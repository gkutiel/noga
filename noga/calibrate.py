'''
Post-hoc calibration of point forecasts to the price-dependent cost.

All calibrators are fitted on the calibration split (2024-09..12) and applied
to the test split. Inputs are plain arrays in MW and ₪/MWh; the hidden
under-prediction price p_u is only used for fitting, never at prediction time.
'''
import numpy as np
import pandas as pd
import torch

from noga.price import P_DA, P_O, P_U

N_BINS = 20
HOURS = 24
C3_STEPS = 2000
C3_LR = 1e-2


def excess(pred, y, p_da, p_o, p_u):
    '''Per-slot cost above a perfect forecast (₪/MWh × MW).'''
    return (p_u - p_da) * np.maximum(y - pred, 0) + p_o * np.maximum(pred - y, 0)


class RatioMap:
    '''Binned estimate of E[p_u / p_o | p_da] fitted on known (past) prices.'''

    def fit(self, p_da, p_o, p_u):
        log_p = np.log(p_da)
        self.edges = np.quantile(log_p, np.linspace(0, 1, N_BINS + 1))
        b = self._bin(log_p)
        r = p_u / p_o
        self.r = np.array([r[b == i].mean() for i in range(N_BINS)])
        return self

    def _bin(self, log_p):
        return np.clip(np.searchsorted(self.edges, log_p) - 1, 0, N_BINS - 1)

    def tau(self, p_da, p_o):
        '''Estimated cost-optimal quantile for each slot.'''
        p_u = p_o * self.r[self._bin(np.log(p_da))]
        c_u = p_u - p_da
        return c_u / (c_u + p_o)


def hour(meta: pd.DataFrame):
    return (meta["time"].to_numpy() // 60).astype(int)


def affine(pred_cal, meta_cal: pd.DataFrame):
    '''C1: p = a * pred + b minimising the excess cost on the calibration set.'''
    feats = lambda pred, _: np.stack([pred, np.ones_like(pred)], axis=1)
    return _linear(feats, pred_cal, meta_cal)


def newsvendor(pred_cal, meta_cal: pd.DataFrame, ratio: RatioMap):
    '''
    C2 (predict-then-optimise): p = pred + Q_h(tau_hat), where Q_h is the
    empirical quantile of the calibration residuals y - pred in the same hour.
    '''
    res = meta_cal["actual"].to_numpy() - pred_cal
    h_cal = hour(meta_cal)
    by_hour = [res[h_cal == h] for h in range(HOURS)]

    def apply(pred, meta: pd.DataFrame):
        t = ratio.tau(meta[P_DA].to_numpy(), meta[P_O].to_numpy())
        h = hour(meta)
        shift = np.empty_like(pred)
        for i in range(HOURS):
            m = h == i
            shift[m] = np.quantile(by_hour[i], t[m])
        return pred + shift

    return apply


def residual(pred_cal, meta_cal: pd.DataFrame, ratio: RatioMap):
    '''
    C3: p = pred + w . phi(pred, p_da, p_o, tau_hat, hour), with w fitted to
    minimise the excess cost on the calibration set.
    '''
    def feats(pred, meta):
        p_da = meta[P_DA].to_numpy()
        p_o = meta[P_O].to_numpy()
        onehot = np.eye(HOURS)[hour(meta)]
        return np.concatenate([
            np.stack([
                pred,
                np.log(p_da) - 5,
                10 * p_o / p_da,
                ratio.tau(p_da, p_o),
            ], axis=1),
            onehot,
        ], axis=1)

    return _linear(feats, pred_cal, meta_cal, offset=True)


def _linear(feats, pred_cal, meta_cal, offset=False):
    '''Fits p = [pred +] feats @ w on the excess cost (scaled to ~1).'''
    scale = 100.0
    X = torch.tensor(feats(pred_cal / scale, meta_cal), dtype=torch.float32)
    base = torch.tensor(pred_cal / scale if offset else 0 * pred_cal,
                        dtype=torch.float32)
    y = torch.tensor(meta_cal["actual"].to_numpy() / scale, dtype=torch.float32)
    c = torch.tensor(meta_cal[[P_DA, P_O, P_U]].to_numpy() / scale,
                     dtype=torch.float32)

    w = torch.zeros(X.shape[1], requires_grad=True)
    if not offset:
        with torch.no_grad():
            w[0] = 1  # start from the identity: p = pred

    opt = torch.optim.Adam([w], lr=C3_LR)
    for _ in range(C3_STEPS):
        opt.zero_grad()
        pred = base + X @ w
        under = torch.relu(y - pred)
        over = torch.relu(pred - y)
        loss = ((c[:, 2] - c[:, 0]) * under + c[:, 1] * over).mean()
        loss.backward()
        opt.step()

    w = w.detach().numpy()

    def apply(pred, meta: pd.DataFrame):
        out = feats(pred / scale, meta) @ w
        return (pred / scale * offset + out) * scale

    return apply


def bank(preds: dict[float, np.ndarray], meta: pd.DataFrame, ratio: RatioMap):
    '''
    C4: given fixed-quantile models {tau_r: pred}, interpolate per slot
    between the two models whose tau brackets tau_hat (clamped at the ends).
    '''
    taus = np.array(sorted(preds))
    P = np.stack([preds[t] for t in taus], axis=1)
    t = ratio.tau(meta[P_DA].to_numpy(), meta[P_O].to_numpy())
    t = np.clip(t, taus[0], taus[-1])

    i = np.clip(np.searchsorted(taus, t) - 1, 0, len(taus) - 2)
    w = (t - taus[i]) / (taus[i + 1] - taus[i])
    rows = np.arange(len(t))
    return (1 - w) * P[rows, i] + w * P[rows, i + 1]
