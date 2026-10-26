'''
Post-hoc cost adjustment of point forecasts.

Both adjustments correct a baseline trained on the wrong loss: they are fitted
on the baseline's predictions for the train split and applied to the test
split.
Prices are used only for fitting (they are unknown at decision time), so the
adjusted forecast depends on the base forecast and the hour only.
'''
import numpy as np
import pandas as pd
import torch

from noga.price import P_DA, P_O, P_U

HOURS = 24
FIT_STEPS = 2000
FIT_LR = 1e-2


def excess(pred, y, p_da, p_o, p_u):
    '''Per-slot cost above a perfect forecast (₪/MWh × MW).'''
    return (p_u - p_da) * np.maximum(y - pred, 0) + p_o * np.maximum(pred - y, 0)


def hour(meta: pd.DataFrame):
    return (meta["time"].to_numpy() // 60).astype(int)


def affine(pred_fit, meta_fit: pd.DataFrame):
    '''C1: p = a * pred + b minimising the excess cost on the fitting set.'''
    scale = 100.0
    X = torch.tensor(
        np.stack([pred_fit / scale, np.ones_like(pred_fit)], axis=1),
        dtype=torch.float32)
    y = torch.tensor(meta_fit["actual"].to_numpy() / scale, dtype=torch.float32)
    c = torch.tensor(meta_fit[[P_DA, P_O, P_U]].to_numpy() / scale,
                     dtype=torch.float32)

    w = torch.tensor([1.0, 0.0], requires_grad=True)  # start from p = pred
    opt = torch.optim.Adam([w], lr=FIT_LR)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        pred = X @ w
        under = torch.relu(y - pred)
        over = torch.relu(pred - y)
        loss = ((c[:, 2] - c[:, 0]) * under + c[:, 1] * over).mean()
        loss.backward()
        opt.step()

    a, b = w.detach().numpy()
    return lambda pred, meta: a * pred + b * scale


def cost_shift(err, c_u, c_o):
    '''
    argmin_s sum_i c_u[i] (err[i] - s)+ + c_o[i] (s - err[i])+ :
    the smallest err[k] (sorted) at which the surplus cost of errors below
    balances the shortfall cost of errors above, i.e. a cost-weighted quantile.
    '''
    order = np.argsort(err)
    cum = np.cumsum((c_u + c_o)[order])
    k = np.searchsorted(cum, c_u.sum())
    return err[order][min(k, len(err) - 1)]


def newsvendor_shift(pred_fit, meta_fit: pd.DataFrame):
    '''
    Per-hour cost-optimal constant shift s_h of the errors y - pred (exact,
    a cost-weighted quantile); the starting point of C2.
    '''
    err = meta_fit["actual"].to_numpy() - pred_fit
    c_u = (meta_fit[P_U] - meta_fit[P_DA]).to_numpy()
    c_o = meta_fit[P_O].to_numpy()
    h_fit = hour(meta_fit)
    return np.array([
        cost_shift(err[h_fit == h], c_u[h_fit == h], c_o[h_fit == h])
        for h in range(HOURS)])


def hourly_affine(pred_fit, meta_fit: pd.DataFrame):
    '''
    C2 (predict-then-optimise): p = pred + b_h + a_h * (pred - m_h) for
    hour-of-day h, minimising the excess cost on the fitting set. m_h is the
    hour's mean forecast (centring decouples slope and shift). Starts from the
    exact shift-only solution (a_h = 0), so C1 and the per-hour shift are
    special cases.
    '''
    scale = 100.0
    h_fit = hour(meta_fit)
    m = np.array([pred_fit[h_fit == h].mean() for h in range(HOURS)])
    hh = torch.tensor(h_fit)
    x = torch.tensor(pred_fit / scale, dtype=torch.float32)
    xc = x - torch.tensor(m / scale, dtype=torch.float32)[hh]
    y = torch.tensor(meta_fit["actual"].to_numpy() / scale, dtype=torch.float32)
    c = torch.tensor(meta_fit[[P_DA, P_O, P_U]].to_numpy() / scale,
                     dtype=torch.float32)

    b = torch.tensor(newsvendor_shift(pred_fit, meta_fit) / scale,
                     dtype=torch.float32, requires_grad=True)
    a = torch.zeros(HOURS, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=FIT_LR)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        pred = x + b[hh] + a[hh] * xc
        under = torch.relu(y - pred)
        over = torch.relu(pred - y)
        loss = ((c[:, 2] - c[:, 0]) * under + c[:, 1] * over).mean()
        loss.backward()
        opt.step()

    a, b = a.detach().numpy(), b.detach().numpy() * scale

    def apply(pred, meta):
        h = hour(meta)
        return pred + b[h] + a[h] * (pred - m[h])
    return apply
