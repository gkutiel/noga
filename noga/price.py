from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns produced by synth(): over/under-prediction prices (₪/MWh).
P_DA = "p_da"
P_O = "p_o"
P_U = "p_u"


@dataclass(frozen=True)
class PriceParams:
    '''
    Synthetic imbalance prices, driven by the real day-ahead price p_da:

        p_o = alpha * (1 + eps_o) * p_da                 (known to the model)
        r   = clip((r_lo + (r_hi - r_lo) * u**gamma) * exp(eta), r_lo, r_hi)
        p_u = r * p_o                                    (hidden from the model)

    where u in [0, 1] is the log-price position between the q_lo and q_hi
    quantiles of p_da. Since alpha * r_lo > 1, p_o < p_da < p_u always holds,
    and p_u is a noisy, convex, increasing function of p_da.
    '''
    alpha: float = 0.3
    gamma: float = 2.0
    sigma_o: float = 0.03
    sigma_u: float = 0.15
    r_lo: float = 5.0
    r_hi: float = 20.0
    q_lo: float = 0.05
    q_hi: float = 0.99


def synth(
        prices: pd.DataFrame,
        *,
        seed: int,
        params: PriceParams = PriceParams()) -> pd.DataFrame:
    '''
    prices: one row per settlement period with a `p_da` column.
    Returns a copy with `p_o` and `p_u` columns added.
    '''
    assert params.alpha * params.r_lo > 1, "p_u must exceed p_da"

    rng = np.random.default_rng(seed)
    p_da = prices[P_DA].to_numpy()
    log_p = np.log(p_da)
    lo, hi = np.quantile(log_p, [params.q_lo, params.q_hi])
    u = np.clip((log_p - lo) / (hi - lo), 0, 1)

    eps_o = rng.normal(0, params.sigma_o, len(p_da))
    eta = rng.normal(0, params.sigma_u, len(p_da))

    p_o = params.alpha * (1 + eps_o) * p_da
    r = np.clip(
        (params.r_lo + (params.r_hi - params.r_lo) * u ** params.gamma)
        * np.exp(eta),
        params.r_lo,
        params.r_hi)

    out = prices.copy()
    out[P_O] = p_o
    out[P_U] = r * p_o
    return out


def tau(p_da, p_o, p_u):
    '''Cost-optimal quantile: c_u / (c_u + c_o), c_u = p_u - p_da, c_o = p_o.'''
    c_u = p_u - p_da
    return c_u / (c_u + p_o)


def check(seed=0):
    from scipy.stats import spearmanr

    prices = pd.read_csv("data/price.csv")
    prices = prices.rename(columns={"day_ahead_constrained": P_DA})
    df = synth(prices[[P_DA]], seed=seed)

    r = df[P_U] / df[P_O]
    t = tau(df[P_DA], df[P_O], df[P_U])

    assert (df[P_O] < df[P_DA]).all()
    assert (df[P_DA] < df[P_U]).all()
    assert r.between(5 - 1e-9, 20 + 1e-9).all()

    print(df.describe().T.to_string(float_format="{:.2f}".format))
    print(f"ratio p_u/p_o: {r.min():.2f}..{r.max():.2f}")
    print(f"tau: {t.min():.3f}..{t.max():.3f} (median {t.median():.3f})")
    print(f"spearman(r, p_da): {spearmanr(r, df[P_DA]).statistic:.3f}")


if __name__ == "__main__":
    check()
