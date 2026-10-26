'''
Price-aware evaluation: trains every model variant for each (seed, price seed),
applies the post-hoc calibrators and writes cost tables, significance tests,
the Israeli savings estimate and the paper figures.

    uv run -m noga.report            # full grid (3 seeds x 3 price seeds)
    uv run -m noga.report quick      # one seed, one price seed
'''
import sys
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from noga import calibrate as cal
from noga.cost import PINBALL_TAU, USE_PRICE, Name
from noga.model import load_data, load_model, predict, train
from noga.price import P_DA, P_O, P_U, PriceParams, tau

RES = Path("res")
PLOTS = Path("tex/plot")
SEEDS = [0, 1, 2]
PRICE_SEEDS = [0, 1, 2]
PROPOSED = "cost"
NOGA = "Noga"
SLOT_H = 5 / 60  # one 5-minute slot, in hours
DM_LAG = 7

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#1f2328", "#8a9099"


def metrics(pred, meta: pd.DataFrame):
    y = meta["actual"].to_numpy()
    ex = cal.excess(pred, y, meta[P_DA], meta[P_O], meta[P_U]) * SLOT_H
    days = meta["date_ord"].nunique()
    return {
        "excess": ex.sum() * 365 / days,  # ₪ per year
        "mae": np.abs(pred - y).mean(),
        "rmse": np.sqrt(((pred - y) ** 2).mean()),
        "coverage": (y <= pred).mean(),
    }, pd.Series(ex).groupby(meta["date_ord"].to_numpy()).sum()


def run(
        seed: int,
        price_seed: int,
        params: PriceParams = PriceParams(),
        names: tuple[Name, ...] = get_args(Name)):
    '''Returns {method: test predictions} and the test/cal metadata.'''
    for name in names:
        train(name, seed=seed, price_seed=price_seed, params=params)

    data = {
        p: load_data(use_price=p, price_seed=price_seed, params=params)
        for p in (False, True)}
    meta_tr, meta_cal, meta_te = (ds.meta for ds in data[False])

    # E[p_u/p_o | p_da] from past prices only (train + cal).
    known = pd.concat([meta_tr, meta_cal])
    ratio = cal.RatioMap().fit(
        known[P_DA].to_numpy(), known[P_O].to_numpy(), known[P_U].to_numpy())

    raw_cal, raw_te = {}, {}
    for name in names:
        _, ds_cal, ds_te = data[name in USE_PRICE]
        model = load_model(
            name, seed=seed, price_seed=price_seed, params=params)
        raw_cal[name] = predict(model, ds_cal)
        raw_te[name] = predict(model, ds_te)

    raw_cal[NOGA] = meta_cal["forecast"].to_numpy()
    raw_te[NOGA] = meta_te["forecast"].to_numpy()

    out = {}
    for name in raw_te:
        p_cal, p_te = raw_cal[name], raw_te[name]
        out[(name, "raw")] = p_te
        if name in {PROPOSED, "cost-noprice"}:
            continue
        out[(name, "C1 affine")] = cal.affine(p_cal, meta_cal)(p_te, meta_te)
        out[(name, "C2 newsvendor")] = cal.newsvendor(
            p_cal, meta_cal, ratio)(p_te, meta_te)
        out[(name, "C3 residual")] = cal.residual(
            p_cal, meta_cal, ratio)(p_te, meta_te)

    out[("Pinball bank", "C4 bank")] = cal.bank(
        {PINBALL_TAU[n]: raw_te[n] for n in PINBALL_TAU}, meta_te, ratio)
    out[("Oracle", "raw")] = meta_te["actual"].to_numpy()

    return out, meta_te, ratio


def dm_test(d: np.ndarray, lag=DM_LAG):
    '''Diebold-Mariano with a Newey-West variance; d = loss_a - loss_b.'''
    n = len(d)
    e = d - d.mean()
    var = e @ e / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    t = d.mean() / np.sqrt(var / n)
    return t, 2 * stats.norm.sf(abs(t))


def report(seeds=SEEDS, price_seeds=PRICE_SEEDS):
    RES.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    rows, daily = [], {}
    for ps in price_seeds:
        for s in seeds:
            preds, meta, ratio = run(s, ps)
            for (model, method), p in preds.items():
                m, d = metrics(p, meta)
                rows.append({"model": model, "calibration": method,
                             "seed": s, "price_seed": ps, **m})
                daily.setdefault((model, method), []).append(d)
            if (s, ps) == (seeds[0], price_seeds[0]):
                figures(preds, meta, ratio)
                savings_base = meta

    runs = pd.DataFrame(rows)
    runs.to_csv(RES / "price_runs.csv", index=False, float_format="%.4f")

    summary = runs.groupby(["model", "calibration"])[
        ["excess", "mae", "rmse", "coverage"]].agg(["mean", "std"])
    summary = summary.sort_values(("excess", "mean"))
    summary.to_csv(RES / "price_summary.csv", float_format="%.4f")
    print(summary.to_string(float_format="{:,.3f}".format))

    # Daily excess averaged over runs; DM test of every method vs the proposed.
    daily = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in daily.items()}
    base = daily[(PROPOSED, "raw")]
    dm = []
    for k, d in daily.items():
        if k == (PROPOSED, "raw") or k[0] == "Oracle":
            continue
        t, p = dm_test((d - base).to_numpy())
        dm.append({"model": k[0], "calibration": k[1],
                   "mean_daily_diff": (d - base).mean(), "dm_t": t, "p": p})
    dm = pd.DataFrame(dm).sort_values("mean_daily_diff")
    dm.to_csv(RES / "price_dm.csv", index=False, float_format="%.4g")
    print(dm.to_string(index=False))

    savings(summary, savings_base)


def savings(summary: pd.DataFrame, meta: pd.DataFrame):
    '''Annual ₪ saving of the proposed model vs raw/calibrated Noga.'''
    ex = summary[("excess", "mean")]
    y = meta["actual"].to_numpy()
    days = meta["date_ord"].nunique()
    energy = y.mean() * 8760  # MWh / year
    bill = (meta[P_DA].to_numpy() * y).sum() * SLOT_H * 365 / days

    rows = []
    for method in ["raw", "C1 affine", "C2 newsvendor", "C3 residual"]:
        saved = ex[(NOGA, method)] - ex[(PROPOSED, "raw")]
        rows.append({
            "vs": f"Noga {method}",
            "saving_nis_per_year": saved,
            "saving_nis_per_mwh": saved / energy,
            "saving_pct_of_da_bill": 100 * saved / bill,
        })
    out = pd.DataFrame(rows)
    out.attrs["energy_mwh"] = energy
    out.to_csv(RES / "price_savings.csv", index=False, float_format="%.4g")
    print(f"annual energy {energy:,.0f} MWh, DA bill {bill:,.0f} ₪")
    print(out.to_string(index=False))


def _style(ax):
    ax.grid(True, color="#e6e8eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK)


def figures(preds, meta: pd.DataFrame, ratio: cal.RatioMap):
    p_da = meta[P_DA].to_numpy()
    t = tau(p_da, meta[P_O].to_numpy(), meta[P_U].to_numpy())

    # tau_t vs day-ahead price.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.random.default_rng(0).choice(len(t), 5000, replace=False)
    ax.scatter(p_da[idx], t[idx], s=8, color=BLUE, alpha=0.35,
               linewidths=0, label=r"realised $\tau_t$")
    grid = np.geomspace(p_da.min(), p_da.max(), 200)
    ax.plot(grid, ratio.tau(grid, PriceParams().alpha * grid),
            color=INK, linewidth=2,
            label=r"$\hat\tau$ (known at decision time)")
    ax.set_xscale("log")
    ax.set_xlabel("Day-ahead price (₪/MWh)")
    ax.set_ylabel(r"Cost-optimal quantile $\tau_t$")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "tau_vs_price.png", dpi=150)
    plt.close(fig)

    # Coverage and excess cost by price decile.
    dec = pd.qcut(p_da, 10, labels=False)
    y = meta["actual"].to_numpy()
    shown = [
        ((NOGA, "C2 newsvendor"), "Noga + C2", ORANGE),
        (("l2", "C2 newsvendor"), "MSE + C2", YELLOW),
        (("Pinball bank", "C4 bank"), "Pinball bank (C4)", AQUA),
        ((PROPOSED, "raw"), "Proposed", BLUE),
    ]
    target = pd.Series(t).groupby(dec).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(target.index + 1, target, color=INK, linestyle="--",
            linewidth=1.5, label=r"mean $\tau_t$ (target)")
    for key, label, color in shown:
        cov = pd.Series(y <= preds[key]).groupby(dec).mean()
        ax.plot(cov.index + 1, cov, color=color, linewidth=2,
                marker="o", markersize=5, label=label)
    ax.set_xlabel("Day-ahead price decile")
    ax.set_ylabel(r"Coverage $P(y \leq \hat p)$")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[1]
    width = 0.8 / len(shown)
    for i, (key, label, color) in enumerate(shown):
        ex = cal.excess(preds[key], y, p_da, meta[P_O], meta[P_U]) * SLOT_H
        by = pd.Series(ex).groupby(dec).sum() / 1e6
        ax.bar(by.index + 1 + (i - 1.5) * width, by, width=width * 0.9,
               color=color, label=label)
    ax.set_xlabel("Day-ahead price decile")
    ax.set_ylabel("Excess cost on test year (M₪)")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False)
    _style(ax)

    fig.tight_layout()
    fig.savefig(PLOTS / "cost_by_price.png", dpi=150)
    plt.close(fig)


SENSITIVITY = {
    "default": PriceParams(),
    "alpha=0.25": PriceParams(alpha=0.25),
    "alpha=0.4": PriceParams(alpha=0.4),
    "gamma=1": PriceParams(gamma=1.0),
    "gamma=3": PriceParams(gamma=3.0),
    "sigma_u=0.05": PriceParams(sigma_u=0.05),
    "sigma_u=0.3": PriceParams(sigma_u=0.3),
    "r in [5,10]": PriceParams(r_hi=10.0),
    "r in [10,20]": PriceParams(r_lo=10.0),
}
SENSITIVITY_SHOWN = [
    ((PROPOSED, "raw"), "Proposed"),
    (("l2", "C2 newsvendor"), "MSE + C2"),
    (("l2", "C3 residual"), "MSE + C3"),
    (("Pinball bank", "C4 bank"), "Pinball bank (C4)"),
    ((NOGA, "C2 newsvendor"), "Noga + C2"),
    ((NOGA, "raw"), "Noga"),
]


def sensitivity(seed=0, price_seed=0):
    '''Excess cost (M₪/yr) under alternative imbalance-price generators.'''
    names: tuple[Name, ...] = ("l2", *PINBALL_TAU, PROPOSED)
    rows = []
    for label, params in SENSITIVITY.items():
        preds, meta, _ = run(seed, price_seed, params, names)
        row = {"generator": label}
        for key, name in SENSITIVITY_SHOWN:
            row[name] = metrics(preds[key], meta)[0]["excess"] / 1e6
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "price_sensitivity.csv", index=False, float_format="%.3f")
    print(out.to_string(index=False, float_format="{:.2f}".format))


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["quick"]:
            report(seeds=[0], price_seeds=[0])
        case ["sensitivity"]:
            sensitivity()
        case _:
            report()
