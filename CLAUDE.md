# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Data pipeline (run in order, or use task)
task data_csv          # runs noga_csv + ims_csv + data_csv in dependency order
# Note: tasks are idempotent — they skip if the output file already exists.
# Delete data/noga.csv, data/ims.csv, or data/data.csv to force a rerun.
uv run -m noga.data <fn>   # runs noga.data.<fn>() (e.g. data_csv); sample_csv() if omitted

# Real day-ahead prices (SMP xlsx -> data/price.csv), merged by data_csv
uv run -m noga.data price_csv

# Check the synthetic imbalance-price generator
uv run -m noga.price

# Train one model (refuses if tracked files are uncommitted); short names:
# mse, l1, med, opt, p95, cost. TensorBoard runs: logs/<YYYY-MM-DD-HH-MM>-<sha>-<short>
# with cost/train, cost/test (every 50 epochs, M₪/yr), loss/<name>,
# price/{train,test} (cost model only: relative squared error of predicted p_da, p_o, p_u).
uv run train cost [--seed N] [--price-seed N]

# Train all variants, calibrate, evaluate (writes res/price_*.csv, tex/plot/*)
uv run -m noga.report quick        # seed 0, price seed 0
uv run -m noga.report              # 3 seeds x 3 price seeds (trains JOBS models in parallel)
uv run -m noga.report sensitivity  # alternative price generators

# Generate stat/error plots
uv run noga-stat       # runs noga/stat.py:main() — MAE charts by day/month/hour
uv run -m noga.error   # runs cost-vs-reserve plots

# Generate exploratory plots
uv run -m noga.plot
```

## Architecture

This project predicts Israel's electricity demand, improving on NOGA's existing day-ahead forecast. The pipeline is:

```
data/noga.xlsx  ──► data/noga.csv  ──┐
                                      ├──► data/data.csv ──► model / plots / stats
data/ims.he.csv ──► data/ims.csv   ──┘
```

**Data sources:**
- `data/noga.xlsx`: 5-minute electricity demand + day-ahead forecast from NOGA (Israel's grid operator). Hebrew column headers, translated in `data.py`; `noga_fixed_csv()` interpolates gaps into `data/noga.fixed.csv`.
- `data/ims.he.csv`: Weather data from the Israeli Meteorological Service (IMS) for three stations: Haifa, Jerusalem, Tel Aviv. Hebrew headers, pivoted from long to wide format.
- `data/Smp_01_04_2023-01_01_2026_he-IL.xlsx` → `data/price.csv`: real half-hourly day-ahead SMP (₪/MWh). Real-time columns are empty, so imbalance prices are synthetic.
- `data/data.csv`: 5-minute rows: time (minutes), forecast, actual, weather, `p_da` (unconstrained SMP, `data.DA_PRICE`), year, month (0-based), day (weekday), `period` (30-min settlement index), `date_ord`. Ends 2025-12-31 / 2026-01-01 (price data end).

**Pricing (`noga/price.py`):** `synth()` draws per settlement period `p_o = alpha(1+eps)p_da` and `p_u = r*p_o` with `r in [5,20]` a noisy convex increasing function of `p_da` (`rho > 0`: scarcity variant, noise correlated with the system-wide demand surprise). `PriceParams` holds all generator parameters. A slot's own prices are never model inputs (the SMP is set after procurement decisions; only yesterday's price is an input); they enter only the `cost` loss, the post-hoc adjustments' fitting and the evaluation.

**Model (`noga/model.py`):**
- PyTorch Lightning `Model`: forecast `a * attention + b` (attention = softmax-weighted average of the `HISTORY_LEN` demand readings 24h earlier; `a = exp(s) > 0`, starts at 1; a, b per slot from the network); the network's only inputs are embeddings for month, day-of-week, 5-min time-of-day (`max_norm=1`, so they stay on the weather's scale) and the 9 weather features (forecast for t), standardised per column with the train-split mean and std (`x_mean`/`x_std` buffers saved with the model). Plain Adam, no weight decay. All other inputs are ≥24h old and used only by attentions: the demand history (`HISTORY_LEN` readings ending t−24h) and the day-ahead price history (`PRICE_LEN` prices, `PRICE_STEP` rows = 30 min apart, ending t−24h). `X = [demand history | weather | price history]`. The slot's own prices are never inputs.
- The `cost` model (`PRICE_HEADS`) has a price head on real prices: `p_da = a_p * attention(price history)`, `p_o = exp(s_o) p_da`, `p_u = exp(s_u) p_o` (multi-task loss: relative squared error, weight `PRICE_WEIGHT`); `predict_prices()` returns `(p_da, p_o, p_u)`.
- `Data` is batch-indexable (use `dl()`), yields `(month, day, time, X, y, c)` in real units (MW, ₪/MWh; `c = [p_da, p_o, p_u]`). No fixed scale constants: the model's buffers `y_scale` (std of y(t)−y(t−24h), scales the shift b and the demand losses) and `p_scale` (mean p_da, scales prices in the losses) are set from the train split by `set_scale`, like `x_mean`/`x_std`; `predict`/`predict_prices` return MW and ₪/MWh; `Data.meta` holds per-item context for evaluation.
- Splits: train 2023-04..2024-12, test 2025. Every model trains on train for a fixed `EPOCHS` (no early stopping); C1/C2 are fitted on the baselines' train predictions.
- Models saved as `models/<short name>[-s<seed>][-p<price_seed>]…-<git sha>.pt` (non-default parts only, e.g. `cost-a1b2c3d.pt`, `p95-s1-p2-a1b2c3d.pt`). `train()` skips only if that exact file (same commit) exists; `--force` retrains. `load_model`/`pt.find` take a given `sha`, else the newest file, else the pre-sha name (`cost-s0-p0.pt`, the 42-model matrix). `noga.report` reuses a model of any commit.

**Losses (`noga/cost.py`):** `Name` lists the variants: `l2`, `l1`; `Pinball med/opt/p95` (constant quantile derived from training prices: median of the per-slot optimal tau, the cost-optimal constant E[c_u]/(E[c_u]+E[c_o]), 95th percentile of tau); `cost` (proposed) = mean excess cost `(p_u-p_da)(y-p)+ + p_o(p-y)+`. Losses take `(pred, y, c)`.

**Post-hoc cost adjustment (`noga/calibrate.py`):** C1 global affine `a*pred + b`; C2 hourly affine `pred + b_h + a_h*(pred - m_h)`, started from the exact per-hour cost-optimal shift (`newsvendor_shift`, a cost-weighted quantile of errors). Both fitted on the excess cost of the baseline's train predictions.

**Analysis:**
- `noga/report.py`: experiment driver; results in `res/price_{runs,summary,dm,savings,sensitivity}.csv`, figures in `tex/plot/`.
- `noga/stat.py`, `noga/error.py`, `noga/plot.py`: older exploratory plots/statistics.
- `noga/date.py`: Single `DT_FRMT` constant (`'%Y-%m-%d'`) shared across the pipeline.

Exploratory plots are saved to `plots/`; paper figures to `tex/plot/`. The paper sources are in `tex/` (`task latex:build`).
