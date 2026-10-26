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

# Train all variants, calibrate, evaluate (writes res/price_*.csv, tex/plot/*)
uv run -m noga.report quick        # seed 0, price seed 0
uv run -m noga.report              # 3 seeds x 3 price seeds
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
- `data/data.csv`: 5-minute rows: time (minutes), forecast, actual, weather, `p_da` (constrained SMP), year, month (0-based), day (weekday), `period` (30-min settlement index), `date_ord`. Ends 2025-12-31 / 2026-01-01 (price data end).

**Pricing (`noga/price.py`):** `synth()` draws per settlement period `p_o = alpha(1+eps)p_da` (known to the model) and `p_u = r*p_o` with `r in [5,20]` a noisy convex increasing function of `p_da` (hidden). `PriceParams` holds all generator parameters.

**Model (`noga/model.py`):**
- PyTorch Lightning `Model`: soft attention over the 10 demand readings 24h earlier + additive correction; embeddings for month, day-of-week, 5-min time-of-day; 9 weather features; price-aware variants add `log_p_da`, `p_o_ratio`.
- `Data` is batch-indexable (use `dl()`), yields `(month, day, time, X, y, c)` with `c = [p_da, p_o, p_u] / P_SCALE`; `Data.meta` holds per-item context for evaluation.
- Splits: train 2023-04..2024-08, cal 2024-09..12 (early stopping + calibration), test 2025.
- Models saved as `models/<name>-s<seed>[-p<price_seed>[-params]].pt`.

**Losses (`noga/cost.py`):** `Name` lists the variants (`l2`, `l1`, `Pinball r:1`, `l2+price`, `cost-noprice`, `cost`); losses take `(pred, y, c)`. `cost` = mean excess cost `(p_u-p_da)(y-p)+ + p_o(p-y)+`.

**Calibration (`noga/calibrate.py`):** C1 affine, C2 price-aware newsvendor (hourly residual quantile at tau_hat), C3 learned residual, C4 pinball-model bank. All fitted on the cal split.

**Analysis:**
- `noga/report.py`: experiment driver; results in `res/price_{runs,summary,dm,savings,sensitivity}.csv`, figures in `tex/plot/`.
- `noga/stat.py`, `noga/error.py`, `noga/plot.py`: older exploratory plots/statistics.
- `noga/date.py`: Single `DT_FRMT` constant (`'%Y-%m-%d'`) shared across the pipeline.

Exploratory plots are saved to `plots/`; paper figures to `tex/plot/`. The paper sources are in `tex/` (`task latex:build`).
