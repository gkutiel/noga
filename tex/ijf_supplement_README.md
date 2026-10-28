# Supplementary material: code and data

A unified price-and-demand model for cost-optimal day-ahead electricity demand
forecasting in Israel.

## Contents

- `data/data.csv`: the dataset, one row per 5-minute slot, April 2023 to
  December 2025: time, the system operator's day-ahead forecast, the actual
  national demand (MW), weather at three stations (Haifa, Jerusalem, Tel Aviv),
  the real half-hourly day-ahead price `p_da` (NIS/MWh) and calendar fields.
  Imbalance prices are not in the file; they are drawn from `p_da` by
  `noga/price.py`.
- `noga/`: the code.
  - `price.py`: the synthetic imbalance-price generator.
  - `model.py`: the unified model and the baselines.
  - `cost.py`: the losses.
  - `calibrate.py`: the post-hoc corrections (C1, C2).
  - `report.py`: the experiment driver; reproduces every table and figure.
  - `data.py`: the pipeline that built `data/data.csv` from the raw sources.
  - The other files are helpers and exploratory scripts.
- `pyproject.toml`, `uv.lock`, `.python-version`: the exact Python environment.

## Reproducing the results

Requires [uv](https://docs.astral.sh/uv/). The code names each trained model
after the git commit and refuses to train on uncommitted changes, so commit the
unpacked folder first:

```bash
git init -q && git add -A && git commit -qm supplement
uv sync
uv run -m noga.report quick   # one seed, one price draw
uv run -m noga.report         # 3 seeds x 3 price draws (the paper's tables)
uv run -m noga.report sensitivity
```

Results are written to `res/price_*.csv` and figures to `tex/plot/`. One model
trains in about five minutes on a laptop CPU.
