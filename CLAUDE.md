# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Data pipeline (run in order, or use task)
task data_csv          # runs noga_csv + ims_csv + data_csv in dependency order
uv run -m noga.data    # runs daily_demand() by default (__main__)

# Train the neural network model
uv run -m noga.model

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
- `data/noga.xlsx`: Hourly electricity demand + day-ahead forecast from NOGA (Israel's grid operator). Hebrew column headers, translated in `data.py`.
- `data/ims.he.csv`: Hourly weather data from the Israeli Meteorological Service (IMS) for three stations: Haifa, Jerusalem, Tel Aviv. Hebrew headers, pivoted from long to wide format.
- `data/data.csv`: Merged dataset with time features (year, month, day-of-week, hour) + 15 weather columns + demand columns. This is the main input for the model.
- `data/daily.csv`: Daily aggregates produced by `data.daily_demand()`.

**Model (`noga/model.py`):**
- PyTorch Lightning `Model` with learned embeddings for month (1–12), day-of-week (0–6), and hour (0–23), each of size `EMBED_SIZE=5`.
- Input: 3 embeddings + 15 normalized numeric features (temperature×9, humidity×3, wind_speed×3).
- Architecture: `Linear → Tanh → 6× (Linear → LeakyReLU) → Linear(1)`.
- **Custom asymmetric loss**: under-prediction penalized 5× more than over-prediction (`UNDER=5`). This reflects the real cost asymmetry in electricity reserves.
- Outputs predictions to `data/pred.csv` or `data/pred-new.csv`.

**Analysis:**
- `noga/stat.py`: `noga-stat` entry point. Produces MAE bar charts and error distribution plots broken down by day, month, hour.
- `noga/error.py`: `cost()` function computes total cost under varying reserve multipliers (0.98–1.2) and plots cost-vs-reserve curves for both the NOGA baseline and the model.
- `noga/plot.py`: Exploratory scatter/line plots (demand vs temperature, forecast vs actual, KDE of forecast errors).

All plots are saved to `plots/`.
