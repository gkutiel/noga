# noga

A unified price-and-demand model for cost-optimal day-ahead electricity demand
forecasting in Israel. It forecasts the demand of every 5-minute slot one day
ahead, together with the day-ahead and imbalance prices, and is trained on the
procurement cost instead of the forecast error.

- `noga/`: data pipeline, model, losses, post-hoc corrections and the
  experiment driver.
- `tex/`: the paper (`task latex:build`).
- `docs/model.pdf`: a diagram of the current model.
- `docs/baselines.md`: the baselines and why they were chosen.

See `CLAUDE.md` for the commands, the data and the architecture.
