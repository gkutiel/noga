# The demand model

The model forecasts Israel's electricity demand one day ahead, at 5-minute resolution.
All variants share the same network and differ only in the training loss.
The proposed model is trained on the actual cost of buying electricity, not on the forecast error.

Code: `noga/model.py` (network and data), `noga/cost.py` (losses).

## Idea in one sentence

Start from what demand was at the same time yesterday, and let a small network decide
how to weight yesterday's readings and how much to add or subtract.

## Inputs

For every target slot `t` the model gets only information that is known when the day-ahead
purchase is made:

| Input | Size | Meaning |
|---|---|---|
| Demand history | 10 | The ten 5-minute demand readings ending exactly 24 h before `t` (the same 50 minutes yesterday) |
| Weather | 9 | Temperature, wet-bulb temperature and relative humidity at Haifa, Jerusalem and Tel Aviv |
| Yesterday's price | 1 | Log of the day-ahead price of the same half-hour yesterday (already public) |
| Time of day | 16 | Learned embedding of the 5-minute slot (288 values) |
| Day of week | 3 | Learned embedding (7 values) |
| Month | 2 | Learned embedding (12 values) |

Today's day-ahead price and the imbalance prices are **never** inputs: they are set only
after the purchase decision. Demand and prices are divided by 100 during training.

## Structure

```
inputs (41 numbers)
   │
   ├─ Linear 41→48 → LeakyReLU → Linear 48→48 → LeakyReLU → Linear 48→48 → LeakyReLU
   │                                   = shared features z (48 numbers)
   │
   ├─ demand head:  Linear 48→11  →  10 attention weights + 1 correction
   │
   └─ price head (proposed model only):  Linear 48→2  →  log p_o, log p_u
```

The forecast is

```
ŷ_t = Σ_k softmax(a_t)_k · h_k  +  δ_t
```

- `h_1..h_10` are yesterday's readings. The softmax weights are positive and sum to 1, so the first
  term is always a weighted average of real demand values: it anchors the forecast to yesterday's level.
- `δ_t` is a correction in MW for what changed since yesterday (weekend, heat, holidays) and, for
  the cost-aware models, the safety margin: how much extra to buy.

The network has about 12k parameters.

## Training objectives

| Model | Loss | What it aims for |
|---|---|---|
| MSE / L1 | squared / absolute error | the mean / median demand |
| Pinball med, opt, p95 | pinball loss at a fixed quantile τ = 0.781 / 0.864 / 0.933 | a fixed safety margin; τ is derived from training prices |
| **Proposed** | excess cost + price loss (below) | the cheapest purchase, slot by slot |

The proposed loss for a batch is

```
cost  = mean[ (p_u − p_da)·(y − ŷ)⁺  +  p_o·(ŷ − y)⁺ ]
price = mean[ ½(log p̂_o − log p_o)² + ½(log p̂_u − log p_u)² ]
loss  = cost + 1.0 · price
```

- **cost** is what is paid above a perfect forecast: buying too little costs the
  under-prediction premium `p_u − p_da` per MWh, buying too much costs `p_o` per MWh.
  Expensive slots therefore pull the forecast up more than cheap ones.
- **price** trains the second head to predict the imbalance prices. It is only a training aid:
  it makes the shared features learn what makes a slot expensive. The predicted prices are not
  used by the demand forecast.

## Training setup

- Train: April 2023 – December 2024; test: 2025.
- Adam, learning rate 2·10⁻⁴, batch size 10,000, a fixed 1,500 epochs, no early stopping.
- Same architecture, inputs and schedule for every model, so differences come from the loss.
- Imbalance prices `p_o`, `p_u` are synthetic (see `noga/price.py`); `p_da` is the real
  unconstrained day-ahead price.
