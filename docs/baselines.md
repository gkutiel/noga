# Baselines

This note explains two of the baselines our model is compared against:

- **Baseline 1: MSE + price-aware calibration.** A normal forecast, then a correction that depends on the expected prices.
- **Baseline 3: Pinball opt.** A forecast trained on a fixed asymmetric loss.

Both use the same data, the same split (train 2023-04 … 2024-12, test 2025) and the same rule for inputs (everything at least 24 h old, plus the weather forecast for t). The only thing that changes is **how the model learns where to put its forecast**.

---

## Background: what a good forecast should be

For each 5-minute slot t we commit to a forecast q. Then the real demand y arrives, and we pay extra for any mistake:

```
excess cost = c_u · (y − q)+   +   c_o · (q − y)+
              ───────────────      ──────────────
              under-forecast        over-forecast
```

- `c_u = p_u − p_da` is the extra cost of each missing MWh.
- `c_o = p_o` is the cost of each surplus MWh.

Under-forecasting is much more expensive. On the training data the average `c_u` is **269 ₪/MWh** and the average `c_o` is **42.5 ₪/MWh**, so a missing MWh costs about 6× more than a surplus MWh.

This is the classic **newsvendor problem**. Its answer: if you knew the prices of slot t, the best forecast would be the **τ-quantile** of the demand, where

```
τ = c_u / (c_u + c_o)
```

In words: forecast so high that the demand ends up **below** your forecast a fraction τ of the time.

**Example.** A slot with `c_u = 450`, `c_o = 45` has τ = 450 / 495 = 0.91, so you should cover the demand 91 % of the time. A slot with `c_u = 120`, `c_o = 40` has τ = 0.75, so you cover it only 75 % of the time.

**The key point: τ changes from slot to slot**, because prices change. On the training data:

| | τ |
|---|---|
| 5th percentile | 0.62 |
| median | 0.78 |
| 95th percentile | 0.93 |
| median in the cheapest 20 % of slots (by p_da) | 0.66 |
| median in the most expensive 20 % of slots | 0.92 |

When prices are high, under-forecasting hurts much more, so the forecast should go up. A good method must follow this.

There is a catch: **we don't know the slot's prices when we forecast** (the day-ahead price is set after the decision). We only know things that predict them: yesterday's price, the hour, the weekday, the weather. Every method below differs in how it handles this.

---

## Baseline 1: MSE + price-aware calibration

### The idea in one sentence

First make the best ordinary forecast, then raise it by the amount the expected prices call for.

This is the standard **"predict, then optimise"** pipeline. Most operators would build it, and the paper must beat it.

### Step 1: an ordinary forecast (MSE)

Train our network (same inputs, same trunk, same demand head) with the plain squared-error loss and no price head:

```
loss = mean( (ŷ − y)² )
```

This gives ŷ_t, the **expected** demand. It knows nothing about prices; over- and under-forecasting are equally bad to it.

### Step 2: a separate price model

Train a **separate** network that predicts the slot's prices from the same inputs (calendar, weather, `p_da(t−24h)`), using only the price loss:

```
loss = mean over (p_da, p_o, p_u) of ((p̂ − p) / p)²
```

It has the same architecture as our model's price head, but it is trained on its own, with no demand loss. It must **not** reuse our model's price head; otherwise the baseline would borrow what our model learned.

From its output we get the expected costs and the expected τ for every slot:

```
ĉ_u = p̂_u − p̂_da,     ĉ_o = p̂_o,     τ̂ = ĉ_u / (ĉ_u + ĉ_o)
```

**Example.**

| slot | p̂_da | p̂_o | p̂_u | ĉ_u | ĉ_o | τ̂ |
|---|---|---|---|---|---|---|
| summer evening (expensive) | 200 | 60 | 900 | 700 | 60 | 0.92 |
| spring night (cheap) | 90 | 27 | 200 | 110 | 27 | 0.80 |

### Step 3: the shift (the newsvendor step)

The MSE forecast aims at the middle. We now push it up to the τ̂-quantile. We need to know how the forecast's errors are spread:

```
e = y − ŷ          (positive = demand was above the forecast)
```

**Plain version.** Take the τ̂-quantile of the errors and add it:

```
q_t = ŷ_t + Quantile_τ̂t( e )
```

**Example.** Suppose the errors on the calibration set have a 80th percentile of +250 MW and a 92nd percentile of +450 MW. Then:

- the cheap spring night (τ̂ = 0.80) gets `q = ŷ + 250`;
- the expensive summer evening (τ̂ = 0.92) gets `q = ŷ + 450`.

**Recommended version: bins with the exact cost-optimal shift.** The plain version assumes the errors don't depend on the prices. With our scarcity prices that is false: `p_u` jumps exactly when demand surprises upward, so a plain quantile shifts too little. The fix is simple and needs no assumption:

1. Sort the calibration slots by τ̂ and cut them into 10 equal bins (deciles).
2. In each bin b, find the single shift s_b that minimises the **real** excess cost of the slots in that bin:
   ```
   s_b = argmin_s  Σ_{t in bin b} [ c_u,t · (e_t − s)+  +  c_o,t · (s − e_t)+ ]
   ```
   This has an exact answer: a **cost-weighted quantile** of the errors. `calibrate.cost_shift` already computes it.
3. Forecast: `q_t = ŷ_t + s_{bin(τ̂_t)}`.

Each bin learns its own margin from real costs, so any link between errors and prices is captured automatically. If the error spread also changes a lot by hour, use bins of (τ̂ decile × 4 blocks of the day) = 40 cells. Don't go finer: each cell needs a few thousand slots for a stable shift.

### Where the shift is fitted

- **Not on test**, ever.
- **Preferably not on the same data the MSE model trained on.** Training errors are smaller than real errors (the model has seen those slots), so the shifts come out too small. Use a held-out calibration window: train the MSE model and the price model on 2023-04 … 2024-09, fit the shifts on 2024-10 … 2024-12, then retrain both models on all training data and keep the shifts.
- If that's too much work, fit on the training predictions and say so in the paper. Our C1/C2 do this today, but it makes the baseline a little weaker than it could be.

### How it relates to C1 and C2

| | margin depends on | can follow prices? |
|---|---|---|
| C1 (global affine `a·ŷ + b`) | nothing, one rule for all slots | no |
| C2 (hourly affine) | the hour of the day | only through the hour |
| **Baseline 1** | the expected prices (τ̂), optionally also the hour | **yes** |

Baseline 1 replaces both: it is the strongest post-hoc fix of an MSE forecast.

### What beating it proves

Both Baseline 1 and our model use price information. The difference is **how**:

- **Baseline 1** learns the demand and the prices separately, then glues them with one margin per bin. It assumes the forecast's uncertainty is the same for all slots in a bin.
- **Our model** learns where to put the forecast directly from the real cost. It can give a bigger margin where the forecast is **uncertain** (e.g. a heat wave that started yesterday) and a smaller one where it is **sure**, even at the same τ̂.

So a win over Baseline 1 shows that **training on the cost, jointly with prices, beats predicting first and correcting after**. That is the paper's central claim.

### Useful extra: the oracle version

Run Baseline 1 once more with the **true** τ instead of τ̂. That can't be used in practice (the prices are unknown at decision time), but it shows how much of the gap comes from not knowing the prices. If even the oracle loses to our model, the advantage comes from modelling uncertainty, not from better price guesses.

---

## Baseline 3: Pinball opt

### The idea in one sentence

Train the forecaster with an asymmetric loss that punishes under-forecasting more, using one fixed ratio set from average prices.

This is the approach most earlier papers take: "costs are asymmetric, so use a quantile loss". It answers whether a fixed asymmetry is already enough.

### The pinball loss

For a chosen quantile τ:

```
pinball_τ(q, y) = τ · (y − q)        if y > q   (under-forecast)
                  (1 − τ) · (q − y)  if q ≥ y   (over-forecast)
```

A model trained with it learns the **τ-quantile** of demand given its inputs. With τ = 0.5 it is the median (same as L1); with τ = 0.9 the forecast ends above the demand 90 % of the time.

Note the similarity with the excess cost: both are "under times one weight + over times another weight". The pinball loss uses the **fixed** weights τ and 1 − τ. The excess cost uses the **real** prices of each slot.

### How τ is chosen ("opt")

We pick the single τ that would be cost-optimal if prices and errors were unrelated. Averaging the costs over the training data:

```
τ* = E[c_u] / (E[c_u] + E[c_o]) = 269.4 / (269.4 + 42.5) = 0.864
```

So the Pinball opt model always aims to cover the demand 86.4 % of the time, in every slot.

Notes on the code (`noga/cost.py: pinball_opt`):
- It computes the ratio `fac = Σ c_u / Σ c_o` on **each training batch**, so τ wobbles slightly between batches (around 0.864). Computing it once on the whole training set would be cleaner, but the difference is tiny.
- `pinball()` writes the loss as weights `u = 1/(1+fac)` for over-forecast and `1−u` for under-forecast, which is the same as τ = fac/(1+fac). It multiplies by 2, which only rescales the loss.

### Why τ* is not the median τ

τ* = 0.864 is higher than the median slot's τ (0.78). The average is pulled up by the expensive slots, where `c_u` is huge. Those slots dominate the total cost, so it pays to lean toward them. This is why "opt" beats the median-τ variant ("Pinball med") on cost.

### What it can and cannot do

- **Can:** shape the forecast by calendar and weather. It learns the 86.4 % quantile *given the inputs*, so its margin is bigger where demand is harder to predict (hot afternoons) and smaller where it's easy (nights).
- **Cannot:** change τ with prices. A cheap slot (true τ ≈ 0.66) gets the same 86.4 % cover as an expensive one (true τ ≈ 0.92). It over-forecasts the cheap slots and under-forecasts the expensive ones, and the expensive misses are the costly ones.

**Example.** A cheap night slot with true τ = 0.66 is forecast at its 86 % quantile, so we buy a margin we don't need (cheap but wasteful). An expensive evening slot with true τ = 0.92 is forecast at only 86 %, so it's short more often than it should be, and each shortfall costs ~700 ₪/MWh.

### What beating it proves

Pinball opt already knows the costs are asymmetric. A win over it shows that **knowing the asymmetry isn't enough: the forecast must also follow how the asymmetry changes with prices**. That is what separates our model from the asymmetric-loss literature, and it supports the novelty claim.

---

## Summary

| | knows costs are asymmetric | follows the slot's expected prices | learns from the real cost | what a win shows |
|---|---|---|---|---|
| Baseline 1: MSE + price-aware calibration | yes (after training) | yes, through τ̂ bins | only in the shift | joint training beats predict-then-optimise |
| Baseline 3: Pinball opt | yes (in training) | no, one τ for all | no, a proxy loss | a fixed asymmetry isn't enough |
| **Our model** | yes | yes, via the shared price head and p_da(t−24h) | yes, directly | — |
