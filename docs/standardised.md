# Standardised inputs

This note explains why the network now sees *standardised* inputs, what that means,
and how it is done. Each step comes with a small worked example.

Code: `Model.set_scale`, `Model.hidden` and `train()` in `noga/model.py`.

## The problem in one sentence

The raw inputs are large and sit far from zero, so at the start of training almost
every hidden unit is either always on or always off, and which ones depends on the
random seed. Training then starts from a nearly linear network, and how well it recovers
also depends on the seed.

## Step 1: the inputs live on very different scales

Every slot gives the network 34 numbers (plus the calendar embeddings). On the
training split they look like this:

| Input | Count | Mean | Std |
|---|---|---|---|
| Demand history (MW / 100) | 24 | 91 | 21 |
| Relative humidity (%) | 3 | 56 – 72 | 11 – 22 |
| Temperature, wet-bulb (°C) | 6 | 14 – 23 | 4 – 7 |
| Log of yesterday's price | 1 | 0.3 | 0.3 |

**Example.** A summer afternoon might give:

```
demand readings   ≈ 105, 106, 106, ... (24 numbers)
humidity Haifa    = 85
temp Tel Aviv     = 30
log price         = 0.6
```

The demand readings are about 100 times larger than the price input and are all
almost the same number. That causes the trouble below.

## Step 2: what one hidden unit computes

The first layer has 16 units. Each one computes a weighted sum of all inputs plus a bias,
and then applies LeakyReLU:

```
a = w1·x1 + w2·x2 + ... + w34·x34 + b
out = a        if a > 0      ("on":  passes the signal)
out = 0.01·a   if a < 0      ("off": almost blocks it)
```

A unit is useful only if `a` is **positive for some slots and negative for others**.
That switching is what lets the network bend its answer and not just draw a straight line.

## Step 3: why the units get stuck

At the start the weights are random, about ±0.15 each. Look only at the 24 demand
readings:

```
a ≈ (w1 + w2 + ... + w24) · demand level  + (smaller terms)
```

Say the 24 random weights happen to add up to `S = +0.35`. Then:

| Slot | Demand level | a ≈ 0.35 × level |
|---|---|---|
| Quiet night | 60 | +21 |
| Average | 91 | +32 |
| Hot afternoon | 130 | +45 |

`a` is positive for **every** slot. The unit is always "on" and acts like a plain
linear term. If the sum had come out as `S = −0.35`, `a` would be negative for every
slot and the unit would be always "off", which is almost dead.

The reason is that the demand level is always far from zero. The mean is 91 and the
std is 21, so the mean is about 4.4 std away from zero. For a unit to switch, demand
would have to move more than 4 std, which almost never happens.

The other large inputs (humidity around 60–70, temperature around 20) add the same
kind of fixed offset.

**Measured at the start of training** (16 units, before standardising):

| Seed | Always off | Always on | Actually switching |
|---|---|---|---|
| 0 | 8 | 5 | 3 |
| 1 | 6 | 9 | 1 |
| 2 | 9 | 5 | 2 |

Only 1–3 of the 16 units do any nonlinear work.

## Step 4: why this depends on the seed

The seed picks the random starting weights. The starting weights decide each sum `S`,
and `S` decides which units are on, off or switching. So each seed starts from a
different, mostly broken network.

Training can fix it, but slowly and noisily. Because the inputs are about 91, one Adam
step moves `a` by roughly 1, almost all of it along the "overall demand level"
direction. The first layer therefore moves in big, coarse jumps, and the path it takes
(and so where it ends after a fixed number of epochs) differs from seed to seed. In
our runs this showed up as test costs from 166 to 207 M₪/yr for the same model.

## Step 5: the fix, which is to standardise every input

For each input column, subtract its training mean and divide by its training std:

```
x̃ = (x − mean) / std
```

After this, every column has mean 0 and std 1 on the training data.

**Example**, using the summer afternoon from Step 1:

| Input | Raw | Mean | Std | Standardised |
|---|---|---|---|---|
| Demand reading | 105 | 91.1 | 20.7 | (105 − 91.1) / 20.7 = **+0.67** |
| Humidity Haifa | 85 | 71.9 | 19.8 | (85 − 71.9) / 19.8 = **+0.66** |
| Temp Tel Aviv | 30 | 23.4 | 5.0 | (30 − 23.4) / 5.0 = **+1.31** |
| Log price | 0.6 | 0.31 | 0.27 | (0.6 − 0.31) / 0.27 = **+1.07** |

Now every input is a small number around zero. "+1.31" simply means "1.3 std hotter
than usual", whatever the original unit was.

Back to the unit from Step 3 with `S = +0.35`:

| Slot | Standardised demand | a ≈ 0.35 × value |
|---|---|---|
| Quiet night | (60 − 91) / 21 = −1.5 | −0.52 (off) |
| Average | 0 | 0 |
| Hot afternoon | (130 − 91) / 21 = +1.9 | +0.66 (on) |

The same unit now switches between low-demand and high-demand slots, which is exactly
what we want.

## Step 6: use the training statistics only

The mean and std are computed **once, on the training split** (2023-04 … 2024-12), and
the same numbers are then used for the test year.

**Why.** If we computed them on 2025, the model would quietly learn something about the
test year. For example, if 2025 were hotter on average, its mean temperature would be
higher and that fact would leak into the inputs. Using training numbers only keeps the
test honest.

In code (`train()`):

```python
model = Model(name=name, test=test_ds)
model.set_scale(train_ds.X)      # mean/std of the training inputs
```

`set_scale` stores `X.mean(0)` and `X.std(0)` per column. A std below `1e-6` is raised
to `1e-6` so a constant column can never cause a division by zero.

## Step 7: where the numbers are kept

The mean and std are stored inside the model as two **buffers**, `x_mean` and `x_std`.
A buffer is a tensor that belongs to the model but is not trained:

- it is saved in the `.pt` file together with the weights;
- `load_model` restores it automatically, so predictions after loading use exactly the
  training numbers;
- the optimiser never changes it.

**Example.** Train a model, save it, load it next week: `x_mean[0]` is still 91.1, so a
reading of 105 still becomes +0.67.

Creating the buffers uses no random numbers, so seed 0 still gives the same starting
weights as before.

## Step 8: the forecast itself still uses the raw demand

Only the network's **input** is standardised (`Model.hidden`). The forecast is built in
`Model.demand` from the **raw** demand readings:

```
forecast = Σ attention_i × reading_i  +  correction
```

**Example.** Two readings, 105 and 107 (MW / 100), with attention weights 0.5 / 0.5,
and a correction of +1.2:

```
forecast = 0.5·105 + 0.5·107 + 1.2 = 107.2  →  10,720 MW
```

The network decides the weights and the correction from the standardised inputs, but
the average is taken over real demand values, so the output stays in MW.

## Step 9: before and after

Measured on the training data at the start of training, for seeds 0–2:

| | Before | After |
|---|---|---|
| Typical size of a unit's value `a` | 29 – 36 | 0.4 – 0.5 |
| Units stuck always on or always off | 13 – 15 of 16 | 0 of 16 |
| Starting correction | −1 to −3 | ≈ 0 |
| Largest attention weight at the start | 0.3 – 0.5 | ≈ 0.05 (almost an even average) |

Every seed now starts from a healthy network in which every unit can switch. The
starting point is also more alike from seed to seed: an almost even average of
yesterday's readings with no correction.

## What this does not fix

- **The cost loss is piecewise linear.** Its gradient is just "± price" for each slot, however
  small the error, so with a constant learning rate the model keeps jittering near the
  optimum. A learning-rate decay would address that separately.
- **Old model files cannot be loaded**, because they lack `x_mean` and `x_std`. Every model
  must be retrained after this change, which is needed anyway for a fair comparison.
- It applies to **every** model (baselines included), so the comparison stays fair.
