---
description: Read the model code and draw a detailed flow diagram of the network into docs/model.pdf (TikZ)
allowed-tools: Read, Grep, Glob, Write, Edit, Bash(git rev-parse:*), Bash(git status:*), Bash(latexmk:*), Bash(pdftoppm:*), Bash(rm -f docs/model.*:*)
---

Draw the current demand model as one clear, detailed flow diagram and save it as
`docs/model.pdf`. $ARGUMENTS

## 1. Read the code, never draw from memory

The diagram must match the code **as it is now**, so read it before drawing:

- `noga/model.py`:
  - the config block (`EPOCHS`, `BATCH_SIZE`, `LR`, `WEIGHT_DECAY`, `M_EMBD`, `D_EMBD`, `T_EMBED`, `N_LAYERS`, `HIDDEN_SIZE`, `HISTORY_LEN`, `FEATURES`, `Y_SCALE`, `P_SCALE`, `PRICE_HEADS`, `PRICE_WEIGHT`, `AFFINE_HEAD`, `N`, any input scaling such as `INPUT_GROUPS`);
  - `Model.__init__`, `hidden`, `demand`, `prices`, `training_step`, `price_loss`, `configure_optimizers`;
  - `Data.__init__`, for what each input column is and how it is computed.
- `noga/cost.py`: `loss_fns` and `excess`, for the training losses.

Write down the **actual numbers** (layer widths, embedding sizes, input counts, number of
hidden blocks, activation, any normalisation layers, head sizes) and use them in the
diagram. If a feature is switched off in the config (for example `AFFINE_HEAD` is False),
leave it out or show it as a dashed "optional" box, and say which in the summary.

## 2. What the diagram shows (top to bottom)

1. **Inputs**, grouped and colour-coded, each with its size and a short meaning:
   - the demand history (`HISTORY_LEN` readings, 24 h before t);
   - the weather features (variables × stations);
   - yesterday's day-ahead price;
   - the calendar indices (month, weekday, 5-min slot).
   Mark that the slot's own prices are **not** inputs.
2. **Preprocessing**: input scaling (what it divides or subtracts, and that it uses train statistics), and the embedding lookups with their sizes (e.g. `Embedding(288, 4)`).
3. **Concatenation**, with the total width written as a sum (e.g. `2 + 3 + 4 + 22 = 31`).
4. **Shared trunk**: every layer in order, in the code's exact order (`Linear a→b`, `LayerNorm`, `LeakyReLU`, ...). Repeated blocks are drawn once inside a bracket labelled `× N_LAYERS`. The output is the shared features **z**.
5. **Demand head**: `Linear HIDDEN→HISTORY_LEN+1(+1)`, split into attention logits → softmax → weights, and the correction δ (and the scale, if the affine head is on).
   - Show the **raw** history going around the trunk into a weighted sum.
   - Show the final formula, e.g. `ŷ = Σ wᵢ·histᵢ + δ`.
6. **Price heads** (only if `PRICE_HEADS` is not empty): `Linear HIDDEN→2` from z → log p̂_o, log p̂_u, labelled "cost model only".
7. **Training (side panel or bottom strip, visually separate)**:
   - the loss per model (cost: excess cost with p_da, p_o, p_u; baselines: pinball/L1/L2);
   - the price loss (MSE on log prices × `PRICE_WEIGHT`);
   - the optimiser (AdamW, LR, weight decay applied to Linear weights only), epochs and batch size.

## 3. Style: simple and easy to read

- One `standalone` TikZ file: `\documentclass[tikz,border=8pt]{standalone}` with the `positioning`, `fit`, `arrows.meta`, `calc` and `backgrounds` libraries.
- Rounded boxes with a short **bold title** on the first line and the shapes or formula in a smaller second line. Use `\footnotesize` maths, not long sentences.
- A small fixed palette, one colour per role: inputs, preprocessing, trunk, demand head, price head, training. Use light fills with darker borders, and keep the text black.
- Arrows `-{Stealth}` labelled with the tensor width where it helps (e.g. `31`, `8`).
- Align everything on a grid with `positioning` (`below=6mm of ...`). No overlapping boxes, labels or arrows. Route crossing arrows with `|-` / `-|`.
- A title at the top: "Demand model (`noga/model.py` @ <git short sha>)". Get the sha with `git rev-parse --short HEAD`, and add `+dirty` if `git status --porcelain --untracked-files=no` is not empty.
- A small legend for the colours.

## 4. Build and check

1. Write the source to `docs/model.tex`.
2. Build it: `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=docs docs/model.tex`.
3. Remove the build files: `rm -f docs/model.aux docs/model.log docs/model.fls docs/model.fdb_latexmk`. Keep `model.tex` and `model.pdf`.
4. **Look at it**:
   - render it with `pdftoppm -png -r 110 -singlefile docs/model.pdf <scratchpad>/model` and Read the PNG;
   - fix any overlap, clipped text, crossing arrows or unreadable label, and rebuild;
   - repeat until it is clean.
5. Check every number in the picture against the code once more.

Finish with a short summary: the main structure shown, the sizes used, and anything drawn as optional or left out.
