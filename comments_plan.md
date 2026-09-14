# Plan: Incorporate `comments.docx` edits into `tex/`

## What `comments.docx` contains

`tex/main.docx` (generated from the current `tex/` sources) was sent to Or
Aleksandrowicz for review. `comments.docx` is that file back with Word
Track Changes on: **142 insertions, 67 deletions**, all authored by Or
Aleksandrowicz, plus **one open margin comment**. None of these edits are
in `tex/` yet. Extracted via `pandoc --track-changes=all` and by reading
`word/comments.xml` directly.

## Step 0 — Resolve the one open comment first

Comment (anchored on "wet-bulb temperature" in the Weather features bullet,
`method.tex`/`dataset.tex`): *"Is it something the IMS provides???"*

Checked `noga/data.py`: the Hebrew column `טמפרטורה לחה (°C)` maps directly
to `wet_bulb`, distinct from `dew_point` and plain temperature — the IMS
**does** provide wet-bulb temperature as a raw field, it's not derived by
us. Resolution: keep the term, and reply/resolve the comment by adding a
short clarifying clause (e.g. "...and relative humidity (all raw IMS
fields)...") or a footnote citing the IMS field name, so a future reader
doesn't have the same doubt.

## Step 1 — Apply edits, file by file

**`main.tex`**
- Abstract: "In practice" → "In practice,"; reword "Synthetic asymmetric
  cost functions are generated to assess..." → "We then generate synthetic
  asymmetric cost functions to assess..."
- Add a `Keywords` line — docx only has a placeholder ("..."), so **ask the
  user for the actual keywords** rather than inventing them.
- Add a new **Funding** section before `\printbibliography`: "This work was
  funded by Israel's Ministry of Energy and Infrastructure, grant number
  223-11-056."
- Flag for the user: the docx front matter also carries author emails,
  affiliations-per-author, and an "Authors' contribution" statement that
  have no equivalent in `main.tex` and were *not* part of Or's tracked
  changes (i.e. pre-existing in the submitted docx, possibly journal
  formatting only). Confirm whether these belong in the LaTeX paper too.

**`intro.tex`**
- "the first source, uncertainty" → "the first source of differences,
  uncertainty"
- "the majority of practical forecasting systems" → "most practical
  forecasting systems"
- Spell out "MSE" → "mean squared error (MSE)" on first use in that
  sentence (parallel to "mean absolute error (MAE)")
- Literature-review verbs → past tense: "train" → "trained" (Wang et al.),
  "address"/"propose" → "addressed"/"proposed" (Lin et al.), "extend"/
  "achieves" → "extended"/"achieved" (Wu et al.), "formalise" →
  "formalised" (Zhang et al.)
- "this body of work" → "this recent body of work"
- "leaving open whether" → "leaving open the question of whether"
- Restructure: move the "Our work is designed to close both gaps..."
  paragraph out of the end of §1.2 and into the start of the new
  **Contributions** subsection, reworded as "Our work is designed to
  overcome both gaps currently existing in the literature described
  above. We formulate...", dropping the final "characterised formally in
  Section 2" sentence (now redundant with the itemised list).
- Contribution item 1: drop the trailing "(Sections 2–3)" parenthetical.

**`modeling.tex`**
- Merge the paragraph break after "Let $c_o>0$ and $c_u>c_o$ be the
  per-unit overage and underage costs." into the following sentence ("This
  is a well known problem...") — one paragraph, not two.
- Merge the paragraph break after "...it does not always capture the
  complexities of real-world electricity markets." with the following
  sentence in §2.2.
- Merge the paragraph break after "...synthetic cost functions that
  satisfy the qualitative properties stated in the problem formulation."
  with the following sentence in §2.3.
- "we evaluate our models" → "we evaluated our models"; "we conduct our
  main evaluation" → "we conducted our main evaluation"
- "ratios of $1{:}5$ and $1{:}20$ respectively" → add comma: "...
  respectively,"

**`dataset.tex`** (input of `method.tex` §3.1)
- **Factual correction**: Israel's area 22,145 km² → **20,770 km²**
- Population "9.8 million" → "10 million", add "as of 2026"
- "Despite this small size, the population is highly concentrated" →
  "The population geographic distribution is highly concentrated"
- Population-density figure caption: append "Population density is per 1
  square kilometre."
- Insert new sentence after the Noga/IMS source citations: "In Israel,
  where the percentage of households with air conditioning systems is
  currently 96.5% (ranging from 89% in the lower decile to 98.3% in the
  upper decile) according to data from Israel's Central Bureau of
  Statistics, there is a strong correlation between electricity
  consumption levels and building cooling and heating loads resulting from
  changing weather conditions and building occupancy trends."
- Weather sentence rewrite: "The weather data includes temperature,
  humidity, and wind speed measurements..." → "The weather data includes
  air temperature and relative humidity, whose combined effect controls
  building cooling loads, alongside wind speed measurements..."
- Add "air" before "temperature" consistently: EDA paragraph, Figures 2–4
  captions, and Table 1 row labels ("Temp." → "Air Temp." for Tel Aviv/
  Jerusalem/Haifa rows).
- Figure 2/Table 1 date range: "March 2023–February 2026" → "1 March 2023
  to 28 February 2026" (spell out day numbers, matches the "1,096 days"
  sentence already in the text).
- Figure 2 caption: add "and observed air temperature".

**`method.tex`** (§3.2 Model Training)
- "The objective of this study is" → "was" (past tense, matches the rest
  of the Methods section describing completed work)
- Weather-conditions list: "temperature, wet-bulb temperature" → "air
  temperature, wet-bulb temperature" + resolve the open comment (Step 0)
- Same "temperature" → "air temperature" fix in the Weather-features
  bullet under Neural Network Architecture
- "at learning rate" → "at a learning rate"
- Evaluation Protocol: tense fixes to match past-tense narration — "we
  compare" → "we compared", "we additionally evaluate" → "we additionally
  evaluated", "Models are evaluated" → "Models were evaluated"

**`evaluation.tex`**
- Remove trailing periods from the four `\paragraph*{...}` headings ("Each
  model excels on its own training metric.", "Asymmetric training
  substantially outperforms the symmetric baseline.", "Post-hoc
  calibration narrows but does not close the gap.", "Error distributions
  reflect the targeted quantile.") — headings shouldn't end in a full stop.

**`conclusion.tex`**
- Opening sentence: "This paper addressed day-ahead electricity demand
  forecasting under asymmetric operational costs." → "...under asymmetric
  operational costs using a 3-year national-scale, high-resolution dataset
  of electricity use prediction and actual consumption."
- "In electricity markets the costs" → "In electricity markets, the costs"
- "For a cost ratio of 5 this yields" → "For a cost ratio of 5, this
  yields"
- Practicality finding: "...even under symmetric error metrics such as MSE
  --- and we believe substantial further gains are readily achievable." →
  "...also under symmetric error metrics such as MSE; we believe
  substantial further gains in this aspect are readily achievable."
- Add a lead-in sentence before the Limitations bullet list: "Along with
  these positive findings, this work also has several limitations, as
  follows:"
- Future-work item on temporal modelling: replace the em-dash-parenthetical
  ("--- for example with Transformer architectures ... or temporal
  convolutional networks --- together with...") with a comma-based
  parenthetical, and insert "the" before "time of day" in the concluding
  remarks' fuel-prices clause.
- Concluding Remarks: merge the final two paragraphs ("Realising it is the
  main direction of our future work." / "We hope these findings
  contribute...") into a single paragraph.

## Step 2 — Style-wide dash cleanup (verify, don't blindly apply)

A large share of the tracked "changes" are `---` (em dash) being replaced
by a single `-`, scattered throughout every section. This looks like a
Word round-trip artifact rather than an intentional edit by the reviewer —
applying it literally would strip em-dashes from the whole paper. **Do not
apply these mechanically**; when in doubt, keep the existing LaTeX
`---`/`\emph`-style dashes and only change punctuation where it's paired
with an actual wording change (e.g. the comma-for-dash rewrites already
listed above under `conclusion.tex`).

## Step 3 — Rebuild and proofread

1. Apply all edits above directly to the `.tex` files (not `main.docx`).
2. `cd tex && latexmk -pdf main.tex` (or run the project's normal build) to
   regenerate `main.pdf`.
3. Diff the rendered PDF against the previous version section by section
   against this checklist.
4. Re-export `main.docx` from the updated `.tex` if a clean (no-tracked-
   changes) docx copy is needed for the co-authors.

## Open questions for the user before implementing

1. Actual keywords to fill in `Keywords: ...`.
2. Whether the docx's author emails / per-author affiliations / "Authors'
   contribution" statement should be added to `main.tex`, or are
   submission-portal-only metadata outside the LaTeX source.
3. Confirm the Israel area correction (22,145 → 20,770 km²) and funding
   grant number (223-11-056) are correct before they go in the final PDF.
