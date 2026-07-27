# Reviewer Comments — Action List

This document consolidates every reviewer annotation found in `comments.pdf`
(highlights, strike-outs, carets, and sticky notes) into a concrete list of
changes required to finalize the paper.

The LaTeX sources live in `tex/`:

| File | Current section |
|------|-----------------|
| `tex/intro.tex` | §1 Introduction |
| `tex/related.tex` | §2 Related Work |
| `tex/dataset.tex` | §3 Dataset |
| `tex/modeling.tex` | §4 Cost-Sensitive Forecasting: Problem Formulation |
| `tex/method.tex` | §5 Method |
| `tex/evaluation.tex` | §6 Evaluation |
| `tex/conclusion.tex` | §7 Conclusion (currently **commented out** in `main.tex`) |

---

## 1. The big picture: restructure the paper (5 sections, not 7)

The single largest theme across the review is a **major structural
reorganization**. The reviewer wants "Related Work" dissolved into the
Introduction, the Dataset folded into a unified "Methods" chapter, and
Results+Discussion merged. The caret renumbering marks (2.3→1.2, Results→4,
Conclusion→5) plus the sticky-note on p.7 imply the following target
table of contents:

### Target structure

```
1  Introduction
   1.1  Asymmetric Costs in Electricity Markets
   1.2  Cost-Oriented and Asymmetric Learning for Load Forecasting
        (← absorbs all of the old "Related Work" section)
2  Cost-Sensitive Forecasting: Problem Formulation and Asymmetric Loss
   (the theory/newsvendor framework — old §4)
3  Methods
   [chapter intro: state the research objective + old §4.1 problem-formulation
    material, with the sub-section headings removed]
   3.1  Dataset            (← old §3)
   3.2  Model Training      (← old §5 "Method" + old §6.1/§6.2 evaluation setup,
        incl. the cost functions used for evaluation and Figure 6)
4  Results and Discussion   (← old §6.3 + §6.4 merged)
5  Conclusion and Future Work
   (built from the old "contributions" list text)
```

> **Ambiguity to resolve with the reviewer / advisor:** The p.7 sticky-note
> asks that old §4.1 (Problem Formulation) become the *intro of the Methods
> chapter*. That conflicts with keeping old §4 as a standalone Section 2.
> Two workable readings:
> - **(A, recommended)** Keep the newsvendor/problem-formulation theory as
>   **Section 2**, and open **Section 3 (Methods)** with a short, explicit
>   *research-objective* statement only. This honors the caret numbering
>   (Results=4, Conclusion=5) cleanly.
> - **(B, literal p.7 note)** Move old §4.1 into the Methods intro; then
>   Section 2 becomes the remaining framework material (§4.2 limitations,
>   §4.3 synthetic costs). This leaves Section 2 thin.
>
> Pick one before doing the mechanical moves below. The per-item instructions
> are written to be compatible with either.

Because sections are renumbered, do a **final pass on all `\ref`/`\Cref`
cross-references** (e.g. "As established in Section 4", "described in Section 5",
"Figure 5 (Section 3.2)") so every internal reference points to the new number.

---

## 2. Introduction — `tex/intro.tex`

### §1.1 Asymmetric Costs in Electricity Markets

- **[Market mechanism is wrong for Israel] (p.3, sticky-note).** The generic
  "producers submit bids… market operator clears the market" description does
  not match the Israeli case. Reviewer: *"the future hourly demand is predicted
  by the operator, which then sets a destination price and informs the
  producers what they should deliver for the next day. It is less chaotic and
  much more centralized than what is described here."* → **Rewrite** the
  opening market description to reflect the more centralized Israeli mechanism
  (NOGA predicts demand → sets price → instructs producers on next-day delivery).

- **[Do suppliers keep their own models?] (p.3, highlight + note on "…affects
  the bidding strategies and financial outcomes of all market participants").**
  Reviewer notes it is an open question whether suppliers maintain their own
  models; likely each supplier draws its share from NOGA's published day-ahead
  prediction based on experience, settling on a stable-but-imperfect price with
  significant error margins — and retail price *could be lower* with better
  prediction (by suppliers or NOGA). → **Add a short passage** acknowledging
  this and framing it as motivation (better forecasting → lower retail price).

- **[Terminology: "market" → "system"] (p.3, highlight).** Change "The **market**
  operator clears the market…" → "The **system** operator…".

- **[Who submits demand bids] (p.3, highlight).** Replace "large consumers and
  aggregators" → **"suppliers (and maybe also a handful of large consumers)"**.

- **[Scope: two sources of price divergence] (p.3, highlight + note on
  "…balancing or real-time markets, where prices can be dramatically different
  from day-ahead levels").** Reviewer: price differences arise from **(1)**
  demand-prediction errors and **(2)** production-capacity prediction errors
  (renewables — solar/wind — being inherently unstable). → **Add a clear scope
  statement** that this paper addresses only the first (demand-side
  uncertainty), not renewable production uncertainty.

- **["short notice" → add purpose] (p.3, highlight).** After "…source additional
  electricity in the real-time market at short notice" add **"to prevent
  blackouts"**.

- **[Soften/reword "consequences"] (p.3, highlight).** "the consequences are far
  more severe" → **"the cost of error is usually much higher"**.

- **[Modernize reference] (p.4, highlight on "(Stoft 2002)" + note).** Reviewer
  wants **more recent / updated sources** describing how these markets work,
  since production sources (renewables) have changed substantially since the
  early 2000s. → **Add newer citations** alongside or replacing Stoft 2002 (in
  `tex/ref.bib`).

- **[Move disconnect paragraph here] (p.4, highlight + note).** The paragraph
  beginning *"Despite the theoretical clarity of this framing, the majority of
  practical forecasting systems… continue to use symmetric loss functions…"*
  should be **moved to become the concluding paragraph of §1.1**.

- **[Insert 3 forecasting-overview paragraphs here] (p.6, note).** Move the three
  paragraphs currently in old §2.2 — **Weron (2014)**, **Hong & Fan (2016)**,
  **GEFCom2014** — into §1.1, **right after** the paragraph ending with
  *"…rational market participant should seek a prediction that minimises expected
  asymmetric cost."*

### §1.2 — remove old heading, create new one

- **[Delete "Contributions and Paper Organisation" heading] (p.4, strike-out of
  "1.2" and its title).** Remove the `\subsection{Contributions and Paper
  Organisation}`.

- **[Contributions list → move to Conclusion; add gap statement] (p.4, highlight
  of "This paper makes the following contributions: 1. …" + note).** Reviewer:
  *"the following paragraphs can be deleted or moved to the conclusion section…
  the reader would like you to actually show what you did without declaring you
  are just about to show it… In the intro, you should clearly define the
  knowledge gap."* → **Remove the enumerated contributions + paper-organization
  from the Introduction.** Keep this text for reuse in the Conclusion (see §6
  below). **In its place, write a clear statement of the knowledge gap.**

- **[Delete "remainder of the paper is organised as follows"] (p.5, highlight +
  note).** Reviewer: *"Can be deleted, the paper's structure is conventional and
  does not require an explanation."* → **Delete** that paragraph.

- **[New §1.2 title & opening] (p.5 note + p.6 caret "2.3 → 1.2").** Create a new
  subsection **§1.2 "Cost-Oriented and Asymmetric Learning for Load
  Forecasting"**. Its **opening two paragraphs** are the ones currently in old
  §2.1 — **Koenker & Bassett (1978)** and **Christoffersen & Diebold (1997)**
  (moved from `related.tex`). Its **body** is the old §2.3 content.

- **[Reframe §1.2 around the research gap] (p.6, highlight + note on "The most
  directly related line of work…").** Reviewer wants a *clearer framing*: state
  not only what prior asymmetric-learning work achieved, but **what is still
  missing** that justifies this research. → **Rewrite the framing**, and make the
  **concluding paragraph explicitly define the research gap** (in light of the
  limitations of the surveyed studies), leading directly into the research
  question.

- **[Differentiate our contribution] (p.7, highlight + note on "Our work").**
  Reviewer: *"It is unclear what differentiates our approach from what you just
  presented."* → **Add one or two sentences** that sharply state what is novel
  here vs. Wang/Lin/Wu/Zhang (e.g. non-linear/generalized asymmetric loss trained
  end-to-end, real operational NOGA baseline, etc.).

---

## 3. Related Work — `tex/related.tex` (section is being dissolved)

The entire "Related Work" section is being folded into the Introduction.

- **[Delete section header + intro] (p.5, strike-out).** Remove
  `\section{Related Work}` (in `main.tex`) and the intro paragraph *"Our work
  sits at the intersection of three research streams…"*.
- **[Delete §2.1 heading] (p.5, strike-out).** "Asymmetric Loss and Quantile
  Regression" — content moves to new §1.2 opening (see above).
- **[Delete §2.2 heading] (p.6, strike-out).** "Electricity Forecasting: Overview
  and Probabilistic Methods" — its 3 paragraphs move into §1.1 (see above).
- **[Delete §2.3 heading, renumber to §1.2] (p.6, strike-out "2.3" + caret
  "1.2").** Content becomes the body of new §1.2.
- Net effect: `related.tex` content is redistributed into `intro.tex`; remove
  `\section{Related Work}\input{related}` from `main.tex` once emptied.

---

## 4. Dataset & Method → unified "Methods" chapter

### `tex/dataset.tex` (old §3)

- **[Restructure into Methods] (p.7, sticky-note on "The dataset is collected").**
  This is the key structural instruction:
  > *"Before describing the dataset, please clearly state the research objective.
  > The dataset should appear as part of the Methods section. Suggested
  > structure: Chapter 3: Methods [intro includes what is now §4.1, removing the
  > sub-section headings]; Section 3.1: Dataset [currently §3]; Section 3.2:
  > Model training [currently the "Method" chapter]."*
  Actions:
  - Create a **Methods** chapter (new Section 3).
  - **State the research objective** clearly at the start (before the dataset).
  - Demote current §3 (Dataset) to **§3.1 Dataset**.
  - See ambiguity note in §1 above regarding placing old §4.1 in the Methods
    intro vs. keeping it as Section 2.

### `tex/method.tex` (old §5)

- Becomes **§3.2 "Model Training"** (subsection of Methods).
- Fold in the evaluation-setup material moved from old §6 (below).

---

## 5. Evaluation → split between Methods and a new "Results and Discussion"

### `tex/evaluation.tex` (old §6)

- **[Delete "Evaluation" section header] (p.19, strike-out "6 Evaluation").**
- **[Delete "Experimental Setup" heading; move content to Methods] (p.19,
  strike-out "6.1"; highlight of "All models share the same architecture…").**
  The experimental-setup text moves into **§3.2 Model Training**.
- **[Delete "Evaluation Cost Functions" heading; move to Methods] (p.20,
  strike-out "6.2"; highlight of "…primary metrics are the Pinball 5:1 and
  Pinball 20:1 costs").** Move this content into Methods.
- **[Move Figure 6 + text to Methods] (p.20, highlight of "Figure 6 illustrates
  the three cost functions…" + note "This too should be part of the last
  sub-section of the Methods chapter").** Move the cost-function figure and its
  paragraph into the last subsection of Methods (§3.2).
- **[Rename to "Results and Discussion", renumber to §4] (p.20, highlight
  "Results" → note "Results and discussion"; strike-out "6.3" + caret "4").**
  The old §6.3 Results becomes **Section 4 "Results and Discussion"**.
- **[Merge Discussion into it] (p.21, strike-out "6.4 Discussion").** Remove the
  separate Discussion subsection heading; its content becomes part of Section 4.

---

## 6. Conclusion — `tex/conclusion.tex`

- **[Write the Conclusion from the contributions text] (p.23, sticky-note).**
  Reviewer: *"I would use the text under the current 'this paper makes the
  following contribution' section as the basis of the Conclusions section."*
  → **Write `conclusion.tex`** using the (removed-from-intro) contributions list
  as its basis, and **un-comment** `\input{conclusion}` in `main.tex` (line 81).
- **[Renumber to §5] (p.23, strike-out "7" + caret "5").** Follows automatically
  once Related Work is merged into the Introduction.

---

## 7. Global / cross-cutting

- **References (`tex/ref.bib`):** add more recent electricity-market sources to
  supplement/replace Stoft (2002) (p.4).
- **Cross-references:** after renumbering, verify every "Section N" / "Figure N"
  mention and `\ref`/`\label` resolves to the new structure. Notable in-text
  references to fix: "As established in Section 4", "described in Section 5",
  "we return to this point in Section 6", "Figure 5 (Section 3.2)".
- **Scope framing:** ensure the paper consistently states it addresses
  demand-prediction uncertainty only (not renewable production uncertainty) —
  introduced in §1.1, reinforced where relevant.
- **Author names / date:** `main.tex` still has placeholder `[Author Names]`
  (line 30) — fill in before submission.

---

## Quick checklist

- [ ] Rewrite §1.1 market mechanism for the centralized Israeli case
- [ ] Add supplier-model / retail-price motivation passage
- [ ] "market operator" → "system operator"
- [ ] "large consumers and aggregators" → "suppliers (and maybe … large consumers)"
- [ ] Add scope statement: demand uncertainty only, not renewable production
- [ ] "short notice" → add "to prevent blackouts"
- [ ] "consequences are far more severe" → "cost of error is usually much higher"
- [ ] Add newer references; supplement/replace Stoft 2002
- [ ] Move "Despite the theoretical clarity…" paragraph to end of §1.1
- [ ] Move Weron / Hong&Fan / GEFCom paragraphs into §1.1
- [ ] Delete "Contributions and Paper Organisation" heading
- [ ] Remove contributions list from intro; add knowledge-gap statement
- [ ] Delete "remainder of the paper is organised…" paragraph
- [ ] Create §1.2 "Cost-Oriented and Asymmetric Learning for Load Forecasting"
- [ ] Move Koenker&Bassett + Christoffersen&Diebold paragraphs to open §1.2
- [ ] Reframe §1.2 with explicit research gap in closing paragraph
- [ ] Add 1–2 sentences differentiating our approach ("Our work")
- [ ] Dissolve Related Work section (remove headers/intro; redistribute content)
- [ ] Create "Methods" chapter; state research objective before the dataset
- [ ] Demote Dataset → §3.1; Method → §3.2 Model Training
- [ ] Move experimental setup + cost functions + Figure 6 into Methods
- [ ] Rename Results → "Results and Discussion" (Section 4); merge old Discussion
- [ ] Write Conclusion from contributions text; un-comment `\input{conclusion}`
- [ ] Renumber all sections (1–5) and fix all cross-references
- [ ] Fill in author names / finalize date
