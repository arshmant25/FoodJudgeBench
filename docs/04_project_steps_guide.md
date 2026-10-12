# FoodJudgeBench — Project steps guide (original research plan)

**Core question:** can an LLM judge grade food recommendations **and their explanations** as reliably as human annotators, when both see the same profile, recipe, evidence and rubric?

| RQ | Question | Answered by |
|---|---|---|
| RQ1 | How well do LLM judges agree with humans overall? | Step 10 (agreement, all cases) |
| RQ2 | Does agreement differ for historical (inferred) vs synthetic (stated) profiles? | Step 10 (split by profile source) |
| RQ3 | Can judges rank the explanation techniques the way humans do? | Step 10 (technique means / rankings) |
| RQ4 | Which judging setup best catches invented personalization, factual errors and ignored constraints? | Step 9 + Step 10 (diagnostic cases x judge setup) |

Out of scope: persuasion or real behaviour change.

## The plan in numbers (fixed — do not change mid-project)

| Item | Value |
|---|---|
| Profiles | **30 historical** (from HUMMUS review histories) + **30 synthetic** (designed) = 60 |
| Recipes per profile | **2** -> 120 recipe-profile pairs |
| Explanation techniques | **4** (template, direct LLM, evidence-grounded LLM, grounded LLM that checks its own claims and revises) |
| Explanation cases | 120 x 4 = **480** |
| Human annotators | **3** per case (1,440 human ratings if every case is rated by all three) |
| LLM judges | **several models**, from different families than the explanation generator |
| Generalization | **held-out tests** (see Step 10) |
| Extras (not in the 480) | ~20 goal-framed grounded explanations; ~15 planned diagnostic cases with an injected error |

## Where the project stands

| Step | What | Status |
|---|---|---|
| 1 | Research objectives and plan | done |
| 2 | Data audit (notebook) and recipe catalog | done |
| 3 | 30 historical + 30 synthetic profiles | done |
| 4 | Recommendation pool (2 recipes per profile, plus weak and violating candidates) | done |
| 5 | Generate the 4 explanation techniques | **running** |
| 6 | Case bank (freeze the 480 cases + diagnostics) | next |
| 7 | Annotation rubric | to do (can be written in parallel with 5) |
| 8 | Human annotation (3 annotators) | to do |
| 9 | LLM judging (several models, several setups) | to do |
| 10 | Analysis + held-out tests | to do |
| 11 | Release and report | to do |

Run order, with exact commands for Steps 2–5, is in `05_script_guide.md`.

---

## Step 1 — Research objectives (done)
Fixed the four RQs, the scope above, and the rule that a judge sees **exactly** what a human sees (same profile, recipe, evidence, rubric).

## Step 2 — Data audit and catalog (done)
**Purpose:** make HUMMUS (Food.com: ~507k recipes, ~1.9M interactions) safe to build cases from.
- Notebook `1_HUMMUS_EDA_v2.ipynb` audits the data and writes cleaned tables.
- `02_build_catalog.py` builds a stratified catalog of **6,525 recipes**, percentiles for nutrition, review-weighted baselines, and a diet label per recipe (`violates` / `verified` / `rule_only`).

Key data findings that shape everything downstream:
- Rating `6` is an authorship record, not a rating; rating `2.5` is probably a "no stars" placeholder.
- There is only one timestamp per record, so no temporal train/test split.
- Dietary labels come from ingredient rules, so they can be wrong. `verified` means a recipe tag agrees; `rule_only` means only our rule says so. Every recipe used in a case must be hand-checked (see Step 6).
- Health scores: higher = healthier.

## Step 3 — Profiles (done)
**Purpose:** the "person" each recommendation is for. Both sources use **one schema** so RQ2 compares *where the profile came from*, not its format.
- **Historical (30):** users from the review data, balanced over sparse (20–50 reviews) / rich (100–300 reviews) x with / without a signal. Likes, dislikes, nutrition tendencies and diet patterns are **inferred** with significance tests, and every claim carries a status and confidence. Nothing inferred is ever treated as confirmed. Health goal stays `unknown` unless clearly supported.
- **Synthetic (30):** hand-designed archetypes — 14 standard, 8 with conflicting priorities (for example "low sodium but loves salty food"), 6 with missing information, 2 that explicitly state "no restrictions". Everything in them is **stated**. Each was checked for enough matching recipes in the catalog.

## Step 4 — Recommendation pool (done)
**Purpose:** decide *which* recipes get explained, and make sure not every explanation is for a perfect match.
- One method only: content-based (TF-IDF similarity to the user's liked recipes for historical profiles; liked-term / quality / goal score for synthetic ones). HUMMUS is far too sparse for collaborative filtering.
- Per profile it tags candidates: `strong` (top 3), `weak_taste`, `weak_goal`, `weak_dislike`, `violates_constraint`.
- Two are flagged `selected_for_case`: **slot 1 = best strong match; slot 2 cycles weak_goal -> weak_taste -> second strong match.** The weak picks give the explanations something honest to say about trade-offs ("this does not fit your sodium goal").
- `violates_constraint` candidates are **never** normal cases; they are the planned diagnostics for Step 6.
- Each candidate stores verified `facts` (nutrition, time check, diet checks, matches). These facts are the ground truth for Steps 5, 6 and 9.

Real result (your run): 120 selected pairs over 60 profiles. Slot 1 = 60 strong. Slot 2 = 20 strong, 13 weak_goal, 27 weak_taste. 15 planned violating-constraint diagnostics.

## Step 5 — Explanation generation (running)
**Purpose:** write the explanation text that annotators and judges will grade.

| Technique | Input to the writer | Why it is in the study |
|---|---|---|
| 1 Template | verified facts only, fixed sentences | accurate but stiff baseline |
| 2 Direct LLM | profile card + recipe card | realistic baseline; may invent personalization |
| 3 Evidence-grounded LLM | numbered permitted-fact list | fluent and constrained |
| 4 Grounded + self-check | technique 3's draft, then a claim-by-claim audit and revision (max 2 rounds) | tests whether self-verification improves faithfulness |

Output: `processed/explanations.jsonl` (480 records plus ~20 framed extras). One generator model for all LLM techniques; fact-id citations are stripped before annotators see the text.

## Step 6 — Case bank (next)
**Purpose:** freeze exactly what annotators and judges will see.
1. **Re-check labels.** Recompute the planned diagnostics after `constraint_overrides.csv` (some "violations" may be false), and confirm the dietary label by hand for every recipe in the 480.
2. **Clean the explanations.** Drop empty / errored ones; if technique 4's text is identical to technique 3's (`changed_vs_grounded = False`), keep both but flag it, since a human will see the same text twice.
3. **Assemble each case:** profile (with evidence), recipe (title, ingredients with quantities, directions, nutrition percentiles, health levels), the explanation, and the hidden technique label.
4. **Add ~15 diagnostic cases** with an injected error of a known type: invented preference, wrong nutrition comparison, ignored time limit, ignored dietary constraint.
5. **Scrub** emails/URLs from any review-derived text; randomize case order; hide technique identity.
6. Output: a case bank file, an annotation sheet per annotator, and a hidden answer key.

## Step 7 — Rubric (can run in parallel with Step 5)
Binary violation flags: ignored constraint, factual or nutrition error, invented personalization, claim unsupported by the evidence. Ordinal 1–5: clarity, trade-off communication, nudge proportionality. Goal alignment allows N/A. The rubric measures **suitability for the stated profile**, not personal taste. Write it, pilot it on ~10 cases, fix unclear wording **before** Step 8.

## Step 8 — Human annotation (3 annotators)
- Pilot ~10 cases together, resolve disagreements about wording, then annotate independently and blind.
- Workload: 480 cases x ~2.5 min = about 20 hours per annotator if each rates every case. Decide early whether all three rate all 480, or all three rate a stratified subset (all diagnostics + a balanced sample across source x technique) — state the rule in the report.
- Add ~15 pairwise "which explanation is better" items.
- Output: 3 rating files -> human consensus (majority for flags, mean/median for ordinal) + agreement between humans (Krippendorff's alpha).

## Step 9 — LLM judges
- Several judge models, **different families from the generator** (to limit self-preference), shown the same case and rubric as humans.
- Several judging setups, e.g. plain rubric vs constraint-checklist, with and without the fact list, so RQ4 can say which setup catches which error.
- Randomize pairwise order to measure position bias; run each judge more than once to measure stability.
- Output: one file per judge x setup with the same fields as the human sheets.

## Step 10 — Analysis and held-out tests
- Human consensus vs judge: correlation / agreement per rubric item (RQ1), split by profile source (RQ2) and technique (RQ3), with technique rankings compared against the human ranking.
- Precision / recall per error type and judge setup on the diagnostic cases (RQ4).
- **Held-out generalization tests** (interpretation — confirm before building): choose judge prompts / setups using a development split of **profiles** (never split a profile's cases), then report agreement only on the held-out profiles. Also report held-out by source (develop on historical, test on synthetic, and the reverse) and by technique.
- Always report where judges break down, not only overall agreement.

## Step 11 — Release and report
README, decision log, scripts that rebuild everything from raw HUMMUS, profiles, case bank, annotations, judge prompts and the analysis notebook. **Do not publish raw Food.com text or user IDs** without checking the terms (they limit use to educational/scientific purposes); keep `data/` and `.env` out of any public repository.

---

## Decision log
1. Rating 6 = authorship -> removed from reviews.
2. Rating 2.5 = no-stars placeholder -> treated as unrated.
3. No temporal split -> already-reviewed recipes are excluded instead.
4. Health scores: higher = healthier; percentiles used throughout.
5. `normalization_comment` recipes excluded.
6. Dietary labels come from rules with tag verification; case recipes are hand-checked. 4 label errors and 6 ambiguous labels were found in the Step 4 hand-check -> `constraint_overrides.csv`.
7. Review mentions of diets or health are weak evidence; never promoted to confirmed constraints or goals.
8. Baselines are review-weighted to avoid popularity bias.
9. Plan fixed at 30 + 30 profiles x 2 recipes x 4 techniques = 480 cases, 3 annotators, several judges.
10. Technique 4 (grounded + self-check) was added by the researcher.
11. Free LLM APIs or local Ollama for generation; judges from other model families.

## Open items to decide
- Whether all 3 annotators rate all 480 cases.
- Exact meaning of "held-out generalization" (see Step 10).
- Which models generate and which judge (only generation model is fixed per run).
- Whether to re-run Step 3a with the likes-cleaning patch (see caveats in `05_script_guide.md`); doing so changes all downstream steps.
