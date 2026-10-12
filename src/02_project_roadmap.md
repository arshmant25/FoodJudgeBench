# FoodJudgeBench — Project Roadmap

**Question:** can an LLM judge food recommendations *and their explanations* as reliably as a human, given the same profile, recipe, evidence and rubric?
**RQ1** overall human–LLM agreement · **RQ2** historical vs synthetic profiles · **RQ3** can judges rank explanation techniques like humans · **RQ4** which judging setup catches which failure type (invented personalization, factual error, ignored constraint).
Out of scope: persuasion / real behaviour change.

## Where we are

| Step | Status |
|---|---|
| 1 Research objectives | done |
| 2a Data audit (`1_HUMMUS_EDA_v2.ipynb`) | **done** — see `01_preprocessing_report.md` |
| 2b Catalog build (`src/02_build_catalog.py`) | **next** |
| 3 Profiles (`src/03a_…`, `src/03b_…`) | code ready |
| 4–11 | below |

## Run order for the next session

```
pip install pyarrow            # optional but recommended
python src/02_build_catalog.py
python src/03a_historical_profiles.py
python src/03b_synthetic_profiles.py --n-extra 0
```
Then **read every profile** (`processed/profiles_*_summary.csv`, `profile_text` in the jsonl) and fix thresholds before moving on.

## Project layout

```
FoodJudgeBench/
├── data/                      raw pp_*.csv  (gitignored)
├── processed/                 generated tables, profiles, cases (gitignored if it contains review text)
├── docs/                      reports, rubric, decision log
├── src/                       common.py, 02_…, 03a_…, 03b_…, later 04_…–10_…
├── 1_HUMMUS_EDA_v2.ipynb
└── requirements.txt
```

## Steps from here

**Step 2b — Catalog (½ day).** Run `02_build_catalog.py`. Check `catalog_report.json`. Manually spot-check ~100 `rule_only` recipes for each diet rule (is the "vegetarian"/"gluten-free" label right?). Record precision in the decision log.

**Step 3 — Profiles (1 day).**
- Historical (target 12): stratified sparse/rich × with/without evidence signal. Every claim carries `status` (`observed_pattern`, `possible_self_report`, `unknown`) and `confidence`; health goal stays `unknown` unless ≥3 health mentions *and* a matching "lighter" tendency. Read each profile against the user's actual reviews; drop or edit weak ones.
- Synthetic (12 archetypes + optional random draws): standard, **conflict** (taste vs goal, tight time + vegan + protein), **missing info**, explicit-none baseline. Each profile is feasibility-checked against the catalog (≥30 recipes satisfying hard constraints; warnings for thin liked/goal pools).
- Both use **one schema** (`dietary_constraints`, `health_goal`, `time_budget_minutes`, `taste`, `nutrition_tendencies`, `unknown_fields`, `evidence`, `profile_text`), so RQ2 compares profile *source*, not format.

**Step 4 — Recommendation pool (1 day).** One method: content-based (TF-IDF over normalized ingredients + informative tags, cosine to the centroid of liked recipes for historical profiles; to term/goal-weighted query for synthetic). Per profile: 2 strong picks + 1 deliberately weak pick (e.g. poor taste fit or violates a soft goal, but **not** an unmarked hard-constraint violation unless it is a planned diagnostic). Exclude recipes the user already reviewed. Record method + evidence per candidate. Why content-based: HUMMUS is extremely sparse (99.998%) and the HUMMUS authors found collaborative filtering weak on it.

**Step 5 — Explanations (1 day).** Three techniques, not three models:
1. *Template* (jinja2) from verified attributes only.
2. *Direct LLM* from profile + recipe, free writing.
3. *Evidence-grounded LLM* — may only use a permitted fact list built from the profile evidence and catalog fields (the RAG-flavoured one).
Plus a neutral vs goal-framed pair on a subset to isolate "nudge framing". Hide technique identity.

**Step 6 — Case bank (½ day).** Case = profile + evidence + recipe (title, ingredients with quantities, directions, nutrition percentiles, health levels) + explanation (+ alternative for pairwise). Target **~72 cases** (24 profiles × 1 recipe × 3 techniques) **+ 8 diagnostic cases** with an injected error: invented preference, wrong nutrition comparison, ignored time limit, ignored dietary constraint. Scrub emails/URLs from any review snippet. Verify constraint labels by hand for each recipe used.

**Step 7 — Rubric (½ day, parallel with 4–5).** Binary violation flags (ignored constraint, factual/nutrition error, invented personalization, claim unsupported by evidence); ordinal 1–5 (clarity, trade-off communication, nudge proportionality); goal alignment allows N/A. Measures suitability *for the stated profile*, not personal taste.

**Step 8 — Human annotation (1 day).** Pilot on ~10 cases, fix wording, then both annotators score all cases independently and blind. Add ~15 pairwise "which explanation is better" items.

**Step 9 — LLM judges (1 day, starts as soon as the case bank is frozen).** Two judging styles (plain rubric vs constraint-checklist), 1–2 judge models, same case + same rubric as humans. Randomize pairwise order to measure position bias; check self-preference if a judge also generated explanations. Tool-assisted numeric verification only if time allows.

**Step 10 — Analysis (1 day).** Krippendorff's α between the two humans; human consensus vs judge; precision/recall on diagnostic cases; break down by profile source (RQ2), technique (RQ3), error type × judge style (RQ4); sparse vs rich historical profiles. Always report where the judge breaks down, not just overall agreement.

**Step 11 — Release & report (1 day).** README, decisions log, scripts to rebuild everything from raw HUMMUS, profiles, cases, annotations, judge prompts, analysis notebook. Do **not** publish raw Food.com text without checking the terms.

## Scope (documented limitation)
~12 historical + ~12 synthetic profiles, 1 recommender, 3 explanation techniques, 2 human annotators, 1–2 judge models. This is a pilot; the full plan (30+30 profiles, 480 cases, 3 annotators) is out of reach in the time available. Skip Recall@K/NDCG@K held-out testing.

## Decision log (seed entries)
1. Rating 6 = authorship → removed from reviews.
2. Rating 2.5 = no-stars placeholder → treated as unrated (verify).
3. No temporal split (316 users at best); exclude already-reviewed recipes instead.
4. Health scores: higher = healthier; use percentiles.
5. `normalization_comment` recipes excluded.
6. Dietary labels from ingredient rules with tag-verification status; hand-check case-bank recipes.
7. Review mentions of diets/health = weak evidence; never promoted to confirmed constraints or goals.
8. Baselines are review-weighted to avoid popularity bias in "tendency" tests.
