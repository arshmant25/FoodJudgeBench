# FoodJudgeBench — Script guide (Steps 1–5)

How to run every file built so far, in order: what goes in, what comes out, every parameter, what a good result looks like, and how each piece feeds the project. For the research plan itself see `04_project_steps_guide.md`.

## 0. Setup (once)

```
FoodJudgeBench/
├── data/                     put the HUMMUS CSVs here: pp_recipes.csv, pp_reviews.csv, pp_members.csv  (never commit)
├── processed/                everything the scripts generate (never commit if it holds review text)
├── src/                      all scripts
├── docs/
├── 1_HUMMUS_EDA_v2.ipynb
├── .env                      API keys (never commit)
└── requirements.txt
```

```
pip install -r requirements.txt     # pandas, numpy, pyarrow, scikit-learn, jinja2, requests, jupyter, ...
```
- All commands are run **from the project root** (`FoodJudgeBench/`), e.g. `python src/02_build_catalog.py`.
- Tables are stored as parquet if `pyarrow` is installed, otherwise `csv.gz`. Scripts read either.
- Seed is fixed (`SEED = 42` in `src/common.py`), so reruns give the same profiles and pools.

## 1. Order at a glance

| # | Run | Needs (from earlier) | Produces | Typical time |
|---|---|---|---|---|
| 1 | `1_HUMMUS_EDA_v2.ipynb` (Run All) | `data/*.csv` | cleaned tables in `processed/` | minutes |
| 2 | `python src/02_build_catalog.py` | notebook tables | `catalog`, `recipe_lookup`, `baselines.json`, `catalog_report.json` | minutes |
| 3a | `python src/03a_historical_profiles.py` | step 2 + notebook | `profiles_historical.jsonl`, `…_summary.csv` | < 1 min |
| 3b | `python src/03b_synthetic_profiles.py` | `catalog` | `profiles_synthetic.jsonl`, `…_summary.csv` | < 1 min |
| 4 | `python src/04_recommendation_pool.py` | steps 2, 3a, 3b | `recommendation_pool.jsonl`, `…_summary.csv`, `constraint_check_sheet.csv` | 1–3 min |
| 4b | hand-check, fill `constraint_overrides.csv` | `constraint_check_sheet.csv` | `processed/constraint_overrides.csv` | manual |
| 5 | `python src/llm_client.py … --ping`, then `python src/05_generate_explanations.py …` | steps 3, 4, 4b | `explanations.jsonl` etc. | hours to days (API limits) |

Library files (not run directly): `src/common.py` (used by all), `src/explain_facts.py` and `src/llm_client.py` (used by Step 5; `llm_client.py` can also be run as a connection test).
Optional patches: `patch_03a_oversample.py`, `patch_03a_clean_likes.py` (see Step 3a).

---

## Step 1 — `1_HUMMUS_EDA_v2.ipynb`
**Purpose:** audit HUMMUS and write the cleaned tables everything else uses.
**Input:** `data/pp_recipes.csv`, `pp_reviews.csv`, `pp_members.csv`.
**Output (in `processed/`):** among others `recipes_flagged` (quality flags, diet-rule columns `rule_*`), `reviews_clean` (true reviews, rating 1–5), `authorship`, `user_profile_candidates`.
**Parameters:** one config cell at the top (change there, re-run):

| Name | Default | Meaning |
|---|---|---|
| `MIN_INGREDIENTS` / `MIN_STEPS` | 3 / 2 | recipe quality filter |
| `MAX_DURATION_MIN` | 1440 | longer than a day = suspicious, excluded |
| `MAX_CALORIES` | 5000 | per-serving plausibility limit |
| `MIN_REVIEWS_PER_RECIPE` | 3 | true reviews needed for a recipe to be a catalog candidate |
| `LIKED_RATING` | 4.0 | rating >= this = "liked" |
| `PROFILE_MIN_REVIEWS` | 20 | minimum reviews for a user to be a historical-profile candidate |

**What to expect:** printed audits of ratings (6 = authorship, 2.5 = placeholder), recipes, review text, constraint groundability and profile feasibility. **Contribution:** guarantees the later steps never use authorship rows as ratings and never use junk recipes.

## Step 2 — `src/02_build_catalog.py`
**Purpose:** build the pool of recipes that recommendations can come from, with nutrition percentiles and diet labels.
**Input:** `recipes_flagged`, `data/pp_recipes.csv` (for ingredient quantities and directions).
**Output:** `catalog` (the recipe pool), `recipe_lookup` (slim table of all quality recipes, used for profiling), `baselines.json` (review-weighted base rates), `catalog_report.json`.
**Command:** `python src/02_build_catalog.py` (no CLI flags; constants are at the top of the file):

| Constant | Default | Meaning |
|---|---|---|
| `N_CATALOG` | 6000 | target catalog size (strata minimums can push it higher) |
| `MIN_PER_STRATUM` | 150 | minimum recipes per (health level x duration band) stratum so healthy and unhealthy recipes are both present |
| `MIN_TAG_RECIPES` / `MIN_ING_RECIPES` | 50 / 100 | tags/ingredients rarer than this get no baseline |

**Your result:** 448,652 quality recipes -> 102,317 candidates -> **6,525** in the catalog; 39.1% take 30 minutes or less. Diet status counts (violates / rule_only / verified): vegetarian 2,567/2,950/1,008; vegan 5,312/920/293; dairy-free 3,793/2,727/**5**; gluten-free 3,082/3,316/127.
**Read this:** dairy-free has only 5 tag-verified recipes, so dairy-free cases rely on the ingredient rule. That is why hand-checking labels matters.
**Contribution:** the catalog is the search space for Step 4 and the source of recipe facts for Steps 5–6.

## Step 3a — `src/03a_historical_profiles.py`
**Purpose:** build the 30 historical profiles from real review histories without over-claiming.
**Input:** `user_profile_candidates`, `reviews_clean`, `recipe_lookup`, `baselines.json`.
**Output:** `profiles_historical.jsonl` (full profiles incl. `profile_text` shown to annotators), `profiles_historical_summary.csv`.
**Command:** `python src/03a_historical_profiles.py`
**Parameters (constants at the top):**

| Constant | Default | Meaning |
|---|---|---|
| `N_HISTORICAL` | 30 | number of profiles, spread over sparse/rich x with/without signal |
| `SPARSE_RANGE`, `RICH_RANGE` | (20,50), (100,300) | review-count bands |
| `MIN_RATING_STD`, `MIN_LIKED`, `MIN_SPAN_YEARS`, `MIN_TEXT` | 0.5, 10, 1.0, 15 | user must vary in ratings, have enough liked recipes, history span, text reviews |
| `MIN_N_TENDENCY`, `Z_TENDENCY`, `Z_STRONG` | 10, 3.0, 5.0 | evidence needed before reporting "tends to like higher/lower X" (medium / strong confidence) |
| `MIN_ITEM_COUNT`, `Z_ITEM`, `MIN_LIFT`, `TOP_TAGS`, `TOP_INGS` | 4, 3.0, 1.8, 6, 8 | thresholds for liked/disliked ingredients and tags |
| `MIN_DISLIKED` | 5 | dislikes only reported with at least this many disliked recipes |
| `PATTERN_MIN_N`, `PATTERN_MIN_SHARE`, `PATTERN_Z` | 15, 0.95, 3.0 | a diet "pattern" needs >=95% of liked recipes to satisfy it |

**What to expect:** 30 rows labelled `rich|sparse / with_signal|no_signal`. Every constraint is `possible_self_report` or `observed_pattern`, never confirmed; goal is `unknown` except for a few "possibly prefers lighter eating" profiles.
**Your result:** 16 rich + 14 sparse. Diet patterns appear for 8 profiles; a lighter-eating hint for 3 (H015, H016, H025). Most profiles have no goal and no diet constraint, which is why historical explanations rely on taste and nutrition tendencies.
**Optional patches (run once, then re-run 03a):**
- `python src/patch_03a_oversample.py` — only if Step 4 prints "SKIPPED profiles" (users with too few usable liked recipes).
- `python src/patch_03a_clean_likes.py` — removes junk items (e.g. "use more or less", "ice cubes", equipment/occasion tags, health tags) from "likes".
**Caveat in your current output:** the summary still lists items such as "high-in-something", "bread-machine", "punch", and diet tags like "vegan"/"vegetarian" as likes. Re-running 03a with the patch (and also filtering diet tags from likes) changes the profiles and therefore Steps 4–5, so decide **before** the full generation is finished.

## Step 3b — `src/03b_synthetic_profiles.py`
**Purpose:** build 30 controlled, fully-stated profiles that exercise specific situations.
**Input:** `catalog`.
**Output:** `profiles_synthetic.jsonl`, `profiles_synthetic_summary.csv`.
**Command:** `python src/03b_synthetic_profiles.py [--n-extra 0]`

| Parameter | Default | Meaning |
|---|---|---|
| `--n-extra` | 0 | extra random profiles from a factor grid; keep 0 to stay at exactly 30 |
| `MIN_FEASIBLE` (constant) | 30 | warn if fewer catalog recipes satisfy a profile's hard constraints |

**Composition:** 14 standard (S001–S014), 8 conflict (S015–S022, e.g. low sodium but loves salty), 6 missing-info (S023–S028), 2 explicit "no restrictions" (S029–S030).
**What to expect / your result:** the summary shows how many recipes meet the hard constraints, the liked terms, and the goal. Thin cases: S016 (quick vegan protein, 331 feasible, only 2 meeting all three), S014 (dairy-free + vegetarian, only 1 tag-verified), S007 (dairy-free low sugar, only 4 verified).
**Contribution:** synthetic profiles let RQ2 compare "stated" with "inferred" preferences, and the conflict / missing-info profiles stress-test trade-off communication.

## Step 4 — `src/04_recommendation_pool.py`
**Purpose:** pick which two recipes are explained for each profile, including honest weak matches, and record verified facts about each.
**Input:** `catalog`, both profile files, `reviews_clean`, `recipe_lookup`.
**Output:**
- `recommendation_pool.jsonl` — every candidate with role, score, and `facts` (matches, dislikes hit, per-constraint check with label status, time check, goal check, tendency alignment, nutrition, percentiles, health levels).
- `recommendation_pool_summary.csv` — one readable row per candidate.
- `constraint_check_sheet.csv` — recipes whose diet label needs a human check.
**Command:** `python src/04_recommendation_pool.py` (constants at the top):

| Constant | Default | Meaning |
|---|---|---|
| `N_STRONG` | 3 | strong candidates kept per profile |
| `MMR_MAX_SIM` | 0.85 | 2nd/3rd strong pick may not be more similar than this to an earlier pick |
| `HARD_STATUSES` | stated, observed_pattern | constraint statuses the recommender must respect |
| `W_HIST` | sim 0.85, quality 0.15 | score weights for historical profiles |
| `W_SYN_GOAL` / `W_SYN_LIKE` / `W_SYN_NONE` | 0.5/0.2/0.3; 0.75/0.25; quality only | synthetic score weights by what the profile specifies |
| `MIN_LIKED_FOR_QUERY` | 10 | liked recipes needed to build a historical query |
| `DISLIKE_PENALTY` | 0.5 | centroid-of-disliked weight subtracted for historical |
| `SLOT2_CYCLE` / `SLOT2_FALLBACK` | weak_goal -> weak_taste -> strong#2 | what the second recipe is, with fallbacks |

**Roles:** `strong`, `weak_taste`, `weak_goal`, `weak_dislike`, `violates_constraint` (planned diagnostic, never a normal case). `selected_for_case` + `case_slot` (1 or 2) mark the 120 cases.
**Your result:** 120 selected cases over 60 profiles (slot 1 = 60 strong; slot 2 = 20 strong, 13 weak_goal, 27 weak_taste); 15 planned violating-constraint diagnostics; 16 weak_dislike candidates (not selected).
**Caveats seen in your output:**
- The 120 cases use only **90 distinct recipes**. Historical cases repeat a few popular recipes (e.g. "Banana Spice Bars" 5 times, "Brownie Trifle", "Creamy Oven-Baked Risotto" and "Alice Springs Chicken" 4 times each). This is typical of centroid-similarity recommenders; mention it in the limitations, and consider enforcing a per-recipe cap before Step 6.
- Many historical strong picks have 0 matched liked items, because liked items are sparse; explanations for them will lean on nutrition patterns.
- Slot 2 ended up more weak_taste (27) than the planned ~20/20/20 because many profiles have no goal (no weak_goal candidate).
**Contribution:** the facts stored here are the permitted fact list for Step 5 and the ground truth for diagnostics in Step 6.

## Step 4b — hand-check labels (manual)
Open `constraint_check_sheet.csv`, and for every row check the recipe's ingredient list against the diet label, then record the true status. Corrections go in `processed/constraint_overrides.csv` with columns `recipe_id,constraint,true_status,note`, where `true_status` is `ok`, `violates` or `ambiguous`.
- `ok` -> the check is treated as passed (label "manual").
- `violates` -> the case is replaced by the next clean strong candidate.
- `ambiguous` -> case skipped (unless `--allow-ambiguous`).
**Your result:** all 51 rows were first marked OK; review found 4 wrong labels (croissant / meringue powder / M&M's-type ingredients missing from the rule, and soy/vegan cheese not recognised) and 6 ambiguous. Please confirm each override before the final run.

## Step 5 — Explanation generation

### 5a. `src/llm_client.py` (connection test and shared client)
**Purpose:** one place that talks to any chat API: rate limiting (requests and tokens per minute), retries, 429 handling, a disk cache, and a clean stop when a daily quota is hit.
**Test a connection:** `python src/llm_client.py --provider ollama --model llama3.1:8b --ping`
We will use 

| Parameter | Meaning |
|---|---|
| `--provider` | `groq`, `gemini`, `openrouter`, `nvidia`, `cerebras`, `mistral`, `huggingface`, `ollama`, `custom`, `mock` (offline) |
| `--model` | model name; Groq defaults to `openai/gpt-oss-120b`, Gemini to `gemini-2.5-flash`, others need one (for Ollama, the tag from `ollama list`) |
| `--extra` | JSON of extra request fields, e.g. `'{"reasoning_effort":"low"}'` |
| `--ping` | send one tiny request and print the reply |

**Things to know** 
Speed: about 600 calls takes roughly an hour or two on a GPU, and many hours on CPU. There is no daily cap, so you can leave it running overnight.

Check quality: technique 4 only helps if the model returns valid claim-by-claim JSON. Small models often fail at this. The script retries once, then records check_failed, so you can see how often it happens in the pilot. If it fails often, use a bigger model for just the check step, or use Groq for that part.

One generator for all techniques: Techniques 2, 3 and 4 must all come from the same model. Don’t switch models partway through a full run. The script stops if it detects a change.

Judges: Later you’ll want judge models from different families than the generator. Ollama can serve those too: for example, generate with `llama3.1:8b` and judge with `qwen2.5:14b ` and `gemma2:9b`. Judging will be about 2,000+ calls, which is easy locally and the hardest part on free cloud limits. Local judges are weaker than cloud ones, though, so it’s worth keeping one cloud judge such as Gemini for comparison.
Reproducibility: write the exact model tags (for example llama3.1:8b) in your paper. The cache keys on model and prompt, so reruns reuse results.


### 5b. `src/explain_facts.py` (library)
Builds the **numbered fact list** (F1…Fn) for each case from profile + recipe + match, marks facts that **must be mentioned** (diet failure, time overrun, unmet goal, no liked match, disliked item), cleans ingredient text, renders the **template** technique, strips fact-id citations, and finds numbers that are not on the fact list. Nothing to run.

### 5c. `src/05_generate_explanations.py` (the script you run)
**Purpose:** produce the four explanation techniques for every selected case.
**Input:** `recommendation_pool.jsonl`, `profiles_*.jsonl`, `catalog`, `constraint_overrides.csv`.
**Output:**

| File | Content |
|---|---|
| `case_inputs.jsonl` | per case: profile card, recipe card, facts (what was given to the writers) |
| `explanations.jsonl` | one record per explanation: technique, framing, `text` (clean), `text_raw` (with citations), words, numeric mismatches, cited facts; technique 4 also `audit`, `draft_raw`, `changed_vs_grounded` |
| `explanations_summary.csv` | one row per explanation (easy to open) |
| `generation_report.json` | counts, generator, settings, quality stats, quota status |
| `prompt_preview.txt` | dry run only: facts, template output, prompts |

**Command pattern:**
```
python src/llm_client.py --provider ollama --model llama3.1:8b --ping
python src/05_generate_explanations.py --provider ollama --model llama3.1:8b --dry-run --limit 4  # no API calls; read prompt_preview.txt
python src/05_generate_explanations.py --provider ollama --model llama3.1:8b n --limit 6            # pilot
python src/05_generate_explanations.py --provider ollama --model llama3.1:8b                    # full run, resumable
```

| Parameter | Default | Meaning |
|---|---|---|
| `--provider` | `mock` | API to use; `mock` is an offline fake for testing |
| `--model` | provider default | generator model; **keep the same model for the whole run** |
| `--extra` | – | JSON of extra request fields |
| `--rpm`, `--tpm` | provider default | override requests/min and tokens/min throttles |
| `--techniques` | `template,direct,grounded,check` | subset to run; `check` reuses the grounded draft |
| `--limit N` | all | first N cases, whole profiles at a time, alternating historical/synthetic (pilot) |
| `--framed-n` | 20 | extra goal-framed grounded explanations for profiles with a stated goal (not part of the 480); 0 = none |
| `--max-revisions` | 2 | max revise rounds in technique 4 |
| `--temperature` | 0.4 | writer temperature (the audit always uses 0) |
| `--max-tokens` | 1200 | writer budget (reasoning models spend part on thinking) |
| `--check-max-tokens` | 1800 | budget for the audit |
| `--overrides` | `processed/constraint_overrides.csv` | label corrections file |
| `--allow-ambiguous` | off | keep cases whose label is ambiguous (flagged `label_uncertain`) |
| `--dry-run` | off | build facts and prompts only |
| `--fresh` | off | discard existing `explanations.jsonl` and start over |
| `--preview-n` | 3 | cases shown in the dry-run preview |

**Behaviour to know:**
- **Resumable:** records are appended as they finish; rerun the same command and it continues. Failed ones are retried.
- **Stops cleanly** with the message "STOPPED: quota exhausted" when a daily limit is hit; rerun later.
- **One generator rule:** if `explanations.jsonl` was made by another model, the script refuses to continue unless you pass `--fresh`.
- **Cases:** a case whose recipe is a confirmed violation or ambiguous is replaced by the next clean strong candidate (printed as NOTE) or dropped.
- **Privacy:** only profile summaries, recipe data and facts go to the API — never review text or user IDs.
**Load:** ~600 generation calls for the full set (about 540K tokens). On Groq's free plan (8K tokens/min, 200K/day) this takes several days; on local Ollama it is limited only by your hardware.
**What good output looks like:** 480 explanations of 60–110 words each; template 100% free of invented numbers; direct LLM sometimes with invented personalization; grounded variants with 0% numbers-not-in-facts; in `generation_report.json` -> `check_revise`: a share of explanations revised, `check_failed` close to 0 (if high, the model cannot produce the audit JSON; use a larger model for the check) and a small `unresolved_after_revisions`.
**Contribution:** these are the objects that annotators and judges grade; the four techniques differ in how much evidence the writer gets and whether it self-verifies — exactly the contrast RQ3 and RQ4 test.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `processed/<name>.(parquet|csv.gz|csv) not found` | the earlier step has not been run |
| `missing API key` | add it to `.env` or the shell |
| 404 from the API | wrong model name; run `--ping` and check the provider's model list |
| 429 messages, then "STOPPED: quota exhausted" | daily limit; rerun later, or switch provider / use Ollama |
| `existing explanations were generated by …` | generator changed mid-run; use the same model or `--fresh` |
| many `check_failed` | audit model cannot return valid JSON; use a bigger model |
| Ollama very slow | use a smaller model or a GPU; set `OLLAMA_CONTEXT_LENGTH=8192` |
| `SKIPPED profiles` in Step 4 | run `patch_03a_oversample.py`, then rerun 03a and 04 |

## What comes next
Step 6 (case bank) consumes `explanations.jsonl`, `case_inputs.jsonl` and `recommendation_pool.jsonl`; see `04_project_steps_guide.md`.
