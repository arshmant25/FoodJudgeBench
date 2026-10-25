# FoodJudgeBench

**Can LLM judges grade personalized food recommendations and their explanations as reliably as human annotators?**

FoodJudgeBench is a benchmark and a reproducible pipeline built on the [HUMMUS](https://gitlab.com/felix134/connected-recipe-data-set) recipe dataset (Food.com recipes and reviews with healthiness scores). It builds 60 user profiles (30 inferred from real review histories, 30 synthetic), generates explanations with four techniques, collects human ratings with a ten-question rubric, runs three open-weight LLM judges, and measures how closely the judges agree with humans.

University of Oulu, NLP course project. Team: Arshman Tariq (team leader), Muhammad Faizan Tnaveer, Faisal Khan.
Project ID / group: 22. Report: `report/report.tex` (IEEE template).

## Status at a glance

| Part | Status |
|---|---|
| Data audit, catalog, 60 profiles, recommendation pool | done |
| 4 explanation techniques, 480 blinded cases, 15 diagnostics, 15 pairs, hidden answer key | done |
| Rubric, offline annotation forms, annotator guide | done |
| Human annotation | **pilot only** (10 items x 3 annotators). The 510-item main round is outside the scope of our project |
| LLM judges (3 families x 3 prompt setups) and analysis (RQ1 to RQ4) | done, on the pilot items |
| Public release folder `processed/release/FoodJudgeBench_v0.1` | done (recipe text withheld) |

All numbers produced so far come from the 10-item pilot, so they demonstrate the pipeline and do not rank the judges.

## Research questions

- **RQ1** How closely do LLM judgments agree with human assessments?
- **RQ2** Does agreement differ between historical (inferred) and synthetic (stated) profiles?
- **RQ3** Do judges rank the four explanation techniques like humans?
- **RQ4** Which judging setup best detects invented personalization, factual errors and ignored constraints?

## Repository structure

```
FoodJudgeBench
├── README.md                    this file
├── requirements.txt             Python libraries with exact versions
├── foodjudgebench_pipeline.png  pipeline diagram used in the report
├── report/report.tex            IEEE-format report (upload with the figures to Overleaf)
├── data/                        HUMMUS files (NOT in the repository, see "Get the data")
├── docs/
│   └── annotator_guide.md/.docx instructions given to the human annotators
├── src/                         all code (Jupyter notebooks + config)
│   ├── config.py                paths and helpers shared by all notebooks
│   ├── .env                     API keys, only if a hosted LLM is used (NOT in the repository)
│   └── 1_ ... 10_ *.ipynb       the pipeline, run in numeric order
└── processed/                   every file the notebooks write (see table below)
```

## Which file does which task

Notebooks are run in this order. Each reads the output of the previous ones. The project plan step (from the research plan) and the report section are given for each.

| Notebook | Task (plan step) | Main inputs | Main outputs (in `processed/`) | Report |
|---|---|---|---|---|
| `src/1_HUMMUS_EDA.ipynb` | Audit and clean HUMMUS (2) | `data/*.csv` | `reviews_clean`, `authorship`, `recipes_flagged`, `user_profile_candidates`, `data_quality_summary.json`; figures `NB1_*.png` | II-B, dataset table, EDA figures |
| `src/2_Building_Catalog.ipynb` | Recipe catalog with nutrition percentiles and diet labels (2) | `recipes_flagged` | `catalog`, `recipe_lookup`, `baselines.json`, `catalog_report.json` | III-B |
| `src/3_Profiling.ipynb` | 30 historical + 30 synthetic profiles (3) | `user_profile_candidates`, `reviews_clean`, `catalog` | `profiles_historical.jsonl`, `profiles_synthetic.jsonl` and summaries | III-C |
| `src/4_Recommendation_Pooling.ipynb` | Content-based recommendation pool, selection of 2 recipes per profile (4) | profiles, `catalog` | `recommendation_pool.jsonl`, `constraint_check_sheet.csv` | III-D |
| `src/5_Building_Explanations.ipynb` | Four explanation techniques, check-and-revise loop (5) | pool, profiles, `constraint_overrides.csv` | `explanations.jsonl`, `explanations_summary.csv`, `generation_report.json`, `case_inputs.jsonl` | III-E, generated-text table |
| `src/6_Case_Bank.ipynb` | Blinded case bank, diagnostics, pairs, rubric, annotation forms (6, 7) | `explanations.jsonl`, `case_inputs.jsonl` | `case_bank/` (`case_bank.jsonl`, `answer_key.csv`, `rubric.json`, `forms/*.html`, assignments) | III-F, III-G |
| *(step 7, offline)* | Humans fill the HTML forms and export CSV | `case_bank/forms/` | `annotations/incoming/ratings_<pilot or main>_A<n>.csv` | III-H |
| `src/8_Annotation_Collection.ipynb` | Validate, merge, agreement statistics (8) | `annotations/incoming/` | `annotations/annotations_<mode>_raw.csv`, `..._consensus.csv` | III-H, human-agreement table |
| `src/9_LLM_Judge.ipynb` | Run LLM judges (9) | human-rated items, `case_bank/` | `judgments/judgments_<judge>_<setup>.csv`, `judge_run_report.json` | III-I |
| `src/10_Analysis_and_Release.ipynb` | Agreement analysis RQ1 to RQ4 and release (10, 11) | annotations, judgments | `results/*.csv`, `results/fig_*.png`, `results/insights.md`, `release/FoodJudgeBench_v0.1/` | IV, RQ1 to RQ3 tables |

There is no notebook 7: the rubric is defined in Notebook 6 and saved as `case_bank/rubric.json`.

## Which file produces which table or figure of the report

| Report item | Produced by | File |
|---|---|---|
| Dataset audit table and EDA figures | Notebook 1 | `NB1_*.png`, notebook printouts |
| Generated-text statistics table | Notebook 5 | `generation_report.json`, `explanations_summary.csv` |
| Human-agreement table | Notebook 8 | printed table, `annotations_pilot_consensus.csv` |
| RQ1 table | Notebook 10 | `results/rq1_overall.csv` |
| Agreement heatmap figure | Notebook 10 | `results/fig_rq1_heatmap.png` (`NB10_1_Kappa_Agreement_Heatmap.png`), data in `results/rq1_by_question.csv` |
| Yes-rate table | Notebook 9 | printed summary, judgment CSVs |
| RQ2 table | Notebook 10 | `results/rq2_by_source.csv` |
| RQ3 table | Notebook 10 | `results/rq3_technique_means.csv` |
| RQ4 detection figure | Notebook 10 | `results/fig_rq4_detection.png` (`NB10_2_Detection_of_Injected_errors.png`), `results/rq4_detection.csv` |

## Setup

Python 3.11 or newer (_[state the exact version you used]_). Create an environment and install the pinned libraries:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins every library (for example pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, pyarrow 26.0.0, ollama 0.6.3). Open the notebooks with Jupyter, VS Code or Google Colab.

### Get the data

The HUMMUS files are not redistributed here. Download them from the [HUMMUS repository](https://gitlab.com/felix134/connected-recipe-data-set) and place `pp_recipes.csv`, `pp_reviews.csv` and `pp_members.csv` in `data/`. The original recipe and review data come from Food.com; use is subject to its terms (educational and scientific use). Please cite the HUMMUS paper (Bölz et al., RecSys 2023, doi 10.1145/3604915.3609491).

### Local LLMs (Ollama)

All language models run locally through [Ollama](https://ollama.com). Install it, then pull the models:

```bash
ollama pull llama3.1:8b      # explanation generator (Notebook 5)
ollama pull qwen2.5:14b      # technique-4 auditor (Notebook 5) and judge (Notebook 9)
ollama pull gemma2:9b        # judge
ollama pull phi4             # judge
```

A GPU with about 12 to 16 GB of memory is comfortable for the 14B model. A hosted OpenAI-compatible API can replace Ollama: set the provider in the configuration cell and put the key in `src/.env` (never commit this file).

## How to run

Open each notebook from inside the project folder (they find the root through `src/config.py`) and run all cells in order.

1. **Notebooks 1 to 4** (data, catalog, profiles, pool). Minutes each, no LLM needed.
2. **Notebook 5** (explanations). Needs Ollama with `llama3.1:8b` and `qwen2.5:14b`; about 1.5 to 2 hours for all 480 cases on a laptop GPU. It is resumable: rerun to continue. Put your hand-checked diet labels in `processed/constraint_overrides.csv` before running (columns `recipe_id, constraint, true_status`; allowed values `ok`, `violates`, `ambiguous`).
3. **Notebook 6** (case bank). Writes the annotator forms. Settings at the top: `SEED`, `ANNOTATORS`, `N_PILOT`, `N_PAIRS`. Do not change the seed after annotators have started, because the item identifiers depend on it.
4. **Annotation (offline).** Send each annotator their `case_bank/forms/<mode>_A<n>.html` together with `docs/annotator_guide`. Annotators open the file in a browser (progress autosaves) and send back the exported CSV. Save the files as `processed/annotations/incoming/ratings_pilot_A1.csv`, and so on (`ratings_main_A1.csv` for the main round).
5. **Notebook 8** (collect). Set `MODE = "pilot"` or `"main"`. It validates the files and prints agreement per question (Krippendorff alpha, Cohen kappa, exact agreement, within-one-point), the use of "not applicable", disputed items and diagnostic catch rates.
6. **Notebook 9** (judges). Set `HUMAN_FILES` (for example `["annotations_pilot_raw.csv"]`), `JUDGES`, `SETUPS` (`rubric_facts`, `rubric_nofacts`, `checklist`) and `REPEATS`. Judges only grade items that humans rated. Results are cached and the notebook is resumable. Delete `processed/judgments/` (including the `cache_*.jsonl` files) after changing prompts or model settings. The `mock` judge runs the pipeline without an LLM, for tests only.
7. **Notebook 10** (analysis and release). Use the same `HUMAN_FILES` as in Notebook 9. It writes `processed/results/` and the release folder. Recipe text is withheld from the release unless `RELEASE_RECIPE_TEXT = True` (ask before switching it on).

To use the full main round later: collect `ratings_main_A*.csv`, run Notebook 8 with `MODE = "main"`, then set `HUMAN_FILES = ["annotations_main_raw.csv"]` in Notebooks 9 and 10 and rerun them.

## Design decisions worth knowing

- **Rating 6 is not a rating.** In HUMMUS, rows with rating 6 are authorship records; all analysis uses true reviews (1 to 5). Rating 2.5 is treated as "unrated".
- **Blinding.** Annotators and judges never see the technique, the model or the error type; they are in `case_bank/answer_key.csv`.
- **Diagnostics are separate.** Fifteen cases with injected errors (invented preference, wrong nutrition, ignored time limit, ignored dietary constraint) are reported apart from natural outputs.
- **Judges come from other families than the generator.** Generator: Llama. Judges: Gemma, Qwen, Phi. Qwen also audits technique 4, which is reported as a limitation.
- **Repairs are cosmetic.** `repair_text` in Notebook 6 only fixes formatting artefacts (lost fractions, leaked fact identifiers, a broken sentence start). Factual and personalization errors stay in the text.
- **Agreement is computed only where meaningful.** Yes/no kappa uses items where both raters said yes or no; "not applicable" agreement is reported separately; confidence intervals resample items.

## Reproducibility

Seeds: `SEED = 2026` in Notebook 6 (case bank), `42` for the judges (Notebook 9), `SEED = 42` in the EDA. Generation temperature 0.4, judge temperature 0.1, judge context length 4,096 tokens, one repetition. Model versions are those served by Ollama at run time (`llama3.1:8b`, `qwen2.5:14b`, `gemma2:9b`, `phi4`); LLM output can differ slightly across Ollama versions and hardware, so the cached `llm_cache/` and `judgments/` files are the reference for the reported numbers.

## Release package

`processed/release/FoodJudgeBench_v0.1/` contains `case_bank_public.jsonl`, `answer_key.csv`, `rubric.json`, `ratings/human_ratings.csv`, `ratings/judge_ratings.csv`, `results/` and its own README. Do not publish `data/` or `src/.env`.

## Limitations

Only 10 of 510 items have human ratings, so every comparison between judges has wide confidence intervals; three of the six planned explanation techniques (content-based, model-based, contrastive) and human pairwise preferences are not implemented; see the report (Section V) for the full list.

## Citation and licence

HUMMUS paper: https://dl.acm.org/doi/10.1145/3604915.3609491

HUMMUS dataset: https://gitlab.com/felix134/connected-recipe-data-set Code licence: _GPLv3_.
