# FoodJudgeBench_v0.1

**FoodJudgeBench** tests whether LLM judges can grade food-recommendation *explanations* as reliably as human annotators.

## Contents
| File | Description |
|---|---|
| `case_bank_public.jsonl` | the items shown to raters: profile card, recipe (withheld), checked facts, explanation (technique identity hidden) |
| `answer_key.csv` | per item: explanation technique, profile source (historical/synthetic), injected error type (diagnostics), pair order |
| `ratings/human_ratings.csv` | human ratings (3 annotator(s), 10 items), rubric answers per item |
| `ratings/judge_ratings.csv` | LLM-judge ratings (same columns; `annotator_id` = `judge|setup`) |
| `rubric.json` | the questions and answer options used by humans and judges |
| `results/` | agreement tables (RQ1-RQ4), figures, auto-generated insights |

## How it was built
Profiles: historical (inferred from Food.com reviews) and synthetic (stated). Recipes: content-based pool. Explanations: four techniques (template, direct LLM, evidence-grounded LLM, grounded LLM with self-check). Diagnostic items contain one injected error of a known type. Judges see exactly what human annotators saw.

## Limitations
- Human data in this version: 3 annotator(s), 10 items.
- Explanations come from one writer model; technique 3 and 4 texts are identical in a large share of cases.
- Style differences between techniques (e.g. first-person phrasing) can reveal the technique.
- Limited recipe diversity; profiles are few.

## Terms
Recipe and review data derive from Food.com (HUMMUS dataset) and are limited to educational / scientific use. Recipe text is withheld in this release. No personal data about annotators is included beyond anonymous IDs.
