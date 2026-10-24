# FoodJudgeBench - automatic insights (draft)

- Data: 10 human-rated items (10 case/diagnostic, 0 pairwise) by 3 annotator(s); 9 judge|setup combination(s).

- **Limitation:** small sample - all numbers are indicative.

- Best judge|setup on yes/no problem questions: **qwen14b|checklist** (mean kappa 0.26); weakest: mistral7b|checklist (0.07).
- Best on 1-5 scales (within 1 point): **gemma9b|rubric_facts** (0.88).
- Human-vs-human baseline: kappa 0.29 (yes/no), within-1 0.88 (scales). Judges at or above this level are as consistent with a human as humans are with each other.
- Hardest yes/no question for judges (lowest mean kappa): **factual_error** (0.00); easiest: **invented_personalization** (0.35).
- Error 'invented_preference': best detector gemma9b|checklist (100% of ratings; 2 items).

Always check these against the tables in `processed/results/` before using them.