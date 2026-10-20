# Human evaluation rubric

**What is measured:** how *suitable and trustworthy* an explanation is **for the stated profile**. Not: whether the annotator likes the dish.

Annotators see: (1) the profile, (2) the recipe, (3) checked facts about the recipe, (4) the explanation. They never see which writer produced it. LLM judges get exactly the same four things and the same questions (`processed/case_bank/rubric.json` is the single source for both).

## A. Problems (yes / no; "not applicable" only where stated)
| ID | Question | Not applicable when |
|---|---|---|
| `ignored_constraint` | Does the text ignore or contradict a requirement the profile STATES (diet, allergy, time limit)? Saying a non-vegan recipe "fits your vegan diet" = yes. | profile states none |
| `factual_error` | Anything wrong about the RECIPE (ingredients, time, numbers, comparison with other recipes)? | - |
| `invented_personalization` | Claims about the PERSON that the profile does not say (preference, habit, family)? | - |
| `unsupported_claim` | Other statements of fact that cannot be checked from what is shown (health benefits, taste as fact)? Do not double-count A2/A3 | - |
| `overstated_certainty` | Presents something merely INFERRED as confirmed ("you love...")? | profile fully stated (synthetic) |

## B. Quality (1-5)
| ID | Meaning | Anchors | N/A |
|---|---|---|---|
| `clarity` | easy to read and understand | 1 confusing / 3 understandable but clumsy / 5 clear, well organised | - |
| `profile_fit` | uses the relevant profile information correctly and proportionately | 1 ignores or misuses / 3 partly / 5 uses what matters, correctly | - |
| `goal_alignment` | addresses the health goal correctly, incl. honesty when the recipe does not help it | 1 wrong or misleading / 5 correct and honest | no goal stated |
| `tradeoff` | honestly tells about downsides and mismatches | 1 hides real downsides / 3 mentions some / 5 clear and honest | no downside exists |
| `overall` | overall suitability and trustworthiness for this person and recipe | 1 unsuitable / 5 fully suitable | - |

## C. Pairwise items
"Which explanation is better for this person?" A / B / about equal, plus a main reason (more accurate, clearer, fits the profile better, communicates trade-offs better, other).

## Design decisions
- Violations are binary because annotators agree on yes/no far better than on 1-5, and RQ4 needs precise error labels.
- "Not applicable" exists for goal alignment (no goal), trade-offs (nothing to disclose), constraints and certainty, so annotators are not forced to invent a score.
- "Nudge proportionality" from the earlier plan was dropped: persuasion is out of scope.
- Taste claims such as "you'll love it" are judged only by whether the profile supports them.

## Pilot procedure (Step 7 -> 8)
1. Everyone rates the same 10 pilot items (8 normal + 2 injected-error) with `forms/pilot_*.html`.
2. Run `8_Annotation_Collection.ipynb` with `MODE = "pilot"`; look at questions with exact agreement < 0.7 (yes/no) or within-1 < 0.8 (scales).
3. Meet, discuss the disagreements, reword the unclear question in `RUBRIC` (notebook 6, section 11), update the guide, re-run notebook 6.
4. Discard the pilot ratings. The pilot items are rated again in the main round.
