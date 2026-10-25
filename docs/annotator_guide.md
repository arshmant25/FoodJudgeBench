# Annotator guide - rating food recommendation explanations

You do not need to know anything about the project beforehand. This guide tells you what to do, step by step.

## 1. What this is about
Apps that recommend recipes often add a short sentence or paragraph saying *why* a recipe was chosen for you. We are testing whether computers (language models) can grade such explanations as reliably as humans. For that we need your careful ratings. You will read many short explanations and answer the same questions each time.

Your job is **not** to say whether you like the recipe. Your job is to judge whether the explanation is **correct, honest and suitable for the person it was written for.**

## 2. What you get
A single HTML file (for example `main_A1.html`). Open it in Chrome, Firefox, Edge or Safari - no installation, no internet needed. It shows one **case** at a time.

Each case has four parts:
1. **The person (profile).** A short description. It says whether things were **stated** (the person told us) or **inferred** (guessed from their past reviews). Inferred information is a guess, not a fact.
2. **The recipe.** Title and ingredient list.
3. **Checked facts about the recipe.** Time, nutrition per serving, how it compares with other recipes, diet checks. **Trust these facts.**
4. **The explanation.** The text you rate. Several writers (people-written templates and language models) produced the texts; you are not told which.

You may also get **comparison items**: two explanations (A and B) for the same person and recipe; you pick the better one.

## 3. How to rate a case
Read parts 1-3 first, then the explanation, then answer the questions.

**A. Problems** - answer Yes or No (some also allow "Not applicable"):
1. *Ignores a stated requirement?* Yes if the text ignores or contradicts something the profile **states** as a requirement: a diet, an allergy or a time limit. Example: the profile says "vegan" and the text says the recipe fits their vegan diet, but the ingredients contain cheese -> **Yes**. Not applicable if the profile states no requirement.
2. *Wrong fact about the recipe?* Yes if the text says something about the recipe that contradicts the ingredients or the checked facts. Example: text says "very low in sodium" but the facts say "high in sodium" -> **Yes**.
3. *Invents something about the person?* Yes if the text claims something about the person that the profile does not say. Example: "Since you cook for your large family..." when the profile mentions no family -> **Yes**.
4. *Other unsupported claim?* Yes if the text states as fact something you cannot check from the four parts, for example "this will lower your cholesterol". Do not count again a problem you already marked in 2 or 3.
5. *Treats a guess as certain?* Only for profiles where information was inferred. Yes if the text says "you love X" / "you always..." when the profile only says the person *tends to* like X. Not applicable if the profile is fully stated.

**B. Quality** - scale 1 (very poor) to 5 (excellent):
- *Clarity*: easy to read and understand?
- *Fit with the profile*: uses what matters for this person, correctly, without over-reading weak information?
- *Goal alignment*: if the profile has a health goal (for example "less sodium"), does the text deal with it correctly and say honestly when the recipe does not help? **Not applicable** if there is no goal.
- *Trade-offs*: does it tell the person honestly about downsides (high sodium, takes longer than usual, goal not met)? 5 = clear and honest, 1 = hides real downsides. **Not applicable** if the recipe has no downside for this person.
- *Overall suitability*: your overall judgement.

Use the whole 1-5 range. 3 means "acceptable, with noticeable weaknesses".


## 3b. When to answer "Not applicable" (check the profile, not your opinion)
Whether a question applies is a fact you can read in the profile:
| Question | "Not applicable" ONLY when... |
|---|---|
| Ignores a stated requirement | the profile states **no** diet, no allergy **and** no time limit. If any of those is stated, answer Yes or No. |
| Goal alignment | the profile states **no** health goal ("Health goal: unknown") |
| Treats a guess as certain | the profile is fully **stated** (a designed profile that says "stated"). Profiles built from past reviews are always *inferred*, so answer Yes or No there. |
| Trade-offs | the recipe has genuinely no downside for this person (rare) |

## 4. Rules that keep the study valid
- **Work alone.** Do not discuss cases with the other annotators until the project lead says the round is finished. We measure how much independent people agree.
- **Do not try to guess who or what wrote a text**, and do not let a guess influence a rating.
- **Judge only what is on the screen.** Do not look up the recipe or search the web; do not use outside knowledge about the person.
- **Taste is not the criterion.** A dessert for a person who likes desserts is fine even if you dislike desserts.
- **Short, friendly texts are not better by themselves.** Reward correctness and honesty over enthusiasm.
- Some cases contain a deliberate mistake; that is expected. Just rate what you see. Do not look for "trick" cases.
- If a case is broken or confusing (missing text, odd characters), rate what you can and write a short **comment**.
- If you are unsure between two answers, pick the one you would defend if asked, and add a comment if you wish.

## 5. Step by step
**Pilot round (first, about 30 minutes)**
1. Open `pilot_<your code>.html` (10 cases).
2. Rate them all. Use the comment box whenever a question is unclear.
3. Click **Download my ratings (CSV)** and send the file to the project lead.
4. Join the short meeting where we go through disagreements and improve the wording. Your pilot answers are only for improving the questions and will be discarded.

**Main round (about 20 hours in total; spread over several days, 1-2 hour sessions are best)**
1. Open `main_<your code>.html`. It starts at the first unfinished case.
2. For each case: read, answer every required question, click **Save & next**. The form tells you what is still missing.
3. **Your progress is saved automatically in your browser.** Use the same browser and computer and do not clear site data. Use **Download my ratings (CSV)** at the end of every session as a backup, and send the file when you finish (the last download contains everything).
4. You can click **Back** to change an earlier answer; click Save & next again afterwards.
5. When all cases are done, download the CSV and send it to the project lead. Do not edit the CSV by hand.

## 6. Comparison items
You see two explanations, A and B. Choose which is better **for this person**, or "about equal", and pick the main reason. Judge accuracy, honesty about downsides and fit with the profile - not which dish you prefer.

## 7. Time and quality
Aim for about 2-3 minutes per case. Take breaks; tired ratings are noisy. The form records how long you spend per case; this is only used to check the workload estimate, not to judge you.

## 8. Privacy
The materials come from a public recipe site and contain no personal data. Please do not share the forms or the cases outside the project.

## 9. Quick reference
| Question | Yes means... |
|---|---|
| Ignores a stated requirement | text breaks or contradicts a stated diet / allergy / time limit |
| Wrong fact | contradicts the recipe or checked facts |
| Invents about the person | claims something about the person not in the profile |
| Other unsupported claim | stated as fact but cannot be checked |
| Treats a guess as certain | inferred information presented as confirmed |

Questions? Contact the project lead (Arshman). Thank you!
