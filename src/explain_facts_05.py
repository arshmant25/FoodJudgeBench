"""Step 5 helpers: permitted-fact lists, the template technique, and deterministic text checks.

Nothing here calls an LLM. Everything an explanation may claim comes from `build_facts()`; the template technique
(technique 1) fills fixed sentences from the same structured values, and `numeric_mismatches()` / `cited_ids()` give
cheap, objective checks used by the self-check technique and by the generation report.
"""
from __future__ import annotations

import hashlib
import re

from jinja2 import Environment, StrictUndefined

from common import GOAL_RULES, scrub_text

# goal -> (nutrient label, key in nutrition_per_serving, unit)
GOAL_NUTRIENT = {
    "lower sodium": ("sodium", "sodium_mg", "mg"),
    "fewer calories": ("calories", "calories", "kcal"),
    "less sugar": ("sugar", "sugars_g", "g"),
    "less saturated fat": ("saturated fat", "saturated_fat_g", "g"),
    "higher protein": ("protein", "protein_g", "g"),
    "more fiber": ("fiber", "fiber_g", "g"),
}
assert set(GOAL_NUTRIENT) == set(GOAL_RULES), "GOAL_NUTRIENT must cover every goal in common.GOAL_RULES"

_LEVELS = [(0.2, "very low"), (0.4, "low"), (0.6, "moderate"), (0.8, "high"), (1.01, "very high")]


def level(p: float) -> str:
    """Percentile (0-1, share of dataset recipes with LESS of the nutrient) -> plain word."""
    for cut, word in _LEVELS:
        if p < cut:
            return word
    return "very high"


def join_and(items) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ----------------------------------------------------------------------------- ingredient cleaning
# HUMMUS ingredient strings contain placeholders such as "$template2$" where an alternative ingredient was removed,
# and a few lines that are really cook's notes. Neither may reach an annotator or an LLM.
_TEMPLATE = re.compile(r"\$template\d*\$")
_ASIDE = re.compile(r"\([^)]*\b(?:i|we)\s+(?:like|use|used|prefer)\b[^)]*\)", re.I)
_NOTE_LINE = re.compile(r"\bskip it\b|\bi like to\b", re.I)


def clean_ingredient_line(s) -> str | None:
    s = str(s or "").strip()
    if not s or _NOTE_LINE.search(s):
        return None
    m = _TEMPLATE.search(s)
    if m:
        head = s[: m.start()]
        if not re.search(r"[A-Za-z]{3,}", head):          # the line starts with the placeholder: nothing usable
            return None
        cut = head.rfind(" or ")                           # drop the unknown alternative: "butter or 6 tablespoons $t$"
        s = head[:cut] if cut != -1 else head
    s = _ASIDE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;(")
    return s or None


def main_ingredients(recipe: dict, k: int = 12) -> list[str]:
    seen, out = set(), []
    for raw in recipe.get("ingredient_names") or []:
        line = clean_ingredient_line(raw)
        if not line:
            continue
        name = re.sub(r"\([^)]*\)", "", line).split(",")[0].strip().lower()
        if len(name) >= 3 and name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= k:
            break
    return out


def ingredient_lines(recipe: dict, k: int = 25) -> list[str]:
    lines = [clean_ingredient_line(x) for x in (recipe.get("ingredients_display") or recipe.get("ingredient_names") or [])]
    return [scrub_text(x) for x in lines if x][:k]


# ----------------------------------------------------------------------------- citations, text clean-up, numeric check
CITE_RE = re.compile(r"\s*\[\s*F\d+(?:\s*[,;/]\s*F\d+)*\s*\]")
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def strip_citations(t: str) -> str:
    return CITE_RE.sub("", t)


def cited_ids(t: str) -> set[str]:
    out = set()
    for m in CITE_RE.finditer(t):
        out.update(re.findall(r"F\d+", m.group(0)))
    return out


def clean_text(raw: str) -> str:
    """What annotators and judges see: no citations, no markdown, one paragraph."""
    t = strip_citations(str(raw))
    t = re.sub(r"^\s*(?:explanation|recommendation|answer)\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"[*#`>]+", "", t)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    t = re.sub(r"\s*\n+\s*", " ", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def word_count(t: str) -> int:
    return len(re.findall(r"\S+", t))


def numeric_mismatches(text: str, facts: list[dict]) -> list[str]:
    """Numbers in the text that do not appear (within rounding) in any fact."""
    allowed = [float(x) for f in facts for x in _NUM_RE.findall(f["text"])]
    bad = set()
    for m in _NUM_RE.findall(strip_citations(text)):
        v = float(m)
        if not any(abs(v - a) <= max(1.0, 0.03 * abs(a)) for a in allowed):
            bad.add(m)
    return sorted(bad, key=float)


def split_sentences(t: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


# ----------------------------------------------------------------------------- facts
def _tendency_phrase(metric: str, direction: str) -> str:
    if metric == "cooking time":
        return "longer cooking times" if direction == "higher" else "shorter cooking times"
    if metric == "overall healthiness score":
        return f"a {direction} overall healthiness score"
    return f"{direction} {metric}"


def build_facts(ci: dict) -> list[dict]:
    """Numbered permitted facts for one case. `ci` is the case-input dict built in 05 (profile fields + pool facts + recipe)."""
    f, n, src = ci["pool_facts"], ci["pool_facts"]["nutrition_per_serving"], ci["profile_source"]
    prof = ci["profile_view"]
    facts: list[dict] = []

    def add(kind, text, must=False):
        facts.append({"id": f"F{len(facts) + 1}", "kind": kind, "text": text, "must_mention": bool(must)})

    # ---- what we know about the person
    if src == "synthetic":
        add("profile", "Every preference listed for this person was stated explicitly by them.")
    else:
        add("profile", f"This person's profile is inferred from {prof['n_reviews']} past recipe reviews; nothing in it was confirmed by the person.")
    for c in prof["constraints"]:
        if c["status"] == "stated":
            add("profile", f"Stated dietary requirement: {c['constraint']}.")
        elif c["status"] == "observed_pattern":
            add("profile", f"Past behaviour suggests, without confirmation, a pattern of choosing {c['constraint']} recipes.")
    g = prof["goal"]
    if g["value"] in GOAL_RULES and g["status"] == "stated":
        add("profile", f"Stated health goal: {g['value']}.")
    elif g["value"] == "none":
        add("profile", "The person stated that they have no particular health goal.")
    elif g["status"] == "observed_pattern":
        add("profile", f"Past reviews hint, with low confidence and no confirmation: {g['value']}.")
    else:
        add("profile", "No health goal is known for this person.")
    t = prof["time"]
    if t["status"] == "stated" and t["value"]:
        add("profile", f"Stated time limit: about {int(t['value'])} minutes or less.")
    elif t["status"] == "stated":
        add("profile", "The person stated that they have no time limit.")
    elif t.get("typical_minutes"):
        add("profile", f"No time limit is known; recipes this person liked typically take about {t['typical_minutes']:.0f} minutes.")
    else:
        add("profile", "No time limit is known for this person.")
    if prof["likes"]:
        names = ", ".join(prof["likes"])
        add("profile", f"Stated food likes: {names}." if src == "synthetic"
            else f"Past reviews suggest the person tends to like recipes featuring: {names} (inferred pattern).")
    else:
        add("profile", "No food likes are known for this person." if src == "synthetic"
            else "No clear taste pattern could be inferred from past reviews.")
    if prof["dislikes"]:
        names = ", ".join(prof["dislikes"])
        add("profile", f"Stated food dislikes: {names}." if src == "synthetic"
            else f"Past reviews suggest the person tends to rate lower recipes featuring: {names} (inferred pattern).")
    for tr in ci["tendencies_sorted"][:3]:
        add("profile", f"Recipes this person liked tend to have {_tendency_phrase(tr['metric'], tr['direction'])} than typical recipes "
                       f"({tr['confidence']} confidence, inferred).")
    if prof.get("tension"):
        add("profile", f"About this person's priorities: {prof['tension']}.")
    unknown = [re.sub(r"\s*\(.*", "", u).replace("_", " ") for u in prof["unknown"]]
    unknown = sorted(set(unknown) | {"allergies"})
    add("profile", "Not known about this person: " + ", ".join(unknown) + ".")

    # ---- the recipe
    add("recipe", f'The recipe is called "{ci["title"]}".')
    ings = ci["main_ingredients"]
    if ings:
        add("recipe", "Ingredients (as listed): " + ", ".join(ings) + ".")
    d = f["time"]["duration_minutes"]
    add("recipe", f"It takes about {d:.0f} minutes in total.")
    add("recipe", f"Per serving: {n['calories']:.0f} kcal, {n['protein_g']:.0f} g protein, {n['total_fat_g']:.0f} g fat "
                  f"({n['saturated_fat_g']:.0f} g saturated), {n['carbs_g']:.0f} g carbohydrate ({n['sugars_g']:.0f} g sugar, "
                  f"{n['fiber_g']:.0f} g fibre), {n['sodium_mg']:.0f} mg sodium.")
    p = f["nutrition_percentiles"]
    parts = [f"{level(p[k])} in {k}" for k in ("calories", "sodium", "sugar", "saturated fat", "protein", "fiber") if k in p]
    if parts:
        add("recipe", "Compared with other recipes in the dataset, this recipe is " + ", ".join(parts) + ".")
    hl = f["health_levels"]
    add("recipe", f"Nutrition-score levels (higher is healthier): {hl['nutri_level_0_to_4']} out of 4 (Nutri-Score style) "
                  f"and {hl['fsa_level_0_to_8']} out of 8 (FSA style).")

    # ---- how the recipe matches the person
    for chk in f["constraint_checks"]:
        c = chk["constraint"]
        if chk["rule_passes"]:
            how = {"verified": "the ingredient list and the recipe tags both indicate this",
                   "manual": "this was checked by hand"}.get(chk["label_status"],
                                                             "no ingredient in the list conflicts with it, but the recipe carries no explicit tag for it")
            add("match", f"Dietary check ({c}): passes; {how}.")
        else:
            add("match", f"Dietary check ({c}): FAILS; the ingredient list conflicts with it.", must=True)
    tm = f["time"]
    if tm["limit_minutes"]:
        if tm["within_limit"]:
            add("match", f"Time check: {d:.0f} minutes is within the {int(tm['limit_minutes'])}-minute limit.")
        else:
            add("match", f"Time check: {d:.0f} minutes is longer than the {int(tm['limit_minutes'])}-minute limit.", must=True)
    gl = f.get("goal")
    if gl:
        label, nkey, unit = GOAL_NUTRIENT[gl["goal"]]
        base = f"Goal check ({gl['goal']}): {{}}; this recipe's {label} is {level(gl['recipe_percentile'])} compared with other recipes ({n[nkey]:.0f} {unit} per serving)."
        add("match", base.format("met") if gl["meets_goal"] else base.format("NOT met"), must=not gl["meets_goal"])
    if f["matches_liked"]:
        add("match", "The recipe features liked items: " + ", ".join(f["matches_liked"]) + ".")
    elif prof["likes"]:
        add("match", "The recipe features none of the person's listed likes.", must=True)
    if f["contains_disliked"]:
        add("match", "The recipe contains disliked items: " + ", ".join(f["contains_disliked"]) + ".", must=True)
    for ta in ci["tendencies_sorted"][:2]:
        if ta.get("aligned") is not None:
            add("match", f"Compared with the person's usual pattern ({_tendency_phrase(ta['metric'], ta['direction'])}), this recipe is "
                         f"{'in line' if ta['aligned'] else 'not in line'}.")
    return facts


def facts_to_prompt(facts: list[dict]) -> str:
    return "\n".join(f"{x['id']}: {x['text']}" + (" [MUST MENTION]" if x["must_mention"] else "") for x in facts)


# ----------------------------------------------------------------------------- technique 1: template (jinja2, fixed sentences)
_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
_SENT = {
    "open": ["{{ title }} is recommended for you.", "We suggest {{ title }} for you.", "{{ title }} is one option we can offer you."],
    "taste_match_syn": ["It features {{ items }}, which you said you like.", "It includes {{ items }}, which are on your list of favourites."],
    "taste_match_hist": ["It features {{ items }}, and your past reviews suggest you tend to like recipes with these.",
                         "Your past reviews suggest you tend to enjoy recipes featuring {{ items }}, and this one does."],
    "taste_none_syn": ["It does not feature any of the foods you listed as favourites."],
    "taste_none_hist": ["It does not clearly match the ingredients and themes you usually favour."],
    "dislike_syn": ["Note that it contains {{ items }}, which you said you dislike."],
    "dislike_hist": ["Note that it contains {{ items }}, which your past reviews suggest you tend to rate lower."],
    "cons_verified": ["It meets your {{ items }} {{ 'requirements' if plural else 'requirement' }}: the ingredient list and the recipe tags agree."],
    "cons_rule_only": ["No ingredient in the list conflicts with your {{ items }} {{ 'requirements' if plural else 'requirement' }}."],
    "cons_manual": ["It meets your {{ items }} {{ 'requirements' if plural else 'requirement' }} (checked by hand)."],
    "cons_pattern": ["You often choose {{ items }} recipes, and this one fits that pattern."],
    "cons_fail": ["Warning: it does not meet your {{ items }} requirement."],
    "time_ok": ["It takes about {{ d }} minutes, within your {{ limit }}-minute limit."],
    "time_over": ["Note that it takes about {{ d }} minutes, longer than your {{ limit }}-minute limit."],
    "time_free": ["It takes about {{ d }} minutes."],
    "goal_met": ["It supports your goal of {{ goal }}: its {{ nutrient }} is {{ level }} compared with other recipes ({{ val }} {{ unit }} per serving)."],
    "goal_not": ["Trade-off: it does not fit your goal of {{ goal }}, since its {{ nutrient }} is {{ level }} compared with other recipes ({{ val }} {{ unit }} per serving)."],
    "tend_in": ["It matches the usual pattern of your favourites, which tend to have {{ phrase }}."],
    "tend_out": ["It differs from the usual pattern of your favourites, which tend to have {{ phrase }}."],
    "snapshot": ["Per serving it has about {{ cal }} kcal, {{ protein }} g of protein and {{ sodium }} mg of sodium."],
}
_SENT = {k: [_ENV.from_string(v) for v in vs] for k, vs in _SENT.items()}


def _r(name: str, key: str, **kw) -> str:
    vs = _SENT[name]
    i = int(hashlib.md5(f"{key}|{name}".encode()).hexdigest(), 16) % len(vs)
    return vs[i].render(**kw)


def render_template(ci: dict, key: str) -> str:
    f, n, hist = ci["pool_facts"], ci["pool_facts"]["nutrition_per_serving"], ci["profile_source"] == "historical"
    out = [_r("open", key, title=ci["title"])]
    if f["matches_liked"]:
        out.append(_r("taste_match_hist" if hist else "taste_match_syn", key, items=join_and(f["matches_liked"][:3])))
    elif ci["profile_view"]["likes"]:
        out.append(_r("taste_none_hist" if hist else "taste_none_syn", key))
    if f["contains_disliked"]:
        out.append(_r("dislike_hist" if hist else "dislike_syn", key, items=join_and(f["contains_disliked"][:2])))
    groups = {"verified": [], "rule_only": [], "manual": [], "pattern": []}
    for c in f["constraint_checks"]:
        name = c["constraint"]
        if not c["rule_passes"]:
            out.append(_r("cons_fail", key, items=name))
        elif ci["profile_view"]["constraint_status"].get(name) == "observed_pattern":
            groups["pattern"].append(name)
        else:
            groups[c["label_status"] if c["label_status"] in groups else "rule_only"].append(name)
    for g, names in groups.items():
        if names:
            out.append(_r(f"cons_{g}", key, items=join_and(names), plural=len(names) > 1))
    tm = f["time"]
    d = round(tm["duration_minutes"])
    if tm["limit_minutes"]:
        out.append(_r("time_ok" if tm["within_limit"] else "time_over", key, d=d, limit=int(tm["limit_minutes"])))
    else:
        out.append(_r("time_free", key, d=d))
    gl = f.get("goal")
    if gl:
        label, nkey, unit = GOAL_NUTRIENT[gl["goal"]]
        out.append(_r("goal_met" if gl["meets_goal"] else "goal_not", key, goal=gl["goal"], nutrient=label,
                      level=level(gl["recipe_percentile"]), val=round(n[nkey]), unit=unit))
    elif ci["tendencies_sorted"] and ci["tendencies_sorted"][0].get("aligned") is not None:
        t0 = ci["tendencies_sorted"][0]
        out.append(_r("tend_in" if t0["aligned"] else "tend_out", key, phrase=_tendency_phrase(t0["metric"], t0["direction"])))
    out.append(_r("snapshot", key, cal=round(n["calories"]), protein=round(n["protein_g"]), sodium=round(n["sodium_mg"])))
    return " ".join(out)
