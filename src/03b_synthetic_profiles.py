"""Step 3b - SYNTHETIC profiles (30, explicitly specified, systematically varied) on the same schema as historical ones.

30 hand-designed archetypes: 14 standard, 8 deliberate CONFLICTS between priorities, 6 MISSING-information profiles and
2 explicit-NONE profiles ('none' is a stated fact, 'unknown' is not). Optional --n-extra adds random draws from the factor grid.
Each profile is checked against the catalog for feasibility (enough recipes that satisfy its hard constraints).

Inputs : processed/catalog
Outputs: processed/profiles_synthetic.jsonl, profiles_synthetic_summary.csv
Run    : python src/03b_synthetic_profiles.py [--n-extra 0]
"""
import argparse

import numpy as np
import pandas as pd

from config import (SEED, GOAL_RULES, read_table, parse_list_cols, constraint_mask, verified_mask, goal_mask, term_mask,
                    render_profile_text, write_jsonl, PROC_DIR)

MIN_FEASIBLE = 30

# term label -> regex over normalized ingredients / informative tags (stored in the profile so checks are reproducible)
TERMS = {
    "pasta": r"pasta|spaghetti|macaroni|penne|noodle", "chicken": r"chicken", "beans": r"\bbeans?\b|lentil|chickpea",
    "cheese": r"cheese|parmesan|mozzarella|cheddar", "rice": r"\brice\b", "spicy": r"chili|chile|jalapeno|cayenne|curry|sriracha",
    "salad": r"salad|lettuce|arugula", "soup": r"\bsoup|stew|broth", "chocolate": r"chocolate|cocoa", "potato": r"potato",
    "tofu": r"tofu|tempeh", "salmon": r"salmon|\bfish\b", "creamy": r"cream|cheese|butter", "mushroom": r"mushroom",
    "cilantro": r"cilantro|coriander", "olives": r"olive(?!\s*oil)", "onion": r"onion", "seafood": r"shrimp|crab|salmon|\bfish\b|tuna",
    "soy sauce": r"soy sauce", "cured meat": r"bacon|ham\b|sausage|salami", "baking": r"flour|baking", "tomato": r"tomato",
}


def C(*names):
    return [{"constraint": n, "status": "stated", "confidence": "stated", "evidence": "specified by design"} for n in names]


ARCHETYPES = [
    # (name, group, constraints, goal, time_limit, likes, dislikes, tension)
    # constraints: None = unknown, [] = explicitly none | goal: None = unknown, "none" = explicitly none
    # time_limit : None = unknown, "none" = explicitly no limit, int = minutes
    # ---- standard (14): clear, non-conflicting specs
    ("busy_vegetarian", "standard", ["vegetarian"], None, 30, ["pasta", "cheese"], ["mushroom"], None),
    ("sodium_watcher", "standard", [], "lower sodium", 45, ["chicken", "salad"], [], None),
    ("vegan_protein", "standard", ["vegan"], "higher protein", 45, ["beans", "tofu"], [], None),
    ("nut_free_family", "standard", ["nut-free"], None, 30, ["pasta", "chicken"], ["seafood"], None),
    ("calorie_conscious_quick", "standard", [], "fewer calories", 30, ["salad", "soup"], ["cheese"], None),
    ("gluten_free_baker", "standard", ["gluten-free"], None, 60, ["chocolate", "potato"], [], None),
    ("dairy_free_low_sugar", "standard", ["dairy-free"], "less sugar", 45, ["rice", "chicken"], [], None),
    ("vegetarian_fiber", "standard", ["vegetarian"], "more fiber", 45, ["beans", "soup"], ["mushroom"], None),
    ("low_satfat_chicken", "standard", [], "less saturated fat", 60, ["chicken", "rice"], ["cured meat"], None),
    ("vegan_dessert_lover", "standard", ["vegan"], "none", 60, ["chocolate", "baking"], [], None),
    ("gluten_free_protein", "standard", ["gluten-free"], "higher protein", 45, ["chicken", "salmon"], [], None),
    ("nut_free_quick_lunch", "standard", ["nut-free"], "none", 15, ["salad", "tomato"], [], None),
    ("vegetarian_nut_free", "standard", ["vegetarian", "nut-free"], "none", 45, ["pasta", "tomato"], [], None),
    ("dairy_free_vegetarian_fiber", "standard", ["dairy-free", "vegetarian"], "more fiber", 60, ["rice", "beans"], [], None),
    # ---- conflict (8): stated priorities pull against each other
    ("comfort_vs_light", "conflict", [], "fewer calories", 45, ["creamy", "pasta", "cheese"], [], "stated taste (rich, creamy food) pulls against stated goal (fewer calories)"),
    ("quick_vegan_protein", "conflict", ["vegan"], "higher protein", 15, ["beans"], [], "tight time limit + vegan + protein goal leaves few options"),
    ("low_sodium_loves_salty", "conflict", [], "lower sodium", 45, ["soy sauce", "cured meat"], [], "likes salty foods but wants lower sodium"),
    ("sweet_tooth_less_sugar", "conflict", [], "less sugar", 60, ["chocolate", "baking"], [], "loves sweet baking but wants less sugar"),
    ("low_satfat_loves_cheese", "conflict", [], "less saturated fat", 45, ["cheese", "creamy"], [], "likes cheese/cream but wants less saturated fat"),
    ("fewer_calories_loves_carbs", "conflict", [], "fewer calories", 60, ["pasta", "potato"], [], "likes starchy comfort food but wants fewer calories"),
    ("quick_gluten_free_protein", "conflict", ["gluten-free"], "higher protein", 15, ["chicken"], [], "very short time + gluten-free + protein goal"),
    ("quick_vegetarian_protein", "conflict", ["vegetarian"], "higher protein", 15, ["beans", "cheese"], [], "short time and a protein goal while meat-free"),
    # ---- missing information (6): unknown fields are explicit
    ("minimal_info", "missing_info", None, None, 30, [], [], None),
    ("taste_only_spicy", "missing_info", None, None, None, ["spicy", "rice"], [], None),
    ("goal_only_sodium", "missing_info", None, "lower sodium", None, [], [], None),
    ("constraint_only_vegan", "missing_info", ["vegan"], None, None, [], [], None),
    ("time_and_soup", "missing_info", None, None, 45, ["soup"], [], None),
    ("dislikes_only", "missing_info", None, None, None, [], ["mushroom", "onion"], None),
    # ---- explicit none (2): 'none' is a stated fact, not 'unknown'
    ("no_restrictions_baseline", "explicit_none", [], "none", "none", ["pasta", "tomato"], [], None),
    ("easygoing_no_constraints", "explicit_none", [], "none", "none", ["rice"], ["onion"], None),
]
assert len(ARCHETYPES) == 30


def make_profile(pid, name, label, cons, goal, time_limit, likes, dislikes, tension, known_constraints=True):
    unknown = []
    if cons is None:
        dc, cons = [], []
        unknown += ["dietary_constraints", "allergies"]
    else:
        dc = C(*cons)
    if goal is None:
        hg = {"value": "unknown", "status": "unknown", "confidence": None, "evidence": "not provided"}
        unknown.append("health_goal")
    elif goal == "none":
        hg = {"value": "none", "status": "stated", "confidence": "stated", "evidence": "specified by design"}
    else:
        col, op, thr = GOAL_RULES[goal]
        hg = {"value": goal, "status": "stated", "confidence": "stated", "evidence": "specified by design",
              "operational_definition": f"{col} {op} {thr}"}
    if time_limit == "none":
        tb = {"value": None, "status": "stated", "typical_minutes": None, "evidence": "no time limit, specified by design"}
    elif time_limit:
        tb = {"value": time_limit, "status": "stated", "typical_minutes": None, "evidence": "specified by design"}
    else:
        tb = {"value": None, "status": "unknown", "typical_minutes": None, "evidence": "not provided"}
        unknown.append("time_budget")
    if not likes and not dislikes:
        unknown.append("taste_preferences")
    mk = lambda items: [{"item": i, "kind": "term", "match_regex": TERMS[i], "status": "stated"} for i in items]
    p = {"profile_id": pid, "source": "synthetic", "label": f"{label}/{name}", "dietary_constraints": dc, "health_goal": hg,
         "time_budget_minutes": tb, "taste": {"likes": mk(likes), "dislikes": mk(dislikes)}, "nutrition_tendencies": [],
         "household_context": [], "unknown_fields": unknown, "tension": tension, "history": None,
         "evidence": [{"type": "specified_by_design", "claim": "all attributes explicitly set", "confidence": "stated"}]}
    return p


def feasibility(cat, p):
    cons = [c["constraint"] for c in p["dietary_constraints"]]
    t = p["time_budget_minutes"]["value"]
    hard = constraint_mask(cat, cons, t)
    out = {"n_hard_feasible": int(hard.sum()), "n_verified": int((verified_mask(cat, cons) & hard).sum())}
    like = pd.Series(False, index=cat.index)
    for x in p["taste"]["likes"]:
        like |= term_mask(cat, x["match_regex"])
    dis = pd.Series(False, index=cat.index)
    for x in p["taste"]["dislikes"]:
        dis |= term_mask(cat, x["match_regex"])
    out["n_hard_and_liked"] = int((hard & like & ~dis).sum()) if p["taste"]["likes"] else None
    g = p["health_goal"]["value"]
    out["n_hard_and_goal"] = int((hard & goal_mask(cat, g)).sum()) if g in GOAL_RULES else None
    out["n_all_three"] = int((hard & like & ~dis & goal_mask(cat, g)).sum()) if (g in GOAL_RULES and p["taste"]["likes"]) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-extra", type=int, default=0, help="extra random profiles from the factor grid")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    cat = read_table("catalog")
    parse_list_cols(cat)
    cat["tags_str"] = cat["tags_str"].fillna("")
    cat["ing_text"] = cat["ing_text"].fillna("")

    profiles = [make_profile(f"S{i:03d}", *a) for i, a in enumerate(ARCHETYPES, 1)]
    cons_grid = [[], ["vegetarian"], ["vegan"], ["dairy-free"], ["gluten-free"], ["nut-free"], ["vegetarian", "nut-free"]]
    goals = list(GOAL_RULES) + ["none", None]
    times = [None, 15, 30, 45, 60]
    like_pool = ["pasta", "chicken", "beans", "cheese", "rice", "spicy", "salad", "soup", "chocolate", "potato", "tofu", "salmon"]
    dis_pool = ["mushroom", "cilantro", "olives", "onion", "tomato", "seafood"]
    n_done, tries = 0, 0
    while n_done < args.n_extra and tries < 500:
        tries += 1
        cons = cons_grid[rng.integers(len(cons_grid))]
        goal = goals[rng.integers(len(goals))]
        t = times[rng.integers(len(times))]
        lk = list(rng.choice(like_pool, rng.integers(0, 3), replace=False))
        dk = list(rng.choice(dis_pool, rng.integers(0, 2), replace=False))
        p = make_profile(f"S{len(profiles) + 1:03d}", f"grid{n_done + 1}", "random_grid", cons, goal, t, lk, dk, None)
        if feasibility(cat, p)["n_hard_feasible"] >= MIN_FEASIBLE:
            profiles.append(p)
            n_done += 1

    rows = []
    for p in profiles:
        f = feasibility(cat, p)
        p["feasibility"] = f
        p["profile_text"] = render_profile_text(p)
        rows.append({"profile_id": p["profile_id"], "label": p["label"], **f})
        if f["n_hard_feasible"] < MIN_FEASIBLE:
            print(f"WARNING {p['profile_id']} {p['label']}: only {f['n_hard_feasible']} recipes satisfy the hard constraints - loosen it or enlarge the catalog")
    for p in profiles:
        f = p["feasibility"]
        for k in ("n_hard_and_liked", "n_hard_and_goal", "n_all_three"):
            if f[k] is not None and f[k] < 10:
                print(f"WARNING {p['profile_id']} {p['label']}: {k}={f[k]} (<10) - recommender will struggle to find good matches; revisit the likes/goal terms")
    write_jsonl(profiles, "profiles_synthetic")
    summ = pd.DataFrame(rows)
    summ.to_csv(PROC_DIR / "profiles_synthetic_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\nSample profile text:\n", profiles[14]["profile_text"])


if __name__ == "__main__":
    main()
