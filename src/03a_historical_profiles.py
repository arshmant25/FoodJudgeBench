"""Step 3a - HISTORICAL profiles (30) from HUMMUS review history.

Principle: a profile claim is only made when the evidence clearly differs from what a typical reviewed recipe looks
like (z-test vs review-weighted baseline). Everything inferred is tagged with status + confidence; health goals stay
'unknown' unless clearly supported. Nothing is treated as confirmed.

Inputs : processed/user_profile_candidates, reviews_clean, recipe_lookup, baselines.json
Outputs: processed/profiles_historical.jsonl, profiles_historical_summary.csv
Run    : python src/03a_historical_profiles.py
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import (PROC_DIR, SEED, LIKED_RATING, DISLIKED_MAX_RATING, read_table, split_items, snippet,
                    render_profile_text, write_jsonl)

# ------------------------------------------------------------------ config
N_HISTORICAL = 30                      # total profiles, spread evenly over 2 richness x 2 signal strata
SPARSE_RANGE, RICH_RANGE = (20, 50), (100, 300)
MIN_RATING_STD, MIN_LIKED, MIN_SPAN_YEARS, MIN_TEXT = 0.5, 10, 1.0, 15
MIN_N_TENDENCY, Z_TENDENCY, Z_STRONG = 10, 3.0, 5.0
MIN_ITEM_COUNT, Z_ITEM, MIN_LIFT, TOP_TAGS, TOP_INGS = 4, 3.0, 1.8, 6, 8
MIN_DISLIKED = 5
PATTERN_MIN_N, PATTERN_MIN_SHARE, PATTERN_Z = 15, 0.95, 3.0

PCT_METRICS = {  # label -> (lookup column, human meaning of 'higher')
    "calories": "calories_pct", "sodium": "sodium_pct", "sugar": "sugars_pct", "saturated fat": "satfat_pct",
    "protein": "protein_pct", "fiber": "fiber_pct", "cooking time": "duration_pct", "overall healthiness score": "who_pct",
}
RULES = {"vegetarian": "rule_vegetarian", "vegan": "rule_vegan", "dairy-free": "rule_dairy_free",
         "gluten-free": "rule_gluten_free", "nut-free": "rule_nut_free"}
IMPLIES = {"vegan": {"vegetarian", "dairy-free"}}

MENTION_LABELS = {
    "vegetarian": r"\bvegetarian\b", "vegan": r"\bvegan\b", "gluten-free": r"\b(?:gluten|celiac|coeliac)\b",
    "dairy-free": r"\b(?:lactose|dairy[- ]free)\b", "nut-free": r"\bnut[- ]free\b|\bnut allerg\w*",
    "low-carb/keto": r"\b(?:keto|low[- ]carb)\b", "kosher/halal": r"\b(?:kosher|halal)\b", "allergy (unspecified)": r"\ballerg\w*",
}
HEALTH_RE = re.compile(r"\b(?:diabet\w*|blood sugar|cholesterol|low[- ]fat|low[- ]sodium|low[- ]calorie|weight loss|healthy|healthier)\b", re.I)
HOUSEHOLD_RE = re.compile(r"\b(?:my kids?|my children|my son|my daughter|my husband|my wife|my family|toddler\w*|picky|my boyfriend|my girlfriend)\b", re.I)


def allocate(total, sizes):
    """Round-robin allocation so strata stay balanced (e.g. 30 over 4 strata -> 8,8,7,7), capped by availability."""
    alloc = {k: 0 for k in sizes}
    while sum(alloc.values()) < total:
        moved = False
        for k in alloc:
            if sum(alloc.values()) >= total:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                moved = True
        if not moved:
            break
    return alloc


def select_users(us: pd.DataFrame) -> pd.DataFrame:
    e = us[(us["rating_std"] >= MIN_RATING_STD) & (us["n_liked"] >= MIN_LIKED) & (us["span_years"] >= MIN_SPAN_YEARS)
           & (us["n_text"] >= MIN_TEXT)].copy()
    e["richness"] = np.select([e["n_reviews"].between(*SPARSE_RANGE), e["n_reviews"].between(*RICH_RANGE)], ["sparse", "rich"], "other")
    e["signal"] = np.select([(e["n_restriction"] >= 2) | (e["n_health"] >= 3), (e["n_restriction"] == 0) & (e["n_health"] <= 1)],
                            ["with_signal", "no_signal"], "other")
    e = e[(e["richness"] != "other") & (e["signal"] != "other")]
    print("eligible users per stratum:\n", e.groupby(["richness", "signal"]).size().to_string())
    sizes = e.groupby(["richness", "signal"]).size().to_dict()
    alloc = allocate(N_HISTORICAL, sizes)
    print("allocation per stratum:", alloc)
    sel = [g.sample(alloc[k], random_state=SEED) for k, g in e.groupby(["richness", "signal"]) if alloc[k] > 0]
    if sum(alloc.values()) < N_HISTORICAL:
        print(f"WARNING: only {sum(alloc.values())} eligible users available - relax the thresholds at the top of the file")
    return pd.concat(sel).reset_index(drop=True)


def item_lifts(lists, base_rates, min_count, top_k):
    lists = [l for l in lists if l]
    n = len(lists)
    if n < MIN_N_TENDENCY:
        return [], n
    cnt = Counter(i for l in lists for i in set(l))
    default = min(base_rates.values()) if base_rates else 0.001
    out = []
    for item, c in cnt.items():
        if c < min_count:
            continue
        r, b = c / n, min(max(base_rates.get(item, default), 1e-4), 0.999)
        z = (r - b) / np.sqrt(b * (1 - b) / n)
        if z >= Z_ITEM and r / b >= MIN_LIFT:
            out.append({"item": item, "count": int(c), "n_recipes": n, "rate": round(r, 3), "baseline_rate": round(b, 4),
                        "lift": round(r / b, 2), "z": round(float(z), 1), "status": "observed_pattern"})
    out.sort(key=lambda d: -d["z"])
    return out[:top_k], n


def build_profile(i, uid, label, rows, lookup, base, ustats):
    r = rows.merge(lookup, on="new_recipe_id", how="left")
    liked = r[r["liked"]]
    disliked = r[(r["rating"] <= DISLIKED_MAX_RATING) & (r["rating_type"] == "stars")]
    coverage = float(r["duration"].notna().mean())
    ev, unknown = [], ["confirmed_dietary_restrictions", "allergies", "budget"]

    # --- taste
    likes_tags, n_t = item_lifts([split_items(s) for s in liked["tags_str"]], base["tag_rates"], MIN_ITEM_COUNT, TOP_TAGS)
    likes_ing, n_i = item_lifts([split_items(s) for s in liked["ing_str"]], base["ingredient_rates"], MIN_ITEM_COUNT, TOP_INGS)
    for x in likes_tags:
        x["kind"] = "tag"
    for x in likes_ing:
        x["kind"] = "ingredient"
    likes = sorted(likes_tags + likes_ing, key=lambda d: -d["z"])
    dislikes = []
    if len(disliked) >= MIN_DISLIKED:
        d_tags, _ = item_lifts([split_items(s) for s in disliked["tags_str"]], base["tag_rates"], 3, 3)
        d_ing, _ = item_lifts([split_items(s) for s in disliked["ing_str"]], base["ingredient_rates"], 3, 4)
        for x in d_tags:
            x["kind"] = "tag"
        for x in d_ing:
            x["kind"] = "ingredient"
        dislikes = d_tags + d_ing
    else:
        unknown.append(f"dislikes (only {len(disliked)} low-rated recipes)")

    # --- nutrition tendencies (liked recipes vs review-weighted baseline)
    tend = []
    for name, col in PCT_METRICS.items():
        v = liked[col].dropna()
        if len(v) < MIN_N_TENDENCY:
            continue
        b = base["pct_metrics"][col]
        z = (v.mean() - b["mean"]) / (b["sd"] / np.sqrt(len(v)))
        if abs(z) >= Z_TENDENCY:
            tend.append({"metric": name, "direction": "higher" if z > 0 else "lower", "user_mean_percentile": round(float(v.mean()), 3),
                         "baseline_mean_percentile": round(b["mean"], 3), "z": round(float(z), 1), "n": int(len(v)),
                         "confidence": "high" if abs(z) >= Z_STRONG else "medium", "status": "observed_pattern"})

    # --- dietary patterns from behaviour
    constraints = {}
    for cname, col in RULES.items():
        v = liked[col].dropna()
        if len(v) < PATTERN_MIN_N:
            continue
        share, b = float(v.mean()), base["rule_rates"][col]
        if share >= PATTERN_MIN_SHARE and b < 0.9:
            z = (share - b) / np.sqrt(b * (1 - b) / len(v))
            if z >= PATTERN_Z:
                constraints[cname] = {"constraint": cname, "status": "observed_pattern", "confidence": "medium",
                                      "evidence": f"{share:.0%} of {len(v)} liked recipes satisfy '{cname}' vs {b:.0%} for a typical reviewed recipe (z={z:.1f}). Pattern only - not a confirmed restriction."}
    for c, implied in IMPLIES.items():
        if c in constraints:
            for k in implied:
                constraints.pop(k, None)

    # --- text evidence (self-reports are weak: reviews also talk about friends/family/recipes)
    mentions = {k: [] for k in MENTION_LABELS}
    health_hits, household = [], Counter()
    for _, row in rows[rows["mentions_restriction"] | rows["mentions_health"] | rows["mentions_household"]].iterrows():
        t = str(row["text"])
        for lab, pat in MENTION_LABELS.items():
            m = re.search(pat, t, re.I)
            if m:
                mentions[lab].append(snippet(t, m))
        m = HEALTH_RE.search(t)
        if m:
            health_hits.append(snippet(t, m))
        for m in HOUSEHOLD_RE.finditer(t):
            household[m.group(0).lower()] += 1
    for lab, snips in mentions.items():
        if lab == "allergy (unspecified)" and not snips:
            continue
        if len(snips) >= 2:
            ex = constraints.get(lab)
            if ex:
                ex["confidence"] = "high"
                ex["evidence"] += f" Also mentioned in {len(snips)} reviews."
            elif lab in RULES:
                constraints[lab] = {"constraint": lab, "status": "possible_self_report", "confidence": "low",
                                    "evidence": f"Mentioned in {len(snips)} reviews (may refer to other people or the recipe): " + " | ".join(snips[:2])}
            else:
                ev.append({"type": "repeated_mention", "claim": lab, "confidence": "low", "detail": snips[:2]})
        elif len(snips) == 1:
            ev.append({"type": "single_mention", "claim": lab, "confidence": "very_low", "detail": snips})
    ctx = [{"term": t, "n_reviews": int(n), "status": "possible_self_report"} for t, n in household.most_common(3) if n >= 3]

    # --- health goal: unknown unless several health mentions AND a matching 'lighter' tendency
    goal = {"value": "unknown", "status": "unknown", "confidence": None, "evidence": "No clear evidence of a health goal."}
    lighter = [t for t in tend if t["metric"] in ("calories", "sodium", "sugar", "saturated fat") and t["direction"] == "lower"]
    if len(health_hits) >= 3 and lighter:
        goal = {"value": "possibly prefers lighter eating (" + ", ".join(t["metric"] for t in lighter) + ")", "status": "observed_pattern",
                "confidence": "low", "evidence": f"{len(health_hits)} reviews mention health topics and liked recipes skew lower in those nutrients. Not confirmed."}
    else:
        unknown.append("health_goal")
        if health_hits:
            ev.append({"type": "health_mentions", "claim": f"{len(health_hits)} reviews mention health topics", "confidence": "very_low", "detail": health_hits[:2]})

    dur = liked["duration"].dropna()
    time_budget = {"value": None, "status": "unknown", "typical_minutes": float(dur.median()) if len(dur) >= MIN_N_TENDENCY else None,
                   "evidence": "median duration of liked recipes - typical, not a stated limit"}
    unknown.append("time_budget")

    p = {
        "profile_id": f"H{i:03d}", "source": "historical", "label": label,
        "dietary_constraints": list(constraints.values()), "health_goal": goal, "time_budget_minutes": time_budget,
        "taste": {"likes": likes, "dislikes": dislikes}, "nutrition_tendencies": tend, "household_context": ctx,
        "unknown_fields": unknown, "tension": None,
        "history": {"source_user_id": int(uid), "n_reviews": int(len(rows)), "n_liked": int(len(liked)), "n_disliked": int(len(disliked)),
                    "span_years": float(ustats["span_years"]), "mean_rating": float(rows["rating"].mean()), "rating_std": float(rows["rating"].std()),
                    "feature_coverage": round(coverage, 3), "n_modification_mentions": int(rows["mentions_modification"].sum())},
        "evidence": ev,
    }
    p["profile_text"] = render_profile_text(p)
    return p


def main():
    base = json.load(open(PROC_DIR / "baselines.json"))
    us = read_table("user_profile_candidates")
    sel = select_users(us)
    ids = set(sel["new_member_id"])
    rv = read_table("reviews_clean", usecols=["new_member_id", "new_recipe_id", "rating", "rating_type", "liked", "text", "has_text", "date",
                                              "mentions_restriction", "mentions_health", "mentions_household", "mentions_modification"],
                    filter_col="new_member_id", filter_values=ids)
    for c in ("liked", "has_text", "mentions_restriction", "mentions_health", "mentions_household", "mentions_modification"):
        rv[c] = rv[c].astype(bool)
    lookup = read_table("recipe_lookup", filter_col="new_recipe_id", filter_values=set(rv["new_recipe_id"]))
    profiles = []
    for k, (_, row) in enumerate(sel.iterrows(), 1):
        profiles.append(build_profile(k, row["new_member_id"], f"{row['richness']}/{row['signal']}",
                                      rv[rv["new_member_id"] == row["new_member_id"]], lookup, base, row))
    write_jsonl(profiles, "profiles_historical")
    summ = pd.DataFrame([{
        "profile_id": p["profile_id"], "label": p["label"], "n_reviews": p["history"]["n_reviews"], "n_liked": p["history"]["n_liked"],
        "coverage": p["history"]["feature_coverage"], "likes": ", ".join(x["item"] for x in p["taste"]["likes"][:5]),
        "tendencies": "; ".join(f"{t['direction']} {t['metric']}" for t in p["nutrition_tendencies"]),
        "constraints": "; ".join(f"{c['constraint']}[{c['status']}]" for c in p["dietary_constraints"]),
        "goal": p["health_goal"]["value"]} for p in profiles])
    summ.to_csv(PROC_DIR / "profiles_historical_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\nSample profile text:\n", profiles[0]["profile_text"])


if __name__ == "__main__":
    main()
