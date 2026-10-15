"""Step 4 - recommendation pool (one method: content-based).

For every profile (30 historical + 30 synthetic) build a small, *tagged* pool of candidate recipes from the catalog:

  strong              top-3 content matches that satisfy the profile's hard constraints (diverse, no near-duplicates)
  weak_taste          satisfies hard constraints but is a poor taste match (bottom quartile)
  weak_goal           good taste match that FAILS the stated goal / contradicts the observed nutrition tendency
  weak_dislike        good taste match that contains something the profile dislikes
  violates_constraint good taste match that BREAKS a hard dietary constraint  (planned diagnostic, never a normal pick)

Why weak candidates: annotators and judges need something to disagree about; a pool of only 'best picks' makes every
explanation look fine. Each candidate stores `facts` (verified, checkable evidence) - the permitted fact list for Step 5 and
the ground truth for Step 6 diagnostics. Two candidates per profile are flagged `selected_for_case` (case_slot 1 and 2).

How the method works
  historical : TF-IDF cosine between each catalog recipe and the centroid of the user's liked recipes
               (minus 0.5 x centroid of disliked recipes if >=5), already-reviewed recipes excluded.
  synthetic  : share of the stated liked terms found in the recipe (ingredients/tags), + quality prior (+ goal match).
  Both are mapped to a 0-1 score. Weights are in the config below.

Inputs : processed/catalog, profiles_historical.jsonl, profiles_synthetic.jsonl, reviews_clean, recipe_lookup
Outputs: processed/recommendation_pool.jsonl, recommendation_pool_summary.csv, constraint_check_sheet.csv
Run    : python src/04_recommendation_pool.py
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (PROC_DIR, SEED, GOAL_RULES, CONSTRAINT_COL, CONSTRAINT_TAG, read_table, parse_list_cols, norm_ingredient,
                    split_items, constraint_mask, goal_mask, term_mask, write_jsonl, _json_default)

# ------------------------------------------------------------------ config
N_STRONG = 3
MMR_MAX_SIM = 0.85                       # a 2nd/3rd strong pick may not be this similar to one already chosen
HARD_STATUSES = {"stated", "observed_pattern"}   # constraint statuses the recommender must respect
W_HIST = {"sim": 0.85, "quality": 0.15}
W_SYN_GOAL = {"like": 0.50, "quality": 0.20, "goal": 0.30}      # profile has likes and a stated goal
W_SYN_LIKE = {"like": 0.75, "quality": 0.25}                    # likes, no goal
W_SYN_NONE = {"quality": 1.00}                                  # nothing known -> best-rated recipes
MIN_LIKED_FOR_QUERY = 10
DISLIKE_PENALTY = 0.5
# Each profile is evaluated on TWO recipes (case slots 1 and 2):
#   slot 1 = best strong match (rank 1)
#   slot 2 = cycles weak_goal -> weak_taste -> second strong match, with fallbacks if a role is unavailable for that profile
# Overall: ~60 strong rank-1, ~20 strong rank-2, ~20 weak_goal, ~20 weak_taste  (= 120 recipe-profile pairs)
SLOT2_CYCLE = [("weak_goal", 1), ("weak_taste", 1), ("strong", 2)]
SLOT2_FALLBACK = [("weak_goal", 1), ("weak_taste", 1), ("strong", 2), ("strong", 3)]
PCT_COL = {"calories": "calories_pct", "sodium": "sodium_pct", "sugar": "sugars_pct", "saturated fat": "satfat_pct",
           "protein": "protein_pct", "fiber": "fiber_pct", "cooking time": "duration_pct", "overall healthiness score": "who_pct"}
RAW_NUTR = {"calories": "calories [cal]", "total_fat_g": "totalFat [g]", "saturated_fat_g": "saturatedFat [g]", "sodium_mg": "sodium [mg]",
            "carbs_g": "totalCarbohydrate [g]", "fiber_g": "dietaryFiber [g]", "sugars_g": "sugars [g]", "protein_g": "protein [g]",
            "cholesterol_mg": "cholesterol [mg]"}


# ------------------------------------------------------------------ text -> tokens
def tokens(ing_str, tags_str):
    items = split_items(ing_str)
    words = [w for it in items for w in it.split() if len(w) > 2]
    return items + words + ["tag:" + t for t in split_items(tags_str)]


def load_profiles():
    out = []
    for name in ("profiles_historical", "profiles_synthetic"):
        with open(PROC_DIR / f"{name}.jsonl", encoding="utf-8") as f:
            out += [json.loads(l) for l in f]
    return out


def item_mask(cat, item):
    """Boolean mask over catalog for a taste item (synthetic term with regex | historical tag/ingredient)."""
    if item.get("match_regex"):
        return term_mask(cat, item["match_regex"])
    name = re.escape(item["item"])
    if item.get("kind") == "tag":
        return cat["tags_str"].str.contains(rf"(?:^|\|){name}(?:\||$)", regex=True)
    return cat["ing_norm_str"].str.contains(name, regex=True)


def pct_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


# ------------------------------------------------------------------ historical query vectors
def historical_queries(profiles, vec, lookup_cols=("new_recipe_id", "tags_str", "ing_str")):
    users = {p["history"]["source_user_id"] for p in profiles if p["source"] == "historical"}
    rv = read_table("reviews_clean", usecols=["new_member_id", "new_recipe_id", "rating", "rating_type", "liked"],
                    filter_col="new_member_id", filter_values=users)
    rv["liked"] = rv["liked"].astype(bool)
    lk = read_table("recipe_lookup", usecols=list(lookup_cols), filter_col="new_recipe_id", filter_values=set(rv["new_recipe_id"]))
    lk = lk.set_index("new_recipe_id")
    res = {}
    for p in profiles:
        if p["source"] != "historical":
            continue
        uid = p["history"]["source_user_id"]
        r = rv[rv["new_member_id"] == uid]
        reviewed = set(r["new_recipe_id"])

        def centroid(sub, weights=None):
            sub = sub[sub["new_recipe_id"].isin(lk.index)]
            if len(sub) < 1:
                return None, 0
            docs = [tokens(lk.at[i, "ing_str"], lk.at[i, "tags_str"]) for i in sub["new_recipe_id"]]
            M = vec.transform(docs)
            w = np.ones(len(sub)) if weights is None else weights(sub)
            c = np.asarray(M.multiply(w[:, None]).sum(axis=0)).ravel()
            n = np.linalg.norm(c)
            return (c / n if n > 0 else None), len(sub)

        q, n_like = centroid(r[r["liked"]], lambda s: np.clip(s["rating"].to_numpy() - 3.0, 0.5, None))
        d, n_dis = centroid(r[(r["rating"] <= 3) & (r["rating_type"] == "stars")])
        res[p["profile_id"]] = {"q": q, "n_like": n_like, "d": d if n_dis >= 5 else None, "n_dis": n_dis, "reviewed": reviewed}
    return res


# ------------------------------------------------------------------ per-profile scoring
def hard_spec(p):
    cons = [c["constraint"] for c in p["dietary_constraints"] if c["status"] in HARD_STATUSES]
    t = p["time_budget_minutes"]
    tlim = t["value"] if t.get("status") == "stated" and t.get("value") else None
    return cons, tlim


def recipe_facts(row, p, cons, tlim, goal_key, like_hit, dis_hit, tendencies):
    f = {"matches_liked": like_hit, "contains_disliked": dis_hit}
    checks = []
    for c in cons:
        passes = bool(row[CONSTRAINT_COL[c]])
        status = row[f"{c}_status"] if c in CONSTRAINT_TAG else ("rule_only" if passes else "violates")
        checks.append({"constraint": c, "rule_passes": passes, "label_status": status})
    f["constraint_checks"] = checks
    f["time"] = {"duration_minutes": float(row["duration"]), "limit_minutes": tlim,
                 "within_limit": (bool(row["duration"] <= tlim) if tlim else None)}
    if goal_key:
        col, op, thr = GOAL_RULES[goal_key]
        f["goal"] = {"goal": goal_key, "percentile_column": col, "recipe_percentile": round(float(row[col]), 3),
                     "rule": f"{col} {op} {thr}", "meets_goal": bool(row[col] <= thr if op == "<=" else row[col] >= thr)}
    else:
        f["goal"] = None
    f["tendency_alignment"] = [{"metric": t["metric"], "direction": t["direction"], "recipe_percentile": round(float(row[PCT_COL[t["metric"]]]), 3),
                                "aligned": bool((row[PCT_COL[t["metric"]]] <= 0.5) if t["direction"] == "lower" else (row[PCT_COL[t["metric"]]] >= 0.5))}
                               for t in tendencies]
    f["nutrition_per_serving"] = {k: round(float(row[v]), 1) for k, v in RAW_NUTR.items()}
    f["nutrition_percentiles"] = {k: round(float(row[v]), 3) for k, v in PCT_COL.items() if v in row.index and k != "cooking time"}
    f["health_levels"] = {"nutri_level_0_to_4": int(row["nutri_level"]), "fsa_level_0_to_8": int(row["fsa_level"]), "note": "higher = healthier"}
    return f


def build_pool(p, cat, X, quality_pct, hq, rng):
    cons, tlim = hard_spec(p)
    goal_key = p["health_goal"]["value"] if (p["health_goal"]["value"] in GOAL_RULES and p["health_goal"]["status"] in HARD_STATUSES) else None
    tend = [t for t in p.get("nutrition_tendencies", []) if t["metric"] in PCT_COL]
    likes, dislikes = p["taste"]["likes"], p["taste"]["dislikes"]
    like_masks = {x["item"]: item_mask(cat, x) for x in likes}
    dis_masks = {x["item"]: item_mask(cat, x) for x in dislikes}
    dis_any = pd.Series(False, index=cat.index)
    for m in dis_masks.values():
        dis_any |= m

    # ---- score every catalog recipe
    if p["source"] == "historical":
        info = hq[p["profile_id"]]
        if info["q"] is None or info["n_like"] < MIN_LIKED_FOR_QUERY:
            return None, f"only {info['n_like']} liked recipes with features"
        sim = X @ info["q"]
        if info["d"] is not None:
            sim = sim - DISLIKE_PENALTY * (X @ info["d"])
        score = W_HIST["sim"] * pct_rank(sim) + W_HIST["quality"] * quality_pct
        method, already = "content_based_tfidf_centroid", info["reviewed"]
    else:
        already = set()
        like_frac = np.mean([m.to_numpy(float) for m in like_masks.values()], axis=0) if like_masks else None
        if goal_key and like_frac is not None:
            w = W_SYN_GOAL; score = w["like"] * like_frac + w["quality"] * quality_pct + w["goal"] * goal_mask(cat, goal_key).to_numpy(float)
        elif like_frac is not None:
            w = W_SYN_LIKE; score = w["like"] * like_frac + w["quality"] * quality_pct
        elif goal_key:
            score = 0.6 * quality_pct + 0.4 * goal_mask(cat, goal_key).to_numpy(float)
        else:
            score = W_SYN_NONE["quality"] * quality_pct
        sim = score
        method = "content_based_term_match"
    score = np.asarray(score, dtype=float)

    not_reviewed = ~cat["new_recipe_id"].isin(already)
    hard_ok = constraint_mask(cat, cons, tlim) & not_reviewed
    strong_ok = hard_ok & ~dis_any
    picked = {}

    def add(role, idx):
        picked.setdefault(role, []).append(int(idx))

    # strong: greedy diverse top-k
    order = np.argsort(-score)
    chosen = []
    for i in order:
        if not strong_ok.iloc[i]:
            continue
        if all(float(X[i].multiply(X[j]).sum()) < MMR_MAX_SIM for j in chosen):
            chosen.append(i)
            add("strong", i)
        if len(chosen) >= N_STRONG:
            break
    used = set(chosen)

    def best(mask):
        for i in order:
            if mask.iloc[i] and i not in used:
                return i
        return None

    # weak_taste: random recipe from the bottom quartile of scores among hard-ok recipes
    hok = np.where(hard_ok.to_numpy())[0]
    if len(hok) >= 8:
        q25 = np.quantile(score[hok], 0.25)
        pool_low = [i for i in hok if score[i] <= q25 and i not in used and not dis_any.iloc[i]]
        if pool_low:
            i = int(rng.choice(pool_low)); add("weak_taste", i); used.add(i)
    # weak_goal: good taste match that fails the stated goal / contradicts the strongest tendency
    if goal_key:
        i = best(hard_ok & ~goal_mask(cat, goal_key) & ~dis_any)
    elif tend:
        t = max(tend, key=lambda x: abs(x["z"])); col = PCT_COL[t["metric"]]
        opposite = (cat[col] >= 0.67) if t["direction"] == "lower" else (cat[col] <= 0.33)
        i = best(hard_ok & opposite & ~dis_any)
    else:
        i = None
    if i is not None:
        add("weak_goal", i); used.add(i)
    # weak_dislike
    if dis_masks:
        i = best(hard_ok & dis_any)
        if i is not None:
            add("weak_dislike", i); used.add(i)
    # planned diagnostic: breaks a hard constraint
    if cons:
        viol = ~constraint_mask(cat, cons, None) & not_reviewed
        if tlim:
            viol &= cat["duration"].le(tlim)
        i = best(viol)
        if i is not None:
            add("violates_constraint", i); used.add(i)

    recs = []
    for role, idxs in picked.items():
        for rank, i in enumerate(idxs, 1):
            row = cat.iloc[i]
            lh = [k for k, m in like_masks.items() if m.iloc[i]]
            dh = [k for k, m in dis_masks.items() if m.iloc[i]]
            recs.append({"profile_id": p["profile_id"], "profile_source": p["source"], "recipe_id": int(row["new_recipe_id"]),
                         "role": role, "rank_in_role": rank, "score": round(float(score[i]), 4), "method": method,
                         "planned_diagnostic": role == "violates_constraint", "selected_for_case": False, "case_slot": None,
                         "facts": recipe_facts(row, p, cons, tlim, goal_key, lh, dh, tend)})
    return recs, None


def main():
    rng = np.random.default_rng(SEED)
    cat = read_table("catalog")
    parse_list_cols(cat)
    for c in ("tags_str", "ing_text"):
        cat[c] = cat[c].fillna("")
    cat["ing_norm_str"] = cat["ingredient_names"].apply(lambda l: "|".join(sorted({n for n in map(norm_ingredient, l) if n})))
    for c in CONSTRAINT_COL.values():
        cat[c] = cat[c].astype(bool)
    print(f"catalog: {len(cat):,} recipes")

    vec = TfidfVectorizer(analyzer=lambda x: x, min_df=2, sublinear_tf=True)
    X = vec.fit_transform([tokens(a, b) for a, b in zip(cat["ing_norm_str"], cat["tags_str"])]).tocsr()
    print(f"TF-IDF matrix: {X.shape}")
    # bayesian-smoothed quality prior
    m, C = 5.0, float(cat["mean_true_rating"].mean())
    qual = (cat["n_reviews"] * cat["mean_true_rating"] + m * C) / (cat["n_reviews"] + m)
    quality_pct = pct_rank(qual)

    profiles = load_profiles()
    hq = historical_queries(profiles, vec)
    all_recs, skipped = [], []
    pos = {"historical": 0, "synthetic": 0}
    for p in profiles:
        recs, why = build_pool(p, cat, X, quality_pct, hq, rng)
        if recs is None:
            skipped.append((p["profile_id"], why)); continue
        # choose the two evaluated recipes for this profile
        def find(role, rank):
            c = [r for r in recs if r["role"] == role and r["rank_in_role"] == rank]
            return c[0] if c else None

        slot1 = find("strong", 1)
        want = SLOT2_CYCLE[pos[p["source"]] % len(SLOT2_CYCLE)]
        pos[p["source"]] += 1
        slot2 = None
        for role, rank in [want] + SLOT2_FALLBACK:
            slot2 = find(role, rank)
            if slot2 is not None and slot2 is not slot1:
                break
            slot2 = None
        for slot, r in ((1, slot1), (2, slot2)):
            if r is not None:
                r["selected_for_case"], r["case_slot"] = True, slot
        if slot1 is None or slot2 is None:
            skipped.append((p["profile_id"], f"only {sum(x is not None for x in (slot1, slot2))} of 2 case recipes available"))
        all_recs += recs
    write_jsonl(all_recs, "recommendation_pool")

    summ = pd.DataFrame([{k: r[k] for k in ("profile_id", "profile_source", "recipe_id", "role", "rank_in_role", "score", "selected_for_case", "case_slot", "planned_diagnostic")}
                         | {"title": cat.set_index("new_recipe_id").at[r["recipe_id"], "title"],
                            "duration": r["facts"]["time"]["duration_minutes"],
                            "meets_goal": (r["facts"]["goal"] or {}).get("meets_goal"),
                            "n_matches_liked": len(r["facts"]["matches_liked"]),
                            "n_disliked_hit": len(r["facts"]["contains_disliked"]),
                            "constraint_labels": ",".join(f"{c['constraint']}:{c['label_status']}" for c in r["facts"]["constraint_checks"])}
                         for r in all_recs])
    summ.to_csv(PROC_DIR / "recommendation_pool_summary.csv", index=False)

    # hand-check sheet: every recipe whose constraint label matters for a normal case or a planned violation
    ct = cat.set_index("new_recipe_id")
    rows = []
    for r in all_recs:
        if not (r["selected_for_case"] or r["planned_diagnostic"]):
            continue
        for c in r["facts"]["constraint_checks"]:
            rows.append({"profile_id": r["profile_id"], "recipe_id": r["recipe_id"], "title": ct.at[r["recipe_id"], "title"],
                         "constraint": c["constraint"], "claimed": "OK" if c["rule_passes"] else "VIOLATES", "label_status": c["label_status"],
                         "ingredients": " ; ".join(ct.at[r["recipe_id"], "ingredients_display"] if isinstance(ct.at[r["recipe_id"], "ingredients_display"], list)
                                                  else [str(ct.at[r["recipe_id"], "ing_text"])]),
                         "manual_ok": ""})
    pd.DataFrame(rows).to_csv(PROC_DIR / "constraint_check_sheet.csv", index=False)

    # ---- console report
    print("\nroles per source:")
    print(summ.groupby(["profile_source", "role"]).size().unstack(fill_value=0).to_string())
    sel = summ[summ["selected_for_case"]]
    print(f"\nselected for case: {len(sel)} recipe-profile pairs over {sel['profile_id'].nunique()} profiles (target 120 over 60) | by role: {sel['role'].value_counts().to_dict()}")
    if skipped:
        print("SKIPPED profiles:", skipped)
    few = summ[summ["role"] == "strong"].groupby("profile_id").size()
    short = [pid for pid in summ["profile_id"].unique() if few.get(pid, 0) < N_STRONG]
    if short:
        print("WARNING profiles with <3 strong candidates:", short)
    print("constraint labels that need hand-checking (rule_only share among selected):",
          round(float(summ[summ['selected_for_case']]['constraint_labels'].str.contains('rule_only').mean() * 100), 1), "%")


if __name__ == "__main__":
    main()
