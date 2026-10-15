"""Step 2 - cleaning / catalog build.

Inputs : processed/recipes_flagged, (data/pp_recipes.csv for ingredient quantities + directions)
Outputs: processed/catalog           (~N_CATALOG recipes, full text + nutrition + percentiles + constraint status)
         processed/recipe_lookup     (all quality_ok reviewed recipes, slim; used to build historical profiles)
         processed/baselines.json    (review-weighted base rates used for 'is this tendency real?' tests)
         processed/catalog_report.json
Run    : python src/02_build_catalog.py
"""
import json
import re
import ast
from collections import Counter

import numpy as np
import pandas as pd

from config import (DATA_DIR, PROC_DIR, SEED, read_table, write_table, parse_list_cols, informative_tags,
                    norm_ingredient, CONSTRAINT_COL, CONSTRAINT_TAG)

N_CATALOG = 6000
MIN_PER_STRATUM = 150
MIN_TAG_RECIPES = 50          # tags / ingredients rarer than this get no baseline entry
MIN_ING_RECIPES = 100

PCT_SOURCES = {
    "calories_pct": "calories [cal]", "sodium_pct": "sodium [mg]", "sugars_pct": "sugars [g]",
    "satfat_pct": "saturatedFat [g]", "protein_pct": "protein [g]", "fiber_pct": "dietaryFiber [g]",
    "who_pct": "who_score", "duration_pct": "duration",
}


def wstats(x, w):
    m = np.average(x, weights=w)
    return float(m), float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def format_ingredients(s):
    try:
        d = ast.literal_eval(s)
    except Exception:
        return []
    out = []
    if isinstance(d, dict):
        for items in d.values():
            for it in items:
                qty = re.sub(r"\s*time\(s\)\s*", " ", str(it[1])).strip() if len(it) > 1 else ""
                out.append(re.sub(r"\s+", " ", f"{qty} {it[0]}").strip())
    return out


def format_directions(s):
    try:
        v = ast.literal_eval(s)
    except Exception:
        return str(s) if isinstance(s, str) else ""
    if isinstance(v, dict):
        v = [x for vals in v.values() for x in (vals if isinstance(vals, list) else [vals])]
    return " ".join(f"{i}. {str(x).strip()}" for i, x in enumerate(v, 1)) if isinstance(v, (list, tuple)) else str(v)


def main():
    rng = np.random.default_rng(SEED)
    print("loading recipes_flagged ...")
    rf = read_table("recipes_flagged")
    Q = rf[rf["quality_ok"].astype(bool)].copy()
    del rf
    print(f"quality_ok recipes: {len(Q):,}")
    parse_list_cols(Q)

    # ---- derived columns
    Q["tags_inf"] = Q["tags_list"].apply(informative_tags)
    Q["tags_str"] = Q["tags_inf"].apply("|".join)
    Q["ing_norm"] = Q["ingredient_names"].apply(lambda l: sorted({n for n in map(norm_ingredient, l) if n}))
    Q["ing_str"] = Q["ing_norm"].apply("|".join)
    Q["ing_text"] = Q["ingredient_names"].apply(lambda l: " ; ".join(l).lower())
    for dst, src in PCT_SOURCES.items():
        Q[dst] = Q[src].rank(pct=True)
    Q["nutri_level"] = (Q["nutri_score"] * 4).round().astype(int)      # 0..4, higher = healthier (verified in EDA 9.4)
    Q["fsa_level"] = (Q["fsa_score"] * 8).round().astype(int)          # 0..8, higher = healthier
    for c, rule in CONSTRAINT_COL.items():
        Q[rule] = Q[rule].astype(bool)
    for c, tag in CONSTRAINT_TAG.items():
        has_tag = Q["tags_list"].apply(lambda l, t=tag: t in l)
        Q[f"{c}_status"] = np.where(~Q[CONSTRAINT_COL[c]], "violates", np.where(has_tag, "verified", "rule_only"))

    # ---- review-weighted baselines (what a *typical reviewed* recipe looks like)
    Qr = Q[Q["n_reviews"] > 0]
    w = Qr["n_reviews"].to_numpy(float)
    base = {"pct_metrics": {}, "rule_rates": {}, "tag_rates": {}, "ingredient_rates": {}}
    for dst in PCT_SOURCES:
        m, sd = wstats(Qr[dst].to_numpy(float), w)
        base["pct_metrics"][dst] = {"mean": m, "sd": sd}
    for rule in CONSTRAINT_COL.values():
        base["rule_rates"][rule] = float(np.average(Qr[rule].to_numpy(float), weights=w))

    def weighted_rates(col, min_recipes):
        sub = Qr[Qr[col].apply(len) > 0][[col, "n_reviews"]]
        total = sub["n_reviews"].sum()
        ex = sub.explode(col)
        g = ex.groupby(col).agg(w=("n_reviews", "sum"), n=("n_reviews", "size"))
        g = g[g["n"] >= min_recipes]
        return {k: float(v / total) for k, v in g["w"].items()}

    base["tag_rates"] = weighted_rates("tags_inf", MIN_TAG_RECIPES)
    base["ingredient_rates"] = weighted_rates("ing_norm", MIN_ING_RECIPES)
    with open(PROC_DIR / "baselines.json", "w") as f:
        json.dump(base, f)
    print(f"baselines: {len(base['tag_rates'])} tags, {len(base['ingredient_rates'])} ingredients")

    # ---- slim lookup for historical profiling
    lookup_cols = ["new_recipe_id", "duration", "nutri_level", "fsa_level", "tags_str", "ing_str"] + list(PCT_SOURCES) + list(CONSTRAINT_COL.values())
    write_table(Qr[lookup_cols], "recipe_lookup")

    # ---- catalog: candidate recipes, stratified sample
    C = Q[Q["catalog_candidate"].astype(bool)].copy()
    C["time_bucket"] = pd.cut(C["duration"], [0, 30, 60, 1e9], labels=["<=30", "31-60", ">60"])
    C["stratum"] = C["nutri_level"].astype(str) + "|" + C["time_bucket"].astype(str)
    sizes = C["stratum"].value_counts()
    target = {s: int(min(n, max(MIN_PER_STRATUM, round(N_CATALOG * n / len(C))))) for s, n in sizes.items()}
    picks = [g.sample(target[s], random_state=SEED) for s, g in C.groupby("stratum")]
    cat = pd.concat(picks).reset_index(drop=True)
    print(f"candidates={len(C):,} -> catalog={len(cat):,} recipes over {len(sizes)} strata")

    # ---- attach ingredient quantities + directions from the raw file (catalog ids only)
    ids = set(cat["new_recipe_id"])
    parts = []
    for ch in pd.read_csv(DATA_DIR / "pp_recipes.csv", usecols=["new_recipe_id", "ingredients", "directions"], chunksize=50_000):
        parts.append(ch[ch["new_recipe_id"].isin(ids)])
    raw = pd.concat(parts)
    raw["ingredients_display"] = raw["ingredients"].apply(format_ingredients)
    raw["directions_text"] = raw["directions"].apply(format_directions)
    cat = cat.merge(raw[["new_recipe_id", "ingredients_display", "directions_text"]], on="new_recipe_id", how="left")

    keep = (["new_recipe_id", "title", "description", "duration", "servingsPerRecipe", "servingSize [g]",
             "calories [cal]", "totalFat [g]", "saturatedFat [g]", "cholesterol [mg]", "sodium [mg]",
             "totalCarbohydrate [g]", "dietaryFiber [g]", "sugars [g]", "protein [g]",
             "who_score", "fsa_score", "nutri_score", "nutri_level", "fsa_level", "n_reviews", "mean_true_rating",
             "tags_str", "ing_text", "ingredient_names", "ingredients_display", "directions_text", "stratum"]
            + list(PCT_SOURCES) + list(CONSTRAINT_COL.values()) + [f"{c}_status" for c in CONSTRAINT_TAG]
            + [c for c in cat.columns if c.startswith("under_")])
    cat = cat[keep]
    write_table(cat, "catalog")

    report = {
        "quality_ok_recipes": int(len(Q)), "candidate_recipes": int(len(C)), "catalog_size": int(len(cat)),
        "strata": cat["stratum"].value_counts().to_dict(),
        "constraint_share_pct": {k: round(float(cat[v].mean() * 100), 1) for k, v in CONSTRAINT_COL.items()},
        "status_counts": {c: cat[f"{c}_status"].value_counts().to_dict() for c in CONSTRAINT_TAG},
        "under_30min_pct": round(float(cat["under_30min"].mean() * 100), 1),
        "missing_directions": int((cat["directions_text"].fillna("") == "").sum()),
    }
    with open(PROC_DIR / "catalog_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
