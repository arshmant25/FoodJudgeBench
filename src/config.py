"""Shared helpers for FoodJudgeBench: paths, table IO, text scrubbing, constraint logic, profile text.

Every script in src/ imports this module (python src/02_build_catalog.py puts src/ on sys.path).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = ROOT / "processed"
PROC_DIR.mkdir(exist_ok=True)

SEED = 42
LIKED_RATING = 4.0          # rating >= this counts as "liked"
DISLIKED_MAX_RATING = 3.0   # star rating <= this counts as "disliked" (2.5 'no stars' rows are excluded)

# Tags that are category headers or duplicate other columns (duration etc.) -> useless as taste evidence.
STRUCTURAL_TAGS = {"time-to-make", "course", "preparation", "main-ingredient", "cuisine", "occasion"}
EXCLUDED_TAGS = STRUCTURAL_TAGS | {
    "dietary", "low-in-something", "taste-mood", "equipment", "number-of-servings", "3-steps-or-less",
    "5-ingredients-or-less", "15-minutes-or-less", "30-minutes-or-less", "60-minutes-or-less", "4-hours-or-less",
}

# diet constraint name -> boolean rule column created in notebook 1 (section 11)
CONSTRAINT_COL = {
    "vegetarian": "rule_vegetarian",
    "vegan": "rule_vegan",
    "dairy-free": "rule_dairy_free",
    "gluten-free": "rule_gluten_free",
    "nut-free": "rule_nut_free",
}
# tags (when present) that independently confirm a rule label
CONSTRAINT_TAG = {"vegetarian": "vegetarian", "vegan": "vegan", "dairy-free": "dairy-free", "gluten-free": "gluten-free"}

# health goal -> operational definition on catalog percentile columns (higher pct = more of that nutrient)
GOAL_RULES = {
    "lower sodium": ("sodium_pct", "<=", 0.33),
    "fewer calories": ("calories_pct", "<=", 0.35),
    "less sugar": ("sugars_pct", "<=", 0.33),
    "less saturated fat": ("satfat_pct", "<=", 0.33),
    "higher protein": ("protein_pct", ">=", 0.67),
    "more fiber": ("fiber_pct", ">=", 0.67),
}

LIST_COLS = ("tags_list", "ingredient_names")


# ----------------------------------------------------------------------------- IO
def table_path(name: str) -> Path:
    """Newest of processed/<name>.parquet | .csv.gz | .csv."""
    cands = [PROC_DIR / f"{name}{ext}" for ext in (".parquet", ".csv.gz", ".csv")]
    cands = [p for p in cands if p.exists()]
    if not cands:
        raise FileNotFoundError(f"processed/{name}.(parquet|csv.gz|csv) not found - run the previous step first")
    return max(cands, key=lambda p: p.stat().st_mtime)


def read_table(name, usecols=None, filter_col=None, filter_values=None, chunksize=250_000) -> pd.DataFrame:
    """Read a processed table (parquet or csv.gz). Optionally keep only rows with filter_col in filter_values
    (applied chunk-wise for csv so huge tables never sit fully in memory). filter_col must be in usecols."""
    p = table_path(name)
    if filter_values is not None:
        filter_values = set(filter_values)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p, columns=list(usecols) if usecols else None)
        if filter_col is not None:
            df = df[df[filter_col].isin(filter_values)]
    else:
        parts = []
        for chunk in pd.read_csv(p, usecols=usecols, chunksize=chunksize):
            if filter_col is not None:
                chunk = chunk[chunk[filter_col].isin(filter_values)]
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True)
    for c in ("date", "first_date", "last_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.reset_index(drop=True)


def write_table(df: pd.DataFrame, name: str) -> Path:
    try:
        p = PROC_DIR / f"{name}.parquet"
        df.to_parquet(p, index=False)
    except Exception as e:  # pyarrow missing etc.
        p = PROC_DIR / f"{name}.csv.gz"
        df.to_csv(p, index=False)
        print(f"  (parquet unavailable: {type(e).__name__}; wrote csv.gz - `pip install pyarrow` to get parquet)")
    print(f"saved {p.relative_to(ROOT)}  {df.shape}")
    return p


def as_list(x):
    """list | ndarray | repr-string-of-list | NaN  ->  python list."""
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, np.ndarray)):
        return list(x)
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
        return list(v) if isinstance(v, (list, tuple)) else []
    return []


def parse_list_cols(df: pd.DataFrame, cols=LIST_COLS) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(as_list)
    return df


def split_items(s) -> list[str]:
    """'a|b|c' (possibly NaN) -> ['a','b','c']"""
    return s.split("|") if isinstance(s, str) and s else []


# ----------------------------------------------------------------------------- text helpers
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def scrub_text(s: str) -> str:
    """Remove emails / URLs before any text is shown to annotators or LLMs."""
    s = _EMAIL.sub("[email]", str(s))
    return _URL.sub("[url]", s)


def snippet(text: str, match: re.Match, width: int = 90) -> str:
    a, b = max(0, match.start() - width), min(len(text), match.end() + width)
    return ("..." if a else "") + scrub_text(text[a:b]).replace("\n", " ").strip() + ("..." if b < len(text) else "")


def norm_ingredient(name: str) -> str:
    n = re.sub(r"\(.*?\)", " ", str(name).lower())
    n = n.split(",")[0]
    n = re.sub(r"[^a-z\- ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def informative_tags(tags) -> list[str]:
    return [t for t in as_list(tags) if t and t not in EXCLUDED_TAGS]


# ----------------------------------------------------------------------------- constraint / goal logic (catalog)
def constraint_mask(df: pd.DataFrame, constraints, max_minutes=None) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in constraints:
        m &= df[CONSTRAINT_COL[c]].astype(bool)
    if max_minutes:
        m &= df["duration"].le(max_minutes)
    return m


def verified_mask(df: pd.DataFrame, constraints) -> pd.Series:
    """Rule says OK AND (a confirming tag exists, or no tag exists for that constraint at all, e.g. nut-free)."""
    m = pd.Series(True, index=df.index)
    for c in constraints:
        m &= df[CONSTRAINT_COL[c]].astype(bool)
        if c in CONSTRAINT_TAG:
            m &= df[f"{c}_status"].eq("verified")
    return m


def goal_mask(df: pd.DataFrame, goal: str) -> pd.Series:
    col, op, thr = GOAL_RULES[goal]
    return df[col].le(thr) if op == "<=" else df[col].ge(thr)


def term_mask(df: pd.DataFrame, regex: str) -> pd.Series:
    """Recipe mentions the term in its normalized ingredients or its informative tags."""
    return df["ing_text"].str.contains(regex, case=False, regex=True, na=False) | \
        df["tags_str"].str.contains(regex, case=False, regex=True, na=False)


# ----------------------------------------------------------------------------- profile text
def _fmt_items(items, k=5):
    return ", ".join(i["item"] for i in items[:k])


def render_profile_text(p: dict) -> str:
    """Deterministic, hedged natural-language rendering of a profile (no LLM needed)."""
    lines = []
    h = p.get("history")
    if p["source"] == "historical" and h:
        lines.append(f"Profile summarised from {h['n_reviews']} recipe reviews over about {h['span_years']:.1f} years "
                     f"({h['n_liked']} liked recipes). Everything below is inferred from past behaviour, not confirmed by the person.")
    else:
        lines.append("Profile specified explicitly for this benchmark (synthetic persona).")

    likes, dislikes = p["taste"]["likes"], p["taste"]["dislikes"]
    if likes:
        verb = "States a liking for" if p["source"] == "synthetic" else "Tends to like recipes featuring"
        lines.append(f"{verb}: {_fmt_items(likes)}.")
    else:
        lines.append("Taste preferences: no clear evidence." if p["source"] == "historical" else "Taste preferences: unknown.")
    if dislikes:
        verb = "Dislikes" if p["source"] == "synthetic" else "Tends to rate lower recipes featuring"
        lines.append(f"{verb}: {_fmt_items(dislikes)}.")

    cons = p["dietary_constraints"]
    if cons:
        parts = []
        for c in cons:
            tag = {"stated": "stated", "observed_pattern": "observed pattern, not confirmed",
                   "possible_self_report": "possibly mentioned in reviews, low confidence"}.get(c["status"], c["status"])
            parts.append(f"{c['constraint']} ({tag})")
        lines.append("Dietary constraints: " + "; ".join(parts) + ".")
    else:
        lines.append("Dietary constraints: none established." if p["source"] == "historical" else "Dietary constraints: none.")

    g = p["health_goal"]
    if g["value"] in ("unknown", None):
        lines.append("Health goal: unknown.")
    elif g["value"] == "none":
        lines.append("Health goal: none stated.")
    else:
        lines.append(f"Health goal: {g['value']} ({g['status'].replace('_', ' ')}"
                     + (f", {g['confidence']} confidence" if g.get("confidence") not in (None, "stated") else "") + ").")

    t = p["time_budget_minutes"]
    if t.get("value"):
        lines.append(f"Time budget: about {t['value']} minutes or less.")
    elif t.get("status") == "stated":
        lines.append("Time budget: no limit.")
    elif t.get("typical_minutes"):
        lines.append(f"Time budget: unknown (liked recipes typically take about {t['typical_minutes']:.0f} minutes).")
    else:
        lines.append("Time budget: unknown.")

    tend = p.get("nutrition_tendencies", [])
    if tend:
        lines.append("Nutrition pattern in liked recipes: " + "; ".join(
            f"{x['direction']} {x['metric']} ({x['confidence']} evidence)" for x in tend) + ".")
    ctx = p.get("household_context", [])
    if ctx:
        lines.append("Possible household context (low confidence): " + ", ".join(f"mentions '{c['term']}'" for c in ctx) + ".")
    if p.get("unknown_fields"):
        lines.append("Unknown: " + ", ".join(p["unknown_fields"]) + ".")
    return " ".join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    raise TypeError(f"not serialisable: {type(o)}")


def write_jsonl(profiles: list[dict], name: str) -> Path:
    p = PROC_DIR / f"{name}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for pr in profiles:
            f.write(json.dumps(pr, default=_json_default, ensure_ascii=False) + "\n")
    print(f"saved {p.relative_to(ROOT)}  ({len(profiles)} profiles)")
    return p
