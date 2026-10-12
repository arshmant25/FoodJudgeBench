"""Patch v2 for src/03a_historical_profiles.py: keep ONLY real taste evidence in "likes" / "dislikes".

Why v1 was not enough: besides parsing artefacts, the likes still contained
  * diet tags ("vegan", "vegetarian", ...)      -> diets are handled as constraints, never as taste
  * health / nutrition tags ("high-in-something", "low-fat", "diabetic", ...) -> handled by nutrition tendencies
  * equipment / process / occasion tags ("bread-machine", "microwave", "from-scratch", "seasonal", "fall", ...)
  * over-narrow dish-type tags ("punch")
This version uses an explicit blocklist PLUS regex patterns, applied to likes AND dislikes. It works whether or not v1 was applied.
It does NOT change which users are selected, only the likes/dislikes lists and the profile text.

Run once from the project root:   python src/patch_03a_clean_likes_v2.py     then rerun 03a, 04 (and 05 with --fresh).
Safe: aborts without writing if the expected text is missing; keeps 03a_historical_profiles.py.bak3.
After rerunning 03a, look at the printout "distinct likes" at the end of this script's instructions (python src/patch_03a_clean_likes_v2.py --audit)
and add anything still noisy to EXTRA_JUNK below / in 03a.
"""
from pathlib import Path
import json
import shutil
import sys
from collections import Counter

here = Path(__file__).resolve().parent
path = here / "03a_historical_profiles.py"

if "--audit" in sys.argv:                       # list every distinct like/dislike in the current profiles
    f = here.parent / "processed" / "profiles_historical.jsonl"
    c, d = Counter(), Counter()
    for l in open(f, encoding="utf-8"):
        p = json.loads(l)
        c.update(x["item"] for x in p["taste"]["likes"])
        d.update(x["item"] for x in p["taste"]["dislikes"])
    print("LIKES   :", ", ".join(f"{k} ({v})" for k, v in sorted(c.items())))
    print("DISLIKES:", ", ".join(f"{k} ({v})" for k, v in sorted(d.items())))
    sys.exit()

s = path.read_text(encoding="utf-8")
if "def is_junk_item" in s:
    sys.exit("v2 already applied - nothing to do")

CONST = r'''
# ---- items that must never be reported as taste evidence (patch v2) -------------------------------------------------
JUNK_ITEMS = {
    # parsing artefacts / generic pantry
    "use more or less", "ice cubes", "ice", "water", "warm water", "cold water", "ice water", "hot water", "boiling water",
    "salt", "pepper", "salt and pepper", "meat", "ingredients", "optional", "garnish",
    # diets (handled as constraints) and health tags (handled by nutrition tendencies)
    "vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free", "egg-free", "lactose-free", "kosher", "halal",
    "diabetic", "healthy", "high-in-something", "high-in-something-diabetic-friendly", "low-in-something", "heart-healthy",
    # equipment / process / convenience
    "equipment", "bread-machine", "microwave", "mixer", "blender", "food-processor-blender", "crock-pot-slow-cooker",
    "slow-cooker", "dutch-oven", "pressure-cooker", "stove-top", "oven", "grilling", "from-scratch", "oamc-freezer-make-ahead",
    "freezer", "make-ahead", "easy", "to-go", "inexpensive", "one-dish-meal", "for-large-groups", "for-1-or-2", "small-appliance",
    # occasion / season / mood
    "romantic", "seasonal", "gifts", "holiday-event", "fall", "spring", "summer", "winter", "christmas", "thanksgiving",
    "valentines-day", "st-patricks-day", "new-years", "4th-of-july", "superbowl", "taste-mood", "comfort-food",
    # over-narrow dish-type tags
    "punch", "mixer-drink",
}
JUNK_PATTERNS = re.compile(
    r"^(?:low|very-low|high|reduced|no)[- ](?:in-|carb|fat|sodium|calorie|cholesterol|sugar|protein|fiber|saturated)|"
    r"-free$|-friendly$|something|^\d+-|-or-less$|^(?:number|time)-", re.I)


def is_junk_item(item: str) -> bool:
    return len(item) < 3 or item in JUNK_ITEMS or bool(JUNK_PATTERNS.search(item))
'''
V1 = "    for item, c in cnt.items():\n        if c < min_count or item in JUNK_ITEMS or len(item) < 3:\n            continue\n"
ORIG = "    for item, c in cnt.items():\n        if c < min_count:\n            continue\n"
NEW = "    for item, c in cnt.items():\n        if c < min_count or is_junk_item(item):\n            continue\n"
try:
    if "JUNK_ITEMS = {" in s:                    # v1 applied: drop its constant block, keep one source of truth
        a = s.index("\n# Items that must never be reported as taste evidence")
        b = s.index("}\n", a) + 2
        s = s[:a] + s[b:]
    anchor = "MIN_DISLIKED = 5\n"
    assert anchor in s, "config anchor"
    s = s.replace(anchor, anchor + CONST, 1)
    assert (V1 in s) or (ORIG in s), "item_lifts loop anchor"
    s = s.replace(V1, NEW, 1) if V1 in s else s.replace(ORIG, NEW, 1)
    compile(s, str(path), "exec")
except (AssertionError, SyntaxError, ValueError) as e:
    sys.exit(f"patch aborted, file untouched: {type(e).__name__}: {e}")
shutil.copy(path, path.with_suffix(".py.bak3"))
path.write_text(s, encoding="utf-8")
print("patched", path.name, "(backup: 03a_historical_profiles.py.bak3)")