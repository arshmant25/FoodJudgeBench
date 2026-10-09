"""One-off patch for src/03a_historical_profiles.py: stop junk items from becoming "likes".

Why: your historical summary lists items such as "use more or less", "ice cubes", "microwave", "mixer", "from-scratch",
"romantic", "gifts", "seasonal" as things a user "likes". These are parsing artefacts or recipe-process/occasion tags, not taste.
Health tags ("diabetic", "low-saturated-fat", "very-low-carbs", "healthy", "high-calcium") are also removed from taste likes:
the nutrition tendencies already capture health-related behaviour, and listing a tag as a "like" invites invented personalization.

This does NOT change which users are selected (same seed, same order), only the likes/dislikes lists and the profile text.
Run once from the project root:   python src/patch_03a_clean_likes.py   then rerun 03a.
Safe: aborts without writing if the expected text is missing; keeps 03a_historical_profiles.py.bak2.
"""
from pathlib import Path
import shutil
import sys

path = Path(__file__).resolve().parent / "03a_historical_profiles.py"
s = path.read_text(encoding="utf-8")
if "JUNK_ITEMS" in s:
    sys.exit("already patched - nothing to do")

CONST = '''
# Items that must never be reported as taste evidence (parsing artefacts, process/equipment/occasion tags, health tags).
JUNK_ITEMS = {
    "use more or less", "ice cubes", "ice", "water", "warm water", "cold water", "ice water", "hot water", "boiling water",
    "salt", "pepper", "salt and pepper", "meat", "microwave", "mixer", "from-scratch", "romantic", "seasonal", "gifts",
    "for-large-groups", "oamc-freezer-make-ahead", "equipment", "easy", "to-go", "inexpensive",
    # health/diet tags: handled by nutrition tendencies, not taste
    "diabetic", "low-saturated-fat", "very-low-carbs", "healthy", "high-calcium", "low-carb", "low-sodium", "low-fat",
    "low-calorie", "low-cholesterol", "high-protein", "high-fiber", "low-protein", "high-in-something-diabetic-friendly",
}
'''
try:
    anchor = "MIN_DISLIKED = 5\n"
    assert anchor in s, "config anchor"
    s = s.replace(anchor, anchor + CONST, 1)
    old = "    for item, c in cnt.items():\n        if c < min_count:\n            continue\n"
    assert old in s, "item_lifts loop anchor"
    s = s.replace(old, "    for item, c in cnt.items():\n        if c < min_count or item in JUNK_ITEMS or len(item) < 3:\n            continue\n", 1)
    compile(s, str(path), "exec")
except (AssertionError, SyntaxError) as e:
    sys.exit(f"patch aborted, file untouched: {type(e).__name__}: {e}")

shutil.copy(path, path.with_suffix(".py.bak2"))
path.write_text(s, encoding="utf-8")
print("patched", path.name, "(backup: 03a_historical_profiles.py.bak2)")
