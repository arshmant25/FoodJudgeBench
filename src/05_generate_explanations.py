"""Step 5 - generate the explanations that humans and LLM judges will later score.

For every selected recipe-profile pair from Step 4 (2 per profile -> ~120 cases) this writes FOUR explanations:

  template            fixed jinja2 sentences filled only from verified values (no LLM)
  direct_llm          an LLM writes freely from the profile card and the recipe card
  grounded_llm        an LLM may use ONLY a numbered permitted-fact list and must cite fact ids   (RAG-style)
  grounded_check_llm  the grounded draft is fact-checked claim by claim against the same list, then revised

-> 120 x 4 = 480 explanation cases. Optionally (--framed-n, default 20) a goal-framed variant of the grounded technique is
generated for profiles with a stated health goal, to isolate "nudge framing" (these are EXTRA, not part of the 480).

Design notes
  * One generator model for all LLM techniques, so techniques differ in method, not model.
  * The self-check technique reuses the grounded draft: any difference is caused by the check/revise step alone.
  * Only profile summaries, recipe data and derived facts are sent to the API: no review text, no user ids.
  * Resumable: records are appended as they finish; the LLM cache makes re-runs free. Stops cleanly on a daily quota.

Usage (from the project root)
  python src/llm_client.py --provider ollama --model llama3.1:8b --ping         # offline smoke test, no API
  python src/05_generate_explanations.py --provider ollama --model llama3.1:8b --dry-run --limit 4  # build facts + prompts, no calls
  python src/05_generate_explanations.py --provider ollama --model llama3.1:8b --limit 6          # pilot: 6 cases x 4 techniques
  python src/05_generate_explanations.py --provider ollama --model llama3.1:8b                    # everything (resumes automatically)

Inputs : processed/recommendation_pool.jsonl, profiles_*.jsonl, catalog, constraint_overrides.csv (optional)
Outputs: processed/case_inputs.jsonl, explanations.jsonl, explanations_summary.csv, generation_report.json,
         prompt_preview.txt (dry run)
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from config import PROC_DIR, GOAL_RULES, as_list, read_table, scrub_text
from explain_facts_05 import (build_facts, facts_to_prompt, render_template, clean_text, word_count, numeric_mismatches,
                           cited_ids, main_ingredients, ingredient_lines)
from llm_client_05 import get_client, QuotaExhausted

PROMPT_VERSION = "v1"
TECH_CODE = {"template": "tpl", "direct_llm": "dir", "grounded_llm": "grd", "grounded_check_llm": "chk"}
CLI_TECH = {"template": "template", "direct": "direct_llm", "grounded": "grounded_llm", "check": "grounded_check_llm"}
LO_WORDS, HI_WORDS = 45, 125

# ------------------------------------------------------------------------------ prompts
DIRECT_SYSTEM = "You are a helpful assistant for a food recommendation app."

GROUNDED_SYSTEM = """You write short explanations for a food recommendation system. You explain to one person why one recipe is being recommended to them.

Rules you must follow:
1. Use ONLY the numbered facts provided. Do not add any other information about the person or the recipe, and do not use outside knowledge about the recipe (for example taste, texture, origin or health effects) unless a fact states it.
2. After every sentence, cite the fact numbers that support it in square brackets, for example [F3] or [F2, F5].
3. Facts marked MUST MENTION describe a mismatch or limitation (for example a goal that is not met). State each one honestly as a trade-off, in plain words. Never hide it or turn it into something positive.
4. Facts described as inferred or not confirmed must be phrased cautiously (for example "your past reviews suggest ..."). Never present them as certain.
5. Never assume anything listed under "not known" (for example allergies, budget, family, cooking skill, medical conditions).
6. Use only numbers that appear in the facts.
7. Write one paragraph of 60 to 110 words in plain text, addressed to the person as "you". No lists, headings or bold text."""

NEUTRAL_TONE = "Tone: neutral and informative. Do not use motivational or persuasive language about health."
FRAMED_TONE = ("Tone: frame the explanation around the person's health goal ({goal}). You may be encouraging, but every claim must "
               "still be supported by the facts, and any trade-off must still be stated honestly.")

CHECK_SYSTEM = """You are a strict fact-checker. You are given numbered FACTS and an EXPLANATION written from them.
Break the explanation into its individual factual claims (one checkable statement about the person or the recipe, including any comparison or number). For each claim decide:
- "supported": directly stated by, or a direct restatement of, one or more facts
- "unsupported": not stated by any fact (outside knowledge, invented preferences, assumptions about the person, or vague praise such as "perfect" or "healthy" that no fact supports)
- "contradicted": conflicts with a fact
Return ONLY a JSON object, with no other text, in exactly this form:
{"claims": [{"claim": "...", "verdict": "supported", "fact_ids": ["F3"]}]}"""

REVISE_SYSTEM = GROUNDED_SYSTEM + "\n\nYou are revising a draft that contains problems. Fix every listed problem and keep everything that was correct."


def msgs(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ------------------------------------------------------------------------------ case selection
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_overrides(path) -> dict:
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    bad = set(d["true_status"]) - {"ok", "violates", "ambiguous"}
    if bad:
        sys.exit(f"{path}: true_status must be ok / violates / ambiguous, found {sorted(bad)}")
    return {(int(r.recipe_id), r.constraint): r.true_status for r in d.itertuples()}


def apply_overrides(rec: dict, ov: dict) -> None:
    for ck in rec["facts"]["constraint_checks"]:
        s = ov.get((rec["recipe_id"], ck["constraint"]))
        if s == "ok":
            ck["rule_passes"], ck["label_status"] = True, "manual"
        elif s == "violates":
            ck["rule_passes"], ck["label_status"] = False, "manual"
        elif s == "ambiguous":
            ck["label_status"] = "ambiguous"


def check_state(rec: dict) -> str:
    cks = rec["facts"]["constraint_checks"]
    if any(c["label_status"] == "ambiguous" for c in cks):
        return "ambiguous"
    return "violates" if any(not c["rule_passes"] for c in cks) else "ok"


def select_cases(pool, ov, allow_ambiguous):
    by = defaultdict(list)
    for r in pool:
        apply_overrides(r, ov)
        by[r["profile_id"]].append(r)
    cases, notes = [], []
    for pid, recs in sorted(by.items()):
        chosen = sorted((r for r in recs if r["selected_for_case"]), key=lambda r: r.get("case_slot") or 9)
        final = []
        for r in chosen:
            st = check_state(r)
            if st == "ok" or (st == "ambiguous" and allow_ambiguous):
                r["label_uncertain"] = st == "ambiguous"
                final.append(r)
                continue
            repl = next((c for c in sorted(recs, key=lambda x: x["rank_in_role"]) if c["role"] == "strong" and not c["selected_for_case"]
                         and not c["planned_diagnostic"] and check_state(c) == "ok" and c not in final), None)
            why = f"recipe {r['recipe_id']} ({r['role']}) label is {st}"
            if repl:
                repl["selected_for_case"], repl["case_slot"], repl["replaced_from"] = True, r["case_slot"], f"{r['recipe_id']}:{r['role']}:{st}"
                repl["label_uncertain"] = False
                final.append(repl)
                notes.append(f"{pid}: {why} -> replaced by strong candidate {repl['recipe_id']}")
            else:
                notes.append(f"{pid}: {why} and no replacement available -> case dropped")
        cases += final
    return cases, notes


def build_case_input(rec, prof, recipe) -> dict:
    f = rec["facts"]
    tz = {t["metric"]: t for t in prof.get("nutrition_tendencies", [])}
    tsorted = [{"metric": ta["metric"], "direction": ta["direction"], "aligned": ta["aligned"],
                "confidence": tz.get(ta["metric"], {}).get("confidence", "medium"), "z": tz.get(ta["metric"], {}).get("z", 0)}
               for ta in sorted(f["tendency_alignment"], key=lambda x: -abs(tz.get(x["metric"], {}).get("z", 0)))]
    hg, tb = prof["health_goal"], prof["time_budget_minutes"]
    view = {
        "n_reviews": (prof.get("history") or {}).get("n_reviews"),
        "constraints": [{"constraint": c["constraint"], "status": c["status"]} for c in prof["dietary_constraints"]],
        "constraint_status": {c["constraint"]: c["status"] for c in prof["dietary_constraints"]},
        "goal": {"value": hg["value"], "status": hg["status"]},
        "time": {"value": tb.get("value"), "status": tb.get("status"), "typical_minutes": tb.get("typical_minutes")},
        "likes": [x["item"] for x in prof["taste"]["likes"][:5]],
        "dislikes": [x["item"] for x in prof["taste"]["dislikes"][:3]],
        "tension": prof.get("tension"), "unknown": prof.get("unknown_fields", []),
    }
    ci = {
        "case_key": f"{rec['profile_id']}-r{rec['case_slot']}", "profile_id": rec["profile_id"], "profile_source": rec["profile_source"],
        "profile_label": prof["label"], "profile_text": scrub_text(prof["profile_text"]), "recipe_id": rec["recipe_id"],
        "title": scrub_text(recipe["title"]), "role": rec["role"], "case_slot": rec["case_slot"],
        "replaced_from": rec.get("replaced_from"), "label_uncertain": bool(rec.get("label_uncertain")),
        "pool_facts": f, "profile_view": view, "tendencies_sorted": tsorted,
        "main_ingredients": main_ingredients(recipe), "ingredient_lines": ingredient_lines(recipe),
        "goal_key": f["goal"]["goal"] if f.get("goal") else None,
    }
    ci["facts"] = build_facts(ci)
    return ci


def order_cases(cis, limit):
    """Whole profiles at a time, alternating historical / synthetic, so a pilot (--limit) covers both sources and both slots."""
    by = defaultdict(list)
    for c in cis:
        by[c["profile_id"]].append(c)
    hist = [sorted(v, key=lambda c: c["case_slot"]) for k, v in sorted(by.items()) if v[0]["profile_source"] == "historical"]
    syn = [sorted(v, key=lambda c: c["case_slot"]) for k, v in sorted(by.items()) if v[0]["profile_source"] == "synthetic"]
    out = [c for pair in itertools.zip_longest(hist, syn) for grp in pair if grp for c in grp]
    return out[:limit] if limit else out


# ------------------------------------------------------------------------------ prompts per technique
def direct_messages(ci):
    n, tm = ci["pool_facts"]["nutrition_per_serving"], ci["pool_facts"]["time"]["duration_minutes"]
    ings = "\n".join(f"- {x}" for x in ci["ingredient_lines"])
    user = (f"PERSON\n{ci['profile_text']}\n\nRECIPE\nTitle: {ci['title']}\nIngredients:\n{ings}\nReady in about {tm:.0f} minutes.\n"
            f"Nutrition per serving: {n['calories']:.0f} kcal, {n['protein_g']:.0f} g protein, {n['total_fat_g']:.0f} g fat "
            f"({n['saturated_fat_g']:.0f} g saturated), {n['carbs_g']:.0f} g carbohydrate ({n['sugars_g']:.0f} g sugar, "
            f"{n['fiber_g']:.0f} g fibre), {n['sodium_mg']:.0f} mg sodium.\n\n"
            'Write a short explanation (60 to 110 words, one paragraph, plain text, no lists or headings) addressed to the person as "you", '
            "explaining why this recipe is being recommended to them.")
    return msgs(DIRECT_SYSTEM, user)


def grounded_messages(ci, framing):
    tone = NEUTRAL_TONE if framing == "neutral" else FRAMED_TONE.format(goal=ci["goal_key"])
    return msgs(GROUNDED_SYSTEM, f"FACTS\n{facts_to_prompt(ci['facts'])}\n\n{tone}\n\nWrite the explanation now.")


def parse_claims(text):
    t = text.replace("```json", "").replace("```", "")
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    cl = d.get("claims") if isinstance(d, dict) else None
    if not isinstance(cl, list):
        return None
    out = [{"claim": str(c.get("claim", "")), "verdict": str(c["verdict"]).lower().strip(), "fact_ids": c.get("fact_ids", [])}
           for c in cl if isinstance(c, dict) and "verdict" in c]
    return out or None


class Gen:
    """Holds the client and settings; every LLM call goes through here."""

    def __init__(self, client, args):
        self.client, self.a = client, args

    def write(self, messages, task, meta):
        r = self.client.chat(messages, task=task, temperature=self.a.temperature, max_tokens=self.a.max_tokens, meta=meta)
        text, w = r.text, word_count(clean_text(r.text))
        if w < LO_WORDS or w > HI_WORDS:                                  # one length-repair attempt
            fix = messages + [{"role": "assistant", "content": text},
                              {"role": "user", "content": f"That answer has {w} words. Rewrite it in 60 to 110 words, following the same rules, "
                                                          "and return only the explanation."}]
            text = self.client.chat(fix, task="shorten", temperature=self.a.temperature, max_tokens=self.a.max_tokens,
                                    meta={**meta, "text": text}).text
        return text

    def audit(self, ci, raw):
        facts, clean = ci["facts"], clean_text(raw)
        claims = None
        for attempt in range(2):
            tail = "" if attempt == 0 else "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object."
            r = self.client.chat(msgs(CHECK_SYSTEM, f"FACTS\n{facts_to_prompt(facts)}\n\nEXPLANATION\n{clean}\n\nReturn the JSON now.{tail}"),
                                 task="check", temperature=0.0, max_tokens=self.a.check_max_tokens, meta={"text": clean, "facts": facts})
            claims = parse_claims(r.text)
            if claims:
                break
        check_ok = bool(claims)
        issues = [{"type": c["verdict"], "detail": c["claim"]} for c in (claims or []) if c["verdict"] in ("unsupported", "contradicted")]
        issues += [{"type": "number_not_in_facts", "detail": n} for n in numeric_mismatches(raw, facts)]
        cited = cited_ids(raw)
        missing = [f["id"] for f in facts if f["must_mention"] and f["id"] not in cited]
        issues += [{"type": "missing_tradeoff", "detail": fid} for fid in missing]
        return {"issues": issues, "claims": claims or [], "check_ok": check_ok, "missing_ids": missing}

    def revise(self, ci, raw, a):
        by = {f["id"]: f["text"] for f in ci["facts"]}
        lines = []
        for i in a["issues"]:
            t = i["type"]
            if t in ("unsupported", "contradicted"):
                lines.append(f'- {t.capitalize()} claim - remove or correct it: "{i["detail"]}"')
            elif t == "number_not_in_facts":
                lines.append(f"- The number {i['detail']} does not appear in the facts; use only numbers from the facts, or remove it.")
            elif t == "missing_tradeoff":
                lines.append(f'- Missing trade-off: you must state {i["detail"]} ("{by[i["detail"]]}") honestly.')
        user = (f"FACTS\n{facts_to_prompt(ci['facts'])}\n\nCURRENT DRAFT\n{raw}\n\nPROBLEMS TO FIX\n" + "\n".join(lines) +
                "\n\nRewrite the explanation so that every problem is fixed. Keep the rules. Return only the corrected explanation, with fact citations.")
        return self.write(msgs(REVISE_SYSTEM, user), "revise", {"text": raw, "facts": ci["facts"], "missing_ids": a["missing_ids"], "title": ci["title"], "case": ci["case_key"]})

    def check_and_revise(self, ci, draft):
        text, trail, first, a = draft, [], None, None
        for it in range(self.a.max_revisions + 1):
            a = self.audit(ci, text)
            trail.append({"iteration": it, "n_claims": len(a["claims"]), "check_ok": a["check_ok"], "issues": a["issues"]})
            first = first or a
            if not a["issues"] or it == self.a.max_revisions:
                break
            text = self.revise(ci, text, a)
        return text, {"n_revisions": len(trail) - 1, "initial_issue_count": len(first["issues"]), "final_issue_count": len(a["issues"]),
                      "check_failed": not all(t["check_ok"] for t in trail), "claims_initial": first["claims"], "claims_final": a["claims"], "trail": trail}


# ------------------------------------------------------------------------------ main loop
def new_record(ci, tech, framing, key, gen_info):
    return {"key": key, "case_key": ci["case_key"], "profile_id": ci["profile_id"], "profile_source": ci["profile_source"],
            "recipe_id": ci["recipe_id"], "case_slot": ci["case_slot"], "role": ci["role"], "technique": tech, "framing": framing,
            "prompt_version": PROMPT_VERSION, "provider": gen_info[0], "model": gen_info[1],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mock", help="groq | gemini | openrouter | nvidia | cerebras | mistral | huggingface | custom | mock")
    ap.add_argument("--model")
    ap.add_argument("--extra", help='JSON of provider-specific request fields, e.g. \'{"reasoning_effort": "low"}\'')
    ap.add_argument("--rpm", type=int, help="override requests/minute")
    ap.add_argument("--tpm", type=int, help="override tokens/minute")
    ap.add_argument("--techniques", default="template,direct,grounded,check")
    ap.add_argument("--limit", type=int, help="only the first N cases (whole profiles, alternating sources) - use for a pilot")
    ap.add_argument("--framed-n", type=int, default=20, help="extra goal-framed grounded explanations (0 = none)")
    ap.add_argument("--max-revisions", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1200, help="generation budget (reasoning models spend part of it on thinking)")
    ap.add_argument("--check-max-tokens", type=int, default=1800)
    ap.add_argument("--overrides", default=str(PROC_DIR / "constraint_overrides.csv"))
    ap.add_argument("--allow-ambiguous", action="store_true", help="keep cases whose dietary label is marked ambiguous (flagged label_uncertain)")
    ap.add_argument("--dry-run", action="store_true", help="build facts, template output and prompts only; no API calls")
    ap.add_argument("--fresh", action="store_true", help="ignore and replace existing explanations.jsonl")
    ap.add_argument("--preview-n", type=int, default=3)
    args = ap.parse_args()
    techs = [CLI_TECH[t.strip()] for t in args.techniques.split(",") if t.strip()]

    # ---- cases
    pool = read_jsonl(PROC_DIR / "recommendation_pool.jsonl")
    profiles = {}
    for name in ("profiles_historical", "profiles_synthetic"):
        for p in read_jsonl(PROC_DIR / f"{name}.jsonl"):
            profiles[p["profile_id"]] = p
    ov = load_overrides(__import__("pathlib").Path(args.overrides))
    recs, notes = select_cases(pool, ov, args.allow_ambiguous)
    print(f"{len(ov)} label overrides applied" if ov else "no constraint_overrides.csv found (labels used as-is)")
    for n in notes:
        print("  NOTE", n)
    ids = {r["recipe_id"] for r in recs}
    cat = read_table("catalog", usecols=["new_recipe_id", "title", "ingredient_names", "ingredients_display"], filter_col="new_recipe_id", filter_values=ids)
    cat = cat.drop_duplicates("new_recipe_id").set_index("new_recipe_id")
    cis = []
    for r in recs:
        row = cat.loc[r["recipe_id"]]
        recipe = {"title": row["title"], "ingredient_names": as_list(row["ingredient_names"]), "ingredients_display": as_list(row["ingredients_display"])}
        cis.append(build_case_input(r, profiles[r["profile_id"]], recipe))
    cis = order_cases(cis, args.limit)
    with open(PROC_DIR / "case_inputs.jsonl", "w", encoding="utf-8") as f:
        for ci in cis:
            f.write(json.dumps(ci, ensure_ascii=False, default=str) + "\n")
    per_profile = defaultdict(int)
    for c in cis:
        per_profile[c["profile_id"]] += 1
    short = [p for p, n in per_profile.items() if n < 2]
    print(f"{len(cis)} cases over {len(per_profile)} profiles ({sum(c['profile_source'] == 'historical' for c in cis)} historical, "
          f"{sum(c['profile_source'] == 'synthetic' for c in cis)} synthetic)" + (f" | profiles with <2 cases: {short}" if short else ""))
    uncertain = [c["case_key"] for c in cis if c["label_uncertain"]]
    if uncertain:
        print("  WARNING cases kept with an ambiguous dietary label:", uncertain)

    framed_keys = set()
    if args.framed_n > 0 and "grounded_llm" in techs:
        elig = sorted((c for c in cis if c["goal_key"] and c["profile_view"]["goal"]["status"] == "stated" and c["case_slot"] == 1), key=lambda c: c["profile_id"])
        framed_keys = {c["case_key"] for c in elig[: args.framed_n]}

    n_llm = sum(t != "template" for t in techs)
    est_calls = len(cis) * (("direct_llm" in techs) + ("grounded_llm" in techs) + (3 if "grounded_check_llm" in techs else 0)) + len(framed_keys)
    print(f"techniques: {', '.join(techs)} | framed extras: {len(framed_keys)} | estimated LLM calls: ~{est_calls}" if n_llm else f"techniques: {', '.join(techs)}")

    # ---- dry run
    if args.dry_run:
        lines = []
        for ci in cis[: args.preview_n]:
            lines += [f"{'=' * 100}\nCASE {ci['case_key']}  role={ci['role']}  profile={ci['profile_label']}\n{'=' * 100}",
                      "--- FACT LIST ---", facts_to_prompt(ci["facts"]), "--- TEMPLATE OUTPUT ---", render_template(ci, f"{ci['case_key']}-tpl"),
                      "--- DIRECT PROMPT ---", direct_messages(ci)[1]["content"], "--- GROUNDED PROMPT (user part) ---", grounded_messages(ci, "neutral")[1]["content"], ""]
        (PROC_DIR / "prompt_preview.txt").write_text("\n".join(lines), encoding="utf-8")
        tw = [word_count(render_template(c, f"{c['case_key']}-tpl")) for c in cis]
        print(f"dry run: wrote processed/prompt_preview.txt | template words min/mean/max = {min(tw)}/{sum(tw) / len(tw):.0f}/{max(tw)}")
        return

    # ---- client + resume
    extra = json.loads(args.extra) if args.extra else None
    client = get_client(args.provider, args.model, **({"rpm": args.rpm, "tpm": args.tpm, "extra": extra} if args.provider != "mock" else {}))
    gen_info = (client.provider, client.model)
    out_path = PROC_DIR / "explanations.jsonl"
    done: dict = {}
    if out_path.exists() and not args.fresh:
        for r in read_jsonl(out_path):
            done[r["key"]] = r
        done = {k: r for k, r in done.items() if not r.get("error")}
        mixed = {(r["provider"], r["model"]) for r in done.values() if r["technique"] != "template"} - {gen_info}
        if mixed:
            sys.exit(f"existing explanations were generated by {sorted(mixed)} but this run uses {gen_info}. "
                     "All LLM techniques must use ONE generator: use the same model, or pass --fresh to start over.")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif out_path.exists():
        out_path.unlink()
    if done:
        print(f"resuming: {len(done)} explanations already done")

    gen = Gen(client, args)
    drafts = {(r["case_key"], r["framing"]): r["text_raw"] for r in done.values() if r["technique"] == "grounded_llm"}
    errors, quota_stop = [], False
    t_start = time.time()
    fout = open(out_path, "a", encoding="utf-8")

    def run_one(ci, tech, framing):
        key = f"{ci['case_key']}-{TECH_CODE[tech]}" + ("-framed" if framing == "goal_framed" else "")
        if key in done:
            return
        rec = new_record(ci, tech, framing, key, ("jinja2", "template") if tech == "template" else gen_info)
        meta = {"title": ci["title"], "facts": ci["facts"], "case": key, "minutes": round(ci["pool_facts"]["time"]["duration_minutes"]),
                "ingredients": ci["main_ingredients"]}
        try:
            if tech == "template":
                raw = render_template(ci, key)
            elif tech == "direct_llm":
                raw = gen.write(direct_messages(ci), "explain_direct", meta)
            elif tech == "grounded_llm":
                raw = gen.write(grounded_messages(ci, framing), "explain_grounded", meta)
                drafts[(ci["case_key"], framing)] = raw
            else:
                draft = drafts.get((ci["case_key"], "neutral"))
                if draft is None:
                    draft = gen.write(grounded_messages(ci, "neutral"), "explain_grounded", meta)
                    drafts[(ci["case_key"], "neutral")] = draft
                raw, audit = gen.check_and_revise(ci, draft)
                rec.update({"audit": audit, "draft_raw": draft, "changed_vs_grounded": clean_text(raw) != clean_text(draft)})
            rec["text_raw"], rec["text"] = raw, clean_text(raw)
            rec["words"] = word_count(rec["text"])
            if tech != "template":
                rec["numeric_mismatches"] = numeric_mismatches(raw, ci["facts"])
            if tech in ("grounded_llm", "grounded_check_llm"):
                rec["cited_fact_ids"] = sorted(cited_ids(raw))
        except QuotaExhausted:
            raise
        except Exception as e:  # noqa: BLE001 - keep going, record the failure, retry on the next run
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            errors.append((key, rec["error"]))
            print(f"  ERROR {key}: {rec['error']}")
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if "error" not in rec:
            done[key] = rec

    try:
        for i, ci in enumerate(cis, 1):
            for tech in techs:
                run_one(ci, tech, "neutral")
            if ci["case_key"] in framed_keys:
                run_one(ci, "grounded_llm", "goal_framed")
            if i % 5 == 0 or i == len(cis):
                print(f"  {i}/{len(cis)} cases | {len(done)} explanations | {time.time() - t_start:.0f}s | {client.stats()['api_calls']} API calls, "
                      f"{client.stats()['cache_hits']} cache hits")
    except QuotaExhausted as e:
        quota_stop = True
        print(f"\nSTOPPED: quota exhausted ({e}).\nProgress is saved. Re-run the same command later (or tomorrow) and it resumes.")
    finally:
        fout.close()

    # ---- summary + report
    recs_out = [r for r in done.values()]
    df = pd.DataFrame([{k: r.get(k) for k in ("key", "case_key", "profile_id", "profile_source", "case_slot", "role", "technique", "framing", "words")}
                       | {"n_numeric_mismatch": len(r.get("numeric_mismatches", [])),
                          "n_revisions": (r.get("audit") or {}).get("n_revisions"), "changed_vs_grounded": r.get("changed_vs_grounded"),
                          "text": r["text"]} for r in recs_out])
    df.to_csv(PROC_DIR / "explanations_summary.csv", index=False)
    rep = {"prompt_version": PROMPT_VERSION, "generator": {"provider": gen_info[0], "model": gen_info[1]}, "settings": {
               "temperature": args.temperature, "max_tokens": args.max_tokens, "max_revisions": args.max_revisions, "framed_n": args.framed_n},
           "cases": len(cis), "explanations_done": len(df), "main_480_done": int((df["framing"] == "neutral").sum()) if len(df) else 0,
           "target_main": len(cis) * len(techs), "errors": errors, "stopped_on_quota": quota_stop, "llm_usage": client.stats()}
    if len(df):
        rep["by_technique"] = {t: {"n": int(len(g)), "words_mean": round(g["words"].mean(), 1), "words_min": int(g["words"].min()), "words_max": int(g["words"].max()),
                                   "share_with_number_not_in_facts": round(float((g["n_numeric_mismatch"] > 0).mean()), 3)}
                               for t, g in df.groupby("technique")}
        chk = df[df["technique"] == "grounded_check_llm"]
        if len(chk):
            rep["check_revise"] = {"share_revised": round(float((chk["n_revisions"] > 0).mean()), 3), "mean_revisions": round(float(chk["n_revisions"].mean()), 2),
                                   "share_text_changed_vs_grounded": round(float(chk["changed_vs_grounded"].mean()), 3),
                                   "check_failed": int(sum((r.get("audit") or {}).get("check_failed", False) for r in recs_out if r["technique"] == "grounded_check_llm")),
                                   "unresolved_after_revisions": int(sum((r.get("audit") or {}).get("final_issue_count", 0) > 0 for r in recs_out if r["technique"] == "grounded_check_llm"))}
    (PROC_DIR / "generation_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: rep[k] for k in ("generator", "cases", "explanations_done", "target_main", "stopped_on_quota", "llm_usage")}, indent=2))
    for t, v in rep.get("by_technique", {}).items():
        print(f"  {t:20s} n={v['n']:4d} words {v['words_min']}-{v['words_max']} (mean {v['words_mean']})  numbers-not-in-facts: {v['share_with_number_not_in_facts']:.0%}")
    if "check_revise" in rep:
        print("  check/revise:", rep["check_revise"])
    if errors:
        print(f"{len(errors)} explanations failed; re-run to retry only those.")


if __name__ == "__main__":
    main()
