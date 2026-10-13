"""Check the determination letters separately from the rule decisions.

The headline evaluation scores the decision and the rule statuses. A correct decision can still go out
with a bad letter: a wrong date, a placeholder left in, an undocumented rule described as "not met", no
appeal route on a denial, or a sentence the findings do not support. This script checks every letter.

Code checks (no model, run on every letter):
  decision      the letter states the system's decision and does not state a different one
  placeholder   no template placeholders such as "[Patient Name]"
  dates         every full date in the letter appears in the request or the patient record
  rule_labels   a records request does not call an undocumented rule "not met"; a denial names an unmet rule
  next_steps    a denial explains how to appeal; a records request says what to send

Model check (optional, --judge): Claude Haiku 4.5 at temperature 0 lists statements in the letter that the
verified findings do not support (for example a validity period or a policy term the findings never mention).
It reads only what the letter writer was given, so "unsupported" means "not in the findings", not "false".

Usage:
  python scripts/letter_check.py results/runs/test-authpilot                  # code checks, free
  python scripts/letter_check.py results/runs/test-authpilot --judge --budget 0.9
Output: <run>/letter_check.jsonl and <run>/letter_check.json
"""
from __future__ import annotations

import argparse
import sys
import json
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from authpilot import graph as G  # noqa: E402
from authpilot.letter_rules import code_checks  # noqa: E402
MODEL = "claude-haiku-4-5-20251001"
PRICE = {"input": 1.0 / 1e6, "output": 5.0 / 1e6, "cache_read": 0.1 / 1e6, "cache_write": 1.25 / 1e6}

JUDGE = """You check prior-authorization determination letters before they are sent. You get the FINDINGS the \
letter writer was given (the request, the decision made by code, and each coverage rule's status with its \
reason and its verified quotes from the patient record) and the LETTER.

List every statement in the letter that the findings do not support: facts, dates, numbers, document names, \
policy terms (such as how long an approval lasts or what a plan covers), or promises about process. Rephrasing \
a finding is fine. Generic courtesy text is fine. Also say whether anything in the letter contradicts the \
decision or a rule's status (for example calling an undocumented rule "not met", or approving in the text of \
a denial).

Submit with the submit_check tool."""
TOOL = {"name": "submit_check", "description": "Submit the letter check.", "input_schema": {
    "type": "object", "required": ["unsupported", "contradictions", "verdict"],
    "properties": {
        "unsupported": {"type": "array", "items": {"type": "object", "required": ["quote", "why"], "properties": {
            "quote": {"type": "string"}, "why": {"type": "string"}}}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["clean", "minor_issues", "must_fix"]}}}}


def judge_call(client, findings: str, letter: str) -> tuple[dict, dict]:
    r = client.messages.create(model=MODEL, max_tokens=900,
                               system=[{"type": "text", "text": JUDGE, "cache_control": {"type": "ephemeral"}}],
                               tools=[TOOL], tool_choice={"type": "tool", "name": "submit_check"},
                               messages=[{"role": "user", "content": f"<findings>\n{findings}\n</findings>\n\n<letter>\n{letter}\n</letter>"}],
                               extra_body={"temperature": 0})
    u = r.usage
    return next(b.input for b in r.content if b.type == "tool_use"), {
        "input": u.input_tokens, "output": u.output_tokens,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0, "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}


def rate(k: int, n: int) -> dict:
    return {"count": k, "of": n, "rate": round(k / n, 4) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--budget", type=float, default=0.9)
    ap.add_argument("--sample", type=int, default=0, help="judge only this many letters, drawn at random (seed 3)")
    ap.add_argument("--letters", default="", help="score these letters instead (e.g. letters_v2.jsonl); output letter_check_v2.*")
    args = ap.parse_args()
    run = Path(args.run)
    rows_in = [json.loads(l) for l in open(run / "results.jsonl")]
    suffix, alt = "", {}
    if args.letters:
        alt = {json.loads(l)["case_id"]: json.loads(l) for l in open(run / args.letters)}
        suffix = "_" + Path(args.letters).stem.split("_")[-1]
        rows_in = [{**r, "letter": alt[r["case_id"]]["letter"], "letter_date": alt[r["case_id"]].get("letter_date", "")}
                   for r in rows_in if r["case_id"] in alt]
    client = None
    if args.judge:
        import anthropic
        client = anthropic.Anthropic(max_retries=5)
    prev = {}
    if (run / f"letter_check{suffix}.jsonl").exists():
        prev = {json.loads(l)["case_id"]: json.loads(l) for l in open(run / f"letter_check{suffix}.jsonl")}
    out, spent = [], 0.0
    sample = None
    if args.sample:
        import random
        ids = [r["case_id"] for r in rows_in if r.get("decision")]
        sample = set(random.Random(3).sample(ids, min(args.sample, len(ids))))
    for r in rows_in:
        if not r.get("decision"):
            continue
        case = json.loads((ROOT / "app/site/data/cases" / f"{r['case_id'][5:]}.json").read_text())
        statuses = {k: v.get("status") for k, v in (r["criteria"] or {}).items()}
        row = {"case_id": r["case_id"], "decision": r["decision"], "truth": r["truth"], **code_checks(r["letter"] or "", r["decision"], statuses, case, r.get("letter_date", ""),
                                                                                                        tuple(c["text"] for c in G.POLICIES[r["policy"]]["criteria"]) if args.letters else ())}
        old = prev.get(r["case_id"], {})
        if "judge" in old:
            row["judge"], row["usage"] = old["judge"], old.get("usage", {})
        elif client and spent < args.budget and (sample is None or r["case_id"] in sample):
            findings = json.dumps({"request": {k: case["request"][k] for k in ("service", "code", "order_date", "ordering_provider", "clinical_summary")},
                                   "decision": r["decision"],
                                   "rules": {k: {"status": v.get("status"), "reason": v.get("reason"),
                                                 "verified_quotes": [q["quote"] for q in v.get("citations", []) if q.get("verified")]}
                                             for k, v in (r["criteria"] or {}).items()}}, indent=1)
            for attempt in range(3):
                try:
                    row["judge"], row["usage"] = judge_call(client, findings, r["letter"] or "")
                    spent += sum(row["usage"].get(k, 0) * PRICE[k] for k in PRICE)
                    break
                except Exception as e:  # noqa: BLE001
                    print("retry", r["case_id"], type(e).__name__, str(e)[:120])
                    time.sleep(5 * (attempt + 1))
        out.append(row)
    with open(run / f"letter_check{suffix}.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    n = len(out)
    kinds = Counter(p.split(":")[0] for row in out for p in row["problems"])
    judged = [row for row in out if "judge" in row]
    m = {"letters": n, "letters_with_a_code_problem": rate(sum(bool(row["problems"]) for row in out), n),
         "code_problems_by_kind": {k: rate(v, n) for k, v in kinds.most_common()},
         "by_decision": {d: rate(sum(row["decision"] == d and bool(row["problems"]) for row in out), sum(row["decision"] == d for row in out))
                         for d in ("APPROVE", "DENY", "NEED_INFO")},
         "dates_checked": sum(row["dates"] for row in out),
         "examples": [{"case_id": row["case_id"], "problems": row["problems"]} for row in out if row["problems"]][:12]}
    if judged:
        v = Counter(row["judge"]["verdict"] for row in judged)
        m["judge"] = {"model": MODEL, "letters_judged": len(judged),
                      "clean": rate(v["clean"], len(judged)), "minor_issues": rate(v["minor_issues"], len(judged)), "must_fix": rate(v["must_fix"], len(judged)),
                      "letters_with_an_unsupported_statement": rate(sum(bool(row["judge"]["unsupported"]) for row in judged), len(judged)),
                      "letters_with_a_contradiction": rate(sum(bool(row["judge"]["contradictions"]) for row in judged), len(judged)),
                      "cost_usd": round(sum(sum(row.get("usage", {}).get(k, 0) * PRICE[k] for k in PRICE) for row in judged), 4),
                      "examples": [{"case_id": row["case_id"], "unsupported": row["judge"]["unsupported"][:2], "contradictions": row["judge"]["contradictions"][:2]}
                                   for row in judged if row["judge"]["verdict"] == "must_fix"][:10]}
    (run / f"letter_check{suffix}.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
