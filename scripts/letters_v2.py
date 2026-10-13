"""Regenerate the held-out determination letters with version 2 (authpilot/letter.py) and validate them.

The decisions and rule statuses are the recorded ones from the frozen test run; only the letter step changes, so
the comparison with the original letters is letter against letter on the same findings.

Usage:
  python scripts/letters_v2.py results/runs/test-authpilot --budget 0.5     # needs ANTHROPIC_API_KEY
  python scripts/letters_v2.py results/runs/test-authpilot --provider mock  # offline
Output: <run>/letters_v2.jsonl (letter, letter date, gate problems, usage). Score it with
  python scripts/letter_check.py <run> --letters letters_v2.jsonl [--judge]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from authpilot import graph as G  # noqa: E402
from authpilot.letter import REASONS_PROMPT, TOOL, assemble, parse_reasons, reasons_input, validate  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
PRICE = {"input": 1.0 / 1e6, "output": 5.0 / 1e6, "cache_read": 0.1 / 1e6, "cache_write": 1.25 / 1e6}
LETTER_DATE = "2025-08-01"   # fixed, so the regenerated letters are reproducible
TEMPLATE = (ROOT / "app/template.html").read_text()
LABELS = dict(re.findall(r'(\w+): "([^"]+)"', TEMPLATE[TEMPLATE.index("const LABEL = {"):TEMPLATE.index("const STATUS")]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "mock"])
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--out", default="letters_v2.jsonl")
    ap.add_argument("--reuse", default="", help="reuse the model's reason lines stored in this earlier letters file")
    a = ap.parse_args()
    run = Path(a.run)
    out_path = run / a.out
    reuse = {json.loads(l)["case_id"]: json.loads(l) for l in open(run / a.reuse)} if a.reuse else {}
    done = {json.loads(l)["case_id"]: json.loads(l) for l in open(out_path)} if out_path.exists() else {}
    client = None
    if a.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=5)
    spent, rows = 0.0, []
    for line in open(run / "results.jsonl"):
        r = json.loads(line)
        if not r.get("decision"):
            continue
        if r["case_id"] in done:
            rows.append(done[r["case_id"]])
            continue
        if spent >= a.budget:
            print("budget reached")
            break
        case = json.loads((ROOT / "app/site/data/cases" / f"{r['case_id'][5:]}.json").read_text())
        policy = G.POLICIES[r["policy"]]
        crit = r["criteria"] or {}
        usage = {}
        old = reuse.get(r["case_id"], {})
        if old.get("reasons"):
            reasons = parse_reasons({"reasons": [{"rule": k, "text": v} for k, v in old["reasons"].items()]}, policy)
        if old.get("reasons") and len(reasons) == len(crit):
            pass                                  # the stored reason lines, no new model call
        elif client:
            for attempt in range(3):
                try:
                    m = client.messages.create(model=MODEL, max_tokens=900,
                                               system=[{"type": "text", "text": REASONS_PROMPT, "cache_control": {"type": "ephemeral"}}],
                                               tools=[TOOL], tool_choice={"type": "tool", "name": "submit_reasons"},
                                               messages=[{"role": "user", "content": reasons_input(crit, policy)}],
                                               extra_body={"temperature": 0})
                    reasons = parse_reasons(next(b.input for b in m.content if b.type == "tool_use"), policy)
                    u = m.usage
                    usage = {"input": u.input_tokens, "output": u.output_tokens,
                             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0, "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
                    break
                except Exception as e:  # noqa: BLE001
                    print("retry", r["case_id"], type(e).__name__, str(e)[:120])
                    time.sleep(5 * (attempt + 1))
            else:
                continue
        else:
            reasons = {k: v.get("reason", "")[:200] for k, v in crit.items()}
        spent += sum(usage.get(k, 0) * PRICE[k] for k in PRICE)
        letter = assemble(case["request"], r["decision"], crit, policy, LABELS, reasons, LETTER_DATE)
        row = {"case_id": r["case_id"], "letter": letter, "letter_date": LETTER_DATE, "reasons": reasons,
               "gate": validate(letter, r["decision"], crit, case, LETTER_DATE, policy), "usage": usage}
        rows.append(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    blocked = sum(bool(x["gate"]) for x in rows)
    print(f"{len(rows)} letters, {blocked} blocked by the gate, spent ${spent:.4f}")


if __name__ == "__main__":
    main()
