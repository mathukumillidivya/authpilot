"""Build the demo page (app/site/index.html) from the template, recorded runs and measured results.

Usage: python scripts/build_site.py [--api https://<live service>] [--run results/runs/<recorded run>]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.server import CHOICES, EXAMPLES, TRAPS, public  # noqa: E402
from authpilot import graph as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="")
    ap.add_argument("--run", default="results/runs/dev-authpilot-standard-v5")
    args = ap.parse_args()
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in open(ROOT / "data/cases.jsonl")}
    runs = {json.loads(l)["case_id"]: json.loads(l) for l in open(ROOT / args.run / "results.jsonl")}
    examples = []
    for cid in EXAMPLES:
        r = runs[cid]
        examples.append({**public(cases[cid]), "recorded": {"trace": r["trace"], "criteria": r["criteria"], "decision": r["decision"],
                                                            "letter": r["letter"], "seconds": r["seconds"]},
                         "answer_key": cases[cid]["truth"]})
    results = {}
    for name in ("test-comparison.json", "dev-comparison.json"):
        f = ROOT / "results" / name
        if f.exists():
            results[name.split("-")[0]] = json.loads(f.read_text())
    data = {"api": args.api, "examples": examples, "traps": TRAPS,
            "choices": {p: {k: {"options": v, "text": next(c["text"] for c in G.POLICIES[p]["criteria"] if c["id"] == k)}
                            for k, v in ch.items()} for p, ch in CHOICES.items()},
            "policies": {p: G.POLICIES[p]["title"] for p in CHOICES}, "results": results}
    html = (ROOT / "app/template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = ROOT / "app/site/index.html"
    out.write_text(html)
    print(f"{out}: {out.stat().st_size / 1e3:.0f} kB, {len(examples)} recorded examples, results: {list(results)}")


if __name__ == "__main__":
    main()
