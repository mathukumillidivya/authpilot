"""Build the AuthPilot workspace (app/site) from the template, every recorded agent run and the measured results.

Usage: python scripts/build_site.py [--api https://<live service>]

Writes
  app/site/index.html            the app, with the queue and results inlined
  app/site/data/cases/<n>.json   one file per request: the request, the FHIR record, the recorded run, the answer key
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.server import CHOICES, EXAMPLES, TRAPS, public  # noqa: E402
from authpilot import graph as G  # noqa: E402
from evaluate import cost  # noqa: E402

RUNS = {"dev": "results/runs/dev-authpilot-standard-v5", "test": "results/runs/test-authpilot"}


def policy_info(p: dict) -> dict:
    text = p["text"]
    urls = re.findall(r"https?://\S+?(?=[\s)]|$)", text)
    src = re.search(r"Sources?:\s*(.+?)(?:\n\n|https?://)", text, re.S)
    intro = [b.strip() for b in text.split("\n\n") if b.strip() and not b.startswith("#") and not b.lstrip().startswith("-")
             and not b.startswith("Source") and not b.startswith("<!--")]
    return {"title": p["title"], "codes": p["route_codes"], "url": urls[0].rstrip(".,;") if urls else None,
            "source": re.sub(r"\s+", " ", src.group(1)).strip(" ,;") if src else "",
            "intro": re.sub(r"\s+", " ", intro[0]) if intro else "",
            "criteria": [{"id": c["id"], "text": c["text"], "conditional": bool(c.get("conditional")), "check": c.get("check") or {}}
                         for c in p["criteria"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="")
    args = ap.parse_args()
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in open(ROOT / "data/cases.jsonl")}
    out_dir = ROOT / "app/site/data/cases"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    queue = []
    for split, run in RUNS.items():
        lc_file = ROOT / run / "letter_check.jsonl"
        checks = {json.loads(l)["case_id"]: json.loads(l) for l in open(lc_file)} if lc_file.exists() else {}
        load = lambda f: {json.loads(l)["case_id"]: json.loads(l) for l in open(f)} if f.exists() else {}  # noqa: E731
        v2_letters, v2_checks = load(ROOT / run / "letters_v2.jsonl"), load(ROOT / run / "letter_check_v2.jsonl")
        for line in open(ROOT / run / "results.jsonl"):
            r = json.loads(line)
            c = cases[r["case_id"]]
            n = r["case_id"][5:]
            cites = sum(1 for v in (r["criteria"] or {}).values() for q in v.get("citations", []) if q.get("verified"))
            usage = r.get("usage") or {}
            calls = sum(v.get("calls", 0) for v in usage.values())
            toks = sum(v.get("input", 0) + v.get("output", 0) + v.get("cache_read", 0) for v in usage.values())
            started = (r["trace"][-1]["t"] - r["seconds"]) if r.get("trace") else None
            queue.append([n, c["request"]["request_id"], c["request"]["patient_name"], c["request"]["service"], c["truth"]["policy"],
                          c["request"]["order_date"], r["decision"], r["truth"], cites, r["seconds"], round(cost(usage), 4), split,
                          calls, toks, round(started, 2) if started else None])
            rec = {"trace": r["trace"], "criteria": r["criteria"], "decision": r["decision"], "letter": r["letter"],
                   "seconds": r["seconds"], "cost": round(cost(usage), 4), "usage": usage}
            if r["case_id"] in v2_letters:
                L2, k2 = v2_letters[r["case_id"]], v2_checks.get(r["case_id"], {})
                rec["letter_v2"] = {"letter": L2["letter"], "gate": L2["gate"],
                                    "check": {"problems": k2.get("problems", []), **({"verdict": k2["judge"]["verdict"], "unsupported": k2["judge"]["unsupported"],
                                                                                    "contradictions": k2["judge"]["contradictions"]} if "judge" in k2 else {})}}
            if r["case_id"] in checks:
                k = checks[r["case_id"]]
                rec["letter_check"] = {"problems": k["problems"], **({"verdict": k["judge"]["verdict"], "unsupported": k["judge"]["unsupported"],
                                                                     "contradictions": k["judge"]["contradictions"]} if "judge" in k else {})}
            (out_dir / f"{n}.json").write_text(json.dumps({**public(c), "split": split, "recorded": rec, "answer_key": c["truth"]},
                                                          separators=(",", ":")))
    queue.sort(key=lambda q: q[0])

    results = {}
    for name in ("test-comparison.json", "dev-comparison.json"):
        f = ROOT / "results" / name
        if f.exists():
            results[name.split("-")[0]] = json.loads(f.read_text())
    lc = ROOT / RUNS["test"] / "letter_check.json"
    if lc.exists():
        results["letters"] = json.loads(lc.read_text())
    lc2 = ROOT / RUNS["test"] / "letter_check_v2.json"
    if lc2.exists():
        results["letters_v2"] = json.loads(lc2.read_text())
        gates = [json.loads(l)["gate"] for l in open(ROOT / RUNS["test"] / "letters_v2.jsonl")]
        results["letters_v2"]["gate_blocked"] = {"count": sum(bool(g) for g in gates), "of": len(gates)}
    per_case = {}
    for s in ("baseline", "single_plus", "authpilot"):
        f = ROOT / "results/runs" / f"test-{s}" / "results.jsonl"
        if f.exists():
            rows = sorted((json.loads(l) for l in open(f)), key=lambda r: r["case_id"])
            per_case[s] = [[r["case_id"][5:], r["policy"], r["truth"], r["decision"]] for r in rows]
    windows = {p: {c["id"]: c["check"] for c in G.POLICIES[p]["criteria"]
                   if any(k in (c.get("check") or {}) for k in ("within_days", "at_least_days"))} for p in CHOICES}
    data = {"api": args.api, "featured": [c[5:] for c in EXAMPLES], "queue": queue, "windows": windows, "cases": per_case,
            "traps": TRAPS, "results": results,
            "choices": {p: {k: {"options": v, "text": next(c["text"] for c in G.POLICIES[p]["criteria"] if c["id"] == k)}
                            for k, v in ch.items()} for p, ch in CHOICES.items()},
            "policies": {p: policy_info(G.POLICIES[p]) for p in CHOICES}}
    html = (ROOT / "app/template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = ROOT / "app/site/index.html"
    out.write_text(html)
    size = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"{out}: {out.stat().st_size / 1e3:.0f} kB; {len(queue)} case files, {size / 1e6:.1f} MB; results: {list(results)}")


if __name__ == "__main__":
    main()
