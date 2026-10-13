"""Score one or more runs against the known answers.

Usage: python scripts/evaluate.py results/runs/<run> [results/runs/<other>] [--out file.json]

Error that matters most: a wrongful denial (denying care the policy covers, or denying when the
right step was to ask for documentation). Second: a wrongful approval (paying for care the policy
does not cover, or approving without the documentation the policy requires).
"""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICE = {"input": 1.0 / 1e6, "output": 5.0 / 1e6}       # Claude Haiku 4.5, USD per token
TRUTH = {json.loads(l)["case_id"]: json.loads(l)["truth"] for l in open(ROOT / "data/cases.jsonl")}


def load(run: str) -> list[dict]:
    return [json.loads(l) for l in open(Path(run) / "results.jsonl")]


def cost(usage: dict | None) -> float:
    return sum(v.get("input", 0) * PRICE["input"] + v.get("output", 0) * PRICE["output"] for v in (usage or {}).values())


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if not r["error"]]
    correct = sum(r["decision"] == r["truth"] for r in rows)
    wrong_deny = [r for r in rows if r["decision"] == "DENY" and r["truth"] != "DENY"]
    wrong_approve = [r for r in rows if r["decision"] == "APPROVE" and r["truth"] != "APPROVE"]
    denials = [r for r in rows if r["decision"] == "DENY"]

    def supported(r):
        return any(v["status"] == "not_met" and any(c["verified"] for c in v.get("citations", []))
                   for v in (r.get("criteria") or {}).values())

    cites = [c for r in ok for v in (r.get("criteria") or {}).values() for c in v.get("citations", [])]
    crit_total = crit_right = 0
    for r in ok:
        truth = TRUTH[r["case_id"]]["statuses"]
        for k, v in (r.get("criteria") or {}).items():
            if k in truth:
                crit_total += 1
                crit_right += v["status"] == truth[k]
    by = lambda key: {k: f"{sum(r['decision'] == r['truth'] for r in g)}/{len(g)}" for k, g in sorted(group(rows, key).items())}
    return {
        "cases": n, "errors": n - len(ok),
        "accuracy": round(correct / n, 4), "correct": correct,
        "wrongful_denials": len(wrong_deny), "wrongful_approvals": len(wrong_approve),
        "denials": len(denials), "denials_without_verified_evidence": sum(not supported(r) for r in denials),
        "criterion_accuracy": round(crit_right / crit_total, 4) if crit_total else None,
        "citations": len(cites), "citations_verified": round(sum(c["verified"] for c in cites) / len(cites), 4) if cites else None,
        "confusion": {f"{t}->{p}": c for (t, p), c in sorted(Counter((r["truth"], str(r["decision"])) for r in rows).items())},
        "by_policy": by(lambda r: r["policy"]),
        "by_truth": by(lambda r: r["truth"]),
        "by_trap": {t: f"{sum(r['decision'] == r['truth'] for r in rows if t in TRUTH[r['case_id']]['traps'])}/"
                       f"{sum(1 for r in rows if t in TRUTH[r['case_id']]['traps'])}" for t in ("note_only", "boundary", "claim")},
        "cost_per_case": round(statistics.mean(cost(r["usage"]) for r in ok), 4) if ok else None,
        "model_calls_per_case": round(statistics.mean(sum(v.get("calls", 0) for v in (r["usage"] or {}).values()) for r in ok), 1) if ok else None,
        "median_seconds": statistics.median(r["seconds"] for r in ok) if ok else None,
    }


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def paired(a: list[dict], b: list[dict], n=10000, seed=3) -> dict:
    A, B = {r["case_id"]: r for r in a}, {r["case_id"]: r for r in b}
    ids = sorted(set(A) & set(B))
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [rng.choice(ids) for _ in ids]
        diffs.append(sum((B[i]["decision"] == B[i]["truth"]) - (A[i]["decision"] == A[i]["truth"]) for i in s) / len(s))
    diffs.sort()
    return {"shared_cases": len(ids), "accuracy_gain": round(statistics.mean(diffs), 4),
            "ci95": [round(diffs[int(0.025 * n)], 4), round(diffs[int(0.975 * n)], 4)],
            "b_right_a_wrong": sum(B[i]["decision"] == B[i]["truth"] and A[i]["decision"] != A[i]["truth"] for i in ids),
            "a_right_b_wrong": sum(A[i]["decision"] == A[i]["truth"] and B[i]["decision"] != B[i]["truth"] for i in ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = {r: load(r) for r in args.runs}
    report = {Path(r).name: summarise(v) for r, v in rows.items()}
    if len(args.runs) == 2:
        report["comparison (second vs first)"] = paired(rows[args.runs[0]], rows[args.runs[1]])
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
