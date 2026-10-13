"""Re-score AuthPilot's published runs and check every number against the committed results.

No API key needed; it takes a few seconds. The cases and answer keys are in data/cases.jsonl, each
system's raw output is in results/runs/test-*/results.jsonl, and the scoring code is scripts/evaluate.py.
The system was frozen at commit ba8fc5b before the held-out test was run. To re-run the agents
themselves, see REPRODUCE.md.

Usage: python reproduce.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="reproduce-"))


def run(*args, cwd=None):
    """Run one of the repository's own scripts with this Python, quietly."""
    subprocess.run([sys.executable, *map(str, args)], cwd=cwd or ROOT, check=True, stdout=subprocess.DEVNULL)


def differences(a, b, path=""):
    """Every place two JSON values differ, allowing for rounding in the last decimal places."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one file" for k in set(a) ^ set(b)]
        return out + [d for k in set(a) & set(b) for d in differences(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items against {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= 1e-3 else [f"{path}: {a} against {b}"]
    return [] if a == b else [f"{path}: {str(a)[:40]} against {str(b)[:40]}"]


def check(name, recomputed, published):
    diff = differences(recomputed, published)
    print(f"  {'MATCH    ' if not diff else 'DIFFERENT'} {name}")
    for d in diff[:5]:
        print(f"             {d}")
    return not diff


def rerun_in_place(name, files, *args, cwd=None):
    """For scripts that write next to the published files: run, compare with what was there, then put it back."""
    before = {f: (ROOT / f).read_text() for f in files}
    try:
        run(*args, cwd=cwd)
        return all([check(f"{name}: {f}", json.loads((ROOT / f).read_text()), json.loads(before[f])) for f in files])
    finally:
        for f, text in before.items():
            (ROOT / f).write_text(text)


def main():
    print("AuthPilot: re-scoring the 200 held-out requests with the repository's own scoring code.\n")
    pub = json.loads((ROOT / "results/test-comparison.json").read_text())
    runs = {s: f"results/runs/test-{s}" for s in ("baseline", "single_plus", "authpilot")}
    run("scripts/evaluate.py", *runs.values(), "--out", TMP / "all.json")
    allr = json.loads((TMP / "all.json").read_text())
    ok = all([check(f"{s}: accuracy, wrongful denials and approvals, citations, cost", allr[f"test-{s}"], pub["systems"][s]) for s in runs])
    for other in ("baseline", "single_plus"):
        run("scripts/evaluate.py", runs[other], runs["authpilot"], "--out", TMP / f"{other}.json")
        p = json.loads((TMP / f"{other}.json").read_text())["comparison (second vs first)"]
        ok &= check(f"paired bootstrap, AuthPilot against {other}", p, pub["paired"][f"authpilot_vs_{other}"])
    a, b = allr["test-authpilot"], allr["test-baseline"]
    print(f"\n  Correct   AuthPilot {a['accuracy']:.1%}   single AI call {b['accuracy']:.1%}")
    print(f"  Wrongful denials {a['wrongful_denials']} against {b['wrongful_denials']}; wrongful approvals {a['wrongful_approvals']} against {b['wrongful_approvals']}")
    print("\nEvery number matches the published results." if ok else "\nSome numbers differ; see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
