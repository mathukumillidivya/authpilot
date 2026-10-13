# Reproduce these results

## Check the published numbers (no API key, a few seconds)

```bash
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` re-scores every saved run with the same scoring code and compares each number with the
committed results. It prints `MATCH` or `DIFFERENT` per file.

## What each part is

| | Where |
|---|---|
| System frozen before the held-out test | commit `ba8fc5b` |
| Dataset | 300 synthetic patients with known answers, generated truth-first from five Medicare policies by `authpilot/cases.py`: `data/cases.jsonl`, split in `data/split.json`. No real patient data. |
| Coverage policies | `policies/` (Medicare NCDs and LCDs, with sources) |
| Raw model outputs | `results/runs/test-<system>/results.jsonl` (decision, rule statuses, quotes, letter, trace, tokens) |
| Scoring code | `scripts/evaluate.py` (accuracy, wrongful denials and approvals, quote verification, paired bootstrap) |
| Published results | `results/test-comparison.json`, `results/dev-comparison.json` |

## Re-run the agents from scratch

Needs an Anthropic API key. The held-out test cost about $9 for all three systems.

```bash
git checkout ba8fc5b
export ANTHROPIC_API_KEY=...      # never commit this
python scripts/run.py --system authpilot   --split test --run-id my-authpilot
python scripts/run.py --system baseline    --split test --run-id my-baseline
python scripts/run.py --system single_plus --split test --run-id my-single_plus
python scripts/evaluate.py results/runs/my-baseline results/runs/my-single_plus results/runs/my-authpilot
```

A fresh run lands close to, not exactly on, the published figures. The GitHub workflow `run.yml` does the
same in Actions.
