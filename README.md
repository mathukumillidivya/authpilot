# AuthPilot

**Prior authorization review by a team of AI agents, with every finding quoted from the patient record and checked by code.**

**Live demo:** https://authpilot-app.onrender.com, a utilization review workspace over all 300 recorded agent runs. It has five parts:

**Reproduce:** `python reproduce.py` re-scores every saved run and checks each published number (no API key). Full steps, the frozen commit and the data are in [REPRODUCE.md](REPRODUCE.md). A 40-second demo video is on the site's Reproduce page.

- **Review queue**, with sign-off for denials.
- **Case view**, laid out like a reviewer's desk: the faxed request, the patient record with every quoted passage highlighted, the policy checklist the specialists fill in, and the determination letter with its stamp. Replays the recorded run step by step, with the agent trace and token use.
- **Run monitoring**, covering latency, cost and model calls.
- **Evaluation and policy library.**
- **Patient simulator**, which builds a new case for the real agents to review live.

Before an insurer pays for a CPAP machine, home oxygen, a defibrillator or bariatric surgery, a reviewer checks the
request against the coverage policy and the patient's chart. This usually takes days, and wrong denials are common.
AuthPilot reads the request, picks the matching Medicare policy, pulls the chart over MCP from a FHIR store, and
gives each coverage rule to its own specialist agent. Code then checks every quote and every date window before
deciding to approve, deny or ask for records. Denials are drafted for a human to sign, not sent automatically.

Built with LangGraph, the Model Context Protocol, FHIR-style records, FastAPI and Claude Haiku 4.5.

## Result

200 held-out cases, run once with the system frozen at commit `ba8fc5b`. None of these cases were looked at during development.

| | Correct | Wrongful denials | Wrongful approvals | Denials without verified evidence | Cost / request | Median time |
|---|---|---|---|---|---|---|
| Single AI call | 151/200 (75.5%) | 8 | 25 | 1 | $0.007 | 6.7s |
| Single call + the same code checks | 166/200 (83.0%) | 22 | 7 | 0 | $0.008 | 6.2s |
| **AuthPilot** | **188/200 (94.0%)** | **5** | **7** | **0** | $0.030 | 8.9s |

All three use the same model, the same policies and the same output rules. The only difference is how the work is organised.

- **Versus a single call: +18.5 points** (paired bootstrap, 95% interval +12.5 to +24.5). AuthPilot got 41 cases right that the single call got wrong, and 4 the other way.
- **Versus a single call with the same code checks: +11.0 points** (+5.5 to +16.5). The code checks alone fix wrongful approvals but cause many more wrongful denials (22). Splitting the work across specialists is what brings both kinds of error down.
- **Asking for missing records:** AuthPilot asked for records correctly in 40 of 42 cases, against 28 of 42 for both single-call systems.
- **Key evidence only in free-text notes:** 45/47 correct, against 30/47 for a single call.
- **Citations:** 1,257 quotes, and 99.8% matched the record word for word under code verification.
- **Development comparison:** on the 100 development cases AuthPilot scored 92%, so the held-out result (94%) does not suggest the design was tuned to the cases it was built on.

## How it works

```
request ─▶ intake ─▶ policy ─▶ record ─▶ ┌ specialist (rule 1) ┐ ─▶ verify ─▶ investigator? ─▶ letter
          (AI)      (code,     (MCP)     ├ specialist (rule 2) ┤    (code)    (AI + tools)      (AI)
                    then AI)             └ specialist (rule n) ┘
                                          AI, in parallel (LangGraph Send)
```

1. **Intake**: an AI agent reads the free-text request into a structured form.
2. **Policy**: code routes by procedure code first, and falls back to the model only when no code matches.
3. **Record**: the chart is fetched over MCP (`authpilot/mcp_server.py`, a FastMCP server over a FHIR-like store). Every entry is dated relative to the order.
4. **Specialists**: one agent per coverage rule, run in parallel. Each returns `met`, `not_met`, `missing` or `not_applicable`, with exact quotes from the record.
5. **Verify** (code, no model):
   - every quote must appear in the cited record, and numbers and dates in a quote must match the record;
   - `not_applicable` is accepted only for rules the policy marks as conditional;
   - date windows ("within 30 days", "at least 90 days") are computed by code. If a specialist got one wrong, it is sent back once with the correct day count;
   - covered procedure codes are looked up in the policy.

   The decision comes from code: any rule not met means deny, any rule undocumented means ask for records, otherwise approve.
6. **Investigator**: if a rule is still undocumented, a ReAct agent searches the chart with MCP tools before the system asks the provider for records.
7. **Letter**: drafted only from verified findings.

## The cases

There is no public benchmark for prior authorization, so the cases are constructed. `authpilot/cases.py` generates 300 synthetic patients across five Medicare policies:

- CPAP: LCD L33718
- Home oxygen: NCD 240.2 and LCD L33797
- Bariatric surgery: NCD 100.1
- Defibrillator: NCD 20.4
- Glucose monitor: LCD L33822

Each case is generated truth-first: the status of every rule is chosen, then a chart is written that supports it. Three kinds of traps are mixed in:

- key evidence only in free-text notes;
- values exactly at the threshold;
- a provider summary that claims more than the chart supports.

Every chart also includes unrelated background history. The 300 cases are split into 100 development cases and 200 held-out test cases.

## How the design got here

Five rounds on the development cases. The first multi-agent design was **worse** than a single call.

| Round | Design | AuthPilot | Single call |
|---|---|---|---|
| 1 | Agents search the record; one reviewer judges everything | 69% | 84% |
| 2 | Stricter output formats, records sorted by date | 62% | 88% |
| 3 | One specialist per rule over the whole record; investigator agent | 62% | 68%* |
| 4 | Code checks date windows and sends failures back; three ambiguous case templates rewritten | 73% | 68% |
| 5 | Routing and covered procedure codes decided by code lookup | 92% | 71% |

\*From round 3 the single call uses the same strict output schema and rules, which lowered its score. Earlier numbers are not comparable.

What went wrong, and why each round changed things:

- **Searching lost evidence.** Agents that had to search the record missed entries the single call simply read, so specialists were given the whole record.
- **Routing failed.** The model sometimes failed to pick the right policy, so routing moved to code.
- **Procedure codes were confused.** Specialists mixed up covered and uncovered procedure codes, so that check moved to code.
- **Date arithmetic was wrong.** The model miscounted days between dates, so code computes windows and feeds the correct count back.

The rule throughout: **the model reads and judges; code counts, looks up and decides.**

## What these results do not show

- **The cases are constructed.** Real requests are messier: scanned faxes, missing pages, conflicting notes. The charts here average about 70 entries, while real ones can run to thousands of pages. A long-record variant (`--history long`) exists, but a full run costs about four times as much and was not part of the test.
- **The case design was refined during development.** Three templates were rewritten when review showed their correct answer was ambiguous. This happened before the test set was run, and the test set was run once.
- **Five policies, one model.** Results may differ for other services or models.
- **The remaining 12 errors.** Three oxygen cases marked the Group II condition "not applicable" when it applied. Three glucose monitor cases accepted prediabetes as diabetes. Four misjudged the qualifying test value, three of them right at the threshold. One misjudged a symptom note, and one misread a test date.
- **This is decision support.** A human signs every denial.

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # never commit this

# offline check with a scripted model (no key needed)
python -m pytest -q
python scripts/run.py --system authpilot --split dev --limit 5 --model fake --run-id offline-check

# real runs
python scripts/run.py --system authpilot   --split test --run-id test-authpilot
python scripts/run.py --system baseline    --split test --run-id test-baseline
python scripts/run.py --system single_plus --split test --run-id test-single_plus
python scripts/evaluate.py results/runs/test-baseline results/runs/test-authpilot

# the demo service
uvicorn app.server:app --port 8000      # then open http://localhost:8000
```

The GitHub workflow `run.yml` runs any system on any split and commits the results. Every run's per-case results, traces and token usage are in `results/runs/`.

## Layout

```
authpilot/
  graph.py        LangGraph pipeline, specialists, code checks, single-call baselines
  mcp_server.py   FastMCP server exposing the FHIR record as tools
  fhir.py         record store, quote verification, date helpers
  cases.py        truth-first case generator
policies/         the five coverage policies, with machine-readable checks
data/             cases, long-record cases, dev/test split
scripts/          run.py, evaluate.py, build_site.py
app/              FastAPI live service (SSE streaming) and the demo page
results/          every run, plus dev and test comparisons
tests/            unit tests and a scripted fake model
render.yaml       deployment: agent service and static page
```

Live demo limits: each visitor gets 15 live runs a day, and there is a cap on total runs per day. The API key stays on the server.

## License

MIT
