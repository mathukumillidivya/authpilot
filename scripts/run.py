"""Run AuthPilot (or the single-call baseline) over a split of cases and save every result.

Usage:
  python scripts/run.py --system authpilot --split dev --run-id dev-v1 [--limit 20] [--model claude-haiku-4-5-20251001]
  python scripts/run.py --system baseline  --split test --run-id test-baseline

Writes results/runs/<run-id>/results.jsonl and config.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "--history" in sys.argv and sys.argv[sys.argv.index("--history") + 1] == "long":
    os.environ["AUTHPILOT_CASES"] = "cases_long.jsonl"     # read by the chart store and the MCP server

from authpilot import graph as G  # noqa: E402


def make_model(name: str):
    if name == "fake":
        sys.path.insert(0, str(ROOT / "tests"))
        from fake_model import FakeModel
        return FakeModel()
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=name, temperature=0, max_tokens=2500, max_retries=6, timeout=120)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["authpilot", "authpilot_noguard", "baseline"], default="authpilot")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--history", choices=["standard", "long"], default="standard")
    args = ap.parse_args()

    split = json.loads((ROOT / "data/split.json").read_text())[args.split]
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in open(ROOT / "data" / os.environ.get("AUTHPILOT_CASES", "cases.jsonl"))}
    todo = [cases[c] for c in split][: args.limit or None]
    out = ROOT / "results/runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    model = make_model(args.model)
    sem = asyncio.Semaphore(args.concurrency)

    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(command=sys.executable, args=["-m", "authpilot.mcp_server"], cwd=str(ROOT),
                                   env={**os.environ})
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        app = G.build(model, tools, use_guards=args.system != "authpilot_noguard")

        async def one(case):
            async with sem:
                t0 = time.time()
                try:
                    if args.system == "baseline":
                        res = await G.baseline(model, case)
                    else:
                        res = await app.ainvoke({"request_text": G.request_form(case["request"]), "retries": 0},
                                                {"recursion_limit": 40})
                    err = None
                except Exception as e:  # keep going; a failed case counts as an error, not a decision
                    res, err = {}, f"{type(e).__name__}: {e}"[:500]
                return {"case_id": case["case_id"], "policy": case["truth"]["policy"], "truth": case["truth"]["decision"],
                        "decision": res.get("decision"), "criteria": res.get("verified"), "policy_id": res.get("policy_id"),
                        "usage": res.get("usage"), "trace": res.get("trace"), "letter": res.get("letter"),
                        "seconds": round(time.time() - t0, 1), "error": err}

        results, t0 = [], time.time()
        with open(out / "results.jsonl", "w") as f:
            for i, coro in enumerate(asyncio.as_completed([one(c) for c in todo])):
                r = await coro
                results.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    ok = sum(x["decision"] == x["truth"] for x in results)
                    print(f"{i + 1}/{len(todo)}  correct so far {ok}  errors {sum(1 for x in results if x['error'])}  "
                          f"{time.time() - t0:.0f}s", flush=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "cases": len(todo)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
