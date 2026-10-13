"""AuthPilot live service: runs the real LangGraph agents on a case and streams every step to the browser.

    uvicorn app.server:app --port 8000

Endpoints
  GET  /                    the demo page
  GET  /api/catalog         example cases and what a visitor can change when building a case
  POST /api/case            build a case to order (which criteria hold, which traps) and return it
  POST /api/run             run the agents on a case; Server-Sent Events, one per step
  GET  /health              liveness

Limits (a public page must not become a free model endpoint): runs per visitor per day, runs per day in
total, and runs at the same time. The API key stays on the server.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from authpilot import fhir, graph as G  # noqa: E402
from authpilot.cases import CAN_FAIL, MET, MISSING, NA, NOT_MET, POLICIES as CASE_POLICIES, custom_case  # noqa: E402

MODEL = os.environ.get("AUTHPILOT_MODEL", "claude-haiku-4-5-20251001")
PER_VISITOR = int(os.environ.get("RUNS_PER_VISITOR", 15))
PER_DAY = int(os.environ.get("RUNS_PER_DAY", 200))
AT_ONCE = asyncio.Semaphore(int(os.environ.get("RUNS_AT_ONCE", 3)))

CASES = {json.loads(l)["case_id"]: json.loads(l) for l in open(ROOT / "data/cases.jsonl")}
DEV = set(json.loads((ROOT / "data/split.json").read_text())["dev"])
EXAMPLES = ["case-0181", "case-0246", "case-0123", "case-0002", "case-0188", "case-0061"]  # development cases only
CUSTOM: dict[str, dict] = {}
USAGE = defaultdict(int)

CHOICES = {  # what a visitor may set for each criterion when building a case
    "cpap": {"cpap_test": {MET: "AHI qualifies", NOT_MET: "AHI below 5", MISSING: "Sleep study report missing"},
             "cpap_eval": {MET: "Evaluated before the test", NOT_MET: "Evaluated only after the test", MISSING: "No evaluation"},
             "cpap_symptoms": {NA: "AHI 15 or more (not needed)", MET: "Symptoms documented", NOT_MET: "Denies symptoms", MISSING: "Not documented"},
             "cpap_education": {MET: "Supplier taught the patient", MISSING: "No supplier note"}},
    "oxygen": {"o2_test": {MET: "SpO2 qualifies", NOT_MET: "SpO2 normal", MISSING: "Result pending"},
               "o2_group2": {NA: "Group I value (not needed)", MET: "Pulmonary hypertension / edema", NOT_MET: "None documented"},
               "o2_timing": {MET: "Test within 30 days", NOT_MET: "Test too old"},
               "o2_tester": {MET: "Hospital lab", NOT_MET: "Done by the equipment supplier"}},
    "bariatric": {"bar_bmi": {MET: "BMI 35 or more", NOT_MET: "BMI just under 35"},
                  "bar_comorbid": {MET: "Diabetes, hypertension or apnea", NOT_MET: "No co-morbidity"},
                  "bar_prior": {MET: "Failed supervised program", MISSING: "Not documented"},
                  "bar_procedure": {MET: "Covered procedure", NOT_MET: "Gastric balloon or open sleeve"}},
    "icd": {"icd_indication": {MET: "LVEF 35% or less", NOT_MET: "Latest LVEF above 35%"},
            "icd_gdmt": {NA: "Ischemic (not needed)", MET: "Over 3 months of therapy", NOT_MET: "Therapy too recent"},
            "icd_sdm": {MET: "Shared decision visit", MISSING: "Not documented"},
            "icd_exclusions": {MET: "No recent MI or stent", NOT_MET: "Recent MI, stent or bypass"}},
    "cgm": {"cgm_dx": {MET: "Diabetes", NOT_MET: "Prediabetes only"},
            "cgm_need": {MET: "Insulin or severe lows", NOT_MET: "Metformin only"},
            "cgm_visit": {MET: "Visit in last 6 months", NOT_MET: "Last visit too long ago"}},
}
TRAPS = {"note_only": "Hide the key evidence in free-text notes", "boundary": "Put values right at the threshold",
         "claim": "Provider's summary overstates the case"}

TOOLS = None
APP_GRAPH = None


def make_model():
    if MODEL == "fake":
        sys.path.insert(0, str(ROOT / "tests"))
        from fake_model import FakeModel
        return FakeModel()
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=MODEL, temperature=0, max_tokens=2500, max_retries=4, timeout=90)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to the FHIR MCP server in-process for the life of the service."""
    global TOOLS, APP_GRAPH
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp.shared.memory import create_connected_server_and_client_session

    from authpilot.mcp_server import mcp
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        TOOLS = await load_mcp_tools(session)
        APP_GRAPH = G.build(make_model(), TOOLS)
        yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["content-type"])


def public(case: dict) -> dict:
    """What the page may show. The answer key stays here until the run is over."""
    return {"case_id": case["case_id"], "request": case["request"], "form": G.request_form(case["request"]),
            "chart": [fhir.line(r) for r in sorted(case["chart"], key=fhir.when, reverse=True)],
            "policy": case["truth"]["policy"], "built": case["case_id"].startswith("custom")}


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.get("/api/catalog")
def catalog():
    return {"examples": [public(CASES[c]) for c in EXAMPLES if c in DEV],
            "choices": {p: {k: {"options": v, "text": next(c["text"] for c in G.POLICIES[p]["criteria"] if c["id"] == k)}
                            for k, v in ch.items()} for p, ch in CHOICES.items()},
            "policies": {p: G.POLICIES[p]["title"] for p in CHOICES}, "traps": TRAPS,
            "limits": {"per_visitor": PER_VISITOR, "per_day": PER_DAY}}


@app.post("/api/case")
async def build_case(request: Request):
    body = await request.json()
    policy = body.get("policy")
    if policy not in CHOICES:
        raise HTTPException(400, "unknown policy")
    statuses = {}
    for k, v in (body.get("statuses") or {}).items():
        if k not in CHOICES[policy] or v not in CHOICES[policy][k]:
            raise HTTPException(400, f"bad choice for {k}")
        statuses[k] = v
    traps = {t for t in body.get("traps") or [] if t in TRAPS}
    case = custom_case(policy, statuses, traps, random.randint(1, 10**6))
    if len(CUSTOM) > 500:
        CUSTOM.clear()
    CUSTOM[case["case_id"]] = case
    fhir.register(case)
    return public(case)


def limited(ip: str) -> str | None:
    day = date.today().isoformat()
    if USAGE[day] >= PER_DAY:
        return "The live agents have reached today's limit. Recorded runs still play."
    if USAGE[(day, ip)] >= PER_VISITOR:
        return "You have reached today's limit for live runs. Recorded runs still play."
    USAGE[day] += 1
    USAGE[(day, ip)] += 1
    return None


@app.post("/api/run")
async def run(request: Request):
    body = await request.json()
    case = CASES.get(body.get("case_id")) if body.get("case_id") in DEV else CUSTOM.get(body.get("case_id"))
    if not case:
        raise HTTPException(404, "unknown case")
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    problem = limited(ip)
    if problem:
        return JSONResponse({"error": problem}, status_code=429)

    async def events():
        t0 = time.time()
        yield f"data: {json.dumps({'type': 'start', 'case_id': case['case_id']})}\n\n"
        async with AT_ONCE:
            final = {}
            try:
                async for chunk in APP_GRAPH.astream({"request_text": G.request_form(case["request"]), "retries": 0},
                                                     {"recursion_limit": 40}, stream_mode="updates"):
                    for node, update in chunk.items():
                        for e in (update or {}).get("trace") or []:
                            yield f"data: {json.dumps({'type': 'step', 'elapsed': round(time.time() - t0, 1), **e}, default=str)}\n\n"
                        for k in ("verified", "decision", "letter", "usage", "policy_id"):
                            if update and k in update:
                                final[k] = update[k]
                cost = sum(v.get("input", 0) * 1e-6 + v.get("output", 0) * 5e-6 + v.get("cache_read", 0) * 1e-7
                           + v.get("cache_write", 0) * 1.25e-6 for v in (final.get("usage") or {}).values())
                done = {"type": "done", "decision": final.get("decision"), "criteria": final.get("verified"),
                        "letter": final.get("letter"), "seconds": round(time.time() - t0, 1), "cost": round(cost, 4),
                        "answer_key": case["truth"]}
                yield f"data: {json.dumps(done, default=str)}\n\n"
            except Exception as e:  # the page shows the failure rather than hanging
                yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {str(e)[:200]}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def page():
    return FileResponse(ROOT / "app/site/index.html")


@app.get("/{name}")
def static(name: str):
    f = ROOT / "app/site" / name
    if not f.is_file() or f.suffix not in (".js", ".json", ".css", ".svg"):
        raise HTTPException(404)
    return FileResponse(f)
