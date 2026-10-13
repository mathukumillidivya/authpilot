"""AuthPilot: a multi-agent prior-authorization reviewer built with LangGraph.

    intake -> policy -> record -+-> specialist (criterion 1) -+-> verify --(gaps, once)--> investigator -> specialist(gaps)
                                +-> specialist (criterion 2) -+      |
                                +-> ...                       +      +--> letter -> END

Agents (language model):
  intake        reads the provider's request form and extracts the structured fields
  policy        picks the coverage policy that governs the requested service
  specialist    one per coverage criterion, run in parallel; each judges a single criterion against the whole
                record and quotes what it relied on. They share one cached copy of the record.
  investigator  for criteria still undocumented after the first pass, searches the record again through the
                FHIR MCP tools (notes, orders, encounters) before the specialist takes a second look
  letter        drafts the determination letter

Code steps (deterministic):
  record        fetches the record through the MCP server, in date order, each entry marked with how many
                days before the order it falls, so no agent has to do date arithmetic
  verify        checks that every citation exists and says what the agent quoted; a criterion with no verified
                evidence cannot count as met or not met; 'not applicable' is only allowed where the policy says
                so; the decision is computed from the criteria, never taken from a model
"""
from __future__ import annotations

import json
from datetime import date
import operator
import re
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from authpilot import fhir
from authpilot.cases import decide

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "policies"
STATUSES = ("met", "not_met", "missing", "not_applicable")


# ------------------------------------------------------------------------------------------ policies

def load_policies() -> dict:
    out = {}
    for f in sorted(POLICY_DIR.glob("*.md")):
        text = f.read_text()
        crit = []
        for m in re.finditer(r"^- \*\*([a-z0-9_]+)\.\*\*(.+?)(?=^- \*\*|^\S|\Z)", text, re.S | re.M):
            raw = m.group(2)
            check = {k: (int(v) if v.isdigit() else v.split("|")) for k, v in re.findall(r"<!-- check: (\w+)=([\w|]+) -->", raw)}
            body = " ".join(re.sub(r"<!--.*?-->", "", raw).split())
            crit.append({"id": m.group(1), "text": body, "conditional": body.startswith("Applies only"), "check": check})
        shown = re.sub(r"\n?\s*<!--.*?-->", "", text)          # the machine checks are for code, not for the models
        routes = re.search(r"<!-- route_codes: ([\w,]+) -->", text)
        out[f.stem] = {"id": f.stem, "title": text.splitlines()[0].lstrip("# ").strip(), "text": shown, "criteria": crit,
                       "route_codes": routes.group(1).split(",") if routes else []}
    return out


POLICIES = load_policies()


def request_form(req: dict) -> str:
    """The request as a provider would send it: a form, not tidy JSON."""
    return (f"PRIOR AUTHORIZATION REQUEST  {req['request_id']}\n"
            f"Member: {req['patient_name']}   DOB: {req['birth_date']}   Member ID: {req['patient_id']}\n"
            f"Requested service: {req['service']}   Code: {req['code']}\n"
            f"Diagnosis: {req['diagnosis']['text']} ({req['diagnosis']['code']})\n"
            f"Date of order: {req['order_date']}   Ordering provider: {req['ordering_provider']}\n"
            f"Clinical summary from provider: {req['clinical_summary']}\n")


# The same judging rules are given to the single-call baseline and to the specialists, so a comparison
# measures the architecture, not a better prompt.
RULES = """Statuses:
- "met": the record shows the criterion is satisfied.
- "not_met": the record shows something that fails it: a value on the wrong side of a threshold, a dated event
  outside the allowed window, an excluded condition, a non-covered procedure, or an explicit negative finding.
- "missing": the record does not document it either way. The provider will be asked to send documentation.
- "not_applicable": only for a criterion that says "Applies only ..." when that condition does not hold.
Rules:
- When a measurement was repeated, the most recent one before the order applies.
- The provider's clinical summary is a claim, not evidence. Never decide a criterion from it.
- Every "met" or "not_met" must cite at least one record id with a short quote copied from that record,
  including the number, date or words that decide it."""


# ------------------------------------------------------------------------------------------ output schemas

class IntakeOut(BaseModel):
    patient_id: str
    service: str
    code: str
    diagnosis_code: str
    diagnosis: str
    order_date: str = Field(description="YYYY-MM-DD")
    ordering_provider: str
    provider_claims: list[str] = Field(default_factory=list, description="claims the provider makes, as stated")


class PolicyOut(BaseModel):
    policy_id: str
    reason: str


class Citation(BaseModel):
    id: str = Field(description="record id, e.g. obs-004")
    quote: str = Field(description="text copied from that record that decides the criterion")


class CriterionOut(BaseModel):
    criterion_id: str
    status: str = Field(description="met | not_met | missing | not_applicable")
    reason: str
    citations: list[Citation] = Field(default_factory=list)


class InvestigatorOut(BaseModel):
    record_ids: list[str] = Field(default_factory=list)
    note: str = ""


async def structured(model, schema, messages):
    """Ask for output in a fixed shape (via tool calling) and keep the raw message for token counts."""
    out = await model.with_structured_output(schema, include_raw=True).ainvoke(messages)
    if out.get("parsed") is None:
        raise ValueError(f"unreadable {schema.__name__}: {out.get('parsing_error')}")
    return out["parsed"], out["raw"]


def text_of(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


def tokens(node: str, messages: list) -> dict:
    cur = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0}
    for m in messages:
        md = getattr(m, "usage_metadata", None)
        if md and getattr(m, "type", "") == "ai":
            det = md.get("input_token_details") or {}
            cur["cache_read"] += det.get("cache_read", 0)
            cur["cache_write"] += det.get("cache_creation", 0)
            cur["input"] += md.get("input_tokens", 0) - det.get("cache_read", 0) - det.get("cache_creation", 0)
            cur["output"] += md.get("output_tokens", 0)
            cur["calls"] += 1
    return {node: cur}


def add_usage(a: dict, b: dict) -> dict:
    out = {k: dict(v) for k, v in (a or {}).items()}
    for k, v in (b or {}).items():
        cur = out.setdefault(k, {})
        for kk, vv in v.items():
            cur[kk] = cur.get(kk, 0) + vv
    return out


def merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


def ev(node: str, summary: str, **detail) -> list:
    return [{"node": node, "summary": summary, "t": round(time.time(), 2), **detail}]


# ------------------------------------------------------------------------------------------ state

class State(TypedDict, total=False):
    request_text: str
    request: dict
    policy_id: str
    record: str
    review: Annotated[dict, merge]
    extra: dict                  # criterion -> extra record ids found by the investigator
    verified: dict
    decision: str
    letter: str
    retries: int
    focus: list
    feedback: dict
    window_retried: bool
    route: str
    trace: Annotated[list, operator.add]
    usage: Annotated[dict, add_usage]


class SpecialistInput(TypedDict):
    criterion: dict
    policy_id: str
    request: dict
    record: str
    extra: list
    feedback: str


# ------------------------------------------------------------------------------------------ prompts

INTAKE = "You read prior authorization request forms and extract the fields exactly as written."

POLICY = """You route prior authorization requests to the coverage policy that governs the requested service.
Route by the service and code only. Do not judge whether the patient qualifies; that is decided later.
Policies available:
{catalog}
Return the policy id (one of the ids above, or "none" only if no policy covers this kind of service) and a
one-sentence reason."""

SPECIALIST = """You are a utilization review specialist. You judge ONE coverage criterion for a prior authorization
request, using the patient's medical record below.

Policy:
{policy}

{rules}

The record lists every entry in date order with how many days before the order date it falls. Use those day
counts for any time window; do not compute dates yourself.

Order date: {order_date}

Medical record:
{record}"""

INVESTIGATOR = """You are a clinical evidence investigator. A reviewer could not find documentation for some coverage
criteria in a patient's record. Search the record with the tools, especially the notes, orders and encounters,
for anything that bears on them. Patient ID: {patient_id}. Order date: {order_date}.

Criteria:
{criteria}

When done, return the ids of the records that bear on these criteria (empty if there are none) and one sentence."""

LETTER = """You write prior authorization determination letters to the ordering provider. Plain, professional,
short. State the decision, then each criterion with the evidence relied on (record dates and values), and for a
denial or request for information, exactly what is missing or not met and what the provider can send. Do not
invent any clinical fact that is not in the material given. Use the dates, names and ids given; never write a
placeholder such as [Date] or [Name], leave a line out instead. No more than 220 words."""


# ------------------------------------------------------------------------------------------ code checks

def check_review(policy_id: str, patient_id: str, review: dict, use_guards: bool = True, order_date: str | None = None):
    """Verify each judgement's citations against the record and compute the decision.

    Returns (criteria, notes, decision). With guards on, a criterion judged met or not met without a single
    citation that exists and says what was quoted is treated as missing, and 'not applicable' is only
    accepted for criteria the policy marks as conditional."""
    p = POLICIES[policy_id]
    records = fhir.chart(patient_id)
    out, notes = {}, []
    for c in p["criteria"]:
        v = review.get(c["id"]) or {"status": "missing", "reason": "not assessed", "citations": []}
        if use_guards and "covered_codes" in c.get("check", {}):
            v = code_lookup(c, records, order_date, v, notes)
        status = v.get("status") if v.get("status") in STATUSES else "missing"
        cites = []
        for ct in v.get("citations") or []:
            rid, q = str(ct.get("id", "")), str(ct.get("quote", ""))
            cites.append({"id": rid, "quote": q, "verified": rid in records and fhir.quote_found(records[rid], q, order_date)})
        if use_guards:
            if status == "not_applicable" and not c["conditional"]:
                notes.append(f"{c['id']}: marked not applicable but the criterion always applies; treated as missing")
                status = "missing"
            if status in ("met", "not_met") and not any(x["verified"] for x in cites):
                notes.append(f"{c['id']}: '{status}' had no verified citation; treated as missing")
                status = "missing"
        out[c["id"]] = {"status": status, "reason": v.get("reason", ""), "citations": cites, "model_status": v.get("status")}
        if use_guards and status == "met" and c.get("check") and order_date:
            problem = window_problem(c, [records[x["id"]] for x in cites if x["verified"]], order_date)
            if problem:
                out[c["id"]]["window_problem"] = problem
    return out, notes, decide({k: v["status"] for k, v in out.items()})


def code_lookup(criterion: dict, records: dict, order_date: str | None, judged: dict, notes: list) -> dict:
    """Some criteria are lookups, not judgements: is the ordered procedure code on the covered list? Code decides."""
    orders = [r for r in records.values() if r["resourceType"] == "ServiceRequest" and (not order_date or fhir.when(r) == order_date)]
    if not orders:
        return judged
    o = orders[0]
    code = o["code"]["coding"][0]["code"]
    status = "met" if code in criterion["check"]["covered_codes"] else "not_met"
    if judged.get("status") != status:
        notes.append(f"{criterion['id']}: judged {judged.get('status')}, but code {code} is "
                     f"{'on' if status == 'met' else 'not on'} the covered list; set to {status}")
    return {"status": status, "reason": f"Ordered procedure code {code} is {'a covered' if status == 'met' else 'not a covered'} type.",
            "citations": [{"id": o["id"], "quote": f"{o['code']['text']} (code {code})"}]}


def window_problem(criterion: dict, cited: list, order_date: str) -> str | None:
    """Time windows are arithmetic, so code checks them: a 'met' needs cited evidence inside the window."""
    dates = [fhir.days_before(fhir.when(r), order_date) for r in cited]
    dates = [d for d in dates if d is not None and d >= 0]
    if "within_days" in criterion["check"]:
        n = criterion["check"]["within_days"]
        if not any(d <= n for d in dates):
            return (f"the evidence cited is {', '.join(str(d) for d in sorted(dates)) or 'not'} days before the order, "
                    f"but this criterion requires evidence within {n} days")
    if "at_least_days" in criterion["check"]:
        n = criterion["check"]["at_least_days"]
        if not dates or min(dates) < n:
            return (f"the evidence cited starts {min(dates) if dates else 'an unknown number of'} days before the order, "
                    f"but this criterion requires at least {n} days")
    return None


# ------------------------------------------------------------------------------------------ graph

def build(model, tools, max_retries: int = 1, use_guards: bool = True):
    """model: a LangChain chat model. tools: the FHIR MCP tools (LangChain-wrapped)."""
    from langgraph.prebuilt import create_react_agent

    tool_by_name = {t.name: t for t in tools}
    search_tools = [t for t in tools if t.name != "get_record"]
    catalog = "\n".join(f"- {p['id']}: {p['title']}" for p in POLICIES.values())

    async def intake(state: State) -> State:
        parsed, raw = await structured(model, IntakeOut, [SystemMessage(INTAKE), HumanMessage(state["request_text"])])
        req = parsed.model_dump()
        errors = []
        if req["patient_id"] not in fhir.store():
            errors.append(f"unknown member id {req['patient_id']}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", req["order_date"]):
            errors.append("order date not readable")
        if errors:
            raise ValueError("; ".join(errors))
        return {"request": req, "usage": tokens("intake", [raw]),
                "trace": ev("intake", f"Read the request: {req['service']}, ordered {req['order_date']}", fields=req)}

    async def policy(state: State) -> State:
        r = state["request"]
        by_code = [p for p in POLICIES.values() if r["code"].strip().upper() in p["route_codes"]]
        if len(by_code) == 1:
            p = by_code[0]
            return {"policy_id": p["id"], "trace": ev("policy", f"Applied {p['title']} (matched by code {r['code']})",
                                                    criteria=[c["id"] for c in p["criteria"]])}
        parsed, raw = await structured(model, PolicyOut, [SystemMessage(POLICY.format(catalog=catalog)),
                                                          HumanMessage(f"Service: {r['service']}\nCode: {r['code']}\nDiagnosis: {r['diagnosis']}")])
        if parsed.policy_id not in POLICIES:
            raise ValueError(f"no policy matched ({parsed.policy_id})")
        p = POLICIES[parsed.policy_id]
        return {"policy_id": parsed.policy_id, "usage": tokens("policy", [raw]),
                "trace": ev("policy", f"Applied {p['title']}", criteria=[c["id"] for c in p["criteria"]], reason=parsed.reason)}

    async def record(state: State) -> State:
        r = state["request"]
        res = await tool_by_name["get_record"].ainvoke({"patient_id": r["patient_id"], "as_of": r["order_date"]})
        text = res if isinstance(res, str) else "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in res)
        n = text.count("\n") + 1
        return {"record": text, "retries": state.get("retries") or 0,
                "trace": ev("record", f"Fetched the record over MCP: {n} entries, each dated relative to the order")}

    def fan_out(state: State):
        p = POLICIES[state["policy_id"]]
        focus = state.get("focus") or [c["id"] for c in p["criteria"]]
        return [Send("specialist", {"criterion": c, "policy_id": state["policy_id"], "request": state["request"],
                                    "record": state["record"], "extra": (state.get("extra") or {}).get(c["id"], []),
                                    "feedback": (state.get("feedback") or {}).get(c["id"], "")})
                for c in p["criteria"] if c["id"] in focus]

    async def specialist(inp: SpecialistInput) -> State:
        c, r = inp["criterion"], inp["request"]
        p = POLICIES[inp["policy_id"]]
        system = SPECIALIST.format(policy=p["text"], rules=RULES, order_date=r["order_date"], record=inp["record"])
        # The record is the same for every specialist on a case, so it is cached once and re-read cheaply.
        sys_msg = SystemMessage(content=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
        ask = (f"Judge criterion {c['id']} only: {c['text']}\n"
               f"Provider's clinical summary (a claim, not evidence): {r.get('provider_claims')}")
        if inp["extra"]:
            ask += f"\nAn investigator points to these records as relevant: {', '.join(inp['extra'])}. Look at them closely."
        if inp.get("feedback"):
            ask += (f"\nA date check on your previous answer failed: {inp['feedback']}. Re-read the day counts in the "
                    "record and judge again. If no evidence falls inside the window, the criterion is not met.")
        parsed, raw = await structured(model, CriterionOut, [sys_msg, HumanMessage(ask)])
        out = parsed.model_dump()
        out["criterion_id"] = c["id"]
        return {"review": {c["id"]: out}, "usage": tokens("specialist", [raw]),
                "trace": ev("specialist", f"{c['id']}: {out['status']}", criterion=c["id"], status=out["status"],
                            reason=out["reason"], citations=out["citations"])}

    def verify(state: State) -> State:
        r = state["request"]
        out, notes, decision = check_review(state["policy_id"], r["patient_id"], state.get("review") or {}, use_guards, r["order_date"])
        windows = {k: v["window_problem"] for k, v in out.items() if v.get("window_problem")}
        if windows and not state.get("window_retried"):
            return {"verified": out, "focus": list(windows), "feedback": windows, "window_retried": True, "route": "specialist",
                    "trace": ev("verify", f"Date check failed on {len(windows)} criteria; returned to the specialist with the dates",
                                window_problems=windows, overrides=notes)}
        for k, msg in windows.items():                 # still outside the window after a second look
            out[k]["status"] = "not_met"
            notes.append(f"{k}: judged met, but {msg}; treated as not met")
        decision = decide({k: v["status"] for k, v in out.items()})
        gaps = [k for k, v in out.items() if v["status"] == "missing"]
        retry = bool(use_guards and gaps and (state.get("retries") or 0) < max_retries)
        bad = sum(1 for v in out.values() for x in v["citations"] if not x["verified"])
        return {"verified": out, "decision": decision, "focus": gaps if retry else [], "feedback": {},
                "retries": (state.get("retries") or 0) + (1 if retry else 0), "route": "investigator" if retry else "letter",
                "trace": ev("verify", (f"{len(gaps)} criteria undocumented; sending the investigator back into the record"
                                       if retry else f"Decision computed from the criteria: {decision}"),
                            unverified_citations=bad, overrides=notes, decision=decision,
                            statuses={k: v["status"] for k, v in out.items()})}

    async def investigator(state: State) -> State:
        p, r = POLICIES[state["policy_id"]], state["request"]
        crit = "\n".join(f"- {c['id']}: {c['text']}" for c in p["criteria"] if c["id"] in state["focus"])
        agent = create_react_agent(model, search_tools, prompt=INVESTIGATOR.format(patient_id=r["patient_id"],
                                   order_date=r["order_date"], criteria=crit), response_format=InvestigatorOut)
        res = await agent.ainvoke({"messages": [HumanMessage("Search now.")]}, {"recursion_limit": 24})
        found = res.get("structured_response") or InvestigatorOut()
        if isinstance(found, dict):
            found = InvestigatorOut.model_validate(found)
        valid = [i for i in found.record_ids if i in fhir.chart(r["patient_id"])]
        n_tools = sum(1 for m in res["messages"] if getattr(m, "type", "") == "tool")
        calls = [f"{tc['name']}({', '.join(f'{k}={v}' for k, v in tc['args'].items() if k != 'patient_id')})"
                 for m in res["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
        return {"extra": {k: valid for k in state["focus"]}, "usage": tokens("investigator", res["messages"]),
                "trace": ev("investigator", f"Searched the record again ({n_tools} lookups); found {len(valid)} relevant records",
                            tool_calls=calls, found=valid)}

    async def letter(state: State) -> State:
        r = state["request"]
        body = json.dumps({"letter_date": date.today().isoformat(), "member_id": r.get("patient_id"),
                           "ordering_provider": r.get("ordering_provider"),
                           "decision": state["decision"], "service": r["service"], "order_date": r["order_date"],
                           "criteria": {k: {"status": v["status"], "reason": v["reason"],
                                            "evidence": [x["quote"] + f" ({x['id']})" for x in v["citations"] if x["verified"]]}
                                        for k, v in state["verified"].items()}}, indent=1)
        out = await model.ainvoke([SystemMessage(LETTER), HumanMessage(body)])
        return {"letter": text_of(out), "usage": tokens("letter", [out]), "trace": ev("letter", "Drafted the determination letter")}

    g = StateGraph(State)
    for name, fn in [("intake", intake), ("policy", policy), ("record", record), ("specialist", specialist),
                     ("verify", verify), ("investigator", investigator), ("letter", letter)]:
        g.add_node(name, fn)
    g.set_entry_point("intake")
    g.add_edge("intake", "policy")
    g.add_edge("policy", "record")
    g.add_conditional_edges("record", fan_out, ["specialist"])
    g.add_edge("specialist", "verify")
    g.add_conditional_edges("verify", lambda s: fan_out(s) if s.get("route") == "specialist" else s.get("route", "letter"),
                            ["specialist", "investigator", "letter"])
    g.add_conditional_edges("investigator", fan_out, ["specialist"])
    g.add_edge("letter", END)
    return g.compile()


# ------------------------------------------------------------------------------------------ baseline

BASELINE = """You are a utilization review nurse. Decide this prior authorization request using the policy and the
patient's full medical record below.

Policy:
{policy}

{rules}

Judge every criterion in the policy, then give the overall decision: APPROVE if all apply and are met, DENY if any
is not met, NEED_INFO if none is failed but some are undocumented."""


class BaselineOut(BaseModel):
    criteria: list[CriterionOut]
    decision: str = Field(description="APPROVE | DENY | NEED_INFO")


async def baseline(model, case: dict, annotated: bool = False, use_guards: bool = False) -> dict:
    """One model call with the request, the right policy and the whole record. The common first attempt.
    annotated=True gives it the same day-counted record the specialists get; use_guards=True applies the same
    code checks and computes the decision in code (an ablation, not the baseline)."""
    req = case["request"]
    pol = POLICIES[case["truth"]["policy"]]
    records = fhir.chart(req["patient_id"])
    if annotated:
        chart_text = fhir.timeline(req["patient_id"], req["order_date"])
    else:
        chart_text = "\n".join(fhir.line(r) for r in sorted(records.values(), key=fhir.when, reverse=True))
    parsed, raw = await structured(model, BaselineOut, [SystemMessage(BASELINE.format(policy=pol["text"], rules=RULES)),
                                                        HumanMessage(f"{request_form(req)}\nMedical record:\n{chart_text}")])
    review = {c.criterion_id: c.model_dump() for c in parsed.criteria}
    crit, notes, code_decision = check_review(case["truth"]["policy"], req["patient_id"], review, use_guards, req["order_date"])
    if use_guards:                       # same date rule as AuthPilot, without the second look
        for k, v in crit.items():
            if v.get("window_problem"):
                v["status"] = "not_met"
        code_decision = decide({k: v["status"] for k, v in crit.items()})
    return {"decision": code_decision if use_guards else parsed.decision, "verified": crit, "model_decision": parsed.decision,
            "usage": tokens("single", [raw])}
