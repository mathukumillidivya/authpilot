"""AuthPilot: a multi-agent prior-authorization reviewer built with LangGraph.

    intake -> policy -> chart -> assemble -> review -> verify --(gaps, first pass)--> chart
                                                         |
                                                         +--> letter -> END

Agents (language model):
  intake   reads the provider's request form and extracts the structured fields
  policy   picks the coverage policy that governs the requested service
  chart    searches the patient's record through the FHIR MCP server for evidence on each criterion
  review   judges each criterion against the evidence and quotes what it relied on
  letter   drafts the determination letter

Code steps (deterministic):
  assemble attaches each cited record with its date and how many days before the order it falls
  verify   checks every citation exists and says what the reviewer quoted; a criterion with no
           verified evidence cannot count as met or not met; the decision is computed from the
           criteria, never taken from the model; one extra search is allowed for gaps
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

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
            body = " ".join(m.group(2).split())
            crit.append({"id": m.group(1), "text": body, "conditional": body.startswith("Applies only")})
        out[f.stem] = {"id": f.stem, "title": text.splitlines()[0].lstrip("# ").strip(), "text": text, "criteria": crit}
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


# ------------------------------------------------------------------------------------------ state

class State(TypedDict, total=False):
    request_text: str
    request: dict
    policy_id: str
    evidence_ids: dict
    focus: list
    pack: str
    review: dict
    verified: dict
    decision: str
    letter: str
    retries: int
    trace: list
    usage: dict
    errors: list


def parse_json(text: str) -> Any:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


def text_of(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


class Usage:
    """Adds up tokens per node from the messages a node produced."""

    @staticmethod
    def add(state: State, node: str, messages: list) -> dict:
        u = dict(state.get("usage") or {})
        cur = dict(u.get(node) or {"input": 0, "output": 0, "calls": 0})
        for m in messages:
            md = getattr(m, "usage_metadata", None)
            if isinstance(m, AIMessage) and md:
                cur["input"] += md.get("input_tokens", 0)
                cur["output"] += md.get("output_tokens", 0)
                cur["calls"] += 1
        u[node] = cur
        return u


def event(state: State, node: str, summary: str, **detail) -> list:
    return list(state.get("trace") or []) + [{"node": node, "summary": summary, "t": round(time.time(), 2), **detail}]


# ------------------------------------------------------------------------------------------ prompts

INTAKE = """You read prior authorization request forms and extract the fields exactly as written.
Reply with JSON only: {"patient_id": ..., "service": ..., "code": ..., "diagnosis_code": ..., "diagnosis": ...,
"order_date": "YYYY-MM-DD", "ordering_provider": ..., "provider_claims": [short claims the provider makes about
the patient, as stated]}"""

POLICY = """You route prior authorization requests to the coverage policy that governs them.
Policies available:
{catalog}
Reply with JSON only: {{"policy_id": one of the ids above, or "none", "reason": one sentence}}"""

CHART = """You are a clinical evidence researcher for prior authorization review. You search a patient's medical
record through tools and collect the records that bear on each coverage criterion.

Patient ID: {patient_id}. The service was ordered on {order_date}.

Criteria to research:
{criteria}

How to work:
- Use the tools to look through conditions, observations, medications, procedures, encounters and notes. Search
  notes too: evidence is often written only in a note.
- For each criterion collect every record that supports it OR shows it is not met, including the most recent
  measurement and any older measurement of the same thing, and dates that matter for time windows.
- Do not decide whether criteria are met. Only gather evidence.
- Do not rely on what the provider's request claims; find it in the record.

When finished, reply with JSON only, mapping each criterion id to a list of record ids, e.g.
{{"cpap_test": ["obs-004", "doc-007"], "cpap_eval": []}}. Use [] when nothing relevant exists."""

REVIEW = """You are a utilization review nurse deciding whether each coverage criterion is met, using only the
evidence records provided. Today's review concerns an order dated {order_date}.

Policy:
{policy}

For each criterion choose one status:
- "met": the records show the criterion is satisfied.
- "not_met": the records show it is NOT satisfied (a value on the wrong side of a threshold, a date outside the
  allowed window, an excluded condition, a non-covered procedure, or an explicit negative finding).
- "missing": the records contain nothing that settles it either way.
- "not_applicable": only for criteria that say "Applies only ..." when that condition does not hold.

Rules:
- When a measurement was repeated, the most recent one before the order applies.
- Use the "days before order" given with each record for time windows.
- The provider's clinical summary is a claim, not evidence. Never mark a criterion met or not met from it.
- Every "met" or "not_met" must cite at least one record id with a short quote copied from that record
  (include the number or words that decide it).

Reply with JSON only:
{{"criteria": {{"<criterion id>": {{"status": "...", "reason": "one sentence",
  "citations": [{{"id": "<record id>", "quote": "<copied text>"}}]}}, ...}}}}"""

LETTER = """You write prior authorization determination letters to the ordering provider. Plain, professional,
short. State the decision, then each criterion with the evidence relied on (record dates and values), and for a
denial or request for information, exactly what is missing or not met and what the provider can send. Do not
invent any clinical fact that is not in the material given. No more than 220 words."""


# ------------------------------------------------------------------------------------------ code checks

def check_review(policy_id: str, patient_id: str, review: dict, use_guards: bool = True):
    """Verify the reviewer's citations against the record and compute the decision.

    Returns (criteria, notes, decision). With guards on, a criterion judged met or not met without a
    single citation that exists and says what was quoted is treated as missing, and 'not applicable'
    is only accepted for criteria the policy marks as conditional."""
    p = POLICIES[policy_id]
    records = fhir.chart(patient_id)
    out, notes = {}, []
    for c in p["criteria"]:
        v = review.get(c["id"]) or {"status": "missing", "reason": "not assessed", "citations": []}
        status = v.get("status") if v.get("status") in STATUSES else "missing"
        cites = []
        for ct in v.get("citations") or []:
            rid, q = str(ct.get("id", "")), str(ct.get("quote", ""))
            cites.append({"id": rid, "quote": q, "verified": rid in records and fhir.quote_found(records[rid], q)})
        if use_guards:
            if status == "not_applicable" and not c["conditional"]:
                notes.append(f"{c['id']}: marked not applicable but the criterion always applies; treated as missing")
                status = "missing"
            if status in ("met", "not_met") and not any(x["verified"] for x in cites):
                notes.append(f"{c['id']}: '{status}' had no verified citation; treated as missing")
                status = "missing"
        out[c["id"]] = {"status": status, "reason": v.get("reason", ""), "citations": cites, "model_status": v.get("status")}
    return out, notes, decide({k: v["status"] for k, v in out.items()})


# ------------------------------------------------------------------------------------------ graph

def build(model, tools, max_retries: int = 1, use_guards: bool = True):
    """model: a LangChain chat model. tools: the MCP tools for the chart agent."""
    from langgraph.prebuilt import create_react_agent

    catalog = "\n".join(f"- {p['id']}: {p['title']}" for p in POLICIES.values())

    async def intake(state: State) -> State:
        msgs = [SystemMessage(INTAKE), HumanMessage(state["request_text"])]
        out = await model.ainvoke(msgs)
        req = parse_json(text_of(out))
        errors = []
        if req.get("patient_id") not in fhir.store():
            errors.append(f"unknown member id {req.get('patient_id')}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(req.get("order_date", ""))):
            errors.append("order date not readable")
        return {"request": req, "errors": errors, "usage": Usage.add(state, "intake", [out]),
                "trace": event(state, "intake", f"Read request for {req.get('service')} ordered {req.get('order_date')}",
                               fields=req)}

    async def policy(state: State) -> State:
        r = state["request"]
        msgs = [SystemMessage(POLICY.format(catalog=catalog)),
                HumanMessage(f"Service: {r.get('service')}\nCode: {r.get('code')}\nDiagnosis: {r.get('diagnosis')}")]
        out = await model.ainvoke(msgs)
        pid = parse_json(text_of(out)).get("policy_id")
        if pid not in POLICIES:
            raise ValueError(f"no policy matched ({pid})")
        p = POLICIES[pid]
        return {"policy_id": pid, "usage": Usage.add(state, "policy", [out]),
                "trace": event(state, "policy", f"Applied {p['title']}", criteria=[c["id"] for c in p["criteria"]])}

    async def chart(state: State) -> State:
        p, r = POLICIES[state["policy_id"]], state["request"]
        focus = state.get("focus") or [c["id"] for c in p["criteria"]]
        crit = "\n".join(f"- {c['id']}: {c['text']}" for c in p["criteria"] if c["id"] in focus)
        agent = create_react_agent(model, tools, prompt=CHART.format(patient_id=r["patient_id"], order_date=r["order_date"], criteria=crit))
        res = await agent.ainvoke({"messages": [HumanMessage("Collect the evidence now.")]}, {"recursion_limit": 30})
        final = text_of(res["messages"][-1])
        try:
            found = parse_json(final)
        except ValueError:
            found = {}
        ids = dict(state.get("evidence_ids") or {})
        for k, v in found.items():
            if isinstance(v, list):
                ids[k] = sorted(set(ids.get(k, [])) | {str(x) for x in v})
        n_tools = sum(1 for m in res["messages"] if getattr(m, "type", "") == "tool")
        return {"evidence_ids": ids, "usage": Usage.add(state, "chart", res["messages"]),
                "trace": event(state, "chart", f"Searched the record ({n_tools} lookups) for {len(focus)} criteria",
                               found={k: len(v) for k, v in ids.items()}, tool_calls=n_tools)}

    def assemble(state: State) -> State:
        r = state["request"]
        records = fhir.chart(r["patient_id"])
        lines = []
        for cid, ids in (state.get("evidence_ids") or {}).items():
            for rid in ids:
                if rid in records:
                    res = records[rid]
                    db = fhir.days_before(fhir.when(res), r["order_date"])
                    lines.append((rid, f"{fhir.line(res)}  [days before order: {db}]"))
        seen, pack = set(), []
        for rid, l in lines:
            if rid not in seen:
                seen.add(rid)
                pack.append(l)
        return {"pack": "\n".join(pack) or "(no records found)",
                "trace": event(state, "assemble", f"Attached {len(pack)} records with dates relative to the order")}

    async def review(state: State) -> State:
        p, r = POLICIES[state["policy_id"]], state["request"]
        msgs = [SystemMessage(REVIEW.format(order_date=r["order_date"], policy=p["text"])),
                HumanMessage(f"Provider's clinical summary (a claim, not evidence): {r.get('provider_claims')}\n\n"
                             f"Evidence records:\n{state['pack']}")]
        out = await model.ainvoke(msgs)
        rev = parse_json(text_of(out)).get("criteria", {})
        return {"review": rev, "usage": Usage.add(state, "review", [out]),
                "trace": event(state, "review", "Judged each criterion",
                               statuses={k: v.get("status") for k, v in rev.items()})}

    def verify(state: State) -> State:
        out, notes, decision = check_review(state["policy_id"], state["request"]["patient_id"], state.get("review") or {}, use_guards)
        gaps = [k for k, v in out.items() if v["status"] == "missing"]
        retry = use_guards and gaps and (state.get("retries") or 0) < max_retries
        bad = sum(1 for v in out.values() for x in v["citations"] if not x["verified"])
        return {"verified": out, "decision": decision, "focus": gaps if retry else [],
                "retries": (state.get("retries") or 0) + (1 if retry else 0),
                "trace": event(state, "verify",
                               (f"{len(gaps)} criteria lack evidence; searching the record again" if retry else
                                f"Decision computed from the criteria: {decision}"),
                               unverified_citations=bad, overrides=notes, decision=decision)}

    async def letter(state: State) -> State:
        r = state["request"]
        body = json.dumps({"decision": state["decision"], "service": r.get("service"), "order_date": r.get("order_date"),
                           "criteria": {k: {"status": v["status"], "reason": v["reason"],
                                            "evidence": [x["quote"] + f" ({x['id']})" for x in v["citations"] if x["verified"]]}
                                        for k, v in state["verified"].items()}}, indent=1)
        out = await model.ainvoke([SystemMessage(LETTER), HumanMessage(body)])
        return {"letter": text_of(out), "usage": Usage.add(state, "letter", [out]),
                "trace": event(state, "letter", "Drafted the determination letter")}

    g = StateGraph(State)
    for name, fn in [("intake", intake), ("policy", policy), ("chart", chart), ("assemble", assemble),
                     ("review", review), ("verify", verify), ("letter", letter)]:
        g.add_node(name, fn)
    g.set_entry_point("intake")
    g.add_edge("intake", "policy")
    g.add_edge("policy", "chart")
    g.add_edge("chart", "assemble")
    g.add_edge("assemble", "review")
    g.add_edge("review", "verify")
    g.add_conditional_edges("verify", lambda s: "chart" if s.get("focus") else "letter", {"chart": "chart", "letter": "letter"})
    g.add_edge("letter", END)
    return g.compile()


# ------------------------------------------------------------------------------------------ baseline

BASELINE = """You are a utilization review nurse. Decide this prior authorization request using the policy and the
patient's full medical record below.

Policy:
{policy}

Statuses per criterion: met, not_met, missing (nothing documented either way), not_applicable (only for criteria
that say "Applies only ..."). The most recent measurement applies. Cite record ids with quotes.

Reply with JSON only:
{{"criteria": {{"<criterion id>": {{"status": "...", "reason": "...", "citations": [{{"id": "...", "quote": "..."}}]}}}},
 "decision": "APPROVE" | "DENY" | "NEED_INFO"}}"""


async def baseline(model, case: dict) -> dict:
    """One model call with the request, the right policy and the whole chart. The common first attempt."""
    req = case["request"]
    policy = POLICIES[case["truth"]["policy"]]
    records = fhir.chart(req["patient_id"])
    chart_text = "\n".join(f"{fhir.line(r)}" for r in sorted(records.values(), key=fhir.when, reverse=True))
    out = await model.ainvoke([SystemMessage(BASELINE.format(policy=policy["text"])),
                               HumanMessage(f"{request_form(req)}\nMedical record:\n{chart_text}")])
    try:
        parsed = parse_json(text_of(out))
    except (ValueError, json.JSONDecodeError):
        parsed = {"criteria": {}, "decision": "NEED_INFO", "parse_error": True}
    md = out.usage_metadata or {}
    crit = {}
    for c in policy["criteria"]:
        v = (parsed.get("criteria") or {}).get(c["id"]) or {}
        cites = [{"id": str(x.get("id")), "quote": str(x.get("quote", "")),
                  "verified": str(x.get("id")) in records and fhir.quote_found(records[str(x.get("id"))], str(x.get("quote", "")))}
                 for x in v.get("citations") or []]
        crit[c["id"]] = {"status": v.get("status", "missing"), "citations": cites, "reason": v.get("reason", "")}
    return {"decision": parsed.get("decision", "NEED_INFO"), "verified": crit,
            "usage": {"single": {"input": md.get("input_tokens", 0), "output": md.get("output_tokens", 0), "calls": 1}}}
