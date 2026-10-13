"""A small read-only FHIR store over the generated charts, plus compact text renderings.

The same functions back the MCP server (what the agents use) and the code-level checks.
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

CASES = Path(__file__).resolve().parent.parent / "data" / "cases.jsonl"


@lru_cache(maxsize=1)
def store() -> dict:
    """patient_id -> {resource_id: resource}"""
    out = {}
    with open(CASES) as f:
        for line in f:
            case = json.loads(line)
            pid = case["request"]["patient_id"]
            out[pid] = {r["id"]: r for r in case["chart"]}
    return out


def when(r: dict) -> str:
    return (r.get("effectiveDateTime") or r.get("onsetDateTime") or r.get("performedDateTime") or r.get("authoredOn")
            or r.get("date") or (r.get("period") or {}).get("start") or r.get("birthDate") or "")


def text_of(r: dict) -> str:
    """Everything a citation could quote from: names, values, notes, performers."""
    t = r["resourceType"]
    parts = []
    if t == "Patient":
        parts += [r["name"][0]["text"], r["birthDate"], r["gender"]]
    if t in ("Condition", "Observation", "Procedure"):
        parts.append(r["code"]["text"])
        parts += [c.get("code", "") for c in r["code"].get("coding", [])]
    if t == "Condition":
        parts.append(f"status {r.get('clinicalStatus')}")
    if t == "Observation":
        v = r["valueQuantity"]
        parts.append(f"{v['value']} {v['unit']}")
        parts += [p["display"] for p in r.get("performer", [])]
        parts += [(r.get("method") or {}).get("text", "")]
        parts += [n["text"] for n in r.get("note", [])]
    if t == "MedicationRequest":
        parts += [r["medicationCodeableConcept"]["text"], f"status {r.get('status')}"]
    if t == "ServiceRequest":
        parts += [r["code"]["text"], r["code"]["coding"][0]["code"], r["requester"]["display"]]
    if t == "Encounter":
        parts += [x["text"] for x in r["type"]] + [r["class"]["display"]] + [p["display"] for p in r.get("participant", [])]
    if t == "DocumentReference":
        parts += [r["type"]["text"]] + [a["display"] for a in r.get("author", [])] + [r["content"][0]["attachment"]["data"]]
    return " | ".join(p for p in parts if p)


def line(r: dict) -> str:
    """One line per resource, for listings the agents read."""
    t = r["resourceType"]
    d = when(r)
    if t == "Observation":
        v = r["valueQuantity"]
        extra = "; ".join(x for x in [", ".join(p["display"] for p in r.get("performer", [])),
                                       (r.get("method") or {}).get("text", ""),
                                       " ".join(n["text"] for n in r.get("note", []))] if x)
        return f"[{r['id']}] {d} Observation: {r['code']['text']} = {v['value']} {v['unit']}" + (f" ({extra})" if extra else "")
    if t == "Condition":
        return f"[{r['id']}] onset {d} Condition: {r['code']['text']} ({r['code']['coding'][0]['code']}), {r.get('clinicalStatus')}"
    if t == "MedicationRequest":
        return f"[{r['id']}] started {d} Medication: {r['medicationCodeableConcept']['text']} ({r.get('status')})"
    if t == "Procedure":
        return f"[{r['id']}] {d} Procedure: {r['code']['text']}"
    if t == "ServiceRequest":
        return f"[{r['id']}] {d} Order: {r['code']['text']} (code {r['code']['coding'][0]['code']}) by {r['requester']['display']}"
    if t == "Encounter":
        return f"[{r['id']}] {d} Encounter ({r['class']['display']}): {r['type'][0]['text']} with {r['participant'][0]['display']}"
    if t == "DocumentReference":
        return f"[{r['id']}] {d} Note '{r['type']['text']}' by {r['author'][0]['display']}: {r['content'][0]['attachment']['data']}"
    if t == "Patient":
        return f"[{r['id']}] Patient {r['name'][0]['text']}, born {r['birthDate']}, {r['gender']}"
    return f"[{r['id']}] {t}"


def chart(patient_id: str) -> dict:
    if patient_id not in store():
        raise KeyError(f"no patient {patient_id}")
    return store()[patient_id]


def by_type(patient_id: str, rtype: str) -> list[dict]:
    rs = [r for r in chart(patient_id).values() if r["resourceType"] == rtype]
    return sorted(rs, key=when, reverse=True)


def search(patient_id: str, rtype: str, query: str, limit: int = 25) -> list[dict]:
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1]
    rs = by_type(patient_id, rtype)
    if not words:
        return rs[:limit]
    scored = [(sum(w in text_of(r).lower() for w in words), r) for r in rs]
    return [r for s, r in sorted(scored, key=lambda x: (x[0], when(x[1])), reverse=True) if s > 0][:limit]


def days_before(d: str, order_date: str) -> int | None:
    try:
        return (date.fromisoformat(order_date) - date.fromisoformat(d[:10])).days
    except ValueError:
        return None


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def quote_found(resource: dict, quote: str) -> bool:
    """True if every number and most words in the quote appear in the resource's text."""
    hay = normalise(text_of(resource)) + " " + when(resource)
    q = normalise(quote)
    if not q:
        return False
    if q in hay:
        return True
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    if any(n not in re.findall(r"\d+(?:\.\d+)?", hay) for n in nums):
        return False
    if nums:
        return True          # every number quoted is in the record; wording may be paraphrased
    words = [w for w in re.findall(r"[a-z]{4,}", q)]
    return bool(words) and sum(w in hay for w in words) / len(words) >= 0.7
