import json
from pathlib import Path

from authpilot import fhir
from authpilot.cases import POLICIES as CASE_POLICIES, decide
from authpilot.graph import POLICIES, check_review

CASES = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "data/cases.jsonl")]


def test_decision_rule():
    assert decide({"a": "met", "b": "not_applicable"}) == "APPROVE"
    assert decide({"a": "met", "b": "missing"}) == "NEED_INFO"
    assert decide({"a": "missing", "b": "not_met"}) == "DENY"


def test_policy_files_match_the_case_criteria():
    for pid, spec in CASE_POLICIES.items():
        assert [c["id"] for c in POLICIES[pid]["criteria"]] == spec["criteria"]


def test_every_case_is_internally_consistent():
    for c in CASES:
        t, ids = c["truth"], {r["id"] for r in c["chart"]}
        assert t["decision"] == decide(t["statuses"])
        for crit, ev in t["evidence"].items():
            assert set(ev) <= ids, (c["case_id"], crit)
        for crit, st in t["statuses"].items():
            if st in ("met", "not_met"):
                assert t["evidence"].get(crit), (c["case_id"], crit, "a decided criterion needs evidence in the chart")


def test_quote_check_is_strict_on_numbers():
    c = next(x for x in CASES if x["truth"]["policy"] == "icd")
    pid = c["request"]["patient_id"]
    lvef = next(r for r in c["chart"] if r["resourceType"] == "Observation" and "ejection" in r["code"]["text"])
    v = lvef["valueQuantity"]["value"]
    assert fhir.quote_found(lvef, f"LVEF {v}%")
    assert not fhir.quote_found(lvef, f"LVEF {v + 7}%")


def _case(policy, decision):
    return next(c for c in CASES if c["truth"]["policy"] == policy and c["truth"]["decision"] == decision)


def test_unsupported_judgement_is_downgraded():
    c = _case("cgm", "APPROVE")
    review = {k: {"status": "met", "citations": [{"id": "obs-999", "quote": "invented"}]} for k in c["truth"]["statuses"]}
    out, notes, decision = check_review("cgm", c["request"]["patient_id"], review)
    assert decision == "NEED_INFO" and all(v["status"] == "missing" for v in out.values()) and notes
    out, notes, decision = check_review("cgm", c["request"]["patient_id"], review, use_guards=False)
    assert decision == "APPROVE"          # without the guard, invented evidence goes straight through


def test_verified_denial_stands():
    c = _case("bariatric", "DENY")
    fail = next(k for k, v in c["truth"]["statuses"].items() if v == "not_met")
    rid = c["truth"]["evidence"][fail][-1] if fail != "bar_procedure" else None
    review = {k: {"status": "met", "citations": []} for k in c["truth"]["statuses"]}
    if rid:
        res = fhir.chart(c["request"]["patient_id"])[rid]
        review[fail] = {"status": "not_met", "citations": [{"id": rid, "quote": fhir.text_of(res)[:60]}]}
        out, _, decision = check_review("bariatric", c["request"]["patient_id"], review)
        assert out[fail]["status"] == "not_met" and decision == "DENY"


def test_not_applicable_only_for_conditional_criteria():
    c = _case("icd", "APPROVE")
    review = {"icd_sdm": {"status": "not_applicable", "citations": []}}
    out, notes, _ = check_review("icd", c["request"]["patient_id"], review)
    assert out["icd_sdm"]["status"] == "missing"


def test_quote_check_on_every_record_line():
    """A record's own description (with the true day count) passes; the same with a wrong day count fails."""
    tot = 0
    for c in CASES[:30]:
        od, records = c["request"]["order_date"], fhir.chart(c["request"]["patient_id"])
        for r in records.values():
            if r["resourceType"] == "Patient":
                continue
            db = fhir.days_before(fhir.when(r), od)
            assert fhir.quote_found(r, f"{fhir.line(r)} [days before order: {db}]", od), r["id"]
            assert not fhir.quote_found(r, f"{fhir.line(r)} [days before order: {db + 3}]", od), r["id"]
            tot += 1
    assert tot > 1000


def test_window_check_flags_stale_evidence():
    from authpilot.graph import window_problem
    c = next(x for x in CASES if x["truth"]["policy"] == "cgm")
    od = c["request"]["order_date"]
    crit = next(k for k in POLICIES["cgm"]["criteria"] if k["id"] == "cgm_visit")
    old = {"resourceType": "Encounter", "period": {"start": "2020-01-01"}}
    assert window_problem(crit, [old], od)
    from datetime import date, timedelta
    recent = {"resourceType": "Encounter", "period": {"start": (date.fromisoformat(od) - timedelta(days=40)).isoformat()}}
    assert window_problem(crit, [old, recent], od) is None
