"""Code checks for determination letters, shared by scripts/letter_check.py (evaluation) and authpilot/letter.py
(the sign-off gate). No model is involved."""
from __future__ import annotations

import json
import re
from datetime import date

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})

DECISION_WORDS = {
    "DENY": re.compile(r"\bden(ied|ial|y)\b", re.I),
    "APPROVE": re.compile(r"\bapprov(ed|al)\b(?![^.\n]{0,40}\b(cannot|unable|not)\b)", re.I),
    "NEED_INFO": re.compile(r"(additional|more|further|missing) (information|documentation|records)|request for (information|documentation|records)"
                            r"|records? request|need(s|ed)? (more )?(information|documentation)|information (is )?(needed|required)", re.I),
}
HEADLINE = re.compile(r"^\W*(?:\*\*)?\s*(?:prior authorization\s+)?(?:determination\s*[-–:]\s*)?(.{0,80}?)(?:\*\*)?\s*$", re.I)
PLACEHOLDER = re.compile(r"\[(?:[A-Z][A-Za-z]*(?:\s+[A-Za-z]+){0,3})\]")
APPEAL = re.compile(r"\b(appeal|reconsideration|redetermination|re-?review|request (a )?review)\b", re.I)
SEND = re.compile(r"\b(submit|send|provide|forward|fax|upload)\b", re.I)
NOT_MET = re.compile(r"\bnot\s+met\b", re.I)


def dates_in(text: str) -> list[date]:
    out = []
    for y, m, d in re.findall(r"\b(20\d\d|19\d\d)-(\d\d)-(\d\d)\b", text):
        out.append((int(y), int(m), int(d)))
    for mon, d, y in re.findall(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s*(\d{4})\b", text):
        if mon.lower() in MONTHS:
            out.append((int(y), MONTHS[mon.lower()], int(d)))
    for m, d, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text):
        y = int(y) + (2000 if len(y) == 2 else 0)
        out.append((y, int(m), int(d)))
    res = []
    for y, m, d in out:
        try:
            res.append(date(y, m, d))
        except ValueError:
            res.append(None)
    return res


def known_dates(case: dict, extra: str = "") -> set[date]:
    text = json.dumps(case["request"]) + "\n" + "\n".join(case["chart"]) + "\n" + case.get("form", "") + "\n" + extra
    return {d for d in dates_in(text) if d}


def code_checks(letter: str, decision: str, statuses: dict, case: dict, letter_date: str = "") -> dict:
    problems = []
    if not letter.strip():
        return {"problems": ["no letter"], "dates": 0, "dates_unknown": []}
    head = letter.strip().splitlines()[0]
    if not DECISION_WORDS[decision].search(letter):
        problems.append("decision: the letter never states the decision")
    for other, rx in DECISION_WORDS.items():
        if other != decision and rx.search(head):
            problems.append(f"decision: the heading reads as {other}")
    if PLACEHOLDER.search(letter):
        problems.append("placeholder: " + PLACEHOLDER.search(letter).group(0))
    seen = dates_in(letter)
    known = known_dates(case, letter_date)
    unknown = sorted({str(d) if d else "invalid date" for d in seen if d is None or d not in known})
    if unknown:
        problems.append("dates: not found in the request or record (some are computed, such as a resubmission date): " + ", ".join(unknown))
    has_not_met = any(s == "not_met" for s in statuses.values())
    if decision == "NEED_INFO" and not has_not_met and NOT_MET.search(letter):
        problems.append("rule_labels: an undocumented rule is described as 'not met'")
    if decision == "DENY" and not NOT_MET.search(letter) and not re.search(r"does not meet|not meet|fails? to meet|not satisf|excluded|not covered", letter, re.I):
        problems.append("rule_labels: the denial does not name an unmet rule")
    if decision == "DENY" and not APPEAL.search(letter):
        problems.append("next_steps: the denial does not state appeal rights" + (" (it offers resubmission only)" if SEND.search(letter) else ""))
    if decision == "NEED_INFO" and not SEND.search(letter):
        problems.append("next_steps: the records request does not say what to send")
    return {"problems": problems, "dates": len(seen), "dates_unknown": unknown}


