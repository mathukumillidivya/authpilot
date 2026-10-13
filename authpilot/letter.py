"""Determination letters, version 2: the structure is code, the model writes only the reasons.

The original letter step let the model write the whole letter. Checked separately (scripts/letter_check.py), 94 of
200 held-out letters failed at least one code check: 60 kept a template placeholder, 39 of 82 denials gave no
appeal route, and some called a rule "not met" when it was only undocumented. Version 2 moves everything that can
be decided in code into code:

  header        request id, letter date, patient, member id, service and code, order date, ordering provider
  decision      one fixed sentence per decision
  rule list     every rule with its status written by code (the model cannot relabel a status)
  next steps    a fixed paragraph per decision: appeal route for a denial, the list of records to send for a
                records request, the scope of an approval (no invented validity period)

The model writes one or two sentences per rule from the verified quotes, and nothing else. Then the letter is
validated in code before anyone can sign it (validate()): placeholders, dates that are not in the request, the
record or the letter date, a reason that contradicts its rule's status, and the decision and next steps present.
A letter that fails is not offered for sign-off; it is marked for a person to edit.

Version 2.1: the second-model check of version 2 (results/runs/test-authpilot/letter_check_v2.json) found that the
short rule labels the template borrowed from the review screen ("Three months of therapy", "Insulin or severe
lows") misdescribe some rules. Each rule is now quoted in the policy's own words instead.
"""
from __future__ import annotations

import json
import re

from authpilot.letter_rules import code_checks

STATUS_WORDS = {"met": "Met", "not_met": "Not met", "missing": "Not documented in the records received",
                "not_applicable": "Does not apply to this request"}
DECISION_LINE = {
    "APPROVE": "The request is approved. The records show that every coverage rule that applies is met.",
    "DENY": "The request is denied. The records show that at least one coverage rule is not met.",
    "NEED_INFO": "We cannot decide this request yet. The records we received do not document every coverage rule. "
                 "This is a request for records, not a denial.",
}
APPEAL = ("If you disagree with this decision, you or the patient can ask for it to be reviewed again (an appeal, "
          "also called a reconsideration). Send the request with any records that address the rules marked "
          "\"Not met\" above. Your plan's appeal notice gives the time limit and the address.")
APPROVAL_SCOPE = "This approval covers the service and code above, as ordered on {order_date}."
NEED_INFO_INTRO = "Please send records that document:"
PLACEHOLDER = re.compile(r"\[[A-Za-z][^\]]{0,40}\]")

REASONS_PROMPT = """You write the reason lines of a prior authorization letter to the ordering provider. For each \
rule you get its wording, the status decided by the review (already final), the reviewer's reason and the quotes \
from the patient record that code verified.

For each rule write one or two plain sentences that explain the status using only the verified quotes and the \
reason. Rules:
- Never contradict or soften the status. If the status is "missing", say what is not documented; never say "not met".
- Use only dates, values and names that appear in the quotes or the reason. Do not compute new dates.
- No placeholders, no greeting, no sign-off, no policy terms that are not in the rule wording.

Submit with the submit_reasons tool."""
TOOL = {"name": "submit_reasons", "description": "Submit one reason per rule.", "input_schema": {
    "type": "object", "required": ["reasons"], "properties": {"reasons": {"type": "array", "items": {
        "type": "object", "required": ["rule", "text"], "properties": {
            "rule": {"type": "string", "description": "The rule's id, exactly as given (for example cpap_eval)."},
            "text": {"type": "string"}}}}}}}


def reasons_input(criteria: dict, policy: dict) -> str:
    text = {c["id"]: c["text"] for c in policy["criteria"]}
    return json.dumps({k: {"id": k, "rule": text.get(k, k), "status": v.get("status"), "reason": v.get("reason"),
                           "verified_quotes": [x["quote"] for x in v.get("citations", []) if x.get("verified")]}
                       for k, v in criteria.items()}, indent=1)


def parse_reasons(out, policy: dict | None = None) -> dict:
    """Reason lines keyed by rule id. The model sometimes names a rule by its wording instead of its id, so match the
    wording too (version 2 as first run lost every reason line this way)."""
    if isinstance(out, str):
        out = json.loads(out)
    items = out.get("reasons", out) if isinstance(out, dict) else out
    if isinstance(items, str):
        items = json.loads(items)
    by_text = {re.sub(r"\W+", " ", c["text"]).strip().lower(): c["id"] for c in (policy or {}).get("criteria", [])}
    ids = {c["id"] for c in (policy or {}).get("criteria", [])}
    found = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        key = str(x.get("rule", "")).strip()
        norm = re.sub(r"\W+", " ", key).strip().lower()
        rid = key if not ids or key in ids else by_text.get(norm) or next((i for t, i in by_text.items() if norm and (norm in t or t in norm)), None)
        if rid:
            found[rid] = str(x.get("text", "")).strip()
    return found


def assemble(request: dict, decision: str, criteria: dict, policy: dict, labels: dict, reasons: dict, letter_date: str,
             rule_wording: bool = True) -> str:
    r = request
    who = r.get("patient_name") or ""
    member = f"member {r['patient_id']}" if r.get("patient_id") else ""
    head = [("Request", r.get("request_id")), ("Date of this letter", letter_date),
            ("Patient", f"{who} ({member})" if who and member else who or member),
            ("Service", f"{r.get('service', '')} ({r['code']})" if r.get("code") else r.get("service")),
            ("Order date", r.get("order_date")), ("Ordering provider", r.get("ordering_provider"))]
    head = [f"**{k}:** {v}" for k, v in head if v]
    lines = ["**PRIOR AUTHORIZATION DETERMINATION**", "", *[h + "  " for h in head[:-1]], head[-1], "",
             f"**Decision: {dict(APPROVE='APPROVED', DENY='DENIED', NEED_INFO='RECORDS REQUESTED')[decision]}**", "",
             DECISION_LINE[decision], "", f"**Coverage rules ({policy['title']})**", ""]
    text = {c["id"]: c["text"] for c in policy["criteria"]}
    for k, v in criteria.items():
        reason = PLACEHOLDER.sub("", reasons.get(k, "")).strip()
        if rule_wording:  # version 2.1: the policy's own words, not a short label (see the module docstring)
            lines.append(f"- **{STATUS_WORDS.get(v.get('status'), 'Not assessed')}.** *Rule:* {text.get(k, k)} {reason}".rstrip())
        else:
            lines.append(f"- **{labels.get(k, k)}: {STATUS_WORDS.get(v.get('status'), 'Not assessed')}.** {reason}".rstrip())
    lines.append("")
    if decision == "DENY":
        lines += ["**Your options**", "", APPEAL]
    elif decision == "NEED_INFO":
        lines += ["**What to send**", "", NEED_INFO_INTRO]
        lines += [f"- {text.get(k, '')}" if rule_wording else f"- {labels.get(k, k)}: {text.get(k, '')}"
                  for k, v in criteria.items() if v.get("status") == "missing"]
        lines += ["", "We will complete the review when the records arrive."]
    else:
        lines.append(APPROVAL_SCOPE.format(order_date=r.get("order_date", "the order date")))
    return "\n".join(lines)


REASON_CONTRADICTS = {
    "not_met": re.compile(r"\b(criterion|requirement|rule) is (actually |therefore |thus )?met\b|\bthis criterion is actually met\b", re.I),
    "met": re.compile(r"\b(criterion|requirement|rule) is (actually |therefore )?not met\b", re.I),
}


def validate(letter: str, decision: str, criteria: dict, case: dict, letter_date: str, policy: dict | None = None) -> list[str]:
    """Code checks a letter must pass before sign-off. Returns the problems (empty means it may be signed)."""
    statuses = {k: v.get("status") for k, v in criteria.items()}
    quoted = tuple(c["text"] for c in (policy or {}).get("criteria", []))
    return code_checks(letter, decision, statuses, case, letter_date, quoted)["problems"] + contradictions(criteria)


def contradictions(criteria: dict) -> list[str]:
    """A specialist whose reason concludes the opposite of its own status (the held-out case 0045)."""
    out = []
    for k, v in criteria.items():
        rx = REASON_CONTRADICTS.get(v.get("status"))
        if rx and rx.search(v.get("reason") or ""):
            out.append(f"reason_status: the reason for {k} concludes the opposite of its status ({v.get('status')})")
    return out
