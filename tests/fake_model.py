"""A scripted stand-in for the language model, so the whole graph can be tested without an API key.

It reads the system prompt to tell which agent is calling and answers the way a careful reviewer
would, using the chart directly. It is deliberately simple; it exists to test the plumbing.
"""
import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeModel(BaseChatModel):
    tool_plan: list = ["list_conditions", "search_observations", "list_encounters", "search_notes"]

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        system = next((m.content for m in messages if m.type == "system"), "")
        text = self._reply(system, messages)
        msg = text if isinstance(text, AIMessage) else AIMessage(content=text)
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _reply(self, system, messages):
        last = messages[-1].content if messages else ""
        if system.startswith("You read prior authorization"):
            form = last
            get = lambda label: re.search(label + r":\s*([^\n]+?)(?:\s{3}|\n)", form).group(1).strip()
            return json.dumps({"patient_id": get("Member ID"), "service": get("Requested service"), "code": get("Code"),
                               "diagnosis": get("Diagnosis"), "order_date": get("Date of order"),
                               "ordering_provider": get("Ordering provider"), "provider_claims": []})
        if system.startswith("You route"):
            s = last.lower()
            pid = ("cpap" if "cpap" in s else "oxygen" if "oxygen" in s else "icd" if "defibrillator" in s
                   else "cgm" if "glucose" in s else "bariatric")
            return json.dumps({"policy_id": pid, "reason": "matched by service"})
        if "clinical evidence researcher" in system:
            pid = re.search(r"Patient ID: (\S+)\.", system).group(1)
            done = [m for m in messages if isinstance(m, ToolMessage)]
            if len(done) < len(self.tool_plan):
                name = self.tool_plan[len(done)]
                args = {"patient_id": pid}
                if name.startswith("search"):
                    args["query"] = "the"
                return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call{len(done)}"}])
            flat = lambda c: c if isinstance(c, str) else " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
            ids = re.findall(r"\[([a-z]+-\d+)\]", " ".join(flat(m.content) for m in done))
            crit = re.findall(r"^- ([a-z0-9_]+):", system, re.M)
            return json.dumps({c: ids[:6] for c in crit})
        if "utilization review nurse" in system:
            crit = re.findall(r"\*\*([a-z0-9_]+)\.\*\*", system)
            rec = re.findall(r"\[([a-z]+-\d+)\] (\S+)", last)
            cite = [{"id": rec[0][0], "quote": rec[0][1]}] if rec else []
            out = {"criteria": {c: {"status": "met", "reason": "scripted", "citations": cite} for c in crit}}
            if "Medical record" in last:
                out["decision"] = "APPROVE"
            return json.dumps(out)
        if system.startswith("You write prior authorization determination"):
            return "Dear provider, this is a scripted letter."
        return "{}"
