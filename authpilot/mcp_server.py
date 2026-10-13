"""MCP server exposing the patient chart (FHIR) to the agents.

Run standalone:  python -m authpilot.mcp_server        (stdio transport)
The agents connect to it through the Model Context Protocol, the same way they would connect to
a hospital's FHIR API wrapped as an MCP server. Every tool is read-only.
"""
from mcp.server.fastmcp import FastMCP

from authpilot import fhir

mcp = FastMCP("fhir-chart")


def _lines(rs: list[dict]) -> str:
    return "\n".join(fhir.line(r) for r in rs) or "No matching records."


@mcp.tool()
def get_record(patient_id: str, as_of: str) -> str:
    """The patient's whole record, one line per entry in date order (oldest first), each marked with how many
    days before the as_of date (YYYY-MM-DD) it falls. Use this to read everything at once."""
    return fhir.timeline(patient_id, as_of)


@mcp.tool()
def get_patient(patient_id: str) -> str:
    """Demographics for a patient."""
    return _lines(fhir.by_type(patient_id, "Patient"))


@mcp.tool()
def list_conditions(patient_id: str) -> str:
    """All diagnoses on the problem list, newest first."""
    return _lines(fhir.by_type(patient_id, "Condition"))


@mcp.tool()
def search_observations(patient_id: str, query: str) -> str:
    """Lab results, vital signs and test results whose name matches the query (for example
    'ejection fraction', 'oxygen saturation', 'body mass index', 'glucose'). Newest first within matches."""
    return _lines(fhir.search(patient_id, "Observation", query))


@mcp.tool()
def list_medications(patient_id: str) -> str:
    """Medication orders with their start dates, newest first."""
    return _lines(fhir.by_type(patient_id, "MedicationRequest"))


@mcp.tool()
def list_procedures(patient_id: str) -> str:
    """Procedures performed, newest first."""
    return _lines(fhir.by_type(patient_id, "Procedure"))


@mcp.tool()
def list_orders(patient_id: str) -> str:
    """Orders placed for the patient (the service being requested, tests, referrals), newest first."""
    return _lines(fhir.by_type(patient_id, "ServiceRequest"))


@mcp.tool()
def list_encounters(patient_id: str) -> str:
    """Visits (in person or telehealth) with type and date, newest first."""
    return _lines(fhir.by_type(patient_id, "Encounter"))


@mcp.tool()
def search_notes(patient_id: str, query: str) -> str:
    """Clinical notes and reports containing any of the query words (for example 'sleepiness',
    'shared decision', 'weight management', 'edema'). Returns the full note text."""
    return _lines(fhir.search(patient_id, "DocumentReference", query, limit=12))


if __name__ == "__main__":
    mcp.run()
