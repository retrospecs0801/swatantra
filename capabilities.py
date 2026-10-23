"""
capabilities.py
---------------
Central registry of every capability the orchestrator may execute.

Each capability has:
  - name            : stable identifier used in plans
  - description     : shown to the planner model
  - required_inputs : field names that must appear in step["inputs"]
  - optional_inputs : field names that may appear
  - output_keys     : what this step adds to the shared context
  - execute         : callable(inputs, context) → dict of output_key→value

The model may only propose capabilities that appear in REGISTRY.
The executor never calls any function not listed here.
"""

import os, time
from typing import Dict, Any, Callable

# ---- type alias ------------------------------------------------------------ #
ExecuteFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


class Capability:
    def __init__(
        self,
        name: str,
        description: str,
        required_inputs: list,
        optional_inputs: list,
        output_keys: list,
        execute: ExecuteFn,
    ):
        self.name = name
        self.description = description
        self.required_inputs = required_inputs
        self.optional_inputs = optional_inputs
        self.output_keys = output_keys
        self._execute = execute

    def run(self, inputs: dict, context: dict) -> dict:
        return self._execute(inputs, context)

    def __repr__(self):
        return f"<Capability {self.name}>"


# ---- execute functions ----------------------------------------------------- #

def _execute_extract_requirements(inputs: dict, context: dict) -> dict:
    """Thin adapter around extract_requirements.run()."""
    import extract_requirements as er
    trace_cb = context.get("_trace_callback")
    if trace_cb:
        trace_cb("MODEL", "Extracting compliance obligations from SOP...", {"event": "extract_start"})
    sop_path = inputs["sop_path"]
    output_path = inputs.get("requirements_output", "requirements.json")
    data = er.run(sop_path=sop_path, output_path=output_path)
    return {
        "requirements_path": output_path,
        "requirements_data": data,
    }


def _execute_check_report(inputs: dict, context: dict) -> dict:
    """Thin adapter around check_report.run()."""
    import check_report as cr
    report_path = inputs["report_path"]
    # Use requirements path from previous step if not explicitly supplied
    requirements_path = inputs.get(
        "requirements_path",
        context.get("requirements_path", "requirements.json"),
    )
    output_path = inputs.get("findings_output", "findings.json")
    trace_cb = context.get("_trace_callback")

    def _progress(step, total, req_id):
        msg = f"Checking clause {req_id} ({step}/{total})..."
        print(f"  [MODEL] {msg}")
        if trace_cb:
            trace_cb("MODEL", msg, {"event": "check_clause", "step": step, "total": total, "req_id": req_id})

    def _tool_trace(event_type, payload):
        if event_type == "requested":
            msg = f"Requested tool: {payload.get('tool')}"
            print(f"  [MODEL] {msg}")
            if trace_cb:
                trace_cb("MODEL", msg, {"event": "tool_requested", **payload})
        elif event_type == "validating":
            msg = f"Validating expression: {payload.get('expression')}"
            print(f"  [TOOL] {msg}")
            if trace_cb:
                trace_cb("TOOL", msg, {"event": "tool_validating", **payload})
        elif event_type == "result":
            res = payload.get("result", {})
            if res.get("success"):
                msg = f"Result: {res.get('result')} ({res.get('formatted')})"
                print(f"  [TOOL] {msg}")
                print(f"  [MODEL] Received calculation result")
                if trace_cb:
                    trace_cb("TOOL", msg, {"event": "tool_result", **payload})
                    trace_cb("MODEL", "Received calculation result", {"event": "tool_feedback", **payload})
            else:
                msg = f"Error: {res.get('error')}"
                print(f"  [TOOL] {msg}")
                if trace_cb:
                    trace_cb("TOOL", msg, {"event": "tool_error", **payload})
        elif event_type == "producing_finding":
            msg = "Producing finding"
            print(f"  [MODEL] {msg}")
            if trace_cb:
                trace_cb("MODEL", msg, {"event": "producing_finding", **payload})
        elif event_type == "error":
            msg = f"Error: {payload.get('error')}"
            print(f"  [TOOL] {msg}")
            if trace_cb:
                trace_cb("TOOL", msg, {"event": "tool_error", **payload})

    findings = cr.run(
        report_path=report_path,
        requirements_path=requirements_path,
        output_path=output_path,
        progress_callback=_progress,
        tool_callback=_tool_trace,
    )
    return {
        "findings_path": output_path,
        "findings_data": findings,
    }


def _execute_build_report(inputs: dict, context: dict) -> dict:
    """Thin adapter around build_report.run()."""
    import build_report as br
    findings_path = inputs.get(
        "findings_path",
        context.get("findings_path", "findings.json"),
    )
    output_path = inputs.get("report_output", "Gap_Register.docx")
    trace_cb = context.get("_trace_callback")

    def _progress(item_id):
        msg = f"Generating follow-up for {item_id}..."
        print(f"  [MODEL] {msg}")
        if trace_cb:
            trace_cb("MODEL", msg, {"event": "follow_up", "item_id": item_id})

    path = br.run(
        findings_path=findings_path,
        output_path=output_path,
        progress_callback=_progress,
    )
    return {"artifact_path": path}


def _execute_answer_locally(inputs: dict, context: dict) -> dict:
    """
    Answer a general question using the local LLM.
    Does not touch any finance pipeline.
    """
    from llm import MODEL
    import ollama
    question = inputs["question"]
    trace_cb = context.get("_trace_callback")
    if trace_cb:
        trace_cb("MODEL", f"Querying local model ({MODEL})...", {"event": "chat_start"})

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.7},
    )
    answer = response["message"]["content"]
    print(f"\n{answer}\n")
    if trace_cb:
        trace_cb("MODEL", "Answer generated successfully.", {"event": "chat_complete"})
    return {"answer": answer}


# ---- registry -------------------------------------------------------------- #

REGISTRY: Dict[str, Capability] = {}


def _register(cap: Capability):
    REGISTRY[cap.name] = cap


_register(Capability(
    name="extract_requirements",
    description=(
        "Reads a company SOP PDF and extracts every compliance obligation "
        "as structured JSON. Writes requirements.json."
    ),
    required_inputs=["sop_path"],
    optional_inputs=["requirements_output"],
    output_keys=["requirements_path", "requirements_data"],
    execute=_execute_extract_requirements,
))

_register(Capability(
    name="check_report",
    description=(
        "Compares a financial report PDF against extracted SOP requirements. "
        "Produces a findings JSON with verdicts, evidence quotes and reasoning."
    ),
    required_inputs=["report_path"],
    optional_inputs=["requirements_path", "findings_output"],
    output_keys=["findings_path", "findings_data"],
    execute=_execute_check_report,
))

_register(Capability(
    name="build_report",
    description=(
        "Reads findings.json and generates a Gap Register DOCX document "
        "with executive summary, gap table, review checklist and follow-up actions."
    ),
    required_inputs=[],
    optional_inputs=["findings_path", "report_output"],
    output_keys=["artifact_path"],
    execute=_execute_build_report,
))

_register(Capability(
    name="answer_locally",
    description=(
        "Answers a general natural-language question using the local LLM. "
        "Use for any request that does not involve document compliance analysis."
    ),
    required_inputs=["question"],
    optional_inputs=[],
    output_keys=["answer"],
    execute=_execute_answer_locally,
))


def registry_summary() -> str:
    """Return a human-readable capability summary for planner prompts."""
    lines = []
    for cap in REGISTRY.values():
        req = ", ".join(cap.required_inputs) if cap.required_inputs else "none"
        lines.append(f"  - {cap.name}: {cap.description} (required inputs: {req})")
    return "\n".join(lines)
