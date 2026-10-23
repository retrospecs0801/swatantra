"""
audit_log.py
------------
Appends execution events to a local JSONL audit log (audit.jsonl).

IMPORTANT: This log is NOT tamper-proof. It is a local convenience record
for reviewing what the application did. Do not treat it as a security audit trail.

What IS logged:
  - Timestamps, intents, input file references (paths only, not content)
  - Validated plans (steps and capability names)
  - Capability start/completion, duration, success/failure
  - Model name used for each model call
  - Generated artifact paths

What is NOT logged:
  - Document contents (confidential)
  - Model prompts or responses containing document text
  - Secrets, private keys, or sensitive user data
"""

import json, os, time
from datetime import datetime, timezone
from typing import Any

LOG_PATH = "audit.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(record: dict) -> None:
    """Append a JSON record to the audit log."""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_request(intent: str, files: list, raw_request: str) -> None:
    """Log an incoming user request (file paths only, not file contents)."""
    _append({
        "event": "request",
        "timestamp": _now(),
        "intent": intent,
        "input_files": files,  # paths only
        "request_summary": raw_request[:200],  # truncated for log hygiene
    })


def log_plan(plan: dict) -> None:
    """Log the validated plan (capability names and non-sensitive inputs)."""
    safe_steps = []
    for step in plan.get("steps", []):
        safe_inputs = {
            k: v for k, v in step.get("inputs", {}).items()
            if not any(secret in k.lower() for secret in ("key", "token", "password", "secret"))
        }
        safe_steps.append({
            "capability": step.get("capability"),
            "inputs": safe_inputs,
        })
    _append({
        "event": "plan_validated",
        "timestamp": _now(),
        "intent": plan.get("intent"),
        "steps": safe_steps,
    })


def log_step_start(step_index: int, capability: str, model: str = None) -> float:
    """Log a step starting. Returns the start time for duration calculation."""
    start = time.monotonic()
    _append({
        "event": "step_start",
        "timestamp": _now(),
        "step_index": step_index,
        "capability": capability,
        "model": model,
    })
    return start


def log_step_complete(
    step_index: int,
    capability: str,
    start_time: float,
    outputs: dict,
    success: bool,
    error: str = None,
) -> None:
    """Log a step completing."""
    duration = round(time.monotonic() - start_time, 3)
    # Only log artifact paths, not document content
    safe_outputs = {
        k: v for k, v in outputs.items()
        if isinstance(v, str) and k.endswith("_path")
    }
    record: dict = {
        "event": "step_complete",
        "timestamp": _now(),
        "step_index": step_index,
        "capability": capability,
        "duration_seconds": duration,
        "success": success,
        "artifact_paths": safe_outputs,
    }
    if error:
        record["error"] = error
    _append(record)


def log_tool_call(
    tool_name: str,
    args: dict,
    result: Any,
    duration: float,
    requirement_id: str = None,
    success: bool = True,
    error: str = None,
    model_name: str = "gemma4:e4b",
) -> None:
    """Log a tool execution event."""
    record: dict = {
        "event": "tool_call",
        "timestamp": _now(),
        "tool_name": tool_name,
        "requirement_id": requirement_id,
        "arguments": args,
        "result": result,
        "duration_seconds": round(duration, 4),
        "success": success,
        "model": model_name,
    }
    if error:
        record["error"] = error
    _append(record)


def log_workflow_complete(artifact_paths: list, success: bool, error: str = None) -> None:
    """Log the overall workflow result."""
    record: dict = {
        "event": "workflow_complete",
        "timestamp": _now(),
        "success": success,
        "artifact_paths": artifact_paths,
    }
    if error:
        record["error"] = error
    _append(record)
