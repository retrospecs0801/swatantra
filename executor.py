"""
executor.py
-----------
Deterministic step executor with real-time terminal trace.

Design principles:
- Runs validated steps in declaration order (dependencies are explicit in the plan)
- Injects prior step outputs into subsequent step inputs automatically
- Stops all dependent steps when one fails
- Never claims success when a step has failed
- All progress messages correspond to actual operations
- Logs every step to the audit log
"""

import time, os
from typing import List, Dict, Any, Optional
from capabilities import REGISTRY
import audit_log


class StepResult:
    __slots__ = ("step_index", "capability", "success", "outputs", "error", "duration")

    def __init__(
        self,
        step_index: int,
        capability: str,
        success: bool,
        outputs: dict,
        error: str = None,
        duration: float = 0.0,
    ):
        self.step_index = step_index
        self.capability = capability
        self.success = success
        self.outputs = outputs
        self.error = error
        self.duration = duration


class ExecutionResult:
    def __init__(
        self,
        success: bool,
        step_results: List[StepResult],
        artifact_paths: List[str],
        error: str = None,
    ):
        self.success = success
        self.step_results = step_results
        self.artifact_paths = artifact_paths
        self.error = error


def _trace(tag: str, message: str) -> None:
    """Print a labelled trace line to the terminal."""
    print(f"[{tag}] {message}")


def execute(plan: dict, trace_callback=None) -> ExecutionResult:
    """
    Execute a validated plan.

    Outputs from each step are merged into a shared *context* dict
    and may be referenced by subsequent steps.

    *trace_callback(tag: str, message: str, meta: dict)* is an optional callback
    for UI or real-time event streaming.

    Returns an ExecutionResult. Never raises — errors are captured.
    """
    def _emit(tag: str, message: str, meta: dict = None):
        _trace(tag, message)
        if trace_callback:
            try:
                trace_callback(tag, message, meta or {})
            except Exception:
                pass

    steps = plan.get("steps", [])
    total = len(steps)
    context: Dict[str, Any] = {"_trace_callback": _emit}
    step_results: List[StepResult] = []
    artifact_paths: List[str] = []
    failed = False
    fail_reason: Optional[str] = None

    _emit("EXECUTOR", f"Starting {total} step(s)", {"total": total, "event": "start"})

    for idx, step in enumerate(steps, start=1):
        cap_name = step["capability"]
        raw_inputs = step.get("inputs", {})

        _emit("EXECUTOR", f"Step {idx}/{total}: {cap_name}", {"step": idx, "total": total, "capability": cap_name, "event": "step_start"})

        if failed:
            _emit("SKIP", f"Step {idx} ({cap_name}) — skipped due to prior failure", {"step": idx, "capability": cap_name, "event": "step_skipped"})
            step_results.append(StepResult(idx, cap_name, False, {}, "Skipped due to prior step failure"))
            continue

        cap = REGISTRY[cap_name]

        # Merge context into inputs so prior-step outputs are available
        merged_inputs = {**raw_inputs}
        for key in cap.output_keys:
            # If a prior step produced a key we need, inject it
            pass  # context injection happens below

        # Inject context values for known dependency keys
        merged_inputs = _inject_context(merged_inputs, cap, context)

        start_time = audit_log.log_step_start(idx, cap_name, model="gemma4:e4b")
        t0 = time.monotonic()

        try:
            _emit("STEP", f"Running {cap_name}...", {"step": idx, "capability": cap_name, "event": "step_running"})
            outputs = cap.run(merged_inputs, context)
            duration = time.monotonic() - t0

            # Merge outputs into context for subsequent steps
            context.update(outputs)

            # Collect artifact paths
            for k, v in outputs.items():
                if isinstance(v, str) and k.endswith("_path") and os.path.isfile(v):
                    artifact_paths.append(v)

            audit_log.log_step_complete(idx, cap_name, start_time, outputs, success=True)
            _emit("STEP", f"Completed in {duration:.1f}s", {"step": idx, "capability": cap_name, "duration": duration, "event": "step_completed"})
            step_results.append(StepResult(idx, cap_name, True, outputs, duration=duration))

        except Exception as exc:
            duration = time.monotonic() - t0
            error_msg = f"{type(exc).__name__}: {exc}"
            _emit("ERROR", f"Step {idx} ({cap_name}) failed: {error_msg}", {"step": idx, "capability": cap_name, "error": error_msg, "event": "step_error"})
            audit_log.log_step_complete(idx, cap_name, start_time, {}, success=False, error=error_msg)
            step_results.append(StepResult(idx, cap_name, False, {}, error_msg, duration))
            failed = True
            fail_reason = f"Step {idx} ({cap_name}) failed: {error_msg}"

    overall_success = not failed
    audit_log.log_workflow_complete(artifact_paths, overall_success, fail_reason)

    if overall_success:
        if artifact_paths:
            for p in artifact_paths:
                _emit("OUTPUT", p, {"path": p, "event": "artifact_created"})
        _emit("STATUS", "Workflow completed successfully", {"event": "workflow_success"})
    else:
        _emit("STATUS", f"Workflow FAILED — {fail_reason}", {"event": "workflow_failed", "error": fail_reason})

    return ExecutionResult(
        success=overall_success,
        step_results=step_results,
        artifact_paths=artifact_paths,
        error=fail_reason,
    )


def _inject_context(inputs: dict, cap, context: dict) -> dict:
    """
    Inject values from *context* into *inputs* for fields the capability needs
    but that weren't explicitly set in the plan (they come from prior steps).
    """
    merged = dict(inputs)

    # Known context → input field mappings
    CONTEXT_MAP = {
        "requirements_path": "requirements_path",
        "findings_path": "findings_path",
    }

    for ctx_key, input_key in CONTEXT_MAP.items():
        if input_key not in merged and ctx_key in context:
            merged[input_key] = context[ctx_key]

    return merged
