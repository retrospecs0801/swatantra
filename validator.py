"""
validator.py
------------
Validates a planner-generated plan before any execution occurs.

Checks performed (in order):
  1.  Plan is a dict (valid JSON was already parsed by the planner)
  2.  Required top-level keys: "intent", "steps"
  3.  "steps" is a non-empty list
  4.  Step count within MAX_STEPS
  5.  Each step has "capability" and "inputs"
  6.  Each capability is registered
  7.  Required input fields are present for each capability
  8.  No forbidden fields (shell, code, exec, eval, subprocess)
  9.  File-path inputs resolve to readable files
  10. File extensions are supported for the relevant capability
  11. Output paths stay within the workspace (no path traversal)
  12. Plan contains no arbitrary executable instructions

Rejection returns a ValidationResult with ok=False and a clear reason.
An unvalidated plan is NEVER executed.
"""

import os, re
from typing import List, Tuple
from capabilities import REGISTRY

MAX_STEPS = 8

# File types accepted per capability (lowercase extensions)
ACCEPTED_EXTENSIONS = {
    "extract_requirements": {".pdf"},
    "check_report": {".pdf"},
    "build_report": {".json"},
}

# Input field names that must point to readable files
FILE_INPUT_FIELDS = {
    "sop_path",
    "report_path",
    "requirements_path",
    "findings_path",
}

# Forbidden keys/patterns — model must not inject executable instructions
FORBIDDEN_PATTERNS = [
    r"\bshell\b",
    r"\bexec\b",
    r"\beval\b",
    r"\bsubprocess\b",
    r"\bos\.system\b",
    r"\bimport\b",
    r"\b__import__\b",
    r"\bopen\(",
]

# Workspace root — output paths must stay within this directory
WORKSPACE = os.path.dirname(os.path.abspath(__file__))


class ValidationResult:
    __slots__ = ("ok", "reason", "warnings")

    def __init__(self, ok: bool, reason: str = "", warnings: List[str] = None):
        self.ok = ok
        self.reason = reason
        self.warnings = warnings or []

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return f"<ValidationResult ok={self.ok} reason={self.reason!r}>"


def validate(plan: dict, workspace: str = WORKSPACE, allowed_files: list = None) -> ValidationResult:
    """
    Validate *plan* against all safety and schema rules.

    *allowed_files*: optional list of file paths the user explicitly provided.
    If a plan input path references an allowed file (or matches with collapsed
    whitespace from LLM tokenization), it is canonically resolved to that file.

    Returns a ValidationResult. Never raises; errors are captured in the result.
    """
    # Build a lookup map to resolve canonical paths (including space-collapsed variants)
    allowed_map: dict = {}
    if allowed_files:
        for f in allowed_files:
            try:
                full = f if os.path.isabs(f) else os.path.join(workspace, f)
                canonical = os.path.realpath(full)
                allowed_map[canonical] = canonical
                allowed_map[full] = canonical
                allowed_map[re.sub(r"\s+", " ", canonical)] = canonical
                allowed_map[re.sub(r"\s+", " ", full)] = canonical
                allowed_map[os.path.basename(full)] = canonical
            except Exception:
                pass
    warnings: List[str] = []

    # 1. Type check
    if not isinstance(plan, dict):
        return ValidationResult(False, "Plan is not a JSON object")

    # 2. Required top-level keys
    for key in ("intent", "steps"):
        if key not in plan:
            return ValidationResult(False, f"Plan is missing required key: '{key}'")

    # 3. steps must be a non-empty list
    steps = plan["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        return ValidationResult(False, "'steps' must be a non-empty list")

    # 4. Step count
    if len(steps) > MAX_STEPS:
        return ValidationResult(
            False,
            f"Plan has {len(steps)} steps; maximum is {MAX_STEPS}",
        )

    # Scan the entire plan text for forbidden patterns
    plan_text = str(plan)
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, plan_text, re.IGNORECASE):
            return ValidationResult(
                False,
                f"Plan contains a forbidden pattern: {pat!r}. "
                "The model must not inject executable instructions.",
            )

    # Per-step checks
    seen_outputs: dict = {}  # track what capabilities have already produced

    for idx, step in enumerate(steps, start=1):
        step_label = f"Step {idx}"

        # 5. Step structure
        if not isinstance(step, dict):
            return ValidationResult(False, f"{step_label}: step must be a JSON object")
        if "capability" not in step:
            return ValidationResult(False, f"{step_label}: missing 'capability' field")
        if "inputs" not in step:
            return ValidationResult(False, f"{step_label}: missing 'inputs' field")
        if not isinstance(step["inputs"], dict):
            return ValidationResult(False, f"{step_label}: 'inputs' must be a JSON object")

        cap_name = step["capability"]

        # 6. Capability must be registered
        if cap_name not in REGISTRY:
            known = ", ".join(sorted(REGISTRY.keys()))
            return ValidationResult(
                False,
                f"{step_label}: unknown capability '{cap_name}'. "
                f"Registered capabilities: {known}",
            )

        cap = REGISTRY[cap_name]
        inputs = step["inputs"]

        # 7. Required inputs
        for field in cap.required_inputs:
            # The field may be absent if it will be injected from a prior step's output
            # (e.g. requirements_path from extract_requirements). We only enforce
            # required inputs that are NOT in FILE_INPUT_FIELDS that could be chained.
            # File-path fields that are outputs of preceding steps are allowed to be absent.
            if field not in inputs:
                # Check if a prior step's output covers this field
                # requirements_path can come from extract_requirements output
                # findings_path can come from check_report output
                provided_by_context = _output_covers(field, seen_outputs)
                if not provided_by_context:
                    return ValidationResult(
                        False,
                        f"{step_label} ({cap_name}): required input '{field}' is missing "
                        "and is not provided by any prior step's output",
                    )

        # 8. File existence and extension for file-path inputs
        for field, value in list(inputs.items()):
            if not isinstance(value, str):
                continue
            if field in FILE_INPUT_FIELDS and value:
                # Resolve space-collapsed or relative names against allowed files
                resolved = _resolve_allowed_path(value, allowed_map)
                if resolved:
                    value = resolved
                    inputs[field] = resolved  # canonicalize in place for the executor

                result = _check_file_input(value, cap_name, step_label, workspace, allowed_map)
                if not result.ok:
                    return result
                if result.warnings:
                    warnings.extend(result.warnings)

        # 9. Output paths must stay in workspace
        for field, value in inputs.items():
            if isinstance(value, str) and field.endswith(("_output", "_path")):
                if not _within_workspace(value, workspace):
                    return ValidationResult(
                        False,
                        f"{step_label}: output path '{value}' is outside the "
                        f"allowed workspace '{workspace}'",
                    )

        # Track this step's output keys for dependency resolution
        for key in cap.output_keys:
            seen_outputs[key] = cap_name

    return ValidationResult(True, warnings=warnings)


def _output_covers(field: str, seen_outputs: dict) -> bool:
    """Check if *field* is produced as an output by a preceding step."""
    # requirements_path is output by extract_requirements
    # findings_path is output by check_report
    return field in seen_outputs


def _resolve_allowed_path(path: str, allowed_map: dict) -> str:
    """Resolve a path to its canonical form from allowed_map if matched."""
    if not allowed_map:
        return None
    if path in allowed_map:
        return allowed_map[path]
    norm = re.sub(r"\s+", " ", path)
    if norm in allowed_map:
        return allowed_map[norm]
    base = os.path.basename(path)
    if base in allowed_map:
        return allowed_map[base]
    return None


def _check_file_input(
    path: str, cap_name: str, step_label: str, workspace: str,
    allowed_map: dict = None,
) -> ValidationResult:
    """
    Validate a single file-path input.
    """
    # If already canonicalized/in allowlist, skip workspace boundary check
    in_allowlist = bool(allowed_map and (path in allowed_map.values() or path in allowed_map))

    if not in_allowlist:
        # Path traversal / outside workspace
        if not _within_workspace(path, workspace):
            return ValidationResult(
                False,
                f"{step_label}: file path '{path}' is outside the allowed workspace. "
                "Path traversal is not permitted.",
            )

    # File must exist and be readable
    if not os.path.isfile(path):
        return ValidationResult(
            False,
            f"{step_label}: file not found: '{path}'",
        )
    if not os.access(path, os.R_OK):
        return ValidationResult(
            False,
            f"{step_label}: file is not readable: '{path}'",
        )

    # Extension check
    if cap_name in ACCEPTED_EXTENSIONS:
        ext = os.path.splitext(path)[1].lower()
        if ext not in ACCEPTED_EXTENSIONS[cap_name]:
            accepted = ", ".join(sorted(ACCEPTED_EXTENSIONS[cap_name]))
            return ValidationResult(
                False,
                f"{step_label}: unsupported file type '{ext}' for '{cap_name}'. "
                f"Accepted: {accepted}",
            )

    return ValidationResult(True)


def _within_workspace(path: str, workspace: str) -> bool:
    """Return True if *path* resolves to a location inside *workspace*."""
    try:
        resolved = os.path.realpath(os.path.join(workspace, path) if not os.path.isabs(path) else path)
        workspace_real = os.path.realpath(workspace)
        return resolved.startswith(workspace_real + os.sep) or resolved == workspace_real
    except Exception:
        return False
