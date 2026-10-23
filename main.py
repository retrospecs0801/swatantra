"""
main.py
-------
Terminal entry point for the Sovereign AI Workbench.

Usage:
    python main.py

The user enters a natural-language request and optionally provides local file paths.
The application:
  1. Accepts the request
  2. Runs the local planner to generate a JSON plan
  3. Displays the plan
  4. Validates it
  5. Asks for confirmation (or auto-proceeds in --auto mode)
  6. Executes the plan with real-time trace
  7. Logs everything to audit.jsonl

Exit: type 'exit', 'quit', or press Ctrl-C.

Note: all inference is local via Ollama. No data leaves this machine.
"""

import sys, os, json, textwrap

# Ensure the project directory is in sys.path so all modules import correctly
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# Change working directory to the project directory so relative file paths work
os.chdir(_PROJECT_DIR)

import planner as planner_mod
import validator as validator_mod
import executor as executor_mod
import audit_log


AUTO_MODE = "--auto" in sys.argv  # skip confirmation prompts when True


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_banner():
    print()
    print("=" * 62)
    print("  Sovereign AI Workbench — Phase 2")
    print("  Local inference only. No data leaves this machine.")
    print("=" * 62)
    print()


def _print_plan(plan: dict):
    """Pretty-print the plan before execution."""
    print()
    print("[PLAN] Proposed execution plan:")
    print(f"       Intent : {plan.get('intent', 'unknown')}")
    steps = plan.get("steps", [])
    for i, step in enumerate(steps, start=1):
        cap = step.get("capability", "?")
        inputs = step.get("inputs", {})
        input_str = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
        print(f"  {i:2}. {cap}({input_str})")
    print()


def _confirm_plan() -> bool:
    """Ask the user to confirm or reject the plan."""
    if AUTO_MODE:
        return True
    try:
        answer = input("[VALIDATION] Proceed with this plan? [y/N] ").strip().lower()
        return answer in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


def _parse_user_input(raw: str):
    """
    Split user input into (request_text, [file_paths]).

    File paths may contain spaces if quoted. Any token that ends with a
    recognised file extension (.pdf, .json, .docx) or that resolves to an
    existing file is treated as a path. Everything else is part of the request.
    """
    # Simple tokenise: respect quoted strings
    import shlex
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    FILE_EXTS = {".pdf", ".json", ".docx", ".txt"}
    paths = []
    words = []

    for tok in tokens:
        ext = os.path.splitext(tok)[1].lower()
        if ext in FILE_EXTS or os.path.isfile(tok):
            # Resolve relative paths against project dir
            resolved = tok if os.path.isabs(tok) else os.path.join(_PROJECT_DIR, tok)
            paths.append(resolved)
        else:
            words.append(tok)

    request = raw.strip()
    return request, paths


# ---------------------------------------------------------------------------
# Main REPL loop
# ---------------------------------------------------------------------------

def _process_request(request: str, file_paths: list):
    """Run one full request–plan–validate–execute cycle."""
    print()

    # Show inputs
    print(f"[INPUT] Request: {request}")
    for fp in file_paths:
        ext = os.path.splitext(fp)[1].lower()
        label = {"sop": "SOP", ".pdf": "PDF"}.get(ext, "File")
        print(f"[INPUT] File  : {fp}")

    print()

    # Log the request
    audit_log.log_request(intent="pending", files=file_paths, raw_request=request)

    # Plan
    plan = planner_mod.plan(request, file_paths)

    # Display plan
    _print_plan(plan)

    # Validate
    print("[VALIDATION] Validating plan...")
    result = validator_mod.validate(plan, workspace=_PROJECT_DIR, allowed_files=file_paths)

    if not result.ok:
        print(f"[VALIDATION] REJECTED — {result.reason}")
        print()
        # Log rejection
        audit_log.log_workflow_complete([], success=False, error=f"Plan rejected: {result.reason}")
        return

    if result.warnings:
        for w in result.warnings:
            print(f"[VALIDATION] Warning: {w}")

    print("[VALIDATION] Plan accepted")

    # Confirm
    if not _confirm_plan():
        print("[STATUS] Execution cancelled by user.")
        return

    # Log validated plan
    audit_log.log_plan(plan)

    print()

    # Execute
    exec_result = executor_mod.execute(plan)

    if not exec_result.success:
        print(f"\n[STATUS] Failed: {exec_result.error}")
    else:
        if exec_result.artifact_paths:
            print("\n[OUTPUT] Artifacts generated:")
            for p in exec_result.artifact_paths:
                print(f"         {p}")

    print()


def main():
    _print_banner()
    print("Type your request (and optional file paths, quoted if they contain spaces).")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[STATUS] Exiting. Goodbye.")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            print("[STATUS] Exiting. Goodbye.")
            break

        request, file_paths = _parse_user_input(raw)
        _process_request(request, file_paths)


if __name__ == "__main__":
    main()
