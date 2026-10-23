"""
planner.py
----------
Accepts a natural-language request and available file paths, asks the local
LLM to select an appropriate workflow, and returns a structured JSON plan.

The model may only propose capabilities listed in the registry.
Malformed JSON is retried within the llm.ask_json() retry budget.
An explicit fallback is used when all retries are exhausted.

Security note: The planner produces a *proposal* only. Python validates and
executes it. The model cannot add arbitrary code or shell commands.
"""

import json
from typing import List, Optional
from llm import ask_json, MODEL
from capabilities import registry_summary, REGISTRY


# The plan schema the model must follow.
PLAN_SCHEMA = """{
  "intent": "<short description of what the user wants>",
  "steps": [
    {
      "capability": "<name from the registered capability list>",
      "inputs": {
        "<input_field>": "<value>"
      }
    }
  ],
  "output_format": "docx | json | text"
}"""


def build_planner_prompt(request: str, file_paths: List[str]) -> str:
    # Format files as a numbered reference list with canonical paths
    if file_paths:
        files_str = "\n".join(f"  - file_{i+1}: {p}" for i, p in enumerate(file_paths))
        file_ref_note = (
            "\nIMPORTANT: Use the EXACT file paths listed above. "
            "Do NOT modify, shorten or reconstruct any file path. "
            "Copy each path character-for-character into the plan."
        )
    else:
        files_str = "  (none)"
        file_ref_note = ""

    caps = registry_summary()
    return f"""You are a workflow planner for a local AI workbench.

A user has made the following request:
"{request}"

Available local files:
{files_str}{file_ref_note}

Registered capabilities (you MUST only use these):
{caps}

Your job: decide which capability or sequence of capabilities will satisfy the request.

Rules:
1. Use ONLY capability names from the list above. Do NOT invent new names.
2. If the user asks a general question with no document analysis needed, use only "answer_locally" with their question.
3. For document compliance analysis with a SOP and a financial report, use: extract_requirements → check_report → build_report.
4. Use the exact file paths from the "Available local files" list above — do not alter them.
5. Do NOT include shell commands, Python code, or any executable instructions.
6. Do NOT propose more than 8 steps.
7. For "answer_locally", set "question" to the user's exact request.

Return ONLY a JSON plan matching this schema (no explanation, no markdown fences):
{PLAN_SCHEMA}"""


def plan(request: str, file_paths: List[str] = None) -> dict:
    """
    Ask the local LLM for a JSON execution plan.

    Returns a parsed plan dict on success.
    Returns a safe fallback plan (answer_locally) when the model fails.
    Never raises — callers should check plan["intent"] for "_fallback".
    """
    if file_paths is None:
        file_paths = []

    print(f"[PLANNER] Interpreting request via {MODEL}...")
    prompt = build_planner_prompt(request, file_paths)

    try:
        raw_plan = ask_json(prompt)
        # Sanity: if model returned something that has 'steps' as a list, good enough
        if isinstance(raw_plan, dict) and "steps" in raw_plan:
            print(f"[PLANNER] Selected workflow: {raw_plan.get('intent', 'unknown')}")
            return raw_plan
        else:
            print("[PLANNER] Model returned unexpected plan shape — using fallback")
            return _fallback_plan(request, "Model returned unexpected plan structure")
    except ValueError as exc:
        print(f"[PLANNER] All retries exhausted: {exc}")
        return _fallback_plan(request, str(exc))


def _fallback_plan(request: str, reason: str) -> dict:
    """
    Safe fallback when the planner fails.
    Always resolves to answer_locally so the user gets some response.
    """
    print(f"[PLANNER] Falling back to answer_locally. Reason: {reason}")
    return {
        "intent": "_fallback_answer_locally",
        "steps": [
            {
                "capability": "answer_locally",
                "inputs": {"question": request},
            }
        ],
        "output_format": "text",
        "_fallback_reason": reason,
    }
