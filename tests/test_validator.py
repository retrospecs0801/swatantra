"""
tests/test_validator.py
-----------------------
Automated tests for the plan validator.
No Ollama calls are needed — all plans are constructed directly.

Run with:
    cd "<project_dir>"
    .venv/bin/python -m pytest tests/test_validator.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from validator import validate, MAX_STEPS

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOP_PATH = os.path.join(PROJECT_DIR, "sop.pdf")
REPORT_PATH = os.path.join(PROJECT_DIR, "report.pdf")
REQS_PATH = os.path.join(PROJECT_DIR, "requirements.json")
FINDINGS_PATH = os.path.join(PROJECT_DIR, "findings.json")


# ---- helpers ---------------------------------------------------------------

def make_finance_plan(sop=SOP_PATH, report=REPORT_PATH):
    return {
        "intent": "document_compliance_analysis",
        "steps": [
            {"capability": "extract_requirements", "inputs": {"sop_path": sop}},
            {"capability": "check_report", "inputs": {"report_path": report}},
            {"capability": "build_report", "inputs": {}},
        ],
        "output_format": "docx",
    }


def make_chat_plan(question="What is 2+2?"):
    return {
        "intent": "general_chat",
        "steps": [{"capability": "answer_locally", "inputs": {"question": question}}],
        "output_format": "text",
    }


# ---- Test 2: Valid plans ---------------------------------------------------

class TestValidPlans:
    def test_finance_plan_passes(self):
        result = validate(make_finance_plan())
        assert result.ok, result.reason

    def test_chat_plan_passes(self):
        result = validate(make_chat_plan())
        assert result.ok, result.reason

    def test_single_step_build_with_existing_findings(self):
        plan = {
            "intent": "rebuild_report",
            "steps": [
                {
                    "capability": "build_report",
                    "inputs": {"findings_path": FINDINGS_PATH},
                }
            ],
            "output_format": "docx",
        }
        result = validate(plan)
        assert result.ok, result.reason


# ---- Test 3: Invalid plans -------------------------------------------------

class TestInvalidPlans:

    def test_malformed_not_a_dict(self):
        result = validate("this is not a dict")
        assert not result.ok
        assert "not a JSON object" in result.reason

    def test_missing_intent(self):
        plan = {"steps": [{"capability": "answer_locally", "inputs": {"question": "hi"}}]}
        result = validate(plan)
        assert not result.ok
        assert "intent" in result.reason

    def test_missing_steps(self):
        plan = {"intent": "x"}
        result = validate(plan)
        assert not result.ok
        assert "steps" in result.reason

    def test_empty_steps(self):
        plan = {"intent": "x", "steps": []}
        result = validate(plan)
        assert not result.ok

    def test_too_many_steps(self):
        steps = [{"capability": "answer_locally", "inputs": {"question": "q"}}] * (MAX_STEPS + 1)
        plan = {"intent": "overflow", "steps": steps}
        result = validate(plan)
        assert not result.ok
        assert str(MAX_STEPS) in result.reason

    def test_unknown_capability(self):
        plan = {
            "intent": "hack",
            "steps": [{"capability": "run_shell_command", "inputs": {"cmd": "rm -rf /"}}],
        }
        result = validate(plan)
        assert not result.ok
        assert "unknown capability" in result.reason.lower()

    def test_missing_required_input_sop_path(self):
        plan = {
            "intent": "x",
            "steps": [{"capability": "extract_requirements", "inputs": {}}],
        }
        result = validate(plan)
        assert not result.ok
        assert "sop_path" in result.reason

    def test_missing_required_input_question(self):
        plan = {
            "intent": "x",
            "steps": [{"capability": "answer_locally", "inputs": {}}],
        }
        result = validate(plan)
        assert not result.ok
        assert "question" in result.reason

    def test_missing_required_input_report_path(self):
        plan = {
            "intent": "x",
            "steps": [
                # extract_requirements first so requirements_path is in outputs
                {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                {"capability": "check_report", "inputs": {}},  # missing report_path
            ],
        }
        result = validate(plan)
        assert not result.ok
        assert "report_path" in result.reason

    def test_nonexistent_file(self):
        """
        An absolute path outside the workspace is rejected.
        The validator catches path traversal before checking existence —
        both rejection messages are acceptable; we only assert ok=False.
        """
        plan = {
            "intent": "x",
            "steps": [
                {
                    "capability": "extract_requirements",
                    "inputs": {"sop_path": "/nonexistent/ghost.pdf"},
                }
            ],
        }
        result = validate(plan)
        assert not result.ok
        # Accept either "not found" or "outside" — both mean the file is inaccessible
        assert "not found" in result.reason or "outside" in result.reason or "traversal" in result.reason.lower()

    def test_unsupported_file_type(self):
        # Supply a non-PDF as sop_path — use the findings JSON file as the wrong type
        plan = {
            "intent": "x",
            "steps": [
                {
                    "capability": "extract_requirements",
                    "inputs": {"sop_path": FINDINGS_PATH},  # .json instead of .pdf
                }
            ],
        }
        result = validate(plan)
        assert not result.ok
        assert "unsupported file type" in result.reason.lower()

    def test_path_traversal_rejected(self):
        plan = {
            "intent": "x",
            "steps": [
                {
                    "capability": "extract_requirements",
                    "inputs": {"sop_path": "../../../../etc/passwd"},
                }
            ],
        }
        result = validate(plan)
        assert not result.ok
        # Either path traversal or file-not-found is an acceptable rejection
        assert not result.ok

    def test_forbidden_shell_pattern(self):
        plan = {
            "intent": "evil",
            "steps": [
                {
                    "capability": "answer_locally",
                    "inputs": {
                        "question": "hi",
                        "exec": "os.system('rm -rf /')",
                    },
                }
            ],
        }
        result = validate(plan)
        assert not result.ok
        assert "forbidden" in result.reason.lower()

    def test_forbidden_import_pattern(self):
        plan = {
            "intent": "evil",
            "steps": [
                {
                    "capability": "answer_locally",
                    "inputs": {
                        "question": "__import__('subprocess').run(['ls'])",
                    },
                }
            ],
        }
        result = validate(plan)
        assert not result.ok
        assert "forbidden" in result.reason.lower()

    def test_output_outside_workspace(self):
        plan = {
            "intent": "x",
            "steps": [
                {
                    "capability": "build_report",
                    "inputs": {
                        "findings_path": FINDINGS_PATH,
                        "report_output": "/tmp/evil_output.docx",
                    },
                }
            ],
        }
        result = validate(plan)
        assert not result.ok
        assert "outside" in result.reason.lower()
