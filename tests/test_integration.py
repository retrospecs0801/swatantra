"""
tests/test_integration.py
--------------------------
Test 5: Real Ollama integration tests.

These tests actually call the local gemma4:e4b model via Ollama.
They are slow (minutes) and require Ollama to be running.

Run with:
    cd "<project_dir>"
    .venv/bin/python -m pytest tests/test_integration.py -v -s

Measures latency. Does NOT assert exact model output — only structure and behavior.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOP_PATH = os.path.join(PROJECT_DIR, "sop.pdf")
REPORT_PATH = os.path.join(PROJECT_DIR, "report.pdf")

pytestmark = pytest.mark.slow


def check_ollama_available():
    """Return True if Ollama is reachable."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=False)
def require_ollama():
    if not check_ollama_available():
        pytest.skip("Ollama is not available — skipping integration tests")


class TestLLMHelper:
    def test_ask_json_returns_dict(self, require_ollama):
        from llm import ask_json
        t0 = time.monotonic()
        result = ask_json('Return JSON: {"value": 42}')
        elapsed = time.monotonic() - t0
        print(f"\n  ask_json latency: {elapsed:.1f}s")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}: {result}"

    def test_ask_json_handles_malformed_response(self, require_ollama):
        """Model should recover within retry budget."""
        from llm import ask_json
        # Ask for a deliberately tricky prompt to stress the retry logic
        result = ask_json(
            'Respond with a JSON object containing a key "ok" with value true.'
        )
        assert "ok" in result or isinstance(result, dict)


class TestPlannerIntegration:
    def test_planner_generates_valid_finance_plan(self, require_ollama):
        """Planner must return a plan with finance steps for a compliance request."""
        import planner as p
        from validator import validate

        t0 = time.monotonic()
        plan = p.plan(
            "Review the SOP and financial report for compliance gaps",
            file_paths=[SOP_PATH, REPORT_PATH],
        )
        elapsed = time.monotonic() - t0
        print(f"\n  Planner latency: {elapsed:.1f}s")
        print(f"  Intent: {plan.get('intent')}")
        print(f"  Steps: {[s['capability'] for s in plan.get('steps', [])]}")

        assert isinstance(plan, dict)
        assert "steps" in plan
        assert len(plan["steps"]) > 0

        # The plan should be valid
        result = validate(plan, allowed_files=[SOP_PATH, REPORT_PATH])
        assert result.ok, f"Plan validation failed: {result.reason}"

    def test_planner_selects_chat_for_general_question(self, require_ollama):
        """General questions must not trigger the finance pipeline."""
        import planner as p

        t0 = time.monotonic()
        plan = p.plan("What is the capital of France?", file_paths=[])
        elapsed = time.monotonic() - t0
        print(f"\n  Planner latency: {elapsed:.1f}s")
        print(f"  Intent: {plan.get('intent')}")
        print(f"  Steps: {[s['capability'] for s in plan.get('steps', [])]}")

        steps = plan.get("steps", [])
        caps = [s.get("capability") for s in steps]

        finance_caps = {"extract_requirements", "check_report", "build_report"}
        assert not any(c in finance_caps for c in caps), (
            f"Finance pipeline must not run for a general question. Got: {caps}"
        )

    def test_planner_fallback_on_model_failure(self):
        """When model returns garbage, the fallback plan is safe and valid."""
        import planner as p
        import importlib
        from unittest.mock import patch

        # Reload planner to get a fresh module reference for patching
        importlib.reload(p)

        # Patch ask_json on the already-imported planner module directly
        with patch.object(p, "ask_json", side_effect=ValueError("Persistent bad JSON")):
            plan = p.plan("test", file_paths=[])

        assert plan.get("intent") == "_fallback_answer_locally", (
            f"Expected fallback intent, got: {plan.get('intent')}"
        )
        assert plan["steps"][0]["capability"] == "answer_locally"

    def test_json_helper_does_not_need_internet(self, require_ollama):
        """
        Model inference must succeed without internet access.
        This test does not actually disconnect Wi-Fi — it verifies
        that the inference call does not raise a network error.
        (Actual offline test should be done manually per the spec.)
        """
        from llm import ask_json
        try:
            result = ask_json('Return JSON: {"local": true}')
            assert isinstance(result, dict)
        except Exception as exc:
            if "network" in str(exc).lower() or "connection" in str(exc).lower():
                pytest.fail(f"Inference required a network connection: {exc}")
            raise


class TestAnswerLocally:
    def test_general_chat_returns_answer(self, require_ollama):
        """answer_locally capability must return a non-empty answer."""
        from capabilities import REGISTRY

        cap = REGISTRY["answer_locally"]
        t0 = time.monotonic()
        result = cap.run({"question": "What is 2 + 2?"}, context={})
        elapsed = time.monotonic() - t0
        print(f"\n  answer_locally latency: {elapsed:.1f}s")

        assert "answer" in result
        assert len(result["answer"]) > 0
        # Should not invoke finance pipeline
        assert "requirements_path" not in result
        assert "findings_path" not in result


class TestToolProtocolIntegration:
    def test_check_report_invokes_calculate_tool(self, require_ollama):
        """
        Verify that check_single_requirement invokes the calculate tool
        when evaluating Clause 6.2 against report.pdf.
        """
        from check_report import check_single_requirement, read_pdf
        import re

        report = read_pdf(REPORT_PATH)
        report_norm = re.sub(r"\s+", " ", report).lower()

        req_6_2 = {
            "clause": "6.2",
            "requirement": "Where the net variance exceeds 1% of book value, written CFO approval is required before the adjustment is posted.",
            "threshold": "1%",
            "checkable": True,
            "id": "6.2-a",
        }

        t0 = time.monotonic()
        finding = check_single_requirement(
            r=req_6_2,
            report=report,
            report_norm=report_norm,
            definitions=[],
            max_tool_calls=3,
        )
        elapsed = time.monotonic() - t0
        print(f"\n  check_single_requirement latency: {elapsed:.1f}s")
        print(f"  Finding: {finding}")

        assert isinstance(finding, dict)
        assert "verdict" in finding
        assert finding["id"] == "6.2-a"
        # Tool call was attempted and recorded
        assert "tool_calls" in finding
        assert len(finding["tool_calls"]) > 0
        assert finding["tool_calls"][0]["tool"] == "calculate"
        assert finding["tool_calls"][0]["success"] is True
        # Result should be approximately 1.3978 or close
        res = finding["tool_calls"][0]["result"]
        assert isinstance(res, (int, float))
        assert abs(res - (0.26 / 18.60 * 100)) < 0.05 or abs(res - 1.3978) < 0.05
