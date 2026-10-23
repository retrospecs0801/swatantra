"""
tests/test_tool_protocol.py
---------------------------
Automated tests for the controlled tool-request protocol in check_report.py.
Mocks model responses to test all protocol paths without calling Ollama.

Tests:
  - Valid calculate request followed by finding
  - Direct finding without tool calls
  - Unknown tool name handling and rejection
  - Invalid calculation (division by zero) handled safely
  - Enforcement of max_tool_calls limit (3 calls max)
  - Audit log recording of tool events
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
import check_report as cr
import audit_log


MOCK_REQ = {
    "id": "6.2-a",
    "clause": "6.2",
    "requirement": "Where the net variance exceeds 1% of book value, written CFO approval is required before the adjustment is posted.",
    "threshold": "1%",
    "checkable": True,
}

MOCK_REPORT = "Net variance was 0.26 crore on book value 18.60 crore (1.4%). The variance adjustment was posted on 3 October 2026."
MOCK_REPORT_NORM = cr.re.sub(r"\s+", " ", MOCK_REPORT).lower()


class TestToolProtocol:

    def test_direct_finding_no_tool(self):
        """Model produces finding directly without requesting any tool."""
        mock_finding = {
            "verdict": "Met",
            "evidence_quote": "posted on 3 October 2026",
            "page": 1,
            "reasoning": "Rule was followed.",
        }

        with patch("check_report.ask_json", return_value=mock_finding):
            res = cr.check_single_requirement(
                r=MOCK_REQ,
                report=MOCK_REPORT,
                report_norm=MOCK_REPORT_NORM,
                definitions=[],
            )

        assert res["verdict"] == "Met"
        assert res["id"] == "6.2-a"
        assert res["quote_check"] == "exact"
        assert "tool_calls" not in res  # No tools used

    def test_valid_calculate_tool_request(self):
        """Model requests calculator, receives result, then returns finding."""
        tool_call_response = {
            "tool": "calculate",
            "args": {"expression": "0.26 / 18.60 * 100"},
        }
        final_finding_response = {
            "verdict": "Not enough evidence",
            "evidence_quote": "Net variance was 0.26 crore",
            "page": 1,
            "reasoning": "Variance was 1.4% which exceeds 1%, but CFO approval is not mentioned.",
        }

        # Sequence of model returns: first tool call, then final finding
        side_effects = [tool_call_response, final_finding_response]

        with patch("check_report.ask_json", side_effect=side_effects):
            res = cr.check_single_requirement(
                r=MOCK_REQ,
                report=MOCK_REPORT,
                report_norm=MOCK_REPORT_NORM,
                definitions=[],
            )

        assert res["verdict"] == "Not enough evidence"
        assert res["id"] == "6.2-a"
        assert "tool_calls" in res
        assert len(res["tool_calls"]) == 1
        assert res["tool_calls"][0]["tool"] == "calculate"
        assert round(res["tool_calls"][0]["result"], 4) == 1.3978
        assert res["tool_calls"][0]["success"] is True

    def test_unknown_tool_rejected(self):
        """Model requests unknown tool; error is returned to model and handled."""
        unknown_tool_response = {
            "tool": "execute_shell",
            "args": {"cmd": "echo hello"},
        }
        final_finding_response = {
            "verdict": "Not triggered",
            "evidence_quote": "Net variance was 0.26 crore",
            "page": 1,
            "reasoning": "Fallback finding.",
        }

        side_effects = [unknown_tool_response, final_finding_response]

        trace_events = []
        def capture_trace(event_type, payload):
            trace_events.append((event_type, payload))

        with patch("check_report.ask_json", side_effect=side_effects):
            res = cr.check_single_requirement(
                r=MOCK_REQ,
                report=MOCK_REPORT,
                report_norm=MOCK_REPORT_NORM,
                definitions=[],
                tool_callback=capture_trace,
            )

        assert res["verdict"] == "Not triggered"
        errors = [p for ev, p in trace_events if ev == "error"]
        assert len(errors) == 1
        assert "Unknown tool" in errors[0]["error"]

    def test_tool_failure_handled_gracefully(self):
        """Calculator error (division by zero) passed to model without crashing."""
        tool_call_div_zero = {
            "tool": "calculate",
            "args": {"expression": "10 / 0"},
        }
        final_finding = {
            "verdict": "Not enough evidence",
            "evidence_quote": "Net variance was 0.26 crore",
            "page": 1,
            "reasoning": "Could not compute threshold, evidence insufficient.",
        }

        with patch("check_report.ask_json", side_effect=[tool_call_div_zero, final_finding]):
            res = cr.check_single_requirement(
                r=MOCK_REQ,
                report=MOCK_REPORT,
                report_norm=MOCK_REPORT_NORM,
                definitions=[],
            )

        assert res["verdict"] == "Not enough evidence"
        assert len(res["tool_calls"]) == 1
        assert res["tool_calls"][0]["success"] is False
        assert res["tool_calls"][0]["result"] is None

    def test_max_tool_calls_limit_enforced(self):
        """Loop terminates after MAX_TOOL_CALLS (3) even if model keeps requesting tools."""
        infinite_tool_call = {
            "tool": "calculate",
            "args": {"expression": "1 + 1"},
        }
        forced_final_finding = {
            "verdict": "Met",
            "evidence_quote": "Net variance was 0.26 crore",
            "page": 1,
            "reasoning": "Forced verdict after limit reached.",
        }

        # Model requests tool 4 times, but loop must force verdict on 4th call
        side_effects = [
            infinite_tool_call,
            infinite_tool_call,
            infinite_tool_call,
            forced_final_finding,
        ]

        with patch("check_report.ask_json", side_effect=side_effects):
            res = cr.check_single_requirement(
                r=MOCK_REQ,
                report=MOCK_REPORT,
                report_norm=MOCK_REPORT_NORM,
                definitions=[],
                max_tool_calls=3,
            )

        assert res["verdict"] == "Met"
        assert len(res["tool_calls"]) == 3  # Exactly 3 tool calls executed

    def test_audit_log_records_tool_call(self):
        """Tool calls are appended to audit.jsonl with proper fields."""
        tool_call = {
            "tool": "calculate",
            "args": {"expression": "100 * 0.05"},
        }
        final_finding = {
            "verdict": "Met",
            "evidence_quote": "Net variance was 0.26 crore",
            "page": 1,
            "reasoning": "Done.",
        }

        log_path = audit_log.LOG_PATH
        initial_lines = 0
        if os.path.exists(log_path):
            with open(log_path) as f:
                initial_lines = sum(1 for _ in f)

        with patch("check_report.ask_json", side_effect=[tool_call, final_finding]):
            cr.check_single_requirement(
                r=MOCK_REQ,
                report=MOCK_REPORT,
                report_norm=MOCK_REPORT_NORM,
                definitions=[],
            )

        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()[initial_lines:]

        records = [json.loads(l) for l in lines]
        tool_records = [r for r in records if r.get("event") == "tool_call"]

        assert len(tool_records) == 1
        tr = tool_records[0]
        assert tr["tool_name"] == "calculate"
        assert tr["arguments"] == {"expression": "100 * 0.05"}
        assert tr["result"] == 5.0
        assert tr["success"] is True
        assert tr["requirement_id"] == "6.2-a"
        assert "duration_seconds" in tr

    def test_multi_step_tool_calls_sequence(self):
        """Model makes two consecutive calculations before concluding."""
        step1_call = {
            "tool": "calculate",
            "args": {"expression": "0.26 / 18.60 * 100"},
        }
        step2_call = {
            "tool": "calculate",
            "args": {"expression": "1.3978 > 1.0"},
        }
        final_finding = {
            "verdict": "Not enough evidence",
            "evidence_quote": "Net variance was 0.26 crore",
            "page": 1,
            "reasoning": "Variance was 1.4% which is greater than 1%, but approval is absent.",
        }

        trace_log = []
        def on_trace(ev, payload):
            trace_log.append(ev)

        side_effects = [step1_call, step2_call, final_finding]
        with patch("check_report.ask_json", side_effect=side_effects):
            res = cr.check_single_requirement(
                r=MOCK_REQ,
                report=MOCK_REPORT,
                report_norm=MOCK_REPORT_NORM,
                definitions=[],
                tool_callback=on_trace,
            )

        assert res["verdict"] == "Not enough evidence"
        assert len(res["tool_calls"]) == 2
        assert res["tool_calls"][0]["expression"] == "0.26 / 18.60 * 100"
        assert round(res["tool_calls"][0]["result"], 4) == 1.3978
        assert res["tool_calls"][1]["expression"] == "1.3978 > 1.0"
        assert res["tool_calls"][1]["result"] is True
        assert "requested" in trace_log
        assert "validating" in trace_log
        assert "result" in trace_log
        assert "producing_finding" in trace_log
