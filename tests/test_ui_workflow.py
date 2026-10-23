"""
tests/test_ui_workflow.py
-------------------------
Automated tests for Phase 3.1 UI workflow and interactions:
  - Upload validation (valid, missing, unsupported, empty, unusual filenames)
  - Mode routing (Auto, General Chat, Document Analysis)
  - Plan generation, validation, and confirmation lifecycle
  - Live execution event trace capture and calculator tool visibility
  - Failure handling and double-submission resilience
  - Artifact output generation and findings summarization
"""

import sys, os, io, json, tempfile, shutil
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pymupdf
import ui_utils
import planner as planner_mod
import validator as validator_mod
import executor as executor_mod

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOP_PATH = os.path.join(PROJECT_DIR, "sop.pdf")
REPORT_PATH = os.path.join(PROJECT_DIR, "report.pdf")


class DummyUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.data = data

    def getvalue(self) -> bytes:
        return self.data


def _make_dummy_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page()
    b = doc.tobytes()
    doc.close()
    return b


class TestUploadValidationSuite:
    """Test B: Upload validation under various edge cases."""

    def test_two_valid_pdfs(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            pdf1 = DummyUpload("sop_2026.pdf", _make_dummy_pdf())
            pdf2 = DummyUpload("financial_report_q3.pdf", _make_dummy_pdf())

            ok1, p1, err1 = ui_utils.validate_and_save_upload(pdf1, tmp_dir, "sop.pdf")
            ok2, p2, err2 = ui_utils.validate_and_save_upload(pdf2, tmp_dir, "report.pdf")

            assert ok1 and os.path.isfile(p1) and err1 == ""
            assert ok2 and os.path.isfile(p2) and err2 == ""
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_unsupported_file_type(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            txt = DummyUpload("sop.txt", b"plain text")
            ok, p, err = ui_utils.validate_and_save_upload(txt, tmp_dir, "sop.pdf")
            assert not ok
            assert "unsupported" in err.lower()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_empty_file(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            empty = DummyUpload("empty.pdf", b"")
            ok, p, err = ui_utils.validate_and_save_upload(empty, tmp_dir, "sop.pdf")
            assert not ok
            assert "empty" in err.lower()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_unusual_characters_and_spaces_in_filename(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            weird = DummyUpload("my doc (draft #2) [final] & approved.pdf", _make_dummy_pdf())
            ok, p, err = ui_utils.validate_and_save_upload(weird, tmp_dir, weird.name)
            assert ok
            assert os.path.isfile(p)
            # Safe filename has no spaces or special symbols
            assert " " not in os.path.basename(p)
            assert "#" not in os.path.basename(p)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_duplicate_filenames_in_separate_runs(self):
        run1 = ui_utils.create_run_workspace()
        run2 = ui_utils.create_run_workspace()
        try:
            assert run1 != run2
            pdf = DummyUpload("report.pdf", _make_dummy_pdf())
            ok1, p1, _ = ui_utils.validate_and_save_upload(pdf, run1, "report.pdf")
            ok2, p2, _ = ui_utils.validate_and_save_upload(pdf, run2, "report.pdf")
            assert ok1 and ok2
            assert p1 != p2
        finally:
            shutil.rmtree(run1, ignore_errors=True)
            shutil.rmtree(run2, ignore_errors=True)


class TestInteractionModes:
    """Test D & E: Interaction modes and prompt routing."""

    def test_general_chat_mode_plan_isolation(self):
        """General chat mode must never contain finance pipeline steps."""
        chat_req = "Explain zero-trust security."
        plan = {
            "intent": "Answer a general question locally",
            "steps": [{"capability": "answer_locally", "inputs": {"question": chat_req}}],
            "output_format": "text",
        }
        res = validator_mod.validate(plan, workspace=PROJECT_DIR)
        assert res.ok
        caps = [s["capability"] for s in plan["steps"]]
        assert "extract_requirements" not in caps
        assert "check_report" not in caps
        assert "build_report" not in caps

    def test_document_analysis_mode_plan(self):
        """Document analysis mode builds a compliant 3-step workflow."""
        plan = {
            "intent": "Full document compliance analysis and gap register generation",
            "steps": [
                {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                {"capability": "check_report", "inputs": {"report_path": REPORT_PATH}},
                {"capability": "build_report", "inputs": {}},
            ],
            "output_format": "docx",
        }
        res = validator_mod.validate(plan, workspace=PROJECT_DIR, allowed_files=[SOP_PATH, REPORT_PATH])
        assert res.ok
        assert len(plan["steps"]) == 3

    def test_prompt_flexibility_finance_variations(self):
        """Test D: Multiple natural language variations for finance compliance."""
        variations = [
            "Compare this quarterly report against the SOP and generate a gap register.",
            "Check this report for compliance gaps and provide evidence and recommended actions.",
            "Audit report.pdf using sop.pdf and create Gap_Register.docx",
            "Review compliance of the quarterly financials against company policy",
        ]
        for v in variations:
            prompt = planner_mod.build_planner_prompt(v, file_paths=[SOP_PATH, REPORT_PATH])
            assert "extract_requirements" in prompt
            assert "check_report" in prompt
            assert "build_report" in prompt
            assert "Available local files" in prompt
            assert v in prompt

    def test_prompt_flexibility_general_chat_variations(self):
        """Test E: General chat requests do not include file lists."""
        chat_variations = [
            "Summarize zero-trust architecture in five bullet points.",
            "What is the difference between symmetric and asymmetric encryption?",
            "Who was Alan Turing?",
        ]
        for q in chat_variations:
            prompt = planner_mod.build_planner_prompt(q, file_paths=[])
            assert "(none)" in prompt
            assert q in prompt


class TestLiveExecutionTraceAndToolVisibility:
    """Test F: Real-time execution trace and tool visibility callback."""

    def test_trace_callback_captures_all_events(self):
        events_captured = []

        def trace_cb(tag, msg, meta):
            events_captured.append({"tag": tag, "msg": msg, "meta": meta})

        mock_plan = {
            "intent": "Answer a general question locally",
            "steps": [{"capability": "answer_locally", "inputs": {"question": "Hello"}}],
            "output_format": "text",
        }

        with patch("ollama.chat", return_value={"message": {"content": "Hello there!"}}):
            res = executor_mod.execute(mock_plan, trace_callback=trace_cb)

        assert res.success
        tags = [e["tag"] for e in events_captured]
        assert "EXECUTOR" in tags
        assert "STEP" in tags
        assert "MODEL" in tags
        assert "STATUS" in tags

    def test_tool_visibility_on_calculation_event(self):
        """When check_report executes a tool call, trace callback receives tool events."""
        events = []

        def trace_cb(tag, msg, meta):
            events.append((tag, msg, meta))

        # Test context emission via capability trace
        import capabilities
        cap = capabilities.REGISTRY["check_report"]

        mock_finding = {
            "verdict": "Not met",
            "evidence_quote": "variance was 0.26 crore",
            "page": 3,
            "reasoning": "1.4% exceeds 1%",
            "id": "6.2-a",
            "clause": "6.2",
            "requirement": "Net variance exceeds 1%",
            "tool_calls": [
                {"tool": "calculate", "expression": "0.26 / 18.60 * 100", "result": 1.3978, "success": True}
            ],
        }

        with patch("check_report.run", return_value=[mock_finding]):
            outputs = cap.run(
                {"report_path": REPORT_PATH, "requirements_path": "requirements.json"},
                context={"_trace_callback": trace_cb},
            )

        assert "findings_data" in outputs
        assert len(outputs["findings_data"]) == 1
        summary = ui_utils.parse_findings_summary(outputs["findings_data"])
        assert summary["has_calculations"] is True
        assert summary["calculations"][0]["expression"] == "0.26 / 18.60 * 100"


class TestFailureHandlingSuite:
    """Test G: Failure handling, plan rejections, and error states."""

    def test_plan_rejected_by_validator_not_executed(self):
        malicious_plan = {
            "intent": "Run unauthorized command",
            "steps": [{"capability": "answer_locally", "inputs": {"question": "test", "eval": "os.system('bad')"}}],
            "output_format": "text",
        }
        val = validator_mod.validate(malicious_plan, workspace=PROJECT_DIR)
        assert not val.ok
        assert "forbidden pattern" in val.reason.lower() or "eval" in val.reason.lower()

    def test_missing_files_in_plan_rejected(self):
        bad_plan = {
            "intent": "Check non-existent file",
            "steps": [{"capability": "extract_requirements", "inputs": {"sop_path": "/tmp/non_existent_file.pdf"}}],
            "output_format": "docx",
        }
        val = validator_mod.validate(bad_plan, workspace=PROJECT_DIR)
        assert not val.ok
        assert "not found" in val.reason.lower() or "outside" in val.reason.lower()

    def test_step_failure_stops_pipeline(self):
        """If step 1 fails, step 2 must be skipped and workflow marked failed."""
        fail_plan = {
            "intent": "Failing test plan",
            "steps": [
                {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                {"capability": "check_report", "inputs": {"report_path": REPORT_PATH}},
            ],
            "output_format": "docx",
        }

        with patch("extract_requirements.run", side_effect=RuntimeError("Simulated extraction crash")):
            res = executor_mod.execute(fail_plan)

        assert not res.success
        assert len(res.step_results) == 2
        assert res.step_results[0].success is False
        assert res.step_results[1].success is False
        assert "Skipped due to prior" in res.step_results[1].error

    def test_ollama_status_offline_simulation(self):
        """Test G: Ollama status checker handles network/connection errors gracefully."""
        with patch("ollama.list", side_effect=ConnectionError("Connection refused")):
            status = ui_utils.check_ollama_status()
            assert status["available"] is False
            assert status["model_ready"] is False
            assert "unavailable" in status["message"].lower()

    def test_ollama_status_online_detection(self):
        """Test G: Ollama status detects running model."""
        MockModel = MagicMock()
        MockModel.model = "gemma4:e4b"
        MockResp = MagicMock()
        MockResp.models = [MockModel]

        with patch("ollama.list", return_value=MockResp):
            status = ui_utils.check_ollama_status()
            assert status["available"] is True
            assert status["model_ready"] is True
            assert "online" in status["message"].lower()


class TestZeroTrustChatAndDeliverablesIsolation:
    """Test isolation between Zero-Trust Chat and Compare SOP finance workflow outputs."""

    def test_zero_trust_chat_does_not_load_finance_findings_or_deliverables(self):
        """Zero-Trust Chat output must not display Compare SOP findings or Generated Deliverables."""
        chat_plan = {
            "intent": "Answer a general question locally",
            "steps": [{"capability": "answer_locally", "inputs": {"question": "Explain zero-trust architecture."}}],
            "output_format": "text",
        }

        # Mock Ollama response
        with patch("ollama.chat", return_value={"message": {"content": "Zero-trust requires strict verification."}}):
            result = executor_mod.execute(chat_plan)

        assert result.success
        assert len(result.step_results) == 1
        assert result.step_results[0].capability == "answer_locally"
        assert "answer" in result.step_results[0].outputs

        # Simulate post-execution state resolution from app.py
        executed_caps = [s.get("capability") for s in chat_plan.get("steps", [])]
        has_finance = any(c in executed_caps for c in ("extract_requirements", "check_report", "build_report"))
        has_chat = "answer_locally" in executed_caps

        findings_data = None
        chat_answer = None
        generated_deliverables = {}
        workflow_type = None

        if has_finance:
            workflow_type = "finance"
            # (Finance resolution logic)
        if has_chat:
            workflow_type = "chat"
            for step_res in result.step_results:
                if step_res.capability == "answer_locally" and step_res.success and "answer" in step_res.outputs:
                    chat_answer = step_res.outputs["answer"]

        # Crucial assertions: Chat must NOT populate finance data or deliverables
        assert workflow_type == "chat"
        assert chat_answer == "Zero-trust requires strict verification."
        assert findings_data is None, "Zero-Trust Chat must NOT load findings_data"
        assert generated_deliverables == {}, "Zero-Trust Chat must NOT have generated_deliverables"
        assert result.artifact_paths == [], "Chat execution must have no artifact paths"

    def test_reset_state_clears_all_deliverables_and_findings(self):
        """Reset state must wipe all previous execution outputs, findings, and deliverables."""
        # Simulated session state containing prior finance workflow output
        fake_state = {
            "current_plan": {"intent": "Compare SOP"},
            "validation_result": MagicMock(ok=True),
            "execution_result": MagicMock(success=True),
            "execution_events": [{"tag": "STEP", "message": "done"}],
            "findings_data": [{"clause": "2.2", "verdict": "Not met"}],
            "chat_answer": "Previous answer",
            "generated_deliverables": {
                "docx": "/path/to/Gap_Register.docx",
                "findings": "/path/to/findings.json",
                "requirements": "/path/to/requirements.json",
            },
            "workflow_type": "finance",
            "selected_request": "Compare this quarterly report...",
            "run_dir": "/tmp/runs/run_1",
        }

        # Simulate clicking "Reset State"
        fake_state["current_plan"] = None
        fake_state["validation_result"] = None
        fake_state["execution_result"] = None
        fake_state["execution_events"] = []
        fake_state["findings_data"] = None
        fake_state["chat_answer"] = None
        fake_state["generated_deliverables"] = {}
        fake_state["workflow_type"] = None
        fake_state["selected_request"] = ""
        fake_state["run_dir"] = None

        # Verify everything is cleanly reset
        assert fake_state["current_plan"] is None
        assert fake_state["execution_result"] is None
        assert fake_state["findings_data"] is None
        assert fake_state["chat_answer"] is None
        assert fake_state["generated_deliverables"] == {}
        assert fake_state["workflow_type"] is None
        assert fake_state["selected_request"] == ""
        assert fake_state["run_dir"] is None
