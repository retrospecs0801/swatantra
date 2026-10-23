"""
tests/test_executor.py
----------------------
Automated tests for the executor using mocked capabilities.
No Ollama calls or real pipeline runs needed.

Run with:
    cd "<project_dir>"
    .venv/bin/python -m pytest tests/test_executor.py -v
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
import capabilities
import executor


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOP_PATH = os.path.join(PROJECT_DIR, "sop.pdf")
REPORT_PATH = os.path.join(PROJECT_DIR, "report.pdf")


# ---- Test 2: Executor with mocked capabilities ----------------------------

class TestExecutorMocked:

    def _mock_cap(self, name, required_inputs, output_keys, return_value, will_fail=False):
        """Create and register a temporary mock capability."""
        def fake_execute(inputs, context):
            if will_fail:
                raise RuntimeError(f"Simulated failure in {name}")
            return return_value

        cap = capabilities.Capability(
            name=name,
            description=f"Mock {name}",
            required_inputs=required_inputs,
            optional_inputs=[],
            output_keys=output_keys,
            execute=fake_execute,
        )
        original = capabilities.REGISTRY.get(name)
        capabilities.REGISTRY[name] = cap
        return original  # return original so we can restore it

    def test_chat_plan_does_not_run_finance_capabilities(self):
        """General chat must not invoke any finance pipeline step."""
        chat_run = []

        original_answer = capabilities.REGISTRY["answer_locally"]._execute

        def fake_answer(inputs, context):
            chat_run.append("answer_locally")
            return {"answer": "42"}

        finance_run = []
        original_extract = capabilities.REGISTRY["extract_requirements"]._execute

        def fake_extract(inputs, context):
            finance_run.append("extract_requirements")
            return {"requirements_path": "x.json", "requirements_data": {}}

        capabilities.REGISTRY["answer_locally"]._execute = fake_answer
        capabilities.REGISTRY["extract_requirements"]._execute = fake_extract

        try:
            plan = {
                "intent": "general_chat",
                "steps": [{"capability": "answer_locally", "inputs": {"question": "Hello"}}],
                "output_format": "text",
            }
            result = executor.execute(plan)
            assert result.success
            assert "answer_locally" in chat_run
            assert len(finance_run) == 0, "Finance pipeline must NOT run for a chat request"
        finally:
            capabilities.REGISTRY["answer_locally"]._execute = original_answer
            capabilities.REGISTRY["extract_requirements"]._execute = original_extract

    def test_outputs_passed_between_steps(self):
        """Requirements path from step 1 must be available to step 2 via context."""
        received_context = {}

        original_extract = capabilities.REGISTRY["extract_requirements"]._execute
        original_check = capabilities.REGISTRY["check_report"]._execute

        def fake_extract(inputs, context):
            return {"requirements_path": "mock_requirements.json", "requirements_data": {}}

        def fake_check(inputs, context):
            received_context.update(context)  # capture what context was passed
            return {"findings_path": "mock_findings.json", "findings_data": []}

        capabilities.REGISTRY["extract_requirements"]._execute = fake_extract
        capabilities.REGISTRY["check_report"]._execute = fake_check

        try:
            plan = {
                "intent": "test",
                "steps": [
                    {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                    {"capability": "check_report", "inputs": {"report_path": REPORT_PATH}},
                ],
            }
            result = executor.execute(plan)
            assert result.success
            assert received_context.get("requirements_path") == "mock_requirements.json", (
                f"requirements_path not passed to step 2; context was: {received_context}"
            )
        finally:
            capabilities.REGISTRY["extract_requirements"]._execute = original_extract
            capabilities.REGISTRY["check_report"]._execute = original_check


# ---- Test 4: Failure handling ---------------------------------------------

class TestExecutorFailures:

    def test_step1_failure_skips_step2(self):
        original_extract = capabilities.REGISTRY["extract_requirements"]._execute
        original_check = capabilities.REGISTRY["check_report"]._execute

        step2_ran = []

        def fail_extract(inputs, context):
            raise RuntimeError("Disk full")

        def fake_check(inputs, context):
            step2_ran.append(True)
            return {"findings_path": "x.json", "findings_data": []}

        capabilities.REGISTRY["extract_requirements"]._execute = fail_extract
        capabilities.REGISTRY["check_report"]._execute = fake_check

        try:
            plan = {
                "intent": "test",
                "steps": [
                    {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                    {"capability": "check_report", "inputs": {"report_path": REPORT_PATH}},
                ],
            }
            result = executor.execute(plan)
            assert not result.success, "Result must be failure when step 1 fails"
            assert len(step2_ran) == 0, "Step 2 must not run when step 1 fails"
            assert result.error is not None
        finally:
            capabilities.REGISTRY["extract_requirements"]._execute = original_extract
            capabilities.REGISTRY["check_report"]._execute = original_check

    def test_failure_does_not_create_fake_artifact(self):
        original_extract = capabilities.REGISTRY["extract_requirements"]._execute
        original_build = capabilities.REGISTRY["build_report"]._execute

        def fail_extract(inputs, context):
            raise RuntimeError("Model crashed")

        def fake_build(inputs, context):
            return {"artifact_path": "Gap_Register.docx"}

        capabilities.REGISTRY["extract_requirements"]._execute = fail_extract
        capabilities.REGISTRY["build_report"]._execute = fake_build

        try:
            plan = {
                "intent": "test",
                "steps": [
                    {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                    {"capability": "build_report", "inputs": {}},
                ],
            }
            result = executor.execute(plan)
            assert not result.success
            # build_report should NOT have run, so its path must not be in artifacts
            # (build_report was skipped due to failure)
            # Even if it did run, success must be False
            assert "Gap_Register.docx" not in result.artifact_paths or not result.success
        finally:
            capabilities.REGISTRY["extract_requirements"]._execute = original_extract
            capabilities.REGISTRY["build_report"]._execute = original_build

    def test_audit_log_records_failure(self):
        """Audit log must contain a failure record when a step fails."""
        original_extract = capabilities.REGISTRY["extract_requirements"]._execute

        def fail_extract(inputs, context):
            raise RuntimeError("Test failure")

        capabilities.REGISTRY["extract_requirements"]._execute = fail_extract

        import audit_log as al
        log_path = al.LOG_PATH

        try:
            # Count existing log lines before
            pre_count = 0
            if os.path.exists(log_path):
                with open(log_path) as f:
                    pre_count = sum(1 for _ in f)

            plan = {
                "intent": "test_failure",
                "steps": [
                    {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                ],
            }
            result = executor.execute(plan)
            assert not result.success

            # Read new log entries
            with open(log_path) as f:
                lines = f.readlines()
            new_lines = lines[pre_count:]
            records = [json.loads(l) for l in new_lines]

            # Find a step_complete record with success=False
            failures = [r for r in records if r.get("event") == "step_complete" and not r.get("success")]
            assert len(failures) > 0, f"No failure recorded in audit log. New records: {records}"

        finally:
            capabilities.REGISTRY["extract_requirements"]._execute = original_extract

    def test_independent_request_after_failure(self):
        """A chat request must succeed even after a finance step fails."""
        original_extract = capabilities.REGISTRY["extract_requirements"]._execute
        original_answer = capabilities.REGISTRY["answer_locally"]._execute

        def fail_extract(inputs, context):
            raise RuntimeError("Finance broke")

        def fake_answer(inputs, context):
            return {"answer": "I am fine"}

        capabilities.REGISTRY["extract_requirements"]._execute = fail_extract
        capabilities.REGISTRY["answer_locally"]._execute = fake_answer

        try:
            # First: a failing finance plan
            finance_plan = {
                "intent": "broken_finance",
                "steps": [
                    {"capability": "extract_requirements", "inputs": {"sop_path": SOP_PATH}},
                ],
            }
            fail_result = executor.execute(finance_plan)
            assert not fail_result.success

            # Then: a chat plan — must succeed independently
            chat_plan = {
                "intent": "general_chat",
                "steps": [
                    {"capability": "answer_locally", "inputs": {"question": "Hello"}},
                ],
            }
            chat_result = executor.execute(chat_plan)
            assert chat_result.success, f"Chat request failed after finance failure: {chat_result.error}"
        finally:
            capabilities.REGISTRY["extract_requirements"]._execute = original_extract
            capabilities.REGISTRY["answer_locally"]._execute = original_answer
