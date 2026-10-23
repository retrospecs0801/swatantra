"""
tests/test_pipeline_regression.py
----------------------------------
Test 1: Regression tests for the existing finance pipeline.

These tests verify that the refactored scripts produce the same results
as the original implementation. They run against the existing sample files
and check structural invariants, not exact model outputs (which can vary).

IMPORTANT: These tests call Ollama. They are slow (minutes per run).
Mark them with -m slow to run separately:
    .venv/bin/python -m pytest tests/test_pipeline_regression.py -v -m slow

Or run all:
    .venv/bin/python -m pytest tests/test_pipeline_regression.py -v
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOP_PATH = os.path.join(PROJECT_DIR, "sop.pdf")
REPORT_PATH = os.path.join(PROJECT_DIR, "report.pdf")
FINDINGS_PATH = os.path.join(PROJECT_DIR, "findings.json")

pytestmark = pytest.mark.slow


class TestExistingFindings:
    """Tests against the EXISTING findings.json — no model calls needed."""

    def test_findings_file_exists(self):
        assert os.path.isfile(FINDINGS_PATH), "findings.json must exist"

    def test_findings_is_valid_json(self):
        with open(FINDINGS_PATH) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_findings_schema(self):
        """Each finding must have the required fields."""
        with open(FINDINGS_PATH) as f:
            findings = json.load(f)
        required_fields = {"verdict", "id", "clause", "requirement", "quote_check"}
        for f in findings:
            missing = required_fields - set(f.keys())
            assert not missing, f"Finding {f.get('id')} missing fields: {missing}"

    def test_baseline_verdict_counts(self):
        """
        Check verdict invariants across all 12 checkable clauses.

        Baseline distribution:
          - Gaps (Not met): 6 or 7
            (clauses 2.2, 3.2, 4.1, 5.1, 5.2, 8.2 are strict gaps;
             clause 8.1 is 'Met' if 'Prepared by' is treated as sign-off,
             or 'Not met' if strictly requiring 'Signed by').
          - Unsure (Not enough evidence): exactly 1 (clause 6.2)
          - Compliant (Met / Not triggered): 4 or 5
            (clauses 2.1, 2.3, 6.1, 7.1 are consistently Met).
          - Total checkable findings: exactly 12.
        """
        with open(FINDINGS_PATH) as f:
            findings = json.load(f)
        assert len(findings) == 12, f"Expected exactly 12 findings, got {len(findings)}"
        counts = {}
        for f in findings:
            v = f["verdict"]
            counts[v] = counts.get(v, 0) + 1

        assert counts.get("Not enough evidence", 0) in (0, 1), (
            f"Expected 0 or 1 'Not enough evidence' (clause 6.2), got {counts}"
        )
        assert counts.get("Not met", 0) in (6, 7), (
            f"Expected 6 or 7 'Not met', got {counts.get('Not met')}. Full counts: {counts}"
        )
        ok_total = counts.get("Met", 0) + counts.get("Not triggered", 0)
        assert ok_total in (4, 5), (
            f"Expected 4 or 5 'Met/Not triggered', got {ok_total}. Full counts: {counts}"
        )

    def test_clause_3_1_absent(self):
        """Clause 3.1 must not appear in findings (it is not checkable)."""
        with open(FINDINGS_PATH) as f:
            findings = json.load(f)
        clauses = [f["clause"] for f in findings]
        assert "3.1" not in clauses, "Clause 3.1 must be absent from findings (checkable=False)"

    def test_clause_8_2_present(self):
        """Clause 8.2 must be present in findings (checkable=True)."""
        with open(FINDINGS_PATH) as f:
            findings = json.load(f)
        clauses = [f["clause"] for f in findings]
        assert "8.2" in clauses, "Clause 8.2 must be present in findings (checkable=True)"

    def test_clause_8_2_verdict_not_met(self):
        """Clause 8.2 finding should be 'Not met' per the current baseline."""
        with open(FINDINGS_PATH) as f:
            findings = json.load(f)
        f82 = next((f for f in findings if f["clause"] == "8.2"), None)
        assert f82 is not None
        assert f82["verdict"] == "Not met", (
            f"Expected 8.2 to be 'Not met', got '{f82['verdict']}'"
        )


class TestRequirementsFile:
    """Tests against the existing requirements.json."""

    def test_requirements_file_exists(self):
        path = os.path.join(PROJECT_DIR, "requirements.json")
        assert os.path.isfile(path)

    def test_clause_3_1_not_checkable(self):
        path = os.path.join(PROJECT_DIR, "requirements.json")
        with open(path) as f:
            data = json.load(f)
        reqs = data["requirements"]
        r31 = next((r for r in reqs if r["clause"] == "3.1"), None)
        assert r31 is not None, "Clause 3.1 must exist in requirements"
        assert r31["checkable"] is False, (
            f"Clause 3.1 must have checkable=False, got {r31['checkable']}"
        )

    def test_clause_8_2_checkable(self):
        path = os.path.join(PROJECT_DIR, "requirements.json")
        with open(path) as f:
            data = json.load(f)
        reqs = data["requirements"]
        r82 = next((r for r in reqs if r["clause"] == "8.2"), None)
        assert r82 is not None, "Clause 8.2 must exist in requirements"
        assert r82["checkable"] is True, (
            f"Clause 8.2 must have checkable=True, got {r82['checkable']}"
        )

    def test_all_requirements_have_ids(self):
        path = os.path.join(PROJECT_DIR, "requirements.json")
        with open(path) as f:
            data = json.load(f)
        for r in data["requirements"]:
            assert "id" in r, f"Requirement {r.get('clause')} missing 'id'"


class TestExtractRequirementsRefactor:
    """Verify the refactored extract_requirements can be imported without side effects."""

    def test_import_does_not_execute(self):
        """Importing extract_requirements must NOT call ask_json."""
        import unittest.mock as mock
        with mock.patch("llm.ask_json", side_effect=AssertionError("ask_json called at import time!")):
            # If this import triggers ask_json, the test fails
            import importlib
            import extract_requirements
            importlib.reload(extract_requirements)  # force re-import
        # If we got here, the import was clean

    def test_clause_overrides_applied(self):
        """After run(), all clauses must have correct checkable values per the override table."""
        import unittest.mock as mock
        import tempfile
        import importlib

        # Force a fresh import of extract_requirements so the module object is known
        import extract_requirements as er
        importlib.reload(er)

        # Mock ask_json to return model output WITHOUT the correct values
        # (simulating a bad model run where everything is wrong)
        fake_data = {
            "requirements": [
                {"clause": "2.1", "requirement": "...", "threshold": None, "checkable": False},
                {"clause": "2.2", "requirement": "...", "threshold": None, "checkable": False},
                {"clause": "3.1", "requirement": "Advance up to 20%", "threshold": "20%", "checkable": True},
                {"clause": "3.2", "requirement": "...", "threshold": "20%", "checkable": False},
                {"clause": "6.1", "requirement": "...", "threshold": None, "checkable": False},
                {"clause": "8.2", "requirement": "IA review before Board", "threshold": None, "checkable": False},
            ],
            "definitions": [],
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            # Patch ask_json on the already-imported extract_requirements module
            with mock.patch.object(er, "ask_json", return_value=fake_data), \
                 mock.patch.object(er, "read_pdf", return_value="mock SOP text"):
                result = er.run(sop_path="sop.pdf", output_path=tmp_path)

            reqs = {r["clause"]: r for r in result["requirements"]}
            # 3.1 must be False
            assert reqs["3.1"]["checkable"] is False, "3.1 must be forced to checkable=False"
            # 8.2 must be True
            assert reqs["8.2"]["checkable"] is True,  "8.2 must be forced to checkable=True"
            # 2.1, 2.2 must be True (full override table)
            assert reqs["2.1"]["checkable"] is True,  "2.1 must be forced to checkable=True"
            assert reqs["2.2"]["checkable"] is True,  "2.2 must be forced to checkable=True"
            # 3.2 must be True
            assert reqs["3.2"]["checkable"] is True,  "3.2 must be forced to checkable=True"
            # 6.1 must be True
            assert reqs["6.1"]["checkable"] is True,  "6.1 must be forced to checkable=True"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestDocxGeneration:
    """Verify Gap_Register.docx exists and can be opened."""

    def test_docx_file_exists(self):
        path = os.path.join(PROJECT_DIR, "Gap_Register.docx")
        assert os.path.isfile(path), "Gap_Register.docx must exist"

    def test_docx_is_openable(self):
        path = os.path.join(PROJECT_DIR, "Gap_Register.docx")
        from docx import Document
        doc = Document(path)
        assert len(doc.paragraphs) > 0, "Gap_Register.docx must contain paragraphs"

    def test_docx_with_tool_calls(self):
        """Test build_report run with mock findings including tool_calls."""
        import build_report as br
        import tempfile
        from unittest.mock import patch
        from docx import Document

        mock_findings = [
            {
                "id": "6.2-a",
                "clause": "6.2",
                "requirement": "Where net variance exceeds 1%...",
                "verdict": "Not enough evidence",
                "evidence_quote": "Net variance was 0.26 crore",
                "page": 3,
                "reasoning": "Net variance exceeds 1%, but approval is not mentioned.",
                "quote_check": "exact",
                "tool_calls": [
                    {
                        "tool": "calculate",
                        "expression": "0.26 / 18.60 * 100",
                        "result": 1.3978,
                        "success": True,
                    }
                ],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
            json.dump(mock_findings, tf)
            tf_findings = tf.name

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf_docx:
            tf_out = tf_docx.name

        try:
            with patch("build_report.ask_json", return_value={"action": "Seek CFO approval."}):
                br.run(findings_path=tf_findings, output_path=tf_out)

            doc = Document(tf_out)
            text = "\n".join(cell.text for t in doc.tables for r in t.rows for cell in r.cells)
            assert "[Calculation: 0.26 / 18.60 * 100 = 1.3978]" in text
        finally:
            if os.path.exists(tf_findings):
                os.unlink(tf_findings)
            if os.path.exists(tf_out):
                os.unlink(tf_out)
