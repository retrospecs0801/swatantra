"""
tests/test_ui_utils.py
----------------------
Unit tests for ui_utils: upload validation, sanitization, run workspaces,
and findings summarization.
"""

import sys, os, io, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pymupdf
import ui_utils

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOP_PATH = os.path.join(PROJECT_DIR, "sop.pdf")


class DummyUpload:
    """Mock Streamlit UploadedFile object."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.data = data

    def getvalue(self) -> bytes:
        return self.data


def _create_minimal_pdf_bytes() -> bytes:
    """Generate a minimal valid 1-page PDF in memory."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test Document")
    buf = doc.tobytes()
    doc.close()
    return buf


class TestUIUtils:

    def test_check_ollama_status(self):
        status = ui_utils.check_ollama_status()
        assert isinstance(status, dict)
        assert "available" in status
        assert "model_ready" in status
        assert "target_model" in status

    def test_sanitize_filename(self):
        assert ui_utils.sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
        assert ui_utils.sanitize_filename("my report (Q3) [final].pdf") == "my_report__Q3___final_.pdf"
        assert ui_utils.sanitize_filename("") == "uploaded_document.pdf"

    def test_create_run_workspace(self):
        tmp_base = tempfile.mkdtemp()
        try:
            run_dir = ui_utils.create_run_workspace(tmp_base)
            assert os.path.isdir(run_dir)
            assert "run_" in os.path.basename(run_dir)
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

    def test_validate_and_save_upload_valid_pdf(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            pdf_bytes = _create_minimal_pdf_bytes()
            mock_file = DummyUpload("company_sop.pdf", pdf_bytes)
            ok, path, err = ui_utils.validate_and_save_upload(mock_file, tmp_dir, "sop.pdf")
            assert ok is True
            assert err == ""
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_validate_and_save_upload_empty_file(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            mock_file = DummyUpload("empty.pdf", b"")
            ok, path, err = ui_utils.validate_and_save_upload(mock_file, tmp_dir, "sop.pdf")
            assert ok is False
            assert "empty" in err.lower()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_validate_and_save_upload_unsupported_extension(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            mock_file = DummyUpload("data.docx", b"some content")
            ok, path, err = ui_utils.validate_and_save_upload(mock_file, tmp_dir, "sop.pdf")
            assert ok is False
            assert "unsupported" in err.lower()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_validate_and_save_upload_corrupt_pdf(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            mock_file = DummyUpload("broken.pdf", b"%PDF-1.4 Not a real pdf content junk junk")
            ok, path, err = ui_utils.validate_and_save_upload(mock_file, tmp_dir, "sop.pdf")
            assert ok is False
            assert "corrupt" in err.lower() or "invalid" in err.lower()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_validate_and_save_upload_none(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            ok, path, err = ui_utils.validate_and_save_upload(None, tmp_dir, "sop.pdf")
            assert ok is False
            assert "no file" in err.lower()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_parse_findings_summary(self):
        findings = [
            {"id": "1", "clause": "2.1", "verdict": "Met"},
            {"id": "2", "clause": "2.2", "verdict": "Not met"},
            {"id": "3", "clause": "6.2", "verdict": "Not enough evidence", "tool_calls": [
                {"tool": "calculate", "expression": "0.26 / 18.60 * 100", "result": 1.3978, "success": True}
            ]},
            {"id": "4", "clause": "3.1", "verdict": "Not triggered"},
        ]
        summary = ui_utils.parse_findings_summary(findings)
        assert summary["total"] == 4
        assert summary["met"] == 1
        assert summary["not_met"] == 1
        assert summary["not_enough_evidence"] == 1
        assert summary["not_triggered"] == 1
        assert summary["has_calculations"] is True
        assert len(summary["calculations"]) == 1
        assert summary["calculations"][0]["expression"] == "0.26 / 18.60 * 100"
