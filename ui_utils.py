"""
ui_utils.py
-----------
Helper utilities for the Sovereign AI Workbench web UI.
Handles safe upload storage, file validation, Ollama reachability checks,
run workspace management, and findings summarization.
"""

import os, re, uuid, time, datetime, json
from typing import Tuple, Dict, Any, List, Optional
import pymupdf

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit


def check_ollama_status() -> Dict[str, Any]:
    """
    Check if the local Ollama instance is active and has the required model.
    Never raises an uncaught exception.
    """
    from llm import MODEL
    try:
        import ollama
        resp = ollama.list()
        models = [m.model for m in resp.models]
        # Match base model or tag (e.g. 'gemma4:e4b' or 'gemma4:e4b' in name)
        model_ready = any(MODEL in m or m.startswith(MODEL.split(":")[0]) for m in models)
        return {
            "available": True,
            "models": models,
            "target_model": MODEL,
            "model_ready": model_ready,
            "message": f"Ollama online with {MODEL}" if model_ready else f"Ollama online, but {MODEL} not found in {models}",
        }
    except Exception as exc:
        return {
            "available": False,
            "models": [],
            "target_model": "gemma4:e4b",
            "model_ready": False,
            "message": f"Ollama unavailable: {exc}",
        }


def create_run_workspace(base_dir: Optional[str] = None) -> str:
    """
    Create a unique per-run directory to isolate uploaded files and run artifacts.
    Prevents concurrent requests from overwriting one another.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    runs_dir = os.path.join(base_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    run_dir = os.path.join(runs_dir, f"run_{timestamp}_{unique_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and remove problematic characters."""
    base = os.path.basename(filename)
    clean = re.sub(r"[^\w\.-]", "_", base)
    return clean or "uploaded_document.pdf"


def validate_and_save_upload(
    uploaded_file,
    target_dir: str,
    target_filename: str,
) -> Tuple[bool, str, str]:
    """
    Validate uploaded file buffer and save safely into *target_dir* with *target_filename*.

    Checks:
    1. Buffer exists and has non-zero size
    2. Buffer within size limits
    3. File extension is .pdf
    4. Valid PDF readable by PyMuPDF

    Returns:
        (is_valid: bool, saved_path: str, error_message: str)
    """
    if uploaded_file is None:
        return False, "", "No file provided"

    # Extract bytes from Streamlit UploadedFile or file-like object
    try:
        if hasattr(uploaded_file, "getvalue"):
            content = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            content = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
        else:
            return False, "", "Invalid upload object"
    except Exception as exc:
        return False, "", f"Failed to read file: {exc}"

    if not content or len(content) == 0:
        return False, "", "File is empty (0 bytes)"

    if len(content) > MAX_UPLOAD_BYTES:
        return False, "", f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"

    # Check extension
    orig_name = getattr(uploaded_file, "name", "document.pdf")
    ext = os.path.splitext(orig_name)[1].lower()
    if ext != ".pdf":
        return False, "", f"Unsupported file type '{ext}'. Only .pdf documents are supported."

    # Validate PDF structure via PyMuPDF
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = len(doc)
        if page_count == 0:
            return False, "", "PDF contains no pages"
    except Exception as exc:
        return False, "", f"Corrupted or invalid PDF file: {exc}"

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    safe_name = sanitize_filename(target_filename)
    dest_path = os.path.join(target_dir, safe_name)

    try:
        with open(dest_path, "wb") as f:
            f.write(content)
        return True, os.path.abspath(dest_path), ""
    except Exception as exc:
        return False, "", f"Failed to save file: {exc}"


def parse_findings_summary(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize findings data for UI presentation.
    Counts verdicts and checks for tool calculation records.
    """
    total = len(findings)
    counts = {
        "Met": 0,
        "Not met": 0,
        "Not enough evidence": 0,
        "Not triggered": 0,
    }
    calculations_used = []

    for f in findings:
        v = f.get("verdict", "Unknown")
        counts[v] = counts.get(v, 0) + 1
        if f.get("tool_calls"):
            for tc in f["tool_calls"]:
                calculations_used.append({
                    "clause": f.get("clause"),
                    "requirement_id": f.get("id"),
                    "tool": tc.get("tool"),
                    "expression": tc.get("expression"),
                    "result": tc.get("result"),
                    "success": tc.get("success"),
                })

    return {
        "total": total,
        "not_met": counts.get("Not met", 0),
        "not_enough_evidence": counts.get("Not enough evidence", 0),
        "met": counts.get("Met", 0),
        "not_triggered": counts.get("Not triggered", 0),
        "counts": counts,
        "calculations": calculations_used,
        "has_calculations": len(calculations_used) > 0,
    }


def read_audit_log_entries(limit: int = 50) -> List[Dict[str, Any]]:
    """Read recent audit records from audit.jsonl safely."""
    from audit_log import LOG_PATH
    if not os.path.exists(LOG_PATH):
        return []
    records = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records[-limit:]
    except Exception:
        return []
