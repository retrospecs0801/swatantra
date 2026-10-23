"""
build_report.py
---------------
Reads findings.json and produces a Gap_Register.docx.

Callable as a library function (run) or as a CLI script.
Output is always labelled as a DRAFT for human review.
"""

import json
from datetime import date
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from llm import ask_json

COLORS = {
    "Not met": "F4B6B6",
    "Not enough evidence": "FFE699",
    "Met": "C6E0B4",
    "Not triggered": "D9D9D9",
}


def _shade(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _follow_up(f: dict) -> str:
    out = ask_json(
        f"""An auditor found a possible non-compliance with a company SOP.
SOP rule (clause {f['clause']}): {f['requirement']}
Verdict: {f['verdict']}
Reason: {f.get('reasoning', '')}
Suggest ONE concrete follow-up action in under 25 words. Name the responsible role
(Finance Manager, CFO, Internal Audit or Audit Committee). Make no safety or legal conclusions.
Return JSON: {{"action": "..."}}"""
    )
    return out.get("action", "Review manually.")


def _evidence_text(f: dict) -> str:
    t = f'"{f.get("evidence_quote", "")}" (page {f.get("page", "?")})'
    if f.get("tool_calls"):
        tool_details = [
            f"{tc.get('expression')} = {round(tc.get('result'), 4) if isinstance(tc.get('result'), float) else tc.get('result')}"
            for tc in f["tool_calls"]
            if tc.get("success") and "expression" in tc and "result" in tc
        ]
        if tool_details:
            t += f"\n[Calculation: {', '.join(tool_details)}]"
    qc = f.get("quote_check")
    if qc == "missing":
        t += "\n⚠ Quote not found in report. Verify manually."
    elif qc == "partial":
        t += "\n~ Quote matched in part. Verify wording."
    return t


def _make_table(doc: Document, items: list, followups: dict = None) -> None:
    cols = ["Clause", "Requirement", "Status", "Evidence (from report)"]
    if followups is not None:
        cols.append("Suggested follow-up")
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = "Table Grid"
    for i, c in enumerate(cols):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(c).bold = True
        _shade(cell, "BDD7EE")
    for f in items:
        row = t.add_row().cells
        row[0].text = f["clause"]
        row[1].text = f["requirement"]
        row[2].text = f["verdict"]
        _shade(row[2], COLORS.get(f["verdict"], "FFFFFF"))
        row[3].text = _evidence_text(f)
        if followups is not None:
            row[4].text = followups[f["id"]]


def run(
    findings_path: str = "findings.json",
    output_path: str = "Gap_Register.docx",
    progress_callback=None,
) -> str:
    """
    Build a Gap Register DOCX from *findings_path* and write it to *output_path*.
    Returns the resolved output path string.

    *progress_callback(item_id)* is called before each follow-up model call.
    """
    findings = json.load(open(findings_path))
    gaps   = [f for f in findings if f["verdict"] == "Not met"]
    unsure = [f for f in findings if f["verdict"] == "Not enough evidence"]
    ok     = [f for f in findings if f["verdict"] in ("Met", "Not triggered")]

    # One small model call per flagged item
    followups: dict = {}
    flagged = gaps + unsure
    for f in flagged:
        if progress_callback:
            progress_callback(f["id"])
        print(f"  follow-up for {f['id']}")
        followups[f["id"]] = _follow_up(f)

    # Build document
    doc = Document()
    doc.styles["Normal"].font.size = Pt(9.5)
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, m, Inches(0.7))

    doc.add_heading("Finance Gap Register: Q2 FY2026-27", 0)
    p = doc.add_paragraph()
    p.add_run("DRAFT for human review. ").bold = True
    p.add_run(
        f"Generated locally on {date.today():%d %B %Y} by gemma4:e4b. "
        "No document data left this machine."
    )

    doc.add_heading("1. Executive summary", 1)
    doc.add_paragraph(
        f"{len(findings)} SOP requirements were checked against the quarterly report: "
        f"{len(gaps)} not met, {len(unsure)} could not be determined from the report, "
        f"and {len(ok)} met or not applicable."
    )
    for f in gaps:
        doc.add_paragraph(f"Clause {f['clause']}: {f['reasoning']}", style="List Bullet")

    doc.add_heading("2. Gap register", 1)
    _make_table(doc, gaps + unsure, followups)

    doc.add_heading("3. Human review checklist", 1)
    for f in gaps:
        doc.add_paragraph(
            f"\u2610 Clause {f['clause']}: verify the evidence and confirm the gap with the responsible owner."
        )
    for f in unsure:
        doc.add_paragraph(
            f"\u2610 Clause {f['clause']}: obtain the missing record, then re-assess."
        )
    doc.add_paragraph("\u2610 Verify every quote marked \u26a0 or ~ against the source report.")
    doc.add_paragraph("\u2610 Reviewer approval: ____________________    Date: ____________")

    doc.add_heading("4. Items found compliant or not applicable", 1)
    _make_table(doc, ok)

    doc.save(output_path)
    print(f"[build_report] Saved {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    findings = sys.argv[1] if len(sys.argv) > 1 else "findings.json"
    output   = sys.argv[2] if len(sys.argv) > 2 else "Gap_Register.docx"
    run(findings, output)
