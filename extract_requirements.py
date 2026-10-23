"""
extract_requirements.py
-----------------------
Reads the company SOP PDF and extracts structured compliance requirements.

Callable as a library function (run) or as a CLI script.
Deterministic overrides for clauses 3.1 and 8.2 are applied after model
extraction so they survive every regeneration.
"""

import pymupdf, json, os
from llm import ask_json

# ---------------------------------------------------------------------------
# Deterministic post-processing overrides.
# These corrections always win over model output so results survive regeneration.
#
# FULL OVERRIDE TABLE (derived from human review of the SOP):
#   2.1  checkable=True  — expense register entries are visible in quarterly reports
#   2.2  checkable=True  — second-approver field is visible in the expense register
#   2.3  checkable=True  — Audit Committee resolution reference is in the register
#   3.1  checkable=False — permits advances ≤20%; not an independent compliance duty
#   3.2  checkable=True  — justification and CFO approval appear in the advance register
#   4.1  checkable=True  — reconciliation status is disclosed in the quarterly report
#   5.1  checkable=True  — related-party transactions are disclosed with approval ref
#   5.2  checkable=True  — RPT section is present/absent in the quarterly report
#   6.1  checkable=True  — inventory verification is reported in the quarterly report
#   6.2  checkable=True  — variance and CFO approval appear in the quarterly report
#   7.1  checkable=True  — contingent liabilities section is in the quarterly report
#   8.1  checkable=True  — signatories are present on the report
#   8.2  checkable=True  — Internal Audit sign-off date is in the report
# ---------------------------------------------------------------------------
CLAUSE_OVERRIDES = {
    "2.1": {"checkable": True},
    "2.2": {"checkable": True},
    "2.3": {"checkable": True},
    "3.1": {"checkable": False},  # NOT an independent compliance duty
    "3.2": {"checkable": True},
    "4.1": {"checkable": True},
    "5.1": {"checkable": True},
    "5.2": {"checkable": True},
    "6.1": {"checkable": True},
    "6.2": {"checkable": True},
    "7.1": {"checkable": True},
    "8.1": {"checkable": True},
    "8.2": {"checkable": True},
}


def read_pdf(path: str) -> str:
    """Return the full text of a PDF, page-tagged."""
    doc = pymupdf.open(path)
    return "\n".join(f"[Page {i+1}]\n{p.get_text()}" for i, p in enumerate(doc))


def run(sop_path: str, output_path: str = "requirements.json") -> dict:
    """
    Extract requirements from *sop_path* and write them to *output_path*.

    Returns the parsed requirements dict.
    Raises on model failure or unreadable PDF.
    """
    sop_text = read_pdf(sop_path)

    prompt = f"""You are reading a company's internal Finance SOP.
Extract every individual obligation or rule (must / shall / required / prohibited / may) as one entry.
Rules:
- Copy the clause number exactly. If a clause has several obligations, make one entry per obligation.
- Include all clauses 2.1 through 8.2.
- "threshold": copy the limit EXACTLY as written (e.g. "Rs 5 lakh", "20%", "30 days"). Do NOT convert units. Use null if none.
- "checkable": true if it can be verified from a quarterly financial report (e.g. expenses, advances, bank reconciliations, related parties, inventory verification, contingent liabilities, signatories); false only if it is solely about internal process not recorded in the report.
- "definitions": list clauses that define a term used by the requirements (e.g. who counts as a related party).
- Use ONLY the text below.

Return JSON in exactly this shape:
{{"requirements": [{{"clause": "2.1", "requirement": "...", "threshold": "Rs 5 lakh", "checkable": true}}],
 "definitions": [{{"clause": "5.1", "term": "related party", "meaning": "..."}}]}}

SOP TEXT:
{sop_text}"""

    data = ask_json(prompt)

    # Assign unique IDs (e.g. 2.1-a, 2.1-b for multiple obligations per clause)
    counts: dict = {}
    for r in data["requirements"]:
        counts[r["clause"]] = counts.get(r["clause"], 0) + 1
        r["id"] = f'{r["clause"]}-{chr(96 + counts[r["clause"]])}'

    # Apply deterministic overrides — these always win over model output
    for r in data["requirements"]:
        if r.get("clause") in CLAUSE_OVERRIDES:
            r.update(CLAUSE_OVERRIDES[r["clause"]])

    json.dump(data, open(output_path, "w"), indent=2, ensure_ascii=False)
    print(f"[extract_requirements] Wrote {len(data['requirements'])} requirements → {output_path}")
    return data


if __name__ == "__main__":
    import sys
    sop = sys.argv[1] if len(sys.argv) > 1 else "sop.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "requirements.json"
    result = run(sop, out)
    print(json.dumps(result, indent=2, ensure_ascii=False))
