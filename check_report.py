"""
check_report.py
---------------
Compares extracted SOP requirements against a financial report PDF.
Supports controlled, bounded tool calls (e.g. safe calculation tool).

Callable as a library function (run) or as a CLI script.
Findings schema: verdict, evidence_quote, page, reasoning, quote_check,
                 id, clause, requirement, and optional tool_calls.
"""

import json, re, time, pymupdf
from llm import ask_json
from calculator import safe_calculate
import audit_log

MAX_TOOL_CALLS_PER_REQ = 3


def read_pdf(path: str) -> str:
    """Return the full text of a PDF, page-tagged."""
    doc = pymupdf.open(path)
    return "\n".join(f"[Page {i+1}]\n{p.get_text()}" for i, p in enumerate(doc))


def verify_quote(quote: str, report_norm: str) -> str:
    """
    Check whether *quote* appears in the normalised report text.
    Returns 'exact', 'partial', or 'missing'.
    """
    norm = lambda s: re.sub(r"\s+", " ", s).lower()
    q = norm(quote)
    if not q:
        return "missing"
    if q in report_norm:
        return "exact"
    # Accept slash-separated or ellipsis-joined multi-part quotes
    parts = [p.strip() for p in re.split(r"\s/\s|\.\.\.|…", q) if p.strip()]
    if len(parts) > 1 and all(p in report_norm for p in parts):
        return "exact"
    words = re.findall(r"\w+", q)
    report_words = set(re.findall(r"\w+", report_norm))
    coverage = sum(w in report_words for w in words) / max(len(words), 1)
    return "partial" if coverage >= 0.9 else "missing"


def check_single_requirement(
    r: dict,
    report: str,
    report_norm: str,
    definitions: list,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_REQ,
    tool_callback=None,
) -> dict:
    """
    Evaluate one SOP requirement against the report using a bounded tool-use loop.
    Gemma 4 can optionally call 'calculate' before producing its final audit verdict.
    """
    tool_history = []
    tool_calls_count = 0

    initial_prompt = f"""You are an auditor checking a financial report against ONE rule from the company SOP.

REPORT:
{report}

--- END OF REPORT ---

DEFINITIONS: {json.dumps(definitions, ensure_ascii=False)}

RULE (SOP clause {r['clause']}): {r['requirement']}
Threshold: {r.get('threshold')}

TOOLS AVAILABLE:
- "calculate": performs safe arithmetic operations.
  Args: {{"expression": "<math expression>"}}
  Example request: {{"tool": "calculate", "args": {{"expression": "0.26 / 18.60 * 100"}}}}

INSTRUCTIONS:
1. If you need to verify or calculate any numerical value, threshold, ratio, or percentage difference, request the "calculate" tool:
   Return JSON: {{"tool": "calculate", "args": {{"expression": "..."}}}}
2. Otherwise, return your audit verdict directly:
   Return JSON: {{"verdict": "Met | Not met | Not triggered | Not enough evidence", "evidence_quote": "short verbatim quote, max 12 words", "page": 1, "reasoning": "one sentence"}}

VERDICT RULES:
- "Met": the report shows the rule was followed.
- "Not met": the report shows the rule was broken (a recorded breach, empty approval field, or section visibly missing).
- "Not triggered": the rule only applies above a threshold/condition, and the report shows it was not reached.
- "Not enough evidence": the report does not state whether something required occurred (e.g. an approval that is not mentioned anywhere). Silence in the report is NOT a breach."""

    current_prompt = initial_prompt

    while tool_calls_count < max_tool_calls:
        out = ask_json(current_prompt)

        # 1. Model requested a tool
        if isinstance(out, dict) and "tool" in out:
            tool_name = out.get("tool")
            args = out.get("args", {})
            tool_calls_count += 1

            if tool_callback:
                tool_callback("requested", {"tool": tool_name, "args": args, "req_id": r["id"]})
            else:
                print(f"  [MODEL] Requested tool: {tool_name}")

            t0 = time.monotonic()

            if tool_name == "calculate":
                expr = args.get("expression", "") if isinstance(args, dict) else str(args)
                if tool_callback:
                    tool_callback("validating", {"expression": expr, "req_id": r["id"]})
                else:
                    print(f"  [TOOL] Validating expression: {expr}")

                calc_res = safe_calculate(expr)
                duration = time.monotonic() - t0

                if tool_callback:
                    tool_callback("result", {"result": calc_res, "req_id": r["id"], "duration": duration})
                else:
                    if calc_res["success"]:
                        print(f"  [TOOL] Result: {calc_res['result']} ({calc_res['formatted']})")
                    else:
                        print(f"  [TOOL] Error: {calc_res['error']}")

                # Audit logging
                audit_log.log_tool_call(
                    tool_name="calculate",
                    args={"expression": expr},
                    result=calc_res["result"],
                    duration=duration,
                    requirement_id=r["id"],
                    success=calc_res["success"],
                    error=calc_res["error"],
                )

                tool_history.append({
                    "tool": "calculate",
                    "expression": expr,
                    "result": calc_res["result"],
                    "success": calc_res["success"],
                })

                if calc_res["success"]:
                    tool_feedback = (
                        f"Calculation result for \"{expr}\": {calc_res['result']} "
                        f"(formatted: {calc_res['formatted']}).\n"
                        "Now use this result to determine your audit verdict.\n"
                        "Remember: if the threshold condition occurred but the report is completely "
                        "silent on whether required approval was obtained, the verdict must be "
                        "\"Not enough evidence\" (silence is NOT a breach)."
                    )
                else:
                    tool_feedback = (
                        f"Calculation failed for \"{expr}\": {calc_res['error']}.\n"
                        "Please evaluate using the textual evidence in the report."
                    )

                current_prompt = (
                    f"{initial_prompt}\n\n"
                    f"--- TOOL RESULT (Call {tool_calls_count}/{max_tool_calls}) ---\n"
                    f"{tool_feedback}\n\n"
                    f"Return your final verdict JSON: "
                    f'{{"verdict": "...", "evidence_quote": "...", "page": 1, "reasoning": "..."}}'
                )

            else:
                duration = time.monotonic() - t0
                err_msg = f"Unknown tool '{tool_name}'. Available tools: calculate."
                if tool_callback:
                    tool_callback("error", {"error": err_msg, "req_id": r["id"]})
                else:
                    print(f"  [TOOL] Error: {err_msg}")

                audit_log.log_tool_call(
                    tool_name=tool_name,
                    args=args,
                    result=None,
                    duration=duration,
                    requirement_id=r["id"],
                    success=False,
                    error=err_msg,
                )

                current_prompt = (
                    f"{initial_prompt}\n\n"
                    f"Tool error: {err_msg}.\n"
                    f"Return your final verdict JSON: "
                    f'{{"verdict": "...", "evidence_quote": "...", "page": 1, "reasoning": "..."}}'
                )

        # 2. Model returned the final audit verdict
        elif isinstance(out, dict) and "verdict" in out:
            if tool_callback:
                tool_callback("producing_finding", {"verdict": out.get("verdict"), "req_id": r["id"]})
            else:
                print(f"  [MODEL] Producing finding")

            out["quote_check"] = verify_quote(out.get("evidence_quote", ""), report_norm)
            out.update(id=r["id"], clause=r["clause"], requirement=r["requirement"])
            if tool_history:
                out["tool_calls"] = tool_history
            return out

        # 3. Model returned unexpected shape — retry prompt once
        else:
            tool_calls_count += 1
            current_prompt = (
                f"{initial_prompt}\n\n"
                f"Please return valid JSON with either a tool request or final verdict."
            )

    # If loop exhausted without a verdict, force one final prompt specifically for verdict
    forced_prompt = (
        f"{initial_prompt}\n\n"
        f"Tool call limit reached ({max_tool_calls}). Provide your final audit verdict now without further tool calls.\n"
        f'Return JSON: {{"verdict": "...", "evidence_quote": "...", "page": 1, "reasoning": "..."}}'
    )
    final_out = ask_json(forced_prompt)
    final_out["quote_check"] = verify_quote(final_out.get("evidence_quote", ""), report_norm)
    final_out.update(id=r["id"], clause=r["clause"], requirement=r["requirement"])
    if tool_history:
        final_out["tool_calls"] = tool_history
    return final_out


def run(
    report_path: str,
    requirements_path: str = "requirements.json",
    output_path: str = "findings.json",
    progress_callback=None,
    tool_callback=None,
) -> list:
    """
    Check *report_path* against requirements in *requirements_path*.
    Writes findings to *output_path* and returns the findings list.

    *progress_callback(step, total, req_id)* is called before each model call.
    *tool_callback(event_type, payload)* is called on tool events.
    """
    report = read_pdf(report_path)
    norm = lambda s: re.sub(r"\s+", " ", s).lower()
    report_norm = norm(report)

    reqs = json.load(open(requirements_path))
    definitions = reqs.get("definitions", [])
    findings = []

    checkable = [r for r in reqs["requirements"] if r.get("checkable", True)]
    total = len(checkable)

    for step, r in enumerate(checkable, start=1):
        if progress_callback:
            progress_callback(step, total, r["id"])

        out = check_single_requirement(
            r=r,
            report=report,
            report_norm=report_norm,
            definitions=definitions,
            tool_callback=tool_callback,
        )
        findings.append(out)

        mark = {"exact": "✓", "partial": "~ partial", "missing": "✗ quote not found"}.get(
            out.get("quote_check"), "✗"
        )
        print(f"  {r['id']:8} {out['verdict']:20} {mark}")
        if out.get("quote_check") == "missing":
            print("           evidence needs manual check")
            print("           quote was:", out.get("evidence_quote"))

    json.dump(findings, open(output_path, "w"), indent=2, ensure_ascii=False)
    print(f"[check_report] Wrote {len(findings)} findings → {output_path}")
    return findings


if __name__ == "__main__":
    import sys
    report = sys.argv[1] if len(sys.argv) > 1 else "report.pdf"
    reqs   = sys.argv[2] if len(sys.argv) > 2 else "requirements.json"
    out    = sys.argv[3] if len(sys.argv) > 3 else "findings.json"
    result = run(report, reqs, out)
    print(f"\n{len(result)} findings written.")
