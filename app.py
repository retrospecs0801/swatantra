"""
app.py
------
Sovereign AI Workbench — Streamlit Web Interface (Phase 3.1)
Private, local-first AI workflows for confidential documents.

Launch command:
    .venv/bin/streamlit run app.py
"""

import os, sys, json, time, io
from datetime import datetime

# Ensure project directory is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import streamlit as st
import planner as planner_mod
import validator as validator_mod
import executor as executor_mod
import audit_log
from llm import MODEL
import ui_utils

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sovereign AI Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for high polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .badge-status {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
    .badge-yellow { background-color: #FEF9C3; color: #854D0E; }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-gray { background-color: #F1F5F9; color: #475569; }
    .log-box {
        font-family: monospace;
        font-size: 0.85rem;
        background-color: #0F172A;
        color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .trace-model { color: #38BDF8; }
    .trace-tool { color: #FBBF24; font-weight: bold; }
    .trace-step { color: #4ADE80; }
    .trace-error { color: #F87171; font-weight: bold; }
    .trace-status { color: #A78BFA; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "run_dir" not in st.session_state:
    st.session_state.run_dir = None
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
if "execution_result" not in st.session_state:
    st.session_state.execution_result = None
if "execution_events" not in st.session_state:
    st.session_state.execution_events = []
if "findings_data" not in st.session_state:
    st.session_state.findings_data = None
if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = None
if "active_files" not in st.session_state:
    st.session_state.active_files = []
if "selected_request" not in st.session_state:
    st.session_state.selected_request = ""


# ---------------------------------------------------------------------------
# Sidebar: System Status & File Selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Sovereign AI Workbench")
    st.caption("Private, local-first AI workflows")

    # 1. System & Model Status
    st.markdown("---")
    st.markdown("#### System Engine")
    ollama_info = ui_utils.check_ollama_status()
    if ollama_info["available"]:
        if ollama_info["model_ready"]:
            st.markdown(f'<span class="badge-status badge-green">🟢 Ollama Online</span>', unsafe_allow_html=True)
            st.caption(f"Model: `{ollama_info['target_model']}` (Apple Metal)")
        else:
            st.markdown(f'<span class="badge-status badge-yellow">🟡 Model Missing</span>', unsafe_allow_html=True)
            st.caption(f"Expected `{ollama_info['target_model']}`")
    else:
        st.markdown(f'<span class="badge-status badge-red">🔴 Ollama Offline</span>', unsafe_allow_html=True)
        st.caption("Run `ollama serve` in a terminal.")

    # 2. Interaction Mode
    st.markdown("---")
    st.markdown("#### Interaction Mode")
    mode = st.radio(
        "Workflow routing mode:",
        ["Auto (Inferred)", "General Chat", "Document Analysis"],
        help="Auto: model decides. General Chat: answers locally without finance pipeline. Document Analysis: forces compliance pipeline.",
    )

    # 3. Document Source & Uploads
    st.markdown("---")
    st.markdown("#### Document Selection")
    use_sample = st.checkbox("Use Demo Sample Files", value=True, help="Loads local sop.pdf and report.pdf from repository.")

    sop_path = None
    report_path = None

    if use_sample:
        sample_sop = os.path.join(PROJECT_DIR, "sop.pdf")
        sample_report = os.path.join(PROJECT_DIR, "report.pdf")
        if os.path.isfile(sample_sop) and os.path.isfile(sample_report):
            sop_path = sample_sop
            report_path = sample_report
            st.success("✓ Loaded sample `sop.pdf` & `report.pdf`")
        else:
            st.warning("Sample files not found in workspace.")
    else:
        st.caption("Upload confidential PDFs (validated locally):")
        uploaded_sop = st.file_uploader("1. Company SOP (PDF)", type=["pdf"], key="upload_sop")
        uploaded_report = st.file_uploader("2. Quarterly Financial Report (PDF)", type=["pdf"], key="upload_report")

        if uploaded_sop or uploaded_report:
            if st.session_state.run_dir is None:
                st.session_state.run_dir = ui_utils.create_run_workspace(PROJECT_DIR)

            if uploaded_sop:
                ok, path, err = ui_utils.validate_and_save_upload(uploaded_sop, st.session_state.run_dir, "sop.pdf")
                if ok:
                    sop_path = path
                    st.caption(f"✓ SOP: `{os.path.basename(path)}` ({len(uploaded_sop.getvalue())//1024} KB)")
                else:
                    st.error(f"SOP Error: {err}")

            if uploaded_report:
                ok, path, err = ui_utils.validate_and_save_upload(uploaded_report, st.session_state.run_dir, "report.pdf")
                if ok:
                    report_path = path
                    st.caption(f"✓ Report: `{os.path.basename(path)}` ({len(uploaded_report.getvalue())//1024} KB)")
                else:
                    st.error(f"Report Error: {err}")

    # Build active file list
    active_files = []
    if sop_path:
        active_files.append(sop_path)
    if report_path:
        active_files.append(report_path)
    st.session_state.active_files = active_files

    # 4. Security & Privacy Notice
    st.markdown("---")
    st.markdown("#### Security Disclosure")
    st.caption(
        "🔒 **Local-first execution**: All inference runs locally via Ollama on Apple Silicon Metal. "
        "No documents or embeddings are transmitted to cloud APIs. "
        "Note: Local execution ensures document privacy against external inference services, "
        "but system integrity depends on host security controls."
    )


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ Sovereign AI Workbench</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Local-first, agentic workbench for confidential document analysis and general local chat.</div>',
    unsafe_allow_html=True,
)

# Request Composer
st.markdown("### 💬 Request Composer")

# Quick prompt helper buttons
st.caption("Sample prompts (click to populate):")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    if st.button("📄 Compare vs SOP", use_container_width=True):
        st.session_state.selected_request = "Compare this quarterly report against the SOP and generate a gap register."
with col_p2:
    if st.button("🔍 Check Compliance", use_container_width=True):
        st.session_state.selected_request = "Check this report for compliance gaps and provide evidence and recommended actions."
with col_p3:
    if st.button("📝 Summarize Report", use_container_width=True):
        st.session_state.selected_request = "Summarize this report in five bullet points."
with col_p4:
    if st.button("🛡️ Zero-Trust Chat", use_container_width=True):
        st.session_state.selected_request = "Explain zero-trust architecture."

user_request = st.text_area(
    "Enter your request in natural language:",
    value=st.session_state.selected_request or "Compare this quarterly report against the SOP and generate a gap register.",
    height=80,
    help="Type any natural language instruction. In Auto mode, the local model determines the workflow.",
)

col_plan_btn, col_clear_btn = st.columns([1, 5])
with col_plan_btn:
    generate_plan = st.button("🎯 Generate Plan", type="primary", use_container_width=True)
with col_clear_btn:
    if st.button("Reset State", use_container_width=False):
        st.session_state.current_plan = None
        st.session_state.validation_result = None
        st.session_state.execution_result = None
        st.session_state.execution_events = []
        st.session_state.findings_data = None
        st.session_state.chat_answer = None
        st.rerun()


# ---------------------------------------------------------------------------
# Planning & Validation Step
# ---------------------------------------------------------------------------
if generate_plan and user_request:
    with st.spinner("Local model (gemma4:e4b) is analyzing request and drafting execution plan..."):
        # Route depending on selected mode
        if mode == "General Chat":
            plan = {
                "intent": "Answer a general question locally",
                "steps": [
                    {
                        "capability": "answer_locally",
                        "inputs": {"question": user_request},
                    }
                ],
                "output_format": "text",
            }
        elif mode == "Document Analysis":
            if len(st.session_state.active_files) < 2:
                st.error("Document Analysis mode requires both the SOP PDF and Quarterly Report PDF. Please upload both files.")
                st.stop()
            plan = {
                "intent": "Full document compliance analysis and gap register generation",
                "steps": [
                    {"capability": "extract_requirements", "inputs": {"sop_path": st.session_state.active_files[0]}},
                    {"capability": "check_report", "inputs": {"report_path": st.session_state.active_files[1]}},
                    {"capability": "build_report", "inputs": {}},
                ],
                "output_format": "docx",
            }
        else:
            # Auto Mode: let planner.plan inspect request and available files
            plan = planner_mod.plan(user_request, file_paths=st.session_state.active_files)

        # Validate using existing validator
        validation = validator_mod.validate(
            plan,
            workspace=PROJECT_DIR,
            allowed_files=st.session_state.active_files,
        )

        st.session_state.current_plan = plan
        st.session_state.validation_result = validation
        st.session_state.execution_result = None
        st.session_state.execution_events = []
        st.session_state.findings_data = None
        st.session_state.chat_answer = None


# ---------------------------------------------------------------------------
# Plan Preview & Approval Section
# ---------------------------------------------------------------------------
if st.session_state.current_plan:
    st.markdown("---")
    st.markdown("### 📋 Proposed Execution Plan")

    plan = st.session_state.current_plan
    validation = st.session_state.validation_result

    col_intent, col_val = st.columns([3, 2])
    with col_intent:
        st.markdown(f"**Detected Intent:** `{plan.get('intent', 'unknown')}`")
        st.markdown(f"**Expected Output:** `{plan.get('output_format', 'text')}`")
    with col_val:
        if validation and validation.ok:
            st.markdown('<span class="badge-status badge-green">✓ Validated by Safety Engine</span>', unsafe_allow_html=True)
            if validation.warnings:
                for w in validation.warnings:
                    st.caption(f"⚠ Warning: {w}")
        else:
            reason = validation.reason if validation else "Unvalidated"
            st.markdown(f'<span class="badge-status badge-red">✗ Rejected: {reason}</span>', unsafe_allow_html=True)

    # Show Steps Table
    steps = plan.get("steps", [])
    step_rows = []
    for i, s in enumerate(steps, 1):
        cap = s.get("capability", "")
        inputs_str = ", ".join(f"{k}='{os.path.basename(str(v))}'" if "path" in k else f"{k}={v!r}" for k, v in s.get("inputs", {}).items())
        step_rows.append({"Step": i, "Capability": cap, "Inputs": inputs_str})

    st.table(step_rows)

    # Approval Actions
    if validation and validation.ok:
        col_exec, col_cancel = st.columns([2, 5])
        with col_exec:
            execute_button = st.button("🚀 Approve & Execute Plan", type="primary", use_container_width=True)
        with col_cancel:
            if st.button("❌ Reject / Cancel", use_container_width=False):
                st.session_state.current_plan = None
                st.session_state.validation_result = None
                st.rerun()

        # -------------------------------------------------------------------
        # Plan Execution & Real-Time Event Streaming
        # -------------------------------------------------------------------
        if execute_button:
            st.session_state.execution_events = []
            st.session_state.execution_result = None

            # Streamlit live progress container
            trace_container = st.empty()
            event_log = []

            def _ui_trace_callback(tag: str, message: str, meta: dict = None):
                timestamp = datetime.now().strftime("%H:%M:%S")
                evt = {
                    "time": timestamp,
                    "tag": tag,
                    "message": message,
                    "meta": meta or {},
                }
                event_log.append(evt)
                st.session_state.execution_events.append(evt)

                # Format color-coded live console
                color_class = {
                    "MODEL": "trace-model",
                    "TOOL": "trace-tool",
                    "STEP": "trace-step",
                    "ERROR": "trace-error",
                    "STATUS": "trace-status",
                }.get(tag, "")

                # Render recent events in real time
                lines_html = []
                for e in event_log[-12:]:
                    c = {
                        "MODEL": "trace-model",
                        "TOOL": "trace-tool",
                        "STEP": "trace-step",
                        "ERROR": "trace-error",
                        "STATUS": "trace-status",
                    }.get(e["tag"], "")
                    lines_html.append(f"<div><span style='color:#64748B;'>[{e['time']}]</span> <span class='{c}'>[{e['tag']}]</span> {e['message']}</div>")

                trace_container.markdown(
                    f'<div class="log-box">{"".join(lines_html)}</div>',
                    unsafe_allow_html=True,
                )

            with st.status("Executing workflow with local AI model...", expanded=True) as status_box:
                audit_log.log_plan(plan)
                result = executor_mod.execute(plan, trace_callback=_ui_trace_callback)
                st.session_state.execution_result = result

                if result.success:
                    status_box.update(label="Workflow completed successfully!", state="complete", expanded=False)
                else:
                    status_box.update(label=f"Workflow failed: {result.error}", state="error", expanded=True)

            # Check if findings.json was generated or modified
            findings_file = "findings.json"
            for step_res in result.step_results:
                if "findings_path" in step_res.outputs and os.path.isfile(step_res.outputs["findings_path"]):
                    findings_file = step_res.outputs["findings_path"]
                if "answer" in step_res.outputs:
                    st.session_state.chat_answer = step_res.outputs["answer"]

            if os.path.isfile(findings_file):
                try:
                    with open(findings_file) as f:
                        st.session_state.findings_data = json.load(f)
                except Exception:
                    st.session_state.findings_data = None

            st.rerun()


# ---------------------------------------------------------------------------
# Results & Downloads Section
# ---------------------------------------------------------------------------
if st.session_state.execution_result:
    res = st.session_state.execution_result
    st.markdown("---")

    if res.success:
        st.markdown("### ✅ Execution Results")

        # 1. General Chat Output
        if st.session_state.chat_answer:
            st.markdown("#### Model Response")
            st.markdown(f"> {st.session_state.chat_answer}")

        # 2. Finance Analysis Output
        if st.session_state.findings_data:
            findings = st.session_state.findings_data
            summary = ui_utils.parse_findings_summary(findings)

            # Metrics row
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Total Clauses", summary["total"])
            col_m2.metric("Gaps Found (Not met)", summary["not_met"], delta=f"-{summary['not_met']}" if summary["not_met"] else "0", delta_color="inverse")
            col_m3.metric("Unsure (Needs review)", summary["not_enough_evidence"])
            col_m4.metric("Compliant (Met)", summary["met"])
            col_m5.metric("Calculations Run", len(summary["calculations"]))

            # Draft Warning
            st.warning("⚠ **DRAFT FOR HUMAN REVIEW** — This compliance report was prepared by local AI inference. Human audit is required before reliance.")

            # Artifact Downloads
            st.markdown("#### Generated Deliverables")
            docx_path = os.path.join(PROJECT_DIR, "Gap_Register.docx")
            for p in res.artifact_paths:
                if p.endswith(".docx") and os.path.isfile(p):
                    docx_path = p

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                if os.path.isfile(docx_path):
                    with open(docx_path, "rb") as f:
                        st.download_button(
                            label="⬇ Download Gap_Register.docx",
                            data=f.read(),
                            file_name="Gap_Register.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary",
                            use_container_width=True,
                        )
            with col_d2:
                findings_json_path = os.path.join(PROJECT_DIR, "findings.json")
                if os.path.isfile(findings_json_path):
                    with open(findings_json_path, "rb") as f:
                        st.download_button(
                            label="⬇ Download findings.json",
                            data=f.read(),
                            file_name="findings.json",
                            mime="application/json",
                            use_container_width=True,
                        )
            with col_d3:
                reqs_json_path = os.path.join(PROJECT_DIR, "requirements.json")
                if os.path.isfile(reqs_json_path):
                    with open(reqs_json_path, "rb") as f:
                        st.download_button(
                            label="⬇ Download requirements.json",
                            data=f.read(),
                            file_name="requirements.json",
                            mime="application/json",
                            use_container_width=True,
                        )

            # Interactive Findings Explorer
            st.markdown("#### Findings Explorer")
            filter_choice = st.radio(
                "Filter by status:",
                ["All", "Not met", "Not enough evidence", "Met", "Not triggered"],
                horizontal=True,
            )

            filtered = findings if filter_choice == "All" else [f for f in findings if f.get("verdict") == filter_choice]

            for item in filtered:
                v = item.get("verdict", "")
                badge_class = {
                    "Not met": "badge-red",
                    "Not enough evidence": "badge-yellow",
                    "Met": "badge-green",
                    "Not triggered": "badge-gray",
                }.get(v, "badge-gray")

                with st.expander(f"Clause {item.get('clause')}: {item.get('requirement')[:70]}..."):
                    st.markdown(f'<span class="badge-status {badge_class}">{v}</span>', unsafe_allow_html=True)
                    st.markdown(f"**Full Rule:** {item.get('requirement')}")
                    st.markdown(f"**Reasoning:** {item.get('reasoning')}")
                    st.markdown(f"**Evidence Quote (Page {item.get('page')}):** *\"{item.get('evidence_quote')}\"*")
                    qc = item.get("quote_check", "")
                    st.caption(f"Quote Verification: `{qc}`")

                    # Highlight tool calculation if used on this finding
                    if item.get("tool_calls"):
                        st.markdown("##### 🧮 Tool Calculation Used:")
                        for tc in item["tool_calls"]:
                            st.code(f"Expression: {tc.get('expression')}\nResult:     {tc.get('result')}\nStatus:     {'Success' if tc.get('success') else 'Failed'}")

    else:
        st.error(f"❌ Execution Failed: {res.error}")

    # Tabs for Inspection: Trace, Tool Calls, Audit Log
    st.markdown("---")
    st.markdown("### 🔍 Technical Inspection & Diagnostics")
    tab_trace, tab_tools, tab_audit = st.tabs(["📜 Live Trace Events", "🧮 Tool Calls", "🔒 Audit Log"])

    with tab_trace:
        if st.session_state.execution_events:
            lines = [f"[{e['time']}] [{e['tag']}] {e['message']}" for e in st.session_state.execution_events]
            st.code("\n".join(lines), language="text")
        else:
            st.caption("No trace events recorded for this run.")

    with tab_tools:
        # Filter tool events
        tool_events = [e for e in st.session_state.execution_events if e["tag"] in ("TOOL", "MODEL") and "tool" in e["message"].lower() or "calc" in e["message"].lower()]
        if tool_events:
            for te in tool_events:
                st.markdown(f"**[{te['time']}] `{te['tag']}`:** {te['message']}")
        else:
            st.caption("No tool calls were requested during this workflow.")

    with tab_audit:
        recent_audit = ui_utils.read_audit_log_entries(limit=15)
        if recent_audit:
            st.json(recent_audit)
        else:
            st.caption("No audit records found in audit.jsonl.")
