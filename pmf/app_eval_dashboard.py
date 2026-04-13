"""
Extended Evaluation Dashboard — Streamlit Application
======================================================

Production-grade 5-tab dashboard for the Healthark GenAI Evaluation Framework.

Tabs:
  1. Run Overview     — metric cards, grade badge, run selector
  2. Section Heatmap  — multi-metric heatmap, section detail drill-down
  3. Trend Analysis   — time-series charts, regression alerts
  4. Model Comparison — file upload, radar chart, cost analysis
  5. Benchmark Mgmt   — statistics, case table, add/export/run

Run standalone:
    streamlit run app_eval_dashboard.py

Or import into app.py:
    from app_eval_dashboard import render_eval_dashboard
    render_eval_dashboard()
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eval.eval_store import list_runs, load_run_by_file
from src.eval.eval_config import get_eval_rules
from src.eval.eval_utils import score_section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe imports
# ---------------------------------------------------------------------------

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from src.eval.eval_metrics import LexicalMetrics, SemanticMetrics, compute_all_metrics
    HAS_METRICS = True
except (ImportError, OSError):
    HAS_METRICS = False

try:
    from src.eval.benchmark_loader import BenchmarkLoader
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark")
EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval_results")

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════

CLR_PRIMARY = "#5340C0"   # purple
CLR_SUCCESS = "#1D9E75"   # teal
CLR_WARNING = "#BA7517"   # amber
CLR_DANGER  = "#D85A30"   # coral

GRADE_COLORS: Dict[str, str] = {
    "A": CLR_SUCCESS,        # teal
    "B": "#3B82F6",          # blue
    "C": CLR_WARNING,        # amber
    "D": CLR_DANGER,         # coral
    "F": "#DC2626",          # red
}

CUSTOM_CSS = f"""
<style>
  .dashboard-header {{
    color: {CLR_PRIMARY};
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }}
  .grade-badge {{
    display: inline-block;
    padding: 4px 16px;
    border-radius: 6px;
    color: white;
    font-weight: 700;
    font-size: 1.4rem;
  }}
  .metric-label {{
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  div[data-testid="stMetric"] label {{
    color: {CLR_PRIMARY} !important;
    font-weight: 600 !important;
  }}
  .stTabs [data-baseweb="tab-list"] button {{
    color: {CLR_PRIMARY};
  }}
  .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    border-bottom-color: {CLR_PRIMARY};
    color: {CLR_PRIMARY};
    font-weight: 700;
  }}
</style>
"""


def _grade_badge_html(grade: str) -> str:
    """Return an HTML span for a letter grade badge."""
    color = GRADE_COLORS.get(grade, "#6b7280")
    return (
        f'<span class="grade-badge" style="background:{color};">'
        f'{grade}</span>'
    )


def _letter_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def _fmt(val: Any, decimals: int = 2) -> str:
    """Format a numeric value for display, or '—' if None."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def _load_runs() -> List[Dict[str, Any]]:
    return list_runs()

@st.cache_data(ttl=30)
def _load_run_payload(run_file: str) -> Dict[str, Any]:
    return load_run_by_file(run_file)

@st.cache_data(ttl=30)
def _load_evalsuite_results() -> List[Dict[str, Any]]:
    """Load all JSON result files saved by EvalSuite.save_results()."""
    results: List[Dict[str, Any]] = []
    if not os.path.isdir(EVAL_RESULTS_DIR):
        return results
    for fname in sorted(os.listdir(EVAL_RESULTS_DIR)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(EVAL_RESULTS_DIR, fname), "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return results


def _extract_extended(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pull extended_eval_summary from a run payload if present."""
    return payload.get("run_artifacts", payload).get("extended_eval_summary", {})


def _section_extended(section: Dict[str, Any]) -> Dict[str, Any]:
    """Pull extended_eval from a section dict."""
    return section.get("extended_eval", {})


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    """Global sidebar: about, API keys, model selector, metric toggles, export."""
    with st.sidebar:
        st.markdown(
            f"<h3 style='color:{CLR_PRIMARY};'>About</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "- **Multi-layer evaluation** — Combines rule-based checks, "
            "lexical metrics (BLEU/ROUGE), semantic similarity (BERTScore), "
            "and LLM-as-judge scoring into one composite quality grade.\n"
            "- **RAG pipeline monitoring** — Measures faithfulness, context "
            "precision, and answer relevancy to detect hallucination and "
            "retrieval failures before they reach production.\n"
            "- **Regression tracking** — Automatically compares every run "
            "against a stored baseline and alerts when quality drops, "
            "enabling continuous improvement across model and template updates."
        )
        st.divider()

        st.markdown(f"<h3 style='color:{CLR_PRIMARY};'>Settings</h3>",
                    unsafe_allow_html=True)

        st.text_input("Anthropic API Key", type="password",
                       key="sidebar_anthropic_key",
                       help="Used for LLM-as-Judge and RAG evaluation")
        st.text_input("Azure OpenAI API Key", type="password",
                       key="sidebar_azure_key",
                       help="Used for Azure OpenAI-based evaluation")

        st.selectbox("Judge Model", [
            "claude-sonnet-4-6", "gpt-4o",
        ], key="sidebar_judge_model")

        st.markdown("**Metric Toggles**")
        st.checkbox("Run Lexical (BLEU/ROUGE)", value=True, key="toggle_lexical")
        st.checkbox("Run Semantic (BERTScore)", value=False, key="toggle_semantic")
        st.checkbox("Run LLM Judge", value=False, key="toggle_judge")
        st.checkbox("Run RAG Metrics", value=False, key="toggle_rag")

        st.divider()

        # Export current run as JSON download
        if st.session_state.get("_current_payload"):
            data_str = json.dumps(st.session_state["_current_payload"], indent=2, default=str)
            st.download_button(
                "Export Current Run (JSON)",
                data_str,
                file_name="eval_run_export.json",
                mime="application/json",
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_overview(runs: List[Dict[str, Any]]) -> None:
    """Metric cards, grade badge, run selector."""

    # ── Run selector ─────────────────────────────────────────────────────
    if not runs:
        st.info("No evaluation runs found. Generate a PMF document first.")
        return

    labels = [
        f"{r.get('timestamp', '')}  |  {r.get('site_name', '')}  |  "
        f"score={r.get('overall_score', '?')}"
        for r in runs
    ]
    idx = st.selectbox("Select evaluation run", range(len(runs)),
                       format_func=lambda i: labels[i], key="tab1_run_sel")
    run_meta = runs[idx]
    run_file = run_meta.get("run_file", "")
    if not run_file or not os.path.exists(run_file):
        st.warning("Run file not found.")
        return

    payload = _load_run_payload(run_file)
    st.session_state["_current_payload"] = payload  # for sidebar export

    eval_data = payload.get("evaluation", {}).get("document_scores", {})
    ext = _extract_extended(payload)

    # Delta computation (vs previous run)
    prev_eval: Dict[str, Any] = {}
    if idx + 1 < len(runs):
        prev_file = runs[idx + 1].get("run_file", "")
        if prev_file and os.path.exists(prev_file):
            prev_payload = _load_run_payload(prev_file)
            prev_eval = prev_payload.get("evaluation", {}).get("document_scores", {})

    def _delta(cur: Any, prev: Any) -> Optional[float]:
        try:
            c, p = float(cur), float(prev)
            return round(c - p, 2)
        except (ValueError, TypeError):
            return None

    rule_score = eval_data.get("overall_score", 0)
    prev_rule = prev_eval.get("overall_score")

    bert_f1 = ext.get("mean_bertscore_f1")
    judge_score = ext.get("mean_judge_normalized")
    ragas_score = ext.get("mean_ragas")
    composite = ext.get("mean_composite", rule_score)

    # ── Metric cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rule Score", _fmt(rule_score),
              delta=_fmt(_delta(rule_score, prev_rule)))
    c2.metric("BERTScore F1", _fmt(bert_f1, 4))
    c3.metric("Judge Score", _fmt(judge_score))
    c4.metric("RAGAS Score", _fmt(ragas_score, 4))
    c5.metric("Composite", _fmt(composite))

    # ── Metric tooltips ─────────────────────────────────────────────────
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **BERTScore** — Measures how semantically similar the "
            "generated text is to a reference document, using AI-based "
            "language understanding.\n"
            "- **Judge Score** — Score assigned by a second AI acting as "
            "a domain expert evaluating the regulatory quality of the "
            "output.\n"
            "- **Faithfulness** (in RAGAS) — Measures whether the AI's "
            "answer is fully supported by the source documents it "
            "retrieved — a low score indicates hallucination.\n"
            "- **RAGAS Score** — Combined quality score of the retrieval "
            "and generation pipeline."
        )

    # ── Grade badge ──────────────────────────────────────────────────────
    grade = ext.get("overall_grade") or _letter_grade(float(composite or rule_score or 0))
    passed = float(composite or rule_score or 0) >= 65.0
    badge_html = _grade_badge_html(grade)
    status = "PASS" if passed else "FAIL"
    status_clr = CLR_SUCCESS if passed else CLR_DANGER
    st.markdown(
        f"{badge_html} &nbsp; "
        f'<span style="color:{status_clr};font-weight:700;">{status}</span>',
        unsafe_allow_html=True,
    )

    # ── Section count, retrieval, missing sections ───────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Section Count", eval_data.get("section_count", 0))
    mc2.metric("Retrieval Coverage",
               f"{eval_data.get('retrieval_coverage', 0)}%")
    missing = eval_data.get("missing_required_sections", [])
    if missing:
        mc3.error(f"Missing: {', '.join(missing)}")
    else:
        mc3.success("All required sections present")

    # ── Download Report (DOCX) ───────────────────────────────────────────
    _render_download_report(payload, eval_data, ext, grade, rule_score)

    # ── Live Evaluation mode ─────────────────────────────────────────────
    st.divider()
    _render_live_evaluation()


# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOAD REPORT (DOCX)
# ═══════════════════════════════════════════════════════════════════════════


def _render_download_report(
    payload: Dict[str, Any],
    eval_data: Dict[str, Any],
    ext: Dict[str, Any],
    grade: str,
    rule_score: Any,
) -> None:
    """Generate a formatted DOCX summary report and offer it for download."""
    try:
        from docx import Document as DocxDocument
        from docx.shared import Pt, Inches, RGBColor
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
    except ImportError:
        st.caption("Install python-docx to enable DOCX report download.")
        return

    import io

    if not st.button("Download Report (DOCX)", key="dl_report_btn"):
        return

    doc = DocxDocument()

    # ── Title ────────────────────────────────────────────────────────
    title = doc.add_heading("PMF Evaluation Report", level=0)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x53, 0x40, 0xC0)

    # ── Run metadata table ───────────────────────────────────────────
    doc.add_heading("Run Metadata", level=1)
    run_arts = payload.get("run_artifacts", payload)
    meta_items = [
        ("Timestamp", run_arts.get("timestamp", "—")),
        ("Site Name", run_arts.get("site_name", "—")),
        ("Model", run_arts.get("model_name", "—")),
        ("Template", run_arts.get("template_file", "—")),
        ("Grade", grade),
    ]
    tbl = doc.add_table(rows=len(meta_items), cols=2, style="Light Grid Accent 1")
    for i, (label, val) in enumerate(meta_items):
        tbl.rows[i].cells[0].text = label
        tbl.rows[i].cells[1].text = str(val)

    # ── Metric scores table ──────────────────────────────────────────
    doc.add_heading("Metric Scores", level=1)
    metrics = [
        ("Rule Score", _fmt(rule_score)),
        ("BERTScore F1", _fmt(ext.get("mean_bertscore_f1"), 4)),
        ("Judge Normalized", _fmt(ext.get("mean_judge_normalized"))),
        ("RAGAS Score", _fmt(ext.get("mean_ragas"), 4)),
        ("Composite", _fmt(ext.get("mean_composite", rule_score))),
        ("Retrieval Coverage", f"{eval_data.get('retrieval_coverage', 0)}%"),
    ]
    tbl2 = doc.add_table(rows=len(metrics), cols=2, style="Light Grid Accent 1")
    for i, (label, val) in enumerate(metrics):
        tbl2.rows[i].cells[0].text = label
        tbl2.rows[i].cells[1].text = str(val)

    # ── Top 3 best and worst sections ────────────────────────────────
    sections = eval_data.get("sections", [])
    sorted_secs = sorted(sections, key=lambda s: s.get("score", 0), reverse=True)

    doc.add_heading("Top 3 Best Sections", level=1)
    for s in sorted_secs[:3]:
        doc.add_paragraph(
            f"{s.get('section_key', '?')} — Score: {s.get('score', 0)}",
            style="List Bullet",
        )

    doc.add_heading("Top 3 Worst Sections", level=1)
    for s in sorted_secs[-3:]:
        doc.add_paragraph(
            f"{s.get('section_key', '?')} — Score: {s.get('score', 0)}",
            style="List Bullet",
        )

    # ── Judge findings ───────────────────────────────────────────────
    doc.add_heading("Key Findings from LLM Judge", level=1)
    run_secs = run_arts.get("sections", [])
    all_critical: List[str] = []
    all_strengths: List[str] = []
    all_suggestions: List[str] = []
    for rs in run_secs:
        ext_sec = _section_extended(rs)
        judge = ext_sec.get("judge_scores") or {}
        if judge.get("judge_error"):
            continue
        key = rs.get("section_key", "?")
        for iss in judge.get("critical_issues", []):
            all_critical.append(f"[{key}] {iss}")
        for s in judge.get("strengths", [])[:1]:
            all_strengths.append(f"[{key}] {s}")
        for sg in judge.get("improvement_suggestions", [])[:1]:
            all_suggestions.append(f"[{key}] {sg}")

    if all_critical:
        doc.add_heading("Critical Issues", level=2)
        for item in all_critical[:10]:
            doc.add_paragraph(item, style="List Bullet")
    else:
        doc.add_paragraph("No critical issues identified.", style="List Bullet")

    if all_strengths:
        doc.add_heading("Top Strengths", level=2)
        for item in all_strengths[:5]:
            doc.add_paragraph(item, style="List Bullet")

    # ── Recommendations ──────────────────────────────────────────────
    doc.add_heading("Recommendations", level=1)
    if all_suggestions:
        for item in all_suggestions[:5]:
            doc.add_paragraph(item, style="List Bullet")
    else:
        doc.add_paragraph(
            "Run the extended evaluation with the LLM judge enabled to "
            "generate specific improvement recommendations.",
        )

    # ── Footer ───────────────────────────────────────────────────────
    doc.add_paragraph("")
    footer = doc.add_paragraph(
        f"Generated by Healthark GenAI Evaluation Framework v1.0 "
        f"on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    footer.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x9C, 0xA3, 0xAF)

    # ── Serve as download ────────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    st.download_button(
        "Save Report",
        buf.getvalue(),
        file_name=f"eval_report_{run_arts.get('timestamp', 'unknown')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key="dl_report_file",
    )


# ═══════════════════════════════════════════════════════════════════════════
# LIVE EVALUATION (CEO demo feature)
# ═══════════════════════════════════════════════════════════════════════════


def _render_live_evaluation() -> None:
    """Paste-and-evaluate: runs metrics + judge on arbitrary text."""
    st.subheader("Live Evaluation")
    st.caption(
        "Paste any generated and reference text below to evaluate quality "
        "in real time — the primary demo feature for stakeholder presentations."
    )

    lc1, lc2 = st.columns(2)
    with lc1:
        gen_text = st.text_area(
            "Paste generated text here", height=180, key="live_gen_txt",
        )
    with lc2:
        ref_text = st.text_area(
            "Paste reference text here", height=180, key="live_ref_txt",
        )

    live_section = st.text_input(
        "Section key (optional)", value="LIVE EVAL", key="live_sec_key",
    )

    if not st.button("Evaluate Now", key="live_eval_now_btn", type="primary"):
        return

    if not gen_text.strip():
        st.warning("Please paste generated text.")
        return

    # ── Phase 1: Lexical metrics (fast, <2s) ─────────────────────────
    with st.spinner("Computing lexical metrics..."):
        lex_result: Dict[str, Any] = {}
        sim_result: float = 0.0
        if HAS_METRICS and ref_text.strip():
            lex_result = LexicalMetrics.compute_all_lexical(gen_text, ref_text)

    # Show lexical results immediately
    st.subheader("Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("BLEU", _fmt(lex_result.get("bleu"), 2))
    r2.metric("ROUGE-1 F1", _fmt(lex_result.get("rouge1_fmeasure"), 4))
    r3.metric("ROUGE-L F1", _fmt(lex_result.get("rougeL_fmeasure"), 4))

    # ── Phase 2: BERTScore (if enabled, ~3-5s) ──────────────────────
    bert_f1: Any = None
    if st.session_state.get("toggle_semantic", False) and HAS_METRICS and ref_text.strip():
        with st.spinner("Computing BERTScore..."):
            try:
                sm = SemanticMetrics(model_type="distilbert-base-uncased")
                bs = sm.compute_bertscore([gen_text], [ref_text])
                bert_f1 = bs.get("bertscore_f1_mean")
                sim_result = sm.compute_semantic_similarity(gen_text, ref_text)
            except Exception as exc:
                logger.warning("BERTScore failed in live eval: %s", exc)
    r4.metric("BERTScore F1", _fmt(bert_f1, 4))

    # ── Phase 3: LLM Judge (if enabled, ~5-15s) ─────────────────────
    judge_result: Dict[str, Any] = {}
    if st.session_state.get("toggle_judge", False):
        with st.spinner("Running LLM Judge evaluation..."):
            try:
                from src.eval.eval_judge import PMFJudge
                api_key = st.session_state.get("sidebar_anthropic_key", "")
                model = st.session_state.get("sidebar_judge_model", "claude-sonnet-4-6")
                judge = PMFJudge(
                    provider="anthropic",
                    model=model,
                    api_key=api_key,
                    cache_enabled=False,
                )
                judge_result = judge.score_section(
                    section_key=live_section,
                    section_instruction="Evaluate this section.",
                    retrieved_context="",
                    generated_output=gen_text,
                    site_name="",
                    reference_output=ref_text,
                )
            except Exception as exc:
                logger.warning("Judge failed in live eval: %s", exc)
                judge_result = {"judge_error": True, "error": str(exc)}

    # ── Display judge results ────────────────────────────────────────
    if judge_result and not judge_result.get("judge_error"):
        st.subheader("LLM Judge Assessment")
        jc1, jc2, jc3, jc4, jc5 = st.columns(5)
        scores = judge_result.get("scores", {})
        jc1.metric("Factual Accuracy", scores.get("factual_accuracy", "—"))
        jc2.metric("Regulatory Lang.", scores.get("regulatory_language", "—"))
        jc3.metric("Site Specificity", scores.get("site_specificity", "—"))
        jc4.metric("Completeness", scores.get("completeness", "—"))
        jc5.metric("Coherence", scores.get("structural_coherence", "—"))

        norm = judge_result.get("normalized_score", 0)
        j_grade = _letter_grade(float(norm))
        st.markdown(
            f"**Normalized Score:** {_fmt(norm)} / 100 &nbsp; "
            f"{_grade_badge_html(j_grade)}",
            unsafe_allow_html=True,
        )

        with st.expander("Judge Details"):
            st.markdown(f"**Strengths:** {', '.join(judge_result.get('strengths', [])) or '—'}")
            st.markdown(f"**Weaknesses:** {', '.join(judge_result.get('weaknesses', [])) or '—'}")
            critical = judge_result.get("critical_issues", [])
            if critical:
                st.error(f"Critical: {', '.join(critical)}")
            st.markdown(f"**Notes:** {judge_result.get('evaluation_notes', '—')}")
    elif judge_result.get("judge_error"):
        st.warning(
            f"Judge evaluation failed: {judge_result.get('error', 'unknown')}. "
            "Check your API key in the sidebar."
        )

    # ── Full JSON result (expandable) ────────────────────────────────
    with st.expander("Full Results (JSON)"):
        full = {
            "lexical": lex_result,
            "bertscore_f1": bert_f1,
            "semantic_similarity": sim_result,
            "judge": judge_result if judge_result else None,
        }
        st.json(full)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — SECTION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_heatmap(runs: List[Dict[str, Any]]) -> None:
    """Multi-metric heatmap (rows=sections, cols=metrics) with detail view."""

    if not runs:
        st.info("No runs available.")
        return

    # Reuse the same run selector
    labels = [
        f"{r.get('timestamp', '')}  |  {r.get('site_name', '')}"
        for r in runs
    ]
    idx = st.selectbox("Select run for heatmap", range(len(runs)),
                       format_func=lambda i: labels[i], key="tab2_run_sel")
    run_file = runs[idx].get("run_file", "")
    if not run_file or not os.path.exists(run_file):
        return

    payload = _load_run_payload(run_file)
    eval_data = payload.get("evaluation", {}).get("document_scores", {})
    sections = eval_data.get("sections", [])
    run_sections = payload.get("run_artifacts", payload).get("sections", [])

    if not sections:
        st.info("No section data available.")
        return

    # Build a lookup from section_key to the run_artifacts section
    run_sec_map: Dict[str, Dict[str, Any]] = {}
    for rs in run_sections:
        run_sec_map[rs.get("section_key", "")] = rs

    # ── Build heatmap DataFrame ──────────────────────────────────────────
    heatmap_rows: List[Dict[str, Any]] = []
    for s in sections:
        key = s.get("section_key", "?")
        ext = _section_extended(run_sec_map.get(key, {}))
        lex = ext.get("lexical_scores") or {}
        sem = ext.get("semantic_scores") or {}
        judge = ext.get("judge_scores") or {}
        rag = ext.get("rag_scores") or {}

        heatmap_rows.append({
            "Section": key[:35],
            "Rule Score": s.get("score", 0),
            "BERTScore F1": _safe_pct(sem.get("bertscore_f1_mean")),
            "Judge Score": judge.get("normalized_score"),
            "Faithfulness": _safe_pct(rag.get("faithfulness")),
            "Ctx Precision": _safe_pct(rag.get("context_precision")),
        })

    df = pd.DataFrame(heatmap_rows).set_index("Section")

    # Replace None with NaN for Styler
    df = df.where(df.notna(), other=float("nan"))

    metric_cols = ["Rule Score", "BERTScore F1", "Judge Score",
                   "Faithfulness", "Ctx Precision"]

    st.subheader("Section Heatmap")
    st.caption("Green (>80) | Yellow (50-80) | Red (<50). Blank = metric not computed.")

    styled = df[metric_cols].style.background_gradient(
        cmap="RdYlGn", vmin=0, vmax=100, axis=None,
    ).format("{:.1f}", na_rep="—")

    st.dataframe(styled, use_container_width=True, height=min(len(df) * 40 + 60, 600))

    # ── Section detail drill-down ────────────────────────────────────────
    st.subheader("Section Detail")
    sec_names = [s.get("section_key", "?") for s in sections]
    sel_sec = st.selectbox("Select section", sec_names, key="tab2_sec_detail")

    rule_sec = next((s for s in sections if s.get("section_key") == sel_sec), {})
    run_sec = run_sec_map.get(sel_sec, {})
    ext_sec = _section_extended(run_sec)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Section Instruction**")
        st.text(run_sec.get("prompt_text", "—")[:500])

        with st.expander("Retrieved Context"):
            paths = run_sec.get("retrieved_paths", [])
            if paths:
                st.write(f"{len(paths)} documents retrieved: {paths}")
            else:
                st.write("No retrieval data available.")

    with col_b:
        st.markdown("**Generated Text**")
        gen_text = run_sec.get("generated_text", "")
        st.text(gen_text[:800] if gen_text else "—")

    # Metric scores for this section
    st.markdown("**Metric Scores**")
    lex = ext_sec.get("lexical_scores") or {}
    sem = ext_sec.get("semantic_scores") or {}
    judge = ext_sec.get("judge_scores") or {}
    rag = ext_sec.get("rag_scores") or {}

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Rule", _fmt(rule_sec.get("score")))
    sc2.metric("BLEU", _fmt(lex.get("bleu")))
    sc3.metric("BERTScore F1", _fmt(sem.get("bertscore_f1_mean"), 4))
    sc4.metric("Judge Norm.", _fmt(judge.get("normalized_score")))
    sc5.metric("Faithfulness", _fmt(rag.get("faithfulness"), 4))

    # Judge qualitative output
    if judge and not judge.get("judge_error"):
        with st.expander("Judge Output"):
            st.markdown(f"**Strengths:** {', '.join(judge.get('strengths', [])) or '—'}")
            st.markdown(f"**Weaknesses:** {', '.join(judge.get('weaknesses', [])) or '—'}")
            critical = judge.get("critical_issues", [])
            if critical:
                st.error(f"Critical Issues: {', '.join(critical)}")
            st.markdown(f"**Notes:** {judge.get('evaluation_notes', '—')}")


def _safe_pct(val: Any) -> Optional[float]:
    """Convert a 0-1 metric to 0-100, or return None."""
    if val is None:
        return None
    try:
        v = float(val)
        return round(v * 100, 2) if v <= 1.0 else round(v, 2)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_trends(runs: List[Dict[str, Any]]) -> None:
    """Time-series charts and regression alerts."""

    if len(runs) < 2:
        st.info("Need at least 2 runs for trend analysis.")
        return

    df = pd.DataFrame(runs)
    df["overall_score"] = pd.to_numeric(df.get("overall_score"), errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Regression alert ─────────────────────────────────────────────────
    latest = df.iloc[-1]["overall_score"]
    prev = df.iloc[-2]["overall_score"]
    if pd.notna(latest) and pd.notna(prev):
        drop = prev - latest
        if drop > 5:
            latest_ts = df.iloc[-1].get("timestamp", "?")
            st.markdown(
                f'<div style="background:{CLR_DANGER};color:white;padding:12px;'
                f'border-radius:8px;margin-bottom:16px;">'
                f'<strong>Regression detected:</strong> Rule score dropped from '
                f'{prev:.1f} to {latest:.1f} (delta {-drop:+.1f}) on {latest_ts}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Composite score over time ────────────────────────────────────────
    st.subheader("Score Trend Over Time")

    if HAS_PLOTLY:
        fig = px.line(df, x="timestamp", y="overall_score", markers=True,
                      labels={"overall_score": "Rule Score (%)", "timestamp": "Run"})
        fig.update_traces(line_color=CLR_PRIMARY)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index("timestamp")["overall_score"])

    # ── Multi-metric trend ───────────────────────────────────────────────
    # Load extended data from each run payload to build multi-metric trend
    multi_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rf = row.get("run_file", "")
        if rf and os.path.exists(rf):
            p = _load_run_payload(rf)
            ext = _extract_extended(p)
            ed = p.get("evaluation", {}).get("document_scores", {})
            multi_rows.append({
                "timestamp": row["timestamp"],
                "Rule Score": ed.get("overall_score"),
                "BERTScore F1": _safe_pct(ext.get("mean_bertscore_f1")),
                "Judge Score": ext.get("mean_judge_normalized"),
                "Faithfulness": _safe_pct(ext.get("mean_faithfulness")),
                "RAGAS": _safe_pct(ext.get("mean_ragas")),
            })

    if multi_rows:
        mdf = pd.DataFrame(multi_rows).set_index("timestamp")
        # Only show columns that have at least one non-null value
        active_cols = [c for c in mdf.columns if mdf[c].notna().any()]
        if active_cols:
            st.subheader("Multi-Metric Trend")
            if HAS_PLOTLY:
                fig = go.Figure()
                colors = [CLR_PRIMARY, CLR_SUCCESS, "#3B82F6", CLR_WARNING, CLR_DANGER]
                for i, col in enumerate(active_cols):
                    fig.add_trace(go.Scatter(
                        x=mdf.index, y=mdf[col], name=col, mode="lines+markers",
                        line=dict(color=colors[i % len(colors)]),
                    ))
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis_title="Score (0-100)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(mdf[active_cols])

    # ── Section score distribution (histogram) ───────────────────────────
    st.subheader("Section Score Distribution (Latest Run)")
    latest_file = df.iloc[-1].get("run_file", "")
    if latest_file and os.path.exists(latest_file):
        lp = _load_run_payload(latest_file)
        secs = lp.get("evaluation", {}).get("document_scores", {}).get("sections", [])
        sec_scores = [s.get("score", 0) for s in secs]
        if sec_scores:
            if HAS_PLOTLY:
                fig = px.histogram(
                    x=sec_scores, nbins=10,
                    labels={"x": "Section Rule Score", "count": "Count"},
                    color_discrete_sequence=[CLR_PRIMARY],
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                hist_df = pd.DataFrame({"Score": sec_scores})
                st.bar_chart(hist_df["Score"].value_counts().sort_index())


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON (unchanged from previous)
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_model_comparison() -> None:
    """Upload JSON results and compare models."""
    st.subheader("Model Comparison")

    uploaded = st.file_uploader(
        "Upload evaluation result JSON files (one per model run)",
        type=["json"], accept_multiple_files=True, key="mc_upload",
    )
    if not uploaded:
        st.info("Upload two or more JSON result files to compare models.")
        return

    model_data: List[Dict[str, Any]] = []
    for f in uploaded:
        try:
            data = json.loads(f.read().decode("utf-8"))
            data["_source_file"] = f.name
            model_data.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            st.warning(f"Could not parse {f.name}: {exc}")

    if len(model_data) < 2:
        st.warning("Upload at least 2 files to compare.")
        return

    rows: List[Dict[str, Any]] = []
    for d in model_data:
        label = d.get("_source_file", "unknown")
        judge = d.get("judge_scores") or {}
        rag = d.get("rag_scores") or {}
        lex = d.get("lexical_scores") or {}
        sem = d.get("semantic_scores") or {}
        rows.append({
            "Model / File": label,
            "Section": d.get("section_key", "—"),
            "Composite": d.get("composite_score", 0),
            "Grade": d.get("grade", "?"),
            "Rule Score": d.get("rule_score"),
            "BLEU": lex.get("bleu"),
            "BERTScore F1": sem.get("bertscore_f1_mean"),
            "Judge Norm.": judge.get("normalized_score"),
            "RAGAS": rag.get("ragas_score"),
        })

    cdf = pd.DataFrame(rows)
    st.dataframe(
        cdf.style.background_gradient(subset=["Composite"], cmap="RdYlGn",
                                       vmin=0, vmax=100),
        use_container_width=True, hide_index=True,
    )

    # Radar chart
    st.subheader("Metric Radar")
    radar_metrics = ["Rule Score", "BLEU", "BERTScore F1", "Judge Norm.", "RAGAS"]
    if HAS_PLOTLY:
        fig = go.Figure()
        for _, row in cdf.iterrows():
            vals = []
            for m in radar_metrics:
                v = row.get(m)
                if v is None:
                    vals.append(0)
                elif m in ("BERTScore F1", "RAGAS"):
                    vals.append(float(v) * 100)
                else:
                    vals.append(float(v))
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_metrics + [radar_metrics[0]],
                fill="toself", name=str(row.get("Model / File", "")), opacity=0.6,
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          height=400, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(cdf[["Model / File"] + radar_metrics].set_index("Model / File"))

    # Cost analysis
    st.subheader("Cost Analysis")
    cost_entries: Dict[str, float] = {}
    cost_cols = st.columns(min(len(model_data), 4))
    for i, d in enumerate(model_data):
        label = d.get("_source_file", f"Model {i+1}")
        with cost_cols[i % len(cost_cols)]:
            cost = st.number_input(f"Cost ($) — {label}", min_value=0.0,
                                   value=0.0, step=0.01, key=f"cost_{i}")
            cost_entries[label] = cost

    if any(c > 0 for c in cost_entries.values()):
        st.subheader("Pareto Recommendation")
        pareto_rows = []
        for _, row in cdf.iterrows():
            label = row["Model / File"]
            cost = cost_entries.get(label, 0)
            comp = row.get("Composite", 0) or 0
            eff = round(comp / cost, 2) if cost > 0 else 0
            pareto_rows.append({"Model": label, "Composite": comp,
                                "Cost ($)": cost, "Score / $": eff})
        pdf = pd.DataFrame(pareto_rows).sort_values("Score / $", ascending=False)
        st.dataframe(pdf, use_container_width=True, hide_index=True)
        best = pdf.iloc[0]
        st.success(
            f"Recommended: **{best['Model']}** — "
            f"Composite {best['Composite']:.1f} at ${best['Cost ($)']:.2f} "
            f"({best['Score / $']:.1f} points per dollar)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — BENCHMARK MANAGEMENT (unchanged from previous)
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_benchmark() -> None:
    """Benchmark management interface."""
    if not HAS_BENCHMARK:
        st.info("Benchmark loader not available.")
        return

    st.subheader("Benchmark Management")
    loader = BenchmarkLoader(BENCHMARK_DIR)
    stats = loader.get_statistics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", stats.get("total_cases", 0))
    c2.metric("With Reference", stats.get("cases_with_reference", 0))
    c3.metric("Expert Annotated", stats.get("cases_with_expert_scores", 0))
    c4.metric("Auto Scored", stats.get("cases_with_automated_scores", 0))

    # Filterable table
    st.subheader("Browse Cases")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        ft = st.selectbox("Section Type",
                          ["All"] + list(stats.get("by_section_type", {}).keys()),
                          key="bm_ft")
    with fc2:
        fd = st.selectbox("Difficulty",
                          ["All"] + list(stats.get("by_difficulty", {}).keys()),
                          key="bm_fd")
    with fc3:
        fs = st.text_input("Search section key", "", key="bm_fs")

    filters: Dict[str, Any] = {}
    if ft != "All":
        filters["section_type"] = ft
    if fd != "All":
        filters["difficulty"] = fd
    if fs.strip():
        filters["section_key"] = fs.strip()

    cases = loader.load_cases(filters if filters else None)
    if cases:
        rows = [{
            "Case ID": c.get("case_id", ""),
            "Section": c.get("section_key", ""),
            "Site": c.get("site_name", ""),
            "Type": c.get("section_type", ""),
            "Difficulty": c.get("difficulty", ""),
            "Ref Len": len(c.get("reference_output", "")),
            "Tags": ", ".join(c.get("tags", [])),
        } for c in cases]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No cases match filters.")

    # Add case form
    st.subheader("Add New Case")
    with st.expander("New benchmark case form"):
        nsk = st.text_input("Section Key", key="bn_sk")
        nsite = st.text_input("Site Name", key="bn_site")
        ninstr = st.text_area("Section Instruction", height=80, key="bn_instr")
        nquery = st.text_input("Retrieval Query", key="bn_query")
        nref = st.text_area("Reference Output", height=120, key="bn_ref")
        ntype = st.selectbox("Type", ["text", "table", "image", "static"], key="bn_type")
        ndiff = st.selectbox("Difficulty", ["easy", "medium", "hard"], key="bn_diff")
        ntags = st.text_input("Tags (comma-separated)", key="bn_tags")
        if st.button("Add Case", key="bn_add"):
            if not nsk.strip() or not nref.strip():
                st.warning("Section Key and Reference Output are required.")
            else:
                new_case: Dict[str, Any] = {
                    "created_at": datetime.now().isoformat(),
                    "created_by": "human", "validated_by": None,
                    "site_name": nsite, "section_key": nsk,
                    "section_instruction": ninstr, "retrieval_query": nquery,
                    "source_documents": [], "retrieved_context": "",
                    "generated_output": {}, "reference_output": nref,
                    "expert_scores": {}, "automated_scores": {},
                    "tags": [t.strip() for t in ntags.split(",") if t.strip()],
                    "difficulty": ndiff, "section_type": ntype,
                }
                try:
                    cid = loader.add_case(new_case, validate=True)
                    st.success(f"Case **{cid}** added.")
                except ValueError as exc:
                    st.error(f"Validation failed: {exc}")

    # Export + Run
    st.subheader("Export & Run")
    ec1, ec2 = st.columns(2)
    with ec1:
        if st.button("Export to CSV", key="bn_csv"):
            csv_path = os.path.join(BENCHMARK_DIR, "benchmark_export.csv")
            loader.export_to_csv(csv_path)
            st.success(f"Exported to `{csv_path}`")
            with open(csv_path, "r", encoding="utf-8") as fh:
                st.download_button("Download CSV", fh.read(),
                                   file_name="benchmark_export.csv",
                                   mime="text/csv", key="bn_dl")
    with ec2:
        do_run = st.button("Run Full Benchmark", key="bn_run")
    if do_run:
        all_cases = loader.load_cases()
        if not all_cases:
            st.warning("No cases.")
            return
        try:
            from healthark_eval import EvalSuite
        except ImportError:
            st.error("healthark_eval not available.")
            return
        suite = EvalSuite(task="pmf", run_judge=False, run_rag=False,
                          run_semantic=False)
        prog = st.progress(0, text="Evaluating...")
        res: List[Dict[str, Any]] = []
        for i, case in enumerate(all_cases):
            ref = case.get("reference_output", "")
            r = suite.run(generated=ref, reference=ref,
                          section_key=case.get("section_key", ""),
                          section_instruction=case.get("section_instruction", ""),
                          site_name=case.get("site_name", ""))
            res.append({"Case ID": case.get("case_id"), "Section": case.get("section_key"),
                         "Composite": r.composite_score, "Grade": r.grade,
                         "Rule Score": r.rule_score})
            prog.progress((i + 1) / len(all_cases),
                          text=f"Evaluated {i+1}/{len(all_cases)}...")
        prog.empty()
        st.success(f"Benchmark complete: {len(res)} cases.")
        rdf = pd.DataFrame(res)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        st.metric("Mean Composite", f"{rdf['Composite'].mean():.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════

def render_eval_dashboard() -> None:
    """Render the complete 5-tab evaluation dashboard."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-header">'
        'Healthark GenAI Evaluation Framework v1.0'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("PMF Document Generator — LLM Evaluation & Benchmarking")

    _render_sidebar()

    runs = _load_runs()

    t1, t2, t3, t4, t5 = st.tabs([
        "Run Overview", "Section Heatmap", "Trend Analysis",
        "Model Comparison", "Benchmark Management",
    ])

    with t1:
        _render_tab_overview(runs)
    with t2:
        _render_tab_heatmap(runs)
    with t3:
        _render_tab_trends(runs)
    with t4:
        _render_tab_model_comparison()
    with t5:
        _render_tab_benchmark()


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" or "streamlit" in getattr(
    sys.modules.get("__main__"), "__module__", ""
):
    st.set_page_config(
        page_title="PMF Evaluation Dashboard",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
    render_eval_dashboard()
