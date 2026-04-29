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

# Lexical metrics (BLEU/ROUGE) — sacrebleu + rouge-score, no torch dependency
try:
    from src.eval.eval_metrics import LexicalMetrics
    HAS_LEXICAL = True
except (ImportError, OSError):
    HAS_LEXICAL = False
    logger.warning("LexicalMetrics unavailable — BLEU/ROUGE disabled")

# Semantic metrics (BERTScore) — bert-score + sentence-transformers, needs torch
try:
    from src.eval.eval_metrics import SemanticMetrics, compute_all_metrics
    HAS_SEMANTIC = True
except (ImportError, OSError):
    HAS_SEMANTIC = False
    logger.warning("SemanticMetrics unavailable — BERTScore disabled")

HAS_METRICS = HAS_LEXICAL or HAS_SEMANTIC

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

        # Pre-fill keys from .env so the user doesn't have to retype them
        _default_anthropic = os.getenv("ANTHROPIC_API_KEY", "")
        _default_azure = os.getenv("AZURE_KEY", os.getenv("AZURE_OPENAI_API_KEY", ""))

        st.text_input(
            "Anthropic API Key",
            value=_default_anthropic,
            type="password",
            key="sidebar_anthropic_key",
            help="Used for LLM-as-Judge. Auto-loaded from .env if ANTHROPIC_API_KEY is set.",
        )
        st.text_input(
            "Azure OpenAI API Key",
            value=_default_azure,
            type="password",
            key="sidebar_azure_key",
            help="Auto-loaded from .env (AZURE_KEY).",
        )

        st.selectbox("Judge Model", [
            "claude-sonnet-4-6", "gpt-4o",
        ], key="sidebar_judge_model")

        st.markdown("**Quick Eval Toggles** *(for the text box below)*")
        st.checkbox(
            "Run LLM Judge",
            value=False,
            key="toggle_judge",
            help="Calls an LLM to score the pasted text on 5 PMF criteria (needs API key above).",
        )
        st.checkbox(
            "Run DeepEval RAG Triad",
            value=False,
            key="toggle_rag",
            help="Faithfulness + Contextual Precision + Answer Relevancy (DeepEval methodology, needs LLM API key).",
        )
        st.info("Judge + RAG Triad run **automatically** after every document generation — no checkbox needed there.")

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
        st.info(
            "No evaluation runs found yet. "
            "Generate a PMF document using the **Plant Master File** option in the sidebar."
        )
        return

    # Auto-select the latest run (index 0 — list is sorted newest first)
    labels = [
        f"{r.get('timestamp', '')}  |  {r.get('site_name', '')}  |  "
        f"score={r.get('overall_score', '?')}"
        for r in runs
    ]
    idx = st.selectbox(
        "Evaluation run",
        range(len(runs)),
        format_func=lambda i: labels[i],
        key="tab1_run_sel",
        help="Auto-selected: most recent run. Change to compare older runs.",
    )
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
    rag_triad = ext.get("mean_rag_triad_score") or ext.get("mean_ragas")
    mean_faith = ext.get("mean_faithfulness")
    composite = ext.get("mean_composite", rule_score)

    # ── Smart health summary (plain-English) ─────────────────────────────
    _grade_val = _letter_grade(float(composite or rule_score or 0))
    _passed = float(composite or rule_score or 0) >= 65.0
    _site = run_meta.get("site_name", "")
    _sec_count = eval_data.get("section_count", 0)
    _missing = eval_data.get("missing_required_sections", [])
    _retrieval = eval_data.get("retrieval_coverage", 100)
    _sections = eval_data.get("sections", [])
    _worst = sorted(_sections, key=lambda s: s.get("score", 100))[:3]

    # Colour band
    _health_clr = CLR_SUCCESS if _passed else CLR_DANGER
    _health_label = "Good Quality" if _grade_val in ("A", "B") else (
        "Acceptable" if _grade_val == "C" else "Needs Improvement"
    )
    st.markdown(
        f'<div style="background:{_health_clr}22;border-left:4px solid {_health_clr};'
        f'padding:12px 16px;border-radius:6px;margin-bottom:16px;">'
        f'<span style="font-size:1.1rem;font-weight:700;color:{_health_clr};">'
        f'{_grade_badge_html(_grade_val)} &nbsp; {_health_label}</span><br>'
        f'<span style="color:#374151;">'
        f'Document for <strong>{_site or "unknown site"}</strong> — '
        f'{_sec_count} sections evaluated, '
        f'{_retrieval:.0f}% retrieval coverage'
        + (f', <strong style="color:{CLR_DANGER};">{len(_missing)} missing required sections</strong>' if _missing else '')
        + f'</span></div>',
        unsafe_allow_html=True,
    )

    if _worst:
        _low_names = [s.get("section_key", "?")[:40] for s in _worst if s.get("score", 100) < 75]
        if _low_names:
            st.warning(
                f"Sections needing attention: **{', '.join(_low_names)}** — "
                "see Section Heatmap tab for details."
            )

    # ── Metric cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Rule Score", _fmt(rule_score),
        delta=_fmt(_delta(rule_score, prev_rule)),
        help="Structural checks: min length, required sections, keyword presence",
    )
    c2.metric(
        "Judge Score",
        _fmt(judge_score) if judge_score is not None else "—",
        help="LLM-as-Judge: regulatory quality scored on 5 criteria (factual accuracy, regulatory language, site specificity, completeness, coherence)",
    )
    c3.metric(
        "Faithfulness",
        _fmt(mean_faith, 3) if mean_faith is not None else "—",
        help="DeepEval: fraction of generated claims grounded in source documents (0–1). Low = hallucination.",
    )
    c4.metric(
        "RAG Triad",
        _fmt(rag_triad, 3) if rag_triad is not None else "—",
        help="DeepEval RAG Triad: harmonic mean of Faithfulness + Contextual Precision + Answer Relevancy (0–1)",
    )
    c5.metric("Composite", _fmt(composite))

    # ── Opik-style metric row ────────────────────────────────────────────
    _hallucination = ext.get("mean_hallucination_score")
    _answer_rel = ext.get("mean_answer_relevance_score")
    _reg_tone = ext.get("mean_regulatory_tone_score")
    _opik_composite = ext.get("mean_opik_composite")

    _has_opik = any(v is not None for v in [_hallucination, _answer_rel, _reg_tone])
    if _has_opik:
        st.markdown(
            '<p style="font-size:0.78rem;color:#6b7280;margin:12px 0 4px 0;font-weight:600;">'
            'OPIK-STYLE METRICS (direct continuous scoring)</p>',
            unsafe_allow_html=True,
        )
        oc1, oc2, oc3, oc4 = st.columns(4)
        _hall_disp = f"{(1.0 - _hallucination):.3f}" if _hallucination is not None else "—"
        oc1.metric(
            "Groundedness",
            _hall_disp,
            help="Opik: 1 − Hallucination Score. 1.0 = fully grounded, 0 = complete hallucination. "
                 "Only invented factual claims (names, numbers, certifications) are penalised — "
                 "regulatory boilerplate is NOT penalised.",
        )
        oc2.metric(
            "Answer Relevance",
            _fmt(_answer_rel, 3) if _answer_rel is not None else "—",
            help="Opik: Direct continuous score (0–1) for how well the output addresses the "
                 "section instruction. Faster and more holistic than DeepEval's reverse-question method.",
        )
        oc3.metric(
            "Regulatory Tone",
            _fmt(_reg_tone, 3) if _reg_tone is not None else "—",
            help="PMF-specific metric (not in vanilla Opik): scores language appropriateness for "
                 "EU GMP Annex 4 / ICH Q10 regulatory submission (0 = informal, 1 = exemplary).",
        )
        oc4.metric(
            "Opik Composite",
            _fmt(_opik_composite, 3) if _opik_composite is not None else "—",
            help="Mean of Groundedness + Answer Relevance + Regulatory Tone.",
        )

    # ── MLflow tracking link ─────────────────────────────────────────────
    _run_arts = payload.get("run_artifacts", payload)
    _mlflow_run_id = _run_arts.get("mlflow_run_id")
    _mlflow_url = _run_arts.get("mlflow_ui_url", "http://localhost:5000")
    if _mlflow_run_id:
        st.markdown(
            f'<div style="margin:8px 0 4px 0;">'
            f'<a href="{_mlflow_url}" target="_blank" style="font-size:0.85rem;color:#5340C0;">'
            f'🔗 Open in MLflow UI &nbsp;<span style="color:#9ca3af;font-size:0.75rem;">'
            f'(run: {_mlflow_run_id[:8]}…)</span></a>'
            f'&nbsp;&nbsp;<span style="color:#6b7280;font-size:0.78rem;">'
            f'Run <code>mlflow ui</code> in the project folder to start the server.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "MLflow tracking: run `mlflow ui` in the project folder → "
            "open http://localhost:5000 to compare runs over time."
        )

    # ── Metric tooltips ─────────────────────────────────────────────────
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "**Rule Score** — Structural quality checks: minimum character length, "
            "required sections present, site name referenced.\n\n"
            "**Judge Score** — A second LLM acts as a domain expert and scores the "
            "output on 5 PMF-specific criteria: Factual Accuracy, Regulatory Language, "
            "Site Specificity, Completeness, Structural Coherence. Score: 0–100.\n\n"
            "**Faithfulness** *(DeepEval)* — Extracts every factual claim from the "
            "generated text and checks each one against the retrieved source documents. "
            "Score = supported_claims / total_claims. Low score = hallucination.\n\n"
            "**Contextual Precision** *(DeepEval)* — Checks whether the most relevant "
            "retrieved context chunks are ranked first. Rank-weighted Average Precision.\n\n"
            "**Answer Relevancy** *(DeepEval)* — Generates questions the answer would "
            "address, then measures how closely they match the original instruction.\n\n"
            "**RAG Triad** — Harmonic mean of Faithfulness + Contextual Precision + "
            "Answer Relevancy (0–1 scale). Industry-standard composite for RAG quality.\n\n"
            "**Composite** — Weighted blend: Rule 20% + Judge 55% + RAG Triad 25%.\n\n"
            "---\n\n"
            "**Groundedness** *(Opik-style)* — 1 minus the hallucination score. Single LLM "
            "call asking for a direct continuous score. Only penalises invented factual claims "
            "(names, numbers, certifications) — NOT regulatory boilerplate.\n\n"
            "**Answer Relevance** *(Opik-style)* — How well the output addresses the section "
            "instruction, scored as a direct continuous value (0–1). One LLM call per section.\n\n"
            "**Regulatory Tone** *(PMF-specific, not in vanilla Opik)* — Language quality for "
            "EU GMP Annex 4 / ICH Q10 regulatory submission. 1 = exemplary formal regulatory "
            "language, 0 = completely inappropriate.\n\n"
            "**Opik Composite** — Mean of Groundedness + Answer Relevance + Regulatory Tone.\n\n"
            "*DeepEval vs Opik*: DeepEval extracts discrete claims and checks each one (binary, "
            "multi-call). Opik asks the LLM for one direct continuous score per section (faster, "
            "captures nuance). Both frameworks complement each other."
        )

    # ── Grade badge ──────────────────────────────────────────────────────
    grade = ext.get("overall_grade") or _letter_grade(float(composite or rule_score or 0))
    passed = float(composite or rule_score or 0) >= 65.0
    badge_html = _grade_badge_html(grade)
    status = "PASS" if passed else "FAIL"
    status_clr = CLR_SUCCESS if passed else CLR_DANGER
    _framework_label = ""
    _fw = ext.get("framework", "")
    if "opik_style" in _fw:
        _framework_label = ' &nbsp;<span style="font-size:0.75rem;color:#6b7280;font-weight:400;">DeepEval + Opik-style</span>'
    elif "deepeval" in _fw or ext.get("mean_rag_triad_score") is not None:
        _framework_label = ' &nbsp;<span style="font-size:0.75rem;color:#6b7280;font-weight:400;">DeepEval RAG Triad</span>'
    st.markdown(
        f"{badge_html} &nbsp; "
        f'<span style="color:{status_clr};font-weight:700;">{status}</span>'
        f"{_framework_label}",
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
    _rag_val = ext.get("mean_rag_triad_score") or ext.get("mean_ragas")
    _hall = ext.get("mean_hallucination_score")
    metrics = [
        ("Rule Score", _fmt(rule_score)),
        ("Judge Score", _fmt(ext.get("mean_judge_normalized"))),
        ("Faithfulness (DeepEval)", _fmt(ext.get("mean_faithfulness"), 4)),
        ("RAG Triad Score (DeepEval)", _fmt(_rag_val, 4)),
        ("Composite", _fmt(ext.get("mean_composite", rule_score))),
        ("Groundedness (Opik)", _fmt((1.0 - _hall) if _hall is not None else None, 4)),
        ("Answer Relevance (Opik)", _fmt(ext.get("mean_answer_relevance_score"), 4)),
        ("Regulatory Tone (Opik)", _fmt(ext.get("mean_regulatory_tone_score"), 4)),
        ("Opik Composite", _fmt(ext.get("mean_opik_composite"), 4)),
        ("Retrieval Coverage", f"{eval_data.get('retrieval_coverage', 0)}%"),
        ("Framework", ext.get("framework", "rule-based")),
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
    """Smart evaluation panel — rule-based runs automatically, reference metrics optional."""
    st.subheader("Quick Text Evaluation")
    st.caption(
        "Paste any generated PMF text to evaluate it instantly. "
        "**Rule-based checks run without any reference text.** "
        "Add a reference text on the right to also compute BLEU / ROUGE / BERTScore."
    )

    # ── Auto-populate from the last generated run ─────────────────────
    auto_gen = ""
    auto_sec = "EXECUTIVE SUMMARY"
    _runs = _load_runs()
    if _runs:
        _rf = _runs[0].get("run_file", "")
        if _rf and os.path.exists(_rf):
            try:
                _payload = _load_run_payload(_rf)
                for _sec in _payload.get("run_artifacts", {}).get("sections", []):
                    _gt = _sec.get("generated_text", "")
                    if _gt.strip() and "static" not in _sec.get("section_key", "").lower():
                        auto_gen = _gt[:2500]
                        auto_sec = _sec.get("section_key", auto_sec)
                        break
            except Exception:
                pass

    lc1, lc2 = st.columns(2)
    with lc1:
        gen_text = st.text_area(
            "Generated text",
            value=auto_gen,
            height=200,
            key="live_gen_txt",
            help="Auto-populated from your last run. Edit or paste any text to evaluate.",
        )
    with lc2:
        ref_text = st.text_area(
            "Reference text (optional)",
            height=200,
            key="live_ref_txt",
            help=(
                "Leave blank — rule-based and judge metrics work without it. "
                "Paste a ground-truth PMF section here to also compute BLEU / ROUGE / BERTScore."
            ),
        )

    live_section = st.text_input(
        "Section type", value=auto_sec, key="live_sec_key",
        help="E.g. EXECUTIVE SUMMARY, SITE DESCRIPTION, MANUFACTURING PROCESSES",
    )

    if not st.button("Evaluate Now", key="live_eval_now_btn", type="primary"):
        return

    if not gen_text.strip():
        st.warning("Paste some generated text first.")
        return

    has_ref = bool(ref_text.strip())

    # ── Phase 1: Rule-based checks (always — no reference needed) ─────
    rule_score = 0.0
    rule_checks: Dict[str, Any] = {}
    missing_kw: List[str] = []
    with st.spinner("Running rule-based quality checks..."):
        try:
            _rule_result = score_section(
                section_key=live_section,
                section_text=gen_text,
                rules=get_eval_rules(),
                context={},
            )
            rule_score = float(_rule_result.get("score", 0))
            rule_checks = _rule_result.get("checks", {})
            missing_kw = _rule_result.get("missing_keywords", [])
        except Exception as exc:
            logger.warning("Rule scoring failed in live eval: %s", exc)

    # ── Phase 2: Lexical metrics (BLEU / ROUGE) — needs reference ─────
    lex_result: Dict[str, Any] = {}
    if HAS_LEXICAL and has_ref:
        with st.spinner("Computing BLEU / ROUGE..."):
            try:
                lex_result = LexicalMetrics.compute_all_lexical(gen_text, ref_text)
            except Exception as exc:
                logger.warning("Lexical metrics failed in live eval: %s", exc)

    # ── Phase 3: BERTScore — needs reference + semantic toggle ────────
    bert_f1: Any = None
    sim_result: float = 0.0
    if st.session_state.get("toggle_semantic", False) and HAS_SEMANTIC and has_ref:
        with st.spinner("Computing BERTScore (this may take a few seconds)..."):
            try:
                sm = SemanticMetrics(model_type="distilbert-base-uncased")
                bs = sm.compute_bertscore([gen_text], [ref_text])
                bert_f1 = bs.get("bertscore_f1_mean")
                sim_result = sm.compute_semantic_similarity(gen_text, ref_text)
            except Exception as exc:
                logger.warning("BERTScore failed in live eval: %s", exc)

    # ── Phase 4: LLM Judge (if enabled, ~30-60s) ─────────────────────
    judge_result: Dict[str, Any] = {}
    if st.session_state.get("toggle_judge", False):
        with st.spinner("Running LLM Judge (30-60 s)..."):
            try:
                from src.eval.eval_judge import PMFJudge
                from dotenv import load_dotenv
                load_dotenv()
                # Prefer Azure OpenAI (already configured) over Anthropic
                _azure_key = os.getenv("AZURE_KEY", os.getenv("AZURE_OPENAI_API_KEY", ""))
                _azure_ep = os.getenv("AZURE_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
                _azure_ver = os.getenv("AZURE_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"))
                _azure_name = os.getenv("AZURE_NAME", "gpt-4o")

                if _azure_key and _azure_ep:
                    judge = PMFJudge(
                        provider="azure_openai",
                        model=_azure_name,
                        api_key=_azure_key,
                        azure_endpoint=_azure_ep,
                        azure_api_version=_azure_ver,
                        cache_enabled=False,
                    )
                else:
                    # Fallback to sidebar Anthropic key
                    api_key = st.session_state.get("sidebar_anthropic_key", "")
                    model = st.session_state.get("sidebar_judge_model", "claude-sonnet-4-6")
                    judge = PMFJudge(provider="anthropic", model=model, api_key=api_key, cache_enabled=False)

                judge_result = judge.score_section(
                    section_key=live_section,
                    section_instruction="Evaluate this PMF section for regulatory quality.",
                    retrieved_context="",
                    generated_output=gen_text,
                    site_name="",
                    reference_output=ref_text,
                )
            except Exception as exc:
                logger.warning("Judge failed in live eval: %s", exc)
                judge_result = {"judge_error": True, "error": str(exc)}

    # ── Phase 5: DeepEval RAG Triad (if enabled) ──────────────────────
    rag_result: Dict[str, Any] = {}
    if st.session_state.get("toggle_rag", False):
        with st.spinner("Running DeepEval RAG Triad (Faithfulness + Contextual Precision + Answer Relevancy)..."):
            try:
                from src.eval.eval_rag import RAGEvaluator
                from openai import AzureOpenAI
                from dotenv import load_dotenv
                load_dotenv()
                _azure_key = os.getenv("AZURE_KEY", os.getenv("AZURE_OPENAI_API_KEY", ""))
                _azure_ep = os.getenv("AZURE_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
                _azure_ver = os.getenv("AZURE_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"))
                _azure_name = os.getenv("AZURE_NAME", "gpt-4o")

                _client = AzureOpenAI(api_key=_azure_key, api_version=_azure_ver, azure_endpoint=_azure_ep)
                rag_ev = RAGEvaluator(llm_client=_client, model=_azure_name, cache_enabled=False)
                rag_result = rag_ev.evaluate_section(
                    section_key=live_section,
                    section_instruction=live_section,
                    retrieved_chunks=[ref_text] if ref_text.strip() else [],
                    generated_answer=gen_text,
                )
            except Exception as exc:
                logger.warning("RAG Triad eval failed in live eval: %s", exc)
                rag_result = {"error": str(exc)}

    # ── Results display ───────────────────────────────────────────────
    st.subheader("Evaluation Results")

    grade = _letter_grade(rule_score)
    badge_html = _grade_badge_html(grade)
    passed = rule_score >= 65.0
    status = "PASS" if passed else "NEEDS IMPROVEMENT"
    status_clr = CLR_SUCCESS if passed else CLR_WARNING
    st.markdown(
        f"{badge_html} &nbsp;"
        f'<span style="color:{status_clr};font-weight:700;">{status}</span>'
        f"&nbsp; Rule Score: <strong>{_fmt(rule_score)}/100</strong>",
        unsafe_allow_html=True,
    )

    st.write("")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rule Score", _fmt(rule_score),
              help="Structural quality: min length, required sections, site name presence")
    m2.metric("BLEU", _fmt(lex_result.get("bleu"), 2) if lex_result else "N/A",
              help="Lexical overlap with reference (0–100). Needs reference text.")
    m3.metric("ROUGE-1 F1", _fmt(lex_result.get("rouge1_fmeasure"), 4) if lex_result else "N/A",
              help="Unigram overlap F1 with reference. Needs reference text.")
    _faith_live = rag_result.get("faithfulness")
    _rt_live = rag_result.get("rag_triad_score") or rag_result.get("ragas_score")
    m4.metric("Faithfulness",
              _fmt(_faith_live, 3) if _faith_live is not None else "N/A",
              help="DeepEval: claims grounded in source context. Enable 'DeepEval RAG Triad' in sidebar.")
    m5.metric("RAG Triad",
              _fmt(_rt_live, 3) if _rt_live is not None else "N/A",
              help="DeepEval composite: Faithfulness + Contextual Precision + Answer Relevancy")

    # Rule check breakdown
    if rule_checks:
        with st.expander("Rule Check Details", expanded=True):
            for check_name, check_passed in rule_checks.items():
                icon = "✓" if check_passed else "✗"
                clr = CLR_SUCCESS if check_passed else CLR_DANGER
                label = check_name.replace("_", " ").replace("passed", "").strip().title()
                st.markdown(
                    f'<span style="color:{clr};font-weight:600;">{icon} {label}</span>',
                    unsafe_allow_html=True,
                )
        if missing_kw:
            st.warning(f"Missing keywords: {', '.join(missing_kw)}")

    if not has_ref:
        st.info(
            "**BLEU / ROUGE / BERTScore** require a reference text — paste a ground-truth "
            "PMF section in the right box to compute them. "
            "The rule-based score above works without any reference."
        )

    if not st.session_state.get("toggle_semantic") and has_ref and HAS_SEMANTIC:
        st.caption(
            "BERTScore is available but disabled. "
            "Enable 'Run Semantic (BERTScore)' in the sidebar to compute it."
        )

    # ── Judge results ─────────────────────────────────────────────────
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
            f"**Judge Score:** {_fmt(norm)} / 100 &nbsp; {_grade_badge_html(j_grade)}",
            unsafe_allow_html=True,
        )

        with st.expander("Judge Details"):
            st.markdown(f"**Strengths:** {', '.join(judge_result.get('strengths', [])) or '—'}")
            st.markdown(f"**Weaknesses:** {', '.join(judge_result.get('weaknesses', [])) or '—'}")
            critical = judge_result.get("critical_issues", [])
            if critical:
                st.error(f"Critical Issues: {', '.join(critical)}")
            st.markdown(f"**Notes:** {judge_result.get('evaluation_notes', '—')}")

    elif judge_result.get("judge_error"):
        st.warning(
            f"Judge evaluation failed: {judge_result.get('error', 'unknown')}. "
            "Check Azure OpenAI credentials in .env file."
        )

    # ── DeepEval RAG Triad results ────────────────────────────────────
    if rag_result and not rag_result.get("error"):
        st.subheader("DeepEval RAG Triad")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Faithfulness", _fmt(rag_result.get("faithfulness"), 3),
                   help="Claims grounded in context (0–1)")
        rc2.metric("Ctx Precision", _fmt(rag_result.get("contextual_precision") or rag_result.get("context_precision"), 3),
                   help="Relevant contexts ranked first (0–1)")
        rc3.metric("Answer Relevancy", _fmt(rag_result.get("answer_relevancy"), 3),
                   help="Answer addresses the question (0–1)")
        _rt = rag_result.get("rag_triad_score") or rag_result.get("ragas_score")
        rc4.metric("RAG Triad", _fmt(_rt, 3), help="Harmonic mean of the three metrics (0–1)")

        with st.expander("RAG Triad Details"):
            claims = rag_result.get("faithfulness_claims", [])
            if claims:
                st.markdown(f"**Claims extracted:** {len(claims)}")
                for c in claims[:5]:
                    icon = "✓" if c.get("supported") else "✗"
                    clr = CLR_SUCCESS if c.get("supported") else CLR_DANGER
                    st.markdown(
                        f'<span style="color:{clr};">{icon}</span> {c.get("claim", "?")}',
                        unsafe_allow_html=True,
                    )
            gen_qs = rag_result.get("generated_questions", [])
            if gen_qs:
                st.markdown(f"**Generated questions (Answer Relevancy):** {', '.join(gen_qs)}")
    elif rag_result.get("error"):
        st.warning(f"DeepEval RAG Triad failed: {rag_result.get('error')}. Check Azure credentials.")

    # ── Full JSON (expandable) ────────────────────────────────────────
    with st.expander("Full Results (JSON)"):
        st.json({
            "rule_score": rule_score,
            "rule_checks": rule_checks,
            "lexical": lex_result if lex_result else "N/A — no reference text",
            "deepeval_rag_triad": rag_result if rag_result else None,
            "judge": judge_result if judge_result else None,
        })


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — SECTION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def _cell_color(val: Any) -> str:
    """Return a hex background color based on a 0-100 score."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "#f3f4f6"  # light grey for missing
    v = float(val)
    if v >= 80:
        return "#86efac"  # green
    if v >= 50:
        return "#fde68a"  # yellow
    return "#fca5a5"      # red


def _render_heatmap_html(df: "pd.DataFrame", metric_cols: List[str]) -> None:
    """Render a color-coded heatmap table using pure HTML — no matplotlib or plotly needed."""
    rows_html = ""
    for section, row in df[metric_cols].iterrows():
        cells = f'<td style="padding:8px 12px;font-weight:600;background:#f8fafc;border:1px solid #e2e8f0;white-space:nowrap;">{section}</td>'
        for col in metric_cols:
            val = row[col]
            bg = _cell_color(val)
            label = f"{float(val):.1f}" if (val is not None and not (isinstance(val, float) and pd.isna(val))) else "—"
            cells += (
                f'<td style="padding:8px 14px;text-align:center;'
                f'background:{bg};border:1px solid #e2e8f0;">{label}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    header_cells = '<th style="padding:8px 12px;background:#5340C0;color:white;border:1px solid #4338ca;">Section</th>'
    for col in metric_cols:
        header_cells += (
            f'<th style="padding:8px 14px;background:#5340C0;color:white;'
            f'text-align:center;border:1px solid #4338ca;">{col}</th>'
        )

    html = f"""
    <div style="overflow-x:auto;margin-bottom:16px;">
      <table style="border-collapse:collapse;width:100%;font-size:14px;font-family:sans-serif;">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_tab_heatmap(runs: List[Dict[str, Any]]) -> None:
    """Multi-metric heatmap (rows=sections, cols=metrics) with detail view."""

    if not runs:
        st.info("No evaluation runs available. Generate a PMF document first to see results here.")
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
        st.warning(f"Run file not found: {run_file}")
        return

    payload = _load_run_payload(run_file)
    eval_data = payload.get("evaluation", {}).get("document_scores", {})
    sections = eval_data.get("sections", [])
    run_sections = payload.get("run_artifacts", payload).get("sections", [])

    if not sections:
        st.warning("No section evaluation data available in this run. Check if the evaluation completed successfully.")
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
            "Judge Score": judge.get("normalized_score"),
            "Faithfulness": _safe_pct(rag.get("faithfulness")),
            "Ctx Precision": _safe_pct(rag.get("contextual_precision") or rag.get("context_precision")),
            "RAG Triad": _safe_pct(rag.get("rag_triad_score") or rag.get("ragas_score")),
        })

    df = pd.DataFrame(heatmap_rows).set_index("Section")

    metric_cols = ["Rule Score", "Judge Score", "Faithfulness", "Ctx Precision", "RAG Triad"]

    # Ensure all metric columns are numeric (None -> NaN)
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    st.subheader("Section Heatmap")
    st.caption("Green (>80) | Yellow (50-80) | Red (<50). Blank = metric not computed.")

    _render_heatmap_html(df, metric_cols)

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
    sc2.metric("Judge Score", _fmt(judge.get("normalized_score")))
    sc3.metric("Faithfulness", _fmt(rag.get("faithfulness"), 4))
    _cp = rag.get("contextual_precision") or rag.get("context_precision")
    sc4.metric("Ctx Precision", _fmt(_cp, 4))
    _rt = rag.get("rag_triad_score") or rag.get("ragas_score")
    sc5.metric("RAG Triad", _fmt(_rt, 4))

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

# DEPRECATED: Removed from dashboard
# def _render_tab_trends(runs: List[Dict[str, Any]]) -> None:
#     """Time-series charts and regression alerts."""
#
#     if len(runs) < 2:
#         st.info("Need at least 2 runs for trend analysis.")
#         return
#
##     df = pd.DataFrame(runs)
#    df["overall_score"] = pd.to_numeric(df.get("overall_score"), errors="coerce")
#    df = df.sort_values("timestamp").reset_index(drop=True)
#
#    # ── Regression alert ─────────────────────────────────────────────────
#    latest = df.iloc[-1]["overall_score"]
#    prev = df.iloc[-2]["overall_score"]
#    if pd.notna(latest) and pd.notna(prev):
#        drop = prev - latest
#        if drop > 5:
#            latest_ts = df.iloc[-1].get("timestamp", "?")
#            st.markdown(
#                f'<div style="background:{CLR_DANGER};color:white;padding:12px;'
#                f'border-radius:8px;margin-bottom:16px;">'
#                f'<strong>Regression detected:</strong> Rule score dropped from '
#                f'{prev:.1f} to {latest:.1f} (delta {-drop:+.1f}) on {latest_ts}'
#                f'</div>',
#                unsafe_allow_html=True,
#            )
#
#    # ── Composite score over time ────────────────────────────────────────
#    st.subheader("Score Trend Over Time")
#
#    if HAS_PLOTLY:
#        fig = px.line(df, x="timestamp", y="overall_score", markers=True,
#                      labels={"overall_score": "Rule Score (%)", "timestamp": "Run"})
#        fig.update_traces(line_color=CLR_PRIMARY)
#        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
#        st.plotly_chart(fig, use_container_width=True)
#    else:
#        st.line_chart(df.set_index("timestamp")["overall_score"])
#
#    # ── Multi-metric trend ───────────────────────────────────────────────
#    # Load extended data from each run payload to build multi-metric trend
#    multi_rows: List[Dict[str, Any]] = []
#    for _, row in df.iterrows():
#        rf = row.get("run_file", "")
#        if rf and os.path.exists(rf):
#            p = _load_run_payload(rf)
#            ext = _extract_extended(p)
#            ed = p.get("evaluation", {}).get("document_scores", {})
#            multi_rows.append({
#                "timestamp": row["timestamp"],
#                "Rule Score": ed.get("overall_score"),
#                "BERTScore F1": _safe_pct(ext.get("mean_bertscore_f1")),
#                "Judge Score": ext.get("mean_judge_normalized"),
#                "Faithfulness": _safe_pct(ext.get("mean_faithfulness")),
#                "RAGAS": _safe_pct(ext.get("mean_ragas")),
#            })
#
#    if multi_rows:
#        mdf = pd.DataFrame(multi_rows).set_index("timestamp")
#        # Only show columns that have at least one non-null value
#        active_cols = [c for c in mdf.columns if mdf[c].notna().any()]
#        if active_cols:
#            st.subheader("Multi-Metric Trend")
#            if HAS_PLOTLY:
#                fig = go.Figure()
#                colors = [CLR_PRIMARY, CLR_SUCCESS, "#3B82F6", CLR_WARNING, CLR_DANGER]
#                for i, col in enumerate(active_cols):
#                    fig.add_trace(go.Scatter(
#                        x=mdf.index, y=mdf[col], name=col, mode="lines+markers",
#                        line=dict(color=colors[i % len(colors)]),
#                    ))
#                fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
#                                  yaxis_title="Score (0-100)")
#                st.plotly_chart(fig, use_container_width=True)
#            else:
#                st.line_chart(mdf[active_cols])
#
#    # ── Section score distribution (histogram) ───────────────────────────
#    st.subheader("Section Score Distribution (Latest Run)")
#    latest_file = df.iloc[-1].get("run_file", "")
#    if latest_file and os.path.exists(latest_file):
#        lp = _load_run_payload(latest_file)
#        secs = lp.get("evaluation", {}).get("document_scores", {}).get("sections", [])
#        sec_scores = [s.get("score", 0) for s in secs]
#        if sec_scores:
#            if HAS_PLOTLY:
#                fig = px.histogram(
#                    x=sec_scores, nbins=10,
#                    labels={"x": "Section Rule Score", "count": "Count"},
#                    color_discrete_sequence=[CLR_PRIMARY],
#                )
#                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
#                st.plotly_chart(fig, use_container_width=True)
#            else:
#                hist_df = pd.DataFrame({"Score": sec_scores})
#                st.bar_chart(hist_df["Score"].value_counts().sort_index())


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON (unchanged from previous)
# ═══════════════════════════════════════════════════════════════════════════

#def _render_tab_model_comparison() -> None:
#    """Upload JSON results and compare models."""
#    st.subheader("Model Comparison")
#
#    uploaded = st.file_uploader(
#        "Upload evaluation result JSON files (one per model run)",
#        type=["json"], accept_multiple_files=True, key="mc_upload",
#    )
#    if not uploaded:
#        st.info("Upload two or more JSON result files to compare models.")
#        return
#
#    model_data: List[Dict[str, Any]] = []
#    for f in uploaded:
#        try:
#            data = json.loads(f.read().decode("utf-8"))
#            data["_source_file"] = f.name
#            model_data.append(data)
#        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
#            st.warning(f"Could not parse {f.name}: {exc}")
#
#    if len(model_data) < 2:
#        st.warning("Upload at least 2 files to compare.")
#        return
#
#    rows: List[Dict[str, Any]] = []
#    for d in model_data:
#        label = d.get("_source_file", "unknown")
#        judge = d.get("judge_scores") or {}
#        rag = d.get("rag_scores") or {}
#        lex = d.get("lexical_scores") or {}
#        sem = d.get("semantic_scores") or {}
#        rows.append({
#            "Model / File": label,
#            "Section": d.get("section_key", "—"),
#            "Composite": d.get("composite_score", 0),
#            "Grade": d.get("grade", "?"),
#            "Rule Score": d.get("rule_score"),
#            "BLEU": lex.get("bleu"),
#            "BERTScore F1": sem.get("bertscore_f1_mean"),
#            "Judge Norm.": judge.get("normalized_score"),
#            "RAGAS": rag.get("ragas_score"),
#        })
#
#    cdf = pd.DataFrame(rows)
#    st.dataframe(
#        cdf.style.background_gradient(subset=["Composite"], cmap="RdYlGn",
#                                       vmin=0, vmax=100),
#        use_container_width=True, hide_index=True,
#    )
#
#    # Radar chart
#    st.subheader("Metric Radar")
#    radar_metrics = ["Rule Score", "BLEU", "BERTScore F1", "Judge Norm.", "RAGAS"]
#    if HAS_PLOTLY:
#        fig = go.Figure()
#        for _, row in cdf.iterrows():
#            vals = []
#            for m in radar_metrics:
#                v = row.get(m)
#                if v is None:
#                    vals.append(0)
#                elif m in ("BERTScore F1", "RAGAS"):
#                    vals.append(float(v) * 100)
#                else:
#                    vals.append(float(v))
#            fig.add_trace(go.Scatterpolar(
#                r=vals + [vals[0]], theta=radar_metrics + [radar_metrics[0]],
#                fill="toself", name=str(row.get("Model / File", "")), opacity=0.6,
#            ))
#        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
#                          height=400, margin=dict(l=40, r=40, t=20, b=20))
#        st.plotly_chart(fig, use_container_width=True)
#    else:
#        st.bar_chart(cdf[["Model / File"] + radar_metrics].set_index("Model / File"))
#
#    # Cost analysis
#    st.subheader("Cost Analysis")
#    cost_entries: Dict[str, float] = {}
#    cost_cols = st.columns(min(len(model_data), 4))
#    for i, d in enumerate(model_data):
#        label = d.get("_source_file", f"Model {i+1}")
#        with cost_cols[i % len(cost_cols)]:
#            cost = st.number_input(f"Cost ($) — {label}", min_value=0.0,
#                                   value=0.0, step=0.01, key=f"cost_{i}")
#            cost_entries[label] = cost
#
#    if any(c > 0 for c in cost_entries.values()):
#        st.subheader("Pareto Recommendation")
#        pareto_rows = []
#        for _, row in cdf.iterrows():
#            label = row["Model / File"]
#            cost = cost_entries.get(label, 0)
#            comp = row.get("Composite", 0) or 0
#            eff = round(comp / cost, 2) if cost > 0 else 0
#            pareto_rows.append({"Model": label, "Composite": comp,
#                                "Cost ($)": cost, "Score / $": eff})
#        pdf = pd.DataFrame(pareto_rows).sort_values("Score / $", ascending=False)
#        st.dataframe(pdf, use_container_width=True, hide_index=True)
#        best = pdf.iloc[0]
#        st.success(
#            f"Recommended: **{best['Model']}** — "
#            f"Composite {best['Composite']:.1f} at ${best['Cost ($)']:.2f} "
#            f"({best['Score / $']:.1f} points per dollar)"
#        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — BENCHMARK MANAGEMENT (unchanged from previous)
# ═══════════════════════════════════════════════════════════════════════════

#def _render_tab_benchmark() -> None:
#    """Benchmark management interface."""
#    if not HAS_BENCHMARK:
#        st.info("Benchmark loader not available.")
#        return
#
#    st.subheader("Benchmark Management")
#    loader = BenchmarkLoader(BENCHMARK_DIR)
#    stats = loader.get_statistics()
#
#    c1, c2, c3, c4 = st.columns(4)
#    c1.metric("Total Cases", stats.get("total_cases", 0))
#    c2.metric("With Reference", stats.get("cases_with_reference", 0))
#    c3.metric("Expert Annotated", stats.get("cases_with_expert_scores", 0))
#    c4.metric("Auto Scored", stats.get("cases_with_automated_scores", 0))
#
#    # Filterable table
#    st.subheader("Browse Cases")
#    fc1, fc2, fc3 = st.columns(3)
#    with fc1:
#        ft = st.selectbox("Section Type",
#                          ["All"] + list(stats.get("by_section_type", {}).keys()),
#                          key="bm_ft")
#    with fc2:
#        fd = st.selectbox("Difficulty",
#                          ["All"] + list(stats.get("by_difficulty", {}).keys()),
#                          key="bm_fd")
#    with fc3:
#        fs = st.text_input("Search section key", "", key="bm_fs")
#
#    filters: Dict[str, Any] = {}
#    if ft != "All":
#        filters["section_type"] = ft
#    if fd != "All":
#        filters["difficulty"] = fd
#    if fs.strip():
#        filters["section_key"] = fs.strip()
#
#    cases = loader.load_cases(filters if filters else None)
#    if cases:
#        rows = [{
#            "Case ID": c.get("case_id", ""),
#            "Section": c.get("section_key", ""),
#            "Site": c.get("site_name", ""),
#            "Type": c.get("section_type", ""),
#            "Difficulty": c.get("difficulty", ""),
#            "Ref Len": len(c.get("reference_output", "")),
#            "Tags": ", ".join(c.get("tags", [])),
#        } for c in cases]
#        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
#    else:
#        st.info("No cases match filters.")
#
#    # Add case form
#    st.subheader("Add New Case")
#    with st.expander("New benchmark case form"):
#        nsk = st.text_input("Section Key", key="bn_sk")
#        nsite = st.text_input("Site Name", key="bn_site")
#        ninstr = st.text_area("Section Instruction", height=80, key="bn_instr")
#        nquery = st.text_input("Retrieval Query", key="bn_query")
#        nref = st.text_area("Reference Output", height=120, key="bn_ref")
#        ntype = st.selectbox("Type", ["text", "table", "image", "static"], key="bn_type")
#        ndiff = st.selectbox("Difficulty", ["easy", "medium", "hard"], key="bn_diff")
#        ntags = st.text_input("Tags (comma-separated)", key="bn_tags")
#        if st.button("Add Case", key="bn_add"):
#            if not nsk.strip() or not nref.strip():
#                st.warning("Section Key and Reference Output are required.")
#            else:
#                new_case: Dict[str, Any] = {
#                    "created_at": datetime.now().isoformat(),
#                    "created_by": "human", "validated_by": None,
#                    "site_name": nsite, "section_key": nsk,
#                    "section_instruction": ninstr, "retrieval_query": nquery,
#                    "source_documents": [], "retrieved_context": "",
#                    "generated_output": {}, "reference_output": nref,
#                    "expert_scores": {}, "automated_scores": {},
#                    "tags": [t.strip() for t in ntags.split(",") if t.strip()],
#                    "difficulty": ndiff, "section_type": ntype,
#                }
#                try:
#                    cid = loader.add_case(new_case, validate=True)
#                    st.success(f"Case **{cid}** added.")
#                except ValueError as exc:
#                    st.error(f"Validation failed: {exc}")
#
#    # Export + Run
#    st.subheader("Export & Run")
#    ec1, ec2 = st.columns(2)
#    with ec1:
#        if st.button("Export to CSV", key="bn_csv"):
#            csv_path = os.path.join(BENCHMARK_DIR, "benchmark_export.csv")
#            loader.export_to_csv(csv_path)
#            st.success(f"Exported to `{csv_path}`")
#            with open(csv_path, "r", encoding="utf-8") as fh:
#                st.download_button("Download CSV", fh.read(),
#                                   file_name="benchmark_export.csv",
#                                   mime="text/csv", key="bn_dl")
#    with ec2:
#        do_run = st.button("Run Full Benchmark", key="bn_run")
#    if do_run:
#        all_cases = loader.load_cases()
#        if not all_cases:
#            st.warning("No cases.")
#            return
#        try:
#            from healthark_eval import EvalSuite
#        except ImportError:
#            st.error("healthark_eval not available.")
#            return
#        suite = EvalSuite(task="pmf", run_judge=False, run_rag=False,
#                          run_semantic=False)
#        prog = st.progress(0, text="Evaluating...")
#        res: List[Dict[str, Any]] = []
#        for i, case in enumerate(all_cases):
#            ref = case.get("reference_output", "")
#            r = suite.run(generated=ref, reference=ref,
#                          section_key=case.get("section_key", ""),
#                          section_instruction=case.get("section_instruction", ""),
#                          site_name=case.get("site_name", ""))
#            res.append({"Case ID": case.get("case_id"), "Section": case.get("section_key"),
#                         "Composite": r.composite_score, "Grade": r.grade,
#                         "Rule Score": r.rule_score})
#            prog.progress((i + 1) / len(all_cases),
#                          text=f"Evaluated {i+1}/{len(all_cases)}...")
#        prog.empty()
#        st.success(f"Benchmark complete: {len(res)} cases.")
#        rdf = pd.DataFrame(res)
#        st.dataframe(rdf, use_container_width=True, hide_index=True)
#        st.metric("Mean Composite", f"{rdf['Composite'].mean():.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — PERFORMANCE (latency · failures · improvements)
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_performance(runs: List[Dict[str, Any]]) -> None:
    """Latency breakdown, failure analysis, and improvement recommendations."""

    if not runs:
        st.info("No runs available. Generate a PMF document first.")
        return

    # ── Run selector ──────────────────────────────────────────────────
    labels = [
        f"{r.get('timestamp', '')}  |  {r.get('site_name', '')}  |  score={r.get('overall_score', '?')}"
        for r in runs
    ]
    idx = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i],
                       key="perf_run_sel")
    run_meta = runs[idx]
    payload = _load_run_payload(run_meta.get("run_file", ""))
    run_arts = payload.get("run_artifacts", payload)
    perf = run_arts.get("performance_report", {})

    # ── View toggle: Plain English / Technical ─────────────────────────
    view = st.radio("View mode", ["Plain English", "Technical"], horizontal=True,
                    key="perf_view_mode",
                    help="Plain English: for non-technical stakeholders. Technical: for engineers.")
    is_tech = view == "Technical"

    st.divider()

    # ── Summary banner ────────────────────────────────────────────────
    summary = perf.get("summary_technical" if is_tech else "summary_plain", "")
    if summary:
        st.info(summary)
    else:
        st.info("No performance data available for this run. Performance tracking is captured from the next document generation onwards.")
        _render_performance_legend()
        return

    # ── Overall timing cards ──────────────────────────────────────────
    ot = perf.get("overall_timing", {})
    total_s = (ot.get("total_pipeline_ms") or 0) / 1000
    gen_s   = (ot.get("total_generation_ms") or 0) / 1000
    ret_s   = (ot.get("total_retrieval_ms") or 0) / 1000
    eval_s  = (ot.get("total_eval_ms") or 0) / 1000
    avg_s   = (ot.get("avg_section_ms") or 0) / 1000

    st.subheader("Overall Timing")
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    tc1.metric("Total Time",       f"{total_s:.1f}s" if total_s else "—",
               help="End-to-end pipeline: retrieval + generation + evaluation")
    tc2.metric("LLM Generation",   f"{gen_s:.1f}s"   if gen_s  else "—",
               help="Cumulative time spent calling the LLM across all sections")
    tc3.metric("Retrieval",        f"{ret_s:.1f}s"   if ret_s  else "—",
               help="Cumulative time spent querying the vector database")
    tc4.metric("Evaluation",       f"{eval_s:.1f}s"  if eval_s else "—",
               help="Cumulative time running DeepEval + Opik metrics")
    tc5.metric("Avg / Section",    f"{avg_s:.1f}s"   if avg_s  else "—",
               help="Average time per section (total ÷ section count)")

    # ── Time breakdown donut chart ────────────────────────────────────
    if HAS_PLOTLY and gen_s + ret_s + eval_s > 0:
        other_s = max(0, total_s - gen_s - ret_s - eval_s)
        fig_donut = go.Figure(go.Pie(
            labels=["LLM Generation", "Retrieval", "Evaluation", "Other"],
            values=[gen_s, ret_s, eval_s, other_s],
            hole=0.55,
            marker_colors=["#5340C0", "#1D9E75", "#BA7517", "#d1d5db"],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}s (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            margin=dict(t=30, b=10, l=10, r=10),
            height=260,
            showlegend=False,
            title_text="Time Breakdown",
            title_x=0.5,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # ── Per-section latency bar chart ─────────────────────────────────
    st.subheader("Section Latency")
    section_timings = perf.get("section_timings", [])
    if section_timings:
        df_timing = pd.DataFrame([
            {
                "Section": t["section_key"][:45],
                "Retrieval (s)":  round((t.get("retrieval_ms") or 0) / 1000, 2),
                "Generation (s)": round((t.get("generation_ms") or 0) / 1000, 2),
                "Evaluation (s)": round((t.get("eval_ms") or 0) / 1000, 2),
                "Total (s)":      round((t.get("total_ms") or 0) / 1000, 2),
                "Static":         t.get("is_static", False),
            }
            for t in section_timings
        ])
        # Sort slowest first
        df_timing = df_timing.sort_values("Total (s)", ascending=False)

        if HAS_PLOTLY:
            fig_bar = px.bar(
                df_timing,
                x="Total (s)",
                y="Section",
                orientation="h",
                color="Total (s)",
                color_continuous_scale=["#1D9E75", "#BA7517", "#D85A30"],
                labels={"Total (s)": "Time (seconds)", "Section": ""},
                title="Time per Section (slowest first)",
                height=max(300, len(df_timing) * 28),
            )
            fig_bar.update_coloraxes(showscale=False)
            fig_bar.update_layout(margin=dict(t=40, b=20, l=10, r=20), yaxis_autorange="reversed")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Stacked breakdown chart
            df_stack = df_timing[["Section", "Retrieval (s)", "Generation (s)", "Evaluation (s)"]].copy()
            fig_stack = px.bar(
                df_stack.melt(id_vars="Section", var_name="Phase", value_name="Time (s)"),
                x="Time (s)",
                y="Section",
                color="Phase",
                orientation="h",
                color_discrete_map={
                    "Retrieval (s)":  "#1D9E75",
                    "Generation (s)": "#5340C0",
                    "Evaluation (s)": "#BA7517",
                },
                title="Phase Breakdown per Section",
                height=max(300, len(df_timing) * 28),
            )
            fig_stack.update_layout(margin=dict(t=40, b=20, l=10, r=20), yaxis_autorange="reversed",
                                    legend_title_text="Phase")
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.dataframe(df_timing[["Section", "Retrieval (s)", "Generation (s)", "Evaluation (s)", "Total (s)"]],
                         use_container_width=True, hide_index=True)

        slowest = ot.get("slowest_section")
        if slowest:
            st.caption(f"Slowest section: **{slowest}** ({(ot.get('slowest_section_ms') or 0)/1000:.1f}s)")
    else:
        st.info("Section timing data not available for this run.")

    st.divider()

    # ── Failures table ────────────────────────────────────────────────
    st.subheader("Issues Detected")
    failures = perf.get("failures", [])

    if failures:
        SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

        df_fail = pd.DataFrame([
            {
                "Severity":  SEVERITY_ICON.get(f["severity"], "") + " " + f["severity"].upper(),
                "Section":   f["section_key"][:45],
                "Type":      f["failure_type"].replace("_", " ").title(),
                "Details":   f["plain_english"] if not is_tech else f["technical"],
                "Metric":    f"{f['metric_value']:.3f}" if f.get("metric_value") is not None else "—",
                "_sev_ord":  SEVERITY_ORDER.get(f["severity"], 9),
            }
            for f in failures
        ])
        df_fail = df_fail.sort_values("_sev_ord").drop(columns=["_sev_ord"])

        st.dataframe(df_fail, use_container_width=True, hide_index=True,
                     column_config={
                         "Details": st.column_config.TextColumn("Details", width="large"),
                     })

        n_crit = sum(1 for f in failures if f["severity"] == "critical")
        n_warn = sum(1 for f in failures if f["severity"] == "warning")
        n_info = sum(1 for f in failures if f["severity"] == "info")
        st.caption(f"🔴 {n_crit} critical &nbsp; 🟡 {n_warn} warnings &nbsp; 🔵 {n_info} informational")
    else:
        st.success("No issues detected. All sections generated and evaluated successfully.")

    st.divider()

    # ── Improvement recommendations ───────────────────────────────────
    st.subheader("Improvement Recommendations")
    improvements = perf.get("improvements", [])

    if improvements:
        PRIORITY_ICON = {"high": "🔥", "medium": "📌", "low": "💡"}
        PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
        improvements_sorted = sorted(improvements, key=lambda x: PRIORITY_ORDER.get(x.get("priority", "low"), 9))

        for imp in improvements_sorted:
            priority = imp.get("priority", "low")
            area = imp.get("area", "General")
            icon = PRIORITY_ICON.get(priority, "💡")
            desc = imp["technical"] if is_tech else imp["plain_english"]
            affected = imp.get("affected_sections", [])

            with st.expander(f"{icon} **{area}** — {priority.upper()} priority", expanded=(priority == "high")):
                st.markdown(desc)
                if affected:
                    st.caption(
                        "Affected sections: "
                        + ", ".join(f"`{s[:40]}`" for s in affected[:6])
                        + (" …" if len(affected) > 6 else "")
                    )
    else:
        st.success("No improvements identified — the pipeline is running optimally.")

    _render_performance_legend()


def _render_performance_legend() -> None:
    with st.expander("How to read this tab"):
        st.markdown(
            "**Total Time** — how long the entire document generation + evaluation took.\n\n"
            "**LLM Generation** — time the AI spent writing each section. "
            "This is usually the largest chunk.\n\n"
            "**Retrieval** — time spent searching your uploaded documents for relevant content.\n\n"
            "**Evaluation** — time spent running quality checks (DeepEval + Opik metrics).\n\n"
            "**🔴 Critical issues** — sections that could not be generated at all. Must be fixed.\n\n"
            "**🟡 Warnings** — sections with quality concerns (possible inaccuracies, low scores). "
            "Should be reviewed before submission.\n\n"
            "**🔵 Informational** — minor observations (e.g. slightly informal language). "
            "Optional to address.\n\n"
            "**🔥 High priority improvements** — actions that will most improve document quality.\n\n"
            "**📌 Medium priority** — worthwhile improvements for the next run.\n\n"
            "**💡 Low priority** — optimisations for speed or efficiency."
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═══════════════════════════════════════════════════════════════════════════

def render_eval_dashboard() -> None:
    """Render the complete 6-tab evaluation dashboard."""
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

    t1, t2, t3 = st.tabs([
        "Run Overview", "Section Heatmap", "⚡ Performance",
    ])

    with t1:
        _render_tab_overview(runs)
    with t2:
        _render_tab_heatmap(runs)
    with t3:
        _render_tab_performance(runs)


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
