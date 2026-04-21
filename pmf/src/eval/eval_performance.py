"""
Performance Analyzer for PMF Document Generation
=================================================

Analyzes latency, failures, and improvement opportunities from a completed
PMF generation run. Designed to be readable by both technical and
non-technical stakeholders — every finding has a plain-English and a
technical description.

No LLM calls. Pure analytical post-processing of run_artifacts.

Outputs:
  PerformanceReport
    .section_timings  — per-section latency breakdown (retrieval / generation / eval)
    .overall_timing   — pipeline totals and slowest-section pointer
    .failures         — issues detected (empty sections, hallucination, low scores, etc.)
    .improvements     — prioritised recommendations (technical + plain English)
    .summary_technical — one-paragraph technical summary
    .summary_plain     — one-paragraph plain-English summary

Part of the Healthark GenAI Evaluation Framework (Initiative 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Thresholds ────────────────────────────────────────────────────────────────
SLOW_SECTION_S = 20.0        # seconds: flag as "slow"
VERY_SLOW_SECTION_S = 60.0   # seconds: flag as "very slow"
LOW_RULE_SCORE = 50.0        # /100
LOW_FAITHFULNESS = 0.25      # 0–1
HIGH_HALLUCINATION = 0.50    # 0–1 (Opik hallucination score)
LOW_REG_TONE = 0.50          # 0–1
LOW_ANSWER_RELEVANCE = 0.40  # 0–1
EVAL_OVERHEAD_RATIO = 0.40   # warn if eval > 40% of total pipeline time


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SectionTiming:
    section_key: str
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    eval_ms: Optional[float] = None
    total_ms: Optional[float] = None
    is_static: bool = False

    @property
    def total_s(self) -> Optional[float]:
        return self.total_ms / 1000.0 if self.total_ms is not None else None

    @property
    def generation_s(self) -> Optional[float]:
        return self.generation_ms / 1000.0 if self.generation_ms is not None else None


@dataclass
class SectionFailure:
    section_key: str
    failure_type: str   # "error" | "missing_chunks" | "low_score" | "hallucination" | "low_tone" | "low_relevance"
    severity: str       # "critical" | "warning" | "info"
    technical: str
    plain_english: str
    metric_value: Optional[float] = None


@dataclass
class Improvement:
    area: str           # "Source Documents" | "Factual Accuracy" | etc.
    priority: str       # "high" | "medium" | "low"
    technical: str
    plain_english: str
    affected_sections: List[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    section_timings: List[SectionTiming]
    overall_timing: Dict[str, Any]
    failures: List[SectionFailure]
    improvements: List[Improvement]
    summary_technical: str
    summary_plain: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_timings": [
                {
                    "section_key": t.section_key,
                    "retrieval_ms": t.retrieval_ms,
                    "generation_ms": t.generation_ms,
                    "eval_ms": t.eval_ms,
                    "total_ms": t.total_ms,
                    "is_static": t.is_static,
                }
                for t in self.section_timings
            ],
            "overall_timing": self.overall_timing,
            "failures": [
                {
                    "section_key": f.section_key,
                    "failure_type": f.failure_type,
                    "severity": f.severity,
                    "technical": f.technical,
                    "plain_english": f.plain_english,
                    "metric_value": f.metric_value,
                }
                for f in self.failures
            ],
            "improvements": [
                {
                    "area": i.area,
                    "priority": i.priority,
                    "technical": i.technical,
                    "plain_english": i.plain_english,
                    "affected_sections": i.affected_sections,
                }
                for i in self.improvements
            ],
            "summary_technical": self.summary_technical,
            "summary_plain": self.summary_plain,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceAnalyzer:
    """Analyzes run_artifacts to produce a PerformanceReport.

    Usage:
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(run_artifacts, evaluation)
    """

    def __init__(
        self,
        slow_threshold_s: float = SLOW_SECTION_S,
        very_slow_threshold_s: float = VERY_SLOW_SECTION_S,
    ):
        self.slow_s = slow_threshold_s
        self.very_slow_s = very_slow_threshold_s

    # ─────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        run_artifacts: Dict[str, Any],
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> PerformanceReport:
        sections = run_artifacts.get("sections", [])
        timing_meta = run_artifacts.get("timing", {})

        # Build flat lookup: section_key → rule-eval row
        eval_section_map: Dict[str, Any] = {}
        if evaluation:
            for row in evaluation.get("document_scores", {}).get("sections", []):
                eval_section_map[row.get("section_key", "")] = row

        timings = self._build_section_timings(sections)
        overall = self._build_overall_timing(timings, timing_meta)
        failures = self._detect_failures(sections, eval_section_map)
        improvements = self._generate_improvements(timings, failures, overall)
        tech_summary, plain_summary = self._build_summaries(timings, overall, failures, improvements)

        return PerformanceReport(
            section_timings=timings,
            overall_timing=overall,
            failures=failures,
            improvements=improvements,
            summary_technical=tech_summary,
            summary_plain=plain_summary,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_section_timings(self, sections: List[Dict]) -> List[SectionTiming]:
        timings: List[SectionTiming] = []
        for s in sections:
            t = s.get("timing") or {}
            total = t.get("total_ms")
            if total is None:
                parts = [t.get("retrieval_ms"), t.get("generation_ms"), t.get("eval_ms")]
                valid = [p for p in parts if p is not None]
                total = sum(valid) if valid else None
            timings.append(SectionTiming(
                section_key=s.get("section_key", "unknown"),
                retrieval_ms=t.get("retrieval_ms"),
                generation_ms=t.get("generation_ms"),
                eval_ms=t.get("eval_ms"),
                total_ms=total,
                is_static=s.get("is_static", False),
            ))
        return sorted(timings, key=lambda x: (x.total_ms or 0), reverse=True)

    def _build_overall_timing(
        self,
        timings: List[SectionTiming],
        timing_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        gen_ms = sum(t.generation_ms or 0 for t in timings)
        ret_ms = sum(t.retrieval_ms or 0 for t in timings)
        eval_ms = sum(t.eval_ms or 0 for t in timings)
        total_ms = timing_meta.get("total_pipeline_ms") or sum(t.total_ms or 0 for t in timings)
        slowest = timings[0] if timings else None
        n = len(timings)
        return {
            "total_pipeline_ms": total_ms,
            "total_pipeline_s": round(total_ms / 1000, 1) if total_ms else None,
            "total_generation_ms": gen_ms,
            "total_retrieval_ms": ret_ms,
            "total_eval_ms": eval_ms,
            "section_count": n,
            "slowest_section": slowest.section_key if slowest else None,
            "slowest_section_ms": slowest.total_ms if slowest else None,
            "avg_section_ms": round(total_ms / n, 1) if (total_ms and n) else None,
            "pct_generation": round(gen_ms / total_ms * 100, 1) if (total_ms and gen_ms) else None,
            "pct_retrieval": round(ret_ms / total_ms * 100, 1) if (total_ms and ret_ms) else None,
            "pct_eval": round(eval_ms / total_ms * 100, 1) if (total_ms and eval_ms) else None,
        }

    def _detect_failures(
        self,
        sections: List[Dict],
        eval_map: Dict[str, Any],
    ) -> List[SectionFailure]:
        failures: List[SectionFailure] = []

        for s in sections:
            sk = s.get("section_key", "")
            is_static = s.get("is_static", False)
            gen_text = s.get("generated_text", "")

            # ── Generation failed entirely ────────────────────────────
            if not gen_text.strip():
                failures.append(SectionFailure(
                    section_key=sk,
                    failure_type="error",
                    severity="critical",
                    technical=(
                        "generated_text is empty — the LLM call returned no output "
                        "or threw an exception during handle_user_message()."
                    ),
                    plain_english=(
                        "This section could not be generated. The AI produced no content — "
                        "possibly a timeout, network error, or missing input data."
                    ),
                ))
                continue  # no point checking metrics on an empty section

            # ── No retrieval chunks (dynamic sections only) ───────────
            if not is_static and not s.get("retrieved_paths"):
                failures.append(SectionFailure(
                    section_key=sk,
                    failure_type="missing_chunks",
                    severity="warning",
                    technical=(
                        "retrieved_paths is empty — no source documents were found "
                        "in the vector DB for this section's retrieval query. "
                        "Generation proceeded using only excel data."
                    ),
                    plain_english=(
                        "No relevant documents were found for this section. "
                        "The AI wrote it without referencing your uploaded source files, "
                        "which increases the risk of inaccurate or generic content."
                    ),
                ))

            # ── Low rule score ────────────────────────────────────────
            es = eval_map.get(sk, {})
            rule_score = es.get("score")
            if rule_score is not None and float(rule_score) < LOW_RULE_SCORE:
                missing_kw = es.get("missing_keywords") or []
                failures.append(SectionFailure(
                    section_key=sk,
                    failure_type="low_score",
                    severity="warning",
                    technical=(
                        f"Rule score {float(rule_score):.1f}/100 — below threshold {LOW_RULE_SCORE}. "
                        f"Missing keywords: {missing_kw}. "
                        f"Checks: {es.get('checks', {})}."
                    ),
                    plain_english=(
                        f"This section scored {float(rule_score):.0f}/100 on quality checks. "
                        "It may be too short, missing required regulatory keywords, "
                        "or not mentioning the site name."
                    ),
                    metric_value=float(rule_score),
                ))

            # ── DeepEval: low faithfulness ────────────────────────────
            deep = s.get("extended_eval") or {}
            faith = deep.get("faithfulness")
            if faith is not None and float(faith) < LOW_FAITHFULNESS:
                failures.append(SectionFailure(
                    section_key=sk,
                    failure_type="hallucination",
                    severity="warning",
                    technical=(
                        f"DeepEval faithfulness {float(faith):.3f} — "
                        f"only {float(faith)*100:.0f}% of generated claims are "
                        "grounded in retrieved source documents."
                    ),
                    plain_english=(
                        f"About {(1-float(faith))*100:.0f}% of the content in this section "
                        "may not be directly supported by your uploaded documents. "
                        "A human reviewer should verify the factual claims before submission."
                    ),
                    metric_value=float(faith),
                ))

            # ── Opik: high hallucination score ────────────────────────
            opik = s.get("opik_eval") or {}
            hall = opik.get("hallucination_score")
            if hall is not None and float(hall) > HIGH_HALLUCINATION:
                failures.append(SectionFailure(
                    section_key=sk,
                    failure_type="hallucination",
                    severity="warning",
                    technical=(
                        f"Opik hallucination score {float(hall):.3f} > threshold {HIGH_HALLUCINATION} — "
                        "significant portion of factual claims not grounded in context."
                    ),
                    plain_english=(
                        "The AI appears to have introduced details in this section that "
                        "are not found in your source documents. Please review carefully."
                    ),
                    metric_value=float(hall),
                ))

            # ── Opik: low regulatory tone ─────────────────────────────
            tone = opik.get("regulatory_tone_score")
            if tone is not None and float(tone) < LOW_REG_TONE:
                failures.append(SectionFailure(
                    section_key=sk,
                    failure_type="low_tone",
                    severity="info",
                    technical=(
                        f"Regulatory tone score {float(tone):.3f} < {LOW_REG_TONE} — "
                        "language may not meet EU GMP Annex 4 / ICH Q10 formality standards."
                    ),
                    plain_english=(
                        "The language in this section may not sound formal enough for a "
                        "regulatory submission. It may need editing before filing."
                    ),
                    metric_value=float(tone),
                ))

            # ── Opik: low answer relevance ────────────────────────────
            relev = opik.get("answer_relevance_score")
            if relev is not None and float(relev) < LOW_ANSWER_RELEVANCE:
                failures.append(SectionFailure(
                    section_key=sk,
                    failure_type="low_relevance",
                    severity="info",
                    technical=(
                        f"Opik answer relevance {float(relev):.3f} < {LOW_ANSWER_RELEVANCE} — "
                        "generated output does not closely address the section instruction."
                    ),
                    plain_english=(
                        "This section does not fully address what was asked of it. "
                        "The content may be off-topic or too generic."
                    ),
                    metric_value=float(relev),
                ))

        return failures

    def _generate_improvements(
        self,
        timings: List[SectionTiming],
        failures: List[SectionFailure],
        overall: Dict[str, Any],
    ) -> List[Improvement]:
        improvements: List[Improvement] = []
        total_ms = overall.get("total_pipeline_ms") or 0
        eval_ms = overall.get("total_eval_ms") or 0

        # Group failures by type
        by_type: Dict[str, List[str]] = {}
        for f in failures:
            by_type.setdefault(f.failure_type, []).append(f.section_key)

        # ── Missing source documents ──────────────────────────────────
        if "missing_chunks" in by_type:
            sks = by_type["missing_chunks"]
            improvements.append(Improvement(
                area="Source Documents",
                priority="high",
                technical=(
                    f"{len(sks)} section(s) had empty retrieved_paths: {sks}. "
                    "Action: (1) expand the ZIP upload with domain-specific SOPs, batch records, "
                    "or quality manuals; (2) lower the similarity threshold in "
                    "DocumentRetriever; (3) review retrieval_query templates in the PMF template "
                    "to ensure queries are specific enough to match uploaded documents."
                ),
                plain_english=(
                    "Some sections had no documents to reference. "
                    "To fix this, include more relevant documents in your ZIP file — "
                    "especially site-specific SOPs, equipment qualification records, "
                    "and quality manuals that relate to those sections."
                ),
                affected_sections=sks,
            ))

        # ── Hallucination / low faithfulness ─────────────────────────
        if "hallucination" in by_type:
            sks = list(dict.fromkeys(by_type["hallucination"]))  # deduplicate
            improvements.append(Improvement(
                area="Factual Accuracy",
                priority="high",
                technical=(
                    f"{len(sks)} section(s) flagged for hallucination: {sks}. "
                    "Recommended actions: "
                    "(1) Add explicit grounding in prompts: 'Use ONLY information from the provided context. Do not invent details.' "
                    "(2) Increase top_k retrieval (currently 5) to provide more context. "
                    "(3) Consider a post-generation fact-check pass using DeepEval Faithfulness. "
                    "(4) Add site-specific factual documents (certificates, equipment lists) to the corpus."
                ),
                plain_english=(
                    "The AI may have invented some details in these sections. "
                    "The best fix is to upload more specific documents — "
                    "especially certificates, equipment lists, and site-specific quality records — "
                    "so the AI has accurate facts to draw from instead of guessing. "
                    "A regulatory expert should review these sections before submission."
                ),
                affected_sections=sks,
            ))

        # ── Low rule scores / completeness ────────────────────────────
        if "low_score" in by_type:
            sks = by_type["low_score"]
            improvements.append(Improvement(
                area="Section Completeness",
                priority="medium",
                technical=(
                    f"{len(sks)} section(s) failed rule-based checks: {sks}. "
                    "Review eval_config.py — check min_chars thresholds and required keyword lists. "
                    "Consider enriching the PMF template prompts for these sections to explicitly "
                    "request the missing information (regulatory keywords, site-specific details)."
                ),
                plain_english=(
                    "Some sections are too short or missing key regulatory words. "
                    "The PMF template may need to be updated to give the AI more specific "
                    "instructions about what to include in these sections."
                ),
                affected_sections=sks,
            ))

        # ── Low regulatory tone ───────────────────────────────────────
        if "low_tone" in by_type:
            sks = by_type["low_tone"]
            improvements.append(Improvement(
                area="Regulatory Language",
                priority="medium",
                technical=(
                    f"{len(sks)} section(s) have low regulatory tone scores: {sks}. "
                    "Add to the system prompt: 'Write in formal passive voice using ICH Q10 and EU GMP Annex 4 terminology. "
                    "Avoid informal phrasing.' Consider a post-processing linguistic normalisation step."
                ),
                plain_english=(
                    "Some sections use language that is too informal for a regulatory submission. "
                    "These sections should be reviewed and revised by a regulatory affairs professional "
                    "before the document is filed."
                ),
                affected_sections=sks,
            ))

        # ── Low answer relevance ──────────────────────────────────────
        if "low_relevance" in by_type:
            sks = by_type["low_relevance"]
            improvements.append(Improvement(
                area="Prompt Design",
                priority="medium",
                technical=(
                    f"{len(sks)} section(s) have low answer relevance: {sks}. "
                    "The section instruction in the PMF template may be ambiguous or too broad. "
                    "Refine the prompt template to include explicit required sub-topics "
                    "and output structure constraints."
                ),
                plain_english=(
                    "Some sections don't fully address what the template asks for. "
                    "The section instructions in the PMF template may need to be "
                    "made more specific and detailed."
                ),
                affected_sections=sks,
            ))

        # ── Slow sections ─────────────────────────────────────────────
        slow = [t for t in timings if t.total_s and t.total_s > self.slow_s]
        if slow:
            slow_names = [t.section_key for t in slow]
            avg_gen_s = (
                sum(t.generation_ms or 0 for t in slow) / len(slow) / 1000
            )
            improvements.append(Improvement(
                area="Performance / Speed",
                priority="low",
                technical=(
                    f"{len(slow)} section(s) exceeded {self.slow_s:.0f}s: {slow_names}. "
                    f"Average LLM generation time for slow sections: {avg_gen_s:.1f}s. "
                    "Options: "
                    "(1) Cache static section outputs — they never change between runs. "
                    "(2) Reduce max_tokens for sections where output is already long enough. "
                    "(3) Use GPT-4o-mini for low-complexity sections (lower cost + 2× faster). "
                    "(4) Run independent sections concurrently using asyncio or threading."
                ),
                plain_english=(
                    f"{len(slow)} section(s) took more than {self.slow_s:.0f} seconds each to generate. "
                    "This is normal for complex AI generation, but can be improved by "
                    "providing more focused, specific source documents so the AI "
                    "spends less time processing irrelevant content."
                ),
                affected_sections=slow_names,
            ))

        # ── Eval overhead ─────────────────────────────────────────────
        if total_ms > 0 and eval_ms > 0 and (eval_ms / total_ms) > EVAL_OVERHEAD_RATIO:
            improvements.append(Improvement(
                area="Evaluation Speed",
                priority="low",
                technical=(
                    f"Evaluation accounts for {eval_ms/total_ms*100:.0f}% of total runtime "
                    f"({eval_ms/1000:.1f}s / {total_ms/1000:.1f}s). "
                    "The DeepEval claim-extraction pipeline is the primary driver. "
                    "Ensure cache_enabled=True in both eval_rag.py and eval_opik_style.py. "
                    "Skip rag/opik eval on static sections (already gated by is_static check). "
                    "Consider reducing claim extraction calls by batching sections."
                ),
                plain_english=(
                    "The quality-checking phase is taking up a lot of total time. "
                    "This can be reduced automatically on subsequent runs — "
                    "sections that haven't changed are cached and won't be re-evaluated."
                ),
                affected_sections=[],
            ))

        return improvements

    def _build_summaries(
        self,
        timings: List[SectionTiming],
        overall: Dict[str, Any],
        failures: List[SectionFailure],
        improvements: List[Improvement],
    ) -> Tuple[str, str]:
        total_s = (overall.get("total_pipeline_ms") or 0) / 1000
        gen_s = (overall.get("total_generation_ms") or 0) / 1000
        eval_s = (overall.get("total_eval_ms") or 0) / 1000
        n_crit = sum(1 for f in failures if f.severity == "critical")
        n_warn = sum(1 for f in failures if f.severity == "warning")
        n_info = sum(1 for f in failures if f.severity == "info")
        slowest_name = overall.get("slowest_section", "—")
        slowest_ms = overall.get("slowest_section_ms") or 0
        n_high = sum(1 for i in improvements if i.priority == "high")

        technical = (
            f"Pipeline completed in {total_s:.1f}s total "
            f"(LLM generation: {gen_s:.1f}s, evaluation: {eval_s:.1f}s, "
            f"other: {max(0, total_s - gen_s - eval_s):.1f}s). "
            f"Slowest section: '{slowest_name}' ({slowest_ms/1000:.1f}s). "
            f"Issues detected: {n_crit} critical, {n_warn} warnings, {n_info} informational. "
            f"{len(improvements)} improvements identified ({n_high} high-priority)."
        )

        plain = (
            f"The document was generated in {_friendly_duration(total_s)}. "
            f"The slowest section was '{slowest_name}'. "
        )
        if n_crit:
            plain += f"⚠️ {n_crit} section(s) failed to generate and need immediate attention. "
        if n_warn:
            plain += f"{n_warn} section(s) have quality concerns (possible inaccuracies or incomplete content). "
        if not failures:
            plain += "All sections generated successfully with no quality issues detected. "
        if n_high:
            plain += f"{n_high} high-priority improvement(s) were identified for the next run."

        return technical, plain


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _friendly_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s} sec"
