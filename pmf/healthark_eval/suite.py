"""
EvalSuite — Primary Public API for healthark_eval
===================================================

Central orchestrator that runs lexical, semantic, LLM-judge, and RAG
metrics, and returns structured ``EvalResult`` / ``DocumentEvalResult``
objects.  Any future project that needs evaluation imports this.

Usage:
    from healthark_eval import EvalSuite

    suite = EvalSuite(task="pmf")
    result = suite.run(
        generated="The site manufactures connectors.",
        reference="The site produces sterile connectors.",
        section_key="DEVICE DESCRIPTION",
    )
    print(result.grade, result.composite_score)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """Result of evaluating a single section.

    Attributes:
        section_key:       Section identifier.
        run_id:            Unique run identifier.
        timestamp:         ISO 8601 evaluation timestamp.
        rule_score:        Rule-based score (0-100) or None.
        lexical_scores:    BLEU / ROUGE dict or None.
        semantic_scores:   BERTScore dict or None.
        judge_scores:      PMFJudge result dict or None.
        rag_scores:        RAGEvaluator result dict or None.
        composite_score:   Weighted combination of all available scores (0-100).
        grade:             Letter grade: A (>=90), B (>=75), C (>=60),
                           D (>=45), F (<45).
        passed_threshold:  True if composite >= task threshold.
        summary:           One-sentence natural-language summary.
    """

    section_key: str = ""
    run_id: str = ""
    timestamp: str = ""
    rule_score: Optional[float] = None
    lexical_scores: Optional[Dict[str, Any]] = None
    semantic_scores: Optional[Dict[str, Any]] = None
    judge_scores: Optional[Dict[str, Any]] = None
    rag_scores: Optional[Dict[str, Any]] = None
    composite_score: float = 0.0
    grade: str = "F"
    passed_threshold: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentEvalResult:
    """Aggregate evaluation result for a full document.

    Attributes:
        run_id:               Unique document-level run ID.
        timestamp:            ISO 8601 timestamp.
        section_results:      Per-section ``EvalResult`` list.
        mean_composite:       Mean composite score across sections.
        mean_rule_score:      Mean rule score across sections.
        grade_distribution:   Count of each grade (A/B/C/D/F).
        lowest_sections:      Up to 3 lowest-scoring section keys.
        overall_grade:        Grade derived from mean_composite.
        passed_threshold:     True if mean_composite >= task threshold.
    """

    run_id: str = ""
    timestamp: str = ""
    section_results: List[EvalResult] = field(default_factory=list)
    mean_composite: float = 0.0
    mean_rule_score: Optional[float] = None
    grade_distribution: Dict[str, int] = field(default_factory=dict)
    lowest_sections: List[str] = field(default_factory=list)
    overall_grade: str = "F"
    passed_threshold: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["section_results"] = [r.to_dict() for r in self.section_results]
        return d


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

_COMPOSITE_WEIGHTS = {
    "rule": 0.15,
    "bertscore_f1": 0.20,
    "judge_normalized": 0.40,
    "ragas_score": 0.25,
}


def _compute_composite(scores: Dict[str, Optional[float]]) -> float:
    """Weighted mean of available scores, renormalising when some are null.

    Args:
        scores: Dict mapping weight-key to score value (0-100 scale) or None.

    Returns:
        Composite score in [0, 100].
    """
    active: List[tuple] = []
    for key, weight in _COMPOSITE_WEIGHTS.items():
        val = scores.get(key)
        if val is not None:
            active.append((weight, val))
    if not active:
        return 0.0
    total_w = sum(w for w, _ in active)
    return round(sum(w / total_w * v for w, v in active), 2)


def _grade(score: float) -> str:
    """Assign a letter grade based on composite score."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def _build_summary(
    section_key: str,
    grade: str,
    composite: float,
    judge_scores: Optional[Dict[str, Any]],
) -> str:
    """Generate a one-sentence natural-language summary."""
    detail = ""
    if judge_scores and not judge_scores.get("judge_error"):
        strengths = judge_scores.get("strengths", [])
        weaknesses = judge_scores.get("weaknesses", [])
        if strengths:
            detail = f" — strength: {strengths[0]}"
        elif weaknesses:
            detail = f" — weakness: {weaknesses[0]}"
    return (
        f"Section '{section_key}' achieved grade {grade} "
        f"({composite:.1f}/100){detail}."
    )


# ═══════════════════════════════════════════════════════════════════════════
# EvalSuite
# ═══════════════════════════════════════════════════════════════════════════


class EvalSuite:
    """Primary public API for the healthark_eval package.

    Orchestrates all evaluation metric modules and returns structured
    ``EvalResult`` objects.

    Args:
        task:          Task name — ``"pmf"`` loads PMF thresholds/config.
        llm_provider:  ``"anthropic"`` or ``"azure_openai"``.
        llm_model:     Model name for judge and RAG LLM calls.
        api_key:       API key (falls back to env vars).
        run_lexical:   Enable BLEU + ROUGE.
        run_semantic:  Enable BERTScore.
        run_judge:     Enable LLM-as-Judge.
        run_rag:       Enable RAGAS-style metrics.
        output_dir:    Directory for saved result files.
        verbose:       Enable verbose logging.

    Example:
        >>> suite = EvalSuite(task="pmf", run_judge=False)
        >>> result = suite.run(generated="...", reference="...")
        >>> result.grade
        'B'
    """

    def __init__(
        self,
        task: str = "pmf",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        run_lexical: bool = True,
        run_semantic: bool = True,
        run_judge: bool = True,
        run_rag: bool = True,
        output_dir: str = "eval_results",
        verbose: bool = False,
    ):
        from healthark_eval.config import get_task_config, TaskConfig

        self.task_name = task
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key
        self.run_lexical = run_lexical
        self.run_semantic = run_semantic
        self.run_judge = run_judge
        self.run_rag = run_rag
        self.output_dir = output_dir
        self.verbose = verbose

        # Load task config
        try:
            self.task_config: TaskConfig = get_task_config(task)
        except KeyError:
            self.task_config = TaskConfig(task_name=task)

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        # Lazy-initialized components
        self._lexical = None
        self._semantic = None
        self._judge = None
        self._rag = None

    # ── lazy loaders ─────────────────────────────────────────────────────

    def _get_lexical(self) -> Any:
        """Lazy-load LexicalMetrics class."""
        if self._lexical is None:
            from src.eval.eval_metrics import LexicalMetrics
            self._lexical = LexicalMetrics
        return self._lexical

    def _get_semantic(self) -> Any:
        """Lazy-load SemanticMetrics instance (caches model)."""
        if self._semantic is None:
            from src.eval.eval_metrics import SemanticMetrics
            self._semantic = SemanticMetrics(
                model_type=self.task_config.bertscore_model
            )
        return self._semantic

    def _get_judge(self) -> Any:
        """Lazy-load PMFJudge instance."""
        if self._judge is None:
            from src.eval.eval_judge import PMFJudge
            self._judge = PMFJudge(
                provider=self.llm_provider,
                model=self.llm_model,
                api_key=self.api_key,
            )
        return self._judge

    def _get_rag(self) -> Any:
        """Lazy-load RAGEvaluator instance."""
        if self._rag is None:
            from src.eval.eval_rag import RAGEvaluator
            judge = self._get_judge()
            self._rag = RAGEvaluator(
                llm_client=judge._client,
                model=self.llm_model,
            )
        return self._rag

    # ══════════════════════════════════════════════════════════════════════
    # run() — single section
    # ══════════════════════════════════════════════════════════════════════

    def run(
        self,
        generated: str,
        retrieved: Optional[List[str]] = None,
        reference: str = "",
        section_key: str = "",
        section_instruction: str = "",
        site_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Evaluate a single generated section.

        Args:
            generated:            Generated output text.
            retrieved:            Retrieved context chunks.
            reference:            Ground-truth reference text.
            section_key:          Section identifier.
            section_instruction:  Original generation instruction.
            site_name:            Manufacturing site name.
            metadata:             Free-form metadata dict.

        Returns:
            ``EvalResult`` with all computed scores, composite, and grade.

        Raises:
            Nothing — individual metric failures are captured and logged.

        Example:
            >>> r = suite.run("The site...", reference="The site...",
            ...               section_key="EXEC SUMMARY")
            >>> r.composite_score
            78.5
        """
        retrieved = retrieved or []
        run_id = uuid.uuid4().hex[:12]
        ts = datetime.now(timezone.utc).isoformat()

        # --- rule score ---
        rule_score: Optional[float] = None
        try:
            from src.eval.eval_utils import score_section
            from src.eval.eval_config import get_eval_rules
            rules = get_eval_rules()
            rule_result = score_section(
                section_key=section_key,
                section_text=generated,
                rules=rules,
                context={"site_name": site_name},
            )
            rule_score = rule_result.get("score")
        except Exception as exc:
            logger.warning("Rule scoring failed: %s", exc)

        # --- lexical ---
        lexical_scores: Optional[Dict[str, Any]] = None
        if self.run_lexical and reference:
            try:
                lex_cls = self._get_lexical()
                lexical_scores = lex_cls.compute_all_lexical(generated, reference)
            except Exception as exc:
                logger.warning("Lexical metrics failed: %s", exc)

        # --- semantic (BERTScore) ---
        semantic_scores: Optional[Dict[str, Any]] = None
        if self.run_semantic and reference:
            try:
                sm = self._get_semantic()
                semantic_scores = sm.compute_bertscore([generated], [reference])
            except Exception as exc:
                logger.warning("BERTScore failed: %s", exc)

        # --- judge ---
        judge_scores: Optional[Dict[str, Any]] = None
        if self.run_judge:
            try:
                judge = self._get_judge()
                judge_scores = judge.score_section(
                    section_key=section_key,
                    section_instruction=section_instruction,
                    retrieved_context="\n\n".join(retrieved),
                    generated_output=generated,
                    site_name=site_name,
                    reference_output=reference,
                )
            except Exception as exc:
                logger.warning("Judge scoring failed: %s", exc)

        # --- RAG ---
        rag_scores: Optional[Dict[str, Any]] = None
        if self.run_rag and retrieved:
            try:
                rag = self._get_rag()
                rag_scores = rag.evaluate_section(
                    section_key=section_key,
                    section_instruction=section_instruction or section_key,
                    retrieved_chunks=retrieved,
                    generated_answer=generated,
                    reference_answer=reference,
                )
            except Exception as exc:
                logger.warning("RAG metrics failed: %s", exc)

        # --- composite ---
        composite_inputs: Dict[str, Optional[float]] = {
            "rule": rule_score,
            "bertscore_f1": None,
            "judge_normalized": None,
            "ragas_score": None,
        }

        if semantic_scores:
            f1 = semantic_scores.get("bertscore_f1_mean")
            if f1 is not None:
                composite_inputs["bertscore_f1"] = f1 * 100.0

        if judge_scores and not judge_scores.get("judge_error"):
            ns = judge_scores.get("normalized_score")
            if ns is not None:
                composite_inputs["judge_normalized"] = float(ns)

        if rag_scores:
            rs = rag_scores.get("ragas_score")
            if rs is not None:
                composite_inputs["ragas_score"] = float(rs) * 100.0

        composite = _compute_composite(composite_inputs)
        g = _grade(composite)
        passed = composite >= self.task_config.composite_threshold
        summary = _build_summary(section_key, g, composite, judge_scores)

        return EvalResult(
            section_key=section_key,
            run_id=run_id,
            timestamp=ts,
            rule_score=rule_score,
            lexical_scores=lexical_scores,
            semantic_scores=semantic_scores,
            judge_scores=judge_scores,
            rag_scores=rag_scores,
            composite_score=composite,
            grade=g,
            passed_threshold=passed,
            summary=summary,
        )

    # ══════════════════════════════════════════════════════════════════════
    # run_document()
    # ══════════════════════════════════════════════════════════════════════

    def run_document(
        self,
        sections: List[Dict[str, Any]],
        parallel: bool = False,
    ) -> DocumentEvalResult:
        """Evaluate all sections in a document.

        Args:
            sections: List of dicts with at least ``section_key`` and
                      ``generated_text``.  May also include ``retrieved``,
                      ``reference``, ``section_instruction``, ``site_name``.
            parallel: Reserved for future parallel execution.

        Returns:
            ``DocumentEvalResult`` with per-section results and aggregates.

        Example:
            >>> doc = suite.run_document(run_artifact["sections"])
            >>> doc.overall_grade
            'B'
        """
        doc_id = uuid.uuid4().hex[:12]
        ts = datetime.now(timezone.utc).isoformat()

        # --- Phase 1: run all sections with BERTScore disabled ---
        # We batch BERTScore in Phase 2 for performance (Rule 10).
        orig_semantic = self.run_semantic
        self.run_semantic = False
        results: List[EvalResult] = []
        gen_ref_pairs: List[tuple] = []  # (index, generated, reference)
        for idx, sec in enumerate(sections):
            gen = sec.get("generated_text", "")
            ref = sec.get("reference", sec.get("reference_output", ""))
            r = self.run(
                generated=gen,
                retrieved=sec.get("retrieved", sec.get("retrieved_chunks", [])),
                reference=ref,
                section_key=sec.get("section_key", ""),
                section_instruction=sec.get(
                    "section_instruction", sec.get("prompt_text", "")
                ),
                site_name=sec.get("site_name", ""),
            )
            results.append(r)
            if orig_semantic and gen.strip() and ref.strip():
                gen_ref_pairs.append((idx, gen, ref))
        self.run_semantic = orig_semantic

        # --- Phase 2: batched BERTScore across all sections at once ---
        if gen_ref_pairs and self.run_semantic:
            try:
                sm = self._get_semantic()
                hyps = [p[1] for p in gen_ref_pairs]
                refs = [p[2] for p in gen_ref_pairs]
                batch_result = sm.compute_bertscore(hyps, refs)
                per_ex = batch_result.get("bertscore_per_example", [])
                for i, (idx, _, _) in enumerate(gen_ref_pairs):
                    if i < len(per_ex):
                        results[idx].semantic_scores = {
                            "bertscore_precision_mean": per_ex[i].get("precision", 0.0),
                            "bertscore_recall_mean": per_ex[i].get("recall", 0.0),
                            "bertscore_f1_mean": per_ex[i].get("f1", 0.0),
                            "bertscore_per_example": [per_ex[i]],
                        }
                        # Recompute composite with BERTScore now available
                        f1 = per_ex[i].get("f1")
                        if f1 is not None:
                            ci: Dict[str, Optional[float]] = {
                                "rule": results[idx].rule_score,
                                "bertscore_f1": float(f1) * 100.0,
                                "judge_normalized": None,
                                "ragas_score": None,
                            }
                            js = results[idx].judge_scores
                            if js and not js.get("judge_error"):
                                ci["judge_normalized"] = js.get("normalized_score")
                            rs = results[idx].rag_scores
                            if rs:
                                ci["ragas_score"] = (
                                    float(rs["ragas_score"]) * 100.0
                                    if rs.get("ragas_score") is not None
                                    else None
                                )
                            new_comp = _compute_composite(ci)
                            results[idx].composite_score = new_comp
                            results[idx].grade = _grade(new_comp)
                            results[idx].passed_threshold = (
                                new_comp >= self.task_config.composite_threshold
                            )
            except Exception as exc:
                logger.warning("Batched BERTScore in run_document failed: %s", exc)

        # aggregate
        composites = [r.composite_score for r in results]
        mean_comp = (
            round(sum(composites) / len(composites), 2)
            if composites else 0.0
        )

        rules = [r.rule_score for r in results if r.rule_score is not None]
        mean_rule = round(sum(rules) / len(rules), 2) if rules else None

        grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for r in results:
            grades[r.grade] = grades.get(r.grade, 0) + 1

        sorted_results = sorted(results, key=lambda r: r.composite_score)
        lowest = [r.section_key for r in sorted_results[:3]]

        return DocumentEvalResult(
            run_id=doc_id,
            timestamp=ts,
            section_results=results,
            mean_composite=mean_comp,
            mean_rule_score=mean_rule,
            grade_distribution=grades,
            lowest_sections=lowest,
            overall_grade=_grade(mean_comp),
            passed_threshold=mean_comp >= self.task_config.composite_threshold,
        )

    # ══════════════════════════════════════════════════════════════════════
    # run_benchmark()
    # ══════════════════════════════════════════════════════════════════════

    def run_benchmark(
        self,
        benchmark_dir: str = "data/benchmark",
        model_override: Optional[str] = None,
    ) -> Any:
        """Run full evaluation on all benchmark cases.

        Args:
            benchmark_dir: Benchmark dataset directory.
            model_override: Optional model name to override the suite default.

        Returns:
            ``pandas.DataFrame`` with one row per benchmark case and
            columns for every metric.

        Example:
            >>> df = suite.run_benchmark()
            >>> df["composite_score"].mean()
            72.5
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required — pip install pandas")

        from src.eval.benchmark_loader import BenchmarkLoader

        loader = BenchmarkLoader(benchmark_dir)
        cases = loader.load_cases()

        if model_override:
            self.llm_model = model_override

        rows: List[Dict[str, Any]] = []
        for case in cases:
            r = self.run(
                generated=case.get("reference_output", ""),
                retrieved=[case.get("retrieved_context", "")],
                reference=case.get("reference_output", ""),
                section_key=case.get("section_key", ""),
                section_instruction=case.get("section_instruction", ""),
                site_name=case.get("site_name", ""),
            )
            rows.append({
                "case_id": case.get("case_id"),
                "section_key": case.get("section_key"),
                "difficulty": case.get("difficulty"),
                "section_type": case.get("section_type"),
                "rule_score": r.rule_score,
                "bleu": (r.lexical_scores or {}).get("bleu"),
                "rouge_l": (r.lexical_scores or {}).get("rougeL_fmeasure"),
                "bertscore_f1": (r.semantic_scores or {}).get("bertscore_f1_mean"),
                "judge_normalized": (r.judge_scores or {}).get("normalized_score"),
                "faithfulness": (r.rag_scores or {}).get("faithfulness"),
                "ragas_score": (r.rag_scores or {}).get("ragas_score"),
                "composite_score": r.composite_score,
                "grade": r.grade,
                "passed": r.passed_threshold,
            })

        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════════════════
    # compare_models()
    # ══════════════════════════════════════════════════════════════════════

    def compare_models(
        self,
        sections: List[Dict[str, Any]],
        model_configs: List[Dict[str, str]],
    ) -> Any:
        """Evaluate the same sections across multiple model configurations.

        Args:
            sections:      List of section dicts with ``section_key``,
                           ``generated_outputs`` (dict: model_name → text),
                           and other standard fields.
            model_configs:  List of dicts with ``name`` and ``provider``.

        Returns:
            ``pandas.DataFrame`` with columns: ``section_key``,
            ``model_name``, ``composite_score``, ``grade``, plus
            individual metric columns.

        Example:
            >>> df = suite.compare_models(secs, configs)
            >>> df.groupby("model_name")["composite_score"].mean()
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required — pip install pandas")

        rows: List[Dict[str, Any]] = []
        for sec in sections:
            gen_outputs = sec.get("generated_outputs", {})
            for cfg in model_configs:
                model_name = cfg["name"]
                gen_text = gen_outputs.get(model_name, "")
                if not gen_text:
                    continue

                r = self.run(
                    generated=gen_text,
                    retrieved=sec.get("retrieved", []),
                    reference=sec.get("reference", ""),
                    section_key=sec.get("section_key", ""),
                    section_instruction=sec.get("section_instruction", ""),
                    site_name=sec.get("site_name", ""),
                )
                rows.append({
                    "section_key": sec.get("section_key"),
                    "model_name": model_name,
                    "composite_score": r.composite_score,
                    "grade": r.grade,
                    "rule_score": r.rule_score,
                    "bleu": (r.lexical_scores or {}).get("bleu"),
                    "bertscore_f1": (r.semantic_scores or {}).get(
                        "bertscore_f1_mean"
                    ),
                    "judge_normalized": (r.judge_scores or {}).get(
                        "normalized_score"
                    ),
                    "ragas_score": (r.rag_scores or {}).get("ragas_score"),
                })

        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════════════════
    # save_results()
    # ══════════════════════════════════════════════════════════════════════

    def save_results(
        self,
        result: EvalResult,
        output_path: Optional[str] = None,
    ) -> str:
        """Save an EvalResult to a JSON file.

        Args:
            result:      The evaluation result to persist.
            output_path: Explicit file path.  If None, auto-generates a
                         path in ``self.output_dir``.

        Returns:
            The file path where the result was saved.

        Example:
            >>> path = suite.save_results(result)
        """
        if output_path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            fname = f"eval_{result.run_id}_{result.section_key or 'unknown'}.json"
            # Sanitise filename
            fname = fname.replace(" ", "_").replace("/", "_")
            output_path = os.path.join(self.output_dir, fname)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)

        logger.info("Saved eval result to %s", output_path)
        return output_path
