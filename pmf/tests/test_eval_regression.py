"""
Regression Testing Suite for PMF Evaluation Framework
======================================================

Complete pytest suite for the Healthark GenAI Evaluation Framework.

Run:
    pytest tests/test_eval_regression.py -v
    pytest tests/test_eval_regression.py -v --junitxml=reports/eval_report.xml
    EVAL_RUN_SLOW=1 pytest tests/test_eval_regression.py -v   # include LLM tests

Architecture:
    - Thresholds defined in tests/eval_thresholds.yaml
    - Fixtures in tests/conftest.py (session-scoped where possible)
    - @pytest.mark.slow for tests that call LLM APIs (skipped unless
      EVAL_RUN_SLOW=1 env var is set)
    - JUnit XML report via --junitxml flag

Dependencies:
    pip install pytest pyyaml sacrebleu rouge-score bert-score sentence-transformers
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eval.eval_metrics import LexicalMetrics, SemanticMetrics, compute_all_metrics
from src.eval.eval_config import get_eval_rules
from src.eval.eval_utils import score_section
from src.eval.benchmark_loader import BenchmarkLoader, validate_case
from src.eval.eval_judge import (
    PMFJudge,
    JUDGE_RUBRIC,
    CRITERIA_NAMES,
    CRITERION_WEIGHTS,
    _parse_judge_response,
)
from src.eval.eval_rag import RAGEvaluator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark")
EVAL_RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "eval_runs")
BASELINE_PATH = os.path.join(BENCHMARK_DIR, "baseline_scores.json")


# ═══════════════════════════════════════════════════════════════════════════
# 1. BENCHMARK DATASET INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkDatasetIntegrity:
    """Pure data validation — no external calls."""

    def test_all_required_fields_present(
        self, benchmark_cases: List[Dict[str, Any]]
    ) -> None:
        """Every benchmark case must contain all required schema fields."""
        for case in benchmark_cases:
            errors = validate_case(case)
            assert len(errors) == 0, (
                f"Case {case.get('case_id', '?')} validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def test_all_case_ids_unique(
        self, benchmark_cases: List[Dict[str, Any]]
    ) -> None:
        """All case_id values must be unique across the dataset."""
        ids = [c.get("case_id", "") for c in benchmark_cases]
        seen = set()
        duplicates = []
        for cid in ids:
            if cid in seen:
                duplicates.append(cid)
            seen.add(cid)
        assert len(duplicates) == 0, f"Duplicate case IDs: {duplicates}"

    def test_all_section_types_valid(
        self, benchmark_cases: List[Dict[str, Any]]
    ) -> None:
        """Every section_type must be one of text, table, image, static."""
        valid = {"text", "table", "image", "static"}
        for case in benchmark_cases:
            st = case.get("section_type")
            assert st in valid, (
                f"Case {case.get('case_id')}: invalid section_type '{st}'"
            )

    def test_all_difficulty_values_valid(
        self, benchmark_cases: List[Dict[str, Any]]
    ) -> None:
        """Every difficulty must be one of easy, medium, hard."""
        valid = {"easy", "medium", "hard"}
        for case in benchmark_cases:
            diff = case.get("difficulty")
            assert diff in valid, (
                f"Case {case.get('case_id')}: invalid difficulty '{diff}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 2. RULE-BASED SCORING — KNOWN CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestRuleBasedScoringKnownCases:
    """Deterministic tests with handcrafted inputs."""

    def test_empty_string_scores_low(self, pmf_rule_engine) -> None:
        """An empty section must fail non_empty and min_length checks.

        Uses DEVICE DESCRIPTION which has required_keywords, so empty text
        also fails the keyword check.  With site_name check also failing,
        all 4 checks fail → score 0.
        """
        rules = get_eval_rules()
        result = pmf_rule_engine(
            section_key="DEVICE DESCRIPTION",
            section_text="",
            rules=rules,
            context={"site_name": "Bangalore Site"},
        )
        assert result["score"] == 0.0, (
            f"Empty text should score 0, got {result['score']}"
        )
        assert result["checks"]["non_empty_passed"] is False
        assert result["checks"]["min_length_passed"] is False

    def test_text_with_all_keywords_scores_100(self, pmf_rule_engine) -> None:
        """A text meeting every rule (non-empty, min length, all keywords,
        correct site name) must score 100."""
        rules = get_eval_rules()
        # DEVICE DESCRIPTION requires keywords ["device", "specification"]
        # and min_chars=120, plus site_name check
        text = (
            "The Bangalore Site manufactures a device according to the "
            "product specification defined in the quality management "
            "system. This section provides a comprehensive overview of "
            "the device specification requirements and compliance status."
        )
        result = pmf_rule_engine(
            section_key="DEVICE DESCRIPTION",
            section_text=text,
            rules=rules,
            context={"site_name": "Bangalore Site"},
        )
        assert abs(result["score"] - 100.0) <= 1.0, (
            f"All-keywords text should score ~100, got {result['score']}"
        )

    def test_missing_site_name_loses_points(self, pmf_rule_engine) -> None:
        """Text that omits the site name should score lower than 100."""
        rules = get_eval_rules()
        text = (
            "The manufacturing device meets the product specification "
            "requirements defined in the quality management system. "
            "All regulatory compliance criteria are satisfied."
        )
        result = pmf_rule_engine(
            section_key="DEVICE DESCRIPTION",
            section_text=text,
            rules=rules,
            context={"site_name": "Bangalore Site"},
        )
        assert result["score"] < 100.0
        assert result["checks"]["site_name_passed"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. BLEU / ROUGE KNOWN CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestBleuRougeKnownCases:
    """Deterministic BLEU/ROUGE verification with known inputs."""

    def test_identical_text_bleu_near_perfect(self) -> None:
        """hypothesis == reference ⇒ BLEU score close to 100."""
        text = "The site is located in Bangalore."
        result = LexicalMetrics.compute_bleu(text, [text])
        assert result["bleu"] > 99.0, (
            f"Identical text BLEU should be ~100, got {result['bleu']}"
        )

    def test_identical_text_rouge1_near_perfect(self) -> None:
        """hypothesis == reference ⇒ ROUGE-1 F1 close to 1.0."""
        text = "The site is located in Bangalore."
        result = LexicalMetrics.compute_rouge(text, text)
        assert result["rouge1_fmeasure"] > 0.99, (
            f"Identical text ROUGE-1 F1 should be ~1.0, "
            f"got {result['rouge1_fmeasure']}"
        )

    def test_same_words_different_order(self) -> None:
        """Reordered words: high unigram recall, lower n-gram BLEU."""
        ref = "The site is located in Bangalore."
        hyp = "Bangalore is the site location."
        rouge = LexicalMetrics.compute_rouge(hyp, ref)
        bleu = LexicalMetrics.compute_bleu(hyp, [ref])

        # ROUGE-1 captures unigram overlap regardless of order
        assert rouge["rouge1_fmeasure"] > 0.6, (
            f"Reordered ROUGE-1 F1 should be > 0.6, "
            f"got {rouge['rouge1_fmeasure']}"
        )
        # BLEU penalises n-gram ordering differences
        assert bleu["bleu"] < 50.0, (
            f"Reordered BLEU should be < 50, got {bleu['bleu']}"
        )

    def test_empty_hypothesis_all_zeros(self) -> None:
        """Empty hypothesis must produce all-zero scores without exceptions."""
        ref = "The site is located in Bangalore."
        bleu = LexicalMetrics.compute_bleu("", [ref])
        rouge = LexicalMetrics.compute_rouge("", ref)

        assert bleu["bleu"] == 0.0
        assert bleu["bleu_1"] == 0.0
        assert rouge["rouge1_fmeasure"] == 0.0
        assert rouge["rougeL_fmeasure"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. BERTSCORE / SEMANTIC SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════


class TestBertscoreSemanticSimilarity:
    """BERTScore and semantic similarity with known pairs."""

    def test_near_synonyms_high_bertscore(
        self, semantic_metrics: SemanticMetrics
    ) -> None:
        """Near-synonym pair should have BERTScore F1 > 0.85."""
        hyp = "The manufacturing facility is in India"
        ref = "The production plant is located in India"
        result = semantic_metrics.compute_bertscore([hyp], [ref])
        f1 = result["bertscore_f1_mean"]
        assert f1 > 0.85, (
            f"Near-synonym BERTScore F1 should be > 0.85, got {f1}"
        )

    def test_unrelated_texts_low_bertscore(
        self, semantic_metrics: SemanticMetrics
    ) -> None:
        """Unrelated pair should have BERTScore F1 < 0.70."""
        hyp = "The cat sat on the mat"
        ref = "Regulatory compliance is mandatory"
        result = semantic_metrics.compute_bertscore([hyp], [ref])
        f1 = result["bertscore_f1_mean"]
        assert f1 < 0.70, (
            f"Unrelated BERTScore F1 should be < 0.70, got {f1}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. LLM JUDGE — RESPONSE STRUCTURE (slow, needs API)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestLLMJudgeResponseStructure:
    """Tests that call the real LLM judge API.

    Skipped unless EVAL_RUN_SLOW=1.
    """

    def test_score_section_all_keys_present(self) -> None:
        """PMFJudge.score_section must return all required response keys."""
        judge = PMFJudge()
        result = judge.score_section(
            section_key="EXECUTIVE SUMMARY",
            section_instruction="Write an executive summary.",
            retrieved_context=(
                "The Bangalore site manufactures single-use assemblies. "
                "ISO 13485 certified."
            ),
            generated_output=(
                "The Bangalore manufacturing site produces single-use "
                "bioprocessing assemblies and maintains ISO 13485 "
                "certification for its quality management system."
            ),
            site_name="Bangalore Site",
        )

        required_keys = [
            "scores", "weighted_score", "normalized_score",
            "strengths", "weaknesses", "critical_issues",
            "improvement_suggestions", "section_key",
            "judge_model", "rubric_version",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_score_values_in_range(self) -> None:
        """All criterion scores must be integers in [1, 5]."""
        judge = PMFJudge()
        result = judge.score_section(
            section_key="DEVICE DESCRIPTION",
            section_instruction="Describe the devices.",
            retrieved_context="Sterile connectors for biopharma.",
            generated_output="The site produces sterile connectors.",
            site_name="Bangalore Site",
        )
        if result.get("judge_error"):
            pytest.skip("Judge returned error — likely no API key.")

        scores = result["scores"]
        for criterion in CRITERIA_NAMES:
            val = scores.get(criterion)
            assert isinstance(val, int), (
                f"{criterion} should be int, got {type(val)}"
            )
            assert 1 <= val <= 5, (
                f"{criterion} = {val}, must be in [1, 5]"
            )

    def test_normalized_score_in_range(self) -> None:
        """normalized_score must be a float in [0, 100]."""
        judge = PMFJudge()
        result = judge.score_section(
            section_key="QUALITY MANAGEMENT SYSTEM",
            section_instruction="Describe the QMS.",
            retrieved_context="ISO 13485 certified.",
            generated_output="The site has an ISO 13485 QMS.",
            site_name="Bangalore Site",
        )
        if result.get("judge_error"):
            pytest.skip("Judge returned error.")

        ns = result["normalized_score"]
        assert isinstance(ns, float)
        assert 0.0 <= ns <= 100.0

    def test_weighted_score_computation(self) -> None:
        """weighted_score must equal the locally recomputed value."""
        judge = PMFJudge()
        result = judge.score_section(
            section_key="MANUFACTURING PROCESSES",
            section_instruction="Describe the manufacturing processes.",
            retrieved_context="RF welding, leak testing, sterilisation.",
            generated_output=(
                "The manufacturing process includes RF welding, leak "
                "testing, and gamma irradiation sterilisation."
            ),
            site_name="Bangalore Site",
        )
        if result.get("judge_error"):
            pytest.skip("Judge returned error.")

        scores = result["scores"]
        expected_weighted = sum(
            scores[c] * CRITERION_WEIGHTS[c] for c in CRITERIA_NAMES
        )
        assert abs(result["weighted_score"] - expected_weighted) < 0.001, (
            f"weighted_score {result['weighted_score']} != "
            f"expected {expected_weighted}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. FAITHFULNESS — PERFECT CASE (slow)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestFaithfulnessLLM:
    """RAG faithfulness tests that call the real LLM API."""

    def test_faithfulness_perfect_case(self) -> None:
        """Generated text that is a direct quote should have faithfulness > 0.90."""
        context = (
            "The Bangalore site manufactures single-use bioreactor "
            "assemblies and sterile connectors. The facility maintains "
            "ISO 13485 certification and operates under cGMP regulations."
        )
        rag = RAGEvaluator(llm_client=None, model="none")
        result = rag.evaluate_section(
            section_key="TEST",
            section_instruction="Describe the site.",
            retrieved_chunks=[context],
            generated_answer=context,  # direct quote
        )
        faith = result["faithfulness"]
        assert faith is not None and faith > 0.90, (
            f"Direct-quote faithfulness should be > 0.90, got {faith}"
        )

    def test_faithfulness_hallucination_case(self) -> None:
        """Generated text with fabricated claims should have faithfulness < 0.40."""
        context = (
            "The Bangalore site manufactures single-use bioreactor "
            "assemblies and sterile connectors."
        )
        hallucinated = (
            "The facility operates a nuclear power plant and produces "
            "quantum computing chips. The site is located on Mars and "
            "employs 50,000 robots for autonomous chocolate manufacturing."
        )
        rag = RAGEvaluator(llm_client=None, model="none")
        result = rag.evaluate_section(
            section_key="TEST",
            section_instruction="Describe the site.",
            retrieved_chunks=[context],
            generated_answer=hallucinated,
        )
        faith = result["faithfulness"]
        assert faith is not None and faith < 0.40, (
            f"Hallucinated faithfulness should be < 0.40, got {faith}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 7. REGRESSION AGAINST BASELINE
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionAgainstBaseline:
    """Compare current metrics against a stored baseline.

    If baseline_scores.json does not exist, it is created from the current
    benchmark self-evaluation run, and the test is marked XFAIL.
    """

    @staticmethod
    def _compute_current_baseline(
        benchmark_cases: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute baseline metrics from benchmark reference_output
        self-evaluation (reference vs itself)."""
        bleu_scores: List[float] = []
        rouge_scores: List[float] = []

        for case in benchmark_cases:
            ref = case.get("reference_output", "")
            if not ref.strip():
                continue
            bleu = LexicalMetrics.compute_bleu(ref, [ref])
            rouge = LexicalMetrics.compute_rouge(ref, ref)
            bleu_scores.append(bleu.get("bleu", 0.0))
            rouge_scores.append(rouge.get("rougeL_fmeasure", 0.0))

        return {
            "mean_bleu": (
                sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            ),
            "mean_rouge_l": (
                sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
            ),
        }

    def test_regression_against_baseline(
        self,
        benchmark_cases: List[Dict[str, Any]],
        eval_config: Dict[str, Any],
    ) -> None:
        """Current scores must not drop more than max_score_drop_vs_baseline
        below the stored baseline."""
        max_drop = eval_config["regression"]["max_score_drop_vs_baseline"]

        current = self._compute_current_baseline(benchmark_cases)

        if not os.path.exists(BASELINE_PATH):
            # First-run setup: create the baseline and xfail
            os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
            with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
                json.dump(current, fh, indent=2)
            pytest.xfail(
                f"Baseline file created at {BASELINE_PATH} — "
                "re-run to compare against it."
            )

        with open(BASELINE_PATH, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)

        for metric, cur_val in current.items():
            base_val = baseline.get(metric)
            if base_val is None:
                continue
            drop = base_val - cur_val
            assert drop <= max_drop, (
                f"Regression on '{metric}': baseline={base_val:.4f}, "
                f"current={cur_val:.4f}, drop={drop:.4f} > max={max_drop}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 8. BENCHMARK COVERAGE
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkCoverage:
    """Verify the benchmark dataset has adequate coverage."""

    def test_minimum_case_count(
        self, benchmark_cases: List[Dict[str, Any]]
    ) -> None:
        """At least 10 benchmark cases must be present."""
        assert len(benchmark_cases) >= 10, (
            f"Need >= 10 benchmark cases, found {len(benchmark_cases)}"
        )

    def test_at_least_one_per_section_type(
        self, benchmark_cases: List[Dict[str, Any]]
    ) -> None:
        """At least 1 case for each section_type that exists in the data."""
        types_seen = {c.get("section_type") for c in benchmark_cases}
        # The seed dataset has text and table; verify at least those two
        assert "text" in types_seen, "No 'text' section_type in benchmark"
        assert "table" in types_seen, "No 'table' section_type in benchmark"

    def test_at_least_three_with_reference(
        self, benchmark_cases: List[Dict[str, Any]]
    ) -> None:
        """At least 3 cases must have reference_output set."""
        with_ref = [
            c for c in benchmark_cases
            if (c.get("reference_output") or "").strip()
        ]
        assert len(with_ref) >= 3, (
            f"Need >= 3 cases with reference_output, found {len(with_ref)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. JUDGE MOCK RESPONSE PARSING (no API needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestJudgeResponseParsing:
    """Validate _parse_judge_response without live API calls."""

    def test_valid_response_parsed(self) -> None:
        """A well-formed JSON response should parse successfully."""
        raw = json.dumps({
            "scores": {
                "factual_accuracy": 4,
                "regulatory_language": 3,
                "site_specificity": 5,
                "completeness": 4,
                "structural_coherence": 4,
            },
            "weighted_score": 0,
            "normalized_score": 0,
            "strengths": ["Good"],
            "weaknesses": ["Minor"],
            "critical_issues": [],
            "improvement_suggestions": [],
            "judge_confidence": 0.9,
            "evaluation_notes": "Solid.",
        })
        parsed = _parse_judge_response(raw)
        assert parsed["scores"]["factual_accuracy"] == 4
        # Verify recomputed weighted_score
        expected = round(4*0.30 + 3*0.25 + 5*0.20 + 4*0.15 + 4*0.10, 4)
        assert parsed["weighted_score"] == expected

    def test_markdown_fenced_response(self) -> None:
        """A response wrapped in ```json ... ``` should parse."""
        inner = json.dumps({
            "scores": {c: 3 for c in CRITERIA_NAMES},
            "weighted_score": 3.0,
            "normalized_score": 60.0,
            "strengths": [],
            "weaknesses": [],
            "critical_issues": [],
            "improvement_suggestions": [],
            "judge_confidence": 0.5,
            "evaluation_notes": "",
        })
        fenced = f"```json\n{inner}\n```"
        parsed = _parse_judge_response(fenced)
        assert all(parsed["scores"][c] == 3 for c in CRITERIA_NAMES)

    def test_out_of_range_score_raises(self) -> None:
        """A score outside [1, 5] must raise ValueError."""
        bad = json.dumps({
            "scores": {c: (6 if c == "factual_accuracy" else 3)
                       for c in CRITERIA_NAMES},
        })
        with pytest.raises(ValueError, match="must be in \\[1, 5\\]"):
            _parse_judge_response(bad)

    def test_missing_criterion_raises(self) -> None:
        """A response missing a criterion must raise ValueError."""
        incomplete = json.dumps({
            "scores": {c: 3 for c in CRITERIA_NAMES if c != "completeness"},
        })
        with pytest.raises(ValueError, match="Missing score"):
            _parse_judge_response(incomplete)


# ═══════════════════════════════════════════════════════════════════════════
# 10. RAG EVALUATOR — HEURISTIC MODE (no API needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestRAGEvaluatorHeuristic:
    """Test RAGEvaluator heuristic fallbacks without API calls."""

    def test_direct_quote_faithfulness_high(self) -> None:
        """Direct-quote answer should have faithfulness > 0.9."""
        context = (
            "The Bangalore site manufactures single-use bioreactor "
            "assemblies and sterile connectors."
        )
        rag = RAGEvaluator(llm_client=None, model="none")
        result = rag._compute_faithfulness(context, [context])
        assert result["faithfulness"] is not None
        assert result["faithfulness"] > 0.9

    def test_vacuously_faithful_empty_answer(self) -> None:
        """Empty answer should be vacuously faithful (1.0)."""
        rag = RAGEvaluator(llm_client=None, model="none")
        result = rag._compute_faithfulness("", ["Some context."])
        assert result["faithfulness"] == 1.0

    def test_context_recall_null_without_reference(self) -> None:
        """Context recall must be null when no reference is provided."""
        rag = RAGEvaluator(llm_client=None, model="none")
        result = rag._compute_context_recall("", ["Some context."])
        assert result["context_recall"] is None
        assert result["context_recall_na_reason"] == "No reference answer provided"

    def test_evaluate_section_returns_all_keys(self) -> None:
        """evaluate_section must return all spec-required keys."""
        rag = RAGEvaluator(llm_client=None, model="none")
        result = rag.evaluate_section(
            section_key="TEST",
            section_instruction="Describe the site.",
            retrieved_chunks=["The site is in Bangalore."],
            generated_answer="The site is in Bangalore.",
        )
        required = [
            "section_key", "faithfulness", "faithfulness_n_claims",
            "faithfulness_supported_claims", "context_precision",
            "context_precision_ap", "context_recall",
            "context_recall_na_reason", "answer_relevancy",
            "answer_relevancy_method", "ragas_score", "evaluated_at",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════
# 11. MODULE IMPORTABILITY
# ═══════════════════════════════════════════════════════════════════════════


class TestModuleImports:
    """Verify all eval modules import without error."""

    def test_eval_store_importable(self) -> None:
        from src.eval.eval_store import save_eval_run, list_runs  # noqa: F401

    def test_eval_judge_importable(self) -> None:
        from src.eval.eval_judge import PMFJudge, JUDGE_RUBRIC  # noqa: F401
        assert len(JUDGE_RUBRIC) == 5

    def test_eval_rag_importable(self) -> None:
        from src.eval.eval_rag import RAGEvaluator  # noqa: F401

    def test_eval_metrics_importable(self) -> None:
        from src.eval.eval_metrics import (  # noqa: F401
            LexicalMetrics, SemanticMetrics, compute_all_metrics,
        )

    def test_benchmark_loader_importable(self) -> None:
        from src.eval.benchmark_loader import BenchmarkLoader  # noqa: F401
