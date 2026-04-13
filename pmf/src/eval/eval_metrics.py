"""
LLM Evaluation Metrics Module — Lexical & Semantic Scoring
===========================================================

Provides BLEU, ROUGE, BERTScore, and semantic-similarity metrics for
evaluating generated PMF document sections against ground-truth references.

Part of the Healthark GenAI Evaluation Framework (Initiative 4).

Dependencies:
    pip install sacrebleu rouge-score bert-score sentence-transformers
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe, lazy imports — metrics degrade gracefully when a library is missing.
# ---------------------------------------------------------------------------

try:
    import sacrebleu as _sacrebleu
except ImportError:
    _sacrebleu = None
    logger.warning("sacrebleu not installed — BLEU metrics will be unavailable.")

try:
    from rouge_score import rouge_scorer as _rouge_scorer
except ImportError:
    _rouge_scorer = None
    logger.warning("rouge-score not installed — ROUGE metrics will be unavailable.")

try:
    import bert_score as _bert_score
except (ImportError, OSError):
    _bert_score = None
    logger.warning("bert-score not available — BERTScore metrics will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except (ImportError, OSError):
    _SentenceTransformer = None
    logger.warning(
        "sentence-transformers not available — semantic similarity will be unavailable."
    )


def _round(value: float, decimals: int = 4) -> float:
    """Round a float to *decimals* places (JSON-safe)."""
    return round(float(value), decimals)


# ═══════════════════════════════════════════════════════════════════════════
# LEXICAL METRICS
# ═══════════════════════════════════════════════════════════════════════════


class LexicalMetrics:
    """Corpus-/sentence-level BLEU and ROUGE metrics.

    All results are JSON-serializable dicts of floats rounded to 4 d.p.
    """

    # ── BLEU ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_bleu(
        hypothesis: str, references: List[str]
    ) -> Dict[str, float]:
        """Compute sentence-level BLEU and per-n-gram precisions.

        Uses *sacrebleu* for reproducibility (not nltk).

        Args:
            hypothesis:  Single generated string.
            references:  One or more reference strings.

        Returns:
            Dictionary with keys ``bleu``, ``bleu_1`` … ``bleu_4``.
            Values are floats in [0, 100] for ``bleu`` and [0, 100] for
            individual n-gram precisions (sacrebleu convention).

        Raises:
            Nothing — returns zeros on error or empty input.

        Example:
            >>> LexicalMetrics.compute_bleu("the cat sat", ["the cat sat on the mat"])
            {'bleu': 48.5271, 'bleu_1': 100.0, 'bleu_2': 66.6667, ...}
        """
        zero = {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        if _sacrebleu is None:
            return {**zero, "error": "sacrebleu not installed", "value": None}
        try:
            if not hypothesis or not hypothesis.strip():
                return zero

            # sacrebleu.sentence_bleu expects refs as List[str]
            result = _sacrebleu.sentence_bleu(hypothesis, references)

            precisions = list(result.precisions)  # length 4
            # If hypothesis is shorter than n-gram order, precision is 0
            out = {
                "bleu": _round(result.score),
                "bleu_1": _round(precisions[0]) if len(precisions) > 0 else 0.0,
                "bleu_2": _round(precisions[1]) if len(precisions) > 1 else 0.0,
                "bleu_3": _round(precisions[2]) if len(precisions) > 2 else 0.0,
                "bleu_4": _round(precisions[3]) if len(precisions) > 3 else 0.0,
            }
            return out
        except Exception as exc:
            logger.warning("BLEU computation failed: %s", exc)
            return {**zero, "error": "bleu computation failed", "value": None}

    # ── ROUGE ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_rouge(
        hypothesis: str, reference: str
    ) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L (precision, recall, F1).

        Uses Google's ``rouge_score`` library with stemming enabled.

        Args:
            hypothesis: Single generated string.
            reference:  Single reference string.

        Returns:
            Dict with 9 keys: ``rouge{1,2,L}_{precision,recall,fmeasure}``.

        Raises:
            Nothing — returns zeros on error or empty input.

        Example:
            >>> LexicalMetrics.compute_rouge("the cat sat", "the cat sat on the mat")
            {'rouge1_precision': 1.0, 'rouge1_recall': 0.5, ...}
        """
        zero_keys = [
            f"{name}_{part}"
            for name in ("rouge1", "rouge2", "rougeL")
            for part in ("precision", "recall", "fmeasure")
        ]
        zero = {k: 0.0 for k in zero_keys}

        if _rouge_scorer is None:
            return {**zero, "error": "rouge-score not installed", "value": None}
        try:
            if not hypothesis or not hypothesis.strip() or not reference or not reference.strip():
                return zero

            scorer = _rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            scores = scorer.score(reference, hypothesis)

            out: Dict[str, float] = {}
            for metric_name in ("rouge1", "rouge2", "rougeL"):
                s = scores[metric_name]
                out[f"{metric_name}_precision"] = _round(s.precision)
                out[f"{metric_name}_recall"] = _round(s.recall)
                out[f"{metric_name}_fmeasure"] = _round(s.fmeasure)
            return out
        except Exception as exc:
            logger.warning("ROUGE computation failed: %s", exc)
            return {**zero, "error": "rouge computation failed", "value": None}

    # ── combined ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_all_lexical(
        hypothesis: str, reference: str
    ) -> Dict[str, float]:
        """Run BLEU *and* ROUGE and merge results.

        Args:
            hypothesis: Generated text.
            reference:  Reference text (also used as the single BLEU reference).

        Returns:
            Merged dictionary of all BLEU + ROUGE keys.

        Raises:
            Nothing — individual sub-metrics degrade independently.

        Example:
            >>> LexicalMetrics.compute_all_lexical("the cat", "the cat sat on the mat")
        """
        bleu = LexicalMetrics.compute_bleu(hypothesis, [reference])
        rouge = LexicalMetrics.compute_rouge(hypothesis, reference)
        return {**bleu, **rouge}


# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC METRICS
# ═══════════════════════════════════════════════════════════════════════════


class SemanticMetrics:
    """BERTScore and sentence-embedding cosine similarity.

    The BERTScore model is loaded lazily and cached across calls within the
    same ``SemanticMetrics`` instance.
    """

    _NUM_LAYERS = {
        "distilbert-base-uncased": 5,
        "roberta-large": 17,
    }

    def __init__(
        self,
        model_type: str = "distilbert-base-uncased",
        device: str = "cpu",
        batch_size: int = 8,
    ):
        """Initialise the semantic-metrics evaluator.

        Args:
            model_type: BERTScore model. ``"distilbert-base-uncased"`` for
                speed, ``"roberta-large"`` for highest accuracy.
            device:     ``"cpu"`` or ``"cuda"``.
            batch_size: Batch size for BERTScore inference.
        """
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self.num_layers = self._NUM_LAYERS.get(model_type)
        self._embedder: Optional[Any] = None  # lazy SentenceTransformer

    # ── BERTScore ─────────────────────────────────────────────────────────

    def compute_bertscore(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """Compute BERTScore for aligned hypothesis-reference pairs.

        Args:
            hypotheses: List of generated strings.
            references: List of reference strings (same length).

        Returns:
            Dict with corpus-level means (``bertscore_precision_mean``, etc.)
            and ``bertscore_per_example`` list of per-pair dicts.

        Raises:
            Nothing — returns zeros on error.

        Example:
            >>> sm = SemanticMetrics()
            >>> sm.compute_bertscore(["the cat"], ["the cat sat on mat"])
        """
        empty_result: Dict[str, Any] = {
            "bertscore_precision_mean": 0.0,
            "bertscore_recall_mean": 0.0,
            "bertscore_f1_mean": 0.0,
            "bertscore_per_example": [],
        }

        if _bert_score is None:
            return {**empty_result, "error": "bert-score not installed", "value": None}
        try:
            if len(hypotheses) != len(references):
                raise ValueError(
                    f"Length mismatch: {len(hypotheses)} hypotheses vs "
                    f"{len(references)} references."
                )

            # Handle all-empty edge case
            if all(not h or not h.strip() for h in hypotheses):
                per_ex = [{"precision": 0.0, "recall": 0.0, "f1": 0.0}] * len(hypotheses)
                return {**empty_result, "bertscore_per_example": per_ex}

            kwargs: Dict[str, Any] = {
                "cands": hypotheses,
                "refs": references,
                "model_type": self.model_type,
                "device": self.device,
                "batch_size": self.batch_size,
                "rescale_with_baseline": True,
                "lang": "en",
            }
            if self.num_layers is not None:
                kwargs["num_layers"] = self.num_layers

            P, R, F1 = _bert_score.score(**kwargs)

            per_example = []
            for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
                per_example.append(
                    {"precision": _round(p), "recall": _round(r), "f1": _round(f)}
                )

            return {
                "bertscore_precision_mean": _round(P.mean().item()),
                "bertscore_recall_mean": _round(R.mean().item()),
                "bertscore_f1_mean": _round(F1.mean().item()),
                "bertscore_per_example": per_example,
            }
        except Exception as exc:
            logger.warning("BERTScore computation failed: %s", exc)
            return {**empty_result, "error": "bertscore computation failed", "value": None}

    # ── Semantic similarity (sentence embeddings) ─────────────────────────

    def compute_semantic_similarity(
        self,
        text_a: str,
        text_b: str,
        embedder: Optional[Any] = None,
    ) -> float:
        """Cosine similarity between sentence embeddings of two texts.

        Uses ``all-MiniLM-L6-v2`` by default — lightweight and fast.
        This is *separate* from BERTScore; it measures overall section
        coherence via a single embedding per text.

        Args:
            text_a:   First text (typically the generated section).
            text_b:   Second text (typically the reference).
            embedder: Optional pre-loaded ``SentenceTransformer`` instance.

        Returns:
            Cosine similarity as a float in [-1, 1] (usually [0, 1]).

        Raises:
            Nothing — returns 0.0 on error.

        Example:
            >>> sm = SemanticMetrics()
            >>> sm.compute_semantic_similarity("the cat sat", "a cat sitting")
            0.7832
        """
        if _SentenceTransformer is None:
            logger.warning("sentence-transformers not installed — returning 0.0")
            return 0.0
        try:
            if not text_a or not text_a.strip() or not text_b or not text_b.strip():
                return 0.0

            if embedder is None:
                if self._embedder is None:
                    self._embedder = _SentenceTransformer("all-MiniLM-L6-v2")
                embedder = self._embedder

            embeddings = embedder.encode([text_a, text_b], normalize_embeddings=True)
            # Dot product of L2-normalised vectors == cosine similarity
            similarity = float(embeddings[0] @ embeddings[1])
            return _round(similarity)
        except Exception as exc:
            logger.warning("Semantic similarity computation failed: %s", exc)
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def compute_all_metrics(
    hypothesis: str,
    reference: str,
    include_bertscore: bool = True,
    bertscore_model: str = "distilbert-base-uncased",
) -> Dict[str, Any]:
    """Run every available metric and return a unified results dict.

    This is the primary entry point for the evaluation pipeline — call it
    once per section with the generated text and its ground-truth reference.

    Args:
        hypothesis:      Generated text.
        reference:       Ground-truth reference text.
        include_bertscore: If ``False``, skip BERTScore (saves ~2 s per call).
        bertscore_model: Model for BERTScore (see ``SemanticMetrics``).

    Returns:
        Combined dictionary::

            {
                "hypothesis_len": int,
                "reference_len": int,
                "lexical": { ...BLEU + ROUGE keys... },
                "semantic": { ...BERTScore keys... },
                "semantic_similarity": float,
                "computed_at": "2026-04-11T..."
            }

    Raises:
        Nothing — individual sub-metrics degrade independently and log warnings.

    Example:
        >>> result = compute_all_metrics(
        ...     "The Langensbold site manufactures sterile injectable products.",
        ...     "The Langensbold manufacturing site produces sterile injectables."
        ... )
        >>> result["lexical"]["bleu"]
        34.1234
    """
    lexical = LexicalMetrics.compute_all_lexical(hypothesis, reference)

    semantic: Dict[str, Any] = {}
    similarity = 0.0

    sm = SemanticMetrics(model_type=bertscore_model)

    if include_bertscore:
        semantic = sm.compute_bertscore([hypothesis], [reference])

    similarity = sm.compute_semantic_similarity(hypothesis, reference)

    return {
        "hypothesis_len": len(hypothesis) if hypothesis else 0,
        "reference_len": len(reference) if reference else 0,
        "lexical": lexical,
        "semantic": semantic,
        "semantic_similarity": similarity,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    # --- Synthetic PMF-style hypothesis / reference pair ---
    hyp = (
        "The Langensbold manufacturing site is a state-of-the-art facility "
        "operated by Thermo Fisher Scientific. The site specialises in the "
        "production of single-use bioreactor assemblies and sterile "
        "connectors used in biopharmaceutical manufacturing. Quality "
        "management follows ISO 13485 and cGMP guidelines to ensure "
        "product safety and regulatory compliance."
    )

    ref = (
        "The Langensbold site, operated by Thermo Fisher Scientific, "
        "manufactures single-use bioprocessing components including "
        "bioreactor bags and sterile connectors. The facility maintains "
        "ISO 13485 certification and operates under cGMP regulations to "
        "guarantee consistent product quality and compliance with "
        "regulatory requirements."
    )

    print("=" * 72)
    print("EVAL METRICS — SMOKE TEST")
    print("=" * 72)

    # 1. Lexical
    print("\n--- LexicalMetrics.compute_bleu ---")
    bleu = LexicalMetrics.compute_bleu(hyp, [ref])
    print(json.dumps(bleu, indent=2))

    print("\n--- LexicalMetrics.compute_rouge ---")
    rouge = LexicalMetrics.compute_rouge(hyp, ref)
    print(json.dumps(rouge, indent=2))

    # 2. Semantic
    sm = SemanticMetrics(model_type="distilbert-base-uncased")

    print("\n--- SemanticMetrics.compute_bertscore ---")
    bs = sm.compute_bertscore([hyp], [ref])
    print(json.dumps(bs, indent=2))

    print("\n--- SemanticMetrics.compute_semantic_similarity ---")
    sim = sm.compute_semantic_similarity(hyp, ref)
    print(f"Cosine similarity: {sim}")

    # 3. Combined
    print("\n--- compute_all_metrics ---")
    result = compute_all_metrics(hyp, ref, include_bertscore=True)
    print(json.dumps(result, indent=2))

    # 4. Edge cases
    print("\n--- Edge case: empty hypothesis ---")
    edge = compute_all_metrics("", ref, include_bertscore=False)
    print(json.dumps(edge, indent=2))

    print("\n" + "=" * 72)
    print("SMOKE TEST COMPLETE")
    print("=" * 72)
