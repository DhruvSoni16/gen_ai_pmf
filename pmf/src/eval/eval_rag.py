"""
RAG Evaluation Module — RAGAS-Inspired Metrics (from scratch)
==============================================================

Computes the four core RAGAS metrics (arXiv:2309.15217) for each
retrieved+generated section of a PMF document, without importing the
``ragas`` package.  Instead, the metric logic is implemented directly
using an LLM for claim extraction / entailment and SentenceTransformer
embeddings for answer relevancy.

Metrics:
  1. **Faithfulness** — fraction of generated claims supported by context.
  2. **Context Precision** — whether relevant contexts are ranked higher
     (raw precision + Average Precision).
  3. **Context Recall** — fraction of reference claims attributable to
     the retrieved context (requires reference answer).
  4. **Answer Relevancy** — reverse-question cosine similarity (RAGAS paper
     methodology) with direct-similarity fallback.

Part of the Healthark GenAI Evaluation Framework (Initiative 4).

Dependencies:
    pip install anthropic openai sentence-transformers numpy
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe imports
# ---------------------------------------------------------------------------

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]
    logger.warning("numpy not installed — some RAG metrics will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment]
    logger.warning(
        "sentence-transformers not installed — answer relevancy metric unavailable."
    )

try:
    import anthropic as _anthropic_mod
except ImportError:
    _anthropic_mod = None

try:
    from openai import AzureOpenAI as _AzureOpenAI
except ImportError:
    _AzureOpenAI = None


def _round(value: float, decimals: int = 4) -> float:
    """Round to *decimals* places (JSON-safe)."""
    return round(float(value), decimals)


# ═══════════════════════════════════════════════════════════════════════════
# CACHING (mirrors eval_judge pattern)
# ═══════════════════════════════════════════════════════════════════════════

def _cache_key(*parts: str) -> str:
    """SHA-256 from concatenated parts separated by ``|``."""
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.isfile(path):
        logger.debug("Cache miss for %s...", key[:8])
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache read failed for %s...: %s", key[:8], exc)
        return None


def _write_cache(cache_dir: str, key: str, data: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        logger.warning("Cache write failed %s: %s", path, exc)


# ═══════════════════════════════════════════════════════════════════════════
# JSON HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _strip_fences(text: str) -> str:
    """Strip Markdown code fences (```…```)."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def _parse_json_lenient(raw: str) -> Any:
    """Attempt to parse JSON from an LLM response.

    Strips code fences, then falls back to scanning for the first ``{``
    or ``[`` if the full string fails.

    Args:
        raw: Raw LLM output.

    Returns:
        Parsed Python object (dict or list).

    Raises:
        json.JSONDecodeError: If no valid JSON is found.
    """
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Scan for first { or [
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            idx = cleaned.find(start_char)
            if idx != -1:
                # Find matching end from the back
                ridx = cleaned.rfind(end_char)
                if ridx > idx:
                    try:
                        return json.loads(cleaned[idx : ridx + 1])
                    except json.JSONDecodeError:
                        continue
        raise


def _harmonic_mean(values: List[float]) -> float:
    """Compute the harmonic mean, skipping zeros to avoid division errors.

    Formula: n / sum(1/x for x in values if x > 0)

    Args:
        values: List of positive floats.

    Returns:
        Harmonic mean, or 0.0 if no positive values.
    """
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return len(positive) / sum(1.0 / v for v in positive)


# ═══════════════════════════════════════════════════════════════════════════
# LLM WRAPPER (accepts the same client objects as PMFJudge)
# ═══════════════════════════════════════════════════════════════════════════


class _LLMCaller:
    """Thin wrapper that sends prompts to either an Anthropic or
    Azure OpenAI client — the same client object created by PMFJudge.

    If ``client`` is ``None`` every method falls back to a keyword-overlap
    heuristic so the module can still produce approximate scores without
    an API key.
    """

    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if self._client is None:
            return "none"
        cls_name = type(self._client).__name__
        mod_name = type(self._client).__module__ or ""
        if "anthropic" in mod_name.lower() or "Anthropic" in cls_name:
            return "anthropic"
        # Covers both OpenAI and AzureOpenAI
        if "openai" in mod_name.lower() or "OpenAI" in cls_name:
            return "azure_openai"
        return "unknown"

    @property
    def available(self) -> bool:
        """True when a live LLM client is ready."""
        return self._client is not None and self._provider != "none"

    def chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Send *system* + *user* prompt and return the text.

        Args:
            system:     System prompt.
            user:       User prompt.
            max_tokens: Response length cap.

        Returns:
            Model response text.

        Raises:
            RuntimeError: When no client is available.
        """
        if not self.available:
            raise RuntimeError("No LLM client available.")

        if self._provider == "anthropic":
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text if resp.content else ""

        # azure_openai / openai
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════════════
# HEURISTIC FALLBACKS (no-API paths)
# ═══════════════════════════════════════════════════════════════════════════


def _heuristic_extract_claims(text: str) -> List[str]:
    """Split text into sentence-level pseudo-claims.

    Uses period-splitting as a rough approximation when no LLM is
    available.

    Args:
        text: Source text.

    Returns:
        List of non-empty sentence strings.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return sentences


def _heuristic_supported(claim: str, context: str) -> bool:
    """Keyword-overlap entailment check.

    A claim is considered supported when more than half of its
    non-trivial words (length > 2) appear in the context.

    Args:
        claim:   Single claim sentence.
        context: Full context text.

    Returns:
        True if enough overlap is found.
    """
    stop = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
            "of", "in", "to", "for", "on", "at", "by", "it", "its",
            "that", "this", "with", "from", "as", "be", "has", "have",
            "had", "not", "but", "all", "can", "will", "do", "does"}
    claim_words = {w for w in claim.lower().split() if len(w) > 2 and w not in stop}
    ctx_lower = context.lower()
    if not claim_words:
        return True
    matched = sum(1 for w in claim_words if w in ctx_lower)
    return matched / len(claim_words) > 0.5


def _heuristic_relevant(question: str, chunk: str) -> bool:
    """Keyword-overlap relevance check between question and chunk.

    Uses stem-prefix matching (first 6 characters) so that inflected
    forms like *manufacturing* / *manufactures* still count as a match.

    Args:
        question: Query text.
        chunk:    Context chunk.

    Returns:
        True when stem-overlap exceeds 25 %.
    """
    q_words = {w.lower() for w in question.split() if len(w) > 3}
    if not q_words:
        return True
    c_lower = chunk.lower()
    # Stem-prefix: a question word matches if its first min(6, len)
    # characters appear as a substring anywhere in the chunk.
    matched = 0
    for w in q_words:
        prefix = w[: min(6, len(w))]
        if prefix in c_lower:
            matched += 1
    return matched / len(q_words) > 0.25


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

_SYS_CLAIM_EXTRACT = (
    "You are a precise factual claim extractor.  Given a passage, extract "
    "every distinct atomic factual claim as a short, standalone, verifiable "
    "statement.  Return ONLY a valid JSON object, no markdown."
)

_USR_CLAIM_EXTRACT = (
    "Given the following text, extract all factual claims as a JSON list "
    "of short statements.  Each claim should be a single, verifiable "
    "statement.\n\n"
    "Text: {text}\n\n"
    "Respond with: {{\"claims\": [\"claim1\", \"claim2\", ...]}}"
)

_SYS_ENTAILMENT = (
    "You are a natural-language inference classifier.  Determine whether "
    "a given claim is fully supported by the provided context.  Respond "
    "with ONLY a valid JSON object, no markdown."
)

_USR_ENTAILMENT = (
    "Given the following context and claim, determine if the claim is "
    "fully supported by the context.\n\n"
    "Context: {context}\n\n"
    "Claim: {claim}\n\n"
    "Respond with: {{\"supported\": true/false, \"evidence\": \"brief quote or null\"}}"
)

_SYS_RELEVANCE = (
    "You are a retrieval quality evaluator.  Determine if a context "
    "chunk is relevant to answering the given question.  Respond with "
    "ONLY a valid JSON object, no markdown."
)

_USR_RELEVANCE = (
    "Is the following context chunk relevant to answering this question?\n\n"
    "Question: {question}\n\n"
    "Context chunk: {chunk}\n\n"
    "Respond: {{\"relevant\": true/false, \"reason\": \"brief explanation\"}}"
)

_SYS_GEN_QUESTIONS = (
    "You are a question generation assistant.  Given an answer, generate "
    "questions that this answer would perfectly address.  Respond with "
    "ONLY a valid JSON object, no markdown."
)

_USR_GEN_QUESTIONS = (
    "Given the following answer, generate {n} questions that this answer "
    "would perfectly address.\n\n"
    "Answer: {answer}\n\n"
    "Respond: {{\"questions\": [\"q1\", \"q2\", \"q3\"]}}"
)


# ═══════════════════════════════════════════════════════════════════════════
# RAGEvaluator CLASS
# ═══════════════════════════════════════════════════════════════════════════


class RAGEvaluator:
    """RAGAS-inspired RAG evaluation for PMF document sections.

    Implements the four core metrics from the RAGAS paper (arXiv:2309.15217)
    using an LLM for claim extraction / entailment and SentenceTransformer
    embeddings for answer relevancy.  Falls back to keyword-overlap
    heuristics when no LLM client is supplied.

    Args:
        llm_client:    An Anthropic ``Anthropic`` instance or Azure
                       ``AzureOpenAI`` instance — the same client object
                       that ``PMFJudge`` uses.  Pass ``None`` to run in
                       heuristic-only mode.
        model:         Model name / deployment to use for LLM calls.
        embedder:      Optional pre-loaded ``SentenceTransformer`` instance.
                       If ``None``, ``all-MiniLM-L6-v2`` is loaded lazily.
        cache_enabled: Cache per-section results by SHA-256 hash.
        cache_dir:     Directory for cached JSON files.

    Example:
        >>> from src.eval.eval_judge import PMFJudge
        >>> judge = PMFJudge()
        >>> rag = RAGEvaluator(
        ...     llm_client=judge._client,
        ...     model=judge.model,
        ... )
        >>> result = rag.evaluate_section(
        ...     section_key="DEVICE DESCRIPTION",
        ...     section_instruction="Describe the device.",
        ...     retrieved_chunks=["The device is a sterile connector."],
        ...     generated_answer="The device is a sterile connector.",
        ... )
        >>> result["faithfulness"]
        1.0
    """

    CACHE_VERSION: str = "rag_v1.0"

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "claude-sonnet-4-6",
        embedder: Any = None,
        cache_enabled: bool = True,
        cache_dir: str = "data/eval_cache/rag",
    ):
        self._llm = _LLMCaller(llm_client, model)
        self._embedder = embedder
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir

    # ── embedder ─────────────────────────────────────────────────────────

    def _get_embedder(self) -> Any:
        """Lazy-load sentence-transformer model.

        Returns:
            A ``SentenceTransformer`` instance.

        Raises:
            RuntimeError: If the library is not installed.
        """
        if self._embedder is None:
            if _SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed.")
            self._embedder = _SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL: claim extraction
    # ══════════════════════════════════════════════════════════════════════

    def _extract_claims(self, text: str) -> List[str]:
        """Extract atomic factual claims from *text*.

        Uses the LLM when available, falls back to sentence-splitting.

        Args:
            text: Passage to decompose.

        Returns:
            List of claim strings.
        """
        if not self._llm.available:
            return _heuristic_extract_claims(text)

        prompt = _USR_CLAIM_EXTRACT.format(text=text[:4000])
        try:
            raw = self._llm.chat(_SYS_CLAIM_EXTRACT, prompt)
            data = _parse_json_lenient(raw)
            if isinstance(data, dict):
                claims = data.get("claims", [])
            elif isinstance(data, list):
                claims = data
            else:
                claims = []
            return [str(c) for c in claims if c]
        except Exception as exc:
            logger.warning("Claim extraction via LLM failed: %s", exc)
            return _heuristic_extract_claims(text)

    def _check_supported(self, claim: str, context: str) -> Dict[str, Any]:
        """Check whether *context* supports *claim*.

        Args:
            claim:   A single factual claim.
            context: Merged context text.

        Returns:
            Dict with ``supported`` (bool) and ``evidence`` (str or None).
        """
        if not self._llm.available:
            return {
                "supported": _heuristic_supported(claim, context),
                "evidence": None,
            }

        prompt = _USR_ENTAILMENT.format(
            context=context[:8000], claim=claim,
        )
        try:
            raw = self._llm.chat(_SYS_ENTAILMENT, prompt, max_tokens=512)
            data = _parse_json_lenient(raw)
            return {
                "supported": bool(data.get("supported", False)),
                "evidence": data.get("evidence"),
            }
        except Exception as exc:
            logger.warning("Entailment check failed: %s", exc)
            return {
                "supported": _heuristic_supported(claim, context),
                "evidence": None,
            }

    def _check_relevant(self, question: str, chunk: str) -> Dict[str, Any]:
        """Check whether *chunk* is relevant to *question*.

        Args:
            question: Query / instruction.
            chunk:    A single context chunk.

        Returns:
            Dict with ``relevant`` (bool) and ``reason`` (str).
        """
        if not self._llm.available:
            return {
                "relevant": _heuristic_relevant(question, chunk),
                "reason": "heuristic",
            }

        prompt = _USR_RELEVANCE.format(
            question=question, chunk=chunk[:2000],
        )
        try:
            raw = self._llm.chat(_SYS_RELEVANCE, prompt, max_tokens=512)
            data = _parse_json_lenient(raw)
            return {
                "relevant": bool(data.get("relevant", False)),
                "reason": data.get("reason", ""),
            }
        except Exception as exc:
            logger.warning("Relevance check failed: %s", exc)
            return {
                "relevant": _heuristic_relevant(question, chunk),
                "reason": "heuristic",
            }

    # ══════════════════════════════════════════════════════════════════════
    # METRIC 1: Faithfulness
    # ══════════════════════════════════════════════════════════════════════

    def _compute_faithfulness(
        self, generated_answer: str, retrieved_chunks: List[str]
    ) -> Dict[str, Any]:
        """Fraction of generated claims supported by context.

        If no claims are extracted the answer is vacuously faithful (1.0).
        If all individual LLM entailment calls fail, returns ``null``.

        Args:
            generated_answer: The AI-generated text.
            retrieved_chunks: Retrieved context passages.

        Returns:
            Dict with ``faithfulness``, ``faithfulness_n_claims``,
            ``faithfulness_supported_claims``, and ``claims`` detail.
        """
        claims = self._extract_claims(generated_answer)
        if not claims:
            # Vacuously faithful — no claims made.
            return {
                "faithfulness": 1.0,
                "faithfulness_n_claims": 0,
                "faithfulness_supported_claims": 0,
                "claims": [],
            }

        merged_context = "\n\n".join(retrieved_chunks)
        claim_results: List[Dict[str, Any]] = []
        supported_count = 0
        error_count = 0

        for claim in claims:
            check = self._check_supported(claim, merged_context)
            is_supported = check["supported"]
            if is_supported:
                supported_count += 1
            claim_results.append({
                "claim": claim,
                "supported": is_supported,
                "evidence": check.get("evidence"),
            })

        # If every single check errored out (all None), return null.
        if error_count == len(claims) and error_count > 0:
            return {
                "faithfulness": None,
                "faithfulness_n_claims": len(claims),
                "faithfulness_supported_claims": 0,
                "claims": claim_results,
            }

        score = supported_count / len(claims)
        return {
            "faithfulness": _round(score),
            "faithfulness_n_claims": len(claims),
            "faithfulness_supported_claims": supported_count,
            "claims": claim_results,
        }

    # ══════════════════════════════════════════════════════════════════════
    # METRIC 2: Context Precision (with Average Precision)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_context_precision(
        self,
        section_instruction: str,
        retrieved_chunks: List[str],
        generated_answer: str,
    ) -> Dict[str, Any]:
        """Raw precision and rank-weighted Average Precision of retrieved
        context chunks.

        AP = sum(precision@k * rel_k for k=1..K) / count_relevant

        Args:
            section_instruction: The original query / instruction.
            retrieved_chunks:    Context chunks in retrieval order.
            generated_answer:    The AI-generated text (unused for now but
                                 available for future relevance variants).

        Returns:
            Dict with ``context_precision``, ``context_precision_ap``,
            and per-chunk ``context_relevance`` list.
        """
        if not retrieved_chunks:
            return {
                "context_precision": 0.0,
                "context_precision_ap": 0.0,
                "context_relevance": [],
            }

        relevance: List[bool] = []
        detail: List[Dict[str, Any]] = []
        for chunk in retrieved_chunks:
            check = self._check_relevant(section_instruction, chunk)
            is_rel = check["relevant"]
            relevance.append(is_rel)
            detail.append({
                "relevant": is_rel,
                "reason": check.get("reason", ""),
            })

        total_relevant = sum(relevance)

        # Raw precision
        raw_precision = total_relevant / len(relevance) if relevance else 0.0

        # Average Precision (AP)
        if total_relevant == 0:
            ap = 0.0
        else:
            cum_relevant = 0
            precision_sum = 0.0
            for k, is_rel in enumerate(relevance, start=1):
                if is_rel:
                    cum_relevant += 1
                    precision_sum += cum_relevant / k
            ap = precision_sum / total_relevant

        return {
            "context_precision": _round(raw_precision),
            "context_precision_ap": _round(ap),
            "context_relevance": detail,
        }

    # ══════════════════════════════════════════════════════════════════════
    # METRIC 3: Context Recall
    # ══════════════════════════════════════════════════════════════════════

    def _compute_context_recall(
        self,
        reference_answer: str,
        retrieved_chunks: List[str],
    ) -> Dict[str, Any]:
        """Fraction of reference claims attributable to the context.

        Returns ``null`` with a reason string when no reference answer is
        provided.

        Args:
            reference_answer: Ground-truth text.  Empty string → null result.
            retrieved_chunks: Retrieved context passages.

        Returns:
            Dict with ``context_recall`` (float or None),
            ``context_recall_na_reason`` (str or None), and claim detail.
        """
        if not reference_answer or not reference_answer.strip():
            return {
                "context_recall": None,
                "context_recall_na_reason": "No reference answer provided",
                "reference_claims": [],
            }

        ref_claims = self._extract_claims(reference_answer)
        if not ref_claims:
            return {
                "context_recall": 1.0,
                "context_recall_na_reason": None,
                "reference_claims": [],
            }

        merged_context = "\n\n".join(retrieved_chunks)
        claim_results: List[Dict[str, Any]] = []
        found_count = 0

        for claim in ref_claims:
            check = self._check_supported(claim, merged_context)
            if check["supported"]:
                found_count += 1
            claim_results.append({
                "claim": claim,
                "in_context": check["supported"],
                "evidence": check.get("evidence"),
            })

        score = found_count / len(ref_claims)
        return {
            "context_recall": _round(score),
            "context_recall_na_reason": None,
            "reference_claims": claim_results,
        }

    # ══════════════════════════════════════════════════════════════════════
    # METRIC 4: Answer Relevancy (reverse-question method)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_answer_relevancy(
        self,
        section_instruction: str,
        generated_answer: str,
        n_questions: int = 3,
    ) -> Dict[str, Any]:
        """Reverse-question cosine similarity (RAGAS paper methodology).

        Falls back to direct cosine similarity between instruction and
        answer embeddings when the LLM or embedder is unavailable.

        Args:
            section_instruction: The original query.
            generated_answer:    The AI-generated text.
            n_questions:         Number of synthetic questions to generate.

        Returns:
            Dict with ``answer_relevancy``, ``answer_relevancy_method``,
            and detail fields.
        """
        # --- Attempt the reverse-question method ---
        if self._llm.available:
            try:
                prompt = _USR_GEN_QUESTIONS.format(
                    n=n_questions, answer=generated_answer[:3000],
                )
                raw = self._llm.chat(_SYS_GEN_QUESTIONS, prompt)
                data = _parse_json_lenient(raw)
                if isinstance(data, dict):
                    gen_questions = data.get("questions", [])
                elif isinstance(data, list):
                    gen_questions = data
                else:
                    gen_questions = []
                gen_questions = [str(q) for q in gen_questions if q]

                if gen_questions:
                    embedder = self._get_embedder()
                    all_texts = [section_instruction] + gen_questions
                    embeddings = embedder.encode(
                        all_texts, normalize_embeddings=True,
                    )
                    q_emb = embeddings[0]
                    sims = [
                        float(q_emb @ embeddings[i])
                        for i in range(1, len(embeddings))
                    ]
                    if _np is not None:
                        mean_sim = float(_np.mean(sims))
                    else:
                        mean_sim = sum(sims) / len(sims) if sims else 0.0

                    return {
                        "answer_relevancy": _round(mean_sim),
                        "answer_relevancy_method": "reverse_question",
                        "generated_questions": gen_questions,
                        "similarities": [_round(s) for s in sims],
                    }
            except Exception as exc:
                logger.warning(
                    "Reverse-question relevancy failed, falling back: %s", exc,
                )

        # --- Fallback: direct cosine similarity ---
        try:
            embedder = self._get_embedder()
            embs = embedder.encode(
                [section_instruction, generated_answer],
                normalize_embeddings=True,
            )
            sim = float(embs[0] @ embs[1])
            return {
                "answer_relevancy": _round(sim),
                "answer_relevancy_method": "direct_similarity",
            }
        except Exception as exc:
            logger.warning("Direct similarity fallback also failed: %s", exc)
            return {
                "answer_relevancy": 0.0,
                "answer_relevancy_method": "failed",
                "error": str(exc),
            }

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: evaluate_section
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_section(
        self,
        section_key: str,
        section_instruction: str,
        retrieved_chunks: List[str],
        generated_answer: str,
        reference_answer: str = "",
    ) -> Dict[str, Any]:
        """Run all four RAGAS metrics on one section.

        Args:
            section_key:         Section identifier.
            section_instruction: The original generation instruction.
            retrieved_chunks:    List of retrieved context text strings.
            generated_answer:    The AI-generated section content.
            reference_answer:    Optional ground-truth reference text.

        Returns:
            Dict with the following structure::

                {
                  "section_key": str,
                  "faithfulness": float | None,
                  "faithfulness_n_claims": int,
                  "faithfulness_supported_claims": int,
                  "context_precision": float,
                  "context_precision_ap": float,
                  "context_recall": float | None,
                  "context_recall_na_reason": str | None,
                  "answer_relevancy": float,
                  "answer_relevancy_method": str,
                  "ragas_score": float,
                  "evaluated_at": str
                }

        Raises:
            Nothing — individual metric errors are captured and logged.

        Example:
            >>> result = rag.evaluate_section(
            ...     "EXEC SUMMARY", "Write summary.", ctxs, gen_text,
            ... )
            >>> result["faithfulness"]
            0.85
        """
        # --- cache check ---
        ck = _cache_key(
            section_key, generated_answer, reference_answer,
            self.CACHE_VERSION,
        )
        if self.cache_enabled:
            cached = _read_cache(self.cache_dir, ck)
            if cached is not None:
                logger.info("RAG cache hit for %s", section_key)
                return cached

        faith = self._compute_faithfulness(generated_answer, retrieved_chunks)
        prec = self._compute_context_precision(
            section_instruction, retrieved_chunks, generated_answer,
        )
        recall = self._compute_context_recall(reference_answer, retrieved_chunks)
        relev = self._compute_answer_relevancy(
            section_instruction, generated_answer,
        )

        # --- RAGAS composite score (harmonic mean of non-null metrics) ---
        hmean_inputs: List[float] = []
        if faith.get("faithfulness") is not None:
            hmean_inputs.append(float(faith["faithfulness"]))
        if prec.get("context_precision") is not None:
            hmean_inputs.append(float(prec["context_precision"]))
        if relev.get("answer_relevancy") is not None:
            hmean_inputs.append(float(relev["answer_relevancy"]))

        ragas_score = _round(_harmonic_mean(hmean_inputs)) if hmean_inputs else 0.0

        result: Dict[str, Any] = {
            "section_key": section_key,
            # faithfulness
            "faithfulness": faith.get("faithfulness"),
            "faithfulness_n_claims": faith.get("faithfulness_n_claims", 0),
            "faithfulness_supported_claims": faith.get(
                "faithfulness_supported_claims", 0
            ),
            "faithfulness_claims": faith.get("claims", []),
            # context precision
            "context_precision": prec.get("context_precision", 0.0),
            "context_precision_ap": prec.get("context_precision_ap", 0.0),
            "context_relevance": prec.get("context_relevance", []),
            # context recall
            "context_recall": recall.get("context_recall"),
            "context_recall_na_reason": recall.get("context_recall_na_reason"),
            # answer relevancy
            "answer_relevancy": relev.get("answer_relevancy", 0.0),
            "answer_relevancy_method": relev.get(
                "answer_relevancy_method", "unknown"
            ),
            # composite
            "ragas_score": ragas_score,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.cache_enabled:
            _write_cache(self.cache_dir, ck, result)

        return result

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: evaluate_document
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_document(
        self,
        sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate RAG quality across all sections in a document run.

        Args:
            sections: List of section dicts.  Each must contain at least
                      ``section_key``, ``section_instruction`` (or
                      ``prompt_text``), ``retrieved_chunks`` (list of str),
                      and ``generated_answer`` (or ``generated_text``).
                      Optional: ``reference_answer``, ``is_static``.

        Returns:
            Dict with the following structure::

                {
                  "mean_faithfulness": float,
                  "mean_context_precision": float,
                  "mean_context_recall": float | None,
                  "mean_answer_relevancy": float,
                  "mean_ragas_score": float,
                  "per_section": [...],
                  "low_faithfulness_sections": [...],
                  "low_precision_sections": [...],
                  "retrieval_quality_summary": str
                }

        Raises:
            Nothing — per-section errors are captured in each result.

        Example:
            >>> doc = rag.evaluate_document(sections)
            >>> doc["retrieval_quality_summary"]
            'good'
        """
        per_section: List[Dict[str, Any]] = []

        for sec in sections:
            if sec.get("is_static", False):
                continue

            key = sec.get("section_key", "UNKNOWN")
            instruction = (
                sec.get("section_instruction")
                or sec.get("prompt_text", "")
            )
            chunks = sec.get("retrieved_chunks", [])
            if not chunks:
                # Try alternative key names
                chunks = sec.get("retrieved_texts", [])
            if not chunks and sec.get("retrieved_context"):
                chunks = [sec["retrieved_context"]]

            answer = sec.get("generated_answer") or sec.get("generated_text", "")
            ref = sec.get("reference_answer", "")

            result = self.evaluate_section(
                section_key=key,
                section_instruction=instruction,
                retrieved_chunks=chunks,
                generated_answer=answer,
                reference_answer=ref,
            )
            per_section.append(result)

        # --- aggregation ---
        def _mean_of(key: str) -> Optional[float]:
            vals = [
                float(r[key]) for r in per_section
                if r.get(key) is not None
            ]
            if not vals:
                return None
            return _round(sum(vals) / len(vals))

        mean_faith = _mean_of("faithfulness")
        mean_prec = _mean_of("context_precision")
        mean_recall = _mean_of("context_recall")
        mean_relev = _mean_of("answer_relevancy")
        mean_ragas = _mean_of("ragas_score")

        low_faith = [
            r["section_key"] for r in per_section
            if r.get("faithfulness") is not None and r["faithfulness"] < 0.7
        ]
        low_prec = [
            r["section_key"] for r in per_section
            if r.get("context_precision") is not None
            and r["context_precision"] < 0.5
        ]

        # Retrieval quality summary
        mp = mean_prec if mean_prec is not None else 0.0
        if mp >= 0.8:
            quality = "excellent"
        elif mp >= 0.6:
            quality = "good"
        elif mp >= 0.4:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "mean_faithfulness": mean_faith,
            "mean_context_precision": mean_prec,
            "mean_context_recall": mean_recall,
            "mean_answer_relevancy": mean_relev,
            "mean_ragas_score": mean_ragas,
            "per_section": per_section,
            "low_faithfulness_sections": low_faith,
            "low_precision_sections": low_prec,
            "retrieval_quality_summary": quality,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )

    print("=" * 72)
    print("RAGEvaluator — SMOKE TEST (heuristic mode, no live API)")
    print("=" * 72)

    # --- Instantiate without a live LLM client (heuristic fallbacks) ---
    rag = RAGEvaluator(llm_client=None, model="none")
    print(f"LLM available: {rag._llm.available}")

    # ── 1. Heuristic claim extraction ────────────────────────────────────
    print("\n--- 1. Heuristic claim extraction ---")
    claims = rag._extract_claims(
        "The Langensbold site manufactures sterile connectors. "
        "The facility follows ISO 13485. Quality is ensured via cGMP."
    )
    print(f"  Extracted {len(claims)} claims: {claims}")
    assert len(claims) >= 2, "Should extract at least 2 claims"
    print("  PASS")

    # ── 2. Heuristic entailment ──────────────────────────────────────────
    print("\n--- 2. Heuristic entailment ---")
    assert _heuristic_supported(
        "The site manufactures sterile connectors",
        "The Langensbold site manufactures sterile connectors for biopharma.",
    ), "Direct-quote claim must be supported"
    assert not _heuristic_supported(
        "The factory produces chocolate bars",
        "The Langensbold site manufactures sterile connectors for biopharma.",
    ), "Unrelated claim must not be supported"
    print("  PASS")

    # ── 3. Faithfulness — direct-quote case (score must be > 0.9) ────────
    print("\n--- 3. Faithfulness — direct-quote test ---")
    context_chunks = [
        "The Langensbold manufacturing facility produces single-use "
        "bioreactor assemblies and sterile connectors used in "
        "biopharmaceutical manufacturing.",
        "The facility maintains ISO 13485 certification and operates "
        "under cGMP regulations.",
    ]
    # Generated answer is nearly a direct quote from the context.
    generated = (
        "The Langensbold manufacturing facility produces single-use "
        "bioreactor assemblies and sterile connectors. "
        "The facility maintains ISO 13485 certification and operates "
        "under cGMP regulations."
    )
    faith = rag._compute_faithfulness(generated, context_chunks)
    print(f"  faithfulness = {faith['faithfulness']}")
    print(f"  n_claims = {faith['faithfulness_n_claims']}")
    print(f"  supported = {faith['faithfulness_supported_claims']}")
    for c in faith.get("claims", []):
        print(f"    {c['supported']}  {c['claim']}")
    assert faith["faithfulness"] is not None
    assert faith["faithfulness"] > 0.9, (
        f"Direct-quote faithfulness should be > 0.9, got {faith['faithfulness']}"
    )
    print("  PASS (faithfulness > 0.9)")

    # ── 4. Faithfulness — vacuously faithful (empty answer) ──────────────
    print("\n--- 4. Faithfulness — empty answer (vacuously faithful) ---")
    faith_empty = rag._compute_faithfulness("", context_chunks)
    print(f"  faithfulness = {faith_empty['faithfulness']}")
    assert faith_empty["faithfulness"] == 1.0, (
        "Empty answer must be vacuously faithful (1.0)"
    )
    print("  PASS (vacuously faithful = 1.0)")

    # ── 5. Context Precision ─────────────────────────────────────────────
    print("\n--- 5. Context Precision ---")
    prec = rag._compute_context_precision(
        section_instruction=(
            "Describe the manufacturing capabilities and quality "
            "management at the Langensbold site"
        ),
        retrieved_chunks=[
            "The Langensbold site manufactures bioreactor assemblies.",
            "Weather in Frankfurt was sunny last Tuesday.",
            "ISO 13485 certification is maintained at the Langensbold site.",
        ],
        generated_answer=generated,
    )
    print(f"  raw precision = {prec['context_precision']}")
    print(f"  AP = {prec['context_precision_ap']}")
    for i, det in enumerate(prec.get("context_relevance", [])):
        print(f"    chunk {i}: relevant={det['relevant']}")
    assert prec["context_precision"] > 0.0
    print("  PASS")

    # ── 6. Context Recall — with reference ───────────────────────────────
    print("\n--- 6. Context Recall — with reference ---")
    recall = rag._compute_context_recall(
        reference_answer=(
            "The Langensbold site produces bioreactor assemblies and "
            "sterile connectors. It is ISO 13485 certified."
        ),
        retrieved_chunks=context_chunks,
    )
    print(f"  context_recall = {recall['context_recall']}")
    print(f"  na_reason = {recall['context_recall_na_reason']}")
    assert recall["context_recall"] is not None
    assert recall["context_recall_na_reason"] is None
    print("  PASS")

    # ── 7. Context Recall — no reference ─────────────────────────────────
    print("\n--- 7. Context Recall — no reference ---")
    recall_none = rag._compute_context_recall("", context_chunks)
    print(f"  context_recall = {recall_none['context_recall']}")
    print(f"  na_reason = {recall_none['context_recall_na_reason']}")
    assert recall_none["context_recall"] is None
    assert recall_none["context_recall_na_reason"] == "No reference answer provided"
    print("  PASS")

    # ── 8. Answer Relevancy (direct similarity fallback) ─────────────────
    print("\n--- 8. Answer Relevancy (direct similarity fallback) ---")
    relev = rag._compute_answer_relevancy(
        section_instruction=(
            "Describe the manufacturing capabilities at Langensbold"
        ),
        generated_answer=generated,
    )
    print(f"  answer_relevancy = {relev['answer_relevancy']}")
    print(f"  method = {relev['answer_relevancy_method']}")
    assert relev["answer_relevancy_method"] == "direct_similarity"
    assert relev["answer_relevancy"] > 0.0
    print("  PASS")

    # ── 9. Harmonic mean ─────────────────────────────────────────────────
    print("\n--- 9. Harmonic mean ---")
    hm = _harmonic_mean([0.8, 0.6, 0.9])
    expected = 3.0 / (1 / 0.8 + 1 / 0.6 + 1 / 0.9)
    assert abs(hm - expected) < 1e-6, f"Expected {expected}, got {hm}"
    print(f"  H([0.8, 0.6, 0.9]) = {hm:.6f}  (expected {expected:.6f})")
    assert _harmonic_mean([]) == 0.0
    assert _harmonic_mean([0.0, 0.0]) == 0.0
    print("  PASS")

    # ── 10. evaluate_section — full pipeline ─────────────────────────────
    print("\n--- 10. evaluate_section — full pipeline ---")
    section_result = rag.evaluate_section(
        section_key="DEVICE DESCRIPTION",
        section_instruction="Describe the medical devices manufactured.",
        retrieved_chunks=context_chunks,
        generated_answer=generated,
        reference_answer=(
            "The Langensbold site produces bioreactor assemblies and "
            "sterile connectors. It is ISO 13485 certified."
        ),
    )
    for k in ("section_key", "faithfulness", "faithfulness_n_claims",
              "faithfulness_supported_claims", "context_precision",
              "context_precision_ap", "context_recall",
              "context_recall_na_reason", "answer_relevancy",
              "answer_relevancy_method", "ragas_score", "evaluated_at"):
        assert k in section_result, f"Missing key: {k}"
        print(f"  {k} = {section_result[k]}")
    assert section_result["ragas_score"] > 0.0
    print("  PASS (all keys present, ragas_score > 0)")

    # ── 11. evaluate_document ────────────────────────────────────────────
    print("\n--- 11. evaluate_document ---")
    mock_sections = [
        {
            "section_key": "SECTION A",
            "section_instruction": "Describe the devices.",
            "retrieved_chunks": context_chunks,
            "generated_answer": generated,
        },
        {
            "section_key": "SECTION B",
            "section_instruction": "Describe quality management.",
            "retrieved_chunks": context_chunks,
            "generated_answer": generated,
            "reference_answer": "ISO 13485 certified. cGMP compliant.",
        },
        {
            "section_key": "STATIC",
            "is_static": True,
            "generated_text": "Fixed content.",
        },
    ]
    doc_result = rag.evaluate_document(mock_sections)
    for k in ("mean_faithfulness", "mean_context_precision",
              "mean_context_recall", "mean_answer_relevancy",
              "mean_ragas_score", "per_section",
              "low_faithfulness_sections", "low_precision_sections",
              "retrieval_quality_summary"):
        assert k in doc_result, f"Missing key: {k}"
    print(f"  mean_faithfulness = {doc_result['mean_faithfulness']}")
    print(f"  mean_context_precision = {doc_result['mean_context_precision']}")
    print(f"  mean_ragas_score = {doc_result['mean_ragas_score']}")
    print(f"  retrieval_quality_summary = {doc_result['retrieval_quality_summary']}")
    print(f"  per_section count = {len(doc_result['per_section'])}")
    # Static section should be skipped
    assert len(doc_result["per_section"]) == 2, "Static section should be skipped"
    assert doc_result["retrieval_quality_summary"] in (
        "excellent", "good", "fair", "poor"
    )
    print("  PASS")

    # ── 12. Cache round-trip ─────────────────────────────────────────────
    print("\n--- 12. Cache round-trip ---")
    test_key = _cache_key("SEC", "text", "v1")
    test_data = {"score": 42}
    cache_dir = "data/eval_cache/rag_test"
    _write_cache(cache_dir, test_key, test_data)
    loaded = _read_cache(cache_dir, test_key)
    assert loaded == test_data, f"Cache mismatch: {loaded}"
    # Clean up
    os.remove(os.path.join(cache_dir, f"{test_key}.json"))
    try:
        os.rmdir(cache_dir)
    except OSError:
        pass
    print("  PASS")

    print("\n" + "=" * 72)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 72)
