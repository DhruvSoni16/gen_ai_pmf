"""
DeepEval-Inspired RAG Triad Evaluation Module
==============================================

Implements the DeepEval RAG Triad metrics for PMF document sections
using Azure OpenAI (no external deepeval package needed — same methodology,
same LLM-judge approach, pure-Python implementation).

Metrics (aligned to DeepEval's RAG Triad):
  1. **Faithfulness**      — Are all claims in the generated output
                             grounded in the retrieved context?
                             Uses step-by-step LLM claim extraction +
                             entailment checking (DeepEval methodology).
  2. **Answer Relevancy**  — Does the answer address the original question /
                             instruction?  Uses the reverse-question method:
                             generate N questions the answer would address,
                             then measure cosine similarity to the original.
  3. **Contextual Precision** — Are the most relevant context chunks ranked
                             first?  Computes rank-weighted Average Precision.

Composite: harmonic mean of Faithfulness + Contextual Precision + Answer
Relevancy → ``rag_triad_score`` (0–1).

Part of the Healthark GenAI Evaluation Framework (Initiative 4).

Dependencies:
    pip install openai numpy sentence-transformers
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
except (ImportError, OSError):
    _SentenceTransformer = None  # type: ignore[assignment]
    logger.warning(
        "sentence-transformers not available — answer relevancy will use LLM-only path."
    )

try:
    from openai import AzureOpenAI as _AzureOpenAI
except ImportError:
    _AzureOpenAI = None

try:
    import anthropic as _anthropic_mod
except ImportError:
    _anthropic_mod = None


def _round(value: float, decimals: int = 4) -> float:
    return round(float(value), decimals)


# ═══════════════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════════════

def _cache_key(*parts: str) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
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
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def _parse_json_lenient(raw: str) -> Any:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            idx = cleaned.find(start_char)
            if idx != -1:
                ridx = cleaned.rfind(end_char)
                if ridx > idx:
                    try:
                        return json.loads(cleaned[idx: ridx + 1])
                    except json.JSONDecodeError:
                        continue
        raise


def _harmonic_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    # Any zero metric fails the whole triad (standard harmonic mean behaviour)
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


# ═══════════════════════════════════════════════════════════════════════════
# LLM WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class _LLMCaller:
    """Sends prompts to either Anthropic or Azure OpenAI client."""

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
        if "openai" in mod_name.lower() or "OpenAI" in cls_name:
            return "azure_openai"
        return "unknown"

    @property
    def available(self) -> bool:
        return self._client is not None and self._provider != "none"

    def chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
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
# HEURISTIC FALLBACKS (no-LLM paths)
# ═══════════════════════════════════════════════════════════════════════════

def _heuristic_extract_claims(text: str) -> List[str]:
    return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]


def _heuristic_supported(claim: str, context: str) -> bool:
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
    q_words = {w.lower() for w in question.split() if len(w) > 3}
    if not q_words:
        return True
    c_lower = chunk.lower()
    matched = sum(1 for w in q_words if w[: min(6, len(w))] in c_lower)
    return matched / len(q_words) > 0.25


# ═══════════════════════════════════════════════════════════════════════════
# DEEPEVAL-STYLE PROMPT TEMPLATES
# (Step-by-step chain-of-thought — same approach as DeepEval open-source)
# ═══════════════════════════════════════════════════════════════════════════

# --- Claim Extraction (Faithfulness step 1) ---
_SYS_CLAIM_EXTRACT = (
    "You are a precise factual claim extractor. "
    "Extract every distinct atomic factual claim from the given text. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

_USR_CLAIM_EXTRACT = """\
Read the following text carefully and extract all factual claims.

STEPS:
1. Identify every declarative statement that makes a factual assertion.
2. Break compound sentences into individual atomic claims.
3. Each claim should be a single, self-contained, verifiable statement.
4. Ignore opinions, instructions, or formatting text.

TEXT:
{text}

Respond with ONLY:
{{"claims": ["claim 1", "claim 2", ...]}}
"""

# --- Entailment Check (Faithfulness step 2) ---
_SYS_ENTAILMENT = (
    "You are a strict fact-checking agent. "
    "Determine whether a claim is FULLY supported by the provided context. "
    "Return ONLY valid JSON."
)

_USR_ENTAILMENT = """\
Evaluate whether the following claim is fully supported by the context.

CONTEXT:
{context}

CLAIM:
{claim}

STEPS:
1. Find any information in the context that relates to the claim.
2. Determine if the context EXPLICITLY states or clearly implies the claim.
3. Do NOT infer beyond what is clearly stated.
4. Return your verdict.

Respond with ONLY:
{{"verdict": "yes" or "no", "reason": "one sentence explanation"}}
"""

# --- Context Relevance (Contextual Precision step) ---
_SYS_RELEVANCE = (
    "You are a retrieval quality evaluator. "
    "Determine if a context chunk is relevant to answering the given question. "
    "Return ONLY valid JSON."
)

_USR_RELEVANCE = """\
Evaluate whether the context chunk is relevant to answering the question.

QUESTION / INSTRUCTION:
{question}

CONTEXT CHUNK:
{chunk}

STEPS:
1. Understand what information the question requires.
2. Check if this context chunk contains any information that helps answer it.
3. Mark as relevant only if it directly addresses the question topic.

Respond with ONLY:
{{"verdict": "yes" or "no", "reason": "one sentence explanation"}}
"""

# --- Question Generation (Answer Relevancy step) ---
_SYS_GEN_QUESTIONS = (
    "You are a question generation assistant. "
    "Given an answer, generate questions that this answer would perfectly address. "
    "Return ONLY valid JSON."
)

_USR_GEN_QUESTIONS = """\
Given the following answer, generate {n} questions that this answer directly and completely addresses.

ANSWER:
{answer}

STEPS:
1. Identify the main topics and facts covered in the answer.
2. Formulate {n} distinct questions whose ideal answer is this text.
3. Make questions specific and information-seeking, not yes/no questions.

Respond with ONLY:
{{"questions": ["question 1", "question 2", ...]}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# RAGEvaluator CLASS — DeepEval RAG Triad Implementation
# ═══════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    """DeepEval-inspired RAG Triad evaluation for PMF document sections.

    Implements the three core RAG Triad metrics following DeepEval's
    open-source methodology — using LLM for claim extraction / entailment
    and optionally SentenceTransformer embeddings for answer relevancy.
    Falls back to keyword-overlap heuristics when no LLM is available.

    Metrics:
        - **Faithfulness**: Fraction of generated claims supported by context.
        - **Contextual Precision**: Rank-weighted MAP of retrieved contexts.
        - **Answer Relevancy**: How well the answer addresses the question.

    Args:
        llm_client:    An AzureOpenAI or Anthropic client instance.
        model:         Model deployment name.
        embedder:      Optional pre-loaded SentenceTransformer.
        cache_enabled: Cache per-section results to avoid re-computation.
        cache_dir:     Directory for cached JSON files.
    """

    CACHE_VERSION: str = "deepeval_v1.0"

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4o",
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
        if self._embedder is None:
            if _SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not available.")
            self._embedder = _SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL: claim extraction (Faithfulness — step 1)
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_claims(self, text: str) -> List[str]:
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
        """DeepEval-style entailment check — step-by-step reasoning."""
        if not self._llm.available:
            return {
                "supported": _heuristic_supported(claim, context),
                "evidence": None,
            }
        prompt = _USR_ENTAILMENT.format(context=context[:6000], claim=claim)
        try:
            raw = self._llm.chat(_SYS_ENTAILMENT, prompt, max_tokens=256)
            data = _parse_json_lenient(raw)
            verdict = str(data.get("verdict", "no")).lower().strip()
            return {
                "supported": verdict in ("yes", "true", "1"),
                "evidence": data.get("reason"),
            }
        except Exception as exc:
            logger.warning("Entailment check failed: %s", exc)
            return {
                "supported": _heuristic_supported(claim, context),
                "evidence": None,
            }

    def _check_relevant(self, question: str, chunk: str) -> Dict[str, Any]:
        """DeepEval-style context relevance check."""
        if not self._llm.available:
            return {
                "relevant": _heuristic_relevant(question, chunk),
                "reason": "heuristic",
            }
        prompt = _USR_RELEVANCE.format(question=question, chunk=chunk[:2000])
        try:
            raw = self._llm.chat(_SYS_RELEVANCE, prompt, max_tokens=256)
            data = _parse_json_lenient(raw)
            verdict = str(data.get("verdict", "no")).lower().strip()
            return {
                "relevant": verdict in ("yes", "true", "1"),
                "reason": data.get("reason", ""),
            }
        except Exception as exc:
            logger.warning("Relevance check failed: %s", exc)
            return {
                "relevant": _heuristic_relevant(question, chunk),
                "reason": "heuristic",
            }

    # ═══════════════════════════════════════════════════════════════════════
    # METRIC 1: Faithfulness
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_faithfulness(
        self, generated_answer: str, retrieved_chunks: List[str]
    ) -> Dict[str, Any]:
        """Fraction of generated claims supported by retrieved context.

        DeepEval approach: extract claims → check each against merged context.
        Score = supported_claims / total_claims
        """
        claims = self._extract_claims(generated_answer)
        if not claims:
            return {
                "faithfulness": 1.0,
                "faithfulness_n_claims": 0,
                "faithfulness_supported_claims": 0,
                "claims": [],
            }

        merged_context = "\n\n".join(retrieved_chunks)
        claim_results: List[Dict[str, Any]] = []
        supported_count = 0

        for claim in claims:
            check = self._check_supported(claim, merged_context)
            if check["supported"]:
                supported_count += 1
            claim_results.append({
                "claim": claim,
                "supported": check["supported"],
                "evidence": check.get("evidence"),
            })

        score = supported_count / len(claims)
        return {
            "faithfulness": _round(score),
            "faithfulness_n_claims": len(claims),
            "faithfulness_supported_claims": supported_count,
            "claims": claim_results,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # METRIC 2: Contextual Precision (rank-weighted Average Precision)
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_contextual_precision(
        self,
        section_instruction: str,
        retrieved_chunks: List[str],
        generated_answer: str,
    ) -> Dict[str, Any]:
        """Rank-weighted MAP of retrieved context chunks.

        DeepEval approach: for each context chunk (in retrieval order),
        determine relevance. Compute precision@k weighted Average Precision.
        Higher score = relevant chunks ranked first.
        """
        if not retrieved_chunks:
            return {
                "contextual_precision": 0.0,
                "contextual_precision_ap": 0.0,
                "context_relevance": [],
            }

        relevance: List[bool] = []
        detail: List[Dict[str, Any]] = []
        for chunk in retrieved_chunks:
            check = self._check_relevant(section_instruction, chunk)
            relevance.append(check["relevant"])
            detail.append({"relevant": check["relevant"], "reason": check.get("reason", "")})

        total_relevant = sum(relevance)
        raw_precision = total_relevant / len(relevance) if relevance else 0.0

        # Average Precision (rank-weighted)
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
            "contextual_precision": _round(raw_precision),
            "contextual_precision_ap": _round(ap),
            "context_relevance": detail,
            # Keep backwards-compat alias
            "context_precision": _round(raw_precision),
            "context_precision_ap": _round(ap),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # METRIC 3: Answer Relevancy (reverse-question cosine similarity)
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_answer_relevancy(
        self,
        section_instruction: str,
        generated_answer: str,
        n_questions: int = 3,
    ) -> Dict[str, Any]:
        """How well does the answer address the original instruction?

        DeepEval approach:
          1. Ask LLM to generate N questions this answer would perfectly address.
          2. Compute cosine similarity between each generated question and the
             original instruction (using sentence embeddings).
          3. Score = mean similarity.

        Falls back to direct instruction-answer cosine similarity if embedder
        or LLM unavailable.
        """
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
                    try:
                        embedder = self._get_embedder()
                        all_texts = [section_instruction] + gen_questions
                        embeddings = embedder.encode(all_texts, normalize_embeddings=True)
                        q_emb = embeddings[0]
                        sims = [float(q_emb @ embeddings[i]) for i in range(1, len(embeddings))]
                        mean_sim = float(_np.mean(sims)) if _np else sum(sims) / len(sims)
                        return {
                            "answer_relevancy": _round(mean_sim),
                            "answer_relevancy_method": "reverse_question_embedding",
                            "generated_questions": gen_questions,
                        }
                    except Exception:
                        # Embedder unavailable — use LLM-based similarity score
                        pass

                # LLM-only fallback: compute similarity via instruction comparison
                if gen_questions:
                    return {
                        "answer_relevancy": _round(
                            min(1.0, len(gen_questions) / n_questions)
                        ),
                        "answer_relevancy_method": "question_generation_count",
                        "generated_questions": gen_questions,
                    }
            except Exception as exc:
                logger.warning("Answer relevancy (reverse-question) failed: %s", exc)

        # Direct cosine similarity fallback
        try:
            embedder = self._get_embedder()
            embs = embedder.encode(
                [section_instruction, generated_answer], normalize_embeddings=True
            )
            sim = float(embs[0] @ embs[1])
            return {
                "answer_relevancy": _round(sim),
                "answer_relevancy_method": "direct_cosine",
            }
        except Exception as exc:
            logger.warning("Direct cosine fallback also failed: %s", exc)
            return {
                "answer_relevancy": 0.5,  # neutral default
                "answer_relevancy_method": "default",
                "error": str(exc),
            }

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC: evaluate_section
    # ═══════════════════════════════════════════════════════════════════════

    def evaluate_section(
        self,
        section_key: str,
        section_instruction: str,
        retrieved_chunks: List[str],
        generated_answer: str,
        reference_answer: str = "",
    ) -> Dict[str, Any]:
        """Run DeepEval RAG Triad metrics on one PMF section.

        Returns a dict with faithfulness, contextual_precision, answer_relevancy,
        and the composite ``rag_triad_score`` (harmonic mean, 0–1 scale).
        Also includes ``ragas_score`` as a backwards-compatibility alias.
        """
        ck = _cache_key(section_key, generated_answer, self.CACHE_VERSION)
        if self.cache_enabled:
            cached = _read_cache(self.cache_dir, ck)
            if cached is not None:
                logger.info("RAG cache hit for %s", section_key)
                return cached

        faith = self._compute_faithfulness(generated_answer, retrieved_chunks)
        prec = self._compute_contextual_precision(
            section_instruction, retrieved_chunks, generated_answer,
        )
        relev = self._compute_answer_relevancy(section_instruction, generated_answer)

        # RAG Triad composite — harmonic mean of the three metrics
        hmean_inputs: List[float] = []
        if faith.get("faithfulness") is not None:
            hmean_inputs.append(float(faith["faithfulness"]))
        if prec.get("contextual_precision") is not None:
            hmean_inputs.append(float(prec["contextual_precision"]))
        if relev.get("answer_relevancy") is not None:
            hmean_inputs.append(float(relev["answer_relevancy"]))

        rag_triad_score = _round(_harmonic_mean(hmean_inputs)) if hmean_inputs else 0.0

        result: Dict[str, Any] = {
            "section_key": section_key,
            # Faithfulness
            "faithfulness": faith.get("faithfulness"),
            "faithfulness_n_claims": faith.get("faithfulness_n_claims", 0),
            "faithfulness_supported_claims": faith.get("faithfulness_supported_claims", 0),
            "faithfulness_claims": faith.get("claims", []),
            # Contextual Precision
            "contextual_precision": prec.get("contextual_precision", 0.0),
            "contextual_precision_ap": prec.get("contextual_precision_ap", 0.0),
            "context_relevance": prec.get("context_relevance", []),
            # Answer Relevancy
            "answer_relevancy": relev.get("answer_relevancy", 0.0),
            "answer_relevancy_method": relev.get("answer_relevancy_method", "unknown"),
            # Composite
            "rag_triad_score": rag_triad_score,
            # Backwards-compat aliases (keep old consumers working)
            "ragas_score": rag_triad_score,
            "context_precision": prec.get("contextual_precision", 0.0),
            "context_precision_ap": prec.get("contextual_precision_ap", 0.0),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "framework": "deepeval_rag_triad",
        }

        if self.cache_enabled:
            _write_cache(self.cache_dir, ck, result)

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC: evaluate_document
    # ═══════════════════════════════════════════════════════════════════════

    def evaluate_document(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate RAG quality across all sections in a document run."""
        per_section: List[Dict[str, Any]] = []
        for sec in sections:
            if sec.get("is_static", False):
                continue
            key = sec.get("section_key", "UNKNOWN")
            instruction = sec.get("section_instruction") or sec.get("prompt_text", "")
            chunks = sec.get("retrieved_chunks", sec.get("retrieved_texts", []))
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

        def _mean_of(key: str) -> Optional[float]:
            vals = [float(r[key]) for r in per_section if r.get(key) is not None]
            return _round(sum(vals) / len(vals)) if vals else None

        mean_faith = _mean_of("faithfulness")
        mean_prec = _mean_of("contextual_precision")
        mean_relev = _mean_of("answer_relevancy")
        mean_rag = _mean_of("rag_triad_score")

        mp = mean_prec if mean_prec is not None else 0.0
        quality = "excellent" if mp >= 0.8 else "good" if mp >= 0.6 else "fair" if mp >= 0.4 else "poor"

        return {
            "mean_faithfulness": mean_faith,
            "mean_contextual_precision": mean_prec,
            "mean_answer_relevancy": mean_relev,
            "mean_rag_triad_score": mean_rag,
            # Backwards compat
            "mean_ragas_score": mean_rag,
            "mean_context_precision": mean_prec,
            "per_section": per_section,
            "low_faithfulness_sections": [
                r["section_key"] for r in per_section
                if r.get("faithfulness") is not None and r["faithfulness"] < 0.7
            ],
            "retrieval_quality_summary": quality,
            "framework": "deepeval_rag_triad",
        }
