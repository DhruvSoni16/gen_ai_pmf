"""
Opik-Inspired Evaluation Metrics (no opik package required)
============================================================

Implements Opik's (Comet) hallucination and answer-relevance metrics
from scratch using the existing Azure OpenAI client.

Opik's approach differs from DeepEval in one key way:
  - DeepEval: extract claims → check each claim one-by-one (binary per claim)
  - Opik:     ask the LLM for a DIRECT continuous score (0-1) for the whole
              section in a single call — faster and captures nuance better.

Metrics implemented:
  1. **HallucinationScorer**  — 0 (no hallucination) to 1 (complete hallucination)
  2. **AnswerRelevanceScorer** — 0 (completely irrelevant) to 1 (perfectly relevant)
  3. **RegulatoryToneScorer** — 0 (informal/inappropriate) to 1 (formal regulatory)
     (PMF-specific, not in vanilla Opik — added for regulatory domain value)

Part of the Healthark GenAI Evaluation Framework (Initiative 4).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


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
    try:
        with open(os.path.join(cache_dir, f"{key}.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        logger.warning("Cache write failed: %s", exc)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def _parse_json(raw: str) -> Any:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for s, e in [("{", "}"), ("[", "]")]:
            i = cleaned.find(s)
            if i != -1:
                j = cleaned.rfind(e)
                if j > i:
                    try:
                        return json.loads(cleaned[i: j + 1])
                    except json.JSONDecodeError:
                        continue
        raise


# ═══════════════════════════════════════════════════════════════════════════
# LLM CALLER (reuses same client pattern as the rest of the framework)
# ═══════════════════════════════════════════════════════════════════════════

class _LLMCaller:
    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model
        mod = type(client).__module__ or "" if client else ""
        self._provider = (
            "anthropic" if "anthropic" in mod else
            "azure_openai" if "openai" in mod else
            "none" if client is None else "unknown"
        )

    @property
    def available(self) -> bool:
        return self._client is not None and self._provider != "none"

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        if not self.available:
            raise RuntimeError("No LLM client available.")
        if self._provider == "anthropic":
            resp = self._client.messages.create(
                model=self._model, max_tokens=max_tokens, temperature=0.0,
                system=system, messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text if resp.content else ""
        resp = self._client.chat.completions.create(
            model=self._model, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════════════
# OPIK-STYLE PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

# --- Hallucination (Opik methodology: direct continuous score) ---
_SYS_HALLUCINATION = (
    "You are an expert fact-checker evaluating AI-generated regulatory documents. "
    "Your task is to identify hallucinations — content that is not supported by or "
    "directly contradicts the provided source context. "
    "Return ONLY valid JSON, no markdown."
)

_USR_HALLUCINATION = """\
Evaluate the HALLUCINATION LEVEL of the following AI-generated output.

SOURCE CONTEXT (what the AI had access to):
{context}

AI-GENERATED OUTPUT (evaluate this):
{output}

DEFINITION:
- Hallucination = any factual claim in the output that is NOT supported by or
  directly contradicts the source context.
- Do NOT penalise for regulatory boilerplate language or standard PMF formatting —
  only penalise for invented factual claims (names, numbers, certifications, dates,
  site-specific details not in the context).

SCORING:
  0.0 = No hallucinations — all factual claims are grounded in the context
  0.25 = Minor hallucinations — 1-2 small unsupported claims
  0.5 = Moderate — several unsupported claims
  0.75 = Significant — majority of specific facts are not in context
  1.0 = Complete hallucination — virtually nothing is grounded

Respond with ONLY:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>", "examples": ["<example unsupported claim if any>"]}}
"""

# --- Answer Relevance (Opik methodology: direct score) ---
_SYS_ANSWER_RELEVANCE = (
    "You are an expert evaluator assessing whether an AI response addresses the "
    "given instruction. Return ONLY valid JSON, no markdown."
)

_USR_ANSWER_RELEVANCE = """\
Evaluate how RELEVANT the AI-generated output is to the original instruction.

INSTRUCTION / SECTION REQUIREMENT:
{instruction}

AI-GENERATED OUTPUT:
{output}

SCORING:
  1.0 = Perfectly relevant — directly and completely addresses the instruction
  0.75 = Mostly relevant — addresses the instruction with minor gaps
  0.5 = Partially relevant — addresses some aspects but misses key requirements
  0.25 = Mostly irrelevant — barely addresses the instruction
  0.0 = Completely irrelevant — does not address the instruction at all

Respond with ONLY:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
"""

# --- Regulatory Tone (PMF-specific, not in vanilla Opik) ---
_SYS_REG_TONE = (
    "You are a regulatory affairs expert evaluating whether AI-generated text "
    "uses language appropriate for a Plant Master File (PMF) regulatory submission "
    "under EU GMP Annex 4 / ICH Q10. Return ONLY valid JSON."
)

_USR_REG_TONE = """\
Evaluate the REGULATORY TONE AND LANGUAGE QUALITY of this AI-generated PMF section.

SECTION KEY: {section_key}

AI-GENERATED OUTPUT:
{output}

SCORING CRITERIA:
  1.0 = Exemplary — formal passive construction, precise regulatory terminology,
        ICH Q10/EU GMP language conventions, no informal phrases
  0.75 = Good — mostly formal, minor informal phrases or non-standard terms
  0.5 = Acceptable — mix of formal/informal, would need moderate revision
  0.25 = Poor — predominantly informal language, non-regulatory style
  0.0 = Inappropriate — completely unsuitable for a regulatory submission

Respond with ONLY:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>", "issues": ["<example issue if any>"]}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# SCORERS
# ═══════════════════════════════════════════════════════════════════════════

class OpikStyleScorer:
    """Opik-inspired evaluation metrics using direct LLM scoring.

    All three scorers use a single LLM call per section (vs DeepEval's
    multiple calls per claim), making them faster and giving a holistic
    score that captures nuance a binary claim-checker misses.

    Args:
        llm_client:    AzureOpenAI or Anthropic client instance.
        model:         Model deployment name.
        cache_enabled: Cache results by SHA-256 hash.
        cache_dir:     Cache directory path.
    """

    CACHE_VERSION = "opik_v1.0"

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4o",
        cache_enabled: bool = True,
        cache_dir: str = "data/eval_cache/opik",
    ):
        self._llm = _LLMCaller(llm_client, model)
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir

    # ─────────────────────────────────────────────────────────────────────
    # Hallucination Score (Opik methodology)
    # ─────────────────────────────────────────────────────────────────────

    def score_hallucination(
        self,
        output: str,
        context: str,
        section_key: str = "",
    ) -> Dict[str, Any]:
        """Score hallucination level (0 = none, 1 = complete).

        Lower is better. Opik's direct-score approach gives a continuous
        value that's more informative than DeepEval's binary per-claim check.
        """
        null_result = {
            "hallucination_score": None,
            "hallucination_reason": "LLM unavailable",
            "hallucination_examples": [],
            "framework": "opik_style",
        }

        if not self._llm.available:
            return null_result

        if not output.strip() or not context.strip():
            return {**null_result, "hallucination_score": 0.0, "hallucination_reason": "empty input"}

        ck = _cache_key(section_key, output[:500], "hallucination", self.CACHE_VERSION)
        if self.cache_enabled:
            cached = _read_cache(self.cache_dir, ck)
            if cached:
                return cached

        prompt = _USR_HALLUCINATION.format(
            context=context[:6000],
            output=output[:4000],
        )
        try:
            raw = self._llm.chat(_SYS_HALLUCINATION, prompt)
            data = _parse_json(raw)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            result = {
                "hallucination_score": round(score, 4),
                "hallucination_reason": data.get("reason", ""),
                "hallucination_examples": data.get("examples", []),
                "framework": "opik_style",
            }
            if self.cache_enabled:
                _write_cache(self.cache_dir, ck, result)
            return result
        except Exception as exc:
            logger.warning("Hallucination scoring failed for '%s': %s", section_key, exc)
            return null_result

    # ─────────────────────────────────────────────────────────────────────
    # Answer Relevance Score (Opik methodology)
    # ─────────────────────────────────────────────────────────────────────

    def score_answer_relevance(
        self,
        output: str,
        instruction: str,
        section_key: str = "",
    ) -> Dict[str, Any]:
        """Score how relevant the output is to the instruction (0–1).

        Higher is better. Opik's direct-score approach vs DeepEval's
        reverse-question cosine similarity — gives a more holistic view.
        """
        null_result = {
            "answer_relevance_score": None,
            "answer_relevance_reason": "LLM unavailable",
            "framework": "opik_style",
        }

        if not self._llm.available:
            return null_result

        if not output.strip() or not instruction.strip():
            return {**null_result, "answer_relevance_score": 0.0}

        ck = _cache_key(section_key, output[:500], "answer_relevance", self.CACHE_VERSION)
        if self.cache_enabled:
            cached = _read_cache(self.cache_dir, ck)
            if cached:
                return cached

        prompt = _USR_ANSWER_RELEVANCE.format(
            instruction=instruction[:2000],
            output=output[:4000],
        )
        try:
            raw = self._llm.chat(_SYS_ANSWER_RELEVANCE, prompt)
            data = _parse_json(raw)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            result = {
                "answer_relevance_score": round(score, 4),
                "answer_relevance_reason": data.get("reason", ""),
                "framework": "opik_style",
            }
            if self.cache_enabled:
                _write_cache(self.cache_dir, ck, result)
            return result
        except Exception as exc:
            logger.warning("Answer relevance scoring failed for '%s': %s", section_key, exc)
            return null_result

    # ─────────────────────────────────────────────────────────────────────
    # Regulatory Tone Score (PMF-domain-specific)
    # ─────────────────────────────────────────────────────────────────────

    def score_regulatory_tone(
        self,
        output: str,
        section_key: str = "",
    ) -> Dict[str, Any]:
        """Score regulatory language quality (0 = inappropriate, 1 = exemplary).

        PMF-specific metric not in vanilla Opik. Checks whether the language
        is appropriate for EU GMP / ICH Q10 regulatory submission.
        """
        null_result = {
            "regulatory_tone_score": None,
            "regulatory_tone_reason": "LLM unavailable",
            "regulatory_tone_issues": [],
            "framework": "opik_style",
        }

        if not self._llm.available:
            return null_result

        if not output.strip():
            return {**null_result, "regulatory_tone_score": 0.0}

        ck = _cache_key(section_key, output[:500], "reg_tone", self.CACHE_VERSION)
        if self.cache_enabled:
            cached = _read_cache(self.cache_dir, ck)
            if cached:
                return cached

        prompt = _USR_REG_TONE.format(section_key=section_key, output=output[:4000])
        try:
            raw = self._llm.chat(_SYS_REG_TONE, prompt)
            data = _parse_json(raw)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            result = {
                "regulatory_tone_score": round(score, 4),
                "regulatory_tone_reason": data.get("reason", ""),
                "regulatory_tone_issues": data.get("issues", []),
                "framework": "opik_style",
            }
            if self.cache_enabled:
                _write_cache(self.cache_dir, ck, result)
            return result
        except Exception as exc:
            logger.warning("Regulatory tone scoring failed for '%s': %s", section_key, exc)
            return null_result

    # ─────────────────────────────────────────────────────────────────────
    # Evaluate Section (all three in one call)
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_section(
        self,
        section_key: str,
        output: str,
        context: str,
        instruction: str,
    ) -> Dict[str, Any]:
        """Run all three Opik-style metrics on one section.

        Returns a merged dict suitable for storage in run_artifacts.
        """
        hall = self.score_hallucination(output, context, section_key)
        relev = self.score_answer_relevance(output, instruction, section_key)
        tone = self.score_regulatory_tone(output, section_key)

        # Opik-style composite: lower hallucination is better, so invert it
        h = hall.get("hallucination_score")
        r = relev.get("answer_relevance_score")
        t = tone.get("regulatory_tone_score")

        scores = [v for v in [
            (1.0 - h) if h is not None else None,   # inverted hallucination
            r,
            t,
        ] if v is not None]

        opik_composite = round(sum(scores) / len(scores), 4) if scores else None

        return {
            "section_key": section_key,
            **hall,
            **relev,
            **tone,
            "opik_composite": opik_composite,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
