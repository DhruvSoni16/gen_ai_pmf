"""
LLM-as-Judge Evaluation Module — Domain-Specific Rubric Scoring
================================================================

Uses an LLM (Claude claude-sonnet-4-6 preferred, GPT-4o fallback) as a domain
expert evaluator for Plant Master File (PMF) regulatory document sections.

The judge reads the section instruction, retrieved source documents, and
generated output, then scores on a structured 5-criterion rubric capturing
quality dimensions that automated metrics cannot.

Part of the Healthark GenAI Evaluation Framework (Initiative 4).

Dependencies:
    pip install anthropic openai pandas
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe imports — degrade gracefully when a provider SDK is missing.
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore[assignment]
    logger.warning("anthropic SDK not installed — Claude-based judging unavailable.")

try:
    from openai import AzureOpenAI as _AzureOpenAI
except ImportError:
    _AzureOpenAI = None  # type: ignore[assignment,misc]
    logger.warning("openai SDK not installed — Azure OpenAI judging unavailable.")

try:
    import pandas as _pd
except ImportError:
    _pd = None  # type: ignore[assignment]
    logger.warning("pandas not installed — compare_models will be unavailable.")


# ═══════════════════════════════════════════════════════════════════════════
# RUBRIC DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_RUBRIC: Dict[str, Dict[str, Any]] = {
    "factual_accuracy": {
        "weight": 0.30,
        "description": "Factual Accuracy",
        "levels": {
            5: (
                "All factual claims in the output are directly supported by the "
                "retrieved source documents. No information is fabricated."
            ),
            4: (
                "Majority of claims supported; minor inferences that are "
                "reasonable but not explicitly in source documents."
            ),
            3: (
                "Most claims supported but 1-2 factual inaccuracies or "
                "unsupported assertions present."
            ),
            2: (
                "Several factual inaccuracies; significant content not found "
                "in source documents."
            ),
            1: (
                "Output is largely hallucinated or contradicts source documents."
            ),
        },
    },
    "regulatory_language": {
        "weight": 0.25,
        "description": "Regulatory Language Quality",
        "levels": {
            5: (
                "Uses precise regulatory terminology throughout. Formal, passive "
                "construction appropriate for PMF/CTD submissions. Compliant with "
                "ICH Q10 and ISO 13485 language conventions."
            ),
            4: "Mostly formal language; minor lapses in regulatory style.",
            3: "Mix of formal and informal language; some non-standard terminology.",
            2: (
                "Predominantly informal or unclear language; regulatory reviewers "
                "would require substantial revision."
            ),
            1: "Inappropriate language for a regulatory submission.",
        },
    },
    "site_specificity": {
        "weight": 0.20,
        "description": "Site Specificity",
        "levels": {
            5: (
                "Correctly and consistently references the specific manufacturing "
                "site name, location, and any site-specific processes mentioned in "
                "source documents."
            ),
            4: "Site name correctly used; minor missing site-specific detail.",
            3: (
                "Site name present but some site-specific content is generic or "
                "could apply to any site."
            ),
            2: "Site name inconsistently used or mostly generic content.",
            1: "Site name missing or incorrect; content is entirely generic.",
        },
    },
    "completeness": {
        "weight": 0.15,
        "description": "Completeness",
        "levels": {
            5: (
                "All sub-topics required for this section type are addressed. "
                "No significant gaps relative to the section instruction."
            ),
            4: "Minor gaps; most required content present.",
            3: "Some required content missing; section partially complete.",
            2: "Significant content gaps; section substantially incomplete.",
            1: "Section is a stub or nearly empty relative to requirements.",
        },
    },
    "structural_coherence": {
        "weight": 0.10,
        "description": "Structural Coherence",
        "levels": {
            5: (
                "Well-organized, logical flow. Uses appropriate headings, "
                "bullet points, or tables as instructed. Easy to navigate."
            ),
            4: "Good organization; minor structural issues.",
            3: "Acceptable structure; some organizational inconsistencies.",
            2: "Poor structure; difficult to follow or navigate.",
            1: "Unstructured or random ordering of content.",
        },
    },
}

CRITERION_WEIGHTS: Dict[str, float] = {
    name: info["weight"] for name, info in JUDGE_RUBRIC.items()
}

CRITERIA_NAMES: List[str] = list(JUDGE_RUBRIC.keys())


def _format_rubric_for_prompt() -> str:
    """Render the JUDGE_RUBRIC as human-readable text for the LLM prompt.

    Returns:
        Multi-line string presenting each criterion with its weight and
        score-level descriptions.
    """
    lines: List[str] = []
    for i, (name, info) in enumerate(JUDGE_RUBRIC.items(), start=1):
        header = (
            f"CRITERION {i} — {info['description']} "
            f"(weight: {info['weight']:.2f})"
        )
        lines.append(header)
        for level in (5, 4, 3, 2, 1):
            lines.append(f"  Score {level}: {info['levels'][level]}")
        lines.append("")
    return "\n".join(lines)


# Pre-render once at module load.
_RUBRIC_TEXT: str = _format_rubric_for_prompt()


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM / USER PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are an expert regulatory affairs specialist with 15+ years of "
    "experience reviewing Plant Master Files (PMF) and pharmaceutical "
    "regulatory submissions. You evaluate AI-generated PMF sections against "
    "a structured rubric. You are precise, consistent, and always provide "
    "a structured JSON response. You never deviate from the requested JSON "
    "format."
)

_USER_TEMPLATE = """\
## Evaluation Task

You are evaluating one section of an AI-generated Plant Master File (PMF).

## Section Being Evaluated
Section Key: {section_key}

## Section Instruction (what the AI was asked to write)
{section_instruction}

## Retrieved Source Documents (context the AI had access to)
{retrieved_context}

## Site Name
{site_name}

{reference_section}

## Generated Output (what the AI actually wrote — evaluate this)
{generated_output}

## Scoring Rubric
{rubric_text}

## Your Task
Score the Generated Output on each of the 5 criteria above.
Respond with ONLY a valid JSON object in this exact format:

{{
  "scores": {{
    "factual_accuracy": <integer 1-5>,
    "regulatory_language": <integer 1-5>,
    "site_specificity": <integer 1-5>,
    "completeness": <integer 1-5>,
    "structural_coherence": <integer 1-5>
  }},
  "weighted_score": <float, weighted average using defined weights, 0-5 scale>,
  "normalized_score": <float, weighted_score / 5 * 100, giving 0-100 scale>,
  "strengths": ["<specific strength 1>", "<specific strength 2>"],
  "weaknesses": ["<specific weakness 1>", "<specific weakness 2>"],
  "critical_issues": ["<any regulatory compliance issue>"],
  "improvement_suggestions": ["<actionable suggestion 1>"],
  "judge_confidence": <float 0-1, your confidence in this evaluation>,
  "evaluation_notes": "<one paragraph of free-text assessment>"
}}
"""

_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous response was not valid JSON. Respond with "
    "ONLY the JSON object, no markdown, no explanation."
)


# ═══════════════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════════════


def _cache_key(section_key: str, generated_text: str, rubric_version: str) -> str:
    """Compute a SHA-256 cache key from the evaluation inputs.

    Args:
        section_key:    Section identifier.
        generated_text: The text being evaluated.
        rubric_version: Rubric version string for cache invalidation.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    payload = f"{section_key}|{generated_text}|{rubric_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    """Read a cached judge result if it exists.

    Args:
        cache_dir: Directory containing cache JSON files.
        key:       SHA-256 cache key.

    Returns:
        Parsed dict if cache hit, else None.
    """
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
    """Persist a judge result to the cache.

    Args:
        cache_dir: Directory for cache JSON files.
        key:       SHA-256 cache key.
        data:      Judge result dict to cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        logger.warning("Failed to write judge cache %s: %s", path, exc)


# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════════════════


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences (```json ... ```) if present.

    Args:
        text: Raw LLM response string.

    Returns:
        Cleaned string without fences.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove the opening fence line
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        # Remove the closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """Parse and validate the structured JSON response from the judge LLM.

    Recomputes weighted_score and normalized_score locally to avoid trusting
    the model's arithmetic.

    Args:
        raw: Raw text response from the LLM.

    Returns:
        Validated and enriched result dict containing ``scores``,
        ``weighted_score``, ``normalized_score``, and all qualitative fields.

    Raises:
        ValueError: If the response is not valid JSON or scores are out of range.
    """
    cleaned = _strip_code_fences(raw)

    data = json.loads(cleaned)  # may raise JSONDecodeError → caught by caller

    scores: Dict[str, int] = data.get("scores", {})

    # --- validate each criterion is present and in [1, 5] ---
    for criterion in CRITERIA_NAMES:
        val = scores.get(criterion)
        if val is None:
            raise ValueError(f"Missing score for criterion '{criterion}'")
        val = int(val)
        if not 1 <= val <= 5:
            raise ValueError(
                f"Score for '{criterion}' is {val}; must be in [1, 5]"
            )
        scores[criterion] = val
    data["scores"] = scores

    # --- recompute weighted_score locally (never trust model arithmetic) ---
    weighted = sum(
        scores[c] * CRITERION_WEIGHTS[c] for c in CRITERIA_NAMES
    )
    data["weighted_score"] = round(weighted, 4)
    data["normalized_score"] = round(weighted / 5.0 * 100.0, 4)

    # --- ensure list fields exist ---
    for list_key in ("strengths", "weaknesses", "critical_issues",
                     "improvement_suggestions"):
        if not isinstance(data.get(list_key), list):
            data[list_key] = []

    # --- clamp judge_confidence to [0, 1] ---
    conf = data.get("judge_confidence")
    if conf is not None:
        try:
            conf = float(conf)
            data["judge_confidence"] = max(0.0, min(1.0, round(conf, 4)))
        except (ValueError, TypeError):
            data["judge_confidence"] = None
    else:
        data["judge_confidence"] = None

    if not isinstance(data.get("evaluation_notes"), str):
        data["evaluation_notes"] = ""

    return data


def _error_result() -> Dict[str, Any]:
    """Return a structured error-state result with all fields set to null.

    Returns:
        Dict matching the expected score_section output shape with
        ``judge_error`` set to True and all scores null.
    """
    return {
        "scores": {c: None for c in CRITERIA_NAMES},
        "weighted_score": None,
        "normalized_score": None,
        "strengths": [],
        "weaknesses": [],
        "critical_issues": [],
        "improvement_suggestions": [],
        "judge_confidence": None,
        "evaluation_notes": "",
        "judge_error": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PMFJudge CLASS
# ═══════════════════════════════════════════════════════════════════════════


class PMFJudge:
    """Evaluate PMF document sections using an LLM as a rubric-based judge.

    Supports Anthropic Claude (preferred) and Azure OpenAI GPT-4o as
    backend providers.  Responses are cached by SHA-256 hash of the
    evaluation inputs to avoid redundant API calls.

    Args:
        provider:          ``"anthropic"`` or ``"azure_openai"``.
        model:             Model name or Azure deployment name.
        api_key:           Provider API key.  Falls back to env vars
                           ``ANTHROPIC_API_KEY`` / ``AZURE_OPENAI_API_KEY``.
        azure_endpoint:    Azure OpenAI endpoint URL.  Falls back to
                           ``AZURE_OPENAI_ENDPOINT`` env var.
        azure_api_version: Azure API version.  Falls back to
                           ``AZURE_OPENAI_API_VERSION`` env var or
                           ``"2024-06-01"``.
        temperature:       Sampling temperature (0.0 = deterministic).
        cache_enabled:     If True, cache results by SHA-256 hash to avoid
                           re-evaluating unchanged sections.
        cache_dir:         Directory for cached JSON files.

    Example:
        >>> judge = PMFJudge(provider="anthropic")
        >>> result = judge.score_section(
        ...     section_key="DEVICE DESCRIPTION",
        ...     section_instruction="Describe the medical devices...",
        ...     retrieved_context="Source: sterile connectors are produced...",
        ...     generated_output="The Langensbold site manufactures...",
        ...     site_name="Langensbold",
        ... )
        >>> result["normalized_score"]
        84.0
    """

    RUBRIC_VERSION: str = "v1.0"

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        temperature: float = 0.0,
        cache_enabled: bool = True,
        cache_dir: str = "data/eval_cache",
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir

        # --- resolve API key ---
        if api_key is None:
            if self.provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            else:
                api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._api_key = api_key

        # --- Azure-specific config ---
        self._azure_endpoint = azure_endpoint or os.environ.get(
            "AZURE_OPENAI_ENDPOINT", ""
        )
        self._azure_api_version = azure_api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-06-01"
        )

        # --- build client ---
        self._client: Any = None
        self._init_client()

    # ── client init ──────────────────────────────────────────────────────

    def _init_client(self) -> None:
        """Initialise the LLM client based on provider."""
        if self.provider == "anthropic":
            if _anthropic is None:
                logger.error(
                    "anthropic SDK not installed — "
                    "run `pip install anthropic`."
                )
                return
            self._client = _anthropic.Anthropic(api_key=self._api_key)
        elif self.provider == "azure_openai":
            if _AzureOpenAI is None:
                logger.error(
                    "openai SDK not installed — "
                    "run `pip install openai`."
                )
                return
            self._client = _AzureOpenAI(
                api_key=self._api_key,
                api_version=self._azure_api_version,
                azure_endpoint=self._azure_endpoint,
            )
        else:
            logger.error("Unsupported provider: '%s'", self.provider)

    # ── low-level LLM call ───────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> str:
        """Send system + user prompt to the configured LLM.

        Args:
            user_prompt: The user-role prompt text.

        Returns:
            Raw text from the model's response.

        Raises:
            RuntimeError: If no client is available.
        """
        if self._client is None:
            raise RuntimeError(
                f"No LLM client available for provider '{self.provider}'. "
                "Check SDK installation and API key."
            )

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=self.temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text if response.content else ""

        # azure_openai
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""

    # ── prompt construction ──────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(
        section_key: str,
        section_instruction: str,
        retrieved_context: str,
        generated_output: str,
        site_name: str,
        reference_output: str,
    ) -> str:
        """Render the user prompt from the template.

        Args:
            section_key:        Section identifier.
            section_instruction: The instruction given to the generator.
            retrieved_context:  Source documents (truncated to 8 000 chars).
            generated_output:   The text to evaluate.
            site_name:          Manufacturing site name.
            reference_output:   Optional ground-truth reference.

        Returns:
            Fully rendered user prompt string.
        """
        # Truncate context to 8000 chars as specified.
        if len(retrieved_context) > 8000:
            retrieved_context = retrieved_context[:8000]

        reference_section = ""
        if reference_output:
            reference_section = (
                "## Reference Output (ground truth for comparison)\n"
                f"{reference_output}\n"
            )

        return _USER_TEMPLATE.format(
            section_key=section_key,
            section_instruction=section_instruction,
            retrieved_context=retrieved_context,
            site_name=site_name,
            reference_section=reference_section,
            generated_output=generated_output,
            rubric_text=_RUBRIC_TEXT,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API: score_section
    # ══════════════════════════════════════════════════════════════════════

    def score_section(
        self,
        section_key: str,
        section_instruction: str,
        retrieved_context: str,
        generated_output: str,
        site_name: str = "",
        reference_output: str = "",
    ) -> Dict[str, Any]:
        """Evaluate a single PMF section with the LLM judge.

        Builds a structured prompt, sends it to the configured LLM,
        parses the JSON response, and returns a validated result dict.
        Results are cached by SHA-256 hash of (section_key, generated_output,
        rubric_version) when caching is enabled.

        Args:
            section_key:        Section name / identifier (e.g. ``"DEVICE DESCRIPTION"``).
            section_instruction: The original instruction given to the AI generator.
            retrieved_context:  Concatenated text from retrieved source documents.
            generated_output:   The AI-generated section text to evaluate.
            site_name:          Manufacturing site name for site-specificity checks.
            reference_output:   Optional ground-truth reference for comparison.

        Returns:
            Dict with the following structure::

                {
                  "scores": {
                    "factual_accuracy": int,        # 1-5
                    "regulatory_language": int,      # 1-5
                    "site_specificity": int,         # 1-5
                    "completeness": int,             # 1-5
                    "structural_coherence": int      # 1-5
                  },
                  "weighted_score": float,           # 0-5
                  "normalized_score": float,         # 0-100
                  "strengths": [...],
                  "weaknesses": [...],
                  "critical_issues": [...],
                  "improvement_suggestions": [...],
                  "judge_confidence": float | None,  # 0-1
                  "evaluation_notes": str,
                  "section_key": str,
                  "site_name": str,
                  "judge_model": str,
                  "judge_provider": str,
                  "rubric_version": str,
                  "cached": bool,
                  "evaluated_at": str               # ISO timestamp
                }

        Raises:
            Nothing — returns a structured error dict with ``judge_error: true``
            and all scores set to ``null`` if both parse attempts fail.

        Example:
            >>> result = judge.score_section(
            ...     "EXECUTIVE SUMMARY",
            ...     "Write an executive summary ...",
            ...     "Source: The Langensbold site ...",
            ...     "The Langensbold manufacturing site is ...",
            ...     site_name="Langensbold",
            ... )
            >>> result["normalized_score"]
            82.0
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- metadata appended to every result ---
        meta = {
            "section_key": section_key,
            "site_name": site_name,
            "judge_model": self.model,
            "judge_provider": self.provider,
            "rubric_version": self.RUBRIC_VERSION,
            "cached": False,
            "evaluated_at": now_iso,
        }

        # --- cache check ---
        ck = _cache_key(section_key, generated_output, self.RUBRIC_VERSION)
        if self.cache_enabled:
            cached = _read_cache(self.cache_dir, ck)
            if cached is not None:
                logger.info("Cache hit for %s (hash=%s…)", section_key, ck[:12])
                cached.update(meta)
                cached["cached"] = True
                cached["evaluated_at"] = now_iso
                return cached

        # --- build prompt ---
        user_prompt = self._build_user_prompt(
            section_key=section_key,
            section_instruction=section_instruction,
            retrieved_context=retrieved_context,
            generated_output=generated_output,
            site_name=site_name,
            reference_output=reference_output,
        )

        # --- first LLM call ---
        result: Optional[Dict[str, Any]] = None
        try:
            raw = self._call_llm(user_prompt)
            result = _parse_judge_response(raw)
        except Exception as exc:
            logger.warning(
                "First judge attempt for '%s' failed: %s — retrying.",
                section_key, exc,
            )
            # --- retry with stricter prompt ---
            try:
                raw_retry = self._call_llm(user_prompt + _RETRY_SUFFIX)
                result = _parse_judge_response(raw_retry)
            except Exception as exc2:
                logger.error(
                    "Second judge attempt for '%s' also failed: %s",
                    section_key, exc2,
                )

        if result is None:
            result = _error_result()

        result.update(meta)

        # --- write to cache on success ---
        if self.cache_enabled and not result.get("judge_error"):
            _write_cache(self.cache_dir, ck, result)

        return result

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API: score_document
    # ══════════════════════════════════════════════════════════════════════

    def score_document(
        self,
        sections: List[Dict[str, Any]],
        site_name: str = "",
        parallel: bool = False,
        max_workers: int = 3,
    ) -> Dict[str, Any]:
        """Evaluate all sections in a PMF document run.

        Args:
            sections:    List of section dicts.  Each must contain at least
                         ``section_key``, ``prompt_text`` (or
                         ``section_instruction``), and ``generated_text``.
                         ``retrieved_context`` or ``retrieved_paths`` should
                         also be provided.
            site_name:   Manufacturing site name.
            parallel:    If True, score sections concurrently using a
                         ``ThreadPoolExecutor``.  Be mindful of rate limits.
            max_workers: Maximum concurrent threads when ``parallel=True``.

        Returns:
            Dict with the following structure::

                {
                  "section_scores": [... per-section result dicts ...],
                  "document_weighted_score": float,
                  "document_normalized_score": float,
                  "lowest_scoring_sections": [... top 3 worst ...],
                  "critical_issues_summary": [... all issues ...],
                  "sections_with_judge_error": [...],
                  "total_sections": int,
                  "scored_sections": int
                }

        Raises:
            Nothing — per-section failures are captured in each result dict.

        Example:
            >>> doc = judge.score_document(run_artifact["sections"], "Langensbold")
            >>> doc["document_normalized_score"]
            78.5
        """

        def _score_one(section: Dict[str, Any]) -> Dict[str, Any]:
            key = section.get("section_key", "UNKNOWN")
            instruction = (
                section.get("prompt_text")
                or section.get("section_instruction", "")
            )
            context = section.get("retrieved_context", "")
            output = section.get("generated_text", "")
            ref = section.get("reference_output", "")
            return self.score_section(
                section_key=key,
                section_instruction=instruction,
                retrieved_context=context,
                generated_output=output,
                site_name=site_name,
                reference_output=ref,
            )

        # --- execute (serial or parallel) ---
        section_scores: List[Dict[str, Any]]
        if parallel and len(sections) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(_score_one, s) for s in sections
                ]
                section_scores = [
                    f.result() for f in concurrent.futures.as_completed(futures)
                ]
        else:
            section_scores = [_score_one(s) for s in sections]

        # --- aggregate ---
        valid_scores = [
            s for s in section_scores if not s.get("judge_error")
        ]
        error_sections = [
            s.get("section_key", "?") for s in section_scores
            if s.get("judge_error")
        ]

        if valid_scores:
            doc_weighted = round(
                sum(s["weighted_score"] for s in valid_scores)
                / len(valid_scores),
                4,
            )
            doc_normalized = round(doc_weighted / 5.0 * 100.0, 4)
        else:
            doc_weighted = 0.0
            doc_normalized = 0.0

        # lowest scoring (up to 3)
        sorted_valid = sorted(
            valid_scores, key=lambda s: s.get("normalized_score", 0)
        )
        lowest = sorted_valid[:3]

        # collect all critical issues
        all_critical: List[str] = []
        for s in section_scores:
            for issue in s.get("critical_issues", []):
                all_critical.append(
                    f"[{s.get('section_key', '?')}] {issue}"
                )

        return {
            "section_scores": section_scores,
            "document_weighted_score": doc_weighted,
            "document_normalized_score": doc_normalized,
            "lowest_scoring_sections": lowest,
            "critical_issues_summary": all_critical,
            "sections_with_judge_error": error_sections,
            "total_sections": len(sections),
            "scored_sections": len(valid_scores),
        }

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API: compare_models
    # ══════════════════════════════════════════════════════════════════════

    def compare_models(
        self,
        test_cases: List[Dict[str, Any]],
        model_configs: List[Dict[str, str]],
    ) -> Any:
        """Compare normalised scores across multiple model configurations.

        For each model, a fresh ``PMFJudge`` instance is created with the
        given config and run against every test case.

        Args:
            test_cases:    List of dicts, each with ``section_key``,
                           ``section_instruction``, ``retrieved_context``,
                           and ``generated_outputs`` (a dict mapping
                           model_name → generated text).
            model_configs: List of dicts, each with ``name`` and ``provider``
                           (e.g. ``{"name": "gpt-4o", "provider": "azure_openai"}``).

        Returns:
            ``pandas.DataFrame`` with columns: ``section_key``,
            ``model_name``, ``normalized_score``, ``factual_accuracy``,
            ``regulatory_language``, ``site_specificity``, ``completeness``,
            ``structural_coherence``.

        Raises:
            RuntimeError: If pandas is not installed.

        Example:
            >>> df = judge.compare_models(cases, configs)
            >>> df.groupby("model_name")["normalized_score"].mean()
        """
        if _pd is None:
            raise RuntimeError(
                "pandas is required for compare_models — "
                "run `pip install pandas`."
            )

        rows: List[Dict[str, Any]] = []

        for case in test_cases:
            gen_outputs = case.get("generated_outputs", {})
            for cfg in model_configs:
                model_name = cfg["name"]
                gen_text = gen_outputs.get(model_name, "")
                if not gen_text:
                    continue

                result = self.score_section(
                    section_key=case.get("section_key", ""),
                    section_instruction=case.get("section_instruction", ""),
                    retrieved_context=case.get("retrieved_context", ""),
                    generated_output=gen_text,
                    site_name=case.get("site_name", ""),
                )
                scores = result.get("scores", {})
                rows.append({
                    "section_key": case.get("section_key", ""),
                    "model_name": model_name,
                    "normalized_score": result.get("normalized_score", 0),
                    "factual_accuracy": scores.get("factual_accuracy"),
                    "regulatory_language": scores.get("regulatory_language"),
                    "site_specificity": scores.get("site_specificity"),
                    "completeness": scores.get("completeness"),
                    "structural_coherence": scores.get("structural_coherence"),
                })

        return _pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )

    print("=" * 72)
    print("PMFJudge — SMOKE TEST (no live API calls)")
    print("=" * 72)

    # ── 1. Rubric inspection ─────────────────────────────────────────────
    print("\n--- JUDGE_RUBRIC criteria ---")
    for name, info in JUDGE_RUBRIC.items():
        print(f"  {name}  (weight={info['weight']})")
    total_weight = sum(info["weight"] for info in JUDGE_RUBRIC.values())
    print(f"  Total weight = {total_weight}")
    assert abs(total_weight - 1.0) < 1e-9, "Weights must sum to 1.0"
    print("  Weights sum check: PASS")

    print(f"\n--- Formatted rubric length: {len(_RUBRIC_TEXT)} chars ---")
    print(_RUBRIC_TEXT[:300] + "...\n")

    # ── 2. Prompt rendering ──────────────────────────────────────────────
    prompt = PMFJudge._build_user_prompt(
        section_key="DEVICE DESCRIPTION",
        section_instruction="Describe the medical devices manufactured at the site.",
        retrieved_context=(
            "The Langensbold site manufactures sterile connectors and "
            "bioreactor assemblies for biopharmaceutical applications."
        ),
        generated_output=(
            "The Langensbold manufacturing site produces single-use "
            "bioreactor assemblies and sterile connectors used in "
            "biopharmaceutical manufacturing. Quality management follows "
            "ISO 13485 and cGMP guidelines."
        ),
        site_name="Langensbold",
        reference_output="",
    )
    print(f"--- User prompt length: {len(prompt)} chars ---")
    assert "Section Key: DEVICE DESCRIPTION" in prompt
    assert "Langensbold" in prompt
    assert "CRITERION 1" in prompt
    print("  Prompt structure check: PASS")

    # ── 3. Response parsing — valid response ─────────────────────────────
    mock_response = json.dumps({
        "scores": {
            "factual_accuracy": 4,
            "regulatory_language": 3,
            "site_specificity": 5,
            "completeness": 4,
            "structural_coherence": 4,
        },
        "weighted_score": 3.95,
        "normalized_score": 79.0,
        "strengths": [
            "Accurate site reference",
            "Good coverage of device types",
        ],
        "weaknesses": [
            "Could use more precise regulatory terminology",
        ],
        "critical_issues": [],
        "improvement_suggestions": [
            "Add ISO classification references",
        ],
        "judge_confidence": 0.85,
        "evaluation_notes": (
            "The output accurately reflects source documents and correctly "
            "references the Langensbold site. Regulatory language could be "
            "improved by using more ICH Q10 terminology."
        ),
    })
    print("\n--- Parsing valid mock response ---")
    parsed = _parse_judge_response(mock_response)
    print(f"  Scores: {parsed['scores']}")
    print(f"  Weighted (recomputed): {parsed['weighted_score']}")
    print(f"  Normalized (recomputed): {parsed['normalized_score']}")

    # Verify recomputation
    expected_weighted = round(
        4 * 0.30 + 3 * 0.25 + 5 * 0.20 + 4 * 0.15 + 4 * 0.10, 4
    )
    assert parsed["weighted_score"] == expected_weighted, (
        f"Expected {expected_weighted}, got {parsed['weighted_score']}"
    )
    expected_normalized = round(expected_weighted / 5.0 * 100.0, 4)
    assert parsed["normalized_score"] == expected_normalized
    print(f"  Weighted score recomputation check: PASS ({expected_weighted})")
    print(f"  Normalized score check: PASS ({expected_normalized})")
    print(f"  Strengths: {parsed['strengths']}")
    print(f"  Weaknesses: {parsed['weaknesses']}")
    print(f"  Confidence: {parsed['judge_confidence']}")

    # ── 4. Response parsing — markdown-fenced response ───────────────────
    fenced = "```json\n" + mock_response + "\n```"
    print("\n--- Parsing fenced mock response ---")
    parsed2 = _parse_judge_response(fenced)
    assert parsed2["scores"] == parsed["scores"]
    print("  Fence stripping: PASS")

    # ── 5. Response parsing — invalid response ───────────────────────────
    print("\n--- Parsing invalid response ---")
    try:
        _parse_judge_response("This is not JSON at all.")
        print("  ERROR: should have raised ValueError")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Correctly raised error: {type(e).__name__}")

    # ── 6. Response parsing — out-of-range score ─────────────────────────
    print("\n--- Parsing out-of-range score ---")
    bad_scores = json.dumps({
        "scores": {
            "factual_accuracy": 7,  # out of range
            "regulatory_language": 3,
            "site_specificity": 5,
            "completeness": 4,
            "structural_coherence": 4,
        },
    })
    try:
        _parse_judge_response(bad_scores)
        print("  ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly caught: {e}")

    # ── 7. Cache key determinism ─────────────────────────────────────────
    print("\n--- Cache key check ---")
    k1 = _cache_key("SEC_A", "text one", "v1.0")
    k2 = _cache_key("SEC_A", "text one", "v1.0")
    k3 = _cache_key("SEC_A", "text two", "v1.0")
    k4 = _cache_key("SEC_A", "text one", "v2.0")
    assert k1 == k2, "Same inputs must produce same hash"
    assert k1 != k3, "Different text must produce different hash"
    assert k1 != k4, "Different rubric version must produce different hash"
    print(f"  Hash (SEC_A/text one/v1.0): {k1[:16]}…")
    print("  Determinism check: PASS")
    print("  Invalidation check: PASS")

    # ── 8. Error result structure ────────────────────────────────────────
    print("\n--- Error result structure ---")
    err = _error_result()
    assert err["judge_error"] is True
    assert all(v is None for v in err["scores"].values())
    assert err["weighted_score"] is None
    print("  Error result check: PASS")

    # ── 9. PMFJudge instantiation (no API key — client will be None) ─────
    print("\n--- PMFJudge instantiation (no API key) ---")
    judge = PMFJudge(provider="anthropic", api_key="")
    print(f"  Provider: {judge.provider}")
    print(f"  Model: {judge.model}")
    print(f"  Rubric version: {judge.RUBRIC_VERSION}")
    print(f"  Cache enabled: {judge.cache_enabled}")
    print(f"  Client available: {judge._client is not None}")

    # ── 10. score_section without live client → error result ─────────────
    print("\n--- score_section without API (expects error result) ---")
    result = judge.score_section(
        section_key="TEST SECTION",
        section_instruction="Write a test section.",
        retrieved_context="Test context.",
        generated_output="Test output.",
        site_name="TestSite",
    )
    print(f"  judge_error: {result.get('judge_error')}")
    print(f"  section_key: {result.get('section_key')}")
    print(f"  judge_model: {result.get('judge_model')}")
    print(f"  rubric_version: {result.get('rubric_version')}")
    assert result.get("section_key") == "TEST SECTION"
    assert result.get("judge_model") == "claude-sonnet-4-6"
    assert result.get("rubric_version") == "v1.0"
    print("  Metadata fields: PASS")

    # ── 11. score_document ───────────────────────────────────────────────
    print("\n--- score_document with mock sections ---")
    mock_sections = [
        {
            "section_key": "SECTION A",
            "prompt_text": "Write section A.",
            "generated_text": "Content A.",
            "retrieved_context": "Context A.",
        },
        {
            "section_key": "SECTION B",
            "prompt_text": "Write section B.",
            "generated_text": "Content B.",
            "retrieved_context": "Context B.",
        },
    ]
    doc_result = judge.score_document(mock_sections, site_name="TestSite")
    print(f"  total_sections: {doc_result['total_sections']}")
    print(f"  scored_sections: {doc_result['scored_sections']}")
    print(f"  sections_with_judge_error: {doc_result['sections_with_judge_error']}")
    assert doc_result["total_sections"] == 2
    print("  score_document structure: PASS")

    print("\n" + "=" * 72)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 72)
