"""
Shared pytest fixtures for the PMF Evaluation regression test suite.

Session-scoped fixtures are used where possible to avoid repeated
model loading and file I/O across test functions.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest
import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark")
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "eval_thresholds.yaml")
BASELINE_PATH = os.path.join(BENCHMARK_DIR, "baseline_scores.json")


# ═══════════════════════════════════════════════════════════════════════════
# benchmark_cases — loads all cases from BenchmarkLoader
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def benchmark_cases() -> List[Dict[str, Any]]:
    """Load all benchmark cases from data/benchmark/ once per session."""
    from src.eval.benchmark_loader import BenchmarkLoader

    loader = BenchmarkLoader(BENCHMARK_DIR)
    return loader.load_cases()


# ═══════════════════════════════════════════════════════════════════════════
# eval_config — loads thresholds from eval_thresholds.yaml
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def eval_config() -> Dict[str, Any]:
    """Load eval_thresholds.yaml once per session."""
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ═══════════════════════════════════════════════════════════════════════════
# mock_llm_client — deterministic judge response without any API call
# ═══════════════════════════════════════════════════════════════════════════


class _MockLLMClient:
    """Fake LLM client that returns a valid PMFJudge-shaped JSON response.

    Used by non-slow tests so they never touch a real API.
    """

    MOCK_RESPONSE = json.dumps({
        "scores": {
            "factual_accuracy": 4,
            "regulatory_language": 4,
            "site_specificity": 5,
            "completeness": 3,
            "structural_coherence": 4,
        },
        "weighted_score": 4.05,
        "normalized_score": 81.0,
        "strengths": ["Accurate site reference"],
        "weaknesses": ["Minor completeness gap"],
        "critical_issues": [],
        "improvement_suggestions": ["Add more detail"],
        "judge_confidence": 0.90,
        "evaluation_notes": "Overall good quality section.",
    })

    class _Messages:
        """Mimics the anthropic messages.create interface."""

        class _ContentBlock:
            def __init__(self, text: str):
                self.text = text

        @staticmethod
        def create(**kwargs: Any) -> Any:
            class _Response:
                content = [_MockLLMClient._Messages._ContentBlock(
                    _MockLLMClient.MOCK_RESPONSE
                )]
            return _Response()

    messages = _Messages()


@pytest.fixture(scope="session")
def mock_llm_client() -> _MockLLMClient:
    """A mock client matching the Anthropic interface, returning a valid
    judge response dict without calling any API."""
    return _MockLLMClient()


# ═══════════════════════════════════════════════════════════════════════════
# pmf_rule_engine — score_section callable
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def pmf_rule_engine():
    """Return the score_section function from eval_utils.py."""
    from src.eval.eval_utils import score_section
    return score_section


# ═══════════════════════════════════════════════════════════════════════════
# Shared SemanticMetrics (caches model across test session)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def semantic_metrics():
    """Session-scoped SemanticMetrics with distilbert (fast)."""
    from src.eval.eval_metrics import SemanticMetrics
    return SemanticMetrics(model_type="distilbert-base-uncased")


# ═══════════════════════════════════════════════════════════════════════════
# slow marker registration
# ═══════════════════════════════════════════════════════════════════════════


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: marks tests that call LLM APIs (skipped unless EVAL_RUN_SLOW=1)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list
) -> None:
    """Auto-skip @pytest.mark.slow tests unless EVAL_RUN_SLOW=1."""
    if os.environ.get("EVAL_RUN_SLOW", "0") == "1":
        return
    skip_slow = pytest.mark.skip(reason="Set EVAL_RUN_SLOW=1 to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
