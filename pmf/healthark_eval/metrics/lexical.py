"""Thin re-export of LexicalMetrics from src.eval.eval_metrics."""

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.eval_metrics import LexicalMetrics  # noqa: F401, E402

__all__ = ["LexicalMetrics"]
