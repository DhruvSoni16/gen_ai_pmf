"""
MLflow Experiment Tracker for PMF Document Evaluation
======================================================

Logs every evaluation run as an MLflow experiment run so that scores
can be compared across document versions, model changes, and template
updates — standard enterprise ML experiment tracking.

Local mode (default): stores run data in ./mlruns — no server needed.
View UI: run `mlflow ui` in the project folder, then open
         http://localhost:5000 in your browser.

Logged per run:
  Parameters  — site_name, template_file, model_name, timestamp
  Metrics     — rule_score, judge_score, faithfulness, rag_triad_score,
                hallucination_score, answer_relevance_score,
                regulatory_tone_score, opik_composite, composite_score
  Tags        — overall_grade, framework, sections_evaluated

Part of the Healthark GenAI Evaluation Framework (Initiative 4).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy import — only resolved at call time to avoid slow startup
_mlflow = None


def _get_mlflow():
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow as _m
            _mlflow = _m
        except ImportError:
            logger.warning("mlflow not installed — experiment tracking disabled.")
    return _mlflow


# ═══════════════════════════════════════════════════════════════════════════
# TRACKER CLASS
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME = "PMF_Document_Generation"
TRACKING_URI = "mlruns"   # local file-based store; change to a remote URI if needed


class MLflowTracker:
    """Log PMF evaluation runs to MLflow for experiment tracking.

    Usage:
        tracker = MLflowTracker()
        run_id = tracker.log_run(run_artifacts, eval_summary)
        print(f"MLflow run: {tracker.run_url(run_id)}")

    Args:
        tracking_uri: Local path or remote MLflow server URI.
                      Defaults to './mlruns' (local file store).
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        tracking_uri: str = TRACKING_URI,
        experiment_name: str = EXPERIMENT_NAME,
    ):
        self.tracking_uri = os.path.abspath(tracking_uri)
        self.experiment_name = experiment_name
        self._enabled = _get_mlflow() is not None

    @property
    def enabled(self) -> bool:
        return self._enabled and _get_mlflow() is not None

    # ─────────────────────────────────────────────────────────────────────

    def _ensure_experiment(self) -> Optional[str]:
        """Create or retrieve the MLflow experiment. Returns experiment_id."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return None
        try:
            mlflow.set_tracking_uri(f"file:///{self.tracking_uri}")
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                return mlflow.create_experiment(self.experiment_name)
            return exp.experiment_id
        except Exception as exc:
            logger.warning("MLflow experiment setup failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────

    def log_run(
        self,
        run_artifacts: Dict[str, Any],
        eval_summary: Dict[str, Any],
        extended_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Log a complete PMF evaluation run to MLflow.

        Args:
            run_artifacts:    The run_artifacts dict from extraction_pmf().
            eval_summary:     The rule-based evaluation dict (document_scores).
            extended_summary: The extended_eval_summary dict (judge/RAG/Opik).

        Returns:
            MLflow run_id string, or None if logging failed.
        """
        if not self.enabled:
            logger.info("MLflow not available — skipping experiment logging.")
            return None

        mlflow = _get_mlflow()
        exp_id = self._ensure_experiment()
        if exp_id is None:
            return None

        ext = extended_summary or {}
        doc_scores = eval_summary.get("document_scores", eval_summary)

        # ── Parameters (stable descriptors of the run) ─────────────────
        params = {
            "site_name": run_artifacts.get("site_name", ""),
            "template_file": os.path.basename(run_artifacts.get("template_file", "")),
            "model_name": run_artifacts.get("model_name", ""),
            "timestamp": run_artifacts.get("timestamp", ""),
            "sections_evaluated": str(doc_scores.get("section_count", 0)),
            "framework": ext.get("framework", "rule_based"),
        }

        # ── Metrics (numeric values to track over time) ─────────────────
        metrics: Dict[str, float] = {}

        def _add(key: str, val: Any) -> None:
            if val is not None:
                try:
                    metrics[key] = round(float(val), 4)
                except (ValueError, TypeError):
                    pass

        # Rule-based
        _add("rule_score", doc_scores.get("overall_score"))
        _add("section_count", doc_scores.get("section_count"))
        _add("retrieval_coverage_pct", doc_scores.get("retrieval_coverage"))

        # Extended — DeepEval RAG Triad
        _add("judge_score", ext.get("mean_judge_normalized"))
        _add("faithfulness", ext.get("mean_faithfulness"))
        _add("rag_triad_score", ext.get("mean_rag_triad_score") or ext.get("mean_ragas"))
        _add("composite_score", ext.get("mean_composite"))

        # Opik-style
        _add("hallucination_score", ext.get("mean_hallucination_score"))
        _add("answer_relevance_score", ext.get("mean_answer_relevance_score"))
        _add("regulatory_tone_score", ext.get("mean_regulatory_tone_score"))
        _add("opik_composite", ext.get("mean_opik_composite"))

        # ── Tags (categorical labels) ───────────────────────────────────
        tags = {
            "overall_grade": ext.get("overall_grade", "?"),
            "site_name": run_artifacts.get("site_name", ""),
            "model": run_artifacts.get("model_name", ""),
        }
        missing = doc_scores.get("missing_required_sections", [])
        tags["missing_sections"] = ", ".join(missing) if missing else "none"

        # ── Log to MLflow ───────────────────────────────────────────────
        try:
            mlflow.set_tracking_uri(f"file:///{self.tracking_uri}")
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(experiment_id=exp_id) as run:
                mlflow.log_params(params)
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.set_tags(tags)

                run_id = run.info.run_id
                logger.info(
                    "MLflow run logged: experiment=%s, run_id=%s, metrics=%s",
                    self.experiment_name, run_id[:8], list(metrics.keys()),
                )
                return run_id

        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────────

    def run_url(self, run_id: Optional[str] = None) -> str:
        """Return the local MLflow UI URL for a run (or the experiment overview)."""
        base = "http://localhost:5000"
        if run_id:
            return f"{base}/#/runs/{run_id}"
        return base

    def get_all_runs(self) -> Any:
        """Return a pandas DataFrame of all logged runs for this experiment."""
        mlflow = _get_mlflow()
        if mlflow is None:
            return None
        try:
            mlflow.set_tracking_uri(f"file:///{self.tracking_uri}")
            return mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["start_time DESC"],
            )
        except Exception as exc:
            logger.warning("MLflow search_runs failed: %s", exc)
            return None
