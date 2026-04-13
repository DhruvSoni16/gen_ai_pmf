"""
healthark_eval — Internal LLM Evaluation Framework
====================================================

Installable package exposing the ``EvalSuite`` API for evaluating
LLM-generated documents across lexical, semantic, RAG, and LLM-judge
metrics.

Quick start::

    from healthark_eval import EvalSuite

    suite = EvalSuite(task="pmf", run_judge=False)
    result = suite.run(
        generated="The site manufactures connectors.",
        reference="The site produces sterile connectors.",
        section_key="DEVICE DESCRIPTION",
    )
    print(result.grade, result.composite_score)
"""

from healthark_eval.suite import EvalSuite, EvalResult, DocumentEvalResult
from healthark_eval.config import TaskConfig, PMF_TASK_CONFIG

__all__ = [
    "EvalSuite",
    "EvalResult",
    "DocumentEvalResult",
    "TaskConfig",
    "PMF_TASK_CONFIG",
]
__version__ = "0.1.0"
