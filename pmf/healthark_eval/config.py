"""
Task Configuration — Central configuration for healthark_eval.
================================================================

Provides ``TaskConfig`` (alias for ``BaseTaskConfig``) and the pre-built
``PMF_TASK_CONFIG`` for PMF regulatory document evaluation.
"""

from __future__ import annotations

from healthark_eval.tasks.base import BaseTaskConfig
from healthark_eval.tasks.pmf import PMF_TASK_CONFIG

# Public alias so users can write ``from healthark_eval.config import TaskConfig``
TaskConfig = BaseTaskConfig

# Registry of built-in task configs, keyed by task_name.
TASK_REGISTRY = {
    "pmf": PMF_TASK_CONFIG,
}


def get_task_config(task: str) -> TaskConfig:
    """Look up a task config by name.

    Args:
        task: Task name (e.g. ``"pmf"``).

    Returns:
        The matching ``TaskConfig``.

    Raises:
        KeyError: If no config is registered for *task*.
    """
    if task not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{task}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task]


__all__ = ["TaskConfig", "PMF_TASK_CONFIG", "TASK_REGISTRY", "get_task_config"]
