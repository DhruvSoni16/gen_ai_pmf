"""
Base task configuration — extended by domain-specific tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseTaskConfig:
    """Abstract base configuration for an evaluation task.

    Subclass or instantiate with domain-specific values to customise
    scoring thresholds, required sections, and model choices.

    Attributes:
        task_name:                     Short identifier for this task.
        required_sections:             Section keys that must be present.
        min_section_chars:             Minimum character count per section.
        judge_rubric:                  Name of the rubric to use.
        bertscore_model:               Model name for BERTScore.
        faithfulness_threshold:        Minimum faithfulness for a pass.
        context_precision_threshold:   Minimum context precision for a pass.
        composite_threshold:           Minimum composite score for a pass.
    """

    task_name: str = "base"
    required_sections: List[str] = field(default_factory=list)
    min_section_chars: int = 80
    judge_rubric: str = "default"
    bertscore_model: str = "distilbert-base-uncased"
    faithfulness_threshold: float = 0.60
    context_precision_threshold: float = 0.40
    composite_threshold: float = 60.0
