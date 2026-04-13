"""
PMF-specific task configuration.
"""

from healthark_eval.tasks.base import BaseTaskConfig

PMF_TASK_CONFIG = BaseTaskConfig(
    task_name="pmf",
    required_sections=[
        "EXECUTIVE SUMMARY",
        "DEVICE DESCRIPTION",
        "PRODUCT SPECIFICATION",
        "MANUFACTURING",
        "QUALITY MANAGEMENT",
    ],
    min_section_chars=100,
    judge_rubric="pmf_regulatory",
    bertscore_model="distilbert-base-uncased",
    faithfulness_threshold=0.70,
    context_precision_threshold=0.50,
    composite_threshold=65.0,
)
