from typing import Dict, List, Any


DEFAULT_MIN_CHARS = 80


def get_eval_rules() -> Dict[str, Any]:
    """
    Rule-based evaluation config for PMF generation.
    Section keys are matched using case-insensitive substring matching.
    """
    return {
        "required_section_patterns": [
            "EXECUTIVE SUMMARY",
            "DEVICE DESCRIPTION",
            "PRODUCT SPECIFICATION",
        ],
        "global_required_keywords": [],
        "section_rules": {
            "DEVICE DESCRIPTION": {
                "min_chars": 120,
                "required_keywords": ["device", "specification"],
            },
            "PRODUCT SPECIFICATION": {
                "min_chars": 120,
                "required_keywords": ["product", "specification"],
            },
            "EXECUTIVE SUMMARY": {
                "min_chars": 100,
                "required_keywords": [],
            },
        },
        "fallback_rule": {
            "min_chars": DEFAULT_MIN_CHARS,
            "required_keywords": [],
        },
    }


def resolve_rule_for_section(section_key: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    section_key_upper = (section_key or "").upper()
    for pattern, rule in rules.get("section_rules", {}).items():
        if pattern.upper() in section_key_upper:
            return rule
    return rules.get("fallback_rule", {"min_chars": DEFAULT_MIN_CHARS, "required_keywords": []})
