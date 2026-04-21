from typing import Dict, List, Any


DEFAULT_MIN_CHARS = 80


def get_eval_rules() -> Dict[str, Any]:
    """
    Rule-based evaluation config for PMF (Plant Master File) generation.
    Section keys are matched using case-insensitive substring matching.

    Required sections align with EU GMP Annex 4 / ICH Q10 PMF structure.
    """
    return {
        "required_section_patterns": [
            "GENERAL INFORMATION",
            "PERSONNEL",
            "PREMISES",
            "PRODUCTION",
            "QUALITY ASSURANCE",
        ],
        "global_required_keywords": [],
        "section_rules": {
            "GENERAL INFORMATION": {
                "min_chars": 100,
                "required_keywords": [],
            },
            "MANUFACTURING ACTIVITIES": {
                "min_chars": 80,
                "required_keywords": [],
            },
            "PERSONNEL": {
                "min_chars": 100,
                "required_keywords": [],
            },
            "PREMISES": {
                "min_chars": 100,
                "required_keywords": [],
            },
            "EQUIPMENT": {
                "min_chars": 80,
                "required_keywords": [],
            },
            "SANITATION": {
                "min_chars": 80,
                "required_keywords": [],
            },
            "PRODUCTION": {
                "min_chars": 120,
                "required_keywords": [],
            },
            "QUALITY ASSURANCE": {
                "min_chars": 100,
                "required_keywords": [],
            },
            "STORAGE": {
                "min_chars": 80,
                "required_keywords": [],
            },
            "DOCUMENTATION": {
                "min_chars": 80,
                "required_keywords": [],
            },
            "INTERNAL AUDIT": {
                "min_chars": 80,
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
