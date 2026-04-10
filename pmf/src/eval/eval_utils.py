from typing import Dict, Any, List

from src.eval.eval_config import resolve_rule_for_section


def _contains_all_keywords(text: str, keywords: List[str]) -> Dict[str, Any]:
    text_lower = (text or "").lower()
    missing = [kw for kw in keywords if kw.lower() not in text_lower]
    return {
        "missing_keywords": missing,
        "keywords_passed": len(missing) == 0,
    }


def score_section(section_key: str, section_text: str, rules: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    rule = resolve_rule_for_section(section_key, rules)
    min_chars = int(rule.get("min_chars", 80))
    required_keywords = rule.get("required_keywords", [])
    char_len = len((section_text or "").strip())

    keyword_check = _contains_all_keywords(section_text, required_keywords)
    min_length_passed = char_len >= min_chars
    non_empty_passed = char_len > 0

    site_name = (context.get("site_name") or "").strip()
    site_name_passed = True
    if site_name:
        site_name_passed = site_name.lower() in (section_text or "").lower()

    checks = {
        "non_empty_passed": non_empty_passed,
        "min_length_passed": min_length_passed,
        "keywords_passed": keyword_check["keywords_passed"],
        "site_name_passed": site_name_passed,
    }
    passed_count = sum(1 for value in checks.values() if value)
    total_count = len(checks)
    score = round((passed_count / total_count) * 100.0, 2) if total_count else 0.0

    return {
        "section_key": section_key,
        "score": score,
        "char_len": char_len,
        "required_min_chars": min_chars,
        "required_keywords": required_keywords,
        "missing_keywords": keyword_check["missing_keywords"],
        "checks": checks,
    }


def score_document(run_artifacts: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    section_results = []
    sections = run_artifacts.get("sections", [])
    for section in sections:
        section_key = section.get("section_key", "")
        section_text = section.get("generated_text", "")
        section_results.append(
            score_section(
                section_key=section_key,
                section_text=section_text,
                rules=rules,
                context={"site_name": run_artifacts.get("site_name", "")},
            )
        )

    required_patterns = [p.upper() for p in rules.get("required_section_patterns", [])]
    seen_keys_upper = [s.get("section_key", "").upper() for s in section_results]
    missing_required_sections = [
        pat for pat in required_patterns if not any(pat in key for key in seen_keys_upper)
    ]

    retrieval_sections = [s for s in sections if not s.get("is_static", False)]
    retrieval_non_empty = [
        s for s in retrieval_sections if len(s.get("retrieved_paths", [])) > 0
    ]
    retrieval_coverage = round(
        (len(retrieval_non_empty) / len(retrieval_sections)) * 100.0, 2
    ) if retrieval_sections else 100.0

    overall = round(
        sum(s["score"] for s in section_results) / len(section_results), 2
    ) if section_results else 0.0

    return {
        "overall_score": overall,
        "section_count": len(section_results),
        "missing_required_sections": missing_required_sections,
        "retrieval_coverage": retrieval_coverage,
        "sections": section_results,
    }


def evaluate_run(run_artifacts: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    document_scores = score_document(run_artifacts, rules)
    return {
        "run_meta": {
            "timestamp": run_artifacts.get("timestamp"),
            "site_name": run_artifacts.get("site_name"),
            "template_file": run_artifacts.get("template_file"),
            "final_doc_path": run_artifacts.get("final_doc_path"),
        },
        "document_scores": document_scores,
    }
