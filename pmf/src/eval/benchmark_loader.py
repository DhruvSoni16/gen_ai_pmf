"""
Benchmark Dataset Loader
=========================

Loads, validates, filters, and manages benchmark cases from JSONL files
in a benchmark directory for regression testing and evaluation of the
PMF Document Generator.

Part of the Healthark GenAI Evaluation Framework (Initiative 4).

Usage:
    from src.eval.benchmark_loader import BenchmarkLoader

    loader = BenchmarkLoader("data/benchmark")
    cases  = loader.load_cases()
    hard   = loader.load_cases({"difficulty": "hard"})
    case   = loader.get_case_by_id("pmf_case_001")
    stats  = loader.get_statistics()
    loader.export_to_csv("benchmark_review.csv")
"""

from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: set = {
    "case_id",
    "created_at",
    "created_by",
    "site_name",
    "section_key",
    "section_instruction",
    "retrieval_query",
    "source_documents",
    "retrieved_context",
    "generated_output",
    "reference_output",
    "difficulty",
    "section_type",
}

VALID_DIFFICULTIES: set = {"easy", "medium", "hard"}
VALID_SECTION_TYPES: set = {"text", "table", "image", "static"}
VALID_CREATED_BY: set = {"human", "synthetic"}

EXPERT_SCORE_FIELDS: set = {
    "factual_accuracy",
    "regulatory_language",
    "site_specificity",
    "completeness",
    "structural_coherence",
}


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def validate_case(data: Dict[str, Any], source: str = "") -> List[str]:
    """Validate a single benchmark case against the schema.

    Args:
        data:   Parsed JSON object for one case.
        source: File/line reference for error messages.

    Returns:
        List of validation error strings (empty if valid).

    Example:
        >>> errors = validate_case({"case_id": "pmf_case_001", ...})
        >>> assert len(errors) == 0
    """
    errors: List[str] = []
    prefix = source or f"case_id={data.get('case_id', '?')}"

    # --- required fields ---
    for field_name in REQUIRED_FIELDS:
        if field_name not in data or data[field_name] is None:
            errors.append(f"{prefix}: missing required field '{field_name}'")

    # --- string-not-empty checks ---
    for field_name in ("case_id", "section_key", "site_name",
                       "section_instruction", "reference_output"):
        val = data.get(field_name)
        if isinstance(val, str) and not val.strip():
            errors.append(f"{prefix}: field '{field_name}' is empty")

    # --- enum checks ---
    if data.get("difficulty") and data["difficulty"] not in VALID_DIFFICULTIES:
        errors.append(
            f"{prefix}: 'difficulty' must be one of {VALID_DIFFICULTIES}, "
            f"got '{data['difficulty']}'"
        )
    if data.get("section_type") and data["section_type"] not in VALID_SECTION_TYPES:
        errors.append(
            f"{prefix}: 'section_type' must be one of {VALID_SECTION_TYPES}, "
            f"got '{data['section_type']}'"
        )
    if data.get("created_by") and data["created_by"] not in VALID_CREATED_BY:
        errors.append(
            f"{prefix}: 'created_by' must be one of {VALID_CREATED_BY}, "
            f"got '{data['created_by']}'"
        )

    # --- type checks ---
    if "source_documents" in data and not isinstance(data["source_documents"], list):
        errors.append(f"{prefix}: 'source_documents' must be a list")
    if "generated_output" in data and not isinstance(data["generated_output"], dict):
        errors.append(f"{prefix}: 'generated_output' must be an object")
    if "tags" in data and not isinstance(data.get("tags", []), list):
        errors.append(f"{prefix}: 'tags' must be a list")

    # --- expert_scores sub-schema ---
    expert = data.get("expert_scores")
    if expert and isinstance(expert, dict):
        for model_name, scores in expert.items():
            if not isinstance(scores, dict):
                errors.append(
                    f"{prefix}: expert_scores.{model_name} must be an object"
                )
                continue
            for dim in EXPERT_SCORE_FIELDS:
                val = scores.get(dim)
                if val is not None:
                    if not isinstance(val, int) or not 1 <= val <= 5:
                        errors.append(
                            f"{prefix}: expert_scores.{model_name}.{dim} "
                            f"must be int 1-5, got {val!r}"
                        )

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# BenchmarkLoader CLASS
# ═══════════════════════════════════════════════════════════════════════════


class BenchmarkLoader:
    """Load, filter, manage, and export benchmark cases from a directory
    of JSONL files.

    Args:
        benchmark_dir: Path to the directory containing ``.jsonl`` benchmark
                       files.  All ``.jsonl`` files in this directory are
                       loaded.

    Example:
        >>> loader = BenchmarkLoader("data/benchmark")
        >>> stats = loader.get_statistics()
        >>> stats["total_cases"]
        10
    """

    def __init__(self, benchmark_dir: str = "data/benchmark"):
        self._dir = benchmark_dir
        self._cases: Optional[List[Dict[str, Any]]] = None

    # ── internal: discover and parse ─────────────────────────────────────

    def _jsonl_files(self) -> List[str]:
        """Return sorted list of .jsonl file paths in the benchmark dir."""
        if not os.path.isdir(self._dir):
            return []
        return sorted(
            os.path.join(self._dir, f)
            for f in os.listdir(self._dir)
            if f.endswith(".jsonl")
        )

    def _load_all(self) -> List[Dict[str, Any]]:
        """Parse every .jsonl file in the benchmark directory.

        Returns:
            List of raw case dicts.
        """
        cases: List[Dict[str, Any]] = []
        for filepath in self._jsonl_files():
            fname = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        cases.append(data)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "%s line %d: invalid JSON — %s", fname, line_num, exc
                        )
        self._cases = cases
        logger.info(
            "Loaded %d benchmark cases from %s", len(cases), self._dir
        )
        return cases

    def _ensure_loaded(self) -> List[Dict[str, Any]]:
        if self._cases is None:
            self._load_all()
        assert self._cases is not None
        return self._cases

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: load_cases
    # ══════════════════════════════════════════════════════════════════════

    def load_cases(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Load benchmark cases, optionally applying filters.

        All ``.jsonl`` files in ``benchmark_dir`` are scanned.  If
        *filters* is ``None`` or empty, every case is returned.

        Supported filter keys:
            ``section_type``  — exact match (``"text"``, ``"table"``, …).
            ``difficulty``    — exact match (``"easy"``, ``"medium"``, ``"hard"``).
            ``tags``          — list of tags; case must contain **all**.
            ``site_name``     — case-insensitive exact match.
            ``section_key``   — case-insensitive substring match.
            ``created_by``    — exact match (``"human"`` or ``"synthetic"``).

        Args:
            filters: Optional dict of filter criteria.

        Returns:
            Filtered list of case dicts.

        Raises:
            Nothing — returns an empty list if the directory is missing.

        Example:
            >>> loader.load_cases({"section_type": "text", "difficulty": "hard"})
        """
        cases = list(self._ensure_loaded())
        if not filters:
            return cases

        if "section_type" in filters:
            val = filters["section_type"]
            cases = [c for c in cases if c.get("section_type") == val]

        if "difficulty" in filters:
            val = filters["difficulty"]
            cases = [c for c in cases if c.get("difficulty") == val]

        if "tags" in filters:
            required_tags = set(filters["tags"])
            cases = [
                c for c in cases
                if required_tags.issubset(set(c.get("tags", [])))
            ]

        if "site_name" in filters:
            val = filters["site_name"].lower()
            cases = [
                c for c in cases
                if (c.get("site_name") or "").lower() == val
            ]

        if "section_key" in filters:
            val = filters["section_key"].upper()
            cases = [
                c for c in cases
                if val in (c.get("section_key") or "").upper()
            ]

        if "created_by" in filters:
            val = filters["created_by"]
            cases = [c for c in cases if c.get("created_by") == val]

        return cases

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: get_case_by_id
    # ══════════════════════════════════════════════════════════════════════

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single case by its unique ``case_id``.

        Args:
            case_id: The identifier to look up.

        Returns:
            Case dict if found, else ``None``.

        Example:
            >>> loader.get_case_by_id("pmf_case_001")
        """
        for case in self._ensure_loaded():
            if case.get("case_id") == case_id:
                return case
        return None

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: add_case
    # ══════════════════════════════════════════════════════════════════════

    def add_case(
        self,
        case: Dict[str, Any],
        validate: bool = True,
    ) -> str:
        """Append a new benchmark case to ``cases.jsonl``.

        Auto-generates ``case_id`` if not provided.  Validates against the
        schema when *validate* is ``True``.

        Args:
            case:     Case dict following the benchmark schema.
            validate: If True, raise ``ValueError`` on schema violations.

        Returns:
            The ``case_id`` of the added case.

        Raises:
            ValueError: If validation fails and *validate* is True.

        Example:
            >>> cid = loader.add_case({"section_key": "NEW", ...})
        """
        # Auto-generate case_id
        if not case.get("case_id"):
            case["case_id"] = f"pmf_case_{uuid.uuid4().hex[:8]}"

        # Auto-fill created_at if absent
        if not case.get("created_at"):
            case["created_at"] = datetime.now(timezone.utc).isoformat()

        # Default empty containers
        case.setdefault("generated_output", {})
        case.setdefault("expert_scores", {})
        case.setdefault("automated_scores", {})
        case.setdefault("tags", [])

        if validate:
            errors = validate_case(case)
            if errors:
                raise ValueError(
                    f"Case validation failed ({len(errors)} errors):\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

        # Check for duplicate case_id
        existing = self.get_case_by_id(case["case_id"])
        if existing is not None:
            raise ValueError(
                f"Duplicate case_id: '{case['case_id']}' already exists."
            )

        # Append to cases.jsonl
        target = os.path.join(self._dir, "cases.jsonl")
        os.makedirs(self._dir, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

        # Update in-memory cache
        if self._cases is not None:
            self._cases.append(case)

        logger.info("Added case '%s' to %s", case["case_id"], target)
        return case["case_id"]

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: export_to_csv
    # ══════════════════════════════════════════════════════════════════════

    def export_to_csv(self, output_path: str) -> None:
        """Export benchmark cases to a flat CSV for expert review.

        Nested fields (``generated_output``, ``expert_scores``,
        ``automated_scores``, ``source_documents``, ``tags``) are
        serialised as JSON strings.  ``reference_output`` and
        ``retrieved_context`` are truncated to 500 chars for readability.

        Args:
            output_path: Destination CSV file path.

        Raises:
            Nothing — writes an empty CSV header if no cases exist.

        Example:
            >>> loader.export_to_csv("benchmark_review.csv")
        """
        cases = self._ensure_loaded()

        columns = [
            "case_id", "created_at", "created_by", "validated_by",
            "site_name", "section_key", "section_instruction",
            "retrieval_query", "source_documents", "retrieved_context",
            "generated_output", "reference_output", "expert_scores",
            "automated_scores", "tags", "difficulty", "section_type",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for case in cases:
                row: Dict[str, Any] = {}
                for col in columns:
                    val = case.get(col, "")
                    # Serialise complex types
                    if isinstance(val, (dict, list)):
                        val = json.dumps(val, ensure_ascii=False)
                    # Truncate long text for readability
                    if col in ("reference_output", "retrieved_context",
                               "section_instruction"):
                        val = str(val)[:500]
                    row[col] = val
                writer.writerow(row)

        logger.info("Exported %d cases to %s", len(cases), output_path)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: get_statistics
    # ══════════════════════════════════════════════════════════════════════

    def get_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the benchmark dataset.

        Returns:
            Dict with the following structure::

                {
                  "total_cases": int,
                  "by_section_type": {"text": int, "table": int, ...},
                  "by_difficulty": {"easy": int, "medium": int, "hard": int},
                  "cases_with_reference": int,
                  "cases_with_expert_scores": int,
                  "cases_with_automated_scores": int
                }

        Example:
            >>> loader.get_statistics()["total_cases"]
            10
        """
        cases = self._ensure_loaded()

        by_type: Dict[str, int] = {}
        by_diff: Dict[str, int] = {}
        with_ref = 0
        with_expert = 0
        with_auto = 0

        for case in cases:
            st = case.get("section_type", "unknown")
            by_type[st] = by_type.get(st, 0) + 1

            diff = case.get("difficulty", "unknown")
            by_diff[diff] = by_diff.get(diff, 0) + 1

            if case.get("reference_output", "").strip():
                with_ref += 1

            expert = case.get("expert_scores")
            if expert and isinstance(expert, dict) and len(expert) > 0:
                with_expert += 1

            auto = case.get("automated_scores")
            if auto and isinstance(auto, dict) and len(auto) > 0:
                with_auto += 1

        return {
            "total_cases": len(cases),
            "by_section_type": by_type,
            "by_difficulty": by_diff,
            "cases_with_reference": with_ref,
            "cases_with_expert_scores": with_expert,
            "cases_with_automated_scores": with_auto,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="PMF Benchmark Dataset Tool"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all cases in the benchmark directory.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics.",
    )
    parser.add_argument(
        "--export-csv",
        metavar="PATH",
        help="Export cases to a flat CSV at PATH.",
    )
    parser.add_argument(
        "--dir",
        default="data/benchmark",
        help="Benchmark directory (default: data/benchmark).",
    )
    args = parser.parse_args()

    loader = BenchmarkLoader(args.dir)

    if args.validate:
        cases = loader.load_cases()
        all_errors: List[str] = []
        seen_ids: set = set()
        for case in cases:
            cid = case.get("case_id", "?")
            errors = validate_case(case, source=f"case_id={cid}")
            all_errors.extend(errors)
            if cid in seen_ids:
                all_errors.append(f"Duplicate case_id: '{cid}'")
            seen_ids.add(cid)
        if all_errors:
            print(f"VALIDATION FAILED — {len(all_errors)} error(s):")
            for e in all_errors:
                print(f"  - {e}")
        else:
            print(f"VALIDATION PASSED — {len(cases)} cases OK")

    elif args.stats:
        stats = loader.get_statistics()
        print(json.dumps(stats, indent=2))

    elif args.export_csv:
        loader.export_to_csv(args.export_csv)
        print(f"Exported to {args.export_csv}")

    else:
        # Default: smoke test
        print("=" * 60)
        print("BENCHMARK LOADER — SMOKE TEST")
        print("=" * 60)

        cases = loader.load_cases()
        print(f"\nLoaded {len(cases)} cases")

        stats = loader.get_statistics()
        print(f"\nStatistics:\n{json.dumps(stats, indent=2)}")

        # Filters
        hard = loader.load_cases({"difficulty": "hard"})
        print(f"\nHard cases: {len(hard)}")

        text = loader.load_cases({"section_type": "text"})
        print(f"Text cases: {len(text)}")

        tagged = loader.load_cases({"tags": ["high_priority"]})
        print(f"High-priority cases: {len(tagged)}")

        # ID lookup
        c1 = loader.get_case_by_id("pmf_case_001")
        print(f"\nCase pmf_case_001: {c1['section_key'] if c1 else 'NOT FOUND'}")

        # Validation
        all_errors = []
        for case in cases:
            all_errors.extend(validate_case(case))
        print(f"\nValidation errors: {len(all_errors)}")
        for e in all_errors:
            print(f"  - {e}")

        print("\n" + "=" * 60)
        print("SMOKE TEST COMPLETE")
        print("=" * 60)
