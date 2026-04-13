# Benchmark Dataset Guide

## Purpose

The benchmark dataset provides ground-truth reference data for regression
testing, model comparison, and quality monitoring of the PMF Document
Generator.  Each case represents one document section with its full
generation context and expected output.

---

## Dataset Location

```
data/benchmark/
├── seed_cases.jsonl      ← 10 initial seed cases
├── cases.jsonl           ← user-added cases (created by BenchmarkLoader.add_case())
└── schema.md             ← full schema reference
```

All `.jsonl` files in `data/benchmark/` are loaded automatically by
`BenchmarkLoader`.

---

## How to Create a New Benchmark Case

### Step 1: Prepare the case data

Each case is a single JSON object with these required fields:

```json
{
  "case_id": "pmf_case_011",
  "created_at": "2026-04-15T10:00:00Z",
  "created_by": "human",
  "validated_by": "Dr. Smith",
  "site_name": "Munich Site",
  "section_key": "ENVIRONMENTAL MONITORING",
  "section_instruction": "Describe the environmental monitoring programme...",
  "retrieval_query": "environmental monitoring particulate microbial...",
  "source_documents": ["EM_SOP_030.pdf", "Cleanroom_Report_2025.pdf"],
  "retrieved_context": "The site monitors particles at 0.5 and 5.0 micron...",
  "generated_output": {"gpt-4o-azure": null, "claude-sonnet": null},
  "reference_output": "The environmental monitoring programme covers all classified areas...",
  "expert_scores": {},
  "automated_scores": {},
  "tags": ["environmental_monitoring", "high_priority"],
  "difficulty": "hard",
  "section_type": "text"
}
```

### Step 2: Add via API or manually

**Option A: Python API**

```python
from src.eval.benchmark_loader import BenchmarkLoader

loader = BenchmarkLoader("data/benchmark")
case_id = loader.add_case(case_dict, validate=True)
print(f"Added: {case_id}")
```

This validates the schema, checks for duplicate IDs, and appends to
`data/benchmark/cases.jsonl`.

**Option B: Manual append**

Append one JSON line to any `.jsonl` file in `data/benchmark/`, then
validate:

```bash
python -m src.eval.benchmark_loader --validate
```

### Step 3: Verify

```python
loader = BenchmarkLoader("data/benchmark")
stats = loader.get_statistics()
print(stats["total_cases"])  # should include your new case
```

---

## How to Annotate Cases with Expert Scores

Expert scores provide human-judged quality ratings that serve as the gold
standard for evaluating the LLM judge's alignment with human judgment.

### Scoring Rubric

Each model output is scored on 5 criteria (1-5 scale):

| Criterion | What to evaluate |
|---|---|
| `factual_accuracy` | Are all claims supported by source documents? |
| `regulatory_language` | Does it use formal PMF/ICH Q10/ISO 13485 language? |
| `site_specificity` | Is the specific site correctly and consistently referenced? |
| `completeness` | Are all required sub-topics for this section covered? |
| `structural_coherence` | Is the content well-organised with logical flow? |

### Annotation Format

```json
"expert_scores": {
  "gpt-4o-azure": {
    "factual_accuracy": 4,
    "regulatory_language": 5,
    "site_specificity": 3,
    "completeness": 4,
    "structural_coherence": 5,
    "annotator": "Dr. Maria Chen",
    "annotation_date": "2026-04-15",
    "notes": "Good regulatory tone but site name missing in paragraph 2."
  }
}
```

### Annotation Workflow

1. Generate outputs for the benchmark cases using the PMF pipeline.
2. Export cases to CSV for spreadsheet review:
   ```python
   loader.export_to_csv("benchmark_review.csv")
   ```
3. Domain experts score each output using the rubric above.
4. Import scores back into the JSONL file.
5. Use expert scores to validate LLM judge calibration.

---

## Inter-Annotator Agreement: Cohen's Kappa

When multiple experts annotate the same cases, measure agreement using
Cohen's Kappa to ensure the rubric is applied consistently.

### Formula

```
        p_o - p_e
  k = ───────────
        1 - p_e
```

Where:
- **p_o** = observed agreement (fraction of items where both annotators
  gave the same score)
- **p_e** = expected agreement by chance (probability both annotators
  would agree if scoring randomly)

### Interpretation

| Kappa | Agreement Level |
|---|---|
| < 0.20 | Poor |
| 0.21 - 0.40 | Fair |
| 0.41 - 0.60 | Moderate |
| 0.61 - 0.80 | Substantial |
| 0.81 - 1.00 | Almost perfect |

### Worked Example

Two annotators (A and B) scored 10 sections on `factual_accuracy` (1-5).
For simplicity, we collapse to binary: score >= 4 is "Good", < 4 is "Needs Work".

```
Section   Annotator A   Annotator B   Agree?
───────   ───────────   ───────────   ──────
  1         Good          Good          Yes
  2         Good          Good          Yes
  3         Needs Work    Good          No
  4         Needs Work    Needs Work    Yes
  5         Good          Good          Yes
  6         Good          Needs Work    No
  7         Good          Good          Yes
  8         Needs Work    Needs Work    Yes
  9         Good          Good          Yes
 10         Needs Work    Needs Work    Yes
```

**Step 1: Observed agreement (p_o)**

8 out of 10 sections agree: `p_o = 8/10 = 0.80`

**Step 2: Expected agreement by chance (p_e)**

```
Annotator A: 6 Good, 4 Needs Work  →  P(A=Good) = 0.6
Annotator B: 7 Good, 3 Needs Work  →  P(B=Good) = 0.7

P(both Good by chance)       = 0.6 × 0.7 = 0.42
P(both Needs Work by chance) = 0.4 × 0.3 = 0.12

p_e = 0.42 + 0.12 = 0.54
```

**Step 3: Compute Kappa**

```
k = (0.80 - 0.54) / (1 - 0.54) = 0.26 / 0.46 = 0.565
```

**Interpretation**: Kappa = 0.565 indicates **moderate** agreement.
The annotators should discuss the 2 disagreements (sections 3 and 6)
and clarify the rubric boundary between scores 3 and 4.

---

## Ground Truth Quality Requirements

For a benchmark case to be considered production-quality:

1. **Reference output** must be written or validated by a domain expert
   with pharmaceutical regulatory experience.
2. **Reference output** must be 4-8 sentences of formal regulatory
   language.
3. **Retrieved context** must be realistic — actual extracted text from
   source documents, not summaries.
4. **Section instruction** must match the template format used in
   production (including the `@!` delimiter convention).
5. **Source documents** field must list real document names so cases can
   be traced back to source material.
6. Cases should cover all three difficulty levels (easy, medium, hard)
   and at least text + table section types.
7. Cases tagged `high_priority` should have at least one expert
   annotation.

### Minimum Dataset Size

For meaningful regression testing:
- At least **10 cases** total
- At least **3 cases** per difficulty level
- At least **1 case** per section type used in production
- At least **5 cases** with expert annotations (for judge calibration)
