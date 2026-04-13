# Benchmark Dataset Schema — PMF Evaluation Framework

## Purpose

This dataset provides ground-truth reference data, expert annotations,
and model outputs for regression testing and quantitative evaluation of
the PMF Document Generator.  Each record represents one document
**section** with its full generation context and scoring data.

## File Format

- **Format**: JSON Lines (`.jsonl`) — one JSON object per line.
- **Encoding**: UTF-8.
- **Location**: `data/benchmark/` — all `.jsonl` files in this directory
  are loaded by `BenchmarkLoader`.
- **Primary file**: `data/benchmark/seed_cases.jsonl` (initial seed data).
- **User-added cases**: `data/benchmark/cases.jsonl` (appended via
  `BenchmarkLoader.add_case()`).

## Record Schema

```json
{
  "case_id":              "string  — REQUIRED — unique identifier, e.g. 'pmf_case_001'",
  "created_at":           "string  — REQUIRED — ISO 8601 timestamp",
  "created_by":           "string  — REQUIRED — 'human' or 'synthetic'",
  "validated_by":         "string  — OPTIONAL — expert name or null",
  "site_name":            "string  — REQUIRED — manufacturing site name",
  "section_key":          "string  — REQUIRED — template section header",
  "section_instruction":  "string  — REQUIRED — full instruction text from template",
  "retrieval_query":      "string  — REQUIRED — FAISS query used for retrieval",
  "source_documents":     ["string — REQUIRED — names/descriptions of source docs"],
  "retrieved_context":    "string  — REQUIRED — combined extracted text from retrieved docs",
  "generated_output": {
    "gpt-4o-azure":       "string  — model output or null if not run",
    "claude-sonnet":      "string  — model output or null if not run"
  },
  "reference_output":     "string  — REQUIRED — expert-written reference section text",
  "expert_scores": {
    "<model_name>": {
      "factual_accuracy":      "int 1-5",
      "regulatory_language":   "int 1-5",
      "site_specificity":      "int 1-5",
      "completeness":          "int 1-5",
      "structural_coherence":  "int 1-5",
      "annotator":             "string — expert name",
      "annotation_date":       "string — ISO date",
      "notes":                 "string — free text"
    }
  },
  "automated_scores":     "object  — OPTIONAL — filled by evaluation runs",
  "tags":                 ["string — OPTIONAL — e.g. 'executive_summary', 'high_priority'"],
  "difficulty":           "string  — REQUIRED — 'easy', 'medium', or 'hard'",
  "section_type":         "string  — REQUIRED — 'text', 'table', 'image', or 'static'"
}
```

## Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `case_id` | `string` | yes | Unique identifier for this benchmark case. Convention: `pmf_case_NNN`. |
| `created_at` | `string` | yes | ISO 8601 creation timestamp. |
| `created_by` | `string` | yes | `"human"` for expert-authored, `"synthetic"` for auto-generated. |
| `validated_by` | `string\|null` | no | Name of domain expert who validated the case, or `null`. |
| `site_name` | `string` | yes | Manufacturing site name (e.g. `"Bangalore Site"`). |
| `section_key` | `string` | yes | Template section heading, matched by `$` delimiters. |
| `section_instruction` | `string` | yes | Full LLM instruction text (left of `@!` delimiter in template). |
| `retrieval_query` | `string` | yes | FAISS retrieval query (right of `@!` delimiter in template). |
| `source_documents` | `list[string]` | yes | Names or descriptions of the source documents retrieved. |
| `retrieved_context` | `string` | yes | Combined extracted text from retrieved source documents. |
| `generated_output` | `object` | yes | Dict mapping model names to generated text (or `null` if not yet run). |
| `reference_output` | `string` | yes | Expert-written ground-truth section text (4-8 sentences, regulatory style). |
| `expert_scores` | `object` | no | Dict mapping model names to expert annotation dicts. See sub-schema below. |
| `automated_scores` | `object` | no | Filled by evaluation pipeline (BLEU, ROUGE, BERTScore, etc.). |
| `tags` | `list[string]` | no | Arbitrary tags for filtering (e.g. `["high_priority", "text_section"]`). |
| `difficulty` | `string` | yes | `"easy"`, `"medium"`, or `"hard"`. |
| `section_type` | `string` | yes | `"text"`, `"table"`, `"image"`, or `"static"`. |

### Expert Scores Sub-Schema

Each entry under `expert_scores.<model_name>` contains:

| Field | Type | Description |
|---|---|---|
| `factual_accuracy` | `int` 1-5 | Accuracy of claims vs source documents. |
| `regulatory_language` | `int` 1-5 | ICH Q10 / ISO 13485 language quality. |
| `site_specificity` | `int` 1-5 | Correct, consistent site references. |
| `completeness` | `int` 1-5 | Coverage of required sub-topics. |
| `structural_coherence` | `int` 1-5 | Logical flow and organisation. |
| `annotator` | `string` | Expert annotator name. |
| `annotation_date` | `string` | ISO date of annotation. |
| `notes` | `string` | Free-text annotation notes. |

## Difficulty Levels

- **easy**: Straightforward section with clear source material and simple
  structure.  Generator should reliably produce high-quality output.
- **medium**: Section requiring synthesis across multiple sources or
  moderate domain knowledge.  Output quality may vary.
- **hard**: Complex section requiring deep regulatory knowledge, multi-source
  synthesis, or structured formatting.  Known to be challenging for LLMs.

## Section Types

- **text**: Prose section generated by the Text Agent.
- **table**: Tabular data generated by the Table Agent.
- **image**: Section involving image extraction/embedding by the Image Agent.
- **static**: Fixed content returned as-is by the Static Agent.

## Usage

```python
from src.eval.benchmark_loader import BenchmarkLoader

loader = BenchmarkLoader("data/benchmark")
cases = loader.load_cases()                                  # all cases
hard_text = loader.load_cases({"section_type": "text", "difficulty": "hard"})
case = loader.get_case_by_id("pmf_case_001")
stats = loader.get_statistics()
loader.export_to_csv("benchmark_report.csv")
```

## Adding New Cases

```python
new_case = {
    "case_id": "pmf_case_011",
    "created_at": "2026-04-15T10:00:00Z",
    "created_by": "human",
    ...
}
loader.add_case(new_case, validate=True)  # appends to cases.jsonl
```

Or manually append a JSON line to any `.jsonl` file in `data/benchmark/`
and validate with:

```bash
python -m src.eval.benchmark_loader --validate
```
