# Quick Start Guide

## Installation

From the project root (`pmf/`):

```bash
# Install the healthark_eval package in development mode
pip install -e .

# Or install dependencies directly
pip install sacrebleu rouge-score bert-score sentence-transformers anthropic pandas pyyaml pytest
```

---

## Minimal Working Example

```python
from healthark_eval import EvalSuite

# Create a suite — disable judge and RAG to avoid needing an API key
suite = EvalSuite(
    task="pmf",
    run_judge=False,
    run_rag=False,
    run_semantic=False,  # set True if BERTScore is needed (adds ~2-5s)
)

result = suite.run(
    generated="The Bangalore site manufactures single-use bioprocessing assemblies.",
    retrieved=["Site description document text about Bangalore manufacturing."],
    reference="The manufacturing site located in Bangalore produces assemblies.",
    section_key="SITE DESCRIPTION",
    site_name="Bangalore Site",
)

print(result.grade)            # "B"
print(result.composite_score)  # 75.0
print(result.summary)          # "Section 'SITE DESCRIPTION' achieved grade B (75.0/100)."
```

### With all metrics enabled (requires API key)

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

suite = EvalSuite(
    task="pmf",
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-6",
    run_lexical=True,
    run_semantic=True,
    run_judge=True,
    run_rag=True,
)

result = suite.run(
    generated="The Bangalore Site is a multi-product manufacturing facility...",
    retrieved=[
        "The Bangalore site produces single-use assemblies...",
        "ISO 13485 certification is maintained...",
    ],
    reference="The Bangalore manufacturing site produces single-use bioprocessing...",
    section_key="EXECUTIVE SUMMARY",
    section_instruction="Write an executive summary for the PMF.",
    site_name="Bangalore Site",
)

print(f"Grade: {result.grade}")
print(f"Composite: {result.composite_score}")
print(f"Rule Score: {result.rule_score}")
print(f"BLEU: {result.lexical_scores.get('bleu') if result.lexical_scores else 'N/A'}")
print(f"Judge: {result.judge_scores.get('normalized_score') if result.judge_scores else 'N/A'}")
print(f"Faithfulness: {result.rag_scores.get('faithfulness') if result.rag_scores else 'N/A'}")
```

---

## Evaluate a Full Document

```python
sections = [
    {
        "section_key": "EXECUTIVE SUMMARY",
        "generated_text": "The Bangalore site is a manufacturing facility...",
        "reference": "The site produces single-use assemblies...",
        "site_name": "Bangalore Site",
    },
    {
        "section_key": "DEVICE DESCRIPTION",
        "generated_text": "The site manufactures sterile connectors...",
        "reference": "Sterile connectors and bioreactor assemblies...",
        "site_name": "Bangalore Site",
    },
]

doc_result = suite.run_document(sections)

print(f"Overall Grade: {doc_result.overall_grade}")
print(f"Mean Composite: {doc_result.mean_composite}")
print(f"Grade Distribution: {doc_result.grade_distribution}")
print(f"Lowest Sections: {doc_result.lowest_sections}")

for r in doc_result.section_results:
    print(f"  {r.section_key}: {r.grade} ({r.composite_score})")
```

---

## Run the Benchmark

```python
suite = EvalSuite(task="pmf", run_judge=False, run_rag=False)

df = suite.run_benchmark(benchmark_dir="data/benchmark")
print(df[["case_id", "section_key", "composite_score", "grade"]])
print(f"\nMean composite: {df['composite_score'].mean():.1f}")
```

---

## Run the Test Suite

```bash
# Fast tests only (no API calls)
pytest tests/test_eval_regression.py -v

# Include slow tests that call LLM APIs
EVAL_RUN_SLOW=1 pytest tests/test_eval_regression.py -v

# Generate JUnit XML report
pytest tests/test_eval_regression.py -v --junitxml=reports/eval_report.xml
```

---

## Save and Load Results

```python
# Save
path = suite.save_results(result)
print(f"Saved to: {path}")

# Load
import json
with open(path) as f:
    data = json.load(f)
print(data["grade"], data["composite_score"])
```

---

## Interpreting Results

| Grade | Action |
|---|---|
| **A** (>= 90) | Production-ready. No changes needed. |
| **B** (>= 75) | Good quality. Minor polish optional. |
| **C** (>= 60) | Acceptable but gaps exist. Review flagged sections. |
| **D** (>= 45) | Below standard. Investigate retrieval quality and prompts. |
| **F** (< 45) | Failing. Check for empty sections, retrieval failures, or hallucination. |

**When a section scores poorly:**

1. Check `result.rule_score` — is the section empty or missing keywords?
2. Check `result.rag_scores["faithfulness"]` — is the output hallucinating?
3. Check `result.judge_scores["scores"]` — which criterion scored lowest?
4. Check `result.rag_scores["context_precision"]` — are the retrieved docs
   relevant, or is the retrieval query poor?
