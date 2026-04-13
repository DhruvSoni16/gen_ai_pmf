# Evaluation Dashboard User Guide

## Accessing the Dashboard

The evaluation dashboard is integrated into the main Streamlit app.
Click **"Evaluation Dashboard"** in the sidebar menu.

Alternatively, run the standalone version:

```bash
streamlit run app_eval_dashboard.py
```

---

## Dashboard Tabs

The dashboard has **4 tabs**: Overview, Run Details, Benchmark, and
Live Eval.

---

### Tab 1: Overview

Provides a high-level snapshot of evaluation health across all historical
runs.

**KPI Cards (top row)**
- **Latest Score** — The overall rule-based score (0-100%) from the most
  recent evaluation run.
- **Total Runs** — Number of evaluation runs stored in `data/eval_runs/`.
- **Latest Site** — The site name from the most recent run.
- **Score Delta** — Difference between the latest run and the previous
  run.  Positive = improvement, negative = regression.

**Regression Alerts**

If the latest score drops by more than 5 points compared to the previous
run, a red alert banner appears:

> REGRESSION ALERT: Score dropped by X.X points (Y.Y -> Z.Z).
> Investigate the latest run's section scores for the root cause.

A drop of 2-5 points shows a yellow warning.

**Score Trend Chart**

A line chart plotting `overall_score` over time across all runs.  Use
this to spot gradual quality drift after model updates or template
changes.

**Configuration Comparison**

When multiple template files or models have been used, a bar chart
compares mean scores grouped by configuration.  Error bars show standard
deviation.

---

### Tab 2: Run Details

Drill into a specific evaluation run to see section-by-section results.

**Run Selector**

A dropdown listing all runs by timestamp, site name, and score.  Select
a run to load its full results.

**Rule-Based Scores Panel**

Three metric cards:
- **Overall Score** — Average of all per-section rule scores (0-100%).
- **Sections** — Total number of sections evaluated.
- **Retrieval Coverage** — Percentage of non-static sections where the
  FAISS retriever found at least one source document.

A warning appears if any required sections (e.g. EXECUTIVE SUMMARY,
DEVICE DESCRIPTION) are missing from the generated document.

**Section Detail Table**

A table with one row per section showing:

| Column | Description |
|---|---|
| Section | Section key from the template |
| Score | Rule-based score (0-100) |
| Chars | Character count of the generated text |
| Min Chars | Minimum required by the rule config |
| Non-Empty | Pass/Fail — is the section non-empty? |
| Length | Pass/Fail — does it meet minimum character length? |
| Keywords | Pass/Fail — are all required keywords present? |
| Site Name | Pass/Fail — is the site name mentioned? |
| Missing KW | Comma-separated list of missing keywords |

**Section Score Heatmap**

A colour-coded grid showing pass (green) / fail (red) for each of the
four rule checks across all sections.  This reveals patterns at a glance,
such as "site name is consistently missing from table sections".

**LLM Metrics Panel**

If extended evaluation was run (checkbox enabled during generation),
shows:
- **BLEU** — Overlap of n-grams between generated and reference text.
- **ROUGE-L F1** — Longest common subsequence overlap.
- **BERTScore F1** — Semantic similarity using contextual embeddings.
- **Semantic Sim** — Cosine similarity of sentence embeddings.

---

### Tab 3: Benchmark

Overview of the benchmark dataset used for regression testing.

**Dataset Statistics**
- **Total Cases** — Number of benchmark cases loaded.
- **Agent Types** — Count of distinct section_type values.
- **Quality Tiers** — Count of difficulty levels.
- **Tags** — Count of distinct tags.

**Breakdown Expander**

Expandable section showing counts by agent type (text, table), by
difficulty (easy, medium, hard), and listing all section keys in the
dataset.

---

### Tab 4: Live Eval

An interactive tool for evaluating arbitrary text on the fly without
running the full generation pipeline.

**How to use:**
1. Paste generated text into the left text area.
2. Paste reference (ground truth) text into the right text area.
3. Optionally check "Include BERTScore" (adds ~2-5 seconds).
4. Click **Evaluate**.

**Results shown:**
- BLEU score
- ROUGE-L F1
- BERTScore F1 (if enabled)
- Semantic Similarity (cosine)
- Full JSON result in an expandable section

This is useful for quick spot-checks during development or for comparing
two versions of a section before committing to a full pipeline run.

---

## How to Interpret Grades

The composite score from `EvalSuite` is converted to a letter grade:

| Grade | Score Range | Meaning |
|---|---|---|
| **A** | >= 90 | Excellent. Production-ready, audit-ready quality. |
| **B** | 75 - 89 | Good. Minor improvements possible but acceptable. |
| **C** | 60 - 74 | Adequate. Noticeable gaps; revision recommended. |
| **D** | 45 - 59 | Below standard. Significant rework needed. |
| **F** | < 45 | Failing. Major quality issues; do not use as-is. |

The composite score is a weighted mean of available metrics:
- Rule-based score: **15%**
- BERTScore F1: **20%**
- LLM Judge normalized score: **40%**
- RAGAS composite score: **25%**

When a metric is unavailable (e.g. no reference text for BERTScore, or
judge not enabled), its weight is redistributed proportionally among the
remaining metrics.

---

## How to Read the Heatmap

The section score heatmap in the Run Details tab is a grid:

- **X-axis**: Section names (truncated to 30 characters)
- **Y-axis**: The four rule checks: Non-Empty, Min Length, Keywords,
  Site Name
- **Colour**: Green = pass, Red = fail
- **Cell text**: "Pass" or "Fail"

**What to look for:**

- A full red column means a section failed every check (likely empty or
  a generation failure).
- A red row across "Site Name" means the site name is systematically
  missing from all sections — check the template's `[Site Name]`
  placeholder substitution.
- A red row across "Keywords" suggests the retrieval step is not finding
  relevant source documents for those sections.

---

## How to Set Up Model Comparison

Use the `EvalSuite.compare_models()` method to produce a comparison
DataFrame:

```python
from healthark_eval import EvalSuite

suite = EvalSuite(task="pmf")

sections = [
    {
        "section_key": "EXECUTIVE SUMMARY",
        "section_instruction": "Write an executive summary.",
        "generated_outputs": {
            "gpt-4o": "GPT-4o generated text...",
            "claude-sonnet": "Claude generated text...",
        },
        "reference": "Reference text...",
    },
]

configs = [
    {"name": "gpt-4o", "provider": "azure_openai"},
    {"name": "claude-sonnet", "provider": "anthropic"},
]

df = suite.compare_models(sections, configs)
print(df[["section_key", "model_name", "composite_score", "grade"]])
```

The resulting DataFrame can be displayed in Streamlit using
`st.dataframe()` or plotted with `px.bar()`.

---

## How to Configure Regression Thresholds

Thresholds are defined in `tests/eval_thresholds.yaml`:

```yaml
document_level:
  min_overall_rule_score: 65.0
  min_bertscore_f1: 0.75
  min_judge_normalized_score: 60.0
  min_faithfulness: 0.70
  min_context_precision: 0.50
  min_answer_relevancy: 0.65
  min_ragas_score: 0.60
  max_missing_required_sections: 0

section_level:
  min_rule_score: 40.0
  min_bertscore_f1: 0.65
  min_judge_score: 50.0
  min_faithfulness: 0.60

regression:
  max_score_drop_vs_baseline: 5.0
```

**To adjust thresholds:**
1. Edit the YAML values.
2. Run `pytest tests/test_eval_regression.py -v` to verify current
   scores meet the new thresholds.
3. If a baseline update is needed, delete
   `data/benchmark/baseline_scores.json` and re-run the tests (the
   first run creates a new baseline, subsequent runs compare against it).

---

## Metric Tooltips (Plain English)

| Metric | What it measures |
|---|---|
| **BLEU** | How many exact word sequences (n-grams) from the reference appear in the generated text. Higher = more precise word-level overlap. Scale: 0-100. |
| **ROUGE-1 F1** | Overlap of individual words between generated and reference text, balancing precision and recall. Scale: 0-1. |
| **ROUGE-L F1** | Length of the longest common word subsequence between generated and reference, as a fraction. Captures sentence-level structure. Scale: 0-1. |
| **BERTScore F1** | Semantic similarity using contextual word embeddings (not just exact words). Captures meaning even when different words are used. Scale: 0-1. |
| **Semantic Similarity** | Cosine similarity between sentence-level embeddings of the generated and reference texts. Measures overall topical alignment. Scale: -1 to 1 (typically 0-1). |
| **Factual Accuracy** | LLM judge score: are all claims in the output supported by the source documents? Scale: 1-5. |
| **Regulatory Language** | LLM judge score: does the text use formal, precise regulatory terminology appropriate for PMF submissions? Scale: 1-5. |
| **Site Specificity** | LLM judge score: does the text correctly and consistently reference the specific manufacturing site? Scale: 1-5. |
| **Completeness** | LLM judge score: are all required sub-topics for this section type addressed? Scale: 1-5. |
| **Structural Coherence** | LLM judge score: is the content well-organised with logical flow and appropriate formatting? Scale: 1-5. |
| **Faithfulness** | Fraction of claims in the generated answer that are supported by the retrieved context. 1.0 = perfectly grounded, 0.0 = fully hallucinated. |
| **Context Precision** | Are the retrieved documents actually relevant? Measured as rank-weighted average precision. Higher = relevant docs ranked first. Scale: 0-1. |
| **Context Recall** | Fraction of reference claims that can be found in the retrieved context. Measures retrieval completeness. Scale: 0-1. |
| **Answer Relevancy** | Does the generated answer actually address the question/instruction? Measured via reverse-question cosine similarity. Scale: 0-1. |
| **RAGAS Score** | Harmonic mean of faithfulness, context precision, and answer relevancy. Overall RAG pipeline health. Scale: 0-1. |
| **Composite Score** | Weighted combination of rule score (15%), BERTScore (20%), judge score (40%), and RAGAS score (25%). Scale: 0-100. |
