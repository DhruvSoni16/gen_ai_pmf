# Evaluation Framework Overview

## Purpose

The Healthark GenAI Evaluation Framework (Initiative 4) provides a
production-grade, multi-layer evaluation system for LLM-generated Plant
Master File (PMF) regulatory documents.  It combines rule-based checks,
lexical metrics, semantic metrics, LLM-as-judge scoring, and RAG quality
metrics into a single composite score with letter grades.

---

## Architecture Diagram

```
 USER INPUT                          GENERATION PIPELINE
 ──────────                          ───────────────────
 ZIP (PDFs, DOCX, XLSX)
 + Excel reference                  ┌───────────────────┐
 + Site name                        │  FAISS Vector DB   │
       │                            │  (all-mpnet-base-v2,│
       ▼                            │   768-dim, HNSW)   │
 ┌─────────────┐    top-5 docs      └────────┬──────────┘
 │  Streamlit  │───────────────────────────►  │
 │   app.py    │                              ▼
 └──────┬──────┘                  ┌───────────────────────┐
        │                         │  Triage Agent (GPT-4o) │
        │  ZIP extract            │  ┌──────┐ ┌──────┐    │
        │  + template parse       │  │ Text │ │Table │    │
        ▼                         │  │Agent │ │Agent │    │
 data/artifacts/                  │  └──────┘ └──────┘    │
 Extracted_folder/                │  ┌──────┐ ┌──────┐    │
                                  │  │Image │ │Static│    │
                                  │  │Agent │ │Agent │    │
                                  └──────┬────────────────┘
                                         │
                                         ▼ generated_text (per section)
                                         │
        ┌────────────────────────────────┤
        │                                │
        ▼                                ▼
 ┌──────────────┐              ┌──────────────────┐
 │  Rule-Based  │              │  Extended Eval    │
 │  Evaluation  │              │  (opt-in via UI)  │
 │              │              │                   │
 │ eval_config  │              │  ┌─────────────┐  │
 │ eval_utils   │              │  │eval_metrics  │  │
 │ eval_store   │              │  │ BLEU, ROUGE  │  │
 │              │              │  │ BERTScore    │  │
 │ Checks:      │              │  └─────────────┘  │
 │ - non-empty  │              │  ┌─────────────┐  │
 │ - min length │              │  │eval_judge    │  │
 │ - keywords   │              │  │ PMFJudge     │  │
 │ - site name  │              │  │ 5-criterion  │  │
 │              │              │  │ rubric       │  │
 │ Score: 0-100 │              │  └─────────────┘  │
 └──────┬───────┘              │  ┌─────────────┐  │
        │                      │  │eval_rag      │  │
        │                      │  │ Faithfulness │  │
        │                      │  │ Ctx Precision│  │
        │                      │  │ Ctx Recall   │  │
        │                      │  │ Answer Relev.│  │
        │                      │  └─────────────┘  │
        │                      └────────┬──────────┘
        │                               │
        ▼                               ▼
 ┌──────────────────────────────────────────────┐
 │              healthark_eval.EvalSuite          │
 │                                                │
 │  Composite = 0.15*rule + 0.20*BERTScore       │
 │             + 0.40*judge + 0.25*RAGAS          │
 │                                                │
 │  → EvalResult(grade, composite_score, summary) │
 └───────────────────────┬────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Streamlit          │
              │   Eval Dashboard     │
              │                      │
              │   - Overview tab     │
              │   - Run Details tab  │
              │   - Benchmark tab    │
              │   - Live Eval tab    │
              └──────────────────────┘
```

---

## Module Descriptions

### src/eval/eval_config.py
Rule-based evaluation configuration.  Defines required section patterns,
per-section keyword requirements, and minimum character lengths.  Used by
`eval_utils.py` to score sections on a 0-100 scale across four boolean
checks (non-empty, min length, keywords, site name).

### src/eval/eval_utils.py
Scoring engine.  `score_section()` evaluates one section against rules;
`score_document()` aggregates across all sections; `evaluate_run()` wraps
everything with run metadata.

### src/eval/eval_store.py
JSON persistence layer.  Stores evaluation runs as individual JSON files
in `data/eval_runs/` with a line-delimited index file for fast listing.

### src/eval/eval_metrics.py
Lexical and semantic metrics: BLEU (via sacrebleu), ROUGE (via
rouge-score), BERTScore (via bert-score), and cosine similarity (via
sentence-transformers).  All results are JSON-serializable floats rounded
to 4 decimal places.  Graceful degradation when libraries are missing.

### src/eval/eval_judge.py
LLM-as-judge module.  `PMFJudge` uses Claude or GPT-4o to evaluate
sections on a 5-criterion rubric: factual accuracy (0.30), regulatory
language (0.25), site specificity (0.20), completeness (0.15), structural
coherence (0.10).  Returns weighted scores (0-5), normalized scores
(0-100), strengths/weaknesses, and critical issues.  Responses are cached
by SHA-256 hash with rubric version for invalidation.

### src/eval/eval_rag.py
RAGAS-inspired RAG evaluation.  `RAGEvaluator` computes four metrics:
faithfulness (claim-level entailment), context precision (rank-weighted
average precision), context recall (reference claim attribution), and
answer relevancy (reverse-question cosine similarity).  Falls back to
keyword-overlap heuristics when no LLM client is available.

### src/eval/benchmark_loader.py
Loads, validates, filters, and manages benchmark cases from JSONL files
in `data/benchmark/`.  Supports filtering by section_type, difficulty,
tags, and site_name.  Can add new cases, export to CSV, and compute
dataset statistics.

### healthark_eval/suite.py
Primary public API.  `EvalSuite` orchestrates all metric modules and
returns structured `EvalResult` / `DocumentEvalResult` objects with
composite scores and letter grades.  `run_document()` batches BERTScore
across all sections in a single call for performance.

### healthark_eval/config.py
Task configuration registry.  `TaskConfig` holds domain-specific settings
(required sections, thresholds, model choices).  `PMF_TASK_CONFIG` is the
built-in configuration for PMF regulatory documents.

---

## How Metrics Relate

```
Rule Score (0-100)           ─┐
                               │  composite_score = weighted mean
BERTScore F1 (0-1 → ×100)   ─┤  of available metrics (null ones
                               │  excluded, weights renormalized)
Judge Normalized (0-100)     ─┤
                               │  Weights:
RAGAS Score (0-1 → ×100)    ─┘    rule=0.15, bert=0.20,
                                    judge=0.40, ragas=0.25

Grade:  A (>=90)  B (>=75)  C (>=60)  D (>=45)  F (<45)
```

The rule score catches structural issues (empty sections, missing
keywords).  BERTScore captures semantic similarity to a reference.  The
LLM judge evaluates regulatory quality holistically.  RAGAS metrics
assess RAG pipeline health (are retrieved docs faithful and relevant?).

---

## How to Add a New Task Type

1. Create `healthark_eval/tasks/your_task.py`:

```python
from healthark_eval.tasks.base import BaseTaskConfig

YOUR_TASK_CONFIG = BaseTaskConfig(
    task_name="your_task",
    required_sections=["INTRODUCTION", "METHODOLOGY"],
    min_section_chars=80,
    judge_rubric="your_domain",
    bertscore_model="distilbert-base-uncased",
    faithfulness_threshold=0.65,
    context_precision_threshold=0.45,
    composite_threshold=60.0,
)
```

2. Register it in `healthark_eval/config.py`:

```python
from healthark_eval.tasks.your_task import YOUR_TASK_CONFIG
TASK_REGISTRY["your_task"] = YOUR_TASK_CONFIG
```

3. Use it:

```python
suite = EvalSuite(task="your_task")
```

---

## How to Extend the Rubric for a New Domain

The judge rubric lives in `src/eval/eval_judge.py` as `JUDGE_RUBRIC`.
To create a domain-specific rubric:

1. Define your criteria with weights summing to 1.0:

```python
CUSTOM_RUBRIC = {
    "accuracy": {"weight": 0.35, "description": "...", "levels": {5: "...", ...}},
    "clarity":  {"weight": 0.25, "description": "...", "levels": {5: "...", ...}},
    ...
}
```

2. Create a subclass of `PMFJudge` that overrides the rubric:

```python
class CustomJudge(PMFJudge):
    RUBRIC_VERSION = "custom_v1.0"
    # Override _build_user_prompt to inject your rubric
```

3. Wire it through `EvalSuite._get_judge()` by passing a custom
   `judge_rubric` in your `TaskConfig`.
