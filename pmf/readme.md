# PMF Document Generator with LLM Evaluation
This project generates Plant Master File (PMF) documents using a retrieval-augmented LLM workflow and evaluates output quality using a rule-based benchmarking layer.
## Overview
The system has two major flows:
1. **PMF Generation Flow**
   - Upload source documents (ZIP)
   - Upload reference Excel
   - Split template into:
     - retrieval query segment (for vector DB search)
     - generation instruction segment (for LLM response generation)
   - Generate section-wise PMF output
   - Assemble final output document
2. **LLM Evaluation & Benchmarking Flow**
   - Capture per-run artifacts (prompts, retrieval context, generated section text)
   - Run rule-based evaluation (section and document level)
   - Store run results in an evaluation store
   - Visualize quality metrics in Streamlit dashboard
---
## Current Architecture
### Entry Point
- `app.py`
  - Streamlit UI
  - Main features:
    - `Plant Master File` generation
    - `Evaluation Dashboard`
### PMF Generation Core
- `src/document_analyzer/Extraction_module_PMF.py`
  - Handles template split and section loop
  - Builds retriever using extracted documents
  - Calls LLM generation via `handle_user_message(...)`
  - Assembles DOCX output
  - Triggers evaluation and stores results
### Section Generation
- `src/document_generate/dynamic_template_PMF.py`
  - Routes section requests via triage logic to:
    - Text agent
    - Table agent
    - Image agent
    - Static agent
  - Returns structured generation metadata for evaluation
### Retriever
- `src/document_retriever/Vector_db.py`
  - Builds embeddings with `sentence-transformers`
  - Stores FAISS index
  - Performs top-k similarity search
---
## Template Split Logic
Template content is split into two parts:
- **Part A (retrieval query half)**: used to query vector DB and fetch relevant chunks
- **Part B (generation half)**: sent to LLM with retrieved content to produce final section output
In `Extraction_module_PMF.py`:
- Section prompt is split using `@!`
  - `value_ls[0]` → instruction to LLM
  - `value_ls[1]` → retrieval query (if present)
---
## LLM Evaluation Framework (Phase 1)
This project currently uses a **custom rule-based evaluation framework** designed for compliance-style document generation.
### Why rule-based?
PMF output quality depends on:
- required section presence
- content completeness
- consistency with site metadata
- retrieval usage quality
These are easier to control and audit with deterministic checks.
### Evaluation modules
#### 1) Rule Configuration
- `src/eval/eval_config.py`
- Defines:
  - required section patterns
  - section-specific constraints
    - minimum length
    - required keywords
  - fallback rules for unspecified sections
#### 2) Scoring Engine
- `src/eval/eval_utils.py`
- Computes:
  - **Section-level checks**
    - non-empty content
    - minimum length pass/fail
    - keyword presence
    - site name consistency
  - **Document-level checks**
    - overall score (average of section scores)
    - missing required sections
    - retrieval coverage (non-static sections with retrieved docs)
#### 3) Evaluation Store
- `src/eval/eval_store.py`
- Persists:
  - full run artifacts + evaluation payload as JSON
  - run index entries in `data/eval_runs/index.jsonl`
---
## Run Artifacts Captured
For each generation run, the system stores:
- run metadata
  - timestamp
  - site name
  - template path
  - model name
  - final doc path
- section artifacts
  - section key
  - LLM prompt text
  - retrieval query
  - retrieved file paths
  - input text size
  - static/non-static flag
  - selected agent/tool metadata
  - generated section text
This makes each run reproducible and benchmarkable.
---
## Evaluation Dashboard
In Streamlit (`app.py` → `Evaluation Dashboard`):
- List historical runs
- Select a run and inspect:
  - overall score
  - section count
  - retrieval coverage %
  - missing required sections
  - per-section diagnostics
- View score trend across runs
---
## Data Paths
### Generated document artifacts
- `data/artifacts/generated output file/`
### Evaluation artifacts
- `data/eval_runs/`
  - `<run_id>.json` (full payload)
  - `index.jsonl` (run index)
---
## How to Use
1. Run app:
   ```bash
   streamlit run app.py
Open Plant Master File
Upload ZIP source docs
Upload reference Excel
Enter site name
Submit generation
Open Evaluation Dashboard
Select the latest run
Review score and per-section failures
Scoring Interpretation (Current)
100: all checks pass for a section/document
Lower scores indicate one or more failures:
missing text
too short content
missing mandatory keywords
site name mismatch
retrieval not used for non-static sections
Limitations (Phase 1)
Rule-based only (no human-reference similarity scoring yet)
No multi-model side-by-side comparison in one execution run
Section matching is pattern-based and may need domain tuning
Planned Enhancements
Weighted criticality by section type
richer compliance checks (regex/rulesets per region/product type)
CI regression gating
optional integration with external observability/evaluation platforms (LangSmith/Phoenix/TruLens/Promptfoo)
Security Note
Secrets should be loaded from environment variables (.env) and never hardcoded in source files.

Quick File Map
app.py
src/document_analyzer/Extraction_module_PMF.py
src/document_generate/dynamic_template_PMF.py
src/document_retriever/Vector_db.py
src/eval/eval_config.py
src/eval/eval_utils.py
src/eval/eval_store.py