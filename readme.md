# Plant Master File (PMF) Document Generator

> AI-powered regulatory document generation with LLM evaluation, RAG quality assessment, and real-time performance analytics — built for pharmaceutical and medical device manufacturing compliance.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Prerequisites](#prerequisites)
5. [Quickstart — Fork & Run](#quickstart--fork--run)
   - [1. Clone the Repository](#1-clone-the-repository)
   - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
   - [3. Install Dependencies](#3-install-dependencies)
   - [4. Configure Environment Variables](#4-configure-environment-variables)
   - [5. Create Required Directories](#5-create-required-directories)
   - [6. Run the Application](#6-run-the-application)
   - [7. Run the MLflow Tracking Server (Optional)](#7-run-the-mlflow-tracking-server-optional)
6. [Project Structure](#project-structure)
7. [How to Use](#how-to-use)
8. [Evaluation Framework](#evaluation-framework)
   - [Rule-Based Evaluation](#rule-based-evaluation)
   - [LLM-as-Judge](#llm-as-judge)
   - [DeepEval RAG Triad](#deepeval-rag-triad)
   - [Opik-Style Observability Metrics](#opik-style-observability-metrics)
   - [Composite Scoring](#composite-scoring)
9. [Performance Analytics](#performance-analytics)
10. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
11. [Evaluation Dashboard](#evaluation-dashboard)
12. [Template Format](#template-format)
13. [Data Paths](#data-paths)
14. [Troubleshooting](#troubleshooting)
15. [Environment Variables Reference](#environment-variables-reference)
16. [Known Limitations](#known-limitations)

---

## What This Project Does

A **Plant Master File (PMF)** is a regulated document required under EU GMP Annex 4 / ICH Q10 guidelines. It describes a manufacturing site's capabilities across 15–20 standard sections (Personnel, Premises, Equipment, Production, Quality Assurance, etc.).

Writing a PMF manually is time-consuming and error-prone. This system automates the process:

1. You upload your site documents (PDFs, DOCX, XLSX) as a ZIP file.
2. The system builds a FAISS vector index from those documents.
3. For each template section, it retrieves relevant context and calls **Azure OpenAI GPT-4o** to generate compliant content.
4. The generated document is evaluated using four independent evaluation layers.
5. Every run is logged to an evaluation store, and a live dashboard shows quality trends, failures, and improvement recommendations.

---

## Key Features

| Feature | Description |
|---|---|
| **Document Generation** | Section-by-section PMF generation from source documents via RAG + GPT-4o |
| **Multi-Agent Routing** | Triage agent routes each section to Text / Table / Image / Static handler |
| **Vector Search** | FAISS index with `all-mpnet-base-v2` sentence embeddings for context retrieval |
| **Rule-Based Evaluation** | Deterministic compliance checks (length, keywords, site name, section presence) |
| **LLM-as-Judge** | GPT-4o scored rubric: factual accuracy, regulatory language, completeness, coherence |
| **DeepEval RAG Triad** | Faithfulness, Contextual Precision, Answer Relevancy via claim-entailment methods |
| **Opik-Style Metrics** | Continuous LLM scoring for Hallucination, Answer Relevance, Regulatory Tone |
| **Composite Grade** | Weighted A–F grade: Rule 20% + Judge 55% + RAG Triad 25% |
| **Performance Analyzer** | Per-section and pipeline-level latency, failure detection, prioritised improvements |
| **MLflow Tracking** | Experiment logging with params, metrics, and tags per generation run |
| **Evaluation Dashboard** | 6-tab Streamlit dashboard: overview, heatmap, trends, model comparison, benchmark, performance |
| **SHA-256 Caching** | Evaluation results cached to avoid redundant LLM calls on repeat inputs |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          app.py (Streamlit UI)                       │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  PMF Generator       │  │  Evaluation Dashboard                │ │
│  │  - ZIP upload        │  │  - Run history                       │ │
│  │  - Excel reference   │  │  - Section heatmap                   │ │
│  │  - Site name         │  │  - Score trends                      │ │
│  └──────────┬───────────┘  │  - Model comparison                  │ │
│             │              │  - Benchmark management               │ │
│             ▼              │  - Performance analytics              │ │
│  ┌──────────────────────┐  └──────────────────────────────────────┘ │
│  │  Extraction_module_  │                                            │
│  │  PMF.py              │                                            │
│  │  (Orchestrator)      │                                            │
│  └──────────┬───────────┘                                            │
└─────────────┼───────────────────────────────────────────────────────┘
              │
     ┌────────▼─────────────────────────────────────────────┐
     │              Per-Section Generation Loop              │
     │                                                        │
     │  Template Section                                      │
     │       │                                                │
     │       ├──[retrieval query]──► DocumentRetriever       │
     │       │                       (FAISS + embeddings)    │
     │       │                              │                 │
     │       ├──[LLM instruction]──► LangChain / GPT-4o      │
     │       │       ◄──────────────────────┘                 │
     │       │                                                │
     │       └──► dynamic_template_PMF.py (Triage Agent)     │
     │               ├── Text Agent                           │
     │               ├── Table Agent                          │
     │               ├── Image Agent                          │
     │               └── Static Agent                         │
     └───────────────────────┬──────────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────────┐
              │           Evaluation Pipeline                 │
              │                                               │
              │  ① Rule-Based  (eval_utils.py)               │
              │  ② LLM Judge   (eval_judge.py)               │
              │  ③ RAG Triad   (eval_rag.py)                 │
              │  ④ Opik-Style  (eval_opik_style.py)          │
              │  ⑤ Performance (eval_performance.py)         │
              │  ⑥ MLflow      (eval_mlflow_tracker.py)      │
              └──────────────┬──────────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────────┐
              │          eval_store.py                        │
              │  data/eval_runs/{timestamp}_{site}.json       │
              │  data/eval_runs/index.jsonl                   │
              └─────────────────────────────────────────────┘
```

---

## Prerequisites

Before you begin, make sure you have the following installed and configured:

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.12.x | **Required.** Python 3.13 has NumPy 1.26.x build conflicts on Windows. |
| **pip** | latest | `python -m pip install --upgrade pip` |
| **Git** | any | For cloning the repository |
| **Microsoft Office** | Word 2016+ | Required on Windows for DOCX-to-PDF conversion via WIN32COM automation |
| **Tesseract OCR** | 5.x | Required for image-based sections. [Download for Windows](https://github.com/tesseract-ocr/tesseract) |
| **Azure OpenAI** | — | An active Azure OpenAI deployment of **gpt-4o** with a valid API key and endpoint |
| **MLflow** (optional) | 3.x | For experiment tracking UI. Installed automatically via requirements. |

> **Windows Note:** This project uses `pywin32` and `comtypes` for Office automation. These packages only work on Windows with Microsoft Word installed.

---

## Quickstart — Fork & Run

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/pmf
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.12 -m venv venv
source venv/bin/activate
```

> Make sure `python --version` reports **3.12.x** before creating the venv.

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r config/requirements.txt
```

**Post-install steps:**

```bash
# Install Playwright browsers (required for web scraping agent)
playwright install chromium

# Verify sentence-transformers installed correctly
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

> If you get a `torch` DLL conflict with `faiss-cpu` on Windows, install the CPU-only version of torch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Configure Environment Variables

Create a `.env` file in the `pmf/` directory (the same folder as `app.py`):

```bash
# macOS/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Then open `.env` and fill in your values:

```env
# ── Azure OpenAI (Required) ──────────────────────────────────────────
AZURE_KEY=your_azure_openai_api_key_here
AZURE_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_NAME=gpt-4o
AZURE_VERSION=2024-05-01-preview
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here   # same as AZURE_KEY

# ── OpenAI Direct (Optional — only if using non-Azure judge) ─────────
OPENAI_API_KEY=sk-proj-your_openai_key_here

# ── Opik (Optional — if using Opik Cloud instead of local scoring) ───
OPIK_API_KEY=your_opik_api_key_here
```

> **Security:** Never commit `.env` to version control. The `.gitignore` already excludes it.

**Getting your Azure credentials:**
1. Go to [Azure Portal](https://portal.azure.com) → Azure OpenAI resource
2. Under **Keys and Endpoint**, copy **Key 1** and the **Endpoint URL**
3. Under **Deployments**, note the deployment name (e.g., `gpt-4o`)
4. API version: use `2024-05-01-preview` unless your deployment requires otherwise

### 5. Create Required Directories

These directories are not tracked by Git and must be created manually before the first run:

```bash
# macOS / Linux
mkdir -p data/artifacts/Extracted_folder \
         data/eval_runs \
         data/eval_cache/rag \
         data/benchmark \
         vector_db \
         logs
```

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path `
  data/artifacts/Extracted_folder, `
  data/eval_runs, `
  data/eval_cache/rag, `
  data/benchmark, `
  vector_db, `
  logs
```

### 6. Run the Application

Open **two terminal windows** (both with the venv activated).

**Terminal 1 — Streamlit App:**

```bash
streamlit run app.py --server.maxUploadSize=1000
```

The app opens at `http://localhost:8501`.

**Terminal 2 — MLflow Tracking UI (optional but recommended):**

```bash
mlflow ui
```

The MLflow dashboard opens at `http://localhost:5000`.

### 7. Run the MLflow Tracking Server (Optional)

MLflow stores experiments locally in `./mlruns/`. To specify a custom location:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

---

## Project Structure

```
pmf/
├── app.py                              # Main Streamlit entry point
├── app_eval_dashboard.py               # Extended 6-tab evaluation dashboard
├── generate_docs.py                    # Script to regenerate technical DOCX docs
├── setup.py                            # Package setup (healthark_eval library)
│
├── config/
│   ├── requirements.txt                # Pinned Python dependencies
│   ├── configuration.json              # App-level settings
│   └── unpin.py                        # Utility to unpin versions
│
├── src/
│   ├── document_analyzer/
│   │   ├── Extraction_module_PMF.py    # Core orchestrator: template split, loop, eval trigger
│   │   ├── Extraction_module.py        # Legacy extraction module
│   │   ├── text.py                     # GPT-based text extraction
│   │   ├── table.py                    # Table extraction with LLM refinement
│   │   ├── image.py                    # Image extraction with OCR fallback
│   │   ├── json_converter.py           # JSON conversion utilities
│   │   └── contents.py                 # TOC refresh and heading extraction
│   │
│   ├── document_generate/
│   │   ├── dynamic_template_PMF.py     # Triage agent: routes to Text/Table/Image/Static
│   │   ├── dynamic_template.py         # Legacy triage module
│   │   ├── doc_generate.py             # DOCX assembly utilities
│   │   ├── Assembling_appendix_PMF.py  # Appendix assembly + DOCX-to-PDF conversion
│   │   └── Assembling_appendix.py      # Legacy appendix assembly
│   │
│   ├── document_ingestion/
│   │   ├── Input_files_loading.py      # ZIP extraction, file type detection, fuzzy matching
│   │   ├── data_collection.py          # MSG/DOCX/PDF/XLSX extraction, WIN32COM
│   │   └── paths.py                    # Path configuration
│   │
│   ├── document_retriever/
│   │   └── Vector_db.py                # FAISS DocumentRetriever (all-mpnet-base-v2)
│   │
│   └── eval/
│       ├── eval_config.py              # Rule definitions per PMF section
│       ├── eval_utils.py               # Section + document scoring logic
│       ├── eval_store.py               # JSON persistence for evaluation runs
│       ├── eval_metrics.py             # Lexical (BLEU/ROUGE) + semantic (BERTScore)
│       ├── eval_judge.py               # PMFJudge: LLM-based factual/regulatory scoring
│       ├── eval_rag.py                 # RAGEvaluator: Faithfulness, Precision, Relevancy
│       ├── eval_opik_style.py          # Opik-style: Hallucination, Relevance, Tone
│       ├── eval_mlflow_tracker.py      # MLflow experiment logging
│       ├── eval_performance.py         # Latency analysis + failure detection
│       └── benchmark_loader.py         # Benchmark dataset manager
│
├── healthark_eval/                     # Installable evaluation library
│   └── suite.py                        # EvalSuite: parallel multi-metric runner
│
├── templates/
│   ├── PMF_Template_With_vector_DB.docx           # Primary PMF template
│   └── PMF_Template_With_vector_DB - Copy.docx    # Working copy used by app
│
├── data/
│   ├── artifacts/
│   │   ├── Extracted_folder/           # Temp: extracted ZIP contents (cleared per run)
│   │   └── generated output file/      # Final generated PMF DOCX + PDF
│   ├── eval_runs/
│   │   ├── index.jsonl                 # Run index (one JSON object per line)
│   │   └── {timestamp}_{site}.json     # Full run payload per generation
│   ├── eval_cache/rag/                 # SHA-256 cached evaluation results
│   └── benchmark/
│       ├── seed_cases.jsonl            # Expert-annotated benchmark test cases
│       └── baseline_scores.json        # Reference scores for regression testing
│
├── docs/
│   ├── PMF_LLM_Evaluation_Framework_Technical_Documentation.docx
│   ├── eval_framework_overview.md
│   ├── benchmark_guide.md
│   ├── dashboard_user_guide.md
│   └── quick_start.md
│
├── vector_db/                          # Persisted FAISS indexes (per site)
├── logs/app.log                        # Application log file
├── static/logo1.png                    # Sidebar logo
└── .env                                # Environment variables (not committed)
```

---

## How to Use

### Generating a PMF Document

1. Open the app at `http://localhost:8501`
2. In the sidebar, click **Plant Master File**
3. Upload a **ZIP file** containing your source documents (PDFs, DOCX, XLSX, or images)
4. Upload a **Reference Excel file** containing site-specific metadata
5. Enter the **Site Name** (e.g., `Thermo Fisher Scientific - Cincinnati`)
6. Click **Submit**
7. Wait for generation to complete — typically 3–8 minutes depending on document size and section count
8. Download the generated `.docx` file using the link that appears
9. Review the **quality grade card** shown below the download button

### Reviewing Evaluation Results

1. In the sidebar, click **Evaluation Dashboard**
2. Select a run from the dropdown
3. Navigate across the six tabs for different views

---

## Evaluation Framework

The system runs four independent evaluation layers on every generated document.

### Rule-Based Evaluation

**File:** `src/eval/eval_utils.py`, `src/eval/eval_config.py`

Deterministic compliance checks per section:

| Check | Description | Penalty |
|---|---|---|
| Non-empty | Section has content | −100 if empty |
| Minimum length | Content meets `min_chars` threshold | −40 pts |
| Required keywords | Domain-specific terms are present | −30 pts (proportional to missing) |
| Site name | Site name appears in content | −15 pts |
| Retrieval usage | Non-static section used retrieved chunks | −15 pts |

Document-level score = average of section scores. Missing required sections are flagged separately.

### LLM-as-Judge

**File:** `src/eval/eval_judge.py`

`PMFJudge` sends each section to GPT-4o with a structured rubric scoring five criteria:

| Criterion | Weight | What It Measures |
|---|---|---|
| Factual Accuracy | 30% | Claims match source documents |
| Regulatory Language | 25% | GMP/ICH-compliant terminology |
| Site Specificity | 20% | Site name and context correctly referenced |
| Completeness | 15% | All expected sub-topics covered |
| Coherence | 10% | Logical structure and readability |

Final judge score = weighted sum normalised to 0–100.

### DeepEval RAG Triad

**File:** `src/eval/eval_rag.py`

Three retrieval quality metrics computed per section:

| Metric | Method | Measures |
|---|---|---|
| **Faithfulness** | Claim extraction → NLI entailment check | Are claims in the output supported by retrieved chunks? |
| **Contextual Precision** | Rank-weighted MAP over relevance labels | Are the most relevant chunks ranked highest? |
| **Answer Relevancy** | Reverse question generation + cosine similarity | Does the output actually answer the retrieval query? |

All three are normalised to 0–1. RAG Triad score = mean of the three.

### Opik-Style Observability Metrics

**File:** `src/eval/eval_opik_style.py`

Continuous LLM scoring — a single GPT-4o call per section returns:

| Metric | Range | Measures |
|---|---|---|
| **Hallucination Score** | 0–1 (lower is better) | Proportion of unsupported claims |
| **Answer Relevance** | 0–1 | How directly the output answers the query |
| **Regulatory Tone** | 0–1 | Formal GMP-compliant language quality |

The dashboard displays **Groundedness** = `1 − Hallucination Score`.

### Composite Scoring

```
Composite = (Rule × 0.20) + (Judge × 0.55) + (RAG Triad × 0.25)
```

| Score Range | Grade | Label |
|---|---|---|
| ≥ 90 | A | Excellent |
| ≥ 75 | B | Good Quality |
| ≥ 60 | C | Acceptable |
| ≥ 45 | D | Needs Improvement |
| < 45 | F | Poor |

---

## Performance Analytics

**File:** `src/eval/eval_performance.py`

Every section is timed across three phases:

| Phase | What Is Measured |
|---|---|
| **Retrieval** | FAISS vector search + document loading |
| **Generation** | LLM API call latency (wall-clock) |
| **Evaluation** | All four eval layers combined |

**Failure Detection** — the `PerformanceAnalyzer` flags sections that breach thresholds:

| Failure Type | Severity | Trigger Condition |
|---|---|---|
| `error` | Critical | Section threw an exception during generation |
| `missing_chunks` | Warning | No retrieval context found for a non-static section |
| `low_score` | Warning | Rule score < 50 |
| `hallucination` | Warning | Faithfulness < 0.25 or Hallucination score > 0.50 |
| `low_tone` | Info | Regulatory Tone < 0.50 |
| `low_relevance` | Info | Answer Relevance < 0.40 |

**Improvement recommendations** are generated in two formats:
- **Plain English** — for document reviewers and project managers who need actionable guidance without technical details
- **Technical** — for developers, with specific thresholds, module names, and fix strategies

Access the full performance report in the **Performance tab** of the Evaluation Dashboard.

---

## MLflow Experiment Tracking

Each generation run logs the following to MLflow automatically:

```
Params:   site_name, template_file, section_count, model_name
Metrics:  rule_score, judge_score, rag_faithfulness, rag_precision,
          rag_relevancy, opik_hallucination, opik_relevance,
          opik_tone, composite_score, total_pipeline_ms
Tags:     run_id, grade, framework
```

Experiments are stored locally in `./mlruns/`. View them at `http://localhost:5000` after starting `mlflow ui`. A direct link to the specific run appears in the app immediately after generation completes.

---

## Evaluation Dashboard

**File:** `app_eval_dashboard.py`

Six tabs provide different views into run quality:

| Tab | What It Shows |
|---|---|
| **Run Overview** | Grade badge, 9 metric cards (Rule, Judge, RAG Triad, Opik), MLflow run link |
| **Section Heatmap** | Multi-metric heatmap across all sections; click any cell for drill-down details |
| **Trend Analysis** | Score time-series charts; regression alerts when score drops more than 10 points |
| **Model Comparison** | Upload two run files; side-by-side radar chart and delta table |
| **Benchmark** | Seed case table, add/export cases, run the regression test suite |
| **Performance** | Timing cards, donut chart, bar charts (slowest sections first), failures table, improvement expanders |

---

## Template Format

The PMF template is a DOCX file. Sections are separated by `$`. Each section follows this format:

```
SECTION_KEY
Generation instruction for the LLM @! Retrieval query for vector search
```

- Content **before** `@!` is the **LLM prompt** — sent to GPT-4o to generate the section
- Content **after** `@!` is the **retrieval query** — used to search the FAISS vector index
- If there is no `@!`, the section is **static** and no LLM call is made

Section key naming conventions control routing:

| Key Pattern | Routes To |
|---|---|
| Contains `TABLE` | Table Agent |
| Contains `IMAGE` | Image Agent |
| Starts with `STATIC` | Static Agent (no LLM) |
| Anything else | Text Agent |

---

## Data Paths

| Path | Purpose |
|---|---|
| `data/artifacts/Extracted_folder/` | Temp extraction directory — cleared before each run |
| `data/artifacts/generated output file/` | Final generated DOCX and PDF files |
| `data/eval_runs/index.jsonl` | One JSON object per line: `{timestamp, site_name, overall_score, run_file}` |
| `data/eval_runs/{ts}_{site}.json` | Full run payload including all evaluation scores and section artifacts |
| `data/eval_cache/rag/` | SHA-256 keyed evaluation cache (avoids re-scoring identical inputs) |
| `data/benchmark/seed_cases.jsonl` | Expert-annotated benchmark test cases |
| `vector_db/` | Persisted FAISS index files |
| `logs/app.log` | Application log at INFO level |
| `mlruns/` | MLflow local experiment store |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'docx'`**
```bash
pip install python-docx
```

**`ModuleNotFoundError: No module named 'win32com'`**
```bash
pip install pywin32
python venv/Scripts/pywin32_postinstall.py -install
```

**Dashboard shows no runs / empty index**

Generate at least one PMF document first. The evaluation store populates automatically after each successful run.

**FAISS index empty / no retrieval results**

Check that your ZIP contains readable PDF, DOCX, or XLSX files. Review `logs/app.log` for extraction errors.

**Azure OpenAI `AuthenticationError`**

- Verify `AZURE_KEY` and `AZURE_ENDPOINT` in your `.env`
- Confirm the deployment name in `AZURE_NAME` matches your Azure portal deployment
- API version default: `2024-05-01-preview`

**`torch` DLL conflict with `faiss-cpu` on Windows**
```bash
pip uninstall torch faiss-cpu
pip install faiss-cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Streamlit upload size error with large ZIP files**
```bash
streamlit run app.py --server.maxUploadSize=2000
```

**MLflow `protobuf` warning at startup**

Known compatibility note between `mlflow 3.x` and `protobuf 6.x`. Non-fatal — all tracking functions work correctly.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `AZURE_KEY` | Yes | Azure OpenAI API key |
| `AZURE_ENDPOINT` | Yes | Azure OpenAI endpoint URL (e.g. `https://xxx.openai.azure.com/`) |
| `AZURE_NAME` | Yes | Azure deployment name (e.g. `gpt-4o`) |
| `AZURE_VERSION` | Yes | API version string (e.g. `2024-05-01-preview`) |
| `AZURE_OPENAI_API_KEY` | Yes | Same as `AZURE_KEY` — required by some SDK call paths |
| `OPENAI_API_KEY` | No | Direct OpenAI key (only for non-Azure judge fallback) |
| `OPIK_API_KEY` | No | Opik Cloud API key (not required for local Opik-style scoring) |

---

## Known Limitations

- **Windows only** — `pywin32` and `comtypes` are required for DOCX-to-PDF conversion. On macOS/Linux, PDF output is skipped but DOCX download still works.
- **Python 3.12 only** — Python 3.13 introduces NumPy build incompatibilities with the current pinned dependency set.
- **OpenAI 0.28 pinned** — Several extraction modules use the legacy `ChatCompletion` API. Do not upgrade `openai` to 1.x or 2.x without migrating those call sites.
- **BERTScore disabled on Windows** — DLL conflicts between `torch` and `faiss-cpu` cause BERTScore to be skipped automatically; cosine similarity is used as the fallback.
- **Extended evaluation adds latency** — Running all four evaluation layers adds 60–180 seconds per run. Rule-based scores appear immediately; extended scores populate after async completion.
