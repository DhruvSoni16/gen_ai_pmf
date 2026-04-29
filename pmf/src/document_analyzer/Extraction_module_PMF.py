import io
import os
import time
import logging
from docx import Document
import streamlit as st
from datetime import datetime
from docxcompose.composer import Composer
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from src.document_generate.dynamic_template_PMF import handle_user_message
from src.document_ingestion.data_collection import data_extraction, convert_all_doc_to_docx_in_folder, extract_text_from_xlsx
from src.document_analyzer.json_converter import extract_text_from_word
from src.document_analyzer.contents import refresh_toc_with_word, extract_headings_with_tables
from src.document_retriever.Vector_db import DocumentRetriever
from src.eval.eval_config import get_eval_rules
from src.eval.eval_utils import evaluate_run
from src.eval.eval_store import save_eval_run
from healthark_eval.suite import EvalSuite



def _derive_section_title(key: str, value: str) -> str:
    """Extract a human-readable section title from a template value.

    Template sections have a type prefix as the key (e.g. "Text: -",
    "Static_text:-") while the actual section heading is the first
    meaningful line of the value (e.g. "1.0 GENERAL INFORMATION:").
    This is used to produce meaningful section keys for the eval dashboard.
    """
    for line in (value or "").split("\n"):
        clean = line.strip().replace("**", "").replace("*", "").strip()
        if clean and len(clean) > 3:
            return clean.rstrip(":").strip()[:70]
    return key  # fallback to the type key if value is empty


# Function to split the template text into two parts based on the section header
def Template_to_list(text):
    if text is None:
        return [], []     # Handle the case where text is None by returning empty lists

    sections = [section.strip() for section in text.split('$') if section.strip()]
    split_index = None
    for i, item in enumerate(sections):
        if "DEVICE DESCRIPTION & PRODUCT SPECIFICATION".lower() in item.lower():
            split_index = i
            break

    if split_index is not None:
        part1 = sections[:split_index]
        part2 = sections[split_index:]
    else:
        part1 = sections
        part2 = []

    return part1, part2


# Convert the list of strings into a dictionary with unique keys
def convert_dict(list):

    result_dict = {}
    key_count = {}

    for item in list:
    # Split into key and value at first newline
        parts = item.split('\n', 1)
        if len(parts) == 2:
            key, value = parts[0].strip(), parts[1].strip()
        else:
            # If no newline found, use whole as key and empty value
            key, value = parts[0].strip(), ""

        # Handle duplicate keys by counting
        if key in key_count:
            key_count[key] += 1
            new_key = f"{key}_{key_count[key]}"
        else:
            key_count[key] = 1
            new_key = key

        result_dict[new_key] = value
    return result_dict



def _build_opik_scorer(run_artifacts):
    """Construct the OpikStyleScorer using Azure creds from run_artifacts."""
    from src.eval.eval_opik_style import OpikStyleScorer
    from openai import AzureOpenAI

    key = run_artifacts.get("_azure_key", "")
    endpoint = run_artifacts.get("_azure_endpoint", "")
    version = run_artifacts.get("_azure_version", "2024-06-01")
    name = run_artifacts.get("model_name", "gpt-4o")

    if not (key and endpoint):
        return None

    try:
        client = AzureOpenAI(api_key=key, api_version=version, azure_endpoint=endpoint)
        return OpikStyleScorer(llm_client=client, model=name)
    except Exception as exc:
        logging.warning("Opik scorer init failed: %s", exc)
        return None


def _load_retrieved_chunks(retrieved_paths):
    """Extract text from retrieved files using the same extractor as generation.

    Uses data_extraction() so PDFs/docx/xlsx are parsed properly — reading them
    as UTF-8 yields binary garbage, which makes the faithfulness judge score 0
    even when the docs contain the supporting evidence.
    """
    chunks = []
    for p in retrieved_paths:
        try:
            text = data_extraction([p])
            if text:
                chunks.append(text[:8000])
        except Exception as exc:
            logging.warning("Failed to extract text from '%s': %s", p, exc)
    return chunks


def _run_deepeval_for_section(eval_suite, section, generated, retrieved_chunks, site_name, section_results):
    """Run DeepEval (Judge + RAG Triad) on one section; mutate section + section_results."""
    try:
        result = eval_suite.run(
            generated=generated,
            retrieved=retrieved_chunks,
            reference="",
            section_key=section.get("section_key", ""),
            section_instruction=section.get("prompt_text", ""),
            site_name=site_name,
        )
        section["extended_eval"] = result.to_dict()
        section_results.append(result)
    except Exception as exc:
        logging.warning("DeepEval failed for '%s': %s", section.get("section_key", ""), exc)


def _run_opik_for_section(opik_scorer, section, generated, merged_context):
    """Run Opik-style scoring on one section; mutate section in place."""
    if not opik_scorer:
        return
    try:
        section["opik_eval"] = opik_scorer.evaluate_section(
            section_key=section.get("section_key", ""),
            output=generated,
            context=merged_context,
            instruction=section.get("prompt_text", ""),
        )
    except Exception as exc:
        logging.warning("Opik scoring failed for '%s': %s", section.get("section_key", ""), exc)


def _stamp_section_eval_timing(section, sec_eval_ms):
    """Update the section's timing dict with eval_ms and total_ms."""
    if "timing" not in section or section["timing"] is None:
        section["timing"] = {}
    section["timing"]["eval_ms"] = sec_eval_ms
    section["timing"]["total_ms"] = round(
        (section["timing"].get("retrieval_ms") or 0)
        + (section["timing"].get("generation_ms") or 0)
        + sec_eval_ms,
        1,
    )


def _evaluate_one_section(section, eval_suite, opik_scorer, site_name, section_results):
    """Run all per-section eval steps. Returns False if section was skipped."""
    generated = section.get("generated_text", "")
    if not generated.strip():
        return False

    retrieved_chunks = _load_retrieved_chunks(section.get("retrieved_paths", []))
    merged_context = "\n\n".join(retrieved_chunks)

    sec_start = time.perf_counter()
    _run_deepeval_for_section(eval_suite, section, generated, retrieved_chunks, site_name, section_results)
    _run_opik_for_section(opik_scorer, section, generated, merged_context)
    _stamp_section_eval_timing(section, round((time.perf_counter() - sec_start) * 1000, 1))
    return True


def _mean(vals):
    """Mean of non-None values, rounded to 4 decimals; None if empty."""
    clean = [float(v) for v in vals if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _aggregate_deepeval(section_results):
    """Aggregate DeepEval section results into doc-level means + grade distribution."""
    from collections import Counter

    if not section_results:
        return {"mean_composite": 0.0, "grade_dist": {},
                "mean_judge": None, "mean_rag": None, "mean_faith": None}

    composites = [r.composite_score for r in section_results]
    return {
        "mean_composite": round(sum(composites) / len(composites), 2),
        "grade_dist": dict(Counter(r.grade for r in section_results)),
        "mean_judge": _mean([
            r.judge_scores.get("normalized_score")
            for r in section_results
            if r.judge_scores and not r.judge_scores.get("judge_error")
        ]),
        "mean_rag": _mean([
            r.rag_scores.get("rag_triad_score") or r.rag_scores.get("ragas_score")
            for r in section_results if r.rag_scores
        ]),
        "mean_faith": _mean([
            r.rag_scores.get("faithfulness")
            for r in section_results if r.rag_scores
        ]),
    }


def _aggregate_opik(run_artifacts):
    """Aggregate Opik-style scores from all sections that ran Opik."""
    opik_sections = [
        s.get("opik_eval", {})
        for s in run_artifacts.get("sections", [])
        if s.get("opik_eval")
    ]
    return {
        "mean_hallucination": _mean([s.get("hallucination_score") for s in opik_sections]),
        "mean_answer_rel": _mean([s.get("answer_relevance_score") for s in opik_sections]),
        "mean_reg_tone": _mean([s.get("regulatory_tone_score") for s in opik_sections]),
        "mean_opik_composite": _mean([s.get("opik_composite") for s in opik_sections]),
    }


def _build_extended_summary(deep, opik, sections_evaluated):
    """Combine DeepEval + Opik aggregates into the final extended_eval_summary dict."""
    from healthark_eval.suite import _grade

    return {
        "mean_composite": deep["mean_composite"],
        "overall_grade": _grade(deep["mean_composite"]),
        "sections_evaluated": sections_evaluated,
        "grade_distribution": deep["grade_dist"],
        "mean_judge_normalized": deep["mean_judge"],
        "mean_rag_triad_score": deep["mean_rag"],
        "mean_faithfulness": deep["mean_faith"],
        "mean_bertscore_f1": None,
        "mean_hallucination_score": opik["mean_hallucination"],
        "mean_answer_relevance_score": opik["mean_answer_rel"],
        "mean_regulatory_tone_score": opik["mean_reg_tone"],
        "mean_opik_composite": opik["mean_opik_composite"],
        "mean_ragas": deep["mean_rag"],
        "framework": "deepeval_rag_triad + opik_style",
    }


def _finalise_pipeline_timing(run_artifacts, eval_phase_start):
    """Stamp total_eval_ms and total_pipeline_ms on run_artifacts['timing']."""
    total_eval_ms = round((time.perf_counter() - eval_phase_start) * 1000, 1)
    timing = run_artifacts.get("timing")
    if timing is None:
        return
    timing["total_eval_ms"] = total_eval_ms
    gen_phase = timing.get("generation_phase_ms") or 0
    timing["total_pipeline_ms"] = round(gen_phase + total_eval_ms, 1)


def _run_performance_analysis(run_artifacts, evaluation):
    """Run PerformanceAnalyzer and stash report on run_artifacts."""
    try:
        from src.eval.eval_performance import PerformanceAnalyzer
        perf_report = PerformanceAnalyzer().analyze(run_artifacts, evaluation)
        run_artifacts["performance_report"] = perf_report.to_dict()
        logging.info(
            "Performance: %s | failures=%d | improvements=%d",
            perf_report.summary_technical[:120],
            len(perf_report.failures),
            len(perf_report.improvements),
        )
    except Exception as exc:
        logging.warning("Performance analysis failed: %s", exc)


def _log_to_mlflow(run_artifacts, evaluation):
    """Log this run to MLflow and store run id + UI URL on run_artifacts."""
    from src.eval.eval_mlflow_tracker import MLflowTracker
    try:
        tracker = MLflowTracker()
        mlflow_run_id = tracker.log_run(
            run_artifacts=run_artifacts,
            eval_summary=evaluation,
            extended_summary=run_artifacts["extended_eval_summary"],
        )
        run_artifacts["mlflow_run_id"] = mlflow_run_id
        run_artifacts["mlflow_ui_url"] = tracker.run_url(mlflow_run_id)
        logging.info("MLflow run logged: %s", mlflow_run_id)
    except Exception as exc:
        logging.warning("MLflow logging failed: %s", exc)


def _run_extended_evaluation(run_artifacts, eval_suite):
    """Run extended evaluation: DeepEval RAG Triad + Opik-style + MLflow logging.

    Per section: appends 'extended_eval' (DeepEval results) and 'opik_eval'
    (hallucination / answer_relevance / regulatory_tone). Document level:
    saves 'extended_eval_summary' and logs to MLflow.
    """
    from healthark_eval.suite import _grade

    logging.info("Running extended evaluation (DeepEval + Opik-style + MLflow)...")

    opik_scorer = _build_opik_scorer(run_artifacts)
    section_results = []
    eval_phase_start = time.perf_counter()
    site_name = run_artifacts.get("site_name", "")

    for section in run_artifacts.get("sections", []):
        _evaluate_one_section(section, eval_suite, opik_scorer, site_name, section_results)

    deep = _aggregate_deepeval(section_results)
    opik = _aggregate_opik(run_artifacts)
    run_artifacts["extended_eval_summary"] = _build_extended_summary(
        deep, opik, sections_evaluated=len(section_results)
    )

    _finalise_pipeline_timing(run_artifacts, eval_phase_start)

    rules = get_eval_rules()
    evaluation = evaluate_run(run_artifacts, rules)

    _run_performance_analysis(run_artifacts, evaluation)
    _log_to_mlflow(run_artifacts, evaluation)
    save_eval_run(run_artifacts, evaluation)

    logging.info(
        "Evaluation complete — composite=%.1f grade=%s | judge=%.1f | "
        "rag_triad=%.3f | hallucination=%.3f | reg_tone=%.3f",
        deep["mean_composite"], _grade(deep["mean_composite"]),
        deep["mean_judge"] or 0.0, deep["mean_rag"] or 0.0,
        opik["mean_hallucination"] or 0.0, opik["mean_reg_tone"] or 0.0,
    )


# ── extraction_pmf helpers ────────────────────────────────────────────────────

def _init_azure_clients():
    """Load env vars and initialise the LangChain LLM + raw OpenAI client."""
    load_dotenv()
    key = os.getenv('AZURE_KEY', '')
    endpoint = os.getenv('AZURE_ENDPOINT', '')
    name = os.getenv('AZURE_NAME', '')
    version = os.getenv('AZURE_VERSION', '')
    llm = None
    client = None
    try:
        llm = AzureChatOpenAI(
            azure_deployment=name,
            api_key=key,
            azure_endpoint=endpoint,
            api_version=version,
            temperature=0.1,
        )
        client = AzureOpenAI(
            api_key=key,
            api_version=version,
            azure_endpoint=endpoint,
        )
    except Exception as e:
        print("Failed to initialize AzureChatOpenAI!")
        print("Error:", e)
    return llm, client, key, endpoint, name, version


def _resolve_base_folder():
    """Return the extraction base folder, descending into the first subfolder."""
    base = "data/artifacts/Extracted_folder"
    folders = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
    st.write(folders)
    if folders:
        return f"{base}/{folders[0]}"
    return base


def _load_template_docs(site_name, timestamp):
    """Load cover + body docx templates and personalise [Site Name] placeholders."""
    template_path_1 = r"templates\\output_template_1.docx"
    template_path = r"templates\\output_template_PMF.docx"
    new_file_name = f"PMF_Output_{timestamp}.docx"
    new_file_name_pdf = f"PMF_Output_{timestamp}.pdf"

    first_doc = Document(template_path)
    for paragraph in first_doc.paragraphs:
        for run in paragraph.runs:
            if "[Site Name]" in run.text:
                run.text = run.text.replace("[Site Name]", site_name)
                run.bold = True

    doc = Document(template_path_1)
    return first_doc, doc, new_file_name, new_file_name_pdf


def _retrieve_context(retriever, key, value_ls, data_as_string, base_folder):
    """Retrieve relevant document chunks for one template section."""
    input_text = ""
    paths = []
    ret_ms = None

    if len(value_ls) > 1:
        t0 = time.perf_counter()
        results = retriever.search(value_ls[1], top_k=8)
        ret_ms = (time.perf_counter() - t0) * 1000

        # Filter out low-relevance results; fall back to top-3 if all filtered
        MAX_DISTANCE = 0.85
        paths = [path for path, _, score in results if score <= MAX_DISTANCE]
        if not paths:
            paths = [r[0] for r in results[:3]]

        input_text += "excel data:\n" + data_as_string + "\n\n########################################\n\n"
        input_text += data_extraction(paths)
        st.write("-" * 125)
    elif "static" not in key.lower():
        input_text += "excel data:\n" + data_as_string + "\n\n########################################\n\n"
        input_text += data_extraction([base_folder])

    return input_text, paths, ret_ms


def _build_section_record(key, value_ls, value, paths, input_text, response_data, sec_start, ret_ms, gen_ms):
    """Build the success artifact dict for one processed section."""
    display_key = _derive_section_title(key, value_ls[0] if value_ls else value)
    return {
        "section_key": display_key,
        "template_key": key,
        "prompt_text": value_ls[0] if len(value_ls) > 0 else "",
        "retrieval_query": value_ls[1] if len(value_ls) > 1 else "",
        "retrieved_paths": paths,
        "input_text_size": len(input_text or ""),
        "is_static": "static" in key.lower(),
        "agent_result": response_data,
        "generated_text": (response_data or {}).get("result", {}).get("generated_text", ""),
        "timing": {
            "retrieval_ms": round(ret_ms, 1) if ret_ms is not None else None,
            "generation_ms": round(gen_ms, 1) if gen_ms is not None else None,
            "eval_ms": None,
            "total_ms": round((time.perf_counter() - sec_start) * 1000, 1),
        },
    }


def _build_error_section_record(key, value_ls, value, sec_start, ret_ms, gen_ms, error):
    """Build the error artifact dict when section processing fails."""
    display_key = _derive_section_title(key, value_ls[0] if value_ls else value)
    return {
        "section_key": display_key,
        "template_key": key,
        "prompt_text": value_ls[0] if len(value_ls) > 0 else "",
        "retrieval_query": value_ls[1] if len(value_ls) > 1 else "",
        "retrieved_paths": [],
        "input_text_size": 0,
        "is_static": "static" in key.lower(),
        "agent_result": None,
        "generated_text": "",
        "generation_error": str(error),
        "timing": {
            "retrieval_ms": round(ret_ms, 1) if ret_ms is not None else None,
            "generation_ms": round(gen_ms, 1) if gen_ms is not None else None,
            "eval_ms": None,
            "total_ms": round((time.perf_counter() - sec_start) * 1000, 1),
        },
    }


def _process_one_section(llm, client, key, value, doc_1, retriever, data_as_string, base_folder, input_file_path):
    """Run retrieval + generation for a single template section."""
    value = value.replace("[Site Name]", st.session_state.SiteName)
    value_ls = value.split('@!')

    flag = 1
    index = 1
    st.write(key)
    st.write(value[:80])
    st.write("")
    st.write("")

    sec_start = time.perf_counter()
    ret_ms = None
    gen_ms = None

    try:
        input_text, paths, ret_ms = _retrieve_context(retriever, key, value_ls, data_as_string, base_folder)

        gen_t0 = time.perf_counter()
        if "static" not in key.lower():
            response_data = handle_user_message(
                llm, client, key, value_ls[0], doc_1, flag, index, input_text, input_file_path
            )
        else:
            response_data = handle_user_message(llm, client, key, value_ls[0], doc_1, flag, index)
        gen_ms = (time.perf_counter() - gen_t0) * 1000

        return _build_section_record(key, value_ls, value, paths, input_text, response_data, sec_start, ret_ms, gen_ms)

    except Exception as e:
        st.write("Error in processing the section:")
        st.write(f"An error occurred: {e}")
        return _build_error_section_record(key, value_ls, value, sec_start, ret_ms, gen_ms, e)


def _compute_pipeline_timing(run_artifacts, pipeline_start):
    """Aggregate generation-phase timing across all sections."""
    return {
        "generation_phase_ms": round((time.perf_counter() - pipeline_start) * 1000, 1),
        "total_pipeline_ms": None,
        "total_generation_ms": round(sum(
            (s.get("timing") or {}).get("generation_ms") or 0
            for s in run_artifacts["sections"]
        ), 1),
        "total_retrieval_ms": round(sum(
            (s.get("timing") or {}).get("retrieval_ms") or 0
            for s in run_artifacts["sections"]
        ), 1),
        "total_eval_ms": None,
    }


def _assemble_document(doc_1, first_doc, output_dir, new_file_name, new_file_name_pdf):
    """Save body doc, inject TOC, prepend cover page, return BytesIO + paths."""
    new_file_path_temp = os.path.join(output_dir, f"Temp_{new_file_name}")
    final_document_doc = os.path.join(output_dir, f"Final_{new_file_name}")
    final_document = os.path.join(output_dir, f"Final_{new_file_name_pdf}")

    composer = Composer(doc_1)
    composer.save(new_file_path_temp)

    extract_headings_with_tables(new_file_path_temp, 0, final_document_doc)
    refresh_toc_with_word(os.path.abspath(final_document_doc))

    doc1 = Document(final_document_doc)
    composer1 = Composer(first_doc)
    composer1.append(doc1)
    composer1.save(new_file_path_temp)

    link_doc = Document(new_file_path_temp)
    output = io.BytesIO()
    link_doc.save(output)
    output.seek(0)

    return output, new_file_path_temp, final_document


def _run_rule_evaluation(run_artifacts):
    """Run rule-based evaluation and persist results to session state."""
    rules = get_eval_rules()
    evaluation = evaluate_run(run_artifacts, rules)
    eval_file = save_eval_run(run_artifacts, evaluation)
    st.session_state["last_eval_file"] = eval_file
    st.session_state["last_eval_score"] = evaluation.get("document_scores", {}).get("overall_score")


def _run_extended_eval_suite(run_artifacts, azure_name, azure_key, azure_endpoint, azure_version):
    """Run EvalSuite (Judge + RAG Triad) and persist results to session state."""
    try:
        eval_suite = EvalSuite(
            task="pmf",
            llm_provider="azure_openai",
            llm_model=azure_name,
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_version,
            run_judge=True,
            run_rag=True,
            run_semantic=False,   # BERTScore skipped — torch DLL issue on Windows
            run_lexical=False,    # No reference text available during generation
        )
        run_artifacts["_azure_key"] = azure_key
        run_artifacts["_azure_endpoint"] = azure_endpoint
        run_artifacts["_azure_version"] = azure_version
        _run_extended_evaluation(run_artifacts, eval_suite)
        ext = run_artifacts.get("extended_eval_summary", {})
        st.session_state["last_extended_composite"] = ext.get("mean_composite", 0)
        st.session_state["last_extended_grade"] = ext.get("overall_grade", "?")
        st.session_state["last_extended_judge"] = ext.get("mean_judge_normalized")
        st.session_state["last_extended_rag"] = ext.get("mean_rag_triad_score")
        st.session_state["last_mlflow_run_id"] = run_artifacts.get("mlflow_run_id")
        st.session_state["last_mlflow_url"] = run_artifacts.get("mlflow_ui_url", "http://localhost:5000")
    except Exception as exc:
        logging.warning("Extended evaluation failed: %s", exc)


# ── Main orchestrator ─────────────────────────────────────────────────────────

def extraction_pmf(template_file_path):
    llm, client, azure_key, azure_endpoint, azure_name, azure_version = _init_azure_clients()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    site_name = st.session_state.get("SiteName", "")

    run_artifacts = {
        "timestamp": timestamp,
        "template_file": template_file_path,
        "site_name": site_name,
        "model_name": azure_name,
        "sections": [],
    }

    template_content = extract_text_from_word(template_file_path)
    part1, part2 = Template_to_list(template_content)
    template_json = convert_dict(part1)

    first_doc, doc_1, new_file_name, new_file_name_pdf = _load_template_docs(site_name, timestamp)

    base_folder = _resolve_base_folder()
    convert_all_doc_to_docx_in_folder(base_folder)
    retriever = DocumentRetriever(base_folder)
    retriever.process_documents()

    st.write("Extracting text from documents...")
    pipeline_start = time.perf_counter()

    data_as_string = ""
    if st.session_state.uploaded_excel is not None:
        data_as_string = extract_text_from_xlsx(st.session_state.uploaded_excel)
        st.write(data_as_string)

    # TEST MODE: limit to first 5 sections for fast iteration — remove for full runs
    template_json = dict(list(template_json.items())[:5])

    for key, value in template_json.items():
        record = _process_one_section(
            llm, client, key, value, doc_1, retriever, data_as_string, base_folder, ""
        )
        run_artifacts["sections"].append(record)

    run_artifacts["timing"] = _compute_pipeline_timing(run_artifacts, pipeline_start)

    output_dir = os.path.join("data", "artifacts", "generated output file")
    os.makedirs(output_dir, exist_ok=True)

    output, doc_path, final_document = _assemble_document(
        doc_1, first_doc, output_dir, new_file_name, new_file_name_pdf
    )
    run_artifacts["final_doc_path"] = doc_path

    _run_rule_evaluation(run_artifacts)
    _run_extended_eval_suite(run_artifacts, azure_name, azure_key, azure_endpoint, azure_version)

    return output, final_document, new_file_name_pdf
