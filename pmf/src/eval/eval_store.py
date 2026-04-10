import json
import os
from datetime import datetime
from typing import Dict, Any, List


EVAL_DIR = "data/eval_runs"
INDEX_PATH = os.path.join(EVAL_DIR, "index.jsonl")


def _ensure_eval_dir() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)


def save_eval_run(run_artifacts: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    _ensure_eval_dir()
    timestamp = run_artifacts.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_site = (run_artifacts.get("site_name") or "unknown_site").replace(" ", "_")
    run_id = f"{timestamp}_{safe_site}"
    run_file = os.path.join(EVAL_DIR, f"{run_id}.json")

    payload = {
        "run_id": run_id,
        "run_artifacts": run_artifacts,
        "evaluation": evaluation,
    }
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    index_row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "site_name": run_artifacts.get("site_name"),
        "template_file": run_artifacts.get("template_file"),
        "overall_score": evaluation.get("document_scores", {}).get("overall_score"),
        "final_doc_path": run_artifacts.get("final_doc_path"),
        "run_file": run_file,
    }
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(index_row) + "\n")

    return run_file


def list_runs() -> List[Dict[str, Any]]:
    if not os.path.exists(INDEX_PATH):
        return []

    rows: List[Dict[str, Any]] = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return rows


def load_run_by_file(run_file: str) -> Dict[str, Any]:
    with open(run_file, "r", encoding="utf-8") as f:
        return json.load(f)
