# Documentation Index

Healthark GenAI Evaluation Framework — PMF Document Generator

## Guides

| Document | Description |
|---|---|
| [Quick Start](quick_start.md) | Installation, minimal example, how to run benchmarks and tests |
| [Framework Overview](eval_framework_overview.md) | Architecture diagram, module descriptions, how to extend |
| [Benchmark Guide](benchmark_guide.md) | Creating cases, expert annotation, inter-annotator agreement |
| [Dashboard User Guide](dashboard_user_guide.md) | Tab walkthroughs, metric tooltips, configuration |

## Quick Links

- **Run the app**: `streamlit run app.py`
- **Run the dashboard standalone**: `streamlit run app_eval_dashboard.py`
- **Run tests**: `pytest tests/test_eval_regression.py -v`
- **Validate benchmark data**: `python -m src.eval.benchmark_loader --validate`
- **Install package**: `pip install -e .`
