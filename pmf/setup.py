from setuptools import setup, find_packages

setup(
    name="healthark-eval",
    version="0.1.0",
    author="Healthark GenAI Team",
    description="Internal LLM Evaluation Framework",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "sacrebleu>=2.3.1",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "sentence-transformers>=2.2.2",
        "anthropic>=0.25.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": ["pytest-cov", "black", "isort"],
        "ragas": ["ragas>=0.1.0"],
    },
)
