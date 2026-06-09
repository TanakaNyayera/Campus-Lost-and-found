import os

BASE = "MSU_Lost_Found_ML_System"

# All directories to create
dirs = [
    "data/raw/images/lost",
    "data/raw/images/found",
    "data/processed",
    "models/nlp",
    "models/cv",
    "models/hybrid",
    "notebooks",
    "src/data_preprocessing",
    "src/models",
    "src/evaluation",
    "src/web_app/templates",
    "src/web_app/static",
    "reports/figures",
    "reports/evaluation_metrics",
    "logs",
    "config",
]

# Files to create (path: content)
files = {
    "data/raw/descriptions.csv": "id,item_name,description,category,date_reported\n",
    "data/processed/nlp_features.pkl": "",
    "data/processed/image_features.pkl": "",
    "data/processed/combined_dataset.csv": "id,item_name,description,image_path,label\n",
    "notebooks/01_data_exploration.ipynb": '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}',
    "notebooks/02_nlp_model.ipynb":        '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}',
    "notebooks/03_cv_model.ipynb":         '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}',
    "notebooks/04_hybrid_model.ipynb":     '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}',
    "src/data_preprocessing/text_preprocessing.py":  "# Text preprocessing utilities\n",
    "src/data_preprocessing/image_preprocessing.py": "# Image preprocessing utilities\n",
    "src/models/nlp_model.py":    "# NLP model (BERT / SBERT)\n",
    "src/models/cv_model.py":     "# Computer Vision model\n",
    "src/models/hybrid_model.py": "# Hybrid (NLP + CV) model\n",
    "src/evaluation/metrics.py":  "# Evaluation metrics\n",
    "src/web_app/app.py":         "# Flask application\nfrom flask import Flask\napp = Flask(__name__)\n",
    "reports/figures/.gitkeep":            "",
    "reports/evaluation_metrics/.gitkeep": "",
    "config/config.yaml": (
        "# Project configuration\n"
        "project_name: MSU_Lost_Found_ML_System\n"
        "data_path: data/\n"
        "model_path: models/\n"
    ),
    "requirements.txt": (
        "flask\n"
        "numpy\n"
        "pandas\n"
        "scikit-learn\n"
        "torch\n"
        "transformers\n"
        "opencv-python\n"
        "Pillow\n"
        "pyyaml\n"
        "jupyter\n"
    ),
    "README.md": (
        "# MSU Lost and Found ML System\n\n"
        "A machine learning system to match lost and found items using NLP and Computer Vision.\n\n"
        "## Project Structure\n"
        "- `data/` — Raw and processed datasets\n"
        "- `models/` — Trained model files\n"
        "- `notebooks/` — Jupyter notebooks for exploration\n"
        "- `src/` — Source code\n"
        "- `reports/` — Figures and evaluation metrics\n"
        "- `config/` — Configuration files\n"
    ),
    ".gitignore": (
        "__pycache__/\n"
        "*.py[cod]\n"
        "*.pkl\n"
        "*.egg-info/\n"
        ".env\n"
        "venv/\n"
        "*.log\n"
        "data/raw/images/\n"
    ),
}

# ── Create directories ──────────────────────────────────────────────────────
print(f"Creating project: {BASE}/\n")
for d in dirs:
    path = os.path.join(BASE, d)
    os.makedirs(path, exist_ok=True)
    print(f"  [DIR]  {path}")

# ── Create files ────────────────────────────────────────────────────────────
print()
for rel_path, content in files.items():
    path = os.path.join(BASE, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  [FILE] {path}")

print("\n✅ Project structure created successfully!")
print(f"   Navigate into it with:  cd {BASE}")
