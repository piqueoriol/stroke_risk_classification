# Stroke Risk Classification Workflow

This repository contains an **end‑to‑end data science workflow** for a binary classification task: predicting stroke risk from publicly available health data. The goal is to demonstrate data science best practices and a systematic approach, **not** to deep‑dive into stroke analysis, but to showcase a reproducible project structure.

This project provides two complementary implementations:
- Notebook‑based approach under the notebooks/ folder, offering detailed plots and step‑by‑step insights.
- Modular script‑based pipeline with source code in the src/ folder, orchestrated by main.py, and packaged for production via a Dockerfile and Makefile.

---

## Repository Structure

```
├── data/
│   ├── in/           # Input data (stroke_data.csv)
│   └── out/          # Model outputs (train.csv, test.csv, predictions.csv)
├── notebooks/        # Jupyter notebooks with exploratory analysis & plots
│   └── stroke_risk_classification.ipynb # A complete analysis of this project, EDA, research, models, analysis, feature importance, ...
├── src/              # Source code
│   ├── main.py       # Orchestrates full workflow (preprocess, train, optimize, output)
│   ├── stroke_prediction_model.py  # `StrokePredictor` class (preprocessing, modeling)
│   └── utils.py      # Visualization functions for notebooks
├── config.yaml       # All tunable parameters (paths, splits, hyperparameters, metrics)
├── Dockerfile        # Builds a container able to read/write `data/` and run `main.py`
├── Makefile          # Convenience targets (`make docker-build`, `make docker-run`)
├── requirements.txt  # Python dependencies
└── README.md         # Project overview and usage instructions
```

---

## Getting Started

### Prerequisites

* Docker
* Make

### Build & Run

From the project root, simply:

```bash
make docker-run
```

This will:

1. Build the Docker image (`stroke-predictor`).
2. Bind‑mount your local `data/` directory into the container at `/app/data`.
3. Execute `main.py` using the parameters in `config.yaml`.

The workflow will:

* Read `data/in/stroke_data.csv`.
* Preprocess, split, and balance the data.
* Train and optimize **Random Forest** and **Logistic Regression** based on the configured metric.
* Select the best model, then generate:

  * `data/out/train.csv`
  * `data/out/test.csv`
  * `data/out/predictions.csv` (probabilities for the positive class)
* Print progress, hyperparameter search results, and final evaluation metrics to the console.

---

## Notebook

A complete approach to the project can be found in the notebook. EDA, modelling, analysis, feature importances, ..., see:

```
notebooks/stroke_risk_classification.ipynb
```

This notebook uses the `utils.py` visualization functions to generate plots of missing values, distributions, correlations, and ROC/PR curves.

---

## Configuration

All parameters are centralized in **`config.yaml`**:

* **general\_parameters**: data paths, train/test split, CV folds, optimization metric
* **model\_hyperparameters**: grid search settings for each model
* **output**: evaluation metrics and prediction output path

Modify these values to customize the workflow without changing code.

---
