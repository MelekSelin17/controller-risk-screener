# Pre-Release Performance Risk Screening for Backend Components

Replication package for the paper:

> **Pre-Release Performance Risk Screening for Backend Components Using Static Code and Database-Aware Features**  
> Melek Selin Uysal, Feza Buzluca  
> UBMK 2026 — IEEE Conference

## Overview

This repository contains the code, evaluation scripts, and an anonymized dataset used to produce all results reported in the paper. All system-specific identifiers (schema owner, package names, stored procedure names, table names, controller keys) have been replaced with generic placeholders; all numeric values are preserved exactly as used in the experiments.

The approach screens controller-level backend components for persistent relative tail-latency risk using only development-time evidence: controller code metrics, downstream dependency complexity, and database-aware features extracted from stored procedure metadata and table statistics.

## Repository structure

```
controller-risk-screener/
├── pipeline/               # Core ML pipeline
│   ├── config.py           # All constants and feature definitions
│   ├── build_dataset.py    # Dataset construction (labels + features)
│   ├── feature_extractor.py # Live feature extraction from .cs files
│   ├── train_and_evaluate.py # Model training and LOO/CV evaluation
│   ├── predict.py          # Risk score inference for a single controller
│   └── warn.py             # CLI warning tool
├── scripts/                # Evaluation and analysis scripts
│   ├── bootstrap_ci.py     # Bootstrap confidence intervals for LOO AUC-PR
│   ├── heuristic_baselines.py # Single-feature heuristic baselines
│   ├── feature_ablation.py # Cumulative feature group ablation
│   ├── model_comparison.py # Model comparison (LR vs ensemble)
│   ├── plot_pr_curve.py    # Precision-recall curve figure
│   ├── temporal_label_split.py # Temporal robustness check (S1/S2/S3)
│   ├── shap_analysis.py    # SHAP-based feature importance analysis
│   └── run_full_pipeline.py # End-to-end pipeline runner
├── src/
│   └── controller_keys.py  # Controller key normalization utilities
├── data/
│   └── README.md           # Data schema documentation
├── figures/
│   └── pr_curve_ml_vs_heuristic.png  # Figure 2 from the paper
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Reproducing the paper results

### 1. Prepare data

Follow the schema described in `data/README.md`. Place files in `data/raw/` and `data/processed/`.

### 2. Build the dataset

```bash
python -m pipeline.build_dataset
```

Produces `data/processed/controller_dataset.csv` (113 controllers x 14 features + label).

### 3. Run the full evaluation

```bash
python scripts/run_full_pipeline.py
```

Or run individual scripts:

```bash
# Main LOO evaluation + model training
python -m pipeline.train_and_evaluate

# Bootstrap confidence intervals (Table II in paper)
python scripts/bootstrap_ci.py

# Heuristic baselines comparison (Table II in paper)
python scripts/heuristic_baselines.py

# Feature group ablation (Table III in paper)
python scripts/feature_ablation.py

# Temporal label-split robustness check (Table IV in paper)
python scripts/temporal_label_split.py

# SHAP analysis (Section V.A in paper)
python scripts/shap_analysis.py

# Precision-recall curve figure (Figure 2 in paper)
python scripts/plot_pr_curve.py
```

### 4. Screen a new controller (live use)

```bash
python -m pipeline.warn --file path/to/OrderController.cs --project-root . --verbose
```

## Model configuration

All model parameters are defined in `pipeline/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Q_THRESHOLD` | 0.70 | Monthly percentile threshold (top-30% slowest) |
| `MIN_RISKY_MONTHS` | 3 | Minimum months flagged for label=1 |
| `CALL_COUNT_MIN` | 10 | Minimum calls for a reliable controller-month |
| `MODEL_C` | 0.02 | L2 regularization strength |
| `DECISION_THRESHOLD` | 0.35 | Post-hoc operating threshold |

## Key results (paper)

| Metric | Value |
|--------|-------|
| AUC-PR (LOO, n=113) | 0.674 |
| 95% bootstrap CI | [0.523, 0.823] |
| Lift over random | 2.11x |
| Precision@10 | 70% |
| Best heuristic (AUC-PR) | 0.548 (controller complexity) |
| Temporal check S3 AUC-PR | 0.678 |

## Citation

```bibtex
@inproceedings{uysal2026perfscreen,
  title     = {Pre-Release Performance Risk Screening for Backend Components
               Using Static Code and Database-Aware Features},
  author    = {Uysal, Melek Selin and Buzluca, Feza},
  booktitle = {Proceedings of UBMK 2026},
  year      = {2026},
  publisher = {IEEE}
}
```

## License

MIT License. See LICENSE file.

## Data availability

The `data/` directory contains an anonymized version of the industrial dataset used in the paper. All system-specific identifiers have been replaced with generic placeholders:

- Database schema owner (`DFAB` → `CORP`)
- Package names (`PCK_*` → `PKG_001`, `PKG_002`, …)
- Stored procedure names → `SP_001_001`, …
- Table names → `TBL_001`, …
- Controller keys → `ctrl_001`, …

All numeric values (latency measurements, row counts, complexity scores, join counts, etc.) are preserved exactly as used in the paper experiments.
