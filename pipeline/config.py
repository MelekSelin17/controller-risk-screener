"""
config.py — Single source of truth: all constants, feature definitions, paths.

Configuration decisions (locked):
  - Label    : Q75_gte2 — controller appears in top-25% slowest for at least 2 reliable months
  - Transform: PowerTransformer (Yeo-Johnson) — better distribution than log alone
  - Model    : LogisticRegression (C=0.02, L2)
  - Threshold: 0.30 — Precision≈47%, Recall≈72%
  - Eval     : RepeatedStratifiedKFold(n_splits=5, n_repeats=20) = 100 fold
  - APM      : used only for label construction, NEVER as a feature
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # ThesisProject/

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
APM_MONTHLY     = ROOT / "data" / "processed" / "apm_monthly.csv"
SONAR_PROCESSED = ROOT / "data" / "processed" / "sonar_features.csv"
ORACLE_SP       = ROOT / "data" / "raw" / "sp_metrics.csv"
ORACLE_DEPS     = ROOT / "data" / "raw" / "sp_table_deps.csv"
ORACLE_TABLES   = ROOT / "data" / "raw" / "table_stats.csv"
SP_MAPPING      = ROOT / "data" / "raw" / "controller_sp_mapping.csv"

# Output
OUTPUT_DIR      = ROOT / "data" / "models"
ARTIFACT_PATH   = OUTPUT_DIR / "model_bundle.joblib"
REPORT_PATH     = OUTPUT_DIR / "evaluation_report.json"
DATASET_PATH    = ROOT / "data" / "processed" / "controller_dataset.csv"

# ---------------------------------------------------------------------------
# Label parameters
# ---------------------------------------------------------------------------
CALL_COUNT_MIN  = 10      # minimum call count for a reliable controller-month
Q_THRESHOLD     = 0.70    # monthly "slow" threshold (top-30%)
MIN_RISKY_MONTHS = 3      # minimum slow months required for label=1

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# 9 base features (validated through ablation)
BASE_FEATURES = [
    "sp_aggregation_ratio",
    "sp_join_per_select",
    "dep_complexity_per_function",
    "sp_dml_zero",
    "log_ctrl_functions",
    "log_dep_complexity_sum",
    "log_ctrl_complexity",
    "log_max_table_rows",
    "sp_read_heavy_ratio",
]

# 2 enhanced DB features (added in response to Reviewer 2: "coarse proxies")
# sp_loop_ratio was tested but provided no incremental signal → removed
ENHANCED_DB_FEATURES = [
    "sp_order_ratio",
    "log_table_volume",
]

# 3 interaction features
INTERACTION_FEATURES = [
    "agg_x_dmlzero",
    "complexity_total",
    "agg_x_table",
]

FINAL_FEATURES    = BASE_FEATURES + INTERACTION_FEATURES             # 12 features (baseline)
ENHANCED_FEATURES = BASE_FEATURES + ENHANCED_DB_FEATURES + INTERACTION_FEATURES  # 14 features (final model)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
MODEL_C         = 0.02
DECISION_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# CV parameters
# ---------------------------------------------------------------------------
CV_N_SPLITS  = 5
CV_N_REPEATS = 20
CV_SEED      = 42
