"""
feature_ablation.py — Feature group ablation (for reviewer W2).

Four cumulative feature sets evaluated with LOO + CV:
  1. Static code only
  2. Code + Dependency
  3. Code + Dependency + DB-aware
  4. Full model (+ interaction features)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    DATASET_PATH, MODEL_C, CV_SEED,
    CV_N_SPLITS, CV_N_REPEATS, DECISION_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Feature groups (consistent with config.py)
# ---------------------------------------------------------------------------
CODE_FEATURES = [
    "log_ctrl_complexity",
    "log_ctrl_functions",
]
DEP_FEATURES = [
    "dep_complexity_per_function",
    "log_dep_complexity_sum",
]
DB_FEATURES = [
    "sp_aggregation_ratio",
    "sp_join_per_select",
    "sp_dml_zero",
    "log_max_table_rows",
    "sp_read_heavy_ratio",
]
ENHANCED_DB_FEATURES = [
    "sp_loop_ratio",
    "sp_order_ratio",
    "log_table_volume",
]
INTERACTION_FEATURES = [
    "agg_x_dmlzero",
    "complexity_total",
    "agg_x_table",
]

FEATURE_SETS = [
    {
        "label": "Static code only",
        "features": CODE_FEATURES,
    },
    {
        "label": "Code + Dependency",
        "features": CODE_FEATURES + DEP_FEATURES,
    },
    {
        "label": "Code + Dep + DB-aware",
        "features": CODE_FEATURES + DEP_FEATURES + DB_FEATURES,
    },
    {
        "label": "Full model (+ interactions)",
        "features": CODE_FEATURES + DEP_FEATURES + DB_FEATURES + INTERACTION_FEATURES,
    },
    {
        "label": "Full + Enhanced DB",
        "features": CODE_FEATURES + DEP_FEATURES + DB_FEATURES + ENHANCED_DB_FEATURES + INTERACTION_FEATURES,
    },
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_model() -> Pipeline:
    return Pipeline([
        ("pt", PowerTransformer(method="yeo-johnson")),
        ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
    ])


# ---------------------------------------------------------------------------
# LOO
# ---------------------------------------------------------------------------
def eval_loo(X: np.ndarray, y: np.ndarray) -> dict:
    loo = LeaveOneOut()
    oof = np.zeros(len(y))
    for tr, te in loo.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    pred = (oof >= DECISION_THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    top10 = int(y[np.argsort(oof)[::-1][:10]].sum())
    return {
        "auc_pr":  round(average_precision_score(y, oof), 4),
        "auc_roc": round(roc_auc_score(y, oof), 4),
        "prec":    round(prec, 4),
        "rec":     round(rec, 4),
        "f1":      round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "prec_at_10": round(top10 / 10, 2),
    }


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
def eval_cv(X: np.ndarray, y: np.ndarray) -> dict:
    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=CV_SEED
    )
    pr_l, roc_l = [], []
    for tr, te in cv.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            pr_l.append(average_precision_score(y[te], proba))
            roc_l.append(roc_auc_score(y[te], proba))
    pos_rate = y.sum() / len(y)
    return {
        "auc_pr":  (round(np.mean(pr_l), 4), round(np.std(pr_l), 4)),
        "auc_roc": (round(np.mean(roc_l), 4), round(np.std(roc_l), 4)),
        "lift":    round(np.mean(pr_l) / pos_rate, 2),
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    ds = pd.read_csv(DATASET_PATH)
    y  = ds["label"].values
    n, n_pos = len(y), int(y.sum())
    print("Dataset: n=%d, pos=%d (%.1f%%)" % (n, n_pos, 100 * y.mean()))
    print()

    SEP = "=" * 72
    results = []

    for fs in FEATURE_SETS:
        feats = fs["features"]
        label = fs["label"]

        missing = [f for f in feats if f not in ds.columns]
        if missing:
            print("WARNING — missing features: %s" % missing)
            continue

        X = ds[feats].values

        print(SEP)
        print("%-35s  (%d features)" % (label, len(feats)))
        print(SEP)

        s1 = eval_loo(X, y)
        lift_loo = round(s1["auc_pr"] / (n_pos / n), 2)
        print("  LOO   AUC-PR=%.4f  AUC-ROC=%.4f  Lift=%.2fx" % (
            s1["auc_pr"], s1["auc_roc"], lift_loo))
        print("        Prec=%.1f%%  Rec=%.1f%%  F1=%.3f  P@10=%.0f%%" % (
            100*s1["prec"], 100*s1["rec"], s1["f1"], 100*s1["prec_at_10"]))

        s2 = eval_cv(X, y)
        print("  CV    AUC-PR=%.4f+-%.4f  AUC-ROC=%.4f+-%.4f  Lift=%.2fx" % (
            s2["auc_pr"][0], s2["auc_pr"][1],
            s2["auc_roc"][0], s2["auc_roc"][1],
            s2["lift"]))
        print()

        results.append({
            "feature_set":  label,
            "n_features":   len(feats),
            "loo_auc_pr":   s1["auc_pr"],
            "loo_auc_roc":  s1["auc_roc"],
            "loo_lift":     lift_loo,
            "loo_prec":     round(s1["prec"], 4),
            "loo_rec":      round(s1["rec"], 4),
            "loo_f1":       round(s1["f1"], 4),
            "loo_prec_at10": s1["prec_at_10"],
            "cv_auc_pr":    s2["auc_pr"][0],
            "cv_auc_pr_std": s2["auc_pr"][1],
            "cv_auc_roc":   s2["auc_roc"][0],
            "cv_lift":      s2["lift"],
        })

    # Summary table
    print()
    print(SEP)
    print("ABLATION SUMMARY TABLE (for paper)")
    print(SEP)
    hdr = "%-33s  %3s  %6s  %6s  %5s  %5s  %5s  %5s" % (
        "Feature Set", "#F", "AUC-PR", "AUC-ROC", "Prec", "Rec", "F1", "Lift")
    print(hdr)
    print("-" * 72)
    for r in results:
        marker = " <-- full" if r["n_features"] == 12 else ""
        print("%-33s  %3d  %.4f  %.4f  %5.1f%%  %5.1f%%  %.3f  %.2fx%s" % (
            r["feature_set"], r["n_features"],
            r["loo_auc_pr"], r["loo_auc_roc"],
            100*r["loo_prec"], 100*r["loo_rec"], r["loo_f1"],
            r["loo_lift"], marker))

    # Save CSV
    out = ROOT / "data" / "output" / "ablation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print("\nSaved: %s" % out)


if __name__ == "__main__":
    main()
