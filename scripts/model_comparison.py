"""
model_comparison.py — LR, RF, GradientBoosting comparison.

Three models compared on the same final dataset (n=113, 12 features)
using the same LOO + CV protocol. Produces a table for the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    DATASET_PATH, MODEL_C, CV_SEED,
    CV_N_SPLITS, CV_N_REPEATS, DECISION_THRESHOLD, FINAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "Logistic Regression": Pipeline([
        ("pt", PowerTransformer(method="yeo-johnson")),
        ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=CV_SEED,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=CV_SEED,
    ),
}

if HAS_XGB:
    MODELS["XGBoost"] = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(77 / 36),  # neg/pos ratio (113-36)/36
        eval_metric="logloss",
        random_state=CV_SEED,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# LOO evaluation
# ---------------------------------------------------------------------------
def eval_loo(model_template, X: np.ndarray, y: np.ndarray) -> dict:
    import copy
    loo = LeaveOneOut()
    oof = np.zeros(len(y))
    for tr, te in loo.split(X, y):
        m = copy.deepcopy(model_template)
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
        "auc_pr":      round(average_precision_score(y, oof), 4),
        "auc_roc":     round(roc_auc_score(y, oof), 4),
        "prec":        round(prec, 4),
        "rec":         round(rec, 4),
        "f1":          round(f1, 4),
        "prec_at_10":  round(top10 / 10, 2),
    }


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------
def eval_cv(model_template, X: np.ndarray, y: np.ndarray) -> dict:
    import copy
    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=CV_SEED
    )
    pr_l, roc_l = [], []
    for tr, te in cv.split(X, y):
        m = copy.deepcopy(model_template)
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            pr_l.append(average_precision_score(y[te], proba))
            roc_l.append(roc_auc_score(y[te], proba))
    pos_rate = y.sum() / len(y)
    return {
        "auc_pr":     (round(np.mean(pr_l), 4), round(np.std(pr_l), 4)),
        "auc_roc":    (round(np.mean(roc_l), 4), round(np.std(roc_l), 4)),
        "lift":       round(np.mean(pr_l) / pos_rate, 2),
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    ds = pd.read_csv(DATASET_PATH)
    y  = ds["label"].values
    n, n_pos = len(y), int(y.sum())
    print("Dataset: n=%d, pos=%d (%.1f%%)" % (n, n_pos, 100 * y.mean()))

    missing = [f for f in FINAL_FEATURES if f not in ds.columns]
    if missing:
        print("MISSING FEATURES:", missing)
        sys.exit(1)

    X = ds[FINAL_FEATURES].values

    SEP = "=" * 72
    results = []

    for name, model in MODELS.items():
        print()
        print(SEP)
        print(name)
        print(SEP)

        print("  Computing LOO...", flush=True)
        s1 = eval_loo(model, X, y)
        lift_loo = round(s1["auc_pr"] / (n_pos / n), 2)
        print("  LOO   AUC-PR=%.4f  AUC-ROC=%.4f  Lift=%.2fx" % (
            s1["auc_pr"], s1["auc_roc"], lift_loo))
        print("        Prec=%.1f%%  Rec=%.1f%%  F1=%.3f  P@10=%.0f%%" % (
            100 * s1["prec"], 100 * s1["rec"], s1["f1"], 100 * s1["prec_at_10"]))

        print("  Computing CV  (5x20=100 folds)...", flush=True)
        s2 = eval_cv(model, X, y)
        print("  CV    AUC-PR=%.4f+-%.4f  AUC-ROC=%.4f+-%.4f  Lift=%.2fx" % (
            s2["auc_pr"][0], s2["auc_pr"][1],
            s2["auc_roc"][0], s2["auc_roc"][1],
            s2["lift"]))

        results.append({
            "model":          name,
            "loo_auc_pr":     s1["auc_pr"],
            "loo_auc_roc":    s1["auc_roc"],
            "loo_lift":       lift_loo,
            "loo_f1":         s1["f1"],
            "loo_prec":       s1["prec"],
            "loo_rec":        s1["rec"],
            "loo_prec_at10":  s1["prec_at_10"],
            "cv_auc_pr":      s2["auc_pr"][0],
            "cv_auc_pr_std":  s2["auc_pr"][1],
            "cv_auc_roc":     s2["auc_roc"][0],
            "cv_lift":        s2["lift"],
        })

    # Summary table
    print()
    print(SEP)
    print("MODEL COMPARISON TABLE")
    print(SEP)
    hdr = "%-22s  %6s  %6s  %5s  %5s  %5s  %5s" % (
        "Model", "AUC-PR", "AUC-ROC", "Prec", "Rec", "F1", "Lift")
    print(hdr)
    print("-" * 62)
    for r in results:
        print("%-22s  %.4f  %.4f  %5.1f%%  %5.1f%%  %.3f  %.2fx" % (
            r["model"],
            r["loo_auc_pr"], r["loo_auc_roc"],
            100 * r["loo_prec"], 100 * r["loo_rec"],
            r["loo_f1"], r["loo_lift"]))

    # Save CSV
    out = ROOT / "data" / "output" / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print("\nSaved: %s" % out)


if __name__ == "__main__":
    main()
