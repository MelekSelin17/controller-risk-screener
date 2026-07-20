"""
train_and_evaluate.py — Model training and comprehensive evaluation.

Separate metrics for two scenarios:
  - Scenario 1: Existing controller update (LOO protocol)
  - Scenario 2: New controller (entity-aware 5x20 CV)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    ARTIFACT_PATH, REPORT_PATH, DATASET_PATH,
    FINAL_FEATURES, MODEL_C, DECISION_THRESHOLD,
    CV_N_SPLITS, CV_N_REPEATS, CV_SEED,
    MIN_RISKY_MONTHS, Q_THRESHOLD,
)


# ---------------------------------------------------------------------------
# MODEL PIPELINE
# ---------------------------------------------------------------------------

def make_model() -> Pipeline:
    return Pipeline([
        ("pt", PowerTransformer(method="yeo-johnson")),
        ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
    ])


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def _metrics_at_threshold(y_true: np.ndarray, probas: np.ndarray, t: float) -> dict:
    pred = (probas >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged": int(pred.sum()),
    }


def _top_k_metrics(y_true: np.ndarray, probas: np.ndarray) -> dict:
    sorted_idx = np.argsort(probas)[::-1]
    n_pos = y_true.sum()
    result = {}
    for k in [10, 15, 20, 25]:
        top = sorted_idx[:k]
        tp  = int(y_true[top].sum())
        result["top%d" % k] = {
            "precision": round(tp / k, 4),
            "recall":    round(tp / n_pos, 4) if n_pos > 0 else 0.0,
            "tp": tp,
        }
    return result


# ---------------------------------------------------------------------------
# EVALUATION: Scenario 2 — Entity-aware CV (new controller)
# ---------------------------------------------------------------------------

def evaluate_new_controller(X: np.ndarray, y: np.ndarray) -> dict:
    """
    RepeatedStratifiedKFold: each controller is either in train or test.
    Hardest scenario — model has never seen this controller.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=CV_SEED
    )
    pr_list, roc_list, f1_list, prec_list, rec_list = [], [], [], [], []

    for tr, te in cv.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        pred  = (proba >= DECISION_THRESHOLD).astype(int)
        if len(np.unique(y[te])) > 1:
            pr_list.append(average_precision_score(y[te], proba))
            roc_list.append(roc_auc_score(y[te], proba))
        f1_list.append(f1_score(y[te], pred, zero_division=0))
        prec_list.append(precision_score(y[te], pred, zero_division=0))
        rec_list.append(recall_score(y[te], pred, zero_division=0))

    return {
        "protocol":    "entity_aware_cv (5x20=100 fold)",
        "n_folds":     CV_N_SPLITS * CV_N_REPEATS,
        "threshold":   DECISION_THRESHOLD,
        "auc_pr":      {"mean": round(np.mean(pr_list), 4), "std": round(np.std(pr_list), 4)},
        "auc_roc":     {"mean": round(np.mean(roc_list), 4), "std": round(np.std(roc_list), 4)},
        "f1":          {"mean": round(np.mean(f1_list), 4), "std": round(np.std(f1_list), 4)},
        "precision":   {"mean": round(np.mean(prec_list), 4), "std": round(np.std(prec_list), 4)},
        "recall":      {"mean": round(np.mean(rec_list), 4), "std": round(np.std(rec_list), 4)},
        "lift_vs_random": round(np.mean(pr_list) / (y.sum() / len(y)), 2),
    }


# ---------------------------------------------------------------------------
# EVALUATION: Scenario 1 — LOO (existing controller)
# ---------------------------------------------------------------------------

def evaluate_existing_controller(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Leave-One-Out: each controller is tested once, but the model was trained
    including that controller's history. Models the existing controller update scenario.
    """
    loo = LeaveOneOut()
    oof_probas = np.zeros(len(y))

    for tr, te in loo.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof_probas[te] = m.predict_proba(X[te])[:, 1]

    threshold_results = {}
    for t in [0.25, 0.30, 0.35, 0.40, 0.45]:
        threshold_results["t_%.2f" % t] = _metrics_at_threshold(y, oof_probas, t)

    return {
        "protocol":     "leave_one_out (controller known)",
        "auc_pr":       round(average_precision_score(y, oof_probas), 4),
        "auc_roc":      round(roc_auc_score(y, oof_probas), 4),
        "threshold_grid": threshold_results,
        "top_k":          _top_k_metrics(y, oof_probas),
        "oof_at_decision_threshold": _metrics_at_threshold(y, oof_probas, DECISION_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# FINAL MODEL TRAINING (full dataset)
# ---------------------------------------------------------------------------

def train_final_model(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Train on all data, save artifact."""
    model = make_model()
    model.fit(X, y)

    # Feature coefficients (interpretability)
    coefs = model.named_steps["m"].coef_[0]
    feature_importance = {
        feat: round(float(abs(coef)), 6)
        for feat, coef in zip(feature_names, coefs)
    }
    feature_direction = {
        feat: "risk_increases" if coef > 0 else "risk_decreases"
        for feat, coef in zip(feature_names, coefs)
    }

    # In-sample check (expected to be higher than CV results)
    proba_is = model.predict_proba(X)[:, 1]
    in_sample = {
        "auc_pr":  round(average_precision_score(y, proba_is), 4),
        "auc_roc": round(roc_auc_score(y, proba_is), 4),
        "note":    "In-sample check — expected to be higher than CV results",
    }

    # Training statistics (for median imputation at inference)
    training_stats = {
        feat: {
            "mean":   round(float(X[:, i].mean()), 6),
            "std":    round(float(X[:, i].std()), 6),
            "median": round(float(np.median(X[:, i])), 6),
        }
        for i, feat in enumerate(feature_names)
    }

    bundle = {
        "model":            model,
        "feature_names":    feature_names,
        "training_stats":   training_stats,
        "feature_importance": feature_importance,
        "feature_direction":  feature_direction,
        "decision_threshold": DECISION_THRESHOLD,
        "n_train":          int(len(y)),
        "n_pos":            int(y.sum()),
        "positive_rate":    round(float(y.mean()), 4),
        "in_sample_check":  in_sample,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH)
    print("  Model saved: %s" % ARTIFACT_PATH)
    return bundle


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    # Load dataset (run build_dataset.py first if missing)
    if not DATASET_PATH.exists():
        print("Dataset not found. Run build_dataset.py first.")
        sys.exit(1)

    ds = pd.read_csv(DATASET_PATH)
    X  = ds[FINAL_FEATURES].values
    y  = ds["label"].values
    print("Dataset: n=%d, pos=%d (%.1f%%)" % (len(y), y.sum(), 100 * y.mean()))
    print()

    # --- Scenario 2: New controller ---
    print("[1/3] Scenario 2 — New controller (entity-aware CV, 5x20=100 fold)...")
    s2 = evaluate_new_controller(X, y)
    print("  AUC-PR  = %.4f +/- %.4f" % (s2["auc_pr"]["mean"], s2["auc_pr"]["std"]))
    print("  AUC-ROC = %.4f +/- %.4f" % (s2["auc_roc"]["mean"], s2["auc_roc"]["std"]))
    print("  Prec    = %.4f +/- %.4f  (t=%.2f)" % (
        s2["precision"]["mean"], s2["precision"]["std"], DECISION_THRESHOLD))
    print("  Rec     = %.4f +/- %.4f" % (s2["recall"]["mean"], s2["recall"]["std"]))
    print("  Lift    = %.2fx" % s2["lift_vs_random"])
    print()

    # --- Scenario 1: Existing controller ---
    print("[2/3] Scenario 1 — Existing controller (LOO)...")
    s1 = evaluate_existing_controller(X, y)
    oof_m = s1["oof_at_decision_threshold"]
    print("  AUC-PR  = %.4f" % s1["auc_pr"])
    print("  AUC-ROC = %.4f" % s1["auc_roc"])
    print("  Prec    = %.4f  (t=%.2f)" % (oof_m["precision"], DECISION_THRESHOLD))
    print("  Rec     = %.4f" % oof_m["recall"])
    print()
    print("  Threshold grid (OOF):")
    for t_key, m in s1["threshold_grid"].items():
        print("    %s  Prec=%.1f%%  Rec=%.1f%%  Flag=%d  TP=%d/FP=%d" % (
            t_key, 100*m["precision"], 100*m["recall"], m["flagged"], m["tp"], m["fp"]))

    print()

    # --- Final model training ---
    print("[3/3] Final model training (full dataset)...")
    bundle = train_final_model(X, y, FINAL_FEATURES)

    print()
    print("  Feature importances (absolute coefficient):")
    sorted_imp = sorted(bundle["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp:
        direction = bundle["feature_direction"][feat]
        print("    %-32s  %.4f  (%s)" % (feat, imp, direction))

    # --- Save report ---
    report = {
        "config": {
            "label":            "Q75_gte%d (>= %d risky months)" % (MIN_RISKY_MONTHS, MIN_RISKY_MONTHS),
            "q_threshold":      Q_THRESHOLD,
            "min_risky_months": MIN_RISKY_MONTHS,
            "call_count_min":   20,
            "model":            "LogisticRegression(C=%.3f)" % MODEL_C,
            "transform":        "PowerTransformer(yeo-johnson)",
            "features":         FINAL_FEATURES,
            "n_features":       len(FINAL_FEATURES),
            "decision_threshold": DECISION_THRESHOLD,
        },
        "scenario_2_new_controller":      s2,
        "scenario_1_existing_controller": s1,
        "feature_importance": bundle["feature_importance"],
        "feature_direction":  bundle["feature_direction"],
        "in_sample_check":    bundle["in_sample_check"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print()
    print("Report saved: %s" % REPORT_PATH)
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Scenario 1 (existing)  AUC-PR=%.4f  AUC-ROC=%.4f" % (s1["auc_pr"], s1["auc_roc"]))
    print("  Scenario 2 (new)       AUC-PR=%.4f  AUC-ROC=%.4f" % (s2["auc_pr"]["mean"], s2["auc_roc"]["mean"]))
    print("  t=%.2f selected: Prec~%.0f%%  Rec~%.0f%%" % (
        DECISION_THRESHOLD, 100*s2["precision"]["mean"], 100*s2["recall"]["mean"]))


if __name__ == "__main__":
    main()
