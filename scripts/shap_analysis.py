"""
shap_analysis.py — SHAP feature importance for the enhanced model.

Two outputs:
  1. Global: mean absolute SHAP value per feature (bar chart + CSV)
  2. Individual: SHAP values per controller over LOO OOF
     -> explanation for FP and FN examples

Reviewer 2/3: answers "why does the model flag this component?"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    DATASET_PATH, MODEL_C, CV_SEED,
    ENHANCED_FEATURES, DECISION_THRESHOLD,
)

FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR  = ROOT / "data" / "output"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_LABELS = {
    "sp_aggregation_ratio":     "SP Aggregation Ratio",
    "sp_join_per_select":       "SP Join/Select Ratio",
    "dep_complexity_per_function": "Dep. Complexity/Function",
    "sp_dml_zero":              "Write-Absence Flag",
    "log_ctrl_functions":       "log(Controller Functions)",
    "log_dep_complexity_sum":   "log(Dep. Complexity Sum)",
    "log_ctrl_complexity":      "log(Controller Complexity)",
    "log_max_table_rows":       "log(Max Table Rows)",
    "sp_read_heavy_ratio":      "Read-Heavy Ratio",
    "sp_order_ratio":           "SP Order/Select Ratio",
    "log_table_volume":         "log(Max Table Volume)",
    "agg_x_dmlzero":            "Aggregation x Write-Absence",
    "complexity_total":         "Total Layered Complexity",
    "agg_x_table":              "Aggregation x Table Size",
}


def make_model() -> Pipeline:
    return Pipeline([
        ("pt", PowerTransformer(method="yeo-johnson")),
        ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
    ])


def get_loo_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    loo = LeaveOneOut()
    oof = np.zeros(len(y))
    for tr, te in loo.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    ds = pd.read_csv(DATASET_PATH)
    y  = ds["label"].values
    X  = ds[ENHANCED_FEATURES].values
    feature_labels = [FEATURE_LABELS.get(f, f) for f in ENHANCED_FEATURES]

    print("=" * 60)
    print("SHAP ANALYSIS — Enhanced Model (14 features)")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Train full-data model (for SHAP)
    # ----------------------------------------------------------------
    print("[1/4] Training full-data model...")
    model = make_model()
    model.fit(X, y)

    # Transformed X (SHAP uses transformed space for linear model)
    X_transformed = model.named_steps["pt"].transform(X)
    lr = model.named_steps["m"]

    # LinearExplainer — correct SHAP for logistic regression
    explainer = shap.LinearExplainer(lr, X_transformed, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_transformed)  # (n, n_features)

    # ----------------------------------------------------------------
    # 2. Global feature importance (mean |SHAP|)
    # ----------------------------------------------------------------
    print("[2/4] Computing global feature importance...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":       ENHANCED_FEATURES,
        "feature_label": feature_labels,
        "mean_abs_shap": mean_abs_shap,
        "coef":          lr.coef_[0],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n  Global SHAP importance (mean |SHAP|):")
    print("  %-35s  %8s  %8s" % ("Feature", "SHAP", "Coef"))
    print("  " + "-" * 55)
    for _, row in shap_df.iterrows():
        direction = "+" if row["coef"] > 0 else "-"
        print("  %-35s  %.4f  %s%.4f" % (
            row["feature_label"], row["mean_abs_shap"],
            direction, abs(row["coef"])))

    # Save CSV
    shap_df.to_csv(OUTPUT_DIR / "shap_global_importance.csv", index=False)

    # ----------------------------------------------------------------
    # 3. SHAP bar plot — for paper
    # ----------------------------------------------------------------
    print("\n[3/4] Saving SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in shap_df["coef"]]
    bars = ax.barh(
        shap_df["feature_label"][::-1],
        shap_df["mean_abs_shap"][::-1],
        color=colors[::-1],
        edgecolor="white",
        height=0.65,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance (Enhanced Model, 14 features)", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Risk-increasing"),
        Patch(facecolor="#1f77b4", label="Risk-decreasing"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "shap_global_bar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: %s" % fig_path)

    # ----------------------------------------------------------------
    # 4. FP/FN example explanation over LOO OOF
    # ----------------------------------------------------------------
    print("\n[4/4] FP/FN SHAP explanation over LOO OOF...")
    oof = get_loo_oof(X, y)
    pred = (oof >= DECISION_THRESHOLD).astype(int)

    fn_idx = np.where((pred == 0) & (y == 1))[0]
    fp_idx = np.where((pred == 1) & (y == 0))[0]
    tp_idx = np.where((pred == 1) & (y == 1))[0]

    # Average SHAP values per group
    def group_shap(indices):
        if len(indices) == 0:
            return np.zeros(len(ENHANCED_FEATURES))
        return shap_values[indices].mean(axis=0)

    shap_fn = group_shap(fn_idx)
    shap_fp = group_shap(fp_idx)
    shap_tp = group_shap(tp_idx)

    summary_df = pd.DataFrame({
        "feature":       ENHANCED_FEATURES,
        "feature_label": feature_labels,
        "shap_tp_mean":  shap_tp,
        "shap_fp_mean":  shap_fp,
        "shap_fn_mean":  shap_fn,
    })
    summary_df.to_csv(OUTPUT_DIR / "shap_group_analysis.csv", index=False)

    print("\n  Group-average SHAP (positive = risk-increasing):")
    print("  %-35s  %8s  %8s  %8s" % ("Feature", "TP", "FP", "FN"))
    print("  " + "-" * 65)
    for _, row in summary_df.iterrows():
        print("  %-35s  %+7.4f  %+7.4f  %+7.4f" % (
            row["feature_label"],
            row["shap_tp_mean"], row["shap_fp_mean"], row["shap_fn_mean"]))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    top_pos = shap_df[shap_df["coef"] > 0]
    top_neg = shap_df[shap_df["coef"] < 0]
    print("  Most influential risk-increasing feature : %s" %
          (top_pos.iloc[0]["feature_label"] if len(top_pos) > 0 else "none"))
    print("  Most influential risk-decreasing feature : %s" %
          (top_neg.iloc[0]["feature_label"] if len(top_neg) > 0 else "all features are risk-increasing"))
    print("  FN average aggregation SHAP              : %.4f (low -> model misses these)" %
          summary_df[summary_df["feature"] == "sp_aggregation_ratio"]["shap_fn_mean"].values[0])
    print()
    print("  Outputs:")
    print("    data/output/shap_global_importance.csv")
    print("    data/output/shap_group_analysis.csv")
    print("    figures/shap_global_bar.png")


if __name__ == "__main__":
    main()
