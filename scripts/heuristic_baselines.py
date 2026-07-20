"""
heuristic_baselines.py — Response to Reviewer 2.

"Compare the model against rankings based only on table size,
dependency complexity, or historical latency."

Each heuristic ranks by a single feature and computes AUC-PR.
Compared against the ML model (LOO). Lift vs random is also reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    DATASET_PATH, MODEL_C, CV_SEED,
    FINAL_FEATURES, ENHANCED_FEATURES, DECISION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Heuristic baseline definitions
# ---------------------------------------------------------------------------

HEURISTICS = [
    {
        "name": "Largest table (rows)",
        "feature": "log_max_table_rows",
        "desc": "Rank by max accessed table size (row count)",
    },
    {
        "name": "Largest table (volume)",
        "feature": "log_table_volume",
        "desc": "Rank by max accessed table volume (rows × avg_row_len)",
    },
    {
        "name": "Dependency complexity",
        "feature": "log_dep_complexity_sum",
        "desc": "Rank by total downstream dependency complexity",
    },
    {
        "name": "Controller complexity",
        "feature": "log_ctrl_complexity",
        "desc": "Rank by controller cyclomatic complexity",
    },
    {
        "name": "SP aggregation ratio",
        "feature": "sp_aggregation_ratio",
        "desc": "Rank by GROUP BY / SELECT ratio in stored procedures",
    },
    {
        "name": "SP join ratio",
        "feature": "sp_join_per_select",
        "desc": "Rank by JOIN / SELECT ratio in stored procedures",
    },
]


# ---------------------------------------------------------------------------
# LOO evaluator for ML model
# ---------------------------------------------------------------------------

def eval_ml_loo(ds: pd.DataFrame, y: np.ndarray, features: list[str]) -> dict:
    X = ds[features].values
    loo = LeaveOneOut()
    oof = np.zeros(len(y))
    for tr, te in loo.split(X, y):
        m = Pipeline([
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
        ])
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    pred = (oof >= DECISION_THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    top10_tp = int(y[np.argsort(oof)[::-1][:10]].sum())

    pos_rate = y.mean()
    return {
        "auc_pr":   round(average_precision_score(y, oof), 4),
        "auc_roc":  round(roc_auc_score(y, oof), 4),
        "prec":     round(prec, 4),
        "rec":      round(rec, 4),
        "f1":       round(f1, 4),
        "lift":     round(average_precision_score(y, oof) / pos_rate, 2),
        "p_at_10":  round(top10_tp / 10, 2),
        "tp": tp, "fp": fp,
    }


# ---------------------------------------------------------------------------
# Heuristic evaluator — rank by single feature only
# ---------------------------------------------------------------------------

def eval_heuristic(ds: pd.DataFrame, y: np.ndarray, feature: str) -> dict:
    if feature not in ds.columns:
        return {"auc_pr": None, "auc_roc": None, "lift": None, "p_at_10": None}

    scores = ds[feature].fillna(0).values
    pos_rate = y.mean()
    auc_pr  = round(average_precision_score(y, scores), 4)
    auc_roc = round(roc_auc_score(y, scores), 4)
    top10_tp = int(y[np.argsort(scores)[::-1][:10]].sum())

    # threshold @ top-40 (same review budget as ML flagging ~40)
    cutoff = np.sort(scores)[::-1][39]
    pred = (scores >= cutoff).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "auc_pr":  auc_pr,
        "auc_roc": auc_roc,
        "prec":    round(prec, 4),
        "rec":     round(rec, 4),
        "f1":      round(f1, 4),
        "lift":    round(auc_pr / pos_rate, 2),
        "p_at_10": round(top10_tp / 10, 2),
        "tp": tp, "fp": fp,
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main():
    ds = pd.read_csv(DATASET_PATH)
    y  = ds["label"].values
    pos_rate = y.mean()
    n_pos = int(y.sum())

    print("=" * 72)
    print("HEURISTIC BASELINE COMPARISON")
    print("=" * 72)
    print(f"Dataset: n={len(y)}, pos={n_pos} ({100*pos_rate:.1f}%)")
    print(f"Random baseline AUC-PR = {pos_rate:.4f}  (Lift=1.00x)")
    print()

    results = []

    # --- Heuristic baselines ---
    print("Heuristic baselines (single-feature ranking):")
    print("-" * 72)
    for h in HEURISTICS:
        r = eval_heuristic(ds, y, h["feature"])
        results.append({"model": h["name"], "type": "heuristic", **r})
        print(f"  {h['name']:<30}  AUC-PR={r['auc_pr']:.4f}  "
              f"Lift={r['lift']:.2f}x  P@10={r['p_at_10']:.0%}  "
              f"Rec={r['rec']:.3f}  TP={r['tp']}/FP={r['fp']}")

    print()

    # --- ML model (original 12 features) ---
    print("ML Model (LOO):")
    print("-" * 72)
    print("  [Running: Original 12-feature model...]")
    r12 = eval_ml_loo(ds, y, FINAL_FEATURES)
    results.append({"model": "ML model (12 feat, original)", "type": "ml", **r12})
    print(f"  {'ML model (12 feat, original)':<30}  AUC-PR={r12['auc_pr']:.4f}  "
          f"Lift={r12['lift']:.2f}x  P@10={r12['p_at_10']:.0%}  "
          f"Rec={r12['rec']:.3f}  TP={r12['tp']}/FP={r12['fp']}")

    print("  [Running: Enhanced 14-feature model...]")
    r14 = eval_ml_loo(ds, y, ENHANCED_FEATURES)
    results.append({"model": "ML model (14 feat, enhanced)", "type": "ml", **r14})
    print(f"  {'ML model (14 feat, enhanced)':<30}  AUC-PR={r14['auc_pr']:.4f}  "
          f"Lift={r14['lift']:.2f}x  P@10={r14['p_at_10']:.0%}  "
          f"Rec={r14['rec']:.3f}  TP={r14['tp']}/FP={r14['fp']}")

    print()

    # --- Summary table ---
    print("=" * 72)
    print("SUMMARY TABLE (for paper)")
    print("=" * 72)
    header = f"{'Method':<32}  {'Type':<10}  {'AUC-PR':>6}  {'Lift':>6}  {'Recall':>6}  {'P@10':>5}"
    print(header)
    print("-" * 72)
    for r in results:
        marker = " <--" if r["type"] == "ml" else ""
        print(f"  {r['model']:<30}  {r['type']:<10}  "
              f"{r['auc_pr']:>6.4f}  {r['lift']:>5.2f}x  "
              f"{r['rec']:>6.3f}  {r['p_at_10']:>5.0%}{marker}")
    print()
    print(f"  {'Random (baseline)':<30}  {'random':<10}  "
          f"{pos_rate:>6.4f}  {'1.00x':>6}  {'—':>6}  {'—':>5}")

    # Save CSV
    out = ROOT / "data" / "output" / "heuristic_baseline_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
