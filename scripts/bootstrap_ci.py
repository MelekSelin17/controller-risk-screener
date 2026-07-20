"""
bootstrap_ci.py — Bootstrap confidence intervals for LOO AUC-PR.

Reviewer 2: "Small differences may not be reliable without significance testing."

LOO produces a single point estimate. Bootstrap resampling is used to compute
CI95 and to test the statistical significance of the difference between the
original and enhanced models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    DATASET_PATH, MODEL_C, CV_SEED,
    FINAL_FEATURES, ENHANCED_FEATURES,
)

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(CV_SEED)


# ---------------------------------------------------------------------------
# Generate LOO out-of-fold scores
# ---------------------------------------------------------------------------

def get_loo_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    loo = LeaveOneOut()
    oof = np.zeros(len(y))
    for tr, te in loo.split(X, y):
        m = Pipeline([
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
        ])
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_auc_pr(y: np.ndarray, scores: np.ndarray, n: int = N_BOOTSTRAP) -> dict:
    """Compute AUC-PR CI95 using bootstrap resampling."""
    boot_vals = []
    n_samples = len(y)
    for _ in range(n):
        idx = RNG.integers(0, n_samples, size=n_samples)
        y_b = y[idx]
        s_b = scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_vals.append(average_precision_score(y_b, s_b))
    boot_arr = np.array(boot_vals)
    return {
        "mean":  round(float(np.mean(boot_arr)), 4),
        "std":   round(float(np.std(boot_arr)), 4),
        "ci95_lo": round(float(np.percentile(boot_arr, 2.5)), 4),
        "ci95_hi": round(float(np.percentile(boot_arr, 97.5)), 4),
        "n_boot": len(boot_vals),
    }


# ---------------------------------------------------------------------------
# Paired bootstrap test — is the difference between two models significant?
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n: int = N_BOOTSTRAP,
) -> dict:
    """
    H0: AUC-PR(A) == AUC-PR(B)
    p-value computed via bootstrap.
    """
    obs_diff = average_precision_score(y, scores_b) - average_precision_score(y, scores_a)
    n_samples = len(y)
    count_ge = 0
    for _ in range(n):
        idx = RNG.integers(0, n_samples, size=n_samples)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        diff_b = (
            average_precision_score(y_b, scores_b[idx]) -
            average_precision_score(y_b, scores_a[idx])
        )
        if diff_b >= obs_diff:
            count_ge += 1
    p_val = count_ge / n
    return {
        "observed_diff": round(obs_diff, 4),
        "p_value": round(p_val, 4),
        "significant_at_0.10": p_val < 0.10,
        "significant_at_0.05": p_val < 0.05,
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main():
    ds = pd.read_csv(DATASET_PATH)
    y  = ds["label"].values
    pos_rate = y.mean()

    print("=" * 65)
    print("BOOTSTRAP CONFIDENCE INTERVALS — LOO AUC-PR")
    print(f"n_bootstrap={N_BOOTSTRAP}, seed={CV_SEED}")
    print("=" * 65)
    print(f"Dataset: n={len(y)}, pos={int(y.sum())} ({100*pos_rate:.1f}%)")
    print()

    # --- Original 12-feature model ---
    print("[1/2] Computing LOO scores for original model (12 features)...")
    X12 = ds[FINAL_FEATURES].values
    oof12 = get_loo_scores(X12, y)
    obs_auc12 = round(average_precision_score(y, oof12), 4)
    ci12 = bootstrap_auc_pr(y, oof12)
    print(f"  LOO AUC-PR = {obs_auc12:.4f}")
    print(f"  Bootstrap  = {ci12['mean']:.4f} +/- {ci12['std']:.4f}")
    print(f"  CI95       = [{ci12['ci95_lo']:.4f}, {ci12['ci95_hi']:.4f}]")
    print()

    # --- Enhanced 14-feature model ---
    print("[2/2] Computing LOO scores for enhanced model (14 features)...")
    X14 = ds[ENHANCED_FEATURES].values
    oof14 = get_loo_scores(X14, y)
    obs_auc14 = round(average_precision_score(y, oof14), 4)
    ci14 = bootstrap_auc_pr(y, oof14)
    print(f"  LOO AUC-PR = {obs_auc14:.4f}")
    print(f"  Bootstrap  = {ci14['mean']:.4f} +/- {ci14['std']:.4f}")
    print(f"  CI95       = [{ci14['ci95_lo']:.4f}, {ci14['ci95_hi']:.4f}]")
    print()

    # --- Paired test ---
    print("Paired bootstrap test (Enhanced vs Original):")
    test = paired_bootstrap_test(y, oof12, oof14)
    print(f"  Observed delta AUC-PR = +{test['observed_diff']:.4f}")
    print(f"  p-value (one-sided)   = {test['p_value']:.4f}")
    print(f"  Significant at 0.10   = {test['significant_at_0.10']}")
    print(f"  Significant at 0.05   = {test['significant_at_0.05']}")
    print()

    # --- Summary table ---
    print("=" * 65)
    print("SUMMARY TABLE (for paper)")
    print("=" * 65)
    print(f"{'Model':<30}  {'AUC-PR':>6}  {'CI95':>20}  {'Lift':>6}")
    print("-" * 65)
    print(f"  {'Original (12 feat)':<28}  {obs_auc12:.4f}  "
          f"[{ci12['ci95_lo']:.4f}, {ci12['ci95_hi']:.4f}]  "
          f"{obs_auc12/pos_rate:.2f}x")
    print(f"  {'Enhanced (14 feat)':<28}  {obs_auc14:.4f}  "
          f"[{ci14['ci95_lo']:.4f}, {ci14['ci95_hi']:.4f}]  "
          f"{obs_auc14/pos_rate:.2f}x")
    print(f"  {'Random baseline':<28}  {pos_rate:.4f}  {'—':>20}  1.00x")

    # Save CSV
    out = ROOT / "data" / "output" / "bootstrap_ci_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"model": "Original (12 feat)", "loo_auc_pr": obs_auc12,
         "boot_mean": ci12["mean"], "boot_std": ci12["std"],
         "ci95_lo": ci12["ci95_lo"], "ci95_hi": ci12["ci95_hi"]},
        {"model": "Enhanced (14 feat)", "loo_auc_pr": obs_auc14,
         "boot_mean": ci14["mean"], "boot_std": ci14["std"],
         "ci95_lo": ci14["ci95_lo"], "ci95_hi": ci14["ci95_hi"]},
    ]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
