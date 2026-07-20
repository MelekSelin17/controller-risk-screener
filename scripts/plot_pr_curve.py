"""
plot_pr_curve.py — IEEE-ready PR curve figure for UBMK 2026.
Output: paper/figures/pr_curve_ml_vs_heuristic.png  (300 DPI, 3.5 x 2.8 in)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.config import DATASET_PATH, MODEL_C, CV_SEED, ENHANCED_FEATURES, DECISION_THRESHOLD

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def run_loo(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X, y):
        pipe = Pipeline([
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
        ])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def smooth_pr(prec, rec, n=400, sigma=7.0):
    idx = np.argsort(rec)
    rg  = np.linspace(0.0, 1.0, n)
    pi  = np.interp(rg, rec[idx], prec[idx])
    return rg, np.clip(gaussian_filter1d(pi, sigma=sigma), 0.0, 1.0)


def make_fig(ds: pd.DataFrame, oof_ml: np.ndarray) -> None:
    y          = ds["label"].values
    prevalence = float(y.mean())

    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["Times New Roman", "DejaVu Serif"],
        "font.size":       8,
        "axes.labelsize":  8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth":  0.7,
        "grid.linewidth":  0.4,
        "grid.color":      "#D5D5D5",
        "figure.dpi":      300,
    })

    # -----------------------------------------------------------------------
    # Color palette:
    #   ML model blue (single color) + three heuristics in different dark colors +
    #   random light gray.  Colors are muted (not bright).
    # -----------------------------------------------------------------------
    C_ML   = "#2166AC"  # muted blue      — main model
    C_H1   = "#4D4D4D"  # dark gray       — Controller complexity (best)
    C_H2   = "#8C510A"  # dark brown      — Max. table rows
    C_H3   = "#35617A"  # muted steel blue-gray — Dependency complexity
    C_RAND = "#BBBBBB"  # light gray      — random

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, linewidth=0.4, color="#D5D5D5", zorder=0)
    ax.set_axisbelow(True)

    # -- ML model ----------------------------------------------------------
    prec_ml, rec_ml, _ = precision_recall_curve(y, oof_ml)
    auc_ml = average_precision_score(y, oof_ml)
    rx, px = smooth_pr(prec_ml, rec_ml)
    ax.plot(rx, px, color=C_ML, lw=2.0, ls="-", zorder=5,
            label=f"ML model (AUC-PR = {auc_ml:.3f})")

    # -- Heuristic baselines ------------------------------------------------
    heuristics = [
        ("Controller complexity", ds["log_ctrl_complexity"].values,    C_H1, "--",           1.4),
        ("Max. table rows",       ds["log_max_table_rows"].values,     C_H2, (0,(5,2,1,2)), 1.3),
        ("Dependency complexity", ds["log_dep_complexity_sum"].values, C_H3, (0,(2,2)),      1.3),
    ]
    for name, scores, col, ls, lw in heuristics:
        prec_h, rec_h, _ = precision_recall_curve(y, scores)
        auc_h = average_precision_score(y, scores)
        rh, ph = smooth_pr(prec_h, rec_h)
        ax.plot(rh, ph, color=col, lw=lw, linestyle=ls, zorder=4,
                label=f"{name} ({auc_h:.3f})")

    # -- Random baseline ----------------------------------------------------
    ax.axhline(prevalence, color=C_RAND, lw=0.85, ls=(0,(6,4)), zorder=3,
               label=f"Random ({prevalence:.3f})")

    # -- Operating point ----------------------------------------------------
    oof_bin = (oof_ml >= DECISION_THRESHOLD).astype(int)
    tp  = int(((oof_bin == 1) & (y == 1)).sum())
    fp  = int(((oof_bin == 1) & (y == 0)).sum())
    fn  = int(((oof_bin == 0) & (y == 1)).sum())
    op_prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    op_rec  = tp / (tp + fn) if tp + fn > 0 else 0.0

    # Show as a simple marker in the legend
    ax.plot(op_rec, op_prec,
            marker="D", ms=5.0, ls="none",
            color="white", mec=C_ML, mew=1.4, zorder=8,
            label=f"Operating threshold ($t={DECISION_THRESHOLD}$)")

    # -- Axes -----------------------------------------------------------
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    # -- Legend: lower left, not overlapping curves ------------------
    leg = ax.legend(
        loc="lower left",
        framealpha=0.97,
        edgecolor="#C0C0C0",
        handlelength=2.4,
        labelspacing=0.30,
        borderpad=0.55,
    )
    leg.get_frame().set_linewidth(0.5)

    # -- Save -------------------------------------------------------------
    plt.tight_layout(pad=0.35)
    out_path = OUT / "pr_curve_ml_vs_heuristic.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Loading dataset...")
    ds = pd.read_csv(DATASET_PATH)
    print(f"  n={len(ds)}, pos={int(ds['label'].sum())} ({100*ds['label'].mean():.1f}%)")
    print(f"Running LOO (n={len(ds)})...")
    oof_ml = run_loo(ds[ENHANCED_FEATURES].values, ds["label"].values)
    print("Generating figure...")
    make_fig(ds, oof_ml)
    print("Done.")


if __name__ == "__main__":
    main()
