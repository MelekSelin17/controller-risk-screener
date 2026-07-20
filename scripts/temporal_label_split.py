"""
temporal_label_split.py — Temporal label split evaluation.

Reviewer 2: "features and labels derived from overlapping historical periods"

Solution:
  - TRAIN LABEL : early period (Oct-Jan, 4 months) -> risky?
  - TEST  LABEL : late  period (Feb-Mar, 2 months) -> risky?
  - FEATURE     : same static feature vector for both periods

Three scenario comparison:
  S1 - Standard LOO      : full label (7 months), standard paper protocol
  S2 - Temporal LOO same : early label -> early label prediction (within-period)
  S3 - Temporal LOO cross: early label -> late label prediction (cross-period, strictest)

S3 directly addresses Reviewer 2's request: "past data predicts later behavior."
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    DATASET_PATH, APM_MONTHLY,
    MODEL_C, CV_SEED, ENHANCED_FEATURES, DECISION_THRESHOLD,
    CALL_COUNT_MIN, Q_THRESHOLD,
)

OUTPUT_DIR = ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Period definitions
EARLY_MONTHS = ['2025-10', '2025-11', '2025-12', '2026-01']  # 4 months
LATE_MONTHS  = ['2026-02', '2026-03']                         # 2 months
K_EARLY = 2   # early period: risky in at least 2 months
K_LATE  = 1   # late  period: risky in at least 1 month


def build_temporal_labels(apm: pd.DataFrame) -> pd.DataFrame:
    """
    Build separate labels for early and late periods.
    """
    rel = apm[apm['call_count'] >= CALL_COUNT_MIN].copy()

    # Monthly relative threshold (within each period independently)
    monthly_q = rel.groupby('month')['p95_ms'].quantile(Q_THRESHOLD)
    rel['is_risky_month'] = rel.apply(
        lambda r: int(r['p95_ms'] >= monthly_q[r['month']]), axis=1
    )

    # Early period label
    early = (rel[rel['month'].isin(EARLY_MONTHS)]
             .groupby('controller')
             .agg(n_early=('month','count'),
                  n_risky_early=('is_risky_month','sum'))
             .reset_index())
    early['early_label'] = (early['n_risky_early'] >= K_EARLY).astype(int)

    # Late period label
    late = (rel[rel['month'].isin(LATE_MONTHS)]
            .groupby('controller')
            .agg(n_late=('month','count'),
                 n_risky_late=('is_risky_month','sum'))
            .reset_index())
    late['late_label'] = (late['n_risky_late'] >= K_LATE).astype(int)

    merged = early.merge(late, on='controller', how='inner')
    merged['ctrl_key'] = (merged['controller']
                          .str.lower()
                          .str.replace(r'controller$', '', regex=True))
    return merged


def loo_eval(X: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    LOO: train with y_train on each fold, evaluate with y_test.
    y_train == y_test -> standard LOO
    y_train != y_test -> temporal cross-period LOO
    """
    loo = LeaveOneOut()
    oof = np.zeros(len(y_test))

    for tr, te in loo.split(X, y_train):
        m = Pipeline([
            ('pt', PowerTransformer(method='yeo-johnson')),
            ('m',  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
        ])
        m.fit(X[tr], y_train[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    # Compute metrics on y_test
    auc_pr  = round(average_precision_score(y_test, oof), 4)
    auc_roc = round(roc_auc_score(y_test, oof), 4)

    pred = (oof >= DECISION_THRESHOLD).astype(int)
    tp = int(((pred==1) & (y_test==1)).sum())
    fp = int(((pred==1) & (y_test==0)).sum())
    fn = int(((pred==0) & (y_test==1)).sum())
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    top10 = int(y_test[np.argsort(oof)[::-1][:10]].sum())

    pos_rate = y_test.mean()
    return {
        'n':        len(y_test),
        'n_pos':    int(y_test.sum()),
        'pos_rate': round(float(pos_rate), 4),
        'auc_pr':   auc_pr,
        'auc_roc':  auc_roc,
        'prec':     round(prec, 4),
        'rec':      round(rec, 4),
        'f1':       round(f1, 4),
        'lift':     round(auc_pr / pos_rate, 2) if pos_rate > 0 else 0,
        'p_at_10':  round(top10/10, 2),
        'tp': tp, 'fp': fp, 'fn': fn,
    }


def print_result(name: str, r: dict):
    print('  %-38s  n=%d  pos=%d(%.0f%%)' % (name, r['n'], r['n_pos'], 100*r['pos_rate']))
    print('  %-38s  AUC-PR=%.4f  AUC-ROC=%.4f  Lift=%.2fx' % ('', r['auc_pr'], r['auc_roc'], r['lift']))
    print('  %-38s  Prec=%.1f%%  Rec=%.1f%%  F1=%.3f  P@10=%.0f%%  TP=%d/FP=%d' % (
        '', 100*r['prec'], 100*r['rec'], r['f1'], 100*r['p_at_10'], r['tp'], r['fp']))


def main():
    print('=' * 65)
    print('TEMPORAL LABEL SPLIT EVALUATION')
    print('=' * 65)
    print('Train label : Early period (%s) K>=%d' % (str(EARLY_MONTHS), K_EARLY))
    print('Test  label : Late  period (%s) K>=%d' % (str(LATE_MONTHS), K_LATE))
    print()

    # Load data
    ds  = pd.read_csv(DATASET_PATH)
    apm = pd.read_csv(APM_MONTHLY)

    # Temporal labels
    temp_labels = build_temporal_labels(apm)
    # LEFT JOIN: missing period = label 0
    # Rationale: no traffic means no observable performance issue -> safe (0)
    # This approach preserves all 113 controllers and stays consistent with the main model
    merged = ds[['ctrl_key', 'label'] + ENHANCED_FEATURES].merge(
        temp_labels[['ctrl_key', 'early_label', 'late_label']],
        on='ctrl_key', how='left'
    )
    merged['early_label'] = merged['early_label'].fillna(0).astype(int)
    merged['late_label']  = merged['late_label'].fillna(0).astype(int)

    print('Label distribution:')
    print('  Full label (S1)    : n=%d, pos=%d (%d%%)' % (
        len(merged), merged['label'].sum(), 100*merged['label'].mean()))
    print('  Early label (S2/S3): n=%d, pos=%d (%d%%)' % (
        len(merged), merged['early_label'].sum(), 100*merged['early_label'].mean()))
    print('  Late  label (S3)   : n=%d, pos=%d (%d%%)' % (
        len(merged), merged['late_label'].sum(), 100*merged['late_label'].mean()))
    print()

    # Label stability
    stable = (merged['early_label'] == merged['late_label']).sum()
    print('  Label stability (early==late): %d/%d = %.1f%%' % (
        stable, len(merged), 100*stable/len(merged)))
    print()

    X = merged[ENHANCED_FEATURES].values
    y_full  = merged['label'].values
    y_early = merged['early_label'].values
    y_late  = merged['late_label'].values

    results = []
    SEP = '-' * 65

    # --- S1: Standard LOO (paper protocol) ---
    print(SEP)
    print('S1: Standard LOO (full label -> full label)')
    print('    Same as paper protocol — reference point')
    print(SEP)
    s1 = loo_eval(X, y_full, y_full)
    print_result('Standard LOO', s1)
    results.append({'scenario': 'S1_standard_loo', **s1})
    print()

    # --- S2: Early -> Early (within-period temporal) ---
    print(SEP)
    print('S2: Temporal LOO — within-period (early label -> early label)')
    print('    Train and evaluate the model on the early period behavior')
    print(SEP)
    s2 = loo_eval(X, y_early, y_early)
    print_result('Temporal LOO (same period)', s2)
    results.append({'scenario': 'S2_temporal_same', **s2})
    print()

    # --- S3: Early -> Late (cross-period temporal) ---
    print(SEP)
    print('S3: Temporal LOO — cross-period (early label -> LATE label)')
    print('    Train on early-period labels, evaluate on LATE-period labels')
    print('    Strictest temporal test: "does past structural risk predict future behavior?"')
    print(SEP)
    s3 = loo_eval(X, y_early, y_late)
    print_result('Temporal LOO (cross-period)', s3)
    results.append({'scenario': 'S3_temporal_cross', **s3})
    print()

    # --- Summary table ---
    print('=' * 65)
    print('SUMMARY COMPARISON')
    print('=' * 65)
    print('%-30s  %5s  %6s  %6s  %5s  %6s  %6s' % (
        'Scenario', 'n_pos', 'AUC-PR', 'AUC-ROC', 'Lift', 'Recall', 'F1'))
    print('-' * 65)
    names = {
        'S1_standard_loo':   'S1: Standard (full label)',
        'S2_temporal_same':  'S2: Temporal same-period',
        'S3_temporal_cross': 'S3: Temporal cross-period',
    }
    for r in results:
        print('  %-28s  %5d  %.4f  %.4f  %4.2fx  %.3f  %.3f' % (
            names[r['scenario']], r['n_pos'],
            r['auc_pr'], r['auc_roc'], r['lift'],
            r['rec'], r['f1']))

    print()
    print('Interpretation:')
    delta = s3['auc_pr'] - s1['auc_pr']
    print('  S3 vs S1 delta AUC-PR : %+.4f' % delta)
    if s3['auc_pr'] > s3['pos_rate']:
        print('  S3 Lift = %.2fx > 1.0 -> model beats random (temporal generalization confirmed)' % s3['lift'])
    else:
        print('  S3 Lift < 1.0 -> limited temporal generalization')

    # Save CSV
    out = OUTPUT_DIR / 'temporal_split_results.csv'
    pd.DataFrame(results).to_csv(out, index=False)
    print()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
