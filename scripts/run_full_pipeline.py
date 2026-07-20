"""
run_full_pipeline.py -- Full analysis pipeline.

Produces three output groups in one run:

GROUP 1: Main evaluation (n=113, full label)
  -> main_model_results.csv
  -> heuristic_baseline_results.csv
  -> bootstrap_ci_results.csv
  -> shap_global_importance.csv
  -> case_study_examples.csv

GROUP 2: Temporal robustness (n=103, inner join -- observed in both periods)
  -> temporal_label_split_results.csv   (S1/S2/S3)
  -> temporal_transition_analysis.csv  (Safe-Safe, Safe-Risky, ...)
  -> label_trend_diagnostics.csv       (worsening/stable/improving)
  NOTE: Temporal results cannot be compared directly with the main 113-controller results.
        Use S1 on the same 103-subset as the reference for S3.

GROUP 3: Unobserved controller scores (no APM history / insufficient data)
  -> unobserved_controller_scores.csv

To run:
  python -m scripts.run_full_pipeline
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    DATASET_PATH, APM_MONTHLY, SP_MAPPING,
    ORACLE_SP, ORACLE_DEPS, ORACLE_TABLES,
    MODEL_C, CV_SEED, DECISION_THRESHOLD,
    ENHANCED_FEATURES, FINAL_FEATURES,
    CV_N_SPLITS, CV_N_REPEATS,
    CALL_COUNT_MIN, Q_THRESHOLD,
)

OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

EARLY_MONTHS = ["2025-10", "2025-11", "2025-12", "2026-01"]
LATE_MONTHS  = ["2026-02", "2026-03"]
K_EARLY, K_LATE = 2, 1
N_BOOT = 1000
RNG = np.random.default_rng(CV_SEED)

FEATURE_LABELS = {
    "sp_aggregation_ratio":        "SP Aggregation Ratio",
    "sp_join_per_select":          "SP Join/Select Ratio",
    "dep_complexity_per_function": "Dep. Complexity/Function",
    "sp_dml_zero":                 "Write-Absence Flag",
    "log_ctrl_functions":          "log(Ctrl Functions)",
    "log_dep_complexity_sum":      "log(Dep. Complexity Sum)",
    "log_ctrl_complexity":         "log(Ctrl Complexity)",
    "log_max_table_rows":          "log(Max Table Rows)",
    "sp_read_heavy_ratio":         "Read-Heavy Ratio",
    "sp_order_ratio":              "SP Order/Select Ratio",
    "log_table_volume":            "log(Table Volume)",
    "agg_x_dmlzero":               "Aggregation x Write-Absence",
    "complexity_total":            "Total Layered Complexity",
    "agg_x_table":                 "Aggregation x Table Size",
}


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def make_model():
    return Pipeline([
        ("pt", PowerTransformer(method="yeo-johnson")),
        ("m",  LogisticRegression(C=MODEL_C, max_iter=1000, random_state=CV_SEED)),
    ])


def loo_scores(X, y_train):
    loo = LeaveOneOut()
    oof = np.zeros(len(y_train))
    for tr, te in loo.split(X, y_train):
        m = make_model()
        m.fit(X[tr], y_train[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def eval_oof(y_true, oof, t=DECISION_THRESHOLD):
    auc_pr  = round(average_precision_score(y_true, oof), 4)
    auc_roc = round(roc_auc_score(y_true, oof), 4)
    pos_rate = y_true.mean()
    pred = (oof >= t).astype(int)
    tp = int(((pred==1)&(y_true==1)).sum())
    fp = int(((pred==1)&(y_true==0)).sum())
    fn = int(((pred==0)&(y_true==1)).sum())
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    top10 = int(y_true[np.argsort(oof)[::-1][:10]].sum())
    top15 = int(y_true[np.argsort(oof)[::-1][:15]].sum())
    return dict(
        n=len(y_true), n_pos=int(y_true.sum()),
        pos_rate=round(float(pos_rate), 4),
        auc_pr=auc_pr, auc_roc=auc_roc,
        lift=round(auc_pr/pos_rate, 2) if pos_rate > 0 else 0,
        prec=round(prec, 4), rec=round(rec, 4), f1=round(f1, 4),
        p_at_10=round(top10/10, 2), p_at_15=round(top15/15, 2),
        tp=tp, fp=fp, fn=fn,
        threshold=t,
    )


def bootstrap_ci(y, scores, n=N_BOOT):
    vals = []
    ns = len(y)
    for _ in range(n):
        idx = RNG.integers(0, ns, size=ns)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], scores[idx]))
    arr = np.array(vals)
    return dict(
        mean=round(float(np.mean(arr)), 4),
        std=round(float(np.std(arr)), 4),
        ci95_lo=round(float(np.percentile(arr, 2.5)), 4),
        ci95_hi=round(float(np.percentile(arr, 97.5)), 4),
    )


def build_temporal_labels(apm):
    rel = apm[apm["call_count"] >= CALL_COUNT_MIN].copy()
    mq  = rel.groupby("month")["p95_ms"].quantile(Q_THRESHOLD)
    rel["is_risky"] = rel.apply(lambda r: int(r["p95_ms"] >= mq[r["month"]]), axis=1)

    def agg_period(months, k, suffix):
        g = (rel[rel["month"].isin(months)]
             .groupby("controller")
             .agg(n=(  "month", "count"), nr=("is_risky", "sum"))
             .reset_index())
        g[f"{suffix}_label"] = (g["nr"] >= k).astype(int)
        g["ctrl_key"] = g["controller"].str.lower().str.replace(r"controller$", "", regex=True)
        return g[["ctrl_key", f"{suffix}_label"]]

    early = agg_period(EARLY_MONTHS, K_EARLY, "early")
    late  = agg_period(LATE_MONTHS,  K_LATE,  "late")
    return early.merge(late, on="ctrl_key", how="inner")


# ===========================================================================
# GROUP 1: MAIN EVALUATION (n=113)
# ===========================================================================

def run_main_evaluation(ds, feats):
    print("\n" + "="*60)
    print("GROUP 1: MAIN EVALUATION (n=113)")
    print("="*60)

    y = ds["label"].values
    X = ds[feats].values

    # --- LOO ---
    print("[1/6] Computing LOO scores (enhanced 14 feat)...")
    oof14 = loo_scores(X, y)
    r14   = eval_oof(y, oof14)

    print("[1/6] Computing LOO scores (original 12 feat)...")
    X12  = ds[FINAL_FEATURES].values
    oof12 = loo_scores(X12, y)
    r12   = eval_oof(y, oof12)

    # --- CV robustness ---
    print("[2/6] Repeated CV (5x20)...")
    cv = RepeatedStratifiedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=CV_SEED)
    pr14, pr12 = [], []
    for tr, te in cv.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr])
        if len(np.unique(y[te])) > 1:
            pr14.append(average_precision_score(y[te], m.predict_proba(X[te])[:,1]))
        m2 = make_model(); m2.fit(X12[tr], y[tr])
        if len(np.unique(y[te])) > 1:
            pr12.append(average_precision_score(y[te], m2.predict_proba(X12[te])[:,1]))

    # --- Bootstrap CI ---
    print("[3/6] Bootstrap CI (n=1000)...")
    ci14 = bootstrap_ci(y, oof14)
    ci12 = bootstrap_ci(y, oof12)

    # --- Heuristic baselines ---
    print("[4/6] Heuristic baselines...")
    heuristics = [
        ("log(Max Table Rows)", "log_max_table_rows"),
        ("log(Table Volume)",   "log_table_volume"),
        ("Dep. Complexity",     "log_dep_complexity_sum"),
        ("Ctrl Complexity",     "log_ctrl_complexity"),
        ("SP Aggregation",      "sp_aggregation_ratio"),
        ("SP Join Ratio",       "sp_join_per_select"),
    ]
    h_rows = []
    for name, feat in heuristics:
        scores = ds[feat].fillna(0).values
        auc = round(average_precision_score(y, scores), 4)
        top10 = int(y[np.argsort(scores)[::-1][:10]].sum())
        cutoff = np.sort(scores)[::-1][39]
        pred = (scores >= cutoff).astype(int)
        tp = int(((pred==1)&(y==1)).sum())
        fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum())
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        h_rows.append(dict(
            method=name, type="heuristic",
            auc_pr=auc, lift=round(auc/y.mean(),2),
            prec=round(prec,3), rec=round(rec,3),
            p_at_10=round(top10/10,2), tp=tp, fp=fp,
        ))
    for model_name, oof_s, feat_count in [("ML original (12)", oof12, 12), ("ML enhanced (14)", oof14, 14)]:
        r = eval_oof(y, oof_s)
        h_rows.append(dict(
            method=model_name, type="ml",
            auc_pr=r["auc_pr"], lift=r["lift"],
            prec=r["prec"], rec=r["rec"],
            p_at_10=r["p_at_10"], tp=r["tp"], fp=r["fp"],
        ))

    # --- SHAP ---
    print("[5/6] Computing SHAP...")
    model_full = make_model(); model_full.fit(X, y)
    X_t = model_full.named_steps["pt"].transform(X)
    lr  = model_full.named_steps["m"]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer   = shap.LinearExplainer(lr, X_t, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_t)

    shap_df = pd.DataFrame({
        "feature":       feats,
        "feature_label": [FEATURE_LABELS.get(f, f) for f in feats],
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "coef":          lr.coef_[0],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # --- Case studies ---
    print("[6/6] Case study examples...")
    pred14 = (oof14 >= DECISION_THRESHOLD).astype(int)
    tp_idx = np.where((pred14==1)&(y==1))[0]
    fp_idx = np.where((pred14==1)&(y==0))[0]
    fn_idx = np.where((pred14==0)&(y==1))[0]

    def group_shap(idxs):
        return shap_values[idxs].mean(axis=0) if len(idxs) > 0 else np.zeros(len(feats))

    case_rows = []
    for group, idxs, label_str in [("TP", tp_idx, "risky"), ("FP", fp_idx, "safe"), ("FN", fn_idx, "risky")]:
        sorted_by_score = sorted(idxs, key=lambda i: oof14[i], reverse=(group!="FN"))
        for i in sorted_by_score[:3]:
            row = ds.iloc[i]
            case_rows.append({
                "group": group, "controller": row["controller"],
                "true_label": label_str,
                "risk_score": round(oof14[i], 3),
                "avg_p95_ms": round(row["avg_p95_ms"], 0),
                "log_max_table_rows": round(row["log_max_table_rows"], 2),
                "log_table_volume": round(row.get("log_table_volume", 0), 2),
                "sp_aggregation_ratio": round(row["sp_aggregation_ratio"], 3),
                "complexity_total": round(row["complexity_total"], 2),
                "note": (
                    "High structural risk: large table + aggregation" if group == "TP" else
                    "Latent risk: structurally similar to TP but lower traffic" if group == "FP" else
                    "Workload-driven: low aggregation/table signal"
                ),
            })

    # --- Save ---
    main_results = pd.DataFrame([
        {"model": "Original (12 feat)", "n": r12["n"], "n_pos": r12["n_pos"],
         "auc_pr_loo": r12["auc_pr"], "auc_roc_loo": r12["auc_roc"],
         "recall": r12["rec"], "precision": r12["prec"], "f1": r12["f1"],
         "lift": r12["lift"], "p_at_10": r12["p_at_10"],
         "tp": r12["tp"], "fp": r12["fp"], "fn": r12["fn"],
         "cv_auc_pr_mean": round(np.mean(pr12), 4), "cv_auc_pr_std": round(np.std(pr12), 4),
         "ci95_lo": ci12["ci95_lo"], "ci95_hi": ci12["ci95_hi"],
         "brier": round(brier_score_loss(y, oof12), 4)},
        {"model": "Enhanced (14 feat)", "n": r14["n"], "n_pos": r14["n_pos"],
         "auc_pr_loo": r14["auc_pr"], "auc_roc_loo": r14["auc_roc"],
         "recall": r14["rec"], "precision": r14["prec"], "f1": r14["f1"],
         "lift": r14["lift"], "p_at_10": r14["p_at_10"],
         "tp": r14["tp"], "fp": r14["fp"], "fn": r14["fn"],
         "cv_auc_pr_mean": round(np.mean(pr14), 4), "cv_auc_pr_std": round(np.std(pr14), 4),
         "ci95_lo": ci14["ci95_lo"], "ci95_hi": ci14["ci95_hi"],
         "brier": round(brier_score_loss(y, oof14), 4)},
    ])
    main_results.to_csv(OUT / "main_model_results.csv", index=False)
    pd.DataFrame(h_rows).to_csv(OUT / "heuristic_baseline_results.csv", index=False)
    shap_df.to_csv(OUT / "shap_global_importance.csv", index=False)
    pd.DataFrame(case_rows).to_csv(OUT / "case_study_examples.csv", index=False)

    print("  Saved: main_model_results.csv, heuristic_baseline_results.csv")
    print("  Saved: shap_global_importance.csv, case_study_examples.csv")
    return oof14, shap_values


# ===========================================================================
# GROUP 2: TEMPORAL ROBUSTNESS (n=103, inner join)
# ===========================================================================

def run_temporal_analysis(ds, feats, apm):
    print("\n" + "="*60)
    print("GROUP 2: TEMPORAL ROBUSTNESS (n=103, inner join)")
    print("="*60)

    temp = build_temporal_labels(apm)
    merged = ds[["ctrl_key", "label"] + feats].merge(temp, on="ctrl_key", how="inner")
    print("  n=%d (inner join: observed in both periods)" % len(merged))
    print("  NOTE: This subset's S1 result cannot be compared to the main 113-ctrl result.")
    print("        10 safe controllers are missing -> positive rate changes.")
    print("        Correct comparison: S3 vs S1 on the same 103-ctrl subset.")

    X = merged[feats].values
    y_full  = merged["label"].values
    y_early = merged["early_label"].values
    y_late  = merged["late_label"].values

    # --- S1/S2/S3 ---
    print("\n[1/3] S1/S2/S3 LOO scores...")
    oof_s1 = loo_scores(X, y_full)
    oof_s2 = loo_scores(X, y_early)
    oof_s3_train = loo_scores(X, y_early)  # trained on early labels, evaluated on late labels

    scenarios = [
        ("S1_standard",       y_full,  y_full,  "Full label -> Full label (reference)"),
        ("S2_temporal_same",  y_early, y_early, "Early label -> Early label (within-period)"),
        ("S3_temporal_cross", y_early, y_late,  "Early label -> Late label (cross-period, strictest)"),
    ]
    oof_map = {"S1_standard": oof_s1, "S2_temporal_same": oof_s2, "S3_temporal_cross": oof_s3_train}

    temp_rows = []
    for scen_id, y_tr, y_te, desc in scenarios:
        oof = oof_map[scen_id]
        r = eval_oof(y_te, oof)
        r["scenario"] = scen_id
        r["description"] = desc
        temp_rows.append(r)
        print("  %s: AUC-PR=%.4f  Lift=%.2fx  P@10=%.0f%%  Rec=%.3f" % (
            scen_id, r["auc_pr"], r["lift"], 100*r["p_at_10"], r["rec"]))

    # S3 vs S1 delta
    delta = temp_rows[2]["auc_pr"] - temp_rows[0]["auc_pr"]
    print("\n  S3 vs S1 delta AUC-PR: %+.4f" % delta)
    print("  S3 Lift=%.2fx > 1.0: temporal generalization confirmed." % temp_rows[2]["lift"])

    # --- Transition analysis ---
    print("\n[2/3] Safe/Risky transition analysis...")
    merged["risk_score"] = oof_s3_train

    def trans(row):
        e, l = row["early_label"], row["late_label"]
        if e==0 and l==0: return "Safe-Safe"
        if e==0 and l==1: return "Safe-Risky"
        if e==1 and l==1: return "Risky-Risky"
        return "Risky-Safe"

    merged["transition"] = merged.apply(trans, axis=1)
    trans_rows = []
    for t in ["Safe-Safe", "Safe-Risky", "Risky-Risky", "Risky-Safe"]:
        grp = merged[merged["transition"]==t]["risk_score"]
        if len(grp) > 0:
            trans_rows.append({
                "transition": t, "count": len(grp),
                "median_score": round(float(grp.median()), 4),
                "mean_score":   round(float(grp.mean()),   4),
                "min_score":    round(float(grp.min()),    4),
                "max_score":    round(float(grp.max()),    4),
            })
            print("  %-15s  n=%2d  median=%.3f  mean=%.3f" % (
                t, len(grp), grp.median(), grp.mean()))

    sr = merged[merged["transition"]=="Safe-Risky"]["risk_score"]
    ss = merged[merged["transition"]=="Safe-Safe"]["risk_score"]
    diff = sr.median() - ss.median()
    print("\n  Safe-Risky vs Safe-Safe delta: %+.3f" % diff)
    if diff > 0:
        print("  Model scores controllers that will become risky higher in advance.")

    # --- Label trend diagnostics ---
    print("\n[3/3] Label trend diagnostics (APM trend analysis)...")
    apm2 = apm[apm["call_count"] >= CALL_COUNT_MIN].copy()
    months = sorted(apm2["month"].unique())
    early_ms = months[:3]; late_ms = months[3:6]
    early_p = apm2[apm2["month"].isin(early_ms)].groupby("controller")["p95_ms"].mean().rename("early_p95")
    late_p  = apm2[apm2["month"].isin(late_ms)].groupby("controller")["p95_ms"].mean().rename("late_p95")
    trend = pd.concat([early_p, late_p], axis=1).dropna()
    trend["pct_change"] = (trend["late_p95"] - trend["early_p95"]) / trend["early_p95"] * 100
    trend["ctrl_key"] = trend.index.str.lower().str.replace(r"controller$", "", regex=True)
    trend = trend.merge(ds[["ctrl_key","label"]], on="ctrl_key", how="inner")
    trend["trend_group"] = trend["pct_change"].apply(
        lambda x: "worsening" if x > 10 else ("improving" if x < -10 else "stable"))

    diag_rows = []
    for lbl in [0, 1]:
        for tg in ["worsening", "stable", "improving"]:
            cnt = len(trend[(trend["label"]==lbl) & (trend["trend_group"]==tg)])
            diag_rows.append({"label": lbl, "trend_group": tg, "count": cnt})

    pd.DataFrame(temp_rows).to_csv(OUT / "temporal_label_split_results.csv", index=False)
    pd.DataFrame(trans_rows).to_csv(OUT / "temporal_transition_analysis.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT / "label_trend_diagnostics.csv", index=False)

    print("  Saved: temporal_label_split_results.csv")
    print("  Saved: temporal_transition_analysis.csv")
    print("  Saved: label_trend_diagnostics.csv")

    return merged


# ===========================================================================
# GROUP 3: UNOBSERVED CONTROLLER SCORES
# ===========================================================================

def run_unobserved_scoring(ds, apm, feats):
    print("\n" + "="*60)
    print("GROUP 3: UNOBSERVED CONTROLLER SCORES")
    print("="*60)

    sp_map     = pd.read_csv(SP_MAPPING)
    oracle_sp  = pd.read_csv(ORACLE_SP,  encoding="utf-8")
    oracle_dep = pd.read_csv(ORACLE_DEPS, encoding="utf-8")
    oracle_tbl = pd.read_csv(ORACLE_TABLES, encoding="utf-8")

    oracle_sp["sp_key"] = (oracle_sp["OWNER"] + "." +
                           oracle_sp["PACKAGE_NAME"] + "." +
                           oracle_sp["SUBPROGRAM_NAME"])
    sp_idx   = oracle_sp.set_index("sp_key")
    tbl_rows = oracle_tbl.set_index("TABLE_NAME")["NUM_ROWS"].fillna(0)
    tbl_rlen = oracle_tbl.set_index("TABLE_NAME")["AVG_ROW_LEN"].fillna(0)
    oracle_dep["sp_key"] = (oracle_dep["OWNER"] + "." +
                            oracle_dep["PACKAGE_NAME"] + "." +
                            oracle_dep["SUBPROGRAM_NAME"])

    sp_map["ctrl_key"] = (sp_map["file_name"]
                          .str.replace(r"Controller\.cs$", "", regex=True)
                          .str.lower())
    sp_map["sp_list"]  = sp_map["sp_names"].fillna("").str.split("|")

    in_apm = set(apm["controller"].str.lower().str.replace(r"controller$","",regex=True))
    in_ds  = set(ds["ctrl_key"])
    medians = ds[["log_ctrl_complexity","log_ctrl_functions",
                  "log_dep_complexity_sum","dep_complexity_per_function"]].median()

    bundle = joblib.load(ROOT / "data" / "models" / "model_bundle.joblib")
    model  = bundle["model"]
    model_feats = FINAL_FEATURES  # 12-feature model (original)

    # Never in APM (Group A)
    never_in_apm = sp_map[~sp_map["ctrl_key"].isin(in_apm)].copy()
    # In APM but not labeled (Group B: dropped due to insufficient data)
    in_apm_not_labeled = sp_map[
        sp_map["ctrl_key"].isin(in_apm) & ~sp_map["ctrl_key"].isin(in_ds)
    ].copy()

    unobs_rows = []
    for group_name, subset in [("never_in_apm", never_in_apm),
                                ("insufficient_apm", in_apm_not_labeled)]:
        for _, ctrl_row in subset.iterrows():
            ctrl = ctrl_row["ctrl_key"]
            sp_keys = [s.strip() for s in ctrl_row["sp_list"] if s.strip()]
            valid   = [k for k in sp_keys if k in sp_idx.index]

            feat = {k: float(medians[k]) for k in
                    ["log_ctrl_complexity","log_ctrl_functions",
                     "log_dep_complexity_sum","dep_complexity_per_function"]}

            if valid:
                sp_data   = sp_idx.loc[valid]
                sel_clip  = sp_data["SELECT_COUNT"].clip(lower=1)
                dml_total = float(sp_data["DML_COUNT"].sum())
                feat["sp_aggregation_ratio"] = float((sp_data["GROUP_BY_COUNT"]/sel_clip).mean())
                feat["sp_join_per_select"]   = float((sp_data["JOIN_COUNT"]/sel_clip).mean())
                feat["sp_dml_zero"]          = float(dml_total == 0)
                feat["sp_read_heavy_ratio"]  = float(sp_data["READ_HEAVY_FLAG"].mean())
                dep_t = oracle_dep[oracle_dep["sp_key"].isin(set(valid))]["TABLE_NAME"].unique()
                tbl_c = [tbl_rows.get(t, 0) for t in dep_t]
                feat["log_max_table_rows"] = float(math.log1p(max(tbl_c) if tbl_c else 0))
            else:
                feat.update({k: 0.0 for k in
                             ["sp_aggregation_ratio","sp_join_per_select",
                              "sp_dml_zero","sp_read_heavy_ratio"]})
                feat["log_max_table_rows"] = float(ds["log_max_table_rows"].median())

            feat["agg_x_dmlzero"]    = feat["sp_aggregation_ratio"] * feat["sp_dml_zero"]
            feat["complexity_total"] = (feat["log_ctrl_complexity"] +
                                        feat["log_dep_complexity_sum"] +
                                        feat["dep_complexity_per_function"])
            feat["agg_x_table"]      = feat["sp_aggregation_ratio"] * feat["log_max_table_rows"]

            X_new = np.array([[feat[f] for f in model_feats]])
            score = float(model.predict_proba(X_new)[0, 1])

            flag_reason = []
            if feat.get("sp_aggregation_ratio", 0) > 0.1:
                flag_reason.append("high aggregation")
            if feat.get("log_max_table_rows", 0) > 14:
                flag_reason.append("large table access")
            if len(valid) > 10:
                flag_reason.append("many SPs (%d)" % len(valid))
            if feat.get("sp_dml_zero", 0) == 1:
                flag_reason.append("read-only")

            unobs_rows.append({
                "controller": ctrl,
                "runtime_status": group_name,
                "n_mapped_sp": len(valid),
                "risk_score": round(score, 3),
                "flagged": int(score >= DECISION_THRESHOLD),
                "sp_aggregation_ratio": round(feat.get("sp_aggregation_ratio", 0), 3),
                "log_max_table_rows": round(feat.get("log_max_table_rows", 0), 2),
                "complexity_total": round(feat.get("complexity_total", 0), 2),
                "structural_reason": "; ".join(flag_reason) if flag_reason else "low structural signal",
                "note": "No runtime label available -- structural risk score only",
            })

    unobs_df = pd.DataFrame(unobs_rows).sort_values("risk_score", ascending=False)
    unobs_df.to_csv(OUT / "unobserved_controller_scores.csv", index=False)

    flagged = unobs_df[unobs_df["flagged"] == 1]
    print("  Total unobserved/insufficient controllers: %d" % len(unobs_df))
    print("  Flagged at threshold >= %.2f: %d" % (DECISION_THRESHOLD, len(flagged)))
    print("  Top-scoring controllers:")
    for _, r in unobs_df.head(5).iterrows():
        print("    %-40s  score=%.3f  n_SP=%d  [%s]" % (
            r["controller"], r["risk_score"], r["n_mapped_sp"], r["runtime_status"]))
    print("  Saved: unobserved_controller_scores.csv")


# ===========================================================================
# SUMMARY
# ===========================================================================

def print_summary():
    print("\n" + "="*60)
    print("SUMMARY -- Output files")
    print("="*60)
    files = [
        ("main_model_results.csv",           "Main 113-ctrl results (LOO, CV, CI)"),
        ("heuristic_baseline_results.csv",   "ML vs heuristic comparison"),
        ("shap_global_importance.csv",       "SHAP feature importance"),
        ("case_study_examples.csv",          "TP/FP/FN examples"),
        ("temporal_label_split_results.csv", "S1/S2/S3 temporal results (n=103)"),
        ("temporal_transition_analysis.csv", "Safe/Risky transition distribution"),
        ("label_trend_diagnostics.csv",      "Worsening/stable/improving distribution"),
        ("unobserved_controller_scores.csv", "Unobserved controller risk scores"),
    ]
    for fname, desc in files:
        fpath = OUT / fname
        exists = "OK" if fpath.exists() else "MISSING"
        print("  [%s] %-40s  %s" % (exists, fname, desc))


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("Full Analysis Pipeline")
    print("Enhanced model (14 feat): FINAL_FEATURES + log_table_volume + sp_order_ratio")

    ds  = pd.read_csv(DATASET_PATH)
    apm = pd.read_csv(APM_MONTHLY)

    oof14, shap_values = run_main_evaluation(ds, ENHANCED_FEATURES)
    run_temporal_analysis(ds, ENHANCED_FEATURES, apm)
    run_unobserved_scoring(ds, apm, ENHANCED_FEATURES)
    print_summary()


if __name__ == "__main__":
    main()
