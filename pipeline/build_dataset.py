"""
build_dataset.py — Controller-level dataset construction.

APM   → label (runtime data, used only here for label construction)
Oracle + SonarQube → static features (valid for all scenarios)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import (
    APM_MONTHLY, SONAR_PROCESSED, ORACLE_SP, ORACLE_DEPS,
    ORACLE_TABLES, SP_MAPPING, DATASET_PATH,
    CALL_COUNT_MIN, Q_THRESHOLD, MIN_RISKY_MONTHS,
    BASE_FEATURES, INTERACTION_FEATURES, FINAL_FEATURES,
)


# ---------------------------------------------------------------------------
# 1. LABEL BUILDER
# ---------------------------------------------------------------------------

def build_labels(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    APM monthly data → controller-level binary label.

    Label = 1 : controller appears in top-25% slowest for at least MIN_RISKY_MONTHS reliable months.
    Label = 0 : otherwise.

    Reliable month: call_count >= CALL_COUNT_MIN.
    "Slow"        : p95_ms >= Q_THRESHOLD percentile for that month.
    """
    rel = monthly[monthly["call_count"] >= CALL_COUNT_MIN].copy()
    monthly_q = rel.groupby("month")["p95_ms"].quantile(Q_THRESHOLD)

    rel["is_risky_month"] = rel.apply(
        lambda r: int(r["p95_ms"] >= monthly_q[r["month"]]), axis=1
    )
    agg = rel.groupby("controller").agg(
        n_reliable=("month", "count"),
        n_risky=("is_risky_month", "sum"),
        risky_ratio=("is_risky_month", "mean"),
        avg_p95_ms=("p95_ms", "mean"),
        max_p95_ms=("p95_ms", "max"),
    ).reset_index()

    agg["label"] = (agg["n_risky"] >= MIN_RISKY_MONTHS).astype(int)
    return agg


# ---------------------------------------------------------------------------
# 2. DATABASE FEATURE BUILDER
# ---------------------------------------------------------------------------

def build_oracle_features(
    sp_map: pd.DataFrame,
    oracle_sp: pd.DataFrame,
    oracle_deps: pd.DataFrame,
    oracle_tbl: pd.DataFrame,
) -> pd.DataFrame:
    """
    controller_sp_mapping + DB metadata → controller-level static features.
    All features are static/structural — no runtime APM data.
    """
    sp_map = sp_map.copy()
    sp_map["ctrl_key"] = (
        sp_map["file_name"]
        .str.replace(r"Controller\.cs$", "", regex=True)
        .str.lower()
    )
    sp_map["sp_list"] = sp_map["sp_names"].fillna("").str.split("|")

    # SP lookup: OWNER.PACKAGE_NAME.SUBPROGRAM_NAME → row
    oracle_sp = oracle_sp.copy()
    oracle_sp["sp_key"] = (
        oracle_sp["OWNER"] + "." +
        oracle_sp["PACKAGE_NAME"] + "." +
        oracle_sp["SUBPROGRAM_NAME"]
    )
    sp_idx = oracle_sp.set_index("sp_key")

    # Table sizes and data volume (rows × average row length)
    tbl_rows   = oracle_tbl.set_index("TABLE_NAME")["NUM_ROWS"].fillna(0)
    tbl_rowlen = oracle_tbl.set_index("TABLE_NAME")["AVG_ROW_LEN"].fillna(0)

    # SP → table dependencies
    oracle_deps = oracle_deps.copy()
    oracle_deps["sp_key"] = (
        oracle_deps["OWNER"] + "." +
        oracle_deps["PACKAGE_NAME"] + "." +
        oracle_deps["SUBPROGRAM_NAME"]
    )

    rows = []
    for _, ctrl_row in sp_map.iterrows():
        ctrl = ctrl_row["ctrl_key"]
        sp_keys = [s.strip() for s in ctrl_row["sp_list"] if s.strip()]
        valid_sps = [k for k in sp_keys if k in sp_idx.index]

        if not valid_sps:
            rows.append({"ctrl_key": ctrl})
            continue

        sp_data = sp_idx.loc[valid_sps]
        select_clip = sp_data["SELECT_COUNT"].clip(lower=1)
        dml_total = float(sp_data["DML_COUNT"].sum())

        # Core DB features
        sp_join_per_select    = float((sp_data["JOIN_COUNT"] / select_clip).mean())
        sp_aggregation_ratio  = float((sp_data["GROUP_BY_COUNT"] / select_clip).mean())
        sp_dml_zero           = float(dml_total == 0)
        sp_read_heavy_ratio   = float(sp_data["READ_HEAVY_FLAG"].mean())
        sp_line_count         = float(sp_data["SOURCE_LINE_COUNT"].sum())
        sp_dml_count          = dml_total

        # Enhanced DB features — richer signals
        loop_total            = (sp_data["LOOP_COUNT"] + sp_data["FOR_COUNT"]).sum()
        sp_loop_ratio         = float((loop_total / select_clip.sum()))
        sp_order_ratio        = float((sp_data["ORDER_BY_COUNT"] / select_clip).mean())

        # Table information
        dep_tables   = oracle_deps[oracle_deps["sp_key"].isin(set(valid_sps))]["TABLE_NAME"].unique()
        tbl_counts   = [tbl_rows.get(t, 0) for t in dep_tables]
        tbl_vols     = [tbl_rows.get(t, 0) * tbl_rowlen.get(t, 0) for t in dep_tables]
        max_tbl_rows = max(tbl_counts) if tbl_counts else 0
        max_tbl_vol  = max(tbl_vols)   if tbl_vols   else 0
        n_huge_tbl   = sum(1 for r in tbl_counts if r > 1e8)

        rows.append({
            "ctrl_key":             ctrl,
            "sp_join_per_select":   round(sp_join_per_select, 6),
            "sp_aggregation_ratio": round(sp_aggregation_ratio, 6),
            "sp_dml_zero":          sp_dml_zero,
            "sp_read_heavy_ratio":  round(sp_read_heavy_ratio, 6),
            "sp_line_count":        sp_line_count,
            "sp_dml_count":         sp_dml_count,
            "log_max_table_rows":   float(np.log1p(max_tbl_rows)),
            "n_huge_tables":        float(n_huge_tbl),
            # Enhanced features
            "sp_loop_ratio":        round(sp_loop_ratio, 6),
            "sp_order_ratio":       round(sp_order_ratio, 6),
            "log_table_volume":     float(np.log1p(max_tbl_vol)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. SONARQUBE FEATURES (pre-processed)
# ---------------------------------------------------------------------------

SONAR_COLS = [
    "controller",
    "log_ctrl_complexity",
    "log_ctrl_functions",
    "log_dep_complexity_sum",
    "dep_complexity_per_function",
    "log_sp_line_count",
]


def load_sonar_features(sonar_path=SONAR_PROCESSED) -> pd.DataFrame:
    df = pd.read_csv(sonar_path)[SONAR_COLS].copy()
    df["ctrl_key"] = (
        df["controller"]
        .str.lower()
        .str.replace(r"controller$", "", regex=True)
    )
    return df.drop(columns=["controller"])


# ---------------------------------------------------------------------------
# 4. INTERACTION FEATURES
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features validated through ablation.
    Computed on raw feature values, not after PowerTransform.
    """
    df = df.copy()
    df["agg_x_dmlzero"] = df["sp_aggregation_ratio"] * df["sp_dml_zero"]
    df["complexity_total"] = (
        df["log_ctrl_complexity"] +
        df["log_dep_complexity_sum"] +
        df["dep_complexity_per_function"]
    )
    df["agg_x_table"] = df["sp_aggregation_ratio"] * df["log_max_table_rows"]
    return df


# ---------------------------------------------------------------------------
# 5. MAIN DATASET BUILD FUNCTION
# ---------------------------------------------------------------------------

def build_controller_dataset(save: bool = True) -> pd.DataFrame:
    """
    Merges all sources and returns the controller-level final dataset.
    save=True → saves as CSV to DATASET_PATH.
    """
    print("[1/5] Loading APM data...")
    monthly = pd.read_csv(APM_MONTHLY)

    print("[2/5] Loading Oracle data...")
    oracle_sp   = pd.read_csv(ORACLE_SP,   encoding="utf-8")
    oracle_deps = pd.read_csv(ORACLE_DEPS, encoding="utf-8")
    oracle_tbl  = pd.read_csv(ORACLE_TABLES, encoding="utf-8")
    sp_map      = pd.read_csv(SP_MAPPING)

    print("[3/5] Building labels (Q=%.2f, min_risky_months=%d)..." % (Q_THRESHOLD, MIN_RISKY_MONTHS))
    labels = build_labels(monthly)
    labels["ctrl_key"] = (
        labels["controller"]
        .str.lower()
        .str.replace(r"controller$", "", regex=True)
    )

    print("[4/5] Merging features...")
    oracle_feats = build_oracle_features(sp_map, oracle_sp, oracle_deps, oracle_tbl)
    sonar_feats  = load_sonar_features()

    ds = (
        labels
        .merge(sonar_feats,  on="ctrl_key", how="left")
        .merge(oracle_feats, on="ctrl_key", how="left")
    )
    ds = add_interaction_features(ds)

    # Keep only controllers with all features present
    before = len(ds)
    ds = ds.dropna(subset=FINAL_FEATURES)
    after = len(ds)
    print("  %d/%d controllers have all features." % (after, before))

    print("[5/5] Dataset statistics:")
    print("  Total controllers : %d" % len(ds))
    print("  Positive (risky)  : %d (%.1f%%)" % (ds["label"].sum(), 100*ds["label"].mean()))
    print("  Negative (safe)   : %d" % (ds["label"]==0).sum())

    if save:
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        ds.to_csv(DATASET_PATH, index=False)
        print("  Saved: %s" % DATASET_PATH)

    return ds


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__ + "/../../.."))
    build_controller_dataset()
