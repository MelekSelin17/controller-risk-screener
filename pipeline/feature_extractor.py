"""
feature_extractor.py — Live feature extraction from a .cs file.

Three sources:
  1. .cs file → C# metrics (regex)
  2. Project root handler scan → SP names
  3. Oracle CSV lookup → SP/table metrics
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    ORACLE_SP, ORACLE_DEPS, ORACLE_TABLES, SP_MAPPING,
    BASE_FEATURES, INTERACTION_FEATURES, FINAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Regex patterns — controller .cs file
# ---------------------------------------------------------------------------
_BRANCH_RE    = re.compile(r"\b(if|for(each)?|while|switch|case|catch)\b|&&|\|\||\?", re.I)
_FUNC_RE      = re.compile(r"\b(public|private|protected|internal)\b[^{;=]+\(", re.I)
_DISPATCH_RE  = re.compile(r"\bdispatcher\.(?:Query|Command)\s*<\s*(\w+Request)", re.I)
_HANDLER_RE   = re.compile(r"IQueryHandler\s*<\s*(\w+Request)\s*,", re.I)
# Pattern matches OWNER.PACKAGE.PROCEDURE format — adapt to your DB schema owner
_SP_NAME_RE   = re.compile(r'"([A-Z0-9_]+\.[A-Z0-9_]+\.[A-Z0-9_]+)"')
_SP_CALL_RE   = re.compile(
    r"dataContext\.(GetList|GetSingle|GetScalar|ExecuteNonQuery|Execute)\s*[<(]", re.I
)
_CTOR_RE      = re.compile(r"public\s+\w+Controller\s*\(([^)]*)\)", re.I | re.DOTALL)


# ---------------------------------------------------------------------------
# Cache Oracle tables in memory (loaded on first access)
# ---------------------------------------------------------------------------

_oracle_cache: dict | None = None


def _load_oracle_cache() -> dict:
    global _oracle_cache
    if _oracle_cache is not None:
        return _oracle_cache

    sp_df = pd.read_csv(ORACLE_SP, encoding="utf-8")
    sp_df["sp_key"] = (
        sp_df["OWNER"] + "." + sp_df["PACKAGE_NAME"] + "." + sp_df["SUBPROGRAM_NAME"]
    )
    sp_idx = sp_df.set_index("sp_key")

    dep_df = pd.read_csv(ORACLE_DEPS, encoding="utf-8")
    dep_df["sp_key"] = (
        dep_df["OWNER"] + "." + dep_df["PACKAGE_NAME"] + "." + dep_df["SUBPROGRAM_NAME"]
    )

    tbl_df = pd.read_csv(ORACLE_TABLES, encoding="utf-8")
    tbl_rows = tbl_df.set_index("TABLE_NAME")["NUM_ROWS"].fillna(0)

    sp_map_df = pd.read_csv(SP_MAPPING)
    sp_map_df["ctrl_key"] = (
        sp_map_df["file_name"]
        .str.replace(r"Controller\.cs$", "", regex=True)
        .str.lower()
    )
    sp_map_df["sp_list"] = sp_map_df["sp_names"].fillna("").str.split("|")

    _oracle_cache = {
        "sp_idx":   sp_idx,
        "dep_df":   dep_df,
        "tbl_rows": tbl_rows,
        "sp_map":   sp_map_df,
    }
    return _oracle_cache


# ---------------------------------------------------------------------------
# Compute SP metrics from SP keys
# ---------------------------------------------------------------------------

def _sp_metrics_from_keys(sp_keys: list[str]) -> dict:
    cache = _load_oracle_cache()
    sp_idx   = cache["sp_idx"]
    dep_df   = cache["dep_df"]
    tbl_rows = cache["tbl_rows"]

    valid = [k for k in sp_keys if k in sp_idx.index]
    if not valid:
        return {}

    sp_data = sp_idx.loc[valid]
    sel_clip  = sp_data["SELECT_COUNT"].clip(lower=1)
    dml_total = float(sp_data["DML_COUNT"].sum())

    dep_tables   = dep_df[dep_df["sp_key"].isin(set(valid))]["TABLE_NAME"].unique()
    tbl_counts   = [tbl_rows.get(t, 0) for t in dep_tables]
    max_tbl_rows = max(tbl_counts) if tbl_counts else 0

    return {
        "sp_aggregation_ratio": float((sp_data["GROUP_BY_COUNT"] / sel_clip).mean()),
        "sp_join_per_select":   float((sp_data["JOIN_COUNT"]     / sel_clip).mean()),
        "sp_dml_zero":          float(dml_total == 0),
        "sp_read_heavy_ratio":  float(sp_data["READ_HEAVY_FLAG"].mean()),
        "log_max_table_rows":   float(math.log1p(max_tbl_rows)),
    }


# ---------------------------------------------------------------------------
# Handler scan — request type → SP names
# ---------------------------------------------------------------------------

def _build_handler_index(project_root: Path) -> dict[str, list[str]]:
    """Scans *Handler.cs files under project_root, returns request→SP names mapping."""
    index: dict[str, list[str]] = {}
    for path in project_root.rglob("*Handler.cs"):
        if any(p.lower() in {"obj", "bin", "test", "tests"} for p in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        request_types = [m.group(1) for m in _HANDLER_RE.finditer(text)]
        sp_names = [m.group(1) for m in _SP_NAME_RE.finditer(text)]
        for req in request_types:
            index.setdefault(req, []).extend(sp_names)
    return index


# ---------------------------------------------------------------------------
# Main extraction result
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFeatures:
    controller_name: str
    feature_values:  dict[str, float] = field(default_factory=dict)
    feature_sources: dict[str, str]   = field(default_factory=dict)   # "parsed" | "oracle" | "historical" | "median"
    missing:         list[str]        = field(default_factory=list)
    notes:           list[str]        = field(default_factory=list)

    @property
    def completeness(self) -> float:
        extracted = sum(1 for s in self.feature_sources.values() if s != "median")
        return extracted / max(len(FINAL_FEATURES), 1)


# ---------------------------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------------------------------

def extract_features(
    file_path: Path,
    project_root: Path | None = None,
    historical_profile: dict[str, float] | None = None,
    training_stats: dict[str, dict] | None = None,
) -> ExtractedFeatures:
    """
    Extract features from a .cs file and Oracle DB metadata.

    historical_profile : known feature values if controller is in the training dataset
    training_stats     : median imputation for missing features
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    ctrl_name = file_path.stem.lower().replace("controller", "")

    feat: dict[str, float] = {}
    src:  dict[str, str]   = {}
    notes: list[str]       = []

    # ---- 1. C# metrics (.cs parse) ----
    branch_count   = len(_BRANCH_RE.findall(text))
    func_count     = len(_FUNC_RE.findall(text))
    dispatch_reqs  = [m.group(1) for m in _DISPATCH_RE.finditer(text)]

    log_complexity = float(math.log1p(branch_count + 1))
    log_functions  = float(math.log1p(func_count))

    feat["log_ctrl_complexity"] = log_complexity
    feat["log_ctrl_functions"]  = log_functions
    src["log_ctrl_complexity"]  = "parsed"
    src["log_ctrl_functions"]   = "parsed"

    # ---- 2. SP names — directly in the controller? ----
    direct_sp_keys = [m.group(1) for m in _SP_NAME_RE.finditer(text)]

    # ---- 3. Handler scan → SP names ----
    handler_sp_keys: list[str] = []
    if project_root and project_root.exists() and dispatch_reqs:
        handler_index = _build_handler_index(project_root)
        for req in dispatch_reqs:
            handler_sp_keys.extend(handler_index.get(req, []))

    all_sp_keys = list(set(direct_sp_keys + handler_sp_keys))

    # ---- 4. Oracle lookup ----
    sp_metrics = _sp_metrics_from_keys(all_sp_keys)
    if sp_metrics:
        for k, v in sp_metrics.items():
            feat[k] = v
            src[k]  = "oracle"
    elif all_sp_keys:
        notes.append("SP names found (%d) but no match in DB metadata." % len(all_sp_keys))
    else:
        notes.append("No SP names found. DB metrics will be imputed from historical data or training medians.")

    # ---- 5. Find controller in SP mapping (direct match) ----
    if not sp_metrics:
        cache = _load_oracle_cache()
        ctrl_map_row = cache["sp_map"][cache["sp_map"]["ctrl_key"] == ctrl_name]
        if not ctrl_map_row.empty:
            mapped_keys = [
                s.strip()
                for s in ctrl_map_row.iloc[0]["sp_list"]
                if s.strip()
            ]
            if mapped_keys:
                sp_metrics = _sp_metrics_from_keys(mapped_keys)
                for k, v in sp_metrics.items():
                    feat[k] = v
                    src[k]  = "oracle_mapping"

    # ---- 6. Dependency metrics: historical if available, else median ----
    dep_features = ["log_dep_complexity_sum", "dep_complexity_per_function"]
    for df in dep_features:
        if df not in feat:
            if historical_profile and df in historical_profile:
                feat[df] = historical_profile[df]
                src[df]  = "historical"
            elif training_stats and df in training_stats:
                feat[df] = training_stats[df]["median"]
                src[df]  = "median"
                notes.append("%s: No SonarQube dependency data, using training median." % df)

    # ---- 7. Fallback for all missing BASE features ----
    for f in BASE_FEATURES:
        if f not in feat:
            if historical_profile and f in historical_profile:
                feat[f] = historical_profile[f]
                src[f]  = "historical"
            elif training_stats and f in training_stats:
                feat[f] = training_stats[f]["median"]
                src[f]  = "median"
            else:
                feat[f] = 0.0
                src[f]  = "zero_fallback"

    # ---- 8. Interaction features (deterministic) ----
    feat["agg_x_dmlzero"]    = feat.get("sp_aggregation_ratio", 0) * feat.get("sp_dml_zero", 0)
    feat["complexity_total"] = (
        feat.get("log_ctrl_complexity", 0) +
        feat.get("log_dep_complexity_sum", 0) +
        feat.get("dep_complexity_per_function", 0)
    )
    feat["agg_x_table"]      = feat.get("sp_aggregation_ratio", 0) * feat.get("log_max_table_rows", 0)
    src["agg_x_dmlzero"]    = "computed"
    src["complexity_total"] = "computed"
    src["agg_x_table"]      = "computed"

    missing = [f for f in FINAL_FEATURES if src.get(f) in {"median", "zero_fallback", None}]

    return ExtractedFeatures(
        controller_name=ctrl_name,
        feature_values=feat,
        feature_sources=src,
        missing=missing,
        notes=notes,
    )
