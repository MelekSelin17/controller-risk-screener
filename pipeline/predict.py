"""
predict.py — Model loading, risk score, delta analysis, explanation.

Two modes:
  "edit" — controller is in the training dataset; delta is shown
  "new"  — unseen controller; absolute risk score only
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.config import (
    ARTIFACT_PATH, DATASET_PATH, FINAL_FEATURES, DECISION_THRESHOLD,
)
from pipeline.feature_extractor import ExtractedFeatures, extract_features

# ---------------------------------------------------------------------------
# Risk levels and alert layer
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {"high": 0.55, "medium": 0.35}

# Alert levels — based on risk score only, independent of confidence
ALERT_SCORE_THRESHOLDS = {"critical": 0.45, "high": 0.35, "medium": 0.25}

# Confidence-adjusted threshold: the less real data, the stricter
# weighted_conf >= 0.60 → base (0.30)
# 0.35 <= weighted_conf < 0.60 → base + 0.05
# weighted_conf < 0.35 → base + 0.10
_CONF_THRESHOLD_BUMPS = [(0.60, 0.0), (0.35, 0.05), (0.0, 0.10)]

FEATURE_LABELS = {
    "sp_aggregation_ratio":    "SQL GROUP BY density",
    "sp_join_per_select":      "SQL JOIN density",
    "dep_complexity_per_function": "Complexity per dependency",
    "sp_dml_zero":             "Read-only controller (no writes)",
    "log_ctrl_functions":      "Controller function count",
    "log_dep_complexity_sum":  "Total dependency complexity",
    "log_ctrl_complexity":     "Controller cyclomatic complexity",
    "log_max_table_rows":      "Largest accessed table size",
    "sp_read_heavy_ratio":     "SP read-heavy ratio",
    "agg_x_dmlzero":           "Read-only + aggregation combination (high risk)",
    "complexity_total":        "Total complexity score",
    "agg_x_table":             "GROUP BY density on large table",
}

SUGGESTIONS = {
    "sp_aggregation_ratio":    "Review GROUP BY queries in stored procedures; remove unnecessary aggregations or move them to the application layer.",
    "sp_join_per_select":      "Reduce JOIN count in stored procedures; check index usage on large tables.",
    "dep_complexity_per_function": "Simplify business logic in handlers or split into separate handlers.",
    "log_max_table_rows":      "Optimize queries against very large tables (100M+ rows) with partitioning or caching strategies.",
    "agg_x_dmlzero":           "This controller is read-only with high aggregation — the riskiest pattern. Consider caching results.",
    "complexity_total":        "Total complexity is high. Break the controller into smaller, single-responsibility handlers.",
    "agg_x_table":             "High GROUP BY query density on a very large table detected. Cache results or move aggregation to the application layer.",
    "log_ctrl_functions":      "Controller has too many actions. Move unrelated actions to separate controllers.",
    "sp_dml_zero":             "Controller is read-only — heavy read patterns often lead to latency.",
    "log_dep_complexity_sum":  "Total dependency complexity is high. Simplify the service layer.",
    "log_ctrl_complexity":     "High cyclomatic complexity inside the controller. Move business logic to handlers.",
    "sp_read_heavy_ratio":     "Stored procedures are predominantly read-heavy — risk of latency on large datasets.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    if score >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


def _alert_level(score: float) -> str:
    """Score-based alert level — independent of confidence."""
    if score >= ALERT_SCORE_THRESHOLDS["critical"]:
        return "CRITICAL"
    if score >= ALERT_SCORE_THRESHOLDS["high"]:
        return "HIGH"
    if score >= ALERT_SCORE_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "OK"


def _weighted_confidence(
    feat_sources: dict[str, str],
    feature_importance: dict[str, float],
) -> float:
    """
    Weighted confidence score based on feature importance.
    A missing high-importance Oracle feature incurs a much larger penalty
    than a missing low-importance SonarQube feature.
    """
    total_w = sum(feature_importance.values()) or 1.0
    real_w = sum(
        w for feat, w in feature_importance.items()
        if feat_sources.get(feat) not in {"median", "zero_fallback", None}
    )
    return round(real_w / total_w, 3)


def _effective_threshold(base: float, weighted_conf: float) -> float:
    """Raise the threshold based on weighted confidence."""
    for min_conf, bump in _CONF_THRESHOLD_BUMPS:
        if weighted_conf >= min_conf:
            return round(base + bump, 2)
    return round(base + 0.10, 2)


def _fmt_feature_value(feat: str, val: float) -> str:
    """Format a feature value for human-readable display."""
    if feat == "sp_dml_zero":
        return "Yes" if val > 0.5 else "No"
    if feat.startswith("log_"):
        raw = math.expm1(val)
        if raw > 1e6:
            return "%.0fM (log=%.2f)" % (raw / 1e6, val)
        return "%.0f (log=%.2f)" % (raw, val)
    return "%.3f" % val


def _top_driving_features(
    bundle: dict,
    feat_vec: dict[str, float],
    feat_sources: dict[str, str],
    n: int = 3,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns the most important features by weighted model contribution.
    Only features from "parsed" or "oracle*" sources are selected (not median).

    Returns: (reasons, suggestions, top_feature_names)
    """
    model = bundle["model"]
    stats = bundle["training_stats"]
    coefs = model.named_steps["m"].coef_[0]
    feat_names = bundle["feature_names"]

    contributions: dict[str, float] = {}
    for feat, coef in zip(feat_names, coefs):
        src = feat_sources.get(feat, "median")
        if src in {"median", "zero_fallback"}:
            continue   # exclude median-imputed features from explanation
        s = stats[feat]
        std = s["std"] if s["std"] > 1e-8 else 1.0
        z = (feat_vec[feat] - s["mean"]) / std
        contributions[feat] = float(coef * z)

    # Only features with positive contribution to risk
    positive = {k: v for k, v in contributions.items() if v > 0}
    pool = positive if positive else contributions
    ordered = sorted(pool.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

    reasons     = []
    suggestions = []
    top_feats   = []
    for feat, _ in ordered:
        label = FEATURE_LABELS.get(feat, feat)
        val   = _fmt_feature_value(feat, feat_vec[feat])
        mean_val = _fmt_feature_value(feat, stats[feat]["mean"])
        reasons.append("%s = %s (fleet average: %s)" % (label, val, mean_val))
        if feat in SUGGESTIONS:
            suggestions.append(SUGGESTIONS[feat])
        top_feats.append(feat)

    return reasons, suggestions, top_feats


# ---------------------------------------------------------------------------
# DELTA ANALYSIS
# ---------------------------------------------------------------------------

def _compute_delta(
    current: dict[str, float],
    historical: dict[str, float],
    feat_names: list[str],
    coefs: np.ndarray,
    stats: dict,
) -> list[dict]:
    """Which features changed and how much did they impact the risk score."""
    changes = []
    for feat, coef in zip(feat_names, coefs):
        if feat not in historical or feat not in current:
            continue
        delta_raw = current[feat] - historical[feat]
        if abs(delta_raw) < 1e-6:
            continue
        s   = stats[feat]
        std = s["std"] if s["std"] > 1e-8 else 1.0
        # coefficient * normalized delta → impact on risk score
        risk_impact = float(coef * delta_raw / std)
        changes.append({
            "feature":     feat,
            "label":       FEATURE_LABELS.get(feat, feat),
            "before":      round(historical[feat], 4),
            "after":       round(current[feat], 4),
            "delta":       round(delta_raw, 4),
            "risk_impact": round(risk_impact, 4),
            "direction":   "risk increased" if risk_impact > 0 else "risk decreased",
        })
    return sorted(changes, key=lambda x: abs(x["risk_impact"]), reverse=True)[:5]


# ---------------------------------------------------------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------------------------------------------------------

def predict(
    file_path: Path,
    project_root: Path | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """
    Analyse a controller .cs file and return a risk score with explanation.

    Return fields:
      controller_name, mode, risk_score, risk_level,
      top_reasons, suggestions, confidence,
      [score_delta, changed_features]  <- only in "edit" mode
      feature_values, missing_features, notes
    """
    # 1. Load model bundle
    apath = artifact_path or ARTIFACT_PATH
    if not apath.exists():
        raise FileNotFoundError("Model not found: %s — run train_and_evaluate.py first." % apath)
    bundle: dict = joblib.load(apath)
    stats   = bundle["training_stats"]
    model   = bundle["model"]
    coefs   = model.named_steps["m"].coef_[0]

    # 2. Historical profile: is this controller in the training dataset?
    historical_profile: dict[str, float] | None = None
    if DATASET_PATH.exists():
        ds = pd.read_csv(DATASET_PATH)
        ctrl_key = file_path.stem.lower().replace("controller", "")
        match = ds[ds["ctrl_key"].astype(str).str.lower() == ctrl_key]
        if not match.empty:
            row = match.iloc[0]
            historical_profile = {
                f: float(row[f]) for f in FINAL_FEATURES if f in row.index
            }

    # 3. Feature extraction
    extracted: ExtractedFeatures = extract_features(
        file_path=file_path,
        project_root=project_root,
        historical_profile=historical_profile,
        training_stats=stats,
    )

    # 4. Build feature vector
    feat_vec = extracted.feature_values
    X = np.array([[feat_vec.get(f, stats[f]["median"]) for f in FINAL_FEATURES]])

    # 5. Risk score
    risk_score = float(model.predict_proba(X)[0, 1])
    risk_lvl   = _risk_level(risk_score)

    # 6. Determine mode
    mode = "edit" if historical_profile else "new"

    # 7. Explanation — only from real-source features
    reasons, suggestions, top_feats = _top_driving_features(
        bundle, feat_vec, extracted.feature_sources, n=3
    )
    if not reasons:
        reasons = ["Static code metrics are close to fleet average."]

    # 8. Confidence score (two layers)
    feature_importance = bundle.get("feature_importance", {})

    # 8a. Naive confidence — how many features from real sources
    n_real = sum(
        1 for s in extracted.feature_sources.values()
        if s not in {"median", "zero_fallback"}
    )
    confidence_score = round(n_real / max(len(FINAL_FEATURES), 1), 2)

    # 8b. Weighted confidence — by feature importance
    weighted_conf = _weighted_confidence(extracted.feature_sources, feature_importance)
    confidence_label = (
        "high"   if weighted_conf >= 0.60 else
        "medium" if weighted_conf >= 0.35 else
        "low"
    )

    # 9. Confidence-adjusted threshold and alert level
    eff_threshold  = _effective_threshold(DECISION_THRESHOLD, weighted_conf)
    alert_lvl      = _alert_level(risk_score)

    # Which high-importance features are missing? (only heavy features listed)
    missing_heavy = [
        feat for feat in FINAL_FEATURES
        if extracted.feature_sources.get(feat) in {"median", "zero_fallback", None}
        and feature_importance.get(feat, 0) >= 0.09
    ]

    result: dict[str, Any] = {
        "controller_name":        extracted.controller_name,
        "mode":                   mode,
        "risk_score":             round(risk_score, 4),
        "risk_level":             risk_lvl,
        "alert_level":            alert_lvl,
        "flagged":                risk_score >= eff_threshold,
        "effective_threshold":    eff_threshold,
        "top_reasons":            reasons,
        "suggestions":            suggestions,
        "confidence_score":       confidence_score,
        "weighted_confidence":    weighted_conf,
        "confidence":             confidence_label,
        "missing_heavy_features": missing_heavy,
        "feature_values":         {f: round(feat_vec.get(f, 0), 4) for f in FINAL_FEATURES},
        "feature_sources":        extracted.feature_sources,
        "missing_features":       extracted.missing,
        "notes":                  extracted.notes,
    }

    # 10. Delta analysis — only in "edit" mode
    if mode == "edit" and historical_profile:
        historical_X = np.array([[historical_profile.get(f, stats[f]["median"]) for f in FINAL_FEATURES]])
        historical_score = float(model.predict_proba(historical_X)[0, 1])
        score_delta      = round(risk_score - historical_score, 4)

        changed = _compute_delta(
            current=feat_vec,
            historical=historical_profile,
            feat_names=FINAL_FEATURES,
            coefs=coefs,
            stats=stats,
        )

        result["historical_risk_score"] = round(historical_score, 4)
        result["score_delta"]           = score_delta
        result["delta_direction"]       = (
            "risk increased" if score_delta > 0.01 else
            "risk decreased" if score_delta < -0.01 else
            "unchanged"
        )
        result["changed_features"] = changed

    return result


def predict_to_json(
    file_path: Path,
    project_root: Path | None = None,
    artifact_path: Path | None = None,
    indent: int = 2,
) -> str:
    payload = predict(file_path=file_path, project_root=project_root, artifact_path=artifact_path)
    return json.dumps(payload, ensure_ascii=False, indent=indent)
