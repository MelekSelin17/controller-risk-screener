"""
warn.py — CLI entrypoint.

Usage:
  python -m pipeline.warn --file OrderController.cs
  python -m pipeline.warn --file OrderController.cs --project-root . --verbose
  python -m pipeline.warn --file OrderController.cs --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.predict import predict, DECISION_THRESHOLD


# ---------------------------------------------------------------------------
# Colored terminal output
# ---------------------------------------------------------------------------
_COLORS = {
    "CRITICAL": "\033[91m",  # red
    "HIGH":     "\033[91m",  # red
    "MEDIUM":   "\033[93m",  # yellow
    "LOW":      "\033[92m",  # green
    "OK":       "\033[92m",  # green
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
    "DIM":      "\033[2m",
}

_ALERT_LABELS = {
    "CRITICAL": "!! CRITICAL — must review before release",
    "HIGH":     "!! HIGH     — review recommended before merge",
    "MEDIUM":   "!  MEDIUM   — flagged with lower confidence",
    "OK":       "OK SAFE     — below risk threshold",
}


def _c(text: str, *keys: str) -> str:
    if not sys.stdout.isatty():
        return text
    prefix = "".join(_COLORS.get(k, "") for k in keys)
    return prefix + text + _COLORS["RESET"]


def _conf_bar(weighted_conf: float, width: int = 20) -> str:
    filled = int(round(weighted_conf * width))
    bar = "#" * filled + "-" * (width - filled)
    pct = int(round(weighted_conf * 100))
    return "[%s] %d%%" % (bar, pct)


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def _print_human(result: dict, verbose: bool = False) -> None:
    ctrl    = result["controller_name"].capitalize() + "Controller"
    score   = result["risk_score"]
    alert   = result["alert_level"]
    mode    = result["mode"]
    flagged = result["flagged"]
    eff_t   = result["effective_threshold"]
    wconf   = result["weighted_confidence"]
    conf    = result["confidence"]

    # ----- Header -----
    print()
    print(_c("=" * 58, "BOLD"))
    print(_c("  CONTROLLER RISK ANALYSIS", "BOLD"))
    print(_c("=" * 58, "BOLD"))
    print("  Controller : %s" % ctrl)
    print("  Mode       : %s" % ("Edit (existing)" if mode == "edit" else "New controller"))
    print()

    # ----- Alert band -----
    alert_text = _ALERT_LABELS.get(alert, alert)
    color_key  = "HIGH" if alert in ("CRITICAL", "HIGH") else ("MEDIUM" if alert == "MEDIUM" else "OK")
    print("  %s" % _c(alert_text, color_key, "BOLD"))
    print()

    # ----- Score + threshold -----
    print("  Risk Score     : %.4f  (threshold=%.2f, base=%.2f)" % (score, eff_t, DECISION_THRESHOLD))
    if eff_t > DECISION_THRESHOLD:
        print(_c(
            "  Threshold raised: Low confidence — +%.2f (%.0f%% weighted data)"
            % (eff_t - DECISION_THRESHOLD, 100 * wconf),
            "DIM",
        ))

    # ----- Confidence bar -----
    bar_color = "OK" if conf == "high" else ("MEDIUM" if conf == "medium" else "HIGH")
    print("  Confidence [%s]: %s" % (conf.upper(), _c(_conf_bar(wconf), bar_color)))

    if result.get("missing_heavy_features"):
        heavy = ", ".join(result["missing_heavy_features"])
        print(_c("  Critical missing: %s (imputed from median)" % heavy, "DIM"))

    # ----- Delta — only in edit mode -----
    if mode == "edit" and "score_delta" in result:
        delta = result["score_delta"]
        hist  = result["historical_risk_score"]
        dir_  = result["delta_direction"]
        print()
        print(_c("  CHANGE ANALYSIS", "BOLD"))
        print("  Previous score : %.4f" % hist)
        print("  New score      : %.4f  (%+.4f — %s)" % (
            score, delta, _c(dir_, "HIGH" if delta > 0.01 else "OK")))

        if result.get("changed_features"):
            print()
            print("  Changed features:")
            for ch in result["changed_features"]:
                impact_str = _c("%+.4f risk impact" % ch["risk_impact"],
                                "HIGH" if ch["risk_impact"] > 0 else "OK")
                print("    * %-38s  %.4f -> %.4f  (%s)" % (
                    ch["label"], ch["before"], ch["after"], impact_str))

    # ----- Risk reasons -----
    print()
    print(_c("  RISK REASONS", "BOLD"))
    if result["top_reasons"]:
        for i, reason in enumerate(result["top_reasons"], 1):
            print("  %d. %s" % (i, reason))
    else:
        print("  (No real-source features — no explanation available)")

    # ----- Suggestions -----
    if result.get("suggestions"):
        print()
        print(_c("  SUGGESTIONS", "BOLD"))
        for sug in result["suggestions"]:
            print("  -> %s" % sug)

    # ----- Verbose extra info -----
    if verbose:
        if result.get("missing_features"):
            print()
            print(_c("  Imputed from median : %s" % ", ".join(result["missing_features"]), "DIM"))
        if result.get("notes"):
            print()
            for note in result["notes"]:
                print(_c("  [Note] %s" % note, "DIM"))

    print()
    print(_c("=" * 58, "BOLD"))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Controller performance risk screening tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--file",         type=Path, required=True,
                   help="Path to the .cs file to analyse")
    p.add_argument("--project-root", type=Path, default=None,
                   help="Project root directory for handler SP resolution (optional)")
    p.add_argument("--artifact",     type=Path, default=None,
                   help="Model artifact path (default: data/models/model_bundle.joblib)")
    p.add_argument("--json",         action="store_true",
                   help="Output results in JSON format")
    p.add_argument("--verbose",      action="store_true",
                   help="Show missing features and notes")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.file.exists():
        print("Error: File not found: %s" % args.file, file=sys.stderr)
        sys.exit(1)

    try:
        result = predict(
            file_path=args.file,
            project_root=args.project_root,
            artifact_path=args.artifact,
        )
    except FileNotFoundError as e:
        print("Error: %s" % e, file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result, verbose=args.verbose)

    # Exit code: 1 = warning issued (for CI/CD pipeline integration)
    sys.exit(1 if result["flagged"] else 0)


if __name__ == "__main__":
    main()
