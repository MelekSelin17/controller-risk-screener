"""Shared controller key normalization helpers."""

from __future__ import annotations

import re


HTTP_METHOD_RE = re.compile(r"^(GET|POST|PUT|DELETE|HEAD|PATCH|OPTIONS)\s+", re.IGNORECASE)
SKIP_CONTROLLER_KEYS = {
    "",
    "base",
    "content",
    # Add application-specific prefixes to skip here
    "elasticlogger",
    "unknown route",
    "index.html",
}

# Known naming mismatches (APM key -> canonical controller key)
CONTROLLER_OVERRIDES = {
    # Add system-specific controller key aliases here, e.g. "old_name": "canonical_name"
}


def normalize_controller_key(value: str) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("controller.cs", "")
    s = s.replace("controller", "")
    s = s.replace(".cs", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return CONTROLLER_OVERRIDES.get(s, s)


def extract_controller_from_endpoint(endpoint: str) -> str:
    """Extract normalized controller key from raw endpoint string."""
    s = HTTP_METHOD_RE.sub("", str(endpoint or "").strip())
    s = s.split(" ")[0].lstrip("/")
    first_segment = s.split("/")[0]
    return normalize_controller_key(first_segment)


def is_valid_controller_key(value: str) -> bool:
    s = str(value or "").strip().lower()
    return s not in SKIP_CONTROLLER_KEYS and s != ""


def normalize_ctrl_key_from_file(file_name: str) -> str:
    # Backward-compatible behavior for legacy joins.
    return (
        str(file_name)
        .lower()
        .replace("controller.cs", "controller")
        .replace(".cs", "")
        .strip()
    )


def class_name_from_file(file_name: str) -> str:
    return str(file_name).replace(".cs", "").strip()
