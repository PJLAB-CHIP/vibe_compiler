#!/usr/bin/env python3
"""Canonical schema-v6 runtime launch contracts used by board tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PACKAGE_SCHEMA_VERSION = 6
KERNEL_LAUNCH_KIND = "kernel"
MODEL_LAUNCH_KIND = "model"

RANK_ONE_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "per-rank",
    "entry_abi": "rank-local-pointer-block-v1",
    "phases": ["main"],
}

GRID_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "grid",
    "entry_abi": "rank-major-pointer-table-v1",
    "phases": ["main"],
}

CLUSTER_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "cluster",
    "entry_abi": "rank-major-pointer-table-v1",
    "phases": ["prepare", "main"],
}

MODEL_LAUNCH: dict[str, Any] = {
    "kind": MODEL_LAUNCH_KIND,
    "entry_abi": "tx81-model-bootparam-v1",
    "phases": ["main"],
}


def require_manifest_launch(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{context} package schema is not "
            f"{PACKAGE_SCHEMA_VERSION}"
        )
    target = manifest.get("target")
    if not isinstance(target, Mapping) or target.get("launch") != expected:
        raise RuntimeError(f"{context} runtime launch contract is invalid")
