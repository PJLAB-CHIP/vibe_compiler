#!/usr/bin/env python3
"""Canonical current runtime launch contracts used by board tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PACKAGE_SCHEMA_VERSION = 7
KERNEL_LAUNCH_KIND = "kernel"
MODEL_LAUNCH_KIND = "model"

RANK_ONE_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "per-rank",
    "entry_abi": "rank-local-pointer-block",
    "phases": ["main"],
}

GRID_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "grid",
    "entry_abi": "rank-major-pointer-table",
    "phases": ["main"],
}

CLUSTER_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "cluster",
    "entry_abi": "rank-major-pointer-table",
    "phases": ["prepare", "main"],
}

MODEL_LAUNCH: dict[str, Any] = {
    "kind": MODEL_LAUNCH_KIND,
    "entry_abi": "tx81-model-bootparam",
    "phases": ["main"],
}

KERNEL_PREPARE_EXPORT = {
    "role": "prepare",
    "symbol": "__wafer_kernel_prepare",
}
KERNEL_MAIN_EXPORT = {"role": "main", "symbol": "main"}


def expected_kernel_module_exports(
    launch: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return the canonical module exports for one kernel launch contract."""

    if launch.get("kind") != KERNEL_LAUNCH_KIND:
        raise RuntimeError("kernel module exports require a kernel launch")
    phases = launch.get("phases")
    if not isinstance(phases, list):
        raise RuntimeError("kernel launch phases must be a list")
    exports_by_phase = {
        "prepare": KERNEL_PREPARE_EXPORT,
        "main": KERNEL_MAIN_EXPORT,
    }
    if any(phase not in exports_by_phase for phase in phases):
        raise RuntimeError("kernel launch contains an unsupported phase")
    return [dict(exports_by_phase[phase]) for phase in phases]


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
