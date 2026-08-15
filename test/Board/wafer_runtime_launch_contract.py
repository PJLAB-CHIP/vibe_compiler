#!/usr/bin/env python3
"""Canonical current runtime launch contracts used by board tests."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


KERNEL_LAUNCH_KIND = "kernel"
MODEL_LAUNCH_KIND = "model"
TARGET_TILE_COUNT = 16
LOCAL_DRAIN_COMPLETION = "return_after_local_drain"
PACKAGE_FIELDS = frozenset(
    {
        "program",
        "target",
        "card_count",
        "tile_count",
        "resources",
        "modules",
        "entries",
    }
)
TARGET_FIELDS = frozenset(
    {"identity", "runtime_abi", "launch", "module_format"}
)

GRID_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "grid",
    "entry_abi": "tile-major-pointer-table",
    "phases": ["main"],
}

CLUSTER_KERNEL_LAUNCH: dict[str, Any] = {
    "kind": KERNEL_LAUNCH_KIND,
    "form": "cluster",
    "entry_abi": "tile-major-pointer-table",
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


@dataclasses.dataclass(frozen=True)
class SharedBoundaryResourceSpec:
    """One program-boundary resource shared by every Tile entry."""

    role: str
    role_index: int
    name: str
    type: Mapping[str, Any]
    bytes: int
    alignment: int
    access: str
    host_visible: bool


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
    if frozenset(manifest) != PACKAGE_FIELDS:
        raise RuntimeError(f"{context} package fields are not current")
    target = manifest.get("target")
    if (
        not isinstance(target, Mapping)
        or frozenset(target) != TARGET_FIELDS
        or target.get("launch") != expected
    ):
        raise RuntimeError(f"{context} runtime launch contract is invalid")


def require_complete_tile_domain(
    manifest: Mapping[str, Any], *, context: str
) -> tuple[Mapping[str, Any], ...]:
    """Validate and return entries in canonical Tile order."""

    entries = manifest.get("entries")
    if (
        manifest.get("card_count") != 1
        or manifest.get("tile_count") != TARGET_TILE_COUNT
        or not isinstance(entries, list)
        or len(entries) != TARGET_TILE_COUNT
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        raise RuntimeError(f"{context} package does not cover the complete Tile domain")
    entries_by_tile = {entry.get("tile_id"): entry for entry in entries}
    if set(entries_by_tile) != set(range(TARGET_TILE_COUNT)):
        raise RuntimeError(f"{context} package Tile IDs are not exact")
    if {entry.get("card_id") for entry in entries} != {0}:
        raise RuntimeError(f"{context} package has an invalid card domain")
    if {entry.get("launch_slot") for entry in entries} != set(
        range(TARGET_TILE_COUNT)
    ):
        raise RuntimeError(f"{context} launch-slot domain is not exact")
    if {
        entry.get("completion") for entry in entries
    } != {LOCAL_DRAIN_COMPLETION}:
        raise RuntimeError(f"{context} completion contract is invalid")
    return tuple(entries_by_tile[tile] for tile in range(TARGET_TILE_COUNT))


def require_board_completion(stdout: str, *, context: str) -> None:
    """Validate the current complete-Tile runtime completion report."""

    lines = stdout.splitlines()
    expected = [
        f"completion: {LOCAL_DRAIN_COMPLETION} tile_id={tile}"
        for tile in range(TARGET_TILE_COUNT)
    ]
    actual = [line for line in lines if line.startswith("completion:")]
    if actual != expected:
        raise RuntimeError(
            f"{context}: wafer-run completion report does not match the "
            "complete Tile domain"
        )
    for line in (
        f"invocation_tiles: {TARGET_TILE_COUNT}",
        f"physical_tile_domain: 0..{TARGET_TILE_COUNT - 1}",
        "board_execution: true",
    ):
        if lines.count(line) != 1:
            raise RuntimeError(
                f"{context}: wafer-run omitted or repeated {line!r}"
            )


def configure_direct_dte_tile_package(
    manifest: dict[str, Any],
    *,
    resources: tuple[SharedBoundaryResourceSpec, ...],
    status_abi: str,
    status_bytes: int,
    status_alignment: int,
    context: str,
) -> dict[tuple[int, str, int], int]:
    """Configure one shared module for Direct-DTE execution on all Tiles."""

    require_manifest_launch(manifest, GRID_KERNEL_LAUNCH, context=context)
    seed_entries = require_complete_tile_domain(manifest, context=context)
    modules = manifest.get("modules")
    if not isinstance(modules, list) or len(modules) != 1:
        raise RuntimeError(f"{context} requires one shared module")
    module = modules[0]
    if not isinstance(module, dict) or not isinstance(module.get("id"), int):
        raise RuntimeError(f"{context} module record is invalid")
    if not resources or len({(spec.role, spec.role_index) for spec in resources}) != len(
        resources
    ):
        raise RuntimeError(f"{context} resource roles are not unique")
    if any(
        spec.bytes <= 0
        or spec.alignment <= 0
        or spec.role_index < 0
        or spec.access not in {"read_only", "write_only", "read_write"}
        for spec in resources
    ):
        raise RuntimeError(f"{context} resource specification is invalid")

    manifest["target"]["launch"] = dict(CLUSTER_KERNEL_LAUNCH)
    module["exports"] = expected_kernel_module_exports(CLUSTER_KERNEL_LAUNCH)
    manifest_resources: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    bindings: dict[tuple[int, str, int], int] = {}
    next_resource_id = 0
    shared_resources: list[tuple[SharedBoundaryResourceSpec, int]] = []
    for spec in resources:
        resource_id = next_resource_id
        next_resource_id += 1
        shared_resources.append((spec, resource_id))
        manifest_resources.append(
            {
                "id": resource_id,
                "scope": {"kind": "card", "card_id": 0},
                "role": spec.role,
                "role_index": spec.role_index,
                "name": spec.name,
                "type": dict(spec.type),
                "bytes": spec.bytes,
                "alignment": spec.alignment,
                "access": spec.access,
                "host_visible": spec.host_visible,
            }
        )
    for tile_id in range(TARGET_TILE_COUNT):
        slots: list[dict[str, Any]] = []
        for spec, resource_id in shared_resources:
            bindings[(tile_id, spec.role, spec.role_index)] = resource_id
            slots.append(
                {
                    "ordinal": len(slots),
                    "resource": resource_id,
                    "access": spec.access,
                }
            )
        status_id = next_resource_id
        next_resource_id += 1
        bindings[(tile_id, "transport_status", 0)] = status_id
        manifest_resources.append(
            {
                "id": status_id,
                "scope": {"kind": "tile", "card_id": 0, "tile_id": tile_id},
                "role": "transport_status",
                "role_index": 0,
                "name": f"transport_status_tile_{tile_id}",
                "type": {"dtype": "u32", "shape": [1]},
                "bytes": status_bytes,
                "alignment": status_alignment,
                "access": "read_write",
                "host_visible": False,
            }
        )
        slots.append(
            {
                "ordinal": len(slots),
                "resource": status_id,
                "access": "read_write",
            }
        )
        entries.append(
            {
                "id": tile_id,
                "card_id": 0,
                "tile_id": tile_id,
                "launch_slot": tile_id,
                "module": module["id"],
                "slots": slots,
                "completion": LOCAL_DRAIN_COMPLETION,
                "transport": {
                    "kind": "direct_dte",
                    "status_resource": status_id,
                    "status_abi": status_abi,
                    "host_watchdog_required": True,
                },
            }
        )
    if len(seed_entries) != len(entries):
        raise RuntimeError(f"{context} Tile domain changed unexpectedly")
    manifest["resources"] = manifest_resources
    manifest["entries"] = entries
    return bindings
