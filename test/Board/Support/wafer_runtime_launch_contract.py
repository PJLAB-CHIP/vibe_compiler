#!/usr/bin/env python3
"""Canonical current runtime launch contracts used by board tests."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


KERNEL_LAUNCH_KIND = "kernel"
TARGET_TILE_COUNT = 16
LOCAL_DRAIN_COMPLETION = "return_after_local_drain"
PACKAGE_FIELDS = frozenset(
    {
        "program",
        "target",
        "launch",
        "card_count",
        "tile_count",
        "program_data",
        "program_tensors",
        "target_tensors",
        "inputs",
        "outputs",
        "modules",
        "entries",
    }
)
TARGET_FIELDS = frozenset({"identity", "runtime_abi", "module_format"})
LAUNCH_FIELDS = frozenset({"kind", "form", "entry_abi", "phases"})

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

KERNEL_PREPARE_EXPORT = {
    "role": "prepare",
    "symbol": "__wafer_kernel_prepare",
}
KERNEL_MAIN_EXPORT = {"role": "main", "symbol": "main"}


@dataclasses.dataclass(frozen=True)
class SharedBoundaryPortSpec:
    """One external program-boundary port shared by every Tile entry."""

    table: str  # "inputs" | "outputs"
    role_index: int
    logical_dtype: str
    logical_shape: list[int]
    dtype: str
    layout: str
    shape: list[int]
    bytes: int
    alignment: int
    access: str


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
    launch = manifest.get("launch")
    if (
        not isinstance(target, Mapping)
        or frozenset(target) != TARGET_FIELDS
        or not isinstance(launch, Mapping)
        or frozenset(launch) != LAUNCH_FIELDS
        or launch != expected
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
    ports: tuple[SharedBoundaryPortSpec, ...],
    status_abi: str,
    status_bytes: int,
    status_alignment: int,
    context: str,
) -> dict[tuple[str, str, int], int]:
    """Configure one shared module for Direct-DTE execution on all Tiles."""

    require_manifest_launch(manifest, GRID_KERNEL_LAUNCH, context=context)
    seed_entries = require_complete_tile_domain(manifest, context=context)
    modules = manifest.get("modules")
    if not isinstance(modules, list) or len(modules) != 1:
        raise RuntimeError(f"{context} requires one shared module")
    module = modules[0]
    if not isinstance(module, dict) or not isinstance(module.get("id"), int):
        raise RuntimeError(f"{context} module record is invalid")
    if not ports or len(
        {(spec.table, spec.role_index) for spec in ports}
    ) != len(ports):
        raise RuntimeError(f"{context} port roles are not unique")
    if any(
        spec.bytes <= 0
        or spec.alignment <= 0
        or spec.role_index < 0
        or spec.access not in {"read_only", "write_only", "read_write"}
        for spec in ports
    ):
        raise RuntimeError(f"{context} port specification is invalid")

    manifest["launch"] = dict(CLUSTER_KERNEL_LAUNCH)
    module["exports"] = expected_kernel_module_exports(CLUSTER_KERNEL_LAUNCH)
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    next_port_id = {"inputs": 0, "outputs": 0}
    entries: list[dict[str, Any]] = []
    bindings: dict[tuple[str, str, int], int] = {}
    for spec in ports:
        port_id = next_port_id[spec.table]
        next_port_id[spec.table] += 1
        bindings[(spec.table, spec.role_index)] = port_id
        record = {
            "id": port_id,
            "role_index": spec.role_index,
            "logical_dtype": spec.logical_dtype,
            "logical_shape": list(spec.logical_shape),
            "dtype": spec.dtype,
            "layout": spec.layout,
            "shape": list(spec.shape),
            "bytes": spec.bytes,
            "alignment": spec.alignment,
        }
        (inputs if spec.table == "inputs" else outputs).append(record)
    for tile_id in range(TARGET_TILE_COUNT):
        arguments: list[dict[str, Any]] = []
        for spec in ports:
            kind = "external_input" if spec.table == "inputs" else "external_output"
            arguments.append(
                {
                    "ordinal": len(arguments),
                    "kind": kind,
                    "port": bindings[(spec.table, spec.role_index)],
                    "access": spec.access,
                }
            )
        arguments.append(
            {
                "ordinal": len(arguments),
                "kind": "transport_status",
                "status_abi": status_abi,
                "bytes": status_bytes,
                "alignment": status_alignment,
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
                "arguments": arguments,
                "completion": LOCAL_DRAIN_COMPLETION,
                "transport": {
                    "kind": "direct_dte",
                    "status_abi": status_abi,
                    "host_watchdog_required": True,
                },
            }
        )
    if len(seed_entries) != len(entries):
        raise RuntimeError(f"{context} Tile domain changed unexpectedly")
    manifest["inputs"] = inputs
    manifest["outputs"] = outputs
    manifest["entries"] = entries
    return bindings
