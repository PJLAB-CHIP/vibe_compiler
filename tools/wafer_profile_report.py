#!/usr/bin/env python3
"""Validate 16-tile profiler evidence and build a self-contained report.

The input is the versioned ``wafer.profile.evidence`` JSON contract.  Timing
samples and the explanatory TSM trace intentionally remain separate:

* uninstrumented execution-package host samples decide the performance
  verdict;
* TSM records describe wrapper call/return issue points, never engine
  completion;
* PMU counters are aggregate correlation evidence and are never apportioned to
  individual TSM records.

Only Python's standard library is used so a profile companion can be analyzed
offline in the same environment that runs ``wafer-run``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import pathlib
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any


SCHEMA_NAME = "wafer.profile.evidence"
SCHEMA_VERSION = 2
ANALYSIS_SCHEMA_NAME = "wafer.profile.analysis"
ANALYSIS_SCHEMA_VERSION = 1
TILES = tuple(range(16))
ENGINES = ("CT", "NE", "RDMA", "WDMA", "TDMA")
WORKERS = (0, 1, 2)
AGGREGATE_COUNTERS = (
    "statistics_window",
    "fu",
    "ct",
    "ne",
    "rdma",
    "wdma",
    "tdma",
    "scalar",
)
UINT64_MAX = (1 << 64) - 1
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0x57414645
TRACE_STATE_COMPLETE = 2
TRACE_STATE_OVERFLOW = 3
TRACE_FLAG_ENABLED = 1 << 0
TRACE_FLAG_ENTRY_BEGUN = 1 << 1
TRACE_FLAG_ENTRY_ENDED = 1 << 2
TRACE_FLAG_OVERFLOW = 1 << 3
TRACE_FLAG_SITE_PROTOCOL_ERROR = 1 << 4
TRACE_FLAG_SUB_INDEX_OVERFLOW = 1 << 5
TRACE_FLAG_PUBLISHED = 1 << 6
TRACE_FLAG_COUNT_ONLY = 1 << 7
TRACE_EXPECTED_COMPLETE_FLAGS = (
    TRACE_FLAG_ENABLED
    | TRACE_FLAG_ENTRY_BEGUN
    | TRACE_FLAG_ENTRY_ENDED
    | TRACE_FLAG_PUBLISHED
)
TRACE_KNOWN_FLAGS = (
    TRACE_EXPECTED_COMPLETE_FLAGS
    | TRACE_FLAG_OVERFLOW
    | TRACE_FLAG_SITE_PROTOCOL_ERROR
    | TRACE_FLAG_SUB_INDEX_OVERFLOW
    | TRACE_FLAG_COUNT_ONLY
)
MAX_COMPLETION_RESOLUTION_FRACTION = 0.0025


class EvidenceError(ValueError):
    """The evidence does not conform to the v2 structural contract."""


def _fail(path: str, message: str) -> None:
    raise EvidenceError(f"{path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(path, "expected an array")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing keys {missing}")
    if unknown:
        details.append(f"unknown keys {unknown}")
    _fail(path, "; ".join(details))


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "expected a non-empty string")
    return value


def _integer(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "expected an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be at most {maximum}")
    return value


def _number(value: object, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _fail(path, "expected a finite number")
    return float(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


def _nullable_boolean(value: object, path: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path)


def _tile_rows(
    value: object, path: str
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        _mapping(row, f"{path}[{index}]")
        for index, row in enumerate(_sequence(value, path))
    )
    tile_ids: list[int] = []
    for index, row in enumerate(rows):
        tile_ids.append(_integer(row.get("tile"), f"{path}[{index}].tile"))
    counts = Counter(tile_ids)
    duplicates = sorted(tile for tile, count in counts.items() if count != 1)
    missing = sorted(set(TILES) - set(tile_ids))
    extra = sorted(set(tile_ids) - set(TILES))
    if len(rows) != len(TILES) or duplicates or missing or extra:
        _fail(
            path,
            "must contain all-and-only tiles 0..15 exactly once "
            f"(duplicates={duplicates}, missing={missing}, extra={extra})",
        )
    return rows


def _validate_runtime_launch(
    value: object, path: str
) -> Mapping[str, Any]:
    launch = _mapping(value, path)
    kind = _string(launch.get("kind"), f"{path}.kind")
    if kind == "kernel":
        _exact_keys(
            launch,
            {"kind", "form", "entry_abi", "phases"},
            path,
        )
        form = _string(launch["form"], f"{path}.form")
        entry_abi = _string(launch["entry_abi"], f"{path}.entry_abi")
        phases = tuple(
            _string(phase, f"{path}.phases[{index}]")
            for index, phase in enumerate(
                _sequence(launch["phases"], f"{path}.phases")
            )
        )
        canonical = {
            "per-rank": ("rank-local-pointer-block-v1", ("main",)),
            "grid": ("rank-major-pointer-table-v1", ("main",)),
            "cluster": (
                "rank-major-pointer-table-v1",
                ("prepare", "main"),
            ),
        }
        expected = canonical.get(form)
        if expected is None:
            _fail(f"{path}.form", "unknown kernel launch form")
        if (entry_abi, phases) != expected:
            _fail(
                path,
                "kernel form, entry ABI and ordered phases are incompatible",
            )
        return launch
    if kind == "model":
        _exact_keys(launch, {"kind", "entry_abi", "phases"}, path)
        entry_abi = _string(launch["entry_abi"], f"{path}.entry_abi")
        phases = tuple(
            _string(phase, f"{path}.phases[{index}]")
            for index, phase in enumerate(
                _sequence(launch["phases"], f"{path}.phases")
            )
        )
        if (
            entry_abi != "tx81-model-bootparam-v1"
            or phases != ("main",)
        ):
            _fail(
                path,
                "model entry ABI and ordered phases are incompatible",
            )
        return launch
    _fail(f"{path}.kind", "must be either 'kernel' or 'model'")


def _validate_identity(evidence: Mapping[str, Any]) -> None:
    identity = _mapping(evidence["identity"], "identity")
    _exact_keys(
        identity,
        {
            "production_manifest_sha256",
            "profile_companion_schema_version",
            "target_profile",
            "launch",
            "execution_ranks",
            "baseline_same_as_winner",
            "site_correlation_basis",
        },
        "identity",
    )
    _string(
        identity["production_manifest_sha256"],
        "identity.production_manifest_sha256",
    )
    version = _integer(
        identity["profile_companion_schema_version"],
        "identity.profile_companion_schema_version",
        minimum=1,
    )
    if version != 1:
        _fail(
            "identity.profile_companion_schema_version",
            "profiler v2 requires companion schema version 1",
        )
    _string(identity["target_profile"], "identity.target_profile")
    _validate_runtime_launch(identity["launch"], "identity.launch")
    ranks = _integer(
        identity["execution_ranks"],
        "identity.execution_ranks",
        minimum=1,
    )
    if ranks != len(TILES):
        _fail("identity.execution_ranks", "profiler v2 requires exactly 16 ranks")
    _boolean(
        identity["baseline_same_as_winner"],
        "identity.baseline_same_as_winner",
    )
    correlation_basis = _string(
        identity["site_correlation_basis"],
        "identity.site_correlation_basis",
    )
    if correlation_basis != "heuristic-target-call-signature-occurrence-v1":
        _fail(
            "identity.site_correlation_basis",
            "profiler v2 requires the declared heuristic correlation basis",
        )


def _validate_topology(evidence: Mapping[str, Any]) -> None:
    rows = _tile_rows(evidence["topology"], "topology")
    coordinates: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        path = f"topology[{index}]"
        _exact_keys(row, {"tile", "x", "y"}, path)
        x = _integer(row["x"], f"{path}.x", minimum=0)
        y = _integer(row["y"], f"{path}.y", minimum=0)
        if (x, y) in coordinates:
            _fail(path, f"duplicate physical coordinate ({x}, {y})")
        coordinates.add((x, y))


def _validate_measurement(evidence: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    measurement = _mapping(evidence["measurement"], "measurement")
    _exact_keys(measurement, {"blocks", "samples"}, "measurement")
    blocks = tuple(
        _mapping(row, f"measurement.blocks[{index}]")
        for index, row in enumerate(
            _sequence(measurement["blocks"], "measurement.blocks")
        )
    )
    if len(blocks) != 5:
        _fail("measurement.blocks", "v1 requires exactly five balanced blocks")
    by_id: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(blocks):
        path = f"measurement.blocks[{index}]"
        _exact_keys(row, {"block", "held_out", "order"}, path)
        block = _integer(row["block"], f"{path}.block", minimum=0)
        if block in by_id:
            _fail(f"{path}.block", f"duplicate block {block}")
        _boolean(row["held_out"], f"{path}.held_out")
        order = _string(row["order"], f"{path}.order")
        if order not in {"ABBA", "BAAB"}:
            _fail(f"{path}.order", "must be ABBA or BAAB")
        by_id[block] = row

    ordered = [by_id[key] for key in sorted(by_id)]
    held_out = [bool(row["held_out"]) for row in ordered]
    if held_out != [False, False, False, False, True]:
        _fail(
            "measurement.blocks",
            "the first four ordered blocks must calibrate and the fifth "
            "must be held out",
        )
    orders = [str(row["order"]) for row in ordered]
    if any(orders[index] == orders[index - 1] for index in range(1, 5)):
        _fail("measurement.blocks", "ABBA and BAAB orders must alternate")

    samples = tuple(
        _mapping(row, f"measurement.samples[{index}]")
        for index, row in enumerate(
            _sequence(measurement["samples"], "measurement.samples")
        )
    )
    if len(samples) != 20:
        _fail(
            "measurement.samples",
            "v1 requires exactly twenty uninstrumented execution samples",
        )
    sample_ids: set[str] = set()
    positions_by_candidate_block: dict[
        tuple[str, int], set[int]
    ] = defaultdict(set)
    for index, row in enumerate(samples):
        path = f"measurement.samples[{index}]"
        _exact_keys(
            row,
            {
                "sample_id",
                "candidate",
                "block",
                "position",
                "host_elapsed_ns",
                "completion_observation_resolution_ns",
            },
            path,
        )
        sample_id = _string(row["sample_id"], f"{path}.sample_id")
        if sample_id in sample_ids:
            _fail(f"{path}.sample_id", f"duplicate sample {sample_id!r}")
        sample_ids.add(sample_id)
        candidate = _string(row["candidate"], f"{path}.candidate")
        if candidate not in {"baseline", "winner"}:
            _fail(f"{path}.candidate", "must be baseline or winner")
        block = _integer(row["block"], f"{path}.block", minimum=0)
        if block not in by_id:
            _fail(f"{path}.block", f"unknown block {block}")
        position = _integer(
            row["position"], f"{path}.position", minimum=0, maximum=3
        )
        key = (candidate, block)
        if position in positions_by_candidate_block[key]:
            _fail(
                f"{path}.position",
                f"duplicate {candidate} position {position} in block {block}",
            )
        positions_by_candidate_block[key].add(position)
        _integer(
            row["host_elapsed_ns"],
            f"{path}.host_elapsed_ns",
            minimum=1,
            maximum=UINT64_MAX,
        )
        _integer(
            row["completion_observation_resolution_ns"],
            f"{path}.completion_observation_resolution_ns",
            minimum=1,
            maximum=UINT64_MAX,
        )

    for block, block_row in by_id.items():
        order = str(block_row["order"])
        for candidate, letter in (("baseline", "A"), ("winner", "B")):
            expected = {
                index
                for index, order_letter in enumerate(order)
                if order_letter == letter
            }
            actual = positions_by_candidate_block[(candidate, block)]
            if actual != expected:
                _fail(
                    "measurement.samples",
                    f"block {block} {candidate} positions {sorted(actual)} "
                    f"do not match {order}",
                )
    return by_id


def _validate_output_validation(evidence: Mapping[str, Any]) -> None:
    output = _mapping(evidence["output_validation"], "output_validation")
    _exact_keys(output, {"mode", "resources"}, "output_validation")
    mode = _string(output["mode"], "output_validation.mode")
    allowed_modes = {
        "external-exact",
        "mixed",
        "same-session-production-winner",
    }
    if mode not in allowed_modes:
        _fail(
            "output_validation.mode",
            f"must be one of {sorted(allowed_modes)}",
        )
    rows = tuple(
        _mapping(row, f"output_validation.resources[{index}]")
        for index, row in enumerate(
            _sequence(
                output["resources"], "output_validation.resources"
            )
        )
    )
    if not rows:
        _fail(
            "output_validation.resources",
            "must describe at least one writable semantic resource",
        )
    keys: set[tuple[int, str, int]] = set()
    external_states: list[bool | None] = []
    for index, row in enumerate(rows):
        path = f"output_validation.resources[{index}]"
        _exact_keys(
            row,
            {
                "logical_rank",
                "role",
                "role_index",
                "bytes",
                "reference_sha256",
                "external_expected_exact",
                "production_winner_repeat_exact",
                "candidate_equivalent_exact",
            },
            path,
        )
        logical_rank = _integer(
            row["logical_rank"], f"{path}.logical_rank", minimum=-1
        )
        role = _string(row["role"], f"{path}.role")
        role_index = _integer(
            row["role_index"], f"{path}.role_index", minimum=0
        )
        key = (logical_rank, role, role_index)
        if key in keys:
            _fail(path, f"duplicate writable semantic key {key}")
        keys.add(key)
        _integer(
            row["bytes"],
            f"{path}.bytes",
            minimum=1,
            maximum=UINT64_MAX,
        )
        digest = _string(row["reference_sha256"], f"{path}.reference_sha256")
        if (
            not digest.startswith("sha256:")
            or len(digest) != len("sha256:") + 64
            or any(
                character not in "0123456789abcdef"
                for character in digest[len("sha256:") :]
            )
        ):
            _fail(f"{path}.reference_sha256", "must be canonical SHA-256")
        external_states.append(
            _nullable_boolean(
                row["external_expected_exact"],
                f"{path}.external_expected_exact",
            )
        )
        _boolean(
            row["production_winner_repeat_exact"],
            f"{path}.production_winner_repeat_exact",
        )
        _boolean(
            row["candidate_equivalent_exact"],
            f"{path}.candidate_equivalent_exact",
        )
    expected_mode = (
        "external-exact"
        if all(state is not None for state in external_states)
        else "same-session-production-winner"
        if all(state is None for state in external_states)
        else "mixed"
    )
    if mode != expected_mode:
        _fail(
            "output_validation.mode",
            f"{mode!r} conflicts with resource expected coverage; "
            f"expected {expected_mode!r}",
        )


def _validate_sites(
    evidence: Mapping[str, Any],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    rows = tuple(
        _mapping(row, f"sites[{index}]")
        for index, row in enumerate(_sequence(evidence["sites"], "sites"))
    )
    by_id: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    correlation_keys: set[tuple[str, int, str]] = set()
    for index, row in enumerate(rows):
        path = f"sites[{index}]"
        required = {
            "candidate",
            "tile",
            "site_id",
            "correlation_key",
            "engine",
            "target_call_ordinal",
            "target_call_symbol",
        }
        allowed = required | {"position"}
        missing = sorted(required - set(row))
        unknown = sorted(set(row) - allowed)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing keys {missing}")
            if unknown:
                details.append(f"unknown keys {unknown}")
            _fail(path, "; ".join(details))
        candidate = _string(row["candidate"], f"{path}.candidate")
        if candidate not in {"baseline", "winner"}:
            _fail(
                f"{path}.candidate", "must be baseline or winner"
            )
        tile = _integer(row["tile"], f"{path}.tile", minimum=0, maximum=15)
        site_id = _integer(row["site_id"], f"{path}.site_id", minimum=0)
        site_key = (candidate, tile, site_id)
        if site_key in by_id:
            _fail(
                f"{path}.site_id",
                f"duplicate {candidate} tile {tile} site {site_id}",
            )
        correlation_key = _string(
            row["correlation_key"], f"{path}.correlation_key"
        )
        candidate_correlation = (candidate, tile, correlation_key)
        if candidate_correlation in correlation_keys:
            _fail(
                f"{path}.correlation_key",
                f"duplicate {candidate} tile {tile} correlation key "
                f"{correlation_key!r}",
            )
        correlation_keys.add(candidate_correlation)
        engine = _string(row["engine"], f"{path}.engine")
        if engine not in ENGINES:
            _fail(f"{path}.engine", f"must be one of {list(ENGINES)}")
        _integer(
            row["target_call_ordinal"],
            f"{path}.target_call_ordinal",
            minimum=0,
        )
        _string(row["target_call_symbol"], f"{path}.target_call_symbol")
        if "position" in row:
            _string(row["position"], f"{path}.position")
        by_id[site_key] = row
    return by_id


def _validate_clock(
    candidate: Mapping[str, Any], candidate_name: str
) -> dict[int, Mapping[str, Any]]:
    rows = _tile_rows(candidate["clock"], f"experiments.{candidate_name}.clock")
    by_tile: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        path = f"experiments.{candidate_name}.clock[{index}]"
        _exact_keys(
            row,
            {
                "tile",
                "slope",
                "offset",
                "uncertainty",
                "round_trips",
                "valid",
                "monotonic",
            },
            path,
        )
        tile = _integer(row["tile"], f"{path}.tile")
        slope = _number(row["slope"], f"{path}.slope")
        if slope <= 0:
            _fail(f"{path}.slope", "must be positive")
        _number(row["offset"], f"{path}.offset")
        uncertainty = _number(row["uncertainty"], f"{path}.uncertainty")
        if uncertainty < 0:
            _fail(f"{path}.uncertainty", "must not be negative")
        _integer(row["round_trips"], f"{path}.round_trips", minimum=0)
        _boolean(row["valid"], f"{path}.valid")
        _boolean(row["monotonic"], f"{path}.monotonic")
        by_tile[tile] = row
    return by_tile


def _validate_summary(
    candidate: Mapping[str, Any],
    candidate_name: str,
    clock: Mapping[int, Mapping[str, Any]],
) -> None:
    path = f"experiments.{candidate_name}.summary"
    summary = _mapping(candidate["summary"], path)
    _exact_keys(summary, {"tiles"}, path)
    timings = _tile_rows(summary["tiles"], f"{path}.tiles")
    for timing_index, timing in enumerate(timings):
        timing_path = f"{path}.tiles[{timing_index}]"
        _exact_keys(
            timing, {"tile", "entry_begin", "entry_end"}, timing_path
        )
        tile = _integer(timing["tile"], f"{timing_path}.tile")
        begin = _integer(
            timing["entry_begin"],
            f"{timing_path}.entry_begin",
            minimum=0,
            maximum=UINT64_MAX,
        )
        end = _integer(
            timing["entry_end"],
            f"{timing_path}.entry_end",
            minimum=0,
            maximum=UINT64_MAX,
        )
        if end < begin:
            _fail(timing_path, "entry_end precedes entry_begin")
        mapping = clock[tile]
        # Parse through decimal strings so u64 device cycles are never first
        # rounded through an IEEE-754 float.  The mapping itself is qualified
        # separately; this check only proves that the declared finite values
        # can be evaluated.
        Decimal(str(mapping["slope"])) * Decimal(begin) + Decimal(
            str(mapping["offset"])
        )
        Decimal(str(mapping["slope"])) * Decimal(end) + Decimal(
            str(mapping["offset"])
        )


def _validate_trace(
    candidate: Mapping[str, Any],
    candidate_name: str,
    sites: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> None:
    trace_path = f"experiments.{candidate_name}.trace"
    trace = _mapping(candidate["trace"], trace_path)
    _exact_keys(trace, {"complete", "tiles"}, trace_path)
    _boolean(trace["complete"], f"{trace_path}.complete")
    rows = _tile_rows(trace["tiles"], f"{trace_path}.tiles")
    for index, row in enumerate(rows):
        path = f"{trace_path}.tiles[{index}]"
        _exact_keys(
            row,
            {
                "tile",
                "entry_begin_cycle",
                "entry_end_cycle",
                "capacity",
                "count",
                "preflight_count",
                "next_sequence",
                "dropped_event_count",
                "record_flags",
                "trace_state",
                "overflow",
                "events",
            },
            path,
        )
        _integer(row["tile"], f"{path}.tile")
        entry_begin = _integer(
            row["entry_begin_cycle"],
            f"{path}.entry_begin_cycle",
            minimum=0,
            maximum=UINT64_MAX,
        )
        entry_end = _integer(
            row["entry_end_cycle"],
            f"{path}.entry_end_cycle",
            minimum=0,
            maximum=UINT64_MAX,
        )
        if entry_end < entry_begin:
            _fail(path, "entry_end_cycle precedes entry_begin_cycle")
        capacity = _integer(row["capacity"], f"{path}.capacity", minimum=0)
        count = _integer(row["count"], f"{path}.count", minimum=0)
        preflight_count = _integer(
            row["preflight_count"],
            f"{path}.preflight_count",
            minimum=0,
            maximum=UINT64_MAX,
        )
        next_sequence = _integer(
            row["next_sequence"],
            f"{path}.next_sequence",
            minimum=0,
            maximum=UINT64_MAX,
        )
        dropped = _integer(
            row["dropped_event_count"],
            f"{path}.dropped_event_count",
            minimum=0,
            maximum=(1 << 32) - 1,
        )
        record_flags = _integer(
            row["record_flags"],
            f"{path}.record_flags",
            minimum=0,
            maximum=(1 << 32) - 1,
        )
        if record_flags & ~TRACE_KNOWN_FLAGS:
            _fail(
                f"{path}.record_flags",
                "contains flags unknown to profiler evidence v2",
            )
        trace_state = _integer(
            row["trace_state"],
            f"{path}.trace_state",
            minimum=0,
            maximum=4,
        )
        overflow = _boolean(row["overflow"], f"{path}.overflow")
        events = tuple(
            _mapping(event, f"{path}.events[{event_index}]")
            for event_index, event in enumerate(
                _sequence(row["events"], f"{path}.events")
            )
        )
        if count != len(events):
            _fail(
                f"{path}.count",
                f"declares {count} events but carries {len(events)}",
            )
        if count > capacity:
            _fail(path, "event count exceeds trace capacity")
        if next_sequence != count + dropped:
            _fail(
                path,
                "next_sequence must equal event count plus dropped count",
            )
        raw_overflow = bool(
            dropped
            or record_flags & TRACE_FLAG_OVERFLOW
            or trace_state == TRACE_STATE_OVERFLOW
        )
        if overflow != raw_overflow:
            _fail(path, "overflow summary conflicts with raw terminal fields")
        sequences: list[int] = []
        for event_index, event in enumerate(events):
            event_path = f"{path}.events[{event_index}]"
            _exact_keys(
                event,
                {
                    "sequence",
                    "site_id",
                    "sub_index",
                    "engine",
                    "begin_cycle",
                    "return_cycle",
                    "raw_result",
                    "valid",
                },
                event_path,
            )
            sequence = _integer(
                event["sequence"], f"{event_path}.sequence", minimum=0
            )
            sequences.append(sequence)
            site_id = _integer(
                event["site_id"], f"{event_path}.site_id", minimum=0
            )
            site_key = (candidate_name, int(row["tile"]), site_id)
            if site_key not in sites:
                _fail(
                    f"{event_path}.site_id",
                    f"unmapped {candidate_name} tile {row['tile']} "
                    f"site {site_id}",
                )
            _integer(
                event["sub_index"], f"{event_path}.sub_index", minimum=0
            )
            engine = _string(event["engine"], f"{event_path}.engine")
            if engine not in ENGINES:
                _fail(
                    f"{event_path}.engine",
                    f"must be one of {list(ENGINES)}",
                )
            if engine != sites[site_key]["engine"]:
                _fail(
                    f"{event_path}.engine",
                    f"{engine} conflicts with site {site_id} engine "
                    f"{sites[site_key]['engine']}",
                )
            begin = _integer(
                event["begin_cycle"],
                f"{event_path}.begin_cycle",
                minimum=0,
                maximum=UINT64_MAX,
            )
            end = _integer(
                event["return_cycle"],
                f"{event_path}.return_cycle",
                minimum=0,
                maximum=UINT64_MAX,
            )
            if end < begin:
                _fail(event_path, "return_cycle precedes begin_cycle")
            if begin < entry_begin or end > entry_end:
                _fail(
                    event_path,
                    "TSM call interval falls outside the entry interval",
                )
            _integer(
                event["raw_result"],
                f"{event_path}.raw_result",
                minimum=0,
                maximum=UINT64_MAX,
            )
            _boolean(event["valid"], f"{event_path}.valid")
        if sequences != list(range(len(events))):
            _fail(
                f"{path}.events",
                "sequence must be contiguous, unique, and ordered from zero",
            )
        # The count preflight is independent evidence.  A mismatch is
        # structurally representable so the report can explain why trace
        # comparison was withheld; it is never silently repaired here.
        del preflight_count


def _validate_counter_snapshot(
    value: object, path: str, *, maximum: int = UINT64_MAX
) -> None:
    snapshot = _mapping(value, path)
    required = {"start", "end", "stable", "enabled"}
    allowed = required | {"recovery"}
    missing = sorted(required - set(snapshot))
    unknown = sorted(set(snapshot) - allowed)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        _fail(path, "; ".join(details))
    for field in ("start", "end"):
        _integer(
            snapshot[field],
            f"{path}.{field}",
            minimum=0,
            maximum=maximum,
        )
    if "recovery" in snapshot:
        _integer(
            snapshot["recovery"],
            f"{path}.recovery",
            minimum=0,
            maximum=maximum,
        )
    _boolean(snapshot["stable"], f"{path}.stable")
    _boolean(snapshot["enabled"], f"{path}.enabled")


def _validate_pmu(candidate: Mapping[str, Any], candidate_name: str) -> None:
    pmu_path = f"experiments.{candidate_name}.pmu"
    pmu = _mapping(candidate["pmu"], pmu_path)
    _exact_keys(pmu, {"tiles"}, pmu_path)
    rows = _tile_rows(pmu["tiles"], f"{pmu_path}.tiles")
    for tile_index, row in enumerate(rows):
        path = f"{pmu_path}.tiles[{tile_index}]"
        _exact_keys(row, {"tile", "aggregates", "workers"}, path)
        _integer(row["tile"], f"{path}.tile")
        aggregates = _mapping(row["aggregates"], f"{path}.aggregates")
        _exact_keys(
            aggregates, set(AGGREGATE_COUNTERS), f"{path}.aggregates"
        )
        for counter in AGGREGATE_COUNTERS:
            _validate_counter_snapshot(
                aggregates[counter], f"{path}.aggregates.{counter}"
            )

        workers = tuple(
            _mapping(worker, f"{path}.workers[{index}]")
            for index, worker in enumerate(
                _sequence(row["workers"], f"{path}.workers")
            )
        )
        worker_ids: list[int] = []
        for worker_index, worker in enumerate(workers):
            worker_path = f"{path}.workers[{worker_index}]"
            _exact_keys(worker, {"worker", "engines"}, worker_path)
            worker_id = _integer(
                worker["worker"], f"{worker_path}.worker"
            )
            worker_ids.append(worker_id)
            engines = tuple(
                _mapping(engine, f"{worker_path}.engines[{index}]")
                for index, engine in enumerate(
                    _sequence(
                        worker["engines"], f"{worker_path}.engines"
                    )
                )
            )
            engine_names: list[str] = []
            for engine_index, engine in enumerate(engines):
                engine_path = f"{worker_path}.engines[{engine_index}]"
                _exact_keys(
                    engine,
                    {"engine", "instructions", "blocking"},
                    engine_path,
                )
                engine_name = _string(
                    engine["engine"], f"{engine_path}.engine"
                )
                engine_names.append(engine_name)
                for metric in ("instructions", "blocking"):
                    _validate_counter_snapshot(
                        engine[metric],
                        f"{engine_path}.{metric}",
                        maximum=(1 << 32) - 1,
                    )
            if set(engine_names) != set(ENGINES) or len(
                engine_names
            ) != len(ENGINES):
                _fail(
                    f"{worker_path}.engines",
                    f"must contain all-and-only engines {list(ENGINES)}",
                )
        if sorted(worker_ids) != list(WORKERS) or len(set(worker_ids)) != 3:
            _fail(
                f"{path}.workers",
                "must contain all-and-only workers 0, 1, and 2",
            )


def _validate_candidate(
    evidence: Mapping[str, Any],
    candidate_name: str,
    sites: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> None:
    candidate_path = f"experiments.{candidate_name}"
    candidate = _mapping(
        _mapping(evidence["experiments"], "experiments")[candidate_name],
        candidate_path,
    )
    _exact_keys(
        candidate,
        {
            "artifact",
            "clock",
            "summary",
            "trace",
            "pmu",
        },
        candidate_path,
    )
    artifact = _mapping(candidate["artifact"], f"{candidate_path}.artifact")
    _exact_keys(
        artifact,
        {
            "digest",
            "target_profile",
            "launch",
            "execution_ranks",
        },
        f"{candidate_path}.artifact",
    )
    for field in ("digest", "target_profile"):
        _string(artifact[field], f"{candidate_path}.artifact.{field}")
    _validate_runtime_launch(
        artifact["launch"], f"{candidate_path}.artifact.launch"
    )
    ranks = _integer(
        artifact["execution_ranks"],
        f"{candidate_path}.artifact.execution_ranks",
        minimum=1,
    )
    if ranks != len(TILES):
        _fail(
            f"{candidate_path}.artifact.execution_ranks",
            "profiler v2 requires exactly 16 ranks",
        )
    clock = _validate_clock(candidate, candidate_name)
    _validate_summary(candidate, candidate_name, clock)
    _validate_trace(candidate, candidate_name, sites)
    _validate_pmu(candidate, candidate_name)


def validate_evidence(value: object) -> Mapping[str, Any]:
    """Return structurally valid v2 evidence or raise :class:`EvidenceError`."""

    evidence = _mapping(value, "evidence")
    _exact_keys(
        evidence,
        {
            "schema",
            "schema_version",
            "run_id",
            "identity",
            "topology",
            "measurement",
            "output_validation",
            "sites",
            "validity",
            "experiments",
        },
        "evidence",
    )
    if evidence["schema"] != SCHEMA_NAME:
        _fail("schema", f"expected {SCHEMA_NAME!r}")
    if evidence["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"expected version {SCHEMA_VERSION}")
    _string(evidence["run_id"], "run_id")
    _validate_identity(evidence)
    _validate_topology(evidence)
    _validate_measurement(evidence)
    _validate_output_validation(evidence)
    sites = _validate_sites(evidence)
    validity = _mapping(evidence["validity"], "validity")
    _exact_keys(
        validity,
        {"environment", "package_companion", "measurement_basis"},
        "validity",
    )
    for key in ("environment", "package_companion", "measurement_basis"):
        _boolean(validity[key], f"validity.{key}")
    experiments = _mapping(evidence["experiments"], "experiments")
    _exact_keys(experiments, {"baseline", "winner"}, "experiments")
    for candidate in ("baseline", "winner"):
        _validate_candidate(evidence, candidate, sites)
    return evidence


def load_evidence(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    """Load and validate a profile evidence JSON file."""

    source = pathlib.Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvidenceError(f"{source}: cannot read evidence: {error}") from error
    except json.JSONDecodeError as error:
        raise EvidenceError(
            f"{source}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}"
        ) from error
    return validate_evidence(value)


def _by_tile(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["tile"]): row for row in rows}


def _clock_map(candidate: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return _by_tile(candidate["clock"])


def _mapped_decimal(local: int, mapping: Mapping[str, Any]) -> Decimal:
    """Map one exact device-cycle value without first rounding its u64."""

    return Decimal(str(mapping["slope"])) * Decimal(local) + Decimal(
        str(mapping["offset"])
    )


def _summary_metrics(
    summary: Mapping[str, Any], clock: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    begins: dict[int, Decimal] = {}
    ends: dict[int, Decimal] = {}
    durations: dict[int, int] = {}
    for row in summary["tiles"]:
        tile = int(row["tile"])
        entry_begin = int(row["entry_begin"])
        entry_end = int(row["entry_end"])
        begins[tile] = _mapped_decimal(entry_begin, clock[tile])
        ends[tile] = _mapped_decimal(entry_end, clock[tile])
        # Duration is rank-local.  It remains meaningful without a qualified
        # cross-rank offset and therefore uses the raw local cycle interval.
        durations[tile] = entry_end - entry_begin
    begin_tile = min(begins, key=begins.__getitem__)
    end_tile = max(ends, key=ends.__getitem__)
    uncertainty = float(clock[begin_tile]["uncertainty"]) + float(
        clock[end_tile]["uncertainty"]
    )
    return {
        "cluster_span": float(ends[end_tile] - begins[begin_tile]),
        "uncertainty": uncertainty,
        "entry_skew": float(max(begins.values()) - min(begins.values())),
        "finish_skew": float(max(ends.values()) - min(ends.values())),
        "durations": durations,
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _paired_bootstrap_ci(
    baseline_blocks: Sequence[Sequence[float]],
    winner_blocks: Sequence[Sequence[float]],
) -> tuple[float, float]:
    if (
        len(baseline_blocks) != len(winner_blocks)
        or not baseline_blocks
        or any(not block for block in baseline_blocks)
        or any(not block for block in winner_blocks)
    ):
        raise ValueError("paired bootstrap requires non-empty equal-length rows")
    if any(value <= 0 for block in baseline_blocks for value in block) or any(
        value <= 0 for block in winner_blocks for value in block
    ):
        raise ValueError("launch-to-completion latencies must be positive")
    rng = random.Random(BOOTSTRAP_SEED)
    ratios: list[float] = []
    count = len(baseline_blocks)
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(count) for _ in range(count)]
        baseline = [
            value
            for index in indices
            for value in baseline_blocks[index]
        ]
        winner = [
            value for index in indices for value in winner_blocks[index]
        ]
        baseline_median = _median(baseline)
        winner_median = _median(winner)
        ratios.append(baseline_median / winner_median)
    ratios.sort()
    return _percentile(ratios, 0.025), _percentile(ratios, 0.975)


def _counter_status(snapshot: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not snapshot["enabled"]:
        return False, "counter was not enabled"
    if not snapshot["stable"]:
        return False, "high-low-high read was not stable"
    start = int(snapshot["start"])
    end = int(snapshot["end"])
    if end < start:
        return False, "counter moved backwards or wrapped without proof"
    if "recovery" not in snapshot:
        return False, "terminal recovery snapshot is missing"
    recovery = int(snapshot["recovery"])
    if recovery != end:
        return False, "counter changed after terminal recovery snapshot"
    return True, None


def _identity_validity(
    evidence: Mapping[str, Any], diagnostics: list[dict[str, Any]]
) -> bool:
    identity = evidence["identity"]
    valid = True
    baseline_digest = str(
        evidence["experiments"]["baseline"]["artifact"]["digest"]
    )
    winner_digest = str(
        evidence["experiments"]["winner"]["artifact"]["digest"]
    )
    if identity["baseline_same_as_winner"] and baseline_digest != winner_digest:
        valid = False
        diagnostics.append(
            {
                "severity": "error",
                "code": "identity_mismatch",
                "candidate": "baseline",
                "message": (
                    "companion declares reserved baseline same_as winner, "
                    "but their execution-package digests differ"
                ),
            }
        )
    for candidate_name in ("baseline", "winner"):
        artifact = evidence["experiments"][candidate_name]["artifact"]
        for field in (
            "target_profile",
            "launch",
            "execution_ranks",
        ):
            if artifact[field] != identity[field]:
                valid = False
                diagnostics.append(
                    {
                        "severity": "error",
                        "code": "identity_mismatch",
                        "candidate": candidate_name,
                        "message": (
                            f"{candidate_name} artifact {field} does not "
                            "match the profile companion"
                        ),
                    }
                )
        if (
            candidate_name == "winner"
            and artifact["digest"] != identity["production_manifest_sha256"]
        ):
            valid = False
            diagnostics.append(
                {
                    "severity": "error",
                    "code": "identity_mismatch",
                    "candidate": candidate_name,
                    "message": (
                        "winner artifact digest does not match the exact "
                        "production package bound by the companion"
                    ),
                }
            )
    return valid


def _clock_validity(
    evidence: Mapping[str, Any], diagnostics: list[dict[str, Any]]
) -> bool:
    all_valid = True
    for candidate_name in ("baseline", "winner"):
        grouped: dict[str, list[int]] = defaultdict(list)
        for mapping in evidence["experiments"][candidate_name]["clock"]:
            problems: list[str] = []
            if not mapping["valid"]:
                problems.append("calibration marked invalid")
            if not mapping["monotonic"]:
                problems.append("mapped clock is non-monotonic")
            if int(mapping["round_trips"]) < 32:
                problems.append("fewer than 32 calibration round trips")
            if problems:
                all_valid = False
                grouped[", ".join(problems)].append(int(mapping["tile"]))
        for message, tiles in sorted(grouped.items()):
            if tiles == list(TILES):
                tile_text = "tiles 0–15"
            else:
                tile_text = "tiles " + ", ".join(str(tile) for tile in tiles)
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "clock_invalid",
                    "candidate": candidate_name,
                    "message": f"{tile_text}: {message}",
                }
            )
    return all_valid


def _output_validity(
    evidence: Mapping[str, Any], diagnostics: list[dict[str, Any]]
) -> tuple[bool, bool | None, dict[str, Any]]:
    output = evidence["output_validation"]
    rows = output["resources"]
    repeated = all(
        bool(row["production_winner_repeat_exact"]) for row in rows
    )
    equivalent = all(bool(row["candidate_equivalent_exact"]) for row in rows)
    output_equivalence = repeated and equivalent
    external = [row["external_expected_exact"] for row in rows]
    if any(state is False for state in external):
        semantic_correctness: bool | None = False
    elif all(state is True for state in external):
        semantic_correctness = True
    else:
        semantic_correctness = None
    if not output_equivalence:
        diagnostics.append(
            {
                "severity": "error",
                "code": "output_equivalence_failed",
                "message": (
                    "production repeatability or baseline/winner/capture "
                    "writable-output equivalence failed"
                ),
            }
        )
    if semantic_correctness is False:
        diagnostics.append(
            {
                "severity": "error",
                "code": "external_expected_failed",
                "message": (
                    "at least one independently supplied expected output "
                    "failed exact comparison"
                ),
            }
        )
    elif semantic_correctness is None:
        diagnostics.append(
            {
                "severity": "warning",
                "code": "semantic_correctness_unknown",
                "message": (
                    "at least one writable resource has no independent "
                    "expected output; same-session exact equivalence cannot "
                    "prove absolute semantic correctness"
                ),
            }
        )
    summary = {
        "mode": str(output["mode"]),
        "resource_count": len(rows),
        "external_expected_resource_count": sum(
            state is not None for state in external
        ),
        "output_equivalence": output_equivalence,
        "semantic_correctness": semantic_correctness,
    }
    return output_equivalence, semantic_correctness, summary


def _trace_validity(
    evidence: Mapping[str, Any], diagnostics: list[dict[str, Any]]
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for candidate_name in ("baseline", "winner"):
        candidate = evidence["experiments"][candidate_name]
        trace = candidate["trace"]
        valid = bool(trace["complete"])
        if not trace["complete"]:
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "trace_incomplete",
                    "candidate": candidate_name,
                    "message": "count preflight/full trace did not complete",
                }
            )
        for row in trace["tiles"]:
            problems: list[str] = []
            preflight_count = int(row["preflight_count"])
            next_sequence = int(row["next_sequence"])
            count = int(row["count"])
            dropped = int(row["dropped_event_count"])
            flags = int(row["record_flags"])
            state = int(row["trace_state"])
            if preflight_count != next_sequence:
                problems.append(
                    "count preflight differs from trace next_sequence"
                )
            if next_sequence != count:
                problems.append(
                    "trace next_sequence differs from published event count"
                )
            if dropped:
                problems.append(f"{dropped} events were dropped")
            if state != TRACE_STATE_COMPLETE:
                problems.append(f"terminal trace state is {state}, not complete")
            if flags != TRACE_EXPECTED_COMPLETE_FLAGS:
                problems.append(
                    "terminal record flags are not the exact complete-trace set"
                )
            if row["overflow"]:
                problems.append("trace buffer overflowed")
            if count > int(row["capacity"]):
                problems.append("event count exceeded capacity")
            invalid_events = sum(not event["valid"] for event in row["events"])
            if invalid_events:
                problems.append(f"{invalid_events} events marked invalid")
            if problems:
                valid = False
                diagnostics.append(
                    {
                        "severity": "warning",
                        "code": "trace_invalid",
                        "candidate": candidate_name,
                        "tile": int(row["tile"]),
                        "message": ", ".join(problems),
                    }
                )
        result[candidate_name] = valid
    return result


def _pmu_validity_and_deltas(
    evidence: Mapping[str, Any], diagnostics: list[dict[str, Any]]
) -> tuple[
    dict[str, bool],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_rows: dict[str, dict[int, Mapping[str, Any]]] = {}
    candidate_valid = {"baseline": True, "winner": True}
    for candidate_name in ("baseline", "winner"):
        rows = evidence["experiments"][candidate_name]["pmu"]["tiles"]
        candidate_rows[candidate_name] = _by_tile(rows)
        for tile, row in candidate_rows[candidate_name].items():
            for counter in AGGREGATE_COUNTERS:
                valid, reason = _counter_status(row["aggregates"][counter])
                if not valid:
                    candidate_valid[candidate_name] = False
                    diagnostics.append(
                        {
                            "severity": "warning",
                            "code": "pmu_counter_invalid",
                            "candidate": candidate_name,
                            "tile": tile,
                            "scope": f"aggregate:{counter}",
                            "message": str(reason),
                        }
                    )
            for worker in row["workers"]:
                for engine in worker["engines"]:
                    for metric in ("instructions", "blocking"):
                        valid, reason = _counter_status(engine[metric])
                        if not valid:
                            candidate_valid[candidate_name] = False
                            diagnostics.append(
                                {
                                    "severity": "warning",
                                    "code": "pmu_counter_invalid",
                                    "candidate": candidate_name,
                                    "tile": tile,
                                    "scope": (
                                        f"worker:{worker['worker']}:"
                                        f"{engine['engine']}:{metric}"
                                    ),
                                    "message": str(reason),
                                }
                            )

    aggregate_deltas: list[dict[str, Any]] = []
    for counter in AGGREGATE_COUNTERS:
        totals: dict[str, int] = {}
        invalid_tiles: set[int] = set()
        for candidate_name in ("baseline", "winner"):
            total = 0
            for tile in TILES:
                snapshot = candidate_rows[candidate_name][tile]["aggregates"][
                    counter
                ]
                valid, _ = _counter_status(snapshot)
                if not valid:
                    invalid_tiles.add(tile)
                    continue
                total += int(snapshot["end"]) - int(snapshot["start"])
            totals[candidate_name] = total
        complete = not invalid_tiles
        aggregate_deltas.append(
            {
                "counter": counter,
                "valid": complete,
                "invalid_tiles": sorted(invalid_tiles),
                "baseline": totals["baseline"] if complete else None,
                "winner": totals["winner"] if complete else None,
                "delta": (
                    totals["winner"] - totals["baseline"]
                    if complete
                    else None
                ),
            }
        )

    engine_deltas: list[dict[str, Any]] = []
    for engine_name in ENGINES:
        for metric in ("instructions", "execution", "blocking"):
            totals: dict[str, int] = {}
            invalid_tiles: set[int] = set()
            for candidate_name in ("baseline", "winner"):
                total = 0
                for tile in TILES:
                    tile_row = candidate_rows[candidate_name][tile]
                    if metric == "execution":
                        snapshot = tile_row["aggregates"][engine_name.lower()]
                        valid, _ = _counter_status(snapshot)
                        if not valid:
                            invalid_tiles.add(tile)
                            continue
                        total += int(snapshot["end"]) - int(snapshot["start"])
                    else:
                        tile_valid = True
                        tile_total = 0
                        for worker in tile_row["workers"]:
                            engine = next(
                                row
                                for row in worker["engines"]
                                if row["engine"] == engine_name
                            )
                            valid, _ = _counter_status(engine[metric])
                            if not valid:
                                tile_valid = False
                                break
                            tile_total += int(engine[metric]["end"]) - int(
                                engine[metric]["start"]
                            )
                        if not tile_valid:
                            invalid_tiles.add(tile)
                            continue
                        total += tile_total
                totals[candidate_name] = total
            complete = not invalid_tiles
            engine_deltas.append(
                {
                    "engine": engine_name,
                    "metric": metric,
                    "valid": complete,
                    "invalid_tiles": sorted(invalid_tiles),
                    "baseline": totals["baseline"] if complete else None,
                    "winner": totals["winner"] if complete else None,
                    "delta": (
                        totals["winner"] - totals["baseline"]
                        if complete
                        else None
                    ),
                }
            )

    worker_engine_deltas: list[dict[str, Any]] = []
    for worker_id in WORKERS:
        for engine_name in ENGINES:
            for metric in ("instructions", "blocking"):
                totals: dict[str, int] = {}
                invalid_tiles: set[int] = set()
                for candidate_name in ("baseline", "winner"):
                    total = 0
                    for tile in TILES:
                        worker = next(
                            row
                            for row in candidate_rows[candidate_name][tile][
                                "workers"
                            ]
                            if int(row["worker"]) == worker_id
                        )
                        engine = next(
                            row
                            for row in worker["engines"]
                            if row["engine"] == engine_name
                        )
                        valid, _ = _counter_status(engine[metric])
                        if not valid:
                            invalid_tiles.add(tile)
                            continue
                        total += int(engine[metric]["end"]) - int(
                            engine[metric]["start"]
                        )
                    totals[candidate_name] = total
                complete = not invalid_tiles
                worker_engine_deltas.append(
                    {
                        "worker": worker_id,
                        "engine": engine_name,
                        "metric": metric,
                        "valid": complete,
                        "invalid_tiles": sorted(invalid_tiles),
                        "baseline": (
                            totals["baseline"] if complete else None
                        ),
                        "winner": totals["winner"] if complete else None,
                        "delta": (
                            totals["winner"] - totals["baseline"]
                            if complete
                            else None
                        ),
                    }
                )
    return (
        candidate_valid,
        engine_deltas,
        aggregate_deltas,
        worker_engine_deltas,
    )


def _trace_site_counts(
    evidence: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    sites_by_id = {
        (
            str(site["candidate"]),
            int(site["tile"]),
            int(site["site_id"]),
        ): site
        for site in evidence["sites"]
    }
    sites_by_correlation: dict[
        str, dict[str, list[Mapping[str, Any]]]
    ] = {
        "baseline": defaultdict(list),
        "winner": defaultdict(list),
    }
    for site in evidence["sites"]:
        sites_by_correlation[str(site["candidate"])][
            str(site["correlation_key"])
        ].append(site)
    counts: dict[str, Counter[str]] = {
        "baseline": Counter(),
        "winner": Counter(),
    }
    tile_counts: dict[str, dict[str, Counter[int]]] = {
        "baseline": defaultdict(Counter),
        "winner": defaultdict(Counter),
    }
    engine_counts: dict[str, Counter[str]] = {
        "baseline": Counter(),
        "winner": Counter(),
    }
    for candidate_name in ("baseline", "winner"):
        for tile in evidence["experiments"][candidate_name]["trace"]["tiles"]:
            for event in tile["events"]:
                site = sites_by_id[
                    (
                        candidate_name,
                        int(tile["tile"]),
                        int(event["site_id"]),
                    )
                ]
                correlation_key = str(site["correlation_key"])
                counts[candidate_name][correlation_key] += 1
                tile_counts[candidate_name][correlation_key][
                    int(tile["tile"])
                ] += 1
                engine_counts[candidate_name][str(event["engine"])] += 1
    changed: list[dict[str, Any]] = []
    all_correlations = sorted(
        set(sites_by_correlation["baseline"])
        | set(sites_by_correlation["winner"])
    )
    for correlation_key in all_correlations:
        baseline_sites = sites_by_correlation["baseline"].get(
            correlation_key, []
        )
        winner_sites = sites_by_correlation["winner"].get(
            correlation_key, []
        )
        baseline = counts["baseline"][correlation_key]
        winner = counts["winner"][correlation_key]
        metadata = lambda rows: {
            (
                str(row["engine"]),
                int(row["target_call_ordinal"]),
                str(row["target_call_symbol"]),
            )
            for row in rows
        }
        metadata_changed = (
            bool(baseline_sites)
            and bool(winner_sites)
            and metadata(baseline_sites) != metadata(winner_sites)
        )
        tile_count_changes = [
            {
                "tile": tile,
                "baseline": int(
                    tile_counts["baseline"][correlation_key][tile]
                ),
                "winner": int(tile_counts["winner"][correlation_key][tile]),
                "delta": int(
                    tile_counts["winner"][correlation_key][tile]
                    - tile_counts["baseline"][correlation_key][tile]
                ),
            }
            for tile in TILES
            if tile_counts["baseline"][correlation_key][tile]
            != tile_counts["winner"][correlation_key][tile]
        ]
        placement_changed = bool(tile_count_changes)
        if (
            baseline_sites
            and winner_sites
            and baseline == winner
            and not metadata_changed
            and not placement_changed
        ):
            continue
        display_site = (winner_sites or baseline_sites)[0]
        if not baseline_sites:
            change_kind = "added"
        elif not winner_sites:
            change_kind = "removed"
        elif metadata_changed:
            change_kind = "modified"
        elif baseline == winner and placement_changed:
            change_kind = "placement_changed"
        else:
            change_kind = "count_changed"
        changed.append(
            {
                "correlation_key": correlation_key,
                "change_kind": change_kind,
                "baseline_sites": [
                    {
                        "tile": int(site["tile"]),
                        "site_id": int(site["site_id"]),
                    }
                    for site in sorted(
                        baseline_sites,
                        key=lambda site: (
                            int(site["tile"]),
                            int(site["site_id"]),
                        ),
                    )
                ],
                "winner_sites": [
                    {
                        "tile": int(site["tile"]),
                        "site_id": int(site["site_id"]),
                    }
                    for site in sorted(
                        winner_sites,
                        key=lambda site: (
                            int(site["tile"]),
                            int(site["site_id"]),
                        ),
                    )
                ],
                "baseline_site_ids": sorted(
                    {int(site["site_id"]) for site in baseline_sites}
                ),
                "winner_site_ids": sorted(
                    {int(site["site_id"]) for site in winner_sites}
                ),
                "baseline_engines": sorted(
                    {str(site["engine"]) for site in baseline_sites}
                ),
                "winner_engines": sorted(
                    {str(site["engine"]) for site in winner_sites}
                ),
                "engine": str(display_site["engine"]),
                "target_call_ordinal": int(
                    display_site["target_call_ordinal"]
                ),
                "target_call_symbol": str(
                    display_site["target_call_symbol"]
                ),
                "position": display_site.get("position"),
                "correlation_confidence": "heuristic",
                "positions": [
                    {
                        "tile": int(site["tile"]),
                        "position": site.get("position"),
                    }
                    for site in sorted(
                        winner_sites or baseline_sites,
                        key=lambda site: int(site["tile"]),
                    )
                ],
                "baseline": baseline,
                "winner": winner,
                "delta": winner - baseline,
                "tile_count_changes": tile_count_changes,
                "placement_delta_magnitude": sum(
                    abs(int(row["delta"])) for row in tile_count_changes
                ),
            }
        )
    engine_summary = {
        candidate: {
            engine: int(engine_counts[candidate][engine]) for engine in ENGINES
        }
        for candidate in ("baseline", "winner")
    }
    return changed, engine_summary


def _trace_issue_metrics(
    evidence: Mapping[str, Any],
    trace_validity: Mapping[str, bool],
) -> dict[str, dict[int, dict[str, float | None]]]:
    result: dict[str, dict[int, dict[str, float | None]]] = {}
    for candidate_name in ("baseline", "winner"):
        by_tile: dict[int, dict[str, float | None]] = {}
        for row in evidence["experiments"][candidate_name]["trace"]["tiles"]:
            tile = int(row["tile"])
            events = row["events"]
            span: float | None = None
            median_gap: float | None = None
            if events and trace_validity[candidate_name]:
                begins = [int(event["begin_cycle"]) for event in events]
                returns = [int(event["return_cycle"]) for event in events]
                span = float(max(returns) - min(begins))
                gaps = [
                    begins[index] - returns[index - 1]
                    for index in range(1, len(events))
                ]
                if gaps:
                    median_gap = _median(gaps)
            by_tile[tile] = {
                "issue_span": span,
                "local_inter_call_gap": median_gap,
            }
        result[candidate_name] = by_tile
    return result


def _cross_tile_order_summary(
    evidence: Mapping[str, Any],
    trace_validity: Mapping[str, bool],
    clock_valid: bool,
) -> dict[str, dict[str, int | bool]]:
    """Count only pairwise cross-tile relations proven by disjoint intervals.

    A mapped event interval includes the calibration uncertainty on both ends.
    No transitive or total order is inferred from these counts.
    """

    result: dict[str, dict[str, int | bool]] = {}
    for candidate_name in ("baseline", "winner"):
        if not trace_validity[candidate_name] or not clock_valid:
            result[candidate_name] = {
                "valid": False,
                "proven_pairs": 0,
                "overlapping_pairs": 0,
                "total_pairs": 0,
            }
            continue
        candidate = evidence["experiments"][candidate_name]
        clock = _clock_map(candidate)
        intervals: list[tuple[int, Decimal, Decimal]] = []
        for tile_row in candidate["trace"]["tiles"]:
            tile = int(tile_row["tile"])
            uncertainty = Decimal(str(clock[tile]["uncertainty"]))
            for event in tile_row["events"]:
                intervals.append(
                    (
                        tile,
                        _mapped_decimal(
                            int(event["begin_cycle"]), clock[tile]
                        )
                        - uncertainty,
                        _mapped_decimal(
                            int(event["return_cycle"]), clock[tile]
                        )
                        + uncertainty,
                    )
                )
        proven = 0
        overlapping = 0
        for left_index, left in enumerate(intervals):
            for right in intervals[left_index + 1 :]:
                if left[0] == right[0]:
                    continue
                if left[2] < right[1] or right[2] < left[1]:
                    proven += 1
                else:
                    overlapping += 1
        result[candidate_name] = {
            "valid": True,
            "proven_pairs": proven,
            "overlapping_pairs": overlapping,
            "total_pairs": proven + overlapping,
        }
    return result


def _timeline_events(
    evidence: Mapping[str, Any],
    clock_valid: bool,
    trace_validity: Mapping[str, bool],
) -> list[dict[str, Any]]:
    """Build browser-safe relative event coordinates.

    Raw cycle values remain decimal strings.  Plot coordinates are derived
    only after exact integer subtraction (entry-local mode) or Decimal clock
    mapping followed by subtraction of a common origin (qualified mode).
    This keeps a long-running device clock above 2**53 from corrupting short
    TsmExecute intervals in JavaScript.
    """

    staged: list[tuple[dict[str, Any], Decimal, Decimal]] = []
    for candidate_name in ("baseline", "winner"):
        if not trace_validity[candidate_name]:
            continue
        candidate = evidence["experiments"][candidate_name]
        clock = _clock_map(candidate)
        for tile_row in candidate["trace"]["tiles"]:
            tile = int(tile_row["tile"])
            entry_begin = int(tile_row["entry_begin_cycle"])
            for event in tile_row["events"]:
                begin = int(event["begin_cycle"])
                end = int(event["return_cycle"])
                if clock_valid:
                    plot_begin = _mapped_decimal(begin, clock[tile])
                    plot_end = _mapped_decimal(end, clock[tile])
                    uncertainty: float | None = float(
                        clock[tile]["uncertainty"]
                    )
                else:
                    plot_begin = Decimal(begin - entry_begin)
                    plot_end = Decimal(end - entry_begin)
                    uncertainty = None
                staged.append(
                    (
                        {
                            "candidate": candidate_name,
                            "tile": tile,
                            "sequence": int(event["sequence"]),
                            "site_id": int(event["site_id"]),
                            "sub_index": int(event["sub_index"]),
                            "engine": str(event["engine"]),
                            "local_begin_cycle": str(begin),
                            "local_return_cycle": str(end),
                            "raw_result": str(int(event["raw_result"])),
                            "valid": bool(event["valid"]),
                            "uncertainty": uncertainty,
                        },
                        plot_begin,
                        plot_end,
                    )
                )
    origin = (
        min((begin for _, begin, _ in staged), default=Decimal(0))
        if clock_valid
        else Decimal(0)
    )
    result: list[dict[str, Any]] = []
    for row, begin, end in staged:
        row["plot_begin"] = float(begin - origin)
        row["plot_return"] = float(end - origin)
        result.append(row)
    return result


def _build_insights(
    *,
    verdict: str,
    performance: Mapping[str, Any],
    tile_metrics: Sequence[Mapping[str, Any]],
    changed_sites: Sequence[Mapping[str, Any]],
    engine_deltas: Sequence[Mapping[str, Any]],
    validity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    labels = {
        "improved": "Measured improvement",
        "regressed": "Measured regression",
        "inconclusive": "Performance direction is inconclusive",
        "invalid": "Performance measurement is invalid",
    }
    if verdict == "invalid":
        performance_body = (
            "One or more output-equivalence, identity, environment, package, or "
            "measurement-basis gates failed. Clock qualification is not a "
            "host-latency gate. No speed claim is published."
        )
    elif not validity["distinct_candidates"]:
        performance_body = (
            "Reserved baseline aliases, or is byte-identical to, the "
            "production winner. The timing samples characterize repeatability "
            "of one artifact; they cannot establish an optimization effect."
        )
    else:
        ratio = float(performance["speedup_ratio"])
        low, high = performance["paired_bootstrap_ci95"]
        performance_body = (
            f"Training blocks give {ratio:.4f}× baseline/winner "
            f"launch-to-completion speedup "
            f"(paired bootstrap 95% CI {low:.4f}–{high:.4f}); the "
            f"held-out block gives "
            f"{float(performance['held_out_speedup_ratio']):.4f}×."
        )
        if validity["semantic_correctness"] is None:
            performance_body += (
                " Writable outputs were byte-identical to the same-session "
                "production-winner reference, but absolute semantic "
                "correctness is unknown without independent expected bytes."
            )
    insights: list[dict[str, Any]] = [
        {
            "strength": "measured",
            "title": labels[verdict],
            "body": performance_body,
            "target": {"mode": "difference"},
        }
    ]
    if validity["semantic_correctness"] is None:
        insights.append(
            {
                "strength": "unresolved",
                "title": "Absolute output correctness is unqualified",
                "body": (
                    "At least one writable resource used the same-session "
                    "production winner as its exact reference. This proves "
                    "candidate and instrumentation equivalence, not that a "
                    "shared deterministic result is semantically correct."
                ),
                "target": {"mode": "difference"},
            }
        )

    critical = max(tile_metrics, key=lambda row: row["winner_duration"])
    insights.append(
        {
            "strength": "measured",
            "title": (
                "Longest winner rank-local entry interval is "
                f"T{int(critical['tile']):02d}"
            ),
            "body": (
                f"Its diagnostic summary entry duration is "
                f"{float(critical['winner_duration']):.1f} local cycles. "
                "This rank-local comparison does not establish which physical "
                "tile completed last or a global critical path."
            ),
            "target": {
                "mode": "winner",
                "tile": int(critical["tile"]),
            },
        }
    )

    trace_comparable = (
        validity["cross_candidate_explanation"]
        and validity["trace"]["baseline"]
        and validity["trace"]["winner"]
    )
    if not validity["cross_candidate_explanation"]:
        insights.append(
            {
                "strength": "unresolved",
                "title": "Cross-candidate explanation is withheld",
                "body": (
                    "Identity, exact companion binding, output equivalence, "
                    "environment, and distinct-artifact gates must all pass "
                    "before tile, TSM-site, or PMU differences are published."
                ),
                "target": {"mode": "baseline"},
            }
        )
    elif not trace_comparable:
        insights.append(
            {
                "strength": "unresolved",
                "title": "TSM structural comparison is withheld",
                "body": (
                    "At least one trace failed completeness, capacity, or "
                    "event-validity checks, so site and engine call-count "
                    "differences are not analyzed."
                ),
                "target": {"mode": "difference"},
            }
        )
    elif changed_sites:
        largest = max(
            changed_sites,
            key=lambda row: max(
                abs(int(row["delta"])),
                int(row["placement_delta_magnitude"]),
            ),
        )
        if int(largest["delta"]) == 0:
            structural_change = (
                "has equal full-card call count, but its calls moved between "
                f"{len(largest['tile_count_changes'])} tile rows"
            )
        else:
            structural_change = (
                "has full-card call-count delta "
                f"{int(largest['delta']):+d}"
            )
        insights.append(
            {
                "strength": "correlated",
                "title": (
                    f"{len(changed_sites)} heuristic TSM correlation"
                    f"{'s' if len(changed_sites) != 1 else ''} differ"
                ),
                "body": (
                    f"Correlation {largest['correlation_key']} "
                    f"({largest['engine']} "
                    f"{largest['target_call_symbol']}) {structural_change}. "
                    "The key is a target-call signature/occurrence heuristic, "
                    "not stable IR provenance; candidate-local IDs and this "
                    "correlation must not be read as causal attribution."
                ),
                "target": {
                    "mode": "difference",
                    "correlation_key": str(largest["correlation_key"]),
                },
            }
        )
    else:
        insights.append(
            {
                "strength": "unresolved",
                "title": "No TSM call-count difference",
                "body": (
                    "The captured traces have equal per-tile counts under "
                    "every heuristic target-call correlation. Any measured "
                    "change is not explained by this structural comparison."
                ),
                "target": {"mode": "difference"},
            }
        )

    valid_execution = [
        row
        for row in engine_deltas
        if validity["cross_candidate_explanation"]
        and row["metric"] == "execution"
        and row["valid"]
    ]
    if valid_execution:
        largest = max(valid_execution, key=lambda row: abs(int(row["delta"])))
        insights.append(
            {
                "strength": "correlated",
                "title": f"{largest['engine']} has the largest PMU delta",
                "body": (
                    f"The full-card aggregate execution counter changes by "
                    f"{int(largest['delta']):+d}. Aggregate PMU evidence is "
                    "not assigned to individual TSM calls."
                ),
                "target": {"mode": "difference"},
            }
        )
    else:
        insights.append(
            {
                "strength": "unresolved",
                "title": "PMU comparison is unavailable",
                "body": (
                    "Cross-candidate explanation is gated off, or no engine "
                    "has a complete valid 16-tile counter pair. PMU data is "
                    "excluded from correlation."
                ),
                "target": {"mode": "difference"},
            }
        )

    insights.append(
        {
            "strength": "unresolved",
            "title": "TSM records are issue evidence",
            "body": (
                "Each mark covers only the wrapper call-to-return interval. "
                "It does not identify NCC hardware completion. The report "
                "uses entry-local tile axes unless cross-tile clock mapping "
                "is separately qualified; only disjoint uncertainty "
                "intervals support an order claim."
            ),
            "target": {"mode": "difference"},
        }
    )
    if not validity["clock"]:
        insights.append(
            {
                "strength": "unresolved",
                "title": "Cross-tile order is withheld",
                "body": (
                    "Clock alignment did not pass qualification. All 16 tile "
                    "rows remain visible on entry-local cycle axes, but their "
                    "horizontal positions do not define a global order."
                ),
                "target": {"mode": "difference"},
            }
        )
    return insights


def analyze_evidence(value: object) -> dict[str, Any]:
    """Validate evidence and return the deterministic analysis record."""

    evidence = validate_evidence(value)
    diagnostics: list[dict[str, Any]] = []
    identity_valid = _identity_validity(evidence, diagnostics)
    clock_valid = _clock_validity(evidence, diagnostics)
    (
        output_equivalence_valid,
        semantic_correctness,
        output_validation,
    ) = _output_validity(evidence, diagnostics)
    trace_validity = _trace_validity(evidence, diagnostics)
    (
        pmu_validity,
        engine_deltas,
        aggregate_deltas,
        worker_engine_deltas,
    ) = _pmu_validity_and_deltas(evidence, diagnostics)

    samples: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "winner": [],
    }
    for sample in evidence["measurement"]["samples"]:
        samples[str(sample["candidate"])].append(
            {
                "sample_id": str(sample["sample_id"]),
                "block": int(sample["block"]),
                "position": int(sample["position"]),
                "host_elapsed_ns": int(sample["host_elapsed_ns"]),
                "completion_observation_resolution_ns": int(
                    sample["completion_observation_resolution_ns"]
                ),
            }
        )

    summary_metrics: dict[str, dict[str, Any]] = {}
    for candidate_name in ("baseline", "winner"):
        candidate = evidence["experiments"][candidate_name]
        clock = _clock_map(candidate)
        summary_metrics[candidate_name] = _summary_metrics(
            candidate["summary"], clock
        )

    block_rows = {
        int(row["block"]): row for row in evidence["measurement"]["blocks"]
    }
    block_latencies: dict[str, dict[int, float]] = {
        "baseline": {},
        "winner": {},
    }
    for candidate_name in ("baseline", "winner"):
        grouped: dict[int, list[float]] = defaultdict(list)
        for sample in samples[candidate_name]:
            grouped[int(sample["block"])].append(
                float(sample["host_elapsed_ns"])
            )
        block_latencies[candidate_name] = {
            block: _median(values) for block, values in grouped.items()
        }

    training_blocks = sorted(
        block for block, row in block_rows.items() if not row["held_out"]
    )
    held_out_block = next(
        block for block, row in block_rows.items() if row["held_out"]
    )
    baseline_block_medians = [
        block_latencies["baseline"][block] for block in training_blocks
    ]
    winner_block_medians = [
        block_latencies["winner"][block] for block in training_blocks
    ]
    sample_latencies_by_block: dict[str, dict[int, list[float]]] = {
        "baseline": defaultdict(list),
        "winner": defaultdict(list),
    }
    for candidate_name in ("baseline", "winner"):
        for sample in samples[candidate_name]:
            sample_latencies_by_block[candidate_name][
                int(sample["block"])
            ].append(float(sample["host_elapsed_ns"]))
    baseline_training_blocks = [
        sample_latencies_by_block["baseline"][block]
        for block in training_blocks
    ]
    winner_training_blocks = [
        sample_latencies_by_block["winner"][block]
        for block in training_blocks
    ]
    baseline_training = [
        value for block in baseline_training_blocks for value in block
    ]
    winner_training = [
        value for block in winner_training_blocks for value in block
    ]
    if any(latency <= 0 for latency in baseline_training + winner_training):
        _fail(
            "measurement.samples",
            "launch-to-completion host latencies must be positive",
        )
    baseline_median = _median(baseline_training)
    winner_median = _median(winner_training)
    speedup_ratio = baseline_median / winner_median
    bootstrap_ci = _paired_bootstrap_ci(
        baseline_training_blocks, winner_training_blocks
    )
    paired_effects = [
        (baseline - winner) / baseline
        for baseline, winner in zip(
            baseline_block_medians, winner_block_medians, strict=True
        )
    ]
    effect_median = _median(paired_effects)
    paired_mad = _median(
        [abs(effect - effect_median) for effect in paired_effects]
    )
    threshold = max(0.01, 3.0 * paired_mad)
    held_out_speedup = (
        block_latencies["baseline"][held_out_block]
        / block_latencies["winner"][held_out_block]
    )
    resolution_fractions = [
        float(sample["completion_observation_resolution_ns"])
        / float(sample["host_elapsed_ns"])
        for candidate_samples in samples.values()
        for sample in candidate_samples
    ]
    maximum_resolution_fraction = max(resolution_fractions)
    maximum_resolution_ns = max(
        int(sample["completion_observation_resolution_ns"])
        for candidate_samples in samples.values()
        for sample in candidate_samples
    )
    resolution_valid = (
        maximum_resolution_fraction <= MAX_COMPLETION_RESOLUTION_FRACTION
    )

    global_validity = evidence["validity"]
    performance_valid = (
        identity_valid
        and output_equivalence_valid
        and semantic_correctness is not False
        and bool(global_validity["environment"])
        and bool(global_validity["package_companion"])
        and bool(global_validity["measurement_basis"])
        and resolution_valid
    )
    baseline_digest = str(
        evidence["experiments"]["baseline"]["artifact"]["digest"]
    )
    winner_digest = str(
        evidence["experiments"]["winner"]["artifact"]["digest"]
    )
    distinct_candidates = baseline_digest != winner_digest
    cross_candidate_explanation = (
        identity_valid
        and output_equivalence_valid
        and semantic_correctness is not False
        and bool(global_validity["environment"])
        and bool(global_validity["package_companion"])
        and distinct_candidates
    )
    if not distinct_candidates:
        diagnostics.append(
            {
                "severity": "warning",
                "code": "identical_candidate_artifacts",
                "message": (
                    "baseline and winner execution-package digests are "
                    "identical; optimization-effect claims are disabled"
                ),
            }
        )
    for field in ("environment", "package_companion", "measurement_basis"):
        if not global_validity[field]:
            diagnostics.append(
                {
                    "severity": "error",
                    "code": f"{field}_invalid",
                    "message": f"global {field} gate failed",
                }
            )
    if not resolution_valid:
        diagnostics.append(
            {
                "severity": "error",
                "code": "completion_resolution_too_coarse",
                "message": (
                    "maximum completion-observation gap is "
                    f"{maximum_resolution_fraction * 100.0:.3f}% of its "
                    "sample latency; profiler v2 requires at most "
                    f"{MAX_COMPLETION_RESOLUTION_FRACTION * 100.0:.2f}%"
                ),
            }
        )

    if not performance_valid:
        verdict = "invalid"
    elif not distinct_candidates:
        verdict = "inconclusive"
    elif (
        speedup_ratio - 1.0 > threshold
        and bootstrap_ci[0] > 1.0
        and held_out_speedup > 1.0
    ):
        verdict = "improved"
    elif (
        1.0 - speedup_ratio > threshold
        and bootstrap_ci[1] < 1.0
        and held_out_speedup < 1.0
    ):
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    performance = {
        "basis": "host launch-to-completion steady-clock latency",
        "candidate_artifacts_distinct": distinct_candidates,
        "maximum_completion_observation_resolution_ns": (
            maximum_resolution_ns
        ),
        "maximum_completion_resolution_fraction": (
            maximum_resolution_fraction
        ),
        "baseline_median_latency_ns": baseline_median,
        "winner_median_latency_ns": winner_median,
        "speedup_ratio": speedup_ratio,
        "speedup_percent": (speedup_ratio - 1.0) * 100.0,
        "paired_bootstrap_ci95": list(bootstrap_ci),
        "paired_mad_noise": paired_mad,
        "materiality_threshold": threshold,
        "held_out_speedup_ratio": held_out_speedup,
        "held_out_block": held_out_block,
        "training_blocks": training_blocks,
        "block_latencies": {
            candidate: [
                {
                    "block": block,
                    "latency_ns": block_latencies[candidate][block],
                    "held_out": bool(block_rows[block]["held_out"]),
                }
                for block in sorted(block_rows)
            ]
            for candidate in ("baseline", "winner")
        },
        "sample_latencies": {
            candidate: [
                {
                    "sample_id": row["sample_id"],
                    "block": row["block"],
                    "position": row["position"],
                    "latency_ns": row["host_elapsed_ns"],
                    "completion_observation_resolution_ns": row[
                        "completion_observation_resolution_ns"
                    ],
                }
                for row in samples[candidate]
            ]
            for candidate in ("baseline", "winner")
        },
        "cross_tile_entry_skew": (
            {
                "baseline": float(
                    summary_metrics["baseline"]["entry_skew"]
                ),
                "winner": float(summary_metrics["winner"]["entry_skew"]),
            }
            if clock_valid
            else None
        ),
        "cross_tile_finish_skew": (
            {
                "baseline": float(
                    summary_metrics["baseline"]["finish_skew"]
                ),
                "winner": float(summary_metrics["winner"]["finish_skew"]),
            }
            if clock_valid
            else None
        ),
    }

    topology = {
        int(row["tile"]): (int(row["x"]), int(row["y"]))
        for row in evidence["topology"]
    }
    tile_metrics: list[dict[str, Any]] = []
    for tile in TILES:
        baseline_duration = int(
            summary_metrics["baseline"]["durations"][tile]
        )
        winner_duration = int(summary_metrics["winner"]["durations"][tile])
        x, y = topology[tile]
        tile_metrics.append(
            {
                "tile": tile,
                "x": x,
                "y": y,
                "baseline_duration": baseline_duration,
                "winner_duration": winner_duration,
                "delta": (
                    winner_duration - baseline_duration
                    if cross_candidate_explanation
                    else None
                ),
                "improvement_percent": (
                    (baseline_duration - winner_duration)
                    / baseline_duration
                    * 100.0
                    if baseline_duration and cross_candidate_explanation
                    else None
                ),
            }
        )

    if (
        cross_candidate_explanation
        and trace_validity["baseline"]
        and trace_validity["winner"]
    ):
        changed_sites, engine_counts = _trace_site_counts(evidence)
    else:
        changed_sites = []
        engine_counts = {
            candidate: {engine: None for engine in ENGINES}
            for candidate in ("baseline", "winner")
        }
    issue_metrics = _trace_issue_metrics(evidence, trace_validity)
    cross_tile_order = _cross_tile_order_summary(
        evidence, trace_validity, clock_valid
    )
    for row in tile_metrics:
        tile = int(row["tile"])
        row["baseline_issue_span"] = issue_metrics["baseline"][tile][
            "issue_span"
        ]
        row["winner_issue_span"] = issue_metrics["winner"][tile]["issue_span"]
        row["baseline_local_inter_call_gap"] = issue_metrics["baseline"][
            tile
        ]["local_inter_call_gap"]
        row["winner_local_inter_call_gap"] = issue_metrics["winner"][tile][
            "local_inter_call_gap"
        ]

    validity = {
        "performance": performance_valid,
        "distinct_candidates": distinct_candidates,
        "cross_candidate_explanation": cross_candidate_explanation,
        "identity": identity_valid,
        "output_equivalence": output_equivalence_valid,
        "semantic_correctness": semantic_correctness,
        "clock": clock_valid,
        "trace": trace_validity,
        "pmu": pmu_validity,
        "environment": bool(global_validity["environment"]),
        "package_companion": bool(global_validity["package_companion"]),
        "measurement_basis": (
            bool(global_validity["measurement_basis"]) and resolution_valid
        ),
        "completion_resolution": resolution_valid,
    }
    if not cross_candidate_explanation:
        def withhold(
            rows: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for row in rows:
                withheld = dict(row)
                withheld.update(
                    {
                        "valid": False,
                        "baseline": None,
                        "winner": None,
                        "delta": None,
                        "withheld": True,
                    }
                )
                result.append(withheld)
            return result

        engine_deltas = withhold(engine_deltas)
        aggregate_deltas = withhold(aggregate_deltas)
        worker_engine_deltas = withhold(worker_engine_deltas)
    insights = _build_insights(
        verdict=verdict,
        performance=performance,
        tile_metrics=tile_metrics,
        changed_sites=changed_sites,
        engine_deltas=engine_deltas,
        validity=validity,
    )
    return {
        "schema": ANALYSIS_SCHEMA_NAME,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_id": evidence["run_id"],
        "verdict": verdict,
        "validity": validity,
        "performance": performance,
        "output_validation": output_validation,
        "tile_metrics": tile_metrics,
        "engine_counts": engine_counts,
        "changed_sites": changed_sites,
        "pmu_engine_deltas": engine_deltas,
        "pmu_aggregate_deltas": aggregate_deltas,
        "pmu_worker_engine_deltas": worker_engine_deltas,
        "cross_tile_order": cross_tile_order,
        "timeline_events": _timeline_events(
            evidence, clock_valid, trace_validity
        ),
        "insights": insights,
        "diagnostics": diagnostics,
        "method": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "materiality": "max(1%, 3 * paired MAD)",
            "maximum_completion_resolution_fraction": (
                MAX_COMPLETION_RESOLUTION_FRACTION
            ),
            "performance_basis": (
                "host steady-clock submit-to-all-completion latency"
            ),
            "trace_semantics": "wrapper call-to-return issue interval",
            "timeline_semantics": (
                "global uncertainty-bounded mapping when qualified; "
                "otherwise per-tile entry-local cycles with no cross-tile "
                "order"
            ),
            "pmu_semantics": "full-card aggregate correlation only",
        },
    }


def _summarize_diagnostics_for_report(
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated PMU failures without changing raw analysis evidence."""

    result: list[dict[str, Any]] = []
    group_indices: dict[tuple[str, str, str], int] = {}
    grouped_tiles: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    grouped_scopes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for source in diagnostics:
        row = dict(source)
        if row.get("code") != "pmu_counter_invalid":
            row["grouped"] = False
            row["grouped_count"] = 1
            result.append(row)
            continue

        severity = str(row.get("severity", "warning"))
        candidate = str(row.get("candidate", "global"))
        message = str(row.get("message", "PMU counter is invalid"))
        key = (severity, candidate, message)
        if key not in group_indices:
            group_indices[key] = len(result)
            result.append(
                {
                    "severity": severity,
                    "code": "pmu_counter_invalid",
                    "candidate": candidate,
                    "message": message,
                    "grouped": True,
                    "grouped_count": 0,
                    "tiles": [],
                    "scopes": [],
                }
            )
        grouped = result[group_indices[key]]
        grouped["grouped_count"] = int(grouped["grouped_count"]) + 1
        if row.get("tile") is not None:
            grouped_tiles[key].add(int(row["tile"]))
        if row.get("scope") is not None:
            grouped_scopes[key].add(str(row["scope"]))

    for key, index in group_indices.items():
        result[index]["tiles"] = sorted(grouped_tiles[key])
        result[index]["scopes"] = sorted(grouped_scopes[key])
    return result


_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wafer Profile · __RUN_ID__</title>
<style>
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #687187;
  --line: #dce2ec;
  --panel: #ffffff;
  --wash: #f4f7fb;
  --accent: #3558d4;
  --ct: #5367dc;
  --ne: #a457c4;
  --rdma: #087c75;
  --wdma: #d47623;
  --tdma: #627489;
  --good: #08765c;
  --bad: #a6404c;
  --warn: #9b6500;
  --shadow: 0 12px 36px rgba(41, 53, 85, .09);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 85% -10%, #e8edff 0, transparent 30rem),
    var(--wash);
  color: var(--ink);
  font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
}
button, input { font: inherit; }
.page { max-width: 1500px; margin: 0 auto; padding: 28px; }
.topline {
  display: flex; justify-content: space-between; align-items: start; gap: 24px;
  margin-bottom: 18px;
}
.eyebrow { color: var(--accent); font-weight: 760; letter-spacing: .08em;
  text-transform: uppercase; font-size: 11px; }
h1 { font-size: clamp(25px, 4vw, 42px); letter-spacing: -.035em;
  margin: 4px 0 5px; line-height: 1.07; }
.subtitle { color: var(--muted); }
.artifact-links { display: flex; gap: 11px; justify-content: flex-end; }
.artifact-links a { color: var(--accent); text-decoration: none; font-weight: 680; }
.artifact-links a:hover { text-decoration: underline; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.card {
  background: color-mix(in srgb, var(--panel) 96%, transparent);
  border: 1px solid var(--line); border-radius: 15px; box-shadow: var(--shadow);
}
.overall { display: grid; grid-template-columns: minmax(250px, 1.1fr) repeat(4, minmax(125px, .55fr));
  overflow: hidden; margin-bottom: 16px; }
.metric { padding: 19px 21px; border-left: 1px solid var(--line); min-height: 105px; }
.metric:first-child { border-left: 0; }
.metric-label { color: var(--muted); font-size: 12px; margin-bottom: 7px; }
.metric-value { font-size: 24px; font-weight: 760; letter-spacing: -.025em; }
.metric-value.large { font-size: 29px; }
.metric-detail { color: var(--muted); font-size: 12px; margin-top: 5px; }
.verdict-improved { color: var(--good); }
.verdict-regressed, .verdict-invalid { color: var(--bad); }
.verdict-inconclusive { color: var(--warn); }
.latency-card { margin-bottom: 16px; }
.latency-grid { display: grid; gap: 10px; }
.latency-row { display: grid; grid-template-columns: 76px 1fr; gap: 12px;
  align-items: center; padding: 8px 0; border-bottom: 1px solid var(--line); }
.latency-row:last-child { border-bottom: 0; }
.latency-label { font: 11px ui-monospace, monospace; color: var(--muted); }
.latency-bars { display: grid; gap: 5px; }
.latency-bar-line { display: grid; grid-template-columns: 58px 1fr 105px;
  gap: 8px; align-items: center; font-size: 11px; }
.latency-track { height: 9px; border-radius: 999px; background: #edf0f5;
  overflow: hidden; }
.latency-fill { display: block; height: 100%; border-radius: inherit;
  background: #929bad; }
.latency-fill.winner { background: var(--accent); }
.grid { display: grid; grid-template-columns: minmax(275px, .68fr) minmax(420px, 1.32fr);
  gap: 16px; margin-bottom: 16px; }
.section { padding: 19px; }
.section-head { display: flex; align-items: baseline; justify-content: space-between;
  gap: 18px; margin-bottom: 14px; }
h2 { font-size: 16px; margin: 0; letter-spacing: -.01em; }
.hint { color: var(--muted); font-size: 12px; }
.tile-map { display: grid; grid-template-columns: repeat(var(--tile-cols, 4), minmax(58px, 1fr));
  gap: 8px; overflow: auto; }
.tile {
  border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  border-radius: 10px; min-height: 75px; padding: 9px; text-align: left;
  cursor: pointer; position: relative; overflow: hidden;
}
.tile::after { content: ""; position: absolute; inset: auto 0 0; height: 5px;
  background: var(--heat, #9da6b7); }
.tile:hover, .tile.selected { border-color: var(--accent); outline: 2px solid #dfe5ff; }
.tile-id { font-weight: 750; }
.tile-delta { font-size: 13px; margin-top: 10px; }
.insights { display: grid; gap: 8px; max-height: 410px; overflow: auto; padding-right: 3px; }
.insight { border: 1px solid var(--line); border-radius: 10px; padding: 11px 12px;
  cursor: pointer; background: #fbfcff; }
.insight:hover { border-color: #aab8f2; }
.insight-top { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.badge { border-radius: 999px; padding: 2px 7px; font-size: 10px; font-weight: 740;
  letter-spacing: .04em; text-transform: uppercase; background: #e7ebf5; }
.badge.measured { color: #075e4a; background: #dff4ed; }
.badge.correlated { color: #684b00; background: #fff0c8; }
.badge.unresolved { color: #6d506e; background: #f2e8f2; }
.insight-title { font-weight: 720; }
.insight-body { color: var(--muted); font-size: 12px; }
.timeline-card { margin-bottom: 16px; overflow: hidden; }
.timeline-toolbar { display: flex; gap: 13px; flex-wrap: wrap; align-items: center;
  padding: 16px 19px; border-bottom: 1px solid var(--line); }
.tabs { display: flex; background: #edf1f7; border-radius: 9px; padding: 3px; }
.tab { border: 0; background: transparent; border-radius: 7px; padding: 6px 11px;
  color: var(--muted); cursor: pointer; }
.tab.active { background: white; color: var(--ink); box-shadow: 0 2px 8px #cbd3e2; }
.tab:disabled { cursor: not-allowed; opacity: .45; }
.slider { display: flex; gap: 7px; align-items: center; color: var(--muted); font-size: 12px; }
.slider input { width: 120px; accent-color: var(--accent); }
.legend { display: flex; gap: 11px; flex-wrap: wrap; margin-left: auto; color: var(--muted);
  font-size: 11px; }
.legend-item { display: flex; align-items: center; gap: 4px; }
.glyph { color: var(--engine); font-weight: 900; }
.timeline-wrap { padding: 12px 18px 17px; overflow: hidden; }
.axis { height: 24px; position: relative; margin-left: 58px; border-bottom: 1px solid var(--line); }
.axis-label { position: absolute; bottom: 4px; transform: translateX(-50%);
  color: var(--muted); font: 10px ui-monospace, monospace; }
.timeline-row { height: 29px; display: grid; grid-template-columns: 52px 1fr; align-items: center; }
.row-label { border: 0; padding: 0 7px 0 0; background: transparent; color: var(--muted);
  text-align: right; font: 11px ui-monospace, monospace; cursor: pointer; }
.row-label:hover, .row-label.selected { color: var(--accent); font-weight: 760; }
.track { height: 17px; position: relative; border-left: 1px solid var(--line);
  background: repeating-linear-gradient(90deg, #f0f2f6 0 1px, transparent 1px 20%); }
.event { position: absolute; min-width: 4px; height: 10px; top: 3px;
  border-radius: 2px; background: var(--engine); border: 1px solid color-mix(in srgb, var(--engine) 80%, #172033);
  cursor: crosshair; opacity: .9; }
.event::after { content: attr(data-glyph); color: white; position: absolute; font-size: 7px;
  left: 1px; top: -1px; font-weight: 900; }
.event.baseline-overlay { top: 9px; height: 6px; background: transparent; border-width: 1px; }
.event.winner-overlay { top: 1px; height: 6px; }
.event.focused { outline: 2px solid #1b2334; z-index: 4; }
.detail { margin: 12px 0 0 58px; padding: 11px 12px; border-radius: 10px;
  border: 1px solid var(--line); background: #fafbfe; display: none; }
.detail.visible { display: block; }
.engine-row { display: grid; grid-template-columns: 58px 1fr; min-height: 24px; align-items: center; }
.engine-label { color: var(--muted); font: 10px ui-monospace, monospace; }
.engine-track { height: 15px; position: relative; border-left: 1px solid var(--line);
  background: #f2f4f8; }
.lower { display: grid; grid-template-columns: 1fr 1.1fr; gap: 16px; }
.table-wrap { overflow: auto; max-height: 390px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: right;
  white-space: nowrap; }
th { position: sticky; top: 0; background: white; z-index: 1; color: var(--muted);
  font-size: 10px; text-transform: uppercase; letter-spacing: .04em; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
.delta-pos, .delta-neg { color: var(--ink); }
.diagnostics { margin-top: 16px; }
.diag { padding: 8px 0; border-bottom: 1px solid var(--line); color: var(--muted); }
.diag-group summary { cursor: pointer; list-style-position: outside; }
.diag-group-body { padding: 7px 0 2px 24px; font-size: 12px; color: var(--muted); }
.tooltip { position: fixed; z-index: 20; pointer-events: none; display: none; width: 315px;
  padding: 10px 11px; border-radius: 9px; background: #172033; color: white;
  box-shadow: 0 12px 30px rgba(0,0,0,.24); font-size: 11px; }
.tooltip strong { color: white; }
.tooltip .dim { color: #bfc8dc; }
.footer { margin: 18px 2px 4px; color: var(--muted); font-size: 11px; }
@media (max-width: 950px) {
  .overall { grid-template-columns: 1fr 1fr; }
  .metric:first-child { grid-column: 1 / -1; }
  .grid, .lower { grid-template-columns: 1fr; }
  .legend { width: 100%; margin-left: 0; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
</head>
<body>
<main class="page">
  <div class="topline">
    <div>
      <div class="eyebrow">16-tile TSM profiler</div>
      <h1>Optimization evidence</h1>
      <div class="subtitle">Run <span class="mono">__RUN_ID__</span> · profile <span class="mono">__TARGET_PROFILE__</span></div>
    </div>
    <div>
      <div class="subtitle">Offline report · evidence schema v1</div>
      <div class="artifact-links"><a href="evidence.json">Raw evidence</a>
        <a href="analysis.json">Analysis JSON</a></div>
    </div>
  </div>

  <section class="card overall">
    <div class="metric">
      <div class="metric-label">Overall verdict</div>
      <div id="verdict" class="metric-value large"></div>
      <div id="verdictDetail" class="metric-detail"></div>
    </div>
    <div class="metric"><div class="metric-label">Speedup</div>
      <div id="speedup" class="metric-value mono"></div>
      <div id="speedupPct" class="metric-detail"></div></div>
    <div class="metric"><div class="metric-label">Paired bootstrap 95% CI</div>
      <div id="ci" class="metric-value mono"></div>
      <div class="metric-detail">10,000 deterministic block resamples</div></div>
    <div class="metric"><div class="metric-label">Output validation</div>
      <div id="outputValidation" class="metric-value"></div>
      <div id="outputValidationDetail" class="metric-detail"></div></div>
    <div class="metric"><div class="metric-label">Trace / PMU</div>
      <div id="evidenceHealth" class="metric-value"></div>
      <div id="evidenceHealthDetail" class="metric-detail"></div></div>
  </section>

  <section class="card section latency-card">
    <div class="section-head"><h2>Balanced launch-to-completion latency</h2>
      <span class="hint">two uninstrumented samples per candidate and block · fifth block held out</span></div>
    <div id="latencyBlocks" class="latency-grid"></div>
  </section>

  <div class="grid">
    <section class="card section">
      <div class="section-head"><h2>Physical tile map</h2>
        <span id="tileMapHint" class="hint">winner local-cycle duration change</span></div>
      <div id="tileMap" class="tile-map"></div>
    </section>
    <section class="card section">
      <div class="section-head"><h2>Ranked findings</h2>
        <span class="hint">click to focus evidence</span></div>
      <div id="insights" class="insights"></div>
    </section>
  </div>

  <section class="card timeline-card">
    <div class="timeline-toolbar">
      <h2>Full-card TSM issue timeline</h2>
      <div class="tabs">
        <button class="tab" data-mode="baseline">Baseline</button>
        <button class="tab" data-mode="winner">Winner</button>
        <button class="tab active" data-mode="difference">Difference</button>
      </div>
      <label class="slider">Zoom <input id="zoom" type="range" min="1" max="20" step=".25" value="1"></label>
      <label class="slider">Pan <input id="pan" type="range" min="0" max="1000" step="1" value="0"></label>
      <div id="legend" class="legend"></div>
    </div>
    <div class="timeline-wrap">
      <div id="timelineBasisHint" class="hint"></div>
      <div id="axis" class="axis"></div>
      <div id="timeline">__EMPTY_TIMELINE_ROWS__</div>
      <div id="detail" class="detail"></div>
    </div>
  </section>

  <div class="lower">
    <section class="card section">
      <div class="section-head"><h2>Engine PMU deltas</h2>
        <span class="hint">full-card aggregate correlation</span></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Engine</th><th>Metric</th><th>Baseline</th><th>Winner</th><th>Δ</th></tr></thead>
        <tbody id="pmuRows"></tbody>
      </table></div>
    </section>
    <section class="card section">
      <div class="section-head"><h2>Heuristic TSM structure differences</h2>
        <span class="hint">signature/occurrence correlation, not stable provenance or causality</span></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Candidate sites</th><th>Correlation / engine / TSM symbol</th><th>Final position</th><th>Baseline</th><th>Winner</th><th>Δ</th></tr></thead>
        <tbody id="siteRows"></tbody>
      </table></div>
    </section>
  </div>

  <section id="diagnosticCard" class="card section diagnostics">
    <div class="section-head"><h2>Validity diagnostics</h2>
      <span id="diagnosticCount" class="hint"></span></div>
    <div id="diagnostics"></div>
  </section>
  <div class="footer">Performance uses balanced host steady-clock launch-to-completion samples. Rank-local entry cycles, TSM call records, and aggregate PMU counters provide explanation only.</div>
</main>
<div id="tooltip" class="tooltip"></div>
<script>
"use strict";
const payload = __PAYLOAD__;
const evidence = payload.evidence;
const analysis = payload.analysis;
const displayDiagnostics = payload.display_diagnostics;
const diagnosticRawCount = payload.diagnostic_raw_count;
const engines = ["CT", "NE", "RDMA", "WDMA", "TDMA"];
const engineStyle = {
  CT: ["var(--ct)", "◆"], NE: ["var(--ne)", "●"],
  RDMA: ["var(--rdma)", "▶"], WDMA: ["var(--wdma)", "◀"],
  TDMA: ["var(--tdma)", "■"]
};
const traceComparable = analysis.validity.cross_candidate_explanation &&
  analysis.validity.trace.baseline && analysis.validity.trace.winner;
const defaultTimelineMode = traceComparable
  ? "difference"
  : analysis.validity.trace.baseline
  ? "baseline"
  : analysis.validity.trace.winner
  ? "winner"
  : "baseline";
const state = { mode: defaultTimelineMode,
  zoom: 1, pan: 0, selectedTile: null, focusedCorrelation: null };
const fmt = (value, digits=1) => value == null ? "—" : Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
const signed = value => value == null ? "—" : `${value > 0 ? "+" : ""}${fmt(value, 1)}`;
const esc = value => String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const verdictLabels = {improved:"Effective improvement", regressed:"Effective regression", inconclusive:"Inconclusive", invalid:"Measurement invalid"};

function initializeSummary() {
  const verdict = document.querySelector("#verdict");
  verdict.textContent = verdictLabels[analysis.verdict];
  verdict.className = `metric-value large verdict-${analysis.verdict}`;
  const publishPerformance = analysis.validity.performance && analysis.validity.distinct_candidates;
  document.querySelector("#verdictDetail").textContent = publishPerformance
    ? `host launch→completion · materiality ${(analysis.performance.materiality_threshold*100).toFixed(2)}% · held-out ${analysis.performance.held_out_speedup_ratio.toFixed(4)}×`
    : "speed claim withheld because a required measurement gate failed";
  document.querySelector("#speedup").textContent = publishPerformance
    ? `${analysis.performance.speedup_ratio.toFixed(4)}×` : "withheld";
  document.querySelector("#speedupPct").textContent = publishPerformance
    ? `${analysis.performance.speedup_percent >= 0 ? "+" : ""}${analysis.performance.speedup_percent.toFixed(2)}% baseline / winner`
    : "raw values remain in analysis.json";
  document.querySelector("#ci").textContent = publishPerformance
    ? analysis.performance.paired_bootstrap_ci95.map(v=>v.toFixed(4)).join("–")
    : "withheld";
  const outputExact = analysis.validity.output_equivalence;
  const semantic = analysis.validity.semantic_correctness;
  document.querySelector("#outputValidation").textContent =
    !outputExact
      ? "Failed"
      : semantic === false
      ? "Expected mismatch"
      : semantic === true
      ? "Expected exact"
      : outputExact
      ? "Pair exact"
      : "Failed";
  document.querySelector("#outputValidation").className =
    `metric-value ${outputExact && semantic !== false ? "verdict-improved" : "verdict-invalid"}`;
  document.querySelector("#outputValidationDetail").textContent =
    !outputExact
      ? "production repeatability or candidate/instrumentation equivalence failed"
      : semantic === false
      ? "one or more independently supplied expected outputs did not match"
      : semantic === true
      ? `${analysis.output_validation.resource_count} writable resource(s) matched independent expected bytes`
      : "candidate/instrumentation equivalent · absolute correctness unknown";
  const traceOK = analysis.validity.trace.baseline && analysis.validity.trace.winner;
  const pmuOK = analysis.validity.pmu.baseline && analysis.validity.pmu.winner;
  document.querySelector("#evidenceHealth").textContent = `${traceOK ? "Trace ✓" : "Trace ✕"}`;
  document.querySelector("#evidenceHealthDetail").textContent = `${pmuOK ? "PMU complete" : "PMU partially excluded"}`;
}

function renderLatencyBlocks() {
  const root = document.querySelector("#latencyBlocks");
  if (!analysis.validity.performance || !analysis.validity.distinct_candidates) {
    root.innerHTML = `<div class="hint">Cross-candidate latency visualization is withheld because the measurement gates failed or both candidates are the same artifact.</div>`;
    return;
  }
  const baseline = new Map(analysis.performance.block_latencies.baseline.map(row=>[row.block,row]));
  const winner = new Map(analysis.performance.block_latencies.winner.map(row=>[row.block,row]));
  const maximum = Math.max(...[...baseline.values(),...winner.values()].map(row=>row.latency_ns));
  root.innerHTML = [...baseline.keys()].sort((a,b)=>a-b).map(block => {
    const left = baseline.get(block), right = winner.get(block);
    const badge = left.held_out ? `<span class="badge correlated">held out</span>` : `<span class="badge measured">training</span>`;
    const line = (label,row,kind) => `<div class="latency-bar-line">
      <span>${label}</span><span class="latency-track"><span class="latency-fill ${kind}" style="width:${row.latency_ns/maximum*100}%"></span></span>
      <span class="mono">${fmt(row.latency_ns/1000,1)} µs</span></div>`;
    return `<div class="latency-row"><div class="latency-label">Block ${block}<br>${badge}</div>
      <div class="latency-bars">${line("Baseline",left,"")}${line("Winner",right,"winner")}</div></div>`;
  }).join("");
}

function heatColor(value, maxAbs) {
  if (!maxAbs || value === 0) return "#9da6b7";
  const strength = .3 + .7 * Math.min(1, Math.abs(value) / maxAbs);
  return value < 0
    ? `color-mix(in srgb, var(--good) ${Math.round(strength*100)}%, white)`
    : `color-mix(in srgb, var(--bad) ${Math.round(strength*100)}%, white)`;
}

function renderTileMap() {
  const root = document.querySelector("#tileMap");
  root.innerHTML = "";
  const maxX = Math.max(...analysis.tile_metrics.map(row => row.x));
  const maxY = Math.max(...analysis.tile_metrics.map(row => row.y));
  root.style.setProperty("--tile-cols", String(maxX + 1));
  document.querySelector("#tileMapHint").textContent =
    `${maxX + 1}×${maxY + 1} reported coordinates · summary diagnostic, not verdict basis`;
  const maxAbs = Math.max(...analysis.tile_metrics.map(row => Math.abs(row.delta ?? 0)));
  [...analysis.tile_metrics].sort((a,b) => a.y-b.y || a.x-b.x).forEach(row => {
    const button = document.createElement("button");
    button.className = `tile ${state.selectedTile === row.tile ? "selected" : ""}`;
    button.style.setProperty("--heat", heatColor(row.delta ?? 0, maxAbs));
    button.style.gridColumn = String(row.x + 1);
    button.style.gridRow = String(row.y + 1);
    button.innerHTML = `<div class="tile-id mono">T${String(row.tile).padStart(2,"0")}</div>
      <div class="tile-delta mono">${row.delta == null ? "withheld" : `${signed(row.delta)} cyc`}</div>
      <div class="hint">${row.improvement_percent == null ? "candidate-local values only" : `${signed(-row.improvement_percent)}% duration`}</div>`;
    button.onclick = () => selectTile(row.tile);
    root.appendChild(button);
  });
}

function renderInsights() {
  const root = document.querySelector("#insights");
  root.innerHTML = analysis.insights.map((row,index) => `
    <div class="insight" data-index="${index}">
      <div class="insight-top"><span class="badge ${row.strength}">${esc(row.strength)}</span>
        <span class="insight-title">${esc(row.title)}</span></div>
      <div class="insight-body">${esc(row.body)}</div>
    </div>`).join("");
  root.querySelectorAll(".insight").forEach(node => node.onclick = () => focusInsight(Number(node.dataset.index)));
}

const timelineEvents = analysis.timeline_events.map(row => ({
  ...row, plotBegin: row.plot_begin, plotReturn: row.plot_return
}));
const fullMin = timelineEvents.length ? Math.min(...timelineEvents.map(row=>row.plotBegin)) : 0;
const fullMax = timelineEvents.length ? Math.max(...timelineEvents.map(row=>row.plotReturn)) : 1;

function visibleWindow() {
  const total = Math.max(1, fullMax-fullMin);
  const span = total / state.zoom;
  const start = fullMin + (total-span)*(state.pan/1000);
  return [start,start+span];
}
function eventStyle(row, start, end) {
  const span = Math.max(1e-9,end-start);
  const left = (row.plotBegin-start)/span*100;
  const right = (row.plotReturn-start)/span*100;
  const width = Math.max(.3,right-left);
  return `left:${left}%;width:${width}%;--engine:${engineStyle[row.engine][0]}`;
}
function eventsFor(tile, candidate) {
  return timelineEvents.filter(row => row.tile === tile && row.candidate === candidate);
}
function siteFor(candidate, tile, siteId) {
  return evidence.sites.find(item=>item.candidate===candidate && item.tile===tile && item.site_id===siteId);
}
function eventNode(row, overlay="") {
  const [start,end] = visibleWindow();
  if (row.plotReturn < start || row.plotBegin > end) return "";
  const site = siteFor(row.candidate,row.tile,row.site_id);
  const focused = state.focusedCorrelation === site.correlation_key ? "focused" : "";
  return `<span class="event ${overlay} ${focused}" style="${eventStyle(row,start,end)}"
    data-glyph="${engineStyle[row.engine][1]}" data-candidate="${row.candidate}" data-tile="${row.tile}"
    data-sequence="${row.sequence}" aria-label="${row.engine} site ${row.site_id}"></span>`;
}
function rowEvents(tile) {
  if (state.mode === "difference" && traceComparable) {
    return eventsFor(tile,"baseline").map(row=>eventNode(row,"baseline-overlay")).join("") +
      eventsFor(tile,"winner").map(row=>eventNode(row,"winner-overlay")).join("");
  }
  return eventsFor(tile,state.mode).map(row=>eventNode(row)).join("");
}
function renderAxis() {
  const [start,end] = visibleWindow();
  const selectedTraceValid = state.mode === "difference"
    ? traceComparable
    : analysis.validity.trace[state.mode];
  document.querySelector("#timelineBasisHint").textContent = !selectedTraceValid
    ? `${state.mode} trace is invalid; event marks are withheld, and blank rows do not mean zero TsmExecute calls.`
    : analysis.validity.clock
    ? "All 16 rows use the qualified reference-clock mapping. Marks are TsmExecute wrapper call → return; only disjoint uncertainty intervals support cross-tile order."
    : "All 16 rows use each tile's entry-local cycle zero. Marks are TsmExecute wrapper call → return; horizontal positions across different rows do not establish cross-tile order.";
  document.querySelector("#axis").innerHTML = [0,.25,.5,.75,1].map(part =>
    `<span class="axis-label" style="left:${part*100}%">${fmt(start+(end-start)*part,0)}</span>`).join("");
}
function renderTimeline() {
  renderAxis();
  const root = document.querySelector("#timeline");
  root.innerHTML = Array.from({length:16},(_,tile) => `
    <div class="timeline-row">
      <button class="row-label ${state.selectedTile === tile ? "selected" : ""}" data-tile="${tile}">T${String(tile).padStart(2,"0")}</button>
      <div class="track">${rowEvents(tile)}</div>
    </div>`).join("");
  root.querySelectorAll(".row-label").forEach(node => node.onclick = () => selectTile(Number(node.dataset.tile)));
  root.querySelectorAll(".event").forEach(bindTooltip);
  renderDetail();
}
function renderDetail() {
  const root = document.querySelector("#detail");
  if (state.selectedTile == null) { root.className = "detail"; root.innerHTML = ""; return; }
  root.className = "detail visible";
  root.innerHTML = `<div class="section-head"><h2>T${String(state.selectedTile).padStart(2,"0")} engine lanes</h2>
    <span class="hint">full-card overview remains visible</span></div>` +
    engines.map(engine => {
      let rows;
      if (state.mode === "difference") rows = [
        ...eventsFor(state.selectedTile,"baseline").filter(row=>row.engine===engine),
        ...eventsFor(state.selectedTile,"winner").filter(row=>row.engine===engine)
      ]; else rows = eventsFor(state.selectedTile,state.mode).filter(row=>row.engine===engine);
      return `<div class="engine-row"><div class="engine-label">${engineStyle[engine][1]} ${engine}</div>
        <div class="engine-track">${rows.map(row=>eventNode(row,state.mode==="difference" ? `${row.candidate}-overlay` : "")).join("")}</div></div>`;
    }).join("");
  root.querySelectorAll(".event").forEach(bindTooltip);
}
function bindTooltip(node) {
  const row = timelineEvents.find(event => event.candidate===node.dataset.candidate &&
    event.tile===Number(node.dataset.tile) && event.sequence===Number(node.dataset.sequence));
  const site = siteFor(row.candidate,row.tile,row.site_id);
  node.onmouseenter = event => {
    const tip = document.querySelector("#tooltip");
    const relation = pairwiseOrderSummary(row);
    const relationText = relation
      ? `${relation.before} before / ${relation.after} after / ${relation.overlap} unresolved`
      : "unavailable because trace or clock calibration is invalid";
    const timeDetail = analysis.validity.clock
      ? `<span class="dim">reference-clock call → return</span> ${fmt(row.plotBegin,1)} → ${fmt(row.plotReturn,1)} ±${fmt(row.uncertainty,1)}<br>`
      : `<span class="dim">entry-local call → return</span> ${fmt(row.plotBegin,1)} → ${fmt(row.plotReturn,1)} cycles; no cross-tile order<br>`;
    tip.innerHTML = `<strong>${row.candidate} · T${String(row.tile).padStart(2,"0")} · ${row.engine}</strong><br>
      <span class="dim">sequence / local site / sub-index</span> ${row.sequence} / ${row.site_id} / ${row.sub_index}<br>
      <span class="dim">correlation key</span> ${esc(site.correlation_key)}<br>
      <span class="dim">local call → return</span> ${esc(row.local_begin_cycle)} → ${esc(row.local_return_cycle)}<br>
      ${timeDetail}
      <span class="dim">raw adapter result</span> ${row.raw_result}<br>
      <span class="dim">final TSM target call</span> #${site.target_call_ordinal} ${esc(site.target_call_symbol)}<br>
      <span class="dim">diagnostic position</span> ${esc(site.position ?? "unavailable")}<br>
      <span class="dim">cross-tile pairs</span> ${relationText}<br>
      <span class="dim">semantics</span> wrapper call-return; hardware completion unknown`;
    tip.style.display = "block"; positionTooltip(event);
  };
  node.onmousemove = positionTooltip;
  node.onmouseleave = () => document.querySelector("#tooltip").style.display = "none";
}
function pairwiseOrderSummary(row) {
  if (!analysis.validity.clock || !analysis.validity.trace[row.candidate]) return null;
  const ownLow = row.plotBegin-row.uncertainty, ownHigh = row.plotReturn+row.uncertainty;
  const result = {before:0, after:0, overlap:0};
  timelineEvents.forEach(other => {
    if (other === row || other.candidate !== row.candidate || other.tile === row.tile) return;
    const otherLow = other.plotBegin-other.uncertainty;
    const otherHigh = other.plotReturn+other.uncertainty;
    if (ownHigh < otherLow) result.before++;
    else if (otherHigh < ownLow) result.after++;
    else result.overlap++;
  });
  return result;
}
function positionTooltip(event) {
  const tip = document.querySelector("#tooltip");
  const x = Math.min(window.innerWidth-tip.offsetWidth-12,event.clientX+15);
  const y = Math.min(window.innerHeight-tip.offsetHeight-12,event.clientY+15);
  tip.style.left = `${Math.max(8,x)}px`; tip.style.top = `${Math.max(8,y)}px`;
}
function selectTile(tile) {
  state.selectedTile = state.selectedTile === tile ? null : tile;
  renderTileMap(); renderTimeline();
}
function focusInsight(index) {
  const target = analysis.insights[index].target || {};
  if (target.mode) setMode(target.mode);
  if (target.tile != null) state.selectedTile = target.tile;
  state.focusedCorrelation = target.correlation_key ?? null;
  if (state.focusedCorrelation != null) {
    const row = timelineEvents.find(item=>
      siteFor(item.candidate,item.tile,item.site_id).correlation_key===state.focusedCorrelation);
    if (row) {
      state.zoom = Math.max(state.zoom,4);
      const total = Math.max(1,fullMax-fullMin), span=total/state.zoom;
      state.pan = Math.max(0,Math.min(1000,(row.plotBegin-fullMin-span/2)/(total-span)*1000 || 0));
      document.querySelector("#zoom").value=state.zoom; document.querySelector("#pan").value=state.pan;
    }
  }
  renderTileMap(); renderTimeline();
  document.querySelector(".timeline-card").scrollIntoView({behavior:"smooth",block:"start"});
}
function setMode(mode) {
  if (mode === "difference" && !traceComparable) mode = "baseline";
  if ((mode === "baseline" || mode === "winner") &&
      !analysis.validity.trace[mode]) {
    const other = mode === "baseline" ? "winner" : "baseline";
    if (analysis.validity.trace[other]) mode = other;
  }
  state.mode = mode;
  document.querySelectorAll(".tab").forEach(node=>node.classList.toggle("active",node.dataset.mode===mode));
  renderTimeline();
}

function renderLegend() {
  document.querySelector("#legend").innerHTML = engines.map(engine =>
    `<span class="legend-item"><span class="glyph" style="--engine:${engineStyle[engine][0]}">${engineStyle[engine][1]}</span>${engine}</span>`).join("");
}
function renderTables() {
  document.querySelector("#pmuRows").innerHTML = analysis.pmu_engine_deltas.map(row => {
    const deltaClass = row.delta > 0 ? "delta-pos" : row.delta < 0 ? "delta-neg" : "";
    return `<tr><td>${row.engine}</td><td>${row.metric}</td><td>${fmt(row.baseline,0)}</td>
      <td>${fmt(row.winner,0)}</td><td class="${deltaClass}">${row.valid?signed(row.delta):"excluded"}</td></tr>`;
  }).join("");
  const siteRefs = rows => {
    if (!rows.length) return "—";
    const ids = [...new Set(rows.map(row=>row.site_id))];
    if (rows.length === 16 && ids.length === 1) return `all tiles · #${ids[0]}`;
    const shown = rows.slice(0,3).map(row=>`T${String(row.tile).padStart(2,"0")}:#${row.site_id}`);
    return shown.join(", ") + (rows.length > shown.length ? ` +${rows.length-shown.length}` : "");
  };
  document.querySelector("#siteRows").innerHTML = !traceComparable
    ? `<tr><td colspan="6" class="hint">Withheld because cross-candidate identity/output-equivalence or trace gates failed</td></tr>`
    : analysis.changed_sites.length
    ? analysis.changed_sites.map(row => `<tr><td class="mono">b:${siteRefs(row.baseline_sites)} → w:${siteRefs(row.winner_sites)}</td>
      <td><span class="mono">${esc(row.correlation_key)}</span> · ${esc(row.baseline_engines.join("/") || "—")}→${esc(row.winner_engines.join("/") || "—")} · ${esc(row.target_call_symbol)}</td>
      <td class="mono">#${row.target_call_ordinal} · ${esc(row.position ?? "unavailable")}<br><span class="hint">${esc(row.tile_count_changes.map(item=>`T${String(item.tile).padStart(2,"0")} ${item.delta>0?"+":""}${item.delta}`).join(", ") || "no tile movement")}</span></td>
      <td>${row.baseline}</td><td>${row.winner}</td>
      <td class="${row.delta>0?"delta-pos":"delta-neg"}">${signed(row.delta)}</td></tr>`).join("")
    : `<tr><td colspan="6" class="hint">No site count changes</td></tr>`;
}
function renderDiagnostics() {
  const compactTiles = tiles => {
    if (!tiles.length) return "no tile scope";
    if (tiles.length === 16 && tiles.every((tile,index)=>tile===index)) return "T00–T15";
    const shown = tiles.slice(0,8).map(tile=>`T${String(tile).padStart(2,"0")}`);
    return shown.join(", ") + (tiles.length>shown.length?` +${tiles.length-shown.length}`:"");
  };
  document.querySelector("#diagnosticCount").textContent =
    diagnosticRawCount === displayDiagnostics.length
      ? `${diagnosticRawCount} finding(s)`
      : `${diagnosticRawCount} raw finding(s) · ${displayDiagnostics.length} displayed row(s)`;
  document.querySelector("#diagnostics").innerHTML = displayDiagnostics.length
    ? displayDiagnostics.map(row => {
      const badge = `<span class="badge ${row.severity==="error"?"unresolved":"correlated"}">${esc(row.severity)}</span>`;
      const identity = `<span class="mono">${esc(row.code)}</span> · ${esc(row.candidate ?? "global")}`;
      if (!row.grouped)
        return `<div class="diag">${badge} ${identity}${row.tile!=null?` T${String(row.tile).padStart(2,"0")}`:""} · ${esc(row.message)}</div>`;
      const scopes = row.scopes.slice(0,8).map(esc);
      const scopeText = scopes.join(", ") +
        (row.scopes.length>scopes.length?` +${row.scopes.length-scopes.length}`:"");
      const findingLabel = row.grouped_count === 1 ? "counter finding" : "counter findings";
      return `<details class="diag diag-group"><summary>${badge} ${identity} · ${row.grouped_count} ${findingLabel} across ${compactTiles(row.tiles)} · ${esc(row.message)}</summary>
        <div class="diag-group-body">Affected scopes (${row.scopes.length}): ${scopeText || "unavailable"}. Full per-counter details remain in <a href="analysis.json">Analysis JSON</a>.</div></details>`;
    }).join("")
    : `<div class="diag">All structural and semantic evidence gates passed.</div>`;
}

document.querySelectorAll(".tab").forEach(node => {
  if (node.dataset.mode === "difference" && !traceComparable)
    node.disabled = true;
  if ((node.dataset.mode === "baseline" || node.dataset.mode === "winner") &&
      !analysis.validity.trace[node.dataset.mode])
    node.disabled = true;
  node.classList.toggle("active", node.dataset.mode === state.mode);
  node.onclick=()=>setMode(node.dataset.mode);
});
document.querySelector("#zoom").oninput = event => { state.zoom=Number(event.target.value); renderTimeline(); };
document.querySelector("#pan").oninput = event => { state.pan=Number(event.target.value); renderTimeline(); };
initializeSummary(); renderLatencyBlocks(); renderLegend(); renderTileMap(); renderInsights(); renderTimeline(); renderTables(); renderDiagnostics();
</script>
</body>
</html>
"""


def render_report(
    evidence: object, analysis: Mapping[str, Any] | None = None
) -> str:
    """Render a self-contained offline HTML report."""

    valid_evidence = validate_evidence(evidence)
    if analysis is None:
        analysis = analyze_evidence(valid_evidence)
    # The browser receives site metadata, normalized analysis coordinates, and
    # bounded display diagnostics. Raw u64 cycles and per-counter diagnostics
    # stay in evidence.json/analysis.json; JSON.parse must never round the
    # cycles before timeline math.
    report_analysis = dict(analysis)
    raw_diagnostics = tuple(
        _mapping(row, f"analysis.diagnostics[{index}]")
        for index, row in enumerate(
            _sequence(analysis.get("diagnostics"), "analysis.diagnostics")
        )
    )
    report_analysis.pop("diagnostics", None)
    payload = json.dumps(
        {
            "evidence": {"sites": valid_evidence["sites"]},
            "analysis": report_analysis,
            "display_diagnostics": _summarize_diagnostics_for_report(
                raw_diagnostics
            ),
            "diagnostic_raw_count": len(raw_diagnostics),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # A profile can contain source strings.  Keep them from terminating the
    # inline script while retaining a fully offline single-file report.
    payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    empty_rows = "".join(
        (
            '<div class="timeline-row" data-static-tile="'
            f'{tile}"><span class="row-label">T{tile:02d}</span>'
            '<div class="track"></div></div>'
        )
        for tile in TILES
    )
    return (
        _REPORT_TEMPLATE.replace(
            "__RUN_ID__", html.escape(str(valid_evidence["run_id"]))
        )
        .replace(
            "__TARGET_PROFILE__",
            html.escape(str(valid_evidence["identity"]["target_profile"])),
        )
        .replace("__EMPTY_TIMELINE_ROWS__", empty_rows)
        .replace("__PAYLOAD__", payload)
    )


def _atomic_write(path: pathlib.Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def generate_report(
    evidence_path: os.PathLike[str] | str,
    output_directory: os.PathLike[str] | str,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate ``index.html`` and the machine-readable ``analysis.json``."""

    evidence = load_evidence(evidence_path)
    analysis = analyze_evidence(evidence)
    output = pathlib.Path(output_directory)
    evidence_output = output / "evidence.json"
    html_path = output / "index.html"
    analysis_path = output / "analysis.json"
    if pathlib.Path(evidence_path).resolve() != evidence_output.resolve():
        _atomic_write(
            evidence_output,
            json.dumps(
                evidence, indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n",
        )
    _atomic_write(
        analysis_path,
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
    )
    _atomic_write(html_path, render_report(evidence, analysis))
    return html_path, analysis_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate 16-tile wafer profile evidence and generate an offline "
            "HTML report."
        )
    )
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument(
        "--output-directory",
        required=True,
        type=pathlib.Path,
        help="run report directory that will receive index.html and analysis.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        html_path, analysis_path = generate_report(
            args.evidence, args.output_directory
        )
    except EvidenceError as error:
        print(f"profile evidence rejected: {error}", file=os.sys.stderr)
        return 2
    except OSError as error:
        print(f"cannot publish profile report: {error}", file=os.sys.stderr)
        return 3
    print(f"profile_report: {html_path}")
    print(f"profile_analysis: {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
