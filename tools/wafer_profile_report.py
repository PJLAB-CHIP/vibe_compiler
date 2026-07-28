#!/usr/bin/env python3
"""Validate final-artifact profiler evidence and publish an offline report."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import pathlib
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_NAME = "wafer.profile.evidence"
SCHEMA_VERSION = 4
COMPANION_SCHEMA_VERSION = 2
ANALYSIS_SCHEMA_NAME = "wafer.profile.analysis"
ANALYSIS_SCHEMA_VERSION = 2
TILES = tuple(range(16))
NCC_ENGINES = ("CT", "NE", "RDMA", "WDMA", "TDMA")
ENGINES = NCC_ENGINES + ("DIRECT_DTE",)
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
WORKERS = (0, 1, 2)
UINT64_MAX = (1 << 64) - 1
TRACE_COMPLETE_FLAGS = 71
TRACE_COMPLETE_STATE = 2
MAX_COMPLETION_RESOLUTION_FRACTION = 0.0025


class EvidenceError(ValueError):
    """The evidence does not conform to the final-artifact contract."""


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
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        _fail(path, "; ".join(details))


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "expected a non-empty string")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if (
        len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        _fail(path, "expected sha256: followed by 64 lowercase hex digits")
    return digest


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


def _tile_rows(value: object, path: str) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        _mapping(row, f"{path}[{index}]")
        for index, row in enumerate(_sequence(value, path))
    )
    ids = [
        _integer(row.get("tile"), f"{path}[{index}].tile")
        for index, row in enumerate(rows)
    ]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    missing = sorted(set(TILES) - set(ids))
    extra = sorted(set(ids) - set(TILES))
    if len(rows) != 16 or duplicates or missing or extra:
        _fail(
            path,
            "must contain all-and-only tiles 0..15 exactly once "
            f"(duplicates={duplicates}, missing={missing}, extra={extra})",
        )
    return rows


def _validate_launch(value: object, path: str) -> Mapping[str, Any]:
    launch = _mapping(value, path)
    kind = _string(launch.get("kind"), f"{path}.kind")
    if kind == "kernel":
        _exact_keys(launch, {"kind", "form", "entry_abi", "phases"}, path)
        form = _string(launch["form"], f"{path}.form")
        entry_abi = _string(launch["entry_abi"], f"{path}.entry_abi")
        phases = tuple(
            _string(item, f"{path}.phases[{index}]")
            for index, item in enumerate(
                _sequence(launch["phases"], f"{path}.phases")
            )
        )
        contracts = {
            "per-rank": ("rank-local-pointer-block-v1", ("main",)),
            "grid": ("rank-major-pointer-table-v1", ("main",)),
            "cluster": (
                "rank-major-pointer-table-v1",
                ("prepare", "main"),
            ),
        }
        if contracts.get(form) != (entry_abi, phases):
            _fail(path, "kernel form, entry ABI and phases are incompatible")
        return launch
    if kind == "model":
        _exact_keys(launch, {"kind", "entry_abi", "phases"}, path)
        entry_abi = _string(launch["entry_abi"], f"{path}.entry_abi")
        phases = tuple(
            _string(item, f"{path}.phases[{index}]")
            for index, item in enumerate(
                _sequence(launch["phases"], f"{path}.phases")
            )
        )
        if (entry_abi, phases) != ("tx81-model-bootparam-v1", ("main",)):
            _fail(path, "model entry ABI and phases are incompatible")
        return launch
    _fail(f"{path}.kind", "must be 'kernel' or 'model'")


def _validate_counter(value: object, path: str) -> None:
    counter = _mapping(value, path)
    _exact_keys(counter, {"start", "end", "recovery", "stable", "enabled"}, path)
    for key in ("start", "end", "recovery"):
        _integer(counter[key], f"{path}.{key}", minimum=0, maximum=UINT64_MAX)
    _boolean(counter["stable"], f"{path}.stable")
    _boolean(counter["enabled"], f"{path}.enabled")


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
            "site_correlation_basis",
        },
        "identity",
    )
    _sha256(
        identity["production_manifest_sha256"],
        "identity.production_manifest_sha256",
    )
    version = _integer(
        identity["profile_companion_schema_version"],
        "identity.profile_companion_schema_version",
    )
    if version != COMPANION_SCHEMA_VERSION:
        _fail(
            "identity.profile_companion_schema_version",
            f"must be {COMPANION_SCHEMA_VERSION}",
        )
    _string(identity["target_profile"], "identity.target_profile")
    _validate_launch(identity["launch"], "identity.launch")
    if _integer(identity["execution_ranks"], "identity.execution_ranks") != 16:
        _fail("identity.execution_ranks", "must be 16")
    basis = _string(
        identity["site_correlation_basis"],
        "identity.site_correlation_basis",
    )
    if basis != "heuristic-target-call-signature-occurrence-v1":
        _fail("identity.site_correlation_basis", "unknown correlation basis")


def _validate_topology(evidence: Mapping[str, Any]) -> None:
    coordinates: set[tuple[int, int]] = set()
    for index, row in enumerate(_tile_rows(evidence["topology"], "topology")):
        path = f"topology[{index}]"
        _exact_keys(row, {"tile", "x", "y"}, path)
        coordinate = (
            _integer(row["x"], f"{path}.x", minimum=0),
            _integer(row["y"], f"{path}.y", minimum=0),
        )
        if coordinate in coordinates:
            _fail(path, f"duplicate physical coordinate {coordinate}")
        coordinates.add(coordinate)


def _validate_measurement(evidence: Mapping[str, Any]) -> None:
    measurement = _mapping(evidence["measurement"], "measurement")
    _exact_keys(measurement, {"samples"}, "measurement")
    samples = tuple(
        _mapping(row, f"measurement.samples[{index}]")
        for index, row in enumerate(
            _sequence(measurement["samples"], "measurement.samples")
        )
    )
    if len(samples) != 10:
        _fail("measurement.samples", "must contain exactly 10 final executions")
    indices: list[int] = []
    ids: set[str] = set()
    for index, row in enumerate(samples):
        path = f"measurement.samples[{index}]"
        _exact_keys(
            row,
            {
                "sample_id",
                "sample_index",
                "host_elapsed_ns",
                "completion_observation_resolution_ns",
            },
            path,
        )
        sample_id = _string(row["sample_id"], f"{path}.sample_id")
        if sample_id in ids:
            _fail(f"{path}.sample_id", f"duplicate sample id {sample_id!r}")
        ids.add(sample_id)
        indices.append(
            _integer(row["sample_index"], f"{path}.sample_index", minimum=0)
        )
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
    if sorted(indices) != list(range(10)):
        _fail("measurement.samples", "duplicate sample index or incomplete 0..9 domain")


def _validate_output(evidence: Mapping[str, Any]) -> None:
    output = _mapping(evidence["output_validation"], "output_validation")
    _exact_keys(output, {"mode", "resources"}, "output_validation")
    mode = _string(output["mode"], "output_validation.mode")
    allowed_modes = {
        "external-expected",
        "mixed",
        "same-session-production",
    }
    if mode not in allowed_modes:
        _fail("output_validation.mode", f"must be one of {sorted(allowed_modes)}")
    rows = tuple(
        _mapping(row, f"output_validation.resources[{index}]")
        for index, row in enumerate(
            _sequence(output["resources"], "output_validation.resources")
        )
    )
    if not rows:
        _fail("output_validation.resources", "must not be empty")
    keys: set[tuple[int, str, int]] = set()
    comparisons: list[str | None] = []
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
                "external_expected_comparison",
                "production_repeats_exact",
                "diagnostic_captures_exact",
            },
            path,
        )
        key = (
            _integer(row["logical_rank"], f"{path}.logical_rank", minimum=-1),
            _string(row["role"], f"{path}.role"),
            _integer(row["role_index"], f"{path}.role_index", minimum=0),
        )
        if key in keys:
            _fail(path, f"duplicate semantic output key {key}")
        keys.add(key)
        _integer(row["bytes"], f"{path}.bytes", minimum=1)
        _sha256(row["reference_sha256"], f"{path}.reference_sha256")
        comparison = row["external_expected_comparison"]
        if comparison not in {None, "exact", "relaxed-f16"}:
            _fail(
                f"{path}.external_expected_comparison",
                "must be null, 'exact', or 'relaxed-f16'",
            )
        comparisons.append(comparison)
        _boolean(
            row["production_repeats_exact"],
            f"{path}.production_repeats_exact",
        )
        _boolean(
            row["diagnostic_captures_exact"],
            f"{path}.diagnostic_captures_exact",
        )
    expected_mode = (
        "external-expected"
        if all(value is not None for value in comparisons)
        else "same-session-production"
        if all(value is None for value in comparisons)
        else "mixed"
    )
    if mode != expected_mode:
        _fail("output_validation.mode", f"expected {expected_mode!r}")


def _validate_sites(
    evidence: Mapping[str, Any],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    sites: dict[tuple[int, int], Mapping[str, Any]] = {}
    for index, row_value in enumerate(_sequence(evidence["sites"], "sites")):
        row = _mapping(row_value, f"sites[{index}]")
        path = f"sites[{index}]"
        required = {
            "tile",
            "site_id",
            "correlation_key",
            "engine",
            "target_call_ordinal",
            "target_call_symbol",
        }
        if not required.issubset(row) or not set(row).issubset(
            required | {"position"}
        ):
            _exact_keys(row, required | ({"position"} if "position" in row else set()), path)
        tile = _integer(row["tile"], f"{path}.tile", minimum=0, maximum=15)
        site_id = _integer(row["site_id"], f"{path}.site_id", minimum=0)
        key = (tile, site_id)
        if key in sites:
            _fail(path, f"duplicate tile/site key {key}")
        _string(row["correlation_key"], f"{path}.correlation_key")
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
        sites[key] = row
    return sites


def _validate_experiment(
    evidence: Mapping[str, Any],
    sites: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    experiment = _mapping(evidence["experiment"], "experiment")
    _exact_keys(
        experiment, {"artifact", "clock", "summary", "trace", "pmu"}, "experiment"
    )
    artifact = _mapping(experiment["artifact"], "experiment.artifact")
    _exact_keys(
        artifact,
        {"digest", "target_profile", "launch", "execution_ranks"},
        "experiment.artifact",
    )
    _sha256(artifact["digest"], "experiment.artifact.digest")
    _string(artifact["target_profile"], "experiment.artifact.target_profile")
    _validate_launch(artifact["launch"], "experiment.artifact.launch")
    if _integer(
        artifact["execution_ranks"], "experiment.artifact.execution_ranks"
    ) != 16:
        _fail("experiment.artifact.execution_ranks", "must be 16")

    for index, row in enumerate(
        _tile_rows(experiment["clock"], "experiment.clock")
    ):
        path = f"experiment.clock[{index}]"
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
        if _number(row["slope"], f"{path}.slope") <= 0:
            _fail(f"{path}.slope", "must be positive")
        _number(row["offset"], f"{path}.offset")
        if _number(row["uncertainty"], f"{path}.uncertainty") < 0:
            _fail(f"{path}.uncertainty", "must not be negative")
        _integer(row["round_trips"], f"{path}.round_trips", minimum=0)
        _boolean(row["valid"], f"{path}.valid")
        _boolean(row["monotonic"], f"{path}.monotonic")

    summary = _mapping(experiment["summary"], "experiment.summary")
    _exact_keys(summary, {"tiles"}, "experiment.summary")
    for index, row in enumerate(
        _tile_rows(summary["tiles"], "experiment.summary.tiles")
    ):
        path = f"experiment.summary.tiles[{index}]"
        _exact_keys(row, {"tile", "entry_begin", "entry_end"}, path)
        _integer(row["entry_begin"], f"{path}.entry_begin", minimum=0)
        _integer(row["entry_end"], f"{path}.entry_end", minimum=0)

    trace = _mapping(experiment["trace"], "experiment.trace")
    _exact_keys(trace, {"complete", "tiles"}, "experiment.trace")
    _boolean(trace["complete"], "experiment.trace.complete")
    for index, row in enumerate(
        _tile_rows(trace["tiles"], "experiment.trace.tiles")
    ):
        path = f"experiment.trace.tiles[{index}]"
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
        tile = int(row["tile"])
        for key in (
            "entry_begin_cycle",
            "entry_end_cycle",
            "capacity",
            "count",
            "preflight_count",
            "next_sequence",
            "dropped_event_count",
            "record_flags",
            "trace_state",
        ):
            _integer(row[key], f"{path}.{key}", minimum=0, maximum=UINT64_MAX)
        _boolean(row["overflow"], f"{path}.overflow")
        for event_index, event_value in enumerate(
            _sequence(row["events"], f"{path}.events")
        ):
            event = _mapping(event_value, f"{path}.events[{event_index}]")
            event_path = f"{path}.events[{event_index}]"
            _exact_keys(
                event,
                {
                    "sequence",
                    "site_id",
                    "sub_index",
                    "engine",
                    "observed_begin_cycle",
                    "observed_end_cycle",
                    "counter_delta",
                    "activity_valid",
                    "dte_role",
                    "dte_counter_valid",
                },
                event_path,
            )
            _integer(event["sequence"], f"{event_path}.sequence", minimum=0)
            site_id = _integer(
                event["site_id"], f"{event_path}.site_id", minimum=0
            )
            _integer(event["sub_index"], f"{event_path}.sub_index", minimum=0)
            engine = _string(event["engine"], f"{event_path}.engine")
            if engine not in ENGINES:
                _fail(f"{event_path}.engine", f"must be one of {list(ENGINES)}")
            site = sites.get((tile, site_id))
            if site is None:
                _fail(f"{event_path}.site_id", "does not map to a final site")
            if site["engine"] != engine:
                _fail(f"{event_path}.engine", "conflicts with the site engine")
            _integer(
                event["observed_begin_cycle"],
                f"{event_path}.observed_begin_cycle",
                minimum=0,
            )
            _integer(
                event["observed_end_cycle"],
                f"{event_path}.observed_end_cycle",
                minimum=0,
            )
            raw = _integer(
                event["counter_delta"],
                f"{event_path}.counter_delta",
                minimum=0,
            )
            activity_valid = _boolean(
                event["activity_valid"], f"{event_path}.activity_valid"
            )
            if engine == "DIRECT_DTE":
                if event["dte_role"] not in {"send", "receive"}:
                    _fail(f"{event_path}.dte_role", "must be send or receive")
                if not activity_valid:
                    _fail(
                        f"{event_path}.activity_valid",
                        "must be true for a completed Direct-DTE wait window",
                    )
                valid = _boolean(
                    event["dte_counter_valid"],
                    f"{event_path}.dte_counter_valid",
                )
                if not valid and raw != 0:
                    _fail(
                        f"{event_path}.counter_delta",
                        "must be zero when the DTE counter is unusable",
                    )
            elif event["dte_role"] is not None or event["dte_counter_valid"] is not None:
                _fail(
                    event_path,
                    "NCC activity must carry null DTE role and counter validity",
                )

    pmu = _mapping(experiment["pmu"], "experiment.pmu")
    _exact_keys(pmu, {"tiles"}, "experiment.pmu")
    for index, row in enumerate(_tile_rows(pmu["tiles"], "experiment.pmu.tiles")):
        path = f"experiment.pmu.tiles[{index}]"
        _exact_keys(row, {"tile", "aggregates", "workers"}, path)
        aggregates = _mapping(row["aggregates"], f"{path}.aggregates")
        _exact_keys(aggregates, set(AGGREGATE_COUNTERS), f"{path}.aggregates")
        for name in AGGREGATE_COUNTERS:
            _validate_counter(aggregates[name], f"{path}.aggregates.{name}")
        workers = tuple(
            _mapping(item, f"{path}.workers[{worker_index}]")
            for worker_index, item in enumerate(
                _sequence(row["workers"], f"{path}.workers")
            )
        )
        if sorted(int(item.get("worker", -1)) for item in workers) != list(WORKERS):
            _fail(f"{path}.workers", "must contain workers 0, 1 and 2")
        for worker_index, worker in enumerate(workers):
            worker_path = f"{path}.workers[{worker_index}]"
            _exact_keys(worker, {"worker", "engines"}, worker_path)
            engines = tuple(
                _mapping(item, f"{worker_path}.engines[{engine_index}]")
                for engine_index, item in enumerate(
                    _sequence(worker["engines"], f"{worker_path}.engines")
                )
            )
            if {item.get("engine") for item in engines} != set(NCC_ENGINES):
                _fail(f"{worker_path}.engines", "must contain the five NCC engines")
            for engine_index, engine in enumerate(engines):
                engine_path = f"{worker_path}.engines[{engine_index}]"
                _exact_keys(engine, {"engine", "instructions", "blocking"}, engine_path)
                _validate_counter(engine["instructions"], f"{engine_path}.instructions")
                _validate_counter(engine["blocking"], f"{engine_path}.blocking")


def validate_evidence(value: object) -> Mapping[str, Any]:
    evidence = _mapping(value, "$")
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
            "experiment",
        },
        "$",
    )
    if evidence["schema"] != SCHEMA_NAME:
        _fail("schema", f"must be {SCHEMA_NAME!r}")
    if evidence["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"must be {SCHEMA_VERSION}")
    _string(evidence["run_id"], "run_id")
    _validate_identity(evidence)
    _validate_topology(evidence)
    _validate_measurement(evidence)
    _validate_output(evidence)
    sites = _validate_sites(evidence)
    validity = _mapping(evidence["validity"], "validity")
    _exact_keys(
        validity,
        {"environment", "package_companion", "measurement_basis"},
        "validity",
    )
    for key in validity:
        _boolean(validity[key], f"validity.{key}")
    _validate_experiment(evidence, sites)
    identity = evidence["identity"]
    artifact = evidence["experiment"]["artifact"]
    for key, left, right in (
        (
            "digest",
            identity["production_manifest_sha256"],
            artifact["digest"],
        ),
        ("target_profile", identity["target_profile"], artifact["target_profile"]),
        ("launch", identity["launch"], artifact["launch"]),
        (
            "execution_ranks",
            identity["execution_ranks"],
            artifact["execution_ranks"],
        ),
    ):
        if left != right:
            _fail(f"experiment.artifact.{key}", "conflicts with identity")
    return evidence


def load_evidence(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    try:
        value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read evidence: {error}") from error
    return validate_evidence(value)


def _by_tile(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["tile"]): row for row in rows}


def _counter_value(counter: Mapping[str, Any]) -> tuple[int | None, str | None]:
    if not counter["enabled"]:
        return None, "counter collection was disabled"
    if not counter["stable"]:
        return None, "split counter read was not stable"
    start, end, recovery = (
        int(counter["start"]),
        int(counter["end"]),
        int(counter["recovery"]),
    )
    if end < start:
        return None, "counter end precedes start"
    if recovery < end:
        return None, "counter recovery precedes end"
    return end - start, None


def analyze_evidence(value: object) -> dict[str, Any]:
    evidence = validate_evidence(value)
    diagnostics: list[dict[str, Any]] = []

    def diagnose(
        severity: str,
        code: str,
        message: str,
        *,
        tile: int | None = None,
        scope: str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "severity": severity,
            "code": code,
            "message": message,
        }
        if tile is not None:
            row["tile"] = tile
        if scope is not None:
            row["scope"] = scope
        diagnostics.append(row)

    source_validity = evidence["validity"]
    output = evidence["output_validation"]
    resources = tuple(output["resources"])
    production_exact = all(row["production_repeats_exact"] for row in resources)
    captures_exact = all(row["diagnostic_captures_exact"] for row in resources)
    output_equivalence = production_exact and captures_exact
    if not output_equivalence:
        diagnose(
            "error",
            "output_equivalence_failed",
            "a final execution or diagnostic capture differs from the first "
            "same-session production result",
        )
    comparisons = [row["external_expected_comparison"] for row in resources]
    all_expected = all(value is not None for value in comparisons)
    any_expected = any(value is not None for value in comparisons)
    semantic_correctness: bool | None = True if all_expected else None
    if all_expected and all(value == "exact" for value in comparisons):
        correctness_status = "expected-exact"
    elif all_expected:
        correctness_status = "expected-relaxed-f16"
    elif any_expected:
        correctness_status = "partially-expected"
    else:
        correctness_status = "repeat-exact"

    samples = sorted(
        evidence["measurement"]["samples"], key=lambda row: row["sample_index"]
    )
    elapsed = [int(row["host_elapsed_ns"]) for row in samples]
    resolution = [
        int(row["completion_observation_resolution_ns"]) for row in samples
    ]
    resolution_fractions = [
        value / duration for value, duration in zip(resolution, elapsed)
    ]
    high_resolution = max(resolution_fractions) <= MAX_COMPLETION_RESOLUTION_FRACTION
    if not high_resolution:
        diagnose(
            "warning",
            "completion_resolution_too_coarse",
            "host latency remains usable, but completion observation resolution "
            "is too coarse for the high-resolution label",
        )
    qualified = bool(
        source_validity["environment"]
        and source_validity["package_companion"]
        and source_validity["measurement_basis"]
        and output_equivalence
    )
    for key in ("environment", "package_companion", "measurement_basis"):
        if not source_validity[key]:
            diagnose("error", f"{key}_failed", f"{key} qualification failed")

    experiment = evidence["experiment"]
    topology = _by_tile(evidence["topology"])
    summary = _by_tile(experiment["summary"]["tiles"])
    trace = _by_tile(experiment["trace"]["tiles"])
    pmu = _by_tile(experiment["pmu"]["tiles"])
    clocks = _by_tile(experiment["clock"])
    sites = {(int(row["tile"]), int(row["site_id"])): row for row in evidence["sites"]}
    timeline_events: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    trace_all = bool(experiment["trace"]["complete"])
    pmu_all = True
    summary_all = True

    for tile in TILES:
        summary_row = summary[tile]
        summary_begin = int(summary_row["entry_begin"])
        summary_end = int(summary_row["entry_end"])
        if summary_end < summary_begin:
            summary_cycles = None
            summary_all = False
            diagnose(
                "warning",
                "summary_entry_span_unusable",
                "summary entry end precedes begin",
                tile=tile,
            )
        else:
            summary_cycles = summary_end - summary_begin

        trace_row = trace[tile]
        trace_begin = int(trace_row["entry_begin_cycle"])
        trace_end = int(trace_row["entry_end_cycle"])
        trace_axis = trace_end - trace_begin if trace_end >= trace_begin else None
        protocol_ok = bool(
            experiment["trace"]["complete"]
            and trace_axis is not None
            and not trace_row["overflow"]
            and int(trace_row["dropped_event_count"]) == 0
            and int(trace_row["record_flags"]) == TRACE_COMPLETE_FLAGS
            and int(trace_row["trace_state"]) == TRACE_COMPLETE_STATE
            and int(trace_row["count"]) == len(trace_row["events"])
            and int(trace_row["count"]) <= int(trace_row["capacity"])
            and int(trace_row["preflight_count"]) == int(trace_row["count"])
            and int(trace_row["next_sequence"]) == int(trace_row["count"])
            and [int(event["sequence"]) for event in trace_row["events"]]
            == list(range(len(trace_row["events"])))
        )
        if not protocol_ok:
            trace_all = False
            diagnose(
                "warning",
                "trace_protocol_unusable",
                "trace terminal fields, count preflight, or entry span are inconsistent",
                tile=tile,
            )

        usable_events: list[dict[str, Any]] = []
        dte_source_event_count = sum(
            event["engine"] == "DIRECT_DTE" for event in trace_row["events"]
        )
        dte_windows_valid = protocol_ok
        for event in trace_row["events"]:
            begin = int(event["observed_begin_cycle"])
            end = int(event["observed_end_cycle"])
            engine = str(event["engine"])
            site = sites.get((tile, int(event["site_id"])))
            window_ok = bool(
                protocol_ok
                and event["activity_valid"]
                and end >= begin
                and begin >= trace_begin
                and end <= trace_end
                and site is not None
                and site["engine"] == engine
            )
            if engine in NCC_ENGINES and int(event["counter_delta"]) == 0:
                window_ok = False
            if not window_ok:
                if engine == "DIRECT_DTE":
                    dte_windows_valid = False
                if event["activity_valid"]:
                    trace_all = False
                    diagnose(
                        "warning",
                        (
                            "direct_dte_wait_window_unusable"
                            if engine == "DIRECT_DTE"
                            else "ncc_activity_window_unusable"
                        ),
                        "activity observation is reversed, outside the trace "
                        "entry span, or inconsistent with its site",
                        tile=tile,
                        scope=f"site:{event['site_id']}:{event['sub_index']}",
                    )
                continue
            axis = int(trace_axis)
            offset_begin = begin - trace_begin
            offset_end = end - trace_begin
            base = {
                "tile": tile,
                "sequence": int(event["sequence"]),
                "site_id": int(event["site_id"]),
                "sub_index": int(event["sub_index"]),
                "engine": engine,
                "trace_entry_offset_begin": offset_begin,
                "trace_entry_offset_end": offset_end,
                "activity_window_cycles": end - begin,
                "plot_begin_fraction": offset_begin / axis if axis else 0.0,
                "plot_end_fraction": offset_end / axis if axis else 0.0,
                "dte_role": event["dte_role"],
                "ncc_busy_cycles": (
                    int(event["counter_delta"]) if engine in NCC_ENGINES else None
                ),
                "direct_dte_wait_cycles": (
                    end - begin if engine == "DIRECT_DTE" else None
                ),
                "direct_dte_raw_pmu_activity": (
                    int(event["counter_delta"])
                    if engine == "DIRECT_DTE" and event["dte_counter_valid"]
                    else None
                ),
                "direct_dte_raw_pmu_valid": (
                    bool(event["dte_counter_valid"])
                    if engine == "DIRECT_DTE"
                    else None
                ),
            }
            timeline_events.append(base)
            usable_events.append(base)
            if engine == "DIRECT_DTE" and not event["dte_counter_valid"]:
                diagnose(
                    "warning",
                    "direct_dte_raw_pmu_unusable",
                    "Direct-DTE wait window is usable, but its uncalibrated raw "
                    "PMU activity is unavailable",
                    tile=tile,
                    scope=f"site:{event['site_id']}:{event['sub_index']}",
                )

        engine_rows: list[dict[str, Any]] = []
        aggregate = pmu[tile]["aggregates"]
        for engine in NCC_ENGINES:
            busy, reason = _counter_value(aggregate[engine.lower()])
            valid = busy is not None
            pmu_all = pmu_all and valid
            if not valid:
                diagnose(
                    "warning",
                    "ncc_busy_counter_unusable",
                    reason or "NCC busy counter is unavailable",
                    tile=tile,
                    scope=engine,
                )
            engine_rows.append(
                {
                    "engine": engine,
                    "measurement_kind": "ncc-hardware-busy-cycles",
                    "busy_cycles": busy,
                    "busy_cycles_valid": valid,
                    "activity_window_count": sum(
                        row["engine"] == engine for row in usable_events
                    ),
                }
            )

        dte_events = [
            row for row in usable_events if row["engine"] == "DIRECT_DTE"
        ]
        raw_values = [
            row["direct_dte_raw_pmu_activity"] for row in dte_events
        ]
        raw_valid = bool(
            dte_windows_valid
            and dte_events
            and all(value is not None for value in raw_values)
        )
        engine_rows.append(
            {
                "engine": "DIRECT_DTE",
                "measurement_kind": "direct-dte-wait-completion-windows",
                "wait_window_cycles": (
                    sum(
                        int(row["direct_dte_wait_cycles"])
                        for row in dte_events
                    )
                    if dte_windows_valid
                    else None
                ),
                "wait_windows_valid": dte_windows_valid,
                "wait_window_count": dte_source_event_count,
                "raw_pmu_activity": (
                    sum(int(value) for value in raw_values) if raw_valid else None
                ),
                "raw_pmu_activity_valid": raw_valid,
                "activity_window_count": len(dte_events),
            }
        )
        topology_row = topology[tile]
        tile_rows.append(
            {
                "tile": tile,
                "x": int(topology_row["x"]),
                "y": int(topology_row["y"]),
                "summary_entry_cycles": summary_cycles,
                "trace_entry_cycles": trace_axis,
                "timeline_axis": "trace-capture-entry-span",
                "engines": engine_rows,
            }
        )

    clock_alignment = all(
        row["valid"] and row["monotonic"] for row in clocks.values()
    )
    validity = {
        "identity": True,
        "output_equivalence": output_equivalence,
        "semantic_correctness": semantic_correctness,
        "environment": bool(source_validity["environment"]),
        "package_companion": bool(source_validity["package_companion"]),
        "measurement_basis": bool(source_validity["measurement_basis"]),
        "completion_resolution": high_resolution,
        "summary": summary_all,
        "trace": trace_all,
        "pmu": pmu_all,
        "clock_alignment": clock_alignment,
    }
    return {
        "schema": ANALYSIS_SCHEMA_NAME,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_id": evidence["run_id"],
        "valid": qualified,
        "validity": validity,
        "final_artifact": {
            "artifact_digest": experiment["artifact"]["digest"],
            "target_profile": experiment["artifact"]["target_profile"],
            "latency": {
                "sample_count": len(elapsed),
                "samples_ns": elapsed,
                "median_ns": statistics.median(elapsed),
                "mean_ns": statistics.fmean(elapsed),
                "minimum_ns": min(elapsed),
                "maximum_ns": max(elapsed),
                "maximum_completion_observation_resolution_ns": max(resolution),
                "maximum_completion_observation_fraction": max(
                    resolution_fractions
                ),
                "high_resolution": high_resolution,
                "qualified": qualified,
            },
            "output": {
                "resource_count": len(resources),
                "production_repeats_exact": production_exact,
                "diagnostic_captures_exact": captures_exact,
                "independent_expected_coverage": (
                    "complete" if all_expected else "partial" if any_expected else "none"
                ),
                "correctness_status": correctness_status,
            },
            "tiles": tile_rows,
            "timeline_events": timeline_events,
        },
        "diagnostics": diagnostics,
        "method": {
            "overall_latency": "host steady-clock first submit through all-rank trusted completion",
            "summary_entry_span": "summary capture per-tile entry span",
            "timeline_axis": "trace capture per-tile entry span",
            "ncc_engine_time": "per-tile hardware PMU busy cycles; engines may overlap",
            "direct_dte_time": "sum of captured wait/completion windows",
            "direct_dte_raw_pmu": "uncalibrated activity only; never interpreted as elapsed time",
        },
    }


_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>最终产物板卡 Profile · __RUN_ID__</title>
<style>
:root{--ink:#17202a;--muted:#66717e;--line:#d8dee7;--paper:#f5f7fa;--card:#fff;--accent:#2457c5;--ok:#16745a;--warn:#a05a00;--ct:#2457c5;--ne:#7047a8;--rdma:#087c78;--wdma:#3b7d44;--tdma:#b05b17;--dte:#48515c}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,system-ui,sans-serif}.page{max-width:1240px;margin:auto;padding:28px 22px 42px}.mono{font-family:Consolas,monospace}h1{margin:4px 0 18px;font-size:29px}.sub,.note{color:var(--muted);font-size:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:17px;margin-bottom:14px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;padding:0;overflow:hidden}.kpi{padding:18px;border-left:1px solid var(--line)}.kpi:first-child{border:0}.label{font-size:12px;color:var(--muted)}.value{font-size:24px;font-weight:760;margin-top:7px}.hero{font-size:32px;color:var(--accent)}.ok{color:var(--ok)}.warn{color:var(--warn)}.samples{display:grid;gap:6px}.sample{display:grid;grid-template-columns:42px 1fr 90px;gap:8px;align-items:center;font-size:11px}.track{position:relative;height:18px;background:#edf0f4}.fill{height:100%;background:var(--accent)}.layout{display:grid;grid-template-columns:360px 1fr;gap:14px}.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.tile{border:1px solid var(--line);background:white;border-radius:8px;padding:8px;text-align:left;cursor:pointer}.tile.selected{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}.lane{display:grid;grid-template-columns:82px 1fr;gap:8px;align-items:center;margin:6px 0}.event{position:absolute;height:14px;top:2px;min-width:2px;border-radius:3px}.event-CT{background:var(--ct)}.event-NE{background:var(--ne)}.event-RDMA{background:var(--rdma)}.event-WDMA{background:var(--wdma)}.event-TDMA{background:var(--tdma)}.event-DIRECT_DTE{background:var(--dte)}.diag{font-size:12px;color:var(--muted);padding:7px 0;border-bottom:1px solid var(--line)}a{color:var(--accent);text-decoration:none}.links{float:right;display:flex;gap:10px}@media(max-width:900px){.kpis{grid-template-columns:1fr 1fr}.layout{grid-template-columns:1fr}}@media(max-width:560px){.page{padding:16px 10px}.kpis{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main class="page">
<div class="links"><a href="evidence.json">evidence.json</a><a href="analysis.json">analysis.json</a></div>
<div class="sub">Wafer · 16 Tile · Hardware Activity</div><h1>最终编译产物板卡 Profile</h1>
<section class="card kpis"><div class="kpi"><div class="label">最终产物整体耗时 · 中位数</div><div id="median" class="value hero"></div><div id="range" class="note"></div></div><div class="kpi"><div class="label">板卡实测</div><div id="samples" class="value"></div><div class="note">同一最终产物逐次串行执行</div></div><div class="kpi"><div class="label">结果校验</div><div id="correctness" class="value"></div><div id="correctnessNote" class="note"></div></div><div class="kpi"><div class="label">测量资格</div><div id="qualification" class="value"></div><div id="qualificationNote" class="note"></div></div></section>
<section class="card"><h2>每次最终产物耗时</h2><div id="latencies" class="samples"></div></section>
<div class="layout"><section class="card"><h2>Tile 总览</h2><div class="note">summary span 与 trace timeline 轴分别展示</div><div id="tiles" class="tiles"></div></section><section class="card"><h2 id="tileTitle">Engine 活动 Timeline</h2><div id="spans" class="note"></div><table><thead><tr><th>Engine</th><th>真实测量</th><th>活动段</th><th>Busy/Wait 状态</th></tr></thead><tbody id="engines"></tbody></table><h3>Engine 活动 Timeline</h3><div id="timeline"></div><div class="note">NCC 为硬件 busy cycles；DIRECT_DTE 为 wait/completion 窗口。DTE raw PMU 仅表示未校准活动量，显示“—”即该 raw 计数不可用；Busy/Wait 状态不替它背书。横轴只使用本 tile 的 trace capture entry span。</div></section></div>
<details class="card"><summary>诊断与测量说明 · <span id="diagnosticCount"></span></summary><div id="diagnostics"></div></details>
<div class="note">本目录只有三个产物：index.html、analysis.json、evidence.json；目录和文件均为 0777。</div>
</main><script>"use strict";const analysis=__ANALYSIS__,final=analysis.final_artifact,engines=["CT","NE","RDMA","WDMA","TDMA","DIRECT_DTE"];const num=v=>v==null?"—":Number(v).toLocaleString();const ms=v=>(Number(v)/1e6).toFixed(3)+" ms";const latency=final.latency;document.querySelector("#median").textContent=ms(latency.median_ns);document.querySelector("#range").textContent=`范围 ${ms(latency.minimum_ns)}–${ms(latency.maximum_ns)}`;document.querySelector("#samples").textContent=latency.sample_count+" 次";const output=final.output;document.querySelector("#correctness").textContent=output.correctness_status==="expected-exact"?"全部正确":output.correctness_status==="expected-relaxed-f16"?"全部正确（Relaxed F16）":output.correctness_status==="partially-expected"?"重复一致（部分 expected）":"重复一致";document.querySelector("#correctness").className="value "+(output.production_repeats_exact&&output.diagnostic_captures_exact?"ok":"warn");document.querySelector("#correctnessNote").textContent=output.resource_count+" 个输出资源";document.querySelector("#qualification").textContent=latency.qualified?"测量有效":"测量失败";document.querySelector("#qualification").className="value "+(latency.qualified?"ok":"warn");document.querySelector("#qualificationNote").textContent=latency.high_resolution?"高分辨率 completion 观察":"completion 观察较粗，耗时仍保留";const lo=latency.minimum_ns,hi=latency.maximum_ns,span=Math.max(1,hi-lo);document.querySelector("#latencies").innerHTML=latency.samples_ns.map((v,i)=>`<div class="sample"><span>#${String(i+1).padStart(2,"0")}</span><div class="track"><div class="fill" style="width:${8+92*(v-lo)/span}%"></div></div><span>${ms(v)}</span></div>`).join("");let selected=0;function renderTiles(){document.querySelector("#tiles").innerHTML=[...final.tiles].sort((a,b)=>a.y-b.y||a.x-b.x).map(t=>`<button class="tile ${t.tile===selected?"selected":""}" data-tile="${t.tile}"><b>T${String(t.tile).padStart(2,"0")}</b><div class="note">summary ${num(t.summary_entry_cycles)}</div><div class="note">trace ${num(t.trace_entry_cycles)}</div></button>`).join("");document.querySelectorAll(".tile").forEach(n=>n.onclick=()=>{selected=Number(n.dataset.tile);renderTiles();renderDetail()})}function renderDetail(){const tile=final.tiles.find(t=>t.tile===selected),events=final.timeline_events.filter(e=>e.tile===selected);document.querySelector("#tileTitle").textContent=`T${String(selected).padStart(2,"0")} · 六类 Engine`;document.querySelector("#spans").textContent=`summary entry ${num(tile.summary_entry_cycles)} cycles · trace timeline axis ${num(tile.trace_entry_cycles)} cycles`;document.querySelector("#engines").innerHTML=tile.engines.map(e=>{if(e.engine==="DIRECT_DTE")return `<tr><td>${e.engine}</td><td>${num(e.wait_window_cycles)} wait cycles · raw ${num(e.raw_pmu_activity)}</td><td>${e.wait_window_count}</td><td>${e.wait_windows_valid?"有效":"不可用"}</td></tr>`;return `<tr><td>${e.engine}</td><td>${num(e.busy_cycles)} busy cycles</td><td>${e.activity_window_count}</td><td>${e.busy_cycles_valid?"有效":"不可用"}</td></tr>`}).join("");document.querySelector("#timeline").innerHTML=engines.map(engine=>{const marks=events.filter(e=>e.engine===engine).map(e=>`<span class="event event-${engine}" style="left:${100*e.plot_begin_fraction}%;width:${Math.max(.25,100*(e.plot_end_fraction-e.plot_begin_fraction))}%" title="${engine} · ${e.activity_window_cycles} cycle observation"></span>`).join("");return `<div class="lane"><b>${engine}</b><div class="track">${marks}</div></div>`}).join("")}document.querySelector("#diagnosticCount").textContent=analysis.diagnostics.length+" 项";document.querySelector("#diagnostics").innerHTML=analysis.diagnostics.length?analysis.diagnostics.map(d=>`<div class="diag">${d.code}${d.tile==null?"":" · T"+String(d.tile).padStart(2,"0")} · ${d.message}</div>`).join(""):'<div class="diag ok">全部结构与语义检查通过。</div>';renderTiles();renderDetail();</script></body></html>"""


def render_report(
    evidence: object, analysis: Mapping[str, Any] | None = None
) -> str:
    valid = validate_evidence(evidence)
    result = analyze_evidence(valid) if analysis is None else analysis
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return (
        _REPORT_TEMPLATE.replace(
            "__RUN_ID__", html.escape(str(valid["run_id"]))
        ).replace("__ANALYSIS__", payload)
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
        os.chmod(path, 0o777)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def generate_report(
    evidence_path: os.PathLike[str] | str,
    output_directory: os.PathLike[str] | str,
) -> tuple[pathlib.Path, pathlib.Path]:
    evidence = load_evidence(evidence_path)
    analysis = analyze_evidence(evidence)
    output = pathlib.Path(output_directory)
    evidence_output = output / "evidence.json"
    analysis_path = output / "analysis.json"
    html_path = output / "index.html"
    if pathlib.Path(evidence_path).resolve() != evidence_output.resolve():
        _atomic_write(
            evidence_output,
            json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
        )
    _atomic_write(
        analysis_path,
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_write(html_path, render_report(evidence, analysis))
    os.chmod(output, 0o777)
    return html_path, analysis_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate final-artifact profile evidence and generate HTML."
    )
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument(
        "--output-directory", required=True, type=pathlib.Path
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
