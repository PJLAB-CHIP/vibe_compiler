#!/usr/bin/env python3
"""Validate final-artifact profiler evidence and publish an offline report."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import pathlib
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_NAME = "wafer.profile.evidence"
SCHEMA_VERSION = 5
COMPANION_SCHEMA_VERSION = 3
ANALYSIS_SCHEMA_NAME = "wafer.profile.analysis"
ANALYSIS_SCHEMA_VERSION = 3
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
UINT32_MAX = (1 << 32) - 1
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
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, "expected a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        _fail(path, "expected a finite number")
    if not math.isfinite(number):
        _fail(path, "expected a finite number")
    return number


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


def _validate_counter(
    value: object, path: str, *, maximum: int = UINT64_MAX
) -> None:
    counter = _mapping(value, path)
    _exact_keys(counter, {"start", "end", "recovery", "stable", "enabled"}, path)
    for key in ("start", "end", "recovery"):
        _integer(counter[key], f"{path}.{key}", minimum=0, maximum=maximum)
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
    if len(samples) != 1:
        _fail("measurement.samples", "must contain exactly one primary execution")
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
        if sample_id != "primary":
            _fail(f"{path}.sample_id", "must be 'primary'")
        if _integer(row["sample_index"], f"{path}.sample_index", minimum=0) != 0:
            _fail(f"{path}.sample_index", "must be 0")
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
                "production_execution_validated",
                "diagnostic_captures_match_primary",
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
        _integer(
            row["bytes"],
            f"{path}.bytes",
            minimum=1,
            maximum=UINT64_MAX,
        )
        _sha256(row["reference_sha256"], f"{path}.reference_sha256")
        comparison = row["external_expected_comparison"]
        if comparison is not None and (
            not isinstance(comparison, str)
            or comparison not in ("exact", "relaxed-f16")
        ):
            _fail(
                f"{path}.external_expected_comparison",
                "must be null, 'exact', or 'relaxed-f16'",
            )
        comparisons.append(comparison)
        _boolean(
            row["production_execution_validated"],
            f"{path}.production_execution_validated",
        )
        _boolean(
            row["diagnostic_captures_match_primary"],
            f"{path}.diagnostic_captures_match_primary",
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
    _exact_keys(experiment, {"artifact", "clock", "trace", "pmu"}, "experiment")
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
        for key in ("entry_begin_cycle", "entry_end_cycle"):
            _integer(row[key], f"{path}.{key}", minimum=0, maximum=UINT64_MAX)
        for key in ("capacity", "count"):
            _integer(row[key], f"{path}.{key}", minimum=0)
        for key in ("preflight_count", "next_sequence"):
            _integer(row[key], f"{path}.{key}", minimum=0, maximum=UINT64_MAX)
        for key in ("dropped_event_count", "record_flags"):
            _integer(row[key], f"{path}.{key}", minimum=0, maximum=UINT32_MAX)
        _integer(row["trace_state"], f"{path}.trace_state", minimum=0, maximum=4)
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
                maximum=UINT64_MAX,
            )
            _integer(
                event["observed_end_cycle"],
                f"{event_path}.observed_end_cycle",
                minimum=0,
                maximum=UINT64_MAX,
            )
            raw = _integer(
                event["counter_delta"],
                f"{event_path}.counter_delta",
                minimum=0,
                maximum=UINT64_MAX,
            )
            activity_valid = _boolean(
                event["activity_valid"], f"{event_path}.activity_valid"
            )
            if engine == "DIRECT_DTE":
                dte_role = event["dte_role"]
                if not isinstance(dte_role, str) or dte_role not in (
                    "send",
                    "receive",
                ):
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
        if len(workers) != len(WORKERS):
            _fail(f"{path}.workers", "must contain exactly workers 0, 1 and 2")
        worker_ids: list[int] = []
        for worker_index, worker in enumerate(workers):
            worker_path = f"{path}.workers[{worker_index}]"
            _exact_keys(worker, {"worker", "engines"}, worker_path)
            worker_ids.append(
                _integer(
                    worker["worker"],
                    f"{worker_path}.worker",
                    minimum=0,
                    maximum=2,
                )
            )
            engines = tuple(
                _mapping(item, f"{worker_path}.engines[{engine_index}]")
                for engine_index, item in enumerate(
                    _sequence(worker["engines"], f"{worker_path}.engines")
                )
            )
            if len(engines) != len(NCC_ENGINES):
                _fail(
                    f"{worker_path}.engines",
                    "must contain each of the five NCC engines exactly once",
                )
            engine_names: list[str] = []
            for engine_index, engine in enumerate(engines):
                engine_path = f"{worker_path}.engines[{engine_index}]"
                _exact_keys(engine, {"engine", "instructions", "blocking"}, engine_path)
                engine_name = _string(engine["engine"], f"{engine_path}.engine")
                if engine_name not in NCC_ENGINES:
                    _fail(
                        f"{engine_path}.engine",
                        f"must be one of {list(NCC_ENGINES)}",
                    )
                engine_names.append(engine_name)
                _validate_counter(
                    engine["instructions"],
                    f"{engine_path}.instructions",
                    maximum=UINT32_MAX,
                )
                _validate_counter(
                    engine["blocking"],
                    f"{engine_path}.blocking",
                    maximum=UINT32_MAX,
                )
            if set(engine_names) != set(NCC_ENGINES):
                _fail(
                    f"{worker_path}.engines",
                    "must contain each of the five NCC engines exactly once",
                )
        if sorted(worker_ids) != list(WORKERS):
            _fail(f"{path}.workers", "must contain exactly workers 0, 1 and 2")


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
    production_validated = all(
        row["production_execution_validated"] for row in resources
    )
    captures_match_primary = all(
        row["diagnostic_captures_match_primary"] for row in resources
    )
    output_equivalence = production_validated and captures_match_primary
    if not output_equivalence:
        diagnose(
            "error",
            "output_equivalence_failed",
            "the primary production execution failed validation or a "
            "diagnostic capture differs from its primary output",
        )
    comparisons = [row["external_expected_comparison"] for row in resources]
    all_expected = all(value is not None for value in comparisons)
    any_expected = any(value is not None for value in comparisons)
    if not production_validated:
        semantic_correctness: bool | None = False
        correctness_status = "production-validation-failed"
    elif all_expected and all(value == "exact" for value in comparisons):
        semantic_correctness = True
        correctness_status = "expected-exact"
    elif all_expected:
        semantic_correctness = True
        correctness_status = "expected-relaxed-f16"
    elif any_expected:
        semantic_correctness = None
        correctness_status = "partially-expected"
    else:
        semantic_correctness = None
        correctness_status = "primary-validated"

    sample = evidence["measurement"]["samples"][0]
    elapsed = int(sample["host_elapsed_ns"])
    resolution = int(sample["completion_observation_resolution_ns"])
    resolution_fraction = resolution / elapsed
    high_resolution = resolution_fraction <= MAX_COMPLETION_RESOLUTION_FRACTION
    if not high_resolution:
        diagnose(
            "warning",
            "completion_resolution_too_coarse",
            "the primary host duration remains usable, but completion "
            "observation resolution is too coarse for the high-resolution label",
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
    trace = _by_tile(experiment["trace"]["tiles"])
    pmu = _by_tile(experiment["pmu"]["tiles"])
    clocks = _by_tile(experiment["clock"])
    sites = {(int(row["tile"]), int(row["site_id"])): row for row in evidence["sites"]}
    timeline_events: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    trace_all = bool(experiment["trace"]["complete"])
    pmu_all = True

    for tile in TILES:
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
        trace_status = (
            "Bounded"
            if protocol_ok
            else "Invalid"
            if trace_axis is None
            else "Incomplete"
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
            is_direct_dte = engine == "DIRECT_DTE"
            base = {
                "tile": tile,
                "sequence": int(event["sequence"]),
                "site_id": int(event["site_id"]),
                "sub_index": int(event["sub_index"]),
                "engine": engine,
                "correlation_key": site["correlation_key"],
                "target_call_ordinal": int(site["target_call_ordinal"]),
                "target_call_symbol": site["target_call_symbol"],
                "position": site.get("position"),
                "trace_entry_offset_begin": offset_begin,
                "trace_entry_offset_end": offset_end,
                "activity_window_cycles": end - begin,
                "duration_status": "Measured" if is_direct_dte else "Bounded",
                "counter_status": (
                    "Sampled"
                    if is_direct_dte and event["dte_counter_valid"]
                    else "Unavailable"
                    if is_direct_dte
                    else "Sampled"
                ),
                "timeline_scope": "tile-local",
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
                    "busy_cycles_status": "Measured" if valid else "Unavailable",
                    "activity_window_count": sum(
                        row["engine"] == engine for row in usable_events
                    ),
                    "activity_window_status": (
                        "Bounded"
                        if any(row["engine"] == engine for row in usable_events)
                        else "Incomplete"
                        if not protocol_ok
                        else "Unavailable"
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
        wait_status = (
            "Measured"
            if dte_windows_valid
            else "Invalid"
            if trace_axis is None
            else "Incomplete"
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
                "wait_window_status": wait_status,
                "wait_window_count": dte_source_event_count,
                "raw_pmu_activity": (
                    sum(int(value) for value in raw_values) if raw_valid else None
                ),
                "raw_pmu_activity_valid": raw_valid,
                "raw_pmu_activity_status": (
                    "Sampled" if raw_valid else "Unavailable"
                ),
                "activity_window_count": len(dte_events),
            }
        )
        topology_row = topology[tile]
        clock_row = clocks[tile]
        clock_valid = bool(clock_row["valid"] and clock_row["monotonic"])
        tile_rows.append(
            {
                "tile": tile,
                "x": int(topology_row["x"]),
                "y": int(topology_row["y"]),
                "trace_entry_cycles": trace_axis,
                "timeline_axis": "trace-capture-entry-span",
                "timeline_scope": "tile-local",
                "trace_status": trace_status,
                "clock_status": "Measured" if clock_valid else "Unavailable",
                "clock_mapping": {
                    "slope": float(clock_row["slope"]),
                    "offset": float(clock_row["offset"]),
                    "uncertainty": float(clock_row["uncertainty"]),
                    "round_trips": int(clock_row["round_trips"]),
                    "valid": clock_valid,
                },
                "engines": engine_rows,
            }
        )

    clock_alignment = all(
        row["valid"] and row["monotonic"] for row in clocks.values()
    )
    event_counts = Counter(
        (int(row["tile"]), int(row["site_id"])) for row in timeline_events
    )
    site_rows: list[dict[str, Any]] = []
    for (tile, site_id), site in sorted(sites.items()):
        event_count = event_counts[(tile, site_id)]
        site_rows.append(
            {
                "tile": tile,
                "site_id": site_id,
                "engine": site["engine"],
                "correlation_key": site["correlation_key"],
                "target_call_ordinal": int(site["target_call_ordinal"]),
                "target_call_symbol": site["target_call_symbol"],
                "position": site.get("position"),
                "event_count": event_count,
                "observation_status": (
                    "Measured"
                    if event_count and site["engine"] == "DIRECT_DTE"
                    else "Bounded"
                    if event_count
                    else "Unavailable"
                ),
            }
        )
    direct_dte_events = [
        row for row in timeline_events if row["engine"] == "DIRECT_DTE"
    ]
    validity = {
        "identity": True,
        "output_equivalence": output_equivalence,
        "semantic_correctness": semantic_correctness,
        "environment": bool(source_validity["environment"]),
        "package_companion": bool(source_validity["package_companion"]),
        "measurement_basis": bool(source_validity["measurement_basis"]),
        "completion_resolution": high_resolution,
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
            "duration": {
                "sample_id": "primary",
                "sample_index": 0,
                "host_elapsed_ns": elapsed,
                "completion_observation_resolution_ns": resolution,
                "completion_observation_fraction": resolution_fraction,
                "high_resolution": high_resolution,
                "qualified": qualified,
                "status": "Measured" if qualified else "Invalid",
            },
            "output": {
                "resource_count": len(resources),
                "production_execution_validated": production_validated,
                "diagnostic_captures_match_primary": captures_match_primary,
                "independent_expected_coverage": (
                    "complete" if all_expected else "partial" if any_expected else "none"
                ),
                "correctness_status": correctness_status,
            },
            "tiles": tile_rows,
            "sites": site_rows,
            "timeline_events": timeline_events,
            "communication": {
                "direct_dte_event_count": len(direct_dte_events),
                "send_count": sum(
                    row["dte_role"] == "send" for row in direct_dte_events
                ),
                "receive_count": sum(
                    row["dte_role"] == "receive" for row in direct_dte_events
                ),
                "tiles_with_activity": sorted(
                    {int(row["tile"]) for row in direct_dte_events}
                ),
                "timeline_scope": "tile-local",
                "cross_tile_order_available": False,
            },
        },
        "diagnostics": diagnostics,
        "method": {
            "primary_duration": "one primary production execution: host steady-clock first submit through all-rank trusted completion",
            "timeline_axis": "trace capture per-tile entry span",
            "timeline_scope": "tile-local only; no cross-tile order is inferred",
            "ncc_engine_time": "Measured per-tile aggregate hardware PMU busy cycles; engines may overlap",
            "ncc_activity_window": "Bounded observation only; never presented as an exact execution interval",
            "direct_dte_time": "Measured wait/completion windows, separate from raw PMU activity",
            "direct_dte_raw_pmu": "Sampled uncalibrated activity only; never interpreted as elapsed time",
        },
    }




_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wafer Profiler · __RUN_ID__</title>
<style>
:root{
  color-scheme:light;--bg:#f4f6f8;--panel:#fff;--soft:#f8fafb;
  --line:#dce2e8;--line-strong:#c5ced8;--ink:#17212b;--muted:#657180;
  --accent:#175cd3;--accent-soft:#eaf2ff;--good:#087a55;--warn:#a45b06;
  --bad:#b42318;--ct:#2563eb;--ne:#7c3aed;--rdma:#07857e;
  --wdma:#368147;--tdma:#c4600c;--dte:#475467
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,select{font:inherit}
button{color:inherit}
.shell{display:flex;align-items:flex-start;min-height:100vh}
.sidebar{position:sticky;top:0;flex:0 0 268px;width:268px;height:100vh;overflow:auto;background:#fbfcfd;border-right:1px solid var(--line);padding:18px 14px}
.brand{display:flex;align-items:center;gap:10px;padding:2px 6px 17px}
.brand-mark{display:grid;place-items:center;width:34px;height:34px;border-radius:9px;background:#17212b;color:#fff;font-weight:800}
.brand strong{display:block;font-size:14px}.brand small{color:var(--muted)}
.nav{display:grid;gap:3px}
.nav button,.tree button{width:100%;border:0;background:transparent;text-align:left;border-radius:7px;cursor:pointer}
.nav button{padding:9px 10px;color:#465364}
.nav button:hover,.nav button.active{background:var(--accent-soft);color:var(--accent);font-weight:650}
.side-heading{margin:20px 8px 7px;color:var(--muted);font-size:10px;font-weight:750;letter-spacing:.1em;text-transform:uppercase}
.tree{font-size:11px}.tree details{border-left:1px solid var(--line);margin-left:7px;padding-left:8px}.tree summary{cursor:pointer;padding:4px 2px;font-weight:700}
.tree button{padding:3px 5px;color:var(--muted)}.tree button:hover{color:var(--accent);background:var(--accent-soft)}
.main{flex:1 1 auto;min-width:0;padding:20px 24px 34px}
.topbar{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:16px}
.eyebrow{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:750}
h1{font-size:23px;margin:3px 0 2px;line-height:1.2}.run-id{color:var(--muted);font:11px ui-monospace,SFMono-Regular,Consolas,monospace}
.artifact-links{display:flex;gap:7px}.artifact-links a{padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--accent);text-decoration:none;font-size:11px}
.view{display:none}.view.active{display:block}
.section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin:0 0 10px}
.section-head h2{font-size:17px;margin:0}.section-head p{margin:2px 0 0;color:var(--muted);font-size:11px}
.grid{display:flex;flex-wrap:wrap;display:grid;gap:10px}.grid>*{min-width:0}.kpis{grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:10px}.kpis>.card{flex:1 1 210px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.metric-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-weight:700}
.metric-value{font-size:24px;font-weight:760;margin:5px 0 2px;white-space:nowrap}.metric-value.primary{font-size:31px;color:var(--accent)}
.metric-note{font-size:11px;color:var(--muted)}
.status{display:inline-flex;align-items:center;gap:5px;border:1px solid currentColor;border-radius:999px;padding:2px 7px;font-size:10px;font-weight:750;white-space:nowrap}
.status::before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor}
.status-Measured{color:#087a55;background:#edf9f4}.status-Sampled{color:#175cd3;background:#eef4ff}.status-Bounded{color:#6e4fc4;background:#f5f1ff}
.status-Unavailable{color:#667085;background:#f2f4f7}.status-Incomplete{color:#a45b06;background:#fff7e8}.status-Invalid{color:#b42318;background:#fff1f0}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.split{display:flex;align-items:flex-start;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:10px}.split>*:first-child{flex:1 1 auto;min-width:0}.split>*:last-child{flex:0 0 300px;margin-left:10px}
.tile-grid{display:flex;flex-wrap:wrap;display:grid;grid-template-columns:repeat(4,minmax(68px,1fr));gap:5px}
.tile-card{flex:0 0 calc(25% - 6px);margin:3px;border:1px solid var(--line);border-radius:7px;padding:8px;background:var(--soft);cursor:pointer;text-align:left}
.tile-card:hover{border-color:var(--accent)}.tile-card b{display:flex;justify-content:space-between}.tile-card small{display:block;color:var(--muted);margin-top:4px}
.method-list{display:grid;gap:8px}.method-row{padding-bottom:7px;border-bottom:1px solid var(--line)}.method-row:last-child{border:0}
.method-row b{display:block;font-size:11px}.method-row span{color:var(--muted);font-size:11px}
.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:9px 10px;background:var(--panel);border:1px solid var(--line);border-radius:9px;margin-bottom:9px}
.toolbar label{display:flex;align-items:center;gap:5px;color:var(--muted);font-size:11px}.toolbar select,.toolbar input[type=search]{border:1px solid var(--line-strong);border-radius:6px;background:#fff;padding:5px 7px;color:var(--ink)}
.toolbar input[type=search]{min-width:250px}.toolbar .spacer{flex:1}.toolbar button{border:1px solid var(--line-strong);border-radius:6px;background:#fff;padding:5px 8px;cursor:pointer}
.engine-toggle{display:inline-flex!important;padding:3px 6px;border:1px solid var(--line);border-radius:5px;background:var(--soft);color:var(--ink)!important}
.timeline-shell{overflow:auto;border:1px solid var(--line);border-radius:9px;background:var(--panel)}
.timeline-canvas{min-width:720px;padding:12px 14px 16px;transition:min-width .15s ease}
.ruler{display:grid;grid-template-columns:92px 1fr;gap:8px;align-items:end;margin-bottom:5px}
.ruler-track{position:relative;height:24px;border-bottom:1px solid var(--line-strong)}
.tick{position:absolute;bottom:-1px;height:6px;border-left:1px solid var(--line-strong)}
.tick span{position:absolute;bottom:8px;transform:translateX(-50%);color:var(--muted);font:9px ui-monospace,SFMono-Regular,monospace}
.lane{display:grid;grid-template-columns:92px 1fr;gap:8px;align-items:center;min-height:33px;border-top:1px solid #edf0f3}
.lane-label{display:flex;align-items:center;justify-content:space-between;font-size:11px}.lane-track{position:relative;height:20px;background:#f6f8fa;border-left:1px solid var(--line);border-right:1px solid var(--line)}
.lane-gridline{position:absolute;inset:0 auto 0 0;border-left:1px solid #e7ebef;pointer-events:none}
.event{position:absolute;top:3px;height:14px;min-width:3px;border:0;border-radius:3px;cursor:pointer;opacity:.92;box-shadow:0 0 0 1px rgba(0,0,0,.07)}
.event:hover,.event.selected{outline:2px solid #17212b;outline-offset:1px;z-index:2}
.event-CT{background:var(--ct)}.event-NE{background:var(--ne)}.event-RDMA{background:var(--rdma)}.event-WDMA{background:var(--wdma)}.event-TDMA{background:var(--tdma)}.event-DIRECT_DTE{background:var(--dte)}
.detail{min-height:210px}.detail h3{font-size:13px;margin:0 0 10px}.kv{display:grid;grid-template-columns:110px 1fr;gap:5px 8px;font-size:11px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
table{width:100%;border-collapse:collapse;background:var(--panel);font-size:11px}th{position:sticky;top:0;background:#f6f8fa;color:#5a6675;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.05em}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.interactive-row{cursor:pointer}.interactive-row:hover,.interactive-row:focus{background:var(--accent-soft);outline:none}.interactive-row:focus-visible{box-shadow:inset 3px 0 var(--accent)}
.table-wrap{overflow:auto;max-height:calc(100vh - 180px);border:1px solid var(--line);border-radius:9px}.empty{padding:28px;text-align:center;color:var(--muted)}
.diag{display:grid;grid-template-columns:74px 220px 70px 1fr;gap:8px;padding:8px;border-bottom:1px solid var(--line);font-size:11px}.diag:last-child{border:0}.severity-error{color:var(--bad)}.severity-warning{color:var(--warn)}
.raw-tabs{display:flex;gap:5px;margin:12px 0 7px}.raw-tabs button{border:1px solid var(--line);border-radius:6px;background:#fff;padding:5px 8px;cursor:pointer}.raw-tabs button.active{background:var(--accent);border-color:var(--accent);color:#fff}
pre{margin:0;max-height:520px;overflow:auto;background:#111827;color:#dbe7f5;border-radius:8px;padding:12px;font:10px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
.notice{border-left:3px solid var(--accent);background:var(--accent-soft);padding:8px 10px;color:#344054;font-size:11px;margin-bottom:9px}
@supports(display:grid){.split>*:last-child{margin-left:0}.tile-card{margin:0}}
@media(max-width:1050px){.sidebar{flex-basis:220px;width:220px}.kpis{grid-template-columns:repeat(2,1fr)}.split{grid-template-columns:1fr;flex-direction:column}.split>*:last-child{width:100%;margin:10px 0 0}}
@media(max-width:720px){.shell{display:block}.sidebar{position:static;width:auto;height:auto}.tree{display:none}.main{padding:14px}.nav{grid-template-columns:repeat(2,1fr)}.kpis{grid-template-columns:1fr}.tile-grid{grid-template-columns:repeat(2,1fr)}.tile-card{flex-basis:calc(50% - 6px)}.topbar{display:block}.artifact-links{margin-top:10px}}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand"><div class="brand-mark">W</div><div><strong>Wafer Profiler</strong><small>Final artifact · Card 0</small></div></div>
    <nav class="nav" aria-label="Profile views">
      <button class="active" data-view="overview">Overview</button>
      <button data-view="timeline">Trace Timeline</button>
      <button data-view="engines">Tile / Engine</button>
      <button data-view="sites">Program / Sites</button>
      <button data-view="communication">Communication / DTE</button>
      <button data-view="diagnostics">Diagnostics / Raw</button>
    </nav>
    <div class="side-heading">Resource tree</div>
    <div id="resourceTree" class="tree" aria-label="Card tile engine tree"></div>
  </aside>
  <main class="main">
    <header class="topbar">
      <div><div class="eyebrow">Production artifact / primary execution</div><h1>最终编译产物板卡 Profile</h1><div class="run-id">run __RUN_ID__</div></div>
      <div class="artifact-links"><a href="evidence.json">Evidence</a><a href="analysis.json">Analysis</a></div>
    </header>

    <section id="view-overview" class="view active" data-view-panel="overview">
      <div class="section-head"><div><h2>Overview</h2><p>单次 primary production execution；不是多轮统计，也没有 winner/baseline。</p></div></div>
      <div id="statusLegend" class="legend"></div>
      <div class="grid kpis">
        <div class="card"><div class="metric-label">Primary duration</div><div id="primaryDuration" class="metric-value primary"></div><div id="primaryResolution" class="metric-note"></div></div>
        <div class="card"><div class="metric-label">Measurement status</div><div id="measurementStatus" class="metric-value"></div><div class="metric-note">Host first-submit → all-rank trusted completion</div></div>
        <div class="card"><div class="metric-label">Output validation</div><div id="outputStatus" class="metric-value"></div><div id="outputNote" class="metric-note"></div></div>
        <div class="card"><div class="metric-label">Trace coverage</div><div id="traceCoverage" class="metric-value"></div><div id="traceNote" class="metric-note"></div></div>
      </div>
      <div class="split">
        <div class="card"><div class="section-head"><div><h2>Card 0 · Tile map</h2><p>点击 tile 进入本地 engine timeline。</p></div></div><div id="overviewTiles" class="tile-grid"></div></div>
        <div class="card"><div class="section-head"><div><h2>Measurement contract</h2></div></div><div id="methodList" class="method-list"></div></div>
      </div>
      <div class="card" style="margin-top:10px"><div class="section-head"><div><h2>Engine summary</h2><p>跨 tile 求和是 derived activity volume，不是全卡 elapsed duration；Direct-DTE wait 单独列示。</p></div></div><div class="table-wrap" style="max-height:none"><table><thead><tr><th>Engine</th><th>Metric</th><th class="num">Derived sum</th><th class="num">Tiles</th><th class="num">Windows</th><th>Status</th></tr></thead><tbody id="overviewEngineRows"></tbody></table></div></div>
    </section>

    <section id="view-timeline" class="view" data-view-panel="timeline">
      <div class="section-head"><div><h2>Trace Timeline</h2><p>六条 engine lane；横轴始终是所选 tile 的 trace entry span。</p></div></div>
      <div class="notice">Tile-local clock domain。报告不推断跨 tile 的先后关系；NCC 矩形仅是 <b>Bounded observation window</b>，不是精确执行起止。</div>
      <div class="toolbar">
        <label>Tile <select id="timelineTile"></select></label>
        <span id="engineFilters"></span>
        <span class="spacer"></span>
        <label>Zoom <input id="timelineZoom" type="range" min="1" max="8" value="1" step="0.25"></label>
        <button id="timelineFit" type="button">Fit</button>
      </div>
      <div class="split">
        <div class="timeline-shell"><div id="timelineCanvas" class="timeline-canvas"><div id="timelineRuler"></div><div id="timelineLanes"></div></div></div>
        <aside id="eventDetail" class="card detail" aria-live="polite"></aside>
      </div>
    </section>

    <section id="view-engines" class="view" data-view-panel="engines">
      <div class="section-head"><div><h2>Tile / Engine</h2><p>Aggregate PMU busy 是 Measured；activity window 只作 Bounded 证据。</p></div></div>
      <div class="toolbar"><label>Tile <select id="engineTile"></select></label><span id="engineTileMeta"></span></div>
      <div class="table-wrap"><table><thead><tr><th>Engine</th><th>Metric</th><th class="num">Cycles / raw</th><th class="num">Windows</th><th>Status</th><th>Interpretation</th></tr></thead><tbody id="engineRows"></tbody></table></div>
    </section>

    <section id="view-sites" class="view" data-view-panel="sites">
      <div class="section-head"><div><h2>Program / Sites</h2><p>从 trace event 下钻到 compiler-published correlation identity。</p></div></div>
      <div class="toolbar"><label>Search <input id="siteSearch" type="search" placeholder="symbol, correlation, position"></label><label>Engine <select id="siteEngine"><option value="">All engines</option></select></label><span id="siteCount" class="spacer"></span></div>
      <div class="table-wrap"><table><thead><tr><th>Tile</th><th>Site</th><th>Engine</th><th>Correlation</th><th>Target call</th><th>Position</th><th class="num">Events</th><th>Status</th></tr></thead><tbody id="siteRows"></tbody></table></div>
    </section>

    <section id="view-communication" class="view" data-view-panel="communication">
      <div class="section-head"><div><h2>Communication / Direct-DTE</h2><p>wait/completion duration 与未校准 raw PMU activity 分开显示。</p></div></div>
      <div class="notice">每一行只属于自身 tile-local clock domain；不同 tile 的位置和 duration 不组成全局通信序列。</div>
      <div id="communicationSummary" class="grid kpis"></div>
      <div class="table-wrap"><table><thead><tr><th>Tile</th><th>Role</th><th>Site / symbol</th><th class="num">Wait cycles</th><th>Wait status</th><th class="num">Raw PMU</th><th>Raw status</th></tr></thead><tbody id="communicationRows"></tbody></table></div>
    </section>

    <section id="view-diagnostics" class="view" data-view-panel="diagnostics">
      <div class="section-head"><div><h2>Diagnostics / Raw</h2><p>整体 duration 资格与局部 trace/PMU 可用性分开陈述。</p></div></div>
      <div id="validityMatrix" class="legend"></div>
      <div class="card" id="diagnosticRows"></div>
      <div class="raw-tabs"><button class="active" data-raw-target="analysis" type="button">Analysis JSON</button><button data-raw-target="evidence" type="button">Evidence JSON</button></div>
      <pre id="rawPayload"></pre>
    </section>

    <div class="metric-note" style="margin-top:14px">Stable publication: index.html · analysis.json · evidence.json. Directory and regular files are 0777.</div>
  </main>
</div>
<script>
"use strict";
const analysis=__ANALYSIS__;
const evidence=__EVIDENCE__;
const finalArtifact=analysis.final_artifact;
const ENGINES=["CT","NE","RDMA","WDMA","TDMA","DIRECT_DTE"];
const STATES=["Measured","Sampled","Bounded","Unavailable","Incomplete","Invalid"];
const state={view:"overview",tile:0,zoom:1,engines:new Set(ENGINES),selectedEvent:null,raw:"analysis"};
const q=(selector,root)=>(root||document).querySelector(selector);
const qa=(selector,root)=>Array.prototype.slice.call((root||document).querySelectorAll(selector));
const escapeHtml=value=>String(value==null?"—":value).replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const number=value=>value==null?"—":Number(value).toLocaleString();
const milliseconds=value=>(Number(value)/1e6).toFixed(3)+" ms";
const tileLabel=tile=>"T"+(Number(tile)<10?"0":"")+String(tile);
const statusBadge=value=>`<span class="status status-${escapeHtml(value)}">${escapeHtml(value)}</span>`;
const selectedTile=()=>finalArtifact.tiles.find(row=>row.tile===state.tile);
const tileEvents=()=>finalArtifact.timeline_events.filter(row=>row.tile===state.tile);

function navigate(view){
  state.view=view;
  qa("[data-view-panel]").forEach(node=>node.classList.toggle("active",node.dataset.viewPanel===view));
  qa("[data-view]").forEach(node=>node.classList.toggle("active",node.dataset.view===view));
  window.scrollTo(0,0);
  if(view==="timeline")renderTimeline();
  if(view==="engines")renderEngines();
  if(view==="sites")renderSites();
  if(view==="communication")renderCommunication();
  if(view==="diagnostics")renderDiagnostics();
}

function selectTile(tile,view="timeline",engine=null){
  state.tile=Number(tile);
  if(engine){state.engines=new Set([engine]);renderEngineFilters()}
  q("#timelineTile").value=String(state.tile);
  q("#engineTile").value=String(state.tile);
  navigate(view);
}

function focusEvent(tile,siteId,engine){
  const event=finalArtifact.timeline_events.find(row=>row.tile===Number(tile)&&row.site_id===Number(siteId)&&row.engine===engine);
  if(!event)return;
  state.tile=Number(tile);
  state.selectedEvent=event;
  state.engines.add(engine);
  renderEngineFilters();
  q("#timelineTile").value=String(state.tile);
  q("#engineTile").value=String(state.tile);
  navigate("timeline");
}

function bindEventLinks(selector){
  qa(selector).forEach(node=>{
    const activate=()=>focusEvent(node.dataset.eventTile,node.dataset.eventSite,node.dataset.eventEngine);
    node.addEventListener("click",activate);
    node.addEventListener("keydown",event=>{
      if(event.key==="Enter"||event.key===" "){event.preventDefault();activate()}
    });
  });
}

function buildResourceTree(){
  q("#resourceTree").innerHTML=`<details open><summary>Card 0 · ${finalArtifact.tiles.length} tiles</summary>${finalArtifact.tiles.map(tile=>`<details><summary>${tileLabel(tile.tile)} · ${escapeHtml(tile.trace_status)}</summary>${ENGINES.map(engine=>`<button type="button" data-tree-tile="${tile.tile}" data-tree-engine="${engine}">${engine}</button>`).join("")}</details>`).join("")}</details>`;
  qa("[data-tree-tile]").forEach(node=>node.addEventListener("click",()=>selectTile(node.dataset.treeTile,"timeline",node.dataset.treeEngine)));
}

function renderOverview(){
  const duration=finalArtifact.duration;
  q("#statusLegend").innerHTML=STATES.map(statusBadge).join("");
  q("#primaryDuration").textContent=milliseconds(duration.host_elapsed_ns);
  q("#primaryResolution").textContent=`completion observation ≤ ${number(duration.completion_observation_resolution_ns)} ns · primary`;
  q("#measurementStatus").innerHTML=statusBadge(duration.status);
  q("#outputStatus").innerHTML=statusBadge(analysis.validity.output_equivalence?"Measured":"Invalid");
  q("#outputNote").textContent=`${finalArtifact.output.resource_count} resources · ${finalArtifact.output.correctness_status}`;
  const completeTiles=finalArtifact.tiles.filter(tile=>tile.trace_status==="Bounded").length;
  q("#traceCoverage").textContent=`${completeTiles} / ${finalArtifact.tiles.length}`;
  q("#traceNote").textContent="tile-local complete trace spans";
  q("#overviewTiles").innerHTML=[...finalArtifact.tiles].sort((left,right)=>left.y-right.y||left.x-right.x).map(tile=>`<button class="tile-card" data-overview-tile="${tile.tile}" type="button"><b>${tileLabel(tile.tile)} ${statusBadge(tile.trace_status)}</b><small>${number(tile.trace_entry_cycles)} cycles · (${tile.x},${tile.y})</small></button>`).join("");
  qa("[data-overview-tile]").forEach(node=>node.addEventListener("click",()=>selectTile(node.dataset.overviewTile)));
  const methodLabels={primary_duration:"Primary duration",timeline_axis:"Timeline axis",timeline_scope:"Clock/order",ncc_engine_time:"NCC busy",ncc_activity_window:"NCC window",direct_dte_time:"Direct-DTE wait",direct_dte_raw_pmu:"Direct-DTE raw"};
  q("#methodList").innerHTML=Object.keys(analysis.method).map(key=>`<div class="method-row"><b>${escapeHtml(methodLabels[key]||key)}</b><span>${escapeHtml(analysis.method[key])}</span></div>`).join("");
  q("#overviewEngineRows").innerHTML=ENGINES.map(engine=>{
    const rows=finalArtifact.tiles.map(tile=>tile.engines.find(item=>item.engine===engine));
    const direct=engine==="DIRECT_DTE";
    const values=rows.map(row=>direct?row.wait_window_cycles:row.busy_cycles);
    const available=values.filter(value=>value!=null);
    const derived=available.reduce((sum,value)=>sum+Number(value),0);
    const windows=rows.reduce((sum,row)=>sum+Number(row.activity_window_count||0),0);
    const states=rows.map(row=>direct?row.wait_window_status:row.busy_cycles_status);
    const aggregateStatus=states.every(value=>value==="Measured")?"Measured":states.some(value=>value==="Invalid")?"Invalid":states.some(value=>value==="Incomplete")?"Incomplete":"Unavailable";
    return `<tr><td><b>${engine}</b></td><td>${direct?"Tile-local wait cycles":"Aggregate PMU busy cycles"}</td><td class="num">${available.length?number(derived):"—"}</td><td class="num">${available.length} / ${rows.length}</td><td class="num">${windows}</td><td>${statusBadge(aggregateStatus)}</td></tr>`;
  }).join("");
}

function populateSelectors(){
  const options=finalArtifact.tiles.map(tile=>`<option value="${tile.tile}">${tileLabel(tile.tile)} · (${tile.x},${tile.y})</option>`).join("");
  q("#timelineTile").innerHTML=options;
  q("#engineTile").innerHTML=options;
  q("#siteEngine").innerHTML+=ENGINES.map(engine=>`<option value="${engine}">${engine}</option>`).join("");
}

function renderEngineFilters(){
  q("#engineFilters").innerHTML=ENGINES.map(engine=>`<label class="engine-toggle"><input type="checkbox" data-engine-filter="${engine}" ${state.engines.has(engine)?"checked":""}>${engine}</label>`).join("");
  qa("[data-engine-filter]").forEach(node=>node.addEventListener("change",()=>{
    node.checked?state.engines.add(node.dataset.engineFilter):state.engines.delete(node.dataset.engineFilter);
    renderTimeline();
  }));
}

function renderRuler(axis){
  const ticks=Array.from({length:6},(_,index)=>({fraction:index/5,value:axis==null?null:Math.round(axis*index/5)}));
  q("#timelineRuler").innerHTML=`<div class="ruler"><span class="metric-note">local cycles</span><div class="ruler-track">${ticks.map(tick=>`<i class="tick" style="left:${tick.fraction*100}%"><span>${number(tick.value)}</span></i>`).join("")}</div></div>`;
}

function renderTimeline(){
  const tile=selectedTile();
  if(!tile)return;
  q("#timelineTile").value=String(state.tile);
  q("#timelineCanvas").style.minWidth=`${Math.max(100,state.zoom*100)}%`;
  renderRuler(tile.trace_entry_cycles);
  const events=tileEvents();
  q("#timelineLanes").innerHTML=ENGINES.map(engine=>{
    const visible=state.engines.has(engine);
    const engineEvents=events.filter(event=>event.engine===engine);
    const marks=visible?engineEvents.map(event=>`<button class="event event-${engine}${state.selectedEvent&&state.selectedEvent.tile===event.tile&&state.selectedEvent.sequence===event.sequence&&state.selectedEvent.site_id===event.site_id?" selected":""}" data-event-sequence="${event.sequence}" data-event-site="${event.site_id}" style="left:${event.plot_begin_fraction*100}%;width:${Math.max(.2,(event.plot_end_fraction-event.plot_begin_fraction)*100)}%" title="${escapeHtml(engine)} · ${escapeHtml(event.duration_status)} · ${number(event.activity_window_cycles)} cycles" type="button"></button>`).join(""):"";
    const gridlines=[20,40,60,80].map(position=>`<i class="lane-gridline" style="left:${position}%"></i>`).join("");
    return `<div class="lane" data-lane-engine="${engine}"><div class="lane-label"><b>${engine}</b><span>${engineEvents.length}</span></div><div class="lane-track">${gridlines}${marks}</div></div>`;
  }).join("");
  qa("[data-event-sequence]").forEach(node=>node.addEventListener("click",()=>{
    const event=events.find(row=>row.sequence===Number(node.dataset.eventSequence)&&row.site_id===Number(node.dataset.eventSite));
    state.selectedEvent=event||null;
    renderEventDetail();
    qa(".event").forEach(mark=>mark.classList.toggle("selected",mark===node));
  }));
  if(!state.selectedEvent||state.selectedEvent.tile!==state.tile)state.selectedEvent=events[0]||null;
  renderEventDetail();
}

function renderEventDetail(){
  const event=state.selectedEvent;
  if(!event){q("#eventDetail").innerHTML=`<h3>Event details</h3><div class="empty">No usable event on this tile.</div>`;return}
  const counterLabel=event.engine==="DIRECT_DTE"?"Raw DTE PMU":"Observed counter delta";
  const counterValue=event.engine==="DIRECT_DTE"?event.direct_dte_raw_pmu_activity:event.ncc_busy_cycles;
  q("#eventDetail").innerHTML=`<h3>${escapeHtml(tileLabel(event.tile)+" · "+event.engine)} ${statusBadge(event.duration_status)}</h3><dl class="kv"><dt>Local interval</dt><dd class="mono">${number(event.trace_entry_offset_begin)} → ${number(event.trace_entry_offset_end)} cycles</dd><dt>Window</dt><dd>${number(event.activity_window_cycles)} cycles · ${statusBadge(event.duration_status)}</dd><dt>${counterLabel}</dt><dd>${number(counterValue)} · ${statusBadge(event.counter_status)}</dd><dt>Site</dt><dd class="mono">${event.site_id}:${event.sub_index}</dd><dt>Correlation</dt><dd>${escapeHtml(event.correlation_key)}</dd><dt>Target call</dt><dd class="mono">#${number(event.target_call_ordinal)} ${escapeHtml(event.target_call_symbol)}</dd><dt>Position</dt><dd class="mono">${escapeHtml(event.position)}</dd><dt>DTE role</dt><dd>${escapeHtml(event.dte_role)}</dd><dt>Scope</dt><dd>tile-local only</dd></dl>`;
}

function renderEngines(){
  state.tile=Number(q("#engineTile").value||state.tile);
  const tile=selectedTile();
  q("#engineTile").value=String(state.tile);
  q("#engineTileMeta").innerHTML=`${statusBadge(tile.trace_status)} · trace ${number(tile.trace_entry_cycles)} cycles · clock ${statusBadge(tile.clock_status)}`;
  q("#engineRows").innerHTML=tile.engines.map(engine=>{
    if(engine.engine==="DIRECT_DTE")return[
      `<tr><td><b>DIRECT_DTE</b></td><td>Wait / completion</td><td class="num">${number(engine.wait_window_cycles)}</td><td class="num">${engine.wait_window_count}</td><td>${statusBadge(engine.wait_window_status)}</td><td>Measured wait interval; not raw PMU.</td></tr>`,
      `<tr><td></td><td>Raw PMU activity</td><td class="num">${number(engine.raw_pmu_activity)}</td><td class="num">${engine.activity_window_count}</td><td>${statusBadge(engine.raw_pmu_activity_status)}</td><td>Sampled and uncalibrated; never elapsed time.</td></tr>`
    ].join("");
    return `<tr><td><b>${engine.engine}</b></td><td>Aggregate hardware busy</td><td class="num">${number(engine.busy_cycles)}</td><td class="num">${engine.activity_window_count}</td><td>${statusBadge(engine.busy_cycles_status)}</td><td>Measured PMU busy cycles. Activity windows are ${escapeHtml(engine.activity_window_status)} only.</td></tr>`;
  }).join("");
}

function renderSites(){
  const query=q("#siteSearch").value.trim().toLowerCase();
  const engine=q("#siteEngine").value;
  const rows=finalArtifact.sites.filter(site=>(!engine||site.engine===engine)&&(!query||[site.correlation_key,site.target_call_symbol,site.position,site.site_id,site.tile].some(value=>String(value==null?"":value).toLowerCase().indexOf(query)!==-1)));
  q("#siteCount").textContent=`${rows.length} / ${finalArtifact.sites.length} sites`;
  q("#siteRows").innerHTML=rows.length?rows.map(site=>`<tr${site.event_count?` class="interactive-row" tabindex="0" title="Open correlated timeline event" data-event-tile="${site.tile}" data-event-site="${site.site_id}" data-event-engine="${site.engine}"`:""}><td>${tileLabel(site.tile)}</td><td class="mono">${site.site_id}</td><td>${site.engine}</td><td>${escapeHtml(site.correlation_key)}</td><td class="mono">#${site.target_call_ordinal} ${escapeHtml(site.target_call_symbol)}</td><td class="mono">${escapeHtml(site.position)}</td><td class="num">${site.event_count}</td><td>${statusBadge(site.observation_status)}</td></tr>`).join(""):`<tr><td colspan="8" class="empty">No matching sites.</td></tr>`;
  bindEventLinks("#siteRows [data-event-site]");
}

function renderCommunication(){
  const events=finalArtifact.timeline_events.filter(event=>event.engine==="DIRECT_DTE");
  const communication=finalArtifact.communication;
  q("#communicationSummary").innerHTML=`<div class="card"><div class="metric-label">DTE waits</div><div class="metric-value">${communication.direct_dte_event_count}</div><div class="metric-note">Measured tile-local windows</div></div><div class="card"><div class="metric-label">Send / receive</div><div class="metric-value">${communication.send_count} / ${communication.receive_count}</div></div><div class="card"><div class="metric-label">Active tiles</div><div class="metric-value">${communication.tiles_with_activity.length}</div></div><div class="card"><div class="metric-label">Cross-tile order</div><div class="metric-value">${statusBadge("Unavailable")}</div><div class="metric-note">No inferred global timeline</div></div>`;
  q("#communicationRows").innerHTML=events.length?events.map(event=>`<tr class="interactive-row" tabindex="0" title="Open correlated timeline event" data-event-tile="${event.tile}" data-event-site="${event.site_id}" data-event-engine="${event.engine}"><td>${tileLabel(event.tile)}</td><td>${escapeHtml(event.dte_role)}</td><td><span class="mono">${event.site_id}</span> · ${escapeHtml(event.target_call_symbol)}</td><td class="num">${number(event.direct_dte_wait_cycles)}</td><td>${statusBadge(event.duration_status)}</td><td class="num">${number(event.direct_dte_raw_pmu_activity)}</td><td>${statusBadge(event.counter_status)}</td></tr>`).join(""):`<tr><td colspan="7" class="empty">No Direct-DTE wait was observed.</td></tr>`;
  bindEventLinks("#communicationRows [data-event-site]");
}

function renderDiagnostics(){
  q("#validityMatrix").innerHTML=Object.keys(analysis.validity).map(key=>`<span>${escapeHtml(key)} ${statusBadge(analysis.validity[key]?"Measured":"Invalid")}</span>`).join("");
  q("#diagnosticRows").innerHTML=analysis.diagnostics.length?analysis.diagnostics.map(row=>`<div class="diag"><b class="severity-${escapeHtml(row.severity)}">${escapeHtml(row.severity)}</b><span class="mono">${escapeHtml(row.code)}</span><span>${row.tile==null?"global":tileLabel(row.tile)}</span><span>${escapeHtml(row.message)}</span></div>`).join(""):`<div class="empty">No analyzer diagnostics.</div>`;
  q("#rawPayload").textContent=JSON.stringify(state.raw==="analysis"?analysis:evidence,null,2);
  qa("[data-raw-target]").forEach(node=>node.classList.toggle("active",node.dataset.rawTarget===state.raw));
}

qa("[data-view]").forEach(node=>node.addEventListener("click",()=>navigate(node.dataset.view)));
q("#timelineTile").addEventListener("change",event=>{state.tile=Number(event.target.value);state.selectedEvent=null;renderTimeline()});
q("#engineTile").addEventListener("change",event=>{state.tile=Number(event.target.value);renderEngines()});
q("#timelineZoom").addEventListener("input",event=>{state.zoom=Number(event.target.value);renderTimeline()});
q("#timelineFit").addEventListener("click",()=>{state.zoom=1;q("#timelineZoom").value="1";renderTimeline()});
q("#siteSearch").addEventListener("input",renderSites);
q("#siteEngine").addEventListener("change",renderSites);
qa("[data-raw-target]").forEach(node=>node.addEventListener("click",()=>{state.raw=node.dataset.rawTarget;renderDiagnostics()}));

populateSelectors();
renderEngineFilters();
buildResourceTree();
renderOverview();
renderTimeline();
renderEngines();
renderSites();
renderCommunication();
renderDiagnostics();
window.__waferProfileUI={analysis,evidence,state,navigate,selectTile,focusEvent,renderTimeline};
</script>
</body>
</html>"""


def render_report(
    evidence: object, analysis: Mapping[str, Any] | None = None
) -> str:
    valid = validate_evidence(evidence)
    result = analyze_evidence(valid) if analysis is None else analysis

    def script_payload(value: object) -> str:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return (
            payload.replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

    return (
        _REPORT_TEMPLATE.replace(
            "__RUN_ID__", html.escape(str(valid["run_id"]))
        )
        .replace("__ANALYSIS__", script_payload(result))
        .replace("__EVIDENCE__", script_payload(valid))
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
    else:
        os.chmod(evidence_output, 0o777)
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
