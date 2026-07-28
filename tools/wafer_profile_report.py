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
SCHEMA_VERSION = 7
COMPANION_SCHEMA_VERSION = 4
RECORD_ABI = "wafer-tx81-profiler-record-v3"
ANALYSIS_SCHEMA_NAME = "wafer.profile.analysis"
ANALYSIS_SCHEMA_VERSION = 6
TILES = tuple(range(16))
NCC_ENGINES = ("CT", "NE", "RDMA", "WDMA", "TDMA")
ENGINES = NCC_ENGINES + ("DIRECT_DTE",)
EVENT_ENGINES = ENGINES
SITE_KINDS = (
    "ncc-command",
    "ncc-completion",
    "direct-dte-control",
    "direct-dte-wait",
)
EVENT_KINDS = (
    "ncc-command",
    "ncc-completion-wait",
    "direct-dte-wait",
    "direct-dte-peer-ready-wait",
    "direct-dte-setup-issue",
    "direct-dte-completion-wait",
    "direct-dte-cleanup",
    "target-site",
)
OBSERVATION_STATUSES = (
    "engine-delta-bounded",
    "counter-no-change",
    "attribution-ambiguous",
    "counter-unavailable",
)
COST_SUMMARY_FIELDS = (
    "ncc_pmu_sample_cycles",
    "dte_pmu_sample_cycles",
    "event_bookkeeping_cycles",
    "status_poll_cycles",
    "site_hook_cycles",
    "completion_loop_bookkeeping_cycles",
    "entry_setup_cycles",
    "entry_teardown_cycles",
)
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
            "record_abi",
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
    if basis != "typed-target-call-ordinal-ssa-identity-occurrence-v1":
        _fail("identity.site_correlation_basis", "unknown correlation basis")
    if _string(identity["record_abi"], "identity.record_abi") != RECORD_ABI:
        _fail("identity.record_abi", f"must be {RECORD_ABI!r}")


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
                "device_elapsed_ns",
                "device_timer_kind",
                "host_submit_ns",
                "host_launch_to_completion_ns",
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
            row["device_elapsed_ns"],
            f"{path}.device_elapsed_ns",
            minimum=0,
            maximum=UINT64_MAX,
        )
        timer_kind = _string(
            row["device_timer_kind"], f"{path}.device_timer_kind"
        )
        if timer_kind != "tx-stream-events":
            _fail(
                f"{path}.device_timer_kind",
                "must be 'tx-stream-events'",
            )
        _integer(
            row["host_submit_ns"],
            f"{path}.host_submit_ns",
            minimum=0,
            maximum=UINT64_MAX,
        )
        host_envelope = _integer(
            row["host_launch_to_completion_ns"],
            f"{path}.host_launch_to_completion_ns",
            minimum=0,
            maximum=UINT64_MAX,
        )
        if row["host_submit_ns"] > host_envelope:
            _fail(
                f"{path}.host_submit_ns",
                "must not exceed host_launch_to_completion_ns",
            )
        _integer(
            row["completion_observation_resolution_ns"],
            f"{path}.completion_observation_resolution_ns",
            minimum=0,
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
            "site_kind",
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
        site_kind = _string(row["site_kind"], f"{path}.site_kind")
        if site_kind not in SITE_KINDS:
            _fail(f"{path}.site_kind", f"must be one of {list(SITE_KINDS)}")
        engine = row["engine"]
        if engine is not None:
            engine = _string(engine, f"{path}.engine")
            if engine not in ENGINES:
                _fail(f"{path}.engine", f"must be null or one of {list(ENGINES)}")
        if site_kind == "ncc-command" and engine not in NCC_ENGINES:
            _fail(f"{path}.engine", "an NCC command must name its NCC engine")
        if site_kind == "ncc-completion" and engine is not None:
            _fail(f"{path}.engine", "an NCC completion site must use null engine")
        if site_kind == "direct-dte-wait" and engine != "DIRECT_DTE":
            _fail(
                f"{path}.engine",
                "a Direct-DTE wait site must name the DIRECT_DTE engine",
            )
        if site_kind == "direct-dte-control" and engine is not None:
            _fail(
                f"{path}.engine",
                "a Direct-DTE control site must use null engine",
            )
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
                "cost_summary",
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
        cost_summary = _mapping(row["cost_summary"], f"{path}.cost_summary")
        _exact_keys(
            cost_summary, set(COST_SUMMARY_FIELDS), f"{path}.cost_summary"
        )
        for field in COST_SUMMARY_FIELDS:
            _integer(
                cost_summary[field],
                f"{path}.cost_summary.{field}",
                minimum=0,
                maximum=UINT64_MAX,
            )
        trace_events = _sequence(row["events"], f"{path}.events")
        active_site_container: Mapping[str, Any] | None = None
        next_site_sub_index = 0
        for event_index, event_value in enumerate(trace_events):
            event = _mapping(event_value, f"{path}.events[{event_index}]")
            event_path = f"{path}.events[{event_index}]"
            _exact_keys(
                event,
                {
                    "sequence",
                    "site_id",
                    "sub_index",
                    "engine",
                    "kind",
                    "observed_begin_cycle",
                    "observed_end_cycle",
                    "counter_delta",
                    "site_begin_cycle",
                    "site_end_cycle",
                    "operation_begin_cycle",
                    "operation_end_cycle",
                    "observation_count",
                    "observed_span_valid",
                    "site_span_valid",
                    "operation_span_valid",
                    "positive_delta",
                    "attribution_ambiguous",
                    "ncc_counter_valid",
                    "observation_status",
                    "worker",
                    "wait_scope",
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
            engine = event["engine"]
            if engine is not None:
                engine = _string(engine, f"{event_path}.engine")
                if engine not in EVENT_ENGINES:
                    _fail(
                        f"{event_path}.engine",
                        f"must be null or one of {list(EVENT_ENGINES)}",
                    )
            kind = _string(event["kind"], f"{event_path}.kind")
            if kind not in EVENT_KINDS:
                _fail(f"{event_path}.kind", f"must be one of {list(EVENT_KINDS)}")
            site = sites.get((tile, site_id))
            if site is None:
                _fail(f"{event_path}.site_id", "does not map to a final site")
            expected_site_kinds = {
                "ncc-command": {
                    "target-site",
                    "ncc-command",
                    "ncc-completion-wait",
                },
                "ncc-completion": {"target-site", "ncc-completion-wait"},
                "direct-dte-control": {"target-site"},
                "direct-dte-wait": {
                    "target-site",
                    "direct-dte-wait",
                    "direct-dte-peer-ready-wait",
                    "direct-dte-setup-issue",
                    "direct-dte-completion-wait",
                    "direct-dte-cleanup",
                },
            }
            if (
                kind not in ("ncc-completion-wait", "target-site")
                and site["engine"] is not None
                and site["engine"] != engine
            ):
                _fail(f"{event_path}.engine", "conflicts with the site engine")
            if kind not in expected_site_kinds[site["site_kind"]]:
                _fail(f"{event_path}.kind", "conflicts with the site kind")
            for field in (
                "observed_begin_cycle",
                "observed_end_cycle",
                "site_begin_cycle",
                "site_end_cycle",
                "operation_begin_cycle",
                "operation_end_cycle",
            ):
                _integer(
                    event[field],
                    f"{event_path}.{field}",
                    minimum=0,
                    maximum=UINT64_MAX,
                )
            raw = _integer(
                event["counter_delta"],
                f"{event_path}.counter_delta",
                minimum=0,
                maximum=UINT64_MAX,
            )
            _integer(
                event["observation_count"],
                f"{event_path}.observation_count",
                minimum=0,
                maximum=UINT32_MAX,
            )
            observed_valid = _boolean(
                event["observed_span_valid"],
                f"{event_path}.observed_span_valid",
            )
            site_valid = _boolean(
                event["site_span_valid"], f"{event_path}.site_span_valid"
            )
            operation_valid = _boolean(
                event["operation_span_valid"],
                f"{event_path}.operation_span_valid",
            )
            positive_delta = _boolean(
                event["positive_delta"], f"{event_path}.positive_delta"
            )
            ambiguous = _boolean(
                event["attribution_ambiguous"],
                f"{event_path}.attribution_ambiguous",
            )
            ncc_counter_valid = event["ncc_counter_valid"]
            status = event["observation_status"]
            if kind == "ncc-command":
                ncc_counter_valid = _boolean(
                    ncc_counter_valid,
                    f"{event_path}.ncc_counter_valid",
                )
                status = _string(status, f"{event_path}.observation_status")
                if status not in OBSERVATION_STATUSES:
                    _fail(
                        f"{event_path}.observation_status",
                        f"must be one of {list(OBSERVATION_STATUSES)}",
                    )
                expected_status = (
                    "counter-unavailable"
                    if not ncc_counter_valid
                    else "attribution-ambiguous"
                    if ambiguous
                    else "engine-delta-bounded"
                    if positive_delta
                    else "counter-no-change"
                )
                if status != expected_status:
                    _fail(
                        f"{event_path}.observation_status",
                        f"must be {expected_status!r} for its metadata",
                    )
                if ncc_counter_valid != observed_valid:
                    _fail(
                        f"{event_path}.ncc_counter_valid",
                        "must agree with NCC observation-span availability",
                    )
                if not ncc_counter_valid and (
                    raw != 0 or event["observation_count"] != 0
                ):
                    _fail(
                        event_path,
                        "an unavailable NCC counter must not retain a delta or sample count",
                    )
            else:
                if ncc_counter_valid is not None:
                    _fail(
                        f"{event_path}.ncc_counter_valid",
                        "must be null outside an NCC command",
                    )
                if status is not None:
                    _fail(
                        f"{event_path}.observation_status",
                        "must be null for a Kcore/DTE phase",
                    )
            if positive_delta != (raw > 0):
                _fail(
                    f"{event_path}.positive_delta",
                    "must agree with whether counter_delta is positive",
                )
            worker = event["worker"]
            if worker is not None:
                _integer(worker, f"{event_path}.worker", minimum=0, maximum=2)
            wait_scope = event["wait_scope"]
            if wait_scope is not None and wait_scope not in ("local", "worker"):
                _fail(f"{event_path}.wait_scope", "must be null, local, or worker")
            if observed_valid and (
                event["observed_end_cycle"] < event["observed_begin_cycle"]
            ):
                _fail(
                    f"{event_path}.observed_end_cycle",
                    "must not precede a valid observed span",
                )
            if site_valid and event["site_end_cycle"] < event["site_begin_cycle"]:
                _fail(
                    f"{event_path}.site_end_cycle",
                    "must not precede a valid site span",
                )
            if operation_valid and (
                event["operation_end_cycle"] < event["operation_begin_cycle"]
            ):
                _fail(
                    f"{event_path}.operation_end_cycle",
                    "must not precede a valid operation span",
                )
            if engine == "DIRECT_DTE":
                dte_role = event["dte_role"]
                if not isinstance(dte_role, str) or dte_role not in (
                    "send",
                    "receive",
                ):
                    _fail(f"{event_path}.dte_role", "must be send or receive")
                if kind == "direct-dte-wait":
                    valid = _boolean(
                        event["dte_counter_valid"],
                        f"{event_path}.dte_counter_valid",
                    )
                    if not valid and raw != 0:
                        _fail(
                            f"{event_path}.counter_delta",
                            "must be zero when the DTE counter is unusable",
                        )
                elif event["dte_counter_valid"] is not None:
                    _fail(
                        f"{event_path}.dte_counter_valid",
                        "must be null outside a Direct-DTE wait",
                    )
            elif event["dte_role"] is not None or event["dte_counter_valid"] is not None:
                _fail(
                    event_path,
                    "a non-DTE-wait event must carry null DTE role and counter validity",
                )
            if kind == "target-site":
                if (
                    engine is not None
                    or not site_valid
                    or observed_valid
                    or operation_valid
                    or raw != 0
                    or event["observation_count"] != 0
                    or ambiguous
                    or worker is not None
                    or wait_scope is not None
                ):
                    _fail(
                        event_path,
                        "a target-site container may carry only its typed site span",
                    )
                if int(event["sub_index"]) != 0:
                    _fail(
                        f"{event_path}.sub_index",
                        "a target-site container must be sub-index zero",
                    )
                active_site_container = event
                next_site_sub_index = 1
            else:
                if active_site_container is None:
                    _fail(
                        event_path,
                        "a dynamic event must follow its target-site container",
                    )
                if (
                    int(active_site_container["site_id"]) != site_id
                    or int(active_site_container["site_begin_cycle"])
                    != int(event["site_begin_cycle"])
                    or int(active_site_container["site_end_cycle"])
                    != int(event["site_end_cycle"])
                ):
                    _fail(
                        event_path,
                        "a dynamic event must match its target-site container",
                    )
                if int(event["sub_index"]) != next_site_sub_index:
                    _fail(
                        f"{event_path}.sub_index",
                        "dynamic site sub-indexes must be contiguous",
                    )
                next_site_sub_index += 1

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
    if recovery != end:
        return None, "counter recovery does not match the terminal end value"
    return end - start, None


def _engine_active_time_summary(
    tile_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a non-latency summary from per-tile NCC PMU execution time."""

    expected_tiles = len(TILES)

    def status(available: int) -> str:
        if available == expected_tiles:
            return "Measured"
        if available:
            return "Incomplete"
        return "Unavailable"

    def statistics(values: Sequence[int]) -> dict[str, int | float | None]:
        if not values:
            return {
                "minimum_ns": None,
                "average_ns": None,
                "maximum_ns": None,
                "work_volume_ns": None,
            }
        work_volume = sum(values)
        return {
            "minimum_ns": min(values),
            "average_ns": work_volume / len(values),
            "maximum_ns": max(values),
            "work_volume_ns": work_volume,
        }

    by_engine: list[dict[str, Any]] = []
    for engine in NCC_ENGINES:
        values: list[int] = []
        for tile in tile_rows:
            row = next(
                item for item in tile["engines"] if item["engine"] == engine
            )
            if (
                row["engine_execution_time_valid"]
                and row["engine_execution_time_ns"] is not None
            ):
                values.append(int(row["engine_execution_time_ns"]))
        summary = statistics(values)
        by_engine.append(
            {
                "engine": engine,
                "measurement_kind": "trace-pmu-engine-active-time",
                "available_tile_count": len(values),
                "active_tile_count": sum(value > 0 for value in values),
                "minimum_per_tile_ns": summary["minimum_ns"],
                "average_per_tile_ns": summary["average_ns"],
                "maximum_per_tile_ns": summary["maximum_ns"],
                "sum_across_tiles_work_ns": summary["work_volume_ns"],
                "status": status(len(values)),
            }
        )

    complete_tile_values: list[int] = []
    for tile in tile_rows:
        engine_rows = [
            item for item in tile["engines"] if item["engine"] in NCC_ENGINES
        ]
        if len(engine_rows) != len(NCC_ENGINES) or any(
            not item["engine_execution_time_valid"]
            or item["engine_execution_time_ns"] is None
            for item in engine_rows
        ):
            continue
        complete_tile_values.append(
            sum(int(item["engine_execution_time_ns"]) for item in engine_rows)
        )

    tile_summary = statistics(complete_tile_values)
    return {
        "measurement_kind": "trace-pmu-engine-active-time",
        "source_capture": "trace",
        "measurement_run": "trace-diagnostic",
        "relation_to_primary": "separate-run-proxy",
        "unit": "ns",
        "scope": "per-tile-per-ncc-engine",
        "additive_to_kernel_launch_envelope": False,
        "cross_tile_sum_role": "work-volume-not-wall-time",
        "card_wide_engine_elapsed_ns": None,
        "card_wide_engine_elapsed_status": "Unavailable",
        "card_wide_engine_elapsed_reason": (
            "per-tile engines may overlap and have no qualified global "
            "activity start/end alignment"
        ),
        "per_tile_work_volume": {
            "available_tile_count": len(complete_tile_values),
            "minimum_per_tile_total_work_ns": tile_summary["minimum_ns"],
            "average_per_tile_total_work_ns": tile_summary["average_ns"],
            "maximum_per_tile_total_work_ns": tile_summary["maximum_ns"],
            "sum_across_tiles_work_ns": tile_summary["work_volume_ns"],
            "status": status(len(complete_tile_values)),
        },
        "by_engine": by_engine,
    }


def _event_span(
    event: Mapping[str, Any], name: str, lower: int, upper: int
) -> tuple[int, int] | None:
    if not event[f"{name}_span_valid"]:
        return None
    begin = int(event[f"{name}_begin_cycle"])
    end = int(event[f"{name}_end_cycle"])
    if begin < lower or end < begin or end > upper:
        return None
    return begin, end


def _operation_cost_contract(kind: str) -> tuple[str, str, str, str]:
    if kind == "ncc-command":
        return (
            "ncc-submit",
            "yes",
            "proxy",
            "reduce descriptor preparation and issue-path serialization",
        )
    if kind == "ncc-completion-wait":
        return (
            "completion-wait-proxy",
            "yes",
            "proxy",
            "move joins later, overlap independent work, or reduce polling",
        )
    dte_categories = {
        "direct-dte-peer-ready-wait": "dte-peer-ready-wait",
        "direct-dte-setup-issue": "dte-setup-issue",
        "direct-dte-completion-wait": "dte-completion-wait",
        "direct-dte-cleanup": "dte-cleanup",
        "direct-dte-wait": "dte-wait-aggregate-fallback",
    }
    return (
        dte_categories[kind],
        "yes",
        "proxy",
        "coalesce transfers and overlap DTE phases with independent NCC work",
    )


def _site_ref(
    site: Mapping[str, Any], instance_sequence: int
) -> dict[str, Any]:
    return {
        "site_id": int(site["site_id"]),
        "instance_sequence": instance_sequence,
        "site_kind": str(site["site_kind"]),
        "target_call_symbol": str(site["target_call_symbol"]),
        "target_call_ordinal": int(site["target_call_ordinal"]),
        "correlation_key": str(site["correlation_key"]),
        "position": site.get("position"),
    }


def _site_label(site: Mapping[str, Any] | None) -> str:
    if site is None:
        return "none"
    return (
        f"site-{site['site_id']}/instance-{site['instance_sequence']}/"
        f"{site['site_kind']}/{site['target_call_symbol']}"
    )


def _semantic_segments(
    lower: int,
    upper: int,
    sites: Sequence[Mapping[str, Any]],
    operations: Sequence[tuple[int, int, str, int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build an exclusive, source-attributed partition of one Trace entry."""
    boundaries = {lower, upper}
    for site in sites:
        boundaries.update((int(site["begin_cycle"]), int(site["end_cycle"])))
    for begin, end, _, _, _ in operations:
        boundaries.update((begin, end))
    ordered_sites = sorted(
        sites,
        key=lambda site: (
            int(site["begin_cycle"]),
            int(site["end_cycle"]),
            int(site["ref"]["instance_sequence"]),
        ),
    )
    first_site = ordered_sites[0] if ordered_sites else None
    last_site = (
        max(
            ordered_sites,
            key=lambda site: (
                int(site["end_cycle"]),
                int(site["ref"]["instance_sequence"]),
            ),
        )
        if ordered_sites
        else None
    )
    result: list[dict[str, Any]] = []
    ordered = sorted(boundaries)
    for begin, end in zip(ordered, ordered[1:]):
        if end <= begin:
            continue
        operation_claims = [
            (kind, sequence, site_ref)
            for op_begin, op_end, kind, sequence, site_ref in operations
            if op_begin < end and op_end > begin
        ]
        site_claims = [
            site
            for site in ordered_sites
            if int(site["begin_cycle"]) < end and int(site["end_cycle"]) > begin
        ]
        previous_site = next(
            (
                site["ref"]
                for site in reversed(ordered_sites)
                if int(site["end_cycle"]) <= begin
            ),
            None,
        )
        next_site = next(
            (
                site["ref"]
                for site in ordered_sites
                if int(site["begin_cycle"]) >= end
            ),
            None,
        )
        reason: str | None = None
        event_kind: str | None = None
        source_event_sequence: int | None = None
        containing_site: Mapping[str, Any] | None = None
        claimant_sites: list[Mapping[str, Any]] = []
        if len(operation_claims) > 1:
            category = "capture-boundary-residual"
            reason = "overlapping-operation-spans"
            primary = relation = "unknown"
            optimization = "inspect overlapping operation instrumentation"
            claimant_sites = [claim[2] for claim in operation_claims]
        elif operation_claims:
            event_kind, source_event_sequence, containing_site = operation_claims[0]
            reason = (
                f"{event_kind}@{_site_label(containing_site)}"
                f"/event-{source_event_sequence}"
            )
            category, primary, relation, optimization = _operation_cost_contract(
                event_kind
            )
        elif len(site_claims) > 1:
            category = "capture-boundary-residual"
            reason = "overlapping-site-spans"
            primary = relation = "unknown"
            optimization = "inspect nested or overlapping target-call sites"
            claimant_sites = [site["ref"] for site in site_claims]
        elif site_claims:
            category = "site-control"
            containing_site = site_claims[0]["ref"]
            source_event_sequence = int(containing_site["instance_sequence"])
            reason = (
                "inside-site-outside-operation@"
                f"{_site_label(containing_site)}"
            )
            primary, relation = "mixed", "mixed/proxy"
            optimization = (
                "inspect this target-call wrapper, descriptor/setup work and "
                "synchronization; use the aggregate Trace-cost overlay to "
                "separate instrumentation"
            )
        elif first_site is None:
            category = "capture-boundary-residual"
            reason = "no-valid-site-boundaries"
            primary = relation = "unknown"
            optimization = "restore site-boundary capture before optimizing"
        elif end <= int(first_site["begin_cycle"]):
            category = "entry-prologue"
            reason = f"before-first-{_site_label(first_site['ref'])}"
            primary, relation = "mixed", "mixed/proxy"
            optimization = (
                "inspect entry ABI/pointer/descriptor preparation; Trace setup "
                "remains separately visible in the non-additive overlay"
            )
        elif begin >= int(last_site["end_cycle"]):
            category = "entry-epilogue"
            reason = f"after-last-{_site_label(last_site['ref'])}"
            primary, relation = "mixed", "mixed/proxy"
            optimization = (
                "inspect final completion, output publication and return; Trace "
                "teardown remains separately visible outside the entry axis"
            )
        else:
            category = "between-site-gap"
            reason = (
                f"between-{_site_label(previous_site)}-and-"
                f"{_site_label(next_site)}"
            )
            primary, relation = "mixed", "mixed/proxy"
            optimization = (
                "inspect the preceding-to-next data/control dependency, join or "
                "wait before deciding whether reordering is legal"
            )
        row = {
            "category": category,
            "reason": reason,
            "event_kind": event_kind,
            "source_event_sequence": source_event_sequence,
            "containing_site": containing_site,
            "previous_site": previous_site,
            "next_site": next_site,
            "claimant_sites": claimant_sites,
            "begin_cycle": begin,
            "end_cycle": end,
            "cycles": end - begin,
            "counts_in_primary_device_elapsed": primary,
            "magnitude_relation": relation,
            "optimization_entry": optimization,
        }
        merge_keys = (
            "category",
            "reason",
            "event_kind",
            "source_event_sequence",
            "containing_site",
            "previous_site",
            "next_site",
            "claimant_sites",
            "counts_in_primary_device_elapsed",
            "magnitude_relation",
            "optimization_entry",
        )
        if (
            result
            and result[-1]["end_cycle"] == begin
            and all(result[-1][key] == row[key] for key in merge_keys)
        ):
            result[-1]["end_cycle"] = end
            result[-1]["cycles"] += end - begin
        else:
            result.append(row)
    return result


def _semantic_cost_rows(
    segments: Sequence[Mapping[str, Any]], total: int
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str | None, str | None, int | None], dict[str, Any]
    ] = {}
    for segment in segments:
        key = (
            str(segment["category"]),
            segment.get("reason"),
            segment.get("event_kind"),
            segment.get("source_event_sequence"),
        )
        row = groups.setdefault(
            key,
            {
                "category": key[0],
                "reason": key[1],
                "event_kind": key[2],
                "source_event_sequence": key[3],
                "containing_site": segment.get("containing_site"),
                "previous_site": segment.get("previous_site"),
                "next_site": segment.get("next_site"),
                "claimant_sites": segment.get("claimant_sites", []),
                "cycles": 0,
                "interval_count": 0,
                "counts_in_primary_device_elapsed": segment[
                    "counts_in_primary_device_elapsed"
                ],
                "magnitude_relation": segment["magnitude_relation"],
                "optimization_entry": segment["optimization_entry"],
                "accounting_scope": "semantic-partition",
            },
        )
        row["cycles"] += int(segment["cycles"])
        row["interval_count"] += 1
    rows = list(groups.values())
    for row in rows:
        row["share_of_trace_entry"] = int(row["cycles"]) / total if total else 0.0
    return sorted(
        rows,
        key=lambda row: (
            -int(row["cycles"]),
            str(row["category"]),
            str(row["reason"]),
        ),
    )


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
    resources = tuple(evidence["output_validation"]["resources"])
    production_validated = all(
        row["production_execution_validated"] for row in resources
    )
    captures_match_primary = all(
        row["diagnostic_captures_match_primary"] for row in resources
    )
    output_equivalence = production_validated and captures_match_primary
    comparisons = [row["external_expected_comparison"] for row in resources]
    any_expected = any(item is not None for item in comparisons)
    all_expected = all(item is not None for item in comparisons)
    all_exact = all(item == "exact" for item in comparisons)
    if not production_validated:
        semantic_correctness: bool | None = False
        correctness_status = "production-validation-failed"
    elif all_expected and all_exact:
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
    device_elapsed = int(sample["device_elapsed_ns"])
    host_submit = int(sample["host_submit_ns"])
    host_envelope = int(sample["host_launch_to_completion_ns"])
    resolution = int(sample["completion_observation_resolution_ns"])
    host_envelope_available = host_envelope > 0
    resolution_fraction = (
        resolution / host_envelope if host_envelope_available else None
    )
    high_resolution = bool(
        host_envelope_available
        and resolution_fraction is not None
        and resolution_fraction <= MAX_COMPLETION_RESOLUTION_FRACTION
    )
    device_zero = device_elapsed == 0
    host_submit_zero = host_submit == 0
    if device_zero:
        diagnose(
            "warning",
            "device_event_elapsed_quantized_zero",
            "the TX stream-event timer returned zero; the launch-to-completion "
            "device envelope "
            "remains measured but is at or below the effective timer resolution",
        )
    if host_submit_zero:
        diagnose(
            "warning",
            "host_submit_quantized_zero",
            "the host provider-submit measurement returned zero and is at or "
            "below the host clock's effective resolution",
        )
    if not host_envelope_available:
        diagnose(
            "warning",
            "host_completion_envelope_unavailable",
            "the host completion envelope is zero/quantized and unavailable; "
            "Primary same-stream device elapsed remains independently usable",
        )
    elif not high_resolution:
        diagnose(
            "warning",
            "completion_resolution_too_coarse",
            "the device stream-event duration remains independent and usable, "
            "but host completion observation is coarse relative to the host "
            "launch-to-completion envelope",
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
    sites = {
        (int(row["tile"]), int(row["site_id"])): row for row in evidence["sites"]
    }
    timeline_events: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    trace_all = bool(experiment["trace"]["complete"])
    accounting_all = True
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
            "Measured"
            if protocol_ok
            else "Invalid"
            if trace_axis is None
            else "Incomplete"
        )
        axis = int(trace_axis or 0)
        site_instances: list[dict[str, Any]] = []
        operation_candidates: list[
            tuple[
                int,
                int,
                str,
                int,
                int | None,
                int | None,
                Mapping[str, Any],
            ]
        ] = []
        tile_events: list[dict[str, Any]] = []
        dte_source_count = 0
        dte_windows_valid = protocol_ok
        current_site_ref: Mapping[str, Any] | None = None

        for event in trace_row["events"]:
            if not protocol_ok:
                continue
            engine = event["engine"]
            kind = str(event["kind"])
            site_id = (
                None if event["site_id"] is None else int(event["site_id"])
            )
            site = (
                sites[(tile, site_id)]
                if site_id is not None
                else {
                    "site_kind": "ncc-completion",
                    "engine": None,
                    "correlation_key": "runtime.ncc-completion",
                    "target_call_ordinal": None,
                    "target_call_symbol": "runtime-completion-phase",
                }
            )
            site_span = (
                _event_span(event, "site", trace_begin, trace_end)
                if protocol_ok
                else None
            )
            operation_span = (
                _event_span(event, "operation", trace_begin, trace_end)
                if protocol_ok
                else None
            )
            observed_span = (
                _event_span(event, "observed", trace_begin, trace_end)
                if protocol_ok
                else None
            )
            scope = (
                f"site:{site_id}:{event['sub_index']}"
                if site_id is not None
                else "runtime:ncc-completion"
            )
            if event["site_span_valid"] and site_span is None:
                trace_all = False
                diagnose(
                    "warning",
                    "site_capture_span_unusable",
                    "site capture span is reversed or outside the trace entry",
                    tile=tile,
                    scope=scope,
                )
            if event["operation_span_valid"] and operation_span is None:
                trace_all = False
                diagnose(
                    "warning",
                    "operation_span_unusable",
                    "operation span is reversed or outside the trace entry",
                    tile=tile,
                    scope=scope,
                )
            if event["observed_span_valid"] and observed_span is None:
                trace_all = False
                diagnose(
                    "warning",
                    "observation_span_unusable",
                    "observation span is reversed or outside the trace entry",
                    tile=tile,
                    scope=scope,
                )
            if kind == "target-site" and site_id is not None:
                current_site_ref = _site_ref(site, int(event["sequence"]))
                if site_span:
                    site_instances.append(
                        {
                            "begin_cycle": site_span[0],
                            "end_cycle": site_span[1],
                            "ref": current_site_ref,
                        }
                    )
            if operation_span:
                if current_site_ref is None:
                    raise AssertionError(
                        "validated operation has no target-site container"
                    )
                operation_candidates.append(
                    (
                        *operation_span,
                        kind,
                        int(event["sequence"]),
                        site_id,
                        (
                            None
                            if event["sub_index"] is None
                            else int(event["sub_index"])
                        ),
                        current_site_ref,
                    )
                )
                if site_span and not (
                    site_span[0] <= operation_span[0]
                    and operation_span[1] <= site_span[1]
                ):
                    trace_all = False
                    diagnose(
                        "warning",
                        "operation_outside_site_span",
                        "operation span is not contained by its site capture span",
                        tile=tile,
                        scope=scope,
                    )

            ncc = kind == "ncc-command"
            dte_event = kind.startswith("direct-dte-")
            dte_wait = kind == "direct-dte-wait"
            target_site = kind == "target-site"
            positive = bool(event["positive_delta"])
            ambiguous = bool(event["attribution_ambiguous"])
            ncc_counter_valid = (
                bool(event["ncc_counter_valid"]) if ncc else None
            )
            if dte_wait:
                dte_source_count += 1
                if operation_span is None:
                    dte_windows_valid = False
            if ncc and not ncc_counter_valid:
                duration_status = counter_status = "Unavailable"
            elif ambiguous:
                duration_status = counter_status = "Attribution ambiguous"
            elif ncc and positive and observed_span:
                duration_status, counter_status = "Bounded", "Measured"
            elif ncc:
                duration_status, counter_status = "Zero-delta marker", "Zero delta"
            elif dte_wait and operation_span:
                duration_status = "Measured"
                counter_status = (
                    "Sampled" if event["dte_counter_valid"] else "Unavailable"
                )
            elif dte_event and operation_span:
                duration_status, counter_status = "Measured", "Unavailable"
            elif target_site and site_span:
                duration_status, counter_status = "Measured", "Unavailable"
            elif operation_span:
                duration_status, counter_status = "Measured", "Unavailable"
            else:
                duration_status = counter_status = "Unavailable"
            if ncc and ncc_counter_valid and positive and not observed_span:
                trace_all = False
                diagnose(
                    "warning",
                    "ncc_activity_window_unusable",
                    "a positive NCC delta has no usable bounded observation span",
                    tile=tile,
                    scope=scope,
                )
            if ncc and not ncc_counter_valid:
                diagnose(
                    "warning",
                    "ncc_event_counter_unavailable",
                    "the exact target site and TsmExecute submit span are "
                    "retained, but this engine event crossed an unstable PMU "
                    "sample and has no counter delta or activity bound",
                    tile=tile,
                    scope=scope,
                )
            if ambiguous:
                diagnose(
                    "warning",
                    "ncc_site_attribution_ambiguous",
                    "same-engine activity overlaps this observation; its delta "
                    "is retained but not assigned as exact site work",
                    tile=tile,
                    scope=scope,
                )

            if ncc and operation_span:
                plot_span = operation_span
                display_interval_role = "command-submit"
            elif operation_span:
                plot_span = operation_span
                display_interval_role = "operation-window"
            elif site_span:
                plot_span = site_span
                display_interval_role = "site-envelope"
            else:
                plot_span = None
                display_interval_role = "marker"
            marker_cycle = plot_span[0] if plot_span else trace_begin
            marker = bool(
                not plot_span
                or ambiguous
                or (ncc and (not ncc_counter_valid or not positive))
            )
            direct_raw = (
                int(event["counter_delta"])
                if dte_wait and event["dte_counter_valid"]
                else None
            )
            row = {
                "tile": tile,
                "sequence": int(event["sequence"]),
                "site_id": site_id,
                "sub_index": (
                    None
                    if event["sub_index"] is None
                    else int(event["sub_index"])
                ),
                "site_kind": site["site_kind"],
                "site_instance_sequence": (
                    None
                    if current_site_ref is None
                    else int(current_site_ref["instance_sequence"])
                ),
                "engine": engine,
                "kind": kind,
                "correlation_key": site["correlation_key"],
                "target_call_ordinal": (
                    None
                    if site["target_call_ordinal"] is None
                    else int(site["target_call_ordinal"])
                ),
                "target_call_symbol": site["target_call_symbol"],
                "position": site.get("position"),
                "observation_status": event["observation_status"],
                "observation_count": int(event["observation_count"]),
                "site_span_declared": bool(event["site_span_valid"]),
                "operation_span_declared": bool(event["operation_span_valid"]),
                "observed_span_declared": bool(event["observed_span_valid"]),
                "ncc_counter_valid": ncc_counter_valid,
                "attribution_ambiguous": ambiguous,
                "zero_delta_marker": bool(
                    ncc
                    and ncc_counter_valid
                    and not positive
                    and not ambiguous
                ),
                "marker": marker,
                "display_interval_role": display_interval_role,
                "marker_cycle_offset": marker_cycle - trace_begin,
                "trace_entry_offset_begin_cpu_cycles": (
                    plot_span[0] - trace_begin if plot_span else marker_cycle - trace_begin
                ),
                "trace_entry_offset_end_cpu_cycles": (
                    plot_span[1] - trace_begin if plot_span else marker_cycle - trace_begin
                ),
                "activity_window_cpu_cycles": (
                    observed_span[1] - observed_span[0] if observed_span else None
                ),
                "activity_trace_entry_offset_begin_cpu_cycles": (
                    observed_span[0] - trace_begin if observed_span else None
                ),
                "activity_trace_entry_offset_end_cpu_cycles": (
                    observed_span[1] - trace_begin if observed_span else None
                ),
                "activity_plot_begin_fraction": (
                    (observed_span[0] - trace_begin) / axis
                    if axis and observed_span
                    else None
                ),
                "activity_plot_end_fraction": (
                    (observed_span[1] - trace_begin) / axis
                    if axis and observed_span
                    else None
                ),
                "activity_window_status": (
                    "Bounded"
                    if ncc
                    and ncc_counter_valid
                    and positive
                    and not ambiguous
                    and observed_span
                    else "Attribution ambiguous"
                    if ncc and ambiguous
                    else "Zero delta"
                    if ncc and ncc_counter_valid and not positive
                    else "Unavailable"
                ),
                "site_capture_window_cpu_cycles": (
                    site_span[1] - site_span[0] if site_span else None
                ),
                "site_capture_status": (
                    "Measured"
                    if site_span
                    else "Invalid"
                    if event["site_span_valid"]
                    else "Unavailable"
                ),
                "operation_window_cpu_cycles": (
                    operation_span[1] - operation_span[0] if operation_span else None
                ),
                "operation_trace_entry_offset_begin_cpu_cycles": (
                    operation_span[0] - trace_begin if operation_span else None
                ),
                "operation_trace_entry_offset_end_cpu_cycles": (
                    operation_span[1] - trace_begin if operation_span else None
                ),
                "operation_window_status": (
                    "Measured"
                    if operation_span
                    else "Invalid"
                    if event["operation_span_valid"]
                    else "Unavailable"
                ),
                "duration_status": duration_status,
                "counter_status": counter_status,
                "timeline_scope": "tile-local",
                "plot_begin_fraction": (
                    (plot_span[0] - trace_begin) / axis if axis and plot_span else 0.0
                ),
                "plot_end_fraction": (
                    (plot_span[1] - trace_begin) / axis if axis and plot_span else 0.0
                ),
                "dte_role": event["dte_role"],
                "ncc_engine_execution_time_ns": (
                    int(event["counter_delta"])
                    if ncc
                    and ncc_counter_valid
                    and positive
                    and not ambiguous
                    else None
                ),
                "direct_dte_wait_cpu_cycles": (
                    operation_span[1] - operation_span[0]
                    if dte_wait and operation_span
                    else None
                ),
                "direct_dte_raw_pmu_activity": direct_raw,
                "direct_dte_raw_pmu_valid": (
                    bool(event["dte_counter_valid"]) if dte_wait else None
                ),
                "worker": event["worker"],
                "wait_scope": event["wait_scope"],
                "engine_lane_visible": ncc or dte_wait,
            }
            timeline_events.append(row)
            tile_events.append(row)
            if dte_wait and not event["dte_counter_valid"]:
                diagnose(
                    "warning",
                    "direct_dte_raw_pmu_unusable",
                    "Direct-DTE operation span is usable, but its uncalibrated "
                    "raw PMU activity is unavailable",
                    tile=tile,
                    scope=scope,
                )

        dte_leaf_intervals = [
            (begin, end)
            for begin, end, kind, _, _, _, _ in operation_candidates
            if kind.startswith("direct-dte-") and kind != "direct-dte-wait"
        ]
        operation_spans: list[
            tuple[int, int, str, int, Mapping[str, Any]]
        ] = []
        for begin, end, kind, sequence, _, _, site_ref in operation_candidates:
            if kind == "direct-dte-wait" and any(
                leaf_begin < end and leaf_end > begin
                for leaf_begin, leaf_end in dte_leaf_intervals
            ):
                continue
            operation_spans.append((begin, end, kind, sequence, site_ref))
        semantic = (
            _semantic_segments(
                trace_begin, trace_end, site_instances, operation_spans
            )
            if trace_axis is not None
            else []
        )
        for segment in semantic:
            segment["plot_begin_fraction"] = (
                (int(segment["begin_cycle"]) - trace_begin) / axis if axis else 0.0
            )
            segment["plot_end_fraction"] = (
                (int(segment["end_cycle"]) - trace_begin) / axis if axis else 0.0
            )
        semantic_cycles = sum(int(row["cycles"]) for row in semantic)
        semantic_valid = trace_axis is not None and semantic_cycles == axis
        if not semantic_valid:
            accounting_all = trace_all = False
            diagnose(
                "error",
                "semantic_partition_not_exclusive",
                "exclusive semantic Kcore partition does not equal the trace entry span",
                tile=tile,
            )

        overlay_rows: list[dict[str, Any]] = []
        for field in COST_SUMMARY_FIELDS:
            cycles = int(trace_row["cost_summary"][field])
            outside_entry = field in (
                "entry_setup_cycles",
                "entry_teardown_cycles",
            )
            primary_relation = "no"
            magnitude_relation = "trace-only"
            if field in (
                "status_poll_cycles",
                "completion_loop_bookkeeping_cycles",
            ):
                optimization_entry = (
                    "Trace replaces production TsmWaitfinish with this sampled "
                    "poll loop; optimize the production completion semantic "
                    "from its proxy row, not from this instrumentation cost"
                )
            else:
                optimization_entry = (
                    "reduce profiler sampling/bookkeeping only; this does "
                    "not optimize the production artifact"
                )
            overlay_rows.append(
                {
                    "category": "trace-run-cost-overlay",
                    "reason": field.removesuffix("_cycles").replace("_", "-"),
                    "cycles": cycles,
                    "share_of_trace_entry": (
                        None
                        if outside_entry
                        else cycles / axis
                        if axis
                        else 0.0
                    ),
                    "counts_in_primary_device_elapsed": primary_relation,
                    "magnitude_relation": magnitude_relation,
                    "optimization_entry": optimization_entry,
                    "accounting_scope": "trace-overhead-overlay",
                    "location_granularity": (
                        "before-entry"
                        if field == "entry_setup_cycles"
                        else "after-entry"
                        if field == "entry_teardown_cycles"
                        else "inside-entry-unpositioned"
                    ),
                    "non_additive_to_semantic_partition": True,
                }
            )
        overlay_cycles = sum(int(row["cycles"]) for row in overlay_rows)
        inside_overlay_cycles = sum(
            int(row["cycles"])
            for row in overlay_rows
            if row["location_granularity"] == "inside-entry-unpositioned"
        )
        outside_overlay_cycles = overlay_cycles - inside_overlay_cycles

        aggregate = pmu[tile]["aggregates"]
        statistics_window, statistics_reason = _counter_value(
            aggregate["statistics_window"]
        )
        statistics_valid = statistics_window is not None
        if not statistics_valid:
            diagnose(
                "warning",
                "statistics_window_unusable",
                statistics_reason or "statistics-window counter is unavailable",
                tile=tile,
                scope="statistics_window",
            )
        engine_rows: list[dict[str, Any]] = []
        for engine in NCC_ENGINES:
            execution, reason = _counter_value(aggregate[engine.lower()])
            valid = execution is not None
            pmu_all = pmu_all and valid
            if not valid:
                diagnose(
                    "warning",
                    "ncc_engine_execution_time_unusable",
                    reason or "NCC execution-time counter is unavailable",
                    tile=tile,
                    scope=engine,
                )
            events = [
                row
                for row in tile_events
                if row["engine"] == engine and row["kind"] == "ncc-command"
            ]
            engine_rows.append(
                {
                    "engine": engine,
                    "measurement_kind": "ncc-engine-execution-time-ns",
                    "engine_execution_time_ns": execution,
                    "engine_execution_time_valid": valid,
                    "engine_execution_time_status": (
                        "Measured" if valid else "Unavailable"
                    ),
                    "activity_window_count": sum(
                        row["duration_status"] == "Bounded" for row in events
                    ),
                    "zero_delta_marker_count": sum(
                        row["zero_delta_marker"] for row in events
                    ),
                    "ambiguous_attribution_count": sum(
                        row["attribution_ambiguous"] for row in events
                    ),
                    "activity_window_status": (
                        "Bounded"
                        if any(row["duration_status"] == "Bounded" for row in events)
                        else "Ambiguous"
                        if any(row["attribution_ambiguous"] for row in events)
                        else "Zero delta"
                        if events
                        else "Incomplete"
                        if not protocol_ok
                        else "Unavailable"
                    ),
                }
            )

        dte_events = [
            row
            for row in tile_events
            if row["kind"] == "direct-dte-wait"
        ]
        raw_values = [row["direct_dte_raw_pmu_activity"] for row in dte_events]
        raw_valid = bool(
            dte_windows_valid
            and dte_events
            and all(item is not None for item in raw_values)
        )
        dte_intervals = sorted(
            (
                int(row["trace_entry_offset_begin_cpu_cycles"]),
                int(row["trace_entry_offset_end_cpu_cycles"]),
            )
            for row in dte_events
            if row["direct_dte_wait_cpu_cycles"] is not None
        )
        dte_union = 0
        union_end = -1
        for begin, end in dte_intervals:
            if end > max(begin, union_end):
                dte_union += end - max(begin, union_end)
            union_end = max(union_end, end)
        engine_rows.append(
            {
                "engine": "DIRECT_DTE",
                "measurement_kind": "direct-dte-wait-completion-windows",
                "wait_window_cpu_cycles": dte_union if dte_windows_valid else None,
                "wait_windows_valid": dte_windows_valid,
                "wait_window_status": (
                    "Measured"
                    if dte_windows_valid
                    else "Invalid"
                    if trace_axis is None
                    else "Incomplete"
                ),
                "wait_window_count": dte_source_count,
                "raw_pmu_activity": (
                    sum(int(item) for item in raw_values) if raw_valid else None
                ),
                "raw_pmu_activity_valid": raw_valid,
                "raw_pmu_activity_status": "Sampled" if raw_valid else "Unavailable",
                "activity_window_count": sum(
                    row["direct_dte_wait_cpu_cycles"] is not None
                    for row in dte_events
                ),
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
                "trace_entry_cpu_cycles": trace_axis,
                "timeline_axis": "kcore-rdcycle-entry-span",
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
                "statistics_window_raw_ticks": statistics_window,
                "statistics_window_valid": statistics_valid,
                "statistics_window_status": (
                    "Measured" if statistics_valid else "Unavailable"
                ),
                "semantic_timeline_segments": semantic,
                "semantic_partition": {
                    "entry_cycles": trace_axis,
                    "exclusive_cycles": semantic_cycles,
                    "exclusive_accounting_valid": semantic_valid,
                    "rows": _semantic_cost_rows(semantic, axis),
                },
                "trace_overhead_overlay": {
                    "exclusive_component_cycles": overlay_cycles,
                    "inside_entry_known_overhead_cycles": inside_overlay_cycles,
                    "outside_entry_overhead_cycles": outside_overlay_cycles,
                    "components_exclusive": True,
                    "non_additive_to_semantic_partition": True,
                    "positioning": "aggregate-only",
                    "coverage": "measured-categories-only",
                    "rows": overlay_rows,
                },
                "engines": engine_rows,
            }
        )

    clock_alignment = all(
        row["valid"] and row["monotonic"] for row in clocks.values()
    )
    site_instance_counts = Counter(
        (int(row["tile"]), int(row["site_id"]))
        for row in timeline_events
        if row["kind"] == "target-site"
    )
    site_capture_statuses: dict[tuple[int, int], set[str]] = {}
    for event in timeline_events:
        if event["kind"] != "target-site" or event["site_id"] is None:
            continue
        site_capture_statuses.setdefault(
            (int(event["tile"]), int(event["site_id"])), set()
        ).add(str(event["site_capture_status"]))
    activity_event_counts = Counter(
        (int(row["tile"]), int(row["site_id"]))
        for row in timeline_events
        if row["kind"] != "target-site"
    )
    statuses: dict[tuple[int, int], set[str | None]] = {}
    for event in timeline_events:
        if event["site_id"] is None:
            continue
        if event["kind"] == "target-site":
            continue
        statuses.setdefault(
            (int(event["tile"]), int(event["site_id"])), set()
        ).add(event["observation_status"])
    site_rows: list[dict[str, Any]] = []
    for (tile, site_id), site in sorted(sites.items()):
        instance_count = site_instance_counts[(tile, site_id)]
        event_count = activity_event_counts[(tile, site_id)]
        site_status = statuses.get((tile, site_id), set())
        capture_status = site_capture_statuses.get((tile, site_id), set())
        observation_status = (
            "Unavailable"
            if "counter-unavailable" in site_status
            else "Attribution ambiguous"
            if "attribution-ambiguous" in site_status
            else "Bounded"
            if "engine-delta-bounded" in site_status
            else "Zero delta"
            if "counter-no-change" in site_status
            else "Measured"
            if event_count
            else "Unavailable"
        )
        site_rows.append(
            {
                "tile": tile,
                "site_id": site_id,
                "site_kind": site["site_kind"],
                "engine": site["engine"],
                "correlation_key": site["correlation_key"],
                "target_call_ordinal": int(site["target_call_ordinal"]),
                "target_call_symbol": site["target_call_symbol"],
                "position": site.get("position"),
                "site_instance_count": instance_count,
                "site_capture_status": (
                    "Invalid"
                    if "Invalid" in capture_status
                    else "Measured"
                    if "Measured" in capture_status
                    else "Unavailable"
                ),
                "event_count": event_count,
                "engine_observation_status": observation_status,
                "observation_status": observation_status,
            }
        )
    dte_events = [
        row
        for row in timeline_events
        if row["kind"] == "direct-dte-wait"
    ]
    engine_active_time = _engine_active_time_summary(tile_rows)
    validity = {
        "identity": True,
        "output_equivalence": output_equivalence,
        "semantic_correctness": semantic_correctness,
        "environment": bool(source_validity["environment"]),
        "package_companion": bool(source_validity["package_companion"]),
        "measurement_basis": bool(source_validity["measurement_basis"]),
        "host_completion_resolution": high_resolution,
        "trace": trace_all,
        "cost_accounting": accounting_all,
        "pmu": pmu_all,
        "clock_alignment": clock_alignment,
    }
    return {
        "schema": ANALYSIS_SCHEMA_NAME,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_id": evidence["run_id"],
        "record_abi": evidence["identity"]["record_abi"],
        "valid": qualified,
        "validity": validity,
        "final_artifact": {
            "artifact_digest": experiment["artifact"]["digest"],
            "target_profile": experiment["artifact"]["target_profile"],
            "duration": {
                "sample_id": "primary",
                "sample_index": 0,
                "measurement_kind": (
                    "tx-stream-kernel-launch-to-completion-envelope"
                ),
                "scope": "production-launch-to-stream-completion",
                "is_engine_only": False,
                "includes_device_control_wait_and_scheduling": True,
                "includes_host_submit": False,
                "includes_host_completion_polling": False,
                "includes_trace_instrumentation": False,
                "device_elapsed_ns": device_elapsed,
                "device_timer_kind": sample["device_timer_kind"],
                "device_elapsed_quantized_zero": device_zero,
                "host_submit_ns": host_submit,
                "host_submit_quantized_zero": host_submit_zero,
                "host_launch_to_completion_ns": host_envelope,
                "host_non_submit_envelope_ns": host_envelope - host_submit,
                "host_envelope_available": host_envelope_available,
                "completion_observation_resolution_ns": resolution,
                "completion_observation_fraction": resolution_fraction,
                "host_completion_high_resolution": high_resolution,
                "qualified": qualified,
                "status": "Measured" if qualified else "Invalid",
            },
            "engine_active_time": engine_active_time,
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
                "direct_dte_event_count": len(dte_events),
                "send_count": sum(row["dte_role"] == "send" for row in dte_events),
                "receive_count": sum(
                    row["dte_role"] == "receive" for row in dte_events
                ),
                "tiles_with_activity": sorted(
                    {int(row["tile"]) for row in dte_events}
                ),
                "timeline_scope": "tile-local",
                "cross_tile_order_available": False,
            },
        },
        "diagnostics": diagnostics,
        "method": {
            "kernel_launch_to_completion": "one uninstrumented Primary production launch measured by same-stream TX start/end events; this launch-to-completion envelope includes device-side dispatch, Kcore control, submits, waits, DTE lifecycle, scheduling gaps, and retirement; it is not engine-only time",
            "host_submit_time": "host steady-clock provider submit path, including provider argument preparation and stream/event setup where applicable; not pure TX API time",
            "host_envelope_time": "host steady-clock first submit through all-rank trusted completion; diagnostic envelope, not kernel execution time",
            "host_envelope_ledger": "host envelope is partitioned only on its own clock into host submit and host non-submit envelope; the latter includes host waiting/polling/phase control concurrent with the Primary TX stream envelope, is not pure overhead, and is never reduced by device elapsed",
            "queue_delay": "unavailable; no queue-delay value is inferred by subtracting host and device measurements",
            "timeline_axis": "tile-local Kcore rdcycle entry span partitioned into named semantic costs; every cycle has an explicit category or reason-specific capture-boundary residual",
            "timeline_scope": "tile-local only; no cross-tile order is inferred",
            "semantic_partition": "exclusive interval-union/subtraction accounting over the trace entry; categories sum exactly once to the entry span",
            "primary_inclusion_labels": "yes means the production semantic phase is inside the Primary device event; mixed means the trace interval combines production-common control with nested trace-only work that is measured only in the separate overhead overlay; unknown means current evidence cannot establish the production correspondence; no is trace-only",
            "trace_overhead_overlay": "eight mutually exclusive Trace-run rdcycle instrumentation components; status-poll and completion-loop bookkeeping belong to the sampled replacement for production TsmWaitfinish, while the production-common wait semantic is represented separately by its proxy operation row; all eight are trace-only, aggregate-positioned, and never added to the semantic partition",
            "engine_active_time": "separate Trace diagnostic-run vendor PMU execution time in nanoseconds for each tile and NCC engine; min/average/max are per-tile distributions and cross-tile sums are work volume only; asynchronous engines/tiles may overlap, so no card-wide engine elapsed is inferred or subtracted from the Primary envelope",
            "statistics_window": "aggregate statistics_window is a raw PMU tick delta, not the rdcycle timeline axis and not converted to elapsed time",
            "ncc_activity_window": "positive per-event deltas carry only bounded observation windows; zero deltas remain markers and ambiguous deltas are not assigned as exact site work",
            "direct_dte_time": "measured operation windows in tile-local Kcore rdcycle, separate from raw PMU activity",
            "direct_dte_raw_pmu": "sampled uncalibrated activity only; never interpreted as elapsed time",
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
.section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin:0 0 10px}.section-head>*{min-width:0}
.section-head h2{font-size:17px;margin:0}.section-head p{margin:2px 0 0;color:var(--muted);font-size:11px}
.grid{display:flex;flex-wrap:wrap;display:grid;gap:10px}.grid>*{min-width:0}.kpis{grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:10px}.kpis>.card{flex:1 1 210px}
.overview-kpis{grid-template-columns:repeat(3,minmax(0,1fr))}
.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.metric-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-weight:700}
.metric-value{font-size:24px;font-weight:760;margin:5px 0 2px;white-space:nowrap}.metric-value.primary{font-size:31px;color:var(--accent)}
.metric-note{font-size:11px;color:var(--muted)}
.status{display:inline-flex;align-items:center;gap:5px;border:1px solid currentColor;border-radius:999px;padding:2px 7px;font-size:10px;font-weight:750;white-space:nowrap}
.status::before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor}
.status-Measured{color:#087a55;background:#edf9f4}.status-Sampled{color:#175cd3;background:#eef4ff}.status-Bounded{color:#6e4fc4;background:#f5f1ff}
.status-Unavailable{color:#667085;background:#f2f4f7}.status-Incomplete{color:#a45b06;background:#fff7e8}.status-Invalid{color:#b42318;background:#fff1f0}
.status-Passed{color:#087a55;background:#edf9f4}.status-Gate-unmet{color:#b42318;background:#fff1f0}.status-Not-assessed{color:#667085;background:#f2f4f7}
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
.ruler{display:grid;grid-template-columns:128px 1fr;gap:8px;align-items:end;margin-bottom:5px}
.ruler-track{position:relative;height:24px;border-bottom:1px solid var(--line-strong)}
.tick{position:absolute;bottom:-1px;height:6px;border-left:1px solid var(--line-strong)}
.tick span{position:absolute;bottom:8px;transform:translateX(-50%);color:var(--muted);font:9px ui-monospace,SFMono-Regular,monospace}
.lane{display:grid;grid-template-columns:128px 1fr;gap:8px;align-items:center;min-height:37px;border-top:1px solid #edf0f3}
.lane-label{display:flex;align-items:center;justify-content:space-between;font-size:11px}.lane-track{position:relative;height:20px;background:#f6f8fa;border-left:1px solid var(--line);border-right:1px solid var(--line)}
.lane-gridline{position:absolute;inset:0 auto 0 0;border-left:1px solid #e7ebef;pointer-events:none}
.lane-track{height:24px}
.event,.observation-bound{--event-stroke:#475467;--event-fill-0:#f2f4f7;--event-fill-1:#e4e7ec;--event-fill-2:#f8fafc;--event-solid-0:#475467;--event-solid-1:#344054;--event-solid-2:#667085}
.event-CT,.observation-CT{--event-stroke:var(--ct);--event-fill-0:#dbeafe;--event-fill-1:#bfdbfe;--event-fill-2:#eff6ff;--event-solid-0:#2563eb;--event-solid-1:#1d4ed8;--event-solid-2:#60a5fa}
.event-NE,.observation-NE{--event-stroke:var(--ne);--event-fill-0:#ede9fe;--event-fill-1:#ddd6fe;--event-fill-2:#f5f3ff;--event-solid-0:#7c3aed;--event-solid-1:#6d28d9;--event-solid-2:#a78bfa}
.event-RDMA,.observation-RDMA{--event-stroke:var(--rdma);--event-fill-0:#ccfbf1;--event-fill-1:#99f6e4;--event-fill-2:#f0fdfa;--event-solid-0:#07857e;--event-solid-1:#0f766e;--event-solid-2:#2dd4bf}
.event-WDMA,.observation-WDMA{--event-stroke:var(--wdma);--event-fill-0:#dcfce7;--event-fill-1:#bbf7d0;--event-fill-2:#f0fdf4;--event-solid-0:#368147;--event-solid-1:#166534;--event-solid-2:#4ade80}
.event-TDMA,.observation-TDMA{--event-stroke:var(--tdma);--event-fill-0:#ffedd5;--event-fill-1:#fed7aa;--event-fill-2:#fff7ed;--event-solid-0:#c4600c;--event-solid-1:#9a3412;--event-solid-2:#fb923c}
.event-DIRECT_DTE,.observation-DIRECT_DTE{--event-stroke:var(--dte);--event-fill-0:#e4e7ec;--event-fill-1:#d0d5dd;--event-fill-2:#f2f4f7;--event-solid-0:#475467;--event-solid-1:#344054;--event-solid-2:#667085}
.event{position:absolute;top:8px;height:8px;min-width:3px;border:1px solid #fff;border-radius:3px;cursor:pointer;z-index:2;box-shadow:0 0 0 1px var(--event-stroke)}
.event-v0{background:var(--event-solid-0)}.event-v1{background:var(--event-solid-1)}.event-v2{background:var(--event-solid-2)}
.observation-bound{position:absolute;top:2px;height:20px;min-width:3px;border:1px dashed var(--event-stroke);border-radius:4px;z-index:1;pointer-events:none;opacity:.78}
.observation-bound.event-v0{background:var(--event-fill-0)}.observation-bound.event-v1{background:var(--event-fill-1)}.observation-bound.event-v2{background:var(--event-fill-2)}
.observation-bound.ambiguous{border-color:#a45b06;background:#fff7e8}.observation-bound.zero{border-color:#667085;background:#f2f4f7}
.event:hover,.event.selected{outline:2px solid #17212b;outline-offset:1px;z-index:2}
.event.marker{width:3px!important;min-width:3px;border-radius:0;box-shadow:0 0 0 1px #fff,0 0 0 2px currentColor}
.event.ambiguous{box-shadow:0 0 0 2px #a45b06}.event.zero{box-shadow:0 0 0 2px #667085}
.timeline-key{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 9px;color:var(--muted);font-size:11px}.timeline-key span{display:inline-flex;align-items:center;gap:6px}.key-swatch{display:inline-block;width:28px;height:10px;border-radius:3px}.key-submit{background:var(--ct);box-shadow:0 0 0 1px #1d4ed8}.key-bound{height:16px;background:#dbeafe;border:1px dashed var(--ct)}.key-pmu{width:auto;height:auto;font-weight:700;color:var(--ink)}
.observation-notice[hidden]{display:none}
.cost-segment{position:absolute;inset:2px auto 2px 0;border:0;border-radius:2px;cursor:pointer;box-shadow:inset 0 0 0 1px rgba(0,0,0,.08)}
.cost-segment:hover,.cost-segment.selected{outline:2px solid #17212b;outline-offset:1px;z-index:3}
.cost-ncc-submit{background:#8fb7f7}.cost-completion-wait-proxy{background:#d8b4fe}.cost-dte-peer-ready-wait{background:#b9ddd8}.cost-dte-setup-issue{background:#94cec6}.cost-dte-completion-wait{background:#69b8ad}.cost-dte-cleanup{background:#a7d7d0}.cost-dte-wait-aggregate-fallback{background:#4c9c92}.cost-site-control{background:#d7dee7}.cost-between-site-gap{background:#eef1f4}.cost-entry-prologue,.cost-entry-epilogue{background:#f3f5f7}
.cost-capture-boundary-residual{background-color:#fff2d6;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8'%3E%3Cpath d='M-2 8L8-2M2 10L10 2' stroke='%23b26a00' stroke-width='1'/%3E%3C/svg%3E")}
.trace-cost{background:#f4b7ad}.lane-production{background:#fbfcfd}.lane-separator{border-top:2px solid var(--line-strong)}
.overlay-stack{display:flex;min-height:28px;margin:2px 0 10px;border:1px solid var(--line);border-radius:5px;overflow:hidden;background:var(--soft)}.overlay-part{min-width:3px;border:0;border-right:1px solid rgba(255,255,255,.8);background:#f4b7ad;cursor:pointer}.overlay-part:nth-child(even){background:#e99387}.overlay-part:hover,.overlay-part.selected{outline:2px solid #17212b;outline-offset:-2px;z-index:2}
.cost-tables{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:10px;margin-top:10px}.cost-tables>*{min-width:0}.cost-tables .table-wrap{max-width:100%;max-height:310px}
.detail{min-height:210px}.detail h3{font-size:13px;margin:0 0 10px}.kv{display:grid;grid-template-columns:110px 1fr;gap:5px 8px;font-size:11px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
table{width:100%;border-collapse:collapse;background:var(--panel);font-size:11px}th{position:sticky;top:0;background:#f6f8fa;color:#5a6675;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.05em}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.interactive-row{cursor:pointer}.interactive-row:hover,.interactive-row:focus{background:var(--accent-soft);outline:none}.interactive-row:focus-visible{box-shadow:inset 3px 0 var(--accent)}
.table-wrap{overflow:auto;max-height:calc(100vh - 180px);border:1px solid var(--line);border-radius:9px}.empty{padding:28px;text-align:center;color:var(--muted)}
.communication-table{min-width:1080px}
.evidence-role{display:inline-flex;border:1px solid var(--line-strong);border-radius:999px;padding:2px 7px;background:var(--soft);color:var(--muted);font-size:10px;font-weight:700;white-space:nowrap}
.diag{display:grid;grid-template-columns:74px 220px 70px minmax(0,1fr);gap:8px;padding:8px;border-bottom:1px solid var(--line);font-size:11px}.diag>*{min-width:0;overflow-wrap:anywhere}.diag:last-child{border:0}.severity-error{color:var(--bad)}.severity-warning{color:var(--warn)}
.raw-tabs{display:flex;gap:5px;margin:12px 0 7px}.raw-tabs button{border:1px solid var(--line);border-radius:6px;background:#fff;padding:5px 8px;cursor:pointer}.raw-tabs button.active{background:var(--accent);border-color:var(--accent);color:#fff}
pre{margin:0;max-height:520px;overflow:auto;background:#111827;color:#dbe7f5;border-radius:8px;padding:12px;font:10px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
.notice{border-left:3px solid var(--accent);background:var(--accent-soft);padding:8px 10px;color:#344054;font-size:11px;margin-bottom:9px}
@supports(display:grid){.split>*:last-child{margin-left:0}.tile-card{margin:0}}
@media(max-width:1050px){.sidebar{flex-basis:220px;width:220px}.kpis,.overview-kpis{grid-template-columns:repeat(2,1fr)}.split,.cost-tables{grid-template-columns:1fr;flex-direction:column}.split>*:last-child{width:100%;margin:10px 0 0}}
@media(max-width:720px){.shell{display:block}.sidebar{position:static;width:auto;height:auto}.tree{display:none}.main{padding:14px}.nav{grid-template-columns:repeat(2,1fr)}.kpis{grid-template-columns:1fr}.tile-grid{grid-template-columns:repeat(2,1fr)}.tile-card{flex-basis:calc(50% - 6px)}.topbar{display:block}.artifact-links{margin-top:10px}.diag{grid-template-columns:64px minmax(0,1fr);gap:3px 8px}.diag>:nth-child(3){grid-column:1}.diag>:nth-child(4){grid-column:2}}
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
      <button data-view="glossary">术语说明</button>
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
      <div class="section-head"><div><h2>Overview</h2><p>单次Primary production launch；kernel包络、Host诊断和Trace PMU engine active time分域展示。</p></div></div>
      <div id="statusLegend" class="legend"></div>
      <div class="metric-note status-glossary"><b>Measured</b>：边界和值均由对应计时源直接取得；<b>Sampled</b>：采样 counter；<b>Bounded</b>：只知道活动发生在保守观测窗内，不代表整段持续 busy；<b>Unavailable / Incomplete / Invalid</b>：分别表示该局部证据缺失、采集未闭合或协议不合法。</div>
      <div class="grid kpis overview-kpis">
        <div class="card"><div class="metric-label">Kernel launch → completion</div><div id="deviceDuration" class="metric-value primary"></div><div id="deviceDurationNote" class="metric-note"></div></div>
        <div class="card"><div class="metric-label">NCC engine active work / Tile</div><div id="engineActiveRange" class="metric-value"></div><div id="engineActiveRangeNote" class="metric-note"></div></div>
        <div class="card"><div class="metric-label">Host submit</div><div id="hostSubmitDuration" class="metric-value"></div><div id="hostSubmitNote" class="metric-note"></div></div>
        <div class="card"><div class="metric-label">Host launch → trusted completion</div><div id="hostEnvelopeDuration" class="metric-value"></div><div id="hostEnvelopeResolution" class="metric-note"></div></div>
        <div class="card"><div class="metric-label">Measurement status</div><div id="measurementStatus" class="metric-value"></div><div class="metric-note">Uninstrumented primary · device event timing</div></div>
        <div class="card"><div class="metric-label">Output validation</div><div id="outputStatus" class="metric-value"></div><div id="outputNote" class="metric-note"></div></div>
        <div class="card"><div class="metric-label">Trace coverage</div><div id="traceCoverage" class="metric-value"></div><div id="traceNote" class="metric-note"></div></div>
      </div>
      <div class="card" style="margin-bottom:10px"><div class="section-head"><div><h2>Timing domains · not additive</h2><p>TX stream包络、Host submit与Host completion使用不同边界；不通过相减虚构queue delay或engine-only时间。</p></div></div><div class="table-wrap" style="max-height:none"><table><thead><tr><th>Scope</th><th>Clock / source</th><th class="num">Duration</th><th class="num">Nanoseconds</th><th>Status</th><th>Interpretation</th></tr></thead><tbody id="timingRows"></tbody></table></div></div>
      <div class="split">
        <div class="card"><div class="section-head"><div><h2>Card 0 · Tile map</h2><p>点击 tile 进入本地 engine timeline。</p></div></div><div id="overviewTiles" class="tile-grid"></div></div>
        <div class="card"><div class="section-head"><div><h2>Measurement contract</h2></div></div><div id="methodList" class="method-list"></div></div>
      </div>
      <div class="card" style="margin-top:10px"><div class="section-head"><div><h2>NCC engine active time · Trace PMU</h2><p>五类NCC engine均为另一轮Trace diagnostic的per-tile PMU ns。最小/平均/最大用于看Tile分布；Σ仅是work volume，不是wall time。Direct-DTE没有calibrated engine ns，单独在Communication展示。</p></div></div><div class="table-wrap" style="max-height:none"><table><thead><tr><th>Engine</th><th>Source / unit</th><th class="num">Min / Tile</th><th class="num">Avg / Tile</th><th class="num">Max / Tile</th><th class="num">Active Tiles</th><th class="num">Available Tiles</th><th class="num">Σ Tile Work</th><th>Status</th></tr></thead><tbody id="overviewEngineRows"></tbody></table></div></div>
    </section>

    <section id="view-timeline" class="view" data-view-panel="timeline">
      <div class="section-head"><div><h2>Trace Timeline</h2><p>Trace-run Kcore ledger 与 engine evidence 分层；横轴是所选 tile 的 rdcycle entry span。</p></div></div>
      <div class="notice">Trace-run Kcore ledger 对 entry span 做排他 interval union/subtraction；它仍包含嵌套的 trace instrumentation，因此不能直接相减得到 Primary 各项成本。In Primary 的 yes 表示同一语义阶段存在于 Primary，mixed 表示 production control 与无法定位的 Trace-only 工作混合，unknown 表示当前证据不足，no 表示纯插桩。Primary TX stream launch envelope包含设备侧控制、等待和真实idle，但不含Trace-only instrumentation。Trace-run cost overlay只在独立aggregate bar中显示，绝不伪造时间位置或与ledger相加；NCC engine ns来自另一轮Trace，只是异步work volume。Engine lane的空背景只表示当前没有observation window，不等于engine idle，也不用于补算Primary。</div>
      <div class="timeline-key" aria-label="Timeline interval legend">
        <span><i class="key-swatch key-submit"></i><b>实心块</b>：精确 command submit / DTE operation 区间</span>
        <span><i class="key-swatch key-bound"></i><b>浅色虚线框</b>：PMU 活动保守观测范围，不是持续 busy</span>
        <span><i class="key-swatch key-pmu">ns</i><b>Engine work</b>：PMU 测得耗时，但没有精确起止位置</span>
      </div>
      <div id="timelineOverlapNotice" class="notice observation-notice" hidden></div>
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
      <div class="cost-tables">
        <div class="card"><div class="section-head"><div><h2>Trace-run Kcore ledger</h2><p>排他分区，合计必须等于 trace entry span；不是 Primary 的可减成本表。</p></div></div><div class="table-wrap"><table><thead><tr><th>Cost / reason</th><th class="num">Cycles</th><th class="num">Share</th><th>In Primary</th><th>Representativeness</th><th>Optimization entry</th></tr></thead><tbody id="semanticCostRows"></tbody></table></div></div>
        <div class="card"><div class="section-head"><div><h2>Trace-run cost overlay</h2><p>已测到的互斥成本类别；与 ledger 嵌套且不可相加，也不伪造它们在 entry 内的精确位置。</p></div></div><div class="metric-note">Inside entry · aggregate / not positioned</div><div id="traceOverheadBar" class="overlay-stack" aria-label="Inside-entry aggregate Trace-run cost overlay breakdown"></div><div class="metric-note">Outside entry · setup / teardown</div><div id="traceOutsideBar" class="overlay-stack" aria-label="Outside-entry Trace-run cost overlay breakdown"></div><div class="table-wrap"><table><thead><tr><th>Cost component</th><th class="num">Cycles</th><th class="num">Entry share</th><th>Location</th><th>In Primary</th></tr></thead><tbody id="traceCostRows"></tbody></table></div></div>
      </div>
    </section>

    <section id="view-engines" class="view" data-view-panel="engines">
      <div class="section-head"><div><h2>Tile / Engine</h2><p>Per-tile NCC engine active time来自Trace PMU ns；activity window只作Kcore rdcycle Bounded证据，二者都不是Primary包络。</p></div></div>
      <div class="toolbar"><label>Tile <select id="engineTile"></select></label><span id="engineTileMeta"></span></div>
      <div class="table-wrap"><table><thead><tr><th>Engine</th><th>Metric</th><th class="num">ns / CPU cycles / raw</th><th class="num">Windows</th><th>Status</th><th>Interpretation</th></tr></thead><tbody id="engineRows"></tbody></table></div>
    </section>

    <section id="view-sites" class="view" data-view-panel="sites">
      <div class="section-head"><div><h2>Program / Sites</h2><p>从 trace event 下钻到 compiler-published correlation identity。</p></div></div>
      <div class="toolbar"><label>Search <input id="siteSearch" type="search" placeholder="symbol, correlation, position"></label><label>Engine <select id="siteEngine"><option value="">All engines</option></select></label><span id="siteCount" class="spacer"></span></div>
      <div class="table-wrap"><table><thead><tr><th>Tile</th><th>Site</th><th>Kind</th><th>Engine</th><th>Correlation</th><th>Target call</th><th>Position</th><th class="num">Instances</th><th>Site capture</th><th class="num">Activity events</th><th>Engine / phase evidence</th></tr></thead><tbody id="siteRows"></tbody></table></div>
    </section>

    <section id="view-communication" class="view" data-view-panel="communication">
      <div class="section-head"><div><h2>Communication / Direct-DTE</h2><p>整次通信等待与等待对端、配置发起、等待完成、收尾等内部步骤分开显示。</p></div></div>
      <div class="notice">每一行只属于自身 tile-local clock domain；不同 tile 的位置和 duration 不组成全局通信序列。“整次通信等待（总计）”和“通信内部步骤”是同一件事的两种粒度，不能重复相加；未校准 raw PMU 只记录在总计行。</div>
      <div id="communicationSummary" class="grid kpis"></div>
      <div class="table-wrap"><table class="communication-table"><thead><tr><th>Tile</th><th>Role</th><th>Phase / kind</th><th>Evidence role</th><th>Site / symbol</th><th class="num">Operation CPU cycles</th><th>Duration status</th><th class="num">Raw PMU</th><th>Raw status</th></tr></thead><tbody id="communicationRows"></tbody></table></div>
    </section>

    <section id="view-glossary" class="view" data-view-panel="glossary">
      <div class="section-head"><div><h2>术语说明</h2><p>界面使用人话名称，机器字段保持不变，便于与 analysis.json 对照。</p></div></div>
      <div class="grid kpis">
        <div class="card"><div class="metric-label">Kernel launch → completion</div><div class="metric-note">未插桩最终产物在同一TX stream event pair之间的设备包络，单位ns；包含device control/wait/engine/retirement，不是纯engine时间。</div></div>
        <div class="card"><div class="metric-label">Command submit</div><div class="metric-note">Kcore 调用 TsmExecute 的精确提交区间，单位 rdcycle；不是 engine busy duration。</div></div>
        <div class="card"><div class="metric-label">PMU activity bound</div><div class="metric-note">只证明 counter 增量发生在保守窗口内；窗口重叠不证明两个 engine 同时执行。</div></div>
        <div class="card"><div class="metric-label">Engine active time · Trace PMU</div><div class="metric-note">另一轮Trace diagnostic的vendor PMU execution-time delta，单位ns；是per-tile/per-engine active work，没有精确起止坐标，也不是Primary的同次分项。</div></div>
      </div>
      <div class="notice">读图顺序：先看实心 submit / operation 块，再看浅色虚线 PMU bound，最后到 Tile / Engine 查看 measured ns。三者不能互相替代，也不能把多个 engine 的 ns 相加成 Primary wall time。</div>
      <div class="table-wrap" style="max-height:none"><table><thead><tr><th>分组</th><th>界面名称</th><th>机器字段</th><th>具体指什么</th><th>单位 / Primary</th><th>不能这样理解</th><th>怎么看 / 优化入口</th></tr></thead><tbody id="glossaryRows"></tbody></table></div>
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
const TERMS={
  semantic:{
    "ncc-submit":{label:"NCC 指令提交",definition:"Kcore 调用 TsmExecute 的精确提交区间。",not:"不是 engine 真正执行区间，也不是 PMU execution ns。"},
    "completion-wait-proxy":{label:"等待 NCC 完成（Trace 代理）",definition:"Trace clone 中等待 NCC completion 的 measured operation 区间。",not:"数值受 Trace 采集方式影响，不能直接从 Primary 扣除。"},
    "dte-peer-ready-wait":{label:"等待 DTE 对端就绪",definition:"Direct-DTE 发送前等待接收端 ready 的 tile-local operation 区间。",not:"不是全卡同步时间，也不能跨 tile 比较先后。"},
    "dte-setup-issue":{label:"DTE 配置与发起",definition:"准备 Direct-DTE 描述符并发起传输的 tile-local operation 区间。",not:"不是完整传输耗时。"},
    "dte-completion-wait":{label:"等待 DTE 完成",definition:"等待 Direct-DTE completion 的 tile-local operation 区间。",not:"不是 calibrated DTE engine busy time。"},
    "dte-cleanup":{label:"DTE 收尾",definition:"Direct-DTE 完成后的状态清理与资源释放区间。",not:"不是传输 payload 时间。"},
    "dte-wait-aggregate-fallback":{label:"整次 DTE 通信等待（缺少分步数据时使用）",definition:"没有可用内部步骤时，用整次 Direct-DTE wait operation 表示这次通信等待。",not:"一旦已有内部步骤，就不能再把这行与它们重复相加。"},
    "site-control":{label:"调用点内控制与准备",definition:"一个动态 target-call site 包络内，扣除已识别 operation 后的剩余区间；可能包含 wrapper、descriptor、同步和 Trace 记录。",not:"不是纯硬件控制、纯软件开销或 engine idle。"},
    "between-site-gap":{label:"调用点之间的控制/等待",definition:"相邻动态 target-call site 之间的 Trace 区间，详情保留前后 site。",not:"不能直接称为硬件空闲；可能混有依赖、等待、控制和插桩。"},
    "entry-prologue":{label:"入口准备区间",definition:"Trace entry 起点到首个动态 site 之前的 entry-local 区间。",not:"不同于 entry-setup；后者是 entry 轴外的纯 Trace 初始化。"},
    "entry-epilogue":{label:"结束收尾区间",definition:"最后一个动态 site 结束到 Trace entry 返回之间的区间。",not:"不同于 entry-teardown；后者是 entry 轴外的纯 Trace 收尾。"},
    "capture-boundary-residual":{label:"未解析的采集边界区间",definition:"只有边界不完整或区间关系非法时才产生的保守 residual。",not:"不是正常 idle，也不应出现在有效 exclusive partition 中。"}
  },
  trace:{
    "ncc-pmu-sample":{label:"NCC PMU 采样",definition:"读取 NCC engine PMU counter 的 Trace-only 成本。",not:"Primary 不包含，不能与语义 ledger 相加。"},
    "dte-pmu-sample":{label:"DTE PMU 采样",definition:"启用、读取或恢复 Direct-DTE PMU 的 Trace-only 成本。",not:"Primary 不包含，也不是 DTE 传输耗时。"},
    "event-bookkeeping":{label:"Trace 事件记录",definition:"写入 profiler event record 和维护序号的 Trace-only 成本。",not:"不是被测 target-call 自身耗时。"},
    "status-poll":{label:"完成状态轮询",definition:"Trace completion loop 读取任务状态的采样成本。",not:"不能冒充 production completion wait；后者在语义 ledger 单列。"},
    "site-hook":{label:"调用点钩子",definition:"进入/退出 TARGET_SITE 记录边界的 Trace-only 成本。",not:"Primary 不包含。"},
    "completion-loop-bookkeeping":{label:"完成循环记录",definition:"Trace completion loop 中除状态读取外的记录与控制成本。",not:"不能与 completion-wait-proxy 重复计算。"},
    "entry-setup":{label:"Trace 入口初始化",definition:"Trace entry 时间轴开始前初始化 record/PMU 的成本。",not:"在 entry 轴外，不是 entry-prologue。"},
    "entry-teardown":{label:"Trace 退出收尾",definition:"Trace entry 时间轴结束后恢复状态和封口 record 的成本。",not:"在 entry 轴外，不是 entry-epilogue。"}
  },
  status:{
    "Measured":{label:"已测得",definition:"边界和值由对应计时源直接取得。",not:"仍需结合该字段的计时源和单位解释。"},
    "Sampled":{label:"采样值",definition:"来自 PMU 或其它离散采样 counter。",not:"不自动等于 elapsed time。"},
    "Bounded":{label:"仅确定活动范围",definition:"只知道正增量发生在保守观测窗口内。",not:"窗口长度不等于持续 busy；两个窗口重叠不证明 engine 同时执行。"},
    "Derived":{label:"派生值",definition:"由已验证字段确定性计算得到。",not:"不是独立硬件计时源。"},
    "Zero delta":{label:"计数未变化",definition:"本次 PMU 观测没有检测到 counter 增量。",not:"不等于证明 engine 从未活动。"},
    "Zero-delta marker":{label:"计数未变化标记",definition:"保留一次没有正增量的 NCC 观测位置。",not:"不是零耗时 operation。"},
    "Ambiguous":{label:"聚合归属混杂",definition:"这个 engine 的聚合活动窗口包含无法唯一归属的事件。",not:"不能把聚合窗口数量或宽度当作精确 engine busy。"},
    "Attribution ambiguous":{label:"归属不唯一",definition:"同 engine outstanding activity 无法唯一归到当前 site。",not:"不能作为精确 site work。"},
    "Passed":{label:"通过",definition:"这项资格检查已经满足。",not:"只代表这一项，不替代其它检查。"},
    "Gate unmet":{label:"门禁未满足",definition:"这项资格要求没有满足，因此不能据此形成完整性能结论。",not:"不一定表示协议损坏；应查看该检查的具体定义。"},
    "Not assessed":{label:"尚未独立判定",definition:"当前证据没有对这项语义做独立判定。",not:"不是失败，也不是已经证明正确。"},
    "Unavailable":{label:"无可用证据",definition:"该局部字段没有可用计时或 counter 证据。",not:"不会自动抹掉同一事件其它有效字段。"},
    "Incomplete":{label:"采集未闭合",definition:"采集生命周期、容量或 terminal 状态没有完整闭合。",not:"不能作为完整 profile。"},
    "Invalid":{label:"协议无效",definition:"字段关系或协议约束不合法。",not:"不能用于性能结论。"}
  },
  accounting:{
    "yes":{label:"生产执行包含该语义",definition:"同一语义阶段位于未插桩 Primary device event pair 内。",not:"Trace 数值仍不等于 Primary 中该阶段的精确耗时。"},
    "mixed":{label:"生产控制与 Trace 影响混合",definition:"区间包含 production control，也可能混有当前无法定位的 Trace 影响。",not:"不能直接从 Primary 扣除。"},
    "unknown":{label:"是否计入尚不可判定",definition:"当前证据不足以判断与 Primary 的关系。",not:"不能当作零或 idle。"},
    "no":{label:"仅 Trace 插桩",definition:"该成本只由 Count/Trace 诊断执行产生。",not:"不计入未插桩 Primary。"}
  },
  magnitude:{
    "proxy":{label:"Trace 代理值",definition:"语义对应 Primary，但数值来自另一轮 Trace clone。",not:"不是 Primary 的精确分项。"},
    "mixed/proxy":{label:"混合代理值",definition:"数值来自 Trace clone，且 production 与插桩影响无法完全剥离。",not:"不能解释成单一硬件成本。"},
    "unknown":{label:"代表关系未知",definition:"当前边界或归属不完整，无法说明该数值怎样代表 production execution。",not:"不能当作零、Primary 分项或 Trace-only 成本。"},
    "trace-only":{label:"仅插桩值",definition:"只描述 Trace instrumentation 本身。",not:"不计入 Primary，也不能与语义 ledger 相加。"}
  },
  interval:{
    "command-submit":{label:"精确指令提交区间",definition:"Kcore 进入到返回 TsmExecute 的 measured rdcycle 区间。",not:"不是 engine 真正执行起止。"},
    "operation-window":{label:"精确 operation 区间",definition:"对应 runtime operation 的 measured tile-local rdcycle 区间。",not:"不能跨 tile 建立全局顺序。"},
    "observation-bound":{label:"PMU 活动保守范围",definition:"PMU positive delta 的保守采样包络。",not:"不是持续 busy；与其它范围重叠不证明并行。"},
    "site-envelope":{label:"调用点包络",definition:"一个动态 TARGET_SITE 的 measured begin/end。",not:"包含 operation 与调用点内其它工作。"},
    "marker":{label:"观测标记",definition:"没有可画区间时保留的事件位置。",not:"不是 duration。"}
  },
  location:{
    "inside-entry-unpositioned":{label:"入口轴内，未定位",definition:"成本在 entry 内测得，但没有足够证据放到某个具体区间。",not:"不能从任意 gap 或 site-control 中扣除。"},
    "before-entry":{label:"入口轴之前",definition:"发生在 Trace entry 时间轴开始之前。",not:"不是 entry-prologue。"},
    "after-entry":{label:"入口轴之后",definition:"发生在 Trace entry 时间轴结束之后。",not:"不是 entry-epilogue。"}
  },
  event:{
    "ncc-command":{label:"NCC engine 指令提交事件",definition:"一次已关联 target-call 的 TsmExecute 提交，以及独立的 PMU 观测证据。",not:"事件条的提交区间不是 engine busy 起止；PMU bound 重叠也不证明并行。"},
    "ncc-completion-wait":{label:"NCC 完成等待事件",definition:"Trace clone 中等待 NCC completion 的 operation 事件。",not:"不是某一个 engine 的执行时长，也不是 Host 等待。"},
    "direct-dte-wait":{label:"整次 Direct-DTE 通信等待",definition:"从这次 Direct-DTE 等待开始到结束的总计行，可携带未校准 raw PMU。",not:"不能再与同一次通信的内部步骤重复相加。"},
    "direct-dte-peer-ready-wait":{label:"等待 DTE 对端就绪",definition:"通信内部先等待接收端 ready 的一步。",not:"不是全卡 barrier，也不能跨 tile 排序。"},
    "direct-dte-setup-issue":{label:"配置并发起 DTE",definition:"通信内部配置描述符并发起传输的一步。",not:"不是完整通信耗时。"},
    "direct-dte-completion-wait":{label:"等待 DTE 传输完成",definition:"通信内部等待 Direct-DTE completion 的一步。",not:"不是已校准的 DTE engine busy time。"},
    "direct-dte-cleanup":{label:"DTE 收尾事件",definition:"Direct-DTE completion 后的状态和资源清理 operation。",not:"不是 payload 传输区间。"},
    "target-site":{label:"动态 target-call 调用点包络",definition:"一次静态 target-call site 的动态进入/退出容器，用于关联内部 operation。",not:"不是额外 engine 工作，也不能与内部 operation 再相加。"}
  },
  site:{
    "ncc-command":{label:"NCC 指令调用点",definition:"会向 CT/NE/RDMA/WDMA/TDMA 之一提交 TsmExecute 的静态 target-call site。",not:"site 包络不是对应 engine 的持续执行时间。"},
    "ncc-completion":{label:"NCC 完成调用点",definition:"观察或等待 NCC completion 的静态 target-call site。",not:"不是多 tile barrier，也不专属于某一个 engine。"},
    "direct-dte-control":{label:"Direct-DTE 控制调用点",definition:"Direct-DTE ready、配置、发起或清理类控制 operation 的静态 site。",not:"不是整次通信等待。"},
    "direct-dte-wait":{label:"Direct-DTE 通信等待调用点",definition:"承载一次整次 Direct-DTE 通信等待及其内部步骤的静态 site。",not:"总计与内部步骤不能重复求和。"}
  },
  engine:{
    "CT":{label:"CT / CGRA 执行部件",definition:"执行算术、逻辑、激活、转换、reduce/pool 及部分 data-move/peripheral packet 的 NCC engine。",not:"CT PMU ns 没有精确起止坐标，也不能与其它 engine ns 相加成 Primary。"},
    "NE":{label:"NE 神经网络执行部件",definition:"执行 Conv、DepthwiseConv、GEMM 等 NE packet 的 NCC engine。",not:"NE PMU ns 不是 tile-local timeline 区间。"},
    "RDMA":{label:"RDMA：DDR / 外部地址到 SPM",definition:"LSU 中把外部或 DDR 数据读入 SPM 的 NCC movement engine。",not:"独立 queue 不等于必然与其它 engine 并行。"},
    "WDMA":{label:"WDMA：SPM 到 DDR / 外部地址",definition:"LSU 中把 SPM 数据写回外部或 DDR 的 NCC movement engine。",not:"与 CT 共用或冲突的 SPM 资源不能由 PMU bound 重叠判断。"},
    "TDMA":{label:"TDMA：SPM 内搬运 / 变换",definition:"LSU 中执行 local move、mirror、pad、img2col、gatherscatter、memset 等 packet 的 NCC engine。",not:"TDMA 名称不涵盖所有历史 data-move helper。"},
    "DIRECT_DTE":{label:"Direct-DTE 通信路径",definition:"跨 tile / 节点 Direct-DTE 通信的 runtime operation lane。",meta:"operation 使用 tile-local Kcore rdcycle；raw PMU 未校准且不是 elapsed time；是否属于 Primary 由对应成本行说明。",not:"raw PMU activity 不是 ns/cycles，也不能与 NCC PMU ns 比较。"}
  },
  role:{
    "send":{label:"发送侧",definition:"当前 tile 在这次 Direct-DTE operation 中承担发送角色。",not:"不提供跨 tile 的绝对先后关系。"},
    "receive":{label:"接收侧",definition:"当前 tile 在这次 Direct-DTE operation 中承担接收角色。",not:"不表示接收端全程 busy。"}
  },
  evidence:{
    "aggregate":{label:"整次通信等待（总计）",definition:"覆盖一次完整 Direct-DTE wait，从开始等待到结束，并承载只在总计行记录的 raw PMU。",not:"不能与等待对端、配置发起、等待完成、收尾这些内部步骤求和。"},
    "leaf":{label:"通信内部步骤",definition:"整次 Direct-DTE 通信等待中的一个具体步骤。",not:"单个步骤不是完整通信耗时。"}
  },
  reason:{
    "overlapping-operation-spans":{label:"两个 operation 区间互相覆盖",definition:"同一段 Kcore rdcycle 同时被多个 operation 声明，分析器无法排他归属。",not:"不是已经证明两个 hardware engine 并行。"},
    "overlapping-site-spans":{label:"两个调用点区间互相覆盖",definition:"同一段 Kcore rdcycle 同时落在多个动态 target-call site 中。",not:"不是可优化的正常重叠；应先检查插桩边界。"},
    "no-valid-site-boundaries":{label:"没有可用的调用点边界",definition:"Trace entry 存在，但没有足够的 site begin/end 来做语义分项。",not:"不能把整段当作 idle 或某个 engine 的成本。"},
    "operation@site/event":{label:"已关联到具体调用点的 operation",definition:"reason 后半段给出 event kind、site、动态 instance 和 event sequence。",not:"reason 是来源定位，不是另一笔可相加成本。"},
    "inside-site-outside-operation":{label:"调用点内、已识别 operation 之外",definition:"位于具体 target-call site 内，但不属于已识别 submit/wait/DTE operation 的剩余区间。",not:"不能直接叫做硬件空闲或纯软件开销。"},
    "before-first-site":{label:"首个调用点之前",definition:"Trace entry 开始后、第一次动态 target-call 之前的准备区间。",not:"不是 Trace entry-setup；后者在 entry 轴外单列。"},
    "after-last-site":{label:"最后调用点之后",definition:"最后一次动态 target-call 结束后、Trace entry 返回前的收尾区间。",not:"不是 Trace entry-teardown；后者在 entry 轴外单列。"},
    "between-sites":{label:"两个调用点之间",definition:"reason 明确列出前一个和后一个动态 site，用于定位依赖、等待或控制来源。",not:"不能直接称为 engine idle。"}
  },
  correctness:{
    "production-validation-failed":{label:"最终产物结果校验失败",definition:"至少一个 production execution 输出没有通过结果/guard 校验。",not:"不能用于性能结论。"},
    "expected-exact":{label:"全部输出与独立期望值精确一致",definition:"所有输出资源都有 independent expected，并按 exact 规则通过 production validation。",not:"不替代 profiler capture 一致性和其它资格门禁。"},
    "expected-relaxed-f16":{label:"全部输出按 FP16 容差通过",definition:"所有输出资源都有 independent expected，并按允许的 FP16 容差通过 production validation。",not:"不是 bitwise exact。"},
    "partially-expected":{label:"仅部分输出有独立期望值",definition:"production 结果校验通过，但只有一部分资源覆盖 independent expected comparison。",not:"不能声称完整的独立语义正确性。"},
    "primary-validated":{label:"生产结果已校验，未提供独立期望值",definition:"production validation 和 guard 已通过，但没有 external independent expected comparison。",not:"不是独立 semantic oracle。"}
  },
  structure:{
    "trace-run-cost-overlay":{label:"Trace 插桩成本分组",definition:"八种 Trace-only 成本行的结构容器；各行互斥，但整体与语义时间轴嵌套。",not:"不能与语义分项相加。"},
    "semantic-partition":{label:"Trace entry 排他语义分项",definition:"把当前 tile 的 Trace entry rdcycle 轴不重不漏地划分为语义成本。",not:"它来自 Trace clone，不能直接还原 Primary 精确分项。"},
    "trace-overhead-overlay":{label:"Trace-only 成本附加层",definition:"单独记录 profiler 采样和记账成本，避免伪造时间位置。",not:"不是 Primary 的组成部分。"}
  },
  validity:{
    "identity":{label:"证据身份与 schema",definition:"evidence schema、ABI 和身份字段已通过验证。",not:"不代表板端环境、输出或 PMU 同时有效。"},
    "output_equivalence":{label:"Primary 与诊断执行输出一致",definition:"production 输出校验通过，且 Count/Trace captures 与 Primary 结果一致。",not:"不等于已有完整 independent expected oracle。"},
    "semantic_correctness":{label:"独立语义期望值覆盖",definition:"有完整 independent expected 时为通过/失败；没有完整覆盖时保持尚未独立判定。",not:"尚未判定不等于 Invalid。"},
    "environment":{label:"板卡环境资格",definition:"本次 evidence 声明的硬件、driver/runtime 与环境门禁已满足。",not:"不能由历史报告替代本次证据。"},
    "package_companion":{label:"产物与 profiler companion 匹配",definition:"最终 package 和同源 profiler companion 的身份/版本关系有效。",not:"不允许混用旧 companion 或另一份产物。"},
    "measurement_basis":{label:"Primary 计时基础",definition:"未插桩最终产物的 same-stream device event 计时基础满足合同。",not:"不包含 Host submit，也不等于 Trace clone duration。"},
    "host_completion_resolution":{label:"Host 完成观测分辨率",definition:"Host completion polling 的观测间隔足够细，可作为次级诊断 envelope。",not:"未满足时不否定 device event elapsed，但 Host envelope 不能作高分辨率结论。"},
    "trace":{label:"Trace 采集完整性",definition:"所有 tile 的 Trace 生命周期、容量、terminal 和区间关系闭合。",not:"不自动证明每个 PMU counter 都可归属。"},
    "cost_accounting":{label:"成本分项会计闭合",definition:"每个 tile 的语义分项排他并覆盖完整 Trace entry，Trace-only overlay 独立。",not:"不表示这些 Trace 数值可以从 Primary 直接相减。"},
    "pmu":{label:"PMU 计数资格",definition:"需要使用的 NCC/DTE/aggregate counter 读取稳定且生命周期闭合。",not:"PMU bound 仍不是精确 engine 起止。"},
    "clock_alignment":{label:"Tile-local 时钟有效",definition:"每个 tile 自己的 Kcore rdcycle 关系有效且单调。",not:"不建立不同 tile 之间的绝对时钟对齐。"}
  }
};
const GLOSSARY_GROUPS={semantic:"语义成本",trace:"Trace 插桩成本",status:"测量状态/标记",accounting:"与 Primary 的关系",magnitude:"数值代表性",interval:"时间区间角色",location:"Trace 成本位置",event:"Timeline 事件类型",site:"Target-call site 类型",engine:"Engine 类型",role:"Direct-DTE 角色",evidence:"通信证据粒度",reason:"成本来源原因",correctness:"输出正确性",structure:"成本结构",validity:"资格检查"};
const TERM_META={
  semantic:"单位为 tile-local Kcore rdcycle；是否属于 Primary 和数值代表性由该实例的 accounting 字段说明。",
  trace:"单位为 tile-local Kcore rdcycle；Primary=no；只属于 Count/Trace 诊断执行。",
  status:"无独立单位；它修饰相邻数值或区间，不能脱离计时源解释。",
  accounting:"无独立单位；说明语义是否存在于未插桩 Primary，不能证明 Trace 数值等于 Primary 分项。",
  magnitude:"无独立单位；说明当前数值与生产执行之间的代表关系。",
  interval:"横轴单位为当前 tile 的 Kcore rdcycle；不建立跨 tile 全局时间。",
  location:"相对当前 tile Trace entry 的位置；Primary 关系另看 accounting。",
  event:"operation 区间使用 tile-local Kcore rdcycle；NCC PMU work 另用 ns；Primary 关系看对应语义成本。",
  site:"site begin/end 使用 tile-local Kcore rdcycle；site 容器本身不额外计账。",
  engine:"NCC engine active work来自另一轮Trace diagnostic的vendor PMU ns，提交/operation横轴使用Kcore rdcycle；不能跨单位相加，也不能与Primary包络相减。",
  role:"无独立单位；只描述当前 tile 在 Direct-DTE operation 中的角色。",
  evidence:"operation 使用 tile-local Kcore rdcycle；整次通信行的 raw PMU 没有校准时间单位。",
  reason:"无独立单位；它解释一行成本来自哪一段边界或哪个动态 site/event。",
  correctness:"无独立单位；说明 production output validation 与 independent expected 的覆盖程度。",
  structure:"结构术语本身无单位；其行内数值通常为 tile-local Kcore rdcycle。",
  validity:"无独立单位；每一项都是独立资格门禁，三态为通过、门禁未满足或尚未独立判定。"
};
const TERM_GUIDANCE={
  semantic:"点击 Timeline 的对应成本块查看该实例边界、Primary 关系和精确 optimization_entry。",
  trace:"只在 profiler 自身开销过大时优化；不要把它当成 production artifact 优化收益。",
  status:"先确认该状态修饰的是 submit、PMU bound、PMU work 还是 capture，再决定是否可用于结论。",
  accounting:"用它判断能否讨论 Primary 语义；不要用它把不同实验轮次的数字直接相减。",
  magnitude:"proxy 用于定位方向，不用于精确拆分 Primary；trace-only 只用于控制观测成本。",
  interval:"实心块看精确 submit/operation，虚线框只看活动可能范围，PMU ns 到 Tile / Engine 查看。",
  location:"结合相邻 site 和点击详情定位来源；未定位项不能武断归入某个 gap。",
  event:"点击事件查看 submit/operation、PMU bound、PMU work 三层证据，再进入对应成本卡片定位优化。",
  site:"用 site id、target-call symbol 和前后 site 定位 wrapper、依赖或等待来源。",
  engine:"先按 engine 类型筛选，再分别比较 submit/operation、PMU work 与 Primary；禁止把各 engine ns 求和。",
  role:"发送/接收要成对理解，但当前报告不构造跨 tile 绝对时间轴。",
  evidence:"按“整次通信等待（总计）”看完整生命周期，按“通信内部步骤”定位分项；两种粒度不重复求和。",
  reason:"先用人话原因定位前后 site 或冲突边界，再查看该成本行的 optimization_entry。",
  correctness:"先确认是否有完整 independent expected；部分覆盖或无覆盖时不要夸大语义正确性结论。",
  structure:"用语义分项优化 production，用 Trace-only 附加层控制 profiler 自身开销；两层不相加。",
  validity:"未通过项要看具体门禁定义；尚未独立判定应补 oracle，而不是当成失败。"
};
const DTE_PHASES=Object.fromEntries(["direct-dte-wait","direct-dte-peer-ready-wait","direct-dte-setup-issue","direct-dte-completion-wait","direct-dte-cleanup"].map(key=>[key,TERMS.event[key].label]));
const STATES=["Measured","Sampled","Bounded","Zero delta","Zero-delta marker","Ambiguous","Attribution ambiguous","Unavailable","Incomplete","Invalid"];
const state={view:"overview",tile:0,zoom:1,engines:new Set(ENGINES),selectedEvent:null,selectedCost:null,raw:"analysis"};
const q=(selector,root)=>(root||document).querySelector(selector);
const qa=(selector,root)=>Array.prototype.slice.call((root||document).querySelectorAll(selector));
const escapeHtml=value=>String(value==null?"—":value).replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const number=value=>value==null?"—":Number(value).toLocaleString();
const durationText=value=>{
  const ns=Number(value);
  if(ns<1000)return `${number(ns)} ns`;
  if(ns<1e6)return `${(ns/1000).toFixed(3)} µs`;
  return `${(ns/1e6).toFixed(3)} ms`;
};
const durationRange=(minimum,maximum)=>minimum==null||maximum==null?"—":minimum===maximum?durationText(minimum):`${durationText(minimum)} – ${durationText(maximum)}`;
const tileLabel=tile=>"T"+(Number(tile)<10?"0":"")+String(tile);
const term=(group,value)=>TERMS[group]&&TERMS[group][value]?TERMS[group][value]:{label:String(value),definition:"尚无展示说明。",not:"请查看原始字段。"};
const termCell=(group,value)=>{const item=term(group,value);return `<b>${escapeHtml(item.label)}</b><br><span class="mono">${escapeHtml(value)}</span>`};
const termValue=(group,value)=>{const item=term(group,value);return `${escapeHtml(item.label)} <span class="mono">(${escapeHtml(value)})</span>`};
const statusBadge=value=>{const item=term("status",value);return `<span class="status status-${escapeHtml(String(value).replace(/[^A-Za-z0-9_-]/g,"-"))}" title="${escapeHtml(value)}">${escapeHtml(item.label)}</span>`};
const validityBadge=value=>statusBadge(value===true?"Passed":value===false?"Gate unmet":"Not assessed");
const semanticReason=value=>{
  if(TERMS.reason[value])return TERMS.reason[value];
  const text=String(value==null?"":value);
  if(text.startsWith("inside-site-outside-operation@"))return TERMS.reason["inside-site-outside-operation"];
  if(text.startsWith("before-first-"))return TERMS.reason["before-first-site"];
  if(text.startsWith("after-last-"))return TERMS.reason["after-last-site"];
  if(text.startsWith("between-")&&text.includes("-and-"))return TERMS.reason["between-sites"];
  if(text.includes("@site-")&&text.includes("/event-"))return TERMS.reason["operation@site/event"];
  return {label:text||"—",definition:"尚无展示说明。",not:"请查看原始字段。"};
};
const siteRefText=site=>site==null?"—":`#${site.site_id} / instance ${site.instance_sequence} · ${term("site",site.site_kind).label} (${site.site_kind}) · ${site.target_call_symbol}`;
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
  if(view==="glossary")renderGlossary();
  if(view==="diagnostics")renderDiagnostics();
}

function selectTile(tile,view="timeline",engine=null){
  state.tile=Number(tile);
  state.selectedEvent=null;
  state.selectedCost=null;
  if(engine){state.engines=new Set([engine]);renderEngineFilters()}
  q("#timelineTile").value=String(state.tile);
  q("#engineTile").value=String(state.tile);
  navigate(view);
}

function focusEvent(tile,siteId,engine,sequence=null){
  const selectedSequence=sequence==null||sequence===""?null:Number(sequence);
  const event=finalArtifact.timeline_events.find(row=>row.tile===Number(tile)&&row.site_id===Number(siteId)&&(engine==null||engine===""||engine==="null"||row.engine===engine)&&(selectedSequence==null||row.sequence===selectedSequence));
  if(!event)return;
  state.tile=Number(tile);
  state.selectedEvent=event;
  state.selectedCost=null;
  if(ENGINES.includes(engine))state.engines.add(engine);
  renderEngineFilters();
  q("#timelineTile").value=String(state.tile);
  q("#engineTile").value=String(state.tile);
  navigate("timeline");
}

function bindEventLinks(selector){
  qa(selector).forEach(node=>{
    const activate=()=>focusEvent(node.dataset.eventTile,node.dataset.eventSite,node.dataset.eventEngine,node.dataset.eventSequence);
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
  const engineActive=finalArtifact.engine_active_time;
  const tileWork=engineActive.per_tile_work_volume;
  q("#statusLegend").innerHTML=STATES.map(statusBadge).join("");
  q("#deviceDuration").textContent=durationText(duration.device_elapsed_ns);
  q("#deviceDurationNote").textContent=duration.device_elapsed_quantized_zero?`${duration.device_timer_kind} · at/below effective timer resolution`:`${duration.device_timer_kind} · TX stream device envelope, not engine-only`;
  q("#engineActiveRange").textContent=durationRange(tileWork.minimum_per_tile_total_work_ns,tileWork.maximum_per_tile_total_work_ns);
  q("#engineActiveRangeNote").innerHTML=`average ${tileWork.average_per_tile_total_work_ns==null?"—":durationText(tileWork.average_per_tile_total_work_ns)} · ${number(tileWork.available_tile_count)} / ${finalArtifact.tiles.length} complete tiles · five-engine work volume, not latency · ${statusBadge(tileWork.status)}`;
  q("#hostSubmitDuration").textContent=durationText(duration.host_submit_ns);
  q("#hostSubmitNote").textContent=duration.host_submit_quantized_zero?"separate host diagnostic · at/below host clock resolution":"separate host diagnostic · provider preparation + submit";
  q("#hostEnvelopeDuration").textContent=durationText(duration.host_launch_to_completion_ns);
  q("#hostEnvelopeResolution").textContent=duration.host_envelope_available?`separate host-clock envelope · non-submit ${durationText(duration.host_non_submit_envelope_ns)} · completion observation ≤ ${number(duration.completion_observation_resolution_ns)} ns`:"host completion diagnostic unavailable / quantized zero";
  q("#measurementStatus").innerHTML=statusBadge(duration.status);
  q("#timingRows").innerHTML=[
    ["Kernel launch → completion","Primary TX stream events",duration.device_elapsed_ns,duration.status,"Uninstrumented production launch envelope: device dispatch, Kcore control, NCC/DTE execution and waits, scheduling gaps, and retirement. Excludes Host and Trace-only work; not engine-only time."],
    ["Host submit","host steady_clock",duration.host_submit_ns,"Measured","Provider argument preparation, stream/event setup, and submit path. Separate Host diagnostic; not pure TX API time."],
    ["Host non-submit envelope","host steady_clock",duration.host_non_submit_envelope_ns,duration.host_envelope_available?"Measured":"Unavailable","Host wait, completion polling and phase control after submit. It overlaps the Primary TX stream envelope, is not pure overhead, and is never reduced by device elapsed."],
    ["Host launch → trusted completion","host steady_clock",duration.host_launch_to_completion_ns,duration.host_envelope_available?"Measured":"Unavailable",duration.host_envelope_available?`First submit through trusted completion; separate Host-clock envelope; observation gap ≤ ${number(duration.completion_observation_resolution_ns)} ns.`:"Quantized/unavailable host diagnostic; does not invalidate TX stream event elapsed."]
  ].map(row=>`<tr><td><b>${escapeHtml(row[0])}</b></td><td class="mono">${escapeHtml(row[1])}</td><td class="num">${durationText(row[2])}</td><td class="num">${number(row[2])}</td><td>${statusBadge(row[3])}</td><td>${escapeHtml(row[4])}</td></tr>`).join("");
  q("#outputStatus").innerHTML=statusBadge(analysis.validity.output_equivalence?"Measured":"Invalid");
  q("#outputNote").innerHTML=`${finalArtifact.output.resource_count} resources · ${termValue("correctness",finalArtifact.output.correctness_status)}`;
  const completeTiles=finalArtifact.tiles.filter(tile=>tile.trace_status==="Measured").length;
  q("#traceCoverage").textContent=`${completeTiles} / ${finalArtifact.tiles.length}`;
  q("#traceNote").textContent="tile-local complete trace spans";
  q("#overviewTiles").innerHTML=[...finalArtifact.tiles].sort((left,right)=>left.y-right.y||left.x-right.x).map(tile=>`<button class="tile-card" data-overview-tile="${tile.tile}" type="button"><b>${tileLabel(tile.tile)} ${statusBadge(tile.trace_status)}</b><small>${number(tile.trace_entry_cpu_cycles)} Kcore CPU cycles · (${tile.x},${tile.y})</small></button>`).join("");
  qa("[data-overview-tile]").forEach(node=>node.addEventListener("click",()=>selectTile(node.dataset.overviewTile)));
  const methodLabels={kernel_launch_to_completion:"Kernel launch → completion",host_submit_time:"Host submit",host_envelope_time:"Host launch → trusted completion",host_envelope_ledger:"Host envelope ledger",queue_delay:"Queue delay",timeline_axis:"Timeline axis",timeline_scope:"Clock/order",semantic_partition:"Semantic partition",trace_overhead_overlay:"Trace-run cost overlay",engine_active_time:"NCC engine active time",statistics_window:"Statistics window",ncc_activity_window:"NCC window",direct_dte_time:"Direct-DTE wait",direct_dte_raw_pmu:"Direct-DTE raw"};
  q("#methodList").innerHTML=Object.keys(analysis.method).map(key=>`<div class="method-row"><b>${escapeHtml(methodLabels[key]||key)}</b><span>${escapeHtml(analysis.method[key])}</span></div>`).join("");
  q("#overviewEngineRows").innerHTML=engineActive.by_engine.map(row=>{
    return `<tr><td>${termCell("engine",row.engine)}</td><td>Trace diagnostic PMU<br><span class="mono">ns · separate-run proxy</span></td><td class="num">${row.minimum_per_tile_ns==null?"—":durationText(row.minimum_per_tile_ns)}</td><td class="num">${row.average_per_tile_ns==null?"—":durationText(row.average_per_tile_ns)}</td><td class="num">${row.maximum_per_tile_ns==null?"—":durationText(row.maximum_per_tile_ns)}</td><td class="num">${number(row.active_tile_count)} / ${finalArtifact.tiles.length}</td><td class="num">${number(row.available_tile_count)} / ${finalArtifact.tiles.length}</td><td class="num">${row.sum_across_tiles_work_ns==null?"—":durationText(row.sum_across_tiles_work_ns)}<br><span class="metric-note">not wall time</span></td><td>${statusBadge(row.status)}</td></tr>`;
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
  q("#timelineRuler").innerHTML=`<div class="ruler"><span class="metric-note">Kcore rdcycle</span><div class="ruler-track">${ticks.map(tick=>`<i class="tick" style="left:${tick.fraction*100}%"><span>${number(tick.value)}</span></i>`).join("")}</div></div>`;
}

function crossEngineBoundOverlaps(events){
  const bounded=events.filter(event=>event.activity_window_status==="Bounded"&&event.activity_trace_entry_offset_begin_cpu_cycles!=null&&event.activity_trace_entry_offset_end_cpu_cycles!=null);
  const overlaps=[];
  for(let left=0;left<bounded.length;left+=1){
    for(let right=left+1;right<bounded.length;right+=1){
      const first=bounded[left],second=bounded[right];
      if(first.engine===second.engine)continue;
      const begin=Math.max(first.activity_trace_entry_offset_begin_cpu_cycles,second.activity_trace_entry_offset_begin_cpu_cycles);
      const end=Math.min(first.activity_trace_entry_offset_end_cpu_cycles,second.activity_trace_entry_offset_end_cpu_cycles);
      if(begin<end)overlaps.push({first,second,begin,end,cycles:end-begin});
    }
  }
  return overlaps;
}

function renderTimeline(){
  const tile=selectedTile();
  if(!tile)return;
  q("#timelineTile").value=String(state.tile);
  q("#timelineCanvas").style.minWidth=`${Math.max(100,state.zoom*100)}%`;
  renderRuler(tile.trace_entry_cpu_cycles);
  const events=tileEvents();
  const overlaps=crossEngineBoundOverlaps(events);
  const overlapNotice=q("#timelineOverlapNotice");
  overlapNotice.hidden=overlaps.length===0;
  if(overlaps.length){
    const pairs=Array.from(new Set(overlaps.map(item=>`${item.first.engine}/${item.second.engine}`))).join("、");
    overlapNotice.innerHTML=`<b>${tileLabel(tile.tile)} 有 ${overlaps.length} 组跨 engine PMU 观测范围相交（${escapeHtml(pairs)}）。</b>浅色虚线框只是 conservative activity bound，常因多个 outstanding event 共用 completion 采样上界而重叠；这不证明 engine 同时执行。实心块才是精确 command submit / operation 区间。`;
  }
  const gridlines=[20,40,60,80].map(position=>`<i class="lane-gridline" style="left:${position}%"></i>`).join("");
  const semanticMarks=tile.semantic_timeline_segments.map((segment,index)=>`<button class="cost-segment cost-${escapeHtml(segment.category)}${state.selectedCost&&state.selectedCost.scope==="semantic"&&state.selectedCost.index===index?" selected":""}" data-cost-index="${index}" style="left:${segment.plot_begin_fraction*100}%;width:${Math.max(.15,(segment.plot_end_fraction-segment.plot_begin_fraction)*100)}%" title="${escapeHtml(term("semantic",segment.category).label)} · ${escapeHtml(segment.reason)} · ${number(segment.cycles)} cycles" aria-label="${escapeHtml(term("semantic",segment.category).label)} · ${number(segment.cycles)} cycles" type="button"></button>`).join("");
  const traceRows=tile.trace_overhead_overlay.rows;
  const traceMarks=(location)=>traceRows.map((row,index)=>({row,index})).filter(item=>location==="outside-entry"?item.row.location_granularity==="before-entry"||item.row.location_granularity==="after-entry":item.row.location_granularity===location).map(item=>`<button class="overlay-part${state.selectedCost&&state.selectedCost.scope==="trace"&&state.selectedCost.index===item.index?" selected":""}" data-trace-cost-index="${item.index}" style="flex:${Math.max(1,item.row.cycles)} 1 0" title="${escapeHtml(term("trace",item.row.reason).label)} · ${number(item.row.cycles)} cycles · aggregate-only / non-additive" aria-label="${escapeHtml(term("trace",item.row.reason).label)} · ${number(item.row.cycles)} cycles" type="button"></button>`).join("");
  const semanticLane=`<div class="lane lane-production" data-lane-engine="Trace-run Kcore ledger"><div class="lane-label"><b>Trace-run Kcore ledger</b><span>${number(tile.semantic_partition.exclusive_cycles)} cyc</span></div><div class="lane-track">${gridlines}${semanticMarks}</div></div>`;
  const engineLanes=ENGINES.map(engine=>{
    const visible=state.engines.has(engine);
    const engineEvents=events.filter(event=>event.engine_lane_visible&&event.engine===engine);
    const engineSummary=tile.engines.find(row=>row.engine===engine);
    const metric=engine==="DIRECT_DTE"?`${number(engineSummary&&engineSummary.wait_window_cpu_cycles)} cyc`:`${number(engineSummary&&engineSummary.engine_execution_time_ns)} ns`;
    const marks=visible?engineEvents.map((event,index)=>{
      const variant=`event-v${index%3}`;
      const bound=event.activity_window_status==="Bounded"&&event.activity_plot_begin_fraction!=null&&event.activity_plot_end_fraction!=null?`<span class="observation-bound observation-${engine} ${variant}${event.attribution_ambiguous?" ambiguous":""}${event.zero_delta_marker?" zero":""}" data-interval-role="observation-bound" style="left:${event.activity_plot_begin_fraction*100}%;width:${Math.max(.2,(event.activity_plot_end_fraction-event.activity_plot_begin_fraction)*100)}%" title="${escapeHtml(engine)} · PMU 正增量发生在此保守范围内的某处；不代表持续 busy 或精确执行起止"></span>`:"";
      const selected=state.selectedEvent&&state.selectedEvent.tile===event.tile&&state.selectedEvent.sequence===event.sequence&&state.selectedEvent.site_id===event.site_id;
      const role=event.display_interval_role;
      const roleLabel=term("interval",role).label;
      const pointRole=role==="marker";
      const main=`<button class="event event-${engine} ${variant}${pointRole?" marker":""}${event.attribution_ambiguous?" ambiguous":""}${event.zero_delta_marker?" zero":""}${selected?" selected":""}" data-interval-role="${escapeHtml(role)}" data-event-sequence="${event.sequence}" data-event-site="${event.site_id}" style="left:${event.plot_begin_fraction*100}%;width:${pointRole?".2":Math.max(.2,(event.plot_end_fraction-event.plot_begin_fraction)*100)}%" title="${escapeHtml(engine)} · ${escapeHtml(roleLabel)} · ${number(event.operation_window_cpu_cycles)} Kcore CPU cycles" aria-label="${escapeHtml(engine)} · ${escapeHtml(roleLabel)} · site ${event.site_id}" type="button"></button>`;
      return bound+main;
    }).join(""):"";
    return `<div class="lane" data-lane-engine="${engine}"><div class="lane-label"><b title="${escapeHtml(term("engine",engine).definition)}">${engine}</b><span>${metric} · ${engineEvents.length}</span></div><div class="lane-track">${gridlines}${marks}</div></div>`;
  }).join("");
  q("#timelineLanes").innerHTML=semanticLane+`<div class="lane-separator"></div>`+engineLanes;
  q("#traceOverheadBar").innerHTML=traceMarks("inside-entry-unpositioned");
  q("#traceOutsideBar").innerHTML=traceMarks("outside-entry");
  qa("[data-cost-index]").forEach(node=>node.addEventListener("click",()=>{
    state.selectedEvent=null;
    state.selectedCost={scope:"semantic",index:Number(node.dataset.costIndex)};
    renderTimeline();
  }));
  qa("[data-trace-cost-index]").forEach(node=>node.addEventListener("click",()=>{
    state.selectedEvent=null;
    state.selectedCost={scope:"trace",index:Number(node.dataset.traceCostIndex)};
    renderTimeline();
  }));
  qa("[data-event-sequence]").forEach(node=>node.addEventListener("click",()=>{
    const event=events.find(row=>row.sequence===Number(node.dataset.eventSequence)&&row.site_id===Number(node.dataset.eventSite));
    state.selectedEvent=event||null;
    state.selectedCost=null;
    renderEventDetail();
    qa(".event").forEach(mark=>mark.classList.toggle("selected",mark===node));
  }));
  if(state.selectedCost)renderCostDetail();
  else{
    if(!state.selectedEvent||state.selectedEvent.tile!==state.tile)state.selectedEvent=events.find(event=>event.engine_lane_visible)||events[0]||null;
    renderEventDetail();
  }
  q("#semanticCostRows").innerHTML=tile.semantic_partition.rows.map(row=>{
    const reason=semanticReason(row.reason);
    return `<tr><td>${termCell("semantic",row.category)}${row.reason?`<br><span title="${escapeHtml(reason.definition)}">${escapeHtml(reason.label)}</span><br><span class="mono">${escapeHtml(row.reason)}</span>`:""}${row.event_kind?`<br>${termCell("event",row.event_kind)}`:""}</td><td class="num">${number(row.cycles)}</td><td class="num">${(row.share_of_trace_entry*100).toFixed(1)}%</td><td>${termValue("accounting",row.counts_in_primary_device_elapsed)}</td><td>${termValue("magnitude",row.magnitude_relation)}</td><td>${escapeHtml(row.optimization_entry)}</td></tr>`;
  }).join("");
  q("#traceCostRows").innerHTML=tile.trace_overhead_overlay.rows.map(row=>`<tr><td>${termCell("trace",row.reason)}</td><td class="num">${number(row.cycles)}</td><td class="num">${row.share_of_trace_entry==null?"axis 外":(row.share_of_trace_entry*100).toFixed(1)+"%"}</td><td>${termValue("location",row.location_granularity)}</td><td>${termValue("accounting",row.counts_in_primary_device_elapsed)}</td></tr>`).join("");
}

function renderCostDetail(){
  const tile=selectedTile();
  const selection=state.selectedCost;
  if(!tile||!selection)return;
  const row=selection.scope==="semantic"?tile.semantic_timeline_segments[selection.index]:tile.trace_overhead_overlay.rows[selection.index];
  if(!row)return;
  const semantic=selection.scope==="semantic";
  const group=semantic?"semantic":"trace";
  const rawKey=semantic?row.category:row.reason;
  const explanation=term(group,rawKey);
  const reasonExplanation=semantic?semanticReason(row.reason):null;
  const share=row.share_of_trace_entry==null?"axis 外 / not applicable":(row.share_of_trace_entry*100).toFixed(1)+"%";
  const sourceContext=semantic?`<dt>Containing site</dt><dd>${escapeHtml(siteRefText(row.containing_site))}</dd><dt>Previous site</dt><dd>${escapeHtml(siteRefText(row.previous_site))}</dd><dt>Next site</dt><dd>${escapeHtml(siteRefText(row.next_site))}</dd><dt>Source event</dt><dd>${number(row.source_event_sequence)}</dd>`:"";
  const claimantContext=semantic&&row.claimant_sites&&row.claimant_sites.length?`<dt>边界冲突涉及</dt><dd>${row.claimant_sites.map(site=>escapeHtml(siteRefText(site))).join("<br>")}</dd>`:"";
  const location=semantic?`${number(row.begin_cycle)} → ${number(row.end_cycle)} Kcore rdcycle`:termValue("location",row.location_granularity);
  q("#eventDetail").innerHTML=`<h3>${escapeHtml(explanation.label)}</h3><dl class="kv"><dt>机器字段</dt><dd class="mono">${escapeHtml(rawKey)}</dd><dt>具体含义</dt><dd>${escapeHtml(explanation.definition)}</dd><dt>不能这样理解</dt><dd>${escapeHtml(explanation.not)}</dd><dt>来源原因</dt><dd>${semantic?`<b>${escapeHtml(reasonExplanation.label)}</b><br>${escapeHtml(reasonExplanation.definition)}<br><span class="metric-note">不能这样理解：${escapeHtml(reasonExplanation.not)}</span><br><span class="mono">${escapeHtml(row.reason)}</span>`:`${escapeHtml(explanation.label)}<br><span class="mono">${escapeHtml(row.reason)}</span>`}</dd><dt>耗时</dt><dd>${number(row.cycles)} Kcore CPU cycles</dd><dt>Entry 占比</dt><dd>${share}</dd><dt>是否属于 Primary</dt><dd>${termValue("accounting",row.counts_in_primary_device_elapsed)}</dd><dt>数值代表性</dt><dd>${termValue("magnitude",row.magnitude_relation)}</dd><dt>计时边界/位置</dt><dd>${location}</dd>${sourceContext}${claimantContext}<dt>优化入口</dt><dd>${escapeHtml(row.optimization_entry)}</dd>${semantic?"<dt>会计关系</dt><dd>Trace-run entry 的排他分区；包含嵌套插桩影响，不能直接从 Primary 相减。</dd>":`<dt>会计关系</dt><dd>非加和 Trace-only overlay；不能与语义 ledger 相加。</dd>`}</dl>`;
}

function renderEventDetail(){
  const event=state.selectedEvent;
  if(!event){q("#eventDetail").innerHTML=`<h3>Event details</h3><div class="empty">No usable event on this tile.</div>`;return}
  const direct=event.engine==="DIRECT_DTE";
  const eventType=term("event",event.kind);
  const siteType=term("site",event.site_kind);
  const engineType=event.engine==null?null:term("engine",event.engine);
  const dteRole=event.dte_role==null?null:term("role",event.dte_role);
  const counterLabel=direct?"Raw DTE PMU":"Attributed engine work";
  const counterValue=direct?event.direct_dte_raw_pmu_activity:event.ncc_engine_execution_time_ns;
  const counterUnit=direct?"":" ns";
  const attribution=event.kind==="ncc-command"&&event.ncc_counter_valid===false?"PMU counter 不可用；精确 submit 仍保留，但没有 engine work 归属":event.attribution_ambiguous?"同 engine 活动归属不唯一，不能作为精确 site work":event.zero_delta_marker?"保留 zero-delta 观测；不等于 operation 零耗时":"在当前局部状态下可用";
  const interval=term("interval",event.display_interval_role);
  const observation=event.activity_trace_entry_offset_begin_cpu_cycles==null?"无可用 PMU 活动范围":`${number(event.activity_trace_entry_offset_begin_cpu_cycles)} → ${number(event.activity_trace_entry_offset_end_cpu_cycles)}；宽度 ${number(event.activity_window_cpu_cycles)} Kcore CPU cycles`;
  q("#eventDetail").innerHTML=`<h3>${escapeHtml(tileLabel(event.tile)+" · "+(event.engine||"Kcore"))} ${statusBadge(event.operation_window_status)}</h3><dl class="kv"><dt>事件类型</dt><dd><b>${escapeHtml(eventType.label)}</b><br><span class="mono">${escapeHtml(event.kind)}</span><br>${escapeHtml(eventType.definition)}<br><span class="metric-note">不能这样理解：${escapeHtml(eventType.not)}</span></dd><dt>Engine 类型</dt><dd>${engineType?`<b>${escapeHtml(engineType.label)}</b><br><span class="mono">${escapeHtml(event.engine)}</span><br>${escapeHtml(engineType.definition)}`:"Kcore / no engine lane"}</dd><dt>实心块含义</dt><dd><b>${escapeHtml(interval.label)}</b><br><span class="mono">${escapeHtml(event.display_interval_role)}</span><br>${escapeHtml(interval.definition)}<br><span class="metric-note">不能这样理解：${escapeHtml(interval.not)}</span></dd><dt>精确提交/operation</dt><dd class="mono">${number(event.operation_trace_entry_offset_begin_cpu_cycles)} → ${number(event.operation_trace_entry_offset_end_cpu_cycles)}；${number(event.operation_window_cpu_cycles)} Kcore CPU cycles · ${statusBadge(event.operation_window_status)}</dd><dt>PMU 活动保守范围</dt><dd>${observation} · ${statusBadge(event.activity_window_status)}<br><span class="metric-note">范围重叠不证明 engine 同时执行。</span></dd><dt>${counterLabel}</dt><dd>${number(counterValue)}${counterUnit} · ${statusBadge(event.counter_status)}<br><span class="metric-note">${direct?"未校准 raw activity，不是 elapsed time。":"Vendor PMU measured work duration；没有精确起止坐标。"}</span></dd><dt>归属质量</dt><dd>${escapeHtml(attribution)}</dd><dt>Site 类型</dt><dd><b>${escapeHtml(siteType.label)}</b><br><span class="mono">${escapeHtml(event.site_kind)}</span><br>${escapeHtml(siteType.definition)}</dd><dt>Site 实例</dt><dd class="mono">${event.site_id}:${event.sub_index} · instance ${number(event.site_instance_sequence)}</dd><dt>Correlation</dt><dd>${escapeHtml(event.correlation_key)}</dd><dt>Target call</dt><dd class="mono">#${number(event.target_call_ordinal)} ${escapeHtml(event.target_call_symbol)}</dd><dt>Position</dt><dd class="mono">${escapeHtml(event.position)}</dd><dt>DTE role</dt><dd>${dteRole?`<b>${escapeHtml(dteRole.label)}</b> <span class="mono">(${escapeHtml(event.dte_role)})</span><br>${escapeHtml(dteRole.definition)}`:"—"}</dd><dt>怎么看 / 优化入口</dt><dd>${escapeHtml(TERM_GUIDANCE.event)}</dd><dt>Scope</dt><dd>仅当前 tile；不能建立跨 tile 全局顺序</dd></dl>`;
}

function renderEngines(){
  state.tile=Number(q("#engineTile").value||state.tile);
  const tile=selectedTile();
  q("#engineTile").value=String(state.tile);
  q("#engineTileMeta").innerHTML=`${statusBadge(tile.trace_status)} · trace ${number(tile.trace_entry_cpu_cycles)} Kcore CPU cycles · clock ${statusBadge(tile.clock_status)}`;
  const statisticsRow=`<tr><td><b>STATISTICS_WINDOW</b></td><td>Aggregate PMU window (raw ticks)</td><td class="num">${number(tile.statistics_window_raw_ticks)}</td><td class="num">—</td><td>${statusBadge(tile.statistics_window_status)}</td><td>Auxiliary raw PMU tick delta; not the rdcycle timeline axis and not elapsed time.</td></tr>`;
  q("#engineRows").innerHTML=statisticsRow+tile.engines.map(engine=>{
    if(engine.engine==="DIRECT_DTE")return[
      `<tr><td><b>DIRECT_DTE</b></td><td>Wait / completion (Kcore CPU cycles)</td><td class="num">${number(engine.wait_window_cpu_cycles)}</td><td class="num">${engine.wait_window_count}</td><td>${statusBadge(engine.wait_window_status)}</td><td>Measured tile-local rdcycle interval; frequency conversion unavailable.</td></tr>`,
      `<tr><td></td><td>Raw PMU activity</td><td class="num">${number(engine.raw_pmu_activity)}</td><td class="num">${engine.activity_window_count}</td><td>${statusBadge(engine.raw_pmu_activity_status)}</td><td>Sampled and uncalibrated; never elapsed time.</td></tr>`
    ].join("");
    return `<tr><td>${termCell("engine",engine.engine)}</td><td>Per-tile engine active time (Trace PMU ns)</td><td class="num">${number(engine.engine_execution_time_ns)}</td><td class="num">${engine.activity_window_count}</td><td>${statusBadge(engine.engine_execution_time_status)}</td><td>来自另一轮Trace diagnostic的vendor PMU execution-time delta；没有精确起止坐标，不能与Primary包络相减。Activity windows: ${termValue("status",engine.activity_window_status)}；仅为Kcore rdcycle bounds。</td></tr>`;
  }).join("");
}

function renderSites(){
  const query=q("#siteSearch").value.trim().toLowerCase();
  const engine=q("#siteEngine").value;
  const rows=finalArtifact.sites.filter(site=>(!engine||site.engine===engine)&&(!query||[site.correlation_key,site.target_call_symbol,site.position,site.site_id,site.tile].some(value=>String(value==null?"":value).toLowerCase().indexOf(query)!==-1)));
  q("#siteCount").textContent=`${rows.length} / ${finalArtifact.sites.length} sites`;
  q("#siteRows").innerHTML=rows.length?rows.map(site=>`<tr${site.site_instance_count?` class="interactive-row" tabindex="0" title="Open correlated timeline event" data-event-tile="${site.tile}" data-event-site="${site.site_id}" data-event-engine="${site.engine}"`:""}><td>${tileLabel(site.tile)}</td><td class="mono">${site.site_id}</td><td>${termCell("site",site.site_kind)}</td><td>${site.engine==null?"—":termCell("engine",site.engine)}</td><td>${escapeHtml(site.correlation_key)}</td><td class="mono">#${site.target_call_ordinal} ${escapeHtml(site.target_call_symbol)}</td><td class="mono">${escapeHtml(site.position)}</td><td class="num">${site.site_instance_count}</td><td>${statusBadge(site.site_capture_status)}</td><td class="num">${site.event_count}</td><td>${statusBadge(site.engine_observation_status)}</td></tr>`).join(""):`<tr><td colspan="11" class="empty">No matching sites.</td></tr>`;
  bindEventLinks("#siteRows [data-event-site]");
}

function renderCommunication(){
  const phaseOrder=Object.keys(DTE_PHASES);
  const events=finalArtifact.timeline_events
    .filter(event=>Object.prototype.hasOwnProperty.call(DTE_PHASES,event.kind))
    .sort((left,right)=>left.tile-right.tile||left.site_id-right.site_id||phaseOrder.indexOf(left.kind)-phaseOrder.indexOf(right.kind)||left.sequence-right.sequence);
  const leafEvents=events.filter(event=>event.kind!=="direct-dte-wait");
  const communication=finalArtifact.communication;
  q("#communicationSummary").innerHTML=`<div class="card"><div class="metric-label">整次通信等待</div><div class="metric-value">${communication.direct_dte_event_count}</div><div class="metric-note">总计行 · ${leafEvents.length} 个通信内部步骤；两者不重复相加</div></div><div class="card"><div class="metric-label">发送侧 / 接收侧</div><div class="metric-value">${communication.send_count} / ${communication.receive_count}</div></div><div class="card"><div class="metric-label">有活动的 tile</div><div class="metric-value">${communication.tiles_with_activity.length}</div></div><div class="card"><div class="metric-label">跨 tile 顺序</div><div class="metric-value">${statusBadge("Unavailable")}</div><div class="metric-note">不推断全局 timeline</div></div>`;
  q("#communicationRows").innerHTML=events.length?events.map(event=>{
    const aggregate=event.kind==="direct-dte-wait";
    const evidenceRole=aggregate?"aggregate":"leaf";
    const rawValue=aggregate?number(event.direct_dte_raw_pmu_activity):"—";
    const rawStatus=aggregate?statusBadge(event.counter_status):`<span class="evidence-role">仅“整次通信等待”行提供</span>`;
    return `<tr class="interactive-row" tabindex="0" title="Open exact correlated phase" data-event-tile="${event.tile}" data-event-site="${event.site_id}" data-event-engine="${event.engine}" data-event-sequence="${event.sequence}" data-event-kind="${escapeHtml(event.kind)}"><td>${tileLabel(event.tile)}</td><td>${termCell("role",event.dte_role)}</td><td><b>${escapeHtml(DTE_PHASES[event.kind])}</b><br><span class="mono">${escapeHtml(event.kind)}</span></td><td><span class="evidence-role">${termCell("evidence",evidenceRole)}</span></td><td><span class="mono">${event.site_id}</span> · ${escapeHtml(event.target_call_symbol)}</td><td class="num">${number(event.operation_window_cpu_cycles)}</td><td>${statusBadge(event.duration_status)}</td><td class="num">${rawValue}</td><td>${rawStatus}</td></tr>`;
  }).join(""):`<tr><td colspan="9" class="empty">No Direct-DTE operation was observed.</td></tr>`;
  bindEventLinks("#communicationRows [data-event-site]");
}

function renderGlossary(){
  q("#glossaryRows").innerHTML=Object.keys(GLOSSARY_GROUPS).flatMap(group=>Object.keys(TERMS[group]).map(key=>{
    const item=TERMS[group][key];
    return `<tr><td>${escapeHtml(GLOSSARY_GROUPS[group])}</td><td><b>${escapeHtml(item.label)}</b></td><td class="mono">${escapeHtml(key)}</td><td>${escapeHtml(item.definition)}</td><td>${escapeHtml(item.meta||TERM_META[group])}</td><td>${escapeHtml(item.not)}</td><td>${escapeHtml(item.guide||TERM_GUIDANCE[group])}</td></tr>`;
  })).join("");
}

function renderDiagnostics(){
  q("#validityMatrix").innerHTML=Object.keys(analysis.validity).map(key=>{
    const item=term("validity",key);
    return `<span title="${escapeHtml(item.definition)}"><b>${escapeHtml(item.label)}</b> <span class="mono">${escapeHtml(key)}</span> ${validityBadge(analysis.validity[key])}</span>`;
  }).join("");
  q("#diagnosticRows").innerHTML=analysis.diagnostics.length?analysis.diagnostics.map(row=>`<div class="diag"><b class="severity-${escapeHtml(row.severity)}">${escapeHtml(row.severity)}</b><span class="mono">${escapeHtml(row.code)}</span><span>${row.tile==null?"global":tileLabel(row.tile)}</span><span>${escapeHtml(row.message)}</span></div>`).join(""):`<div class="empty">No analyzer diagnostics.</div>`;
  q("#rawPayload").textContent=JSON.stringify(state.raw==="analysis"?analysis:evidence,null,2);
  qa("[data-raw-target]").forEach(node=>node.classList.toggle("active",node.dataset.rawTarget===state.raw));
}

qa("[data-view]").forEach(node=>node.addEventListener("click",()=>navigate(node.dataset.view)));
q("#timelineTile").addEventListener("change",event=>{state.tile=Number(event.target.value);state.selectedEvent=null;state.selectedCost=null;renderTimeline()});
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
renderGlossary();
renderDiagnostics();
window.__waferProfileUI={analysis,evidence,state,navigate,selectTile,focusEvent,renderTimeline,renderGlossary,crossEngineBoundOverlaps};
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
