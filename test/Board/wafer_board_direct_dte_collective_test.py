#!/usr/bin/env python3
"""Compile and run a 16-rank sharded Add/reduction Direct-DTE case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys

import numpy as np

import wafer_runtime_launch_contract as runtime_launch


RANK_COUNT = 16
LOCAL_ELEMENTS = 458752
LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
TARGET_IDENTITY = "wafer-tx81-single-card"
STATUS_ABI = "wafer-direct-dte-status-v2"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS = 30
PROFILE_INSTRUMENTATION_READY = (
    "profile_instrumentation: ready schema=7 ranks=16 variants=1 captures=2"
)
PROFILE_CAMPAIGN_LAUNCH_COUNT = 3
PROFILE_PRIMARY_EXECUTION_COUNT = 1
PROFILE_EVIDENCE_SCHEMA_VERSION = 8
PROFILE_ANALYSIS_SCHEMA_VERSION = 7
ELEMENT_TYPES = {
    "f16": ("float16", np.dtype("<f2")),
    "f32": ("float32", np.dtype("<f4")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "compile byte-identical ordinary/profile packages, then execute "
            "one fixed Primary->Count->Trace campaign"
        ),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def run(
    command: list[str], timeout_seconds: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        for partial in (error.stdout, error.stderr):
            if partial:
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(partial, end="", file=sys.stderr)
        raise RuntimeError(
            "one-shot Direct-DTE process exceeded its outer deadline; it was "
            "killed and this test will not retry or invoke reset/power "
            "operations; board state requires external read-only qualification"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")
    return result


def compile_package(
    args: argparse.Namespace,
    source: pathlib.Path,
    package: pathlib.Path,
    *,
    profile: bool,
) -> None:
    command = [
        str(args.wafer_compile),
        "--input-program-dir",
        str(source),
        "--output-program-dir",
        str(package),
        f"--execution-ranks={RANK_COUNT}",
        f"--launch-kind={LAUNCH_KIND}",
    ]
    if profile:
        command.append("--profile")
    result = run(command)
    if (
        f"wrote verified package with execution-ranks={RANK_COUNT}"
        not in result.stdout
    ):
        raise RuntimeError(
            "wafer-compile did not report a verified rank-16 package"
        )
    written_instrumentation = "wafer-compile: wrote profile instrumentation:"
    if profile and written_instrumentation not in result.stdout:
        raise RuntimeError(
            "wafer-compile did not write the requested profile instrumentation"
        )
    if not profile and written_instrumentation in result.stdout:
        raise RuntimeError(
            "ordinary wafer-compile unexpectedly wrote profile instrumentation"
        )


def require_byte_identical_packages(
    ordinary: pathlib.Path, profiled: pathlib.Path
) -> None:
    def file_digests(root: pathlib.Path) -> dict[pathlib.Path, str]:
        files = {
            path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }
        if not files:
            raise RuntimeError(f"compiled package is empty: {root}")
        return files

    if file_digests(ordinary) != file_digests(profiled):
        raise RuntimeError(
            "ordinary and --profile production packages are not byte-identical"
        )


def require_profile_instrumentation_permissions(package: pathlib.Path) -> None:
    instrumentation = pathlib.Path(f"{package}.profile")
    if not instrumentation.is_dir() or instrumentation.is_symlink():
        raise RuntimeError("profile instrumentation is not a real directory")
    for path in (instrumentation, *instrumentation.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_dir() and not path.is_file():
            raise RuntimeError(
                f"profile instrumentation contains a non-regular member: {path}"
            )
        if stat.S_IMODE(path.stat().st_mode) != 0o777:
            raise RuntimeError(
                f"profile instrumentation permission is not 0777: {path}"
            )


def write_fixture(
    work_dir: pathlib.Path,
    local_elements: int = LOCAL_ELEMENTS,
    element_type: str = "f16",
) -> pathlib.Path:
    if element_type not in ELEMENT_TYPES:
        raise RuntimeError(f"unsupported fixture element type {element_type}")
    metadata_dtype, _ = ELEMENT_TYPES[element_type]
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    devices = ",".join(str(rank) for rank in range(RANK_COUNT))
    module = f'''module {{
  wafer.target.topology @default {{card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}}
  wafer.execution.mesh @default_mesh {{topology = @default, axes = ["rank"], shape = array<i64: 16>, policy = "all_available", endpoints = array<i64>}}
  func.func @main(%arg0: tensor<{RANK_COUNT}x{local_elements}x{element_type}>) -> tensor<{RANK_COUNT}x{local_elements}x{element_type}> {{
    %sharded = stablehlo.custom_call @Sharding(%arg0) {{
      backend_config = "",
      mhlo.sharding = "{{devices=[{RANK_COUNT},1]{devices}}}"
    }} : (tensor<{RANK_COUNT}x{local_elements}x{element_type}>) -> tensor<{RANK_COUNT}x{local_elements}x{element_type}>
    %sum = stablehlo.add %sharded, %sharded : tensor<{RANK_COUNT}x{local_elements}x{element_type}>
    %zero = stablehlo.constant dense<0.0> : tensor<{element_type}>
    %result = "stablehlo.reduce"(%sum, %zero) ({{
    ^bb0(%lhs: tensor<{element_type}>, %rhs: tensor<{element_type}>):
      %value = stablehlo.add %lhs, %rhs : tensor<{element_type}>
      stablehlo.return %value : tensor<{element_type}>
    }}) {{dimensions = array<i64: 0>}} : (tensor<{RANK_COUNT}x{local_elements}x{element_type}>, tensor<{element_type}>) -> tensor<{local_elements}x{element_type}>
    %broadcast = "stablehlo.broadcast_in_dim"(%result) {{
      broadcast_dimensions = array<i64: 1>
    }} : (tensor<{local_elements}x{element_type}>) -> tensor<{RANK_COUNT}x{local_elements}x{element_type}>
    %tagged = stablehlo.add %broadcast, %sharded : tensor<{RANK_COUNT}x{local_elements}x{element_type}>
    return %tagged : tensor<{RANK_COUNT}x{local_elements}x{element_type}>
  }}
}}
'''
    metadata = {
        "name": "forward",
        "stablehlo_version": "0.0.0",
        "input_signature": [
            {
                "shape": [RANK_COUNT, local_elements],
                "dtype": metadata_dtype,
                "dynamic_dims": [],
            }
        ],
        "output_signature": [
            {
                "shape": [RANK_COUNT, local_elements],
                "dtype": metadata_dtype,
                "dynamic_dims": [],
            }
        ],
        "input_locations": [
            {"type_": "input_arg", "position": 0, "name": "input"}
        ],
        "unused_inputs": [],
    }
    (source / "functions" / "forward.mlir").write_text(module)
    (source / "functions" / "forward.meta").write_text(
        json.dumps(metadata, separators=(",", ":")) + "\n"
    )
    return source


def validate_manifest(
    package: pathlib.Path,
    local_elements: int = LOCAL_ELEMENTS,
    element_type: str = "f16",
) -> dict[tuple[int, str, int], int]:
    if element_type not in ELEMENT_TYPES:
        raise RuntimeError(f"unsupported fixture element type {element_type}")
    metadata_dtype, element_dtype = ELEMENT_TYPES[element_type]
    metadata = json.loads((package / "functions" / "forward.meta").read_text())
    boundary = metadata.get("distributed_boundary")
    if not isinstance(boundary, dict) or boundary.get("logical_rank_count") != RANK_COUNT:
        raise RuntimeError("SPMD helper did not write the 16-rank boundary")
    inputs = boundary.get("inputs")
    outputs = boundary.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise RuntimeError("SPMD helper did not write the input boundary")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("SPMD helper did not write the output boundary")
    input_binding = inputs[0]
    output_binding = outputs[0]
    expected_input_ranks = [
        {
            "rank": rank,
            "replica_id": 0,
            "offsets": [rank, 0],
            "sizes": [1, local_elements],
            "strides": [1, 1],
        }
        for rank in range(RANK_COUNT)
    ]
    expected_output_ranks = [
        {
            "rank": rank,
            "replica_id": 0,
            "offsets": [rank, 0],
            "sizes": [1, local_elements],
            "strides": [1, 1],
        }
        for rank in range(RANK_COUNT)
    ]
    if input_binding != {
        "argument_index": 0,
        "distribution": "partitioned",
        "global_shape": [RANK_COUNT, local_elements],
        "local_shape": [1, local_elements],
        "dtype": metadata_dtype,
        "ranks": expected_input_ranks,
    }:
        raise RuntimeError("SPMD helper produced an unexpected input partition")
    if output_binding != {
        "result_index": 0,
        "distribution": "partitioned",
        "global_shape": [RANK_COUNT, local_elements],
        "local_shape": [1, local_elements],
        "dtype": metadata_dtype,
        "ranks": expected_output_ranks,
    }:
        raise RuntimeError("SPMD helper produced an unexpected output partition")

    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="Direct-DTE case",
    )
    if (
        manifest.get("rank_count") != RANK_COUNT
    ):
        raise RuntimeError(
            "Direct-DTE case did not produce the closed schema-v7 kernel launch"
        )
    modules = manifest.get("modules")
    if not isinstance(modules, list) or len(modules) != 1:
        raise RuntimeError("Direct-DTE case must produce one shared module")
    module = modules[0]
    if module.get("exports") != runtime_launch.expected_kernel_module_exports(
        runtime_launch.CLUSTER_KERNEL_LAUNCH
    ) or not (package / module["path"]).is_file():
        raise RuntimeError("Direct-DTE shared module exports are invalid")

    resources = manifest.get("resources")
    entries = manifest.get("entries")
    if not isinstance(resources, list) or not isinstance(entries, list):
        raise RuntimeError("Direct-DTE manifest domains are not lists")
    resources_by_rank: dict[int, list[dict[str, object]]] = {
        rank: [] for rank in range(RANK_COUNT)
    }
    for resource in resources:
        if not isinstance(resource, dict) or resource.get("rank") not in resources_by_rank:
            raise RuntimeError("Direct-DTE resource rank is invalid")
        resources_by_rank[resource["rank"]].append(resource)

    host_bindings: dict[tuple[int, str, int], int] = {}
    if sorted(entry.get("rank") for entry in entries) != list(range(RANK_COUNT)):
        raise RuntimeError("Direct-DTE entry rank domain is incomplete")
    for entry in entries:
        rank = entry["rank"]
        if (
            entry.get("module") != module.get("id")
            or entry.get("transport", {}).get("kind") != "direct_dte"
            or entry["transport"].get("status_abi") != STATUS_ABI
            or entry["transport"].get("host_watchdog_required") is not True
        ):
            raise RuntimeError(f"rank {rank} has an invalid Direct-DTE contract")
        rank_resources = resources_by_rank[rank]
        status = [r for r in rank_resources if r.get("role") == "transport_status"]
        if (
            len(status) != 1
            or status[0].get("type") != {"dtype": "u32", "shape": [1]}
            or status[0].get("bytes") != STATUS_STORAGE_BYTES
            or status[0].get("alignment") != STATUS_STORAGE_ALIGNMENT
        ):
            raise RuntimeError(f"rank {rank} has an invalid transport status resource")
        typed_host_resources = {
            (resource.get("role"), resource.get("role_index")): resource.get("type")
            for resource in rank_resources
            if resource.get("host_visible")
        }
        if typed_host_resources != {
            ("user_input", 0): {
                "dtype": element_type,
                "shape": [1, local_elements],
            },
            ("output", 0): {
                "dtype": element_type,
                "shape": [1, local_elements],
            },
        }:
            raise RuntimeError(f"rank {rank} has invalid typed host resources")
        for resource in rank_resources:
            if not resource.get("host_visible"):
                continue
            if resource.get("bytes") != local_elements * element_dtype.itemsize:
                raise RuntimeError(
                    f"rank {rank} host resource byte size is invalid"
                )
            key = (rank, resource.get("role"), resource.get("role_index"))
            if key in host_bindings or key[1:] not in {
                ("user_input", 0),
                ("output", 0),
            }:
                raise RuntimeError(f"rank {rank} has an unexpected host resource")
            host_bindings[key] = resource["id"]
    expected = {
        (rank, role, 0)
        for rank in range(RANK_COUNT)
        for role in ("user_input", "output")
    }
    if set(host_bindings) != expected:
        raise RuntimeError("Direct-DTE host resources are not all-and-only")
    return host_bindings


def write_raw_files(
    work_dir: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    local_elements: int = LOCAL_ELEMENTS,
) -> list[str]:
    raw = work_dir / "raw"
    raw.mkdir()
    arguments: list[str] = []
    lanes = np.arange(local_elements, dtype=np.int32)
    values_i32 = [
        (lanes % 16) * 16
        if rank == 0
        else np.full(local_elements, rank * 4, dtype=np.int32)
        for rank in range(RANK_COUNT)
    ]
    reduced_i32 = np.sum(
        np.stack(values_i32, axis=0) * 2,
        axis=0,
        dtype=np.int32,
    )
    expected_payloads: set[bytes] = set()
    for rank in range(RANK_COUNT):
        input_ = values_i32[rank].astype("<f2")
        expected_i32 = reduced_i32 + values_i32[rank]
        expected = expected_i32.astype("<f2")
        if (
            not np.array_equal(input_.astype(np.int32), values_i32[rank])
            or not np.array_equal(expected.astype(np.int32), expected_i32)
            or not np.all(np.isfinite(input_))
            or not np.all(np.isfinite(expected))
        ):
            raise RuntimeError(
                "Direct-DTE f16 sentinels are not finite and exact"
            )
        expected_bytes = expected.tobytes()
        expected_payloads.add(expected_bytes)
        input_path = raw / f"input_{rank:02d}.f16.raw"
        expected_path = raw / f"expected_{rank:02d}.f16.raw"
        input_path.write_bytes(input_.tobytes())
        expected_path.write_bytes(expected_bytes)
        arguments.extend(
            ["--resource", f"{bindings[(rank, 'user_input', 0)]}={input_path}"]
        )
        arguments.extend(
            ["--expected", f"{bindings[(rank, 'output', 0)]}={expected_path}"]
        )
    if len(expected_payloads) != RANK_COUNT:
        raise RuntimeError("Direct-DTE outputs are not rank-distinct sentinels")
    return arguments


def verify_no_card_evidence(stdout: str, *, instrumentation_expected: bool) -> None:
    required = {
        "package: id=0 schema=7 ranks=16",
        f"invocation_ranks: {RANK_COUNT}",
        "board_execution: false",
    }
    output_lines = set(stdout.splitlines())
    if not required.issubset(output_lines):
        raise RuntimeError("no-card output omitted Direct-DTE validation")
    instrumentation_ready = PROFILE_INSTRUMENTATION_READY in stdout
    if instrumentation_ready != instrumentation_expected:
        raise RuntimeError(
            "no-card launch did not prove the exact profile instrumentation "
            "activation boundary"
        )


def verify_board_evidence(
    stdout: str,
    bindings: dict[tuple[int, str, int], int],
) -> None:
    required_evidence = {
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        f"invocation_ranks: {RANK_COUNT}",
        "launch_pattern: cluster-x16",
        "logical_tile_execution_basis: cluster-pid-and-exact-rank-slices",
        "logical_tile_domain: 0..15",
        "physical_execution_claim: none",
        "board_execution: true",
    }
    if not required_evidence.issubset(set(stdout.splitlines())):
        raise RuntimeError("board output omitted complete Direct-DTE evidence")
    output_matches = re.findall(
        rf"^output_compare: resource=(\d+) bytes={LOCAL_ELEMENTS * 2} "
        r"exact=true$",
        stdout,
        re.MULTILINE,
    )
    expected_output_ids = {
        bindings[(rank, "output", 0)] for rank in range(RANK_COUNT)
    }
    if len(output_matches) != RANK_COUNT or {
        int(resource) for resource in output_matches
    } != expected_output_ids:
        raise RuntimeError("board output omitted exact rank output evidence")
    # BoardRuntime checks every typed Direct-DTE status resource for Success
    # before it runs DeviceToHost/Cleanup. Reaching both stages therefore
    # retains the existing exact 16-rank status gate without inventing a second
    # host-visible status protocol in this harness.


def verify_profile_report(
    package: pathlib.Path, stdout: str
) -> tuple[int, pathlib.Path]:
    require_profile_instrumentation_permissions(package)
    instrumentation = pathlib.Path(f"{package}.profile")
    runs = instrumentation / "runs"
    current = runs / "current"
    if not current.is_symlink():
        raise RuntimeError("profile report current entry is not a symlink")
    run_directory = current.resolve(strict=True)
    if run_directory.parent != runs.resolve(strict=True):
        raise RuntimeError("profile report current entry escapes its runs directory")
    entries = tuple(run_directory.iterdir())
    if len(entries) != 3 or any(not path.is_file() for path in entries):
        raise RuntimeError(
            "profile report does not contain exactly three regular files"
        )
    members = {path.name: path for path in entries}
    if set(members) != {"evidence.json", "analysis.json", "index.html"}:
        raise RuntimeError(
            "profile report does not contain exactly the three public output files"
        )

    evidence = json.loads(members["evidence.json"].read_text())
    analysis = json.loads(members["analysis.json"].read_text())
    if (
        evidence.get("schema") != "wafer.profile.evidence"
        or evidence.get("schema_version") != PROFILE_EVIDENCE_SCHEMA_VERSION
        or evidence.get("run_id") != run_directory.name
    ):
        raise RuntimeError("profile evidence identity is invalid")
    samples = evidence.get("measurement", {}).get("samples")
    trace = evidence.get("experiment", {}).get("trace")
    trace_tiles = trace.get("tiles") if isinstance(trace, dict) else None
    if (
        not isinstance(samples, list)
        or len(samples) != PROFILE_PRIMARY_EXECUTION_COUNT
        or samples[0].get("sample_id") != "primary"
        or samples[0].get("sample_index") != 0
        or not isinstance(samples[0].get("device_elapsed_ns"), int)
        or samples[0]["device_elapsed_ns"] < 0
        or samples[0].get("device_timer_kind") != "tx-stream-events"
        or not isinstance(samples[0].get("host_submit_ns"), int)
        or samples[0]["host_submit_ns"] < 0
        or not isinstance(
            samples[0].get("host_launch_to_completion_ns"), int
        )
        or samples[0]["host_launch_to_completion_ns"] < 0
        or samples[0]["host_submit_ns"]
        > samples[0]["host_launch_to_completion_ns"]
        or not isinstance(
            samples[0].get("completion_observation_resolution_ns"), int
        )
        or samples[0]["completion_observation_resolution_ns"] < 0
        or not isinstance(trace, dict)
        or trace.get("complete") is not True
        or not isinstance(trace_tiles, list)
        or len(trace_tiles) != RANK_COUNT
        or sorted(tile.get("tile") for tile in trace_tiles) != list(
            range(RANK_COUNT)
        )
    ):
        raise RuntimeError(
            "profile evidence does not contain one Primary and 16 trace tiles"
        )
    direct_dte_source_tiles = {
        int(tile["tile"])
        for tile in trace_tiles
        for event in tile.get("events", [])
        if isinstance(event, dict)
        and event.get("engine") == "DIRECT_DTE"
        and event.get("kind")
        in ("direct-dte-wait", "direct-dte-completion-wait")
        and event.get("operation_span_valid") is True
        and isinstance(event.get("operation_begin_cycle"), int)
        and isinstance(event.get("operation_end_cycle"), int)
        and event["operation_end_cycle"] > event["operation_begin_cycle"]
    }
    if direct_dte_source_tiles != set(range(RANK_COUNT)):
        raise RuntimeError(
            "profile trace omitted a real positive Direct-DTE phase event "
            "for one or more tiles"
        )

    final = analysis.get("program")
    validity = analysis.get("validity")
    if (
        analysis.get("schema") != "wafer.profile.analysis"
        or analysis.get("schema_version") != PROFILE_ANALYSIS_SCHEMA_VERSION
        or analysis.get("run_id") != run_directory.name
        or not analysis.get("valid")
        or not isinstance(validity, dict)
        or validity.get("trace") is not True
        or validity.get("cost_accounting") is not True
        or not isinstance(final, dict)
    ):
        raise RuntimeError("profile analysis identity or trace validity is invalid")
    duration = final.get("duration")
    output = final.get("output")
    tiles = final.get("tiles")
    host_submit = samples[0]["host_submit_ns"]
    host_envelope = samples[0]["host_launch_to_completion_ns"]
    resolution = samples[0]["completion_observation_resolution_ns"]
    host_envelope_available = host_envelope > 0
    completion_fraction = (
        duration.get("completion_observation_fraction")
        if isinstance(duration, dict)
        else None
    )
    expected_completion_fraction = (
        resolution / host_envelope if host_envelope_available else None
    )
    if (
        not isinstance(duration, dict)
        or duration.get("sample_id") != "primary"
        or duration.get("sample_index") != 0
        or duration.get("device_elapsed_ns")
        != samples[0]["device_elapsed_ns"]
        or duration.get("device_timer_kind") != "tx-stream-events"
        or duration.get("host_submit_ns") != samples[0]["host_submit_ns"]
        or duration.get("host_launch_to_completion_ns")
        != samples[0]["host_launch_to_completion_ns"]
        or duration.get("completion_observation_resolution_ns")
        != samples[0]["completion_observation_resolution_ns"]
        or duration.get("host_non_submit_envelope_ns")
        != host_envelope - host_submit
        or duration.get("host_envelope_available")
        is not host_envelope_available
        or (
            host_envelope_available
            and (
                isinstance(completion_fraction, bool)
                or not isinstance(completion_fraction, (int, float))
                or abs(
                    completion_fraction - expected_completion_fraction
                )
                > 1e-15
            )
        )
        or (
            not host_envelope_available
            and completion_fraction is not None
        )
        or not isinstance(
            duration.get("host_completion_high_resolution"), bool
        )
        or (
            not host_envelope_available
            and duration.get("host_completion_high_resolution") is not False
        )
        or not duration.get("qualified")
        or not isinstance(output, dict)
        or not output.get("primary_output_validated")
        or not output.get("diagnostic_captures_match_primary")
        or not isinstance(tiles, list)
        or len(tiles) != RANK_COUNT
        or sorted(tile.get("tile") for tile in tiles) != list(range(RANK_COUNT))
    ):
        raise RuntimeError(
            "profile analysis failed latency, output, or tile qualification"
        )

    for tile in tiles:
        entry_cycles = tile.get("trace_entry_cpu_cycles")
        partition = tile.get("semantic_partition")
        segments = tile.get("semantic_timeline_segments")
        overhead = tile.get("trace_overhead_overlay")
        overhead_rows = (
            overhead.get("rows") if isinstance(overhead, dict) else None
        )
        overlay_cycles = (
            overhead.get("exclusive_component_cycles")
            if isinstance(overhead, dict)
            else None
        )
        inside_overlay_cycles = (
            overhead.get("inside_entry_known_overhead_cycles")
            if isinstance(overhead, dict)
            else None
        )
        outside_overlay_cycles = (
            overhead.get("outside_entry_overhead_cycles")
            if isinstance(overhead, dict)
            else None
        )
        if (
            not isinstance(entry_cycles, int)
            or entry_cycles < 0
            or not isinstance(partition, dict)
            or partition.get("exclusive_accounting_valid") is not True
            or partition.get("entry_cycles") != entry_cycles
            or partition.get("exclusive_cycles") != entry_cycles
            or not isinstance(segments, list)
            or any(
                not isinstance(segment, dict)
                or not isinstance(segment.get("cycles"), int)
                or segment["cycles"] < 0
                for segment in segments
            )
            or not isinstance(overhead, dict)
            or not isinstance(overhead_rows, list)
            or not isinstance(overlay_cycles, int)
            or overlay_cycles < 0
            or not isinstance(inside_overlay_cycles, int)
            or inside_overlay_cycles < 0
            or not isinstance(outside_overlay_cycles, int)
            or outside_overlay_cycles < 0
            or overlay_cycles
            != inside_overlay_cycles + outside_overlay_cycles
            or any(
                not isinstance(row, dict)
                or not isinstance(row.get("cycles"), int)
                or row["cycles"] < 0
                for row in overhead_rows
            )
            or overlay_cycles
            != sum(row["cycles"] for row in overhead_rows)
            or overhead.get("components_exclusive") is not True
            or overhead.get("non_additive_to_semantic_partition") is not True
            or overhead.get("positioning") != "aggregate-only"
            or overhead.get("coverage") != "measured-categories-only"
        ):
            raise RuntimeError(
                "profile analysis contains invalid or negative Kcore cost"
            )
        direct_dte = next(
            (
                engine
                for engine in tile.get("engines", [])
                if isinstance(engine, dict)
                and engine.get("engine") == "DIRECT_DTE"
            ),
            None,
        )
        if direct_dte is None:
            raise RuntimeError("profile analysis omitted a Direct-DTE aggregate")
        wait_cycles = direct_dte.get("wait_window_cpu_cycles")
        if not (
            direct_dte.get("wait_windows_valid") is True
            and isinstance(wait_cycles, int)
            and wait_cycles > 0
            and direct_dte.get("wait_window_count", 0) > 0
        ):
            raise RuntimeError(
                "profile analysis omitted a real Direct-DTE aggregate "
                f"for tile {tile.get('tile')}"
            )
    timeline_events = final.get("timeline_events")
    positive_direct_dte_phase_tiles = {
        int(event["tile"])
        for event in timeline_events
        if isinstance(event, dict)
        and isinstance(event.get("tile"), int)
        and event.get("engine") == "DIRECT_DTE"
        and event.get("kind")
        in ("direct-dte-wait", "direct-dte-completion-wait")
        and isinstance(event.get("operation_window_cpu_cycles"), int)
        and event["operation_window_cpu_cycles"] > 0
        and event.get("duration_status") == "Measured"
    } if isinstance(timeline_events, list) else set()
    if positive_direct_dte_phase_tiles != set(range(RANK_COUNT)):
        raise RuntimeError(
            "profile analysis omitted a real measured Direct-DTE phase "
            "for one or more tiles"
        )

    expected_report = current / "index.html"
    if f"profile_report: {expected_report}" not in stdout:
        raise RuntimeError("wafer-run did not return the stable profile report path")
    return samples[0]["device_elapsed_ns"], expected_report


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if args.profile and args.repeat != 1:
        raise RuntimeError("--profile requires exactly one fixed campaign")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("Direct-DTE hardware execution is not armed", file=sys.stderr)
        return 77

    source = write_fixture(args.work_dir)
    package = args.work_dir / "package"
    ordinary_package = package
    if args.profile:
        ordinary_package = args.work_dir / "ordinary-package"
        compile_package(
            args, source, ordinary_package, profile=False
        )
        compile_package(args, source, package, profile=True)
        require_byte_identical_packages(ordinary_package, package)
        require_profile_instrumentation_permissions(package)
    else:
        compile_package(args, source, package, profile=False)

    bindings = validate_manifest(package)
    if args.no_card:
        no_card_packages = (
            (ordinary_package, package)
            if args.profile
            else (package,)
        )
        for no_card_package in no_card_packages:
            result = run([
                str(args.wafer_run),
                "--package-dir",
                str(no_card_package),
                "--all-ranks",
                "--no-card",
                "--direct-dte-status-abi",
                STATUS_ABI,
                "--supports-host-watchdog",
            ])
            verify_no_card_evidence(
                result.stdout,
                instrumentation_expected=(
                    args.profile and no_card_package == package
                ),
            )
        print("direct_dte_no_card: verified")
        print(f"direct_dte_no_card_profile: {str(args.profile).lower()}")
        return 0

    required = [
        args.expected_runtime_version,
        args.expected_device_name,
        args.expected_pci_bus_id,
        args.expected_tile_count,
        args.expected_runtime_library_sha256,
    ]
    if any(value is None for value in required):
        raise RuntimeError("board execution requires complete qualification arguments")
    if args.expected_tile_count != RANK_COUNT:
        raise RuntimeError(f"--expected-tile-count must be {RANK_COUNT}")

    resource_args = write_raw_files(args.work_dir, bindings)
    command = [
        str(args.wafer_run),
        "--package-dir",
        str(package),
        "--all-ranks",
        "--board",
        "--device-id",
        str(args.device_id),
        "--expected-runtime-version",
        str(args.expected_runtime_version),
        "--expected-device-name",
        args.expected_device_name,
        "--expected-pci-bus-id",
        args.expected_pci_bus_id,
        "--expected-tile-count",
        str(args.expected_tile_count),
        "--expected-runtime-library-sha256",
        args.expected_runtime_library_sha256,
        "--completion-timeout-ms",
        str(args.completion_timeout_ms),
        *resource_args,
    ]
    if args.profile:
        result = run(
            command,
            timeout_seconds=(
                max(
                    300.0,
                    PROFILE_CAMPAIGN_LAUNCH_COUNT
                    * args.completion_timeout_ms
                    / 1000.0
                    + DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS,
                )
            ),
        )
        verify_board_evidence(result.stdout, bindings)
        device_duration, report = verify_profile_report(
            package, result.stdout
        )
        print(
            "direct_dte_profile_campaign: pass "
            f"launches={PROFILE_CAMPAIGN_LAUNCH_COUNT} "
            f"primary={PROFILE_PRIMARY_EXECUTION_COUNT}"
        )
        print(f"direct_dte_profile_device_duration_ns: {device_duration}")
        print(f"direct_dte_profile_report: {report}")
        print(result.stdout, end="")
        return 0

    for iteration in range(args.repeat):
        result = run(
            command,
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        verify_board_evidence(result.stdout, bindings)
        print(f"direct_dte_iteration: {iteration + 1}/{args.repeat} exact=true")
        print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
