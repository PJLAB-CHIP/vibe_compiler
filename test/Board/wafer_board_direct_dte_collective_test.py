#!/usr/bin/env python3
"""Compile and run a 16-rank sharded Add/reduction Direct-DTE case."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

import wafer_runtime_launch_contract as runtime_launch


RANK_COUNT = 16
LOCAL_ELEMENTS = 128
LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
STATUS_ABI = "wafer-direct-dte-status-v2"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS = 30
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
        raise RuntimeError("SPMD helper did not publish the 16-rank boundary")
    inputs = boundary.get("inputs")
    outputs = boundary.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise RuntimeError("SPMD helper did not publish the input boundary")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("SPMD helper did not publish the output boundary")
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
            "Direct-DTE case did not produce the closed schema-v6 kernel launch"
        )
    modules = manifest.get("modules")
    if not isinstance(modules, list) or len(modules) != 1:
        raise RuntimeError("Direct-DTE case must publish one shared module")
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
        lanes * 16
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


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("Direct-DTE hardware execution is not armed", file=sys.stderr)
        return 77

    source = write_fixture(args.work_dir)
    package = args.work_dir / "package"
    run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(package),
            f"--execution-ranks={RANK_COUNT}",
            "--target-profile=wafer-tx81-single-card-kernel-v1",
            f"--launch-kind={LAUNCH_KIND}",
        ]
    )
    bindings = validate_manifest(package)
    resource_args = write_raw_files(args.work_dir, bindings)
    if args.no_card:
        result = run(
            [
                str(args.wafer_run),
                "--package-dir",
                str(package),
                "--all-ranks",
                "--no-card",
                "--direct-dte-status-abi",
                STATUS_ABI,
                "--supports-host-watchdog",
            ]
        )
        if "board_execution: false" not in result.stdout:
            raise RuntimeError("no-card output omitted Direct-DTE validation")
        print("direct_dte_no_card: verified")
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
    for iteration in range(args.repeat):
        result = run(
            [
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
            ],
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        required_evidence = {
            "launch_pattern: cluster-x16",
            "logical_tile_execution_basis: cluster-pid-and-exact-rank-slices",
            "logical_tile_domain: 0..15",
            "physical_execution_claim: none",
            "board_execution: true",
        }
        if not required_evidence.issubset(set(result.stdout.splitlines())):
            raise RuntimeError("board output omitted complete Direct-DTE evidence")
        output_matches = re.findall(
            rf"^output_compare: resource=(\d+) bytes={LOCAL_ELEMENTS * 2} "
            r"exact=true$",
            result.stdout,
            re.MULTILINE,
        )
        expected_output_ids = {
            bindings[(rank, "output", 0)] for rank in range(RANK_COUNT)
        }
        if len(output_matches) != RANK_COUNT or {
            int(resource) for resource in output_matches
        } != expected_output_ids:
            raise RuntimeError("board output omitted exact rank output evidence")
        print(f"direct_dte_iteration: {iteration + 1}/{args.repeat} exact=true")
        print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
