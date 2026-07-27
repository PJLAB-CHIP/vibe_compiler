#!/usr/bin/env python3
"""Compile and run a full-4096 K-sharded f16 GEMM on 16 TX ranks."""

from __future__ import annotations

import argparse
import hashlib
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
M = 4096
K = 4096
N = 4096
LOCAL_K = K // RANK_COUNT
F16_BYTES = np.dtype("<f2").itemsize
LHS_LOCAL_SHAPE = [M, LOCAL_K]
RHS_LOCAL_SHAPE = [LOCAL_K, N]
OUTPUT_SHAPE = [M, N]
LHS_BYTES = M * LOCAL_K * F16_BYTES
RHS_BYTES = LOCAL_K * N * F16_BYTES
OUTPUT_BYTES = M * N * F16_BYTES
EXPECTED_SHA256 = "f82ced1cea5d133a8f4640a527025333a80cbf2e49e7a529dc9c7c941e20360f"
TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
STATUS_ABI = "wafer-direct-dte-status-v2"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS = 60
DEVICES = ",".join(str(rank) for rank in range(RANK_COUNT))
RANK_GROUP = ", ".join(str(rank) for rank in range(RANK_COUNT))
LHS_SHARDING = f"{{devices=[1,{RANK_COUNT}]{DEVICES}}}"
RHS_SHARDING = f"{{devices=[{RANK_COUNT},1]{DEVICES}}}"
OUTPUT_SHARDING = "{replicated}"

MODULE = f"""module {{
  func.func @main(
      %lhs: tensor<{M}x{K}xf16> {{mhlo.sharding = "{LHS_SHARDING}"}},
      %rhs: tensor<{K}x{N}xf16> {{mhlo.sharding = "{RHS_SHARDING}"}})
      -> (tensor<{M}x{N}xf16> {{mhlo.sharding = "{OUTPUT_SHARDING}"}}) {{
    %result = "stablehlo.dot_general"(%lhs, %rhs) {{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    }} : (tensor<{M}x{K}xf16>, tensor<{K}x{N}xf16>) -> tensor<{M}x{N}xf16>
    return %result : tensor<{M}x{N}xf16>
  }}
}}
"""

METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [M, K], "dtype": "float16", "dynamic_dims": []},
        {"shape": [K, N], "dtype": "float16", "dynamic_dims": []},
    ],
    "output_signature": [{"shape": [M, N], "dtype": "float16", "dynamic_dims": []}],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "lhs"},
        {"type_": "input_arg", "position": 1, "name": "rhs"},
    ],
    "unused_inputs": [],
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
    parser.add_argument("--completion-timeout-ms", type=int, default=600000)
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
            "one-shot full-4096 Direct-DTE GEMM exceeded its outer deadline; "
            "the process was killed and this test will not retry or invoke "
            "reset/power operations; board state requires external read-only "
            "qualification"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}"
        )
    return result


def write_source_program(work_dir: pathlib.Path) -> pathlib.Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(MODULE)
    (source / "functions" / "forward.meta").write_text(
        json.dumps(METADATA, separators=(",", ":")) + "\n"
    )
    return source


def require_boundary_binding(
    binding: object,
    *,
    index_field: str,
    index: int,
    distribution: str,
    global_shape: list[int],
    local_shape: list[int],
    expected_geometry: dict[int, tuple[int, list[int], list[int]]],
) -> None:
    if not isinstance(binding, dict):
        raise RuntimeError("distributed boundary binding must be an object")
    if binding != {
        index_field: index,
        "distribution": distribution,
        "global_shape": global_shape,
        "local_shape": local_shape,
        "dtype": "float16",
        "ranks": [
            {
                "rank": rank,
                "replica_id": expected_geometry[rank][0],
                "offsets": expected_geometry[rank][1],
                "sizes": expected_geometry[rank][2],
                "strides": [1, 1],
            }
            for rank in range(RANK_COUNT)
        ],
    }:
        raise RuntimeError(
            f"unexpected {index_field}={index} distributed boundary geometry"
        )


def validate_structured_program(package: pathlib.Path) -> None:
    metadata = json.loads((package / "functions" / "forward.meta").read_text())
    boundary = metadata.get("distributed_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("version") != 1
        or boundary.get("logical_rank_count") != RANK_COUNT
    ):
        raise RuntimeError("SPMD helper omitted the exact rank-16 boundary")
    inputs = boundary.get("inputs")
    outputs = boundary.get("outputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 2
        or not isinstance(outputs, list)
        or len(outputs) != 1
    ):
        raise RuntimeError("K-sharded GEMM boundary arity is invalid")
    input_by_index = {
        binding.get("argument_index"): binding
        for binding in inputs
        if isinstance(binding, dict)
    }
    if set(input_by_index) != {0, 1}:
        raise RuntimeError("K-sharded GEMM input indices are not all-and-only")

    lhs_geometry = {
        rank: (0, [0, rank * LOCAL_K], LHS_LOCAL_SHAPE) for rank in range(RANK_COUNT)
    }
    rhs_geometry = {
        rank: (0, [rank * LOCAL_K, 0], RHS_LOCAL_SHAPE) for rank in range(RANK_COUNT)
    }
    output_geometry = {rank: (rank, [0, 0], OUTPUT_SHAPE) for rank in range(RANK_COUNT)}
    require_boundary_binding(
        input_by_index[0],
        index_field="argument_index",
        index=0,
        distribution="partitioned",
        global_shape=[M, K],
        local_shape=LHS_LOCAL_SHAPE,
        expected_geometry=lhs_geometry,
    )
    require_boundary_binding(
        input_by_index[1],
        index_field="argument_index",
        index=1,
        distribution="partitioned",
        global_shape=[K, N],
        local_shape=RHS_LOCAL_SHAPE,
        expected_geometry=rhs_geometry,
    )
    require_boundary_binding(
        outputs[0],
        index_field="result_index",
        index=0,
        distribution="replicated",
        global_shape=[M, N],
        local_shape=OUTPUT_SHAPE,
        expected_geometry=output_geometry,
    )

    structured_ir = (package / "functions" / "forward.mlir").read_text()
    required_ir = (
        f"tensor<{M}x{LOCAL_K}xf16>",
        f"tensor<{LOCAL_K}x{N}xf16>",
        f"tensor<{M}x{N}xf16>",
        "linalg.matmul",
        "wafer.linalg_ext.collective.all_reduce",
        f"rank_group = array<i64: {RANK_GROUP}>",
    )
    missing_ir = [fragment for fragment in required_ir if fragment not in structured_ir]
    if missing_ir:
        raise RuntimeError(
            f"post-SPMD structured program omitted required IR: {missing_ir}"
        )
    if (
        structured_ir.count("linalg.matmul") != 1
        or structured_ir.count("wafer.linalg_ext.collective.all_reduce") != 1
        or "stablehlo." in structured_ir
    ):
        raise RuntimeError(
            "post-SPMD structured program is not one local full-K GEMM plus "
            "one normalized all-reduce"
        )


def validate_manifest(
    package: pathlib.Path,
) -> tuple[dict[tuple[int, str, int], int], set[int]]:
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="full-4096 GEMM",
    )
    if (
        manifest.get("rank_count") != RANK_COUNT
        or manifest.get("target", {}).get("profile") != TARGET_PROFILE
    ):
        raise RuntimeError("full-4096 GEMM package target contract is invalid")
    modules = manifest.get("modules")
    if (
        not isinstance(modules, list)
        or len(modules) != 1
        or modules[0].get("exports")
        != runtime_launch.expected_kernel_module_exports(
            runtime_launch.CLUSTER_KERNEL_LAUNCH
        )
        or not (package / modules[0].get("path", "")).is_file()
    ):
        raise RuntimeError("full-4096 GEMM must publish one cluster shared ELF")
    module_id = modules[0].get("id")

    entries = manifest.get("entries")
    completions = manifest.get("completions")
    resources = manifest.get("resources")
    if not all(isinstance(value, list) for value in (entries, completions, resources)):
        raise RuntimeError("package entry/completion/resource domains must be lists")
    if sorted(entry.get("rank") for entry in entries) != list(range(RANK_COUNT)):
        raise RuntimeError("package entries do not exactly cover ranks 0..15")
    if sorted(item.get("rank") for item in completions) != list(range(RANK_COUNT)):
        raise RuntimeError("package completions do not exactly cover ranks 0..15")
    if len(resources) != 4 * RANK_COUNT:
        raise RuntimeError("full-4096 GEMM requires four resources per rank")

    resources_by_rank: dict[int, list[dict[str, object]]] = {
        rank: [] for rank in range(RANK_COUNT)
    }
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("rank") not in resources_by_rank
        ):
            raise RuntimeError("package contains an invalid resource rank")
        resources_by_rank[resource["rank"]].append(resource)

    bindings: dict[tuple[int, str, int], int] = {}
    output_ids: set[int] = set()
    completion_by_rank = {item["rank"]: item for item in completions}
    for entry in entries:
        rank = entry["rank"]
        rank_resources = resources_by_rank[rank]
        resource_by_role = {
            (resource.get("role"), resource.get("role_index")): resource
            for resource in rank_resources
        }
        expected_resources = {
            ("user_input", 0): (
                {"dtype": "f16", "shape": LHS_LOCAL_SHAPE},
                LHS_BYTES,
                "read_only",
            ),
            ("user_input", 1): (
                {"dtype": "f16", "shape": RHS_LOCAL_SHAPE},
                RHS_BYTES,
                "read_only",
            ),
            ("output", 0): (
                {"dtype": "f16", "shape": OUTPUT_SHAPE},
                OUTPUT_BYTES,
                "write_only",
            ),
            ("transport_status", 0): (
                {"dtype": "u32", "shape": [1]},
                STATUS_STORAGE_BYTES,
                "read_write",
            ),
        }
        if set(resource_by_role) != set(expected_resources):
            raise RuntimeError(f"rank {rank} resource roles are not all-and-only")
        for key, (type_, bytes_, access) in expected_resources.items():
            resource = resource_by_role[key]
            expected_alignment = (
                STATUS_STORAGE_ALIGNMENT if key[0] == "transport_status" else 256
            )
            expected_host_visible = key[0] != "transport_status"
            if (
                resource.get("type") != type_
                or resource.get("bytes") != bytes_
                or resource.get("alignment") != expected_alignment
                or resource.get("access") != access
                or resource.get("host_visible") is not expected_host_visible
                or not isinstance(resource.get("id"), int)
            ):
                raise RuntimeError(f"rank {rank} resource {key} is invalid")
            if expected_host_visible:
                bindings[(rank, key[0], key[1])] = resource["id"]
            if key == ("output", 0):
                output_ids.add(resource["id"])

        status = resource_by_role[("transport_status", 0)]
        completion = completion_by_rank[rank]
        transport = entry.get("transport")
        if (
            entry.get("module") != module_id
            or entry.get("terminal_completion") != completion.get("id")
            or completion.get("kind") != "entry_return"
            or transport
            != {
                "kind": "direct_dte",
                "status_resource": status["id"],
                "status_abi": STATUS_ABI,
                "host_watchdog_required": True,
            }
        ):
            raise RuntimeError(f"rank {rank} Direct-DTE entry contract is invalid")
        expected_slots = [
            (0, resource_by_role[("user_input", 0)]["id"], "read_only"),
            (1, resource_by_role[("user_input", 1)]["id"], "read_only"),
            (2, resource_by_role[("output", 0)]["id"], "write_only"),
            (3, status["id"], "read_write"),
        ]
        actual_slots = [
            (slot.get("ordinal"), slot.get("resource"), slot.get("access"))
            for slot in entry.get("slots", [])
            if isinstance(slot, dict)
        ]
        if actual_slots != expected_slots:
            raise RuntimeError(f"rank {rank} launch slots are invalid")

    expected_bindings = {
        (rank, role, index)
        for rank in range(RANK_COUNT)
        for role, index in (("user_input", 0), ("user_input", 1), ("output", 0))
    }
    if set(bindings) != expected_bindings or len(output_ids) != RANK_COUNT:
        raise RuntimeError("host bindings do not exactly cover all rank resources")
    return bindings, output_ids


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_payloads(
    work_dir: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
) -> tuple[list[str], str]:
    raw = work_dir / "raw"
    raw.mkdir()

    rhs_path = raw / "rhs_all_ranks.f16.raw"
    rhs = np.memmap(rhs_path, dtype="<f2", mode="w+", shape=tuple(RHS_LOCAL_SHAPE))
    rhs[:, :] = (1 + np.arange(N, dtype=np.int32) % 19).astype("<f2")
    rhs.flush()
    del rhs

    expected_path = raw / "expected_all_ranks.f16.raw"
    expected = np.memmap(
        expected_path, dtype="<f2", mode="w+", shape=tuple(OUTPUT_SHAPE)
    )
    column_factor = 1 + np.arange(N, dtype=np.int32) % 19
    row_block = 64
    for begin in range(0, M, row_block):
        end = min(M, begin + row_block)
        rows = np.arange(begin, end, dtype=np.int32)
        rank_sum_numerator = 136 + 16 * (rows % 17)
        block = rank_sum_numerator[:, None] * column_factor[None, :] / 16
        expected[begin:end, :] = block.astype("<f2")
    expected.flush()
    if (
        float(expected[0, 0]) != 8.5
        or float(expected[1, 18]) != 180.5
        or float(expected[M - 1, N - 1]) != 258.5
    ):
        raise RuntimeError("frozen full-4096 GEMM expected formula is invalid")
    del expected

    arguments: list[str] = []
    for rank in range(RANK_COUNT):
        lhs_path = raw / f"lhs_rank_{rank:05d}.f16.raw"
        lhs = np.memmap(lhs_path, dtype="<f2", mode="w+", shape=tuple(LHS_LOCAL_SHAPE))
        lhs[:, :] = (
            (rank + 1 + np.arange(M, dtype=np.int32)[:, None] % 17) / K
        ).astype("<f2")
        lhs.flush()
        del lhs
        arguments.extend(
            ["--resource", f"{bindings[(rank, 'user_input', 0)]}={lhs_path}"]
        )
        arguments.extend(
            ["--resource", f"{bindings[(rank, 'user_input', 1)]}={rhs_path}"]
        )
        arguments.extend(
            ["--expected", f"{bindings[(rank, 'output', 0)]}={expected_path}"]
        )

    digest = sha256_file(expected_path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"full-4096 expected digest changed: {digest} != {EXPECTED_SHA256}"
        )
    print(f"expected_sha256: {digest}")
    return arguments, digest


def verify_no_card_evidence(stdout: str) -> None:
    required = {
        "package: id=0 schema=6 ranks=16",
        "invocation_ranks: 16",
        "board_execution: false",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("no-card output omitted full-4096 Direct-DTE evidence")


def verify_board_evidence(stdout: str, output_ids: set[int]) -> None:
    required = {
        "board_stage: preflight",
        "board_stage: device-selection",
        "board_stage: resource-allocation",
        "board_stage: host-to-device",
        "board_stage: module-load",
        "board_stage: entry-resolve",
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "invocation_ranks: 16",
        "launch_pattern: cluster-prepare-main-x16",
        "logical_tile_execution_basis: cluster-pid-and-exact-rank-slices",
        "logical_tile_domain: 0..15",
        "physical_execution_claim: none",
        "board_execution: true",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("board output omitted full-4096 lifecycle evidence")
    output_matches = re.findall(
        rf"^output_compare: resource=(\d+) bytes={OUTPUT_BYTES} exact=true$",
        stdout,
        re.MULTILINE,
    )
    if (
        len(output_matches) != RANK_COUNT
        or {int(resource) for resource in output_matches} != output_ids
    ):
        raise RuntimeError("board output omitted a complete replicated exact result")
    tile_matches = re.findall(
        r"^board_tile: logical=(\d+) available=true physical_x=\d+ physical_y=\d+$",
        stdout,
        re.MULTILINE,
    )
    if sorted(int(rank) for rank in tile_matches) != list(range(RANK_COUNT)):
        raise RuntimeError("board inventory did not prove logical ranks 0..15")


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print(
            "full-4096 K-sharded GEMM hardware execution is not armed", file=sys.stderr
        )
        return 77
    if not args.no_card:
        required = [
            args.expected_runtime_version,
            args.expected_device_name,
            args.expected_pci_bus_id,
            args.expected_tile_count,
            args.expected_runtime_library_sha256,
        ]
        if any(value is None for value in required):
            raise RuntimeError(
                "board execution requires complete qualification arguments"
            )
        if args.expected_tile_count != RANK_COUNT:
            raise RuntimeError("full-4096 K-sharded GEMM requires exactly 16 tiles")

    source = write_source_program(args.work_dir)
    package = args.work_dir / "package"
    compile_result = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(package),
            f"--execution-ranks={RANK_COUNT}",
            f"--target-profile={TARGET_PROFILE}",
            f"--launch-kind={LAUNCH_KIND}",
        ]
    )
    if (
        "published verified package with execution-ranks=16"
        not in compile_result.stdout
    ):
        raise RuntimeError("wafer-compile did not publish a verified rank-16 package")
    validate_structured_program(package)
    bindings, output_ids = validate_manifest(package)

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
        verify_no_card_evidence(result.stdout)
        print("k_sharded_gemm_no_card: verified")
        print(result.stdout, end="")
        return 0

    resource_arguments, expected_digest = write_payloads(args.work_dir, bindings)
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
        *resource_arguments,
    ]
    for iteration in range(args.repeat):
        result = run(
            command,
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        verify_board_evidence(result.stdout, output_ids)
        print(
            "k_sharded_gemm_iteration: "
            f"{iteration + 1}/{args.repeat} exact=true "
            f"expected_sha256={expected_digest}"
        )
        print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"wafer_board_k_sharded_gemm_test: {error}", file=sys.stderr)
        raise SystemExit(1)
