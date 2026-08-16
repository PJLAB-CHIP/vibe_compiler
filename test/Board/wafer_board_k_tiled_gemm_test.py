#!/usr/bin/env python3
"""Compile and run a full-4096 K-tiled f16 GEMM on the 16-Tile TX domain."""

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

import torch

import wafer_runtime_launch_contract as runtime_launch

PYTORCH_BOARD_DIR = pathlib.Path(__file__).resolve().parent / "PyTorch"
if str(PYTORCH_BOARD_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_BOARD_DIR))

import wafer_pytorch_board_common as torch_reference  # noqa: E402


TILE_COUNT = 16
M = 4096
K = 4096
N = 4096
LOCAL_K = K // TILE_COUNT
F16_BYTES = 2
LHS_LOCAL_SHAPE = [M, LOCAL_K]
RHS_LOCAL_SHAPE = [LOCAL_K, N]
OUTPUT_SHAPE = [M, N]
LHS_BYTES = M * K * F16_BYTES
RHS_BYTES = K * N * F16_BYTES
OUTPUT_BYTES = M * N * F16_BYTES
TARGET_IDENTITY = "wafer-tx81-single-card"
STATUS_ABI = "wafer-direct-dte-status"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS = 60
PYTORCH_SEED = 20260803

MODULE = f"""module {{
  func.func @main(
      %lhs: tensor<{M}x{K}xf16>,
      %rhs: tensor<{K}x{N}xf16>) -> tensor<{M}x{N}xf16> {{
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


def validate_source_program(source: pathlib.Path) -> None:
    metadata = json.loads((source / "functions" / "forward.meta").read_text())
    if metadata != METADATA or "distributed_boundary" in metadata:
        raise RuntimeError("K-tiled GEMM metadata is not the current source form")
    stablehlo = (source / "functions" / "forward.mlir").read_text()
    required = (
        f"tensor<{M}x{K}xf16>",
        f"tensor<{K}x{N}xf16>",
        f"tensor<{M}x{N}xf16>",
        "stablehlo.dot_general",
    )
    if any(fragment not in stablehlo for fragment in required):
        raise RuntimeError("K-tiled GEMM source semantics are incomplete")
    if "mhlo.sharding" in stablehlo or "@Sharding" in stablehlo:
        raise RuntimeError("K-tiled GEMM source contains retired partition hints")


def validate_manifest(
    package: pathlib.Path,
) -> tuple[dict[tuple[str, int], int], int]:
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="full-4096 GEMM",
    )
    if (
        manifest.get("tile_count") != TILE_COUNT
        or manifest.get("target", {}).get("identity") != TARGET_IDENTITY
    ):
        raise RuntimeError("full-4096 GEMM package target fields are invalid")
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
        raise RuntimeError("full-4096 GEMM must produce one cluster shared ELF")
    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="full-4096 GEMM"
    )
    module_id = modules[0].get("id")
    if any(
        entry.get("module") != module_id
        or entry.get("transport", {}).get("kind") != "direct_dte"
        or entry["transport"].get("status_abi") != STATUS_ABI
        or entry["transport"].get("host_watchdog_required") is not True
        for entry in entries
    ):
        raise RuntimeError("full-4096 GEMM Direct-DTE contract is invalid")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise RuntimeError("full-4096 GEMM port tables are not lists")
    expected_resources = {
        ("user_input", 0): ({"dtype": "f16", "shape": [M, K]}, LHS_BYTES),
        ("user_input", 1): ({"dtype": "f16", "shape": [K, N]}, RHS_BYTES),
        ("output", 0): ({"dtype": "f16", "shape": OUTPUT_SHAPE}, OUTPUT_BYTES),
    }
    bindings: dict[tuple[str, int], int] = {}
    for table, role in (("inputs", "user_input"), ("outputs", "output")):
        for record in manifest.get(table, []):
            if not isinstance(record, dict):
                raise RuntimeError("full-4096 GEMM port records must be objects")
            role_index = record.get("role_index")
            key = (role, role_index)
            if key not in expected_resources or key in bindings:
                raise RuntimeError("full-4096 GEMM has an unexpected host port")
            type_, bytes_ = expected_resources[key]
            if (
                record.get("dtype") != type_["dtype"]
                or record.get("shape") != type_["shape"]
                or record.get("bytes") != bytes_
                or not isinstance(record.get("id"), int)
            ):
                raise RuntimeError(f"full-4096 GEMM port {key} is invalid")
            bindings[key] = record["id"]
    if set(bindings) != set(expected_resources):
        raise RuntimeError("full-4096 GEMM host bindings are not exact")
    return bindings, bindings[("output", 0)]


def write_payloads(
    work_dir: pathlib.Path,
    bindings: dict[tuple[str, int], int],
) -> tuple[list[str], str, dict[pathlib.Path, torch.Tensor]]:
    raw = work_dir / "raw"
    raw.mkdir()

    generator = torch.Generator(device="cpu").manual_seed(PYTORCH_SEED)
    lhs_global = torch.randn(
        (M, K), dtype=torch.float16, generator=generator
    )
    rhs_global = torch.randn(
        (K, N), dtype=torch.float16, generator=generator
    )
    with torch.no_grad():
        expected = torch.matmul(lhs_global, rhs_global)

    lhs_path = raw / "lhs.f16.raw"
    rhs_path = raw / "rhs.f16.raw"
    capture_path = raw / "output.capture.f16.raw"
    torch_reference.write_tensor_raw(lhs_path, lhs_global)
    torch_reference.write_tensor_raw(rhs_path, rhs_global)
    arguments = [
        "--resource",
        f"{bindings[('user_input', 0)]}={lhs_path}",
        "--resource",
        f"{bindings[('user_input', 1)]}={rhs_path}",
        "--output",
        f"{bindings[('output', 0)]}={capture_path}",
    ]
    captures = {capture_path: expected}

    digest = hashlib.sha256(
        torch_reference.tensor_raw_bytes(expected)
    ).hexdigest()
    print(f"expected_sha256: {digest}")
    return arguments, digest, captures


def verify_no_card_evidence(stdout: str) -> None:
    required = {
        "package: id=0 cards=1 tiles=16",
        "invocation_tiles: 16",
        "board_execution: false",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("no-card output omitted full-4096 Direct-DTE evidence")


def verify_board_evidence(stdout: str, output_id: int) -> None:
    required = {
        "board_stage: validation",
        "board_stage: device-selection",
        "board_stage: resource-allocation",
        "board_stage: host-to-device",
        "board_stage: module-load",
        "board_stage: entry-resolve",
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "invocation_tiles: 16",
        "launch_pattern: cluster-x16",
        "physical_tile_domain: 0..15",
        "board_execution: true",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("board output omitted full-4096 lifecycle evidence")
    output_matches = re.findall(
        rf"^output_capture: port=(\d+) bytes={OUTPUT_BYTES} path=.+$",
        stdout,
        re.MULTILINE,
    )
    if (
        len(output_matches) != 1
        or int(output_matches[0]) != output_id
    ):
        raise RuntimeError("board output omitted the complete GEMM capture")
    tile_matches = re.findall(
        r"^board_tile: tile_id=(\d+) launch_slot=\d+ available=true "
        r"physical_x=\d+ physical_y=\d+$",
        stdout,
        re.MULTILINE,
    )
    if sorted(int(tile_id) for tile_id in tile_matches) != list(range(TILE_COUNT)):
        raise RuntimeError("board inventory did not prove Tiles 0..15")
    runtime_launch.require_board_completion(stdout, context="full-4096 GEMM")


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print(
            "full-4096 K-tiled GEMM hardware execution is not armed", file=sys.stderr
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
        if args.expected_tile_count != TILE_COUNT:
            raise RuntimeError("full-4096 K-tiled GEMM requires exactly 16 tiles")

    source = write_source_program(args.work_dir)
    validate_source_program(source)
    package = args.work_dir / "package"
    compile_result = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-package-dir",
            str(package),
            "--num-partitions=1",
        ]
    )
    if (
        "wrote verified package with num-partitions=1 tiles=16"
        not in compile_result.stdout
    ):
        raise RuntimeError("wafer-compile did not write a verified 16-Tile package")
    bindings, output_id = validate_manifest(package)

    if args.no_card:
        result = run(
            [
                str(args.wafer_run),
                "--package-dir",
                str(package),
                "--no-card",
                "--direct-dte-status-abi",
                STATUS_ABI,
                "--supports-host-watchdog",
            ]
        )
        verify_no_card_evidence(result.stdout)
        print("k_tiled_gemm_no_card: verified")
        print(result.stdout, end="")
        return 0

    resource_arguments, expected_digest, captures = write_payloads(
        args.work_dir, bindings
    )
    command = [
        str(args.wafer_run),
        "--package-dir",
        str(package),
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
        verify_board_evidence(result.stdout, output_id)
        for capture_path, expected in captures.items():
            torch_reference.assert_raw_capture_matches(
                capture_path,
                expected,
                policy=torch_reference.EXACT,
                context=(
                    f"full-4096 K-tiled GEMM iteration {iteration + 1} "
                    f"{capture_path.name}"
                ),
            )
        print(
            "k_tiled_gemm_iteration: "
            f"{iteration + 1}/{args.repeat} torch_close=true "
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
        print(f"wafer_board_k_tiled_gemm_test: {error}", file=sys.stderr)
        raise SystemExit(1)
