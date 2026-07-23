#!/usr/bin/env python3
"""Compile and execute one standalone f16 GEMM on a configured TX board."""

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


GEMM_CASE_SHAPES = {
    "single-tile": (256, 256, 512),
    "mn-tiled": (4096, 256, 4096),
}
M, K, N = GEMM_CASE_SHAPES["single-tile"]
F16_BYTES = np.dtype("<f2").itemsize
TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
LAUNCH_ABI = "per-rank-pointer-block-v1"
PROCESS_TIMEOUT_MARGIN_SECONDS = 30


def configure_gemm_case(name: str) -> None:
    global M, K, N
    M, K, N = GEMM_CASE_SHAPES[name]


def module_text() -> str:
    return f"""\
module {{
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


def metadata() -> dict[str, object]:
    return {
        "name": "forward",
        "stablehlo_version": "0.0.0",
        "input_signature": [
            {"shape": [M, K], "dtype": "float16", "dynamic_dims": []},
            {"shape": [K, N], "dtype": "float16", "dynamic_dims": []},
        ],
        "output_signature": [
            {"shape": [M, N], "dtype": "float16", "dynamic_dims": []}
        ],
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
    parser.add_argument(
        "--gemm-case", choices=tuple(GEMM_CASE_SHAPES), default="single-tile"
    )
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def run(
    command: list[str], timeout_seconds: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        for partial in (error.stdout, error.stderr):
            if partial:
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(partial, end="", file=sys.stderr)
        raise RuntimeError(
            "one-shot standalone GEMM exceeded its outer deadline; the process "
            "was killed and this test will not retry or invoke reset/power "
            "operations; board state requires read-only qualification"
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
    (source / "functions" / "forward.mlir").write_text(module_text())
    (source / "functions" / "forward.meta").write_text(
        json.dumps(metadata(), separators=(",", ":")) + "\n"
    )
    return source


def validate_structured_program(package: pathlib.Path) -> None:
    structured_ir = (package / "functions" / "forward.mlir").read_text()
    required = (
        f"tensor<{M}x{K}xf16>",
        f"tensor<{K}x{N}xf16>",
        f"tensor<{M}x{N}xf16>",
        "linalg.matmul",
    )
    missing = [fragment for fragment in required if fragment not in structured_ir]
    if missing:
        raise RuntimeError(f"standalone GEMM structured IR omitted: {missing}")
    if (
        structured_ir.count("linalg.matmul") != 1
        or "stablehlo." in structured_ir
        or "collective" in structured_ir
    ):
        raise RuntimeError(
            "standalone program is not exactly one normalized local GEMM"
        )


def validate_manifest(package: pathlib.Path) -> tuple[dict[tuple[str, int], int], int]:
    manifest = json.loads((package / "manifest.json").read_text())
    target = manifest.get("target")
    if (
        manifest.get("schema_version") != 5
        or manifest.get("rank_count") != 1
        or not isinstance(target, dict)
        or target.get("profile") != TARGET_PROFILE
        or target.get("launch_abi") != LAUNCH_ABI
    ):
        raise RuntimeError("standalone GEMM package target contract is invalid")

    modules = manifest.get("modules")
    entries = manifest.get("entries")
    completions = manifest.get("completions")
    resources = manifest.get("resources")
    if (
        not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(completions, list)
        or len(completions) != 1
        or not isinstance(resources, list)
        or len(resources) != 3
    ):
        raise RuntimeError("standalone GEMM package domains are not rank-one exact")
    module = modules[0]
    entry = entries[0]
    completion = completions[0]
    if (
        module.get("exports") != [{"role": "main", "symbol": "main"}]
        or not (package / module.get("path", "")).is_file()
        or entry.get("id") != 0
        or entry.get("rank") != 0
        or entry.get("module") != module.get("id")
        or entry.get("terminal_completion") != completion.get("id")
        or entry.get("transport") != {"kind": "none"}
        or completion.get("rank") != 0
        or completion.get("kind") != "entry_return"
    ):
        raise RuntimeError("standalone GEMM entry/module/completion contract is invalid")

    expected = {
        ("user_input", 0): (
            {"dtype": "f16", "shape": [M, K]},
            M * K * F16_BYTES,
            "read_only",
        ),
        ("user_input", 1): (
            {"dtype": "f16", "shape": [K, N]},
            K * N * F16_BYTES,
            "read_only",
        ),
        ("output", 0): (
            {"dtype": "f16", "shape": [M, N]},
            M * N * F16_BYTES,
            "write_only",
        ),
    }
    bindings: dict[tuple[str, int], int] = {}
    for resource in resources:
        if not isinstance(resource, dict):
            raise RuntimeError("standalone GEMM resources must be objects")
        key = (resource.get("role"), resource.get("role_index"))
        if key not in expected or key in bindings:
            raise RuntimeError(f"unexpected standalone GEMM resource: {key}")
        type_, bytes_, access = expected[key]
        if (
            resource.get("rank") != 0
            or resource.get("type") != type_
            or resource.get("bytes") != bytes_
            or resource.get("alignment") != 256
            or resource.get("access") != access
            or resource.get("host_visible") is not True
            or not isinstance(resource.get("id"), int)
        ):
            raise RuntimeError(f"invalid standalone GEMM resource: {key}")
        bindings[key] = resource["id"]
    if set(bindings) != set(expected):
        raise RuntimeError("standalone GEMM host bindings are incomplete")

    expected_slots = [
        (0, bindings[("user_input", 0)], "read_only"),
        (1, bindings[("user_input", 1)], "read_only"),
        (2, bindings[("output", 0)], "write_only"),
    ]
    actual_slots = [
        (slot.get("ordinal"), slot.get("resource"), slot.get("access"))
        for slot in entry.get("slots", [])
        if isinstance(slot, dict)
    ]
    if actual_slots != expected_slots:
        raise RuntimeError("standalone GEMM launch slots are invalid")
    return bindings, bindings[("output", 0)]


def write_payloads(
    work_dir: pathlib.Path, bindings: dict[tuple[str, int], int]
) -> list[str]:
    raw = work_dir / "raw"
    raw.mkdir()
    rows = np.arange(M, dtype=np.int32)
    lhs = np.zeros((M, K), dtype="<f2")
    lhs[rows, rows % K] = np.float16(1.0)
    k_indices = np.arange(K, dtype=np.int32)[:, None]
    n_indices = np.arange(N, dtype=np.int32)[None, :]
    rhs = (1 + ((k_indices * 17 + n_indices * 3) % 1024) / 8).astype("<f2")
    expected = rhs[rows % K, :].copy()
    if (
        np.count_nonzero(lhs) != M
        or not np.all(lhs[rows, rows % K] == np.float16(1.0))
        or expected.shape != (M, N)
    ):
        raise RuntimeError("standalone GEMM one-hot CPU oracle is invalid")

    paths = {
        ("user_input", 0): raw / "lhs_identity.f16.raw",
        ("user_input", 1): raw / "rhs_pattern.f16.raw",
        ("output", 0): raw / "expected.f16.raw",
    }
    lhs.tofile(paths[("user_input", 0)])
    rhs.tofile(paths[("user_input", 1)])
    expected.tofile(paths[("output", 0)])
    arguments: list[str] = []
    for key, path in paths.items():
        option = "--expected" if key[0] == "output" else "--resource"
        arguments.extend([option, f"{bindings[key]}={path}"])
    return arguments


def verify_no_card_evidence(stdout: str) -> None:
    required = {
        "package: id=0 schema=5 ranks=1",
        "target: wafer-tx81-single-card runtime_abi=wafer-tx81-kernel-v1 "
        f"launch_abi={LAUNCH_ABI} module_format=elf-riscv64",
        "entry: 0 rank=0 symbol=main",
        "board_execution: false",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("no-card output omitted standalone GEMM evidence")


def verify_board_evidence(stdout: str, output_id: int) -> None:
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
        "board_execution: true",
    }
    missing = sorted(required.difference(stdout.splitlines()))
    if missing:
        raise RuntimeError(
            f"board output omitted standalone GEMM lifecycle evidence: {missing}"
        )
    if not re.search(
        rf"^output_compare: resource={output_id} bytes={M * N * F16_BYTES} exact=true$",
        stdout,
        re.MULTILINE,
    ):
        raise RuntimeError("board output omitted standalone GEMM exact comparison")


def main() -> int:
    args = parse_args()
    configure_gemm_case(args.gemm_case)
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("standalone GEMM hardware execution is not armed", file=sys.stderr)
        return 77
    if not args.no_card and any(
        value is None
        for value in (
            args.expected_runtime_version,
            args.expected_device_name,
            args.expected_pci_bus_id,
            args.expected_tile_count,
            args.expected_runtime_library_sha256,
        )
    ):
        raise RuntimeError("board execution requires complete qualification arguments")

    source = write_source_program(args.work_dir)
    package = args.work_dir / "package"
    compile_result = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(package),
            "--execution-ranks=1",
            f"--target-profile={TARGET_PROFILE}",
            f"--launch-abi={LAUNCH_ABI}",
        ]
    )
    if "published verified package with execution-ranks=1" not in compile_result.stdout:
        raise RuntimeError("wafer-compile did not publish a verified rank-one package")
    validate_structured_program(package)
    bindings, output_id = validate_manifest(package)

    if args.no_card:
        result = run(
            [
                str(args.wafer_run),
                "--package-dir",
                str(package),
                "--entry-id",
                "0",
                "--no-card",
            ]
        )
        verify_no_card_evidence(result.stdout)
        print("standalone_gemm_no_card: verified")
        print(result.stdout, end="")
        return 0

    resource_arguments = write_payloads(args.work_dir, bindings)
    command = [
        str(args.wafer_run),
        "--package-dir",
        str(package),
        "--entry-id",
        "0",
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
                + PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        verify_board_evidence(result.stdout, output_id)
        print(f"standalone_gemm_iteration: {iteration + 1}/{args.repeat} exact=true")
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
        print(f"wafer_board_single_gemm_test: {error}", file=sys.stderr)
        raise SystemExit(1)
