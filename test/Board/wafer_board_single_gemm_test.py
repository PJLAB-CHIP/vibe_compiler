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

import torch

import wafer_runtime_launch_contract as runtime_launch

PYTORCH_BOARD_DIR = pathlib.Path(__file__).resolve().parent / "PyTorch"
if str(PYTORCH_BOARD_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_BOARD_DIR))

import wafer_pytorch_board_common as torch_reference  # noqa: E402


GEMM_CASE_SHAPES = {
    "single-tile": (256, 256, 512),
    "mn-tiled": (4096, 256, 4096),
}
M, K, N = GEMM_CASE_SHAPES["single-tile"]
F16_BYTES = 2
TARGET_IDENTITY = "wafer-tx81-single-card"
PROCESS_TIMEOUT_MARGIN_SECONDS = 30
PYTORCH_SEED = 20260803


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
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.GRID_KERNEL_LAUNCH,
        context="standalone GEMM",
    )
    if (
        not isinstance(target, dict)
        or target.get("identity") != TARGET_IDENTITY
    ):
        raise RuntimeError("standalone GEMM package target fields are invalid")

    modules = manifest.get("modules")
    entries = manifest.get("entries")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="standalone GEMM"
    )
    if (
        not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(inputs, list)
        or not isinstance(outputs, list)
    ):
        raise RuntimeError("standalone GEMM package domains are invalid")
    module = modules[0]
    entry = entries[0]
    if (
        module.get("exports") != [{"role": "main", "symbol": "main"}]
        or not (package / module.get("path", "")).is_file()
        or entry.get("id") != 0
        or entry.get("card_id") != 0
        or entry.get("tile_id") != 0
        or entry.get("module") != module.get("id")
        or entry.get("transport") != {"kind": "none"}
    ):
        raise RuntimeError("standalone GEMM entry/module contract is invalid")
    if any(item.get("module") != module.get("id") for item in entries):
        raise RuntimeError("standalone GEMM entries do not share one module")

    expected = {
        ("user_input", 0): (
            {"dtype": "f16", "shape": [M, K]},
            M * K * F16_BYTES,
        ),
        ("user_input", 1): (
            {"dtype": "f16", "shape": [K, N]},
            K * N * F16_BYTES,
        ),
        ("output", 0): (
            {"dtype": "f16", "shape": [M, N]},
            M * N * F16_BYTES,
        ),
    }
    bindings: dict[tuple[str, int], int] = {}
    for table, role in (("inputs", "user_input"), ("outputs", "output")):
        for record in manifest.get(table, []):
            if not isinstance(record, dict):
                raise RuntimeError("standalone GEMM ports must be objects")
            role_index = record.get("role_index")
            key = (role, role_index)
            if key not in expected or key in bindings:
                raise RuntimeError(f"unexpected standalone GEMM port: {key}")
            type_, bytes_ = expected[key]
            if (
                record.get("dtype") != type_["dtype"]
                or record.get("shape") != type_["shape"]
                or record.get("bytes") != bytes_
                or record.get("alignment") != 256
                or not isinstance(record.get("id"), int)
            ):
                raise RuntimeError(f"invalid standalone GEMM port: {key}")
            bindings[key] = record["id"]
    if set(bindings) != set(expected):
        raise RuntimeError("standalone GEMM host bindings are incomplete")

    expected_arguments = [
        {
            "ordinal": 0,
            "kind": "external_input",
            "port": bindings[("user_input", 0)],
            "access": "read_only",
        },
        {
            "ordinal": 1,
            "kind": "external_input",
            "port": bindings[("user_input", 1)],
            "access": "read_only",
        },
        {
            "ordinal": 2,
            "kind": "external_output",
            "port": bindings[("output", 0)],
            "access": "write_only",
        },
    ]
    actual_arguments = [
        {
            "ordinal": argument.get("ordinal"),
            "kind": argument.get("kind"),
            "port": argument.get("port"),
            "access": argument.get("access"),
        }
        for argument in entry.get("arguments", [])[:3]
        if isinstance(argument, dict)
    ]
    if actual_arguments != expected_arguments:
        raise RuntimeError("standalone GEMM launch arguments are invalid")
    return bindings, bindings[("output", 0)]


def write_payloads(
    work_dir: pathlib.Path, bindings: dict[tuple[str, int], int]
) -> tuple[list[str], pathlib.Path, torch.Tensor]:
    raw = work_dir / "raw"
    raw.mkdir()
    generator = torch.Generator(device="cpu").manual_seed(PYTORCH_SEED)
    lhs = torch.randn((M, K), dtype=torch.float16, generator=generator)
    rhs = torch.randn((K, N), dtype=torch.float16, generator=generator)
    with torch.no_grad():
        expected = torch.matmul(lhs, rhs)

    paths = {
        ("user_input", 0): raw / "lhs_random.f16.raw",
        ("user_input", 1): raw / "rhs_random.f16.raw",
    }
    torch_reference.write_tensor_raw(paths[("user_input", 0)], lhs)
    torch_reference.write_tensor_raw(paths[("user_input", 1)], rhs)
    capture_path = raw / "output.capture.f16.raw"
    arguments: list[str] = []
    for key in (("user_input", 0), ("user_input", 1)):
        arguments.extend(["--resource", f"{bindings[key]}={paths[key]}"])
    arguments.extend(
        ["--output", f"{bindings[('output', 0)]}={capture_path}"]
    )
    return arguments, capture_path, expected


def verify_no_card_evidence(stdout: str) -> None:
    required = {
        "package: id=0 cards=1 tiles=16",
        "entry: 0 card_id=0 tile_id=0 launch_slot=0",
        "launch_phase: role=main symbol=main",
        "board_execution: false",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("no-card output omitted standalone GEMM evidence")


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
        "board_execution: true",
    }
    missing = sorted(required.difference(stdout.splitlines()))
    if missing:
        raise RuntimeError(
            f"board output omitted standalone GEMM lifecycle evidence: {missing}"
        )
    if not re.search(
        rf"^output_capture: port={output_id} bytes={M * N * F16_BYTES} path=.+$",
        stdout,
        re.MULTILINE,
    ):
        raise RuntimeError("board output omitted standalone GEMM capture")
    runtime_launch.require_board_completion(stdout, context="standalone GEMM")


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
            "--output-package-dir",
            str(package),
            "--num-partitions=1",
        ]
    )
    if "wrote verified package with num-partitions=1 tiles=16" not in compile_result.stdout:
        raise RuntimeError("wafer-compile did not write a verified Tile package")
    validate_structured_program(package)
    bindings, output_id = validate_manifest(package)

    if args.no_card:
        result = run(
            [
                str(args.wafer_run),
                "--package-dir",
                str(package),
                "--no-card",
            ]
        )
        verify_no_card_evidence(result.stdout)
        print("standalone_gemm_no_card: verified")
        print(result.stdout, end="")
        return 0

    resource_arguments, capture_path, expected = write_payloads(
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
                + PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        verify_board_evidence(result.stdout, output_id)
        torch_reference.assert_raw_capture_matches(
            capture_path,
            expected,
            policy=torch_reference.EXACT,
            context=f"standalone GEMM iteration {iteration + 1}",
        )
        print(
            f"standalone_gemm_iteration: {iteration + 1}/{args.repeat} "
            "torch_close=true"
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
        print(f"wafer_board_single_gemm_test: {error}", file=sys.stderr)
        raise SystemExit(1)
