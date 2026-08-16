#!/usr/bin/env python3
"""Compile and execute one f16 add over the current TX Tile domain."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

import wafer_runtime_launch_contract as runtime_launch


@dataclasses.dataclass(frozen=True)
class RuntimeLaunchCalibrationCase:
    key: str
    tile_count: int
    oracle: str
    completion: str


RUNTIME_LAUNCH_CALIBRATION_CASES = (
    RuntimeLaunchCalibrationCase(
        "grid-add",
        runtime_launch.TARGET_TILE_COUNT,
        "full f16 expected output and exact current resource binding",
        "local drain, device-to-host copy, and normal cleanup",
    ),
)
CALIBRATION_LEAF_BINDINGS = {
    "grid-add": RUNTIME_LAUNCH_CALIBRATION_CASES,
}


MODULE = """\
module {
  func.func @main(%arg0: tensor<16xf16>, %arg1: tensor<16xf16>) -> tensor<16xf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<16xf16>
    return %0 : tensor<16xf16>
  }
}
"""

METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [16], "dtype": "float16", "dynamic_dims": []},
        {"shape": [16], "dtype": "float16", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [16], "dtype": "float16", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "input"},
        {"type_": "parameter", "position": -1, "name": "weight"},
    ],
    "unused_inputs": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int, required=True)
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument("--expected-pci-bus-id", required=True)
    parser.add_argument("--expected-tile-count", type=int, required=True)
    parser.add_argument("--expected-runtime-library-sha256", required=True)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")
    return result


def write_fixture(
    work_dir: pathlib.Path,
) -> tuple[pathlib.Path, dict[tuple[str, int], pathlib.Path]]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(MODULE)
    (source / "functions" / "forward.meta").write_text(
        json.dumps(METADATA, separators=(",", ":")) + "\n"
    )

    # Small integers and their sums are exactly representable in binary16, so
    # this heartbeat keeps exact comparison without conflating launch health
    # with floating-point tolerance.
    input_tensor = np.arange(-8, 8, dtype=np.float16)
    weight = np.arange(16, dtype=np.float16) * np.float16(2.0)
    expected = input_tensor + weight
    with (source / "data" / "weight").open("wb") as output:
        np.save(output, weight)

    raw = work_dir / "raw"
    raw.mkdir()
    paths = {
        ("user_input", 0): raw / "input.f16.raw",
        ("output", 0): raw / "expected.f16.raw",
    }
    input_tensor.tofile(paths[("user_input", 0)])
    expected.tofile(paths[("output", 0)])
    return source, paths


def invocation_arguments(
    manifest_path: pathlib.Path, raw_paths: dict[tuple[str, int], pathlib.Path]
) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.GRID_KERNEL_LAUNCH,
        context="single-op board gate",
    )
    runtime_launch.require_complete_tile_domain(
        manifest, context="single-op board gate"
    )

    arguments: list[str] = []
    seen: set[tuple[str, int]] = set()
    for table, role in (("inputs", "user_input"), ("outputs", "output")):
        for record in manifest.get(table, []):
            if not isinstance(record, dict):
                raise RuntimeError("manifest port record must be an object")
            role_index = record.get("role_index")
            if not isinstance(role_index, int):
                raise RuntimeError("manifest port record requires role_index")
            key = (role, role_index)
            if key in seen or key not in raw_paths:
                raise RuntimeError(f"unexpected manifest port binding: {key}")
            seen.add(key)
            port_id = record.get("id")
            if not isinstance(port_id, int):
                raise RuntimeError("manifest port record requires an integer id")
            if role == "user_input":
                arguments.extend(["--resource", f"{port_id}={raw_paths[key]}"])
            elif role == "output":
                arguments.extend(["--expected", f"{port_id}={raw_paths[key]}"])
            else:
                raise RuntimeError(f"unexpected manifest port role for {key}")
    if seen != set(raw_paths):
        raise RuntimeError(f"package does not expose the expected add bindings: {seen}")
    return arguments


def main() -> int:
    if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print(
            "wafer_board_single_op_add_test: hardware execution is not armed; "
            "set WAFER_EXECUTE_HARDWARE_TESTS=1",
            file=sys.stderr,
        )
        return 77
    args = parse_args()
    if args.repeat < 1:
        raise RuntimeError("--repeat must be positive")
    source, raw_paths = write_fixture(args.work_dir)
    package = args.work_dir / "package"
    compile_result = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(package),
            "--num-partitions=1",
        ]
    )
    if "wrote verified package" not in compile_result.stdout:
        raise RuntimeError("wafer-compile did not report a verified package")

    resource_arguments = invocation_arguments(package / "manifest.json", raw_paths)
    for iteration in range(args.repeat):
        result = run(
            [
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
                *resource_arguments,
            ]
        )
        required = (
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
            "output_compare:",
            "exact=true",
            "board_execution: true",
        )
        missing = [text for text in required if text not in result.stdout]
        if missing:
            raise RuntimeError(f"board invocation omitted evidence: {missing}")
        runtime_launch.require_board_completion(
            result.stdout, context="single-op board gate"
        )
        print(f"board_add_iteration: {iteration + 1}/{args.repeat}")
        print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"wafer_board_single_op_add_test: {error}", file=sys.stderr)
        raise SystemExit(1)
