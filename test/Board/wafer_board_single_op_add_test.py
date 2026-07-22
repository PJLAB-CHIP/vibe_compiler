#!/usr/bin/env python3
"""Compile and execute a rank-one f32 add on a configured TX board."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np


MODULE = """\
module {
  func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
"""

METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [16], "dtype": "float32", "dynamic_dims": []},
        {"shape": [16], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [16], "dtype": "float32", "dynamic_dims": []}
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

    input_tensor = np.arange(-8, 8, dtype=np.float32)
    weight = np.arange(16, dtype=np.float32) * np.float32(2.0)
    expected = input_tensor + weight
    with (source / "data" / "weight").open("wb") as output:
        np.save(output, weight)

    raw = work_dir / "raw"
    raw.mkdir()
    paths = {
        ("user_input", 0): raw / "input.f32.raw",
        ("parameter", 1): raw / "weight.f32.raw",
        ("output", 0): raw / "expected.f32.raw",
    }
    input_tensor.tofile(paths[("user_input", 0)])
    weight.tofile(paths[("parameter", 1)])
    expected.tofile(paths[("output", 0)])
    return source, paths


def invocation_arguments(
    manifest_path: pathlib.Path, raw_paths: dict[tuple[str, int], pathlib.Path]
) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3 or manifest.get("rank_count") != 1:
        raise RuntimeError("single-op board gate requires a schema-v3 rank-one package")
    entries = manifest.get("entries", [])
    if len(entries) != 1 or entries[0].get("id") != 0:
        raise RuntimeError("single-op board gate requires the unique entry ID 0")

    arguments: list[str] = []
    seen: set[tuple[str, int]] = set()
    for resource in manifest.get("resources", []):
        if resource.get("rank") != 0 or not resource.get("host_visible"):
            continue
        role = resource.get("role")
        role_index = resource.get("role_index")
        key = (role, role_index)
        if key in seen or key not in raw_paths:
            raise RuntimeError(f"unexpected host-visible resource binding: {key}")
        seen.add(key)
        resource_id = resource["id"]
        if resource.get("access") == "read_only":
            arguments.extend(["--resource", f"{resource_id}={raw_paths[key]}"])
        elif resource.get("access") == "write_only" and role == "output":
            arguments.extend(["--expected", f"{resource_id}={raw_paths[key]}"])
        else:
            raise RuntimeError(f"unexpected host-visible resource access for {key}")
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
            "--execution-ranks=1",
            "--target-profile=wafer-tx81-single-card-kernel-v1",
        ]
    )
    if "published verified package" not in compile_result.stdout:
        raise RuntimeError("wafer-compile did not report a verified package")

    resource_arguments = invocation_arguments(package / "manifest.json", raw_paths)
    for iteration in range(args.repeat):
        result = run(
            [
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
                *resource_arguments,
            ]
        )
        required = (
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
            "output_compare:",
            "exact=true",
            "board_execution: true",
        )
        missing = [text for text in required if text not in result.stdout]
        if missing:
            raise RuntimeError(f"board invocation omitted evidence: {missing}")
        print(f"board_add_iteration: {iteration + 1}/{args.repeat}")
        print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"wafer_board_single_op_add_test: {error}", file=sys.stderr)
        raise SystemExit(1)
