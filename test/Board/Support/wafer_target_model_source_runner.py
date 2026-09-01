#!/usr/bin/env python3
"""Compile a current StableHLO source and execute its TargetCall in SystemC."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

import wafer_board_source_program as source_program
import wafer_runtime_launch_contract as runtime_launch


SHAPE = (2, 1024, 64)
ELEMENT_COUNT = int(np.prod(SHAPE))
DTYPE = np.dtype("<f2")
SOURCE_MODULE = f"""\
module {{
  func.func @main(
      %lhs: tensor<{SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}xf16>,
      %rhs: tensor<{SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}xf16>)
      -> tensor<{SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}xf16> {{
    %sum = stablehlo.add %lhs, %rhs
        : tensor<{SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}xf16>
    return %sum : tensor<{SHAPE[0]}x{SHAPE[1]}x{SHAPE[2]}xf16>
  }}
}}
"""
SOURCE_METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": list(SHAPE), "dtype": "float16", "dynamic_dims": []},
        {"shape": list(SHAPE), "dtype": "float16", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": list(SHAPE), "dtype": "float16", "dynamic_dims": []}
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
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def run(command: list[str], timeout_seconds: float = 900.0) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "target-model source vertical exceeded its one-shot deadline"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"target-model command failed with exit code {result.returncode}"
        )
    return result.stdout + result.stderr


def write_case(
    work_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    source_program.write_program(source, SOURCE_MODULE, SOURCE_METADATA)

    values = np.arange(ELEMENT_COUNT, dtype=np.int64).reshape(SHAPE)
    lhs = ((values % 17) - 8).astype(DTYPE)
    rhs = ((values % 7) - 3).astype(DTYPE)
    expected = (lhs + rhs).astype(DTYPE)
    input_lhs = work_dir / "lhs.npy"
    input_rhs = work_dir / "rhs.npy"
    output = work_dir / "expected.npy"
    np.save(input_lhs, lhs, allow_pickle=False)
    np.save(input_rhs, rhs, allow_pickle=False)
    np.save(output, expected, allow_pickle=False)
    return source, input_lhs, input_rhs, output


def main() -> int:
    args = parse_args()
    source, input_lhs, input_rhs, expected = write_case(args.work_dir)
    package = args.work_dir / "package"
    output = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-dir",
            str(package),
            "--num-partitions=1",
            "--optimization-policy=none",
            "--target-model",
            "--model-input",
            f"0={input_lhs}",
            "--model-input",
            f"1={input_rhs}",
            "--model-expected",
            f"0={expected}",
            "--model-atol=0",
            "--model-rtol=0",
            "--target-model-max-scalar-evaluations=1000000",
            "--target-model-max-fused-multiply-adds=1000000",
            "--target-model-max-movement-bytes=100000000",
            "--target-model-max-movement-segments=1000000",
        ]
    )
    if "target model outputs matched" not in output:
        raise RuntimeError("TargetCall model did not report matched outputs")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="target-model source vertical"
    )
    if len(entries) != 16:
        raise RuntimeError("target-model package does not contain 16 Tile entries")
    if not (source / "functions" / "forward.stablehlo.bc").is_file():
        raise RuntimeError("target-model source omitted StableHLO bytecode")
    if (source / "functions" / "forward.mlir").exists():
        raise RuntimeError("target-model source retained staging MLIR text")
    print(
        "target-model source vertical passed: shape="
        f"{list(SHAPE)} tiles={len(entries)} package={package}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
