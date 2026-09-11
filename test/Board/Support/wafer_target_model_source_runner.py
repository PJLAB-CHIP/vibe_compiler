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
DTYPE = np.dtype("<f2")
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
    parser.add_argument("--case", choices=("add", "score-rounding"), default="add")
    parser.add_argument("--extent", type=int, choices=(1024, 1025, 1031), default=1024)
    parser.add_argument("--optimization-policy", choices=("none", "search"), default="none")
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
    work_dir: pathlib.Path, case: str, extent: int,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    shape = (2, extent, 64)
    spelling = "x".join(map(str, shape))
    input_type = f"tensor<{spelling}xf16>"
    wide_type = f"tensor<{spelling}xf32>"
    values = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    lhs = ((values % 37) - 18).astype(DTYPE) / np.float16(7)
    rhs = ((values % 17) - 8).astype(DTYPE) / np.float16(19)
    if case == "score-rounding":
        output_type = wide_type
        expected = ((lhs.astype(np.float32) * np.float32(0.0883883461))
                    .astype(DTYPE) + rhs).astype(np.float32)
        unrounded = lhs.astype(np.float32) * np.float32(0.0883883461) + rhs.astype(np.float32)
        if not np.any(expected != unrounded):
            raise RuntimeError("rounding corpus does not distinguish lost casts")
        body = f"""
    %wide = stablehlo.convert %lhs : ({input_type}) -> {wide_type}
    %scale = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %scales = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> {wide_type}
    %scaled = stablehlo.multiply %wide, %scales : {wide_type}
    %rounded = stablehlo.convert %scaled : ({wide_type}) -> {input_type}
    %masked = stablehlo.add %rounded, %rhs : {input_type}
    %result = stablehlo.convert %masked : ({input_type}) -> {wide_type}
"""
    else:
        output_type = input_type
        lhs = ((values % 17) - 8).astype(DTYPE)
        rhs = ((values % 7) - 3).astype(DTYPE)
        expected = (lhs + rhs).astype(DTYPE)
        body = f"    %result = stablehlo.add %lhs, %rhs : {input_type}\n"
    module = f"""module {{
  func.func @main(%lhs: {input_type}, %rhs: {input_type}) -> {output_type} {{
{body}
    return %result : {output_type}
  }}
}}
"""
    metadata = dict(SOURCE_METADATA)
    metadata["input_signature"] = [
        {"shape": list(shape), "dtype": "float16", "dynamic_dims": []} for _ in range(2)
    ]
    metadata["output_signature"] = [{
        "shape": list(shape), "dtype": "float32" if case == "score-rounding" else "float16",
        "dynamic_dims": [],
    }]
    source_program.write_program(source, module, metadata)
    input_lhs = work_dir / "lhs.npy"
    input_rhs = work_dir / "rhs.npy"
    output = work_dir / "expected.npy"
    np.save(input_lhs, lhs, allow_pickle=False)
    np.save(input_rhs, rhs, allow_pickle=False)
    np.save(output, expected, allow_pickle=False)
    return source, input_lhs, input_rhs, output


def main() -> int:
    args = parse_args()
    source, input_lhs, input_rhs, expected = write_case(args.work_dir, args.case, args.extent)
    package = args.work_dir / "package"
    output = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-dir",
            str(package),
            "--num-partitions=1",
            f"--optimization-policy={args.optimization_policy}",
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
        f"{[2, args.extent, 64]} case={args.case} tiles={len(entries)} package={package}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
