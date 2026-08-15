#!/usr/bin/env python3
"""Validate the current source contract for Direct-DTE/compute overlap.

The source and host oracle remain useful.  Executable matched comparison waits
for the current global candidate selector to expose serialized and overlap
schedules through the normal ``none`` and ``search`` policies; retired test-only
selection environment variables are intentionally not supported.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch

import wafer_board_compiler_optimization_comparison_test as comparison


ELEMENT_COUNT = 524288
STRUCTURAL_CHECKS = (
    "same-current-global-source",
    "direct-dte-issue-and-wait-closure",
    "compute-between-issue-and-wait",
    "identical-global-output",
)


def overlap_module() -> str:
    return f"""module {{
  func.func @main(%lhs: tensor<{ELEMENT_COUNT}xf16>,
                  %rhs: tensor<{ELEMENT_COUNT}xf16>)
      -> tensor<{ELEMENT_COUNT}xf16> {{
    %add = stablehlo.add %lhs, %lhs : tensor<{ELEMENT_COUNT}xf16>
    %mul = stablehlo.multiply %rhs, %rhs : tensor<{ELEMENT_COUNT}xf16>
    %result = stablehlo.add %add, %mul : tensor<{ELEMENT_COUNT}xf16>
    return %result : tensor<{ELEMENT_COUNT}xf16>
  }}
}}
"""


def overlap_payloads() -> comparison.SourcePayloads:
    lhs = comparison.random_f16((ELEMENT_COUNT,), 400)
    rhs = comparison.random_f16((ELEMENT_COUNT,), 401)
    return comparison.SourcePayloads((lhs, rhs), (((lhs + lhs) + (rhs * rhs)),))


CASE = comparison.OptimizationComparisonCase(
    "direct-dte-compute-overlap",
    "transport-compute-overlap",
    (
        comparison.TensorSpec((ELEMENT_COUNT,), "f16", "float16"),
        comparison.TensorSpec((ELEMENT_COUNT,), "f16", "float16"),
    ),
    (comparison.TensorSpec((ELEMENT_COUNT,), "f16", "float16"),),
    overlap_module,
    overlap_payloads,
    STRUCTURAL_CHECKS,
)


def validate_source_contract() -> None:
    comparison.validate_source(CASE)
    module = CASE.module_factory()
    if any(
        marker in module
        for marker in ("mhlo.sharding", "@Sharding", "distributed_boundary")
    ):
        raise RuntimeError("overlap source contains a retired partition hint")
    for operation in (
        "stablehlo.add %lhs, %lhs",
        "stablehlo.multiply %rhs, %rhs",
        "stablehlo.add %add, %mul",
    ):
        if operation not in module:
            raise RuntimeError("overlap source dataflow is incomplete")
    comparison.validate_payloads(CASE, CASE.payload_factory())


def write_source(work_dir: pathlib.Path) -> pathlib.Path:
    source = work_dir / "source-program"
    if source.exists():
        shutil.rmtree(source)
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(CASE.module_factory())
    (source / "functions" / "forward.meta").write_text(
        json.dumps(comparison.metadata(CASE), separators=(",", ":")) + "\n"
    )
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_source_contract()
    write_source(args.work_dir)
    print(
        "direct_dte_compute_overlap_source_contract: "
        "policies=none,search executable=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
