#!/usr/bin/env python3
"""Validate the current source contract for Direct-DTE/compute overlap."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch

import wafer_source_program_fixture as source_fixture


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


def overlap_payloads() -> source_fixture.SourcePayloads:
    lhs = source_fixture.random_f16((ELEMENT_COUNT,), 400)
    rhs = source_fixture.random_f16((ELEMENT_COUNT,), 401)
    return source_fixture.SourcePayloads(
        (lhs, rhs), (((lhs + lhs) + (rhs * rhs)),)
    )


CASE = source_fixture.SourceProgramCase(
    "direct-dte-compute-overlap",
    (
        source_fixture.TensorSpec((ELEMENT_COUNT,), "f16", "float16"),
        source_fixture.TensorSpec((ELEMENT_COUNT,), "f16", "float16"),
    ),
    (source_fixture.TensorSpec((ELEMENT_COUNT,), "f16", "float16"),),
    overlap_module,
    overlap_payloads,
)


def validate_source_contract() -> None:
    source_fixture.validate_source(CASE)
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
    source_fixture.validate_payloads(CASE, CASE.payload_factory())


def write_source(work_dir: pathlib.Path) -> pathlib.Path:
    source = work_dir / "source-program"
    if source.exists():
        shutil.rmtree(source)
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(CASE.module_factory())
    (source / "functions" / "forward.meta").write_text(
        json.dumps(source_fixture.metadata(CASE), separators=(",", ":"))
        + "\n"
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
        "executable=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
