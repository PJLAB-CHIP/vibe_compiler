#!/usr/bin/env python3
"""Validate current global-source programs used for optimizer comparisons.

The former board runner selected retired compiler implementations and decoded a
per-partition package.  The current compiler has one global source interface and
one Tile package interface.  Until current global search can execute
every case, this file owns only the reusable source, deterministic host oracle,
and the intended ``none`` versus ``search`` policy comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import shutil
from collections.abc import Callable

import torch


TILE_COUNT = 16
PYTORCH_RANDOM_SEED = 20260803
POLICIES = ("none", "search")


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    mlir_dtype: str
    metadata_dtype: str

    @property
    def element_bytes(self) -> int:
        return {"f16": 2, "bf16": 2, "f32": 4}[self.mlir_dtype]


@dataclasses.dataclass(frozen=True)
class SourcePayloads:
    inputs: tuple[torch.Tensor, ...]
    expected_outputs: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class OptimizationComparisonCase:
    key: str
    family: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    module_factory: Callable[[], str]
    payload_factory: Callable[[], SourcePayloads]
    structural_checks: tuple[str, ...]


F16_16384 = TensorSpec((16384,), "f16", "float16")
F16_32768 = TensorSpec((32768,), "f16", "float16")
F16_524288 = TensorSpec((524288,), "f16", "float16")
F16_8388608 = TensorSpec((8388608,), "f16", "float16")
F16_8192 = TensorSpec((8192,), "f16", "float16")
F16_64_128 = TensorSpec((64, 128), "f16", "float16")
F16_128_128 = TensorSpec((128, 128), "f16", "float16")
F16_64_128_OUT = TensorSpec((64, 128), "f16", "float16")
F16_65_129 = TensorSpec((65, 129), "f16", "float16")
F16_129_129 = TensorSpec((129, 129), "f16", "float16")
F16_65_129_OUT = TensorSpec((65, 129), "f16", "float16")
F16_TILE_REDUCTION_INPUT = TensorSpec((TILE_COUNT, 4096), "f16", "float16")
F16_TILE_REDUCTION_OUTPUT = TensorSpec((4096,), "f16", "float16")

LARGE_GEMM_EXTENT = 4096
M_TILED_GEMM_M = 4096
M_TILED_GEMM_K = 1024
M_TILED_GEMM_N = 4096
F16_LARGE_GEMM_LHS = TensorSpec(
    (LARGE_GEMM_EXTENT, LARGE_GEMM_EXTENT), "f16", "float16"
)
F16_LARGE_GEMM_RHS = F16_LARGE_GEMM_LHS
F16_LARGE_GEMM_OUTPUT = F16_LARGE_GEMM_LHS
F16_M_TILED_GEMM_LHS = TensorSpec(
    (M_TILED_GEMM_M, M_TILED_GEMM_K), "f16", "float16"
)
F16_M_TILED_GEMM_RHS = TensorSpec(
    (M_TILED_GEMM_K, M_TILED_GEMM_N), "f16", "float16"
)
F16_M_TILED_GEMM_OUTPUT = TensorSpec(
    (M_TILED_GEMM_M, M_TILED_GEMM_N), "f16", "float16"
)


def elementwise_module(
    arguments: str, results: str, body: str, return_values: str
) -> str:
    return_types = results[1:-1] if results.startswith("(") else results
    return f"""module {{
  func.func @main({arguments}) -> {results} {{
{body}
    return {return_values} : {return_types}
  }}
}}
"""


def reciprocal_module() -> str:
    return elementwise_module(
        "%input: tensor<8192xf16>",
        "tensor<8192xf16>",
        """    %one = stablehlo.constant dense<1.0> : tensor<8192xf16>
    %result = stablehlo.divide %one, %input : tensor<8192xf16>""",
        "%result",
    )


def f16_factor_module() -> str:
    return """module {
  func.func @main(%a: tensor<16384xf16>, %b: tensor<16384xf16>,
                  %c: tensor<16384xf16>) -> tensor<16384xf16> {
    %ab = stablehlo.multiply %a, %b : tensor<16384xf16>
    %ac = stablehlo.multiply %a, %c : tensor<16384xf16>
    %result = stablehlo.add %ab, %ac : tensor<16384xf16>
    return %result : tensor<16384xf16>
  }
}
"""


def resident_fanout_module() -> str:
    return elementwise_module(
        "%input: tensor<32768xf16>",
        "(tensor<32768xf16>, tensor<32768xf16>, tensor<32768xf16>)",
        """    %producer = stablehlo.multiply %input, %input : tensor<32768xf16>
    %left = stablehlo.add %producer, %producer : tensor<32768xf16>
    %right = stablehlo.multiply %producer, %producer : tensor<32768xf16>""",
        "%producer, %left, %right",
    )


def recompute_module() -> str:
    return elementwise_module(
        "%input: tensor<524288xf16>, %other: tensor<524288xf16>",
        "(tensor<524288xf16>, tensor<524288xf16>, tensor<524288xf16>)",
        """    %producer = stablehlo.multiply %input, %input : tensor<524288xf16>
    %left = stablehlo.add %producer, %producer : tensor<524288xf16>
    %middle = stablehlo.multiply %other, %other : tensor<524288xf16>
    %right = stablehlo.multiply %producer, %producer : tensor<524288xf16>""",
        "%left, %middle, %right",
    )


def long_steady_add_module() -> str:
    return elementwise_module(
        "%lhs: tensor<8388608xf16>, %rhs: tensor<8388608xf16>",
        "tensor<8388608xf16>",
        "    %result = stablehlo.add %lhs, %rhs : tensor<8388608xf16>",
        "%result",
    )


def ready_order_module() -> str:
    return elementwise_module(
        "%a: tensor<8192xf16>, %b: tensor<8192xf16>",
        "tensor<8192xf16>",
        """    %producer = stablehlo.multiply %a, %a : tensor<8192xf16>
    %left = stablehlo.add %producer, %b : tensor<8192xf16>
    %result = stablehlo.add %left, %b : tensor<8192xf16>""",
        "%result",
    )


def gemm_module(m: int, k: int, n: int) -> str:
    return f"""module {{
  func.func @main(%lhs: tensor<{m}x{k}xf16>,
                  %rhs: tensor<{k}x{n}xf16>) -> tensor<{m}x{n}xf16> {{
    %result = "stablehlo.dot_general"(%lhs, %rhs) {{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    }} : (tensor<{m}x{k}xf16>, tensor<{k}x{n}xf16>) -> tensor<{m}x{n}xf16>
    return %result : tensor<{m}x{n}xf16>
  }}
}}
"""


def tile_reduction_module() -> str:
    return f"""module {{
  func.func @main(%input: tensor<{TILE_COUNT}x4096xf16>) -> tensor<4096xf16> {{
    %zero = stablehlo.constant dense<0.0> : tensor<f16>
    %result = "stablehlo.reduce"(%input, %zero) ({{
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f16>
      stablehlo.return %sum : tensor<f16>
    }}) {{dimensions = array<i64: 0>}}
      : (tensor<{TILE_COUNT}x4096xf16>, tensor<f16>) -> tensor<4096xf16>
    return %result : tensor<4096xf16>
  }}
}}
"""


def random_f16(shape: tuple[int, ...], stream: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        PYTORCH_RANDOM_SEED + stream
    )
    return torch.randn(shape, dtype=torch.float16, generator=generator)


def random_unit_f16(shape: tuple[int, ...], stream: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        PYTORCH_RANDOM_SEED + stream
    )
    return torch.rand(shape, dtype=torch.float16, generator=generator)


def payloads(
    inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]
) -> SourcePayloads:
    return SourcePayloads(inputs, outputs)


def reciprocal_payloads() -> SourcePayloads:
    input_ = random_unit_f16((8192,), 0) + 0.5
    return payloads((input_,), (torch.ones_like(input_) / input_,))


def f16_factor_payloads() -> SourcePayloads:
    a = random_unit_f16((16384,), 10)
    b = random_unit_f16((16384,), 11)
    c = random_unit_f16((16384,), 12)
    return payloads((a, b, c), ((a * b) + (a * c),))


def resident_fanout_payloads() -> SourcePayloads:
    input_ = random_f16((32768,), 20)
    producer = input_ * input_
    return payloads((input_,), (producer, producer + producer, producer * producer))


def recompute_payloads() -> SourcePayloads:
    input_ = random_f16((524288,), 30)
    other = random_f16((524288,), 31)
    producer = input_ * input_
    return payloads(
        (input_, other),
        (producer + producer, other * other, producer * producer),
    )


def long_steady_add_payloads() -> SourcePayloads:
    lhs = random_f16((8388608,), 40)
    rhs = random_f16((8388608,), 41)
    return payloads((lhs, rhs), (lhs + rhs,))


def ready_order_payloads() -> SourcePayloads:
    a = random_f16((8192,), 50)
    b = random_f16((8192,), 51)
    return payloads((a, b), (((a * a) + b) + b,))


def gemm_payloads(m: int, k: int, n: int) -> SourcePayloads:
    lhs = random_f16((m, k), 60 + m)
    rhs = random_f16((k, n), 61 + n)
    return payloads((lhs, rhs), (torch.matmul(lhs, rhs),))


def tile_reduction_payloads() -> SourcePayloads:
    input_ = random_f16((TILE_COUNT, 4096), 100)
    return payloads((input_,), (input_.sum(dim=0),))


def large_gemm_payloads() -> SourcePayloads:
    lhs = random_f16((LARGE_GEMM_EXTENT, LARGE_GEMM_EXTENT), 200)
    rhs = random_f16((LARGE_GEMM_EXTENT, LARGE_GEMM_EXTENT), 201)
    return payloads((lhs, rhs), (torch.matmul(lhs, rhs),))


def m_tiled_gemm_payloads() -> SourcePayloads:
    lhs = random_f16((M_TILED_GEMM_M, M_TILED_GEMM_K), 300)
    rhs = random_f16((M_TILED_GEMM_K, M_TILED_GEMM_N), 301)
    return payloads((lhs, rhs), (torch.matmul(lhs, rhs),))


CASES = {
    case.key: case
    for case in (
        OptimizationComparisonCase(
            "reciprocal-implementation", "numeric-implementation",
            (F16_8192,), (F16_8192,), reciprocal_module,
            reciprocal_payloads, ("target-call-count",),
        ),
        OptimizationComparisonCase(
            "f16-common-factor", "numeric-reassociation",
            (F16_16384, F16_16384, F16_16384), (F16_16384,),
            f16_factor_module, f16_factor_payloads, ("target-call-count",),
        ),
        OptimizationComparisonCase(
            "resident-fanout-share", "resident-value-reuse",
            (F16_32768,), (F16_32768, F16_32768, F16_32768),
            resident_fanout_module, resident_fanout_payloads,
            ("workspace-bytes", "movement-call-count"),
        ),
        OptimizationComparisonCase(
            "consumer-local-recompute", "resident-value-reuse",
            (F16_524288, F16_524288),
            (F16_524288, F16_524288, F16_524288), recompute_module,
            recompute_payloads, ("workspace-bytes", "compute-call-count"),
        ),
        OptimizationComparisonCase(
            "long-steady-elementwise-add", "fixed-slot-scheduling",
            (F16_8388608, F16_8388608), (F16_8388608,),
            long_steady_add_module, long_steady_add_payloads,
            ("loop-structure", "target-call-order"),
        ),
        OptimizationComparisonCase(
            "ready-order-movement-first", "ready-operation-order",
            (F16_8192, F16_8192), (F16_8192,), ready_order_module,
            ready_order_payloads, ("target-call-order",),
        ),
        OptimizationComparisonCase(
            "gemm-aligned-physical-route", "gemm-lowering",
            (F16_64_128, F16_128_128), (F16_64_128_OUT,),
            lambda: gemm_module(64, 128, 128),
            lambda: gemm_payloads(64, 128, 128), ("gemm-call-count",),
        ),
        OptimizationComparisonCase(
            "gemm-tail-physical-route", "gemm-lowering",
            (F16_65_129, F16_129_129), (F16_65_129_OUT,),
            lambda: gemm_module(65, 129, 129),
            lambda: gemm_payloads(65, 129, 129),
            ("gemm-call-count", "tail-movement-call-count"),
        ),
        OptimizationComparisonCase(
            "tree-all-reduce", "tile-reduction",
            (F16_TILE_REDUCTION_INPUT,), (F16_TILE_REDUCTION_OUTPUT,),
            tile_reduction_module, tile_reduction_payloads,
            ("transport-call-count", "reduction-tree-depth"),
        ),
        OptimizationComparisonCase(
            "noc-resident-large-gemm", "k-tiled-gemm",
            (F16_LARGE_GEMM_LHS, F16_LARGE_GEMM_RHS),
            (F16_LARGE_GEMM_OUTPUT,),
            lambda: gemm_module(LARGE_GEMM_EXTENT, LARGE_GEMM_EXTENT,
                                LARGE_GEMM_EXTENT),
            large_gemm_payloads, ("ddr-call-count", "workspace-bytes"),
        ),
        OptimizationComparisonCase(
            "noc-resident-m-tiled-gemm", "m-tiled-gemm",
            (F16_M_TILED_GEMM_LHS, F16_M_TILED_GEMM_RHS),
            (F16_M_TILED_GEMM_OUTPUT,),
            lambda: gemm_module(M_TILED_GEMM_M, M_TILED_GEMM_K,
                                M_TILED_GEMM_N),
            m_tiled_gemm_payloads, ("ddr-call-count", "workspace-bytes"),
        ),
    )
}


def metadata(case: OptimizationComparisonCase) -> dict[str, object]:
    return {
        "name": "forward",
        "stablehlo_version": "0.0.0",
        "input_signature": [
            {"shape": list(spec.shape), "dtype": spec.metadata_dtype,
             "dynamic_dims": []}
            for spec in case.inputs
        ],
        "output_signature": [
            {"shape": list(spec.shape), "dtype": spec.metadata_dtype,
             "dynamic_dims": []}
            for spec in case.outputs
        ],
        "input_locations": [
            {"type_": "input_arg", "position": index,
             "name": f"input_{index}"}
            for index in range(len(case.inputs))
        ],
        "unused_inputs": [],
    }


def validate_source(case: OptimizationComparisonCase) -> None:
    module = case.module_factory()
    retired = (
        "mhlo.sharding", "@Sharding", "distributed_boundary",
        "execution-ranks", "logical_rank",
    )
    if any(marker in module for marker in retired):
        raise RuntimeError(f"{case.key}: source contains a retired interface")
    if "func.func @main" not in module or "stablehlo." not in module:
        raise RuntimeError(f"{case.key}: source program is incomplete")
    if not case.structural_checks or set(POLICIES) != {"none", "search"}:
        raise RuntimeError(f"{case.key}: comparison contract is incomplete")


def validate_payloads(
    case: OptimizationComparisonCase, values: SourcePayloads
) -> None:
    for tensors, specs, kind in (
        (values.inputs, case.inputs, "input"),
        (values.expected_outputs, case.outputs, "output"),
    ):
        if len(tensors) != len(specs):
            raise RuntimeError(f"{case.key}: {kind} count is invalid")
        for index, (tensor, spec) in enumerate(zip(tensors, specs, strict=True)):
            if tuple(tensor.shape) != spec.shape:
                raise RuntimeError(
                    f"{case.key}: {kind} {index} shape is invalid"
                )
            if tensor.numel() * tensor.element_size() != (
                math.prod(spec.shape) * spec.element_bytes
            ):
                raise RuntimeError(
                    f"{case.key}: {kind} {index} byte count is invalid"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"{case.key}: {kind} {index} contains a nonfinite value"
                )


def write_source(work_dir: pathlib.Path, case: OptimizationComparisonCase) -> pathlib.Path:
    source = work_dir / "source-program"
    if source.exists():
        shutil.rmtree(source)
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(case.module_factory())
    (source / "functions" / "forward.meta").write_text(
        json.dumps(metadata(case), separators=(",", ":")) + "\n"
    )
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--validate-payloads",
        action="store_true",
        help="construct and validate the deterministic host oracle",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = CASES[args.case]
    validate_source(case)
    if args.validate_payloads:
        validate_payloads(case, case.payload_factory())
    write_source(args.work_dir, case)
    print(
        "compiler_optimization_source_contract: "
        f"case={case.key} policies=none,search executable=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
