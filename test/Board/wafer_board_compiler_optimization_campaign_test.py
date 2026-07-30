#!/usr/bin/env python3
"""Compile and run same-source compiler baseline/winner board A/B cases."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable

import numpy as np

import wafer_runtime_launch_contract as runtime_launch


TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
RANK_ONE_LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
CLUSTER_LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
DIRECT_DTE_STATUS_ABI = "wafer-direct-dte-status-v2"
PROCESS_TIMEOUT_MARGIN_SECONDS = 30
SOURCE_SNAPSHOT_PATHS = (
    pathlib.Path("functions/forward.mlir"),
    pathlib.Path("functions/forward.meta"),
    pathlib.Path("functions/forward.parameter_shards.json"),
)


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    mlir_dtype: str
    metadata_dtype: str
    source_shape: tuple[int, ...] | None = None

    @property
    def element_bytes(self) -> int:
        return {
            "f16": 2,
            "bf16": 2,
            "f32": 4,
        }[self.mlir_dtype]


@dataclasses.dataclass(frozen=True)
class PairedOutputComparisonPolicy:
    class Kind(enum.Enum):
        RawExact = enum.auto()
        FloatingTolerance = enum.auto()

    kind: Kind
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    maximum_ulp: int | None = None
    signed_zero_equal: bool = False


RAW_EXACT_OUTPUT = PairedOutputComparisonPolicy(
    PairedOutputComparisonPolicy.Kind.RawExact
)
RELAXED_F16_ALGEBRA_OUTPUT = PairedOutputComparisonPolicy(
    PairedOutputComparisonPolicy.Kind.FloatingTolerance,
    absolute_tolerance=0.0009765625,
    relative_tolerance=0.001,
    maximum_ulp=1,
    signed_zero_equal=True,
)


@dataclasses.dataclass(frozen=True)
class TargetStructure:
    callsites: tuple[str, ...]
    scheduler_body_sha256: tuple[str, ...]
    straight_line_calls: tuple[str, ...] | None
    workspace_bytes: int

    @property
    def counts(self) -> collections.Counter[str]:
        return collections.Counter(self.callsites)


@dataclasses.dataclass(frozen=True)
class PairedPayloads:
    inputs: list[list[np.ndarray]]
    baseline_outputs: list[list[np.ndarray]]
    winner_outputs: list[list[np.ndarray]]


@dataclasses.dataclass(frozen=True)
class CampaignCase:
    key: str
    family: str
    rank_count: int
    launch_kind: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    module_factory: Callable[[], str]
    payload_factory: Callable[[], PairedPayloads]
    structural_oracle: Callable[
        ["TargetStructure", "TargetStructure"], None
    ]
    output_comparison: PairedOutputComparisonPolicy = RAW_EXACT_OUTPUT
    expected_launch: dict[str, object] | None = None

    @property
    def launch_contract(self) -> dict[str, object]:
        if self.expected_launch is not None:
            return self.expected_launch
        return (
            runtime_launch.RANK_ONE_KERNEL_LAUNCH
            if self.rank_count == 1
            else runtime_launch.CLUSTER_KERNEL_LAUNCH
        )


F16_16384 = TensorSpec((16384,), "f16", "float16")
F16_32768 = TensorSpec((32768,), "f16", "float16")
F16_524288 = TensorSpec((524288,), "f16", "float16")
F16_8388608 = TensorSpec((8388608,), "f16", "float16")
F16_8192 = TensorSpec((8192,), "f16", "float16")
F16_4096 = TensorSpec((4096,), "f16", "float16")
F16_64_128 = TensorSpec((64, 128), "f16", "float16")
F16_128_128 = TensorSpec((128, 128), "f16", "float16")
F16_64_128_OUT = TensorSpec((64, 128), "f16", "float16")
F16_65_129 = TensorSpec((65, 129), "f16", "float16")
F16_129_129 = TensorSpec((129, 129), "f16", "float16")
F16_65_129_OUT = TensorSpec((65, 129), "f16", "float16")
F16_4096_LOCAL = TensorSpec(
    (1, 4096), "f16", "float16", source_shape=(16, 4096)
)
NOC_RESIDENT_GEMM_RANKS = 16
NOC_RESIDENT_GEMM_EXTENT = 4096
NOC_RESIDENT_GEMM_LOCAL_K = (
    NOC_RESIDENT_GEMM_EXTENT // NOC_RESIDENT_GEMM_RANKS
)
F16_NOC_RESIDENT_GEMM_LHS_LOCAL = TensorSpec(
    (NOC_RESIDENT_GEMM_EXTENT, NOC_RESIDENT_GEMM_LOCAL_K),
    "f16",
    "float16",
    source_shape=(NOC_RESIDENT_GEMM_EXTENT, NOC_RESIDENT_GEMM_EXTENT),
)
F16_NOC_RESIDENT_GEMM_RHS_LOCAL = TensorSpec(
    (NOC_RESIDENT_GEMM_LOCAL_K, NOC_RESIDENT_GEMM_EXTENT),
    "f16",
    "float16",
    source_shape=(NOC_RESIDENT_GEMM_EXTENT, NOC_RESIDENT_GEMM_EXTENT),
)
F16_NOC_RESIDENT_GEMM_OUTPUT = TensorSpec(
    (NOC_RESIDENT_GEMM_EXTENT, NOC_RESIDENT_GEMM_EXTENT),
    "f16",
    "float16",
)
NOC_RESIDENT_M_SHARDED_GEMM_RANKS = 16
NOC_RESIDENT_M_SHARDED_GEMM_M = 4096
NOC_RESIDENT_M_SHARDED_GEMM_K = 1024
NOC_RESIDENT_M_SHARDED_GEMM_N = 4096
NOC_RESIDENT_M_SHARDED_GEMM_LOCAL_M = (
    NOC_RESIDENT_M_SHARDED_GEMM_M // NOC_RESIDENT_M_SHARDED_GEMM_RANKS
)
F16_NOC_RESIDENT_M_SHARDED_GEMM_LHS_LOCAL = TensorSpec(
    (
        NOC_RESIDENT_M_SHARDED_GEMM_LOCAL_M,
        NOC_RESIDENT_M_SHARDED_GEMM_K,
    ),
    "f16",
    "float16",
    source_shape=(
        NOC_RESIDENT_M_SHARDED_GEMM_M,
        NOC_RESIDENT_M_SHARDED_GEMM_K,
    ),
)
F16_NOC_RESIDENT_M_SHARDED_GEMM_RHS_LOCAL = TensorSpec(
    (
        NOC_RESIDENT_M_SHARDED_GEMM_K,
        NOC_RESIDENT_M_SHARDED_GEMM_N,
    ),
    "f16",
    "float16",
)
F16_NOC_RESIDENT_M_SHARDED_GEMM_OUTPUT_LOCAL = TensorSpec(
    (
        NOC_RESIDENT_M_SHARDED_GEMM_LOCAL_M,
        NOC_RESIDENT_M_SHARDED_GEMM_N,
    ),
    "f16",
    "float16",
    source_shape=(
        NOC_RESIDENT_M_SHARDED_GEMM_M,
        NOC_RESIDENT_M_SHARDED_GEMM_N,
    ),
)


def elementwise_module(
    arguments: str, results: str, body: str, return_values: str
) -> str:
    return_types = results[1:-1] if results.startswith("(") else results
    return f"""\
module {{
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
        """\
    %one = stablehlo.constant dense<1.000000e+00> : tensor<8192xf16>
    %result = stablehlo.divide %one, %input : tensor<8192xf16>""",
        "%result",
    )


def f16_factor_module() -> str:
    return """\
module {
  func.func @main(
      %a: tensor<16384xf16>,
      %b: tensor<16384xf16>,
      %c: tensor<16384xf16>) -> tensor<16384xf16> {
    %result = "stablehlo.map"(%a, %b, %c) ({
    ^bb0(%av: tensor<f16>, %bv: tensor<f16>, %cv: tensor<f16>):
      %ab = stablehlo.multiply %av, %bv : tensor<f16>
      %ac = stablehlo.multiply %av, %cv : tensor<f16>
      %sum = stablehlo.add %ab, %ac : tensor<f16>
      stablehlo.return %sum : tensor<f16>
    }) {dimensions = array<i64: 0>}
      : (tensor<16384xf16>, tensor<16384xf16>, tensor<16384xf16>)
        -> tensor<16384xf16>
    return %result : tensor<16384xf16>
  }
}
"""


def resident_fanout_module() -> str:
    return elementwise_module(
        "%input: tensor<32768xf16>",
        "(tensor<32768xf16>, tensor<32768xf16>, tensor<32768xf16>)",
        """\
    %producer = stablehlo.multiply %input, %input : tensor<32768xf16>
    %left = stablehlo.add %producer, %producer : tensor<32768xf16>
    %right = stablehlo.multiply %producer, %producer : tensor<32768xf16>""",
        "%producer, %left, %right",
    )


def recompute_module() -> str:
    return elementwise_module(
        "%input: tensor<524288xf16>, %other: tensor<524288xf16>",
        "(tensor<524288xf16>, tensor<524288xf16>, tensor<524288xf16>)",
        """\
    %producer = stablehlo.multiply %input, %input : tensor<524288xf16>
    %left = stablehlo.add %producer, %producer : tensor<524288xf16>
    %middle = stablehlo.multiply %other, %other : tensor<524288xf16>
    %right = stablehlo.multiply %producer, %producer : tensor<524288xf16>""",
        "%left, %middle, %right",
    )


def long_steady_add_module() -> str:
    return elementwise_module(
        "%lhs: tensor<8388608xf16>, %rhs: tensor<8388608xf16>",
        "tensor<8388608xf16>",
        """\
    %result = stablehlo.add %lhs, %rhs : tensor<8388608xf16>""",
        "%result",
    )


def ready_order_module() -> str:
    return elementwise_module(
        "%a: tensor<8192xf16>, %b: tensor<8192xf16>",
        "tensor<8192xf16>",
        """\
    %producer = stablehlo.multiply %a, %a : tensor<8192xf16>
    %left = stablehlo.add %producer, %b : tensor<8192xf16>
    %result = stablehlo.add %left, %b : tensor<8192xf16>""",
        "%result",
    )


def gemm_module(m: int, k: int, n: int) -> str:
    return f"""\
module {{
  func.func @main(
      %lhs: tensor<{m}x{k}xf16>,
      %rhs: tensor<{k}x{n}xf16>) -> tensor<{m}x{n}xf16> {{
    %result = "stablehlo.dot_general"(%lhs, %rhs) {{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    }} : (tensor<{m}x{k}xf16>, tensor<{k}x{n}xf16>) -> tensor<{m}x{n}xf16>
    return %result : tensor<{m}x{n}xf16>
  }}
}}
"""


def all_reduce_module() -> str:
    devices = ",".join(str(rank) for rank in range(16))
    return f"""\
module {{
  func.func @main(%input: tensor<16x4096xf16>) -> tensor<4096xf16> {{
    %sharded = stablehlo.custom_call @Sharding(%input) {{
      backend_config = "",
      mhlo.sharding = "{{devices=[16,1]{devices}}}"
    }} : (tensor<16x4096xf16>) -> tensor<16x4096xf16>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %result = "stablehlo.reduce"(%sharded, %zero) ({{
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f16>
      stablehlo.return %sum : tensor<f16>
    }}) {{dimensions = array<i64: 0>}}
      : (tensor<16x4096xf16>, tensor<f16>) -> tensor<4096xf16>
    return %result : tensor<4096xf16>
  }}
}}
"""


def noc_resident_large_gemm_module() -> str:
    devices = ",".join(str(rank) for rank in range(NOC_RESIDENT_GEMM_RANKS))
    lhs_sharding = (
        f"{{devices=[1,{NOC_RESIDENT_GEMM_RANKS}]{devices}}}"
    )
    rhs_sharding = (
        f"{{devices=[{NOC_RESIDENT_GEMM_RANKS},1]{devices}}}"
    )
    extent = NOC_RESIDENT_GEMM_EXTENT
    return f"""\
module {{
  func.func @main(
      %lhs: tensor<{extent}x{extent}xf16>
          {{mhlo.sharding = "{lhs_sharding}"}},
      %rhs: tensor<{extent}x{extent}xf16>
          {{mhlo.sharding = "{rhs_sharding}"}})
      -> (tensor<{extent}x{extent}xf16>
          {{mhlo.sharding = "{{replicated}}"}}) {{
    %result = "stablehlo.dot_general"(%lhs, %rhs) {{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    }} : (tensor<{extent}x{extent}xf16>,
          tensor<{extent}x{extent}xf16>)
        -> tensor<{extent}x{extent}xf16>
    return %result : tensor<{extent}x{extent}xf16>
  }}
}}
"""


def noc_resident_m_sharded_gemm_module() -> str:
    ranks = NOC_RESIDENT_M_SHARDED_GEMM_RANKS
    devices = ",".join(str(rank) for rank in range(ranks))
    m_sharding = f"{{devices=[{ranks},1]{devices}}}"
    m = NOC_RESIDENT_M_SHARDED_GEMM_M
    k = NOC_RESIDENT_M_SHARDED_GEMM_K
    n = NOC_RESIDENT_M_SHARDED_GEMM_N
    return f"""\
module {{
  func.func @main(
      %lhs: tensor<{m}x{k}xf16>
          {{mhlo.sharding = "{m_sharding}"}},
      %rhs: tensor<{k}x{n}xf16>
          {{mhlo.sharding = "{{replicated}}"}})
      -> (tensor<{m}x{n}xf16>
          {{mhlo.sharding = "{m_sharding}"}}) {{
    %result = "stablehlo.dot_general"(%lhs, %rhs) {{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    }} : (tensor<{m}x{k}xf16>, tensor<{k}x{n}xf16>)
        -> tensor<{m}x{n}xf16>
    return %result : tensor<{m}x{n}xf16>
  }}
}}
"""


def replicated(payloads: list[np.ndarray]) -> list[list[np.ndarray]]:
    return [payloads]


def unchanged_numeric_payloads(
    inputs: list[list[np.ndarray]], outputs: list[list[np.ndarray]]
) -> PairedPayloads:
    return PairedPayloads(
        inputs,
        outputs,
        [[array.copy() for array in arrays] for arrays in outputs],
    )


def reciprocal_payloads() -> PairedPayloads:
    exponents = (np.arange(8192, dtype=np.int32) % 17) - 8
    signs = np.where(np.arange(8192) % 3 == 0, -1.0, 1.0).astype(np.float16)
    input_ = np.ldexp(signs, exponents).astype("<f2")
    baseline = np.divide(np.float16(1.0), input_).astype("<f2")
    winner = np.reciprocal(input_).astype("<f2")
    return PairedPayloads(
        replicated([input_]), replicated([baseline]), replicated([winner])
    )


def f16_factor_payloads() -> PairedPayloads:
    indices = np.arange(16384, dtype=np.int32)
    nonzero_factors = np.array((-4, -3, -2, -1, 1, 2, 3, 4), dtype="<f2")
    a = nonzero_factors[indices % nonzero_factors.size]
    b = (1 + ((indices * 7 + 1) % 4)).astype("<f2")
    c = (1 + ((indices * 11 + 2) % 4)).astype("<f2")
    # Exercise the only intentional raw-bit difference in this paired oracle:
    # the source DAG produces +0 while the factored DAG produces -0.
    a[0] = np.float16(0.0)
    b[0] = np.float16(0.0)
    c[0] = np.float16(-1.0)
    lhs = (a * b).astype("<f2")
    rhs = (a * c).astype("<f2")
    baseline = (lhs + rhs).astype("<f2")
    winner = (a * (b + c).astype("<f2")).astype("<f2")
    expected_i32 = a.astype(np.int32) * (
        b.astype(np.int32) + c.astype(np.int32)
    )
    if not np.array_equal(baseline.astype(np.int32), expected_i32):
        raise RuntimeError("f16 factor source expression is not exact")
    if (
        np.signbit(baseline[0])
        or not np.signbit(winner[0])
        or baseline[0] != winner[0]
    ):
        raise RuntimeError("f16 factor signed-zero control is ineffective")
    return PairedPayloads(
        replicated([a, b, c]),
        replicated([baseline]),
        replicated([winner]),
    )


def resident_fanout_payloads() -> PairedPayloads:
    input_ = ((np.arange(32768, dtype=np.int32) % 7) - 3).astype("<f2")
    producer = (input_ * input_).astype("<f2")
    left = (producer + producer).astype("<f2")
    right = (producer * producer).astype("<f2")
    return unchanged_numeric_payloads(
        replicated([input_]), replicated([producer, left, right])
    )


def recompute_payloads() -> PairedPayloads:
    indices = np.arange(524288, dtype=np.int32)
    input_ = ((indices % 7) - 3).astype("<f2")
    other = (((indices * 5) % 7) - 3).astype("<f2")
    producer = (input_ * input_).astype("<f2")
    left = (producer + producer).astype("<f2")
    middle = (other * other).astype("<f2")
    right = (producer * producer).astype("<f2")
    return unchanged_numeric_payloads(
        replicated([input_, other]), replicated([left, middle, right])
    )


def long_steady_add_payloads() -> PairedPayloads:
    indices = np.arange(8388608, dtype=np.int32)
    lhs = ((indices % 31) - 15).astype("<f2")
    rhs = (((indices * 7) % 29) - 14).astype("<f2")
    result = (lhs + rhs).astype("<f2")
    return unchanged_numeric_payloads(
        replicated([lhs, rhs]), replicated([result])
    )


def ready_order_payloads() -> PairedPayloads:
    indices = np.arange(8192, dtype=np.int32)
    a = ((indices % 15) - 7).astype("<f2")
    b = (((indices * 3) % 11) - 5).astype("<f2")
    expected = ((a * a + b) + b).astype("<f2")
    if np.array_equal(expected, b):
        raise RuntimeError("ready-order expected output does not depend on a")
    changed_a = (a + np.float16(1.0)).astype("<f2")
    changed_expected = ((changed_a * changed_a + b) + b).astype("<f2")
    if np.array_equal(expected, changed_expected):
        raise RuntimeError("ready-order a-path sensitivity control is ineffective")
    return unchanged_numeric_payloads(
        replicated([a, b]), replicated([expected])
    )


def gemm_payloads(
    m: int, k: int, n: int
) -> PairedPayloads:
    k_lanes = np.arange(k, dtype=np.int32)
    lhs = np.zeros((m, k), dtype="<f2")
    lhs[k_lanes % m, k_lanes] = np.float16(1.0)
    k_indices = np.arange(k, dtype=np.int32)[:, None]
    n_indices = np.arange(n, dtype=np.int32)[None, :]
    rhs_i32 = 1 + ((k_indices * 5 + n_indices * 3) % 8)
    rhs_i32 = np.where((k_indices + n_indices) % 2, -rhs_i32, rhs_i32)
    rhs = rhs_i32.astype("<f2")
    expected_i32 = lhs.astype(np.int32) @ rhs_i32.astype(np.int32)
    expected = expected_i32.astype("<f2")
    if (
        not np.all(np.count_nonzero(lhs, axis=0) == 1)
        or lhs[(k - 1) % m, k - 1] == np.float16(0.0)
        or not np.all(np.count_nonzero(rhs, axis=1) > 0)
    ):
        raise RuntimeError("GEMM payload does not exercise every K lane")
    if not np.array_equal(expected.astype(np.int32), expected_i32):
        raise RuntimeError("GEMM expected values are not exact in f16")
    return unchanged_numeric_payloads(
        replicated([lhs, rhs]), replicated([expected])
    )


def all_reduce_payloads() -> PairedPayloads:
    lanes = np.arange(4096, dtype=np.int32)
    inputs = [
        (((lanes * (rank + 3) + rank * 11) % 9) - 4)
        .astype("<f2")[None, :]
        for rank in range(16)
    ]
    total = np.zeros(4096, dtype=np.float32)
    for input_ in inputs:
        total += input_[0].astype(np.float32)
    expected = total.astype("<f2")
    return unchanged_numeric_payloads(
        [[input_] for input_ in inputs],
        [[expected] for _ in range(16)],
    )


def noc_resident_large_gemm_payloads() -> PairedPayloads:
    extent = NOC_RESIDENT_GEMM_EXTENT
    local_k_extent = NOC_RESIDENT_GEMM_LOCAL_K
    columns = np.arange(extent, dtype=np.int32)
    expected = np.empty((extent, extent), dtype="<f2")
    inputs: list[list[np.ndarray]] = []
    for rank in range(NOC_RESIDENT_GEMM_RANKS):
        lhs = np.zeros((extent, local_k_extent), dtype="<f2")
        rhs = np.empty((local_k_extent, extent), dtype="<f2")
        for local_k in range(local_k_extent):
            global_k = rank * local_k_extent + local_k
            lhs[global_k, local_k] = np.float16(1.0)
            row = 1 + ((global_k * 5 + columns * 3) % 8)
            row = np.where((global_k + columns) % 2, -row, row)
            rhs[local_k] = row.astype("<f2")
            expected[global_k] = rhs[local_k]
        inputs.append([lhs, rhs])

    if not np.array_equal(expected, expected.astype(np.int32).astype("<f2")):
        raise RuntimeError(
            "NoC-resident GEMM expected values are not exact integers in f16"
        )
    outputs = [[expected] for _ in range(NOC_RESIDENT_GEMM_RANKS)]
    return PairedPayloads(inputs, outputs, list(outputs))


def noc_resident_m_sharded_gemm_payloads() -> PairedPayloads:
    ranks = NOC_RESIDENT_M_SHARDED_GEMM_RANKS
    local_m = NOC_RESIDENT_M_SHARDED_GEMM_LOCAL_M
    k = NOC_RESIDENT_M_SHARDED_GEMM_K
    n = NOC_RESIDENT_M_SHARDED_GEMM_N
    k_indices = np.arange(k, dtype=np.int32)[:, None]
    columns = np.arange(n, dtype=np.int32)[None, :]
    rhs_i32 = 1 + ((k_indices * 5 + columns * 3) % 8)
    rhs_i32 = np.where((k_indices + columns) % 2, -rhs_i32, rhs_i32)
    rhs = rhs_i32.astype("<f2")

    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    local_rows = np.arange(local_m, dtype=np.int32)
    for rank in range(ranks):
        global_rows = rank * local_m + local_rows
        contributing_k = global_rows % k
        lhs = np.zeros((local_m, k), dtype="<f2")
        lhs[local_rows, contributing_k] = np.float16(1.0)
        expected = rhs[contributing_k].copy()
        inputs.append([lhs, rhs])
        outputs.append([expected])

    return unchanged_numeric_payloads(inputs, outputs)


def require_call(
    counts: collections.Counter[str], fragment: str, *, present: bool
) -> None:
    matching = sum(count for name, count in counts.items() if fragment in name)
    if present and matching == 0:
        raise RuntimeError(
            f"scheduler bodies omitted required target callsite {fragment!r}"
        )
    if not present and matching != 0:
        raise RuntimeError(
            f"scheduler bodies retained forbidden target callsite {fragment!r}"
        )


def count_fragment(structure: TargetStructure, fragment: str) -> int:
    return sum(
        count for name, count in structure.counts.items() if fragment in name
    )


def reciprocal_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    require_call(baseline.counts, "elementwise_div", present=True)
    require_call(winner.counts, "elementwise_recip", present=True)
    require_call(winner.counts, "elementwise_div", present=False)


def f16_factor_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    baseline_mul = count_fragment(baseline, "elementwise_mul")
    winner_mul = count_fragment(winner, "elementwise_mul")
    if not (winner_mul > 0 and winner_mul < baseline_mul):
        raise RuntimeError(
            "f16 factor winner did not reduce target multiply callsites: "
            f"baseline={baseline_mul} winner={winner_mul}"
        )
    require_call(winner.counts, "elementwise_add", present=True)


def resident_fanout_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    baseline_movement = count_fragment(baseline, "_rdma") + count_fragment(
        baseline, "_wdma"
    )
    winner_movement = count_fragment(winner, "_rdma") + count_fragment(
        winner, "_wdma"
    )
    if not (winner_movement < baseline_movement):
        raise RuntimeError(
            "resident/fusion winner did not remove target DDR callsites: "
            f"baseline={baseline_movement} winner={winner_movement}"
        )
    for fragment in ("elementwise_add", "elementwise_mul"):
        require_call(winner.counts, fragment, present=True)


def recompute_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    baseline_mul = count_fragment(baseline, "elementwise_mul")
    winner_mul = count_fragment(winner, "elementwise_mul")
    if not (
        winner_mul > baseline_mul
        and winner.workspace_bytes < baseline.workspace_bytes
    ):
        raise RuntimeError(
            "recompute winner did not exchange compute for spill storage: "
            f"mul {baseline_mul}->{winner_mul}, "
            f"workspace {baseline.workspace_bytes}->{winner.workspace_bytes}"
        )


def long_steady_add_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    require_call(baseline.counts, "elementwise_add", present=True)
    require_call(winner.counts, "elementwise_add", present=True)
    if baseline.scheduler_body_sha256 == winner.scheduler_body_sha256:
        raise RuntimeError(
            "long steady production scheduler is identical to its baseline"
        )
    fragments = ("_rdma", "elementwise_add", "_wdma")
    if not all(
        count_fragment(winner, fragment)
        > count_fragment(baseline, fragment)
        for fragment in fragments
    ):
        raise RuntimeError(
            "production scheduler does not expose distinct "
            "prologue/steady/epilogue target callsites"
        )
    if (
        baseline.straight_line_calls is None
        or winner.straight_line_calls is not None
    ):
        raise RuntimeError(
            "fixed-slot pair must change a straight-line baseline into a looped "
            "production scheduler"
        )


def ready_order_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    def without_fences(
        counts: collections.Counter[str],
    ) -> collections.Counter[str]:
        return collections.Counter(
            {
                name: count
                for name, count in counts.items()
                if "fence" not in name and "wait" not in name
            }
        )

    if (
        baseline.straight_line_calls is None
        or winner.straight_line_calls is None
    ):
        raise RuntimeError(
            "ready-order call order is not proven by straight-line scheduler bodies"
        )
    if (
        without_fences(baseline.counts) != without_fences(winner.counts)
        or baseline.straight_line_calls == winner.straight_line_calls
    ):
        raise RuntimeError(
            "ready-order pair must preserve non-completion calls and change order"
        )
    baseline_completion = sum(
        count
        for name, count in baseline.counts.items()
        if "fence" in name or "wait" in name
    )
    winner_completion = sum(
        count
        for name, count in winner.counts.items()
        if "fence" in name or "wait" in name
    )
    if winner_completion > baseline_completion:
        raise RuntimeError("ready-order winner added completion calls")
    movement = ("wafer_tx81_rdma", "wafer_tx81_wdma")

    def leading_movement(calls: tuple[str, ...]) -> int:
        count = 0
        for call in calls:
            if any(fragment in call for fragment in movement):
                count += 1
                continue
            if "local_fence" in call:
                continue
            break
        return count

    if leading_movement(
        winner.straight_line_calls
    ) <= leading_movement(baseline.straight_line_calls):
        raise RuntimeError("winner is not more movement-first than baseline")


def gemm_route_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    require_call(baseline.counts, "_gemm", present=True)
    require_call(winner.counts, "_gemm", present=True)
    baseline_layout = count_fragment(baseline, "gather_scatter")
    winner_layout = count_fragment(winner, "gather_scatter")
    if winner_layout >= baseline_layout:
        raise RuntimeError(
            "physical-route winner did not reduce layout callsites: "
            f"baseline={baseline_layout} winner={winner_layout}"
        )


def collective_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    required = ("direct_dte_send_issue", "direct_dte_recv_prepare")
    for fragment in required:
        require_call(baseline.counts, fragment, present=True)
        require_call(winner.counts, fragment, present=True)
    baseline_prepares = sum(
        count_fragment(baseline, fragment) for fragment in required
    )
    winner_prepares = sum(
        count_fragment(winner, fragment) for fragment in required
    )
    if not (
        winner_prepares < baseline_prepares
        and winner.workspace_bytes < baseline.workspace_bytes
    ):
        raise RuntimeError(
            "collective winner did not reduce static prepare callsites/staging: "
            f"prepares {baseline_prepares}->{winner_prepares}, "
            f"workspace {baseline.workspace_bytes}->{winner.workspace_bytes}"
        )
    require_call(winner.counts, "elementwise_add", present=True)
    if baseline.scheduler_body_sha256 == winner.scheduler_body_sha256:
        raise RuntimeError(
            "collective baseline and winner scheduler bodies are identical"
        )


def noc_resident_large_gemm_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    for structure in (baseline, winner):
        for fragment in (
            "_gemm",
            "direct_dte_send_prepare",
            "direct_dte_recv_prepare",
            "direct_dte_begin_after_prepare",
            "direct_dte_wait",
        ):
            require_call(structure.counts, fragment, present=True)
    if baseline.scheduler_body_sha256 == winner.scheduler_body_sha256:
        raise RuntimeError(
            "NoC-resident GEMM production scheduler is identical to its baseline"
        )

    baseline_ddr_calls = count_fragment(
        baseline, "_rdma"
    ) + count_fragment(baseline, "_wdma")
    winner_ddr_calls = count_fragment(
        winner, "_rdma"
    ) + count_fragment(winner, "_wdma")
    if not (
        winner_ddr_calls < baseline_ddr_calls
        or winner.workspace_bytes < baseline.workspace_bytes
    ):
        raise RuntimeError(
            "NoC-resident GEMM winner did not reduce DDR callsites or "
            "workspace: "
            f"DDR calls {baseline_ddr_calls}->{winner_ddr_calls}, "
            f"workspace {baseline.workspace_bytes}->{winner.workspace_bytes}"
        )


def noc_resident_m_sharded_gemm_oracle(
    baseline: TargetStructure, winner: TargetStructure
) -> None:
    require_call(baseline.counts, "_gemm", present=True)
    require_call(winner.counts, "_gemm", present=True)
    for fragment in (
        "direct_dte_send_prepare",
        "direct_dte_recv_prepare",
        "direct_dte_begin_after_prepare",
        "direct_dte_wait",
    ):
        require_call(winner.counts, fragment, present=True)
    if baseline.scheduler_body_sha256 == winner.scheduler_body_sha256:
        raise RuntimeError(
            "M-sharded NoC-resident GEMM scheduler is identical to its baseline"
        )

    baseline_ddr_calls = count_fragment(
        baseline, "_rdma"
    ) + count_fragment(baseline, "_wdma")
    winner_ddr_calls = count_fragment(
        winner, "_rdma"
    ) + count_fragment(winner, "_wdma")
    if not (
        winner_ddr_calls < baseline_ddr_calls
        or winner.workspace_bytes < baseline.workspace_bytes
    ):
        raise RuntimeError(
            "M-sharded NoC-resident GEMM winner did not reduce DDR callsites "
            "or workspace: "
            f"DDR calls {baseline_ddr_calls}->{winner_ddr_calls}, "
            f"workspace {baseline.workspace_bytes}->{winner.workspace_bytes}"
        )


CASES = {
    case.key: case
    for case in (
        CampaignCase(
            "reciprocal-implementation",
            "numeric-dag-implementation",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_8192,),
            (F16_8192,),
            reciprocal_module,
            reciprocal_payloads,
            reciprocal_oracle,
        ),
        CampaignCase(
            "f16-common-factor",
            "numeric-dag-implementation",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_16384, F16_16384, F16_16384),
            (F16_16384,),
            f16_factor_module,
            f16_factor_payloads,
            f16_factor_oracle,
            RELAXED_F16_ALGEBRA_OUTPUT,
        ),
        CampaignCase(
            "resident-fanout-share",
            "resident-share-recompute",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_32768,),
            (F16_32768, F16_32768, F16_32768),
            resident_fanout_module,
            resident_fanout_payloads,
            resident_fanout_oracle,
        ),
        CampaignCase(
            "consumer-local-recompute",
            "resident-share-recompute",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_524288, F16_524288),
            (F16_524288, F16_524288, F16_524288),
            recompute_module,
            recompute_payloads,
            recompute_oracle,
        ),
        CampaignCase(
            "long-steady-elementwise-add",
            "static-fixed-slot-candidate",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_8388608, F16_8388608),
            (F16_8388608,),
            long_steady_add_module,
            long_steady_add_payloads,
            long_steady_add_oracle,
        ),
        CampaignCase(
            "ready-order-movement-first",
            "ready-order",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_8192, F16_8192),
            (F16_8192,),
            ready_order_module,
            ready_order_payloads,
            ready_order_oracle,
        ),
        CampaignCase(
            "gemm-aligned-physical-route",
            "tile-physical-route",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_64_128, F16_128_128),
            (F16_64_128_OUT,),
            lambda: gemm_module(64, 128, 128),
            lambda: gemm_payloads(64, 128, 128),
            gemm_route_oracle,
        ),
        CampaignCase(
            "gemm-tail-physical-route",
            "tile-physical-route",
            1,
            RANK_ONE_LAUNCH_KIND,
            (F16_65_129, F16_129_129),
            (F16_65_129_OUT,),
            lambda: gemm_module(65, 129, 129),
            lambda: gemm_payloads(65, 129, 129),
            gemm_route_oracle,
        ),
        CampaignCase(
            "tree-all-reduce",
            "collective-algorithm",
            16,
            CLUSTER_LAUNCH_KIND,
            (F16_4096_LOCAL,),
            (F16_4096,),
            all_reduce_module,
            all_reduce_payloads,
            collective_oracle,
        ),
        CampaignCase(
            "noc-resident-large-gemm",
            "noc-resident-dataflow",
            NOC_RESIDENT_GEMM_RANKS,
            CLUSTER_LAUNCH_KIND,
            (
                F16_NOC_RESIDENT_GEMM_LHS_LOCAL,
                F16_NOC_RESIDENT_GEMM_RHS_LOCAL,
            ),
            (F16_NOC_RESIDENT_GEMM_OUTPUT,),
            noc_resident_large_gemm_module,
            noc_resident_large_gemm_payloads,
            noc_resident_large_gemm_oracle,
        ),
        CampaignCase(
            "noc-resident-m-sharded-gemm",
            "noc-resident-dataflow",
            NOC_RESIDENT_M_SHARDED_GEMM_RANKS,
            CLUSTER_LAUNCH_KIND,
            (
                F16_NOC_RESIDENT_M_SHARDED_GEMM_LHS_LOCAL,
                F16_NOC_RESIDENT_M_SHARDED_GEMM_RHS_LOCAL,
            ),
            (F16_NOC_RESIDENT_M_SHARDED_GEMM_OUTPUT_LOCAL,),
            noc_resident_m_sharded_gemm_module,
            noc_resident_m_sharded_gemm_payloads,
            noc_resident_m_sharded_gemm_oracle,
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-compile-test", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--tx8-objdump", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        for partial in (error.stdout, error.stderr):
            if partial:
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(partial, end="", file=sys.stderr)
        raise RuntimeError(
            "paired compiler-optimization process exceeded its outer deadline; "
            "the test will not retry or invoke reset/power operations"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}"
        )
    return result


def metadata(case: CampaignCase) -> dict[str, object]:
    return {
        "name": "forward",
        "stablehlo_version": "0.0.0",
        "input_signature": [
            {
                "shape": list(spec.source_shape or spec.shape),
                "dtype": spec.metadata_dtype,
                "dynamic_dims": [],
            }
            for spec in case.inputs
        ],
        "output_signature": [
            {
                "shape": list(spec.source_shape or spec.shape),
                "dtype": spec.metadata_dtype,
                "dynamic_dims": [],
            }
            for spec in case.outputs
        ],
        "input_locations": [
            {
                "type_": "input_arg",
                "position": index,
                "name": f"input_{index}",
            }
            for index in range(len(case.inputs))
        ],
        "unused_inputs": [],
    }


def write_source(work_dir: pathlib.Path, case: CampaignCase) -> pathlib.Path:
    known_children = {
        "source-program",
        "baseline-package",
        "winner-package",
        "raw",
        "target-structure.json",
        "board-observations.json",
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    unknown_children = [
        child for child in work_dir.iterdir() if child.name not in known_children
    ]
    if unknown_children:
        raise RuntimeError(
            "--work-dir contains unknown entries; refusing cleanup: "
            + ", ".join(sorted(child.name for child in unknown_children))
        )
    for name in sorted(known_children):
        child = work_dir / name
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(case.module_factory())
    (source / "functions" / "forward.meta").write_text(
        json.dumps(metadata(case), separators=(",", ":")) + "\n"
    )
    return source


def compile_package(
    compiler: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    case: CampaignCase,
    *,
    reserved_baseline: bool,
    profile: bool = False,
) -> None:
    environment = os.environ.copy()
    for failure_injection in (
        "WAFER_TEST_FAIL_AFTER_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_TARGET_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_PACKAGE_LOGICAL_RANK",
    ):
        environment.pop(failure_injection, None)
    environment.pop("WAFER_TEST_SELECT_RESERVED_BASELINE", None)
    command = [
        str(compiler),
        "--input-program-dir",
        str(source),
        "--output-program-dir",
        str(output),
        f"--execution-ranks={case.rank_count}",
        f"--target-profile={TARGET_PROFILE}",
        f"--launch-kind={case.launch_kind}",
    ]
    if reserved_baseline:
        command.append("--optimization-preset=none")
    if profile:
        command.append("--profile")
    result = run(command, environment=environment)
    expected = (
        "wafer-compile: published verified package with "
        f"execution-ranks={case.rank_count}"
    )
    if expected not in result.stdout:
        raise RuntimeError(f"compiler did not publish the {case.key} package")
    published_companion = "wafer-compile: published profile companion:"
    if profile and published_companion not in result.stdout:
        raise RuntimeError(
            f"compiler did not publish the {case.key} profile companion"
        )
    if not profile and published_companion in result.stdout:
        raise RuntimeError(
            f"ordinary {case.key} compilation unexpectedly published a "
            "profile companion"
        )


def normalized_manifest(manifest: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(manifest))
    resources = normalized.get("resources")
    host_resource_keys = (
        {
            resource.get("id"): (
                resource.get("rank"),
                resource.get("role"),
                resource.get("role_index"),
            )
            for resource in resources
            if isinstance(resource, dict) and resource.get("host_visible")
        }
        if isinstance(resources, list)
        else {}
    )
    if isinstance(resources, list):
        visible_resources = [
            dict(resource)
            for resource in resources
            if isinstance(resource, dict) and resource.get("host_visible")
        ]
        for resource in visible_resources:
            resource.pop("id", None)
        normalized["resources"] = sorted(
            visible_resources,
            key=lambda resource: (
                resource.get("rank"),
                resource.get("role"),
                resource.get("role_index"),
            ),
        )
    entries = normalized.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            slots = entry.get("slots")
            if isinstance(slots, list):
                entry["slots"] = [
                    {
                        **slot,
                        "resource": host_resource_keys[slot.get("resource")],
                    }
                    for slot in slots
                    if isinstance(slot, dict)
                    and slot.get("resource") in host_resource_keys
                ]
            transport = entry.get("transport")
            if isinstance(transport, dict):
                transport.pop("status_resource", None)
    modules = normalized.get("modules")
    if isinstance(modules, list):
        for module in modules:
            if isinstance(module, dict):
                module.pop("digest", None)
    return normalized


def validate_paired_packages(
    baseline: pathlib.Path,
    winner: pathlib.Path,
    case: CampaignCase,
    *,
    target_profile: str = TARGET_PROFILE,
) -> tuple[
    dict[str, dict[tuple[int, str, int], int]],
    dict[str, set[int]],
    dict[str, set[tuple[int, int]]],
]:
    manifests = {
        "baseline": json.loads((baseline / "manifest.json").read_text()),
        "winner": json.loads((winner / "manifest.json").read_text()),
    }
    if normalized_manifest(manifests["baseline"]) != normalized_manifest(
        manifests["winner"]
    ):
        raise RuntimeError(
            "baseline/winner host-visible package boundary or ABI differs"
        )
    bindings_by_variant: dict[str, dict[tuple[int, str, int], int]] = {}
    output_ids_by_variant: dict[str, set[int]] = {}
    completion_evidence_by_variant: dict[str, set[tuple[int, int]]] = {}
    for variant, manifest in manifests.items():
        runtime_launch.require_manifest_launch(
            manifest,
            case.launch_contract,
            context=f"paired {variant}",
        )
        if (
            manifest.get("rank_count") != case.rank_count
            or manifest.get("target", {}).get("profile") != target_profile
        ):
            raise RuntimeError("paired package target contract is invalid")
        resources = manifest.get("resources")
        if not isinstance(resources, list):
            raise RuntimeError("package resources are not a list")
        bindings: dict[tuple[int, str, int], int] = {}
        output_ids: set[int] = set()
        for resource in resources:
            if not isinstance(resource, dict) or not resource.get("host_visible"):
                continue
            rank = resource.get("rank")
            role = resource.get("role")
            role_index = resource.get("role_index")
            resource_id = resource.get("id")
            if not all(
                isinstance(value, int)
                for value in (rank, role_index, resource_id)
            ):
                raise RuntimeError(
                    "host-visible package resource identity is invalid"
                )
            key = (rank, role, role_index)
            if key in bindings:
                raise RuntimeError(f"duplicate host-visible resource {key}")
            expected_specs = (
                case.inputs if role == "user_input" else case.outputs
            )
            if role not in ("user_input", "output") or not (
                0 <= role_index < len(expected_specs)
            ):
                raise RuntimeError(f"unexpected host-visible resource {key}")
            spec = expected_specs[role_index]
            if (
                resource.get("type")
                != {"dtype": spec.mlir_dtype, "shape": list(spec.shape)}
                or resource.get("bytes")
                != int(np.prod(spec.shape, dtype=np.int64))
                * spec.element_bytes
                or resource.get("access")
                != ("read_only" if role == "user_input" else "write_only")
            ):
                raise RuntimeError(
                    f"host-visible resource contract differs for {key}"
                )
            bindings[key] = resource_id
            if role == "output":
                output_ids.add(resource_id)
        expected_keys = {
            (rank, role, index)
            for rank in range(case.rank_count)
            for role, specs in (
                ("user_input", case.inputs),
                ("output", case.outputs),
            )
            for index in range(len(specs))
        }
        if set(bindings) != expected_keys:
            raise RuntimeError(
                f"{variant} package host resource domain is incomplete"
            )
        bindings_by_variant[variant] = bindings
        output_ids_by_variant[variant] = output_ids
        completions = manifest.get("completions")
        entries = manifest.get("entries")
        if not isinstance(completions, list) or not isinstance(entries, list):
            raise RuntimeError("package entry/completion records are not lists")
        completion_evidence = {
            (completion.get("id"), completion.get("rank"))
            for completion in completions
            if isinstance(completion, dict)
            and completion.get("kind") == "entry_return"
            and isinstance(completion.get("id"), int)
            and isinstance(completion.get("rank"), int)
        }
        entry_evidence = {
            (entry.get("terminal_completion"), entry.get("rank"))
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("terminal_completion"), int)
            and isinstance(entry.get("rank"), int)
        }
        if (
            len(completions) != case.rank_count
            or len(completion_evidence) != case.rank_count
            or {rank for _, rank in completion_evidence}
            != set(range(case.rank_count))
            or entry_evidence != completion_evidence
        ):
            raise RuntimeError(
                f"{variant} package terminal completion domain is invalid"
            )
        completion_evidence_by_variant[variant] = completion_evidence

    for relative in SOURCE_SNAPSHOT_PATHS:
        if (baseline / relative).read_bytes() != (winner / relative).read_bytes():
            raise RuntimeError(
                f"baseline/winner source snapshot differs at {relative}"
            )

    return (
        bindings_by_variant,
        output_ids_by_variant,
        completion_evidence_by_variant,
    )


def disassembled_functions(disassembly: str) -> dict[str, str]:
    functions = {
        match.group("name"): match.group("body")
        for match in re.finditer(
            (
                r"^[0-9a-fA-F]+ <(?P<name>[^>]+)>:\n"
                r"(?P<body>.*?)(?=^[0-9a-fA-F]+ <[^>]+>:\n|\Z)"
            ),
            disassembly,
            re.MULTILINE | re.DOTALL,
        )
    }
    if "main" not in functions:
        raise RuntimeError("target ELF does not contain a disassemblable main")
    return functions


def disassembled_instructions(body: str) -> list[tuple[str, str]]:
    instructions = [
        (match.group("mnemonic"), match.group("operands") or "")
        for match in re.finditer(
            (
                r"^\s*[0-9a-fA-F]+:\s+"
                r"(?:[0-9a-fA-F]+\s+)+"
                r"(?P<mnemonic>[^\s]+)"
                r"(?:\s+(?P<operands>.*?))?\s*$"
            ),
            body,
            re.MULTILINE,
        )
    ]
    if not instructions:
        raise RuntimeError("target function has no disassembled instructions")
    return instructions


def direct_target(operands: str) -> str | None:
    match = re.search(r"<(?P<target>[^>]+)>", operands)
    if match is None:
        return None
    return match.group("target").split("+", 1)[0]


def scheduler_function_symbols(
    functions: dict[str, str], symbol: str, symbols: set[str]
) -> None:
    if symbol in symbols:
        return
    body = functions.get(symbol)
    if body is None:
        raise RuntimeError(
            f"directly referenced scheduler function {symbol!r} is absent"
        )
    symbols.add(symbol)
    for mnemonic, operands in disassembled_instructions(body):
        if mnemonic not in ("j", "jal"):
            continue
        target = direct_target(operands)
        if (
            target is not None
            and not target.startswith("wafer_tx81_")
            and target != symbol
        ):
            scheduler_function_symbols(functions, target, symbols)


CONDITIONAL_BRANCH_MNEMONICS = frozenset(
    (
        "beq",
        "bne",
        "blt",
        "bge",
        "bltu",
        "bgeu",
        "beqz",
        "bnez",
        "blez",
        "bgez",
        "bltz",
        "bgtz",
    )
)


def straight_line_target_calls(
    functions: dict[str, str], symbol: str, stack: tuple[str, ...] = ()
) -> tuple[str, ...] | None:
    if symbol in stack:
        return None
    body = functions.get(symbol)
    if body is None:
        return None
    instructions = disassembled_instructions(body)
    calls: list[str] = []
    for index, (mnemonic, operands) in enumerate(instructions):
        if (
            mnemonic in CONDITIONAL_BRANCH_MNEMONICS
            or mnemonic.startswith("b")
            or mnemonic.startswith("c.b")
            or mnemonic in ("jalr", "jr")
        ):
            return None
        if mnemonic not in ("j", "jal"):
            continue
        target = direct_target(operands)
        if target is None or target == symbol:
            return None
        if target.startswith("wafer_tx81_"):
            calls.append(target)
        else:
            nested = straight_line_target_calls(
                functions, target, (*stack, symbol)
            )
            if nested is None:
                return None
            calls.extend(nested)
        if mnemonic == "j" and index != len(instructions) - 1:
            return None
    return tuple(calls)


def target_structure(
    package: pathlib.Path, objdump: pathlib.Path
) -> TargetStructure:
    manifest = json.loads((package / "manifest.json").read_text())
    workspace_bytes = sum(
        int(resource.get("bytes", 0))
        for resource in manifest.get("resources", [])
        if isinstance(resource, dict)
        and not resource.get("host_visible")
        and resource.get("role") == "workspace"
    )
    module_paths = {
        module.get("path")
        for module in manifest.get("modules", [])
        if isinstance(module, dict)
    }
    if not module_paths or not all(isinstance(path, str) for path in module_paths):
        raise RuntimeError("package has no target module paths")
    calls: list[str] = []
    scheduler_body_hashes: list[str] = []
    straight_line_calls: list[str] | None = []
    for relative in sorted(module_paths):
        result = run([str(objdump), "-d", str(package / relative)])
        functions = disassembled_functions(result.stdout)
        scheduler_symbols: set[str] = set()
        scheduler_function_symbols(functions, "main", scheduler_symbols)
        scheduler_bodies = [
            f"{symbol}\n{functions[symbol]}"
            for symbol in sorted(scheduler_symbols)
        ]
        for symbol in sorted(scheduler_symbols):
            for mnemonic, operands in disassembled_instructions(
                functions[symbol]
            ):
                target = (
                    direct_target(operands)
                    if mnemonic in ("j", "jal")
                    else None
                )
                if target is not None and target.startswith("wafer_tx81_"):
                    calls.append(target)
        module_straight_line_calls = straight_line_target_calls(
            functions, "main"
        )
        if straight_line_calls is not None:
            if module_straight_line_calls is None:
                straight_line_calls = None
            else:
                straight_line_calls.extend(module_straight_line_calls)
        scheduler_body_hashes.append(
            hashlib.sha256("\n".join(scheduler_bodies).encode()).hexdigest()
        )
    return TargetStructure(
        tuple(calls),
        tuple(scheduler_body_hashes),
        (
            tuple(straight_line_calls)
            if straight_line_calls is not None
            else None
        ),
        workspace_bytes,
    )


def module_digests(package: pathlib.Path) -> tuple[str, ...]:
    manifest = json.loads((package / "manifest.json").read_text())
    digests = tuple(
        module.get("digest")
        for module in manifest.get("modules", [])
        if isinstance(module, dict) and isinstance(module.get("digest"), str)
    )
    if not digests:
        raise RuntimeError("package manifest has no module digests")
    return digests


def floating_ulp_distance(
    lhs: np.generic, rhs: np.generic, element_bytes: int
) -> int:
    if lhs == rhs:
        return 0
    unsigned_dtype = np.dtype("<u2" if element_bytes == 2 else "<u4")
    lhs_bits = int(np.asarray(lhs).view(unsigned_dtype))
    rhs_bits = int(np.asarray(rhs).view(unsigned_dtype))
    sign = 1 << (element_bytes * 8 - 1)

    def ordered(bits: int) -> int:
        return sign - (bits & (sign - 1)) if bits & sign else sign + bits

    return abs(ordered(lhs_bits) - ordered(rhs_bits))


def validate_floating_pair(
    baseline: np.ndarray,
    winner: np.ndarray,
    spec: TensorSpec,
    policy: PairedOutputComparisonPolicy,
    location: tuple[int, str, int],
) -> None:
    if spec.mlir_dtype not in ("f16", "f32"):
        raise RuntimeError(
            f"floating comparison policy does not support {spec.mlir_dtype}"
        )
    baseline_flat = np.ascontiguousarray(baseline).reshape(-1)
    winner_flat = np.ascontiguousarray(winner).reshape(-1)
    nonfinite = np.flatnonzero(
        ~np.isfinite(baseline_flat) | ~np.isfinite(winner_flat)
    )
    if nonfinite.size:
        element = int(nonfinite[0])
        raise RuntimeError(
            "baseline/winner floating oracle contains NaN or infinity for "
            f"{location} at element {element}"
        )

    baseline_wide = baseline_flat.astype(np.float64)
    winner_wide = winner_flat.astype(np.float64)
    absolute_error = np.abs(baseline_wide - winner_wide)
    limit = policy.absolute_tolerance + (
        policy.relative_tolerance * np.abs(baseline_wide)
    )
    accepted = absolute_error <= limit
    if policy.maximum_ulp is not None:
        accepted &= np.fromiter(
            (
                floating_ulp_distance(lhs, rhs, spec.element_bytes)
                <= policy.maximum_ulp
                for lhs, rhs in zip(baseline_flat, winner_flat, strict=True)
            ),
            dtype=np.bool_,
            count=baseline_flat.size,
        )

    opposite_zero = (
        (baseline_flat == 0)
        & (winner_flat == 0)
        & (np.signbit(baseline_flat) != np.signbit(winner_flat))
    )
    if policy.signed_zero_equal:
        accepted |= opposite_zero
    else:
        accepted &= ~opposite_zero
    mismatching = np.flatnonzero(~accepted)
    if mismatching.size:
        element = int(mismatching[0])
        ulp = floating_ulp_distance(
            baseline_flat[element], winner_flat[element], spec.element_bytes
        )
        raise RuntimeError(
            "baseline/winner floating oracles exceed the numeric policy for "
            f"{location} at element {element}: "
            f"baseline={baseline_flat[element]!r} "
            f"winner={winner_flat[element]!r} "
            f"abs={absolute_error[element]!r} ulp={ulp}"
        )


def validate_paired_payloads(
    case: CampaignCase, payloads: PairedPayloads
) -> None:
    if (
        len(payloads.inputs) != case.rank_count
        or len(payloads.baseline_outputs) != case.rank_count
        or len(payloads.winner_outputs) != case.rank_count
    ):
        raise RuntimeError("payload factory rank domain is invalid")
    for rank in range(case.rank_count):
        if len(payloads.inputs[rank]) != len(case.inputs):
            raise RuntimeError(
                f"payload factory input tensor domain is invalid at rank {rank}"
            )
        for index, (array, spec) in enumerate(
            zip(payloads.inputs[rank], case.inputs, strict=True)
        ):
            if tuple(array.shape) != spec.shape or array.nbytes != (
                int(np.prod(spec.shape, dtype=np.int64)) * spec.element_bytes
            ):
                raise RuntimeError(
                    f"payload shape/bytes mismatch for "
                    f"{(rank, 'user_input', index)}"
                )
        baseline = payloads.baseline_outputs[rank]
        winner = payloads.winner_outputs[rank]
        if len(baseline) != len(case.outputs) or len(winner) != len(case.outputs):
            raise RuntimeError(
                f"payload factory output tensor domain is invalid at rank {rank}"
            )
        for index, (baseline_array, winner_array, spec) in enumerate(
            zip(baseline, winner, case.outputs, strict=True)
        ):
            expected_bytes = (
                int(np.prod(spec.shape, dtype=np.int64)) * spec.element_bytes
            )
            for variant, array in (
                ("baseline", baseline_array),
                ("winner", winner_array),
            ):
                if tuple(array.shape) != spec.shape or array.nbytes != expected_bytes:
                    raise RuntimeError(
                        f"{variant} payload shape/bytes mismatch for "
                        f"{(rank, 'output', index)}"
                    )
            location = (rank, "output", index)
            if (
                case.output_comparison.kind
                == PairedOutputComparisonPolicy.Kind.FloatingTolerance
            ):
                validate_floating_pair(
                    baseline_array,
                    winner_array,
                    spec,
                    case.output_comparison,
                    location,
                )
                continue
            baseline_elements = np.ascontiguousarray(baseline_array).view(
                np.uint8
            )
            winner_elements = np.ascontiguousarray(winner_array).view(
                np.uint8
            )
            if not np.array_equal(baseline_elements, winner_elements):
                differing_byte = int(
                    np.flatnonzero(baseline_elements != winner_elements)[0]
                )
                element = differing_byte // spec.element_bytes
                raise RuntimeError(
                    "baseline/winner host oracles are not bit-exact for "
                    f"{location} at element {element}"
                )


def write_payloads(
    work_dir: pathlib.Path,
    case: CampaignCase,
    bindings_by_variant: dict[str, dict[tuple[int, str, int], int]],
    payloads: PairedPayloads,
) -> dict[str, list[str]]:
    inputs = payloads.inputs
    outputs_by_variant = {
        "baseline": payloads.baseline_outputs,
        "winner": payloads.winner_outputs,
    }
    raw = work_dir / "raw"
    raw.mkdir()
    arguments_by_variant = {
        variant: [] for variant in bindings_by_variant
    }
    if set(bindings_by_variant) != set(outputs_by_variant):
        raise RuntimeError("paired payload variants must be baseline and winner")
    for rank in range(case.rank_count):
        for index, (array, spec) in enumerate(
            zip(inputs[rank], case.inputs, strict=True)
        ):
            path = raw / (
                f"rank_{rank:02d}_user_input_{index}.{spec.mlir_dtype}.raw"
            )
            path.write_bytes(np.ascontiguousarray(array).tobytes())
            for variant, bindings in bindings_by_variant.items():
                arguments_by_variant[variant].extend(
                    [
                        "--resource",
                        f"{bindings[(rank, 'user_input', index)]}={path}",
                    ]
                )
        for variant, outputs in outputs_by_variant.items():
            bindings = bindings_by_variant[variant]
            for index, (array, spec) in enumerate(
                zip(outputs[rank], case.outputs, strict=True)
            ):
                path = raw / (
                    f"{variant}_rank_{rank:02d}_output_{index}."
                    f"{spec.mlir_dtype}.raw"
                )
                path.write_bytes(np.ascontiguousarray(array).tobytes())
                expected_option = "--expected"
                if (
                    case.output_comparison.kind
                    == PairedOutputComparisonPolicy.Kind.FloatingTolerance
                ):
                    if spec.mlir_dtype != "f16":
                        raise RuntimeError(
                            "board relaxed output comparison currently "
                            "supports only f16"
                        )
                    expected_option = "--expected-f16-relaxed"
                arguments_by_variant[variant].extend(
                    [
                        expected_option,
                        f"{bindings[(rank, 'output', index)]}={path}",
                    ]
                )
    return arguments_by_variant


def no_card_command(
    wafer_run: pathlib.Path, package: pathlib.Path, case: CampaignCase
) -> list[str]:
    command = [str(wafer_run), "--package-dir", str(package)]
    if case.rank_count == 1:
        command.extend(["--entry-id", "0"])
    else:
        command.extend(
            [
                "--all-ranks",
                "--direct-dte-status-abi",
                DIRECT_DTE_STATUS_ABI,
                "--supports-host-watchdog",
            ]
        )
    command.append("--no-card")
    return command


def board_command(
    args: argparse.Namespace,
    package: pathlib.Path,
    case: CampaignCase,
    resource_arguments: list[str],
) -> list[str]:
    command = [str(args.wafer_run), "--package-dir", str(package)]
    if case.rank_count == 1:
        command.extend(["--entry-id", "0"])
    else:
        command.extend(["--all-ranks"])
    command.extend(
        [
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
    )
    return command


def verify_board_output(
    stdout: str,
    case: CampaignCase,
    output_ids: set[int],
    completion_evidence: set[tuple[int, int]],
) -> None:
    required = {
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "board_execution: true",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("paired board output omitted lifecycle evidence")
    exact_compare_matches = re.findall(
        r"^output_compare: resource=(\d+) bytes=\d+ exact=true$",
        stdout,
        re.MULTILINE,
    )
    relaxed_compare_matches = re.findall(
        r"^output_compare: resource=(\d+) bytes=\d+ "
        r"policy=f16-relaxed abs=([0-9.eE+-]+) rel=([0-9.eE+-]+) "
        r"max_ulp=(\d+) signed_zero_equal=true$",
        stdout,
        re.MULTILINE,
    )
    if (
        case.output_comparison.kind
        == PairedOutputComparisonPolicy.Kind.FloatingTolerance
    ):
        if exact_compare_matches:
            raise RuntimeError(
                "floating paired board output used raw exact comparison"
            )
        for _, absolute, relative, maximum_ulp in relaxed_compare_matches:
            if (
                float(absolute) != case.output_comparison.absolute_tolerance
                or float(relative)
                != case.output_comparison.relative_tolerance
                or int(maximum_ulp) != case.output_comparison.maximum_ulp
                or not case.output_comparison.signed_zero_equal
            ):
                raise RuntimeError(
                    "paired board output used the wrong floating policy"
                )
        compare_matches = [
            resource
            for resource, _, _, _ in relaxed_compare_matches
        ]
    else:
        if relaxed_compare_matches:
            raise RuntimeError(
                "raw-exact paired board output used relaxed comparison"
            )
        compare_matches = exact_compare_matches
    actual_output_ids = {int(resource) for resource in compare_matches}
    if (
        len(compare_matches) != len(output_ids)
        or actual_output_ids != output_ids
    ):
        raise RuntimeError(
            "paired board output compare rows differ: "
            f"expected={output_ids} actual={actual_output_ids} "
            f"rows={len(compare_matches)}"
        )
    if case.rank_count == 1:
        terminal_matches = re.findall(
            r"^terminal_completion: (\d+) kind=entry_return$",
            stdout,
            re.MULTILINE,
        )
        expected_completions = {
            completion
            for completion, rank in completion_evidence
            if rank == 0
        }
        if (
            len(terminal_matches) != 1
            or {int(completion) for completion in terminal_matches}
            != expected_completions
        ):
            raise RuntimeError(
                "rank-one board output omitted its exact terminal completion"
            )
    else:
        terminal_matches = re.findall(
            r"^terminal_completion: (\d+) kind=entry_return rank=(\d+)$",
            stdout,
            re.MULTILINE,
        )
        actual_completions = {
            (int(completion), int(rank))
            for completion, rank in terminal_matches
        }
        if (
            f"launch_pattern: {case.launch_contract['form']}-x"
            f"{case.rank_count}" not in stdout
            or f"logical_tile_domain: 0..{case.rank_count - 1}" not in stdout
            or len(terminal_matches) != case.rank_count
            or len(actual_completions) != case.rank_count
            or actual_completions != completion_evidence
        ):
            raise RuntimeError(
                "multi-rank board output omitted exact all-rank "
                "compare/completion evidence"
            )


def balanced_order(repeat: int) -> list[str]:
    order: list[str] = []
    for iteration in range(repeat):
        order.extend(
            ("baseline", "winner")
            if iteration % 2 == 0
            else ("winner", "baseline")
        )
    return order


def main() -> int:
    args = parse_args()
    case = CASES[args.case]
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("compiler optimization campaign hardware execution is not armed")
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
    if (
        not args.no_card
        and case.rank_count > 1
        and args.expected_tile_count is not None
        and args.expected_tile_count < case.rank_count
    ):
        raise RuntimeError(
            "multi-rank optimization case exceeds the qualified tile count"
        )

    payloads = case.payload_factory()
    validate_paired_payloads(case, payloads)
    source = write_source(args.work_dir, case)
    packages = {
        "baseline": args.work_dir / "baseline-package",
        "winner": args.work_dir / "winner-package",
    }
    compile_package(
        args.wafer_compile_test,
        source,
        packages["baseline"],
        case,
        reserved_baseline=True,
    )
    compile_package(
        args.wafer_compile,
        source,
        packages["winner"],
        case,
        reserved_baseline=False,
    )
    (
        bindings_by_variant,
        output_ids_by_variant,
        completion_evidence_by_variant,
    ) = validate_paired_packages(packages["baseline"], packages["winner"], case)
    structures = {
        name: target_structure(package, args.tx8_objdump)
        for name, package in packages.items()
    }
    case.structural_oracle(structures["baseline"], structures["winner"])
    identity = {
        "case": case.key,
        "source_snapshots_sha256": {
            str(relative): hashlib.sha256(
                (packages["winner"] / relative).read_bytes()
            ).hexdigest()
            for relative in SOURCE_SNAPSHOT_PATHS
        },
        "target_profile": TARGET_PROFILE,
        "launch": case.launch_contract,
        "rank_count": case.rank_count,
        "module_digests": {
            name: list(module_digests(package))
            for name, package in packages.items()
        },
    }
    structure_record = {
        "schema_version": 2,
        "identity": identity,
        "variants": {
            name: {
                "scheduler_body_sha256": list(
                    structure.scheduler_body_sha256
                ),
                "static_target_callsite_counts": dict(
                    sorted(structure.counts.items())
                ),
                "straight_line_call_sequence": (
                    list(structure.straight_line_calls)
                    if structure.straight_line_calls is not None
                    else None
                ),
                "workspace_bytes": structure.workspace_bytes,
            }
            for name, structure in structures.items()
        }
    }
    (args.work_dir / "target-structure.json").write_text(
        json.dumps(structure_record, indent=2, sort_keys=True) + "\n"
    )
    resource_arguments = write_payloads(
        args.work_dir, case, bindings_by_variant, payloads
    )

    if args.no_card:
        for name, package in packages.items():
            result = run(no_card_command(args.wafer_run, package, case))
            if "board_execution: false" not in result.stdout:
                raise RuntimeError(f"{name} no-card output omitted execution state")
        print(
            f"compiler_optimization_no_card: case={case.key} "
            "paired_packages=true target_structure_checked=true"
        )
        return 0

    rows: list[dict[str, object]] = []
    for sample, name in enumerate(balanced_order(args.repeat), start=1):
        command = board_command(
            args, packages[name], case, resource_arguments[name]
        )
        start_ns = time.monotonic_ns()
        result = run(
            command,
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        elapsed_ns = time.monotonic_ns() - start_ns
        verify_board_output(
            result.stdout,
            case,
            output_ids_by_variant[name],
            completion_evidence_by_variant[name],
        )
        row = {
            "case": case.key,
            "sample": sample,
            "variant": name,
            "host_process_elapsed_ns": elapsed_ns,
            "exact": True,
        }
        rows.append(row)
        print("compiler_optimization_sample: " + json.dumps(row, sort_keys=True))
        print(result.stdout, end="")
    (args.work_dir / "board-observations.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "qualification": {
                    "device_id": args.device_id,
                    "expected_runtime_version": args.expected_runtime_version,
                    "expected_device_name": args.expected_device_name,
                    "expected_pci_bus_id": args.expected_pci_bus_id,
                    "expected_tile_count": args.expected_tile_count,
                    "expected_runtime_library_sha256": (
                        args.expected_runtime_library_sha256
                    ),
                },
                "measurement": (
                    "host process elapsed observation; not PMU, device cycles, "
                    "or calibrated compiler cost"
                ),
                "samples": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
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
        print(
            f"wafer_board_compiler_optimization_campaign_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
