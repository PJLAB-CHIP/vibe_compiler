#!/usr/bin/env python3

import argparse
import pathlib

import numpy
import torch

import wafer_pytorch_xla_capture as capture


def _make_payloads(
    size: int, dtype: str
) -> tuple[numpy.ndarray, ...]:
    input_indices = numpy.arange(size * size, dtype=numpy.int32).reshape(size, size)
    weight_indices = (
        numpy.arange(size * size, dtype=numpy.int32).reshape(size, size) * 3
    )
    bias_indices = numpy.arange(size, dtype=numpy.int32)
    input_array = ((input_indices % 5) - 2).astype(numpy.float32) / 8.0
    weight_array = ((weight_indices % 5) - 2).astype(numpy.float32) / 8.0
    bias_array = ((bias_indices % 5) - 2).astype(numpy.float32) / 8.0

    input_storage = capture._cast_gemm_storage(numpy, input_array, dtype)
    weight_storage = capture._cast_gemm_storage(numpy, weight_array, dtype)
    bias_storage = capture._cast_gemm_storage(numpy, bias_array, dtype)
    input_values = capture._gemm_storage_to_float32(
        numpy, input_storage, dtype
    )
    weight_values = capture._gemm_storage_to_float32(
        numpy, weight_storage, dtype
    )
    bias_values = capture._gemm_storage_to_float32(
        numpy, bias_storage, dtype
    )
    expected = numpy.zeros((size, size), dtype=numpy.float32)
    for reduction in range(size):
        expected += (
            input_values[:, reduction : reduction + 1]
            * weight_values[reduction : reduction + 1, :]
        )
    expected += bias_values
    expected_storage = capture._cast_gemm_storage(numpy, expected, dtype)
    return input_storage, weight_storage, bias_storage, expected_storage


def _make_module_factory(
    weight_array: numpy.ndarray,
    bias_array: numpy.ndarray,
    dtype: str,
):
    class RowShardedMatmul(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                capture._torch_from_workload_storage(
                    torch, numpy, weight_array, dtype
                )
            )
            self.bias = torch.nn.Parameter(
                capture._torch_from_workload_storage(
                    torch, numpy, bias_array, dtype
                )
            )

        def forward(self, input_tensor):
            return input_tensor @ self.weight + self.bias

    return RowShardedMatmul


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a deterministic source-captured row-sharded matmul program "
            "and its independent CPU reference"
        )
    )
    parser.add_argument("--output-program-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-input", type=pathlib.Path, required=True)
    parser.add_argument("--output-expected", type=pathlib.Path, required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        required=True,
        help="low-precision source, parameter, input, and reference dtype",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.size <= 0 or args.size % 16 != 0:
        raise RuntimeError("--size must be a positive multiple of 16")

    input_array, weight_array, bias_array, expected = _make_payloads(
        args.size, args.dtype
    )
    module_factory = _make_module_factory(
        weight_array, bias_array, args.dtype
    )
    with torch.no_grad():
        framework_expected = capture._torch_to_workload_storage(
            torch,
            numpy,
            module_factory()(
                capture._torch_from_workload_storage(
                    torch, numpy, input_array, args.dtype
                )
            ),
            args.dtype,
        )
    if not numpy.array_equal(framework_expected, expected):
        raise RuntimeError(
            "framework output differs from the increasing-K CPU reference"
        )

    capture.emit_sharded_stablehlo_program(
        args.output_program_dir,
        strategy_name="row",
        reference_module_factory=module_factory,
        size=args.size,
    )
    args.output_input.parent.mkdir(parents=True, exist_ok=True)
    args.output_expected.parent.mkdir(parents=True, exist_ok=True)
    capture._save_workload_array(numpy, args.output_input, input_array)
    capture._save_workload_array(numpy, args.output_expected, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
