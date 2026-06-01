#!/usr/bin/env python3
"""Temporary PyTorch/XLA SPMD oracle for Wafer tests."""

from __future__ import annotations

import argparse
import pathlib
import sys

from wafer_pytorch_xla_capture import (
    DEFAULT_REFERENCE_MATMUL_SIZE,
    SHARDING_STRATEGY_NAMES,
    emit_partitioned_stablehlo_bundle,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sharding-strategy",
        choices=SHARDING_STRATEGY_NAMES,
        help="user sharding strategy",
    )
    parser.add_argument(
        "--default-input-sharding",
        action="store_true",
        help="apply the no-user-sharding default input seed policy",
    )
    parser.add_argument(
        "--default-tile-count",
        type=int,
        default=16,
        help="tile count for --default-input-sharding",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_REFERENCE_MATMUL_SIZE,
        help="square matmul size for generated reference artifacts",
    )
    parser.add_argument("--output-bundle", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    emit_partitioned_stablehlo_bundle(
        args.output_bundle,
        strategy_name=args.sharding_strategy,
        default_input_sharding=args.default_input_sharding,
        default_tile_count=args.default_tile_count,
        size=args.size,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-spmd-oracle: {error}", file=sys.stderr)
        raise SystemExit(1)
