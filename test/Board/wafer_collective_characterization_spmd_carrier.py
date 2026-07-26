#!/usr/bin/env python3
"""Test-only explicit post-SPMD carrier for collective characterization."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-program-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-program-dir", type=pathlib.Path, required=True)
    parser.add_argument("--entry-function", required=True)
    parser.add_argument("--logical-rank-count", type=int, required=True)
    return parser.parse_args()


def tensor_signature(
    signature: dict[str, object],
) -> tuple[list[int], str]:
    shape = signature.get("shape")
    dtype = signature.get("dtype")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or not all(type(dimension) is int and dimension > 0 for dimension in shape)
        or not isinstance(dtype, str)
    ):
        raise RuntimeError("carrier requires a static one-dimensional signature")
    return shape, dtype


def replicated_binding(
    index_field: str,
    index: int,
    signature: dict[str, object],
    rank_count: int,
) -> dict[str, object]:
    shape, dtype = tensor_signature(signature)
    ranks = [
        {
            "rank": rank,
            "replica_id": rank,
            "offsets": [0] * len(shape),
            "sizes": shape,
            "strides": [1] * len(shape),
        }
        for rank in range(rank_count)
    ]
    return {
        index_field: index,
        "distribution": "replicated",
        "global_shape": shape,
        "local_shape": shape,
        "dtype": dtype,
        "ranks": ranks,
    }


def partitioned_binding(
    index_field: str,
    index: int,
    signature: dict[str, object],
    rank_count: int,
) -> dict[str, object]:
    local_shape, dtype = tensor_signature(signature)
    global_shape = [local_shape[0] * rank_count]
    ranks = [
        {
            "rank": rank,
            "replica_id": 0,
            "offsets": [rank * local_shape[0]],
            "sizes": local_shape,
            "strides": [1],
        }
        for rank in range(rank_count)
    ]
    return {
        index_field: index,
        "distribution": "partitioned",
        "global_shape": global_shape,
        "local_shape": local_shape,
        "dtype": dtype,
        "ranks": ranks,
    }


def collective_boundary(
    collective_marker: str,
    input_signature: dict[str, object],
    output_signature: dict[str, object],
    rank_count: int,
) -> dict[str, object]:
    input_shape, _ = tensor_signature(input_signature)
    output_shape, _ = tensor_signature(output_signature)
    if collective_marker == '"stablehlo.all_gather"':
        if output_shape[0] != input_shape[0] * rank_count:
            raise RuntimeError(
                "all-gather carrier shapes do not form one full rank group"
            )
        input_binding = partitioned_binding(
            "argument_index", 0, input_signature, rank_count
        )
        output_binding = replicated_binding(
            "result_index", 0, output_signature, rank_count
        )
    elif collective_marker == '"stablehlo.reduce_scatter"':
        if input_shape[0] != output_shape[0] * rank_count:
            raise RuntimeError(
                "reduce-scatter carrier shapes do not form one full rank group"
            )
        # The test-only carrier treats the sixteen complete rank-local
        # contributions as one partitioned outer batch. The sixteen distinct
        # result chunks then partition the reduced logical payload.
        input_binding = partitioned_binding(
            "argument_index", 0, input_signature, rank_count
        )
        output_binding = partitioned_binding(
            "result_index", 0, output_signature, rank_count
        )
    else:
        raise RuntimeError("unsupported collective carrier marker")
    return {
        "version": 1,
        "logical_rank_count": rank_count,
        "inputs": [input_binding],
        "outputs": [output_binding],
    }


def main() -> int:
    args = parse_args()
    if args.logical_rank_count != 16 or args.entry_function != "forward":
        raise RuntimeError(
            "collective characterization carrier requires forward on 16 ranks"
        )
    if args.output_program_dir.exists():
        raise RuntimeError("carrier output directory already exists")
    metadata_path = args.input_program_dir / "functions" / "forward.meta"
    module_path = args.input_program_dir / "functions" / "forward.mlir"
    metadata = json.loads(metadata_path.read_text())
    module = module_path.read_text()
    collective_markers = (
        '"stablehlo.all_gather"',
        '"stablehlo.reduce_scatter"',
    )
    present = [marker for marker in collective_markers if marker in module]
    if len(present) != 1:
        raise RuntimeError(
            "carrier requires exactly one explicit StableHLO AG or RS operation"
        )
    if "distributed_boundary" in metadata:
        raise RuntimeError("exporter metadata unexpectedly has a boundary")
    input_signatures = metadata.get("input_signature")
    output_signatures = metadata.get("output_signature")
    if (
        not isinstance(input_signatures, list)
        or not isinstance(output_signatures, list)
        or len(input_signatures) != 1
        or len(output_signatures) != 1
        or not all(
            isinstance(signature, dict)
            for signature in (*input_signatures, *output_signatures)
        )
    ):
        raise RuntimeError("carrier requires one input and one output")

    shutil.copytree(args.input_program_dir, args.output_program_dir)
    output_metadata = dict(metadata)
    output_metadata["distributed_boundary"] = collective_boundary(
        present[0],
        input_signatures[0],
        output_signatures[0],
        args.logical_rank_count,
    )
    (
        args.output_program_dir / "functions" / "forward.meta"
    ).write_text(json.dumps(output_metadata, separators=(",", ":")) + "\n")
    (
        args.output_program_dir
        / "functions"
        / "forward.parameter_shards.json"
    ).write_text(
        json.dumps(
            {
                "parameter_shards_version": 3,
                "function": "forward",
                "logical_rank_count": args.logical_rank_count,
                "parameters": [],
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"wafer_collective_characterization_spmd_carrier: {error}")
        raise SystemExit(1)
