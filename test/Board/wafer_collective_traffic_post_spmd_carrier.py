#!/usr/bin/env python3
"""Test-only explicit post-SPMD carrier for collective traffic behavior cases."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from wafer_collective_traffic_behavior_catalog import PAYLOAD_POINTS


RANK_COUNT = 16
ENTRY_FUNCTION = "forward"
SUPPORTED_MARKERS = (
    '"stablehlo.all_to_all"',
    '"stablehlo.collective_permute"',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-program-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-program-dir", type=pathlib.Path, required=True)
    parser.add_argument("--entry-function", required=True)
    parser.add_argument("--logical-rank-count", type=int, required=True)
    return parser.parse_args()


def tensor_signature(signature: object) -> tuple[list[int], str]:
    if not isinstance(signature, dict):
        raise RuntimeError("carrier tensor signature must be an object")
    shape = signature.get("shape")
    dtype = signature.get("dtype")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(type(dimension) is int and dimension > 0 for dimension in shape)
        or not isinstance(dtype, str)
    ):
        raise RuntimeError("carrier requires a static ranked tensor signature")
    return shape, dtype


def replicated_binding(
    index_field: str,
    index: int,
    signature: object,
    rank_count: int,
) -> dict[str, object]:
    shape, dtype = tensor_signature(signature)
    return {
        index_field: index,
        "distribution": "replicated",
        "global_shape": shape,
        "local_shape": shape,
        "dtype": dtype,
        "ranks": [
            {
                "rank": rank,
                "replica_id": rank,
                "offsets": [0] * len(shape),
                "sizes": shape,
                "strides": [1] * len(shape),
            }
            for rank in range(rank_count)
        ],
    }


def require_supported_module(module: str) -> str:
    counts = {marker: module.count(marker) for marker in SUPPORTED_MARKERS}
    present = [marker for marker, count in counts.items() if count]
    if len(present) != 1:
        raise RuntimeError(
            "carrier requires exactly one collective traffic operation kind"
        )
    marker = present[0]
    expected_count = (
        1 if marker == '"stablehlo.all_to_all"' else counts[marker]
    )
    if (
        marker == '"stablehlo.all_to_all"'
        and counts[marker] != 1
    ) or (
        marker == '"stablehlo.collective_permute"'
        and expected_count not in (1, 2)
    ):
        raise RuntimeError(
            "carrier requires one AllToAll or one/two CollectivePermute epochs"
        )
    return marker


def main() -> int:
    args = parse_args()
    if (
        args.logical_rank_count != RANK_COUNT
        or args.entry_function != ENTRY_FUNCTION
    ):
        raise RuntimeError(
            "collective traffic carrier requires forward on exactly 16 ranks"
        )
    if args.output_program_dir.exists():
        raise RuntimeError("carrier output directory already exists")

    metadata_path = args.input_program_dir / "functions" / "forward.meta"
    module_path = args.input_program_dir / "functions" / "forward.mlir"
    metadata = json.loads(metadata_path.read_text())
    module = module_path.read_text()
    require_supported_module(module)
    if "distributed_boundary" in metadata:
        raise RuntimeError("exporter metadata unexpectedly has a boundary")

    input_signatures = metadata.get("input_signature")
    output_signatures = metadata.get("output_signature")
    if (
        not isinstance(input_signatures, list)
        or not isinstance(output_signatures, list)
        or len(input_signatures) != 1
        or len(output_signatures) != 1
    ):
        raise RuntimeError("carrier requires one input and one output")
    input_shape, input_dtype = tensor_signature(input_signatures[0])
    output_shape, output_dtype = tensor_signature(output_signatures[0])
    input_bytes = 2
    for dimension in input_shape:
        input_bytes *= dimension
    output_bytes = 2
    for dimension in output_shape:
        output_bytes *= dimension
    if input_dtype != "float16" or output_dtype != "float16":
        raise RuntimeError("carrier requires float16 traffic payloads")
    if input_bytes != output_bytes or input_bytes not in PAYLOAD_POINTS:
        raise RuntimeError(
            "carrier requires equal 256/4096/65536-byte local payloads"
        )

    shutil.copytree(args.input_program_dir, args.output_program_dir)
    output_metadata = dict(metadata)
    output_metadata["distributed_boundary"] = {
        "version": 1,
        "logical_rank_count": RANK_COUNT,
        "inputs": [
            replicated_binding(
                "argument_index",
                0,
                input_signatures[0],
                RANK_COUNT,
            )
        ],
        "outputs": [
            replicated_binding(
                "result_index",
                0,
                output_signatures[0],
                RANK_COUNT,
            )
        ],
    }
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
                "function": ENTRY_FUNCTION,
                "logical_rank_count": RANK_COUNT,
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
        print(f"wafer_collective_traffic_post_spmd_carrier: {error}")
        raise SystemExit(1)
