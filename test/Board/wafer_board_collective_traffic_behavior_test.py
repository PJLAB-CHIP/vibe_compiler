#!/usr/bin/env python3
"""Validate current global sources for collective traffic semantics."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil

import numpy as np

from wafer_collective_traffic_behavior_catalog import (
    CASE_KEYS,
    CASES_BY_KEY,
    TILE_COUNT,
    CollectiveKind,
    TrafficBehaviorCase,
    row_major_manhattan_distance,
)


F16_DTYPE = np.dtype("<f2")
PAIR_BITS_BASE = np.uint16(0x4400)
LANE_BITS_BASE = np.uint16(0x0800)
TILE_BITS_BASE = np.uint16(0x4800)


@dataclasses.dataclass(frozen=True)
class TrafficPayloads:
    input: np.ndarray
    expected: np.ndarray


def tensor_type(shape: tuple[int, ...]) -> str:
    return "x".join(str(dimension) for dimension in shape) + "xf16"


def global_input_shape(case: TrafficBehaviorCase) -> tuple[int, ...]:
    if case.collective_kind == CollectiveKind.ALL_TO_ALL:
        return (TILE_COUNT, *case.input_shape)
    return (TILE_COUNT, *case.input_shape)


def global_output_shape(case: TrafficBehaviorCase) -> tuple[int, ...]:
    if case.collective_kind == CollectiveKind.ALL_TO_ALL:
        return (TILE_COUNT, *case.output_shape[1:])
    return (TILE_COUNT, *case.output_shape)


def all_to_all_module(case: TrafficBehaviorCase) -> str:
    lanes = case.input_shape[1]
    input_shape = global_input_shape(case)
    output_shape = global_output_shape(case)
    return f"""module {{
  func.func @main(%input: tensor<{tensor_type(input_shape)}>)
      -> tensor<{tensor_type(output_shape)}> {{
    %by_destination = "stablehlo.transpose"(%input) {{
      permutation = array<i64: 1, 0, 2, 3>
    }} : (tensor<{tensor_type(input_shape)}>)
      -> tensor<{TILE_COUNT}x{TILE_COUNT}x{lanes}x2xf16>
    %result = stablehlo.reshape %by_destination
      : (tensor<{TILE_COUNT}x{TILE_COUNT}x{lanes}x2xf16>)
        -> tensor<{tensor_type(output_shape)}>
    return %result : tensor<{tensor_type(output_shape)}>
  }}
}}
"""


def collective_permute_module(case: TrafficBehaviorCase) -> str:
    local_shape = case.input_shape
    global_shape = global_input_shape(case)
    local_type = tensor_type((1, *local_shape))
    global_type = tensor_type(global_shape)
    lines = [
        "module {",
        f"  func.func @main(%input: tensor<{global_type}>)",
        f"      -> tensor<{global_type}> {{",
        f"    %zero = stablehlo.constant dense<0.0> : tensor<{local_type}>",
    ]
    current = "%input"
    limits = ", ".join(str(dimension) for dimension in local_shape)
    strides = ", ".join("1" for _ in local_shape)
    for epoch_index, epoch in enumerate(case.epochs):
        source_by_target = {target: source for source, target in epoch}
        slices: list[str] = []
        for target in range(TILE_COUNT):
            source = source_by_target.get(target)
            if source is None:
                slices.append("%zero")
                continue
            name = f"%epoch_{epoch_index}_tile_{target}"
            start = ", ".join((str(source), *("0" for _ in local_shape)))
            limit = ", ".join((str(source + 1), limits))
            stride = ", ".join(("1", strides))
            lines.append(
                f"    {name} = stablehlo.slice {current} "
                f"[{start}] [{limit}] [{stride}] : tensor<{global_type}>"
            )
            slices.append(name)
        result = f"%epoch_{epoch_index}"
        lines.append(
            f"    {result} = stablehlo.concatenate "
            + ", ".join(slices)
            + f", dim = 0 : (tensor<{local_type}>) -> tensor<{global_type}>"
        )
        current = result
    lines.extend(
        (
            f"    return {current} : tensor<{global_type}>",
            "  }",
            "}",
            "",
        )
    )
    return "\n".join(lines)


def module_text(case: TrafficBehaviorCase) -> str:
    if case.collective_kind == CollectiveKind.ALL_TO_ALL:
        return all_to_all_module(case)
    if case.collective_kind == CollectiveKind.COLLECTIVE_PERMUTE:
        return collective_permute_module(case)
    raise RuntimeError(f"{case.key}: unsupported collective kind")


def all_to_all_payloads(case: TrafficBehaviorCase) -> TrafficPayloads:
    lanes_per_peer = case.input_shape[1]
    lanes = np.arange(lanes_per_peer, dtype=np.uint16)
    input_ = np.empty(global_input_shape(case), dtype=F16_DTYPE)
    bits = input_.view(np.uint16)
    for source in range(TILE_COUNT):
        for destination in range(TILE_COUNT):
            bits[source, destination, :, 0] = (
                PAIR_BITS_BASE + source * TILE_COUNT + destination
            )
            bits[source, destination, :, 1] = LANE_BITS_BASE + lanes
    expected = np.transpose(input_, (1, 0, 2, 3)).reshape(
        global_output_shape(case)
    )
    return TrafficPayloads(input_, expected)


def tile_payload(tile_id: int, payload_bytes: int) -> np.ndarray:
    lanes = np.arange(payload_bytes // 4, dtype=np.uint16)
    value = np.empty((payload_bytes // 4, 2), dtype=F16_DTYPE)
    bits = value.view(np.uint16)
    bits[:, 0] = TILE_BITS_BASE + tile_id
    bits[:, 1] = LANE_BITS_BASE + lanes
    return value


def simulate_permute_epochs(
    initial: np.ndarray, epochs: tuple[tuple[tuple[int, int], ...], ...]
) -> np.ndarray:
    current = initial
    for epoch in epochs:
        next_values = np.zeros_like(current)
        for source, target in epoch:
            next_values[target] = current[source]
        current = next_values
    return current


def payloads(case: TrafficBehaviorCase) -> TrafficPayloads:
    if case.collective_kind == CollectiveKind.ALL_TO_ALL:
        return all_to_all_payloads(case)
    input_ = np.stack(
        [tile_payload(tile_id, case.payload_bytes) for tile_id in range(TILE_COUNT)]
    )
    return TrafficPayloads(
        input_, simulate_permute_epochs(input_, case.epochs)
    )


def metadata(case: TrafficBehaviorCase) -> dict[str, object]:
    return {
        "name": "forward",
        "stablehlo_version": "0.0.0",
        "input_signature": [
            {"shape": list(global_input_shape(case)), "dtype": "float16",
             "dynamic_dims": []}
        ],
        "output_signature": [
            {"shape": list(global_output_shape(case)), "dtype": "float16",
             "dynamic_dims": []}
        ],
        "input_locations": [
            {"type_": "input_arg", "position": 0, "name": "input"}
        ],
        "unused_inputs": [],
    }


def validate_source(case: TrafficBehaviorCase) -> None:
    module = module_text(case)
    if any(
        marker in module
        for marker in ("mhlo.sharding", "@Sharding", "distributed_boundary")
    ):
        raise RuntimeError(f"{case.key}: source contains a retired interface")
    if "func.func @main" not in module or "stablehlo." not in module:
        raise RuntimeError(f"{case.key}: global source is incomplete")
    values = payloads(case)
    if values.input.shape != global_input_shape(case):
        raise RuntimeError(f"{case.key}: input oracle shape is invalid")
    if values.expected.shape != global_output_shape(case):
        raise RuntimeError(f"{case.key}: output oracle shape is invalid")
    if not np.all(np.isfinite(values.input)) or not np.all(
        np.isfinite(values.expected)
    ):
        raise RuntimeError(f"{case.key}: oracle contains nonfinite data")


def intended_graph_epochs(case: TrafficBehaviorCase) -> tuple[tuple[tuple[int, int], ...], ...]:
    if case.collective_kind == CollectiveKind.ALL_TO_ALL:
        return (
            tuple(
                (source, target)
                for source in range(TILE_COUNT)
                for target in range(TILE_COUNT)
                if source != target
            ),
        )
    return case.epochs


def intended_min_hop_summary(case: TrafficBehaviorCase) -> dict[str, int]:
    distances = [
        row_major_manhattan_distance(source, target)
        for epoch in intended_graph_epochs(case)
        for source, target in epoch
        if source != target
    ]
    return {
        "message_count": len(distances),
        "total_minimum_hops": sum(distances),
        "maximum_minimum_hops": max(distances, default=0),
    }


def write_source(work_dir: pathlib.Path, case: TrafficBehaviorCase) -> pathlib.Path:
    source = work_dir / "source-program"
    if source.exists():
        shutil.rmtree(source)
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(module_text(case))
    (source / "functions" / "forward.meta").write_text(
        json.dumps(metadata(case), separators=(",", ":")) + "\n"
    )
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_KEYS, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = CASES_BY_KEY[args.case]
    validate_source(case)
    write_source(args.work_dir, case)
    print(f"collective_traffic_source_contract: case={case.key} executable=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
