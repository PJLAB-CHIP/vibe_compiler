#!/usr/bin/env python3
"""Validate global sources and host oracles for collective comparisons.

Direct/ring/tree alternatives remain catalogued structural hypotheses.  The
current compiler does not expose a test-only algorithm selector, so this file
does not manufacture executable package comparisons from retired SPMD helpers.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import shutil

import numpy as np

from wafer_collective_hardware_characterization_catalog import (
    CASE_KEYS,
    CASES_BY_KEY,
    CollectiveAlternative,
    CollectiveCharacterizationCase,
    CollectiveKind,
)


TILE_COUNT = 16


@dataclasses.dataclass(frozen=True)
class CollectivePayloads:
    input: np.ndarray
    expected: np.ndarray
    tile_contributions: tuple[np.ndarray, ...]


def finite_tile_values(tile_id: int, elements: int) -> np.ndarray:
    lanes = np.arange(elements, dtype=np.int64)
    values = ((lanes * 13 + tile_id * 7 + (lanes >> 3)) % 17) - 8
    values[(lanes + tile_id) % 29 == 0] += 1
    return values.astype("<f2")


def payloads(case: CollectiveCharacterizationCase) -> CollectivePayloads:
    elements = case.payload_bytes // np.dtype("<f2").itemsize
    if elements % TILE_COUNT:
        raise RuntimeError(f"{case.key}: payload is not Tile divisible")
    chunk_elements = elements // TILE_COUNT
    if case.collective_kind == CollectiveKind.ALL_GATHER:
        contributions = tuple(
            finite_tile_values(tile_id, chunk_elements)
            for tile_id in range(TILE_COUNT)
        )
        input_ = np.stack(contributions)
        gathered = input_.reshape(elements)
        expected = np.broadcast_to(
            gathered, (TILE_COUNT, elements)
        ).copy()
    else:
        contributions = tuple(
            finite_tile_values(tile_id, elements)
            for tile_id in range(TILE_COUNT)
        )
        input_ = np.stack(contributions)
        reduced = np.sum(input_.astype(np.int32), axis=0).astype("<f2")
        if case.collective_kind == CollectiveKind.REDUCE_SCATTER:
            expected = reduced.reshape(TILE_COUNT, chunk_elements)
        elif case.collective_kind == CollectiveKind.ALL_REDUCE:
            expected = np.broadcast_to(
                reduced, (TILE_COUNT, elements)
            ).copy()
        else:
            raise RuntimeError(
                f"{case.key}: unsupported collective kind"
            )
    if not np.all(np.isfinite(input_)) or not np.all(np.isfinite(expected)):
        raise RuntimeError(f"{case.key}: host oracle contains nonfinite data")
    if len({value.tobytes() for value in contributions}) != TILE_COUNT:
        raise RuntimeError(f"{case.key}: Tile contributions are not distinct")
    return CollectivePayloads(input_, expected, contributions)


def reduction_body() -> str:
    return """    %zero = stablehlo.constant dense<0.0> : tensor<f16>
    %reduced = "stablehlo.reduce"(%input, %zero) ({
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f16>
      stablehlo.return %sum : tensor<f16>
    }) {dimensions = array<i64: 0>}"""


def module_text(case: CollectiveCharacterizationCase) -> str:
    elements = case.payload_bytes // 2
    chunk = elements // TILE_COUNT
    if case.collective_kind == CollectiveKind.ALL_GATHER:
        return f"""module {{
  func.func @main(%input: tensor<{TILE_COUNT}x{chunk}xf16>)
      -> tensor<{TILE_COUNT}x{elements}xf16> {{
    %flat = stablehlo.reshape %input
      : (tensor<{TILE_COUNT}x{chunk}xf16>) -> tensor<{elements}xf16>
    %result = "stablehlo.broadcast_in_dim"(%flat) {{
      broadcast_dimensions = array<i64: 1>
    }} : (tensor<{elements}xf16>) -> tensor<{TILE_COUNT}x{elements}xf16>
    return %result : tensor<{TILE_COUNT}x{elements}xf16>
  }}
}}
"""
    if case.collective_kind == CollectiveKind.REDUCE_SCATTER:
        return f"""module {{
  func.func @main(%input: tensor<{TILE_COUNT}x{elements}xf16>)
      -> tensor<{TILE_COUNT}x{chunk}xf16> {{
{reduction_body()}
      : (tensor<{TILE_COUNT}x{elements}xf16>, tensor<f16>)
        -> tensor<{elements}xf16>
    %result = stablehlo.reshape %reduced
      : (tensor<{elements}xf16>) -> tensor<{TILE_COUNT}x{chunk}xf16>
    return %result : tensor<{TILE_COUNT}x{chunk}xf16>
  }}
}}
"""
    if case.collective_kind == CollectiveKind.ALL_REDUCE:
        return f"""module {{
  func.func @main(%input: tensor<{TILE_COUNT}x{elements}xf16>)
      -> tensor<{TILE_COUNT}x{elements}xf16> {{
{reduction_body()}
      : (tensor<{TILE_COUNT}x{elements}xf16>, tensor<f16>)
        -> tensor<{elements}xf16>
    %result = "stablehlo.broadcast_in_dim"(%reduced) {{
      broadcast_dimensions = array<i64: 1>
    }} : (tensor<{elements}xf16>) -> tensor<{TILE_COUNT}x{elements}xf16>
    return %result : tensor<{TILE_COUNT}x{elements}xf16>
  }}
}}
"""
    raise RuntimeError(f"{case.key}: unsupported collective kind")


def metadata(case: CollectiveCharacterizationCase) -> dict[str, object]:
    values = payloads(case)
    return {
        "name": "forward",
        "stablehlo_version": "0.0.0",
        "input_signature": [
            {"shape": list(values.input.shape), "dtype": "float16",
             "dynamic_dims": []}
        ],
        "output_signature": [
            {"shape": list(values.expected.shape), "dtype": "float16",
             "dynamic_dims": []}
        ],
        "input_locations": [
            {"type_": "input_arg", "position": 0, "name": "input"}
        ],
        "unused_inputs": [],
    }


def validate_source(case: CollectiveCharacterizationCase) -> None:
    module = module_text(case)
    if any(
        marker in module
        for marker in ("mhlo.sharding", "@Sharding", "distributed_boundary")
    ):
        raise RuntimeError(f"{case.key}: source contains a retired interface")
    if "func.func @main" not in module or "stablehlo." not in module:
        raise RuntimeError(f"{case.key}: source program is incomplete")
    values = payloads(case)
    if tuple(metadata(case)["input_signature"][0]["shape"]) != values.input.shape:
        raise RuntimeError(f"{case.key}: input metadata shape is invalid")


def require_cross_tile_message_matching(
    rows: dict[int, dict[str, object]], description: str
) -> None:
    sends: collections.Counter[tuple[object, ...]] = collections.Counter()
    receives: collections.Counter[tuple[object, ...]] = collections.Counter()
    for tile_id, row in rows.items():
        for message in row.get("messages", []):
            if not isinstance(message, dict):
                raise RuntimeError(f"{description}: message is not a record")
            direction = message.get("direction")
            peer = message.get("peer")
            key = (
                tile_id if direction == "send" else peer,
                peer if direction == "send" else tile_id,
                message.get("communication_id"), message.get("phase"),
                message.get("round"), message.get("payload_slice"),
                message.get("executed_bytes"),
            )
            if direction == "send":
                sends[key] += 1
            elif direction == "recv":
                receives[key] += 1
            else:
                raise RuntimeError(f"{description}: invalid direction")
    if sends != receives:
        raise RuntimeError(f"{description}: send/receive tuples do not match")


def require_single_cycle(successor: dict[int, int], description: str) -> None:
    if set(successor) != set(range(TILE_COUNT)) or set(successor.values()) != set(
        range(TILE_COUNT)
    ):
        raise RuntimeError(f"{description}: successor domain is incomplete")
    visited: set[int] = set()
    current = 0
    while current not in visited:
        visited.add(current)
        current = successor[current]
    if current != 0 or len(visited) != TILE_COUNT:
        raise RuntimeError(f"{description}: successor relation is not one cycle")


def require_ring(
    rows: dict[int, dict[str, object]], alternative: str,
    round_count: int, chunk_bytes: int,
) -> None:
    for tile_id in range(TILE_COUNT):
        messages = rows[tile_id]["messages"]
        sends = [m for m in messages if m["direction"] == "send"]
        receives = [m for m in messages if m["direction"] == "recv"]
        if len(sends) != round_count or len(receives) != round_count:
            raise RuntimeError(f"{alternative}: ring round count is invalid")
        if {m["round"] for m in sends} != set(range(round_count)):
            raise RuntimeError(f"{alternative}: send rounds are invalid")
        if {m["round"] for m in receives} != set(range(round_count)):
            raise RuntimeError(f"{alternative}: receive rounds are invalid")
        if any(m["executed_bytes"] != chunk_bytes for m in messages):
            raise RuntimeError(f"{alternative}: ring chunk size is invalid")
        if {m["peer"] for m in sends} != {(tile_id + 1) % TILE_COUNT}:
            raise RuntimeError(f"{alternative}: send neighbor is invalid")
        if {m["peer"] for m in receives} != {(tile_id - 1) % TILE_COUNT}:
            raise RuntimeError(f"{alternative}: receive neighbor is invalid")


def require_ordered_tree(
    rows: dict[int, dict[str, object]], alternative: str, payload_bytes: int
) -> None:
    require_cross_tile_message_matching(rows, alternative)
    reduce_edges: set[tuple[int, int]] = set()
    broadcast_edges: set[tuple[int, int]] = set()
    for tile_id, row in rows.items():
        for message in row["messages"]:
            if message["direction"] != "send":
                continue
            edge = (tile_id, message["peer"])
            if message["phase"] == "all_reduce_tree_reduce":
                reduce_edges.add(edge)
            elif message["phase"] == "all_reduce_tree_broadcast":
                broadcast_edges.add(edge)
            if message["executed_bytes"] != payload_bytes:
                raise RuntimeError(f"{alternative}: tree payload size is invalid")
    if len(reduce_edges) != TILE_COUNT - 1:
        raise RuntimeError(f"{alternative}: reduce tree is incomplete")
    if broadcast_edges != {(parent, child) for child, parent in reduce_edges}:
        raise RuntimeError(f"{alternative}: broadcast tree is not the reverse tree")


def write_source(work_dir: pathlib.Path, case: CollectiveCharacterizationCase) -> pathlib.Path:
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
    alternatives = (case.left_alternative.value, case.right_alternative.value)
    if any(
        alternative not in {value.value for value in CollectiveAlternative}
        for alternative in alternatives
    ):
        raise RuntimeError(f"{case.key}: algorithm comparison is invalid")
    print(
        "collective_source_contract: "
        f"case={case.key} alternatives={alternatives[0]},{alternatives[1]} "
        "executable=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
