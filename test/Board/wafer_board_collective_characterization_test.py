#!/usr/bin/env python3
"""Compile and run same-source forced collective implementation board pairs."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

import numpy as np

import wafer_board_compiler_optimization_campaign_test as paired_support
import wafer_direct_dte_board_evidence as direct_dte_evidence
from wafer_collective_hardware_characterization_catalog import (
    CASE_KEYS,
    CASES_BY_KEY,
    CollectiveAlternative,
    CollectiveCharacterizationCase,
    CollectiveKind,
)


TARGET_PROFILE = paired_support.TARGET_PROFILE
LAUNCH_ABI = paired_support.CLUSTER_LAUNCH_ABI
STATUS_ABI = paired_support.DIRECT_DTE_STATUS_ABI
RANK_COUNT = 16
ALTERNATIVE_ENVIRONMENT_VARIABLE = (
    "WAFER_TEST_COLLECTIVE_CHARACTERIZATION_ALTERNATIVE"
)
REPORT_ENVIRONMENT_VARIABLE = "WAFER_TEST_COLLECTIVE_CHARACTERIZATION_REPORT"
SPMD_HELPER_ENVIRONMENT_VARIABLE = "WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER"
SOURCE_SNAPSHOT_PATHS = paired_support.SOURCE_SNAPSHOT_PATHS
PROCESS_TIMEOUT_MARGIN_SECONDS = paired_support.PROCESS_TIMEOUT_MARGIN_SECONDS
POST_SPMD_CARRIER = pathlib.Path(__file__).with_name(
    "wafer_collective_characterization_spmd_carrier.py"
)


@dataclasses.dataclass(frozen=True)
class RuntimeCase:
    contract: CollectiveCharacterizationCase
    inputs: tuple[paired_support.TensorSpec, ...]
    outputs: tuple[paired_support.TensorSpec, ...]

    @property
    def key(self) -> str:
        return self.contract.key

    @property
    def rank_count(self) -> int:
        return self.contract.rank_count

    @property
    def launch_abi(self) -> str:
        return LAUNCH_ABI

    @property
    def payload_factory(self):
        """Adapter for the shared package/runtime payload writer."""

        return lambda: payloads(self)


def runtime_case(contract: CollectiveCharacterizationCase) -> RuntimeCase:
    payload_bytes = contract.payload_bytes
    chunk_bytes = payload_bytes // RANK_COUNT
    if payload_bytes % RANK_COUNT != 0:
        raise RuntimeError("collective payload is not divisible by rank count")
    if contract.collective_kind == CollectiveKind.ALL_GATHER:
        inputs = (paired_support.TensorSpec((chunk_bytes,), "i8", "int8"),)
        outputs = (paired_support.TensorSpec((payload_bytes,), "i8", "int8"),)
    elif contract.collective_kind == CollectiveKind.REDUCE_SCATTER:
        inputs = (paired_support.TensorSpec((payload_bytes,), "i8", "int8"),)
        outputs = (paired_support.TensorSpec((chunk_bytes,), "i8", "int8"),)
    elif contract.collective_kind == CollectiveKind.ALL_REDUCE:
        inputs = (
            paired_support.TensorSpec(
                (1, payload_bytes),
                "i8",
                "int8",
                source_shape=(RANK_COUNT, payload_bytes),
            ),
        )
        outputs = (
            paired_support.TensorSpec((payload_bytes,), "i8", "int8"),
        )
    else:
        raise RuntimeError(f"unsupported collective kind {contract.collective_kind}")
    return RuntimeCase(contract, inputs, outputs)


def rank_group() -> str:
    return ", ".join(str(rank) for rank in range(RANK_COUNT))


def all_gather_module(payload_bytes: int) -> str:
    chunk_bytes = payload_bytes // RANK_COUNT
    return f"""\
module {{
  func.func @main(%input: tensor<{chunk_bytes}xi8>) -> tensor<{payload_bytes}xi8> {{
    %result = "stablehlo.all_gather"(%input) {{
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[{rank_group()}]]> : tensor<1x16xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 47, type = 1>,
      use_global_device_ids
    }} : (tensor<{chunk_bytes}xi8>) -> tensor<{payload_bytes}xi8>
    return %result : tensor<{payload_bytes}xi8>
  }}
}}
"""


def reduce_scatter_module(payload_bytes: int) -> str:
    chunk_bytes = payload_bytes // RANK_COUNT
    return f"""\
module {{
  func.func @main(%input: tensor<{payload_bytes}xi8>) -> tensor<{chunk_bytes}xi8> {{
    %result = "stablehlo.reduce_scatter"(%input) ({{
    ^bb0(%lhs: tensor<i8>, %rhs: tensor<i8>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i8>
      "stablehlo.return"(%sum) : (tensor<i8>) -> ()
    }}) {{
      scatter_dimension = 0 : i64,
      replica_groups = dense<[[{rank_group()}]]> : tensor<1x16xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 53, type = 1>,
      use_global_device_ids
    }} : (tensor<{payload_bytes}xi8>) -> tensor<{chunk_bytes}xi8>
    return %result : tensor<{chunk_bytes}xi8>
  }}
}}
"""


def all_reduce_module(payload_bytes: int) -> str:
    # At 4096 bytes this intentionally remains source-compatible with the
    # existing tree-all-reduce optimizer qualification case.
    devices = ",".join(str(rank) for rank in range(RANK_COUNT))
    return f"""\
module {{
  func.func @main(
      %input: tensor<16x{payload_bytes}xi8>) -> tensor<{payload_bytes}xi8> {{
    %sharded = stablehlo.custom_call @Sharding(%input) {{
      backend_config = "",
      mhlo.sharding = "{{devices=[16,1]{devices}}}"
    }} : (tensor<16x{payload_bytes}xi8>) -> tensor<16x{payload_bytes}xi8>
    %zero = stablehlo.constant dense<0> : tensor<i8>
    %result = "stablehlo.reduce"(%sharded, %zero) ({{
    ^bb0(%lhs: tensor<i8>, %rhs: tensor<i8>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i8>
      stablehlo.return %sum : tensor<i8>
    }}) {{dimensions = array<i64: 0>}}
      : (tensor<16x{payload_bytes}xi8>, tensor<i8>)
        -> tensor<{payload_bytes}xi8>
    return %result : tensor<{payload_bytes}xi8>
  }}
}}
"""


def module_text(case: RuntimeCase) -> str:
    kind = case.contract.collective_kind
    payload_bytes = case.contract.payload_bytes
    if kind == CollectiveKind.ALL_GATHER:
        return all_gather_module(payload_bytes)
    if kind == CollectiveKind.REDUCE_SCATTER:
        return reduce_scatter_module(payload_bytes)
    if kind == CollectiveKind.ALL_REDUCE:
        return all_reduce_module(payload_bytes)
    raise RuntimeError(f"unsupported collective kind {kind}")


def sentinel_bytes(
    rank: int, logical_lanes: np.ndarray, payload_bytes: int
) -> np.ndarray:
    """Return a deterministic rank/lane hash with no intentional short period."""

    mask = (1 << 64) - 1
    rank_seed = ((rank + 1) * 0xD1B54A32D192ED03) & mask
    payload_seed = (payload_bytes * 0x9E3779B97F4A7C15) & mask
    value = logical_lanes.astype(np.uint64, copy=True)
    value ^= np.uint64(rank_seed)
    value ^= np.uint64(payload_seed)
    value += np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    value = (value ^ (value >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    value ^= value >> np.uint64(31)
    folded = value ^ (value >> np.uint64(32))
    folded ^= folded >> np.uint64(16)
    folded ^= folded >> np.uint64(8)
    return (folded & np.uint64(0xFF)).astype(np.uint8).view(np.int8)


def payloads(
    case: RuntimeCase,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    payload_bytes = case.contract.payload_bytes
    kind = case.contract.collective_kind
    if kind == CollectiveKind.ALL_GATHER:
        chunk_bytes = payload_bytes // RANK_COUNT
        local_inputs = [
            sentinel_bytes(
                rank,
                np.arange(
                    rank * chunk_bytes,
                    (rank + 1) * chunk_bytes,
                    dtype=np.uint64,
                ),
                payload_bytes,
            )
            for rank in range(RANK_COUNT)
        ]
        expected = np.concatenate(local_inputs)
        return (
            [[input_] for input_ in local_inputs],
            [[expected] for _ in range(RANK_COUNT)],
        )
    lanes = np.arange(payload_bytes, dtype=np.uint64)
    rank_inputs = [
        sentinel_bytes(rank, lanes, payload_bytes)
        for rank in range(RANK_COUNT)
    ]
    if kind == CollectiveKind.REDUCE_SCATTER:
        chunk_bytes = payload_bytes // RANK_COUNT
        outputs: list[list[np.ndarray]] = []
        for destination in range(RANK_COUNT):
            total = np.zeros(chunk_bytes, dtype=np.int64)
            begin = destination * chunk_bytes
            end = begin + chunk_bytes
            for rank_input in rank_inputs:
                total += rank_input[begin:end].astype(np.int64)
            outputs.append(
                [(total & 0xFF).astype(np.uint8).view(np.int8)]
            )
        return [[input_] for input_ in rank_inputs], outputs
    if kind == CollectiveKind.ALL_REDUCE:
        total = np.zeros(payload_bytes, dtype=np.int64)
        for rank_input in rank_inputs:
            total += rank_input.astype(np.int64)
        expected = (total & 0xFF).astype(np.uint8).view(np.int8)
        return (
            [[input_[None, :]] for input_ in rank_inputs],
            [[expected] for _ in range(RANK_COUNT)],
        )
    raise RuntimeError(f"unsupported collective kind {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-compile-test", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--tx8-objdump", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--case", choices=CASE_KEYS, required=True)
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


def prepare_work_dir(work_dir: pathlib.Path, alternatives: tuple[str, str]) -> None:
    known_children = {
        "source-program",
        "raw",
        "target-structure.json",
        "board-observations.json",
        "collective-characterization.json",
    }
    for alternative in alternatives:
        known_children.add(f"{alternative}-package")
        known_children.add(f"{alternative}-compiler-report.json")
    work_dir.mkdir(parents=True, exist_ok=True)
    unknown = [
        child.name for child in work_dir.iterdir() if child.name not in known_children
    ]
    if unknown:
        raise RuntimeError(
            "--work-dir contains unknown entries; refusing cleanup: "
            + ", ".join(sorted(unknown))
        )
    for name in sorted(known_children):
        child = work_dir / name
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def write_source(work_dir: pathlib.Path, case: RuntimeCase) -> pathlib.Path:
    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(module_text(case))
    (source / "functions" / "forward.meta").write_text(
        json.dumps(paired_support.metadata(case), separators=(",", ":")) + "\n"
    )
    return source


def compile_package(
    compiler: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    report_path: pathlib.Path,
    case: RuntimeCase,
    alternative: str,
) -> None:
    environment = os.environ.copy()
    for name in (
        paired_support.BASELINE_ENVIRONMENT_VARIABLE,
        "WAFER_TEST_FAIL_AFTER_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_TARGET_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_PACKAGE_LOGICAL_RANK",
        ALTERNATIVE_ENVIRONMENT_VARIABLE,
        REPORT_ENVIRONMENT_VARIABLE,
        SPMD_HELPER_ENVIRONMENT_VARIABLE,
    ):
        environment.pop(name, None)
    environment[ALTERNATIVE_ENVIRONMENT_VARIABLE] = alternative
    environment[REPORT_ENVIRONMENT_VARIABLE] = str(report_path)
    if case.contract.collective_kind in (
        CollectiveKind.ALL_GATHER,
        CollectiveKind.REDUCE_SCATTER,
    ):
        environment[SPMD_HELPER_ENVIRONMENT_VARIABLE] = str(POST_SPMD_CARRIER)
    result = paired_support.run(
        [
            str(compiler),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(output),
            f"--execution-ranks={RANK_COUNT}",
            f"--target-profile={TARGET_PROFILE}",
            f"--launch-abi={LAUNCH_ABI}",
        ],
        environment=environment,
    )
    expected = (
        "wafer-compile: published verified package with "
        f"execution-ranks={RANK_COUNT}"
    )
    if expected not in result.stdout or not output.is_dir():
        raise RuntimeError(f"compiler did not publish {alternative}")
    if not report_path.is_file():
        raise RuntimeError(
            f"{alternative} compiler omitted its characterization report"
        )


def require_sorted_unique(values: object, field: str, value_type: type) -> list:
    if (
        not isinstance(values, list)
        or not all(type(value) is value_type for value in values)
        or values != sorted(set(values))
    ):
        raise RuntimeError(f"compiler report {field} is not sorted and unique")
    return values


MESSAGE_FIELDS = frozenset(
    {
        "direction",
        "peer",
        "communication_id",
        "phase",
        "round",
        "payload_slice",
        "issue_bytes",
        "constant_loop_multiplicity",
        "executed_bytes",
    }
)


def message_sort_key(message: dict[str, object]) -> tuple[object, ...]:
    return (
        message["direction"],
        message["peer"],
        message["communication_id"],
        message["phase"],
        message["round"],
        message["payload_slice"],
        message["issue_bytes"],
        message["constant_loop_multiplicity"],
    )


def validate_rank_messages(
    row: dict[str, object],
    rank: int,
    alternative: str,
    expected_phases: frozenset[str],
) -> list[dict[str, object]]:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise RuntimeError(f"{alternative} rank {rank} has no message tuples")
    messages: list[dict[str, object]] = []
    for raw_message in raw_messages:
        if (
            not isinstance(raw_message, dict)
            or set(raw_message) != MESSAGE_FIELDS
            or raw_message.get("direction") not in ("send", "recv")
            or type(raw_message.get("phase")) is not str
            or any(
                type(raw_message.get(field)) is not int
                for field in (
                    "peer",
                    "communication_id",
                    "round",
                    "payload_slice",
                    "issue_bytes",
                    "constant_loop_multiplicity",
                    "executed_bytes",
                )
            )
        ):
            raise RuntimeError(
                f"{alternative} rank {rank} has an invalid message tuple"
            )
        message = dict(raw_message)
        if (
            message["phase"] not in expected_phases
            or not 0 <= message["peer"] < RANK_COUNT
            or message["peer"] == rank
            or message["communication_id"] < 0
            or message["round"] < 0
            or message["payload_slice"] < 0
            or message["issue_bytes"] <= 0
            or message["constant_loop_multiplicity"] <= 0
            or message["executed_bytes"]
            != message["issue_bytes"] * message["constant_loop_multiplicity"]
        ):
            raise RuntimeError(
                f"{alternative} rank {rank} message tuple is out of domain"
            )
        messages.append(message)
    keys = [message_sort_key(message) for message in messages]
    if keys != sorted(set(keys)):
        raise RuntimeError(
            f"{alternative} rank {rank} message tuples are not sorted and unique"
        )

    derived = {
        "phases": sorted({message["phase"] for message in messages}),
        "communication_ids": sorted(
            {message["communication_id"] for message in messages}
        ),
        "rounds": sorted({message["round"] for message in messages}),
        "peers": sorted({message["peer"] for message in messages}),
        "payload_slices": sorted(
            {message["payload_slice"] for message in messages}
        ),
        "send_bytes": sum(
            message["executed_bytes"]
            for message in messages
            if message["direction"] == "send"
        ),
        "recv_bytes": sum(
            message["executed_bytes"]
            for message in messages
            if message["direction"] == "recv"
        ),
    }
    if any(row.get(field) != value for field, value in derived.items()):
        raise RuntimeError(
            f"{alternative} rank {rank} summaries disagree with message tuples"
        )
    return messages


def require_cross_rank_message_matching(
    rows_by_rank: dict[int, dict[str, object]], alternative: str
) -> None:
    sends: collections.Counter[tuple[object, ...]] = collections.Counter()
    recvs: collections.Counter[tuple[object, ...]] = collections.Counter()
    for rank, row in rows_by_rank.items():
        for message in row["messages"]:
            common = (
                message["communication_id"],
                message["phase"],
                message["round"],
                message["payload_slice"],
                message["issue_bytes"],
                message["constant_loop_multiplicity"],
                message["executed_bytes"],
            )
            if message["direction"] == "send":
                sends[(rank, message["peer"], *common)] += 1
            else:
                recvs[(message["peer"], rank, *common)] += 1
    if sends != recvs:
        missing_recvs = sends - recvs
        missing_sends = recvs - sends
        raise RuntimeError(
            f"{alternative} message tuples do not match across ranks: "
            f"missing_recvs={list(missing_recvs.items())[:3]} "
            f"missing_sends={list(missing_sends.items())[:3]}"
        )


def messages_for(
    row: dict[str, object], direction: str, phase: str | None = None
) -> list[dict[str, object]]:
    return [
        message
        for message in row["messages"]
        if message["direction"] == direction
        and (phase is None or message["phase"] == phase)
    ]


def require_direct_all_gather(
    rows_by_rank: dict[int, dict[str, object]], chunk_bytes: int
) -> None:
    expected_peers = set(range(RANK_COUNT))
    expected_rounds = set(range(1, RANK_COUNT))
    for rank, row in rows_by_rank.items():
        sends = messages_for(row, "send")
        recvs = messages_for(row, "recv")
        if (
            len(sends) != RANK_COUNT - 1
            or len(recvs) != RANK_COUNT - 1
            or {message["peer"] for message in sends}
            != expected_peers - {rank}
            or {message["peer"] for message in recvs}
            != expected_peers - {rank}
            or {message["round"] for message in sends} != expected_rounds
            or {message["round"] for message in recvs} != expected_rounds
        ):
            raise RuntimeError(f"all-gather direct rank {rank} fanout is invalid")
        for message in sends:
            if (
                message["peer"] != (rank + message["round"]) % RANK_COUNT
                or message["payload_slice"] != rank
                or message["executed_bytes"] != chunk_bytes
            ):
                raise RuntimeError(
                    f"all-gather direct rank {rank} send tuple is invalid"
                )
        for message in recvs:
            if (
                message["peer"] != (rank - message["round"]) % RANK_COUNT
                or message["payload_slice"] != message["peer"]
                or message["executed_bytes"] != chunk_bytes
            ):
                raise RuntimeError(
                    f"all-gather direct rank {rank} recv tuple is invalid"
                )


def require_direct_reduce_scatter(
    rows_by_rank: dict[int, dict[str, object]], chunk_bytes: int
) -> None:
    expected_peers = set(range(RANK_COUNT))
    for rank, row in rows_by_rank.items():
        sends = messages_for(row, "send")
        recvs = messages_for(row, "recv")
        if (
            len(sends) != RANK_COUNT - 1
            or len(recvs) != RANK_COUNT - 1
            or {message["peer"] for message in sends}
            != expected_peers - {rank}
            or {message["peer"] for message in recvs}
            != expected_peers - {rank}
        ):
            raise RuntimeError(
                f"reduce-scatter direct rank {rank} exchange is invalid"
            )
        for message in sends:
            if (
                message["round"] != rank
                or message["payload_slice"] != message["peer"]
                or message["executed_bytes"] != chunk_bytes
            ):
                raise RuntimeError(
                    f"reduce-scatter direct rank {rank} send tuple is invalid"
                )
        for message in recvs:
            if (
                message["round"] != message["peer"]
                or message["payload_slice"] != rank
                or message["executed_bytes"] != chunk_bytes
            ):
                raise RuntimeError(
                    f"reduce-scatter direct rank {rank} recv tuple is invalid"
                )


def require_single_cycle(successor: dict[int, int], alternative: str) -> None:
    if (
        set(successor) != set(range(RANK_COUNT))
        or set(successor.values()) != set(range(RANK_COUNT))
    ):
        raise RuntimeError(f"{alternative} send graph is not one-in/one-out")
    visited: set[int] = set()
    current = 0
    while current not in visited:
        visited.add(current)
        current = successor[current]
    if current != 0 or visited != set(range(RANK_COUNT)):
        raise RuntimeError(f"{alternative} send graph is not one 16-rank cycle")


def require_ring(
    rows_by_rank: dict[int, dict[str, object]],
    alternative: str,
    round_count: int,
    chunk_bytes: int,
) -> None:
    expected_rounds = set(range(round_count))
    successor: dict[int, int] = {}
    for rank, row in rows_by_rank.items():
        sends = messages_for(row, "send")
        recvs = messages_for(row, "recv")
        send_peers = {message["peer"] for message in sends}
        recv_peers = {message["peer"] for message in recvs}
        if (
            len(sends) != round_count
            or len(recvs) != round_count
            or {message["round"] for message in sends} != expected_rounds
            or {message["round"] for message in recvs} != expected_rounds
            or len(send_peers) != 1
            or len(recv_peers) != 1
            or send_peers == recv_peers
            or any(
                message["executed_bytes"] != chunk_bytes
                for message in (*sends, *recvs)
            )
            or {message["payload_slice"] for message in (*sends, *recvs)}
            != set(range(RANK_COUNT))
        ):
            raise RuntimeError(f"{alternative} rank {rank} ring tuples are invalid")
        successor[rank] = next(iter(send_peers))
    require_single_cycle(successor, alternative)
    predecessor = {next_rank: rank for rank, next_rank in successor.items()}

    def predecessor_power(rank: int, steps: int) -> int:
        result = rank
        for _ in range(steps):
            result = predecessor[result]
        return result

    for rank, row in rows_by_rank.items():
        sends_by_round = {
            message["round"]: message for message in messages_for(row, "send")
        }
        recvs_by_round = {
            message["round"]: message for message in messages_for(row, "recv")
        }
        if (
            {message["peer"] for message in recvs_by_round.values()}
            != {predecessor[rank]}
        ):
            raise RuntimeError(
                f"{alternative} rank {rank} does not receive from predecessor"
            )
        for round_ in range(round_count):
            if alternative == CollectiveAlternative.ALL_GATHER_RING.value:
                expected_send_slice = predecessor_power(rank, round_)
                expected_recv_slice = predecessor_power(rank, round_ + 1)
            elif (
                alternative
                == CollectiveAlternative.REDUCE_SCATTER_RING.value
            ):
                expected_send_slice = predecessor_power(rank, round_ + 1)
                expected_recv_slice = predecessor_power(rank, round_ + 2)
            elif alternative == CollectiveAlternative.ALL_REDUCE_RING.value:
                if round_ < RANK_COUNT - 1:
                    expected_send_slice = predecessor_power(rank, round_)
                    expected_recv_slice = predecessor_power(rank, round_ + 1)
                else:
                    gather_round = round_ - (RANK_COUNT - 1)
                    expected_send_slice = (
                        successor[rank]
                        if gather_round == 0
                        else predecessor_power(rank, gather_round - 1)
                    )
                    expected_recv_slice = predecessor_power(rank, gather_round)
            else:
                raise RuntimeError(
                    f"unknown ring characterization alternative {alternative}"
                )
            if (
                sends_by_round[round_]["payload_slice"]
                != expected_send_slice
                or recvs_by_round[round_]["payload_slice"]
                != expected_recv_slice
            ):
                raise RuntimeError(
                    f"{alternative} rank {rank} round {round_} "
                    "slice forwarding recurrence is invalid"
                )


def require_ordered_tree(
    rows_by_rank: dict[int, dict[str, object]],
    alternative: str,
    payload_bytes: int,
) -> None:
    reduce_phase = "all_reduce_tree_reduce"
    broadcast_phase = "all_reduce_tree_broadcast"
    reduce_sends = [
        (rank, message)
        for rank, row in rows_by_rank.items()
        for message in messages_for(row, "send", reduce_phase)
    ]
    reduce_recvs = [
        (rank, message)
        for rank, row in rows_by_rank.items()
        for message in messages_for(row, "recv", reduce_phase)
    ]
    broadcast_sends = [
        (rank, message)
        for rank, row in rows_by_rank.items()
        for message in messages_for(row, "send", broadcast_phase)
    ]
    broadcast_recvs = [
        (rank, message)
        for rank, row in rows_by_rank.items()
        for message in messages_for(row, "recv", broadcast_phase)
    ]
    all_phase_messages = (
        reduce_sends + reduce_recvs + broadcast_sends + broadcast_recvs
    )
    if (
        any(len(messages) != RANK_COUNT - 1 for messages in (
            reduce_sends,
            reduce_recvs,
            broadcast_sends,
            broadcast_recvs,
        ))
        or any(
            message["executed_bytes"] != payload_bytes
            for _, message in all_phase_messages
        )
    ):
        raise RuntimeError(f"{alternative} tree message cardinality is invalid")

    reduce_by_edge = {
        (rank, message["peer"]): message for rank, message in reduce_sends
    }
    broadcast_by_edge = {
        (rank, message["peer"]): message for rank, message in broadcast_sends
    }
    reduce_edges = set(reduce_by_edge)
    broadcast_edges = set(broadcast_by_edge)
    if (
        len(reduce_edges) != RANK_COUNT - 1
        or broadcast_edges
        != {(parent, child) for child, parent in reduce_edges}
    ):
        raise RuntimeError(
            f"{alternative} reduce/broadcast tree edges are not exact reverses"
        )
    for child, parent in reduce_edges:
        reduce_message = reduce_by_edge[(child, parent)]
        broadcast_message = broadcast_by_edge[(parent, child)]
        if any(
            reduce_message[field] != broadcast_message[field]
            for field in (
                "communication_id",
                "round",
                "payload_slice",
                "issue_bytes",
                "constant_loop_multiplicity",
                "executed_bytes",
            )
        ):
            raise RuntimeError(
                f"{alternative} reverse tree edge fields differ for "
                f"{child}/{parent}"
            )

    parent_by_child: dict[int, int] = {}
    children_by_parent: dict[int, list[int]] = collections.defaultdict(list)
    for child, parent in reduce_edges:
        if child in parent_by_child:
            raise RuntimeError(f"{alternative} tree child has multiple parents")
        parent_by_child[child] = parent
        children_by_parent[parent].append(child)
    roots = set(range(RANK_COUNT)) - set(parent_by_child)
    if len(roots) != 1:
        raise RuntimeError(f"{alternative} tree does not have one root")
    root = next(iter(roots))

    visiting: set[int] = set()
    visited: set[int] = set()

    def inorder(node: int) -> list[int]:
        if node in visiting or node in visited:
            raise RuntimeError(f"{alternative} tree is cyclic or reuses a child")
        visiting.add(node)
        children = children_by_parent.get(node, [])
        lower = [child for child in children if child < node]
        upper = [child for child in children if child > node]
        if len(children) > 2 or len(lower) > 1 or len(upper) > 1:
            raise RuntimeError(
                f"{alternative} tree is not binary in rank-group order"
            )
        order: list[int] = []
        if lower:
            order.extend(inorder(lower[0]))
        order.append(node)
        if upper:
            order.extend(inorder(upper[0]))
        visiting.remove(node)
        visited.add(node)
        return order

    if inorder(root) != list(range(RANK_COUNT)) or len(visited) != RANK_COUNT:
        raise RuntimeError(
            f"{alternative} tree does not preserve rank-group inorder"
        )
    depth_by_rank = {root: 0}
    pending = [root]
    while pending:
        parent = pending.pop()
        for child in children_by_parent.get(parent, []):
            depth_by_rank[child] = depth_by_rank[parent] + 1
            pending.append(child)
    for child, parent in reduce_edges:
        message = reduce_by_edge[(child, parent)]
        if (
            message["payload_slice"] != child
            or message["round"] != depth_by_rank[child]
        ):
            raise RuntimeError(
                f"{alternative} tree edge identity differs from rank/depth"
            )


def validate_compiler_report(
    path: pathlib.Path,
    case: RuntimeCase,
    alternative: str,
    expected_phases: frozenset[str],
) -> dict[str, object]:
    report = json.loads(path.read_text())
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 2
        or report.get("requested_alternative") != alternative
        or report.get("rank_count") != RANK_COUNT
        or report.get("collection_semantics")
        != "sorted-message-tuples-and-constant-loop-weighted-bytes"
        or not isinstance(report.get("ranks"), list)
        or len(report["ranks"]) != RANK_COUNT
    ):
        raise RuntimeError(f"{alternative} compiler report header is invalid")
    rows_by_rank: dict[int, dict[str, object]] = {}
    for row in report["ranks"]:
        if not isinstance(row, dict) or type(row.get("rank")) is not int:
            raise RuntimeError(f"{alternative} compiler report rank is invalid")
        rank = row["rank"]
        if rank in rows_by_rank or not 0 <= rank < RANK_COUNT:
            raise RuntimeError(f"{alternative} compiler report rank domain is invalid")
        phases = require_sorted_unique(row.get("phases"), "phases", str)
        communication_ids = require_sorted_unique(
            row.get("communication_ids"), "communication_ids", int
        )
        rounds = require_sorted_unique(row.get("rounds"), "rounds", int)
        peers = require_sorted_unique(row.get("peers"), "peers", int)
        payload_slices = require_sorted_unique(
            row.get("payload_slices"), "payload_slices", int
        )
        if (
            frozenset(phases) != expected_phases
            or len(communication_ids) != 1
            or communication_ids[0] < 0
            or any(round_ < 0 for round_ in rounds)
            or any(peer < 0 or peer >= RANK_COUNT or peer == rank for peer in peers)
            or any(
                slice_ < 0 or slice_ >= RANK_COUNT
                for slice_ in payload_slices
            )
            or type(row.get("send_bytes")) is not int
            or type(row.get("recv_bytes")) is not int
            or row["send_bytes"] <= 0
            or row["recv_bytes"] <= 0
        ):
            raise RuntimeError(f"{alternative} compiler report rank facts are invalid")
        row["messages"] = validate_rank_messages(
            row, rank, alternative, expected_phases
        )
        rows_by_rank[rank] = row
    if set(rows_by_rank) != set(range(RANK_COUNT)):
        raise RuntimeError(f"{alternative} compiler report rank domain is incomplete")
    if len(
        {
            tuple(row["communication_ids"])
            for row in rows_by_rank.values()
        }
    ) != 1:
        raise RuntimeError(
            f"{alternative} compiler report communication identity differs by rank"
        )
    require_cross_rank_message_matching(rows_by_rank, alternative)

    payload_bytes = case.contract.payload_bytes
    chunk_bytes = payload_bytes // RANK_COUNT
    if alternative == CollectiveAlternative.ALL_GATHER_DIRECT.value:
        require_direct_all_gather(rows_by_rank, chunk_bytes)
    elif alternative == CollectiveAlternative.ALL_GATHER_RING.value:
        require_ring(
            rows_by_rank, alternative, RANK_COUNT - 1, chunk_bytes
        )
    elif alternative == CollectiveAlternative.REDUCE_SCATTER_DIRECT.value:
        require_direct_reduce_scatter(rows_by_rank, chunk_bytes)
    elif alternative == CollectiveAlternative.REDUCE_SCATTER_RING.value:
        require_ring(
            rows_by_rank, alternative, RANK_COUNT - 1, chunk_bytes
        )
    elif alternative == CollectiveAlternative.ALL_REDUCE_RING.value:
        require_ring(
            rows_by_rank, alternative, 2 * (RANK_COUNT - 1), chunk_bytes
        )
    elif alternative == CollectiveAlternative.ALL_REDUCE_TREE.value:
        require_ordered_tree(rows_by_rank, alternative, payload_bytes)
    else:
        raise RuntimeError(f"unknown collective alternative {alternative}")

    return report


def validate_target_pair(
    left: paired_support.TargetStructure,
    right: paired_support.TargetStructure,
    kind: CollectiveKind,
) -> None:
    for structure in (left, right):
        for fragment in ("direct_dte_send_prepare", "direct_dte_recv_prepare"):
            paired_support.require_call(structure.counts, fragment, present=True)
        if kind in (CollectiveKind.REDUCE_SCATTER, CollectiveKind.ALL_REDUCE):
            paired_support.require_call(
                structure.counts, "elementwise_add", present=True
            )
    if left.scheduler_body_sha256 == right.scheduler_body_sha256:
        raise RuntimeError("forced collective variants have identical scheduler bodies")


def remap_package_evidence(
    packages: dict[str, pathlib.Path],
    case: RuntimeCase,
    alternatives: tuple[str, str],
) -> tuple[
    dict[str, dict[tuple[int, str, int], int]],
    dict[str, set[int]],
    dict[str, set[tuple[int, int]]],
]:
    raw = paired_support.validate_paired_packages(
        packages[alternatives[0]], packages[alternatives[1]], case
    )
    names = {"baseline": alternatives[0], "winner": alternatives[1]}
    return tuple(
        {names[name]: evidence for name, evidence in collection.items()}
        for collection in raw
    )  # type: ignore[return-value]


def manifest_boundary_digest(package: pathlib.Path) -> str:
    manifest = json.loads((package / "manifest.json").read_text())
    normalized = paired_support.normalized_manifest(manifest)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def structure_json(
    structure: paired_support.TargetStructure,
) -> dict[str, object]:
    return {
        "scheduler_body_sha256": list(structure.scheduler_body_sha256),
        "static_target_callsite_counts": dict(sorted(structure.counts.items())),
        "straight_line_call_sequence": (
            list(structure.straight_line_calls)
            if structure.straight_line_calls is not None
            else None
        ),
        "workspace_bytes": structure.workspace_bytes,
    }


def manifest_transport_evidence_json(
    evidence: direct_dte_evidence.DirectDTEManifestEvidence,
) -> dict[str, object]:
    return {
        "status_abi": evidence.status_abi,
        "status_resource_by_rank": [
            {"rank": rank, "resource": resource}
            for rank, resource in evidence.status_resource_by_rank
        ],
        "host_watchdog_ranks": list(evidence.host_watchdog_ranks),
        "terminal_completion_by_rank": [
            {"rank": rank, "completion": completion}
            for rank, completion in evidence.terminal_completion_by_rank
        ],
    }


def board_transport_evidence_json(
    evidence: direct_dte_evidence.DirectDTEBoardEvidence,
) -> dict[str, object]:
    return {
        "completion_timeout_ms": evidence.completion_timeout_ms,
        "runtime_all_rank_success_enforced": (
            evidence.runtime_all_rank_success_enforced
        ),
        "observed_all_rank_success": evidence.has_observed_all_rank_success,
        "observed_status_by_rank": (
            [
                dataclasses.asdict(observation)
                for observation in evidence.observed_status_by_rank
            ]
            if evidence.observed_status_by_rank is not None
            else None
        ),
        "status_observation_gap": evidence.status_observation_gap,
        "terminal_completion_by_rank": [
            {"rank": rank, "completion": completion}
            for rank, completion in evidence.terminal_completion_by_rank
        ],
    }


def balanced_order(alternatives: tuple[str, str], repeat: int) -> list[str]:
    order: list[str] = []
    for iteration in range(repeat):
        order.extend(
            alternatives if iteration % 2 == 0 else tuple(reversed(alternatives))
        )
    return order


def main() -> int:
    args = parse_args()
    contract = CASES_BY_KEY[args.case]
    case = runtime_case(contract)
    alternatives = (
        contract.left_alternative.value,
        contract.right_alternative.value,
    )
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not POST_SPMD_CARRIER.is_file():
        raise RuntimeError("explicit post-SPMD carrier is missing")
    args.work_dir = args.work_dir.resolve()
    prepare_work_dir(args.work_dir, alternatives)
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("collective characterization hardware execution is not armed")
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
    if not args.no_card and args.expected_tile_count != RANK_COUNT:
        raise RuntimeError("collective characterization requires exactly 16 tiles")

    source = write_source(args.work_dir, case)
    packages = {
        alternative: args.work_dir / f"{alternative}-package"
        for alternative in alternatives
    }
    report_paths = {
        alternative: args.work_dir / f"{alternative}-compiler-report.json"
        for alternative in alternatives
    }
    for alternative in alternatives:
        compile_package(
            args.wafer_compile_test,
            source,
            packages[alternative],
            report_paths[alternative],
            case,
            alternative,
        )
    expected_phases = {
        alternatives[0]: contract.left_expected_phases,
        alternatives[1]: contract.right_expected_phases,
    }
    compiler_reports = {
        alternative: validate_compiler_report(
            report_paths[alternative],
            case,
            alternative,
            expected_phases[alternative],
        )
        for alternative in alternatives
    }
    (
        bindings_by_variant,
        output_ids_by_variant,
        completion_evidence_by_variant,
    ) = remap_package_evidence(packages, case, alternatives)
    manifest_transport_evidence = {
        alternative: direct_dte_evidence.validate_direct_dte_manifest(
            json.loads((packages[alternative] / "manifest.json").read_text())
        )
        for alternative in alternatives
    }
    structures = {
        alternative: paired_support.target_structure(
            packages[alternative], args.tx8_objdump
        )
        for alternative in alternatives
    }
    validate_target_pair(
        structures[alternatives[0]],
        structures[alternatives[1]],
        contract.collective_kind,
    )
    boundary_digests = {
        alternative: manifest_boundary_digest(packages[alternative])
        for alternative in alternatives
    }
    if len(set(boundary_digests.values())) != 1:
        raise RuntimeError("forced variants differ at normalized manifest boundary")
    identity = {
        "case": case.key,
        "collective_kind": contract.collective_kind.value,
        "payload_bytes": contract.payload_bytes,
        "element_type": contract.element_type,
        "rank_count": RANK_COUNT,
        "target_profile": TARGET_PROFILE,
        "launch_abi": LAUNCH_ABI,
        "source_mode": (
            "explicit-post-spmd-carrier"
            if contract.collective_kind
            in (CollectiveKind.ALL_GATHER, CollectiveKind.REDUCE_SCATTER)
            else "exporter-spmd"
        ),
        "source_snapshots_sha256": {
            str(relative): hashlib.sha256(
                (packages[alternatives[0]] / relative).read_bytes()
            ).hexdigest()
            for relative in SOURCE_SNAPSHOT_PATHS
        },
        "normalized_manifest_sha256": next(iter(boundary_digests.values())),
        "module_digests": {
            alternative: list(
                paired_support.module_digests(packages[alternative])
            )
            for alternative in alternatives
        },
    }
    structure_record = {
        "schema_version": 1,
        "identity": identity,
        "variants": {
            alternative: structure_json(structures[alternative])
            for alternative in alternatives
        },
    }
    (args.work_dir / "target-structure.json").write_text(
        json.dumps(structure_record, indent=2, sort_keys=True) + "\n"
    )
    aggregate_report = {
        "schema_version": 1,
        "identity": identity,
        "variants": {
            alternative: {
                "compiler_report": compiler_reports[alternative],
                "target_structure": structure_json(structures[alternative]),
                "transport_contract": manifest_transport_evidence_json(
                    manifest_transport_evidence[alternative]
                ),
            }
            for alternative in alternatives
        },
    }
    (args.work_dir / "collective-characterization.json").write_text(
        json.dumps(aggregate_report, indent=2, sort_keys=True) + "\n"
    )
    resource_arguments = paired_support.write_payloads(
        args.work_dir, case, bindings_by_variant
    )

    if args.no_card:
        for alternative, package in packages.items():
            result = paired_support.run(
                paired_support.no_card_command(args.wafer_run, package, case)
            )
            if "board_execution: false" not in result.stdout:
                raise RuntimeError(
                    f"{alternative} no-card output omitted execution state"
                )
        print(
            "collective_characterization_no_card: "
            f"case={case.key} paired_packages=true "
            "typed_phases=true target_structure_distinct=true"
        )
        return 0

    rows: list[dict[str, object]] = []
    for sample, alternative in enumerate(
        balanced_order(alternatives, args.repeat), start=1
    ):
        command = paired_support.board_command(
            args,
            packages[alternative],
            case,
            resource_arguments[alternative],
        )
        command_completion_timeout_ms = (
            direct_dte_evidence.validate_direct_dte_board_command(command)
        )
        start_ns = time.monotonic_ns()
        result = paired_support.run(
            command,
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        elapsed_ns = time.monotonic_ns() - start_ns
        paired_support.verify_board_output(
            result.stdout,
            case,
            output_ids_by_variant[alternative],
            completion_evidence_by_variant[alternative],
        )
        transport_evidence = direct_dte_evidence.validate_direct_dte_board_output(
            result.stdout,
            manifest_transport_evidence[alternative],
            completion_timeout_ms=command_completion_timeout_ms,
        )
        row = {
            "case": case.key,
            "sample": sample,
            "alternative": alternative,
            "host_process_elapsed_ns": elapsed_ns,
            "exact": True,
            "transport_evidence": board_transport_evidence_json(
                transport_evidence
            ),
        }
        rows.append(row)
        print(
            "collective_characterization_sample: "
            + json.dumps(row, sort_keys=True)
        )
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
                    "balanced host process elapsed observation; not PMU, "
                    "device cycles, or calibrated compiler cost"
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
            f"wafer_board_collective_characterization_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
