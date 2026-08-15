#!/usr/bin/env python3
"""Compile and run exact collective traffic behavior cases on one TX81 card."""

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

import numpy as np

import wafer_board_compiler_optimization_campaign_test as paired_support
import wafer_direct_dte_board_evidence as direct_dte_evidence
import wafer_runtime_launch_contract as runtime_launch
from wafer_collective_traffic_behavior_catalog import (
    CASE_KEYS,
    CASES_BY_KEY,
    RANK_COUNT,
    CollectiveKind,
    TrafficBehaviorCase,
    row_major_manhattan_distance,
)


TARGET_IDENTITY = paired_support.TARGET_IDENTITY
LAUNCH_KIND = paired_support.CLUSTER_LAUNCH_KIND
STATUS_ABI = paired_support.DIRECT_DTE_STATUS_ABI
PROCESS_TIMEOUT_MARGIN_SECONDS = paired_support.PROCESS_TIMEOUT_MARGIN_SECONDS
SPMD_HELPER_ENVIRONMENT_VARIABLE = "WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER"
POST_SPMD_CARRIER = pathlib.Path(__file__).with_name(
    "wafer_collective_traffic_post_spmd_carrier.py"
)
SOURCE_SNAPSHOT_PATHS = paired_support.SOURCE_SNAPSHOT_PATHS
F16_DTYPE = np.dtype("<f2")
ALL_TO_ALL_PAIR_BITS_BASE = np.uint16(0x4400)
ALL_TO_ALL_LANE_BITS_BASE = np.uint16(0x0800)
PERMUTE_RANK_BITS_BASE = np.uint16(0x4800)
PERMUTE_LANE_BITS_BASE = np.uint16(0x0400)


@dataclasses.dataclass(frozen=True)
class RuntimeCase:
    contract: TrafficBehaviorCase
    inputs: tuple[paired_support.TensorSpec, ...]
    outputs: tuple[paired_support.TensorSpec, ...]

    @property
    def key(self) -> str:
        return self.contract.key

    @property
    def rank_count(self) -> int:
        return self.contract.rank_count

    @property
    def launch_kind(self) -> str:
        return LAUNCH_KIND

    @property
    def payload_factory(self):
        return lambda: payloads(self)

    @property
    def output_comparison(self):
        return paired_support.RAW_EXACT_OUTPUT


@dataclasses.dataclass(frozen=True)
class PackageEvidence:
    bindings: dict[tuple[int, str, int], int]
    output_ids: set[int]
    completion_evidence: set[tuple[int, int]]
    transport: direct_dte_evidence.DirectDTEManifestEvidence


def runtime_case(contract: TrafficBehaviorCase) -> RuntimeCase:
    inputs = (
        paired_support.TensorSpec(
            contract.input_shape,
            contract.element_type,
            "float16",
        ),
    )
    outputs = (
        paired_support.TensorSpec(
            contract.output_shape,
            contract.element_type,
            "float16",
        ),
    )
    return RuntimeCase(contract, inputs, outputs)


def rank_group() -> str:
    return ", ".join(str(rank) for rank in range(RANK_COUNT))


def tensor_type(shape: tuple[int, ...]) -> str:
    return "x".join(str(dimension) for dimension in shape) + "xf16"


def all_to_all_module(case: TrafficBehaviorCase) -> str:
    input_type = tensor_type(case.input_shape)
    output_type = tensor_type(case.output_shape)
    return f"""\
module {{
  func.func @main(
      %input: tensor<{input_type}>) -> tensor<{output_type}> {{
    %result = "stablehlo.all_to_all"(%input) {{
      split_dimension = 0 : i64,
      concat_dimension = 1 : i64,
      split_count = 16 : i64,
      replica_groups = dense<[[{rank_group()}]]> : tensor<1x16xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 71, type = 1>,
      use_global_device_ids
    }} : (tensor<{input_type}>) -> tensor<{output_type}>
    return %result : tensor<{output_type}>
  }}
}}
"""


def dense_pairs(epoch: tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"[{source}, {target}]" for source, target in epoch)


def collective_permute_module(case: TrafficBehaviorCase) -> str:
    payload_type = tensor_type(case.input_shape)
    lines = [
        "module {",
        "  func.func @main(",
        f"      %input: tensor<{payload_type}>) -> tensor<{payload_type}> {{",
    ]
    current = "%input"
    for epoch_index, epoch in enumerate(case.epochs):
        result = f"%epoch_{epoch_index}"
        lines.extend(
            (
                f'    {result} = "stablehlo.collective_permute"({current}) {{',
                (
                    "      source_target_pairs = "
                    f"dense<[{dense_pairs(epoch)}]> "
                    f": tensor<{len(epoch)}x2xi64>,"
                ),
                (
                    "      channel_handle = "
                    "#stablehlo.channel_handle<"
                    f"handle = {79 + epoch_index}, type = 1>"
                ),
                (
                    f"    }} : (tensor<{payload_type}>) "
                    f"-> tensor<{payload_type}>"
                ),
            )
        )
        current = result
    lines.extend(
        (
            f"    return {current} : tensor<{payload_type}>",
            "  }",
            "}",
            "",
        )
    )
    return "\n".join(lines)


def module_text(case: RuntimeCase) -> str:
    if case.contract.collective_kind == CollectiveKind.ALL_TO_ALL:
        return all_to_all_module(case.contract)
    if case.contract.collective_kind == CollectiveKind.COLLECTIVE_PERMUTE:
        return collective_permute_module(case.contract)
    raise RuntimeError(
        f"unsupported collective kind {case.contract.collective_kind}"
    )


def all_to_all_payloads(
    case: TrafficBehaviorCase,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    rank_inputs: list[np.ndarray] = []
    lanes_per_peer = case.input_shape[1]
    lanes = np.arange(lanes_per_peer, dtype=np.uint16)
    for source in range(RANK_COUNT):
        payload = np.empty(case.input_shape, dtype=F16_DTYPE)
        bits = payload.view(np.uint16)
        for destination in range(RANK_COUNT):
            bits[destination, :, 0] = (
                ALL_TO_ALL_PAIR_BITS_BASE + source * RANK_COUNT + destination
            )
            bits[destination, :, 1] = ALL_TO_ALL_LANE_BITS_BASE + lanes
        rank_inputs.append(payload)
    outputs = [
        [
            np.concatenate(
                [
                    rank_inputs[source][destination : destination + 1]
                    for source in range(RANK_COUNT)
                ],
                axis=1,
            )
        ]
        for destination in range(RANK_COUNT)
    ]
    return [[payload] for payload in rank_inputs], outputs


def permute_rank_payload(rank: int, payload_bytes: int = 4096) -> np.ndarray:
    lanes = np.arange(payload_bytes // 4, dtype=np.uint16)
    payload = np.empty((payload_bytes // 4, 2), dtype=F16_DTYPE)
    bits = payload.view(np.uint16)
    bits[:, 0] = PERMUTE_RANK_BITS_BASE + rank
    bits[:, 1] = PERMUTE_LANE_BITS_BASE + lanes
    return payload


def simulate_permute_epochs(
    rank_inputs: list[np.ndarray],
    epochs: tuple[tuple[tuple[int, int], ...], ...],
) -> list[np.ndarray]:
    values = [payload.copy() for payload in rank_inputs]
    for epoch in epochs:
        next_values = [np.zeros_like(values[0]) for _ in range(RANK_COUNT)]
        for source, target in epoch:
            next_values[target] = values[source].copy()
        values = next_values
    return values


def payloads(
    case: RuntimeCase,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    if case.contract.collective_kind == CollectiveKind.ALL_TO_ALL:
        return all_to_all_payloads(case.contract)
    rank_inputs = [
        permute_rank_payload(rank, case.contract.payload_bytes)
        for rank in range(RANK_COUNT)
    ]
    expected = simulate_permute_epochs(
        rank_inputs,
        case.contract.epochs,
    )
    return (
        [[payload] for payload in rank_inputs],
        [[payload] for payload in expected],
    )


def write_exact_payloads(
    work_dir: pathlib.Path,
    case: RuntimeCase,
    bindings: dict[tuple[int, str, int], int],
) -> list[str]:
    inputs, outputs = payloads(case)
    raw = work_dir / "raw"
    raw.mkdir()
    arguments: list[str] = []
    for rank in range(case.rank_count):
        for role, arrays, option in (
            ("user_input", inputs[rank], "--resource"),
            ("output", outputs[rank], "--expected"),
        ):
            for index, (array, spec) in enumerate(
                zip(
                    arrays,
                    case.inputs if role == "user_input" else case.outputs,
                    strict=True,
                )
            ):
                if (
                    array.dtype != F16_DTYPE
                    or tuple(array.shape) != spec.shape
                    or array.nbytes != case.contract.payload_bytes
                    or not np.all(np.isfinite(array))
                ):
                    raise RuntimeError(
                        f"{case.key}: invalid finite f16 payload for "
                        f"{(rank, role, index)}"
                    )
                path = raw / (
                    f"rank_{rank:02d}_{role}_{index}.{spec.mlir_dtype}.raw"
                )
                path.write_bytes(np.ascontiguousarray(array).tobytes())
                arguments.extend(
                    [
                        option,
                        f"{bindings[(rank, role, index)]}={path}",
                    ]
                )
    return arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def prepare_work_dir(work_dir: pathlib.Path) -> None:
    known_children = {
        "source-program",
        "package",
        "raw",
        "target-structure.json",
        "collective-traffic-behavior.json",
        "board-observations.json",
    }
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
    case: RuntimeCase,
) -> None:
    environment = os.environ.copy()
    for name in (
        paired_support.BASELINE_ENVIRONMENT_VARIABLE,
        "WAFER_TEST_FAIL_AFTER_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_TARGET_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_PACKAGE_LOGICAL_RANK",
        "WAFER_TEST_FAIL_TARGET_MODEL_TERMINAL_RANK",
        "WAFER_TEST_COLLECTIVE_CHARACTERIZATION_ALTERNATIVE",
        "WAFER_TEST_COLLECTIVE_CHARACTERIZATION_REPORT",
        SPMD_HELPER_ENVIRONMENT_VARIABLE,
    ):
        environment.pop(name, None)
    environment[SPMD_HELPER_ENVIRONMENT_VARIABLE] = str(POST_SPMD_CARRIER)
    result = paired_support.run(
        [
            str(compiler),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(output),
            f"--execution-ranks={RANK_COUNT}",
            f"--launch-kind={LAUNCH_KIND}",
        ],
        environment=environment,
    )
    expected = (
        "wafer-compile: wrote verified package with "
        f"execution-ranks={RANK_COUNT}"
    )
    if expected not in result.stdout or not output.is_dir():
        raise RuntimeError(f"compiler did not write {case.key}")


def replicated_boundary_binding(
    index_field: str,
    shape: tuple[int, ...],
    dtype: str,
) -> dict[str, object]:
    return {
        index_field: 0,
        "distribution": "replicated",
        "global_shape": list(shape),
        "local_shape": list(shape),
        "dtype": dtype,
        "ranks": [
            {
                "rank": rank,
                "replica_id": rank,
                "offsets": [0] * len(shape),
                "sizes": list(shape),
                "strides": [1] * len(shape),
            }
            for rank in range(RANK_COUNT)
        ],
    }


def validate_distributed_boundary(
    package: pathlib.Path,
    case: RuntimeCase,
) -> None:
    metadata = json.loads(
        (package / "functions" / "forward.meta").read_text()
    )
    expected = {
        "version": 1,
        "logical_rank_count": RANK_COUNT,
        "inputs": [
            replicated_boundary_binding(
                "argument_index",
                case.inputs[0].shape,
                case.inputs[0].metadata_dtype,
            )
        ],
        "outputs": [
            replicated_boundary_binding(
                "result_index",
                case.outputs[0].shape,
                case.outputs[0].metadata_dtype,
            )
        ],
    }
    if metadata.get("distributed_boundary") != expected:
        raise RuntimeError(
            "post-SPMD carrier did not produce the exact replicated boundary"
        )


def validate_package(
    package: pathlib.Path,
    case: RuntimeCase,
) -> PackageEvidence:
    validate_distributed_boundary(package, case)
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="collective traffic",
    )
    if (
        manifest.get("rank_count") != RANK_COUNT
        or manifest.get("target", {}).get("identity") != TARGET_IDENTITY
    ):
        raise RuntimeError("collective traffic package target fields are invalid")
    transport = direct_dte_evidence.validate_direct_dte_manifest(manifest)

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
        if (
            type(rank) is not int
            or type(role_index) is not int
            or type(resource_id) is not int
            or role not in ("user_input", "output")
            or role_index != 0
            or not 0 <= rank < RANK_COUNT
        ):
            raise RuntimeError("host-visible traffic resource identity is invalid")
        key = (rank, role, role_index)
        if key in bindings:
            raise RuntimeError(f"duplicate host-visible traffic resource {key}")
        spec = case.inputs[0] if role == "user_input" else case.outputs[0]
        if (
            resource.get("type")
            != {"dtype": spec.mlir_dtype, "shape": list(spec.shape)}
            or resource.get("bytes") != case.contract.payload_bytes
            or resource.get("access")
            != ("read_only" if role == "user_input" else "write_only")
        ):
            raise RuntimeError(f"host-visible traffic resource differs for {key}")
        bindings[key] = resource_id
        if role == "output":
            output_ids.add(resource_id)
    expected_bindings = {
        (rank, role, 0)
        for rank in range(RANK_COUNT)
        for role in ("user_input", "output")
    }
    if set(bindings) != expected_bindings or len(output_ids) != RANK_COUNT:
        raise RuntimeError("package host-visible traffic domain is incomplete")
    completion_evidence = {
        (completion, rank)
        for rank, completion in transport.terminal_completion_by_rank
    }
    return PackageEvidence(
        bindings,
        output_ids,
        completion_evidence,
        transport,
    )


def validate_target_structure(
    structure: paired_support.TargetStructure,
) -> None:
    for fragment in ("direct_dte_send_issue", "direct_dte_recv_prepare"):
        paired_support.require_call(structure.counts, fragment, present=True)


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


def intended_graph_epochs(
    case: TrafficBehaviorCase,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if case.collective_kind == CollectiveKind.ALL_TO_ALL:
        return (
            tuple(
                (source, target)
                for source in range(RANK_COUNT)
                for target in range(RANK_COUNT)
                if source != target
            ),
        )
    return case.epochs


def intended_graph_json(case: TrafficBehaviorCase) -> list[list[list[int]]]:
    return [
        [[source, target] for source, target in epoch]
        for epoch in intended_graph_epochs(case)
    ]


def intended_min_hop_summary(
    case: TrafficBehaviorCase,
) -> list[dict[str, object]]:
    issue_bytes = (
        case.payload_bytes // RANK_COUNT
        if case.collective_kind == CollectiveKind.ALL_TO_ALL
        else case.payload_bytes
    )
    summaries: list[dict[str, object]] = []
    for epoch_index, epoch in enumerate(intended_graph_epochs(case)):
        distances = [
            row_major_manhattan_distance(source, target)
            for source, target in epoch
            if source != target
        ]
        histogram = collections.Counter(distances)
        summaries.append(
            {
                "epoch": epoch_index,
                "remote_edge_count": len(distances),
                "local_self_count": sum(
                    source == target for source, target in epoch
                ),
                "issue_bytes_per_remote_edge": issue_bytes,
                "row_major_min_hop_histogram": {
                    str(distance): count
                    for distance, count in sorted(histogram.items())
                },
                "row_major_min_hop_byte_demand": (
                    sum(distances) * issue_bytes
                ),
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    case = runtime_case(CASES_BY_KEY[args.case])
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not POST_SPMD_CARRIER.is_file() or not os.access(
        POST_SPMD_CARRIER, os.X_OK
    ):
        raise RuntimeError("executable collective traffic carrier is missing")
    args.work_dir = args.work_dir.resolve()
    prepare_work_dir(args.work_dir)

    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("collective traffic behavior hardware execution is not armed")
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
        raise RuntimeError(
            "collective traffic behavior requires exactly 16 qualified tiles"
        )

    source = write_source(args.work_dir, case)
    package = args.work_dir / "package"
    compile_package(args.wafer_compile_test, source, package, case)
    package_evidence = validate_package(package, case)
    structure = paired_support.target_structure(package, args.tx8_objdump)
    validate_target_structure(structure)

    identity = {
        "case": case.key,
        "collective_kind": case.contract.collective_kind.value,
        "graph_kind": case.contract.graph_kind.value,
        "payload_bytes_per_rank": case.contract.payload_bytes,
        "rank_count": RANK_COUNT,
        "target_identity": TARGET_IDENTITY,
        "launch": runtime_launch.CLUSTER_KERNEL_LAUNCH,
        "source_mode": "explicit-post-spmd-traffic-carrier",
        "source_snapshots_sha256": {
            str(relative): hashlib.sha256(
                (package / relative).read_bytes()
            ).hexdigest()
            for relative in SOURCE_SNAPSHOT_PATHS
        },
        "module_digests": list(paired_support.module_digests(package)),
    }
    structure_record = {
        "schema_version": 1,
        "identity": identity,
        "intended_semantic_graph_epochs": intended_graph_json(case.contract),
        "intended_row_major_min_hop_demand": intended_min_hop_summary(
            case.contract
        ),
        "target_structure": structure_json(structure),
        "claim_boundaries": {
            "generic_direct_dte_calls_observed": True,
            "accepted_target_peer_graph_observed": False,
            "physical_route_observed": False,
            "device_phase_basis_observed": False,
            "contention_cost_observed": False,
            "same_physical_buffer_observed": False,
        },
    }
    (args.work_dir / "target-structure.json").write_text(
        json.dumps(structure_record, indent=2, sort_keys=True) + "\n"
    )
    behavior_record = {
        "schema_version": 1,
        "identity": identity,
        "objectives": list(case.contract.objectives),
        "numeric_oracle": case.contract.numeric_oracle.value,
        "disposition": case.contract.disposition.value,
        "target_structure": structure_json(structure),
        "transport_contract": manifest_transport_evidence_json(
            package_evidence.transport
        ),
        "activation": {
            "board_correctness": "qualified-board-arm-required",
            "target_peer_graph": "blocked-missing-accepted-message-tuple-report",
            "route_or_contention_measurement": (
                case.contract.device_measurement.value
            ),
        },
    }
    (args.work_dir / "collective-traffic-behavior.json").write_text(
        json.dumps(behavior_record, indent=2, sort_keys=True) + "\n"
    )
    resource_arguments = write_exact_payloads(
        args.work_dir,
        case,
        package_evidence.bindings,
    )

    if args.no_card:
        result = paired_support.run(
            paired_support.no_card_command(args.wafer_run, package, case)
        )
        if "board_execution: false" not in result.stdout:
            raise RuntimeError("no-card output omitted execution state")
        print(
            "collective_traffic_behavior_no_card: "
            f"case={case.key} package_verified=true "
            "generic_dte_calls=true peer_graph_observed=false "
            "device_measurement=false"
        )
        return 0

    samples: list[dict[str, object]] = []
    for sample in range(1, args.repeat + 1):
        command = paired_support.board_command(
            args,
            package,
            case,
            resource_arguments,
        )
        command_completion_timeout_ms = (
            direct_dte_evidence.validate_direct_dte_board_command(command)
        )
        result = paired_support.run(
            command,
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        paired_support.verify_board_output(
            result.stdout,
            case,
            package_evidence.output_ids,
            package_evidence.completion_evidence,
        )
        transport_evidence = direct_dte_evidence.validate_direct_dte_board_output(
            result.stdout,
            package_evidence.transport,
            completion_timeout_ms=command_completion_timeout_ms,
        )
        row = {
            "case": case.key,
            "sample": sample,
            "full_output_exact": True,
            "transport_evidence": board_transport_evidence_json(
                transport_evidence
            ),
            "target_peer_graph_observed": False,
            "device_measurement_observed": False,
        }
        samples.append(row)
        print(
            "collective_traffic_behavior_sample: "
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
                "evidence_scope": (
                    "full-output exact collective behavior and Direct-DTE "
                    "lifecycle only; no peer-route, device phase, contention "
                    "cost, or physical-buffer-alias claim"
                ),
                "samples": samples,
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
            f"wafer_board_collective_traffic_behavior_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
