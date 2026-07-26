#!/usr/bin/env python3
"""Build one package and run bounded DMA/DDR/SPM descriptor observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import struct
import sys
from collections.abc import Callable, Iterable

import wafer_board_instruction_family_probe_test as package_support
import wafer_memory_descriptor_calibration_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_memory_descriptor_calibration_probe.c"
MODULE = """\
module {
  func.func @main(
      %request: tensor<524288xf32>,
      %payload: tensor<524288xf32>) -> tensor<524288xf32> {
    %result = stablehlo.add %request, %payload : tensor<524288xf32>
    return %result : tensor<524288xf32>
  }
}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [524288], "dtype": "float32", "dynamic_dims": []},
        {"shape": [524288], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [524288], "dtype": "float32", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request"},
        {"type_": "input_arg", "position": 1, "name": "payload"},
    ],
    "unused_inputs": [],
}
PMU_STABLE_MASK = (1 << 6) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument(
        "--domain",
        action="append",
        choices=tuple(
            sorted({case.domain for case in catalog.CATALOG})
        ),
    )
    parser.add_argument(
        "--oracle", action="append", choices=("exact", "observation")
    )
    parser.add_argument(
        "--relative-spm-offset",
        action="append",
        type=int,
        choices=catalog.SPM_PAIR_RELATIVE_OFFSETS,
        help="select engine-pair rows with this relative SPM offset; repeatable",
    )
    parser.add_argument(
        "--parallel-address-sweep",
        action="store_true",
        help=(
            "select the representative CT+RDMA nine-point 256-byte-step "
            "relative-offset and 64-byte common-base phase controls"
        ),
    )
    parser.add_argument(
        "--parallel-pair-expanded",
        action="store_true",
        help=(
            "select the expanded sustained, cross-worker, and dependency "
            "parallel-pair controls"
        ),
    )
    parser.add_argument(
        "--conflict-equivalence",
        action="store_true",
        help=(
            "select paired same-invocation CT+RDMA common-base controls "
            "with held-out translations, reciprocal issue order, and "
            "counterbalanced repetitions"
        ),
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    return parser.parse_args()


def configure_package_support() -> None:
    package_support.catalog = catalog
    package_support.PROBE_C = PROBE_C
    package_support.MODULE = MODULE
    package_support.METADATA = METADATA


def select_cases(args: argparse.Namespace) -> tuple[catalog.MemoryCase, ...]:
    parallel_pair_expanded = getattr(
        args, "parallel_pair_expanded", False
    )
    conflict_equivalence = getattr(args, "conflict_equivalence", False)
    conflicting_expanded_filters = {
        "--case": bool(args.selected_cases),
        "--domain": bool(args.domain),
        "--relative-spm-offset": bool(args.relative_spm_offset),
        "--parallel-address-sweep": bool(
            getattr(args, "parallel_address_sweep", False)
        ),
    }
    if conflict_equivalence:
        conflicting_equivalence_filters = {
            **conflicting_expanded_filters,
            "--parallel-pair-expanded": parallel_pair_expanded,
        }
        if any(conflicting_equivalence_filters.values()):
            conflicts = sorted(
                name
                for name, enabled in conflicting_equivalence_filters.items()
                if enabled
            )
            raise RuntimeError(
                "--conflict-equivalence cannot be combined with "
                + ", ".join(conflicts)
            )
    if parallel_pair_expanded and any(
        conflicting_expanded_filters.values()
    ):
        conflicts = sorted(
            name
            for name, enabled in conflicting_expanded_filters.items()
            if enabled
        )
        raise RuntimeError(
            "--parallel-pair-expanded cannot be combined with "
            + ", ".join(conflicts)
        )
    if conflict_equivalence:
        selected = catalog.CONFLICT_EQUIVALENCE_CASES
    elif parallel_pair_expanded:
        selected = catalog.PARALLEL_PAIR_CASES
    elif args.selected_cases:
        unknown = sorted(
            set(args.selected_cases) - set(catalog.CASES_BY_NAME)
        )
        if unknown:
            raise RuntimeError(
                f"unknown memory descriptor cases: {unknown}"
            )
        selected = tuple(
            catalog.CASES_BY_NAME[name] for name in args.selected_cases
        )
    else:
        selected = catalog.CATALOG
    if args.domain:
        domains = set(args.domain)
        selected = tuple(
            case for case in selected if case.domain in domains
        )
    if args.oracle:
        exact = {"exact": True, "observation": False}
        accepted = {exact[value] for value in args.oracle}
        selected = tuple(
            case for case in selected if case.is_exact in accepted
        )
    if args.relative_spm_offset:
        offsets = set(args.relative_spm_offset)
        selected = tuple(
            case
            for case in selected
            if case.domain == "spm-bank-engine-pair"
            and abs(case.spm_b - case.spm_a) in offsets
        )
    if getattr(args, "parallel_address_sweep", False):
        sweep = set(catalog.PARALLEL_ADDRESS_SWEEP_CASES)
        selected = tuple(case for case in selected if case in sweep)
    if not selected:
        raise RuntimeError("memory descriptor filters selected no cases")
    if len({case.name for case in selected}) != len(selected):
        raise RuntimeError("memory descriptor selection repeats a case")
    return selected


def _expected_record(
    case: catalog.MemoryCase, sample: int
) -> dict[str, int]:
    descriptor = case.descriptor
    counts = case.expected_counts
    return {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (catalog.SCHEMA << 32) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "KIND": case.kind,
        "ENGINE_A": case.engine_a,
        "ENGINE_B": case.engine_b,
        "SCHEDULE": case.schedule,
        "EFFECT": case.effect,
        "RELATION": case.relation,
        "ORACLE": case.oracle,
        "FORMAT": case.format,
        "SRC_DDR_OFFSET": case.src_ddr_offset,
        "DST_DDR_OFFSET": case.dst_ddr_offset,
        "SPM_A": case.spm_a,
        "SPM_B": case.spm_b,
        "INNER_BYTES": descriptor.inner_bytes,
        "STRIDE0": descriptor.stride0,
        "STRIDE1": descriptor.stride1,
        "STRIDE2": descriptor.stride2,
        "ITERATION0": descriptor.iteration0,
        "ITERATION1": descriptor.iteration1,
        "ITERATION2": descriptor.iteration2,
        "COMPACT_BYTES": descriptor.compact_bytes,
        "ENVELOPE_BYTES": descriptor.envelope_bytes,
        "OUTPUT_BYTES": case.output_bytes,
        "SAMPLE": sample,
        "SPM_GUARD_MISMATCHES": 0,
        "CT_INST_DELTA": counts[catalog.ENGINE_CT],
        "NE_INST_DELTA": counts[catalog.ENGINE_NE],
        "RDMA_INST_DELTA": counts[catalog.ENGINE_RDMA],
        "WDMA_INST_DELTA": counts[catalog.ENGINE_WDMA],
        "TDMA_INST_DELTA": counts[catalog.ENGINE_TDMA],
        "OUTPUT_DATA_OFFSET": catalog.OUTPUT_DATA_OFFSET,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "FLAGS": 0xF,
        "WORKER_A": case.worker_a,
        "WORKER_B": case.worker_b,
        "ROUNDS": case.rounds,
        "BUFFER_COUNT": case.buffer_count,
        "ISSUE_ORDER": case.issue_order,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }


def validate_output(
    path: pathlib.Path,
    case: catalog.MemoryCase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    return validate_output_bytes(path.read_bytes(), case, built, sample)


def validate_output_bytes(
    raw: bytes,
    case: catalog.MemoryCase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(f"{case.name}: invalid output resource size")
    words = struct.unpack_from(f"<{catalog.RECORD_WORDS}Q", raw)
    failures = {
        key: (words[catalog.REC[key]], expected)
        for key, expected in _expected_record(case, sample).items()
        if words[catalog.REC[key]] != expected
    }
    if failures:
        raise RuntimeError(f"{case.name}: record oracle failed: {failures}")
    rec = catalog.REC
    if (
        words[rec["PLAN_CYCLES"]] == 0
        or words[rec["PMU_ENABLE"]] == 0
        or words[rec["SERIAL_MODE"]] & 1
        or words[rec["STABLE_BEFORE"]] != PMU_STABLE_MASK
        or words[rec["STABLE_AFTER"]] != PMU_STABLE_MASK
    ):
        raise RuntimeError(
            f"{case.name}: timing/PMU/parallel-mode evidence is invalid"
        )

    expected_guard = bytes([catalog.SPM_GUARD]) * catalog.SPM_GUARD_BYTES
    for lane, (begin, end), slot_span in zip(
        range(len(catalog.spm_slot_spans(case))),
        catalog.spm_dump_ranges(case),
        catalog.spm_slot_spans(case),
        strict=True,
    ):
        if end - begin != slot_span + 2 * catalog.SPM_GUARD_BYTES:
            raise RuntimeError(
                f"{case.name}: invalid SPM guard dump layout for lane {lane}"
            )
        before = raw[begin : begin + catalog.SPM_GUARD_BYTES]
        after_begin = begin + catalog.SPM_GUARD_BYTES + slot_span
        after = raw[after_begin : after_begin + catalog.SPM_GUARD_BYTES]
        if before != expected_guard or after != expected_guard:
            raise RuntimeError(
                f"{case.name}: host SPM guard oracle failed for lane {lane}"
            )

    parallel: dict[str, object] | None = None
    if case.kind == catalog.KIND_PARALLEL_PAIR:
        lane_instruction_deltas = (
            words[rec["LANE_A_WORKER_INST_DELTA"]],
            words[rec["LANE_B_WORKER_INST_DELTA"]],
        )
        if lane_instruction_deltas != (case.rounds, case.rounds):
            raise RuntimeError(
                f"{case.name}: per-worker instruction oracle failed: "
                f"{lane_instruction_deltas} != {(case.rounds, case.rounds)}"
            )
        expected_worker_mask = (
            (1 << case.worker_a) | (1 << case.worker_b)
        )
        if words[rec["WORKER_MASK"]] != expected_worker_mask:
            raise RuntimeError(
                f"{case.name}: worker mask oracle failed: "
                f"{words[rec['WORKER_MASK']]} != {expected_worker_mask}"
            )
        controls = (
            words[rec["CONTROL_FINAL"]] & 0xFFFFFFFF,
            words[rec["CONTROL_FINAL"]] >> 32,
        )
        invalid_controls = tuple(
            control
            for control in controls
            if control & 0xFF or not control & 0x100
        )
        if invalid_controls:
            raise RuntimeError(
                f"{case.name}: final worker control oracle failed: "
                f"{controls}"
            )
        lane_blocking_deltas = (
            words[rec["LANE_A_WORKER_BLOCKING_DELTA"]],
            words[rec["LANE_B_WORKER_BLOCKING_DELTA"]],
        )
        engine_blocking_deltas = (
            words[
                rec[
                    f"{catalog.ENGINE_NAMES[case.engine_a]}_BLOCKING_DELTA"
                ]
            ],
            words[
                rec[
                    f"{catalog.ENGINE_NAMES[case.engine_b]}_BLOCKING_DELTA"
                ]
            ],
        )
        if any(
            lane > engine
            for lane, engine in zip(
                lane_blocking_deltas,
                engine_blocking_deltas,
                strict=True,
            )
        ):
            raise RuntimeError(
                f"{case.name}: per-worker blocking counters exceed "
                f"engine totals: {lane_blocking_deltas} > "
                f"{engine_blocking_deltas}"
            )
        parallel = {
            "worker_instruction_deltas": list(lane_instruction_deltas),
            "worker_blocking_deltas": list(lane_blocking_deltas),
            "worker_mask": words[rec["WORKER_MASK"]],
            "final_controls": list(controls),
        }

    for begin, end in built.exact_ranges:
        if raw[begin:end] != built.expected_output[begin:end]:
            mismatch = next(
                index
                for index in range(begin, end)
                if raw[index] != built.expected_output[index]
            )
            raise RuntimeError(
                f"{case.name}: exact result differs at byte {mismatch}"
            )

    mutable = bytearray(raw)
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [catalog.RESOURCE_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    for begin, end in built.allowed_ranges:
        mutable[begin:end] = bytes([catalog.RESOURCE_CANARY]) * (end - begin)
    if mutable != bytes([catalog.RESOURCE_CANARY]) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != catalog.RESOURCE_CANARY
        )
        raise RuntimeError(
            f"{case.name}: output changed outside bounded ranges at {mismatch}"
        )

    dump_ranges = set(catalog.spm_dump_ranges(case))
    observed = b"".join(
        raw[begin:end]
        for begin, end in built.allowed_ranges
        if (begin, end) not in dump_ranges
    )
    if not case.is_exact and all(
        value == catalog.RESOURCE_CANARY for value in observed
    ):
        raise RuntimeError(
            f"{case.name}: observation ranges remained entirely canary"
        )
    observation: dict[str, object] = {
        "case": case.as_dict(),
        "sample": sample,
        "correctness": "exact-result+full-spm-dump-guard",
        "result_sha256": hashlib.sha256(observed).hexdigest(),
        "plan_cycles": words[rec["PLAN_CYCLES"]],
        "pmu": {
            "full_execution": words[rec["FULL_EXEC_DELTA"]],
            "ct_execution": words[rec["CT_EXEC_DELTA"]],
            "ne_execution": words[rec["NE_EXEC_DELTA"]],
            "rdma_execution": words[rec["RDMA_EXEC_DELTA"]],
            "wdma_execution": words[rec["WDMA_EXEC_DELTA"]],
            "tdma_execution": words[rec["TDMA_EXEC_DELTA"]],
            "ct_blocking": words[rec["CT_BLOCKING_DELTA"]],
            "ne_blocking": words[rec["NE_BLOCKING_DELTA"]],
            "rdma_blocking": words[rec["RDMA_BLOCKING_DELTA"]],
            "wdma_blocking": words[rec["WDMA_BLOCKING_DELTA"]],
            "tdma_blocking": words[rec["TDMA_BLOCKING_DELTA"]],
        },
    }
    if parallel is not None:
        observation["parallel_pair"] = parallel
    return observation


def _validate_conflict_row_meta(
    raw: bytes,
    invocation: catalog.ConflictEquivalenceInvocation,
    case: catalog.MemoryCase,
    row: int,
) -> dict[str, int]:
    begin = (
        catalog.CONFLICT_RECORD_META_WORD
        + row * catalog.CONFLICT_RECORD_META_STRIDE_WORDS
    ) * 8
    words = struct.unpack_from(
        f"<{catalog.CONFLICT_RECORD_META_WORDS}Q", raw, begin
    )
    rec = catalog.CONFLICT_REC
    expected = {
        "MAGIC": catalog.CONFLICT_RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.CONFLICT_SCHEMA << 32
        ) | catalog.CONFLICT_RECORD_META_WORDS,
        "STATUS": 0,
        "COORDINATE": invocation.pair.coordinate_id,
        "INNER_CASE": case.case_id,
        "SCHEDULE": case.schedule,
        "ISSUE_ORDER": case.issue_order,
        "SAMPLE": invocation.sample,
        "EXECUTION_ORDINAL": (
            0
            if case.schedule == invocation.first_schedule
            else 1
        ),
        "FIRST_SCHEDULE": invocation.first_schedule,
        "SPM_A": case.spm_a,
        "SPM_B": case.spm_b,
        "WINDOW_FLAGS": catalog.CONFLICT_WINDOW_FLAGS,
        "RECORD_GUARD": catalog.CONFLICT_RECORD_GUARD,
    }
    failures = {
        key: (words[rec[key]], value)
        for key, value in expected.items()
        if words[rec[key]] != value
    }
    if failures:
        raise RuntimeError(
            f"{case.name}: conflict row metadata failed: {failures}"
        )

    request_ddr = words[rec["REQUEST_DDR"]]
    payload_ddr = words[rec["PAYLOAD_DDR"]]
    output_ddr = words[rec["OUTPUT_DDR"]]
    request_word = (
        catalog.CONFLICT_SERIAL_REQUEST_WORD
        if row == 0
        else catalog.CONFLICT_WINDOW_REQUEST_WORD
    )
    output_offset = (
        catalog.CONFLICT_SERIAL_OUTPUT_OFFSET
        if row == 0
        else catalog.CONFLICT_WINDOW_OUTPUT_OFFSET
    )
    row_output_ddr = output_ddr + output_offset
    address_expected = {
        "INNER_REQUEST_DDR": request_ddr + request_word * 8,
        "ROW_OUTPUT_DDR": row_output_ddr,
        "CT_INPUT0_DDR": payload_ddr + catalog.PAYLOAD_DATA_OFFSET,
        "CT_INPUT1_DDR": (
            payload_ddr + catalog.PAYLOAD_DATA_OFFSET + 4096
        ),
        "RDMA_INPUT_DDR": (
            payload_ddr + catalog.PAYLOAD_DATA_OFFSET + 16384 + 8192
        ),
        "RESULT_A_DDR": row_output_ddr + catalog.OUTPUT_DATA_OFFSET,
        "RESULT_B_DDR": (
            row_output_ddr
            + catalog.OUTPUT_DATA_OFFSET
            + case.descriptor.compact_bytes
        ),
    }
    address_failures = {
        key: (words[rec[key]], value)
        for key, value in address_expected.items()
        if words[rec[key]] != value
    }
    if address_failures:
        raise RuntimeError(
            f"{case.name}: actual DDR address echo failed: "
            f"{address_failures}"
        )
    return {
        key.lower(): words[rec[key]]
        for key in (
            "REQUEST_DDR",
            "INNER_REQUEST_DDR",
            "PAYLOAD_DDR",
            "OUTPUT_DDR",
            "ROW_OUTPUT_DDR",
            "CT_INPUT0_DDR",
            "CT_INPUT1_DDR",
            "RDMA_INPUT_DDR",
            "RESULT_A_DDR",
            "RESULT_B_DDR",
        )
    }


def validate_conflict_equivalence_output(
    path: pathlib.Path,
    invocation: catalog.ConflictEquivalenceInvocation,
) -> dict[str, dict[str, object]]:
    raw = path.read_bytes()
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(
            f"{invocation.pair.name}: invalid conflict output resource size"
        )
    rows: dict[str, dict[str, object]] = {}
    mutable = bytearray(raw)
    for row, (name, case, built, output_offset) in enumerate(
        (
            (
                "serial",
                invocation.pair.serial,
                invocation.serial_built,
                catalog.CONFLICT_SERIAL_OUTPUT_OFFSET,
            ),
            (
                "window",
                invocation.pair.window,
                invocation.window_built,
                catalog.CONFLICT_WINDOW_OUTPUT_OFFSET,
            ),
        )
    ):
        row_end = output_offset + catalog.CONFLICT_OUTPUT_ROW_BYTES
        synthetic = bytearray(
            [catalog.RESOURCE_CANARY] * catalog.RESOURCE_BYTES
        )
        synthetic[: catalog.CONFLICT_OUTPUT_ROW_BYTES] = raw[
            output_offset:row_end
        ]
        observation = validate_output_bytes(
            bytes(synthetic), case, built, invocation.sample
        )
        actual_addresses = _validate_conflict_row_meta(
            raw, invocation, case, row
        )
        observation["actual_addresses"] = actual_addresses
        observation["execution_ordinal"] = (
            0
            if case.schedule == invocation.first_schedule
            else 1
        )
        rows[name] = observation
        mutable[
            output_offset :
            output_offset + catalog.RECORD_WORDS * 8
        ] = bytes([catalog.RESOURCE_CANARY]) * (
            catalog.RECORD_WORDS * 8
        )
        for begin, end in built.allowed_ranges:
            mutable[
                output_offset + begin : output_offset + end
            ] = bytes([catalog.RESOURCE_CANARY]) * (end - begin)
        meta_begin = (
            catalog.CONFLICT_RECORD_META_WORD
            + row * catalog.CONFLICT_RECORD_META_STRIDE_WORDS
        ) * 8
        meta_end = (
            meta_begin + catalog.CONFLICT_RECORD_META_WORDS * 8
        )
        mutable[meta_begin:meta_end] = bytes(
            [catalog.RESOURCE_CANARY]
        ) * (meta_end - meta_begin)

    serial_addresses = rows["serial"]["actual_addresses"]
    window_addresses = rows["window"]["actual_addresses"]
    assert isinstance(serial_addresses, dict)
    assert isinstance(window_addresses, dict)
    for key in ("request_ddr", "payload_ddr", "output_ddr"):
        if serial_addresses[key] != window_addresses[key]:
            raise RuntimeError(
                f"{invocation.pair.name}: paired rows do not share "
                f"the same {key} allocation"
            )
    allocation_bases = tuple(
        int(serial_addresses[key])
        for key in ("request_ddr", "payload_ddr", "output_ddr")
    )
    if (
        any(base == 0 or base % 256 != 0 for base in allocation_bases)
        or len(set(allocation_bases)) != len(allocation_bases)
    ):
        raise RuntimeError(
            f"{invocation.pair.name}: actual resource allocations are "
            "zero, unaligned, or aliased"
        )
    if mutable != bytes(
        [catalog.RESOURCE_CANARY]
    ) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != catalog.RESOURCE_CANARY
        )
        raise RuntimeError(
            f"{invocation.pair.name}: conflict output changed outside "
            f"typed rows at {mismatch}"
        )
    return rows


def execute_conflict_equivalence(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
) -> None:
    raw_dir = args.work_dir / "raw" / "conflict-equivalence"
    raw_dir.mkdir(parents=True)
    observations: dict[
        str, list[dict[str, dict[str, object]]]
    ] = {}
    for pair in catalog.CONFLICT_EQUIVALENCE_PAIRS:
        pair_rows = observations.setdefault(pair.name, [])
        for sample in range(catalog.CONFLICT_SAMPLES):
            invocation = catalog.build_conflict_equivalence_invocation(
                pair, sample
            )
            prefix = f"{pair.name}.sample-{sample}"
            request = raw_dir / f"{prefix}.request.raw"
            payload = raw_dir / f"{prefix}.payload.raw"
            output = raw_dir / f"{prefix}.output.raw"
            request.write_bytes(invocation.request)
            payload.write_bytes(invocation.payload)
            result = package_support.run(
                package_support.board_command(
                    args, package, resource_ids, request, payload, output
                ),
                timeout_seconds=(
                    args.completion_timeout_ms / 1000.0 + 30.0
                ),
            )
            required = {
                "board_stage: completion",
                "board_stage: device-to-host",
                "board_stage: cleanup",
                "board_execution: true",
            }
            if not required.issubset(set(result.stdout.splitlines())):
                raise RuntimeError(
                    f"{prefix}: wafer-run omitted lifecycle evidence"
                )
            rows = validate_conflict_equivalence_output(
                output, invocation
            )
            pair_rows.append(rows)
            for schedule in ("serial", "window"):
                print(
                    "memory_descriptor_calibration: "
                    + json.dumps(rows[schedule], sort_keys=True)
                )
    emit_conflict_equivalence_summary(observations)


def emit_conflict_equivalence_summary(
    observations: dict[
        str, list[dict[str, dict[str, object]]]
    ],
) -> None:
    metric_names = (
        "full_execution",
        "ct_execution",
        "rdma_execution",
        "ct_blocking",
        "rdma_blocking",
        "plan_cycles",
    )

    def metric(
        row: dict[str, object], name: str
    ) -> int:
        if name == "plan_cycles":
            return int(row["plan_cycles"])
        pmu = row["pmu"]
        assert isinstance(pmu, dict)
        return int(pmu[name])

    paired_summaries: list[dict[str, object]] = []
    for pair in catalog.CONFLICT_EQUIVALENCE_PAIRS:
        invocations = observations.get(pair.name, [])
        if len(invocations) != catalog.CONFLICT_SAMPLES:
            raise RuntimeError(
                f"{pair.name}: incomplete same-invocation sample set"
            )
        medians = {
            schedule: {
                name: statistics.median(
                    metric(rows[schedule], name)
                    for rows in invocations
                )
                for name in metric_names
            }
            for schedule in ("serial", "window")
        }
        paired_deltas = {
            name: statistics.median(
                metric(rows["window"], name)
                - metric(rows["serial"], name)
                for rows in invocations
            )
            for name in metric_names
        }
        allocation_evidence = []
        for rows in invocations:
            serial_addresses = rows["serial"]["actual_addresses"]
            window_addresses = rows["window"]["actual_addresses"]
            assert isinstance(serial_addresses, dict)
            assert isinstance(window_addresses, dict)
            allocation_evidence.append(
                {
                    "request_ddr": serial_addresses["request_ddr"],
                    "payload_ddr": serial_addresses["payload_ddr"],
                    "output_ddr": serial_addresses["output_ddr"],
                    "same_allocation": all(
                        serial_addresses[key] == window_addresses[key]
                        for key in (
                            "request_ddr",
                            "payload_ddr",
                            "output_ddr",
                        )
                    ),
                }
            )
        summary: dict[str, object] = {
            "engine_pair": ["CT", "RDMA"],
            "relative_spm_offset": pair.serial.spm_b - pair.serial.spm_a,
            "spm_base": pair.serial.spm_a,
            "spm_base_translation": pair.translation,
            "spm_base_phase_mod_256": pair.phase,
            "transfer_bytes": pair.transfer_bytes,
            "issue_order": (
                "a-b" if pair.issue_order == 0 else "b-a"
            ),
            "samples_per_cell": catalog.CONFLICT_SAMPLES,
            "correctness": (
                "all-exact-result+full-spm-dump-guard+instruction-count"
            ),
            "pmu_window": (
                "pair-only; setup before PMU and readback after PMU"
            ),
            "pmu_medians": medians,
            "window_minus_serial": paired_deltas,
            "same_invocation_allocations": allocation_evidence,
            "state": "raw-paired-same-invocation-observation",
            "interpretation": (
                "actual resource and instruction addresses are echoed; "
                "no physical bank identity is inferred"
            ),
        }
        paired_summaries.append(summary)
        print(
            "spm_bank_pair_repeat_summary: "
            + json.dumps(summary, sort_keys=True)
        )

    expected_translations = (
        0,
        *catalog.SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS,
    )
    by_coordinate = {
        (
            int(summary["spm_base_phase_mod_256"]),
            int(summary["spm_base_translation"]),
            int(summary["transfer_bytes"]),
            0 if summary["issue_order"] == "a-b" else 1,
        ): summary
        for summary in paired_summaries
    }
    phase_rows = []
    all_consistent = True
    all_informative = True
    for phase in catalog.SPM_PARALLEL_ALIGNMENT_PHASES:
        for transfer_bytes in catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS:
            for issue_order in (
                catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
            ):
                cells = {
                    translation: by_coordinate.get(
                        (
                            phase,
                            translation,
                            transfer_bytes,
                            issue_order,
                        )
                    )
                    for translation in expected_translations
                }
                missing = [
                    translation
                    for translation, summary in cells.items()
                    if summary is None
                ]
                directions = {
                    metric_name: [
                        (
                            0
                            if int(
                                summary["window_minus_serial"][
                                    metric_name
                                ]
                            )
                            == 0
                            else (
                                1
                                if int(
                                    summary["window_minus_serial"][
                                        metric_name
                                    ]
                                )
                                > 0
                                else -1
                            )
                        )
                        for summary in cells.values()
                        if summary is not None
                    ]
                    for metric_name in (
                        "plan_cycles",
                        "full_execution",
                        "ct_blocking",
                        "rdma_blocking",
                    )
                }
                consistent = not missing and all(
                    len(set(values)) == 1
                    for values in directions.values()
                )
                informative = any(
                    directions[metric_name]
                    and directions[metric_name][0] != 0
                    for metric_name in ("plan_cycles", "full_execution")
                )
                all_consistent &= consistent
                all_informative &= informative
                phase_rows.append(
                    {
                        "base_phase_mod_256": phase,
                        "transfer_bytes": transfer_bytes,
                        "issue_order": (
                            "a-b" if issue_order == 0 else "b-a"
                        ),
                        "missing_translations": missing,
                        "window_minus_serial_directions": directions,
                        "proxy_direction_consistent": consistent,
                        "nonzero_cost_signal": informative,
                        "promotable_proxy_pattern": (
                            consistent and informative
                        ),
                    }
                )
    print(
        "spm_conflict_equivalence_summary: "
        + json.dumps(
            {
                "engine_pair": ["CT", "RDMA"],
                "relative_spm_offset": 8192,
                "base_translations": list(expected_translations),
                "transfer_bytes": list(
                    catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS
                ),
                "issue_orders": ["a-b", "b-a"],
                "samples_per_cell": catalog.CONFLICT_SAMPLES,
                "invocation_contract": (
                    "serial+window share request/payload/output "
                    "allocations per sample"
                ),
                "correctness": (
                    "all-exact-result+full-spm-dump-guard+"
                    "instruction-count+actual-address-echo"
                ),
                "phase_rows": phase_rows,
                "state": (
                    "heldout-proxy-direction-consistent"
                    if all_consistent and all_informative
                    else (
                        "consistent-but-no-nonzero-cost-signal"
                        if all_consistent
                        else "inconclusive-or-direction-flip"
                    )
                ),
                "compiler_use": "no-bank-coloring",
            },
            sort_keys=True,
        )
    )


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.MemoryCase],
) -> None:
    cases = tuple(cases)
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    observations: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        case_observations = observations.setdefault(case.name, [])
        for sample in range(case.repetitions):
            built = catalog.build_case_payload(case, sample)
            prefix = f"{case.name}.sample-{sample}"
            request = raw_dir / f"{prefix}.request.raw"
            payload = raw_dir / f"{prefix}.payload.raw"
            output = raw_dir / f"{prefix}.output.raw"
            request.write_bytes(built.request)
            payload.write_bytes(built.payload)
            result = package_support.run(
                package_support.board_command(
                    args, package, resource_ids, request, payload, output
                ),
                timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0,
            )
            required = {
                "board_stage: completion",
                "board_stage: device-to-host",
                "board_stage: cleanup",
                "board_execution: true",
            }
            if not required.issubset(set(result.stdout.splitlines())):
                raise RuntimeError(
                    f"{prefix}: wafer-run omitted lifecycle evidence"
                )
            observation = validate_output(output, case, built, sample)
            case_observations.append(observation)
            print(
                "memory_descriptor_calibration: "
                + json.dumps(observation, sort_keys=True)
            )

    pair_keys = sorted(
        {
            (
                case.engine_a,
                case.engine_b,
                case.spm_b - case.spm_a,
                case.spm_a,
                case.descriptor.compact_bytes,
                case.issue_order,
            )
            for case in cases
            if case.kind == catalog.KIND_ENGINE_PAIR
        }
    )
    metric_names = (
        "full_execution",
        "ct_execution",
        "ne_execution",
        "rdma_execution",
        "wdma_execution",
        "tdma_execution",
        "ct_blocking",
        "ne_blocking",
        "rdma_blocking",
        "wdma_blocking",
        "tdma_blocking",
        "plan_cycles",
    )
    paired_summaries: list[dict[str, object]] = []
    for (
        engine_a,
        engine_b,
        relative_offset,
        spm_base,
        transfer_bytes,
        issue_order,
    ) in pair_keys:
        schedule_cases = {
            case.schedule: case
            for case in cases
            if case.kind == catalog.KIND_ENGINE_PAIR
            and case.engine_a == engine_a
            and case.engine_b == engine_b
            and case.spm_b - case.spm_a == relative_offset
            and case.spm_a == spm_base
            and case.descriptor.compact_bytes == transfer_bytes
            and case.issue_order == issue_order
        }
        schedule_medians: dict[str, dict[str, int | float]] = {}
        missing: list[str] = []
        for schedule, name in (
            (catalog.SCHEDULE_SERIAL, "serial"),
            (catalog.SCHEDULE_WINDOW, "window"),
        ):
            case = schedule_cases.get(schedule)
            if case is None:
                missing.append(name)
                continue
            rows = observations[case.name]
            if len(rows) != case.repetitions:
                raise RuntimeError(
                    f"{case.name}: incomplete repeated PMU sample set"
                )
            schedule_medians[name] = {
                metric: statistics.median(
                    int(
                        row["plan_cycles"]
                        if metric == "plan_cycles"
                        else row["pmu"][metric]
                    )
                    for row in rows
                )
                for metric in metric_names
            }
        summary: dict[str, object] = {
            "engine_pair": [
                catalog.ENGINE_NAMES[engine_a],
                catalog.ENGINE_NAMES[engine_b],
            ],
            "relative_spm_offset": relative_offset,
            "spm_base": spm_base,
            "spm_base_translation": (
                spm_base - catalog.SPM_BASE - spm_base % 256
            ),
            "spm_base_phase_mod_256": spm_base % 256,
            "transfer_bytes": transfer_bytes,
            "issue_order": "a-b" if issue_order == 0 else "b-a",
            "samples_per_cell": catalog.SPM_BANK_PMU_REPETITIONS,
            "correctness": "all-exact-result+full-spm-dump-guard",
            "pmu_medians": schedule_medians,
            "interpretation": (
                "raw repeated paired control; no bank formula or bank name "
                "is inferred"
            ),
        }
        if missing:
            summary["state"] = "inconclusive"
            summary["missing_controls"] = missing
        else:
            summary["state"] = "raw-paired-observation"
            engine_metrics = (
                f"{catalog.ENGINE_NAMES[engine_a].lower()}_execution",
                f"{catalog.ENGINE_NAMES[engine_b].lower()}_execution",
            )
            summary["pairwise_excess"] = {
                schedule: (
                    medians[engine_metrics[0]]
                    + medians[engine_metrics[1]]
                    - medians["full_execution"]
                )
                for schedule, medians in schedule_medians.items()
            }
            serial_plan = schedule_medians["serial"]["plan_cycles"]
            window_plan = schedule_medians["window"]["plan_cycles"]
            serial_full = schedule_medians["serial"]["full_execution"]
            window_full = schedule_medians["window"]["full_execution"]
            summary["serial_window_ratio"] = {
                "plan_cycles": serial_plan / window_plan,
                "full_execution": (
                    serial_full / window_full if window_full else None
                ),
            }
            summary["window_minus_serial"] = {
                metric: (
                    schedule_medians["window"][metric]
                    - schedule_medians["serial"][metric]
                )
                for metric in metric_names
            }
            paired_summaries.append(summary)
        print(
            "spm_bank_pair_repeat_summary: "
            + json.dumps(summary, sort_keys=True)
        )

    parallel_pair_keys = sorted(
        {
            (
                case.engine_a,
                case.engine_b,
                case.descriptor.compact_bytes,
                case.worker_a,
                case.worker_b,
                case.relation,
                case.issue_order,
            )
            for case in cases
            if case.kind == catalog.KIND_PARALLEL_PAIR
        }
    )
    for (
        engine_a,
        engine_b,
        transfer_bytes,
        worker_a,
        worker_b,
        relation,
        issue_order,
    ) in parallel_pair_keys:
        schedule_cases = {
            case.schedule: case
            for case in cases
            if case.kind == catalog.KIND_PARALLEL_PAIR
            and case.engine_a == engine_a
            and case.engine_b == engine_b
            and case.descriptor.compact_bytes == transfer_bytes
            and case.worker_a == worker_a
            and case.worker_b == worker_b
            and case.relation == relation
            and case.issue_order == issue_order
        }
        schedule_medians: dict[str, dict[str, int | float]] = {}
        schedule_contracts: dict[str, dict[str, int]] = {}
        missing: list[str] = []
        for schedule, name in (
            (catalog.SCHEDULE_SERIAL, "serial"),
            (catalog.SCHEDULE_WINDOW, "window"),
        ):
            case = schedule_cases.get(schedule)
            if case is None:
                missing.append(name)
                continue
            rows = observations[case.name]
            if len(rows) != case.repetitions:
                raise RuntimeError(
                    f"{case.name}: incomplete repeated PMU sample set"
                )
            schedule_medians[name] = {
                metric: statistics.median(
                    int(
                        row["plan_cycles"]
                        if metric == "plan_cycles"
                        else row["pmu"][metric]
                    )
                    for row in rows
                )
                for metric in metric_names
            }
            for lane in range(2):
                schedule_medians[name][
                    f"lane_{lane}_worker_blocking"
                ] = statistics.median(
                    int(row["parallel_pair"]["worker_blocking_deltas"][lane])
                    for row in rows
                )
                schedule_medians[name][
                    f"lane_{lane}_worker_instructions"
                ] = statistics.median(
                    int(
                        row["parallel_pair"][
                            "worker_instruction_deltas"
                        ][lane]
                    )
                    for row in rows
                )
            schedule_contracts[name] = {
                "rounds": case.rounds,
                "buffer_count": case.buffer_count,
            }
        summary = {
            "engine_pair": [
                catalog.ENGINE_NAMES[engine_a],
                catalog.ENGINE_NAMES[engine_b],
            ],
            "transfer_bytes": transfer_bytes,
            "workers": [worker_a, worker_b],
            "relation": catalog.RELATION_NAMES[relation],
            "issue_order": "a-b" if issue_order == 0 else "b-a",
            "samples_per_cell": catalog.SPM_BANK_PMU_REPETITIONS,
            "correctness": (
                "all-exact-result+two-lane-guard+worker-control"
            ),
            "contracts": schedule_contracts,
            "pmu_medians": schedule_medians,
            "interpretation": (
                "raw serial/window repeated control for sustained "
                "multi-engine issue; no overlap claim is inferred"
            ),
        }
        if missing:
            summary["state"] = "inconclusive"
            summary["missing_controls"] = missing
        else:
            summary["state"] = "raw-paired-observation"
            summary["window_minus_serial"] = {
                metric: (
                    schedule_medians["window"][metric]
                    - schedule_medians["serial"][metric]
                )
                for metric in schedule_medians["serial"]
            }
            serial_plan = schedule_medians["serial"]["plan_cycles"]
            window_plan = schedule_medians["window"]["plan_cycles"]
            summary["serial_window_plan_cycles_ratio"] = (
                serial_plan / window_plan
            )
        print(
            "parallel_pair_repeat_summary: "
            + json.dumps(summary, sort_keys=True)
        )

    def emit_axis_summary(
        label: str,
        expected_axis: tuple[int, ...],
        axis_name: str,
        selector: Callable[[dict[str, object]], bool],
    ) -> None:
        selected_cells = [
            summary
            for summary in paired_summaries
            if selector(summary)
        ]
        by_axis = {
            int(summary[axis_name]): summary for summary in selected_cells
        }
        missing = sorted(set(expected_axis) - set(by_axis))
        rows = []
        for axis in expected_axis:
            summary = by_axis.get(axis)
            if summary is None:
                continue
            medians = summary["pmu_medians"]
            assert isinstance(medians, dict)
            window = medians["window"]
            assert isinstance(window, dict)
            ratios = summary["serial_window_ratio"]
            assert isinstance(ratios, dict)
            excess = summary["pairwise_excess"]
            assert isinstance(excess, dict)
            rows.append(
                {
                    axis_name: axis,
                    "window_plan_cycles": window["plan_cycles"],
                    "window_full_execution": window["full_execution"],
                    "window_ct_execution": window["ct_execution"],
                    "window_rdma_execution": window["rdma_execution"],
                    "window_ct_blocking": window["ct_blocking"],
                    "window_rdma_blocking": window["rdma_blocking"],
                    "window_pairwise_excess": excess["window"],
                    "serial_window_plan_cycles_ratio": ratios["plan_cycles"],
                }
            )
        report = {
            "engine_pair": ["CT", "RDMA"],
            "expected_axis": list(expected_axis),
            "samples_per_cell": catalog.SPM_BANK_PMU_REPETITIONS,
            "correctness": "all-exact-result+full-spm-dump-guard",
            "rows": rows,
            "missing": missing,
            "state": "complete-raw-classification-input" if not missing
            else "inconclusive-missing-controls",
            "classification": (
                "compare repeated execution/blocking/FU/plan medians; "
                "do not assign bank identities without direct mapping evidence"
            ),
        }
        print(label + ": " + json.dumps(report, sort_keys=True))

    emit_axis_summary(
        "spm_bank_offset_sweep_summary",
        catalog.SPM_BANK_PERIOD_RELATIVE_OFFSETS,
        "relative_spm_offset",
        lambda summary: (
            summary["engine_pair"] == ["CT", "RDMA"]
            and summary["spm_base_translation"] == 0
            and summary["spm_base_phase_mod_256"] == 0
            and summary["transfer_bytes"] == 256
            and summary["issue_order"] == "a-b"
            and summary["relative_spm_offset"]
            in catalog.SPM_BANK_PERIOD_RELATIVE_OFFSETS
        ),
    )
    emit_axis_summary(
        "spm_alignment_phase_sweep_summary",
        catalog.SPM_PARALLEL_ALIGNMENT_PHASES,
        "spm_base_phase_mod_256",
        lambda summary: (
            summary["engine_pair"] == ["CT", "RDMA"]
            and summary["spm_base_translation"] == 0
            and summary["relative_spm_offset"] == 8192
            and summary["transfer_bytes"] == 256
            and summary["issue_order"] == "a-b"
            and summary["spm_base_phase_mod_256"]
            in catalog.SPM_PARALLEL_ALIGNMENT_PHASES
        ),
    )

    if any(
        case in catalog.PENDING_CONFLICT_EQUIVALENCE_CASES
        for case in cases
    ):
        expected_translations = (
            0,
            *catalog.SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS,
        )
        metrics = (
            "plan_cycles",
            "full_execution",
            "ct_blocking",
            "rdma_blocking",
        )
        by_coordinate = {
            (
                int(summary["spm_base_phase_mod_256"]),
                int(summary["spm_base_translation"]),
                int(summary["transfer_bytes"]),
                0 if summary["issue_order"] == "a-b" else 1,
            ): summary
            for summary in paired_summaries
            if (
                summary["engine_pair"] == ["CT", "RDMA"]
                and summary["relative_spm_offset"] == 8192
                and summary["spm_base_phase_mod_256"]
                in catalog.SPM_PARALLEL_ALIGNMENT_PHASES
                and summary["spm_base_translation"]
                in expected_translations
                and summary["transfer_bytes"]
                in catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS
                and (
                    0 if summary["issue_order"] == "a-b" else 1
                )
                in catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
            )
        }
        phase_rows = []
        all_consistent = True
        for phase in catalog.SPM_PARALLEL_ALIGNMENT_PHASES:
            for transfer_bytes in (
                catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS
            ):
                for issue_order in (
                    catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
                ):
                    cells = {
                        translation: by_coordinate.get(
                            (
                                phase,
                                translation,
                                transfer_bytes,
                                issue_order,
                            )
                        )
                        for translation in expected_translations
                    }
                    missing = [
                        translation
                        for translation, summary in cells.items()
                        if summary is None
                    ]
                    directions: dict[str, list[int]] = {}
                    for metric in metrics:
                        directions[metric] = [
                            (
                                0
                                if int(
                                    summary["window_minus_serial"][metric]
                                )
                                == 0
                                else (
                                    1
                                    if int(
                                        summary["window_minus_serial"][
                                            metric
                                        ]
                                    )
                                    > 0
                                    else -1
                                )
                            )
                            for summary in cells.values()
                            if summary is not None
                        ]
                    consistent = not missing and all(
                        len(set(values)) == 1
                        for values in directions.values()
                    )
                    all_consistent &= consistent
                    phase_rows.append(
                        {
                            "base_phase_mod_256": phase,
                            "transfer_bytes": transfer_bytes,
                            "issue_order": (
                                "a-b" if issue_order == 0 else "b-a"
                            ),
                            "missing_translations": missing,
                            "window_minus_serial_directions": directions,
                            "proxy_direction_consistent": consistent,
                        }
                    )
        print(
            "spm_conflict_equivalence_summary: "
            + json.dumps(
                {
                    "engine_pair": ["CT", "RDMA"],
                    "relative_spm_offset": 8192,
                    "base_translations": list(expected_translations),
                    "transfer_bytes": list(
                        catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS
                    ),
                    "issue_orders": ["a-b", "b-a"],
                    "samples_per_cell": catalog.SPM_BANK_PMU_REPETITIONS,
                    "correctness": (
                        "all-exact-result+full-spm-dump-guard"
                    ),
                    "phase_rows": phase_rows,
                    "state": (
                        "heldout-proxy-direction-consistent"
                        if all_consistent
                        else "inconclusive-or-direction-flip"
                    ),
                    "compiler_use": "no-bank-coloring",
                },
                sort_keys=True,
            )
        )


def main() -> int:
    args = parse_args()
    if args.list_cases:
        print(
            json.dumps(
                {
                    "cases": [case.as_dict() for case in catalog.CATALOG],
                    "pending_cases": [
                        case.as_dict() for case in catalog.PENDING_CASES
                    ],
                    "pending_conflict_equivalence_invocations": [
                        pair.as_dict()
                        for pair in catalog.CONFLICT_EQUIVALENCE_PAIRS
                    ],
                    "calibration_leaf_bindings": {
                        key: [case.name for case in cases]
                        for key, cases
                        in catalog.CALIBRATION_LEAF_BINDINGS.items()
                    },
                    "static_boundaries": {
                        "configurable_burst_knob": (
                            "not exposed by the owned RDMA/WDMA CRT; cases "
                            "exercise default burst boundaries only"
                        ),
                        "same_session_cache_stale": (
                            "requires an owner-backed multi-phase runtime "
                            "session and is not dispatched by this package"
                        ),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    configure_package_support()
    package_support.require_build_args(args)
    selected = select_cases(args)
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "wafer_board_memory_descriptor_calibration_probe_test: "
                "hardware execution is not armed; set "
                "WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        package_support.require_board_args(args)
    source = package_support.write_source_program(args)
    package = package_support.compile_seed_package(args, source)
    module_path, resource_ids = package_support.locate_bindings(package)
    package_support.build_probe(args, package, module_path)
    package_support.verify_no_card(args, package)
    if args.no_card:
        return 0
    if args.conflict_equivalence:
        execute_conflict_equivalence(args, package, resource_ids)
    else:
        execute_cases(args, package, resource_ids, selected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            "wafer_board_memory_descriptor_calibration_probe_test: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
