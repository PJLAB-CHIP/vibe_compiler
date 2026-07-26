#!/usr/bin/env python3
"""Build one package and execute four-direction cache/coherence calibration."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import struct
import sys

import wafer_board_instruction_family_probe_test as package_support
import wafer_cache_coherence_calibration_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_cache_coherence_calibration_probe.c"
MODULE = """\
module {
  func.func @main(
      %request: tensor<16384xf32>,
      %payload: tensor<16384xf32>) -> tensor<16384xf32> {
    %result = stablehlo.add %request, %payload : tensor<16384xf32>
    return %result : tensor<16384xf32>
  }
}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [16384], "dtype": "float32", "dynamic_dims": []},
        {"shape": [16384], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [16384], "dtype": "float32", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request"},
        {"type_": "input_arg", "position": 1, "name": "payload"},
    ],
    "unused_inputs": [],
}
EXPECTED_CACHE_MASK = {
    0: 1,
    1: 1 | 2 | 8,
    2: 4 | 8,
    3: 8,
}
PAIR_KIND_CODES = {None: 0, "rdma-rdma": 1, "wdma-wdma": 2, "rdma-wdma": 3}
SCHEDULE_CODES = {None: 0, "serial": 1, "window": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument(
        "--ddr-conflict-equivalence",
        action="store_true",
        help=(
            "select the pending same-invocation DDR pair equivalence "
            "batches"
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


def select_cases(
    args: argparse.Namespace,
) -> tuple[catalog.CacheCoherenceCase, ...]:
    if args.ddr_conflict_equivalence:
        if args.selected_cases:
            raise RuntimeError(
                "--ddr-conflict-equivalence cannot be combined with --case"
            )
        return catalog.PENDING_DDR_CONFLICT_CASES
    if not args.selected_cases:
        return catalog.CATALOG
    unknown = sorted(
        set(args.selected_cases) - set(catalog.CASES_BY_NAME)
    )
    if unknown:
        raise RuntimeError(f"unknown cache/coherence cases: {unknown}")
    return tuple(
        catalog.CASES_BY_NAME[name] for name in args.selected_cases
    )


def _classify_before(
    case: catalog.CacheCoherenceCase,
    built: catalog.CasePayload,
    raw: bytes,
    mismatch_before: int,
) -> str:
    if case.kind in {
        "ddr-bank-pair",
        "ddr-conflict-equivalence",
    }:
        return "not-sampled"
    if case.case_id == 1:
        observed = raw[
            catalog.OUTPUT0_OFFSET :
            catalog.OUTPUT0_OFFSET + catalog.PAYLOAD_BYTES
        ]
        if observed == built.post_control_expected:
            return "new-visible-before-clean"
        if observed == built.input_pattern:
            return "old-visible-before-clean"
        return "mixed-or-other-before-clean"
    if case.case_id == 0:
        return "post-invalidate-exact" if mismatch_before == 0 else (
            "post-invalidate-mismatch"
        )
    if case.case_id == 2:
        return "new-visible-before-control" if mismatch_before == 0 else (
            "stale-or-mixed-before-control"
        )
    return "not-sampled"


def _validate_conflict_output(
    raw: bytes,
    case: catalog.CacheCoherenceCase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    words = struct.unpack_from(f"<{catalog.RECORD_WORDS}Q", raw)
    rec = catalog.REC
    expected_record = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "SAMPLE": sample,
        "PAYLOAD_BYTES": case.transfer_bytes,
        "EXPECTED_RDMA": case.expected_rdma,
        "EXPECTED_WDMA": case.expected_wdma,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT0_OFFSET": catalog.OUTPUT0_OFFSET,
        "OUTPUT1_OFFSET": catalog.OUTPUT1_OFFSET,
        "MISMATCH_BEFORE": 0,
        "MISMATCH_AFTER": 0,
        "SPM_GUARD_MISMATCHES": 0,
        "OUTPUT_GUARD_MISMATCHES": 0,
        "RDMA_INST_DELTA": case.expected_rdma,
        "WDMA_INST_DELTA": case.expected_wdma,
        "CACHE_CONTROL_MASK": 8,
        "PAIR_KIND": PAIR_KIND_CODES[case.pair_kind],
        "SCHEDULE": 0,
        "BANK_OFFSET": case.bank_offset,
        "MODE": catalog.CONFLICT_MODE,
        "TRANSFER_BYTES": case.transfer_bytes,
        "BASE_RELATION": case.base_relation,
        "BASE_TRANSLATION": case.base_translation,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    failures = {
        key: (words[rec[key]], expected)
        for key, expected in expected_record.items()
        if words[rec[key]] != expected
    }
    if failures:
        raise RuntimeError(
            f"{case.name}: conflict record oracle failed: {failures}"
        )
    base_a = words[rec["DDR_BASE_A"]]
    base_b = words[rec["DDR_BASE_B"]]
    address_a = words[rec["DDR_ADDRESS_A"]]
    address_b = words[rec["DDR_ADDRESS_B"]]
    offset_a = catalog.CONFLICT_DATA_BASE + case.base_translation
    offset_b = (
        offset_a + catalog.BANK_REGION_GAP + case.bank_offset
    )
    if (
        base_a == 0
        or base_b == 0
        or base_a % 256
        or base_b % 256
        or address_a != base_a + offset_a
        or address_b != base_b + offset_b
    ):
        raise RuntimeError(
            f"{case.name}: actual DDR base/address echo is invalid"
        )
    expected_same = (
        case.base_relation == catalog.CONFLICT_BASE_RELATION_SAME
    )
    if (base_a == base_b) != expected_same:
        raise RuntimeError(
            f"{case.name}: actual DDR allocation relation is wrong: "
            f"{base_a:#x}, {base_b:#x}"
        )
    if (
        case.pair_kind == "rdma-rdma"
        and words[rec["RDMA_EXEC_DELTA"]] == 0
    ) or (
        case.pair_kind == "wdma-wdma"
        and words[rec["WDMA_EXEC_DELTA"]] == 0
    ):
        raise RuntimeError(
            f"{case.name}: batch engine execution delta is zero"
        )

    expected_pair_counts = (
        (2, 0) if case.pair_kind == "rdma-rdma" else (0, 2)
    )
    rows: list[dict[str, int]] = []
    row_ordinal = 0
    for repetition in range(catalog.CONFLICT_REPETITIONS):
        for schedule in (1, 2):
            for issue_order in (0, 1):
                row_offset = (
                    catalog.RECORD_WORDS
                    + row_ordinal * catalog.CONFLICT_ROW_WORDS
                ) * 8
                row = struct.unpack_from(
                    f"<{catalog.CONFLICT_ROW_WORDS}Q",
                    raw,
                    row_offset,
                )
                control = (schedule - 1) * 2 + issue_order
                archive0 = (
                    catalog.CONFLICT_ARCHIVE_BASE
                    + control * catalog.CONFLICT_ARCHIVE_STRIDE
                    if case.pair_kind == "rdma-rdma"
                    else offset_a - catalog.SPM_GUARD_BYTES
                )
                archive1 = (
                    archive0
                    + case.transfer_bytes
                    + 2 * catalog.SPM_GUARD_BYTES
                    if case.pair_kind == "rdma-rdma"
                    else offset_b - catalog.SPM_GUARD_BYTES
                )
                expected_row = {
                    "MAGIC": catalog.CONFLICT_ROW_MAGIC,
                    "IDENTITY": (
                        repetition
                        | schedule << 8
                        | issue_order << 16
                    ),
                    "REPETITION": repetition,
                    "SCHEDULE": schedule,
                    "ISSUE_ORDER": issue_order,
                    "RDMA_INST_DELTA": expected_pair_counts[0],
                    "WDMA_INST_DELTA": expected_pair_counts[1],
                    "ARCHIVE0_OFFSET": archive0,
                    "ARCHIVE1_OFFSET": archive1,
                    "DDR_ADDRESS_A": address_a,
                    "DDR_ADDRESS_B": address_b,
                    "COMPLETED": 1,
                    "GUARD": catalog.CONFLICT_ROW_GUARD,
                    "RESULT_MISMATCHES": 0,
                    "SCHEDULE_POSITION": (
                        schedule - 1
                        if (repetition + issue_order) % 2 == 0
                        else 2 - schedule
                    ),
                }
                row_failures = {
                    key: (row[catalog.CONFLICT_ROW[key]], expected)
                    for key, expected in expected_row.items()
                    if row[catalog.CONFLICT_ROW[key]] != expected
                }
                execution_field = (
                    "RDMA_EXEC_DELTA"
                    if case.pair_kind == "rdma-rdma"
                    else "WDMA_EXEC_DELTA"
                )
                relevant_execution = row[
                    catalog.CONFLICT_ROW[execution_field]
                ]
                plan_cycles = row[
                    catalog.CONFLICT_ROW["PLAN_CYCLES"]
                ]
                if row_failures or relevant_execution == 0 or plan_cycles == 0:
                    raise RuntimeError(
                        f"{case.name}: conflict row {row_ordinal} "
                        f"oracle failed: {row_failures}, "
                        f"execution={relevant_execution}, "
                        f"plan_cycles={plan_cycles}"
                    )
                rows.append(
                    {
                        "repetition": repetition,
                        "schedule": schedule,
                        "issue_order": issue_order,
                        "plan_cycles": plan_cycles,
                        "rdma_execution": row[
                            catalog.CONFLICT_ROW["RDMA_EXEC_DELTA"]
                        ],
                        "wdma_execution": row[
                            catalog.CONFLICT_ROW["WDMA_EXEC_DELTA"]
                        ],
                        "schedule_position": row[
                            catalog.CONFLICT_ROW["SCHEDULE_POSITION"]
                        ],
                    }
                )
                row_ordinal += 1
    if row_ordinal != catalog.CONFLICT_BATCH_ROWS:
        raise RuntimeError(f"{case.name}: conflict batch row count drifted")

    mutable = bytearray(raw)
    mutable[: catalog.CONFLICT_RECORD_BYTES] = bytes(
        [catalog.OUTPUT_CANARY]
    ) * catalog.CONFLICT_RECORD_BYTES
    for begin, expected in built.expected_regions:
        end = begin + len(expected)
        if bytes(mutable[begin:end]) != expected:
            raise RuntimeError(
                f"{case.name}: conflict slot at {begin} is not exact"
            )
        mutable[begin:end] = bytes([catalog.OUTPUT_CANARY]) * (
            end - begin
        )
    if mutable != bytes([catalog.OUTPUT_CANARY]) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != catalog.OUTPUT_CANARY
        )
        raise RuntimeError(
            f"{case.name}: conflict output changed outside "
            f"record/slots at {mismatch}"
        )
    return {
        "case": case.as_dict(),
        "sample": sample,
        "actual_ddr": {
            "base_a": base_a,
            "base_b": base_b,
            "address_a": address_a,
            "address_b": address_b,
        },
        "batch_rows": rows,
        "correctness": "exact-two-slot+guards+pair-window-counts",
    }


def validate_output(
    path: pathlib.Path,
    case: catalog.CacheCoherenceCase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(f"{case.name}: invalid output resource size")
    if case.kind == "ddr-conflict-equivalence":
        return _validate_conflict_output(raw, case, built, sample)
    words = struct.unpack_from(f"<{catalog.RECORD_WORDS}Q", raw)
    rec = catalog.REC
    expected_record = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (catalog.SCHEMA << 32) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "SAMPLE": sample,
        "PAYLOAD_BYTES": case.payload_bytes,
        "EXPECTED_RDMA": case.expected_rdma,
        "EXPECTED_WDMA": case.expected_wdma,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT0_OFFSET": catalog.OUTPUT0_OFFSET,
        "OUTPUT1_OFFSET": catalog.OUTPUT1_OFFSET,
        "MISMATCH_AFTER": 0,
        "SPM_GUARD_MISMATCHES": 0,
        "OUTPUT_GUARD_MISMATCHES": 0,
        "RDMA_INST_DELTA": case.expected_rdma,
        "WDMA_INST_DELTA": case.expected_wdma,
        "CACHE_CONTROL_MASK": (
            8
            if case.kind == "ddr-bank-pair"
            else EXPECTED_CACHE_MASK[case.case_id]
        ),
        "PAIR_KIND": PAIR_KIND_CODES[case.pair_kind],
        "SCHEDULE": SCHEDULE_CODES[case.schedule],
        "BANK_OFFSET": case.bank_offset,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    failures = {
        key: (words[rec[key]], value)
        for key, value in expected_record.items()
        if words[rec[key]] != value
    }
    if failures:
        raise RuntimeError(f"{case.name}: record oracle failed: {failures}")

    mutable = bytearray(raw)
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [catalog.OUTPUT_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    before_observation: bytes | None = None
    if case.kind == "ddr-bank-pair":
        for begin, expected in built.expected_regions:
            end = begin + len(expected)
            if bytes(mutable[begin:end]) != expected:
                raise RuntimeError(
                    f"{case.name}: DDR pair output at {begin} is not exact"
                )
            mutable[begin:end] = bytes([catalog.OUTPUT_CANARY]) * (
                end - begin
            )
    else:
        if case.output_regions >= 1:
            begin = catalog.OUTPUT0_OFFSET
            end = begin + catalog.PAYLOAD_BYTES
            before_observation = bytes(mutable[begin:end])
            if (
                case.case_id in (2, 3)
                and before_observation != built.post_control_expected
            ):
                raise RuntimeError(
                    f"{case.name}: WDMA output is not host-exact"
                )
            mutable[begin:end] = bytes([catalog.OUTPUT_CANARY]) * (
                end - begin
            )
        if case.output_regions == 2:
            begin = catalog.OUTPUT1_OFFSET
            end = begin + catalog.PAYLOAD_BYTES
            if bytes(mutable[begin:end]) != built.post_control_expected:
                raise RuntimeError(
                    f"{case.name}: post-clean NCC snapshot is not exact"
                )
            mutable[begin:end] = bytes([catalog.OUTPUT_CANARY]) * (
                end - begin
            )
    if mutable != bytes([catalog.OUTPUT_CANARY]) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != catalog.OUTPUT_CANARY
        )
        raise RuntimeError(
            f"{case.name}: output changed outside record/results at {mismatch}"
        )
    mismatch_before = words[rec["MISMATCH_BEFORE"]]
    if mismatch_before > catalog.PAYLOAD_BYTES:
        raise RuntimeError(f"{case.name}: invalid pre-control mismatch count")
    return {
        "case": case.as_dict(),
        "sample": sample,
        "control_observation_state": _classify_before(
            case, built, raw, mismatch_before
        ),
        "mismatch_before": mismatch_before,
        "post_control_exact": True,
        "pmu": {
            "rdma_execution": words[rec["RDMA_EXEC_DELTA"]],
            "wdma_execution": words[rec["WDMA_EXEC_DELTA"]],
        },
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: tuple[catalog.CacheCoherenceCase, ...],
) -> None:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    observations: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        case_observations = observations.setdefault(case.name, [])
        for sample in range(case.phases * case.repetitions):
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
                "cache_coherence_calibration: "
                + json.dumps(observation, sort_keys=True)
            )

    pair_keys = sorted(
        {
            (case.pair_kind, case.bank_offset)
            for case in cases
            if case.kind == "ddr-bank-pair"
        }
    )
    metric_names = ("rdma_execution", "wdma_execution")
    for pair_kind, bank_offset in pair_keys:
        schedule_cases = {
            case.schedule: case
            for case in cases
            if case.kind == "ddr-bank-pair"
            and case.pair_kind == pair_kind
            and case.bank_offset == bank_offset
        }
        schedule_medians: dict[str, dict[str, int | float]] = {}
        missing: list[str] = []
        for schedule in ("serial", "window"):
            case = schedule_cases.get(schedule)
            if case is None:
                missing.append(schedule)
                continue
            rows = observations[case.name]
            if len(rows) != case.repetitions:
                raise RuntimeError(
                    f"{case.name}: incomplete repeated PMU sample set"
                )
            schedule_medians[schedule] = {
                metric: statistics.median(
                    int(row["pmu"][metric]) for row in rows
                )
                for metric in metric_names
            }
        summary: dict[str, object] = {
            "pair_kind": pair_kind,
            "bank_offset": bank_offset,
            "samples_per_cell": catalog.DDR_BANK_PMU_REPETITIONS,
            "correctness": "all-exact",
            "pmu_medians": schedule_medians,
            "interpretation": (
                "raw repeated serial/window control over the same fixed "
                "RDMA-seed/pair/full-slot-WDMA-readback envelope; no DDR "
                "bank/color class inferred"
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
                for metric in metric_names
            }
        print(
            "ddr_bank_pair_repeat_summary: "
            + json.dumps(summary, sort_keys=True)
        )

    for case in cases:
        if case.kind != "ddr-conflict-equivalence":
            continue
        case_rows = observations[case.name]
        if len(case_rows) != 1:
            raise RuntimeError(
                f"{case.name}: conflict coordinate did not run once"
            )
        observation = case_rows[0]
        batch_rows = observation["batch_rows"]
        assert isinstance(batch_rows, list)
        order_summaries: dict[str, object] = {}
        for issue_order, order_name in ((0, "a-b"), (1, "b-a")):
            schedule_medians: dict[str, dict[str, int | float]] = {}
            for schedule, schedule_name in (
                (1, "serial"),
                (2, "window"),
            ):
                selected = [
                    row
                    for row in batch_rows
                    if row["issue_order"] == issue_order
                    and row["schedule"] == schedule
                ]
                if len(selected) != catalog.CONFLICT_REPETITIONS:
                    raise RuntimeError(
                        f"{case.name}: incomplete {schedule_name}/"
                        f"{order_name} repeat set"
                    )
                schedule_medians[schedule_name] = {
                    metric: statistics.median(
                        int(row[metric]) for row in selected
                    )
                    for metric in (
                        "plan_cycles",
                        "rdma_execution",
                        "wdma_execution",
                    )
                }
            order_summaries[order_name] = {
                "medians": schedule_medians,
                "window_minus_serial": {
                    metric: (
                        schedule_medians["window"][metric]
                        - schedule_medians["serial"][metric]
                    )
                    for metric in schedule_medians["serial"]
                },
            }
        print(
            "ddr_conflict_equivalence_summary: "
            + json.dumps(
                {
                    "case": case.as_dict(),
                    "actual_ddr": observation["actual_ddr"],
                    "samples_per_control": (
                        catalog.CONFLICT_REPETITIONS
                    ),
                    "issue_orders": order_summaries,
                    "correctness": observation["correctness"],
                    "state": "raw-paired-equivalence-input",
                    "compiler_use": "no-ddr-bank-coloring",
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
                    "board_cases": [
                        case.as_dict() for case in catalog.CATALOG
                    ],
                    "pending_ddr_conflict_equivalence_cases": [
                        case.as_dict()
                        for case in catalog.PENDING_DDR_CONFLICT_CASES
                    ],
                    "ddr_conflict_equivalence_dispositions": [
                        case.as_dict()
                        for case in catalog.DDR_CONFLICT_DISPOSITIONS
                    ],
                    "deferred_descriptor_dispositions": [
                        case.as_dict()
                        for case in catalog.DDR_LARGE_DESCRIPTOR_DISPOSITIONS
                    ],
                    "session_dispositions": [
                        case.as_dict()
                        for case in catalog.CACHE_SESSION_DISPOSITIONS
                    ],
                    "calibration_leaf_bindings": {
                        key: [getattr(row, "name", "") for row in rows]
                        for key, rows
                        in catalog.CALIBRATION_LEAF_BINDINGS.items()
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
                "wafer_board_cache_coherence_calibration_probe_test: "
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
    execute_cases(args, package, resource_ids, selected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            "wafer_board_cache_coherence_calibration_probe_test: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
