#!/usr/bin/env python3
"""Build and run the matched sustained CT/RDMA SPM conflict pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import struct
import sys

import wafer_board_instruction_family_probe_test as package_support
import wafer_spm_sustained_conflict_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_spm_sustained_conflict_probe.c"
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


def configure_package_support() -> None:
    package_support.catalog = catalog
    package_support.PROBE_C = PROBE_C
    package_support.MODULE = MODULE
    package_support.METADATA = METADATA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_groups",
        help="select one matched 4-row group; repeatable",
    )
    parser.add_argument(
        "--cell",
        action="append",
        dest="selected_cells",
        help=(
            "select a semantic cell; its complete matched group is executed"
        ),
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--emit-board-case-keys", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument(
        "--repeat",
        type=int,
        default=catalog.COUNTERBALANCED_REPEATS,
        help="must remain four so every row occupies every execution ordinal",
    )
    return parser.parse_args()


def select_groups(
    args: argparse.Namespace,
) -> tuple[catalog.SustainedConflictGroup, ...]:
    selected_group_keys = list(args.selected_groups or ())
    unknown_groups = sorted(
        set(selected_group_keys) - set(catalog.GROUPS_BY_KEY)
    )
    selected_cells = list(args.selected_cells or ())
    unknown_cells = sorted(
        set(selected_cells) - set(catalog.CELLS_BY_KEY)
    )
    if unknown_groups or unknown_cells:
        raise RuntimeError(
            "unknown sustained conflict selectors: "
            f"groups={unknown_groups}, cells={unknown_cells}"
        )
    selected_group_keys.extend(
        catalog.CELLS_BY_KEY[cell_key].group_key
        for cell_key in selected_cells
    )
    if not selected_group_keys:
        return catalog.GROUPS
    unique = tuple(dict.fromkeys(selected_group_keys))
    return tuple(catalog.GROUPS_BY_KEY[key] for key in unique)


def _expected_global_record(
    invocation: catalog.InvocationPayload,
) -> dict[str, int]:
    group = invocation.group
    return {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "GROUP": group.group_id,
        "WORK_BYTES": group.work_bytes,
        "ROUNDS": group.rounds,
        "ISSUE_ORDER": group.issue_order,
        "SAMPLE": invocation.sample,
        "ROWS": catalog.ROW_COUNT,
        "COMPLETED_ROWS": catalog.ROW_COUNT,
        "TRANSFER_BYTES": catalog.TRANSFER_BYTES,
        "SPM_BASE": catalog.SPM_BASE,
        "OWNED_SPM_BYTES": group.owned_spm_bytes,
        "EXECUTION_ROTATION": invocation.sample,
        "FLAGS": catalog.REQUIRED_FLAGS,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }


def _expected_row_record(
    invocation: catalog.InvocationPayload,
    cell: catalog.SustainedConflictCell,
) -> dict[str, int]:
    row = cell.row_index
    return {
        "MAGIC": catalog.ROW_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.ROW_RECORD_WORDS,
        "STATUS": 0,
        "CELL": cell.cell_id,
        "ADDRESS_CLASS": cell.address_class.row_index,
        "RELATIVE_OFFSET": cell.relative_offset,
        "SCHEDULE": cell.schedule.wire_value,
        "ISSUE_ORDER": cell.issue_order,
        "SAMPLE": invocation.sample,
        "EXECUTION_ORDINAL": (
            row - invocation.sample
        ) % catalog.ROW_COUNT,
        "ROUNDS": cell.rounds,
        "TRANSFER_BYTES": catalog.TRANSFER_BYTES,
        "SPM_BASE": catalog.SPM_BASE,
        "OWNED_SPM_BYTES": cell.owned_spm_bytes,
        "SPM_CELL_STRIDE": catalog.SPM_CELL_STRIDE,
        "SPM_READ0_OFFSET": catalog.SPM_READ0_OFFSET,
        "SPM_READ1_OFFSET": catalog.SPM_READ1_OFFSET,
        "SPM_WRITE_OFFSET": catalog.SPM_WRITE_OFFSET,
        "RDMA_WRITE_OFFSET": (
            catalog.SPM_WRITE_OFFSET + cell.relative_offset
        ),
        "ARCHIVE_OFFSET": (
            catalog.OUTPUT_ARCHIVE_BASE
            + row * catalog.OUTPUT_ARCHIVE_STRIDE
        ),
        "CT_INST_DELTA": cell.expected_instruction_count,
        "RDMA_INST_DELTA": cell.expected_instruction_count,
        "OTHER_INST_DELTA": 0,
        "WORKER_CT_INST_DELTA": cell.expected_instruction_count,
        "WORKER_RDMA_INST_DELTA": cell.expected_instruction_count,
        "FLAGS": catalog.REQUIRED_FLAGS,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "ROW_GUARD": catalog.ROW_GUARD,
    }


def _record_failures(
    words: tuple[int, ...],
    fields: dict[str, int],
    expected: dict[str, int],
) -> dict[str, tuple[int, int]]:
    return {
        key: (words[fields[key]], value)
        for key, value in expected.items()
        if words[fields[key]] != value
    }


def _validate_actual_addresses(
    values: tuple[int, int, int],
    context: str,
) -> None:
    if (
        any(value == 0 or value % 256 for value in values)
        or len(set(values)) != len(values)
    ):
        raise RuntimeError(
            f"{context}: resource bases are zero, unaligned, or aliased: "
            f"{values}"
        )


def validate_output_bytes(
    raw: bytes,
    invocation: catalog.InvocationPayload,
) -> dict[str, object]:
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(
            f"{invocation.group.key}: output resource has {len(raw)} bytes"
        )
    global_words = struct.unpack_from(
        f"<{catalog.RECORD_WORDS}Q", raw
    )
    failures = _record_failures(
        global_words,
        catalog.REC,
        _expected_global_record(invocation),
    )
    global_addresses = tuple(
        global_words[catalog.REC[key]]
        for key in ("REQUEST_DDR", "PAYLOAD_DDR", "OUTPUT_DDR")
    )
    if failures:
        raise RuntimeError(
            f"{invocation.group.key}: global record failed: {failures}"
        )
    _validate_actual_addresses(
        global_addresses, invocation.group.key
    )

    mutable = bytearray(raw)
    mutable[: catalog.OUTPUT_RECORD_BYTES] = bytes(
        [catalog.RESOURCE_CANARY]
    ) * catalog.OUTPUT_RECORD_BYTES
    row_observations: dict[str, dict[str, object]] = {}
    for cell in invocation.group.cells:
        row = cell.row_index
        begin = (
            catalog.ROW_RECORD_BASE_WORD
            + row * catalog.ROW_RECORD_WORDS
        ) * 8
        row_words = struct.unpack_from(
            f"<{catalog.ROW_RECORD_WORDS}Q", raw, begin
        )
        row_failures = _record_failures(
            row_words,
            catalog.ROW_REC,
            _expected_row_record(invocation, cell),
        )
        row_addresses = tuple(
            row_words[catalog.ROW_REC[key]]
            for key in ("REQUEST_DDR", "PAYLOAD_DDR", "OUTPUT_DDR")
        )
        if row_failures or row_addresses != global_addresses:
            raise RuntimeError(
                f"{cell.key}: row record failed: {row_failures}, "
                f"row_addresses={row_addresses}, "
                f"global_addresses={global_addresses}"
            )
        if (
            row_words[catalog.ROW_REC["PLAN_CYCLES"]] == 0
            or row_words[catalog.ROW_REC["PMU_ENABLE"]] == 0
            or row_words[catalog.ROW_REC["SERIAL_MODE"]] & 1
            or row_words[catalog.ROW_REC["STABLE_BEFORE"]]
            != catalog.PMU_STABLE_MASK
            or row_words[catalog.ROW_REC["STABLE_AFTER"]]
            != catalog.PMU_STABLE_MASK
        ):
            raise RuntimeError(
                f"{cell.key}: PMU/parallel-mode basis is invalid"
            )
        control = row_words[catalog.ROW_REC["FINAL_CONTROL"]]
        if control & 0xFF or not control & 0x100:
            raise RuntimeError(
                f"{cell.key}: bounded completion control is invalid: "
                f"{control:#x}"
            )

        archive_begin = (
            catalog.OUTPUT_ARCHIVE_BASE
            + row * catalog.OUTPUT_ARCHIVE_STRIDE
        )
        archive_end = archive_begin + cell.owned_spm_bytes
        actual_snapshot = raw[archive_begin:archive_end]
        expected_snapshot = invocation.expected_snapshots[
            cell.relative_offset
        ]
        if actual_snapshot != expected_snapshot:
            mismatch = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(
                        actual_snapshot,
                        expected_snapshot,
                        strict=True,
                    )
                )
                if actual != expected
            )
            raise RuntimeError(
                f"{cell.key}: full owned SPM differs at byte {mismatch}: "
                f"actual=0x{actual_snapshot[mismatch]:02x}, "
                f"expected=0x{expected_snapshot[mismatch]:02x}"
            )
        mutable[archive_begin:archive_end] = bytes(
            [catalog.RESOURCE_CANARY]
        ) * cell.owned_spm_bytes
        row_observations[cell.key] = {
            "cell": cell.as_dict(),
            "sample": invocation.sample,
            "execution_ordinal": row_words[
                catalog.ROW_REC["EXECUTION_ORDINAL"]
            ],
            "actual_addresses": {
                "request_ddr": global_addresses[0],
                "payload_ddr": global_addresses[1],
                "output_ddr": global_addresses[2],
            },
            "snapshot_sha256": hashlib.sha256(
                actual_snapshot
            ).hexdigest(),
            "correctness": (
                "full-owned-SPM-exact+all-gap/prefix/suffix-canary"
            ),
            "plan_cycles": row_words[
                catalog.ROW_REC["PLAN_CYCLES"]
            ],
            "pmu": {
                "ct_execution": row_words[
                    catalog.ROW_REC["CT_EXEC_DELTA"]
                ],
                "rdma_execution": row_words[
                    catalog.ROW_REC["RDMA_EXEC_DELTA"]
                ],
                "full_execution": row_words[
                    catalog.ROW_REC["FULL_EXEC_DELTA"]
                ],
                "ct_blocking": row_words[
                    catalog.ROW_REC["CT_BLOCKING_DELTA"]
                ],
                "rdma_blocking": row_words[
                    catalog.ROW_REC["RDMA_BLOCKING_DELTA"]
                ],
            },
        }
    if mutable != bytes(
        [catalog.RESOURCE_CANARY]
    ) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != catalog.RESOURCE_CANARY
        )
        raise RuntimeError(
            f"{invocation.group.key}: output changed outside typed "
            f"records/snapshots at byte {mismatch}"
        )
    return {
        "group": invocation.group.as_dict(),
        "sample": invocation.sample,
        "rows": row_observations,
    }


def validate_output(
    path: pathlib.Path,
    invocation: catalog.InvocationPayload,
) -> dict[str, object]:
    return validate_output_bytes(path.read_bytes(), invocation)


def _metric(row: dict[str, object], name: str) -> int:
    if name == "plan_cycles":
        return int(row[name])
    pmu = row["pmu"]
    assert isinstance(pmu, dict)
    return int(pmu[name])


def summarize_group(
    group: catalog.SustainedConflictGroup,
    observations: list[dict[str, object]],
) -> dict[str, object]:
    if len(observations) != catalog.COUNTERBALANCED_REPEATS:
        raise RuntimeError(
            f"{group.key}: incomplete counterbalanced repetition set"
        )
    by_cell: dict[str, list[dict[str, object]]] = {
        cell.key: [] for cell in group.cells
    }
    for observation in observations:
        rows = observation["rows"]
        assert isinstance(rows, dict)
        for cell in group.cells:
            row = rows[cell.key]
            assert isinstance(row, dict)
            by_cell[cell.key].append(row)
    for cell in group.cells:
        ordinals = {
            int(row["execution_ordinal"])
            for row in by_cell[cell.key]
        }
        if ordinals != set(range(catalog.ROW_COUNT)):
            raise RuntimeError(
                f"{cell.key}: execution positions are not counterbalanced"
            )

    metric_names = (
        "plan_cycles",
        "ct_execution",
        "rdma_execution",
        "full_execution",
        "ct_blocking",
        "rdma_blocking",
    )
    matched_rows: list[dict[str, object]] = []
    informative = False
    for schedule in catalog.Schedule:
        candidate = next(
            cell
            for cell in group.cells
            if cell.address_class == catalog.AddressClass.CANDIDATE
            and cell.schedule == schedule
        )
        control = next(
            cell
            for cell in group.cells
            if cell.address_class == catalog.AddressClass.CONTROL
            and cell.schedule == schedule
        )
        deltas = {
            name: [
                _metric(candidate_row, name)
                - _metric(control_row, name)
                for candidate_row, control_row in zip(
                    by_cell[candidate.key],
                    by_cell[control.key],
                    strict=True,
                )
            ]
            for name in metric_names
        }
        median_deltas = {
            name: statistics.median(values)
            for name, values in deltas.items()
        }
        directions = {
            name: {
                0 if value == 0 else (1 if value > 0 else -1)
                for value in values
            }
            for name, values in deltas.items()
        }
        stable_nonzero = any(
            len(directions[name]) == 1
            and directions[name] != {0}
            for name in ("plan_cycles", "full_execution")
        )
        informative |= stable_nonzero
        matched_rows.append(
            {
                "schedule": schedule.value,
                "candidate_minus_control": deltas,
                "median_candidate_minus_control": median_deltas,
                "direction_sets": {
                    name: sorted(values)
                    for name, values in directions.items()
                },
                "stable_nonzero_plan_or_full": stable_nonzero,
            }
        )
    return {
        "group": group.as_dict(),
        "correctness": (
            "all-four-rows-full-owned-SPM-exact+count+completion"
        ),
        "samples": catalog.COUNTERBALANCED_REPEATS,
        "matched_rows": matched_rows,
        "state": (
            "nonzero-signal-requires-heldout-review"
            if informative
            else "consistent-or-noisy-zero-signal"
        ),
        "compiler_use": "no-bank-coloring",
    }


def execute_groups(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    terminal_completion: int,
    groups: tuple[catalog.SustainedConflictGroup, ...],
) -> None:
    raw_dir = args.work_dir / "raw" / "spm-sustained-conflict"
    raw_dir.mkdir(parents=True)
    for group in groups:
        observations: list[dict[str, object]] = []
        group_dir = raw_dir / group.key
        group_dir.mkdir()
        for sample in range(catalog.COUNTERBALANCED_REPEATS):
            invocation = catalog.build_invocation(group, sample)
            request = group_dir / f"sample-{sample}.request.raw"
            payload = group_dir / f"sample-{sample}.payload.raw"
            output = group_dir / f"sample-{sample}.output.raw"
            request.write_bytes(invocation.request)
            payload.write_bytes(invocation.payload)
            result = package_support.run(
                package_support.board_command(
                    args,
                    package,
                    resource_ids,
                    request,
                    payload,
                    output,
                ),
                timeout_seconds=(
                    args.completion_timeout_ms / 1000.0 + 30.0
                ),
            )
            try:
                package_support.validate_board_lifecycle(
                    result.stdout, terminal_completion
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"{group.key} sample {sample}: {error}"
                ) from error
            observation = validate_output(output, invocation)
            observations.append(observation)
            print(
                "spm_sustained_conflict_observation: "
                + json.dumps(observation, sort_keys=True)
            )
        print(
            "spm_sustained_conflict_summary: "
            + json.dumps(
                summarize_group(group, observations),
                sort_keys=True,
            )
        )


def main() -> int:
    args = parse_args()
    if args.emit_board_case_keys:
        for group in catalog.GROUPS:
            print(group.key)
        return 0
    if args.list_cases:
        print(
            json.dumps(
                {
                    "groups": [
                        group.as_dict() for group in catalog.GROUPS
                    ],
                    "cells": [cell.as_dict() for cell in catalog.CELLS],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.repeat != catalog.COUNTERBALANCED_REPEATS:
        raise RuntimeError(
            "--repeat must be exactly four so every matched row occupies "
            "every execution ordinal once"
        )
    groups = select_groups(args)
    configure_package_support()
    package_support.require_build_args(args)
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "wafer_board_spm_sustained_conflict_probe_test: hardware "
                "execution is not armed; set "
                "WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        package_support.require_board_args(args)

    source = package_support.write_source_program(args)
    package = package_support.compile_seed_package(args, source)
    module_path, resource_ids = package_support.locate_bindings(package)
    terminal_completion = package_support.rank_one_terminal_completion(
        package
    )
    package_support.build_probe(args, package, module_path)
    package_support.verify_no_card(args, package)
    if args.no_card:
        print("spm_sustained_conflict_probe_no_card: passed")
        return 0
    execute_groups(
        args, package, resource_ids, terminal_completion, groups
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            "wafer_board_spm_sustained_conflict_probe_test: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
