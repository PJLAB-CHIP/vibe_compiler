#!/usr/bin/env python3
"""Build one package and run extended raw/composite DataMove rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
from collections.abc import Iterable

import wafer_board_instruction_family_probe_test as package_support
import wafer_datamove_extended_calibration_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_datamove_extended_calibration_probe.c"
MODULE = """\
module {
  func.func @main(
      %request: tensor<65536xf32>,
      %payload: tensor<65536xf32>) -> tensor<65536xf32> {
    %result = stablehlo.add %request, %payload : tensor<65536xf32>
    return %result : tensor<65536xf32>
  }
}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [65536], "dtype": "float32", "dynamic_dims": []},
        {"shape": [65536], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [65536], "dtype": "float32", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request"},
        {"type_": "input_arg", "position": 1, "name": "payload"},
    ],
    "unused_inputs": [],
}

ISOLATED_NATIVE_CONCAT_HW_SEQUENCE = (
    "datamove-raw-concat-c-n2h7w9-c33-c32",
    "datamove-raw-concat-w-n2h7-w4-w5-c65",
    "datamove-raw-concat-h-n2-h3-h4-w9-c65",
    "datamove-raw-concat-hw-n2-2x5-3x7-c65",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--operation", action="append")
    parser.add_argument(
        "--oracle", action="append", choices=("exact", "observation")
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
    parser.add_argument("--observation-samples", type=int, default=3)
    parser.add_argument(
        "--allow-isolated-native-concat-hw",
        action="store_true",
        help=(
            "authorize only the fixed C/W/H control sequence followed by the "
            "native HW Concat requalification case"
        ),
    )
    return parser.parse_args()


def configure_package_support() -> None:
    package_support.catalog = catalog
    package_support.PROBE_C = PROBE_C
    package_support.MODULE = MODULE
    package_support.METADATA = METADATA


def execution_sample_count(
    *, exact: bool, observation_samples: int
) -> int:
    if observation_samples < 1:
        raise ValueError("--observation-samples must be positive")
    return 1 if exact else observation_samples


def select_cases(
    args: argparse.Namespace,
) -> tuple[catalog.ExtendedDataMoveCase, ...]:
    if args.selected_cases:
        unknown = sorted(set(args.selected_cases) - set(catalog.CASES_BY_NAME))
        if unknown:
            raise RuntimeError(f"unknown extended DataMove cases: {unknown}")
        selected = tuple(
            catalog.CASES_BY_NAME[name] for name in args.selected_cases
        )
    else:
        selected = catalog.CATALOG
    if args.operation:
        operations = set(args.operation)
        unknown = sorted(
            operations - {case.operation for case in catalog.CATALOG}
        )
        if unknown:
            raise RuntimeError(
                f"unknown extended DataMove operations: {unknown}"
            )
        selected = tuple(
            case for case in selected if case.operation in operations
        )
    if args.oracle:
        exact = {"exact": True, "observation": False}
        accepted = {exact[value] for value in args.oracle}
        selected = tuple(
            case for case in selected if case.is_exact in accepted
        )
    if not selected:
        raise RuntimeError("extended DataMove filters selected no cases")
    if len({case.name for case in selected}) != len(selected):
        raise RuntimeError("extended DataMove selection repeats a case")
    selected_names = tuple(case.name for case in selected)
    selects_isolated_hw = any(
        case in catalog.ISOLATED_CONCAT_CASES for case in selected
    )
    if selects_isolated_hw:
        if not args.allow_isolated_native_concat_hw:
            raise RuntimeError(
                "native Concat HW requires the explicit isolated authorization"
            )
        if selected_names != ISOLATED_NATIVE_CONCAT_HW_SEQUENCE:
            raise RuntimeError(
                "native Concat HW must be the final case after the fixed "
                "C/W/H control sequence"
            )
    elif args.allow_isolated_native_concat_hw:
        raise RuntimeError(
            "isolated native Concat HW authorization requires the fixed "
            "C/W/H/HW case sequence"
        )
    return selected


def _first_mismatch(actual: bytes, expected: bytes) -> int:
    return next(
        index
        for index, (left, right) in enumerate(
            zip(actual, expected, strict=True)
        )
        if left != right
    )


def output_guard_mismatches(
    output_slot: bytes, output_span: int
) -> int:
    if len(output_slot) != catalog.SLOT_BYTES:
        raise RuntimeError("invalid output slot size")
    allowed_end = catalog.BODY_OFFSET + output_span
    if (
        output_span < 0
        or allowed_end < catalog.BODY_OFFSET
        or allowed_end > catalog.SLOT_BYTES
    ):
        raise RuntimeError("invalid output physical span")
    canary = catalog.SLOT_CANARY
    return sum(
        value != canary
        for value in output_slot[: catalog.BODY_OFFSET]
    ) + sum(value != canary for value in output_slot[allowed_end:])


def validate_output(
    path: pathlib.Path,
    case: catalog.ExtendedDataMoveCase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(f"{case.name}: invalid output resource size")
    words = struct.unpack_from(f"<{catalog.RECORD_WORDS}Q", raw)
    rec = catalog.REC
    expected_record = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (catalog.SCHEMA << 32) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "INPUT_BYTES": case.input_bytes,
        "RESULT_BYTES": case.result_bytes,
        "OUTPUT_SPAN": case.output_span,
        "TDMA_INSTRUCTIONS": case.expected_tdma_instructions,
        "CT_INSTRUCTIONS": case.expected_ct_instructions,
        "NE_INSTRUCTIONS": case.expected_ne_instructions,
        "ORACLE": case.oracle,
        "SAMPLE": sample,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "OUTPUT_GUARD_MISMATCHES": 0,
        "TDMA_INST_DELTA": case.expected_tdma_instructions,
        "CT_INST_DELTA": case.expected_ct_instructions,
        "NE_INST_DELTA": case.expected_ne_instructions,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    failures = {
        key: (words[rec[key]], expected)
        for key, expected in expected_record.items()
        if words[rec[key]] != expected
    }
    if failures:
        raise RuntimeError(f"{case.name}: record oracle failed: {failures}")

    output_slot = raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ]
    if output_guard_mismatches(output_slot, case.output_span) != 0:
        raise RuntimeError(f"{case.name}: output slot guard differs")

    result = output_slot[
        catalog.BODY_OFFSET : catalog.BODY_OFFSET + case.result_bytes
    ]
    observation: dict[str, object] = {}
    if case.is_exact:
        expected = built.expected_output_slot[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + case.result_bytes
        ]
        if result != expected:
            raise RuntimeError(
                f"{case.name}: exact result differs at byte "
                f"{_first_mismatch(result, expected)}"
            )
    else:
        changed = sum(value != catalog.SLOT_CANARY for value in result)
        if changed == 0:
            raise RuntimeError(
                f"{case.name}: raw observation did not modify its result range"
            )
        _, semantic_expected = catalog.build_input_expected(case, sample + 1)
        observation["changed_bytes"] = changed
        if len(semantic_expected) == len(result):
            observation["expected_mismatches"] = sum(
                left != right
                for left, right in zip(
                    result, semantic_expected, strict=True
                )
            )

    mutable = bytearray(raw[: catalog.OUTPUT_DDR_OFFSET])
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [package_support.OUTPUT_INITIAL_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    if mutable != bytes(
        [package_support.OUTPUT_INITIAL_CANARY]
    ) * catalog.OUTPUT_DDR_OFFSET:
        raise RuntimeError(f"{case.name}: output changed outside record/slot")
    return {
        "case": case.as_dict(),
        "sample": sample,
        "result_sha256": hashlib.sha256(result).hexdigest(),
        "observation": observation,
        "raw_execute_rc": words[rec["RAW_EXECUTE_RC"]],
        "pmu": {
            "tdma_execution": words[rec["TDMA_EXEC_DELTA"]],
            "ct_execution": words[rec["CT_EXEC_DELTA"]],
            "ne_execution": words[rec["NE_EXEC_DELTA"]],
        },
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.ExtendedDataMoveCase],
) -> None:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    for case in cases:
        sample_count = execution_sample_count(
            exact=case.is_exact,
            observation_samples=args.observation_samples,
        )
        for sample in range(sample_count):
            built = catalog.build_case_payload(case, sample)
            stem = f"{case.name}.sample-{sample}"
            request = raw_dir / f"{stem}.request.raw"
            payload = raw_dir / f"{stem}.payload.raw"
            output = raw_dir / f"{stem}.output.raw"
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
                    f"{case.name}: wafer-run omitted lifecycle evidence"
                )
            print(
                "datamove_extended_calibration: "
                + json.dumps(
                    validate_output(output, case, built, sample),
                    sort_keys=True,
                )
            )


def main() -> int:
    args = parse_args()
    execution_sample_count(
        exact=False, observation_samples=args.observation_samples
    )
    if args.list_cases:
        print(
            json.dumps(
                {
                    "cases": [case.as_dict() for case in catalog.CATALOG],
                    "isolated_cases": [
                        case.as_dict()
                        for case in catalog.ISOLATED_CONCAT_CASES
                    ],
                    "calibration_leaf_bindings": {
                        key: [case.name for case in cases]
                        for key, cases
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
                "wafer_board_datamove_extended_calibration_probe_test: "
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
            "wafer_board_datamove_extended_calibration_probe_test: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
