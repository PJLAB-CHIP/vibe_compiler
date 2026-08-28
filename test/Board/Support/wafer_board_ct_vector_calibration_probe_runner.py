#!/usr/bin/env python3
"""Build one shared package and run exhaustive CT vector calibration rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import struct
import sys
from collections.abc import Iterable

import wafer_board_instruction_family_probe_runner as package_support
import wafer_ct_vector_calibration_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ct_vector_calibration_probe.c"
OUTPUT_INITIAL_CANARY = 0xA5
MODULE = """\
module {
  func.func @main(
      %request: tensor<32768xf32>,
      %payload: tensor<32768xf32>) -> tensor<32768xf32> {
    %result = stablehlo.add %request, %payload : tensor<32768xf32>
    return %result : tensor<32768xf32>
  }
}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [32768], "dtype": "float32", "dynamic_dims": []},
        {"shape": [32768], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [32768], "dtype": "float32", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request"},
        {"type_": "input_arg", "position": 1, "name": "payload"},
    ],
    "unused_inputs": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument(
        "--suite",
        choices=("exact", "tolerance", "observed", "all"),
        default="exact",
    )
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(name.lower() for name in catalog.FAMILIES),
    )
    parser.add_argument(
        "--dtype",
        action="append",
        choices=tuple(name.lower() for name in catalog.DTYPES),
    )
    parser.add_argument(
        "--shape", action="append", choices=("main", "tail")
    )
    parser.add_argument("--opcode", action="append", type=int)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--continue-on-validation-failure",
        action="store_true",
        help=(
            "continue only after host output-validation failures; board "
            "launch, timeout, and lifecycle failures still stop immediately"
        ),
    )
    return parser.parse_args()


def configure_package_support() -> None:
    package_support.catalog = catalog
    package_support.PROBE_C = PROBE_C
    package_support.MODULE = MODULE
    package_support.METADATA = METADATA


def select_cases(args: argparse.Namespace) -> tuple[catalog.CTVectorCase, ...]:
    if args.selected_cases:
        unknown = [
            name
            for name in args.selected_cases
            if name not in catalog.CASES_BY_NAME
        ]
        if unknown:
            raise RuntimeError(f"unknown CT vector cases: {unknown}")
        selected = tuple(
            catalog.CASES_BY_NAME[name] for name in args.selected_cases
        )
    else:
        selected = catalog.CATALOG
        if args.suite != "all":
            expected = {
                "exact": "BOARD_EXACT",
                "tolerance": "BOARD_TOLERANCE",
                "observed": "BOARD_OBSERVED",
            }[args.suite]
            selected = tuple(
                case
                for case in selected
                if case.disposition_name == expected
            )
        if args.family:
            families = {name.upper() for name in args.family}
            selected = tuple(
                case
                for case in selected
                if case.family_name in families
            )
        if args.dtype:
            dtypes = {name.upper() for name in args.dtype}
            selected = tuple(
                case for case in selected if case.dtype_name in dtypes
            )
        if args.shape:
            shapes = set(args.shape)
            selected = tuple(
                case for case in selected if case.shape_name in shapes
            )
        if args.opcode:
            opcodes = set(args.opcode)
            invalid = sorted(opcodes - set(range(111)))
            if invalid:
                raise RuntimeError(
                    f"CT vector opcode must be in 0..110: {invalid}"
                )
            selected = tuple(
                case for case in selected if case.opcode in opcodes
            )
    if not selected:
        raise RuntimeError("CT vector filters selected no cases")
    if len({case.name for case in selected}) != len(selected):
        raise RuntimeError("CT vector case selection repeats a row")
    return selected


def _validate_record(
    raw: bytes,
    case: catalog.CTVectorCase,
    built: catalog.CasePayload,
    sample: int,
) -> tuple[int, ...]:
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(
            f"{case.name}: output has {len(raw)} bytes, expected "
            f"{catalog.RESOURCE_BYTES}"
        )
    words = struct.unpack_from(f"<{catalog.RECORD_WORDS}Q", raw)
    rec = catalog.REC
    expected_mirror = (
        case.case_id,
        case.disposition,
        case.family,
        case.dtype,
        case.opcode,
        case.result_bytes,
        case.output_span,
        case.elements,
        sample,
        built.input_a_bytes,
        built.input_b_bytes,
        built.scalar_bits,
        case.unit_elements,
        case.domain,
        case.elem_count,
        case.full_elements,
        case.full_unit_elements,
    )
    actual_mirror = (
        words[rec["CASE"]],
        words[rec["DISPOSITION"]],
        words[rec["FAMILY"]],
        words[rec["DTYPE"]],
        words[rec["OPCODE"]],
        words[rec["RESULT_BYTES"]],
        words[rec["OUTPUT_SPAN"]],
        words[rec["ELEMENTS"]],
        words[rec["SAMPLE"]],
        words[rec["INPUT_A_BYTES"]],
        words[rec["INPUT_B_BYTES"]],
        words[rec["SCALAR_BITS"]],
        words[rec["UNIT_ELEMENTS"]],
        words[rec["DOMAIN"]],
        words[rec["ELEM_COUNT"]],
        words[rec["FULL_ELEMENTS"]],
        words[rec["FULL_UNIT_ELEMENTS"]],
    )
    if (
        words[rec["MAGIC"]] != catalog.RECORD_MAGIC
        or words[rec["WORD_COUNT"]]
        != catalog.RECORD_WORDS
        or words[rec["STATUS"]] != 0
        or actual_mirror != expected_mirror
        or words[rec["EXECUTE_RESULT"]] == 0
        or words[rec["REQUEST_GUARD"]] != catalog.REQUEST_GUARD
        or words[rec["OUTPUT_DDR_OFFSET"]] != catalog.OUTPUT_DDR_OFFSET
        or words[rec["SLOT_BYTES"]] != catalog.SLOT_BYTES
        or words[rec["BODY_OFFSET"]] != catalog.BODY_OFFSET
        or words[rec["RECORD_GUARD"]] != catalog.RECORD_GUARD
    ):
        raise RuntimeError(
            f"{case.name}: record/execute mirror failed"
        )
    return words


def _validate_numeric(
    case: catalog.CTVectorCase,
    actual: bytes,
    expected: bytes,
) -> dict[str, float | int]:
    if catalog._is_bool_output(case.opcode):
        full_bytes, valid_tail_bits = divmod(case.elements, 8)
        if actual[:full_bytes] != expected[:full_bytes]:
            mismatch = next(
                index
                for index, (left, right) in enumerate(
                    zip(
                        actual[:full_bytes],
                        expected[:full_bytes],
                        strict=True,
                    )
                )
                if left != right
            )
            raise RuntimeError(
                f"{case.name}: exact result differs at byte {mismatch}: "
                f"actual=0x{actual[mismatch]:02x}, "
                f"expected=0x{expected[mismatch]:02x}"
            )
        if valid_tail_bits:
            valid_mask = (1 << valid_tail_bits) - 1
            tail_actual = actual[full_bytes]
            tail_expected = expected[full_bytes]
            if tail_actual & valid_mask != tail_expected & valid_mask:
                raise RuntimeError(
                    f"{case.name}: exact result differs at byte "
                    f"{full_bytes}: actual=0x{tail_actual:02x}, "
                    f"expected=0x{tail_expected:02x}"
                )
        return {"mismatches": 0}

    if case.exact:
        if actual != expected:
            mismatch = next(
                index
                for index, (left, right) in enumerate(
                    zip(actual, expected, strict=True)
                )
                if left != right
            )
            raise RuntimeError(
                f"{case.name}: exact result differs at byte {mismatch}: "
                f"actual=0x{actual[mismatch]:02x}, "
                f"expected=0x{expected[mismatch]:02x}"
            )
        return {"mismatches": 0}

    actual_values = catalog.decode_values(case.dtype_name, actual)
    expected_values = catalog.decode_values(case.dtype_name, expected)
    absolute_tolerance, relative_tolerance = catalog.tolerance_for(case)
    max_absolute = 0.0
    max_relative = 0.0
    for index, (left, right) in enumerate(
        zip(actual_values, expected_values, strict=True)
    ):
        if not math.isfinite(left) or not math.isfinite(right):
            if left != right:
                raise RuntimeError(
                    f"{case.name}: non-finite result differs at element {index}"
                )
            continue
        absolute = abs(left - right)
        relative = absolute / max(abs(right), 1.0)
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
        if absolute > absolute_tolerance and relative > relative_tolerance:
            raise RuntimeError(
                f"{case.name}: tolerance failed at element {index}: "
                f"actual={left}, expected={right}, "
                f"absolute={absolute}, relative={relative}"
            )
    return {
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
    }


def _validate_output_guard(
    case: catalog.CTVectorCase,
    slot: bytes,
) -> None:
    suffix_begin = catalog.BODY_OFFSET + (
        case.result_bytes
        if catalog._is_bool_output(case.opcode)
        else case.output_span
    )
    for begin, end in (
        (0, catalog.BODY_OFFSET),
        (suffix_begin, catalog.SLOT_BYTES),
    ):
        mismatch = next(
            (
                index
                for index in range(begin, end)
                if slot[index] != catalog.SLOT_CANARY
            ),
            None,
        )
        if mismatch is not None:
            raise RuntimeError(
                f"{case.name}: output guard differs at slot byte {mismatch}"
            )


def validate_output(
    path: pathlib.Path,
    case: catalog.CTVectorCase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    raw = path.read_bytes()
    words = _validate_record(raw, case, built, sample)
    slot = raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ]
    result_begin = catalog.BODY_OFFSET
    result_end = result_begin + case.result_bytes
    actual_result = slot[result_begin:result_end]
    _validate_output_guard(case, slot)

    numeric: dict[str, float | int] = {}
    if built.expected_result is not None:
        numeric = _validate_numeric(
            case, actual_result, built.expected_result
        )

    mutable = bytearray(raw)
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [OUTPUT_INITIAL_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    mutable[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ] = bytes([OUTPUT_INITIAL_CANARY]) * catalog.SLOT_BYTES
    if mutable != bytes([OUTPUT_INITIAL_CANARY]) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != OUTPUT_INITIAL_CANARY
        )
        raise RuntimeError(
            f"{case.name}: output changed outside record/result at byte "
            f"{mismatch}"
        )
    return {
        "case": case.as_dict(),
        "sample": sample,
        "execute_result": words[catalog.REC["EXECUTE_RESULT"]],
        "result_sha256": hashlib.sha256(actual_result).hexdigest(),
        **numeric,
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.CTVectorCase],
) -> None:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    validation_failures: list[dict[str, object]] = []
    for sample, case in enumerate(cases):
        built = catalog.build_case_payload(case, sample)
        request = raw_dir / f"{case.name}.request.raw"
        payload = raw_dir / f"{case.name}.payload.raw"
        output = raw_dir / f"{case.name}.output.raw"
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
                f"{case.name}: wafer-run omitted complete lifecycle evidence"
            )
        try:
            observation = validate_output(output, case, built, sample)
        except (RuntimeError, ValueError) as error:
            if not args.continue_on_validation_failure:
                raise
            failure = {
                "case": case.name,
                "reason": str(error),
                "sample": sample,
            }
            validation_failures.append(failure)
            print(
                "ct_vector_calibration_failure: "
                + json.dumps(failure, sort_keys=True),
                file=sys.stderr,
            )
            continue
        print(
            "ct_vector_calibration: "
            + json.dumps(observation, sort_keys=True)
        )
    if validation_failures:
        raise RuntimeError(
            "CT vector validation failures: "
            + json.dumps(validation_failures, sort_keys=True)
        )


def main() -> int:
    args = parse_args()
    if args.list_cases:
        print(
            json.dumps(
                [case.as_dict() for case in catalog.CATALOG],
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
                "wafer_board_ct_vector_calibration_probe_runner: hardware "
                "execution is not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        package_support.require_board_args(args)

    source = package_support.write_source_program(args)
    package = package_support.compile_seed_package(args, source)
    module_path, resource_ids, slots_per_tile = package_support.locate_bindings(
        package
    )
    package_support.build_probe(args, package, module_path, slots_per_tile)
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
            f"wafer_board_ct_vector_calibration_probe_runner: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
