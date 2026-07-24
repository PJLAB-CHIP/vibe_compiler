#!/usr/bin/env python3
"""Build one shared package and run large/tail/batch NE calibration rows."""

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
import wafer_ne_calibration_catalog as catalog
import wafer_physical_tensor_codec as physical


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ne_calibration_probe.c"
OUTPUT_INITIAL_CANARY = 0xA5
MODULE = """\
module {
  func.func @main(
      %request: tensor<262144xf32>,
      %payload: tensor<262144xf32>) -> tensor<262144xf32> {
    %result = stablehlo.add %request, %payload : tensor<262144xf32>
    return %result : tensor<262144xf32>
  }
}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [262144], "dtype": "float32", "dynamic_dims": []},
        {"shape": [262144], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [262144], "dtype": "float32", "dynamic_dims": []}
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
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument(
        "--dtype",
        action="append",
        choices=tuple(name.lower() for name in catalog.DTYPES),
    )
    parser.add_argument(
        "--geometry",
        action="append",
        choices=tuple(
            sorted(
                {
                    case.geometry_name
                    for case in catalog.SAFE_CASES
                }
            )
        ),
    )
    parser.add_argument(
        "--orientation",
        action="append",
        choices=tuple(name.lower() for name in catalog.ORIENTATIONS),
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=tuple(name.lower() for name in catalog.KINDS),
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(
            name.lower().replace("_", "-")
            for name in catalog.PROFILES
        ),
    )
    parser.add_argument(
        "--option",
        action="append",
        choices=tuple(
            name.lower().replace("_", "-") for name in catalog.OPTIONS
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("exact", "observed", "all"),
        default="exact",
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


def select_cases(args: argparse.Namespace) -> tuple[catalog.NECase, ...]:
    if args.selected_cases:
        unknown = sorted(set(args.selected_cases) - set(catalog.CASES_BY_NAME))
        if unknown:
            raise RuntimeError(f"unknown NE calibration cases: {unknown}")
        selected = tuple(
            catalog.CASES_BY_NAME[name] for name in args.selected_cases
        )
        deferred = tuple(case for case in selected if not case.is_safe)
        if deferred:
            details = {case.name: case.reason for case in deferred}
            raise RuntimeError(
                f"deferred/negative NE cases cannot be issued: {details}"
            )
    else:
        selected = catalog.SAFE_CASES
        if args.suite != "all":
            disposition = {
                "exact": "BOARD_EXACT",
                "observed": "BOARD_OBSERVED",
            }[args.suite]
            selected = tuple(
                case
                for case in selected
                if case.disposition_name == disposition
            )
        if args.dtype:
            dtypes = {name.upper() for name in args.dtype}
            selected = tuple(
                case for case in selected if case.dtype_name in dtypes
            )
        if args.geometry:
            geometries = set(args.geometry)
            selected = tuple(
                case
                for case in selected
                if case.geometry_name in geometries
            )
        if args.orientation:
            orientations = {name.upper() for name in args.orientation}
            selected = tuple(
                case
                for case in selected
                if case.orientation_name in orientations
            )
        if args.kind:
            kinds = {name.upper() for name in args.kind}
            selected = tuple(
                case for case in selected if case.kind_name in kinds
            )
        if args.profile:
            profiles = {
                name.upper().replace("-", "_") for name in args.profile
            }
            selected = tuple(
                case for case in selected if case.profile_name in profiles
            )
        if args.option:
            options = {
                name.upper().replace("-", "_") for name in args.option
            }
            selected = tuple(
                case for case in selected if case.option_name in options
            )
    if not selected:
        raise RuntimeError("NE calibration filters selected no cases")
    return selected


def _validate_record(
    raw: bytes,
    case: catalog.NECase,
    sample: int,
) -> tuple[int, ...]:
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(
            f"{case.name}: output has {len(raw)} bytes, expected "
            f"{catalog.RESOURCE_BYTES}"
        )
    words = struct.unpack_from(f"<{catalog.RECORD_WORDS}Q", raw)
    rec = catalog.REC
    expected = (
        case.case_id,
        case.dtype,
        case.lhs_orientation,
        case.rhs_orientation,
        case.batch,
        case.m,
        case.k,
        case.n,
        case.lhs_span,
        case.rhs_span,
        case.output_span,
        sample,
        case.kind,
        case.profile,
        case.option,
        case.aux_span,
        case.disposition,
        case.lhs_batch,
        case.rhs_batch,
    )
    actual = tuple(
        words[rec[name]]
        for name in (
            "CASE",
            "DTYPE",
            "LHS_ORIENTATION",
            "RHS_ORIENTATION",
            "BATCH",
            "M",
            "K",
            "N",
            "LHS_SPAN",
            "RHS_SPAN",
            "OUTPUT_SPAN",
            "SAMPLE",
            "KIND",
            "PROFILE",
            "OPTION",
            "AUX_SPAN",
            "DISPOSITION",
            "LHS_BATCH",
            "RHS_BATCH",
        )
    )
    if (
        words[rec["MAGIC"]] != catalog.RECORD_MAGIC
        or words[rec["SCHEMA_AND_WORDS"]]
        != (catalog.SCHEMA << 32) | catalog.RECORD_WORDS
        or words[rec["STATUS"]] != 0
        or actual != expected
        or words[rec["EXECUTE_RESULT"]] == 0
        or words[rec["REQUEST_GUARD"]] != catalog.REQUEST_GUARD
        or words[rec["OUTPUT_DDR_OFFSET"]] != catalog.OUTPUT_DDR_OFFSET
        or words[rec["SLOT_BYTES"]] != catalog.SLOT_BYTES
        or words[rec["BODY_OFFSET"]] != catalog.BODY_OFFSET
        or words[rec["OUTPUT_GUARD_MISMATCHES"]] != 0
        or words[rec["RECORD_GUARD"]] != catalog.RECORD_GUARD
    ):
        raise RuntimeError(f"{case.name}: record/execute/guard oracle failed")
    return words


def validate_output(
    path: pathlib.Path,
    case: catalog.NECase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    raw = path.read_bytes()
    words = _validate_record(raw, case, sample)
    slot = raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ]
    begin = catalog.BODY_OFFSET
    end = begin + case.output_span
    if slot[:begin] != bytes([catalog.SLOT_CANARY]) * begin:
        raise RuntimeError(f"{case.name}: output prefix guard changed")
    if slot[end:] != bytes([catalog.SLOT_CANARY]) * (len(slot) - end):
        raise RuntimeError(f"{case.name}: output suffix guard changed")
    actual_physical = slot[begin:end]
    actual_logical = physical.unpack_scalar_bytes(
        case.output_shape,
        case.output_layout,
        case.element_bytes,
        actual_physical,
    )
    if (
        built.expected_logical is not None
        and actual_logical != built.expected_logical
    ):
        mismatch = next(
            index
            for index, (left, right) in enumerate(
                zip(actual_logical, built.expected_logical, strict=True)
            )
            if left != right
        )
        raise RuntimeError(
            f"{case.name}: logical result differs at element {mismatch}"
        )
    canonical_physical = (
        built.expected_physical
        if built.expected_physical is not None
        else physical.pack_scalar_bytes(
            case.output_shape,
            case.output_layout,
            case.element_bytes,
            actual_logical,
            padding=catalog.OUTPUT_PADDING,
        )
    )
    if actual_physical != canonical_physical:
        mismatch = next(
            index
            for index, (left, right) in enumerate(
                zip(
                    actual_physical,
                    canonical_physical,
                    strict=True,
                )
            )
            if left != right
        )
        raise RuntimeError(
            f"{case.name}: physical padding differs at byte {mismatch}"
        )
    initial_physical = built.payload[
        2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET :
        2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET + case.output_span
    ]
    if not case.exact and actual_physical == initial_physical:
        raise RuntimeError(
            f"{case.name}: observation completed without any bounded "
            "writeback"
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
            f"{case.name}: output changed outside record/result at {mismatch}"
        )
    return {
        "case": case.as_dict(),
        "sample": sample,
        "execute_result": words[catalog.REC["EXECUTE_RESULT"]],
        "logical_sha256": hashlib.sha256(
            b"".join(actual_logical)
        ).hexdigest(),
        "physical_sha256": hashlib.sha256(actual_physical).hexdigest(),
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.NECase],
) -> None:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
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
                f"{case.name}: wafer-run omitted lifecycle evidence"
            )
        print(
            "ne_calibration: "
            + json.dumps(
                validate_output(output, case, built, sample),
                sort_keys=True,
            )
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
                "wafer_board_ne_calibration_probe_test: hardware execution "
                "is not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
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
            f"wafer_board_ne_calibration_probe_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
