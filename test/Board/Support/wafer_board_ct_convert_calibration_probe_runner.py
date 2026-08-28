#!/usr/bin/env python3
"""Build one package and execute exact or observed CT convert rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
from collections.abc import Iterable

import wafer_board_instruction_family_probe_runner as package_support
import wafer_ct_convert_calibration_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ct_convert_calibration_probe.c"
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
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--opcode", action="append", type=int)
    parser.add_argument("--shape", choices=("main", "tail"))
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
    parser.add_argument("--completion-timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--stochastic-samples",
        type=int,
        default=3,
        help=(
            "independent board observations per stochastic route; raw output "
            "is retained and no deterministic sequence is assumed"
        ),
    )
    return parser.parse_args()


def configure_package_support() -> None:
    package_support.catalog = catalog
    package_support.PROBE_C = PROBE_C
    package_support.MODULE = MODULE
    package_support.METADATA = METADATA


def select_cases(args: argparse.Namespace) -> tuple[catalog.CTConvertCase, ...]:
    if args.selected_cases:
        unknown = sorted(set(args.selected_cases) - set(catalog.CASES_BY_NAME))
        if unknown:
            raise RuntimeError(f"unknown CT convert cases: {unknown}")
        selected = tuple(
            catalog.CASES_BY_NAME[name] for name in args.selected_cases
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
    deferred = tuple(case for case in selected if not case.is_safe)
    if deferred:
        details = {case.name: case.reason for case in deferred}
        raise RuntimeError(
            f"deferred CT convert cases cannot be issued: {details}"
        )
    if args.opcode:
        opcodes = set(args.opcode)
        invalid = sorted(opcodes - set(range(139, 175)))
        if invalid:
            raise RuntimeError(f"invalid CT convert opcodes: {invalid}")
        selected = tuple(case for case in selected if case.opcode in opcodes)
    if args.shape:
        selected = tuple(
            case for case in selected if case.shape_name == args.shape
        )
    if not selected:
        raise RuntimeError("CT convert filters selected no cases")
    return selected


def validate_output(
    path: pathlib.Path,
    case: catalog.CTConvertCase,
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
        "WORD_COUNT": catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "OPCODE": case.opcode,
        "SRC_TYPE": catalog.TYPE_CODES[case.source_type],
        "DST_TYPE": catalog.TYPE_CODES[case.destination_type],
        "ELEMENTS": case.elements,
        "INPUT_BYTES": case.input_bytes,
        "RESULT_BYTES": case.result_bytes,
        "OUTPUT_SPAN": case.output_span,
        "ROUNDING": case.rounding_mode,
        "DOMAIN": case.domain,
        "ZERO_POINT": case.zero_point,
        "DISPOSITION": case.disposition,
        "SAMPLE": sample,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "CT_INST_DELTA": 1,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    failures = {
        key: (words[rec[key]], value)
        for key, value in expected_record.items()
        if words[rec[key]] != value
    }
    if failures:
        raise RuntimeError(f"{case.name}: record oracle failed: {failures}")
    output_slot = raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ]
    if (
        built.expected_output_slot is not None
        and output_slot != built.expected_output_slot
    ):
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(output_slot, built.expected_output_slot, strict=True)
            )
            if actual != expected
        )
        raise RuntimeError(
            f"{case.name}: result/physical guard differs at byte {mismatch}"
        )
    if built.expected_output_slot is None:
        for begin, end in (
            (0, catalog.BODY_OFFSET),
            (
                catalog.BODY_OFFSET + case.output_span,
                catalog.SLOT_BYTES,
            ),
        ):
            mismatch = next(
                (
                    index
                    for index in range(begin, end)
                    if output_slot[index] != catalog.SLOT_CANARY
                ),
                None,
            )
            if mismatch is not None:
                raise RuntimeError(
                    f"{case.name}: observed output guard differs at "
                    f"slot byte {mismatch}"
                )
    mutable = bytearray(raw[: catalog.OUTPUT_DDR_OFFSET])
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [package_support.OUTPUT_INITIAL_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    if mutable != bytes(
        [package_support.OUTPUT_INITIAL_CANARY]
    ) * catalog.OUTPUT_DDR_OFFSET:
        raise RuntimeError(f"{case.name}: output changed outside record/slot")
    result = output_slot[
        catalog.BODY_OFFSET : catalog.BODY_OFFSET + case.result_bytes
    ]
    physical = output_slot[
        catalog.BODY_OFFSET : catalog.BODY_OFFSET + case.output_span
    ]
    return {
        "case": case.as_dict(),
        "sample": sample,
        "result_sha256": hashlib.sha256(result).hexdigest(),
        "physical_span_sha256": hashlib.sha256(physical).hexdigest(),
        "pmu": {
            "ct_execution": words[rec["CT_EXEC_DELTA"]],
            "ct_blocking": words[rec["CT_BLOCKING_DELTA"]],
        },
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.CTConvertCase],
) -> None:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    if args.stochastic_samples < 1:
        raise RuntimeError("--stochastic-samples must be at least one")
    for case in cases:
        sample_count = (
            args.stochastic_samples
            if case.domain_name == "STOCHASTIC"
            else 1
        )
        for sample in range(sample_count):
            stem = f"{case.name}.sample{sample}"
            built = catalog.build_case_payload(case, sample)
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
                "ct_convert_calibration: "
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
                "wafer_board_ct_convert_calibration_probe_runner: hardware "
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
            f"wafer_board_ct_convert_calibration_probe_runner: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
