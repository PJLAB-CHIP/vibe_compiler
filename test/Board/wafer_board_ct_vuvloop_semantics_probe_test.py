#!/usr/bin/env python3
"""Build and run isolated CT VuVLoop geometry observations."""

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
import wafer_ct_vuvloop_semantics_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ct_vuvloop_semantics_probe.c"
OUTPUT_INITIAL_CANARY = 0xA5
MODULE = """\
module {
  func.func @main(
      %request: tensor<2048xf32>,
      %payload: tensor<2048xf32>) -> tensor<2048xf32> {
    %result = stablehlo.add %request, %payload : tensor<2048xf32>
    return %result : tensor<2048xf32>
  }
}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [2048], "dtype": "float32", "dynamic_dims": []},
        {"shape": [2048], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [2048], "dtype": "float32", "dynamic_dims": []}
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
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=30000)
    parser.add_argument("--observation-samples", type=int, default=3)
    return parser.parse_args()


def configure_package_support() -> None:
    package_support.catalog = catalog
    package_support.PROBE_C = PROBE_C
    package_support.MODULE = MODULE
    package_support.METADATA = METADATA


def select_cases(
    args: argparse.Namespace,
) -> tuple[catalog.VuVLoopSemanticCase, ...]:
    if not args.selected_cases:
        return catalog.CATALOG
    unknown = [
        name
        for name in args.selected_cases
        if name not in catalog.CASES_BY_NAME
    ]
    if unknown:
        raise RuntimeError(f"unknown VuVLoop semantic cases: {unknown}")
    selected = tuple(
        catalog.CASES_BY_NAME[name] for name in args.selected_cases
    )
    if len({case.name for case in selected}) != len(selected):
        raise RuntimeError("VuVLoop case selection repeats a row")
    return selected


def _first_mismatch(actual: bytes, expected: bytes) -> int | None:
    return next(
        (
            index
            for index, (left, right) in enumerate(
                zip(actual, expected, strict=True)
            )
            if left != right
        ),
        None,
    )


def validate_output(
    path: pathlib.Path,
    case: catalog.VuVLoopSemanticCase,
    built: catalog.VuVLoopObservationPayload,
    sample: int,
) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError(
            f"{case.name}: output has {len(raw)} bytes, expected "
            f"{catalog.RESOURCE_BYTES}"
        )
    words = struct.unpack_from(f"<{catalog.RECORD_WORDS}Q", raw)
    rec = catalog.REC
    geometry = case.geometry
    expected_mirror = (
        case.case_id,
        catalog.DISPOSITIONS[case.disposition],
        catalog.OPCODE,
        catalog.DTYPE,
        geometry.elem_count,
        geometry.unit_elem_count,
        geometry.full_elem_count,
        geometry.full_unit_elem_count,
        sample,
        catalog.REQUEST_GUARD,
    )
    actual_mirror = (
        words[rec["CASE"]],
        words[rec["DISPOSITION"]],
        words[rec["OPCODE"]],
        words[rec["DTYPE"]],
        words[rec["ELEM_COUNT"]],
        words[rec["UNIT_ELEM_COUNT"]],
        words[rec["FULL_ELEM_COUNT"]],
        words[rec["FULL_UNIT_ELEM_COUNT"]],
        words[rec["SAMPLE"]],
        words[rec["REQUEST_GUARD"]],
    )
    execute_result = words[rec["EXECUTE_RESULT"]]
    status = words[rec["STATUS"]]
    if case.exact_legality:
        execution_valid = status == 0 and execute_result != 0
    else:
        execution_valid = (
            (status == 0 and execute_result != 0)
            or (status == 3 and execute_result == 0)
        )
    if (
        words[rec["MAGIC"]] != catalog.RECORD_MAGIC
        or words[rec["WORD_COUNT"]]
        != catalog.RECORD_WORDS
        or not execution_valid
        or actual_mirror != expected_mirror
        or words[rec["REQUEST_ECHO_MISMATCHES"]] != 0
        or words[rec["COMPLETION_SEEN"]] != 1
        or words[rec["OUTPUT_DDR_OFFSET"]]
        != catalog.OUTPUT_DDR_OFFSET
        or words[rec["OUTPUT_SLOT_BYTES"]] != catalog.OUTPUT_SLOT_BYTES
        or words[rec["BODY_OFFSET"]] != catalog.BODY_OFFSET
        or words[rec["OUTPUT_OWNED_BYTES"]]
        != catalog.OUTPUT_CAPTURE_BYTES
        or words[rec["OUTPUT_PHYSICAL_SPAN"]]
        != geometry.output_physical_span
        or words[rec["RESOURCE_BYTES"]] != catalog.RESOURCE_BYTES
        or words[rec["TERMINAL_FENCE_COUNT"]] != 1
        or words[rec["RECORD_GUARD"]] != catalog.RECORD_GUARD
    ):
        raise RuntimeError(
            f"{case.name}: record/echo/completion oracle failed"
        )

    output_slot = raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.OUTPUT_SLOT_BYTES
    ]
    prefix = output_slot[: catalog.GUARD_BYTES]
    body = output_slot[
        catalog.BODY_OFFSET :
        catalog.BODY_OFFSET + catalog.OUTPUT_CAPTURE_BYTES
    ]
    suffix = output_slot[
        catalog.BODY_OFFSET + catalog.OUTPUT_CAPTURE_BYTES :
    ]
    if (
        prefix != built.output.prefix_guard
        or suffix != built.output.suffix_guard
    ):
        raise RuntimeError(f"{case.name}: output guard changed")

    if built.expected_result is not None:
        expected_body = bytearray(built.output.body)
        expected_body[: len(built.expected_result)] = built.expected_result
        mismatch = _first_mismatch(body, bytes(expected_body))
        if mismatch is not None:
            raise RuntimeError(
                f"{case.name}: exact control differs at body byte "
                f"{mismatch}: actual=0x{body[mismatch]:02x}, "
                f"expected=0x{expected_body[mismatch]:02x}"
            )

    mutable = bytearray(raw)
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [OUTPUT_INITIAL_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    mutable[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.OUTPUT_SLOT_BYTES
    ] = bytes([OUTPUT_INITIAL_CANARY]) * catalog.OUTPUT_SLOT_BYTES
    if mutable != bytes([OUTPUT_INITIAL_CANARY]) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != OUTPUT_INITIAL_CANARY
        )
        raise RuntimeError(
            f"{case.name}: output changed outside record/capture at byte "
            f"{mismatch}"
        )

    changed_bytes = sum(
        actual != seed
        for actual, seed in zip(body, built.output.body, strict=True)
    )
    return {
        "case": case.as_dict(),
        "sample": sample,
        "status": status,
        "execute_result": execute_result,
        "changed_bytes": changed_bytes,
        "capture_sha256": hashlib.sha256(output_slot).hexdigest(),
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.VuVLoopSemanticCase],
) -> None:
    if args.observation_samples < 1:
        raise RuntimeError("--observation-samples must be at least one")
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    for case in cases:
        samples = 1 if case.exact_legality else args.observation_samples
        for sample in range(samples):
            built = catalog.build_observation_payload(case, sample)
            stem = f"{case.name}.sample-{sample}"
            request = raw_dir / f"{stem}.request.raw"
            payload = raw_dir / f"{stem}.payload.raw"
            output = raw_dir / f"{stem}.output.raw"
            request.write_bytes(built.request_resource)
            payload.write_bytes(built.payload_resource)
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
            observation = validate_output(output, case, built, sample)
            print(
                "ct_vuvloop_semantics: "
                + json.dumps(observation, sort_keys=True)
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
                "wafer_board_ct_vuvloop_semantics_probe_test: hardware "
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
            f"wafer_board_ct_vuvloop_semantics_probe_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
