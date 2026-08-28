#!/usr/bin/env python3
"""Build one package and run SPM boundary/relative-offset calibration rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import struct
import sys
from collections.abc import Iterable

import wafer_board_instruction_family_probe_runner as package_support
import wafer_spm_calibration_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_spm_calibration_probe.c"
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

GUARD_BYTES = 64
GUARD_READBACK_OFFSET = catalog.RECORD_WORDS * 8


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
        choices=(
            "capacity-reservation",
            "alignment-bank",
            "lifetime-reuse",
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
    parser.add_argument("--completion-timeout-ms", type=int, default=30000)
    return parser.parse_args()


def configure_package_support() -> None:
    package_support.catalog = catalog
    package_support.PROBE_C = PROBE_C
    package_support.MODULE = MODULE
    package_support.METADATA = METADATA


def select_cases(args: argparse.Namespace) -> tuple[catalog.SPMCase, ...]:
    if args.selected_cases:
        unknown = sorted(set(args.selected_cases) - set(catalog.CASES_BY_NAME))
        if unknown:
            raise RuntimeError(f"unknown SPM calibration cases: {unknown}")
        selected = tuple(
            catalog.CASES_BY_NAME[name] for name in args.selected_cases
        )
        deferred = [case.name for case in selected if not case.is_safe]
        if deferred:
            raise RuntimeError(
                "static-negative SPM cases cannot be selected for board "
                f"execution: {deferred}"
            )
    else:
        selected = catalog.BOARD_CASES
        if args.domain:
            domains = set(args.domain)
            selected = tuple(
                case for case in selected if case.domain in domains
            )
    if not selected:
        raise RuntimeError("SPM calibration filters selected no board cases")
    return selected


def guard_addresses(case: catalog.SPMCase) -> tuple[int, ...]:
    slots = 2 if case.kind == "lifetime" and case.iterations > 1 else 1
    addresses: list[int] = []
    for slot in range(slots):
        address = case.address + slot * case.slot_stride
        if address >= catalog.ALLOCATABLE_BEGIN + GUARD_BYTES:
            addresses.append(address - GUARD_BYTES)
        if (
            address + case.transfer_bytes + GUARD_BYTES
            <= catalog.ALLOCATABLE_END
        ):
            addresses.append(address + case.transfer_bytes)
    return tuple(addresses)


def validate_output(
    path: pathlib.Path,
    case: catalog.SPMCase,
    built: catalog.CasePayload,
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
    guards = guard_addresses(case)
    expected_ncc_instructions = case.expected_instructions + len(guards)
    if (
        words[rec["MAGIC"]] != catalog.RECORD_MAGIC
        or words[rec["WORD_COUNT"]]
        != catalog.RECORD_WORDS
        or words[rec["STATUS"]] != 0
        or words[rec["CASE"]] != catalog.CASE_IDS[case.name]
        or words[rec["ADDRESS"]] != case.address
        or words[rec["TRANSFER_BYTES"]] != case.transfer_bytes
        or words[rec["SAMPLE"]] != sample
        or words[rec["REQUEST_GUARD"]] != catalog.REQUEST_GUARD
        or words[rec["OUTPUT_DDR_OFFSET"]] != catalog.OUTPUT_DDR_OFFSET
        or words[rec["SLOT_BYTES"]] != catalog.SLOT_BYTES
        or words[rec["RDMA_INST_DELTA"]] != expected_ncc_instructions
        or words[rec["WDMA_INST_DELTA"]] != expected_ncc_instructions
        or words[rec["RECORD_GUARD"]] != catalog.RECORD_GUARD
    ):
        raise RuntimeError(
            f"{case.name}: record/count oracle failed"
        )
    guard_end = GUARD_READBACK_OFFSET + len(guards) * GUARD_BYTES
    if raw[GUARD_READBACK_OFFSET:guard_end] != bytes(
        [catalog.OUTPUT_CANARY]
    ) * (len(guards) * GUARD_BYTES):
        raise RuntimeError(f"{case.name}: SPM guard readback differs")
    begin = catalog.OUTPUT_DDR_OFFSET
    result_bytes = case.transfer_bytes * case.iterations
    end = begin + result_bytes
    if raw[begin:end] != built.expected:
        mismatch = next(
            index
            for index, (left, right) in enumerate(
                zip(raw[begin:end], built.expected, strict=True)
            )
            if left != right
        )
        raise RuntimeError(
            f"{case.name}: round-trip differs at byte {mismatch}"
        )
    mutable = bytearray(raw)
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [catalog.OUTPUT_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    mutable[GUARD_READBACK_OFFSET:guard_end] = bytes(
        [catalog.OUTPUT_CANARY]
    ) * (len(guards) * GUARD_BYTES)
    mutable[begin:end] = bytes([catalog.OUTPUT_CANARY]) * result_bytes
    if mutable != bytes([catalog.OUTPUT_CANARY]) * catalog.RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != catalog.OUTPUT_CANARY
        )
        raise RuntimeError(
            f"{case.name}: output changed outside record/result at {mismatch}"
        )
    return {
        "case": case.as_dict(),
        "sample": sample,
        "result_sha256": hashlib.sha256(raw[begin:end]).hexdigest(),
        "pmu": {
            "rdma_execution": words[rec["RDMA_EXEC_DELTA"]],
            "wdma_execution": words[rec["WDMA_EXEC_DELTA"]],
            "rdma_blocking": words[rec["RDMA_BLOCKING_DELTA"]],
            "wdma_blocking": words[rec["WDMA_BLOCKING_DELTA"]],
        },
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.SPMCase],
) -> None:
    cases = tuple(cases)
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    repeat_observations: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        observations = repeat_observations.setdefault(case.name, [])
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
            observations.append(observation)
            print(
                "spm_calibration: "
                + json.dumps(observation, sort_keys=True)
            )
    for case in cases:
        if case.repetitions <= 1:
            continue
        observations = repeat_observations[case.name]
        if len(observations) != case.repetitions:
            raise RuntimeError(
                f"{case.name}: incomplete repeated PMU sample set"
            )
        print(
            "spm_bank_repeat_summary: "
            + json.dumps(
                {
                    "case": case.name,
                    "relative_offset": case.address - catalog.SWEEP_BASE,
                    "samples": case.repetitions,
                    "correctness": "all-exact",
                    "pmu_medians": {
                        key: statistics.median(
                            int(observation["pmu"][key])
                            for observation in observations
                        )
                        for key in (
                            "rdma_execution",
                            "wdma_execution",
                            "rdma_blocking",
                            "wdma_blocking",
                        )
                    },
                    "interpretation": (
                        "raw repeated control; no bank/color class inferred"
                    ),
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
                "wafer_board_spm_calibration_probe_test: hardware execution "
                "is not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
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
            f"wafer_board_spm_calibration_probe_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
