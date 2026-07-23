#!/usr/bin/env python3
"""Build one package and execute four-direction cache/coherence calibration."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
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
    if not args.selected_cases:
        return catalog.CATALOG
    unknown = sorted(set(args.selected_cases) - set(catalog.CASES_BY_NAME))
    if unknown:
        raise RuntimeError(f"unknown cache/coherence cases: {unknown}")
    return tuple(catalog.CASES_BY_NAME[name] for name in args.selected_cases)


def _classify_before(
    case: catalog.CacheCoherenceCase,
    built: catalog.CasePayload,
    raw: bytes,
    mismatch_before: int,
) -> str:
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
    if case.case_id in (0, 2):
        return "new-visible-before-control" if mismatch_before == 0 else (
            "stale-or-mixed-before-control"
        )
    return "not-sampled"


def validate_output(
    path: pathlib.Path,
    case: catalog.CacheCoherenceCase,
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
        "SAMPLE": sample,
        "PAYLOAD_BYTES": catalog.PAYLOAD_BYTES,
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
        "CACHE_CONTROL_MASK": EXPECTED_CACHE_MASK[case.case_id],
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
    if case.output_regions >= 1:
        begin = catalog.OUTPUT0_OFFSET
        end = begin + catalog.PAYLOAD_BYTES
        before_observation = bytes(mutable[begin:end])
        if case.case_id in (2, 3) and before_observation != built.post_control_expected:
            raise RuntimeError(f"{case.name}: WDMA output is not host-exact")
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
        "pre_control_state": _classify_before(
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
    for case in cases:
        for sample in range(case.phases):
            built = catalog.build_case_payload(case, sample)
            prefix = f"{case.name}.phase-{sample}"
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
            print(
                "cache_coherence_calibration: "
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
