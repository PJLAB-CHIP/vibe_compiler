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
from collections.abc import Iterable

import wafer_board_instruction_family_probe_test as package_support
import wafer_memory_descriptor_calibration_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_memory_descriptor_calibration_probe.c"
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
    if args.selected_cases:
        unknown = sorted(set(args.selected_cases) - set(catalog.CASES_BY_NAME))
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
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }


def validate_output(
    path: pathlib.Path,
    case: catalog.MemoryCase,
    built: catalog.CasePayload,
    sample: int,
) -> dict[str, object]:
    raw = path.read_bytes()
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

    observed = b"".join(raw[begin:end] for begin, end in built.allowed_ranges)
    if not case.is_exact and all(
        value == catalog.RESOURCE_CANARY for value in observed
    ):
        raise RuntimeError(
            f"{case.name}: observation ranges remained entirely canary"
        )
    rec = catalog.REC
    return {
        "case": case.as_dict(),
        "sample": sample,
        "result_sha256": hashlib.sha256(observed).hexdigest(),
        "pmu": {
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
            (case.engine_a, case.engine_b, case.spm_b - case.spm_a)
            for case in cases
            if case.kind == catalog.KIND_ENGINE_PAIR
        }
    )
    metric_names = (
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
    )
    for engine_a, engine_b, relative_offset in pair_keys:
        schedule_cases = {
            case.schedule: case
            for case in cases
            if case.kind == catalog.KIND_ENGINE_PAIR
            and case.engine_a == engine_a
            and case.engine_b == engine_b
            and case.spm_b - case.spm_a == relative_offset
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
                    int(row["pmu"][metric]) for row in rows
                )
                for metric in metric_names
            }
        summary: dict[str, object] = {
            "engine_pair": [
                catalog.ENGINE_NAMES[engine_a],
                catalog.ENGINE_NAMES[engine_b],
            ],
            "relative_spm_offset": relative_offset,
            "samples_per_cell": catalog.SPM_BANK_PMU_REPETITIONS,
            "correctness": "all-exact",
            "pmu_medians": schedule_medians,
            "interpretation": (
                "raw repeated paired control; no bank/color class inferred"
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
            "spm_bank_pair_repeat_summary: "
            + json.dumps(summary, sort_keys=True)
        )


def main() -> int:
    args = parse_args()
    if args.list_cases:
        print(
            json.dumps(
                {
                    "cases": [case.as_dict() for case in catalog.CATALOG],
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
