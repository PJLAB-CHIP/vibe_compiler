#!/usr/bin/env python3
"""Build and run isolated compiler-emitted TX81 instruction qualifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterable

import wafer_instruction_family_catalog as catalog


TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
LAUNCH_ABI = "per-rank-pointer-block-v1"
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_instruction_family_probe.c"
PROBE_LL = INPUT_DIR / "wafer_instruction_family_probe.ll"
OUTPUT_INITIAL_CANARY = 0xA5

MODULE = """\
module {
  func.func @main(
      %lhs: tensor<8192xf16>,
      %rhs: tensor<8192xf16>) -> tensor<8192xf16> {
    %result = stablehlo.add %lhs, %rhs
        : tensor<8192xf16>
    return %result : tensor<8192xf16>
  }
}
"""

METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [8192], "dtype": "float16", "dynamic_dims": []},
        {"shape": [8192], "dtype": "float16", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [8192], "dtype": "float16", "dynamic_dims": []}
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
        choices=("safe", "exact", "observed", "all"),
        default="safe",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="select a safe typed catalog row; repeatable",
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
        "--observation-samples",
        type=int,
        default=3,
        help="repeat each raw-observation row with identical payload",
    )
    return parser.parse_args()


def run(
    command: list[str], timeout_seconds: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        for partial in (error.stdout, error.stderr):
            if partial:
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(partial, end="", file=sys.stderr)
        raise RuntimeError(
            "one isolated instruction context exceeded its outer deadline; "
            "the suite stops immediately without retry, reset, or power calls"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{shlex.join(command)}"
        )
    return result


def require_build_args(args: argparse.Namespace) -> None:
    required = {
        "--repo-root": args.repo_root,
        "--wafer-compile": args.wafer_compile,
        "--wafer-run": args.wafer_run,
        "--llvm-clangxx": args.llvm_clangxx,
        "--work-dir": args.work_dir,
    }
    missing = [option for option, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"probe build requires: {missing}")
    args.repo_root = args.repo_root.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.work_dir in {
        pathlib.Path("/"),
        args.repo_root,
        args.repo_root.parent,
    }:
        raise RuntimeError("instruction probe work directory is too broad")


def require_board_args(args: argparse.Namespace) -> None:
    required = {
        "--expected-runtime-version": args.expected_runtime_version,
        "--expected-device-name": args.expected_device_name,
        "--expected-pci-bus-id": args.expected_pci_bus_id,
        "--expected-tile-count": args.expected_tile_count,
        "--expected-runtime-library-sha256": (
            args.expected_runtime_library_sha256
        ),
    }
    missing = [
        option for option, value in required.items() if value in (None, "")
    ]
    if missing:
        raise RuntimeError(f"board execution requires qualification: {missing}")
    if re.fullmatch(
        r"[0-9a-fA-F]{64}", str(args.expected_runtime_library_sha256)
    ) is None:
        raise RuntimeError("runtime library SHA-256 must contain 64 hex digits")
    if args.completion_timeout_ms <= 0:
        raise RuntimeError("completion timeout must be positive")


def select_cases(args: argparse.Namespace) -> tuple[catalog.InstructionCase, ...]:
    if not args.selected_cases:
        if args.suite in {"safe", "all"}:
            return catalog.SAFE_CASES
        observed = args.suite == "observed"
        selected = tuple(
            case
            for case in catalog.SAFE_CASES
            if case.is_observation == observed
        )
        if not selected:
            raise RuntimeError(
                f"instruction qualification suite {args.suite} is empty"
            )
        return selected
    unknown = [
        name for name in args.selected_cases if name not in catalog.CASES_BY_NAME
    ]
    if unknown:
        raise RuntimeError(f"unknown instruction qualification cases: {unknown}")
    selected = tuple(
        catalog.CASES_BY_NAME[name] for name in args.selected_cases
    )
    deferred = [case.name for case in selected if not case.is_safe]
    if deferred:
        raise RuntimeError(
            "deferred catalog rows cannot enter the safe board suite: "
            f"{deferred}"
        )
    if len({case.name for case in selected}) != len(selected):
        raise RuntimeError("instruction qualification case selection repeats a row")
    return selected


def write_source_program(args: argparse.Namespace) -> pathlib.Path:
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    source = args.work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(MODULE)
    (source / "functions" / "forward.meta").write_text(
        json.dumps(METADATA, separators=(",", ":")) + "\n"
    )
    return source


def compile_seed_package(
    args: argparse.Namespace, source: pathlib.Path
) -> pathlib.Path:
    package = args.work_dir / "package"
    result = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(package),
            "--execution-ranks=1",
            f"--target-profile={TARGET_PROFILE}",
            f"--launch-abi={LAUNCH_ABI}",
        ],
        timeout_seconds=300,
    )
    if "published verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not publish the seed package")
    return package


def locate_bindings(
    package: pathlib.Path,
) -> tuple[pathlib.Path, tuple[int, int, int]]:
    manifest = json.loads((package / "manifest.json").read_text())
    entries = manifest.get("entries")
    modules = manifest.get("modules")
    resources = manifest.get("resources")
    target = manifest.get("target")
    if (
        manifest.get("schema_version") != 5
        or manifest.get("rank_count") != 1
        or not isinstance(target, dict)
        or target.get("profile") != TARGET_PROFILE
        or target.get("launch_abi") != LAUNCH_ABI
        or not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(resources, list)
        or len(resources) != 3
    ):
        raise RuntimeError("instruction probe seed manifest is not rank-one exact")
    entry = entries[0]
    module = modules[0]
    slots = entry.get("slots")
    if (
        entry.get("rank") != 0
        or entry.get("module") != module.get("id")
        or module.get("exports") != [{"role": "main", "symbol": "main"}]
        or not isinstance(slots, list)
        or [slot.get("ordinal") for slot in slots] != [0, 1, 2]
    ):
        raise RuntimeError("instruction probe ABI slots are not canonical")
    resources_by_id = {
        resource.get("id"): resource
        for resource in resources
        if isinstance(resource, dict) and isinstance(resource.get("id"), int)
    }
    resource_ids = tuple(slot.get("resource") for slot in slots)
    expected = (
        ("user_input", "read_only"),
        ("user_input", "read_only"),
        ("output", "write_only"),
    )
    for resource_id, role_access in zip(resource_ids, expected, strict=True):
        resource = resources_by_id.get(resource_id)
        if (
            resource is None
            or (resource.get("role"), resource.get("access")) != role_access
            or resource.get("bytes") != catalog.RESOURCE_BYTES
            or resource.get("alignment") != 256
            or resource.get("host_visible") is not True
        ):
            raise RuntimeError(
                f"instruction probe resource {resource_id} is invalid"
            )
    module_path_value = module.get("path")
    if not isinstance(module_path_value, str):
        raise RuntimeError("instruction probe module path is missing")
    module_path = package / module_path_value
    if not module_path.is_file():
        raise RuntimeError("instruction probe seed module is missing")
    return module_path, resource_ids


def build_probe(
    args: argparse.Namespace, package: pathlib.Path, module_path: pathlib.Path
) -> None:
    deps = args.repo_root / "third_party" / "tx8_deps"
    tool_bin = deps / TOOLCHAIN_DIR / "bin"
    gcc = tool_bin / "riscv64-unknown-elf-gcc"
    objcopy = tool_bin / "riscv64-unknown-elf-objcopy"
    device_linker = args.repo_root / "tools" / "wafer_device_link.py"
    required = (
        gcc,
        objcopy,
        device_linker,
        args.llvm_clangxx,
        PROBE_C,
        PROBE_LL,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"instruction probe build dependencies are missing: {missing}"
        )

    build = args.work_dir / "probe-build"
    build.mkdir()
    helper = build / "wafer_instruction_family_probe.o"
    linked = build / "wafer_instruction_family_probe.so"
    run(
        [
            str(gcc),
            "-std=c11",
            "-O2",
            "-c",
            "-fPIC",
            "-ffreestanding",
            "-fno-stack-protector",
            "-ffunction-sections",
            "-fdata-sections",
            "-fvisibility=hidden",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DCONFIG_NO_PLATFORM_HOOK_H",
            "-DUSING_RISCV",
            f"-I{args.repo_root / 'runtime' / 'wafer_crt' / 'include'}",
            f"-I{args.repo_root / 'include'}",
            f"-I{deps / 'include'}",
            f"-I{INPUT_DIR}",
            "-mcpu=c908",
            "-mabi=lp64d",
            str(PROBE_C),
            "-o",
            str(helper),
        ]
    )
    run([str(objcopy), "-R", ".riscv.attributes", str(helper)])
    run(
        [
            sys.executable,
            str(device_linker),
            "--llvm-ir",
            str(PROBE_LL),
            "--llvm-clangxx",
            str(args.llvm_clangxx),
            "--output",
            str(linked),
            "--loader-abi",
            "tx8-kcore-loader-v1",
            "--extra-object",
            str(helper),
        ],
        timeout_seconds=120,
    )

    staged = module_path.with_name(f".{module_path.name}.instruction-probe")
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    module_id = manifest["entries"][0]["module"]
    matching = [
        module for module in manifest["modules"] if module.get("id") == module_id
    ]
    if len(matching) != 1:
        raise RuntimeError("instruction probe module record is ambiguous")
    matching[0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(".manifest.json.instruction-probe")
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged_manifest, manifest_path)


def verify_no_card(args: argparse.Namespace, package: pathlib.Path) -> None:
    result = run(
        [
            str(args.wafer_run),
            "--package-dir",
            str(package),
            "--entry-id",
            "0",
            "--no-card",
        ]
    )
    if "board_execution: false" not in result.stdout:
        raise RuntimeError("wafer-run did not verify the rewritten probe package")
    print("instruction_family_probe_build: passed")


def board_command(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    request: pathlib.Path,
    payload: pathlib.Path,
    output: pathlib.Path,
) -> list[str]:
    return [
        str(args.wafer_run),
        "--package-dir",
        str(package),
        "--entry-id",
        "0",
        "--board",
        "--device-id",
        str(args.device_id),
        "--expected-runtime-version",
        str(args.expected_runtime_version),
        "--expected-device-name",
        str(args.expected_device_name),
        "--expected-pci-bus-id",
        str(args.expected_pci_bus_id),
        "--expected-tile-count",
        str(args.expected_tile_count),
        "--expected-runtime-library-sha256",
        str(args.expected_runtime_library_sha256),
        "--completion-timeout-ms",
        str(args.completion_timeout_ms),
        "--resource",
        f"{resource_ids[0]}={request}",
        "--resource",
        f"{resource_ids[1]}={payload}",
        "--output",
        f"{resource_ids[2]}={output}",
    ]


def validate_output(
    path: pathlib.Path,
    case: catalog.InstructionCase,
    expected_slot: bytes,
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
    expected_mirror = (
        case.case_id,
        case.disposition,
        case.family,
        case.dtype,
        case.oracle,
        case.result_bytes,
        case.output_span,
        case.aux_span,
        sample,
    )
    actual_mirror = (
        words[rec["CASE"]],
        words[rec["DISPOSITION"]],
        words[rec["FAMILY"]],
        words[rec["DTYPE"]],
        words[rec["ORACLE"]],
        words[rec["RESULT_BYTES"]],
        words[rec["OUTPUT_SPAN"]],
        words[rec["AUX_SPAN"]],
        words[rec["SAMPLE"]],
    )
    if (
        words[rec["MAGIC"]] != catalog.RECORD_MAGIC
        or words[rec["SCHEMA_AND_WORDS"]]
        != (catalog.SCHEMA << 32) | catalog.RECORD_WORDS
        or words[rec["STATUS"]] != 0
        or actual_mirror != expected_mirror
        or words[rec["OUTPUT_GUARD_MISMATCHES"]] != 0
        or words[rec["AUX_GUARD_MISMATCHES"]] != 0
        or words[rec["REQUEST_GUARD"]] != catalog.REQUEST_GUARD
        or words[rec["OUTPUT_DDR_OFFSET"]] != catalog.OUTPUT_DDR_OFFSET
        or words[rec["SLOT_BYTES"]] != catalog.SLOT_BYTES
        or words[rec["BODY_OFFSET"]] != catalog.BODY_OFFSET
        or words[rec["RECORD_GUARD"]] != catalog.RECORD_GUARD
    ):
        raise RuntimeError(f"{case.name}: record or SPM guard oracle failed")
    select = case.family_name == "CT_SELECT_COMPOSITE"
    expected_steps = (
        catalog.STEP_TARGET_ISSUED
        | catalog.STEP_FINAL_FENCE_COMPLETED
        | (
            catalog.STEP_BIT2FP_COMPLETED
            | catalog.STEP_MASK_MOVE_ISSUED
            if select
            else 0
        )
    )
    if (
        words[rec["STEP_FLAGS"]] != expected_steps
        or words[rec["BIT2FP_MISMATCHES"]] != 0
    ):
        raise RuntimeError(
            f"{case.name}: Bit2FP/MaskMove step oracle failed: "
            f"flags={words[rec['STEP_FLAGS']]:#x}, "
            f"bit2fp_mismatches={words[rec['BIT2FP_MISMATCHES']]}"
        )

    actual_slot = raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ]
    exact_ranges = [
        (0, catalog.BODY_OFFSET),
        (
            catalog.BODY_OFFSET + case.output_span,
            catalog.SLOT_BYTES,
        ),
    ]
    if not case.is_observation:
        exact_ranges.insert(
            1,
            (
                catalog.BODY_OFFSET,
                catalog.BODY_OFFSET + case.result_bytes,
            ),
        )
    mismatch = next(
        (
            index
            for begin, end in exact_ranges
            for index in range(begin, end)
            if actual_slot[index] != expected_slot[index]
        ),
        None,
    )
    if mismatch is not None:
        domain = (
            "logical result"
            if catalog.BODY_OFFSET
            <= mismatch
            < catalog.BODY_OFFSET + case.result_bytes
            else "SPM guard"
        )
        raise RuntimeError(
            f"{case.name}: {domain} differs at slot byte {mismatch}: "
            f"actual=0x{actual_slot[mismatch]:02x}, "
            f"expected=0x{expected_slot[mismatch]:02x}"
        )
    if (
        not case.is_observation
        and case.result_bytes == case.output_span
        and actual_slot != expected_slot
    ):
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(
                zip(actual_slot, expected_slot, strict=True)
            )
            if actual != expected
        )
        raise RuntimeError(
            f"{case.name}: full output slot differs at byte {mismatch}: "
            f"actual=0x{actual_slot[mismatch]:02x}, "
            f"expected=0x{expected_slot[mismatch]:02x}"
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
    result = actual_slot[
        catalog.BODY_OFFSET :
        catalog.BODY_OFFSET + case.output_span
    ]
    if (
        case.is_observation
        and result
        == expected_slot[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + case.output_span
        ]
    ):
        raise RuntimeError(
            f"{case.name}: observation completed without any bounded "
            "writeback"
        )
    return {
        "case": case.as_dict(),
        "sample": sample,
        "step_flags": words[rec["STEP_FLAGS"]],
        "bit2fp_mismatches": words[rec["BIT2FP_MISMATCHES"]],
        "output_sha256": hashlib.sha256(actual_slot).hexdigest(),
        "result_sha256": hashlib.sha256(result).hexdigest(),
    }


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.InstructionCase],
) -> None:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    if args.observation_samples < 1:
        raise RuntimeError("--observation-samples must be at least one")
    for case in cases:
        samples = args.observation_samples if case.is_observation else 1
        for sample in range(samples):
            built = catalog.build_case_payload(case, sample)
            stem = f"{case.name}.sample-{sample}"
            request = raw_dir / f"{stem}.request.raw"
            payload = raw_dir / f"{stem}.payload.raw"
            output = raw_dir / f"{stem}.output.raw"
            request.write_bytes(built.request)
            payload.write_bytes(built.payload)
            result = run(
                board_command(
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
            observation = validate_output(
                output, case, built.expected_output_slot, sample
            )
            print(
                "instruction_family_qualification: "
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
    require_build_args(args)
    selected = select_cases(args)
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "wafer_board_instruction_family_probe_test: hardware "
                "execution is not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        require_board_args(args)

    source = write_source_program(args)
    package = compile_seed_package(args, source)
    module_path, resource_ids = locate_bindings(package)
    build_probe(args, package, module_path)
    verify_no_card(args, package)
    if args.no_card:
        return 0
    execute_cases(args, package, resource_ids, selected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            f"wafer_board_instruction_family_probe_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
