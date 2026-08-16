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
import wafer_runtime_launch_contract as runtime_launch


TARGET_IDENTITY = "wafer-tx81-single-card"
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_instruction_family_probe.c"
PROBE_LL = INPUT_DIR / "wafer_instruction_family_probe.ll"
OUTPUT_INITIAL_CANARY = 0xA5
WORK_DIR_CHILDREN = frozenset(
    {
        "source-program",
        "package",
        "probe-build",
        "raw",
    }
)

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

CT_CAPABILITY_CASE_ID_MIN = 143
CT_CAPABILITY_CASE_ID_MAX = 248
QUALIFICATION_REGRESSION_CASES = (
    "peripheral-argmin-f16",
    "peripheral-argmin-negative-f16-observed",
    "unpool-index-f16",
    "peripheral-bilinear-f16",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument(
        "--suite",
        choices=(
            "safe",
            "exact",
            "observed",
            "ct-capability",
            "qualification-regression",
            "all",
        ),
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
        help="repeat each raw-observation row with an independently tagged sample",
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
    if (
        args.work_dir == args.repo_root
        or args.work_dir in args.repo_root.parents
    ):
        raise RuntimeError("instruction probe work directory is too broad")


def prepare_work_dir(args: argparse.Namespace) -> None:
    """Clear only this driver's known children from a validated work directory."""
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    unknown = [
        child.name
        for child in work_dir.iterdir()
        if child.name not in WORK_DIR_CHILDREN
    ]
    if unknown:
        raise RuntimeError(
            "--work-dir contains unknown entries; refusing cleanup: "
            + ", ".join(sorted(unknown))
        )
    for name in sorted(WORK_DIR_CHILDREN):
        child = work_dir / name
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


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
        if args.suite == "ct-capability":
            selected = tuple(
                case
                for case in catalog.SAFE_CASES
                if CT_CAPABILITY_CASE_ID_MIN
                <= case.case_id
                <= CT_CAPABILITY_CASE_ID_MAX
            )
            if not selected:
                raise RuntimeError("CT capability qualification suite is empty")
            return selected
        if args.suite == "qualification-regression":
            return tuple(
                catalog.CASES_BY_NAME[name]
                for name in QUALIFICATION_REGRESSION_CASES
            )
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
    prepare_work_dir(args)
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
            "--num-partitions=1",
        ],
        timeout_seconds=300,
    )
    if "wrote verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not write the seed package")
    return package


def locate_bindings(
    package: pathlib.Path,
) -> tuple[pathlib.Path, tuple[int, int, int], int]:
    manifest = json.loads((package / "manifest.json").read_text())
    entries = manifest.get("entries")
    modules = manifest.get("modules")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    target = manifest.get("target")
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.GRID_KERNEL_LAUNCH,
        context="instruction probe seed",
    )
    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="instruction probe seed"
    )
    if (
        not isinstance(target, dict)
        or target.get("identity") != TARGET_IDENTITY
        or not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or len(inputs) != 2
        or len(outputs) != 1
    ):
        raise RuntimeError("instruction probe seed manifest is invalid")
    entry = entries[0]
    module = modules[0]
    arguments = entry.get("arguments")
    if (
        entry.get("module") != module.get("id")
        or module.get("exports") != [{"role": "main", "symbol": "main"}]
        or not isinstance(arguments, list)
        or [argument.get("ordinal") for argument in arguments[:3]] != [0, 1, 2]
    ):
        raise RuntimeError("instruction probe ABI arguments are not canonical")
    input_records = sorted(
        inputs, key=lambda record: record.get("role_index")
    )
    output_records = sorted(
        outputs, key=lambda record: record.get("role_index")
    )
    expected = (
        ("inputs", "external_input", "read_only"),
        ("inputs", "external_input", "read_only"),
        ("outputs", "external_output", "write_only"),
    )
    port_ids: list[int] = []
    for record, (table, _kind, _access) in zip(
        (input_records[0], input_records[1], output_records[0]),
        expected,
        strict=True,
    ):
        if (
            not isinstance(record, dict)
            or record.get("bytes") != catalog.RESOURCE_BYTES
            or record.get("alignment") != 256
            or not isinstance(record.get("id"), int)
        ):
            raise RuntimeError(
                f"instruction probe port record is invalid: {record}"
            )
        port_ids.append(record["id"])
    for argument, port_id, (table, kind, access) in zip(
        arguments[:3], port_ids, expected, strict=True
    ):
        if (
            not isinstance(argument, dict)
            or argument.get("kind") != kind
            or argument.get("port") != port_id
            or argument.get("access") != access
        ):
            raise RuntimeError(
                "instruction probe ABI argument for "
                f"{table} port {port_id} is invalid"
            )
    module_path_value = module.get("path")
    if not isinstance(module_path_value, str):
        raise RuntimeError("instruction probe module path is missing")
    module_path = package / module_path_value
    if not module_path.is_file():
        raise RuntimeError("instruction probe seed module is missing")
    if any(other.get("module") != module.get("id") for other in entries):
        raise RuntimeError("instruction probe entries do not share one module")
    return module_path, tuple(port_ids), len(arguments)


def package_completion_kind(package: pathlib.Path) -> str:
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_complete_tile_domain(
        manifest, context="instruction probe"
    )
    return runtime_launch.LOCAL_DRAIN_COMPLETION


def build_probe(
    args: argparse.Namespace,
    package: pathlib.Path,
    module_path: pathlib.Path,
    slots_per_tile: int,
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
            f"-DWAFER_IFP_SLOTS_PER_TILE={slots_per_tile}",
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
            "tx8-kcore-loader-grid",
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


def decode_argmin_diagnostic(
    words: tuple[int, ...], status: int
) -> dict[str, int | bool]:
    rec = catalog.REC
    flags = words[rec["ARGMIN_DIAGNOSTIC_FLAGS"]]
    value_raw = words[rec["ARGMIN_VALUE_RAW"]]
    index_raw = words[rec["ARGMIN_INDEX_RAW"]]
    arrival = words[rec["ARGMIN_ARRIVAL_POLLS"]]
    value_poll = arrival & 0xFFFFFFFF
    index_poll = arrival >> 32
    writeback_polls = words[rec["ARGMIN_WRITEBACK_POLLS"]]
    state_polls = words[rec["ARGMIN_STATE_POLLS"]]
    taskstatus = words[rec["ARGMIN_TASKSTATUS_FIRST_LAST"]]
    ibcounter = words[rec["ARGMIN_IBCOUNTER_FIRST_LAST"]]
    taskstatus_last = taskstatus >> 32
    ibcounter_last = ibcounter >> 32
    value_valid = bool(flags & catalog.ARGMIN_VALUE_VALID)
    index_valid = bool(flags & catalog.ARGMIN_INDEX_VALID)
    task_drained = bool(flags & catalog.ARGMIN_TASK_DRAINED)
    if (
        words[rec["ARGMIN_WRITEBACK_BUDGET"]]
        != catalog.ARGMIN_WRITEBACK_POLL_BUDGET
        or words[rec["ARGMIN_STATE_BUDGET"]]
        != catalog.ARGMIN_STATE_POLL_BUDGET
        or not 1 <= writeback_polls <= catalog.ARGMIN_WRITEBACK_POLL_BUDGET
        or not 1 <= state_polls <= catalog.ARGMIN_STATE_POLL_BUDGET
    ):
        raise RuntimeError("ArgMin diagnostic poll budget record is invalid")
    for name, valid, raw, arrival_poll in (
        ("value", value_valid, value_raw, value_poll),
        ("index", index_valid, index_raw, index_poll),
    ):
        if valid != bool(raw & (1 << 32)):
            raise RuntimeError(
                f"ArgMin diagnostic {name} DATA_VALID flag disagrees with raw"
            )
        if valid != (arrival_poll != 0):
            raise RuntimeError(
                f"ArgMin diagnostic {name} arrival poll is inconsistent"
            )
        if arrival_poll > writeback_polls:
            raise RuntimeError(
                f"ArgMin diagnostic {name} arrival exceeds poll count"
            )
    if task_drained != (taskstatus_last == 1 and ibcounter_last == 0):
        raise RuntimeError("ArgMin diagnostic worker-drain record is inconsistent")
    writeback_exhausted = bool(
        flags & catalog.ARGMIN_WRITEBACK_BUDGET_EXHAUSTED
    )
    state_exhausted = bool(flags & catalog.ARGMIN_STATE_BUDGET_EXHAUSTED)
    if writeback_exhausted != (not value_valid or not index_valid):
        raise RuntimeError("ArgMin diagnostic writeback exhaustion is inconsistent")
    if state_exhausted != (not task_drained):
        raise RuntimeError("ArgMin diagnostic state exhaustion is inconsistent")
    if status == 0:
        if not value_valid or not index_valid or not task_drained:
            raise RuntimeError("completed ArgMin diagnostic lacks bounded evidence")
    elif status == catalog.STATUS_DIAGNOSTIC_BOUNDED:
        if not writeback_exhausted and not state_exhausted:
            raise RuntimeError("bounded ArgMin diagnostic did not exhaust a budget")
    else:
        raise RuntimeError(f"unexpected ArgMin diagnostic status {status}")
    return {
        "value_valid": value_valid,
        "index_valid": index_valid,
        "task_drained": task_drained,
        "value": value_raw & 0xFFFFFFFF,
        "index": index_raw & 0xFFFFFFFF,
        "value_arrival_poll": value_poll,
        "index_arrival_poll": index_poll,
        "writeback_polls": writeback_polls,
        "state_polls": state_polls,
        "writeback_budget_exhausted": writeback_exhausted,
        "state_budget_exhausted": state_exhausted,
    }


def validate_output(
    path: pathlib.Path,
    case: catalog.InstructionCase,
    expected_slot: bytes,
    expected_aux_slot: bytes,
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
    argmin_diagnostic = case.symbol in {
        "PERIPHERAL_ARGMIN_TIE_F16_OBSERVED",
        "PERIPHERAL_ARGMIN_NAN_F16_OBSERVED",
    }
    status = words[rec["STATUS"]]
    valid_status = status == 0 or (
        argmin_diagnostic and status == catalog.STATUS_DIAGNOSTIC_BOUNDED
    )
    if (
        words[rec["MAGIC"]] != catalog.RECORD_MAGIC
        or words[rec["WORD_COUNT"]]
        != catalog.RECORD_WORDS
        or not valid_status
        or actual_mirror != expected_mirror
        or words[rec["REQUEST_GUARD"]] != catalog.REQUEST_GUARD
        or words[rec["OUTPUT_DDR_OFFSET"]] != catalog.OUTPUT_DDR_OFFSET
        or words[rec["SLOT_BYTES"]] != catalog.SLOT_BYTES
        or words[rec["BODY_OFFSET"]] != catalog.BODY_OFFSET
        or words[rec["RECORD_GUARD"]] != catalog.RECORD_GUARD
    ):
        raise RuntimeError(f"{case.name}: record mirror failed")
    select = case.family_name == "CT_SELECT_COMPOSITE"
    repeated_unpool = case.symbol in catalog.REPEATED_UNPOOL_SYMBOLS
    argmin_snapshot = case.symbol in {
        "PERIPHERAL_ARGMIN_TIE_F16_OBSERVED",
        "PERIPHERAL_ARGMIN_NAN_F16_OBSERVED",
    }
    expected_steps = (
        catalog.STEP_TARGET_ISSUED
        | (
            catalog.STEP_FINAL_FENCE_COMPLETED
            if status == 0
            else 0
        )
        | (
            catalog.STEP_BIT2FP_COMPLETED
            | catalog.STEP_MASK_MOVE_ISSUED
            if select
            else 0
        )
        | (
            catalog.STEP_REPEATED_OVERLAP_VALUES_STAGED
            | catalog.STEP_REPEATED_OVERLAP_AUX_SNAPSHOTTED
            if repeated_unpool
            else 0
        )
        | (
            catalog.STEP_ARGMIN_INPUT_SNAPSHOTTED
            if argmin_snapshot and status == 0
            else 0
        )
        | (
            catalog.STEP_ARGMIN_DIAGNOSTIC_RETURNED
            if argmin_diagnostic
            else 0
        )
    )
    if words[rec["STEP_FLAGS"]] != expected_steps:
        raise RuntimeError(
            f"{case.name}: instruction step oracle failed: "
            f"flags={words[rec['STEP_FLAGS']]:#x}"
        )

    actual_slot = raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ]
    actual_aux_slot = raw[
        catalog.AUX_DDR_OFFSET :
        catalog.AUX_DDR_OFFSET + catalog.SLOT_BYTES
    ]
    diagnostic_observation = (
        decode_argmin_diagnostic(words, status)
        if argmin_diagnostic
        else None
    )
    if status == catalog.STATUS_DIAGNOSTIC_BOUNDED:
        bounded = bytearray(raw)
        bounded[: catalog.RECORD_WORDS * 8] = bytes(
            [OUTPUT_INITIAL_CANARY]
        ) * (catalog.RECORD_WORDS * 8)
        if bounded != bytes([OUTPUT_INITIAL_CANARY]) * catalog.RESOURCE_BYTES:
            mismatch = next(
                index
                for index, value in enumerate(bounded)
                if value != OUTPUT_INITIAL_CANARY
            )
            raise RuntimeError(
                f"{case.name}: bounded diagnostic wrote outside its record "
                f"at byte {mismatch}"
            )
        return {
            "case": case.as_dict(),
            "sample": sample,
            "step_flags": words[rec["STEP_FLAGS"]],
            "semantic_observation": {
                "classification": "bounded-diagnostic",
                **diagnostic_observation,
            },
        }
    result_offsets = (
        set(catalog.reduce_exact_result_byte_offsets(case))
        if not case.is_observation
        else set()
    )
    exact_indices = [
        *range(0, catalog.BODY_OFFSET),
        *sorted(result_offsets),
        *range(
            catalog.BODY_OFFSET + case.output_span,
            catalog.SLOT_BYTES,
        ),
    ]
    mismatch = next(
        (
            index
            for index in exact_indices
            if actual_slot[index] != expected_slot[index]
        ),
        None,
    )
    if mismatch is not None:
        domain = (
            "logical result"
            if mismatch in result_offsets
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

    if case.family_name == "CT_SELECT_COMPOSITE":
        expected_select_aux = bytearray(expected_aux_slot)
        struct.pack_into(
            "<128H",
            expected_select_aux,
            catalog.BODY_OFFSET,
            *catalog.bit2fp_expected_words(case),
        )
        if actual_aux_slot != bytes(expected_select_aux):
            mismatch = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_aux_slot, expected_select_aux, strict=True)
                )
                if actual != expected
            )
            raise RuntimeError(
                f"{case.name}: Bit2FP/aux guard differs at slot byte "
                f"{mismatch}: actual=0x{actual_aux_slot[mismatch]:02x}, "
                f"expected=0x{expected_select_aux[mismatch]:02x}"
            )
    elif repeated_unpool:
        expected_repeated_aux = catalog.repeated_unpool_expected_aux_slot()
        snapshot_begin = (
            catalog.BODY_OFFSET
            + catalog.REPEATED_UNPOOL_AUX_SNAPSHOT_OFFSET
        )
        snapshot_end = snapshot_begin + catalog.REPEATED_UNPOOL_AUX_BYTES
        snapshot_mismatch = next(
            (
                index
                for index in range(snapshot_begin, snapshot_end)
                if actual_aux_slot[index] != expected_repeated_aux[index]
            ),
            None,
        )
        if snapshot_mismatch is not None:
            raise RuntimeError(
                f"{case.name}: repeated-overlap indexed-pool "
                f"pre-consumer auxiliary snapshot differs at slot byte "
                f"{snapshot_mismatch}: "
                f"actual=0x{actual_aux_slot[snapshot_mismatch]:02x}, "
                f"expected=0x{expected_repeated_aux[snapshot_mismatch]:02x}"
            )
        if actual_aux_slot != expected_repeated_aux:
            mismatch = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(
                        actual_aux_slot,
                        expected_repeated_aux,
                        strict=True,
                    )
                )
                if actual != expected
            )
            raise RuntimeError(
                f"{case.name}: repeated-overlap post-consumer auxiliary "
                f"or guard differs at slot byte {mismatch}: "
                f"actual=0x{actual_aux_slot[mismatch]:02x}, "
                f"expected=0x{expected_repeated_aux[mismatch]:02x}"
            )
    elif argmin_snapshot:
        if actual_aux_slot != expected_aux_slot:
            mismatch = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_aux_slot, expected_aux_slot, strict=True)
                )
                if actual != expected
            )
            raise RuntimeError(
                f"{case.name}: post-ArgMin input snapshot differs at slot "
                f"byte {mismatch}: actual=0x{actual_aux_slot[mismatch]:02x}, "
                f"expected=0x{expected_aux_slot[mismatch]:02x}"
            )
    else:
        aux_exact_indices = [
            *range(0, catalog.BODY_OFFSET),
            *range(
                catalog.BODY_OFFSET + case.aux_span,
                catalog.SLOT_BYTES,
            ),
        ]
        aux_mismatch = next(
            (
                index
                for index in aux_exact_indices
                if actual_aux_slot[index] != expected_aux_slot[index]
            ),
            None,
        )
        if aux_mismatch is not None:
            raise RuntimeError(
                f"{case.name}: auxiliary SPM guard differs at slot byte "
                f"{aux_mismatch}: actual=0x{actual_aux_slot[aux_mismatch]:02x}, "
                f"expected=0x{expected_aux_slot[aux_mismatch]:02x}"
            )

    mutable = bytearray(raw)
    mutable[: catalog.RECORD_WORDS * 8] = bytes(
        [OUTPUT_INITIAL_CANARY]
    ) * (catalog.RECORD_WORDS * 8)
    mutable[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ] = bytes([OUTPUT_INITIAL_CANARY]) * catalog.SLOT_BYTES
    mutable[
        catalog.AUX_DDR_OFFSET :
        catalog.AUX_DDR_OFFSET + catalog.SLOT_BYTES
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
    argmin_observation = catalog.classify_argmin_domain_observation(
        case, sample, result
    )
    semantic_observation = argmin_observation
    if semantic_observation is None:
        semantic_observation = catalog.classify_repeated_unpool_observation(
            case, sample, result
        )
    if argmin_observation is not None:
        expected_padding = expected_slot[
            catalog.BODY_OFFSET + 2 : catalog.BODY_OFFSET + 4
        ]
        if result[2:4] != expected_padding:
            raise RuntimeError(
                f"{case.name}: ArgMin internal padding changed: "
                f"actual={result[2:4].hex()}, "
                f"expected={expected_padding.hex()}"
            )
        if diagnostic_observation is None:
            raise RuntimeError("ArgMin result omitted its diagnostic record")
        result_value = struct.unpack_from("<H", result)[0]
        result_index = struct.unpack_from("<I", result, 4)[0]
        if (
            diagnostic_observation["value"] & 0xFFFF != result_value
            or diagnostic_observation["index"] != result_index
        ):
            raise RuntimeError(
                f"{case.name}: ArgMin result disagrees with bounded CSR record"
            )
    observation = {
        "case": case.as_dict(),
        "sample": sample,
        "step_flags": words[rec["STEP_FLAGS"]],
        "output_sha256": hashlib.sha256(actual_slot).hexdigest(),
        "result_sha256": hashlib.sha256(result).hexdigest(),
    }
    if semantic_observation is not None:
        observation["semantic_observation"] = semantic_observation
    if diagnostic_observation is not None:
        observation["argmin_diagnostic"] = diagnostic_observation
    return observation


def validate_board_lifecycle(stdout: str, completion_kind: str) -> None:
    lines = stdout.splitlines()
    required = {
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "board_execution: true",
    }
    if not required.issubset(set(lines)):
        raise RuntimeError("wafer-run omitted complete lifecycle evidence")
    if completion_kind != runtime_launch.LOCAL_DRAIN_COMPLETION:
        raise RuntimeError("instruction probe completion kind is invalid")
    runtime_launch.require_board_completion(
        stdout, context="instruction probe"
    )


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    completion_kind: str,
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
            try:
                validate_board_lifecycle(result.stdout, completion_kind)
            except RuntimeError as error:
                raise RuntimeError(f"{case.name}: {error}") from error
            observation = validate_output(
                output,
                case,
                built.expected_output_slot,
                built.payload[
                    3 * catalog.SLOT_BYTES : 4 * catalog.SLOT_BYTES
                ],
                sample,
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
    module_path, resource_ids, slots_per_tile = locate_bindings(package)
    completion_kind = package_completion_kind(package)
    build_probe(args, package, module_path, slots_per_tile)
    verify_no_card(args, package)
    if args.no_card:
        return 0
    execute_cases(
        args, package, resource_ids, completion_kind, selected
    )
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
