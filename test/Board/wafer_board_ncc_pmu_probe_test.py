#!/usr/bin/env python3
"""Build and optionally execute the TX81 read-only NCC/PMU probe."""

from __future__ import annotations

import argparse
import dataclasses
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

import wafer_runtime_launch_contract as runtime_launch


MODULE = """\
module {
  func.func @main(
      %lhs: tensor<64xf32>,
      %rhs: tensor<64xf32>) -> tensor<64xf32> {
    %result = stablehlo.add %lhs, %rhs : tensor<64xf32>
    return %result : tensor<64xf32>
  }
}
"""

METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [64], "dtype": "float32", "dynamic_dims": []},
        {"shape": [64], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [64], "dtype": "float32", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "lhs"},
        {"type_": "input_arg", "position": 1, "name": "rhs"},
    ],
    "unused_inputs": [],
}

TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
PROBE_BYTES = 256
PROBE_WORDS = 32
PROBE_MAGIC = 0x3130554D50464157
PROBE_SCHEMA = 1
PMU_STABLE_MASK = (1 << 8) - 1
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ncc_pmu_readonly_probe.c"
PROBE_LL = INPUT_DIR / "wafer_ncc_pmu_readonly_probe.ll"


@dataclasses.dataclass(frozen=True)
class ReadOnlyPmuCalibrationCase:
    key: str
    execution_scope: str
    oracle: str
    completion: str
    resource_budget: str


READ_ONLY_PMU_CALIBRATION_CASES = (
    ReadOnlyPmuCalibrationCase(
        "profile-identity-readonly",
        "rank-one-read-only",
        (
            "runtime version+device name+PCI bus+tile count+runtime library "
            "SHA-256 and schema-v1 record"
        ),
        "bounded process deadline+normal runtime cleanup",
        "read-only NCC/PMU register snapshot",
    ),
    ReadOnlyPmuCalibrationCase(
        "stable-read-enable-scope",
        "rank-one-read-only",
        (
            "high-low-high split stability+stable PMU enable/scope mask+"
            "record guard"
        ),
        "bounded process deadline+normal runtime cleanup",
        "read-only NCC/PMU register snapshot",
    ),
)
READ_ONLY_PMU_CASES_BY_KEY = {
    case.key: case for case in READ_ONLY_PMU_CALIBRATION_CASES
}
CALIBRATION_LEAF_BINDINGS = {
    key: (READ_ONLY_PMU_CASES_BY_KEY[key],)
    for key in READ_ONLY_PMU_CASES_BY_KEY
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=10000)
    return parser.parse_args()


def run(
    command: list[str], timeout_seconds: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        for partial in (error.stdout, error.stderr):
            if partial:
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(partial, end="", file=sys.stderr)
        raise RuntimeError(
            "read-only PMU probe exceeded its one-shot outer deadline; "
            "the test will not retry or perform device recovery"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            "command failed with exit code "
            f"{result.returncode}: {shlex.join(command)}"
        )
    return result


def write_source_program(work_dir: pathlib.Path) -> pathlib.Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(MODULE)
    (source / "functions" / "forward.meta").write_text(
        json.dumps(METADATA, separators=(",", ":")) + "\n"
    )
    return source


def compile_seed_package(args: argparse.Namespace, source: pathlib.Path) -> pathlib.Path:
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
            f"--launch-kind={LAUNCH_KIND}",
        ],
        timeout_seconds=300,
    )
    if "published verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not report a verified seed package")
    return package


def validate_manifest(
    package: pathlib.Path,
) -> tuple[dict[str, object], pathlib.Path, list[int], int]:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target = manifest.get("target")
    modules = manifest.get("modules")
    entries = manifest.get("entries")
    resources = manifest.get("resources")
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.RANK_ONE_KERNEL_LAUNCH,
        context="PMU probe seed",
    )
    if (
        manifest.get("rank_count") != 1
        or not isinstance(target, dict)
        or target.get("profile") != TARGET_PROFILE
        or not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(resources, list)
        or len(resources) != 3
    ):
        raise RuntimeError("PMU probe seed package contract is not rank-one exact")

    module = modules[0]
    entry = entries[0]
    if (
        not isinstance(module, dict)
        or module.get("id") != 0
        or module.get("format") != "elf-riscv64"
        or module.get("exports") != [{"role": "main", "symbol": "main"}]
        or not isinstance(module.get("path"), str)
        or entry.get("id") != 0
        or entry.get("rank") != 0
        or entry.get("module") != 0
    ):
        raise RuntimeError("PMU probe seed module/entry contract is invalid")
    module_path = package / module["path"]
    if (
        not module_path.is_file()
        or module_path.parent.resolve() != (package / "modules").resolve()
    ):
        raise RuntimeError("PMU probe module path is not the unique module payload")

    input_ids: list[int] = []
    output_id: int | None = None
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("rank") != 0
            or resource.get("bytes") != PROBE_BYTES
            or resource.get("alignment") != 256
            or resource.get("host_visible") is not True
            or not isinstance(resource.get("id"), int)
        ):
            raise RuntimeError("PMU probe resource contract is invalid")
        role = resource.get("role")
        access = resource.get("access")
        if role == "user_input" and access == "read_only":
            input_ids.append(resource["id"])
        elif role == "output" and access == "write_only" and output_id is None:
            output_id = resource["id"]
        else:
            raise RuntimeError(f"unexpected PMU probe resource: {resource}")
    if len(input_ids) != 2 or output_id is None:
        raise RuntimeError("PMU probe host-visible bindings are incomplete")

    slots = entry.get("slots")
    if (
        not isinstance(slots, list)
        or [slot.get("ordinal") for slot in slots] != [0, 1, 2]
        or [slot.get("resource") for slot in slots]
        != [input_ids[0], input_ids[1], output_id]
    ):
        raise RuntimeError("PMU probe entry must keep output at pointer-table slot 2")
    return manifest, module_path, input_ids, output_id


def build_probe(
    args: argparse.Namespace, package: pathlib.Path, module_path: pathlib.Path
) -> None:
    deps = args.repo_root / "third_party" / "tx8_deps"
    tool_bin = deps / TOOLCHAIN_DIR / "bin"
    gcc = tool_bin / "riscv64-unknown-elf-gcc"
    objcopy = tool_bin / "riscv64-unknown-elf-objcopy"
    device_linker = args.repo_root / "tools" / "wafer_device_link.py"
    required_files = (
        gcc,
        objcopy,
        device_linker,
        args.llvm_clangxx,
        PROBE_C,
        PROBE_LL,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"PMU probe build dependencies are missing: {missing}")

    build = args.work_dir / "probe-build"
    build.mkdir()
    helper = build / "wafer_ncc_pmu_readonly_probe.o"
    linked = build / "wafer_ncc_pmu_readonly_probe.so"
    run(
        [
            str(gcc),
            str(PROBE_C),
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
            "-mcpu=c908",
            "-mabi=lp64d",
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

    staged = module_path.with_name(f".{module_path.name}.pmu-probe")
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(".manifest.json.pmu-probe")
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged_manifest, manifest_path)
    print("probe_build: read_only_ncc_pmu_snapshot")


def validate_board_qualification(args: argparse.Namespace) -> None:
    required = {
        "--expected-runtime-version": args.expected_runtime_version,
        "--expected-device-name": args.expected_device_name,
        "--expected-pci-bus-id": args.expected_pci_bus_id,
        "--expected-tile-count": args.expected_tile_count,
        "--expected-runtime-library-sha256": args.expected_runtime_library_sha256,
    }
    missing = [option for option, value in required.items() if value in (None, "")]
    if missing:
        raise RuntimeError(f"board PMU probe requires explicit qualification: {missing}")
    digest = str(args.expected_runtime_library_sha256)
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise RuntimeError("runtime library SHA-256 must contain 64 hex digits")
    if args.completion_timeout_ms <= 0:
        raise RuntimeError("--completion-timeout-ms must be positive")


def parse_snapshot(path: pathlib.Path) -> dict[str, object]:
    payload = path.read_bytes()
    if len(payload) != PROBE_BYTES:
        raise RuntimeError(
            f"PMU probe output has {len(payload)} bytes, expected {PROBE_BYTES}"
        )
    words = struct.unpack("<32Q", payload)
    schema = words[1] >> 32
    word_count = words[1] & 0xFFFFFFFF
    if (
        words[0] != PROBE_MAGIC
        or schema != PROBE_SCHEMA
        or word_count != PROBE_WORDS
    ):
        raise RuntimeError("PMU probe output header is invalid")
    if words[2] != PMU_STABLE_MASK:
        raise RuntimeError(
            f"PMU split counters were unstable: mask=0x{words[2]:x}"
        )

    controls = words[7:10]
    engines = ("ct", "ne", "rdma", "wdma", "tdma", "scalar")
    return {
        "schema": schema,
        "stable_counter_mask": f"0x{words[2]:02x}",
        "pmu_enable_raw": words[3],
        "serial_mode_raw": list(words[4:7]),
        "serial_mode": [value & 1 for value in words[4:7]],
        "control": [
            {
                "raw": value,
                "ib_count": value & 0xFF,
                "task_done": bool(value & 0x100),
            }
            for value in controls
        ],
        "statistics_window_cycles": words[10],
        "full_execution_cycles": words[11],
        "engine_execution_cycles": dict(zip(engines, words[12:18])),
        "worker0_instruction_count": dict(zip(engines, words[18:24])),
        "worker0_blocking_count": dict(zip(engines, words[24:30])),
        "pmu_base": f"0x{words[30]:x}",
        "ncc_base": f"0x{words[31]:x}",
    }


def execute_board(
    args: argparse.Namespace, package: pathlib.Path, input_ids: list[int], output_id: int
) -> None:
    validate_board_qualification(args)
    raw = args.work_dir / "raw"
    raw.mkdir()
    input_paths = [raw / f"input{index}.f32.raw" for index in range(2)]
    for path in input_paths:
        path.write_bytes(bytes(PROBE_BYTES))
    output = raw / "ncc_pmu_snapshot.raw"
    command = [
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
    ]
    for resource_id, path in zip(input_ids, input_paths):
        command.extend(["--resource", f"{resource_id}={path}"])
    command.extend(["--output", f"{output_id}={output}"])
    result = run(
        command, timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0
    )
    required = (
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "output_capture:",
        "board_execution: true",
    )
    missing = [text for text in required if text not in result.stdout]
    if missing:
        raise RuntimeError(f"board PMU probe omitted runtime evidence: {missing}")
    print("probe_snapshot: " + json.dumps(parse_snapshot(output), sort_keys=True))
    print(result.stdout, end="")


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print(
            "wafer_board_ncc_pmu_probe_test: hardware execution is not armed; "
            "set WAFER_EXECUTE_HARDWARE_TESTS=1",
            file=sys.stderr,
        )
        return 77

    source = write_source_program(args.work_dir)
    package = compile_seed_package(args, source)
    _, module_path, input_ids, output_id = validate_manifest(package)
    build_probe(args, package, module_path)

    no_card = run(
        [
            str(args.wafer_run),
            "--package-dir",
            str(package),
            "--entry-id",
            "0",
            "--no-card",
        ]
    )
    if "board_execution: false" not in no_card.stdout:
        raise RuntimeError("wafer-run did not verify the rewritten probe package")
    print("probe_package_verification: passed")
    if not args.no_card:
        execute_board(args, package, input_ids, output_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"wafer_board_ncc_pmu_probe_test: {error}", file=sys.stderr)
        raise SystemExit(1)
