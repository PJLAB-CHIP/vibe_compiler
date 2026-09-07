#!/usr/bin/env python3
"""Calibrate ordered Direct-DTE/NCC execution on the 16-Tile TX81 domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import shlex
import statistics
import struct
import subprocess
import sys
import time

import torch

PYTORCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "PyTorch"
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))
import wafer_pytorch_board_common as torch_common

import wafer_runtime_launch_contract as runtime_launch
import wafer_board_source_program as source_program
import wafer_transport_pmu_calibration_catalog as transport_catalog


TILE_COUNT = 16
RESOURCE_BYTES = 264448
HEADER_BYTES = 128
INPUT_BYTES = 8192
MAX_PAYLOAD_BYTES = 4096
ASYNC_TRANSPORT_BYTES = 65536
SPM_GUARD_BYTES = 256
HOST_SLOT_BYTES = ASYNC_TRANSPORT_BYTES + 2 * SPM_GUARD_BYTES
HOST_SLOT_COUNT = 4
PAYLOAD_SWEEP = transport_catalog.PAYLOAD_SWEEP
INITIAL_CANARY = 0xA5
MAGIC = 0x3143434E45544457
CANARY = 0xD7E0CA11D7E0CA11
STATUS_SUCCESS = 1
STATUS_TRANSPORT_ERROR = 2
DTE_ENABLE_MASK = 0x3
SPM_ENABLE_MASK = 0x1F
TRANSPORT_COUNTER_NAMES = transport_catalog.BOARD_COUNTER_NAMES
TRANSPORT_STABLE_MASK = (1 << len(TRANSPORT_COUNTER_NAMES)) - 1
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = (
    transport_catalog.CALIBRATION_LEAF_BINDINGS
)
TARGET_IDENTITY = "wafer-tx81-single-card"
STATUS_ABI = "wafer-direct-dte-status"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_dte_ncc_execution_probe.c"
PROBE_LL = INPUT_DIR / "wafer_dte_ncc_execution_probe.ll"
MODES = dict(transport_catalog.MODE_NAMES)
TRANSPORT_PMU_MODES = transport_catalog.TRANSPORT_PMU_MODES
ERROR_PATH_MODES = transport_catalog.ERROR_PATH_MODES
ASYNC_SENDER_MODES = transport_catalog.ASYNC_SENDER_MODES
ISOLATED_DTE_MODES = transport_catalog.ISOLATED_DTE_MODES
FOUR_SOURCE_FANIN_MODE = transport_catalog.FOUR_SOURCE_FANIN_MODE
RAW_MULTIDEST_MODES = transport_catalog.RAW_MULTIDEST_MODES
RAW_MULTIDEST_CASE_BY_MODE = transport_catalog.RAW_MULTIDEST_CASE_BY_MODE
PENDING_DTE_MODES = (FOUR_SOURCE_FANIN_MODE, *RAW_MULTIDEST_MODES)
DEFAULT_SAFE_MODES = tuple(
    mode
    for mode in MODES
    if mode not in (*ISOLATED_DTE_MODES, *PENDING_DTE_MODES)
)
ASYNC_SENDER_PAYLOAD_BYTES = (
    transport_catalog.ASYNC_SENDER_PAYLOAD_BYTES
)
ASYNC_SENDER_TRANSPORT_BYTES = (
    transport_catalog.ASYNC_SENDER_TRANSPORT_BYTES
)
ASYNC_SENDER_REPETITIONS = transport_catalog.ASYNC_SENDER_REPETITIONS
MODE_PAYLOADS = transport_catalog.MODE_PAYLOADS
CONTRACT_STATUS_ERROR = 1 << 0
CONTRACT_VALID_SEND_EVENT = 1 << 1
CONTRACT_VALID_RECV_EVENT = 1 << 2
CONTRACT_REJECTED_SEND_EVENT = 1 << 3
CONTRACT_REJECTED_RECV_EVENT = 1 << 4
CONTRACT_UNKNOWN_WAIT_RETURNED = 1 << 5
CONTRACT_RAW_ASYNC_ISSUED = 1 << 6
CONTRACT_RAW_ASYNC_COMPLETED = 1 << 7
RAW_ASYNC_RC_MARKER = 0x4153594E
RAW_MULTIDEST_RC_MARKER = 0x4D445445
EXPECTED_INSTRUCTION_COUNTS = {
    1: (1, 1, 0),
    2: (1, 1, 0),
    3: (1, 2, 0),
    4: (1, 2, 0),
    5: (0, 2, 1),
    6: (0, 1, 0),
    7: (0, 1, 0),
    8: (0, 0, 0),
    9: (0, 0, 0),
    10: (1, 0, 0),
    11: (1, 0, 0),
    12: (0, 1, 0),
    14: (0, 0, 0),
    **{mode: (0, 0, 0) for mode in RAW_MULTIDEST_MODES},
}
EXPECTED_CONTRACT_EVIDENCE = {
    **{mode: 0 for mode in TRANSPORT_PMU_MODES},
    7: (
        CONTRACT_STATUS_ERROR
        | CONTRACT_VALID_SEND_EVENT
        | CONTRACT_VALID_RECV_EVENT
        | CONTRACT_REJECTED_SEND_EVENT
    ),
    8: (
        CONTRACT_STATUS_ERROR
        | CONTRACT_REJECTED_SEND_EVENT
        | CONTRACT_REJECTED_RECV_EVENT
    ),
    9: CONTRACT_STATUS_ERROR | CONTRACT_UNKNOWN_WAIT_RETURNED,
    12: (
        CONTRACT_STATUS_ERROR
        | CONTRACT_VALID_SEND_EVENT
        | CONTRACT_VALID_RECV_EVENT
        | CONTRACT_REJECTED_RECV_EVENT
    ),
    **{
        mode: (
            CONTRACT_VALID_RECV_EVENT
            | CONTRACT_RAW_ASYNC_ISSUED
            | CONTRACT_RAW_ASYNC_COMPLETED
        )
        for mode in ASYNC_SENDER_MODES
    },
}

MODULE = f"""\
module {{
  func.func @main(%input: tensor<{RESOURCE_BYTES // 2}xf16>)
      -> tensor<{RESOURCE_BYTES // 2}xf16> {{
    %output = stablehlo.add %input, %input
        : tensor<{RESOURCE_BYTES // 2}xf16>
    return %output : tensor<{RESOURCE_BYTES // 2}xf16>
  }}
}}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {
            "shape": [RESOURCE_BYTES // 2],
            "dtype": "float16",
            "dynamic_dims": [],
        }
    ],
    "output_signature": [
        {
            "shape": [RESOURCE_BYTES // 2],
            "dtype": "float16",
            "dynamic_dims": [],
        }
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "probe_input"}
    ],
    "unused_inputs": [],
}


def expected_contract_evidence(mode: int, tile_id: int) -> int:
    if mode == FOUR_SOURCE_FANIN_MODE:
        if tile_id == transport_catalog.FOUR_SOURCE_FANIN_TARGET:
            return (
                CONTRACT_VALID_RECV_EVENT
                | CONTRACT_RAW_ASYNC_ISSUED
                | CONTRACT_RAW_ASYNC_COMPLETED
            )
        if tile_id in transport_catalog.FOUR_SOURCE_FANIN_SOURCES:
            return CONTRACT_VALID_SEND_EVENT | CONTRACT_RAW_ASYNC_COMPLETED
        return 0
    if mode in RAW_MULTIDEST_MODES:
        case = RAW_MULTIDEST_CASE_BY_MODE[mode]
        if tile_id == transport_catalog.RAW_MULTIDEST_SOURCE_TILE:
            return (
                CONTRACT_VALID_SEND_EVENT
                | CONTRACT_RAW_ASYNC_ISSUED
                | CONTRACT_RAW_ASYNC_COMPLETED
            )
        if tile_id in case.target_tiles:
            return CONTRACT_VALID_RECV_EVENT | CONTRACT_RAW_ASYNC_COMPLETED
        return 0
    return EXPECTED_CONTRACT_EVIDENCE[mode]


def transport_bytes_for_mode(mode: int, payload_bytes: int) -> int:
    if mode in ASYNC_SENDER_MODES:
        return ASYNC_SENDER_TRANSPORT_BYTES
    if mode == FOUR_SOURCE_FANIN_MODE:
        return (
            len(transport_catalog.FOUR_SOURCE_FANIN_SOURCES) * payload_bytes
        )
    if mode in RAW_MULTIDEST_MODES:
        case = RAW_MULTIDEST_CASE_BY_MODE[mode]
        return case.fanout * case.per_destination_bytes
    return payload_bytes


DEFAULT_NO_CARD_TOTAL_TIMEOUT_SECONDS = 420.0
DEFAULT_BOARD_TOTAL_TIMEOUT_SECONDS = 1800.0
_total_deadline: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--mode", type=int, action="append", dest="selected_modes")
    parser.add_argument(
        "--payload-bytes",
        type=int,
        action="append",
        dest="selected_payload_bytes",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--host-contract-only", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--total-timeout-seconds", type=float)
    return parser.parse_args()


def require_build_args(args: argparse.Namespace) -> None:
    required = {
        "--repo-root": args.repo_root,
        "--wafer-compile": args.wafer_compile,
        "--wafer-run": args.wafer_run,
        "--llvm-clangxx": args.llvm_clangxx,
        "--work-dir": args.work_dir,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise RuntimeError(
            "DTE/NCC probe build requires " + ", ".join(missing)
        )


def select_execution_axes(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    modes = tuple(dict.fromkeys(args.selected_modes or DEFAULT_SAFE_MODES))
    payload_bytes = tuple(
        dict.fromkeys(args.selected_payload_bytes or PAYLOAD_SWEEP)
    )
    invalid_modes = tuple(mode for mode in modes if mode not in MODES)
    invalid_payloads = tuple(
        size for size in payload_bytes if size not in PAYLOAD_SWEEP
    )
    if invalid_modes:
        raise RuntimeError(f"unsupported DTE/NCC modes {invalid_modes}")
    if invalid_payloads:
        raise RuntimeError(
            f"unsupported transport payload sizes {invalid_payloads}"
        )
    if any(mode in ISOLATED_DTE_MODES for mode in modes) and (
        len(modes) != 1 or modes[0] not in ISOLATED_DTE_MODES
    ):
        raise RuntimeError(
            "receiver-unprepared must run as the only isolated DTE mode"
        )
    if any(mode in PENDING_DTE_MODES for mode in modes) and (
        len(modes) != 1 or modes[0] not in PENDING_DTE_MODES
    ):
        raise RuntimeError(
            "each pending fan-in/raw-multidestination case must run alone"
        )
    if not any(
        payload in MODE_PAYLOADS[mode]
        for mode in modes
        for payload in payload_bytes
    ):
        raise RuntimeError(
            "selected DTE/NCC modes and payload sizes have no executable case"
        )
    return modes, payload_bytes


def start_total_deadline(timeout_seconds: float) -> None:
    global _total_deadline
    if timeout_seconds <= 0:
        raise RuntimeError("--total-timeout-seconds must be positive")
    _total_deadline = time.monotonic() + timeout_seconds


def run(
    command: list[str], timeout_seconds: float | None = None
) -> subprocess.CompletedProcess[str]:
    if _total_deadline is not None:
        remaining = _total_deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "ordered DTE/NCC probe exhausted its one-shot total "
                "deadline; the batch stops without retry, reset, or power "
                "operations"
            )
        timeout_seconds = (
            remaining
            if timeout_seconds is None
            else min(timeout_seconds, remaining)
        )
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
            "ordered DTE/NCC probe exceeded its one-shot outer deadline; "
            "the batch stops without retry, reset, or power operations"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{shlex.join(command)}"
        )
    return result


def compile_package(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, pathlib.Path, dict[tuple[int, str, int], int]]:
    source = args.work_dir / "source-program"
    if source.exists():
        shutil.rmtree(source)
    (source / "data").mkdir(parents=True)
    source_program.write_program(source, MODULE, METADATA)
    package = args.work_dir / "package"
    if package.exists():
        shutil.rmtree(package)
    result = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-dir",
            str(package),
            "--num-partitions=1",
        ],
        timeout_seconds=300,
    )
    if "wrote verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not write the seed package")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shared_bindings = runtime_launch.configure_direct_dte_tile_package(
        manifest,
        ports=(
            runtime_launch.SharedBoundaryPortSpec(
                table="inputs",
                role_index=0,
                logical_dtype="i8",
                logical_shape=[TILE_COUNT * RESOURCE_BYTES],
                dtype="i8",
                layout="tensor",
                shape=[TILE_COUNT * RESOURCE_BYTES],
                bytes=TILE_COUNT * RESOURCE_BYTES,
                alignment=256,
                access="read_only",
            ),
            runtime_launch.SharedBoundaryPortSpec(
                table="outputs",
                role_index=0,
                logical_dtype="i8",
                logical_shape=[TILE_COUNT * RESOURCE_BYTES],
                dtype="i8",
                layout="tensor",
                shape=[TILE_COUNT * RESOURCE_BYTES],
                bytes=TILE_COUNT * RESOURCE_BYTES,
                alignment=256,
                access="write_only",
            ),
        ),
        status_abi=STATUS_ABI,
        status_bytes=STATUS_STORAGE_BYTES,
        status_alignment=STATUS_STORAGE_ALIGNMENT,
        context="ordered DTE/NCC probe",
    )
    bindings: dict[tuple[int, str, int], int] = {}
    for (table, role_index), port_id in shared_bindings.items():
        role = "user_input" if table == "inputs" else "output"
        for tile_id in range(TILE_COUNT):
            bindings[(tile_id, role, role_index)] = port_id
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest.get("target", {}).get("identity") != TARGET_IDENTITY:
        raise RuntimeError("ordered DTE/NCC package target identity is invalid")
    for table in ("inputs", "outputs"):
        records = manifest.get(table)
        if not isinstance(records, list) or len(records) != 1:
            raise RuntimeError("DTE/NCC probe port table is not exact")
        for record in records:
            if (
                not isinstance(record, dict)
                or record.get("bytes") != TILE_COUNT * RESOURCE_BYTES
            ):
                raise RuntimeError(
                    "DTE/NCC probe host ports do not match the bounded "
                    f"{RESOURCE_BYTES}-byte record"
                )
    module_path = package / manifest["modules"][0]["path"]
    return package, module_path, bindings


def verify_no_card(args: argparse.Namespace, package: pathlib.Path) -> None:
    result = run(
        [
            str(args.wafer_run),
            "--package-dir",
            str(package),
            "--no-card",
            "--direct-dte-status-abi",
            STATUS_ABI,
            "--supports-host-watchdog",
        ]
    )
    if "board_execution: false" not in result.stdout:
        raise RuntimeError("no-card verification omitted the kernel invocation")


def board_base_command(
    args: argparse.Namespace, package: pathlib.Path
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
    ]


def build_probe(
    args: argparse.Namespace, package: pathlib.Path, module_path: pathlib.Path
) -> None:
    deps = args.repo_root / "third_party" / "tx8_deps"
    kcore_include = (
        deps
        / "tx8-yoc-rt-thread-smp"
        / "include"
        / "components"
        / "oplib_tx81"
        / "riscv"
        / "riscv"
        / "include"
    )
    peripheral_include = (
        deps
        / "tx8-yoc-rt-thread-smp"
        / "interface"
        / "op_fw_sim_if"
        / "peripheral"
        / "include"
    )
    board_config_include = (
        deps
        / "tx8-yoc-rt-thread-smp"
        / "include"
        / "bsp"
        / "xuantie_riscv_tx81"
        / "board_riscv_tx81"
        / "include"
    )
    tool_bin = deps / TOOLCHAIN_DIR / "bin"
    gcc = tool_bin / "riscv64-unknown-elf-gcc"
    objcopy = tool_bin / "riscv64-unknown-elf-objcopy"
    objdump = tool_bin / "riscv64-unknown-elf-objdump"
    nm = tool_bin / "riscv64-unknown-elf-nm"
    device_linker = args.repo_root / "tools" / "wafer_device_link.py"
    pmu_register_header = kcore_include / "pmu" / "pmu_reg.h"
    required = (
        gcc,
        objcopy,
        objdump,
        nm,
        device_linker,
        args.llvm_clangxx,
        PROBE_C,
        PROBE_LL,
        pmu_register_header,
        board_config_include / "tx8_config.h",
        peripheral_include / "direct_dte_and_fsm.h",
        peripheral_include / "tx81_spm.h",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"DTE/NCC probe build dependencies are missing: {missing}")

    build = args.work_dir / "probe-build"
    build.mkdir(exist_ok=True)
    helper = build / "wafer_dte_ncc_execution_probe.o"
    linked = build / "wafer_dte_ncc_execution_probe.so"
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
            f"-I{args.repo_root / 'runtime' / 'crt' / 'include'}",
            f"-I{args.repo_root / 'include'}",
            f"-I{deps / 'include'}",
            f"-I{kcore_include}",
            f"-I{peripheral_include}",
            f"-I{board_config_include}",
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
            "tx8-kcore-loader-cluster",
            "--extra-object",
            str(helper),
        ],
        timeout_seconds=120,
    )
    symbols = run([
        str(nm), "--dynamic", "--defined-only", "--format=posix", str(linked)
    ]).stdout
    runtime_launch.require_kernel_exports(
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        {line.split()[0] for line in symbols.splitlines() if line.split()},
        context="DTE/NCC linked probe",
    )
    print("probe_elf_kernel_exports: verified")
    raw_program_disassembly = run(
        [
            str(objdump),
            "-dr",
            "--disassemble=wafer_probe_raw_multidest_program",
            str(linked),
        ]
    ).stdout
    if "<wafer_probe_raw_multidest_program>:" not in raw_program_disassembly:
        raise RuntimeError(
            "final probe ELF dropped the owner-backed raw DTE program"
        )
    required_register_stores = {
        "mode": 20,
        "length": 24,
        "dest_num": 28,
        "cmd_valid": 56,
    }
    missing_register_stores = [
        name
        for name, offset in required_register_stores.items()
        if re.search(
            rf"\bsw\s+[^,\n]+,\s*{offset}\([^)\n]+\)",
            raw_program_disassembly,
        )
        is None
    ]
    if missing_register_stores:
        raise RuntimeError(
            "final probe ELF does not materialize raw DTE register stores: "
            + ", ".join(missing_register_stores)
        )
    if re.search(
        r"\bslli\s+[^,\n]+,\s*[^,\n]+,\s*0x11\b",
        raw_program_disassembly,
    ) is None:
        raise RuntimeError(
            "final probe ELF dropped the remote-SPM route transform"
        )
    if len(re.findall(r"\bsw\s+", raw_program_disassembly)) < 12:
        raise RuntimeError(
            "final probe ELF does not retain multidestination address/user "
            "register programming"
        )
    print("raw_multidest_elf_verification: passed")

    staged = module_path.with_name(f".{module_path.name}.dte-ncc-probe")
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(".manifest.json.dte-ncc-probe")
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged_manifest, manifest_path)
    print("probe_build: cluster_prepare_main_dte_ncc_execution")


def f16_payload(tile_id: int) -> list[float]:
    return [float(tile_id * 4 + lane + 1) for lane in range(INPUT_BYTES // 2)]


def async_sender_pattern(tile_id: int, count: int) -> bytes:
    return bytes(
        (
            tile_id * 53
            + index * 17
            + (index >> 8) * 29
            + 7
        )
        & 0xFF
        for index in range(count)
    )


def raw_multidest_pattern(tile_id: int, count: int) -> bytes:
    """Byte-exact records that make every aligned source slice identifiable."""

    if count % 4:
        raise RuntimeError("raw multidestination pattern requires 4-byte records")
    records = bytearray()
    for record in range(count // 4):
        records.extend(
            (
                tile_id & 0xFF,
                record & 0xFF,
                (record >> 8) & 0xFF,
                (tile_id * 37 + record * 29 + 0x5B) & 0xFF,
            )
        )
    return bytes(records)


def guarded_host_slot(payload: bytes) -> bytes:
    if len(payload) > ASYNC_TRANSPORT_BYTES:
        raise RuntimeError("DTE/NCC host slot payload exceeds 64 KiB")
    return (
        bytes([INITIAL_CANARY]) * SPM_GUARD_BYTES
        + payload
        + bytes([INITIAL_CANARY])
        * (ASYNC_TRANSPORT_BYTES - len(payload) + SPM_GUARD_BYTES)
    )


def write_probe_inputs(
    work_dir: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    mode: int,
    payload_bytes: int,
    sample: int = 0,
) -> tuple[list[str], pathlib.Path]:
    if payload_bytes not in PAYLOAD_SWEEP:
        raise RuntimeError(f"unsupported transport payload size {payload_bytes}")
    raw = (
        work_dir
        / "probe-raw"
        / f"mode-{mode}-bytes-{payload_bytes}-sample-{sample}"
    )
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    input_blob = bytearray(TILE_COUNT * RESOURCE_BYTES)
    values = [f16_payload(tile_id) for tile_id in range(TILE_COUNT)]
    for tile_id in range(TILE_COUNT):
        header = struct.pack("<II", mode, payload_bytes) + bytes(
            HEADER_BYTES - 8
        )
        normal_payload = struct.pack(
            f"<{len(values[tile_id])}e", *values[tile_id]
        )
        if mode in ASYNC_SENDER_MODES:
            slots = (
                guarded_host_slot(
                    async_sender_pattern(tile_id, ASYNC_TRANSPORT_BYTES)
                ),
                guarded_host_slot(
                    struct.pack(
                        f"<{ASYNC_TRANSPORT_BYTES // 2}e",
                        *([1.0] * (ASYNC_TRANSPORT_BYTES // 2)),
                    )
                ),
            )
        elif mode in RAW_MULTIDEST_MODES:
            slots = (
                guarded_host_slot(
                    raw_multidest_pattern(tile_id, MAX_PAYLOAD_BYTES)
                ),
                guarded_host_slot(b""),
            )
        else:
            slots = (
                guarded_host_slot(normal_payload),
                guarded_host_slot(b""),
            )
        input_bytes = header + b"".join(slots)
        tile_input = (
            input_bytes
            + bytes([INITIAL_CANARY]) * (RESOURCE_BYTES - len(input_bytes))
        )
        begin = tile_id * RESOURCE_BYTES
        input_blob[begin : begin + RESOURCE_BYTES] = tile_input
    input_ids = {
        bindings[(tile_id, "user_input", 0)] for tile_id in range(TILE_COUNT)
    }
    output_ids = {
        bindings[(tile_id, "output", 0)] for tile_id in range(TILE_COUNT)
    }
    if len(input_ids) != 1 or len(output_ids) != 1:
        raise RuntimeError("DTE/NCC probe boundary resources are not shared")
    input_path = raw / "input.raw"
    output_path = raw / "output.raw"
    input_path.write_bytes(input_blob)
    # Materialize the eager reference before any device launch, including
    # no-card qualification. Headers/status remain separate ABI evidence.
    (raw / "expected-captures.raw").write_bytes(b"".join(
        expected_capture_blob(mode, tile, payload_bytes)
        for tile in range(TILE_COUNT)
    ))
    return (
        [
            "--resource",
            f"{next(iter(input_ids))}={input_path}",
            "--output",
            f"{next(iter(output_ids))}={output_path}",
        ],
        output_path,
    )


def packed_f16_slice(
    tile_id: int,
    first_lane: int,
    payload_bytes: int,
    *,
    doubled: bool = False,
) -> bytes:
    values = torch.tensor(f16_payload(tile_id), dtype=torch.float16)[
        first_lane : first_lane + payload_bytes // 2
    ]
    if doubled:
        values = values + values
    return torch_common.tensor_raw_bytes(values)


def guarded_capture_slot(active: bytes, region_bytes: int) -> bytes:
    if len(active) > region_bytes or region_bytes > ASYNC_TRANSPORT_BYTES:
        raise RuntimeError("DTE/NCC capture exceeds its guarded SPM region")
    captured = (
        bytes([INITIAL_CANARY]) * SPM_GUARD_BYTES
        + active
        + bytes([INITIAL_CANARY]) * (region_bytes - len(active))
        + bytes([INITIAL_CANARY]) * SPM_GUARD_BYTES
    )
    return captured + bytes([INITIAL_CANARY]) * (
        HOST_SLOT_BYTES - len(captured)
    )


def expected_capture_slots(
    mode: int, tile_id: int, payload_bytes: int
) -> tuple[bytes, bytes, bytes, bytes]:
    if payload_bytes not in PAYLOAD_SWEEP:
        raise RuntimeError(f"unsupported transport payload size {payload_bytes}")
    predecessor = (tile_id - 1) % TILE_COUNT
    second_predecessor = (tile_id - 2) % TILE_COUNT
    second_lane = MAX_PAYLOAD_BYTES // 2
    empty = guarded_capture_slot(b"", MAX_PAYLOAD_BYTES)
    local_first = packed_f16_slice(tile_id, 0, payload_bytes)
    local_second = packed_f16_slice(tile_id, second_lane, payload_bytes)
    remote_first = packed_f16_slice(predecessor, 0, payload_bytes)
    remote_second = packed_f16_slice(
        predecessor, second_lane, payload_bytes
    )

    if mode == FOUR_SOURCE_FANIN_MODE:
        if tile_id == transport_catalog.FOUR_SOURCE_FANIN_TARGET:
            return tuple(
                guarded_capture_slot(
                    packed_f16_slice(source_tile, 0, payload_bytes),
                    MAX_PAYLOAD_BYTES,
                )
                for source_tile in transport_catalog.FOUR_SOURCE_FANIN_SOURCES
            )
        return (
            guarded_capture_slot(local_first, MAX_PAYLOAD_BYTES),
            empty,
            empty,
            empty,
        )
    if mode in RAW_MULTIDEST_MODES:
        case = RAW_MULTIDEST_CASE_BY_MODE[mode]
        receive = b""
        if tile_id in case.target_tiles:
            source = raw_multidest_pattern(
                transport_catalog.RAW_MULTIDEST_SOURCE_TILE,
                MAX_PAYLOAD_BYTES,
            )
            if case.semantic == "shuffle":
                receive = b"".join(
                    source[offset : offset + case.element_bytes]
                    for offset in raw_shuffle_source_offsets(mode)
                )
            else:
                destination_index = case.target_tiles.index(tile_id)
                source_offset = (
                    0
                    if case.semantic == "broadcast"
                    else raw_multidest_reference_offsets(mode)[
                        destination_index
                    ]
                )
                receive = source[
                    source_offset : source_offset
                    + case.per_destination_bytes
                ]
        return (
            guarded_capture_slot(
                raw_multidest_pattern(tile_id, MAX_PAYLOAD_BYTES),
                MAX_PAYLOAD_BYTES,
            ),
            guarded_capture_slot(receive, MAX_PAYLOAD_BYTES),
            empty,
            empty,
        )
    if mode == 1:
        local_doubled = packed_f16_slice(
            tile_id, 0, payload_bytes, doubled=True
        )
        remote_doubled = packed_f16_slice(
            predecessor, 0, payload_bytes, doubled=True
        )
        return (
            guarded_capture_slot(local_doubled, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_doubled, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(local_doubled, MAX_PAYLOAD_BYTES),
            empty,
        )
    if mode == 2:
        remote_doubled = packed_f16_slice(
            predecessor, 0, payload_bytes, doubled=True
        )
        return (
            guarded_capture_slot(local_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_doubled, MAX_PAYLOAD_BYTES),
            empty,
        )
    if mode in (3, 4):
        disjoint_input = packed_f16_slice(
            tile_id, second_lane, 32
        )
        disjoint_output = packed_f16_slice(
            tile_id, second_lane, 32, doubled=True
        )
        return (
            guarded_capture_slot(local_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(disjoint_output, 32),
            guarded_capture_slot(disjoint_input, 32),
        )
    if mode == 5:
        return (
            guarded_capture_slot(local_second, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_second, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_second, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_first, MAX_PAYLOAD_BYTES),
        )
    if mode == 6:
        return (
            guarded_capture_slot(local_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(
                packed_f16_slice(second_predecessor, 0, payload_bytes),
                MAX_PAYLOAD_BYTES,
            ),
        )
    if mode in (7, 12):
        return (
            guarded_capture_slot(local_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_first, MAX_PAYLOAD_BYTES),
            guarded_capture_slot(remote_first, MAX_PAYLOAD_BYTES),
            empty,
        )
    if mode in (8, 9):
        return (empty, empty, empty, empty)
    if mode in ASYNC_SENDER_MODES:
        ones = struct.pack(
            f"<{ASYNC_TRANSPORT_BYTES // 2}H",
            *([0x3C00] * (ASYNC_TRANSPORT_BYTES // 2)),
        )
        twos = struct.pack(
            f"<{ASYNC_TRANSPORT_BYTES // 2}H",
            *([0x4000] * (ASYNC_TRANSPORT_BYTES // 2)),
        )
        return (
            guarded_capture_slot(
                async_sender_pattern(tile_id, ASYNC_TRANSPORT_BYTES),
                ASYNC_TRANSPORT_BYTES,
            ),
            guarded_capture_slot(
                async_sender_pattern(
                    predecessor, ASYNC_TRANSPORT_BYTES
                ),
                ASYNC_TRANSPORT_BYTES,
            ),
            guarded_capture_slot(ones, ASYNC_TRANSPORT_BYTES),
            guarded_capture_slot(twos, ASYNC_TRANSPORT_BYTES),
        )
    raise RuntimeError(f"unsupported DTE/NCC mode {mode}")


def expected_capture_blob(
    mode: int, tile_id: int, payload_bytes: int
) -> bytes:
    slots = b"".join(expected_capture_slots(mode, tile_id, payload_bytes))
    return slots + bytes([INITIAL_CANARY]) * (
        RESOURCE_BYTES - HEADER_BYTES - len(slots)
    )


def raw_multidest_reference_offsets(mode: int) -> tuple[int, ...]:
    """Stimulus reference only; the board oracle does not assume this mapping."""

    case = RAW_MULTIDEST_CASE_BY_MODE[mode]
    if case.semantic == "shuffle":
        raise RuntimeError("shuffle uses source-section offsets")
    if case.semantic == "broadcast":
        return (0,)
    return tuple(
        index * case.per_destination_bytes
        for index in range(case.fanout)
    )


def raw_shuffle_source_offsets(mode: int) -> tuple[int, ...]:
    case = RAW_MULTIDEST_CASE_BY_MODE[mode]
    if case.semantic != "shuffle":
        raise RuntimeError("source-section offsets require shuffle mode")
    return tuple(
        index * 2 * case.element_bytes
        for index in range(case.shuffle_sections)
    )


def validate_raw_multidest_capture(
    payload: bytes, mode: int, tile_id: int
) -> dict[str, object]:
    case = RAW_MULTIDEST_CASE_BY_MODE[mode]
    capture = payload[HEADER_BYTES:]
    slots = tuple(
        capture[index * HOST_SLOT_BYTES : (index + 1) * HOST_SLOT_BYTES]
        for index in range(HOST_SLOT_COUNT)
    )
    trailing = capture[HOST_SLOT_COUNT * HOST_SLOT_BYTES :]
    expected_local = guarded_capture_slot(
        raw_multidest_pattern(tile_id, MAX_PAYLOAD_BYTES),
        MAX_PAYLOAD_BYTES,
    )
    empty = guarded_capture_slot(b"", MAX_PAYLOAD_BYTES)
    if slots[0] != expected_local:
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} raw source/local guarded slot is not exact"
        )
    if slots[2] != empty or slots[3] != empty or trailing != bytes(
        [INITIAL_CANARY]
    ) * len(trailing):
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} raw inactive guarded slots changed"
        )

    is_target = tile_id in case.target_tiles
    if not is_target:
        if slots[1] != empty:
            raise RuntimeError(
                f"tile_id {tile_id} mode {mode} received outside selective fanout"
            )
        return {
            "role": (
                "source"
                if tile_id == transport_catalog.RAW_MULTIDEST_SOURCE_TILE
                else "nonparticipant"
            ),
            "source_offset_bytes": None,
        }

    source = raw_multidest_pattern(
        transport_catalog.RAW_MULTIDEST_SOURCE_TILE, MAX_PAYLOAD_BYTES
    )
    if case.semantic == "shuffle":
        source_offsets = raw_shuffle_source_offsets(mode)
        expected_receive = b"".join(
            source[offset : offset + case.element_bytes]
            for offset in source_offsets
        )
        if slots[1] != guarded_capture_slot(
            expected_receive, MAX_PAYLOAD_BYTES
        ):
            raise RuntimeError(
                f"tile_id {tile_id} mode {mode} raw source shuffle is not exact"
            )
        return {
            "role": "target",
            "destination_slot": 0,
            "source_offset_bytes": source_offsets[0],
            "source_offsets_bytes": source_offsets,
            "received_bytes": len(expected_receive),
        }

    matches = tuple(
        (offset, received_bytes)
        for received_bytes in range(case.per_destination_bytes, 0, -4)
        for offset in range(
            0,
            MAX_PAYLOAD_BYTES - received_bytes + 1,
            4,
        )
        if slots[1]
        == guarded_capture_slot(
            source[offset : offset + received_bytes],
            MAX_PAYLOAD_BYTES,
        )
    )
    if slots[1] == empty:
        matches = ((0, 0),)
    if len(matches) != 1:
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} raw receive is not one exact guarded "
            f"source slice; matching offset/length pairs={matches}"
        )
    return {
        "role": "target",
        "destination_slot": case.target_tiles.index(tile_id),
        "source_offset_bytes": matches[0][0],
        "received_bytes": matches[0][1],
    }


def summarize_raw_multidest_observations(
    mode: int, observations: list[dict[str, object]]
) -> dict[str, object]:
    case = RAW_MULTIDEST_CASE_BY_MODE[mode]
    target_rows = {
        int(row["tile_id"]): row["raw_multidest"]
        for row in observations
        if row["raw_multidest"]["role"] == "target"
    }
    if set(target_rows) != set(case.target_tiles):
        raise RuntimeError(
            f"{case.name} did not observe exactly its configured target tile_ids"
        )
    if case.semantic == "shuffle":
        target_tile = case.target_tiles[0]
        target = target_rows[target_tile]
        source_offsets = tuple(
            int(offset) for offset in target["source_offsets_bytes"]
        )
        received_bytes = int(target["received_bytes"])
        return {
            "semantic": case.semantic,
            "raw_mode": case.raw_mode,
            "fanout": 1,
            "target_layout": case.target_layout,
            "target_tile": target_tile,
            "shuffle_sections": case.shuffle_sections,
            "element_bytes": case.element_bytes,
            "source_stride_bytes": 2 * case.element_bytes,
            "source_offsets_bytes": source_offsets,
            "received_bytes": received_bytes,
            "dest_num_register_value": case.dest_num_register_value,
            "dest_num_encoding": case.dest_num_encoding,
            "reference_capability_match": (
                source_offsets == raw_shuffle_source_offsets(mode)
                and received_bytes == case.per_destination_bytes
            ),
            "mapping_policy": (
                "single-destination source-side 1D shuffle; no "
                "multi-destination or collective claim"
            ),
            "all_selected_targets_exact": True,
            "all_nonparticipants_untouched": True,
        }

    mapping = {
        tile_id: (
            int(target_rows[tile_id]["source_offset_bytes"]),
            int(target_rows[tile_id]["received_bytes"]),
        )
        for tile_id in case.target_tiles
    }
    offsets = tuple(mapping[tile_id][0] for tile_id in case.target_tiles)
    received_lengths = tuple(
        mapping[tile_id][1] for tile_id in case.target_tiles
    )
    reference_offsets = raw_multidest_reference_offsets(mode)
    reference_mapping = (
        (0,) * case.fanout
        if case.semantic == "broadcast"
        else reference_offsets
    )
    unique_offsets = len(set(offsets))
    if unique_offsets == 1:
        observed_mapping_class = "all-targets-same-source-slice"
    elif unique_offsets == case.fanout:
        observed_mapping_class = "all-targets-distinct-source-slices"
    else:
        observed_mapping_class = "partially-duplicated-source-slices"
    return {
        "semantic": case.semantic,
        "raw_mode": case.raw_mode,
        "fanout": case.fanout,
        "target_layout": case.target_layout,
        "dest_num_register_value": case.dest_num_register_value,
        "dest_num_encoding": case.dest_num_encoding,
        "target_to_source_offset_bytes": {
            str(tile_id): {
                "source_offset_bytes": mapping[tile_id][0],
                "received_bytes": mapping[tile_id][1],
            }
            for tile_id in case.target_tiles
        },
        "observed_mapping_class": observed_mapping_class,
        "distinct_source_slice_count": unique_offsets,
        "observed_received_bytes": received_lengths,
        "observed_delivered_bytes": sum(received_lengths),
        "all_targets_received_configured_length": (
            received_lengths
            == (case.per_destination_bytes,) * case.fanout
        ),
        "stimulus_reference_mapping_match": offsets == reference_mapping,
        "reference_capability_match": (
            received_lengths
            == (case.per_destination_bytes,) * case.fanout
            and offsets == reference_mapping
        ),
        "mapping_policy": (
            "classify observed owner-backed behavior; do not assume the "
            "scatter/shuffle source-slice order or dest_num ABI"
        ),
        "all_selected_targets_exact": True,
        "all_nonparticipants_untouched": True,
        "all_guards_and_inactive_suffixes_exact": True,
    }


def parse_probe_payload(
    payload: bytes, mode: int, tile_id: int, payload_bytes: int
) -> dict[str, object]:
    if payload_bytes not in PAYLOAD_SWEEP:
        raise RuntimeError(f"unsupported transport payload size {payload_bytes}")
    if len(payload) != RESOURCE_BYTES:
        raise RuntimeError(f"tile_id {tile_id} mode {mode} output has invalid size")
    words = struct.unpack("<16Q", payload[:HEADER_BYTES])
    stable_mask = (words[1] >> 48) & 0xFF
    reserved = (words[1] >> 32) & 0xFFFF
    recorded_mode = (words[1] >> 24) & 0xFF
    recorded_tile = (words[1] >> 16) & 0xFF
    recorded_bytes = words[1] & 0xFFFF
    contract_status = (words[1] >> 56) & 0xFF
    status = words[3] & 0xFF
    instruction_counts = (
        (words[3] >> 8) & 0xFF,
        (words[3] >> 16) & 0xFF,
        (words[3] >> 24) & 0xFF,
    )
    device_oracle_mismatches = (
        (words[3] >> 32) & 0xFF,
        (words[3] >> 40) & 0xFF,
        (words[3] >> 48) & 0xFF,
        (words[3] >> 56) & 0xFF,
    )
    if (
        words[0] != MAGIC
        or words[2] != CANARY
        or reserved != 0
        or recorded_mode != mode
        or recorded_tile != tile_id
        or recorded_bytes != payload_bytes
        or status != STATUS_SUCCESS
    ):
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} has invalid status/canary header"
        )
    if stable_mask != 0xF:
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} has unstable NCC PMU counters: "
            f"mask=0x{stable_mask:x}"
        )
    transport_stable_mask = words[8] & 0xFFFF
    recorded_transport_mask = (words[8] >> 16) & 0xFFFF
    scope_change = (words[8] >> 32) & 0xFF
    read_only = bool((words[8] >> 40) & 1)
    contract_evidence = (words[8] >> 48) & 0xFF
    if (
        recorded_transport_mask != TRANSPORT_STABLE_MASK
        or transport_stable_mask != TRANSPORT_STABLE_MASK
        or not read_only
        or scope_change & ~0x3
    ):
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} has invalid transport PMU sampling "
            f"metadata: stable=0x{transport_stable_mask:x} "
            f"expected=0x{recorded_transport_mask:x} read_only={read_only}"
        )
    expected_contract_status = (
        STATUS_TRANSPORT_ERROR if mode in ERROR_PATH_MODES else STATUS_SUCCESS
    )
    expected_contract_evidence_value = expected_contract_evidence(mode, tile_id)
    if (
        contract_status != expected_contract_status
        or contract_evidence != expected_contract_evidence_value
    ):
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} has invalid Direct-DTE error-path "
            f"evidence: status={contract_status} "
            f"evidence=0x{contract_evidence:02x} expected_status="
            f"{expected_contract_status} expected_evidence="
            f"0x{expected_contract_evidence_value:02x}"
        )
    expected_counts = EXPECTED_INSTRUCTION_COUNTS.get(mode)
    if expected_counts is None or instruction_counts != expected_counts:
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} has NCC instruction deltas "
            f"{instruction_counts}, expected {expected_counts}"
        )
    if any(device_oracle_mismatches):
        raise RuntimeError(
            f"tile_id {tile_id} mode {mode} device guard/result mismatches "
            f"{device_oracle_mismatches}"
        )
    raw_async_return_codes: dict[str, int] | None = None
    raw_multidest_return_codes: dict[str, int] | None = None
    if mode in (*ASYNC_SENDER_MODES, *RAW_MULTIDEST_MODES):
        def signed32(value: int) -> int:
            return value if value < 1 << 31 else value - (1 << 32)

        return_codes = {
            "send_async": signed32(words[6] & 0xFFFFFFFF),
            "wait_done": signed32((words[6] >> 32) & 0xFFFFFFFF),
            "release": signed32(words[7] & 0xFFFFFFFF),
        }
        marker = words[7] >> 32
        expected_marker = (
            RAW_MULTIDEST_RC_MARKER
            if mode in RAW_MULTIDEST_MODES
            else RAW_ASYNC_RC_MARKER
        )
        if marker != expected_marker or any(return_codes.values()):
            raise RuntimeError(
                f"tile_id {tile_id} mode {mode} has invalid raw DTE return "
                f"codes/marker: {return_codes} marker=0x{marker:x}"
            )
        if mode in RAW_MULTIDEST_MODES:
            raw_multidest_return_codes = return_codes
        else:
            raw_async_return_codes = return_codes
    raw_multidest: dict[str, object] | None = None
    if mode in RAW_MULTIDEST_MODES:
        raw_multidest = validate_raw_multidest_capture(payload, mode, tile_id)
    else:
        expected_readback = expected_capture_blob(mode, tile_id, payload_bytes)
        if mode == 1:
            # This launch witness uses exactly representable FP16 doubling.
            # Compare every numeric slot with eager PyTorch as well as the
            # existing byte-exact transport/guard evidence.
            for slot in range(3):
                begin = slot * HOST_SLOT_BYTES + SPM_GUARD_BYTES
                actual = torch.frombuffer(bytearray(
                    payload[HEADER_BYTES + begin:
                            HEADER_BYTES + begin + payload_bytes]
                ), dtype=torch.float16).clone()
                expected = torch.frombuffer(bytearray(
                    expected_readback[begin:begin + payload_bytes]
                ), dtype=torch.float16).clone()
                torch_common.assert_tensor_matches(
                    actual, expected,
                    policy=torch_common.ComparisonPolicy(rtol=1e-3, atol=1e-5),
                    context=f"DTE producer/send Tile {tile_id} slot {slot}",
                )
        if payload[HEADER_BYTES:] != expected_readback:
            raise RuntimeError(
                f"tile_id {tile_id} mode {mode} guarded SPM readback is not exact"
            )

    dte_enable = words[9] & 0xFFFFFFFF
    spm_enable = words[9] >> 32
    transport_deltas = tuple(words[10:16])
    inconclusive_reasons: list[str] = []
    if dte_enable & DTE_ENABLE_MASK != DTE_ENABLE_MASK:
        inconclusive_reasons.append("dte_pmu_not_enabled")
    if spm_enable & SPM_ENABLE_MASK != SPM_ENABLE_MASK:
        inconclusive_reasons.append("spm_pmu_not_enabled")
    if scope_change & 1:
        inconclusive_reasons.append("dte_pmu_enable_changed")
    if scope_change & 2:
        inconclusive_reasons.append("spm_pmu_enable_changed")
    if not any(transport_deltas):
        inconclusive_reasons.append("all_transport_deltas_zero")
    transport_sample_state = (
        "inconclusive" if inconclusive_reasons else "raw_observation"
    )
    return {
        "tile_id": tile_id,
        "payload_bytes": payload_bytes,
        "transport_bytes": transport_bytes_for_mode(mode, payload_bytes),
        "stable_mask": stable_mask,
        "status": status,
        "contract_status": contract_status,
        "contract_evidence": f"0x{contract_evidence:02x}",
        "raw_async_return_codes": raw_async_return_codes,
        "raw_multidest_return_codes": raw_multidest_return_codes,
        "raw_multidest": raw_multidest,
        "ct_count_delta": instruction_counts[0],
        "rdma_count_delta": instruction_counts[1],
        "wdma_count_delta": instruction_counts[2],
        "host_readback_oracle": {
            "guarded_spm_slots": HOST_SLOT_COUNT,
            "slot_bytes": HOST_SLOT_BYTES,
            "exact": True,
        },
        "full_cycles_delta": words[4],
        "ct_cycles_delta": words[5],
        "rdma_cycles_delta": (
            None
            if mode in (*ASYNC_SENDER_MODES, *RAW_MULTIDEST_MODES)
            else words[6]
        ),
        "wdma_cycles_delta": (
            None
            if mode in (*ASYNC_SENDER_MODES, *RAW_MULTIDEST_MODES)
            else words[7]
        ),
        "transport_pmu": {
            "measurement_basis": "not_calibrated",
            "sample_state": transport_sample_state,
            "inconclusive_reasons": inconclusive_reasons,
            "delta_arithmetic": "raw_modulo_2^64",
            "stable_counter_mask": f"0x{transport_stable_mask:02x}",
            "raw_counter_deltas": dict(
                zip(TRANSPORT_COUNTER_NAMES, transport_deltas, strict=True)
            ),
            "scope": {
                "tile_local": True,
                "read_only": read_only,
                "dte_channels": [0, 1],
                "dte_enable_raw_before": dte_enable,
                "spm_enable_raw_before": spm_enable,
                "dte_enable_changed_during_window": bool(scope_change & 1),
                "spm_enable_changed_during_window": bool(scope_change & 2),
            },
            "dte": {
                "channel0_transfer_raw_delta": words[10],
                "channel1_transfer_raw_delta": words[11],
                "channel0_execution_raw_delta": words[12],
                "channel1_execution_raw_delta": words[13],
            },
            "spm": {
                "t2_port8_raw_delta": words[14],
                "t3_port8_raw_delta": words[15],
            },
            "tmnoc": {
                "sampled": False,
                "scope": "unknown",
                "reason": (
                    "current installed headers expose only TMNOC PMU bases; "
                    "no decoded read-only counter offsets or measurement basis"
                ),
            },
        },
    }


def synthetic_probe_payload(
    mode: int,
    tile_id: int,
    payload_bytes: int,
    *,
    dte_enable: int = DTE_ENABLE_MASK,
    spm_enable: int = SPM_ENABLE_MASK,
    scope_change: int = 0,
    transport_stable_mask: int = TRANSPORT_STABLE_MASK,
    read_only: bool = True,
    transport_deltas: tuple[int, ...] = (64, 0, 17, 0, 64, 64),
    device_oracle_mismatches: tuple[int, int, int, int] = (0, 0, 0, 0),
    raw_async_return_codes: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    ct_count, rdma_count, wdma_count = EXPECTED_INSTRUCTION_COUNTS[mode]
    source_guard, receive_guard, compute_guard, compute_result = (
        device_oracle_mismatches
    )
    words = [0] * 16
    words[0] = MAGIC
    words[1] = (
        (0xF << 48)
        | (
            (
                STATUS_TRANSPORT_ERROR
                if mode in ERROR_PATH_MODES
                else STATUS_SUCCESS
            )
            << 56
        )
        | (mode << 24)
        | (tile_id << 16)
        | payload_bytes
    )
    words[2] = CANARY
    words[3] = (
        STATUS_SUCCESS
        | (ct_count << 8)
        | (rdma_count << 16)
        | (wdma_count << 24)
        | (source_guard << 32)
        | (receive_guard << 40)
        | (compute_guard << 48)
        | (compute_result << 56)
    )
    words[4:6] = [101, 31]
    if mode in (*ASYNC_SENDER_MODES, *RAW_MULTIDEST_MODES):
        send_result, wait_result, release_result = raw_async_return_codes
        words[6] = (
            (send_result & 0xFFFFFFFF)
            | ((wait_result & 0xFFFFFFFF) << 32)
        )
        words[7] = (
            (release_result & 0xFFFFFFFF)
            | (
                (
                    RAW_MULTIDEST_RC_MARKER
                    if mode in RAW_MULTIDEST_MODES
                    else RAW_ASYNC_RC_MARKER
                )
                << 32
            )
        )
    else:
        words[6:8] = [37, 41]
    words[8] = (
        transport_stable_mask
        | (TRANSPORT_STABLE_MASK << 16)
        | (scope_change << 32)
        | ((1 if read_only else 0) << 40)
        | (expected_contract_evidence(mode, tile_id) << 48)
    )
    words[9] = dte_enable | (spm_enable << 32)
    words[10:16] = list(transport_deltas)
    return (
        struct.pack("<16Q", *words)
        + expected_capture_blob(mode, tile_id, payload_bytes)
    )


def run_host_oracle_self_tests() -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"DTE/NCC host oracle self-test: {message}")

    def require_rejected(
        payload: bytes,
        payload_bytes: int,
        message: str,
        mode: int = 1,
    ) -> None:
        try:
            parse_probe_payload(payload, mode, 0, payload_bytes)
        except (RuntimeError, AssertionError):
            return
        raise RuntimeError(
            f"DTE/NCC host oracle self-test accepted {message}"
        )

    for mode in (*DEFAULT_SAFE_MODES, *PENDING_DTE_MODES):
        for payload_bytes in MODE_PAYLOADS[mode]:
            tile_ids = (
                range(TILE_COUNT)
                if mode in PENDING_DTE_MODES
                else (0, TILE_COUNT - 1)
            )
            parsed_rows: list[dict[str, object]] = []
            for tile_id in tile_ids:
                parsed = parse_probe_payload(
                    synthetic_probe_payload(mode, tile_id, payload_bytes),
                    mode,
                    tile_id,
                    payload_bytes,
                )
                parsed_rows.append(parsed)
                require(
                    parsed["transport_pmu"]["sample_state"]
                    == "raw_observation",
                    f"valid mode {mode} tile_id {tile_id} bytes "
                    f"{payload_bytes} was not a raw observation",
                )
                if mode in ASYNC_SENDER_MODES:
                    require(
                        parsed["raw_async_return_codes"]
                        == {
                            "send_async": 0,
                            "wait_done": 0,
                            "release": 0,
                        },
                        f"valid mode {mode} did not expose all-zero raw "
                        "async return codes",
                    )
                if mode in RAW_MULTIDEST_MODES:
                    require(
                        parsed["raw_multidest_return_codes"]
                        == {
                            "send_async": 0,
                            "wait_done": 0,
                            "release": 0,
                        },
                        f"valid mode {mode} did not expose all-zero "
                        "owner-backed raw DTE return codes",
                    )
            if mode in RAW_MULTIDEST_MODES:
                summary = summarize_raw_multidest_observations(
                    mode, parsed_rows
                )
                require(
                    summary["all_nonparticipants_untouched"] is True,
                    f"valid mode {mode} did not preserve nonparticipants",
                )

    inconclusive_cases = (
        (
            {"dte_enable": 0},
            ("dte_pmu_not_enabled",),
        ),
        (
            {"spm_enable": 0},
            ("spm_pmu_not_enabled",),
        ),
        (
            {"scope_change": 3},
            ("dte_pmu_enable_changed", "spm_pmu_enable_changed"),
        ),
        (
            {"transport_deltas": (0, 0, 0, 0, 0, 0)},
            ("all_transport_deltas_zero",),
        ),
    )
    for options, expected_reasons in inconclusive_cases:
        parsed = parse_probe_payload(
            synthetic_probe_payload(1, 0, MAX_PAYLOAD_BYTES, **options),
            1,
            0,
            MAX_PAYLOAD_BYTES,
        )
        reasons = parsed["transport_pmu"]["inconclusive_reasons"]
        require(
            parsed["transport_pmu"]["sample_state"] == "inconclusive"
            and all(reason in reasons for reason in expected_reasons),
            f"{expected_reasons} did not fail closed as inconclusive",
        )

    require_rejected(
        synthetic_probe_payload(
            1,
            0,
            MAX_PAYLOAD_BYTES,
            transport_stable_mask=TRANSPORT_STABLE_MASK ^ 1,
        ),
        MAX_PAYLOAD_BYTES,
        "an unstable transport split counter",
    )
    require_rejected(
        synthetic_probe_payload(
            1, 0, MAX_PAYLOAD_BYTES, read_only=False
        ),
        MAX_PAYLOAD_BYTES,
        "a non-read-only transport sample",
    )
    for index, name in enumerate(
        ("source guard", "receive guard", "compute guard", "compute result")
    ):
        mismatches = [0, 0, 0, 0]
        mismatches[index] = 1
        require_rejected(
            synthetic_probe_payload(
                1,
                0,
                MAX_PAYLOAD_BYTES,
                device_oracle_mismatches=tuple(mismatches),
            ),
            MAX_PAYLOAD_BYTES,
            name,
        )
    require_rejected(
        synthetic_probe_payload(
            ASYNC_SENDER_MODES[1],
            0,
            ASYNC_SENDER_PAYLOAD_BYTES,
            raw_async_return_codes=(0, -1, 0),
        ),
        ASYNC_SENDER_PAYLOAD_BYTES,
        "a failed raw async wait return code",
        mode=ASYNC_SENDER_MODES[1],
    )

    corrupted_guard = bytearray(
        synthetic_probe_payload(1, 0, MAX_PAYLOAD_BYTES)
    )
    corrupted_guard[HEADER_BYTES] ^= 1
    require_rejected(
        bytes(corrupted_guard),
        MAX_PAYLOAD_BYTES,
        "a changed output guard before",
    )
    corrupted_guard = bytearray(
        synthetic_probe_payload(1, 0, MAX_PAYLOAD_BYTES)
    )
    corrupted_guard[-1] ^= 1
    require_rejected(
        bytes(corrupted_guard),
        MAX_PAYLOAD_BYTES,
        "a changed output guard after",
    )
    corrupted_payload = bytearray(
        synthetic_probe_payload(1, 0, MAX_PAYLOAD_BYTES)
    )
    corrupted_payload[HEADER_BYTES + SPM_GUARD_BYTES] ^= 1
    require_rejected(
        bytes(corrupted_payload),
        MAX_PAYLOAD_BYTES,
        "a changed logical payload",
    )
    corrupted_suffix = bytearray(synthetic_probe_payload(1, 0, 16))
    corrupted_suffix[
        HEADER_BYTES + SPM_GUARD_BYTES + 16
    ] ^= 1
    require_rejected(
        bytes(corrupted_suffix),
        16,
        "a changed inactive payload suffix",
    )
    raw_mode = RAW_MULTIDEST_MODES[0]
    raw_nonparticipant = next(
        tile_id
        for tile_id in range(TILE_COUNT)
        if tile_id
        not in (
            transport_catalog.RAW_MULTIDEST_SOURCE_TILE,
            *RAW_MULTIDEST_CASE_BY_MODE[raw_mode].target_tiles,
        )
    )
    corrupted_raw_nonparticipant = bytearray(
        synthetic_probe_payload(
            raw_mode,
            raw_nonparticipant,
            transport_catalog.RAW_MULTIDEST_PAYLOAD_BYTES,
        )
    )
    corrupted_raw_nonparticipant[
        HEADER_BYTES + HOST_SLOT_BYTES + SPM_GUARD_BYTES
    ] ^= 1
    try:
        parse_probe_payload(
            bytes(corrupted_raw_nonparticipant),
            raw_mode,
            raw_nonparticipant,
            transport_catalog.RAW_MULTIDEST_PAYLOAD_BYTES,
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "DTE/NCC host oracle self-test accepted a write to an "
            "unselected raw multicast Tile"
        )

    divergent_mode = next(
        mode
        for mode in RAW_MULTIDEST_MODES
        if RAW_MULTIDEST_CASE_BY_MODE[mode].semantic == "scatter"
        and RAW_MULTIDEST_CASE_BY_MODE[mode].fanout == 2
    )
    divergent_case = RAW_MULTIDEST_CASE_BY_MODE[divergent_mode]
    divergent_target = divergent_case.target_tiles[0]
    divergent_rows: list[dict[str, object]] = []
    for tile_id in range(TILE_COUNT):
        row_payload = bytearray(
            synthetic_probe_payload(
                divergent_mode,
                tile_id,
                transport_catalog.RAW_MULTIDEST_PAYLOAD_BYTES,
            )
        )
        if tile_id == divergent_target:
            alternate_offset = 512
            source = raw_multidest_pattern(
                transport_catalog.RAW_MULTIDEST_SOURCE_TILE,
                MAX_PAYLOAD_BYTES,
            )
            alternate_slot = guarded_capture_slot(
                source[
                    alternate_offset : alternate_offset
                    + divergent_case.per_destination_bytes
                ],
                MAX_PAYLOAD_BYTES,
            )
            slot_start = HEADER_BYTES + HOST_SLOT_BYTES
            row_payload[
                slot_start : slot_start + HOST_SLOT_BYTES
            ] = alternate_slot
        divergent_rows.append(
            parse_probe_payload(
                bytes(row_payload),
                divergent_mode,
                tile_id,
                transport_catalog.RAW_MULTIDEST_PAYLOAD_BYTES,
            )
        )
    divergent_summary = summarize_raw_multidest_observations(
        divergent_mode, divergent_rows
    )
    require(
        divergent_summary["stimulus_reference_mapping_match"] is False
        and divergent_summary["all_selected_targets_exact"] is True,
        "raw observation classifier did not preserve a valid divergent "
        "target/source mapping",
    )


def run_pending_host_contract(
    mode: int, payload_bytes: int
) -> dict[str, object]:
    if mode not in PENDING_DTE_MODES:
        raise RuntimeError(
            "--host-contract-only requires one pending fan-in/raw mode"
        )
    if payload_bytes not in MODE_PAYLOADS[mode]:
        raise RuntimeError(
            f"mode {mode} does not support payload size {payload_bytes}"
        )
    observations = [
        parse_probe_payload(
            synthetic_probe_payload(mode, tile_id, payload_bytes),
            mode,
            tile_id,
            payload_bytes,
        )
        for tile_id in range(TILE_COUNT)
    ]
    return {
        "case": MODES[mode],
        "mode": mode,
        "payload_bytes": payload_bytes,
        "single_mode_isolation": True,
        "all_tile_exact_oracle": True,
        "raw_multidest": (
            summarize_raw_multidest_observations(mode, observations)
            if mode in RAW_MULTIDEST_MODES
            else None
        ),
        "four_source_fanin": (
            {
                "target_tile": transport_catalog.FOUR_SOURCE_FANIN_TARGET,
                "source_tiles": transport_catalog.FOUR_SOURCE_FANIN_SOURCES,
                "four_disjoint_guarded_slots_exact": True,
            }
            if mode == FOUR_SOURCE_FANIN_MODE
            else None
        ),
    }


def execute_probe_modes(
    args: argparse.Namespace,
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    selected_modes: tuple[int, ...],
    selected_payload_bytes: tuple[int, ...],
) -> None:
    sweep_observations: dict[
        int, dict[int, list[dict[str, object]]]
    ] = {mode: {} for mode in selected_modes}
    for payload_bytes in selected_payload_bytes:
        for mode in selected_modes:
            if payload_bytes not in MODE_PAYLOADS[mode]:
                continue
            name = MODES[mode]
            repetitions = (
                ASYNC_SENDER_REPETITIONS
                if mode in ASYNC_SENDER_MODES
                else 1
            )
            for sample in range(repetitions):
                resource_args, output = write_probe_inputs(
                    args.work_dir, bindings, mode, payload_bytes, sample
                )
                result = run(
                    [*board_base_command(args, package), *resource_args],
                    timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0,
                )
                (output.parent / "runtime.log").write_text(
                    result.stdout + result.stderr, encoding="utf-8"
                )
                if result.stdout.count("output_capture:") != 1:
                    raise RuntimeError(
                        f"{name} bytes {payload_bytes} sample {sample} omitted "
                        "the shared Tile output capture"
                    )
                output_blob = output.read_bytes()
                if len(output_blob) != TILE_COUNT * RESOURCE_BYTES:
                    raise RuntimeError("DTE/NCC shared output size is invalid")
                observations = [
                    parse_probe_payload(
                        output_blob[
                            tile_id * RESOURCE_BYTES : (tile_id + 1) * RESOURCE_BYTES
                        ],
                        mode,
                        tile_id,
                        payload_bytes,
                    )
                    for tile_id in range(TILE_COUNT)
                ]
                for observation in observations:
                    observation["sample"] = sample
                if mode == 1:
                    print(
                        "pytorch_compare: case=dte-producer-send "
                        f"tiles={TILE_COUNT} elements={3 * TILE_COUNT * payload_bytes // 2} "
                        f"torch={torch.__version__} rtol=0.001 atol=1e-05 passed=true"
                    )
                raw_multidest_summary = (
                    summarize_raw_multidest_observations(mode, observations)
                    if mode in RAW_MULTIDEST_MODES
                    else None
                )
                sweep_observations[mode].setdefault(
                    payload_bytes, []
                ).extend(observations)
                inconclusive_tiles = [
                    observation["tile_id"]
                    for observation in observations
                    if observation["transport_pmu"]["sample_state"]
                    == "inconclusive"
                ]
                print(
                    "dte_ncc_case: "
                    + json.dumps(
                        {
                            "case": name,
                            "sample": sample,
                            "payload_bytes": payload_bytes,
                            "transport_bytes": transport_bytes_for_mode(
                                mode, payload_bytes
                            ),
                            "tile_ids": TILE_COUNT,
                            "exact": True,
                            "canary": True,
                            "guards": {
                                "guarded_spm_readback": "exact",
                                "host_slot_count": HOST_SLOT_COUNT,
                            },
                            "transport_status": "success",
                            "contract_status": (
                                "transport_error_observed_then_clean_success"
                                if mode in ERROR_PATH_MODES
                                else "success"
                            ),
                            "ncc_pmu": observations,
                            "transport_pmu_sample_state": (
                                "inconclusive"
                                if inconclusive_tiles
                                else "raw_observation"
                            ),
                            "transport_pmu_inconclusive_tiles": (
                                inconclusive_tiles
                            ),
                            "dte_common_timer": None,
                            "dte_ncc_overlap": (
                                "not_claimed_by_this_correctness_probe"
                            ),
                            "raw_multidest": raw_multidest_summary,
                            "four_source_fanin": (
                                {
                                    "target_tile": (
                                        transport_catalog.FOUR_SOURCE_FANIN_TARGET
                                    ),
                                    "source_tiles": (
                                        transport_catalog.FOUR_SOURCE_FANIN_SOURCES
                                    ),
                                    "four_disjoint_guarded_slots_exact": True,
                                }
                                if mode == FOUR_SOURCE_FANIN_MODE
                                else None
                            ),
                        },
                        sort_keys=True,
                    )
                )
                print(result.stdout, end="")

    sweep_summary: dict[str, object] = {}
    for mode in selected_modes:
        if mode not in TRANSPORT_PMU_MODES:
            continue
        name = MODES[mode]
        counter_summaries: dict[str, object] = {}
        for counter_name in TRANSPORT_COUNTER_NAMES:
            samples = {
                payload_bytes: tuple(
                    int(
                        observation["transport_pmu"][
                            "raw_counter_deltas"
                        ][counter_name]
                    )
                    for observation in sweep_observations[mode][payload_bytes]
                )
                for payload_bytes in selected_payload_bytes
                if payload_bytes in sweep_observations[mode]
            }
            counter_summaries[counter_name] = (
                transport_catalog.classify_payload_series(samples)
            )
        sweep_summary[name] = counter_summaries
    print(
        "transport_pmu_payload_sweep: "
        + json.dumps(
            {
                "payload_bytes": selected_payload_bytes,
                "modes": sweep_summary,
                "tmnoc": transport_catalog.COUNTERS_BY_NAME[
                    "tmnoc"
                ].disposition,
                "unit_policy": "report observation; never infer without data",
            },
            sort_keys=True,
        )
    )

    if all(
        mode in sweep_observations
        and ASYNC_SENDER_PAYLOAD_BYTES in sweep_observations[mode]
        for mode in ASYNC_SENDER_MODES
    ):
        metric_names = (
            "full_cycles_delta",
            "ct_cycles_delta",
            *TRANSPORT_COUNTER_NAMES,
        )

        def metric(observation: dict[str, object], name: str) -> int:
            if name in {"full_cycles_delta", "ct_cycles_delta"}:
                return int(observation[name])
            return int(
                observation["transport_pmu"]["raw_counter_deltas"][name]
            )

        per_mode_tile_medians: dict[
            int, dict[int, dict[str, int | float]]
        ] = {}
        for mode in ASYNC_SENDER_MODES:
            rows = sweep_observations[mode][ASYNC_SENDER_PAYLOAD_BYTES]
            per_mode_tile_medians[mode] = {}
            for tile_id in range(TILE_COUNT):
                tile_rows = [
                    row for row in rows if int(row["tile_id"]) == tile_id
                ]
                if len(tile_rows) != ASYNC_SENDER_REPETITIONS:
                    raise RuntimeError(
                        f"{MODES[mode]} Tile {tile_id} has "
                        f"{len(tile_rows)} repeated samples, expected "
                        f"{ASYNC_SENDER_REPETITIONS}"
                    )
                per_mode_tile_medians[mode][tile_id] = {
                    name: statistics.median(
                        metric(row, name) for row in tile_rows
                    )
                    for name in metric_names
                }
        serial_mode, window_mode = ASYNC_SENDER_MODES
        print(
            "dte_sender_async_control_summary: "
            + json.dumps(
                {
                    "transport_bytes": ASYNC_SENDER_TRANSPORT_BYTES,
                    "samples_per_tile": ASYNC_SENDER_REPETITIONS,
                    "tile_ids": TILE_COUNT,
                    "correctness": "all-exact",
                    "receiver_first": True,
                    "return_codes": "all-zero",
                    "safety_completion": (
                        "wait_done+release+receive-event+local-fence+"
                        "runtime-terminal"
                    ),
                    "raw_tile_median_of_medians": {
                        label: {
                            name: statistics.median(
                                per_mode_tile_medians[mode][tile_id][name]
                                for tile_id in range(TILE_COUNT)
                            )
                            for name in metric_names
                        }
                        for label, mode in (
                            ("serial", serial_mode),
                            ("async_window", window_mode),
                        )
                    },
                    "async_window_minus_serial_paired_tile_median": {
                        name: statistics.median(
                            per_mode_tile_medians[window_mode][tile_id][name]
                            - per_mode_tile_medians[serial_mode][tile_id][name]
                            for tile_id in range(TILE_COUNT)
                        )
                        for name in metric_names
                    },
                    "issue_window_observed": True,
                    "temporal_overlap": (
                        "not_claimed_without_a_calibrated_common_timer"
                    ),
                    "interpretation": (
                        "raw repeated same-Tile control; DTE counters and NCC "
                        "FU counters have no calibrated common timer"
                    ),
                },
                sort_keys=True,
            )
        )


def execute_receiver_unprepared(
    args: argparse.Namespace,
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
) -> None:
    mode = ISOLATED_DTE_MODES[0]
    payload_bytes = MODE_PAYLOADS[mode][0]
    resource_args, _ = write_probe_inputs(
        args.work_dir, bindings, mode, payload_bytes, 0
    )
    command = [*board_base_command(args, package), *resource_args]
    timeout_seconds = args.completion_timeout_ms / 1000.0 + 5.0
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        print(
            "dte_receiver_unprepared: outer-timeout; no retry/reset/power",
            flush=True,
        )
        for partial in (error.stdout, error.stderr):
            if partial:
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(partial, end="", file=sys.stderr)
        return
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode == 0:
        raise RuntimeError(
            "receiver-unprepared unexpectedly completed successfully"
        )
    print(
        "dte_receiver_unprepared: provider rejected/timed out; "
        f"exit={result.returncode}; no retry/reset/power",
        flush=True,
    )


def validate_board_args(args: argparse.Namespace) -> None:
    required = (
        args.expected_runtime_version,
        args.expected_device_name,
        args.expected_pci_bus_id,
        args.expected_tile_count,
        args.expected_runtime_library_sha256,
    )
    if any(value in (None, "") for value in required):
        raise RuntimeError("board execution requires complete qualification values")
    if args.expected_tile_count != TILE_COUNT:
        raise RuntimeError("DTE/NCC probe requires the exact 16-tile domain")
    if (
        re.fullmatch(
            r"[0-9a-fA-F]{64}", str(args.expected_runtime_library_sha256)
        )
        is None
    ):
        raise RuntimeError("runtime library digest must be 64 hexadecimal digits")
    if args.completion_timeout_ms <= 0:
        raise RuntimeError("--completion-timeout-ms must be positive")


def main() -> int:
    args = parse_args()
    if args.list_cases:
        print(
            json.dumps(
                {
                    "transport_pmu_cases": [
                        case.as_dict() for case in transport_catalog.CASES
                    ],
                    "contract_cases": [
                        case.as_dict()
                        for case in transport_catalog.CONTRACT_CASES
                    ],
                    "raw_remote_multicast_cases": [
                        case.as_dict()
                        for case in transport_catalog.RAW_MULTIDEST_CASES
                    ],
                    "calibration_leaf_bindings": {
                        key: [
                            getattr(
                                case,
                                "name",
                                MODES.get(getattr(case, "mode", -1), ""),
                            )
                            for case in cases
                        ]
                        for key, cases
                        in CALIBRATION_LEAF_BINDINGS.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.host_contract_only:
        selected_modes, selected_payload_bytes = select_execution_axes(args)
        if len(selected_modes) != 1 or len(selected_payload_bytes) != 1:
            raise RuntimeError(
                "--host-contract-only requires exactly one --mode and one "
                "--payload-bytes"
            )
        print(
            "pending_dte_host_contract: "
            + json.dumps(
                run_pending_host_contract(
                    selected_modes[0], selected_payload_bytes[0]
                ),
                sort_keys=True,
            )
        )
        return 0
    require_build_args(args)
    selected_modes, selected_payload_bytes = select_execution_axes(args)
    args.repo_root = args.repo_root.resolve()
    total_timeout = args.total_timeout_seconds
    if total_timeout is None:
        total_timeout = (
            DEFAULT_NO_CARD_TOTAL_TIMEOUT_SECONDS
            if args.no_card
            else DEFAULT_BOARD_TOTAL_TIMEOUT_SECONDS
        )
    start_total_deadline(total_timeout)
    run_host_oracle_self_tests()
    print("host_oracle_self_test: passed")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print(
            "wafer_board_dte_ncc_execution_probe_test: hardware execution is "
            "not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
            file=sys.stderr,
        )
        return 77
    if not args.no_card:
        validate_board_args(args)

    package, module_path, bindings = compile_package(args)
    verify_no_card(args, package)
    build_probe(args, package, module_path)
    verify_no_card(args, package)
    print("probe_package_verification: passed")
    if args.no_card:
        for mode in selected_modes:
            for payload_bytes in selected_payload_bytes:
                if payload_bytes in MODE_PAYLOADS[mode]:
                    write_probe_inputs(args.work_dir, bindings, mode, payload_bytes)
        print("probe_inputs_and_pytorch_reference: prepared")
    if not args.no_card:
        if any(mode in ISOLATED_DTE_MODES for mode in selected_modes):
            execute_receiver_unprepared(args, package, bindings)
        else:
            execute_probe_modes(
                args,
                package,
                bindings,
                selected_modes,
                selected_payload_bytes,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"wafer_board_dte_ncc_execution_probe_test: {error}", file=sys.stderr)
        raise SystemExit(1)
