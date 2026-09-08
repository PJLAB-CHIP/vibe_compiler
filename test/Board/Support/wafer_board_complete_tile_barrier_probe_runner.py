#!/usr/bin/env python3
"""Build and run a two-epoch 16-Tile production hrt_barrier probe."""

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

import wafer_board_source_program as source_program
import wafer_runtime_launch_contract as runtime_launch


TILE_COUNT = 16
UNSUPPORTED_PARTICIPANT_COUNTS = (1, 2, 4, 8, 15)
RESOURCE_BYTES = 256
STATUS_ABI = "wafer-direct-dte-status"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_complete_tile_barrier_probe.c"
PROBE_LL = INPUT_DIR / "wafer_complete_tile_barrier_probe.ll"

REQUEST_MAGIC = 0x5742464352455154
REQUEST_WORDS = 4
REQUEST_GUARD = 0x86DB3E71A5942FC0
RECORD_MAGIC = 0x5742464352455354
RECORD_WORDS = 16
RECORD_GUARD = 0xC34A6F9128ED750B
INPUT_CANARY = 0xC3
OUTPUT_CANARY = 0xA5
EPOCH1_OFFSET = 0
EPOCH2_OFFSET = 64
RECORD_OFFSET = 128
EPOCH1_BASE = 0xE101CA1B00000000
EPOCH2_BASE = 0xE202CA1B00000000
DELAY1_SCALE = 8192
DELAY2_SCALE = 6144
STATUS_OK = 1
ALL_STEPS = 0xFF

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
        {"shape": [RESOURCE_BYTES // 2], "dtype": "float16", "dynamic_dims": []}
    ],
    "output_signature": [
        {"shape": [RESOURCE_BYTES // 2], "dtype": "float16", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request"}
    ],
    "unused_inputs": [],
}


@dataclasses.dataclass(frozen=True)
class BarrierCalibrationCase:
    name: str
    participants: int
    disposition: str
    epoch: int | None
    oracle: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProbeSlotLayout:
    slots_per_tile: int
    input_ordinal: int
    output_ordinal: int
    status_ordinal: int


BARRIER_POSITIVE_CASES = (
    BarrierCalibrationCase(
        "complete-Tile-domain-epoch1-tile-increasing-delay",
        TILE_COUNT,
        "board-executable",
        1,
        "16 Tile-specific markers, zero mismatch and zero crosstalk",
        "current hrt_barrier contract owns exactly sixteen participant slots",
    ),
    BarrierCalibrationCase(
        "complete-Tile-domain-epoch2-tile-reverse-delay",
        TILE_COUNT,
        "board-executable",
        2,
        "16 Tile-specific markers, zero mismatch and zero crosstalk",
        "the second epoch reverses delay order and reuses the same barrier",
    ),
    BarrierCalibrationCase(
        "complete-Tile-domain-two-epoch-reuse",
        TILE_COUNT,
        "board-executable",
        None,
        "both ordered epochs complete in one launch with disjoint markers",
        "reuse is accepted only when both epoch-specific oracles pass",
    ),
)
BARRIER_NEGATIVE_CASES = tuple(
    BarrierCalibrationCase(
        f"subgroup-{participants}-participants",
        participants,
        "static-negative",
        None,
        "host rejection before compile or device submission",
        "hrt_barrier exposes sixteen fixed participant slots",
    )
    for participants in UNSUPPORTED_PARTICIPANT_COUNTS
)
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "multi-tile-arrival": BARRIER_POSITIVE_CASES,
    "barrier-participant-negative": BARRIER_NEGATIVE_CASES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--participants", type=int, default=TILE_COUNT)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    return parser.parse_args()


def validate_participant_count(participants: int) -> None:
    if participants != TILE_COUNT:
        raise RuntimeError(
            "current hrt_barrier is fixed to exactly 16 participant slots; "
            f"refusing unsafe subgroup size {participants} before compile "
            "or device submission"
        )


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
            "complete-Tile-domain barrier probe timed out; no retry, reset, or power "
            "operation was attempted"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{shlex.join(command)}"
        )
    return result


def validate_work_dir(
    repo_root: pathlib.Path, work_dir: pathlib.Path
) -> pathlib.Path:
    resolved_repo = repo_root.resolve()
    resolved_work = work_dir.resolve()
    if resolved_work == resolved_repo or resolved_work in resolved_repo.parents:
        raise RuntimeError("barrier probe work directory is too broad")
    return resolved_work


def validate_host_contract() -> None:
    epoch1 = {EPOCH1_BASE | tile_id for tile_id in range(TILE_COUNT)}
    epoch2 = {EPOCH2_BASE | tile_id for tile_id in range(TILE_COUNT)}
    if (
        len(epoch1) != TILE_COUNT
        or len(epoch2) != TILE_COUNT
        or epoch1 & epoch2
        or RECORD_OFFSET + 16 * 8 != RESOURCE_BYTES
    ):
        raise RuntimeError("barrier probe host contract is malformed")
    for tile_id in range(TILE_COUNT):
        request = make_request(tile_id)
        words = struct.unpack("<32Q", request)
        if (
            words[0] != REQUEST_MAGIC
            or words[1] != REQUEST_WORDS
            or words[2] != tile_id
            or words[3] != REQUEST_GUARD
            or request[32:] != bytes([INPUT_CANARY]) * (RESOURCE_BYTES - 32)
        ):
            raise RuntimeError(f"Tile {tile_id} request oracle is not exact")


def write_source_program(work_dir: pathlib.Path) -> pathlib.Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    (source / "data").mkdir(parents=True)
    return source_program.write_program(source, MODULE, METADATA)


def compile_package(
    args: argparse.Namespace,
) -> tuple[
    pathlib.Path,
    pathlib.Path,
    dict[tuple[int, str, int], int],
    ProbeSlotLayout,
]:
    source = write_source_program(args.work_dir)
    package = args.work_dir / "package"
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
        raise RuntimeError("wafer-compile did not write the barrier seed")
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
        context="barrier probe",
    )
    bindings: dict[tuple[int, str, int], int] = {}
    for (table, role_index), port_id in shared_bindings.items():
        role = "user_input" if table == "inputs" else "output"
        for tile_id in range(TILE_COUNT):
            bindings[(tile_id, role, role_index)] = port_id
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    slot_layout = validate_terminal_slots(package, bindings)
    module_path = package / manifest["modules"][0]["path"]
    return package, module_path, bindings, slot_layout


def validate_terminal_slots(
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
) -> ProbeSlotLayout:
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="barrier probe",
    )
    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="barrier probe"
    )
    common_layout: ProbeSlotLayout | None = None
    for entry in entries:
        tile_id = entry.get("tile_id")
        arguments = entry.get("arguments")
        transport = entry.get("transport")
        if (
            not isinstance(tile_id, int)
            or not 0 <= tile_id < TILE_COUNT
            or not isinstance(arguments, list)
            or not arguments
            or not all(isinstance(argument, dict) for argument in arguments)
            or [argument.get("ordinal") for argument in arguments]
            != list(range(len(arguments)))
            or not isinstance(transport, dict)
        ):
            raise RuntimeError(
                "barrier Tile does not expose canonical ordered arguments"
            )
        input_id = bindings[(tile_id, "user_input", 0)]
        output_id = bindings[(tile_id, "output", 0)]
        input_arguments = [
            argument
            for argument in arguments
            if argument.get("kind") == "external_input"
            and argument.get("port") == input_id
        ]
        output_arguments = [
            argument
            for argument in arguments
            if argument.get("kind") == "external_output"
            and argument.get("port") == output_id
        ]
        status_arguments = [
            argument
            for argument in arguments
            if argument.get("kind") == "transport_status"
        ]
        if (
            len(input_arguments) != 1
            or input_arguments[0].get("access") != "read_only"
            or len(output_arguments) != 1
            or output_arguments[0].get("access") != "write_only"
            or len(status_arguments) != 1
            or status_arguments[0].get("access") != "read_write"
            or status_arguments[0].get("status_abi") != STATUS_ABI
            or status_arguments[0].get("bytes") != STATUS_STORAGE_BYTES
            or status_arguments[0].get("alignment")
            != STATUS_STORAGE_ALIGNMENT
            or transport.get("kind") != "direct_dte"
            or transport.get("status_abi") != STATUS_ABI
            or transport.get("host_watchdog_required") is not True
        ):
            raise RuntimeError(
                f"Tile {tile_id} transport arguments are not canonically bound"
            )
        layout = ProbeSlotLayout(
            slots_per_tile=len(arguments),
            input_ordinal=input_arguments[0]["ordinal"],
            output_ordinal=output_arguments[0]["ordinal"],
            status_ordinal=status_arguments[0]["ordinal"],
        )
        if common_layout is None:
            common_layout = layout
        elif layout != common_layout:
            raise RuntimeError(
                "barrier tile-major slot layout differs across Tiles"
            )
    if common_layout is None:
        raise RuntimeError("barrier tile-major slot layout is missing")
    return common_layout


def build_probe(
    args: argparse.Namespace,
    package: pathlib.Path,
    module_path: pathlib.Path,
    slot_layout: ProbeSlotLayout,
) -> None:
    deps = args.repo_root / "third_party" / "tx8_deps"
    tool_bin = deps / TOOLCHAIN_DIR / "bin"
    gcc = tool_bin / "riscv64-unknown-elf-gcc"
    objcopy = tool_bin / "riscv64-unknown-elf-objcopy"
    device_linker = args.repo_root / "tools" / "wafer_device_link.py"
    kcore_lib = deps / "tx8-yoc-rt-thread-smp" / "lib"
    required = (
        gcc,
        objcopy,
        device_linker,
        args.llvm_clangxx,
        PROBE_C,
        PROBE_LL,
        kcore_lib / "libkcorert.a",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"barrier probe build dependencies missing: {missing}")

    build = args.work_dir / "probe-build"
    build.mkdir()
    helper = build / "wafer_complete_tile_barrier_probe.o"
    linked = build / "wafer_complete_tile_barrier_probe.so"
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
            (
                "-DWAFER_BARRIER_SLOTS_PER_TILE="
                f"{slot_layout.slots_per_tile}"
            ),
            f"-DWAFER_BARRIER_INPUT_SLOT={slot_layout.input_ordinal}",
            f"-DWAFER_BARRIER_OUTPUT_SLOT={slot_layout.output_ordinal}",
            f"-DWAFER_BARRIER_STATUS_SLOT={slot_layout.status_ordinal}",
            f"-I{INPUT_DIR}",
            f"-I{args.repo_root / 'runtime' / 'crt' / 'include'}",
            f"-I{args.repo_root / 'include'}",
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
            "--extra-library-dir",
            str(kcore_lib),
            "--extra-library",
            "kcorert",
        ],
        timeout_seconds=120,
    )

    staged = module_path.with_name(f".{module_path.name}.complete-tile-barrier")
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(
        ".manifest.json.complete-tile-barrier"
    )
    staged_manifest.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    os.replace(staged_manifest, manifest_path)


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
        raise RuntimeError("no-card output omitted the kernel invocation")
    print("complete_tile_barrier_probe_no_card: passed")


def make_request(tile_id: int) -> bytes:
    payload = bytearray([INPUT_CANARY] * RESOURCE_BYTES)
    payload[:32] = struct.pack(
        "<4Q", REQUEST_MAGIC, REQUEST_WORDS, tile_id, REQUEST_GUARD
    )
    return bytes(payload)


def write_board_resources(
    work_dir: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
) -> tuple[list[str], pathlib.Path]:
    raw = work_dir / "raw"
    raw.mkdir()
    input_ids = {bindings[(tile_id, "user_input", 0)] for tile_id in range(TILE_COUNT)}
    output_ids = {bindings[(tile_id, "output", 0)] for tile_id in range(TILE_COUNT)}
    if len(input_ids) != 1 or len(output_ids) != 1:
        raise RuntimeError("barrier resources are not shared across all Tiles")
    request = raw / "requests.raw"
    output = raw / "results.raw"
    request.write_bytes(b"".join(make_request(tile_id) for tile_id in range(TILE_COUNT)))
    return [
        "--resource",
        f"{next(iter(input_ids))}={request}",
        "--output",
        f"{next(iter(output_ids))}={output}",
    ], output


def parse_output(path: pathlib.Path, tile_id: int) -> dict[str, int]:
    combined = path.read_bytes()
    if len(combined) != TILE_COUNT * RESOURCE_BYTES:
        raise RuntimeError("barrier result size does not cover all Tiles")
    begin = tile_id * RESOURCE_BYTES
    payload = combined[begin : begin + RESOURCE_BYTES]
    epoch1 = struct.unpack_from("<Q", payload, EPOCH1_OFFSET)[0]
    epoch2 = struct.unpack_from("<Q", payload, EPOCH2_OFFSET)[0]
    if epoch1 != EPOCH1_BASE | tile_id or epoch2 != EPOCH2_BASE | tile_id:
        raise RuntimeError(f"Tile {tile_id} has an incorrect epoch marker")
    if (
        payload[8:EPOCH2_OFFSET]
        != bytes([OUTPUT_CANARY]) * (EPOCH2_OFFSET - 8)
        or payload[EPOCH2_OFFSET + 8 : RECORD_OFFSET]
        != bytes([OUTPUT_CANARY]) * (RECORD_OFFSET - EPOCH2_OFFSET - 8)
    ):
        raise RuntimeError(f"Tile {tile_id} marker guard was modified")

    words = struct.unpack_from("<16Q", payload, RECORD_OFFSET)
    metadata = (
        RECORD_WORDS
        | (tile_id << 16)
        | (STATUS_OK << 24)
        | (TILE_COUNT << 32)
    )
    expected = {
        0: RECORD_MAGIC,
        1: metadata,
        2: 0,
        3: 0,
        4: 0,
        5: EPOCH1_BASE | tile_id,
        6: EPOCH2_BASE | tile_id,
        7: (tile_id + 1) * DELAY1_SCALE,
        8: (TILE_COUNT - tile_id) * DELAY2_SCALE,
        11: 2,
        12: ALL_STEPS,
        13: REQUEST_GUARD,
        14: TILE_COUNT,
        15: RECORD_GUARD,
    }
    for index, value in expected.items():
        if words[index] != value:
            raise RuntimeError(
                f"Tile {tile_id} record[{index}]={words[index]:#x}, "
                f"expected {value:#x}"
            )
    if words[9] == 0 or words[10] == 0:
        raise RuntimeError(f"Tile {tile_id} reported a zero barrier duration")
    return {
        "tile_id": tile_id,
        "epoch1_cycles": words[9],
        "epoch2_cycles": words[10],
        "epoch1_mismatches": words[2],
        "epoch2_mismatches": words[3],
        "epoch1_crosstalk": words[4],
    }


def validate_board_args(args: argparse.Namespace) -> None:
    required = {
        "--expected-runtime-version": args.expected_runtime_version,
        "--expected-device-name": args.expected_device_name,
        "--expected-pci-bus-id": args.expected_pci_bus_id,
        "--expected-tile-count": args.expected_tile_count,
        "--expected-runtime-library-sha256": (
            args.expected_runtime_library_sha256
        ),
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise RuntimeError(f"board execution requires qualification: {missing}")
    if args.expected_tile_count != TILE_COUNT:
        raise RuntimeError("complete-Tile-domain hrt_barrier probe requires exactly 16 tiles")
    if args.completion_timeout_ms <= 0:
        raise RuntimeError("completion timeout must be positive")
    if re.fullmatch(
        r"[0-9a-fA-F]{64}", str(args.expected_runtime_library_sha256)
    ) is None:
        raise RuntimeError("runtime library SHA-256 must contain 64 hex digits")


def execute_board(
    args: argparse.Namespace,
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
) -> None:
    resource_args, output = write_board_resources(args.work_dir, bindings)
    result = run(
        [
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
            *resource_args,
        ],
        timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0,
    )
    required = {"launch_pattern: cluster-x16", "board_execution: true"}
    if not required.issubset(set(result.stdout.splitlines())):
        raise RuntimeError("board output omitted 16-Tile launch evidence")
    runtime_launch.require_board_completion(stdout=result.stdout, context="barrier")
    observations = [parse_output(output, tile_id) for tile_id in range(TILE_COUNT)]
    print(
        "complete_tile_barrier_observations: "
        + json.dumps(observations, sort_keys=True)
    )
    print(result.stdout, end="")


def main() -> int:
    args = parse_args()
    validate_participant_count(args.participants)
    args.repo_root = args.repo_root.resolve()
    args.work_dir = validate_work_dir(args.repo_root, args.work_dir)
    validate_host_contract()
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "complete-Tile-domain barrier hardware execution is not armed",
                file=sys.stderr,
            )
            return 77
        validate_board_args(args)

    package, module_path, bindings, slot_layout = compile_package(args)
    build_probe(args, package, module_path, slot_layout)
    verify_no_card(args, package)
    if args.no_card:
        return 0
    execute_board(args, package, bindings)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"wafer_board_complete_tile_barrier_probe_runner: {error}", file=sys.stderr)
        raise SystemExit(1)
