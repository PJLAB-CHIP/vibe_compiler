#!/usr/bin/env python3
"""Build and run the 16-rank DDR allocation/offset latency probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys


RANK_COUNT = 16
ALLOCATION_COUNT = 2
OFFSET_CLASSES = (
    0,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)
DIRECTIONS = ("rdma", "wdma")
CELL_SAMPLES = 3
RESOURCE_BYTES = 0x200000
LOCAL_ELEMENTS = RESOURCE_BYTES // 4
PAYLOAD_BYTES = 256
GUARD_BYTES = 256
SLOT_BYTES = PAYLOAD_BYTES + 2 * GUARD_BYTES
CANARY = 0xC3
OUTPUT_INITIAL = 0xA5
CANARY_SOURCE_OFFSET = 0x2000
ARCHIVE_BASE = 0x10000
SWEEP_BASE = 0x80000
SPM_GUARDED = 0x10000
SPM_PAYLOAD = 0x20000
REQUEST_MAGIC = 0x5744445254494C45
RECORD_MAGIC = 0x5744445252454344
ROW_MAGIC = 0x57444452524F5721
REQUEST_GUARD = 0x8C21A549F0E36DB7
RECORD_GUARD = 0xB41EF09C7263D85A
ROW_GUARD = 0x6D9703F1CA4285BE
SCHEMA = 1
REQUEST_WORDS = 16
HEADER_WORDS = 32
ROW_WORDS = 16
ROW_COUNT = (
    ALLOCATION_COUNT
    * len(OFFSET_CLASSES)
    * len(DIRECTIONS)
    * CELL_SAMPLES
)
RECORD_BYTES = (HEADER_WORDS + ROW_COUNT * ROW_WORDS) * 8
LAUNCH_ABI = "tx81-cluster-direct-dte-prepare-main-v1"
STATUS_ABI = "wafer-direct-dte-status-v2"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ddr_tile_offset_probe.c"
PROBE_LL = INPUT_DIR / "wafer_ddr_tile_offset_probe.ll"

SHARDING = "{devices=[16,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
MODULE = f"""\
module {{
  wafer.target.topology @default {{card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}}
  wafer.execution.mesh @default_mesh {{topology = @default, axes = ["rank"], shape = array<i64: 16>, policy = "all_available", endpoints = array<i64>}}
  func.func @main(
      %arg0: tensor<16x{LOCAL_ELEMENTS}xf32>,
      %arg1: tensor<16x{LOCAL_ELEMENTS}xf32>)
      -> (tensor<16x{LOCAL_ELEMENTS}xf32>,
          tensor<16x{LOCAL_ELEMENTS}xf32>) {{
    %sharded0 = stablehlo.custom_call @Sharding(%arg0) {{
      backend_config = "",
      mhlo.sharding = "{SHARDING}"
    }} : (tensor<16x{LOCAL_ELEMENTS}xf32>) -> tensor<16x{LOCAL_ELEMENTS}xf32>
    %sharded1 = stablehlo.custom_call @Sharding(%arg1) {{
      backend_config = "",
      mhlo.sharding = "{SHARDING}"
    }} : (tensor<16x{LOCAL_ELEMENTS}xf32>) -> tensor<16x{LOCAL_ELEMENTS}xf32>
    %slice = "stablehlo.slice"(%sharded0) {{
      start_indices = array<i64: 0, 0>,
      limit_indices = array<i64: 16, 1>,
      strides = array<i64: 1, 1>
    }} : (tensor<16x{LOCAL_ELEMENTS}xf32>) -> tensor<16x1xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = "stablehlo.reduce"(%slice, %zero) ({{
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %value = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %value : tensor<f32>
    }}) {{dimensions = array<i64: 0>}} : (tensor<16x1xf32>, tensor<f32>) -> tensor<1xf32>
    %broadcast = "stablehlo.broadcast_in_dim"(%sum) {{
      broadcast_dimensions = array<i64: 1>
    }} : (tensor<1xf32>) -> tensor<16x{LOCAL_ELEMENTS}xf32>
    %result0 = stablehlo.add %sharded0, %broadcast : tensor<16x{LOCAL_ELEMENTS}xf32>
    %result1 = stablehlo.add %sharded1, %broadcast : tensor<16x{LOCAL_ELEMENTS}xf32>
    return %result0, %result1 : tensor<16x{LOCAL_ELEMENTS}xf32>,
        tensor<16x{LOCAL_ELEMENTS}xf32>
  }}
}}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float32",
            "dynamic_dims": [],
        },
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float32",
            "dynamic_dims": [],
        },
    ],
    "output_signature": [
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float32",
            "dynamic_dims": [],
        },
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float32",
            "dynamic_dims": [],
        },
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "allocation0"},
        {"type_": "input_arg", "position": 1, "name": "allocation1"},
    ],
    "unused_inputs": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=600000)
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def matrix_cases() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "allocation_ordinal": allocation,
            "offset_class": offset,
            "direction": direction,
            "samples": CELL_SAMPLES,
            "bytes": PAYLOAD_BYTES,
            "disposition": "manual-board-observation",
            "oracle": "full-payload+prefix/suffix-guard+terminal",
        }
        for rank in range(RANK_COUNT)
        for allocation in range(ALLOCATION_COUNT)
        for offset in OFFSET_CLASSES
        for direction in DIRECTIONS
    ]


def validate_static_contract() -> None:
    cases = matrix_cases()
    expected_cells = (
        RANK_COUNT
        * ALLOCATION_COUNT
        * len(OFFSET_CLASSES)
        * len(DIRECTIONS)
    )
    if (
        len(cases) != expected_cells
        or len(
            {
                (
                    row["rank"],
                    row["allocation_ordinal"],
                    row["offset_class"],
                    row["direction"],
                )
                for row in cases
            }
        )
        != expected_cells
        or any(offset % 256 for offset in OFFSET_CLASSES)
        or OFFSET_CLASSES[-1] + SWEEP_BASE + PAYLOAD_BYTES + GUARD_BYTES
        > RESOURCE_BYTES
        or ARCHIVE_BASE
        + len(OFFSET_CLASSES)
        * len(DIRECTIONS)
        * CELL_SAMPLES
        * SLOT_BYTES
        > SWEEP_BASE
        or RECORD_BYTES > ARCHIVE_BASE
        or LOCAL_ELEMENTS * 4 != RESOURCE_BYTES
    ):
        raise RuntimeError("DDR tile/offset matrix has an invalid static contract")


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
            "DDR tile/offset probe timed out; no retry, reset, or power "
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


def expected_rank_slices() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "replica_id": 0,
            "offsets": [rank, 0],
            "sizes": [1, LOCAL_ELEMENTS],
            "strides": [1, 1],
        }
        for rank in range(RANK_COUNT)
    ]


def validate_boundary_binding(
    binding: object, index_name: str, index: int
) -> None:
    if not isinstance(binding, dict) or binding != {
        index_name: index,
        "distribution": "partitioned",
        "global_shape": [RANK_COUNT, LOCAL_ELEMENTS],
        "local_shape": [1, LOCAL_ELEMENTS],
        "dtype": "float32",
        "ranks": expected_rank_slices(),
    }:
        raise RuntimeError(
            f"DDR tile/offset {index_name} {index} is not the exact "
            "rank-partitioned resource"
        )


def validate_manifest(
    package: pathlib.Path,
) -> tuple[pathlib.Path, dict[tuple[int, str, int], int]]:
    metadata = json.loads((package / "functions" / "forward.meta").read_text())
    boundary = metadata.get("distributed_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("logical_rank_count") != RANK_COUNT
    ):
        raise RuntimeError("DDR tile/offset package omitted rank-16 boundary")
    inputs = boundary.get("inputs")
    outputs = boundary.get("outputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != ALLOCATION_COUNT
        or not isinstance(outputs, list)
        or len(outputs) != ALLOCATION_COUNT
    ):
        raise RuntimeError("DDR tile/offset boundary allocation count is wrong")
    for index, binding in enumerate(inputs):
        validate_boundary_binding(binding, "argument_index", index)
    for index, binding in enumerate(outputs):
        validate_boundary_binding(binding, "result_index", index)

    manifest = json.loads((package / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 5
        or manifest.get("rank_count") != RANK_COUNT
        or manifest.get("target", {}).get("launch_abi") != LAUNCH_ABI
    ):
        raise RuntimeError("DDR tile/offset package launch contract is wrong")
    modules = manifest.get("modules")
    entries = manifest.get("entries")
    resources = manifest.get("resources")
    if (
        not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(entries, list)
        or len(entries) != RANK_COUNT
        or not isinstance(resources, list)
        or len(resources) != RANK_COUNT * 5
    ):
        raise RuntimeError("DDR tile/offset package domains are incomplete")
    module = modules[0]
    if module.get("exports") != [
        {"role": "prepare", "symbol": "__wafer_cluster_prepare"},
        {"role": "main", "symbol": "main"},
    ]:
        raise RuntimeError("DDR tile/offset shared module exports are wrong")
    module_path = package / module["path"]
    if not module_path.is_file():
        raise RuntimeError("DDR tile/offset shared module is missing")

    resources_by_id = {
        resource.get("id"): resource
        for resource in resources
        if isinstance(resource, dict)
        and isinstance(resource.get("id"), int)
    }
    if len(resources_by_id) != len(resources):
        raise RuntimeError("DDR tile/offset resource ids are not unique")
    bindings: dict[tuple[int, str, int], int] = {}
    for resource in resources:
        rank = resource.get("rank")
        role = resource.get("role")
        role_index = resource.get("role_index")
        if role == "transport_status":
            if (
                resource.get("type") != {"dtype": "u32", "shape": [1]}
                or resource.get("bytes") != STATUS_STORAGE_BYTES
                or resource.get("alignment") != STATUS_STORAGE_ALIGNMENT
                or resource.get("host_visible") is not False
            ):
                raise RuntimeError("DDR tile/offset status resource is wrong")
            continue
        key = (rank, role, role_index)
        expected_access = (
            "read_only" if role == "user_input" else "write_only"
        )
        if (
            not isinstance(rank, int)
            or not 0 <= rank < RANK_COUNT
            or key in bindings
            or role not in {"user_input", "output"}
            or role_index not in range(ALLOCATION_COUNT)
            or resource.get("type")
            != {"dtype": "f32", "shape": [1, LOCAL_ELEMENTS]}
            or resource.get("bytes") != RESOURCE_BYTES
            or resource.get("access") != expected_access
            or resource.get("host_visible") is not True
        ):
            raise RuntimeError(
                f"DDR tile/offset host resource is invalid: {resource}"
            )
        bindings[key] = resource["id"]
    expected_bindings = {
        (rank, role, ordinal)
        for rank in range(RANK_COUNT)
        for role in ("user_input", "output")
        for ordinal in range(ALLOCATION_COUNT)
    }
    if set(bindings) != expected_bindings:
        raise RuntimeError("DDR tile/offset host resources are not all-and-only")

    for entry in entries:
        rank = entry.get("rank")
        transport = entry.get("transport")
        slots = entry.get("slots")
        if (
            not isinstance(rank, int)
            or not 0 <= rank < RANK_COUNT
            or entry.get("module") != module.get("id")
            or not isinstance(transport, dict)
            or transport.get("kind") != "direct_dte"
            or transport.get("status_abi") != STATUS_ABI
            or transport.get("host_watchdog_required") is not True
            or not isinstance(slots, list)
            or len(slots) != 5
        ):
            raise RuntimeError(f"rank {rank} cluster entry is invalid")
        status_id = transport.get("status_resource")
        expected_slots = [
            (0, bindings[(rank, "user_input", 0)], "read_only"),
            (1, bindings[(rank, "user_input", 1)], "read_only"),
            (2, bindings[(rank, "output", 0)], "write_only"),
            (3, bindings[(rank, "output", 1)], "write_only"),
            (4, status_id, "read_write"),
        ]
        actual_slots = [
            (slot.get("ordinal"), slot.get("resource"), slot.get("access"))
            for slot in slots
            if isinstance(slot, dict)
        ]
        if actual_slots != expected_slots:
            raise RuntimeError(f"rank {rank} slot order is not canonical")
    return module_path, bindings


def compile_package(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, pathlib.Path, dict[tuple[int, str, int], int]]:
    source = write_source_program(args.work_dir)
    package = args.work_dir / "package"
    result = run(
        [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(package),
            f"--execution-ranks={RANK_COUNT}",
            "--target-profile=wafer-tx81-single-card-kernel-v1",
            f"--launch-abi={LAUNCH_ABI}",
        ],
        timeout_seconds=600,
    )
    if "published verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not publish the DDR probe seed")
    module_path, bindings = validate_manifest(package)
    return package, module_path, bindings


def build_probe(
    args: argparse.Namespace, package: pathlib.Path, module_path: pathlib.Path
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
        raise RuntimeError(f"DDR tile/offset build dependencies missing: {missing}")

    build = args.work_dir / "probe-build"
    build.mkdir()
    helper = build / "wafer_ddr_tile_offset_probe.o"
    linked = build / "wafer_ddr_tile_offset_probe.so"
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
            f"-I{INPUT_DIR}",
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
            "tx8-kcore-loader-cluster-v1",
            "--extra-object",
            str(helper),
            "--extra-library-dir",
            str(kcore_lib),
            "--extra-library",
            "kcorert",
        ],
        timeout_seconds=180,
    )
    staged = module_path.with_name(f".{module_path.name}.ddr-tile-offset")
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(".manifest.json.ddr-tile-offset")
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged_manifest, manifest_path)


def verify_no_card(args: argparse.Namespace, package: pathlib.Path) -> None:
    result = run(
        [
            str(args.wafer_run),
            "--package-dir",
            str(package),
            "--all-ranks",
            "--no-card",
            "--direct-dte-status-abi",
            STATUS_ABI,
            "--supports-host-watchdog",
        ]
    )
    if (
        "board_execution: false" not in result.stdout
        or f"launch_abi={LAUNCH_ABI}" not in result.stdout
    ):
        raise RuntimeError("DDR tile/offset no-card launch evidence is incomplete")
    print("ddr_tile_offset_probe_no_card: passed")


def pattern_period(rank: int, allocation: int, launch_sample: int) -> bytes:
    return bytes(
        (
            rank * 37
            + allocation * 83
            + launch_sample * 29
            + lane * 17
            + (lane >> 4) * 11
            + 5
        )
        & 0xFF
        for lane in range(256)
    )


def make_input(
    rank: int, allocation: int, launch_sample: int
) -> tuple[bytes, bytes]:
    payload = bytearray(RESOURCE_BYTES)
    payload[
        CANARY_SOURCE_OFFSET : CANARY_SOURCE_OFFSET + SLOT_BYTES
    ] = bytes([CANARY]) * SLOT_BYTES
    period = pattern_period(rank, allocation, launch_sample)
    sweep_bytes = RESOURCE_BYTES - SWEEP_BASE
    payload[SWEEP_BASE:] = (period * ((sweep_bytes + 255) // 256))[
        :sweep_bytes
    ]
    if allocation == 0:
        words = [0] * REQUEST_WORDS
        words[0] = REQUEST_MAGIC
        words[1] = SCHEMA
        words[2] = rank
        words[3] = launch_sample
        words[4] = RESOURCE_BYTES
        words[5] = PAYLOAD_BYTES
        words[6] = ALLOCATION_COUNT
        words[7] = len(OFFSET_CLASSES)
        words[15] = REQUEST_GUARD
        payload[: REQUEST_WORDS * 8] = struct.pack(
            f"<{REQUEST_WORDS}Q", *words
        )
    expected = (period * ((PAYLOAD_BYTES + 255) // 256))[:PAYLOAD_BYTES]
    return bytes(payload), expected


def write_resources(
    work_dir: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    launch_sample: int,
) -> tuple[list[str], dict[tuple[int, int], pathlib.Path], dict[tuple[int, int], bytes]]:
    raw_dir = work_dir / f"raw-{launch_sample}"
    raw_dir.mkdir()
    arguments: list[str] = []
    outputs: dict[tuple[int, int], pathlib.Path] = {}
    expected_payloads: dict[tuple[int, int], bytes] = {}
    for rank in range(RANK_COUNT):
        for allocation in range(ALLOCATION_COUNT):
            input_path = raw_dir / (
                f"rank-{rank:02d}.allocation-{allocation}.input.raw"
            )
            output_path = raw_dir / (
                f"rank-{rank:02d}.allocation-{allocation}.output.raw"
            )
            payload, expected = make_input(rank, allocation, launch_sample)
            input_path.write_bytes(payload)
            outputs[(rank, allocation)] = output_path
            expected_payloads[(rank, allocation)] = expected
            arguments.extend(
                [
                    "--resource",
                    f"{bindings[(rank, 'user_input', allocation)]}={input_path}",
                    "--output",
                    f"{bindings[(rank, 'output', allocation)]}={output_path}",
                ]
            )
    return arguments, outputs, expected_payloads


def parse_tile_coordinates(stdout: str) -> dict[int, tuple[int, int]]:
    matches = re.findall(
        r"^board_tile: logical=(\d+) available=true "
        r"physical_x=(\d+) physical_y=(\d+)$",
        stdout,
        re.MULTILINE,
    )
    result = {
        int(rank): (int(physical_x), int(physical_y))
        for rank, physical_x, physical_y in matches
    }
    if set(result) != set(range(RANK_COUNT)):
        raise RuntimeError("board output omitted exact 16-tile coordinates")
    return result


def row_ordinal(
    allocation: int, direction: str, offset_index: int, cell_sample: int
) -> int:
    allocation_base = (
        allocation
        * len(DIRECTIONS)
        * len(OFFSET_CLASSES)
        * CELL_SAMPLES
    )
    direction_base = DIRECTIONS.index(direction) * len(OFFSET_CLASSES) * CELL_SAMPLES
    return (
        allocation_base
        + direction_base
        + offset_index * CELL_SAMPLES
        + cell_sample
    )


def archive_offset(
    direction: str, offset_index: int, cell_sample: int
) -> int:
    return ARCHIVE_BASE + (
        DIRECTIONS.index(direction) * len(OFFSET_CLASSES) * CELL_SAMPLES
        + offset_index * CELL_SAMPLES
        + cell_sample
    ) * SLOT_BYTES


def validate_rank_outputs(
    rank: int,
    launch_sample: int,
    paths: dict[tuple[int, int], pathlib.Path],
    expected_payloads: dict[tuple[int, int], bytes],
    physical: tuple[int, int],
) -> list[dict[str, object]]:
    raw_outputs = {
        allocation: paths[(rank, allocation)].read_bytes()
        for allocation in range(ALLOCATION_COUNT)
    }
    if any(len(raw) != RESOURCE_BYTES for raw in raw_outputs.values()):
        raise RuntimeError(f"rank {rank}: output resource size is wrong")
    header = struct.unpack_from(f"<{HEADER_WORDS}Q", raw_outputs[0])
    expected_header = {
        0: RECORD_MAGIC,
        1: (SCHEMA << 32) | HEADER_WORDS,
        2: 0,
        3: rank,
        4: launch_sample,
        5: ROW_COUNT,
        6: PAYLOAD_BYTES,
        7: GUARD_BYTES,
        12: REQUEST_GUARD,
        31: RECORD_GUARD,
    }
    failures = {
        index: (header[index], expected)
        for index, expected in expected_header.items()
        if header[index] != expected
    }
    if failures:
        raise RuntimeError(f"rank {rank}: header oracle failed: {failures}")
    input_bases = header[8:10]
    output_bases = header[10:12]
    if (
        any(base == 0 or base % 256 for base in input_bases + output_bases)
        or len(set(input_bases + output_bases)) != 4
    ):
        raise RuntimeError(
            f"rank {rank}: runtime allocation bases are null, aliased, or "
            "not 256-byte aligned"
        )

    observations: list[dict[str, object]] = []
    for allocation in range(ALLOCATION_COUNT):
        payload = expected_payloads[(rank, allocation)]
        expected_slot = (
            bytes([CANARY]) * GUARD_BYTES
            + payload
            + bytes([CANARY]) * GUARD_BYTES
        )
        for direction in DIRECTIONS:
            for offset_index, offset in enumerate(OFFSET_CLASSES):
                for cell_sample in range(CELL_SAMPLES):
                    ordinal = row_ordinal(
                        allocation, direction, offset_index, cell_sample
                    )
                    row_offset = (HEADER_WORDS + ordinal * ROW_WORDS) * 8
                    row = struct.unpack_from(
                        f"<{ROW_WORDS}Q", raw_outputs[0], row_offset
                    )
                    identity = (
                        rank
                        | allocation << 8
                        | (1 if direction == "rdma" else 2) << 16
                        | offset_index << 24
                        | cell_sample << 32
                    )
                    measured_address = (
                        input_bases[allocation]
                        if direction == "rdma"
                        else output_bases[allocation]
                    ) + SWEEP_BASE + offset
                    expected_spm = (
                        SPM_GUARDED + GUARD_BYTES
                        if direction == "rdma"
                        else SPM_PAYLOAD
                    )
                    expected_archive = archive_offset(
                        direction, offset_index, cell_sample
                    )
                    expected_row = {
                        0: ROW_MAGIC,
                        1: identity,
                        2: offset,
                        3: input_bases[allocation],
                        4: output_bases[allocation],
                        5: measured_address,
                        6: expected_spm,
                        8: 1,
                        13: PAYLOAD_BYTES,
                        14: expected_archive,
                        15: ROW_GUARD,
                    }
                    row_failures = {
                        index: (row[index], expected)
                        for index, expected in expected_row.items()
                        if row[index] != expected
                    }
                    if row_failures or row[7] == 0 or row[10] == 0:
                        raise RuntimeError(
                            f"rank {rank} allocation {allocation} "
                            f"{direction} offset {offset} sample "
                            f"{cell_sample}: row oracle failed "
                            f"{row_failures}, completion={row[7]}, "
                            f"engine={row[10]}"
                        )
                    archived = raw_outputs[allocation][
                        expected_archive : expected_archive + SLOT_BYTES
                    ]
                    if archived != expected_slot:
                        mismatch = next(
                            index
                            for index, (actual, expected) in enumerate(
                                zip(archived, expected_slot, strict=True)
                            )
                            if actual != expected
                        )
                        raise RuntimeError(
                            f"rank {rank} allocation {allocation} "
                            f"{direction} offset {offset} sample "
                            f"{cell_sample}: archive mismatch at {mismatch}"
                        )
                    observations.append(
                        {
                            "rank": rank,
                            "physical_x": physical[0],
                            "physical_y": physical[1],
                            "launch_sample": launch_sample,
                            "allocation_ordinal": allocation,
                            "input_base": input_bases[allocation],
                            "output_base": output_bases[allocation],
                            "offset_class": offset,
                            "direction": direction,
                            "cell_sample": cell_sample,
                            "measured_ddr_address": measured_address,
                            "payload_bytes": PAYLOAD_BYTES,
                            "issue_to_matching_completion_cycles": row[7],
                            "instruction_delta": row[8],
                            "blocking_delta": row[9],
                            "engine_execution_delta": row[10],
                            "fu_execution_delta": row[11],
                            "statistics_window_delta": row[12],
                            "bytes_per_engine_cycle": PAYLOAD_BYTES / row[10],
                            "bytes_per_issue_to_completion_cycle": (
                                PAYLOAD_BYTES / row[7]
                            ),
                            "controller_class": "unclassified",
                            "correctness": "exact-with-prefix-suffix-guard",
                        }
                    )

    for allocation in range(ALLOCATION_COUNT):
        # wafer-run materializes every write-only --output binding with its
        # 0xa5 sentinel before launch.  Preserve that exact host-side
        # publication contract outside the declared device write ranges.
        expected = bytearray([OUTPUT_INITIAL]) * RESOURCE_BYTES
        payload = expected_payloads[(rank, allocation)]
        expected_slot = (
            bytes([CANARY]) * GUARD_BYTES
            + payload
            + bytes([CANARY]) * GUARD_BYTES
        )
        for offset_index, offset in enumerate(OFFSET_CLASSES):
            target = SWEEP_BASE + offset
            for cell_sample in range(CELL_SAMPLES):
                expected[
                    target - GUARD_BYTES : target + PAYLOAD_BYTES + GUARD_BYTES
                ] = bytes([CANARY]) * SLOT_BYTES
                expected[target : target + PAYLOAD_BYTES] = payload
                for direction in DIRECTIONS:
                    begin = archive_offset(
                        direction, offset_index, cell_sample
                    )
                    expected[begin : begin + SLOT_BYTES] = expected_slot
        if allocation == 0:
            expected[:RECORD_BYTES] = raw_outputs[0][:RECORD_BYTES]
        if raw_outputs[allocation] != bytes(expected):
            mismatch = next(
                index
                for index, (actual, wanted) in enumerate(
                    zip(raw_outputs[allocation], expected, strict=True)
                )
                if actual != wanted
            )
            raise RuntimeError(
                f"rank {rank} allocation {allocation}: output changed "
                f"outside declared records/results at {mismatch}"
            )
    return observations


def summarize(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for rank in range(RANK_COUNT):
        for allocation in range(ALLOCATION_COUNT):
            for direction in DIRECTIONS:
                for offset in OFFSET_CLASSES:
                    rows = [
                        row
                        for row in observations
                        if row["rank"] == rank
                        and row["allocation_ordinal"] == allocation
                        and row["direction"] == direction
                        and row["offset_class"] == offset
                    ]
                    if len(rows) != CELL_SAMPLES:
                        raise RuntimeError("DDR tile/offset cell is incomplete")
                    summaries.append(
                        {
                            "rank": rank,
                            "physical_x": rows[0]["physical_x"],
                            "physical_y": rows[0]["physical_y"],
                            "allocation_ordinal": allocation,
                            "actual_base": (
                                rows[0]["input_base"]
                                if direction == "rdma"
                                else rows[0]["output_base"]
                            ),
                            "offset_class": offset,
                            "direction": direction,
                            "samples": CELL_SAMPLES,
                            "completion_cycles_median": statistics.median(
                                int(
                                    row[
                                        "issue_to_matching_completion_cycles"
                                    ]
                                )
                                for row in rows
                            ),
                            "engine_execution_median": statistics.median(
                                int(row["engine_execution_delta"])
                                for row in rows
                            ),
                            "blocking_median": statistics.median(
                                int(row["blocking_delta"]) for row in rows
                            ),
                            "bytes_per_engine_cycle_median": statistics.median(
                                float(row["bytes_per_engine_cycle"])
                                for row in rows
                            ),
                            "controller_class": "unclassified",
                            "metric_basis": (
                                "engine_execution_delta is primary; "
                                "issue-to-completion includes the matching "
                                "terminal wait"
                            ),
                            "interpretation": (
                                "same-rank independent allocation and "
                                "relative-offset observation; no physical "
                                "DDR bank or hop formula inferred"
                            ),
                        }
                    )
    return summaries


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
    if args.expected_tile_count != RANK_COUNT:
        raise RuntimeError("DDR tile/offset probe requires exactly 16 tiles")
    if args.completion_timeout_ms <= 0 or args.repeat <= 0:
        raise RuntimeError("timeout and repeat must be positive")
    if re.fullmatch(
        r"[0-9a-fA-F]{64}", str(args.expected_runtime_library_sha256)
    ) is None:
        raise RuntimeError("runtime library SHA-256 must contain 64 hex digits")


def execute_board(
    args: argparse.Namespace,
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
) -> None:
    all_observations: list[dict[str, object]] = []
    for launch_sample in range(args.repeat):
        resource_args, paths, expected_payloads = write_resources(
            args.work_dir, bindings, launch_sample
        )
        result = run(
            [
                str(args.wafer_run),
                "--package-dir",
                str(package),
                "--all-ranks",
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
        required = {
            "launch_pattern: cluster-prepare-main-x16",
            "logical_tile_execution_basis: cluster-pid-and-exact-rank-slices",
            "logical_tile_domain: 0..15",
            "board_execution: true",
        }
        if not required.issubset(set(result.stdout.splitlines())):
            raise RuntimeError("DDR tile/offset board evidence is incomplete")
        coordinates = parse_tile_coordinates(result.stdout)
        launch_observations: list[dict[str, object]] = []
        for rank in range(RANK_COUNT):
            launch_observations.extend(
                validate_rank_outputs(
                    rank,
                    launch_sample,
                    paths,
                    expected_payloads,
                    coordinates[rank],
                )
            )
        all_observations.extend(launch_observations)
        print(
            "ddr_tile_offset_raw_observations: "
            + json.dumps(launch_observations, sort_keys=True)
        )
        print(
            "ddr_tile_offset_cell_summaries: "
            + json.dumps(summarize(launch_observations), sort_keys=True)
        )
        print(result.stdout, end="")


def main() -> int:
    args = parse_args()
    validate_static_contract()
    if args.list_cases:
        print(
            json.dumps(
                {
                    "matrix": matrix_cases(),
                    "allocation_contract": (
                        "each rank owns two independent input and two "
                        "independent output resources; cross-rank resources "
                        "are distinct allocations"
                    ),
                    "bank_contract": (
                        "offset classes and actual 64-bit bases are observed; "
                        "no physical bank/controller/hop mapping is assumed"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    args.repo_root = args.repo_root.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.work_dir == args.repo_root or args.work_dir in args.repo_root.parents:
        raise RuntimeError("DDR tile/offset work directory is too broad")
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "DDR tile/offset hardware execution is not armed",
                file=sys.stderr,
            )
            return 77
        validate_board_args(args)
    package, module_path, bindings = compile_package(args)
    build_probe(args, package, module_path)
    verify_no_card(args, package)
    if args.no_card:
        return 0
    execute_board(args, package, bindings)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"wafer_board_ddr_tile_offset_probe_test: {error}", file=sys.stderr)
        raise SystemExit(1)
