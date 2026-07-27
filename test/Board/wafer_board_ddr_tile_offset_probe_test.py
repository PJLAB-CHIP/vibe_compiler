#!/usr/bin/env python3
"""Build the 16-rank DDR probe and run one explicitly selected observation mode."""

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

import wafer_runtime_launch_contract as runtime_launch


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
ELEMENT_BYTES = 2
LOCAL_ELEMENTS = RESOURCE_BYTES // ELEMENT_BYTES
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
CONFLICT_BASE_OFFSETS = (0x20000, 0x40000)
CONFLICT_OFFSET_CLASSES = (0, 4096, 32768)
CONFLICT_TRANSFER_BYTES = (256, 4096)
CONFLICT_ISSUE_ORDERS = ("a-b", "b-a")
CONFLICT_SCHEDULES = ("serial", "window")
CONFLICT_CELL_SAMPLES = 2
CONFLICT_PAIR_GAP = 0x10000
CONFLICT_SPM_A = 0x10000
CONFLICT_SPM_B = 0x30000
CONFLICT_SMALL_ARCHIVE_BASE = 0x20000
CONFLICT_LARGE_ARCHIVE_BASE = 0x190000
REQUEST_MAGIC = 0x5744445254494C45
RECORD_MAGIC = 0x5744445252454344
ROW_MAGIC = 0x57444452524F5721
CONFLICT_ROW_MAGIC = 0x5744445243464C54
REQUEST_GUARD = 0x8C21A549F0E36DB7
RECORD_GUARD = 0xB41EF09C7263D85A
ROW_GUARD = 0x6D9703F1CA4285BE
CONFLICT_ROW_GUARD = 0x71A5CE29B406DF83
SCHEMA = 3
REQUEST_WORDS = 16
HEADER_WORDS = 32
ROW_WORDS = 16
ROW_COUNT = (
    ALLOCATION_COUNT
    * len(OFFSET_CLASSES)
    * len(DIRECTIONS)
    * CELL_SAMPLES
)
OFFSET_RECORD_BYTES = (HEADER_WORDS + ROW_COUNT * ROW_WORDS) * 8
CONFLICT_ROW_WORDS = 25
CONFLICT_ROW_COUNT = (
    ALLOCATION_COUNT
    * len(CONFLICT_BASE_OFFSETS)
    * len(CONFLICT_OFFSET_CLASSES)
    * len(CONFLICT_TRANSFER_BYTES)
    * len(CONFLICT_ISSUE_ORDERS)
    * len(CONFLICT_SCHEDULES)
    * CONFLICT_CELL_SAMPLES
)
CONFLICT_ROWS_PER_TRANSFER_PER_ALLOCATION = (
    len(CONFLICT_BASE_OFFSETS)
    * len(CONFLICT_OFFSET_CLASSES)
    * len(CONFLICT_ISSUE_ORDERS)
    * len(CONFLICT_SCHEDULES)
    * CONFLICT_CELL_SAMPLES
)
CONFLICT_RECORD_BYTES = (
    HEADER_WORDS + CONFLICT_ROW_COUNT * CONFLICT_ROW_WORDS
) * 8
MAX_RECORD_BYTES = max(
    OFFSET_RECORD_BYTES,
    CONFLICT_RECORD_BYTES,
)
MODE_OFFSET = 0
MODE_CONFLICT_EQUIVALENCE = 1
LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
STATUS_ABI = "wafer-direct-dte-status-v2"
STATUS_STORAGE_BYTES = 64
STATUS_STORAGE_ALIGNMENT = 64
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ddr_tile_offset_probe.c"
PROBE_LL = INPUT_DIR / "wafer_ddr_tile_offset_probe.ll"
PROTOCOL_H = INPUT_DIR / "wafer_ddr_tile_offset_probe_protocol.h"

SHARDING = "{devices=[16,1]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
MODULE = f"""\
module {{
  wafer.target.topology @default {{card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}}
  wafer.execution.mesh @default_mesh {{topology = @default, axes = ["rank"], shape = array<i64: 16>, policy = "all_available", endpoints = array<i64>}}
  func.func @main(
      %arg0: tensor<16x{LOCAL_ELEMENTS}xf16>,
      %arg1: tensor<16x{LOCAL_ELEMENTS}xf16>)
      -> (tensor<16x{LOCAL_ELEMENTS}xf16>,
          tensor<16x{LOCAL_ELEMENTS}xf16>) {{
    %sharded0 = stablehlo.custom_call @Sharding(%arg0) {{
      backend_config = "",
      mhlo.sharding = "{SHARDING}"
    }} : (tensor<16x{LOCAL_ELEMENTS}xf16>) -> tensor<16x{LOCAL_ELEMENTS}xf16>
    %sharded1 = stablehlo.custom_call @Sharding(%arg1) {{
      backend_config = "",
      mhlo.sharding = "{SHARDING}"
    }} : (tensor<16x{LOCAL_ELEMENTS}xf16>) -> tensor<16x{LOCAL_ELEMENTS}xf16>
    %slice = "stablehlo.slice"(%sharded0) {{
      start_indices = array<i64: 0, {SWEEP_BASE // ELEMENT_BYTES}>,
      limit_indices = array<i64: 16, {SWEEP_BASE // ELEMENT_BYTES + 1}>,
      strides = array<i64: 1, 1>
    }} : (tensor<16x{LOCAL_ELEMENTS}xf16>) -> tensor<16x1xf16>
    %zero = stablehlo.constant dense<0.0> : tensor<f16>
    %sum = "stablehlo.reduce"(%slice, %zero) ({{
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %value = stablehlo.add %lhs, %rhs : tensor<f16>
      stablehlo.return %value : tensor<f16>
    }}) {{dimensions = array<i64: 0>}} : (tensor<16x1xf16>, tensor<f16>) -> tensor<1xf16>
    %broadcast = "stablehlo.broadcast_in_dim"(%sum) {{
      broadcast_dimensions = array<i64: 1>
    }} : (tensor<1xf16>) -> tensor<16x{LOCAL_ELEMENTS}xf16>
    %result0 = stablehlo.add %sharded0, %broadcast : tensor<16x{LOCAL_ELEMENTS}xf16>
    %result1 = stablehlo.add %sharded1, %broadcast : tensor<16x{LOCAL_ELEMENTS}xf16>
    return %result0, %result1 : tensor<16x{LOCAL_ELEMENTS}xf16>,
        tensor<16x{LOCAL_ELEMENTS}xf16>
  }}
}}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float16",
            "dynamic_dims": [],
        },
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float16",
            "dynamic_dims": [],
        },
    ],
    "output_signature": [
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float16",
            "dynamic_dims": [],
        },
        {
            "shape": [RANK_COUNT, LOCAL_ELEMENTS],
            "dtype": "float16",
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
    parser.add_argument(
        "--conflict-equivalence",
        action="store_true",
        help=(
            "select only the pending all-rank DDR conflict-equivalence "
            "suite; the default request remains the established offset probe"
        ),
    )
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


def conflict_equivalence_cases() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "allocation_ordinal": allocation,
            "base_offset": base_offset,
            "relative_offset_candidate": offset,
            "address_delta": CONFLICT_PAIR_GAP + offset,
            "transfer_bytes": transfer_bytes,
            "issue_orders": list(CONFLICT_ISSUE_ORDERS),
            "schedules": list(CONFLICT_SCHEDULES),
            "samples_per_order_schedule": CONFLICT_CELL_SAMPLES,
            "required_launch_orders": ["forward", "reverse"],
            "disposition": "pending-manual-board-observation",
            "oracle": (
                "same-invocation serial/window+a-b/b-a+pair-only-pmu+"
                "counterbalanced-schedule/rank-order+two-exact-payloads+"
                "prefix/suffix-guards+instruction-count"
            ),
            "compiler_use": "no-ddr-bank-coloring-until-board-evidence",
        }
        for rank in range(RANK_COUNT)
        for allocation in range(ALLOCATION_COUNT)
        for base_offset in CONFLICT_BASE_OFFSETS
        for offset in CONFLICT_OFFSET_CLASSES
        for transfer_bytes in CONFLICT_TRANSFER_BYTES
    ]


def validate_static_contract() -> None:
    cases = matrix_cases()
    conflict_cases = conflict_equivalence_cases()
    expected_cells = (
        RANK_COUNT
        * ALLOCATION_COUNT
        * len(OFFSET_CLASSES)
        * len(DIRECTIONS)
    )
    expected_conflict_cells = (
        RANK_COUNT
        * ALLOCATION_COUNT
        * len(CONFLICT_BASE_OFFSETS)
        * len(CONFLICT_OFFSET_CLASSES)
        * len(CONFLICT_TRANSFER_BYTES)
    )
    expected_conflict_rows_per_rank = (
        expected_conflict_cells
        // RANK_COUNT
        * len(CONFLICT_ISSUE_ORDERS)
        * len(CONFLICT_SCHEDULES)
        * CONFLICT_CELL_SAMPLES
    )
    small_pair_slot_bytes = 2 * (
        CONFLICT_TRANSFER_BYTES[0] + 2 * GUARD_BYTES
    )
    large_pair_slot_bytes = 2 * (
        CONFLICT_TRANSFER_BYTES[1] + 2 * GUARD_BYTES
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
        > CONFLICT_SMALL_ARCHIVE_BASE
        or MAX_RECORD_BYTES > ARCHIVE_BASE
        or LOCAL_ELEMENTS * ELEMENT_BYTES != RESOURCE_BYTES
        or len(conflict_cases) != expected_conflict_cells
        or CONFLICT_ROW_COUNT
        != expected_conflict_rows_per_rank
        or len(
            {
                (
                    row["rank"],
                    row["allocation_ordinal"],
                    row["base_offset"],
                    row["relative_offset_candidate"],
                    row["transfer_bytes"],
                )
                for row in conflict_cases
            }
        )
        != expected_conflict_cells
        or CONFLICT_BASE_OFFSETS[-1]
        + CONFLICT_PAIR_GAP
        + CONFLICT_OFFSET_CLASSES[-1]
        + CONFLICT_TRANSFER_BYTES[-1]
        > SWEEP_BASE
        or CONFLICT_SMALL_ARCHIVE_BASE
        + CONFLICT_ROWS_PER_TRANSFER_PER_ALLOCATION
        * small_pair_slot_bytes
        > SWEEP_BASE
        or CONFLICT_LARGE_ARCHIVE_BASE
        <= SWEEP_BASE + OFFSET_CLASSES[-1] + PAYLOAD_BYTES + GUARD_BYTES
        or CONFLICT_LARGE_ARCHIVE_BASE
        + CONFLICT_ROWS_PER_TRANSFER_PER_ALLOCATION
        * large_pair_slot_bytes
        > RESOURCE_BYTES
    ):
        raise RuntimeError("DDR tile/offset matrix has an invalid static contract")
    validate_mode_dispatch_contract()
    validate_coordinate_parser_contract()


def validate_mode_dispatch_contract() -> None:
    request_failures: list[str] = []
    for enabled, expected_mode, expected_conflict_fields in (
        (False, MODE_OFFSET, (0, 0, 0, 0, 0, 0)),
        (
            True,
            MODE_CONFLICT_EQUIVALENCE,
            (
                CONFLICT_CELL_SAMPLES,
                len(CONFLICT_BASE_OFFSETS),
                len(CONFLICT_OFFSET_CLASSES),
                len(CONFLICT_TRANSFER_BYTES),
                len(CONFLICT_ISSUE_ORDERS),
                len(CONFLICT_SCHEDULES),
            ),
        ),
    ):
        request, _ = make_input(0, 0, 0, enabled)
        words = struct.unpack_from(f"<{REQUEST_WORDS}Q", request)
        if (
            words[8] != expected_mode
            or words[9:15] != expected_conflict_fields
        ):
            request_failures.append(
                f"selector={enabled}: mode={words[8]} "
                f"conflict_fields={words[9:15]}"
            )

    source = PROBE_C.read_text()
    protocol = PROTOCOL_H.read_text()
    protocol_fragments = (
        f"#define WAFER_DDR_TILE_SCHEMA {SCHEMA}U",
        (
            "#define WAFER_DDR_TILE_CONFLICT_ROW_WORDS "
            f"{CONFLICT_ROW_WORDS}U"
        ),
        "WAFER_DDR_TILE_MODE_OFFSET = 0",
        "WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE = 1",
        "WAFER_DDR_TILE_REQ_MODE = 8",
        "WAFER_DDR_TILE_HDR_MODE = 13",
        "WAFER_DDR_TILE_HDR_CONFLICT_ROW_COUNT = 14",
        "WAFER_DDR_TILE_HDR_CONFLICT_RANK_ORDER = 19",
        "WAFER_DDR_TILE_CONFLICT_ROW_GUARD_WORD = 23",
        "WAFER_DDR_TILE_CONFLICT_ROW_SCHEDULE_POSITION = 24",
    )
    missing_protocol = [
        fragment for fragment in protocol_fragments if fragment not in protocol
    ]
    try:
        main_begin = source.index(
            "wafer_tx81_ddr_tile_offset_probe("
        )
        offset_begin = source.index(
            "if (mode == WAFER_DDR_TILE_MODE_OFFSET)", main_begin
        )
        conflict_begin = source.index(
            "else if (mode == "
            "WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE)",
            offset_begin,
        )
        dispatch_end = source.index(
            "wafer_ddr_tile_write_header(", conflict_begin
        )
    except ValueError as error:
        raise RuntimeError(
            "DDR tile/offset mode dispatch markers are incomplete"
        ) from error
    offset_branch = source[offset_begin:conflict_begin]
    conflict_branch = source[conflict_begin:dispatch_end]
    branch_failures = []
    if (
        "wafer_ddr_tile_measure_rdma(" not in offset_branch
        or "wafer_ddr_tile_measure_wdma(" not in offset_branch
        or "wafer_ddr_tile_measure_conflict_pair(" in offset_branch
    ):
        branch_failures.append("offset-mode-is-not-offset-only")
    if (
        "wafer_ddr_tile_measure_conflict_pair(" not in conflict_branch
        or "wafer_ddr_tile_measure_rdma(" in conflict_branch
        or "wafer_ddr_tile_measure_wdma(" in conflict_branch
        or "WAFER_DDR_TILE_RANKS - 1U - rank_position" not in source
        or "reverse_schedule" not in conflict_branch
    ):
        branch_failures.append("conflict-mode-is-not-conflict-only")
    if request_failures or branch_failures or missing_protocol:
        raise RuntimeError(
            "DDR tile/offset typed mode contract failed: "
            f"requests={request_failures}, branches={branch_failures}, "
            f"protocol={missing_protocol}"
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
        "dtype": "float16",
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
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="DDR tile/offset",
    )
    if (
        manifest.get("rank_count") != RANK_COUNT
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
    if module.get("exports") != runtime_launch.expected_kernel_module_exports(
        runtime_launch.CLUSTER_KERNEL_LAUNCH
    ):
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
            != {"dtype": "f16", "shape": [1, LOCAL_ELEMENTS]}
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
            f"--launch-kind={LAUNCH_KIND}",
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
    ):
        raise RuntimeError("DDR tile/offset no-card launch evidence is incomplete")
    print(
        "ddr_tile_conflict_equivalence_probe_no_card: passed"
        if args.conflict_equivalence
        else "ddr_tile_offset_probe_no_card: passed"
    )


def pattern_period(rank: int, allocation: int, launch_sample: int) -> bytes:
    return b"".join(
        struct.pack(
            "<e",
            float(
                (
                    rank * 37
                    + allocation * 83
                    + launch_sample * 29
                    + lane * 17
                    + (lane >> 4) * 11
                    + 5
                )
                % 31
                - 15
            ),
        )
        for lane in range(PAYLOAD_BYTES // ELEMENT_BYTES)
    )


def conflict_pattern(
    rank: int,
    allocation: int,
    launch_sample: int,
    address_offset: int,
    count: int,
) -> bytes:
    if address_offset % ELEMENT_BYTES or count % ELEMENT_BYTES:
        raise RuntimeError("f16 conflict pattern range is not element-aligned")
    return b"".join(
        struct.pack(
            "<e",
            float(
                (
                    rank * 37
                    + allocation * 83
                    + launch_sample * 29
                    + element * 17
                    + (element >> 7) * 11
                    + (element >> 15) * 7
                    + 5
                )
                % 63
                - 31
            ),
        )
        for element in range(
            address_offset // ELEMENT_BYTES,
            (address_offset + count) // ELEMENT_BYTES,
        )
    )


def make_input(
    rank: int,
    allocation: int,
    launch_sample: int,
    conflict_equivalence: bool,
) -> tuple[bytes, bytes]:
    payload = bytearray(RESOURCE_BYTES)
    canary_slot_bytes = (
        CONFLICT_TRANSFER_BYTES[-1] + 2 * GUARD_BYTES
        if conflict_equivalence
        else SLOT_BYTES
    )
    payload[
        CANARY_SOURCE_OFFSET :
        CANARY_SOURCE_OFFSET + canary_slot_bytes
    ] = bytes([CANARY]) * canary_slot_bytes
    if conflict_equivalence:
        for base_offset in CONFLICT_BASE_OFFSETS:
            for relative_offset in CONFLICT_OFFSET_CLASSES:
                for address_offset in (
                    base_offset,
                    base_offset + CONFLICT_PAIR_GAP + relative_offset,
                ):
                    payload[
                        address_offset :
                        address_offset + CONFLICT_TRANSFER_BYTES[-1]
                    ] = conflict_pattern(
                        rank,
                        allocation,
                        launch_sample,
                        address_offset,
                        CONFLICT_TRANSFER_BYTES[-1],
                    )
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
        words[8] = (
            MODE_CONFLICT_EQUIVALENCE
            if conflict_equivalence
            else MODE_OFFSET
        )
        if conflict_equivalence:
            words[9] = CONFLICT_CELL_SAMPLES
            words[10] = len(CONFLICT_BASE_OFFSETS)
            words[11] = len(CONFLICT_OFFSET_CLASSES)
            words[12] = len(CONFLICT_TRANSFER_BYTES)
            words[13] = len(CONFLICT_ISSUE_ORDERS)
            words[14] = len(CONFLICT_SCHEDULES)
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
    conflict_equivalence: bool,
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
            payload, expected = make_input(
                rank,
                allocation,
                launch_sample,
                conflict_equivalence,
            )
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
    if (
        len(matches) != RANK_COUNT
        or set(result) != set(range(RANK_COUNT))
        or len(set(result.values())) != RANK_COUNT
    ):
        raise RuntimeError(
            "board output did not provide a one-to-one mapping from all 16 "
            "logical ranks to 16 unique physical tiles"
        )
    return result


def require_stable_tile_coordinates(
    reference: dict[int, tuple[int, int]],
    observed: dict[int, tuple[int, int]],
) -> None:
    if observed != reference:
        raise RuntimeError(
            "logical-to-physical tile mapping changed across paired "
            "forward/reverse launches"
        )


def validate_coordinate_parser_contract() -> None:
    valid = "\n".join(
        f"board_tile: logical={rank} available=true "
        f"physical_x={rank // 4} physical_y={rank % 4}"
        for rank in range(RANK_COUNT)
    )
    expected = {
        rank: (rank // 4, rank % 4) for rank in range(RANK_COUNT)
    }
    if parse_tile_coordinates(valid) != expected:
        raise RuntimeError("valid tile-coordinate fixture was not preserved")
    duplicate_physical = valid.replace(
        "board_tile: logical=15 available=true physical_x=3 physical_y=3",
        "board_tile: logical=15 available=true physical_x=0 physical_y=0",
    )
    try:
        parse_tile_coordinates(duplicate_physical)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("duplicate physical tile was not rejected")
    changed = dict(expected)
    changed[15] = (4, 0)
    try:
        require_stable_tile_coordinates(expected, changed)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("cross-launch tile remapping was not rejected")


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


def conflict_local_ordinal(
    base_index: int,
    offset_index: int,
    issue_order_index: int,
    schedule_index: int,
    cell_sample: int,
) -> int:
    return (
        (
            (
                base_index * len(CONFLICT_OFFSET_CLASSES) + offset_index
            )
            * len(CONFLICT_ISSUE_ORDERS)
            + issue_order_index
        )
        * len(CONFLICT_SCHEDULES)
        + schedule_index
    ) * CONFLICT_CELL_SAMPLES + cell_sample


def conflict_row_ordinal(
    allocation: int,
    base_index: int,
    offset_index: int,
    transfer_index: int,
    issue_order_index: int,
    schedule_index: int,
    cell_sample: int,
) -> int:
    return (
        (
            (
                (
                    (
                        allocation * len(CONFLICT_BASE_OFFSETS) + base_index
                    )
                    * len(CONFLICT_OFFSET_CLASSES)
                    + offset_index
                )
                * len(CONFLICT_TRANSFER_BYTES)
                + transfer_index
            )
            * len(CONFLICT_ISSUE_ORDERS)
            + issue_order_index
        )
        * len(CONFLICT_SCHEDULES)
        + schedule_index
    ) * CONFLICT_CELL_SAMPLES + cell_sample


def conflict_archive_offset(
    transfer_index: int,
    base_index: int,
    offset_index: int,
    issue_order_index: int,
    schedule_index: int,
    cell_sample: int,
) -> tuple[int, int]:
    transfer_bytes = CONFLICT_TRANSFER_BYTES[transfer_index]
    guarded_slot_bytes = transfer_bytes + 2 * GUARD_BYTES
    archive_base = (
        CONFLICT_SMALL_ARCHIVE_BASE
        if transfer_index == 0
        else CONFLICT_LARGE_ARCHIVE_BASE
    )
    local_ordinal = conflict_local_ordinal(
        base_index,
        offset_index,
        issue_order_index,
        schedule_index,
        cell_sample,
    )
    archive_a = archive_base + local_ordinal * 2 * guarded_slot_bytes
    return archive_a, archive_a + guarded_slot_bytes


def validate_rank_outputs(
    rank: int,
    launch_sample: int,
    paths: dict[tuple[int, int], pathlib.Path],
    expected_payloads: dict[tuple[int, int], bytes],
    physical: tuple[int, int],
    conflict_equivalence: bool,
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
        5: 0 if conflict_equivalence else ROW_COUNT,
        6: PAYLOAD_BYTES,
        7: GUARD_BYTES,
        12: REQUEST_GUARD,
        13: (
            MODE_CONFLICT_EQUIVALENCE
            if conflict_equivalence
            else MODE_OFFSET
        ),
        14: CONFLICT_ROW_COUNT if conflict_equivalence else 0,
        15: CONFLICT_CELL_SAMPLES if conflict_equivalence else 0,
        16: len(CONFLICT_BASE_OFFSETS) if conflict_equivalence else 0,
        17: len(CONFLICT_OFFSET_CLASSES) if conflict_equivalence else 0,
        18: len(CONFLICT_TRANSFER_BYTES) if conflict_equivalence else 0,
        19: (
            2
            if conflict_equivalence and launch_sample % 2
            else 1
            if conflict_equivalence
            else 0
        ),
        31: RECORD_GUARD,
    }
    failures = {
        index: (header[index], expected)
        for index, expected in expected_header.items()
        if header[index] != expected
    }
    if failures:
        raise RuntimeError(f"rank {rank}: header oracle failed: {failures}")
    rank_execution_order = (
        "reverse"
        if header[19] == 2
        else "forward"
        if header[19] == 1
        else "not-applicable"
    )
    rank_execution_position = (
        RANK_COUNT - 1 - rank
        if rank_execution_order == "reverse"
        else rank
    )
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
    for allocation in (
        () if conflict_equivalence else range(ALLOCATION_COUNT)
    ):
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
                            "kind": "offset-latency",
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

    for allocation in (
        range(ALLOCATION_COUNT) if conflict_equivalence else ()
    ):
        for base_index, base_offset in enumerate(CONFLICT_BASE_OFFSETS):
            for offset_index, relative_offset in enumerate(
                CONFLICT_OFFSET_CLASSES
            ):
                address_a = input_bases[allocation] + base_offset
                address_b = (
                    address_a + CONFLICT_PAIR_GAP + relative_offset
                )
                for transfer_index, transfer_bytes in enumerate(
                    CONFLICT_TRANSFER_BYTES
                ):
                    expected_a = conflict_pattern(
                        rank,
                        allocation,
                        launch_sample,
                        base_offset,
                        transfer_bytes,
                    )
                    expected_b = conflict_pattern(
                        rank,
                        allocation,
                        launch_sample,
                        base_offset
                        + CONFLICT_PAIR_GAP
                        + relative_offset,
                        transfer_bytes,
                    )
                    guard = bytes([CANARY]) * GUARD_BYTES
                    expected_slot_a = guard + expected_a + guard
                    expected_slot_b = guard + expected_b + guard
                    for issue_order_index, issue_order in enumerate(
                        CONFLICT_ISSUE_ORDERS
                    ):
                        for schedule_index, schedule in enumerate(
                            CONFLICT_SCHEDULES
                        ):
                            for cell_sample in range(
                                CONFLICT_CELL_SAMPLES
                            ):
                                ordinal = conflict_row_ordinal(
                                    allocation,
                                    base_index,
                                    offset_index,
                                    transfer_index,
                                    issue_order_index,
                                    schedule_index,
                                    cell_sample,
                                )
                                row_offset = (
                                    HEADER_WORDS * 8
                                    + ordinal * CONFLICT_ROW_WORDS * 8
                                )
                                row = struct.unpack_from(
                                    f"<{CONFLICT_ROW_WORDS}Q",
                                    raw_outputs[0],
                                    row_offset,
                                )
                                archive_a, archive_b = (
                                    conflict_archive_offset(
                                        transfer_index,
                                        base_index,
                                        offset_index,
                                        issue_order_index,
                                        schedule_index,
                                        cell_sample,
                                    )
                                )
                                identity = (
                                    rank
                                    | allocation << 8
                                    | base_index << 16
                                    | offset_index << 24
                                    | transfer_index << 32
                                    | issue_order_index << 40
                                    | schedule_index << 48
                                    | cell_sample << 56
                                )
                                expected_row = {
                                    0: CONFLICT_ROW_MAGIC,
                                    1: identity,
                                    2: relative_offset,
                                    3: input_bases[allocation],
                                    4: output_bases[allocation],
                                    5: address_a,
                                    6: address_b,
                                    7: CONFLICT_SPM_A + GUARD_BYTES,
                                    8: CONFLICT_SPM_B + GUARD_BYTES,
                                    9: transfer_bytes,
                                    10: archive_a,
                                    11: archive_b,
                                    13: 2,
                                    18: base_offset,
                                    19: allocation,
                                    20: schedule_index + 1,
                                    21: issue_order_index + 1,
                                    22: cell_sample,
                                    23: CONFLICT_ROW_GUARD,
                                    24: (
                                        schedule_index
                                        if (cell_sample + launch_sample) % 2
                                        == 0
                                        else 1 - schedule_index
                                    ),
                                }
                                row_failures = {
                                    index: (row[index], expected)
                                    for index, expected
                                    in expected_row.items()
                                    if row[index] != expected
                                }
                                if (
                                    row_failures
                                    or row[12] == 0
                                    or row[15] == 0
                                ):
                                    raise RuntimeError(
                                        f"rank {rank} allocation "
                                        f"{allocation} conflict base "
                                        f"{base_offset} offset "
                                        f"{relative_offset} bytes "
                                        f"{transfer_bytes} {issue_order} "
                                        f"{schedule} sample {cell_sample}: "
                                        "row oracle failed "
                                        f"{row_failures}, "
                                        f"completion={row[12]}, "
                                        f"engine={row[15]}"
                                    )
                                actual_slot_a = raw_outputs[allocation][
                                    archive_a :
                                    archive_a + len(expected_slot_a)
                                ]
                                actual_slot_b = raw_outputs[allocation][
                                    archive_b :
                                    archive_b + len(expected_slot_b)
                                ]
                                if (
                                    actual_slot_a != expected_slot_a
                                    or actual_slot_b != expected_slot_b
                                ):
                                    raise RuntimeError(
                                        f"rank {rank} allocation "
                                        f"{allocation} conflict base "
                                        f"{base_offset} offset "
                                        f"{relative_offset} bytes "
                                        f"{transfer_bytes} {issue_order} "
                                        f"{schedule} sample {cell_sample}: "
                                        "paired payload/guard archive is "
                                        "not exact"
                                    )
                                observations.append(
                                    {
                                        "kind": "conflict-equivalence",
                                        "rank": rank,
                                        "physical_x": physical[0],
                                        "physical_y": physical[1],
                                        "launch_sample": launch_sample,
                                        "allocation_ordinal": allocation,
                                        "input_base": input_bases[allocation],
                                        "output_base": output_bases[allocation],
                                        "base_offset": base_offset,
                                        "relative_offset_candidate": (
                                            relative_offset
                                        ),
                                        "address_delta": (
                                            CONFLICT_PAIR_GAP
                                            + relative_offset
                                        ),
                                        "address_a": address_a,
                                        "address_b": address_b,
                                        "transfer_bytes": transfer_bytes,
                                        "issue_order": issue_order,
                                        "schedule": schedule,
                                        "cell_sample": cell_sample,
                                        "schedule_position": row[24],
                                        "rank_execution_order": (
                                            rank_execution_order
                                        ),
                                        "rank_execution_position": (
                                            rank_execution_position
                                        ),
                                        "instruction_delta": row[13],
                                        "blocking_delta": row[14],
                                        "engine_execution_delta": row[15],
                                        "fu_execution_delta": row[16],
                                        "statistics_window_delta": row[17],
                                        "issue_to_matching_completion_cycles": (
                                            row[12]
                                        ),
                                        "pmu_interval": (
                                            "target-two-rdma-pair-only"
                                        ),
                                        "controller_class": "unclassified",
                                        "correctness": (
                                            "two-exact-payloads-with-"
                                            "independent-prefix-suffix-guards"
                                        ),
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
        for offset_index, offset in enumerate(
            () if conflict_equivalence else OFFSET_CLASSES
        ):
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
        for base_index, base_offset in enumerate(
            CONFLICT_BASE_OFFSETS if conflict_equivalence else ()
        ):
            for offset_index, relative_offset in enumerate(
                CONFLICT_OFFSET_CLASSES
            ):
                for transfer_index, transfer_bytes in enumerate(
                    CONFLICT_TRANSFER_BYTES
                ):
                    guard = bytes([CANARY]) * GUARD_BYTES
                    expected_a = guard + conflict_pattern(
                        rank,
                        allocation,
                        launch_sample,
                        base_offset,
                        transfer_bytes,
                    ) + guard
                    expected_b = guard + conflict_pattern(
                        rank,
                        allocation,
                        launch_sample,
                        base_offset
                        + CONFLICT_PAIR_GAP
                        + relative_offset,
                        transfer_bytes,
                    ) + guard
                    for issue_order_index in range(
                        len(CONFLICT_ISSUE_ORDERS)
                    ):
                        for schedule_index in range(
                            len(CONFLICT_SCHEDULES)
                        ):
                            for cell_sample in range(
                                CONFLICT_CELL_SAMPLES
                            ):
                                archive_a, archive_b = (
                                    conflict_archive_offset(
                                        transfer_index,
                                        base_index,
                                        offset_index,
                                        issue_order_index,
                                        schedule_index,
                                        cell_sample,
                                    )
                                )
                                expected[
                                    archive_a : archive_a + len(expected_a)
                                ] = expected_a
                                expected[
                                    archive_b : archive_b + len(expected_b)
                                ] = expected_b
        if allocation == 0:
            record_bytes = (
                CONFLICT_RECORD_BYTES
                if conflict_equivalence
                else OFFSET_RECORD_BYTES
            )
            expected[:record_bytes] = raw_outputs[0][:record_bytes]
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
                        if row["kind"] == "offset-latency"
                        and row["rank"] == rank
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


def summarize_conflicts(
    observations: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    metrics = (
        "engine_execution_delta",
        "issue_to_matching_completion_cycles",
        "blocking_delta",
        "fu_execution_delta",
        "statistics_window_delta",
    )
    for rank in range(RANK_COUNT):
        for allocation in range(ALLOCATION_COUNT):
            for base_offset in CONFLICT_BASE_OFFSETS:
                for relative_offset in CONFLICT_OFFSET_CLASSES:
                    for transfer_bytes in CONFLICT_TRANSFER_BYTES:
                        rows = [
                            row
                            for row in observations
                            if row["kind"] == "conflict-equivalence"
                            and row["rank"] == rank
                            and row["allocation_ordinal"] == allocation
                            and row["base_offset"] == base_offset
                            and row["relative_offset_candidate"]
                            == relative_offset
                            and row["transfer_bytes"] == transfer_bytes
                        ]
                        expected_rows = (
                            len(CONFLICT_ISSUE_ORDERS)
                            * len(CONFLICT_SCHEDULES)
                            * CONFLICT_CELL_SAMPLES
                        )
                        if len(rows) != expected_rows:
                            raise RuntimeError(
                                "DDR conflict-equivalence cell is incomplete"
                            )
                        controls: dict[
                            str, dict[str, dict[str, int | float]]
                        ] = {}
                        differences: dict[
                            str, dict[str, int | float]
                        ] = {}
                        for issue_order in CONFLICT_ISSUE_ORDERS:
                            controls[issue_order] = {}
                            for schedule in CONFLICT_SCHEDULES:
                                samples = [
                                    row
                                    for row in rows
                                    if row["issue_order"] == issue_order
                                    and row["schedule"] == schedule
                                ]
                                if len(samples) != CONFLICT_CELL_SAMPLES:
                                    raise RuntimeError(
                                        "DDR conflict schedule/order control "
                                        "is incomplete"
                                    )
                                controls[issue_order][schedule] = {
                                    metric: statistics.median(
                                        int(row[metric]) for row in samples
                                    )
                                    for metric in metrics
                                }
                            differences[issue_order] = {
                                metric: (
                                    controls[issue_order]["window"][metric]
                                    - controls[issue_order]["serial"][metric]
                                )
                                for metric in metrics
                            }
                        summaries.append(
                            {
                                "rank": rank,
                                "physical_x": rows[0]["physical_x"],
                                "physical_y": rows[0]["physical_y"],
                                "allocation_ordinal": allocation,
                                "actual_input_base": rows[0]["input_base"],
                                "base_offset": base_offset,
                                "relative_offset_candidate": relative_offset,
                                "address_delta": (
                                    CONFLICT_PAIR_GAP + relative_offset
                                ),
                                "address_a": rows[0]["address_a"],
                                "address_b": rows[0]["address_b"],
                                "transfer_bytes": transfer_bytes,
                                "samples_per_order_schedule": (
                                    CONFLICT_CELL_SAMPLES
                                ),
                                "controls": controls,
                                "window_minus_serial_by_issue_order": (
                                    differences
                                ),
                                "reciprocal_window_minus_serial_difference": {
                                    metric: (
                                        differences["a-b"][metric]
                                        - differences["b-a"][metric]
                                    )
                                    for metric in metrics
                                },
                                "correctness": (
                                    "all-two-payload-and-four-guard-"
                                    "archives-exact"
                                ),
                                "pmu_interval": "target-two-rdma-pair-only",
                                "controller_class": "unclassified",
                                "interpretation": (
                                    "same-invocation paired schedule/order "
                                    "observation across actual allocation base "
                                    "and physical tile; no DDR bank identity "
                                    "is inferred"
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
    if getattr(args, "conflict_equivalence", False) and (
        args.repeat < 2 or args.repeat % 2 != 0
    ):
        raise RuntimeError(
            "DDR conflict-equivalence requires an even --repeat of at least "
            "2 so every tile is observed under forward and reverse rank order"
        )
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
    reference_coordinates: dict[int, tuple[int, int]] | None = None
    for launch_sample in range(args.repeat):
        resource_args, paths, expected_payloads = write_resources(
            args.work_dir,
            bindings,
            launch_sample,
            args.conflict_equivalence,
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
            "board_stage: completion",
            "board_stage: device-to-host",
            "board_stage: cleanup",
        }
        if not required.issubset(set(result.stdout.splitlines())):
            raise RuntimeError("DDR tile/offset board evidence is incomplete")
        coordinates = parse_tile_coordinates(result.stdout)
        if reference_coordinates is None:
            reference_coordinates = coordinates
        else:
            require_stable_tile_coordinates(
                reference_coordinates, coordinates
            )
        launch_observations: list[dict[str, object]] = []
        for rank in range(RANK_COUNT):
            launch_observations.extend(
                validate_rank_outputs(
                    rank,
                    launch_sample,
                    paths,
                    expected_payloads,
                    coordinates[rank],
                    args.conflict_equivalence,
                )
            )
        all_observations.extend(launch_observations)
        print(
            (
                "ddr_tile_conflict_equivalence_raw_observations: "
                if args.conflict_equivalence
                else "ddr_tile_offset_raw_observations: "
            )
            + json.dumps(launch_observations, sort_keys=True)
        )
        if args.conflict_equivalence:
            print(
                "ddr_tile_conflict_equivalence_summaries: "
                + json.dumps(
                    summarize_conflicts(launch_observations), sort_keys=True
                )
            )
        else:
            print(
                "ddr_tile_offset_cell_summaries: "
                + json.dumps(summarize(launch_observations), sort_keys=True)
            )
        print(result.stdout, end="")
    if args.conflict_equivalence:
        rows_per_coordinate_per_launch = (
            len(CONFLICT_ISSUE_ORDERS)
            * len(CONFLICT_SCHEDULES)
            * CONFLICT_CELL_SAMPLES
        )
        expected_per_rank_order = (
            args.repeat // 2 * rows_per_coordinate_per_launch
        )
        counterbalanced_coordinates = 0
        for rank in range(RANK_COUNT):
            for allocation in range(ALLOCATION_COUNT):
                for base_offset in CONFLICT_BASE_OFFSETS:
                    for relative_offset in CONFLICT_OFFSET_CLASSES:
                        for transfer_bytes in CONFLICT_TRANSFER_BYTES:
                            rows = [
                                row
                                for row in all_observations
                                if row["rank"] == rank
                                and row["allocation_ordinal"] == allocation
                                and row["base_offset"] == base_offset
                                and row["relative_offset_candidate"]
                                == relative_offset
                                and row["transfer_bytes"] == transfer_bytes
                            ]
                            counts = {
                                order: sum(
                                    row["rank_execution_order"] == order
                                    for row in rows
                                )
                                for order in ("forward", "reverse")
                            }
                            if counts != {
                                "forward": expected_per_rank_order,
                                "reverse": expected_per_rank_order,
                            }:
                                raise RuntimeError(
                                    "DDR conflict rank-order controls are "
                                    f"incomplete for rank {rank}, allocation "
                                    f"{allocation}, base {base_offset}, "
                                    f"offset {relative_offset}, transfer "
                                    f"{transfer_bytes}: {counts}"
                                )
                            counterbalanced_coordinates += 1
        print(
            "ddr_tile_conflict_counterbalance: "
            + json.dumps(
                {
                    "coordinates": counterbalanced_coordinates,
                    "launches": args.repeat,
                    "rank_orders": ["forward", "reverse"],
                    "physical_tile_mapping": (
                        "unique-and-stable-across-launches"
                    ),
                    "schedule_position": (
                        "serial/window order alternates by cell sample and "
                        "launch parity"
                    ),
                    "state": "counterbalanced-raw-equivalence-input",
                    "compiler_use": "no-ddr-bank-coloring",
                },
                sort_keys=True,
            )
        )


def main() -> int:
    args = parse_args()
    validate_static_contract()
    if args.list_cases:
        print(
            json.dumps(
                {
                    "default_mode": "offset-latency",
                    "pending_mode_selector": "--conflict-equivalence",
                    "matrix": matrix_cases(),
                    "conflict_equivalence_matrix": (
                        conflict_equivalence_cases()
                    ),
                    "allocation_contract": (
                        "each rank owns two independent input and two "
                        "independent output resources; cross-rank resources "
                        "are distinct allocations"
                    ),
                    "bank_contract": (
                        "offset classes, paired actual 64-bit addresses, "
                        "allocation ordinal and physical tile are observed; "
                        "serial/window and A-B/B-A controls share one device "
                        "invocation, but no physical bank/controller/hop "
                        "mapping is assumed"
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
