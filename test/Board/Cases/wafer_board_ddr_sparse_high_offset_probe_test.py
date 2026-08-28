#!/usr/bin/env python3
"""Build and run a sparse high-offset probe in one large DDR workspace."""

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

import wafer_runtime_launch_contract as runtime_launch


TARGET_IDENTITY = "wafer-tx81-single-card"
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ddr_sparse_high_offset_probe.c"
PROBE_LL = INPUT_DIR / "wafer_ddr_sparse_high_offset_probe.ll"

RESOURCE_BYTES = 0x2000
ELEMENT_BYTES = 2
LOCAL_ELEMENTS = RESOURCE_BYTES // ELEMENT_BYTES
DEFAULT_WORKSPACE_GIB = 40
MIN_WORKSPACE_GIB = 36
MAX_WORKSPACE_GIB = 40
PAYLOAD_BYTES = 256
GUARD_BYTES = 256
SLOT_BYTES = PAYLOAD_BYTES + 2 * GUARD_BYTES
INPUT_BASE = 0x200
ARCHIVE_BASE = 0x400
OUTPUT_INITIAL = 0xA5
OFFSET_NAMES = (
    "base",
    "4-gib",
    "16-gib",
    "32-gib",
    "near-allocation-end",
    "allocation-end",
)

REQUEST_MAGIC = 0x5744534852455154
RECORD_MAGIC = 0x5744534852454344
ROW_MAGIC = 0x57445348524F5721
REQUEST_GUARD = 0x9B52D6407CE183AF
RECORD_GUARD = 0xC8642F9A15BD703E
ROW_GUARD = 0x73E10AB49D5268CF
REQUEST_WORDS = 16
HEADER_WORDS = 16
ROW_WORDS = 8
RECORD_BYTES = (HEADER_WORDS + len(OFFSET_NAMES) * ROW_WORDS) * 8

MODULE = f"""\
module {{
  func.func @main(%arg0: tensor<{LOCAL_ELEMENTS}xf16>)
      -> tensor<{LOCAL_ELEMENTS}xf16> {{
    %result = stablehlo.add %arg0, %arg0 : tensor<{LOCAL_ELEMENTS}xf16>
    return %result : tensor<{LOCAL_ELEMENTS}xf16>
  }}
}}
"""

METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {
            "shape": [LOCAL_ELEMENTS],
            "dtype": "float16",
            "dynamic_dims": [],
        }
    ],
    "output_signature": [
        {
            "shape": [LOCAL_ELEMENTS],
            "dtype": "float16",
            "dynamic_dims": [],
        }
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request_and_windows"}
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
    parser.add_argument(
        "--workspace-gib",
        type=int,
        default=DEFAULT_WORKSPACE_GIB,
        help=(
            "compiler-managed workspace size in GiB; constrained below the "
            "current profile's approximately 44-GiB allocator window"
        ),
    )
    return parser.parse_args()


def workspace_bytes(workspace_gib: int) -> int:
    if not MIN_WORKSPACE_GIB <= workspace_gib <= MAX_WORKSPACE_GIB:
        raise RuntimeError(
            f"--workspace-gib must be in [{MIN_WORKSPACE_GIB}, "
            f"{MAX_WORKSPACE_GIB}]"
        )
    return workspace_gib * 1024**3


def offsets_for(workspace_size: int) -> tuple[int, ...]:
    return (
        0,
        4 * 1024**3,
        16 * 1024**3,
        32 * 1024**3,
        workspace_size - 2 * 1024**3,
        workspace_size - SLOT_BYTES,
    )


def probe_cases(workspace_size: int) -> list[dict[str, object]]:
    offsets = offsets_for(workspace_size)
    return [
        {
            "name": name,
            "workspace_bytes": workspace_size,
            "relative_offset": offset,
            "bytes": SLOT_BYTES,
            "oracle": "exact-roundtrip-with-prefix-and-suffix-guard",
            "address_scope": "compiler-managed-workspace-relative",
            "physical_bank": "unclassified",
        }
        for name, offset in zip(OFFSET_NAMES, offsets, strict=True)
    ]


def validate_static_contract(workspace_size: int) -> None:
    offsets = offsets_for(workspace_size)
    if (
        workspace_size % 256 != 0
        or RESOURCE_BYTES != 8192
        or SLOT_BYTES != 768
        or len(offsets) != 6
        or len(set(offsets)) != len(offsets)
        or offsets != tuple(sorted(offsets))
        or any(offset % 256 for offset in offsets)
        or any(
            left + SLOT_BYTES > right
            for left, right in zip(offsets, offsets[1:])
        )
        or offsets[-1] + SLOT_BYTES != workspace_size
        or not any(offset > 2**32 for offset in offsets)
        or INPUT_BASE + len(offsets) * SLOT_BYTES > RESOURCE_BYTES
        or ARCHIVE_BASE + len(offsets) * SLOT_BYTES > RESOURCE_BYTES
        or RECORD_BYTES > ARCHIVE_BASE
        or LOCAL_ELEMENTS * ELEMENT_BYTES != RESOURCE_BYTES
    ):
        raise RuntimeError("sparse high-offset DDR contract is inconsistent")
    required = (PROBE_C, PROBE_LL)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"sparse high-offset probe files are missing: {missing}")
    symbol = "wafer_tx81_ddr_sparse_high_offset_probe"
    if symbol not in PROBE_C.read_text() or symbol not in PROBE_LL.read_text():
        raise RuntimeError("sparse high-offset device symbol is inconsistent")


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
            "sparse high-offset DDR probe timed out; no retry, reset, or "
            "power operation was attempted"
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
        raise RuntimeError("sparse high-offset work directory is too broad")


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
    if args.repeat <= 0:
        raise RuntimeError("--repeat must be positive")


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
            "--output-dir",
            str(package),
            "--num-partitions=1",
        ],
        timeout_seconds=300,
    )
    if "wrote verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not write the seed package")
    return package


def prepare_workspace_manifest(
    package: pathlib.Path, workspace_size: int,
) -> tuple[pathlib.Path, tuple[int, int], int]:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries")
    modules = manifest.get("modules")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    target = manifest.get("target")
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.GRID_KERNEL_LAUNCH,
        context="sparse high-offset seed",
    )
    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="sparse high-offset seed"
    )
    if (
        not isinstance(target, dict)
        or target.get("identity") != TARGET_IDENTITY
        or not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or len(inputs) != 1
        or len(outputs) != 1
    ):
        raise RuntimeError("sparse high-offset seed manifest is invalid")

    entry = entries[0]
    module = modules[0]
    if (
        not isinstance(entry, dict)
        or entry.get("card_id") != 0
        or entry.get("tile_id") != 0
        or entry.get("module") != module.get("id")
        or module.get("exports") != [{"role": "main", "symbol": "main"}]
    ):
        raise RuntimeError("sparse high-offset seed entry is invalid")

    expected_host = (
        (inputs[0], "user_input"),
        (outputs[0], "output"),
    )
    for record, role in expected_host:
        if (
            not isinstance(record, dict)
            or record.get("role_index") != 0
            or record.get("dtype") != "f16"
            or record.get("shape") != [LOCAL_ELEMENTS]
            or record.get("bytes") != RESOURCE_BYTES
            or record.get("alignment") != 256
            or not isinstance(record.get("id"), int)
        ):
            raise RuntimeError(
                f"sparse high-offset host port is invalid: {record}"
            )

    input_id = int(inputs[0]["id"])
    output_id = int(outputs[0]["id"])
    workspace_arguments = [
        {
            "ordinal": 2,
            "kind": "workspace",
            "bytes": workspace_size,
            "alignment": 256,
            "access": "read_write",
        }
    ]
    for tile_entry in entries:
        tile_entry["arguments"] = [
            {
                "ordinal": 0,
                "kind": "external_input",
                "port": input_id,
                "access": "read_only",
            },
            {
                "ordinal": 1,
                "kind": "external_output",
                "port": output_id,
                "access": "write_only",
            },
            *workspace_arguments,
        ]

    module_path_value = module.get("path")
    if not isinstance(module_path_value, str):
        raise RuntimeError("sparse high-offset seed module path is missing")
    module_path = package / module_path_value
    if not module_path.is_file():
        raise RuntimeError("sparse high-offset seed module is missing")

    staged = manifest_path.with_name(".manifest.json.ddr-sparse-high-offset")
    staged.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged, manifest_path)
    validate_manifest(package, (input_id, output_id), workspace_size)
    return module_path, (input_id, output_id), len(entry["arguments"])


def validate_manifest(
    package: pathlib.Path,
    resource_ids: tuple[int, int],
    workspace_size: int,
) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    tile_entries = runtime_launch.require_complete_tile_domain(
        manifest, context="sparse high-offset package"
    )
    input_id, output_id = resource_ids
    expected_arguments = [
        {
            "ordinal": 0,
            "kind": "external_input",
            "port": input_id,
            "access": "read_only",
            "bytes": None,
            "alignment": None,
        },
        {
            "ordinal": 1,
            "kind": "external_output",
            "port": output_id,
            "access": "write_only",
            "bytes": None,
            "alignment": None,
        },
        {
            "ordinal": 2,
            "kind": "workspace",
            "port": None,
            "access": "read_write",
            "bytes": workspace_size,
            "alignment": 256,
        },
    ]
    for tile_entry in tile_entries:
        arguments = tile_entry.get("arguments")
        actual_arguments = [
            {
                "ordinal": argument.get("ordinal"),
                "kind": argument.get("kind"),
                "port": argument.get("port"),
                "access": argument.get("access"),
                "bytes": argument.get("bytes"),
                "alignment": argument.get("alignment"),
            }
            for argument in arguments
            if isinstance(argument, dict)
        ] if isinstance(arguments, list) else []
        if actual_arguments != expected_arguments:
            raise RuntimeError(
                "sparse high-offset ABI arguments are not canonical"
            )


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
            f"sparse high-offset build dependencies are missing: {missing}"
        )

    build = args.work_dir / "probe-build"
    build.mkdir()
    helper = build / "wafer_ddr_sparse_high_offset_probe.o"
    linked = build / "wafer_ddr_sparse_high_offset_probe.so"
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
            f"-DWAFER_DDR_SPARSE_SLOTS_PER_TILE={slots_per_tile}",
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

    staged_module = module_path.with_name(
        f".{module_path.name}.ddr-sparse-high-offset"
    )
    shutil.copy2(linked, staged_module)
    os.replace(staged_module, module_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    module_id = manifest["entries"][0]["module"]
    matching = [
        module for module in manifest["modules"] if module.get("id") == module_id
    ]
    if len(matching) != 1:
        raise RuntimeError("sparse high-offset module record is ambiguous")
    matching[0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(
        ".manifest.json.ddr-sparse-high-offset-linked"
    )
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged_manifest, manifest_path)


def verify_no_card(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int],
    workspace_size: int,
) -> None:
    result = run(
        [
            str(args.wafer_run),
            "--package-dir",
            str(package),
            "--no-card",
        ]
    )
    workspace_plan = re.compile(
        r"^launch_argument: 2 base=invocation offset=\d+$"
    )
    if (
        "board_execution: false" not in result.stdout
        or workspace_plan.search(result.stdout) is None
    ):
        raise RuntimeError(
            "sparse high-offset no-card launch evidence is incomplete"
        )
    print("ddr_sparse_high_offset_probe_no_card: passed")


def pattern_bytes(sample: int, index: int, domain: int) -> bytes:
    return b"".join(
        struct.pack(
            "<e",
            float(
                (
                    sample * 29
                    + index * 47
                    + domain * 71
                    + lane * 17
                    + (lane >> 4) * 11
                    + 5
                )
                % 63
                - 31
            ),
        )
        for lane in range(PAYLOAD_BYTES // ELEMENT_BYTES)
    )


def make_input(
    sample: int, workspace_size: int
) -> tuple[bytes, tuple[bytes, ...]]:
    raw = bytearray(RESOURCE_BYTES)
    words = [0] * REQUEST_WORDS
    words[0] = REQUEST_MAGIC
    words[1] = REQUEST_WORDS
    words[2] = sample
    words[3] = workspace_size
    words[4] = RESOURCE_BYTES
    words[5] = PAYLOAD_BYTES
    words[6] = GUARD_BYTES
    words[7] = len(OFFSET_NAMES)
    words[15] = REQUEST_GUARD
    raw[: REQUEST_WORDS * 8] = struct.pack(f"<{REQUEST_WORDS}Q", *words)

    slots: list[bytes] = []
    for index in range(len(OFFSET_NAMES)):
        slot = (
            pattern_bytes(sample, index, 1)
            + pattern_bytes(sample, index, 2)
            + pattern_bytes(sample, index, 3)
        )
        begin = INPUT_BASE + index * SLOT_BYTES
        raw[begin : begin + SLOT_BYTES] = slot
        slots.append(slot)
    return bytes(raw), tuple(slots)


def board_command(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int],
    input_path: pathlib.Path,
    output_path: pathlib.Path,
) -> list[str]:
    input_id, output_id = resource_ids
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
        f"{input_id}={input_path}",
        "--output",
        f"{output_id}={output_path}",
    ]


def validate_output(
    output_path: pathlib.Path,
    expected_slots: tuple[bytes, ...],
    sample: int,
    workspace_size: int,
) -> dict[str, object]:
    raw = output_path.read_bytes()
    if len(raw) != RESOURCE_BYTES:
        raise RuntimeError(
            f"sparse high-offset output has {len(raw)} bytes, expected "
            f"{RESOURCE_BYTES}"
        )
    header = struct.unpack_from(f"<{HEADER_WORDS}Q", raw)
    expected_header = {
        0: RECORD_MAGIC,
        1: HEADER_WORDS,
        2: 0,
        3: sample,
        4: len(OFFSET_NAMES),
        6: workspace_size,
        9: REQUEST_GUARD,
        10: PAYLOAD_BYTES,
        11: GUARD_BYTES,
        12: SLOT_BYTES,
        15: RECORD_GUARD,
    }
    failures = {
        index: (header[index], expected)
        for index, expected in expected_header.items()
        if header[index] != expected
    }
    if failures:
        raise RuntimeError(
            f"sparse high-offset header oracle failed: {failures}"
        )
    workspace_base = header[5]
    input_base = header[7]
    output_base = header[8]
    if (
        any(base == 0 or base % 256 for base in (workspace_base, input_base, output_base))
        or len({workspace_base, input_base, output_base}) != 3
        or workspace_base > (2**64 - 1) - workspace_size
    ):
        raise RuntimeError(
            "sparse high-offset runtime bindings are null, aliased, "
            "misaligned, or overflowing"
        )

    rows: list[dict[str, object]] = []
    offsets = offsets_for(workspace_size)
    for index, relative_offset in enumerate(offsets):
        row_offset = (HEADER_WORDS + index * ROW_WORDS) * 8
        row = struct.unpack_from(f"<{ROW_WORDS}Q", raw, row_offset)
        input_offset = INPUT_BASE + index * SLOT_BYTES
        archive_offset = ARCHIVE_BASE + index * SLOT_BYTES
        expected_row = (
            ROW_MAGIC,
            index,
            relative_offset,
            workspace_base + relative_offset,
            input_offset,
            archive_offset,
            SLOT_BYTES,
            ROW_GUARD,
        )
        if row != expected_row:
            raise RuntimeError(
                f"sparse high-offset row {index} differs: "
                f"actual={row}, expected={expected_row}"
            )
        actual_slot = raw[archive_offset : archive_offset + SLOT_BYTES]
        if actual_slot != expected_slots[index]:
            mismatch = next(
                lane
                for lane, (actual, expected) in enumerate(
                    zip(actual_slot, expected_slots[index], strict=True)
                )
                if actual != expected
            )
            raise RuntimeError(
                f"sparse high-offset window {index} differs at byte {mismatch}"
            )
        rows.append(
            {
                "name": OFFSET_NAMES[index],
                "relative_offset": relative_offset,
                "runtime_workspace_address": row[3],
                "bytes": SLOT_BYTES,
                "sha256": hashlib.sha256(actual_slot).hexdigest(),
                "oracle": "exact-roundtrip-with-prefix-and-suffix-guard",
                "address_scope": "compiler-managed-workspace-relative",
                "physical_bank": "unclassified",
            }
        )

    mutable = bytearray(raw)
    mutable[:RECORD_BYTES] = bytes([OUTPUT_INITIAL]) * RECORD_BYTES
    mutable[
        ARCHIVE_BASE : ARCHIVE_BASE + len(offsets) * SLOT_BYTES
    ] = bytes([OUTPUT_INITIAL]) * (len(offsets) * SLOT_BYTES)
    if mutable != bytes([OUTPUT_INITIAL]) * RESOURCE_BYTES:
        mismatch = next(
            index
            for index, value in enumerate(mutable)
            if value != OUTPUT_INITIAL
        )
        raise RuntimeError(
            "sparse high-offset output changed outside record/archive at "
            f"byte {mismatch}"
        )
    return {
        "sample": sample,
        "workspace_bytes": workspace_size,
        "runtime_workspace_base": workspace_base,
        "address_scope": "compiler-managed-workspace-relative",
        "physical_bank": "unclassified",
        "windows": rows,
    }


def execute_board(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int],
    workspace_size: int,
) -> None:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    observations: list[dict[str, object]] = []
    for sample in range(args.repeat):
        input_path = raw_dir / f"sample-{sample}.input.raw"
        output_path = raw_dir / f"sample-{sample}.output.raw"
        input_bytes, expected_slots = make_input(sample, workspace_size)
        input_path.write_bytes(input_bytes)
        result = run(
            board_command(
                args, package, resource_ids, input_path, output_path
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
                "sparse high-offset board run omitted lifecycle evidence"
            )
        observation = validate_output(
            output_path, expected_slots, sample, workspace_size
        )
        observations.append(observation)
        print(json.dumps(observation, sort_keys=True))
    (args.work_dir / "observations.json").write_text(
        json.dumps(observations, indent=2, sort_keys=True) + "\n"
    )
    print("ddr_sparse_high_offset_probe_board: passed")


def main() -> int:
    args = parse_args()
    workspace_size = workspace_bytes(args.workspace_gib)
    validate_static_contract(workspace_size)
    if args.list_cases:
        print(
            json.dumps(
                probe_cases(workspace_size), indent=2, sort_keys=True
            )
        )
        return 0

    require_build_args(args)
    source = write_source_program(args)
    package = compile_seed_package(args, source)
    module_path, resource_ids, slots_per_tile = prepare_workspace_manifest(
        package, workspace_size
    )
    build_probe(args, package, module_path, slots_per_tile)
    validate_manifest(package, resource_ids, workspace_size)
    if args.no_card:
        verify_no_card(args, package, resource_ids, workspace_size)
        return 0

    require_board_args(args)
    execute_board(args, package, resource_ids, workspace_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
