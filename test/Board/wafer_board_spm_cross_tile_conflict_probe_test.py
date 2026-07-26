#!/usr/bin/env python3
"""Build and run paired CT/RDMA SPM controls on every physical tile."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import struct
import sys
import tempfile

import wafer_board_ddr_tile_offset_probe_test as cluster_support
import wafer_board_memory_descriptor_calibration_probe_test as mdc_probe
import wafer_memory_descriptor_calibration_catalog as catalog


RANK_COUNT = 16
REQUEST_MAGIC = 0x31515443504D5357
RECORD_MAGIC = 0x31524343504D5357
REQUEST_GUARD = 0x7BE143A592D06FC8
RECORD_GUARD = 0x49C5E270A6138BDF
SCHEMA = 2
COUNTERBALANCED_ROUNDS = 4
RANK_ORDER_FORWARD = 0
RANK_ORDER_REVERSE = 1
SERIAL_REQUEST_WORD = 0
WINDOW_REQUEST_WORD = catalog.REQUEST_WORDS
REQUEST_META_WORD = 2 * catalog.REQUEST_WORDS
REQUEST_META_WORDS = 20
RECORD_META_WORD = catalog.RECORD_WORDS
RECORD_META_WORDS = 21
REQUEST_META_BEGIN = REQUEST_META_WORD * 8
RECORD_META_BEGIN = RECORD_META_WORD * 8
RECORD_META_END = (RECORD_META_WORD + RECORD_META_WORDS) * 8
LAUNCH_ABI = "tx81-cluster-direct-dte-prepare-main-v1"
STATUS_ABI = "wafer-direct-dte-status-v2"
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_memory_descriptor_calibration_probe.c"
PROBE_LL = INPUT_DIR / "wafer_spm_cross_tile_conflict_probe.ll"

REQ_META = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "RANK": 2,
    "RANK_COUNT": 3,
    "COORDINATE": 4,
    "SERIAL_CASE": 5,
    "WINDOW_CASE": 6,
    "SERIAL_REQUEST_WORD": 7,
    "WINDOW_REQUEST_WORD": 8,
    "SPM_A": 9,
    "SPM_B": 10,
    "TRANSFER_BYTES": 11,
    "ISSUE_ORDER": 12,
    "SAMPLE": 13,
    "RESOURCE_BYTES": 14,
    "GUARD": 15,
    "EXECUTION_ROUND": 16,
    "RANK_ORDER": 17,
    "RANK_PHASE": 18,
    "FIRST_SCHEDULE": 19,
}
REC_META = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "RANK": 3,
    "RANK_COUNT": 4,
    "COORDINATE": 5,
    "INNER_CASE": 6,
    "SCHEDULE": 7,
    "REQUEST_DDR": 8,
    "PAYLOAD_DDR": 9,
    "OUTPUT_DDR": 10,
    "SPM_A": 11,
    "SPM_B": 12,
    "SAMPLE": 13,
    "REQUEST_GUARD": 14,
    "RECORD_GUARD": 15,
    "EXECUTION_ROUND": 16,
    "RANK_ORDER": 17,
    "RANK_PHASE": 18,
    "FIRST_SCHEDULE": 19,
    "EXECUTION_ORDINAL": 20,
}


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    execution_round: int
    rank_order: int
    rank_phase: int
    first_schedule: int

    @property
    def rank_order_name(self) -> str:
        return (
            "forward"
            if self.rank_order == RANK_ORDER_FORWARD
            else "reverse"
        )

    @property
    def first_schedule_name(self) -> str:
        return (
            "serial"
            if self.first_schedule == catalog.SCHEDULE_SERIAL
            else "window"
        )


def execution_plan(rank: int, execution_round: int) -> ExecutionPlan:
    if rank not in range(RANK_COUNT):
        raise RuntimeError("cross-tile execution rank is invalid")
    if execution_round not in range(COUNTERBALANCED_ROUNDS):
        raise RuntimeError("cross-tile execution round is invalid")
    rank_order = (
        RANK_ORDER_FORWARD
        if execution_round % 2 == 0
        else RANK_ORDER_REVERSE
    )
    return ExecutionPlan(
        execution_round=execution_round,
        rank_order=rank_order,
        rank_phase=(
            rank if rank_order == RANK_ORDER_FORWARD
            else RANK_COUNT - 1 - rank
        ),
        first_schedule=(
            catalog.SCHEDULE_SERIAL
            if (rank + execution_round) % 2 == 0
            else catalog.SCHEDULE_WINDOW
        ),
    )


@dataclasses.dataclass(frozen=True)
class ConflictPair:
    coordinate_id: int
    name: str
    translation: int
    phase: int
    transfer_bytes: int
    issue_order: int
    relative_offset: int
    serial: catalog.MemoryCase
    window: catalog.MemoryCase

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.coordinate_id,
            "name": self.name,
            "translation": self.translation,
            "base_phase_mod_256": self.phase,
            "relative_spm_offset": self.relative_offset,
            "transfer_bytes": self.transfer_bytes,
            "issue_order": "a-b" if self.issue_order == 0 else "b-a",
            "schedules": {
                "serial": self.serial.name,
                "window": self.window.name,
            },
            "ranks": RANK_COUNT,
            "samples": COUNTERBALANCED_ROUNDS,
            "execution": (
                "single-active-rank-per-barrier-phase with alternating "
                "forward/reverse rank traversal and rank/round schedule order"
            ),
            "oracle": (
                "paired-same-launch exact result+SPM guards+instruction "
                "counts+completion+actual resource bases"
            ),
        }


@dataclasses.dataclass(frozen=True)
class RankInputs:
    rank: int
    wire_sample: int
    execution: ExecutionPlan
    serial_built: catalog.CasePayload
    window_built: catalog.CasePayload
    serial_output: pathlib.Path
    window_output: pathlib.Path


def _pair_key(
    case: catalog.MemoryCase,
) -> tuple[int, int, int, int, int]:
    phase = case.spm_a % 256
    translation = case.spm_a - catalog.SPM_BASE - phase
    return (
        translation,
        phase,
        case.descriptor.compact_bytes,
        case.issue_order,
        case.spm_b - case.spm_a,
    )


def conflict_pairs() -> tuple[ConflictPair, ...]:
    grouped: dict[
        tuple[int, int, int, int, int],
        dict[int, catalog.MemoryCase],
    ] = {}
    for case in catalog.CONFLICT_EQUIVALENCE_CASES:
        if (
            case.kind != catalog.KIND_ENGINE_PAIR
            or (case.engine_a, case.engine_b)
            != (catalog.ENGINE_CT, catalog.ENGINE_RDMA)
            or not case.is_exact
        ):
            raise RuntimeError(
                f"{case.name}: cross-tile input is not an exact CT/RDMA pair"
            )
        schedules = grouped.setdefault(_pair_key(case), {})
        if case.schedule in schedules:
            raise RuntimeError(
                f"{case.name}: duplicate cross-tile schedule coordinate"
            )
        schedules[case.schedule] = case

    result: list[ConflictPair] = []
    for coordinate_id, key in enumerate(sorted(grouped)):
        translation, phase, transfer, issue_order, relative_offset = key
        schedules = grouped[key]
        if set(schedules) != {
            catalog.SCHEDULE_SERIAL,
            catalog.SCHEDULE_WINDOW,
        }:
            raise RuntimeError(
                f"cross-tile coordinate {key} lacks paired controls"
            )
        serial = schedules[catalog.SCHEDULE_SERIAL]
        window = schedules[catalog.SCHEDULE_WINDOW]
        result.append(
            ConflictPair(
                coordinate_id=coordinate_id,
                name=(
                    "spm-physical-tile-ct-rdma-"
                    f"translation-{translation}-phase-{phase}-"
                    f"bytes-{transfer}-"
                    f"{'a-b' if issue_order == 0 else 'b-a'}"
                ),
                translation=translation,
                phase=phase,
                transfer_bytes=transfer,
                issue_order=issue_order,
                relative_offset=relative_offset,
                serial=serial,
                window=window,
            )
        )
    if not result:
        raise RuntimeError("cross-tile conflict-equivalence matrix is empty")
    return tuple(result)


def _inner_request_words(payload: bytes) -> tuple[int, ...]:
    return struct.unpack_from(f"<{catalog.REQUEST_WORDS}Q", payload)


def build_rank_request(
    pair: ConflictPair, rank: int, launch_sample: int
) -> tuple[bytes, catalog.CasePayload, catalog.CasePayload, int]:
    plan = execution_plan(rank, launch_sample)
    wire_sample = launch_sample * RANK_COUNT + rank
    serial_built = catalog.build_case_payload(pair.serial, wire_sample)
    window_built = catalog.build_case_payload(pair.window, wire_sample)
    serial_words = _inner_request_words(serial_built.request)
    window_words = _inner_request_words(window_built.request)
    differing = {
        index
        for index, (serial, window) in enumerate(
            zip(serial_words, window_words, strict=True)
        )
        if serial != window
    }
    allowed_differences = {
        catalog.REQ["CASE"],
        catalog.REQ["SCHEDULE"],
    }
    if (
        differing - allowed_differences
        or serial_words[catalog.REQ["SCHEDULE"]]
        != catalog.SCHEDULE_SERIAL
        or window_words[catalog.REQ["SCHEDULE"]]
        != catalog.SCHEDULE_WINDOW
        or serial_built.payload != window_built.payload
    ):
        raise RuntimeError(
            f"{pair.name}: serial/window controls are not a one-factor pair"
        )

    request = bytearray(serial_built.request)
    window_begin = WINDOW_REQUEST_WORD * 8
    window_end = window_begin + catalog.REQUEST_WORDS * 8
    request[window_begin:window_end] = window_built.request[
        : catalog.REQUEST_WORDS * 8
    ]
    meta = [0] * REQUEST_META_WORDS
    values = {
        "MAGIC": REQUEST_MAGIC,
        "SCHEMA_AND_WORDS": (SCHEMA << 32) | REQUEST_META_WORDS,
        "RANK": rank,
        "RANK_COUNT": RANK_COUNT,
        "COORDINATE": pair.coordinate_id,
        "SERIAL_CASE": pair.serial.case_id,
        "WINDOW_CASE": pair.window.case_id,
        "SERIAL_REQUEST_WORD": SERIAL_REQUEST_WORD,
        "WINDOW_REQUEST_WORD": WINDOW_REQUEST_WORD,
        "SPM_A": pair.serial.spm_a,
        "SPM_B": pair.serial.spm_b,
        "TRANSFER_BYTES": pair.transfer_bytes,
        "ISSUE_ORDER": pair.issue_order,
        "SAMPLE": wire_sample,
        "RESOURCE_BYTES": catalog.RESOURCE_BYTES,
        "GUARD": REQUEST_GUARD,
        "EXECUTION_ROUND": plan.execution_round,
        "RANK_ORDER": plan.rank_order,
        "RANK_PHASE": plan.rank_phase,
        "FIRST_SCHEDULE": plan.first_schedule,
    }
    for key, value in values.items():
        meta[REQ_META[key]] = value
    struct.pack_into(
        f"<{REQUEST_META_WORDS}Q",
        request,
        REQUEST_META_BEGIN,
        *meta,
    )
    return bytes(request), serial_built, window_built, wire_sample


def validate_static_contract(
    pairs: tuple[ConflictPair, ...],
) -> None:
    names = {pair.name for pair in pairs}
    ids = {pair.coordinate_id for pair in pairs}
    if (
        len(names) != len(pairs)
        or ids != set(range(len(pairs)))
        or any(pair.relative_offset != 8192 for pair in pairs)
    ):
        raise RuntimeError("cross-tile pair inventory is not canonical")
    for execution_round in range(COUNTERBALANCED_ROUNDS):
        plans = tuple(
            execution_plan(rank, execution_round)
            for rank in range(RANK_COUNT)
        )
        if (
            {plan.rank_phase for plan in plans}
            != set(range(RANK_COUNT))
            or {
                plan.first_schedule for plan in plans
            }
            != {
                catalog.SCHEDULE_SERIAL,
                catalog.SCHEDULE_WINDOW,
            }
            or len(
                [
                    plan
                    for plan in plans
                    if plan.first_schedule == catalog.SCHEDULE_SERIAL
                ]
            )
            != RANK_COUNT // 2
        ):
            raise RuntimeError(
                "cross-tile counterbalance plan is incomplete"
            )
    for pair in pairs:
        request, serial, window, wire_sample = build_rank_request(
            pair, RANK_COUNT - 1, 0
        )
        if (
            len(request) != catalog.RESOURCE_BYTES
            or len(serial.payload) != catalog.RESOURCE_BYTES
            or len(window.payload) != catalog.RESOURCE_BYTES
            or wire_sample != RANK_COUNT - 1
        ):
            raise RuntimeError(
                f"{pair.name}: bounded resource contract is invalid"
            )
    # The shared cluster qualification helper must accept other typed
    # all-rank clients that do not expose the DDR-only selector field.
    cluster_support.validate_board_args(
        argparse.Namespace(
            expected_runtime_version=1,
            expected_device_name="contract-device",
            expected_pci_bus_id="0000:00:00.0",
            expected_tile_count=RANK_COUNT,
            expected_runtime_library_sha256="0" * 64,
            completion_timeout_ms=1,
            repeat=COUNTERBALANCED_ROUNDS,
        )
    )


def validate_physical_coordinates(
    coordinates: dict[int, tuple[int, int]],
) -> None:
    if (
        set(coordinates) != set(range(RANK_COUNT))
        or len(set(coordinates.values())) != RANK_COUNT
    ):
        raise RuntimeError(
            "logical ranks do not map to 16 unique physical tiles"
        )


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
    parser.add_argument("--completion-timeout-ms", type=int, default=180000)
    parser.add_argument(
        "--repeat",
        type=int,
        default=COUNTERBALANCED_ROUNDS,
    )
    return parser.parse_args()


def require_build_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "repo_root",
            "wafer_compile",
            "wafer_run",
            "llvm_clangxx",
            "work_dir",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise RuntimeError(
            f"cross-tile probe build arguments are missing: {missing}"
        )
    args.repo_root = args.repo_root.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.work_dir == args.repo_root or args.work_dir in args.repo_root.parents:
        raise RuntimeError("cross-tile probe work directory is too broad")


def select_pair(
    args: argparse.Namespace, pairs: tuple[ConflictPair, ...]
) -> ConflictPair:
    by_name = {pair.name: pair for pair in pairs}
    selected = args.selected_cases or []
    unknown = sorted(set(selected) - set(by_name))
    if unknown:
        raise RuntimeError(f"unknown cross-tile cases: {unknown}")
    if len(selected) > 1:
        raise RuntimeError(
            "cross-tile execution accepts exactly one paired case per run"
        )
    if selected:
        return by_name[selected[0]]
    return pairs[0]


def build_probe(
    args: argparse.Namespace,
    package: pathlib.Path,
    module_path: pathlib.Path,
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
        raise RuntimeError(
            f"cross-tile build dependencies are missing: {missing}"
        )

    build = args.work_dir / "probe-build"
    build.mkdir()
    # Keep the helper basename distinct from PROBE_LL.  The device linker
    # compiles that IR to an object named after the IR stem in its staging
    # directory; sharing the basename could replace the C helper before link.
    helper = build / "wafer_spm_cross_tile_conflict_probe_helper.o"
    linked = build / "wafer_spm_cross_tile_conflict_probe.so"
    cluster_support.run(
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
    cluster_support.run(
        [str(objcopy), "-R", ".riscv.attributes", str(helper)]
    )
    cluster_support.run(
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
    staged = module_path.with_name(
        f".{module_path.name}.spm-cross-tile-conflict"
    )
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(
        ".manifest.json.spm-cross-tile-conflict"
    )
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staged_manifest, manifest_path)


def verify_no_card(
    args: argparse.Namespace, package: pathlib.Path
) -> None:
    result = cluster_support.run(
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
        raise RuntimeError(
            "cross-tile no-card launch evidence is incomplete"
        )
    print("spm_cross_tile_conflict_probe_no_card: passed")


def write_rank_resources(
    args: argparse.Namespace,
    bindings: dict[tuple[int, str, int], int],
    pair: ConflictPair,
    launch_sample: int,
) -> tuple[list[str], tuple[RankInputs, ...]]:
    raw_dir = (
        args.work_dir
        / "raw"
        / pair.name
        / f"sample-{launch_sample}"
    )
    raw_dir.mkdir(parents=True)
    resource_args: list[str] = []
    expectations: list[RankInputs] = []
    for rank in range(RANK_COUNT):
        request, serial_built, window_built, wire_sample = (
            build_rank_request(pair, rank, launch_sample)
        )
        request_path = raw_dir / f"rank-{rank:02d}.request.raw"
        payload_path = raw_dir / f"rank-{rank:02d}.payload.raw"
        serial_output = raw_dir / f"rank-{rank:02d}.serial.raw"
        window_output = raw_dir / f"rank-{rank:02d}.window.raw"
        request_path.write_bytes(request)
        payload_path.write_bytes(serial_built.payload)
        resource_args.extend(
            [
                "--resource",
                f"{bindings[(rank, 'user_input', 0)]}={request_path}",
                "--resource",
                f"{bindings[(rank, 'user_input', 1)]}={payload_path}",
                "--output",
                f"{bindings[(rank, 'output', 0)]}={serial_output}",
                "--output",
                f"{bindings[(rank, 'output', 1)]}={window_output}",
            ]
        )
        expectations.append(
            RankInputs(
                rank=rank,
                wire_sample=wire_sample,
                execution=execution_plan(rank, launch_sample),
                serial_built=serial_built,
                window_built=window_built,
                serial_output=serial_output,
                window_output=window_output,
            )
        )
    return resource_args, tuple(expectations)


def _parse_meta(raw: bytes) -> tuple[int, ...]:
    if len(raw) != catalog.RESOURCE_BYTES:
        raise RuntimeError("cross-tile output resource size is wrong")
    return struct.unpack_from(
        f"<{RECORD_META_WORDS}Q", raw, RECORD_META_BEGIN
    )


def expected_record_meta(
    pair: ConflictPair,
    case: catalog.MemoryCase,
    rank: int,
    wire_sample: int,
    execution: ExecutionPlan,
    schedule: int,
) -> dict[str, int]:
    return {
        "MAGIC": RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (SCHEMA << 32) | RECORD_META_WORDS,
        "STATUS": 0,
        "RANK": rank,
        "RANK_COUNT": RANK_COUNT,
        "COORDINATE": pair.coordinate_id,
        "INNER_CASE": case.case_id,
        "SCHEDULE": schedule,
        "SPM_A": case.spm_a,
        "SPM_B": case.spm_b,
        "SAMPLE": wire_sample,
        "REQUEST_GUARD": REQUEST_GUARD,
        "RECORD_GUARD": RECORD_GUARD,
        "EXECUTION_ROUND": execution.execution_round,
        "RANK_ORDER": execution.rank_order,
        "RANK_PHASE": execution.rank_phase,
        "FIRST_SCHEDULE": execution.first_schedule,
        "EXECUTION_ORDINAL": (
            0 if schedule == execution.first_schedule else 1
        ),
    }


def _validate_output(
    path: pathlib.Path,
    pair: ConflictPair,
    case: catalog.MemoryCase,
    built: catalog.CasePayload,
    rank: int,
    wire_sample: int,
    execution: ExecutionPlan,
    schedule: int,
    physical: tuple[int, int],
) -> dict[str, object]:
    raw = path.read_bytes()
    meta = _parse_meta(raw)
    expected = expected_record_meta(
        pair, case, rank, wire_sample, execution, schedule
    )
    failures = {
        key: (meta[REC_META[key]], value)
        for key, value in expected.items()
        if meta[REC_META[key]] != value
    }
    bases = {
        "request": meta[REC_META["REQUEST_DDR"]],
        "payload": meta[REC_META["PAYLOAD_DDR"]],
        "output": meta[REC_META["OUTPUT_DDR"]],
    }
    if (
        failures
        or any(value == 0 or value % 256 for value in bases.values())
        or len(set(bases.values())) != len(bases)
    ):
        raise RuntimeError(
            f"{pair.name} rank {rank}: cross-tile meta oracle failed: "
            f"{failures}, bases={bases}"
        )

    sanitized = bytearray(raw)
    sanitized[RECORD_META_BEGIN:RECORD_META_END] = bytes(
        [catalog.RESOURCE_CANARY]
    ) * (RECORD_META_END - RECORD_META_BEGIN)
    temp_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.inner-",
            suffix=".raw",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary.write(sanitized)
            temp_path = pathlib.Path(temporary.name)
        observation = mdc_probe.validate_output(
            temp_path, case, built, wire_sample
        )
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    observation.update(
        {
            "rank": rank,
            "physical_x": physical[0],
            "physical_y": physical[1],
            "resource_bases": bases,
            "paired_schedule": (
                "serial"
                if schedule == catalog.SCHEDULE_SERIAL
                else "window"
            ),
            "execution_round": execution.execution_round,
            "rank_order": execution.rank_order_name,
            "rank_phase": execution.rank_phase,
            "first_schedule": execution.first_schedule_name,
            "execution_ordinal": (
                0 if schedule == execution.first_schedule else 1
            ),
        }
    )
    return observation


def _sign(value: int) -> int:
    return 0 if value == 0 else (1 if value > 0 else -1)


def summarize_launch(
    pair: ConflictPair,
    launch_sample: int,
    observations: dict[int, dict[int, dict[str, object]]],
) -> dict[str, object]:
    metrics = (
        "plan_cycles",
        "full_execution",
        "ct_blocking",
        "rdma_blocking",
    )
    rows: list[dict[str, object]] = []
    signs: dict[str, list[int]] = {metric: [] for metric in metrics}
    all_addresses: set[int] = set()
    launch_rank_orders: set[str] = set()
    launch_rank_phases: set[int] = set()
    launch_first_schedules: list[str] = []
    launch_physical_tiles: set[tuple[int, int]] = set()
    for rank in range(RANK_COUNT):
        schedules = observations[rank]
        serial = schedules[catalog.SCHEDULE_SERIAL]
        window = schedules[catalog.SCHEDULE_WINDOW]
        execution_keys = (
            "execution_round",
            "rank_order",
            "rank_phase",
            "first_schedule",
        )
        if any(
            serial[key] != window[key] for key in execution_keys
        ) or {
            int(serial["execution_ordinal"]),
            int(window["execution_ordinal"]),
        } != {0, 1}:
            raise RuntimeError(
                f"{pair.name} rank {rank}: paired controls do not share "
                "one counterbalanced execution phase"
            )
        launch_rank_orders.add(str(serial["rank_order"]))
        launch_rank_phases.add(int(serial["rank_phase"]))
        launch_first_schedules.append(str(serial["first_schedule"]))
        launch_physical_tiles.add(
            (int(serial["physical_x"]), int(serial["physical_y"]))
        )
        serial_bases = serial["resource_bases"]
        window_bases = window["resource_bases"]
        assert isinstance(serial_bases, dict)
        assert isinstance(window_bases, dict)
        if (
            serial_bases["request"] != window_bases["request"]
            or serial_bases["payload"] != window_bases["payload"]
        ):
            raise RuntimeError(
                f"{pair.name} rank {rank}: paired controls used different "
                "input allocations"
            )
        rank_addresses = {
            int(serial_bases["request"]),
            int(serial_bases["payload"]),
            int(serial_bases["output"]),
            int(window_bases["output"]),
        }
        if len(rank_addresses) != 4 or rank_addresses & all_addresses:
            raise RuntimeError(
                f"{pair.name} rank {rank}: runtime resources alias"
            )
        all_addresses.update(rank_addresses)
        deltas: dict[str, int] = {}
        for metric in metrics:
            if metric == "plan_cycles":
                serial_value = int(serial["plan_cycles"])
                window_value = int(window["plan_cycles"])
            else:
                serial_pmu = serial["pmu"]
                window_pmu = window["pmu"]
                assert isinstance(serial_pmu, dict)
                assert isinstance(window_pmu, dict)
                serial_value = int(serial_pmu[metric])
                window_value = int(window_pmu[metric])
            deltas[metric] = window_value - serial_value
            signs[metric].append(_sign(deltas[metric]))
        rows.append(
            {
                "rank": rank,
                "physical_x": serial["physical_x"],
                "physical_y": serial["physical_y"],
                "request_ddr": serial_bases["request"],
                "payload_ddr": serial_bases["payload"],
                "serial_output_ddr": serial_bases["output"],
                "window_output_ddr": window_bases["output"],
                "execution_round": serial["execution_round"],
                "rank_order": serial["rank_order"],
                "rank_phase": serial["rank_phase"],
                "first_schedule": serial["first_schedule"],
                "window_minus_serial": deltas,
            }
        )
    if (
        len(launch_rank_orders) != 1
        or launch_rank_phases != set(range(RANK_COUNT))
        or len(launch_physical_tiles) != RANK_COUNT
        or launch_first_schedules.count("serial") != RANK_COUNT // 2
        or launch_first_schedules.count("window") != RANK_COUNT // 2
    ):
        raise RuntimeError(
            f"{pair.name}: launch counterbalance inventory is incomplete"
        )
    direction_consistent = {
        metric: len(set(values)) == 1
        for metric, values in signs.items()
    }
    informative_signal = any(
        direction_consistent[metric]
        and signs[metric]
        and signs[metric][0] != 0
        for metric in ("plan_cycles", "full_execution")
    )
    return {
        "case": pair.as_dict(),
        "launch_sample": launch_sample,
        "rows": rows,
        "metric_directions": signs,
        "rank_order": next(iter(launch_rank_orders)),
        "first_schedule_counts": {
            "serial": launch_first_schedules.count("serial"),
            "window": launch_first_schedules.count("window"),
        },
        "physical_tile_direction_consistent": direction_consistent,
        "nonzero_cost_signal": informative_signal,
        "state": (
            "physical-tile-proxy-direction-consistent"
            if all(direction_consistent.values()) and informative_signal
            else (
                "physical-tile-consistent-but-zero-signal"
                if all(direction_consistent.values())
                else "inconclusive-physical-tile-direction-flip"
            )
        ),
        "compiler_use": "no-bank-coloring",
    }


def summarize_repeats(
    pair: ConflictPair, launches: list[dict[str, object]]
) -> dict[str, object]:
    metrics = (
        "plan_cycles",
        "full_execution",
        "ct_blocking",
        "rdma_blocking",
    )
    per_metric: dict[str, list[int]] = {metric: [] for metric in metrics}
    stable_bases = True
    reference_bases: dict[int, tuple[int, int]] = {}
    reference_coordinates: dict[int, tuple[int, int]] = {}
    stable_coordinates = True
    rank_orders: dict[int, list[str]] = {
        rank: [] for rank in range(RANK_COUNT)
    }
    first_schedules: dict[int, list[str]] = {
        rank: [] for rank in range(RANK_COUNT)
    }
    execution_rounds: set[int] = set()
    for launch in launches:
        rows = launch["rows"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            rank = int(row["rank"])
            execution_rounds.add(int(row["execution_round"]))
            rank_orders[rank].append(str(row["rank_order"]))
            first_schedules[rank].append(str(row["first_schedule"]))
            coordinate = (
                int(row["physical_x"]),
                int(row["physical_y"]),
            )
            if (
                rank in reference_coordinates
                and reference_coordinates[rank] != coordinate
            ):
                stable_coordinates = False
            reference_coordinates.setdefault(rank, coordinate)
            bases = (int(row["request_ddr"]), int(row["payload_ddr"]))
            if rank in reference_bases and reference_bases[rank] != bases:
                stable_bases = False
            reference_bases.setdefault(rank, bases)
            deltas = row["window_minus_serial"]
            assert isinstance(deltas, dict)
            for metric in metrics:
                per_metric[metric].append(_sign(int(deltas[metric])))
    direction_consistent = {
        metric: len(values) == len(launches) * RANK_COUNT
        and len(set(values)) == 1
        for metric, values in per_metric.items()
    }
    informative_signal = any(
        direction_consistent[metric]
        and per_metric[metric]
        and per_metric[metric][0] != 0
        for metric in ("plan_cycles", "full_execution")
    )
    counterbalanced = (
        execution_rounds == set(range(COUNTERBALANCED_ROUNDS))
        and all(
            values.count("forward") == COUNTERBALANCED_ROUNDS // 2
            and values.count("reverse") == COUNTERBALANCED_ROUNDS // 2
            for values in rank_orders.values()
        )
        and all(
            values.count("serial") == COUNTERBALANCED_ROUNDS // 2
            and values.count("window") == COUNTERBALANCED_ROUNDS // 2
            for values in first_schedules.values()
        )
    )
    complete = (
        len(launches) == COUNTERBALANCED_ROUNDS
        and counterbalanced
        and stable_bases
        and stable_coordinates
        and all(direction_consistent.values())
        and informative_signal
    )
    return {
        "case": pair.as_dict(),
        "launches": len(launches),
        "expected_launches": COUNTERBALANCED_ROUNDS,
        "counterbalanced_execution_complete": counterbalanced,
        "execution_rounds": sorted(execution_rounds),
        "rank_physical_coordinates_stable_across_launches": (
            stable_coordinates
        ),
        "rank_input_bases_stable_across_launches": stable_bases,
        "all_sample_tile_directions_consistent": direction_consistent,
        "nonzero_cost_signal": informative_signal,
        "state": (
            "physical-tile-heldout-proxy-consistent"
            if complete
            else (
                "physical-tile-heldout-consistent-but-zero-signal"
                if (
                    len(launches) == COUNTERBALANCED_ROUNDS
                    and counterbalanced
                    and stable_bases
                    and stable_coordinates
                    and all(direction_consistent.values())
                )
                else "inconclusive"
            )
        ),
        "compiler_use": "no-bank-coloring",
        "interpretation": (
            "local SPM offsets were replayed on each physical tile; this "
            "does not test remote SPM or concurrent cross-tile contention"
        ),
    }


def execute_board(
    args: argparse.Namespace,
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    pair: ConflictPair,
) -> None:
    launches: list[dict[str, object]] = []
    for launch_sample in range(args.repeat):
        resource_args, expectations = write_rank_resources(
            args, bindings, pair, launch_sample
        )
        result = cluster_support.run(
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
            raise RuntimeError(
                f"{pair.name}: board lifecycle evidence is incomplete"
            )
        coordinates = cluster_support.parse_tile_coordinates(result.stdout)
        validate_physical_coordinates(coordinates)
        observations: dict[int, dict[int, dict[str, object]]] = {}
        for expected in expectations:
            observations[expected.rank] = {
                catalog.SCHEDULE_SERIAL: _validate_output(
                    expected.serial_output,
                    pair,
                    pair.serial,
                    expected.serial_built,
                    expected.rank,
                    expected.wire_sample,
                    expected.execution,
                    catalog.SCHEDULE_SERIAL,
                    coordinates[expected.rank],
                ),
                catalog.SCHEDULE_WINDOW: _validate_output(
                    expected.window_output,
                    pair,
                    pair.window,
                    expected.window_built,
                    expected.rank,
                    expected.wire_sample,
                    expected.execution,
                    catalog.SCHEDULE_WINDOW,
                    coordinates[expected.rank],
                ),
            }
        summary = summarize_launch(
            pair, launch_sample, observations
        )
        launches.append(summary)
        print(
            "spm_cross_tile_conflict_launch: "
            + json.dumps(summary, sort_keys=True)
        )
        print(result.stdout, end="")
    print(
        "spm_cross_tile_conflict_summary: "
        + json.dumps(summarize_repeats(pair, launches), sort_keys=True)
    )


def main() -> int:
    args = parse_args()
    pairs = conflict_pairs()
    validate_static_contract(pairs)
    if args.list_cases:
        print(
            json.dumps(
                {
                    "cases": [pair.as_dict() for pair in pairs],
                    "physical_execution": (
                        "rank-16 cluster launch with exact physical "
                        "coordinates, alternating forward/reverse rank "
                        "traversal, alternating schedule-first order, and "
                        "one active rank per barrier phase"
                    ),
                    "non_goals": [
                        "remote SPM access",
                        "simultaneous cross-tile contention",
                        "physical bank naming",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    require_build_args(args)
    pair = select_pair(args, pairs)
    if args.repeat <= 0 or args.completion_timeout_ms <= 0:
        raise RuntimeError("repeat and timeout must be positive")
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "SPM cross-tile hardware execution is not armed",
                file=sys.stderr,
            )
            return 77
        cluster_support.validate_board_args(args)
        if args.repeat != COUNTERBALANCED_ROUNDS:
            raise RuntimeError(
                f"{pair.name}: board qualification requires exactly "
                f"{COUNTERBALANCED_ROUNDS} counterbalanced rounds"
            )

    package, module_path, bindings = cluster_support.compile_package(args)
    build_probe(args, package, module_path)
    verify_no_card(args, package)
    if args.no_card:
        return 0
    execute_board(args, package, bindings, pair)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        struct.error,
    ) as error:
        print(
            "wafer_board_spm_cross_tile_conflict_probe_test: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
