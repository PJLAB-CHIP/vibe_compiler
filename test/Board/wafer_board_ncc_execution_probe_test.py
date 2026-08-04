#!/usr/bin/env python3
"""Build and explicitly run the rank-one TX81 NCC execution probe."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
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
from collections.abc import Iterable

import wafer_ncc_probe_protocol as ncc_protocol


TARGET_IDENTITY = "wafer-tx81-single-card"
LAUNCH_KIND = "kernel"
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ncc_execution_probe.c"
PROBE_PLAN_C = INPUT_DIR / "wafer_ncc_probe_plan.c"
PROBE_HAZARD_C = INPUT_DIR / "wafer_ncc_hazard_relation.c"
PROBE_LL = INPUT_DIR / "wafer_ncc_execution_probe.ll"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("build-smoke", *SUITES),
        default="build-smoke",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="run only the named case from the selected suite; repeatable",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=10000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--require-overlap", action="store_true")
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
            "one-shot NCC execution probe exceeded its outer deadline; "
            "the runner will not retry or call reset/power interfaces"
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
            f"--launch-kind={LAUNCH_KIND}",
        ],
        timeout_seconds=300,
    )
    if "published verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not publish the seed package")
    return package


def locate_probe_bindings(
    package: pathlib.Path,
) -> tuple[pathlib.Path, tuple[int, int, int]]:
    manifest = json.loads((package / "manifest.json").read_text())
    entries = manifest.get("entries")
    modules = manifest.get("modules")
    resources = manifest.get("resources")
    if (
        manifest.get("rank_count") != 1
        or not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(modules, list)
        or len(modules) != 1
        or not isinstance(resources, list)
    ):
        raise RuntimeError("NCC probe requires a unique rank-one package")
    entry = entries[0]
    module = modules[0]
    slots = entry.get("slots")
    if (
        entry.get("rank") != 0
        or entry.get("module") != module.get("id")
        or not isinstance(slots, list)
        or [slot.get("ordinal") for slot in slots] != [0, 1, 2]
    ):
        raise RuntimeError("NCC probe pointer-table slots are not canonical")
    resources_by_id = {
        resource.get("id"): resource
        for resource in resources
        if isinstance(resource, dict) and isinstance(resource.get("id"), int)
    }
    resource_ids = tuple(slot.get("resource") for slot in slots)
    if len(resources_by_id) != len(resources) or any(
        resource_id not in resources_by_id for resource_id in resource_ids
    ):
        raise RuntimeError("NCC probe resources are not uniquely indexed")
    expected = (
        ("user_input", "read_only"),
        ("user_input", "read_only"),
        ("output", "write_only"),
    )
    for resource_id, contract in zip(resource_ids, expected):
        resource = resources_by_id[resource_id]
        if (
            (resource.get("role"), resource.get("access")) != contract
            or resource.get("bytes") != RESOURCE_BYTES
            or resource.get("host_visible") is not True
        ):
            raise RuntimeError(
                f"NCC probe resource {resource_id} does not match {contract}"
            )
    module_path_value = module.get("path")
    if not isinstance(module_path_value, str):
        raise RuntimeError("NCC probe module path is missing")
    module_path = package / module_path_value
    if not module_path.is_file():
        raise RuntimeError("NCC probe seed module is missing")
    return module_path, resource_ids


def build_probe(
    args: argparse.Namespace, package: pathlib.Path, module_path: pathlib.Path
) -> None:
    deps = args.repo_root / "third_party" / "tx8_deps"
    tool_bin = deps / TOOLCHAIN_DIR / "bin"
    gcc = tool_bin / "riscv64-unknown-elf-gcc"
    nm = tool_bin / "riscv64-unknown-elf-nm"
    objcopy = tool_bin / "riscv64-unknown-elf-objcopy"
    device_linker = args.repo_root / "tools" / "wafer_device_link.py"
    required = (
        gcc,
        nm,
        objcopy,
        device_linker,
        args.llvm_clangxx,
        PROBE_C,
        PROBE_PLAN_C,
        PROBE_HAZARD_C,
        PROBE_LL,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"NCC probe build dependencies are missing: {missing}")

    build = args.work_dir / "probe-build"
    build.mkdir()
    helper = build / "wafer_ncc_execution_probe.o"
    plan = build / "wafer_ncc_probe_plan.o"
    hazard = build / "wafer_ncc_hazard_relation.o"
    linked = build / "wafer_ncc_execution_probe.so"
    compile_prefix = [
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
            "-Wframe-larger-than=2048",
            "-DCONFIG_NO_PLATFORM_HOOK_H",
            "-DUSING_RISCV",
            f"-I{args.repo_root / 'runtime' / 'wafer_crt' / 'include'}",
            f"-I{args.repo_root / 'include'}",
            f"-I{deps / 'include'}",
            f"-I{INPUT_DIR}",
            "-mcpu=c908",
            "-mabi=lp64d",
    ]
    run(
        [
            *compile_prefix,
            str(PROBE_C),
            "-o",
            str(helper),
        ]
    )
    run([*compile_prefix, str(PROBE_PLAN_C), "-o", str(plan)])
    run([*compile_prefix, str(PROBE_HAZARD_C), "-o", str(hazard)])
    run([str(objcopy), "-R", ".riscv.attributes", str(helper)])
    run([str(objcopy), "-R", ".riscv.attributes", str(plan)])
    run([str(objcopy), "-R", ".riscv.attributes", str(hazard)])
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
            "--extra-object",
            str(plan),
            "--extra-object",
            str(hazard),
        ],
        timeout_seconds=120,
    )

    staged = module_path.with_name(f".{module_path.name}.ncc-execution-probe")
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    undefined_symbols = {
        line.split()[-1]
        for line in run([str(nm), "-u", str(module_path)]).stdout.splitlines()
        if line.split()
    }
    heap_symbols = undefined_symbols & {
        "rt_malloc",
        "rt_free",
        "csi_kernel_malloc",
        "csi_kernel_free",
    }
    if heap_symbols != {"rt_malloc", "rt_free"}:
        raise RuntimeError(
            "NCC probe package did not retain the exact Kcore RT-Thread heap "
            f"loader ABI: {sorted(heap_symbols)}"
        )
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    module_id = manifest["entries"][0]["module"]
    matching_modules = [
        module for module in manifest["modules"] if module.get("id") == module_id
    ]
    if len(matching_modules) != 1:
        raise RuntimeError("NCC probe could not resolve its unique module record")
    matching_modules[0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(
        ".manifest.json.ncc-execution-probe"
    )
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
        raise RuntimeError("wafer-run did not verify the rewritten NCC package")
    print("ncc_execution_probe_build: passed")


def validate_board_args(args: argparse.Namespace) -> None:
    required = {
        "--expected-runtime-version": args.expected_runtime_version,
        "--expected-device-name": args.expected_device_name,
        "--expected-pci-bus-id": args.expected_pci_bus_id,
        "--expected-tile-count": args.expected_tile_count,
        "--expected-runtime-library-sha256": args.expected_runtime_library_sha256,
    }
    missing = [option for option, value in required.items() if value in (None, "")]
    if missing:
        raise RuntimeError(f"board execution requires qualification: {missing}")
    digest = str(args.expected_runtime_library_sha256)
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise RuntimeError("runtime library SHA-256 must contain 64 hex digits")
    if args.completion_timeout_ms <= 0 or args.repeat <= 0:
        raise RuntimeError("completion timeout and repeat must be positive")


def validate_work_dir(
    repo_root: pathlib.Path, work_dir: pathlib.Path
) -> pathlib.Path:
    resolved_repo = repo_root.resolve()
    resolved_work = work_dir.resolve()
    if resolved_work == resolved_repo or resolved_work in resolved_repo.parents:
        raise RuntimeError("NCC probe work directory is too broad")
    return resolved_work


def delta32(after: int, before: int) -> int:
    return (after - before) & 0xFFFFFFFF


def delta64(after: int, before: int) -> int:
    return (after - before) & 0xFFFFFFFFFFFFFFFF


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


RECORD_BYTES = ncc_protocol.RECORD_WORDS * 8
V2_OUTPUT_SLOT_BASE = 4096
V2_OUTPUT_GUARD_BYTES = 256
V2_REPEATED_SLOT_BYTES = 16384
V2_OUTPUT_SLOT_STRIDE = (
    V2_REPEATED_SLOT_BYTES + 2 * V2_OUTPUT_GUARD_BYTES
)
RESOURCE_BYTES = (
    V2_OUTPUT_SLOT_BASE
    + (ncc_protocol.MAX_ISSUES + 1) * V2_OUTPUT_SLOT_STRIDE
)
RESOURCE_ELEMENTS = RESOURCE_BYTES // 4
V2_RECORD_GUARD = 0xD87C2A916BE4035F
V2_SPM_SLOT_BASE = 0x10000
V2_SPM_SLOT_STRIDE = 0x20000
V2_SPM_READ0_OFFSET = 0x100
V2_SPM_READ1_OFFSET = 0x5100
V2_SPM_WRITE_OFFSET = 0xA100
V2_NE_PHYSICAL_BYTES = 256
V2_NE_RHS_BYTES = 512
V2_NE_RESULT_BYTES = 32
V2_NE_LARGE_M = 64
V2_NE_LARGE_K = 128
V2_NE_LARGE_N = 128
V2_NE_LARGE_LHS_BYTES = 16384
V2_NE_LARGE_RHS_BYTES = 32768
V2_NE_LARGE_RESULT_BYTES = 16384
V2_NE_LARGE_READ1_OFFSET = 0x4300
V2_NE_LARGE_WRITE_OFFSET = 0xC500
V2_NE_SCOPE_M = 64
V2_NE_SCOPE_K = 256
V2_NE_SCOPE_N = 128
V2_NE_SCOPE_LHS_BYTES = 32768
V2_NE_SCOPE_RHS_BYTES = 65536
V2_NE_SCOPE_RESULT_BYTES = 16384
V2_NE_SCOPE_READ1_OFFSET = 0x8300
V2_NE_SCOPE_WRITE_OFFSET = 0x18500
V2_HAZARD_SELECTED_OFFSET = 0x4000
V2_HAZARD_SECOND_BASELINE_OFFSET = 0x6000
V2_HAZARD_UNSELECTED_OFFSETS = (
    (0x8000, 0xA000, 0xC000),
    (0x10000, 0x12000, 0x14000),
)
V2_STRIDED_INITIAL_SOURCE_SLOT = ncc_protocol.MAX_ISSUES
FMT_INT8 = ncc_protocol.DMA_FORMAT_INT8
FMT_FP16 = ncc_protocol.DMA_FORMAT_FP16
FMT_BF16 = 3
FMT_BOOL = 7
PMU64_NAMES = (
    "window",
    "full",
    "ct",
    "ne",
    "rdma",
    "wdma",
    "tdma",
    "scalar",
)
PMU64_COUNTERS = (
    ncc_protocol.REC["PMU64_AFTER"] - ncc_protocol.REC["PMU64_BEFORE"]
)
if len(PMU64_NAMES) != PMU64_COUNTERS:
    raise RuntimeError("NCC PMU name table does not match the wire schema")
PMU_STABLE_MASK = (1 << PMU64_COUNTERS) - 1

MODULE = f"""\
module {{
  func.func @main(
      %request: tensor<{RESOURCE_ELEMENTS}xf32>,
      %payload: tensor<{RESOURCE_ELEMENTS}xf32>) -> tensor<{RESOURCE_ELEMENTS}xf32> {{
    %result = stablehlo.add %request, %payload : tensor<{RESOURCE_ELEMENTS}xf32>
    return %result : tensor<{RESOURCE_ELEMENTS}xf32>
  }}
}}
"""
METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [RESOURCE_ELEMENTS], "dtype": "float32", "dynamic_dims": []},
        {"shape": [RESOURCE_ELEMENTS], "dtype": "float32", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [RESOURCE_ELEMENTS], "dtype": "float32", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request"},
        {"type_": "input_arg", "position": 1, "name": "payload"},
    ],
    "unused_inputs": [],
}


@dataclasses.dataclass(frozen=True)
class GenericProbeCase:
    name: str
    plan: ncc_protocol.Plan

    @property
    def disposition(self) -> str:
        return (
            "board-observation"
            if self.name in NCC_OBSERVATION_CASE_NAMES
            else "board-executable"
        )

    def plan_for_sample(self, sample: int) -> ncc_protocol.Plan:
        return dataclasses.replace(self.plan, sample=sample)

    def request_words(self, sample: int) -> tuple[int, ...]:
        return self.plan_for_sample(sample).request_words()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "engines": [lane.engine.name.lower() for lane in self.plan.lanes],
            "workers": [lane.worker for lane in self.plan.lanes],
            "rounds": self.plan.rounds,
            "effect": self.plan.effect_relation.name.lower(),
            "range": self.plan.range_relation.name.lower(),
            "first_operand": self.plan.first_operand.name.lower(),
            "second_operand": self.plan.second_operand.name.lower(),
            "schedule": self.plan.schedule.name.lower(),
            "wait": self.plan.wait_kind.name.lower(),
            "wait_worker_mask": self.plan.wait_worker_mask,
            "flags": self.plan.flags,
            "issue_limit": self.plan.issue_limit,
            "issue_modes": [
                lane.issue_mode.name.lower() for lane in self.plan.lanes
            ],
            "transfer_bytes": [
                lane.transfer_bytes for lane in self.plan.lanes
            ],
            "formats": [lane.element_format for lane in self.plan.lanes],
            "layouts": [
                {
                    "kind": lane.layout_kind.name.lower(),
                    "inner_bytes": lane.layout_inner_bytes,
                    "strides": [
                        lane.layout_stride0_bytes,
                        lane.layout_stride1_bytes,
                        lane.layout_stride2_bytes,
                    ],
                    "iterations": [
                        lane.layout_iteration0,
                        lane.layout_iteration1,
                        lane.layout_iteration2,
                    ],
                    "envelope_bytes": lane.dma_envelope_bytes(),
                }
                for lane in self.plan.lanes
            ],
        }


@dataclasses.dataclass(frozen=True)
class CalibrationDisposition:
    """Typed disposition for a calibration leaf that must not be issued."""

    name: str
    domain: str
    disposition: str
    reason: str
    completion_oracle: str

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def v2_lane(
    engine: ncc_protocol.Engine,
    *,
    worker: int = 0,
    mode: ncc_protocol.IssueMode = ncc_protocol.IssueMode.RAW,
    transfer_bytes: int | None = None,
    element_format: int | None = None,
) -> ncc_protocol.Lane:
    if transfer_bytes is None:
        transfer_bytes = 256 if engine == ncc_protocol.Engine.NE else 4096
    if element_format is None:
        element_format = FMT_FP16
    if element_format == FMT_FP16 and transfer_bytes % 2 != 0:
        raise ValueError("FP16 transfer_bytes must contain whole 2-byte elements")
    return ncc_protocol.Lane(
        engine=engine,
        worker=worker,
        issue_mode=mode,
        transfer_bytes=transfer_bytes,
        element_format=element_format,
    )


def v2_case(
    name: str,
    lanes: tuple[ncc_protocol.Lane, ...],
    *,
    rounds: int,
    schedule: ncc_protocol.Schedule,
    seed: int,
    effect_relation: ncc_protocol.EffectRelation = (
        ncc_protocol.EffectRelation.NONE
    ),
    range_relation: ncc_protocol.RangeRelation = (
        ncc_protocol.RangeRelation.DISJOINT
    ),
    first_operand: ncc_protocol.Operand = ncc_protocol.Operand.AUTO,
    second_operand: ncc_protocol.Operand = ncc_protocol.Operand.AUTO,
    wait_kind: ncc_protocol.WaitKind = ncc_protocol.WaitKind.BY_WORKER,
    wait_worker_mask_override: int | None = None,
    flags: int = 0,
    issue_limit: int = 0,
) -> GenericProbeCase:
    worker_mask = (
        0
        if wait_worker_mask_override is None
        else wait_worker_mask_override
    )
    if (
        wait_worker_mask_override is None
        and wait_kind == ncc_protocol.WaitKind.BY_WORKER
    ):
        for lane in lanes:
            worker_mask |= 1 << lane.worker
    return GenericProbeCase(
        name,
        ncc_protocol.Plan(
            lanes=lanes,
            rounds=rounds,
            effect_relation=effect_relation,
            range_relation=range_relation,
            schedule=schedule,
            wait_kind=wait_kind,
            wait_worker_mask=worker_mask,
            seed=seed,
            first_operand=first_operand,
            second_operand=second_operand,
            flags=flags,
            issue_limit=issue_limit,
        ),
    )


def v2_dma_strided_lane(
    engine: ncc_protocol.Engine,
    descriptor: tuple[int, int, int, int, int, int, int, int],
) -> ncc_protocol.Lane:
    (
        byte_count,
        inner_bytes,
        stride0_bytes,
        stride1_bytes,
        stride2_bytes,
        iteration0,
        iteration1,
        iteration2,
    ) = descriptor
    return ncc_protocol.Lane(
        engine=engine,
        worker=0,
        issue_mode=ncc_protocol.IssueMode.WRAPPER,
        transfer_bytes=byte_count,
        element_format=FMT_FP16,
        layout_kind=ncc_protocol.LayoutKind.DMA_STRIDED,
        layout_inner_bytes=inner_bytes,
        layout_stride0_bytes=stride0_bytes,
        layout_stride1_bytes=stride1_bytes,
        layout_stride2_bytes=stride2_bytes,
        layout_iteration0=iteration0,
        layout_iteration1=iteration1,
        layout_iteration2=iteration2,
    )


V2_ENGINES = (
    ncc_protocol.Engine.CT,
    ncc_protocol.Engine.NE,
    ncc_protocol.Engine.RDMA,
    ncc_protocol.Engine.WDMA,
    ncc_protocol.Engine.TDMA,
)
V2_DOCUMENTED_QUEUE_DEPTHS = {
    ncc_protocol.Engine.CT: 6,
    ncc_protocol.Engine.NE: 6,
    ncc_protocol.Engine.RDMA: 6,
    ncc_protocol.Engine.WDMA: 6,
    ncc_protocol.Engine.TDMA: 4,
}
V2_DOCUMENTED_QUEUE_DEPTHS_BY_NAME = {
    engine.name.lower(): depth
    for engine, depth in V2_DOCUMENTED_QUEUE_DEPTHS.items()
}


def v2_disjoint_control_rounds_are_safe(
    engine_names: Iterable[str], rounds: object
) -> bool:
    names = tuple(engine_names)
    return (
        type(rounds) is int
        and rounds > 0
        and len(names) == 2
        and len(set(names)) == 2
        and all(
            name in V2_DOCUMENTED_QUEUE_DEPTHS_BY_NAME
            and rounds < V2_DOCUMENTED_QUEUE_DEPTHS_BY_NAME[name]
            for name in names
        )
    )


QUALIFICATION_CASES = (
    GenericProbeCase(
        "environment-readonly",
        ncc_protocol.Plan(
            lanes=(),
            rounds=0,
            effect_relation=ncc_protocol.EffectRelation.NONE,
            range_relation=ncc_protocol.RangeRelation.DISJOINT,
            schedule=ncc_protocol.Schedule.WINDOW,
            wait_kind=ncc_protocol.WaitKind.NONE,
            wait_worker_mask=0,
            seed=0,
            command=ncc_protocol.Command.QUALIFY,
        ),
    ),
)
NO_CARD_PROTOCOL_CASES = (
    v2_case(
        "tdma-crt-i8-physical16",
        (
            v2_lane(
                ncc_protocol.Engine.TDMA,
                mode=ncc_protocol.IssueMode.WRAPPER,
                transfer_bytes=16,
                element_format=FMT_INT8,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x1001,
    ),
    v2_case(
        "tdma-crt-bool-to-i8-physical17",
        (
            v2_lane(
                ncc_protocol.Engine.TDMA,
                mode=ncc_protocol.IssueMode.WRAPPER,
                transfer_bytes=17,
                element_format=FMT_BOOL,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x1002,
    ),
)
TDMA_CRT_MANUAL_CASES = NO_CARD_PROTOCOL_CASES
V2_DMA_STRIDE_DESCRIPTORS = (
    ("1d", (24, 6, 10, 0, 0, 4, 1, 1)),
    ("2d", (24, 4, 8, 28, 0, 3, 2, 1)),
    ("3d", (32, 4, 8, 20, 52, 2, 2, 2)),
)
V2_DMA_STRIDE_MATRIX_CASES = tuple(
    v2_case(
        f"dma-stride-roundtrip-{dimension}",
        (
            v2_dma_strided_lane(ncc_protocol.Engine.RDMA, descriptor),
            v2_dma_strided_lane(ncc_protocol.Engine.WDMA, descriptor),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x7200 + index,
        effect_relation=ncc_protocol.EffectRelation.RAW,
        range_relation=ncc_protocol.RangeRelation.EXACT,
        first_operand=ncc_protocol.Operand.WRITE,
        second_operand=ncc_protocol.Operand.READ0,
    )
    for index, (dimension, descriptor) in enumerate(
        V2_DMA_STRIDE_DESCRIPTORS, start=1
    )
)
V2_STRIDED_RAW_CASES = tuple(
    v2_case(
        (
            f"dependency-strided-{dimension}-"
            f"{schedule.name.lower()}-observation"
        ),
        (
            v2_dma_strided_lane(ncc_protocol.Engine.RDMA, descriptor),
            v2_dma_strided_lane(ncc_protocol.Engine.WDMA, descriptor),
        ),
        rounds=1,
        schedule=schedule,
        seed=0x7300 + index * 0x10 + int(schedule),
        effect_relation=ncc_protocol.EffectRelation.RAW,
        range_relation=ncc_protocol.RangeRelation.STRIDED_ENVELOPE,
        first_operand=ncc_protocol.Operand.WRITE,
        second_operand=ncc_protocol.Operand.READ0,
    )
    for index, (dimension, descriptor) in enumerate(
        V2_DMA_STRIDE_DESCRIPTORS, start=1
    )
    for schedule in (
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    )
)
V2_STRIDED_READ_CASES = tuple(
    v2_case(
        (
            f"dependency-strided-{dimension}-"
            f"{effect.name.lower()}-{schedule.name.lower()}-observation"
        ),
        (
            v2_dma_strided_lane(first_engine, descriptor),
            v2_dma_strided_lane(second_engine, descriptor),
        ),
        rounds=1,
        schedule=schedule,
        seed=seed_base + index * 0x10 + int(schedule),
        effect_relation=effect,
        range_relation=ncc_protocol.RangeRelation.STRIDED_ENVELOPE,
        first_operand=ncc_protocol.Operand.READ0,
        second_operand=second_operand,
    )
    for (
        effect,
        first_engine,
        second_engine,
        second_operand,
        seed_base,
    ) in (
        (
            ncc_protocol.EffectRelation.WAR,
            ncc_protocol.Engine.WDMA,
            ncc_protocol.Engine.RDMA,
            ncc_protocol.Operand.WRITE,
            0x7400,
        ),
        (
            ncc_protocol.EffectRelation.RAR,
            ncc_protocol.Engine.WDMA,
            ncc_protocol.Engine.WDMA,
            ncc_protocol.Operand.READ0,
            0x7500,
        ),
    )
    for index, (dimension, descriptor) in enumerate(
        V2_DMA_STRIDE_DESCRIPTORS, start=1
    )
    for schedule in (
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    )
)
V2_STRIDED_WAW_CASES = tuple(
    v2_case(
        (
            f"dependency-strided-{dimension}-waw-"
            f"{relation.name.lower()}-{schedule.name.lower()}-observation"
        ),
        (
            v2_dma_strided_lane(ncc_protocol.Engine.RDMA, descriptor),
            v2_dma_strided_lane(ncc_protocol.Engine.RDMA, descriptor),
        ),
        rounds=1,
        schedule=schedule,
        seed=(
            0x7600
            + index * 0x40
            + int(relation) * 0x4
            + int(schedule)
        ),
        effect_relation=ncc_protocol.EffectRelation.WAW,
        range_relation=relation,
        first_operand=ncc_protocol.Operand.WRITE,
        second_operand=ncc_protocol.Operand.WRITE,
    )
    for index, (dimension, descriptor) in enumerate(
        V2_DMA_STRIDE_DESCRIPTORS, start=1
    )
    for relation in (
        ncc_protocol.RangeRelation.EXACT,
        ncc_protocol.RangeRelation.PARTIAL,
        ncc_protocol.RangeRelation.ADJACENT,
    )
    for schedule in (
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    )
)
V2_STRIDED_DEPENDENCY_CASES = (
    V2_STRIDED_RAW_CASES
    + V2_STRIDED_READ_CASES
    + V2_STRIDED_WAW_CASES
)
V2_SINGLE_CASES = tuple(
    v2_case(
        f"{engine.name.lower()}-raw-single",
        (v2_lane(engine),),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x100 + int(engine),
    )
    for engine in V2_ENGINES
)
V2_CONSTRUCTOR_CASES = (
    v2_case(
        "ct-constructor-return-address-nonnull",
        (v2_lane(ncc_protocol.Engine.CT),),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x6001,
        flags=ncc_protocol.CONSTRUCTOR_OBSERVATION,
    ),
)
V2_WORKER_CASES = (
    v2_case(
        "ct-worker0-raw-single",
        (v2_lane(ncc_protocol.Engine.CT, worker=0),),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x6100,
    ),
    v2_case(
        "ct-worker1-raw-single",
        (v2_lane(ncc_protocol.Engine.CT, worker=1),),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x6101,
    ),
    v2_case(
        "ct-worker2-raw-single",
        (v2_lane(ncc_protocol.Engine.CT, worker=2),),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x6102,
    ),
    v2_case(
        "ct-workers01-disjoint-r1-window",
        (
            v2_lane(ncc_protocol.Engine.CT, worker=0),
            v2_lane(ncc_protocol.Engine.CT, worker=1),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x6110,
    ),
    v2_case(
        "ct-workers02-disjoint-r1-window",
        (
            v2_lane(ncc_protocol.Engine.CT, worker=0),
            v2_lane(ncc_protocol.Engine.CT, worker=2),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x6111,
    ),
    v2_case(
        "ct-workers12-disjoint-r1-window",
        (
            v2_lane(ncc_protocol.Engine.CT, worker=1),
            v2_lane(ncc_protocol.Engine.CT, worker=2),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x6112,
    ),
    v2_case(
        "ct-workers012-disjoint-r1-window",
        (
            v2_lane(ncc_protocol.Engine.CT, worker=0),
            v2_lane(ncc_protocol.Engine.CT, worker=1),
            v2_lane(ncc_protocol.Engine.CT, worker=2),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x6113,
    ),
)
V2_COMPLETION_SCOPE_CASES = tuple(
    v2_case(
        f"ne-worker1-depth6-{spelling}-wait-window",
        (
            v2_lane(
                ncc_protocol.Engine.TDMA,
                worker=0,
                transfer_bytes=16,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.NE,
                worker=1,
                transfer_bytes=V2_NE_LARGE_RESULT_BYTES,
            ),
            v2_lane(
                ncc_protocol.Engine.NE,
                worker=1,
                transfer_bytes=V2_NE_LARGE_RESULT_BYTES,
            ),
        ),
        rounds=3,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=seed,
        wait_kind=wait_kind,
        wait_worker_mask_override=(
            0b010
            if wait_kind == ncc_protocol.WaitKind.BY_WORKER
            else None
        ),
    )
    for spelling, wait_kind, seed in (
        ("default", ncc_protocol.WaitKind.DEFAULT, 0x6121),
        ("byworker", ncc_protocol.WaitKind.BY_WORKER, 0x6122),
        ("local-fence", ncc_protocol.WaitKind.LOCAL_FENCE, 0x6123),
    )
)
V2_SUBSET_JOIN_CASES = tuple(
    v2_case(
        f"workers012-join{mask:03b}-observe-unjoined",
        (
            v2_lane(
                ncc_protocol.Engine.CT,
                worker=0,
                transfer_bytes=16384,
            ),
            v2_lane(
                ncc_protocol.Engine.CT,
                worker=1,
                transfer_bytes=16384,
            ),
            v2_lane(
                ncc_protocol.Engine.NE,
                worker=2,
                transfer_bytes=V2_NE_LARGE_RESULT_BYTES,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x6200 + mask,
        wait_worker_mask_override=mask,
    )
    for mask in (0b001, 0b010, 0b100, 0b011, 0b101, 0b110)
)
V2_WORKER_SCOPE_TARGETS = (
    ("ne", ncc_protocol.Engine.NE, V2_NE_SCOPE_RESULT_BYTES),
    ("rdma", ncc_protocol.Engine.RDMA, V2_REPEATED_SLOT_BYTES),
)
V2_WORKER_WAIT_SCOPE_CASES = tuple(
    v2_case(
        (
            f"worker-wait-scope-{engine_name}-worker{target_worker}-"
            f"{spelling}-tight-window"
        ),
        (
            v2_lane(
                ncc_protocol.Engine.TDMA,
                worker=(target_worker + 1) % 3,
                transfer_bytes=16,
                element_format=FMT_FP16,
            ),
            v2_lane(
                engine,
                worker=target_worker,
                transfer_bytes=transfer_bytes,
            ),
            v2_lane(
                engine,
                worker=target_worker,
                transfer_bytes=transfer_bytes,
            ),
        ),
        rounds=3,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x6250 + target_worker * 0x10 + int(engine),
        wait_kind=wait_kind,
        wait_worker_mask_override=(
            1 << target_worker
            if wait_kind == ncc_protocol.WaitKind.BY_WORKER
            else None
        ),
        flags=ncc_protocol.TIGHT_WORKER_SCOPE,
    )
    for target_worker in range(3)
    for engine_name, engine, transfer_bytes in V2_WORKER_SCOPE_TARGETS
    for spelling, wait_kind in (
        ("default", ncc_protocol.WaitKind.DEFAULT),
        ("byworker", ncc_protocol.WaitKind.BY_WORKER),
        ("local-fence", ncc_protocol.WaitKind.LOCAL_FENCE),
    )
)


def v2_worker_subset_scope_case(
    target_worker: int,
    target_name: str,
    target_engine: ncc_protocol.Engine,
    target_bytes: int,
    include_target: bool,
) -> GenericProbeCase:
    other_workers = tuple(
        worker for worker in range(3) if worker != target_worker
    )
    wait_mask = sum(1 << worker for worker in other_workers)
    if include_target:
        wait_mask |= 1 << target_worker
    return v2_case(
        (
            f"worker-subset-{target_name}-target{target_worker}-"
            f"{'include' if include_target else 'exclude'}-tight-window"
        ),
        (
            v2_lane(ncc_protocol.Engine.CT, worker=other_workers[0]),
            v2_lane(
                ncc_protocol.Engine.TDMA,
                worker=other_workers[1],
                transfer_bytes=16,
                element_format=FMT_FP16,
            ),
            v2_lane(
                target_engine,
                worker=target_worker,
                transfer_bytes=target_bytes,
            ),
        ),
        rounds=3,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x6260 + target_worker * 0x10 + int(target_engine),
        wait_worker_mask_override=wait_mask,
        flags=ncc_protocol.TIGHT_WORKER_SCOPE,
    )


V2_WORKER_SUBSET_SCOPE_CASES = tuple(
    v2_worker_subset_scope_case(
        target_worker,
        target_name,
        target_engine,
        target_bytes,
        include_target,
    )
    for target_worker in range(3)
    for target_name, target_engine, target_bytes in V2_WORKER_SCOPE_TARGETS
    for include_target in (False, True)
)
V2_WAIT_OVERHEAD_CASES = tuple(
    v2_case(
        f"{engine.name.lower()}-worker0-r2-{spelling}",
        (v2_lane(engine, worker=0),),
        rounds=2,
        schedule=schedule,
        seed=0x6130 + int(engine),
    )
    for engine in V2_ENGINES
    for spelling, schedule in (
        ("wait-each", ncc_protocol.Schedule.SERIAL),
        ("wait-once", ncc_protocol.Schedule.WINDOW),
    )
)
V2_MULTI_ISSUE_CASES = tuple(
    v2_case(
        f"{engine.name.lower()}-raw-rounds{rounds}-window",
        (v2_lane(engine),),
        rounds=rounds,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x200 + int(engine) * 0x10 + rounds,
    )
    for engine, depth in V2_DOCUMENTED_QUEUE_DEPTHS.items()
    for rounds in ((2,) if engine == ncc_protocol.Engine.TDMA else (2, 4))
)
V2_ISSUE_PATH_CASES = tuple(
    v2_case(
        (
            f"{engine.name.lower()}-{mode.name.lower()}-"
            "rounds2-window"
        ),
        (v2_lane(engine, mode=mode),),
        rounds=2,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x2600 + int(engine) * 0x10 + int(mode),
    )
    for engine in V2_ENGINES
    for mode in (ncc_protocol.IssueMode.RAW, ncc_protocol.IssueMode.WRAPPER)
)
V2_DOCUMENTED_DEPTH_CASES = tuple(
    v2_case(
        f"{engine.name.lower()}-raw-documented-depth{depth}-window",
        (
            (v2_lane(engine), v2_lane(engine))
            if depth > ncc_protocol.MAX_ROUNDS
            else (v2_lane(engine),)
        ),
        rounds=(
            depth // 2
            if depth > ncc_protocol.MAX_ROUNDS
            else depth
        ),
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x280 + int(engine) * 0x10 + depth,
    )
    for engine, depth in V2_DOCUMENTED_QUEUE_DEPTHS.items()
)
V2_DEPTH_PLUS_ONE_CASES = tuple(
    v2_case(
        f"{engine.name.lower()}-raw-depth{depth}-plus1-tight-window",
        (v2_lane(engine), v2_lane(engine)),
        rounds=4 if depth == 6 else 3,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x5000 + int(engine),
        flags=ncc_protocol.TIGHT_DEPTH_PLUS_ONE,
        issue_limit=depth + 1,
    )
    for engine, depth in V2_DOCUMENTED_QUEUE_DEPTHS.items()
)
V2_ACTIVE_OCCUPANCY_BYTES = {
    ncc_protocol.Engine.CT: 16384,
    ncc_protocol.Engine.NE: V2_NE_LARGE_RESULT_BYTES,
    ncc_protocol.Engine.RDMA: V2_REPEATED_SLOT_BYTES,
    ncc_protocol.Engine.WDMA: V2_REPEATED_SLOT_BYTES,
    ncc_protocol.Engine.TDMA: V2_REPEATED_SLOT_BYTES,
}
V2_ACTIVE_OCCUPANCY_CASES = tuple(
    v2_case(
        (
            f"{engine.name.lower()}-large-depth{depth}-plus1-"
            "tight-occupancy-observation"
        ),
        (
            v2_lane(
                engine,
                transfer_bytes=V2_ACTIVE_OCCUPANCY_BYTES[engine],
            ),
            v2_lane(
                engine,
                transfer_bytes=V2_ACTIVE_OCCUPANCY_BYTES[engine],
            ),
        ),
        rounds=(depth + 2) // 2,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x5010 + int(engine),
        flags=ncc_protocol.TIGHT_DEPTH_PLUS_ONE,
        issue_limit=depth + 1,
    )
    for engine, depth in V2_DOCUMENTED_QUEUE_DEPTHS.items()
)
V2_QUEUE_SATURATION_CASES = tuple(
    v2_case(
        (
            f"queue-saturation-{engine.name.lower()}-{load_name}-"
            f"depth{issue_limit}-tight-window"
        ),
        (
            v2_lane(engine, transfer_bytes=transfer_bytes),
            v2_lane(engine, transfer_bytes=transfer_bytes),
        ),
        rounds=(issue_limit + 1) // 2,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x5040 + int(engine) * 0x20,
        flags=ncc_protocol.TIGHT_QUEUE_SATURATION,
        issue_limit=issue_limit,
    )
    for engine, depth in V2_DOCUMENTED_QUEUE_DEPTHS.items()
    for load_name, transfer_bytes in (
        ("short", 256 if engine == ncc_protocol.Engine.NE else 4096),
        ("sustained", V2_ACTIVE_OCCUPANCY_BYTES[engine]),
    )
    for issue_limit in (depth - 1, depth, depth + 1)
)
V2_PAIR_CASES = tuple(
    v2_case(
        (
            f"{first.name.lower()}-{second.name.lower()}-disjoint-"
            f"r{rounds}-{schedule.name.lower()}"
        ),
        (v2_lane(first), v2_lane(second)),
        rounds=rounds,
        schedule=schedule,
        seed=(
            0x1000
            + int(first) * 0x100
            + int(second) * 0x10
            + rounds
            + int(schedule)
        ),
    )
    for first, second in itertools.permutations(V2_ENGINES, 2)
    for rounds in (
        (2,)
        if ncc_protocol.Engine.TDMA in (first, second)
        else (2, 4)
    )
    for schedule in (
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    )
)
V2_DEFERRED_CASES: tuple[CalibrationDisposition, ...] = ()

V2_PRODUCER_CONSUMER_CASES = (
    v2_case(
        "ordered-rdma-to-ct",
        (
            v2_lane(
                ncc_protocol.Engine.RDMA,
                transfer_bytes=256,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=256,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x7401,
        effect_relation=ncc_protocol.EffectRelation.RAW,
        range_relation=ncc_protocol.RangeRelation.EXACT,
        first_operand=ncc_protocol.Operand.WRITE,
        second_operand=ncc_protocol.Operand.READ0,
        flags=ncc_protocol.ORDERED_PRODUCER_CONSUMER,
    ),
    v2_case(
        "ordered-ct-to-wdma",
        (
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=256,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.WDMA,
                transfer_bytes=256,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x7402,
        effect_relation=ncc_protocol.EffectRelation.RAW,
        range_relation=ncc_protocol.RangeRelation.EXACT,
        first_operand=ncc_protocol.Operand.WRITE,
        second_operand=ncc_protocol.Operand.READ0,
        flags=ncc_protocol.ORDERED_PRODUCER_CONSUMER,
    ),
    v2_case(
        "ordered-ne-to-wdma",
        (
            v2_lane(
                ncc_protocol.Engine.NE,
                transfer_bytes=V2_NE_PHYSICAL_BYTES,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.WDMA,
                transfer_bytes=V2_NE_RESULT_BYTES,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x7403,
        effect_relation=ncc_protocol.EffectRelation.RAW,
        range_relation=ncc_protocol.RangeRelation.EXACT,
        first_operand=ncc_protocol.Operand.WRITE,
        second_operand=ncc_protocol.Operand.READ0,
        flags=ncc_protocol.ORDERED_PRODUCER_CONSUMER,
    ),
    *(
        v2_case(
            f"ordered-tdma-to-{consumer.name.lower()}",
            (
                v2_lane(
                    ncc_protocol.Engine.TDMA,
                    transfer_bytes=256,
                    element_format=FMT_FP16,
                ),
                v2_lane(
                    consumer,
                    transfer_bytes=256,
                    element_format=FMT_FP16,
                ),
            ),
            rounds=1,
            schedule=ncc_protocol.Schedule.SERIAL,
            seed=0x7404 + int(consumer),
            effect_relation=ncc_protocol.EffectRelation.RAW,
            range_relation=ncc_protocol.RangeRelation.EXACT,
            first_operand=ncc_protocol.Operand.WRITE,
            second_operand=ncc_protocol.Operand.READ0,
            flags=ncc_protocol.ORDERED_PRODUCER_CONSUMER,
        )
        for consumer in (ncc_protocol.Engine.CT, ncc_protocol.Engine.NE)
    ),
)
V2_KCORE_BOUNDARY_CASES = (
    v2_case(
        "ct-to-kcore-read-boundary",
        (
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x7410,
    ),
    v2_case(
        "ne-to-kcore-read-boundary",
        (
            v2_lane(
                ncc_protocol.Engine.NE,
                transfer_bytes=V2_NE_LARGE_RESULT_BYTES,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x7411,
    ),
)
V2_MAPPED_SPM_PURE_NCC_CASES = (
    v2_case(
        "mapped-spm-rdma-ct-wdma-a-terminal-only",
        (
            v2_lane(
                ncc_protocol.Engine.RDMA,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.WDMA,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x7420,
        flags=ncc_protocol.DOUBLE_SLOT_OBSERVATION,
    ),
    v2_case(
        "mapped-spm-rdma-ct-wdma-b-serial-drain-control",
        (
            v2_lane(
                ncc_protocol.Engine.RDMA,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.WDMA,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.SERIAL,
        seed=0x7420,
        flags=ncc_protocol.DOUBLE_SLOT_OBSERVATION,
    ),
)
V2_MAPPED_SPM_NCC_TO_KCORE_CASES = (
    v2_case(
        "mapped-spm-ncc-ne-depth4-to-kcore-a-no-local-wait",
        (
            v2_lane(
                ncc_protocol.Engine.NE,
                transfer_bytes=V2_NE_LARGE_RESULT_BYTES,
                element_format=FMT_FP16,
            ),
        ),
        rounds=4,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x7421,
        wait_kind=ncc_protocol.WaitKind.NONE,
        flags=ncc_protocol.TIGHT_KCORE_BOUNDARY,
    ),
    v2_case(
        "mapped-spm-ncc-ne-depth4-to-kcore-b-local-wait",
        (
            v2_lane(
                ncc_protocol.Engine.NE,
                transfer_bytes=V2_NE_LARGE_RESULT_BYTES,
                element_format=FMT_FP16,
            ),
        ),
        rounds=4,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x7421,
        wait_kind=ncc_protocol.WaitKind.LOCAL_FENCE,
        flags=ncc_protocol.TIGHT_KCORE_BOUNDARY,
    ),
)
V2_MAPPED_SPM_KCORE_TO_NCC_CASES = (
    v2_case(
        "mapped-spm-kcore-to-ncc-ct-a-volatile-fence-sync-only",
        (
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x7422,
        flags=ncc_protocol.MAPPED_SPM_KCORE_WRITE,
    ),
    v2_case(
        "mapped-spm-kcore-to-ncc-ct-b-preissue-local-wait-control",
        (
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
        ),
        rounds=1,
        schedule=ncc_protocol.Schedule.WINDOW,
        seed=0x7422,
        flags=(
            ncc_protocol.MAPPED_SPM_KCORE_WRITE
            | ncc_protocol.PREISSUE_LOCAL_WAIT
        ),
    ),
)
V2_MAPPED_SPM_BOUNDARY_CASES = (
    V2_MAPPED_SPM_PURE_NCC_CASES
    + V2_MAPPED_SPM_NCC_TO_KCORE_CASES
    + V2_MAPPED_SPM_KCORE_TO_NCC_CASES
)
V2_PRODUCER_CONSUMER_ALL_CASES = (
    V2_PRODUCER_CONSUMER_CASES + V2_KCORE_BOUNDARY_CASES
)

V2_LARGE_BACKLOG_CASES = (
    *(
        v2_case(
            f"{engine.name.lower()}-64k-{mode.name.lower()}-single",
            (
                v2_lane(
                    engine,
                    mode=mode,
                    transfer_bytes=65536,
                    element_format=FMT_FP16,
                ),
            ),
            rounds=1,
            schedule=ncc_protocol.Schedule.SERIAL,
            seed=0x7500 + int(engine) * 0x10 + int(mode),
        )
        for engine in (ncc_protocol.Engine.RDMA, ncc_protocol.Engine.WDMA)
        for mode in (
            ncc_protocol.IssueMode.RAW,
            ncc_protocol.IssueMode.WRAPPER,
        )
    ),
    *(
        v2_case(
            f"{engine.name.lower()}-large-{mode.name.lower()}-single",
            (
                v2_lane(
                    engine,
                    mode=mode,
                    transfer_bytes=16384,
                    element_format=FMT_FP16,
                ),
            ),
            rounds=1,
            schedule=ncc_protocol.Schedule.SERIAL,
            seed=0x7520 + int(engine) * 0x10 + int(mode),
        )
        for engine in (ncc_protocol.Engine.CT, ncc_protocol.Engine.NE)
        for mode in (
            ncc_protocol.IssueMode.RAW,
            ncc_protocol.IssueMode.WRAPPER,
        )
    ),
    *(
        v2_case(
            (
                f"rdma64k-{compute.name.lower()}-large-"
                f"{schedule.name.lower()}"
            ),
            (
                v2_lane(
                    ncc_protocol.Engine.RDMA,
                    transfer_bytes=65536,
                    element_format=FMT_FP16,
                ),
                v2_lane(
                    compute,
                    transfer_bytes=16384,
                    element_format=FMT_FP16,
                ),
            ),
            rounds=1,
            schedule=schedule,
            seed=0x7540 + int(compute) * 0x10 + int(schedule),
        )
        for compute in (ncc_protocol.Engine.CT, ncc_protocol.Engine.NE)
        for schedule in (
            ncc_protocol.Schedule.SERIAL,
            ncc_protocol.Schedule.WINDOW,
        )
    ),
)
V2_LARGE_OVERLAP_CASES = tuple(
    case for case in V2_LARGE_BACKLOG_CASES if len(case.plan.lanes) == 2
)
V2_LARGE_BACKLOG_SINGLE_CASES = tuple(
    case for case in V2_LARGE_BACKLOG_CASES if len(case.plan.lanes) == 1
)
V2_PROTOCOL_NEGATIVE_CASES = (
    CalibrationDisposition(
        name="execute-engine-none",
        domain="ncc-protocol",
        disposition="static-negative",
        reason="Engine.NONE is a wire sentinel, never an active lane",
        completion_oracle="typed Plan validation rejects before serialization",
    ),
    CalibrationDisposition(
        name="execute-worker-out-of-range",
        domain="ncc-protocol",
        disposition="static-negative",
        reason="worker ids outside 0..2 would alias in lower-level firmware",
        completion_oracle="typed Lane validation rejects before serialization",
    ),
    CalibrationDisposition(
        name="wait-mask-nonparticipant",
        domain="ncc-protocol",
        disposition="static-negative",
        reason="a worker wait mask may name only participating workers",
        completion_oracle="typed Plan validation rejects before serialization",
    ),
    CalibrationDisposition(
        name="execute-scalar-packet",
        domain="ncc-protocol",
        disposition="static-negative",
        reason="SCALAR is not an ordinary TsmExecute engine lane",
        completion_oracle="the public Engine enum intentionally has no SCALAR",
    ),
    CalibrationDisposition(
        name="execute-csr-packet",
        domain="ncc-protocol",
        disposition="static-negative",
        reason="CSR is not an ordinary TsmExecute engine lane",
        completion_oracle="the public Engine enum intentionally has no CSR",
    ),
)
V2_ADDITIONAL_DISPOSITIONS = (
    CalibrationDisposition(
        "constructor-builder-release",
        "constructor-ownership",
        "board-executable",
        "the raw path releases every builder immediately after packet materialization",
        "all issued packets retain exact result, range, count, and guard oracles",
    ),
    CalibrationDisposition(
        "execute-invalid-type-return",
        "ncc-protocol",
        "static-negative",
        "invalid engine types are rejected by the typed host plan",
        "no invalid packet reaches TsmExecute",
    ),
    CalibrationDisposition(
        "tdma-native-bool-exclusion",
        "tdma-format",
        "static-negative",
        "native BOOL Memset is excluded on the current profile",
        "only BOOL-to-I8 physical canonicalization may reach the device",
    ),
)
V2_ADDITIONAL_BY_NAME = {
    case.name: case for case in V2_ADDITIONAL_DISPOSITIONS
}
V2_PIPELINE_CASES = tuple(
    v2_case(
        f"rdma-ct-wdma-disjoint-r{rounds}-{schedule.name.lower()}",
        (
            v2_lane(ncc_protocol.Engine.RDMA),
            v2_lane(ncc_protocol.Engine.CT),
            v2_lane(ncc_protocol.Engine.WDMA),
        ),
        rounds=rounds,
        schedule=schedule,
        seed=0x3000 + rounds + int(schedule),
    )
    for rounds in (2, 4)
    for schedule in (
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    )
)
V2_DOUBLE_SLOT_OBSERVATION_CASES = tuple(
    v2_case(
        (
            f"double-slot-iterations{iterations}-"
            f"{schedule.name.lower()}-observation"
        ),
        (
            v2_lane(
                ncc_protocol.Engine.RDMA,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.CT,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
            v2_lane(
                ncc_protocol.Engine.WDMA,
                transfer_bytes=16384,
                element_format=FMT_FP16,
            ),
        ),
        rounds=iterations,
        schedule=schedule,
        seed=0x7600 + iterations * 0x10 + int(schedule),
        flags=ncc_protocol.DOUBLE_SLOT_OBSERVATION,
    )
    for iterations in (1, 2, 3, 4)
    for schedule in (
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    )
)
V2_HAZARD_SPECS = (
    (
        ncc_protocol.EffectRelation.RAW,
        ncc_protocol.Engine.RDMA,
        ncc_protocol.Engine.CT,
        ncc_protocol.Operand.WRITE,
        ncc_protocol.Operand.READ0,
    ),
    (
        ncc_protocol.EffectRelation.WAR,
        ncc_protocol.Engine.CT,
        ncc_protocol.Engine.TDMA,
        ncc_protocol.Operand.READ0,
        ncc_protocol.Operand.WRITE,
    ),
    (
        ncc_protocol.EffectRelation.WAW,
        ncc_protocol.Engine.RDMA,
        ncc_protocol.Engine.TDMA,
        ncc_protocol.Operand.WRITE,
        ncc_protocol.Operand.WRITE,
    ),
    (
        ncc_protocol.EffectRelation.RAR,
        ncc_protocol.Engine.CT,
        ncc_protocol.Engine.WDMA,
        ncc_protocol.Operand.READ0,
        ncc_protocol.Operand.READ0,
    ),
)
V2_HAZARD_CASES = tuple(
    v2_case(
        (
            f"{effect.name.lower()}-{first.name.lower()}-"
            f"{second.name.lower()}-{relation.name.lower()}-r2-"
            f"{schedule.name.lower()}"
        ),
        (
            v2_lane(
                first, transfer_bytes=4096, element_format=FMT_FP16
            ),
            v2_lane(
                second, transfer_bytes=4096, element_format=FMT_FP16
            ),
        ),
        rounds=2,
        schedule=schedule,
        seed=(
            0x4000
            + int(effect) * 0x100
            + int(relation) * 0x10
            + int(schedule)
        ),
        effect_relation=effect,
        range_relation=relation,
        first_operand=first_operand,
        second_operand=second_operand,
    )
    for effect, first, second, first_operand, second_operand
    in V2_HAZARD_SPECS
    for relation in (
        ncc_protocol.RangeRelation.EXACT,
        ncc_protocol.RangeRelation.PARTIAL,
        ncc_protocol.RangeRelation.ADJACENT,
    )
    for schedule in (
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    )
)
V2_HAZARD_PAIR_KEYS = {
    (first, second)
    for _, first, second, _, _ in V2_HAZARD_SPECS
}
V2_HAZARD_DISJOINT_CONTROLS = tuple(
    case
    for case in V2_PAIR_CASES
    if tuple(lane.engine for lane in case.plan.lanes)
    in V2_HAZARD_PAIR_KEYS
    and v2_disjoint_control_rounds_are_safe(
        (lane.engine.name.lower() for lane in case.plan.lanes),
        case.plan.rounds,
    )
)
V2_HAZARD_MANUAL_CASES = (
    V2_HAZARD_DISJOINT_CONTROLS + V2_HAZARD_CASES
)
FOCUSED_CASES = V2_SINGLE_CASES + (
    next(
        case
        for case in V2_PAIR_CASES
        if case.name == "ct-rdma-disjoint-r2-window"
    ),
    next(
        case
        for case in V2_PIPELINE_CASES
        if case.name == "rdma-ct-wdma-disjoint-r2-window"
    ),
)
CALIBRATION_CASES = (
    V2_SINGLE_CASES
    + V2_WORKER_CASES
    + V2_MULTI_ISSUE_CASES
    + V2_PAIR_CASES
    + V2_PIPELINE_CASES
)


def _unique_cases(
    *groups: tuple[GenericProbeCase, ...],
) -> tuple[GenericProbeCase, ...]:
    rows: list[GenericProbeCase] = []
    names: set[str] = set()
    for group in groups:
        for case in group:
            if case.name in names:
                continue
            names.add(case.name)
            rows.append(case)
    return tuple(rows)


BOARD_ALL_SAFE_CASES = _unique_cases(
    CALIBRATION_CASES,
    V2_HAZARD_MANUAL_CASES,
    V2_WAIT_OVERHEAD_CASES,
    V2_ISSUE_PATH_CASES,
    TDMA_CRT_MANUAL_CASES,
    V2_DMA_STRIDE_MATRIX_CASES,
    V2_PRODUCER_CONSUMER_ALL_CASES,
    V2_STRIDED_DEPENDENCY_CASES,
    V2_LARGE_BACKLOG_CASES,
    V2_DOUBLE_SLOT_OBSERVATION_CASES,
)
BOARD_ALL_PREFLIGHT_CASES = _unique_cases(
    BOARD_ALL_SAFE_CASES,
    V2_DOCUMENTED_DEPTH_CASES,
    V2_DEPTH_PLUS_ONE_CASES,
    V2_CONSTRUCTOR_CASES,
    V2_COMPLETION_SCOPE_CASES,
    V2_SUBSET_JOIN_CASES,
    V2_ACTIVE_OCCUPANCY_CASES,
    V2_MAPPED_SPM_BOUNDARY_CASES,
)
NCC_OBSERVATION_CASE_NAMES = frozenset(
    case.name
    for case in (
        V2_COMPLETION_SCOPE_CASES
        + V2_SUBSET_JOIN_CASES
        + V2_QUEUE_SATURATION_CASES
        + V2_WORKER_WAIT_SCOPE_CASES
        + V2_WORKER_SUBSET_SCOPE_CASES
        + V2_STRIDED_DEPENDENCY_CASES
        + V2_DOUBLE_SLOT_OBSERVATION_CASES
        + V2_MAPPED_SPM_BOUNDARY_CASES
    )
)
SUITES = {
    "qualification": QUALIFICATION_CASES,
    "focused": FOCUSED_CASES,
    "calibration": CALIBRATION_CASES,
    "board-all-safe": BOARD_ALL_SAFE_CASES,
    "board-all-preflight": BOARD_ALL_PREFLIGHT_CASES,
    "hazard-manual": V2_HAZARD_MANUAL_CASES,
    "constructor-observation": V2_CONSTRUCTOR_CASES,
    "completion-scope-manual": V2_COMPLETION_SCOPE_CASES,
    "active-occupancy-manual": V2_ACTIVE_OCCUPANCY_CASES,
    "cross-worker-boundary-manual": V2_SUBSET_JOIN_CASES,
    "queue-saturation-manual": V2_QUEUE_SATURATION_CASES,
    "worker-wait-scope-manual": V2_WORKER_WAIT_SCOPE_CASES,
    "worker-subset-scope-manual": V2_WORKER_SUBSET_SCOPE_CASES,
    "producer-consumer-observation": V2_PRODUCER_CONSUMER_ALL_CASES,
    "mapped-spm-boundary-observation": V2_MAPPED_SPM_BOUNDARY_CASES,
    "strided-dependency-observation": V2_STRIDED_DEPENDENCY_CASES,
    "large-backlog-observation": V2_LARGE_BACKLOG_CASES,
    "double-slot-observation": V2_DOUBLE_SLOT_OBSERVATION_CASES,
    "wait-overhead-manual": V2_WAIT_OVERHEAD_CASES,
    "issue-path-manual": V2_ISSUE_PATH_CASES,
    "tdma-crt-manual": TDMA_CRT_MANUAL_CASES,
    "dma-stride-matrix-manual": V2_DMA_STRIDE_MATRIX_CASES,
    "documented-depth-manual": V2_DOCUMENTED_DEPTH_CASES,
    "depth-plus-one-manual": V2_DEPTH_PLUS_ONE_CASES,
}
CASE_CATALOGS = {
    "no-card-protocol": (
        NO_CARD_PROTOCOL_CASES
        + V2_DMA_STRIDE_MATRIX_CASES
        + V2_CONSTRUCTOR_CASES
        + V2_COMPLETION_SCOPE_CASES
        + V2_ACTIVE_OCCUPANCY_CASES
        + V2_SUBSET_JOIN_CASES
        + V2_PRODUCER_CONSUMER_ALL_CASES
        + V2_MAPPED_SPM_BOUNDARY_CASES
        + V2_STRIDED_DEPENDENCY_CASES
        + V2_LARGE_BACKLOG_CASES
        + V2_DOUBLE_SLOT_OBSERVATION_CASES
    ),
    **SUITES,
}
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "constructor-return-address-nonnull": V2_CONSTRUCTOR_CASES,
    "constructor-builder-release": (
        V2_ADDITIONAL_BY_NAME["constructor-builder-release"],
        *V2_SINGLE_CASES,
    ),
    "execute-success-requires-side-effects": V2_SINGLE_CASES,
    "execute-invalid-type-return": (
        V2_ADDITIONAL_BY_NAME["execute-invalid-type-return"],
    ),
    "routing-five-engines-worker0": V2_SINGLE_CASES,
    "routing-ct-workers012": V2_WORKER_CASES,
    "range-materialization-ct-ne": tuple(
        case
        for case in V2_SINGLE_CASES
        if case.plan.lanes[0].engine
        in (ncc_protocol.Engine.CT, ncc_protocol.Engine.NE)
    ),
    "range-materialization-rdma-wdma-tdma": tuple(
        case
        for case in V2_SINGLE_CASES
        if case.plan.lanes[0].engine
        in (
            ncc_protocol.Engine.RDMA,
            ncc_protocol.Engine.WDMA,
            ncc_protocol.Engine.TDMA,
        )
    ),
    "dma-contiguous-64k": tuple(
        case
        for case in V2_LARGE_BACKLOG_CASES
        if "64k" in case.name
    ),
    "dma-1d-2d-3d-stride-holes": V2_DMA_STRIDE_MATRIX_CASES,
    "tdma-i8-whole-positive": (NO_CARD_PROTOCOL_CASES[0],),
    "tdma-native-bool-exclusion": (
        V2_ADDITIONAL_BY_NAME["tdma-native-bool-exclusion"],
    ),
    "tdma-bool-to-i8-physical-fill": (NO_CARD_PROTOCOL_CASES[1],),
    "five-engine-n1-n2-n4": V2_SINGLE_CASES + V2_MULTI_ISSUE_CASES,
    "documented-depth-manual": V2_DOCUMENTED_DEPTH_CASES,
    "depth-plus-one-manual": (
        V2_DEPTH_PLUS_ONE_CASES + V2_ACTIVE_OCCUPANCY_CASES
    ),
    "workers012-disjoint-routing": V2_WORKER_CASES,
    "default-byworker-wait-controls": V2_WORKER_CASES,
    "default-wait-nondefault-scope": V2_COMPLETION_SCOPE_CASES,
    "ten-engine-pairs-two-orders-controls": V2_PAIR_CASES,
    "large-backlog-compute-movement": V2_LARGE_BACKLOG_CASES,
    "double-slot-hardware-observation": V2_DOUBLE_SLOT_OBSERVATION_CASES,
    "dependency-raw-war-waw-rar": V2_HAZARD_CASES,
    "dependency-exact-partial-adjacent": V2_HAZARD_CASES,
    "dependency-strided-envelope": V2_STRIDED_DEPENDENCY_CASES,
    "wrapper-prebuilt-same-sequence": V2_ISSUE_PATH_CASES,
    "five-engine-wait-each-window": V2_WAIT_OVERHEAD_CASES,
    "ncc-producer-consumer-representative-positive": (
        V2_HAZARD_CASES + V2_DMA_STRIDE_MATRIX_CASES
        + V2_PRODUCER_CONSUMER_ALL_CASES
    ),
    "ncc-producer-consumer-all-directions": V2_MAPPED_SPM_BOUNDARY_CASES,
    "cross-worker-single-pair-triple-masks": V2_WORKER_CASES,
    "cross-worker-unjoined-boundary": V2_SUBSET_JOIN_CASES,
    "execute-engine-none-static-negative": tuple(
        case
        for case in V2_PROTOCOL_NEGATIVE_CASES
        if case.name == "execute-engine-none"
    ),
    "execute-worker-out-of-range-static-negative": tuple(
        case
        for case in V2_PROTOCOL_NEGATIVE_CASES
        if case.name == "execute-worker-out-of-range"
    ),
    "wait-mask-nonparticipant-static-negative": tuple(
        case
        for case in V2_PROTOCOL_NEGATIVE_CASES
        if case.name == "wait-mask-nonparticipant"
    ),
    "pmu-workload-delta-basis": V2_SINGLE_CASES + V2_PAIR_CASES,
    "scalar-csr-ordinary-issue-rejected": tuple(
        case
        for case in V2_PROTOCOL_NEGATIVE_CASES
        if case.name in ("execute-scalar-packet", "execute-csr-packet")
    ),
}


def validate_no_card_protocol_cases() -> None:
    expected_lanes = {
        "tdma-crt-i8-physical16": (FMT_INT8, 16),
        "tdma-crt-bool-to-i8-physical17": (FMT_BOOL, 17),
    }
    if {case.name for case in NO_CARD_PROTOCOL_CASES} != set(expected_lanes):
        raise RuntimeError("typed TDMA no-card catalog changed unexpectedly")
    for case in NO_CARD_PROTOCOL_CASES:
        plan = case.plan
        expected_format, expected_bytes = expected_lanes[case.name]
        if (
            len(plan.lanes) != 1
            or plan.lanes[0].engine != ncc_protocol.Engine.TDMA
            or plan.lanes[0].issue_mode != ncc_protocol.IssueMode.WRAPPER
            or plan.lanes[0].element_format != expected_format
            or plan.lanes[0].transfer_bytes != expected_bytes
        ):
            raise RuntimeError(
                f"{case.name}: no-card protocol case lost its typed lane"
            )
        plan.request_words()
    expected_envelopes = (36, 48, 84)
    for case, expected_envelope in zip(
        V2_DMA_STRIDE_MATRIX_CASES, expected_envelopes, strict=True
    ):
        plan = case.plan
        if (
            len(plan.lanes) != 2
            or tuple(lane.engine for lane in plan.lanes)
            != (ncc_protocol.Engine.RDMA, ncc_protocol.Engine.WDMA)
            or any(
                lane.issue_mode != ncc_protocol.IssueMode.WRAPPER
                or lane.element_format != FMT_FP16
                or lane.layout_kind
                != ncc_protocol.LayoutKind.DMA_STRIDED
                or lane.dma_envelope_bytes() != expected_envelope
                for lane in plan.lanes
            )
            or plan.rounds != 1
            or plan.schedule != ncc_protocol.Schedule.SERIAL
            or plan.effect_relation != ncc_protocol.EffectRelation.RAW
            or plan.range_relation != ncc_protocol.RangeRelation.EXACT
        ):
            raise RuntimeError(
                f"{case.name}: DMA stride matrix lost its serial roundtrip plan"
            )
        plan.request_words()
    if len(V2_SINGLE_CASES) != 5 or len(V2_MULTI_ISSUE_CASES) != 9:
        raise RuntimeError("generic catalog lost a single/multi-issue case")
    worker_specs = {
        "ct-worker0-raw-single": (
            (0,),
            ncc_protocol.Schedule.SERIAL,
            0b001,
        ),
        "ct-worker1-raw-single": (
            (1,),
            ncc_protocol.Schedule.SERIAL,
            0b010,
        ),
        "ct-worker2-raw-single": (
            (2,),
            ncc_protocol.Schedule.SERIAL,
            0b100,
        ),
        "ct-workers01-disjoint-r1-window": (
            (0, 1),
            ncc_protocol.Schedule.WINDOW,
            0b011,
        ),
        "ct-workers02-disjoint-r1-window": (
            (0, 2),
            ncc_protocol.Schedule.WINDOW,
            0b101,
        ),
        "ct-workers12-disjoint-r1-window": (
            (1, 2),
            ncc_protocol.Schedule.WINDOW,
            0b110,
        ),
        "ct-workers012-disjoint-r1-window": (
            (0, 1, 2),
            ncc_protocol.Schedule.WINDOW,
            0b111,
        ),
    }
    if {case.name for case in V2_WORKER_CASES} != set(worker_specs):
        raise RuntimeError("generic catalog lost a worker/join case")
    for case in V2_WORKER_CASES:
        plan = case.plan
        workers, schedule, wait_mask = worker_specs[case.name]
        if (
            tuple(lane.worker for lane in plan.lanes) != workers
            or any(
                lane.engine != ncc_protocol.Engine.CT
                or lane.issue_mode != ncc_protocol.IssueMode.RAW
                or lane.element_format != FMT_FP16
                or lane.transfer_bytes != 4096
                for lane in plan.lanes
            )
            or plan.rounds != 1
            or plan.schedule != schedule
            or plan.wait_kind != ncc_protocol.WaitKind.BY_WORKER
            or plan.wait_worker_mask != wait_mask
            or len(plan.issue_identities()) != len(workers)
        ):
            raise RuntimeError(
                f"{case.name}: worker/join protocol is malformed"
            )
        plan.request_words()
    completion_scope_specs = {
        "ne-worker1-depth6-default-wait-window": (
            ncc_protocol.WaitKind.DEFAULT,
            0x6121,
        ),
        "ne-worker1-depth6-byworker-wait-window": (
            ncc_protocol.WaitKind.BY_WORKER,
            0x6122,
        ),
        "ne-worker1-depth6-local-fence-wait-window": (
            ncc_protocol.WaitKind.LOCAL_FENCE,
            0x6123,
        ),
    }
    if {
        case.name for case in V2_COMPLETION_SCOPE_CASES
    } != set(completion_scope_specs):
        raise RuntimeError("generic catalog lost a completion-scope case")
    for case in V2_COMPLETION_SCOPE_CASES:
        wait_kind, seed = completion_scope_specs[case.name]
        plan = case.plan
        if (
            tuple(lane.worker for lane in plan.lanes) != (0, 1, 1)
            or plan.lanes[0].engine != ncc_protocol.Engine.TDMA
            or plan.lanes[0].issue_mode != ncc_protocol.IssueMode.RAW
            or plan.lanes[0].element_format != FMT_FP16
            or plan.lanes[0].transfer_bytes != 16
            or any(
                lane.engine != ncc_protocol.Engine.NE
                or lane.issue_mode != ncc_protocol.IssueMode.RAW
                or lane.element_format != FMT_FP16
                or lane.transfer_bytes != V2_NE_LARGE_RESULT_BYTES
                for lane in plan.lanes[1:]
            )
            or plan.rounds != 3
            or plan.schedule != ncc_protocol.Schedule.WINDOW
            or plan.wait_kind != wait_kind
            or plan.wait_worker_mask
            != (
                0b010
                if wait_kind == ncc_protocol.WaitKind.BY_WORKER
                else 0
            )
            or plan.seed != seed
            or plan.issue_order() != (0, 4, 8, 1, 5, 9, 2, 6, 10)
        ):
            raise RuntimeError(
                f"{case.name}: completion-scope protocol is malformed"
            )
        plan.request_words()
    if len(V2_CONSTRUCTOR_CASES) != 1 or not (
        V2_CONSTRUCTOR_CASES[0].plan.is_constructor_observation()
    ):
        raise RuntimeError("constructor observation catalog is malformed")
    if len(V2_SUBSET_JOIN_CASES) != 6 or {
        case.plan.wait_worker_mask for case in V2_SUBSET_JOIN_CASES
    } != {1, 2, 3, 4, 5, 6}:
        raise RuntimeError("subset-join catalog must cover six proper masks")
    if len(V2_PRODUCER_CONSUMER_CASES) != 5 or any(
        not case.plan.is_ordered_producer_consumer()
        for case in V2_PRODUCER_CONSUMER_CASES
    ):
        raise RuntimeError("ordered producer-consumer catalog is malformed")
    if len(V2_KCORE_BOUNDARY_CASES) != 2:
        raise RuntimeError("Kcore completion boundary catalog is malformed")
    mapped_pairs = (
        (
            V2_MAPPED_SPM_PURE_NCC_CASES,
            ncc_protocol.REQ["SCHEDULE"],
        ),
        (
            V2_MAPPED_SPM_NCC_TO_KCORE_CASES,
            ncc_protocol.REQ["WAIT_KIND"],
        ),
        (
            V2_MAPPED_SPM_KCORE_TO_NCC_CASES,
            ncc_protocol.REQ["FLAGS"],
        ),
    )
    if len(V2_MAPPED_SPM_BOUNDARY_CASES) != 6:
        raise RuntimeError("mapped-SPM boundary catalog must contain three A/B pairs")
    for pair, changed_word in mapped_pairs:
        if len(pair) != 2:
            raise RuntimeError("mapped-SPM boundary comparison lost an A/B case")
        first_words = pair[0].plan.request_words()
        second_words = pair[1].plan.request_words()
        differences = {
            index
            for index, (first, second) in enumerate(
                zip(first_words, second_words, strict=True)
            )
            if first != second
        }
        if differences != {changed_word}:
            raise RuntimeError(
                "mapped-SPM A/B requests must change exactly one typed field"
            )
    terminal_only, serial_control = V2_MAPPED_SPM_PURE_NCC_CASES
    if (
        terminal_only.plan.issue_order() != (0, 4, 8)
        or serial_control.plan.issue_order() != (0, 4, 8)
        or terminal_only.plan.wait_kind
        != ncc_protocol.WaitKind.BY_WORKER
        or serial_control.plan.wait_kind
        != ncc_protocol.WaitKind.BY_WORKER
    ):
        raise RuntimeError("pure NCC mapped-SPM chain lost its terminal wait")
    no_wait, local_wait = V2_MAPPED_SPM_NCC_TO_KCORE_CASES
    if (
        no_wait.plan.wait_kind != ncc_protocol.WaitKind.NONE
        or local_wait.plan.wait_kind
        != ncc_protocol.WaitKind.LOCAL_FENCE
        or no_wait.plan.wait_worker_mask != 0
        or local_wait.plan.wait_worker_mask != 0
        or any(
            not case.plan.is_tight_kcore_boundary_observation()
            or case.plan.flags != ncc_protocol.TIGHT_KCORE_BOUNDARY
            or case.plan.issue_limit != 0
            or case.plan.issue_order() != (0, 1, 2, 3)
            or len(case.plan.lanes) != 1
            or any(
                lane.engine != ncc_protocol.Engine.NE
                or lane.worker != 0
                or lane.issue_mode != ncc_protocol.IssueMode.RAW
                or lane.element_format != FMT_FP16
                or lane.transfer_bytes != V2_NE_LARGE_RESULT_BYTES
                for lane in case.plan.lanes
            )
            for case in V2_MAPPED_SPM_NCC_TO_KCORE_CASES
        )
    ):
        raise RuntimeError("NCC-to-Kcore mapped-SPM wait control is malformed")
    ordered_store, preissue_wait = V2_MAPPED_SPM_KCORE_TO_NCC_CASES
    if (
        not ordered_store.plan.is_mapped_spm_kcore_write_observation()
        or not preissue_wait.plan.is_mapped_spm_kcore_write_observation()
        or ordered_store.plan.flags
        != ncc_protocol.MAPPED_SPM_KCORE_WRITE
        or preissue_wait.plan.flags
        != (
            ncc_protocol.MAPPED_SPM_KCORE_WRITE
            | ncc_protocol.PREISSUE_LOCAL_WAIT
        )
    ):
        raise RuntimeError("Kcore-to-NCC mapped-SPM wait control is malformed")
    strided_effect_counts = {
        effect: sum(
            case.plan.effect_relation == effect
            for case in V2_STRIDED_DEPENDENCY_CASES
        )
        for effect in (
            ncc_protocol.EffectRelation.RAW,
            ncc_protocol.EffectRelation.WAR,
            ncc_protocol.EffectRelation.WAW,
            ncc_protocol.EffectRelation.RAR,
        )
    }
    if (
        len(V2_STRIDED_DEPENDENCY_CASES) != 36
        or strided_effect_counts
        != {
            ncc_protocol.EffectRelation.RAW: 6,
            ncc_protocol.EffectRelation.WAR: 6,
            ncc_protocol.EffectRelation.WAW: 18,
            ncc_protocol.EffectRelation.RAR: 6,
        }
        or any(
            not case.plan.is_strided_dependency_observation()
            for case in V2_STRIDED_DEPENDENCY_CASES
        )
    ):
        raise RuntimeError("strided dependency catalog is malformed")
    if (
        len(V2_LARGE_BACKLOG_CASES) != 12
        or len(V2_LARGE_BACKLOG_SINGLE_CASES) != 8
        or len(V2_LARGE_OVERLAP_CASES) != 4
        or not any(
            lane.transfer_bytes == 65536
            for case in V2_LARGE_BACKLOG_CASES
            for lane in case.plan.lanes
        )
        or not any(
            v2_is_large_ne_lane(lane)
            for case in V2_LARGE_BACKLOG_CASES
            for lane in case.plan.lanes
        )
    ):
        raise RuntimeError("large backlog catalog lost movement/NE coverage")
    if len(V2_DOUBLE_SLOT_OBSERVATION_CASES) != 8 or any(
        not case.plan.is_double_slot_observation()
        for case in V2_DOUBLE_SLOT_OBSERVATION_CASES
    ):
        raise RuntimeError("double-slot hardware observation is malformed")
    for case in BOARD_ALL_PREFLIGHT_CASES:
        case.plan.request_words()
    overhead_specs = {
        f"{engine.name.lower()}-worker0-r2-{spelling}": schedule
        for engine in V2_ENGINES
        for spelling, schedule in (
            ("wait-each", ncc_protocol.Schedule.SERIAL),
            ("wait-once", ncc_protocol.Schedule.WINDOW),
        )
    }
    if {
        case.name for case in V2_WAIT_OVERHEAD_CASES
    } != set(overhead_specs):
        raise RuntimeError("generic catalog lost a wait-overhead case")
    for case in V2_WAIT_OVERHEAD_CASES:
        plan = case.plan
        if (
            len(plan.lanes) != 1
            or any(
                lane.worker != 0
                or lane.issue_mode != ncc_protocol.IssueMode.RAW
                for lane in plan.lanes
            )
            or plan.rounds != 2
            or plan.schedule != overhead_specs[case.name]
            or plan.wait_kind != ncc_protocol.WaitKind.BY_WORKER
            or plan.wait_worker_mask != 1
            or plan.issue_order() != (0, 1)
        ):
            raise RuntimeError(
                f"{case.name}: wait-overhead protocol is malformed"
            )
        plan.request_words()
    for engine, depth in V2_DOCUMENTED_QUEUE_DEPTHS.items():
        issue_counts = {
            len(case.plan.issue_identities())
            for case in V2_MULTI_ISSUE_CASES
            if case.plan.lanes[0].engine == engine
        }
        if issue_counts != (
            {2}
            if engine == ncc_protocol.Engine.TDMA
            else {2, 4}
        ):
            raise RuntimeError(
                f"{engine.name}: generic catalog must cover ordinary "
                "N=2 and the engine-safe N=4 boundary"
            )
        depth_cases = [
            case
            for case in V2_DOCUMENTED_DEPTH_CASES
            if case.plan.lanes[0].engine == engine
        ]
        if len(depth_cases) != 1 or len(
            depth_cases[0].plan.issue_identities()
        ) != depth:
            raise RuntimeError(
                f"{engine.name}: documented-depth catalog is malformed"
            )
        plus_one_cases = [
            case
            for case in V2_DEPTH_PLUS_ONE_CASES
            if case.plan.lanes[0].engine == engine
        ]
        if (
            len(plus_one_cases) != 1
            or len(plus_one_cases[0].plan.issue_identities()) != depth + 1
            or plus_one_cases[0].plan.wait_kind
            != ncc_protocol.WaitKind.BY_WORKER
        ):
            raise RuntimeError(
                f"{engine.name}: depth-plus-one catalog is malformed"
            )
    if (
        len(V2_ACTIVE_OCCUPANCY_CASES) != len(V2_ENGINES)
        or {
            case.plan.lanes[0].engine
            for case in V2_ACTIVE_OCCUPANCY_CASES
        }
        != set(V2_ENGINES)
    ):
        raise RuntimeError(
            "active-occupancy catalog must cover every NCC engine"
        )
    for case in V2_ACTIVE_OCCUPANCY_CASES:
        occupancy = case.plan
        engine = occupancy.lanes[0].engine
        depth = V2_DOCUMENTED_QUEUE_DEPTHS[engine]
        if (
            tuple(lane.engine for lane in occupancy.lanes)
            != (engine, engine)
            or any(
                lane.worker != 0
                or lane.issue_mode != ncc_protocol.IssueMode.RAW
                or lane.transfer_bytes
                != V2_ACTIVE_OCCUPANCY_BYTES[engine]
                for lane in occupancy.lanes
            )
            or occupancy.flags != ncc_protocol.TIGHT_DEPTH_PLUS_ONE
            or occupancy.issue_limit != depth + 1
            or len(occupancy.issue_identities()) != depth + 1
        ):
            raise RuntimeError(
                f"{engine.name}: active-occupancy tight plan is malformed"
            )
        occupancy.request_words()
    if len(V2_PAIR_CASES) != 64:
        raise RuntimeError("generic catalog lost a disjoint pair control")
    for first, second in itertools.permutations(V2_ENGINES, 2):
        schedules = {
            (case.plan.rounds, case.plan.schedule)
            for case in V2_PAIR_CASES
            if tuple(lane.engine for lane in case.plan.lanes)
            == (first, second)
        }
        expected_rounds = (
            (2,)
            if ncc_protocol.Engine.TDMA in (first, second)
            else (2, 4)
        )
        if schedules != {
            (rounds, schedule)
            for rounds in expected_rounds
            for schedule in (
                ncc_protocol.Schedule.SERIAL,
                ncc_protocol.Schedule.WINDOW,
            )
        }:
            raise RuntimeError(
                "generic catalog lost an oriented disjoint pair control: "
                f"{first.name.lower()}->{second.name.lower()}"
            )
    if len(V2_PIPELINE_CASES) != 4:
        raise RuntimeError("generic catalog lost the three-lane controls")
    if len(V2_HAZARD_CASES) != 24:
        raise RuntimeError("generic catalog lost a typed hazard control")
    if len(V2_ISSUE_PATH_CASES) != 10:
        raise RuntimeError("generic catalog lost a raw/wrapper issue-path case")
    for engine in V2_ENGINES:
        modes = {
            case.plan.lanes[0].issue_mode
            for case in V2_ISSUE_PATH_CASES
            if case.plan.lanes[0].engine == engine
        }
        if modes != {
            ncc_protocol.IssueMode.RAW,
            ncc_protocol.IssueMode.WRAPPER,
        }:
            raise RuntimeError(
                f"{engine.name}: issue path lacks a raw/wrapper control"
            )
    real_objects = {
        id(case)
        for case in (
            NO_CARD_PROTOCOL_CASES
            + V2_DMA_STRIDE_MATRIX_CASES
            + tuple(
                case
                for cases in SUITES.values()
                for case in cases
            )
            + V2_DEFERRED_CASES
            + V2_PROTOCOL_NEGATIVE_CASES
            + V2_ADDITIONAL_DISPOSITIONS
        )
    }
    if not CALIBRATION_LEAF_BINDINGS or any(
        not cases or any(id(case) not in real_objects for case in cases)
        for cases in CALIBRATION_LEAF_BINDINGS.values()
    ):
        raise RuntimeError(
            "NCC calibration leaf bindings must reference real catalog objects"
        )


def validate_catalog_resource_layout(
    cases: Iterable[GenericProbeCase],
) -> None:
    for case in cases:
        occupied: list[tuple[int, int]] = []
        for identity in case.plan.issue_identities():
            lane = case.plan.lanes[identity.lane]
            span = lane.dma_envelope_bytes()
            begin = (
                V2_OUTPUT_SLOT_BASE
                + identity.slot * V2_OUTPUT_SLOT_STRIDE
            )
            end = begin + 2 * V2_OUTPUT_GUARD_BYTES + span
            if end > RESOURCE_BYTES:
                raise RuntimeError(
                    f"{case.name}: slot {identity.slot} exceeds the "
                    "shared DDR resource"
                )
            if any(begin < other_end and other_begin < end
                   for other_begin, other_end in occupied):
                raise RuntimeError(
                    f"{case.name}: slot {identity.slot} overlaps another "
                    "shared DDR slot"
                )
            occupied.append((begin, end))


def write_request(
    path: pathlib.Path, case: GenericProbeCase, sample: int
) -> None:
    payload = bytearray([0xD3] * RESOURCE_BYTES)
    encoded = struct.pack(
        f"<{ncc_protocol.REQUEST_WORDS}Q", *case.request_words(sample)
    )
    payload[: len(encoded)] = encoded
    path.write_bytes(payload)


def v2_pattern_byte(slot: int, index: int) -> int:
    return ((slot + 1) * 29 + index * 17) & 0xFF


def v2_scatter_compact(
    buffer: bytearray,
    begin: int,
    lane: ncc_protocol.Lane,
    compact: bytes,
) -> None:
    if lane.layout_kind != ncc_protocol.LayoutKind.DMA_STRIDED:
        buffer[begin : begin + len(compact)] = compact
        return
    cursor = 0
    for offset in lane.dma_chunk_offsets():
        chunk_end = cursor + lane.layout_inner_bytes
        buffer[
            begin + offset : begin + offset + lane.layout_inner_bytes
        ] = compact[cursor:chunk_end]
        cursor = chunk_end
    if cursor != len(compact):
        raise RuntimeError("DMA stride descriptor did not consume compact bytes")


def write_payload(path: pathlib.Path, case: GenericProbeCase) -> None:
    payload = bytearray([0xCC] * RESOURCE_BYTES)
    for identity in case.plan.issue_identities():
        lane = case.plan.lanes[identity.lane]
        begin = (
            identity.slot * V2_OUTPUT_SLOT_STRIDE + V2_OUTPUT_GUARD_BYTES
        )
        if (
            case.plan.is_double_slot_observation()
            and lane.engine == ncc_protocol.Engine.RDMA
        ):
            value = struct.pack("<H", v2_half(identity.round + 1))
            payload[begin : begin + lane.transfer_bytes] = value * (
                lane.transfer_bytes // len(value)
            )
            continue
        if lane.layout_kind == ncc_protocol.LayoutKind.DMA_STRIDED:
            if lane.engine == ncc_protocol.Engine.RDMA:
                compact = bytes(
                    v2_pattern_byte(identity.slot, index)
                    for index in range(lane.transfer_bytes)
                )
                v2_scatter_compact(payload, begin, lane, compact)
            continue
        if (
            case.plan.effect_relation
            != ncc_protocol.EffectRelation.NONE
            and identity.lane == 0
            and lane.engine == ncc_protocol.Engine.RDMA
        ):
            value = struct.pack("<H", v2_half(4 + identity.round))
            payload[begin : begin + lane.transfer_bytes] = value * (
                lane.transfer_bytes // len(value)
            )
        else:
            payload[begin : begin + lane.transfer_bytes] = bytes(
                v2_pattern_byte(identity.slot, index)
                for index in range(lane.transfer_bytes)
            )
    path.write_bytes(payload)


def v2_half(value: int) -> int:
    return ncc_protocol.F16_POSITIVE_INTEGERS[value]


def v2_completion_marker(
    plan: ncc_protocol.Plan,
) -> tuple[int, int]:
    last_slot = plan.issue_order()[-1]
    last_identity = next(
        identity
        for identity in plan.issue_identities()
        if identity.slot == last_slot
    )
    lane = plan.lanes[last_identity.lane]
    write = v2_operand_spm_address(
        plan, last_identity, ncc_protocol.Operand.WRITE
    )
    if lane.engine == ncc_protocol.Engine.RDMA:
        return (
            v2_pattern_byte(last_slot, lane.transfer_bytes - 1),
            write + lane.transfer_bytes - 1,
        )
    if lane.engine == ncc_protocol.Engine.NE:
        return (
            v2_half(last_slot + 1) >> 8,
            write + v2_ne_result_bytes(lane) - 1,
        )
    raise RuntimeError(
        "completion-scope marker requires a result-producing final lane"
    )


def v2_hazard_ranges(
    plan: ncc_protocol.Plan, round_index: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    pair_base = V2_SPM_SLOT_BASE + round_index * V2_SPM_SLOT_STRIDE
    byte_count = plan.lanes[0].transfer_bytes
    first = (
        pair_base + V2_HAZARD_SELECTED_OFFSET,
        pair_base + V2_HAZARD_SELECTED_OFFSET + byte_count,
    )
    if plan.range_relation == ncc_protocol.RangeRelation.EXACT:
        second_begin = first[0]
    elif plan.range_relation == ncc_protocol.RangeRelation.PARTIAL:
        second_begin = first[1] - byte_count // 2
    elif plan.range_relation == ncc_protocol.RangeRelation.ADJACENT:
        second_begin = first[1]
    else:
        raise RuntimeError("hazard plan has no concrete alias relation")
    return first, (second_begin, second_begin + byte_count)


def v2_strided_dependency_bases(
    plan: ncc_protocol.Plan,
) -> tuple[int, int]:
    if not plan.is_strided_dependency_observation():
        raise RuntimeError("plan is not a strided dependency observation")
    lane = plan.lanes[0]
    first = V2_SPM_SLOT_BASE + V2_SPM_WRITE_OFFSET
    if plan.range_relation == ncc_protocol.RangeRelation.PARTIAL:
        shift = lane.transfer_bytes // 2
    elif plan.range_relation == ncc_protocol.RangeRelation.ADJACENT:
        shift = lane.dma_envelope_bytes()
    else:
        shift = 0
    return first, first + shift


def v2_strided_compact_index(
    lane: ncc_protocol.Lane, base: int, address: int
) -> int | None:
    """Map an address in the strided DDR descriptor to compact payload."""
    relative = address - base
    if relative < 0:
        return None
    for chunk, offset in enumerate(lane.dma_chunk_offsets()):
        if offset <= relative < offset + lane.layout_inner_bytes:
            return (
                chunk * lane.layout_inner_bytes + relative - offset
            )
    return None


def v2_dma_local_compact_index(
    lane: ncc_protocol.Lane, base: int, address: int
) -> int | None:
    """Map the compact SPM endpoint of an RDMA/WDMA descriptor."""
    relative = address - base
    return relative if 0 <= relative < lane.transfer_bytes else None


def v2_strided_final_byte(
    plan: ncc_protocol.Plan, address: int
) -> int:
    lane = plan.lanes[0]
    first_base, second_base = v2_strided_dependency_bases(plan)
    first_index = v2_dma_local_compact_index(lane, first_base, address)
    second_index = v2_dma_local_compact_index(lane, second_base, address)
    if plan.effect_relation == ncc_protocol.EffectRelation.RAW:
        return (
            v2_pattern_byte(0, first_index)
            if first_index is not None
            else 0xC3
        )
    if plan.effect_relation == ncc_protocol.EffectRelation.WAR:
        return (
            v2_pattern_byte(ncc_protocol.MAX_ROUNDS, second_index)
            if second_index is not None
            else 0xC3
        )
    if plan.effect_relation == ncc_protocol.EffectRelation.WAW:
        if second_index is not None:
            return v2_pattern_byte(
                ncc_protocol.MAX_ROUNDS, second_index
            )
        return (
            v2_pattern_byte(0, first_index)
            if first_index is not None
            else 0xC3
        )
    if plan.effect_relation == ncc_protocol.EffectRelation.RAR:
        selected_index = (
            first_index if first_index is not None else second_index
        )
        return (
            v2_pattern_byte(
                V2_STRIDED_INITIAL_SOURCE_SLOT, selected_index
            )
            if selected_index is not None
            else 0xC3
        )
    raise RuntimeError("unsupported strided dependency effect")


def v2_strided_expected_result(
    identity: ncc_protocol.IssueIdentity,
    lane: ncc_protocol.Lane,
    plan: ncc_protocol.Plan,
) -> bytes:
    if lane.engine == ncc_protocol.Engine.WDMA:
        source_slot = {
            ncc_protocol.EffectRelation.RAW: 0,
            ncc_protocol.EffectRelation.WAR: (
                V2_STRIDED_INITIAL_SOURCE_SLOT
            ),
            ncc_protocol.EffectRelation.RAR: (
                V2_STRIDED_INITIAL_SOURCE_SLOT
            ),
        }.get(plan.effect_relation)
        if source_slot is None:
            raise RuntimeError("unsupported strided WDMA sink")
        return bytes(
            v2_pattern_byte(source_slot, index)
            for index in range(lane.transfer_bytes)
        )
    if lane.engine != ncc_protocol.Engine.RDMA:
        raise RuntimeError("strided dependency requires RDMA/WDMA lanes")
    base = v2_strided_dependency_bases(plan)[identity.lane]
    return bytes(
        v2_strided_final_byte(plan, base + byte)
        for byte in range(lane.transfer_bytes)
    )


def v2_strided_dependency_evidence(
    plan: ncc_protocol.Plan,
) -> dict[str, object]:
    first_base, second_base = v2_strided_dependency_bases(plan)
    lane = plan.lanes[0]
    first_bytes = set(range(first_base, first_base + lane.transfer_bytes))
    second_bytes = set(
        range(second_base, second_base + lane.transfer_bytes)
    )
    envelope = lane.dma_envelope_bytes()
    envelope_overlap = max(
        0,
        min(first_base + envelope, second_base + envelope)
        - max(first_base, second_base),
    )
    exact_waw = (
        plan.effect_relation == ncc_protocol.EffectRelation.WAW
        and plan.range_relation == ncc_protocol.RangeRelation.EXACT
    )
    return {
        "effect": plan.effect_relation.name.lower(),
        "range": plan.range_relation.name.lower(),
        "independent_ddr_sinks": 2,
        "scatter_holes_checked": True,
        "spm_envelope_and_guards_checked": True,
        "packet_envelope_overlap_bytes": envelope_overlap,
        "selected_overlap_bytes": len(first_bytes & second_bytes),
        "first_only_bytes": len(first_bytes - second_bytes),
        "first_payload_independently_proven": not exact_waw,
        "command_semantics_sufficient": True,
        "evidence": (
            "dual-rdma-count-plus-second-wins-final; "
            "first-payload-not-independently-observable"
            if exact_waw
            else {
                ncc_protocol.EffectRelation.RAW: (
                    "producer-final-copy-and-dependent-read-sink"
                ),
                ncc_protocol.EffectRelation.WAR: (
                    "pre-write-read-sink-and-post-write-final-copy"
                ),
                ncc_protocol.EffectRelation.WAW: (
                    "first-only-and-second-wins-final-regions"
                ),
                ncc_protocol.EffectRelation.RAR: (
                    "two-independent-read-sinks"
                ),
            }[plan.effect_relation]
        ),
    }


def v2_waw_evidence(plan: ncc_protocol.Plan) -> dict[str, object]:
    if plan.effect_relation != ncc_protocol.EffectRelation.WAW:
        raise RuntimeError("WAW evidence requires a WAW plan")
    exact = plan.range_relation == ncc_protocol.RangeRelation.EXACT
    return {
        "two_write_instruction_counts_required": True,
        "second_wins_final_checked": True,
        "first_payload_independently_proven": not exact,
        "command_semantics_sufficient": True,
        "evidence": (
            "dual-write-count-plus-second-wins-final; "
            "first-payload-not-independently-observable"
            if exact
            else "first-only-and-second-wins-final-regions"
        ),
    }


def v2_hazard_source_value(
    plan: ncc_protocol.Plan,
    round_index: int,
    address: int,
    *,
    after_second_write: bool,
) -> int:
    first, second = v2_hazard_ranges(plan, round_index)
    if not (
        min(first[0], second[0])
        <= address
        < max(first[1], second[1])
    ):
        raise RuntimeError("hazard oracle address is outside its footprint")
    value = 2 + round_index
    if (
        plan.first_operand == ncc_protocol.Operand.WRITE
        and first[0] <= address < first[1]
    ):
        value = 4 + round_index
    if (
        after_second_write
        and plan.second_operand == ncc_protocol.Operand.WRITE
        and second[0] <= address < second[1]
    ):
        value = 8 + round_index
    return value


def v2_hazard_result(
    identity: ncc_protocol.IssueIdentity,
    lane: ncc_protocol.Lane,
    plan: ncc_protocol.Plan,
) -> bytes:
    first, second = v2_hazard_ranges(plan, identity.round)
    selected = first if identity.lane == 0 else second
    if lane.engine == ncc_protocol.Engine.CT:
        values = (
            v2_hazard_source_value(
                plan,
                identity.round,
                selected[0] + index * 2,
                after_second_write=False,
            )
            + 1
            for index in range(lane.transfer_bytes // 2)
        )
    elif lane.engine == ncc_protocol.Engine.WDMA:
        values = (
            v2_hazard_source_value(
                plan,
                identity.round,
                selected[0] + index * 2,
                after_second_write=False,
            )
            for index in range(lane.transfer_bytes // 2)
        )
    elif lane.engine in (
        ncc_protocol.Engine.RDMA,
        ncc_protocol.Engine.TDMA,
    ):
        values = (
            v2_hazard_source_value(
                plan,
                identity.round,
                selected[0] + index * 2,
                after_second_write=True,
            )
            for index in range(lane.transfer_bytes // 2)
        )
    else:
        raise RuntimeError(
            f"hazard oracle does not support {lane.engine.name}"
        )
    return struct.pack(
        f"<{lane.transfer_bytes // 2}H",
        *(v2_half(value) for value in values),
    )


def v2_expected_result(
    identity: ncc_protocol.IssueIdentity,
    lane: ncc_protocol.Lane,
    plan: ncc_protocol.Plan,
) -> bytes:
    if plan.is_strided_dependency_observation():
        return v2_strided_expected_result(identity, lane, plan)
    if lane.layout_kind == ncc_protocol.LayoutKind.DMA_STRIDED:
        source_slot = identity.round
        return bytes(
            v2_pattern_byte(source_slot, index)
            for index in range(lane.transfer_bytes)
        )
    if plan.is_double_slot_observation():
        value = v2_half(
            identity.round + (
                1 if lane.engine == ncc_protocol.Engine.RDMA else 2
            )
        )
        return struct.pack(
            f"<{lane.transfer_bytes // 2}H",
            *([value] * (lane.transfer_bytes // 2)),
        )
    if plan.is_ordered_producer_consumer():
        producer = plan.lanes[0].engine
        if identity.lane == 0:
            value = {
                ncc_protocol.Engine.RDMA: 4,
                ncc_protocol.Engine.TDMA: 4,
                ncc_protocol.Engine.CT: 2,
                ncc_protocol.Engine.NE: 1,
            }[lane.engine]
        else:
            value = {
                ncc_protocol.Engine.CT: 5,
                ncc_protocol.Engine.NE: 4,
                ncc_protocol.Engine.WDMA: (
                    1 if producer == ncc_protocol.Engine.NE else 2
                ),
            }[lane.engine]
        result_bytes = (
            v2_ne_result_bytes(lane)
            if lane.engine == ncc_protocol.Engine.NE
            else lane.transfer_bytes
        )
        return struct.pack(
            f"<{result_bytes // 2}H",
            *([v2_half(value)] * (result_bytes // 2)),
        )
    if plan.effect_relation != ncc_protocol.EffectRelation.NONE:
        return v2_hazard_result(identity, lane, plan)
    if lane.engine == ncc_protocol.Engine.CT:
        return struct.pack(
            f"<{lane.transfer_bytes // 2}H",
            *([v2_half(identity.slot + 2)] * (lane.transfer_bytes // 2)),
        )
    if lane.engine == ncc_protocol.Engine.NE:
        return struct.pack(
            f"<{v2_ne_result_bytes(lane) // 2}H",
            *(
                [v2_half(identity.slot + 1)]
                * (v2_ne_result_bytes(lane) // 2)
            ),
        )
    if lane.engine in (
        ncc_protocol.Engine.RDMA,
        ncc_protocol.Engine.WDMA,
    ):
        return bytes(
            v2_pattern_byte(identity.slot, index)
            for index in range(lane.transfer_bytes)
        )
    if lane.element_format == FMT_INT8:
        return bytes([0x31 + identity.slot * 7]) * lane.transfer_bytes
    if lane.element_format == FMT_BOOL:
        return bytes([0xFF]) * lane.transfer_bytes
    if lane.element_format == FMT_FP16:
        value = struct.pack("<H", v2_half(identity.slot + 1))
    elif lane.element_format == FMT_BF16:
        value = struct.pack("<H", 0x3F80)
    else:
        raise RuntimeError(f"unsupported TDMA format {lane.element_format}")
    repeats, remainder = divmod(lane.transfer_bytes, len(value))
    return value * repeats + value[:remainder]


def v2_spm_address(slot: int, offset: int) -> int:
    return V2_SPM_SLOT_BASE + slot * V2_SPM_SLOT_STRIDE + offset


def v2_is_large_ne_lane(lane: ncc_protocol.Lane) -> bool:
    return (
        lane.engine == ncc_protocol.Engine.NE
        and lane.transfer_bytes == V2_NE_LARGE_RESULT_BYTES
    )


def v2_is_scope_ne_lane(
    plan: ncc_protocol.Plan, lane: ncc_protocol.Lane
) -> bool:
    return (
        lane.engine == ncc_protocol.Engine.NE
        and lane.transfer_bytes == V2_NE_SCOPE_RESULT_BYTES
        and plan.flags == ncc_protocol.TIGHT_WORKER_SCOPE
    )


def v2_ne_lhs_bytes(
    plan: ncc_protocol.Plan, lane: ncc_protocol.Lane
) -> int:
    if v2_is_scope_ne_lane(plan, lane):
        return V2_NE_SCOPE_LHS_BYTES
    if v2_is_large_ne_lane(lane):
        return V2_NE_LARGE_LHS_BYTES
    return V2_NE_PHYSICAL_BYTES


def v2_ne_rhs_bytes(
    plan: ncc_protocol.Plan, lane: ncc_protocol.Lane
) -> int:
    if v2_is_scope_ne_lane(plan, lane):
        return V2_NE_SCOPE_RHS_BYTES
    if v2_is_large_ne_lane(lane):
        return V2_NE_LARGE_RHS_BYTES
    return V2_NE_RHS_BYTES


def v2_ne_result_bytes(lane: ncc_protocol.Lane) -> int:
    if v2_is_large_ne_lane(lane):
        return V2_NE_LARGE_RESULT_BYTES
    return V2_NE_RESULT_BYTES


def v2_ne_output_span(lane: ncc_protocol.Lane) -> int:
    if v2_is_large_ne_lane(lane):
        return V2_NE_LARGE_RESULT_BYTES
    return V2_NE_PHYSICAL_BYTES


def v2_ne_read1_offset(
    plan: ncc_protocol.Plan, lane: ncc_protocol.Lane
) -> int:
    if v2_is_scope_ne_lane(plan, lane):
        return V2_NE_SCOPE_READ1_OFFSET
    if v2_is_large_ne_lane(lane):
        return V2_NE_LARGE_READ1_OFFSET
    return V2_SPM_READ1_OFFSET


def v2_ne_write_offset(
    plan: ncc_protocol.Plan, lane: ncc_protocol.Lane
) -> int:
    if v2_is_scope_ne_lane(plan, lane):
        return V2_NE_SCOPE_WRITE_OFFSET
    if v2_is_large_ne_lane(lane):
        return V2_NE_LARGE_WRITE_OFFSET
    return V2_SPM_WRITE_OFFSET


def v2_operand_uses_spm(
    engine: ncc_protocol.Engine, operand: ncc_protocol.Operand
) -> bool:
    if operand == ncc_protocol.Operand.READ1:
        return engine in (ncc_protocol.Engine.CT, ncc_protocol.Engine.NE)
    if operand == ncc_protocol.Operand.READ0:
        return engine not in (
            ncc_protocol.Engine.RDMA,
            ncc_protocol.Engine.TDMA,
        )
    if operand == ncc_protocol.Operand.WRITE:
        return engine != ncc_protocol.Engine.WDMA
    return False


def v2_operand_spm_address(
    plan: ncc_protocol.Plan,
    identity: ncc_protocol.IssueIdentity,
    operand: ncc_protocol.Operand,
) -> int:
    lane = plan.lanes[identity.lane]
    if plan.is_double_slot_observation():
        physical_slot = identity.round & 1
        if lane.engine == ncc_protocol.Engine.RDMA:
            return v2_spm_address(physical_slot, V2_SPM_READ0_OFFSET)
        if lane.engine == ncc_protocol.Engine.CT:
            return v2_spm_address(
                physical_slot,
                (
                    V2_SPM_READ0_OFFSET,
                    V2_SPM_READ1_OFFSET,
                    V2_SPM_WRITE_OFFSET,
                )[operand],
            )
        if lane.engine == ncc_protocol.Engine.WDMA:
            return v2_spm_address(physical_slot, V2_SPM_WRITE_OFFSET)
    if (
        plan.is_ordered_producer_consumer()
        and identity.lane == 1
        and operand == ncc_protocol.Operand.READ0
    ):
        first = next(
            item for item in plan.issue_identities() if item.lane == 0
        )
        return v2_operand_spm_address(
            plan, first, ncc_protocol.Operand.WRITE
        )
    if plan.is_ordered_producer_consumer():
        offsets = (
            V2_SPM_READ0_OFFSET,
            (
                v2_ne_read1_offset(plan, lane)
                if lane.engine == ncc_protocol.Engine.NE
                else V2_SPM_READ1_OFFSET
            ),
            (
                v2_ne_write_offset(plan, lane)
                if lane.engine == ncc_protocol.Engine.NE
                else V2_SPM_WRITE_OFFSET
            ),
        )
        return v2_spm_address(identity.slot, offsets[operand])
    if (
        plan.is_serial_dma_roundtrip()
        and operand
        == (
            plan.first_operand
            if identity.lane == 0
            else plan.second_operand
        )
    ):
        # The RDMA destination and WDMA source are one shared physical
        # envelope.  This is intentionally the ordinary slot-0 write buffer,
        # not the generic hazard-composition offset used by compute hazards.
        return v2_spm_address(0, V2_SPM_WRITE_OFFSET)
    if (
        plan.is_strided_dependency_observation()
        and operand
        == (
            plan.first_operand
            if identity.lane == 0
            else plan.second_operand
        )
    ):
        return v2_strided_dependency_bases(plan)[identity.lane]
    if plan.effect_relation != ncc_protocol.EffectRelation.NONE:
        selected = (
            plan.first_operand
            if identity.lane == 0
            else plan.second_operand
        )
        if operand == selected:
            return v2_hazard_ranges(plan, identity.round)[identity.lane][0]
        engine = lane.engine
        if v2_operand_uses_spm(engine, operand):
            pair_base = (
                V2_SPM_SLOT_BASE
                + identity.round * V2_SPM_SLOT_STRIDE
            )
            return (
                pair_base
                + V2_HAZARD_UNSELECTED_OFFSETS[identity.lane][operand]
            )
    offsets = (
        V2_SPM_READ0_OFFSET,
        (
            v2_ne_read1_offset(plan, lane)
            if lane.engine == ncc_protocol.Engine.NE
            else V2_SPM_READ1_OFFSET
        ),
        (
            v2_ne_write_offset(plan, lane)
            if lane.engine == ncc_protocol.Engine.NE
            else V2_SPM_WRITE_OFFSET
        ),
    )
    return v2_spm_address(identity.slot, offsets[operand])


def validate_observed_ranges_v2(
    case: GenericProbeCase,
    plan: ncc_protocol.Plan,
    observations: tuple[ncc_protocol.IssueObservation, ...],
) -> None:
    ddr_bases: dict[str, int] = {}

    def check_spm(
        observation: ncc_protocol.IssueObservation,
        name: str,
        begin: int,
        byte_count: int,
    ) -> None:
        actual = getattr(observation, name)
        expected = ncc_protocol.InclusiveRange(
            begin, begin + byte_count - 1
        )
        if actual != expected:
            raise RuntimeError(
                f"{case.name}: slot {observation.identity.slot} {name} "
                f"range is {actual}, expected {expected}"
            )

    def check_ddr(
        observation: ncc_protocol.IssueObservation,
        name: str,
        role: str,
        offset: int,
        byte_count: int,
    ) -> None:
        actual = getattr(observation, name)
        if actual is None or actual.end - actual.begin + 1 != byte_count:
            raise RuntimeError(
                f"{case.name}: slot {observation.identity.slot} {name} "
                f"does not cover {byte_count} inclusive DDR bytes"
            )
        if actual.begin < offset:
            raise RuntimeError(
                f"{case.name}: slot {observation.identity.slot} {name} "
                "precedes its planned DDR offset"
            )
        base = actual.begin - offset
        previous = ddr_bases.setdefault(role, base)
        if previous != base:
            raise RuntimeError(
                f"{case.name}: slot {observation.identity.slot} {name} "
                f"does not share the {role} resource base"
            )

    for observation in observations:
        identity = observation.identity
        lane = plan.lanes[identity.lane]
        expected_flags = (
            ncc_protocol.PACKET_OBSERVED
            if (
                lane.issue_mode == ncc_protocol.IssueMode.RAW
                and plan.flags
                not in (
                    ncc_protocol.TIGHT_DEPTH_PLUS_ONE,
                    ncc_protocol.TIGHT_KCORE_BOUNDARY,
                    ncc_protocol.TIGHT_QUEUE_SATURATION,
                    ncc_protocol.TIGHT_WORKER_SCOPE,
                )
            )
            else 0
        )
        read0 = v2_operand_spm_address(
            plan, identity, ncc_protocol.Operand.READ0
        )
        read1 = v2_operand_spm_address(
            plan, identity, ncc_protocol.Operand.READ1
        )
        write = v2_operand_spm_address(
            plan, identity, ncc_protocol.Operand.WRITE
        )
        if lane.engine in (
            ncc_protocol.Engine.CT,
            ncc_protocol.Engine.NE,
        ):
            expected_flags |= (
                ncc_protocol.READ0_VALID
                | ncc_protocol.READ1_VALID
                | ncc_protocol.WRITE_VALID
            )
        elif lane.engine in (
            ncc_protocol.Engine.RDMA,
            ncc_protocol.Engine.WDMA,
        ):
            expected_flags |= (
                ncc_protocol.READ0_VALID | ncc_protocol.WRITE_VALID
            )
        elif lane.engine == ncc_protocol.Engine.TDMA:
            expected_flags |= ncc_protocol.WRITE_VALID
        if (
            plan.flags == ncc_protocol.TIGHT_DEPTH_PLUS_ONE
            and identity.slot == plan.issue_order()[-1]
        ):
            expected_flags |= ncc_protocol.WINDOW_CONTROL_VALID
        if observation.flags != expected_flags:
            raise RuntimeError(
                f"{case.name}: slot {identity.slot} issue flags are "
                f"{observation.flags:#x}, expected {expected_flags:#x}"
            )

        if lane.engine == ncc_protocol.Engine.CT:
            check_spm(observation, "read0", read0, lane.transfer_bytes)
            check_spm(observation, "read1", read1, lane.transfer_bytes)
            check_spm(observation, "write", write, lane.transfer_bytes)
        elif lane.engine == ncc_protocol.Engine.NE:
            check_spm(
                observation, "read0", read0, v2_ne_lhs_bytes(plan, lane)
            )
            check_spm(
                observation, "read1", read1, v2_ne_rhs_bytes(plan, lane)
            )
            check_spm(
                observation, "write", write, v2_ne_output_span(lane)
            )
        elif lane.engine == ncc_protocol.Engine.RDMA:
            check_ddr(
                observation,
                "read0",
                "payload",
                identity.slot * V2_OUTPUT_SLOT_STRIDE
                + V2_OUTPUT_GUARD_BYTES,
                lane.dma_envelope_bytes(),
            )
            check_spm(
                observation, "write", write, lane.dma_envelope_bytes()
            )
        elif lane.engine == ncc_protocol.Engine.WDMA:
            check_spm(
                observation, "read0", read0, lane.dma_envelope_bytes()
            )
            check_ddr(
                observation,
                "write",
                "output",
                V2_OUTPUT_SLOT_BASE
                + identity.slot * V2_OUTPUT_SLOT_STRIDE
                + V2_OUTPUT_GUARD_BYTES,
                lane.dma_envelope_bytes(),
            )
        else:
            check_spm(observation, "write", write, lane.transfer_bytes)


def v2_disjoint_ranges(
    case: GenericProbeCase,
    plan: ncc_protocol.Plan,
    observations: tuple[ncc_protocol.IssueObservation, ...],
) -> None:
    if plan.is_double_slot_observation():
        return
    ranges: list[
        tuple[ncc_protocol.IssueObservation, int, int, str]
    ] = []
    for observation in observations:
        for name in ("read0", "read1", "write"):
            value = getattr(observation, name)
            if value is not None:
                ranges.append(
                    (
                        observation,
                        value.begin,
                        value.end + 1,
                        name,
                    )
                )
    for index, first in enumerate(ranges):
        for second in ranges[index + 1 :]:
            if first[0].identity.slot == second[0].identity.slot:
                continue
            if first[2] <= second[1] or second[2] <= first[1]:
                continue
            allowed_hazard = False
            if plan.effect_relation != ncc_protocol.EffectRelation.NONE:
                identities = (first[0].identity, second[0].identity)
                by_lane = {identity.lane: item for identity, item in zip(
                    identities, (first, second), strict=True
                )}
                if (
                    set(by_lane) == {0, 1}
                    and identities[0].round == identities[1].round
                ):
                    operand_names = {
                        ncc_protocol.Operand.READ0: "read0",
                        ncc_protocol.Operand.READ1: "read1",
                        ncc_protocol.Operand.WRITE: "write",
                    }
                    allowed_hazard = (
                        by_lane[0][3]
                        == operand_names[plan.first_operand]
                        and by_lane[1][3]
                        == operand_names[plan.second_operand]
                    )
            if not allowed_hazard:
                raise RuntimeError(
                    f"{case.name}: slots "
                    f"{first[0].identity.slot}/"
                    f"{second[0].identity.slot} "
                    f"{first[3]}/{second[3]} ranges overlap"
                )


def validate_output_payload_v2(
    output: bytes,
    case: GenericProbeCase,
    plan: ncc_protocol.Plan,
) -> None:
    mutable = bytearray(output)
    mutable[:RECORD_BYTES] = bytes([0xA5]) * RECORD_BYTES
    for identity in plan.issue_identities():
        lane = plan.lanes[identity.lane]
        if (
            plan.is_double_slot_observation()
            and lane.engine != ncc_protocol.Engine.WDMA
        ):
            continue
        expected = v2_expected_result(identity, lane, plan)
        begin = (
            V2_OUTPUT_SLOT_BASE
            + identity.slot * V2_OUTPUT_SLOT_STRIDE
            + V2_OUTPUT_GUARD_BYTES
        )
        if lane.layout_kind == ncc_protocol.LayoutKind.DMA_STRIDED:
            cursor = 0
            for offset in lane.dma_chunk_offsets():
                chunk_end = cursor + lane.layout_inner_bytes
                actual = output[
                    begin + offset :
                    begin + offset + lane.layout_inner_bytes
                ]
                wanted = expected[cursor:chunk_end]
                if actual != wanted:
                    raise RuntimeError(
                        f"{case.name}: slot {identity.slot} scatter chunk "
                        f"at byte {offset} differs"
                    )
                mutable[
                    begin + offset :
                    begin + offset + lane.layout_inner_bytes
                ] = bytes([0xA5]) * lane.layout_inner_bytes
                cursor = chunk_end
            if cursor != len(expected):
                raise RuntimeError(
                    f"{case.name}: scatter oracle did not consume compact bytes"
                )
        else:
            actual = output[begin : begin + len(expected)]
            if actual != expected:
                mismatch = next(
                    index
                    for index, (actual_byte, expected_byte) in enumerate(
                        zip(actual, expected)
                    )
                    if actual_byte != expected_byte
                )
                raise RuntimeError(
                    f"{case.name}: slot {identity.slot} differs at byte "
                    f"{mismatch}: expected=0x{expected[mismatch]:02x} "
                    f"actual=0x{actual[mismatch]:02x}"
                )
            mutable[begin : begin + len(expected)] = (
                bytes([0xA5]) * len(expected)
            )
    if mutable != bytes([0xA5]) * RESOURCE_BYTES:
        mismatch = next(
            index for index, value in enumerate(mutable) if value != 0xA5
        )
        raise RuntimeError(
            f"{case.name}: output changed outside record/result at "
            f"byte {mismatch}"
        )


def classify_ncc_to_kcore_boundary(
    *,
    local_wait: bool,
    wait_cycles: int,
    boundary_done: bool,
    marker_complete: bool,
    boundary_exact: bool,
) -> str:
    if local_wait:
        if (
            wait_cycles == 0
            or not boundary_done
            or not marker_complete
            or not boundary_exact
        ):
            raise RuntimeError(
                "local-wait NCC-to-Kcore boundary did not complete"
            )
        return "local-wait-complete"
    if wait_cycles != 0:
        raise RuntimeError(
            "no-wait NCC-to-Kcore boundary unexpectedly recorded a wait"
        )
    if not boundary_done:
        return "pending-at-snapshot"
    if marker_complete and boundary_exact:
        return "naturally-completed-before-snapshot"
    return "task-done-output-incomplete-at-snapshot"


def parse_record(
    path: pathlib.Path, case: GenericProbeCase, sample: int
) -> dict[str, object]:
    output = path.read_bytes()
    if len(output) != RESOURCE_BYTES:
        raise RuntimeError(
            f"{case.name}: captured {len(output)} bytes, expected "
            f"{RESOURCE_BYTES}"
        )
    words = struct.unpack_from(
        f"<{ncc_protocol.RECORD_WORDS}Q", output
    )
    plan = case.plan_for_sample(sample)
    observations = ncc_protocol.validate_record(words, plan)
    rec = ncc_protocol.REC
    is_wait_scope = case in (
        V2_COMPLETION_SCOPE_CASES + V2_WORKER_WAIT_SCOPE_CASES
    )
    is_subset_join = case in (
        V2_SUBSET_JOIN_CASES + V2_WORKER_SUBSET_SCOPE_CASES
    )
    is_queue_saturation = case in V2_QUEUE_SATURATION_CASES
    is_tight_worker_scope = case in (
        V2_WORKER_WAIT_SCOPE_CASES + V2_WORKER_SUBSET_SCOPE_CASES
    )
    is_mapped_pure_ncc = case in V2_MAPPED_SPM_PURE_NCC_CASES
    is_mapped_ncc_to_kcore = case in V2_MAPPED_SPM_NCC_TO_KCORE_CASES
    is_mapped_kcore_to_ncc = case in V2_MAPPED_SPM_KCORE_TO_NCC_CASES
    allows_pending_boundary = (
        is_wait_scope
        or is_subset_join
        or (
            is_mapped_ncc_to_kcore
            and plan.wait_kind == ncc_protocol.WaitKind.NONE
        )
    )
    if (
        words[rec["OUTPUT_SLOT_BASE"]] != V2_OUTPUT_SLOT_BASE
        or words[rec["OUTPUT_SLOT_STRIDE"]] != V2_OUTPUT_SLOT_STRIDE
        or words[rec["OUTPUT_GUARD_BYTES"]] != V2_OUTPUT_GUARD_BYTES
        or words[rec["RESOURCE_BYTES"]] != RESOURCE_BYTES
        or words[rec["RECORD_GUARD"]] != V2_RECORD_GUARD
    ):
        raise RuntimeError(f"{case.name}: output layout metadata is invalid")
    if (
        words[rec["STABLE_BEFORE"]] != PMU_STABLE_MASK
        or words[rec["STABLE_AFTER"]] != PMU_STABLE_MASK
        or words[rec["PMU_ENABLE"]] == 0
    ):
        raise RuntimeError(f"{case.name}: PMU snapshot is not stable/enabled")
    serial_modes = [
        words[rec["SERIAL_MODE"] + worker] & 1 for worker in range(3)
    ]
    if serial_modes != [0, 0, 0]:
        raise RuntimeError(
            f"{case.name}: parallel queue mode is not active: {serial_modes}"
        )
    pre_wait_controls: dict[int, int] | None = None
    if is_queue_saturation or is_tight_worker_scope:
        pre_wait_controls = {
            worker: words[rec["CONTROL_PRE_WAIT"] + worker]
            for worker in range(3)
        }
    serial_wait_count = words[rec["SERIAL_WAIT_COUNT"]]
    expected_serial_wait_count = (
        len(plan.issue_identities())
        if plan.schedule == ncc_protocol.Schedule.SERIAL
        else 0
    )
    if (
        words[rec["PLAN_CYCLES"]] == 0
        or serial_wait_count != expected_serial_wait_count
        or (
            serial_wait_count != 0
            and words[rec["SERIAL_WAIT_CYCLES"]] == 0
        )
    ):
        raise RuntimeError(f"{case.name}: wait timing record is malformed")
    serial_wait_samples = [
        words[ncc_protocol.WAIT_SAMPLE_BASE + index]
        for index in range(
            min(serial_wait_count, ncc_protocol.MAX_WAIT_SAMPLES)
        )
    ]
    if (
        case in V2_WAIT_OVERHEAD_CASES
        and (
            serial_wait_count > ncc_protocol.MAX_WAIT_SAMPLES
            or sum(serial_wait_samples)
            != words[rec["SERIAL_WAIT_CYCLES"]]
        )
    ):
        raise RuntimeError(
            f"{case.name}: per-wait cycle samples are incomplete"
        )
    timing = {
        "plan_cycles": words[rec["PLAN_CYCLES"]],
        "requested_wait_cycles": words[rec["WAIT_CYCLES"]],
        "serial_wait_count": serial_wait_count,
        "serial_wait_cycles": words[rec["SERIAL_WAIT_CYCLES"]],
        "serial_wait_samples": serial_wait_samples,
        "serial_wait_average_cycles": (
            words[rec["SERIAL_WAIT_CYCLES"]] / serial_wait_count
            if serial_wait_count
            else 0
        ),
        "bounded_window_drain_count": words[
            rec["BOUNDED_WINDOW_DRAIN_COUNT"]
        ],
        "bounded_window_drain_cycles": words[
            rec["BOUNDED_WINDOW_DRAIN_CYCLES"]
        ],
    }
    for observation in observations:
        if (
            observation.inter_type & 0xFF != observation.engine
            or observation.inter_type >> 8 & 0x3 != observation.worker
        ):
            raise RuntimeError(
                f"{case.name}: slot {observation.identity.slot} route is "
                f"{observation.inter_type:#x}"
            )
        lane = plan.lanes[observation.identity.lane]
        if (
            lane.issue_mode == ncc_protocol.IssueMode.RAW
            and observation.execute_rc != 1
        ):
            raise RuntimeError(
                f"{case.name}: raw execute rc is "
                f"{observation.execute_rc}, expected current value 1"
            )
        if (
            observation.boundary_guard_mismatches
            or observation.final_mismatches
            or observation.final_guard_mismatches
            or (
                observation.boundary_mismatches
                and not allows_pending_boundary
            )
        ):
            raise RuntimeError(
                f"{case.name}: slot {observation.identity.slot} oracle "
                "reported a mismatch"
            )
    validate_observed_ranges_v2(case, plan, observations)
    v2_disjoint_ranges(case, plan, observations)

    expected_counts: dict[tuple[int, int], int] = {}
    for identity in plan.issue_identities():
        lane = plan.lanes[identity.lane]
        key = (lane.worker, int(lane.engine))
        expected_counts[key] = expected_counts.get(key, 0) + 1
    instruction_delta: dict[str, int] = {}
    blocking_delta: dict[str, int] = {}
    for worker in range(3):
        for engine in V2_ENGINES:
            index = worker * 5 + int(engine)
            count = delta32(
                words[rec["INSTRUCTION_AFTER"] + index],
                words[rec["INSTRUCTION_BEFORE"] + index],
            )
            blocking = delta32(
                words[rec["BLOCKING_AFTER"] + index],
                words[rec["BLOCKING_BEFORE"] + index],
            )
            key = f"worker{worker}.{engine.name.lower()}"
            instruction_delta[key] = count
            blocking_delta[key] = blocking
            if count != expected_counts.get((worker, int(engine)), 0):
                raise RuntimeError(
                    f"{case.name}: {key} instruction delta is {count}"
                )
    execution_delta = {
        name: delta64(
            words[rec["PMU64_AFTER"] + index],
            words[rec["PMU64_BEFORE"] + index],
        )
        for index, name in enumerate(PMU64_NAMES)
    }
    for worker in {lane.worker for lane in plan.lanes}:
        if not words[rec["CONTROL_FINAL"] + worker] & 0x100:
            raise RuntimeError(
                f"{case.name}: worker {worker} did not reach task_done"
            )
    wait_scope: dict[str, object] | None = None
    if is_wait_scope:
        target_worker = (
            plan.lanes[-1].worker
            if case in V2_WORKER_WAIT_SCOPE_CASES
            else 1
        )
        control_worker = (
            plan.lanes[0].worker
            if case in V2_WORKER_WAIT_SCOPE_CASES
            else 0
        )
        expected_marker, expected_marker_address = v2_completion_marker(plan)
        boundary_marker = words[rec["COMPLETION_MARKER_BOUNDARY"]]
        final_marker = words[rec["COMPLETION_MARKER_FINAL"]]
        target_task_done = bool(
            words[rec["CONTROL_BOUNDARY"] + target_worker] & 0x100
        )
        control_task_done = bool(
            words[rec["CONTROL_BOUNDARY"] + control_worker] & 0x100
        )
        target_boundary_exact = all(
            observation.boundary_mismatches == 0
            for observation in observations
            if observation.worker == target_worker
        )
        if (
            words[rec["WAIT_CYCLES"]] == 0
            or words[rec["COMPLETION_MARKER_EXPECTED"]]
            != expected_marker
            or words[rec["COMPLETION_MARKER_ADDRESS"]]
            != expected_marker_address
            or final_marker != expected_marker
        ):
            raise RuntimeError(
                f"{case.name}: wait/marker observation is malformed"
            )
        if (
            plan.wait_kind == ncc_protocol.WaitKind.BY_WORKER
            and not (
                target_task_done
                and boundary_marker == expected_marker
                and target_boundary_exact
            )
        ):
            raise RuntimeError(
                f"{case.name}: by-worker wait returned before worker"
                f"{target_worker} "
                "completed"
            )
        if (
            case in V2_COMPLETION_SCOPE_CASES
            and plan.wait_kind
            in (
                ncc_protocol.WaitKind.DEFAULT,
                ncc_protocol.WaitKind.LOCAL_FENCE,
            )
            and not control_task_done
        ):
            raise RuntimeError(
                f"{case.name}: default/local fence returned before control "
                f"worker{control_worker} completed"
            )
        scope_name = (
            "default"
            if plan.wait_kind == ncc_protocol.WaitKind.DEFAULT
            else "local-fence"
        )
        target_pending_before_wait = (
            pre_wait_controls is not None
            and not bool(pre_wait_controls[target_worker] & 0x100)
        )
        wait_scope = {
            "wait_cycles": words[rec["WAIT_CYCLES"]],
            "target_pending_before_wait": target_pending_before_wait,
            "distinguishing": (
                target_pending_before_wait
                if case in V2_WORKER_WAIT_SCOPE_CASES
                else True
            ),
            "worker_control_before_wait": pre_wait_controls,
            "target_worker": target_worker,
            "target_engine": plan.lanes[-1].engine.name.lower(),
            "control_worker": control_worker,
            "control_worker_task_status": words[
                rec["CONTROL_BOUNDARY"] + control_worker
            ],
            "control_worker_task_done": control_task_done,
            "target_worker_task_status": words[
                rec["CONTROL_BOUNDARY"] + target_worker
            ],
            "target_worker_task_done": target_task_done,
            "marker_expected": expected_marker,
            "marker_boundary": boundary_marker,
            "marker_final": final_marker,
            "marker_done_at_boundary": boundary_marker == expected_marker,
            "target_results_exact_at_boundary": target_boundary_exact,
            "interpretation": (
                "non-distinguishing-target-drained-before-wait"
                if (
                    case in V2_WORKER_WAIT_SCOPE_CASES
                    and not target_pending_before_wait
                )
                else (
                    f"byworker{target_worker}-completed-target"
                    if plan.wait_kind == ncc_protocol.WaitKind.BY_WORKER
                    else (
                        f"{scope_name}-returned-before-target"
                        if not (
                            target_task_done
                            and boundary_marker == expected_marker
                            and target_boundary_exact
                        )
                        else f"{scope_name}-covered-target"
                    )
                )
            ),
        }
    subset_join: dict[str, object] | None = None
    if is_subset_join:
        boundary_exact = {
            worker: all(
                observation.boundary_mismatches == 0
                for observation in observations
                if observation.worker == worker
            )
            for worker in range(3)
        }
        boundary_done = {
            worker: bool(
                words[rec["CONTROL_BOUNDARY"] + worker] & 0x100
            )
            for worker in range(3)
        }
        joined = {
            worker
            for worker in range(3)
            if plan.wait_worker_mask & (1 << worker)
        }
        if any(
            not boundary_exact[worker] or not boundary_done[worker]
            for worker in joined
        ):
            raise RuntimeError(
                f"{case.name}: subset join returned before a joined worker "
                "completed"
            )
        subset_join = {
            "wait_worker_mask": plan.wait_worker_mask,
            "safety_worker_mask": words[rec["SAFETY_WORKER_MASK"]],
            "target_worker": (
                plan.lanes[-1].worker
                if case in V2_WORKER_SUBSET_SCOPE_CASES
                else None
            ),
            "target_pending_before_wait": (
                pre_wait_controls is not None
                and not bool(
                    pre_wait_controls[plan.lanes[-1].worker] & 0x100
                )
            ),
            "worker_control_before_wait": pre_wait_controls,
            "worker_boundary_exact": boundary_exact,
            "worker_boundary_done": boundary_done,
            "unjoined_workers": sorted(set(range(3)) - joined),
            "unjoined_pending_observed": any(
                not boundary_exact[worker] or not boundary_done[worker]
                for worker in set(range(3)) - joined
            ),
            "interpretation": (
                "unjoined-worker-pending-observed"
                if any(
                    not boundary_exact[worker]
                    or not boundary_done[worker]
                    for worker in set(range(3)) - joined
                )
                else "unjoined-workers-drained-before-boundary"
            ),
        }
    queue_saturation: dict[str, object] | None = None
    if is_queue_saturation:
        assert pre_wait_controls is not None
        engine = plan.lanes[0].engine
        depth = V2_DOCUMENTED_QUEUE_DEPTHS[engine]
        boundary_task_done = bool(
            words[rec["CONTROL_BOUNDARY"]] & 0x100
        )
        if not boundary_task_done:
            raise RuntimeError(
                f"{case.name}: matching worker wait returned before "
                "worker0 reached task_done"
            )
        execute_cycles = [
            observation.execute_cycles for observation in observations
        ]
        queue_saturation = {
            "engine": engine.name.lower(),
            "documented_depth": depth,
            "issue_count": len(observations),
            "load": (
                "sustained"
                if plan.lanes[0].transfer_bytes
                == V2_ACTIVE_OCCUPANCY_BYTES[engine]
                else "short"
            ),
            "worker0_pending_before_wait": not bool(
                pre_wait_controls[0] & 0x100
            ),
            "worker_control_before_wait": pre_wait_controls,
            "execute_cycles": execute_cycles,
            "last_execute_cycles": execute_cycles[-1],
            "boundary_task_done": boundary_task_done,
            "boundary_exact": all(
                observation.boundary_mismatches == 0
                for observation in observations
            ),
            "final_exact": True,
        }
    mapped_spm_boundary: dict[str, object] | None = None
    mapped_mismatches = {
        "boundary_mismatches": sum(
            observation.boundary_mismatches
            for observation in observations
        ),
        "boundary_guard_mismatches": sum(
            observation.boundary_guard_mismatches
            for observation in observations
        ),
        "final_mismatches": sum(
            observation.final_mismatches for observation in observations
        ),
        "final_guard_mismatches": sum(
            observation.final_guard_mismatches
            for observation in observations
        ),
    }
    if is_mapped_pure_ncc:
        terminal_only = plan.schedule == ncc_protocol.Schedule.WINDOW
        if (
            plan.wait_kind != ncc_protocol.WaitKind.BY_WORKER
            or words[rec["WAIT_CYCLES"]] == 0
            or any(
                observation.boundary_mismatches
                or observation.boundary_guard_mismatches
                for observation in observations
            )
        ):
            raise RuntimeError(
                f"{case.name}: pure NCC terminal boundary is malformed"
            )
        mapped_spm_boundary = {
            "direction": "pure-ncc-rdma-ct-wdma",
            "variant": (
                "terminal-only"
                if terminal_only
                else "serial-drain-control"
            ),
            "issue_order": list(plan.issue_order()),
            "intermediate_wait_count": serial_wait_count,
            "intermediate_wait_cycles": words[
                rec["SERIAL_WAIT_CYCLES"]
            ],
            "terminal_wait_cycles": words[rec["WAIT_CYCLES"]],
            "boundary_exact": True,
            "final_exact": True,
            **mapped_mismatches,
        }
    elif is_mapped_ncc_to_kcore:
        expected_marker, expected_marker_address = v2_completion_marker(plan)
        boundary_exact = all(
            observation.boundary_mismatches == 0
            for observation in observations
        )
        boundary_done = bool(words[rec["CONTROL_BOUNDARY"]] & 0x100)
        boundary_marker = words[rec["COMPLETION_MARKER_BOUNDARY"]]
        local_wait = plan.wait_kind == ncc_protocol.WaitKind.LOCAL_FENCE
        marker_complete = boundary_marker == expected_marker
        if (
            words[rec["COMPLETION_MARKER_EXPECTED"]] != expected_marker
            or words[rec["COMPLETION_MARKER_ADDRESS"]]
            != expected_marker_address
            or words[rec["COMPLETION_MARKER_FINAL"]] != expected_marker
        ):
            raise RuntimeError(
                f"{case.name}: NCC-to-Kcore boundary control is malformed"
            )
        try:
            boundary_state = classify_ncc_to_kcore_boundary(
                local_wait=local_wait,
                wait_cycles=words[rec["WAIT_CYCLES"]],
                boundary_done=boundary_done,
                marker_complete=marker_complete,
                boundary_exact=boundary_exact,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"{case.name}: NCC-to-Kcore boundary control is malformed"
            ) from error
        mapped_spm_boundary = {
            "direction": "ncc-ne-tight-window-to-kcore-mapped-read",
            "variant": "local-wait" if local_wait else "no-local-wait",
            "issue_count": len(observations),
            "tight_submission": True,
            "requested_wait_cycles": words[rec["WAIT_CYCLES"]],
            "boundary_task_done": boundary_done,
            "boundary_pending_observed": not boundary_done,
            "boundary_marker_expected": expected_marker,
            "boundary_marker_actual": boundary_marker,
            "boundary_marker_complete": marker_complete,
            "boundary_exact": boundary_exact,
            "boundary_state": boundary_state,
            "boundary_control_distinguishing": (
                boundary_state
                != "naturally-completed-before-snapshot"
            ),
            "boundary_mismatches": sum(
                observation.boundary_mismatches
                for observation in observations
            ),
            "safety_drain_completed": True,
            "final_exact": True,
            **mapped_mismatches,
        }
    elif is_mapped_kcore_to_ncc:
        preissue_wait_requested = bool(
            plan.flags & ncc_protocol.PREISSUE_LOCAL_WAIT
        )
        preissue_wait_done = bool(
            words[rec["FLAGS"]]
            & ncc_protocol.PREISSUE_LOCAL_WAIT_DONE
        )
        preissue_wait_cycles = words[rec["PREISSUE_WAIT_CYCLES"]]
        if (
            preissue_wait_done != preissue_wait_requested
            or (
                preissue_wait_requested
                and preissue_wait_cycles == 0
            )
            or (
                not preissue_wait_requested
                and preissue_wait_cycles != 0
            )
            or plan.wait_kind != ncc_protocol.WaitKind.BY_WORKER
            or words[rec["WAIT_CYCLES"]] == 0
            or any(
                observation.boundary_mismatches
                or observation.boundary_guard_mismatches
                for observation in observations
            )
        ):
            raise RuntimeError(
                f"{case.name}: Kcore-to-NCC boundary control is malformed"
            )
        mapped_spm_boundary = {
            "direction": "kcore-mapped-write-to-ncc-ct-read",
            "variant": (
                "preissue-local-wait-control"
                if preissue_wait_requested
                else "volatile-fence-sync-only"
            ),
            "preissue_local_wait_requested": preissue_wait_requested,
            "preissue_local_wait_done": preissue_wait_done,
            "preissue_local_wait_cycles": preissue_wait_cycles,
            "terminal_wait_cycles": words[rec["WAIT_CYCLES"]],
            "boundary_exact": True,
            "final_exact": True,
            **mapped_mismatches,
        }
    constructor: dict[str, object] | None = None
    if case in V2_CONSTRUCTOR_CASES:
        address = words[rec["CONSTRUCTOR_ADDRESS"]]
        captured = bool(
            words[rec["FLAGS"]] & ncc_protocol.CONSTRUCTOR_CAPTURED
        )
        if not captured or address == 0:
            raise RuntimeError(
                f"{case.name}: constructor return address was not nonnull"
            )
        constructor = {
            "return_address": address,
            "nonnull": True,
            "builder_released_before_issue": True,
            "packet_completed_after_release": True,
        }
    validate_output_payload_v2(output, case, plan)
    result = {
        "case": case.as_dict(),
        "sample": sample,
        "serial_mode": serial_modes,
        "instruction_delta": instruction_delta,
        "blocking_delta": blocking_delta,
        "execution_delta": execution_delta,
        "timing": timing,
        "issues": [dataclasses.asdict(item) for item in observations],
    }
    if wait_scope is not None:
        result["wait_scope"] = wait_scope
    if subset_join is not None:
        result["subset_join"] = subset_join
    if queue_saturation is not None:
        result["queue_saturation"] = queue_saturation
    if mapped_spm_boundary is not None:
        result["mapped_spm_boundary"] = mapped_spm_boundary
    if constructor is not None:
        result["constructor"] = constructor
    if plan.is_strided_dependency_observation():
        result["strided_dependency"] = (
            v2_strided_dependency_evidence(plan)
        )
    elif plan.effect_relation == ncc_protocol.EffectRelation.WAW:
        result["waw_evidence"] = v2_waw_evidence(plan)
    if plan.is_double_slot_observation():
        result["double_slot"] = {
            "hardware_observation_only": True,
            "production_compiler_vertical": False,
            "iterations": plan.rounds,
            "physical_slots": 2,
            "issue_order": list(plan.issue_order()),
        }
    return result


def hazard_pair_key(plan: ncc_protocol.Plan) -> tuple[str, str]:
    if len(plan.lanes) != 2:
        raise ValueError("hazard qualification requires two lanes")
    return (
        plan.lanes[0].engine.name.lower(),
        plan.lanes[1].engine.name.lower(),
    )


def qualified_disjoint_pairs(
    observations: Iterable[dict[str, object]],
) -> set[tuple[str, str]]:
    groups: dict[
        tuple[tuple[str, str], int],
        dict[str, list[dict[str, object]]],
    ] = {}
    for observation in observations:
        case = observation.get("case")
        execution = observation.get("execution_delta")
        if not isinstance(case, dict) or not isinstance(execution, dict):
            continue
        engines = case.get("engines")
        if (
            not isinstance(engines, list)
            or len(engines) != 2
            or case.get("effect") != "none"
            or case.get("range") != "disjoint"
        ):
            continue
        engine_names = tuple(str(item) for item in engines)
        rounds = case.get("rounds")
        if not v2_disjoint_control_rounds_are_safe(engine_names, rounds):
            continue
        schedule = case.get("schedule")
        if schedule not in ("serial", "window"):
            continue
        key = (engine_names, int(rounds))
        groups.setdefault(key, {}).setdefault(str(schedule), []).append(
            execution
        )

    qualified: set[tuple[str, str]] = set()
    for (pair, _), schedules in groups.items():
        serial = schedules.get("serial", [])
        window = schedules.get("window", [])
        if not serial or not window:
            continue
        engines = pair
        serial_excess = statistics.median(
            sum(int(sample[engine]) for engine in engines)
            - int(sample["full"])
            for sample in serial
        )
        window_excess = statistics.median(
            sum(int(sample[engine]) for engine in engines)
            - int(sample["full"])
            for sample in window
        )
        if window_excess > max(0, serial_excess):
            qualified.add(pair)
    return qualified


def validate_hazard_selection(cases: Iterable[GenericProbeCase]) -> None:
    selected = tuple(cases)
    hazards = tuple(
        case
        for case in selected
        if (
            case.plan.effect_relation != ncc_protocol.EffectRelation.NONE
            and not case.plan.is_serial_dma_roundtrip()
            and not case.plan.is_strided_dependency_observation()
            and not case.plan.is_ordered_producer_consumer()
        )
    )
    if not hazards:
        return
    control_groups: dict[
        tuple[tuple[str, str], int], set[ncc_protocol.Schedule]
    ] = {}
    for case in selected:
        plan = case.plan
        if (
            plan.effect_relation != ncc_protocol.EffectRelation.NONE
            or len(plan.lanes) != 2
            or plan.range_relation != ncc_protocol.RangeRelation.DISJOINT
            or not v2_disjoint_control_rounds_are_safe(
                (lane.engine.name.lower() for lane in plan.lanes),
                plan.rounds,
            )
        ):
            continue
        control_groups.setdefault(
            (hazard_pair_key(plan), plan.rounds), set()
        ).add(plan.schedule)
    required_schedules = {
        ncc_protocol.Schedule.SERIAL,
        ncc_protocol.Schedule.WINDOW,
    }
    qualified_control_pairs = {
        pair
        for (pair, _), schedules in control_groups.items()
        if required_schedules.issubset(schedules)
    }
    missing_controls = {
        hazard_pair_key(case.plan)
        for case in hazards
        if hazard_pair_key(case.plan) not in qualified_control_pairs
    }
    if missing_controls:
        missing = sorted(sorted(pair) for pair in missing_controls)
        raise RuntimeError(
            "hazard selection requires below-depth disjoint serial/window "
            f"controls at one common rounds value: {missing}"
        )

    groups: dict[
        tuple[
            ncc_protocol.EffectRelation,
            ncc_protocol.RangeRelation,
            tuple[ncc_protocol.Engine, ...],
            ncc_protocol.Operand,
            ncc_protocol.Operand,
        ],
        set[ncc_protocol.Schedule],
    ] = {}
    for case in hazards:
        key = (
            case.plan.effect_relation,
            case.plan.range_relation,
            tuple(lane.engine for lane in case.plan.lanes),
            case.plan.first_operand,
            case.plan.second_operand,
        )
        groups.setdefault(key, set()).add(case.plan.schedule)
    if any(
        schedules
        != {
            ncc_protocol.Schedule.SERIAL,
            ncc_protocol.Schedule.WINDOW,
        }
        for schedules in groups.values()
    ):
        raise RuntimeError(
            "every hazard relation needs paired serial/window controls"
        )


def case_sample_count(case: GenericProbeCase, repeat: int) -> int:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if case in (
        V2_LARGE_OVERLAP_CASES
        + V2_ACTIVE_OCCUPANCY_CASES
        + V2_QUEUE_SATURATION_CASES
        + V2_WORKER_WAIT_SCOPE_CASES
        + V2_WORKER_SUBSET_SCOPE_CASES
        + V2_STRIDED_DEPENDENCY_CASES
        + V2_MAPPED_SPM_BOUNDARY_CASES
    ):
        return repeat
    if case in (
        V2_DOCUMENTED_DEPTH_CASES
        + V2_DEPTH_PLUS_ONE_CASES
        + V2_DMA_STRIDE_MATRIX_CASES
        + V2_COMPLETION_SCOPE_CASES
        + V2_WAIT_OVERHEAD_CASES
        + V2_CONSTRUCTOR_CASES
        + V2_SUBSET_JOIN_CASES
        + V2_PRODUCER_CONSUMER_ALL_CASES
        + V2_LARGE_BACKLOG_SINGLE_CASES
        + V2_DOUBLE_SLOT_OBSERVATION_CASES
    ):
        return 1
    return repeat if len(case.plan.lanes) > 1 else 1


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[GenericProbeCase],
) -> list[dict[str, object]]:
    raw = args.work_dir / "raw"
    raw.mkdir()
    observations: list[dict[str, object]] = []
    for case in cases:
        # Bounded hazard correctness is independent of whether the disjoint
        # control establishes a performance overlap capability.  The catalog
        # still requires paired serial/window controls, but zero overlap must
        # not suppress exact/partial/adjacent result and guard observations.
        samples = case_sample_count(case, args.repeat)
        for sample in range(samples):
            request = raw / f"{case.name}.{sample}.request.raw"
            payload = raw / f"{case.name}.{sample}.payload.raw"
            output = raw / f"{case.name}.{sample}.output.raw"
            write_request(request, case, sample)
            write_payload(payload, case)
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
                    f"{case.name}: wafer-run omitted lifecycle evidence"
                )
            observation = parse_record(output, case, sample)
            observations.append(observation)
            print(
                "ncc_execution_sample: "
                + json.dumps(observation, sort_keys=True)
            )
    report_worker_wait_scope(observations, cases)
    return observations


def report_worker_wait_scope(
    observations: list[dict[str, object]],
    cases: Iterable[GenericProbeCase],
) -> None:
    expected_names = {
        case.name for case in cases if case in V2_WORKER_WAIT_SCOPE_CASES
    }
    if not expected_names:
        return

    grouped: dict[str, list[dict[str, object]]] = {
        name: [] for name in expected_names
    }
    for observation in observations:
        case_record = observation.get("case")
        wait_scope = observation.get("wait_scope")
        if not isinstance(case_record, dict):
            continue
        name = case_record.get("name")
        if name not in expected_names:
            continue
        if not isinstance(wait_scope, dict):
            raise RuntimeError(
                f"{name}: worker wait-scope observation is malformed"
            )
        pending = wait_scope.get("target_pending_before_wait")
        distinguishing = wait_scope.get("distinguishing")
        interpretation = wait_scope.get("interpretation")
        if (
            type(pending) is not bool
            or type(distinguishing) is not bool
            or distinguishing != pending
            or not isinstance(interpretation, str)
        ):
            raise RuntimeError(
                f"{name}: worker wait-scope qualification is malformed"
            )
        grouped[str(name)].append(wait_scope)

    for name in sorted(expected_names):
        samples = grouped[name]
        distinguishing_samples = [
            sample for sample in samples if sample["distinguishing"]
        ]
        if not distinguishing_samples:
            print(
                "ncc_worker_wait_scope_decision: "
                + json.dumps(
                    {
                        "case": name,
                        "decision": "inconclusive",
                        "distinguishing_samples": 0,
                        "non_distinguishing_samples": len(samples),
                        "reason": (
                            "target-drained-before-wait-in-every-sample"
                        ),
                    },
                    sort_keys=True,
                )
            )
            raise RuntimeError(
                f"{name}: worker wait-scope is inconclusive; no sample "
                "observed the target pending before the requested wait"
            )
        decisions = {
            str(sample["interpretation"])
            for sample in distinguishing_samples
        }
        if len(decisions) != 1:
            raise RuntimeError(
                f"{name}: distinguishing worker wait-scope samples "
                f"disagree: {sorted(decisions)}"
            )
        print(
            "ncc_worker_wait_scope_decision: "
            + json.dumps(
                {
                    "case": name,
                    "decision": next(iter(decisions)),
                    "distinguishing_samples": len(
                        distinguishing_samples
                    ),
                    "non_distinguishing_samples": (
                        len(samples) - len(distinguishing_samples)
                    ),
                },
                sort_keys=True,
            )
        )


def overlap_metrics(
    engines: tuple[str, ...], execution: dict[str, object]
) -> dict[str, int | bool]:
    engine_cycles = sum(int(execution[engine]) for engine in engines)
    full_cycles = int(execution["full"])
    pairwise_excess = engine_cycles - full_cycles
    metrics: dict[str, int | bool] = {
        "pairwise_excess": pairwise_excess,
    }
    if len(engines) == 2:
        metrics["overlap_observed"] = pairwise_excess > 0
    elif len(engines) == 3:
        triple_lower_bound = max(0, engine_cycles - 2 * full_cycles)
        metrics["simultaneous_triple_lower_bound"] = triple_lower_bound
        metrics["triple_overlap_observed"] = engine_cycles > 2 * full_cycles
    else:
        raise ValueError("overlap metrics require two or three engines")
    return metrics


def validate_overlap_selection(cases: Iterable[GenericProbeCase]) -> None:
    groups: dict[tuple[tuple[str, ...], int], set[str]] = {}
    for case in cases:
        if len(case.plan.lanes) < 2:
            continue
        key = (
            tuple(lane.engine.name.lower() for lane in case.plan.lanes),
            case.plan.rounds,
        )
        groups.setdefault(key, set()).add(case.plan.schedule.name.lower())
    if not groups:
        raise RuntimeError(
            "--require-overlap selected no multi-engine control group"
        )
    incomplete = {
        key: sorted({"serial", "window"} - schedules)
        for key, schedules in groups.items()
        if not {"serial", "window"}.issubset(schedules)
    }
    if incomplete:
        raise RuntimeError(
            "--require-overlap needs paired serial/window controls: "
            + repr(incomplete)
        )


def report_overlap(
    observations: list[dict[str, object]], require_overlap: bool
) -> None:
    groups: dict[
        tuple[tuple[str, ...], int],
        dict[str, list[dict[str, int | bool]]],
    ] = {}
    for observation in observations:
        case = observation["case"]
        execution = observation["execution_delta"]
        if (
            not isinstance(case, dict)
            or not isinstance(execution, dict)
            or len(case["engines"]) < 2
        ):
            continue
        engines = tuple(str(engine) for engine in case["engines"])
        if len(engines) not in (2, 3):
            continue
        metrics = overlap_metrics(engines, execution)
        key = (engines, int(case["rounds"]))
        groups.setdefault(key, {}).setdefault(
            str(case["schedule"]), []
        ).append(metrics)
    for (engines, rounds), schedules in groups.items():
        serial = schedules.get("serial", [])
        window = schedules.get("window", [])
        if not serial or not window:
            if require_overlap:
                raise RuntimeError(
                    f"{engines} r{rounds} is missing a serial/window control"
                )
            continue
        serial_pairwise = statistics.median(
            int(item["pairwise_excess"]) for item in serial
        )
        window_pairwise = statistics.median(
            int(item["pairwise_excess"]) for item in window
        )
        report = {
            "engines": engines,
            "rounds": rounds,
            "serial_pairwise_excess_median": serial_pairwise,
            "window_pairwise_excess_median": window_pairwise,
        }
        if len(engines) == 2:
            observed = window_pairwise > max(0, serial_pairwise)
            report["overlap_observed"] = observed
        else:
            serial_triple = statistics.median(
                int(item["simultaneous_triple_lower_bound"])
                for item in serial
            )
            window_triple = statistics.median(
                int(item["simultaneous_triple_lower_bound"])
                for item in window
            )
            observed = window_triple > max(0, serial_triple)
            report.update(
                {
                    "serial_simultaneous_triple_lower_bound_median": (
                        serial_triple
                    ),
                    "window_simultaneous_triple_lower_bound_median": (
                        window_triple
                    ),
                    "triple_overlap_observed": observed,
                }
            )
        print("ncc_overlap_decision: " + json.dumps(report, sort_keys=True))
        if require_overlap and not observed:
            raise RuntimeError(
                f"{engines} r{rounds} did not satisfy its overlap criterion"
            )
    if require_overlap and not groups:
        raise RuntimeError("no multi-engine overlap observations were reported")


def report_stable_large_overlap(
    observations: list[dict[str, object]], expected_repeats: int
) -> None:
    expected_names = {case.name for case in V2_LARGE_OVERLAP_CASES}
    selected: list[dict[str, object]] = []
    by_name: dict[str, list[dict[str, object]]] = {}
    for observation in observations:
        case_record = observation.get("case")
        if (
            not isinstance(case_record, dict)
            or case_record.get("name") not in expected_names
        ):
            continue
        selected.append(observation)
        by_name.setdefault(str(case_record["name"]), []).append(observation)
    if set(by_name) != expected_names:
        raise RuntimeError(
            "large overlap summary requires every serial/window control"
        )
    if any(
        len(samples) != expected_repeats
        or {int(sample["sample"]) for sample in samples}
        != set(range(expected_repeats))
        for samples in by_name.values()
    ):
        raise RuntimeError(
            "large overlap controls did not retain the requested repeats"
        )
    print(
        "ncc_large_overlap_basis: "
        + json.dumps(
            {
                "case_count": len(expected_names),
                "repeats_per_case": expected_repeats,
                "decision": (
                    "paired serial/window median; correctness already passed"
                ),
            },
            sort_keys=True,
        )
    )
    report_overlap(selected, require_overlap=False)


def report_depth_plus_one(
    observations: list[dict[str, object]],
) -> None:
    if len(observations) != 1:
        raise RuntimeError(
            "depth-plus-one suite must produce exactly one observation"
        )
    observation = observations[0]
    case_record = observation.get("case")
    issues = observation.get("issues")
    blocking = observation.get("blocking_delta")
    if (
        not isinstance(case_record, dict)
        or not isinstance(issues, list)
        or not isinstance(blocking, dict)
    ):
        raise RuntimeError("depth-plus-one observation is malformed")
    case = next(
        (
            candidate
            for candidate in V2_DEPTH_PLUS_ONE_CASES
            if candidate.name == case_record.get("name")
        ),
        None,
    )
    if case is None:
        raise RuntimeError("depth-plus-one observation has an unknown case")
    by_slot = {
        int(issue["identity"]["slot"]): issue
        for issue in issues
        if isinstance(issue, dict) and isinstance(issue.get("identity"), dict)
    }
    ordered = [by_slot[slot] for slot in case.plan.issue_order()]
    cycles = [int(issue["execute_cycles"]) for issue in ordered]
    final_control = int(ordered[-1]["control_after_issue"])
    engine = case.plan.lanes[0].engine.name.lower()
    backpressure_cycles = int(blocking[f"worker0.{engine}"])
    print(
        "ncc_depth_plus_one_decision: "
        + json.dumps(
            {
                "engine": engine,
                "issue_count": len(ordered),
                "classification": (
                    "completed-with-pmu-backpressure"
                    if backpressure_cycles > 0
                    else "completed-without-pmu-backpressure"
                ),
                "execute_cycles": cycles,
                "last_call_cycle_dominates": (
                    cycles[-1] > max(cycles[:-1])
                ),
                "control_after_tight_window": final_control,
                "blocking_delta": backpressure_cycles,
                "interpretation": (
                    "completion/count/output/guard prove depth+1 total "
                    "submission; occupancy requires independent evidence"
                ),
            },
            sort_keys=True,
        )
    )


def report_active_occupancy(
    observations: list[dict[str, object]],
    expected_repeats: int,
) -> None:
    if (
        len(observations) != expected_repeats
        or {
            int(observation.get("sample", -1))
            for observation in observations
        }
        != set(range(expected_repeats))
    ):
        raise RuntimeError(
            "active-occupancy suite did not retain the requested repeats"
        )
    case_names = {
        case_record.get("name")
        for observation in observations
        if isinstance(
            case_record := observation.get("case"), dict
        )
    }
    if len(case_names) != 1:
        raise RuntimeError(
            "active-occupancy observations must select one engine case"
        )
    case = next(
        (
            candidate
            for candidate in V2_ACTIVE_OCCUPANCY_CASES
            if candidate.name in case_names
        ),
        None,
    )
    if case is None:
        raise RuntimeError("active-occupancy observation has an unknown case")
    engine = case.plan.lanes[0].engine
    engine_name = engine.name.lower()
    samples: list[dict[str, object]] = []
    for observation in sorted(
        observations, key=lambda item: int(item["sample"])
    ):
        case_record = observation.get("case")
        issues = observation.get("issues")
        blocking = observation.get("blocking_delta")
        if (
            not isinstance(case_record, dict)
            or case_record.get("name") != case.name
            or not isinstance(issues, list)
            or not isinstance(blocking, dict)
        ):
            raise RuntimeError("active-occupancy observation is malformed")
        by_slot = {
            int(issue["identity"]["slot"]): issue
            for issue in issues
            if isinstance(issue, dict)
            and isinstance(issue.get("identity"), dict)
        }
        try:
            ordered = [
                by_slot[slot] for slot in case.plan.issue_order()
            ]
            execute_cycles = [
                int(issue["execute_cycles"]) for issue in ordered
            ]
            control = int(ordered[-1]["control_after_issue"])
            blocking_cycles = int(
                blocking[f"worker0.{engine_name}"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "active-occupancy issue record is malformed"
            ) from error
        samples.append(
            {
                "sample": int(observation["sample"]),
                "control_after_tight_window": control,
                "active_at_observation": not bool(control & 0x100),
                "pmu_backpressure_observed": blocking_cycles > 0,
                "blocking_delta": blocking_cycles,
                "execute_cycles": execute_cycles,
                "last_call_cycle_dominates": (
                    execute_cycles[-1] > max(execute_cycles[:-1])
                ),
            }
        )
    active_samples = sum(
        bool(sample["active_at_observation"]) for sample in samples
    )
    backpressure_samples = sum(
        bool(sample["pmu_backpressure_observed"]) for sample in samples
    )
    print(
        "ncc_active_occupancy_decision: "
        + json.dumps(
            {
                "engine": engine_name,
                "workload": (
                    "large-gemm"
                    if engine == ncc_protocol.Engine.NE
                    else "16k-vector"
                ),
                "issue_count": len(ordered),
                "documented_depth": (
                    V2_DOCUMENTED_QUEUE_DEPTHS[engine]
                ),
                "sample_count": len(samples),
                "active_samples": active_samples,
                "backpressure_samples": backpressure_samples,
                "samples": samples,
                "queue_full_response": (
                    f"pmu-backpressure-in-{backpressure_samples}-of-"
                    f"{len(samples)}-samples"
                    if backpressure_samples
                    else "not-observed"
                ),
                "resident_count": "not-observable-from-control",
            },
            sort_keys=True,
        )
    )


def report_wait_overhead(
    observations: list[dict[str, object]],
) -> None:
    if len(observations) != 2 * len(V2_ENGINES):
        raise RuntimeError(
            "wait-overhead suite requires one serial/window pair per engine"
        )
    by_engine: dict[str, dict[str, dict[str, object]]] = {}
    for observation in observations:
        case = observation.get("case")
        timing = observation.get("timing")
        if not isinstance(case, dict) or not isinstance(timing, dict):
            raise RuntimeError("wait-overhead observation is malformed")
        engines = case.get("engines")
        if not isinstance(engines, list) or len(engines) != 1:
            raise RuntimeError("wait-overhead case must contain one engine")
        by_engine.setdefault(str(engines[0]), {})[
            str(case["schedule"])
        ] = timing
    expected_engines = {engine.name.lower() for engine in V2_ENGINES}
    if set(by_engine) != expected_engines or any(
        set(schedules) != {"serial", "window"}
        for schedules in by_engine.values()
    ):
        raise RuntimeError("wait-overhead serial/window controls are incomplete")
    for engine, by_schedule in sorted(by_engine.items()):
        serial = by_schedule["serial"]
        window = by_schedule["window"]
        report = {
            "engine": engine,
            "workload": "2 issues on worker0",
            "wait_each_plan_cycles": int(serial["plan_cycles"]),
            "wait_once_plan_cycles": int(window["plan_cycles"]),
            "wait_each_calls": int(serial["serial_wait_count"]),
            "wait_each_total_cycles": int(serial["serial_wait_cycles"]),
            "wait_each_average_cycles": serial[
                "serial_wait_average_cycles"
            ],
            "wait_each_cycle_samples": serial["serial_wait_samples"],
            "empty_wait_cycles": int(serial["requested_wait_cycles"]),
            "window_final_wait_cycles": int(window["requested_wait_cycles"]),
            "frequent_wait_extra_plan_cycles": (
                int(serial["plan_cycles"]) - int(window["plan_cycles"])
            ),
        }
        print(
            "ncc_wait_overhead_decision: "
            + json.dumps(report, sort_keys=True)
        )


def report_mapped_spm_boundaries(
    observations: list[dict[str, object]],
) -> None:
    grouped: dict[
        tuple[str, int], dict[str, dict[str, object]]
    ] = {}
    for observation in observations:
        case = observation.get("case")
        boundary = observation.get("mapped_spm_boundary")
        sample = observation.get("sample")
        if (
            not isinstance(case, dict)
            or not isinstance(boundary, dict)
            or type(sample) is not int
        ):
            raise RuntimeError("mapped-SPM boundary observation is malformed")
        direction = boundary.get("direction")
        variant = boundary.get("variant")
        if not isinstance(direction, str) or not isinstance(variant, str):
            raise RuntimeError("mapped-SPM boundary identity is malformed")
        grouped.setdefault((direction, sample), {})[variant] = {
            "case": case.get("name"),
            **boundary,
        }
    for (direction, sample), variants in sorted(grouped.items()):
        print(
            "ncc_mapped_spm_boundary_pair: "
            + json.dumps(
                {
                    "direction": direction,
                    "sample": sample,
                    "pair_complete": len(variants) == 2,
                    "variants": variants,
                },
                sort_keys=True,
            )
        )


def write_qualification(
    args: argparse.Namespace,
    package: pathlib.Path,
    observations: list[dict[str, object]],
) -> None:
    if len(observations) != 1:
        raise RuntimeError("qualification must produce one snapshot")
    manifest = json.loads((package / "manifest.json").read_text())
    modules = manifest.get("modules")
    if not isinstance(modules, list) or len(modules) != 1:
        raise RuntimeError("qualification package does not have one probe ELF")
    record = {
        "probe_schema": ncc_protocol.SCHEMA,
        "probe_elf_digest": modules[0].get("digest"),
        "runtime_library_sha256": str(
            args.expected_runtime_library_sha256
        ).lower(),
        "device_id": args.device_id,
        "expected_device_name": args.expected_device_name,
        "expected_pci_bus_id": args.expected_pci_bus_id,
        "expected_runtime_version": args.expected_runtime_version,
        "expected_tile_count": args.expected_tile_count,
        "observation": observations[0],
    }
    path = args.work_dir / "qualification.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"ncc_execution_qualification: {path}")


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    if args.list_cases:
        print(
            json.dumps(
                {
                    name: [case.as_dict() for case in cases]
                    for name, cases in CASE_CATALOGS.items()
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    args.work_dir = validate_work_dir(args.repo_root, args.work_dir)
    if args.suite == "build-smoke" and not args.no_card:
        raise RuntimeError("build-smoke requires --no-card")
    if args.suite == "build-smoke" and args.selected_cases:
        raise RuntimeError("--case requires a board execution suite")
    if args.suite == "board-all-preflight" and not args.no_card:
        raise RuntimeError("board-all-preflight is a no-card-only suite")
    if args.suite in (
        "documented-depth-manual",
        "depth-plus-one-manual",
        "active-occupancy-manual",
        "completion-scope-manual",
        "constructor-observation",
        "cross-worker-boundary-manual",
        "queue-saturation-manual",
        "worker-wait-scope-manual",
        "worker-subset-scope-manual",
        "producer-consumer-observation",
        "strided-dependency-observation",
        "large-backlog-observation",
        "double-slot-observation",
    ) and (
        not args.no_card
        and (
            not args.selected_cases or len(args.selected_cases) != 1
        )
    ):
        raise RuntimeError(
            f"{args.suite} requires exactly one --case"
        )
    if args.suite == "wait-overhead-manual" and args.selected_cases:
        raise RuntimeError(
            "wait-overhead-manual always runs its serial/window pair"
        )
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "wafer_board_ncc_execution_probe_test: hardware execution "
                "is not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        validate_board_args(args)
    else:
        validate_no_card_protocol_cases()

    validate_catalog_resource_layout(BOARD_ALL_PREFLIGHT_CASES)
    selected_cases: tuple[GenericProbeCase, ...] = ()
    if args.suite != "build-smoke":
        selected_cases = SUITES[args.suite]
        if args.selected_cases:
            by_name = {case.name: case for case in selected_cases}
            missing = [
                name for name in args.selected_cases if name not in by_name
            ]
            if missing:
                raise RuntimeError(
                    f"cases are not in suite {args.suite}: {missing}"
                )
            selected_cases = tuple(
                by_name[name] for name in args.selected_cases
            )
    validate_catalog_resource_layout(selected_cases)
    validate_hazard_selection(selected_cases)
    if args.require_overlap:
        validate_overlap_selection(selected_cases)

    source = write_source_program(args.work_dir)
    package = compile_seed_package(args, source)
    module_path, resource_ids = locate_probe_bindings(package)
    build_probe(args, package, module_path)
    verify_no_card(args, package)
    if args.no_card:
        return 0

    observations = execute_cases(
        args, package, resource_ids, selected_cases
    )
    if args.suite == "qualification":
        write_qualification(args, package, observations)
    if args.suite == "calibration":
        report_overlap(observations, args.require_overlap)
    if {case.name for case in V2_LARGE_OVERLAP_CASES}.issubset(
        {case.name for case in selected_cases}
    ):
        report_stable_large_overlap(observations, args.repeat)
    if args.suite == "depth-plus-one-manual":
        report_depth_plus_one(observations)
    if args.suite == "active-occupancy-manual":
        report_active_occupancy(observations, args.repeat)
    if args.suite == "wait-overhead-manual":
        report_wait_overhead(observations)
    if args.suite == "mapped-spm-boundary-observation":
        report_mapped_spm_boundaries(observations)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            f"wafer_board_ncc_execution_probe_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
