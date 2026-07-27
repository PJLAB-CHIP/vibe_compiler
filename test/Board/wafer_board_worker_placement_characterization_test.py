#!/usr/bin/env python3
"""Build and explicitly run rank-one worker placement/progress probes."""

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
from collections.abc import Iterable

import wafer_worker_placement_characterization_catalog as catalog
import wafer_worker_placement_probe_protocol as protocol


TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
LAUNCH_KIND = "kernel"
TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_worker_placement_probe.c"
PROBE_LL = INPUT_DIR / "wafer_worker_placement_probe.ll"
RESOURCE_ELEMENTS = protocol.RESOURCE_BYTES // 4
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
        {
            "shape": [RESOURCE_ELEMENTS],
            "dtype": "float32",
            "dynamic_dims": [],
        },
        {
            "shape": [RESOURCE_ELEMENTS],
            "dtype": "float32",
            "dynamic_dims": [],
        },
    ],
    "output_signature": [
        {
            "shape": [RESOURCE_ELEMENTS],
            "dtype": "float32",
            "dynamic_dims": [],
        }
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "request"},
        {"type_": "input_arg", "position": 1, "name": "payload"},
    ],
    "unused_inputs": [],
}
LIFECYCLE_LINES = {
    "board_stage: completion",
    "board_stage: device-to-host",
    "board_stage: cleanup",
    "board_execution: true",
}
WORK_DIR_CHILDREN = frozenset(
    {"source-program", "package", "probe-build", "raw"}
)


class PreparationBlocked(RuntimeError):
    """A typed unobservable worker request was rejected before board launch."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group", choices=catalog.GROUPS)
    selection.add_argument("--case", choices=tuple(catalog.CASES_BY_KEY))
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--emit-board-group-keys", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=15000)
    parser.add_argument("--repeat", type=int, default=catalog.REPEATS)
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
            "worker placement probe exceeded its bounded deadline; "
            "no retry, reset, or power action was attempted"
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
        raise RuntimeError("worker probe work directory is too broad")
    return resolved_work


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
    missing = [
        option for option, value in required.items() if value in (None, "")
    ]
    if missing:
        raise RuntimeError(f"board execution requires qualification: {missing}")
    if re.fullmatch(
        r"[0-9a-fA-F]{64}",
        str(args.expected_runtime_library_sha256),
    ) is None:
        raise RuntimeError(
            "runtime library SHA-256 must contain 64 hex digits"
        )
    if args.expected_tile_count != 16:
        raise RuntimeError(
            "worker placement campaign requires the qualified 16-rank card"
        )
    if args.completion_timeout_ms <= 0 or args.repeat < catalog.REPEATS:
        raise RuntimeError(
            "timeout must be positive and repeat must be at least three"
        )


def prepare_selection(key: str) -> dict[str, object]:
    if key in catalog.BOUNDARIES_BY_KEY:
        boundary = catalog.BOUNDARIES_BY_KEY[key]
        raise PreparationBlocked(
            f"{boundary.key}: {boundary.typed_gate}; "
            f"safe alternative: {boundary.safe_alternative}"
        )
    case = catalog.CASES_BY_KEY[key]
    return {
        **case.as_dict(),
        "request_words": list(protocol.request_words(case, 0)),
        "board_request_serializable": True,
    }


def selected_cases(args: argparse.Namespace) -> tuple[catalog.WorkerCase, ...]:
    if args.case:
        return (catalog.CASES_BY_KEY[args.case],)
    if args.group:
        return catalog.GROUP_CASES[args.group]
    if args.no_card:
        return catalog.CASES
    raise RuntimeError("board execution requires one explicit --group or --case")


def write_source_program(work_dir: pathlib.Path) -> pathlib.Path:
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
    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(
        MODULE, encoding="utf-8"
    )
    (source / "functions" / "forward.meta").write_text(
        json.dumps(METADATA, separators=(",", ":")) + "\n",
        encoding="utf-8",
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
            f"--launch-kind={LAUNCH_KIND}",
        ],
        timeout_seconds=300,
    )
    if "published verified package" not in result.stdout:
        raise RuntimeError("wafer-compile did not publish the seed package")
    return package


def locate_probe_bindings(
    package: pathlib.Path,
) -> tuple[pathlib.Path, tuple[int, int, int], int]:
    manifest = json.loads(
        (package / "manifest.json").read_text(encoding="utf-8")
    )
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
        raise RuntimeError("worker probe requires one rank and one module")
    entry = entries[0]
    module = modules[0]
    slots = entry.get("slots")
    if (
        entry.get("rank") != 0
        or entry.get("module") != module.get("id")
        or not isinstance(slots, list)
        or [slot.get("ordinal") for slot in slots] != [0, 1, 2]
    ):
        raise RuntimeError("worker probe pointer slots are not canonical")
    resources_by_id = {
        item.get("id"): item
        for item in resources
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    resource_ids = tuple(slot.get("resource") for slot in slots)
    terminal_completion = entry.get("terminal_completion")
    if not isinstance(terminal_completion, int):
        raise RuntimeError("worker probe terminal completion is missing")
    expected = (
        ("user_input", "read_only"),
        ("user_input", "read_only"),
        ("output", "write_only"),
    )
    for resource_id, contract in zip(resource_ids, expected):
        resource = resources_by_id.get(resource_id)
        if (
            resource is None
            or (resource.get("role"), resource.get("access")) != contract
            or resource.get("bytes") != protocol.RESOURCE_BYTES
            or resource.get("host_visible") is not True
        ):
            raise RuntimeError(
                f"worker probe resource {resource_id} differs from {contract}"
            )
    module_path_value = module.get("path")
    if not isinstance(module_path_value, str):
        raise RuntimeError("worker probe module path is absent")
    module_path = package / module_path_value
    if not module_path.is_file():
        raise RuntimeError("worker probe seed module is absent")
    return module_path, resource_ids, terminal_completion


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
        PROBE_LL,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"worker probe build dependencies are missing: {missing}"
        )
    build = args.work_dir / "probe-build"
    build.mkdir()
    helper = build / "wafer_worker_placement_probe.o"
    linked = build / "wafer_worker_placement_probe.so"
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
            "-Wframe-larger-than=2048",
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
    staged = module_path.with_name(
        f".{module_path.name}.worker-placement-probe"
    )
    shutil.copy2(linked, staged)
    os.replace(staged, module_path)
    undefined = {
        line.split()[-1]
        for line in run([str(nm), "-u", str(module_path)]).stdout.splitlines()
        if line.split()
    }
    heap_symbols = undefined & {
        "rt_malloc",
        "rt_free",
        "csi_kernel_malloc",
        "csi_kernel_free",
    }
    if heap_symbols != {"rt_malloc", "rt_free"}:
        raise RuntimeError(
            "worker probe did not retain the exact Kcore heap loader ABI: "
            f"{sorted(heap_symbols)}"
        )
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module_id = manifest["entries"][0]["module"]
    matches = [
        module
        for module in manifest["modules"]
        if module.get("id") == module_id
    ]
    if len(matches) != 1:
        raise RuntimeError("worker probe module record is ambiguous")
    matches[0]["digest"] = (
        "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    staged_manifest = manifest_path.with_name(
        ".manifest.json.worker-placement-probe"
    )
    staged_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
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
        raise RuntimeError("wafer-run did not verify the worker package")
    print("worker_placement_probe_build: passed")


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


def validate_board_lifecycle(
    stdout: str, terminal_completion: int, context: str
) -> None:
    lines = stdout.splitlines()
    terminal_line = (
        f"terminal_completion: {terminal_completion} kind=entry_return"
    )
    if (
        not LIFECYCLE_LINES.issubset(set(lines))
        or lines.count(terminal_line) != 1
    ):
        raise RuntimeError(
            f"{context}: wafer-run omitted lifecycle or matching "
            "terminal-completion evidence"
        )


def pattern(case: catalog.WorkerCase, sample: int, slot: int, index: int) -> int:
    del case
    return (
        0x51 + sample * 13 + slot * 29 + index * 17 + (index >> 8) * 7
    ) & 0xFF


def write_request(
    path: pathlib.Path, case: catalog.WorkerCase, sample: int
) -> None:
    payload = bytearray([0xD3]) * protocol.RESOURCE_BYTES
    encoded = struct.pack(
        f"<{protocol.REQUEST_WORDS}Q",
        *protocol.request_words(case, sample),
    )
    payload[: len(encoded)] = encoded
    path.write_bytes(payload)


def write_payload(
    path: pathlib.Path, case: catalog.WorkerCase, sample: int
) -> None:
    payload = bytearray([0xCC]) * protocol.RESOURCE_BYTES
    for issue in protocol.issues_for_case(case, sample):
        if issue.engine != catalog.Engine.RDMA:
            continue
        begin = issue.slot * protocol.OUTPUT_SLOT_STRIDE + protocol.GUARD_BYTES
        payload[begin : begin + issue.output_bytes] = bytes(
            pattern(case, sample, issue.slot, index)
            for index in range(issue.output_bytes)
        )
    path.write_bytes(payload)


def expected_result(
    case: catalog.WorkerCase, sample: int, issue: protocol.Issue
) -> bytes:
    if issue.engine == catalog.Engine.CT:
        return struct.pack("<H", 0x4200) * (issue.output_bytes // 2)
    if issue.engine == catalog.Engine.NE:
        return struct.pack("<H", 0x3C00) * (issue.output_bytes // 2)
    return bytes(
        pattern(case, sample, issue.slot, index)
        for index in range(issue.output_bytes)
    )


def parse_output(
    path: pathlib.Path, case: catalog.WorkerCase, sample: int
) -> dict[str, object]:
    output = path.read_bytes()
    if len(output) != protocol.RESOURCE_BYTES:
        raise RuntimeError(
            f"{case.key}: captured {len(output)} bytes, expected "
            f"{protocol.RESOURCE_BYTES}"
        )
    words = struct.unpack_from(f"<{protocol.RECORD_WORDS}Q", output)
    observation = protocol.validate_record(words, case, sample)
    for issue in protocol.issues_for_case(case, sample):
        slot_begin = (
            protocol.OUTPUT_SLOT_BASE
            + issue.slot * protocol.OUTPUT_SLOT_STRIDE
        )
        result_begin = slot_begin + protocol.GUARD_BYTES
        result_end = result_begin + issue.output_bytes
        expected = expected_result(case, sample, issue)
        actual = output[result_begin:result_end]
        if actual != expected:
            mismatch = next(
                index
                for index, (left, right) in enumerate(zip(actual, expected))
                if left != right
            )
            raise RuntimeError(
                f"{case.key}: slot {issue.slot} differs at byte {mismatch}"
            )
        guard = bytes([0xA5]) * protocol.GUARD_BYTES
        if (
            output[slot_begin:result_begin] != guard
            or output[
                result_end : result_end + protocol.GUARD_BYTES
            ]
            != guard
        ):
            raise RuntimeError(
                f"{case.key}: slot {issue.slot} archive guard changed"
            )
    return observation


def ordered_cases_for_sample(
    cases: Iterable[catalog.WorkerCase], sample: int
) -> tuple[catalog.WorkerCase, ...]:
    ordered = tuple(cases)
    if not ordered:
        raise RuntimeError("worker execution plan has no cases")
    if sample < 0:
        raise RuntimeError("worker execution plan has a negative sample")
    rotation = sample % len(ordered)
    rotated = ordered[rotation:] + ordered[:rotation]
    if sample & 1:
        rotated = tuple(reversed(rotated))
    return rotated


def execution_plan(
    cases: Iterable[catalog.WorkerCase], repeat: int
) -> tuple[tuple[int, catalog.WorkerCase], ...]:
    ordered = tuple(cases)
    if repeat <= 0:
        raise RuntimeError("worker execution plan has no repeats")
    return tuple(
        (sample, case)
        for sample in range(repeat)
        for case in ordered_cases_for_sample(ordered, sample)
    )


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    terminal_completion: int,
    cases: Iterable[catalog.WorkerCase],
) -> list[dict[str, object]]:
    raw = args.work_dir / "raw"
    raw.mkdir()
    observations: list[dict[str, object]] = []
    for execution_ordinal, (sample, case) in enumerate(
        execution_plan(cases, args.repeat)
    ):
        request = raw / f"{case.key}.{sample}.request.raw"
        payload = raw / f"{case.key}.{sample}.payload.raw"
        output = raw / f"{case.key}.{sample}.output.raw"
        write_request(request, case, sample)
        write_payload(payload, case, sample)
        result = run(
            board_command(
                args, package, resource_ids, request, payload, output
            ),
            timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0,
        )
        validate_board_lifecycle(
            result.stdout, terminal_completion, case.key
        )
        observation = parse_output(output, case, sample)
        observation["execution_ordinal"] = execution_ordinal
        observations.append(observation)
        print(
            "worker_placement_sample: "
            + json.dumps(observation, sort_keys=True)
        )
    return observations


def report_group(
    cases: tuple[catalog.WorkerCase, ...],
    observations: list[dict[str, object]],
    repeat: int,
) -> None:
    by_case = {
        case.key: [
            observation
            for observation in observations
            if observation["case"] == case.key
        ]
        for case in cases
    }
    if any(
        len(samples) != repeat
        or {int(sample["sample"]) for sample in samples}
        != set(range(repeat))
        for samples in by_case.values()
    ):
        raise RuntimeError("worker activation group lost a repeat")
    if all(case.kind == catalog.Kind.PLACEMENT for case in cases):
        rows = []
        for case in cases:
            samples = by_case[case.key]
            active_workers = [
                worker
                for worker, count in enumerate(case.worker_issues)
                if count
            ]
            blocking_imbalances = []
            for sample in samples:
                blocking = sample["blocking_delta"]
                assert isinstance(blocking, dict)
                values = [
                    int(
                        blocking[
                            f"worker{worker}.queue{int(case.engine)}"
                        ]
                    )
                    for worker in active_workers
                ]
                blocking_imbalances.append(max(values) - min(values))
            rows.append(
                {
                    "case": case.key,
                    "worker_issues": case.worker_issues,
                    "median_plan_cycles": statistics.median(
                        int(sample["plan_cycles"]) for sample in samples
                    ),
                    "median_full_execution_cycles": statistics.median(
                        int(sample["pmu64_delta"][1])
                        for sample in samples
                    ),
                    "median_active_worker_blocking_imbalance": (
                        statistics.median(blocking_imbalances)
                    ),
                    "per_worker_blocking_samples": [
                        sample["blocking_delta"] for sample in samples
                    ],
                }
            )
        decision = {
            "group": cases[0].group,
            "fixed_total_issues": catalog.PLACEMENT_ISSUES,
            "rotated_issue_and_join_order": True,
            "rows": rows,
            "interpretation": (
                "raw placement scaling and blocking imbalance only; no "
                "absolute per-worker completion-cycle or arbiter policy"
            ),
        }
    else:
        concurrent = next(
            case for case in cases if case.kind == catalog.Kind.CONCURRENT
        )
        concurrent_samples = by_case[concurrent.key]
        variant_cycles = {
            case.kind.name.lower().replace("_", "-"): {
                "median_plan_cycles": statistics.median(
                    int(sample["plan_cycles"])
                    for sample in by_case[case.key]
                ),
                "median_full_execution_cycles": statistics.median(
                    int(sample["pmu64_delta"][1])
                    for sample in by_case[case.key]
                ),
            }
            for case in cases
        }
        decision = {
            "group": concurrent.group,
            "matched_variants": [
                "backlog-only",
                "sentinel-only",
                "concurrent",
            ],
            "repeat_count": repeat,
            "observer_completed_samples": sum(
                bool(sample["observer_done_at_boundary"])
                for sample in concurrent_samples
            ),
            "target_pending_at_observer_boundary_samples": sum(
                bool(sample["target_pending_at_observer_boundary"])
                for sample in concurrent_samples
            ),
            "variant_device_cycles": variant_cycles,
            "classification": (
                "observer-progress-before-target-drain-observed"
                if any(
                    bool(sample["target_pending_at_observer_boundary"])
                    for sample in concurrent_samples
                )
                else "observer-completed-after-or-with-natural-target-drain"
            ),
            "interpretation": (
                "bounded no-starvation/progress observation; not a general "
                "fairness constant"
            ),
        }
    decision["case_execution_order_by_sample"] = [
        [
            case.key
            for case in ordered_cases_for_sample(cases, sample)
        ]
        for sample in range(repeat)
    ]
    print(
        "worker_placement_decision: "
        + json.dumps(decision, sort_keys=True)
    )


def validate_static_contract() -> None:
    catalog.validate_catalog()
    if protocol.RESOURCE_BYTES % 4:
        raise RuntimeError("worker resource is not tensor-element aligned")
    for case in catalog.CASES:
        for sample in range(catalog.REPEATS):
            words = protocol.request_words(case, sample)
            if len(words) != protocol.REQUEST_WORDS:
                raise RuntimeError(f"{case.key}: request size differs")
            issues = protocol.issues_for_case(case, sample)
            for issue in issues:
                end = (
                    protocol.OUTPUT_SLOT_BASE
                    + issue.slot * protocol.OUTPUT_SLOT_STRIDE
                    + 2 * protocol.GUARD_BYTES
                    + issue.output_bytes
                )
                if end > protocol.RESOURCE_BYTES:
                    raise RuntimeError(
                        f"{case.key}: issue {issue.slot} exceeds resource"
                    )
    for boundary in catalog.BOUNDARIES:
        try:
            prepare_selection(boundary.key)
        except PreparationBlocked:
            pass
        else:
            raise RuntimeError(
                f"{boundary.key}: typed boundary became serializable"
            )


def main() -> int:
    args = parse_args()
    if args.list_cases:
        print(
            json.dumps(
                {
                    "cases": [case.as_dict() for case in catalog.CASES],
                    "boundaries": [
                        boundary.as_dict()
                        for boundary in catalog.BOUNDARIES
                    ],
                    "groups": {
                        group: [
                            case.key for case in catalog.GROUP_CASES[group]
                        ]
                        for group in catalog.GROUPS
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.emit_board_group_keys:
        print("\n".join(catalog.GROUPS))
        return 0
    required_execution_args = {
        "--wafer-compile": args.wafer_compile,
        "--wafer-run": args.wafer_run,
        "--llvm-clangxx": args.llvm_clangxx,
        "--work-dir": args.work_dir,
    }
    missing_execution_args = [
        option
        for option, value in required_execution_args.items()
        if value is None
    ]
    if missing_execution_args:
        raise RuntimeError(
            "worker probe execution is missing: "
            + ", ".join(missing_execution_args)
        )
    validate_static_contract()
    args.repo_root = args.repo_root.resolve()
    assert args.work_dir is not None
    args.work_dir = validate_work_dir(args.repo_root, args.work_dir)
    cases = selected_cases(args)
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "worker placement hardware execution is not armed; set "
                "WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        validate_board_args(args)

    source = write_source_program(args.work_dir)
    package = compile_seed_package(args, source)
    module_path, resource_ids, terminal_completion = locate_probe_bindings(
        package
    )
    build_probe(args, package, module_path)
    if args.no_card:
        verify_no_card(args, package)
        return 0

    observations = execute_cases(
        args, package, resource_ids, terminal_completion, cases
    )
    report_group(cases, observations, args.repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
