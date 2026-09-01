#!/usr/bin/env python3
"""Compile and execute a 16-tile sharded f16 add on a configured TX board."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys

import numpy as np

import wafer_board_source_program as source_program
import wafer_runtime_launch_contract as runtime_launch


@dataclasses.dataclass(frozen=True)
class RuntimeLaunchCalibrationCase:
    key: str
    tile_count: int
    oracle: str
    completion: str


TILE_COUNT = 16
LOCAL_ELEMENTS = 458752
GLOBAL_ELEMENTS = TILE_COUNT * LOCAL_ELEMENTS
ELEMENT_DTYPE = np.dtype("<f2")
TARGET_IDENTITY = "wafer-tx81-single-card"
PROFILE_INSTRUMENTATION_READY = "profile_instrumentation: ready cards=1 tiles=16"
PROFILE_MEASUREMENT_COUNT = 3
PROFILE_PRIMARY_EXECUTION_COUNT = 1
BOARD_PROCESS_TIMEOUT_MARGIN_SECONDS = 120.0
LAUNCH_PATTERN = "kernel-grid-x16"
TILE_EXECUTION_BASIS = "scheduler-pid-x-and-exact-tile-slices"
RUNTIME_LAUNCH_CALIBRATION_CASES = (
    RuntimeLaunchCalibrationCase(
        "tile16-kernel-add",
        TILE_COUNT,
        "full f16 output and exact complete-Tile runtime domain",
        "all-Tile local drain, device-to-host copy, and normal cleanup",
    ),
)
CALIBRATION_LEAF_BINDINGS = {
    "tile16-kernel-add": RUNTIME_LAUNCH_CALIBRATION_CASES,
}
MODULE = f"""\
module {{
  func.func @main(
      %lhs: tensor<{GLOBAL_ELEMENTS}xf16>,
      %rhs: tensor<{GLOBAL_ELEMENTS}xf16>) -> tensor<{GLOBAL_ELEMENTS}xf16> {{
    %sum = stablehlo.add %lhs, %rhs : tensor<{GLOBAL_ELEMENTS}xf16>
    return %sum : tensor<{GLOBAL_ELEMENTS}xf16>
  }}
}}
"""

METADATA = {
    "name": "forward",
    "stablehlo_version": "0.0.0",
    "input_signature": [
        {"shape": [GLOBAL_ELEMENTS], "dtype": "float16", "dynamic_dims": []},
        {"shape": [GLOBAL_ELEMENTS], "dtype": "float16", "dynamic_dims": []},
    ],
    "output_signature": [
        {"shape": [GLOBAL_ELEMENTS], "dtype": "float16", "dynamic_dims": []}
    ],
    "input_locations": [
        {"type_": "input_arg", "position": 0, "name": "lhs"},
        {"type_": "input_arg", "position": 1, "name": "rhs"},
    ],
    "unused_inputs": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--no-card",
        action="store_true",
        help="compile and validate the complete no-card launch pipeline",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "compile byte-identical ordinary/profile packages, then execute "
            "one fixed Primary->Count->Trace profile collection"
        ),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--repeat", type=int, default=2)
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
            "one-shot complete-Tile process exceeded its outer deadline; it was "
            "killed and this test will not retry or invoke reset/power "
            "operations; board state requires external read-only qualification"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}"
        )
    return result


def write_source_program(work_dir: pathlib.Path) -> pathlib.Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    (source / "data").mkdir(parents=True)
    return source_program.write_program(source, MODULE, METADATA)


def compile_package(
    args: argparse.Namespace,
    source: pathlib.Path,
    package: pathlib.Path,
    *,
    profile: bool,
) -> None:
    command = [
        str(args.wafer_compile),
        "--input-program-dir",
        str(source),
        "--output-dir",
        str(package),
        "--num-partitions=1",
    ]
    if profile:
        command.append("--profile")
    result = run(command)
    if (
        "wrote verified package with num-partitions=1 tiles=16"
        not in result.stdout
    ):
        raise RuntimeError(
            "wafer-compile did not report a verified complete-Tile package"
        )
    written_instrumentation = "wafer-compile: wrote profile instrumentation:"
    if profile and written_instrumentation not in result.stdout:
        raise RuntimeError(
            "wafer-compile did not write the requested profile instrumentation"
        )
    if not profile and written_instrumentation in result.stdout:
        raise RuntimeError(
            "ordinary wafer-compile unexpectedly wrote profile instrumentation"
        )


def require_byte_identical_packages(
    ordinary: pathlib.Path, profiled: pathlib.Path
) -> None:
    def file_digests(root: pathlib.Path) -> dict[pathlib.Path, str]:
        files = {
            path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }
        if not files:
            raise RuntimeError(f"compiled package is empty: {root}")
        return files

    if file_digests(ordinary) != file_digests(profiled):
        raise RuntimeError(
            "ordinary and --profile production packages are not byte-identical"
        )


def require_profile_instrumentation_permissions(package: pathlib.Path) -> None:
    instrumentation = pathlib.Path(f"{package}.profile")
    if not instrumentation.is_dir() or instrumentation.is_symlink():
        raise RuntimeError("profile instrumentation is not a real directory")
    for path in (instrumentation, *instrumentation.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_dir() and not path.is_file():
            raise RuntimeError(
                f"profile instrumentation contains a non-regular member: {path}"
            )
        if stat.S_IMODE(path.stat().st_mode) != 0o777:
            raise RuntimeError(
                f"profile instrumentation permission is not 0777: {path}"
            )


def validate_manifest(
    package: pathlib.Path,
) -> dict[tuple[str, int], int]:
    manifest = json.loads((package / "manifest.json").read_text())
    target = manifest.get("target")
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.GRID_KERNEL_LAUNCH,
        context="all-tile board gate",
    )
    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="all-Tile board gate"
    )
    if not isinstance(target, dict) or target.get("identity") != TARGET_IDENTITY:
        raise RuntimeError("all-Tile board gate has an invalid target")

    modules = manifest.get("modules")
    if not isinstance(modules, list):
        raise RuntimeError("modules must be a list")
    if len(modules) != 1:
        raise RuntimeError("runtime launch has an invalid unique module count")
    module = modules[0]
    if (
        not isinstance(module, dict)
        or not isinstance(module.get("id"), int)
        or module.get("exports") != [runtime_launch.KERNEL_MAIN_EXPORT]
    ):
        raise RuntimeError("package module export contract is invalid")

    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise RuntimeError("package port tables must be lists")
    bindings: dict[tuple[str, int], int] = {}
    for table, role in (("inputs", "user_input"), ("outputs", "output")):
        records = manifest.get(table, [])
        table_resource_ids: set[int] = set()
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError("package port records must be objects")
            resource_id = record.get("id")
            role_index = record.get("role_index")
            key = (role, role_index)
            if (
                not isinstance(resource_id, int)
                or resource_id in table_resource_ids
                or key in bindings
                or record.get("dtype") != "f16"
                or record.get("shape") != [GLOBAL_ELEMENTS]
                or record.get("bytes") != GLOBAL_ELEMENTS * ELEMENT_DTYPE.itemsize
            ):
                raise RuntimeError(f"unexpected tile-16 Add port: {record}")
            table_resource_ids.add(resource_id)
            bindings[key] = resource_id
        if table_resource_ids != set(range(len(records))):
            raise RuntimeError(f"{table} port ids are not a dense domain")

    expected_keys = {("user_input", 0), ("user_input", 1), ("output", 0)}
    if set(bindings) != expected_keys:
        raise RuntimeError(
            "package ports do not exactly cover the Add bindings"
        )

    for entry in entries:
        if (
            entry.get("module") != module["id"]
            or entry.get("transport") != {"kind": "none"}
        ):
            raise RuntimeError("Tile entry contract is invalid")
        expected_arguments = [
            {
                "ordinal": 0,
                "kind": "external_input",
                "port": bindings[("user_input", 0)],
                "access": "read_only",
            },
            {
                "ordinal": 1,
                "kind": "external_input",
                "port": bindings[("user_input", 1)],
                "access": "read_only",
            },
            {
                "ordinal": 2,
                "kind": "external_output",
                "port": bindings[("output", 0)],
                "access": "write_only",
            },
        ]
        actual_arguments = [
            {
                "ordinal": argument.get("ordinal"),
                "kind": argument.get("kind"),
                "port": argument.get("port"),
                "access": argument.get("access"),
            }
            for argument in entry.get("arguments", [])[:3]
            if isinstance(argument, dict)
        ]
        if actual_arguments != expected_arguments:
            raise RuntimeError("Tile launch arguments do not match typed ports")
    return bindings


def write_tile_payloads(
    work_dir: pathlib.Path,
    package: pathlib.Path,
    bindings: dict[tuple[str, int], int],
) -> tuple[list[str], set[int], set[tuple[int, int, int]]]:
    raw = work_dir / "raw"
    raw.mkdir()
    indices = np.arange(GLOBAL_ELEMENTS, dtype=np.int32)
    tiles = indices // LOCAL_ELEMENTS
    lanes = indices % LOCAL_ELEMENTS
    lhs_i32 = tiles * 32 + lanes % 32
    rhs_i32 = 512 + tiles * 16 + lanes % 16
    expected_i32 = lhs_i32 + rhs_i32
    lhs = lhs_i32.astype(ELEMENT_DTYPE)
    rhs = rhs_i32.astype(ELEMENT_DTYPE)
    expected = expected_i32.astype(ELEMENT_DTYPE)
    if (
        not np.array_equal(lhs.astype(np.int32), lhs_i32)
        or not np.array_equal(rhs.astype(np.int32), rhs_i32)
        or not np.array_equal(expected.astype(np.int32), expected_i32)
        or not np.array_equal((lhs + rhs).astype(np.int32), expected_i32)
    ):
        raise RuntimeError("tile-16 f16 Add sentinels are not exactly representable")

    payloads = {
        ("user_input", 0): lhs,
        ("user_input", 1): rhs,
        ("output", 0): expected,
    }
    arguments: list[str] = []
    for (role, role_index), payload in payloads.items():
        path = raw / f"{role}_{role_index}.f16.raw"
        payload.tofile(path)
        option = "--resource" if role == "user_input" else "--expected"
        arguments.extend([option, f"{bindings[(role, role_index)]}={path}"])

    manifest = json.loads((package / "manifest.json").read_text())
    entry_evidence = {
        (entry["id"], entry["tile_id"], entry["module"])
        for entry in manifest["entries"]
    }
    return arguments, {bindings[("output", 0)]}, entry_evidence


def verify_board_evidence(
    stdout: str,
    output_ids: set[int],
    entry_evidence: set[tuple[int, int, int]],
) -> None:
    launch_pattern = LAUNCH_PATTERN
    tile_execution_basis = TILE_EXECUTION_BASIS
    required = (
        "board_stage: validation",
        "board_stage: device-selection",
        "board_stage: resource-allocation",
        "board_stage: host-to-device",
        "board_stage: module-load",
        "board_stage: entry-resolve",
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        f"invocation_tiles: {TILE_COUNT}",
        f"launch_pattern: {launch_pattern}",
        f"physical_tile_execution_basis: {tile_execution_basis}",
        "physical_tile_domain: 0..15",
        "physical_execution_claim: none",
        "board_execution: true",
    )
    output_lines = set(stdout.splitlines())
    missing = [text for text in required if text not in output_lines]
    if missing:
        raise RuntimeError(f"all-tile board invocation omitted evidence: {missing}")

    entry_matches = re.findall(
        r"^entry: (\d+) card_id=0 tile_id=(\d+) launch_slot=\d+ module=(\d+)$",
        stdout, re.MULTILINE,
    )
    output_matches = re.findall(
        rf"^output_compare: port=(\d+) "
        rf"bytes={GLOBAL_ELEMENTS * ELEMENT_DTYPE.itemsize} exact=true$",
        stdout,
        re.MULTILINE,
    )
    tile_matches = re.findall(
        r"^board_tile: tile_id=(\d+) launch_slot=\d+ available=(true|false) "
        r"physical_x=(\d+) physical_y=(\d+)$",
        stdout,
        re.MULTILINE,
    )
    actual_entries = {
        (int(entry), int(tile), int(module)) for entry, tile, module in entry_matches
    }
    actual_outputs = {int(resource) for resource in output_matches}
    available_tiles = [
        int(logical)
        for logical, available, _physical_x, _physical_y in tile_matches
        if available == "true"
    ]
    if (
        len(available_tiles) != TILE_COUNT
        or len(set(available_tiles)) != TILE_COUNT
        or sorted(available_tiles) != list(range(TILE_COUNT))
    ):
        raise RuntimeError(
            "board inventory did not prove available Tiles 0..15"
        )
    if (
        len(entry_matches) != TILE_COUNT
        or actual_entries != entry_evidence
        or len(actual_entries) != TILE_COUNT
    ):
        raise RuntimeError("board result did not prove all 16 tile entries")
    runtime_launch.require_board_completion(stdout, context="all-Tile Add")
    if (
        len(output_matches) != 1
        or actual_outputs != output_ids
        or len(actual_outputs) != 1
    ):
        raise RuntimeError("board result did not prove the exact output comparison")


def verify_no_card_evidence(stdout: str) -> None:
    required = (
        "package: id=0 cards=1 tiles=16",
        f"invocation_tiles: {TILE_COUNT}",
        "board_execution: false",
    )
    output_lines = set(stdout.splitlines())
    missing = [text for text in required if text not in output_lines]
    if missing:
        raise RuntimeError(f"no-card launch invocation omitted evidence: {missing}")

    entry_tiles = sorted(
        int(tile)
        for tile in re.findall(
            r"^entry: \d+ card_id=0 tile_id=(\d+) launch_slot=\d+$",
            stdout, re.MULTILINE
        )
    )
    completion_count = len(
        re.findall(
            r"^completion: return_after_local_drain$", stdout, re.MULTILINE
        )
    )
    if entry_tiles != list(range(TILE_COUNT)):
        raise RuntimeError("no-card launch invocation omitted a tile entry")
    if completion_count != TILE_COUNT:
        raise RuntimeError("no-card launch invocation omitted a Tile completion")


def verify_profile_report(
    package: pathlib.Path, stdout: str, expected_active_engine: str = "CT"
) -> tuple[int, pathlib.Path]:
    instrumentation = pathlib.Path(f"{package}.profile")
    require_profile_instrumentation_permissions(package)
    runs = instrumentation / "runs"
    current = runs / "current"
    if not current.is_symlink():
        raise RuntimeError("profile report current entry is not a symlink")
    run_directory = current.resolve(strict=True)
    if run_directory.parent != runs.resolve(strict=True):
        raise RuntimeError("profile report current entry escapes its runs directory")
    entries = tuple(run_directory.iterdir())
    if len(entries) != 3 or any(not path.is_file() for path in entries):
        raise RuntimeError(
            "profile report does not contain exactly three regular files"
        )
    members = {
        path.name: path
        for path in entries
    }
    if set(members) != {"evidence.json", "analysis.json", "index.html"}:
        raise RuntimeError(
            "profile report does not contain exactly the three public output files"
        )
    evidence = json.loads(members["evidence.json"].read_text())
    analysis = json.loads(members["analysis.json"].read_text())
    html = members["index.html"].read_text()
    if (
        evidence.get("schema") != "wafer.profile.evidence"
        or evidence.get("run_id") != run_directory.name
    ):
        raise RuntimeError("profile evidence identity is invalid")
    samples = evidence.get("measurement", {}).get("samples")
    trace = evidence.get("experiment", {}).get("trace", {})
    trace_tiles = trace.get("tiles")
    if (
        not isinstance(samples, list)
        or len(samples) != PROFILE_PRIMARY_EXECUTION_COUNT
        or samples[0].get("sample_id") != "primary"
        or samples[0].get("sample_index") != 0
        or not isinstance(samples[0].get("device_elapsed_ns"), int)
        or samples[0]["device_elapsed_ns"] < 0
        or samples[0].get("device_timer_kind") != "tx-stream-events"
        or not isinstance(samples[0].get("host_submit_ns"), int)
        or samples[0]["host_submit_ns"] < 0
        or not isinstance(
            samples[0].get("host_launch_to_completion_ns"), int
        )
        or samples[0]["host_launch_to_completion_ns"] < 0
        or samples[0]["host_submit_ns"]
        > samples[0]["host_launch_to_completion_ns"]
        or not isinstance(
            samples[0].get("completion_observation_resolution_ns"), int
        )
        or samples[0]["completion_observation_resolution_ns"] < 0
        or trace.get("complete") is not True
        or not isinstance(trace_tiles, list)
        or len(trace_tiles) != TILE_COUNT
    ):
        raise RuntimeError(
            "profile evidence does not contain one Primary and 16 trace tiles"
        )

    final = analysis.get("program")
    if not isinstance(final, dict):
        raise RuntimeError("profile analysis omitted the final output")
    duration = final.get("duration")
    output = final.get("output")
    tiles = final.get("tiles")
    validity = analysis.get("validity")
    expected_engines = {
        "CT",
        "NE",
        "RDMA",
        "WDMA",
        "TDMA",
        "DIRECT_DTE",
    }
    if (
        not isinstance(duration, dict)
        or duration.get("sample_id") != "primary"
        or duration.get("sample_index") != 0
        or not isinstance(duration.get("device_elapsed_ns"), int)
        or duration["device_elapsed_ns"] < 0
        or duration.get("device_timer_kind") != "tx-stream-events"
        or duration["device_elapsed_ns"] != samples[0]["device_elapsed_ns"]
        or not isinstance(duration.get("host_submit_ns"), int)
        or duration["host_submit_ns"] < 0
        or duration["host_submit_ns"] != samples[0]["host_submit_ns"]
        or not isinstance(
            duration.get("host_launch_to_completion_ns"), int
        )
        or duration["host_launch_to_completion_ns"] < 0
        or duration["host_launch_to_completion_ns"]
        != samples[0]["host_launch_to_completion_ns"]
        or not isinstance(
            duration.get("completion_observation_resolution_ns"), int
        )
        or duration["completion_observation_resolution_ns"]
        != samples[0]["completion_observation_resolution_ns"]
        or not duration.get("qualified")
        or not isinstance(output, dict)
        or not output.get("primary_output_validated")
        or not output.get("diagnostic_captures_match_primary")
        or not isinstance(validity, dict)
        or not validity.get("trace")
        or not validity.get("pmu")
        or not isinstance(tiles, list)
        or len(tiles) != TILE_COUNT
    ):
        raise RuntimeError(
            "profile analysis failed latency, output, or tile qualification"
        )
    for tile in tiles:
        if (
            not isinstance(tile, dict)
            or tile.get("trace_entry_cpu_cycles") is None
            or tile.get("trace_entry_cpu_cycles") < 0
            or {
                engine.get("engine")
                for engine in tile.get("engines", [])
                if isinstance(engine, dict)
            }
            != expected_engines
        ):
            raise RuntimeError(
                "profile analysis has an invalid tile span or engine domain"
            )
        for engine in tile["engines"]:
            for key in (
                "engine_execution_time_ns",
                "wait_window_cpu_cycles",
                "raw_pmu_activity",
            ):
                value = engine.get(key)
                if value is not None and value < 0:
                    raise RuntimeError(
                        f"profile analysis contains a negative {key}"
                    )
        active_engine = next(
            engine
            for engine in tile["engines"]
            if engine["engine"] == expected_active_engine
        )
        if (
            not active_engine.get("engine_execution_time_valid")
            or active_engine.get("engine_execution_time_ns") is None
            or active_engine["engine_execution_time_ns"] <= 0
            or active_engine.get("activity_window_count", 0) <= 0
        ):
            raise RuntimeError(
                "profile analysis did not capture real "
                f"{expected_active_engine} activity on every tile"
            )
    for event in final.get("timeline_events", []):
        if not isinstance(event, dict):
            continue
        activity_window = event.get("activity_window_cpu_cycles")
        if activity_window is not None and (
            not isinstance(activity_window, int)
            or isinstance(activity_window, bool)
            or activity_window < 0
        ):
            raise RuntimeError(
                "profile timeline contains an invalid activity-window cycle value"
            )
        for key in (
            "trace_entry_offset_begin_cpu_cycles",
            "trace_entry_offset_end_cpu_cycles",
        ):
            offset = event.get(key)
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or offset < 0
            ):
                raise RuntimeError(
                    f"profile timeline contains an invalid {key}"
                )
    for retired_text in ("Measurement invalid", "ABBA", "BAAB", "speedup"):
        if retired_text in html:
            raise RuntimeError(
                f"profile HTML retains retired content: {retired_text}"
            )

    expected_report = current / "index.html"
    if f"profile_report: {expected_report}" not in stdout:
        raise RuntimeError("wafer-run did not return the stable profile report path")
    return duration["device_elapsed_ns"], expected_report


def main() -> int:
    args = parse_args()
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print(
            "wafer_board_complete_tile_add_runner: hardware execution is not armed; "
            "set WAFER_EXECUTE_HARDWARE_TESTS=1",
            file=sys.stderr,
        )
        return 77

    if not args.no_card:
        required_board_options = {
            "--expected-runtime-version": args.expected_runtime_version,
            "--expected-device-name": args.expected_device_name,
            "--expected-pci-bus-id": args.expected_pci_bus_id,
            "--expected-tile-count": args.expected_tile_count,
            "--expected-runtime-library-sha256": (
                args.expected_runtime_library_sha256
            ),
        }
        missing = [name for name, value in required_board_options.items() if value is None]
        if missing:
            raise RuntimeError(
                "hardware execution requires explicit qualification options: "
                + ", ".join(missing)
            )
        if args.profile and args.repeat != 1:
            raise RuntimeError("--profile requires exactly one fixed profile collection")
        if not args.profile and args.repeat < 2:
            raise RuntimeError("--repeat must be at least 2")
        if args.completion_timeout_ms <= 0:
            raise RuntimeError("--completion-timeout-ms must be positive")
        if args.expected_tile_count != TILE_COUNT:
            raise RuntimeError(f"--expected-tile-count must be {TILE_COUNT}")

    source = write_source_program(args.work_dir)
    package = args.work_dir / "package"
    ordinary_package = package
    if args.profile:
        ordinary_package = args.work_dir / "ordinary-package"
        compile_package(
            args, source, ordinary_package, profile=False
        )
        compile_package(args, source, package, profile=True)
        # The profiled output directory is the common delivery root; the
        # package root inside it keeps the <root>.profile sibling rule.
        package = package / "package"
        require_byte_identical_packages(ordinary_package, package)
        require_profile_instrumentation_permissions(package)
    else:
        compile_package(args, source, package, profile=False)

    bindings = validate_manifest(package)
    if args.no_card:
        no_card_packages = (
            (ordinary_package, package)
            if args.profile
            else (package,)
        )
        for no_card_package in no_card_packages:
            result = run([
                str(args.wafer_run),
                "--package-dir",
                str(no_card_package),
                "--no-card",
            ])
            verify_no_card_evidence(result.stdout)
            instrumentation_ready = (
                PROFILE_INSTRUMENTATION_READY in result.stdout
            )
            if instrumentation_ready != (
                args.profile and no_card_package == package
            ):
                raise RuntimeError(
                    "no-card launch did not prove the exact profile instrumentation "
                    "activation boundary"
                )
        print("no_card_launch_kind: kernel")
        print(f"no_card_profile: {str(args.profile).lower()}")
        return 0

    (
        resource_arguments,
        output_ids,
        entry_evidence,
    ) = write_tile_payloads(args.work_dir, package, bindings)
    command_tail = [
        "--board",
        "--device-id",
        str(args.device_id),
        "--expected-runtime-version",
        str(args.expected_runtime_version),
        "--expected-device-name",
        args.expected_device_name,
        "--expected-pci-bus-id",
        args.expected_pci_bus_id,
        "--expected-tile-count",
        str(args.expected_tile_count),
        "--expected-runtime-library-sha256",
        args.expected_runtime_library_sha256,
        "--completion-timeout-ms",
        str(args.completion_timeout_ms),
        *resource_arguments,
    ]
    if args.profile:
        profile_result = run(
            [
                str(args.wafer_run),
                "--package-dir",
                str(package),
                *command_tail,
            ],
            timeout_seconds=max(
                300.0,
                (
                    PROFILE_MEASUREMENT_COUNT
                    * args.completion_timeout_ms
                    / 1000.0
                    + BOARD_PROCESS_TIMEOUT_MARGIN_SECONDS
                ),
            ),
        )
        verify_board_evidence(
            profile_result.stdout,
            output_ids,
            entry_evidence,
        )
        device_duration, report = verify_profile_report(
            package, profile_result.stdout
        )
        print(
            "board_profile_collection: pass "
            f"launches={PROFILE_MEASUREMENT_COUNT} "
            f"primary={PROFILE_PRIMARY_EXECUTION_COUNT}"
        )
        print(f"board_profile_device_duration_ns: {device_duration}")
        print(f"board_profile_report: {report}")
        print(profile_result.stdout, end="")
        return 0

    command = [
        str(args.wafer_run),
        "--package-dir",
        str(package),
        *command_tail,
    ]
    for iteration in range(args.repeat):
        result = run(command)
        verify_board_evidence(
            result.stdout,
            output_ids,
            entry_evidence,
        )
        print(
            "board_complete_tile_add_iteration: "
            f"{iteration + 1}/{args.repeat} launch_kind=kernel"
        )
        print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"wafer_board_complete_tile_add_runner: {error}", file=sys.stderr)
        raise SystemExit(1)
