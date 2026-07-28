#!/usr/bin/env python3
"""Compile and execute a 16-rank sharded f16 add on a configured TX board."""

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

import wafer_runtime_launch_contract as runtime_launch


@dataclasses.dataclass(frozen=True)
class RuntimeLaunchCalibrationCase:
    key: str
    rank_count: int
    launch_kind: str
    oracle: str
    completion: str


RANK_COUNT = 16
GLOBAL_ELEMENTS = 512
LOCAL_ELEMENTS = GLOBAL_ELEMENTS // RANK_COUNT
ELEMENT_DTYPE = np.dtype("<f2")
TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
PROFILE_COMPANION_READY = (
    "profile_companion: ready schema=2 ranks=16 variants=1 captures=3"
)
PROFILE_CAMPAIGN_LAUNCH_COUNT = 14
PROFILE_MEASUREMENT_SAMPLE_COUNT = 10
BOARD_PROCESS_TIMEOUT_MARGIN_SECONDS = 120.0
LAUNCH_EVIDENCE = {
    runtime_launch.KERNEL_LAUNCH_KIND: (
        "kernel-grid-x16",
        "scheduler-pid-x-and-exact-rank-slices",
    ),
}
LAUNCH_CONTRACTS = {
    runtime_launch.KERNEL_LAUNCH_KIND: runtime_launch.GRID_KERNEL_LAUNCH,
}
RUNTIME_LAUNCH_CALIBRATION_CASES = tuple(
    RuntimeLaunchCalibrationCase(
        f"rank16-{launch_kind}-add",
        RANK_COUNT,
        launch_kind,
        "all-rank exact slices+full f16 output+schema-v6 rank domain",
        "all-rank terminal+D2H+normal cleanup",
    )
    for launch_kind in LAUNCH_EVIDENCE
)
CALIBRATION_LEAF_BINDINGS = {
    "rank16-kernel-add": RUNTIME_LAUNCH_CALIBRATION_CASES,
}
SHARDING = "{devices=[16]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

MODULE = f"""\
module {{
  func.func @main(
      %lhs: tensor<{GLOBAL_ELEMENTS}xf16> {{mhlo.sharding = "{SHARDING}"}},
      %rhs: tensor<{GLOBAL_ELEMENTS}xf16> {{mhlo.sharding = "{SHARDING}"}})
      -> (tensor<{GLOBAL_ELEMENTS}xf16> {{mhlo.sharding = "{SHARDING}"}}) {{
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
            "compile byte-identical ordinary/profile packages, run the "
            "ordinary correctness gate first, then execute one fixed profile "
            "campaign"
        ),
    )
    parser.add_argument(
        "--launch-kind",
        choices=tuple(LAUNCH_EVIDENCE),
        default=runtime_launch.KERNEL_LAUNCH_KIND,
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
            "one-shot all-rank process exceeded its outer deadline; it was "
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
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(MODULE)
    (source / "functions" / "forward.meta").write_text(
        json.dumps(METADATA, separators=(",", ":")) + "\n"
    )
    return source


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
        "--output-program-dir",
        str(package),
        f"--execution-ranks={RANK_COUNT}",
        f"--target-profile={TARGET_PROFILE}",
        f"--launch-kind={args.launch_kind}",
    ]
    if profile:
        command.append("--profile")
    result = run(command)
    if (
        "published verified package with execution-ranks=16"
        not in result.stdout
    ):
        raise RuntimeError(
            "wafer-compile did not report a verified rank-16 package"
        )
    published_companion = "wafer-compile: published profile companion:"
    if profile and published_companion not in result.stdout:
        raise RuntimeError(
            "wafer-compile did not publish the requested profile companion"
        )
    if not profile and published_companion in result.stdout:
        raise RuntimeError(
            "ordinary wafer-compile unexpectedly published a profile companion"
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


def require_profile_companion_permissions(package: pathlib.Path) -> None:
    companion = pathlib.Path(f"{package}.profile")
    if not companion.is_dir() or companion.is_symlink():
        raise RuntimeError("profile companion is not a real directory")
    for path in (companion, *companion.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_dir() and not path.is_file():
            raise RuntimeError(
                f"profile companion contains a non-regular member: {path}"
            )
        if stat.S_IMODE(path.stat().st_mode) != 0o777:
            raise RuntimeError(
                f"profile companion permission is not 0777: {path}"
            )


def require_rank_domain(records: object, name: str) -> list[dict[str, object]]:
    if not isinstance(records, list) or len(records) != RANK_COUNT:
        raise RuntimeError(f"package must contain exactly {RANK_COUNT} {name}")
    if any(not isinstance(record, dict) for record in records):
        raise RuntimeError(f"package {name} must be objects")
    typed_records = records
    ranks = [record.get("rank") for record in typed_records]
    if any(not isinstance(rank, int) for rank in ranks):
        raise RuntimeError(f"package {name} ranks must be integers")
    if sorted(ranks) != list(range(RANK_COUNT)):
        raise RuntimeError(f"package {name} do not exactly cover all logical ranks")
    if len({record.get("id") for record in typed_records}) != RANK_COUNT:
        raise RuntimeError(f"package {name} IDs are not unique")
    return typed_records


def require_partitioned_axis0_binding(
    binding: dict[str, object], index_field: str, index: int, name: str
) -> dict[int, slice]:
    if (
        binding.get(index_field) != index
        or binding.get("distribution") != "partitioned"
        or binding.get("global_shape") != [GLOBAL_ELEMENTS]
        or binding.get("local_shape") != [LOCAL_ELEMENTS]
        or binding.get("dtype") != "float16"
    ):
        raise RuntimeError(f"{name} is not the required axis-0 f16 partition")
    ranks = binding.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != RANK_COUNT:
        raise RuntimeError(f"{name} does not describe every logical rank")

    slices: dict[int, slice] = {}
    covered = np.zeros(GLOBAL_ELEMENTS, dtype=np.bool_)
    for rank_record in ranks:
        if not isinstance(rank_record, dict):
            raise RuntimeError(f"{name} rank geometry must be objects")
        rank = rank_record.get("rank")
        offsets = rank_record.get("offsets")
        sizes = rank_record.get("sizes")
        strides = rank_record.get("strides")
        if (
            not isinstance(rank, int)
            or rank < 0
            or rank >= RANK_COUNT
            or rank in slices
            or rank_record.get("replica_id") != 0
            or not isinstance(offsets, list)
            or not isinstance(sizes, list)
            or not isinstance(strides, list)
            or len(offsets) != 1
            or sizes != [LOCAL_ELEMENTS]
            or strides != [1]
            or not isinstance(offsets[0], int)
        ):
            raise RuntimeError(f"{name} has invalid rank slice geometry")
        begin = offsets[0]
        end = begin + LOCAL_ELEMENTS
        if begin != rank * LOCAL_ELEMENTS or np.any(covered[begin:end]):
            raise RuntimeError(f"{name} rank slices overlap or leave the global domain")
        covered[begin:end] = True
        slices[rank] = slice(begin, end)
    if set(slices) != set(range(RANK_COUNT)) or not np.all(covered):
        raise RuntimeError(f"{name} rank slices do not exactly cover the global domain")
    return slices


def load_boundary_slices(package: pathlib.Path) -> dict[int, slice]:
    metadata = json.loads((package / "functions" / "forward.meta").read_text())
    boundary = metadata.get("distributed_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("version") != 1
        or boundary.get("logical_rank_count") != RANK_COUNT
    ):
        raise RuntimeError("production SPMD output omitted the rank-16 boundary")
    inputs = boundary.get("inputs")
    outputs = boundary.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise RuntimeError("rank-16 Add must have two distributed inputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("rank-16 Add must have one distributed output")
    if any(not isinstance(binding, dict) for binding in inputs + outputs):
        raise RuntimeError("distributed boundary bindings must be objects")
    input_by_index = {binding.get("argument_index"): binding for binding in inputs}
    if set(input_by_index) != {0, 1}:
        raise RuntimeError("rank-16 Add input boundary indices are not exact")
    lhs_slices = require_partitioned_axis0_binding(
        input_by_index[0], "argument_index", 0, "lhs"
    )
    rhs_slices = require_partitioned_axis0_binding(
        input_by_index[1], "argument_index", 1, "rhs"
    )
    output_slices = require_partitioned_axis0_binding(
        outputs[0], "result_index", 0, "output"
    )
    if lhs_slices != rhs_slices or lhs_slices != output_slices:
        raise RuntimeError("rank-16 Add boundaries disagree on rank slice geometry")
    return output_slices


def validate_manifest(
    package: pathlib.Path, launch_kind: str
) -> dict[tuple[int, str, int], int]:
    manifest = json.loads((package / "manifest.json").read_text())
    target = manifest.get("target")
    runtime_launch.require_manifest_launch(
        manifest,
        LAUNCH_CONTRACTS[launch_kind],
        context="all-rank board gate",
    )
    if (
        manifest.get("rank_count") != RANK_COUNT
        or not isinstance(target, dict)
        or target.get("profile") != TARGET_PROFILE
    ):
        raise RuntimeError(
            "all-rank board gate requires a schema-v6 rank-16 TX package"
        )

    modules = manifest.get("modules")
    if not isinstance(modules, list):
        raise RuntimeError("modules must be a list")
    if len(modules) != 1:
        raise RuntimeError("runtime launch has an invalid unique module count")
    module_by_id: dict[int, dict[str, object]] = {}
    for module in modules:
        if not isinstance(module, dict) or not isinstance(module.get("id"), int):
            raise RuntimeError("package module identity is invalid")
        module_id = module["id"]
        if module_id in module_by_id or module.get("exports") != [
            {"role": "main", "symbol": "main"}
        ]:
            raise RuntimeError("package module export contract is invalid")
        module_by_id[module_id] = module
    entries = require_rank_domain(manifest.get("entries"), "entries")
    completions = require_rank_domain(manifest.get("completions"), "completions")
    completion_by_rank = {record["rank"]: record for record in completions}

    resources = manifest.get("resources")
    if not isinstance(resources, list) or len(resources) != 3 * RANK_COUNT:
        raise RuntimeError("rank-16 Add must contain exactly three resources per rank")
    resource_ids: set[int] = set()
    bindings: dict[tuple[int, str, int], int] = {}
    for resource in resources:
        if not isinstance(resource, dict):
            raise RuntimeError("package resources must be objects")
        resource_id = resource.get("id")
        rank = resource.get("rank")
        role = resource.get("role")
        role_index = resource.get("role_index")
        key = (rank, role, role_index)
        if (
            not isinstance(resource_id, int)
            or resource_id in resource_ids
            or not isinstance(rank, int)
            or rank < 0
            or rank >= RANK_COUNT
            or key in bindings
            or not resource.get("host_visible")
            or resource.get("type") != {"dtype": "f16", "shape": [LOCAL_ELEMENTS]}
            or resource.get("bytes") != LOCAL_ELEMENTS * ELEMENT_DTYPE.itemsize
        ):
            raise RuntimeError(f"unexpected rank-16 Add resource: {resource}")
        if role == "user_input" and role_index in (0, 1):
            expected_access = "read_only"
        elif role == "output" and role_index == 0:
            expected_access = "write_only"
        else:
            raise RuntimeError(f"unexpected host-visible resource binding: {key}")
        if resource.get("access") != expected_access:
            raise RuntimeError(f"unexpected resource access for binding: {key}")
        resource_ids.add(resource_id)
        bindings[key] = resource_id

    expected_keys = {
        (rank, role, role_index)
        for rank in range(RANK_COUNT)
        for role, role_index in (("user_input", 0), ("user_input", 1), ("output", 0))
    }
    if set(bindings) != expected_keys:
        raise RuntimeError(
            "package resources do not exactly cover all rank Add bindings"
        )

    entry_evidence: set[tuple[int, int, int]] = set()
    for entry in entries:
        rank = entry["rank"]
        module = module_by_id.get(entry.get("module"))
        completion = completion_by_rank[rank]
        if (
            module is None
            or entry.get("terminal_completion") != completion.get("id")
            or completion.get("kind") != "entry_return"
            or entry.get("transport") != {"kind": "none"}
        ):
            raise RuntimeError(f"rank {rank} entry/completion contract is invalid")
        expected_slots = [
            (0, bindings[(rank, "user_input", 0)], "read_only"),
            (1, bindings[(rank, "user_input", 1)], "read_only"),
            (2, bindings[(rank, "output", 0)], "write_only"),
        ]
        actual_slots = [
            (slot.get("ordinal"), slot.get("resource"), slot.get("access"))
            for slot in entry.get("slots", [])
            if isinstance(slot, dict)
        ]
        if actual_slots != expected_slots:
            raise RuntimeError(f"rank {rank} launch slots do not match typed resources")
        entry_evidence.add((entry["id"], rank, entry["module"]))
    if len(entry_evidence) != RANK_COUNT:
        raise RuntimeError("rank-16 entry evidence is not unique")
    referenced_modules = {entry["module"] for entry in entries}
    if referenced_modules != set(module_by_id):
        raise RuntimeError("entry-to-module coverage is not all-and-only")
    if len(referenced_modules) != 1:
        raise RuntimeError("kernel entries do not share one aggregate module")
    return bindings


def write_rank_payloads(
    work_dir: pathlib.Path,
    package: pathlib.Path,
    slices: dict[int, slice],
    bindings: dict[tuple[int, str, int], int],
) -> tuple[list[str], set[int], set[tuple[int, int, int]], set[tuple[int, int]]]:
    raw = work_dir / "raw"
    raw.mkdir()
    indices = np.arange(GLOBAL_ELEMENTS, dtype=np.int32)
    ranks = indices // LOCAL_ELEMENTS
    lanes = indices % LOCAL_ELEMENTS
    lhs_i32 = ranks * 32 + lanes
    rhs_i32 = 512 + ranks * 16 + lanes
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
        raise RuntimeError("rank-16 f16 Add sentinels are not exactly representable")

    arguments: list[str] = []
    output_ids: set[int] = set()
    lhs_payloads: set[bytes] = set()
    rhs_payloads: set[bytes] = set()
    reconstructed = np.empty(GLOBAL_ELEMENTS, dtype=ELEMENT_DTYPE)
    covered = np.zeros(GLOBAL_ELEMENTS, dtype=np.bool_)
    for rank in range(RANK_COUNT):
        rank_slice = slices[rank]
        rank_payloads = {
            ("user_input", 0): lhs[rank_slice],
            ("user_input", 1): rhs[rank_slice],
            ("output", 0): expected[rank_slice],
        }
        lhs_payloads.add(rank_payloads[("user_input", 0)].tobytes())
        rhs_payloads.add(rank_payloads[("user_input", 1)].tobytes())
        if np.any(covered[rank_slice]):
            raise RuntimeError("output reconstruction encountered overlapping slices")
        covered[rank_slice] = True
        reconstructed[rank_slice] = rank_payloads[("output", 0)]
        for (role, role_index), payload in rank_payloads.items():
            path = raw / f"rank_{rank:05d}.{role}_{role_index}.f16.raw"
            payload.tofile(path)
            resource_id = bindings[(rank, role, role_index)]
            option = "--resource" if role == "user_input" else "--expected"
            arguments.extend([option, f"{resource_id}={path}"])
            if role == "output":
                output_ids.add(resource_id)

    if len(lhs_payloads) != RANK_COUNT or len(rhs_payloads) != RANK_COUNT:
        raise RuntimeError("rank inputs are not distinct across all logical ranks")
    if not np.all(covered) or not np.array_equal(reconstructed, expected):
        raise RuntimeError(
            "rank output slices do not reconstruct the global CPU result"
        )

    manifest = json.loads((package / "manifest.json").read_text())
    entry_evidence = {
        (entry["id"], entry["rank"], entry["module"]) for entry in manifest["entries"]
    }
    completion_evidence = {
        (completion["id"], completion["rank"]) for completion in manifest["completions"]
    }
    return arguments, output_ids, entry_evidence, completion_evidence


def verify_board_evidence(
    stdout: str,
    launch_kind: str,
    output_ids: set[int],
    entry_evidence: set[tuple[int, int, int]],
    completion_evidence: set[tuple[int, int]],
) -> None:
    launch_pattern, logical_tile_basis = LAUNCH_EVIDENCE[launch_kind]
    required = (
        "board_stage: preflight",
        "board_stage: device-selection",
        "board_stage: resource-allocation",
        "board_stage: host-to-device",
        "board_stage: module-load",
        "board_stage: entry-resolve",
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        f"invocation_ranks: {RANK_COUNT}",
        f"launch_pattern: {launch_pattern}",
        f"logical_tile_execution_basis: {logical_tile_basis}",
        "logical_tile_domain: 0..15",
        "physical_execution_claim: none",
        "board_execution: true",
    )
    output_lines = set(stdout.splitlines())
    missing = [text for text in required if text not in output_lines]
    if missing:
        raise RuntimeError(f"all-rank board invocation omitted evidence: {missing}")

    entry_matches = re.findall(
        r"^entry: (\d+) rank=(\d+) module=(\d+)$", stdout, re.MULTILINE
    )
    completion_matches = re.findall(
        r"^terminal_completion: (\d+) kind=entry_return rank=(\d+)$",
        stdout,
        re.MULTILINE,
    )
    output_matches = re.findall(
        r"^output_compare: resource=(\d+) bytes=64 exact=true$",
        stdout,
        re.MULTILINE,
    )
    tile_matches = re.findall(
        r"^board_tile: logical=(\d+) available=(true|false) "
        r"physical_x=(\d+) physical_y=(\d+)$",
        stdout,
        re.MULTILINE,
    )
    actual_entries = {
        (int(entry), int(rank), int(module)) for entry, rank, module in entry_matches
    }
    actual_completions = {
        (int(completion), int(rank)) for completion, rank in completion_matches
    }
    actual_outputs = {int(resource) for resource in output_matches}
    available_tiles = [
        int(logical)
        for logical, available, _physical_x, _physical_y in tile_matches
        if available == "true"
    ]
    if (
        len(available_tiles) != RANK_COUNT
        or len(set(available_tiles)) != RANK_COUNT
        or sorted(available_tiles) != list(range(RANK_COUNT))
    ):
        raise RuntimeError(
            "board inventory did not prove available logical tiles 0..15"
        )
    if (
        len(entry_matches) != RANK_COUNT
        or actual_entries != entry_evidence
        or len(actual_entries) != RANK_COUNT
    ):
        raise RuntimeError("board result did not prove all 16 rank entries")
    if (
        len(completion_matches) != RANK_COUNT
        or actual_completions != completion_evidence
        or len(actual_completions) != RANK_COUNT
    ):
        raise RuntimeError("board result did not prove all 16 terminal completions")
    if (
        len(output_matches) != RANK_COUNT
        or actual_outputs != output_ids
        or len(actual_outputs) != RANK_COUNT
    ):
        raise RuntimeError("board result did not prove all 16 exact output comparisons")


def verify_no_card_evidence(stdout: str) -> None:
    required = (
        "package: id=0 schema=6 ranks=16",
        f"invocation_ranks: {RANK_COUNT}",
        "board_execution: false",
    )
    output_lines = set(stdout.splitlines())
    missing = [text for text in required if text not in output_lines]
    if missing:
        raise RuntimeError(f"no-card launch invocation omitted evidence: {missing}")

    entry_ranks = sorted(
        int(rank)
        for rank in re.findall(
            r"^entry: \d+ rank=(\d+)$", stdout, re.MULTILINE
        )
    )
    completion_ids = sorted(
        int(completion)
        for completion in re.findall(
            r"^terminal_completion: (\d+) kind=entry_return$",
            stdout,
            re.MULTILINE,
        )
    )
    if entry_ranks != list(range(RANK_COUNT)):
        raise RuntimeError("no-card launch invocation omitted a rank entry")
    if completion_ids != list(range(RANK_COUNT)):
        raise RuntimeError("no-card launch invocation omitted a terminal completion")


def verify_profile_report(
    package: pathlib.Path, stdout: str
) -> tuple[int | float, int, int, pathlib.Path]:
    companion = pathlib.Path(f"{package}.profile")
    require_profile_companion_permissions(package)
    runs = companion / "runs"
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
            "profile report does not contain exactly the three public artifacts"
        )
    evidence = json.loads(members["evidence.json"].read_text())
    analysis = json.loads(members["analysis.json"].read_text())
    html = members["index.html"].read_text()
    if (
        evidence.get("schema") != "wafer.profile.evidence"
        or evidence.get("schema_version") != 4
        or evidence.get("run_id") != run_directory.name
    ):
        raise RuntimeError("profile evidence identity is invalid")
    samples = evidence.get("measurement", {}).get("samples")
    trace = evidence.get("experiment", {}).get("trace", {})
    trace_tiles = trace.get("tiles")
    if (
        not isinstance(samples, list)
        or len(samples) != PROFILE_MEASUREMENT_SAMPLE_COUNT
        or trace.get("complete") is not True
        or not isinstance(trace_tiles, list)
        or len(trace_tiles) != RANK_COUNT
    ):
        raise RuntimeError(
            "profile evidence does not contain 10 samples and 16 trace tiles"
        )

    final = analysis.get("final_artifact")
    if not isinstance(final, dict):
        raise RuntimeError("profile analysis omitted the final artifact")
    latency = final.get("latency")
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
        not isinstance(latency, dict)
        or latency.get("sample_count") != PROFILE_MEASUREMENT_SAMPLE_COUNT
        or not latency.get("qualified")
        or not isinstance(output, dict)
        or not output.get("production_repeats_exact")
        or not output.get("diagnostic_captures_exact")
        or not isinstance(validity, dict)
        or not validity.get("summary")
        or not validity.get("trace")
        or not validity.get("pmu")
        or not isinstance(tiles, list)
        or len(tiles) != RANK_COUNT
    ):
        raise RuntimeError(
            "profile analysis failed latency, output, or tile qualification"
        )
    for tile in tiles:
        if (
            not isinstance(tile, dict)
            or tile.get("summary_entry_cycles") is None
            or tile.get("summary_entry_cycles") < 0
            or tile.get("trace_entry_cycles") is None
            or tile.get("trace_entry_cycles") < 0
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
                "busy_cycles",
                "wait_window_cycles",
                "raw_pmu_activity",
            ):
                value = engine.get(key)
                if value is not None and value < 0:
                    raise RuntimeError(
                        f"profile analysis contains a negative {key}"
                    )
        ct = next(
            engine
            for engine in tile["engines"]
            if engine["engine"] == "CT"
        )
        if (
            not ct.get("busy_cycles_valid")
            or ct.get("busy_cycles") is None
            or ct["busy_cycles"] <= 0
            or ct.get("activity_window_count", 0) <= 0
        ):
            raise RuntimeError(
                "profile analysis did not capture real CT activity on every tile"
            )
    if any(
        event.get("activity_window_cycles", -1) < 0
        or event.get("trace_entry_offset_begin", -1) < 0
        or event.get("trace_entry_offset_end", -1) < 0
        for event in final.get("timeline_events", [])
        if isinstance(event, dict)
    ):
        raise RuntimeError("profile timeline contains a negative cycle value")
    for legacy in ("Measurement invalid", "ABBA", "BAAB", "speedup"):
        if legacy in html:
            raise RuntimeError(f"profile HTML retains legacy content: {legacy}")

    expected_report = current / "index.html"
    if f"profile_report: {expected_report}" not in stdout:
        raise RuntimeError("wafer-run did not publish the stable profile report path")
    return (
        latency["median_ns"],
        latency["minimum_ns"],
        latency["maximum_ns"],
        expected_report,
    )


def main() -> int:
    args = parse_args()
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print(
            "wafer_board_all_rank_add_test: hardware execution is not armed; "
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
            raise RuntimeError("--profile requires exactly one fixed campaign")
        if not args.profile and args.repeat < 2:
            raise RuntimeError("--repeat must be at least 2")
        if args.completion_timeout_ms <= 0:
            raise RuntimeError("--completion-timeout-ms must be positive")
        if args.expected_tile_count != RANK_COUNT:
            raise RuntimeError(f"--expected-tile-count must be {RANK_COUNT}")

    source = write_source_program(args.work_dir)
    package = args.work_dir / "package"
    ordinary_package = package
    if args.profile:
        ordinary_package = args.work_dir / "ordinary-package"
        compile_package(
            args, source, ordinary_package, profile=False
        )
        compile_package(args, source, package, profile=True)
        require_byte_identical_packages(ordinary_package, package)
        require_profile_companion_permissions(package)
    else:
        compile_package(args, source, package, profile=False)

    slices = load_boundary_slices(package)
    bindings = validate_manifest(package, args.launch_kind)
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
                "--all-ranks",
                "--no-card",
            ])
            verify_no_card_evidence(result.stdout)
            companion_ready = PROFILE_COMPANION_READY in result.stdout
            if companion_ready != (
                args.profile and no_card_package == package
            ):
                raise RuntimeError(
                    "no-card launch did not prove the exact profile companion "
                    "activation boundary"
                )
        print(f"no_card_launch_kind: {args.launch_kind}")
        print(f"no_card_profile: {str(args.profile).lower()}")
        return 0

    (
        resource_arguments,
        output_ids,
        entry_evidence,
        completion_evidence,
    ) = write_rank_payloads(args.work_dir, package, slices, bindings)
    command_tail = [
        "--all-ranks",
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
        ordinary_result = run(
            [
                str(args.wafer_run),
                "--package-dir",
                str(ordinary_package),
                *command_tail,
            ],
            timeout_seconds=(
                args.completion_timeout_ms / 1000.0
                + BOARD_PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        verify_board_evidence(
            ordinary_result.stdout,
            args.launch_kind,
            output_ids,
            entry_evidence,
            completion_evidence,
        )
        print("board_profile_ordinary_gate: pass")
        print(ordinary_result.stdout, end="")

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
                    PROFILE_CAMPAIGN_LAUNCH_COUNT
                    * args.completion_timeout_ms
                    / 1000.0
                    + BOARD_PROCESS_TIMEOUT_MARGIN_SECONDS
                ),
            ),
        )
        verify_board_evidence(
            profile_result.stdout,
            args.launch_kind,
            output_ids,
            entry_evidence,
            completion_evidence,
        )
        median, minimum, maximum, report = verify_profile_report(
            package, profile_result.stdout
        )
        print(
            "board_profile_campaign: pass "
            f"launches={PROFILE_CAMPAIGN_LAUNCH_COUNT} "
            f"samples={PROFILE_MEASUREMENT_SAMPLE_COUNT}"
        )
        print(f"board_profile_latency_ns: median={median} range={minimum}..{maximum}")
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
            args.launch_kind,
            output_ids,
            entry_evidence,
            completion_evidence,
        )
        print(
            "board_all_rank_add_iteration: "
            f"{iteration + 1}/{args.repeat} launch_kind={args.launch_kind}"
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
        print(f"wafer_board_all_rank_add_test: {error}", file=sys.stderr)
        raise SystemExit(1)
