#!/usr/bin/env python3
"""Export, compile, and execute PyTorch-owned board tensor cases."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence

import torch

import wafer_pytorch_board_common as common
import wafer_pytorch_board_cases as board_cases


TARGET_IDENTITY = "wafer-tx81-single-card"
LAUNCH_KIND = "kernel"
PHYSICAL_TILE_COUNT = 16
PROCESS_TIMEOUT_MARGIN_SECONDS = 60
COMPILE_TIMEOUT_SECONDS = 1800
DIRECT_DTE_STATUS_ABI = "wafer-direct-dte-status"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(board_cases.CASE_FACTORIES), required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dump-compiler-ir", type=pathlib.Path)
    parser.add_argument("--compile-timing", action="store_true")
    parser.add_argument(
        "--optimization-policy",
        choices=board_cases.OPTIMIZATION_POLICIES,
        default="search",
    )
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def run(
    command: list[str], *, timeout_seconds: float | None = None
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
            "one-shot PyTorch board vertical exceeded its bounded deadline; "
            "the test will not retry or invoke reset/power"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}"
        )
    return result


def validate_compiler_search_evidence(
    stderr: str,
    case: board_cases.PyTorchBoardCase,
    optimization_policy: str,
) -> None:
    minimum = getattr(case, "minimum_search_actual_fused_edges", 0)
    if optimization_policy != "search" or minimum == 0:
        return
    selected = re.findall(
        r"^wafer-compile: card-executable-selection .*?"
        r"\bactual_fused_edges=(\d+)(?:\s|$)",
        stderr,
        re.MULTILINE,
    )
    if len(selected) != 1:
        raise RuntimeError(
            "search-policy compiler output omitted the unique selected "
            "card fusion evidence"
        )
    actual = int(selected[0])
    if actual < minimum:
        raise RuntimeError(
            "search-policy card winner did not materialize the required "
            f"operator fusion: expected at least {minimum}, got {actual}"
        )


def prepare_work_dir(work_dir: pathlib.Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)


def case_step_paths(
    work_dir: pathlib.Path, *, step_index: int, is_chain: bool
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    step_dir = (
        work_dir / f"step_{step_index + 1:02d}" if is_chain else work_dir
    )
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir, step_dir / "source-program", step_dir / "package"


def validate_structured_program(
    package: pathlib.Path, case: board_cases.PyTorchBoardCase
) -> None:
    structured_ir = (package / "functions" / "forward.mlir").read_text(
        encoding="utf-8"
    )
    missing = [
        fragment
        for fragment in case.required_structured_ir
        if fragment not in structured_ir
    ]
    if missing:
        raise RuntimeError(
            f"PyTorch case {case.name} structured IR omitted: {missing}"
        )
    if "stablehlo." in structured_ir:
        raise RuntimeError(
            f"PyTorch case {case.name} retained StableHLO after normalization"
        )
    all_reduce_count = structured_ir.count(
        "wafer.linalg_ext.collective.all_reduce"
    )
    expected_all_reduce_count = case.expected_all_reduce_count
    if all_reduce_count != expected_all_reduce_count:
        raise RuntimeError(
            f"PyTorch case {case.name} expected {expected_all_reduce_count} "
            f"AllReduce op(s), got {all_reduce_count}"
        )


def _boundary_tensor(
    tensor: torch.Tensor,
    binding: dict[str, object],
    partition_id: int,
) -> torch.Tensor:
    partitions = binding.get("partitions")
    if not isinstance(partitions, list):
        raise RuntimeError("distributed boundary partitions must be a list")
    records = [
        record
        for record in partitions
        if isinstance(record, dict) and record.get("partition_id") == partition_id
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"distributed boundary must contain exactly one record for partition_id {partition_id}"
        )
    record = records[0]
    offsets = record.get("offsets")
    sizes = record.get("sizes")
    strides = record.get("strides")
    if (
        not isinstance(offsets, list)
        or not isinstance(sizes, list)
        or not isinstance(strides, list)
        or len(offsets) != tensor.ndim
        or len(sizes) != tensor.ndim
        or strides != [1] * tensor.ndim
    ):
        raise RuntimeError(f"partition_id {partition_id} distributed boundary geometry is invalid")
    if not all(
        isinstance(offset, int)
        and isinstance(size, int)
        and offset >= 0
        and size >= 0
        and offset + size <= extent
        for offset, size, extent in zip(
            offsets, sizes, tensor.shape, strict=True
        )
    ):
        raise RuntimeError(f"partition_id {partition_id} distributed boundary slice is invalid")
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, sizes, strict=True)
    )
    return tensor[slices].contiguous()


def _boundary_maps(
    package: pathlib.Path,
    case: board_cases.PyTorchBoardCase,
    expected_outputs: tuple[torch.Tensor, ...],
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    metadata = json.loads(
        (package / "functions" / "forward.meta").read_text(encoding="utf-8")
    )
    boundary = metadata.get("distributed_boundary")
    if not isinstance(boundary, dict):
        if case.num_partitions != 1:
            raise RuntimeError("multi-partition_id PyTorch case has no distributed boundary")
        return (
            {(0, index): tensor for index, tensor in enumerate(case.inputs)},
            {
                (0, index): tensor
                for index, tensor in enumerate(expected_outputs)
            },
        )
    if boundary.get("num_partitions") != case.num_partitions:
        raise RuntimeError(
            "distributed boundary partition count differs from the case"
        )
    locations = metadata.get("input_locations")
    inputs = boundary.get("inputs")
    outputs = boundary.get("outputs")
    if not isinstance(locations, list) or not isinstance(inputs, list) or not isinstance(outputs, list):
        raise RuntimeError("distributed boundary domains must be lists")

    local_inputs: dict[tuple[int, int], torch.Tensor] = {}
    for binding in inputs:
        if not isinstance(binding, dict):
            raise RuntimeError("distributed input binding must be an object")
        argument_index = binding.get("argument_index")
        if not isinstance(argument_index, int) or argument_index >= len(locations):
            raise RuntimeError("distributed input argument index is invalid")
        location = locations[argument_index]
        if not isinstance(location, dict) or location.get("type_") != "input_arg":
            raise RuntimeError("distributed runtime input is not an input_arg")
        position = location.get("position")
        if not isinstance(position, int) or position >= len(case.inputs):
            raise RuntimeError("distributed input position is invalid")
        for partition_id in range(case.num_partitions):
            # Manifest user_input.role_index preserves the function argument
            # identity even when earlier arguments are parameters.
            key = (partition_id, argument_index)
            if key in local_inputs:
                raise RuntimeError(f"duplicate distributed input binding: {key}")
            local_inputs[key] = _boundary_tensor(
                case.inputs[position], binding, partition_id
            )

    local_outputs: dict[tuple[int, int], torch.Tensor] = {}
    for binding in outputs:
        if not isinstance(binding, dict):
            raise RuntimeError("distributed output binding must be an object")
        result_index = binding.get("result_index")
        if not isinstance(result_index, int) or result_index >= len(expected_outputs):
            raise RuntimeError("distributed result index is invalid")
        for partition_id in range(case.num_partitions):
            key = (partition_id, result_index)
            if key in local_outputs:
                raise RuntimeError(f"duplicate distributed output binding: {key}")
            local_outputs[key] = _boundary_tensor(
                expected_outputs[result_index], binding, partition_id
            )
    return local_inputs, local_outputs


def _manifest_resources(package: pathlib.Path) -> tuple[
    dict[tuple[str, int], dict[str, object]],
    set[int],
]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("card_count") != 1
        or manifest.get("tile_count") != PHYSICAL_TILE_COUNT
        or manifest.get("target", {}).get("identity") != TARGET_IDENTITY
    ):
        raise RuntimeError("PyTorch package physical target contract is invalid")
    resources = manifest.get("resources")
    entries = manifest.get("entries")
    if not isinstance(resources, list) or not isinstance(entries, list):
        raise RuntimeError("PyTorch package domains must be lists")

    physical_tiles: set[int] = set()
    launch_slots: set[int] = set()
    if len(entries) != PHYSICAL_TILE_COUNT:
        raise RuntimeError("PyTorch package must contain one entry per Tile")
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("manifest entry must be an object")
        card_id = entry.get("card_id")
        tile_id = entry.get("tile_id")
        launch_slot = entry.get("launch_slot")
        if (
            type(card_id) is not int
            or type(tile_id) is not int
            or type(launch_slot) is not int
            or card_id != 0
            or not 0 <= tile_id < PHYSICAL_TILE_COUNT
            or not 0 <= launch_slot < PHYSICAL_TILE_COUNT
            or entry.get("completion") != "return_after_local_drain"
        ):
            raise RuntimeError("manifest entry physical binding is invalid")
        if tile_id in physical_tiles or launch_slot in launch_slots:
            raise RuntimeError("manifest entry physical binding is duplicated")
        physical_tiles.add(tile_id)
        launch_slots.add(launch_slot)
    expected_domain = set(range(PHYSICAL_TILE_COUNT))
    if physical_tiles != expected_domain or launch_slots != expected_domain:
        raise RuntimeError("manifest entries do not cover the Tile domains")

    host_resources: dict[tuple[str, int], dict[str, object]] = {}
    output_ids: set[int] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            raise RuntimeError("manifest resource must be an object")
        role = resource.get("role")
        if role not in ("user_input", "output"):
            continue
        role_index = resource.get("role_index")
        resource_id = resource.get("id")
        if (
            type(role_index) is not int
            or type(resource_id) is not int
            or role_index < 0
            or resource_id < 0
            or resource.get("scope") != {"kind": "card", "card_id": 0}
        ):
            raise RuntimeError("host tensor resource identity is invalid")
        key = (role, role_index)
        if key in host_resources or resource.get("host_visible") is not True:
            raise RuntimeError(f"invalid duplicate/non-visible host resource: {key}")
        host_resources[key] = resource
        if role == "output":
            output_ids.add(resource_id)
    return host_resources, output_ids


def prepare_runtime_payloads(
    work_dir: pathlib.Path,
    package: pathlib.Path,
    case: board_cases.PyTorchBoardCase,
    expected_outputs: tuple[torch.Tensor, ...],
) -> tuple[
    list[str],
    dict[pathlib.Path, torch.Tensor],
    set[int],
    dict[int, pathlib.Path],
]:
    local_inputs, local_outputs = _boundary_maps(package, case, expected_outputs)
    if (
        case.num_partitions != 1
        or any(partition_id != 0 for partition_id, _ in local_inputs)
        or any(partition_id != 0 for partition_id, _ in local_outputs)
    ):
        raise RuntimeError(
            "single-card package writing requires one source partition"
        )
    resources, output_ids = _manifest_resources(package)
    expected_keys = {
        *(("user_input", index) for _, index in local_inputs),
        *(("output", index) for _, index in local_outputs),
    }
    if set(resources) != expected_keys:
        raise RuntimeError(
            "manifest host tensor resources differ from PyTorch boundary: "
            f"missing={sorted(expected_keys - set(resources))} "
            f"extra={sorted(set(resources) - expected_keys)}"
        )

    raw = work_dir / "raw"
    raw.mkdir()
    arguments: list[str] = []
    captures: dict[pathlib.Path, torch.Tensor] = {}
    result_capture_paths: dict[int, pathlib.Path] = {}
    for key, resource in sorted(resources.items()):
        role, index = key
        tensor = (
            local_inputs[(0, index)]
            if role == "user_input"
            else local_outputs[(0, index)]
        )
        common.require_manifest_tensor(resource, tensor, context=str(key))
        resource_id = resource["id"]
        manifest_dtype = resource["type"]["dtype"]
        if role == "user_input":
            input_path = raw / (
                f"card_00_{role}_{index}.{manifest_dtype}.raw"
            )
            common.write_tensor_raw(input_path, tensor)
            arguments.extend(["--resource", f"{resource_id}={input_path}"])
            continue
        capture_path = raw / (
            f"card_00_output_{index}.capture.{manifest_dtype}.raw"
        )
        arguments.extend(["--output", f"{resource_id}={capture_path}"])
        captures[capture_path] = tensor
        result_capture_paths[index] = capture_path
    if len(captures) != len(output_ids):
        raise RuntimeError("PyTorch output captures are not all-and-only")
    return arguments, captures, output_ids, result_capture_paths


def read_single_card_continuation_outputs(
    result_capture_paths: dict[int, pathlib.Path],
    expected_outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    expected_keys = set(range(len(expected_outputs)))
    if set(result_capture_paths) != expected_keys:
        raise RuntimeError(
            "functional continuation requires complete single-card result "
            "captures"
        )
    return tuple(
        common.read_tensor_raw(
            result_capture_paths[index],
            dtype=expected.dtype,
            shape=expected.shape,
        )
        for index, expected in enumerate(expected_outputs)
    )


def verify_no_card(stdout: str) -> None:
    required = {
        "package: id=0 cards=1 tiles=16",
        "board_execution: false",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("no-card output omitted PyTorch package evidence")


def verify_board(
    stdout: str,
    case: board_cases.PyTorchBoardCase,
    output_ids: set[int],
    captures: dict[pathlib.Path, torch.Tensor],
) -> None:
    required = {
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "board_execution: true",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("board output omitted PyTorch lifecycle evidence")
    matches = re.findall(
        r"^output_capture: resource=(\d+) bytes=\d+ path=.+$",
        stdout,
        re.MULTILINE,
    )
    if len(matches) != len(output_ids) or {int(value) for value in matches} != output_ids:
        raise RuntimeError("board output did not capture every PyTorch output")
    for capture, expected in captures.items():
        common.assert_raw_capture_matches(
            capture,
            expected,
            policy=case.comparison_policy,
            context=f"{case.name} {capture.name}",
        )


def prepare_case_step(
    args: argparse.Namespace,
    case: board_cases.PyTorchBoardCase,
    *,
    step_index: int,
    step_dir: pathlib.Path,
    source: pathlib.Path,
    package: pathlib.Path,
    dump_compiler_ir: pathlib.Path | None,
) -> tuple[
    tuple[torch.Tensor, ...],
    list[str],
    dict[pathlib.Path, torch.Tensor],
    set[int],
    dict[int, pathlib.Path],
]:
    step_number = step_index + 1
    export_start_ns = time.monotonic_ns()
    case.export_program(source)
    print(
        "pytorch-board-timing stage=source-export "
        f"step={step_number} "
        f"wall_ms={(time.monotonic_ns() - export_start_ns) // 1_000_000}"
    )
    compile_command = [
        str(args.wafer_compile),
        "--input-program-dir",
        str(source),
        "--output-program-dir",
        str(package),
        f"--num-partitions={case.num_partitions}",
        f"--launch-kind={LAUNCH_KIND}",
        f"--optimization-policy={args.optimization_policy}",
    ]
    if args.compile_timing:
        compile_command.append("--compile-timing")
    if dump_compiler_ir is not None:
        compile_command.extend(
            ["--dump-compiler-ir", str(dump_compiler_ir)]
        )
    compile_result = run(
        compile_command,
        timeout_seconds=COMPILE_TIMEOUT_SECONDS,
    )
    validate_compiler_search_evidence(
        compile_result.stderr, case, args.optimization_policy
    )
    if args.compile_timing:
        print(compile_result.stderr, end="", file=sys.stderr)
    if (
        "wrote verified package with num-partitions=1 tiles=16"
        not in compile_result.stdout
    ):
        raise RuntimeError("wafer-compile did not write the PyTorch package")
    if dump_compiler_ir is not None:
        expected_stems = [
            f"tile_{tile_id:05d}" for tile_id in range(PHYSICAL_TILE_COUNT)
        ]
        tile_files = sorted(
            (dump_compiler_ir / "tile-dataflow").glob("tile_*.mlir")
        )
        instruction_files = sorted(
            (dump_compiler_ir / "instruction").glob("tile_*.mlir")
        )
        target_files = sorted(
            (dump_compiler_ir / "target-llvm").glob("tile_*.ll")
        )
        if (
            [path.stem for path in tile_files] != expected_stems
            or [path.stem for path in instruction_files] != expected_stems
            or [path.stem for path in target_files] != expected_stems
            or any(
                path.stat().st_size == 0
                for path in (*tile_files, *instruction_files, *target_files)
            )
        ):
            raise RuntimeError("compiler IR dump is incomplete")
        for tile_file in tile_files:
            tile_ir = tile_file.read_text(encoding="utf-8")
            if "wafer.instr." in tile_ir:
                raise RuntimeError(
                    "compiler Tile/dataflow evidence is not a selected "
                    "pre-Instr IR"
                )
            # A selected MPMD candidate still contains all-and-only the 16
            # Tile interfaces. Tiles outside the winner's active
            # placement intentionally contain the typed function boundary and
            # observable empty results but no TileRegion. Accept that canonical
            # inactive pre-Instr IR; requiring every interface to carry
            # work would invalidate the legal less-than-16-Tile spatial axis.
            if (
                "wafer.tile.region" not in tile_ir
                and (
                    "func.func @main" not in tile_ir
                    or "wafer.tile." in tile_ir
                    or "memref.alloc" in tile_ir
                )
            ):
                raise RuntimeError(
                    "compiler inactive Tile/dataflow evidence is not a "
                    "canonical selected pre-Instr IR"
                )
    validate_structured_program(package, case)

    oracle_start_ns = time.monotonic_ns()
    expected_outputs = case.materialize_expected_outputs()
    print(
        "pytorch-board-timing stage=torch-eager-reference "
        f"step={step_number} "
        f"wall_ms={(time.monotonic_ns() - oracle_start_ns) // 1_000_000}"
    )
    payload_start_ns = time.monotonic_ns()
    (
        resource_arguments,
        captures,
        output_ids,
        result_capture_paths,
    ) = prepare_runtime_payloads(
        step_dir, package, case, expected_outputs
    )
    print(
        "pytorch-board-timing stage=runtime-payload "
        f"step={step_number} "
        f"wall_ms={(time.monotonic_ns() - payload_start_ns) // 1_000_000}"
    )
    return (
        expected_outputs,
        resource_arguments,
        captures,
        output_ids,
        result_capture_paths,
    )


def base_runtime_command(
    wafer_run: pathlib.Path,
    package: pathlib.Path,
) -> list[str]:
    return [
        str(wafer_run),
        "--package-dir",
        str(package),
    ]


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("PyTorch board vertical hardware execution is not armed")
        return 77
    if not args.no_card:
        qualification = (
            args.expected_runtime_version,
            args.expected_device_name,
            args.expected_pci_bus_id,
            args.expected_tile_count,
            args.expected_runtime_library_sha256,
        )
        if any(value is None for value in qualification):
            raise RuntimeError("board execution requires complete qualification arguments")

    dtype = board_cases.parse_torch_dtype(args.dtype)
    case_start_ns = time.monotonic_ns()
    case = board_cases.make_case(args.case, dtype=dtype, seed=args.seed)
    os.environ["CPU_NUM_DEVICES"] = str(case.num_partitions)
    os.environ["PJRT_DEVICE"] = "CPU"
    print(
        "pytorch-board-timing stage=case-materialization "
        "step=1 "
        f"wall_ms={(time.monotonic_ns() - case_start_ns) // 1_000_000}"
    )
    if not args.no_card and args.expected_tile_count != PHYSICAL_TILE_COUNT:
        raise RuntimeError("PyTorch package requires the qualified 16-Tile card")
    is_chain = case.continuation_factory is not None
    prepare_work_dir(args.work_dir)
    current_case = case
    step_index = 0
    while True:
        step_dir, source, package = case_step_paths(
            args.work_dir,
            step_index=step_index,
            is_chain=is_chain,
        )
        dump_compiler_ir = args.dump_compiler_ir
        if dump_compiler_ir is not None and is_chain:
            dump_compiler_ir = (
                dump_compiler_ir / f"step_{step_index + 1:02d}"
            )
        (
            expected_outputs,
            resource_arguments,
            captures,
            output_ids,
            result_capture_paths,
        ) = prepare_case_step(
            args,
            current_case,
            step_index=step_index,
            step_dir=step_dir,
            source=source,
            package=package,
            dump_compiler_ir=dump_compiler_ir,
        )

        command = base_runtime_command(args.wafer_run, package)
        if args.no_card:
            # Direct DTE is selected by the common card search, so a
            # board-ready no-card runner must advertise the same transport
            # capabilities regardless of which candidate wins.  This remains
            # side-effect-free validation; it does not claim hardware execution.
            command.extend(
                [
                    "--no-card",
                    "--direct-dte-status-abi",
                    DIRECT_DTE_STATUS_ABI,
                    "--supports-host-watchdog",
                ]
            )
            result = run(command)
            verify_no_card(result.stdout)
            print(
                f"pytorch_board_no_card: case={current_case.name} "
                f"dtype={args.dtype} seed={args.seed} "
                f"step={step_index + 1} "
                f"optimization_policy={args.optimization_policy} "
                "source_export=true torch_eager_reference=true "
                "runtime_payload=true"
            )
            print(result.stdout, end="")
            continuation_outputs = expected_outputs
        else:
            command.extend(
                [
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
            )
            for iteration in range(args.repeat):
                iteration_start_ns = time.monotonic_ns()
                result = run(
                    command,
                    timeout_seconds=(
                        args.completion_timeout_ms / 1000
                        + PROCESS_TIMEOUT_MARGIN_SECONDS
                    ),
                )
                verify_board(
                    result.stdout,
                    current_case,
                    output_ids,
                    captures,
                )
                print(
                    f"pytorch_board_iteration: case={current_case.name} "
                    f"dtype={args.dtype} seed={args.seed} "
                    f"step={step_index + 1} "
                    f"optimization_policy={args.optimization_policy} "
                    f"iteration={iteration + 1}/{args.repeat} "
                    f"wall_ms="
                    f"{(time.monotonic_ns() - iteration_start_ns) // 1_000_000} "
                    "torch_close=true"
                )
                print(result.stdout, end="")
            continuation_outputs = (
                read_single_card_continuation_outputs(
                    result_capture_paths, expected_outputs
                )
                if current_case.continuation_factory is not None
                else expected_outputs
            )

        continuation_factory = current_case.continuation_factory
        if continuation_factory is None:
            break
        if current_case.num_partitions != 1 or step_index != 0:
            raise RuntimeError(
                "functional state chain must be one bounded single-card "
                "continuation"
            )
        continuation_start_ns = time.monotonic_ns()
        current_case = continuation_factory(continuation_outputs)
        step_index += 1
        print(
            "pytorch-board-timing stage=case-materialization "
            f"step={step_index + 1} "
            f"wall_ms="
            f"{(time.monotonic_ns() - continuation_start_ns) // 1_000_000}"
        )
        if current_case.num_partitions != 1:
            raise RuntimeError("PyTorch continuation changed source partitions")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AssertionError,
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"wafer_board_pytorch_test: {error}", file=sys.stderr)
        raise SystemExit(1)
