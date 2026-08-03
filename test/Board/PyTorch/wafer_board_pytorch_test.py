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
from collections.abc import Sequence

import torch

import wafer_pytorch_board_common as common
import wafer_pytorch_board_cases as board_cases


TARGET_PROFILE = "wafer-tx81-single-card-kernel-v1"
LAUNCH_KIND = "kernel"
DIRECT_DTE_STATUS_ABI = "wafer-direct-dte-status-v2"
PROCESS_TIMEOUT_MARGIN_SECONDS = 60
COMPILE_TIMEOUT_SECONDS = 1800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(board_cases.CASE_FACTORIES), required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--compile-timing", action="store_true")
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


def prepare_work_dir(work_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    source = work_dir / "source-program"
    package = work_dir / "package"
    work_dir.mkdir(parents=True)
    return source, package


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
    rank: int,
) -> torch.Tensor:
    ranks = binding.get("ranks")
    if not isinstance(ranks, list):
        raise RuntimeError("distributed boundary ranks must be a list")
    records = [
        record
        for record in ranks
        if isinstance(record, dict) and record.get("rank") == rank
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"distributed boundary must contain exactly one record for rank {rank}"
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
        raise RuntimeError(f"rank {rank} distributed boundary geometry is invalid")
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
        raise RuntimeError(f"rank {rank} distributed boundary slice is invalid")
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, sizes, strict=True)
    )
    return tensor[slices].contiguous()


def _boundary_maps(
    package: pathlib.Path,
    case: board_cases.PyTorchBoardCase,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    metadata = json.loads(
        (package / "functions" / "forward.meta").read_text(encoding="utf-8")
    )
    boundary = metadata.get("distributed_boundary")
    if not isinstance(boundary, dict):
        if case.rank_count != 1:
            raise RuntimeError("multi-rank PyTorch case has no distributed boundary")
        return (
            {(0, index): tensor for index, tensor in enumerate(case.inputs)},
            {
                (0, index): tensor
                for index, tensor in enumerate(case.expected_outputs)
            },
        )
    if boundary.get("logical_rank_count") != case.rank_count:
        raise RuntimeError("distributed boundary rank count differs from the case")
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
        for rank in range(case.rank_count):
            # Manifest user_input.role_index preserves the function argument
            # identity even when earlier arguments are parameters.
            key = (rank, argument_index)
            if key in local_inputs:
                raise RuntimeError(f"duplicate distributed input binding: {key}")
            local_inputs[key] = _boundary_tensor(case.inputs[position], binding, rank)

    local_outputs: dict[tuple[int, int], torch.Tensor] = {}
    for binding in outputs:
        if not isinstance(binding, dict):
            raise RuntimeError("distributed output binding must be an object")
        result_index = binding.get("result_index")
        if not isinstance(result_index, int) or result_index >= len(case.expected_outputs):
            raise RuntimeError("distributed result index is invalid")
        for rank in range(case.rank_count):
            key = (rank, result_index)
            if key in local_outputs:
                raise RuntimeError(f"duplicate distributed output binding: {key}")
            local_outputs[key] = _boundary_tensor(
                case.expected_outputs[result_index], binding, rank
            )
    return local_inputs, local_outputs


def _manifest_resources(
    package: pathlib.Path, case: board_cases.PyTorchBoardCase
) -> tuple[
    dict[tuple[int, str, int], dict[str, object]],
    set[int],
]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("rank_count") != case.rank_count
        or manifest.get("target", {}).get("profile") != TARGET_PROFILE
    ):
        raise RuntimeError("PyTorch package target/rank contract is invalid")
    resources = manifest.get("resources")
    entries = manifest.get("entries")
    completions = manifest.get("completions")
    if not all(isinstance(value, list) for value in (resources, entries, completions)):
        raise RuntimeError("PyTorch package domains must be lists")
    expected_ranks = list(range(case.rank_count))
    if sorted(entry.get("rank") for entry in entries) != expected_ranks:
        raise RuntimeError("PyTorch package entries do not cover all ranks")
    if sorted(completion.get("rank") for completion in completions) != expected_ranks:
        raise RuntimeError("PyTorch package completions do not cover all ranks")

    host_resources: dict[tuple[int, str, int], dict[str, object]] = {}
    output_ids: set[int] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            raise RuntimeError("manifest resource must be an object")
        role = resource.get("role")
        if role not in ("user_input", "output"):
            continue
        rank = resource.get("rank")
        role_index = resource.get("role_index")
        resource_id = resource.get("id")
        if not all(isinstance(value, int) for value in (rank, role_index, resource_id)):
            raise RuntimeError("host tensor resource identity is invalid")
        key = (rank, role, role_index)
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
) -> tuple[list[str], dict[pathlib.Path, torch.Tensor], set[int]]:
    local_inputs, local_outputs = _boundary_maps(package, case)
    resources, output_ids = _manifest_resources(package, case)
    expected_keys = {
        *(rank_role_index for rank_role_index in (
            (rank, "user_input", index) for rank, index in local_inputs
        )),
        *(rank_role_index for rank_role_index in (
            (rank, "output", index) for rank, index in local_outputs
        )),
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
    for key, resource in sorted(resources.items()):
        rank, role, index = key
        tensor = (
            local_inputs[(rank, index)]
            if role == "user_input"
            else local_outputs[(rank, index)]
        )
        common.require_manifest_tensor(resource, tensor, context=str(key))
        resource_id = resource["id"]
        manifest_dtype = resource["type"]["dtype"]
        if role == "user_input":
            input_path = raw / (
                f"rank_{rank:02d}_{role}_{index}.{manifest_dtype}.raw"
            )
            common.write_tensor_raw(input_path, tensor)
            arguments.extend(["--resource", f"{resource_id}={input_path}"])
            continue
        capture_path = raw / (
            f"rank_{rank:02d}_output_{index}.capture.{manifest_dtype}.raw"
        )
        arguments.extend(["--output", f"{resource_id}={capture_path}"])
        captures[capture_path] = tensor
    if len(captures) != len(output_ids):
        raise RuntimeError("PyTorch output captures are not all-and-only")
    return arguments, captures, output_ids


def verify_no_card(stdout: str, case: board_cases.PyTorchBoardCase) -> None:
    required = {
        f"package: id=0 schema=6 ranks={case.rank_count}",
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


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if args.case != "rank-one-gemm":
        os.environ.setdefault("CPU_NUM_DEVICES", str(board_cases.RANK_COUNT))
        os.environ.setdefault("PJRT_DEVICE", "CPU")
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
    case = board_cases.make_case(args.case, dtype=dtype, seed=args.seed)
    if not args.no_card and args.expected_tile_count < case.rank_count:
        raise RuntimeError("PyTorch case exceeds the qualified tile count")
    source, package = prepare_work_dir(args.work_dir)
    case.export_program(source)
    compile_command = [
        str(args.wafer_compile),
        "--input-program-dir",
        str(source),
        "--output-program-dir",
        str(package),
        f"--execution-ranks={case.rank_count}",
        f"--target-profile={TARGET_PROFILE}",
        f"--launch-kind={LAUNCH_KIND}",
    ]
    if args.compile_timing:
        compile_command.append("--compile-timing")
    compile_result = run(
        compile_command,
        timeout_seconds=COMPILE_TIMEOUT_SECONDS,
    )
    if args.compile_timing:
        print(compile_result.stderr, end="", file=sys.stderr)
    if f"published verified package with execution-ranks={case.rank_count}" not in compile_result.stdout:
        raise RuntimeError("wafer-compile did not publish the PyTorch package")
    validate_structured_program(package, case)
    resource_arguments, captures, output_ids = prepare_runtime_payloads(
        args.work_dir, package, case
    )

    command = [str(args.wafer_run), "--package-dir", str(package)]
    if case.rank_count == 1:
        command.extend(["--entry-id", "0"])
    else:
        command.extend(["--all-ranks"])
    if args.no_card:
        if case.rank_count > 1:
            command.extend(
                [
                    "--direct-dte-status-abi",
                    DIRECT_DTE_STATUS_ABI,
                    "--supports-host-watchdog",
                ]
            )
        command.append("--no-card")
        result = run(command)
        verify_no_card(result.stdout, case)
        print(
            f"pytorch_board_no_card: case={case.name} "
            f"dtype={args.dtype} seed={args.seed} "
            "source_export=true torch_eager_reference=true"
        )
        print(result.stdout, end="")
        return 0

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
        result = run(
            command,
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        verify_board(result.stdout, case, output_ids, captures)
        print(
            f"pytorch_board_iteration: case={case.name} "
            f"dtype={args.dtype} seed={args.seed} "
            f"iteration={iteration + 1}/{args.repeat} torch_close=true"
        )
        print(result.stdout, end="")
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
