#!/usr/bin/env python3
"""Export, compile, and execute PyTorch-owned board tensor cases."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
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
    parser.add_argument(
        "--prepared-work-dir", type=pathlib.Path,
        help="reuse this round's source/package; write fresh payloads to --work-dir",
    )
    parser.add_argument("--dump-compiler-ir", type=pathlib.Path)
    parser.add_argument("--compile-timing", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-trace-event-limit", type=int)
    parser.add_argument("--device-timing", action="store_true")
    parser.add_argument(
        "--qualify-communication",
        choices=(
            "ring-allgather", "direct-alltoall", "direct-reduce-scatter",
            "all-reduce", "shared-input", "pipelined-loads",
        ),
        help="explicit implementation qualification using wafer-compile-test",
    )
    parser.add_argument(
        "--optimization-policy",
        choices=board_cases.OPTIMIZATION_POLICIES,
        default="none",
    )
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--target-model", action="store_true",
                        help="compare full outputs through wafer-compile-test's functional model")
    parser.add_argument("--target-model-max-scalar-evaluations", type=int, default=10000000)
    parser.add_argument("--target-model-max-fused-multiply-adds", type=int, default=1000000)
    parser.add_argument("--target-model-max-movement-bytes", type=int, default=536870912)
    parser.add_argument("--target-model-max-movement-segments", type=int, default=1000000)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--search-width", type=int)
    parser.add_argument("--search-trials", type=int)
    parser.add_argument("--compile-timeout-seconds", type=int, default=COMPILE_TIMEOUT_SECONDS)
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
            f"host process deadline exceeded after {error.timeout}s: {command[0]}"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}"
        )
    return result


def verify_widened_convolution(directory: pathlib.Path, dtype: torch.dtype) -> None:
    element, format_code = {torch.float16: ("f16", 2), torch.bfloat16: ("bf16", 3)}[dtype]
    operations = [
        line for path in sorted((directory / "instruction").glob("tile_*.mlir"))
        for line in path.read_text().splitlines() if "wafer.instr.conv " in line
    ]
    if not operations:
        raise RuntimeError("biased convolution has no actual native convolution")
    for line in operations:
        types = re.findall(r"memref<[^>]+>", line)
        if (len(types) != 3 or f"x{element}," not in types[0]
                or f"x{element}," not in types[1] or "xf32," not in types[2]):
            raise RuntimeError("convolution did not preserve low-precision inputs and F32 output")
    calls = [
        line for path in sorted((directory / "target-llvm").glob("tile_*.ll"))
        for line in path.read_text().splitlines() if "call void @wafer_tx81_conv(" in line
    ]
    if len(calls) != len(operations) or any(
        re.search(rf"i32 {format_code}, i32 5, i32 [0-9]+\)$", line) is None
        for line in calls
    ):
        raise RuntimeError("convolution target call lost its independent input/output formats")


def verify_ordered_convolution(directory: pathlib.Path) -> None:
    operations = [
        line for path in sorted((directory / "instruction").glob("tile_*.mlir"))
        for line in path.read_text().splitlines()
    ]
    if any("wafer.instr.conv " in line for line in operations):
        raise RuntimeError("ordered convolution was replaced by native accumulation")
    for kind in ("mul", "add"):
        matching = [line for line in operations
                    if f"wafer.instr.elementwise <{kind}>" in line and "xf32," in line]
        if not matching:
            raise RuntimeError(f"ordered convolution omitted its F32 {kind}")


def prepare_work_dir(work_dir: pathlib.Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)


def validate_prepared_directories(
    prepared: pathlib.Path, output: pathlib.Path,
) -> None:
    prepared, output = prepared.resolve(), output.resolve()
    if output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise RuntimeError("prepared and output directories must not overlap")
    if not prepared.is_dir():
        raise RuntimeError("prepared work directory does not exist")
    if output.exists():
        raise RuntimeError("reuse requires a fresh output directory")


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_output_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if not expected.dtype.is_floating_point:
        mismatches = int(torch.count_nonzero(actual != expected).item())
        return {
            "shape": list(expected.shape), "dtype": str(expected.dtype),
            "elements": expected.numel(), "mismatched_elements": mismatches,
            "max_abs_error": 0 if mismatches == 0 else None,
        }
    maximum = (actual.float() - expected.float()).abs().max().item()
    return {
        "shape": list(expected.shape), "dtype": str(expected.dtype),
        "elements": expected.numel(),
        "max_abs_error": maximum if math.isfinite(maximum) else None,
    }


def verify_prepared_source(prepared: pathlib.Path, fresh: pathlib.Path) -> None:
    def contents(directory: pathlib.Path) -> dict[str, str]:
        return {
            str(path.relative_to(directory)): file_sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
        }
    original, current = contents(prepared), contents(fresh)
    if not original or original != current:
        changed = sorted(key for key in original.keys() | current.keys()
                         if original.get(key) != current.get(key))
        raise RuntimeError(f"prepared source differs from current case: {changed}")


def case_step_paths(
    work_dir: pathlib.Path, *, step_index: int, is_chain: bool
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    step_dir = (
        work_dir / f"step_{step_index + 1:02d}" if is_chain else work_dir
    )
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir, step_dir / "source-program", step_dir / "package"


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
    source_program: pathlib.Path,
    case: board_cases.PyTorchBoardCase,
    expected_outputs: tuple[torch.Tensor, ...],
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    metadata = json.loads(
        (source_program / "functions" / "forward.meta").read_text(
            encoding="utf-8"
        )
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


def _manifest_ports(package: pathlib.Path) -> tuple[
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
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    entries = manifest.get("entries")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or not isinstance(entries, list)
    ):
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

    host_ports: dict[tuple[str, int], dict[str, object]] = {}
    output_ids: set[int] = set()
    for table, role in ((inputs, "user_input"), (outputs, "output")):
        for record in table:
            if not isinstance(record, dict):
                raise RuntimeError("manifest port record must be an object")
            role_index = record.get("role_index")
            port_id = record.get("id")
            if (
                type(role_index) is not int
                or type(port_id) is not int
                or role_index < 0
                or port_id < 0
            ):
                raise RuntimeError("host tensor port identity is invalid")
            key = (role, role_index)
            if key in host_ports:
                raise RuntimeError(f"invalid duplicate host port: {key}")
            host_ports[key] = record
            if role == "output":
                output_ids.add(port_id)
    return host_ports, output_ids


def prepare_runtime_payloads(
    work_dir: pathlib.Path,
    source_program: pathlib.Path,
    package: pathlib.Path,
    case: board_cases.PyTorchBoardCase,
    expected_outputs: tuple[torch.Tensor, ...],
) -> tuple[
    list[str],
    dict[pathlib.Path, torch.Tensor],
    set[int],
    dict[int, pathlib.Path],
]:
    local_inputs, local_outputs = _boundary_maps(
        source_program, case, expected_outputs
    )
    if (
        case.num_partitions != 1
        or any(partition_id != 0 for partition_id, _ in local_inputs)
        or any(partition_id != 0 for partition_id, _ in local_outputs)
    ):
        raise RuntimeError(
            "single-card package writing requires one source partition"
        )
    ports, output_ids = _manifest_ports(package)
    expected_keys = {
        *(("user_input", index) for _, index in local_inputs),
        *(("output", index) for _, index in local_outputs),
    }
    if set(ports) != expected_keys:
        raise RuntimeError(
            "manifest host tensor resources differ from PyTorch boundary: "
            f"missing={sorted(expected_keys - set(ports))} "
            f"extra={sorted(set(ports) - expected_keys)}"
        )

    raw = work_dir / "raw"
    raw.mkdir()
    arguments: list[str] = []
    captures: dict[pathlib.Path, torch.Tensor] = {}
    result_capture_paths: dict[int, pathlib.Path] = {}
    for key, record in sorted(ports.items()):
        role, index = key
        tensor = (
            local_inputs[(0, index)]
            if role == "user_input"
            else local_outputs[(0, index)]
        )
        common.require_manifest_tensor(record, tensor, context=str(key))
        port_id = record["id"]
        manifest_dtype = record["dtype"]
        if role == "user_input":
            input_path = raw / (
                f"card_00_{role}_{index}.{manifest_dtype}.raw"
            )
            common.write_tensor_raw(input_path, tensor)
            arguments.extend(["--resource", f"{port_id}={input_path}"])
            continue
        capture_path = raw / (
            f"card_00_output_{index}.capture.{manifest_dtype}.raw"
        )
        common.write_tensor_raw(
            raw / f"card_00_output_{index}.expected.{manifest_dtype}.raw", tensor
        )
        arguments.extend(["--output", f"{port_id}={capture_path}"])
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
        "arithmetic_execution: false",
        "completion_execution: false",
        "numeric_readback: false",
        "board_execution: false",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("no-card output omitted PyTorch package evidence")


def verify_pre_instruction_boundary(tile_ir: str, instruction_ir: str) -> None:
    if "wafer.instr." in tile_ir:
        raise RuntimeError("compiler Tile/dataflow evidence is not a selected pre-Instr IR")
    # Inactive interfaces can return a typed DDR allocation and retain the
    # source function's symbol. Neither spelling nor allocation alone proves
    # computation. The compiler owns typed IR verification; this check only
    # cross-checks the two already-verified dump stages for unexpected work.
    if "wafer.tile.region" not in tile_ir and "wafer.instr." in instruction_ir:
        raise RuntimeError("compiler inactive Tile/dataflow evidence gained instructions")


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
        r"^output_capture: port=(\d+) bytes=\d+ path=.+$",
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


def verify_ring_allgather(
    package: pathlib.Path, dump: pathlib.Path, payload_bytes: int
) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    for entry in manifest["entries"]:
        if not any(arg["kind"] == "transport_status" for arg in entry["arguments"]):
            raise RuntimeError(
                "AllGather requires actual Direct-DTE status on every Tile"
            )
        if any(arg["kind"] == "shared_workspace" for arg in entry["arguments"]):
            raise RuntimeError("AllGather unexpectedly uses a shared DDR boundary")
    for tile in range(PHYSICAL_TILE_COUNT):
        ir = (dump / "instruction" / f"tile_{tile:05d}.mlir").read_text()
        tokens: list[str] = []
        for operation in ("send", "recv"):
            issues = re.findall(
                rf"(%[\w]+) = wafer\.instr\.dte_{operation} .*?"
                r"bytes = (\d+) : i64, message = .*?round = (\d+),",
                ir,
            )
            if (
                len(issues) != 15
                or sorted(int(round_) for _, _, round_ in issues) != list(range(15))
                or any(int(size) != payload_bytes for _, size, _ in issues)
            ):
                raise RuntimeError(
                    f"Tile {tile} lacks exact 15-round AllGather {operation}"
                )
            tokens.extend(token for token, _, _ in issues)
        waits = re.findall(r"wafer\.instr\.dte_wait (%[\w]+) :", ir)
        if sorted(waits) != sorted(tokens):
            raise RuntimeError(f"Tile {tile} lacks exact AllGather token waits")
        pending: dict[int, str] = {}
        for line in ir.splitlines():
            receive = re.search(
                r"(%[\w]+) = wafer\.instr\.dte_recv .*peer = (\d+) : i64",
                line,
            )
            if receive:
                token, peer_text = receive.groups()
                peer = int(peer_text)
                if peer in pending:
                    raise RuntimeError(f"Tile {tile} overwrites peer {peer} ready")
                pending[peer] = token
            wait = re.search(r"wafer\.instr\.dte_wait (%[\w]+) :", line)
            if wait:
                pending = {
                    peer: token for peer, token in pending.items()
                    if token != wait.group(1)
                }
    print(
        "qualified_allgather: tiles=16 rounds=15 sends=240 receives=240 "
        f"waits=480 payload_bytes={payload_bytes}"
    )



def verify_reduce_scatter_contributions(
    ir: str, tile: int, extent: int, output_shard: tuple[int, int] | None
) -> tuple[int, int]:
    """Follow the actual receive buffers into the selected sum's input slots."""
    regions: list[str] = []
    lines: list[str] = []
    depth = 0
    for line in ir.splitlines():
        if not lines and "wafer.tile.region(" not in line:
            continue
        lines.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            region = "\n".join(lines)
            if "wafer.tile.peer_recv " in region:
                regions.append(region)
            lines = []
    if len(regions) != 1:
        raise RuntimeError("ReduceScatter requires one actual contribution exchange region")
    body = regions[0]
    views = {
        result: (source, tuple(map(int, offsets.split(", "))),
                 tuple(map(int, sizes.split(", "))))
        for result, source, offsets, sizes in re.findall(
            r"(%[\w]+) = memref\.subview (%[\w]+)\[([\d, ]+)\] "
            r"\[([\d, ]+)\] \[1, 1, 1\]", body
        )
    }
    layouts = dict(re.findall(
        r"(%[\w]+) = wafer\.tile\.materialize_layout (%[\w]+) :", body
    ))
    reductions = re.findall(
        r"wafer\.tile\.reduce <sum> (%[\w]+), %\w+ "
        r"\{dimensions = array<i64: 0>\} : \(memref<16x(\d+)x1xf16,", body
    )
    if len(reductions) != 1:
        raise RuntimeError("ReduceScatter lacks the full 16-source sum")
    size = int(reductions[0][1])

    def before_layout(value: str) -> str:
        visited: set[str] = set()
        while value in layouts:
            if value in visited:
                raise RuntimeError("ReduceScatter layout chain contains a cycle")
            visited.add(value)
            value = layouts[value]
        return value

    reduced = before_layout(reductions[0][0])
    reduced_view = views.get(reduced)
    if reduced_view is None:
        assembled, assembly_column = reduced, 0
    else:
        assembled, offsets, sizes = reduced_view
        if offsets[0] != 0 or offsets[2] != 0 or sizes != (16, size, 1):
            raise RuntimeError("ReduceScatter sum consumes the wrong shard")
        assembly_column = offsets[1]
    offset = None if output_shard is None else output_shard[0]
    if output_shard is not None and size != output_shard[1]:
        raise RuntimeError("ReduceScatter sum has the wrong shard size")
    receives = {
        buffer: int(peer) for buffer, peer in re.findall(
            r"wafer\.tile\.peer_recv (%[\w]+) \{.*?peer = (\d+) : i64", body
        )
    }
    contributions: dict[int, str] = {}
    copies = re.findall(r"memref\.copy (%[\w]+), (%[\w]+) :", body)
    copies += re.findall(r"wafer\.tile\.copy_into (%[\w]+) into (%[\w]+) :", body)
    for source, destination in copies:
        source = before_layout(source)
        view = views.get(destination)
        if view is None or view[0] != assembled:
            continue
        participant, column, last = view[1]
        if (column, last) != (assembly_column, 0) or view[2] != (1, size, 1):
            raise RuntimeError("ReduceScatter contribution has the wrong window")
        if participant in contributions:
            raise RuntimeError("ReduceScatter duplicates a contribution")
        if participant != tile and receives.get(source) != participant:
            raise RuntimeError("ReduceScatter places a receive in the wrong source slot")
        if participant == tile:
            local = views.get(source)
            if local is None:
                raise RuntimeError("ReduceScatter omits its local source contribution")
            if offset is None:
                offset = local[1][1]
            if local[1:] != ((tile, offset, 0), (1, size, 1)):
                raise RuntimeError("ReduceScatter has the wrong local source window")
        contributions[participant] = source
    if sorted(contributions) != list(range(PHYSICAL_TILE_COUNT)):
        raise RuntimeError("ReduceScatter does not consume every source exactly once")
    if offset is None or offset < 0 or offset + size > extent:
        raise RuntimeError("ReduceScatter source window is outside the full domain")
    return offset, size

def verify_personalized_exchange(
    package: pathlib.Path, dump: pathlib.Path, extent: int, *, reduce_scatter: bool = False
) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    for entry in manifest["entries"]:
        kinds = {argument["kind"] for argument in entry["arguments"]}
        if "transport_status" not in kinds or "shared_workspace" in kinds:
            raise RuntimeError("AllToAll requires Direct DTE without shared DDR")
    rows: dict[int, tuple[int, int]] = {}
    instructions: dict[int, str] = {}
    for tile in range(PHYSICAL_TILE_COUNT):
        ir = (dump / "instruction" / f"tile_{tile:05d}.mlir").read_text()
        instructions[tile] = ir
        output_shape = f"1x{extent}x1" if reduce_scatter else f"{extent}x16x1"
        returned = re.findall(rf"return (%[\w]+) : memref<{output_shape}xf16,", ir)
        if len(returned) != 1:
            raise RuntimeError(f"AllToAll Tile {tile} lacks a unique output")
        slice_pattern = (
            r"\[0, (\d+), 0\] \[1, (\d+), 1\] \[1, 1, 1\]"
            if reduce_scatter else
            r"\[(\d+), 0, 0\] \[(\d+), 16, 1\] \[1, 1, 1\]"
        )
        views = re.findall(
            rf"memref\.subview {re.escape(returned[0])}"
            + slice_pattern, ir
        )
        if len(views) != 1:
            raise RuntimeError(f"AllToAll Tile {tile} lacks one output shard")
        rows[tile] = tuple(map(int, views[0]))
    end = 0
    for offset, size in sorted(rows.values()):
        if offset != end or size <= 0:
            raise RuntimeError("AllToAll output has a gap or overlapping shard")
        end += size
    if end != extent:
        raise RuntimeError("AllToAll output does not cover every row")
    sends: dict[tuple[int, int, str], int] = {}
    receives: dict[tuple[int, int, str], int] = {}
    for tile, ir in instructions.items():
        if reduce_scatter:
            tile_ir = (dump / "tile-dataflow" / f"tile_{tile:05d}.mlir").read_text()
            verify_reduce_scatter_contributions(tile_ir, tile, extent, rows[tile])
        tokens = []
        for operation in ("send", "recv"):
            issues = re.findall(
                rf"(%[\w]+) = wafer\.instr\.dte_{operation} .*?"
                r"bytes = (\d+) : i64, message = (#wafer\.dte_message<[^>]+>), "
                r"peer = (\d+) : i64", ir
            )
            peers = [int(peer) for _, _, _, peer in issues]
            if sorted(peers) != [peer for peer in rows if peer != tile]:
                raise RuntimeError(f"AllToAll Tile {tile} lacks exact personalized {operation} peers")
            for token, size_text, message, peer_text in issues:
                peer, size = int(peer_text), int(size_text)
                source, destination = (tile, peer) if operation == "send" else (peer, tile)
                if size != rows[destination][1] * 2:
                    raise RuntimeError("AllToAll transfers a whole carrier or wrong piece")
                records = sends if operation == "send" else receives
                records[(source, destination, message)] = size
                tokens.append(token)
        if sorted(re.findall(r"wafer\.instr\.dte_wait (%[\w]+) :", ir)) != sorted(tokens):
            raise RuntimeError(f"AllToAll Tile {tile} lacks exact token completion")
    if sends != receives:
        raise RuntimeError("AllToAll send/receive messages do not match")
    kind = "reduce_scatter" if reduce_scatter else "alltoall"
    print(
        f"qualified_{kind}: extent={extent} tiles=16 sends=240 receives=240 "
        "waits=480 exact_output_coverage=true personalized_payload=true"
    )


@dataclasses.dataclass(frozen=True)
class _PeerIssue:
    position: int
    kind: str
    token: str
    buffer: str
    size: int
    communication: int
    round: int
    slice: int
    peer: int


def _read_peer_issues(ir: str, dialect: str) -> list[_PeerIssue]:
    pattern = (
        rf"(%\w+) = wafer\.{dialect}_(send|recv) (%\w+) \{{[^\n]*?"
        r"bytes = (\d+) : i64, message = #wafer\.dte_message<"
        r"communication = (\d+), round = (\d+), slice = (\d+)>, "
        r"peer = (\d+) : i64} : memref<[\dx]+f16,"
    )
    return [
        _PeerIssue(match.start(), match[2], match[1], match[3],
                   *map(int, match.groups()[3:]))
        for match in re.finditer(pattern, ir)
    ]


def _verify_all_reduce_ir(
    tile_irs: dict[int, str], instruction_irs: dict[int, str], extent: int
) -> None:
    shards: dict[int, tuple[int, int]] = {}
    phases: dict[tuple[int, int, str], list[_PeerIssue]] = {}
    seeds: dict[int, str] = {}
    views_by_tile = {}
    copies_by_tile = {}
    final_buffers: dict[int, str] = {}
    for tile, ir in tile_irs.items():
        reductions = list(re.finditer(
            r"(%\w+) = wafer\.tile\.reduce <sum> .*?"
            r"\{dimensions = array<i64: 0>} : \(memref<16x(\d+)x1xf16,", ir
        ))
        if len(reductions) != 1:
            raise RuntimeError("AllReduce requires one full contribution sum per Tile")
        reduction, = reductions
        views = {
            result: (source, tuple(map(int, offsets.split(", "))),
                     tuple(map(int, sizes.split(", "))))
            for result, source, offsets, sizes in re.findall(
                r"(%\w+) = memref\.subview (%\w+)\[([\d, ]+)\] "
                r"\[([\d, ]+)\] \[[1, ]+\]", ir
            )
        }
        shards[tile] = verify_reduce_scatter_contributions(ir, tile, extent, None)
        issues = _read_peer_issues(ir, "tile.peer")
        for phase in (0, 1):
            for kind in ("send", "recv"):
                selected = [issue for issue in issues if issue.kind == kind
                            and (issue.position > reduction.start()) == bool(phase)]
                if len(selected) != 15:
                    raise RuntimeError("AllReduce lacks both complete exchanges")
                phases[(tile, phase, kind)] = sorted(selected, key=lambda issue: issue.round)
        gather_sends = phases[(tile, 1, "send")]
        seed = gather_sends[0].buffer
        layouts = {match[1]: (match[2], match.start()) for match in re.finditer(
            r"(%\w+) = wafer\.tile\.materialize_layout (%\w+) :", ir
        )}
        if seed not in layouts:
            raise RuntimeError("AllReduce lacks the published result layout")
        destination, copied_at = layouts[seed]
        # The local sum must write its destination directly, before the layout
        # snapshot. This also witnesses elimination of the temporary writeback.
        publications = re.findall(
            rf"wafer\.tile\.elementwise_into <add> (%\w+), (%\w+) "
            rf"into {re.escape(destination)} :",
            ir[reduction.end():copied_at],
        )
        if (len(publications) != 1 or publications[0] not in (
                (destination, reduction[1]), (reduction[1], destination))):
            raise RuntimeError("AllReduce broadcasts a value other than its local sum")
        if re.search(rf"wafer\.tile\.copy_into .* into {re.escape(destination)} :",
                     ir[reduction.end():copied_at]):
            raise RuntimeError("AllReduce retains a redundant local sum writeback")
        seeds[tile] = seed
        final_inputs = re.findall(
            rf"wafer\.tile\.elementwise <add> %\w+, (%\w+) [^\n]*"
            rf": \(memref<1x{extent}x1xf16,[^\n]*?>, "
            rf"memref<{extent}x1xf16,", ir
        )
        if len(final_inputs) != 1:
            raise RuntimeError("AllReduce does not consume the complete gathered result")
        final_input = final_inputs[0]
        if final_input in views:
            root, offsets, sizes = views[final_input]
            if (offsets, sizes) != ((0, 0), (extent, 1)):
                raise RuntimeError("AllReduce consumes a partial gathered result")
            final_input = root
        final_buffers[tile] = final_input
        views_by_tile[tile] = views
        copies_by_tile[tile] = re.findall(r"memref\.copy (%\w+), (%\w+) :", ir)

        instr = instruction_irs[tile]
        local_updates = re.findall(
            r"wafer\.instr\.elementwise <add> (%\w+), (%\w+) into (%\w+) : "
            rf"memref<{shards[tile][1]}x1xf16, #wafer\.memory<spm, cx>>",
            instr,
        )
        if len(local_updates) != 1 or local_updates[0][2] not in local_updates[0][:2]:
            raise RuntimeError("AllReduce Instr retains an out-of-place local sum")
        final_issues = _read_peer_issues(instr, "instr.dte")
        signature = lambda issue: (issue.kind, issue.size, issue.communication,
                                   issue.round, issue.slice, issue.peer)
        if sorted(map(signature, issues)) != sorted(map(signature, final_issues)):
            raise RuntimeError("AllReduce Tile/Instr messages differ")
        waits = re.findall(r"wafer\.instr\.dte_wait (%\w+) :", instr)
        if sorted(waits) != sorted(issue.token for issue in final_issues):
            raise RuntimeError("AllReduce lacks exact token completion")
        returned = re.findall(rf"return (%\w+) : memref<16x{extent}x1xf16,", instr)
        if len(returned) != 1 or len(re.findall(
            rf"memref\.subview {re.escape(returned[0])}\[{tile}, 0, 0\] "
            rf"\[1, {extent}, 1\] \[1, 1, 1\]", instr
        )) != 1:
            raise RuntimeError("AllReduce lacks exact replicated output ownership")
    end = 0
    for offset, size in sorted(shards.values()):
        if offset != end or size <= 0:
            raise RuntimeError("AllReduce contribution shards overlap or omit values")
        end += size
    if end != extent:
        raise RuntimeError("AllReduce contribution shards omit the tail")
    origin = {phases[(tile, 1, "send")][0].communication: tile for tile in tile_irs}
    if len(origin) != PHYSICAL_TILE_COUNT:
        raise RuntimeError("AllReduce has duplicate gathered contributions")
    messages: dict[str, dict[tuple[int, ...], int]] = {"send": {}, "recv": {}}
    for (tile, phase, kind), issues in phases.items():
        if phase == 0:
            if sorted(issue.peer for issue in issues) != [peer for peer in tile_irs if peer != tile]:
                raise RuntimeError("AllReduce misses a direct contribution peer")
        elif [issue.round for issue in issues] != list(range(15)):
            raise RuntimeError("AllReduce lacks the 15-round result gather")
        for issue in issues:
            source, destination = (tile, issue.peer) if kind == "send" else (issue.peer, tile)
            owner = destination if phase == 0 else origin.get(issue.communication)
            if owner not in shards or issue.size != shards[owner][1] * 2:
                raise RuntimeError("AllReduce transfers the wrong contribution payload")
            key = (phase, source, destination, issue.communication, issue.round, issue.slice)
            if key in messages[kind]:
                raise RuntimeError("AllReduce duplicates a message")
            messages[kind][key] = issue.size
            if phase == 1 and kind == "send" and issue.round:
                previous = phases[(tile, 1, "recv")][issue.round - 1]
                if (issue.buffer, issue.communication) != (previous.buffer, previous.communication):
                    raise RuntimeError("AllReduce forwards the wrong gathered contribution")
    if messages["send"] != messages["recv"]:
        raise RuntimeError("AllReduce sends and receives do not match")
    for tile in tile_irs:
        labels = {issue.buffer: origin[issue.communication]
                  for issue in phases[(tile, 1, "recv")]}
        labels[seeds[tile]] = tile
        views = views_by_tile[tile]
        writes = {}
        for source, destination in copies_by_tile[tile]:
            if destination in views:
                key = views[destination]
                if key in writes:
                    raise RuntimeError("AllReduce overwrites a result fragment")
                writes[key] = source

        def origin_of(buffer: str, visited: frozenset[str] = frozenset()) -> int:
            if buffer in labels:
                return labels[buffer]
            if buffer in visited or buffer not in views or views[buffer] not in writes:
                raise RuntimeError("AllReduce result fragment has no contribution origin")
            return origin_of(writes[views[buffer]], visited | {buffer})

        copied = set()
        for (root, offsets, sizes), source in writes.items():
            if root != final_buffers[tile]:
                continue
            owner = origin_of(source)
            offset, size = shards[owner]
            if owner in copied or (offsets, sizes) != ((offset, 0), (size, 1)):
                raise RuntimeError("AllReduce places a gathered fragment in the wrong output slot")
            copied.add(owner)
        if copied != set(tile_irs):
            raise RuntimeError("AllReduce does not replicate every contribution to every Tile")


def verify_all_reduce(package: pathlib.Path, dump: pathlib.Path, extent: int) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    for entry in manifest["entries"]:
        kinds = {argument["kind"] for argument in entry["arguments"]}
        if "transport_status" not in kinds or "shared_workspace" in kinds:
            raise RuntimeError("AllReduce qualification requires DTE in both exchanges")
    tiles = {tile: (dump / "tile-dataflow" / f"tile_{tile:05d}.mlir").read_text()
             for tile in range(PHYSICAL_TILE_COUNT)}
    instructions = {tile: (dump / "instruction" / f"tile_{tile:05d}.mlir").read_text()
                    for tile in tiles}
    _verify_all_reduce_ir(tiles, instructions, extent)
    first_receive = next(issue for issue in _read_peer_issues(tiles[0], "tile.peer")
                         if issue.kind == "recv")
    reduction = re.search(r"(%\w+) = wafer\.tile\.reduce <sum>", tiles[0])
    assert reduction is not None  # Established by the successful verifier above.
    sum_add = next(match for match in re.finditer(
        r"wafer\.tile\.elementwise_into <add> (%\w+), (%\w+) into (%\w+)", tiles[0]
    ) if reduction[1] in match.groups()[:2])
    wrong_sum = (
        tiles[0][:sum_add.start()]
        + sum_add[0].replace(reduction[1], sum_add[3])
        + tiles[0][sum_add.end():]
    )
    received_layout = re.search(
        rf"(%\w+) = wafer\.tile\.materialize_layout {re.escape(first_receive.buffer)} :",
        tiles[0],
    )
    contribution = received_layout[1] if received_layout else first_receive.buffer
    faults = (
        ("wrong-published-sum", "tile", wrong_sum),
        ("missing-peer", "tile", re.sub(
            r"^.*wafer\.tile\.peer_recv .*\n", "", tiles[0], count=1, flags=re.MULTILINE)),
        ("missing-contribution", "tile", re.sub(
            rf"^.*(?:memref\.copy {re.escape(contribution)},|"
            rf"wafer\.tile\.copy_into {re.escape(contribution)} into).*\n", "",
            tiles[0], count=1, flags=re.MULTILINE)),
        ("wrong-payload", "instr", re.sub(
            r"(wafer\.instr\.dte_recv[^\n]*?bytes = )(\d+)",
            lambda match: match[1] + str(int(match[2]) + 2), instructions[0], count=1)),
        ("missing-wait", "instr", re.sub(
            r"^.*wafer\.instr\.dte_wait .*\n", "", instructions[0],
            count=1, flags=re.MULTILINE)),
    )
    for fault, stage, corrupted in faults:
        original = tiles if stage == "tile" else instructions
        if corrupted == original[0]:
            raise RuntimeError(f"AllReduce fault {fault} was not injected")
        changed = {**original, 0: corrupted}
        try:
            _verify_all_reduce_ir(changed if stage == "tile" else tiles,
                                  changed if stage == "instr" else instructions, extent)
        except RuntimeError:
            continue
        raise RuntimeError(f"AllReduce accepted {fault}")
    print(f"qualified_all_reduce: extent={extent} tiles=16 sends=480 receives=480 "
          "waits=960 exact_contributions=true replicated_results=true rejected_faults=5")


def verify_row_sharded_gemm(
    dump: pathlib.Path, dimensions: tuple[int, int, int], dtype: torch.dtype
) -> None:
    """Check the actual baseline GEMM and returned output slices on every Tile."""
    m, k, n = dimensions
    ir_dtype = next(name for name, value in common.MANIFEST_DTYPES.items()
                    if value == dtype)
    slices: list[tuple[int, int]] = []
    for tile in range(PHYSICAL_TILE_COUNT):
        ir = (dump / "instruction" / f"tile_{tile:05d}.mlir").read_text()
        returned = re.findall(
            rf"return (%[\w]+) : memref<1x{m}x{n}x{ir_dtype}, "
            r"#wafer\.memory<ddr, tensor>>", ir,
        )
        if len(returned) != 1:
            raise RuntimeError(f"GEMM Tile {tile} has no unique complete output")
        views = re.findall(
            rf"memref\.subview {re.escape(returned[0])}"
            rf"\[0, (\d+), 0\] \[1, (\d+), {n}\] \[1, 1, 1\]", ir,
        )
        gemms = re.findall(r"wafer\.instr\.gemm .*?\{([^\n]+)\}", ir)
        if len(views) != 1 or len(gemms) != 1:
            raise RuntimeError(f"GEMM Tile {tile} must own one output slice and GEMM")
        offset, extent = map(int, views[0])
        fields = dict(re.findall(r"\b(batch_count|m|k|n) = (\d+) : i64", gemms[0]))
        if fields != {"batch_count": "1", "m": str(extent),
                      "k": str(k), "n": str(n)}:
            raise RuntimeError(f"GEMM Tile {tile} does not cover its exact M/K/N")
        if extent <= 0:
            raise RuntimeError(f"GEMM Tile {tile} has an empty output slice")
        slices.append((offset, extent))
    end = 0
    for offset, extent in sorted(slices):
        if offset != end:
            raise RuntimeError("GEMM output slices contain a gap or overlap")
        end += extent
    if end != m:
        raise RuntimeError("GEMM output slices do not cover every row")
    print(
        f"production_gemm: tiles={PHYSICAL_TILE_COUNT} batch=1 "
        f"m={m} k={k} n={n} exact_output_coverage=true "
        f"local_m_sizes={sorted({extent for _, extent in slices})}"
    )


def verify_shared_gemm_input(
    dump: pathlib.Path, dimensions: tuple[int, int, int], dtype: torch.dtype
) -> None:
    """Qualify the existing row-sharded GEMM with its complete RHS shared."""
    _, k, n = dimensions
    expected_bytes = k * n * common.element_bytes(dtype)
    sends, receives = {}, {}
    rhs_loads = []
    donor_buffers = set()
    for tile in range(PHYSICAL_TILE_COUNT):
        ir = (dump / "tile-dataflow" / f"tile_{tile:05d}.mlir").read_text()
        rhs_loads.extend((tile, match[1]) for match in re.finditer(
            rf"wafer\.tile\.load %\w+ into (%\w+) : [^\n]* into "
            rf"memref<1x{k}x{n}xf16, #wafer\.memory<spm, tensor>>", ir
        ))
        for issue in _read_peer_issues(ir, "tile.peer"):
            source, destination = (tile, issue.peer) if issue.kind == "send" else (issue.peer, tile)
            if source != 0 or not 0 < destination < PHYSICAL_TILE_COUNT or issue.size != expected_bytes:
                raise RuntimeError("shared GEMM RHS has an incorrect endpoint or payload")
            key = (issue.communication, issue.round, issue.slice, source, destination)
            entries = sends if issue.kind == "send" else receives
            if issue.kind == "send":
                donor_buffers.add(issue.buffer)
            if key in entries:
                raise RuntimeError("shared GEMM RHS repeats a message")
            entries[key] = issue.size
    if len(rhs_loads) != 1 or rhs_loads[0][0] != 0 or sends != receives or len(sends) != PHYSICAL_TILE_COUNT - 1:
        raise RuntimeError("shared GEMM RHS lacks one donor load and one transfer per receiver")
    if donor_buffers != {rhs_loads[0][1]} or {key[-1] for key in sends} != set(range(1, PHYSICAL_TILE_COUNT)):
        raise RuntimeError("shared GEMM RHS does not forward the loaded buffer to every receiver")
    print(f"qualified_shared_input: rhs_loads={len(rhs_loads)} sends={len(sends)} receives={len(receives)} "
          f"payload_bytes={expected_bytes} exact_endpoints=true")


def prepare_case_step(
    args: argparse.Namespace,
    case: board_cases.PyTorchBoardCase,
    *,
    step_index: int,
    step_dir: pathlib.Path,
    source: pathlib.Path,
    package: pathlib.Path,
    dump_compiler_ir: pathlib.Path | None,
    prepared_step: pathlib.Path | None = None,
) -> tuple[
    tuple[torch.Tensor, ...],
    list[str],
    dict[pathlib.Path, torch.Tensor],
    set[int],
    dict[int, pathlib.Path],
]:
    step_number = step_index + 1
    expected_outputs = None
    if args.target_model and prepared_step is not None:
        raise RuntimeError("target-model validation requires this run's source compilation")
    export_start_ns = time.monotonic_ns()
    case.export_program(source)
    print(
        "pytorch-board-timing stage=source-export "
        f"step={step_number} "
        f"wall_ms={(time.monotonic_ns() - export_start_ns) // 1_000_000}"
    )
    if prepared_step is not None:
        verify_prepared_source(prepared_step / "source-program", source)
        if not (package / "manifest.json").is_file():
            raise RuntimeError("prepared package manifest is missing")
        print(f"pytorch-board-prepared: step={step_number} source_equal=true compile=false")
    else:
        compile_command = [
            str(args.wafer_compile),
            "--input-program-dir",
            str(source),
            "--output-dir",
            str(package.parent if args.profile else package),
            f"--num-partitions={case.num_partitions}",
            f"--optimization-policy={args.optimization_policy}",
        ]
        if args.qualify_communication is not None:
            if args.optimization_policy != "none":
                raise RuntimeError("explicit communication qualification cannot run search")
            selected = args.qualify_communication if args.qualify_communication in ("shared-input", "pipelined-loads") else "peer"
            compile_command.append(f"--test-communication-candidate={selected}")
        if args.compile_timing:
            compile_command.append("--compile-timing")
        if args.profile:
            compile_command.append("--profile")
        if dump_compiler_ir is not None:
            compile_command.extend(
                ["--dump-compiler-ir", str(dump_compiler_ir)]
            )
        for option in ("search_width", "search_trials"):
            value = getattr(args, option)
            if value is not None:
                compile_command.extend(["--" + option.replace("_", "-"), str(value)])
        if args.target_model:
            expected_outputs = case.materialize_expected_outputs()
            model_dir = step_dir / "model"
            model_dir.mkdir()
            compile_command.append("--target-model")
            for role, tensors in (("input", case.inputs), ("expected", expected_outputs)):
                for index, tensor in enumerate(tensors):
                    path = model_dir / f"{role}_{index}.npy"
                    board_cases.capture.save_torch_tensor_npy(path, tensor)
                    compile_command.extend([f"--model-{role}", f"{index}={path}"])
            compile_command.extend(common.model_comparison_arguments(case.comparison_policy))
            for option in ("target_model_max_scalar_evaluations",
                           "target_model_max_fused_multiply_adds",
                           "target_model_max_movement_bytes",
                           "target_model_max_movement_segments"):
                compile_command.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
        compile_result = run(
            compile_command,
            timeout_seconds=args.compile_timeout_seconds,
        )
        if args.compile_timing:
            print(compile_result.stderr, end="", file=sys.stderr)
        if args.target_model:
            if "target model outputs matched; tiles=16" not in compile_result.stdout:
                raise RuntimeError("functional target model did not compare every output")
            print(compile_result.stdout, end="")
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
        if case.widened_convolution:
            verify_widened_convolution(dump_compiler_ir, case.dtype)
        if case.ordered_convolution:
            verify_ordered_convolution(dump_compiler_ir)
        for tile_file, instruction_file in zip(tile_files, instruction_files, strict=True):
            verify_pre_instruction_boundary(
                tile_file.read_text(encoding="utf-8"),
                instruction_file.read_text(encoding="utf-8"),
            )
    if args.qualify_communication == "ring-allgather":
        if case.allgather_payload_elements is None:
            raise RuntimeError("Ring AllGather qualification requires an AllGather source")
        if dump_compiler_ir is None:
            raise RuntimeError("AllGather qualification requires current compiler IR")
        verify_ring_allgather(
            package,
            dump_compiler_ir,
            case.allgather_payload_elements * common.element_bytes(case.dtype),
        )
    if case.gemm_dimensions is not None and args.optimization_policy == "none":
        if dump_compiler_ir is None:
            raise RuntimeError("GEMM qualification requires current compiler IR")
        verify_row_sharded_gemm(dump_compiler_ir, case.gemm_dimensions, case.dtype)
    if args.qualify_communication == "direct-alltoall":
        if case.alltoall_extent is None:
            raise RuntimeError("direct AllToAll qualification requires an AllToAll source")
        if dump_compiler_ir is None:
            raise RuntimeError("AllToAll qualification requires current compiler IR")
        verify_personalized_exchange(package, dump_compiler_ir, case.alltoall_extent)
    if args.qualify_communication == "direct-reduce-scatter":
        if case.reduce_scatter_extent is None:
            raise RuntimeError("ReduceScatter qualification requires a reduction source")
        if dump_compiler_ir is None:
            raise RuntimeError("ReduceScatter qualification requires current compiler IR")
        verify_personalized_exchange(
            package, dump_compiler_ir, case.reduce_scatter_extent, reduce_scatter=True
        )
    if args.qualify_communication == "all-reduce":
        if case.all_reduce_extent is None or dump_compiler_ir is None:
            raise RuntimeError("AllReduce qualification requires its source and current IR")
        verify_all_reduce(package, dump_compiler_ir, case.all_reduce_extent)
    if args.qualify_communication == "shared-input":
        if case.gemm_dimensions is None or dump_compiler_ir is None:
            raise RuntimeError("shared-input qualification requires GEMM source and current IR")
        verify_shared_gemm_input(dump_compiler_ir, case.gemm_dimensions, case.dtype)
    if args.qualify_communication == "pipelined-loads":
        if dump_compiler_ir is None:
            raise RuntimeError("load pipeline qualification requires current IR")
        rotating_loads = 0
        for path in (dump_compiler_ir / "tile-dataflow").glob("tile_*.mlir"):
            ir = path.read_text()
            slots = re.findall(r"(%\w+) = arith\.select [^\n]* : memref<[^\n]*#wafer\.memory<spm,", ir)
            rotating_loads += sum(len(re.findall(rf"wafer\.tile\.load %\w+ into {re.escape(slot)} :", ir)) for slot in slots)
        if not rotating_loads:
            raise RuntimeError("load pipeline qualification did not materialize rotating loads")
        print(f"qualified_pipelined_loads: actual_rotating_load_sites={rotating_loads}")
    oracle_start_ns = time.monotonic_ns()
    if expected_outputs is None:
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
        step_dir, source, package, case, expected_outputs
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
    if args.compile_timeout_seconds < 1:
        raise RuntimeError("compile timeout must be positive")
    for limit in (args.search_width, args.search_trials):
        if limit is not None and (limit < 1 or args.optimization_policy != "search"):
            raise RuntimeError("search limits require search policy and positive values")
    if args.prepared_work_dir is not None:
        validate_prepared_directories(args.prepared_work_dir, args.work_dir)
    if args.profile_trace_event_limit is not None and (
        not args.profile or not 0 < args.profile_trace_event_limit <= (1 << 32) - 1
    ):
        raise RuntimeError("profile trace event limit requires --profile and a positive uint32 limit")
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
        prepared_step = None
        if args.prepared_work_dir is not None:
            prepared_step = args.prepared_work_dir.resolve()
            if is_chain:
                prepared_step = prepared_step / f"step_{step_index + 1:02d}"
            package = prepared_step / "package"
        if args.profile:
            package = package / "package"
        dump_compiler_ir = args.dump_compiler_ir
        if (
            dump_compiler_ir is None
            and (
                current_case.allgather_payload_elements is not None
                or current_case.gemm_dimensions is not None
                or current_case.alltoall_extent is not None
                or current_case.reduce_scatter_extent is not None
                or current_case.all_reduce_extent is not None
                or current_case.widened_convolution
                or current_case.ordered_convolution
                or current_case.prefill_extent is not None
            )
        ):
            dump_compiler_ir = (prepared_step or step_dir) / "compiler-ir"
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
            prepared_step=prepared_step,
        )

        command = base_runtime_command(args.wafer_run, package)
        if args.profile_trace_event_limit is not None:
            command.extend(["--profile-trace-event-limit", str(args.profile_trace_event_limit)])
        # Direct DTE is selected by the common card search, so a
        # board-ready no-card runner must advertise the same transport
        # capabilities regardless of which candidate wins.  This remains
        # side-effect-free validation; it does not claim hardware execution.
        no_card_command = command + (
            [
                "--no-card",
                "--direct-dte-status-abi",
                DIRECT_DTE_STATUS_ABI,
                "--supports-host-watchdog",
            ]
        )
        result = run(no_card_command)
        verify_no_card(result.stdout)
        if args.no_card:
            print(
                f"pytorch_board_no_card: case={current_case.name} "
                f"dtype={args.dtype} seed={args.seed} "
                f"step={step_index + 1} "
                f"optimization_policy={args.optimization_policy} "
                "source_export=true torch_eager_reference=true "
                "runtime_payload=true numeric_execution=false "
                "continuation_reference_reused=true"
            )
            print(result.stdout, end="")
            continuation_outputs = expected_outputs
        else:
            if args.device_timing:
                command.append("--device-timing")
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
                (step_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                # Profile includes host report generation after device collection.
                # Each launch still has --completion-timeout-ms; that device
                # deadline must not also bound host report processing.
                result = run(
                    command,
                    timeout_seconds=(
                        None
                        if args.profile else
                        args.completion_timeout_ms / 1000 + PROCESS_TIMEOUT_MARGIN_SECONDS
                    ),
                )
                (step_dir / f"board-{iteration + 1:02d}.log").write_text(
                    result.stdout + result.stderr
                )
                print(result.stdout, end="")
                verify_board(
                    result.stdout,
                    current_case,
                    output_ids,
                    captures,
                )
                actual_outputs = read_single_card_continuation_outputs(
                    result_capture_paths, expected_outputs
                )
                if current_case.validate_actual_outputs is not None:
                    current_case.validate_actual_outputs(actual_outputs)
                audit = {
                    "case": current_case.name,
                    "step": step_index + 1,
                    "policy": args.optimization_policy,
                    "search_width": args.search_width,
                    "search_trials": args.search_trials,
                    "manifest_sha256": file_sha256(package / "manifest.json"),
                    "comparison_passed": True,
                    "comparison": dataclasses.asdict(current_case.comparison_policy),
                    "inputs": {path.name: file_sha256(path) for path in sorted(
                        (step_dir / "raw").glob("*user_input*")
                    )},
                    "outputs": [
                        {
                            **summarize_output_error(actual, expected),
                            **(dataclasses.asdict(common.compute_tensor_similarity(
                                actual, expected, policy=current_case.comparison_policy,
                            )) if current_case.comparison_policy.min_cosine is not None
                               and actual.dtype.is_floating_point else {}),
                        }
                        for actual, expected in zip(actual_outputs, expected_outputs, strict=True)
                    ],
                    "timing": [line for line in result.stdout.splitlines()
                               if line.startswith(("board_timing:", "profile_run:"))],
                }
                (step_dir / f"numeric-audit-{iteration + 1:02d}.json").write_text(
                    json.dumps(audit, indent=2) + "\n"
                )
                print(
                    f"pytorch_board_iteration: case={current_case.name} "
                    f"dtype={args.dtype} seed={args.seed} "
                    f"step={step_index + 1} "
                    f"optimization_policy={args.optimization_policy} "
                    f"iteration={iteration + 1}/{args.repeat} "
                    f"wall_ms="
                    f"{(time.monotonic_ns() - iteration_start_ns) // 1_000_000} "
                    "comparison_passed=true"
                )
            continuation_outputs = actual_outputs

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
