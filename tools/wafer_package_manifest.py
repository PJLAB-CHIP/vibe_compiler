#!/usr/bin/env python3
"""Validate and roundtrip Wafer runtime package manifests."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


ALLOWED_COMPLETION_SOURCES = {
    "hpgr_stream_event",
    "hpgr_command_slot",
    "kcore_local_drain",
    "legacy_tsm_run_sync",
}

KNOWN_STUB_FENCES = {
    "TsmDeviceSynchronize",
    "TsmLaunch",
    "TsmLaunchPg",
    "KmdDoorbellOnly",
}

SUPPORTED_INSTRUCTION_OPS = {
    "wafer.instr.rdma",
    "wafer.instr.wdma",
    "wafer.instr.gemm",
    "wafer.instr.elementwise",
    "wafer.instr.reduce",
}

SUPPORTED_ELEMENTWISE_KINDS = {
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "neg",
    "recip",
    "sqrt",
    "rsqrt",
    "exp",
    "tanh",
}

SUPPORTED_REDUCE_KINDS = {
    "sum",
    "max",
    "min",
}


def canonical_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def fail(message: str) -> None:
    raise ValueError(message)


def require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{name} must be an object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{name} must be a list")
    return value


def require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{name} must be a non-empty string")
    return value


def require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        fail(f"{name} must be a positive integer")
    return value


def require_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        fail(f"{name} must be a non-negative integer")
    return value


def require_number(value: Any, name: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail(f"{name} must be a number")
    return value


DTYPE_BYTES = {
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
}


def compact_tensor_bytes(shape: list[int], dtype: str, name: str) -> int:
    if dtype not in DTYPE_BYTES:
        fail(f"{name}.dtype is not supported for byte validation")
    elements = 1
    for dim in shape:
        elements *= dim
    return elements * DTYPE_BYTES[dtype]


def validate_tensor(tensor: Any, name: str) -> tuple[str, list[int]]:
    item = require_dict(tensor, name)
    tensor_name = require_non_empty_string(item.get("name"), f"{name}.name")
    shape = require_list(item.get("shape"), f"{name}.shape")
    if not shape:
        fail(f"{name}.shape must be non-empty")
    checked_shape = []
    for index, dim in enumerate(shape):
        checked_shape.append(require_positive_int(dim, f"{name}.shape[{index}]"))
    require_non_empty_string(item.get("dtype"), f"{name}.dtype")
    if item.get("layout") != "tensor":
        fail(f"{name}.layout must be tensor")
    return tensor_name, checked_shape


def validate_tensor_storage_demand(item: Any, name: str) -> tuple[str, int]:
    demand = require_dict(item, name)
    demand_name = require_non_empty_string(demand.get("name"), f"{name}.name")
    shape = require_list(demand.get("shape"), f"{name}.shape")
    if not shape:
        fail(f"{name}.shape must be non-empty")
    checked_shape = [
        require_positive_int(dim, f"{name}.shape[{index}]")
        for index, dim in enumerate(shape)
    ]
    dtype = require_non_empty_string(demand.get("dtype"), f"{name}.dtype")
    if demand.get("layout") != "tensor":
        fail(f"{name}.layout must be tensor")
    bytes_value = require_positive_int(demand.get("bytes"), f"{name}.bytes")
    expected_bytes = compact_tensor_bytes(checked_shape, dtype, name)
    if bytes_value != expected_bytes:
        fail(f"{name}.bytes must match compact tensor storage size")
    require_positive_int(demand.get("alignment"), f"{name}.alignment")
    return demand_name, bytes_value


def validate_workspace_buffers(value: Any) -> int:
    total = 0
    seen_names = set()
    for index, workspace in enumerate(require_list(value, "workspace_buffers")):
        name = f"workspace_buffers[{index}]"
        item = require_dict(workspace, name)
        buffer_name, bytes_value = validate_tensor_storage_demand(item, name)
        if buffer_name in seen_names:
            fail("workspace buffer names must be unique")
        seen_names.add(buffer_name)
        require_non_empty_string(item.get("producer"), f"{name}.producer")
        require_non_empty_string(item.get("last_consumer"), f"{name}.last_consumer")
        total += bytes_value
    return total


def validate_resident_constants(
    value: Any, input_names: set[str], output_names: set[str]
) -> int:
    total = 0
    seen_names = set()
    for index, constant in enumerate(require_list(value, "resident_constants")):
        name = f"resident_constants[{index}]"
        item = require_dict(constant, name)
        constant_name, bytes_value = validate_tensor_storage_demand(item, name)
        if constant_name in seen_names:
            fail("resident constant names must be unique")
        seen_names.add(constant_name)

        source = require_non_empty_string(item.get("source"), f"{name}.source")
        if source == "launch_input":
            if constant_name not in input_names:
                fail(f"{name}.name must refer to a launch input")
        elif source != "embedded_constant":
            fail(f"{name}.source must be launch_input or embedded_constant")
        if constant_name in output_names:
            fail(f"{name}.name must not refer to a launch output")
        total += bytes_value
    return total


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        fail("schema_version must be 1")
    require_non_empty_string(manifest.get("package_name"), "package_name")

    runtime = require_dict(manifest.get("runtime"), "runtime")
    completion_source = require_non_empty_string(
        runtime.get("completion_source"), "runtime.completion_source"
    )
    if completion_source in KNOWN_STUB_FENCES:
        fail("completion source is a known stub fence")
    if completion_source not in ALLOWED_COMPLETION_SOURCES:
        fail("completion source is not in the allowed runtime fence set")

    signature = require_dict(manifest.get("launch_signature"), "launch_signature")
    input_tensors = [
        validate_tensor(item, f"launch_signature.inputs[{index}]")
        for index, item in enumerate(
            require_list(signature.get("inputs"), "launch_signature.inputs")
        )
    ]
    output_tensors = [
        validate_tensor(item, f"launch_signature.outputs[{index}]")
        for index, item in enumerate(
            require_list(signature.get("outputs"), "launch_signature.outputs")
        )
    ]
    input_names = {name for name, _ in input_tensors}
    output_names = {name for name, _ in output_tensors}

    resources = require_dict(manifest.get("resources"), "resources")
    spm_bytes = require_positive_int(resources.get("spm_bytes"), "resources.spm_bytes")
    if spm_bytes > 0x2F0000:
        fail("resources.spm_bytes exceeds usable SPM capacity")
    workspace_bytes = require_non_negative_int(
        resources.get("workspace_bytes"), "resources.workspace_bytes"
    )
    resident_constant_bytes = require_non_negative_int(
        resources.get("resident_constant_bytes"),
        "resources.resident_constant_bytes",
    )
    workspace_buffer_bytes = validate_workspace_buffers(
        manifest.get("workspace_buffers")
    )
    if workspace_bytes != workspace_buffer_bytes:
        fail("resources.workspace_bytes does not match workspace buffers")
    resident_bytes = validate_resident_constants(
        manifest.get("resident_constants"), input_names, output_names
    )
    if resident_constant_bytes != resident_bytes:
        fail("resources.resident_constant_bytes does not match resident constants")

    instructions = require_list(manifest.get("instructions"), "instructions")
    for index, op in enumerate(instructions):
        item = require_dict(op, f"instructions[{index}]")
        mnemonic = require_non_empty_string(item.get("op"), f"instructions[{index}].op")
        if mnemonic not in SUPPORTED_INSTRUCTION_OPS:
            fail(f"instructions[{index}].op is not supported by the package manifest")
        if item.get("wait_policy") != "issue_only":
            fail(f"instructions[{index}].wait_policy must be issue_only")
        if mnemonic in {"wafer.instr.rdma", "wafer.instr.wdma"}:
            require_positive_int(item.get("bytes"), f"instructions[{index}].bytes")
        elif mnemonic == "wafer.instr.gemm":
            require_positive_int(item.get("m"), f"instructions[{index}].m")
            require_positive_int(item.get("k"), f"instructions[{index}].k")
            require_positive_int(item.get("n"), f"instructions[{index}].n")
            if "batch_count" in item:
                require_positive_int(
                    item.get("batch_count"), f"instructions[{index}].batch_count"
                )
        elif mnemonic == "wafer.instr.elementwise":
            kind = require_non_empty_string(item.get("kind"), f"instructions[{index}].kind")
            if kind not in SUPPORTED_ELEMENTWISE_KINDS:
                fail(f"instructions[{index}].kind is not supported")
        elif mnemonic == "wafer.instr.reduce":
            kind = require_non_empty_string(item.get("kind"), f"instructions[{index}].kind")
            if kind not in SUPPORTED_REDUCE_KINDS:
                fail(f"instructions[{index}].kind is not supported")
            dimensions = require_list(
                item.get("dimensions"), f"instructions[{index}].dimensions"
            )
            if not dimensions:
                fail(f"instructions[{index}].dimensions must be non-empty")
            if len(dimensions) > 4:
                fail(f"instructions[{index}].dimensions supports at most four dimensions")
            for dim_index, dim in enumerate(dimensions):
                require_non_negative_int(
                    dim, f"instructions[{index}].dimensions[{dim_index}]"
                )
            require_number(item.get("init_value"), f"instructions[{index}].init_value")
        elif "kind" in item:
            fail(f"instructions[{index}].kind is only valid for elementwise or reduce ops")

    input_bytes = 0
    output_bytes = 0
    for index, binding in enumerate(
        require_list(manifest.get("ddr_bindings"), "ddr_bindings")
    ):
        item = require_dict(binding, f"ddr_bindings[{index}]")
        kind = require_non_empty_string(item.get("kind"), f"ddr_bindings[{index}].kind")
        name = require_non_empty_string(item.get("name"), f"ddr_bindings[{index}].name")
        bytes_value = require_positive_int(item.get("bytes"), f"ddr_bindings[{index}].bytes")
        require_positive_int(item.get("alignment"), f"ddr_bindings[{index}].alignment")
        if not isinstance(item.get("host_visible"), bool):
            fail(f"ddr_bindings[{index}].host_visible must be boolean")
        if not isinstance(item.get("read_only"), bool):
            fail(f"ddr_bindings[{index}].read_only must be boolean")

        if kind == "input":
            if name not in input_names:
                fail(f"ddr_bindings[{index}] input name is not in launch signature")
            if item["read_only"] is not True:
                fail(f"ddr_bindings[{index}] input must be read-only")
            input_bytes += bytes_value
        elif kind == "output":
            if name not in output_names:
                fail(f"ddr_bindings[{index}] output name is not in launch signature")
            if item["read_only"] is not False:
                fail(f"ddr_bindings[{index}] output must be writable")
            output_bytes += bytes_value
        else:
            fail(f"ddr_bindings[{index}].kind must be input or output")

    if resources.get("ddr_external_input_bytes") != input_bytes:
        fail("resources.ddr_external_input_bytes does not match DDR bindings")
    if resources.get("ddr_external_output_bytes") != output_bytes:
        fail("resources.ddr_external_output_bytes does not match DDR bindings")


def load_manifest(path: str) -> dict[str, Any]:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    return require_dict(json.loads(text), "manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--roundtrip")
    mode.add_argument("--validate")
    args = parser.parse_args()

    manifest = load_manifest(args.roundtrip or args.validate)
    validate_manifest(manifest)
    if args.roundtrip:
        sys.stdout.write(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
