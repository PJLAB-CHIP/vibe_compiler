#!/usr/bin/env python3
"""Validate and roundtrip Wafer runtime package metadata."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


ALLOWED_COMPLETION_SOURCES = {
    "runtime_stream_wait",
    "runtime_command_completion",
    "kcore_local_drain",
    "legacy_model_sync",
}

ALLOWED_MODULE_FORMATS = {
    "tx.bpm",
    "tx.graph",
    "tx.kcore",
}

ALLOWED_RUNTIME_MODES = {
    "tx",
    "legacy_tsm",
}

ALLOWED_ABI_VERSIONS = {
    "wafer-cabi-v0",
}

ALLOWED_ENTRYPOINT_EXECUTORS = {
    "legacy.tsm",
    "tx.cluster",
    "tx.graph",
    "tx.model",
    "tx.module",
}

ALLOWED_BPM_DESCRIPTOR_STATES = {
    "descriptor_only",
    "materialized",
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
    "wafer.instr.gather_scatter",
    "wafer.instr.fill",
    "wafer.instr.gemm",
    "wafer.instr.elementwise",
    "wafer.instr.reduce",
    "wafer.instr.convert",
    "wafer.instr.dte_send",
    "wafer.instr.dte_recv",
    "wafer.instr.dte_wait",
    "wafer.instr.local_fence",
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
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
}

SUPPORTED_REDUCE_KINDS = {
    "sum",
    "max",
    "min",
    "avg",
}


def canonical_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, indent=2, allow_nan=False) + "\n"


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


def reject_deprecated_keys(item: dict[str, Any], name: str, keys: set[str]) -> None:
    for key in sorted(keys):
        if key in item:
            fail(f"{name}.{key} is deprecated")


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
    if isinstance(value, float) and not math.isfinite(value):
        fail(f"{name} must be a finite number")
    return value


def validate_reduce_init_value(value: Any, name: str) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        require_number(value, name)
        return

    item = require_dict(value, name)
    kind = require_non_empty_string(item.get("kind"), f"{name}.kind")
    if kind != "non_finite":
        fail(f"{name}.kind is not supported")
    encoded = require_non_empty_string(item.get("value"), f"{name}.value")
    if encoded not in {"-inf", "inf", "nan"}:
        fail(f"{name}.value is not a supported non-finite value")
    dtype = require_non_empty_string(item.get("dtype"), f"{name}.dtype")
    if dtype not in {"f16", "bf16", "f32", "f64"}:
        fail(f"{name}.dtype is not supported for non-finite reduce init")
    if "bits" in item:
        require_non_negative_int(item.get("bits"), f"{name}.bits")


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


def validate_modules(value: Any) -> dict[str, dict[str, Any]]:
    modules = require_list(value, "modules")
    if not modules:
        fail("modules must be non-empty")
    by_name: dict[str, dict[str, Any]] = {}
    for index, module in enumerate(modules):
        name = f"modules[{index}]"
        item = require_dict(module, name)
        reject_deprecated_keys(item, name, {"kind"})
        module_name = require_non_empty_string(item.get("name"), f"{name}.name")
        if module_name in by_name:
            fail("module names must be unique")
        module_format = require_non_empty_string(item.get("format"), f"{name}.format")
        if module_format not in ALLOWED_MODULE_FORMATS:
            fail(f"{name}.format is not supported")
        path = require_non_empty_string(item.get("path"), f"{name}.path")
        if module_format == "tx.kcore" and not path.endswith(".so"):
            fail(f"{name}.path must name a kcore shared object")
        by_name[module_name] = item
    return by_name


def validate_runtime(value: Any) -> None:
    runtime = require_dict(value, "runtime")
    mode = require_non_empty_string(runtime.get("mode"), "runtime.mode")
    if mode not in ALLOWED_RUNTIME_MODES:
        fail("runtime.mode is not supported")
    completion_source = require_non_empty_string(
        runtime.get("completion_source"), "runtime.completion_source"
    )
    if completion_source in KNOWN_STUB_FENCES:
        fail("completion source is a known stub fence")
    if completion_source not in ALLOWED_COMPLETION_SOURCES:
        fail("completion source is not in the allowed runtime fence set")
    if completion_source == "legacy_model_sync" and mode != "legacy_tsm":
        fail("legacy_model_sync completion source requires legacy_tsm runtime mode")
    if mode == "legacy_tsm" and completion_source != "legacy_model_sync":
        fail("legacy_tsm runtime mode requires legacy_model_sync completion source")
    requirements = require_dict(runtime.get("requirements"), "runtime.requirements")
    for field in (
        "device_selection",
        "tile_selection",
        "stream_policy",
        "copy_policy",
    ):
        require_non_empty_string(
            requirements.get(field), f"runtime.requirements.{field}"
        )


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


def validate_workspace_buffers(
    value: Any, name: str = "workspace"
) -> tuple[int, set[str], dict[str, int]]:
    total = 0
    seen_names = set()
    binding_bytes: dict[str, int] = {}
    for index, workspace in enumerate(require_list(value, name)):
        item_name = f"{name}[{index}]"
        item = require_dict(workspace, item_name)
        buffer_name, bytes_value = validate_tensor_storage_demand(item, item_name)
        if buffer_name in seen_names:
            fail("workspace buffer names must be unique")
        seen_names.add(buffer_name)
        require_non_empty_string(item.get("producer"), f"{item_name}.producer")
        require_non_empty_string(
            item.get("last_consumer"), f"{item_name}.last_consumer"
        )
        total += bytes_value
        binding_bytes[buffer_name] = bytes_value
    return total, seen_names, binding_bytes


def validate_resident_constants(
    value: Any,
    input_names: set[str],
    output_names: set[str],
    parameter_names: set[str],
    name: str = "resident_constants",
) -> tuple[int, set[str], dict[str, int]]:
    total = 0
    seen_names = set()
    binding_bytes: dict[str, int] = {}
    for index, constant in enumerate(require_list(value, name)):
        item_name = f"{name}[{index}]"
        item = require_dict(constant, item_name)
        constant_name, bytes_value = validate_tensor_storage_demand(item, item_name)
        if constant_name in seen_names:
            fail("resident constant names must be unique")
        seen_names.add(constant_name)

        source = require_non_empty_string(item.get("source"), f"{item_name}.source")
        if source == "launch_input":
            source_binding = require_non_empty_string(
                item.get("source_binding"), f"{item_name}.source_binding"
            )
            if source_binding not in input_names:
                fail(f"{item_name}.source_binding must refer to a model input")
        elif source == "parameter":
            source_binding = require_non_empty_string(
                item.get("source_binding"), f"{item_name}.source_binding"
            )
            if source_binding not in parameter_names:
                fail(f"{item_name}.source_binding must refer to a model parameter")
        elif source != "embedded_constant":
            fail(
                f"{item_name}.source must be launch_input, parameter, or embedded_constant"
            )
        if constant_name in output_names:
            fail(f"{item_name}.name must not refer to a model output")
        total += bytes_value
        binding_bytes[constant_name] = bytes_value
    return total, seen_names, binding_bytes


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{name} must be boolean")
    return value


def validate_external_tensor(
    tensor: Any, name: str, binding_kind: str, read_only: bool
) -> tuple[str, int]:
    item = require_dict(tensor, name)
    tensor_name, shape = validate_tensor(item, name)
    dtype = require_non_empty_string(item.get("dtype"), f"{name}.dtype")
    binding = require_dict(item.get("binding"), f"{name}.binding")
    kind = require_non_empty_string(binding.get("kind"), f"{name}.binding.kind")
    if kind != binding_kind:
        fail(f"{name}.binding.kind must be {binding_kind}")
    bytes_value = require_positive_int(binding.get("bytes"), f"{name}.binding.bytes")
    expected_bytes = compact_tensor_bytes(shape, dtype, name)
    if bytes_value != expected_bytes:
        fail(f"{name}.binding.bytes must match compact tensor storage size")
    require_positive_int(binding.get("alignment"), f"{name}.binding.alignment")
    if (
        require_bool(binding.get("read_only"), f"{name}.binding.read_only")
        != read_only
    ):
        expected = "read-only" if read_only else "writable"
        fail(f"{name}.binding must be {expected}")
    require_bool(binding.get("host_visible"), f"{name}.binding.host_visible")
    return tensor_name, bytes_value


def merge_binding(
    binding_bytes: dict[str, int], name: str, bytes_value: int, description: str
) -> None:
    if name in binding_bytes:
        fail(f"model interface binding name {name} is duplicated in {description}")
    binding_bytes[name] = bytes_value


def validate_model(value: Any) -> tuple[str, str, dict[str, int]]:
    model = require_dict(value, "model")
    reject_deprecated_keys(model, "model", {"abi_version"})
    model_id = require_non_empty_string(model.get("id"), "model.id")
    abi = require_non_empty_string(model.get("abi"), "model.abi")
    if abi not in ALLOWED_ABI_VERSIONS:
        fail("model.abi is not supported")

    interface = require_dict(model.get("interface"), "model.interface")
    binding_bytes: dict[str, int] = {}

    input_bytes = 0
    input_names: set[str] = set()
    for index, tensor in enumerate(
        require_list(interface.get("inputs"), "model.interface.inputs")
    ):
        name, bytes_value = validate_external_tensor(
            tensor, f"model.interface.inputs[{index}]", "external_input", True
        )
        merge_binding(binding_bytes, name, bytes_value, "inputs")
        input_names.add(name)
        input_bytes += bytes_value

    output_bytes = 0
    output_names: set[str] = set()
    for index, tensor in enumerate(
        require_list(interface.get("outputs"), "model.interface.outputs")
    ):
        name, bytes_value = validate_external_tensor(
            tensor, f"model.interface.outputs[{index}]", "external_output", False
        )
        merge_binding(binding_bytes, name, bytes_value, "outputs")
        output_names.add(name)
        output_bytes += bytes_value

    parameter_bytes = 0
    parameter_names: set[str] = set()
    for index, tensor in enumerate(
        require_list(interface.get("parameters"), "model.interface.parameters")
    ):
        name, bytes_value = validate_external_tensor(
            tensor, f"model.interface.parameters[{index}]", "parameter", True
        )
        merge_binding(binding_bytes, name, bytes_value, "parameters")
        parameter_names.add(name)
        parameter_bytes += bytes_value

    workspace_bytes, workspace_names, workspace_binding_bytes = validate_workspace_buffers(
        interface.get("workspace"), "model.interface.workspace"
    )
    for name, bytes_value in workspace_binding_bytes.items():
        merge_binding(binding_bytes, name, bytes_value, "workspace")

    (
        resident_constant_bytes,
        resident_names,
        resident_binding_bytes,
    ) = validate_resident_constants(
        interface.get("resident_constants"),
        input_names,
        output_names,
        parameter_names,
        "model.interface.resident_constants",
    )
    for name, bytes_value in resident_binding_bytes.items():
        merge_binding(binding_bytes, name, bytes_value, "resident constants")

    if input_names & output_names:
        fail("model input and output names must be distinct")
    if (input_names | output_names | parameter_names) & workspace_names:
        fail("workspace names must be distinct from external model bindings")
    if (
        input_names | output_names | parameter_names | workspace_names
    ) & resident_names:
        fail("resident constant names must be distinct from other model bindings")

    resources = require_dict(model.get("resources"), "model.resources")
    spm_bytes = require_positive_int(
        resources.get("spm_bytes"), "model.resources.spm_bytes"
    )
    if spm_bytes > 0x2F0000:
        fail("model.resources.spm_bytes exceeds usable SPM capacity")
    if resources.get("ddr_external_input_bytes") != input_bytes:
        fail("model.resources.ddr_external_input_bytes does not match model inputs")
    if resources.get("ddr_external_output_bytes") != output_bytes:
        fail("model.resources.ddr_external_output_bytes does not match model outputs")
    if resources.get("workspace_bytes") != workspace_bytes:
        fail("model.resources.workspace_bytes does not match workspace buffers")
    if resources.get("resident_constant_bytes") != resident_constant_bytes:
        fail(
            "model.resources.resident_constant_bytes does not match resident constants"
        )
    if (
        "ddr_parameter_bytes" in resources
        and resources.get("ddr_parameter_bytes") != parameter_bytes
    ):
        fail("model.resources.ddr_parameter_bytes does not match model parameters")
    if parameter_bytes and "ddr_parameter_bytes" not in resources:
        fail("model.resources.ddr_parameter_bytes is required when parameters exist")

    return model_id, abi, binding_bytes


def validate_dim3(value: Any, name: str) -> None:
    dims = require_list(value, name)
    if len(dims) != 3:
        fail(f"{name} must contain three dimensions")
    for index, dim in enumerate(dims):
        require_positive_int(dim, f"{name}[{index}]")


def validate_binding_order(
    value: Any, name: str, binding_bytes: dict[str, int]
) -> None:
    order = require_list(value, name)
    if not order:
        fail(f"{name} must be non-empty")
    for index, binding_name in enumerate(order):
        item = require_non_empty_string(binding_name, f"{name}[{index}]")
        if item not in binding_bytes:
            fail(f"{name}[{index}] does not name a model binding")


def validate_entrypoints(
    value: Any,
    modules: dict[str, dict[str, Any]],
    binding_bytes: dict[str, int],
    runtime_mode: str,
) -> None:
    entrypoints = require_list(value, "entrypoints")
    if not entrypoints:
        fail("entrypoints must be non-empty")
    seen_names = set()
    for index, entrypoint in enumerate(entrypoints):
        name = f"entrypoints[{index}]"
        item = require_dict(entrypoint, name)
        reject_deprecated_keys(
            item,
            name,
            {
                "abi_version",
                "debug_or_bringup",
                "entrypoint",
                "graph_module",
                "kind",
            },
        )
        entrypoint_name = require_non_empty_string(item.get("name"), f"{name}.name")
        if entrypoint_name in seen_names:
            fail("entrypoint names must be unique")
        seen_names.add(entrypoint_name)
        executor = require_non_empty_string(item.get("executor"), f"{name}.executor")
        if executor not in ALLOWED_ENTRYPOINT_EXECUTORS:
            fail(f"{name}.executor is not supported")
        validate_binding_order(
            item.get("binding_order"), f"{name}.binding_order", binding_bytes
        )

        if executor in {"tx.module", "tx.cluster"}:
            module = require_non_empty_string(item.get("module"), f"{name}.module")
            if module not in modules:
                fail(f"{name}.module does not name a package module")
            if modules[module]["format"] != "tx.kcore":
                fail(f"{name}.module must reference a tx.kcore module")
            function = require_non_empty_string(
                item.get("function"), f"{name}.function"
            )
            if function.startswith("@"):
                fail(f"{name}.function must not include @")
            if require_bool(item.get("debug"), f"{name}.debug") is not True:
                fail(f"{name}.debug must be true")
            validate_dim3(item.get("grid"), f"{name}.grid")
            validate_dim3(item.get("block"), f"{name}.block")
            if executor == "tx.cluster":
                validate_dim3(item.get("cluster"), f"{name}.cluster")
        elif executor == "tx.model":
            descriptor = require_dict(
                item.get("bpm_descriptor"), f"{name}.bpm_descriptor"
            )
            state = require_non_empty_string(
                descriptor.get("state"), f"{name}.bpm_descriptor.state"
            )
            if state not in ALLOWED_BPM_DESCRIPTOR_STATES:
                fail(f"{name}.bpm_descriptor.state is not supported")
        elif executor == "tx.graph":
            module = require_non_empty_string(item.get("module"), f"{name}.module")
            if module not in modules:
                fail(f"{name}.module does not name a package module")
            if modules[module]["format"] != "tx.graph":
                fail(f"{name}.module must reference a tx.graph module")
            require_non_empty_string(item.get("mod_symbol"), f"{name}.mod_symbol")
        elif executor == "legacy.tsm" and runtime_mode != "legacy_tsm":
            fail(f"{name}.executor legacy.tsm requires legacy_tsm runtime mode")


def validate_package_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("schema_version") != 2:
        fail("schema_version must be 2")
    reject_deprecated_keys(
        metadata,
        "metadata",
        {"artifacts", "backend_strategies", "package_name"},
    )
    require_non_empty_string(metadata.get("name"), "name")

    validate_runtime(metadata.get("runtime"))
    runtime_mode = require_dict(metadata.get("runtime"), "runtime")["mode"]
    _model_id, _model_abi, binding_bytes = validate_model(metadata.get("model"))
    modules = validate_modules(metadata.get("modules"))
    validate_entrypoints(
        metadata.get("entrypoints"),
        modules,
        binding_bytes,
        runtime_mode,
    )

    instructions = require_list(metadata.get("instructions"), "instructions")
    for index, op in enumerate(instructions):
        item = require_dict(op, f"instructions[{index}]")
        mnemonic = require_non_empty_string(
            item.get("op"), f"instructions[{index}].op"
        )
        if mnemonic not in SUPPORTED_INSTRUCTION_OPS:
            fail(f"instructions[{index}].op is not supported by the package metadata")
        wait_policy = item.get("wait_policy")
        if mnemonic in {"wafer.instr.local_fence", "wafer.instr.dte_wait"}:
            if wait_policy != "local_wait":
                fail(f"instructions[{index}].wait_policy must be local_wait")
            continue
        if wait_policy != "issue_only":
            fail(f"instructions[{index}].wait_policy must be issue_only")
        if mnemonic in {
            "wafer.instr.rdma",
            "wafer.instr.wdma",
            "wafer.instr.gather_scatter",
            "wafer.instr.dte_send",
            "wafer.instr.dte_recv",
        }:
            require_positive_int(item.get("bytes"), f"instructions[{index}].bytes")
            if "inner_bytes" in item:
                require_positive_int(
                    item.get("inner_bytes"), f"instructions[{index}].inner_bytes"
                )
        elif mnemonic == "wafer.instr.gemm":
            require_positive_int(item.get("m"), f"instructions[{index}].m")
            require_positive_int(item.get("k"), f"instructions[{index}].k")
            require_positive_int(item.get("n"), f"instructions[{index}].n")
            if "batch_count" in item:
                require_positive_int(
                    item.get("batch_count"), f"instructions[{index}].batch_count"
                )
        elif mnemonic == "wafer.instr.elementwise":
            kind = require_non_empty_string(
                item.get("kind"), f"instructions[{index}].kind"
            )
            if kind not in SUPPORTED_ELEMENTWISE_KINDS:
                fail(f"instructions[{index}].kind is not supported")
        elif mnemonic == "wafer.instr.reduce":
            kind = require_non_empty_string(
                item.get("kind"), f"instructions[{index}].kind"
            )
            if kind not in SUPPORTED_REDUCE_KINDS:
                fail(f"instructions[{index}].kind is not supported")
            dimensions = require_list(
                item.get("dimensions"), f"instructions[{index}].dimensions"
            )
            if not dimensions:
                fail(f"instructions[{index}].dimensions must be non-empty")
            if len(dimensions) > 4:
                fail(
                    f"instructions[{index}].dimensions supports at most four dimensions"
                )
            for dim_index, dim in enumerate(dimensions):
                require_non_negative_int(
                    dim, f"instructions[{index}].dimensions[{dim_index}]"
                )
            validate_reduce_init_value(
                item.get("init_value"), f"instructions[{index}].init_value"
            )
        elif "kind" in item:
            fail(
                f"instructions[{index}].kind is only valid for elementwise or reduce ops"
            )


def load_package_metadata(path: str) -> dict[str, Any]:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    return require_dict(json.loads(text), "metadata")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--roundtrip")
    mode.add_argument("--validate")
    args = parser.parse_args()

    metadata = load_package_metadata(args.roundtrip or args.validate)
    validate_package_metadata(metadata)
    if args.roundtrip:
        sys.stdout.write(canonical_json(metadata))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
