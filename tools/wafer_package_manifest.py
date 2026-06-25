#!/usr/bin/env python3
"""Validate and roundtrip Wafer runtime package manifests."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


ALLOWED_COMPLETION_SOURCES = {
    "runtime_stream_wait",
    "runtime_command_completion",
    "kcore_local_drain",
    "legacy_model_sync",
}

ALLOWED_ARTIFACT_KINDS = {
    "bpm_table_descriptor",
    "graph_directory",
    "kcore_shared_object",
}

ALLOWED_RUNTIME_MODES = {
    "tx",
    "legacy_tsm",
}

ALLOWED_ABI_VERSIONS = {
    "wafer-cabi-v0",
}

ALLOWED_BACKEND_STRATEGY_KINDS = {
    "legacy_tsm_run",
    "tx_cluster_kernel",
    "tx_graph",
    "tx_model_bpm",
    "tx_module_kernel",
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
    "wafer.instr.gemm",
    "wafer.instr.elementwise",
    "wafer.instr.reduce",
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


def validate_artifacts(value: Any) -> dict[str, dict[str, Any]]:
    artifacts = require_list(value, "artifacts")
    if not artifacts:
        fail("artifacts must be non-empty")
    by_name: dict[str, dict[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        name = f"artifacts[{index}]"
        item = require_dict(artifact, name)
        artifact_name = require_non_empty_string(item.get("name"), f"{name}.name")
        if artifact_name in by_name:
            fail("artifact names must be unique")
        kind = require_non_empty_string(item.get("kind"), f"{name}.kind")
        if kind not in ALLOWED_ARTIFACT_KINDS:
            fail(f"{name}.kind is not supported")
        path = require_non_empty_string(item.get("path"), f"{name}.path")
        if kind == "kcore_shared_object" and not path.endswith(".so"):
            fail(f"{name}.path must name a kcore shared object")
        by_name[artifact_name] = item
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
        require_non_empty_string(requirements.get(field), f"runtime.requirements.{field}")


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


def validate_workspace_buffers(value: Any, name: str = "workspace") -> tuple[int, set[str], dict[str, int]]:
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
        require_non_empty_string(item.get("last_consumer"), f"{item_name}.last_consumer")
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
            fail(f"{item_name}.source must be launch_input, parameter, or embedded_constant")
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
    if require_bool(binding.get("read_only"), f"{name}.binding.read_only") != read_only:
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
    model_id = require_non_empty_string(model.get("id"), "model.id")
    abi_version = require_non_empty_string(model.get("abi_version"), "model.abi_version")
    if abi_version not in ALLOWED_ABI_VERSIONS:
        fail("model.abi_version is not supported")

    interface = require_dict(model.get("interface"), "model.interface")
    binding_bytes: dict[str, int] = {}

    input_bytes = 0
    input_names: set[str] = set()
    for index, tensor in enumerate(require_list(interface.get("inputs"), "model.interface.inputs")):
        name, bytes_value = validate_external_tensor(
            tensor, f"model.interface.inputs[{index}]", "external_input", True
        )
        merge_binding(binding_bytes, name, bytes_value, "inputs")
        input_names.add(name)
        input_bytes += bytes_value

    output_bytes = 0
    output_names: set[str] = set()
    for index, tensor in enumerate(require_list(interface.get("outputs"), "model.interface.outputs")):
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
    if (input_names | output_names | parameter_names | workspace_names) & resident_names:
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
    if "ddr_parameter_bytes" in resources and resources.get("ddr_parameter_bytes") != parameter_bytes:
        fail("model.resources.ddr_parameter_bytes does not match model parameters")
    if parameter_bytes and "ddr_parameter_bytes" not in resources:
        fail("model.resources.ddr_parameter_bytes is required when parameters exist")

    return model_id, abi_version, binding_bytes


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


def validate_backend_strategies(
    value: Any,
    artifacts: dict[str, dict[str, Any]],
    model_abi_version: str,
    binding_bytes: dict[str, int],
    runtime_mode: str,
) -> None:
    strategies = require_list(value, "backend_strategies")
    if not strategies:
        fail("backend_strategies must be non-empty")
    seen_names = set()
    for index, strategy in enumerate(strategies):
        name = f"backend_strategies[{index}]"
        item = require_dict(strategy, name)
        strategy_name = require_non_empty_string(item.get("name"), f"{name}.name")
        if strategy_name in seen_names:
            fail("backend strategy names must be unique")
        seen_names.add(strategy_name)
        kind = require_non_empty_string(item.get("kind"), f"{name}.kind")
        if kind not in ALLOWED_BACKEND_STRATEGY_KINDS:
            fail(f"{name}.kind is not supported")
        validate_binding_order(item.get("binding_order"), f"{name}.binding_order", binding_bytes)

        if kind in {"tx_module_kernel", "tx_cluster_kernel"}:
            artifact = require_non_empty_string(item.get("artifact"), f"{name}.artifact")
            if artifact not in artifacts:
                fail(f"{name}.artifact does not name a package artifact")
            if artifacts[artifact]["kind"] != "kcore_shared_object":
                fail(f"{name}.artifact must reference a kcore_shared_object")
            entrypoint = require_non_empty_string(item.get("entrypoint"), f"{name}.entrypoint")
            if entrypoint.startswith("@"):
                fail(f"{name}.entrypoint must not include @")
            abi_version = require_non_empty_string(item.get("abi_version"), f"{name}.abi_version")
            if abi_version != model_abi_version:
                fail(f"{name}.abi_version must match model.abi_version")
            if require_bool(item.get("debug_or_bringup"), f"{name}.debug_or_bringup") is not True:
                fail(f"{name}.debug_or_bringup must be true")
            validate_dim3(item.get("grid"), f"{name}.grid")
            validate_dim3(item.get("block"), f"{name}.block")
            if kind == "tx_cluster_kernel":
                validate_dim3(item.get("cluster"), f"{name}.cluster")
        elif kind == "tx_model_bpm":
            descriptor = require_dict(item.get("bpm_descriptor"), f"{name}.bpm_descriptor")
            state = require_non_empty_string(
                descriptor.get("state"), f"{name}.bpm_descriptor.state"
            )
            if state not in ALLOWED_BPM_DESCRIPTOR_STATES:
                fail(f"{name}.bpm_descriptor.state is not supported")
        elif kind == "tx_graph":
            artifact = require_non_empty_string(item.get("graph_artifact"), f"{name}.graph_artifact")
            if artifact not in artifacts:
                fail(f"{name}.graph_artifact does not name a package artifact")
            if artifacts[artifact]["kind"] != "graph_directory":
                fail(f"{name}.graph_artifact must reference a graph_directory")
            require_non_empty_string(item.get("mod_symbol"), f"{name}.mod_symbol")
        elif kind == "legacy_tsm_run" and runtime_mode != "legacy_tsm":
            fail(f"{name}.kind legacy_tsm_run requires legacy_tsm runtime mode")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 2:
        fail("schema_version must be 2")
    require_non_empty_string(manifest.get("package_name"), "package_name")

    validate_runtime(manifest.get("runtime"))
    runtime_mode = require_dict(manifest.get("runtime"), "runtime")["mode"]
    _model_id, model_abi_version, binding_bytes = validate_model(manifest.get("model"))
    artifacts = validate_artifacts(manifest.get("artifacts"))
    validate_backend_strategies(
        manifest.get("backend_strategies"),
        artifacts,
        model_abi_version,
        binding_bytes,
        runtime_mode,
    )

    instructions = require_list(manifest.get("instructions"), "instructions")
    for index, op in enumerate(instructions):
        item = require_dict(op, f"instructions[{index}]")
        mnemonic = require_non_empty_string(item.get("op"), f"instructions[{index}].op")
        if mnemonic not in SUPPORTED_INSTRUCTION_OPS:
            fail(f"instructions[{index}].op is not supported by the package manifest")
        wait_policy = item.get("wait_policy")
        if mnemonic == "wafer.instr.local_fence":
            if wait_policy != "local_wait":
                fail(f"instructions[{index}].wait_policy must be local_wait")
            continue
        if wait_policy != "issue_only":
            fail(f"instructions[{index}].wait_policy must be issue_only")
        if mnemonic in {
            "wafer.instr.rdma",
            "wafer.instr.wdma",
            "wafer.instr.gather_scatter",
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
