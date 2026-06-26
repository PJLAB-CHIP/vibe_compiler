#!/usr/bin/env python3
"""Export Wafer runtime package metadata from compiler modules."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from wafer_package_metadata import (  # noqa: E402
    canonical_json,
    compact_tensor_bytes,
    validate_package_metadata,
)


DTYPE_BITS = {
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "index": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "f64": 64,
}

SUPPORTED_INSTRUCTION_OPS = {
    "rdma",
    "wdma",
    "gather_scatter",
    "fill",
    "gemm",
    "elementwise",
    "reduce",
    "convert",
    "dte_send",
    "dte_recv",
    "dte_wait",
    "local_fence",
}


@dataclass(frozen=True)
class MemRefInfo:
    shape: list[int]
    dtype: str
    space: str
    layout: str
    physical_bytes: int
    compact_bytes: int


def fail(message: str) -> None:
    raise ValueError(message)


def read_text(path: str, description: str) -> str:
    item = pathlib.Path(path)
    if not item.exists():
        fail(f"{description} input does not exist: {item}")
    if not item.is_file():
        fail(f"{description} input is not a file: {item}")
    return item.read_text(encoding="utf-8")


def load_json_object(path: str, description: str) -> dict[str, Any]:
    value = json.loads(read_text(path, description))
    if not isinstance(value, dict):
        fail(f"{description} must be a JSON object")
    return value


def strip_mlir_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def top_level_split(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    angle_depth = 0
    square_depth = 0
    paren_depth = 0
    for index, char in enumerate(text):
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif (
            char == sep
            and angle_depth == 0
            and square_depth == 0
            and paren_depth == 0
        ):
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def find_memref_type(segment: str) -> str | None:
    start = segment.find("memref<")
    if start < 0:
        return None
    index = start + len("memref<")
    depth = 1
    while index < len(segment):
        char = segment[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return segment[start : index + 1]
        index += 1
    return None


def parse_shape_dtype(head: str) -> tuple[list[int], str]:
    pieces = head.split("x")
    dtype = pieces[-1]
    shape: list[int] = []
    for piece in pieces[:-1]:
        if piece == "?":
            fail("dynamic memref dimensions cannot be exported to package metadata")
        shape.append(int(piece))
    if dtype not in DTYPE_BITS:
        fail(f"unsupported memref dtype for package metadata export: {dtype}")
    return shape, dtype


def ceil_div(value: int, divisor: int) -> int:
    return value // divisor + (1 if value % divisor else 0)


def ceil_div_to_bytes(bits: int) -> int:
    return ceil_div(bits, 8)


def element_storage_bytes(dtype: str) -> int:
    return ceil_div_to_bytes(DTYPE_BITS[dtype])


def compact_bytes(shape: list[int], dtype: str) -> int:
    elements = math.prod(shape) if shape else 1
    return ceil_div_to_bytes(elements * DTYPE_BITS[dtype])


def align_to(value: int, alignment: int) -> int:
    remainder = value % alignment
    if remainder == 0:
        return value
    return value + alignment - remainder


def align_cx_tail(tail: int) -> int:
    if tail <= 4:
        return 4
    if tail <= 8:
        return 8
    if tail <= 16:
        return 16
    if tail <= 32:
        return 32
    return 64


def compute_cx_aligned_c(dtype: str, channels: int) -> int:
    c_block = 128 if dtype == "i8" else 64
    retain_threshold = 64 if dtype == "i8" else 32
    quotient, remainder = divmod(channels, c_block)
    if remainder == 0:
        return quotient * c_block
    if remainder <= retain_threshold:
        return quotient * c_block + align_cx_tail(remainder)
    return (quotient + 1) * c_block


def parse_strided_elements(memref_type: str, shape: list[int]) -> int | None:
    match = re.search(r"strided<\s*\[([^\]]*)\]", memref_type)
    if not match:
        return None
    strides = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
    if len(strides) != len(shape):
        fail(f"strided memref rank mismatch in {memref_type}")
    if not shape:
        return 1
    span = 1
    for dim, stride in zip(shape, strides, strict=True):
        if dim <= 0 or stride < 0:
            fail(f"unsupported memref shape/stride in {memref_type}")
        span += (dim - 1) * stride
    return span


def physical_bytes(shape: list[int], dtype: str, layout: str, memref_type: str) -> int:
    if layout in {"tensor", "ntensor"}:
        elements = parse_strided_elements(memref_type, shape)
        if elements is None:
            elements = math.prod(shape) if shape else 1
        return ceil_div_to_bytes(elements * DTYPE_BITS[dtype])

    if layout not in {"cx", "ncx"}:
        fail(f"unsupported Wafer memory layout for package export: {layout}")
    if not shape:
        return compact_bytes(shape, dtype)

    aligned_c = compute_cx_aligned_c(dtype, shape[-1])
    bank_align_elements = 256 // element_storage_bytes(dtype)
    if layout == "ncx":
        n = shape[0] if len(shape) > 1 else 1
        hw_shape = shape[1:-1] if len(shape) > 1 else []
        hw_elements = math.prod(hw_shape) if hw_shape else 1
        batch_elements = align_to(hw_elements * aligned_c, bank_align_elements)
        elements = n * batch_elements
    else:
        outer_shape = shape[:-1]
        outer_elements = math.prod(outer_shape) if outer_shape else 1
        elements = align_to(outer_elements * aligned_c, bank_align_elements)
    return ceil_div_to_bytes(elements * DTYPE_BITS[dtype])


def parse_memref_type(memref_type: str) -> MemRefInfo:
    if not memref_type.startswith("memref<") or not memref_type.endswith(">"):
        fail(f"invalid memref type: {memref_type}")
    body = memref_type[len("memref<") : -1]
    parts = top_level_split(body)
    if len(parts) < 2:
        fail(f"Wafer memref type is missing memory attr: {memref_type}")
    shape, dtype = parse_shape_dtype(parts[0])
    memory_match = re.search(r"#wafer\.memory<\s*([^,\s>]+)\s*,\s*([^>\s]+)\s*>", body)
    if not memory_match:
        fail(f"Wafer memref type is missing #wafer.memory attr: {memref_type}")
    space = memory_match.group(1)
    layout = memory_match.group(2)
    return MemRefInfo(
        shape=shape,
        dtype=dtype,
        space=space,
        layout=layout,
        physical_bytes=physical_bytes(shape, dtype, layout, memref_type),
        compact_bytes=compact_bytes(shape, dtype),
    )


def parse_function(llvm_ir: str, explicit_function: str | None) -> str:
    if explicit_function:
        return (
            explicit_function[1:]
            if explicit_function.startswith("@")
            else explicit_function
        )
    defined = re.findall(
        r'(?m)^\s*define\b[^@]*@(?:"([^"]+)"|([A-Za-z_.$][A-Za-z0-9_.$]*))\s*\(',
        llvm_ir,
    )
    names = [quoted or bare for quoted, bare in defined]
    if not names:
        fail("LLVM IR contains no defined entrypoint")
    abi_names = [name for name in names if name.endswith("_abi")]
    if len(abi_names) == 1:
        return abi_names[0]
    if len(names) == 1:
        return names[0]
    fail("LLVM IR has multiple candidate functions; pass --function")


def parse_int_attr(segment: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*(-?\d+)\s*:\s*i\d+", segment)
    if not match:
        fail(f"instruction is missing integer attr {name}")
    return int(match.group(1))


def parse_optional_int_attr(segment: str, name: str) -> int | None:
    match = re.search(rf"\b{name}\s*=\s*(-?\d+)\s*:\s*i\d+", segment)
    return int(match.group(1)) if match else None


def parse_array_i64_attr(segment: str, name: str) -> list[int]:
    match = re.search(rf"\b{name}\s*=\s*array<i64:\s*([^>]*)>", segment)
    if not match:
        fail(f"instruction is missing array attr {name}")
    return [int(item.strip()) for item in match.group(1).split(",") if item.strip()]


def parse_kind(segment: str, op_name: str) -> str:
    match = re.search(
        rf"wafer\.instr\.{op_name}\s+(?:#wafer\.[a-z_]+<|<)([a-z_]+)>",
        segment,
    )
    if not match:
        fail(f"wafer.instr.{op_name} is missing kind attr")
    return match.group(1)


def parse_constant_values(text: str) -> dict[str, int | float]:
    constants: dict[str, int | float] = {}
    for match in re.finditer(
        r"(%[A-Za-z0-9_.$]+)\s*=\s*arith\.constant\s+([^:\s]+)\s*:",
        text,
    ):
        raw = match.group(2)
        constants[match.group(1)] = (
            float(raw) if "." in raw or "e" in raw.lower() else int(raw)
        )
    return constants


def split_instruction_segments(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"wafer\.instr\.([A-Za-z_][A-Za-z0-9_]*)\b", text))
    segments: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        op_name = match.group(1)
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments.append((op_name, text[start:end]))
    return segments


def parse_instructions(instruction_ir: str) -> list[dict[str, Any]]:
    text = strip_mlir_comments(instruction_ir)
    constants = parse_constant_values(text)
    instructions: list[dict[str, Any]] = []
    for op_name, segment in split_instruction_segments(text):
        if op_name not in SUPPORTED_INSTRUCTION_OPS:
            fail(f"unsupported package instruction op wafer.instr.{op_name}")

        if op_name in {"local_fence", "dte_wait"}:
            instructions.append(
                {"op": f"wafer.instr.{op_name}", "wait_policy": "local_wait"}
            )
            continue

        item: dict[str, Any] = {
            "op": f"wafer.instr.{op_name}",
            "wait_policy": "issue_only",
        }
        if op_name in {"rdma", "wdma", "gather_scatter"}:
            item["bytes"] = parse_int_attr(segment, "byte_count")
            item["inner_bytes"] = parse_int_attr(segment, "inner_bytes")
        elif op_name in {"dte_send", "dte_recv"}:
            item["bytes"] = parse_int_attr(segment, "bytes")
        elif op_name == "gemm":
            item["m"] = parse_int_attr(segment, "m")
            item["k"] = parse_int_attr(segment, "k")
            item["n"] = parse_int_attr(segment, "n")
            batch_count = parse_optional_int_attr(segment, "batch_count")
            if batch_count is not None:
                item["batch_count"] = batch_count
        elif op_name == "elementwise":
            item["kind"] = parse_kind(segment, "elementwise")
        elif op_name == "reduce":
            item["kind"] = parse_kind(segment, "reduce")
            item["dimensions"] = parse_array_i64_attr(segment, "dimensions")
            init_match = re.search(
                r"wafer\.instr\.reduce\b.*?\sinto\s+%[A-Za-z0-9_.$]+,\s+(%[A-Za-z0-9_.$]+)\s*:",
                segment,
                re.S,
            )
            if not init_match or init_match.group(1) not in constants:
                fail("wafer.instr.reduce init value must resolve to arith.constant")
            item["init_value"] = constants[init_match.group(1)]
        elif op_name == "convert":
            src_dtype = re.search(r"\bsrc_dtype\s*=\s*([A-Za-z0-9]+)", segment)
            dst_dtype = re.search(r"\bdst_dtype\s*=\s*([A-Za-z0-9]+)", segment)
            if not src_dtype or not dst_dtype:
                fail("wafer.instr.convert is missing src_dtype/dst_dtype")
            item["src_dtype"] = src_dtype.group(1)
            item["dst_dtype"] = dst_dtype.group(1)
        instructions.append(item)

    return instructions


def parse_spm_bytes(instruction_ir: str) -> int:
    text = strip_mlir_comments(instruction_ir)
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"memref\.alloc\(\)", text):
        segment = text[match.start() : match.start() + 600]
        offset_match = re.search(
            r"wafer\.spm\.offset\s*=\s*#wafer\.spm_offset<(\d+)>", segment
        )
        if not offset_match:
            continue
        memref_type = find_memref_type(segment)
        if not memref_type:
            fail("SPM allocation with wafer.spm.offset is missing memref type")
        info = parse_memref_type(memref_type)
        if info.space != "spm":
            continue
        start = int(offset_match.group(1))
        ranges.append((start, start + info.physical_bytes))
    if not ranges:
        fail("instruction IR contains no accepted SPM offset facts")
    return max(end for _, end in ranges) - min(start for start, _ in ranges)


def model_external_tensor(tensor: dict[str, Any], kind: str) -> dict[str, Any]:
    name = tensor.get("name")
    shape = tensor.get("shape")
    dtype = tensor.get("dtype")
    binding_kind = "external_input" if kind == "input" else "external_output"
    item = dict(tensor)
    item["binding"] = {
        "kind": binding_kind,
        "bytes": compact_tensor_bytes(shape, dtype, f"model.interface.{kind}.{name}"),
        "alignment": 256,
        "read_only": kind == "input",
        "host_visible": True,
    }
    return item


def binding_bytes(tensor: dict[str, Any]) -> int:
    binding = tensor["binding"]
    if not isinstance(binding, dict):
        fail("model tensor binding must be an object")
    value = binding.get("bytes")
    if not isinstance(value, int):
        fail("model tensor binding bytes must be an integer")
    return value


def binding_order(*groups: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for group in groups:
        for item in group:
            name = item.get("name")
            if not isinstance(name, str) or not name:
                fail("model binding name must be a non-empty string")
            names.append(name)
    return names


def module_name_from_path(path: str) -> str:
    module = pathlib.Path(path)
    return module.stem


def validate_model_interface(
    model_interface: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    inputs = model_interface.get("inputs")
    outputs = model_interface.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        fail("model interface must contain inputs and outputs lists")
    return {"inputs": inputs, "outputs": outputs}


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    model_interface = validate_model_interface(
        load_json_object(args.model_interface, "model interface")
    )
    instruction_ir = read_text(args.instruction_ir, "instruction IR")
    llvm_ir = read_text(args.llvm_ir, "LLVM IR")
    function = parse_function(llvm_ir, args.function)

    model_inputs = [
        model_external_tensor(tensor, "input") for tensor in model_interface["inputs"]
    ]
    model_outputs = [
        model_external_tensor(tensor, "output") for tensor in model_interface["outputs"]
    ]
    input_bytes = sum(binding_bytes(tensor) for tensor in model_inputs)
    output_bytes = sum(binding_bytes(tensor) for tensor in model_outputs)
    module_name = module_name_from_path(args.module_path)

    metadata = {
        "schema_version": 2,
        "name": args.name,
        "runtime": {
            "mode": args.runtime_mode,
            "completion_source": args.completion_source,
            "requirements": {
                "device_selection": "single_device",
                "tile_selection": "full_or_pg",
                "stream_policy": "explicit_or_default",
                "copy_policy": "explicit_h2d_d2h",
            },
        },
        "model": {
            "id": args.model_id,
            "abi": args.abi,
            "interface": {
                "inputs": model_inputs,
                "outputs": model_outputs,
                "parameters": [],
                "workspace": [],
                "resident_constants": [],
            },
            "resources": {
                "spm_bytes": parse_spm_bytes(instruction_ir),
                "ddr_external_input_bytes": input_bytes,
                "ddr_external_output_bytes": output_bytes,
                "workspace_bytes": 0,
                "resident_constant_bytes": 0,
            },
        },
        "modules": [
            {
                "name": module_name,
                "format": "tx.kcore",
                "path": args.module_path,
            }
        ],
        "entrypoints": [
            {
                "name": "debug_kernel",
                "executor": "tx.module",
                "module": module_name,
                "function": function,
                "debug": True,
                "grid": [1, 1, 1],
                "block": [1, 1, 1],
                "binding_order": binding_order(model_inputs, model_outputs),
            }
        ],
        "instructions": parse_instructions(instruction_ir),
    }
    validate_package_metadata(metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--abi", required=True)
    parser.add_argument("--model-interface", required=True)
    parser.add_argument("--instruction-ir", required=True)
    parser.add_argument("--llvm-ir", required=True)
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--function")
    parser.add_argument("--runtime-mode", default="tx")
    parser.add_argument("--completion-source", default="runtime_stream_wait")
    args = parser.parse_args()

    sys.stdout.write(canonical_json(build_metadata(args)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
