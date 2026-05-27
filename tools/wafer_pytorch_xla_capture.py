#!/usr/bin/env python3
"""PyTorch/XLA capture adapter for Wafer frontend artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable


P2F1_SMOKE_SIZE = 4096


_DTYPE_TO_MLIR = {
    "torch.float32": "f32",
    "float32": "f32",
    "float": "f32",
    "torch.float16": "f16",
    "float16": "f16",
    "half": "f16",
    "torch.bfloat16": "bf16",
    "bfloat16": "bf16",
    "torch.float64": "f64",
    "float64": "f64",
    "double": "f64",
    "torch.int8": "i8",
    "int8": "i8",
    "torch.int16": "i16",
    "int16": "i16",
    "torch.int32": "i32",
    "int32": "i32",
    "torch.int64": "i64",
    "int64": "i64",
}

_DTYPE_BYTES = {
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
}


@dataclass(frozen=True)
class ConstantArgument:
    arg: int
    resource_key: str
    shape: list[int]
    dtype: str
    byte_size: int


def _dtype_to_mlir(dtype: Any) -> str:
    dtype_name = str(dtype)
    result = _DTYPE_TO_MLIR.get(dtype_name)
    if result is None:
        raise RuntimeError(f"unsupported frontend dtype for sidecar: {dtype_name}")
    return result


def _byte_size(shape: Iterable[int], dtype: str) -> int:
    if dtype not in _DTYPE_BYTES:
        raise RuntimeError(f"unsupported sidecar dtype byte width: {dtype}")
    return math.prod(shape) * _DTYPE_BYTES[dtype]


def _location_kind(location: Any) -> str:
    kind = getattr(location, "type_")
    value = getattr(kind, "value", kind)
    return str(value).split(".")[-1].lower()


def _is_resource_backed_constant(location: Any) -> bool:
    return _location_kind(location) == "parameter"


def _symbol_name_from_func_prefix(prefix: str) -> str:
    match = re.search(r"@([^\s(]+)", prefix)
    if not match:
        raise RuntimeError("failed to find func.func symbol name in StableHLO artifact")
    return match.group(1).strip('"')


def _find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise RuntimeError("unterminated func.func argument list in StableHLO artifact")


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    paren_depth = 0
    angle_depth = 0
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif (
            char == ","
            and paren_depth == 0
            and angle_depth == 0
            and brace_depth == 0
            and bracket_depth == 0
        ):
            parts.append(text[start:index])
            start = index + 1

    parts.append(text[start:])
    return parts


def _annotate_argument(argument: str, resource_key: str) -> str:
    attr = f'wafer.frontend.constant = "{resource_key}"'
    if attr in argument:
        return argument
    if "{" in argument and "}" in argument:
        return re.sub(r"\{", "{%s, " % attr, argument, count=1)
    return f"{argument} {{{attr}}}"


def _annotate_stablehlo_text(
    stablehlo_text: str, constants: list[ConstantArgument]
) -> tuple[str, str]:
    func_index = stablehlo_text.find("func.func")
    if func_index < 0:
        raise RuntimeError("StableHLO artifact does not contain a func.func")
    open_index = stablehlo_text.find("(", func_index)
    if open_index < 0:
        raise RuntimeError("StableHLO func.func does not have an argument list")
    close_index = _find_matching_paren(stablehlo_text, open_index)

    prefix = stablehlo_text[func_index:open_index]
    function_name = _symbol_name_from_func_prefix(prefix)
    args_text = stablehlo_text[open_index + 1 : close_index]
    args = _split_top_level_commas(args_text) if args_text.strip() else []

    for constant in constants:
        if constant.arg >= len(args):
            raise RuntimeError(
                f"StableHLO metadata references argument {constant.arg}, "
                f"but func.func has only {len(args)} arguments"
            )
        args[constant.arg] = _annotate_argument(args[constant.arg], constant.resource_key)

    annotated_args = ",".join(args)
    annotated_text = (
        stablehlo_text[: open_index + 1]
        + annotated_args
        + stablehlo_text[close_index:]
    )
    return function_name, annotated_text


def _constant_arguments_from_meta(meta: Any) -> list[ConstantArgument]:
    constants: list[ConstantArgument] = []
    for arg_index, (location, signature) in enumerate(
        zip(meta.input_locations, meta.input_signature)
    ):
        if not _is_resource_backed_constant(location):
            continue

        shape = [int(dim) for dim in signature.shape]
        dtype = _dtype_to_mlir(signature.dtype)
        resource_key = f"const_arg_{arg_index}"
        constants.append(
            ConstantArgument(
                arg=arg_index,
                resource_key=resource_key,
                shape=shape,
                dtype=dtype,
                byte_size=_byte_size(shape, dtype),
            )
        )
    return constants


def _sidecar(function_name: str, constants: list[ConstantArgument]) -> dict[str, Any]:
    return {
        "version": 0,
        "constants": [
            {
                "function": function_name,
                "arg": constant.arg,
                "resource_key": constant.resource_key,
                "shape": constant.shape,
                "dtype": constant.dtype,
                "byte_size": constant.byte_size,
            }
            for constant in constants
        ],
    }


def _compile_config(size: int) -> dict[str, Any]:
    return {
        "version": 0,
        "artifact_kind": "p2f1_framework_capture_smoke",
        "input_shape": [size, size],
        "input_dtype": "f32",
        "constant_binding": "function_arg_sidecar",
        "dynamic_shape_policy": "static_smoke",
        "sharding_import_policy": "preserve_stablehlo_sdy_annotations",
    }


def _make_smoke_module(torch_module: Any, size: int) -> Any:
    class WaferCaptureSmoke4096(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch_module.nn.Parameter(
                torch_module.empty(size, size, dtype=torch_module.float32)
            )
            self.bias = torch_module.nn.Parameter(
                torch_module.empty(size, dtype=torch_module.float32)
            )

        def forward(self, x):
            y = x @ self.weight
            y = y + self.bias
            z = torch_module.tanh(y)
            return z + y

    return WaferCaptureSmoke4096()


def _import_runtime_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch_xla import stablehlo
    except ImportError as error:
        raise RuntimeError(
            "PyTorch/XLA importer runtime is unavailable; install pinned torch "
            "Python packages, then build/install torch_xla from "
            "third_party/pytorch-xla with tools/build_pytorch_xla_runtime.py"
        ) from error
    return torch, stablehlo


def emit_p2f1_smoke_artifact(
    mlir_path: pathlib.Path,
    sidecar_path: pathlib.Path,
    config_path: pathlib.Path | None = None,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
    smoke_module_factory: Callable[[], Any] | None = None,
    size: int = P2F1_SMOKE_SIZE,
) -> None:
    if torch_module is None or stablehlo_module is None:
        torch_module, stablehlo_module = _import_runtime_modules()

    if smoke_module_factory is None:
        smoke_module = _make_smoke_module(torch_module, size)
    else:
        smoke_module = smoke_module_factory()
    smoke_module.eval()
    input_tensor = torch_module.empty(size, size, dtype=torch_module.float32)

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = False
    options.inline_all_constant = False
    options.include_human_readable_text = True

    with torch_module.no_grad():
        exported = torch_module.export.export(smoke_module, (input_tensor,))
        stablehlo_program = stablehlo_module.exported_program_to_stablehlo(
            exported, options=options
        )

    stablehlo_text = stablehlo_program.get_stablehlo_text()
    if not stablehlo_text:
        raise RuntimeError("PyTorch/XLA did not emit human-readable StableHLO text")

    meta = stablehlo_program._bundle.stablehlo_funcs[0].meta
    constants = _constant_arguments_from_meta(meta)
    function_name, annotated_text = _annotate_stablehlo_text(stablehlo_text, constants)

    mlir_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    mlir_path.write_text(annotated_text, encoding="utf-8")
    sidecar_path.write_text(
        json.dumps(_sidecar(function_name, constants), indent=2) + "\n",
        encoding="utf-8",
    )
    if config_path is not None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(_compile_config(size), indent=2) + "\n",
            encoding="utf-8",
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-p2f1-smoke",
        action="store_true",
        help="emit the P2.F1 4096x4096 matmul+bias+tanh+residual artifact",
    )
    parser.add_argument("--output-mlir", type=pathlib.Path)
    parser.add_argument("--output-sidecar", type=pathlib.Path)
    parser.add_argument("--output-config", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not args.emit_p2f1_smoke:
        raise RuntimeError("missing --emit-p2f1-smoke")
    if args.output_mlir is None:
        raise RuntimeError("missing --output-mlir")
    if args.output_sidecar is None:
        raise RuntimeError("missing --output-sidecar")

    emit_p2f1_smoke_artifact(args.output_mlir, args.output_sidecar, args.output_config)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-capture: {error}", file=sys.stderr)
        raise SystemExit(1)
