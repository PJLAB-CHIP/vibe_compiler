"""Product framework export entry for Wafer source programs."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any


def _state_dict_payloads(torch: Any, exported_program: Any) -> dict[str, Any]:
    state = getattr(exported_program, "state_dict", None)
    if not isinstance(state, dict):
        raise RuntimeError("PyTorch ExportedProgram state_dict must be a mapping")
    payloads: dict[str, Any] = {}
    for name, value in state.items():
        if not isinstance(name, str):
            raise RuntimeError("PyTorch ExportedProgram state names must be strings")
        if not isinstance(value, torch.Tensor):
            payloads[name] = value
            continue
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            payloads[name] = tensor.view(torch.uint16).numpy().view("|V2")
        else:
            payloads[name] = tensor.numpy()
    return payloads


def _export_stablehlo(torch: Any, stablehlo: Any, exported_program: Any) -> Any:
    options = stablehlo.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True
    state = getattr(exported_program, "state_dict", None)
    has_bfloat16 = isinstance(state, dict) and any(
        isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16
        for value in state.values()
    )
    if not has_bfloat16:
        return stablehlo.exported_program_to_stablehlo(
            exported_program, options=options
        )

    graph_options = copy.copy(options)
    graph_options.export_weights = False
    program = stablehlo.exported_program_to_stablehlo(
        exported_program, options=graph_options
    )
    bundle = getattr(program, "_bundle", None)
    if bundle is None or not hasattr(bundle, "state_dict"):
        raise RuntimeError("pinned PyTorch/XLA result omitted exported state")
    if bundle.state_dict:
        raise RuntimeError("PyTorch/XLA exported weights despite disabled export")
    bundle.state_dict = _state_dict_payloads(torch, exported_program)
    return program


def _finish_program_directory(directory: pathlib.Path) -> None:
    functions = directory / "functions"
    bytecode = functions / "forward.bytecode"
    portable = functions / "forward.stablehlo.bc"
    metadata = functions / "forward.meta"
    text = functions / "forward.mlir"
    if not bytecode.is_file() or not metadata.is_file():
        raise RuntimeError("PyTorch/XLA export omitted StableHLO or metadata")
    if portable.exists():
        raise RuntimeError("PyTorch/XLA export unexpectedly reused Wafer artifact name")
    os.replace(bytecode, portable)
    if text.exists():
        text.unlink()
    artifact = portable.read_bytes()
    if not artifact.startswith(b"ML\xefR") or b"StableHLO_v" not in artifact[:64]:
        raise RuntimeError("PyTorch/XLA bytecode is not a StableHLO portable artifact")
    with metadata.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("name") != "forward":
        raise RuntimeError("PyTorch/XLA export did not produce the forward entry")


def export_pytorch_program(
    module: Any, example_inputs: Any, output_directory: os.PathLike[str] | str
) -> pathlib.Path:
    """Export one static PyTorch module to the current Wafer source directory.

    The destination must not already exist. Target, search, runtime, oracle,
    workload and seed policy are intentionally not arguments of this API.
    """
    try:
        import torch
        import torch_xla._internal.custom_kernel  # noqa: F401
        from torch_xla import stablehlo
    except ImportError as error:
        raise RuntimeError(
            "the pinned PyTorch/XLA exporter runtime is unavailable"
        ) from error
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if isinstance(example_inputs, tuple):
        inputs = example_inputs
    elif isinstance(example_inputs, list):
        inputs = tuple(example_inputs)
    else:
        inputs = (example_inputs,)
    if not inputs:
        raise ValueError("example_inputs must not be empty")
    destination = pathlib.Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        exported = torch.export.export(module, inputs, strict=True)
        if getattr(exported, "range_constraints", None):
            raise RuntimeError("dynamic PyTorch export constraints are unsupported")
        program = _export_stablehlo(torch, stablehlo, exported)

    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.wafer-export-",
                         dir=destination.parent)
    )
    try:
        program.save(str(temporary))
        _finish_program_directory(temporary)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = ["export_pytorch_program"]
