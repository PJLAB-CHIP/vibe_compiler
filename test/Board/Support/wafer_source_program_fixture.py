"""Shared construction and validation for current source-program fixtures."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import shutil
from collections.abc import Callable

import torch


PYTORCH_RANDOM_SEED = 20260803


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    mlir_dtype: str
    metadata_dtype: str

    @property
    def element_bytes(self) -> int:
        return {"f16": 2, "bf16": 2, "f32": 4}[self.mlir_dtype]


@dataclasses.dataclass(frozen=True)
class SourcePayloads:
    inputs: tuple[torch.Tensor, ...]
    expected_outputs: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class SourceProgramCase:
    key: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    module_factory: Callable[[], str]
    payload_factory: Callable[[], SourcePayloads]


def random_f16(shape: tuple[int, ...], stream: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        PYTORCH_RANDOM_SEED + stream
    )
    return torch.randn(shape, dtype=torch.float16, generator=generator)


def metadata(case: SourceProgramCase) -> dict[str, object]:
    return {
        "name": "forward",
        "stablehlo_version": "0.0.0",
        "input_signature": [
            {
                "shape": list(spec.shape),
                "dtype": spec.metadata_dtype,
                "dynamic_dims": [],
            }
            for spec in case.inputs
        ],
        "output_signature": [
            {
                "shape": list(spec.shape),
                "dtype": spec.metadata_dtype,
                "dynamic_dims": [],
            }
            for spec in case.outputs
        ],
        "input_locations": [
            {
                "type_": "input_arg",
                "position": index,
                "name": f"input_{index}",
            }
            for index in range(len(case.inputs))
        ],
        "unused_inputs": [],
    }


def validate_source(case: SourceProgramCase) -> None:
    module = case.module_factory()
    retired = (
        "mhlo.sharding",
        "@Sharding",
        "distributed_boundary",
        "execution-ranks",
        "logical_rank",
    )
    if any(marker in module for marker in retired):
        raise RuntimeError(f"{case.key}: source contains a retired interface")
    if "func.func @main" not in module or "stablehlo." not in module:
        raise RuntimeError(f"{case.key}: source program is incomplete")


def validate_payloads(case: SourceProgramCase, values: SourcePayloads) -> None:
    for tensors, specs, kind in (
        (values.inputs, case.inputs, "input"),
        (values.expected_outputs, case.outputs, "output"),
    ):
        if len(tensors) != len(specs):
            raise RuntimeError(f"{case.key}: {kind} count is invalid")
        for index, (tensor, spec) in enumerate(zip(tensors, specs, strict=True)):
            if tuple(tensor.shape) != spec.shape:
                raise RuntimeError(
                    f"{case.key}: {kind} {index} shape is invalid"
                )
            if tensor.numel() * tensor.element_size() != (
                math.prod(spec.shape) * spec.element_bytes
            ):
                raise RuntimeError(
                    f"{case.key}: {kind} {index} byte count is invalid"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"{case.key}: {kind} {index} contains a nonfinite value"
                )


def write_source(work_dir: pathlib.Path, case: SourceProgramCase) -> pathlib.Path:
    source = work_dir / "source-program"
    if source.exists():
        shutil.rmtree(source)
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(case.module_factory())
    (source / "functions" / "forward.meta").write_text(
        json.dumps(metadata(case), separators=(",", ":")) + "\n"
    )
    return source
