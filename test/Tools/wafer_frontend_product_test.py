#!/usr/bin/env python3
"""Product frontend API contract exercised with the pinned exporter."""

from __future__ import annotations

import argparse
import inspect
import pathlib

import torch

from wafer.frontend import export_pytorch_program


class StaticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(64, dtype=torch.float16).reshape(8, 8))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.weight


class StaticBranch(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value + value, value - value


class DataDependentGraphBreak(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.sum().item() > 0:
            return value + value
        return value - value


def snapshot(directory: pathlib.Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    first = args.output_root / "program"
    second = args.output_root / "repeat"
    branched = args.output_root / "branched"
    value = torch.arange(64, dtype=torch.float16).reshape(8, 8)
    export_pytorch_program(StaticModel().eval(), (value,), first)
    export_pytorch_program(StaticModel().eval(), (value,), second)
    export_pytorch_program(StaticBranch().eval(), (value,), branched)
    if snapshot(first) != snapshot(second):
        raise RuntimeError("repeated product exports are not byte-equivalent")
    if not (first / "functions" / "forward.stablehlo.bc").is_file():
        raise RuntimeError("portable StableHLO artifact is missing")
    if (first / "functions" / "forward.mlir").exists():
        raise RuntimeError("diagnostic text leaked into the program directory")
    if not (branched / "functions" / "forward.stablehlo.bc").is_file():
        raise RuntimeError("branched static export is missing")
    parameters = inspect.signature(export_pytorch_program).parameters
    if tuple(parameters) != ("module", "example_inputs", "output_directory"):
        raise RuntimeError("product frontend API exposes non-frontend policy")
    try:
        export_pytorch_program(DataDependentGraphBreak(), (value,), args.output_root / "bad")
    except Exception:
        pass
    else:
        raise RuntimeError("data-dependent graph break was accepted")
    print("wafer_frontend_product: portable=true repeat_equal=true graph_break=rejected")


if __name__ == "__main__":
    main()
