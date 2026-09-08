#!/usr/bin/env python3
"""Product frontend API contract exercised with the pinned exporter."""

from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import subprocess

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


class BiasedConvolution(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, *, separate_bias: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((3, 2, 3, 3), 0.03125, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.full((3,), 0.015625, dtype=dtype))
        self.separate_bias = separate_bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.separate_bias:
            convolved = torch.nn.functional.conv2d(value, self.weight, padding=1)
            return convolved + self.bias[None, :, None, None]
        return torch.nn.functional.conv2d(value, self.weight, self.bias, padding=1)


def check_convolution_precision(output_root: pathlib.Path) -> None:
    for dtype, extent in ((torch.float16, 1024), (torch.bfloat16, 1025),
                          (torch.float32, 1031)):
        for separate in (False, True):
            model = BiasedConvolution(dtype, separate_bias=separate).eval()
            value = torch.linspace(-1, 1, 8 * extent).reshape(1, 2, 4, extent).to(dtype)
            before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            directory = output_root / f"conv-{dtype}-{separate}"
            export_pytorch_program(model, (value,), directory)
            text = subprocess.check_output([
                os.environ["WAFER_STABLEHLO_TRANSLATE"], "--deserialize",
                str(directory / "functions" / "forward.stablehlo.bc"),
            ], text=True)
            convolution, = [line for line in text.splitlines()
                            if "stablehlo.convolution" in line]
            addition, = [line for line in text.splitlines() if "stablehlo.add" in line]
            element = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}[dtype]
            compute = element if separate else "f32"
            for line in (convolution, addition):
                if not line.rstrip().endswith(f"x{compute}>"):
                    raise RuntimeError(f"incorrect convolution/bias compute dtype: {line}")
            if not separate and element != "f32":
                # No low-precision intermediate is allowed between the two ops.
                between = text[text.index(convolution):text.index(addition)]
                if f"-> tensor<1x3x4x{extent}x{element}>" in between:
                    raise RuntimeError("biased convolution rounded before adding bias")
                if f"-> tensor<1x3x4x{extent}x{element}>" not in text[text.index(addition):]:
                    raise RuntimeError("convolution result did not restore its dtype")
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
            if model(value).dtype != dtype:
                raise RuntimeError("export modified the original module")


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
    check_convolution_precision(args.output_root)
    try:
        export_pytorch_program(DataDependentGraphBreak(), (value,), args.output_root / "bad")
    except Exception:
        pass
    else:
        raise RuntimeError("data-dependent graph break was accepted")
    print("wafer_frontend_product: portable=true repeat_equal=true graph_break=rejected")


if __name__ == "__main__":
    main()
