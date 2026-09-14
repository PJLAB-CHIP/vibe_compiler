#!/usr/bin/env python3
"""Product frontend API contract exercised with the pinned exporter."""

from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import subprocess

import torch

from wafer.frontend import (
    _decompose_batch_norm_inference,
    _decompose_composite_opmath,
    export_pytorch_program,
)


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


def check_batch_norm_precision(output_root: pathlib.Path) -> None:
    for dtype, extent in ((torch.float16, 1024), (torch.bfloat16, 1025),
                          (torch.float32, 1031)):
        for affine in (False, True):
            model = torch.nn.BatchNorm2d(4, affine=affine).to(dtype).eval()
            with torch.no_grad():
                model.running_mean.copy_(torch.tensor([.5, -.8, .2, .9]))
                model.running_var.copy_(torch.tensor([.05, .7, 1.3, 2.0]))
                if affine:
                    model.weight.copy_(torch.tensor([.7, -.4, 1.2, .2]))
                    model.bias.copy_(torch.tensor([.3, -.2, .4, .1]))
            value = torch.linspace(-2, 2, 24 * extent).reshape(2, 4, 3, extent).to(dtype)
            before = {name: tensor.detach().clone()
                      for name, tensor in model.state_dict().items()}
            exported = torch.export.export(model, (value,))
            decomposed = _decompose_batch_norm_inference(torch, exported)
            with torch.no_grad():
                torch.testing.assert_close(decomposed.module()(value), model(value))
            if any("batch_norm" in str(node.target) for node in decomposed.graph.nodes
                   if node.op == "call_function"):
                raise RuntimeError("inference decomposition left a BatchNorm call")
            directory = output_root / f"batch-norm-{dtype}-{affine}"
            export_pytorch_program(model, (value,), directory)
            text = subprocess.check_output([
                os.environ["WAFER_STABLEHLO_TRANSLATE"], "--deserialize",
                str(directory / "functions" / "forward.stablehlo.bc"),
            ], text=True)
            for operation in ("sqrt", "subtract", "multiply", "add"):
                lines = [line for line in text.splitlines()
                         if f"stablehlo.{operation} " in line]
                if not lines or any(not line.rstrip().endswith("xf32>") for line in lines):
                    raise RuntimeError(f"BatchNorm {operation} lost framework opmath")
            if "stablehlo.batch_norm" in text:
                raise RuntimeError("portable source lost inference decomposition")
            lowered = subprocess.check_output([
                "wafer-opt", "--wafer-lower-stablehlo-to-linalg",
            ], input=text, text=True)
            if "stablehlo." in lowered or "arith.divf" not in lowered:
                raise RuntimeError("BatchNorm source did not reach structured arithmetic")
            element = {torch.float16: "f16", torch.bfloat16: "bf16",
                       torch.float32: "f32"}[dtype]
            if f"tensor<2x4x3x{extent}x{element}>" not in lowered:
                raise RuntimeError("BatchNorm output shape/dtype changed")
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    # Exercise the two conditional ATen schemas as well as the module's
    # dedicated no-training schema, using the same nontrivial runtime values.
    for operation in (torch.ops.aten.native_batch_norm.default,
                      torch.ops.aten._native_batch_norm_legit.default):
        class NativeBatchNorm(torch.nn.Module):
            def forward(self, x, scale, bias, mean, variance):
                return operation(x, scale, bias, mean, variance, False, .1, 1e-5)[0]

        native = NativeBatchNorm()
        inputs = (value, model.weight, model.bias, model.running_mean, model.running_var)
        exported = torch.export.export(native, inputs)
        decomposed = _decompose_batch_norm_inference(torch, exported)
        if decomposed is exported:
            raise RuntimeError("conditional inference schema was not decomposed")
        with torch.no_grad():
            torch.testing.assert_close(decomposed.module()(*inputs), native(*inputs))

    # Training and ordinary primitive graphs do not enter this inference policy.
    value = torch.ones((2, 4, 3, 1024))
    for model in (torch.nn.BatchNorm2d(4).train(), StaticBranch()):
        exported = torch.export.export(model, (value,))
        if _decompose_batch_norm_inference(torch, exported) is not exported:
            raise RuntimeError("inference policy changed an unrelated graph")
    print("batch_norm_inference: numeric_cases=8 source_to_linalg=6 unrelated_unchanged=true")


def check_composite_precision(output_root: pathlib.Path) -> None:
    # The pinned CPU MKL erf first call is not reproducible with the default
    # worker pool (observed independently of export). Use one CPU worker for
    # this numerical oracle, including cold calls, and restore the test process.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for extent in (1024, 1025, 1031):
                for approximate in ("none", "tanh"):
                    model = torch.nn.GELU(approximate=approximate).eval()
                    value = torch.linspace(-6, 6, extent * 16).reshape(1, extent, 16).to(dtype)
                    exported = torch.export.export(model, (value,))
                    decomposed = _decompose_composite_opmath(torch, exported)
                    torch.testing.assert_close(decomposed.module()(value), model(value))
                    special = value.clone()
                    special.flatten()[:7] = torch.tensor(
                        [-float("inf"), float("inf"), float("nan"), -0., 0., -1e-6, 1e-6],
                        dtype=dtype,
                    )
                    # Pinned oneDNN GELU returns NaN for +Inf in BF16/F32,
                    # unlike PyTorch's ATen kernel and official decomposition.
                    # Qualify special values against ATen explicitly; finite
                    # acceptance above still uses the unchanged default backend.
                    with torch.backends.mkldnn.flags(enabled=False):
                        torch.testing.assert_close(decomposed.module()(special), model(special), equal_nan=True)
                    for node in decomposed.graph.nodes:
                        if node.op == "call_function" and node.target in (
                            torch.ops.aten.erf.default, torch.ops.aten.tanh.default,
                            torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor,
                        ) and node.meta["val"].dtype != torch.float32:
                            raise RuntimeError("GELU lost framework opmath")
                    directory = output_root / f"composite-gelu-{dtype}-{extent}-{approximate}"
                    export_pytorch_program(model, (value,), directory)
                    text = subprocess.check_output([
                        os.environ["WAFER_STABLEHLO_TRANSLATE"], "--deserialize",
                        str(directory / "functions" / "forward.stablehlo.bc"),
                    ], text=True)
                    nonlinear = [line for line in text.splitlines()
                                 if "@mhlo.erf(" in line or "stablehlo.tanh " in line]
                    if len(nonlinear) != 1 or not nonlinear[0].rstrip().endswith("xf32>"):
                        raise RuntimeError("GELU portable source lost its opmath/approximation")
                    if ("@mhlo.erf(" in text) != (approximate == "none"):
                        raise RuntimeError("GELU approximation choice changed")

        for dtype, extent in ((torch.float16, 1024), (torch.bfloat16, 1025),
                              (torch.float32, 1031)):
            for multiple_axes in (False, True):
                for affine in (False, True):
                    normalized = (8, extent) if multiple_axes else (extent,)

                    class NativeLayerNorm(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.norm = torch.nn.LayerNorm(normalized, eps=1e-5,
                                                           elementwise_affine=affine).to(dtype)
                            if affine:
                                with torch.no_grad():
                                    self.norm.weight.copy_(torch.linspace(.7, 1.3, self.norm.weight.numel())
                                                           .reshape(normalized))
                                    self.norm.bias.copy_(torch.linspace(-.2, .1, self.norm.bias.numel())
                                                         .reshape(normalized))

                        def forward(self, value):
                            return torch.ops.aten.native_layer_norm.default(
                                value, self.norm.normalized_shape, self.norm.weight,
                                self.norm.bias, self.norm.eps)

                    model = NativeLayerNorm().eval()
                    value = (torch.arange(8 * extent).reshape(1, 8, extent).float().sin() + .7).to(dtype)
                    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                    exported = torch.export.export(model, (value,))
                    decomposed = _decompose_composite_opmath(torch, exported)
                    with torch.no_grad():
                        torch.testing.assert_close(decomposed.module()(value), model(value))
                    if any(node.op == "call_function" and node.target == torch.ops.aten.native_layer_norm.default
                           for node in decomposed.graph.nodes):
                        raise RuntimeError("native LayerNorm was not decomposed")
                    directory = output_root / f"composite-layer-norm-{dtype}-{multiple_axes}-{affine}"
                    export_pytorch_program(model, (value,), directory)
                    text = subprocess.check_output([
                        os.environ["WAFER_STABLEHLO_TRANSLATE"], "--deserialize",
                        str(directory / "functions" / "forward.stablehlo.bc"),
                    ], text=True)
                    if "stablehlo.batch_norm" in text or "stablehlo.custom_call" in text:
                        raise RuntimeError("LayerNorm source left opaque/statistics operations")
                    lowered = subprocess.check_output([
                        "wafer-opt", "--wafer-lower-stablehlo-to-linalg",
                    ], input=text, text=True)
                    if "stablehlo." in lowered or "linalg.generic" not in lowered:
                        raise RuntimeError("LayerNorm did not reach structured arithmetic")
                    for name, tensor in model.state_dict().items():
                        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

        # The public LayerNorm module exposes the outer schema and only output;
        # qualify that detection independently of the three-result native schema.
        value = torch.ones((1, 1025, 16), dtype=torch.float16)
        model = torch.nn.LayerNorm(16).half().eval()
        exported = torch.export.export(model, (value,))
        torch.testing.assert_close(_decompose_composite_opmath(torch, exported).module()(value), model(value))
        unrelated = torch.export.export(StaticBranch(), (value,))
        if _decompose_composite_opmath(torch, unrelated) is not unrelated:
            raise RuntimeError("composite policy changed unrelated primitive IR")
    finally:
        torch.set_num_threads(threads)
    print("composite_opmath: gelu=18 layer_norm=13 source_layer_norm_to_linalg=12 unrelated_unchanged=true")


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
    check_batch_norm_precision(args.output_root)
    check_composite_precision(args.output_root)
    try:
        export_pytorch_program(DataDependentGraphBreak(), (value,), args.output_root / "bad")
    except Exception:
        pass
    else:
        raise RuntimeError("data-dependent graph break was accepted")
    print("wafer_frontend_product: portable=true repeat_equal=true graph_break=rejected")


if __name__ == "__main__":
    main()
