#!/usr/bin/env python3
"""Framework-owned GEMM and HuggingFace Transformer board source cases."""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import shutil
import sys
from collections.abc import Callable

import torch

import wafer_pytorch_board_common as common


TOOLS_INPUTS = pathlib.Path(__file__).resolve().parents[2] / "Tools" / "Inputs"
if str(TOOLS_INPUTS) not in sys.path:
    sys.path.insert(0, str(TOOLS_INPUTS))

import wafer_pytorch_xla_capture as capture  # noqa: E402


RANK_COUNT = 16
LARGE_GEMM_EXTENT = 4096
HF_LLAMA2_7B_CONFIG = (
    TOOLS_INPUTS / "hf" / "llama-2-7b-block-config.json"
)
HF_LLAMA2_7B_SEQUENCE_LENGTH = 16
HF_LLAMA2_7B_COMPARISON = common.ComparisonPolicy(rtol=0.002, atol=0.004)


@dataclasses.dataclass(frozen=True)
class PyTorchBoardCase:
    name: str
    rank_count: int
    dtype: torch.dtype
    inputs: tuple[torch.Tensor, ...]
    expected_outputs: tuple[torch.Tensor, ...]
    export_program: Callable[[pathlib.Path], None]
    required_structured_ir: tuple[str, ...]
    expected_all_reduce_count: int
    comparison_policy: common.ComparisonPolicy


class Gemm(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)


def parse_torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unknown torch dtype: {name}")
    return dtype


def _random_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, generator=generator)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, generator=generator)
    return torch.randint(-4, 5, shape, dtype=dtype, generator=generator)


def _stablehlo_export_options(stablehlo_module: object) -> object:
    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True
    return options


def _save_exported_program(
    program_dir: pathlib.Path,
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    torch_module, stablehlo_module = capture._import_runtime_modules()
    module.eval()
    options = _stablehlo_export_options(stablehlo_module)
    with torch_module.no_grad():
        exported = torch_module.export.export(module, inputs)
        program = stablehlo_module.exported_program_to_stablehlo(
            exported, options=options
        )
    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(program_dir))
    capture._verify_program_dir_layout(program_dir)


def _rank_one_gemm(dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    m, k, n = 256, 256, 512
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lhs = _random_tensor((m, k), dtype=dtype, generator=generator)
    rhs = _random_tensor((k, n), dtype=dtype, generator=generator)
    module = Gemm().eval()
    with torch.no_grad():
        expected = module(lhs, rhs)
    return PyTorchBoardCase(
        name="rank-one-gemm",
        rank_count=1,
        dtype=dtype,
        inputs=(lhs, rhs),
        expected_outputs=(expected,),
        export_program=lambda output: _save_exported_program(
            output, module, (lhs, rhs)
        ),
        required_structured_ir=(
            "linalg.matmul",
        ),
        expected_all_reduce_count=0,
        comparison_policy=common.PYTORCH_DEFAULT,
    )


def _k_sharded_gemm_all_reduce(
    dtype: torch.dtype, seed: int
) -> PyTorchBoardCase:
    size = LARGE_GEMM_EXTENT
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_tensor = _random_tensor(
        (size, size), dtype=dtype, generator=generator
    )
    weight = _random_tensor((size, size), dtype=dtype, generator=generator)

    class KShardedGemm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                weight.clone(), requires_grad=False
            )
            self.bias = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.matmul(value, self.weight)

    def module_factory() -> torch.nn.Module:
        return KShardedGemm()

    eager_module = module_factory().eval()
    with torch.no_grad():
        expected = eager_module(input_tensor)

    def export_program(output: pathlib.Path) -> None:
        capture.emit_sharded_stablehlo_program(
            output,
            strategy_name="row",
            reference_module_factory=module_factory,
            example_input_tensor=input_tensor,
            size=size,
        )

    return PyTorchBoardCase(
        name="k-sharded-gemm-all-reduce",
        rank_count=RANK_COUNT,
        dtype=dtype,
        inputs=(input_tensor,),
        expected_outputs=(expected,),
        export_program=export_program,
        required_structured_ir=(
            "linalg.matmul",
            "wafer.linalg_ext.collective.all_reduce",
        ),
        expected_all_reduce_count=1,
        comparison_policy=common.PYTORCH_DEFAULT,
    )


def _hf_megatron_transformer_block(
    dtype: torch.dtype, seed: int
) -> PyTorchBoardCase:
    dtype_names = {
        torch.float16: "float16",
        torch.float32: "float32",
    }
    try:
        dtype_name = dtype_names[dtype]
    except KeyError as error:
        raise RuntimeError(
            "HuggingFace Llama board case supports float16 or float32"
        ) from error

    config = dict(capture.load_hf_transformer_config(HF_LLAMA2_7B_CONFIG))
    config["torch_dtype"] = dtype_name
    batch_size = 1
    sequence_length = HF_LLAMA2_7B_SEQUENCE_LENGTH
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    if (
        num_attention_heads % RANK_COUNT != 0
        or hidden_size % RANK_COUNT != 0
        or intermediate_size % RANK_COUNT != 0
    ):
        raise RuntimeError(
            "HuggingFace Megatron heads/hidden/intermediate dimensions must "
            "divide the TP16 mesh"
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_tensor = _random_tensor(
        (batch_size, sequence_length, hidden_size),
        dtype=dtype,
        generator=generator,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        module = capture._make_hf_llama_decoder_block_module(
            torch,
            config,
            sequence_length,
        )
    module.eval()
    parameter_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        for parameter, sharding_spec in module.wafer_parameter_sharding_specs():
            values = _random_tensor(
                tuple(parameter.shape),
                dtype=dtype,
                generator=parameter_generator,
            )
            if sharding_spec == capture.HF_MEGATRON_REPLICATED_VECTOR_SPEC:
                values = 1.0 + (values * 0.02)
            else:
                values = values * float(config["initializer_range"])
            parameter.copy_(values)
            parameter.requires_grad_(False)
            if parameter.dtype != dtype:
                raise RuntimeError(
                    "HuggingFace Megatron parameter dtype differs from the case"
                )
    with torch.no_grad():
        expected = module(input_tensor)
    if (
        not torch.isfinite(input_tensor).all()
        or not torch.isfinite(expected).all()
    ):
        raise RuntimeError(
            "HuggingFace Megatron random input/reference must be finite"
        )

    def export_program(output: pathlib.Path) -> None:
        resolved_config = output.parent / "hf-megatron-tp16-config.json"
        resolved_config.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        capture.emit_hf_megatron_transformer_block_program(
            output,
            config_path=resolved_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            reference_module_factory=lambda: copy.deepcopy(module),
            example_input_tensor=input_tensor,
        )

    return PyTorchBoardCase(
        name="hf-megatron-transformer-block",
        rank_count=RANK_COUNT,
        dtype=dtype,
        inputs=(input_tensor,),
        expected_outputs=(expected,),
        export_program=export_program,
        required_structured_ir=(
            "linalg.matmul",
            "wafer.linalg_ext.collective.all_reduce",
        ),
        expected_all_reduce_count=2,
        comparison_policy=HF_LLAMA2_7B_COMPARISON,
    )


CASE_FACTORIES: dict[
    str, Callable[[torch.dtype, int], PyTorchBoardCase]
] = {
    "rank-one-gemm": _rank_one_gemm,
    "k-sharded-gemm-all-reduce": _k_sharded_gemm_all_reduce,
    "hf-megatron-transformer-block": _hf_megatron_transformer_block,
}


def make_case(name: str, *, dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    try:
        case = CASE_FACTORIES[name](dtype, seed)
    except KeyError as error:
        raise RuntimeError(f"unknown PyTorch board case: {name}") from error
    for tensor in (*case.inputs, *case.expected_outputs):
        if tensor.device.type != "cpu" or tensor.dtype != dtype:
            raise RuntimeError(
                f"PyTorch board case {name} must preserve CPU dtype {dtype}"
            )
    return case
