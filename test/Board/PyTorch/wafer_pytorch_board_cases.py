#!/usr/bin/env python3
"""Framework-owned GEMM and HuggingFace Transformer board source cases."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
from collections.abc import Callable

import torch

import wafer_pytorch_board_common as common


TOOLS_INPUTS = pathlib.Path(__file__).resolve().parents[2] / "Tools" / "Inputs"
if str(TOOLS_INPUTS) not in sys.path:
    sys.path.insert(0, str(TOOLS_INPUTS))

import wafer_pytorch_xla_capture as capture  # noqa: E402


HF_LLAMA2_7B_CONFIG = (
    TOOLS_INPUTS / "hf" / "llama-2-7b-block-config.json"
)
HF_LLAMA2_7B_SEQUENCE_LENGTH = 16
HF_LLAMA2_7B_COMPARISON = common.ComparisonPolicy(rtol=0.002, atol=0.004)
ATTENTION_COMPARISON = common.ComparisonPolicy(rtol=0.006, atol=0.008)
ATTENTION_HEAD_DIM = 64
OPTIMIZATION_POLICIES = ("search", "none")
SOURCE_NO_CARD_OPTIMIZATION_POLICIES = OPTIMIZATION_POLICIES


@dataclasses.dataclass(frozen=True)
class PyTorchBoardCase:
    name: str
    num_partitions: int
    dtype: torch.dtype
    inputs: tuple[torch.Tensor, ...]
    expected_outputs_factory: Callable[[], tuple[torch.Tensor, ...]]
    export_program: Callable[[pathlib.Path], None]
    comparison_policy: common.ComparisonPolicy
    widened_convolution: bool = False
    ordered_convolution: bool = False
    allgather_payload_elements: int | None = None
    gemm_dimensions: tuple[int, int, int] | None = None
    alltoall_extent: int | None = None
    reduce_scatter_extent: int | None = None
    all_reduce_extent: int | None = None
    prefill_extent: int | None = None
    continuation_factory: (
        Callable[[tuple[torch.Tensor, ...]], "PyTorchBoardCase"] | None
    ) = None

    def materialize_expected_outputs(self) -> tuple[torch.Tensor, ...]:
        outputs = self.expected_outputs_factory()
        if not outputs:
            raise RuntimeError(f"PyTorch board case {self.name} has no outputs")
        for tensor in outputs:
            if tensor.device.type != "cpu" or tensor.dtype != self.dtype:
                raise RuntimeError(
                    f"PyTorch board case {self.name} eager oracle must preserve "
                    f"CPU dtype {self.dtype}"
                )
        return outputs


@dataclasses.dataclass(frozen=True)
class SourceNoCardWorkload:
    """One retained source/oracle/package/no-card qualification case."""

    ctest_name: str
    case_name: str
    dtype_name: str
    num_partitions: int
    optimization_policy: str
    package_count: int = 1


PRODUCTION_SOURCE_CASES = (
    "conv-mixed-dag",
    "attention-prefill",
    "attention-decode-kv-cache",
    "llama-2-7b-block",
)
PRODUCTION_SOURCE_DTYPES = (
    ("float16", "fp16"),
    ("bfloat16", "bf16"),
)
SOURCE_NO_CARD_WORKLOADS = tuple(
    SourceNoCardWorkload(
        ctest_name=(
            f"wafer-runtime-pytorch-{case_name}-{dtype_label}-"
            f"optimization-{optimization_policy}-no-card"
        ),
        case_name=case_name,
        dtype_name=dtype_name,
        num_partitions=1,
        optimization_policy=optimization_policy,
        package_count=(
            2 if case_name == "attention-decode-kv-cache" else 1
        ),
    )
    for case_name in PRODUCTION_SOURCE_CASES
    for dtype_name, dtype_label in PRODUCTION_SOURCE_DTYPES
    for optimization_policy in SOURCE_NO_CARD_OPTIMIZATION_POLICIES
)


class Gemm(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)


class CompleteTileAdd(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + rhs


class AllGatherAdd(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + (rhs + rhs).unsqueeze(0)


def make_allgather_add(
    dtype: torch.dtype, seed: int, *, extent: int = 1024
) -> PyTorchBoardCase:
    """Ordinary broadcast dataflow requires every computed rhs shard."""
    module = AllGatherAdd().eval()
    lanes = torch.arange(extent, dtype=torch.int32)
    tiles = torch.arange(16, dtype=torch.int32)
    rhs = (tiles[:, None, None] * 8 + (lanes + seed) % 8).to(dtype)
    lhs = (
        tiles[:, None, None, None] * 4
        + tiles[None, :, None, None]
        + lanes[None, None, None, :] % 4
    ).to(dtype)
    inputs = (lhs, rhs)

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(*inputs),)

    return PyTorchBoardCase(
        name=f"allgather-add-{extent}",
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, inputs
        ),
        comparison_policy=common.ComparisonPolicy(rtol=1e-3, atol=1e-5),
        allgather_payload_elements=extent,
    )


def make_launch_case(
    name: str, *, local_elements: int | None = None
) -> PyTorchBoardCase:
    """Large launch ABI inputs; source and oracle share one PyTorch module."""
    if local_elements is None:
        local_elements = 458752
    if local_elements <= 0:
        raise ValueError("launch case requires positive local elements")
    tile_count = 16
    lanes = torch.arange(local_elements, dtype=torch.int32)
    tiles = torch.arange(tile_count, dtype=torch.int32).unsqueeze(1)
    if name == "complete-tile-add":
        module = CompleteTileAdd().eval()
        inputs = (
            (tiles * 32 + lanes % 32).flatten().to(torch.float16),
            (512 + tiles * 16 + lanes % 16).flatten().to(torch.float16),
        )
    else:
        raise ValueError(f"unknown launch case: {name}")

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(*inputs),)

    return PyTorchBoardCase(
        name=name,
        num_partitions=1,
        dtype=torch.float16,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(output, module, inputs),
        comparison_policy=common.ComparisonPolicy(rtol=1e-3, atol=1e-5),
    )


class HeterogeneousTilingDataflow(torch.nn.Module):
    """Mixed-shape structured dataflow with no compiler-facing markers."""

    def forward(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.matmul(lhs, rhs)
        activated = torch.relu(projected + bias)
        mixed = activated * (projected - bias)
        return mixed, torch.sum(mixed, dim=1)


class LocalReduce(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value.sum(dim=3), value.sum(dim=(2, 3))


def make_local_reduce(
    dtype: torch.dtype, seed: int, *, extent: int = 1024
) -> PyTorchBoardCase:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randint(
        -4, 5, (1, 24, 8, extent), generator=generator
    ).to(dtype) / 64
    module = LocalReduce().eval()
    inputs = (value,)

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return module(*inputs)

    return PyTorchBoardCase(
        name=f"local-reduce-{extent}",
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(output, module, inputs),
        comparison_policy=common.PYTORCH_DEFAULT,
    )


class LocalConv(torch.nn.Module):
    def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(value, weight, padding=(1, 1))


def make_local_conv(
    dtype: torch.dtype, seed: int, *, extent: int = 1024,
    kernel: tuple[int, int] = (3, 3),
) -> PyTorchBoardCase:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Exact binary fractions make swapped kernel axes and physical padding
    # visible independently of accumulation rounding. The mixed DAG retains
    # its original random floating-point inputs as the numerical regression.
    value = torch.randint(
        -4, 5, (1, 16, 8, extent), generator=generator
    ).to(dtype) / 8
    weight = torch.randint(
        -4, 5, (24, 16, *kernel), generator=generator
    ).to(dtype) / 16
    module = LocalConv().eval()
    inputs = (value, weight)

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(*inputs),)

    return PyTorchBoardCase(
        name=f"local-conv-{kernel[0]}x{kernel[1]}-{extent}",
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(output, module, inputs),
        comparison_policy=common.PYTORCH_DEFAULT,
    )


class BiasedConv(torch.nn.Module):
    def forward(self, value, weight, bias):
        return torch.nn.functional.conv2d(value, weight, bias, padding=1)


class Sigmoid(torch.nn.Module):
    def forward(self, value):
        return torch.sigmoid(value)


def make_biased_conv(dtype: torch.dtype, seed: int, *, extent: int = 1024):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = (
        _random_tensor((1, 16, 8, extent), dtype=dtype, generator=generator) * 0.125,
        _random_tensor((24, 16, 3, 3), dtype=dtype, generator=generator) * 0.03125,
        _random_tensor((24,), dtype=dtype, generator=generator) * 0.015625,
    )
    module = BiasedConv().eval()
    return PyTorchBoardCase(
        name=f"biased-conv-{extent}", num_partitions=1, dtype=dtype, inputs=inputs,
        widened_convolution=dtype == torch.bfloat16,
        ordered_convolution=dtype == torch.float16,
        expected_outputs_factory=lambda: (module(*inputs),),
        export_program=lambda out: _save_exported_program(out, module, inputs),
        comparison_policy=common.PYTORCH_DEFAULT,
    )


def make_sigmoid(dtype: torch.dtype, seed: int, *, extent: int = 1024):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((2, 4, extent), generator=generator).to(dtype) * 4
    # Saturation and signed zero complement the random positive/negative domain.
    value.flatten()[:8] = torch.tensor(
        [-65504, -20, -0.001, -0.0, 0.0, 0.001, 20, 65504], dtype=dtype
    )
    module = Sigmoid().eval()
    inputs = (value,)
    return PyTorchBoardCase(
        name=f"sigmoid-{extent}", num_partitions=1, dtype=dtype, inputs=inputs,
        expected_outputs_factory=lambda: (module(*inputs),),
        export_program=lambda out: _save_exported_program(out, module, inputs),
        comparison_policy=common.PYTORCH_DEFAULT,
    )


class Division(torch.nn.Module):
    def forward(self, lhs, rhs):
        return lhs / rhs


def make_division(dtype: torch.dtype, seed: int, *, extent: int = 1024):
    if dtype != torch.float32:
        raise RuntimeError("division refinement qualification requires F32")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lhs = torch.randn((2, 4, extent), generator=generator) * 3
    rhs = torch.randn((2, 4, extent), generator=generator) * 2
    # Exercise the semantic division boundary, including native-result lanes.
    lhs.flatten()[:12] = torch.tensor(
        [0, -0.0, 0, -0.0, 1, -1, torch.inf, -torch.inf, 0, torch.inf, torch.nan, 1]
    )
    rhs.flatten()[:12] = torch.tensor(
        [1, 1, -1, -1, torch.inf, torch.inf, 1, 1, 0, torch.inf, 1, torch.nan]
    )
    module = Division().eval()
    inputs = (lhs, rhs)
    return PyTorchBoardCase(
        name=f"division-{extent}", num_partitions=1, dtype=dtype, inputs=inputs,
        expected_outputs_factory=lambda: (module(*inputs),),
        export_program=lambda out: _save_exported_program(out, module, inputs),
        comparison_policy=common.ComparisonPolicy(equal_nan=True),
    )


class ConvMixedDataflow(torch.nn.Module):
    """Convolution feeding branch, fanin, fanout, and reduction dataflow."""

    def forward(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        convolved = torch.nn.functional.conv2d(
            input_tensor, weight, bias, stride=1, padding=1
        )
        positive = torch.relu(convolved)
        gated = torch.sigmoid(convolved) * positive
        joined = gated + convolved
        return joined, torch.sum(joined, dim=(2, 3))


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
        program = capture.exported_program_to_stablehlo(
            torch_module,
            stablehlo_module,
            exported,
            options=options,
        )
    capture._save_program(program, program_dir, options)
    # Some StableHLO serializer versions omit non-parameter ExportedProgram
    # state (for example BF16 rotary and causal-mask buffers) while still
    # emitting parameter metadata for it.  The source program must be
    # self-contained, so materialize any missing state payload from the same
    # exported snapshot before validating the directory.  Existing serializer
    # output remains authoritative and is not rewritten.
    required_parameters: set[str] = set()
    for metadata_path in (program_dir / "functions").glob("*.meta"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for location in metadata.get("input_locations", []):
            if location.get("type_") == "parameter":
                required_parameters.add(location.get("name", ""))
    state = dict(module.state_dict())
    state.update(dict(module.named_buffers()))
    numpy_module = capture._import_numpy()
    for name in sorted(required_parameters):
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise RuntimeError("exported state name is not a safe data filename")
        payload = program_dir / "data" / name
        if payload.exists():
            continue
        tensor = state.get(name)
        if tensor is None:
            raise RuntimeError(f"exported parameter has no source state: {name}")
        payload.parent.mkdir(parents=True, exist_ok=True)
        value = capture._torch_tensor_to_workload_storage(torch_module, tensor)
        with payload.open("wb") as stream:
            numpy_module.save(stream, value, allow_pickle=False)
    capture._verify_program_dir_layout(program_dir)


def make_reduce_scatter_sum(
    dtype: torch.dtype, seed: int, *, extent: int = 1024
) -> PyTorchBoardCase:
    if dtype != torch.float16:
        raise RuntimeError("ReduceScatter qualification currently requires FP16")

    class Reduction(torch.nn.Module):
        def forward(self, rhs: torch.Tensor) -> torch.Tensor:
            return (rhs + rhs).sum(dim=0, keepdim=True)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (16, extent, 1)
    # Exact binary contributions isolate missing/duplicated transport and merge
    # values while retaining a full-size PyTorch sum with both signs.
    rhs = torch.randint(1, 9, shape, generator=generator).to(dtype) / 16
    rhs *= torch.randint(0, 2, shape, generator=generator).to(dtype) * 2 - 1
    module = Reduction().eval()

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(rhs),)

    return PyTorchBoardCase(
        name=f"reduce-scatter-sum-l{extent}",
        num_partitions=1,
        dtype=dtype,
        inputs=(rhs,),
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(output, module, (rhs,)),
        comparison_policy=common.PYTORCH_DEFAULT,
        reduce_scatter_extent=extent,
    )


def make_all_reduce_sum(
    dtype: torch.dtype, seed: int, *, extent: int = 1024
) -> PyTorchBoardCase:
    if dtype != torch.float16:
        raise RuntimeError("AllReduce qualification currently requires FP16")

    class Reduction(torch.nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs + (rhs + rhs).sum(dim=0, keepdim=True)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (16, extent, 1)
    rhs = torch.randint(1, 9, shape, generator=generator).to(dtype) / 16
    rhs *= torch.randint(0, 2, shape, generator=generator).to(dtype) * 2 - 1
    lhs = torch.randint(-8, 9, shape, generator=generator).to(dtype) / 16
    module = Reduction().eval()

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(lhs, rhs),)

    return PyTorchBoardCase(
        name=f"all-reduce-sum-l{extent}",
        num_partitions=1,
        dtype=dtype,
        inputs=(lhs, rhs),
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(output, module, (lhs, rhs)),
        comparison_policy=common.PYTORCH_DEFAULT,
        all_reduce_extent=extent,
    )


def make_alltoall_transpose(
    dtype: torch.dtype, seed: int, *, extent: int = 1024
) -> PyTorchBoardCase:
    if dtype != torch.float16:
        raise RuntimeError("AllToAll qualification currently requires FP16")

    class Exchange(torch.nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs + (rhs + rhs).transpose(0, 1)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = (
        _random_tensor((extent, 16, 1), dtype=dtype, generator=generator),
        _random_tensor((16, extent, 1), dtype=dtype, generator=generator),
    )
    module = Exchange().eval()

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(*inputs),)

    return PyTorchBoardCase(
        name=f"alltoall-transpose-l{extent}",
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(output, module, inputs),
        comparison_policy=common.PYTORCH_DEFAULT,
        alltoall_extent=extent,
    )


def _single_card_gemm(
    dtype: torch.dtype, seed: int, *,
    m: int = 1024, k: int = 256, n: int = 512,
) -> PyTorchBoardCase:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lhs = _random_tensor((1, m, k), dtype=dtype, generator=generator)
    rhs = _random_tensor((1, k, n), dtype=dtype, generator=generator)
    module = Gemm().eval()

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(lhs, rhs),)

    return PyTorchBoardCase(
        name=f"single-card-gemm-m{m}-k{k}-n{n}",
        num_partitions=1,
        dtype=dtype,
        inputs=(lhs, rhs),
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, (lhs, rhs)
        ),
        comparison_policy=common.PYTORCH_DEFAULT,
        gemm_dimensions=(m, k, n),
    )


def _heterogeneous_tiling_dataflow(
    dtype: torch.dtype,
    seed: int,
    *,
    name: str,
    num_partitions: int,
) -> PyTorchBoardCase:
    if dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "heterogeneous tiling dataflow requires float16 or bfloat16"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lhs = _random_tensor(
        (96, 64), dtype=dtype, generator=generator
    ) * 0.125
    rhs = _random_tensor(
        (64, 80), dtype=dtype, generator=generator
    ) * 0.125
    bias = _random_tensor((80,), dtype=dtype, generator=generator) * 0.0625
    module = HeterogeneousTilingDataflow().eval()
    inputs = (lhs, rhs, bias)

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            outputs = module(*inputs)
        if any(output.dtype != dtype for output in outputs) or any(
            not torch.isfinite(output).all() for output in outputs
        ):
            raise RuntimeError(
                "heterogeneous tiling eager reference must be finite and "
                "preserve dtype"
            )
        return outputs

    return PyTorchBoardCase(
        name=name,
        num_partitions=num_partitions,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, inputs
        ),
        comparison_policy=common.PYTORCH_DEFAULT,
    )


def _heterogeneous_tiling_single_card(
    dtype: torch.dtype, seed: int
) -> PyTorchBoardCase:
    return _heterogeneous_tiling_dataflow(
        dtype,
        seed,
        name="heterogeneous-tiling-dataflow",
        num_partitions=1,
    )


def _conv_mixed_dag(dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    if dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("conv mixed DAG requires float16 or bfloat16")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    spatial_width = 1024 if dtype == torch.float16 else 1025
    input_tensor = _random_tensor(
        (1, 16, 8, spatial_width), dtype=dtype, generator=generator
    ) * 0.125
    weight = _random_tensor(
        (24, 16, 3, 3), dtype=dtype, generator=generator
    ) * 0.03125
    bias = _random_tensor((24,), dtype=dtype, generator=generator) * 0.015625
    module = ConvMixedDataflow().eval()
    inputs = (input_tensor, weight, bias)

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            outputs = module(*inputs)
        if any(output.dtype != dtype for output in outputs) or any(
            not torch.isfinite(output).all() for output in outputs
        ):
            raise RuntimeError(
                "conv mixed DAG eager reference must be finite and preserve dtype"
            )
        return outputs

    return PyTorchBoardCase(
        name="conv-mixed-dag",
        widened_convolution=dtype == torch.bfloat16,
        ordered_convolution=dtype == torch.float16,
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, inputs
        ),
        comparison_policy=common.PYTORCH_DEFAULT,
    )


def _llama_2_7b_block(dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    dtype_names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    try:
        dtype_name = dtype_names[dtype]
    except KeyError as error:
        raise RuntimeError(
            "Hugging Face Llama block requires float16 or bfloat16"
        ) from error

    config = dict(capture.load_hf_transformer_config(HF_LLAMA2_7B_CONFIG))
    config["torch_dtype"] = dtype_name
    sequence_length = HF_LLAMA2_7B_SEQUENCE_LENGTH
    hidden_size = int(config["hidden_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = _random_tensor(
        (1, sequence_length, hidden_size),
        dtype=dtype,
        generator=generator,
    ) * 0.125
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        module = capture._make_hf_llama_decoder_block_module(
            torch, config, sequence_length
        )
    module.eval()
    inputs = (hidden_states,)

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            output = module(*inputs)
        if output.dtype != dtype or not torch.isfinite(output).all():
            raise RuntimeError(
                "Hugging Face Llama block eager reference must be finite "
                "and preserve dtype"
            )
        return (output,)

    return PyTorchBoardCase(
        name="llama-2-7b-block",
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, inputs
        ),
        comparison_policy=HF_LLAMA2_7B_COMPARISON,
    )


def _read_only_attention(
    dtype: torch.dtype,
    seed: int,
    *,
    name: str,
    query_length: int,
    key_value_length: int,
    causal: bool,
    num_heads: int = 1,
    head_dim: int = ATTENTION_HEAD_DIM,
) -> PyTorchBoardCase:
    if dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "read-only attention board cases require float16 or bfloat16"
        )
    try:
        from transformers import LlamaConfig
        from transformers.masking_utils import create_causal_mask
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            eager_attention_forward,
        )
    except ImportError as error:
        raise RuntimeError(
            "read-only attention board cases require the pinned Hugging Face "
            "Transformers importer dependency"
        ) from error

    config = LlamaConfig(
        hidden_size=num_heads * head_dim,
        intermediate_size=num_heads * head_dim * 4,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=key_value_length,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"

    class HuggingFaceLlamaEagerAttention(torch.nn.Module):
        """Export adapter around the official HF Llama eager backend."""

        def __init__(self) -> None:
            super().__init__()
            self.attention = LlamaAttention(config, layer_idx=0).eval()

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            output, _ = eager_attention_forward(
                self.attention,
                query,
                key,
                value,
                attention_mask,
                scaling=self.attention.scaling,
                dropout=0.0,
            )
            # HF returns [batch, query, heads, dim]. The case observes the
            # canonical [batch, heads, query, dim] layout; this adapter is only
            # a layout view around the official backend and contains no
            # attention arithmetic.
            return output.transpose(1, 2).contiguous()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = _random_tensor(
        (1, num_heads, query_length, head_dim),
        dtype=dtype,
        generator=generator,
    ) * 0.125
    key = _random_tensor(
        (1, num_heads, key_value_length, head_dim),
        dtype=dtype,
        generator=generator,
    ) * 0.125
    value = _random_tensor(
        (1, num_heads, key_value_length, head_dim),
        dtype=dtype,
        generator=generator,
    )
    if causal:
        positions = torch.arange(query_length, dtype=torch.long).unsqueeze(0)
        mask_input = query.transpose(1, 2).reshape(
            1, query_length, num_heads * head_dim
        )
        additive_mask = create_causal_mask(
            config=config,
            inputs_embeds=mask_input,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        if not isinstance(additive_mask, torch.Tensor):
            raise RuntimeError(
                "the official HF eager prefill path did not produce an "
                "additive tensor mask"
            )
    else:
        additive_mask = torch.zeros(
            (1, 1, query_length, key_value_length), dtype=dtype
        )
    module = HuggingFaceLlamaEagerAttention().eval()
    inputs = (query, key, value, additive_mask)

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            expected = module(*inputs)
        if expected.dtype != dtype or not torch.isfinite(expected).all():
            raise RuntimeError(
                f"PyTorch {name} eager reference must be finite and preserve dtype"
            )
        return (expected,)

    def export_program(output: pathlib.Path) -> None:
        _save_exported_program(output, module, inputs)

    return PyTorchBoardCase(
        name=name,
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=export_program,
        comparison_policy=ATTENTION_COMPARISON,
        prefill_extent=query_length if causal else None,
    )


def _attention_prefill(
    dtype: torch.dtype, seed: int, *, extent: int = 1024
) -> PyTorchBoardCase:
    return _read_only_attention(
        dtype,
        seed,
        name=(
            "attention-prefill" if extent == 1024 else f"attention-prefill-tail-{extent}"
        ),
        query_length=extent,
        key_value_length=extent,
        causal=True,
    )


def _llama_2_7b_attention_prefill(dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    config = json.loads(HF_LLAMA2_7B_CONFIG.read_text())
    return _read_only_attention(
        dtype,
        seed,
        name="attention-prefill-llama-2-7b",
        query_length=config["max_position_embeddings"],
        key_value_length=config["max_position_embeddings"],
        causal=True,
        num_heads=config["num_attention_heads"],
        head_dim=config["hidden_size"] // config["num_attention_heads"],
    )


def _attention_decode_kv_cache_step(
    dtype: torch.dtype,
    seed: int,
    *,
    past_length: int,
    past_key_input: torch.Tensor | None,
    past_value_input: torch.Tensor | None,
    step_ordinal: int,
    add_continuation: bool,
) -> PyTorchBoardCase:
    if dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "decode attention board cases require float16 or bfloat16"
        )
    try:
        from transformers import LlamaConfig
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaRotaryEmbedding,
        )
    except ImportError as error:
        raise RuntimeError(
            "decode attention board cases require the pinned Hugging Face "
            "Transformers importer dependency"
        ) from error

    class HuggingFaceLlamaFunctionalDecode(torch.nn.Module):
        """Explicit-state adapter over the official HF attention layer."""

        def __init__(self, position_id: int) -> None:
            super().__init__()
            config = LlamaConfig.from_dict(
                capture.load_hf_transformer_config(HF_LLAMA2_7B_CONFIG)
            )
            config._attn_implementation = "eager"
            self.hidden_size = int(config.hidden_size)
            self.head_dim = int(config.head_dim)
            self.num_key_value_heads = int(config.num_key_value_heads)
            self.attention = LlamaAttention(config, layer_idx=0).to(
                dtype=dtype
            ).eval()
            rotary = LlamaRotaryEmbedding(config).eval()
            rotary_input = torch.empty(
                (1, 1, self.head_dim), dtype=dtype
            )
            position_ids = torch.tensor([[position_id]], dtype=torch.long)
            with torch.no_grad():
                rotary_cos, rotary_sin = rotary(rotary_input, position_ids)
            self.register_buffer("rotary_cos", rotary_cos)
            self.register_buffer("rotary_sin", rotary_sin)

        def forward(
            self,
            hidden_states: torch.Tensor,
            past_key: torch.Tensor,
            past_value: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cache = DynamicCache([(past_key, past_value)])
            output, _ = self.attention(
                hidden_states,
                position_embeddings=(self.rotary_cos, self.rotary_sin),
                attention_mask=attention_mask,
                past_key_values=cache,
            )
            return (
                output,
                cache.layers[0].keys,
                cache.layers[0].values,
            )

    updated_length = past_length + 1
    hidden_generator = torch.Generator(device="cpu").manual_seed(
        seed + step_ordinal
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        module = HuggingFaceLlamaFunctionalDecode(past_length).eval()
    hidden_states = _random_tensor(
        (1, 1, module.hidden_size),
        dtype=dtype,
        generator=hidden_generator,
    ) * 0.125
    if (past_key_input is None) != (past_value_input is None):
        raise RuntimeError(
            "functional decode continuation requires both K and V state"
        )
    if past_key_input is None:
        cache_generator = torch.Generator(device="cpu").manual_seed(
            seed + 4096
        )
        past_key = _random_tensor(
            (1, module.num_key_value_heads, past_length, module.head_dim),
            dtype=dtype,
            generator=cache_generator,
        ) * 0.125
        past_value = _random_tensor(
            (1, module.num_key_value_heads, past_length, module.head_dim),
            dtype=dtype,
            generator=cache_generator,
        )
    else:
        past_key = past_key_input.detach().clone().contiguous()
        past_value = past_value_input.detach().clone().contiguous()
        expected_cache_shape = (
            1,
            module.num_key_value_heads,
            past_length,
            module.head_dim,
        )
        if (
            tuple(past_key.shape) != expected_cache_shape
            or tuple(past_value.shape) != expected_cache_shape
            or past_key.dtype != dtype
            or past_value.dtype != dtype
            or past_key.device.type != "cpu"
            or past_value.device.type != "cpu"
        ):
            raise RuntimeError(
                "functional decode continuation state differs from the "
                "official layer boundary"
            )
    position_ids = torch.tensor([[past_length]], dtype=torch.long)
    attention_mask = create_causal_mask(
        config=module.attention.config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=DynamicCache([(past_key, past_value)]),
        position_ids=position_ids,
    )
    if not isinstance(attention_mask, torch.Tensor):
        raise RuntimeError(
            "the official HF eager decode path did not produce an additive "
            "tensor mask"
        )
    inputs = (
        hidden_states,
        past_key,
        past_value,
        attention_mask,
    )

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            outputs = module(*inputs)
        if any(output.dtype != dtype for output in outputs) or any(
            not torch.isfinite(output).all() for output in outputs
        ):
            raise RuntimeError(
                "HF functional decode reference must be finite and preserve dtype"
            )
        if not torch.equal(outputs[1][..., :-1, :], past_key) or not torch.equal(
            outputs[2][..., :-1, :], past_value
        ):
            raise RuntimeError("HF functional decode did not preserve past cache")
        if (
            outputs[1].shape[-2] != updated_length
            or outputs[2].shape[-2] != updated_length
        ):
            raise RuntimeError("HF functional decode did not append one K/V token")
        return outputs

    continuation_factory = None
    if add_continuation:

        def continuation_factory(
            outputs: tuple[torch.Tensor, ...],
        ) -> PyTorchBoardCase:
            if len(outputs) != 3:
                raise RuntimeError(
                    "functional decode continuation requires attention, K, V"
                )
            return _attention_decode_kv_cache_step(
                dtype,
                seed,
                past_length=updated_length,
                past_key_input=outputs[1],
                past_value_input=outputs[2],
                step_ordinal=step_ordinal + 1,
                add_continuation=False,
            )

    return PyTorchBoardCase(
        name=(
            "attention-decode-kv-cache"
            if step_ordinal == 1
            else "attention-decode-kv-cache-continuation"
        ),
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, inputs
        ),
        comparison_policy=ATTENTION_COMPARISON,
        continuation_factory=continuation_factory,
    )


def _attention_decode_kv_cache(
    dtype: torch.dtype, seed: int
) -> PyTorchBoardCase:
    return _attention_decode_kv_cache_step(
        dtype,
        seed,
        past_length=1023,
        past_key_input=None,
        past_value_input=None,
        step_ordinal=1,
        add_continuation=True,
    )


CASE_FACTORIES: dict[
    str, Callable[[torch.dtype, int], PyTorchBoardCase]
] = {
    "allgather-add": make_allgather_add,
    "allgather-add-tail-1025": lambda dtype, seed: make_allgather_add(
        dtype, seed, extent=1025
    ),
    "allgather-add-tail-1031": lambda dtype, seed: make_allgather_add(
        dtype, seed, extent=1031
    ),
    "single-card-gemm": _single_card_gemm,
    "alltoall-transpose": make_alltoall_transpose,
    "reduce-scatter-sum": make_reduce_scatter_sum,
    "all-reduce-sum": make_all_reduce_sum,
    "all-reduce-sum-tail-1025": lambda dtype, seed: make_all_reduce_sum(
        dtype, seed, extent=1025
    ),
    "all-reduce-sum-tail-1031": lambda dtype, seed: make_all_reduce_sum(
        dtype, seed, extent=1031
    ),
    "reduce-scatter-sum-tail-1025": lambda dtype, seed: make_reduce_scatter_sum(
        dtype, seed, extent=1025
    ),
    "reduce-scatter-sum-tail-1031": lambda dtype, seed: make_reduce_scatter_sum(
        dtype, seed, extent=1031
    ),
    "alltoall-transpose-tail-1025": lambda dtype, seed: make_alltoall_transpose(
        dtype, seed, extent=1025
    ),
    "alltoall-transpose-tail-1031": lambda dtype, seed: make_alltoall_transpose(
        dtype, seed, extent=1031
    ),
    "single-card-gemm-tail-1025": lambda dtype, seed: _single_card_gemm(
        dtype, seed, m=1025, k=257, n=513
    ),
    "single-card-gemm-tail-1031": lambda dtype, seed: _single_card_gemm(
        dtype, seed, m=1031, k=263, n=519
    ),
    "heterogeneous-tiling-dataflow": _heterogeneous_tiling_single_card,
    "conv-mixed-dag": _conv_mixed_dag,
    "biased-conv": make_biased_conv,
    "biased-conv-tail-1025": lambda dtype, seed: make_biased_conv(dtype, seed, extent=1025),
    "biased-conv-tail-1031": lambda dtype, seed: make_biased_conv(dtype, seed, extent=1031),
    "division": make_division,
    "division-tail-1025": lambda dtype, seed: make_division(dtype, seed, extent=1025),
    "division-tail-1031": lambda dtype, seed: make_division(dtype, seed, extent=1031),
    "sigmoid": make_sigmoid,
    "sigmoid-tail-1025": lambda dtype, seed: make_sigmoid(dtype, seed, extent=1025),
    "sigmoid-tail-1031": lambda dtype, seed: make_sigmoid(dtype, seed, extent=1031),
    "local-conv": make_local_conv,
    "local-conv-tail-1025": lambda dtype, seed: make_local_conv(
        dtype, seed, extent=1025, kernel=(2, 3)
    ),
    "local-conv-tail-1031": lambda dtype, seed: make_local_conv(
        dtype, seed, extent=1031, kernel=(3, 2)
    ),
    "local-reduce": make_local_reduce,
    "local-reduce-tail-1025": lambda dtype, seed: make_local_reduce(
        dtype, seed, extent=1025
    ),
    "local-reduce-tail-1031": lambda dtype, seed: make_local_reduce(
        dtype, seed, extent=1031
    ),
    "attention-prefill": _attention_prefill,
    "attention-prefill-llama-2-7b": _llama_2_7b_attention_prefill,
    "attention-prefill-tail-1025": lambda dtype, seed: _attention_prefill(
        dtype, seed, extent=1025
    ),
    "attention-prefill-tail-1031": lambda dtype, seed: _attention_prefill(
        dtype, seed, extent=1031
    ),
    "attention-decode-kv-cache": _attention_decode_kv_cache,
    "llama-2-7b-block": _llama_2_7b_block,
}


def make_case(name: str, *, dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    try:
        case = CASE_FACTORIES[name](dtype, seed)
    except KeyError as error:
        raise RuntimeError(f"unknown PyTorch board case: {name}") from error
    for tensor in case.inputs:
        if tensor.device.type != "cpu" or tensor.dtype != dtype:
            raise RuntimeError(
                f"PyTorch board case {name} must preserve CPU dtype {dtype}"
            )
    return case
