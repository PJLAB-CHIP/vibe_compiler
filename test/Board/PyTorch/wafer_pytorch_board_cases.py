#!/usr/bin/env python3
"""Framework-owned GEMM and HuggingFace Transformer board source cases."""

from __future__ import annotations

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


HF_LLAMA2_7B_CONFIG = (
    TOOLS_INPUTS / "hf" / "llama-2-7b-block-config.json"
)
HF_LLAMA2_7B_SEQUENCE_LENGTH = 16
HF_LLAMA2_7B_COMPARISON = common.ComparisonPolicy(rtol=0.002, atol=0.004)
ATTENTION_COMPARISON = common.ComparisonPolicy(rtol=0.006, atol=0.008)
ATTENTION_HEAD_DIM = 64
OPTIMIZATION_POLICIES = ("search", "none")


@dataclasses.dataclass(frozen=True)
class PyTorchBoardCase:
    name: str
    num_partitions: int
    dtype: torch.dtype
    inputs: tuple[torch.Tensor, ...]
    expected_outputs_factory: Callable[[], tuple[torch.Tensor, ...]]
    export_program: Callable[[pathlib.Path], None]
    required_structured_ir: tuple[str, ...]
    expected_all_reduce_count: int
    comparison_policy: common.ComparisonPolicy
    continuation_factory: (
        Callable[[tuple[torch.Tensor, ...]], "PyTorchBoardCase"] | None
    ) = None
    minimum_search_actual_fused_edges: int = 0

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
    """One explicitly callable source/oracle/package/no-card vertical."""

    ctest_name: str
    case_name: str
    dtype_name: str
    num_partitions: int
    optimization_policy: str = "search"
    package_count: int = 1


SOURCE_NO_CARD_WORKLOADS = (
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-single-card-gemm-no-card",
        "single-card-gemm",
        "float16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-single-card-gemm-bf16-no-card",
        "single-card-gemm",
        "bfloat16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-heterogeneous-tiling-dataflow-fp16-no-card",
        "heterogeneous-tiling-dataflow",
        "float16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-heterogeneous-tiling-dataflow-bf16-no-card",
        "heterogeneous-tiling-dataflow",
        "bfloat16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-conv-mixed-dag-fp16-no-card",
        "conv-mixed-dag",
        "float16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-conv-mixed-dag-bf16-no-card",
        "conv-mixed-dag",
        "bfloat16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-attention-prefill-fp16-no-card",
        "attention-prefill",
        "float16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-attention-prefill-fp16-optimization-none-no-card",
        "attention-prefill",
        "float16",
        1,
        optimization_policy="none",
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-attention-prefill-bf16-no-card",
        "attention-prefill",
        "bfloat16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-attention-decode-kv-cache-no-card",
        "attention-decode-kv-cache",
        "float16",
        1,
        package_count=2,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-attention-decode-kv-cache-fp16-optimization-none-no-card",
        "attention-decode-kv-cache",
        "float16",
        1,
        optimization_policy="none",
        package_count=2,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-attention-decode-kv-cache-bf16-no-card",
        "attention-decode-kv-cache",
        "bfloat16",
        1,
        package_count=2,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-llama-2-7b-block-fp16-no-card",
        "llama-2-7b-block",
        "float16",
        1,
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-llama-2-7b-block-fp16-optimization-none-no-card",
        "llama-2-7b-block",
        "float16",
        1,
        optimization_policy="none",
    ),
    SourceNoCardWorkload(
        "wafer-runtime-pytorch-llama-2-7b-block-bf16-no-card",
        "llama-2-7b-block",
        "bfloat16",
        1,
    ),
)


class Gemm(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)


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
    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(program_dir))
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


def _single_card_gemm(dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    m, k, n = 256, 256, 512
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lhs = _random_tensor((m, k), dtype=dtype, generator=generator)
    rhs = _random_tensor((k, n), dtype=dtype, generator=generator)
    module = Gemm().eval()

    def expected_outputs_factory() -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return (module(lhs, rhs),)

    return PyTorchBoardCase(
        name="single-card-gemm",
        num_partitions=1,
        dtype=dtype,
        inputs=(lhs, rhs),
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, (lhs, rhs)
        ),
        required_structured_ir=(
            "linalg.matmul",
        ),
        expected_all_reduce_count=0,
        comparison_policy=common.PYTORCH_DEFAULT,
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
        required_structured_ir=("linalg.matmul", "linalg.generic"),
        expected_all_reduce_count=0,
        comparison_policy=common.PYTORCH_DEFAULT,
        minimum_search_actual_fused_edges=1,
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
    input_tensor = _random_tensor(
        (1, 16, 32, 32), dtype=dtype, generator=generator
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
        num_partitions=1,
        dtype=dtype,
        inputs=inputs,
        expected_outputs_factory=expected_outputs_factory,
        export_program=lambda output: _save_exported_program(
            output, module, inputs
        ),
        required_structured_ir=(
            "tensor.pad",
            "linalg.generic",
        ),
        expected_all_reduce_count=0,
        comparison_policy=common.PYTORCH_DEFAULT,
        minimum_search_actual_fused_edges=1,
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
        required_structured_ir=("linalg.matmul", "math.exp"),
        expected_all_reduce_count=0,
        comparison_policy=HF_LLAMA2_7B_COMPARISON,
        minimum_search_actual_fused_edges=1,
    )


def _read_only_attention(
    dtype: torch.dtype,
    seed: int,
    *,
    name: str,
    query_length: int,
    key_value_length: int,
    causal: bool,
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
        hidden_size=ATTENTION_HEAD_DIM,
        intermediate_size=ATTENTION_HEAD_DIM * 4,
        num_attention_heads=1,
        num_key_value_heads=1,
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
        (1, 1, query_length, ATTENTION_HEAD_DIM),
        dtype=dtype,
        generator=generator,
    ) * 0.125
    key = _random_tensor(
        (1, 1, key_value_length, ATTENTION_HEAD_DIM),
        dtype=dtype,
        generator=generator,
    ) * 0.125
    value = _random_tensor(
        (1, 1, key_value_length, ATTENTION_HEAD_DIM),
        dtype=dtype,
        generator=generator,
    )
    if causal:
        positions = torch.arange(query_length, dtype=torch.long).unsqueeze(0)
        mask_input = query.transpose(1, 2).reshape(
            1, query_length, ATTENTION_HEAD_DIM
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
        required_structured_ir=("linalg.generic", "math.exp"),
        expected_all_reduce_count=0,
        comparison_policy=ATTENTION_COMPARISON,
        minimum_search_actual_fused_edges=1,
    )


def _attention_prefill(dtype: torch.dtype, seed: int) -> PyTorchBoardCase:
    return _read_only_attention(
        dtype,
        seed,
        name="attention-prefill",
        query_length=1024,
        key_value_length=1024,
        causal=True,
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
        required_structured_ir=(
            "tensor.insert_slice",
            "math.exp",
        ),
        expected_all_reduce_count=0,
        comparison_policy=ATTENTION_COMPARISON,
        continuation_factory=continuation_factory,
        minimum_search_actual_fused_edges=1,
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
    "single-card-gemm": _single_card_gemm,
    "heterogeneous-tiling-dataflow": _heterogeneous_tiling_single_card,
    "conv-mixed-dag": _conv_mixed_dag,
    "attention-prefill": _attention_prefill,
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
