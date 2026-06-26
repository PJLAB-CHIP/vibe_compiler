#!/usr/bin/env python3
"""PyTorch/XLA StableHLO program directory generator for Wafer frontend gates."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import operator
import pathlib
import shutil
import sys
from typing import Any, Callable


DEFAULT_REFERENCE_MATMUL_SIZE = 4096
DEFAULT_HF_TRANSFORMER_BATCH_SIZE = 1
DEFAULT_HF_TRANSFORMER_SEQUENCE_LENGTH = 4
HF_MEGATRON_TP_MESH_SHAPE = (16,)
HF_MEGATRON_TP_AXIS_NAMES = ("tensor",)
HF_MEGATRON_INPUT_SPEC = (None, None, None)
HF_MEGATRON_REPLICATED_VECTOR_SPEC = (None,)
HF_MEGATRON_COLUMN_PARALLEL_WEIGHT_SPEC = ("tensor", None)
HF_MEGATRON_ROW_PARALLEL_WEIGHT_SPEC = (None, "tensor")
HF_LLAMA_COLUMN_PARALLEL_WEIGHT_NAMES = frozenset(
    {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
    }
)
HF_LLAMA_ROW_PARALLEL_WEIGHT_NAMES = frozenset(
    {
        "o_proj.weight",
        "down_proj.weight",
    }
)
HF_LLAMA_REPLICATED_VECTOR_NAMES = frozenset(
    {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }
)


@dataclasses.dataclass(frozen=True)
class ShardingStrategy:
    name: str
    mesh_shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    input_spec: tuple[Any, ...]
    weight_spec: tuple[Any, ...]
    bias_spec: tuple[Any, ...]

    @property
    def device_count(self) -> int:
        return functools.reduce(operator.mul, self.mesh_shape, 1)


SHARDING_STRATEGIES: tuple[ShardingStrategy, ...] = (
    ShardingStrategy(
        name="data",
        mesh_shape=(16,),
        axis_names=("dp",),
        input_spec=("dp", None),
        weight_spec=(None, None),
        bias_spec=(None,),
    ),
    ShardingStrategy(
        name="column",
        mesh_shape=(16,),
        axis_names=("tp",),
        input_spec=(None, None),
        weight_spec=(None, "tp"),
        bias_spec=("tp",),
    ),
    ShardingStrategy(
        name="row",
        mesh_shape=(16,),
        axis_names=("tp",),
        input_spec=(None, "tp"),
        weight_spec=("tp", None),
        bias_spec=(None,),
    ),
    ShardingStrategy(
        name="2d-output",
        mesh_shape=(4, 4),
        axis_names=("dp", "tp"),
        input_spec=("dp", None),
        weight_spec=(None, "tp"),
        bias_spec=("tp",),
    ),
    ShardingStrategy(
        name="2d-contracting-output",
        mesh_shape=(2, 4, 2),
        axis_names=("dp", "tp", "mp"),
        input_spec=("dp", "mp"),
        weight_spec=("mp", "tp"),
        bias_spec=("tp",),
    ),
    ShardingStrategy(
        name="partial-replication",
        mesh_shape=(4, 4),
        axis_names=("dp", "tp"),
        input_spec=(None, "tp"),
        weight_spec=("tp", None),
        bias_spec=(None,),
    ),
)

SHARDING_STRATEGY_NAMES: tuple[str, ...] = tuple(
    strategy.name for strategy in SHARDING_STRATEGIES
)


def get_sharding_strategy(name: str) -> ShardingStrategy:
    for strategy in SHARDING_STRATEGIES:
        if strategy.name == name:
            return strategy
    valid = ", ".join(SHARDING_STRATEGY_NAMES)
    raise RuntimeError(f"unknown sharding strategy '{name}'; valid: {valid}")


def create_spmd_mesh(spmd_module: Any, strategy: ShardingStrategy) -> Any:
    return spmd_module.Mesh(
        list(range(strategy.device_count)),
        strategy.mesh_shape,
        strategy.axis_names,
    )


def apply_strategy_marks(
    *,
    spmd_module: Any,
    strategy: ShardingStrategy,
    mesh: Any,
    input_tensor: Any,
    reference_module: Any,
) -> None:
    spmd_module.mark_sharding(input_tensor, mesh, strategy.input_spec)
    spmd_module.mark_sharding(reference_module.weight, mesh, strategy.weight_spec)
    spmd_module.mark_sharding(reference_module.bias, mesh, strategy.bias_spec)


def create_hf_megatron_mesh(spmd_module: Any) -> Any:
    return spmd_module.Mesh(
        list(range(HF_MEGATRON_TP_MESH_SHAPE[0])),
        HF_MEGATRON_TP_MESH_SHAPE,
        HF_MEGATRON_TP_AXIS_NAMES,
    )


def apply_hf_megatron_sharding_marks(
    *,
    spmd_module: Any,
    mesh: Any,
    input_tensor: Any,
    reference_module: Any,
) -> None:
    spmd_module.mark_sharding(input_tensor, mesh, HF_MEGATRON_INPUT_SPEC)
    for name, parameter in _named_parameters(reference_module):
        if name in HF_LLAMA_REPLICATED_VECTOR_NAMES:
            spec = HF_MEGATRON_REPLICATED_VECTOR_SPEC
        elif name in HF_LLAMA_COLUMN_PARALLEL_WEIGHT_NAMES:
            spec = HF_MEGATRON_COLUMN_PARALLEL_WEIGHT_SPEC
        elif name in HF_LLAMA_ROW_PARALLEL_WEIGHT_NAMES:
            spec = HF_MEGATRON_ROW_PARALLEL_WEIGHT_SPEC
        else:
            raise RuntimeError(
                f"unsupported HF Megatron transformer parameter '{name}'"
            )
        spmd_module.mark_sharding(parameter, mesh, spec)


def _verify_program_dir_layout(program_dir: pathlib.Path) -> None:
    for relative in [
        pathlib.Path("functions") / "forward.mlir",
        pathlib.Path("functions") / "forward.meta",
        pathlib.Path("functions") / "forward.bytecode",
    ]:
        path = program_dir / relative
        if not path.is_file():
            raise RuntimeError(f"missing StableHLO program directory file: {relative}")
    if not (program_dir / "data").is_dir():
        raise RuntimeError("missing StableHLO program data directory: data")


def _make_reference_matmul_module(torch_module: Any, size: int) -> Any:
    class WaferReferenceMatmul(torch_module.nn.Module):
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

    return WaferReferenceMatmul()


def load_hf_transformer_config(config_path: pathlib.Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("model_type") != "llama":
        raise RuntimeError(
            "only Llama-family HuggingFace transformer configs are supported "
            f"by this test emitter, got model_type={config.get('model_type')!r}"
        )

    required_fields = [
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "rms_norm_eps",
    ]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise RuntimeError(
            "missing required HuggingFace transformer config fields: "
            + ", ".join(missing)
        )

    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    if hidden_size % num_attention_heads != 0:
        raise RuntimeError(
            "hidden_size must be divisible by num_attention_heads for the "
            "Llama decoder block test emitter"
        )
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))
    if hidden_size != num_attention_heads * head_dim:
        raise RuntimeError(
            "hidden_size must match num_attention_heads * head_dim for the "
            "Llama decoder block test emitter"
        )
    if int(config.get("num_key_value_heads", num_attention_heads)) != num_attention_heads:
        raise RuntimeError(
            "grouped-query attention is not part of this Megatron TP compiler gate"
        )

    return config


def _make_hf_llama_decoder_block_module(
    torch_module: Any,
    config: dict[str, Any],
    sequence_length: int,
) -> Any:
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))
    rms_norm_eps = float(config["rms_norm_eps"])
    rope_theta = float(config.get("rope_theta", 10000.0))
    initializer_range = float(config.get("initializer_range", 0.02))

    if sequence_length <= 0:
        raise RuntimeError("--sequence-length must be positive")

    class WaferLlamaRMSNorm(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch_module.nn.Parameter(
                torch_module.empty(hidden_size, dtype=torch_module.float32)
            )

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch_module.rsqrt(variance + rms_norm_eps)
            return x * self.weight

    class WaferLinearNoBias(torch_module.nn.Module):
        def __init__(self, out_features: int, in_features: int):
            super().__init__()
            self.weight = torch_module.nn.Parameter(
                torch_module.empty(
                    out_features, in_features, dtype=torch_module.float32
                )
            )

        def forward(self, x):
            return x @ self.weight.transpose(0, 1)

    def build_rope_cache() -> tuple[Any, Any]:
        positions = torch_module.arange(sequence_length, dtype=torch_module.float32)
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch_module.arange(0, head_dim, 2, dtype=torch_module.float32)
                / head_dim
            )
        )
        freqs = torch_module.outer(positions, inv_freq)
        embedding = torch_module.cat((freqs, freqs), dim=-1)
        return (
            torch_module.cos(embedding)[None, None, :, :],
            torch_module.sin(embedding)[None, None, :, :],
        )

    class WaferHFLlamaDecoderBlock(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = WaferLlamaRMSNorm()
            self.post_attention_layernorm = WaferLlamaRMSNorm()
            self.q_proj = WaferLinearNoBias(hidden_size, hidden_size)
            self.k_proj = WaferLinearNoBias(hidden_size, hidden_size)
            self.v_proj = WaferLinearNoBias(hidden_size, hidden_size)
            self.o_proj = WaferLinearNoBias(hidden_size, hidden_size)
            self.gate_proj = WaferLinearNoBias(intermediate_size, hidden_size)
            self.up_proj = WaferLinearNoBias(intermediate_size, hidden_size)
            self.down_proj = WaferLinearNoBias(hidden_size, intermediate_size)
            cos_cached, sin_cached = build_rope_cache()
            self.register_buffer("cos_cached", cos_cached, persistent=False)
            self.register_buffer("sin_cached", sin_cached, persistent=False)
            for _, parameter in self.named_parameters():
                torch_module.nn.init.normal_(
                    parameter, mean=0.0, std=initializer_range
                )

        def _shape_projection(self, x):
            batch, seq, _ = x.shape
            return (
                x.reshape(batch, seq, num_attention_heads, head_dim)
                .transpose(1, 2)
            )

        def _rotate_half(self, x):
            first_half = x[..., : head_dim // 2]
            second_half = x[..., head_dim // 2 :]
            return torch_module.cat((-second_half, first_half), dim=-1)

        def _apply_rope(self, query, key):
            seq = query.shape[-2]
            cos = self.cos_cached[:, :, :seq, :]
            sin = self.sin_cached[:, :, :seq, :]
            query = (query * cos) + (self._rotate_half(query) * sin)
            key = (key * cos) + (self._rotate_half(key) * sin)
            return query, key

        def forward(self, hidden_states):
            residual = hidden_states
            normed_states = self.input_layernorm(hidden_states)
            query = self._shape_projection(self.q_proj(normed_states))
            key = self._shape_projection(self.k_proj(normed_states))
            value = self._shape_projection(self.v_proj(normed_states))
            query, key = self._apply_rope(query, key)
            attn_scores = torch_module.matmul(
                query, key.transpose(-2, -1)
            ) * (1.0 / math.sqrt(head_dim))
            attn_weights = torch_module.softmax(attn_scores, dim=-1)
            attn_output = torch_module.matmul(attn_weights, value)
            batch, _, seq, _ = attn_output.shape
            attn_output = (
                attn_output.transpose(1, 2)
                .reshape(batch, seq, hidden_size)
            )
            hidden_states = residual + self.o_proj(attn_output)

            residual = hidden_states
            normed_states = self.post_attention_layernorm(hidden_states)
            gated = self.gate_proj(normed_states)
            up = self.up_proj(normed_states)
            mlp_output = torch_module.nn.functional.silu(gated) * up
            return residual + self.down_proj(mlp_output)

    return WaferHFLlamaDecoderBlock()


def _import_runtime_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch_xla._internal.custom_kernel  # noqa: F401
        from torch_xla import stablehlo
    except ImportError as error:
        raise RuntimeError(
            "PyTorch/XLA importer runtime is unavailable; install pinned torch "
            "Python packages, then build/install torch_xla from "
            "third_party/pytorch-xla with tools/build_pytorch_xla_runtime.py"
        ) from error
    return torch, stablehlo


def _import_spmd_runtime_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
        from torch_xla import stablehlo
    except ImportError as error:
        raise RuntimeError(
            "PyTorch/XLA SPMD importer runtime is unavailable; build/install "
            "torch_xla from third_party/pytorch-xla with the pinned "
            "repository dependencies"
        ) from error
    return torch, stablehlo, xr, xm, xs, torch_xla._XLAC


def _tensor_signature(stablehlo_module: Any, tensor: Any) -> Any:
    return stablehlo_module.VariableSignature(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        dynamic_dims=[],
    )


def _named_parameters(reference_module: Any) -> list[tuple[str, Any]]:
    named_parameters = getattr(reference_module, "named_parameters", None)
    if callable(named_parameters):
        parameters = [(name, parameter) for name, parameter in named_parameters()]
        if parameters:
            return parameters

    parameters = []
    for name in ["weight", "bias"]:
        if hasattr(reference_module, name):
            parameters.append((name, getattr(reference_module, name)))
    return parameters


def _state_dict_numpy(reference_module: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().numpy()
        for name, parameter in _named_parameters(reference_module)
    }


def _move_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _build_lazy_stablehlo_program(
    *,
    stablehlo_module: Any,
    xla_model_module: Any,
    xlac_module: Any,
    output_tensor: Any,
    input_tensor: Any,
    reference_module: Any,
    state_dict: dict[str, Any],
) -> Any:
    graph_input_tensor_ids, graph_input_xla_values = (
        xlac_module._get_tensors_xla_device_data_node([output_tensor])
    )
    id_to_location = {
        xlac_module._xla_get_tensor_id(input_tensor):
            stablehlo_module.InputLocation.input_arg(position=0),
    }
    for name, parameter in _named_parameters(reference_module):
        id_to_location[
            xlac_module._xla_get_tensor_id(parameter)
        ] = stablehlo_module.InputLocation.parameter(name=name)

    stablehlo_text = xla_model_module.get_stablehlo([output_tensor])
    stablehlo_bytecode = xla_model_module.get_stablehlo_bytecode([output_tensor])

    input_locations = []
    runtime_input_signatures = []
    additional_constants = []
    for tensor_id, tensor_value in zip(graph_input_tensor_ids,
                                       graph_input_xla_values):
        location = id_to_location.get(tensor_id)
        if location is None:
            location = stablehlo_module.InputLocation.constant(
                position=len(additional_constants)
            )
            additional_constants.append(tensor_value.detach().cpu().numpy())
        input_locations.append(location)
        runtime_input_signatures.append(
            _tensor_signature(stablehlo_module, tensor_value)
        )

    meta = stablehlo_module.StableHLOFunctionMeta(
        name="forward",
        stablehlo_version="0.0.0",
        input_signature=runtime_input_signatures,
        output_signature=[_tensor_signature(stablehlo_module, output_tensor)],
        input_locations=input_locations,
        unused_inputs=[],
        input_pytree_spec=None,
        output_pytree_spec=None,
    )
    func = stablehlo_module.StableHLOFunc(
        meta=meta,
        bytecode=stablehlo_bytecode,
        text=stablehlo_text,
    )
    return stablehlo_module.StableHLOModelBundle(
        state_dict=state_dict,
        additional_constants=additional_constants,
        stablehlo_funcs=[func],
    )


def emit_reference_stablehlo_program(
    program_dir: pathlib.Path,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
    reference_module_factory: Callable[[], Any] | None = None,
    size: int = DEFAULT_REFERENCE_MATMUL_SIZE,
) -> None:
    if torch_module is None or stablehlo_module is None:
        torch_module, stablehlo_module = _import_runtime_modules()

    if reference_module_factory is None:
        reference_module = _make_reference_matmul_module(torch_module, size)
    else:
        reference_module = reference_module_factory()
    reference_module.eval()
    input_tensor = torch_module.empty(size, size, dtype=torch_module.float32)

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        exported = torch_module.export.export(reference_module, (input_tensor,))
        stablehlo_program = stablehlo_module.exported_program_to_stablehlo(
            exported, options=options
        )

    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_program.save(str(program_dir))
    _verify_program_dir_layout(program_dir)


def emit_sharded_stablehlo_program(
    program_dir: pathlib.Path,
    strategy_name: str,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
    runtime_module: Any | None = None,
    xla_model_module: Any | None = None,
    spmd_module: Any | None = None,
    xlac_module: Any | None = None,
    reference_module_factory: Callable[[], Any] | None = None,
    size: int = DEFAULT_REFERENCE_MATMUL_SIZE,
) -> None:
    strategy = get_sharding_strategy(strategy_name)
    if (
        torch_module is None
        or stablehlo_module is None
        or runtime_module is None
        or xla_model_module is None
        or spmd_module is None
        or xlac_module is None
    ):
        (
            torch_module,
            stablehlo_module,
            runtime_module,
            xla_model_module,
            spmd_module,
            xlac_module,
        ) = _import_spmd_runtime_modules()

    runtime_module.use_spmd()
    device_count = runtime_module.global_runtime_device_count()
    if device_count != strategy.device_count:
        raise RuntimeError(
            f"sharding strategy '{strategy.name}' requires "
            f"{strategy.device_count} XLA devices, got {device_count}; "
            "for CPU program directory tests set CPU_NUM_DEVICES=16 before importing "
            "torch_xla"
        )

    if reference_module_factory is None:
        reference_module = _make_reference_matmul_module(torch_module, size)
    else:
        reference_module = reference_module_factory()
    reference_module.eval()
    state_dict = _state_dict_numpy(reference_module)

    device = xla_model_module.xla_device()
    reference_module = _move_to_device(reference_module, device)
    input_tensor = _move_to_device(
        torch_module.empty(size, size, dtype=torch_module.float32), device
    )

    mesh = create_spmd_mesh(spmd_module, strategy)
    apply_strategy_marks(
        spmd_module=spmd_module,
        strategy=strategy,
        mesh=mesh,
        input_tensor=input_tensor,
        reference_module=reference_module,
    )

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        output_tensor = reference_module(input_tensor)
        stablehlo_graph = _build_lazy_stablehlo_program(
            stablehlo_module=stablehlo_module,
            xla_model_module=xla_model_module,
            xlac_module=xlac_module,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            reference_module=reference_module,
            state_dict=state_dict,
        )

    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_module.StableHLOGraphModule(stablehlo_graph).save(
        str(program_dir), options
    )
    _verify_program_dir_layout(program_dir)


def emit_hf_megatron_transformer_block_program(
    program_dir: pathlib.Path,
    config_path: pathlib.Path,
    batch_size: int = DEFAULT_HF_TRANSFORMER_BATCH_SIZE,
    sequence_length: int = DEFAULT_HF_TRANSFORMER_SEQUENCE_LENGTH,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
    runtime_module: Any | None = None,
    xla_model_module: Any | None = None,
    spmd_module: Any | None = None,
    xlac_module: Any | None = None,
) -> None:
    config = load_hf_transformer_config(config_path)
    if batch_size <= 0:
        raise RuntimeError("--batch-size must be positive")

    if (
        torch_module is None
        or stablehlo_module is None
        or runtime_module is None
        or xla_model_module is None
        or spmd_module is None
        or xlac_module is None
    ):
        (
            torch_module,
            stablehlo_module,
            runtime_module,
            xla_model_module,
            spmd_module,
            xlac_module,
        ) = _import_spmd_runtime_modules()

    runtime_module.use_spmd()
    device_count = runtime_module.global_runtime_device_count()
    required_device_count = functools.reduce(
        operator.mul, HF_MEGATRON_TP_MESH_SHAPE, 1
    )
    if device_count != required_device_count:
        raise RuntimeError(
            "HF Megatron transformer block requires "
            f"{required_device_count} XLA devices, got {device_count}; "
            "for CPU program directory tests set CPU_NUM_DEVICES=16 before importing "
            "torch_xla"
        )

    torch_module.manual_seed(0)
    reference_module = _make_hf_llama_decoder_block_module(
        torch_module, config, sequence_length
    )
    reference_module.eval()
    state_dict = _state_dict_numpy(reference_module)

    device = xla_model_module.xla_device()
    reference_module = _move_to_device(reference_module, device)
    input_tensor = _move_to_device(
        torch_module.empty(
            batch_size,
            sequence_length,
            int(config["hidden_size"]),
            dtype=torch_module.float32,
        ),
        device,
    )

    mesh = create_hf_megatron_mesh(spmd_module)
    apply_hf_megatron_sharding_marks(
        spmd_module=spmd_module,
        mesh=mesh,
        input_tensor=input_tensor,
        reference_module=reference_module,
    )

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        output_tensor = reference_module(input_tensor)
        stablehlo_graph = _build_lazy_stablehlo_program(
            stablehlo_module=stablehlo_module,
            xla_model_module=xla_model_module,
            xlac_module=xlac_module,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            reference_module=reference_module,
            state_dict=state_dict,
        )

    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_module.StableHLOGraphModule(stablehlo_graph).save(
        str(program_dir), options
    )
    _verify_program_dir_layout(program_dir)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-reference-program",
        action="store_true",
        help="emit the reference 4096x4096 matmul+bias+tanh+residual StableHLO program directory",
    )
    parser.add_argument(
        "--emit-sharded-program",
        action="store_true",
        help="emit the 4096x4096 sharded StableHLO program directory",
    )
    parser.add_argument(
        "--emit-hf-megatron-transformer-block",
        action="store_true",
        help="emit a HuggingFace Llama decoder block StableHLO program with Megatron tensor-parallel mark_sharding on a 16-rank mesh",
    )
    parser.add_argument(
        "--sharding-strategy",
        choices=SHARDING_STRATEGY_NAMES,
        help="user sharding strategy",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_REFERENCE_MATMUL_SIZE,
        help="square matmul size for generated reference program directories",
    )
    parser.add_argument(
        "--hf-config-json",
        type=pathlib.Path,
        help="HuggingFace transformer config JSON used by --emit-hf-megatron-transformer-block",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_HF_TRANSFORMER_BATCH_SIZE,
        help="batch size for --emit-hf-megatron-transformer-block",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_HF_TRANSFORMER_SEQUENCE_LENGTH,
        help="sequence length for --emit-hf-megatron-transformer-block",
    )
    parser.add_argument("--output-program-dir", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.emit_reference_program:
        if args.output_program_dir is None:
            raise RuntimeError("missing --output-program-dir")

        emit_reference_stablehlo_program(args.output_program_dir, size=args.size)
        return 0

    if args.emit_sharded_program:
        if args.output_program_dir is None:
            raise RuntimeError("missing --output-program-dir")
        if args.sharding_strategy is None:
            raise RuntimeError("missing --sharding-strategy")

        emit_sharded_stablehlo_program(
            args.output_program_dir,
            strategy_name=args.sharding_strategy,
            size=args.size,
        )
        return 0

    if args.emit_hf_megatron_transformer_block:
        if args.output_program_dir is None:
            raise RuntimeError("missing --output-program-dir")
        if args.hf_config_json is None:
            raise RuntimeError("missing --hf-config-json")

        emit_hf_megatron_transformer_block_program(
            args.output_program_dir,
            config_path=args.hf_config_json,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
        return 0

    raise RuntimeError(
        "missing --emit-reference-program, --emit-sharded-program, or "
        "--emit-hf-megatron-transformer-block"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-capture: {error}", file=sys.stderr)
        raise SystemExit(1)
