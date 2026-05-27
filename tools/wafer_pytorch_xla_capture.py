#!/usr/bin/env python3
"""PyTorch/XLA capture adapter for Wafer frontend artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import operator
import os
import pathlib
import re
import shutil
import sys
from typing import Any, Callable


P2F1_SMOKE_SIZE = 4096


@dataclasses.dataclass(frozen=True)
class P2S1ShardingStrategy:
    name: str
    mesh_shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    input_spec: tuple[Any, ...]
    weight_spec: tuple[Any, ...]
    bias_spec: tuple[Any, ...]

    @property
    def device_count(self) -> int:
        return functools.reduce(operator.mul, self.mesh_shape, 1)


P2S1_SHARDING_STRATEGIES: tuple[P2S1ShardingStrategy, ...] = (
    P2S1ShardingStrategy(
        name="data",
        mesh_shape=(16,),
        axis_names=("dp",),
        input_spec=("dp", None),
        weight_spec=(None, None),
        bias_spec=(None,),
    ),
    P2S1ShardingStrategy(
        name="column",
        mesh_shape=(16,),
        axis_names=("tp",),
        input_spec=(None, None),
        weight_spec=(None, "tp"),
        bias_spec=("tp",),
    ),
    P2S1ShardingStrategy(
        name="row",
        mesh_shape=(16,),
        axis_names=("tp",),
        input_spec=(None, "tp"),
        weight_spec=("tp", None),
        bias_spec=(None,),
    ),
    P2S1ShardingStrategy(
        name="2d-output",
        mesh_shape=(4, 4),
        axis_names=("dp", "tp"),
        input_spec=("dp", None),
        weight_spec=(None, "tp"),
        bias_spec=("tp",),
    ),
    P2S1ShardingStrategy(
        name="2d-contracting-output",
        mesh_shape=(2, 4, 2),
        axis_names=("dp", "tp", "mp"),
        input_spec=("dp", "mp"),
        weight_spec=("mp", "tp"),
        bias_spec=("tp",),
    ),
    P2S1ShardingStrategy(
        name="partial-replication",
        mesh_shape=(4, 4),
        axis_names=("dp", "tp"),
        input_spec=(None, "tp"),
        weight_spec=("tp", None),
        bias_spec=(None,),
    ),
)

P2S1_SHARDING_STRATEGY_NAMES: tuple[str, ...] = tuple(
    strategy.name for strategy in P2S1_SHARDING_STRATEGIES
)


def create_p2s1_default_input_strategy(
    *, tile_count: int, size: int = P2F1_SMOKE_SIZE
) -> P2S1ShardingStrategy:
    if tile_count < 1 or tile_count > 16:
        raise RuntimeError("P2.S1 default tile count must be in [1, 16]")

    def default_spec(shape: tuple[int, ...]) -> tuple[Any, ...]:
        if tile_count == 1:
            return tuple(None for _ in shape)
        for index, dim in enumerate(shape):
            if dim % tile_count == 0:
                spec = [None for _ in shape]
                spec[index] = "tile"
                return tuple(spec)
        return tuple(None for _ in shape)

    return P2S1ShardingStrategy(
        name=f"default-input-seed-{tile_count}",
        mesh_shape=(tile_count,),
        axis_names=("tile",),
        input_spec=default_spec((size, size)),
        weight_spec=default_spec((size, size)),
        bias_spec=default_spec((size,)),
    )


def get_p2s1_sharding_strategy(name: str) -> P2S1ShardingStrategy:
    for strategy in P2S1_SHARDING_STRATEGIES:
        if strategy.name == name:
            return strategy
    valid = ", ".join(P2S1_SHARDING_STRATEGY_NAMES)
    raise RuntimeError(f"unknown P2.S1 sharding strategy '{name}'; valid: {valid}")


def create_p2s1_mesh(spmd_module: Any, strategy: P2S1ShardingStrategy) -> Any:
    return spmd_module.Mesh(
        list(range(strategy.device_count)),
        strategy.mesh_shape,
        strategy.axis_names,
    )


def apply_p2s1_strategy_marks(
    *,
    spmd_module: Any,
    strategy: P2S1ShardingStrategy,
    mesh: Any,
    input_tensor: Any,
    smoke_module: Any,
) -> None:
    spmd_module.mark_sharding(input_tensor, mesh, strategy.input_spec)
    spmd_module.mark_sharding(smoke_module.weight, mesh, strategy.weight_spec)
    spmd_module.mark_sharding(smoke_module.bias, mesh, strategy.bias_spec)


def _with_disabled_hlo_pass(flags: str, pass_name: str) -> str:
    tokens = flags.split()
    prefixes = ("--xla_disable_hlo_passes=", "xla_disable_hlo_passes=")
    for index, token in enumerate(tokens):
        for prefix in prefixes:
            if token.startswith(prefix):
                passes = [value for value in token[len(prefix):].split(",") if value]
                if pass_name not in passes:
                    passes.append(pass_name)
                    tokens[index] = prefix + ",".join(passes)
                return " ".join(tokens)
    tokens.append(f"--xla_disable_hlo_passes={pass_name}")
    return " ".join(tokens)


def configure_p2s1_partitioned_export_environment() -> None:
    os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "1"
    os.environ["XLA_FLAGS"] = _with_disabled_hlo_pass(
        os.environ.get("XLA_FLAGS", ""), "fusion"
    )


def _parse_stablehlo_tensor_type(type_text: str) -> tuple[list[int], str]:
    match = re.fullmatch(r"tensor<(.+)>", type_text.strip())
    if not match:
        raise RuntimeError(f"unsupported StableHLO function type: {type_text}")

    body = match.group(1)
    parts = body.split("x")
    dtype = parts[-1]
    shape = []
    for dim in parts[:-1]:
        if dim == "?":
            raise RuntimeError("dynamic partitioned StableHLO signatures are unsupported")
        shape.append(int(dim))

    dtype_map = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "ui8": "uint8",
        "ui16": "uint16",
        "ui32": "uint32",
        "ui64": "uint64",
    }
    if dtype not in dtype_map:
        raise RuntimeError(f"unsupported StableHLO function dtype: {dtype}")
    return shape, dtype_map[dtype]


def _split_result_types(result_text: str) -> list[str]:
    result_text = result_text.strip()
    if result_text.startswith("(") and result_text.endswith(")"):
        result_text = result_text[1:-1].strip()
    if not result_text:
        return []
    return re.findall(r"tensor<[^>]+>", result_text)


def _parse_stablehlo_main_signatures(
    stablehlo_module: Any, stablehlo_text: str
) -> tuple[list[Any], list[Any]]:
    match = re.search(
        r"func\.func\s+@main\s*\((?P<args>.*?)\)\s*->\s*"
        r"(?P<results>.*?)\s*\{",
        stablehlo_text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("partitioned StableHLO text is missing func.func @main")

    arg_types = re.findall(r"%arg\d+:\s*(tensor<[^>]+>)", match.group("args"))
    result_types = _split_result_types(match.group("results"))
    if not result_types:
        raise RuntimeError("partitioned StableHLO function must have a result")

    def make_signature(type_text: str) -> Any:
        shape, dtype = _parse_stablehlo_tensor_type(type_text)
        return stablehlo_module.VariableSignature(
            shape=shape, dtype=dtype, dynamic_dims=[]
        )

    return (
        [make_signature(type_text) for type_text in arg_types],
        [make_signature(type_text) for type_text in result_types],
    )


def _verify_bundle_layout(bundle_path: pathlib.Path) -> None:
    for relative in [
        pathlib.Path("functions") / "forward.mlir",
        pathlib.Path("functions") / "forward.meta",
        pathlib.Path("functions") / "forward.bytecode",
    ]:
        path = bundle_path / relative
        if not path.is_file():
            raise RuntimeError(f"missing StableHLO bundle file: {relative}")
    if not (bundle_path / "data").is_dir():
        raise RuntimeError("missing StableHLO bundle directory: data")


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


def _state_dict_numpy(smoke_module: Any) -> dict[str, Any]:
    return {
        "weight": smoke_module.weight.detach().cpu().numpy(),
        "bias": smoke_module.bias.detach().cpu().numpy(),
    }


def _move_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _build_lazy_stablehlo_bundle(
    *,
    stablehlo_module: Any,
    xla_model_module: Any,
    xlac_module: Any,
    output_tensor: Any,
    input_tensor: Any,
    smoke_module: Any,
    state_dict: dict[str, Any],
    use_exported_function_signatures: bool = False,
) -> Any:
    graph_input_tensor_ids, graph_input_xla_values = (
        xlac_module._get_tensors_xla_device_data_node([output_tensor])
    )
    id_to_location = {
        xlac_module._xla_get_tensor_id(input_tensor):
            stablehlo_module.InputLocation.input_arg(position=0),
        xlac_module._xla_get_tensor_id(smoke_module.weight):
            stablehlo_module.InputLocation.parameter(name="weight"),
        xlac_module._xla_get_tensor_id(smoke_module.bias):
            stablehlo_module.InputLocation.parameter(name="bias"),
    }

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

    if use_exported_function_signatures:
        input_signatures, output_signatures = _parse_stablehlo_main_signatures(
            stablehlo_module, stablehlo_text
        )
        if len(input_signatures) != len(input_locations):
            raise RuntimeError(
                "partitioned StableHLO function input count does not match "
                "PyTorch/XLA graph inputs"
            )
    else:
        input_signatures = runtime_input_signatures
        output_signatures = [_tensor_signature(stablehlo_module, output_tensor)]

    meta = stablehlo_module.StableHLOFunctionMeta(
        name="forward",
        stablehlo_version="0.0.0",
        input_signature=input_signatures,
        output_signature=output_signatures,
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


def emit_p2f1_smoke_bundle(
    bundle_path: pathlib.Path,
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
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        exported = torch_module.export.export(smoke_module, (input_tensor,))
        stablehlo_program = stablehlo_module.exported_program_to_stablehlo(
            exported, options=options
        )

    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_program.save(str(bundle_path))
    _verify_bundle_layout(bundle_path)


def emit_p2s1_sharded_smoke_bundle(
    bundle_path: pathlib.Path,
    strategy_name: str,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
    runtime_module: Any | None = None,
    xla_model_module: Any | None = None,
    spmd_module: Any | None = None,
    xlac_module: Any | None = None,
    smoke_module_factory: Callable[[], Any] | None = None,
    size: int = P2F1_SMOKE_SIZE,
    partitioned: bool = False,
) -> None:
    strategy = get_p2s1_sharding_strategy(strategy_name)
    if (
        torch_module is None
        or stablehlo_module is None
        or runtime_module is None
        or xla_model_module is None
        or spmd_module is None
        or xlac_module is None
    ):
        if partitioned:
            configure_p2s1_partitioned_export_environment()
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
            f"P2.S1 strategy '{strategy.name}' requires "
            f"{strategy.device_count} XLA devices, got {device_count}; "
            "for CPU smoke tests set CPU_NUM_DEVICES=16 before importing "
            "torch_xla"
        )

    if smoke_module_factory is None:
        smoke_module = _make_smoke_module(torch_module, size)
    else:
        smoke_module = smoke_module_factory()
    smoke_module.eval()
    state_dict = _state_dict_numpy(smoke_module)

    device = xla_model_module.xla_device()
    smoke_module = _move_to_device(smoke_module, device)
    input_tensor = _move_to_device(
        torch_module.empty(size, size, dtype=torch_module.float32), device
    )

    mesh = create_p2s1_mesh(spmd_module, strategy)
    apply_p2s1_strategy_marks(
        spmd_module=spmd_module,
        strategy=strategy,
        mesh=mesh,
        input_tensor=input_tensor,
        smoke_module=smoke_module,
    )

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        output_tensor = smoke_module(input_tensor)
        bundle = _build_lazy_stablehlo_bundle(
            stablehlo_module=stablehlo_module,
            xla_model_module=xla_model_module,
            xlac_module=xlac_module,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            smoke_module=smoke_module,
            state_dict=state_dict,
            use_exported_function_signatures=partitioned,
        )

    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_module.StableHLOGraphModule(bundle).save(str(bundle_path), options)
    _verify_bundle_layout(bundle_path)


def emit_p2s1_partitioned_smoke_bundle(
    bundle_path: pathlib.Path,
    *,
    strategy_name: str | None = None,
    default_input_sharding: bool = False,
    default_tile_count: int = 16,
    size: int = P2F1_SMOKE_SIZE,
) -> None:
    if (strategy_name is None) == (not default_input_sharding):
        raise RuntimeError(
            "--emit-p2s1-partitioned-smoke requires exactly one of "
            "--sharding-strategy or --default-input-sharding"
        )

    if default_input_sharding:
        strategy = create_p2s1_default_input_strategy(
            tile_count=default_tile_count, size=size
        )
        strategy_name = strategy.name
    else:
        strategy = get_p2s1_sharding_strategy(strategy_name or "")

    _emit_p2s1_smoke_bundle_with_strategy(
        bundle_path=bundle_path,
        strategy=strategy,
        size=size,
        partitioned=True,
    )


def _emit_p2s1_smoke_bundle_with_strategy(
    *,
    bundle_path: pathlib.Path,
    strategy: P2S1ShardingStrategy,
    size: int,
    partitioned: bool,
) -> None:
    if partitioned:
        configure_p2s1_partitioned_export_environment()
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
            f"P2.S1 strategy '{strategy.name}' requires "
            f"{strategy.device_count} XLA devices, got {device_count}; "
            "for CPU smoke tests set CPU_NUM_DEVICES to the strategy device "
            "count before importing torch_xla"
        )

    smoke_module = _make_smoke_module(torch_module, size)
    smoke_module.eval()
    state_dict = _state_dict_numpy(smoke_module)

    device = xla_model_module.xla_device()
    smoke_module = _move_to_device(smoke_module, device)
    input_tensor = _move_to_device(
        torch_module.empty(size, size, dtype=torch_module.float32), device
    )

    mesh = create_p2s1_mesh(spmd_module, strategy)
    apply_p2s1_strategy_marks(
        spmd_module=spmd_module,
        strategy=strategy,
        mesh=mesh,
        input_tensor=input_tensor,
        smoke_module=smoke_module,
    )

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        output_tensor = smoke_module(input_tensor)
        bundle = _build_lazy_stablehlo_bundle(
            stablehlo_module=stablehlo_module,
            xla_model_module=xla_model_module,
            xlac_module=xlac_module,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            smoke_module=smoke_module,
            state_dict=state_dict,
            use_exported_function_signatures=partitioned,
        )

    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_module.StableHLOGraphModule(bundle).save(str(bundle_path), options)
    _verify_bundle_layout(bundle_path)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-p2f1-smoke",
        action="store_true",
        help="emit the P2.F1 4096x4096 matmul+bias+tanh+residual artifact",
    )
    parser.add_argument(
        "--emit-p2s1-sharded-smoke",
        action="store_true",
        help="emit the P2.S1 4096x4096 sharded matmul artifact",
    )
    parser.add_argument(
        "--emit-p2s1-partitioned-smoke",
        action="store_true",
        help="emit the P2.S1 4096x4096 post-XLA-SPMD partitioned artifact",
    )
    parser.add_argument(
        "--sharding-strategy",
        choices=P2S1_SHARDING_STRATEGY_NAMES,
        help="P2.S1 user sharding strategy",
    )
    parser.add_argument(
        "--default-input-sharding",
        action="store_true",
        help="apply the P2.S1 no-user-sharding default input seed policy",
    )
    parser.add_argument(
        "--default-tile-count",
        type=int,
        default=16,
        help="tile count for --default-input-sharding",
    )
    parser.add_argument("--output-bundle", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.emit_p2f1_smoke:
        if args.output_bundle is None:
            raise RuntimeError("missing --output-bundle")

        emit_p2f1_smoke_bundle(args.output_bundle)
        return 0

    if args.emit_p2s1_sharded_smoke:
        if args.output_bundle is None:
            raise RuntimeError("missing --output-bundle")
        if args.sharding_strategy is None:
            raise RuntimeError("missing --sharding-strategy")

        emit_p2s1_sharded_smoke_bundle(
            args.output_bundle, strategy_name=args.sharding_strategy
        )
        return 0

    if args.emit_p2s1_partitioned_smoke:
        if args.output_bundle is None:
            raise RuntimeError("missing --output-bundle")

        emit_p2s1_partitioned_smoke_bundle(
            args.output_bundle,
            strategy_name=args.sharding_strategy,
            default_input_sharding=args.default_input_sharding,
            default_tile_count=args.default_tile_count,
        )
        return 0

    raise RuntimeError(
        "missing --emit-p2f1-smoke, --emit-p2s1-sharded-smoke, or "
        "--emit-p2s1-partitioned-smoke"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-capture: {error}", file=sys.stderr)
        raise SystemExit(1)
