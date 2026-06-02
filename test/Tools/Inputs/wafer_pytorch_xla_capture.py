#!/usr/bin/env python3
"""PyTorch/XLA StableHLO program directory generator for Wafer frontend gates."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import operator
import pathlib
import shutil
import sys
from typing import Any, Callable


DEFAULT_REFERENCE_MATMUL_SIZE = 4096


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


def _state_dict_numpy(reference_module: Any) -> dict[str, Any]:
    return {
        "weight": reference_module.weight.detach().cpu().numpy(),
        "bias": reference_module.bias.detach().cpu().numpy(),
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
        xlac_module._xla_get_tensor_id(reference_module.weight):
            stablehlo_module.InputLocation.parameter(name="weight"),
        xlac_module._xla_get_tensor_id(reference_module.bias):
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

    raise RuntimeError(
        "missing --emit-reference-program or --emit-sharded-program"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-capture: {error}", file=sys.stderr)
        raise SystemExit(1)
