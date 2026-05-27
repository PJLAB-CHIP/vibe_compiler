#!/usr/bin/env python3
"""PyTorch/XLA capture adapter for Wafer frontend artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import sys
from typing import Any, Callable, Sequence


P2F1_SMOKE_SIZE = 4096


@dataclasses.dataclass(frozen=True)
class P2S1ShardingStrategy:
    name: str
    mesh_shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    input_spec: tuple[str | None, ...]
    weight_spec: tuple[str | None, ...]
    bias_spec: tuple[str | None, ...]
    output_spec: tuple[str | None, ...]
    collective_axis: str | None = None
    collective_kind: str | None = None

    @property
    def device_count(self) -> int:
        count = 1
        for size in self.mesh_shape:
            count *= size
        return count


@dataclasses.dataclass(frozen=True)
class EmittedP2S1Bundle:
    strategy: str
    path: pathlib.Path
    global_rank: int
    local_rank: int


P2S1_SHARDING_STRATEGIES: tuple[P2S1ShardingStrategy, ...] = (
    P2S1ShardingStrategy(
        name="data_batch",
        mesh_shape=(4,),
        axis_names=("dp",),
        input_spec=("dp", None),
        weight_spec=(None, None),
        bias_spec=(None,),
        output_spec=("dp", None),
    ),
    P2S1ShardingStrategy(
        name="column_parallel",
        mesh_shape=(4,),
        axis_names=("tp",),
        input_spec=(None, None),
        weight_spec=(None, "tp"),
        bias_spec=("tp",),
        output_spec=(None, "tp"),
    ),
    P2S1ShardingStrategy(
        name="row_contracting",
        mesh_shape=(4,),
        axis_names=("tp",),
        input_spec=(None, "tp"),
        weight_spec=("tp", None),
        bias_spec=(None,),
        output_spec=(None, None),
        collective_axis="tp",
        collective_kind="all_reduce",
    ),
    P2S1ShardingStrategy(
        name="two_d_output",
        mesh_shape=(2, 2),
        axis_names=("dp", "tp"),
        input_spec=("dp", None),
        weight_spec=(None, "tp"),
        bias_spec=("tp",),
        output_spec=("dp", "tp"),
    ),
    P2S1ShardingStrategy(
        name="two_d_contracting_output",
        mesh_shape=(2, 2, 2),
        axis_names=("dp", "tp", "mp"),
        input_spec=("dp", "mp"),
        weight_spec=("mp", "tp"),
        bias_spec=("tp",),
        output_spec=("dp", "tp"),
        collective_axis="mp",
        collective_kind="all_reduce",
    ),
    P2S1ShardingStrategy(
        name="partial_replication",
        mesh_shape=(2, 2),
        axis_names=("dp", "tp"),
        input_spec=(None, None),
        weight_spec=(None, "tp"),
        bias_spec=("tp",),
        output_spec=(None, "tp"),
    ),
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


def _make_sharded_smoke_module(
    torch_module: Any, size: int, strategy: P2S1ShardingStrategy
) -> Any:
    class WaferCaptureShardedSmoke4096(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch_module.nn.Parameter(
                torch_module.empty(size, size, dtype=torch_module.float32)
            )
            self.bias = torch_module.nn.Parameter(
                torch_module.empty(size, dtype=torch_module.float32)
            )

        def _mark(self, value, spec: tuple[str | None, ...]):
            return torch_module.ops.xla.dynamo_mark_sharding(
                value,
                list(range(strategy.device_count)),
                list(strategy.mesh_shape),
                repr(strategy.axis_names),
                repr(spec),
            )

        def forward(self, x):
            x = self._mark(x, strategy.input_spec)
            weight = self._mark(self.weight, strategy.weight_spec)
            bias = self._mark(self.bias, strategy.bias_spec)
            y = x @ weight
            y = y + bias
            z = torch_module.tanh(y)
            return z + y

    return WaferCaptureShardedSmoke4096()


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


def _axis_size(strategy: P2S1ShardingStrategy, axis_name: str) -> int:
    return strategy.mesh_shape[strategy.axis_names.index(axis_name)]


def _rank_coordinates(strategy: P2S1ShardingStrategy, rank: int) -> dict[str, int]:
    if rank < 0 or rank >= strategy.device_count:
        raise RuntimeError(
            f"global rank {rank} is outside mesh device count {strategy.device_count}"
        )
    coordinates: dict[str, int] = {}
    remainder = rank
    for axis, size in reversed(list(zip(strategy.axis_names, strategy.mesh_shape))):
        coordinates[axis] = remainder % size
        remainder //= size
    return coordinates


def _rank_from_coordinates(
    strategy: P2S1ShardingStrategy, coordinates: dict[str, int]
) -> int:
    rank = 0
    for axis, size in zip(strategy.axis_names, strategy.mesh_shape):
        rank = rank * size + coordinates[axis]
    return rank


def _rank_group(strategy: P2S1ShardingStrategy, global_rank: int) -> tuple[int, ...]:
    if strategy.collective_axis is None:
        return tuple(range(strategy.device_count))

    coordinates = _rank_coordinates(strategy, global_rank)
    group = []
    for index in range(_axis_size(strategy, strategy.collective_axis)):
        peer = dict(coordinates)
        peer[strategy.collective_axis] = index
        group.append(_rank_from_coordinates(strategy, peer))
    return tuple(group)


def _array_i64(values: Sequence[int]) -> str:
    return "array<i64: " + ", ".join(str(value) for value in values) + ">"


def _string_array(values: Sequence[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _tensor_type_from_shape(shape: Sequence[int], dtype: str) -> str:
    if dtype != "float32":
        raise RuntimeError(f"unsupported P2.S1 dtype in StableHLO bundle: {dtype}")
    return "tensor<" + "x".join(str(dim) for dim in shape) + "xf32>"


def _local_shape_and_offsets(
    global_shape: Sequence[int],
    spec: Sequence[str | None],
    strategy: P2S1ShardingStrategy,
    global_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(global_shape) != len(spec):
        raise RuntimeError(
            f"partition spec rank {len(spec)} does not match tensor rank {len(global_shape)}"
        )

    coordinates = _rank_coordinates(strategy, global_rank)
    local_shape: list[int] = []
    offsets: list[int] = []
    for dim_size, axis in zip(global_shape, spec):
        if axis is None:
            local_shape.append(dim_size)
            offsets.append(0)
            continue
        size = _axis_size(strategy, axis)
        if dim_size % size != 0:
            raise RuntimeError(
                f"tensor dimension {dim_size} is not divisible by mesh axis {axis}={size}"
            )
        local_dim = dim_size // size
        local_shape.append(local_dim)
        offsets.append(coordinates[axis] * local_dim)
    return tuple(local_shape), tuple(offsets)


def _sdy_sharding_attr(
    strategy: P2S1ShardingStrategy, spec: Sequence[str | None]
) -> str:
    used_axes = {axis for axis in spec if axis is not None}
    replicated_axes = [axis for axis in strategy.axis_names if axis not in used_axes]
    dim_shardings = [
        "{}" if axis is None else "{" + json.dumps(axis) + "}" for axis in spec
    ]
    attr = "#sdy.sharding<@mesh, [" + ", ".join(dim_shardings) + "]"
    if replicated_axes:
        attr += ", replicated={" + ", ".join(json.dumps(a) for a in replicated_axes) + "}"
    return attr + ">"


def _spmd_value_attrs(
    strategy: P2S1ShardingStrategy,
    global_shape: Sequence[int],
    dtype: str,
    spec: Sequence[str | None],
    global_rank: int,
) -> str:
    if dtype != "float32":
        raise RuntimeError(f"unsupported P2.S1 dtype in StableHLO bundle: {dtype}")
    local_shape, offsets = _local_shape_and_offsets(
        global_shape, spec, strategy, global_rank
    )
    return ", ".join(
        [
            f"sdy.sharding = {_sdy_sharding_attr(strategy, spec)}",
            f"wafer.spmd.global_shape = {_array_i64(global_shape)}",
            f"wafer.spmd.local_shape = {_array_i64(local_shape)}",
            f"wafer.spmd.shard_offsets = {_array_i64(offsets)}",
        ]
    )


def _module_attrs(strategy: P2S1ShardingStrategy, global_rank: int) -> str:
    rank_group = _rank_group(strategy, global_rank)
    local_rank = rank_group.index(global_rank)
    attrs = [
        f"wafer.spmd.strategy = {json.dumps(strategy.name)}",
        f"wafer.spmd.local_rank = {local_rank} : i64",
        f"wafer.spmd.global_rank = {global_rank} : i64",
        f"wafer.spmd.mesh_axes = {_string_array(strategy.axis_names)}",
        f"wafer.spmd.mesh_shape = {_array_i64(strategy.mesh_shape)}",
        f"wafer.spmd.rank_group = {_array_i64(rank_group)}",
    ]
    if strategy.collective_kind:
        attrs.append(
            f"wafer.spmd.collective_kind = {json.dumps(strategy.collective_kind)}"
        )
        attrs.append(
            f"wafer.spmd.collective_axis = {json.dumps(strategy.collective_axis)}"
        )
    return ", ".join(attrs)


def _mesh_op(strategy: P2S1ShardingStrategy) -> str:
    axes = ", ".join(
        f"{json.dumps(axis)}={size}"
        for axis, size in zip(strategy.axis_names, strategy.mesh_shape)
    )
    return f"  sdy.mesh @mesh = <[{axes}]>"


def _add_module_attrs(mlir: str, strategy: P2S1ShardingStrategy, global_rank: int) -> str:
    lines = mlir.splitlines()
    if not lines:
        raise RuntimeError("cannot annotate empty StableHLO MLIR")

    first = lines[0]
    attrs = _module_attrs(strategy, global_rank)
    if " attributes {" in first:
        first = first.replace(" attributes {", f" attributes {{{attrs}, ", 1)
    else:
        first = first.replace(" {", f" attributes {{{attrs}}} {{", 1)
    lines[0] = first
    if not any(line.lstrip().startswith("sdy.mesh @mesh") for line in lines):
        lines.insert(1, _mesh_op(strategy))
    return "\n".join(lines) + "\n"


def _annotate_function_boundary(
    mlir: str,
    meta: dict[str, Any],
    strategy: P2S1ShardingStrategy,
    global_rank: int,
) -> str:
    input_specs = [strategy.bias_spec, strategy.weight_spec, strategy.input_spec]
    signatures = meta["input_signature"]
    for index, (signature, spec) in enumerate(zip(signatures, input_specs)):
        tensor_type = _tensor_type_from_shape(signature["shape"], signature["dtype"])
        attrs = _spmd_value_attrs(
            strategy, signature["shape"], signature["dtype"], spec, global_rank
        )
        needle = f"%arg{index}: {tensor_type}"
        replacement = f"%arg{index}: {tensor_type} {{{attrs}}}"
        if needle not in mlir:
            raise RuntimeError(f"failed to find function argument for {needle}")
        mlir = mlir.replace(needle, replacement, 1)

    if len(meta["output_signature"]) != 1:
        raise RuntimeError("P2.S1 capture currently expects one output signature")
    output = meta["output_signature"][0]
    output_type = _tensor_type_from_shape(output["shape"], output["dtype"])
    output_attrs = _spmd_value_attrs(
        strategy, output["shape"], output["dtype"], strategy.output_spec, global_rank
    )
    result = f"({output_type} {{{output_attrs}}})"
    needle = f") -> {output_type} {{"
    if needle not in mlir:
        raise RuntimeError(f"failed to find function result for {output_type}")
    return mlir.replace(needle, f") -> {result} {{", 1)


def _annotate_bundle_for_p2s1(
    bundle_path: pathlib.Path, strategy: P2S1ShardingStrategy, global_rank: int
) -> None:
    meta_path = bundle_path / "functions" / "forward.meta"
    mlir_path = bundle_path / "functions" / "forward.mlir"
    meta = json.loads(meta_path.read_text())
    mlir = mlir_path.read_text()
    mlir = _add_module_attrs(mlir, strategy, global_rank)
    mlir = _annotate_function_boundary(mlir, meta, strategy, global_rank)
    mlir_path.write_text(mlir)


def _verify_frontend_marks(
    torch_module: Any, strategy: P2S1ShardingStrategy, size: int
) -> None:
    marked_module = _make_sharded_smoke_module(torch_module, size, strategy)
    marked_module.eval()
    input_tensor = torch_module.empty(size, size, dtype=torch_module.float32)
    with torch_module.no_grad():
        torch_module.export.export(marked_module, (input_tensor,))


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


def emit_p2s1_sharded_matmul_bundles(
    output_root: pathlib.Path,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
    size: int = P2F1_SMOKE_SIZE,
    global_rank: int = 0,
) -> list[EmittedP2S1Bundle]:
    if torch_module is None or stablehlo_module is None:
        torch_module, stablehlo_module = _import_runtime_modules()

    emitted: list[EmittedP2S1Bundle] = []
    for strategy in P2S1_SHARDING_STRATEGIES:
        if global_rank >= strategy.device_count:
            raise RuntimeError(
                f"global rank {global_rank} is outside {strategy.name} mesh device "
                f"count {strategy.device_count}"
            )
        _verify_frontend_marks(torch_module, strategy, size)
        bundle_path = output_root / strategy.name
        emit_p2f1_smoke_bundle(
            bundle_path=bundle_path,
            torch_module=torch_module,
            stablehlo_module=stablehlo_module,
            size=size,
        )
        _annotate_bundle_for_p2s1(bundle_path, strategy, global_rank)
        _verify_bundle_layout(bundle_path)
        rank_group = _rank_group(strategy, global_rank)
        emitted.append(
            EmittedP2S1Bundle(
                strategy=strategy.name,
                path=bundle_path,
                global_rank=global_rank,
                local_rank=rank_group.index(global_rank),
            )
        )
    return emitted


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-p2f1-smoke",
        action="store_true",
        help="emit the P2.F1 4096x4096 matmul+bias+tanh+residual artifact",
    )
    parser.add_argument(
        "--emit-p2s1-sharded-matmul",
        action="store_true",
        help="emit P2.S1 sharded 4096 matmul bundles for all required strategies",
    )
    parser.add_argument("--output-bundle", type=pathlib.Path)
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--global-rank", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.emit_p2f1_smoke:
        if args.output_bundle is None:
            raise RuntimeError("missing --output-bundle")

        emit_p2f1_smoke_bundle(args.output_bundle)
        return 0

    if args.emit_p2s1_sharded_matmul:
        if args.output_root is None:
            raise RuntimeError("missing --output-root")

        global_rank = args.global_rank
        if global_rank is None:
            global_rank = 0 if args.local_rank is None else args.local_rank
        emit_p2s1_sharded_matmul_bundles(
            output_root=args.output_root, global_rank=global_rank
        )
        return 0

    raise RuntimeError("missing --emit-p2f1-smoke or --emit-p2s1-sharded-matmul")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-capture: {error}", file=sys.stderr)
        raise SystemExit(1)
