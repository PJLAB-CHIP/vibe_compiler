"""Product framework export entry for Wafer source programs."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any


def _capture_xla_module(torch: Any, stablehlo: Any, module: Any,
                        inputs: tuple[Any, ...]) -> Any:
    """Capture an original module; the output graph owns all port bindings."""
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as metrics
    from torch._decomp import get_decompositions
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    if any(child.training for child in module.modules()):
        raise ValueError("direct XLA export requires an eval module")
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
           for value in inputs):
        raise TypeError("direct XLA export inputs must be CPU tensors")
    if len({id(value) for value in inputs}) != len(inputs):
        raise ValueError("aliased input arguments are unsupported")

    def payload(value):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy().view("|V2")
        return tensor.numpy()

    def state(model):
        return dict((*model.named_parameters(), *model.named_buffers()))

    # Preserve the caller's CPU module, including buffers, aliases and Python
    # state. Nothing in the lazy graph may refer back to the caller's tensors.
    captured = copy.deepcopy(module)
    if any(not name or name in (".", "..") or "/" in name or "\\" in name
           for name in state(captured)):
        raise ValueError("module state name is not a safe data filename")
    state_payloads = {name: payload(value) for name, value in state(captured).items()}
    slots = [(table, name, value) for child in captured.modules()
             for table in (child._parameters, child._buffers)
             for name, value in table.items() if value is not None]
    device = xm.xla_device()
    xlac = torch_xla._XLAC
    special_scalars = xlac._get_xla_handle_special_scalars()
    try:
        xlac._set_xla_handle_special_scalars(False)
        captured.to(device)
        # CPU -> XLA Module._apply may replace a shared Parameter separately
        # at each registration. Preserve the actual pre-transfer aliases.
        transferred = {}
        for table, name, original in slots:
            table[name] = transferred.setdefault(id(original), table[name])
        xla_inputs = tuple(value.detach().clone().to(device) for value in inputs)
    finally:
        xlac._set_xla_handle_special_scalars(special_scalars)

    locations = {}
    protected = (*xla_inputs, *state(captured).values())
    versions = tuple(value._version for value in protected)
    identities = tuple(xlac._xla_get_tensor_id(value) for value in protected)
    for name, value in state(captured).items():
        locations[xlac._xla_get_tensor_id(value)] = stablehlo.InputLocation.parameter(name)
    for position, value in enumerate(xla_inputs):
        locations[xlac._xla_get_tensor_id(value)] = stablehlo.InputLocation.input_arg(position)
    state_before = state(captured)
    counters = {name: metrics.counter_value(name) for name in metrics.counter_names()}
    inference = torch.ops.aten._native_batch_norm_legit_no_training.default
    conditional_bn = (
        torch.ops.aten.native_batch_norm.default,
        torch.ops.aten._native_batch_norm_legit.default,
        torch.ops.aten._native_batch_norm_legit_functional.default,
    )
    decompositions = get_decompositions((
        inference, *conditional_bn, torch.ops.aten.gelu.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.native_layer_norm.default,
    ))
    native_layer_norm = torch.ops.aten.native_layer_norm.default
    decompose_layer_norm = decompositions[native_layer_norm]

    def cpu_layer_norm_results(input, *args, **kwargs):
        output, mean, rstd = decompose_layer_norm(input, *args, **kwargs)
        # Pinned torch._refs.native_layer_norm casts its statistics to input
        # dtype for CPU callers. Moving the original module to XLA must not
        # change these public result dtypes; internal opmath remains F32.
        return output, mean.to(input.dtype), rstd.to(input.dtype)

    decompositions[native_layer_norm] = cpu_layer_norm_results
    average_pools = (
        torch.ops.aten.avg_pool2d.default,
        torch.ops.aten._adaptive_avg_pool2d.default,
    )

    class CompositeOpmath(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            native = torch.ops.aten.native_layer_norm.default
            # CompositeImplicitAutograd can lower LayerNorm to a training
            # BatchNorm before Python dispatch sees it. Preserve the original
            # operator's official decomposition at its public call boundary.
            if func == native:
                return decompositions[native](*args, **kwargs)
            if func in (torch.layer_norm, torch.nn.functional.layer_norm,
                        torch.ops.aten.layer_norm.default):
                arguments = dict(zip(("input", "normalized_shape", "weight", "bias", "eps"), args))
                arguments.update(kwargs)
                return decompositions[native](
                    arguments["input"], arguments["normalized_shape"],
                    arguments.get("weight"), arguments.get("bias"),
                    arguments.get("eps", 1e-5),
                )[0]
            # The public adaptive operator takes a CompositeImplicitAutograd
            # mean fast path for a 1x1 result, before _adaptive_avg_pool2d is
            # dispatched. Preserve CPU half/BF16 accumulation on both paths.
            if func in (torch.nn.functional.adaptive_avg_pool2d,
                        torch.ops.aten.adaptive_avg_pool2d.default):
                input_name = ("self" if func == torch.ops.aten.adaptive_avg_pool2d.default
                              else "input")
                arguments = dict(zip((input_name, "output_size"), args))
                arguments.update(kwargs)
                value = arguments[input_name]
                if value.dtype in (torch.float16, torch.bfloat16):
                    return func(value.float(), arguments["output_size"]).to(value.dtype)
            return func(*args, **kwargs)

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten._local_scalar_dense.default:
                value = args[0]
                if value.device.type == "xla":
                    ids, _ = xlac._get_tensors_xla_device_data_node([value])
                    if ids:
                        raise RuntimeError("data-dependent scalar extraction during XLA export")
            if func == torch.ops.aten._to_copy.default:
                destination = kwargs.get("device")
                if (args[0].device.type == "xla" and destination is not None
                        and torch.device(destination).type != "xla"):
                    raise RuntimeError("XLA export cannot move graph tensors to the host")
            if func in decompositions:
                if func in conditional_bn and (args[5] if len(args) > 5
                                              else kwargs.get("training")) is not False:
                    raise RuntimeError("training BatchNorm is unsupported during XLA export")
                with self:
                    return decompositions[func](*args, **kwargs)
            if func in average_pools and args[0].dtype in (torch.float16, torch.bfloat16):
                return func(args[0].float(), *args[1:], **kwargs).to(args[0].dtype)
            if (func == torch.ops.aten.convolution.default and args[2] is not None
                    and args[0].dtype in (torch.float16, torch.bfloat16)):
                wide = tuple(value.float() for value in args[:3])
                return func(*wide, *args[3:], **kwargs).to(args[0].dtype)
            return func(*args, **kwargs)

    with torch.no_grad(), Capture(), CompositeOpmath():
        result = captured(*xla_inputs)
    if (metrics.counter_value("MarkStep") or 0) > (counters.get("MarkStep") or 0):
        raise RuntimeError("XLA export forward executed a graph step")
    outputs, _ = tree_flatten(result)
    if not outputs or any(not isinstance(value, torch.Tensor)
                          or value.device.type != "xla" for value in outputs):
        raise RuntimeError("XLA export outputs must be tensors in the captured graph")
    if (versions != tuple(value._version for value in protected)
            or identities != tuple(xlac._xla_get_tensor_id(value) for value in protected)):
        raise RuntimeError("XLA export does not support input or state mutation")
    state_after = state(captured)
    if (state_before.keys() != state_after.keys()
            or any(value is not state_after[name] for name, value in state_before.items())):
        raise RuntimeError("XLA export does not support replacing module state")
    fallbacks = [name for name in metrics.executed_fallback_ops()
                 if name != "aten::_local_scalar_dense"
                 and (metrics.counter_value(name) or 0) > (counters.get(name) or 0)]
    if fallbacks:
        raise RuntimeError(f"unsupported XLA CPU fallback operations: {sorted(fallbacks)}")

    ids, leaves = xlac._get_tensors_xla_device_data_node(outputs)
    bytecode = xm.get_stablehlo_bytecode(outputs)
    input_locations = []
    constants = []
    used_state = {}
    used_inputs = set()
    for identity, value in zip(ids, leaves):
        location = locations.get(identity)
        if location is None:
            location = stablehlo.InputLocation.constant(len(constants))
            constants.append(payload(value))
        elif location.type_ == stablehlo.VariableType.PARAMETER:
            used_state[location.name] = state_payloads[location.name]
        else:
            used_inputs.add(location.position)
        input_locations.append(location)
    if used_inputs != set(range(len(inputs))):
        raise RuntimeError("unused runtime inputs are unsupported by the source port contract")

    def signature(value):
        return stablehlo.VariableSignature(list(value.shape), str(value.dtype).removeprefix("torch."))

    meta = stablehlo.StableHLOFunctionMeta(
        name="forward", stablehlo_version="0.0.0",
        input_signature=[signature(value) for value in leaves],
        output_signature=[signature(value) for value in outputs],
        input_locations=input_locations, unused_inputs=[],
    )
    return stablehlo.StableHLOGraphModule(stablehlo.StableHLOModelBundle(
        state_dict=used_state, additional_constants=constants,
        stablehlo_funcs=[stablehlo.StableHLOFunc(meta, bytecode, None)],
    ))


def _finish_program_directory(directory: pathlib.Path) -> None:
    functions = directory / "functions"
    bytecode = functions / "forward.bytecode"
    portable = functions / "forward.stablehlo.bc"
    metadata = functions / "forward.meta"
    text = functions / "forward.mlir"
    if not bytecode.is_file() or not metadata.is_file():
        raise RuntimeError("PyTorch/XLA export omitted StableHLO or metadata")
    if portable.exists():
        raise RuntimeError("PyTorch/XLA export unexpectedly reused Wafer artifact name")
    os.replace(bytecode, portable)
    if text.exists():
        text.unlink()
    artifact = portable.read_bytes()
    if not artifact.startswith(b"ML\xefR") or b"StableHLO_v" not in artifact[:64]:
        raise RuntimeError("PyTorch/XLA bytecode is not a StableHLO portable artifact")
    with metadata.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("name") != "forward":
        raise RuntimeError("PyTorch/XLA export did not produce the forward entry")


def export_pytorch_program(
    module: Any, example_inputs: Any, output_directory: os.PathLike[str] | str
) -> pathlib.Path:
    """Export one static PyTorch module to the current Wafer source directory.

    The destination must not already exist. Target, search, runtime, oracle,
    workload and seed policy are intentionally not arguments of this API.
    """
    try:
        import torch
        import torch_xla._internal.custom_kernel  # noqa: F401
        from torch_xla import stablehlo
    except ImportError as error:
        raise RuntimeError(
            "the pinned PyTorch/XLA exporter runtime is unavailable"
        ) from error
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if isinstance(example_inputs, tuple):
        inputs = example_inputs
    elif isinstance(example_inputs, list):
        inputs = tuple(example_inputs)
    else:
        inputs = (example_inputs,)
    if not inputs:
        raise ValueError("example_inputs must not be empty")
    destination = pathlib.Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    program = _capture_xla_module(torch, stablehlo, module, inputs)

    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.wafer-export-",
                         dir=destination.parent)
    )
    try:
        program.save(str(temporary))
        _finish_program_directory(temporary)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = ["export_pytorch_program"]
