# torch_txda PyTorch Runtime Wheel Analysis

This note records the static reverse engineering results for:

```text
/root/dlc_dev/torch_txda-0.1.0+20251230.71a1e5a6-cp310-cp310-manylinux2014_x86_64.whl
```

The wheel was extracted to `/tmp/torch_txda_wheel_analysis` for inspection. The
analysis used Python sources, bundled headers, ELF dynamic dependencies, symbol
tables, strings, and targeted disassembly of runtime/copy/stream/event paths.

The artifact metadata says:

- package: `torch-txda`
- version: `0.1.0+20251230.71a1e5a6`
- wheel tag: `cp310-cp310-linux_x86_64`
- archive hash: `sha256=b6aa5426e97edd35c1ed56001f70d3d94a320a90b9e3643d506822f05952a425`
- source URL embedded by pip: `TX8_SDK_1.9.0_Triton/pack/torch_txda-...whl`

The source path mentions a Triton SDK package, but this should not be treated as
the organizing model for the reverse engineering. The wheel itself is mainly a
PyTorch `PrivateUse1` backend and runtime adapter layer.

## 1. Main Conclusion

`torch_txda` implements a PyTorch backend named `txda` by registering PyTorch's
`PrivateUse1` device type. Its stack is:

```text
PyTorch Python compatibility layer
  -> torch_txda._C binding
  -> libtorch_txda.so
  -> tx_runtime + txdnn + hpgr + external txops package
```

This wheel is not the same layer as the low-level TX8 compiler runtime described
in `tx8-interface-contract.md`. It does not expose `TsmRun`, bootparam/dyn TLV
serialization, Kcore firmware loading, NCC packet wrappers, or direct DTE
programming APIs. Those remain the compiler/runtime contract recovered from
`tx8_deps`.

The useful new signal is that the vendor PyTorch runtime has a CUDA-like
`tx_runtime`/`txdnn` API layer above the lower TX8 runtime:

- device APIs: `txGetDeviceCount`, `txGetDevice`, `txSetDevice`,
  `txDeviceSynchronize`
- memory APIs: `txMalloc`, `txFree`, `txMemcpy`
- stream/event APIs: `txStreamCreate`, `txStreamQuery`,
  `txStreamSynchronize`, `txStreamWaitEvent`, `txEventCreate`,
  `txEventRecord`, `txEventQuery`, `txEventSynchronize`,
  `txEventElapsedTime`, `txEventDestroy`
- descriptor/op APIs: `txCreateTensorDescriptor`, `txSetTensorDescriptor`,
  `txDestroyTensorDescriptor`, and many `txdnn*` eager tensor kernels

That layer is useful for a PyTorch integration shim, eager bring-up, and host
memory management comparison. It should not replace the Wafer compiler package
launch path unless the missing `tx_runtime` headers/libraries later expose a
compatible compiled-model API.

## 2. Wheel Contents

Important files:

| file | role |
| --- | --- |
| `torch_txda/__init__.py` | Registers backend name `txda`, imports `txops`, registers device module, exposes top-level device/RNG helpers. |
| `torch_txda/txda/__init__.py` | CUDA-like runtime API: device, stream, synchronize, current/default stream. |
| `torch_txda/txda/streams.py` | Python `Stream`, `ExternalStream`, and `Event` wrappers over native base classes. |
| `torch_txda/transfer_to_txda.py` | Monkey-patches CUDA-oriented PyTorch code to use TXDA/flagcx. |
| `torch_txda/generator.py` | Default generator and `manual_seed_all` wrappers. |
| `torch_txda/autocast_utils.py` | AMP dtype policy for TXDA. |
| `torch_txda/_C.cpython-310-x86_64-linux-gnu.so` | Python extension binding layer. |
| `torch_txda/libtorch_txda.so` | Main native PyTorch backend implementation. |
| `torch_txda/include/*.h` | Bundled C++ headers for device, stream, event, guard, exception, dtype, and txdnn descriptor helpers. |

`txops` is imported but not bundled in this wheel. `libtorch_txda.so` also has a
RUNPATH pointing at `txops/lib`, so real operator coverage depends on an external
`txops` installation in addition to this wheel.

## 3. Native Dependencies

`_C.cpython-310-x86_64-linux-gnu.so` depends on:

- `libtorch_txda.so`
- `libtorch.so`
- `libtorch_cpu.so`
- `libtorch_python.so`

`libtorch_txda.so` depends on:

- `libhpgr.so`
- `libtxdnn.so`
- `libtorch.so`
- `libtorch_cpu.so`
- `libc10.so`
- `libtorch_python.so`

Its RUNPATH is:

```text
/usr/local/lib/python3.10/site-packages/torch/lib:
/usr/local/lib/python3.10/site-packages/txops/lib:
/usr/local/kuiper/lib
```

After the later `firmware_kuiper` pass, `libhpgr.so` and `tx_runtime.h` were
found under `/root/dlc_dev/firmware_kuiper/kuiper`.  `libtxdnn.so` and
`txdnn.h` are still not present in the workspace.  The conclusions below remain
static ABI/behavior findings, not board-executed validation.

## 4. Python-Facing Runtime API

Top-level `torch_txda/__init__.py` does the backend registration:

- imports `txops`
- imports `torch_txda._C`
- calls `rename_privateuse1_backend("txda")`
- calls `torch._register_device_module("txda", txda_module)`
- calls `generate_methods_for_privateuse1_backend(for_storage=True, for_tensor=True)`
- if devices are available, selects device 0 at import time

Top-level APIs:

| Python API | native binding / behavior |
| --- | --- |
| `torch.txda.set_device(device)` / `torch_txda.set_device` | `_txda_setDevice(device_index)` |
| `torch.txda.current_device()` | `_txda_getDevice()` |
| `torch.txda.device_count()` | `_txda_getDeviceCount()` |
| `torch.txda.is_available()` | `_txda_getDeviceCount() > 0` |
| `torch.txda.synchronize(device=None)` | `_txda_synchronize()` under optional device guard |
| `torch.txda.current_stream(device=None)` | `_txda_getCurrentStream(device_index)` and wraps `(stream_id, device_index, device_type)` |
| `torch.txda.default_stream(device=None)` | `_txda_getDefaultStream(device_index)` |
| `torch.txda.set_stream(stream)` | `_txda_setStream(stream_id, device_index, device_type)` |
| `torch.txda.stream(stream)` | context manager that swaps current stream and restores it |
| `torch_txda.txda_get_generator_by_device(device)` | native generator lookup |

`device_init`, `device_deinit`, `init_device`, and `cleanup_device` are Python
no-ops in this wheel. They should not be used as proof of hardware/device
initialization.

AMP/autocast support is narrow: `autocast_utils.py` advertises only
`torch.float16` and `torch.bfloat16` as supported autocast dtypes.

## 5. CUDA Compatibility Patching

`transfer_to_txda.py` is a compatibility shim for CUDA-oriented PyTorch code.
It is invasive and should be treated as application-layer patching, not a stable
compiler runtime ABI.

Key behavior:

- rewrites device strings and `torch.device("cuda:*")` to `txda`
- patches selected `torch.*`, `torch.Tensor.*`, and `torch.nn.Module.*` factory
  and transfer APIs so device kwargs become TXDA devices
- sets `torch.Tensor.cuda = torch.Tensor.txda`
- sets `torch.Tensor.is_cuda = torch.Tensor.is_txda`
- sets `torch.nn.Module.cuda = torch.nn.Module.txda`
- replaces `torch.cuda` with `torch_txda.txda` public APIs
- rewrites distributed backends from `nccl`/`NCCL` to `flagcx`
- wraps FSDP, DDP, `init_device_mesh`, `new_group`, and process-group backend
  lookups for CUDA-to-TXDA/flagcx conversion
- defaults `DataLoader(pin_memory=True)` to `pin_memory_device="txda"` if no
  pin-memory device is supplied
- disables `torch.jit.script` by returning the original object after a warning
- maps matching `torch._C` CUDA/CUDA symbols to TXDA/TXDA symbols when present
- sets `torch.cuda._lazy_init = torch.txda._lazy_init`
- rewrites `torch.serialization.default_restore_location`
- aliases `torch.cuda.Stream.cuda_stream` to `torch.txda.Stream.txda_stream`

This confirms the vendor's PyTorch story is mainly compatibility through device
renaming and monkey patching, not an upstream-native `torch.cuda` replacement.

## 6. Device, Allocator, and Copy Semantics

The native device API wraps CUDA-like `tx_runtime` calls:

| native layer | recovered behavior |
| --- | --- |
| `c10::txda::device_count()` | calls `txGetDeviceCount`; native result is cached in the runtime. |
| `c10::txda::is_available()` | true when device count is nonzero. |
| `c10::txda::current_device()` | calls `txGetDevice`. |
| `c10::txda::set_device()` | calls `txSetDevice`. |
| `c10::txda::device_synchronize()` | calls `txDeviceSynchronize`; this is device-wide, not stream-local. |

The allocator is registered as a PyTorch `PrivateUse1` allocator. Disassembly of
`TXDADeviceAllocator` shows:

- allocation uses the current `txGetDevice` result and calls `txMalloc` when
  `nbytes != 0`
- returned `DataPtr` is tagged with PyTorch device type `PrivateUse1`
- delete path calls `txFree`
- allocator `copy_data` uses `txMemcpy(..., kind=3)`, matching a device-to-device
  copy kind

Copy behavior from `txda__copy_from`:

- accepts CPU tensors and TXDA tensors; other backends are rejected
- contiguous same-shape paths use direct `txMemcpy`
- observed copy-kind constants match CUDA-style convention:
  - `1`: host to device
  - `2`: device to host
  - `3`: device to device
- non-contiguous or stride-changing copy paths build tensor descriptors and call
  `txdnnStrideCopy`
- dtype conversion paths call a native `type_convert_kernel` backed by
  `txdnnDtypeConvert`

Important limitation: a native error string says `only support data transfer
between cpu and txda`, but D2D code paths are present for TXDA tensors. Treat
the practical scope as CPU<->TXDA plus TXDA<->TXDA inside this backend, not
general cross-backend or arbitrary device transfer.

## 7. Stream and Event Semantics

`torch_txda` streams/events are host runtime queue primitives backed by
`tx_runtime`. They are not the same thing as the TX8 Kcore stream FSM/mailbox
register protocol documented in the register-level spec.

Native stream API:

| API | behavior |
| --- | --- |
| `getDefaultTXDAStream(device)` | returns stream id 0 for the device. |
| `getCurrentTXDAStream(device)` | returns the thread-local current stream for the device, defaulting to stream 0. |
| `setCurrentTXDAStream(stream)` | changes current stream for the stream's device; it does not change current device. |
| `getStreamFromPool(priority, device)` | returns lazily created streams from per-device pools. |
| `getStreamFromExternal(ext_stream, device)` | wraps an externally allocated `txStream_t`. |

The bundled `TXDAStream.h` describes three stream pools per device:

- default stream pool: only stream 0
- low/default-priority pool: 32 streams per device, round-robin
- high-priority pool: same round-robin shape

The 33rd stream request in one priority pool aliases the first stream in that
pool, so code that expects many long-lived independent streams can accidentally
serialize work.

Native event API:

| API | behavior |
| --- | --- |
| event creation | lazy; an event is created on first record. |
| `record(stream)` | creates the event if needed, then records on the stream. |
| `block(stream)` / Python `wait(stream)` | calls `txStreamWaitEvent`; only future stream work waits. |
| `query()` | calls `txEventQuery`; not-created events query as complete. |
| `elapsed_time(end_event)` | calls `txEventElapsedTime`. |
| `synchronize()` | calls `txEventSynchronize`. |
| destructor | switches to the event device and calls `txEventDestroy`. |

Design implication: these are useful for PyTorch eager stream ordering and host
queue dependencies. They do not express NCC packet ordering, Direct DTE receive
completion, Kcore mailbox completion, or `serial_mode`/SPM bank constraints.
Wafer's compiler runtime still needs explicit NCC/DTE/Kcore issue/drain rules.

## 8. txdnn Descriptor and Dtype Contract

`txda_init.h` maps PyTorch dtypes to `txdnn` dtypes:

| PyTorch dtype | txdnn dtype |
| --- | --- |
| `torch.bool` | `BOOL` |
| `torch.uint8` | `UINT8` |
| `torch.int8` | `INT8` |
| `torch.float16` | `FP16` |
| `torch.bfloat16` | `BF16` |
| `torch.int16` | `INT16` |
| `torch.int32` | `INT32` |
| `torch.float32` | `FP32` |
| `torch.int64` | `INT64` |

The helper macro uses `FP16` as a default fallback, and `IS_SUPPORTED_DTYPE`
rejects unmapped dtypes by checking whether a non-half dtype mapped to `FP16`.

Tensor descriptors are initialized as:

```c
txCreateTensorDescriptor(&desc);
txSetTensorDescriptor(
    desc,
    GET_TX_DTYPE(tensor.dtype().toScalarType()),
    txLayout_t::NHWC,
    shape.size(),
    shape.data(),
    strides.data());
```

A scalar non-empty tensor is represented with shape `{1}`. The descriptor passes
raw PyTorch shape and stride arrays while tagging layout as `NHWC`.

This is a high-level eager tensor descriptor convention. It is not evidence that
low-level TX8 SPM tensor layouts are NHWC. The Wafer SPM planner should continue
to use the register-level Cx/NCx/SPM layout constraints recovered from
`tx8_deps`; `txdnn` descriptors are only relevant to a PyTorch eager runtime
interop layer.

## 9. Native Operator Coverage

`libtorch_txda.so` registers a small eager operator set for
`PrivateUse1`/`AutogradPrivateUse1`. Exported symbols and strings show these
families:

| family | representative functions |
| --- | --- |
| binary arithmetic | `add_out`, `sub_out`, `mul_out`, `div_out`, `pow_out`, `pow_tensor_scalar_out` |
| comparisons | `eq_out`, `ne_out`, `lt_out`, `le_out`, `gt_out`, `ge_out`, `txda_equal` |
| unary math | `abs_out`, `ceil`, `floor`, `sin_out`, `cos_out`, `exp_out`, `neg_out`, `sqrt_out`, `rsqrt_out`, `recip_out`, `square_out` |
| movement / view | `empty`, `empty_strided`, `_copy_from`, `_copy_from_and_resize`, `type_convert_kernel`, `as_strided`, `view`, `_reshape_alias`, `detach`, `contiguous`, `resize_` |
| indexing / fill | `gather`, `scatter`, `split`, `split_with_sizes`, `fill_`, `zero_`, `masked_fill_.Scalar` |
| reductions / selection | `mean`, `count_nonzero`, `topk.values`, `sort.values_stable`, `local_scalar_dense` |
| random | `uniform_`, `normal_`, generator-by-device, seed setup |

Most real math work is delegated to `txdnn` calls. Undefined dynamic symbols
include:

```text
txdnnArith
txdnnComp
txdnnComp_Tensor_Scalar
txdnnEqual
txdnnDtypeConvert
txdnnStrideCopy
txdnnFillInplace
txdnnMaskFillInplace
txdnnMean
txdnnMeanByDim
txdnnCountNonzero
txdnnCountNonzeroDimInt
txdnnTopk
txdnnSort
txdnnSplit
txdnnAbs/Ceil/Floor/Sin/Cos/Exp/Neg/Sqrt/Rsqrt/Recip/Square
txdnnPow
txdnnPow_Tensor_Scalar
txdnnUniform
txdnnNormal_float_float
```

Observed limitations from strings/error paths:

- many paths require contiguous tensors
- `mean` supports only a single dim or all dims
- `masked_fill` warns that unsupported input datatype can fail
- random APIs require `txdnn*GetWorkspaceSize` and workspace allocation
- `topk`, `sort`, `split`, `count_nonzero`, and stride copy report `txdnn`
  failures directly through PyTorch checks

This operator set is useful for eager compatibility checks, but it is
too small and too opaque to be the Wafer compiler's operator lowering contract.

## 10. Fallback Environment Variables

`libtorch_txda.so` contains fallback/skip controls:

| variable | recovered role |
| --- | --- |
| `TXDA_FALLBACK_CPU_OPS` | parsed as a comma-separated set by `get_fallback_cpu_ops_from_env()`; used by custom CPU fallback paths. |
| `TXDA_SKIP_OPS` | parsed as a comma-separated set by `get_skip_ops_from_env()`; used by custom skip/override logic. |

The exact policy per operator still needs source-level confirmation or more
targeted control-flow recovery, but the ABI shape is clear: both are process
environment variables containing comma-separated operator names.

## 11. Relationship to Existing TX8 Runtime Findings

Do not conflate these layers:

| layer | source | role |
| --- | --- | --- |
| PyTorch TXDA backend | this wheel | PyTorch `PrivateUse1` device, eager ops, CUDA compatibility patching, host queue streams/events. |
| `tx_runtime` / HPGR | `firmware_kuiper` SDK | CUDA-like device/memory/stream/event API plus model/module/kernel/graph launch and HPGR command completion. |
| `txdnn` | linked by this wheel but not bundled or found locally | Eager tensor kernels and tensor descriptor API. |
| Host TX8 runtime | `tx8_deps` `libtx8_runtime.so` | device memory, bootparam, dyn TLV, `TsmRun`, D2D/P2P via Kcore programs, profiling. |
| Kcore/NCC/DTE layer | `tx8_deps` static libs and headers | instruction wrappers, Direct DTE, stream FSM/mailbox, PMU, reserved SPM. |
| KMD/driver | `firmware_kuiper` decrypted driver payload | BO/job/NPU/DTE/C2C/log/info/topology UAPI, BAR/ATU windows, PG maps, and firmware loading. |
| Wafer compiler runtime | our design | stable C ABI, SPM planner, package format, explicit issue/drain, verifier and golden tests. |

New design constraints implied by this wheel:

- A PyTorch frontend can plausibly expose `torch.txda` through PyTorch
  `PrivateUse1`, but that is separate from the compiled package ABI.
- `torch_txda` stream/event semantics are host queue semantics. They should not
  be used as a proof that NCC/CT/NE/DTE tasks have safe SPM reuse or bank-safe
  parallelism.
- `torch.txda.synchronize()` maps to `txDeviceSynchronize`, which is too coarse
  for normal compiled program scheduling. It is useful for debugging and host
  boundary fences only.  It should not be confused with HPGR model command-slot
  completion or KMD compute fences.
- The wheel's Python device init/cleanup hooks are no-ops, so initialization and
  teardown must be owned by the lower runtime adapter or deployment system.
- `txdnn` descriptor layout should not drive Wafer SPM layout. It is an eager
  tensor-library contract.
- The distributed runtime expects `flagcx` when CUDA/NCCL code is migrated.
  Wafer multi-card runtime design should keep a separate collective backend
  boundary rather than baking NCCL semantics into compiler IR.
- Environment-controlled CPU fallback is present; compiled Wafer correctness
  tests should avoid silently passing through CPU fallback unless the test
  explicitly opts into it.

## 12. What This Adds to Our Runtime Picture

The wheel confirms that there are at least two host-facing runtime surfaces in
the TX8/TXDA ecosystem:

1. A compiled-model/runtime surface recovered from `tx8_deps`
   (`TsmRun`, bootparam, dyn TLV, Kcore DTE, profiling).
2. A primary HPGR host runtime surface recovered from `firmware_kuiper`
   (`tx_runtime.h`, `libhpgr.so`, model/module/kernel/graph launch, streams,
   events, device memory).
3. A PyTorch eager surface recovered here (`PrivateUse1`, `tx_runtime`,
   `txdnn`, `txops`, streams/events, CUDA monkey patches).

For Wafer, the practical split should be:

- compiler package execution can target HPGR/`tx_runtime` when that SDK is
  available; the old `TsmRun`/bootparam/dyn-TLV/Kcore path remains compatibility
  and reverse-engineering evidence;
- PyTorch user-facing integration can use a `torch.txda` compatibility module or
  a similar `PrivateUse1` path;
- tests should explicitly state which layer is under test: PyTorch eager op,
  runtime allocation/copy, compiled package launch, Kcore instruction wrapper,
  or hardware register protocol.

## 13. Remaining Gaps

Static analysis of this wheel does not close these items:

- `txdnn.h` and `libtxdnn.so` are still absent locally, so exact txdnn enum
  values and full public eager-op signatures still need the missing package.
- The external `txops` package is required by import and RUNPATH, but was not
  bundled in the wheel.
- No low-level model launch API equivalent to HPGR `txLaunchModel*` or old
  `TsmRun` was found in the exposed Python/native symbols of this wheel.
- `TXDA_FALLBACK_CPU_OPS` and `TXDA_SKIP_OPS` policy is recovered at the
  environment-variable and parser level, but individual fallback decisions need
  more control-flow recovery or source.
- Board-level behavior for `txDeviceSynchronize`, stream concurrency, event
  timing, allocator failure, and txdnn workspace limits requires a target
  runtime environment.
