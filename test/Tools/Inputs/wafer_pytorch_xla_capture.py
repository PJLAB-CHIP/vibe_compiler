#!/usr/bin/env python3
"""PyTorch/XLA StableHLO program directory generator for Wafer frontend gates."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import math
import operator
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


DEFAULT_REFERENCE_MATMUL_SIZE = 4096
DEFAULT_HF_TRANSFORMER_BATCH_SIZE = 1
DEFAULT_HF_TRANSFORMER_SEQUENCE_LENGTH = 4
WORKLOAD_CORPUS_SCHEMA_VERSION = 1
DEFAULT_WORKLOAD_CORPUS_SPEC = (
    pathlib.Path(__file__).resolve().parent
    / "workloads"
    / "single-card-vertical-v1.json"
)
HF_CARD_PARTITION_MESH_SHAPE = (1,)
HF_CARD_PARTITION_AXIS_NAMES = ("card_partition",)
LLAMA_SCALE_PAYLOAD_ALGORITHM = (
    "wafer-exact-f16-splitmix64-counter-byte-scaled-v3"
)
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_STREAM_DOMAIN = 0xD1B54A32D192ED03
_SPLITMIX64_MULTIPLIER_0 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MULTIPLIER_1 = 0x94D049BB133111EB


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError(
            "the workload corpus reference requires NumPy in the importer "
            "Python environment"
        ) from error
    return numpy


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_array_bytes(
    numpy_module: Any, value: Any, *, require_float32: bool = True
) -> bytes:
    array = numpy_module.asarray(value)
    if require_float32 and array.dtype != numpy_module.dtype("float32"):
        raise RuntimeError(
            f"workload corpus only admits float32 arrays, got {array.dtype}"
        )
    if array.dtype.hasobject:
        raise RuntimeError("workload corpus does not admit object array payloads")
    canonical_dtype = (
        numpy_module.dtype("<f4")
        if require_float32
        else array.dtype.newbyteorder("<")
    )
    little_endian = numpy_module.ascontiguousarray(
        array.astype(canonical_dtype, copy=False)
    )
    header = _canonical_json_bytes(
        {
            "dtype": "float32" if require_float32 else canonical_dtype.str,
            "shape": [int(dim) for dim in array.shape],
        }
    )
    return len(header).to_bytes(8, "little") + header + little_endian.tobytes()


def _canonical_little_endian_array(numpy_module: Any, value: Any) -> Any:
    """Materialize one public NPY payload in canonical little-endian order."""
    array = numpy_module.asarray(value)
    original_shape = array.shape
    if array.dtype.hasobject:
        raise RuntimeError("workload corpus does not admit object array payloads")
    if array.dtype.kind == "V":
        # BF16 uses |V2 because NumPy has no portable public BF16 dtype. Its
        # producer already stores the raw 16-bit encoding little-endian.
        if array.dtype.itemsize != 2:
            raise RuntimeError(
                "workload corpus only admits two-byte opaque BF16 payloads"
            )
        return numpy_module.ascontiguousarray(array).reshape(original_shape)
    canonical_dtype = array.dtype.newbyteorder("<")
    return numpy_module.ascontiguousarray(
        array.astype(canonical_dtype, copy=False)
    ).reshape(original_shape)


def _save_workload_array(
    numpy_module: Any, destination: pathlib.Path, value: Any
) -> None:
    numpy_module.save(
        destination,
        _canonical_little_endian_array(numpy_module, value),
        allow_pickle=False,
    )


def _array_digest(
    numpy_module: Any, value: Any, *, require_float32: bool = True
) -> str:
    return _sha256_bytes(
        _canonical_array_bytes(
            numpy_module, value, require_float32=require_float32
        )
    )


def _named_array_digest(
    numpy_module: Any,
    values: dict[str, Any],
    *,
    require_float32: bool = True,
) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        canonical_array = _canonical_array_bytes(
            numpy_module, values[name], require_float32=require_float32
        )
        digest.update(len(canonical_array).to_bytes(8, "little"))
        digest.update(canonical_array)
    return "sha256:" + digest.hexdigest()


def _quantized_array_digest(
    numpy_module: Any, value: Any, decimals: int
) -> str:
    quantized = numpy_module.round(
        numpy_module.asarray(value, dtype=numpy_module.float32), decimals=decimals
    ).astype(numpy_module.float32)
    return _array_digest(numpy_module, quantized)


def _verify_finite_workload_array(
    numpy_module: Any,
    value: Any,
    *,
    case_id: str,
    value_name: str,
) -> None:
    array = numpy_module.asarray(value)
    is_bfloat16_storage = array.dtype.kind == "V" and array.dtype.itemsize == 2
    if not is_bfloat16_storage and array.dtype.kind not in {"f", "c"}:
        return

    flat = array.reshape(-1)
    chunk_elements = 1 << 20
    nonfinite_count = 0
    first_flat_index = None
    first_value = None
    for begin in range(0, int(flat.size), chunk_elements):
        chunk = flat[begin : begin + chunk_elements]
        finite_values = (
            _bfloat16_storage_to_float32(numpy_module, chunk)
            if is_bfloat16_storage
            else chunk
        )
        finite = numpy_module.isfinite(finite_values)
        chunk_nonfinite = int(numpy_module.count_nonzero(~finite))
        if chunk_nonfinite == 0:
            continue
        nonfinite_count += chunk_nonfinite
        if first_flat_index is None:
            first_in_chunk = int(numpy_module.flatnonzero(~finite)[0])
            first_flat_index = begin + first_in_chunk
            first_value = finite_values.reshape(-1)[first_in_chunk]
    if first_flat_index is None:
        return
    first_index = tuple(
        int(index)
        for index in numpy_module.unravel_index(first_flat_index, array.shape)
    )
    raise RuntimeError(
        f"workload case {case_id!r} {value_name} contains "
        f"{nonfinite_count} non-finite value(s); first_index={first_index}, "
        f"first_value={first_value!r}"
    )


def _verify_workload_payload_finite(
    numpy_module: Any,
    *,
    case_id: str,
    input_array: Any,
    extra_inputs: dict[int, Any],
    parameters: dict[str, Any],
    expected: Any,
) -> None:
    _verify_finite_workload_array(
        numpy_module,
        input_array,
        case_id=case_id,
        value_name="input 0",
    )
    for index, value in sorted(extra_inputs.items()):
        _verify_finite_workload_array(
            numpy_module,
            value,
            case_id=case_id,
            value_name=f"input {index}",
        )
    for name, value in sorted(parameters.items()):
        _verify_finite_workload_array(
            numpy_module,
            value,
            case_id=case_id,
            value_name=f"parameter {name!r}",
        )
    _verify_finite_workload_array(
        numpy_module,
        expected,
        case_id=case_id,
        value_name="expected output 0",
    )


def _load_llama_scale_payload_config(
    case_id: str, config: dict[str, Any]
) -> tuple[int, int, int]:
    payload = config.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"workload case {case_id!r} requires explicit config.payload"
        )
    required_fields = {
        "algorithm",
        "input_denominator",
        "projection_denominator",
        "normalization_delta_denominator",
    }
    actual_fields = set(payload)
    if actual_fields != required_fields:
        missing = sorted(required_fields - actual_fields)
        extra = sorted(actual_fields - required_fields)
        raise RuntimeError(
            f"workload case {case_id!r} config.payload fields mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if payload["algorithm"] != LLAMA_SCALE_PAYLOAD_ALGORITHM:
        raise RuntimeError(
            f"workload case {case_id!r} has unsupported payload algorithm "
            f"{payload['algorithm']!r}; expected "
            f"{LLAMA_SCALE_PAYLOAD_ALGORITHM!r}"
        )

    denominators = []
    for field in (
        "input_denominator",
        "projection_denominator",
        "normalization_delta_denominator",
    ):
        value = payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value & (value - 1)
        ):
            raise RuntimeError(
                f"workload case {case_id!r} config.payload.{field} must be "
                "a positive power-of-two integer"
            )
        denominators.append(value)
    return tuple(denominators)


def _verify_framework_cpu_output(
    actual: Any, expected: Any, *, rtol: float, atol: float
) -> None:
    try:
        _import_numpy().testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol
        )
    except AssertionError as error:
        raise RuntimeError(
            "framework CPU output does not match the independent NumPy reference"
        ) from error


def _deterministic_float32_array(
    numpy_module: Any,
    shape: tuple[int, ...],
    *,
    seed: int,
    stream: int,
    denominator: int,
) -> Any:
    if denominator <= 0 or denominator & (denominator - 1):
        raise RuntimeError("deterministic payload denominator must be a power of two")
    count = math.prod(shape)
    indices = numpy_module.arange(count, dtype=numpy_module.uint64)
    raw = (
        indices * numpy_module.uint64(1103515245)
        + numpy_module.uint64(seed) * numpy_module.uint64(2654435761)
        + numpy_module.uint64(stream) * numpy_module.uint64(2246822519)
    ) % numpy_module.uint64(257)
    signed = raw.astype(numpy_module.int32) - numpy_module.int32(128)
    return (
        signed.astype(numpy_module.float32) / numpy_module.float32(denominator)
    ).reshape(shape)


def _deterministic_float_storage_array(
    numpy_module: Any,
    shape: tuple[int, ...],
    *,
    seed: int,
    stream: int,
    denominator: int,
    dtype: str,
) -> Any:
    if dtype == "float32":
        return _deterministic_float32_array(
            numpy_module,
            shape,
            seed=seed,
            stream=stream,
            denominator=denominator,
        )
    if dtype != "float16":
        raise RuntimeError(
            f"deterministic Llama payload does not support dtype {dtype!r}"
        )
    if denominator <= 0 or denominator & (denominator - 1):
        raise RuntimeError("deterministic payload denominator must be a power of two")

    # A standard 7B block contains 90M-element projection matrices.  Fill the
    # final FP16 allocation in bounded chunks instead of constructing several
    # full-size uint64/int32/float32 temporaries alongside it.
    count = math.prod(shape)
    result = numpy_module.empty(count, dtype=numpy_module.float16)
    chunk_elements = 1 << 20
    for begin in range(0, count, chunk_elements):
        end = min(begin + chunk_elements, count)
        indices = numpy_module.arange(begin, end, dtype=numpy_module.uint64)
        raw = (
            indices * numpy_module.uint64(1103515245)
            + numpy_module.uint64(seed) * numpy_module.uint64(2654435761)
            + numpy_module.uint64(stream) * numpy_module.uint64(2246822519)
        ) % numpy_module.uint64(257)
        signed = raw.astype(numpy_module.int32) - numpy_module.int32(128)
        result[begin:end] = (
            signed.astype(numpy_module.float32)
            / numpy_module.float32(denominator)
        ).astype(numpy_module.float16)
    return result.reshape(shape)


def _splitmix64_final_scalar(value: int) -> int:
    value &= _UINT64_MASK
    value = (
        (value ^ (value >> 30)) * _SPLITMIX64_MULTIPLIER_0
    ) & _UINT64_MASK
    value = (
        (value ^ (value >> 27)) * _SPLITMIX64_MULTIPLIER_1
    ) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _deterministic_counter_float16_array(
    numpy_module: Any,
    shape: tuple[int, ...],
    *,
    seed: int,
    stream: int,
    denominator: int,
    chunk_elements: int = 1 << 20,
) -> Any:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > _UINT64_MASK
    ):
        raise RuntimeError("deterministic payload seed must fit uint64")
    if (
        isinstance(stream, bool)
        or not isinstance(stream, int)
        or stream < 0
        or stream >= _UINT64_MASK
    ):
        raise RuntimeError("deterministic payload stream must fit uint64 - 1")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
        or denominator & (denominator - 1)
        or denominator > (1 << 24)
    ):
        raise RuntimeError(
            "exact FP16 counter payload denominator must be a power of two "
            "not exceeding 2^24"
        )
    if (
        isinstance(chunk_elements, bool)
        or not isinstance(chunk_elements, int)
        or chunk_elements <= 0
    ):
        raise RuntimeError("deterministic payload chunk size must be positive")

    count = math.prod(shape)
    if count < 0 or count > _UINT64_MASK:
        raise RuntimeError("deterministic payload element count exceeds uint64")
    result = numpy_module.empty(count, dtype=numpy_module.float16)
    domain = ((stream + 1) * _SPLITMIX64_STREAM_DOMAIN) & _UINT64_MASK
    key = numpy_module.uint64(_splitmix64_final_scalar(seed ^ domain))
    multiplier_0 = numpy_module.uint64(_SPLITMIX64_MULTIPLIER_0)
    multiplier_1 = numpy_module.uint64(_SPLITMIX64_MULTIPLIER_1)
    for begin in range(0, count, chunk_elements):
        end = min(begin + chunk_elements, count)
        mixed = numpy_module.arange(begin, end, dtype=numpy_module.uint64) ^ key
        mixed ^= mixed >> numpy_module.uint64(30)
        mixed *= multiplier_0
        mixed ^= mixed >> numpy_module.uint64(27)
        mixed *= multiplier_1
        mixed ^= mixed >> numpy_module.uint64(31)
        code = (mixed >> numpy_module.uint64(56)).astype(numpy_module.int16)
        signed = code - numpy_module.int16(128)
        result[begin:end] = (
            signed.astype(numpy_module.float32)
            / numpy_module.float32(denominator)
        ).astype(numpy_module.float16)
    return result.reshape(shape)


def _float32_to_bfloat16_storage(numpy_module: Any, value: Any) -> Any:
    bits = numpy_module.ascontiguousarray(
        numpy_module.asarray(value, dtype=numpy_module.dtype("<f4"))
    ).view(numpy_module.dtype("<u4"))
    rounding = numpy_module.uint32(0x7FFF) + ((bits >> 16) & 1)
    rounded = ((bits + rounding) >> 16).astype(numpy_module.dtype("<u2"))
    return numpy_module.ascontiguousarray(rounded).view(numpy_module.dtype("|V2"))


def _bfloat16_storage_to_float32(numpy_module: Any, value: Any) -> Any:
    bits = numpy_module.ascontiguousarray(value).view(numpy_module.dtype("<u2"))
    widened = (bits.astype(numpy_module.dtype("<u4")) << 16).astype(
        numpy_module.dtype("<u4"), copy=False
    )
    return numpy_module.ascontiguousarray(widened).view(numpy_module.dtype("<f4"))


def _cast_gemm_storage(numpy_module: Any, value: Any, dtype: str) -> Any:
    if dtype == "float32":
        return numpy_module.asarray(value, dtype=numpy_module.float32)
    if dtype == "float16":
        return numpy_module.asarray(value, dtype=numpy_module.float16)
    if dtype == "bfloat16":
        return _float32_to_bfloat16_storage(numpy_module, value)
    raise RuntimeError(f"unsupported simple GEMM dtype {dtype!r}")


def _gemm_storage_to_float32(numpy_module: Any, value: Any, dtype: str) -> Any:
    if dtype == "bfloat16":
        return _bfloat16_storage_to_float32(numpy_module, value)
    return numpy_module.asarray(value, dtype=numpy_module.float32)


def _deterministic_gemm_array(
    numpy_module: Any,
    shape: tuple[int, ...],
    *,
    seed: int,
    stream: int,
) -> Any:
    count = math.prod(shape)
    indices = numpy_module.arange(count, dtype=numpy_module.uint64)
    raw = (
        indices * numpy_module.uint64(17)
        + numpy_module.uint64(seed) * numpy_module.uint64(13)
        + numpy_module.uint64(stream) * numpy_module.uint64(7)
    ) % numpy_module.uint64(9)
    signed = raw.astype(numpy_module.int32) - numpy_module.int32(4)
    return (
        signed.astype(numpy_module.float32) / numpy_module.float32(4.0)
    ).reshape(shape)


def _increasing_k_gemm_reference(
    numpy_module: Any, lhs: Any, rhs: Any, dtype: str
) -> Any:
    lhs_f32 = _gemm_storage_to_float32(numpy_module, lhs, dtype)
    rhs_f32 = _gemm_storage_to_float32(numpy_module, rhs, dtype)
    if lhs_f32.ndim != 2 or rhs_f32.ndim != 2:
        raise RuntimeError("simple GEMM reference currently requires rank-2 inputs")
    m, k = lhs_f32.shape
    rhs_k, n = rhs_f32.shape
    if k != rhs_k:
        raise RuntimeError("simple GEMM reference contracting dimensions differ")
    output = numpy_module.zeros((m, n), dtype=numpy_module.float32)
    for reduction in range(k):
        output = (
            output
            + lhs_f32[:, reduction : reduction + 1]
            * rhs_f32[reduction : reduction + 1, :]
        ).astype(numpy_module.float32)
    return _cast_gemm_storage(numpy_module, output, dtype)


def load_workload_corpus_spec(spec_path: pathlib.Path) -> dict[str, Any]:
    with spec_path.open("r", encoding="utf-8") as file:
        spec = json.load(file)
    if spec.get("schema_version") != WORKLOAD_CORPUS_SCHEMA_VERSION:
        raise RuntimeError(
            "unsupported workload corpus schema_version "
            f"{spec.get('schema_version')!r}; expected "
            f"{WORKLOAD_CORPUS_SCHEMA_VERSION}"
        )
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("workload corpus must contain a non-empty cases list")
    case_ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise RuntimeError("every workload corpus case must have a non-empty id")
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("workload corpus case ids must be unique")
    return spec


def _find_workload_case(spec: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in spec["cases"]:
        if case["id"] == case_id:
            return case
    valid = ", ".join(case["id"] for case in spec["cases"])
    raise RuntimeError(f"unknown workload corpus case '{case_id}'; valid: {valid}")


def _find_git_checkout(path: pathlib.Path) -> pathlib.Path | None:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _verify_workload_runtime_provenance(
    spec: dict[str, Any], torch_module: Any, torch_xla_module: Any
) -> None:
    source = spec.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("workload corpus is missing source provenance")
    framework = source.get("framework")
    exporter = source.get("exporter")
    if not isinstance(framework, dict) or not isinstance(exporter, dict):
        raise RuntimeError("workload corpus framework/exporter provenance is invalid")

    actual_torch_version = str(getattr(torch_module, "__version__", ""))
    actual_torch_revision = str(
        getattr(getattr(torch_module, "version", None), "git_version", "")
    )
    if actual_torch_version != framework.get("version"):
        raise RuntimeError(
            "workload corpus PyTorch version mismatch: "
            f"expected {framework.get('version')!r}, got {actual_torch_version!r}"
        )
    if actual_torch_revision != framework.get("git_revision"):
        raise RuntimeError(
            "workload corpus PyTorch git revision mismatch: "
            f"expected {framework.get('git_revision')!r}, "
            f"got {actual_torch_revision!r}"
        )

    actual_xla_version = str(getattr(torch_xla_module, "__version__", ""))
    if actual_xla_version != exporter.get("version"):
        raise RuntimeError(
            "workload corpus PyTorch/XLA version mismatch: "
            f"expected {exporter.get('version')!r}, got {actual_xla_version!r}"
        )
    module_path_value = getattr(torch_xla_module, "__file__", None)
    if not isinstance(module_path_value, str):
        raise RuntimeError("cannot locate PyTorch/XLA source checkout")
    checkout = _find_git_checkout(pathlib.Path(module_path_value))
    if checkout is None:
        raise RuntimeError(
            "PyTorch/XLA workload exporter is not loaded from a git source checkout"
        )
    try:
        actual_xla_revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        actual_xla_origin = subprocess.run(
            ["git", "-C", str(checkout), "remote", "get-url", "origin"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        xla_status = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot read PyTorch/XLA source git revision") from error
    if actual_xla_revision != exporter.get("git_revision"):
        raise RuntimeError(
            "workload corpus PyTorch/XLA git revision mismatch: "
            f"expected {exporter.get('git_revision')!r}, "
            f"got {actual_xla_revision!r}"
        )
    if actual_xla_origin != exporter.get("git_repository"):
        raise RuntimeError(
            "workload corpus PyTorch/XLA git origin mismatch: "
            f"expected {exporter.get('git_repository')!r}, "
            f"got {actual_xla_origin!r}"
        )
    if xla_status:
        raise RuntimeError(
            "workload corpus PyTorch/XLA source checkout is dirty; "
            "cannot attest the pinned exporter revision"
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
    bias = getattr(reference_module, "bias", None)
    if bias is not None:
        spmd_module.mark_sharding(bias, mesh, strategy.bias_spec)


def create_hf_card_partition_mesh(spmd_module: Any) -> Any:
    return spmd_module.Mesh(
        [0],
        HF_CARD_PARTITION_MESH_SHAPE,
        HF_CARD_PARTITION_AXIS_NAMES,
    )


def apply_hf_card_partition_marks(
    *,
    spmd_module: Any,
    mesh: Any,
    input_tensor: Any,
    reference_module: Any,
) -> None:
    # The source boundary has one logical card partition. Replicated marks
    # keep parameters explicit for the program-directory ABI without encoding
    # any physical-Tile placement or operator-specific partitioning.
    spmd_module.mark_sharding(
        input_tensor, mesh, tuple(None for _ in input_tensor.shape)
    )
    for parameter in reference_module.parameters():
        spmd_module.mark_sharding(
            parameter, mesh, tuple(None for _ in parameter.shape)
        )


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


def _make_linear_residual_mlp_module(
    torch_module: Any, parameters: dict[str, Any]
) -> Any:
    class WaferLinearResidualMlp(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_weight = torch_module.nn.Parameter(
                torch_module.from_numpy(parameters["first_weight"].copy())
            )
            self.first_bias = torch_module.nn.Parameter(
                torch_module.from_numpy(parameters["first_bias"].copy())
            )
            self.second_weight = torch_module.nn.Parameter(
                torch_module.from_numpy(parameters["second_weight"].copy())
            )
            self.second_bias = torch_module.nn.Parameter(
                torch_module.from_numpy(parameters["second_bias"].copy())
            )

        def forward(self, x):
            hidden = torch_module.tanh(x @ self.first_weight + self.first_bias)
            projection = hidden @ self.second_weight + self.second_bias
            return projection + x

    return WaferLinearResidualMlp()


def _make_simple_gemm_module(torch_module: Any) -> Any:
    class WaferSimpleGemm(torch_module.nn.Module):
        def forward(self, lhs, rhs):
            return lhs @ rhs

    return WaferSimpleGemm()


def _torch_from_workload_storage(
    torch_module: Any, numpy_module: Any, value: Any, dtype: str
) -> Any:
    if dtype == "bfloat16":
        raw = numpy_module.ascontiguousarray(value).view(
            numpy_module.dtype("<u2")
        )
        raw = numpy_module.ascontiguousarray(
            raw.astype(numpy_module.dtype("=u2"), copy=False)
        )
        return torch_module.from_numpy(raw.copy()).view(torch_module.bfloat16)
    return torch_module.from_numpy(numpy_module.ascontiguousarray(value).copy())


def _torch_to_workload_storage(
    torch_module: Any, numpy_module: Any, value: Any, dtype: str
) -> Any:
    cpu = value.detach().cpu()
    if dtype == "bfloat16":
        raw = numpy_module.ascontiguousarray(
            cpu.view(torch_module.uint16).numpy()
        ).astype(numpy_module.dtype("<u2"), copy=False)
        return numpy_module.ascontiguousarray(raw).view(
            numpy_module.dtype("|V2")
        )
    return cpu.numpy()


def _torch_tensor_to_workload_storage(torch_module: Any, value: Any) -> Any:
    """Convert any Torch tensor to its canonical public NPY representation."""
    numpy_module = _import_numpy()
    dtype = str(value.dtype).replace("torch.", "")
    return _canonical_little_endian_array(
        numpy_module,
        _torch_to_workload_storage(
            torch_module, numpy_module, value, dtype
        ),
    )


def _assign_named_parameter_arrays(
    torch_module: Any, reference_module: Any, parameters: dict[str, Any]
) -> None:
    named_parameters = dict(_named_parameters(reference_module))
    if set(named_parameters) != set(parameters):
        missing = sorted(set(named_parameters) - set(parameters))
        extra = sorted(set(parameters) - set(named_parameters))
        raise RuntimeError(
            "workload parameter set does not match model parameters: "
            f"missing={missing}, extra={extra}"
        )
    with torch_module.no_grad():
        for name, parameter in named_parameters.items():
            source = torch_module.from_numpy(parameters[name].copy())
            if tuple(parameter.shape) != tuple(source.shape):
                raise RuntimeError(
                    f"workload parameter shape mismatch for {name}: "
                    f"model={tuple(parameter.shape)}, source={tuple(source.shape)}"
                )
            parameter.copy_(source)


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
            "grouped-query attention is not part of this card-local compiler gate"
        )

    return config


def _make_hf_llama_decoder_block_module(
    torch_module: Any,
    config: dict[str, Any],
    sequence_length: int,
    parameter_arrays: dict[str, Any] | None = None,
) -> Any:
    """Build a thin export adapter over the installed official HF Llama layer.

    Hugging Face owns every mathematical operation in the block.  This
    adapter only freezes the official RoPE/mask outputs for the fixture's
    static sequence. Card-local spatial and temporal scheduling remain wholly
    compiler-owned after export.
    """

    if sequence_length <= 0:
        raise RuntimeError("--sequence-length must be positive")

    try:
        from transformers import LlamaConfig
        from transformers.masking_utils import create_causal_mask
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRotaryEmbedding,
        )
    except ImportError as error:
        raise RuntimeError(
            "the Llama block fixture requires the pinned Hugging Face "
            "Transformers dependency"
        ) from error

    decoder_parameters = inspect.signature(
        LlamaDecoderLayer.forward
    ).parameters
    required_decoder_features = {
        "attention_mask",
        "position_ids",
        "position_embeddings",
        "use_cache",
    }
    missing_decoder_features = sorted(
        required_decoder_features - set(decoder_parameters)
    )
    mask_parameters = inspect.signature(create_causal_mask).parameters
    required_mask_features = {
        "config",
        "inputs_embeds",
        "attention_mask",
        "past_key_values",
        "position_ids",
    }
    missing_mask_features = sorted(
        required_mask_features - set(mask_parameters)
    )
    if missing_decoder_features or missing_mask_features:
        raise RuntimeError(
            "the installed Hugging Face Llama API cannot express the "
            "fixture contract: "
            f"decoder_missing={missing_decoder_features} "
            f"mask_missing={missing_mask_features}"
        )

    hidden_size = int(config["hidden_size"])
    storage_dtype_name = str(config.get("torch_dtype", "float32"))
    if storage_dtype_name in {"float16", "torch.float16"}:
        storage_dtype = torch_module.float16
    elif storage_dtype_name in {"bfloat16", "torch.bfloat16"}:
        storage_dtype = torch_module.bfloat16
    elif storage_dtype_name in {"float32", "torch.float32"}:
        storage_dtype = torch_module.float32
    else:
        raise RuntimeError(
            "Llama decoder block test emitter supports torch_dtype float16, "
            f"bfloat16, or float32, got {storage_dtype_name!r}"
        )

    hf_config = LlamaConfig.from_dict(copy.deepcopy(config))
    hf_config._attn_implementation = "eager"
    hf_config.use_cache = False

    decoder_init_parameters = inspect.signature(
        LlamaDecoderLayer.__init__
    ).parameters
    decoder_init_kwargs = (
        {"layer_idx": 0} if "layer_idx" in decoder_init_parameters else {}
    )

    class HuggingFaceLlamaDecoderBlockAdapter(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder_layer = LlamaDecoderLayer(
                hf_config, **decoder_init_kwargs
            ).to(dtype=storage_dtype)

            positions = torch_module.arange(
                sequence_length, dtype=torch_module.long
            ).unsqueeze(0)
            rotary_input = torch_module.empty(
                (1, sequence_length, hidden_size), dtype=storage_dtype
            )
            rotary = LlamaRotaryEmbedding(hf_config).eval()
            with torch_module.no_grad():
                rotary_cos, rotary_sin = rotary(rotary_input, positions)
                causal_mask = create_causal_mask(
                    config=hf_config,
                    inputs_embeds=rotary_input,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=positions,
                )
            if not isinstance(causal_mask, torch_module.Tensor):
                raise RuntimeError(
                    "the official Hugging Face eager Llama mask path did not "
                    "produce an additive tensor mask"
                )
            self.register_buffer("position_ids", positions, persistent=False)
            self.register_buffer(
                "rotary_cos", rotary_cos, persistent=False
            )
            self.register_buffer(
                "rotary_sin", rotary_sin, persistent=False
            )
            self.register_buffer(
                "causal_mask", causal_mask, persistent=False
            )

        def wafer_named_parameters(self):
            layer = self.decoder_layer
            return (
                ("input_layernorm.weight", layer.input_layernorm.weight),
                (
                    "post_attention_layernorm.weight",
                    layer.post_attention_layernorm.weight,
                ),
                ("q_proj.weight", layer.self_attn.q_proj.weight),
                ("k_proj.weight", layer.self_attn.k_proj.weight),
                ("v_proj.weight", layer.self_attn.v_proj.weight),
                ("o_proj.weight", layer.self_attn.o_proj.weight),
                ("gate_proj.weight", layer.mlp.gate_proj.weight),
                ("up_proj.weight", layer.mlp.up_proj.weight),
                ("down_proj.weight", layer.mlp.down_proj.weight),
            )

        def forward(self, hidden_states):
            return self.decoder_layer(
                hidden_states,
                attention_mask=self.causal_mask,
                position_ids=self.position_ids,
                position_embeddings=(self.rotary_cos, self.rotary_sin),
                use_cache=False,
            )

    module = HuggingFaceLlamaDecoderBlockAdapter()
    if parameter_arrays is not None:
        named_parameters = dict(module.wafer_named_parameters())
        if set(named_parameters) != set(parameter_arrays):
            missing = sorted(set(named_parameters) - set(parameter_arrays))
            extra = sorted(set(parameter_arrays) - set(named_parameters))
            raise RuntimeError(
                "workload parameter set does not match the official HF "
                f"decoder layer: missing={missing}, extra={extra}"
            )
        with torch_module.no_grad():
            for name, parameter in named_parameters.items():
                source = torch_module.from_numpy(
                    parameter_arrays[name].copy()
                )
                if tuple(parameter.shape) != tuple(source.shape):
                    raise RuntimeError(
                        f"workload parameter shape mismatch for {name}: "
                        f"model={tuple(parameter.shape)}, "
                        f"source={tuple(source.shape)}"
                    )
                parameter.copy_(source)
    return module


def _linear_residual_mlp_cpu_reference(
    numpy_module: Any,
    input_array: Any,
    parameters: dict[str, Any],
) -> Any:
    hidden = numpy_module.tanh(
        input_array @ parameters["first_weight"] + parameters["first_bias"]
    ).astype(numpy_module.float32)
    projection = (
        hidden @ parameters["second_weight"] + parameters["second_bias"]
    ).astype(numpy_module.float32)
    return (projection + input_array).astype(numpy_module.float32)


def _llama_decoder_block_cpu_reference(
    numpy_module: Any,
    hidden_states: Any,
    parameters: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    float32 = numpy_module.float32
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))
    rms_norm_eps = float32(config["rms_norm_eps"])
    rope_theta = float32(config.get("rope_theta", 10000.0))

    def rms_norm(value: Any, weight: Any) -> Any:
        variance = numpy_module.mean(
            value * value, axis=-1, keepdims=True, dtype=float32
        )
        normalized = value * numpy_module.reciprocal(
            numpy_module.sqrt(variance + rms_norm_eps).astype(float32)
        )
        return (normalized * weight).astype(float32)

    def apply_linear_layer(value: Any, name: str) -> Any:
        return (value @ parameters[f"{name}.weight"].T).astype(float32)

    def shape_projection(value: Any) -> Any:
        batch, sequence_length, _ = value.shape
        return value.reshape(
            batch, sequence_length, num_attention_heads, head_dim
        ).transpose(0, 2, 1, 3)

    def rotate_half(value: Any) -> Any:
        first_half = value[..., : head_dim // 2]
        second_half = value[..., head_dim // 2 :]
        return numpy_module.concatenate((-second_half, first_half), axis=-1)

    sequence_length = int(hidden_states.shape[1])
    positions = numpy_module.arange(sequence_length, dtype=float32)
    exponents = (
        numpy_module.arange(0, head_dim, 2, dtype=float32) / float32(head_dim)
    )
    inv_freq = numpy_module.reciprocal(
        numpy_module.power(rope_theta, exponents).astype(float32)
    )
    frequencies = numpy_module.outer(positions, inv_freq).astype(float32)
    embedding = numpy_module.concatenate((frequencies, frequencies), axis=-1)
    cosine = numpy_module.cos(embedding).astype(float32)[None, None, :, :]
    sine = numpy_module.sin(embedding).astype(float32)[None, None, :, :]

    residual = hidden_states
    normed_states = rms_norm(
        hidden_states, parameters["input_layernorm.weight"]
    )
    query = shape_projection(apply_linear_layer(normed_states, "q_proj"))
    key = shape_projection(apply_linear_layer(normed_states, "k_proj"))
    value = shape_projection(apply_linear_layer(normed_states, "v_proj"))
    query = (query * cosine + rotate_half(query) * sine).astype(float32)
    key = (key * cosine + rotate_half(key) * sine).astype(float32)
    attention_scores = (
        query @ key.transpose(0, 1, 3, 2)
    ).astype(float32) * float32(1.0 / math.sqrt(head_dim))
    causal_mask = numpy_module.triu(
        numpy_module.ones(
            (sequence_length, sequence_length), dtype=numpy_module.bool_
        ),
        k=1,
    )
    attention_scores = numpy_module.where(
        causal_mask[None, None, :, :],
        float32(-numpy_module.inf),
        attention_scores,
    ).astype(float32)
    shifted_scores = attention_scores - numpy_module.max(
        attention_scores, axis=-1, keepdims=True
    )
    attention_exp = numpy_module.exp(shifted_scores).astype(float32)
    attention_weights = (
        attention_exp
        / numpy_module.sum(attention_exp, axis=-1, keepdims=True, dtype=float32)
    ).astype(float32)
    attention_output = (attention_weights @ value).astype(float32)
    batch, _, sequence_length, _ = attention_output.shape
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
        batch, sequence_length, hidden_size
    )
    hidden_states = (
        residual + apply_linear_layer(attention_output, "o_proj")
    ).astype(float32)

    residual = hidden_states
    normed_states = rms_norm(
        hidden_states, parameters["post_attention_layernorm.weight"]
    )
    gated = apply_linear_layer(normed_states, "gate_proj")
    up = apply_linear_layer(normed_states, "up_proj")
    silu = (
        gated / (float32(1.0) + numpy_module.exp(-gated).astype(float32))
    ).astype(float32)
    mlp_output = (silu * up).astype(float32)
    if mlp_output.shape[-1] != intermediate_size:
        raise RuntimeError("Llama CPU reference intermediate shape mismatch")
    return (residual + apply_linear_layer(mlp_output, "down_proj")).astype(float32)


def _build_workload_case_payload(
    spec_path: pathlib.Path,
    case: dict[str, Any],
    numpy_module: Any,
    *,
    verify_fixed_digests: bool = True,
) -> dict[str, Any]:
    if case.get("dtype") not in {"float16", "bfloat16", "float32"}:
        raise RuntimeError(
            f"workload case {case['id']} has unsupported dtype {case.get('dtype')!r}"
        )
    seed = int(case["seed"])
    kind = case.get("kind")
    config = case.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"workload case {case['id']} is missing config")

    extra_inputs: dict[int, Any] = {}
    if kind == "simple_gemm":
        dtype = str(case["dtype"])
        m = int(config["m"])
        k = int(config["k"])
        n = int(config["n"])
        if min(m, k, n) <= 0:
            raise RuntimeError("simple GEMM dimensions must be positive")
        if config.get("model_semantics_revision") != "wafer-simple-gemm-v1":
            raise RuntimeError("unsupported simple GEMM semantics revision")
        source_config = config
        lhs_f32 = _deterministic_gemm_array(
            numpy_module, (m, k), seed=seed, stream=0
        )
        rhs_f32 = _deterministic_gemm_array(
            numpy_module, (k, n), seed=seed, stream=1
        )
        input_array = _cast_gemm_storage(numpy_module, lhs_f32, dtype)
        rhs_array = _cast_gemm_storage(numpy_module, rhs_f32, dtype)
        extra_inputs[1] = rhs_array
        parameters = {}
        expected = _increasing_k_gemm_reference(
            numpy_module, input_array, rhs_array, dtype
        )
    elif kind == "linear_residual_mlp":
        batch_size = int(config["batch_size"])
        input_features = int(config["input_features"])
        hidden_features = int(config["hidden_features"])
        output_features = int(config["output_features"])
        if min(batch_size, input_features, hidden_features, output_features) <= 0:
            raise RuntimeError("linear/MLP workload dimensions must be positive")
        if output_features != input_features:
            raise RuntimeError(
                "linear residual workload requires output_features == input_features"
            )
        if config.get("model_semantics_revision") != "wafer-linear-mlp-v1":
            raise RuntimeError("unsupported linear/MLP model semantics revision")
        source_config = config
        input_array = _deterministic_float32_array(
            numpy_module,
            (batch_size, input_features),
            seed=seed,
            stream=0,
            denominator=128,
        )
        parameters = {
            "first_weight": _deterministic_float32_array(
                numpy_module,
                (input_features, hidden_features),
                seed=seed,
                stream=1,
                denominator=4096,
            ),
            "first_bias": _deterministic_float32_array(
                numpy_module,
                (hidden_features,),
                seed=seed,
                stream=2,
                denominator=4096,
            ),
            "second_weight": _deterministic_float32_array(
                numpy_module,
                (hidden_features, output_features),
                seed=seed,
                stream=3,
                denominator=4096,
            ),
            "second_bias": _deterministic_float32_array(
                numpy_module,
                (output_features,),
                seed=seed,
                stream=4,
                denominator=4096,
            ),
        }
        expected = _linear_residual_mlp_cpu_reference(
            numpy_module, input_array, parameters
        )
    elif kind in {"tiny_llama_decoder_block", "llama_decoder_block"}:
        config_path_value = config.get("hf_config")
        if not isinstance(config_path_value, str):
            raise RuntimeError("Llama workload requires config.hf_config")
        config_path = (spec_path.parent / config_path_value).resolve()
        if config.get("causal_attention") is not True:
            raise RuntimeError("Llama decoder workload requires causal attention")
        hf_config = load_hf_transformer_config(config_path)
        source_config = {
            "hf_config": hf_config,
            "workload_config": config,
            "model_semantics_revision": "wafer-causal-llama-decoder-v1",
        }
        batch_size = int(config["batch_size"])
        sequence_length = int(config["sequence_length"])
        hidden_size = int(hf_config["hidden_size"])
        intermediate_size = int(hf_config["intermediate_size"])
        if batch_size <= 0 or sequence_length <= 0:
            raise RuntimeError("Llama batch and sequence dimensions must be positive")
        storage_dtype = str(case["dtype"])
        config_storage_dtype = str(hf_config.get("torch_dtype", "float32"))
        if config_storage_dtype != storage_dtype:
            raise RuntimeError(
                "Llama workload dtype must match hf_config torch_dtype: "
                f"case={storage_dtype!r}, config={config_storage_dtype!r}"
            )
        if kind == "llama_decoder_block":
            (
                input_denominator,
                projection_denominator,
                normalization_delta_denominator,
            ) = _load_llama_scale_payload_config(case["id"], config)
        else:
            # The frozen tiny diagnostic corpus predates the scale-payload
            # schema. Its existing source digest and values remain unchanged.
            input_denominator = 128
            projection_denominator = 4096
            normalization_delta_denominator = 4096
        if kind == "llama_decoder_block" and storage_dtype != "float16":
            raise RuntimeError(
                "the versioned Llama scale payload requires float16 storage"
            )
        payload_array = (
            _deterministic_counter_float16_array
            if kind == "llama_decoder_block"
            else functools.partial(
                _deterministic_float_storage_array, dtype=storage_dtype
            )
        )
        input_array = payload_array(
            numpy_module,
            (batch_size, sequence_length, hidden_size),
            seed=seed,
            stream=0,
            denominator=input_denominator,
        )
        parameter_specs = (
            (1, "input_layernorm.weight", (hidden_size,)),
            (2, "post_attention_layernorm.weight", (hidden_size,)),
            (3, "q_proj.weight", (hidden_size, hidden_size)),
            (4, "k_proj.weight", (hidden_size, hidden_size)),
            (5, "v_proj.weight", (hidden_size, hidden_size)),
            (6, "o_proj.weight", (hidden_size, hidden_size)),
            (7, "gate_proj.weight", (intermediate_size, hidden_size)),
            (8, "up_proj.weight", (intermediate_size, hidden_size)),
            (9, "down_proj.weight", (hidden_size, intermediate_size)),
        )
        parameters = {}
        for stream, name, shape in parameter_specs:
            denominator = (
                normalization_delta_denominator
                if name.endswith("layernorm.weight")
                else projection_denominator
            )
            value = payload_array(
                numpy_module,
                shape,
                seed=seed,
                stream=stream,
                denominator=denominator,
            )
            if name.endswith("layernorm.weight"):
                value = (
                    numpy_module.asarray(1.0, dtype=value.dtype) + value
                ).astype(value.dtype)
            parameters[name] = value
        if kind == "llama_decoder_block":
            # The scale corpus deliberately uses the complete PyTorch eager
            # block output as the external truth.  The hand-written NumPy
            # implementation stays available for the tiny diagnostic case but
            # cannot generate this scale gate's expected.npy.
            torch_module, _ = _import_runtime_modules()
            reference_module = _make_hf_llama_decoder_block_module(
                torch_module,
                hf_config,
                sequence_length,
                parameter_arrays=parameters,
            )
            reference_module.eval()
            with torch_module.no_grad():
                expected_tensor = reference_module(
                    torch_module.from_numpy(input_array.copy())
                )
            expected = _torch_tensor_to_workload_storage(
                torch_module, expected_tensor
            )
        else:
            expected = _llama_decoder_block_cpu_reference(
                numpy_module, input_array, parameters, hf_config
            )
    else:
        raise RuntimeError(
            f"workload case {case['id']} has unsupported kind {kind!r}"
        )

    # These checks precede both canonical digest construction and every NPY /
    # StableHLO program write. In particular, an all-NaN eager reference must
    # never become a fixed digest that can compare equal to itself later.
    _verify_workload_payload_finite(
        numpy_module,
        case_id=case["id"],
        input_array=input_array,
        extra_inputs=extra_inputs,
        parameters=parameters,
        expected=expected,
    )

    if kind == "simple_gemm":
        input_digest = _named_array_digest(
            numpy_module,
            {"0": input_array, "1": extra_inputs[1]},
            require_float32=False,
        )
        parameters_digest = _named_array_digest(
            numpy_module, parameters, require_float32=False
        )
        output_digest = _array_digest(
            numpy_module, expected, require_float32=False
        )
        quantized_output_digest = output_digest
    elif kind == "llama_decoder_block":
        input_digest = _array_digest(
            numpy_module, input_array, require_float32=False
        )
        parameters_digest = _named_array_digest(
            numpy_module, parameters, require_float32=False
        )
        output_digest = _array_digest(
            numpy_module, expected, require_float32=False
        )
        quantized_output_digest = _quantized_array_digest(
            numpy_module,
            expected,
            int(case["reference"]["digest_quantization_decimals"]),
        )
    else:
        input_digest = _array_digest(numpy_module, input_array)
        parameters_digest = _named_array_digest(numpy_module, parameters)
        output_digest = _array_digest(numpy_module, expected)
        quantized_output_digest = _quantized_array_digest(
            numpy_module,
            expected,
            int(case["reference"]["digest_quantization_decimals"]),
        )
    digests = {
        "source_config": _sha256_bytes(_canonical_json_bytes(source_config)),
        "input": input_digest,
        "parameters": parameters_digest,
        "reference_output": output_digest,
        "reference_output_quantized": quantized_output_digest,
    }
    if verify_fixed_digests:
        expected_digests = case.get("digests")
        if not isinstance(expected_digests, dict):
            raise RuntimeError(
                f"workload case {case['id']} is missing fixed digests"
            )
        for name, digest in digests.items():
            expected_digest = expected_digests.get(name)
            if expected_digest != digest:
                raise RuntimeError(
                    f"workload case {case['id']} {name} digest mismatch: "
                    f"expected {expected_digest!r}, got {digest!r}"
                )
    if case.get("source_revision") != digests["source_config"]:
        raise RuntimeError(
            f"workload case {case['id']} source_revision must equal its "
            "canonical source_config digest"
        )
    return {
        "case": case,
        "source_config": source_config,
        "input": input_array,
        "extra_inputs": extra_inputs,
        "parameters": parameters,
        "expected": expected,
        "digests": digests,
    }


def emit_workload_cpu_reference(
    spec_path: pathlib.Path,
    case_id: str,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    numpy_module = _import_numpy()
    spec = load_workload_corpus_spec(spec_path)
    case = _find_workload_case(spec, case_id)
    payload = _build_workload_case_payload(spec_path, case, numpy_module)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    parameter_dir = output_dir / "parameters"
    parameter_dir.mkdir(parents=True)
    _save_workload_array(
        numpy_module, output_dir / "input.npy", payload["input"]
    )
    for index, value in sorted(payload["extra_inputs"].items()):
        _save_workload_array(
            numpy_module, output_dir / f"input-{index}.npy", value
        )
    _save_workload_array(
        numpy_module, output_dir / "expected.npy", payload["expected"]
    )
    for name, value in payload["parameters"].items():
        if pathlib.PurePath(name).name != name:
            raise RuntimeError(f"invalid workload parameter name {name!r}")
        _save_workload_array(
            numpy_module, parameter_dir / f"{name}.npy", value
        )

    bulk_qualification = case.get("bulk_qualification")
    if bulk_qualification is not None:
        if not isinstance(bulk_qualification, dict):
            raise RuntimeError("bulk_qualification must be an object")
        m = int(bulk_qualification["m"])
        k = int(bulk_qualification["k"])
        n = int(bulk_qualification["n"])
        calibration_seed = int(bulk_qualification["calibration_seed"])
        qualification_dir = output_dir / "qualification"
        qualification_dir.mkdir()
        calibration_lhs = _cast_gemm_storage(
            numpy_module,
            _deterministic_gemm_array(
                numpy_module, (m, k), seed=calibration_seed, stream=0
            ),
            case["dtype"],
        )
        calibration_rhs = _cast_gemm_storage(
            numpy_module,
            _deterministic_gemm_array(
                numpy_module, (k, n), seed=calibration_seed, stream=1
            ),
            case["dtype"],
        )
        _save_workload_array(
            numpy_module,
            qualification_dir / "calibration-lhs.npy",
            calibration_lhs,
        )
        _save_workload_array(
            numpy_module,
            qualification_dir / "calibration-rhs.npy",
            calibration_rhs,
        )
        (qualification_dir / "qualification.json").write_text(
            json.dumps(bulk_qualification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    reference_record = {
        "schema_version": WORKLOAD_CORPUS_SCHEMA_VERSION,
        "corpus_id": spec["corpus_id"],
        "case_id": case_id,
        "kind": case["kind"],
        "source_revision": case["source_revision"],
        "seed": case["seed"],
        "dtype": case["dtype"],
        "input_shape": [int(dim) for dim in payload["input"].shape],
        "output_shape": [int(dim) for dim in payload["expected"].shape],
        "digests": payload["digests"],
        "reference": case["reference"],
        "parameter_files": {
            name: f"parameters/{name}.npy" for name in payload["parameters"]
        },
        "input_files": {
            "0": "input.npy",
            **{
                str(index): f"input-{index}.npy"
                for index in sorted(payload["extra_inputs"])
            },
        },
        "bulk_qualification": bulk_qualification,
    }
    (output_dir / "reference.json").write_text(
        json.dumps(reference_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


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
    explicit_parameters = getattr(
        reference_module, "wafer_named_parameters", None
    )
    if callable(explicit_parameters):
        parameters = list(explicit_parameters())
        if parameters:
            return parameters

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


def _state_dict_numpy(
    torch_module: Any, reference_module: Any
) -> dict[str, Any]:
    return {
        name: _torch_tensor_to_workload_storage(torch_module, parameter)
        for name, parameter in _named_parameters(reference_module)
    }


def _exported_state_dict_numpy(
    torch_module: Any, exported_program: Any
) -> dict[str, Any]:
    """Convert an ExportedProgram state dict to canonical NPY payload arrays."""
    state_dict = getattr(exported_program, "state_dict", None)
    if not isinstance(state_dict, dict):
        raise RuntimeError("PyTorch ExportedProgram state_dict must be a mapping")
    converted = {}
    for name, value in state_dict.items():
        if not isinstance(name, str):
            raise RuntimeError("PyTorch ExportedProgram state names must be strings")
        if not isinstance(value, torch_module.Tensor):
            converted[name] = value
            continue
        converted[name] = _torch_tensor_to_workload_storage(
            torch_module, value
        )
    return converted


def exported_program_to_stablehlo(
    torch_module: Any,
    stablehlo_module: Any,
    exported_program: Any,
    *,
    options: Any | None = None,
) -> Any:
    """Export StableHLO while preserving BF16 parameter and buffer payloads.

    PyTorch 2.5 intentionally has no public NumPy BF16 scalar type, while the
    pinned PyTorch/XLA exporter serializes every ExportedProgram state tensor
    through ``Tensor.numpy()``.  Keep the upstream path for ordinary state.  If
    BF16 state is present, let PyTorch/XLA build the same graph and parameter
    locations without serializing weights, then attach canonical little-endian
    ``|V2`` payload arrays to its returned model.  This applies uniformly to
    parameters and persistent buffers; callers do not need model-specific dtype
    handling.
    """
    resolved_options = options
    if resolved_options is None:
        resolved_options = stablehlo_module.StableHLOExportOptions()
    state_dict = getattr(exported_program, "state_dict", None)
    has_bfloat16_state = isinstance(state_dict, dict) and any(
        isinstance(value, torch_module.Tensor)
        and value.dtype == torch_module.bfloat16
        for value in state_dict.values()
    )
    if not resolved_options.export_weights or not has_bfloat16_state:
        return stablehlo_module.exported_program_to_stablehlo(
            exported_program, options=resolved_options
        )

    graph_options = copy.copy(resolved_options)
    graph_options.export_weights = False
    program = stablehlo_module.exported_program_to_stablehlo(
        exported_program, options=graph_options
    )
    model_state = getattr(program, "_bundle", None)
    if model_state is None or not hasattr(model_state, "state_dict"):
        raise RuntimeError(
            "pinned PyTorch/XLA StableHLO result omitted its exported state"
        )
    if model_state.state_dict:
        raise RuntimeError(
            "PyTorch/XLA exported weights despite export_weights=False"
        )
    model_state.state_dict = _exported_state_dict_numpy(
        torch_module, exported_program
    )
    return program


def _move_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _build_lazy_stablehlo_program(
    *,
    torch_module: Any,
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
            additional_constants.append(
                _torch_tensor_to_workload_storage(
                    torch_module, tensor_value
                )
            )
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
        stablehlo_program = exported_program_to_stablehlo(
            torch_module,
            stablehlo_module,
            exported,
            options=options,
        )

    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_program.save(str(program_dir))
    _verify_program_dir_layout(program_dir)


def emit_simple_gemm_program(
    program_dir: pathlib.Path,
    lhs_array: Any,
    rhs_array: Any,
    expected_cpu_output: Any,
    dtype: str,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
) -> None:
    if torch_module is None or stablehlo_module is None:
        torch_module, stablehlo_module = _import_runtime_modules()
    numpy_module = _import_numpy()
    reference_module = _make_simple_gemm_module(torch_module)
    reference_module.eval()
    lhs_tensor = _torch_from_workload_storage(
        torch_module, numpy_module, lhs_array, dtype
    )
    rhs_tensor = _torch_from_workload_storage(
        torch_module, numpy_module, rhs_array, dtype
    )
    with torch_module.no_grad():
        torch_cpu_output = reference_module(lhs_tensor, rhs_tensor)
    actual = _torch_to_workload_storage(
        torch_module, numpy_module, torch_cpu_output, dtype
    )
    if not numpy_module.array_equal(actual, expected_cpu_output):
        raise RuntimeError(
            "framework CPU simple GEMM output does not match the independent "
            "increasing-K reference"
        )

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True
    with torch_module.no_grad():
        exported = torch_module.export.export(
            reference_module, (lhs_tensor, rhs_tensor)
        )
        stablehlo_program = exported_program_to_stablehlo(
            torch_module,
            stablehlo_module,
            exported,
            options=options,
        )
    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_program.save(str(program_dir))
    _verify_program_dir_layout(program_dir)


def emit_linear_residual_mlp_program(
    program_dir: pathlib.Path,
    input_array: Any,
    parameters: dict[str, Any],
    expected_cpu_output: Any,
    torch_module: Any | None = None,
    stablehlo_module: Any | None = None,
) -> None:
    if torch_module is None or stablehlo_module is None:
        torch_module, stablehlo_module = _import_runtime_modules()

    reference_module = _make_linear_residual_mlp_module(torch_module, parameters)
    reference_module.eval()
    input_tensor = torch_module.from_numpy(input_array.copy())
    with torch_module.no_grad():
        torch_cpu_output = reference_module(input_tensor)
    _verify_framework_cpu_output(
        torch_cpu_output.detach().cpu().numpy(),
        expected_cpu_output,
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    with torch_module.no_grad():
        exported = torch_module.export.export(reference_module, (input_tensor,))
        stablehlo_program = exported_program_to_stablehlo(
            torch_module,
            stablehlo_module,
            exported,
            options=options,
        )

    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_program.save(str(program_dir))
    _verify_program_dir_layout(program_dir)


def _verify_exported_parameter_payloads(
    program_dir: pathlib.Path,
    parameters: dict[str, Any],
    numpy_module: Any,
) -> None:
    for name, expected in parameters.items():
        path = program_dir / "data" / name
        if not path.is_file():
            raise RuntimeError(
                f"exported workload parameter payload is missing: data/{name}"
            )
        try:
            actual = numpy_module.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"exported workload parameter payload is not NPY: data/{name}"
            ) from error
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise RuntimeError(
                f"exported workload parameter metadata mismatch for {name}"
            )
        if not numpy_module.array_equal(actual, expected):
            raise RuntimeError(
                f"exported workload parameter payload differs from source: {name}"
            )


def _canonical_program_digest(
    program_dir: pathlib.Path, numpy_module: Any
) -> str:
    digest = hashlib.sha256()

    def add_record(name: str, payload: bytes) -> None:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    mlir_path = program_dir / "functions" / "forward.mlir"
    meta_path = program_dir / "functions" / "forward.meta"
    add_record("functions/forward.mlir", mlir_path.read_bytes())
    with meta_path.open("r", encoding="utf-8") as file:
        add_record("functions/forward.meta", _canonical_json_bytes(json.load(file)))

    for root_name in ("data", "constants"):
        root = program_dir / root_name
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(program_dir).as_posix()
            try:
                array = numpy_module.load(path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"workload program payload is not NPY: {relative}"
                ) from error
            add_record(
                relative,
                _canonical_array_bytes(
                    numpy_module, array, require_float32=False
                ),
            )
    return "sha256:" + digest.hexdigest()


def _emit_workload_program(
    *,
    spec_path: pathlib.Path,
    payload: dict[str, Any],
    program_dir: pathlib.Path,
    runtime_modules: dict[str, Any],
) -> None:
    case = payload["case"]
    if case["kind"] == "simple_gemm":
        torch_module = runtime_modules.get("torch")
        stablehlo_module = runtime_modules.get("stablehlo")
        if torch_module is None or stablehlo_module is None:
            torch_module, stablehlo_module = _import_runtime_modules()
            runtime_modules["torch"] = torch_module
            runtime_modules["stablehlo"] = stablehlo_module
        emit_simple_gemm_program(
            program_dir,
            payload["input"],
            payload["extra_inputs"][1],
            payload["expected"],
            str(case["dtype"]),
            torch_module=torch_module,
            stablehlo_module=stablehlo_module,
        )
        return
    if case["kind"] == "linear_residual_mlp":
        torch_module = runtime_modules.get("torch")
        stablehlo_module = runtime_modules.get("stablehlo")
        if torch_module is None or stablehlo_module is None:
            torch_module, stablehlo_module = _import_runtime_modules()
            runtime_modules["torch"] = torch_module
            runtime_modules["stablehlo"] = stablehlo_module
        emit_linear_residual_mlp_program(
            program_dir,
            payload["input"],
            payload["parameters"],
            payload["expected"],
            torch_module=torch_module,
            stablehlo_module=stablehlo_module,
        )
        return

    if case["kind"] in {"tiny_llama_decoder_block", "llama_decoder_block"}:
        modules = runtime_modules.get("spmd")
        if modules is None:
            modules = _import_spmd_runtime_modules()
            runtime_modules["spmd"] = modules
        (
            torch_module,
            stablehlo_module,
            runtime_module,
            xla_model_module,
            spmd_module,
            xlac_module,
        ) = modules
        config_value = case["config"]["hf_config"]
        config_path = (spec_path.parent / config_value).resolve()
        emit_hf_llama_block_program(
            program_dir,
            config_path=config_path,
            batch_size=int(case["config"]["batch_size"]),
            sequence_length=int(case["config"]["sequence_length"]),
            torch_module=torch_module,
            stablehlo_module=stablehlo_module,
            runtime_module=runtime_module,
            xla_model_module=xla_model_module,
            spmd_module=spmd_module,
            xlac_module=xlac_module,
            parameter_arrays=payload["parameters"],
            input_array=payload["input"],
            expected_cpu_output=payload["expected"],
        )
        return

    raise RuntimeError(f"unsupported workload corpus kind {case['kind']!r}")


def emit_workload_corpus(
    spec_path: pathlib.Path,
    output_dir: pathlib.Path,
    case_ids: list[str] | None = None,
    verify_reproducible: bool = False,
) -> None:
    numpy_module = _import_numpy()
    spec = load_workload_corpus_spec(spec_path)
    selected_case_ids = case_ids or [case["id"] for case in spec["cases"]]
    if len(set(selected_case_ids)) != len(selected_case_ids):
        raise RuntimeError("--corpus-case values must be unique")
    selected_cases = [
        _find_workload_case(spec, case_id) for case_id in selected_case_ids
    ]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    runtime_modules: dict[str, Any] = {}
    # Initialize the exporter in one-partition SPMD mode before any XLA value
    # exists. This keeps every corpus case on the same card-local frontend
    # boundary without introducing a physical-Tile mesh in source IR.
    spmd_modules = _import_spmd_runtime_modules()
    spmd_modules[2].use_spmd()
    torch_xla_module = sys.modules.get("torch_xla")
    if torch_xla_module is None:
        raise RuntimeError("PyTorch/XLA importer did not load torch_xla")
    _verify_workload_runtime_provenance(
        spec, spmd_modules[0], torch_xla_module
    )
    runtime_modules["spmd"] = spmd_modules
    runtime_modules["torch"] = spmd_modules[0]
    runtime_modules["stablehlo"] = spmd_modules[1]
    corpus_records = []
    for case in selected_cases:
        case_dir = output_dir / case["id"]
        reference_dir = case_dir / "reference"
        payload = emit_workload_cpu_reference(
            spec_path, case["id"], reference_dir
        )
        program_dir = case_dir / "program"
        _emit_workload_program(
            spec_path=spec_path,
            payload=payload,
            program_dir=program_dir,
            runtime_modules=runtime_modules,
        )
        _verify_exported_parameter_payloads(
            program_dir, payload["parameters"], numpy_module
        )
        program_digest = _canonical_program_digest(program_dir, numpy_module)
        expected_program_digest = case["digests"].get("exported_program")
        if expected_program_digest != program_digest:
            raise RuntimeError(
                f"workload case {case['id']} exported_program digest mismatch: "
                f"expected {expected_program_digest!r}, got {program_digest!r}"
            )

        repeat_program_digest = None
        if verify_reproducible:
            with tempfile.TemporaryDirectory(
                prefix=f"wafer-{case['id']}-repeat-", dir=output_dir.parent
            ) as repeat_tmp:
                repeat_program_dir = pathlib.Path(repeat_tmp) / "program"
                _emit_workload_program(
                    spec_path=spec_path,
                    payload=payload,
                    program_dir=repeat_program_dir,
                    runtime_modules=runtime_modules,
                )
                _verify_exported_parameter_payloads(
                    repeat_program_dir, payload["parameters"], numpy_module
                )
                repeat_program_digest = _canonical_program_digest(
                    repeat_program_dir, numpy_module
                )
            if repeat_program_digest != program_digest:
                raise RuntimeError(
                    f"workload case {case['id']} repeat export is not "
                    "canonical-equivalent"
                )

        case_record = {
            "schema_version": WORKLOAD_CORPUS_SCHEMA_VERSION,
            "corpus_id": spec["corpus_id"],
            "case_id": case["id"],
            "source": spec["source"],
            "kind": case["kind"],
            "source_revision": case["source_revision"],
            "seed": case["seed"],
            "dtype": case["dtype"],
            "config": case["config"],
            "bulk_qualification": case.get("bulk_qualification"),
            "digests": {
                **payload["digests"],
                "exported_program": program_digest,
            },
            "reference": case["reference"],
            "canonical_export_digest_algorithm": (
                "forward.mlir bytes + canonical forward.meta + canonical "
                "typed data/constants; redundant bytecode excluded"
            ),
            "repeat_export_digest": repeat_program_digest,
            "repeat_export_canonical_equivalent": (
                repeat_program_digest == program_digest
                if verify_reproducible
                else None
            ),
        }
        (case_dir / "case.json").write_text(
            json.dumps(case_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        corpus_records.append(case_record)

    (output_dir / "corpus.json").write_text(
        json.dumps(
            {
                "schema_version": WORKLOAD_CORPUS_SCHEMA_VERSION,
                "corpus_id": spec["corpus_id"],
                "cases": corpus_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _make_diagnostic_variant_case(
    base_case: dict[str, Any], seed: int
) -> dict[str, Any]:
    if base_case.get("kind") != "llama_decoder_block":
        raise RuntimeError(
            "diagnostic workload variants require a Llama scale base case"
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > _UINT64_MASK
    ):
        raise RuntimeError("diagnostic workload variant seed must fit uint64")
    variant = copy.deepcopy(base_case)
    base_case_id = str(base_case["id"])
    variant["id"] = f"{base_case_id}-diagnostic-seed-{seed}"
    variant["seed"] = seed
    variant.pop("digests", None)
    variant["diagnostic_base_case_id"] = base_case_id
    return variant


def emit_workload_variant(
    spec_path: pathlib.Path,
    base_case_id: str,
    seed: int,
    output_dir: pathlib.Path,
) -> None:
    """Emits one explicitly non-admitted seed variant for characterization."""
    numpy_module = _import_numpy()
    spec = load_workload_corpus_spec(spec_path)
    base_case = _find_workload_case(spec, base_case_id)
    variant = _make_diagnostic_variant_case(base_case, seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    reference_dir = output_dir / "reference"
    parameter_dir = reference_dir / "parameters"
    parameter_dir.mkdir(parents=True)

    spmd_modules = _import_spmd_runtime_modules()
    spmd_modules[2].use_spmd()
    torch_xla_module = sys.modules.get("torch_xla")
    if torch_xla_module is None:
        raise RuntimeError("PyTorch/XLA importer did not load torch_xla")
    _verify_workload_runtime_provenance(spec, spmd_modules[0], torch_xla_module)
    runtime_modules = {
        "spmd": spmd_modules,
        "torch": spmd_modules[0],
        "stablehlo": spmd_modules[1],
    }
    payload = _build_workload_case_payload(
        spec_path, variant, numpy_module, verify_fixed_digests=False
    )
    if payload["extra_inputs"]:
        raise RuntimeError("Llama diagnostic variant has unexpected extra inputs")

    _save_workload_array(
        numpy_module, reference_dir / "input.npy", payload["input"]
    )
    _save_workload_array(
        numpy_module, reference_dir / "expected.npy", payload["expected"]
    )
    for name, value in payload["parameters"].items():
        if pathlib.PurePath(name).name != name:
            raise RuntimeError(f"invalid workload parameter name {name!r}")
        _save_workload_array(
            numpy_module, parameter_dir / f"{name}.npy", value
        )

    program_dir = output_dir / "program"
    _emit_workload_program(
        spec_path=spec_path,
        payload=payload,
        program_dir=program_dir,
        runtime_modules=runtime_modules,
    )
    _verify_exported_parameter_payloads(
        program_dir, payload["parameters"], numpy_module
    )
    program_digest = _canonical_program_digest(program_dir, numpy_module)
    record = {
        "schema_version": WORKLOAD_CORPUS_SCHEMA_VERSION,
        "included_in_corpus": False,
        "variant_kind": "diagnostic-seed",
        "base_corpus_id": spec["corpus_id"],
        "base_case_id": base_case_id,
        "case_id": variant["id"],
        "source": spec["source"],
        "source_revision": base_case["source_revision"],
        "seed": seed,
        "dtype": variant["dtype"],
        "config": variant["config"],
        "reference": variant["reference"],
        "digests": {
            **payload["digests"],
            "exported_program": program_digest,
        },
        "reference_paths": {
            "input": "reference/input.npy",
            "expected": "reference/expected.npy",
            "parameters": "reference/parameters",
        },
        "program_path": "program",
    }
    (output_dir / "variant.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    example_input_tensor: Any | None = None,
    size: int = DEFAULT_REFERENCE_MATMUL_SIZE,
    timing_callback: Callable[[str, int], None] | None = None,
) -> None:
    def report_timing(stage: str, start_ns: int) -> None:
        if timing_callback is not None:
            timing_callback(stage, (time.monotonic_ns() - start_ns) // 1_000_000)

    strategy = get_sharding_strategy(strategy_name)
    import_start_ns = time.monotonic_ns()
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
    report_timing("runtime-import", import_start_ns)

    if not runtime_module.is_spmd():
        runtime_module.use_spmd()
    device_count = runtime_module.global_runtime_device_count()
    if device_count != strategy.device_count:
        raise RuntimeError(
            f"sharding strategy '{strategy.name}' requires "
            f"{strategy.device_count} XLA devices, got {device_count}; "
            "for CPU program directory tests set CPU_NUM_DEVICES=16 before importing "
            "torch_xla"
        )

    model_start_ns = time.monotonic_ns()
    if reference_module_factory is None:
        reference_module = _make_reference_matmul_module(torch_module, size)
    else:
        reference_module = reference_module_factory()
    reference_module.eval()
    state_dict = _state_dict_numpy(torch_module, reference_module)
    report_timing("model-state-materialization", model_start_ns)

    device_start_ns = time.monotonic_ns()
    device = xla_model_module.xla_device()
    reference_module = _move_to_device(reference_module, device)
    first_parameter = next(reference_module.parameters(), None)
    input_dtype = (
        first_parameter.dtype
        if first_parameter is not None
        else torch_module.float32
    )
    if example_input_tensor is None:
        example_input_tensor = torch_module.empty(
            size, size, dtype=input_dtype
        )
    elif (
        tuple(example_input_tensor.shape) != (size, size)
        or example_input_tensor.dtype != input_dtype
    ):
        raise RuntimeError(
            "sharded StableHLO example input must match the module: "
            f"input={tuple(example_input_tensor.shape)}/"
            f"{example_input_tensor.dtype} module={(size, size)}/{input_dtype}"
        )
    input_tensor = _move_to_device(
        example_input_tensor.detach().clone(), device
    )

    mesh = create_spmd_mesh(spmd_module, strategy)
    apply_strategy_marks(
        spmd_module=spmd_module,
        strategy=strategy,
        mesh=mesh,
        input_tensor=input_tensor,
        reference_module=reference_module,
    )
    report_timing("xla-device-materialization", device_start_ns)

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = True
    options.include_human_readable_text = True

    graph_start_ns = time.monotonic_ns()
    with torch_module.no_grad():
        output_tensor = reference_module(input_tensor)
        stablehlo_graph = _build_lazy_stablehlo_program(
            torch_module=torch_module,
            stablehlo_module=stablehlo_module,
            xla_model_module=xla_model_module,
            xlac_module=xlac_module,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            reference_module=reference_module,
            state_dict=state_dict,
        )
    report_timing("xla-graph-capture", graph_start_ns)

    save_start_ns = time.monotonic_ns()
    if program_dir.exists():
        shutil.rmtree(program_dir)
    program_dir.parent.mkdir(parents=True, exist_ok=True)
    stablehlo_module.StableHLOGraphModule(stablehlo_graph).save(
        str(program_dir), options
    )
    _verify_program_dir_layout(program_dir)
    report_timing("program-save", save_start_ns)


def emit_hf_llama_block_program(
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
    reference_module_factory: Callable[[], Any] | None = None,
    example_input_tensor: Any | None = None,
    parameter_arrays: dict[str, Any] | None = None,
    input_array: Any | None = None,
    expected_cpu_output: Any | None = None,
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

    if not runtime_module.is_spmd():
        runtime_module.use_spmd()
    device_count = runtime_module.global_runtime_device_count()
    if device_count != 1:
        raise RuntimeError(
            "card-local HF Llama block export requires one logical XLA "
            f"device, got {device_count}; for CPU program directory tests "
            "set CPU_NUM_DEVICES=1 before importing "
            "torch_xla"
        )

    if reference_module_factory is not None and parameter_arrays is not None:
        raise RuntimeError(
            "HF Llama exporter accepts either a Torch module factory or "
            "transport parameter arrays, not both"
        )
    if example_input_tensor is not None and input_array is not None:
        raise RuntimeError(
            "HF Llama exporter accepts either a Torch example input or a "
            "transport input array, not both"
        )
    if reference_module_factory is None:
        torch_module.manual_seed(0)
        reference_module = _make_hf_llama_decoder_block_module(
            torch_module,
            config,
            sequence_length,
            parameter_arrays=parameter_arrays,
        )
    else:
        reference_module = reference_module_factory()
    reference_module.eval()
    state_dict = _state_dict_numpy(torch_module, reference_module)

    if input_array is not None:
        cpu_input_tensor = torch_module.from_numpy(input_array.copy())
        if tuple(cpu_input_tensor.shape) != (
            batch_size,
            sequence_length,
            int(config["hidden_size"]),
        ):
            raise RuntimeError("tiny Llama workload input shape mismatch")
        if expected_cpu_output is not None:
            with torch_module.no_grad():
                torch_cpu_output = reference_module(cpu_input_tensor)
            _verify_framework_cpu_output(
                torch_cpu_output.detach().cpu().numpy(),
                expected_cpu_output,
                rtol=1.0e-4,
                atol=1.0e-5,
            )

    device = xla_model_module.xla_device()
    reference_module = _move_to_device(reference_module, device)
    input_dtype = next(reference_module.parameters()).dtype
    input_shape = (
        batch_size,
        sequence_length,
        int(config["hidden_size"]),
    )
    if example_input_tensor is not None:
        if (
            tuple(example_input_tensor.shape) != input_shape
            or example_input_tensor.dtype != input_dtype
        ):
            raise RuntimeError(
                "HF Llama example input must match the module: "
                f"input={tuple(example_input_tensor.shape)}/"
                f"{example_input_tensor.dtype} module={input_shape}/{input_dtype}"
            )
        input_tensor = example_input_tensor.detach().clone()
    elif input_array is None:
        input_tensor = torch_module.empty(
            *input_shape,
            dtype=input_dtype,
        )
    else:
        input_tensor = torch_module.from_numpy(input_array.copy())
    input_tensor = _move_to_device(input_tensor, device)

    mesh = create_hf_card_partition_mesh(spmd_module)
    apply_hf_card_partition_marks(
        spmd_module=spmd_module,
        mesh=mesh,
        input_tensor=input_tensor,
        reference_module=reference_module,
    )

    options = stablehlo_module.StableHLOExportOptions()
    options.export_weights = True
    options.save_weights = True
    options.inline_all_constant = False
    options.include_human_readable_text = True

    with torch_module.no_grad():
        output_tensor = reference_module(input_tensor)
        stablehlo_graph = _build_lazy_stablehlo_program(
            torch_module=torch_module,
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
        "--emit-hf-llama-block",
        action="store_true",
        help="emit a card-local HuggingFace Llama decoder block StableHLO program",
    )
    parser.add_argument(
        "--emit-workload-corpus",
        action="store_true",
        help="emit the source-backed linear/MLP and tiny Llama StableHLO corpus plus independent CPU references",
    )
    parser.add_argument(
        "--emit-cpu-reference",
        action="store_true",
        help="emit one deterministic NumPy CPU reference without importing PyTorch/XLA",
    )
    parser.add_argument(
        "--emit-workload-variant",
        action="store_true",
        help="emit one explicitly non-admitted workload seed variant for numeric characterization",
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
        help="HuggingFace transformer config JSON used by --emit-hf-llama-block",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_HF_TRANSFORMER_BATCH_SIZE,
        help="batch size for --emit-hf-llama-block",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_HF_TRANSFORMER_SEQUENCE_LENGTH,
        help="sequence length for --emit-hf-llama-block",
    )
    parser.add_argument(
        "--workload-corpus-spec",
        type=pathlib.Path,
        default=DEFAULT_WORKLOAD_CORPUS_SPEC,
        help="source-backed workload corpus specification",
    )
    parser.add_argument(
        "--corpus-case",
        action="append",
        help="workload corpus case id; repeat to select multiple cases (default: all)",
    )
    parser.add_argument(
        "--payload-seed",
        type=int,
        help="explicit uint64 payload seed for --emit-workload-variant",
    )
    parser.add_argument(
        "--verify-corpus-reproducibility",
        action="store_true",
        help="repeat each real exporter invocation and require canonical-equivalent output",
    )
    parser.add_argument("--output-program-dir", type=pathlib.Path)
    parser.add_argument("--output-corpus-dir", type=pathlib.Path)
    parser.add_argument("--output-reference-dir", type=pathlib.Path)
    parser.add_argument("--output-variant-dir", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    selected_actions = sum(
        int(action)
        for action in (
            args.emit_reference_program,
            args.emit_sharded_program,
            args.emit_hf_llama_block,
            args.emit_workload_corpus,
            args.emit_cpu_reference,
            args.emit_workload_variant,
        )
    )
    if selected_actions != 1:
        raise RuntimeError("select exactly one emit action")

    if args.emit_workload_variant:
        if args.output_variant_dir is None:
            raise RuntimeError("missing --output-variant-dir")
        if args.corpus_case is None or len(args.corpus_case) != 1:
            raise RuntimeError(
                "--emit-workload-variant requires exactly one --corpus-case"
            )
        if args.payload_seed is None:
            raise RuntimeError("missing --payload-seed")
        emit_workload_variant(
            args.workload_corpus_spec,
            args.corpus_case[0],
            args.payload_seed,
            args.output_variant_dir,
        )
        return 0

    if args.emit_cpu_reference:
        if args.output_reference_dir is None:
            raise RuntimeError("missing --output-reference-dir")
        if args.corpus_case is None or len(args.corpus_case) != 1:
            raise RuntimeError("--emit-cpu-reference requires exactly one --corpus-case")
        emit_workload_cpu_reference(
            args.workload_corpus_spec,
            args.corpus_case[0],
            args.output_reference_dir,
        )
        return 0

    if args.emit_workload_corpus:
        if args.output_corpus_dir is None:
            raise RuntimeError("missing --output-corpus-dir")
        emit_workload_corpus(
            args.workload_corpus_spec,
            args.output_corpus_dir,
            case_ids=args.corpus_case,
            verify_reproducible=args.verify_corpus_reproducibility,
        )
        return 0

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

    if args.emit_hf_llama_block:
        if args.output_program_dir is None:
            raise RuntimeError("missing --output-program-dir")
        if args.hf_config_json is None:
            raise RuntimeError("missing --hf-config-json")

        emit_hf_llama_block_program(
            args.output_program_dir,
            config_path=args.hf_config_json,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
        return 0

    raise RuntimeError(
        "missing --emit-reference-program, --emit-sharded-program, or "
        "--emit-hf-llama-block"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"wafer-pytorch-xla-capture: {error}", file=sys.stderr)
        raise SystemExit(1)
