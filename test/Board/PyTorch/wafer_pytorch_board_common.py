#!/usr/bin/env python3
"""Torch-only raw tensor transport and output comparison helpers."""

from __future__ import annotations

import ctypes
import dataclasses
import math
import pathlib
from collections.abc import Sequence

import torch


MANIFEST_DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "u32": torch.uint32,
}


@dataclasses.dataclass(frozen=True)
class ComparisonPolicy:
    rtol: float | None = None
    atol: float | None = None
    equal_nan: bool = False
    min_cosine: float | None = None
    max_relative_l2: float | None = None

    def __post_init__(self) -> None:
        if any(value is not None and (not math.isfinite(value) or value < 0)
               for value in (self.rtol, self.atol)):
            raise ValueError("torch comparison tolerances must be finite and non-negative")
        if (self.rtol is None) != (self.atol is None):
            raise ValueError("rtol and atol must both be set or both use PyTorch defaults")
        if (self.min_cosine is None) != (self.max_relative_l2 is None):
            raise ValueError("min_cosine and max_relative_l2 must both be set")
        if self.min_cosine is not None:
            if (not math.isfinite(self.min_cosine) or not 0 <= self.min_cosine <= 1
                    or not math.isfinite(self.max_relative_l2) or self.max_relative_l2 < 0):
                raise ValueError("similarity requires finite cosine in [0,1] and non-negative relative L2")
            if self.equal_nan or self.atol is None:
                raise ValueError("similarity requires finite tensors and explicit diagnostic atol/rtol")


PYTORCH_DEFAULT = ComparisonPolicy()
EXACT = ComparisonPolicy(rtol=0.0, atol=0.0)


@dataclasses.dataclass(frozen=True)
class TensorSimilarityStatistics:
    cosine_similarity: float
    relative_l2_error: float
    elementwise_mismatches: int
    mean_abs_error: float
    max_abs_error: float


def compute_tensor_similarity(
    actual: torch.Tensor, expected: torch.Tensor, *, policy: ComparisonPolicy,
) -> TensorSimilarityStatistics:
    """Compare complete vectors in F64; zero norms have no epsilon fallback."""
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise RuntimeError("similarity requires identical dtype and shape")
    if actual.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("similarity supports only F16, BF16 and F32")
    if not actual.numel():
        raise RuntimeError("similarity requires a non-empty tensor")
    if policy.atol is None or policy.rtol is None:
        raise ValueError("similarity requires explicit diagnostic atol/rtol")
    a = require_cpu_contiguous(actual, context="similarity actual").to(torch.float64).flatten()
    b = require_cpu_contiguous(expected, context="similarity expected").to(torch.float64).flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("similarity requires finite actual and expected values")
    delta = a - b
    anorm = torch.linalg.vector_norm(a).item()
    bnorm = torch.linalg.vector_norm(b).item()
    error_norm = torch.linalg.vector_norm(delta).item()
    cosine = (max(-1.0, min(1.0, torch.dot(a, b).item() / anorm / bnorm))
              if anorm and bnorm else float(anorm == bnorm))
    relative_l2 = (error_norm / bnorm if bnorm else
                   (0.0 if error_norm == 0.0 else math.inf))
    tolerance = policy.atol + policy.rtol * b.abs()
    if not torch.isfinite(tolerance).all():
        raise ValueError("computed tolerance is not finite")
    return TensorSimilarityStatistics(
        cosine, relative_l2, int(torch.count_nonzero(delta.abs() > tolerance).item()),
        delta.abs().mean().item(), delta.abs().max().item(),
    )


def model_comparison_arguments(policy: ComparisonPolicy) -> list[str]:
    arguments = []
    for field in ("atol", "rtol", "min_cosine", "max_relative_l2"):
        value = getattr(policy, field)
        if value is not None:
            arguments.extend(["--model-" + field.replace("_", "-"), str(value)])
    return arguments


def dtype_from_manifest(name: str) -> torch.dtype:
    try:
        return MANIFEST_DTYPES[name]
    except KeyError as error:
        raise RuntimeError(f"unsupported manifest tensor dtype: {name}") from error


def element_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * tensor.element_size()


def require_cpu_contiguous(tensor: torch.Tensor, *, context: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{context} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise RuntimeError(f"{context} must reside on CPU, got {tensor.device}")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.detach()


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    tensor = require_cpu_contiguous(tensor, context="raw tensor")
    expected_bytes = tensor_nbytes(tensor)
    if (
        tensor.storage_offset() != 0
        or tensor.untyped_storage().nbytes() != expected_bytes
    ):
        tensor = tensor.clone(memory_format=torch.contiguous_format)
    # The retained CPU-contiguous tensor owns exactly this checked byte span.
    # Iterating UntypedStorage copies one Python element per byte.
    raw = ctypes.string_at(tensor.data_ptr(), expected_bytes)
    if len(raw) != expected_bytes:
        raise RuntimeError(
            "torch storage contains bytes outside the logical tensor: "
            f"storage={len(raw)} tensor={expected_bytes}"
        )
    return raw


def write_tensor_raw(path: pathlib.Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tensor_raw_bytes(tensor))


def read_tensor_raw(
    path: pathlib.Path,
    *,
    dtype: torch.dtype,
    shape: Sequence[int],
) -> torch.Tensor:
    normalized_shape = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in normalized_shape):
        raise RuntimeError(f"raw tensor shape must be static: {normalized_shape}")
    raw = path.read_bytes()
    expected_bytes = math.prod(normalized_shape) * element_bytes(dtype)
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"raw tensor byte count mismatch for {path}: "
            f"expected={expected_bytes} actual={len(raw)}"
        )
    # bytearray owns writable storage, and clone detaches the result from it.
    return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(
        normalized_shape
    )


def assert_tensor_matches(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    policy: ComparisonPolicy = PYTORCH_DEFAULT,
    context: str,
) -> None:
    actual = require_cpu_contiguous(actual, context=f"{context} actual")
    expected = require_cpu_contiguous(expected, context=f"{context} expected")
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{context} shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise RuntimeError(
            f"{context} dtype mismatch: actual={actual.dtype} "
            f"expected={expected.dtype}"
        )
    if not (actual.dtype.is_floating_point or actual.dtype.is_complex):
        if not torch.equal(actual, expected):
            mismatches = int(torch.count_nonzero(actual != expected).item())
            raise AssertionError(
                f"{context}: integer output differs in {mismatches}/{actual.numel()} elements"
            )
        return
    if policy.min_cosine is not None:
        statistics = compute_tensor_similarity(actual, expected, policy=policy)
        detail = (
            f"{context}: comparison=similarity cosine={statistics.cosine_similarity:.17g} "
            f"minimum_cosine={policy.min_cosine} relative_l2={statistics.relative_l2_error:.17g} "
            f"maximum_relative_l2={policy.max_relative_l2} "
            f"elementwise_mismatches={statistics.elementwise_mismatches}/{actual.numel()} "
            f"atol={policy.atol} rtol={policy.rtol} "
            f"mean_abs={statistics.mean_abs_error:.17g} max_abs={statistics.max_abs_error:.17g}"
        )
        if (statistics.cosine_similarity < policy.min_cosine
                or statistics.relative_l2_error > policy.max_relative_l2):
            raise AssertionError(detail)
        print(detail)
        return
    options = {
        "equal_nan": policy.equal_nan,
        "check_device": True,
        "check_dtype": True,
        "msg": lambda message: f"{context}: {message}",
    }
    if policy.rtol is not None and policy.atol is not None:
        options.update(rtol=policy.rtol, atol=policy.atol)
    torch.testing.assert_close(actual, expected, **options)


def assert_raw_capture_matches(
    path: pathlib.Path,
    expected: torch.Tensor,
    *,
    policy: ComparisonPolicy = PYTORCH_DEFAULT,
    context: str,
) -> None:
    expected = require_cpu_contiguous(expected, context=f"{context} expected")
    actual = read_tensor_raw(path, dtype=expected.dtype, shape=expected.shape)
    assert_tensor_matches(actual, expected, policy=policy, context=context)


def manifest_tensor_spec(record: dict[str, object]) -> tuple[torch.dtype, tuple[int, ...]]:
    dtype = record.get("dtype")
    shape = record.get("shape")
    if not isinstance(dtype, str) or not isinstance(shape, list):
        raise RuntimeError("manifest port record requires dtype and shape")
    if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise RuntimeError("manifest port shape must be static non-negative ints")
    return dtype_from_manifest(dtype), tuple(shape)


def require_manifest_tensor(
    record: dict[str, object], tensor: torch.Tensor, *, context: str
) -> None:
    dtype, shape = manifest_tensor_spec(record)
    tensor = require_cpu_contiguous(tensor, context=context)
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise RuntimeError(
            f"{context} does not match manifest: tensor="
            f"{tuple(tensor.shape)}/{tensor.dtype} manifest={shape}/{dtype}"
        )
    expected_bytes = tensor_nbytes(tensor)
    if record.get("bytes") != expected_bytes:
        raise RuntimeError(
            f"{context} manifest byte count mismatch: "
            f"expected={expected_bytes} actual={record.get('bytes')}"
        )
