#!/usr/bin/env python3
"""Torch-only raw tensor transport and output comparison helpers."""

from __future__ import annotations

import ctypes
import dataclasses
import math
import pathlib
import sys
from collections.abc import Sequence

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "Support"))
import wafer_physical_tensor_codec as physical_codec


MANIFEST_DTYPES = {
    "bool": torch.bool,
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


def make_similarity_policy(
    dtype: torch.dtype, *, rtol: float | None = None, atol: float | None = None,
) -> ComparisonPolicy:
    """Select ordinary floating computation acceptance, retaining point diagnostics."""
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("similarity policy requires F16, BF16 or F32")
    if rtol is None and atol is None:
        from torch.testing._comparison import default_tolerances

        rtol, atol = default_tolerances(dtype)
    return ComparisonPolicy(
        rtol=rtol, atol=atol, min_cosine=0.9999, max_relative_l2=0.01,
    )


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
    if tensor.dtype == torch.bool:
        return (int(tensor.numel()) + 7) // 8
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
    if tensor.dtype == torch.bool:
        # The manifest Bool format is packed least-significant bit first;
        # PyTorch's in-memory bool is byte-sized.
        flat = tensor.flatten().to(torch.uint8)
        if flat.numel() % 8:
            flat = torch.cat((flat, torch.zeros(8 - flat.numel() % 8,
                                                dtype=torch.uint8)))
        packed = (flat.reshape(-1, 8) << torch.arange(8)).sum(dim=1).to(torch.uint8)
        return tensor_raw_bytes(packed)
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


@dataclasses.dataclass(frozen=True)
class TensorCapture:
    expected: torch.Tensor
    layout: str


def physical_storage_elements(shape: Sequence[int], dtype: torch.dtype,
                              layout: str) -> int:
    if layout in ("tensor", "ntensor"):
        return math.prod(shape)
    names = {"cx": "Cx", "ncx": "NCx"}
    if layout not in names or dtype == torch.bool:
        raise RuntimeError(f"unsupported board payload layout/dtype: {layout}/{dtype}")
    return physical_codec.physical_layout(shape, names[layout], element_bytes(dtype)).physical_elements


def pack_tensor(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    tensor = require_cpu_contiguous(tensor, context="physical payload")
    if layout in ("tensor", "ntensor"):
        return tensor
    physical_storage_elements(tensor.shape, tensor.dtype, layout)
    geometry = physical_codec.physical_layout(
        tensor.shape, {"cx": "Cx", "ncx": "NCx"}[layout], tensor.element_size())
    batches = tensor.shape[0] if layout == "ncx" else 1
    rows = geometry.hw_elements if layout == "ncx" else geometry.outer_elements
    padded = torch.zeros((batches, rows, geometry.aligned_c), dtype=tensor.dtype)
    padded[:, :, :tensor.shape[-1]] = tensor.reshape(batches, rows, tensor.shape[-1])
    full_c = geometry.full_blocks * geometry.c_block
    blocks = padded[:, :, :full_c].reshape(
        batches, rows, geometry.full_blocks, geometry.c_block).transpose(1, 2)
    storage = torch.zeros((batches, geometry.batch_elements), dtype=tensor.dtype)
    storage[:, :rows * full_c] = blocks.reshape(batches, -1)
    if geometry.tail_width:
        storage[:, rows * full_c:rows * geometry.aligned_c] = padded[:, :, full_c:].reshape(batches, -1)
    return storage.flatten()


def unpack_tensor(storage: torch.Tensor, shape: Sequence[int], layout: str) -> torch.Tensor:
    if layout in ("tensor", "ntensor"):
        return storage.reshape(tuple(shape))
    physical_storage_elements(shape, storage.dtype, layout)
    geometry = physical_codec.physical_layout(
        shape, {"cx": "Cx", "ncx": "NCx"}[layout], storage.element_size())
    batches = shape[0] if layout == "ncx" else 1
    rows = geometry.hw_elements if layout == "ncx" else geometry.outer_elements
    storage = storage.reshape(batches, geometry.batch_elements)
    full_c = geometry.full_blocks * geometry.c_block
    values = torch.empty((batches, rows, geometry.aligned_c), dtype=storage.dtype)
    values[:, :, :full_c] = storage[:, :rows * full_c].reshape(
        batches, geometry.full_blocks, rows, geometry.c_block).transpose(1, 2).reshape(batches, rows, full_c)
    if geometry.tail_width:
        values[:, :, full_c:] = storage[:, rows * full_c:rows * geometry.aligned_c].reshape(batches, rows, geometry.tail_width)
    return values[:, :, :shape[-1]].contiguous().reshape(tuple(shape))


def write_tensor_raw(path: pathlib.Path, tensor: torch.Tensor, *, layout: str = "tensor") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tensor_raw_bytes(pack_tensor(tensor, layout)))


def read_tensor_raw(
    path: pathlib.Path,
    *,
    dtype: torch.dtype,
    shape: Sequence[int],
    layout: str = "tensor",
) -> torch.Tensor:
    normalized_shape = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in normalized_shape):
        raise RuntimeError(f"raw tensor shape must be static: {normalized_shape}")
    raw = path.read_bytes()
    count = physical_storage_elements(normalized_shape, dtype, layout)
    expected_bytes = (count + 7) // 8 if dtype == torch.bool else count * element_bytes(dtype)
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"raw tensor byte count mismatch for {path}: "
            f"expected={expected_bytes} actual={len(raw)}"
        )
    # bytearray owns writable storage, and clone detaches the result from it.
    if dtype == torch.bool:
        if not raw:
            return torch.empty(normalized_shape, dtype=dtype)
        packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        values = ((packed[:, None] >> torch.arange(8)) & 1).flatten()[:count]
        return values.to(torch.bool).reshape(normalized_shape)
    storage = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
    return unpack_tensor(storage, normalized_shape, layout)


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
    layout: str = "tensor",
) -> None:
    expected = require_cpu_contiguous(expected, context=f"{context} expected")
    actual = read_tensor_raw(path, dtype=expected.dtype, shape=expected.shape, layout=layout)
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
    layout = record.get("layout")
    if not isinstance(layout, str):
        raise RuntimeError(f"{context} manifest port requires physical layout")
    elements = physical_storage_elements(shape, dtype, layout)
    expected_bytes = (elements + 7) // 8 if dtype == torch.bool else elements * element_bytes(dtype)
    if record.get("bytes") != expected_bytes:
        raise RuntimeError(
            f"{context} manifest byte count mismatch: "
            f"expected={expected_bytes} actual={record.get('bytes')}"
        )
