#!/usr/bin/env python3
"""Large/tail/batch/orientation NE calibration rows and exact host oracles."""

from __future__ import annotations

import dataclasses
import functools
import math
import struct

import wafer_physical_tensor_codec as physical


REQUEST_MAGIC = 0x31514552454E4357
RECORD_MAGIC = 0x31434552454E4357
REQUEST_GUARD = 0xB7A6958473625140
RECORD_GUARD = 0x0F1E2D3C4B5A6978
SCHEMA = 4
REQUEST_WORDS = 24
RECORD_WORDS = 32
RESOURCE_BYTES = 1048576
SLOT_BYTES = 262144
BODY_OFFSET = 256
OUTPUT_DDR_OFFSET = SLOT_BYTES
SLOT_CANARY = 0xA7
# NE owns its complete physical result span and zero-fills invalid lanes.
OUTPUT_PADDING = 0x00
CASE_BASE = 20000
SPECIAL_CASE_BASE = 21000
GEMM_OPTION_CASE_BASE = 22000
CONV_CASE_BASE = 23000
CONV_OPTION_CASE_BASE = 23100
DEFERRED_CASE_BASE = 24000

DTYPES = {"F16": 0, "BF16": 1, "I8": 2}
FLOAT_DTYPES = ("F16", "BF16")
KINDS = {
    "GEMM": 0,
    "CONV": 1,
    "DEPTHWISE_CONV": 2,
    "BACKWARD_CONV": 3,
}
PROFILES = {
    "DENSE": 0,
    "BF16_CANCELLATION": 1,
    "BF16_ROUNDING": 2,
    "BF16_SIGNED_ZERO": 3,
    "BF16_SUBNORMAL": 4,
    "BF16_OVERFLOW_INF": 5,
    "BF16_NAN": 6,
    "CONV_LARGE": 16,
    "CONV_HELDOUT": 17,
    "GEMM_QUANT": 32,
    "GEMM_L1_R2": 33,
    "GEMM_L2_R1": 34,
    "DEPTHWISE_1X1": 35,
    "BACKWARD_1X1": 36,
}
OPTIONS = {
    "NONE": 0,
    "BIAS": 1,
    "PSUM": 2,
    "RELU": 3,
    "LEAKY_RELU": 4,
    "POSITIVE_AXIS_SCALE": 5,
    "NEGATIVE_AXIS_SCALE": 6,
}
DISPOSITIONS = {
    "BOARD_EXACT": 0,
    "BOARD_OBSERVED": 1,
    "ISOLATED_DEFERRED": 2,
    "STATIC_NEGATIVE": 3,
}
BF16_NAN_INPUT_BITS = (
    0x7FC1,  # positive quiet NaN, payload 1
    0x7FFF,  # positive quiet NaN, maximal payload
    0xFFC2,  # negative quiet NaN, payload 2
    0x7F81,  # positive signaling NaN, payload 1
    0x7FA5,  # positive signaling NaN, nontrivial payload
    0xFF81,  # negative signaling NaN, payload 1
)
ORIENTATIONS = {
    "NN": (0, 0),
    "NT": (0, 1),
    "TN": (1, 0),
    "TT": (1, 1),
}
GEOMETRIES = {
    "large": (1, 64, 128, 128, "Cx"),
    "tail": (1, 65, 129, 129, "Cx"),
    "batch2": (2, 32, 64, 65, "NCx"),
}
REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "DTYPE": 3,
    "LHS_ORIENTATION": 4,
    "RHS_ORIENTATION": 5,
    "BATCH": 6,
    "M": 7,
    "K": 8,
    "N": 9,
    "LHS_SPAN": 10,
    "RHS_SPAN": 11,
    "OUTPUT_SPAN": 12,
    "SAMPLE": 13,
    "RESOURCE_BYTES": 14,
    "SLOT_BYTES": 15,
    "BODY_OFFSET": 16,
    "KIND": 17,
    "PROFILE": 18,
    "OPTION": 19,
    "AUX_SPAN": 20,
    "DISPOSITION": 21,
    "BATCH_PAIR": 22,
    "GUARD": 23,
}
REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "DTYPE": 4,
    "LHS_ORIENTATION": 5,
    "RHS_ORIENTATION": 6,
    "BATCH": 7,
    "M": 8,
    "K": 9,
    "N": 10,
    "LHS_SPAN": 11,
    "RHS_SPAN": 12,
    "OUTPUT_SPAN": 13,
    "SAMPLE": 14,
    "EXECUTE_RESULT": 15,
    "REQUEST_GUARD": 16,
    "OUTPUT_DDR_OFFSET": 17,
    "SLOT_BYTES": 18,
    "BODY_OFFSET": 19,
    "OUTPUT_GUARD_MISMATCHES": 20,
    "KIND": 21,
    "PROFILE": 22,
    "OPTION": 23,
    "AUX_SPAN": 24,
    "DISPOSITION": 25,
    "LHS_BATCH": 26,
    "RHS_BATCH": 27,
    "RECORD_GUARD": 31,
}


def _stored_shape(
    batch: int,
    rows: int,
    columns: int,
    transpose: int,
) -> tuple[int, ...]:
    matrix = (columns, rows) if transpose else (rows, columns)
    return ((batch,) + matrix) if batch > 1 else matrix


@dataclasses.dataclass(frozen=True)
class NECase:
    case_id: int
    name: str
    dtype_name: str
    geometry_name: str
    orientation_name: str
    batch: int
    m: int
    k: int
    n: int
    kind_name: str
    profile_name: str
    option_name: str
    disposition_name: str
    reason: str
    lhs_layout: str
    rhs_layout: str
    output_layout: str
    lhs_orientation: int
    rhs_orientation: int
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    lhs_span: int
    rhs_span: int
    output_span: int
    aux_span: int

    @property
    def dtype(self) -> int:
        return DTYPES[self.dtype_name]

    @property
    def kind(self) -> int:
        return KINDS[self.kind_name]

    @property
    def profile(self) -> int:
        return PROFILES[self.profile_name]

    @property
    def option(self) -> int:
        return OPTIONS[self.option_name]

    @property
    def disposition(self) -> int:
        return DISPOSITIONS[self.disposition_name]

    @property
    def element_bytes(self) -> int:
        return 1 if self.dtype_name == "I8" else 2

    @property
    def lhs_batch(self) -> int:
        return 1 if self.profile_name == "GEMM_L1_R2" else self.batch

    @property
    def rhs_batch(self) -> int:
        return 1 if self.profile_name == "GEMM_L2_R1" else self.batch

    @property
    def is_safe(self) -> bool:
        return self.disposition_name not in {
            "ISOLATED_DEFERRED",
            "STATIC_NEGATIVE",
        }

    @property
    def exact(self) -> bool:
        return self.disposition_name == "BOARD_EXACT"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "dtype": self.dtype_name.lower(),
            "geometry": self.geometry_name,
            "orientation": self.orientation_name,
            "batch": self.batch,
            "lhs_batch": self.lhs_batch,
            "rhs_batch": self.rhs_batch,
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "layout": self.layout.lower(),
            "kind": self.kind_name.lower(),
            "profile": self.profile_name.lower().replace("_", "-"),
            "option": self.option_name.lower().replace("_", "-"),
            "disposition": self.disposition_name.lower().replace("_", "-"),
            "reason": self.reason,
            "lhs_layout": self.lhs_layout.lower(),
            "rhs_layout": self.rhs_layout.lower(),
            "output_layout": self.output_layout.lower(),
            "lhs_shape": self.lhs_shape,
            "rhs_shape": self.rhs_shape,
            "output_shape": self.output_shape,
            "lhs_span": self.lhs_span,
            "rhs_span": self.rhs_span,
            "output_span": self.output_span,
            "aux_span": self.aux_span,
            "oracle": (
                "exact-logical-bits+physical-guard"
                if self.exact
                else "raw-logical-bits+physical-guard"
            ),
        }

    @property
    def layout(self) -> str:
        return self.output_layout


def _make_case(
    dtype_name: str,
    geometry_name: str,
    orientation_name: str,
) -> NECase:
    batch, m, k, n, layout = GEOMETRIES[geometry_name]
    lhs_orientation, rhs_orientation = ORIENTATIONS[orientation_name]
    lhs_shape = _stored_shape(batch, m, k, lhs_orientation)
    rhs_shape = _stored_shape(batch, k, n, rhs_orientation)
    output_shape = ((batch, m, n) if batch > 1 else (m, n))
    lhs_span = physical.physical_layout(lhs_shape, layout, 2).physical_bytes
    rhs_span = physical.physical_layout(rhs_shape, layout, 2).physical_bytes
    output_span = physical.physical_layout(
        output_shape, layout, 2
    ).physical_bytes
    ordinal = (
        FLOAT_DTYPES.index(dtype_name) * 12
        + tuple(GEOMETRIES).index(geometry_name) * 4
        + tuple(ORIENTATIONS).index(orientation_name)
    )
    return NECase(
        CASE_BASE + ordinal,
        (
            f"ne-{dtype_name.lower()}-{geometry_name}-"
            f"{orientation_name.lower()}"
        ),
        dtype_name,
        geometry_name,
        orientation_name,
        batch,
        m,
        k,
        n,
        "GEMM",
        "DENSE",
        "NONE",
        "BOARD_EXACT",
        "",
        layout,
        layout,
        layout,
        lhs_orientation,
        rhs_orientation,
        lhs_shape,
        rhs_shape,
        output_shape,
        lhs_span,
        rhs_span,
        output_span,
        0,
    )


_BASE_CASES = tuple(
    _make_case(dtype, geometry, orientation)
    for dtype in FLOAT_DTYPES
    for geometry in GEOMETRIES
    for orientation in ORIENTATIONS
)


def _make_special_case(profile_name: str) -> NECase:
    base = _make_case("BF16", "large", "NN")
    profile = PROFILES[profile_name]
    observed = profile_name in {
        "BF16_SIGNED_ZERO",
        "BF16_SUBNORMAL",
        "BF16_OVERFLOW_INF",
        "BF16_NAN",
    }
    return dataclasses.replace(
        base,
        case_id=SPECIAL_CASE_BASE + profile - 1,
        name=f"ne-bf16-{profile_name.lower().replace('_', '-')}",
        profile_name=profile_name,
        disposition_name=(
            "BOARD_OBSERVED" if observed else "BOARD_EXACT"
        ),
        reason=(
            "raw result records signed-zero/subnormal/overflow/NaN behavior without "
            "assuming an accumulator propagation policy"
            if observed
            else ""
        ),
    )


def _option_disposition(option_name: str) -> str:
    return (
        "BOARD_OBSERVED"
        if option_name
        in {
            "BIAS",
            "RELU",
            "LEAKY_RELU",
            "POSITIVE_AXIS_SCALE",
            "NEGATIVE_AXIS_SCALE",
        }
        else "BOARD_EXACT"
    )


def _option_reason(option_name: str) -> str:
    if option_name == "BIAS":
        return (
            "current FP bias profile has no exact numeric contract: its "
            "first GEMM calibration produced the base output unchanged "
            "despite bias_en and a populated bias address; raw output "
            "records the observed behavior without treating that no-op as "
            "additive bias semantics"
        )
    if option_name == "RELU":
        return (
            "current FP16 GEMM profile retained negative base results despite "
            "relu_en; FP16 and BF16 share this wrapper option row, so raw "
            "output records the profile behavior without treating the "
            "observed no-op as ReLU semantics"
        )
    if _option_disposition(option_name) == "BOARD_OBSERVED":
        return (
            "raw output calibrates the option formula without promoting it "
            "to production numeric support"
        )
    return ""


def _option_aux_span(
    option_name: str,
    output_shape: tuple[int, ...],
    output_layout: str,
    channels: int,
) -> int:
    if option_name == "PSUM":
        return physical.physical_layout(
            output_shape, output_layout, 2
        ).physical_bytes
    if option_name in {
        "BIAS",
        "POSITIVE_AXIS_SCALE",
        "NEGATIVE_AXIS_SCALE",
    }:
        return physical.physical_layout((channels,), "Cx", 2).physical_bytes
    return 0


def _make_gemm_option_case(
    dtype_name: str, option_name: str
) -> NECase:
    base = _make_case(dtype_name, "large", "NN")
    option = OPTIONS[option_name]
    return dataclasses.replace(
        base,
        case_id=(
            GEMM_OPTION_CASE_BASE
            + DTYPES[dtype_name] * 6
            + option
            - 1
        ),
        name=(
            f"ne-{dtype_name.lower()}-large-nn-"
            f"{option_name.lower().replace('_', '-')}"
        ),
        option_name=option_name,
        disposition_name=_option_disposition(option_name),
        reason=_option_reason(option_name),
        aux_span=_option_aux_span(
            option_name,
            base.output_shape,
            base.output_layout,
            base.n,
        ),
    )


def _make_conv_case(
    dtype_name: str,
    profile_name: str,
    option_name: str = "NONE",
) -> NECase:
    output_c = 96 if profile_name == "CONV_LARGE" else 65
    input_shape = (2, 17, 19, 65)
    weight_shape = (3, 2, output_c, 65)
    output_shape = (2, 17, 10, output_c)
    lhs_layout = "NCx"
    rhs_layout = "Cx"
    output_layout = "NCx"
    option = OPTIONS[option_name]
    if option_name == "NONE":
        case_id = (
            CONV_CASE_BASE
            + DTYPES[dtype_name] * 2
            + (profile_name == "CONV_HELDOUT")
        )
    else:
        case_id = (
            CONV_OPTION_CASE_BASE
            + DTYPES[dtype_name] * 6
            + option
            - 1
        )
    disposition = "BOARD_OBSERVED"
    reason = (
        "the bounded nontrivial Conv output does not match the current host "
        "NCx/HWOI numeric oracle beyond the first output pixel; until feature, "
        "weight, and output physical indexing are distinguished, raw output "
        "must not establish exact Conv or fused-option semantics"
    )
    return NECase(
        case_id=case_id,
        name=(
            f"ne-{dtype_name.lower()}-"
            f"{profile_name.lower().replace('_', '-')}"
            + (
                ""
                if option_name == "NONE"
                else f"-{option_name.lower().replace('_', '-')}"
            )
        ),
        dtype_name=dtype_name,
        geometry_name=profile_name.lower().replace("conv_", ""),
        orientation_name="NN",
        batch=2,
        m=17,
        k=19,
        n=output_c,
        kind_name="CONV",
        profile_name=profile_name,
        option_name=option_name,
        disposition_name=disposition,
        reason=reason,
        lhs_layout=lhs_layout,
        rhs_layout=rhs_layout,
        output_layout=output_layout,
        lhs_orientation=0,
        rhs_orientation=0,
        lhs_shape=input_shape,
        rhs_shape=weight_shape,
        output_shape=output_shape,
        lhs_span=physical.physical_layout(
            input_shape, lhs_layout, 2
        ).physical_bytes,
        rhs_span=physical.physical_layout(
            weight_shape, rhs_layout, 2
        ).physical_bytes,
        output_span=physical.physical_layout(
            output_shape, output_layout, 2
        ).physical_bytes,
        aux_span=_option_aux_span(
            option_name, output_shape, output_layout, output_c
        ),
    )


@dataclasses.dataclass(frozen=True)
class NEDeferredCase:
    case_id: int
    name: str
    capability: str
    disposition_name: str
    reason: str

    @property
    def is_safe(self) -> bool:
        return False

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "capability": self.capability,
            "disposition": self.disposition_name.lower().replace("_", "-"),
            "reason": self.reason,
            "oracle": "not-issued",
        }


_SPECIAL_CASE_ROWS = tuple(
    _make_special_case(profile)
    for profile in (
        "BF16_CANCELLATION",
        "BF16_ROUNDING",
        "BF16_SIGNED_ZERO",
        "BF16_SUBNORMAL",
        "BF16_OVERFLOW_INF",
        "BF16_NAN",
    )
)
_GEMM_OPTION_CASES = tuple(
    _make_gemm_option_case(dtype, option)
    for dtype in FLOAT_DTYPES
    for option in tuple(OPTIONS)[1:]
)
_CONV_CASES = tuple(
    _make_conv_case(dtype, profile)
    for dtype in FLOAT_DTYPES
    for profile in ("CONV_LARGE", "CONV_HELDOUT")
)
_CONV_OPTION_CASES = tuple(
    _make_conv_case(dtype, "CONV_LARGE", option)
    for dtype in FLOAT_DTYPES
    for option in tuple(OPTIONS)[1:]
)


def _make_quant_case() -> NECase:
    shape = (16, 16)
    layout = "Cx"
    span = physical.physical_layout(shape, layout, 1).physical_bytes
    return NECase(
        case_id=DEFERRED_CASE_BASE,
        name="ne-gemm-quant-observed",
        dtype_name="I8",
        geometry_name="quant-16x16",
        orientation_name="NN",
        batch=1,
        m=16,
        k=16,
        n=16,
        kind_name="GEMM",
        profile_name="GEMM_QUANT",
        option_name="NONE",
        disposition_name="BOARD_OBSERVED",
        reason=(
            "nonzero q0/q1 and left/right zero points are captured as raw "
            "INT8 output without assuming a requantization formula"
        ),
        lhs_layout=layout,
        rhs_layout=layout,
        output_layout=layout,
        lhs_orientation=0,
        rhs_orientation=0,
        lhs_shape=shape,
        rhs_shape=shape,
        output_shape=shape,
        lhs_span=span,
        rhs_span=span,
        output_span=span,
        aux_span=0,
    )


def _make_conv_kind_case(dtype_name: str, kind_name: str) -> NECase:
    input_shape = (1, 4, 4, 64)
    if kind_name == "DEPTHWISE_CONV":
        profile_name = "DEPTHWISE_1X1"
        weight_shape = (1, 1, 64, 1)
        output_shape = input_shape
        ordinal_base = 4
        spelling = "depthwise-conv"
        reason = (
            "raw output closes bounded wrapper execution while the distinct "
            "weight/channel relation remains a calibration result"
        )
    elif kind_name == "BACKWARD_CONV":
        profile_name = "BACKWARD_1X1"
        weight_shape = (1, 1, 64, 64)
        # For backward Conv the wrapper's AddWeight shape owns tfr_1, so the
        # physical output transfer footprint follows the weight-gradient
        # shape rather than the shape argument passed to AddOutput.
        output_shape = weight_shape
        ordinal_base = 6
        spelling = "backward-conv"
        reason = (
            "raw output closes bounded wrapper execution; backward Conv "
            "takes its physical output transfer shape from the full weight "
            "shape stored in tfr_1, while the numeric relation remains a "
            "calibration result"
        )
    else:
        raise RuntimeError(f"unknown observed convolution kind {kind_name}")
    layout = "Cx"
    return NECase(
        case_id=DEFERRED_CASE_BASE
        + ordinal_base
        + FLOAT_DTYPES.index(dtype_name),
        name=f"ne-{spelling}-{dtype_name.lower()}-observed",
        dtype_name=dtype_name,
        geometry_name="1x1-n1h4w4c64",
        orientation_name="NN",
        batch=1,
        m=4,
        k=4,
        n=64,
        kind_name=kind_name,
        profile_name=profile_name,
        option_name="NONE",
        disposition_name="BOARD_OBSERVED",
        reason=reason,
        lhs_layout=layout,
        rhs_layout=layout,
        output_layout=layout,
        lhs_orientation=0,
        rhs_orientation=0,
        lhs_shape=input_shape,
        rhs_shape=weight_shape,
        output_shape=output_shape,
        lhs_span=physical.physical_layout(
            input_shape, layout, 2
        ).physical_bytes,
        rhs_span=physical.physical_layout(
            weight_shape, layout, 2
        ).physical_bytes,
        output_span=physical.physical_layout(
            output_shape, layout, 2
        ).physical_bytes,
        aux_span=0,
    )


def _make_unequal_batch_case(
    dtype_name: str, profile_name: str
) -> NECase:
    batch = 2
    m, k, n = 32, 64, 65
    lhs_batch = 1 if profile_name == "GEMM_L1_R2" else 2
    rhs_batch = 1 if profile_name == "GEMM_L2_R1" else 2
    lhs_shape = (
        (m, k) if lhs_batch == 1 else (lhs_batch, m, k)
    )
    rhs_shape = (
        (k, n) if rhs_batch == 1 else (rhs_batch, k, n)
    )
    output_shape = (batch, m, n)
    lhs_layout = "Cx" if lhs_batch == 1 else "NCx"
    rhs_layout = "Cx" if rhs_batch == 1 else "NCx"
    ordinal = (
        FLOAT_DTYPES.index(dtype_name) * 2
        + (profile_name == "GEMM_L2_R1")
    )
    return NECase(
        case_id=DEFERRED_CASE_BASE + 8 + ordinal,
        name=(
            f"ne-gemm-{dtype_name.lower()}-"
            f"{profile_name.lower().replace('_', '-')}"
        ),
        dtype_name=dtype_name,
        geometry_name="unequal-batch",
        orientation_name="NN",
        batch=batch,
        m=m,
        k=k,
        n=n,
        kind_name="GEMM",
        profile_name=profile_name,
        option_name="NONE",
        disposition_name="BOARD_EXACT",
        reason="",
        lhs_layout=lhs_layout,
        rhs_layout=rhs_layout,
        output_layout="NCx",
        lhs_orientation=0,
        rhs_orientation=0,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        output_shape=output_shape,
        lhs_span=physical.physical_layout(
            lhs_shape, lhs_layout, 2
        ).physical_bytes,
        rhs_span=physical.physical_layout(
            rhs_shape, rhs_layout, 2
        ).physical_bytes,
        output_span=physical.physical_layout(
            output_shape, "NCx", 2
        ).physical_bytes,
        aux_span=0,
    )


_QUANT_CASES = (_make_quant_case(),)
_DEPTHWISE_BACKWARD_CASES = tuple(
    _make_conv_kind_case(dtype, kind)
    for kind in ("DEPTHWISE_CONV", "BACKWARD_CONV")
    for dtype in FLOAT_DTYPES
)
_UNEQUAL_BATCH_CASES = tuple(
    _make_unequal_batch_case(dtype, profile)
    for dtype in FLOAT_DTYPES
    for profile in ("GEMM_L1_R2", "GEMM_L2_R1")
)
_NONEXECUTABLE_CASES = (
    NEDeferredCase(
        DEFERRED_CASE_BASE + 1,
        "ne-gemm-sparse-static-unsupported",
        "gemm-sparse",
        "STATIC_NEGATIVE",
        "TsmGemm has no SetSparse entry; Conv/Depthwise SetSparse cannot "
        "express GEMM sparse through the typed ABI",
    ),
    NEDeferredCase(
        DEFERRED_CASE_BASE + 2,
        "ne-gemm-pad-static-negative",
        "gemm-pad",
        "STATIC_NEGATIVE",
        "the typed TsmGemm ABI has no pad field",
    ),
    NEDeferredCase(
        DEFERRED_CASE_BASE + 3,
        "ne-gemm-unpad-static-negative",
        "gemm-unpad",
        "STATIC_NEGATIVE",
        "the typed TsmGemm ABI has no unpad field",
    ),
)

SAFE_CASES = (
    _BASE_CASES
    + _SPECIAL_CASE_ROWS
    + _GEMM_OPTION_CASES
    + _CONV_CASES
    + _CONV_OPTION_CASES
    + _QUANT_CASES
    + _DEPTHWISE_BACKWARD_CASES
    + _UNEQUAL_BATCH_CASES
)
CATALOG = SAFE_CASES + _NONEXECUTABLE_CASES
CASES_BY_NAME = {case.name: case for case in CATALOG}
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "ne-gemm-dtype-orientation-main-tail-batch": _BASE_CASES,
    "ne-long-accumulation-cancellation": tuple(
        case
        for case in _SPECIAL_CASE_ROWS
        if case.profile_name
        in {"BF16_CANCELLATION", "BF16_ROUNDING"}
    ),
    "ne-bf16-special-values": tuple(
        case
        for case in _SPECIAL_CASE_ROWS
        if case.profile_name
        in {
            "BF16_SIGNED_ZERO",
            "BF16_SUBNORMAL",
            "BF16_OVERFLOW_INF",
            "BF16_NAN",
        }
    ),
    "ne-one-factor-options-positive": tuple(
        case
        for case in _GEMM_OPTION_CASES + _CONV_OPTION_CASES
        if case.disposition_name == "BOARD_EXACT"
    ),
    "ne-one-factor-options-observed": tuple(
        case
        for case in _GEMM_OPTION_CASES + _CONV_OPTION_CASES
        if case.disposition_name == "BOARD_OBSERVED"
    ),
    "ne-quant-observed": _QUANT_CASES,
    "ne-sparse-static-negative": _NONEXECUTABLE_CASES[:1],
    "ne-pad-unpad-static-negative": _NONEXECUTABLE_CASES[1:],
    "ne-conv-large-heldout": _CONV_CASES,
    "ne-depthwise-backward-conv-observed": (
        _DEPTHWISE_BACKWARD_CASES
    ),
    "ne-batch-broadcast-positive": _UNEQUAL_BATCH_CASES,
}


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_logical: tuple[bytes, ...] | None
    expected_physical: bytes | None


def _encode(dtype_name: str, value: float) -> bytes:
    if dtype_name == "I8":
        integer = int(value)
        if integer < -128 or integer > 127 or integer != value:
            raise RuntimeError(f"INT8 calibration value is invalid: {value}")
        return struct.pack("<b", integer)
    if dtype_name == "F16":
        return struct.pack("<e", value)
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", ((bits + rounding_bias) >> 16) & 0xFFFF)


def _encoded_operand(
    case: NECase,
    role: str,
    values: tuple[float, ...],
) -> tuple[bytes, ...]:
    if case.profile_name != "BF16_NAN" or role != "lhs":
        return tuple(_encode(case.dtype_name, value) for value in values)
    if case.dtype_name != "BF16" or case.lhs_orientation != 0:
        raise RuntimeError("BF16 NaN raw-bit profile requires NN BF16 input")
    encoded: list[bytes] = []
    for index, value in enumerate(values):
        row = (index // case.k) % case.m
        inner = index % case.k
        variant = row % (len(BF16_NAN_INPUT_BITS) + 1)
        if inner == 0 and variant < len(BF16_NAN_INPUT_BITS):
            encoded.append(struct.pack("<H", BF16_NAN_INPUT_BITS[variant]))
        else:
            encoded.append(_encode(case.dtype_name, value))
    return tuple(encoded)


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def _dense_lhs(case: NECase) -> tuple[float, ...]:
    return tuple(
        1.0
        if _mix32(
            ((batch * case.m + row) * case.k + inner) + 0x13579
        )
        & 1
        else -1.0
        for batch in range(case.lhs_batch)
        for row in range(case.m)
        for inner in range(case.k)
    )


def _dense_rhs(case: NECase) -> tuple[float, ...]:
    return tuple(
        1.0
        if _mix32(
            ((batch * case.k + inner) * case.n + column) + 0x2468A
        )
        & 1
        else -1.0
        for batch in range(case.rhs_batch)
        for inner in range(case.k)
        for column in range(case.n)
    )


def _gemm_inputs(case: NECase) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if case.profile_name in {
        "DENSE",
        "GEMM_L1_R2",
        "GEMM_L2_R1",
    }:
        return _dense_lhs(case), _dense_rhs(case)
    if case.profile_name == "GEMM_QUANT":
        lhs = tuple(
            float((row * 5 + inner * 3) % 15 - 7)
            for row in range(case.m)
            for inner in range(case.k)
        )
        rhs = tuple(
            float((inner * 7 + column * 3) % 13 - 6)
            for inner in range(case.k)
            for column in range(case.n)
        )
        return lhs, rhs
    if case.profile_name == "BF16_CANCELLATION":
        lhs = (1.0,) * (case.batch * case.m * case.k)
        rhs = tuple(
            1.0 if inner % 2 == 0 else -1.0
            for _batch in range(case.batch)
            for inner in range(case.k)
            for _column in range(case.n)
        )
        return lhs, rhs
    if case.profile_name == "BF16_ROUNDING":
        lhs = (1.0,) * (case.batch * case.m * case.k)
        rhs = tuple(
            (
                1.0 + (column % 8) / 128.0
                if inner == 0
                else 1.0 / 512.0
                if column % 2 == 0
                else 1.0 / 4096.0
                if inner <= case.k // 2
                else -1.0 / 4096.0
            )
            for _batch in range(case.batch)
            for inner in range(case.k)
            for column in range(case.n)
        )
        return lhs, rhs
    if case.profile_name == "BF16_SIGNED_ZERO":
        lhs = tuple(
            (
                0.0
                if inner == 0
                else -0.0
                if inner == 1
                else 1.0
            )
            for _batch in range(case.batch)
            for _row in range(case.m)
            for inner in range(case.k)
        )
        rhs = tuple(
            (
                1.0
                if column % 5 == 0 and inner == 0
                else -1.0
                if column % 5 == 1 and inner == 0
                else 1.0
                if column % 5 == 2 and inner == 1
                else -1.0
                if column % 5 == 3 and inner == 1
                else 1.0
                if column % 5 == 4 and inner == 2
                else 0.0
            )
            for _batch in range(case.batch)
            for inner in range(case.k)
            for column in range(case.n)
        )
        return lhs, rhs
    if case.profile_name == "BF16_SUBNORMAL":
        values = (
            2.0**-133,
            -(2.0**-133),
            127.0 * (2.0**-133),
            -127.0 * (2.0**-133),
            2.0**-126,
            -(2.0**-126),
        )
        lhs = tuple(
            values[inner % len(values)]
            for _batch in range(case.batch)
            for _row in range(case.m)
            for inner in range(case.k)
        )
        rhs = tuple(
            (
                1.0
                if column % 8 < 6
                and inner == column % 8
                else 1.0
                if column % 8 == 6
                else -1.0
                if column % 8 == 7 and inner % 2
                else 1.0
                if column % 8 == 7
                else 0.0
            )
            for _batch in range(case.batch)
            for inner in range(case.k)
            for column in range(case.n)
        )
        return lhs, rhs
    if case.profile_name == "BF16_OVERFLOW_INF":
        values = (
            3.3895313892515355e38,
            -3.3895313892515355e38,
            math.inf,
            -math.inf,
            2.0,
            -2.0,
        )
        lhs = tuple(
            values[inner] if inner < len(values) else 1.0
            for _batch in range(case.batch)
            for _row in range(case.m)
            for inner in range(case.k)
        )
        rhs = tuple(
            (
                2.0
                if column % 8 in {0, 1, 5, 6}
                and inner == column % 8
                else 1.0
                if column % 8 in {2, 3}
                and inner == column % 8
                else 1.0
                if column % 8 == 4 and inner in {2, 3}
                else 1.0
                if column % 8 == 7 and inner >= len(values)
                else 0.0
            )
            for _batch in range(case.batch)
            for inner in range(case.k)
            for column in range(case.n)
        )
        return lhs, rhs
    if case.profile_name == "BF16_NAN":
        lhs = tuple(
            (
                math.nan
                if row % (len(BF16_NAN_INPUT_BITS) + 1)
                < len(BF16_NAN_INPUT_BITS)
                and inner == 0
                else 1.0
            )
            for _batch in range(case.batch)
            for row in range(case.m)
            for inner in range(case.k)
        )
        rhs = tuple(
            1.0 if _mix32(inner * case.n + column + 0x51A7) & 1 else -1.0
            for _batch in range(case.batch)
            for inner in range(case.k)
            for column in range(case.n)
        )
        return lhs, rhs
    raise RuntimeError(f"{case.name}: unknown GEMM numeric profile")


def _gemm_expected_values(
    case: NECase,
    lhs: tuple[float, ...],
    rhs: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        sum(
            lhs[
                (
                    (0 if case.lhs_batch == 1 else batch) * case.m
                    + row
                )
                * case.k
                + inner
            ]
            * rhs[
                (
                    (0 if case.rhs_batch == 1 else batch) * case.k
                    + inner
                )
                * case.n
                + column
            ]
            for inner in range(case.k)
        )
        for batch in range(case.batch)
        for row in range(case.m)
        for column in range(case.n)
    )


def _stored_values(
    values: tuple[float, ...],
    *,
    batch: int,
    rows: int,
    columns: int,
    transpose: int,
) -> tuple[float, ...]:
    def logical(batch_index: int, row: int, column: int) -> float:
        return values[
            (batch_index * rows + row) * columns + column
        ]

    if not transpose:
        return values
    return tuple(
        logical(batch_index, row, column)
        for batch_index in range(batch)
        for column in range(columns)
        for row in range(rows)
    )


@functools.lru_cache(maxsize=None)
def _conv_source_values() -> tuple[float, ...]:
    return tuple(
        1.0 if (batch * 19 + y * 7 + x * 5 + channel * 3) % 2 else -1.0
        for batch in range(2)
        for y in range(17)
        for x in range(19)
        for channel in range(65)
    )


@functools.lru_cache(maxsize=None)
def _conv_weight_values(output_channels: int) -> tuple[float, ...]:
    return tuple(
        1.0
        if (kernel_x * 17 + kernel_y * 11 + output * 5 + channel * 3) % 3
        else -1.0
        for kernel_x in range(3)
        for kernel_y in range(2)
        for output in range(output_channels)
        for channel in range(65)
    )


@functools.lru_cache(maxsize=None)
def _conv_expected_values(output_channels: int) -> tuple[float, ...]:
    source = _conv_source_values()
    weight = _conv_weight_values(output_channels)

    def source_value(
        batch: int, y: int, x: int, channel: int
    ) -> float:
        return source[((batch * 17 + y) * 19 + x) * 65 + channel]

    def weight_value(
        kernel_x: int, kernel_y: int, output: int, channel: int
    ) -> float:
        return weight[
            (
                (kernel_x * 2 + kernel_y) * output_channels
                + output
            )
            * 65
            + channel
        ]

    values: list[float] = []
    for batch in range(2):
        for output_y in range(17):
            for output_x in range(10):
                for output in range(output_channels):
                    result = 0.0
                    for kernel_x in range(3):
                        input_x = output_x * 2 + kernel_x - 2
                        if input_x < 0 or input_x >= 19:
                            continue
                        for kernel_y in range(2):
                            input_y = output_y + kernel_y - 1
                            if input_y < 0 or input_y >= 17:
                                continue
                            for channel in range(65):
                                result += source_value(
                                    batch, input_y, input_x, channel
                                ) * weight_value(
                                    kernel_x,
                                    kernel_y,
                                    output,
                                    channel,
                                )
                    values.append(result)
    return tuple(values)


def _observed_conv_inputs(
    case: NECase,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    source = tuple(
        float((y * 11 + x * 7 + channel * 3) % 17 - 8)
        for _batch in range(1)
        for y in range(4)
        for x in range(4)
        for channel in range(64)
    )
    if case.kind_name == "DEPTHWISE_CONV":
        weight = tuple(
            1.0 if channel % 3 else -1.0
            for _kernel_x in range(1)
            for _kernel_y in range(1)
            for channel in range(64)
            for _multiplier in range(1)
        )
    elif case.kind_name == "BACKWARD_CONV":
        weight = tuple(
            1.0
            if output_channel == input_channel
            else -1.0
            if (output_channel + input_channel) % 31 == 0
            else 0.0
            for _kernel_x in range(1)
            for _kernel_y in range(1)
            for output_channel in range(64)
            for input_channel in range(64)
        )
    else:
        raise RuntimeError(f"{case.name}: unknown observed convolution kind")
    return source, weight


def _option_auxiliary(
    case: NECase,
) -> tuple[tuple[float, ...], tuple[int, ...], str]:
    if case.option_name == "BIAS":
        values = tuple(float((column % 7) - 3) for column in range(case.n))
        return values, (case.n,), "Cx"
    if case.option_name == "PSUM":
        count = math.prod(case.output_shape)
        values = tuple(float((index % 5) - 2) for index in range(count))
        return values, case.output_shape, case.output_layout
    if case.option_name in {
        "POSITIVE_AXIS_SCALE",
        "NEGATIVE_AXIS_SCALE",
    }:
        values = tuple(
            0.5 if column % 2 else 2.0 for column in range(case.n)
        )
        return values, (case.n,), "Cx"
    return (), (), "Tensor"


def _apply_exact_option(
    case: NECase,
    base: tuple[float, ...],
    auxiliary: tuple[float, ...],
) -> tuple[float, ...] | None:
    if not case.exact:
        return None
    if case.option_name == "NONE":
        return base
    if case.option_name == "BIAS":
        return tuple(
            value + auxiliary[index % case.n]
            for index, value in enumerate(base)
        )
    if case.option_name == "PSUM":
        return tuple(
            value + extra
            for value, extra in zip(base, auxiliary, strict=True)
        )
    raise RuntimeError(
        f"{case.name}: observed option must not request an exact oracle"
    )


def build_case_payload(case: NECase, sample: int = 0) -> CasePayload:
    if not case.is_safe:
        raise RuntimeError(f"{case.name}: deferred/negative case is not issued")
    if case.kind_name == "GEMM":
        lhs_logical, rhs_logical = _gemm_inputs(case)
        lhs_stored = _stored_values(
            lhs_logical,
            batch=case.lhs_batch,
            rows=case.m,
            columns=case.k,
            transpose=case.lhs_orientation,
        )
        rhs_stored = _stored_values(
            rhs_logical,
            batch=case.rhs_batch,
            rows=case.k,
            columns=case.n,
            transpose=case.rhs_orientation,
        )
        base_expected = _gemm_expected_values(
            case, lhs_logical, rhs_logical
        )
    elif case.kind_name == "CONV":
        lhs_stored = _conv_source_values()
        rhs_stored = _conv_weight_values(case.n)
        base_expected = _conv_expected_values(case.n)
    elif case.kind_name in {"DEPTHWISE_CONV", "BACKWARD_CONV"}:
        lhs_stored, rhs_stored = _observed_conv_inputs(case)
        base_expected = ()
    else:
        raise RuntimeError(f"{case.name}: unknown NE instruction kind")
    lhs = physical.pack_scalar_bytes(
        case.lhs_shape,
        case.lhs_layout,
        case.element_bytes,
        _encoded_operand(case, "lhs", lhs_stored),
    )
    rhs = physical.pack_scalar_bytes(
        case.rhs_shape,
        case.rhs_layout,
        case.element_bytes,
        _encoded_operand(case, "rhs", rhs_stored),
    )
    auxiliary_values, auxiliary_shape, auxiliary_layout = (
        _option_auxiliary(case)
    )
    expected_values = _apply_exact_option(
        case, base_expected, auxiliary_values
    )
    expected_logical: tuple[bytes, ...] | None = None
    expected_physical: bytes | None = None
    if expected_values is not None:
        expected_logical = tuple(
            _encode(case.dtype_name, value) for value in expected_values
        )
        expected_physical = physical.pack_scalar_bytes(
            case.output_shape,
            case.output_layout,
            case.element_bytes,
            expected_logical,
            padding=OUTPUT_PADDING,
        )

    slots = [bytearray([SLOT_CANARY] * SLOT_BYTES) for _ in range(4)]
    for slot, data in ((slots[0], lhs), (slots[1], rhs)):
        slot[BODY_OFFSET : BODY_OFFSET + len(data)] = data
    if auxiliary_values:
        auxiliary = physical.pack_scalar_bytes(
            auxiliary_shape,
            auxiliary_layout,
            2,
            (
                _encode(case.dtype_name, value)
                for value in auxiliary_values
            ),
        )
        if len(auxiliary) != case.aux_span:
            raise RuntimeError(f"{case.name}: auxiliary span mismatch")
        slots[3][BODY_OFFSET : BODY_OFFSET + len(auxiliary)] = auxiliary
    output_seed = physical.pack_scalar_bytes(
        case.output_shape,
        case.output_layout,
        case.element_bytes,
        (
            _encode(case.dtype_name, -113.0 if case.dtype_name == "I8" else -13.0)
            for _ in range(math.prod(case.output_shape))
        ),
        padding=SLOT_CANARY,
    )
    slots[2][BODY_OFFSET : BODY_OFFSET + len(output_seed)] = output_seed

    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
    words[REQ["CASE"]] = case.case_id
    words[REQ["DTYPE"]] = case.dtype
    words[REQ["LHS_ORIENTATION"]] = case.lhs_orientation
    words[REQ["RHS_ORIENTATION"]] = case.rhs_orientation
    words[REQ["BATCH"]] = case.batch
    words[REQ["M"]] = case.m
    words[REQ["K"]] = case.k
    words[REQ["N"]] = case.n
    words[REQ["LHS_SPAN"]] = case.lhs_span
    words[REQ["RHS_SPAN"]] = case.rhs_span
    words[REQ["OUTPUT_SPAN"]] = case.output_span
    words[REQ["SAMPLE"]] = sample
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["SLOT_BYTES"]] = SLOT_BYTES
    words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    words[REQ["KIND"]] = case.kind
    words[REQ["PROFILE"]] = case.profile
    words[REQ["OPTION"]] = case.option
    words[REQ["AUX_SPAN"]] = case.aux_span
    words[REQ["DISPOSITION"]] = case.disposition
    words[REQ["BATCH_PAIR"]] = (
        (case.lhs_batch << 32) | case.rhs_batch
    )
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    return CasePayload(
        request + bytes([SLOT_CANARY]) * (RESOURCE_BYTES - len(request)),
        b"".join(bytes(slot) for slot in slots),
        expected_logical,
        expected_physical,
    )
