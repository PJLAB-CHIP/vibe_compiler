#!/usr/bin/env python3
"""Typed cases and host references for the TX81 instruction probe."""

from __future__ import annotations

import dataclasses
import pathlib
import re
import struct
from collections.abc import Callable, Iterable

import wafer_physical_tensor_codec as physical


REQUEST_MAGIC = 0x3151455246494657
RECORD_MAGIC = 0x3143455246494657
SCHEMA = 1
REQUEST_WORDS = 16
RECORD_WORDS = 32
RESOURCE_BYTES = 16384
SLOT_BYTES = 4096
BODY_OFFSET = 256
OUTPUT_DDR_OFFSET = 4096
SLOT_CANARY = 0xA7
BIT2FP_TRUE_WORDS = {"F16": 0x3C00, "BF16": 0x3F80}
BIT2FP_FALSE_WORD = 0x0000
REQUEST_GUARD = 0x8E7C6A5948372615
RECORD_GUARD = 0x192A3B4C5D6E7F80

DISPOSITIONS = {"SAFE": 0, "DEFERRED": 1}
FAMILIES = {
    "CT_ELEMENTWISE": 0,
    "CT_CONVERT": 1,
    "CT_REDUCE": 2,
    "CT_SELECT_COMPOSITE": 3,
    "NE_GEMM": 4,
    "TDMA_PAD": 5,
    "TDMA_IMG2COL": 6,
    "CONV": 7,
    "POOL": 8,
    "UNPOOL": 9,
    "PERIPHERAL": 10,
}
DTYPES = {
    "F16": 0,
    "BF16": 1,
    "I8_TO_F16": 2,
    "I8_TO_BF16": 3,
    "BF16_TO_F16": 4,
    "F16_TO_BF16": 5,
    "F16_TO_I16": 6,
    "F32": 7,
}
ORACLES = {"EXACT_BITS": 0, "EXACT_COMPOSITE": 1, "NO_ORACLE": 2}
REASONS = {
    "REASON_NONE": 0,
    "REASON_ISOLATED_COMPLETION_WRITEBACK_UNQUALIFIED": 1,
    "REASON_GEOMETRY_UNQUALIFIED": 2,
    "REASON_NUMERIC_UNQUALIFIED": 3,
}

REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "DISPOSITION": 3,
    "FAMILY": 4,
    "DTYPE": 5,
    "ORACLE": 6,
    "RESULT_BYTES": 7,
    "OUTPUT_SPAN": 8,
    "AUX_SPAN": 9,
    "SAMPLE": 10,
    "RESOURCE_BYTES": 11,
    "SLOT_BYTES": 12,
    "BODY_OFFSET": 13,
    "GUARD": 15,
}
REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "DISPOSITION": 4,
    "FAMILY": 5,
    "DTYPE": 6,
    "ORACLE": 7,
    "RESULT_BYTES": 8,
    "OUTPUT_SPAN": 9,
    "AUX_SPAN": 10,
    "SAMPLE": 11,
    "OUTPUT_GUARD_MISMATCHES": 12,
    "AUX_GUARD_MISMATCHES": 13,
    "REQUEST_GUARD": 14,
    "OUTPUT_DDR_OFFSET": 15,
    "SLOT_BYTES": 16,
    "BODY_OFFSET": 17,
    "STEP_FLAGS": 18,
    "BIT2FP_MISMATCHES": 19,
    "RECORD_GUARD": 31,
}

STEP_TARGET_ISSUED = 1 << 0
STEP_BIT2FP_COMPLETED = 1 << 1
STEP_MASK_MOVE_ISSUED = 1 << 2
STEP_FINAL_FENCE_COMPLETED = 1 << 3

_HEADER = (
    pathlib.Path(__file__).resolve().parent
    / "Inputs"
    / "wafer_instruction_family_probe_protocol.h"
)
_ROW = re.compile(
    r"\bX\(\s*([A-Z0-9_]+)\s*,\s*(\d+)\s*,\s*\"([^\"]+)\"\s*,"
    r"\s*([A-Z0-9_]+)\s*,\s*([A-Z0-9_]+)\s*,\s*([A-Z0-9_]+)\s*,"
    r"\s*([A-Z0-9_]+)\s*,\s*([A-Z0-9_]+)\s*,\s*(\d+)\s*,"
    r"\s*(\d+)\s*,\s*(\d+)\s*\)"
)


@dataclasses.dataclass(frozen=True)
class InstructionCase:
    symbol: str
    case_id: int
    name: str
    disposition_name: str
    family_name: str
    dtype_name: str
    oracle_name: str
    reason_name: str
    result_bytes: int
    output_span: int
    aux_span: int

    @property
    def disposition(self) -> int:
        return DISPOSITIONS[self.disposition_name]

    @property
    def family(self) -> int:
        return FAMILIES[self.family_name]

    @property
    def dtype(self) -> int:
        return DTYPES[self.dtype_name]

    @property
    def oracle(self) -> int:
        return ORACLES[self.oracle_name]

    @property
    def reason(self) -> int:
        return REASONS[self.reason_name]

    @property
    def is_safe(self) -> bool:
        return self.disposition_name == "SAFE"

    @property
    def is_observation(self) -> bool:
        return self.is_safe and self.oracle_name == "NO_ORACLE"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "disposition": self.disposition_name.lower(),
            "family": self.family_name.lower(),
            "dtype": self.dtype_name.lower(),
            "oracle": self.oracle_name.lower(),
            "deferred_reason": (
                None
                if self.reason_name == "REASON_NONE"
                else self.reason_name.removeprefix("REASON_").lower()
            ),
            "result_bytes": self.result_bytes,
            "output_span": self.output_span,
            "aux_span": self.aux_span,
        }


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_output_slot: bytes


def load_catalog(header: pathlib.Path = _HEADER) -> tuple[InstructionCase, ...]:
    flattened = header.read_text().replace("\\\n", " ")
    rows = tuple(
        InstructionCase(
            symbol=match.group(1),
            case_id=int(match.group(2)),
            name=match.group(3),
            disposition_name=match.group(4),
            family_name=match.group(5),
            dtype_name=match.group(6),
            oracle_name=match.group(7),
            reason_name=match.group(8),
            result_bytes=int(match.group(9)),
            output_span=int(match.group(10)),
            aux_span=int(match.group(11)),
        )
        for match in _ROW.finditer(flattened)
    )
    if not rows:
        raise RuntimeError("instruction qualification catalog is empty")
    ids = {case.case_id for case in rows}
    names = {case.name for case in rows}
    symbols = {case.symbol for case in rows}
    if len(ids) != len(rows) or len(names) != len(rows) or len(symbols) != len(rows):
        raise RuntimeError("instruction qualification catalog keys are not unique")
    for case in rows:
        # Force every closed token through the Python-side typed enum tables.
        _ = (
            case.disposition,
            case.family,
            case.dtype,
            case.oracle,
            case.reason,
        )
        if case.is_safe != (case.reason == REASONS["REASON_NONE"]):
            raise RuntimeError(f"{case.name}: disposition/reason disagree")
        if case.is_safe and (
            case.result_bytes <= 0
            or case.output_span < case.result_bytes
            or case.output_span > SLOT_BYTES - BODY_OFFSET
            or case.aux_span > SLOT_BYTES - BODY_OFFSET
        ):
            raise RuntimeError(f"{case.name}: safe geometry is invalid")
        if not case.is_safe and any(
            (case.result_bytes, case.output_span, case.aux_span)
        ):
            raise RuntimeError(f"{case.name}: deferred row carries geometry")
    return rows


CATALOG = load_catalog()
CASES_BY_NAME = {case.name: case for case in CATALOG}
SAFE_CASES = tuple(case for case in CATALOG if case.is_safe)


def _repeat(values: Iterable[float], count: int = 128) -> list[float]:
    source = tuple(values)
    return [source[index % len(source)] for index in range(count)]


def _f16(values: Iterable[float]) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}e", *values)


def _f16_from_bits(bits: Iterable[int]) -> bytes:
    bits = tuple(bits)
    return struct.pack(f"<{len(bits)}H", *bits)


def _f32(values: Iterable[float]) -> bytes:
    values = tuple(values)
    return struct.pack(f"<{len(values)}f", *values)


def _bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def _bf16(values: Iterable[float]) -> bytes:
    encoded = tuple(_bf16_bits(value) for value in values)
    return struct.pack(f"<{len(encoded)}H", *encoded)


def _fp(dtype_name: str, values: Iterable[float]) -> bytes:
    if dtype_name in ("F16", "I8_TO_F16", "BF16_TO_F16"):
        return _f16(values)
    if dtype_name in ("BF16", "I8_TO_BF16", "F16_TO_BF16"):
        return _bf16(values)
    if dtype_name == "F32":
        return _f32(values)
    raise RuntimeError(f"no floating encoder for {dtype_name}")


def _slot() -> bytearray:
    return bytearray([SLOT_CANARY] * SLOT_BYTES)


def _put(slot: bytearray, raw: bytes) -> None:
    end = BODY_OFFSET + len(raw)
    if end > len(slot):
        raise RuntimeError("instruction probe payload exceeds one SPM slot")
    slot[BODY_OFFSET:end] = raw


def _special_add_zero(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol == "CT_ADD_SPECIAL_F16":
        distinguishing = (
            (0x0000, 0x0000, 0x0000),
            (0x8000, 0x0000, 0x0000),
            (0x0000, 0x8000, 0x0000),
            (0x8000, 0x8000, 0x8000),
            (0x7C00, 0x0000, 0x7C00),
            (0xFC00, 0x0000, 0xFC00),
            (0x7BFF, 0x0000, 0x7BFF),
            (0xFBFF, 0x0000, 0xFBFF),
            (0x0400, 0x0000, 0x0400),
            (0x8400, 0x0000, 0x8400),
            (0x0001, 0x0000, 0x0001),
            (0x8001, 0x0000, 0x8001),
        )
    elif case.symbol == "CT_ADD_SPECIAL_BF16":
        distinguishing = (
            (0x0000, 0x0000, 0x0000),
            (0x8000, 0x0000, 0x0000),
            (0x0000, 0x8000, 0x0000),
            (0x8000, 0x8000, 0x8000),
            (0x7F80, 0x0000, 0x7F80),
            (0xFF80, 0x0000, 0xFF80),
            (0x7F7F, 0x0000, 0x7F7F),
            (0xFF7F, 0x0000, 0xFF7F),
            (0x0080, 0x0000, 0x0080),
            (0x8080, 0x0000, 0x8080),
            (0x0001, 0x0000, 0x0001),
            (0x8001, 0x0000, 0x8001),
        )
    else:
        raise RuntimeError(f"{case.name}: unknown special-value add kind")
    vectors = tuple(
        distinguishing[index % len(distinguishing)] for index in range(128)
    )
    return (
        _f16_from_bits(entry[0] for entry in vectors),
        _f16_from_bits(entry[1] for entry in vectors),
        _f16_from_bits(entry[2] for entry in vectors),
    )


def _arithmetic(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol in ("CT_ADD_SPECIAL_F16", "CT_ADD_SPECIAL_BF16"):
        return _special_add_zero(case)
    element_count = (
        130
        if case.symbol in ("CT_ADD_F16_TAIL130", "CT_ADD_BF16_TAIL130")
        else 128
    )
    if case.symbol == "CT_ADD_F32":
        base_a = _repeat(
            (-16.0, -7.0, -1.0, 0.0, 1.0, 7.0, 16.0, 31.0),
            element_count,
        )
        base_b = _repeat(
            (3.0, -2.0, 5.0, 9.0, -1.0, 4.0, -8.0, 2.0),
            element_count,
        )
    else:
        base_a = _repeat(
            (-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0),
            element_count,
        )
        base_b = _repeat(
            (1.0, -1.0, 2.0, 2.0, 0.5, -2.0, 4.0, -0.5),
            element_count,
        )
    operation = case.name.split("-")[1]
    if operation == "pow2":
        base_a = _repeat((-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0))
        expected = [2.0**value for value in base_a]
    elif operation == "relu":
        expected = [max(value, 0.0) for value in base_a]
    elif operation == "neg":
        expected = [-value for value in base_a]
    else:
        binary: dict[str, Callable[[float, float], float]] = {
            "add": lambda lhs, rhs: lhs + rhs,
            "sub": lambda lhs, rhs: lhs - rhs,
            "mul": lambda lhs, rhs: lhs * rhs,
            "max": max,
            "min": min,
        }
        expected = [
            binary[operation](lhs, rhs)
            for lhs, rhs in zip(base_a, base_b, strict=True)
        ]
    return _fp(case.dtype_name, base_a), _fp(case.dtype_name, base_b), _fp(
        case.dtype_name, expected
    )


def _convert(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    integer_values = tuple((index % 15) - 7 for index in range(128))
    floating_values = tuple(float(value) for value in integer_values)
    if case.dtype_name == "I8_TO_F16":
        return struct.pack("<128b", *integer_values), b"", _f16(floating_values)
    if case.dtype_name == "I8_TO_BF16":
        return struct.pack("<128b", *integer_values), b"", _bf16(floating_values)
    if case.dtype_name == "BF16_TO_F16":
        return _bf16(floating_values), b"", _f16(floating_values)
    if case.dtype_name == "F16_TO_BF16":
        tie_and_fraction_bits = (
            0x3C0C,  # positive tie, retained bf16 LSB odd: round upward
            0x3C04,  # positive tie, retained bf16 LSB even: remain even
            0x3C0D,  # greater than half
            0x3C0B,  # less than half
            0xBC0C,  # negative tie with odd retained bf16 LSB
            0xBC04,  # negative tie with even retained bf16 LSB
            0x400C,
            0xC00C,
        )
        source_bits = tuple(
            tie_and_fraction_bits[index % len(tie_and_fraction_bits)]
            for index in range(128)
        )
        source = _f16_from_bits(source_bits)
        source_values = struct.unpack("<128e", source)
        return source, b"", _bf16(source_values)
    if case.dtype_name == "F16_TO_I16":
        fractions = _repeat(
            (1.5, 2.5, -1.5, -2.5, 3.25, -3.75, 0.5, -0.5)
        )
        return (
            _f16(fractions),
            b"",
            struct.pack("<128h", *(round(value) for value in fractions)),
        )
    raise RuntimeError(f"{case.name}: unknown convert signature")


_REDUCE_MATRIX_BASE = 144
_REDUCE_MATRIX_END = 191
_REDUCE_CX_BASE = 192
_REDUCE_CX_END = 195
_REDUCE_RAW_BASE = 196
_REDUCE_RAW_END = 203
_REDUCE_OPERATIONS = ("sum", "avg", "max", "min")
_REDUCE_DTYPES = ("F16", "BF16", "F32")
_REDUCE_AXES = ("C", "W", "H", "HW")


def _reduce_config(
    case: InstructionCase,
) -> tuple[str, str, str, tuple[int, int, int, int]]:
    if _REDUCE_MATRIX_BASE <= case.case_id <= _REDUCE_MATRIX_END:
        offset = case.case_id - _REDUCE_MATRIX_BASE
        operation = _REDUCE_OPERATIONS[offset // 12]
        dtype_name = _REDUCE_DTYPES[(offset % 12) // 4]
        axis = _REDUCE_AXES[offset % 4]
        shape = {
            "C": (1, 2, 3, 65),
            "W": (1, 1, 4, 64),
            "H": (1, 3, 2, 65),
            "HW": (1, 3, 2, 65),
        }[axis]
        return operation, dtype_name, axis, shape
    if _REDUCE_CX_BASE <= case.case_id <= _REDUCE_CX_END:
        return (
            _REDUCE_OPERATIONS[case.case_id - _REDUCE_CX_BASE],
            "F16",
            "C",
            (1, 1, 4, 8),
        )
    if _REDUCE_RAW_BASE <= case.case_id <= _REDUCE_RAW_END:
        offset = case.case_id - _REDUCE_RAW_BASE
        axis = ("N", "HWC")[offset % 2]
        shape = (2, 1, 1, 64) if axis == "N" else (1, 2, 2, 64)
        return _REDUCE_OPERATIONS[offset // 2], "F16", axis, shape
    return case.name.split("-")[1], case.dtype_name, "W", (1, 1, 4, 64)


def _reduce_element_bytes(dtype_name: str) -> int:
    return 4 if dtype_name == "F32" else 2


def _reduce_layout_name(case: InstructionCase) -> str:
    return (
        "Cx"
        if _REDUCE_CX_BASE <= case.case_id <= _REDUCE_CX_END
        else "NCx"
    )


def _reduce_physical_shape(
    case: InstructionCase,
    axis: str,
    shape: tuple[int, int, int, int],
    *,
    output: bool,
) -> tuple[int, ...]:
    n_size, h_size, w_size, c_size = shape
    if output:
        if axis == "C":
            c_size = 1
        elif axis == "W":
            w_size = 1
        elif axis == "H":
            h_size = 1
        elif axis == "HW":
            h_size = 1
            w_size = 1
    if _reduce_layout_name(case) == "Cx":
        return (w_size, c_size)
    return (n_size, h_size, w_size, c_size)


def reduce_exact_result_byte_offsets(
    case: InstructionCase,
) -> tuple[int, ...]:
    """Return exact logical-result bytes inside the output SPM slot.

    Native Reduce retains a four-dimensional result shape and packs it in
    NCx/Cx storage.  For channel-tail geometries the logical scalars are not a
    compact prefix of the hardware write span, so exact checking must follow
    physical offsets while leaving internal padding unconstrained.
    """

    if not (
        _REDUCE_MATRIX_BASE <= case.case_id <= _REDUCE_CX_END
        and case.oracle_name == "EXACT_BITS"
    ):
        return tuple(range(BODY_OFFSET, BODY_OFFSET + case.result_bytes))

    _operation, dtype_name, axis, shape = _reduce_config(case)
    element_bytes = _reduce_element_bytes(dtype_name)
    output_shape = _reduce_physical_shape(
        case, axis, shape, output=True
    )
    layout_name = _reduce_layout_name(case)
    layout = physical.physical_layout(
        output_shape, layout_name, element_bytes
    )
    if layout.physical_bytes != case.output_span:
        raise RuntimeError(
            f"{case.name}: physical output span {layout.physical_bytes} "
            f"does not match catalog {case.output_span}"
        )
    offsets = tuple(
        BODY_OFFSET
        + physical.physical_element_offset(layout, coordinate)
        * element_bytes
        + byte
        for coordinate in physical.coordinates(output_shape)
        for byte in range(element_bytes)
    )
    if len(offsets) != case.result_bytes:
        raise RuntimeError(
            f"{case.name}: physical logical byte count {len(offsets)} "
            f"does not match catalog {case.result_bytes}"
        )
    return offsets


def _apply_reduce(operation: str, values: list[float]) -> float:
    if operation == "sum":
        return sum(values)
    if operation == "avg":
        return sum(values) / len(values)
    if operation == "max":
        return max(values)
    if operation == "min":
        return min(values)
    raise RuntimeError(f"unknown reduce kind: {operation}")


def _reduce(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    operation, dtype_name, axis, (n_size, h_size, w_size, c_size) = (
        _reduce_config(case)
    )
    if dtype_name != case.dtype_name:
        raise RuntimeError(f"{case.name}: encoded dtype and case dtype disagree")

    dimension_sizes = (n_size, h_size, w_size, c_size)
    reduced_dimensions = {
        "C": (3,),
        "W": (2,),
        "H": (1,),
        "HW": (1, 2),
        "N": (0,),
        "HWC": (1, 2, 3),
    }[axis]
    retained_dimensions = tuple(
        dimension
        for dimension in range(4)
        if dimension not in reduced_dimensions
    )

    def linear_ordinal(
        coordinate: tuple[int, int, int, int],
        dimensions: tuple[int, ...],
    ) -> int:
        ordinal = 0
        for dimension in dimensions:
            ordinal = (
                ordinal * dimension_sizes[dimension]
                + coordinate[dimension]
            )
        return ordinal

    def value(n: int, h: int, w: int, c: int) -> float:
        coordinate = (n, h, w, c)
        base = float(
            1 << (linear_ordinal(coordinate, retained_dimensions) % 5)
        )
        if operation in ("sum", "avg"):
            # Repeating powers of two keeps every partial sum exact in
            # f16/bf16/f32, including the C=65 tail geometry.
            return base
        return base + linear_ordinal(
            coordinate, reduced_dimensions
        ) * 0.25

    values = [
        value(n, h, w, c)
        for n in range(n_size)
        for h in range(h_size)
        for w in range(w_size)
        for c in range(c_size)
    ]
    groups: list[list[float]] = []
    if axis == "C":
        groups = [
            [value(n, h, w, c) for c in range(c_size)]
            for n in range(n_size)
            for h in range(h_size)
            for w in range(w_size)
        ]
    elif axis == "W":
        groups = [
            [value(n, h, w, c) for w in range(w_size)]
            for n in range(n_size)
            for h in range(h_size)
            for c in range(c_size)
        ]
    elif axis == "H":
        groups = [
            [value(n, h, w, c) for h in range(h_size)]
            for n in range(n_size)
            for w in range(w_size)
            for c in range(c_size)
        ]
    elif axis == "HW":
        groups = [
            [
                value(n, h, w, c)
                for h in range(h_size)
                for w in range(w_size)
            ]
            for n in range(n_size)
            for c in range(c_size)
        ]
    elif axis == "N":
        groups = [
            [value(n, h, w, c) for n in range(n_size)]
            for h in range(h_size)
            for w in range(w_size)
            for c in range(c_size)
        ]
    elif axis == "HWC":
        groups = [
            [
                value(n, h, w, c)
                for h in range(h_size)
                for w in range(w_size)
                for c in range(c_size)
            ]
            for n in range(n_size)
        ]
    else:
        raise RuntimeError(f"{case.name}: unknown reduce axis {axis}")
    encoded_values = _fp(dtype_name, values)
    element_bytes = _reduce_element_bytes(dtype_name)
    logical_scalars = tuple(
        encoded_values[offset : offset + element_bytes]
        for offset in range(0, len(encoded_values), element_bytes)
    )
    input_shape = _reduce_physical_shape(
        case,
        axis,
        (n_size, h_size, w_size, c_size),
        output=False,
    )
    input_bytes = physical.pack_scalar_bytes(
        input_shape,
        _reduce_layout_name(case),
        element_bytes,
        logical_scalars,
        padding=SLOT_CANARY,
    )
    if case.is_observation:
        return input_bytes, b"", bytes(case.result_bytes)
    expected = [_apply_reduce(operation, group) for group in groups]
    return input_bytes, b"", _fp(dtype_name, expected)


def _select(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    true_values = _repeat((1.0, 2.0, 3.0, 4.0))
    false_values = _repeat((-1.0, -2.0, -3.0, -4.0))
    mask = bytes(0xFF if byte % 2 == 0 else 0x00 for byte in range(16))
    expected = [
        true_value if (index // 8) % 2 == 0 else false_value
        for index, (true_value, false_value) in enumerate(
            zip(true_values, false_values, strict=True)
        )
    ]
    return _fp(case.dtype_name, true_values), mask, _fp(
        case.dtype_name, expected
    )


def _gemm(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol in ("GEMM_F16", "GEMM_BF16"):
        lhs = _repeat((1.0, 2.0, 3.0, 4.0), 16)
        rhs = [
            1.0 if row == column else 0.0
            for row in range(16)
            for column in range(16)
        ]
        expected = lhs
    elif case.symbol == "GEMM_BF16_ACCUM_ROUND":
        lhs = [1.0] * 16
        rhs = [
            1.0 + column / 128.0
            if row == 0
            else 1.0 / 512.0
            if column % 2 == 0
            else 1.0 / 4096.0
            for row in range(16)
            for column in range(16)
        ]
        expected = [
            1.0 + (column + 4) / 128.0
            if column % 2 == 0
            else 1.0 + column / 128.0
            for column in range(16)
        ]
    elif case.symbol == "GEMM_F16_ACCUM_ROUND":
        lhs = [1.0] * 16
        rhs = [
            1.0 + column / 1024.0
            if row == 0
            else 1.0 / 4096.0
            if column % 2 == 0
            else 1.0 / 8192.0
            if row <= 8
            else -1.0 / 8192.0
            for row in range(16)
            for column in range(16)
        ]
        expected = [
            1.0 + (column + 4) / 1024.0
            if column % 2 == 0
            else 1.0 + column / 1024.0
            for column in range(16)
        ]
    elif case.symbol == "GEMM_F16_M4":
        lhs = [
            float(
                column + 1
                if row == 0
                else 32 + column + 1
                if row == 1
                else -(64 + column + 1)
                if row == 2
                else 96 + column + 1
            )
            for row in range(4)
            for column in range(16)
        ]
        rhs = [
            1.0 if row == column else 0.0
            for row in range(16)
            for column in range(16)
        ]
        expected = lhs
    elif case.symbol in ("GEMM_F16_BATCH2_M8", "GEMM_BF16_BATCH2_M8"):
        lhs = [
            float(batch * 128 + row * 16 + column + 1)
            for batch in range(2)
            for row in range(8)
            for column in range(16)
        ]
        rhs = [
            float(
                1
                if batch == 0 and row == column
                else -1
                if batch == 1 and row == column
                else 0
            )
            for batch in range(2)
            for row in range(16)
            for column in range(16)
        ]
        expected = lhs[:128] + [-value for value in lhs[128:]]
    elif case.symbol == "GEMM_F16_N17":
        lhs = [1.0] * 16
        rhs = [
            float(
                column + 1
                if column < 17 and row == column % 16
                else -29
                if column >= 17
                else 0
            )
            for row in range(16)
            for column in range(32)
        ]
        expected = [float(column + 1) for column in range(17)]
    elif case.symbol == "GEMM_F16_K17":
        lhs = (
            [float(index + 1) for index in range(17)]
            + [-29.0] * 15
            + [0.0] * 96
        )
        rhs = (
            [
                float(1 if row < 16 else column + 2)
                for row in range(17)
                for column in range(16)
            ]
            + [-31.0] * 112
        )
        expected = [float(170 + 17 * column) for column in range(16)]
    elif case.symbol == "GEMM_BF16_K17":
        lhs = [1.0] * 17 + [-29.0] * 15 + [0.0] * 96
        rhs = (
            [
                float(1 if row < 16 else column + 2)
                for row in range(17)
                for column in range(16)
            ]
            + [-31.0] * 112
        )
        expected = [float(18 + column) for column in range(16)]
    elif case.symbol in ("GEMM_F16_N65", "GEMM_BF16_N65"):
        lhs = [1.0] * 16
        semantic_rhs = [
            float(1 if row < 15 else 100 + column)
            for row in range(16)
            for column in range(65)
        ]
        rhs = (
            [
                semantic_rhs[row * 65 + column]
                for row in range(16)
                for column in range(64)
            ]
            + [
                value
                for row in range(16)
                for value in (
                    semantic_rhs[row * 65 + 64],
                    -29.0,
                    -29.0,
                    -29.0,
                )
            ]
            + [-31.0] * 64
        )
        expected = [float(115 + column) for column in range(65)]
    elif case.symbol in (
        "GEMM_F16_ORIENTED_NT",
        "GEMM_BF16_ORIENTED_NT",
    ):
        semantic_lhs = [
            float((row * 5 + contracting * 3) % 7 - 3)
            for row in range(4)
            for contracting in range(16)
        ]
        semantic_rhs = [
            float((contracting * 2 + column * 5) % 9 - 4)
            for contracting in range(16)
            for column in range(8)
        ]
        lhs = semantic_lhs
        rhs = [
            semantic_rhs[contracting * 8 + column]
            for column in range(8)
            for contracting in range(16)
        ]
        expected = [
            float(
                sum(
                    semantic_lhs[row * 16 + contracting]
                    * semantic_rhs[contracting * 8 + column]
                    for contracting in range(16)
                )
            )
            for row in range(4)
            for column in range(8)
        ]
    elif case.symbol == "GEMM_F16_ORIENTED_TN":
        semantic_lhs = [
            float((row * 5 + contracting * 3) % 7 - 3)
            for row in range(4)
            for contracting in range(16)
        ]
        semantic_rhs = [
            float((contracting * 2 + column * 5) % 9 - 4)
            for contracting in range(16)
            for column in range(8)
        ]
        lhs = [
            semantic_lhs[row * 16 + contracting]
            for contracting in range(16)
            for row in range(4)
        ]
        rhs = semantic_rhs
        expected = [
            float(
                sum(
                    semantic_lhs[row * 16 + contracting]
                    * semantic_rhs[contracting * 8 + column]
                    for contracting in range(16)
                )
            )
            for row in range(4)
            for column in range(8)
        ]
    elif case.symbol == "GEMM_F16_ORIENTED_TT":
        semantic_lhs = [
            float((row * 5 + contracting * 3) % 7 - 3)
            for row in range(4)
            for contracting in range(16)
        ]
        semantic_rhs = [
            float((contracting * 2 + column * 5) % 9 - 4)
            for contracting in range(16)
            for column in range(8)
        ]
        lhs = [
            semantic_lhs[row * 16 + contracting]
            for contracting in range(16)
            for row in range(4)
        ]
        rhs = [
            semantic_rhs[contracting * 8 + column]
            for column in range(8)
            for contracting in range(16)
        ]
        expected = [
            float(
                sum(
                    semantic_lhs[row * 16 + contracting]
                    * semantic_rhs[contracting * 8 + column]
                    for contracting in range(16)
                )
            )
            for row in range(4)
            for column in range(8)
        ]
    elif case.symbol == "GEMM_F16_PSUM":
        lhs = [1.0] * 16
        rhs = [
            float(column + 1)
            for _row in range(16)
            for column in range(16)
        ]
        expected = [float(1016 + 17 * column) for column in range(16)]
    else:
        raise RuntimeError(f"{case.name}: unknown GEMM kind")
    lhs_physical = lhs + [0.0] * (128 - len(lhs))
    return (
        _fp(case.dtype_name, lhs_physical),
        _fp(case.dtype_name, rhs),
        _fp(case.dtype_name, expected),
    )


def _pad() -> tuple[bytes, bytes, bytes]:
    source = [
        float(1 + row * 2 + column + channel % 4)
        for row in range(2)
        for column in range(2)
        for channel in range(64)
    ]
    expected: list[float] = []
    for row in range(4):
        for column in range(4):
            for channel in range(64):
                if 1 <= row < 3 and 1 <= column < 3:
                    source_index = (
                        ((row - 1) * 2 + (column - 1)) * 64 + channel
                    )
                    expected.append(source[source_index])
                else:
                    expected.append(0.0)
    return _f16(source), b"", _f16(expected)


def _img2col(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol == "TDMA_IMG2COL_F16":
        source = [
            float(1 + 256 * row + 64 * column + channel)
            for row in range(3)
            for column in range(3)
            for channel in range(64)
        ]
    elif case.symbol == "TDMA_IMG2COL_BF16":
        source = [
            float(1 + 64 * row + 16 * column + channel % 8)
            for row in range(3)
            for column in range(3)
            for channel in range(64)
        ]
    else:
        raise RuntimeError(f"{case.name}: unknown Img2Col kind")
    expected = [
        source[((output_row + kernel_y) * 3 + output_column + kernel_x) * 64
               + channel]
        for kernel_y in range(2)
        for kernel_x in range(2)
        for output_row in range(2)
        for output_column in range(2)
        for channel in range(64)
    ]
    return _fp(case.dtype_name, source), b"", _fp(case.dtype_name, expected)


def _conv(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol not in ("CONV_F16", "CONV_BF16"):
        raise RuntimeError(f"{case.name}: unknown convolution kind")
    source = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    weight = (
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
        17.0,
    )
    expected = (1.0, 16.0, 3.0, 11.0, 2.0, 32.0, 5.0, 13.0)
    return (
        _fp(case.dtype_name, source),
        _fp(case.dtype_name, weight),
        _fp(case.dtype_name, expected),
    )


_POOL_SYMMETRIC_BASE = 204
_POOL_SYMMETRIC_END = 221
_POOL_ASYMMETRIC_BASE = 222
_POOL_ASYMMETRIC_END = 227
_POOL_PADDED_BASE = 228
_POOL_PADDED_END = 233
_POOL_TIE_BASE = 234
_POOL_TIE_END = 235
_POOL_OPERATIONS = (
    "avg",
    "sum",
    "max",
    "indexedmax",
    "min",
    "indexedmin",
)


def _pool_config(
    case: InstructionCase,
) -> tuple[str, str, str]:
    if _POOL_SYMMETRIC_BASE <= case.case_id <= _POOL_SYMMETRIC_END:
        offset = case.case_id - _POOL_SYMMETRIC_BASE
        return (
            _POOL_OPERATIONS[offset // 3],
            _REDUCE_DTYPES[offset % 3],
            "symmetric",
        )
    if _POOL_ASYMMETRIC_BASE <= case.case_id <= _POOL_ASYMMETRIC_END:
        return (
            _POOL_OPERATIONS[case.case_id - _POOL_ASYMMETRIC_BASE],
            "F16",
            "asymmetric",
        )
    if _POOL_PADDED_BASE <= case.case_id <= _POOL_PADDED_END:
        return (
            _POOL_OPERATIONS[case.case_id - _POOL_PADDED_BASE],
            "F16",
            "padded",
        )
    if _POOL_TIE_BASE <= case.case_id <= _POOL_TIE_END:
        return (
            ("indexedmax", "indexedmin")[
                case.case_id - _POOL_TIE_BASE
            ],
            "F16",
            "tie",
        )
    raise RuntimeError(f"{case.name}: not a capability-matrix pool case")


def _pool_capability(
    case: InstructionCase,
) -> tuple[bytes, bytes, bytes]:
    operation, dtype_name, geometry = _pool_config(case)
    if dtype_name != case.dtype_name:
        raise RuntimeError(f"{case.name}: encoded dtype and case dtype disagree")
    if geometry in {"symmetric", "tie"}:
        src_h, src_w, dst_h, dst_w = 2, 4, 1, 2
        kernel_x, kernel_y, stride_x, stride_y = 2, 2, 2, 2
        pad_top = pad_left = 0
    elif geometry == "asymmetric":
        src_h, src_w, dst_h, dst_w = 3, 5, 2, 2
        kernel_x, kernel_y, stride_x, stride_y = 3, 2, 2, 1
        pad_top = pad_left = 0
    elif geometry == "padded":
        src_h, src_w, dst_h, dst_w = 2, 3, 2, 2
        kernel_x, kernel_y, stride_x, stride_y = 3, 2, 2, 1
        pad_top = pad_left = 1
    else:
        raise RuntimeError(f"{case.name}: unknown pool geometry {geometry}")

    source = [0.0] * (src_h * src_w * 64)
    for row in range(src_h):
        for column in range(src_w):
            for channel in range(64):
                if geometry in {"asymmetric", "padded"}:
                    value = float(
                        6 * (1 + row * src_w + column) + channel % 4
                    )
                else:
                    value = float(
                        4 * (1 + row * src_w + column) + channel % 4
                    )
                if operation in {"min", "indexedmin"}:
                    value = -value
                source[(row * src_w + column) * 64 + channel] = value

    if geometry == "tie":
        for output_column in range(dst_w):
            for channel in range(64):
                extremum = (
                    float(100 + channel)
                    if operation == "indexedmax"
                    else float(-100 - channel)
                )
                for position in (0, 3):
                    row = position // 2
                    column = 2 * output_column + position % 2
                    source[(row * src_w + column) * 64 + channel] = extremum

    expected_values: list[float] = []
    expected_indices: list[int] = []
    for output_row in range(dst_h):
        for output_column in range(dst_w):
            for channel in range(64):
                window: list[float] = []
                for kernel_row in range(kernel_y):
                    input_row = (
                        output_row * stride_y + kernel_row - pad_top
                    )
                    for kernel_column in range(kernel_x):
                        input_column = (
                            output_column * stride_x
                            + kernel_column
                            - pad_left
                        )
                        if (
                            input_row < 0
                            or input_row >= src_h
                            or input_column < 0
                            or input_column >= src_w
                        ):
                            window.append(0.0)
                        else:
                            window.append(
                                source[
                                    (input_row * src_w + input_column) * 64
                                    + channel
                                ]
                            )
                if operation == "avg":
                    result = sum(window) / len(window)
                elif operation == "sum":
                    result = sum(window)
                elif operation in {"max", "indexedmax"}:
                    result = max(window)
                elif operation in {"min", "indexedmin"}:
                    result = min(window)
                else:
                    raise RuntimeError(
                        f"{case.name}: unknown pool operation {operation}"
                    )
                expected_values.append(result)
                if operation in {"indexedmax", "indexedmin"}:
                    expected_indices.append(window.index(result))

    expected = _fp(dtype_name, expected_values)
    if expected_indices:
        expected += struct.pack(
            f"<{len(expected_indices)}H", *expected_indices
        )
    return _fp(dtype_name, source), b"", expected


def _pool(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if _POOL_SYMMETRIC_BASE <= case.case_id <= _POOL_TIE_END:
        return _pool_capability(case)
    if case.symbol in ("POOL_F16", "POOL_BF16"):
        source = [
            float(100 * row + 10 * column + channel % 8)
            for row in range(2)
            for column in range(4)
            for channel in range(64)
        ]
        expected = [
            float(100 + 10 * (2 * output_column + 1) + channel % 8)
            for output_column in range(2)
            for channel in range(64)
        ]
        return _fp(case.dtype_name, source), b"", _fp(
            case.dtype_name, expected
        )
    if case.symbol in {
        "POOL_AVG_F16",
        "POOL_SUM_F16",
        "POOL_MIN_F16",
    }:
        source = [
            float(
                4 * (row * 4 + column)
                + 4 * (channel % 4)
                + 32
            )
            for row in range(2)
            for column in range(4)
            for channel in range(64)
        ]
        windows = tuple(
            tuple(
                source[(row * 4 + column) * 64 + channel]
                for row in range(2)
                for column in range(2 * output_column, 2 * output_column + 2)
            )
            for output_column in range(2)
            for channel in range(64)
        )
        operation = {
            "POOL_AVG_F16": lambda values: sum(values) / 4.0,
            "POOL_SUM_F16": sum,
            "POOL_MIN_F16": min,
        }[case.symbol]
        expected = [operation(values) for values in windows]
        return _f16(source), b"", _f16(expected)
    if case.symbol == "POOL_INDEXED_MIN_F16":
        source = [0.0] * (2 * 4 * 64)
        expected_values: list[float] = []
        expected_indices: list[int] = []
        for output_column in range(2):
            for channel in range(64):
                selected = (channel + output_column) % 4
                minimum = float(-64 - output_column * 16 - channel % 8)
                expected_values.append(minimum)
                expected_indices.append(selected)
                for position in range(4):
                    row = position // 2
                    column = 2 * output_column + position % 2
                    source[(row * 4 + column) * 64 + channel] = (
                        minimum if position == selected else float(8 + position)
                    )
        return (
            _f16(source),
            b"",
            _f16(expected_values)
            + struct.pack("<128H", *expected_indices),
        )
    if case.symbol == "POOL_INDEXED_MAX_F16_ASYMMETRIC":
        source = [
            float(100 * row + 10 * column + channel % 8)
            for row in range(3)
            for column in range(5)
            for channel in range(64)
        ]
        expected_values = [
            float(100 * (output_row + 1)
                  + 10 * (2 * output_column + 2)
                  + channel % 8)
            for output_row in range(2)
            for output_column in range(2)
            for channel in range(64)
        ]
        # A 3x2 window is flattened row-major (Y then X); the monotonic
        # pattern makes its lower-right value uniquely maximal at index 5.
        expected_indices = [5] * (2 * 2 * 64)
        return (
            _f16(source),
            b"",
            _f16(expected_values)
            + struct.pack("<256H", *expected_indices),
        )
    else:
        raise RuntimeError(f"{case.name}: unknown pool kind")


def _unpool(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol in {
        "UNPOOL_AVG_F16",
        "UNPOOL_AVG_BF16",
        "UNPOOL_AVG_F32",
    }:
        source = [float(4 * (channel % 16 + 1)) for channel in range(64)]
        expected = [
            source[channel] / 4.0
            for _position in range(4)
            for channel in range(64)
        ]
        return (
            _fp(case.dtype_name, source),
            b"",
            _fp(case.dtype_name, expected),
        )

    if case.symbol in {
        "UNPOOL_F16",
        "UNPOOL_INDEX_F16",
        "UNPOOL_INDEX_BF16_OBSERVED",
        "UNPOOL_INDEX_F32_OBSERVED",
        "UNPOOL_MASK_BF16",
        "UNPOOL_MASK_F32",
    }:
        source = [
            float(
                128 + channel
                if position == channel % 4
                else 1 + position
            )
            for position in range(4)
            for channel in range(64)
        ]
        expected = [
            float(128 + channel if position == channel % 4 else 0)
            for position in range(4)
            for channel in range(64)
        ]
        return (
            _fp(case.dtype_name, source),
            b"",
            _fp(case.dtype_name, expected),
        )

    if case.symbol == "UNPOOL_AVG_F16_ASYMMETRIC_OBSERVED":
        source = [
            float(6 * (1 + position) + channel % 4)
            for position in range(4)
            for channel in range(64)
        ]
        expected = [0.0] * (3 * 5 * 64)
        for source_row in range(2):
            for source_column in range(2):
                for kernel_row in range(2):
                    for kernel_column in range(3):
                        output_row = source_row + kernel_row
                        output_column = (
                            2 * source_column + kernel_column
                        )
                        for channel in range(64):
                            source_index = (
                                (source_row * 2 + source_column) * 64
                                + channel
                            )
                            output_index = (
                                (output_row * 5 + output_column) * 64
                                + channel
                            )
                            expected[output_index] += (
                                source[source_index] / 6.0
                            )
        return _f16(source), b"", _f16(expected)

    asymmetric_symbols = {
        "UNPOOL_INDEX_F16_ASYMMETRIC_OBSERVED",
        "UNPOOL_MASK_F16_ASYMMETRIC",
    }
    repeated_symbols = {
        "UNPOOL_INDEX_F16_REPEATED_OVERLAP_OBSERVED",
        "UNPOOL_MASK_F16_REPEATED_OVERLAP_OBSERVED",
    }
    if case.symbol in asymmetric_symbols | repeated_symbols:
        source = [
            float(100 * row + 10 * column + channel % 8)
            for row in range(3)
            for column in range(5)
            for channel in range(64)
        ]
        expected = [0.0] * (3 * 5 * 64)
        if case.symbol in asymmetric_symbols:
            for output_row in range(2):
                for output_column in range(2):
                    input_row = output_row + 1
                    input_column = 2 * output_column + 2
                    for channel in range(64):
                        index = (
                            (input_row * 5 + input_column) * 64 + channel
                        )
                        expected[index] = source[index]
        else:
            for channel in range(64):
                index = (1 * 5 + 2) * 64 + channel
                source[index] = float(1000 + channel)
        return _f16(source), b"", _f16(expected)

    raise RuntimeError(f"{case.name}: unknown unpool kind")


def _peripheral_arg_extrema(
    case: InstructionCase,
    sample: int,
) -> tuple[bytes, bytes, bytes]:
    if case.symbol == "PERIPHERAL_ARGMAX_F16":
        values = [float((index * 17) % 61 - 30) for index in range(128)]
        result_index = 73
        result_value = 100.0
    elif case.symbol == "PERIPHERAL_ARGMIN_F16":
        # Current hardware is qualified only for this positive finite domain;
        # a negative-domain probe returned element zero instead of the minimum.
        values = [float((index * 17) % 61 + 1) for index in range(128)]
        result_index = 42
        result_value = 0.5
    elif case.symbol == "PERIPHERAL_ARGMIN_NEGATIVE_F16_OBSERVED":
        variant = sample % 3
        values = [
            -float(((index * 17 + variant * 13) % 61) + 1)
            for index in range(128)
        ]
        result_index = (37, 83, 109)[variant]
        result_value = float(-100 - variant)
    else:
        raise RuntimeError(f"{case.name}: unknown peripheral extrema kind")
    values[result_index] = result_value
    expected = (
        _f16((result_value,))
        + _f16((-13.0,))
        + struct.pack("<I", result_index)
    )
    return _f16(values), b"", expected


def _peripheral_lut16() -> tuple[bytes, bytes, bytes]:
    table_indices = tuple((37 * index + 11) % 128 for index in range(128))
    source = struct.pack(
        "<128H", *(2 * table_index for table_index in table_indices)
    )
    table_values = tuple(float(index - 64) for index in range(128))
    expected = _f16(table_values[index] for index in table_indices)
    return source, _f16(table_values), expected


def _peripheral_bilinear() -> tuple[bytes, bytes, bytes]:
    source = _f16(
        float((index * 11) % 97 - 48) for index in range(128)
    )
    # The public wrapper proves the shape/scale ABI, but board output disproves
    # pass-through as a numeric oracle even for equal source/destination shapes.
    # Retain the semantic reference as discriminating input; observation mode
    # deliberately does not compare the result against it.
    return source, b"", source


def _peripheral_factorize_observation() -> tuple[bytes, bytes, bytes]:
    source = _f32(
        (
            float((-1 if index % 2 else 1) * (index + 1))
            + 0.25 * (index % 3)
        )
        for index in range(32)
    )
    return source, b"", bytes(1536)


def _peripheral_lut32_observation() -> tuple[bytes, bytes, bytes]:
    indices = tuple((29 * index + 7) % 128 for index in range(128))
    source = struct.pack("<128I", *(4 * index for index in indices))
    table = struct.pack(
        "<128I",
        *(0x3F000000 ^ (index * 0x00010101) for index in range(128)),
    )
    return source, table, bytes(512)


def _peripheral_randgen_observation() -> tuple[bytes, bytes, bytes]:
    seed0 = struct.pack(
        "<16Q",
        *(0x0123456789ABCDEF ^ (index * 0x1111111111111111)
          for index in range(16)),
    )
    seed1 = struct.pack(
        "<16Q",
        *(0xFEDCBA9876543210 ^ (index * 0x0102040810204080)
          for index in range(16)),
    )
    return seed0, seed1, bytes(1536)


def _peripheral_elemmask_observation() -> tuple[bytes, bytes, bytes]:
    source = _f16(
        float((index % 31) - 15) / 4.0 for index in range(128)
    )
    return source, b"", bytes(256)


def _peripheral(
    case: InstructionCase, sample: int
) -> tuple[bytes, bytes, bytes]:
    if case.symbol in (
        "PERIPHERAL_ARGMAX_F16",
        "PERIPHERAL_ARGMIN_F16",
        "PERIPHERAL_ARGMIN_NEGATIVE_F16_OBSERVED",
    ):
        return _peripheral_arg_extrema(case, sample)
    if case.symbol == "PERIPHERAL_LUT16_F16":
        return _peripheral_lut16()
    if case.symbol == "PERIPHERAL_BILINEAR_F16":
        return _peripheral_bilinear()
    if case.symbol == "PERIPHERAL_FACTORIZE_F32_OBSERVED":
        return _peripheral_factorize_observation()
    if case.symbol == "PERIPHERAL_LUT32_OBSERVED":
        return _peripheral_lut32_observation()
    if case.symbol == "PERIPHERAL_RANDGEN_F16_OBSERVED":
        return _peripheral_randgen_observation()
    if case.symbol == "PERIPHERAL_ELEMMASK_F16_OBSERVED":
        return _peripheral_elemmask_observation()
    raise RuntimeError(f"{case.name}: unknown peripheral kind")


def build_case_payload(case: InstructionCase, sample: int = 0) -> CasePayload:
    if not case.is_safe:
        raise RuntimeError(
            f"{case.name}: deferred case cannot produce an executable request"
        )
    if case.family_name == "CT_ELEMENTWISE":
        input_a, input_b, expected = _arithmetic(case)
    elif case.family_name == "CT_CONVERT":
        input_a, input_b, expected = _convert(case)
    elif case.family_name == "CT_REDUCE":
        input_a, input_b, expected = _reduce(case)
    elif case.family_name == "CT_SELECT_COMPOSITE":
        input_a, input_b, expected = _select(case)
    elif case.family_name == "NE_GEMM":
        input_a, input_b, expected = _gemm(case)
    elif case.family_name == "TDMA_PAD":
        input_a, input_b, expected = _pad()
    elif case.family_name == "TDMA_IMG2COL":
        input_a, input_b, expected = _img2col(case)
    elif case.family_name == "CONV":
        input_a, input_b, expected = _conv(case)
    elif case.family_name == "POOL":
        input_a, input_b, expected = _pool(case)
    elif case.family_name == "UNPOOL":
        input_a, input_b, expected = _unpool(case)
    elif case.family_name == "PERIPHERAL":
        input_a, input_b, expected = _peripheral(case, sample)
    else:
        raise RuntimeError(f"{case.name}: safe case has no oracle builder")
    if len(expected) != case.result_bytes:
        raise RuntimeError(
            f"{case.name}: expected {len(expected)} bytes, catalog declares "
            f"{case.result_bytes}"
        )
    if not any(input_a) or (input_b and not any(input_b)):
        raise RuntimeError(f"{case.name}: discriminating input became all-zero")

    slots = [_slot() for _ in range(4)]
    _put(slots[0], input_a)
    if input_b:
        _put(slots[1], input_b)
    if case.symbol == "GEMM_F16_PSUM":
        _put(
            slots[3],
            _f16(
                [float(1000 + column) for column in range(16)]
                + [-23.0] * 112
            ),
        )

    if case.family_name == "CT_SELECT_COMPOSITE":
        false_values = _repeat((-1.0, -2.0, -3.0, -4.0))
        output_seed = _fp(case.dtype_name, false_values)
    elif case.family_name == "NE_GEMM":
        output_seed = _fp(case.dtype_name, _repeat((-13.0,), 128))
    elif case.family_name == "UNPOOL" and not case.is_observation:
        seed_element_bytes = 4 if case.dtype_name == "F32" else 2
        output_seed = _fp(
            case.dtype_name,
            [0.0] * (case.output_span // seed_element_bytes),
        )
    else:
        seed_element_bytes = 4 if case.dtype_name == "F32" else 2
        seed_elements = case.output_span // seed_element_bytes
        seed_dtype = (
            "BF16"
            if case.dtype_name == "BF16"
            else "F32"
            if case.dtype_name == "F32"
            else "F16"
        )
        output_seed = _fp(
            seed_dtype,
            _repeat((-13.0,), seed_elements),
        )
    _put(slots[2], output_seed)
    expected_slot = bytearray(slots[2])
    if not case.is_observation:
        exact_offsets = reduce_exact_result_byte_offsets(case)
        if len(exact_offsets) != len(expected):
            raise RuntimeError(
                f"{case.name}: exact result offset count disagrees with "
                "logical oracle"
            )
        for offset, value in zip(exact_offsets, expected, strict=True):
            expected_slot[offset] = value

    request_words = [0] * REQUEST_WORDS
    request_words[REQ["MAGIC"]] = REQUEST_MAGIC
    request_words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
    request_words[REQ["CASE"]] = case.case_id
    request_words[REQ["DISPOSITION"]] = case.disposition
    request_words[REQ["FAMILY"]] = case.family
    request_words[REQ["DTYPE"]] = case.dtype
    request_words[REQ["ORACLE"]] = case.oracle
    request_words[REQ["RESULT_BYTES"]] = case.result_bytes
    request_words[REQ["OUTPUT_SPAN"]] = case.output_span
    request_words[REQ["AUX_SPAN"]] = case.aux_span
    request_words[REQ["SAMPLE"]] = sample
    request_words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    request_words[REQ["SLOT_BYTES"]] = SLOT_BYTES
    request_words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    request_words[REQ["GUARD"]] = REQUEST_GUARD
    request = bytearray([0xD3] * RESOURCE_BYTES)
    struct.pack_into(f"<{REQUEST_WORDS}Q", request, 0, *request_words)
    return CasePayload(
        request=bytes(request),
        payload=b"".join(bytes(slot) for slot in slots),
        expected_output_slot=bytes(expected_slot),
    )


def bit2fp_expected_words(case: InstructionCase) -> tuple[int, ...]:
    if case.family_name != "CT_SELECT_COMPOSITE":
        raise RuntimeError(f"{case.name}: case does not execute Bit2FP")
    true_word = BIT2FP_TRUE_WORDS[case.dtype_name]
    return tuple(
        true_word if (index // 8) % 2 == 0 else BIT2FP_FALSE_WORD
        for index in range(128)
    )
