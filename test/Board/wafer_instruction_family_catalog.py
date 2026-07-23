#!/usr/bin/env python3
"""Typed cases and exact host oracles for the TX81 instruction probe."""

from __future__ import annotations

import dataclasses
import pathlib
import re
import struct
from collections.abc import Callable, Iterable


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
    raise RuntimeError(f"no floating encoder for {dtype_name}")


def _slot() -> bytearray:
    return bytearray([SLOT_CANARY] * SLOT_BYTES)


def _put(slot: bytearray, raw: bytes) -> None:
    end = BODY_OFFSET + len(raw)
    if end > len(slot):
        raise RuntimeError("instruction probe payload exceeds one SPM slot")
    slot[BODY_OFFSET:end] = raw


def _arithmetic(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    base_a = _repeat((-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0))
    base_b = _repeat((1.0, -1.0, 2.0, 2.0, 0.5, -2.0, 4.0, -0.5))
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


def _reduce(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    values = [
        float(width + 1 + channel % 4)
        for width in range(4)
        for channel in range(64)
    ]
    by_channel = [
        [values[width * 64 + channel] for width in range(4)]
        for channel in range(64)
    ]
    operation = case.name.split("-")[1]
    if operation == "sum":
        expected = [sum(group) for group in by_channel]
    elif operation == "avg":
        expected = [sum(group) / 4.0 for group in by_channel]
    elif operation == "max":
        expected = [max(group) for group in by_channel]
    elif operation == "min":
        expected = [min(group) for group in by_channel]
    else:
        raise RuntimeError(f"{case.name}: unknown reduce kind")
    return _fp(case.dtype_name, values), b"", _fp(case.dtype_name, expected)


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
    if case.symbol not in ("GEMM_F16", "GEMM_BF16"):
        raise RuntimeError(f"{case.name}: unknown GEMM kind")
    lhs = _repeat((1.0, 2.0, 3.0, 4.0), 16)
    lhs_physical = lhs + [0.0] * (128 - len(lhs))
    rhs = [1.0 if row == column else 0.0 for row in range(16) for column in range(16)]
    return (
        _fp(case.dtype_name, lhs_physical),
        _fp(case.dtype_name, rhs),
        _fp(case.dtype_name, lhs),
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
    if case.symbol != "CONV_F16":
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
    return _f16(source), _f16(weight), _f16(expected)


def _pool(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol not in ("POOL_F16", "POOL_BF16"):
        raise RuntimeError(f"{case.name}: unknown pool kind")
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
    return _fp(case.dtype_name, source), b"", _fp(case.dtype_name, expected)


def _peripheral_arg_extrema(
    case: InstructionCase,
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


def _peripheral(case: InstructionCase) -> tuple[bytes, bytes, bytes]:
    if case.symbol in (
        "PERIPHERAL_ARGMAX_F16",
        "PERIPHERAL_ARGMIN_F16",
    ):
        return _peripheral_arg_extrema(case)
    if case.symbol == "PERIPHERAL_LUT16_F16":
        return _peripheral_lut16()
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
    elif case.family_name == "PERIPHERAL":
        input_a, input_b, expected = _peripheral(case)
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

    if case.family_name == "CT_SELECT_COMPOSITE":
        false_values = _repeat((-1.0, -2.0, -3.0, -4.0))
        output_seed = _fp(case.dtype_name, false_values)
    elif case.family_name == "NE_GEMM":
        output_seed = _fp(case.dtype_name, _repeat((-13.0,), 128))
    else:
        seed_elements = case.output_span // 2
        output_seed = _fp(
            "BF16" if case.dtype_name == "BF16" else "F16",
            _repeat((-13.0,), seed_elements),
        )
    _put(slots[2], output_seed)
    expected_slot = bytearray(slots[2])
    expected_slot[BODY_OFFSET : BODY_OFFSET + len(expected)] = expected

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
