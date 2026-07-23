#!/usr/bin/env python3
"""Large/tail/batch/orientation NE calibration rows and exact host oracles."""

from __future__ import annotations

import dataclasses
import struct

import wafer_physical_tensor_codec as physical


REQUEST_MAGIC = 0x31514552454E4357
RECORD_MAGIC = 0x31434552454E4357
REQUEST_GUARD = 0xB7A6958473625140
RECORD_GUARD = 0x0F1E2D3C4B5A6978
SCHEMA = 1
REQUEST_WORDS = 24
RECORD_WORDS = 32
RESOURCE_BYTES = 262144
SLOT_BYTES = 65536
BODY_OFFSET = 256
OUTPUT_DDR_OFFSET = SLOT_BYTES
SLOT_CANARY = 0xA7
CASE_BASE = 20000

DTYPES = {"F16": 0, "BF16": 1}
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
    layout: str
    lhs_orientation: int
    rhs_orientation: int
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    lhs_span: int
    rhs_span: int
    output_span: int

    @property
    def dtype(self) -> int:
        return DTYPES[self.dtype_name]

    @property
    def is_safe(self) -> bool:
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "dtype": self.dtype_name.lower(),
            "geometry": self.geometry_name,
            "orientation": self.orientation_name,
            "batch": self.batch,
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "layout": self.layout.lower(),
            "lhs_shape": self.lhs_shape,
            "rhs_shape": self.rhs_shape,
            "output_shape": self.output_shape,
            "lhs_span": self.lhs_span,
            "rhs_span": self.rhs_span,
            "output_span": self.output_span,
            "oracle": "exact-logical-bits+physical-guard",
        }


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
        tuple(DTYPES).index(dtype_name) * 12
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
        layout,
        lhs_orientation,
        rhs_orientation,
        lhs_shape,
        rhs_shape,
        output_shape,
        lhs_span,
        rhs_span,
        output_span,
    )


CATALOG = tuple(
    _make_case(dtype, geometry, orientation)
    for dtype in DTYPES
    for geometry in GEOMETRIES
    for orientation in ORIENTATIONS
)
CASES_BY_NAME = {case.name: case for case in CATALOG}
SAFE_CASES = CATALOG


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_logical: tuple[bytes, ...]
    expected_physical: bytes


def _encode(dtype_name: str, value: float) -> bytes:
    if dtype_name == "F16":
        return struct.pack("<e", value)
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", ((bits + rounding_bias) >> 16) & 0xFFFF)


def _logical_lhs(case: NECase) -> tuple[float, ...]:
    return tuple(
        float(((batch * 11 + row * 5 + inner * 3) % 9) - 4)
        for batch in range(case.batch)
        for row in range(case.m)
        for inner in range(case.k)
    )


def _logical_rhs(case: NECase) -> tuple[float, ...]:
    return tuple(
        1.0 if inner == column else 0.0
        for _batch in range(case.batch)
        for inner in range(case.k)
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


def build_case_payload(case: NECase, sample: int = 0) -> CasePayload:
    lhs_logical = _logical_lhs(case)
    rhs_logical = _logical_rhs(case)
    lhs_stored = _stored_values(
        lhs_logical,
        batch=case.batch,
        rows=case.m,
        columns=case.k,
        transpose=case.lhs_orientation,
    )
    rhs_stored = _stored_values(
        rhs_logical,
        batch=case.batch,
        rows=case.k,
        columns=case.n,
        transpose=case.rhs_orientation,
    )
    lhs = physical.pack_scalar_bytes(
        case.lhs_shape,
        case.layout,
        2,
        (_encode(case.dtype_name, value) for value in lhs_stored),
    )
    rhs = physical.pack_scalar_bytes(
        case.rhs_shape,
        case.layout,
        2,
        (_encode(case.dtype_name, value) for value in rhs_stored),
    )
    expected_values = tuple(
        lhs_logical[
            (batch * case.m + row) * case.k + column
        ]
        if column < case.k
        else 0.0
        for batch in range(case.batch)
        for row in range(case.m)
        for column in range(case.n)
    )
    expected_logical = tuple(
        _encode(case.dtype_name, value) for value in expected_values
    )
    expected_physical = physical.pack_scalar_bytes(
        case.output_shape,
        case.layout,
        2,
        expected_logical,
        padding=SLOT_CANARY,
    )

    slots = [bytearray([SLOT_CANARY] * SLOT_BYTES) for _ in range(4)]
    for slot, data in ((slots[0], lhs), (slots[1], rhs)):
        slot[BODY_OFFSET : BODY_OFFSET + len(data)] = data
    output_zero = physical.pack_scalar_bytes(
        case.output_shape,
        case.layout,
        2,
        (_encode(case.dtype_name, 0.0) for _ in expected_logical),
        padding=SLOT_CANARY,
    )
    slots[2][BODY_OFFSET : BODY_OFFSET + len(output_zero)] = output_zero

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
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    return CasePayload(
        request + bytes([SLOT_CANARY]) * (RESOURCE_BYTES - len(request)),
        b"".join(bytes(slot) for slot in slots),
        expected_logical,
        expected_physical,
    )
