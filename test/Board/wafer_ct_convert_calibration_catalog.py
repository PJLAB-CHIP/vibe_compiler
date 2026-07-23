#!/usr/bin/env python3
"""Exact CT convert opcode inventory, large-shape inputs and host oracles."""

from __future__ import annotations

import dataclasses
import math
import struct

import wafer_ct_vector_calibration_catalog as ct_inventory


REQUEST_MAGIC = 0x3151455256434357
RECORD_MAGIC = 0x3143455256434357
REQUEST_GUARD = 0xE7D6C5B4A3928170
RECORD_GUARD = 0x0F1E2D3C4B5A6978
SCHEMA = 1
REQUEST_WORDS = 16
RECORD_WORDS = 32
RESOURCE_BYTES = 131072
SLOT_BYTES = 65536
BODY_OFFSET = 256
OUTPUT_DDR_OFFSET = SLOT_BYTES
SLOT_CANARY = 0xA7
MAIN_ELEMENTS = 8192
TAIL_ELEMENTS = 8197
RND_NEAREST_EVEN = 0

TYPE_CODES = {
    "int8": 0,
    "int16": 1,
    "int32": 2,
    "fp16": 3,
    "bf16": 4,
    "fp32": 5,
    "tf32": 6,
}
TYPE_BYTES = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "tf32": 4,
}
REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "OPCODE": 3,
    "SRC_TYPE": 4,
    "DST_TYPE": 5,
    "ELEMENTS": 6,
    "INPUT_BYTES": 7,
    "RESULT_BYTES": 8,
    "OUTPUT_SPAN": 9,
    "ROUNDING": 10,
    "SAMPLE": 11,
    "RESOURCE_BYTES": 12,
    "SLOT_BYTES": 13,
    "BODY_OFFSET": 14,
    "GUARD": 15,
}
REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "OPCODE": 4,
    "SRC_TYPE": 5,
    "DST_TYPE": 6,
    "ELEMENTS": 7,
    "INPUT_BYTES": 8,
    "RESULT_BYTES": 9,
    "OUTPUT_SPAN": 10,
    "ROUNDING": 11,
    "SAMPLE": 12,
    "REQUEST_GUARD": 13,
    "OUTPUT_DDR_OFFSET": 14,
    "SLOT_BYTES": 15,
    "BODY_OFFSET": 16,
    "OUTPUT_GUARD_MISMATCHES": 17,
    "CT_INST_DELTA": 18,
    "CT_EXEC_DELTA": 19,
    "CT_BLOCKING_DELTA": 20,
    "RECORD_GUARD": 31,
}


def _parse_route(opcode: int) -> tuple[str, str]:
    name = ct_inventory.OPCODE_NAMES[opcode]
    marker = "ConvertOp_V_V_"
    if marker not in name:
        raise RuntimeError(f"opcode {opcode} is not a convert opcode: {name}")
    source, destination = name.split(marker, 1)[1].split("_", 1)
    return source, destination


@dataclasses.dataclass(frozen=True)
class CTConvertCase:
    case_id: int
    name: str
    opcode: int
    opcode_name: str
    source_type: str
    destination_type: str
    shape_name: str
    elements: int
    input_bytes: int
    result_bytes: int
    output_span: int
    rounding_mode: int = RND_NEAREST_EVEN

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "opcode": self.opcode,
            "opcode_name": self.opcode_name,
            "family": "convert",
            "form": "V",
            "source_type": self.source_type,
            "destination_type": self.destination_type,
            "shape": self.shape_name,
            "elements": self.elements,
            "rounding": "nearest-even",
            "input_bytes": self.input_bytes,
            "result_bytes": self.result_bytes,
            "output_span": self.output_span,
            "disposition": "board-exact",
            "oracle": "full-result-bits+physical-padding+guard",
        }


def build_catalog() -> tuple[CTConvertCase, ...]:
    cases: list[CTConvertCase] = []
    for opcode in range(139, 175):
        source, destination = _parse_route(opcode)
        for shape_index, (shape_name, elements) in enumerate(
            (("main", MAIN_ELEMENTS), ("tail", TAIL_ELEMENTS))
        ):
            result_bytes = elements * TYPE_BYTES[destination]
            cases.append(
                CTConvertCase(
                    case_id=(opcode - 139) * 2 + shape_index,
                    name=(
                        f"ct-op{opcode:03d}-convert-{source}-"
                        f"{destination}-{shape_name}"
                    ),
                    opcode=opcode,
                    opcode_name=ct_inventory.OPCODE_NAMES[opcode],
                    source_type=source,
                    destination_type=destination,
                    shape_name=shape_name,
                    elements=elements,
                    input_bytes=elements * TYPE_BYTES[source],
                    result_bytes=result_bytes,
                    output_span=(result_bytes + 255) // 256 * 256,
                )
            )
    return tuple(cases)


CATALOG = build_catalog()
CASES_BY_NAME = {case.name: case for case in CATALOG}
CASES_BY_ID = {case.case_id: case for case in CATALOG}


def _round_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    exponent = bits & 0x7F800000
    fraction = bits & 0x007FFFFF
    if exponent == 0x7F800000 and fraction:
        return ((bits >> 16) | 0x0040) & 0xFFFF
    bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + bias) >> 16) & 0xFFFF


def _round_to_tf32_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    exponent = bits & 0x7F800000
    fraction = bits & 0x007FFFFF
    if exponent == 0x7F800000:
        return bits if fraction == 0 else bits | 0x00400000
    bias = 0xFFF + ((bits >> 13) & 1)
    return (bits + bias) & 0xFFFFE000


def _encode_scalar(type_name: str, value: int | float) -> bytes:
    if type_name == "int8":
        return struct.pack("<b", int(value))
    if type_name == "int16":
        return struct.pack("<h", int(value))
    if type_name == "int32":
        return struct.pack("<i", int(value))
    if type_name == "fp16":
        return struct.pack("<e", float(value))
    if type_name == "bf16":
        return struct.pack("<H", _round_to_bf16_bits(float(value)))
    if type_name == "fp32":
        return struct.pack("<f", float(value))
    if type_name == "tf32":
        return struct.pack("<I", _round_to_tf32_bits(float(value)))
    raise RuntimeError(f"unknown convert scalar type {type_name}")


def _decode_scalar(type_name: str, raw: bytes) -> int | float:
    if type_name == "int8":
        return struct.unpack("<b", raw)[0]
    if type_name == "int16":
        return struct.unpack("<h", raw)[0]
    if type_name == "int32":
        return struct.unpack("<i", raw)[0]
    if type_name == "fp16":
        return struct.unpack("<e", raw)[0]
    if type_name == "bf16":
        bits = struct.unpack("<H", raw)[0] << 16
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    if type_name == "fp32":
        return struct.unpack("<f", raw)[0]
    if type_name == "tf32":
        bits = struct.unpack("<I", raw)[0]
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    raise RuntimeError(f"unknown convert scalar type {type_name}")


def _source_values(type_name: str, destination_type: str) -> tuple[int | float, ...]:
    if type_name.startswith("int"):
        common: tuple[int | float, ...] = (
            -17,
            -9,
            -3,
            -1,
            0,
            1,
            2,
            7,
            15,
            17,
        )
        if type_name == "int16":
            common += (255, 257, 259, 2049, 2051, -2049, -2051)
        if type_name == "int32":
            common += (4097, 8193, 16385, -4097, -8193)
        return common

    finite: tuple[int | float, ...] = (
        -17.5,
        -9.5,
        -3.5,
        -2.5,
        -1.5,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        1.5,
        2.5,
        3.5,
        7.5,
        15.5,
        31.5,
    )
    if destination_type.startswith("int"):
        return finite
    return finite + (math.inf, -math.inf)


def _convert_scalar(
    source_type: str, destination_type: str, raw: bytes
) -> bytes:
    value = _decode_scalar(source_type, raw)
    if destination_type.startswith("int"):
        if not math.isfinite(float(value)):
            raise RuntimeError("non-finite value cannot enter exact integer oracle")
        value = round(float(value))
    return _encode_scalar(destination_type, value)


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_output_slot: bytes


def build_case_payload(case: CTConvertCase, sample: int = 0) -> CasePayload:
    pattern = _source_values(case.source_type, case.destination_type)
    source_scalars = tuple(
        _encode_scalar(case.source_type, pattern[index % len(pattern)])
        for index in range(case.elements)
    )
    expected_scalars = tuple(
        _convert_scalar(case.source_type, case.destination_type, raw)
        for raw in source_scalars
    )
    input_raw = b"".join(source_scalars)
    expected_raw = b"".join(expected_scalars)
    if len(input_raw) != case.input_bytes or len(expected_raw) != case.result_bytes:
        raise RuntimeError(f"{case.name}: scalar codec size mismatch")

    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
    words[REQ["CASE"]] = case.case_id
    words[REQ["OPCODE"]] = case.opcode
    words[REQ["SRC_TYPE"]] = TYPE_CODES[case.source_type]
    words[REQ["DST_TYPE"]] = TYPE_CODES[case.destination_type]
    words[REQ["ELEMENTS"]] = case.elements
    words[REQ["INPUT_BYTES"]] = case.input_bytes
    words[REQ["RESULT_BYTES"]] = case.result_bytes
    words[REQ["OUTPUT_SPAN"]] = case.output_span
    words[REQ["ROUNDING"]] = case.rounding_mode
    words[REQ["SAMPLE"]] = sample
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["SLOT_BYTES"]] = SLOT_BYTES
    words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)

    payload = bytearray([SLOT_CANARY] * RESOURCE_BYTES)
    payload[BODY_OFFSET : BODY_OFFSET + len(input_raw)] = input_raw
    expected = bytearray([SLOT_CANARY] * SLOT_BYTES)
    expected[BODY_OFFSET : BODY_OFFSET + len(expected_raw)] = expected_raw
    return CasePayload(
        request + bytes([SLOT_CANARY]) * (RESOURCE_BYTES - len(request)),
        bytes(payload),
        bytes(expected),
    )
