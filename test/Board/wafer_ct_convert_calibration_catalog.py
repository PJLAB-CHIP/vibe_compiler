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
SCHEMA = 2
REQUEST_WORDS = 24
RECORD_WORDS = 32
RESOURCE_BYTES = 131072
SLOT_BYTES = 65536
BODY_OFFSET = 256
OUTPUT_DDR_OFFSET = SLOT_BYTES
SLOT_CANARY = 0xA7
MAIN_ELEMENTS = 8192
TAIL_ELEMENTS = 8197
RND_NEAREST_EVEN = 0
RND_ZERO = 1
RND_POS_INF = 2
RND_NEG_INF = 3
RND_STOCHASTIC = 4
ROUNDING_NAMES = {
    RND_NEAREST_EVEN: "nearest-even",
    RND_ZERO: "toward-zero",
    RND_POS_INF: "toward-positive",
    RND_NEG_INF: "toward-negative",
    RND_STOCHASTIC: "stochastic",
}
DOMAINS = {
    "NORMAL": 0,
    "DIRECTED": 1,
    "ZERO_POINT": 2,
    "EXTREMA": 3,
    "STOCHASTIC": 4,
}
DISPOSITIONS = {
    "BOARD_EXACT": 0,
    "BOARD_OBSERVED": 1,
    "ISOLATED_DEFERRED": 2,
}
DIRECTED_CASE_BASE = 1000
ZERO_POINT_CASE_BASE = 2000
EXTREMA_CASE_BASE = 3000
STOCHASTIC_CASE_BASE = 4000

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
    "DOMAIN": 15,
    "ZERO_POINT": 16,
    "DISPOSITION": 17,
    "GUARD": 23,
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
    # Wire-compatibility slot; host validation uses the complete output slot.
    "OUTPUT_GUARD_MISMATCHES": 17,
    "CT_INST_DELTA": 18,
    "CT_EXEC_DELTA": 19,
    "CT_BLOCKING_DELTA": 20,
    "DOMAIN": 21,
    "ZERO_POINT": 22,
    "DISPOSITION": 23,
    "RECORD_GUARD": 31,
}

ZERO_POINT_OPCODES = frozenset(range(139, 143))
PLAIN_OPCODES = frozenset(
    (143, 151, 154, 155, 156, 161, 162, 172, 174)
)
ROUNDING_OPCODES = frozenset(
    set(range(139, 175)) - ZERO_POINT_OPCODES - PLAIN_OPCODES
)
# Int16 values are all exactly representable in FP32.  The three directed
# modes are retained as an explicit mode-invariance control, not mislabeled as
# evidence that the rounding selector changed a result.
ROUNDING_INVARIANT_OPCODES = frozenset({145})


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
    zero_point: int = 0
    domain_name: str = "NORMAL"
    disposition_name: str = "BOARD_EXACT"
    reason: str = ""

    @property
    def domain(self) -> int:
        return DOMAINS[self.domain_name]

    @property
    def disposition(self) -> int:
        return DISPOSITIONS[self.disposition_name]

    @property
    def is_safe(self) -> bool:
        return self.disposition_name != "ISOLATED_DEFERRED"

    @property
    def exact(self) -> bool:
        return self.disposition_name == "BOARD_EXACT"

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
            "rounding": ROUNDING_NAMES[self.rounding_mode],
            "zero_point": self.zero_point,
            "numeric_domain": self.domain_name.lower().replace("_", "-"),
            "input_bytes": self.input_bytes,
            "result_bytes": self.result_bytes,
            "output_span": self.output_span,
            "disposition": self.disposition_name.lower().replace("_", "-"),
            "oracle": (
                "full-result-bits+physical-padding+guard"
                if self.exact
                else "raw-result-bits+physical-padding+guard"
                if self.is_safe
                else "not-issued"
            ),
            "reason": self.reason,
        }


def _make_case(
    *,
    case_id: int,
    opcode: int,
    shape_name: str,
    elements: int,
    suffix: str,
    rounding_mode: int = RND_NEAREST_EVEN,
    zero_point: int = 0,
    domain_name: str = "NORMAL",
    disposition_name: str = "BOARD_EXACT",
    reason: str = "",
) -> CTConvertCase:
    source, destination = _parse_route(opcode)
    result_bytes = elements * TYPE_BYTES[destination]
    return CTConvertCase(
        case_id=case_id,
        name=(
            f"ct-op{opcode:03d}-convert-{source}-{destination}-{suffix}"
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
        rounding_mode=rounding_mode,
        zero_point=zero_point,
        domain_name=domain_name,
        disposition_name=disposition_name,
        reason=reason,
    )


def _extrema_disposition(source: str, destination: str) -> str:
    if destination.startswith("int"):
        return "BOARD_OBSERVED"
    if destination == "fp16" and source in {
        "int32",
        "bf16",
        "fp32",
        "tf32",
    }:
        return "BOARD_OBSERVED"
    if destination == "bf16" and source in {"fp32", "tf32"}:
        return "BOARD_OBSERVED"
    if destination == "tf32" and source == "fp32":
        return "BOARD_OBSERVED"
    return "BOARD_EXACT"


def build_catalog() -> tuple[CTConvertCase, ...]:
    cases: list[CTConvertCase] = []
    for opcode in range(139, 175):
        for shape_index, (shape_name, elements) in enumerate(
            (("main", MAIN_ELEMENTS), ("tail", TAIL_ELEMENTS))
        ):
            cases.append(
                _make_case(
                    case_id=(opcode - 139) * 2 + shape_index,
                    opcode=opcode,
                    shape_name=shape_name,
                    elements=elements,
                    suffix=shape_name,
                )
            )
    for opcode in sorted(ROUNDING_OPCODES):
        route = opcode - 139
        for rounding_mode in (RND_ZERO, RND_POS_INF, RND_NEG_INF):
            cases.append(
                _make_case(
                    case_id=(
                        DIRECTED_CASE_BASE
                        + route * 3
                        + rounding_mode
                        - 1
                    ),
                    opcode=opcode,
                    shape_name="main",
                    elements=MAIN_ELEMENTS,
                    suffix=ROUNDING_NAMES[rounding_mode],
                    rounding_mode=rounding_mode,
                    domain_name="DIRECTED",
                )
            )
    for opcode in sorted(ZERO_POINT_OPCODES):
        cases.append(
            _make_case(
                case_id=ZERO_POINT_CASE_BASE + opcode - 139,
                opcode=opcode,
                shape_name="main",
                elements=MAIN_ELEMENTS,
                suffix="zero-point-7",
                zero_point=7,
                domain_name="ZERO_POINT",
                disposition_name="BOARD_OBSERVED",
                reason=(
                    "zero-point formula is intentionally calibrated from raw "
                    "board output and is not yet a production numeric policy"
                ),
            )
        )
    for opcode in range(139, 175):
        source, destination = _parse_route(opcode)
        cases.append(
            _make_case(
                case_id=EXTREMA_CASE_BASE + opcode - 139,
                opcode=opcode,
                shape_name="main",
                elements=MAIN_ELEMENTS,
                suffix="extrema",
                domain_name="EXTREMA",
                disposition_name=_extrema_disposition(
                    source, destination
                ),
                reason=(
                    "overflow/saturation policy is observed rather than "
                    "assumed"
                    if _extrema_disposition(source, destination)
                    == "BOARD_OBSERVED"
                    else ""
                ),
            )
        )
    for opcode in sorted(ROUNDING_OPCODES):
        cases.append(
            _make_case(
                case_id=STOCHASTIC_CASE_BASE + opcode - 139,
                opcode=opcode,
                shape_name="main",
                elements=MAIN_ELEMENTS,
                suffix="stochastic-observed",
                rounding_mode=RND_STOCHASTIC,
                domain_name="STOCHASTIC",
                disposition_name="BOARD_OBSERVED",
                reason=(
                    "the ABI exposes stochastic mode without seed/state; "
                    "repeat the bounded request and retain raw output without "
                    "assuming an exact sequence or distribution"
                ),
            )
        )
    return tuple(cases)


CATALOG = build_catalog()
CASES_BY_NAME = {case.name: case for case in CATALOG}
CASES_BY_ID = {case.case_id: case for case in CATALOG}
SAFE_CASES = tuple(case for case in CATALOG if case.is_safe)
DEFERRED_CASES = tuple(case for case in CATALOG if not case.is_safe)
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "ct-convert-all-routes-main-tail-nearest-even": tuple(
        case for case in CATALOG if case.domain_name == "NORMAL"
    ),
    "ct-convert-halfway-extrema-positive": tuple(
        case
        for case in CATALOG
        if case.domain_name == "EXTREMA"
        and case.disposition_name == "BOARD_EXACT"
    ),
    "ct-convert-halfway-extrema-observed": tuple(
        case
        for case in CATALOG
        if case.domain_name == "EXTREMA"
        and case.disposition_name == "BOARD_OBSERVED"
    ),
    "ct-convert-directed-rounding-positive": tuple(
        case
        for case in CATALOG
        if case.domain_name == "DIRECTED"
    ),
    "ct-convert-zero-point-observed": tuple(
        case for case in CATALOG if case.domain_name == "ZERO_POINT"
    ),
    "ct-convert-stochastic-observed": tuple(
        case for case in CATALOG if case.domain_name == "STOCHASTIC"
    ),
}


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


def _directed_source_values(
    source_type: str, destination_type: str
) -> tuple[int | float, ...]:
    """Return source-representable values that distinguish directed modes."""
    route_values: dict[tuple[str, str], tuple[int | float, ...]] = {
        ("int32", "fp32"): (
            -16777219,
            -16777217,
            16777217,
            16777219,
        ),
        ("fp16", "bf16"): (
            -1.01171875,
            -1.00390625,
            1.00390625,
            1.01171875,
        ),
        ("fp32", "fp16"): (
            -1.00146484375,
            -1.00048828125,
            1.00048828125,
            1.00146484375,
        ),
        ("fp32", "bf16"): (
            -1.01171875,
            -1.00390625,
            1.00390625,
            1.01171875,
        ),
        ("fp32", "tf32"): (
            -1.00146484375,
            -1.00048828125,
            1.00048828125,
            1.00146484375,
        ),
        ("tf32", "bf16"): (
            -1.01171875,
            -1.00390625,
            1.00390625,
            1.01171875,
        ),
    }
    return route_values.get(
        (source_type, destination_type),
        _source_values(source_type, destination_type),
    )


def _extrema_source_values(type_name: str) -> tuple[int | float, ...]:
    if type_name == "int8":
        return (-128, -127, -1, 0, 1, 126, 127)
    if type_name == "int16":
        return (-32768, -32767, -129, -128, 127, 128, 32766, 32767)
    if type_name == "int32":
        return (
            -2147483648,
            -2147483647,
            -65537,
            -32769,
            32768,
            65536,
            2147483646,
            2147483647,
        )
    if type_name == "fp16":
        return (
            -65504.0,
            -32769.0,
            -129.0,
            -0.0,
            0.0,
            127.0,
            32768.0,
            65504.0,
        )
    return (
        -3.0e38,
        -65537.0,
        -32769.0,
        -129.0,
        -0.0,
        0.0,
        127.0,
        32768.0,
        65536.0,
        3.0e38,
        -math.inf,
        math.inf,
    )


def _step_float_bits(type_name: str, raw: bytes, upward: bool) -> bytes:
    code = "H" if type_name in ("fp16", "bf16") else "I"
    bits = struct.unpack(f"<{code}", raw)[0]
    sign = 1 << (15 if code == "H" else 31)
    step = 0x2000 if type_name == "tf32" else 1
    if upward:
        if bits == sign:
            bits = step
        elif bits & sign:
            bits -= step
        else:
            bits += step
    else:
        if bits == 0:
            bits = sign | step
        elif bits & sign:
            bits += step
        else:
            bits -= step
    return struct.pack(f"<{code}", bits)


def _encode_float_with_rounding(
    destination_type: str, value: float, rounding_mode: int
) -> bytes:
    nearest = _encode_scalar(destination_type, value)
    if (
        rounding_mode == RND_NEAREST_EVEN
        or not math.isfinite(value)
    ):
        return nearest
    rounded = float(_decode_scalar(destination_type, nearest))
    if rounded == value:
        return nearest
    if rounding_mode == RND_ZERO:
        needs_step = abs(rounded) > abs(value)
        upward = value < 0.0
    elif rounding_mode == RND_POS_INF:
        needs_step = rounded < value
        upward = True
    elif rounding_mode == RND_NEG_INF:
        needs_step = rounded > value
        upward = False
    else:
        raise RuntimeError("stochastic rounding has no exact host oracle")
    return (
        _step_float_bits(destination_type, nearest, upward)
        if needs_step
        else nearest
    )


def _convert_scalar(
    source_type: str,
    destination_type: str,
    raw: bytes,
    rounding_mode: int,
) -> bytes:
    value = _decode_scalar(source_type, raw)
    if destination_type.startswith("int"):
        if not math.isfinite(float(value)):
            raise RuntimeError("non-finite value cannot enter exact integer oracle")
        if rounding_mode == RND_NEAREST_EVEN:
            value = round(float(value))
        elif rounding_mode == RND_ZERO:
            value = math.trunc(float(value))
        elif rounding_mode == RND_POS_INF:
            value = math.ceil(float(value))
        elif rounding_mode == RND_NEG_INF:
            value = math.floor(float(value))
        else:
            raise RuntimeError("stochastic rounding has no exact host oracle")
        return _encode_scalar(destination_type, value)
    return _encode_float_with_rounding(
        destination_type, float(value), rounding_mode
    )


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_output_slot: bytes | None


def build_case_payload(case: CTConvertCase, sample: int = 0) -> CasePayload:
    if not case.is_safe:
        raise RuntimeError(
            f"{case.name}: deferred case cannot produce a board request: "
            f"{case.reason}"
        )
    pattern = (
        _extrema_source_values(case.source_type)
        if case.domain_name == "EXTREMA"
        else _directed_source_values(
            case.source_type, case.destination_type
        )
        if case.domain_name in ("DIRECTED", "STOCHASTIC")
        else _source_values(case.source_type, case.destination_type)
    )
    source_scalars = tuple(
        _encode_scalar(case.source_type, pattern[index % len(pattern)])
        for index in range(case.elements)
    )
    input_raw = b"".join(source_scalars)
    expected_raw: bytes | None = None
    if case.exact:
        expected_raw = b"".join(
            _convert_scalar(
                case.source_type,
                case.destination_type,
                raw,
                case.rounding_mode,
            )
            for raw in source_scalars
        )
    if len(input_raw) != case.input_bytes or (
        expected_raw is not None and len(expected_raw) != case.result_bytes
    ):
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
    words[REQ["DOMAIN"]] = case.domain
    words[REQ["ZERO_POINT"]] = case.zero_point
    words[REQ["DISPOSITION"]] = case.disposition
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)

    payload = bytearray([SLOT_CANARY] * RESOURCE_BYTES)
    payload[BODY_OFFSET : BODY_OFFSET + len(input_raw)] = input_raw
    expected: bytearray | None = None
    if expected_raw is not None:
        expected = bytearray([SLOT_CANARY] * SLOT_BYTES)
        expected[BODY_OFFSET : BODY_OFFSET + len(expected_raw)] = expected_raw
    return CasePayload(
        request + bytes([SLOT_CANARY]) * (RESOURCE_BYTES - len(request)),
        bytes(payload),
        bytes(expected) if expected is not None else None,
    )
