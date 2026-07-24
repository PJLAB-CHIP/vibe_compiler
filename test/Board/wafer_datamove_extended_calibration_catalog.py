#!/usr/bin/env python3
"""Bounded raw/composite DataMove observations missing from the base suite."""

from __future__ import annotations

import dataclasses
import struct

import wafer_physical_tensor_codec as codec


REQUEST_MAGIC = 0x325145584D444357
RECORD_MAGIC = 0x324345584D444357
REQUEST_GUARD = 0xC6B5A4938271605F
RECORD_GUARD = 0x2132435465768798
SCHEMA = 1
REQUEST_WORDS = 24
RECORD_WORDS = 48
RESOURCE_BYTES = 262144
SLOT_BYTES = 131072
BODY_OFFSET = 512
OUTPUT_DDR_OFFSET = SLOT_BYTES
SLOT_CANARY = 0xA7

REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "INPUT_BYTES": 3,
    "RESULT_BYTES": 4,
    "OUTPUT_SPAN": 5,
    "TDMA_INSTRUCTIONS": 6,
    "CT_INSTRUCTIONS": 7,
    "NE_INSTRUCTIONS": 8,
    "ORACLE": 9,
    "SAMPLE": 10,
    "RESOURCE_BYTES": 11,
    "SLOT_BYTES": 12,
    "BODY_OFFSET": 13,
    "GUARD": 23,
}
REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "INPUT_BYTES": 4,
    "RESULT_BYTES": 5,
    "OUTPUT_SPAN": 6,
    "TDMA_INSTRUCTIONS": 7,
    "CT_INSTRUCTIONS": 8,
    "NE_INSTRUCTIONS": 9,
    "ORACLE": 10,
    "SAMPLE": 11,
    "REQUEST_GUARD": 12,
    "OUTPUT_DDR_OFFSET": 13,
    "SLOT_BYTES": 14,
    "BODY_OFFSET": 15,
    "OUTPUT_GUARD_MISMATCHES": 16,
    "TDMA_INST_DELTA": 17,
    "CT_INST_DELTA": 18,
    "NE_INST_DELTA": 19,
    "TDMA_EXEC_DELTA": 20,
    "CT_EXEC_DELTA": 21,
    "NE_EXEC_DELTA": 22,
    "RAW_EXECUTE_RC": 23,
    "RECORD_GUARD": 47,
}

ORACLE_EXACT = 0
ORACLE_OBSERVATION = 1


@dataclasses.dataclass(frozen=True)
class ExtendedDataMoveCase:
    case_id: int
    name: str
    operation: str
    opcode: int
    input_bytes: int
    result_bytes: int
    output_span: int
    expected_tdma_instructions: int
    expected_ct_instructions: int
    expected_ne_instructions: int
    oracle: int
    dtype: str
    semantic_axis: str | None = None

    @property
    def is_exact(self) -> bool:
        return self.oracle == ORACLE_EXACT

    @property
    def disposition(self) -> str:
        return (
            "board-executable"
            if self.is_exact
            else "board-observation"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "disposition": self.disposition,
            "oracle_name": (
                "exact-logical+physical-guard"
                if self.is_exact
                else "raw-output+expected-diff+physical-guard"
            ),
        }


def _span(result_bytes: int) -> int:
    return (result_bytes + 255) // 256 * 256


def _case(
    case_id: int,
    name: str,
    operation: str,
    opcode: int,
    input_bytes: int,
    result_bytes: int,
    *,
    tdma: int = 0,
    ct: int = 0,
    ne: int = 0,
    oracle: int = ORACLE_EXACT,
    dtype: str = "FP16",
    semantic_axis: str | None = None,
    output_span: int | None = None,
) -> ExtendedDataMoveCase:
    return ExtendedDataMoveCase(
        case_id,
        name,
        operation,
        opcode,
        input_bytes,
        result_bytes,
        _span(result_bytes) if output_span is None else output_span,
        tdma,
        ct,
        ne,
        oracle,
        dtype,
        semantic_axis,
    )


LAYOUT_COMPOSITE_SHAPE = (2, 1, 1, 65)
LAYOUT_LOGICAL_BYTES = 2 * 1 * 1 * 65 * 2
LAYOUT_CX_BYTES = codec.physical_layout(
    LAYOUT_COMPOSITE_SHAPE, "Cx", 2
).physical_bytes
LAYOUT_NCX_BYTES = codec.physical_layout(
    LAYOUT_COMPOSITE_SHAPE, "NCx", 2
).physical_bytes

PAD_SOURCE_SHAPE = (2, 5, 7, 65)
PAD_DESTINATION_SHAPE = (2, 7, 10, 65)
PAD_INPUT_BYTES = 2 * 5 * 7 * 65 * 2
PAD_RESULT_BYTES = 2 * 7 * 10 * 65 * 2

IMG2COL_SOURCE_SHAPE = (2, 9, 11, 65)
IMG2COL_DESTINATION_SHAPE = (2, 6, 54, 65)
IMG2COL_INPUT_BYTES = 2 * 9 * 11 * 65 * 2
IMG2COL_RESULT_BYTES = 2 * 6 * 54 * 65 * 2

CONCAT_CASES = (
    _case(
        0,
        "datamove-raw-concat-c-n2h7w9-c33-c32",
        "raw-concat",
        131,
        16380,
        16380,
        ct=1,
        oracle=ORACLE_OBSERVATION,
        semantic_axis="C",
    ),
    _case(
        1,
        "datamove-raw-concat-w-n2h7-w4-w5-c65",
        "raw-concat",
        131,
        16380,
        16380,
        ct=1,
        oracle=ORACLE_OBSERVATION,
        semantic_axis="W",
    ),
    _case(
        2,
        "datamove-raw-concat-h-n2-h3-h4-w9-c65",
        "raw-concat",
        131,
        16380,
        16380,
        ct=1,
        oracle=ORACLE_OBSERVATION,
        semantic_axis="H",
    ),
    _case(
        3,
        "datamove-raw-concat-hw-n2-2x5-3x7-c65",
        "raw-concat",
        131,
        8060,
        8060,
        ct=1,
        oracle=ORACLE_OBSERVATION,
        semantic_axis="HW",
    ),
)

LARGE_TYPED_CASES = (
    _case(
        4,
        "datamove-pad-large-n2h5w7c65-to-n2h7w10c65",
        "pad-large",
        132,
        PAD_INPUT_BYTES,
        PAD_RESULT_BYTES,
        tdma=1,
    ),
    _case(
        5,
        "datamove-img2col-large-n2h9w11c65-k3x2-s2x1",
        "img2col-large",
        138,
        IMG2COL_INPUT_BYTES,
        IMG2COL_RESULT_BYTES,
        tdma=1,
    ),
)

RAW_OBSERVATION_CASES = (
    _case(
        6,
        "datamove-raw-maskgather-f16-tail130",
        "mask-gather",
        136,
        4608,
        260,
        ct=1,
        oracle=ORACLE_OBSERVATION,
        output_span=512,
    ),
    _case(
        7,
        "datamove-raw-maskgather-bv-f16-tail130",
        "mask-gather-bv",
        137,
        4608,
        260,
        ct=1,
        oracle=ORACLE_OBSERVATION,
        output_span=512,
    ),
    _case(
        8,
        "datamove-raw-tensornom-n2h7w9c65",
        "tensor-nom",
        133,
        16380,
        16380,
        tdma=1,
        oracle=ORACLE_OBSERVATION,
    ),
)

COMPOSITE_CASES = (
    _case(
        9,
        "datamove-cx-materialize-ct-add-n2c65",
        "cx-materialize-ct",
        135,
        LAYOUT_CX_BYTES,
        LAYOUT_LOGICAL_BYTES,
        tdma=2,
        ct=1,
        output_span=512,
    ),
    _case(
        10,
        "datamove-ncx-materialize-ct-add-n2c65",
        "ncx-materialize-ct",
        135,
        LAYOUT_NCX_BYTES,
        LAYOUT_LOGICAL_BYTES,
        tdma=4,
        ct=1,
        output_span=512,
    ),
    _case(
        11,
        "datamove-tensor-materialize-ne-identity-m1k16n16",
        "tensor-materialize-ne",
        135,
        4608,
        32,
        tdma=2,
        ne=1,
        output_span=256,
    ),
)

TDMA_DESCRIPTOR_CASES = (
    _case(
        12,
        "tdma-raw-i8-strided-128b-x32",
        "tdma-raw-i8-strided-128",
        123,
        0,
        8064,
        tdma=1,
        dtype="I8",
        output_span=8192,
    ),
    _case(
        13,
        "tdma-raw-i8-strided-64b-x64",
        "tdma-raw-i8-strided-64",
        123,
        0,
        8128,
        tdma=1,
        dtype="I8",
        output_span=8192,
    ),
    _case(
        14,
        "tdma-raw-memset-fp16-4096b",
        "tdma-raw-memset",
        123,
        0,
        4096,
        tdma=1,
        dtype="FP16",
    ),
    _case(
        15,
        "tdma-crt-memset-fp16-4096b",
        "tdma-crt-memset",
        123,
        0,
        4096,
        tdma=1,
        dtype="FP16",
    ),
    _case(
        16,
        "tdma-raw-memset-bf16-4096b",
        "tdma-raw-memset",
        123,
        0,
        4096,
        tdma=1,
        dtype="BF16",
    ),
    _case(
        17,
        "tdma-crt-memset-bf16-4096b",
        "tdma-crt-memset",
        123,
        0,
        4096,
        tdma=1,
        dtype="BF16",
    ),
)

CATALOG = (
    CONCAT_CASES
    + LARGE_TYPED_CASES
    + RAW_OBSERVATION_CASES
    + COMPOSITE_CASES
    + TDMA_DESCRIPTOR_CASES
)
CASES_BY_ID = {case.case_id: case for case in CATALOG}
CASES_BY_NAME = {case.name: case for case in CATALOG}


def _f16_bits(value: int) -> bytes:
    return struct.pack("<e", float(value))


def _logical_values(count: int, seed: int) -> tuple[bytes, ...]:
    return tuple(
        _f16_bits(1 + ((seed * 7 + index * 5 + (index >> 3)) % 23))
        for index in range(count)
    )


def _concat_payload(
    case: ExtendedDataMoveCase, seed: int
) -> tuple[bytes, bytes]:
    specs = {
        "C": ((2, 7, 9, 33), (2, 7, 9, 32), 3),
        "W": ((2, 7, 4, 65), (2, 7, 5, 65), 2),
        "H": ((2, 3, 9, 65), (2, 4, 9, 65), 1),
        "HW": ((2, 2, 5, 65), (2, 3, 7, 65), 1),
    }
    left_shape, right_shape, axis = specs[str(case.semantic_axis)]

    def product(shape: tuple[int, ...]) -> int:
        result = 1
        for dimension in shape:
            result *= dimension
        return result

    left = _logical_values(product(left_shape), seed)
    right = _logical_values(product(right_shape), seed + 41)
    outer = product(left_shape[:axis])
    left_chunk = product(left_shape[axis:])
    right_chunk = product(right_shape[axis:])
    result: list[bytes] = []
    for index in range(outer):
        result.extend(left[index * left_chunk : (index + 1) * left_chunk])
        result.extend(
            right[index * right_chunk : (index + 1) * right_chunk]
        )
    return b"".join(left + right), b"".join(result)


def _pad_payload(seed: int) -> tuple[bytes, bytes]:
    n, h, w, c = PAD_SOURCE_SHAPE
    values = _logical_values(n * h * w * c, seed)
    zero = _f16_bits(0)

    def source(ni: int, hi: int, wi: int, ci: int) -> bytes:
        return values[((ni * h + hi) * w + wi) * c + ci]

    result = (
        source(ni, hi - 1, wi - 2, ci)
        if 1 <= hi < 6 and 2 <= wi < 9
        else zero
        for ni in range(2)
        for hi in range(7)
        for wi in range(10)
        for ci in range(65)
    )
    return b"".join(values), b"".join(result)


def _img2col_payload(seed: int) -> tuple[bytes, bytes]:
    n, h, w, c = IMG2COL_SOURCE_SHAPE
    values = _logical_values(n * h * w * c, seed)
    zero = _f16_bits(0)

    def source(ni: int, hi: int, wi: int, ci: int) -> bytes:
        if hi < 0 or hi >= h or wi < 0 or wi >= w:
            return zero
        return values[((ni * h + hi) * w + wi) * c + ci]

    result = (
        source(ni, oh + ky - 1, ow * 2 + kx - 2, ci)
        for ni in range(2)
        for ky in range(2)
        for kx in range(3)
        for oh in range(9)
        for ow in range(6)
        for ci in range(65)
    )
    return b"".join(values), b"".join(result)


def _mask_payload(case: ExtendedDataMoveCase, seed: int) -> bytes:
    payload = bytearray([SLOT_CANARY] * case.input_bytes)
    values = _logical_values(260, seed)
    payload[: len(values) * 2] = b"".join(values)
    if case.operation == "mask-gather":
        indices = tuple((index * 37 + 11) % 260 for index in range(130))
        payload[4096 : 4096 + 260] = struct.pack("<130H", *indices)
    else:
        bits = bytearray(17)
        for index in range(130):
            if ((index * 29 + 7) % 11) < 5:
                bits[index // 8] |= 1 << (index % 8)
        payload[4096 : 4096 + len(bits)] = bits
    return bytes(payload)


def _layout_composite_payload(
    case: ExtendedDataMoveCase, seed: int
) -> tuple[bytes, bytes]:
    logical = _logical_values(130, seed)
    layout = "Cx" if case.operation == "cx-materialize-ct" else "NCx"
    source = codec.pack_scalar_bytes(
        LAYOUT_COMPOSITE_SHAPE,
        layout,
        2,
        logical,
        padding=SLOT_CANARY,
    )
    expected = b"".join(
        _f16_bits(2 * int(struct.unpack("<e", value)[0]))
        for value in logical
    )
    return source, expected


def _ne_composite_payload(seed: int) -> tuple[bytes, bytes]:
    payload = bytearray([SLOT_CANARY] * 4608)
    lhs = tuple(_f16_bits(1 + ((seed + index * 3) % 15)) for index in range(16))
    payload[:32] = b"".join(lhs)
    identity = bytearray(512)
    one = _f16_bits(1)
    for index in range(16):
        identity[(index * 16 + index) * 2 : (index * 16 + index + 1) * 2] = one
    payload[4096:4608] = identity
    return bytes(payload), b"".join(lhs)


def _tdma_fill_expected(case: ExtendedDataMoveCase) -> bytes:
    if case.operation == "tdma-raw-i8-strided-128":
        result = bytearray([SLOT_CANARY] * case.result_bytes)
        for iteration in range(32):
            begin = iteration * 256
            result[begin : begin + 128] = bytes([0x5A]) * 128
        return bytes(result)
    if case.operation == "tdma-raw-i8-strided-64":
        result = bytearray([SLOT_CANARY] * case.result_bytes)
        for iteration in range(64):
            begin = iteration * 128
            result[begin : begin + 64] = bytes([0xC3]) * 64
        return bytes(result)
    value = 0x3C00 if case.dtype == "FP16" else 0x3F80
    return struct.pack("<H", value) * (case.result_bytes // 2)


def build_input_expected(
    case: ExtendedDataMoveCase, seed: int
) -> tuple[bytes, bytes]:
    if case.operation == "raw-concat":
        return _concat_payload(case, seed)
    if case.operation == "pad-large":
        return _pad_payload(seed)
    if case.operation == "img2col-large":
        return _img2col_payload(seed)
    if case.operation in {"mask-gather", "mask-gather-bv"}:
        return _mask_payload(case, seed), bytes()
    if case.operation == "tensor-nom":
        values = _logical_values(case.input_bytes // 2, seed)
        return b"".join(values), bytes()
    if case.operation in {"cx-materialize-ct", "ncx-materialize-ct"}:
        return _layout_composite_payload(case, seed)
    if case.operation == "tensor-materialize-ne":
        return _ne_composite_payload(seed)
    if case.operation.startswith("tdma-"):
        return bytes(), _tdma_fill_expected(case)
    raise RuntimeError(f"{case.name}: missing host payload builder")


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_output_slot: bytes


def build_case_payload(
    case: ExtendedDataMoveCase, sample: int = 0
) -> CasePayload:
    source, expected = build_input_expected(case, sample + 1)
    if len(source) != case.input_bytes:
        raise RuntimeError(f"{case.name}: generated input size mismatch")
    if case.is_exact and len(expected) != case.result_bytes:
        raise RuntimeError(f"{case.name}: generated result size mismatch")
    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
    words[REQ["CASE"]] = case.case_id
    words[REQ["INPUT_BYTES"]] = case.input_bytes
    words[REQ["RESULT_BYTES"]] = case.result_bytes
    words[REQ["OUTPUT_SPAN"]] = case.output_span
    words[REQ["TDMA_INSTRUCTIONS"]] = case.expected_tdma_instructions
    words[REQ["CT_INSTRUCTIONS"]] = case.expected_ct_instructions
    words[REQ["NE_INSTRUCTIONS"]] = case.expected_ne_instructions
    words[REQ["ORACLE"]] = case.oracle
    words[REQ["SAMPLE"]] = sample
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["SLOT_BYTES"]] = SLOT_BYTES
    words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    payload = bytearray([SLOT_CANARY] * RESOURCE_BYTES)
    payload[BODY_OFFSET : BODY_OFFSET + len(source)] = source
    output = bytearray([SLOT_CANARY] * SLOT_BYTES)
    if case.is_exact:
        output[BODY_OFFSET : BODY_OFFSET + len(expected)] = expected
    return CasePayload(
        request + bytes([SLOT_CANARY]) * (RESOURCE_BYTES - len(request)),
        bytes(payload),
        bytes(output),
    )


EVIDENCE_BY_OPCODE = {
    opcode: tuple(case for case in CATALOG if case.opcode == opcode)
    for opcode in range(121, 139)
}

CALIBRATION_LEAF_BINDINGS = {
    "raw-concat-c-w-h-hw": CONCAT_CASES,
    "large-pad-img2col": LARGE_TYPED_CASES,
    "mask-gather-and-bit-vector": RAW_OBSERVATION_CASES[:2],
    "tensor-nom": RAW_OBSERVATION_CASES[2:],
    "materialize-then-consume": COMPOSITE_CASES,
    "tdma-i8-strided": TDMA_DESCRIPTOR_CASES[:2],
    "tdma-fp16-bf16-raw-crt": TDMA_DESCRIPTOR_CASES[2:],
}
