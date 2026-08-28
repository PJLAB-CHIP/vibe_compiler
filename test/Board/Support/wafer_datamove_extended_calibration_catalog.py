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
REQUEST_WORDS = 24
RECORD_WORDS = 48
RESOURCE_BYTES = 262144
SLOT_BYTES = 131072
BODY_OFFSET = 512
OUTPUT_DDR_OFFSET = SLOT_BYTES
SLOT_CANARY = 0xA7
FP16_BYTES = 2

REQ = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
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
    "WORD_COUNT": 1,
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


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _compact_bytes(shape: tuple[int, ...]) -> int:
    return _product(shape) * FP16_BYTES


def _raw_concat_operand_span(shape: tuple[int, ...]) -> int:
    return codec.physical_layout(shape, "NCx", FP16_BYTES).physical_bytes


RAW_CONCAT_SPECS = {
    "C": (
        (2, 7, 9, 33),
        (2, 7, 9, 32),
        (2, 7, 9, 65),
        3,
    ),
    "W": (
        (2, 7, 4, 65),
        (2, 7, 5, 65),
        (2, 7, 9, 65),
        2,
    ),
    "H": (
        (2, 3, 9, 65),
        (2, 4, 9, 65),
        (2, 7, 9, 65),
        1,
    ),
}
# The production op_concat wrapper asserts axis == rank - 1 and maps
# that logical last dimension to native dims=0 (C).  This is a conservative
# production contract, not proof that hardware rejects every other encoding:
# W/H have bounded completion+guard observations.  Native dims=HW is an invalid
# instruction use and must never become an executable catalog case.
NATIVE_CONCAT_AXES = frozenset({"C", "W", "H"})
INVALID_NATIVE_CONCAT_AXES = frozenset({"HW"})


def _raw_concat_physical_span(axis: str) -> int:
    left_shape, right_shape, _, _ = RAW_CONCAT_SPECS[axis]
    return (
        _raw_concat_operand_span(left_shape)
        + _raw_concat_operand_span(right_shape)
    )


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
PAD_INPUT_BYTES = codec.physical_layout(
    PAD_SOURCE_SHAPE, "NCx", FP16_BYTES
).physical_bytes
PAD_RESULT_BYTES = codec.physical_layout(
    PAD_DESTINATION_SHAPE, "NCx", FP16_BYTES
).physical_bytes

IMG2COL_SOURCE_SHAPE = (2, 9, 11, 65)
IMG2COL_DESTINATION_SHAPE = (2, 6, 54, 65)
IMG2COL_INPUT_BYTES = codec.physical_layout(
    IMG2COL_SOURCE_SHAPE, "NCx", FP16_BYTES
).physical_bytes
IMG2COL_RESULT_BYTES = codec.physical_layout(
    IMG2COL_DESTINATION_SHAPE, "NCx", FP16_BYTES
).physical_bytes

TENSOR_NOM_SHAPE = (2, 7, 9, 65)
TENSOR_NOM_INPUT_BYTES = _compact_bytes(TENSOR_NOM_SHAPE)
TENSOR_NOM_RESULT_BYTES = codec.physical_layout(
    TENSOR_NOM_SHAPE, "NCx", FP16_BYTES
).physical_bytes

CONCAT_CASES = (
    _case(
        0,
        "datamove-raw-concat-c-n2h7w9-c33-c32",
        "raw-concat",
        131,
        _raw_concat_physical_span("C"),
        _compact_bytes(RAW_CONCAT_SPECS["C"][2]),
        ct=1,
        oracle=ORACLE_OBSERVATION,
        semantic_axis="C",
        output_span=_raw_concat_physical_span("C"),
    ),
    _case(
        1,
        "datamove-raw-concat-w-n2h7-w4-w5-c65",
        "raw-concat",
        131,
        _raw_concat_physical_span("W"),
        _compact_bytes(RAW_CONCAT_SPECS["W"][2]),
        ct=1,
        oracle=ORACLE_OBSERVATION,
        semantic_axis="W",
        output_span=_raw_concat_physical_span("W"),
    ),
    _case(
        2,
        "datamove-raw-concat-h-n2-h3-h4-w9-c65",
        "raw-concat",
        131,
        _raw_concat_physical_span("H"),
        _compact_bytes(RAW_CONCAT_SPECS["H"][2]),
        ct=1,
        oracle=ORACLE_OBSERVATION,
        semantic_axis="H",
        output_span=_raw_concat_physical_span("H"),
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
        TENSOR_NOM_INPUT_BYTES,
        TENSOR_NOM_RESULT_BYTES,
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
        "tdma-fp16-strided-128b-x32",
        "tdma-fp16-strided-128",
        123,
        0,
        8064,
        tdma=1,
        dtype="FP16",
        output_span=8192,
    ),
    _case(
        13,
        "tdma-bf16-strided-64b-x64",
        "tdma-bf16-strided-64",
        123,
        0,
        8128,
        tdma=1,
        dtype="BF16",
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
ALL_CASES = tuple(
    sorted(CATALOG, key=lambda case: case.case_id)
)
CASES_BY_ID = {case.case_id: case for case in ALL_CASES}
CASES_BY_NAME = {case.name: case for case in ALL_CASES}


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
    left_shape, right_shape, _, logical_axis = RAW_CONCAT_SPECS[
        str(case.semantic_axis)
    ]
    left = _logical_values(_product(left_shape), seed)
    right = _logical_values(_product(right_shape), seed + 41)
    physical_left = codec.pack_scalar_bytes(
        left_shape,
        "NCx",
        FP16_BYTES,
        left,
        padding=SLOT_CANARY,
    )
    physical_right = codec.pack_scalar_bytes(
        right_shape,
        "NCx",
        FP16_BYTES,
        right,
        padding=SLOT_CANARY,
    )
    outer = _product(left_shape[:logical_axis])
    left_chunk = _product(left_shape[logical_axis:])
    right_chunk = _product(right_shape[logical_axis:])
    result: list[bytes] = []
    for index in range(outer):
        result.extend(left[index * left_chunk : (index + 1) * left_chunk])
        result.extend(
            right[index * right_chunk : (index + 1) * right_chunk]
        )
    return physical_left + physical_right, b"".join(result)


def _pad_payload(seed: int) -> tuple[bytes, bytes]:
    n, h, w, c = PAD_SOURCE_SHAPE
    values = _logical_values(n * h * w * c, seed)
    zero = _f16_bits(0)

    def source(ni: int, hi: int, wi: int, ci: int) -> bytes:
        return values[((ni * h + hi) * w + wi) * c + ci]

    result = tuple(
        source(ni, hi - 1, wi - 2, ci)
        if 1 <= hi < 6 and 2 <= wi < 9
        else zero
        for ni in range(2)
        for hi in range(7)
        for wi in range(10)
        for ci in range(65)
    )
    return (
        codec.pack_scalar_bytes(
            PAD_SOURCE_SHAPE,
            "NCx",
            FP16_BYTES,
            values,
            padding=0,
        ),
        codec.pack_scalar_bytes(
            PAD_DESTINATION_SHAPE,
            "NCx",
            FP16_BYTES,
            result,
            padding=0,
        ),
    )


def _img2col_payload(seed: int) -> tuple[bytes, bytes]:
    n, h, w, c = IMG2COL_SOURCE_SHAPE
    values = _logical_values(n * h * w * c, seed)
    zero = _f16_bits(0)

    def source(ni: int, hi: int, wi: int, ci: int) -> bytes:
        if hi < 0 or hi >= h or wi < 0 or wi >= w:
            return zero
        return values[((ni * h + hi) * w + wi) * c + ci]

    result = tuple(
        source(ni, oh + ky - 1, ow * 2 + kx - 2, ci)
        for ni in range(2)
        for ky in range(2)
        for kx in range(3)
        for oh in range(9)
        for ow in range(6)
        for ci in range(65)
    )
    return (
        codec.pack_scalar_bytes(
            IMG2COL_SOURCE_SHAPE,
            "NCx",
            FP16_BYTES,
            values,
            padding=0,
        ),
        codec.pack_scalar_bytes(
            IMG2COL_DESTINATION_SHAPE,
            "NCx",
            FP16_BYTES,
            result,
            padding=0,
            batch_padding=SLOT_CANARY,
        ),
    )


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
    if case.operation == "tdma-fp16-strided-128":
        result = bytearray([SLOT_CANARY] * case.result_bytes)
        value = struct.pack("<H", 0x3C00)
        for iteration in range(32):
            begin = iteration * 256
            result[begin : begin + 128] = value * 64
        return bytes(result)
    if case.operation == "tdma-bf16-strided-64":
        result = bytearray([SLOT_CANARY] * case.result_bytes)
        value = struct.pack("<H", 0x3F80)
        for iteration in range(64):
            begin = iteration * 128
            result[begin : begin + 64] = value * 32
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
        values = _logical_values(_product(TENSOR_NOM_SHAPE), seed)
        return (
            b"".join(values),
            codec.pack_scalar_bytes(
                TENSOR_NOM_SHAPE,
                "NCx",
                FP16_BYTES,
                values,
                padding=0,
            ),
        )
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
    words[REQ["WORD_COUNT"]] = REQUEST_WORDS
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
    opcode: tuple(case for case in ALL_CASES if case.opcode == opcode)
    for opcode in range(121, 139)
}

CALIBRATION_LEAF_BINDINGS = {
    "raw-concat-c-w-h": CONCAT_CASES,
    "large-pad-img2col": LARGE_TYPED_CASES,
    "mask-gather-and-bit-vector": RAW_OBSERVATION_CASES[:2],
    "tensor-nom": RAW_OBSERVATION_CASES[2:],
    "materialize-then-consume": COMPOSITE_CASES,
    "tdma-fp16-bf16-strided": TDMA_DESCRIPTOR_CASES[:2],
    "tdma-fp16-bf16-raw-crt": TDMA_DESCRIPTOR_CASES[2:],
}
