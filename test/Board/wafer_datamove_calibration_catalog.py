#!/usr/bin/env python3
"""Large-shape DataMove/layout board cases and complete public dispositions."""

from __future__ import annotations

import dataclasses
import struct

import wafer_ct_vector_calibration_catalog as ct_inventory
import wafer_physical_tensor_codec as codec


REQUEST_MAGIC = 0x315145524D444357
RECORD_MAGIC = 0x314345524D444357
REQUEST_GUARD = 0xF6E5D4C3B2A1908F
RECORD_GUARD = 0x1029384756AABBCC
SCHEMA = 1
REQUEST_WORDS = 16
RECORD_WORDS = 32
RESOURCE_BYTES = 131072
SLOT_BYTES = 65536
BODY_OFFSET = 256
OUTPUT_DDR_OFFSET = SLOT_BYTES
SLOT_CANARY = 0xA7
ELEMENT_BYTES = 2
REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "INPUT_BYTES": 3,
    "RESULT_BYTES": 4,
    "OUTPUT_SPAN": 5,
    "EXPECTED_INSTRUCTIONS": 6,
    "SAMPLE": 7,
    "RESOURCE_BYTES": 8,
    "SLOT_BYTES": 9,
    "BODY_OFFSET": 10,
    "GUARD": 15,
}
REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "INPUT_BYTES": 4,
    "RESULT_BYTES": 5,
    "OUTPUT_SPAN": 6,
    "EXPECTED_INSTRUCTIONS": 7,
    "SAMPLE": 8,
    "REQUEST_GUARD": 9,
    "OUTPUT_DDR_OFFSET": 10,
    "SLOT_BYTES": 11,
    "BODY_OFFSET": 12,
    "OUTPUT_GUARD_MISMATCHES": 13,
    "TDMA_INST_DELTA": 14,
    "TDMA_EXEC_DELTA": 15,
    "TDMA_BLOCKING_DELTA": 16,
    "RECORD_GUARD": 31,
}


@dataclasses.dataclass(frozen=True)
class DataMoveCase:
    case_id: int
    name: str
    operation: str
    source_layout: str
    destination_layout: str
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]
    input_bytes: int
    result_bytes: int
    output_span: int
    expected_instructions: int
    opcode: int | None
    realization: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "operation": self.operation,
            "opcode": self.opcode,
            "source_layout": self.source_layout,
            "destination_layout": self.destination_layout,
            "source_shape": self.source_shape,
            "destination_shape": self.destination_shape,
            "input_bytes": self.input_bytes,
            "result_bytes": self.result_bytes,
            "output_span": self.output_span,
            "expected_instructions": self.expected_instructions,
            "realization": self.realization,
            "disposition": "board-exact",
            "oracle": "all-logical-points+physical-padding+full-slot-canary",
        }


def _span(result_bytes: int) -> int:
    return (result_bytes + 255) // 256 * 256


def _compact_bytes(shape: tuple[int, ...]) -> int:
    result = ELEMENT_BYTES
    for dimension in shape:
        result *= dimension
    return result


def _case(
    case_id: int,
    name: str,
    operation: str,
    source_layout: str,
    destination_layout: str,
    source_shape: tuple[int, ...],
    destination_shape: tuple[int, ...],
    input_bytes: int,
    result_bytes: int,
    instructions: int,
    opcode: int | None,
    realization: str = "gather-scatter-materialized",
) -> DataMoveCase:
    return DataMoveCase(
        case_id,
        name,
        operation,
        source_layout,
        destination_layout,
        source_shape,
        destination_shape,
        input_bytes,
        result_bytes,
        _span(result_bytes),
        instructions,
        opcode,
        realization,
    )


TRANSFORM_SHAPE = (37, 53)
ROTATE_180_SHAPE = (17, 19)
NCHW_SHAPE = (2, 5, 17, 19)
NHWC_SHAPE = (2, 17, 19, 5)
LAYOUT_SHAPE = (2, 7, 9, 65)
LAYOUT_COMPACT_BYTES = _compact_bytes(LAYOUT_SHAPE)
CX_BYTES = codec.physical_layout(LAYOUT_SHAPE, "Cx", ELEMENT_BYTES).physical_bytes
NCX_BYTES = codec.physical_layout(LAYOUT_SHAPE, "NCx", ELEMENT_BYTES).physical_bytes
CONCAT_LEFT_SHAPE = (2, 17, 19, 7)
CONCAT_RIGHT_SHAPE = (2, 17, 19, 11)
CONCAT_OUTPUT_SHAPE = (2, 17, 19, 18)

CATALOG = (
    _case(
        0, "datamove-transpose-37x53", "transpose", "Tensor", "Tensor",
        TRANSFORM_SHAPE, (53, 37), _compact_bytes(TRANSFORM_SHAPE),
        _compact_bytes(TRANSFORM_SHAPE), 1, 125,
    ),
    _case(
        1, "datamove-mirror-axis1-37x53", "mirror", "Tensor", "Tensor",
        TRANSFORM_SHAPE, TRANSFORM_SHAPE, _compact_bytes(TRANSFORM_SHAPE),
        _compact_bytes(TRANSFORM_SHAPE), 53, 124,
    ),
    _case(
        2, "datamove-rotate90-37x53", "rotate90", "Tensor", "Tensor",
        TRANSFORM_SHAPE, (53, 37), _compact_bytes(TRANSFORM_SHAPE),
        _compact_bytes(TRANSFORM_SHAPE), 37, 126,
    ),
    _case(
        3, "datamove-rotate180-17x19", "rotate180", "Tensor", "Tensor",
        ROTATE_180_SHAPE, ROTATE_180_SHAPE,
        _compact_bytes(ROTATE_180_SHAPE),
        _compact_bytes(ROTATE_180_SHAPE), 17 * 19, 127,
    ),
    _case(
        4, "datamove-rotate270-37x53", "rotate270", "Tensor", "Tensor",
        TRANSFORM_SHAPE, (53, 37), _compact_bytes(TRANSFORM_SHAPE),
        _compact_bytes(TRANSFORM_SHAPE), 53, 128,
    ),
    _case(
        5, "datamove-nchw2nhwc-n2c5h17w19", "nchw2nhwc", "Tensor",
        "Tensor", NCHW_SHAPE, NHWC_SHAPE, _compact_bytes(NCHW_SHAPE),
        _compact_bytes(NHWC_SHAPE), 2, 129,
    ),
    _case(
        6, "datamove-nhwc2nchw-n2h17w19c5", "nhwc2nchw", "Tensor",
        "Tensor", NHWC_SHAPE, NCHW_SHAPE, _compact_bytes(NHWC_SHAPE),
        _compact_bytes(NCHW_SHAPE), 2, 130,
    ),
    _case(
        7, "datamove-concat-c-n2h17w19-c7-c11", "concat", "Tensor",
        "Tensor", CONCAT_LEFT_SHAPE + CONCAT_RIGHT_SHAPE,
        CONCAT_OUTPUT_SHAPE,
        _compact_bytes(CONCAT_LEFT_SHAPE)
        + _compact_bytes(CONCAT_RIGHT_SHAPE),
        _compact_bytes(CONCAT_OUTPUT_SHAPE), 2, 131,
    ),
    _case(
        8, "datamove-broadcast-row-53-to-37x53", "broadcast-row",
        "Tensor", "Tensor", (53,), TRANSFORM_SHAPE, _compact_bytes((53,)),
        _compact_bytes(TRANSFORM_SHAPE), 1, 135,
    ),
    _case(
        9, "datamove-broadcast-column-37-to-37x53", "broadcast-column",
        "Tensor", "Tensor", (37,), TRANSFORM_SHAPE, _compact_bytes((37,)),
        _compact_bytes(TRANSFORM_SHAPE), 1, 135,
    ),
    _case(
        10, "datamove-tensor-to-cx-n2h7w9c65", "tensor-to-cx", "Tensor",
        "Cx", LAYOUT_SHAPE, LAYOUT_SHAPE, LAYOUT_COMPACT_BYTES, CX_BYTES, 2,
        133,
    ),
    _case(
        11, "datamove-cx-to-tensor-n2h7w9c65", "cx-to-tensor", "Cx",
        "Tensor", LAYOUT_SHAPE, LAYOUT_SHAPE, CX_BYTES, LAYOUT_COMPACT_BYTES,
        2, 133,
    ),
    _case(
        12, "datamove-tensor-to-ncx-n2h7w9c65", "tensor-to-ncx", "Tensor",
        "NCx", LAYOUT_SHAPE, LAYOUT_SHAPE, LAYOUT_COMPACT_BYTES, NCX_BYTES, 4,
        133,
    ),
    _case(
        13, "datamove-ncx-to-tensor-n2h7w9c65", "ncx-to-tensor", "NCx",
        "Tensor", LAYOUT_SHAPE, LAYOUT_SHAPE, NCX_BYTES, LAYOUT_COMPACT_BYTES,
        4, 133,
    ),
    _case(
        14, "datamove-gather-even-columns-64x65", "gather-scatter-strided",
        "Tensor", "Tensor", (64, 65), (64, 33), _compact_bytes((64, 65)),
        _compact_bytes((64, 33)), 1, 135,
    ),
)
CASES_BY_NAME = {case.name: case for case in CATALOG}
CASES_BY_ID = {case.case_id: case for case in CATALOG}


@dataclasses.dataclass(frozen=True)
class PublicMovementDisposition:
    opcode: int
    opcode_name: str
    operation: str
    disposition: str
    evidence: tuple[str, ...]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


_BOARD_CASES_BY_OPCODE = {
    opcode: tuple(case.name for case in CATALOG if case.opcode == opcode)
    for opcode in range(121, 139)
}
_EXISTING_EVIDENCE = {
    121: ("unpool-f16",),
    123: ("unpool-f16",),
    132: ("tdma-pad-f16",),
    134: (
        "select-bit2fp-maskmove-f16",
        "select-bit2fp-maskmove-bf16",
    ),
    138: ("tdma-img2col-f16", "tdma-img2col-bf16"),
}
_DEFERRED = {
    122: "unpool-avg geometry lacks an independent large-shape physical oracle",
    136: "MaskGather index ownership and bounded write span are not qualified",
    137: "bit-vector MaskGather index ownership is not qualified",
}


def build_public_dispositions() -> tuple[PublicMovementDisposition, ...]:
    rows: list[PublicMovementDisposition] = []
    for opcode in range(121, 139):
        name = ct_inventory.OPCODE_NAMES[opcode]
        operation = name.split("DataMoveOp_", 1)[1].lower()
        evidence = _BOARD_CASES_BY_OPCODE[opcode] + _EXISTING_EVIDENCE.get(
            opcode, ()
        )
        if opcode in _DEFERRED:
            disposition = "isolated-deferred"
            reason = _DEFERRED[opcode]
        elif evidence:
            disposition = "board-executable"
            reason = None
        else:
            raise RuntimeError(f"public DataMove opcode {opcode} has no disposition")
        rows.append(
            PublicMovementDisposition(
                opcode, name, operation, disposition, evidence, reason
            )
        )
    return tuple(rows)


PUBLIC_DISPOSITIONS = build_public_dispositions()


@dataclasses.dataclass(frozen=True)
class InstructionLayoutDisposition:
    instruction_family: str
    layout: str
    disposition: str
    evidence: tuple[str, ...]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


INSTRUCTION_LAYOUT_DISPOSITIONS = (
    InstructionLayoutDisposition(
        "CT", "Tensor", "native-board-executable",
        ("wafer-ct-vector-calibration-catalog-python",),
    ),
    InstructionLayoutDisposition(
        "CT", "NTensor", "static-negative", (),
        "direct CT NTensor qualification is absent; materialize compact Tensor first",
    ),
    InstructionLayoutDisposition(
        "CT", "Cx", "materialize-then-tensor",
        ("datamove-cx-to-tensor-n2h7w9c65",),
    ),
    InstructionLayoutDisposition(
        "CT", "NCx", "materialize-then-tensor",
        ("datamove-ncx-to-tensor-n2h7w9c65",),
    ),
    InstructionLayoutDisposition(
        "NE", "Tensor", "materialize-to-native",
        ("datamove-tensor-to-cx-n2h7w9c65",),
    ),
    InstructionLayoutDisposition(
        "NE", "NTensor", "static-negative", (),
        "NE packet geometry requires explicit Cx/NCx materialization",
    ),
    InstructionLayoutDisposition(
        "NE", "Cx", "native-board-executable",
        ("wafer-ne-calibration-catalog-python",),
    ),
    InstructionLayoutDisposition(
        "NE", "NCx", "native-board-executable",
        ("wafer-ne-calibration-catalog-python",),
    ),
    InstructionLayoutDisposition(
        "RDMA", "Tensor", "native-board-executable",
        ("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    InstructionLayoutDisposition(
        "RDMA", "NTensor", "static-negative", (),
        "no independent NTensor descriptor/range board oracle is qualified",
    ),
    InstructionLayoutDisposition(
        "RDMA", "Cx", "static-negative", (),
        "Cx logical-point transfer must use explicit physical segments",
    ),
    InstructionLayoutDisposition(
        "RDMA", "NCx", "static-negative", (),
        "NCx batch-stride transfer must use explicit physical segments",
    ),
    InstructionLayoutDisposition(
        "WDMA", "Tensor", "native-board-executable",
        ("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    InstructionLayoutDisposition(
        "WDMA", "NTensor", "static-negative", (),
        "no independent NTensor descriptor/range board oracle is qualified",
    ),
    InstructionLayoutDisposition(
        "WDMA", "Cx", "static-negative", (),
        "Cx logical-point writeback must use explicit physical segments",
    ),
    InstructionLayoutDisposition(
        "WDMA", "NCx", "static-negative", (),
        "NCx batch-stride writeback must use explicit physical segments",
    ),
    InstructionLayoutDisposition(
        "TDMA", "Tensor", "native-board-executable",
        ("datamove-transpose-37x53", "datamove-concat-c-n2h17w19-c7-c11"),
    ),
    InstructionLayoutDisposition(
        "TDMA", "NTensor", "static-negative", (),
        "NTensor has no distinct qualified physical movement contract",
    ),
    InstructionLayoutDisposition(
        "TDMA", "Cx", "native-board-executable",
        (
            "datamove-tensor-to-cx-n2h7w9c65",
            "datamove-cx-to-tensor-n2h7w9c65",
        ),
    ),
    InstructionLayoutDisposition(
        "TDMA", "NCx", "native-board-executable",
        (
            "datamove-tensor-to-ncx-n2h7w9c65",
            "datamove-ncx-to-tensor-n2h7w9c65",
        ),
    ),
)


def _logical_values(count: int, seed: int) -> tuple[bytes, ...]:
    return tuple(
        struct.pack("<H", (seed * 257 + index * 73 + (index >> 5) * 19) & 0xFFFF)
        for index in range(count)
    )


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _matrix(values: tuple[bytes, ...], rows: int, columns: int) -> list[list[bytes]]:
    return [
        list(values[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]


def _build_input_expected(case: DataMoveCase, seed: int) -> tuple[bytes, bytes]:
    operation = case.operation
    if operation in ("transpose", "mirror", "rotate90", "rotate270"):
        rows, columns = TRANSFORM_SHAPE
        values = _logical_values(rows * columns, seed)
        source = _matrix(values, rows, columns)
        if operation == "transpose":
            output = [
                source[row][column]
                for column in range(columns)
                for row in range(rows)
            ]
        elif operation == "mirror":
            output = [
                source[row][columns - 1 - column]
                for row in range(rows)
                for column in range(columns)
            ]
        elif operation == "rotate90":
            output = [
                source[rows - 1 - column][row]
                for row in range(columns)
                for column in range(rows)
            ]
        else:
            output = [
                source[column][columns - 1 - row]
                for row in range(columns)
                for column in range(rows)
            ]
        return b"".join(values), b"".join(output)

    if operation == "rotate180":
        rows, columns = ROTATE_180_SHAPE
        values = _logical_values(rows * columns, seed)
        source = _matrix(values, rows, columns)
        output = [
            source[rows - 1 - row][columns - 1 - column]
            for row in range(rows)
            for column in range(columns)
        ]
        return b"".join(values), b"".join(output)

    if operation == "nchw2nhwc":
        n, c, h, w = NCHW_SHAPE
        values = _logical_values(n * c * h * w, seed)
        def at(ni: int, ci: int, hi: int, wi: int) -> bytes:
            return values[((ni * c + ci) * h + hi) * w + wi]
        output = (
            at(ni, ci, hi, wi)
            for ni in range(n)
            for hi in range(h)
            for wi in range(w)
            for ci in range(c)
        )
        return b"".join(values), b"".join(output)

    if operation == "nhwc2nchw":
        n, h, w, c = NHWC_SHAPE
        values = _logical_values(n * h * w * c, seed)
        def at(ni: int, hi: int, wi: int, ci: int) -> bytes:
            return values[((ni * h + hi) * w + wi) * c + ci]
        output = (
            at(ni, hi, wi, ci)
            for ni in range(n)
            for ci in range(c)
            for hi in range(h)
            for wi in range(w)
        )
        return b"".join(values), b"".join(output)

    if operation == "concat":
        outer = 2 * 17 * 19
        left = _logical_values(outer * 7, seed)
        right = _logical_values(outer * 11, seed + 37)
        output: list[bytes] = []
        for index in range(outer):
            output.extend(left[index * 7 : (index + 1) * 7])
            output.extend(right[index * 11 : (index + 1) * 11])
        return b"".join(left + right), b"".join(output)

    if operation == "broadcast-row":
        values = _logical_values(53, seed)
        return b"".join(values), b"".join(values * 37)

    if operation == "broadcast-column":
        values = _logical_values(37, seed)
        output = tuple(value for value in values for _ in range(53))
        return b"".join(values), b"".join(output)

    if operation in (
        "tensor-to-cx",
        "cx-to-tensor",
        "tensor-to-ncx",
        "ncx-to-tensor",
    ):
        logical = _logical_values(_product(LAYOUT_SHAPE), seed)
        if operation == "tensor-to-cx":
            return b"".join(logical), codec.pack_scalar_bytes(
                LAYOUT_SHAPE, "Cx", ELEMENT_BYTES, logical, padding=SLOT_CANARY
            )
        if operation == "cx-to-tensor":
            return (
                codec.pack_scalar_bytes(
                    LAYOUT_SHAPE, "Cx", ELEMENT_BYTES, logical,
                    padding=SLOT_CANARY,
                ),
                b"".join(logical),
            )
        if operation == "tensor-to-ncx":
            return b"".join(logical), codec.pack_scalar_bytes(
                LAYOUT_SHAPE, "NCx", ELEMENT_BYTES, logical,
                padding=SLOT_CANARY,
            )
        return (
            codec.pack_scalar_bytes(
                LAYOUT_SHAPE, "NCx", ELEMENT_BYTES, logical,
                padding=SLOT_CANARY,
            ),
            b"".join(logical),
        )

    if operation == "gather-scatter-strided":
        values = _logical_values(64 * 65, seed)
        output = tuple(
            values[row * 65 + column]
            for row in range(64)
            for column in range(0, 65, 2)
        )
        return b"".join(values), b"".join(output)

    raise RuntimeError(f"{case.name}: no host movement oracle")


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_output_slot: bytes


def build_case_payload(case: DataMoveCase, sample: int = 0) -> CasePayload:
    source, result = _build_input_expected(case, sample + 1)
    if len(source) != case.input_bytes or len(result) != case.result_bytes:
        raise RuntimeError(f"{case.name}: generated movement size mismatch")
    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
    words[REQ["CASE"]] = case.case_id
    words[REQ["INPUT_BYTES"]] = case.input_bytes
    words[REQ["RESULT_BYTES"]] = case.result_bytes
    words[REQ["OUTPUT_SPAN"]] = case.output_span
    words[REQ["EXPECTED_INSTRUCTIONS"]] = case.expected_instructions
    words[REQ["SAMPLE"]] = sample
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["SLOT_BYTES"]] = SLOT_BYTES
    words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    payload = bytearray([SLOT_CANARY] * RESOURCE_BYTES)
    payload[BODY_OFFSET : BODY_OFFSET + len(source)] = source
    expected = bytearray([SLOT_CANARY] * SLOT_BYTES)
    expected[BODY_OFFSET : BODY_OFFSET + len(result)] = result
    return CasePayload(
        request + bytes([SLOT_CANARY]) * (RESOURCE_BYTES - len(request)),
        bytes(payload),
        bytes(expected),
    )
