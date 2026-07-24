#!/usr/bin/env python3
"""Large-shape DataMove/layout board cases and complete public dispositions."""

from __future__ import annotations

import dataclasses
import struct

import wafer_ct_vector_calibration_catalog as ct_inventory
import wafer_datamove_extended_calibration_catalog as extended
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
    source1_shape: tuple[int, ...] | None
    destination_shape: tuple[int, ...]
    input_bytes: int
    result_bytes: int
    output_span: int
    expected_instructions: int
    opcode: int | None
    realization: str
    semantic_axis: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "operation": self.operation,
            "opcode": self.opcode,
            "source_layout": self.source_layout,
            "destination_layout": self.destination_layout,
            "source_shape": self.source_shape,
            "source1_shape": self.source1_shape,
            "destination_shape": self.destination_shape,
            "input_bytes": self.input_bytes,
            "result_bytes": self.result_bytes,
            "output_span": self.output_span,
            "expected_instructions": self.expected_instructions,
            "realization": self.realization,
            "semantic_axis": self.semantic_axis,
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
    *,
    source1_shape: tuple[int, ...] | None = None,
    semantic_axis: str | None = None,
) -> DataMoveCase:
    return DataMoveCase(
        case_id,
        name,
        operation,
        source_layout,
        destination_layout,
        source_shape,
        source1_shape,
        destination_shape,
        input_bytes,
        result_bytes,
        _span(result_bytes),
        instructions,
        opcode,
        realization,
        semantic_axis,
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
        "Tensor", CONCAT_LEFT_SHAPE,
        CONCAT_OUTPUT_SHAPE,
        _compact_bytes(CONCAT_LEFT_SHAPE)
        + _compact_bytes(CONCAT_RIGHT_SHAPE),
        _compact_bytes(CONCAT_OUTPUT_SHAPE), 2, 135,
        source1_shape=CONCAT_RIGHT_SHAPE, semantic_axis="C",
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
        135,
    ),
    _case(
        11, "datamove-cx-to-tensor-n2h7w9c65", "cx-to-tensor", "Cx",
        "Tensor", LAYOUT_SHAPE, LAYOUT_SHAPE, CX_BYTES, LAYOUT_COMPACT_BYTES,
        2, 135,
    ),
    _case(
        12, "datamove-tensor-to-ncx-n2h7w9c65", "tensor-to-ncx", "Tensor",
        "NCx", LAYOUT_SHAPE, LAYOUT_SHAPE, LAYOUT_COMPACT_BYTES, NCX_BYTES, 4,
        135,
    ),
    _case(
        13, "datamove-ncx-to-tensor-n2h7w9c65", "ncx-to-tensor", "NCx",
        "Tensor", LAYOUT_SHAPE, LAYOUT_SHAPE, NCX_BYTES, LAYOUT_COMPACT_BYTES,
        4, 135,
    ),
    _case(
        14, "datamove-gather-even-columns-64x65", "gather-scatter-strided",
        "Tensor", "Tensor", (64, 65), (64, 33), _compact_bytes((64, 65)),
        _compact_bytes((64, 33)), 1, 135,
    ),
)


def _concat_case(
    case_id: int,
    axis: str,
    left: tuple[int, ...],
    right: tuple[int, ...],
    output: tuple[int, ...],
) -> DataMoveCase:
    return _case(
        case_id,
        f"datamove-concat-{axis.lower()}-"
        + "x".join(str(value) for value in output),
        "concat",
        "Tensor",
        "Tensor",
        left,
        output,
        _compact_bytes(left) + _compact_bytes(right),
        _compact_bytes(output),
        2,
        135,
        source1_shape=right,
        semantic_axis=axis,
    )


_LARGE_LAYOUT_SHAPE = (2, 7, 9, 65)
_CONCAT_CASES = (
    _concat_case(
        15, "C", (2, 7, 9, 33), (2, 7, 9, 32), _LARGE_LAYOUT_SHAPE
    ),
    _concat_case(
        16, "W", (2, 7, 4, 65), (2, 7, 5, 65), _LARGE_LAYOUT_SHAPE
    ),
    _concat_case(
        17, "H", (2, 3, 9, 65), (2, 4, 9, 65), _LARGE_LAYOUT_SHAPE
    ),
    _concat_case(
        18, "HW", (2, 2, 5, 65), (2, 3, 7, 65), (2, 1, 31, 65)
    ),
    _concat_case(
        19, "N", (1, 7, 9, 65), (1, 7, 9, 65), _LARGE_LAYOUT_SHAPE
    ),
)

_BROADCAST_CASES = (
    _case(
        20,
        "datamove-broadcast-scalar-to-n2h7w9c65",
        "broadcast-scalar",
        "Tensor",
        "Tensor",
        (1,),
        _LARGE_LAYOUT_SHAPE,
        ELEMENT_BYTES,
        _compact_bytes(_LARGE_LAYOUT_SHAPE),
        1,
        135,
    ),
    _case(
        21,
        "datamove-broadcast-channel-c65-to-n2h7w9c65",
        "broadcast-channel",
        "Tensor",
        "Tensor",
        (65,),
        _LARGE_LAYOUT_SHAPE,
        _compact_bytes((65,)),
        _compact_bytes(_LARGE_LAYOUT_SHAPE),
        1,
        135,
    ),
    _case(
        22,
        "datamove-broadcast-row-w9c65-to-n2h7w9c65",
        "broadcast-row-large",
        "Tensor",
        "Tensor",
        (9, 65),
        _LARGE_LAYOUT_SHAPE,
        _compact_bytes((9, 65)),
        _compact_bytes(_LARGE_LAYOUT_SHAPE),
        1,
        135,
    ),
)


def _layout_cases(first_case_id: int) -> tuple[DataMoveCase, ...]:
    rows: list[DataMoveCase] = []
    case_id = first_case_id
    for channels in (63, 64, 127, 129):
        shape = (2, 7, 9, channels)
        compact = _compact_bytes(shape)
        cx_bytes = codec.physical_layout(shape, "Cx", ELEMENT_BYTES).physical_bytes
        ncx_bytes = codec.physical_layout(
            shape, "NCx", ELEMENT_BYTES
        ).physical_bytes
        cx_instructions = codec.physical_layout(
            shape, "Cx", ELEMENT_BYTES
        ).full_blocks + (
            codec.physical_layout(shape, "Cx", ELEMENT_BYTES).tail_width != 0
        )
        ncx_instructions = 2 * cx_instructions
        rows.extend(
            (
                _case(
                    case_id,
                    f"datamove-tensor-to-cx-n2h7w9c{channels}",
                    "tensor-to-cx",
                    "Tensor",
                    "Cx",
                    shape,
                    shape,
                    compact,
                    cx_bytes,
                    int(cx_instructions),
                    135,
                ),
                _case(
                    case_id + 1,
                    f"datamove-cx-to-tensor-n2h7w9c{channels}",
                    "cx-to-tensor",
                    "Cx",
                    "Tensor",
                    shape,
                    shape,
                    cx_bytes,
                    compact,
                    int(cx_instructions),
                    135,
                ),
                _case(
                    case_id + 2,
                    f"datamove-tensor-to-ncx-n2h7w9c{channels}",
                    "tensor-to-ncx",
                    "Tensor",
                    "NCx",
                    shape,
                    shape,
                    compact,
                    ncx_bytes,
                    int(ncx_instructions),
                    135,
                ),
                _case(
                    case_id + 3,
                    f"datamove-ncx-to-tensor-n2h7w9c{channels}",
                    "ncx-to-tensor",
                    "NCx",
                    "Tensor",
                    shape,
                    shape,
                    ncx_bytes,
                    compact,
                    int(ncx_instructions),
                    135,
                ),
            )
        )
        case_id += 4
    return tuple(rows)


_LARGE_GATHER_CASES = (
    _case(
        39,
        "datamove-gather-contiguous-32768b",
        "gather-contiguous-large",
        "Tensor",
        "Tensor",
        (16384,),
        (16384,),
        32768,
        32768,
        1,
        135,
    ),
    _case(
        40,
        "datamove-gather-1d-holes-32768b-to-16384b",
        "gather-1d-holes-large",
        "Tensor",
        "Tensor",
        (16384,),
        (8192,),
        32768,
        16384,
        1,
        135,
    ),
    _case(
        41,
        "datamove-gather-2d-holes-128x256b-to-128x128b",
        "gather-2d-holes-large",
        "Tensor",
        "Tensor",
        (128, 128),
        (128, 64),
        32768,
        16384,
        1,
        135,
    ),
    _case(
        42,
        "datamove-gather-3d-holes-7x16x256b-to-7x16x128b",
        "gather-3d-holes-large",
        "Tensor",
        "Tensor",
        (7, 16, 128),
        (7, 16, 64),
        53248,
        14336,
        1,
        135,
    ),
    _case(
        43,
        "datamove-gather-tail-16385xf16",
        "gather-tail-large",
        "Tensor",
        "Tensor",
        (16385,),
        (16385,),
        32770,
        32770,
        1,
        135,
    ),
)

_LARGE_NCHW_CASES = (
    _case(
        44,
        "datamove-nchw2nhwc-n2c65h7w9",
        "nchw2nhwc",
        "Tensor",
        "Tensor",
        (2, 65, 7, 9),
        (2, 7, 9, 65),
        16380,
        16380,
        2,
        129,
    ),
    _case(
        45,
        "datamove-nhwc2nchw-n2h7w9c65",
        "nhwc2nchw",
        "Tensor",
        "Tensor",
        (2, 7, 9, 65),
        (2, 65, 7, 9),
        16380,
        16380,
        2,
        130,
    ),
)

CATALOG = (
    CATALOG
    + _CONCAT_CASES
    + _BROADCAST_CASES
    + _layout_cases(23)
    + _LARGE_GATHER_CASES
    + _LARGE_NCHW_CASES
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

    @property
    def key(self) -> str:
        return f"public-datamove-opcode-{self.opcode}"

    def as_dict(self) -> dict[str, object]:
        return {"key": self.key, **dataclasses.asdict(self)}


_BOARD_CASES_BY_OPCODE = {
    opcode: tuple(case.name for case in CATALOG if case.opcode == opcode)
    for opcode in range(121, 139)
}
_EXISTING_EVIDENCE = {
    121: ("unpool-index-f16",),
    122: ("unpool-avg-f16",),
    123: ("unpool-f16",),
    132: ("tdma-pad-f16",),
    134: (
        "select-bit2fp-maskmove-f16",
        "select-bit2fp-maskmove-bf16",
    ),
    138: ("tdma-img2col-f16", "tdma-img2col-bf16"),
}
_CT_DATAMOVE_OPCODES = frozenset({121, 122, 123, 131, 134, 136, 137})


def _matches_public_opcode_engine(
    opcode: int, case: extended.ExtendedDataMoveCase
) -> bool:
    """Do not confuse the probe's case opcode with another engine's opcode."""
    if opcode in _CT_DATAMOVE_OPCODES:
        return case.expected_ct_instructions > 0
    return case.expected_tdma_instructions > 0


_EXTENDED_EVIDENCE = {
    opcode: tuple(
        case.name
        for case in rows
        if _matches_public_opcode_engine(opcode, case)
    )
    for opcode, rows in extended.EVIDENCE_BY_OPCODE.items()
}
EXISTING_EVIDENCE_OPCODES = {
    "unpool-index-f16": frozenset({118, 121}),
    "unpool-avg-f16": frozenset({122}),
    "unpool-f16": frozenset({118, 123}),
    "tdma-pad-f16": frozenset({132}),
    "select-bit2fp-maskmove-f16": frozenset({134}),
    "select-bit2fp-maskmove-bf16": frozenset({134}),
    "tdma-img2col-f16": frozenset({138}),
    "tdma-img2col-bf16": frozenset({138}),
}


def build_public_dispositions() -> tuple[PublicMovementDisposition, ...]:
    rows: list[PublicMovementDisposition] = []
    for opcode in range(121, 139):
        name = ct_inventory.OPCODE_NAMES[opcode]
        operation = name.split("DataMoveOp_", 1)[1].lower()
        evidence = (
            _BOARD_CASES_BY_OPCODE[opcode]
            + _EXISTING_EVIDENCE.get(opcode, ())
            + _EXTENDED_EVIDENCE.get(opcode, ())
        )
        if opcode in {121, 122, 123}:
            disposition = "typed-profile"
            evidence = (
                "wafer-ct-reduce-pool-capability-catalog-python",
            )
            reason = (
                "unpool capability is keyed by "
                "(opcode,dtype,index/layout,geometry); a representative "
                "instruction-family case does not qualify the bare opcode"
            )
        elif evidence:
            disposition = (
                "board-observation"
                if opcode in {121, 131, 133, 136, 137}
                else "board-executable"
            )
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
class SemanticDisposition:
    name: str
    semantic: str
    disposition: str
    evidence: tuple[str, ...]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


RAW_CONCAT_DISPOSITIONS = tuple(
    SemanticDisposition(
        f"raw-concat-{axis.lower()}",
        f"concat-axis-{axis}",
        (
            "board-observation"
            if axis in extended.NATIVE_CONCAT_AXES
            else "isolated-deferred"
        ),
        tuple(
            case.name
            for case in (
                extended.CONCAT_CASES
                if axis in extended.NATIVE_CONCAT_AXES
                else extended.ISOLATED_CONCAT_CASES
            )
            if case.semantic_axis == axis
        ),
        (
            None
            if axis in extended.NATIVE_CONCAT_AXES
            else (
                "native dims=HW previously timed out at matching completion; "
                "exclude it from the default dispatcher and allow only an "
                "explicit final isolated selection after the bounded C/W/H "
                "alternatives"
            )
        ),
    )
    for axis in ("C", "W", "H", "HW")
)

EXTENDED_DATAMOVE_DISPOSITIONS = (
    SemanticDisposition(
        "pad-large-n2h5w7c65-to-n2h7w10c65",
        "pad-large-non-symmetric",
        "board-executable",
        (extended.LARGE_TYPED_CASES[0].name,),
    ),
    SemanticDisposition(
        "img2col-large-n2h9w11c65",
        "img2col-large-non-square",
        "board-executable",
        (extended.LARGE_TYPED_CASES[1].name,),
    ),
)


@dataclasses.dataclass(frozen=True)
class InstructionLayoutDisposition:
    instruction_family: str
    layout: str
    disposition: str
    evidence: tuple[str, ...]
    reason: str | None = None

    @property
    def key(self) -> str:
        return f"{self.instruction_family.lower()}-{self.layout.lower()}"

    def as_dict(self) -> dict[str, object]:
        return {"key": self.key, **dataclasses.asdict(self)}


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
        "CT", "Cx", "composite-board-executable",
        ("datamove-cx-materialize-ct-add-n2c65",),
    ),
    InstructionLayoutDisposition(
        "CT", "NCx", "composite-board-executable",
        ("datamove-ncx-materialize-ct-add-n2c65",),
    ),
    InstructionLayoutDisposition(
        "NE", "Tensor", "composite-board-executable",
        ("datamove-tensor-materialize-ne-identity-m1k16n16",),
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

INSTRUCTION_LAYOUT_POSITIVE_DISPOSITIONS = tuple(
    row
    for row in INSTRUCTION_LAYOUT_DISPOSITIONS
    if row.disposition
    in {"native-board-executable", "composite-board-executable"}
)

INSTRUCTION_LAYOUT_COMPOSITE_DEFERRED_DISPOSITIONS = tuple(
    row
    for row in INSTRUCTION_LAYOUT_DISPOSITIONS
    if row.disposition == "isolated-deferred"
)

INSTRUCTION_LAYOUT_COMPOSITE_POSITIVE_DISPOSITIONS = tuple(
    row
    for row in INSTRUCTION_LAYOUT_DISPOSITIONS
    if row.disposition == "composite-board-executable"
)

INSTRUCTION_LAYOUT_NONBOARD_DISPOSITIONS = tuple(
    row
    for row in INSTRUCTION_LAYOUT_DISPOSITIONS
    if row.disposition == "static-negative"
)


def _cases_with(**fields: object) -> tuple[DataMoveCase, ...]:
    return tuple(
        case
        for case in CATALOG
        if all(getattr(case, name) == value for name, value in fields.items())
    )


CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "instruction-layout-native-positive": (
        tuple(
            row
            for row in INSTRUCTION_LAYOUT_POSITIVE_DISPOSITIONS
            if row.disposition == "native-board-executable"
        )
    ),
    "instruction-layout-composite-positive": (
        INSTRUCTION_LAYOUT_COMPOSITE_POSITIVE_DISPOSITIONS
    ),
    "instruction-layout-static-negative": (
        INSTRUCTION_LAYOUT_NONBOARD_DISPOSITIONS
    ),
    "cx-ncx-channel-boundaries": tuple(
        case
        for case in CATALOG
        if case.operation
        in {"tensor-to-cx", "cx-to-tensor", "tensor-to-ncx", "ncx-to-tensor"}
    ),
    "cx-ncx-padding-poison-n-slice": tuple(
        case
        for case in CATALOG
        if case.operation in {"tensor-to-ncx", "ncx-to-tensor"}
    ),
    "transpose-mirror-rotate-large": tuple(
        case
        for case in CATALOG
        if case.operation
        in {"transpose", "mirror", "rotate90", "rotate180", "rotate270"}
    ),
    "nchw-nhwc-large": _LARGE_NCHW_CASES,
    "concat-c-w-h-hw": tuple(
        case
        for case in CATALOG
        if case.operation == "concat"
        and case.semantic_axis in {"C", "W", "H", "HW"}
        and case.case_id >= 15
    ),
    "raw-concat-observation": tuple(
        row
        for row in RAW_CONCAT_DISPOSITIONS
        if row.disposition == "board-observation"
    ),
    "raw-concat-hw-isolated": tuple(
        row
        for row in RAW_CONCAT_DISPOSITIONS
        if row.disposition == "isolated-deferred"
    ),
    "compiler-concat-materialization": _cases_with(
        operation="concat", semantic_axis="N"
    ),
    "broadcast-scalar-channel-row": _BROADCAST_CASES,
    "pad-img2col-large": EXTENDED_DATAMOVE_DISPOSITIONS,
    "gather-contiguous-strided-tail": _LARGE_GATHER_CASES,
    "mask-gather-disposition": tuple(
        row for row in PUBLIC_DISPOSITIONS if row.opcode in {136, 137}
    ),
    "tensor-normalization": tuple(
        row for row in PUBLIC_DISPOSITIONS if row.opcode == 133
    ),
    "tdma-layout-materialization": tuple(
        case
        for case in CATALOG
        if case.operation
        in {"tensor-to-cx", "cx-to-tensor", "tensor-to-ncx", "ncx-to-tensor"}
    ),
}


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
        n, c, h, w = case.source_shape
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
        n, h, w, c = case.source_shape
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
        if case.source1_shape is None or case.semantic_axis is None:
            raise RuntimeError(f"{case.name}: concat metadata is incomplete")
        # HW concatenates each rank's flattened H*W plane.  It deliberately
        # uses unequal H/W source shapes so an implementation that treats HW
        # as ordinary W concatenation produces a different ordering.
        axis_by_name = {"N": 0, "H": 1, "W": 2, "HW": 1, "C": 3}
        axis = axis_by_name[case.semantic_axis]
        left_shape = case.source_shape
        right_shape = case.source1_shape
        outer = _product(left_shape[:axis])
        left_chunk = _product(left_shape[axis:])
        right_chunk = _product(right_shape[axis:])
        left = _logical_values(_product(left_shape), seed)
        right = _logical_values(_product(right_shape), seed + 37)
        output: list[bytes] = []
        for index in range(outer):
            output.extend(
                left[index * left_chunk : (index + 1) * left_chunk]
            )
            output.extend(
                right[index * right_chunk : (index + 1) * right_chunk]
            )
        return b"".join(left + right), b"".join(output)

    if operation == "broadcast-row":
        values = _logical_values(53, seed)
        return b"".join(values), b"".join(values * 37)

    if operation == "broadcast-column":
        values = _logical_values(37, seed)
        output = tuple(value for value in values for _ in range(53))
        return b"".join(values), b"".join(output)

    if operation == "broadcast-scalar":
        value = _logical_values(1, seed)
        return b"".join(value), b"".join(
            value * _product(case.destination_shape)
        )

    if operation == "broadcast-channel":
        values = _logical_values(case.source_shape[-1], seed)
        repeats = _product(case.destination_shape[:-1])
        return b"".join(values), b"".join(values * repeats)

    if operation == "broadcast-row-large":
        values = _logical_values(_product(case.source_shape), seed)
        repeats = (
            _product(case.destination_shape)
            // _product(case.source_shape)
        )
        return b"".join(values), b"".join(values * repeats)

    if operation in (
        "tensor-to-cx",
        "cx-to-tensor",
        "tensor-to-ncx",
        "ncx-to-tensor",
    ):
        logical_shape = case.destination_shape
        logical = _logical_values(_product(logical_shape), seed)
        if operation == "tensor-to-cx":
            return b"".join(logical), codec.pack_scalar_bytes(
                logical_shape, "Cx", ELEMENT_BYTES, logical,
                padding=SLOT_CANARY,
            )
        if operation == "cx-to-tensor":
            return (
                codec.pack_scalar_bytes(
                    logical_shape, "Cx", ELEMENT_BYTES, logical,
                    padding=SLOT_CANARY,
                ),
                b"".join(logical),
            )
        if operation == "tensor-to-ncx":
            return b"".join(logical), codec.pack_scalar_bytes(
                logical_shape, "NCx", ELEMENT_BYTES, logical,
                padding=SLOT_CANARY,
            )
        return (
            codec.pack_scalar_bytes(
                logical_shape, "NCx", ELEMENT_BYTES, logical,
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

    if operation in ("gather-contiguous-large", "gather-tail-large"):
        values = _logical_values(_product(case.source_shape), seed)
        return b"".join(values), b"".join(values)

    if operation == "gather-1d-holes-large":
        values = _logical_values(_product(case.source_shape), seed)
        return b"".join(values), b"".join(values[::2])

    if operation == "gather-2d-holes-large":
        values = _logical_values(_product(case.source_shape), seed)
        output = tuple(
            values[row * 128 + column]
            for row in range(128)
            for column in range(64)
        )
        return b"".join(values), b"".join(output)

    if operation == "gather-3d-holes-large":
        logical_values = _logical_values(7 * 16 * 128, seed)
        source = bytearray([SLOT_CANARY] * case.input_bytes)
        output: list[bytes] = []
        logical_index = 0
        for plane in range(7):
            for row in range(16):
                row_base = plane * 8192 + row * 256
                row_values = logical_values[
                    logical_index : logical_index + 128
                ]
                logical_index += 128
                source[row_base : row_base + 256] = b"".join(row_values)
                output.extend(row_values[:64])
        return bytes(source), b"".join(output)

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
