#!/usr/bin/env python3
"""Exhaustive CT vector opcode/form/dtype calibration cases and host oracles."""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
import struct
from collections.abc import Callable, Iterable


REQUEST_MAGIC = 0x3151455256544357
RECORD_MAGIC = 0x3143455256544357
SCHEMA = 2
REQUEST_WORDS = 16
RECORD_WORDS = 32
CASE_BASE = 10000
CASE_STRIDE = 6
SPECIAL_CASE_BASE = 12000
RESOURCE_BYTES = 131072
SLOT_BYTES = 65536
BODY_OFFSET = 256
OUTPUT_DDR_OFFSET = 65536
SLOT_CANARY = 0xA7
REQUEST_GUARD = 0xC7B6A59483726150
RECORD_GUARD = 0x8F7E6D5C4B3A2910
MAIN_ELEMENTS = 8192
TAIL_ELEMENTS = 8197
UNIT_ELEMENTS = 37

DISPOSITIONS = {
    "BOARD_EXACT": 0,
    "BOARD_TOLERANCE": 1,
    "BOARD_OBSERVED": 2,
}
DTYPES = {"F16": 0, "BF16": 1, "F32": 2, "BOOL": 3}
FLOAT_DTYPES = ("F16", "BF16", "F32")
HARDWARE_FORMATS = {"F16": 2, "BF16": 3, "F32": 5, "BOOL": 7}
_CASE_DTYPE_ORDINAL = {"F16": 0, "BF16": 1, "F32": 2, "BOOL": 0}
FAMILIES = {
    "UNARY": 0,
    "BINARY": 1,
    "RELATION": 2,
    "LOGIC_VALUE": 3,
    "LOGIC_BOOL": 4,
    "TRANSCENDENTAL": 5,
    "ACTIVATION": 6,
}
DOMAINS = {
    "NORMAL": 0,
    "RECIP_ZERO": 1,
    "SQRT_ZERO": 2,
    "SQRT_NEGATIVE": 3,
    "RSQRT_ZERO": 4,
    "RSQRT_NEGATIVE": 5,
    "SPECIAL": 6,
    "QNAN": 7,
    "SNAN": 8,
    "SIGNED_ZERO": 9,
}
OUTPUT_ZERO_SIGN_POLICIES = {
    "AS_COMPUTED",
    "CANONICAL_POSITIVE",
}
REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "DISPOSITION": 3,
    "FAMILY": 4,
    "DTYPE": 5,
    "OPCODE": 6,
    "RESULT_BYTES": 7,
    "OUTPUT_SPAN": 8,
    "ELEMENTS": 9,
    "SAMPLE": 10,
    "RESOURCE_BYTES": 11,
    "SLOT_BYTES": 12,
    "BODY_OFFSET": 13,
    "DOMAIN": 14,
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
    "OPCODE": 7,
    "RESULT_BYTES": 8,
    "OUTPUT_SPAN": 9,
    "ELEMENTS": 10,
    "SAMPLE": 11,
    "OUTPUT_GUARD_MISMATCHES": 12,
    "EXECUTE_RESULT": 13,
    "REQUEST_GUARD": 14,
    "OUTPUT_DDR_OFFSET": 15,
    "SLOT_BYTES": 16,
    "BODY_OFFSET": 17,
    "INPUT_A_BYTES": 18,
    "INPUT_B_BYTES": 19,
    "SCALAR_BITS": 20,
    "UNIT_ELEMENTS": 21,
    "DOMAIN": 22,
    "RECORD_GUARD": 31,
}

_INSTR_DEF = (
    pathlib.Path(__file__).resolve().parents[2]
    / "third_party"
    / "tx8_deps"
    / "include"
    / "instr_def.h"
)
_OPCODE_ROW = re.compile(
    r"OP_FUNC_CGRATensor_([A-Za-z0-9_]+)\s*=\s*(\d+)"
)
_FORMAT_ROW = re.compile(
    r"\b(Fmt_(?:FP16|BF16|FP32|BOOL))\s*=\s*(\d+)"
)
_FORMAT_ENUM_NAMES = {
    "F16": "Fmt_FP16",
    "BF16": "Fmt_BF16",
    "F32": "Fmt_FP32",
    "BOOL": "Fmt_BOOL",
}


def load_opcode_inventory() -> tuple[str, ...]:
    rows = {
        int(match.group(2)): match.group(1)
        for match in _OPCODE_ROW.finditer(_INSTR_DEF.read_text())
    }
    if set(rows) != set(range(187)):
        missing = sorted(set(range(187)) - set(rows))
        extra = sorted(set(rows) - set(range(187)))
        raise RuntimeError(
            f"CT opcode inventory is not exactly 0..186: "
            f"missing={missing}, extra={extra}"
        )
    return tuple(rows[index] for index in range(187))


OPCODE_NAMES = load_opcode_inventory()


def validate_hardware_format_inventory() -> None:
    rows = {
        match.group(1): int(match.group(2))
        for match in _FORMAT_ROW.finditer(_INSTR_DEF.read_text())
    }
    actual = {
        dtype_name: rows.get(enum_name)
        for dtype_name, enum_name in _FORMAT_ENUM_NAMES.items()
    }
    if actual != HARDWARE_FORMATS:
        raise RuntimeError(
            "CT catalog Data_Format ids do not match instr_def.h: "
            f"expected={HARDWARE_FORMATS}, actual={actual}"
        )


validate_hardware_format_inventory()


def _family(opcode: int) -> str:
    if opcode <= 5:
        return "UNARY"
    if opcode <= 29:
        return "BINARY"
    if opcode <= 77:
        return "RELATION"
    if opcode <= 87:
        return "LOGIC_VALUE"
    if opcode <= 97:
        return "LOGIC_BOOL"
    if opcode <= 104:
        return "TRANSCENDENTAL"
    if opcode <= 110:
        return "ACTIVATION"
    raise ValueError(f"opcode {opcode} is not a CT vector entry")


def _is_bool_output(opcode: int) -> bool:
    return (
        30 <= opcode <= 77
        and (opcode - 30) % 8 in (1, 3, 6, 7)
    ) or 88 <= opcode <= 97


def _is_vs(opcode: int) -> bool:
    return (
        6 <= opcode <= 29
        and (opcode - 6) % 4 == 1
    ) or (
        30 <= opcode <= 77
        and (opcode - 30) % 8 in (2, 3)
    )


def _is_vuv(opcode: int) -> bool:
    return (
        6 <= opcode <= 29
        and (opcode - 6) % 4 in (2, 3)
    ) or (
        30 <= opcode <= 77
        and (opcode - 30) % 8 in (4, 5, 6, 7)
    ) or opcode in tuple(range(82, 88)) + tuple(range(92, 98))


def _is_loop(opcode: int) -> bool:
    return (
        6 <= opcode <= 29
        and (opcode - 6) % 4 == 3
    ) or (
        30 <= opcode <= 77
        and (opcode - 30) % 8 in (5, 7)
    ) or opcode in (85, 86, 87, 95, 96, 97)


def _disposition(opcode: int) -> str:
    if opcode in (108, 109):
        return "BOARD_OBSERVED"
    if opcode in (1, 3, 4) or 98 <= opcode <= 106 or opcode == 110:
        return "BOARD_TOLERANCE"
    return "BOARD_EXACT"


def _dtype_size(dtype_name: str) -> int:
    if dtype_name == "F32":
        return 4
    if dtype_name in ("F16", "BF16"):
        return 2
    raise ValueError(f"{dtype_name} is not a value dtype")


def _case_id(opcode: int, dtype_name: str, shape_name: str) -> int:
    return (
        CASE_BASE
        + opcode * CASE_STRIDE
        + _CASE_DTYPE_ORDINAL[dtype_name] * 2
        + (shape_name == "tail")
    )


_SPECIAL_CASES = (
    (1, "RECIP_ZERO"),
    (3, "SQRT_ZERO"),
    (3, "SQRT_NEGATIVE"),
    (4, "RSQRT_ZERO"),
    (4, "RSQRT_NEGATIVE"),
    (14, "SPECIAL"),
    (14, "QNAN"),
    (14, "SNAN"),
    (5, "SIGNED_ZERO"),
)


def _special_case_id(
    opcode: int, dtype_name: str, domain_name: str
) -> int:
    ordinal = _SPECIAL_CASES.index((opcode, domain_name))
    return (
        SPECIAL_CASE_BASE
        + ordinal * len(FLOAT_DTYPES)
        + FLOAT_DTYPES.index(dtype_name)
    )


def _domain_disposition(opcode: int, domain_name: str) -> str:
    if domain_name == "NORMAL":
        return _disposition(opcode)
    if domain_name in ("SQRT_ZERO", "SPECIAL", "SIGNED_ZERO"):
        return "BOARD_TOLERANCE" if opcode in (3, 4) else "BOARD_EXACT"
    return "BOARD_OBSERVED"


def _output_zero_sign_policy(opcode: int, dtype_name: str) -> str:
    # The TX81 F16 and BF16 Neg entries canonicalize a zero result to +0.
    # Keep this per-opcode and per-dtype: the calibration contract does not
    # infer F32 behavior from the F16/BF16 board observations.
    if opcode == 5 and dtype_name in {"F16", "BF16"}:
        return "CANONICAL_POSITIVE"
    return "AS_COMPUTED"


@dataclasses.dataclass(frozen=True)
class CTVectorCase:
    case_id: int
    name: str
    opcode: int
    opcode_name: str
    family_name: str
    dtype_name: str
    disposition_name: str
    domain_name: str
    output_zero_sign_policy_name: str
    shape_name: str
    elements: int
    result_bytes: int
    output_span: int

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
    def domain(self) -> int:
        return DOMAINS[self.domain_name]

    @property
    def is_safe(self) -> bool:
        return True

    @property
    def exact(self) -> bool:
        return self.disposition_name == "BOARD_EXACT"

    @property
    def tolerance(self) -> bool:
        return self.disposition_name == "BOARD_TOLERANCE"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "opcode": self.opcode,
            "opcode_name": self.opcode_name,
            "family": self.family_name.lower(),
            "dtype": self.dtype_name.lower(),
            "form": form_name(self.opcode),
            "output_storage": (
                "bitpacked-bool"
                if _is_bool_output(self.opcode)
                else "value"
            ),
            "disposition": self.disposition_name.lower(),
            "numeric_domain": self.domain_name.lower().replace("_", "-"),
            "output_zero_sign_policy": (
                self.output_zero_sign_policy_name.lower().replace("_", "-")
            ),
            "shape": self.shape_name,
            "elements": self.elements,
            "unit_elements": UNIT_ELEMENTS if _is_vuv(self.opcode) else 0,
            "result_bytes": self.result_bytes,
            "output_span": self.output_span,
        }


def form_name(opcode: int) -> str:
    if opcode <= 5 or 78 == opcode or 88 == opcode or 98 <= opcode <= 110:
        return "V"
    if 6 <= opcode <= 29:
        return ("VV", "VS", "VuV", "VuVLoop")[(opcode - 6) % 4]
    if 30 <= opcode <= 77:
        return (
            "VV",
            "VV",
            "VS",
            "VS",
            "VuV",
            "VuVLoop",
            "VuV",
            "VuVLoop",
        )[(opcode - 30) % 8]
    if 79 <= opcode <= 87 or 89 <= opcode <= 97:
        position = (opcode - (79 if opcode <= 87 else 89)) % 10
        return (
            "VV"
            if position <= 2
            else "VuV" if position <= 5 else "VuVLoop"
        )
    raise ValueError(f"opcode {opcode} has no vector form")


def build_catalog() -> tuple[CTVectorCase, ...]:
    cases: list[CTVectorCase] = []
    for opcode in range(111):
        dtype_names = ("BOOL",) if 88 <= opcode <= 97 else FLOAT_DTYPES
        for dtype_name in dtype_names:
            for shape_name, elements in (
                ("main", MAIN_ELEMENTS),
                ("tail", TAIL_ELEMENTS),
            ):
                result_bytes = (
                    (elements + 7) // 8
                    if _is_bool_output(opcode)
                    else elements * _dtype_size(dtype_name)
                )
                output_span = (result_bytes + 255) // 256 * 256
                cases.append(
                    CTVectorCase(
                        case_id=_case_id(opcode, dtype_name, shape_name),
                        name=(
                            f"ct-op{opcode:03d}-"
                            f"{OPCODE_NAMES[opcode].lower()}-"
                            f"{dtype_name.lower()}-{shape_name}"
                        ),
                        opcode=opcode,
                        opcode_name=OPCODE_NAMES[opcode],
                        family_name=_family(opcode),
                        dtype_name=dtype_name,
                        disposition_name=_disposition(opcode),
                        domain_name="NORMAL",
                        output_zero_sign_policy_name=(
                            _output_zero_sign_policy(opcode, dtype_name)
                        ),
                        shape_name=shape_name,
                        elements=elements,
                        result_bytes=result_bytes,
                        output_span=output_span,
                    )
                )
    for opcode, domain_name in _SPECIAL_CASES:
        for dtype_name in FLOAT_DTYPES:
            elements = MAIN_ELEMENTS
            result_bytes = elements * _dtype_size(dtype_name)
            cases.append(
                CTVectorCase(
                    case_id=_special_case_id(
                        opcode, dtype_name, domain_name
                    ),
                    name=(
                        f"ct-op{opcode:03d}-"
                        f"{OPCODE_NAMES[opcode].lower()}-"
                        f"{dtype_name.lower()}-"
                        f"{domain_name.lower().replace('_', '-')}"
                    ),
                    opcode=opcode,
                    opcode_name=OPCODE_NAMES[opcode],
                    family_name=_family(opcode),
                    dtype_name=dtype_name,
                    disposition_name=_domain_disposition(
                        opcode, domain_name
                    ),
                    domain_name=domain_name,
                    output_zero_sign_policy_name=(
                        _output_zero_sign_policy(opcode, dtype_name)
                    ),
                    shape_name="main",
                    elements=elements,
                    result_bytes=result_bytes,
                    output_span=(result_bytes + 255) // 256 * 256,
                )
            )
    ids = {case.case_id for case in cases}
    names = {case.name for case in cases}
    if len(ids) != len(cases) or len(names) != len(cases):
        raise RuntimeError("CT vector catalog has duplicate case keys")
    if any(
        case.output_span > SLOT_BYTES - BODY_OFFSET for case in cases
    ):
        raise RuntimeError("CT vector output does not fit its guarded slot")
    return tuple(cases)


CATALOG = build_catalog()
CASES_BY_NAME = {case.name: case for case in CATALOG}
SAFE_CASES = CATALOG
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "ct-vector-forms-main-tail": tuple(
        case
        for case in CATALOG
        if case.domain_name == "NORMAL" and case.family_name == "BINARY"
    ),
    "ct-finite-boundaries-all-float-dtypes": tuple(
        case
        for case in CATALOG
        if case.domain_name == "NORMAL" and case.family_name == "UNARY"
    ),
    "ct-zero-domain-positive": tuple(
        case
        for case in CATALOG
        if case.domain_name
        in {
            "RECIP_ZERO",
            "SQRT_ZERO",
            "SQRT_NEGATIVE",
            "RSQRT_ZERO",
            "RSQRT_NEGATIVE",
        }
        and case.disposition_name != "BOARD_OBSERVED"
    ),
    "ct-zero-domain-observed": tuple(
        case
        for case in CATALOG
        if case.domain_name
        in {
            "RECIP_ZERO",
            "SQRT_ZERO",
            "SQRT_NEGATIVE",
            "RSQRT_ZERO",
            "RSQRT_NEGATIVE",
        }
        and case.disposition_name == "BOARD_OBSERVED"
    ),
    "ct-special-values-positive": tuple(
        case
        for case in CATALOG
        if case.domain_name in {"SPECIAL", "QNAN", "SNAN", "SIGNED_ZERO"}
        and case.disposition_name != "BOARD_OBSERVED"
    ),
    "ct-special-values-observed": tuple(
        case
        for case in CATALOG
        if case.domain_name in {"SPECIAL", "QNAN", "SNAN", "SIGNED_ZERO"}
        and case.disposition_name == "BOARD_OBSERVED"
    ),
    "ct-relation-logic-value-bool": tuple(
        case
        for case in CATALOG
        if case.domain_name == "NORMAL"
        and case.family_name
        in {"RELATION", "LOGIC_VALUE", "LOGIC_BOOL"}
    ),
    "ct-transcendental-activation": tuple(
        case
        for case in CATALOG
        if case.domain_name == "NORMAL"
        and (
            case.family_name == "TRANSCENDENTAL"
            or (
                case.family_name == "ACTIVATION"
                and case.opcode not in {108, 109}
            )
        )
    ),
    "ct-observed-activation-semantics": tuple(
        case
        for case in CATALOG
        if case.domain_name == "NORMAL" and case.opcode in {108, 109}
    ),
}


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_result: bytes | None
    expected_output_slot: bytes
    input_a_bytes: int
    input_b_bytes: int
    scalar_bits: int


def _bf16_word(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _encode(dtype_name: str, values: Iterable[float]) -> bytes:
    values = tuple(values)
    if dtype_name == "F16":
        return struct.pack(f"<{len(values)}e", *values)
    if dtype_name == "BF16":
        return struct.pack(
            f"<{len(values)}H", *(_bf16_word(value) for value in values)
        )
    if dtype_name == "F32":
        return struct.pack(f"<{len(values)}f", *values)
    raise ValueError(f"unknown CT vector dtype {dtype_name}")


def _pack_words(dtype_name: str, words: Iterable[int]) -> bytes:
    values = tuple(words)
    code = "I" if dtype_name == "F32" else "H"
    return struct.pack(f"<{len(values)}{code}", *values)


def _special_words(dtype_name: str, domain_name: str) -> tuple[int, ...]:
    if dtype_name == "F16":
        finite = (
            0x0000,
            0x8000,
            0x7C00,
            0xFC00,
            0x7BFF,
            0xFBFF,
            0x0400,
            0x8400,
            0x0001,
            0x8001,
        )
        qnan = (0x7E01, 0xFE55)
        snan = (0x7D01, 0xFD55)
    elif dtype_name == "BF16":
        finite = (
            0x0000,
            0x8000,
            0x7F80,
            0xFF80,
            0x7F7F,
            0xFF7F,
            0x0080,
            0x8080,
            0x0001,
            0x8001,
        )
        qnan = (0x7FC1, 0xFFC5)
        snan = (0x7F81, 0xFF85)
    else:
        finite = (
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x00800000,
            0x80800000,
            0x00000001,
            0x80000001,
        )
        qnan = (0x7FC01234, 0xFFC05678)
        snan = (0x7F801234, 0xFF805678)
    if domain_name == "SPECIAL":
        return finite
    if domain_name == "QNAN":
        return qnan
    if domain_name == "SNAN":
        return snan
    raise ValueError(f"{domain_name} is not a raw special-value domain")


def _repeat_words(words: Iterable[int], count: int) -> tuple[int, ...]:
    source = tuple(words)
    return tuple(source[index % len(source)] for index in range(count))


def decode_values(dtype_name: str, raw: bytes) -> tuple[float, ...]:
    if dtype_name == "F16":
        return struct.unpack(f"<{len(raw) // 2}e", raw)
    if dtype_name == "BF16":
        words = struct.unpack(f"<{len(raw) // 2}H", raw)
        return tuple(
            struct.unpack("<f", struct.pack("<I", word << 16))[0]
            for word in words
        )
    if dtype_name == "F32":
        return struct.unpack(f"<{len(raw) // 4}f", raw)
    raise ValueError(f"unknown CT vector dtype {dtype_name}")


def _repeat(values: Iterable[float], count: int) -> tuple[float, ...]:
    source = tuple(values)
    return tuple(source[index % len(source)] for index in range(count))


def _numeric_inputs(case: CTVectorCase) -> tuple[bytes, bytes, int]:
    if case.domain_name in ("SPECIAL", "QNAN", "SNAN"):
        lhs = _pack_words(
            case.dtype_name,
            _repeat_words(
                _special_words(case.dtype_name, case.domain_name),
                case.elements,
            ),
        )
        zero = 0
        rhs = _pack_words(
            case.dtype_name, _repeat_words((zero,), case.elements)
        )
        return lhs, rhs, 0
    if case.domain_name == "SIGNED_ZERO":
        return (
            _pack_words(
                case.dtype_name,
                _repeat_words(
                    (
                        0,
                        0x80000000
                        if case.dtype_name == "F32"
                        else 0x8000,
                    ),
                    case.elements,
                ),
            ),
            b"",
            0,
        )
    if case.domain_name in ("RECIP_ZERO", "SQRT_ZERO", "RSQRT_ZERO"):
        return (
            _pack_words(
                case.dtype_name,
                _repeat_words(
                    (
                        0,
                        0x80000000
                        if case.dtype_name == "F32"
                        else 0x8000,
                    ),
                    case.elements,
                ),
            ),
            b"",
            0,
        )
    if case.domain_name in ("SQRT_NEGATIVE", "RSQRT_NEGATIVE"):
        lhs = _repeat((-1.0, -4.0, -9.0, -16.0), case.elements)
        return _encode(case.dtype_name, lhs), b"", 0
    operation = case.opcode_name.rsplit("_", 1)[-1].removesuffix("_loop")
    if case.family_name == "UNARY":
        patterns = {
            "abs": (-8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0),
            "recip": (-8.0, -4.0, -2.0, -0.5, 0.5, 2.0, 4.0, 8.0),
            "square": (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
            "sqrt": (0.0, 0.25, 1.0, 4.0, 9.0, 16.0, 25.0, 64.0),
            "rsqrt": (0.25, 1.0, 4.0, 16.0, 64.0),
            "neg": (-8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0),
        }
        lhs = _repeat(patterns[operation], case.elements)
        return _encode(case.dtype_name, lhs), b"", 0
    if case.family_name in ("TRANSCENDENTAL", "ACTIVATION"):
        patterns = {
            "log2": (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            "ln": (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            "pow2": (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
            "exp": (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
            "lp": (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
            "sin": (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0),
            "cos": (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0),
            "tanh": (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0),
            "sigmoid": (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0),
            "relu": (-8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0),
            "satrelu": (-8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0),
            "leakyrelu": (-8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0),
            "softplus": (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0),
        }
        key = "lp" if case.opcode == 102 else operation
        lhs = _repeat(patterns[key], case.elements)
        return _encode(case.dtype_name, lhs), b"", 0

    lhs_values = _repeat(
        (-8.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 8.0),
        case.elements,
    )
    rhs_count = UNIT_ELEMENTS if _is_vuv(case.opcode) else case.elements
    if (
        case.family_name == "RELATION"
        and 30 <= case.opcode <= 45
        and form_name(case.opcode) == "VV"
    ):
        rhs_values = tuple(
            value if index % 2 == 0 else value + 0.5
            for index, value in enumerate(lhs_values)
        )
    else:
        rhs_values = _repeat(
            (2.0, -2.0, 4.0, 1.0, -1.0, 0.5, 8.0, -4.0),
            rhs_count,
        )
    scalar_bits = {
        "F16": 0x4000,
        "BF16": 0x4000,
        "F32": 0x40000000,
    }[case.dtype_name]
    return (
        _encode(case.dtype_name, lhs_values),
        b"" if _is_vs(case.opcode) else _encode(case.dtype_name, rhs_values),
        scalar_bits if _is_vs(case.opcode) else 0,
    )


def _logic_inputs(case: CTVectorCase) -> tuple[bytes, bytes, int]:
    if case.family_name == "LOGIC_BOOL":
        byte_count = (case.elements + 7) // 8
        lhs = bytes((0x96, 0x5A, 0xF0, 0x0F)[index % 4] for index in range(byte_count))
        rhs_count = (
            (UNIT_ELEMENTS + 7) // 8
            if _is_vuv(case.opcode)
            else byte_count
        )
        rhs = bytes((0x69, 0xC3, 0xAA, 0x55)[index % 4] for index in range(rhs_count))
        return lhs, rhs if case.opcode != 88 else b"", 0

    word_bytes = _dtype_size(case.dtype_name)
    mask = (1 << (word_bytes * 8)) - 1
    lhs_words = tuple(
        (0x1357_9BDF ^ (index * 0x1021)) & mask
        for index in range(case.elements)
    )
    rhs_count = UNIT_ELEMENTS if _is_vuv(case.opcode) else case.elements
    rhs_words = tuple(
        (0x2468_ACE0 ^ (index * 0x2043)) & mask
        for index in range(rhs_count)
    )
    code = {2: "H", 4: "I"}[word_bytes]
    lhs = struct.pack(f"<{len(lhs_words)}{code}", *lhs_words)
    rhs = struct.pack(f"<{len(rhs_words)}{code}", *rhs_words)
    return lhs, rhs if case.opcode != 78 else b"", 0


def _numeric_expected(
    case: CTVectorCase, input_a: bytes, input_b: bytes, scalar_bits: int
) -> bytes | None:
    if case.disposition_name == "BOARD_OBSERVED":
        return None
    lhs = decode_values(case.dtype_name, input_a)
    if _is_vs(case.opcode):
        scalar_raw = scalar_bits.to_bytes(_dtype_size(case.dtype_name), "little")
        rhs_source = decode_values(case.dtype_name, scalar_raw)
    elif input_b:
        rhs_source = decode_values(case.dtype_name, input_b)
    else:
        rhs_source = ()
    rhs = tuple(
        rhs_source[index % len(rhs_source)] for index in range(case.elements)
    ) if rhs_source else ()

    if case.family_name == "UNARY":
        operations: tuple[Callable[[float], float], ...] = (
            abs,
            lambda value: 1.0 / value,
            lambda value: value * value,
            math.sqrt,
            lambda value: 1.0 / math.sqrt(value),
            lambda value: -value,
        )
        values = tuple(operations[case.opcode](value) for value in lhs)
    elif case.family_name == "BINARY":
        operation = (case.opcode - 6) // 4
        functions: tuple[Callable[[float, float], float], ...] = (
            max,
            min,
            lambda left, right: left + right,
            lambda left, right: left - right,
            lambda left, right: left * right,
            lambda left, right: left / right,
        )
        values = tuple(
            functions[operation](left, right)
            for left, right in zip(lhs, rhs, strict=True)
        )
    elif case.family_name == "RELATION":
        operation = (case.opcode - 30) // 8
        predicates: tuple[Callable[[float, float], bool], ...] = (
            lambda left, right: left == right,
            lambda left, right: left != right,
            lambda left, right: left >= right,
            lambda left, right: left > right,
            lambda left, right: left <= right,
            lambda left, right: left < right,
        )
        truth = tuple(
            predicates[operation](left, right)
            for left, right in zip(lhs, rhs, strict=True)
        )
        if _is_bool_output(case.opcode):
            packed = bytearray((case.elements + 7) // 8)
            for index, value in enumerate(truth):
                packed[index // 8] |= int(value) << (index % 8)
            return bytes(packed)
        values = tuple(float(value) for value in truth)
    elif case.family_name == "TRANSCENDENTAL":
        functions = (
            math.log2,
            math.log,
            lambda value: 2.0**value,
            math.exp,
            math.exp,
            math.sin,
            math.cos,
        )
        values = tuple(
            functions[case.opcode - 98](value) for value in lhs
        )
    elif case.family_name == "ACTIVATION":
        functions: tuple[Callable[[float], float] | None, ...] = (
            math.tanh,
            lambda value: 1.0 / (1.0 + math.exp(-value)),
            lambda value: max(value, 0.0),
            None,
            None,
            lambda value: math.log1p(math.exp(value)),
        )
        function = functions[case.opcode - 105]
        if function is None:
            return None
        values = tuple(function(value) for value in lhs)
    else:
        raise RuntimeError(f"{case.name}: numeric oracle family is invalid")
    if case.output_zero_sign_policy_name == "CANONICAL_POSITIVE":
        values = tuple(0.0 if value == 0.0 else value for value in values)
    return _encode(case.dtype_name, values)


def _logic_expected(
    case: CTVectorCase, input_a: bytes, input_b: bytes
) -> bytes:
    if case.family_name == "LOGIC_BOOL":
        result = bytearray((case.elements + 7) // 8)
        functions = (
            lambda left, right: left & right,
            lambda left, right: left | right,
            lambda left, right: left ^ right,
        )
        for index in range(case.elements):
            left = (input_a[index // 8] >> (index % 8)) & 1
            if case.opcode == 88:
                value = left ^ 1
            else:
                right_index = (
                    index % UNIT_ELEMENTS
                    if _is_vuv(case.opcode)
                    else index
                )
                right = (
                    input_b[right_index // 8] >> (right_index % 8)
                ) & 1
                value = functions[(case.opcode - 89) % 3](left, right)
            result[index // 8] |= value << (index % 8)
        return bytes(result)

    word_bytes = _dtype_size(case.dtype_name)
    code = {2: "H", 4: "I"}[word_bytes]
    lhs = struct.unpack(f"<{len(input_a) // word_bytes}{code}", input_a)
    if case.opcode == 78:
        mask = (1 << (word_bytes * 8)) - 1
        values = tuple(value ^ mask for value in lhs)
    else:
        rhs_words = struct.unpack(
            f"<{len(input_b) // word_bytes}{code}", input_b
        )
        operation = (case.opcode - 79) % 3
        functions = (
            lambda left, right: left & right,
            lambda left, right: left | right,
            lambda left, right: left ^ right,
        )
        values = tuple(
            functions[operation](
                left,
                rhs_words[index % len(rhs_words)],
            )
            for index, left in enumerate(lhs)
        )
    return struct.pack(f"<{len(values)}{code}", *values)


def build_case_payload(case: CTVectorCase, sample: int = 0) -> CasePayload:
    if case.family_name.startswith("LOGIC"):
        input_a, input_b, scalar_bits = _logic_inputs(case)
        expected = _logic_expected(case, input_a, input_b)
    else:
        input_a, input_b, scalar_bits = _numeric_inputs(case)
        expected = _numeric_expected(case, input_a, input_b, scalar_bits)
    if len(input_a) > SLOT_BYTES or len(input_b) > SLOT_BYTES:
        raise RuntimeError(f"{case.name}: input exceeds one payload slot")
    if expected is not None and len(expected) != case.result_bytes:
        raise RuntimeError(f"{case.name}: host oracle has the wrong byte size")

    payload = bytearray([SLOT_CANARY] * RESOURCE_BYTES)
    payload[: len(input_a)] = input_a
    payload[SLOT_BYTES : SLOT_BYTES + len(input_b)] = input_b
    expected_slot = bytearray([SLOT_CANARY] * SLOT_BYTES)
    if expected is not None:
        begin = BODY_OFFSET
        expected_slot[begin : begin + len(expected)] = expected

    request_words = [0] * REQUEST_WORDS
    request_words[REQ["MAGIC"]] = REQUEST_MAGIC
    request_words[REQ["SCHEMA_AND_WORDS"]] = (
        SCHEMA << 32
    ) | REQUEST_WORDS
    request_words[REQ["CASE"]] = case.case_id
    request_words[REQ["DISPOSITION"]] = case.disposition
    request_words[REQ["FAMILY"]] = case.family
    request_words[REQ["DTYPE"]] = case.dtype
    request_words[REQ["OPCODE"]] = case.opcode
    request_words[REQ["RESULT_BYTES"]] = case.result_bytes
    request_words[REQ["OUTPUT_SPAN"]] = case.output_span
    request_words[REQ["ELEMENTS"]] = case.elements
    request_words[REQ["SAMPLE"]] = sample
    request_words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    request_words[REQ["SLOT_BYTES"]] = SLOT_BYTES
    request_words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    request_words[REQ["DOMAIN"]] = case.domain
    request_words[REQ["GUARD"]] = REQUEST_GUARD
    request = bytearray([0xD3] * RESOURCE_BYTES)
    struct.pack_into(f"<{REQUEST_WORDS}Q", request, 0, *request_words)
    return CasePayload(
        request=bytes(request),
        payload=bytes(payload),
        expected_result=expected,
        expected_output_slot=bytes(expected_slot),
        input_a_bytes=len(input_a),
        input_b_bytes=len(input_b),
        scalar_bits=scalar_bits,
    )


def tolerance_for(case: CTVectorCase) -> tuple[float, float]:
    if case.dtype_name == "F16":
        return (2.0e-2, 2.0e-2)
    if case.dtype_name == "BF16":
        return (8.0e-2, 8.0e-2)
    return (2.0e-5, 2.0e-5)
