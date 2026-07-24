#!/usr/bin/env python3
"""Isolated supported-contract control for CT VuVLoop."""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Callable


REQUEST_MAGIC = 0x31514552564C5657
RECORD_MAGIC = 0x31434552564C5657
SCHEMA = 1
REQUEST_WORDS = 16
RECORD_WORDS = 24
CASE_BASE = 20000
OPCODE = 17
DTYPE = 0
ELEMENT_BYTES = 2
PHYSICAL_BLOCK_BYTES = 256
RESOURCE_BYTES = 8192
LHS_OWNED_BYTES = 1024
RHS_OWNED_BYTES = 512
OUTPUT_CAPTURE_BYTES = 1024
GUARD_BYTES = 256
GUARD_BYTE = 0xA7
REQUEST_PADDING_BYTE = 0xC9
REQUEST_GUARD = 0xC14D2E3F50617283
RECORD_GUARD = 0x8A79685746352413

EXACT_CONTROL = 0
BOUNDED_OBSERVATION = 1
DISPOSITIONS = {
    "exact-control": EXACT_CONTROL,
    "bounded-observation": BOUNDED_OBSERVATION,
}

LHS_SLOT_BYTES = 2 * GUARD_BYTES + LHS_OWNED_BYTES
RHS_SLOT_BYTES = 2 * GUARD_BYTES + RHS_OWNED_BYTES
OUTPUT_SLOT_BYTES = 2 * GUARD_BYTES + OUTPUT_CAPTURE_BYTES
PAYLOAD_LHS_OFFSET = 0
PAYLOAD_RHS_OFFSET = PAYLOAD_LHS_OFFSET + LHS_SLOT_BYTES
PAYLOAD_OUTPUT_SEED_OFFSET = PAYLOAD_RHS_OFFSET + RHS_SLOT_BYTES
RECORD_RESERVED_BYTES = 512
OUTPUT_DDR_OFFSET = RECORD_RESERVED_BYTES
BODY_OFFSET = GUARD_BYTES

SPM_LHS = 0x10000
SPM_RHS = 0x30000
SPM_OUTPUT = 0x50000

REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "DISPOSITION": 3,
    "OPCODE": 4,
    "DTYPE": 5,
    "ELEM_COUNT": 6,
    "UNIT_ELEM_COUNT": 7,
    "FULL_ELEM_COUNT": 8,
    "FULL_UNIT_ELEM_COUNT": 9,
    "LHS_OWNED_BYTES": 10,
    "RHS_OWNED_BYTES": 11,
    "OUTPUT_CAPTURE_BYTES": 12,
    "GUARD_BYTES": 13,
    "SAMPLE": 14,
    "GUARD": 15,
}

REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "DISPOSITION": 4,
    "OPCODE": 5,
    "DTYPE": 6,
    "ELEM_COUNT": 7,
    "UNIT_ELEM_COUNT": 8,
    "FULL_ELEM_COUNT": 9,
    "FULL_UNIT_ELEM_COUNT": 10,
    "EXECUTE_RESULT": 11,
    "COMPLETION_SEEN": 12,
    "REQUEST_ECHO_MISMATCHES": 13,
    "SAMPLE": 14,
    "REQUEST_GUARD": 15,
    "OUTPUT_DDR_OFFSET": 16,
    "OUTPUT_SLOT_BYTES": 17,
    "BODY_OFFSET": 18,
    "OUTPUT_OWNED_BYTES": 19,
    "OUTPUT_PHYSICAL_SPAN": 20,
    "RESOURCE_BYTES": 21,
    "TERMINAL_FENCE_COUNT": 22,
    "RECORD_GUARD": 23,
}

OBSERVATION_CONTRACT = (
    "request-echo",
    "execute-result",
    "matching-completion",
    "full-owned-output-capture",
    "output-guards",
    "single-terminal-fence",
)


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclasses.dataclass(frozen=True)
class VuVLoopGeometry:
    elem_count: int
    unit_elem_count: int
    full_elem_count: int
    full_unit_elem_count: int

    @property
    def full_elem_divmod(self) -> tuple[int, int]:
        return divmod(self.full_elem_count, self.elem_count)

    @property
    def full_unit_divmod(self) -> tuple[int, int]:
        return divmod(self.full_unit_elem_count, self.unit_elem_count)

    @property
    def elem_unit_remainder(self) -> int:
        return self.elem_count % self.unit_elem_count

    @property
    def boundary_tags(self) -> frozenset[str]:
        tags: set[str] = set()
        elem_outer, elem_tail = self.full_elem_divmod
        unit_outer, unit_tail = self.full_unit_divmod
        if elem_tail:
            tags.add("partial-full-elem")
        if unit_tail:
            tags.add("partial-full-unit")
        if not elem_tail and not unit_tail and elem_outer != unit_outer:
            tags.add("unequal-outer-quotient")
        if self.elem_unit_remainder:
            tags.add("elem-unit-remainder")
        if not tags:
            tags.add("rectangular-proportional")
        return frozenset(tags)

    @property
    def satisfies_existing_catalog_gate(self) -> bool:
        return (
            self.unit_elem_count == 64
            and self.elem_count > 0
            and self.elem_unit_remainder == 0
            and self.full_elem_count > self.elem_count
            and self.full_unit_elem_count > self.unit_elem_count
            and self.full_elem_count * self.unit_elem_count
            == self.elem_count * self.full_unit_elem_count
        )

    @property
    def lhs_declared_bytes(self) -> int:
        return self.full_elem_count * ELEMENT_BYTES

    @property
    def rhs_declared_bytes(self) -> int:
        return self.full_unit_elem_count * ELEMENT_BYTES

    @property
    def output_declared_bytes(self) -> int:
        return self.full_elem_count * ELEMENT_BYTES

    @property
    def output_physical_span(self) -> int:
        return _round_up(
            self.output_declared_bytes, PHYSICAL_BLOCK_BYTES
        )


@dataclasses.dataclass(frozen=True)
class VuVLoopSemanticCase:
    case_id: int
    name: str
    distinction: str
    geometry: VuVLoopGeometry
    question: str
    candidate_models: tuple[str, ...]
    disposition: str = "bounded-observation"
    evidence_maturity: str = "unknown"
    exact_legality: bool = False
    expected_result: bytes | None = None
    observation_contract: tuple[str, ...] = OBSERVATION_CONTRACT

    def request_words(self, sample: int = 0) -> tuple[int, ...]:
        words = [0] * REQUEST_WORDS
        values = {
            "MAGIC": REQUEST_MAGIC,
            "SCHEMA_AND_WORDS": (SCHEMA << 32) | REQUEST_WORDS,
            "CASE": self.case_id,
            "DISPOSITION": DISPOSITIONS[self.disposition],
            "OPCODE": OPCODE,
            "DTYPE": DTYPE,
            "ELEM_COUNT": self.geometry.elem_count,
            "UNIT_ELEM_COUNT": self.geometry.unit_elem_count,
            "FULL_ELEM_COUNT": self.geometry.full_elem_count,
            "FULL_UNIT_ELEM_COUNT": self.geometry.full_unit_elem_count,
            "LHS_OWNED_BYTES": LHS_OWNED_BYTES,
            "RHS_OWNED_BYTES": RHS_OWNED_BYTES,
            "OUTPUT_CAPTURE_BYTES": OUTPUT_CAPTURE_BYTES,
            "GUARD_BYTES": GUARD_BYTES,
            "SAMPLE": sample,
            "GUARD": REQUEST_GUARD,
        }
        for field, value in values.items():
            words[REQ[field]] = value
        return tuple(words)

    def encode_request(self, sample: int = 0) -> bytes:
        return struct.pack(
            f"<{REQUEST_WORDS}Q", *self.request_words(sample)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "distinction": self.distinction,
            "disposition": self.disposition,
            "evidence_maturity": self.evidence_maturity,
            "exact_legality": self.exact_legality,
            "expected_result_bytes": (
                len(self.expected_result)
                if self.expected_result is not None
                else None
            ),
            "elem_count": self.geometry.elem_count,
            "unit_elem_count": self.geometry.unit_elem_count,
            "full_elem_count": self.geometry.full_elem_count,
            "full_unit_elem_count": self.geometry.full_unit_elem_count,
            "boundary_tags": tuple(sorted(self.geometry.boundary_tags)),
            "question": self.question,
            "candidate_models": self.candidate_models,
            "observation_contract": self.observation_contract,
        }


@dataclasses.dataclass(frozen=True)
class GuardedRegion:
    body: bytes
    prefix_guard: bytes
    suffix_guard: bytes

    @property
    def encoded(self) -> bytes:
        return self.prefix_guard + self.body + self.suffix_guard


@dataclasses.dataclass(frozen=True)
class VuVLoopObservationPayload:
    request: bytes
    lhs: GuardedRegion
    rhs: GuardedRegion
    output: GuardedRegion
    expected_result: bytes | None = None

    @property
    def request_resource(self) -> bytes:
        resource = bytearray([REQUEST_PADDING_BYTE]) * RESOURCE_BYTES
        resource[: len(self.request)] = self.request
        return bytes(resource)

    @property
    def payload_resource(self) -> bytes:
        resource = bytearray([REQUEST_PADDING_BYTE]) * RESOURCE_BYTES
        for offset, encoded in (
            (PAYLOAD_LHS_OFFSET, self.lhs.encoded),
            (PAYLOAD_RHS_OFFSET, self.rhs.encoded),
            (PAYLOAD_OUTPUT_SEED_OFFSET, self.output.encoded),
        ):
            resource[offset : offset + len(encoded)] = encoded
        return bytes(resource)


@dataclasses.dataclass(frozen=True)
class ExistingRectangularReference:
    case_name: str
    geometry: VuVLoopGeometry
    role: str


EXISTING_RECTANGULAR_REFERENCE = ExistingRectangularReference(
    case_name="ct-op017-arithop_v_vuv_add_loop-f16-main",
    geometry=VuVLoopGeometry(
        elem_count=256,
        unit_elem_count=64,
        full_elem_count=8192,
        full_unit_elem_count=2048,
    ),
    role=(
        "read-only reference to the existing 653-row catalog; it does not "
        "qualify any non-rectangular geometry"
    ),
)

STATIC_NEGATIVE_GEOMETRIES = {
    "unit-not-64": VuVLoopGeometry(128, 32, 384, 96),
    "ratio-mismatch": VuVLoopGeometry(128, 64, 384, 128),
    "elem-unit-remainder": VuVLoopGeometry(222, 64, 8214, 2368),
}


def _case(
    ordinal: int,
    name: str,
    distinction: str,
    geometry: VuVLoopGeometry,
    question: str,
    *candidate_models: str,
    disposition: str = "bounded-observation",
    evidence_maturity: str = "unknown",
    exact_legality: bool = False,
    expected_result: bytes | None = None,
) -> VuVLoopSemanticCase:
    return VuVLoopSemanticCase(
        case_id=CASE_BASE + ordinal,
        name=name,
        distinction=distinction,
        geometry=geometry,
        question=question,
        candidate_models=tuple(candidate_models),
        disposition=disposition,
        evidence_maturity=evidence_maturity,
        exact_legality=exact_legality,
        expected_result=expected_result,
    )


def _lhs_value(index: int) -> float:
    return float((index % 23) - 11)


def _rhs_value(index: int, geometry: VuVLoopGeometry) -> float:
    return float(
        4 * (index // geometry.unit_elem_count + 1) + index % 4
    )


def _rectangular_expected(geometry: VuVLoopGeometry) -> bytes:
    if not geometry.satisfies_existing_catalog_gate:
        raise RuntimeError("exact VuVLoop control must be rectangular")
    values = []
    for index in range(geometry.full_elem_count):
        outer = index // geometry.elem_count
        within_base = index % geometry.elem_count
        rhs_index = (
            outer * geometry.unit_elem_count
            + within_base % geometry.unit_elem_count
        )
        values.append(_lhs_value(index) + _rhs_value(rhs_index, geometry))
    return b"".join(struct.pack("<e", value) for value in values)


def build_catalog() -> tuple[VuVLoopSemanticCase, ...]:
    rectangular_geometry = VuVLoopGeometry(128, 64, 384, 192)
    held_out_geometry = VuVLoopGeometry(192, 64, 384, 128)
    cases = (
        _case(
            0,
            "ct-vuvloop-rectangular-control",
            "rectangular-proportional",
            rectangular_geometry,
            "Does the isolated protocol reproduce the rectangular control?",
            "rectangular-repeat",
            "reject-request",
            disposition="exact-control",
            evidence_maturity="supported-contract-control",
            exact_legality=True,
            expected_result=_rectangular_expected(rectangular_geometry),
        ),
        _case(
            1,
            "ct-vuvloop-supported-outer2-control",
            "supported-outer2",
            held_out_geometry,
            "Does a second supported proportional geometry execute exactly?",
            "supported-contract",
            "reject-request",
            disposition="exact-control",
            evidence_maturity="supported-contract-control",
            exact_legality=True,
            expected_result=_rectangular_expected(held_out_geometry),
        ),
    )

    ids = {case.case_id for case in cases}
    names = {case.name for case in cases}
    distinctions = {case.distinction for case in cases}
    if len(ids) != len(cases) or len(names) != len(cases):
        raise RuntimeError("VuVLoop semantic catalog has duplicate keys")
    if len(distinctions) != len(cases):
        raise RuntimeError(
            "VuVLoop semantic catalog has duplicate distinctions"
        )
    for case in cases:
        geometry = case.geometry
        if not (
            geometry.elem_count > 0
            and geometry.unit_elem_count == 64
            and geometry.full_elem_count > 0
            and geometry.full_unit_elem_count > 0
        ):
            raise RuntimeError(f"{case.name}: non-positive geometry")
        if geometry.lhs_declared_bytes > LHS_OWNED_BYTES:
            raise RuntimeError(f"{case.name}: lhs exceeds owned span")
        if geometry.rhs_declared_bytes > RHS_OWNED_BYTES:
            raise RuntimeError(f"{case.name}: rhs exceeds owned span")
        if geometry.output_physical_span > OUTPUT_CAPTURE_BYTES:
            raise RuntimeError(
                f"{case.name}: output exceeds bounded capture"
            )
        if case.exact_legality:
            if (
                case.disposition != "exact-control"
                or case.evidence_maturity
                != "supported-contract-control"
                or not case.exact_legality
                or case.expected_result is None
            ):
                raise RuntimeError(
                    f"{case.name}: rectangular control lost its exact oracle"
                )
        if not geometry.satisfies_existing_catalog_gate:
            raise RuntimeError(f"{case.name}: unsupported VuVLoop geometry")
    if any(
        geometry.satisfies_existing_catalog_gate
        for geometry in STATIC_NEGATIVE_GEOMETRIES.values()
    ):
        raise RuntimeError("VuVLoop static negative became board-legal")
    return cases


CATALOG = build_catalog()
CASES_BY_NAME = {case.name: case for case in CATALOG}


def _guarded(body: bytes) -> GuardedRegion:
    guard = bytes([GUARD_BYTE]) * GUARD_BYTES
    return GuardedRegion(body, guard, guard)


def _f16_body(
    declared_elements: int,
    owned_bytes: int,
    value: Callable[[int], float],
    poison: Callable[[int], float],
) -> bytes:
    owned_elements = owned_bytes // ELEMENT_BYTES
    return b"".join(
        struct.pack(
            "<e",
            value(index)
            if index < declared_elements
            else poison(index),
        )
        for index in range(owned_elements)
    )


def build_observation_payload(
    case: VuVLoopSemanticCase, sample: int = 0
) -> VuVLoopObservationPayload:
    geometry = case.geometry
    lhs = _f16_body(
        geometry.full_elem_count,
        LHS_OWNED_BYTES,
        _lhs_value,
        lambda index: 48.0 if index % 2 else -48.0,
    )
    rhs = _f16_body(
        geometry.full_unit_elem_count,
        RHS_OWNED_BYTES,
        lambda index: _rhs_value(index, geometry),
        lambda index: 56.0 if index % 2 else -56.0,
    )
    output_seed = _f16_body(
        0,
        OUTPUT_CAPTURE_BYTES,
        lambda _index: -13.0,
        lambda _index: -13.0,
    )
    return VuVLoopObservationPayload(
        request=case.encode_request(sample),
        lhs=_guarded(lhs),
        rhs=_guarded(rhs),
        output=_guarded(output_seed),
        expected_result=case.expected_result,
    )
