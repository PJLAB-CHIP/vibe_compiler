#!/usr/bin/env python3
"""Typed SPM capacity/reservation/alignment calibration case inventory."""

from __future__ import annotations

import dataclasses
import struct

import wafer_datamove_calibration_catalog as datamove
import wafer_memory_descriptor_calibration_catalog as memory_descriptor


ALLOCATABLE_BEGIN = 0x10000
ALLOCATABLE_END = 0x2F0000
HARDWARE_END = 0x300000
RESERVED_BEGIN = ALLOCATABLE_END
SWEEP_BASE = 0x100000
TRANSFER_BYTES = 4096
PMU_REPETITIONS = 3
DISPOSITIONS = {
    "BOARD_ROUNDTRIP",
    "BOARD_OBSERVATION",
    "BOARD_LIFETIME",
    "DELEGATED_BOARD_CASE",
    "ISOLATED_DEFERRED",
    "STATIC_NEGATIVE",
}
REQUEST_MAGIC = 0x315145524D505357
RECORD_MAGIC = 0x314345524D505357
REQUEST_GUARD = 0xD7C6B5A493827160
RECORD_GUARD = 0x1021324354657687
REQUEST_WORDS = 16
RECORD_WORDS = 32
RESOURCE_BYTES = 131072
SLOT_BYTES = 65536
OUTPUT_DDR_OFFSET = SLOT_BYTES
OUTPUT_CANARY = 0xA5
REQ = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
    "CASE": 2,
    "ADDRESS": 3,
    "TRANSFER_BYTES": 4,
    "SAMPLE": 5,
    "RESOURCE_BYTES": 6,
    "SLOT_BYTES": 7,
    "GUARD": 15,
}
REC = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
    "STATUS": 2,
    "CASE": 3,
    "ADDRESS": 4,
    "TRANSFER_BYTES": 5,
    "SAMPLE": 6,
    "REQUEST_GUARD": 7,
    "OUTPUT_DDR_OFFSET": 8,
    "SLOT_BYTES": 9,
    "SPM_GUARD_MISMATCHES": 10,
    "RDMA_INST_DELTA": 11,
    "WDMA_INST_DELTA": 12,
    "RDMA_EXEC_DELTA": 13,
    "WDMA_EXEC_DELTA": 14,
    "RDMA_BLOCKING_DELTA": 15,
    "WDMA_BLOCKING_DELTA": 16,
    "RECORD_GUARD": 31,
}


@dataclasses.dataclass(frozen=True)
class SPMCase:
    name: str
    domain: str
    disposition: str
    address: int
    transfer_bytes: int
    expected_violation: str | None = None
    kind: str = "roundtrip"
    iterations: int = 1
    slot_stride: int = 0
    address_b: int | None = None
    effect: str | None = None
    relation: str | None = None
    evidence: tuple[object, ...] = ()
    repetitions: int = 1

    @property
    def end(self) -> int:
        return self.address + self.transfer_bytes

    @property
    def is_safe(self) -> bool:
        return self.disposition in {
            "BOARD_ROUNDTRIP",
            "BOARD_OBSERVATION",
            "BOARD_LIFETIME",
        }

    @property
    def expected_instructions(self) -> int:
        return self.iterations

    def iteration_payload(self, seed: int, iteration: int) -> bytes:
        return bytes(
            (
                seed * 29
                + iteration * 43
                + index * 17
                + (index >> 8) * 7
            )
            & 0xFF
            for index in range(self.transfer_bytes)
        )

    def payload(self, seed: int) -> bytes:
        if not self.is_safe:
            raise RuntimeError(
                f"{self.name}: static negative cannot produce board payload"
            )
        return b"".join(
            self.iteration_payload(seed, iteration)
            for iteration in range(self.iterations)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "domain": self.domain,
            "disposition": self.disposition.lower(),
            "address": self.address,
            "end_exclusive": self.end,
            "transfer_bytes": self.transfer_bytes,
            "kind": self.kind,
            "iterations": self.iterations,
            "slot_stride": self.slot_stride,
            "address_b": self.address_b,
            "effect": self.effect,
            "relation": self.relation,
            "repetitions": self.repetitions,
            "relative_offset": self.address - SWEEP_BASE,
            "mod_256": self.address % 256,
            "mod_1024": self.address % 1024,
            "expected_violation": self.expected_violation,
            "oracle": (
                "exact-rdma-wdma-roundtrip+full-range-guard+completion+pmu"
                if self.is_safe
                else (
                    "delegated-exact-board-case"
                    if self.disposition == "DELEGATED_BOARD_CASE"
                    else "typed-rejection-or-isolated-deferred"
                )
            ),
            "evidence": tuple(
                getattr(row, "name", repr(row)) for row in self.evidence
            ),
        }


CAPACITY_CASES = (
    SPMCase(
        "spm-lower-bound-256",
        "capacity-reservation",
        "BOARD_ROUNDTRIP",
        ALLOCATABLE_BEGIN,
        256,
    ),
    SPMCase(
        "spm-lower-heldout-8192",
        "capacity-reservation",
        "BOARD_ROUNDTRIP",
        ALLOCATABLE_BEGIN,
        8192,
    ),
    SPMCase(
        "spm-upper-heldout-4096",
        "capacity-reservation",
        "BOARD_ROUNDTRIP",
        ALLOCATABLE_END - 4096,
        4096,
    ),
    SPMCase(
        "spm-upper-bound-256",
        "capacity-reservation",
        "BOARD_ROUNDTRIP",
        ALLOCATABLE_END - 256,
        256,
    ),
)

ALIGNMENT_OFFSETS = (
    0,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    65792,
)
ALIGNMENT_CASES = tuple(
    SPMCase(
        f"spm-offset-{offset}",
        "alignment-bank",
        "BOARD_ROUNDTRIP",
        SWEEP_BASE + offset,
        TRANSFER_BYTES,
        repetitions=PMU_REPETITIONS,
    )
    for offset in ALIGNMENT_OFFSETS
)

LARGE_TRANSFER_CASES = (
    SPMCase(
        "spm-offset-sweep-large-65536",
        "alignment-bank",
        "BOARD_ROUNDTRIP",
        0x180000,
        65536,
        kind="roundtrip-large",
        repetitions=PMU_REPETITIONS,
    ),
)

LIFETIME_CASES = (
    SPMCase(
        "spm-lifetime-single-iteration",
        "lifetime-reuse",
        "BOARD_LIFETIME",
        0x140000,
        4096,
        kind="lifetime",
        iterations=1,
        slot_stride=8192,
    ),
    SPMCase(
        "spm-lifetime-double-slot-even-4",
        "lifetime-reuse",
        "BOARD_LIFETIME",
        0x140000,
        4096,
        kind="lifetime",
        iterations=4,
        slot_stride=8192,
    ),
    SPMCase(
        "spm-lifetime-double-slot-odd-5",
        "lifetime-reuse",
        "BOARD_LIFETIME",
        0x140000,
        4096,
        kind="lifetime",
        iterations=5,
        slot_stride=8192,
    ),
)

NON_PREFERRED_GEOMETRY_CASES = (
    SPMCase(
        "spm-base-offset-64-observation",
        "alignment-bank",
        "BOARD_OBSERVATION",
        SWEEP_BASE + 64,
        TRANSFER_BYTES,
        kind="roundtrip-non-preferred-geometry",
    ),
    SPMCase(
        "spm-base-offset-128-observation",
        "alignment-bank",
        "BOARD_OBSERVATION",
        SWEEP_BASE + 128,
        TRANSFER_BYTES,
        kind="roundtrip-non-preferred-geometry",
    ),
    SPMCase(
        "spm-base-offset-192-observation",
        "alignment-bank",
        "BOARD_OBSERVATION",
        SWEEP_BASE + 192,
        TRANSFER_BYTES,
        kind="roundtrip-non-preferred-geometry",
    ),
    SPMCase(
        "spm-length-128-observation",
        "alignment-bank",
        "BOARD_OBSERVATION",
        SWEEP_BASE,
        128,
        kind="roundtrip-non-preferred-geometry",
    ),
    SPMCase(
        "spm-length-384-observation",
        "alignment-bank",
        "BOARD_OBSERVATION",
        SWEEP_BASE,
        384,
        kind="roundtrip-non-preferred-geometry",
    ),
)

NEGATIVE_CASES = (
    SPMCase(
        "spm-below-allocatable-base",
        "capacity-reservation",
        "STATIC_NEGATIVE",
        ALLOCATABLE_BEGIN - 256,
        256,
        "below-allocatable-window",
    ),
    SPMCase(
        "spm-crosses-reserved-boundary",
        "capacity-reservation",
        "STATIC_NEGATIVE",
        ALLOCATABLE_END - 128,
        256,
        "crosses-reserved-window",
    ),
    SPMCase(
        "spm-reserved-window-first-line",
        "capacity-reservation",
        "STATIC_NEGATIVE",
        RESERVED_BEGIN,
        256,
        "reserved-window",
    ),
    SPMCase(
        "spm-hardware-end-crossing",
        "capacity-reservation",
        "STATIC_NEGATIVE",
        HARDWARE_END - 128,
        256,
        "outside-hardware-window",
    ),
    SPMCase(
        "spm-zero-length",
        "capacity-reservation",
        "STATIC_NEGATIVE",
        ALLOCATABLE_BEGIN,
        0,
        "zero-length",
    ),
    SPMCase(
        "spm-address-wrap",
        "capacity-reservation",
        "STATIC_NEGATIVE",
        (1 << 64) - 127,
        256,
        "address-overflow",
    ),
)


_RELATION_EFFECTS = ("RAW", "WAR", "WAW", "RAR")
_RELATIONS = ("exact", "half-partial", "adjacent", "far-disjoint", "strided")
ADDRESS_RELATION_CASES = tuple(
    SPMCase(
        f"spm-relation-{effect.lower()}-{relation}",
        "address-relation",
        "DELEGATED_BOARD_CASE",
        SWEEP_BASE,
        TRANSFER_BYTES,
        None,
        address_b=(
            SWEEP_BASE
            if relation == "exact"
            else SWEEP_BASE + TRANSFER_BYTES // 2
            if relation == "half-partial"
            else SWEEP_BASE + TRANSFER_BYTES
            if relation == "adjacent"
            else SWEEP_BASE + 4 * TRANSFER_BYTES
        ),
        effect=effect,
        relation=relation,
        evidence=(
            memory_descriptor.CASES_BY_NAME[
                f"spm-relation-{effect.lower()}-{relation}"
            ],
        ),
    )
    for effect in _RELATION_EFFECTS
    for relation in _RELATIONS
)


def _layout_evidence(channels: int) -> tuple[object, ...]:
    return tuple(
        case
        for case in datamove.CATALOG
        if case.destination_shape == (2, 7, 9, channels)
        and case.operation
        in {"tensor-to-cx", "cx-to-tensor", "tensor-to-ncx", "ncx-to-tensor"}
    )


PHYSICAL_LAYOUT_CASES = tuple(
    SPMCase(
        f"spm-physical-layout-c{channels}",
        "physical-layout",
        "DELEGATED_BOARD_CASE",
        ALLOCATABLE_BEGIN,
        256,
        kind="delegated-datamove",
        evidence=_layout_evidence(channels),
    )
    for channels in (63, 64, 65, 127, 129)
)

LIFETIME_NEGATIVE_CASES = (
    SPMCase(
        "spm-lifetime-reuse-before-completion",
        "lifetime-reuse",
        "STATIC_NEGATIVE",
        0x140000,
        4096,
        "reuse-without-matching-wait-or-token",
        kind="lifetime-negative",
    ),
    SPMCase(
        "spm-lifetime-dangling-view",
        "lifetime-reuse",
        "STATIC_NEGATIVE",
        0x140000,
        4096,
        "view-outlives-owning-slot",
        kind="lifetime-negative",
    ),
    SPMCase(
        "spm-lifetime-double-slot-capacity-insufficient",
        "lifetime-reuse",
        "STATIC_NEGATIVE",
        ALLOCATABLE_END - 4096,
        4096,
        "double-slot-exceeds-allocatable-window",
        kind="lifetime-negative",
        slot_stride=8192,
    ),
    SPMCase(
        "spm-lifetime-double-slot-alias",
        "lifetime-reuse",
        "STATIC_NEGATIVE",
        0x140000,
        4096,
        "distinct-logical-slots-have-overlapping-physical-ranges",
        kind="lifetime-negative",
        slot_stride=2048,
    ),
)

FIVE_ENGINE_ACCESS_CASES = tuple(
    SPMCase(
        f"spm-engine-access-{engine.lower()}",
        "engine-access",
        "DELEGATED_BOARD_CASE",
        SWEEP_BASE,
        TRANSFER_BYTES,
        None,
        kind=f"engine-{engine.lower()}",
        effect=engine,
        evidence=(
            memory_descriptor.CASES_BY_NAME[
                f"spm-engine-access-{engine.lower()}"
            ],
        ),
    )
    for engine in ("CT", "NE", "RDMA", "WDMA", "TDMA")
)

BANK_ENGINE_PAIR_CASES = tuple(
    SPMCase(
        case.name,
        "bank-engine-pair",
        "DELEGATED_BOARD_CASE",
        case.spm_a,
        case.descriptor.compact_bytes,
        None,
        address_b=case.spm_b,
        effect=(
            f"{memory_descriptor.ENGINE_NAMES[case.engine_a]}-"
            f"{memory_descriptor.ENGINE_NAMES[case.engine_b]}"
        ),
        relation=(
            "serial"
            if case.schedule == memory_descriptor.SCHEDULE_SERIAL
            else "window"
        ),
        evidence=(case,),
    )
    for case in memory_descriptor.PAIR_CASES
)

CATALOG = (
    CAPACITY_CASES
    + ALIGNMENT_CASES
    + LARGE_TRANSFER_CASES
    + LIFETIME_CASES
    + NON_PREFERRED_GEOMETRY_CASES
    + NEGATIVE_CASES
    + ADDRESS_RELATION_CASES
    + PHYSICAL_LAYOUT_CASES
    + LIFETIME_NEGATIVE_CASES
    + FIVE_ENGINE_ACCESS_CASES
    + BANK_ENGINE_PAIR_CASES
)
CASES_BY_NAME = {case.name: case for case in CATALOG}
BOARD_CASES = tuple(case for case in CATALOG if case.is_safe)
STATIC_NEGATIVE_CASES = tuple(
    case for case in CATALOG if case.disposition == "STATIC_NEGATIVE"
)
DEFERRED_CASES = tuple(
    case for case in CATALOG if case.disposition == "ISOLATED_DEFERRED"
)
DELEGATED_CASES = tuple(
    case for case in CATALOG if case.disposition == "DELEGATED_BOARD_CASE"
)
CASE_IDS = {case.name: index for index, case in enumerate(BOARD_CASES)}

CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "lower-upper-capacity-boundaries": CAPACITY_CASES,
    "reservation-range-negatives": tuple(
        case
        for case in NEGATIVE_CASES
        if case.domain == "capacity-reservation"
    ),
    "relative-offset-sweep": ALIGNMENT_CASES + LARGE_TRANSFER_CASES,
    "non-preferred-geometry-roundtrip": NON_PREFERRED_GEOMETRY_CASES,
    "rdma-wdma-engine-access-delegated": tuple(
        case
        for case in FIVE_ENGINE_ACCESS_CASES
        if case.effect in {"RDMA", "WDMA"}
    ),
    "ct-ne-tdma-engine-access-delegated": tuple(
        case
        for case in FIVE_ENGINE_ACCESS_CASES
        if case.effect in {"CT", "NE", "TDMA"}
    ),
    "exact-partial-adjacent-disjoint-strided": (
        ADDRESS_RELATION_CASES
    ),
    "tensor-cx-ncx-physical-layout": PHYSICAL_LAYOUT_CASES,
    "slot-lifetime-reuse": LIFETIME_CASES,
    "slot-lifetime-negatives": LIFETIME_NEGATIVE_CASES,
    "bank-engine-pair-controls": BANK_ENGINE_PAIR_CASES,
}


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected: bytes


def build_case_payload(case: SPMCase, sample: int = 0) -> CasePayload:
    if not case.is_safe:
        raise RuntimeError(
            f"{case.name}: static negative cannot produce a device request"
        )
    data = case.payload(sample + 1)
    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["WORD_COUNT"]] = REQUEST_WORDS
    words[REQ["CASE"]] = CASE_IDS[case.name]
    words[REQ["ADDRESS"]] = case.address
    words[REQ["TRANSFER_BYTES"]] = case.transfer_bytes
    words[REQ["SAMPLE"]] = sample
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["SLOT_BYTES"]] = SLOT_BYTES
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    return CasePayload(
        request + bytes([OUTPUT_CANARY]) * (RESOURCE_BYTES - len(request)),
        data + bytes([OUTPUT_CANARY]) * (RESOURCE_BYTES - len(data)),
        data,
    )
