#!/usr/bin/env python3
"""Typed SPM capacity/reservation/alignment calibration case inventory."""

from __future__ import annotations

import dataclasses
import struct


ALLOCATABLE_BEGIN = 0x10000
ALLOCATABLE_END = 0x2F0000
HARDWARE_END = 0x300000
RESERVED_BEGIN = ALLOCATABLE_END
SWEEP_BASE = 0x100000
TRANSFER_BYTES = 4096
DISPOSITIONS = {"BOARD_ROUNDTRIP", "STATIC_NEGATIVE"}
REQUEST_MAGIC = 0x315145524D505357
RECORD_MAGIC = 0x314345524D505357
REQUEST_GUARD = 0xD7C6B5A493827160
RECORD_GUARD = 0x1021324354657687
SCHEMA = 1
REQUEST_WORDS = 16
RECORD_WORDS = 32
RESOURCE_BYTES = 65536
SLOT_BYTES = 16384
OUTPUT_DDR_OFFSET = SLOT_BYTES
OUTPUT_CANARY = 0xA5
REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
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
    "SCHEMA_AND_WORDS": 1,
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

    @property
    def end(self) -> int:
        return self.address + self.transfer_bytes

    @property
    def is_safe(self) -> bool:
        return self.disposition == "BOARD_ROUNDTRIP"

    def payload(self, seed: int) -> bytes:
        if not self.is_safe:
            raise RuntimeError(
                f"{self.name}: static negative cannot produce board payload"
            )
        return bytes(
            ((seed * 29 + index * 17 + (index >> 8) * 7) & 0xFF)
            for index in range(self.transfer_bytes)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "domain": self.domain,
            "disposition": self.disposition.lower(),
            "address": self.address,
            "end_exclusive": self.end,
            "transfer_bytes": self.transfer_bytes,
            "relative_offset": self.address - SWEEP_BASE,
            "mod_256": self.address % 256,
            "mod_1024": self.address % 1024,
            "expected_violation": self.expected_violation,
            "oracle": (
                "exact-rdma-wdma-roundtrip+full-range-guard+pmu"
                if self.is_safe
                else "typed-range-rejection"
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
    64,
    128,
    192,
    256,
    320,
    512,
    768,
    1024,
    4096,
    65536,
    65728,
)
ALIGNMENT_CASES = tuple(
    SPMCase(
        f"spm-offset-{offset}",
        "alignment-bank",
        "BOARD_ROUNDTRIP",
        SWEEP_BASE + offset,
        TRANSFER_BYTES,
    )
    for offset in ALIGNMENT_OFFSETS
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

CATALOG = CAPACITY_CASES + ALIGNMENT_CASES + NEGATIVE_CASES
CASES_BY_NAME = {case.name: case for case in CATALOG}
BOARD_CASES = tuple(case for case in CATALOG if case.is_safe)
STATIC_NEGATIVE_CASES = tuple(case for case in CATALOG if not case.is_safe)
CASE_IDS = {case.name: index for index, case in enumerate(BOARD_CASES)}


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
    words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
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
