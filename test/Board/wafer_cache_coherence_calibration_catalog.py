#!/usr/bin/env python3
"""Four-direction DDR/Kcore/NCC/host visibility calibration contract."""

from __future__ import annotations

import dataclasses
import struct


REQUEST_MAGIC = 0x3151455248434357
RECORD_MAGIC = 0x3143455248434357
REQUEST_GUARD = 0xABCDEF0123456789
RECORD_GUARD = 0x9876543210FEDCBA
SCHEMA = 1
REQUEST_WORDS = 16
RECORD_WORDS = 32
RESOURCE_BYTES = 65536
BODY_OFFSET = 256
PAYLOAD_BYTES = 16384
OUTPUT0_OFFSET = 8192
OUTPUT1_OFFSET = 32768
OUTPUT_CANARY = 0xA5
SPM_GUARD_BYTES = 64
REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "SAMPLE": 3,
    "PAYLOAD_BYTES": 4,
    "EXPECTED_RDMA": 5,
    "EXPECTED_WDMA": 6,
    "RESOURCE_BYTES": 7,
    "BODY_OFFSET": 8,
    "OUTPUT0_OFFSET": 9,
    "OUTPUT1_OFFSET": 10,
    "GUARD": 15,
}
REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "SAMPLE": 4,
    "PAYLOAD_BYTES": 5,
    "EXPECTED_RDMA": 6,
    "EXPECTED_WDMA": 7,
    "REQUEST_GUARD": 8,
    "OUTPUT0_OFFSET": 9,
    "OUTPUT1_OFFSET": 10,
    "MISMATCH_BEFORE": 11,
    "MISMATCH_AFTER": 12,
    "SPM_GUARD_MISMATCHES": 13,
    "OUTPUT_GUARD_MISMATCHES": 14,
    "RDMA_INST_DELTA": 15,
    "WDMA_INST_DELTA": 16,
    "CACHE_CONTROL_MASK": 17,
    "RDMA_EXEC_DELTA": 18,
    "WDMA_EXEC_DELTA": 19,
    "RECORD_GUARD": 31,
}


@dataclasses.dataclass(frozen=True)
class CacheCoherenceCase:
    case_id: int
    name: str
    direction: str
    phases: int
    expected_rdma: int
    expected_wdma: int
    cache_control: str
    output_regions: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "direction": self.direction,
            "phases": self.phases,
            "payload_bytes": PAYLOAD_BYTES,
            "expected_rdma": self.expected_rdma,
            "expected_wdma": self.expected_wdma,
            "cache_control": self.cache_control,
            "output_regions": self.output_regions,
            "oracle": (
                "pre-control-observation+post-control-exact+guards+"
                "matching-completion"
            ),
            "disposition": "board-executable",
        }


CATALOG = (
    CacheCoherenceCase(
        0,
        "cache-host-h2d-to-kcore-read",
        "host-h2d->kcore",
        2,
        0,
        0,
        "phase0 invalidate+prime; phase1 read-before/invalidate/read-after",
        0,
    ),
    CacheCoherenceCase(
        1,
        "cache-kcore-store-to-ncc-read",
        "kcore-store->rdma",
        1,
        2,
        2,
        "rdma-before-clean; clean+fence; rdma-after-clean",
        2,
    ),
    CacheCoherenceCase(
        2,
        "cache-ncc-write-to-kcore-read",
        "wdma->kcore",
        1,
        1,
        1,
        "prime-old-cache; wdma+completion; read-before/invalidate/read-after",
        1,
    ),
    CacheCoherenceCase(
        3,
        "cache-wdma-to-host-d2h",
        "wdma->host-d2h",
        1,
        1,
        1,
        "matching local completion before runtime terminal/readback",
        1,
    ),
)
CASES_BY_NAME = {case.name: case for case in CATALOG}
CASES_BY_ID = {case.case_id: case for case in CATALOG}


def payload_pattern(case_id: int, sample: int) -> bytes:
    return bytes(
        (
            case_id * 41
            + (sample + 1) * 13
            + index * 17
            + (index >> 7) * 29
            + 3
        )
        & 0xFF
        for index in range(PAYLOAD_BYTES)
    )


def kcore_store_pattern(case_id: int, sample: int) -> bytes:
    return bytes(
        (
            case_id * 41
            + (sample + 101) * 13
            + index * 17
            + (index >> 7) * 29
            + 3
        )
        & 0xFF
        for index in range(PAYLOAD_BYTES)
    )


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    input_pattern: bytes
    post_control_expected: bytes


def build_case_payload(
    case: CacheCoherenceCase, sample: int
) -> CasePayload:
    if sample < 0 or sample >= case.phases:
        raise RuntimeError(f"{case.name}: sample {sample} is outside phase count")
    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
    words[REQ["CASE"]] = case.case_id
    words[REQ["SAMPLE"]] = sample
    words[REQ["PAYLOAD_BYTES"]] = PAYLOAD_BYTES
    words[REQ["EXPECTED_RDMA"]] = case.expected_rdma
    words[REQ["EXPECTED_WDMA"]] = case.expected_wdma
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    words[REQ["OUTPUT0_OFFSET"]] = OUTPUT0_OFFSET
    words[REQ["OUTPUT1_OFFSET"]] = OUTPUT1_OFFSET
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    input_pattern = payload_pattern(case.case_id, sample)
    payload = bytearray([OUTPUT_CANARY] * RESOURCE_BYTES)
    payload[BODY_OFFSET : BODY_OFFSET + PAYLOAD_BYTES] = input_pattern
    expected = (
        kcore_store_pattern(case.case_id, sample)
        if case.case_id == 1
        else input_pattern
    )
    return CasePayload(
        request + bytes([OUTPUT_CANARY]) * (RESOURCE_BYTES - len(request)),
        bytes(payload),
        input_pattern,
        expected,
    )
