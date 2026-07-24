#!/usr/bin/env python3
"""Four-direction DDR/Kcore/NCC/host visibility calibration contract."""

from __future__ import annotations

import dataclasses
import struct


REQUEST_MAGIC = 0x3151455248434357
RECORD_MAGIC = 0x3143455248434357
REQUEST_GUARD = 0xABCDEF0123456789
RECORD_GUARD = 0x9876543210FEDCBA
SCHEMA = 2
REQUEST_WORDS = 16
RECORD_WORDS = 32
RESOURCE_BYTES = 65536
BODY_OFFSET = 256
PAYLOAD_BYTES = 16384
OUTPUT0_OFFSET = 8192
OUTPUT1_OFFSET = 32768
OUTPUT_CANARY = 0xA5
SPM_GUARD_BYTES = 64
BANK_PAYLOAD_BYTES = 4096
BANK_REGION_GAP = 8192
BANK_OFFSETS = (0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
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
    "PAIR_KIND": 11,
    "SCHEDULE": 12,
    "BANK_OFFSET": 13,
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
    "PAIR_KIND": 20,
    "SCHEDULE": 21,
    "BANK_OFFSET": 22,
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
    kind: str = "coherence"
    pair_kind: str | None = None
    schedule: str | None = None
    bank_offset: int = 0
    payload_bytes: int = PAYLOAD_BYTES
    disposition: str = "board-executable"
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        oracle = (
            "single-invocation-invalidate+post-control-exact+guards"
            if self.case_id == 0 and self.kind == "coherence"
            else (
                "pre-control-observation+post-control-exact+guards+"
                "matching-completion"
            )
        )
        return {
            "id": self.case_id,
            "name": self.name,
            "direction": self.direction,
            "phases": self.phases,
            "payload_bytes": self.payload_bytes,
            "expected_rdma": self.expected_rdma,
            "expected_wdma": self.expected_wdma,
            "cache_control": self.cache_control,
            "output_regions": self.output_regions,
            "kind": self.kind,
            "pair_kind": self.pair_kind,
            "schedule": self.schedule,
            "bank_offset": self.bank_offset,
            "oracle": oracle,
            "disposition": self.disposition,
            "reason": self.reason,
        }


CACHE_CASES = (
    CacheCoherenceCase(
        0,
        "cache-host-h2d-to-kcore-read",
        "host-h2d->kcore",
        1,
        0,
        0,
        "single invocation invalidate then exact Kcore read",
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


def _bank_cases(first_case_id: int) -> tuple[CacheCoherenceCase, ...]:
    rows: list[CacheCoherenceCase] = []
    case_id = first_case_id
    counts = {
        "rdma-rdma": (2, 0, 0),
        "wdma-wdma": (0, 2, 2),
        "rdma-wdma": (1, 1, 1),
    }
    for pair_kind in ("rdma-rdma", "wdma-wdma", "rdma-wdma"):
        expected_rdma, expected_wdma, output_regions = counts[pair_kind]
        for offset in BANK_OFFSETS:
            for schedule in ("serial", "window"):
                rows.append(
                    CacheCoherenceCase(
                        case_id,
                        f"ddr-bank-{pair_kind}-{schedule}-offset-{offset}",
                        f"ddr-{pair_kind}",
                        1,
                        expected_rdma,
                        expected_wdma,
                        (
                            "same-allocation disjoint ranges; matching local "
                            f"completion; {schedule} issue order"
                        ),
                        output_regions,
                        kind="ddr-bank-pair",
                        pair_kind=pair_kind,
                        schedule=schedule,
                        bank_offset=offset,
                        payload_bytes=BANK_PAYLOAD_BYTES,
                    )
                )
                case_id += 1
    return tuple(rows)


DDR_BANK_CASES = _bank_cases(len(CACHE_CASES))

DDR_LARGE_DESCRIPTOR_DISPOSITIONS = (
    CacheCoherenceCase(
        len(CACHE_CASES) + len(DDR_BANK_CASES) + 0,
        "ddr-large-64k-disjoint-pair",
        "ddr-large-descriptor",
        0,
        0,
        0,
        "not-issued",
        0,
        kind="descriptor-disposition",
        payload_bytes=65536,
        disposition="isolated-deferred",
        reason=(
            "two disjoint 64KiB ranges plus guards do not fit the current "
            "shared 64KiB resource; do not silently downsize the case"
        ),
    ),
    CacheCoherenceCase(
        len(CACHE_CASES) + len(DDR_BANK_CASES) + 1,
        "ddr-strided-holes-pair",
        "ddr-large-descriptor",
        0,
        0,
        0,
        "not-issued",
        0,
        kind="descriptor-disposition",
        disposition="isolated-deferred",
        reason=(
            "the cache probe record does not encode independent 1D/2D/3D "
            "descriptor envelopes; use the NCC DMA descriptor suite first"
        ),
    ),
    CacheCoherenceCase(
        len(CACHE_CASES) + len(DDR_BANK_CASES) + 2,
        "ddr-burst-boundary-tail-pair",
        "ddr-large-descriptor",
        0,
        0,
        0,
        "not-issued",
        0,
        kind="descriptor-disposition",
        disposition="isolated-deferred",
        reason=(
            "burst and tail unit semantics need a bounded descriptor record "
            "before a bank-classification packet is safe"
        ),
    ),
)

CACHE_SESSION_DISPOSITIONS = (
    CacheCoherenceCase(
        len(CACHE_CASES) + len(DDR_BANK_CASES)
        + len(DDR_LARGE_DESCRIPTOR_DISPOSITIONS),
        "cache-host-h2d-stale-same-session",
        "host-h2d->kcore-stale",
        0,
        0,
        0,
        "not-issued",
        0,
        kind="session-disposition",
        disposition="isolated-deferred",
        reason=(
            "a stale-cache oracle requires prime, host mutation and "
            "read-before/invalidate/read-after in one BoardRuntime session "
            "over the same allocation; separate wafer-run invocations unload "
            "and free resources"
        ),
    ),
)

CATALOG = CACHE_CASES + DDR_BANK_CASES
CASES_BY_NAME = {case.name: case for case in CATALOG}
CASES_BY_ID = {case.case_id: case for case in CATALOG}


def payload_pattern(case_id: int, sample: int, count: int = PAYLOAD_BYTES) -> bytes:
    return bytes(
        (
            case_id * 41
            + (sample + 1) * 13
            + index * 17
            + (index >> 7) * 29
            + 3
        )
        & 0xFF
        for index in range(count)
    )


def kcore_store_pattern(
    case_id: int, sample: int, count: int = PAYLOAD_BYTES
) -> bytes:
    return bytes(
        (
            case_id * 41
            + (sample + 101) * 13
            + index * 17
            + (index >> 7) * 29
            + 3
        )
        & 0xFF
        for index in range(count)
    )


def bank_pattern(case_id: int, lane: int) -> bytes:
    return bytes(
        (
            case_id * 31
            + lane * 97
            + index * 19
            + (index >> 6) * 11
            + 5
        )
        & 0xFF
        for index in range(BANK_PAYLOAD_BYTES)
    )


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    input_pattern: bytes
    post_control_expected: bytes
    expected_regions: tuple[tuple[int, bytes], ...] = ()


def build_case_payload(
    case: CacheCoherenceCase, sample: int
) -> CasePayload:
    if case.disposition != "board-executable":
        raise RuntimeError(
            f"{case.name}: deferred case cannot produce a device request"
        )
    if sample < 0 or sample >= case.phases:
        raise RuntimeError(f"{case.name}: sample {sample} is outside phase count")
    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
    words[REQ["CASE"]] = case.case_id
    words[REQ["SAMPLE"]] = sample
    words[REQ["PAYLOAD_BYTES"]] = case.payload_bytes
    words[REQ["EXPECTED_RDMA"]] = case.expected_rdma
    words[REQ["EXPECTED_WDMA"]] = case.expected_wdma
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["BODY_OFFSET"]] = BODY_OFFSET
    words[REQ["OUTPUT0_OFFSET"]] = OUTPUT0_OFFSET
    words[REQ["OUTPUT1_OFFSET"]] = OUTPUT1_OFFSET
    pair_codes = {None: 0, "rdma-rdma": 1, "wdma-wdma": 2, "rdma-wdma": 3}
    schedule_codes = {None: 0, "serial": 1, "window": 2}
    words[REQ["PAIR_KIND"]] = pair_codes[case.pair_kind]
    words[REQ["SCHEDULE"]] = schedule_codes[case.schedule]
    words[REQ["BANK_OFFSET"]] = case.bank_offset
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    input_pattern = payload_pattern(case.case_id, sample, case.payload_bytes)
    payload = bytearray([OUTPUT_CANARY] * RESOURCE_BYTES)
    payload[BODY_OFFSET : BODY_OFFSET + case.payload_bytes] = input_pattern
    expected_regions: tuple[tuple[int, bytes], ...] = ()
    if case.kind == "ddr-bank-pair":
        lane0 = bank_pattern(case.case_id, 0)
        lane1 = bank_pattern(case.case_id, 1)
        second = BODY_OFFSET + BANK_REGION_GAP + case.bank_offset
        payload[BODY_OFFSET : BODY_OFFSET + BANK_PAYLOAD_BYTES] = lane0
        payload[second : second + BANK_PAYLOAD_BYTES] = lane1
        if case.pair_kind == "wdma-wdma":
            expected_regions = (
                (OUTPUT0_OFFSET, lane0),
                (
                    OUTPUT0_OFFSET + BANK_REGION_GAP + case.bank_offset,
                    lane1,
                ),
            )
        elif case.pair_kind == "rdma-wdma":
            expected_regions = ((OUTPUT0_OFFSET, lane0),)
        input_pattern = lane0
    expected = (
        kcore_store_pattern(case.case_id, sample, case.payload_bytes)
        if case.case_id == 1
        else input_pattern
    )
    return CasePayload(
        request + bytes([OUTPUT_CANARY]) * (RESOURCE_BYTES - len(request)),
        bytes(payload),
        input_pattern,
        expected,
        expected_regions,
    )


CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "four-visibility-directions": CACHE_CASES,
    "cache-same-session-stale-invalidation": CACHE_SESSION_DISPOSITIONS,
    "ddr-same-allocation-offset-sweep": DDR_BANK_CASES,
    "ddr-rdma-wdma-pair-controls": DDR_BANK_CASES,
    "ddr-large-stride-burst-tail": DDR_LARGE_DESCRIPTOR_DISPOSITIONS,
}
