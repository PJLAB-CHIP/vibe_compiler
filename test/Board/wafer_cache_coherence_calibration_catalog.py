#!/usr/bin/env python3
"""Four-direction DDR/Kcore/NCC/host visibility calibration contract."""

from __future__ import annotations

import dataclasses
import struct

import wafer_memory_descriptor_calibration_catalog as memory_descriptor


REQUEST_MAGIC = 0x3151455248434357
RECORD_MAGIC = 0x3143455248434357
REQUEST_GUARD = 0xABCDEF0123456789
RECORD_GUARD = 0x9876543210FEDCBA
REQUEST_WORDS = 24
RECORD_WORDS = 32
RESOURCE_BYTES = 65536
BODY_OFFSET = 256
PAYLOAD_BYTES = 16384
OUTPUT0_OFFSET = 8192
OUTPUT1_OFFSET = 32768
OUTPUT_CANARY = 0xA5
SPM_GUARD_BYTES = 64
SPM_GUARD_VALUE = 0x6D
BANK_PAYLOAD_BYTES = 4096
BANK_SLOT_BYTES = BANK_PAYLOAD_BYTES + 2 * SPM_GUARD_BYTES
BANK_REGION_GAP = 8192
BANK_OFFSETS = (0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DDR_BANK_PMU_REPETITIONS = 3
BANK_SEED_RDMA_INSTRUCTIONS = 2
BANK_READBACK_WDMA_INSTRUCTIONS = 2
CONFLICT_MODE = 1
CONFLICT_TRANSFER_BYTES = (256, 4096)
CONFLICT_BASE_RELATION_SAME = 1
CONFLICT_BASE_RELATION_CROSS = 2
CONFLICT_BASE_TRANSLATIONS = (0, 4096)
CONFLICT_REPETITIONS = 4
CONFLICT_SCHEDULES = ("serial", "window")
CONFLICT_ISSUE_ORDERS = ("a-b", "b-a")
CONFLICT_BATCH_ROWS = (
    CONFLICT_REPETITIONS
    * len(CONFLICT_SCHEDULES)
    * len(CONFLICT_ISSUE_ORDERS)
)
CONFLICT_DATA_BASE = 8192
CONFLICT_ARCHIVE_BASE = 8192
CONFLICT_ARCHIVE_STRIDE = 2 * (
    max(CONFLICT_TRANSFER_BYTES) + 2 * SPM_GUARD_BYTES
)
CONFLICT_ROW_MAGIC = 0x5743434844524F57
CONFLICT_ROW_GUARD = 0xA18E42C7956B30DF
CONFLICT_ROW_WORDS = 18
CONFLICT_FIRST_CASE_ID = 1000
CONFLICT_RECORD_BYTES = (
    RECORD_WORDS + CONFLICT_BATCH_ROWS * CONFLICT_ROW_WORDS
) * 8
REQ = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
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
    "MODE": 14,
    "TRANSFER_BYTES": 15,
    "BASE_RELATION": 16,
    "BASE_TRANSLATION": 17,
    "REPETITIONS": 18,
    "BATCH_ROWS": 19,
    "ARCHIVE_BASE": 20,
    "ARCHIVE_STRIDE": 21,
    "GUARD": 23,
}
REC = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
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
    "MODE": 23,
    "TRANSFER_BYTES": 24,
    "BASE_RELATION": 25,
    "BASE_TRANSLATION": 26,
    "DDR_BASE_A": 27,
    "DDR_BASE_B": 28,
    "DDR_ADDRESS_A": 29,
    "DDR_ADDRESS_B": 30,
    "RECORD_GUARD": 31,
}
CONFLICT_ROW = {
    "MAGIC": 0,
    "IDENTITY": 1,
    "REPETITION": 2,
    "SCHEDULE": 3,
    "ISSUE_ORDER": 4,
    "RDMA_INST_DELTA": 5,
    "WDMA_INST_DELTA": 6,
    "RDMA_EXEC_DELTA": 7,
    "WDMA_EXEC_DELTA": 8,
    "PLAN_CYCLES": 9,
    "ARCHIVE0_OFFSET": 10,
    "ARCHIVE1_OFFSET": 11,
    "DDR_ADDRESS_A": 12,
    "DDR_ADDRESS_B": 13,
    "COMPLETED": 14,
    "GUARD": 15,
    "RESULT_MISMATCHES": 16,
    "SCHEDULE_POSITION": 17,
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
    evidence: tuple[object, ...] = ()
    repetitions: int = 1
    mode: int = 0
    transfer_bytes: int = 0
    base_relation: int = 0
    base_translation: int = 0
    batch_repetitions: int = 0

    def as_dict(self) -> dict[str, object]:
        if self.kind == "ddr-conflict-equivalence":
            oracle = (
                "same-invocation-serial/window+a-b/b-a+four-repeats+"
                "counterbalanced-schedule-position+pair-window-counts+"
                "actual-address-echo+per-row-result/guard-mismatch+"
                "host-final-archive"
            )
        elif self.kind == "ddr-bank-pair":
            oracle = (
                "host-payload-rdma-slot-seed+pair+wdma-full-slot-readback+"
                "host-exact+guards+terminal-completion"
            )
        elif self.case_id == 0:
            oracle = "single-invocation-invalidate+post-control-exact+guards"
        else:
            oracle = (
                "pre-control-observation+post-control-exact+guards+"
                "matching-completion"
            )
        return {
            "id": self.case_id,
            "name": self.name,
            "direction": self.direction,
            "phases": self.phases,
            "repetitions": self.repetitions,
            "payload_bytes": self.payload_bytes,
            "expected_rdma": self.expected_rdma,
            "expected_wdma": self.expected_wdma,
            "cache_control": self.cache_control,
            "output_regions": self.output_regions,
            "kind": self.kind,
            "pair_kind": self.pair_kind,
            "schedule": self.schedule,
            "bank_offset": self.bank_offset,
            "mode": self.mode,
            "transfer_bytes": self.transfer_bytes,
            "base_relation": {
                0: None,
                CONFLICT_BASE_RELATION_SAME: "same-allocation",
                CONFLICT_BASE_RELATION_CROSS: "cross-allocation",
            }[self.base_relation],
            "base_translation": self.base_translation,
            "batch_repetitions": self.batch_repetitions,
            "oracle": oracle,
            "disposition": self.disposition,
            "reason": self.reason,
            "evidence": tuple(
                getattr(row, "name", repr(row)) for row in self.evidence
            ),
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
    # PMU and instruction-count deltas cover the same fixed pure-NCC
    # end-to-end envelope for serial and window controls:
    #   two RDMA slot seeds + tested pair + two full-slot WDMA readbacks.
    counts = {
        "rdma-rdma": (4, 2, 2),
        "wdma-wdma": (2, 4, 2),
        "rdma-wdma": (3, 3, 2),
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
                            "same-allocation disjoint ranges; fixed RDMA-seed/"
                            "pair/full-slot-WDMA-readback envelope; matching "
                            f"terminal completion; {schedule} issue order"
                        ),
                        output_regions,
                        kind="ddr-bank-pair",
                        pair_kind=pair_kind,
                        schedule=schedule,
                        bank_offset=offset,
                        payload_bytes=BANK_PAYLOAD_BYTES,
                        repetitions=DDR_BANK_PMU_REPETITIONS,
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
        disposition="delegated-board-case",
        reason=(
            "the dedicated memory-descriptor package owns the 64KiB payload "
            "and its independent output/SPM guards"
        ),
        evidence=(
            memory_descriptor.CASES_BY_NAME[
                "ddr-large-contiguous-65536"
            ],
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
        disposition="delegated-board-case",
        reason=(
            "the dedicated memory-descriptor package owns independent "
            "1D/2D/3D envelopes and descriptor-hole guards"
        ),
        evidence=tuple(
            memory_descriptor.CASES_BY_NAME[name]
            for name in (
                "ddr-large-1d-stride-holes",
                "ddr-large-2d-stride-holes",
                "ddr-large-3d-stride-holes",
            )
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
        disposition="delegated-board-case",
        reason=(
            "the dedicated memory-descriptor package owns default burst "
            "boundary and large-tail observations"
        ),
        evidence=tuple(
            memory_descriptor.CASES_BY_NAME[name]
            for name in (
                "ddr-default-burst-boundary-4095",
                "ddr-default-burst-boundary-4096",
                "ddr-default-burst-boundary-4097",
                "ddr-large-tail-65535",
            )
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
            "the current one-shot BoardRuntime frees every allocation after "
            "one launch, so this state is unreachable and does not block the "
            "current compiler/runtime.  Reactivate the probe when a persistent "
            "session can reuse one allocation across launches: prime value A, "
            "host-overwrite the same address with B, then compare Kcore reads "
            "before and after invalidate"
        ),
        evidence=(
            CACHE_CASES[0],
            CACHE_CASES[2],
        ),
    ),
)

CATALOG = CACHE_CASES + DDR_BANK_CASES
CASES_BY_NAME = {case.name: case for case in CATALOG}
CASES_BY_ID = {case.case_id: case for case in CATALOG}


def _conflict_expected_counts(pair_kind: str) -> tuple[int, int]:
    if pair_kind == "rdma-rdma":
        return (
            BANK_SEED_RDMA_INSTRUCTIONS
            + CONFLICT_BATCH_ROWS * 2,
            CONFLICT_BATCH_ROWS * BANK_READBACK_WDMA_INSTRUCTIONS,
        )
    if pair_kind == "wdma-wdma":
        # Two initial RDMA slot seeds, four one-guard WDMA initializers, the
        # measured two-WDMA pair, and an after-PMU two-slot RDMA readback for
        # every row's independent exact mismatch oracle.
        return (
            BANK_SEED_RDMA_INSTRUCTIONS + CONFLICT_BATCH_ROWS * 2,
            4 + CONFLICT_BATCH_ROWS * 2,
        )
    raise ValueError(f"unsupported conflict pair kind: {pair_kind}")


def _conflict_cases(first_case_id: int) -> tuple[CacheCoherenceCase, ...]:
    rows: list[CacheCoherenceCase] = []
    case_id = first_case_id
    for pair_kind in ("rdma-rdma", "wdma-wdma"):
        relations = (
            (
                CONFLICT_BASE_RELATION_SAME,
                CONFLICT_BASE_RELATION_CROSS,
            )
            if pair_kind == "rdma-rdma"
            else (CONFLICT_BASE_RELATION_SAME,)
        )
        expected_rdma, expected_wdma = _conflict_expected_counts(pair_kind)
        for transfer_bytes in CONFLICT_TRANSFER_BYTES:
            for base_relation in relations:
                relation_name = (
                    "same-allocation"
                    if base_relation == CONFLICT_BASE_RELATION_SAME
                    else "cross-allocation"
                )
                for base_translation in CONFLICT_BASE_TRANSLATIONS:
                    for bank_offset in BANK_OFFSETS:
                        rows.append(
                            CacheCoherenceCase(
                                case_id,
                                (
                                    "ddr-conflict-equivalence-"
                                    f"{pair_kind}-{transfer_bytes}b-"
                                    f"{relation_name}-"
                                    f"translation-{base_translation}-"
                                    f"offset-{bank_offset}"
                                ),
                                f"ddr-{pair_kind}",
                                1,
                                expected_rdma,
                                expected_wdma,
                                (
                                    "one invocation batches serial/window "
                                    "and both issue orders with four "
                                    "pair-window repetitions"
                                ),
                                2,
                                kind="ddr-conflict-equivalence",
                                pair_kind=pair_kind,
                                schedule=None,
                                bank_offset=bank_offset,
                                payload_bytes=transfer_bytes,
                                disposition="pending-board-executable",
                                mode=CONFLICT_MODE,
                                transfer_bytes=transfer_bytes,
                                base_relation=base_relation,
                                base_translation=base_translation,
                                batch_repetitions=CONFLICT_REPETITIONS,
                            )
                        )
                        case_id += 1
    return tuple(rows)


PENDING_DDR_CONFLICT_CASES = _conflict_cases(CONFLICT_FIRST_CASE_ID)
ALL_CASES = CATALOG + PENDING_DDR_CONFLICT_CASES
ALL_CASES_BY_NAME = {case.name: case for case in ALL_CASES}
ALL_CASES_BY_ID = {case.case_id: case for case in ALL_CASES}

DDR_CONFLICT_DISPOSITIONS = (
    CacheCoherenceCase(
        CONFLICT_FIRST_CASE_ID + len(PENDING_DDR_CONFLICT_CASES),
        "ddr-conflict-equivalence-wdma-cross-allocation",
        "ddr-wdma-wdma",
        0,
        0,
        0,
        "not-issued",
        0,
        kind="ddr-conflict-equivalence",
        pair_kind="wdma-wdma",
        disposition="isolated-deferred",
        reason=(
            "the program-local ABI exposes only one device-to-host output "
            "allocation; writing the second WDMA lane into an input resource "
            "would not provide a host exact oracle"
        ),
        mode=CONFLICT_MODE,
        base_relation=CONFLICT_BASE_RELATION_CROSS,
    ),
)


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


def bank_pattern(
    case_id: int,
    sample: int,
    lane: int,
    count: int = BANK_PAYLOAD_BYTES,
) -> bytes:
    return bytes(
        (
            case_id * 31
            + (sample + 1) * 43
            + lane * 97
            + index * 19
            + (index >> 6) * 11
            + 5
        )
        & 0xFF
        for index in range(count)
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
    if case.disposition not in {
        "board-executable",
        "pending-board-executable",
    }:
        raise RuntimeError(
            f"{case.name}: deferred case cannot produce a device request"
        )
    sample_count = case.phases * case.repetitions
    if sample < 0 or sample >= sample_count:
        raise RuntimeError(
            f"{case.name}: sample {sample} is outside sample count"
        )
    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["WORD_COUNT"]] = REQUEST_WORDS
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
    words[REQ["MODE"]] = case.mode
    words[REQ["TRANSFER_BYTES"]] = case.transfer_bytes
    words[REQ["BASE_RELATION"]] = case.base_relation
    words[REQ["BASE_TRANSLATION"]] = case.base_translation
    words[REQ["REPETITIONS"]] = case.batch_repetitions
    words[REQ["BATCH_ROWS"]] = (
        CONFLICT_BATCH_ROWS
        if case.kind == "ddr-conflict-equivalence"
        else 0
    )
    words[REQ["ARCHIVE_BASE"]] = (
        CONFLICT_ARCHIVE_BASE
        if case.kind == "ddr-conflict-equivalence"
        else 0
    )
    words[REQ["ARCHIVE_STRIDE"]] = (
        CONFLICT_ARCHIVE_STRIDE
        if case.kind == "ddr-conflict-equivalence"
        else 0
    )
    words[REQ["GUARD"]] = REQUEST_GUARD
    request = bytearray([OUTPUT_CANARY] * RESOURCE_BYTES)
    request[: REQUEST_WORDS * 8] = struct.pack(
        f"<{REQUEST_WORDS}Q", *words
    )
    input_pattern = payload_pattern(case.case_id, sample, case.payload_bytes)
    payload = bytearray([OUTPUT_CANARY] * RESOURCE_BYTES)
    payload[BODY_OFFSET : BODY_OFFSET + case.payload_bytes] = input_pattern
    expected_regions: tuple[tuple[int, bytes], ...] = ()
    if case.kind == "ddr-conflict-equivalence":
        lane0 = bank_pattern(
            case.case_id, sample, 0, case.transfer_bytes
        )
        lane1 = bank_pattern(
            case.case_id, sample, 1, case.transfer_bytes
        )
        guard = bytes([SPM_GUARD_VALUE]) * SPM_GUARD_BYTES
        slot0 = guard + lane0 + guard
        slot1 = guard + lane1 + guard
        first = CONFLICT_DATA_BASE + case.base_translation
        second = (
            CONFLICT_DATA_BASE
            + case.base_translation
            + BANK_REGION_GAP
            + case.bank_offset
        )
        if case.base_relation == CONFLICT_BASE_RELATION_CROSS:
            request[
                first - SPM_GUARD_BYTES :
                first + case.transfer_bytes + SPM_GUARD_BYTES
            ] = slot0
            payload[
                second - SPM_GUARD_BYTES :
                second + case.transfer_bytes + SPM_GUARD_BYTES
            ] = slot1
        else:
            payload[
                first - SPM_GUARD_BYTES :
                first + case.transfer_bytes + SPM_GUARD_BYTES
            ] = slot0
            payload[
                second - SPM_GUARD_BYTES :
                second + case.transfer_bytes + SPM_GUARD_BYTES
            ] = slot1
        if case.pair_kind == "wdma-wdma":
            # WDMA always seeds its two SPM sources from the payload
            # allocation, even though its measured destinations are output.
            payload[
                first - SPM_GUARD_BYTES :
                first + case.transfer_bytes + SPM_GUARD_BYTES
            ] = slot0
            payload[
                second - SPM_GUARD_BYTES :
                second + case.transfer_bytes + SPM_GUARD_BYTES
            ] = slot1
            expected_regions = (
                (first - SPM_GUARD_BYTES, slot0),
                (second - SPM_GUARD_BYTES, slot1),
            )
        else:
            regions: list[tuple[int, bytes]] = []
            for schedule_index in range(len(CONFLICT_SCHEDULES)):
                for order_index in range(len(CONFLICT_ISSUE_ORDERS)):
                    control = (
                        schedule_index * len(CONFLICT_ISSUE_ORDERS)
                        + order_index
                    )
                    first_archive = (
                        CONFLICT_ARCHIVE_BASE
                        + control * CONFLICT_ARCHIVE_STRIDE
                    )
                    regions.append((first_archive, slot0))
                    regions.append(
                        (first_archive + len(slot0), slot1)
                    )
            expected_regions = tuple(regions)
        input_pattern = lane0
    elif case.kind == "ddr-bank-pair":
        lane0 = bank_pattern(case.case_id, sample, 0)
        lane1 = bank_pattern(case.case_id, sample, 1)
        second = BODY_OFFSET + BANK_REGION_GAP + case.bank_offset
        guard = bytes([SPM_GUARD_VALUE]) * SPM_GUARD_BYTES
        slot0 = guard + lane0 + guard
        slot1 = guard + lane1 + guard
        payload[
            BODY_OFFSET - SPM_GUARD_BYTES :
            BODY_OFFSET + BANK_PAYLOAD_BYTES + SPM_GUARD_BYTES
        ] = slot0
        payload[
            second - SPM_GUARD_BYTES :
            second + BANK_PAYLOAD_BYTES + SPM_GUARD_BYTES
        ] = slot1
        expected_regions = (
            (OUTPUT0_OFFSET - SPM_GUARD_BYTES, slot0),
            (
                OUTPUT0_OFFSET
                + BANK_REGION_GAP
                + case.bank_offset
                - SPM_GUARD_BYTES,
                slot1,
            ),
        )
        input_pattern = lane0
    expected = (
        kcore_store_pattern(case.case_id, sample, case.payload_bytes)
        if case.case_id == 1
        else input_pattern
    )
    return CasePayload(
        bytes(request),
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
