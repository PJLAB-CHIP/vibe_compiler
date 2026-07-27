#!/usr/bin/env python3
"""Bounded DMA/DDR descriptors and SPM engine/relation observations."""

from __future__ import annotations

import dataclasses
import struct


REQUEST_MAGIC = 0x315145444D454D57
RECORD_MAGIC = 0x314345444D454D57
REQUEST_GUARD = 0xB5A4938271605F4E
RECORD_GUARD = 0x32435465768798A9
SCHEMA = 3
REQUEST_WORDS = 40
RECORD_WORDS = 64
RESOURCE_BYTES = 2097152
PAYLOAD_DATA_OFFSET = 4096
OUTPUT_DATA_OFFSET = 4096
OUTPUT_SINK0_OFFSET = 65536
OUTPUT_SINK1_OFFSET = 73728
OUTPUT_LAYOUT_ALIGNMENT = 256
RESOURCE_CANARY = 0xA5
SPM_GUARD = 0x6D
SPM_GUARD_BYTES = 64
SPM_PAIR_SLOT_BYTES = 2560
SPM_ALLOCATABLE_BEGIN = 0x10000
SPM_ALLOCATABLE_END = 0x2F0000
PAYLOAD_SPM_SEED_OFFSET = 135168
PAYLOAD_SPM_SEED_BYTES = 65536 + 2 * SPM_GUARD_BYTES
PAYLOAD_SPM_SCRATCH0_OFFSET = 204800
PAYLOAD_SPM_SCRATCH1_OFFSET = 212992

KIND_DMA = 0
KIND_ENGINE_ACCESS = 1
KIND_ADDRESS_RELATION = 2
KIND_ENGINE_PAIR = 3
KIND_PARALLEL_PAIR = 4

ENGINE_CT = 0
ENGINE_NE = 1
ENGINE_RDMA = 2
ENGINE_WDMA = 3
ENGINE_TDMA = 4
ENGINE_NONE = 255
ENGINE_NAMES = {
    ENGINE_CT: "CT",
    ENGINE_NE: "NE",
    ENGINE_RDMA: "RDMA",
    ENGINE_WDMA: "WDMA",
    ENGINE_TDMA: "TDMA",
}

SCHEDULE_SERIAL = 0
SCHEDULE_WINDOW = 1

EFFECT_NONE = 0
EFFECT_RAW = 1
EFFECT_WAR = 2
EFFECT_WAW = 3
EFFECT_RAR = 4
EFFECT_NAMES = {
    EFFECT_RAW: "RAW",
    EFFECT_WAR: "WAR",
    EFFECT_WAW: "WAW",
    EFFECT_RAR: "RAR",
}

RELATION_DISJOINT = 0
RELATION_EXACT = 1
RELATION_PARTIAL = 2
RELATION_ADJACENT = 3
RELATION_STRIDED = 4
RELATION_NAMES = {
    RELATION_DISJOINT: "far-disjoint",
    RELATION_EXACT: "exact",
    RELATION_PARTIAL: "half-partial",
    RELATION_ADJACENT: "adjacent",
    RELATION_STRIDED: "strided",
}

ORACLE_EXACT = 0
ORACLE_OBSERVATION = 1
FMT_FP16 = 2
FMT_UINT8 = 8

SPM_BASE = 0x100000
SPM_GENERAL_PAIR_RELATIVE_OFFSETS = (8192, 4352, 65536)
SPM_BANK_PERIOD_RELATIVE_OFFSETS = tuple(
    4096 + index * 256 for index in range(9)
)
SPM_PAIR_RELATIVE_OFFSETS = tuple(
    sorted(
        set(SPM_GENERAL_PAIR_RELATIVE_OFFSETS)
        | set(SPM_BANK_PERIOD_RELATIVE_OFFSETS)
    )
)
SPM_PARALLEL_ALIGNMENT_PHASES = (0, 64, 128, 192)
SPM_PARALLEL_ENGINES = (ENGINE_CT, ENGINE_RDMA)
SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS = (0x20000, 0x40000)
SPM_CONFLICT_EQUIVALENCE_TRANSFERS = (256, 512)
SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS = (0, 1)
SPM_BANK_PMU_REPETITIONS = 3
PERF_MAX_ROUNDS = 4
PERF_MAX_BUFFERS = 2
PERF_READ0_OFFSET = 0x100
PERF_READ1_OFFSET = 0x20100
PERF_WRITE_OFFSET = 0x40100
PERF_DISJOINT_A = 0x10100
PERF_DISJOINT_B = 0x110100
PERF_PAYLOAD_BASE = 0x20000
PERF_PAYLOAD_LANE_STRIDE = 0x80000
PERF_PAYLOAD_READ0_OFFSET = 0
PERF_PAYLOAD_READ1_OFFSET = 0x20000
PERF_PAYLOAD_RDMA_OFFSET = 0x40000
PERF_PAYLOAD_GUARD_SEED_OFFSET = 0x140000
PERF_OUTPUT_SINK_BASE = 0x10000
PERF_OUTPUT_RESULT_BASE = 0x90000
PERF_OUTPUT_LANE_STRIDE = 0x40000
PERF_OUTPUT_GUARD_BASE = 0x190000
PERF_OUTPUT_GUARD_LANE_STRIDE = 0x100

CONFLICT_REQUEST_MAGIC = 0x314551434D4D5357
CONFLICT_RECORD_MAGIC = 0x315245434D4D5357
CONFLICT_REQUEST_GUARD = 0xE7B3D98264A15C0F
CONFLICT_RECORD_GUARD = 0x19F04CB267D38AE5
CONFLICT_SCHEMA = 1
CONFLICT_ROWS = 2
CONFLICT_SAMPLES = 4
CONFLICT_SERIAL_REQUEST_WORD = 0
CONFLICT_WINDOW_REQUEST_WORD = REQUEST_WORDS
CONFLICT_REQUEST_META_WORD = 2 * REQUEST_WORDS
CONFLICT_REQUEST_META_WORDS = 20
CONFLICT_SERIAL_OUTPUT_OFFSET = 0
CONFLICT_WINDOW_OUTPUT_OFFSET = 0x4000
CONFLICT_OUTPUT_ROW_BYTES = 0x4000
CONFLICT_RECORD_META_WORD = 4096
CONFLICT_RECORD_META_STRIDE_WORDS = 32
CONFLICT_RECORD_META_WORDS = 24
CONFLICT_WINDOW_FLAGS = 0x7

REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "CASE": 2,
    "KIND": 3,
    "ENGINE_A": 4,
    "ENGINE_B": 5,
    "SCHEDULE": 6,
    "EFFECT": 7,
    "RELATION": 8,
    "ORACLE": 9,
    "FORMAT": 10,
    "SRC_DDR_OFFSET": 11,
    "DST_DDR_OFFSET": 12,
    "SPM_A": 13,
    "SPM_B": 14,
    "INNER_BYTES": 15,
    "STRIDE0": 16,
    "STRIDE1": 17,
    "STRIDE2": 18,
    "ITERATION0": 19,
    "ITERATION1": 20,
    "ITERATION2": 21,
    "COMPACT_BYTES": 22,
    "ENVELOPE_BYTES": 23,
    "OUTPUT_BYTES": 24,
    "SAMPLE": 25,
    "CT_INSTRUCTIONS": 26,
    "NE_INSTRUCTIONS": 27,
    "RDMA_INSTRUCTIONS": 28,
    "WDMA_INSTRUCTIONS": 29,
    "TDMA_INSTRUCTIONS": 30,
    "RESOURCE_BYTES": 31,
    "PAYLOAD_DATA_OFFSET": 32,
    "OUTPUT_DATA_OFFSET": 33,
    "WORKER_A": 34,
    "WORKER_B": 35,
    "ROUNDS": 36,
    "BUFFER_COUNT": 37,
    "ISSUE_ORDER": 38,
    "GUARD": 39,
}
REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "CASE": 3,
    "KIND": 4,
    "ENGINE_A": 5,
    "ENGINE_B": 6,
    "SCHEDULE": 7,
    "EFFECT": 8,
    "RELATION": 9,
    "ORACLE": 10,
    "FORMAT": 11,
    "SRC_DDR_OFFSET": 12,
    "DST_DDR_OFFSET": 13,
    "SPM_A": 14,
    "SPM_B": 15,
    "INNER_BYTES": 16,
    "STRIDE0": 17,
    "STRIDE1": 18,
    "STRIDE2": 19,
    "ITERATION0": 20,
    "ITERATION1": 21,
    "ITERATION2": 22,
    "COMPACT_BYTES": 23,
    "ENVELOPE_BYTES": 24,
    "OUTPUT_BYTES": 25,
    "SAMPLE": 26,
    "SPM_GUARD_MISMATCHES": 27,
    "CT_INST_DELTA": 28,
    "NE_INST_DELTA": 29,
    "RDMA_INST_DELTA": 30,
    "WDMA_INST_DELTA": 31,
    "TDMA_INST_DELTA": 32,
    "CT_EXEC_DELTA": 33,
    "NE_EXEC_DELTA": 34,
    "RDMA_EXEC_DELTA": 35,
    "WDMA_EXEC_DELTA": 36,
    "TDMA_EXEC_DELTA": 37,
    "CT_BLOCKING_DELTA": 38,
    "NE_BLOCKING_DELTA": 39,
    "RDMA_BLOCKING_DELTA": 40,
    "WDMA_BLOCKING_DELTA": 41,
    "TDMA_BLOCKING_DELTA": 42,
    "OUTPUT_DATA_OFFSET": 43,
    "REQUEST_GUARD": 44,
    "FLAGS": 45,
    "FULL_EXEC_DELTA": 46,
    "PLAN_CYCLES": 47,
    "PMU_ENABLE": 48,
    "SERIAL_MODE": 49,
    "STABLE_BEFORE": 50,
    "STABLE_AFTER": 51,
    "WORKER_A": 52,
    "WORKER_B": 53,
    "ROUNDS": 54,
    "BUFFER_COUNT": 55,
    "ISSUE_ORDER": 56,
    "LANE_A_WORKER_INST_DELTA": 57,
    "LANE_B_WORKER_INST_DELTA": 58,
    "LANE_A_WORKER_BLOCKING_DELTA": 59,
    "LANE_B_WORKER_BLOCKING_DELTA": 60,
    "WORKER_MASK": 61,
    "CONTROL_FINAL": 62,
    "RECORD_GUARD": 63,
}
CONFLICT_REQ = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "COORDINATE": 2,
    "SERIAL_CASE": 3,
    "WINDOW_CASE": 4,
    "SERIAL_REQUEST_WORD": 5,
    "WINDOW_REQUEST_WORD": 6,
    "SPM_A": 7,
    "SPM_B": 8,
    "TRANSFER_BYTES": 9,
    "ISSUE_ORDER": 10,
    "SAMPLE": 11,
    "RESOURCE_BYTES": 12,
    "SERIAL_OUTPUT_OFFSET": 13,
    "WINDOW_OUTPUT_OFFSET": 14,
    "OUTPUT_ROW_BYTES": 15,
    "RECORD_META_WORD": 16,
    "FIRST_SCHEDULE": 17,
    "ROWS": 18,
    "GUARD": 19,
}
CONFLICT_REC = {
    "MAGIC": 0,
    "SCHEMA_AND_WORDS": 1,
    "STATUS": 2,
    "COORDINATE": 3,
    "INNER_CASE": 4,
    "SCHEDULE": 5,
    "ISSUE_ORDER": 6,
    "SAMPLE": 7,
    "EXECUTION_ORDINAL": 8,
    "FIRST_SCHEDULE": 9,
    "REQUEST_DDR": 10,
    "INNER_REQUEST_DDR": 11,
    "PAYLOAD_DDR": 12,
    "OUTPUT_DDR": 13,
    "ROW_OUTPUT_DDR": 14,
    "CT_INPUT0_DDR": 15,
    "CT_INPUT1_DDR": 16,
    "RDMA_INPUT_DDR": 17,
    "RESULT_A_DDR": 18,
    "RESULT_B_DDR": 19,
    "SPM_A": 20,
    "SPM_B": 21,
    "WINDOW_FLAGS": 22,
    "RECORD_GUARD": 23,
}


@dataclasses.dataclass(frozen=True)
class Descriptor:
    inner_bytes: int
    stride0: int = 0
    stride1: int = 0
    stride2: int = 0
    iteration0: int = 1
    iteration1: int = 1
    iteration2: int = 1

    @property
    def compact_bytes(self) -> int:
        return (
            self.inner_bytes
            * self.iteration0
            * self.iteration1
            * self.iteration2
        )

    @property
    def envelope_bytes(self) -> int:
        return (
            self.inner_bytes
            + self.stride0 * (self.iteration0 - 1)
            + self.stride1 * (self.iteration1 - 1)
            + self.stride2 * (self.iteration2 - 1)
        )

    def offsets(self) -> tuple[int, ...]:
        return tuple(
            outer * self.stride2
            + middle * self.stride1
            + inner * self.stride0
            for outer in range(self.iteration2)
            for middle in range(self.iteration1)
            for inner in range(self.iteration0)
        )


@dataclasses.dataclass(frozen=True)
class MemoryCase:
    case_id: int
    name: str
    domain: str
    kind: int
    engine_a: int
    engine_b: int
    schedule: int
    effect: int
    relation: int
    oracle: int
    format: int
    src_ddr_offset: int
    dst_ddr_offset: int
    spm_a: int
    spm_b: int
    descriptor: Descriptor
    output_bytes: int
    expected_counts: tuple[int, int, int, int, int]
    repetitions: int = 1
    sweep: str = "general"
    worker_a: int = 0
    worker_b: int = 0
    rounds: int = 1
    buffer_count: int = 1
    issue_order: int = 0

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
            "id": self.case_id,
            "name": self.name,
            "domain": self.domain,
            "kind": self.kind,
            "engine_a": ENGINE_NAMES.get(self.engine_a, "NONE"),
            "engine_b": ENGINE_NAMES.get(self.engine_b, "NONE"),
            "schedule": (
                "serial"
                if self.schedule == SCHEDULE_SERIAL
                else "window"
            ),
            "effect": EFFECT_NAMES.get(self.effect, "none"),
            "relation": RELATION_NAMES[self.relation],
            "disposition": self.disposition,
            "format": self.format,
            "src_ddr_offset": self.src_ddr_offset,
            "dst_ddr_offset": self.dst_ddr_offset,
            "spm_a": self.spm_a,
            "spm_b": self.spm_b,
            "relative_spm_offset": self.spm_b - self.spm_a,
            "spm_base_phase_mod_256": self.spm_a % 256,
            "sweep": self.sweep,
            "workers": [self.worker_a, self.worker_b],
            "rounds": self.rounds,
            "buffer_count": self.buffer_count,
            "issue_order": self.issue_order,
            "descriptor": dataclasses.asdict(self.descriptor),
            "compact_bytes": self.descriptor.compact_bytes,
            "envelope_bytes": self.descriptor.envelope_bytes,
            "output_bytes": self.output_bytes,
            "expected_counts": self.expected_counts,
            "repetitions": self.repetitions,
            "oracle": (
                "exact-result+descriptor-holes+SPM/output-guards"
                if self.is_exact
                else "raw-result+PMU-counts+SPM/output-guards"
            ),
        }


def _counts(*engines: int) -> tuple[int, int, int, int, int]:
    result = [0, 0, 0, 0, 0]
    for engine in engines:
        result[engine] += 1
    return tuple(result)  # type: ignore[return-value]


def _dma_case(
    case_id: int,
    name: str,
    descriptor: Descriptor,
    src_offset: int = 0,
    dst_offset: int = 0,
    format: int = FMT_UINT8,
) -> MemoryCase:
    return MemoryCase(
        case_id,
        name,
        "dma-ddr-descriptor",
        KIND_DMA,
        ENGINE_RDMA,
        ENGINE_WDMA,
        SCHEDULE_SERIAL,
        EFFECT_RAW,
        RELATION_EXACT,
        ORACLE_EXACT,
        format,
        src_offset,
        dst_offset,
        SPM_BASE,
        SPM_BASE,
        descriptor,
        descriptor.envelope_bytes,
        _counts(ENGINE_RDMA, ENGINE_WDMA),
    )


DMA_CASES = (
    _dma_case(
        0, "dma-src-offset256-dst-offset4096-tail4097-u8",
        Descriptor(4097), 256, 4096,
    ),
    _dma_case(
        1, "dma-src-offset4096-dst-offset256-tail8193-u8",
        Descriptor(8193), 4096, 256,
    ),
    _dma_case(
        2, "dma-src-offset65536-dst-offset32768-tail16385-fp16",
        Descriptor(32770), 65536, 32768, FMT_FP16,
    ),
    _dma_case(3, "dma-default-burst-boundary-255", Descriptor(255)),
    _dma_case(4, "dma-default-burst-boundary-256", Descriptor(256)),
    _dma_case(5, "dma-default-burst-boundary-257", Descriptor(257)),
    _dma_case(6, "ddr-large-contiguous-65536", Descriptor(65536)),
    _dma_case(
        7, "ddr-large-1d-stride-holes",
        Descriptor(64, stride0=128, iteration0=512),
    ),
    _dma_case(
        8, "ddr-large-2d-stride-holes",
        Descriptor(
            64, stride0=128, stride1=2048, iteration0=8, iteration1=64
        ),
    ),
    _dma_case(
        9, "ddr-large-3d-stride-holes",
        Descriptor(
            64,
            stride0=128,
            stride1=1024,
            stride2=4096,
            iteration0=4,
            iteration1=4,
            iteration2=32,
        ),
    ),
    _dma_case(10, "ddr-default-burst-boundary-4095", Descriptor(4095)),
    _dma_case(11, "ddr-default-burst-boundary-4096", Descriptor(4096)),
    _dma_case(12, "ddr-default-burst-boundary-4097", Descriptor(4097)),
    _dma_case(13, "ddr-large-tail-65535", Descriptor(65535)),
)


def _engine_case(case_id: int, engine: int) -> MemoryCase:
    transfer = 256 if engine == ENGINE_NE else 4096
    return MemoryCase(
        case_id,
        f"spm-engine-access-{ENGINE_NAMES[engine].lower()}",
        "spm-engine-access",
        KIND_ENGINE_ACCESS,
        engine,
        ENGINE_NONE,
        SCHEDULE_SERIAL,
        EFFECT_NONE,
        RELATION_DISJOINT,
        ORACLE_EXACT,
        FMT_FP16 if engine in {ENGINE_CT, ENGINE_NE} else FMT_UINT8,
        0,
        0,
        SPM_BASE,
        0,
        Descriptor(transfer),
        transfer,
        _counts(engine),
    )


ENGINE_ACCESS_CASES = tuple(
    _engine_case(14 + engine, engine) for engine in range(5)
)


def _relation_engines(
    effect: int, relation: int
) -> tuple[int, int]:
    if relation == RELATION_STRIDED:
        return ENGINE_TDMA, ENGINE_TDMA
    return {
        EFFECT_RAW: (ENGINE_RDMA, ENGINE_WDMA),
        EFFECT_WAR: (ENGINE_WDMA, ENGINE_RDMA),
        EFFECT_WAW: (ENGINE_RDMA, ENGINE_RDMA),
        EFFECT_RAR: (ENGINE_WDMA, ENGINE_WDMA),
    }[effect]


def _relation_case(
    case_id: int, effect: int, relation: int
) -> MemoryCase:
    descriptor = (
        Descriptor(128, stride0=256, iteration0=16)
        if relation == RELATION_STRIDED
        else Descriptor(2048)
    )
    address_b = {
        RELATION_EXACT: SPM_BASE,
        RELATION_PARTIAL: SPM_BASE + 1024,
        RELATION_ADJACENT: SPM_BASE + 2048,
        RELATION_DISJOINT: SPM_BASE + 8192,
        RELATION_STRIDED: SPM_BASE + 128,
    }[relation]
    engines = _relation_engines(effect, relation)
    return MemoryCase(
        case_id,
        f"spm-relation-{EFFECT_NAMES[effect].lower()}-"
        f"{RELATION_NAMES[relation]}",
        "spm-address-relation",
        KIND_ADDRESS_RELATION,
        engines[0],
        engines[1],
        SCHEDULE_WINDOW,
        effect,
        relation,
        ORACLE_OBSERVATION,
        FMT_UINT8,
        0,
        0,
        SPM_BASE,
        address_b,
        descriptor,
        16384,
        _counts(*engines),
    )


RELATION_CASES = tuple(
    _relation_case(
        19 + effect_index * 5 + relation_index, effect, relation
    )
    for effect_index, effect in enumerate(
        (EFFECT_RAW, EFFECT_WAR, EFFECT_WAW, EFFECT_RAR)
    )
    for relation_index, relation in enumerate(
        (
            RELATION_EXACT,
            RELATION_PARTIAL,
            RELATION_ADJACENT,
            RELATION_DISJOINT,
            RELATION_STRIDED,
        )
    )
)


def _pair_case(
    case_id: int,
    engine_a: int,
    engine_b: int,
    schedule: int,
    relative_offset: int,
) -> MemoryCase:
    pair_name = (
        f"spm-bank-pair-{ENGINE_NAMES[engine_a].lower()}-"
        f"{ENGINE_NAMES[engine_b].lower()}"
    )
    schedule_name = (
        "serial" if schedule == SCHEDULE_SERIAL else "window"
    )
    name = (
        f"{pair_name}-{schedule_name}"
        if relative_offset == 8192
        else f"{pair_name}-offset-{relative_offset}-{schedule_name}"
    )
    return MemoryCase(
        case_id,
        name,
        "spm-bank-engine-pair",
        KIND_ENGINE_PAIR,
        engine_a,
        engine_b,
        schedule,
        EFFECT_NONE,
        RELATION_DISJOINT,
        ORACLE_EXACT,
        FMT_UINT8,
        0,
        0,
        SPM_BASE,
        SPM_BASE + relative_offset,
        Descriptor(256),
        512,
        _counts(engine_a, engine_b),
        repetitions=SPM_BANK_PMU_REPETITIONS,
        sweep="general-offset",
    )


GENERAL_PAIR_CASES = tuple(
    _pair_case(
        39 + offset_index * 20 + pair_index * 2 + schedule,
        engine_a,
        engine_b,
        schedule,
        relative_offset,
    )
    for offset_index, relative_offset in enumerate(
        SPM_GENERAL_PAIR_RELATIVE_OFFSETS
    )
    for pair_index, (engine_a, engine_b) in enumerate(
        (left, right)
        for left in range(5)
        for right in range(left + 1, 5)
    )
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)


def _focused_pair_case(
    case_id: int,
    relative_offset: int,
    base_phase: int,
    schedule: int,
    sweep: str,
) -> MemoryCase:
    engine_a, engine_b = SPM_PARALLEL_ENGINES
    schedule_name = (
        "serial" if schedule == SCHEDULE_SERIAL else "window"
    )
    if sweep == "bank-period":
        name = (
            "spm-bank-period-ct-rdma-"
            f"offset-{relative_offset}-{schedule_name}"
        )
    else:
        name = (
            "spm-parallel-alignment-ct-rdma-"
            f"base-mod256-{base_phase}-{schedule_name}"
        )
    return MemoryCase(
        case_id,
        name,
        "spm-bank-engine-pair",
        KIND_ENGINE_PAIR,
        engine_a,
        engine_b,
        schedule,
        EFFECT_NONE,
        RELATION_DISJOINT,
        ORACLE_EXACT,
        FMT_UINT8,
        0,
        0,
        SPM_BASE + base_phase,
        SPM_BASE + base_phase + relative_offset,
        Descriptor(256),
        512,
        _counts(engine_a, engine_b),
        repetitions=SPM_BANK_PMU_REPETITIONS,
        sweep=sweep,
    )


_GENERAL_CASE_COUNT = (
    len(DMA_CASES)
    + len(ENGINE_ACCESS_CASES)
    + len(RELATION_CASES)
    + len(GENERAL_PAIR_CASES)
)
_ADDITIONAL_BANK_OFFSETS = tuple(
    offset
    for offset in SPM_BANK_PERIOD_RELATIVE_OFFSETS
    if offset not in SPM_GENERAL_PAIR_RELATIVE_OFFSETS
)
BANK_PERIOD_PAIR_CASES = tuple(
    _focused_pair_case(
        _GENERAL_CASE_COUNT + offset_index * 2 + schedule,
        relative_offset,
        0,
        schedule,
        "bank-period",
    )
    for offset_index, relative_offset in enumerate(_ADDITIONAL_BANK_OFFSETS)
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)
_ALIGNMENT_CASE_BASE = _GENERAL_CASE_COUNT + len(BANK_PERIOD_PAIR_CASES)
ALIGNMENT_PAIR_CASES = tuple(
    _focused_pair_case(
        _ALIGNMENT_CASE_BASE + phase_index * 2 + schedule,
        8192,
        base_phase,
        schedule,
        "alignment-phase",
    )
    for phase_index, base_phase in enumerate(
        phase
        for phase in SPM_PARALLEL_ALIGNMENT_PHASES
        if phase != 0
    )
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)
PAIR_CASES = (
    GENERAL_PAIR_CASES + BANK_PERIOD_PAIR_CASES + ALIGNMENT_PAIR_CASES
)


def _parallel_pair_transfer(engine_a: int, engine_b: int) -> int:
    if ENGINE_NE in (engine_a, engine_b) or ENGINE_CT in (
        engine_a,
        engine_b,
    ):
        return 16384
    return 65536


def _parallel_pair_case(
    case_id: int,
    engine_a: int,
    engine_b: int,
    schedule: int,
    *,
    transfer: int,
    relation: int = RELATION_DISJOINT,
    effect: int = EFFECT_NONE,
    worker_a: int = 0,
    worker_b: int = 0,
    rounds: int = PERF_MAX_ROUNDS,
    buffer_count: int = PERF_MAX_BUFFERS,
    issue_order: int = 0,
    sweep: str = "sustained-pair",
) -> MemoryCase:
    schedule_name = (
        "serial" if schedule == SCHEDULE_SERIAL else "window"
    )
    order_name = "a-b" if issue_order == 0 else "b-a"
    relation_name = RELATION_NAMES[relation]
    worker_name = (
        f"workers-{worker_a}-{worker_b}"
        if worker_a != worker_b
        else f"worker-{worker_a}"
    )
    if relation == RELATION_DISJOINT:
        spm_b = PERF_DISJOINT_B
    else:
        # RDMA lane A writes at WRITE; WDMA lane B reads at READ0.
        spm_b = (
            PERF_DISJOINT_A
            + PERF_WRITE_OFFSET
            - PERF_READ0_OFFSET
        )
        if relation == RELATION_PARTIAL:
            spm_b += transfer // 2
    return MemoryCase(
        case_id,
        (
            f"parallel-{ENGINE_NAMES[engine_a].lower()}-"
            f"{ENGINE_NAMES[engine_b].lower()}-{transfer // 1024}k-"
            f"{relation_name}-{order_name}-{worker_name}-{schedule_name}"
        ),
        "multi-engine-parallel-window",
        KIND_PARALLEL_PAIR,
        engine_a,
        engine_b,
        schedule,
        effect,
        relation,
        ORACLE_EXACT,
        (
            FMT_FP16
            if ENGINE_CT in (engine_a, engine_b)
            or ENGINE_NE in (engine_a, engine_b)
            else FMT_UINT8
        ),
        0,
        0,
        PERF_DISJOINT_A,
        spm_b,
        Descriptor(transfer),
        2 * rounds * transfer,
        tuple(
            rounds * value
            for value in _counts(engine_a, engine_b)
        ),
        repetitions=SPM_BANK_PMU_REPETITIONS,
        sweep=sweep,
        worker_a=worker_a,
        worker_b=worker_b,
        rounds=rounds,
        buffer_count=buffer_count,
        issue_order=issue_order,
    )


_PARALLEL_CASE_BASE = (
    len(DMA_CASES)
    + len(ENGINE_ACCESS_CASES)
    + len(RELATION_CASES)
    + len(PAIR_CASES)
)
_PARALLEL_ENGINE_PAIRS = tuple(
    (left, right)
    for left in range(5)
    for right in range(left + 1, 5)
)
SUSTAINED_PARALLEL_PAIR_CASES = tuple(
    _parallel_pair_case(
        _PARALLEL_CASE_BASE + pair_index * 2 + schedule,
        engine_a,
        engine_b,
        schedule,
        transfer=_parallel_pair_transfer(engine_a, engine_b),
    )
    for pair_index, (engine_a, engine_b) in enumerate(
        _PARALLEL_ENGINE_PAIRS
    )
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)
_CROSS_WORKER_CASE_BASE = (
    _PARALLEL_CASE_BASE + len(SUSTAINED_PARALLEL_PAIR_CASES)
)
CROSS_WORKER_PARALLEL_PAIR_CASES = tuple(
    _parallel_pair_case(
        _CROSS_WORKER_CASE_BASE + schedule,
        ENGINE_CT,
        ENGINE_RDMA,
        schedule,
        transfer=16384,
        worker_a=0,
        worker_b=1,
        sweep="cross-worker",
    )
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)
_DEPENDENCY_CASE_BASE = (
    _CROSS_WORKER_CASE_BASE + len(CROSS_WORKER_PARALLEL_PAIR_CASES)
)
DEPENDENCY_PARALLEL_PAIR_CASES = tuple(
    _parallel_pair_case(
        _DEPENDENCY_CASE_BASE
        + relation_index * 4
        + issue_order * 2
        + schedule,
        ENGINE_RDMA,
        ENGINE_WDMA,
        schedule,
        transfer=4096,
        relation=relation,
        effect=EFFECT_RAW if issue_order == 0 else EFFECT_WAR,
        rounds=1,
        buffer_count=1,
        issue_order=issue_order,
        sweep="dependency-control",
    )
    for relation_index, relation in enumerate(
        (RELATION_DISJOINT, RELATION_EXACT, RELATION_PARTIAL)
    )
    for issue_order in (0, 1)
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)
PARALLEL_PAIR_CASES = (
    SUSTAINED_PARALLEL_PAIR_CASES
    + CROSS_WORKER_PARALLEL_PAIR_CASES
    + DEPENDENCY_PARALLEL_PAIR_CASES
)

CATALOG = (
    DMA_CASES
    + ENGINE_ACCESS_CASES
    + RELATION_CASES
    + PAIR_CASES
    + PARALLEL_PAIR_CASES
)
CASES_BY_ID = {case.case_id: case for case in CATALOG}
CASES_BY_NAME = {case.name: case for case in CATALOG}
_CONFLICT_EQUIVALENCE_COORDINATES = tuple(
    (
        translation,
        phase,
        transfer,
        issue_order,
        schedule,
    )
    for translation in (0, *SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS)
    for phase in SPM_PARALLEL_ALIGNMENT_PHASES
    for transfer in SPM_CONFLICT_EQUIVALENCE_TRANSFERS
    for issue_order in SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)
_PENDING_CONFLICT_EQUIVALENCE_COORDINATES = tuple(
    coordinate
    for coordinate in _CONFLICT_EQUIVALENCE_COORDINATES
    if not (
        coordinate[0] == 0
        and coordinate[2] == 256
        and coordinate[3] == 0
    )
)
PENDING_CONFLICT_EQUIVALENCE_CASES = tuple(
    MemoryCase(
        len(CATALOG) + pending_index,
        (
            "spm-conflict-equivalence-ct-rdma-"
            f"translation-{translation}-phase-{phase}-"
            f"bytes-{transfer}-"
            f"{'a-b' if issue_order == 0 else 'b-a'}-"
            f"{'serial' if schedule == SCHEDULE_SERIAL else 'window'}"
        ),
        "spm-bank-engine-pair",
        KIND_ENGINE_PAIR,
        ENGINE_CT,
        ENGINE_RDMA,
        schedule,
        EFFECT_NONE,
        RELATION_DISJOINT,
        ORACLE_EXACT,
        FMT_UINT8,
        0,
        0,
        SPM_BASE + translation + phase,
        SPM_BASE + translation + phase + 8192,
        Descriptor(transfer),
        2 * transfer,
        _counts(ENGINE_CT, ENGINE_RDMA),
        repetitions=SPM_BANK_PMU_REPETITIONS,
        sweep="conflict-equivalence",
        issue_order=issue_order,
    )
    for pending_index, (
        translation,
        phase,
        transfer,
        issue_order,
        schedule,
    ) in enumerate(
        _PENDING_CONFLICT_EQUIVALENCE_COORDINATES
    )
)
PENDING_CASES = PENDING_CONFLICT_EQUIVALENCE_CASES
CONFLICT_EQUIVALENCE_BASELINE_CASES = tuple(
    case
    for case in PAIR_CASES
    if (
        (case.engine_a, case.engine_b) == SPM_PARALLEL_ENGINES
        and case.spm_b - case.spm_a == 8192
        and case.spm_a - SPM_BASE in SPM_PARALLEL_ALIGNMENT_PHASES
    )
)
CONFLICT_EQUIVALENCE_CASES = (
    CONFLICT_EQUIVALENCE_BASELINE_CASES
    + PENDING_CONFLICT_EQUIVALENCE_CASES
)
ALL_CASES = CATALOG + PENDING_CASES
ALL_CASES_BY_ID = {case.case_id: case for case in ALL_CASES}
ALL_CASES_BY_NAME = {case.name: case for case in ALL_CASES}


@dataclasses.dataclass(frozen=True)
class ConflictEquivalencePair:
    coordinate_id: int
    translation: int
    phase: int
    transfer_bytes: int
    issue_order: int
    serial: MemoryCase
    window: MemoryCase

    @property
    def name(self) -> str:
        return (
            "spm-conflict-equivalence-ct-rdma-"
            f"translation-{self.translation}-phase-{self.phase}-"
            f"bytes-{self.transfer_bytes}-"
            f"{'a-b' if self.issue_order == 0 else 'b-a'}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.coordinate_id,
            "name": self.name,
            "translation": self.translation,
            "base_phase_mod_256": self.phase,
            "relative_spm_offset": self.serial.spm_b - self.serial.spm_a,
            "transfer_bytes": self.transfer_bytes,
            "issue_order": "a-b" if self.issue_order == 0 else "b-a",
            "samples": CONFLICT_SAMPLES,
            "schedules": {
                "serial": self.serial.name,
                "window": self.window.name,
            },
            "execution": (
                "one invocation per sample with paired serial/window rows "
                "sharing request, payload, and output allocations"
            ),
        }


def _conflict_pair_key(
    case: MemoryCase,
) -> tuple[int, int, int, int]:
    phase = case.spm_a % 256
    return (
        case.spm_a - SPM_BASE - phase,
        phase,
        case.descriptor.compact_bytes,
        case.issue_order,
    )


def _conflict_equivalence_pairs() -> tuple[ConflictEquivalencePair, ...]:
    grouped: dict[
        tuple[int, int, int, int], dict[int, MemoryCase]
    ] = {}
    for case in CONFLICT_EQUIVALENCE_CASES:
        schedules = grouped.setdefault(_conflict_pair_key(case), {})
        if case.schedule in schedules:
            raise RuntimeError(
                f"{case.name}: duplicate conflict-equivalence schedule"
            )
        schedules[case.schedule] = case
    result = []
    for coordinate_id, key in enumerate(sorted(grouped)):
        schedules = grouped[key]
        if set(schedules) != {SCHEDULE_SERIAL, SCHEDULE_WINDOW}:
            raise RuntimeError(
                f"conflict-equivalence coordinate {key} lacks paired controls"
            )
        translation, phase, transfer_bytes, issue_order = key
        result.append(
            ConflictEquivalencePair(
                coordinate_id,
                translation,
                phase,
                transfer_bytes,
                issue_order,
                schedules[SCHEDULE_SERIAL],
                schedules[SCHEDULE_WINDOW],
            )
        )
    return tuple(result)


CONFLICT_EQUIVALENCE_PAIRS = _conflict_equivalence_pairs()


@dataclasses.dataclass(frozen=True)
class StaticBoundary:
    name: str
    disposition: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


STATIC_BOUNDARIES = (
    StaticBoundary(
        "ddr-configurable-burst-knob",
        "static-negative",
        (
            "the owned RDMA/WDMA CRT exposes no configurable burst field; "
            "the executable rows only cross default burst boundaries"
        ),
    ),
)


def _pattern(seed: int, salt: int, length: int) -> bytes:
    return bytes(
        (
            seed * 31
            + salt * 47
            + index * 17
            + (index >> 7) * 11
        )
        & 0xFF
        for index in range(length)
    )


def _scatter(
    destination: bytearray,
    base: int,
    descriptor: Descriptor,
    compact: bytes,
) -> None:
    cursor = 0
    for offset in descriptor.offsets():
        destination[
            base + offset : base + offset + descriptor.inner_bytes
        ] = compact[cursor : cursor + descriptor.inner_bytes]
        cursor += descriptor.inner_bytes
    if cursor != len(compact):
        raise RuntimeError("descriptor compact scatter did not consume input")


def _engine_payload_expected(
    engine: int, seed: int, lane: int, transfer: int
) -> tuple[bytes, bytes, int]:
    payload = bytearray([RESOURCE_CANARY] * 12288)
    if engine == ENGINE_CT:
        payload[:transfer] = struct.pack("<H", 0x3C00) * (transfer // 2)
        payload[4096 : 4096 + transfer] = (
            struct.pack("<H", 0x4000) * (transfer // 2)
        )
        return bytes(payload), struct.pack("<H", 0x4200) * (transfer // 2), transfer
    if engine == ENGINE_NE:
        lhs = tuple(struct.pack("<e", float(1 + (index + seed) % 15)) for index in range(16))
        payload[:32] = b"".join(lhs)
        identity = bytearray(512)
        for index in range(16):
            identity[(index * 16 + index) * 2 : (index * 16 + index + 1) * 2] = struct.pack("<H", 0x3C00)
        payload[4096:4608] = identity
        return bytes(payload), b"".join(lhs), 32
    if engine == ENGINE_RDMA:
        value = _pattern(seed, 20 + lane, transfer)
        payload[8192 : 8192 + transfer] = value
        return bytes(payload), value, transfer
    value = _pattern(seed, 30 + engine * 3 + lane, transfer)
    payload[:transfer] = value
    return bytes(payload), value, transfer


@dataclasses.dataclass(frozen=True)
class CasePayload:
    request: bytes
    payload: bytes
    expected_output: bytes
    allowed_ranges: tuple[tuple[int, int], ...]
    exact_ranges: tuple[tuple[int, int], ...]


@dataclasses.dataclass(frozen=True)
class ConflictEquivalenceInvocation:
    pair: ConflictEquivalencePair
    sample: int
    first_schedule: int
    request: bytes
    payload: bytes
    serial_built: CasePayload
    window_built: CasePayload


def validate_pair_case(case: MemoryCase) -> None:
    if case.kind != KIND_ENGINE_PAIR:
        return
    relative_offset = case.spm_b - case.spm_a
    phase = case.spm_a % 256
    general = (
        phase == 0
        and relative_offset in SPM_GENERAL_PAIR_RELATIVE_OFFSETS
    )
    bank_period = (
        (case.engine_a, case.engine_b) == SPM_PARALLEL_ENGINES
        and phase == 0
        and relative_offset in SPM_BANK_PERIOD_RELATIVE_OFFSETS
    )
    alignment_phase = (
        (case.engine_a, case.engine_b) == SPM_PARALLEL_ENGINES
        and relative_offset == 8192
        and phase in SPM_PARALLEL_ALIGNMENT_PHASES
    )
    conflict_equivalence = (
        case.sweep == "conflict-equivalence"
        and alignment_phase
        and case.descriptor.compact_bytes
        in SPM_CONFLICT_EQUIVALENCE_TRANSFERS
        and case.issue_order in SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
    )
    legacy_pair = (
        case.descriptor == Descriptor(256)
        and case.issue_order == 0
        and (general or bank_period or alignment_phase)
    )
    if (
        case.engine_a >= case.engine_b
        or case.output_bytes != 2 * case.descriptor.compact_bytes
        or not (legacy_pair or conflict_equivalence)
    ):
        raise RuntimeError(
            f"{case.name}: unsupported SPM parallel address coordinate"
        )
    guarded_ranges = tuple(
        (
            base - SPM_GUARD_BYTES,
            base + SPM_PAIR_SLOT_BYTES + SPM_GUARD_BYTES,
        )
        for base in (case.spm_a, case.spm_b)
    )
    if any(
        begin < SPM_ALLOCATABLE_BEGIN
        or end > SPM_ALLOCATABLE_END
        or begin >= end
        for begin, end in guarded_ranges
    ):
        raise RuntimeError(
            f"{case.name}: guarded SPM range leaves the allocatable arena"
        )
    if guarded_ranges[0][1] > guarded_ranges[1][0]:
        raise RuntimeError(
            f"{case.name}: guarded SPM lane ranges overlap"
        )


def _parallel_active_range(
    case: MemoryCase, lane: int
) -> tuple[int, int]:
    engine = (case.engine_a, case.engine_b)[lane]
    base = (case.spm_a, case.spm_b)[lane]
    transfer = case.descriptor.compact_bytes
    if engine == ENGINE_CT:
        return (
            base + PERF_READ0_OFFSET,
            max(
                base
                + PERF_READ1_OFFSET
                + case.buffer_count * transfer,
                base + PERF_WRITE_OFFSET + case.rounds * transfer,
            ),
        )
    if engine == ENGINE_NE:
        return (
            base + PERF_READ0_OFFSET,
            max(
                base + PERF_READ1_OFFSET + 32768,
                base + PERF_WRITE_OFFSET + case.rounds * transfer,
            ),
        )
    if engine == ENGINE_RDMA:
        return (
            base + PERF_WRITE_OFFSET,
            base + PERF_WRITE_OFFSET + case.rounds * transfer,
        )
    if engine == ENGINE_WDMA:
        return (
            base + PERF_READ0_OFFSET,
            base + PERF_READ0_OFFSET + case.buffer_count * transfer,
        )
    if engine == ENGINE_TDMA:
        return (
            base + PERF_WRITE_OFFSET,
            base + PERF_WRITE_OFFSET + case.rounds * transfer,
        )
    raise RuntimeError(f"{case.name}: invalid parallel engine")


def validate_parallel_pair_case(case: MemoryCase) -> None:
    if case.kind != KIND_PARALLEL_PAIR:
        return
    transfer = case.descriptor.compact_bytes
    if (
        case.engine_a == case.engine_b
        or case.engine_a not in ENGINE_NAMES
        or case.engine_b not in ENGINE_NAMES
        or case.schedule not in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
        or case.oracle != ORACLE_EXACT
        or transfer not in (4096, 16384, 65536)
        or not 1 <= case.rounds <= PERF_MAX_ROUNDS
        or not 1 <= case.buffer_count <= PERF_MAX_BUFFERS
        or case.issue_order not in (0, 1)
        or case.worker_a not in range(3)
        or case.worker_b not in range(3)
        or case.output_bytes != 2 * case.rounds * transfer
        or case.expected_counts
        != tuple(
            case.rounds * value
            for value in _counts(case.engine_a, case.engine_b)
        )
    ):
        raise RuntimeError(f"{case.name}: invalid parallel pair contract")
    if case.sweep == "dependency-control":
        if (
            (case.engine_a, case.engine_b)
            != (ENGINE_RDMA, ENGINE_WDMA)
            or transfer != 4096
            or case.rounds != 1
            or case.buffer_count != 1
            or case.relation
            not in (RELATION_DISJOINT, RELATION_EXACT, RELATION_PARTIAL)
            or case.effect
            != (EFFECT_RAW if case.issue_order == 0 else EFFECT_WAR)
        ):
            raise RuntimeError(
                f"{case.name}: invalid dependency control"
            )
    elif (
        case.relation != RELATION_DISJOINT
        or case.effect != EFFECT_NONE
        or case.issue_order != 0
        or case.rounds != PERF_MAX_ROUNDS
        or case.buffer_count != PERF_MAX_BUFFERS
    ):
        raise RuntimeError(f"{case.name}: invalid sustained pair contract")
    active = tuple(_parallel_active_range(case, lane) for lane in range(2))
    if any(
        begin < SPM_ALLOCATABLE_BEGIN + SPM_GUARD_BYTES
        or end + SPM_GUARD_BYTES > SPM_ALLOCATABLE_END
        or begin >= end
        for begin, end in active
    ):
        raise RuntimeError(f"{case.name}: parallel SPM range is invalid")
    if (
        case.relation == RELATION_DISJOINT
        and active[0][1] + SPM_GUARD_BYTES
        > active[1][0] - SPM_GUARD_BYTES
    ):
        raise RuntimeError(
            f"{case.name}: disjoint parallel lanes overlap"
        )


def spm_slot_spans(case: MemoryCase) -> tuple[int, ...]:
    if case.kind == KIND_DMA:
        return (case.descriptor.compact_bytes,)
    if case.kind == KIND_ENGINE_ACCESS:
        return (0x6000 if case.descriptor.compact_bytes > 512 else 0x1000,)
    if case.kind == KIND_ENGINE_PAIR:
        return (SPM_PAIR_SLOT_BYTES, SPM_PAIR_SLOT_BYTES)
    if case.kind == KIND_PARALLEL_PAIR:
        # Parallel rows capture the two exterior guard bands, not a
        # multi-megabyte full arena dump.
        return (0, 0)
    return (case.output_bytes,)


def output_result_ranges(
    case: MemoryCase,
) -> tuple[tuple[int, int], ...]:
    if case.kind == KIND_DMA:
        begin = OUTPUT_DATA_OFFSET + case.dst_ddr_offset
        return ((begin, begin + case.descriptor.envelope_bytes),)
    if case.kind in {KIND_ENGINE_ACCESS, KIND_ENGINE_PAIR}:
        engines = (
            (case.engine_a,)
            if case.kind == KIND_ENGINE_ACCESS
            else (case.engine_a, case.engine_b)
        )
        transfer = (
            case.descriptor.compact_bytes
        )
        ranges: list[tuple[int, int]] = []
        for lane, engine in enumerate(engines):
            output_begin = OUTPUT_DATA_OFFSET + (
                lane * transfer if case.kind == KIND_ENGINE_PAIR else 0
            )
            ranges.append((output_begin, output_begin + transfer))
            if engine == ENGINE_WDMA:
                sink = (
                    OUTPUT_SINK0_OFFSET
                    if lane == 0
                    else OUTPUT_SINK1_OFFSET
                )
                ranges.append((sink, sink + transfer))
        return tuple(ranges)
    if case.kind == KIND_PARALLEL_PAIR:
        transfer = case.descriptor.compact_bytes
        ranges = []
        for lane, engine in enumerate((case.engine_a, case.engine_b)):
            result_begin = (
                PERF_OUTPUT_RESULT_BASE
                + lane * PERF_OUTPUT_LANE_STRIDE
            )
            ranges.append(
                (result_begin, result_begin + case.rounds * transfer)
            )
            if engine == ENGINE_WDMA:
                sink_begin = (
                    PERF_OUTPUT_SINK_BASE
                    + lane * PERF_OUTPUT_LANE_STRIDE
                )
                ranges.append(
                    (sink_begin, sink_begin + case.rounds * transfer)
                )
        return tuple(ranges)
    ranges = [
        (OUTPUT_DATA_OFFSET, OUTPUT_DATA_OFFSET + case.output_bytes)
    ]
    if case.engine_a == ENGINE_WDMA:
        ranges.append((OUTPUT_SINK0_OFFSET, OUTPUT_SINK0_OFFSET + 4096))
    if case.engine_b == ENGINE_WDMA:
        ranges.append((OUTPUT_SINK1_OFFSET, OUTPUT_SINK1_OFFSET + 4096))
    return tuple(ranges)


def _align_output_offset(offset: int) -> int:
    return (
        offset + OUTPUT_LAYOUT_ALIGNMENT - 1
    ) // OUTPUT_LAYOUT_ALIGNMENT * OUTPUT_LAYOUT_ALIGNMENT


def spm_dump_ranges(case: MemoryCase) -> tuple[tuple[int, int], ...]:
    if case.kind == KIND_PARALLEL_PAIR:
        return tuple(
            (
                PERF_OUTPUT_GUARD_BASE
                + lane * PERF_OUTPUT_GUARD_LANE_STRIDE,
                PERF_OUTPUT_GUARD_BASE
                + lane * PERF_OUTPUT_GUARD_LANE_STRIDE
                + 2 * SPM_GUARD_BYTES,
            )
            for lane in range(2)
        )
    occupied = output_result_ranges(case)
    cursor = _align_output_offset(
        max(RECORD_WORDS * 8, *(end for _, end in occupied))
    )
    ranges: list[tuple[int, int]] = []
    for span in spm_slot_spans(case):
        end = cursor + span + 2 * SPM_GUARD_BYTES
        if end > RESOURCE_BYTES:
            raise RuntimeError(
                f"{case.name}: dynamic SPM dump leaves output resource"
            )
        ranges.append((cursor, end))
        cursor = _align_output_offset(end)
    return tuple(ranges)


def _perf_payload_offset(
    lane: int, section: int, index: int, transfer: int
) -> int:
    return (
        PERF_PAYLOAD_BASE
        + lane * PERF_PAYLOAD_LANE_STRIDE
        + section
        + index * transfer
    )


def _perf_read0_bytes(
    engine: int, seed: int, lane: int, buffer: int, transfer: int
) -> bytes:
    if engine in (ENGINE_CT, ENGINE_NE):
        value = float(1 + seed + lane + buffer)
        return struct.pack("<e", value) * (transfer // 2)
    return _pattern(seed + buffer, 110 + engine * 7 + lane, transfer)


def _perf_read1_bytes(
    engine: int, seed: int, lane: int, buffer: int, transfer: int
) -> bytes:
    if engine == ENGINE_CT:
        value = float(2 + (seed & 1))
        return struct.pack("<e", value) * (transfer // 2)
    if engine == ENGINE_NE:
        identity = bytearray(32768)
        one = struct.pack("<e", 1.0)
        for index in range(128):
            begin = (index * 128 + index) * 2
            identity[begin : begin + 2] = one
        return bytes(identity)
    return bytes(transfer)


def _perf_engine_result(
    engine: int,
    seed: int,
    lane: int,
    round_index: int,
    buffer: int,
    transfer: int,
) -> bytes:
    if engine == ENGINE_CT:
        lhs = float(1 + seed + lane + buffer)
        rhs = float(2 + (seed & 1))
        return struct.pack("<e", lhs + rhs) * (transfer // 2)
    if engine == ENGINE_NE:
        return _perf_read0_bytes(engine, seed, lane, buffer, transfer)
    if engine == ENGINE_RDMA:
        return _pattern(
            seed + round_index, 170 + lane * 11, transfer
        )
    if engine == ENGINE_WDMA:
        return _perf_read0_bytes(engine, seed, lane, buffer, transfer)
    if engine == ENGINE_TDMA:
        return bytes([(seed + lane + round_index + 1) & 0xFF]) * transfer
    raise RuntimeError("invalid parallel engine")


def _overlay_range(
    destination: bytearray,
    destination_begin: int,
    source: bytes,
    source_begin: int,
) -> None:
    begin = max(destination_begin, source_begin)
    end = min(
        destination_begin + len(destination),
        source_begin + len(source),
    )
    if begin >= end:
        return
    destination[begin - destination_begin : end - destination_begin] = (
        source[begin - source_begin : end - source_begin]
    )


def _build_parallel_payload(
    case: MemoryCase,
    seed: int,
    payload: bytearray,
    expected: bytearray,
    allowed: list[tuple[int, int]],
    exact: list[tuple[int, int]],
) -> None:
    transfer = case.descriptor.compact_bytes
    engines = (case.engine_a, case.engine_b)
    lane_results: list[list[bytes]] = [[], []]
    for lane, engine in enumerate(engines):
        for buffer in range(case.buffer_count):
            read0 = _perf_read0_bytes(
                engine, seed, lane, buffer, transfer
            )
            begin = _perf_payload_offset(
                lane, PERF_PAYLOAD_READ0_OFFSET, buffer, transfer
            )
            payload[begin : begin + len(read0)] = read0
            read1 = _perf_read1_bytes(
                engine, seed, lane, buffer, transfer
            )
            read1_index = 0 if engine == ENGINE_NE else buffer
            if engine == ENGINE_NE and buffer != 0:
                continue
            begin = _perf_payload_offset(
                lane, PERF_PAYLOAD_READ1_OFFSET, read1_index, transfer
            )
            payload[begin : begin + len(read1)] = read1
        for round_index in range(case.rounds):
            buffer = round_index % case.buffer_count
            result = _perf_engine_result(
                engine,
                seed,
                lane,
                round_index,
                buffer,
                transfer,
            )
            lane_results[lane].append(result)
            if engine == ENGINE_RDMA:
                begin = _perf_payload_offset(
                    lane,
                    PERF_PAYLOAD_RDMA_OFFSET,
                    round_index,
                    transfer,
                )
                payload[begin : begin + transfer] = result

    if case.sweep == "dependency-control":
        rdma = lane_results[0][0]
        wdma_initial = lane_results[1][0]
        rdma_begin = case.spm_a + PERF_WRITE_OFFSET
        wdma_begin = case.spm_b + PERF_READ0_OFFSET
        wdma_final = bytearray(wdma_initial)
        _overlay_range(wdma_final, wdma_begin, rdma, rdma_begin)
        lane_results[1][0] = bytes(wdma_final)
        sink_value = (
            bytes(wdma_final)
            if case.issue_order == 0
            else wdma_initial
        )
    else:
        sink_value = b""

    for lane, engine in enumerate(engines):
        result_begin = (
            PERF_OUTPUT_RESULT_BASE + lane * PERF_OUTPUT_LANE_STRIDE
        )
        result = b"".join(lane_results[lane])
        expected[result_begin : result_begin + len(result)] = result
        allowed.append((result_begin, result_begin + len(result)))
        exact.append((result_begin, result_begin + len(result)))
        if engine == ENGINE_WDMA:
            sink_begin = (
                PERF_OUTPUT_SINK_BASE + lane * PERF_OUTPUT_LANE_STRIDE
            )
            if case.sweep == "dependency-control":
                sinks = sink_value
            else:
                sinks = b"".join(
                    _perf_engine_result(
                        engine,
                        seed,
                        lane,
                        round_index,
                        round_index % case.buffer_count,
                        transfer,
                    )
                    for round_index in range(case.rounds)
                )
            expected[sink_begin : sink_begin + len(sinks)] = sinks
            allowed.append((sink_begin, sink_begin + len(sinks)))
            exact.append((sink_begin, sink_begin + len(sinks)))

    for begin, end in spm_dump_ranges(case):
        expected[begin:end] = bytes([SPM_GUARD]) * (end - begin)


def build_case_payload(case: MemoryCase, sample: int = 0) -> CasePayload:
    validate_pair_case(case)
    validate_parallel_pair_case(case)
    seed = sample + 1
    payload = bytearray([RESOURCE_CANARY] * RESOURCE_BYTES)
    expected = bytearray([RESOURCE_CANARY] * RESOURCE_BYTES)
    allowed: list[tuple[int, int]] = []
    exact: list[tuple[int, int]] = []
    payload[
        PAYLOAD_SPM_SEED_OFFSET :
        PAYLOAD_SPM_SEED_OFFSET + PAYLOAD_SPM_SEED_BYTES
    ] = bytes([SPM_GUARD]) * PAYLOAD_SPM_SEED_BYTES

    if case.kind == KIND_DMA:
        compact = _pattern(seed, 1, case.descriptor.compact_bytes)
        _scatter(
            payload,
            PAYLOAD_DATA_OFFSET + case.src_ddr_offset,
            case.descriptor,
            compact,
        )
        _scatter(
            expected,
            OUTPUT_DATA_OFFSET + case.dst_ddr_offset,
            case.descriptor,
            compact,
        )
        begin = OUTPUT_DATA_OFFSET + case.dst_ddr_offset
        end = begin + case.descriptor.envelope_bytes
        allowed.append((begin, end))
        exact.append((begin, end))
    elif case.kind in {KIND_ENGINE_ACCESS, KIND_ENGINE_PAIR}:
        engines = (
            (case.engine_a,)
            if case.kind == KIND_ENGINE_ACCESS
            else (case.engine_a, case.engine_b)
        )
        for lane, engine in enumerate(engines):
            transfer = (
                case.descriptor.compact_bytes
            )
            lane_payload, result, exact_bytes = _engine_payload_expected(
                engine, seed, lane, transfer
            )
            payload_begin = PAYLOAD_DATA_OFFSET + lane * 16384
            payload[
                payload_begin : payload_begin + len(lane_payload)
            ] = lane_payload
            output_begin = OUTPUT_DATA_OFFSET + lane * transfer
            expected[output_begin : output_begin + len(result)] = result
            allowed.append((output_begin, output_begin + transfer))
            exact.append((output_begin, output_begin + exact_bytes))
            if engine == ENGINE_WDMA:
                sink = (
                    OUTPUT_SINK0_OFFSET
                    if lane == 0
                    else OUTPUT_SINK1_OFFSET
                )
                expected[sink : sink + transfer] = result[:transfer]
                allowed.append((sink, sink + transfer))
                exact.append((sink, sink + transfer))
    elif case.kind == KIND_PARALLEL_PAIR:
        payload[
            PERF_PAYLOAD_GUARD_SEED_OFFSET :
            PERF_PAYLOAD_GUARD_SEED_OFFSET + 65536
        ] = bytes([SPM_GUARD]) * 65536
        _build_parallel_payload(
            case, seed, payload, expected, allowed, exact
        )
    else:
        relation_seed = bytes(
            (sample * 17 + index * 13 + 5) & 0xFF
            for index in range(case.output_bytes)
        )
        payload[
            PAYLOAD_SPM_SEED_OFFSET + SPM_GUARD_BYTES :
            PAYLOAD_SPM_SEED_OFFSET
            + SPM_GUARD_BYTES
            + case.output_bytes
        ] = relation_seed
        for lane in range(2):
            source = _pattern(seed, 80 + lane, 4096)
            begin = PAYLOAD_DATA_OFFSET + lane * 16384
            payload[begin : begin + len(source)] = source
        if case.relation == RELATION_STRIDED:
            payload[
                PAYLOAD_SPM_SCRATCH0_OFFSET :
                PAYLOAD_SPM_SCRATCH0_OFFSET + 8192
            ] = bytes([0x39]) * 8192
            payload[
                PAYLOAD_SPM_SCRATCH1_OFFSET :
                PAYLOAD_SPM_SCRATCH1_OFFSET + 8192
            ] = bytes([0xC7]) * 8192
        allowed.append(
            (OUTPUT_DATA_OFFSET, OUTPUT_DATA_OFFSET + case.output_bytes)
        )
        if case.engine_a == ENGINE_WDMA:
            allowed.append(
                (OUTPUT_SINK0_OFFSET, OUTPUT_SINK0_OFFSET + 4096)
            )
        if case.engine_b == ENGINE_WDMA:
            allowed.append(
                (OUTPUT_SINK1_OFFSET, OUTPUT_SINK1_OFFSET + 4096)
            )
    result_ranges = output_result_ranges(case)
    if tuple(allowed) != result_ranges:
        raise RuntimeError(f"{case.name}: output range contract drifted")
    allowed.extend(spm_dump_ranges(case))

    words = [0] * REQUEST_WORDS
    counts = case.expected_counts
    values = {
        "MAGIC": REQUEST_MAGIC,
        "SCHEMA_AND_WORDS": (SCHEMA << 32) | REQUEST_WORDS,
        "CASE": case.case_id,
        "KIND": case.kind,
        "ENGINE_A": case.engine_a,
        "ENGINE_B": case.engine_b,
        "SCHEDULE": case.schedule,
        "EFFECT": case.effect,
        "RELATION": case.relation,
        "ORACLE": case.oracle,
        "FORMAT": case.format,
        "SRC_DDR_OFFSET": case.src_ddr_offset,
        "DST_DDR_OFFSET": case.dst_ddr_offset,
        "SPM_A": case.spm_a,
        "SPM_B": case.spm_b,
        "INNER_BYTES": case.descriptor.inner_bytes,
        "STRIDE0": case.descriptor.stride0,
        "STRIDE1": case.descriptor.stride1,
        "STRIDE2": case.descriptor.stride2,
        "ITERATION0": case.descriptor.iteration0,
        "ITERATION1": case.descriptor.iteration1,
        "ITERATION2": case.descriptor.iteration2,
        "COMPACT_BYTES": case.descriptor.compact_bytes,
        "ENVELOPE_BYTES": case.descriptor.envelope_bytes,
        "OUTPUT_BYTES": case.output_bytes,
        "SAMPLE": sample,
        "CT_INSTRUCTIONS": counts[ENGINE_CT],
        "NE_INSTRUCTIONS": counts[ENGINE_NE],
        "RDMA_INSTRUCTIONS": counts[ENGINE_RDMA],
        "WDMA_INSTRUCTIONS": counts[ENGINE_WDMA],
        "TDMA_INSTRUCTIONS": counts[ENGINE_TDMA],
        "RESOURCE_BYTES": RESOURCE_BYTES,
        "PAYLOAD_DATA_OFFSET": PAYLOAD_DATA_OFFSET,
        "OUTPUT_DATA_OFFSET": OUTPUT_DATA_OFFSET,
        "WORKER_A": case.worker_a,
        "WORKER_B": case.worker_b,
        "ROUNDS": case.rounds,
        "BUFFER_COUNT": case.buffer_count,
        "ISSUE_ORDER": case.issue_order,
        "GUARD": REQUEST_GUARD,
    }
    for key, value in values.items():
        words[REQ[key]] = value
    request = struct.pack(f"<{REQUEST_WORDS}Q", *words)
    return CasePayload(
        request
        + bytes([RESOURCE_CANARY]) * (RESOURCE_BYTES - len(request)),
        bytes(payload),
        bytes(expected),
        tuple(allowed),
        tuple(exact),
    )


def build_conflict_equivalence_invocation(
    pair: ConflictEquivalencePair, sample: int
) -> ConflictEquivalenceInvocation:
    if pair not in CONFLICT_EQUIVALENCE_PAIRS:
        raise RuntimeError(
            f"{pair.name}: conflict-equivalence pair is not canonical"
        )
    if sample not in range(CONFLICT_SAMPLES):
        raise RuntimeError(
            f"{pair.name}: conflict-equivalence sample is invalid"
        )
    serial_built = build_case_payload(pair.serial, sample)
    window_built = build_case_payload(pair.window, sample)
    serial_words = struct.unpack_from(
        f"<{REQUEST_WORDS}Q", serial_built.request
    )
    window_words = struct.unpack_from(
        f"<{REQUEST_WORDS}Q", window_built.request
    )
    differing = {
        index
        for index, (serial, window) in enumerate(
            zip(serial_words, window_words, strict=True)
        )
        if serial != window
    }
    if (
        differing != {REQ["CASE"], REQ["SCHEDULE"]}
        or serial_words[REQ["SCHEDULE"]] != SCHEDULE_SERIAL
        or window_words[REQ["SCHEDULE"]] != SCHEDULE_WINDOW
        or serial_built.payload != window_built.payload
    ):
        raise RuntimeError(
            f"{pair.name}: serial/window controls are not a one-factor pair"
        )
    for case, built in (
        (pair.serial, serial_built),
        (pair.window, window_built),
    ):
        row_end = max(
            RECORD_WORDS * 8,
            *(end for _, end in built.allowed_ranges),
        )
        if row_end > CONFLICT_OUTPUT_ROW_BYTES:
            raise RuntimeError(
                f"{case.name}: conflict output row exceeds its slot"
            )

    first_schedule = (
        SCHEDULE_SERIAL
        if (pair.coordinate_id + sample) % 2 == 0
        else SCHEDULE_WINDOW
    )
    request = bytearray(serial_built.request)
    window_begin = CONFLICT_WINDOW_REQUEST_WORD * 8
    request[
        window_begin : window_begin + REQUEST_WORDS * 8
    ] = window_built.request[: REQUEST_WORDS * 8]
    meta = [0] * CONFLICT_REQUEST_META_WORDS
    values = {
        "MAGIC": CONFLICT_REQUEST_MAGIC,
        "SCHEMA_AND_WORDS": (
            CONFLICT_SCHEMA << 32
        ) | CONFLICT_REQUEST_META_WORDS,
        "COORDINATE": pair.coordinate_id,
        "SERIAL_CASE": pair.serial.case_id,
        "WINDOW_CASE": pair.window.case_id,
        "SERIAL_REQUEST_WORD": CONFLICT_SERIAL_REQUEST_WORD,
        "WINDOW_REQUEST_WORD": CONFLICT_WINDOW_REQUEST_WORD,
        "SPM_A": pair.serial.spm_a,
        "SPM_B": pair.serial.spm_b,
        "TRANSFER_BYTES": pair.transfer_bytes,
        "ISSUE_ORDER": pair.issue_order,
        "SAMPLE": sample,
        "RESOURCE_BYTES": RESOURCE_BYTES,
        "SERIAL_OUTPUT_OFFSET": CONFLICT_SERIAL_OUTPUT_OFFSET,
        "WINDOW_OUTPUT_OFFSET": CONFLICT_WINDOW_OUTPUT_OFFSET,
        "OUTPUT_ROW_BYTES": CONFLICT_OUTPUT_ROW_BYTES,
        "RECORD_META_WORD": CONFLICT_RECORD_META_WORD,
        "FIRST_SCHEDULE": first_schedule,
        "ROWS": CONFLICT_ROWS,
        "GUARD": CONFLICT_REQUEST_GUARD,
    }
    for key, value in values.items():
        meta[CONFLICT_REQ[key]] = value
    struct.pack_into(
        f"<{CONFLICT_REQUEST_META_WORDS}Q",
        request,
        CONFLICT_REQUEST_META_WORD * 8,
        *meta,
    )
    return ConflictEquivalenceInvocation(
        pair,
        sample,
        first_schedule,
        bytes(request),
        serial_built.payload,
        serial_built,
        window_built,
    )


CALIBRATION_LEAF_BINDINGS = {
    "dma-offset-alignment-tail": DMA_CASES[:6],
    "ddr-large-stride-default-burst-tail": DMA_CASES[6:],
    "spm-five-engine-access": ENGINE_ACCESS_CASES,
    "spm-exact-partial-adjacent-disjoint-strided": RELATION_CASES,
    "spm-bank-engine-pair-controls": PAIR_CASES,
    "multi-engine-sustained-parallel-controls": PARALLEL_PAIR_CASES,
    "ddr-configurable-burst-static-boundary": STATIC_BOUNDARIES,
}
