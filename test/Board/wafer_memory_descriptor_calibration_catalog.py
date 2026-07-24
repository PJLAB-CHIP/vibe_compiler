#!/usr/bin/env python3
"""Bounded DMA/DDR descriptors and SPM engine/relation observations."""

from __future__ import annotations

import dataclasses
import struct


REQUEST_MAGIC = 0x315145444D454D57
RECORD_MAGIC = 0x314345444D454D57
REQUEST_GUARD = 0xB5A4938271605F4E
RECORD_GUARD = 0x32435465768798A9
SCHEMA = 1
REQUEST_WORDS = 40
RECORD_WORDS = 64
RESOURCE_BYTES = 262144
PAYLOAD_DATA_OFFSET = 4096
OUTPUT_DATA_OFFSET = 4096
OUTPUT_SINK0_OFFSET = 65536
OUTPUT_SINK1_OFFSET = 73728
RESOURCE_CANARY = 0xA5
SPM_GUARD = 0x6D

KIND_DMA = 0
KIND_ENGINE_ACCESS = 1
KIND_ADDRESS_RELATION = 2
KIND_ENGINE_PAIR = 3

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
SPM_PAIR_B = SPM_BASE + 8192

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
    "RECORD_GUARD": 63,
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
            "descriptor": dataclasses.asdict(self.descriptor),
            "compact_bytes": self.descriptor.compact_bytes,
            "envelope_bytes": self.descriptor.envelope_bytes,
            "output_bytes": self.output_bytes,
            "expected_counts": self.expected_counts,
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
    case_id: int, engine_a: int, engine_b: int, schedule: int
) -> MemoryCase:
    return MemoryCase(
        case_id,
        f"spm-bank-pair-{ENGINE_NAMES[engine_a].lower()}-"
        f"{ENGINE_NAMES[engine_b].lower()}-"
        f"{'serial' if schedule == SCHEDULE_SERIAL else 'window'}",
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
        SPM_PAIR_B,
        Descriptor(256),
        512,
        _counts(engine_a, engine_b),
    )


PAIR_CASES = tuple(
    _pair_case(
        39 + pair_index * 2 + schedule,
        engine_a,
        engine_b,
        schedule,
    )
    for pair_index, (engine_a, engine_b) in enumerate(
        (left, right)
        for left in range(5)
        for right in range(left + 1, 5)
    )
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WINDOW)
)

CATALOG = DMA_CASES + ENGINE_ACCESS_CASES + RELATION_CASES + PAIR_CASES
CASES_BY_ID = {case.case_id: case for case in CATALOG}
CASES_BY_NAME = {case.name: case for case in CATALOG}


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


def build_case_payload(case: MemoryCase, sample: int = 0) -> CasePayload:
    seed = sample + 1
    payload = bytearray([RESOURCE_CANARY] * RESOURCE_BYTES)
    expected = bytearray([RESOURCE_CANARY] * RESOURCE_BYTES)
    allowed: list[tuple[int, int]] = []
    exact: list[tuple[int, int]] = []

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
                if case.kind == KIND_ENGINE_ACCESS
                else 256
            )
            lane_payload, result, exact_bytes = _engine_payload_expected(
                engine, seed, lane, transfer
            )
            payload_begin = PAYLOAD_DATA_OFFSET + lane * 16384
            payload[
                payload_begin : payload_begin + len(lane_payload)
            ] = lane_payload
            output_begin = OUTPUT_DATA_OFFSET + lane * 256
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
    else:
        for lane in range(2):
            source = _pattern(seed, 80 + lane, 4096)
            begin = PAYLOAD_DATA_OFFSET + lane * 16384
            payload[begin : begin + len(source)] = source
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


CALIBRATION_LEAF_BINDINGS = {
    "dma-offset-alignment-tail": DMA_CASES[:6],
    "ddr-large-stride-default-burst-tail": DMA_CASES[6:],
    "spm-five-engine-access": ENGINE_ACCESS_CASES,
    "spm-exact-partial-adjacent-disjoint-strided": RELATION_CASES,
    "spm-bank-engine-pair-controls": PAIR_CASES,
    "ddr-configurable-burst-static-boundary": STATIC_BOUNDARIES,
}
