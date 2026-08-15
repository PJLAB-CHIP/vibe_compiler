#!/usr/bin/env python3
"""Run complete-Tile-domain DDR active-Tile cases and gate typed boundaries.

The DDR path reuses the qualified 16-Tile cluster package, direct-DTE
status lifecycle, and wafer-run launch contract from the complete-Tile DDR
probe.  Every launch has all 16 Tiles present; an explicit request mask
selects 1/2/4/8/16 target Tiles.  Executable worker/SPM families are owned by
their dedicated adapters; genuine worker/SPM and DDR-bank boundaries retained
here remain non-serializable typed host blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
from collections.abc import Mapping, Sequence

import wafer_board_ddr_tile_offset_probe_test as cluster_driver
import wafer_direct_dte_board_evidence as direct_dte_evidence
import wafer_worker_memory_contention_characterization_catalog as catalog


INPUT_DIR = pathlib.Path(__file__).resolve().parent / "Inputs"
PROBE_C = INPUT_DIR / "wafer_ddr_active_tile_contention_probe.c"
PROTOCOL_H = (
    INPUT_DIR / "wafer_ddr_active_tile_contention_probe_protocol.h"
)
_PROTOCOL_TEXT = PROTOCOL_H.read_text(encoding="utf-8")
CANARY = 0xA5
STATUS_OK = 0


class PreparationBlocked(RuntimeError):
    """A typed host gate rejected board serialization."""


def _c_integer(expression: str) -> int:
    expression = re.sub(r"UINT(?:32|64)_C\(([^)]+)\)", r"\1", expression)
    expression = re.sub(
        r"(?<=\d)[uUlL]+\b|(?<=[a-fA-F0-9])[uUlL]+\b",
        "",
        expression,
    )
    if re.fullmatch(r"[\s0-9a-fA-FxX()<>|+-]+", expression) is None:
        raise RuntimeError(f"unsupported protocol integer {expression!r}")
    return int(eval(expression, {"__builtins__": {}}, {}))


def _macro(name: str) -> int:
    match = re.search(
        rf"^#define\s+{re.escape(name)}\s+(.+?)\s*$",
        _PROTOCOL_TEXT,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"protocol macro {name} is missing")
    return _c_integer(match.group(1))


def _enumerator(name: str) -> int:
    match = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*([^,]+),?\s*$",
        _PROTOCOL_TEXT,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"protocol enumerator {name} is missing")
    return _c_integer(match.group(1))


REQUEST_MAGIC = _macro("WAFER_DAR_REQUEST_MAGIC")
RECORD_MAGIC = _macro("WAFER_DAR_RECORD_MAGIC")
REQUEST_GUARD = _macro("WAFER_DAR_REQUEST_GUARD")
RECORD_GUARD = _macro("WAFER_DAR_RECORD_GUARD")
REQUEST_WORDS = _macro("WAFER_DAR_REQUEST_WORDS")
RECORD_WORDS = _macro("WAFER_DAR_RECORD_WORDS")
TILE_COUNT = _macro("WAFER_DAR_TILES")
GUARD_BYTES = _macro("WAFER_DAR_GUARD_BYTES")
INACTIVE_CANARY_BYTES = _macro("WAFER_DAR_INACTIVE_CANARY_BYTES")
RESOURCE_BYTES = _macro("WAFER_DAR_RESOURCE_BYTES")
SOURCE_OFFSET = _macro("WAFER_DAR_SOURCE_OFFSET")
SEED_RDMA_OFFSET = _macro("WAFER_DAR_SEED_RDMA_OFFSET")
SEED_WDMA_OFFSET = _macro("WAFER_DAR_SEED_WDMA_OFFSET")
RDMA_ARCHIVE_OFFSET = _macro("WAFER_DAR_RDMA_ARCHIVE_OFFSET")
WDMA_TARGET_OFFSET = _macro("WAFER_DAR_WDMA_TARGET_OFFSET")
SPM_RDMA = _macro("WAFER_DAR_SPM_RDMA")
SPM_WDMA = _macro("WAFER_DAR_SPM_WDMA")
DIRECTION = {
    catalog.DDRDirection.RDMA: _enumerator("WAFER_DAR_DIRECTION_RDMA"),
    catalog.DDRDirection.WDMA: _enumerator("WAFER_DAR_DIRECTION_WDMA"),
    catalog.DDRDirection.BIDIRECTIONAL: _enumerator(
        "WAFER_DAR_DIRECTION_BIDIRECTIONAL"
    ),
}
STATUS_PMU_UNSTABLE = _enumerator("WAFER_DAR_STATUS_PMU_UNSTABLE")
REQ = {
    name: _enumerator(f"WAFER_DAR_REQ_{name}")
    for name in (
        "MAGIC",
        "WORD_COUNT",
        "TILE_ID",
        "SAMPLE",
        "ACTIVE_MASK",
        "DIRECTION",
        "PAYLOAD_BYTES",
        "INNER_BYTES",
        "STRIDE0",
        "ITERATION0",
        "EXPECTED_RDMA",
        "EXPECTED_WDMA",
        "ISSUE_ORDER",
        "RESOURCE_BYTES",
        "PAYLOAD_SEED",
        "GUARD",
    )
}
REC = {
    name: _enumerator(f"WAFER_DAR_REC_{name}")
    for name in (
        "MAGIC",
        "WORD_COUNT",
        "STATUS",
        "TILE_ID",
        "SAMPLE",
        "ACTIVE_MASK",
        "ACTIVE",
        "DIRECTION",
        "PAYLOAD_BYTES",
        "INNER_BYTES",
        "STRIDE0",
        "ITERATION0",
        "ENVELOPE_BYTES",
        "RDMA_INST_DELTA",
        "WDMA_INST_DELTA",
        "RDMA_BLOCKING_DELTA",
        "WDMA_BLOCKING_DELTA",
        "RDMA_EXEC_DELTA",
        "WDMA_EXEC_DELTA",
        "FU_EXEC_DELTA",
        "WINDOW_DELTA",
        "COMPLETION_CYCLES",
        "INPUT0_BASE",
        "INPUT1_BASE",
        "OUTPUT0_BASE",
        "OUTPUT1_BASE",
        "SPM_RDMA",
        "SPM_WDMA",
        "ISSUE_ORDER",
        "REQUEST_GUARD",
        "PAYLOAD_SEED",
        "TARGET_ONLY_WINDOW",
        "PMU_BEFORE_STABLE",
        "PMU_AFTER_STABLE",
        "RECORD_GUARD",
    )
}


def all_objects() -> dict[str, object]:
    return {
        **catalog.CASES_BY_KEY,
        **catalog.DELEGATED_BY_KEY,
        **catalog.BOUNDARIES_BY_KEY,
    }


def resolve_object(key: str) -> object:
    objects = all_objects()
    if key not in objects:
        raise KeyError(key)
    return objects[key]


def case_keys() -> tuple[str, ...]:
    return tuple(sorted(all_objects()))


def _blocked_record(value: object) -> Mapping[str, object]:
    if isinstance(value, catalog.CharacterizationCase):
        return value.blocker
    if isinstance(value, catalog.TypedBoundary):
        return {
            "reason": value.reason,
            "typed_gate": value.typed_gate,
            "safe_alternative": value.safe_alternative,
        }
    raise TypeError("object is not a typed blocker")


def _shape(
    case: catalog.CharacterizationCase,
) -> tuple[int, int, int, int]:
    if case.domain != catalog.Domain.DDR_ACTIVE_TILE:
        raise ValueError(f"{case.key}: not an active-Tile DDR case")
    stream = case.streams[0]
    if stream.stride_bytes:
        return (stream.payload_bytes, 4096, stream.stride_bytes, 16)
    return (stream.payload_bytes, stream.payload_bytes, 0, 1)


def active_tiles_for_sample(
    case: catalog.CharacterizationCase, sample: int
) -> tuple[int, ...]:
    if sample < 0:
        raise ValueError("sample is outside the request domain")
    count = len(case.active_tiles)
    if count not in {1, 2, 4, 8}:
        return case.active_tiles
    phase_count = TILE_COUNT // count
    phase = sample % phase_count
    return tuple(
        sorted((tile_id + phase) % TILE_COUNT for tile_id in case.active_tiles)
    )


def inactive_tiles_for_sample(
    case: catalog.CharacterizationCase, sample: int
) -> tuple[int, ...]:
    active = set(active_tiles_for_sample(case, sample))
    return tuple(tile_id for tile_id in range(TILE_COUNT) if tile_id not in active)


def active_mask(case: catalog.CharacterizationCase, sample: int = 0) -> int:
    return sum(1 << tile_id for tile_id in active_tiles_for_sample(case, sample))


def ordered_group_cases(
    cases: Sequence[catalog.CharacterizationCase], sample: int
) -> tuple[catalog.CharacterizationCase, ...]:
    if not cases:
        raise ValueError("active-Tile group is empty")
    values = tuple(cases)
    phase = sample % 4
    if phase == 0:
        return values
    if phase == 1:
        return tuple(reversed(values))
    offset = len(values) // 2
    rotated = values[offset:] + values[:offset]
    return rotated if phase == 2 else tuple(reversed(rotated))


def group_execution_plan(
    cases: Sequence[catalog.CharacterizationCase], repeat: int
) -> tuple[tuple[int, catalog.CharacterizationCase], ...]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    return tuple(
        (sample, case)
        for sample in range(repeat)
        for case in ordered_group_cases(cases, sample)
    )


def request_words(
    case_or_key: catalog.CharacterizationCase | str,
    tile_id: int,
    sample: int,
) -> tuple[int, ...]:
    case = (
        catalog.CASES_BY_KEY[case_or_key]
        if isinstance(case_or_key, str)
        else case_or_key
    )
    if (
        case not in catalog.DDR_ACTIVE_TILE_CASES
        or case.disposition != catalog.Disposition.BOARD_EXECUTABLE
    ):
        raise PreparationBlocked(f"{case.key}: no executable DDR adapter")
    if tile_id not in range(TILE_COUNT) or sample < 0:
        raise ValueError("tile_id/sample is outside the request domain")
    assert case.ddr_direction is not None
    direction = DIRECTION[case.ddr_direction]
    payload_bytes, inner_bytes, stride0, iteration0 = _shape(case)
    words = [0] * REQUEST_WORDS
    words[REQ["MAGIC"]] = REQUEST_MAGIC
    words[REQ["WORD_COUNT"]] = REQUEST_WORDS
    words[REQ["TILE_ID"]] = tile_id
    words[REQ["SAMPLE"]] = sample
    words[REQ["ACTIVE_MASK"]] = active_mask(case, sample)
    words[REQ["DIRECTION"]] = direction
    words[REQ["PAYLOAD_BYTES"]] = payload_bytes
    words[REQ["INNER_BYTES"]] = inner_bytes
    words[REQ["STRIDE0"]] = stride0
    words[REQ["ITERATION0"]] = iteration0
    words[REQ["EXPECTED_RDMA"]] = int(
        case.ddr_direction != catalog.DDRDirection.WDMA
    )
    words[REQ["EXPECTED_WDMA"]] = int(
        case.ddr_direction != catalog.DDRDirection.RDMA
    )
    words[REQ["ISSUE_ORDER"]] = (
        sample & 1
        if case.ddr_direction == catalog.DDRDirection.BIDIRECTIONAL
        else 0
    )
    words[REQ["RESOURCE_BYTES"]] = RESOURCE_BYTES
    words[REQ["PAYLOAD_SEED"]] = case.payload_seed
    words[REQ["GUARD"]] = REQUEST_GUARD
    return tuple(words)


def request_bytes(
    case: catalog.CharacterizationCase,
    tile_id: int,
    sample: int,
) -> bytes:
    return struct.pack(
        f"<{REQUEST_WORDS}Q", *request_words(case, tile_id, sample)
    )


def _compact_payload_pattern(
    case: catalog.CharacterizationCase,
    tile_id: int,
    sample: int,
    purpose: str,
) -> bytes:
    payload_bytes, _, _, _ = _shape(case)
    purpose_delta = {"rdma": 0, "wdma": 73}[purpose]
    return bytes(
        (
            case.payload_seed
            + tile_id * 37
            + sample * 29
            + compact_index * 17
            + (compact_index >> 8) * 11
            + purpose_delta
        )
        & 0xFF
        for compact_index in range(payload_bytes)
    )


def _ddr_span_pattern(
    case: catalog.CharacterizationCase,
    tile_id: int,
    sample: int,
    purpose: str,
) -> bytes:
    payload_bytes, inner_bytes, stride0, iteration0 = _shape(case)
    envelope = inner_bytes + stride0 * (iteration0 - 1)
    result = bytearray([CANARY] * (envelope + 2 * GUARD_BYTES))
    payload = _compact_payload_pattern(case, tile_id, sample, purpose)
    compact_begin = 0
    for iteration in range(iteration0):
        begin = GUARD_BYTES + (
            iteration * stride0 if iteration0 > 1 else 0
        )
        compact_end = compact_begin + inner_bytes
        result[begin : begin + inner_bytes] = payload[
            compact_begin:compact_end
        ]
        compact_begin = compact_end
    if compact_begin != payload_bytes:
        raise AssertionError("DDR span did not consume compact payload")
    return bytes(result)


def _spm_span_pattern(
    case: catalog.CharacterizationCase,
    tile_id: int,
    sample: int,
    purpose: str,
) -> bytes:
    payload_bytes, inner_bytes, stride0, iteration0 = _shape(case)
    envelope = inner_bytes + stride0 * (iteration0 - 1)
    result = bytearray([CANARY] * (envelope + 2 * GUARD_BYTES))
    payload = _compact_payload_pattern(case, tile_id, sample, purpose)
    result[GUARD_BYTES : GUARD_BYTES + payload_bytes] = payload
    return bytes(result)


def tile_inputs(
    case: catalog.CharacterizationCase,
    tile_id: int,
    sample: int,
) -> tuple[bytes, bytes]:
    input0 = bytearray(RESOURCE_BYTES)
    input1 = bytearray(RESOURCE_BYTES)
    request = request_bytes(case, tile_id, sample)
    input0[: len(request)] = request
    rdma_ddr_span = _ddr_span_pattern(case, tile_id, sample, "rdma")
    rdma_spm_span = _spm_span_pattern(case, tile_id, sample, "rdma")
    wdma_spm_span = _spm_span_pattern(case, tile_id, sample, "wdma")
    input0[
        SOURCE_OFFSET : SOURCE_OFFSET + len(rdma_ddr_span)
    ] = rdma_ddr_span
    input1[
        SEED_RDMA_OFFSET : SEED_RDMA_OFFSET + len(rdma_spm_span)
    ] = bytes([CANARY]) * len(rdma_spm_span)
    input1[
        SEED_WDMA_OFFSET : SEED_WDMA_OFFSET + len(wdma_spm_span)
    ] = wdma_spm_span
    return bytes(input0), bytes(input1)


def _inactive_canary(tile_id: int, sample: int) -> bytes:
    words = [
        (
            0xC39A57E10D2468BF
            ^ (tile_id << 48)
            ^ (sample << 24)
            ^ (word * 0x9E3779B97F4A7C15)
        )
        & ((1 << 64) - 1)
        for word in range(INACTIVE_CANARY_BYTES // 8)
    ]
    return struct.pack(f"<{len(words)}Q", *words)


def prepare_board_selection(key: str) -> dict[str, object]:
    value = resolve_object(key)
    if isinstance(value, catalog.CharacterizationCase):
        if value.disposition == catalog.Disposition.BOARD_EXECUTABLE:
            return {
                "key": value.key,
                "disposition": value.disposition.value,
                "driver": str(pathlib.Path(__file__).resolve()),
                "carrier": str(PROBE_C),
                "protocol": str(PROTOCOL_H),
                "launch": "wafer-run --board cluster status",
                "tile_count": TILE_COUNT,
                "active_mask": active_mask(value),
                "sample_active_masks": [
                    active_mask(value, sample) for sample in range(4)
                ],
                "request_words_tile0_sample0": list(
                    request_words(value, 0, 0)
                ),
                "measurement": (
                    "per-Tile target-only device rdcycle/PMU; no host time"
                ),
            }
        blocker = _blocked_record(value)
    elif isinstance(value, catalog.DelegatedAsset):
        return {
            "key": value.key,
            "disposition": value.disposition.value,
            "board_request_serializable_here": False,
            "route": {
                "source_file": value.source_file,
                "selector": value.selector,
                "object_count": value.object_count,
            },
            "evidence_scope": value.evidence_scope,
        }
    else:
        blocker = _blocked_record(value)
    raise PreparationBlocked(
        f"{key}: {blocker['typed_gate']}; reason: {blocker['reason']}; "
        f"safe alternative: {blocker['safe_alternative']}"
    )


def inventory() -> dict[str, object]:
    by_disposition: dict[str, int] = {}
    for value in all_objects().values():
        disposition = value.disposition.value
        by_disposition[disposition] = (
            by_disposition.get(disposition, 0) + 1
        )
    return {
        "executable_new_cases": len(catalog.EXECUTABLE_CASES),
        "executable_ddr_active_tile_matrix": len(
            catalog.DDR_ACTIVE_TILE_CASES
        ),
        "delegated_assets": len(catalog.DELEGATED_ASSETS),
        "typed_boundaries": len(catalog.TYPED_BOUNDARIES),
        "by_disposition": dict(sorted(by_disposition.items())),
        "objects": [
            all_objects()[key].as_dict() for key in case_keys()
        ],
    }


def validate_static_contract() -> None:
    if TILE_COUNT != catalog.TILE_COUNT:
        raise RuntimeError("catalog/device tile_id domains differ")
    if RESOURCE_BYTES != cluster_driver.RESOURCE_BYTES:
        raise RuntimeError("carrier and cluster package resource sizes differ")
    if set(REQ.values()) != {
        *range(15),
        REQUEST_WORDS - 1,
    }:
        raise RuntimeError("request word layout is not exact")
    if set(REC.values()) != set(range(RECORD_WORDS)):
        raise RuntimeError("record word layout is not exact")
    if (
        RDMA_ARCHIVE_OFFSET + 65536 + 16 * 4096 + 2 * GUARD_BYTES
        > WDMA_TARGET_OFFSET
    ):
        raise RuntimeError("DDR output oracle regions can overlap")
    source = PROBE_C.read_text(encoding="utf-8")
    required = (
        '#include "wafer_ddr_tile_offset_probe.c"',
        "wafer_tx81_direct_dte_begin_after_prepare",
        "wafer_tx81_direct_dte_finish",
        "request[WAFER_DAR_REQ_ACTIVE_MASK] >> tile_id",
        "hrt_barrier();",
        "WaferDDRTilePMU before",
        "WaferDDRTilePMU after",
        "WAFER_DAR_STATUS_TARGET_COUNT_MISMATCH",
        "WAFER_DAR_STATUS_PMU_UNSTABLE",
        "wafer_dar_read64_stable",
        "before_stable == 0U || after_stable == 0U",
        "wafer_dar_write_inactive_canary",
        "wafer_dar_copy_chunks",
    )
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        raise RuntimeError(f"active-Tile carrier contract is missing {missing}")
    stable_reader = source[
        source.index("static uint64_t wafer_dar_read64_stable") :
        source.index("static WaferDDRTilePMU wafer_dar_read_pmu")
    ]
    stable_condition = "if (high_before == high_after)"
    stable_assignment = "*stable = 1U;"
    if (
        stable_reader.count(stable_assignment) != 1
        or stable_reader.index(stable_assignment)
        < stable_reader.index(stable_condition)
    ):
        raise RuntimeError(
            "PMU stable-read success is not gated by high-word equality"
        )


def _expected_output0(
    case: catalog.CharacterizationCase,
    tile_id: int,
    sample: int,
    record_bytes: bytes,
) -> bytes:
    expected = bytearray([CANARY]) * RESOURCE_BYTES
    expected[: len(record_bytes)] = record_bytes
    if tile_id in active_tiles_for_sample(case, sample):
        assert case.ddr_direction is not None
        if case.ddr_direction != catalog.DDRDirection.WDMA:
            span = _spm_span_pattern(case, tile_id, sample, "rdma")
            expected[
                RDMA_ARCHIVE_OFFSET : RDMA_ARCHIVE_OFFSET + len(span)
            ] = span
        if case.ddr_direction != catalog.DDRDirection.RDMA:
            span = _ddr_span_pattern(case, tile_id, sample, "wdma")
            expected[
                WDMA_TARGET_OFFSET : WDMA_TARGET_OFFSET + len(span)
            ] = span
    return bytes(expected)


def _expected_output1(tile_id: int, sample: int) -> bytes:
    expected = bytearray([CANARY]) * RESOURCE_BYTES
    expected[:INACTIVE_CANARY_BYTES] = _inactive_canary(tile_id, sample)
    return bytes(expected)


def validate_tile_outputs(
    case: catalog.CharacterizationCase,
    tile_id: int,
    sample: int,
    output0_path: pathlib.Path,
    output1_path: pathlib.Path,
    physical: tuple[int, int],
) -> dict[str, object]:
    output0 = output0_path.read_bytes()
    output1 = output1_path.read_bytes()
    if len(output0) != RESOURCE_BYTES or len(output1) != RESOURCE_BYTES:
        raise RuntimeError(f"{case.key}: tile_id {tile_id} output size is wrong")
    record_bytes = output0[: RECORD_WORDS * 8]
    record = struct.unpack(f"<{RECORD_WORDS}Q", record_bytes)
    payload_bytes, inner_bytes, stride0, iteration0 = _shape(case)
    envelope = inner_bytes + stride0 * (iteration0 - 1)
    assert case.ddr_direction is not None
    direction = DIRECTION[case.ddr_direction]
    sample_active_tiles = active_tiles_for_sample(case, sample)
    active = int(tile_id in sample_active_tiles)
    expected_rdma = int(
        active and case.ddr_direction != catalog.DDRDirection.WDMA
    )
    expected_wdma = int(
        active and case.ddr_direction != catalog.DDRDirection.RDMA
    )
    expected_fields = {
        REC["MAGIC"]: RECORD_MAGIC,
        REC["WORD_COUNT"]: RECORD_WORDS,
        REC["STATUS"]: STATUS_OK,
        REC["TILE_ID"]: tile_id,
        REC["SAMPLE"]: sample,
        REC["ACTIVE_MASK"]: active_mask(case, sample),
        REC["ACTIVE"]: active,
        REC["DIRECTION"]: direction,
        REC["PAYLOAD_BYTES"]: payload_bytes,
        REC["INNER_BYTES"]: inner_bytes,
        REC["STRIDE0"]: stride0,
        REC["ITERATION0"]: iteration0,
        REC["ENVELOPE_BYTES"]: envelope,
        REC["RDMA_INST_DELTA"]: expected_rdma,
        REC["WDMA_INST_DELTA"]: expected_wdma,
        REC["SPM_RDMA"]: SPM_RDMA,
        REC["SPM_WDMA"]: SPM_WDMA,
        REC["ISSUE_ORDER"]: (
            sample & 1
            if case.ddr_direction == catalog.DDRDirection.BIDIRECTIONAL
            else 0
        ),
        REC["REQUEST_GUARD"]: REQUEST_GUARD,
        REC["PAYLOAD_SEED"]: case.payload_seed,
        REC["TARGET_ONLY_WINDOW"]: 1,
        REC["PMU_BEFORE_STABLE"]: 1,
        REC["PMU_AFTER_STABLE"]: 1,
        REC["RECORD_GUARD"]: RECORD_GUARD,
    }
    failures = {
        index: (record[index], expected)
        for index, expected in expected_fields.items()
        if record[index] != expected
    }
    if failures:
        raise RuntimeError(
            f"{case.key}: tile_id {tile_id} record oracle failed: {failures}"
        )
    bases = tuple(
        record[REC[name]]
        for name in (
            "INPUT0_BASE",
            "INPUT1_BASE",
            "OUTPUT0_BASE",
            "OUTPUT1_BASE",
        )
    )
    if (
        any(base == 0 or base % 256 for base in bases)
        or len(set(bases)) != 4
    ):
        raise RuntimeError(
            f"{case.key}: tile_id {tile_id} resource bases alias or misalign"
        )
    completion_cycles = record[REC["COMPLETION_CYCLES"]]
    if active and completion_cycles == 0:
        raise RuntimeError(
            f"{case.key}: tile_id {tile_id} has no device completion cycle"
        )
    if active:
        if expected_rdma and record[REC["RDMA_EXEC_DELTA"]] == 0:
            raise RuntimeError(
                f"{case.key}: tile_id {tile_id} RDMA PMU delta is zero"
            )
        if expected_wdma and record[REC["WDMA_EXEC_DELTA"]] == 0:
            raise RuntimeError(
                f"{case.key}: tile_id {tile_id} WDMA PMU delta is zero"
            )
    else:
        inactive_metrics = (
            "RDMA_INST_DELTA",
            "WDMA_INST_DELTA",
            "RDMA_BLOCKING_DELTA",
            "WDMA_BLOCKING_DELTA",
            "RDMA_EXEC_DELTA",
            "WDMA_EXEC_DELTA",
        )
        if any(record[REC[name]] != 0 for name in inactive_metrics):
            raise RuntimeError(
                f"{case.key}: inactive tile_id {tile_id} changed target PMU"
            )
    expected0 = _expected_output0(
        case, tile_id, sample, record_bytes
    )
    if output0 != expected0:
        mismatch = next(
            index
            for index, (actual, wanted) in enumerate(
                zip(output0, expected0, strict=True)
            )
            if actual != wanted
        )
        raise RuntimeError(
            f"{case.key}: tile_id {tile_id} output0 mismatch at {mismatch}"
        )
    expected1 = _expected_output1(tile_id, sample)
    if output1 != expected1:
        mismatch = next(
            index
            for index, (actual, wanted) in enumerate(
                zip(output1, expected1, strict=True)
            )
            if actual != wanted
        )
        raise RuntimeError(
            f"{case.key}: tile_id {tile_id} inactive canary mismatch at {mismatch}"
        )
    return {
        "case_key": case.key,
        "tile_id": tile_id,
        "physical_x": physical[0],
        "physical_y": physical[1],
        "active": bool(active),
        "active_mask": active_mask(case, sample),
        "direction": case.ddr_direction.value,
        "payload_bytes": payload_bytes,
        "inner_bytes": inner_bytes,
        "stride0": stride0,
        "iteration0": iteration0,
        "rdma_instruction_delta": record[REC["RDMA_INST_DELTA"]],
        "wdma_instruction_delta": record[REC["WDMA_INST_DELTA"]],
        "rdma_blocking_delta": record[REC["RDMA_BLOCKING_DELTA"]],
        "wdma_blocking_delta": record[REC["WDMA_BLOCKING_DELTA"]],
        "rdma_execution_delta": record[REC["RDMA_EXEC_DELTA"]],
        "wdma_execution_delta": record[REC["WDMA_EXEC_DELTA"]],
        "completion_cycles": completion_cycles,
        "fu_execution_delta": record[REC["FU_EXEC_DELTA"]],
        "statistics_window_delta": record[REC["WINDOW_DELTA"]],
        "exact_full_output": True,
        "physical_guards_and_holes": True,
        "inactive_canary": True,
        "target_only_window": True,
        "pmu_before_stable": True,
        "pmu_after_stable": True,
        "measurement_basis": "device-rdcycle-and-local-pmu",
        "host_elapsed_used": False,
        "record_words": list(record),
        "output0_sha256": hashlib.sha256(output0).hexdigest(),
        "output1_sha256": hashlib.sha256(output1).hexdigest(),
    }


def _write_resources(
    work_dir: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    case: catalog.CharacterizationCase,
    sample: int,
) -> tuple[
    list[str],
    dict[tuple[int, int], pathlib.Path],
    pathlib.Path,
    tuple[pathlib.Path, ...],
    dict[tuple[int, int], str],
]:
    raw_dir = work_dir / "raw" / case.key / f"sample-{sample}"
    raw_dir.mkdir(parents=True)
    arguments: list[str] = []
    outputs: dict[tuple[int, int], pathlib.Path] = {}
    resource_paths: list[pathlib.Path] = []
    input_hashes: dict[tuple[int, int], str] = {}
    for tile_id in range(TILE_COUNT):
        input0, input1 = tile_inputs(case, tile_id, sample)
        for ordinal, payload in enumerate((input0, input1)):
            path = raw_dir / f"tile-{tile_id:02d}.input-{ordinal}.raw"
            path.write_bytes(payload)
            resource_paths.append(path)
            input_hashes[(tile_id, ordinal)] = hashlib.sha256(
                payload
            ).hexdigest()
            arguments.extend(
                [
                    "--resource",
                    f"{bindings[(tile_id, 'user_input', ordinal)]}={path}",
                ]
            )
        for ordinal in range(2):
            path = raw_dir / f"tile-{tile_id:02d}.output-{ordinal}.raw"
            outputs[(tile_id, ordinal)] = path
            resource_paths.append(path)
            arguments.extend(
                [
                    "--output",
                    f"{bindings[(tile_id, 'output', ordinal)]}={path}",
                ]
            )
    return (
        arguments,
        outputs,
        raw_dir,
        tuple(resource_paths),
        input_hashes,
    )


def _discard_validated_resources(
    work_dir: pathlib.Path,
    raw_dir: pathlib.Path,
    resource_paths: Sequence[pathlib.Path],
) -> None:
    resolved_work = work_dir.resolve()
    resolved_raw = raw_dir.resolve()
    raw_root = resolved_work / "raw"
    if (
        resolved_raw == raw_root
        or not resolved_raw.is_relative_to(raw_root)
    ):
        raise RuntimeError("validated raw resource directory escaped work-dir")
    expected_names = {path.name for path in resource_paths}
    actual = tuple(resolved_raw.iterdir())
    if (
        {path.name for path in actual} != expected_names
        or len(expected_names) != TILE_COUNT * 4
        or any(path.is_symlink() or not path.is_file() for path in actual)
    ):
        raise RuntimeError(
            "validated raw resource cleanup saw an unexpected file set"
        )
    for path in sorted(actual):
        path.unlink()
    resolved_raw.rmdir()


def _runtime_args(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_args: list[str],
) -> list[str]:
    return [
        str(args.wafer_run),
        "--package-dir",
        str(package),
        "--board",
        "--device-id",
        str(args.device_id),
        "--expected-runtime-version",
        str(args.expected_runtime_version),
        "--expected-device-name",
        str(args.expected_device_name),
        "--expected-pci-bus-id",
        str(args.expected_pci_bus_id),
        "--expected-tile-count",
        str(args.expected_tile_count),
        "--expected-runtime-library-sha256",
        str(args.expected_runtime_library_sha256),
        "--completion-timeout-ms",
        str(args.completion_timeout_ms),
        *resource_args,
    ]


def validate_board_lifecycle(
    stdout: str,
    manifest: direct_dte_evidence.DirectDTEManifestEvidence,
    completion_timeout_ms: int,
) -> dict[str, object]:
    evidence = direct_dte_evidence.validate_direct_dte_board_output(
        stdout,
        manifest,
        completion_timeout_ms=completion_timeout_ms,
    )
    lines = stdout.splitlines()
    exact_basis = (
        "physical_tile_execution_basis: cluster-pid-and-exact-tile-slices"
    )
    if lines.count(exact_basis) != 1:
        raise RuntimeError(
            "DDR active-Tile board output omitted its exact Tile-slice basis"
        )
    return {
        "completion_by_tile": [
            {"tile_id": tile_id, "completion": completion}
            for tile_id, completion in evidence.completion_by_tile
        ],
        "runtime_all_tile_success_enforced": (
            evidence.runtime_all_tile_success_enforced
        ),
        "completion_timeout_ms": evidence.completion_timeout_ms,
    }


def _validate_work_dir(
    repo_root: pathlib.Path, work_dir: pathlib.Path
) -> pathlib.Path:
    resolved_repo = repo_root.resolve()
    resolved = work_dir.resolve()
    if resolved == resolved_repo or resolved in resolved_repo.parents:
        raise RuntimeError("active-Tile work directory is too broad")
    return resolved


def _prepare_package(
    args: argparse.Namespace,
) -> tuple[
    pathlib.Path,
    dict[tuple[int, str, int], int],
    direct_dte_evidence.DirectDTEManifestEvidence,
]:
    validate_static_contract()
    original_probe = cluster_driver.PROBE_C
    cluster_driver.PROBE_C = PROBE_C
    try:
        package, module_path, bindings = cluster_driver.compile_package(args)
        cluster_driver.build_probe(args, package, module_path)
        cluster_driver.verify_no_card(args, package)
    finally:
        cluster_driver.PROBE_C = original_probe
    manifest = json.loads((package / "manifest.json").read_text())
    completion_evidence = (
        direct_dte_evidence.validate_direct_dte_manifest(manifest)
    )
    return package, bindings, completion_evidence


def _execute_launch(
    args: argparse.Namespace,
    case: catalog.CharacterizationCase,
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    completion_manifest: direct_dte_evidence.DirectDTEManifestEvidence,
    *,
    sample: int,
    execution_ordinal: int,
    reference_coordinates: dict[int, tuple[int, int]] | None,
) -> tuple[
    dict[str, object],
    dict[int, tuple[int, int]],
    pathlib.Path,
    tuple[pathlib.Path, ...],
]:
    (
        resource_args,
        outputs,
        raw_dir,
        resource_paths,
        input_hashes,
    ) = _write_resources(args.work_dir, bindings, case, sample)
    runtime_command = _runtime_args(args, package, resource_args)
    completion_timeout_ms = (
        direct_dte_evidence.validate_direct_dte_board_command(
            runtime_command
        )
    )
    result = cluster_driver.run(
        runtime_command,
        timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0,
    )
    lifecycle_evidence = validate_board_lifecycle(
        result.stdout, completion_manifest, completion_timeout_ms
    )
    coordinates = cluster_driver.parse_tile_coordinates(result.stdout)
    if reference_coordinates is not None:
        cluster_driver.require_stable_tile_coordinates(
            reference_coordinates, coordinates
        )
    rows = []
    for tile_id in range(TILE_COUNT):
        row = validate_tile_outputs(
            case,
            tile_id,
            sample,
            outputs[(tile_id, 0)],
            outputs[(tile_id, 1)],
            coordinates[tile_id],
        )
        row["request_words"] = list(request_words(case, tile_id, sample))
        row["input0_sha256"] = input_hashes[(tile_id, 0)]
        row["input1_sha256"] = input_hashes[(tile_id, 1)]
        rows.append(row)
    sample_active_tiles = active_tiles_for_sample(case, sample)
    active_rows = [row for row in rows if row["active"]]
    if len(active_rows) != len(sample_active_tiles):
        raise RuntimeError(f"{case.key}: active row count differs")
    launch = {
        "sample": sample,
        "execution_ordinal": execution_ordinal,
        "case_key": case.key,
        "active_tiles": list(sample_active_tiles),
        "inactive_tiles": list(inactive_tiles_for_sample(case, sample)),
        "active_mask": active_mask(case, sample),
        "complete_tile_max_cycles": max(
            int(row["completion_cycles"]) for row in active_rows
        ),
        "per_tile": rows,
        "all_tile_status": "terminal-success",
        **lifecycle_evidence,
        "lifecycle": (
            "cluster-prepare/completion/d2h/cleanup with bounded timeout"
        ),
        "measurement_basis": "device-rdcycle-and-local-pmu",
        "host_elapsed_used": False,
        "full_resources_validated": True,
        "full_resource_hashes_retained": True,
    }
    print("ddr_active_tile_launch: " + json.dumps(launch, sort_keys=True))
    print(result.stdout, end="")
    return launch, coordinates, raw_dir, resource_paths


def _validate_completed_launches(
    case: catalog.CharacterizationCase,
    launches: Sequence[dict[str, object]],
    repeat: int,
) -> None:
    if [launch["sample"] for launch in launches] != list(range(repeat)):
        raise RuntimeError(f"{case.key}: sample set/order is incomplete")
    if (
        case.ddr_direction == catalog.DDRDirection.BIDIRECTIONAL
        and {
            int(launch["sample"]) & 1 for launch in launches
        }
        != {0, 1}
    ):
        raise RuntimeError(f"{case.key}: issue order was not counterbalanced")
    if len(case.active_tiles) in {1, 2, 4, 8}:
        observed_masks = {
            int(launch["active_mask"]) for launch in launches
        }
        required_phases = min(
            repeat, TILE_COUNT // len(case.active_tiles)
        )
        if len(observed_masks) != required_phases:
            raise RuntimeError(
                f"{case.key}: physical active-Tile phases are incomplete"
            )


def _write_case_archive(
    args: argparse.Namespace,
    case: catalog.CharacterizationCase,
    launches: Sequence[dict[str, object]],
) -> dict[str, object]:
    archive = {
        "case": case.as_dict(),
        "repeat": args.repeat,
        "completed_launches": len(launches),
        "launches": list(launches),
        "raw_resource_retention": (
            "full 2MiB resources are deleted only after exact validation and "
            "durable request/record/full-resource SHA-256 evidence is written"
        ),
        "interpretation": (
            "hash-backed correctness-closed complete-Tile-domain contention "
            "observation; no DDR bank/controller identity and no host timing"
        ),
    }
    archive_path = args.work_dir / f"{case.key}.observations.json"
    _write_json_atomic(archive_path, archive)
    return archive


def _write_json_atomic(
    path: pathlib.Path, value: Mapping[str, object]
) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    with staged.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(staged, path)


def execute_board(
    args: argparse.Namespace,
    case: catalog.CharacterizationCase,
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    completion_manifest: direct_dte_evidence.DirectDTEManifestEvidence,
) -> dict[str, object]:
    launches: list[dict[str, object]] = []
    reference_coordinates: dict[int, tuple[int, int]] | None = None
    archive: dict[str, object] | None = None
    for sample in range(args.repeat):
        launch, coordinates, raw_dir, resource_paths = _execute_launch(
            args,
            case,
            package,
            bindings,
            completion_manifest,
            sample=sample,
            execution_ordinal=sample,
            reference_coordinates=reference_coordinates,
        )
        if reference_coordinates is None:
            reference_coordinates = coordinates
        launches.append(launch)
        archive = _write_case_archive(args, case, launches)
        _discard_validated_resources(
            args.work_dir, raw_dir, resource_paths
        )
    _validate_completed_launches(case, launches, args.repeat)
    archive = _write_case_archive(args, case, launches)
    archive_path = args.work_dir / f"{case.key}.observations.json"
    print(f"ddr_active_tile_archive: {archive_path}")
    return archive


def execute_group(
    args: argparse.Namespace,
    cases: Sequence[catalog.CharacterizationCase],
    package: pathlib.Path,
    bindings: dict[tuple[int, str, int], int],
    completion_manifest: direct_dte_evidence.DirectDTEManifestEvidence,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    launches_by_key: dict[str, list[dict[str, object]]] = {
        case.key: [] for case in cases
    }
    reference_coordinates: dict[int, tuple[int, int]] | None = None
    execution_order: list[dict[str, object]] = []
    for execution_ordinal, (sample, case) in enumerate(
        group_execution_plan(cases, args.repeat)
    ):
        launch, coordinates, raw_dir, resource_paths = _execute_launch(
            args,
            case,
            package,
            bindings,
            completion_manifest,
            sample=sample,
            execution_ordinal=execution_ordinal,
            reference_coordinates=reference_coordinates,
        )
        if reference_coordinates is None:
            reference_coordinates = coordinates
        launches_by_key[case.key].append(launch)
        execution_order.append(
            {
                "execution_ordinal": execution_ordinal,
                "sample": sample,
                "case_key": case.key,
                "active_tile_count": len(case.active_tiles),
                "active_mask": active_mask(case, sample),
            }
        )
        _write_case_archive(args, case, launches_by_key[case.key])
        _discard_validated_resources(
            args.work_dir, raw_dir, resource_paths
        )
    archives: list[dict[str, object]] = []
    for case in cases:
        launches = launches_by_key[case.key]
        _validate_completed_launches(case, launches, args.repeat)
        archives.append(_write_case_archive(args, case, launches))
        print(
            "ddr_active_tile_archive: "
            + str(args.work_dir / f"{case.key}.observations.json")
        )
    return archives, execution_order


def _require_tools(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("wafer_compile", "wafer_run", "llvm_clangxx", "work_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise RuntimeError(
            "no-card/board mode requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    args.repo_root = args.repo_root.resolve()
    args.wafer_compile = args.wafer_compile.resolve()
    args.wafer_run = args.wafer_run.resolve()
    args.llvm_clangxx = args.llvm_clangxx.resolve()
    args.work_dir = _validate_work_dir(args.repo_root, args.work_dir)
    args.conflict_equivalence = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("audit", "no-card", "board"),
        default="audit",
    )
    parser.add_argument("--list", action="store_true", help="list inventory")
    parser.add_argument("--case", help="select one exact stable key")
    parser.add_argument(
        "--group",
        choices=tuple(catalog.DDR_ACTIVE_TILE_GROUP_KEYS),
        help="select one complete five-point active-Tile sweep",
    )
    parser.add_argument(
        "--emit-board-group-keys",
        action="store_true",
        help="print one complete board activation group key per line",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=600000)
    parser.add_argument("--repeat", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalog.validate_catalog()
        validate_static_contract()
        if args.emit_board_group_keys:
            print("\n".join(catalog.DDR_ACTIVE_TILE_GROUP_KEYS))
            return 0
        if args.list:
            print(json.dumps(inventory(), indent=2, sort_keys=True))
            return 0
        if (args.case is None) == (args.group is None):
            raise ValueError(
                "select exactly one --case or one complete --group"
            )
        values = (
            (resolve_object(args.case),)
            if args.case is not None
            else catalog.DDR_ACTIVE_TILE_GROUPS[args.group]
        )
        if args.mode == "audit":
            print(
                json.dumps(
                    {
                        "selection": args.case or args.group,
                        "cases": [
                            prepare_board_selection(value.key)
                            for value in values
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        for value in values:
            if (
                not isinstance(value, catalog.CharacterizationCase)
                or value.disposition
                != catalog.Disposition.BOARD_EXECUTABLE
            ):
                prepare_board_selection(value.key)
                raise AssertionError(
                    "non-executable selection passed host gate"
                )
        _require_tools(args)
        for value in values:
            if args.repeat < value.measurement.minimum_repeats:
                raise RuntimeError(
                    f"{value.key}: repeat must be >= "
                    f"{value.measurement.minimum_repeats}"
                )
            if (
                value.ddr_direction
                == catalog.DDRDirection.BIDIRECTIONAL
                and args.repeat % 2 != 0
            ):
                raise RuntimeError(
                    f"{value.key}: bidirectional repeat must be even"
                )
        if args.group is not None and args.repeat % 4 != 0:
            raise RuntimeError(
                "active-Tile group repeat must be a multiple of four for "
                "position counterbalance"
            )
        if args.mode == "board":
            if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
                print(
                    "DDR active-Tile hardware execution is not armed; set "
                    "WAFER_EXECUTE_HARDWARE_TESTS=1",
                    file=sys.stderr,
                )
                return 77
            cluster_driver.validate_board_args(args)
        package, bindings, completion_manifest = _prepare_package(args)
        if args.mode == "no-card":
            serialized_requests = b"".join(
                request_bytes(value, tile_id, sample)
                for sample in range(args.repeat)
                for value in values
                for tile_id in range(TILE_COUNT)
            )
            print(
                "ddr_active_tile_no_card: "
                + json.dumps(
                    {
                        "case_keys": [value.key for value in values],
                        "serialized_tile_requests": (
                            TILE_COUNT * len(values) * args.repeat
                        ),
                        "request_sha256": hashlib.sha256(
                            serialized_requests
                        ).hexdigest(),
                        "active_masks_by_sample": [
                            {
                                "sample": sample,
                                "masks": [
                                    active_mask(value, sample)
                                    for value in values
                                ],
                            }
                            for sample in range(args.repeat)
                        ],
                        "package": str(package),
                        "board_execution": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        execution_order: list[dict[str, object]] = []
        if args.group is None:
            archives = [
                execute_board(
                    args,
                    value,
                    package,
                    bindings,
                    completion_manifest,
                )
                for value in values
            ]
        else:
            archives, execution_order = execute_group(
                args,
                values,
                package,
                bindings,
                completion_manifest,
            )
        if args.group is not None:
            group_archive = {
                "group": args.group,
                "case_keys": [value.key for value in values],
                "active_tile_counts": [
                    len(value.active_tiles) for value in values
                ],
                "archives": [
                    str(args.work_dir / f"{value.key}.observations.json")
                    for value in values
                ],
                "execution_order": execution_order,
                "completion_by_tile": [
                    {"tile_id": tile_id, "completion": completion}
                    for tile_id, completion in (
                        completion_manifest.completion_by_tile
                    )
                ],
                "execution_order_basis": (
                    "sample-major four-sample forward/reverse/half-rotation "
                    "with equal mean position; not a complete five-position "
                    "Latin rotation"
                ),
                "physical_tile_phase_basis": (
                    "1/2/4/8-active masks shift by sample modulo uniform "
                    "Tile spacing"
                ),
                "measurement_basis": (
                    "per-Tile device rdcycle and target-only PMU; "
                    "complete-Tile-domain max at the same launch boundary"
                ),
                "activation": (
                    "observation-only; cost promotion requires calibration and "
                    "held-out directions to agree across all groups"
                ),
            }
            group_path = (
                args.work_dir / f"{args.group}.group-observations.json"
            )
            _write_json_atomic(group_path, group_archive)
            print(f"ddr_active_tile_group_archive: {group_path}")
        if len(archives) != len(values):
            raise AssertionError("active-Tile archive count differs")
        return 0
    except (
        KeyError,
        OSError,
        PreparationBlocked,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
