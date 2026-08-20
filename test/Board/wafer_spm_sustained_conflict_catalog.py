#!/usr/bin/env python3
"""Typed sustained CT/RDMA SPM pilot with matched port-PMU response.

Pipeline position:
- Upstream IR / input:
  A program-local package with three 2 MiB host-visible resources and a typed
  characterization group selected by work size and reciprocal issue order.
- Current stage responsibility:
  Materialize four same-invocation matched rows (8192/control-4352 times
  serial/window) from compact, disjoint 4 KiB CT/RDMA operand cells.  For
  each row, read back the RDMA-written range with the same number of WDMA
  packets and bytes while sampling the owner-backed SPM port-0/port-6 PMU.
- Output IR / files:
  Exact full-SPM snapshots plus current device records containing actual
  resource addresses, instruction counts, completion state, pair-only NCC
  PMU, and stable raw SPM port-counter snapshots with restored enable scope.
- Downstream consumer:
  The hardware-calibration evidence record and, only after a stable non-zero
  held-out signal, a narrow current-target cost feature.
- User-level driver / named pipeline:
  ``wafer_board_spm_sustained_conflict_probe_test.py``.
- Explicit non-goals:
  This catalog does not name physical banks, infer coloring from offsets,
  infer a counter unit or bank conflict from port response, duplicate the
  existing Direct-DTE port-8 probe, permit overlapping operand ranges, or
  treat no-card output as board evidence.
- Completion gate:
  Every one of the 16 cells belongs to one of four matched groups; each group
  executes all four rows in one invocation for four counterbalanced repeats
  with exact full-footprint and matched WDMA readback, guard, per-window
  count, completion, SPM-PMU enable/restore, stable split-read validation, and
  phase-scoped accepted/pending cleanup evidence that poisons the board batch
  when a bounded safety drain cannot restore the matching worker to idle.
"""

from __future__ import annotations

import dataclasses
import enum
import struct


REQUEST_MAGIC = 0x31514353504D5357
RECORD_MAGIC = 0x31524353504D5357
ROW_MAGIC = 0x31575253504D5357
REQUEST_GUARD = 0x6F4A31C9E2B587D0
RECORD_GUARD = 0x91D8B42E63CA705F
ROW_GUARD = 0x38C5E719A46DB20F
PORT_RESPONSE_GUARD = 0x5A1CE07D93B2468F
CLEANUP_RECORD_GUARD = 0xC47A029D6E18B35F
REQUEST_WORDS = 36
RECORD_WORDS = 32
ROW_RECORD_WORDS = 86
ROW_COUNT = 4
COUNTERBALANCED_REPEATS = 4
RESOURCE_BYTES = 2 * 1024 * 1024
RESOURCE_CANARY = 0xA5
SPM_CANARY = 0x6D

TRANSFER_BYTES = 4096
MAX_DMA_CHUNK = 65536
SPM_BASE = 0x80000
SPM_CELL_STRIDE = 0x8000
SPM_READ0_OFFSET = 0x100
SPM_READ1_OFFSET = 0x2100
SPM_WRITE_OFFSET = 0x4100
CANDIDATE_OFFSET = 8192
CONTROL_OFFSET = 4352
RELATIVE_OFFSETS = (CANDIDATE_OFFSET, CONTROL_OFFSET)
WORK_BYTES = (4096, 16384)
ISSUE_ORDERS = (0, 1)
SCHEDULES = (0, 1)

PAYLOAD_READ0_OFFSET = 0x10000
PAYLOAD_READ1_OFFSET = 0x20000
PAYLOAD_RDMA_OFFSET = 0x30000
PAYLOAD_CANARY_OFFSET = 0x50000
PAYLOAD_CANARY_BYTES = 65536
OUTPUT_ARCHIVE_BASE = 0x10000
OUTPUT_ARCHIVE_STRIDE = 0x20000
PORT_RESPONSE_BASE = 0x100000
PORT_RESPONSE_STRIDE = 0x10000
ROW_RECORD_BASE_WORD = RECORD_WORDS
OUTPUT_RECORD_BYTES = (
    ROW_RECORD_BASE_WORD + ROW_COUNT * ROW_RECORD_WORDS
) * 8

SPM_PORT_PMU_BASE = 0x580000
SPM_PORT_PMU_REQUIRED_ENABLE = 0x1F
SPM_PORT_PMU_COUNTERS = 4
SPM_PORT_PMU_STABLE_MASK = (1 << SPM_PORT_PMU_COUNTERS) - 1
SPM_PORT_PMU_SCOPE_MASK = (1 << 5) - 1

STATUS_OK = 0
FLAG_PREPARED = 1 << 0
FLAG_ISSUED = 1 << 1
FLAG_COMPLETED = 1 << 2
FLAG_PMU_RECORDED = 1 << 3
FLAG_READBACK = 1 << 4
FLAG_EXACT_OWNED_RANGE = 1 << 5
FLAG_PORT_PMU_RECORDED = 1 << 6
FLAG_PORT_MATCHED_READBACK = 1 << 7
FLAG_PORT_ENABLE_RESTORED = 1 << 8
FLAG_CLEANUP_EVIDENCE = 1 << 9
REQUIRED_FLAGS = (
    FLAG_PREPARED
    | FLAG_ISSUED
    | FLAG_COMPLETED
    | FLAG_PMU_RECORDED
    | FLAG_READBACK
    | FLAG_EXACT_OWNED_RANGE
    | FLAG_PORT_PMU_RECORDED
    | FLAG_PORT_MATCHED_READBACK
    | FLAG_PORT_ENABLE_RESTORED
    | FLAG_CLEANUP_EVIDENCE
)
PMU_STABLE_MASK = (1 << 6) - 1
STATUS_CLEANUP_FAILED = 6
CLEANUP_SETUP = 1 << 0
CLEANUP_MEASURED_PAIR = 1 << 1
CLEANUP_MATCHED_WDMA = 1 << 2
CLEANUP_ARCHIVE = 1 << 3
CLEANUP_PHASE_MASK = (
    CLEANUP_SETUP
    | CLEANUP_MEASURED_PAIR
    | CLEANUP_MATCHED_WDMA
    | CLEANUP_ARCHIVE
)


class AddressClass(str, enum.Enum):
    CANDIDATE = "candidate-8192"
    CONTROL = "control-4352"

    @property
    def relative_offset(self) -> int:
        return (
            CANDIDATE_OFFSET
            if self == AddressClass.CANDIDATE
            else CONTROL_OFFSET
        )

    @property
    def row_index(self) -> int:
        return 0 if self == AddressClass.CANDIDATE else 1


class Schedule(str, enum.Enum):
    SERIAL = "serial"
    WINDOW = "window"

    @property
    def wire_value(self) -> int:
        return 0 if self == Schedule.SERIAL else 1


@dataclasses.dataclass(frozen=True)
class ActiveRange:
    role: str
    begin: int
    end: int

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SustainedConflictCell:
    cell_id: int
    key: str
    group_key: str
    work_bytes: int
    rounds: int
    issue_order: int
    address_class: AddressClass
    schedule: Schedule
    relative_offset: int

    @property
    def issue_order_name(self) -> str:
        return "a-b" if self.issue_order == 0 else "b-a"

    @property
    def row_index(self) -> int:
        return self.address_class.row_index * 2 + self.schedule.wire_value

    @property
    def owned_spm_bytes(self) -> int:
        return self.rounds * SPM_CELL_STRIDE

    @property
    def expected_instruction_count(self) -> int:
        return self.rounds

    @property
    def expected_archive_accepted(self) -> int:
        return (
            self.owned_spm_bytes + MAX_DMA_CHUNK - 1
        ) // MAX_DMA_CHUNK

    @property
    def expected_setup_accepted(self) -> int:
        return self.expected_archive_accepted + 2 * self.rounds

    def active_ranges(self) -> tuple[ActiveRange, ...]:
        ranges: list[ActiveRange] = []
        for round_index in range(self.rounds):
            cell_base = SPM_BASE + round_index * SPM_CELL_STRIDE
            ranges.extend(
                (
                    ActiveRange(
                        f"ct-read0-r{round_index}",
                        cell_base + SPM_READ0_OFFSET,
                        cell_base + SPM_READ0_OFFSET + TRANSFER_BYTES,
                    ),
                    ActiveRange(
                        f"ct-read1-r{round_index}",
                        cell_base + SPM_READ1_OFFSET,
                        cell_base + SPM_READ1_OFFSET + TRANSFER_BYTES,
                    ),
                    ActiveRange(
                        f"ct-write-r{round_index}",
                        cell_base + SPM_WRITE_OFFSET,
                        cell_base + SPM_WRITE_OFFSET + TRANSFER_BYTES,
                    ),
                    ActiveRange(
                        f"rdma-write-r{round_index}",
                        cell_base
                        + SPM_WRITE_OFFSET
                        + self.relative_offset,
                        cell_base
                        + SPM_WRITE_OFFSET
                        + self.relative_offset
                        + TRANSFER_BYTES,
                    ),
                )
            )
        return tuple(ranges)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.cell_id,
            "key": self.key,
            "group": self.group_key,
            "work_bytes": self.work_bytes,
            "transfer_bytes": TRANSFER_BYTES,
            "rounds": self.rounds,
            "issue_order": self.issue_order_name,
            "address_class": self.address_class.value,
            "relative_offset": self.relative_offset,
            "schedule": self.schedule.value,
            "owned_spm_begin": SPM_BASE,
            "owned_spm_bytes": self.owned_spm_bytes,
            "active_ranges": [
                active_range.as_dict()
                for active_range in self.active_ranges()
            ],
            "repeats": COUNTERBALANCED_REPEATS,
            "execution": (
                "four matched rows share one request/payload/output "
                "allocation and one owned SPM envelope per invocation"
            ),
            "oracle": (
                "full-owned-SPM-exact+gap/prefix/suffix-canary+"
                "matched-rdma/wdma-readback+per-window-instruction-count+"
                "bounded-completion+pair-only-NCC-PMU+raw-port-response+"
                "partial-accept-safety-drain-or-poison"
            ),
            "spm_port_response": {
                "mmio_base": SPM_PORT_PMU_BASE,
                "rdma_port": 0,
                "wdma_port": 6,
                "pair_window": {
                    "ct_packets": self.rounds,
                    "rdma_packets": self.rounds,
                    "rdma_requested_bytes": self.work_bytes,
                    "ct_counter_unit": "unclassified",
                },
                "matched_readback_window": {
                    "wdma_packets": self.rounds,
                    "wdma_requested_bytes": self.work_bytes,
                },
                "split_read": "high-low-high",
                "counter_unit": "unclassified",
                "bank_identity": "not-observed",
            },
            "disposition": "pending-board-execution",
            "compiler_use": "no-bank-coloring-until-stable-nonzero-heldout",
        }


@dataclasses.dataclass(frozen=True)
class SustainedConflictGroup:
    group_id: int
    key: str
    work_bytes: int
    rounds: int
    issue_order: int
    cells: tuple[SustainedConflictCell, ...]

    @property
    def issue_order_name(self) -> str:
        return "a-b" if self.issue_order == 0 else "b-a"

    @property
    def owned_spm_bytes(self) -> int:
        return self.rounds * SPM_CELL_STRIDE

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.group_id,
            "key": self.key,
            "work_bytes": self.work_bytes,
            "transfer_bytes": TRANSFER_BYTES,
            "rounds": self.rounds,
            "issue_order": self.issue_order_name,
            "cell_keys": [cell.key for cell in self.cells],
            "repeats": COUNTERBALANCED_REPEATS,
        }


def _build_inventory() -> tuple[
    tuple[SustainedConflictCell, ...],
    tuple[SustainedConflictGroup, ...],
]:
    cells: list[SustainedConflictCell] = []
    groups: list[SustainedConflictGroup] = []
    for work_index, work_bytes in enumerate(WORK_BYTES):
        rounds = work_bytes // TRANSFER_BYTES
        for issue_order in ISSUE_ORDERS:
            group_id = work_index * len(ISSUE_ORDERS) + issue_order
            order_name = "a-b" if issue_order == 0 else "b-a"
            group_key = (
                f"spm-conflict-pilot-{work_bytes // 1024}k-"
                f"{order_name}"
            )
            group_cells: list[SustainedConflictCell] = []
            for address_class in AddressClass:
                for schedule in Schedule:
                    cell_id = (
                        work_index * 8
                        + issue_order * 4
                        + address_class.row_index * 2
                        + schedule.wire_value
                    )
                    cell = SustainedConflictCell(
                        cell_id=cell_id,
                        key=(
                            f"{group_key}-{address_class.value}-"
                            f"{schedule.value}"
                        ),
                        group_key=group_key,
                        work_bytes=work_bytes,
                        rounds=rounds,
                        issue_order=issue_order,
                        address_class=address_class,
                        schedule=schedule,
                        relative_offset=address_class.relative_offset,
                    )
                    cells.append(cell)
                    group_cells.append(cell)
            groups.append(
                SustainedConflictGroup(
                    group_id=group_id,
                    key=group_key,
                    work_bytes=work_bytes,
                    rounds=rounds,
                    issue_order=issue_order,
                    cells=tuple(group_cells),
                )
            )
    return tuple(cells), tuple(groups)


CELLS, GROUPS = _build_inventory()
CELLS_BY_KEY = {cell.key: cell for cell in CELLS}
CELLS_BY_ID = {cell.cell_id: cell for cell in CELLS}
GROUPS_BY_KEY = {group.key: group for group in GROUPS}
GROUPS_BY_ID = {group.group_id: group for group in GROUPS}


REQ = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
    "GROUP": 2,
    "WORK_BYTES": 3,
    "ROUNDS": 4,
    "ISSUE_ORDER": 5,
    "SAMPLE": 6,
    "REPEATS": 7,
    "ROWS": 8,
    "TRANSFER_BYTES": 9,
    "SPM_BASE": 10,
    "SPM_CELL_STRIDE": 11,
    "SPM_READ0_OFFSET": 12,
    "SPM_READ1_OFFSET": 13,
    "SPM_WRITE_OFFSET": 14,
    "CANDIDATE_OFFSET": 15,
    "CONTROL_OFFSET": 16,
    "OWNED_SPM_BYTES": 17,
    "RESOURCE_BYTES": 18,
    "PAYLOAD_READ0_OFFSET": 19,
    "PAYLOAD_READ1_OFFSET": 20,
    "PAYLOAD_RDMA_OFFSET": 21,
    "PAYLOAD_CANARY_OFFSET": 22,
    "PAYLOAD_CANARY_BYTES": 23,
    "OUTPUT_ARCHIVE_BASE": 24,
    "OUTPUT_ARCHIVE_STRIDE": 25,
    "ROW_RECORD_BASE_WORD": 26,
    "ROW_RECORD_WORDS": 27,
    "RESOURCE_CANARY": 28,
    "SPM_CANARY": 29,
    "GUARD": 30,
    "PORT_RESPONSE_BASE": 31,
    "PORT_RESPONSE_STRIDE": 32,
    "SPM_PORT_PMU_BASE": 33,
    "SPM_PORT_PMU_REQUIRED_ENABLE": 34,
    "RESERVED": 35,
}

REC = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
    "STATUS": 2,
    "GROUP": 3,
    "WORK_BYTES": 4,
    "ROUNDS": 5,
    "ISSUE_ORDER": 6,
    "SAMPLE": 7,
    "ROWS": 8,
    "COMPLETED_ROWS": 9,
    "TRANSFER_BYTES": 10,
    "SPM_BASE": 11,
    "OWNED_SPM_BYTES": 12,
    "REQUEST_DDR": 13,
    "PAYLOAD_DDR": 14,
    "OUTPUT_DDR": 15,
    "EXECUTION_ROTATION": 16,
    "FLAGS": 17,
    "REQUEST_GUARD": 18,
    "RECORD_GUARD": 19,
}

ROW_REC = {
    "MAGIC": 0,
    "WORD_COUNT": 1,
    "STATUS": 2,
    "CELL": 3,
    "ADDRESS_CLASS": 4,
    "RELATIVE_OFFSET": 5,
    "SCHEDULE": 6,
    "ISSUE_ORDER": 7,
    "SAMPLE": 8,
    "EXECUTION_ORDINAL": 9,
    "ROUNDS": 10,
    "TRANSFER_BYTES": 11,
    "SPM_BASE": 12,
    "OWNED_SPM_BYTES": 13,
    "SPM_CELL_STRIDE": 14,
    "SPM_READ0_OFFSET": 15,
    "SPM_READ1_OFFSET": 16,
    "SPM_WRITE_OFFSET": 17,
    "RDMA_WRITE_OFFSET": 18,
    "ARCHIVE_OFFSET": 19,
    "REQUEST_DDR": 20,
    "PAYLOAD_DDR": 21,
    "OUTPUT_DDR": 22,
    "CT_INST_DELTA": 23,
    "RDMA_INST_DELTA": 24,
    "OTHER_INST_DELTA": 25,
    "WORKER_CT_INST_DELTA": 26,
    "WORKER_RDMA_INST_DELTA": 27,
    "CT_EXEC_DELTA": 28,
    "RDMA_EXEC_DELTA": 29,
    "FULL_EXEC_DELTA": 30,
    "CT_BLOCKING_DELTA": 31,
    "RDMA_BLOCKING_DELTA": 32,
    "PLAN_CYCLES": 33,
    "PMU_ENABLE": 34,
    "SERIAL_MODE": 35,
    "STABLE_BEFORE": 36,
    "STABLE_AFTER": 37,
    "FINAL_CONTROL": 38,
    "FLAGS": 39,
    "REQUEST_GUARD": 40,
    "ROW_GUARD": 41,
    "SPM_PORT_PMU_BASE": 42,
    "SPM_PORT_PMU_REQUIRED_ENABLE": 43,
    "SPM_PORT_PMU_ENABLE_ORIGINAL": 44,
    "SPM_PORT_PMU_ENABLE_BEFORE": 45,
    "SPM_PORT_PMU_ENABLE_BOUNDARY": 46,
    "SPM_PORT_PMU_ENABLE_AFTER": 47,
    "SPM_PORT_PMU_ENABLE_RESTORED": 48,
    "SPM_PORT_PMU_STABLE_BEFORE": 49,
    "SPM_PORT_PMU_STABLE_BOUNDARY": 50,
    "SPM_PORT_PMU_STABLE_AFTER": 51,
    "SPM_PORT0_T2_BEFORE": 52,
    "SPM_PORT0_T2_BOUNDARY": 53,
    "SPM_PORT0_T2_AFTER": 54,
    "SPM_PORT0_T3_BEFORE": 55,
    "SPM_PORT0_T3_BOUNDARY": 56,
    "SPM_PORT0_T3_AFTER": 57,
    "SPM_PORT6_T2_BEFORE": 58,
    "SPM_PORT6_T2_BOUNDARY": 59,
    "SPM_PORT6_T2_AFTER": 60,
    "SPM_PORT6_T3_BEFORE": 61,
    "SPM_PORT6_T3_BOUNDARY": 62,
    "SPM_PORT6_T3_AFTER": 63,
    "WDMA_INST_DELTA": 64,
    "WDMA_OTHER_INST_DELTA": 65,
    "WORKER_WDMA_INST_DELTA": 66,
    "WDMA_FINAL_CONTROL": 67,
    "PORT_READBACK_OFFSET": 68,
    "PORT_READBACK_BYTES": 69,
    "SPM_PORT_PMU_SCOPE_FLAGS": 70,
    "PORT_RESPONSE_GUARD": 71,
    "SETUP_ACCEPTED": 72,
    "SETUP_PENDING_AFTER": 73,
    "SETUP_FINAL_CONTROL": 74,
    "MEASURED_PAIR_ACCEPTED": 75,
    "MEASURED_PAIR_PENDING_AFTER": 76,
    "MATCHED_WDMA_ACCEPTED": 77,
    "MATCHED_WDMA_PENDING_AFTER": 78,
    "ARCHIVE_ACCEPTED": 79,
    "ARCHIVE_PENDING_AFTER": 80,
    "ARCHIVE_FINAL_CONTROL": 81,
    "CLEANUP_ATTEMPTED_MASK": 82,
    "CLEANUP_SUCCEEDED_MASK": 83,
    "POISONED_PHASE_MASK": 84,
    "CLEANUP_RECORD_GUARD": 85,
}


@dataclasses.dataclass(frozen=True)
class InvocationPayload:
    group: SustainedConflictGroup
    sample: int
    request: bytes
    payload: bytes
    expected_snapshots: dict[int, bytes]
    expected_port_readback: bytes


def _pattern(sample: int, round_index: int, length: int) -> bytes:
    return bytes(
        (
            sample * 53
            + round_index * 71
            + index * 17
            + (index >> 6) * 29
            + 0x3B
        )
        & 0xFF
        for index in range(length)
    )


def _fp16_constant(raw_bits: int) -> bytes:
    return struct.pack("<H", raw_bits) * (TRANSFER_BYTES // 2)


def expected_snapshot(
    group: SustainedConflictGroup, relative_offset: int, sample: int
) -> bytes:
    snapshot = bytearray([SPM_CANARY] * group.owned_spm_bytes)
    read0 = _fp16_constant(0x3C00)
    read1 = _fp16_constant(0x4000)
    result = _fp16_constant(0x4200)
    for round_index in range(group.rounds):
        cell = round_index * SPM_CELL_STRIDE
        snapshot[
            cell + SPM_READ0_OFFSET :
            cell + SPM_READ0_OFFSET + TRANSFER_BYTES
        ] = read0
        snapshot[
            cell + SPM_READ1_OFFSET :
            cell + SPM_READ1_OFFSET + TRANSFER_BYTES
        ] = read1
        snapshot[
            cell + SPM_WRITE_OFFSET :
            cell + SPM_WRITE_OFFSET + TRANSFER_BYTES
        ] = result
        rdma_begin = cell + SPM_WRITE_OFFSET + relative_offset
        snapshot[rdma_begin : rdma_begin + TRANSFER_BYTES] = _pattern(
            sample, round_index, TRANSFER_BYTES
        )
    return bytes(snapshot)


def expected_port_readback(
    group: SustainedConflictGroup, sample: int
) -> bytes:
    return b"".join(
        _pattern(sample, round_index, TRANSFER_BYTES)
        for round_index in range(group.rounds)
    )


def build_invocation(
    group: SustainedConflictGroup, sample: int
) -> InvocationPayload:
    if sample not in range(COUNTERBALANCED_REPEATS):
        raise RuntimeError(f"{group.key}: invalid sample {sample}")
    request = bytearray([RESOURCE_CANARY] * RESOURCE_BYTES)
    words = [0] * REQUEST_WORDS
    values = {
        "MAGIC": REQUEST_MAGIC,
        "WORD_COUNT": REQUEST_WORDS,
        "GROUP": group.group_id,
        "WORK_BYTES": group.work_bytes,
        "ROUNDS": group.rounds,
        "ISSUE_ORDER": group.issue_order,
        "SAMPLE": sample,
        "REPEATS": COUNTERBALANCED_REPEATS,
        "ROWS": ROW_COUNT,
        "TRANSFER_BYTES": TRANSFER_BYTES,
        "SPM_BASE": SPM_BASE,
        "SPM_CELL_STRIDE": SPM_CELL_STRIDE,
        "SPM_READ0_OFFSET": SPM_READ0_OFFSET,
        "SPM_READ1_OFFSET": SPM_READ1_OFFSET,
        "SPM_WRITE_OFFSET": SPM_WRITE_OFFSET,
        "CANDIDATE_OFFSET": CANDIDATE_OFFSET,
        "CONTROL_OFFSET": CONTROL_OFFSET,
        "OWNED_SPM_BYTES": group.owned_spm_bytes,
        "RESOURCE_BYTES": RESOURCE_BYTES,
        "PAYLOAD_READ0_OFFSET": PAYLOAD_READ0_OFFSET,
        "PAYLOAD_READ1_OFFSET": PAYLOAD_READ1_OFFSET,
        "PAYLOAD_RDMA_OFFSET": PAYLOAD_RDMA_OFFSET,
        "PAYLOAD_CANARY_OFFSET": PAYLOAD_CANARY_OFFSET,
        "PAYLOAD_CANARY_BYTES": PAYLOAD_CANARY_BYTES,
        "OUTPUT_ARCHIVE_BASE": OUTPUT_ARCHIVE_BASE,
        "OUTPUT_ARCHIVE_STRIDE": OUTPUT_ARCHIVE_STRIDE,
        "ROW_RECORD_BASE_WORD": ROW_RECORD_BASE_WORD,
        "ROW_RECORD_WORDS": ROW_RECORD_WORDS,
        "RESOURCE_CANARY": RESOURCE_CANARY,
        "SPM_CANARY": SPM_CANARY,
        "GUARD": REQUEST_GUARD,
        "PORT_RESPONSE_BASE": PORT_RESPONSE_BASE,
        "PORT_RESPONSE_STRIDE": PORT_RESPONSE_STRIDE,
        "SPM_PORT_PMU_BASE": SPM_PORT_PMU_BASE,
        "SPM_PORT_PMU_REQUIRED_ENABLE": SPM_PORT_PMU_REQUIRED_ENABLE,
        "RESERVED": 0,
    }
    for key, value in values.items():
        words[REQ[key]] = value
    struct.pack_into(f"<{REQUEST_WORDS}Q", request, 0, *words)

    payload = bytearray([RESOURCE_CANARY] * RESOURCE_BYTES)
    read0 = _fp16_constant(0x3C00)
    read1 = _fp16_constant(0x4000)
    for round_index in range(group.rounds):
        begin0 = PAYLOAD_READ0_OFFSET + round_index * TRANSFER_BYTES
        begin1 = PAYLOAD_READ1_OFFSET + round_index * TRANSFER_BYTES
        begin_dma = PAYLOAD_RDMA_OFFSET + round_index * TRANSFER_BYTES
        payload[begin0 : begin0 + TRANSFER_BYTES] = read0
        payload[begin1 : begin1 + TRANSFER_BYTES] = read1
        payload[begin_dma : begin_dma + TRANSFER_BYTES] = _pattern(
            sample, round_index, TRANSFER_BYTES
        )
    payload[
        PAYLOAD_CANARY_OFFSET :
        PAYLOAD_CANARY_OFFSET + PAYLOAD_CANARY_BYTES
    ] = bytes([SPM_CANARY]) * PAYLOAD_CANARY_BYTES

    return InvocationPayload(
        group=group,
        sample=sample,
        request=bytes(request),
        payload=bytes(payload),
        expected_snapshots={
            relative_offset: expected_snapshot(
                group, relative_offset, sample
            )
            for relative_offset in RELATIVE_OFFSETS
        },
        expected_port_readback=expected_port_readback(group, sample),
    )


def validate_static_contract() -> None:
    if (
        len(CELLS) != 16
        or len(GROUPS) != 4
        or set(CELLS_BY_ID) != set(range(16))
        or set(GROUPS_BY_ID) != set(range(4))
        or any(len(group.cells) != ROW_COUNT for group in GROUPS)
        or any(
            {cell.row_index for cell in group.cells}
            != set(range(ROW_COUNT))
            for group in GROUPS
        )
        or OUTPUT_ARCHIVE_BASE
        + ROW_COUNT * OUTPUT_ARCHIVE_STRIDE
        > PORT_RESPONSE_BASE
        or PORT_RESPONSE_BASE
        + ROW_COUNT * PORT_RESPONSE_STRIDE
        > RESOURCE_BYTES
        or max(WORK_BYTES) > PORT_RESPONSE_STRIDE
        or OUTPUT_RECORD_BYTES > OUTPUT_ARCHIVE_BASE
        or PAYLOAD_CANARY_OFFSET + PAYLOAD_CANARY_BYTES
        > RESOURCE_BYTES
    ):
        raise RuntimeError("sustained conflict inventory is not bounded")

    for cell in CELLS:
        ranges = cell.active_ranges()
        if (
            len(ranges) != 4 * cell.rounds
            or any(
                active.begin < SPM_BASE
                or active.end > SPM_BASE + cell.owned_spm_bytes
                or active.begin >= active.end
                for active in ranges
            )
        ):
            raise RuntimeError(f"{cell.key}: active range leaves owned SPM")
        ordered = sorted(ranges, key=lambda active: active.begin)
        for left, right in zip(ordered, ordered[1:]):
            if left.end > right.begin:
                raise RuntimeError(
                    f"{cell.key}: {left.role} overlaps {right.role}"
                )
        snapshot = expected_snapshot(
            GROUPS_BY_KEY[cell.group_key],
            cell.relative_offset,
            0,
        )
        port_readback = expected_port_readback(
            GROUPS_BY_KEY[cell.group_key], 0
        )
        if (
            len(snapshot) != cell.owned_spm_bytes
            or snapshot[:64] != bytes([SPM_CANARY]) * 64
            or snapshot[-64:] != bytes([SPM_CANARY]) * 64
            or len(port_readback) != cell.work_bytes
        ):
            raise RuntimeError(f"{cell.key}: full snapshot guard is invalid")


validate_static_contract()
