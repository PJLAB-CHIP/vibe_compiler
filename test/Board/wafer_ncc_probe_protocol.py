"""Typed host view of the bounded NCC probe wire protocol.

Numeric protocol values are loaded from ``wafer_ncc_probe_protocol.h`` so the
device and host do not maintain parallel schema constants.  This module is
deliberately independent of named workloads: catalogs build typed lane plans,
while packet construction and golden semantics stay in engine adapters.
"""

from __future__ import annotations

import dataclasses
import enum
import pathlib
import re
from collections.abc import Iterable, Sequence


PROTOCOL_HEADER = (
    pathlib.Path(__file__).resolve().parent
    / "Inputs"
    / "wafer_ncc_probe_protocol.h"
)
_HEADER_TEXT = PROTOCOL_HEADER.read_text()


def _c_integer(expression: str) -> int:
    expression = re.sub(r"UINT(?:32|64)_C\(([^)]+)\)", r"\1", expression)
    expression = re.sub(
        r"(?<=\d)[uUlL]+\b|(?<=[a-fA-F0-9])[uUlL]+\b", "", expression
    )
    if re.fullmatch(r"[\s0-9a-fA-FxX()<>|+-]+", expression) is None:
        raise RuntimeError(f"unsupported protocol integer: {expression!r}")
    return int(eval(expression, {"__builtins__": {}}, {}))


def _macro(name: str) -> int:
    match = re.search(
        rf"^#define\s+{re.escape(name)}\s+(.+?)\s*$",
        _HEADER_TEXT,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"protocol macro {name} is missing")
    return _c_integer(match.group(1))


def _enumerator(name: str) -> int:
    match = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*([^,]+),?\s*$",
        _HEADER_TEXT,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"protocol enumerator {name} is missing")
    return _c_integer(match.group(1))


REQUEST_MAGIC = _macro("WAFER_NCC_PROTOCOL_REQUEST_MAGIC")
RECORD_MAGIC = _macro("WAFER_NCC_PROTOCOL_RECORD_MAGIC")
SCHEMA = _macro("WAFER_NCC_PROTOCOL_SCHEMA")
REQUEST_WORDS = _macro("WAFER_NCC_PROTOCOL_REQUEST_WORDS")
RECORD_WORDS = _macro("WAFER_NCC_PROTOCOL_RECORD_WORDS")
MAX_LANES = _macro("WAFER_NCC_PROTOCOL_MAX_LANES")
MAX_ROUNDS = _macro("WAFER_NCC_PROTOCOL_MAX_ROUNDS")
MAX_THREE_LANE_ROUNDS = _macro(
    "WAFER_NCC_PROTOCOL_MAX_THREE_LANE_ROUNDS"
)
MAX_ISSUES = _macro("WAFER_NCC_PROTOCOL_MAX_ISSUES")
WORKERS = _macro("WAFER_NCC_PROTOCOL_WORKERS")
MAX_DMA_ENVELOPE_BYTES = _macro(
    "WAFER_NCC_PROTOCOL_MAX_DMA_ENVELOPE_BYTES"
)
DMA_FORMAT_INT8 = _macro("WAFER_NCC_PROTOCOL_DMA_FORMAT_INT8")
DMA_FORMAT_FP16 = _macro("WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16")
DMA_FORMAT_UINT8 = _macro("WAFER_NCC_PROTOCOL_DMA_FORMAT_UINT8")
REQUEST_RESERVED_BASE = _macro(
    "WAFER_NCC_PROTOCOL_REQUEST_RESERVED_BASE"
)
REQUEST_RESERVED_WORDS = _macro(
    "WAFER_NCC_PROTOCOL_REQUEST_RESERVED_WORDS"
)
LANE_BASE = _macro("WAFER_NCC_PROTOCOL_LANE_BASE")
LANE_STRIDE = _macro("WAFER_NCC_PROTOCOL_LANE_STRIDE")
ISSUE_BASE = _macro("WAFER_NCC_PROTOCOL_ISSUE_BASE")
ISSUE_STRIDE = _macro("WAFER_NCC_PROTOCOL_ISSUE_STRIDE")
WAIT_SAMPLE_BASE = _macro("WAFER_NCC_PROTOCOL_WAIT_SAMPLE_BASE")
MAX_WAIT_SAMPLES = _macro("WAFER_NCC_PROTOCOL_MAX_WAIT_SAMPLES")
TIGHT_DEPTH_PLUS_ONE = _enumerator(
    "WAFER_NCC_REQUEST_TIGHT_DEPTH_PLUS_ONE"
)
CONSTRUCTOR_OBSERVATION = _enumerator(
    "WAFER_NCC_REQUEST_CONSTRUCTOR_OBSERVATION"
)
ORDERED_PRODUCER_CONSUMER = _enumerator(
    "WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER"
)
DOUBLE_SLOT_OBSERVATION = _enumerator(
    "WAFER_NCC_REQUEST_DOUBLE_SLOT_OBSERVATION"
)
MAPPED_SPM_KCORE_WRITE = _enumerator(
    "WAFER_NCC_REQUEST_MAPPED_SPM_KCORE_WRITE"
)
PREISSUE_LOCAL_WAIT = _enumerator(
    "WAFER_NCC_REQUEST_PREISSUE_LOCAL_WAIT"
)
TIGHT_KCORE_BOUNDARY = _enumerator(
    "WAFER_NCC_REQUEST_TIGHT_KCORE_BOUNDARY"
)
TIGHT_QUEUE_SATURATION = _enumerator(
    "WAFER_NCC_REQUEST_TIGHT_QUEUE_SATURATION"
)
TIGHT_WORKER_SCOPE = _enumerator(
    "WAFER_NCC_REQUEST_TIGHT_WORKER_SCOPE"
)
BOUNDED_PAIR_WINDOW = _enumerator(
    "WAFER_NCC_REQUEST_BOUNDED_PAIR_WINDOW"
)


def _typed_enum(
    class_name: str, prefix: str, members: Iterable[str]
) -> type[enum.IntEnum]:
    return enum.IntEnum(
        class_name,
        {
            member: _enumerator(f"{prefix}{member}")
            for member in members
        },
    )


Command = _typed_enum(
    "Command", "WAFER_NCC_COMMAND_", ("QUALIFY", "EXECUTE")
)
Engine = _typed_enum(
    "Engine",
    "WAFER_NCC_ENGINE_",
    ("CT", "NE", "RDMA", "WDMA", "TDMA", "NONE"),
)
EffectRelation = _typed_enum(
    "EffectRelation",
    "WAFER_NCC_EFFECT_",
    ("NONE", "RAW", "WAR", "WAW", "RAR"),
)
Operand = _typed_enum(
    "Operand",
    "WAFER_NCC_OPERAND_",
    ("READ0", "READ1", "WRITE", "AUTO"),
)
RangeRelation = _typed_enum(
    "RangeRelation",
    "WAFER_NCC_RANGE_",
    ("DISJOINT", "EXACT", "PARTIAL", "ADJACENT", "STRIDED_ENVELOPE"),
)
Schedule = _typed_enum(
    "Schedule", "WAFER_NCC_SCHEDULE_", ("WINDOW", "SERIAL")
)
WaitKind = _typed_enum(
    "WaitKind",
    "WAFER_NCC_WAIT_",
    ("NONE", "BY_WORKER", "DEFAULT", "LOCAL_FENCE"),
)
IssueMode = _typed_enum(
    "IssueMode", "WAFER_NCC_ISSUE_", ("RAW", "WRAPPER")
)
LayoutKind = _typed_enum(
    "LayoutKind",
    "WAFER_NCC_LAYOUT_",
    ("CONTIGUOUS", "INNER_STRIDED", "DMA_STRIDED"),
)
Status = _typed_enum(
    "Status",
    "WAFER_NCC_STATUS_",
    (
        "OK",
        "BAD_REQUEST",
        "MISSING_ADAPTER",
        "UNSUPPORTED_COMBINATION",
        "UNSAFE_WINDOW",
        "SEED_FAILED",
        "PREPARE_FAILED",
        "ISSUE_FAILED",
        "OBSERVATION_FAILED",
        "WAIT_FAILED",
        "DRAIN_FAILED",
    ),
)


def _word(prefix: str, name: str) -> int:
    return _enumerator(f"{prefix}{name}")


REQ = {
    name: _word("WAFER_NCC_REQ_", name)
    for name in (
        "MAGIC",
        "SCHEMA_AND_WORDS",
        "COMMAND",
        "LANE_COUNT",
        "ROUNDS",
        "EFFECT_RELATION",
        "RANGE_RELATION",
        "SCHEDULE",
        "WAIT_KIND",
        "WAIT_WORKER_MASK",
        "SEED",
        "SAMPLE",
        "FLAGS",
        "FIRST_OPERAND",
        "SECOND_OPERAND",
        "ISSUE_LIMIT",
    )
}
LANE = {
    name: _word("WAFER_NCC_LANE_", name)
    for name in (
        "ENGINE",
        "WORKER",
        "ISSUE_MODE",
        "TRANSFER_BYTES",
        "ELEMENT_FORMAT",
        "LAYOUT_KIND",
        "LAYOUT_INNER_BYTES",
        "LAYOUT_STRIDE0_BYTES",
        "LAYOUT_STRIDE1_BYTES",
        "LAYOUT_STRIDE2_BYTES",
        "LAYOUT_ITERATION0",
        "LAYOUT_ITERATION1",
        "LAYOUT_ITERATION2",
        "FLAGS",
    )
}
REC = {
    name: _word("WAFER_NCC_REC_", name)
    for name in (
        "MAGIC",
        "SCHEMA_AND_WORDS",
        "STATUS",
        "FLAGS",
        "COMMAND",
        "LANE_COUNT",
        "ROUNDS",
        "ISSUE_COUNT",
        "EFFECT_RELATION",
        "RANGE_RELATION",
        "SCHEDULE",
        "WAIT_KIND",
        "WAIT_WORKER_MASK",
        "SAFETY_WORKER_MASK",
        "SEED",
        "SAMPLE",
        "BOUNDARY_MISMATCHES",
        "FINAL_MISMATCHES",
        "BOUNDARY_GUARD_MISMATCHES",
        "FINAL_GUARD_MISMATCHES",
        "STABLE_BEFORE",
        "STABLE_AFTER",
        "FIRST_OPERAND",
        "SECOND_OPERAND",
        "CONTROL_BEFORE",
        "CONTROL_BOUNDARY",
        "CONTROL_FINAL",
        "PMU64_BEFORE",
        "PMU64_AFTER",
        "INSTRUCTION_BEFORE",
        "INSTRUCTION_AFTER",
        "BLOCKING_BEFORE",
        "BLOCKING_AFTER",
        "PMU_ENABLE",
        "SERIAL_MODE",
        "OUTPUT_SLOT_BASE",
        "OUTPUT_SLOT_STRIDE",
        "OUTPUT_GUARD_BYTES",
        "RESOURCE_BYTES",
        "ISSUE_LIMIT",
        "WAIT_CYCLES",
        "COMPLETION_MARKER_EXPECTED",
        "COMPLETION_MARKER_BOUNDARY",
        "COMPLETION_MARKER_FINAL",
        "COMPLETION_MARKER_ADDRESS",
        "PLAN_CYCLES",
        "SERIAL_WAIT_CYCLES",
        "SERIAL_WAIT_COUNT",
        "CONSTRUCTOR_ADDRESS",
        "RECORD_GUARD",
        "PREISSUE_WAIT_CYCLES",
        "CONTROL_PRE_WAIT",
        "BOUNDED_WINDOW_DRAIN_COUNT",
        "BOUNDED_WINDOW_DRAIN_CYCLES",
    )
}
ISSUE = {
    name: _word("WAFER_NCC_ISSUE_", name)
    for name in (
        "ORDINAL",
        "LANE",
        "ROUND",
        "SLOT",
        "ENGINE",
        "WORKER",
        "TAG",
        "EXECUTE_RC",
        "INTER_TYPE",
        "READ0_BEGIN",
        "READ0_END",
        "READ1_BEGIN",
        "READ1_END",
        "WRITE_BEGIN",
        "WRITE_END",
        "BOUNDARY_MISMATCHES",
        "BOUNDARY_GUARD_MISMATCHES",
        "FINAL_MISMATCHES",
        "FINAL_GUARD_MISMATCHES",
        "FLAGS",
        "CONTROL_AFTER_ISSUE",
        "EXECUTE_CYCLES",
    )
}

READ0_VALID = _enumerator("WAFER_NCC_ISSUE_READ0_VALID")
READ1_VALID = _enumerator("WAFER_NCC_ISSUE_READ1_VALID")
WRITE_VALID = _enumerator("WAFER_NCC_ISSUE_WRITE_VALID")
PACKET_OBSERVED = _enumerator("WAFER_NCC_ISSUE_PACKET_OBSERVED")
WINDOW_CONTROL_VALID = _enumerator(
    "WAFER_NCC_ISSUE_WINDOW_CONTROL_VALID"
)
PREPARE_STAGE_FLAGS = {
    name: _macro(f"WAFER_NCC_ISSUE_PREPARE_{name}")
    for name in (
        "ENTERED",
        "BUILDER_ACQUIRED",
        "PACKET_MATERIALIZED",
        "BUILDER_RELEASED",
        "COMPLETED",
    )
}
PREPARE_STAGE_MASK = sum(PREPARE_STAGE_FLAGS.values())
CONSTRUCTOR_CAPTURED = _enumerator(
    "WAFER_NCC_RECORD_CONSTRUCTOR_CAPTURED"
)
PREISSUE_LOCAL_WAIT_DONE = _enumerator(
    "WAFER_NCC_RECORD_PREISSUE_LOCAL_WAIT_DONE"
)
PRE_WAIT_CAPTURED = _enumerator(
    "WAFER_NCC_RECORD_PRE_WAIT_CAPTURED"
)
ALL_PHASE_FLAGS = sum(
    _enumerator(name)
    for name in (
        "WAFER_NCC_RECORD_BEFORE_CAPTURED",
        "WAFER_NCC_RECORD_REQUESTED_WAIT_DONE",
        "WAFER_NCC_RECORD_BOUNDARY_CAPTURED",
        "WAFER_NCC_RECORD_BOUNDARY_ORACLE_DONE",
        "WAFER_NCC_RECORD_SAFETY_DRAIN_DONE",
        "WAFER_NCC_RECORD_FINAL_CAPTURED",
        "WAFER_NCC_RECORD_FINAL_ORACLE_DONE",
    )
)


@dataclasses.dataclass(frozen=True)
class Lane:
    engine: Engine
    worker: int
    issue_mode: IssueMode
    transfer_bytes: int
    element_format: int
    layout_kind: LayoutKind = LayoutKind.CONTIGUOUS
    layout_inner_bytes: int = 0
    layout_stride0_bytes: int = 0
    layout_stride1_bytes: int = 0
    layout_stride2_bytes: int = 0
    layout_iteration0: int = 0
    layout_iteration1: int = 0
    layout_iteration2: int = 0
    flags: int = 0

    def validate(self) -> None:
        if self.engine == Engine.NONE:
            raise ValueError("an active lane cannot use Engine.NONE")
        if not 0 <= self.worker < WORKERS:
            raise ValueError(f"worker must be in [0, {WORKERS})")
        if self.transfer_bytes <= 0:
            raise ValueError("transfer_bytes must be positive")
        if self.flags != 0:
            raise ValueError("unknown lane flags are not safe")
        extra_layout = (
            self.layout_stride0_bytes,
            self.layout_stride1_bytes,
            self.layout_stride2_bytes,
            self.layout_iteration0,
            self.layout_iteration1,
            self.layout_iteration2,
        )
        if any(type(value) is not int or value < 0 for value in extra_layout):
            raise ValueError("layout strides and iterations must be nonnegative")
        if self.layout_kind == LayoutKind.CONTIGUOUS:
            if self.layout_inner_bytes != 0 or any(extra_layout):
                raise ValueError("contiguous lanes have no layout descriptor")
        elif self.layout_kind == LayoutKind.INNER_STRIDED:
            if any(extra_layout):
                raise ValueError(
                    "inner-strided lanes do not carry a DMA descriptor"
                )
            if (
                self.layout_inner_bytes <= 0
                or self.layout_inner_bytes > self.transfer_bytes
                or self.transfer_bytes % self.layout_inner_bytes
            ):
                raise ValueError(
                    "inner-strided layout must exactly tile the span"
                )
        elif self.layout_kind == LayoutKind.DMA_STRIDED:
            if self.engine not in (Engine.RDMA, Engine.WDMA):
                raise ValueError("DMA-strided layout requires RDMA or WDMA")
            if self.issue_mode != IssueMode.WRAPPER:
                raise ValueError("DMA-strided layout uses the public CRT wrapper")
            if self.element_format != DMA_FORMAT_FP16:
                raise ValueError("DMA-strided layout is qualified for FP16")
            if (
                self.layout_inner_bytes
                | self.layout_stride0_bytes
                | self.layout_stride1_bytes
                | self.layout_stride2_bytes
            ) & 1:
                raise ValueError(
                    "FP16 DMA inner span and strides must be 2-byte aligned"
                )
            iterations = (
                self.layout_iteration0,
                self.layout_iteration1,
                self.layout_iteration2,
            )
            if self.layout_inner_bytes <= 0 or any(
                iteration <= 0 for iteration in iterations
            ):
                raise ValueError(
                    "DMA-strided layout requires a positive inner span "
                    "and three positive iterations"
                )
            compact_bytes = self.layout_inner_bytes
            for iteration in iterations:
                compact_bytes *= iteration
            if compact_bytes != self.transfer_bytes:
                raise ValueError(
                    "DMA-strided byte_count must equal "
                    "inner_bytes*iteration0*iteration1*iteration2"
                )
            row_bytes = (
                (self.layout_iteration0 - 1)
                * self.layout_stride0_bytes
                + self.layout_inner_bytes
            )
            plane_bytes = (
                (self.layout_iteration1 - 1)
                * self.layout_stride1_bytes
                + row_bytes
            )
            envelope_bytes = (
                (self.layout_iteration2 - 1)
                * self.layout_stride2_bytes
                + plane_bytes
            )
            if (
                (
                    self.layout_iteration0 > 1
                    and self.layout_stride0_bytes
                    < self.layout_inner_bytes
                )
                or (
                    self.layout_iteration1 > 1
                    and self.layout_stride1_bytes < row_bytes
                )
                or (
                    self.layout_iteration2 > 1
                    and self.layout_stride2_bytes < plane_bytes
                )
            ):
                raise ValueError("DMA-strided chunks must not overlap")
            if envelope_bytes > MAX_DMA_ENVELOPE_BYTES:
                raise ValueError("DMA-strided envelope exceeds its DDR slot")
        else:
            raise ValueError("unknown layout kind")

    def dma_chunk_offsets(self) -> tuple[int, ...]:
        self.validate()
        if self.layout_kind != LayoutKind.DMA_STRIDED:
            return (0,)
        return tuple(
            outer * self.layout_stride2_bytes
            + middle * self.layout_stride1_bytes
            + inner * self.layout_stride0_bytes
            for outer in range(self.layout_iteration2)
            for middle in range(self.layout_iteration1)
            for inner in range(self.layout_iteration0)
        )

    def dma_envelope_bytes(self) -> int:
        return max(self.dma_chunk_offsets()) + (
            self.layout_inner_bytes
            if self.layout_kind == LayoutKind.DMA_STRIDED
            else self.transfer_bytes
        )


@dataclasses.dataclass(frozen=True)
class IssueIdentity:
    ordinal: int
    lane: int
    round: int
    slot: int
    tag: int


@dataclasses.dataclass(frozen=True)
class Plan:
    lanes: tuple[Lane, ...]
    rounds: int
    effect_relation: EffectRelation
    range_relation: RangeRelation
    schedule: Schedule
    wait_kind: WaitKind
    wait_worker_mask: int
    seed: int
    first_operand: Operand = Operand.AUTO
    second_operand: Operand = Operand.AUTO
    issue_limit: int = 0
    sample: int = 0
    command: Command = Command.EXECUTE
    flags: int = 0

    def is_serial_dma_roundtrip(self) -> bool:
        if len(self.lanes) != 2:
            return False
        first, second = self.lanes
        descriptor_fields = (
            "transfer_bytes",
            "element_format",
            "layout_kind",
            "layout_inner_bytes",
            "layout_stride0_bytes",
            "layout_stride1_bytes",
            "layout_stride2_bytes",
            "layout_iteration0",
            "layout_iteration1",
            "layout_iteration2",
        )
        return (
            self.rounds == 1
            and self.effect_relation == EffectRelation.RAW
            and self.range_relation == RangeRelation.EXACT
            and self.schedule == Schedule.SERIAL
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask == 1
            and self.first_operand == Operand.WRITE
            and self.second_operand == Operand.READ0
            and self.flags == 0
            and self.issue_limit == 0
            and first.engine == Engine.RDMA
            and second.engine == Engine.WDMA
            and first.worker == second.worker == 0
            and first.layout_kind == LayoutKind.DMA_STRIDED
            and second.layout_kind == LayoutKind.DMA_STRIDED
            and all(
                getattr(first, field) == getattr(second, field)
                for field in descriptor_fields
            )
        )

    def is_strided_dependency_observation(self) -> bool:
        if len(self.lanes) != 2:
            return False
        first, second = self.lanes
        descriptor_fields = (
            "transfer_bytes",
            "element_format",
            "layout_kind",
            "layout_inner_bytes",
            "layout_stride0_bytes",
            "layout_stride1_bytes",
            "layout_stride2_bytes",
            "layout_iteration0",
            "layout_iteration1",
            "layout_iteration2",
        )
        common = (
            self.rounds == 1
            and self.schedule in (Schedule.SERIAL, Schedule.WINDOW)
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask == 1
            and self.flags == 0
            and self.issue_limit == 0
            and first.worker == second.worker == 0
            and first.layout_kind == LayoutKind.DMA_STRIDED
            and second.layout_kind == LayoutKind.DMA_STRIDED
            and all(
                getattr(first, field) == getattr(second, field)
                for field in descriptor_fields
            )
        )
        if not common:
            return False
        shapes = {
            EffectRelation.RAW: (
                Engine.RDMA,
                Engine.WDMA,
                Operand.WRITE,
                Operand.READ0,
                (RangeRelation.STRIDED_ENVELOPE,),
            ),
            EffectRelation.WAR: (
                Engine.WDMA,
                Engine.RDMA,
                Operand.READ0,
                Operand.WRITE,
                (RangeRelation.STRIDED_ENVELOPE,),
            ),
            EffectRelation.WAW: (
                Engine.RDMA,
                Engine.RDMA,
                Operand.WRITE,
                Operand.WRITE,
                (
                    RangeRelation.EXACT,
                    RangeRelation.PARTIAL,
                    RangeRelation.ADJACENT,
                ),
            ),
            EffectRelation.RAR: (
                Engine.WDMA,
                Engine.WDMA,
                Operand.READ0,
                Operand.READ0,
                (RangeRelation.STRIDED_ENVELOPE,),
            ),
        }
        shape = shapes.get(self.effect_relation)
        if shape is None:
            return False
        (
            first_engine,
            second_engine,
            first_operand,
            second_operand,
            relations,
        ) = shape
        if (
            first.engine != first_engine
            or second.engine != second_engine
            or self.first_operand != first_operand
            or self.second_operand != second_operand
            or self.range_relation not in relations
        ):
            return False
        return (
            self.range_relation != RangeRelation.PARTIAL
            or len(first.dma_chunk_offsets()) > 1
        )

    def is_constructor_observation(self) -> bool:
        return (
            self.flags == CONSTRUCTOR_OBSERVATION
            and len(self.lanes) == 1
            and self.rounds == 1
            and self.lanes[0].engine == Engine.CT
            and self.lanes[0].worker == 0
            and self.lanes[0].issue_mode == IssueMode.RAW
            and self.effect_relation == EffectRelation.NONE
            and self.range_relation == RangeRelation.DISJOINT
            and self.schedule == Schedule.SERIAL
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask == 1
            and self.first_operand == Operand.AUTO
            and self.second_operand == Operand.AUTO
            and self.issue_limit == 0
        )

    def is_ordered_producer_consumer(self) -> bool:
        if len(self.lanes) != 2:
            return False
        first, second = self.lanes
        return (
            self.flags == ORDERED_PRODUCER_CONSUMER
            and self.rounds == 1
            and self.effect_relation == EffectRelation.RAW
            and self.range_relation == RangeRelation.EXACT
            and self.schedule == Schedule.SERIAL
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask
            == ((1 << first.worker) | (1 << second.worker))
            and self.first_operand == Operand.WRITE
            and self.second_operand == Operand.READ0
            and self.issue_limit == 0
            and (first.engine, second.engine)
            in (
                (Engine.RDMA, Engine.CT),
                (Engine.CT, Engine.WDMA),
                (Engine.NE, Engine.WDMA),
                (Engine.TDMA, Engine.CT),
                (Engine.TDMA, Engine.NE),
            )
        )

    def is_double_slot_observation(self) -> bool:
        if len(self.lanes) != 3:
            return False
        rdma, ct, wdma = self.lanes
        return (
            self.flags == DOUBLE_SLOT_OBSERVATION
            and 1 <= self.rounds <= MAX_ROUNDS
            and self.effect_relation == EffectRelation.NONE
            and self.range_relation == RangeRelation.DISJOINT
            and self.schedule in (Schedule.SERIAL, Schedule.WINDOW)
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask == 1
            and self.first_operand == Operand.AUTO
            and self.second_operand == Operand.AUTO
            and self.issue_limit == 0
            and tuple(lane.engine for lane in self.lanes)
            == (Engine.RDMA, Engine.CT, Engine.WDMA)
            and all(lane.worker == 0 for lane in self.lanes)
            and all(
                lane.transfer_bytes == rdma.transfer_bytes
                and lane.element_format == DMA_FORMAT_FP16
                and lane.layout_kind == LayoutKind.CONTIGUOUS
                for lane in (rdma, ct, wdma)
            )
        )

    def is_mapped_spm_kcore_write_observation(self) -> bool:
        if len(self.lanes) != 1:
            return False
        (consumer,) = self.lanes
        return (
            self.flags
            in (
                MAPPED_SPM_KCORE_WRITE,
                MAPPED_SPM_KCORE_WRITE | PREISSUE_LOCAL_WAIT,
            )
            and self.rounds == 1
            and self.effect_relation == EffectRelation.NONE
            and self.range_relation == RangeRelation.DISJOINT
            and self.schedule == Schedule.WINDOW
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask == 1
            and self.first_operand == Operand.AUTO
            and self.second_operand == Operand.AUTO
            and self.issue_limit == 0
            and consumer.engine == Engine.CT
            and consumer.worker == 0
            and consumer.issue_mode == IssueMode.RAW
            and consumer.element_format == DMA_FORMAT_FP16
            and consumer.layout_kind == LayoutKind.CONTIGUOUS
            and consumer.transfer_bytes % 2 == 0
        )

    def is_tight_kcore_boundary_observation(self) -> bool:
        if len(self.lanes) != 1:
            return False
        (producer,) = self.lanes
        queue_depths = {
            Engine.CT: 6,
            Engine.NE: 6,
            Engine.RDMA: 6,
            Engine.WDMA: 6,
            Engine.TDMA: 4,
        }
        return (
            self.flags == TIGHT_KCORE_BOUNDARY
            and self.effect_relation == EffectRelation.NONE
            and self.range_relation == RangeRelation.DISJOINT
            and self.schedule == Schedule.WINDOW
            and self.wait_kind in (WaitKind.NONE, WaitKind.LOCAL_FENCE)
            and self.wait_worker_mask == 0
            and self.first_operand == Operand.AUTO
            and self.second_operand == Operand.AUTO
            and producer.worker == 0
            and producer.issue_mode == IssueMode.RAW
            and self.issue_limit == 0
            and self.rounds <= queue_depths[producer.engine]
        )

    def is_tight_queue_saturation(self) -> bool:
        if len(self.lanes) != 2:
            return False
        first, second = self.lanes
        queue_depths = {
            Engine.CT: 6,
            Engine.NE: 6,
            Engine.RDMA: 6,
            Engine.WDMA: 6,
            Engine.TDMA: 4,
        }
        depth = queue_depths[first.engine]
        return (
            self.flags == TIGHT_QUEUE_SATURATION
            and first == second
            and first.worker == 0
            and first.issue_mode == IssueMode.RAW
            and self.effect_relation == EffectRelation.NONE
            and self.range_relation == RangeRelation.DISJOINT
            and self.schedule == Schedule.WINDOW
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask == 1
            and self.first_operand == Operand.AUTO
            and self.second_operand == Operand.AUTO
            and self.issue_limit in (depth - 1, depth, depth + 1)
            and self.issue_limit <= len(self.lanes) * self.rounds
            and self.issue_limit > (len(self.lanes) - 1) * self.rounds
        )

    def is_tight_worker_scope(self) -> bool:
        participants = {lane.worker for lane in self.lanes}
        return (
            self.flags == TIGHT_WORKER_SCOPE
            and len(self.lanes) == 3
            and len(participants) >= 2
            and all(
                lane.issue_mode == IssueMode.RAW for lane in self.lanes
            )
            and self.effect_relation == EffectRelation.NONE
            and self.range_relation == RangeRelation.DISJOINT
            and self.schedule == Schedule.WINDOW
            and self.wait_kind != WaitKind.NONE
            and self.first_operand == Operand.AUTO
            and self.second_operand == Operand.AUTO
            and self.issue_limit == 0
        )

    def is_bounded_pair_window(self) -> bool:
        if len(self.lanes) != 2:
            return False
        first, second = self.lanes
        return (
            self.flags == BOUNDED_PAIR_WINDOW
            and self.rounds == 8
            and first.engine != second.engine
            and self.effect_relation == EffectRelation.NONE
            and self.range_relation == RangeRelation.DISJOINT
            and self.schedule == Schedule.WINDOW
            and self.wait_kind == WaitKind.BY_WORKER
            and self.wait_worker_mask
            == ((1 << first.worker) | (1 << second.worker))
            and self.first_operand == Operand.AUTO
            and self.second_operand == Operand.AUTO
            and self.issue_limit == 0
        )

    def validate(self) -> None:
        if self.command == Command.QUALIFY:
            if (
                self.lanes
                or self.rounds != 0
                or self.effect_relation != EffectRelation.NONE
                or self.range_relation != RangeRelation.DISJOINT
                or self.schedule != Schedule.WINDOW
                or self.wait_kind != WaitKind.NONE
                or self.wait_worker_mask
                or self.first_operand != Operand.AUTO
                or self.second_operand != Operand.AUTO
                or self.issue_limit
                or self.flags
            ):
                raise ValueError("qualification must not carry an execution plan")
            return
        if not 1 <= len(self.lanes) <= MAX_LANES:
            raise ValueError(f"execute plans require 1..{MAX_LANES} lanes")
        if not 1 <= self.rounds <= MAX_ROUNDS:
            raise ValueError(f"rounds must be in [1, {MAX_ROUNDS}]")
        if (
            len(self.lanes) == 3
            and self.rounds > MAX_THREE_LANE_ROUNDS
        ):
            raise ValueError(
                "three-lane plans exceed the qualified four-round slot arena"
            )
        for lane in self.lanes:
            lane.validate()
        if any(
            lane.layout_kind == LayoutKind.DMA_STRIDED
            for lane in self.lanes
        ) and not (
            self.is_serial_dma_roundtrip()
            or self.is_strided_dependency_observation()
        ):
            raise ValueError(
                "DMA-strided lanes require a bounded worker0 DMA "
                "roundtrip or dependency observation"
            )
        if len(self.lanes) == 1 and (
            self.effect_relation != EffectRelation.NONE
            or self.range_relation != RangeRelation.DISJOINT
        ):
            raise ValueError("single-lane plans do not have a cross-lane relation")
        if len(self.lanes) == 3 and (
            self.effect_relation != EffectRelation.NONE
            or self.range_relation != RangeRelation.DISJOINT
        ):
            raise ValueError("three-lane plans are disjoint windows, not hazards")
        if self.effect_relation == EffectRelation.NONE:
            if (
                self.first_operand != Operand.AUTO
                or self.second_operand != Operand.AUTO
            ):
                raise ValueError(
                    "non-hazard plans cannot select hazard operands"
                )
        else:
            if len(self.lanes) != 2:
                raise ValueError("hazard plans require exactly two lanes")
            selected = (self.first_operand, self.second_operand)
            if any(
                operand not in (Operand.READ0, Operand.READ1, Operand.WRITE)
                for operand in selected
            ):
                raise ValueError(
                    "hazard plans require explicit operand selections"
                )
            reads = (Operand.READ0, Operand.READ1)
            expected = {
                EffectRelation.RAW: (Operand.WRITE, reads),
                EffectRelation.WAR: (reads, Operand.WRITE),
                EffectRelation.WAW: (Operand.WRITE, Operand.WRITE),
                EffectRelation.RAR: (reads, reads),
            }[self.effect_relation]
            if (
                (
                    self.first_operand not in expected[0]
                    if isinstance(expected[0], tuple)
                    else self.first_operand != expected[0]
                )
                or (
                    self.second_operand not in expected[1]
                    if isinstance(expected[1], tuple)
                    else self.second_operand != expected[1]
                )
            ):
                raise ValueError(
                    "selected operands do not implement the effect relation"
                )
            engine_operands = {
                Engine.CT: (Operand.READ0, Operand.READ1, Operand.WRITE),
                Engine.NE: (Operand.READ0, Operand.READ1, Operand.WRITE),
                Engine.RDMA: (Operand.READ0, Operand.WRITE),
                Engine.WDMA: (Operand.READ0, Operand.WRITE),
                Engine.TDMA: (Operand.WRITE,),
            }
            if (
                self.first_operand not in engine_operands[self.lanes[0].engine]
                or self.second_operand
                not in engine_operands[self.lanes[1].engine]
            ):
                raise ValueError(
                    "selected operand is not exposed by its engine"
                )
        if (
            self.range_relation == RangeRelation.STRIDED_ENVELOPE
            and not any(
                lane.layout_kind == LayoutKind.INNER_STRIDED
                or lane.layout_kind == LayoutKind.DMA_STRIDED
                for lane in self.lanes
            )
        ):
            raise ValueError("strided-envelope relation requires a strided lane")
        participants = 0
        for lane in self.lanes:
            participants |= 1 << lane.worker
        if self.wait_kind == WaitKind.BY_WORKER:
            if not self.wait_worker_mask:
                raise ValueError("worker wait requires an explicit nonzero mask")
            if self.wait_worker_mask & ~participants:
                raise ValueError("wait mask contains a non-participant worker")
        elif self.wait_worker_mask:
            raise ValueError("only BY_WORKER consumes a worker mask")
        if self.flags not in (
            0,
            TIGHT_DEPTH_PLUS_ONE,
            CONSTRUCTOR_OBSERVATION,
            ORDERED_PRODUCER_CONSUMER,
            DOUBLE_SLOT_OBSERVATION,
            MAPPED_SPM_KCORE_WRITE,
            MAPPED_SPM_KCORE_WRITE | PREISSUE_LOCAL_WAIT,
            TIGHT_KCORE_BOUNDARY,
            TIGHT_QUEUE_SATURATION,
            TIGHT_WORKER_SCOPE,
            BOUNDED_PAIR_WINDOW,
        ):
            raise ValueError("unknown plan flags are not safe")
        queue_depths = {
            Engine.CT: 6,
            Engine.NE: 6,
            Engine.RDMA: 6,
            Engine.WDMA: 6,
            Engine.TDMA: 4,
        }
        tight_depth_plus_one = self.flags == TIGHT_DEPTH_PLUS_ONE
        tight_kcore_boundary = self.is_tight_kcore_boundary_observation()
        tight_queue_saturation = self.is_tight_queue_saturation()
        tight_worker_scope = self.is_tight_worker_scope()
        bounded_pair_window = self.is_bounded_pair_window()
        special_queue_bound = (
            tight_depth_plus_one
            or tight_kcore_boundary
            or tight_queue_saturation
            or bounded_pair_window
        )
        if tight_depth_plus_one:
            if (
                len(self.lanes) != 2
                or self.lanes[0] != self.lanes[1]
                or self.lanes[0].worker != 0
                or self.lanes[0].issue_mode != IssueMode.RAW
                or self.effect_relation != EffectRelation.NONE
                or self.range_relation != RangeRelation.DISJOINT
                or self.schedule != Schedule.WINDOW
                or self.wait_kind != WaitKind.BY_WORKER
                or self.wait_worker_mask != 1
                or self.issue_limit
                != queue_depths[self.lanes[0].engine] + 1
                or self.issue_limit > len(self.lanes) * self.rounds
                or self.issue_limit
                <= (len(self.lanes) - 1) * self.rounds
            ):
                raise ValueError(
                    "tight depth-plus-one must issue exactly depth+1 raw "
                    "entries on one worker/engine with a matching wait"
                )
        elif self.flags == TIGHT_KCORE_BOUNDARY:
            if not tight_kcore_boundary:
                raise ValueError(
                    "tight Kcore boundary must issue a bounded raw window on "
                    "one worker/engine with no wait or a local fence"
                )
        elif self.flags == TIGHT_QUEUE_SATURATION:
            if not tight_queue_saturation:
                raise ValueError(
                    "tight queue saturation must issue exactly depth-1, "
                    "depth, or depth+1 identical raw entries with a matching "
                    "worker wait"
                )
        elif self.flags == TIGHT_WORKER_SCOPE:
            if not tight_worker_scope:
                raise ValueError(
                    "tight worker scope requires a three-lane raw window "
                    "across at least two workers and a requested wait"
                )
        elif self.flags == BOUNDED_PAIR_WINDOW:
            if not bounded_pair_window:
                raise ValueError(
                    "bounded pair window requires two different engines, "
                    "eight rounds, and a matching participant wait"
                )
        elif self.issue_limit:
            raise ValueError(
                "only a typed tight submission consumes issue_limit"
            )
        if (
            self.flags == CONSTRUCTOR_OBSERVATION
            and not self.is_constructor_observation()
        ):
            raise ValueError("constructor observation requires one raw CT issue")
        if (
            self.flags == ORDERED_PRODUCER_CONSUMER
            and not self.is_ordered_producer_consumer()
        ):
            raise ValueError(
                "ordered producer-consumer observation is malformed"
            )
        if (
            self.flags == DOUBLE_SLOT_OBSERVATION
            and not self.is_double_slot_observation()
        ):
            raise ValueError("double-slot observation is malformed")
        if (
            self.flags & MAPPED_SPM_KCORE_WRITE
            and not self.is_mapped_spm_kcore_write_observation()
        ):
            raise ValueError(
                "mapped-SPM Kcore-write observation is malformed"
            )
        if self.schedule == Schedule.WINDOW and not special_queue_bound:
            outstanding: dict[tuple[Engine, int], int] = {}
            for lane in self.lanes:
                key = (lane.engine, lane.worker)
                outstanding[key] = outstanding.get(key, 0) + self.rounds
                if outstanding[key] > queue_depths[lane.engine]:
                    raise ValueError(
                        "window exceeds the documented queue depth"
                    )

    def issue_identities(self) -> tuple[IssueIdentity, ...]:
        self.validate()
        identities = []
        slot_stride = (
            MAX_ROUNDS
            if self.rounds > MAX_THREE_LANE_ROUNDS
            else MAX_THREE_LANE_ROUNDS
        )
        for lane_index in range(len(self.lanes)):
            for round_index in range(self.rounds):
                ordinal = lane_index * self.rounds + round_index
                slot = lane_index * slot_stride + round_index
                tag = (
                    (self.seed * 0x9E3779B97F4A7C15)
                    & 0xFFFFFFFFFFFFFF00
                ) | (slot + 1)
                identities.append(
                    IssueIdentity(
                        ordinal, lane_index, round_index, slot, tag
                    )
                )
        if self.issue_limit:
            identities = identities[: self.issue_limit]
        return tuple(identities)

    def issue_order(self) -> tuple[int, ...]:
        identities = self.issue_identities()
        if self.is_double_slot_observation():
            by_pair = {
                (identity.lane, identity.round): identity.slot
                for identity in identities
            }
            ordered = []
            for step in range(self.rounds + 2):
                for lane, delay in ((0, 0), (1, 1), (2, 2)):
                    round_index = step - delay
                    if 0 <= round_index < self.rounds:
                        ordered.append(by_pair[(lane, round_index)])
            return tuple(ordered)
        by_ordinal = {identity.ordinal: identity for identity in identities}
        return tuple(
            by_ordinal[ordinal].slot
            for round_index in range(self.rounds)
            for lane_index in range(len(self.lanes))
            if (
                ordinal := lane_index * self.rounds + round_index
            )
            in by_ordinal
        )

    def request_words(self) -> tuple[int, ...]:
        self.validate()
        words = [0] * REQUEST_WORDS
        words[REQ["MAGIC"]] = REQUEST_MAGIC
        words[REQ["SCHEMA_AND_WORDS"]] = (SCHEMA << 32) | REQUEST_WORDS
        words[REQ["COMMAND"]] = self.command
        words[REQ["LANE_COUNT"]] = len(self.lanes)
        words[REQ["ROUNDS"]] = self.rounds
        words[REQ["EFFECT_RELATION"]] = self.effect_relation
        words[REQ["RANGE_RELATION"]] = self.range_relation
        words[REQ["SCHEDULE"]] = self.schedule
        words[REQ["WAIT_KIND"]] = self.wait_kind
        words[REQ["WAIT_WORKER_MASK"]] = self.wait_worker_mask
        words[REQ["SEED"]] = self.seed
        words[REQ["SAMPLE"]] = self.sample
        words[REQ["FLAGS"]] = self.flags
        words[REQ["FIRST_OPERAND"]] = self.first_operand
        words[REQ["SECOND_OPERAND"]] = self.second_operand
        words[REQ["ISSUE_LIMIT"]] = self.issue_limit
        for lane_index, lane in enumerate(self.lanes):
            base = LANE_BASE + lane_index * LANE_STRIDE
            words[base + LANE["ENGINE"]] = lane.engine
            words[base + LANE["WORKER"]] = lane.worker
            words[base + LANE["ISSUE_MODE"]] = lane.issue_mode
            words[base + LANE["TRANSFER_BYTES"]] = lane.transfer_bytes
            words[base + LANE["ELEMENT_FORMAT"]] = lane.element_format
            words[base + LANE["LAYOUT_KIND"]] = lane.layout_kind
            words[base + LANE["LAYOUT_INNER_BYTES"]] = (
                lane.layout_inner_bytes
            )
            words[base + LANE["LAYOUT_STRIDE0_BYTES"]] = (
                lane.layout_stride0_bytes
            )
            words[base + LANE["LAYOUT_STRIDE1_BYTES"]] = (
                lane.layout_stride1_bytes
            )
            words[base + LANE["LAYOUT_STRIDE2_BYTES"]] = (
                lane.layout_stride2_bytes
            )
            words[base + LANE["LAYOUT_ITERATION0"]] = (
                lane.layout_iteration0
            )
            words[base + LANE["LAYOUT_ITERATION1"]] = (
                lane.layout_iteration1
            )
            words[base + LANE["LAYOUT_ITERATION2"]] = (
                lane.layout_iteration2
            )
            words[base + LANE["FLAGS"]] = lane.flags
        return tuple(int(word) for word in words)


@dataclasses.dataclass(frozen=True)
class InclusiveRange:
    begin: int
    end: int


@dataclasses.dataclass(frozen=True)
class IssueObservation:
    identity: IssueIdentity
    engine: Engine
    worker: int
    execute_rc: int
    inter_type: int
    read0: InclusiveRange | None
    read1: InclusiveRange | None
    write: InclusiveRange | None
    boundary_mismatches: int
    boundary_guard_mismatches: int
    final_mismatches: int
    final_guard_mismatches: int
    flags: int
    control_after_issue: int
    execute_cycles: int


def _optional_range(
    words: Sequence[int],
    base: int,
    valid_flag: int,
    begin_field: str,
    end_field: str,
    flags: int,
) -> InclusiveRange | None:
    begin = words[base + ISSUE[begin_field]]
    end = words[base + ISSUE[end_field]]
    if flags & valid_flag:
        if end < begin:
            raise ValueError(f"{begin_field[:-6]} inclusive range is inverted")
        return InclusiveRange(begin, end)
    if begin != 0 or end != 0:
        raise ValueError(f"{begin_field[:-6]} range lacks its validity flag")
    return None


def describe_prepare_failure(words: Sequence[int], plan: Plan) -> str:
    """Describe the one preparation callback that did not complete."""
    plan.validate()
    failed: list[tuple[IssueIdentity, int]] = []
    for identity in plan.issue_identities():
        base = ISSUE_BASE + identity.ordinal * ISSUE_STRIDE
        flags = words[base + ISSUE["FLAGS"]] & PREPARE_STAGE_MASK
        if (
            flags & PREPARE_STAGE_FLAGS["ENTERED"]
            and not flags & PREPARE_STAGE_FLAGS["COMPLETED"]
        ):
            failed.append((identity, flags))
    if len(failed) != 1:
        return f"prepare progress identifies {len(failed)} incomplete issues"

    identity, flags = failed[0]
    lane = plan.lanes[identity.lane]
    stages = [
        name.lower().replace("_", "-")
        for name, flag in PREPARE_STAGE_FLAGS.items()
        if flags & flag
    ]
    if (
        lane.issue_mode == IssueMode.RAW
        and not flags & PREPARE_STAGE_FLAGS["BUILDER_ACQUIRED"]
    ):
        outcome = "builder-not-acquired"
    elif (
        lane.issue_mode == IssueMode.RAW
        and not flags & PREPARE_STAGE_FLAGS["PACKET_MATERIALIZED"]
    ):
        outcome = "packet-not-materialized"
    elif (
        lane.issue_mode == IssueMode.RAW
        and not flags & PREPARE_STAGE_FLAGS["BUILDER_RELEASED"]
    ):
        outcome = "builder-not-released"
    else:
        outcome = "adapter-rejected"
    return (
        f"prepare issue {identity.ordinal} slot={identity.slot} "
        f"engine={lane.engine.name} worker={lane.worker} "
        f"stages={','.join(stages)} outcome={outcome}"
    )


def validate_record(
    words: Sequence[int], plan: Plan
) -> tuple[IssueObservation, ...]:
    if len(words) < RECORD_WORDS:
        raise ValueError("record is truncated")
    if words[REC["MAGIC"]] != RECORD_MAGIC:
        raise ValueError("record magic is invalid")
    if words[REC["SCHEMA_AND_WORDS"]] != (SCHEMA << 32) | RECORD_WORDS:
        raise ValueError("record schema/length is invalid")
    status = Status(words[REC["STATUS"]])
    if status != Status.OK:
        detail = (
            f": {describe_prepare_failure(words, plan)}"
            if status == Status.PREPARE_FAILED
            else ""
        )
        raise ValueError(f"probe status is {status.name}{detail}")
    plan.validate()
    identities = plan.issue_identities()
    if (
        words[REC["COMMAND"]] != plan.command
        or words[REC["LANE_COUNT"]] != len(plan.lanes)
        or words[REC["ROUNDS"]] != plan.rounds
        or words[REC["ISSUE_COUNT"]] != len(identities)
        or words[REC["EFFECT_RELATION"]] != plan.effect_relation
        or words[REC["RANGE_RELATION"]] != plan.range_relation
        or words[REC["SCHEDULE"]] != plan.schedule
        or words[REC["WAIT_KIND"]] != plan.wait_kind
        or words[REC["WAIT_WORKER_MASK"]] != plan.wait_worker_mask
        or words[REC["FIRST_OPERAND"]] != plan.first_operand
        or words[REC["SECOND_OPERAND"]] != plan.second_operand
        or words[REC["ISSUE_LIMIT"]] != plan.issue_limit
        or words[REC["SEED"]] != plan.seed
        or words[REC["SAMPLE"]] != plan.sample
    ):
        raise ValueError("record does not echo the requested typed plan")
    if words[REC["FLAGS"]] & ALL_PHASE_FLAGS != ALL_PHASE_FLAGS:
        raise ValueError("record did not complete every observation phase")
    needs_pre_wait = plan.flags in (
        TIGHT_QUEUE_SATURATION,
        TIGHT_WORKER_SCOPE,
    )
    has_pre_wait = bool(words[REC["FLAGS"]] & PRE_WAIT_CAPTURED)
    if has_pre_wait != needs_pre_wait:
        raise ValueError("record pre-wait capture does not match the plan")
    bounded_drain_count = words[REC["BOUNDED_WINDOW_DRAIN_COUNT"]]
    bounded_drain_cycles = words[REC["BOUNDED_WINDOW_DRAIN_CYCLES"]]
    if plan.is_bounded_pair_window():
        if bounded_drain_count != 1 or bounded_drain_cycles == 0:
            raise ValueError(
                "bounded pair window omitted its one intermediate drain"
            )
    elif bounded_drain_count != 0 or bounded_drain_cycles != 0:
        raise ValueError(
            "ordinary plan unexpectedly recorded a bounded-window drain"
        )

    observations = []
    seen_slots: set[int] = set()
    seen_tags: set[int] = set()
    for expected in identities:
        base = ISSUE_BASE + expected.ordinal * ISSUE_STRIDE
        if (
            words[base + ISSUE["ORDINAL"]] != expected.ordinal
            or words[base + ISSUE["LANE"]] != expected.lane
            or words[base + ISSUE["ROUND"]] != expected.round
            or words[base + ISSUE["SLOT"]] != expected.slot
            or words[base + ISSUE["TAG"]] != expected.tag
        ):
            raise ValueError("per-issue identity does not match the plan")
        if expected.slot in seen_slots or expected.tag in seen_tags:
            raise ValueError("per-issue slot/tag is not unique")
        seen_slots.add(expected.slot)
        seen_tags.add(expected.tag)
        lane = plan.lanes[expected.lane]
        if (
            words[base + ISSUE["ENGINE"]] != lane.engine
            or words[base + ISSUE["WORKER"]] != lane.worker
        ):
            raise ValueError("per-issue engine/worker routing is inconsistent")
        flags = words[base + ISSUE["FLAGS"]]
        observations.append(
            IssueObservation(
                identity=expected,
                engine=lane.engine,
                worker=lane.worker,
                execute_rc=words[base + ISSUE["EXECUTE_RC"]],
                inter_type=words[base + ISSUE["INTER_TYPE"]],
                read0=_optional_range(
                    words,
                    base,
                    READ0_VALID,
                    "READ0_BEGIN",
                    "READ0_END",
                    flags,
                ),
                read1=_optional_range(
                    words,
                    base,
                    READ1_VALID,
                    "READ1_BEGIN",
                    "READ1_END",
                    flags,
                ),
                write=_optional_range(
                    words,
                    base,
                    WRITE_VALID,
                    "WRITE_BEGIN",
                    "WRITE_END",
                    flags,
                ),
                boundary_mismatches=words[
                    base + ISSUE["BOUNDARY_MISMATCHES"]
                ],
                boundary_guard_mismatches=words[
                    base + ISSUE["BOUNDARY_GUARD_MISMATCHES"]
                ],
                final_mismatches=words[base + ISSUE["FINAL_MISMATCHES"]],
                final_guard_mismatches=words[
                    base + ISSUE["FINAL_GUARD_MISMATCHES"]
                ],
                flags=flags,
                control_after_issue=words[
                    base + ISSUE["CONTROL_AFTER_ISSUE"]
                ],
                execute_cycles=words[
                    base + ISSUE["EXECUTE_CYCLES"]
                ],
            )
        )
    return tuple(observations)
