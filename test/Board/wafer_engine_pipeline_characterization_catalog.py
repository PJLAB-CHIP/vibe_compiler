#!/usr/bin/env python3
"""Typed board matrix for NCC throughput, balance, and software pipelines.

Pipeline position:
- Upstream IR / input:
  For raw engine characterization, a typed bounded NCC request whose packet
  adapters already have exact result/range/count support.  For the three-stage
  family, the accepted production Instr program after scheduling and before
  package writing.
- Current stage responsibility:
  Select matched serial/window experiments, retain exact correctness and
  measurement activation contracts, and reject cells that the current bounded
  probe or production compiler cannot honestly produce.
- Output IR / files:
  Raw board observations plus a narrow qualification-scoped activation decision.
  This catalog never writes latency constants or a shadow schedule into IR.
- Downstream consumer:
  Future scheduling/cost calibration and the production multi-buffer
  qualification gate.
- User-level driver / named pipeline:
  ``wafer_board_engine_pipeline_characterization_test.py`` reuses the normal
  wafer-compile/package/wafer-run lifecycle and the existing typed NCC device
  dispatcher.
- Explicit non-goals:
  Re-running already-qualified overlap smoke, treating host elapsed time as a
  device signal, or allowing the handwritten double-slot probe to stand in for
  a production compiler pipeline.
- Completion gate:
  Every safe raw cell serializes, exact result/guards/instruction counts pass,
  every performance cell has at least three sample-major counterbalanced board
  launches with exact manifest-matched terminal completion, and its complete
  matched activation group has a stable device-PMU direction.  The three-stage
  preparation gate accepts only the compiler-owned, manifest-bound fixed-slot
  accepted-Instr qualification sibling.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import pathlib
import statistics
import zlib
from collections import defaultdict
from collections.abc import Iterable, Mapping

import wafer_ncc_probe_protocol as ncc_protocol
import wafer_memory_descriptor_calibration_catalog as memory_catalog


FMT_FP16 = ncc_protocol.DMA_FORMAT_FP16
FP16_ELEMENT_BYTES = 2
MIN_REPEATS = 3
QUALIFICATION_SPM_ALIGNMENT = 256
CALIBRATION_SESSION_ENVIRONMENT = "WAFER_CALIBRATION_SESSION_ID"
RUNTIME_LIFECYCLE = (
    "validation",
    "device-selection",
    "resource-allocation",
    "host-to-device",
    "module-load",
    "entry-resolve",
    "launch",
    "completion",
    "device-to-host",
    "cleanup",
)
NE_TAIL_CHARACTERIZATION_KEY = "single/ne/tail/f16-m65-k129-n129"
NE_WORK_MACS_BY_LABEL = {
    "small": 1 * 16 * 16,
    "steady-16k": 64 * 128 * 128,
}
NE_TAIL_WORK_MACS = 65 * 129 * 129
RAW_PROBE_ASSET = "test/Board/wafer_board_ncc_execution_probe_test.py"
MEMORY_PAIR_ASSET = (
    "test/Board/wafer_memory_descriptor_calibration_catalog.py"
)
NE_CALIBRATION_ASSET = "test/Board/wafer_ne_calibration_catalog.py"
NE_TAIL_THROUGHPUT_ASSET = (
    "test/Board/wafer_ne_tail_throughput_catalog.py"
)
MEMORY_PAIR_SYMBOL = "SUSTAINED_PARALLEL_PAIR_CASES"
MEMORY_CROSS_WORKER_SYMBOL = "CROSS_WORKER_PARALLEL_PAIR_CASES"
HANDWRITTEN_PIPELINE_SYMBOL = "V2_DOUBLE_SLOT_OBSERVATION_CASES"


class Family(str, enum.Enum):
    SINGLE_ENGINE_THROUGHPUT = "single-engine-throughput"
    ENGINE_PAIR_STAGE_BALANCE = "engine-pair-stage-balance"
    PRODUCTION_THREE_STAGE = "production-three-stage"


class Disposition(str, enum.Enum):
    BOARD_EXECUTABLE = "board-executable"
    BOARD_EXECUTABLE_EXTERNAL = "board-executable-external"
    DELEGATED_BOARD_EVIDENCE = "delegated-board-evidence"
    PENDING_CONFIGURED_BOARD = "pending-configured-board"
    HOST_EXACT_CLOSED = "host-exact-closed"
    FAIL_CLOSED_MISSING_ADAPTER = "fail-closed-missing-adapter"
    FAIL_CLOSED_UNSAFE_WINDOW = "fail-closed-unsafe-window"


class Phase(str, enum.Enum):
    CALIBRATION = "calibration"
    HELD_OUT = "held-out"


class Engine(str, enum.Enum):
    CT = "ct"
    NE = "ne"
    RDMA = "rdma"
    WDMA = "wdma"
    TDMA = "tdma"


class Schedule(str, enum.Enum):
    SERIAL = "serial"
    WINDOW = "window"
    ONE_SLOT = "one-slot"
    TWO_SLOT = "two-slot"


class Placement(str, enum.Enum):
    SAME_WORKER = "same-worker"
    CROSS_WORKER = "cross-worker"


class Balance(str, enum.Enum):
    A_HEAVY = "a-heavy"
    EQUAL_BYTES = "equal-bytes"
    B_HEAVY = "b-heavy"


class PipelineRegime(str, enum.Enum):
    MOVEMENT_HEAVY = "movement-heavy"
    EQUAL_BYTES = "equal-bytes"
    COMPUTE_HEAVY = "compute-heavy"


class PipelineCapacity(str, enum.Enum):
    FITS_TWO_SLOTS = "fits-two-slots"
    FALLBACK_REQUIRED = "fallback-required"


@dataclasses.dataclass(frozen=True)
class PipelineContract:
    upstream_input: str
    current_stage_responsibility: str
    output_files: str
    downstream_consumer: str
    user_level_driver: str
    explicit_non_goals: tuple[str, ...]
    completion_gate: tuple[str, ...]


PIPELINE_CONTRACT = PipelineContract(
    upstream_input=(
        "typed bounded NCC request, or accepted production Instr program"
    ),
    current_stage_responsibility=(
        "matched measurement selection and fail-closed activation"
    ),
    output_files=(
        "raw board observations plus qualification-scoped activation decision"
    ),
    downstream_consumer="future scheduling and cost calibration",
    user_level_driver=(
        "wafer_board_engine_pipeline_characterization_test.py"
    ),
    explicit_non_goals=(
        "host-elapsed performance evidence",
        "shadow scheduling plan",
        "handwritten raw pipeline as production proof",
    ),
    completion_gate=(
        "exact result, physical guards, and instruction count",
        "at least three sample-major counterbalanced board samples",
        "exact manifest-matched terminal completion and ordered lifecycle",
        "runner session and exact board-qualification provenance",
        "complete matched device-PMU activation group",
        "authenticated accepted-Instr structure for production pipeline",
    ),
)


ENGINE_ORDER = (
    Engine.CT,
    Engine.NE,
    Engine.RDMA,
    Engine.WDMA,
    Engine.TDMA,
)
ENGINE_TO_PROTOCOL = {
    Engine.CT: ncc_protocol.Engine.CT,
    Engine.NE: ncc_protocol.Engine.NE,
    Engine.RDMA: ncc_protocol.Engine.RDMA,
    Engine.WDMA: ncc_protocol.Engine.WDMA,
    Engine.TDMA: ncc_protocol.Engine.TDMA,
}
ENGINE_PAIRS = tuple(
    (left, right)
    for left_index, left in enumerate(ENGINE_ORDER)
    for right in ENGINE_ORDER[left_index + 1 :]
)


@dataclasses.dataclass(frozen=True)
class CorrectnessOracle:
    full_exact_result: bool = True
    physical_prefix_suffix_guards: bool = True
    exact_instruction_count: bool = True
    matching_completion: bool = True
    terminal_status_and_cleanup: bool = True


@dataclasses.dataclass(frozen=True)
class MeasurementOracle:
    signal: str
    minimum_repeats: int = MIN_REPEATS
    requires_device_pmu: bool = True
    requires_serial_window_pair: bool = False
    requires_reciprocal_issue_order: bool = False
    requires_held_out_direction: bool = False
    host_elapsed_is_evidence: bool = False


@dataclasses.dataclass(frozen=True)
class EvidenceReference:
    asset: str
    symbol: str
    case_names: tuple[str, ...]
    scope: str


@dataclasses.dataclass(frozen=True)
class ProbeCase:
    """One concrete invocation compatible with the existing NCC dispatcher."""

    key: str
    cell_key: str
    family: Family
    phase: Phase
    plan: ncc_protocol.Plan
    repeat: int
    measurement_engines: tuple[Engine, ...]
    correctness: CorrectnessOracle
    measurement: MeasurementOracle

    @property
    def name(self) -> str:
        return self.key.replace("/", "-")

    def plan_for_sample(self, sample: int) -> ncc_protocol.Plan:
        if sample < 0:
            raise ValueError(f"{self.key}: sample must be nonnegative")
        return dataclasses.replace(self.plan, sample=sample)

    def request_words(self, sample: int) -> tuple[int, ...]:
        return self.plan_for_sample(sample).request_words()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "characterization_key": self.key,
            "characterization_cell": self.cell_key,
            "family": self.family.value,
            "phase": self.phase.value,
            "engines": [
                lane.engine.name.lower() for lane in self.plan.lanes
            ],
            "workers": [lane.worker for lane in self.plan.lanes],
            "rounds": self.plan.rounds,
            "effect": self.plan.effect_relation.name.lower(),
            "range": self.plan.range_relation.name.lower(),
            "first_operand": self.plan.first_operand.name.lower(),
            "second_operand": self.plan.second_operand.name.lower(),
            "schedule": self.plan.schedule.name.lower(),
            "wait": self.plan.wait_kind.name.lower(),
            "wait_worker_mask": self.plan.wait_worker_mask,
            "flags": self.plan.flags,
            "issue_limit": self.plan.issue_limit,
            "issue_modes": [
                lane.issue_mode.name.lower() for lane in self.plan.lanes
            ],
            "transfer_bytes": [
                lane.transfer_bytes for lane in self.plan.lanes
            ],
            "element_counts": [
                lane.transfer_bytes // FP16_ELEMENT_BYTES
                for lane in self.plan.lanes
            ],
            "formats": [lane.element_format for lane in self.plan.lanes],
            "layouts": [
                {
                    "kind": lane.layout_kind.name.lower(),
                    "inner_bytes": lane.layout_inner_bytes,
                    "strides": [
                        lane.layout_stride0_bytes,
                        lane.layout_stride1_bytes,
                        lane.layout_stride2_bytes,
                    ],
                    "iterations": [
                        lane.layout_iteration0,
                        lane.layout_iteration1,
                        lane.layout_iteration2,
                    ],
                    "envelope_bytes": lane.dma_envelope_bytes(),
                }
                for lane in self.plan.lanes
            ],
        }


@dataclasses.dataclass(frozen=True)
class MatrixCell:
    key: str
    family: Family
    phase: Phase
    disposition: Disposition
    dimensions: tuple[tuple[str, str | int], ...]
    probes: tuple[ProbeCase, ...]
    correctness: CorrectnessOracle
    measurement: MeasurementOracle
    reason: str
    delegated: tuple[EvidenceReference, ...] = ()
    safe_alternative_keys: tuple[str, ...] = ()

    def dimension(self, name: str) -> str | int:
        values = dict(self.dimensions)
        if name not in values:
            raise KeyError(f"{self.key}: no dimension {name!r}")
        return values[name]

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "family": self.family.value,
            "phase": self.phase.value,
            "disposition": self.disposition.value,
            "dimensions": dict(self.dimensions),
            "probes": [probe.key for probe in self.probes],
            "correctness": dataclasses.asdict(self.correctness),
            "measurement": dataclasses.asdict(self.measurement),
            "reason": self.reason,
            "delegated": [
                dataclasses.asdict(reference) for reference in self.delegated
            ],
            "safe_alternative_keys": list(self.safe_alternative_keys),
        }


@dataclasses.dataclass(frozen=True)
class ActivationDecision:
    family: Family
    scope: str
    state: str
    reasons: tuple[str, ...]
    medians: tuple[tuple[str, float], ...] = ()

    @property
    def activated(self) -> bool:
        return self.state == "activated"

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "scope": self.scope,
            "state": self.state,
            "reasons": list(self.reasons),
            "medians": dict(self.medians),
        }


@dataclasses.dataclass(frozen=True)
class BoardActivationGroup:
    key: str
    cell_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "cell_keys": list(self.cell_keys),
        }


@dataclasses.dataclass(frozen=True)
class ProductionPreparationDecision:
    ready: bool
    reasons: tuple[str, ...]
    checked_package: str | None
    rejected_delegated_asset: str

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


CORRECTNESS = CorrectnessOracle()
SINGLE_MEASUREMENT = MeasurementOracle(
    signal="per-engine-execution-cycles-versus-work",
    requires_held_out_direction=True,
)
PAIR_MEASUREMENT = MeasurementOracle(
    signal="per-engine-execution-and-global-union-stage-imbalance",
    requires_serial_window_pair=True,
    requires_reciprocal_issue_order=True,
    requires_held_out_direction=True,
)
PIPELINE_MEASUREMENT = MeasurementOracle(
    signal="production-prologue-steady-epilogue-stage-overlap",
    requires_serial_window_pair=True,
    requires_held_out_direction=True,
)


def _seed(key: str) -> int:
    return 0x80000000 | zlib.crc32(key.encode("utf-8"))


def _lane(
    engine: Engine,
    transfer_bytes: int,
    *,
    worker: int = 0,
    layout_kind: ncc_protocol.LayoutKind = (
        ncc_protocol.LayoutKind.CONTIGUOUS
    ),
    layout_inner_bytes: int = 0,
    layout_stride0_bytes: int = 0,
    layout_iteration0: int = 0,
) -> ncc_protocol.Lane:
    if transfer_bytes % FP16_ELEMENT_BYTES != 0:
        raise ValueError(
            "generic engine characterization requires a whole FP16 element span"
        )
    return ncc_protocol.Lane(
        engine=ENGINE_TO_PROTOCOL[engine],
        worker=worker,
        issue_mode=(
            ncc_protocol.IssueMode.WRAPPER
            if layout_kind == ncc_protocol.LayoutKind.DMA_STRIDED
            else ncc_protocol.IssueMode.RAW
        ),
        transfer_bytes=transfer_bytes,
        element_format=FMT_FP16,
        layout_kind=layout_kind,
        layout_inner_bytes=layout_inner_bytes,
        layout_stride0_bytes=layout_stride0_bytes,
        layout_stride1_bytes=0,
        layout_stride2_bytes=0,
        layout_iteration0=layout_iteration0,
        layout_iteration1=(1 if layout_iteration0 else 0),
        layout_iteration2=(1 if layout_iteration0 else 0),
    )


def _plan(
    key: str,
    lanes: tuple[ncc_protocol.Lane, ...],
    rounds: int,
    schedule: Schedule,
    *,
    effect: ncc_protocol.EffectRelation = ncc_protocol.EffectRelation.NONE,
    relation: ncc_protocol.RangeRelation = (
        ncc_protocol.RangeRelation.DISJOINT
    ),
    first_operand: ncc_protocol.Operand = ncc_protocol.Operand.AUTO,
    second_operand: ncc_protocol.Operand = ncc_protocol.Operand.AUTO,
    flags: int = 0,
) -> ncc_protocol.Plan:
    participants = 0
    for lane in lanes:
        participants |= 1 << lane.worker
    raw_schedule = {
        Schedule.SERIAL: ncc_protocol.Schedule.SERIAL,
        Schedule.WINDOW: ncc_protocol.Schedule.WINDOW,
    }[schedule]
    return ncc_protocol.Plan(
        lanes=lanes,
        rounds=rounds,
        effect_relation=effect,
        range_relation=relation,
        schedule=raw_schedule,
        wait_kind=ncc_protocol.WaitKind.BY_WORKER,
        wait_worker_mask=participants,
        seed=_seed(key),
        first_operand=first_operand,
        second_operand=second_operand,
        flags=flags,
    )


def _probe(
    key: str,
    cell_key: str,
    family: Family,
    phase: Phase,
    plan: ncc_protocol.Plan,
    measurement_engines: tuple[Engine, ...],
    measurement: MeasurementOracle,
) -> ProbeCase:
    return ProbeCase(
        key=key,
        cell_key=cell_key,
        family=family,
        phase=phase,
        plan=plan,
        repeat=MIN_REPEATS,
        measurement_engines=measurement_engines,
        correctness=CORRECTNESS,
        measurement=measurement,
    )


def _single_probe(
    engine: Engine,
    label: str,
    transfer_bytes: int,
    phase: Phase,
    *,
    layout_kind: ncc_protocol.LayoutKind = (
        ncc_protocol.LayoutKind.CONTIGUOUS
    ),
    layout_inner_bytes: int = 0,
) -> ProbeCase:
    cell_key = f"single/{engine.value}/{label}"
    key = f"{cell_key}/serial"
    lane = _lane(
        engine,
        transfer_bytes,
        layout_kind=layout_kind,
        layout_inner_bytes=layout_inner_bytes,
    )
    return _probe(
        key,
        cell_key,
        Family.SINGLE_ENGINE_THROUGHPUT,
        phase,
        _plan(key, (lane,), 1, Schedule.SERIAL),
        (engine,),
        SINGLE_MEASUREMENT,
    )


def _single_cells() -> tuple[MatrixCell, ...]:
    cells: list[MatrixCell] = []

    def add_probe(probe: ProbeCase, label: str) -> None:
        dimensions: list[tuple[str, str | int]] = [
            ("engine", probe.measurement_engines[0].value),
            ("workload", label),
            (
                "bytes",
                probe.plan.lanes[0].transfer_bytes,
            ),
            (
                "layout",
                probe.plan.lanes[0].layout_kind.name.lower(),
            ),
        ]
        if probe.measurement_engines[0] == Engine.NE:
            dimensions.append(("work_macs", NE_WORK_MACS_BY_LABEL[label]))
        cells.append(
            MatrixCell(
                key=probe.cell_key,
                family=probe.family,
                phase=probe.phase,
                disposition=Disposition.BOARD_EXECUTABLE,
                dimensions=tuple(dimensions),
                probes=(probe,),
                correctness=CORRECTNESS,
                measurement=SINGLE_MEASUREMENT,
                reason=(
                    "bounded one-engine invocation; exact output, physical "
                    "guards, instruction count, and PMU delta precede slope use"
                ),
            )
        )

    for label, size, phase in (
        ("small", 256, Phase.CALIBRATION),
        ("steady-4k", 4096, Phase.CALIBRATION),
        ("steady-16k", 16384, Phase.CALIBRATION),
        ("tail-4352", 4352, Phase.HELD_OUT),
    ):
        add_probe(_single_probe(Engine.CT, label, size, phase), label)

    for label, size in (("small", 256), ("steady-16k", 16384)):
        add_probe(
            _single_probe(Engine.NE, label, size, Phase.CALIBRATION),
            label,
        )
    cells.append(
        MatrixCell(
            key="single/ne/tail",
            family=Family.SINGLE_ENGINE_THROUGHPUT,
            phase=Phase.HELD_OUT,
            disposition=Disposition.BOARD_EXECUTABLE_EXTERNAL,
            dimensions=(
                ("engine", Engine.NE.value),
                ("workload", "tail"),
                ("bytes", 0),
                ("layout", "non-divisible-gemm-tail"),
                ("work_macs", NE_TAIL_WORK_MACS),
            ),
            probes=(),
            correctness=CORRECTNESS,
            measurement=SINGLE_MEASUREMENT,
            reason=(
                "the dedicated NE tail throughput adapter executes the exact "
                "FP16 M65/K129/N129 GEMM with a compatible device-PMU window; "
                "the same CTest first executes fresh small/steady raw NCC "
                "points before any slope activation decision"
            ),
            delegated=(
                EvidenceReference(
                    NE_CALIBRATION_ASSET,
                    "SAFE_CASES",
                    ("ne-f16-tail-nn", "ne-bf16-tail-nn"),
                    (
                        "numeric/layout correctness only; no throughput PMU "
                        "activation signal"
                    ),
                ),
                EvidenceReference(
                    NE_TAIL_THROUGHPUT_ASSET,
                    "CASES",
                    ("single/ne/tail/f16-m65-k129-n129",),
                    (
                        "external board-executable PMU tail group; fresh raw "
                        "small/steady points run in the same invocation"
                    ),
                ),
            ),
            safe_alternative_keys=("single/ne/small", "single/ne/steady-16k"),
        )
    )

    for engine in (Engine.RDMA, Engine.WDMA, Engine.TDMA):
        for label, size, phase in (
            ("contiguous-256b", 256, Phase.CALIBRATION),
            ("contiguous-4k", 4096, Phase.CALIBRATION),
            ("contiguous-16k", 16384, Phase.CALIBRATION),
            ("contiguous-64k", 65536, Phase.HELD_OUT),
        ):
            add_probe(_single_probe(engine, label, size, phase), label)

    tdma_strided = _single_probe(
        Engine.TDMA,
        "inner-strided-16k",
        16384,
        Phase.HELD_OUT,
        layout_kind=ncc_protocol.LayoutKind.INNER_STRIDED,
        layout_inner_bytes=256,
    )
    add_probe(tdma_strided, "inner-strided-16k")

    # The current typed protocol intentionally accepts DMA stride only in an
    # exact RDMA->WDMA roundtrip.  This is still a useful held-out per-engine
    # PMU observation because the two operations are serial and both retain an
    # independent engine counter, while one exact compact/envelope oracle
    # closes the descriptor.
    cell_key = "single/rdma-wdma/strided-roundtrip-16k"
    probe_key = f"{cell_key}/serial"
    rdma = _lane(
        Engine.RDMA,
        16384,
        layout_kind=ncc_protocol.LayoutKind.DMA_STRIDED,
        layout_inner_bytes=256,
        layout_stride0_bytes=512,
        layout_iteration0=64,
    )
    wdma = dataclasses.replace(
        rdma, engine=ENGINE_TO_PROTOCOL[Engine.WDMA]
    )
    strided_probe = _probe(
        probe_key,
        cell_key,
        Family.SINGLE_ENGINE_THROUGHPUT,
        Phase.HELD_OUT,
        _plan(
            probe_key,
            (rdma, wdma),
            1,
            Schedule.SERIAL,
            effect=ncc_protocol.EffectRelation.RAW,
            relation=ncc_protocol.RangeRelation.EXACT,
            first_operand=ncc_protocol.Operand.WRITE,
            second_operand=ncc_protocol.Operand.READ0,
        ),
        (Engine.RDMA, Engine.WDMA),
        SINGLE_MEASUREMENT,
    )
    cells.append(
        MatrixCell(
            key=cell_key,
            family=Family.SINGLE_ENGINE_THROUGHPUT,
            phase=Phase.HELD_OUT,
            disposition=Disposition.BOARD_EXECUTABLE,
            dimensions=(
                ("engine", "rdma+wdma"),
                ("workload", "strided-roundtrip-16k"),
                ("bytes", 16384),
                ("layout", "dma-strided"),
            ),
            probes=(strided_probe,),
            correctness=CORRECTNESS,
            measurement=SINGLE_MEASUREMENT,
            reason=(
                "serial exact DMA roundtrip provides separate RDMA/WDMA PMU "
                "deltas without claiming a two-engine overlap measurement"
            ),
        )
    )
    return tuple(cells)


def _pair_transfers(
    engine_a: Engine, engine_b: Engine, balance: Balance
) -> tuple[int, int]:
    low_a = 256 if engine_a == Engine.NE else 4096
    low_b = 256 if engine_b == Engine.NE else 4096
    if balance == Balance.A_HEAVY:
        return (16384, low_b)
    if balance == Balance.B_HEAVY:
        return (low_a, 16384)
    return (16384, 16384)


def _memory_pair_case_names(
    engine_a: Engine,
    engine_b: Engine,
    *,
    cross_worker: bool,
) -> tuple[str, ...]:
    worker = "workers-0-1" if cross_worker else "worker-0"
    transfer_k = (
        16
        if Engine.CT in (engine_a, engine_b)
        or Engine.NE in (engine_a, engine_b)
        else 64
    )
    return tuple(
        (
            f"parallel-{engine_a.value}-{engine_b.value}-{transfer_k}k-"
            f"far-disjoint-a-b-{worker}-{schedule}"
        )
        for schedule in ("serial", "window")
    )


def _delegated_pair(
    engine_a: Engine,
    engine_b: Engine,
    placement: Placement,
    direction: str,
    balance: Balance,
    iterations: int,
) -> tuple[EvidenceReference, ...]:
    same_worker = placement == Placement.SAME_WORKER
    canonical = direction == "a-to-b"
    if (
        iterations == 4
        and balance == Balance.EQUAL_BYTES
        and canonical
        and same_worker
    ):
        return (
            EvidenceReference(
                MEMORY_PAIR_ASSET,
                MEMORY_PAIR_SYMBOL,
                _memory_pair_case_names(
                    engine_a, engine_b, cross_worker=False
                ),
                "same-worker four-round equal-16KiB sustained smoke",
            ),
        )
    if (
        (engine_a, engine_b) == (Engine.CT, Engine.RDMA)
        and iterations == 4
        and balance == Balance.EQUAL_BYTES
        and canonical
        and placement == Placement.CROSS_WORKER
    ):
        return (
            EvidenceReference(
                MEMORY_PAIR_ASSET,
                MEMORY_CROSS_WORKER_SYMBOL,
                _memory_pair_case_names(
                    engine_a, engine_b, cross_worker=True
                ),
                "cross-worker four-round equal-16KiB sustained smoke",
            ),
        )
    return ()


def _pair_cell(
    engine_a: Engine,
    engine_b: Engine,
    placement: Placement,
    direction: str,
    balance: Balance,
    iterations: int,
) -> MatrixCell:
    phase = (
        Phase.HELD_OUT
        if balance == Balance.EQUAL_BYTES
        else Phase.CALIBRATION
    )
    key = (
        f"pair/{engine_a.value}-{engine_b.value}/{placement.value}/"
        f"{direction}/{balance.value}/i{iterations}"
    )
    dimensions = (
        ("engine_a", engine_a.value),
        ("engine_b", engine_b.value),
        ("placement", placement.value),
        ("direction", direction),
        ("balance", balance.value),
        ("iterations", iterations),
    )
    delegated = _delegated_pair(
        engine_a, engine_b, placement, direction, balance, iterations
    )
    bytes_a, bytes_b = _pair_transfers(engine_a, engine_b, balance)
    worker_a, worker_b = (
        (0, 0)
        if placement == Placement.SAME_WORKER
        else (0, 1)
    )
    by_engine = {
        engine_a: _lane(engine_a, bytes_a, worker=worker_a),
        engine_b: _lane(engine_b, bytes_b, worker=worker_b),
    }
    ordered_engines = (
        (engine_a, engine_b)
        if direction == "a-to-b"
        else (engine_b, engine_a)
    )
    probes: list[ProbeCase] = []
    for schedule in (Schedule.SERIAL, Schedule.WINDOW):
        probe_key = f"{key}/{schedule.value}"
        probes.append(
            _probe(
                probe_key,
                key,
                Family.ENGINE_PAIR_STAGE_BALANCE,
                phase,
                _plan(
                    key,
                    tuple(by_engine[engine] for engine in ordered_engines),
                    iterations,
                    schedule,
                    flags=(
                        ncc_protocol.BOUNDED_PAIR_WINDOW
                        if (
                            iterations == 8
                            and schedule == Schedule.WINDOW
                        )
                        else 0
                    ),
                ),
                (engine_a, engine_b),
                PAIR_MEASUREMENT,
            )
        )
    return MatrixCell(
        key,
        Family.ENGINE_PAIR_STAGE_BALANCE,
        phase,
        Disposition.BOARD_EXECUTABLE,
        dimensions,
        tuple(probes),
        CORRECTNESS,
        PAIR_MEASUREMENT,
        (
            "one-factor serial/window pair; A/B byte ratio, issue direction, "
            "worker placement, and iteration count are explicit dimensions. "
            "Eight-iteration windows are two device-side four-round chunks "
            "with one recorded participant drain, never an unmatched D+2 "
            "submission. Existing sustained smoke evidence remains provenance "
            "only; this matched campaign executes the cell on the same basis "
            "as its ratio and reciprocal controls"
        ),
        delegated=delegated,
    )


def _pair_cells() -> tuple[MatrixCell, ...]:
    return tuple(
        _pair_cell(
            engine_a,
            engine_b,
            placement,
            direction,
            balance,
            iterations,
        )
        for engine_a, engine_b in ENGINE_PAIRS
        for placement in (Placement.SAME_WORKER, Placement.CROSS_WORKER)
        for direction in ("a-to-b", "b-to-a")
        for balance in (
            Balance.A_HEAVY,
            Balance.EQUAL_BYTES,
            Balance.B_HEAVY,
        )
        for iterations in (2, 4, 8)
    )


def _three_stage_core_cells() -> tuple[MatrixCell, ...]:
    rows: list[MatrixCell] = []
    for compute in (Engine.CT, Engine.NE):
        for regime in PipelineRegime:
            for schedule in (
                Schedule.SERIAL,
                Schedule.ONE_SLOT,
                Schedule.TWO_SLOT,
            ):
                for iterations in (1, 2, 3, 4, 8):
                    capacity = PipelineCapacity.FITS_TWO_SLOTS
                    key = (
                        f"pipeline/rdma-{compute.value}-wdma/"
                        f"{regime.value}/{schedule.value}/i{iterations}/"
                        f"{capacity.value}"
                    )
                    rows.append(
                        MatrixCell(
                            key=key,
                            family=Family.PRODUCTION_THREE_STAGE,
                            phase=(
                                Phase.HELD_OUT
                                if iterations in (1, 3, 8)
                                else Phase.CALIBRATION
                            ),
                            disposition=Disposition.PENDING_CONFIGURED_BOARD,
                            dimensions=(
                                ("ingress", Engine.RDMA.value),
                                ("compute", compute.value),
                                ("egress", Engine.WDMA.value),
                                ("regime", regime.value),
                                ("schedule", schedule.value),
                                ("iterations", iterations),
                                ("capacity", capacity.value),
                            ),
                            probes=(),
                            correctness=CORRECTNESS,
                            measurement=PIPELINE_MEASUREMENT,
                            reason=(
                                "the compiler-owned accepted-Instr fixed-slot "
                                "qualification closes current-IR structure and "
                                "host package/model/no-card gates; this cell "
                                "still requires fresh configured-board "
                                "baseline/winner correctness, which a raw "
                                "three-lane request cannot provide"
                            ),
                            safe_alternative_keys=(
                                "delegated/handwritten-double-slot-negative",
                            ),
                        )
                    )
    return tuple(rows)


def _three_stage_negative_cells() -> tuple[MatrixCell, ...]:
    rows: list[MatrixCell] = []
    for compute in (Engine.CT, Engine.NE):
        for regime in PipelineRegime:
            for iterations in (4, 8):
                key = (
                    f"pipeline/rdma-{compute.value}-wdma/"
                    f"{regime.value}/two-slot/i{iterations}/"
                    "fallback-required"
                )
                rows.append(
                    MatrixCell(
                        key=key,
                        family=Family.PRODUCTION_THREE_STAGE,
                        phase=Phase.HELD_OUT,
                        disposition=Disposition.HOST_EXACT_CLOSED,
                        dimensions=(
                            ("ingress", Engine.RDMA.value),
                            ("compute", compute.value),
                            ("egress", Engine.WDMA.value),
                            ("regime", regime.value),
                            ("schedule", Schedule.TWO_SLOT.value),
                            ("iterations", iterations),
                            (
                                "capacity",
                                PipelineCapacity.FALLBACK_REQUIRED.value,
                            ),
                            ("negative_kind", "capacity-fallback"),
                        ),
                        probes=(),
                        correctness=CORRECTNESS,
                        measurement=PIPELINE_MEASUREMENT,
                        reason=(
                            "an over-capacity two-slot candidate must be "
                            "rejected before allocation and retain the "
                            "one-slot/serial production fallback; the "
                            "compiler-owned producer and host exact rejection "
                            "gate now materialize and distinguish these "
                            "alternatives without board execution"
                        ),
                    )
                )
        for alias_relation in (
            "exact-input-output-alias",
            "partial-slot-overlap",
            "loop-carried-write-after-read",
        ):
            key = (
                f"pipeline/rdma-{compute.value}-wdma/alias/"
                f"{alias_relation}"
            )
            rows.append(
                MatrixCell(
                    key=key,
                    family=Family.PRODUCTION_THREE_STAGE,
                    phase=Phase.HELD_OUT,
                    disposition=Disposition.HOST_EXACT_CLOSED,
                    dimensions=(
                        ("ingress", Engine.RDMA.value),
                        ("compute", compute.value),
                        ("egress", Engine.WDMA.value),
                        ("alias_relation", alias_relation),
                        ("negative_kind", "alias-legality"),
                    ),
                    probes=(),
                    correctness=CORRECTNESS,
                    measurement=PIPELINE_MEASUREMENT,
                    reason=(
                        "typed SSA/range effects and the production host exact "
                        "gate reject the unsafe slot rotation; a disjoint raw "
                        "request cannot stand in for this legality proof"
                    ),
                )
            )
    rows.append(
        MatrixCell(
            key="pipeline/production-provenance/handwritten-double-slot",
            family=Family.PRODUCTION_THREE_STAGE,
            phase=Phase.HELD_OUT,
            disposition=Disposition.HOST_EXACT_CLOSED,
            dimensions=(
                ("negative_kind", "handwritten-provenance"),
                ("iterations", "1/2/3/4"),
                ("schedule", "serial/window"),
            ),
            probes=(),
            correctness=CORRECTNESS,
            measurement=PIPELINE_MEASUREMENT,
            reason=(
                "the existing raw double-slot cases prove only hardware issue "
                "and reuse behavior, never compiler materialization"
            ),
            delegated=(
                EvidenceReference(
                    RAW_PROBE_ASSET,
                    HANDWRITTEN_PIPELINE_SYMBOL,
                    tuple(
                        (
                            f"double-slot-iterations{iterations}-{schedule}-"
                            "observation"
                        )
                        for iterations in (1, 2, 3, 4)
                        for schedule in ("serial", "window")
                    ),
                    "negative provenance control only",
                ),
            ),
        )
    )
    return tuple(rows)


SINGLE_ENGINE_CELLS = _single_cells()
ENGINE_PAIR_CELLS = _pair_cells()
THREE_STAGE_CORE_CELLS = _three_stage_core_cells()
THREE_STAGE_NEGATIVE_CELLS = _three_stage_negative_cells()
THREE_STAGE_CELLS = THREE_STAGE_CORE_CELLS + THREE_STAGE_NEGATIVE_CELLS
THREE_STAGE_PENDING_BOARD_CELLS = THREE_STAGE_CORE_CELLS
THREE_STAGE_HOST_EXACT_CELLS = THREE_STAGE_NEGATIVE_CELLS
ALL_CELLS = SINGLE_ENGINE_CELLS + ENGINE_PAIR_CELLS + THREE_STAGE_CELLS
CELLS_BY_KEY = {cell.key: cell for cell in ALL_CELLS}
BOARD_PROBES = tuple(
    probe
    for cell in ALL_CELLS
    if cell.disposition == Disposition.BOARD_EXECUTABLE
    for probe in cell.probes
)
PROBES_BY_KEY = {probe.key: probe for probe in BOARD_PROBES}
PROBES_BY_NAME = {probe.name: probe for probe in BOARD_PROBES}
CASE_KEYS_BY_FAMILY = {
    family: tuple(
        cell.key for cell in ALL_CELLS if cell.family == family
    )
    for family in Family
}
DELEGATED_EVIDENCE = (
    EvidenceReference(
        RAW_PROBE_ASSET,
        HANDWRITTEN_PIPELINE_SYMBOL,
        tuple(
            f"double-slot-iterations{iterations}-{schedule}-observation"
            for iterations in (1, 2, 3, 4)
            for schedule in ("serial", "window")
        ),
        (
            "hardware-only negative control; never satisfies a production "
            "three-stage cell"
        ),
    ),
)


def cells_for_family(family: Family) -> tuple[MatrixCell, ...]:
    return tuple(cell for cell in ALL_CELLS if cell.family == family)


def probes_for_cells(keys: Iterable[str]) -> tuple[ProbeCase, ...]:
    selected: list[ProbeCase] = []
    for key in keys:
        if key not in CELLS_BY_KEY:
            raise KeyError(f"unknown characterization cell {key!r}")
        cell = CELLS_BY_KEY[key]
        if cell.disposition != Disposition.BOARD_EXECUTABLE:
            raise RuntimeError(
                f"{key}: disposition is {cell.disposition.value}, not "
                "board-executable"
            )
        selected.extend(cell.probes)
    return tuple(selected)


def pair_activation_cell_keys(
    engine_a: Engine,
    engine_b: Engine,
    placement: Placement,
    iterations: int,
) -> tuple[str, ...]:
    if ENGINE_ORDER.index(engine_a) >= ENGINE_ORDER.index(engine_b):
        raise ValueError("activation pair must use canonical engine order")
    return tuple(
        (
            f"pair/{engine_a.value}-{engine_b.value}/{placement.value}/"
            f"{direction}/{balance.value}/i{iterations}"
        )
        for direction in ("a-to-b", "b-to-a")
        for balance in (
            Balance.A_HEAVY,
            Balance.EQUAL_BYTES,
            Balance.B_HEAVY,
        )
    )


def _board_activation_groups() -> tuple[BoardActivationGroup, ...]:
    groups = [
        BoardActivationGroup(
            "single-engines",
            tuple(
                cell.key
                for cell in SINGLE_ENGINE_CELLS
                if cell.disposition == Disposition.BOARD_EXECUTABLE
            ),
        )
    ]
    for engine_a, engine_b in ENGINE_PAIRS:
        for placement in Placement:
            for iterations in (2, 4, 8):
                groups.append(
                    BoardActivationGroup(
                        (
                            f"pair-{engine_a.value}-{engine_b.value}-"
                            f"{placement.value}-i{iterations}"
                        ),
                        pair_activation_cell_keys(
                            engine_a,
                            engine_b,
                            placement,
                            iterations,
                        ),
                    )
                )
    return tuple(groups)


BOARD_ACTIVATION_GROUPS = _board_activation_groups()
BOARD_ACTIVATION_GROUPS_BY_KEY = {
    group.key: group for group in BOARD_ACTIVATION_GROUPS
}
EXTERNAL_BOARD_GROUP_KEYS = ("single-ne-tail",)


def single_activation_cell_keys(engine: Engine) -> tuple[str, ...]:
    keys = [
        cell.key
        for cell in SINGLE_ENGINE_CELLS
        if (
            cell.disposition == Disposition.BOARD_EXECUTABLE
            and (
                cell.dimension("engine") == engine.value
                or (
                    engine in (Engine.RDMA, Engine.WDMA)
                    and cell.key
                    == "single/rdma-wdma/strided-roundtrip-16k"
                )
            )
        )
    ]
    return tuple(keys)


def _observation_case_key(observation: Mapping[str, object]) -> str | None:
    case = observation.get("case")
    if not isinstance(case, Mapping):
        return None
    key = case.get("characterization_key")
    return key if isinstance(key, str) else None


def _validate_runtime_evidence(
    observation: Mapping[str, object],
) -> tuple[bool, str | None]:
    lifecycle = observation.get("runtime_lifecycle")
    terminal_completion = observation.get("runtime_terminal_completion")
    if (
        not isinstance(lifecycle, (list, tuple))
        or tuple(lifecycle) != RUNTIME_LIFECYCLE
    ):
        return False, "runtime lifecycle is not the exact ordered rank-one path"
    if type(terminal_completion) is not int or terminal_completion < 0:
        return False, "runtime terminal completion is absent"
    return True, None


def _validate_observation(
    probe: ProbeCase, observation: Mapping[str, object]
) -> tuple[bool, str | None]:
    runtime_valid, runtime_reason = _validate_runtime_evidence(observation)
    if not runtime_valid:
        return False, runtime_reason
    execution = observation.get("execution_delta")
    instructions = observation.get("instruction_delta")
    if not isinstance(execution, Mapping) or not isinstance(
        instructions, Mapping
    ):
        return False, "missing typed execution/instruction deltas"
    expected: dict[str, int] = defaultdict(int)
    for identity in probe.plan.issue_identities():
        lane = probe.plan.lanes[identity.lane]
        expected[
            f"worker{lane.worker}.{lane.engine.name.lower()}"
        ] += 1
    for key, count in expected.items():
        if instructions.get(key) != count:
            return False, f"{key} count does not match {count}"
    for engine in probe.measurement_engines:
        value = execution.get(engine.value)
        if type(value) is not int or value <= 0:
            return False, f"{engine.value} execution delta is not positive"
    full = execution.get("full")
    if type(full) is not int or full <= 0:
        return False, "global union execution delta is not positive"
    return True, None


def _ne_tail_median(
    rows: list[Mapping[str, object]],
) -> float:
    values: list[int] = []
    for row in rows:
        runtime_valid, runtime_reason = _validate_runtime_evidence(row)
        if not runtime_valid:
            raise ValueError(
                f"{NE_TAIL_CHARACTERIZATION_KEY}: {runtime_reason}"
            )
        case = row.get("case")
        instructions = row.get("instruction_delta")
        execution = row.get("execution_delta")
        if (
            not isinstance(case, Mapping)
            or case.get("characterization_key")
            != NE_TAIL_CHARACTERIZATION_KEY
            or case.get("characterization_cell") != "single/ne/tail"
            or case.get("work_macs") != NE_TAIL_WORK_MACS
            or instructions != {"worker0.ne": 1}
            or not isinstance(execution, Mapping)
            or type(execution.get("ne")) is not int
            or execution["ne"] <= 0
            or type(row.get("pmu_enable")) is not int
            or row["pmu_enable"] == 0
            or type(row.get("execute_result")) is not int
            or row["execute_result"] == 0
        ):
            raise ValueError(
                f"{NE_TAIL_CHARACTERIZATION_KEY}: incompatible typed tail "
                "observation"
            )
        for digest_name in ("logical_sha256", "physical_sha256"):
            digest = row.get(digest_name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"{NE_TAIL_CHARACTERIZATION_KEY}: {digest_name} is invalid"
                )
        values.append(execution["ne"])
    return float(statistics.median(values))


def _group_observations(
    observations: Iterable[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for observation in observations:
        key = _observation_case_key(observation)
        if key is not None:
            grouped[key].append(observation)
    return grouped


def _probe_median(
    probe: ProbeCase,
    rows: list[Mapping[str, object]],
    metric: str,
) -> float:
    values: list[int] = []
    for row in rows:
        valid, reason = _validate_observation(probe, row)
        if not valid:
            raise ValueError(f"{probe.key}: {reason}")
        execution = row["execution_delta"]
        assert isinstance(execution, Mapping)
        value = execution.get(metric)
        if type(value) is not int:
            raise ValueError(f"{probe.key}: metric {metric!r} is absent")
        values.append(value)
    return float(statistics.median(values))


def evaluate_single_engine_activation(
    engine: Engine,
    observations: Iterable[Mapping[str, object]],
) -> ActivationDecision:
    grouped = _group_observations(observations)
    required_cells = single_activation_cell_keys(engine)
    required_probes = tuple(
        probe
        for key in required_cells
        for probe in CELLS_BY_KEY[key].probes
    )
    reasons: list[str] = []
    medians: dict[str, float] = {}
    for probe in required_probes:
        rows = grouped.get(probe.key, [])
        samples = {
            row.get("sample")
            for row in rows
            if type(row.get("sample")) is int
        }
        if len(rows) < probe.repeat or len(samples) < probe.repeat:
            reasons.append(
                f"{probe.key}: needs {probe.repeat} distinct samples"
            )
            continue
        try:
            medians[probe.key] = _probe_median(
                probe, rows, engine.value
            )
        except ValueError as error:
            reasons.append(str(error))

    if engine == Engine.NE:
        tail_rows = grouped.get(NE_TAIL_CHARACTERIZATION_KEY, [])
        tail_samples = {
            row.get("sample")
            for row in tail_rows
            if type(row.get("sample")) is int
        }
        if (
            len(tail_rows) < MIN_REPEATS
            or len(tail_samples) < MIN_REPEATS
        ):
            reasons.append(
                f"{NE_TAIL_CHARACTERIZATION_KEY}: needs {MIN_REPEATS} "
                "distinct external samples"
            )
        else:
            try:
                medians[NE_TAIL_CHARACTERIZATION_KEY] = _ne_tail_median(
                    tail_rows
                )
            except ValueError as error:
                reasons.append(str(error))
    if reasons:
        return ActivationDecision(
            Family.SINGLE_ENGINE_THROUGHPUT,
            engine.value,
            "unknown",
            tuple(sorted(set(reasons))),
            tuple(sorted(medians.items())),
        )

    if engine == Engine.NE:
        work_points = sorted(
            [
                (
                    int(CELLS_BY_KEY[probe.cell_key].dimension("work_macs")),
                    medians[probe.key],
                )
                for probe in required_probes
            ]
            + [
                (
                    int(CELLS_BY_KEY["single/ne/tail"].dimension("work_macs")),
                    medians[NE_TAIL_CHARACTERIZATION_KEY],
                )
            ]
        )
        if len(work_points) != 3:
            reasons.append("NE small/steady/tail work triplet is incomplete")
        elif not (
            all(
                next_cycles >= cycles
                for (_, cycles), (_, next_cycles) in zip(
                    work_points, work_points[1:]
                )
            )
            and work_points[-1][1] > work_points[0][1]
        ):
            reasons.append(
                "NE small/steady/tail medians do not retain a positive "
                "work direction"
            )
        return ActivationDecision(
            Family.SINGLE_ENGINE_THROUGHPUT,
            engine.value,
            "unknown" if reasons else "activated",
            tuple(reasons),
            tuple(sorted(medians.items())),
        )

    contiguous = [
        (
            int(CELLS_BY_KEY[probe.cell_key].dimension("bytes")),
            medians[probe.key],
        )
        for probe in required_probes
        if (
            probe.plan.lanes[0].layout_kind
            == ncc_protocol.LayoutKind.CONTIGUOUS
            and len(probe.plan.lanes) == 1
            and (
                CELLS_BY_KEY[probe.cell_key].dimension("workload")
                != "tail-4352"
            )
        )
    ]
    contiguous.sort()
    if len(contiguous) < 3:
        reasons.append("fewer than three contiguous work points")
    elif not (
        all(
            next_cycles >= cycles
            for (_, cycles), (_, next_cycles) in zip(
                contiguous, contiguous[1:]
            )
        )
        and contiguous[-1][1] > contiguous[0][1]
    ):
        reasons.append("contiguous medians do not retain a positive slope")

    held_out = [
        medians[probe.key]
        for probe in required_probes
        if probe.phase == Phase.HELD_OUT
    ]
    if not held_out or any(value <= contiguous[0][1] for value in held_out):
        reasons.append(
            "held-out workload does not preserve the positive direction"
        )
    return ActivationDecision(
        Family.SINGLE_ENGINE_THROUGHPUT,
        engine.value,
        "unknown" if reasons else "activated",
        tuple(reasons),
        tuple(sorted(medians.items())),
    )


def evaluate_pair_activation(
    engine_a: Engine,
    engine_b: Engine,
    placement: Placement,
    iterations: int,
    observations: Iterable[Mapping[str, object]],
) -> ActivationDecision:
    scope = (
        f"{engine_a.value}-{engine_b.value}/{placement.value}/i{iterations}"
    )
    grouped = _group_observations(observations)
    cell_keys = pair_activation_cell_keys(
        engine_a, engine_b, placement, iterations
    )
    reasons: list[str] = []
    signed_imbalance: dict[tuple[str, str], float] = {}
    medians: dict[str, float] = {}
    for cell_key in cell_keys:
        cell = CELLS_BY_KEY[cell_key]
        if cell.disposition != Disposition.BOARD_EXECUTABLE:
            reasons.append(
                f"{cell_key}: requires {cell.disposition.value} evidence"
            )
            continue
        by_schedule = {
            probe.plan.schedule.name.lower(): probe for probe in cell.probes
        }
        schedule_metrics: dict[str, tuple[float, float, float]] = {}
        for schedule in ("serial", "window"):
            probe = by_schedule[schedule]
            rows = grouped.get(probe.key, [])
            samples = {
                row.get("sample")
                for row in rows
                if type(row.get("sample")) is int
            }
            if len(rows) < probe.repeat or len(samples) < probe.repeat:
                reasons.append(
                    f"{probe.key}: needs {probe.repeat} distinct samples"
                )
                continue
            try:
                a_cycles = _probe_median(probe, rows, engine_a.value)
                b_cycles = _probe_median(probe, rows, engine_b.value)
                full_cycles = _probe_median(probe, rows, "full")
            except ValueError as error:
                reasons.append(str(error))
                continue
            schedule_metrics[schedule] = (
                a_cycles,
                b_cycles,
                full_cycles,
            )
            medians[f"{probe.key}:{engine_a.value}"] = a_cycles
            medians[f"{probe.key}:{engine_b.value}"] = b_cycles
            medians[f"{probe.key}:full"] = full_cycles
        if set(schedule_metrics) != {"serial", "window"}:
            continue
        serial = schedule_metrics["serial"]
        window = schedule_metrics["window"]
        serial_excess = serial[0] + serial[1] - serial[2]
        window_excess = window[0] + window[1] - window[2]
        if window_excess <= max(0.0, serial_excess):
            reasons.append(
                f"{cell_key}: no stable serial-relative FU overlap signal"
            )
        direction = str(cell.dimension("direction"))
        balance = str(cell.dimension("balance"))
        denominator = window[0] + window[1]
        signed_imbalance[(direction, balance)] = (
            (window[0] - window[1]) / denominator
        )

    for direction in ("a-to-b", "b-to-a"):
        values = [
            signed_imbalance.get((direction, balance.value))
            for balance in (
                Balance.A_HEAVY,
                Balance.EQUAL_BYTES,
                Balance.B_HEAVY,
            )
        ]
        if any(value is None for value in values):
            reasons.append(f"{direction}: ratio triplet is incomplete")
        elif not (values[0] > values[1] > values[2]):
            reasons.append(
                f"{direction}: measured imbalance does not order A-heavy, "
                "equal-bytes, B-heavy"
            )
    return ActivationDecision(
        Family.ENGINE_PAIR_STAGE_BALANCE,
        scope,
        "unknown" if reasons else "activated",
        tuple(sorted(set(reasons))),
        tuple(sorted(medians.items())),
    )


def validate_production_pipeline_inputs(
    package_dir: pathlib.Path | None,
) -> ProductionPreparationDecision:
    """Validate the closed compiler-owned fixed-slot qualification sibling."""

    def fail(message: str) -> None:
        raise ValueError(message)

    def load_closed_json(path: pathlib.Path, label: str) -> dict[str, object]:
        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    fail(f"{label} contains duplicate field {key!r}")
                result[key] = value
            return result

        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=reject_duplicates,
            )
        except (OSError, json.JSONDecodeError) as error:
            fail(f"{label} is unreadable: {error}")
        if not isinstance(value, dict):
            fail(f"{label} is not an object")
        return value

    def require_fields(
        value: object, fields: set[str], label: str
    ) -> dict[str, object]:
        if not isinstance(value, dict):
            fail(f"{label} is not an object")
        actual = set(value)
        if actual != fields:
            fail(
                f"{label} fields differ: missing={sorted(fields - actual)!r} "
                f"unknown={sorted(actual - fields)!r}"
            )
        return value

    def exact_int(value: object, label: str, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            fail(f"{label} must be an integer >= {minimum}")
        return value

    def signed_int(value: object, label: str) -> int:
        if type(value) is not int:
            fail(f"{label} must be an integer")
        return value

    def exact_bool(value: object, label: str) -> bool:
        if type(value) is not bool:
            fail(f"{label} must be a boolean")
        return value

    def exact_string(value: object, label: str) -> str:
        if not isinstance(value, str) or not value:
            fail(f"{label} must be a non-empty string")
        return value

    def digest_bytes(payload: bytes) -> str:
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def require_digest(value: object, label: str) -> str:
        digest = exact_string(value, label)
        if (
            not digest.startswith("sha256:")
            or len(digest) != len("sha256:") + 64
            or any(
                character not in "0123456789abcdef"
                for character in digest[len("sha256:") :]
            )
        ):
            fail(f"{label} is not a canonical sha256 digest")
        return digest

    def require_list(value: object, label: str) -> list[object]:
        if not isinstance(value, list):
            fail(f"{label} is not an array")
        return value

    reasons: list[str] = []
    checked: str | None = None
    if package_dir is None:
        reasons.append("no production package was supplied")
    else:
        package = package_dir.absolute()
        checked = str(package)
        instrumentation = pathlib.Path(str(package) + ".qualification")
        try:
            if not package.is_dir() or package.is_symlink():
                fail("production package is not a regular directory")
            manifest_path = package / "manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                fail("production package has no regular manifest.json")
            manifest = load_closed_json(
                manifest_path, "production package manifest"
            )
            if manifest.get("schema_version") != 7:
                fail("production package schema is not the accepted schema-v7")
            manifest_rank_count = exact_int(
                manifest.get("rank_count"),
                "production manifest rank_count",
                1,
            )
            manifest_target = manifest.get("target")
            if not isinstance(manifest_target, dict):
                fail("production manifest target is not an object")
            manifest_identity = exact_string(
                manifest_target.get("identity"),
                "production manifest target identity",
            )
            manifest_digest = digest_bytes(manifest_path.read_bytes())

            if not instrumentation.is_dir() or instrumentation.is_symlink():
                fail("production package has no regular qualification sibling")
            entries = {entry.name: entry for entry in instrumentation.iterdir()}
            expected_entries = {"activation.json", "attestation.json"}
            if set(entries) != expected_entries:
                fail(
                    "qualification sibling members differ: "
                    f"missing={sorted(expected_entries - set(entries))!r} "
                    f"unknown={sorted(set(entries) - expected_entries)!r}"
                )
            for name, entry in entries.items():
                if not entry.is_file() or entry.is_symlink():
                    fail(
                        f"qualification sibling member {name!r} is not a "
                        "regular file"
                    )

            attestation_path = entries["attestation.json"]
            activation = require_fields(
                load_closed_json(
                    entries["activation.json"],
                    "qualification activation",
                ),
                {
                    "schema",
                    "schema_version",
                    "manifest_sha256",
                    "attestation_sha256",
                },
                "qualification activation",
            )
            if (
                activation["schema"]
                != "wafer-static-fixed-slot-qualification-activation"
                or exact_int(
                    activation["schema_version"],
                    "qualification activation schema_version",
                    1,
                )
                != 1
            ):
                fail("qualification activation schema is unknown")
            if (
                require_digest(
                    activation["manifest_sha256"],
                    "qualification activation manifest digest",
                )
                != manifest_digest
            ):
                fail("qualification activation has a stale manifest binding")
            if (
                require_digest(
                    activation["attestation_sha256"],
                    "qualification activation attestation digest",
                )
                != digest_bytes(attestation_path.read_bytes())
            ):
                fail("qualification attestation bytes were tampered")

            attestation = require_fields(
                load_closed_json(
                    attestation_path, "qualification attestation"
                ),
                {
                    "schema",
                    "schema_version",
                    "selection_kind",
                    "manifest_sha256",
                    "accepted_instr_digest_basis",
                    "target",
                    "ranks",
                },
                "qualification attestation",
            )
            if (
                attestation["schema"]
                != "wafer-static-fixed-slot-qualification"
                or exact_int(
                    attestation["schema_version"],
                    "qualification attestation schema_version",
                    1,
                )
                != 1
            ):
                fail("qualification attestation schema is unknown")
            if attestation["selection_kind"] != "static-fixed-slot":
                fail("qualification selection kind is not static-fixed-slot")
            if (
                attestation["accepted_instr_digest_basis"]
                != "final-accepted-instr-module-text-v1"
            ):
                fail("qualification accepted-Instr digest basis is unknown")
            if (
                require_digest(
                    attestation["manifest_sha256"],
                    "qualification attestation manifest digest",
                )
                != manifest_digest
            ):
                fail("qualification attestation has a stale manifest binding")

            target = require_fields(
                attestation["target"],
                {"identity", "rank_count", "logical_ranks"},
                "qualification target",
            )
            if target["identity"] != manifest_identity:
                fail("qualification target identity differs from manifest")
            if manifest_identity != "wafer-tx81-single-card":
                fail("qualification target identity is not closed")
            if (
                exact_int(
                    target["rank_count"],
                    "qualification target rank_count",
                    1,
                )
                != manifest_rank_count
            ):
                fail("qualification rank_count differs from manifest")
            logical_ranks = require_list(
                target["logical_ranks"],
                "qualification target logical_ranks",
            )
            if (
                any(type(rank) is not int for rank in logical_ranks)
                or logical_ranks != list(range(manifest_rank_count))
            ):
                fail("qualification logical rank domain is not canonical")

            ranks = require_list(attestation["ranks"], "qualification ranks")
            if len(ranks) != manifest_rank_count:
                fail("qualification ranks do not cover the manifest domain")
            engine_order = {
                "ct": 0,
                "ne": 1,
                "rdma": 2,
                "wdma": 3,
                "tdma": 4,
            }
            for logical_rank, raw_rank in enumerate(ranks):
                rank = require_fields(
                    raw_rank,
                    {
                        "logical_rank",
                        "accepted_instr_sha256",
                        "spm_alloc_roots",
                        "static_loops",
                        "engine_worker_issues",
                        "dte",
                        "completion",
                    },
                    f"qualification rank {logical_rank}",
                )
                if (
                    exact_int(
                        rank["logical_rank"],
                        f"qualification rank {logical_rank} identity",
                    )
                    != logical_rank
                ):
                    fail(
                        f"qualification rank {logical_rank} identity is not "
                        "canonical"
                    )
                require_digest(
                    rank["accepted_instr_sha256"],
                    f"qualification rank {logical_rank} accepted-Instr digest",
                )

                roots = require_list(
                    rank["spm_alloc_roots"],
                    f"qualification rank {logical_rank} SPM roots",
                )
                if len(roots) < 2:
                    fail(
                        f"qualification rank {logical_rank} has fewer than "
                        "two SPM roots"
                    )
                root_intervals: list[tuple[int, int, int]] = []
                for ordinal, raw_root in enumerate(roots):
                    root = require_fields(
                        raw_root,
                        {
                            "ordinal",
                            "offset",
                            "bytes",
                            "range_begin",
                            "range_end",
                        },
                        f"qualification rank {logical_rank} SPM root {ordinal}",
                    )
                    if (
                        exact_int(
                            root["ordinal"],
                            f"qualification rank {logical_rank} root ordinal",
                        )
                        != ordinal
                    ):
                        fail(
                            f"qualification rank {logical_rank} SPM root "
                            "ordinals are not dense"
                        )
                    offset = exact_int(
                        root["offset"],
                        f"qualification rank {logical_rank} root offset",
                    )
                    byte_count = exact_int(
                        root["bytes"],
                        f"qualification rank {logical_rank} root bytes",
                        1,
                    )
                    range_begin = exact_int(
                        root["range_begin"],
                        f"qualification rank {logical_rank} range_begin",
                    )
                    range_end = exact_int(
                        root["range_end"],
                        f"qualification rank {logical_rank} range_end",
                        1,
                    )
                    if range_begin != offset or range_end != offset + byte_count:
                        fail(
                            f"qualification rank {logical_rank} SPM root "
                            "range is inconsistent"
                        )
                    if (
                        range_begin
                        < memory_catalog.SPM_ALLOCATABLE_BEGIN
                        or range_end > memory_catalog.SPM_ALLOCATABLE_END
                    ):
                        fail(
                            f"qualification rank {logical_rank} SPM root "
                            "is outside the allocatable target arena"
                        )
                    if range_begin % QUALIFICATION_SPM_ALIGNMENT != 0:
                        fail(
                            f"qualification rank {logical_rank} SPM root "
                            "is not placement-aligned"
                        )
                    root_intervals.append((range_begin, range_end, ordinal))
                root_intervals.sort()
                for previous, current in zip(
                    root_intervals, root_intervals[1:]
                ):
                    if previous[1] > current[0]:
                        fail(
                            f"qualification rank {logical_rank} SPM root "
                            "intervals overlap"
                        )

                loops = require_list(
                    rank["static_loops"],
                    f"qualification rank {logical_rank} static loops",
                )
                if not loops:
                    fail(
                        f"qualification rank {logical_rank} has no static loop"
                    )
                saw_root_cycle = False
                for ordinal, raw_loop in enumerate(loops):
                    loop = require_fields(
                        raw_loop,
                        {
                            "ordinal",
                            "lower",
                            "upper",
                            "step",
                            "trip_count",
                            "iter_arg_count",
                            "spm_iter_arg_rotations",
                        },
                        f"qualification rank {logical_rank} loop {ordinal}",
                    )
                    if (
                        exact_int(
                            loop["ordinal"],
                            f"qualification rank {logical_rank} loop ordinal",
                        )
                        != ordinal
                    ):
                        fail(
                            f"qualification rank {logical_rank} loop "
                            "ordinals are not dense"
                        )
                    lower = signed_int(
                        loop["lower"],
                        f"qualification rank {logical_rank} loop lower",
                    )
                    upper = signed_int(
                        loop["upper"],
                        f"qualification rank {logical_rank} loop upper",
                    )
                    step = exact_int(
                        loop["step"],
                        f"qualification rank {logical_rank} loop step",
                        1,
                    )
                    trip_count = exact_int(
                        loop["trip_count"],
                        f"qualification rank {logical_rank} loop trip_count",
                        1,
                    )
                    expected_trips = (
                        0 if upper <= lower else (upper - lower + step - 1) // step
                    )
                    if trip_count != expected_trips:
                        fail(
                            f"qualification rank {logical_rank} loop trip "
                            "count is inconsistent"
                        )
                    iter_arg_count = exact_int(
                        loop["iter_arg_count"],
                        f"qualification rank {logical_rank} iter_arg_count",
                    )
                    rotations = require_list(
                        loop["spm_iter_arg_rotations"],
                        f"qualification rank {logical_rank} SPM rotations",
                    )
                    spm_args: set[int] = set()
                    initial_roots: set[int] = set()
                    next_roots: set[int] = set()
                    for rotation_ordinal, raw_rotation in enumerate(rotations):
                        rotation = require_fields(
                            raw_rotation,
                            {
                                "iter_arg",
                                "initial_root",
                                "next_iter_arg",
                                "next_root",
                            },
                            (
                                f"qualification rank {logical_rank} loop "
                                f"{ordinal} rotation {rotation_ordinal}"
                            ),
                        )
                        iter_arg = exact_int(
                            rotation["iter_arg"],
                            "qualification rotation iter_arg",
                        )
                        initial_root = exact_int(
                            rotation["initial_root"],
                            "qualification rotation initial_root",
                        )
                        next_iter_arg = exact_int(
                            rotation["next_iter_arg"],
                            "qualification rotation next_iter_arg",
                        )
                        next_root = exact_int(
                            rotation["next_root"],
                            "qualification rotation next_root",
                        )
                        if (
                            iter_arg >= iter_arg_count
                            or next_iter_arg >= iter_arg_count
                            or initial_root >= len(roots)
                            or next_root >= len(roots)
                            or iter_arg in spm_args
                            or initial_root in initial_roots
                            or next_root in next_roots
                            or initial_root == next_root
                        ):
                            fail(
                                f"qualification rank {logical_rank} SPM "
                                "root cycle is not a bounded permutation"
                            )
                        spm_args.add(iter_arg)
                        initial_roots.add(initial_root)
                        next_roots.add(next_root)
                    if initial_roots != next_roots:
                        fail(
                            f"qualification rank {logical_rank} SPM root "
                            "cycle is not closed"
                        )
                    saw_root_cycle |= len(initial_roots) >= 2
                if not saw_root_cycle:
                    fail(
                        f"qualification rank {logical_rank} has no proven "
                        "multi-root nonidentity SPM cycle"
                    )

                issue_rows = require_list(
                    rank["engine_worker_issues"],
                    f"qualification rank {logical_rank} engine issues",
                )
                issue_keys: list[tuple[int, int]] = []
                issue_engines: set[str] = set()
                issue_total = 0
                for issue_ordinal, raw_issue in enumerate(issue_rows):
                    issue = require_fields(
                        raw_issue,
                        {"engine", "worker", "count"},
                        (
                            f"qualification rank {logical_rank} engine issue "
                            f"{issue_ordinal}"
                        ),
                    )
                    engine = exact_string(
                        issue["engine"], "qualification issue engine"
                    )
                    if engine not in engine_order:
                        fail("qualification issue engine is unknown")
                    worker = exact_int(
                        issue["worker"], "qualification issue worker"
                    )
                    if worker >= 3:
                        fail("qualification issue worker is outside [0, 3)")
                    count = exact_int(
                        issue["count"], "qualification issue count", 1
                    )
                    issue_keys.append((engine_order[engine], worker))
                    issue_engines.add(engine)
                    issue_total += count
                if issue_keys != sorted(set(issue_keys)):
                    fail(
                        f"qualification rank {logical_rank} engine issue "
                        "inventory is not canonical"
                    )
                if not (
                    {"rdma", "wdma"} <= issue_engines
                    and issue_engines & {"ct", "ne"}
                ):
                    fail(
                        f"qualification rank {logical_rank} lacks a typed "
                        "load/compute/store pipeline"
                    )

                dte = require_fields(
                    rank["dte"],
                    {"token_count", "issues", "waits"},
                    f"qualification rank {logical_rank} DTE inventory",
                )
                token_count = exact_int(
                    dte["token_count"], "qualification DTE token_count"
                )
                dte_issues = require_list(
                    dte["issues"], "qualification DTE issues"
                )
                if len(dte_issues) != token_count:
                    fail("qualification DTE token count differs from issues")
                for ordinal, raw_issue in enumerate(dte_issues):
                    issue = require_fields(
                        raw_issue,
                        {"ordinal", "kind", "peer", "bytes", "token"},
                        f"qualification DTE issue {ordinal}",
                    )
                    if (
                        exact_int(
                            issue["ordinal"],
                            "qualification DTE issue ordinal",
                        )
                        != ordinal
                        or exact_int(
                            issue["token"],
                            "qualification DTE issue token",
                        )
                        != ordinal
                    ):
                        fail("qualification DTE issue ordinals are not dense")
                    if issue["kind"] not in {"send", "recv"}:
                        fail("qualification DTE issue kind is unknown")
                    peer = exact_int(issue["peer"], "qualification DTE peer")
                    if (
                        peer >= manifest_rank_count
                        or peer == logical_rank
                    ):
                        fail("qualification DTE peer is outside rank domain")
                    exact_int(
                        issue["bytes"], "qualification DTE issue bytes", 1
                    )
                dte_waits = require_list(
                    dte["waits"], "qualification DTE waits"
                )
                waited_tokens: list[int] = []
                for ordinal, raw_wait in enumerate(dte_waits):
                    wait = require_fields(
                        raw_wait,
                        {"ordinal", "tokens"},
                        f"qualification DTE wait {ordinal}",
                    )
                    if (
                        exact_int(
                            wait["ordinal"],
                            "qualification DTE wait ordinal",
                        )
                        != ordinal
                    ):
                        fail("qualification DTE wait ordinals are not dense")
                    tokens = require_list(
                        wait["tokens"], "qualification DTE wait tokens"
                    )
                    if not tokens:
                        fail("qualification DTE wait has no tokens")
                    waited_tokens.extend(
                        exact_int(token, "qualification DTE wait token")
                        for token in tokens
                    )
                if sorted(waited_tokens) != list(range(token_count)):
                    fail(
                        "qualification DTE waits do not consume every token "
                        "exactly once"
                    )

                completion = require_fields(
                    rank["completion"],
                    {"participant_joins", "behaviors"},
                    f"qualification rank {logical_rank} completion inventory",
                )
                joins = require_list(
                    completion["participant_joins"],
                    "qualification participant joins",
                )
                if not joins:
                    fail("qualification has no typed participant join")
                saw_terminal_join = False
                for ordinal, raw_join in enumerate(joins):
                    join = require_fields(
                        raw_join,
                        {
                            "ordinal",
                            "participants",
                            "inside_static_loop",
                        },
                        f"qualification participant join {ordinal}",
                    )
                    if (
                        exact_int(
                            join["ordinal"],
                            "qualification participant join ordinal",
                        )
                        != ordinal
                    ):
                        fail(
                            "qualification participant join ordinals are "
                            "not dense"
                        )
                    participants = require_list(
                        join["participants"],
                        "qualification join participants",
                    )
                    if (
                        not participants
                        or participants
                        != sorted(set(participants))
                        or any(
                            type(participant) is not int
                            or participant < 0
                            or participant >= 3
                            for participant in participants
                        )
                    ):
                        fail(
                            "qualification participant set is not canonical"
                        )
                    inside = exact_bool(
                        join["inside_static_loop"],
                        "qualification join loop location",
                    )
                    saw_terminal_join |= not inside
                if not saw_terminal_join:
                    fail(
                        "qualification has no completion join outside the "
                        "static loop"
                    )
                behaviors = require_list(
                    completion["behaviors"],
                    "qualification completion behaviors",
                )
                expected_behaviors = (
                    "ordered-pending",
                    "participant-join",
                    "synchronous-writeback",
                )
                if len(behaviors) != len(expected_behaviors):
                    fail("qualification completion behaviors are incomplete")
                behavior_counts: dict[str, int] = {}
                for expected_kind, raw_behavior in zip(
                    expected_behaviors, behaviors, strict=True
                ):
                    behavior = require_fields(
                        raw_behavior,
                        {"kind", "count"},
                        f"qualification completion behavior {expected_kind}",
                    )
                    if behavior["kind"] != expected_kind:
                        fail(
                            "qualification completion behavior order is not "
                            "canonical"
                        )
                    behavior_counts[expected_kind] = exact_int(
                        behavior["count"],
                        f"qualification completion count {expected_kind}",
                    )
                if behavior_counts["participant-join"] != len(joins):
                    fail(
                        "qualification participant join count is inconsistent"
                    )
                if (
                    behavior_counts["ordered-pending"]
                    + behavior_counts["synchronous-writeback"]
                    != issue_total
                ):
                    fail(
                        "qualification engine issue and completion counts "
                        "differ"
                    )
        except (OSError, ValueError) as error:
            reasons.append(str(error))
    return ProductionPreparationDecision(
        ready=not reasons,
        reasons=tuple(reasons),
        checked_package=checked,
        rejected_delegated_asset=(
            f"{RAW_PROBE_ASSET}:{HANDWRITTEN_PIPELINE_SYMBOL}"
        ),
    )


def validate_catalog() -> None:
    if len(CELLS_BY_KEY) != len(ALL_CELLS):
        raise RuntimeError("characterization cell keys are not unique")
    if len(PROBES_BY_KEY) != len(BOARD_PROBES):
        raise RuntimeError("characterization probe keys are not unique")
    if len(PROBES_BY_NAME) != len(BOARD_PROBES):
        raise RuntimeError("characterization probe names are not unique")
    if len(ENGINE_PAIRS) != 10:
        raise RuntimeError("engine stage-balance matrix lost a pair")
    if len(ENGINE_PAIR_CELLS) != 10 * 2 * 2 * 3 * 3:
        raise RuntimeError("engine stage-balance matrix is not full")
    if len(THREE_STAGE_CORE_CELLS) != 2 * 3 * 3 * 5:
        raise RuntimeError("production three-stage matrix is not full")
    if len(THREE_STAGE_NEGATIVE_CELLS) != 2 * 3 * 2 + 2 * 3 + 1:
        raise RuntimeError("production three-stage negative matrix is not full")
    if any(
        cell.disposition == Disposition.BOARD_EXECUTABLE
        and not cell.probes
        for cell in ALL_CELLS
    ):
        raise RuntimeError("board-executable cell has no concrete probe")
    if any(
        cell.disposition != Disposition.BOARD_EXECUTABLE
        and cell.probes
        for cell in ALL_CELLS
    ):
        raise RuntimeError("non-board cell unexpectedly has a probe")
    expected_ne_work = {
        "single/ne/small": NE_WORK_MACS_BY_LABEL["small"],
        "single/ne/steady-16k": NE_WORK_MACS_BY_LABEL["steady-16k"],
        "single/ne/tail": NE_TAIL_WORK_MACS,
    }
    if any(
        CELLS_BY_KEY[key].dimension("work_macs") != work_macs
        for key, work_macs in expected_ne_work.items()
    ):
        raise RuntimeError("NE small/steady/tail work units are inconsistent")
    for probe in BOARD_PROBES:
        if probe.repeat < MIN_REPEATS:
            raise RuntimeError(f"{probe.key}: performance repeat is too small")
        probe.plan.request_words()
        if probe.plan.command != ncc_protocol.Command.EXECUTE:
            raise RuntimeError(f"{probe.key}: probe is not executable")
        if not all(dataclasses.asdict(probe.correctness).values()):
            raise RuntimeError(f"{probe.key}: correctness oracle is incomplete")
        if probe.measurement.host_elapsed_is_evidence:
            raise RuntimeError(f"{probe.key}: host elapsed cannot activate cost")
    for cell in ENGINE_PAIR_CELLS:
        if cell.disposition == Disposition.BOARD_EXECUTABLE:
            schedules = {
                probe.plan.schedule for probe in cell.probes
            }
            if schedules != {
                ncc_protocol.Schedule.SERIAL,
                ncc_protocol.Schedule.WINDOW,
            }:
                raise RuntimeError(
                    f"{cell.key}: serial/window controls are incomplete"
                )
    if any(
        cell.disposition != Disposition.PENDING_CONFIGURED_BOARD
        for cell in THREE_STAGE_CORE_CELLS
    ):
        raise RuntimeError(
            "three-stage positive cell lost its configured-board gate"
        )
    if any(
        cell.disposition != Disposition.HOST_EXACT_CLOSED
        for cell in THREE_STAGE_NEGATIVE_CELLS
    ):
        raise RuntimeError(
            "three-stage structural negative lost its host-exact owner"
        )
    grouped_cells = tuple(
        cell_key
        for group in BOARD_ACTIVATION_GROUPS
        for cell_key in group.cell_keys
    )
    executable_cells = tuple(
        cell.key
        for cell in ALL_CELLS
        if cell.disposition == Disposition.BOARD_EXECUTABLE
    )
    if (
        len(BOARD_ACTIVATION_GROUPS_BY_KEY)
        != len(BOARD_ACTIVATION_GROUPS)
        or len(grouped_cells) != len(set(grouped_cells))
        or set(grouped_cells) != set(executable_cells)
    ):
        raise RuntimeError(
            "board activation groups do not partition executable cells"
        )


validate_catalog()
