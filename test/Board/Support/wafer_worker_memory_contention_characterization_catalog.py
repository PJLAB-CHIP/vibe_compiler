#!/usr/bin/env python3
"""Typed complete-Tile-domain DDR-contention characterization matrix.

Pipeline position:
- Upstream IR / input:
  Verified program-local NCC plans, the 16-Tile cluster lifecycle contract, owned
  SPM/DDR ranges, and the already-qualified PMU/status readback surfaces.
- Current stage responsibility:
  Bind active-Tile DDR-contention claims to concrete matched positive cases
  and retain non-duplicating references or typed boundaries for neighboring
  behavior.  Executable worker-placement and sustained-SPM matrices are owned
  by their dedicated board adapters and are not duplicated here.
- Output IR / files:
  A test-only complete-Tile-domain active-Tile DDR matrix, a real request contract, and
  bindings to already-existing board evidence or genuine typed boundaries.
  They are not compiler IR, scheduler hints, or a bank/color side channel.
- Downstream consumer:
  Profile-scoped hardware behavior evidence and, only after every activation
  gate passes, bounded candidate-cost calibration.
- User-level driver / named pipeline:
  ``wafer_worker_memory_contention_characterization_driver.py`` launches the
  active-Tile DDR family through the normal 16-Tile wafer-run lifecycle and is
  a fail-closed host gate for genuine unrepresentable boundaries.
- Explicit non-goals:
  Do not infer fairness from final completion, infer DDR/SPM banks from
  offsets, use host elapsed time as device cost, duplicate existing wait/
  subset/conflict-equivalence probes, or emit unsafe cross-worker hazards.
- Completion gate:
  Every active-Tile DDR claim resolves to an executable case and every
  neighboring claim retained here resolves to delegated board evidence or a
  genuine typed boundary.  Executable DDR cases require all-Tile status,
  exact result/guard/count/canary checks, and device-cycle observations.
"""

from __future__ import annotations

import dataclasses
import enum
from collections import Counter, defaultdict
from collections.abc import Iterable


TILE_COUNT = 16
WORKER_COUNT = 3
MINIMUM_REPEATS = 3
SPM_CANDIDATE_OFFSET = 8192
SPM_CONTROL_OFFSET = 4352
SPM_SAFE_TRANSLATION = 0x20000


class Domain(str, enum.Enum):
    WORKER_PLACEMENT = "worker-placement-arbitration"
    DDR_ACTIVE_TILE = "ddr-active-tile-contention"
    SPM_CONFLICT = "spm-conflict-pilot"
    DDR_BANK_COLORING = "ddr-bank-coloring"


class Disposition(str, enum.Enum):
    BOARD_EXECUTABLE = "board-executable"
    DELEGATED_BOARD_OBSERVED = "delegated-board-observed"
    DELEGATED_PENDING = "delegated-pending"
    STATIC_NEGATIVE = "static-negative"
    BLOCKED = "blocked"


class Layer(str, enum.Enum):
    CALIBRATION = "calibration"
    HELD_OUT = "held-out"
    CONTROL = "control"
    BOUNDARY = "boundary"


class Engine(str, enum.Enum):
    CT = "ct"
    NE = "ne"
    RDMA = "rdma"
    WDMA = "wdma"
    TDMA = "tdma"


class DDRDirection(str, enum.Enum):
    RDMA = "rdma-only"
    WDMA = "wdma-only"
    BIDIRECTIONAL = "rdma-wdma"


class MeasurementScope(str, enum.Enum):
    PER_WORKER = "per-worker-device-cycles"
    PER_TILE = "per-tile-device-cycles-and-complete-Tile-domain-max"
    PAIRED_SPM = "paired-target-window-device-cycles"


@dataclasses.dataclass(frozen=True)
class CorrectnessGate:
    exact_full_output: bool
    prefix_guard: bool
    suffix_guard: bool
    instruction_count: bool
    request_echo: bool
    status_scope: str
    local_drain_completion: bool
    cleanup: bool
    inactive_participant_canary: bool = False

    def validate(self) -> None:
        if not all(
            (
                self.exact_full_output,
                self.prefix_guard,
                self.suffix_guard,
                self.instruction_count,
                self.request_echo,
                self.local_drain_completion,
                self.cleanup,
            )
        ):
            raise RuntimeError("positive case has a weak correctness gate")
        if self.status_scope not in {
            "tile-zero-record",
            "all-tile-status",
        }:
            raise RuntimeError("positive case has an unknown status scope")


@dataclasses.dataclass(frozen=True)
class MeasurementGate:
    scope: MeasurementScope
    minimum_repeats: int
    required_metrics: tuple[str, ...]
    counterbalanced_order: bool
    calibration_and_held_out: bool
    requires_stable_read: bool = True
    permits_host_elapsed: bool = False
    promotion_rule: str = ""

    def validate(self) -> None:
        if (
            self.minimum_repeats < MINIMUM_REPEATS
            or not self.required_metrics
            or not self.requires_stable_read
            or self.permits_host_elapsed
            or not self.promotion_rule
        ):
            raise RuntimeError("measurement activation gate is incomplete")


@dataclasses.dataclass(frozen=True)
class Stream:
    stream_id: str
    tile_id: int
    worker: int
    engine: Engine
    payload_bytes: int
    expected_instructions: int
    stride_bytes: int = 0
    role: str = "work"

    def validate(self, participants: tuple[int, ...]) -> None:
        if (
            not self.stream_id
            or self.tile_id not in participants
            or self.worker not in range(WORKER_COUNT)
            or self.payload_bytes <= 0
            or self.expected_instructions <= 0
            or self.stride_bytes < 0
        ):
            raise RuntimeError(f"invalid stream contract {self.stream_id!r}")
        if self.stride_bytes and self.stride_bytes <= self.payload_bytes // 16:
            raise RuntimeError(
                f"{self.stream_id}: held-out stride does not leave holes"
            )


@dataclasses.dataclass(frozen=True)
class CharacterizationCase:
    key: str
    domain: Domain
    layer: Layer
    matched_group: str
    variant: str
    participant_tiles: tuple[int, ...]
    active_tiles: tuple[int, ...]
    streams: tuple[Stream, ...]
    payload_seed: int
    correctness: CorrectnessGate
    measurement: MeasurementGate
    activation_requirements: tuple[str, ...]
    disposition: Disposition = Disposition.BLOCKED
    ddr_direction: DDRDirection | None = None
    spm_relative_offset: int | None = None
    spm_translation: int | None = None
    spm_issue_order: str | None = None
    spm_schedule: str | None = None

    @property
    def inactive_tiles(self) -> tuple[int, ...]:
        active = set(self.active_tiles)
        return tuple(tile_id for tile_id in self.participant_tiles if tile_id not in active)

    @property
    def workers(self) -> tuple[int, ...]:
        return tuple(sorted({stream.worker for stream in self.streams}))

    @property
    def blocker(self) -> dict[str, str]:
        if self.domain == Domain.WORKER_PLACEMENT:
            return {
                "reason": (
                    "the current NCC record exposes per-worker instruction/"
                    "blocking counts but no per-worker completion or progress "
                    "cycle on one common start basis"
                ),
                "typed_gate": (
                    "performance request serialization is rejected until a "
                    "a current per-worker completion-cycle surface exists"
                ),
                "safe_alternative": (
                    "retain existing low-depth routing, wait-scope, subset, "
                    "and far-disjoint correctness evidence"
                ),
            }
        if self.domain == Domain.DDR_ACTIVE_TILE:
            return {
                "reason": (
                    "the current 16-Tile DDR carrier intentionally executes "
                    "one active tile_id per barrier phase and has no active-mask "
                    "contention mode or same-basis per-Tile duration record"
                ),
                "typed_gate": (
                    "active-Tile request serialization is rejected until an "
                    "all-Tile active-mask carrier and per-Tile device-cycle "
                    "record share one current protocol"
                ),
                "safe_alternative": (
                    "retain the existing single-active-Tile tile/offset "
                    "legality evidence"
                ),
            }
        raise RuntimeError(f"{self.key}: unknown blocked case domain")

    def expected_counts(self) -> dict[str, int]:
        counts: Counter[tuple[int, int, Engine]] = Counter()
        for stream in self.streams:
            counts[(stream.tile_id, stream.worker, stream.engine)] += (
                stream.expected_instructions
            )
        return {
            f"tile_id{tile_id}.worker{worker}.{engine.value}": value
            for (tile_id, worker, engine), value in sorted(
                counts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2].value,
                ),
            )
        }

    def validate(self) -> None:
        if (
            not self.key
            or not self.matched_group
            or not self.variant
            or self.disposition
            not in {Disposition.BLOCKED, Disposition.BOARD_EXECUTABLE}
            or not self.participant_tiles
            or tuple(sorted(set(self.participant_tiles)))
            != self.participant_tiles
            or tuple(sorted(set(self.active_tiles))) != self.active_tiles
            or not set(self.active_tiles).issubset(self.participant_tiles)
            or not self.streams
            or self.payload_seed <= 0
            or not self.activation_requirements
        ):
            raise RuntimeError(f"{self.key}: invalid positive case envelope")
        self.correctness.validate()
        self.measurement.validate()
        for stream in self.streams:
            stream.validate(self.participant_tiles)
        if len({stream.stream_id for stream in self.streams}) != len(
            self.streams
        ):
            raise RuntimeError(f"{self.key}: duplicate stream identity")
        stream_tiles = {stream.tile_id for stream in self.streams}
        if stream_tiles != set(self.active_tiles):
            raise RuntimeError(f"{self.key}: stream/active-Tile domain differs")
        if self.domain == Domain.DDR_ACTIVE_TILE:
            if (
                self.disposition != Disposition.BOARD_EXECUTABLE
                or
                self.participant_tiles != tuple(range(TILE_COUNT))
                or self.ddr_direction is None
                or not self.correctness.inactive_participant_canary
                or self.measurement.scope != MeasurementScope.PER_TILE
            ):
                raise RuntimeError(
                    f"{self.key}: invalid active-Tile DDR contract"
                )
        elif self.domain == Domain.WORKER_PLACEMENT:
            if (
                self.disposition != Disposition.BLOCKED
                or self.participant_tiles != (0,)
                or self.active_tiles != (0,)
                or self.measurement.scope != MeasurementScope.PER_WORKER
            ):
                raise RuntimeError(
                    f"{self.key}: invalid worker-placement contract"
                )
        elif self.domain == Domain.SPM_CONFLICT:
            if (
                self.participant_tiles != (0,)
                or self.active_tiles != (0,)
                or self.measurement.scope != MeasurementScope.PAIRED_SPM
                or self.spm_relative_offset
                not in {SPM_CANDIDATE_OFFSET, SPM_CONTROL_OFFSET}
                or self.spm_translation != SPM_SAFE_TRANSLATION
                or self.spm_issue_order not in {"a-b", "b-a"}
                or self.spm_schedule not in {"serial", "window"}
            ):
                raise RuntimeError(f"{self.key}: invalid SPM pilot contract")
        else:
            raise RuntimeError(
                f"{self.key}: positive DDR-bank case is forbidden"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "domain": self.domain.value,
            "disposition": self.disposition.value,
            "layer": self.layer.value,
            "matched_group": self.matched_group,
            "variant": self.variant,
            "participant_tiles": list(self.participant_tiles),
            "active_tiles": list(self.active_tiles),
            "inactive_tiles": list(self.inactive_tiles),
            "payload_seed": self.payload_seed,
            "streams": [
                {
                    **dataclasses.asdict(stream),
                    "engine": stream.engine.value,
                }
                for stream in self.streams
            ],
            "expected_counts": self.expected_counts(),
            "correctness": dataclasses.asdict(self.correctness),
            "measurement": {
                **dataclasses.asdict(self.measurement),
                "scope": self.measurement.scope.value,
            },
            "activation_requirements": list(self.activation_requirements),
            "ddr_direction": (
                self.ddr_direction.value
                if self.ddr_direction is not None
                else None
            ),
            "spm_relative_offset": self.spm_relative_offset,
            "spm_translation": self.spm_translation,
            "spm_issue_order": self.spm_issue_order,
            "spm_schedule": self.spm_schedule,
            "board_request_serializable": (
                self.disposition == Disposition.BOARD_EXECUTABLE
            ),
            "blocker": (
                self.blocker
                if self.disposition == Disposition.BLOCKED
                else None
            ),
            "board_adapter": (
                {
                    "driver": (
                        "test/Board/Support/"
                        "wafer_worker_memory_contention_"
                        "characterization_driver.py"
                    ),
                    "carrier": (
                        "test/Board/Inputs/"
                        "wafer_ddr_active_tile_contention_probe.c"
                    ),
                    "lifecycle": "cluster-x16/status",
                }
                if self.disposition == Disposition.BOARD_EXECUTABLE
                else None
            ),
        }


@dataclasses.dataclass(frozen=True)
class BlockedMatrixCase:
    key: str
    domain: Domain
    layer: Layer
    matched_group: str
    variant: str
    reason: str
    typed_gate: str
    safe_alternative: str
    activation_requirements: tuple[str, ...]
    spm_relative_offset: int
    spm_issue_order: str
    spm_schedule: str
    work_bytes: int
    disposition: Disposition = Disposition.BLOCKED

    def validate(self) -> None:
        if (
            not self.key
            or self.domain != Domain.SPM_CONFLICT
            or self.disposition != Disposition.BLOCKED
            or not self.matched_group
            or not self.variant
            or not self.reason
            or not self.typed_gate
            or not self.safe_alternative
            or not self.activation_requirements
            or self.spm_relative_offset
            not in {SPM_CANDIDATE_OFFSET, SPM_CONTROL_OFFSET}
            or self.spm_issue_order not in {"a-b", "b-a"}
            or self.spm_schedule not in {"serial", "window"}
            or self.work_bytes not in {4096, 16384}
        ):
            raise RuntimeError(f"{self.key}: invalid blocked matrix case")

    def as_dict(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "domain": self.domain.value,
            "layer": self.layer.value,
            "disposition": self.disposition.value,
            "board_request_serializable": False,
        }


@dataclasses.dataclass(frozen=True)
class DelegatedAsset:
    key: str
    domain: Domain
    disposition: Disposition
    source_file: str
    selector: str
    object_count: int
    evidence_scope: str
    claims: tuple[str, ...]

    def validate(self) -> None:
        if (
            self.disposition
            not in {
                Disposition.DELEGATED_BOARD_OBSERVED,
                Disposition.DELEGATED_PENDING,
            }
            or not self.source_file
            or not self.selector
            or self.object_count <= 0
            or not self.evidence_scope
            or not self.claims
        ):
            raise RuntimeError(f"{self.key}: invalid delegated evidence binding")

    def as_dict(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "domain": self.domain.value,
            "disposition": self.disposition.value,
        }


@dataclasses.dataclass(frozen=True)
class TypedBoundary:
    key: str
    domain: Domain
    disposition: Disposition
    reason: str
    typed_gate: str
    safe_alternative: str
    claims: tuple[str, ...]

    def validate(self) -> None:
        if (
            self.disposition
            not in {Disposition.STATIC_NEGATIVE, Disposition.BLOCKED}
            or not self.reason
            or not self.typed_gate
            or not self.safe_alternative
            or not self.claims
        ):
            raise RuntimeError(f"{self.key}: invalid typed boundary")

    def as_dict(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "domain": self.domain.value,
            "disposition": self.disposition.value,
        }


PROGRAM_LOCAL_CORRECTNESS = CorrectnessGate(
    exact_full_output=True,
    prefix_guard=True,
    suffix_guard=True,
    instruction_count=True,
    request_echo=True,
    status_scope="tile-zero-record",
    local_drain_completion=True,
    cleanup=True,
)
ALL_TILE_CORRECTNESS = dataclasses.replace(
    PROGRAM_LOCAL_CORRECTNESS,
    status_scope="all-tile-status",
    inactive_participant_canary=True,
)

DDR_MEASUREMENT = MeasurementGate(
    scope=MeasurementScope.PER_TILE,
    minimum_repeats=MINIMUM_REPEATS,
    required_metrics=(
        "per_tile_duration_cycles",
        "complete_tile_max_cycles",
        "per_tile_instruction_delta",
        "target_only_pmu_window",
    ),
    counterbalanced_order=True,
    calibration_and_held_out=True,
    promotion_rule=(
        "payload slope and strided held-out agree across active-Tile counts; "
        "all-Tile max and every active-Tile duration use one device basis"
    ),
)
def _stream(
    prefix: str,
    *,
    tile_id: int,
    worker: int,
    engine: Engine,
    payload_bytes: int,
    instructions: int,
    stride_bytes: int = 0,
    role: str = "work",
) -> Stream:
    return Stream(
        stream_id=f"{prefix}-r{tile_id}-w{worker}-{engine.value}",
        tile_id=tile_id,
        worker=worker,
        engine=engine,
        payload_bytes=payload_bytes,
        expected_instructions=instructions,
        stride_bytes=stride_bytes,
        role=role,
    )


ACTIVE_TILE_SETS = {
    1: (0,),
    2: (0, 8),
    4: (0, 4, 8, 12),
    8: tuple(range(0, TILE_COUNT, 2)),
    16: tuple(range(TILE_COUNT)),
}


def _ddr_active_tile_cases() -> tuple[CharacterizationCase, ...]:
    workloads = (
        ("4k-contiguous", Layer.CALIBRATION, 4096, 0),
        ("64k-contiguous", Layer.CALIBRATION, 65536, 0),
        ("64k-stride-heldout", Layer.HELD_OUT, 65536, 8192),
    )
    cases: list[CharacterizationCase] = []
    for direction_index, direction in enumerate(DDRDirection):
        engines = (
            (Engine.RDMA,)
            if direction == DDRDirection.RDMA
            else (
                (Engine.WDMA,)
                if direction == DDRDirection.WDMA
                else (Engine.RDMA, Engine.WDMA)
            )
        )
        for workload_index, (
            workload,
            layer,
            payload_bytes,
            stride_bytes,
        ) in enumerate(workloads):
            group = f"ddr-contention-{direction.value}-{workload}"
            for active_count, Tiles in ACTIVE_TILE_SETS.items():
                streams = tuple(
                    _stream(
                        group,
                        tile_id=tile_id,
                        worker=0,
                        engine=engine,
                        payload_bytes=payload_bytes,
                        instructions=1,
                        stride_bytes=stride_bytes,
                    )
                    for tile_id in Tiles
                    for engine in engines
                )
                cases.append(
                    CharacterizationCase(
                        key=f"{group}-active-{active_count}",
                        domain=Domain.DDR_ACTIVE_TILE,
                        layer=layer,
                        matched_group=group,
                        variant=f"active-{active_count}",
                        participant_tiles=tuple(range(TILE_COUNT)),
                        active_tiles=Tiles,
                        streams=streams,
                        payload_seed=(
                            0x9000
                            + direction_index * 0x400
                            + workload_index * 0x80
                        ),
                        correctness=ALL_TILE_CORRECTNESS,
                        measurement=DDR_MEASUREMENT,
                        activation_requirements=(
                            "all 16 Tiles enter one start and terminal lifecycle",
                            "inactive Tiles issue zero NCC instructions",
                            "per-Tile device duration and complete-Tile-domain max share basis",
                            "strided holes remain canary",
                        ),
                        ddr_direction=direction,
                        disposition=Disposition.BOARD_EXECUTABLE,
                    )
                )
    return tuple(cases)


DDR_ACTIVE_TILE_CASES = _ddr_active_tile_cases()
# Worker placement and sustained SPM conflict are real executable families
# owned by wafer_worker_placement_characterization_catalog.py and
# wafer_spm_sustained_conflict_catalog.py respectively.  Do not re-materialize
# their superseded synthetic blockers in this DDR-owned catalog.
CASES = DDR_ACTIVE_TILE_CASES
CASES_BY_KEY = {case.key: case for case in CASES}
EXECUTABLE_CASES = DDR_ACTIVE_TILE_CASES
DDR_ACTIVE_TILE_GROUPS = {
    group_key: tuple(
        case
        for case in DDR_ACTIVE_TILE_CASES
        if case.matched_group == group_key
    )
    for group_key in sorted(
        {case.matched_group for case in DDR_ACTIVE_TILE_CASES}
    )
}
DDR_ACTIVE_TILE_GROUP_KEYS = tuple(DDR_ACTIVE_TILE_GROUPS)


DELEGATED_ASSETS = (
    DelegatedAsset(
        key="worker-routing-and-low-depth-disjoint",
        domain=Domain.WORKER_PLACEMENT,
        disposition=Disposition.DELEGATED_BOARD_OBSERVED,
        source_file="test/Board/Support/wafer_board_ncc_execution_probe_runner.py",
        selector="WORKER_CASES",
        object_count=7,
        evidence_scope=(
            "CT w0/w1/w2 routing, all dual placements, and triple matching join"
        ),
        claims=("worker-routing", "worker-low-depth-disjoint-completion"),
    ),
    DelegatedAsset(
        key="cross-worker-ct-rdma-sustained-control",
        domain=Domain.WORKER_PLACEMENT,
        disposition=Disposition.DELEGATED_BOARD_OBSERVED,
        source_file="test/Board/Support/wafer_memory_descriptor_calibration_catalog.py",
        selector="CROSS_WORKER_PARALLEL_PAIR_CASES",
        object_count=2,
        evidence_scope="16KiB CT/RDMA workers 0/1 serial/window exact control",
        claims=("worker-cross-engine-correctness",),
    ),
    DelegatedAsset(
        key="worker-wait-scope-exclusion",
        domain=Domain.WORKER_PLACEMENT,
        disposition=Disposition.DELEGATED_PENDING,
        source_file="test/Board/Support/wafer_board_ncc_execution_probe_runner.py",
        selector="WORKER_WAIT_SCOPE_CASES",
        object_count=18,
        evidence_scope=(
            "NE/RDMA target worker 0/1/2 × default/byworker/local fence"
        ),
        claims=("worker-default-local-wait-scope",),
    ),
    DelegatedAsset(
        key="worker-subset-join-exclusion",
        domain=Domain.WORKER_PLACEMENT,
        disposition=Disposition.DELEGATED_PENDING,
        source_file="test/Board/Support/wafer_board_ncc_execution_probe_runner.py",
        selector="WORKER_SUBSET_SCOPE_CASES",
        object_count=12,
        evidence_scope="NE/RDMA target 0/1/2 × include/exclude target",
        claims=("worker-subset-mask-exclusion",),
    ),
    DelegatedAsset(
        key="ddr-single-active-tile-offset",
        domain=Domain.DDR_ACTIVE_TILE,
        disposition=Disposition.DELEGATED_BOARD_OBSERVED,
        source_file="test/Board/Support/wafer_board_ddr_tile_offset_probe_runner.py",
        selector="matrix_cases",
        object_count=896,
        evidence_scope=(
            "one active tile_id per barrier phase across 16 tiles, two "
            "allocations, 14 offsets, RDMA/WDMA"
        ),
        claims=("ddr-single-active-address-legality",),
    ),
    DelegatedAsset(
        key="spm-sustained-ct-rdma-far-disjoint-control",
        domain=Domain.SPM_CONFLICT,
        disposition=Disposition.DELEGATED_BOARD_OBSERVED,
        source_file="test/Board/Support/wafer_memory_descriptor_calibration_catalog.py",
        selector="SUSTAINED_PARALLEL_PAIR_CASES",
        object_count=20,
        evidence_scope=(
            "all ten engine-pair serial/window sustained controls; the "
            "CT/RDMA pair is the far-disjoint provenance baseline"
        ),
        claims=("spm-sustained-far-disjoint-control",),
    ),
    DelegatedAsset(
        key="spm-conflict-equivalence-program-local",
        domain=Domain.SPM_CONFLICT,
        disposition=Disposition.DELEGATED_PENDING,
        source_file="test/Board/Support/wafer_memory_descriptor_calibration_catalog.py",
        selector="CONFLICT_EQUIVALENCE_PAIRS",
        object_count=48,
        evidence_scope=(
            "same-invocation 256/512B reciprocal-order/base-translation pairs"
        ),
        claims=("spm-conflict-equivalence",),
    ),
    DelegatedAsset(
        key="spm-conflict-equivalence-cross-tile",
        domain=Domain.SPM_CONFLICT,
        disposition=Disposition.DELEGATED_PENDING,
        source_file=(
            "test/Board/Support/wafer_board_spm_cross_tile_conflict_probe_runner.py"
        ),
        selector="conflict_pairs",
        object_count=48,
        evidence_scope="counterbalanced single-active-Tile mapped execution",
        claims=("spm-conflict-equivalence-cross-tile",),
    ),
    DelegatedAsset(
        key="ddr-conflict-equivalence-program-local",
        domain=Domain.DDR_BANK_COLORING,
        disposition=Disposition.DELEGATED_PENDING,
        source_file="test/Board/Support/wafer_cache_coherence_calibration_catalog.py",
        selector="PENDING_DDR_CONFLICT_CASES",
        object_count=108,
        evidence_scope=(
            "program-local same-lifetime serial/window/order/workload controls"
        ),
        claims=("ddr-conflict-equivalence",),
    ),
    DelegatedAsset(
        key="ddr-conflict-equivalence-cross-tile",
        domain=Domain.DDR_BANK_COLORING,
        disposition=Disposition.DELEGATED_PENDING,
        source_file="test/Board/Support/wafer_board_ddr_tile_offset_probe_runner.py",
        selector="conflict_equivalence_cases",
        object_count=384,
        evidence_scope=(
            "RDMA/RDMA same-allocation cross-Tile held-out coordinates"
        ),
        claims=("ddr-conflict-equivalence-cross-tile",),
    ),
)
DELEGATED_BY_KEY = {asset.key: asset for asset in DELEGATED_ASSETS}


TYPED_BOUNDARIES = (
    TypedBoundary(
        key="worker-id-out-of-range",
        domain=Domain.WORKER_PLACEMENT,
        disposition=Disposition.STATIC_NEGATIVE,
        reason="worker ids outside 0..2 alias in lower-level firmware",
        typed_gate="Lane validation rejects worker not in range(3)",
        safe_alternative="select an explicit worker 0, 1, or 2",
        claims=("worker-id-domain",),
    ),
    TypedBoundary(
        key="cross-worker-unordered-same-address",
        domain=Domain.WORKER_PLACEMENT,
        disposition=Disposition.STATIC_NEGATIVE,
        reason=(
            "no owner-backed cross-worker same-address ordering contract exists"
        ),
        typed_gate=(
            "effect/range verifier rejects positive unordered cross-worker alias"
        ),
        safe_alternative=(
            "use disjoint ranges or an explicit producer completion and consumer"
        ),
        claims=("worker-cross-address-hazard",),
    ),
    TypedBoundary(
        key="remote-spm-conflict",
        domain=Domain.SPM_CONFLICT,
        disposition=Disposition.BLOCKED,
        reason=(
            "owned local-SPM aliases do not expose a safe remote-SPM range or "
            "remote contention counter"
        ),
        typed_gate="no remote-SPM request object is serializable",
        safe_alternative="run local-SPM cross-Tile held-out sequentially",
        claims=("spm-remote-concurrent-conflict",),
    ),
    TypedBoundary(
        key="spm-unowned-or-overlapping-guarded-range",
        domain=Domain.SPM_CONFLICT,
        disposition=Disposition.STATIC_NEGATIVE,
        reason="the pilot requires two disjoint guarded owned half-open ranges",
        typed_gate="range validator rejects overlap/reservation escape pre-board",
        safe_alternative=(
            "use planner-proven translated allocations and retain both guards"
        ),
        claims=("spm-owned-range-safety",),
    ),
    TypedBoundary(
        key="ddr-bank-owner-mapping-absent",
        domain=Domain.DDR_BANK_COLORING,
        disposition=Disposition.BLOCKED,
        reason=(
            "runtime exposes allocation addresses but no owner-backed "
            "address-to-bank/controller class"
        ),
        typed_gate=(
            "DDR bank/color request construction is non-executable without "
            "a current mapping provider"
        ),
        safe_alternative="retain ordinary allocator placement",
        claims=("ddr-bank-coloring", "ddr-controller-channel-hop"),
    ),
    TypedBoundary(
        key="ddr-bank-specific-counter-absent",
        domain=Domain.DDR_BANK_COLORING,
        disposition=Disposition.BLOCKED,
        reason=(
            "current PMU has no bank-specific counter and offset directions "
            "flip with allocation base/tile"
        ),
        typed_gate=(
            "measurement activation rejects generic RDMA/WDMA timing as a "
            "physical bank label"
        ),
        safe_alternative=(
            "retain conflict-equivalence observations without bank naming"
        ),
        claims=("ddr-bank-counter",),
    ),
    TypedBoundary(
        key="ddr-wdma-cross-allocation-heldout-missing",
        domain=Domain.DDR_BANK_COLORING,
        disposition=Disposition.BLOCKED,
        reason=(
            "program-local ABI owns one host-visible output allocation and cannot "
            "form the required WDMA cross-allocation exact oracle"
        ),
        typed_gate=(
            "cross-allocation promotion rejects an absent independent output owner"
        ),
        safe_alternative=(
            "limit equivalence evidence to explicitly covered axes"
        ),
        claims=("ddr-wdma-cross-allocation",),
    ),
)
BOUNDARIES_BY_KEY = {boundary.key: boundary for boundary in TYPED_BOUNDARIES}


CLAIM_COVERAGE: dict[str, tuple[str, ...]] = {
    "worker-routing": ("worker-routing-and-low-depth-disjoint",),
    "worker-low-depth-disjoint-completion": (
        "worker-routing-and-low-depth-disjoint",
    ),
    "worker-cross-engine-correctness": (
        "cross-worker-ct-rdma-sustained-control",
    ),
    "worker-default-local-wait-scope": ("worker-wait-scope-exclusion",),
    "worker-subset-mask-exclusion": ("worker-subset-join-exclusion",),
    "worker-id-domain": ("worker-id-out-of-range",),
    "worker-cross-address-hazard": (
        "cross-worker-unordered-same-address",
    ),
    "ddr-single-active-address-legality": (
        "ddr-single-active-tile-offset",
    ),
    "ddr-active-tile-contention": tuple(
        case.key for case in DDR_ACTIVE_TILE_CASES
    ),
    "spm-sustained-far-disjoint-control": (
        "spm-sustained-ct-rdma-far-disjoint-control",
    ),
    "spm-conflict-equivalence": ("spm-conflict-equivalence-program-local",),
    "spm-conflict-equivalence-cross-tile": (
        "spm-conflict-equivalence-cross-tile",
    ),
    "spm-remote-concurrent-conflict": ("remote-spm-conflict",),
    "spm-owned-range-safety": (
        "spm-unowned-or-overlapping-guarded-range",
    ),
    "ddr-conflict-equivalence": ("ddr-conflict-equivalence-program-local",),
    "ddr-conflict-equivalence-cross-tile": (
        "ddr-conflict-equivalence-cross-tile",
    ),
    "ddr-bank-coloring": ("ddr-bank-owner-mapping-absent",),
    "ddr-controller-channel-hop": ("ddr-bank-owner-mapping-absent",),
    "ddr-bank-counter": ("ddr-bank-specific-counter-absent",),
    "ddr-wdma-cross-allocation": (
        "ddr-wdma-cross-allocation-heldout-missing",
    ),
}


def matched_groups(
    cases: Iterable[CharacterizationCase] = CASES,
) -> dict[str, tuple[CharacterizationCase, ...]]:
    grouped: defaultdict[str, list[CharacterizationCase]] = defaultdict(list)
    for case in cases:
        grouped[case.matched_group].append(case)
    return {
        key: tuple(sorted(group, key=lambda case: case.key))
        for key, group in sorted(grouped.items())
    }


def validate_catalog() -> None:
    if (
        len(CASES_BY_KEY) != len(CASES)
        or len(DELEGATED_BY_KEY) != len(DELEGATED_ASSETS)
        or len(BOUNDARIES_BY_KEY) != len(TYPED_BOUNDARIES)
    ):
        raise RuntimeError("catalog object keys are not unique")
    all_keys = (
        set(CASES_BY_KEY)
        | set(DELEGATED_BY_KEY)
        | set(BOUNDARIES_BY_KEY)
    )
    if len(all_keys) != (
        len(CASES)
        + len(DELEGATED_ASSETS)
        + len(TYPED_BOUNDARIES)
    ):
        raise RuntimeError("catalog/delegated/boundary namespaces overlap")
    for case in CASES:
        case.validate()
    for asset in DELEGATED_ASSETS:
        asset.validate()
    for boundary in TYPED_BOUNDARIES:
        boundary.validate()

    groups = matched_groups()
    for group_name, group in groups.items():
        domain = {case.domain for case in group}
        seeds = {case.payload_seed for case in group}
        if len(domain) != 1 or len(seeds) != 1:
            raise RuntimeError(f"{group_name}: malformed matched group")
        if next(iter(domain)) == Domain.DDR_ACTIVE_TILE:
            if {len(case.active_tiles) for case in group} != {
                1,
                2,
                4,
                8,
                16,
            }:
                raise RuntimeError(f"{group_name}: active-Tile sweep incomplete")
    if (
        len(DDR_ACTIVE_TILE_CASES) != 45
    ):
        raise RuntimeError("matrix cardinality changed")
    if any(case.domain == Domain.DDR_BANK_COLORING for case in CASES):
        raise RuntimeError("DDR bank/coloring must have no executable case")
    if {
        len(case.active_tiles) for case in DDR_ACTIVE_TILE_CASES
    } != {1, 2, 4, 8, 16}:
        raise RuntimeError("DDR active-Tile axis is incomplete")
    if {
        case.ddr_direction for case in DDR_ACTIVE_TILE_CASES
    } != set(DDRDirection):
        raise RuntimeError("DDR direction axis is incomplete")
    if {
        (case.streams[0].payload_bytes, case.streams[0].stride_bytes)
        for case in DDR_ACTIVE_TILE_CASES
    } != {(4096, 0), (65536, 0), (65536, 8192)}:
        raise RuntimeError("DDR workload/held-out axis is incomplete")
    if (
        len(DDR_ACTIVE_TILE_GROUPS) != 9
        or any(
            len(group) != len(ACTIVE_TILE_SETS)
            or {len(case.active_tiles) for case in group}
            != set(ACTIVE_TILE_SETS)
            for group in DDR_ACTIVE_TILE_GROUPS.values()
        )
    ):
        raise RuntimeError("DDR active-Tile matched groups are incomplete")

    objects = {
        **CASES_BY_KEY,
        **DELEGATED_BY_KEY,
        **BOUNDARIES_BY_KEY,
    }
    if not CLAIM_COVERAGE or any(
        not bindings
        or any(binding not in objects for binding in bindings)
        for bindings in CLAIM_COVERAGE.values()
    ):
        raise RuntimeError("unresolved claim lacks a concrete catalog object")
    referenced = Counter(
        binding
        for bindings in CLAIM_COVERAGE.values()
        for binding in bindings
    )
    allowed_shared = {
        "worker-routing-and-low-depth-disjoint",
        "ddr-bank-owner-mapping-absent",
    }
    if any(
        count > 1 and key not in allowed_shared
        for key, count in referenced.items()
    ):
        raise RuntimeError("claim coverage duplicates a non-shared object")
    if set(objects) - set(referenced):
        raise RuntimeError(
            "catalog contains objects that no unresolved claim consumes: "
            + repr(sorted(set(objects) - set(referenced)))
        )


validate_catalog()
