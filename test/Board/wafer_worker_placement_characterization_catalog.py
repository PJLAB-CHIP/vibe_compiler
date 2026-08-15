#!/usr/bin/env python3
"""Typed real-board worker placement and bounded-progress cases.

Pipeline position:
- Upstream IR / input:
  Qualified rank-one package ABI, explicit NCC engine/worker routing, owned
  DDR/SPM ranges, matching-worker completion, and stable PMU readback.
- Current stage responsibility:
  Characterize fixed-total-work placement, low/high outstanding response, and
  bounded observer progress under a disjoint worker backlog.  Every
  executable row has an exact guarded result oracle.  Placement/outstanding
  rows retain raw per-issue CONTROL/IB_COUNTER samples; progress rows use a
  tight submit burst followed by a target-only device observation window.
- Output IR / files:
  Test-only request contracts and profile-scoped raw observations.  No case
  name, inferred arbiter identity, or measured number enters compiler IR.
- Downstream consumer:
  Hardware calibration evidence and later profile-scoped worker-placement
  cost selection after independent held-out qualification.
- User-level driver / named pipeline:
  The normal wafer-compile package path, a test-only linked device adapter,
  and wafer-run's rank-one board lifecycle.
- Explicit non-goals:
  Do not infer an exact per-worker completion timestamp, priority direction, a
  physical arbiter algorithm, cross-worker address ordering, or a fixed cycle
  constant.
- Completion gate:
  All 44 cases serialize to the versioned device protocol; each activation
  group contains its matched controls; board execution checks request echo,
  exact full results, both guards, routing/counts, stable device counters,
  participant completion through rotating polling and matching joins,
  lifecycle, and normal cleanup.
"""

from __future__ import annotations

import dataclasses
import enum


WORKERS = 3
REPEATS = 3
PLACEMENT_ISSUES = 6
BACKLOG_ISSUES = 4
OUTSTANDING_LOW_ISSUES = 2
OUTSTANDING_HIGH_ISSUES = 6
PLACEMENT_BYTES = 16384
RDMA_BACKLOG_BYTES = 65536
NE_RESULT_BYTES = 16384
SENTINEL_BYTES = 4096
LONG_ISSUE_BYTES = 65536


class Engine(enum.IntEnum):
    CT = 0
    NE = 1
    RDMA = 2


class Kind(enum.IntEnum):
    PLACEMENT = 0
    BACKLOG_ONLY = 1
    SENTINEL_ONLY = 2
    CONCURRENT = 3
    OUTSTANDING = 4


class Layer(str, enum.Enum):
    CALIBRATION = "calibration"
    HELD_OUT = "held-out"
    CONTROL = "control"


@dataclasses.dataclass(frozen=True)
class WorkerCase:
    key: str
    group: str
    kind: Kind
    engine: Engine
    layer: Layer
    worker_issues: tuple[int, int, int]
    target_worker: int
    observer_worker: int
    primary_bytes: int
    sentinel_bytes: int
    claims: tuple[str, ...]

    @property
    def issue_count(self) -> int:
        return sum(self.worker_issues)

    @property
    def worker_mask(self) -> int:
        return sum(
            1 << worker
            for worker, count in enumerate(self.worker_issues)
            if count
        )

    @property
    def disposition(self) -> str:
        return "board-executable"

    def validate(self) -> None:
        if (
            not self.key
            or not self.group
            or len(self.worker_issues) != WORKERS
            or any(count < 0 for count in self.worker_issues)
            or self.target_worker not in range(WORKERS)
            or self.observer_worker not in range(WORKERS)
            or self.target_worker == self.observer_worker
            or not self.claims
        ):
            raise RuntimeError(f"{self.key}: malformed worker case")
        if self.kind == Kind.PLACEMENT:
            nonzero = sorted(
                count for count in self.worker_issues if count
            )
            if (
                self.engine not in {Engine.CT, Engine.RDMA}
                or self.issue_count != PLACEMENT_ISSUES
                or nonzero not in ([6], [3, 3], [2, 2, 2])
                or self.primary_bytes != PLACEMENT_BYTES
                or self.sentinel_bytes != 0
            ):
                raise RuntimeError(
                    f"{self.key}: placement work is not matched"
                )
            return
        if self.kind == Kind.OUTSTANDING:
            if (
                self.engine not in {Engine.CT, Engine.RDMA}
                or self.primary_bytes != LONG_ISSUE_BYTES
                or self.sentinel_bytes != 0
                or self.worker_issues[self.target_worker]
                not in {
                    OUTSTANDING_LOW_ISSUES,
                    OUTSTANDING_HIGH_ISSUES,
                }
                or any(
                    count
                    for worker, count in enumerate(self.worker_issues)
                    if worker != self.target_worker
                )
            ):
                raise RuntimeError(
                    f"{self.key}: malformed outstanding-response workload"
                )
            return
        expected_primary = (
            NE_RESULT_BYTES
            if self.engine == Engine.NE
            else RDMA_BACKLOG_BYTES
        )
        if (
            self.engine not in {Engine.NE, Engine.RDMA}
            or self.primary_bytes != expected_primary
            or self.sentinel_bytes != SENTINEL_BYTES
        ):
            raise RuntimeError(f"{self.key}: malformed progress workload")
        target_count = self.worker_issues[self.target_worker]
        observer_count = self.worker_issues[self.observer_worker]
        expected = {
            Kind.BACKLOG_ONLY: (BACKLOG_ISSUES, 0),
            Kind.SENTINEL_ONLY: (0, 1),
            Kind.CONCURRENT: (BACKLOG_ISSUES, 1),
        }[self.kind]
        if (
            (target_count, observer_count) != expected
            or any(
                count
                for worker, count in enumerate(self.worker_issues)
                if worker not in {self.target_worker, self.observer_worker}
            )
        ):
            raise RuntimeError(f"{self.key}: progress participants differ")

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "group": self.group,
            "kind": self.kind.name.lower().replace("_", "-"),
            "engine": self.engine.name.lower(),
            "layer": self.layer.value,
            "worker_issues": list(self.worker_issues),
            "worker_mask": self.worker_mask,
            "target_worker": self.target_worker,
            "observer_worker": self.observer_worker,
            "primary_bytes": self.primary_bytes,
            "sentinel_bytes": self.sentinel_bytes,
            "repeat_count": REPEATS,
            "claims": list(self.claims),
            "disposition": self.disposition,
            "adapter": {
                "carrier": (
                    "test/Board/Inputs/wafer_worker_placement_probe.c"
                ),
                "driver": (
                    "test/Board/"
                    "wafer_board_worker_placement_characterization_test.py"
                ),
                "lifecycle": "rank-one/package/wafer-run/status/cleanup",
            },
            "oracle": [
                "full exact output",
                "prefix/suffix guards",
                "request/routing/instruction-count echo",
                "stable PMU and worker controls",
                (
                    "tight progress submission, or per-issue "
                    "CONTROL/IB_COUNTER for placement/outstanding"
                ),
                "rotating first-observed task-done polling",
                "matching participant joins and terminal completion",
            ],
        }


@dataclasses.dataclass(frozen=True)
class TypedBoundary:
    key: str
    reason: str
    typed_gate: str
    safe_alternative: str

    @property
    def disposition(self) -> str:
        return "fail-closed"

    def validate(self) -> None:
        if not all(
            (self.key, self.reason, self.typed_gate, self.safe_alternative)
        ):
            raise RuntimeError("worker boundary is incomplete")

    def as_dict(self) -> dict[str, str]:
        return {
            **dataclasses.asdict(self),
            "disposition": self.disposition,
        }


def _placement_cases() -> tuple[WorkerCase, ...]:
    placements = {
        "w0": (6, 0, 0),
        "w1": (0, 6, 0),
        "w2": (0, 0, 6),
        "w01": (3, 3, 0),
        "w02": (3, 0, 3),
        "w12": (0, 3, 3),
        "w012": (2, 2, 2),
    }
    cases: list[WorkerCase] = []
    for engine, layer in (
        (Engine.CT, Layer.CALIBRATION),
        (Engine.RDMA, Layer.HELD_OUT),
    ):
        group = f"worker-placement-{engine.name.lower()}"
        for placement, counts in placements.items():
            participants = [
                worker for worker, count in enumerate(counts) if count
            ]
            target = participants[0]
            observer = next(
                worker for worker in range(WORKERS) if worker != target
            )
            cases.append(
                WorkerCase(
                    key=f"{group}-{placement}",
                    group=group,
                    kind=Kind.PLACEMENT,
                    engine=engine,
                    layer=layer,
                    worker_issues=counts,
                    target_worker=target,
                    observer_worker=observer,
                    primary_bytes=PLACEMENT_BYTES,
                    sentinel_bytes=0,
                    claims=(
                        "fixed-total-work-placement",
                        (
                            "same-worker-control"
                            if len(participants) == 1
                            else "cross-worker-placement"
                        ),
                        "per-worker-instruction-and-blocking-basis",
                    ),
                )
            )
    return tuple(cases)


def _progress_cases() -> tuple[WorkerCase, ...]:
    cases: list[WorkerCase] = []
    for engine, layer in (
        (Engine.NE, Layer.CALIBRATION),
        (Engine.RDMA, Layer.HELD_OUT),
    ):
        primary_bytes = (
            NE_RESULT_BYTES
            if engine == Engine.NE
            else RDMA_BACKLOG_BYTES
        )
        for target in range(WORKERS):
            observer = (target + 1) % WORKERS
            group = (
                f"worker-progress-{engine.name.lower()}-"
                f"target-w{target}-observer-w{observer}"
            )
            for kind, target_count, observer_count, row_layer in (
                (Kind.BACKLOG_ONLY, BACKLOG_ISSUES, 0, Layer.CONTROL),
                (Kind.SENTINEL_ONLY, 0, 1, Layer.CONTROL),
                (Kind.CONCURRENT, BACKLOG_ISSUES, 1, layer),
            ):
                counts = [0, 0, 0]
                counts[target] = target_count
                counts[observer] = observer_count
                cases.append(
                    WorkerCase(
                        key=(
                            f"{group}-"
                            f"{kind.name.lower().replace('_', '-')}"
                        ),
                        group=group,
                        kind=kind,
                        engine=engine,
                        layer=row_layer,
                        worker_issues=tuple(counts),
                        target_worker=target,
                        observer_worker=observer,
                        primary_bytes=primary_bytes,
                        sentinel_bytes=SENTINEL_BYTES,
                        claims=(
                            "bounded-observer-progress",
                            "backlog-and-sentinel-matched-controls",
                            "tight-submit-without-per-issue-control-reads",
                            "target-pending-before-safety-drain-classification",
                        ),
                    )
                )
    return tuple(cases)


def _outstanding_cases() -> tuple[WorkerCase, ...]:
    cases: list[WorkerCase] = []
    for engine, layer in (
        (Engine.CT, Layer.CALIBRATION),
        (Engine.RDMA, Layer.HELD_OUT),
    ):
        for worker in range(WORKERS):
            observer = (worker + 1) % WORKERS
            group = (
                f"worker-outstanding-{engine.name.lower()}-w{worker}"
            )
            for level, issues in (
                ("low", OUTSTANDING_LOW_ISSUES),
                ("high", OUTSTANDING_HIGH_ISSUES),
            ):
                counts = [0, 0, 0]
                counts[worker] = issues
                cases.append(
                    WorkerCase(
                        key=f"{group}-{level}",
                        group=group,
                        kind=Kind.OUTSTANDING,
                        engine=engine,
                        layer=layer,
                        worker_issues=tuple(counts),
                        target_worker=worker,
                        observer_worker=observer,
                        primary_bytes=LONG_ISSUE_BYTES,
                        sentinel_bytes=0,
                        claims=(
                            "matched-low-high-outstanding-response",
                            "per-issue-control-and-ib-counter",
                            "same-worker-long-operation-control",
                        ),
                    )
                )
    return tuple(cases)


CASES = (
    _placement_cases()
    + _progress_cases()
    + _outstanding_cases()
)
CASES_BY_KEY = {case.key: case for case in CASES}
GROUPS = tuple(sorted({case.group for case in CASES}))
GROUP_CASES = {
    group: tuple(case for case in CASES if case.group == group)
    for group in GROUPS
}

BOUNDARIES = (
    TypedBoundary(
        key="worker-absolute-per-worker-completion-cycle",
        reason=(
            "the current PMU has a common global/engine cycle window and "
            "per-worker instruction/blocking counters, but no versioned "
            "per-worker completion timestamp on that same start basis"
        ),
        typed_gate=(
            "reject serialization of an absolute per-worker completion-cycle "
            "cost field"
        ),
        safe_alternative=(
            "run fixed-total placement with rotated issue order and rotating "
            "task-done polling; retain first-observed cycles, poll gaps, raw "
            "global cycles, per-worker blocking, and controls"
        ),
    ),
    TypedBoundary(
        key="worker-physical-arbiter-policy",
        reason=(
            "no owner-backed register or ABI identifies the physical arbiter "
            "or its round-robin/priority policy"
        ),
        typed_gate=(
            "reject any request that names an arbiter class or policy"
        ),
        safe_alternative=(
            "classify only bounded progress and measured imbalance for the "
            "observed profile"
        ),
    ),
)
BOUNDARIES_BY_KEY = {boundary.key: boundary for boundary in BOUNDARIES}


def validate_catalog() -> None:
    if len(CASES) != 44 or len(CASES_BY_KEY) != len(CASES):
        raise RuntimeError("worker catalog must contain 44 unique cases")
    for case in CASES:
        case.validate()
    for boundary in BOUNDARIES:
        boundary.validate()
    if len(GROUPS) != 14:
        raise RuntimeError("worker catalog must contain 14 matched groups")
    for group, cases in GROUP_CASES.items():
        if group.startswith("worker-placement-"):
            if len(cases) != 7:
                raise RuntimeError(f"{group}: incomplete placement matrix")
        elif group.startswith("worker-outstanding-"):
            if (
                len(cases) != 2
                or {case.issue_count for case in cases}
                != {
                    OUTSTANDING_LOW_ISSUES,
                    OUTSTANDING_HIGH_ISSUES,
                }
            ):
                raise RuntimeError(
                    f"{group}: incomplete outstanding pair"
                )
        else:
            if {case.kind for case in cases} != {
                Kind.BACKLOG_ONLY,
                Kind.SENTINEL_ONLY,
                Kind.CONCURRENT,
            }:
                raise RuntimeError(f"{group}: missing matched progress control")


validate_catalog()
