#!/usr/bin/env python3
"""Run the configured TX board calibration tests as one fail-fast session.

This driver deliberately owns only host-side orchestration.  Every hardware
operation remains inside an explicitly registered CTest, with that test's
bounded timeout and normal runtime cleanup.  The driver runs one CTest process
at a time, never retries a failure, and has no reset or power-control path.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence

BOARD_TEST_DIRECTORY = (
    pathlib.Path(__file__).resolve().parents[1] / "test" / "Board"
)
if str(BOARD_TEST_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BOARD_TEST_DIRECTORY))

import wafer_collective_traffic_behavior_catalog as collective_traffic_catalog
import wafer_engine_pipeline_characterization_catalog as engine_pipeline_catalog
import wafer_pending_hardware_calibration_inventory as pending_inventory
import wafer_spm_sustained_conflict_catalog as spm_sustained_catalog
import wafer_transport_pmu_calibration_catalog as transport_catalog
import wafer_worker_memory_contention_characterization_catalog as contention_catalog
import wafer_worker_placement_characterization_catalog as worker_placement_catalog


ARM_ENVIRONMENT_VARIABLE = "WAFER_EXECUTE_HARDWARE_TESTS"
SESSION_ENVIRONMENT_VARIABLE = "WAFER_CALIBRATION_SESSION_ID"
HEARTBEAT_CTEST = "wafer-board-single-op-add"
SUMMARY_SCHEMA_VERSION = 1
COMPILER_OPTIMIZATION_PAIRED_CASES = (
    "reciprocal-implementation",
    "f16-common-factor",
    "resident-fanout-share",
    "consumer-local-recompute",
    "long-steady-elementwise-add",
    "ready-order-movement-first",
    "gemm-aligned-physical-route",
    "gemm-tail-physical-route",
    "tree-all-reduce",
)
COLLECTIVE_CHARACTERIZATION_CASES = (
    "all-gather-direct-vs-ring-256b",
    "all-gather-direct-vs-ring-4096b",
    "all-gather-direct-vs-ring-65536b",
    "reduce-scatter-direct-vs-ring-256b",
    "reduce-scatter-direct-vs-ring-4096b",
    "reduce-scatter-direct-vs-ring-65536b",
    "all-reduce-ring-vs-tree-256b",
    "all-reduce-ring-vs-tree-4096b",
    "all-reduce-ring-vs-tree-65536b",
)
COLLECTIVE_TRAFFIC_BEHAVIOR_CASES = tuple(
    collective_traffic_catalog.CASE_KEYS
)
DTE_PENDING_CASE_SPECS = (
    (
        transport_catalog.FOUR_SOURCE_FANIN_MODE,
        transport_catalog.MODE_NAMES[
            transport_catalog.FOUR_SOURCE_FANIN_MODE
        ],
    ),
    *tuple(
        (case.mode, case.key)
        for case in transport_catalog.RAW_REMOTE_MULTICAST_CASES
    ),
)
DTE_PENDING_CASE_NAMES = tuple(
    case_name for _, case_name in DTE_PENDING_CASE_SPECS
)
DTE_SHUFFLE_CASE_NAMES = tuple(
    case_name
    for case_name in DTE_PENDING_CASE_NAMES
    if "dte-raw-shuffle-" in case_name
)
DTE_NON_SHUFFLE_CASE_NAMES = tuple(
    case_name
    for case_name in DTE_PENDING_CASE_NAMES
    if case_name not in DTE_SHUFFLE_CASE_NAMES
)
UNPOOL_PENDING_COLLISION_CASE_NAMES = (
    "unpool-index-f16-repeated-overlap-observed",
    "unpool-mask-f16-repeated-overlap-observed",
)
ENGINE_PIPELINE_BOARD_CELL_KEYS = tuple(
    cell.key
    for cell in engine_pipeline_catalog.ALL_CELLS
    if cell.disposition
    in {
        engine_pipeline_catalog.Disposition.BOARD_EXECUTABLE,
        engine_pipeline_catalog.Disposition.BOARD_EXECUTABLE_EXTERNAL,
    }
)
ENGINE_PIPELINE_BOARD_GROUP_KEYS = tuple(
    group.key
    for group in engine_pipeline_catalog.BOARD_ACTIVATION_GROUPS
)
ENGINE_PIPELINE_EXTERNAL_GROUP_KEYS = tuple(
    engine_pipeline_catalog.EXTERNAL_BOARD_GROUP_KEYS
)
SPM_SUSTAINED_BOARD_GROUP_KEYS = tuple(
    group.key for group in spm_sustained_catalog.GROUPS
)
DDR_ACTIVE_RANK_CASE_KEYS = tuple(
    case.key for case in contention_catalog.DDR_ACTIVE_RANK_CASES
)
DDR_ACTIVE_RANK_BOARD_GROUP_KEYS = tuple(
    contention_catalog.DDR_ACTIVE_RANK_GROUP_KEYS
)


class CalibrationRunnerError(RuntimeError):
    """A preflight or execution failure that must stop the board session."""


@dataclasses.dataclass(frozen=True)
class CalibrationStep:
    key: str
    batch: str
    ctest_name: str
    purpose: str


@dataclasses.dataclass(frozen=True)
class RegisteredTest:
    name: str
    command: tuple[str, ...]
    labels: frozenset[str]
    timeout_seconds: float | None
    resource_lock: str | None
    disabled: bool = False


CALIBRATION_STEPS = (
    CalibrationStep(
        "initial-profile-heartbeat",
        "preflight-profile-heartbeat",
        HEARTBEAT_CTEST,
        "qualify the configured rank-one runtime path and establish card health",
    ),
    CalibrationStep(
        "pmu-readonly",
        "preflight-profile-heartbeat",
        "wafer-board-ncc-pmu-readonly-probe",
        "qualify read-only PMU enable, scope, split-counter, and delta basis",
    ),
    CalibrationStep(
        "instruction-family-ct-capability",
        "rank-one-instruction",
        "wafer-board-instruction-family-ct-capability",
        "new CT Reduce/Pool/Unpool typed exact and bounded observations",
    ),
    CalibrationStep(
        "instruction-family-regression",
        "rank-one-instruction",
        "wafer-board-instruction-family-qualification-regression",
        "ArgMin, legacy Unpool-index, and Bilinear qualification regressions",
    ),
    CalibrationStep(
        "ct-vector",
        "rank-one-instruction",
        "wafer-board-ct-vector-calibration",
        "CT form, dtype, relation, logic, reduction, and layout matrix",
    ),
    CalibrationStep(
        "ct-convert",
        "rank-one-instruction",
        "wafer-board-ct-convert-calibration",
        "CT conversion routes including repeated stochastic observations",
    ),
    CalibrationStep(
        "datamove-core",
        "rank-one-datamove",
        "wafer-board-datamove-calibration",
        "core DataMove packet, layout, and physical-span cases",
    ),
    CalibrationStep(
        "datamove-extended",
        "rank-one-datamove",
        "wafer-board-datamove-extended-calibration",
        "concat, broadcast-like materialization, gather, and large-shape cases",
    ),
    CalibrationStep(
        "memory-engine-pair-new-offsets",
        "rank-one-memory",
        "wafer-board-memory-engine-pair-new-offsets",
        "new 4352/65536-byte SPM engine-pair serial/window controls",
    ),
    CalibrationStep(
        "cache-coherence",
        "rank-one-memory",
        "wafer-board-cache-coherence-calibration",
        "single-invocation DDR/cache visibility cases",
    ),
    CalibrationStep(
        "ne-calibration",
        "rank-one-ne",
        "wafer-board-ne-calibration",
        "NE dtype, numeric, broadcast, convolution, and layout cases",
    ),
    CalibrationStep(
        "ne-single-gemm",
        "rank-one-ne",
        "wafer-board-single-gemm",
        "rank-one production GEMM runtime path",
    ),
    CalibrationStep(
        "ne-mn-tiled-gemm",
        "rank-one-ne",
        "wafer-board-mn-tiled-gemm",
        "rank-one tiled GEMM runtime path",
    ),
    CalibrationStep(
        "spm-non-preferred-geometry",
        "rank-one-spm",
        "wafer-board-spm-non-preferred-geometry",
        "bounded non-preferred SPM base and length observations",
    ),
    CalibrationStep(
        "ncc-all-safe",
        "rank-one-ncc",
        "wafer-board-ncc-all-safe-observations",
        "safe queue, worker, engine, dependency, backlog, and overlap observations",
    ),
    CalibrationStep(
        "ncc-constructor-return",
        "rank-one-ncc",
        "wafer-board-ncc-constructor-return-address",
        "typed constructor non-null return-address observation",
    ),
    CalibrationStep(
        "ncc-default-wait-worker1",
        "rank-one-ncc",
        "wafer-board-ncc-default-wait-worker1-boundary",
        "default-wait boundary with worker-one work and final safety drain",
    ),
    CalibrationStep(
        "ncc-byworker-wait-worker1",
        "rank-one-ncc",
        "wafer-board-ncc-byworker1-completion-control",
        "matching by-worker completion control",
    ),
    CalibrationStep(
        "ncc-local-fence-worker1",
        "rank-one-ncc",
        "wafer-board-ncc-local-fence-worker1-boundary",
        "local-fence boundary with worker-one work and final safety drain",
    ),
    CalibrationStep(
        "ncc-ct-active-occupancy",
        "rank-one-ncc",
        "wafer-board-ncc-ct-active-occupancy",
        "tight long CT depth-plus-one activity and backpressure observation",
    ),
    CalibrationStep(
        "ncc-ne-active-occupancy",
        "rank-one-ncc",
        "wafer-board-ncc-ne-active-occupancy",
        "tight large NE depth-plus-one activity and backpressure observation",
    ),
    CalibrationStep(
        "ncc-rdma-active-occupancy",
        "rank-one-ncc",
        "wafer-board-ncc-rdma-active-occupancy",
        "tight long RDMA depth-plus-one activity and backpressure observation",
    ),
    CalibrationStep(
        "ncc-wdma-active-occupancy",
        "rank-one-ncc",
        "wafer-board-ncc-wdma-active-occupancy",
        "tight long WDMA depth-plus-one activity and backpressure observation",
    ),
    CalibrationStep(
        "ncc-tdma-active-occupancy",
        "rank-one-ncc",
        "wafer-board-ncc-tdma-active-occupancy",
        "tight long TDMA depth-plus-one activity and backpressure observation",
    ),
    *tuple(
        CalibrationStep(
            f"ncc-join-{mask}",
            "rank-one-ncc",
            f"wafer-board-ncc-join{mask}-unjoined-boundary",
            f"worker subset join mask {mask} with bounded unjoined observation",
        )
        for mask in ("001", "010", "100", "011", "101", "110")
    ),
    CalibrationStep(
        "runtime-kernel-grid-profile",
        "full-card-runtime",
        "wafer-board-kernel-grid-add-profile",
        (
            "16-rank Add correctness and final-artifact profiler campaign "
            "over the same production package"
        ),
    ),
    CalibrationStep(
        "full-card-barrier",
        "full-card-barrier",
        "wafer-board-full-card-barrier-probe",
        "cluster arrival and barrier completion",
    ),
    CalibrationStep(
        "direct-dte-collective",
        "full-card-dte",
        "wafer-board-cluster-direct-dte",
        "16-rank Direct DTE collective",
    ),
    CalibrationStep(
        "direct-dte-ncc",
        "full-card-dte",
        "wafer-board-dte-ncc-execution-probe",
        "ordered NCC producer/consumer and Direct DTE interactions",
    ),
    CalibrationStep(
        "full-card-sharded-gemm",
        "full-card-runtime",
        "wafer-board-k-sharded-gemm",
        "16-rank sharded GEMM production workload",
    ),
)

STRIDED_DEPENDENCY_CASE_NAMES = (
    *tuple(
        f"dependency-strided-{dimension}-{schedule}-observation"
        for dimension in ("1d", "2d", "3d")
        for schedule in ("serial", "window")
    ),
    *tuple(
        f"dependency-strided-{dimension}-{effect}-{schedule}-observation"
        for effect in ("war", "rar")
        for dimension in ("1d", "2d", "3d")
        for schedule in ("serial", "window")
    ),
    *tuple(
        "dependency-strided-"
        f"{dimension}-waw-{relation}-{schedule}-observation"
        for dimension in ("1d", "2d", "3d")
        for relation in ("exact", "partial", "adjacent")
        for schedule in ("serial", "window")
    ),
)

QUEUE_SATURATION_CASE_NAMES = (
    *tuple(
        f"queue-saturation-{engine}-{load}-depth{issue_limit}-tight-window"
        for engine in ("ct", "ne", "rdma", "wdma")
        for load in ("short", "sustained")
        for issue_limit in (5, 6, 7)
    ),
    *tuple(
        f"queue-saturation-tdma-{load}-depth{issue_limit}-tight-window"
        for load in ("short", "sustained")
        for issue_limit in (3, 4, 5)
    ),
)
WORKER_WAIT_SCOPE_CASE_NAMES = tuple(
    f"worker-wait-scope-{engine}-worker{worker}-{wait_kind}-tight-window"
    for worker in range(3)
    for engine in ("ne", "rdma")
    for wait_kind in ("default", "byworker", "local-fence")
)
WORKER_SUBSET_SCOPE_CASE_NAMES = tuple(
    f"worker-subset-{engine}-target{worker}-{membership}-tight-window"
    for worker in range(3)
    for engine in ("ne", "rdma")
    for membership in ("exclude", "include")
)
WORKER_PLACEMENT_CASE_KEYS = tuple(
    case.key for case in worker_placement_catalog.CASES
)
WORKER_PLACEMENT_BOARD_GROUP_KEYS = tuple(worker_placement_catalog.GROUPS)
ARGMIN_PENDING_DOMAIN_CASE_NAMES = (
    "peripheral-argmin-tie-f16-observed",
    "peripheral-argmin-nan-f16-observed",
)


EXPLICIT_ONLY_STEPS = (
    CalibrationStep(
        "m-sharded-replicated-gemm-profile",
        "full-card-compiler-search",
        "wafer-board-m-sharded-replicated-gemm-profile",
        (
            "16-rank FP16 M-sharded replicated-operand GEMM exact "
            "correctness and final-artifact profiler campaign"
        ),
    ),
    CalibrationStep(
        "runtime-kernel-grid",
        "full-card-runtime-smoke",
        "wafer-board-kernel-grid-add",
        (
            "standalone legacy 16-rank kernel-grid Add smoke; superseded by "
            "the default final-artifact profiler gate"
        ),
    ),
    CalibrationStep(
        "ct-vuvloop-semantics",
        "rank-one-instruction-focused",
        "wafer-board-ct-vuvloop-semantics",
        "contract-legal unit-64 VuVLoop semantics and post-case integrity",
    ),
    CalibrationStep(
        "memory-parallel-expanded",
        "rank-one-memory-focused",
        "wafer-board-memory-descriptor-parallel-expanded",
        "expanded sustained, cross-worker, and dependency pair observations",
    ),
    CalibrationStep(
        "memory-strided-dma-focused",
        "rank-one-memory-focused",
        "wafer-board-memory-descriptor-strided-dma-focused",
        "corrected compact-SPM 1D/2D/3D DMA descriptor observations",
    ),
    CalibrationStep(
        "ncc-mapped-spm-ne-depth4-local-wait",
        "rank-one-ncc-focused",
        "wafer-board-ncc-mapped-spm-ne-depth4-local-wait",
        "depth-four NCC-to-Kcore boundary with matching local completion",
    ),
    CalibrationStep(
        "ncc-mapped-spm-ne-depth4-no-local-wait",
        "rank-one-ncc-focused",
        "wafer-board-ncc-mapped-spm-ne-depth4-no-local-wait",
        "depth-four NCC-to-Kcore boundary without local completion",
    ),
    *tuple(
        CalibrationStep(
            f"ncc-{case_name}",
            "rank-one-ncc-strided-focused",
            f"wafer-board-ncc-{case_name}",
            f"corrected compact-SPM oracle for {case_name}",
        )
        for case_name in STRIDED_DEPENDENCY_CASE_NAMES
    ),
    CalibrationStep(
        "ddr-tile-offset",
        "full-card-memory-focused",
        "wafer-board-ddr-tile-offset-probe",
        "16-rank actual-allocation-base and relative-offset correctness",
    ),
    CalibrationStep(
        "ddr-sparse-high-offset",
        "rank-one-memory-focused",
        "wafer-board-ddr-sparse-high-offset-probe",
        "40-GiB compiler-managed workspace sparse relative-offset correctness",
    ),
    *tuple(
        CalibrationStep(
            f"compiler-optimization-{case_name}",
            "rank-one-compiler-optimization",
            f"wafer-board-compiler-optimization-{case_name}",
            (
                "same-source reserved-baseline versus production-winner "
                f"correctness and target-structure pair for {case_name}"
            ),
        )
        for case_name in COMPILER_OPTIMIZATION_PAIRED_CASES[:-1]
    ),
    CalibrationStep(
        f"compiler-optimization-{COMPILER_OPTIMIZATION_PAIRED_CASES[-1]}",
        "full-card-compiler-optimization",
        (
            "wafer-board-compiler-optimization-"
            f"{COMPILER_OPTIMIZATION_PAIRED_CASES[-1]}"
        ),
        (
            "same-source reserved-baseline versus production tree all-reduce "
            "correctness and final-target communication-work pair"
        ),
    ),
    *tuple(
        CalibrationStep(
            f"collective-characterization-{case_name}",
            "full-card-collective-characterization",
            f"wafer-board-collective-characterization-{case_name}",
            (
                "same-source forced collective alternatives with exact "
                f"all-rank output and pending board observations for {case_name}"
            ),
        )
        for case_name in COLLECTIVE_CHARACTERIZATION_CASES
    ),
    *tuple(
        CalibrationStep(
            f"collective-traffic-behavior-{case_name}",
            "full-card-pending-collective-traffic",
            f"wafer-board-collective-traffic-behavior-{case_name}",
            (
                "structured AllToAll/CollectivePermute traffic semantics "
                f"with exact all-rank output for {case_name}"
            ),
        )
        for case_name in COLLECTIVE_TRAFFIC_BEHAVIOR_CASES
    ),
    *tuple(
        CalibrationStep(
            f"direct-dte-pending-{case_name}",
            "full-card-pending-direct-dte-raw",
            f"wafer-board-{case_name}",
            (
                "isolated owner-backed raw DTE correctness observation for "
                f"{case_name}"
            ),
        )
        for case_name in DTE_NON_SHUFFLE_CASE_NAMES
    ),
    *tuple(
        CalibrationStep(
            f"engine-pipeline-{group_key}",
            "rank-one-pending-engine-pipeline",
            f"wafer-board-engine-pipeline-{group_key}",
            (
                "complete single-engine slope or engine-pair stage-balance "
                f"activation group {group_key}"
            ),
        )
        for group_key in ENGINE_PIPELINE_BOARD_GROUP_KEYS
    ),
    CalibrationStep(
        "engine-pipeline-single-ne-tail",
        "rank-one-pending-engine-pipeline",
        "wafer-board-ne-tail-throughput-single-ne-tail",
        (
            "fresh same-session NE small/steady controls followed by exact "
            "non-divisible NE GEMM tail and merged device-PMU activation"
        ),
    ),
    *tuple(
        CalibrationStep(
            f"instruction-family-{case_name}",
            "rank-one-pending-numeric-domain",
            f"wafer-board-instruction-family-{case_name}",
            (
                "bounded ArgMin value/index classification with three "
                f"domain-discriminating samples for {case_name}"
            ),
        )
        for case_name in ARGMIN_PENDING_DOMAIN_CASE_NAMES
    ),
    *tuple(
        CalibrationStep(
            f"instruction-family-{case_name}",
            "rank-one-pending-unpool-collision",
            f"wafer-board-instruction-family-{case_name}",
            (
                "bounded repeated-overlap Unpool collision observation for "
                f"{case_name}"
            ),
        )
        for case_name in UNPOOL_PENDING_COLLISION_CASE_NAMES
    ),
    CalibrationStep(
        "spm-conflict-equivalence-rank-one",
        "rank-one-pending-spm-conflict",
        "wafer-board-spm-conflict-equivalence-rank-one",
        "same-invocation SPM base/workload/order matched conflict controls",
    ),
    CalibrationStep(
        "spm-conflict-equivalence-cross-tile",
        "full-card-pending-spm-conflict",
        "wafer-board-spm-conflict-equivalence-cross-tile",
        "held-out physical-tile SPM conflict-equivalence control",
    ),
    *tuple(
        CalibrationStep(
            f"ddr-active-rank-{group_key}",
            "full-card-pending-ddr-active-rank",
            f"wafer-board-ddr-active-rank-{group_key}",
            (
                "one/two/four/eight/sixteen active-rank matched device-cycle "
                f"sweep for {group_key}"
            ),
        )
        for group_key in DDR_ACTIVE_RANK_BOARD_GROUP_KEYS
    ),
    CalibrationStep(
        "ddr-conflict-equivalence-rank-one",
        "rank-one-pending-ddr-conflict",
        "wafer-board-ddr-conflict-equivalence-rank-one",
        "same-invocation DDR address/order/schedule equivalence controls",
    ),
    CalibrationStep(
        "ddr-conflict-equivalence-cross-tile",
        "full-card-pending-ddr-conflict",
        "wafer-board-ddr-conflict-equivalence-cross-tile",
        "actual-allocation and physical-tile DDR equivalence held-out",
    ),
    # Session-risk tail.  These cases can intentionally fill queues, observe
    # pending work, require bounded partial-accept cleanup, or exercise raw
    # instruction programming.  Keep them after ordinary board observations.
    *tuple(
        CalibrationStep(
            f"spm-sustained-{group_key}",
            "rank-one-pending-spm-sustained",
            f"wafer-board-spm-sustained-{group_key}",
            (
                "matched candidate/control and serial/window sustained SPM "
                f"conflict observation for {group_key}"
            ),
        )
        for group_key in SPM_SUSTAINED_BOARD_GROUP_KEYS
    ),
    *tuple(
        CalibrationStep(
            f"ncc-{case_name}",
            "rank-one-pending-queue-saturation",
            f"wafer-board-ncc-{case_name}",
            (
                "tight documented-boundary queue admission/backpressure "
                f"observation for {case_name}"
            ),
        )
        for case_name in QUEUE_SATURATION_CASE_NAMES
    ),
    *tuple(
        CalibrationStep(
            f"ncc-{case_name}",
            "rank-one-pending-worker-wait-scope",
            f"wafer-board-ncc-{case_name}",
            (
                "pending-at-snapshot worker wait-scope exclusion "
                f"observation for {case_name}"
            ),
        )
        for case_name in WORKER_WAIT_SCOPE_CASE_NAMES
    ),
    *tuple(
        CalibrationStep(
            f"ncc-{case_name}",
            "rank-one-pending-worker-subset-scope",
            f"wafer-board-ncc-{case_name}",
            (
                "pending-at-snapshot worker subset include/exclude "
                f"observation for {case_name}"
            ),
        )
        for case_name in WORKER_SUBSET_SCOPE_CASE_NAMES
    ),
    *tuple(
        CalibrationStep(
            f"worker-placement-{group_key}",
            "rank-one-pending-worker-placement",
            f"wafer-board-worker-placement-{group_key}",
            (
                "fixed-total worker placement or matched bounded-progress "
                f"activation group {group_key}"
            ),
        )
        for group_key in WORKER_PLACEMENT_BOARD_GROUP_KEYS
    ),
    *tuple(
        CalibrationStep(
            f"direct-dte-pending-{case_name}",
            "full-card-pending-direct-dte-raw",
            f"wafer-board-{case_name}",
            (
                "isolated owner-backed raw DTE correctness observation for "
                f"{case_name}"
            ),
        )
        for case_name in DTE_SHUFFLE_CASE_NAMES
    ),
)

SESSION_RISK_STEP_KEYS = tuple(
    step.key
    for step in EXPLICIT_ONLY_STEPS
    if (
        step.batch
        in {
            "rank-one-pending-spm-sustained",
            "rank-one-pending-queue-saturation",
            "rank-one-pending-worker-wait-scope",
            "rank-one-pending-worker-subset-scope",
            "rank-one-pending-worker-placement",
        }
        or any(
            case_name in step.key for case_name in DTE_SHUFFLE_CASE_NAMES
        )
    )
)

ALL_CALIBRATION_STEPS = (
    *CALIBRATION_STEPS,
    *EXPLICIT_ONLY_STEPS,
)

SELECTABLE_BATCHES = {
    "compiler-search-scalability": (
        "m-sharded-replicated-gemm-profile",
    ),
    "compiler-optimization-paired": tuple(
        step.key
        for step in EXPLICIT_ONLY_STEPS
        if step.key.startswith("compiler-optimization-")
    ),
    "collective-characterization": tuple(
        step.key
        for step in EXPLICIT_ONLY_STEPS
        if step.key.startswith("collective-characterization-")
    ),
    "collective-traffic-behavior": tuple(
        step.key
        for step in EXPLICIT_ONLY_STEPS
        if step.key.startswith("collective-traffic-behavior-")
    ),
    "direct-dte-raw-behavior": tuple(
        step.key
        for step in EXPLICIT_ONLY_STEPS
        if step.key.startswith("direct-dte-pending-")
    ),
    "engine-pipeline-characterization": tuple(
        step.key
        for step in EXPLICIT_ONLY_STEPS
        if step.key.startswith("engine-pipeline-")
    ),
    "pending-execution-boundaries": tuple(
        step.key
        for step in EXPLICIT_ONLY_STEPS
        if step.batch.startswith("rank-one-pending-")
        or step.batch.startswith("full-card-pending-")
    ),
}
SELECTABLE_BATCHES["compiler-optimization-campaign"] = (
    "direct-dte-collective",
    *SELECTABLE_BATCHES["compiler-optimization-paired"],
)
_pending_board_ctests = {
    ctest
    for family in pending_inventory.FAMILIES
    if family.disposition == pending_inventory.PENDING_BOARD
    for ctest in family.board_ctests
}
_pending_steps_by_ctest = {
    step.ctest_name: step for step in EXPLICIT_ONLY_STEPS
}
_missing_pending_ctests = _pending_board_ctests - set(_pending_steps_by_ctest)
if _missing_pending_ctests:
    raise RuntimeError(
        "pending hardware calibration CTests lack runner steps: "
        + repr(sorted(_missing_pending_ctests))
    )
SELECTABLE_BATCHES["pending-hardware-calibration"] = tuple(
    step.key
    for step in EXPLICIT_ONLY_STEPS
    if step.ctest_name in _pending_board_ctests
)
if SELECTABLE_BATCHES["pending-hardware-calibration"][
    -len(SESSION_RISK_STEP_KEYS) :
] != SESSION_RISK_STEP_KEYS:
    raise RuntimeError(
        "potentially session-poisoning board cases must remain at the end "
        "of pending-hardware-calibration"
    )
del _missing_pending_ctests
del _pending_steps_by_ctest
del _pending_board_ctests


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--list",
        action="store_true",
        help="list and validate the ordered CTest plan without building or executing",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help=(
            "explicitly authorize the default fail-fast board plan, or the "
            "subset named by --step/--batch"
        ),
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=repo_root / "build" / "wafer-dev",
        help="configured board-enabled CMake build directory",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        help=(
            "new directory for the session summary, per-step logs, and JUnit "
            "files; defaults below the build directory"
        ),
    )
    parser.add_argument(
        "--step",
        action="append",
        dest="selected_steps",
        metavar="KEY",
        help=(
            "execute/list only this named calibration step; repeatable. "
            "The initial known-good heartbeat is added once automatically, "
            "and execution retains canonical order."
        ),
    )
    parser.add_argument(
        "--batch",
        action="append",
        dest="selected_batches",
        metavar="NAME",
        help=(
            "execute/list every step in this named explicit batch; repeatable. "
            "Named batches retain canonical order after the one automatic "
            "initial heartbeat."
        ),
    )
    parser.add_argument("--ctest", default="ctest", help="CTest executable")
    parser.add_argument("--cmake", default="cmake", help="CMake executable")
    return parser.parse_args(argv)


def property_map(test: Mapping[str, object]) -> dict[str, object]:
    properties: dict[str, object] = {}
    raw_properties = test.get("properties", [])
    if not isinstance(raw_properties, list):
        raise CalibrationRunnerError("CTest JSON contains malformed test properties")
    for raw_property in raw_properties:
        if not isinstance(raw_property, dict):
            raise CalibrationRunnerError("CTest JSON contains a malformed property")
        name = raw_property.get("name")
        if not isinstance(name, str):
            raise CalibrationRunnerError("CTest JSON property has no string name")
        properties[name] = raw_property.get("value")
    return properties


def load_registered_tests(
    ctest: str,
    build_dir: pathlib.Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, RegisteredTest]:
    if not build_dir.is_dir():
        raise CalibrationRunnerError(f"build directory does not exist: {build_dir}")
    command = [ctest, "--test-dir", str(build_dir), "--show-only=json-v1"]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=None if environment is None else dict(environment),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CalibrationRunnerError(
            f"failed to query CTest inventory ({result.returncode}): {detail}"
        )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CalibrationRunnerError(f"CTest returned invalid JSON: {error}") from error
    raw_tests = document.get("tests")
    if not isinstance(raw_tests, list):
        raise CalibrationRunnerError("CTest JSON has no test inventory")

    registered: dict[str, RegisteredTest] = {}
    for raw_test in raw_tests:
        if not isinstance(raw_test, dict):
            raise CalibrationRunnerError("CTest JSON contains a malformed test")
        name = raw_test.get("name")
        command_value = raw_test.get("command")
        if not isinstance(name, str):
            raise CalibrationRunnerError("CTest JSON test has no string name")
        if command_value is None:
            command: tuple[str, ...] = ()
        elif isinstance(command_value, list):
            command = tuple(str(argument) for argument in command_value)
        else:
            raise CalibrationRunnerError(
                f"CTest {name} has a malformed command"
            )
        if name in registered:
            raise CalibrationRunnerError(f"CTest registered duplicate test name: {name}")
        properties = property_map(raw_test)
        raw_labels = properties.get("LABELS", [])
        if isinstance(raw_labels, str):
            labels = frozenset((raw_labels,))
        elif isinstance(raw_labels, list) and all(
            isinstance(label, str) for label in raw_labels
        ):
            labels = frozenset(raw_labels)
        else:
            raise CalibrationRunnerError(f"CTest {name} has malformed LABELS")
        raw_timeout = properties.get("TIMEOUT")
        timeout = (
            float(raw_timeout)
            if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool)
            else None
        )
        raw_lock = properties.get("RESOURCE_LOCK")
        if isinstance(raw_lock, str):
            resource_lock = raw_lock
        elif (
            isinstance(raw_lock, list)
            and raw_lock
            and all(isinstance(lock, str) and lock for lock in raw_lock)
        ):
            resource_lock = ";".join(raw_lock)
        else:
            resource_lock = None
        raw_disabled = properties.get("DISABLED", False)
        disabled = raw_disabled is True or (
            isinstance(raw_disabled, str)
            and raw_disabled.upper() in {"1", "ON", "TRUE", "YES"}
        )
        registered[name] = RegisteredTest(
            name=name,
            command=command,
            labels=labels,
            timeout_seconds=timeout,
            resource_lock=resource_lock,
            disabled=disabled,
        )
    return registered


def is_forbidden_control_argument(argument: str) -> bool:
    lowered = argument.lower()
    if lowered in {"reset", "power", "power-cycle", "power_cycle"}:
        return True
    if not lowered.startswith("-"):
        return False
    option = lowered.split("=", maxsplit=1)[0]
    return "reset" in option or "power" in option


def validate_step_definition(steps: Sequence[CalibrationStep]) -> None:
    if not steps:
        raise CalibrationRunnerError("hardware calibration plan is empty")
    keys = [step.key for step in steps]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count != 1)
    if duplicate_keys:
        raise CalibrationRunnerError(f"duplicate calibration step keys: {duplicate_keys}")
    for step in steps:
        if not step.batch or not step.ctest_name or not step.purpose:
            raise CalibrationRunnerError(f"incomplete calibration step: {step.key}")
        if is_forbidden_control_argument(step.ctest_name):
            raise CalibrationRunnerError(
                f"calibration step names a forbidden control action: {step.ctest_name}"
            )
    duplicates = {
        name: count
        for name, count in Counter(step.ctest_name for step in steps).items()
        if count > 1
    }
    if duplicates:
        raise CalibrationRunnerError(
            f"calibration CTests must not repeat: {duplicates}"
        )


def validate_default_plan() -> None:
    validate_step_definition(CALIBRATION_STEPS)
    validate_step_definition(ALL_CALIBRATION_STEPS)
    if (
        CALIBRATION_STEPS[0].key != "initial-profile-heartbeat"
        or CALIBRATION_STEPS[0].ctest_name != HEARTBEAT_CTEST
    ):
        raise CalibrationRunnerError(
            "default plan must begin with the known-good heartbeat"
        )
    if ALL_CALIBRATION_STEPS[0] != CALIBRATION_STEPS[0]:
        raise CalibrationRunnerError(
            "selectable plan must retain the initial heartbeat"
        )
    default_keys = {step.key for step in CALIBRATION_STEPS}
    explicit_keys = {step.key for step in EXPLICIT_ONLY_STEPS}
    if default_keys & explicit_keys:
        raise CalibrationRunnerError(
            "explicit-only calibration steps must not enter the default plan"
        )


def select_calibration_steps(
    selected_keys: Sequence[str] | None,
    selected_batches: Sequence[str] | None = None,
) -> tuple[CalibrationStep, ...]:
    validate_default_plan()
    if not selected_keys and not selected_batches:
        return CALIBRATION_STEPS
    selected_keys = tuple(selected_keys or ())
    selected_batches = tuple(selected_batches or ())
    duplicate_batches = sorted(
        batch
        for batch, count in Counter(selected_batches).items()
        if count > 1
    )
    if duplicate_batches:
        raise CalibrationRunnerError(
            f"calibration batches were selected more than once: "
            f"{duplicate_batches}"
        )
    unknown_batches = sorted(set(selected_batches) - set(SELECTABLE_BATCHES))
    if unknown_batches:
        raise CalibrationRunnerError(
            f"unknown calibration batch names: {unknown_batches}"
        )
    expanded_keys = list(selected_keys)
    for batch in selected_batches:
        expanded_keys.extend(SELECTABLE_BATCHES[batch])
    duplicates = sorted(
        key for key, count in Counter(expanded_keys).items() if count > 1
    )
    if duplicates:
        raise CalibrationRunnerError(
            f"calibration steps were selected more than once: {duplicates}"
        )
    automatic_keys = {CALIBRATION_STEPS[0].key}
    explicitly_automatic = sorted(set(expanded_keys) & automatic_keys)
    if explicitly_automatic:
        raise CalibrationRunnerError(
            "heartbeat steps are automatic and cannot be selected explicitly: "
            f"{explicitly_automatic}"
        )
    selectable = {
        step.key: step for step in ALL_CALIBRATION_STEPS[1:]
    }
    unknown = sorted(set(expanded_keys) - set(selectable))
    if unknown:
        raise CalibrationRunnerError(
            f"unknown calibration step keys: {unknown}"
        )
    selected = set(expanded_keys)
    ordered = tuple(
        step
        for step in ALL_CALIBRATION_STEPS[1:]
        if step.key in selected
    )
    return (
        CALIBRATION_STEPS[0],
        *ordered,
    )


def validate_inventory(
    registered: Mapping[str, RegisteredTest],
    steps: Sequence[CalibrationStep],
    *,
    reject_unplanned_board_tests: bool = True,
) -> None:
    validate_step_definition(steps)
    planned_names = {step.ctest_name for step in steps}
    missing = sorted(planned_names - set(registered))
    if missing:
        raise CalibrationRunnerError(
            "board-enabled build is missing planned CTests: " + ", ".join(missing)
        )

    for name in sorted(planned_names):
        test = registered[name]
        if not {"board", "hardware"}.issubset(test.labels):
            raise CalibrationRunnerError(
                f"planned CTest {name} lacks board+hardware labels: "
                f"{sorted(test.labels)}"
            )
        if "no-card" in test.labels or "--no-card" in test.command:
            raise CalibrationRunnerError(
                f"planned CTest {name} resolves to a no-card path"
            )
        if test.timeout_seconds is None or test.timeout_seconds <= 0:
            raise CalibrationRunnerError(
                f"planned CTest {name} has no positive bounded TIMEOUT"
            )
        if not test.resource_lock:
            raise CalibrationRunnerError(
                f"planned CTest {name} has no board RESOURCE_LOCK"
            )
        if test.disabled:
            raise CalibrationRunnerError(f"planned CTest {name} is disabled")
        forbidden = [
            argument
            for argument in test.command
            if is_forbidden_control_argument(argument)
        ]
        if forbidden:
            raise CalibrationRunnerError(
                f"planned CTest {name} exposes reset/power control: {forbidden}"
            )

    if reject_unplanned_board_tests:
        eligible = {
            name
            for name, test in registered.items()
            if {"board", "hardware"}.issubset(test.labels)
            and "no-card" not in test.labels
        }
        unplanned = sorted(eligible - planned_names)
        if unplanned:
            raise CalibrationRunnerError(
                "new board CTests require an explicit batch decision before execution: "
                + ", ".join(unplanned)
            )


def default_log_dir(build_dir: pathlib.Path) -> pathlib.Path:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        build_dir
        / "hardware-calibration-logs"
        / f"{stamp}-{os.getpid()}"
    )


def write_summary(path: pathlib.Path, summary: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stream_command(
    command: Sequence[str],
    *,
    log_path: pathlib.Path,
    cwd: pathlib.Path,
    environment: Mapping[str, str],
) -> int:
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"started_at: {utc_now()}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"command: {shlex.join(command)}\n\n")
        log.flush()
        with subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise


def junit_result(junit_path: pathlib.Path, expected_name: str) -> str | None:
    if not junit_path.is_file():
        return "CTest did not produce the expected JUnit result"
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError as error:
        return f"CTest produced malformed JUnit XML: {error}"
    cases = root.findall(".//testcase")
    if root.tag == "testcase":
        cases = [root]
    if len(cases) != 1:
        return f"CTest JUnit contains {len(cases)} cases instead of exactly one"
    case = cases[0]
    if case.get("name") != expected_name:
        return (
            f"CTest JUnit reported {case.get('name')!r}, expected {expected_name!r}"
        )
    if case.find("skipped") is not None:
        return "hardware CTest was skipped"
    if case.find("failure") is not None or case.find("error") is not None:
        return "hardware CTest reported a failure"
    for node in (root, case):
        for attribute in ("failures", "errors", "skipped", "disabled"):
            value = node.get(attribute)
            if value not in (None, "0"):
                return f"hardware CTest JUnit has {attribute}={value}"
    return None


def ctest_command(
    ctest: str,
    build_dir: pathlib.Path,
    test_name: str,
    junit_path: pathlib.Path,
) -> list[str]:
    return [
        ctest,
        "--test-dir",
        str(build_dir),
        "--verbose",
        "--output-on-failure",
        "--no-tests=error",
        "--parallel",
        "1",
        "--tests-regex",
        f"^{re.escape(test_name)}$",
        "--output-junit",
        str(junit_path),
    ]


def command_option_value(
    command: Sequence[str], option: str
) -> str | None:
    prefix = option + "="
    for index, argument in enumerate(command):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option:
            if index + 1 >= len(command):
                raise CalibrationRunnerError(
                    f"CTest command ends after {option}"
                )
            return command[index + 1]
    return None


def profile_report_members(
    test: RegisteredTest, work_dir: pathlib.Path
) -> tuple[pathlib.Path, ...]:
    if "profiler" not in test.labels:
        return ()

    runs = work_dir / "package.profile" / "runs"
    current = runs / "current"
    if not current.is_symlink():
        raise CalibrationRunnerError(
            f"{test.name}: profiler did not publish a managed current report"
        )
    try:
        canonical_runs = runs.resolve(strict=True)
        run_directory = current.resolve(strict=True)
    except OSError as error:
        raise CalibrationRunnerError(
            f"{test.name}: profiler current report cannot be resolved: {error}"
        ) from error
    if run_directory.parent != canonical_runs:
        raise CalibrationRunnerError(
            f"{test.name}: profiler current report escapes its runs directory"
        )
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise CalibrationRunnerError(
            f"{test.name}: profiler current target is not a real directory"
        )

    expected_names = {"evidence.json", "analysis.json", "index.html"}
    actual_members = tuple(run_directory.iterdir())
    if (
        {path.name for path in actual_members} != expected_names
        or len(actual_members) != len(expected_names)
        or any(not path.is_file() or path.is_symlink() for path in actual_members)
    ):
        raise CalibrationRunnerError(
            f"{test.name}: profiler current report is not the complete "
            "three-file public artifact group"
        )
    evidence_path = run_directory / "evidence.json"
    try:
        evidence = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationRunnerError(
            f"{test.name}: profiler evidence is not readable JSON: {error}"
        ) from error
    if (
        not isinstance(evidence, dict)
        or evidence.get("run_id") != run_directory.name
    ):
        raise CalibrationRunnerError(
            f"{test.name}: profiler evidence run_id does not match current "
            "report directory"
        )
    return tuple(
        run_directory / name
        for name in ("evidence.json", "analysis.json", "index.html")
    )


def archive_step_artifacts(
    test: RegisteredTest,
    destination: pathlib.Path,
) -> dict[str, object] | None:
    raw_work_dir = command_option_value(test.command, "--work-dir")
    if raw_work_dir is None:
        return None
    work_dir = pathlib.Path(raw_work_dir)
    if not work_dir.is_dir():
        raise CalibrationRunnerError(
            f"{test.name}: declared work directory was not produced: {work_dir}"
        )
    durable_suffixes = {".raw", ".json", ".jsonl"}
    preserves_compiler_artifacts = (
        "pending" in test.labels
        or "compiler-optimization" in test.labels
        or "collective-characterization" in test.labels
        or "collective-traffic" in test.labels
        or "engine" in test.labels
        or "pipeline" in test.labels
        or test.name == "wafer-board-cluster-direct-dte"
    )
    if preserves_compiler_artifacts:
        # Preserve the exact source snapshots and final linked ELFs as
        # read-only audit evidence for pending calibration and paired results.
        durable_suffixes.update({".mlir", ".meta", ".so"})
    required_profile_report = profile_report_members(test, work_dir)
    evidence_files = tuple(
        sorted(
            {
                *required_profile_report,
                *(
                    path
                    for path in work_dir.rglob("*")
                    if path.is_file()
                    and (
                        path.suffix in durable_suffixes
                        or path.name in {"session.txt", "summary.txt"}
                    )
                ),
            }
        )
    )
    destination.mkdir(parents=True, exist_ok=False)
    manifest_files: list[dict[str, object]] = []
    for source in evidence_files:
        relative = source.relative_to(work_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest_files.append(
            {
                "path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": digest,
            }
        )
    tool_options = (
        "--wafer-compile",
        "--wafer-compile-test",
        "--wafer-run",
        "--tx8-objdump",
        "--llvm-clangxx",
    )
    tools: list[dict[str, object]] = []
    for option in tool_options:
        value = command_option_value(test.command, option)
        if value is None:
            continue
        tool = pathlib.Path(value)
        if not tool.is_file():
            raise CalibrationRunnerError(
                f"{test.name}: {option} tool does not exist: {tool}"
            )
        tools.append(
            {
                "option": option,
                "path": str(tool),
                "bytes": tool.stat().st_size,
                "sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 2,
        "ctest": test.name,
        "ctest_command": list(test.command),
        "source_work_dir": str(work_dir),
        "files": manifest_files,
        "tools": tools,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {
        "directory": str(destination),
        "manifest": str(manifest_path),
        "file_count": len(manifest_files),
    }


def execute_calibration(
    *,
    build_dir: pathlib.Path,
    log_dir: pathlib.Path,
    ctest: str,
    cmake: str,
    steps: Sequence[CalibrationStep],
    inventory_steps: Sequence[CalibrationStep] | None = None,
    environment: Mapping[str, str],
) -> int:
    validate_step_definition(steps)
    audited_steps = steps if inventory_steps is None else inventory_steps
    validate_step_definition(audited_steps)
    if not {step.ctest_name for step in steps}.issubset(
        {step.ctest_name for step in audited_steps}
    ):
        raise CalibrationRunnerError(
            "execution plan contains a CTest outside the audited inventory"
        )
    if environment.get(ARM_ENVIRONMENT_VARIABLE) != "1":
        raise CalibrationRunnerError(
            f"execution requires {ARM_ENVIRONMENT_VARIABLE}=1"
        )
    if not build_dir.is_dir():
        raise CalibrationRunnerError(f"build directory does not exist: {build_dir}")
    registered = load_registered_tests(ctest, build_dir, environment)
    validate_inventory(registered, audited_steps)
    log_dir.mkdir(parents=True, exist_ok=False)
    summary_path = log_dir / "session.json"
    calibration_session_id = secrets.token_hex(16)
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "build_dir": str(build_dir),
        "calibration_session_id": calibration_session_id,
        "hardware_execution_armed": (
            environment.get(ARM_ENVIRONMENT_VARIABLE) == "1"
        ),
        "automatic_retry": False,
        "reset_or_power_control": False,
        "steps": [],
    }
    write_summary(summary_path, summary)

    child_environment = dict(environment)
    child_environment[SESSION_ENVIRONMENT_VARIABLE] = calibration_session_id
    child_environment["CTEST_PARALLEL_LEVEL"] = "1"
    child_environment["CTEST_OUTPUT_ON_FAILURE"] = "1"

    build_log = log_dir / "000-incremental-build.log"
    build_targets = ["wafer-compile", "wafer-run"]
    if any(
        step.key.startswith(
            (
                "compiler-optimization-",
                "collective-characterization-",
                "collective-traffic-behavior-",
            )
        )
        for step in steps
    ):
        build_targets.append("wafer-compile-test")
    build_command = [
        cmake,
        "--build",
        str(build_dir),
        "--target",
        *build_targets,
        "--parallel",
        "128",
    ]
    print(f"[build] {shlex.join(build_command)}")
    build_started = time.monotonic()
    build_returncode = stream_command(
        build_command,
        log_path=build_log,
        cwd=build_dir,
        environment=child_environment,
    )
    summary["incremental_build"] = {
        "status": "passed" if build_returncode == 0 else "failed",
        "returncode": build_returncode,
        "elapsed_seconds": round(time.monotonic() - build_started, 3),
        "log": str(build_log),
    }
    if build_returncode != 0:
        summary["status"] = "failed"
        summary["failure"] = "incremental build failed before board execution"
        summary["ended_at"] = utc_now()
        write_summary(summary_path, summary)
        return 1

    try:
        registered = load_registered_tests(ctest, build_dir, child_environment)
        validate_inventory(registered, audited_steps)
    except CalibrationRunnerError as error:
        summary["status"] = "failed"
        summary["failure"] = str(error)
        summary["ended_at"] = utc_now()
        write_summary(summary_path, summary)
        print(f"hardware_calibration: {error}", file=sys.stderr)
        return 1

    summary["planned_ctest_count"] = len(steps)
    summary["unique_ctest_count"] = len({step.ctest_name for step in steps})
    write_summary(summary_path, summary)

    step_results = summary["steps"]
    assert isinstance(step_results, list)
    for index, step in enumerate(steps, start=1):
        stem = f"{index:03d}-{step.batch}-{step.key}"
        log_path = log_dir / f"{stem}.log"
        junit_path = log_dir / f"{stem}.junit.xml"
        command = ctest_command(ctest, build_dir, step.ctest_name, junit_path)
        print(
            f"[{index}/{len(steps)}] {step.batch}: {step.ctest_name}",
            flush=True,
        )
        started_at = utc_now()
        started = time.monotonic()
        try:
            returncode = stream_command(
                command,
                log_path=log_path,
                cwd=build_dir,
                environment=child_environment,
            )
        except KeyboardInterrupt:
            result = {
                "key": step.key,
                "batch": step.batch,
                "ctest": step.ctest_name,
                "status": "interrupted",
                "started_at": started_at,
                "ended_at": utc_now(),
                "log": str(log_path),
                "junit": str(junit_path),
            }
            step_results.append(result)
            summary["status"] = "interrupted"
            summary["failure"] = (
                "host orchestration was interrupted; no retry, reset, or power "
                "action was attempted"
            )
            summary["ended_at"] = utc_now()
            write_summary(summary_path, summary)
            raise
        junit_error = (
            junit_result(junit_path, step.ctest_name)
            if returncode == 0
            else f"CTest process exited with code {returncode}"
        )
        artifact_error: str | None = None
        artifacts: dict[str, object] | None = None
        try:
            artifacts = archive_step_artifacts(
                registered[step.ctest_name],
                log_dir / "artifacts" / stem,
            )
        except (CalibrationRunnerError, OSError) as error:
            artifact_error = f"failed to archive durable evidence: {error}"
            if junit_error is None:
                junit_error = artifact_error
        status = "passed" if junit_error is None else "failed"
        result = {
            "key": step.key,
            "batch": step.batch,
            "ctest": step.ctest_name,
            "purpose": step.purpose,
            "status": status,
            "returncode": returncode,
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path),
            "junit": str(junit_path),
        }
        if artifacts is not None:
            result["artifacts"] = artifacts
        if artifact_error is not None:
            result["artifact_error"] = artifact_error
        if junit_error is not None:
            result["failure"] = junit_error
        step_results.append(result)
        write_summary(summary_path, summary)
        if junit_error is not None:
            summary["status"] = "failed"
            summary["failure"] = (
                f"{step.ctest_name}: {junit_error}; all later board tests were "
                "left unexecuted"
            )
            summary["ended_at"] = utc_now()
            write_summary(summary_path, summary)
            print(f"hardware_calibration: {summary['failure']}", file=sys.stderr)
            return 1

    summary["status"] = "passed"
    summary["ended_at"] = utc_now()
    write_summary(summary_path, summary)
    print(f"hardware_calibration: passed; evidence: {log_dir}")
    return 0


def print_plan(
    registered: Mapping[str, RegisteredTest], steps: Sequence[CalibrationStep]
) -> None:
    print("ORDER  BATCH                         CTEST")
    for index, step in enumerate(steps, start=1):
        test = registered[step.ctest_name]
        print(
            f"{index:>5}  {step.batch:<28}  {step.ctest_name} "
            f"(timeout={test.timeout_seconds:g}s)"
        )
    print(
        f"\n{len(steps)} ordered executions, "
        f"{len({step.ctest_name for step in steps})} unique board CTests; "
        "one process at a time, no retry/reset/power."
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    args = parse_args(argv)
    active_environment = dict(os.environ if environment is None else environment)
    build_dir = args.build_dir.resolve()
    try:
        validate_default_plan()
        selected_steps = select_calibration_steps(
            args.selected_steps, args.selected_batches
        )
        if args.list:
            registered = load_registered_tests(
                args.ctest, build_dir, active_environment
            )
            validate_inventory(registered, ALL_CALIBRATION_STEPS)
            print_plan(registered, selected_steps)
            return 0

        if active_environment.get(ARM_ENVIRONMENT_VARIABLE) != "1":
            raise CalibrationRunnerError(
                f"--execute also requires {ARM_ENVIRONMENT_VARIABLE}=1"
            )
        log_dir = (
            args.log_dir.resolve()
            if args.log_dir is not None
            else default_log_dir(build_dir)
        )
        return execute_calibration(
            build_dir=build_dir,
            log_dir=log_dir,
            ctest=args.ctest,
            cmake=args.cmake,
            steps=selected_steps,
            inventory_steps=ALL_CALIBRATION_STEPS,
            environment=active_environment,
        )
    except CalibrationRunnerError as error:
        print(f"hardware_calibration: {error}", file=sys.stderr)
        return 2
    except FileExistsError as error:
        print(
            "hardware_calibration: refusing to overwrite an existing log "
            f"directory: {error.filename}",
            file=sys.stderr,
        )
        return 2
    except OSError as error:
        print(f"hardware_calibration: host orchestration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "hardware_calibration: interrupted; stopped without retry/reset/power",
            file=sys.stderr,
        )
        raise SystemExit(130)
