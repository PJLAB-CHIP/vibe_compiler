#!/usr/bin/env python3
"""Executable inventory for calibration rows that still need board evidence."""

from __future__ import annotations

import dataclasses

import wafer_collective_traffic_behavior_catalog as collective_traffic
import wafer_engine_pipeline_characterization_catalog as engine_pipeline
import wafer_spm_sustained_conflict_catalog as spm_sustained
import wafer_unrepresentable_hardware_behavior_catalog as unrepresentable
import wafer_worker_memory_contention_characterization_catalog as contention
import wafer_worker_placement_characterization_catalog as worker_placement


PENDING_BOARD = "pending-board"
HOST_NEGATIVE = "host-negative"
BLOCKED_EXTERNAL = "blocked-external"


@dataclasses.dataclass(frozen=True)
class CatalogBinding:
    asset: str
    symbol: str
    identifiers: tuple[str, ...]
    identifier_field: str = "name"


@dataclasses.dataclass(frozen=True)
class PendingCalibrationFamily:
    key: str
    disposition: str
    execution_scope: str
    bindings: tuple[CatalogBinding, ...]
    board_ctests: tuple[str, ...]
    no_card_ctests: tuple[str, ...]
    runner_batch: str | None
    oracle: tuple[str, ...]
    activation_gate: tuple[str, ...]
    blocker: str | None = None


def _binding(
    asset: str,
    symbol: str,
    identifiers: tuple[str, ...],
    identifier_field: str = "name",
) -> CatalogBinding:
    return CatalogBinding(asset, symbol, identifiers, identifier_field)


COMPILER_OPTIMIZATION_CASES = (
    "reciprocal-implementation",
    "modular-common-factor",
    "resident-fanout-share",
    "consumer-local-recompute",
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
QUEUE_SATURATION_CASES = (
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
WORKER_WAIT_SCOPE_CASES = tuple(
    f"worker-wait-scope-{engine}-worker{worker}-{wait_kind}-tight-window"
    for worker in range(3)
    for engine in ("ne", "rdma")
    for wait_kind in ("default", "byworker", "local-fence")
)
WORKER_SUBSET_SCOPE_CASES = tuple(
    f"worker-subset-{engine}-target{worker}-{membership}-tight-window"
    for worker in range(3)
    for engine in ("ne", "rdma")
    for membership in ("exclude", "include")
)
WORKER_PLACEMENT_CASE_KEYS = tuple(case.key for case in worker_placement.CASES)
WORKER_PLACEMENT_GROUP_KEYS = tuple(worker_placement.GROUPS)
WORKER_PLACEMENT_BOUNDARY_KEYS = tuple(
    boundary.key for boundary in worker_placement.BOUNDARIES
)
ARGMIN_DOMAIN_CASES = (
    "peripheral-argmin-tie-f16-observed",
    "peripheral-argmin-nan-f16-observed",
)
UNPOOL_COLLISION_CASES = (
    "unpool-index-f16-repeated-overlap-observed",
    "unpool-mask-f16-repeated-overlap-observed",
)
COLLECTIVE_TRAFFIC_CASES = tuple(collective_traffic.CASE_KEYS)
COLLECTIVE_TRAFFIC_BLOCKED_COVERAGE_KEYS = tuple(
    item.key
    for item in collective_traffic.COVERAGE_ITEMS
    if (
        item.disposition
        == collective_traffic.CoverageDisposition.BLOCKED_FAIL_CLOSED
    )
)
ENGINE_PIPELINE_BOARD_CELL_KEYS = tuple(
    cell.key
    for cell in engine_pipeline.ALL_CELLS
    if cell.disposition
    in {
        engine_pipeline.Disposition.BOARD_EXECUTABLE,
        engine_pipeline.Disposition.BOARD_EXECUTABLE_EXTERNAL,
    }
)
ENGINE_PIPELINE_BOARD_GROUP_KEYS = tuple(
    group.key for group in engine_pipeline.BOARD_ACTIVATION_GROUPS
)
ENGINE_PIPELINE_EXTERNAL_GROUP_KEYS = tuple(
    engine_pipeline.EXTERNAL_BOARD_GROUP_KEYS
)
ENGINE_PIPELINE_BLOCKED_CELL_KEYS = tuple(
    cell.key
    for cell in engine_pipeline.ALL_CELLS
    if cell.disposition
    in {
        engine_pipeline.Disposition.FAIL_CLOSED_MISSING_ADAPTER,
        engine_pipeline.Disposition.FAIL_CLOSED_MISSING_PRODUCER,
        engine_pipeline.Disposition.FAIL_CLOSED_UNSAFE_WINDOW,
    }
)
SPM_SUSTAINED_CELL_KEYS = tuple(cell.key for cell in spm_sustained.CELLS)
SPM_SUSTAINED_GROUP_KEYS = tuple(group.key for group in spm_sustained.GROUPS)
DDR_ACTIVE_RANK_CASE_KEYS = tuple(
    case.key for case in contention.DDR_ACTIVE_RANK_CASES
)
DDR_ACTIVE_RANK_GROUP_KEYS = tuple(contention.DDR_ACTIVE_RANK_GROUP_KEYS)
MEMORY_UNREPRESENTABLE_BOUNDARY_KEYS = tuple(
    boundary.key for boundary in contention.TYPED_BOUNDARIES
)
UNREPRESENTABLE_BEHAVIOR_KEYS = tuple(
    behavior.key for behavior in unrepresentable.BEHAVIORS
)


def _engine_pipeline_ctest(group_key: str) -> str:
    return "wafer-board-engine-pipeline-" + group_key


def _spm_sustained_ctest(group_key: str) -> str:
    return "wafer-board-spm-sustained-" + group_key


def _ddr_active_rank_ctest(group_key: str) -> str:
    return "wafer-board-ddr-active-rank-" + group_key


def _worker_placement_ctest(group_key: str) -> str:
    return "wafer-board-worker-placement-" + group_key


FAMILIES = (
    PendingCalibrationFamily(
        key="production-optimizer-paired-qualification",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-and-full-card",
        bindings=(
            _binding(
                "test/Board/wafer_compiler_optimization_campaign_catalog.py",
                "CAMPAIGN_CASES",
                COMPILER_OPTIMIZATION_CASES,
                "key",
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-compiler-optimization-{case}"
            for case in COMPILER_OPTIMIZATION_CASES
        ),
        no_card_ctests=tuple(
            f"wafer-runtime-compiler-optimization-{case}-no-card"
            for case in COMPILER_OPTIMIZATION_CASES
        ),
        runner_batch="compiler-optimization-paired",
        oracle=(
            "same-source-baseline-versus-winner",
            "full-cpu-expected",
            "normalized-manifest-and-final-elf-structure",
            "terminal-status-and-cleanup",
        ),
        activation_gate=("explicit-batch", "board-profile-qualified"),
    ),
    PendingCalibrationFamily(
        key="collective-algorithm-characterization",
        disposition=PENDING_BOARD,
        execution_scope="full-card-16-rank",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_collective_hardware_characterization_catalog.py",
                "CASES",
                COLLECTIVE_CHARACTERIZATION_CASES,
                "key",
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-collective-characterization-{case}"
            for case in COLLECTIVE_CHARACTERIZATION_CASES
        ),
        no_card_ctests=tuple(
            f"wafer-runtime-collective-characterization-{case}-no-card"
            for case in COLLECTIVE_CHARACTERIZATION_CASES
        ),
        runner_batch="collective-characterization",
        oracle=(
            "full-rank-output-exact",
            "accepted-instr-message-tuples",
            "cross-rank-message-matching",
            "algorithm-specific-graph",
            "transport-status-and-cleanup",
        ),
        activation_gate=("explicit-batch", "board-profile-qualified"),
    ),
    PendingCalibrationFamily(
        key="collective-traffic-semantics",
        disposition=PENDING_BOARD,
        execution_scope="full-card-16-rank",
        bindings=(
            _binding(
                "test/Board/wafer_collective_traffic_behavior_catalog.py",
                "CASES",
                COLLECTIVE_TRAFFIC_CASES,
                "key",
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-collective-traffic-behavior-{case}"
            for case in COLLECTIVE_TRAFFIC_CASES
        ),
        no_card_ctests=tuple(
            f"wafer-runtime-collective-traffic-behavior-{case}-no-card"
            for case in COLLECTIVE_TRAFFIC_CASES
        ),
        runner_batch="collective-traffic-behavior",
        oracle=(
            "structured-source-to-status-v2-package",
            "full-rank-output-exact",
            "source-graph-and-min-hop-demand-record",
            "transport-status-final-elf-terminal-and-cleanup",
        ),
        activation_gate=(
            "explicit-batch",
            "correctness-and-logical-traffic-only",
            "no-device-cost-without-phase-basis",
        ),
    ),
    PendingCalibrationFamily(
        key="collective-traffic-unsupported-surfaces",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/wafer_collective_traffic_behavior_catalog.py",
                "COVERAGE_ITEMS",
                COLLECTIVE_TRAFFIC_BLOCKED_COVERAGE_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-collective-traffic-behavior-catalog-python",
        ),
        runner_batch=None,
        oracle=(
            "typed-surface-or-abi-rejection",
            "no-sequential-unicast-as-concurrent-hotspot",
            "no-logical-endpoint-as-physical-route",
        ),
        activation_gate=("host-gate-only", "promotion-surface-required"),
        blocker=(
            "concurrent 4/8/15-way endpoint ABI, device phase basis, physical "
            "route report, native multicast or multi-card transport is absent"
        ),
    ),
    PendingCalibrationFamily(
        key="single-engine-and-engine-pair-characterization",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-complete-activation-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_engine_pipeline_characterization_catalog.py",
                "ALL_CELLS",
                ENGINE_PIPELINE_BOARD_CELL_KEYS,
                "key",
            ),
        ),
        board_ctests=tuple(
            _engine_pipeline_ctest(group)
            for group in ENGINE_PIPELINE_BOARD_GROUP_KEYS
        )
        + ("wafer-board-ne-tail-throughput-single-ne-tail",),
        no_card_ctests=(
            "wafer-runtime-engine-pipeline-characterization-no-card",
            "wafer-runtime-ne-tail-throughput-no-card",
            "wafer-ne-tail-throughput-catalog-python",
        ),
        runner_batch="engine-pipeline-characterization",
        oracle=(
            "full-result-and-prefix-suffix-guards",
            "exact-instruction-count-matching-completion-and-cleanup",
            "per-engine-and-global-union-device-counters",
            "three-repeats-and-matched-reciprocal-controls",
            "external-ne-tail-exact-pmu-archive",
        ),
        activation_gate=(
            "one-complete-activation-group-per-ctest",
            "no-host-elapsed-as-cost",
            "complete-activation-group-required",
            "ne-small-steady-tail-session-merge-required",
        ),
    ),
    PendingCalibrationFamily(
        key="engine-pipeline-unrepresentable-cells",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_engine_pipeline_characterization_catalog.py",
                "ALL_CELLS",
                ENGINE_PIPELINE_BLOCKED_CELL_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-engine-pipeline-characterization-catalog-python",
            "wafer-engine-pipeline-production-gate-python",
        ),
        runner_batch=None,
        oracle=(
            "adapter-capacity-and-tdma-window-rejection",
            "accepted-instr-production-provenance-gate",
            "no-handwritten-double-slot-as-production-vertical",
        ),
        activation_gate=(
            "host-gate-only",
            "real-adapter-or-production-producer-required",
        ),
        blocker=(
            "the production three-stage accepted-Instr multi-buffer producer "
            "is not implemented yet"
        ),
    ),
    PendingCalibrationFamily(
        key="queue-saturation-response",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/wafer_board_ncc_execution_probe_test.py",
                "V2_QUEUE_SATURATION_CASES",
                QUEUE_SATURATION_CASES,
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-ncc-{case}" for case in QUEUE_SATURATION_CASES
        ),
        no_card_ctests=("wafer-runtime-ncc-queue-saturation-no-card",),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "per-issue-admission-cycle-and-return",
            "pre-wait-control-snapshot",
            "instruction-count-full-result-and-guards",
            "matching-wait-terminal-and-cleanup",
        ),
        activation_gate=(
            "one-case-per-process",
            "documented-depth-minus-one-depth-depth-plus-one-only",
            "no-retry-reset-or-power",
        ),
    ),
    PendingCalibrationFamily(
        key="worker-wait-scope-exclusion",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/wafer_board_ncc_execution_probe_test.py",
                "V2_WORKER_WAIT_SCOPE_CASES",
                WORKER_WAIT_SCOPE_CASES,
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-ncc-{case}" for case in WORKER_WAIT_SCOPE_CASES
        ),
        no_card_ctests=("wafer-runtime-ncc-worker-wait-scope-no-card",),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "target-pending-before-wait",
            "boundary-worker-control-and-result",
            "matching-safety-drain",
            "full-result-guards-count-and-cleanup",
        ),
        activation_gate=("one-case-per-process", "pending-snapshot-required"),
    ),
    PendingCalibrationFamily(
        key="worker-subset-join-exclusion",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/wafer_board_ncc_execution_probe_test.py",
                "V2_WORKER_SUBSET_SCOPE_CASES",
                WORKER_SUBSET_SCOPE_CASES,
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-ncc-{case}" for case in WORKER_SUBSET_SCOPE_CASES
        ),
        no_card_ctests=("wafer-runtime-ncc-worker-subset-scope-no-card",),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "target-pending-before-join",
            "include-exclude-mask-pair",
            "boundary-worker-control-and-result",
            "final-all-worker-drain-guards-count-and-cleanup",
        ),
        activation_gate=("one-case-per-process", "pending-snapshot-required"),
    ),
    PendingCalibrationFamily(
        key="worker-placement-and-bounded-progress",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-matched-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_worker_placement_characterization_catalog.py",
                "CASES",
                WORKER_PLACEMENT_CASE_KEYS,
                "key",
            ),
        ),
        board_ctests=tuple(
            _worker_placement_ctest(group)
            for group in WORKER_PLACEMENT_GROUP_KEYS
        ),
        no_card_ctests=(
            "wafer-worker-placement-characterization-python",
            "wafer-runtime-worker-placement-characterization-no-card",
        ),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "fixed-total-work-placement-and-matched-progress-controls",
            "full-output-prefix-suffix-guards-and-routing-counts",
            "per-worker-instruction-blocking-and-global-pmu",
            "observer-boundary-target-safety-drain-terminal-and-cleanup",
        ),
        activation_gate=(
            "one-complete-matched-group-per-ctest",
            "three-serial-repeats",
            "no-absolute-worker-timestamp-or-arbiter-name",
        ),
    ),
    PendingCalibrationFamily(
        key="worker-placement-unrepresentable-surfaces",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_worker_placement_characterization_catalog.py",
                "BOUNDARIES",
                WORKER_PLACEMENT_BOUNDARY_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-worker-placement-characterization-python",
        ),
        runner_batch=None,
        oracle=(
            "typed-preparation-rejection",
            "no-absolute-per-worker-completion-cycle",
            "no-unowned-physical-arbiter-policy",
        ),
        activation_gate=(
            "host-gate-only",
            "owner-backed-timestamp-or-arbiter-surface-required",
        ),
        blocker=(
            "the current PMU exposes no common-basis per-worker completion "
            "timestamp, and no owner-backed ABI identifies the physical "
            "worker arbiter policy"
        ),
    ),
    PendingCalibrationFamily(
        key="spm-conflict-equivalence",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-plus-full-card-held-out",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_memory_descriptor_calibration_catalog.py",
                "PENDING_CONFLICT_EQUIVALENCE_CASES",
                (),
            ),
            _binding(
                "test/Board/"
                "wafer_board_spm_cross_tile_conflict_probe_test.py",
                "conflict_pairs",
                (),
                "@callable",
            ),
        ),
        board_ctests=(
            "wafer-board-spm-conflict-equivalence-rank-one",
            "wafer-board-spm-conflict-equivalence-cross-tile",
        ),
        no_card_ctests=(
            "wafer-runtime-spm-conflict-equivalence-probe-no-card",
            "wafer-runtime-spm-cross-tile-conflict-probe-no-card",
        ),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "same-invocation-serial-window-pair",
            "reciprocal-order-base-workload-controls",
            "full-result-guards-count-and-completion",
            "physical-tile-held-out",
        ),
        activation_gate=(
            "nonzero-stable-proxy-required-for-promotion",
            "no-bank-name-without-owner-backed-mapping-or-counter",
        ),
    ),
    PendingCalibrationFamily(
        key="spm-sustained-conflict-pilot",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-matched-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/wafer_spm_sustained_conflict_catalog.py",
                "CELLS",
                SPM_SUSTAINED_CELL_KEYS,
                "key",
            ),
        ),
        board_ctests=tuple(
            _spm_sustained_ctest(group)
            for group in SPM_SUSTAINED_GROUP_KEYS
        ),
        no_card_ctests=(
            "wafer-runtime-spm-sustained-conflict-probe-no-card",
            "wafer-spm-sustained-conflict-contract-python",
        ),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "same-invocation-candidate-control-serial-window",
            "four-way-execution-order-rotation",
            "full-owned-spm-and-gap-canary-exact",
            "pair-only-pmu-count-completion-address-and-lifecycle",
        ),
        activation_gate=(
            "one-complete-matched-group-per-ctest",
            "stable-nonzero-heldout-required-for-cost",
            "no-physical-bank-name-from-offset-equivalence",
        ),
    ),
    PendingCalibrationFamily(
        key="ddr-conflict-equivalence",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-plus-full-card-held-out",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_cache_coherence_calibration_catalog.py",
                "PENDING_DDR_CONFLICT_CASES",
                (),
            ),
            _binding(
                "test/Board/wafer_board_ddr_tile_offset_probe_test.py",
                "conflict_equivalence_cases",
                (),
                "@callable",
            ),
        ),
        board_ctests=(
            "wafer-board-ddr-conflict-equivalence-rank-one",
            "wafer-board-ddr-conflict-equivalence-cross-tile",
        ),
        no_card_ctests=(
            "wafer-runtime-ddr-conflict-equivalence-probe-no-card",
            "wafer-runtime-ddr-cross-tile-conflict-probe-no-card",
        ),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "same-invocation-address-schedule-order-controls",
            "actual-allocation-base",
            "full-payload-holes-guards-count-and-completion",
            "physical-tile-held-out",
        ),
        activation_gate=(
            "explicit-batch-only",
            "no-controller-or-bank-name-without-owned-observation",
        ),
    ),
    PendingCalibrationFamily(
        key="ddr-active-rank-contention",
        disposition=PENDING_BOARD,
        execution_scope="full-card-five-point-matched-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_worker_memory_contention_characterization_catalog.py",
                "DDR_ACTIVE_RANK_CASES",
                DDR_ACTIVE_RANK_CASE_KEYS,
                "key",
            ),
        ),
        board_ctests=tuple(
            _ddr_active_rank_ctest(group)
            for group in DDR_ACTIVE_RANK_GROUP_KEYS
        ),
        no_card_ctests=(
            "wafer-runtime-ddr-active-rank-contention-no-card",
            "wafer-ddr-active-rank-contention-contract-python",
        ),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "all-sixteen-ranks-one-lifecycle-and-explicit-active-mask",
            "active-exact-output-guards-count-and-device-cycles",
            "inactive-rank-output-and-pmu-canary",
            "one-two-four-eight-sixteen-rank-matched-sweep",
        ),
        activation_gate=(
            "one-complete-five-point-group-per-ctest",
            "calibration-and-strided-heldout-direction-must-agree",
            "no-host-elapsed-or-bank-identity",
        ),
    ),
    PendingCalibrationFamily(
        key="memory-unrepresentable-physical-and-unsafe-surfaces",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_worker_memory_contention_characterization_catalog.py",
                "TYPED_BOUNDARIES",
                MEMORY_UNREPRESENTABLE_BOUNDARY_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-ddr-active-rank-contention-contract-python",
        ),
        runner_batch=None,
        oracle=(
            "typed-preparation-rejection",
            "no-worker-alias-or-unordered-same-address-surrogate",
            "no-remote-or-unowned-spm-probe",
            "no-ddr-bank-name-without-mapping-and-counter",
        ),
        activation_gate=(
            "host-gate-only",
            "owner-backed-abi-or-readonly-counter-required",
        ),
        blocker=(
            "worker aliases, unordered same-address access, remote/unowned "
            "SPM, DDR bank/controller identity, bank counter, and WDMA "
            "cross-allocation held-out are not representable by the current "
            "owned ABI"
        ),
    ),
    PendingCalibrationFamily(
        key="unrepresentable-hardware-surfaces",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_unrepresentable_hardware_behavior_catalog.py",
                "BEHAVIORS",
                UNREPRESENTABLE_BEHAVIOR_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-unrepresentable-hardware-behavior-python",
        ),
        runner_batch=None,
        oracle=(
            "record-field-absence-audit",
            "numeric-domain-whitelist-staleness-audit",
            "typed-preparation-rejection",
            "safe-executable-alternative-named",
        ),
        activation_gate=(
            "host-gate-only",
            "owner-backed-readonly-surface-required",
        ),
        blocker=(
            "the current NCC record has no queue resident occupancy field, "
            "and the SPM contract has no physical bank/port mapping or "
            "bank-specific counter; ArgMin domains outside the four explicit "
            "F16 rows have no typed case or coherent value/index oracle"
        ),
    ),
    PendingCalibrationFamily(
        key="argmin-tie-and-nan-domain",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/wafer_instruction_family_catalog.py",
                "CATALOG",
                ARGMIN_DOMAIN_CASES,
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-instruction-family-{case}"
            for case in ARGMIN_DOMAIN_CASES
        ),
        no_card_ctests=(
            "wafer-instruction-family-catalog-python",
            "wafer-runtime-instruction-family-probe-no-card",
        ),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "three-distinguishing-inputs-per-domain",
            "coherent-value-index-pair",
            "tie-or-nan-behavior-classification",
            "internal-padding-span-guards-terminal-and-cleanup",
        ),
        activation_gate=("one-case-per-process", "bounded-observation-only"),
    ),
    PendingCalibrationFamily(
        key="unpool-repeated-overlap-collision",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/wafer_instruction_family_catalog.py",
                "CATALOG",
                UNPOOL_COLLISION_CASES,
            ),
        ),
        board_ctests=tuple(
            f"wafer-board-instruction-family-{case}"
            for case in UNPOOL_COLLISION_CASES
        ),
        no_card_ctests=(
            "wafer-instruction-family-catalog-python",
            "wafer-runtime-instruction-family-probe-no-card",
        ),
        runner_batch="pending-execution-boundaries",
        oracle=(
            "sample-rotated-four-source-sentinels",
            "exact-5-3-2-0-indexed-pool-auxiliary",
            "consistent-nonempty-source-subset-per-64-channel-target",
            "logical-nontarget-physical-tail-slot-guards",
            "matching-terminal-and-cleanup",
        ),
        activation_gate=(
            "one-case-per-process",
            "bounded-collision-observation-only",
            "no-unspecified-winner-rule",
        ),
    ),
    PendingCalibrationFamily(
        key="native-concat-hw-isolated-requalification",
        disposition=PENDING_BOARD,
        execution_scope="rank-one-isolated-final",
        bindings=(
            _binding(
                "test/Board/"
                "wafer_datamove_extended_calibration_catalog.py",
                "CASES_BY_NAME",
                ("datamove-raw-concat-hw-n2-2x5-3x7-c65",),
                "@mapping",
            ),
        ),
        board_ctests=("wafer-board-datamove-native-concat-hw-isolated",),
        no_card_ctests=(
            "wafer-runtime-datamove-extended-calibration-probe-no-card",
        ),
        runner_batch=None,
        oracle=(
            "native-c-w-h-controls-first",
            "bounded-output-span-and-guards",
            "terminal-and-cleanup",
        ),
        activation_gate=("must-run-last", "stop-on-timeout"),
    ),
)


FAMILIES_BY_KEY = {family.key: family for family in FAMILIES}


def validate_inventory() -> None:
    if len(FAMILIES_BY_KEY) != len(FAMILIES):
        raise RuntimeError("pending calibration family keys are not unique")
    for family in FAMILIES:
        if family.disposition not in {
            PENDING_BOARD,
            HOST_NEGATIVE,
            BLOCKED_EXTERNAL,
        }:
            raise RuntimeError(f"{family.key}: invalid disposition")
        if not family.oracle or not family.activation_gate:
            raise RuntimeError(f"{family.key}: oracle or activation gate is empty")
        if family.disposition == PENDING_BOARD:
            if (
                not family.bindings
                or not family.board_ctests
                or not family.no_card_ctests
            ):
                raise RuntimeError(
                    f"{family.key}: executable family lacks concrete assets"
                )
            if family.blocker is not None:
                raise RuntimeError(
                    f"{family.key}: executable family unexpectedly has blocker"
                )
        else:
            if family.board_ctests or family.runner_batch is not None:
                raise RuntimeError(
                    f"{family.key}: non-board family exposes board execution"
                )
            if (
                not family.bindings
                or not family.no_card_ctests
                or not family.blocker
            ):
                raise RuntimeError(
                    f"{family.key}: non-board family lacks a concrete "
                    "host gate or blocker"
                )


validate_inventory()
