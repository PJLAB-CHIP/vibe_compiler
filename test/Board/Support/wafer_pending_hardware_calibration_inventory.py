#!/usr/bin/env python3
"""Executable inventory for calibration rows that still need board evidence."""

from __future__ import annotations

import dataclasses

import wafer_board_complete_tile_add_runner as complete_tile_add
import wafer_board_complete_tile_barrier_probe_runner as complete_tile_barrier
import wafer_collective_traffic_behavior_catalog as collective_traffic
import wafer_engine_pipeline_characterization_catalog as engine_pipeline
import wafer_spm_sustained_conflict_catalog as spm_sustained
import wafer_transport_pmu_calibration_catalog as transport
import wafer_unrepresentable_hardware_behavior_catalog as unrepresentable
import wafer_worker_memory_contention_characterization_catalog as contention
import wafer_worker_placement_characterization_catalog as worker_placement


PENDING_BOARD = "pending-board"
HOST_NEGATIVE = "host-negative"
BLOCKED_EXTERNAL = "blocked-external"
BLOCKED_CURRENT_PIPELINE = "blocked-current-pipeline"
CURRENT_PIPELINE_BLOCKER = (
    "current none controller is unavailable until baseline current-IR "
    "integration recreates source-to-package and no-card gates"
)


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
    host_ctests: tuple[str, ...] = ()
    blocker: str | None = None


def _binding(
    asset: str,
    symbol: str,
    identifiers: tuple[str, ...],
    identifier_field: str = "name",
) -> CatalogBinding:
    return CatalogBinding(asset, symbol, identifiers, identifier_field)


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
COMPLETE_TILE_BARRIER_CASES = tuple(
    case.name for case in complete_tile_barrier.BARRIER_POSITIVE_CASES
)
COMPLETE_TILE_ADD_CASES = tuple(
    case.key for case in complete_tile_add.RUNTIME_LAUNCH_CALIBRATION_CASES
)
COLLECTIVE_TRAFFIC_CASES = tuple(collective_traffic.CASE_KEYS)
DTE_FOUR_SOURCE_FANIN_CASE_KEYS = (
    transport.MODE_NAMES[transport.FOUR_SOURCE_FANIN_MODE],
)
DTE_RAW_MULTIDEST_CASE_KEYS = tuple(
    case.key for case in transport.RAW_MULTIDEST_CASES
)
PENDING_DTE_CASE_KEYS = (
    *DTE_FOUR_SOURCE_FANIN_CASE_KEYS,
    *DTE_RAW_MULTIDEST_CASE_KEYS,
)
PENDING_DTE_CASE_SPECS = (
    (transport.FOUR_SOURCE_FANIN_MODE, DTE_FOUR_SOURCE_FANIN_CASE_KEYS[0]),
    *tuple(
        (case.mode, case.key)
        for case in transport.RAW_MULTIDEST_CASES
    ),
)
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
        engine_pipeline.Disposition.FAIL_CLOSED_UNSAFE_WINDOW,
    }
)
ENGINE_PIPELINE_PENDING_PRODUCTION_CELL_KEYS = tuple(
    cell.key
    for cell in engine_pipeline.THREE_STAGE_PENDING_BOARD_CELLS
    if (
        cell.disposition
        == engine_pipeline.Disposition.PENDING_CONFIGURED_BOARD
    )
)
SPM_SUSTAINED_CELL_KEYS = tuple(cell.key for cell in spm_sustained.CELLS)
SPM_SUSTAINED_GROUP_KEYS = tuple(group.key for group in spm_sustained.GROUPS)
DDR_ACTIVE_TILE_CASE_KEYS = tuple(
    case.key for case in contention.DDR_ACTIVE_TILE_CASES
)
DDR_ACTIVE_TILE_GROUP_KEYS = tuple(contention.DDR_ACTIVE_TILE_GROUP_KEYS)
MEMORY_UNREPRESENTABLE_BOUNDARY_KEYS = tuple(
    boundary.key for boundary in contention.TYPED_BOUNDARIES
)
UNREPRESENTABLE_BEHAVIOR_KEYS = tuple(
    behavior.key for behavior in unrepresentable.BEHAVIORS
)


FAMILIES = (
    PendingCalibrationFamily(
        key="collective-algorithm-characterization",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="current-global-source-contract",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_collective_hardware_characterization_catalog.py",
                "CASES",
                COLLECTIVE_CHARACTERIZATION_CASES,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(),
        runner_batch=None,
        oracle=(
            "current-global-source-and-exact-output",
            "cross-Tile-message-matching",
            "algorithm-specific-graph",
        ),
        activation_gate=("current-collective-algorithm-selection",),
        host_ctests=("wafer-collective-algorithm-source-contract",),
        blocker=(
            "normal global lowering has no current direct/ring/tree "
            "algorithm comparison interface"
        ),
    ),
    PendingCalibrationFamily(
        key="collective-traffic-semantics",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="current-global-source-contract",
        bindings=(
            _binding(
                "test/Board/Support/wafer_collective_traffic_behavior_catalog.py",
                "CASES",
                COLLECTIVE_TRAFFIC_CASES,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(),
        runner_batch=None,
        oracle=(
            "current-global-source",
            "full-output-exact",
            "source-graph-and-min-hop-demand-record",
        ),
        activation_gate=(
            "current-global-collective-lowering",
        ),
        host_ctests=("wafer-collective-traffic-source-contract",),
        blocker=(
            "current global lowering must produce the typed Direct-DTE "
            "graph before no-card or board execution is valid"
        ),
    ),
    PendingCalibrationFamily(
        key="complete-tile-barrier",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="complete-Tile-domain",
        bindings=(
            _binding(
                "test/Board/Support/wafer_board_complete_tile_barrier_probe_runner.py",
                "BARRIER_POSITIVE_CASES",
                COMPLETE_TILE_BARRIER_CASES,
                "name",
            ),
        ),
        board_ctests=("wafer-board-complete-tile-barrier-probe",),
        no_card_ctests=(
            "wafer-runtime-complete-tile-barrier-probe-no-card",
        ),
        runner_batch="synchronization",
        oracle=(
            "two-epoch-Tile-specific-markers",
            "zero-mismatch-and-zero-crosstalk",
            "terminal-status-and-cleanup",
        ),
        activation_gate=(
            "exact-sixteen-Tile-domain",
            "subgroups-rejected-before-device-submission",
        ),
        host_ctests=("wafer-complete-tile-barrier-contract",),
    ),
    PendingCalibrationFamily(
        key="complete-tile-add",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="complete-Tile-domain",
        bindings=(
            _binding(
                "test/Board/Support/wafer_board_complete_tile_add_runner.py",
                "RUNTIME_LAUNCH_CALIBRATION_CASES",
                COMPLETE_TILE_ADD_CASES,
                "key",
            ),
        ),
        board_ctests=(
            "wafer-board-complete-tile-add",
            "wafer-board-complete-tile-add-profile",
        ),
        no_card_ctests=(
            "wafer-runtime-complete-tile-add-no-card",
            "wafer-runtime-complete-tile-add-profile-no-card",
        ),
        runner_batch="runtime-execution",
        oracle=(
            "complete-f16-output",
            "exact-complete-Tile-runtime-domain",
            "ordinary-and-profile-package-byte-equality",
            "Primary-Count-Trace-profile-report",
        ),
        activation_gate=(
            "sixteen-Tile-grid-launch",
            "two-ordinary-runs-or-one-bounded-profile-collection",
            "terminal-local-drain-and-normal-cleanup",
        ),
        host_ctests=("wafer-complete-tile-add-profile-contract",),
    ),
    PendingCalibrationFamily(
        key="direct-dte-raw-multidestination-and-fanin",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="complete-Tile-domain-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/Support/wafer_transport_pmu_calibration_catalog.py",
                "CONTRACT_CASES",
                DTE_FOUR_SOURCE_FANIN_CASE_KEYS,
            ),
            _binding(
                "test/Board/Support/wafer_transport_pmu_calibration_catalog.py",
                "RAW_MULTIDEST_CASES",
                DTE_RAW_MULTIDEST_CASE_KEYS,
                "key",
            ),
        ),
        board_ctests=("wafer-board-dte-ncc-execution-probe",),
        no_card_ctests=("wafer-runtime-dte-ncc-execution-probe-no-card",),
        runner_batch="transport",
        oracle=(
            "owner-backed-raw-dte-register-programming",
            "complete-Tile-exact-or-semantics-classified-output-and-guards",
            "accepted-send-receive-counts-terminal-and-cleanup",
            "final-elf-register-store-and-launch-contract",
        ),
        activation_gate=(
            "one-mode-per-process",
            "four-source-fanin-and-raw-fanout-two-four-eight-fifteen",
            "raw-destination-count-encoding-remains-board-observation",
        ),
    ),
    PendingCalibrationFamily(
        key="collective-traffic-unsupported-surfaces",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/Support/wafer_collective_traffic_behavior_catalog.py",
                "COVERAGE_ITEMS",
                COLLECTIVE_TRAFFIC_BLOCKED_COVERAGE_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(),
        runner_batch=None,
        oracle=(
            "typed-surface-or-abi-rejection",
            "no-sequential-unicast-as-concurrent-hotspot",
            "no-logical-endpoint-as-physical-route",
        ),
        activation_gate=("host-gate-only", "promotion-surface-required"),
        blocker=(
            "eight/fifteen-source fan-in exceeds the current receive-state "
            "surface; device phase basis, physical route report, ragged/"
            "same-buffer collective surfaces, and multi-card transport "
            "remain absent"
        ),
        host_ctests=("wafer-collective-traffic-source-contract",),
    ),
    PendingCalibrationFamily(
        key="single-engine-and-engine-pair-characterization",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-complete-activation-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_engine_pipeline_characterization_catalog.py",
                "ALL_CELLS",
                ENGINE_PIPELINE_BOARD_CELL_KEYS,
                "key",
            ),
        ),
        board_ctests=(
            "wafer-board-engine-pipeline-characterization-probe",
            "wafer-board-ne-tail-throughput-probe",
        ),
        no_card_ctests=(
            "wafer-runtime-engine-pipeline-characterization-no-card",
            "wafer-runtime-ne-tail-throughput-no-card",
        ),
        runner_batch="engine-execution",
        host_ctests=(
            "wafer-engine-pipeline-catalog",
            "wafer-ne-tail-throughput-catalog",
        ),
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
        key="production-software-pipeline-configured-board-qualification",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="configured-board-source-package-qualification",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_engine_pipeline_characterization_catalog.py",
                "ALL_CELLS",
                ENGINE_PIPELINE_PENDING_PRODUCTION_CELL_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(),
        runner_batch=None,
        oracle=(
            "manifest-bound-accepted-instr-fixed-slot-qualification",
            "typed-worker-and-current-ir-completion-proof",
            "same-source-baseline-winner-full-correctness",
            "no-handwritten-double-slot-as-production-vertical",
        ),
        activation_gate=(
            "host-exact-package-model-no-card-closed",
            "fresh-configured-board-session-required",
        ),
        blocker=(
            "the compiler-owned multi-buffer producer and host gates are "
            "closed; only fresh configured-board baseline/winner "
            "qualification remains, and no repo-owned board test collection asset "
            "currently supplies that evidence"
        ),
        host_ctests=("wafer-engine-pipeline-catalog",),
    ),
    PendingCalibrationFamily(
        key="queue-saturation-response",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/Support/wafer_board_ncc_execution_probe_runner.py",
                "QUEUE_SATURATION_CASES",
                QUEUE_SATURATION_CASES,
            ),
        ),
        board_ctests=(),
        no_card_ctests=("wafer-runtime-ncc-execution-probe-no-card",),
        runner_batch=None,
        oracle=(
            "per-issue-acceptance-cycle-and-return",
            "pre-wait-control-snapshot",
            "instruction-count-full-result-and-guards",
            "matching-wait-terminal-and-cleanup",
        ),
        activation_gate=(
            "one-case-per-process",
            "documented-depth-minus-one-depth-depth-plus-one-only",
            "no-retry-reset-or-power",
        ),
        host_ctests=(
            "wafer-ncc-protocol-contract",
            "wafer-ncc-probe-plan-contract",
        ),
    ),
    PendingCalibrationFamily(
        key="worker-wait-scope-exclusion",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/Support/wafer_board_ncc_execution_probe_runner.py",
                "WORKER_WAIT_SCOPE_CASES",
                WORKER_WAIT_SCOPE_CASES,
            ),
        ),
        board_ctests=(),
        no_card_ctests=("wafer-runtime-ncc-execution-probe-no-card",),
        runner_batch=None,
        oracle=(
            "target-pending-before-wait",
            "boundary-worker-control-and-result",
            "matching-safety-drain",
            "full-result-guards-count-and-cleanup",
        ),
        activation_gate=("one-case-per-process", "pending-snapshot-required"),
        host_ctests=(
            "wafer-ncc-protocol-contract",
            "wafer-ncc-probe-plan-contract",
        ),
    ),
    PendingCalibrationFamily(
        key="worker-subset-join-exclusion",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/Support/wafer_board_ncc_execution_probe_runner.py",
                "WORKER_SUBSET_SCOPE_CASES",
                WORKER_SUBSET_SCOPE_CASES,
            ),
        ),
        board_ctests=(),
        no_card_ctests=("wafer-runtime-ncc-execution-probe-no-card",),
        runner_batch=None,
        oracle=(
            "target-pending-before-join",
            "include-exclude-mask-pair",
            "boundary-worker-control-and-result",
            "final-all-worker-drain-guards-count-and-cleanup",
        ),
        activation_gate=("one-case-per-process", "pending-snapshot-required"),
        host_ctests=(
            "wafer-ncc-protocol-contract",
            "wafer-ncc-probe-plan-contract",
        ),
    ),
    PendingCalibrationFamily(
        key="worker-placement-and-bounded-progress",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-matched-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_worker_placement_characterization_catalog.py",
                "CASES",
                WORKER_PLACEMENT_CASE_KEYS,
                "key",
            ),
        ),
        board_ctests=("wafer-board-worker-placement-probe",),
        no_card_ctests=(
            "wafer-runtime-worker-placement-probe-no-card",
        ),
        runner_batch="worker-execution",
        oracle=(
            "fixed-total-work-placement-and-matched-progress-controls",
            "full-output-prefix-suffix-guards-and-routing-counts",
            "per-worker-instruction-blocking-and-global-pmu",
            "observer-boundary-target-safety-drain-terminal-and-cleanup",
            "partial-accept-single-bounded-cleanup-or-poisoned-stop",
        ),
        activation_gate=(
            "one-complete-matched-group-per-ctest",
            "three-serial-repeats",
            "no-absolute-worker-timestamp-or-arbiter-name",
            "no-retry-after-cleanup-deadline",
        ),
        host_ctests=("wafer-worker-placement-contract",),
    ),
    PendingCalibrationFamily(
        key="worker-placement-unrepresentable-surfaces",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_worker_placement_characterization_catalog.py",
                "BOUNDARIES",
                WORKER_PLACEMENT_BOUNDARY_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(),
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
        host_ctests=("wafer-worker-placement-contract",),
    ),
    PendingCalibrationFamily(
        key="spm-conflict-equivalence",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-plus-complete-Tile-domain-held-out",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_memory_descriptor_calibration_catalog.py",
                "PENDING_CONFLICT_EQUIVALENCE_CASES",
                (),
            ),
            _binding(
                "test/Board/Support/"
                "wafer_board_spm_cross_tile_conflict_probe_runner.py",
                "conflict_pairs",
                (),
                "@callable",
            ),
        ),
        board_ctests=(
            "wafer-board-spm-cross-tile-conflict-probe",
        ),
        no_card_ctests=(
            "wafer-runtime-spm-cross-tile-conflict-probe-no-card",
        ),
        runner_batch="memory",
        oracle=(
            "same-invocation-serial-window-pair",
            "reciprocal-order-base-workload-controls",
            "full-result-guards-count-and-completion",
            "cross-Tile-held-out",
        ),
        activation_gate=(
            "nonzero-stable-proxy-required-for-promotion",
            "no-bank-name-without-owner-backed-mapping-or-counter",
        ),
    ),
    PendingCalibrationFamily(
        key="spm-sustained-conflict-pilot",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-matched-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/Support/wafer_spm_sustained_conflict_catalog.py",
                "CELLS",
                SPM_SUSTAINED_CELL_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-runtime-spm-sustained-probe-no-card",
        ),
        runner_batch=None,
        host_ctests=("wafer-spm-sustained-contract",),
        oracle=(
            "same-invocation-candidate-control-serial-window",
            "four-way-execution-order-rotation",
            "full-owned-spm-and-gap-canary-exact",
            "pair-only-pmu-count-completion-address-and-lifecycle",
            "partial-accept-single-bounded-cleanup-before-pmu-restore",
        ),
        activation_gate=(
            "one-complete-matched-group-per-ctest",
            "stable-nonzero-heldout-required-for-cost",
            "no-physical-bank-name-from-offset-equivalence",
            "cleanup-timeout-poisons-batch-without-retry",
        ),
    ),
    PendingCalibrationFamily(
        key="ddr-conflict-equivalence",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-plus-complete-Tile-domain-held-out",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_cache_coherence_calibration_catalog.py",
                "PENDING_DDR_CONFLICT_CASES",
                (),
            ),
            _binding(
                "test/Board/Support/wafer_board_ddr_tile_offset_probe_runner.py",
                "conflict_equivalence_cases",
                (),
                "@callable",
            ),
        ),
        board_ctests=(
            "wafer-board-ddr-tile-offset-probe",
        ),
        no_card_ctests=(
            "wafer-runtime-ddr-tile-offset-probe-no-card",
        ),
        runner_batch="memory",
        oracle=(
            "same-invocation-address-schedule-order-controls",
            "actual-allocation-base",
            "full-payload-holes-guards-count-and-completion",
            "cross-Tile-held-out",
        ),
        activation_gate=(
            "explicit-batch-only",
            "no-controller-or-bank-name-without-owned-observation",
        ),
    ),
    PendingCalibrationFamily(
        key="ddr-active-tile-contention",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="complete-Tile-domain-five-point-matched-group-per-ctest",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_worker_memory_contention_characterization_catalog.py",
                "DDR_ACTIVE_TILE_CASES",
                DDR_ACTIVE_TILE_CASE_KEYS,
                "key",
            ),
        ),
        board_ctests=("wafer-board-ddr-active-tile-contention-probe",),
        no_card_ctests=(
            "wafer-runtime-ddr-active-tile-contention-no-card",
        ),
        runner_batch="memory",
        host_ctests=("wafer-worker-memory-contention-contract",),
        oracle=(
            "all-sixteen-Tiles-one-lifecycle-and-explicit-active-mask",
            "active-exact-output-guards-count-and-device-cycles",
            "inactive-Tile-output-and-pmu-canary",
            "one-two-four-eight-sixteen-Tile-matched-sweep",
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
                "test/Board/Support/"
                "wafer_worker_memory_contention_characterization_catalog.py",
                "TYPED_BOUNDARIES",
                MEMORY_UNREPRESENTABLE_BOUNDARY_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(),
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
        host_ctests=("wafer-worker-memory-contention-contract",),
    ),
    PendingCalibrationFamily(
        key="unrepresentable-hardware-surfaces",
        disposition=BLOCKED_EXTERNAL,
        execution_scope="host-fail-closed",
        bindings=(
            _binding(
                "test/Board/Support/"
                "wafer_unrepresentable_hardware_behavior_catalog.py",
                "BEHAVIORS",
                UNREPRESENTABLE_BEHAVIOR_KEYS,
                "key",
            ),
        ),
        board_ctests=(),
        no_card_ctests=(),
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
            "and the owner-backed SPM port counters have no physical bank/"
            "port mapping or bank-specific attribution; ArgMin domains "
            "outside the four explicit F16 rows have no typed case or "
            "coherent value/index oracle"
        ),
        host_ctests=("wafer-unrepresentable-hardware-catalog",),
    ),
    PendingCalibrationFamily(
        key="argmin-tie-and-nan-domain",
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/Support/wafer_instruction_family_catalog.py",
                "CATALOG",
                ARGMIN_DOMAIN_CASES,
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-runtime-instruction-family-probe-no-card",
        ),
        runner_batch=None,
        host_ctests=("wafer-instruction-family-catalog",),
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
        disposition=BLOCKED_CURRENT_PIPELINE,
        blocker=CURRENT_PIPELINE_BLOCKER,
        execution_scope="program-local-isolated-per-case",
        bindings=(
            _binding(
                "test/Board/Support/wafer_instruction_family_catalog.py",
                "CATALOG",
                UNPOOL_COLLISION_CASES,
            ),
        ),
        board_ctests=(),
        no_card_ctests=(
            "wafer-runtime-instruction-family-probe-no-card",
        ),
        runner_batch=None,
        host_ctests=("wafer-instruction-family-catalog",),
        oracle=(
            "sample-rotated-four-source-sentinels",
            "exact-5-3-2-0-indexed-pool-auxiliary",
            "indexed-zero-fill-baseline-or-mask-nonempty-source-subset",
            "logical-nontarget-physical-tail-slot-guards",
            "matching-terminal-and-cleanup",
        ),
        activation_gate=(
            "one-case-per-process",
            "bounded-collision-observation-only",
            "no-unspecified-winner-rule",
        ),
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
            BLOCKED_CURRENT_PIPELINE,
        }:
            raise RuntimeError(f"{family.key}: invalid disposition")
        if not family.oracle or not family.activation_gate:
            raise RuntimeError(f"{family.key}: oracle or activation gate is empty")
        if family.disposition == PENDING_BOARD:
            if not family.bindings or not family.no_card_ctests:
                raise RuntimeError(
                    f"{family.key}: pending family lacks current no-card assets"
                )
            if family.runner_batch is not None and not family.board_ctests:
                raise RuntimeError(
                    f"{family.key}: runner batch has no registered Board CTest"
                )
            if family.blocker is not None:
                raise RuntimeError(
                    f"{family.key}: executable family unexpectedly has blocker"
                )
        elif family.disposition == BLOCKED_CURRENT_PIPELINE:
            if not family.bindings or not family.no_card_ctests:
                raise RuntimeError(
                    f"{family.key}: pipeline-blocked family lacks retained "
                    "no-card test identities"
                )
            if family.blocker != CURRENT_PIPELINE_BLOCKER:
                raise RuntimeError(
                    f"{family.key}: pipeline-blocked family has the wrong blocker"
                )
        else:
            if family.board_ctests or family.runner_batch is not None:
                raise RuntimeError(
                    f"{family.key}: non-board family exposes board execution"
                )
            if (
                not family.bindings
                or not (family.host_ctests or family.no_card_ctests)
                or not family.blocker
            ):
                raise RuntimeError(
                    f"{family.key}: non-board family lacks a concrete "
                    "host gate or blocker"
                )


validate_inventory()
