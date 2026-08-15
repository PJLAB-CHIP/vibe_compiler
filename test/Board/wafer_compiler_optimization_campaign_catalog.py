#!/usr/bin/env python3
"""Typed coverage contract for the production-compiler board campaign.

The catalog groups choices by board-observable behavior, not by pass name.  A
single paired package case may therefore account for several implementation
mechanisms, while exact legality and rejection proofs stay on the host.
"""

from __future__ import annotations

import dataclasses
import enum
from collections import defaultdict


class AxisDisposition(str, enum.Enum):
    NEW_PAIRED_PACKAGE_BOARD_FAMILY = "new-paired-package-board-family"
    EXISTING_BOARD_FAMILY = "existing-board-family"
    HOST_ONLY_EXACT_NEGATIVE = "host-only-exact-negative"
    PENDING_CONFIGURED_BOARD = "pending-configured-board"


class SourceKind(str, enum.Enum):
    STABLEHLO_SOURCE_PROGRAM = "stablehlo-source-program"
    VERIFIED_STRUCTURED_SOURCE_PROGRAM = "verified-structured-source-program"


class BoardBatch(str, enum.Enum):
    RANK_ONE_LOCAL = "rank-one-local"
    RANK_SIXTEEN_COMMUNICATION = "rank-sixteen-communication"


class PackagePairRequirement(str, enum.Enum):
    REQUIRED_SAME_SEMANTICS = "required-same-semantics-control-and-optimized"
    EXISTING_SINGLE_PACKAGE_BASELINE = "existing-single-package-baseline"


class StructuralOracleKind(str, enum.Enum):
    TARGET_CALL_RELATION = "target-call-multiset-order-and-control-flow"
    PHYSICAL_ROUTE_RELATION = "target-gemm-and-gather-scatter-call-relation"
    TRANSPORT_CONTRACT_RELATION = "manifest-transport-and-target-call-relation"


class NumericOracleKind(str, enum.Enum):
    FULL_EXACT = "full-output-exact"
    FULL_FLOATING_TOLERANCE = "full-output-floating-tolerance"


class PerformanceOracleKind(str, enum.Enum):
    BALANCED_HOST_PROCESS_OBSERVATION = "balanced-host-process-observation-only"
    REPEATED_LIFECYCLE_OBSERVATION = "repeated-lifecycle-observation-only"


class ExecutableEvidenceCapability(str, enum.Enum):
    """Machine-checkable evidence emitted or consumed by an execution asset."""

    PACKAGE_PUBLIC_SEMANTICS_EQUAL = "package-public-semantics-equal"
    MANIFEST_TRANSPORT_CONTRACT = "manifest-transport-contract"
    TARGET_CALL_PRESENCE = "target-call-presence"
    TARGET_CALL_COUNT_RELATION = "target-call-count-relation"
    TARGET_CALL_ORDER_RELATION = "target-call-order-relation"
    TARGET_WORKSPACE_RELATION = "target-workspace-relation"
    SCHEDULER_BODY_DIFFERENCE = "scheduler-body-difference"
    FULL_OUTPUT_EXACT = "full-output-exact"
    FULL_OUTPUT_FLOATING_TOLERANCE = "full-output-floating-tolerance"
    COMPLEMENT_PREFILL_CANARY = "complement-prefill-canary"
    ALL_RANK_STATUS = "all-rank-status"
    BOUNDED_LIFECYCLE = "bounded-lifecycle"
    BALANCED_PAIR_ORDER = "balanced-pair-order"
    REPEATED_LIFECYCLE = "repeated-lifecycle"


STRUCTURAL_EVIDENCE_CAPABILITIES = frozenset(
    {
        ExecutableEvidenceCapability.MANIFEST_TRANSPORT_CONTRACT,
        ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
        ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
        ExecutableEvidenceCapability.TARGET_CALL_ORDER_RELATION,
        ExecutableEvidenceCapability.TARGET_WORKSPACE_RELATION,
        ExecutableEvidenceCapability.SCHEDULER_BODY_DIFFERENCE,
    }
)
@dataclasses.dataclass(frozen=True)
class OracleContract:
    structural_kind: StructuralOracleKind
    structural_checks: tuple[str, ...]
    numeric_kind: NumericOracleKind
    numeric_checks: tuple[str, ...]
    performance_kind: PerformanceOracleKind
    performance_checks: tuple[str, ...]
    executable_capabilities: frozenset[ExecutableEvidenceCapability]
    performance_is_promotion_evidence: bool = False

    @property
    def is_strong(self) -> bool:
        """Whether typed executable evidence covers structure, value and life cycle."""

        capabilities = self.executable_capabilities
        required_execution = {
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION: (
                ExecutableEvidenceCapability.BALANCED_PAIR_ORDER
            ),
            PerformanceOracleKind.REPEATED_LIFECYCLE_OBSERVATION: (
                ExecutableEvidenceCapability.REPEATED_LIFECYCLE
            ),
        }[self.performance_kind]
        numeric_evidence = (
            self.numeric_kind == NumericOracleKind.FULL_EXACT
            and ExecutableEvidenceCapability.FULL_OUTPUT_EXACT in capabilities
        ) or (
            self.numeric_kind
            == NumericOracleKind.FULL_FLOATING_TOLERANCE
            and ExecutableEvidenceCapability.FULL_OUTPUT_FLOATING_TOLERANCE
            in capabilities
        )
        return (
            bool(capabilities & STRUCTURAL_EVIDENCE_CAPABILITIES)
            and numeric_evidence
            and ExecutableEvidenceCapability.BOUNDED_LIFECYCLE in capabilities
            and required_execution in capabilities
            and not self.performance_is_promotion_evidence
        )


@dataclasses.dataclass(frozen=True)
class DriverOracleBinding:
    structural_oracle_symbol: str
    payload_factory_marker: str
    structural_capability_markers: tuple[
        tuple[ExecutableEvidenceCapability, str], ...
    ]


@dataclasses.dataclass(frozen=True)
class CampaignCase:
    key: str
    family: str
    priority: str
    disposition: AxisDisposition
    source_kind: SourceKind
    rank_count: int
    board_batch: BoardBatch
    board_order: int
    pair_requirement: PackagePairRequirement
    package_roles: tuple[str, ...]
    pair_contract: tuple[str, ...]
    oracle: OracleContract
    completion_contract: tuple[str, ...]
    execution_asset: str
    existing_assets: tuple[str, ...] = ()
    driver_binding: DriverOracleBinding | None = None

    @property
    def requires_pair(self) -> bool:
        return (
            self.pair_requirement
            == PackagePairRequirement.REQUIRED_SAME_SEMANTICS
        )


@dataclasses.dataclass(frozen=True)
class OptimizationAxis:
    key: str
    pipeline_stage: str
    disposition: AxisDisposition
    evidence_key: str
    board_observables: tuple[str, ...]
    reason: str
    production_owner: "SourceAnchor"
    host_assets: tuple[str, ...] = ()

    @property
    def is_board_mapped(self) -> bool:
        return self.disposition in BOARD_DISPOSITIONS


BOARD_DISPOSITIONS = frozenset(
    {
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        AxisDisposition.EXISTING_BOARD_FAMILY,
    }
)
PAIRED_CAMPAIGN_DRIVER = (
    "test/Board/wafer_board_compiler_optimization_campaign_test.py"
)


@dataclasses.dataclass(frozen=True)
class SourceAnchor:
    asset: str
    markers: tuple[str, ...]


def _anchor(asset: str, *markers: str) -> SourceAnchor:
    return SourceAnchor(asset=asset, markers=markers)


PRODUCTION_PIPELINE_ANCHORS = (
    _anchor(
        "lib/Wafer/Compiler/CompilationOrchestration.cpp",
        "buildStablehloToLinalgPipeline",
        "stageTargetPackage",
    ),
    _anchor(
        "lib/Wafer/Compiler/RankCandidateSearch.cpp",
        "RankCandidateSearchSession::create",
    ),
    _anchor(
        "lib/Wafer/Compiler/RankCandidateEvaluation.cpp",
        "beginRankCandidateEvaluation",
        "advanceRankCandidateEvaluation",
    ),
    _anchor(
        "lib/Wafer/Compiler/RankCandidateSelection.cpp",
        "selectEvaluatedRankCandidate",
    ),
    _anchor(
        "lib/Wafer/Pipelines/Pipelines.cpp",
        "buildLowerTileRegionToInstrPipeline",
        "addAssignDDROffsetsPass",
    ),
)


PAIRED_DRIVER_BINDINGS = {
    "reciprocal-implementation": DriverOracleBinding(
        "reciprocal_oracle",
        "def reciprocal_payloads",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                "require_call",
            ),
        ),
    ),
    "f16-common-factor": DriverOracleBinding(
        "f16_factor_oracle",
        "def f16_factor_payloads",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                "require_call",
            ),
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "winner_mul < baseline_mul",
            ),
        ),
    ),
    "resident-fanout-share": DriverOracleBinding(
        "resident_fanout_oracle",
        "def resident_fanout_payloads",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                "require_call",
            ),
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "winner_movement < baseline_movement",
            ),
        ),
    ),
    "consumer-local-recompute": DriverOracleBinding(
        "recompute_oracle",
        "def recompute_payloads",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "winner_mul > baseline_mul",
            ),
            (
                ExecutableEvidenceCapability.TARGET_WORKSPACE_RELATION,
                "winner.workspace_bytes < baseline.workspace_bytes",
            ),
        ),
    ),
    "long-steady-elementwise-add": DriverOracleBinding(
        "long_steady_add_oracle",
        "def long_steady_add_payloads",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                "require_call",
            ),
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "count_fragment(winner, fragment)",
            ),
            (
                ExecutableEvidenceCapability.SCHEDULER_BODY_DIFFERENCE,
                "baseline.scheduler_body_sha256 == winner.scheduler_body_sha256",
            ),
        ),
    ),
    "ready-order-movement-first": DriverOracleBinding(
        "ready_order_oracle",
        "def ready_order_payloads",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "without_fences(baseline.counts)",
            ),
            (
                ExecutableEvidenceCapability.TARGET_CALL_ORDER_RELATION,
                "winner.straight_line_calls",
            ),
        ),
    ),
    "gemm-aligned-physical-route": DriverOracleBinding(
        "gemm_route_oracle",
        "gemm_payloads(64, 128, 128)",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                "require_call",
            ),
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "winner_layout >= baseline_layout",
            ),
        ),
    ),
    "gemm-tail-physical-route": DriverOracleBinding(
        "gemm_route_oracle",
        "gemm_payloads(65, 129, 129)",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                "require_call",
            ),
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "winner_layout >= baseline_layout",
            ),
        ),
    ),
    "tree-all-reduce": DriverOracleBinding(
        "collective_oracle",
        "def all_reduce_payloads",
        (
            (
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                "require_call",
            ),
            (
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                "winner_prepares < baseline_prepares",
            ),
            (
                ExecutableEvidenceCapability.TARGET_WORKSPACE_RELATION,
                "winner.workspace_bytes < baseline.workspace_bytes",
            ),
            (
                ExecutableEvidenceCapability.SCHEDULER_BODY_DIFFERENCE,
                "baseline.scheduler_body_sha256 == winner.scheduler_body_sha256",
            ),
        ),
    ),
}

PAIRED_COMMON_CAPABILITIES = frozenset(
    {
        ExecutableEvidenceCapability.PACKAGE_PUBLIC_SEMANTICS_EQUAL,
        ExecutableEvidenceCapability.FULL_OUTPUT_EXACT,
        ExecutableEvidenceCapability.COMPLEMENT_PREFILL_CANARY,
        ExecutableEvidenceCapability.BOUNDED_LIFECYCLE,
        ExecutableEvidenceCapability.BALANCED_PAIR_ORDER,
    }
)

EXECUTION_CAPABILITY_ANCHORS = {
    ExecutableEvidenceCapability.PACKAGE_PUBLIC_SEMANTICS_EQUAL: _anchor(
        PAIRED_CAMPAIGN_DRIVER,
        "normalized_manifest(manifests[\"baseline\"])",
    ),
    ExecutableEvidenceCapability.FULL_OUTPUT_EXACT: _anchor(
        PAIRED_CAMPAIGN_DRIVER,
        "else torch_reference.EXACT",
        "assert_raw_capture_matches",
    ),
    ExecutableEvidenceCapability.FULL_OUTPUT_FLOATING_TOLERANCE: _anchor(
        PAIRED_CAMPAIGN_DRIVER,
        "torch_reference.ComparisonPolicy(",
        "assert_raw_capture_matches",
    ),
    ExecutableEvidenceCapability.COMPLEMENT_PREFILL_CANARY: _anchor(
        "tools/wafer-run/WaferRunBoardIO.cpp",
        "std::fill(bytes.begin(), bytes.end(), UINT8_C(0xa5))",
    ),
    ExecutableEvidenceCapability.ALL_RANK_STATUS: _anchor(
        PAIRED_CAMPAIGN_DRIVER,
        "\"--direct-dte-status-abi\"",
        "actual_completions",
    ),
    ExecutableEvidenceCapability.BOUNDED_LIFECYCLE: _anchor(
        PAIRED_CAMPAIGN_DRIVER,
        "timeout_seconds=(",
        "not retry or invoke reset/power operations",
    ),
    ExecutableEvidenceCapability.BALANCED_PAIR_ORDER: _anchor(
        PAIRED_CAMPAIGN_DRIVER,
        "def balanced_order",
    ),
    ExecutableEvidenceCapability.REPEATED_LIFECYCLE: _anchor(
        "test/Board/wafer_board_direct_dte_collective_test.py",
        "for iteration in range(args.repeat)",
        "timeout_seconds=(",
    ),
}


def _oracle(
    structural_kind: StructuralOracleKind,
    structural_checks: tuple[str, ...],
    numeric_kind: NumericOracleKind,
    numeric_checks: tuple[str, ...],
    performance_kind: PerformanceOracleKind,
    performance_checks: tuple[str, ...],
    executable_capabilities: frozenset[ExecutableEvidenceCapability],
) -> OracleContract:
    return OracleContract(
        structural_kind=structural_kind,
        structural_checks=structural_checks,
        numeric_kind=numeric_kind,
        numeric_checks=numeric_checks,
        performance_kind=performance_kind,
        performance_checks=performance_checks,
        executable_capabilities=executable_capabilities,
    )


def _paired_case(
    key: str,
    family: str,
    source_kind: SourceKind,
    rank_count: int,
    board_batch: BoardBatch,
    board_order: int,
    oracle: OracleContract,
) -> CampaignCase:
    return CampaignCase(
        key=key,
        family=family,
        priority="P0",
        disposition=AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        source_kind=source_kind,
        rank_count=rank_count,
        board_batch=board_batch,
        board_order=board_order,
        pair_requirement=PackagePairRequirement.REQUIRED_SAME_SEMANTICS,
        package_roles=("control", "optimized"),
        pair_contract=(
            "same public input/output signature, shape, dtype, and rank domain",
            "same deterministic nonzero payload and independent host oracle",
            "control blocks only the target choice; optimized uses production default",
            "compile both packages before the board batch and archive both digests",
        ),
        oracle=oracle,
        completion_contract=(
            "bounded timeout",
            "all expected rank completions",
            "full output, complement-prefill canary, and status validation",
            "normal runtime cleanup",
            "no retry, reset, or power action",
        ),
        execution_asset=PAIRED_CAMPAIGN_DRIVER,
        driver_binding=PAIRED_DRIVER_BINDINGS[key],
    )


CAMPAIGN_CASES = (
    _paired_case(
        "gemm-aligned-physical-route",
        "tile-layout-route",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        70,
        _oracle(
            StructuralOracleKind.PHYSICAL_ROUTE_RELATION,
            (
                "package manifests have equivalent host-visible resources and ABI",
                "both statically reachable scheduler bodies call GEMM",
                "the optimized scheduler bodies contain fewer gather/scatter callsites",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "sparse integer-valued f16 inputs exercise every K lane including k-1",
                "the sparse GEMM has an independently exact full output",
                "complement-prefilled write-only output detects partial writes",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
            },
        ),
    ),
    _paired_case(
        "gemm-tail-physical-route",
        "tile-layout-route",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        80,
        _oracle(
            StructuralOracleKind.PHYSICAL_ROUTE_RELATION,
            (
                "package manifests have equivalent host-visible resources and ABI",
                "both statically reachable scheduler bodies call GEMM",
                "the optimized scheduler bodies contain fewer gather/scatter callsites",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "sparse integer-valued odd-shape inputs cover every K lane including k-1",
                "the sparse odd-shape GEMM has an exact full output",
                "tail bytes and complement-prefilled full output are all checked",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
            },
        ),
    ),
    _paired_case(
        "resident-fanout-share",
        "resident-dataflow",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        30,
        _oracle(
            StructuralOracleKind.PHYSICAL_ROUTE_RELATION,
            (
                "the optimized scheduler bodies contain fewer RDMA/WDMA callsites",
                "the optimized target still calls add and multiply",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "integer or exactly representable float payload checks every fanout",
                "mixed-shape consumers use separately computed full expected outputs",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
            },
        ),
    ),
    _paired_case(
        "consumer-local-recompute",
        "resident-dataflow",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        40,
        _oracle(
            StructuralOracleKind.TARGET_CALL_RELATION,
            (
                "the optimized scheduler bodies contain more multiply callsites",
                "the optimized manifest declares fewer workspace bytes",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "independent full-output oracle covers every local recomputation",
                "distinct consumer payloads reject accidental value sharing",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                ExecutableEvidenceCapability.TARGET_WORKSPACE_RELATION,
            },
        ),
    ),
    _paired_case(
        "long-steady-elementwise-add",
        "static-fixed-slot-overlap",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        45,
        _oracle(
            StructuralOracleKind.TARGET_CALL_RELATION,
            (
                "both final scheduler bodies call elementwise add",
                "the production scheduler body differs from its reserved baseline",
                "the production body carries the generated prologue, steady loop, "
                "and epilogue structure",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "the complete 16 MiB f16 output is checked exactly",
                "complement-prefilled output detects partial writes",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                ExecutableEvidenceCapability.SCHEDULER_BODY_DIFFERENCE,
            },
        ),
    ),
    _paired_case(
        "f16-common-factor",
        "numeric-dag-and-implementation",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        20,
        _oracle(
            StructuralOracleKind.TARGET_CALL_RELATION,
            (
                "the optimized scheduler bodies have fewer multiply callsites",
                "the optimized scheduler bodies still contain an add callsite",
            ),
            NumericOracleKind.FULL_FLOATING_TOLERANCE,
            (
                "the full f16 output uses the explicit abs/rel/ULP policy",
                "signed zero compares numerically while nonfinite values fail",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            (
                PAIRED_COMMON_CAPABILITIES
                - {ExecutableEvidenceCapability.FULL_OUTPUT_EXACT}
            )
            | {
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                ExecutableEvidenceCapability.FULL_OUTPUT_FLOATING_TOLERANCE,
            },
        ),
    ),
    _paired_case(
        "reciprocal-implementation",
        "numeric-dag-and-implementation",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        10,
        _oracle(
            StructuralOracleKind.TARGET_CALL_RELATION,
            (
                "the optimized scheduler body calls reciprocal",
                "the control scheduler body calls division",
                "the optimized scheduler body contains no division callsite",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "power-of-two inputs cover signed small, large, and non-unit values",
                "both implementations compare against the same exact full output",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {ExecutableEvidenceCapability.TARGET_CALL_PRESENCE},
        ),
    ),
    _paired_case(
        "ready-order-movement-first",
        "resource-aware-order",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        1,
        BoardBatch.RANK_ONE_LOCAL,
        50,
        _oracle(
            StructuralOracleKind.TARGET_CALL_RELATION,
            (
                "non-completion target-call multisets remain equal",
                "straight-line final linked scheduler bodies prove call order",
                "the optimized straight-line call order is more movement-first",
                "the optimized target adds no completion call",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "full complement-prefilled outputs reject early-consumer races",
                "distinct source ranges expose stale or overwritten buffer values",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                ExecutableEvidenceCapability.TARGET_CALL_ORDER_RELATION,
            },
        ),
    ),
    CampaignCase(
        # Compatibility key retained for existing CTest/runner selections.  This
        # asset proves the shared Direct-DTE transport vertical; it does not
        # identify a Direct all-reduce compiler algorithm.
        key="direct-all-reduce",
        family="direct-dte-transport-evidence",
        priority="P0",
        disposition=AxisDisposition.EXISTING_BOARD_FAMILY,
        source_kind=SourceKind.STABLEHLO_SOURCE_PROGRAM,
        rank_count=16,
        board_batch=BoardBatch.RANK_SIXTEEN_COMMUNICATION,
        board_order=90,
        pair_requirement=PackagePairRequirement.EXISTING_SINGLE_PACKAGE_BASELINE,
        package_roles=("selected",),
        pair_contract=(
            "legacy direct-all-reduce key names transport evidence only",
            "existing production source-to-package Direct-DTE vertical",
            "serves as correctness and transport baseline, not a speedup claim",
        ),
        oracle=_oracle(
            StructuralOracleKind.TRANSPORT_CONTRACT_RELATION,
            (
                "schema-v7 manifest has all-and-only 16 ranked Direct-DTE entries",
                "each rank manifest entry carries the Direct-DTE status contract",
                "the executable fixture owns one rank-sharded reduction source",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "rank-distinct finite f16 input yields an independently exact full output",
                "all rank outputs and transport status resources are validated",
            ),
            PerformanceOracleKind.REPEATED_LIFECYCLE_OBSERVATION,
            (
                "the existing vertical repeats the full lifecycle",
                "no PMU, device-cycle, or comparative performance claim is inferred",
            ),
            frozenset(
                {
                    ExecutableEvidenceCapability.MANIFEST_TRANSPORT_CONTRACT,
                    ExecutableEvidenceCapability.FULL_OUTPUT_EXACT,
                    ExecutableEvidenceCapability.ALL_RANK_STATUS,
                    ExecutableEvidenceCapability.BOUNDED_LIFECYCLE,
                    ExecutableEvidenceCapability.REPEATED_LIFECYCLE,
                }
            ),
        ),
        completion_contract=(
            "bounded Direct-DTE watchdog",
            "all 16 terminal completions and status records",
            "full output validation",
            "normal runtime cleanup",
            "no retry, reset, or power action",
        ),
        execution_asset=(
            "test/Board/wafer_board_direct_dte_collective_test.py"
        ),
        existing_assets=(
            "test/Board/wafer_board_direct_dte_collective_test.py",
        ),
        driver_binding=DriverOracleBinding(
            "validate_manifest",
            "def write_raw_files",
            (
                (
                    ExecutableEvidenceCapability.MANIFEST_TRANSPORT_CONTRACT,
                    "STATUS_ABI",
                ),
            ),
        ),
    ),
    _paired_case(
        "tree-all-reduce",
        "collective-algorithms",
        SourceKind.STABLEHLO_SOURCE_PROGRAM,
        16,
        BoardBatch.RANK_SIXTEEN_COMMUNICATION,
        110,
        _oracle(
            StructuralOracleKind.TRANSPORT_CONTRACT_RELATION,
            (
                "both scheduler bodies call Direct-DTE send/receive prepare",
                "the optimized target has fewer prepare calls and workspace bytes",
                "the scheduler-body digests differ and optimized calls add",
            ),
            NumericOracleKind.FULL_EXACT,
            (
                "rank-distinct f16 values use an exact full-output reference",
                "all 16 outputs compare against the same exact full result",
            ),
            PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION,
            (
                "balanced baseline/winner host-process elapsed samples are archived",
                "the samples are not PMU, device cycles, or promotion evidence",
            ),
            PAIRED_COMMON_CAPABILITIES
            | {
                ExecutableEvidenceCapability.TARGET_CALL_PRESENCE,
                ExecutableEvidenceCapability.TARGET_CALL_COUNT_RELATION,
                ExecutableEvidenceCapability.TARGET_WORKSPACE_RELATION,
                ExecutableEvidenceCapability.SCHEDULER_BODY_DIFFERENCE,
                ExecutableEvidenceCapability.ALL_RANK_STATUS,
            },
        ),
    ),
)

CASES_BY_KEY = {case.key: case for case in CAMPAIGN_CASES}


FRONTEND_HOST_GATES = (
    "test/Frontend/normalize-stablehlo-constant-cleanup.mlir",
    "test/Frontend/lower-stablehlo-official-linalg-coverage.mlir",
)
RELATION_HOST_GATES = (
    "unittests/Analysis/PhysicalDataflow/IndexRelationTest.cpp",
)
REWRITE_HOST_GATES = (
    "unittests/Transforms/PhysicalDataflow/CandidateRewritesTest.cpp",
)
ORDER_HOST_GATES = (
    "unittests/Transforms/PhysicalDataflow/ReadyOrderTest.cpp",
)
SELECTION_HOST_GATES = (
    "unittests/Compiler/RankCandidateSearchTest.cpp",
    "unittests/Compiler/LowerRankInstrModulesTest.cpp",
    "unittests/Compiler/RankCandidateSelectionTest.cpp",
    "unittests/Compiler/RankResourceCostValidationTest.cpp",
)
MEMORY_HOST_GATES = (
    "unittests/Transforms/MemoryPlanning/LifetimeAnalysisTest.cpp",
    "unittests/Transforms/MemoryPlanning/MiniMallocPackingTest.cpp",
    "unittests/Transforms/MemoryPlanning/StaticMemoryPackingTest.cpp",
)
LOWERING_HOST_GATES = (
    "unittests/Conversion/WaferTensorProgramToTileRegionTest.cpp",
    "unittests/Transforms/Target/LowerInstrToTargetLLVMTest.cpp",
)
SOFTWARE_PIPELINE_PLAN = (
    "tasks/plans/multi-engine-software-pipelining.md",
)


PRODUCTION_OWNER_BY_AXIS = {
    "target-implementation-selection": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/StructuredOpInterfaceModels.cpp",
        "TargetImplementationKind::GenericReciprocal",
    ),
    "dependent-tiling-and-tail-coverage": _anchor(
        "lib/Wafer/Compiler/RankCandidateSearch.cpp",
        "buildStructuredTraversalProposals",
    ),
    "producer-fusion-and-relation-propagation": _anchor(
        "lib/Wafer/Conversion/WaferTensorProgramToTileRegion/TileMaterialization.cpp",
        "fuseCandidateProducerSlices",
    ),
    "physical-encoding-view-materialization": _anchor(
        "lib/Wafer/Conversion/WaferTileRegionToInstr/MovementLowering.cpp",
        "ViewReshapeLowering",
    ),
    "transfer-route-storage-realization": _anchor(
        "lib/Wafer/Conversion/WaferTileRegionToInstr/MovementSupport.cpp",
        "getRelationMovementDescriptors",
    ),
    "fixed-cx-ncx-gemm-absorption": _anchor(
        "lib/Wafer/Conversion/WaferTensorProgramToTileRegion/NamedComputeLowering.cpp",
        "getOrMaterializeStructuredInput(",
    ),
    "physical-version-reuse": _anchor(
        "lib/Wafer/Conversion/WaferTensorProgramToTileRegion/BodyEmitter.cpp",
        "TileRegionBodyEmitter::record",
    ),
    "movement-resident-cut-elimination": _anchor(
        "lib/Wafer/Compiler/RankCandidateSearch.cpp",
        "materializeCompleteRankTileResidencySibling",
    ),
    "whole-tensor-share-winner": _anchor(
        "lib/Wafer/Compiler/RankCandidateSearch.cpp",
        "CandidateTileResidencyAction::SelectiveSpill",
    ),
    "consumer-local-recompute-winner": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "materializeConsumerLocalTensorRecomputation",
    ),
    "relaxed-f16-bf16-algebra": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "NumericDomain::Floating",
    ),
    "integer-modular-reassociation": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "reassociateElementwiseExpressions",
    ),
    "integer-modular-reduction-tree": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "balanceElementwiseReductionTrees",
    ),
    "integer-modular-distribution": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "contractDistributiveExpressions",
    ),
    "integer-modular-common-factor": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "factorElementwiseExpressions",
    ),
    "static-loop-invariant-hoist": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "hoistStaticLoopInvariantOperations",
    ),
    "static-buffering-ready-order": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/ReadyOrder.cpp",
        "scheduleIndependentInstructionsByReadyOrder",
    ),
    "static-fixed-slot-overlap-selection": _anchor(
        "lib/Wafer/Compiler/RankCandidateEvaluation.cpp",
        "deriveFixedSlotAction",
    ),
    "collective-direct": _anchor(
        "lib/Wafer/Conversion/WaferTensorProgramToTileRegion/CollectiveLowering.cpp",
        "TileRegionBodyEmitter::convertAllReduce",
    ),
    "collective-ring-all-gather": _anchor(
        "lib/Wafer/Conversion/WaferTileRegionToInstr/CollectiveLowering.cpp",
        "DTEProtocolPhase::AllGatherRing",
    ),
    "collective-ordered-tree-all-reduce": _anchor(
        "lib/Wafer/Conversion/WaferTileRegionToInstr/CollectiveLowering.cpp",
        "AllReduceSchedule::Tree",
    ),
    "stablehlo-normalization-and-canonicalization": _anchor(
        "lib/Wafer/Pipelines/Pipelines.cpp",
        "buildStablehloToLinalgPipeline",
    ),
    "function-boundary-bufferization-alias": _anchor(
        "lib/Wafer/Pipelines/Pipelines.cpp",
        "addInstrFunctionBoundaryBufferizationPass",
    ),
    "index-relation-and-traversal-legality": _anchor(
        "lib/Wafer/Analysis/PhysicalDataflow/IndexRelation.cpp",
        "IndexRelation::compose",
    ),
    "candidate-rewrite-effect-and-numeric-negatives": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/CandidateRewrites.cpp",
        "isPureTensorProducer",
    ),
    "ready-order-hazard-negatives": _anchor(
        "lib/Wafer/Transforms/PhysicalDataflow/ReadyOrder.cpp",
        "scheduleIndependentInstructionsByReadyOrder",
    ),
    "resource-aware-neighbor-generation": _anchor(
        "lib/Wafer/Compiler/RankCandidateSearch.cpp",
        "buildLocalConnectionChoicePool",
    ),
    "bounded-joint-search-and-baseline-fallback": _anchor(
        "lib/Wafer/Compiler/RankCandidateSearch.cpp",
        "deriveStructuredCandidates",
    ),
    "rank-candidate-pareto-and-atomic-selection": _anchor(
        "lib/Wafer/Compiler/RankCandidateSelection.cpp",
        "selectEvaluatedRankCandidate",
    ),
    "spm-lifetime-placement-and-packing": _anchor(
        "lib/Wafer/Transforms/SPM/PlanSPMMemory.cpp",
        "planSPMMemoryModule",
    ),
    "ddr-lifetime-placement-and-packing": _anchor(
        "lib/Wafer/Transforms/DDR/PlanDDRMemory.cpp",
        "planDDRMemoryModule",
    ),
    "typed-instruction-lowering-legality": _anchor(
        "lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp",
        "createLowerInstrToTargetLLVMPass",
    ),
    "software-pipeline-queue-occupancy": _anchor(
        "lib/Wafer/Transforms/Scheduling/FixedSlotPipeline.cpp",
        "deriveStaticFixedSlotPipelineCandidate",
    ),
    "software-pipeline-spm-bank-cost": _anchor(
        "lib/Wafer/Transforms/SPM/PlanSPMMemory.cpp",
        "planSPMMemoryModule",
    ),
    "software-pipeline-ddr-bank-cost": _anchor(
        "lib/Wafer/Transforms/DDR/PlanDDRMemory.cpp",
        "planDDRMemoryModule",
    ),
    "software-pipeline-cross-worker-completion": _anchor(
        "lib/Wafer/Transforms/Scheduling/WorkerPlacement.cpp",
        "deriveDisjointNCCWorkerPlacementCandidate",
    ),
}


def _axis(
    key: str,
    stage: str,
    disposition: AxisDisposition,
    evidence_key: str,
    reason: str,
    *,
    observables: tuple[str, ...] = (),
    host_assets: tuple[str, ...] = (),
) -> OptimizationAxis:
    return OptimizationAxis(
        key=key,
        pipeline_stage=stage,
        disposition=disposition,
        evidence_key=evidence_key,
        board_observables=observables,
        reason=reason,
        production_owner=PRODUCTION_OWNER_BY_AXIS[key],
        host_assets=host_assets,
    )


OPTIMIZATION_AXES = (
    _axis(
        "target-implementation-selection",
        "candidate-selection",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "reciprocal-implementation",
        "Different legal target implementations change issued CT work.",
        observables=("target calls", "full output", "balanced host observation"),
    ),
    _axis(
        "dependent-tiling-and-tail-coverage",
        "structured-to-selected-physical",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "gemm-tail-physical-route",
        "The odd-shape pair checks a distinct target-call relation and exact output.",
        observables=("target call count relation", "exact odd-shape output"),
    ),
    _axis(
        "producer-fusion-and-relation-propagation",
        "structured-to-selected-physical",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "resident-fanout-share",
        "The fanout pair distinguishes reduced movement from unchanged compute kinds.",
        observables=("RDMA/WDMA call count relation", "full fanout output"),
    ),
    _axis(
        "physical-encoding-view-materialization",
        "selected-physical-dataflow",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "gemm-aligned-physical-route",
        "Encoding/view materialization is visible as a gather/scatter count relation.",
        observables=("gather/scatter call count relation", "full output"),
    ),
    _axis(
        "transfer-route-storage-realization",
        "selected-physical-dataflow",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "gemm-aligned-physical-route",
        "The pair distinguishes the emitted gather/scatter path from fewer such calls.",
        observables=("gather/scatter call count relation", "full output"),
    ),
    _axis(
        "fixed-cx-ncx-gemm-absorption",
        "selected-physical-dataflow",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "gemm-aligned-physical-route",
        "Absorption is valuable only if target layout movement really disappears.",
        observables=("GEMM call presence", "gather/scatter call count relation"),
    ),
    _axis(
        "physical-version-reuse",
        "selected-physical-dataflow",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "resident-fanout-share",
        "Reuse is mapped only to the observed reduction in RDMA/WDMA work.",
        observables=("RDMA/WDMA call count relation", "consumer output"),
    ),
    _axis(
        "movement-resident-cut-elimination",
        "selected-physical-dataflow",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "resident-fanout-share",
        "The useful result is removal of a real spill/reload cut.",
        observables=("removed target calls", "full output", "host observation"),
    ),
    _axis(
        "whole-tensor-share-winner",
        "structured-to-selected-physical",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "resident-fanout-share",
        "The shared winner is distinguished by reduced movement and exact fanout output.",
        observables=("RDMA/WDMA call count relation", "fanout output"),
    ),
    _axis(
        "consumer-local-recompute-winner",
        "structured-to-selected-physical",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "consumer-local-recompute",
        "Recompute trades visible compute for a shorter lifetime and less movement.",
        observables=("multiply call count", "workspace bytes", "full output"),
    ),
    _axis(
        "relaxed-f16-bf16-algebra",
        "numeric-candidate-rewrite",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "f16-common-factor",
        "The relaxed-policy f16 pair changes target multiply count.",
        observables=(
            "add/multiply count",
            "full typed-tolerance output",
            "host observation",
        ),
    ),
    _axis(
        "integer-modular-reassociation",
        "numeric-candidate-rewrite",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "rewrite-exact-negative-gates",
        "No paired board fixture currently isolates reassociation from other DAG work.",
        host_assets=REWRITE_HOST_GATES,
    ),
    _axis(
        "integer-modular-reduction-tree",
        "numeric-candidate-rewrite",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "rewrite-exact-negative-gates",
        "No paired board fixture currently isolates the balanced rank-local tree.",
        host_assets=REWRITE_HOST_GATES,
    ),
    _axis(
        "integer-modular-distribution",
        "numeric-candidate-rewrite",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "rewrite-exact-negative-gates",
        "No paired board fixture currently isolates modular distribution.",
        host_assets=REWRITE_HOST_GATES,
    ),
    _axis(
        "integer-modular-common-factor",
        "numeric-candidate-rewrite",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "rewrite-exact-negative-gates",
        "INT8 CT arithmetic is rejected before target code generation.",
        host_assets=REWRITE_HOST_GATES,
    ),
    _axis(
        "static-loop-invariant-hoist",
        "candidate-rewrite-and-order",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "loop-invariant-host-gates",
        (
            "The public StableHLO source-to-package path does not currently "
            "materialize the SCF loop consumed by this rewrite, so a board "
            "fixture would require a non-production bypass."
        ),
        host_assets=(*REWRITE_HOST_GATES, *SELECTION_HOST_GATES),
    ),
    _axis(
        "static-buffering-ready-order",
        "instruction-order",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "ready-order-movement-first",
        "A non-source order is useful only if it changes board blocking or overlap.",
        observables=("target call order", "full output", "host observation"),
    ),
    _axis(
        "static-fixed-slot-overlap-selection",
        "candidate-selection",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "long-steady-elementwise-add",
        (
            "A legal rotating fixed-slot candidate is useful only when normal "
            "production selection produces a distinct executable."
        ),
        observables=(
            "scheduler body difference",
            "full exact output",
            "host observation",
        ),
    ),
    _axis(
        "collective-direct",
        "all-rank-communication",
        AxisDisposition.EXISTING_BOARD_FAMILY,
        "direct-all-reduce",
        (
            "The legacy evidence key covers the production Direct-DTE transport "
            "vertical, not a Direct collective-algorithm choice."
        ),
        observables=("manifest transport contract", "all-rank status", "full output"),
    ),
    _axis(
        "collective-ring-all-gather",
        "all-rank-communication",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "ring-all-gather-host-gates",
        (
            "The public SPMD source boundary currently keeps the generated "
            "all-gather on its reserved baseline, so no distinct production "
            "winner package exists for a board A/B."
        ),
        host_assets=(
            "unittests/Conversion/CommunicationAlternativesTest.cpp",
            "unittests/Compiler/RankCandidateEvaluationTest.cpp",
        ),
    ),
    _axis(
        "collective-ordered-tree-all-reduce",
        "all-rank-communication",
        AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY,
        "tree-all-reduce",
        "The pair distinguishes target prepare/workspace work with exact f16 output.",
        observables=("prepare call count", "workspace bytes", "full exact output"),
    ),
    _axis(
        "stablehlo-normalization-and-canonicalization",
        "frontend-normalization",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "frontend-exact-gates",
        (
            "Canonicalization has no independent board signature after "
            "equivalent lowering."
        ),
        host_assets=FRONTEND_HOST_GATES,
    ),
    _axis(
        "function-boundary-bufferization-alias",
        "bufferization",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "memory-lifetime-exact-gates",
        "Alias, ownership, and effect correctness require exhaustive host negatives.",
        host_assets=MEMORY_HOST_GATES,
    ),
    _axis(
        "index-relation-and-traversal-legality",
        "structured-scheduling-scope",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "relation-exact-gates",
        "Board success cannot prove all-and-only relation or traversal legality.",
        host_assets=RELATION_HOST_GATES,
    ),
    _axis(
        "candidate-rewrite-effect-and-numeric-negatives",
        "candidate-rewrite",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "rewrite-exact-negative-gates",
        (
            "Poison, speculation, visibility, and unsupported effects are "
            "rejection proofs."
        ),
        host_assets=REWRITE_HOST_GATES,
    ),
    _axis(
        "ready-order-hazard-negatives",
        "instruction-order",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "ready-order-exact-negative-gates",
        (
            "RAW/WAR/WAW, view alias, DTE wait, and writeback barriers need "
            "host negatives."
        ),
        host_assets=ORDER_HOST_GATES,
    ),
    _axis(
        "resource-aware-neighbor-generation",
        "candidate-generation",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "selection-resource-exact-gates",
        (
            "Resource bounds prune search; a board run cannot prove rejected "
            "neighbors sound."
        ),
        host_assets=SELECTION_HOST_GATES,
    ),
    _axis(
        "bounded-joint-search-and-baseline-fallback",
        "candidate-set",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "selection-resource-exact-gates",
        (
            "Hard caps, deterministic enumeration, and reserved baseline are "
            "host invariants."
        ),
        host_assets=SELECTION_HOST_GATES,
    ),
    _axis(
        "rank-candidate-pareto-and-atomic-selection",
        "all-rank-coordination",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "selection-resource-exact-gates",
        "Atomicity and exact Pareto accounting require rejected-tuples host coverage.",
        host_assets=SELECTION_HOST_GATES,
    ),
    _axis(
        "spm-lifetime-placement-and-packing",
        "spm-offsets",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "memory-lifetime-exact-gates",
        (
            "Exact conflict graphs and capacity rejection are deterministic "
            "host contracts."
        ),
        host_assets=MEMORY_HOST_GATES,
    ),
    _axis(
        "ddr-lifetime-placement-and-packing",
        "ddr-offsets",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "memory-lifetime-exact-gates",
        (
            "DDR non-overlap, alignment, and fallback correctness are host "
            "proof obligations."
        ),
        host_assets=MEMORY_HOST_GATES,
    ),
    _axis(
        "typed-instruction-lowering-legality",
        "instruction-and-target-lowering",
        AxisDisposition.HOST_ONLY_EXACT_NEGATIVE,
        "typed-lowering-exact-gates",
        (
            "Verifier and conversion coverage must reject every illegal typed "
            "form on host."
        ),
        host_assets=LOWERING_HOST_GATES,
    ),
    _axis(
        "software-pipeline-queue-occupancy",
        "software-pipeline-qualification",
        AxisDisposition.PENDING_CONFIGURED_BOARD,
        "queue-saturation-response",
        (
            "The compiler-owned fixed-slot producer and exact "
            "package/model/no-card host gates are closed; safe outstanding "
            "capacity and winner correctness await fresh configured-board "
            "qualification."
        ),
        host_assets=SOFTWARE_PIPELINE_PLAN,
    ),
    _axis(
        "software-pipeline-spm-bank-cost",
        "spm-cost-calibration",
        AxisDisposition.PENDING_CONFIGURED_BOARD,
        "spm-conflict-equivalence",
        "No calibrated SPM bank identity or directional cost may guide scheduling yet.",
        host_assets=SOFTWARE_PIPELINE_PLAN,
    ),
    _axis(
        "software-pipeline-ddr-bank-cost",
        "ddr-cost-calibration",
        AxisDisposition.PENDING_CONFIGURED_BOARD,
        "ddr-conflict-equivalence",
        "No calibrated DDR bank identity or directional cost may guide scheduling yet.",
        host_assets=SOFTWARE_PIPELINE_PLAN,
    ),
    _axis(
        "software-pipeline-cross-worker-completion",
        "worker-placement-qualification",
        AxisDisposition.PENDING_CONFIGURED_BOARD,
        "worker-wait-and-subset-join-exclusion",
        (
            "Typed disjoint-worker placement and exact host completion gates "
            "are closed; same-candidate winner correctness awaits fresh "
            "configured-board qualification."
        ),
        host_assets=SOFTWARE_PIPELINE_PLAN,
    ),
)

AXES_BY_KEY = {axis.key: axis for axis in OPTIMIZATION_AXES}

_axes_by_case: dict[str, list[OptimizationAxis]] = defaultdict(list)
for _axis_entry in OPTIMIZATION_AXES:
    if _axis_entry.is_board_mapped:
        _axes_by_case[_axis_entry.evidence_key].append(_axis_entry)
AXES_BY_CASE = {
    case_key: tuple(axis_entries)
    for case_key, axis_entries in _axes_by_case.items()
}

BOARD_AXES = tuple(axis for axis in OPTIMIZATION_AXES if axis.is_board_mapped)
HOST_ONLY_AXES = tuple(
    axis
    for axis in OPTIMIZATION_AXES
    if axis.disposition == AxisDisposition.HOST_ONLY_EXACT_NEGATIVE
)
PENDING_CONFIGURED_BOARD_AXES = tuple(
    axis
    for axis in OPTIMIZATION_AXES
    if axis.disposition == AxisDisposition.PENDING_CONFIGURED_BOARD
)
