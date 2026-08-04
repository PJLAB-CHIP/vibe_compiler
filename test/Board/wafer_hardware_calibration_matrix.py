#!/usr/bin/env python3
"""Machine-auditable preparation state for every hardware calibration row."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CatalogBinding:
    """Resolvable references from one semantic leaf to concrete catalog rows."""

    asset: str
    symbol: str
    identifiers: tuple[str | int, ...]
    identifier_field: str = "name"


@dataclasses.dataclass(frozen=True)
class CalibrationLeaf:
    """Smallest independently auditable hardware-calibration requirement."""

    key: str
    layer: str
    disposition: str
    execution_scope: str
    bindings: tuple[CatalogBinding, ...]
    oracle: tuple[str, ...]
    guards: tuple[str, ...]
    completion: tuple[str, ...]
    resource_budget: str
    reason: str | None = None
    remaining_preparation: tuple[str, ...] = ()

    @property
    def preparation(self) -> str:
        return "in-progress" if self.remaining_preparation else "ready"


@dataclasses.dataclass(frozen=True)
class CalibrationDomain:
    key: str
    document_label: str
    execution_scope: str
    positive_assets: tuple[str, ...]
    negative_assets: tuple[str, ...]
    no_card_tests: tuple[str, ...]

    @property
    def remaining_preparation(self) -> tuple[str, ...]:
        leaves = CALIBRATION_LEAVES_BY_DOMAIN.get(self.key, ())
        return tuple(
            leaf.key
            for leaf in leaves
            if leaf.preparation != "ready"
        )

    @property
    def preparation(self) -> str:
        leaves = CALIBRATION_LEAVES_BY_DOMAIN.get(self.key, ())
        return (
            "ready"
            if leaves and all(leaf.preparation == "ready" for leaf in leaves)
            else "in-progress"
        )


def _binding(
    asset: str,
    symbol: str,
    *identifiers: str | int,
    identifier_field: str = "name",
) -> CatalogBinding:
    return CatalogBinding(
        asset=asset,
        symbol=symbol,
        identifiers=tuple(identifiers),
        identifier_field=identifier_field,
    )


def _leaf(
    key: str,
    layer: str,
    disposition: str,
    execution_scope: str,
    *,
    bindings: tuple[CatalogBinding, ...] = (),
    oracle: tuple[str, ...] = (),
    guards: tuple[str, ...] = (),
    completion: tuple[str, ...] = (),
    resource_budget: str = "",
    reason: str | None = None,
    remaining: tuple[str, ...] = (),
) -> CalibrationLeaf:
    return CalibrationLeaf(
        key=key,
        layer=layer,
        disposition=disposition,
        execution_scope=execution_scope,
        bindings=bindings,
        oracle=oracle,
        guards=guards,
        completion=completion,
        resource_budget=resource_budget,
        reason=reason,
        remaining_preparation=remaining,
    )


def _catalog_groups(
    asset: str, *group_keys: str
) -> tuple[CatalogBinding, ...]:
    return (
        _binding(
            asset,
            "CALIBRATION_LEAF_BINDINGS",
            *group_keys,
            identifier_field="@mapping",
        ),
    )


def _board_leaf(
    key: str,
    layer: str,
    execution_scope: str,
    asset: str,
    *group_keys: str,
    resource_budget: str,
    oracle: tuple[str, ...] = ("independent-expected", "full-result"),
    guards: tuple[str, ...] = ("physical-span", "prefix-suffix-canary"),
    completion: tuple[str, ...] = (
        "matching-completion",
        "terminal-status",
        "cleanup",
    ),
) -> CalibrationLeaf:
    return _leaf(
        key,
        layer,
        "board-positive",
        execution_scope,
        bindings=_catalog_groups(asset, *group_keys),
        oracle=oracle,
        guards=guards,
        completion=completion,
        resource_budget=resource_budget,
    )


def _non_board_leaf(
    key: str,
    layer: str,
    disposition: str,
    asset: str,
    *group_keys: str,
    reason: str,
) -> CalibrationLeaf:
    return _leaf(
        key,
        layer,
        disposition,
        "host-gate",
        bindings=_catalog_groups(asset, *group_keys),
        resource_budget="no-device-submission",
        reason=reason,
    )


def _observation_leaf(
    key: str,
    layer: str,
    execution_scope: str,
    asset: str,
    *group_keys: str,
    resource_budget: str,
    reason: str,
) -> CalibrationLeaf:
    return _leaf(
        key,
        layer,
        "board-observation",
        execution_scope,
        bindings=_catalog_groups(asset, *group_keys),
        oracle=("raw-full-result-capture", "request-echo"),
        guards=("physical-span", "prefix-suffix-canary"),
        completion=(
            "matching-completion",
            "terminal-status",
            "cleanup",
        ),
        resource_budget=resource_budget,
        reason=reason,
    )


def _domain(
    key: str,
    document_label: str,
    execution_scope: str,
    *,
    positive: tuple[str, ...] = (),
    negative: tuple[str, ...] = (),
    tests: tuple[str, ...] = (),
) -> CalibrationDomain:
    return CalibrationDomain(
        key=key,
        document_label=document_label,
        execution_scope=execution_scope,
        positive_assets=positive,
        negative_assets=negative,
        no_card_tests=tests,
    )


NCC_EXECUTION = (
    "test/Board/wafer_board_ncc_execution_probe_test.py",
    "test/Board/Inputs/wafer_ncc_execution_probe.c",
    "test/Board/wafer_ncc_probe_protocol.py",
)
INSTRUCTION_FAMILY = (
    "test/Board/wafer_board_instruction_family_probe_test.py",
    "test/Board/wafer_instruction_family_catalog.py",
    "test/Board/Inputs/wafer_instruction_family_probe.c",
)
CT_VECTOR_CALIBRATION = (
    "test/Board/wafer_board_ct_vector_calibration_probe_test.py",
    "test/Board/wafer_ct_vector_calibration_catalog.py",
    "test/Board/Inputs/wafer_ct_vector_calibration_probe.c",
    "test/Board/Inputs/wafer_ct_vector_calibration_probe_protocol.h",
)
CT_CONVERT_CALIBRATION = (
    "test/Board/wafer_board_ct_convert_calibration_probe_test.py",
    "test/Board/wafer_ct_convert_calibration_catalog.py",
    "test/Board/Inputs/wafer_ct_convert_calibration_probe.c",
    "test/Board/Inputs/wafer_ct_convert_calibration_probe_protocol.h",
)
CT_OPCODE_DISPOSITIONS = (
    "test/Board/wafer_ct_opcode_disposition_catalog.py",
    "test/Board/wafer_ct_opcode_disposition_catalog_test.py",
)
DATAMOVE_CALIBRATION = (
    "test/Board/wafer_board_datamove_calibration_probe_test.py",
    "test/Board/wafer_datamove_calibration_catalog.py",
    "test/Board/Inputs/wafer_datamove_calibration_probe.c",
    "test/Board/Inputs/wafer_datamove_calibration_probe_protocol.h",
)
DATAMOVE_EXTENDED_CALIBRATION = (
    "test/Board/wafer_board_datamove_extended_calibration_probe_test.py",
    "test/Board/wafer_datamove_extended_calibration_catalog.py",
    "test/Board/Inputs/wafer_datamove_extended_calibration_probe.c",
    (
        "test/Board/Inputs/"
        "wafer_datamove_extended_calibration_probe_protocol.h"
    ),
)
PHYSICAL_TENSOR_CODEC = (
    "test/Board/wafer_physical_tensor_codec.py",
    "test/Board/wafer_physical_tensor_codec_test.py",
)
NE_CALIBRATION = (
    "test/Board/wafer_board_ne_calibration_probe_test.py",
    "test/Board/wafer_ne_calibration_catalog.py",
    "test/Board/Inputs/wafer_ne_calibration_probe.c",
    "test/Board/Inputs/wafer_ne_calibration_probe_protocol.h",
)
SPM_CALIBRATION = (
    "test/Board/wafer_board_spm_calibration_probe_test.py",
    "test/Board/wafer_spm_calibration_catalog.py",
    "test/Board/Inputs/wafer_spm_calibration_probe.c",
    "test/Board/Inputs/wafer_spm_calibration_probe_protocol.h",
)
CACHE_COHERENCE_CALIBRATION = (
    "test/Board/wafer_board_cache_coherence_calibration_probe_test.py",
    "test/Board/wafer_cache_coherence_calibration_catalog.py",
    "test/Board/Inputs/wafer_cache_coherence_calibration_probe.c",
    "test/Board/Inputs/wafer_cache_coherence_calibration_probe_protocol.h",
)
MEMORY_DESCRIPTOR_CALIBRATION = (
    "test/Board/wafer_board_memory_descriptor_calibration_probe_test.py",
    "test/Board/wafer_memory_descriptor_calibration_catalog.py",
    "test/Board/Inputs/wafer_memory_descriptor_calibration_probe.c",
    (
        "test/Board/Inputs/"
        "wafer_memory_descriptor_calibration_probe_protocol.h"
    ),
)
DTE_NCC = (
    "test/Board/wafer_board_dte_ncc_execution_probe_test.py",
    "test/Board/Inputs/wafer_dte_ncc_execution_probe.c",
)
TRANSPORT_PMU_CALIBRATION = (
    "test/Board/wafer_transport_pmu_calibration_catalog.py",
    "test/Board/wafer_transport_pmu_calibration_catalog_test.py",
)
FULL_CARD_BARRIER = (
    "test/Board/wafer_board_full_card_barrier_probe_test.py",
    "test/Board/Inputs/wafer_full_card_barrier_probe.c",
)
NCC_PMU = (
    "test/Board/wafer_board_ncc_pmu_probe_test.py",
    "test/Board/Inputs/wafer_ncc_pmu_readonly_probe.c",
)

INSTRUCTION_CATALOG = "test/Board/wafer_instruction_family_catalog.py"
CT_VECTOR_CATALOG = "test/Board/wafer_ct_vector_calibration_catalog.py"
CT_CONVERT_CATALOG = "test/Board/wafer_ct_convert_calibration_catalog.py"
CT_DISPOSITION_CATALOG = (
    "test/Board/wafer_ct_opcode_disposition_catalog.py"
)
DATAMOVE_CATALOG = "test/Board/wafer_datamove_calibration_catalog.py"
DATAMOVE_EXTENDED_CATALOG = (
    "test/Board/wafer_datamove_extended_calibration_catalog.py"
)
NE_CATALOG = "test/Board/wafer_ne_calibration_catalog.py"
SPM_CATALOG = "test/Board/wafer_spm_calibration_catalog.py"
CACHE_CATALOG = "test/Board/wafer_cache_coherence_calibration_catalog.py"
MEMORY_DESCRIPTOR_CATALOG = (
    "test/Board/wafer_memory_descriptor_calibration_catalog.py"
)
NCC_CATALOG = "test/Board/wafer_board_ncc_execution_probe_test.py"
DTE_NCC_CATALOG = (
    "test/Board/wafer_transport_pmu_calibration_catalog.py"
)
BARRIER_CATALOG = "test/Board/wafer_board_full_card_barrier_probe_test.py"
TRANSPORT_PMU_CATALOG = (
    "test/Board/wafer_transport_pmu_calibration_catalog.py"
)
NCC_PMU_CATALOG = "test/Board/wafer_board_ncc_pmu_probe_test.py"
RUNTIME_CATALOG = "test/Board/wafer_board_all_rank_add_test.py"
RUNTIME_RANK_ONE_CATALOG = (
    "test/Board/wafer_board_single_op_add_test.py"
)


CALIBRATION_DOMAINS = (
    _domain(
        "profile-qualification",
        "profile qualification",
        "rank-one-read-only-and-heartbeat",
        positive=NCC_PMU + (RUNTIME_RANK_ONE_CATALOG,),
        tests=("wafer-runtime-ncc-pmu-readonly-probe-no-card",),
    ),
    _domain(
        "constructor-ownership",
        "constructor ownership",
        "rank-one-worker0",
        positive=NCC_EXECUTION,
        tests=(
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-safe-observations-no-card",
        ),
    ),
    _domain(
        "execute-result",
        "execute result",
        "rank-one-worker0",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-safe-observations-no-card",),
    ),
    _domain(
        "packet-routing-range",
        "packet routing/range",
        "rank-one-workers012",
        positive=NCC_EXECUTION,
        tests=(
            "wafer-ncc-probe-plan",
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-safe-observations-no-card",
        ),
    ),
    _domain(
        "ct-numeric-form",
        "CT numeric/form",
        "rank-one-worker0",
        positive=(
            INSTRUCTION_FAMILY
            + CT_VECTOR_CALIBRATION
            + CT_CONVERT_CALIBRATION
            + CT_OPCODE_DISPOSITIONS
        ),
        negative=CT_OPCODE_DISPOSITIONS,
        tests=(
            "wafer-instruction-family-catalog-python",
            "wafer-ct-vector-calibration-catalog-python",
            "wafer-ct-convert-calibration-catalog-python",
            "wafer-ct-opcode-disposition-catalog-python",
            "wafer-runtime-ct-vector-calibration-probe-no-card",
            "wafer-runtime-ct-convert-calibration-probe-no-card",
        ),
    ),
    _domain(
        "instruction-physical-layout",
        "instruction × physical layout",
        "rank-one-worker0",
        positive=(
            INSTRUCTION_FAMILY
            + CT_VECTOR_CALIBRATION
            + PHYSICAL_TENSOR_CODEC
            + NE_CALIBRATION
            + DATAMOVE_CALIBRATION
            + DATAMOVE_EXTENDED_CALIBRATION
        ),
        negative=(
            "test/Board/wafer_datamove_calibration_catalog.py",
            "test/Board/wafer_datamove_calibration_catalog_test.py",
        ),
        tests=(
            "wafer-physical-tensor-codec-python",
            "wafer-ne-calibration-catalog-python",
            "wafer-datamove-calibration-catalog-python",
            "wafer-datamove-extended-calibration-catalog-python",
            "wafer-runtime-ne-calibration-probe-no-card",
            "wafer-runtime-datamove-calibration-probe-no-card",
            "wafer-runtime-datamove-extended-calibration-probe-no-card",
        ),
    ),
    _domain(
        "datamove-layout",
        "DataMove/layout",
        "rank-one-worker0",
        positive=(
            INSTRUCTION_FAMILY
            + DATAMOVE_CALIBRATION
            + DATAMOVE_EXTENDED_CALIBRATION
        ),
        negative=(
            "test/Board/wafer_datamove_calibration_catalog.py",
            "test/Board/wafer_datamove_calibration_catalog_test.py",
        ),
        tests=(
            "wafer-datamove-calibration-catalog-python",
            "wafer-datamove-extended-calibration-catalog-python",
            "wafer-runtime-datamove-calibration-probe-no-card",
            "wafer-runtime-datamove-extended-calibration-probe-no-card",
        ),
    ),
    _domain(
        "ne-numeric-layout",
        "NE numeric/layout",
        "rank-one-worker0",
        positive=INSTRUCTION_FAMILY + PHYSICAL_TENSOR_CODEC + NE_CALIBRATION,
        tests=(
            "wafer-instruction-family-catalog-python",
            "wafer-physical-tensor-codec-python",
            "wafer-ne-calibration-catalog-python",
            "wafer-runtime-ne-calibration-probe-no-card",
        ),
    ),
    _domain(
        "rdma-wdma-descriptor",
        "RDMA/WDMA descriptor",
        "rank-one-worker0",
        positive=NCC_EXECUTION + MEMORY_DESCRIPTOR_CALIBRATION,
        tests=(
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-safe-observations-no-card",
            "wafer-memory-descriptor-calibration-catalog-python",
            "wafer-runtime-memory-descriptor-calibration-probe-no-card",
        ),
    ),
    _domain(
        "tdma-memset",
        "TDMA Memset",
        "rank-one-worker0",
        positive=NCC_EXECUTION + DATAMOVE_EXTENDED_CALIBRATION,
        tests=(
            "wafer-runtime-ncc-safe-observations-no-card",
            "wafer-datamove-extended-calibration-catalog-python",
            "wafer-runtime-datamove-extended-calibration-probe-no-card",
        ),
    ),
    _domain(
        "tdma-bool-fill",
        "TDMA BOOL fill",
        "rank-one-worker0-manual",
        positive=NCC_EXECUTION,
        negative=("test/Board/wafer_ncc_probe_protocol_test.py",),
        tests=(
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-safe-observations-no-card",
        ),
    ),
    _domain(
        "tdma-movement-variants",
        "TDMA movement variants",
        "rank-one-worker0",
        positive=(
            INSTRUCTION_FAMILY
            + NCC_EXECUTION
            + DATAMOVE_CALIBRATION
            + DATAMOVE_EXTENDED_CALIBRATION
        ),
        negative=(
            "test/Board/wafer_datamove_calibration_catalog.py",
            "test/Board/wafer_datamove_calibration_catalog_test.py",
        ),
        tests=(
            "wafer-datamove-calibration-catalog-python",
            "wafer-datamove-extended-calibration-catalog-python",
            "wafer-runtime-datamove-calibration-probe-no-card",
            "wafer-runtime-datamove-extended-calibration-probe-no-card",
        ),
    ),
    _domain(
        "spm-capacity-reservation",
        "SPM capacity/reservation",
        "rank-one-static-and-worker0",
        positive=SPM_CALIBRATION,
        negative=(
            "test/Board/wafer_spm_calibration_catalog.py",
            "test/Board/wafer_spm_calibration_catalog_test.py",
        ),
        tests=(
            "wafer-spm-calibration-catalog-python",
            "wafer-runtime-spm-calibration-probe-no-card",
        ),
    ),
    _domain(
        "spm-alignment-bank",
        "SPM alignment/bank",
        "rank-one-worker0",
        positive=(
            NCC_EXECUTION
            + SPM_CALIBRATION
            + MEMORY_DESCRIPTOR_CALIBRATION
        ),
        tests=(
            "wafer-spm-calibration-catalog-python",
            "wafer-runtime-spm-calibration-probe-no-card",
            "wafer-memory-descriptor-calibration-catalog-python",
            "wafer-runtime-memory-descriptor-calibration-probe-no-card",
        ),
    ),
    _domain(
        "ddr-cache-coherence",
        "DDR/cache/coherence",
        "rank-one-and-full-card",
        positive=(
            DTE_NCC
            + NCC_EXECUTION
            + CACHE_COHERENCE_CALIBRATION
            + MEMORY_DESCRIPTOR_CALIBRATION
        ),
        tests=(
            "wafer-cache-coherence-calibration-catalog-python",
            "wafer-runtime-cache-coherence-calibration-probe-no-card",
            "wafer-memory-descriptor-calibration-catalog-python",
            "wafer-runtime-memory-descriptor-calibration-probe-no-card",
        ),
    ),
    _domain(
        "queue-shape-submission",
        "queue shape与连续提交边界",
        "rank-one-worker0-manual",
        positive=NCC_EXECUTION,
        tests=(
            "wafer-ncc-probe-plan",
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-safe-observations-no-card",
        ),
    ),
    _domain(
        "worker-scope",
        "worker scope",
        "rank-one-workers012-manual",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-safe-observations-no-card",),
    ),
    _domain(
        "cross-engine-overlap",
        "cross-engine overlap",
        "rank-one-worker0",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-safe-observations-no-card",),
    ),
    _domain(
        "address-dependency",
        "address dependency",
        "rank-one-worker0-manual",
        positive=NCC_EXECUTION,
        negative=(
            "test/Board/wafer_ncc_hazard_relation_test.c",
            "test/Board/Inputs/wafer_ncc_hazard_relation.c",
            "test/Board/Inputs/wafer_ncc_hazard_relation.h",
        ),
        tests=(
            "wafer-ncc-hazard-relation",
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-safe-observations-no-card",
        ),
    ),
    _domain(
        "issue-overhead",
        "issue overhead",
        "rank-one-worker0-manual",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-safe-observations-no-card",),
    ),
    _domain(
        "local-completion",
        "local completion",
        "rank-one-workers012-manual",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-safe-observations-no-card",),
    ),
    _domain(
        "cross-worker-join",
        "cross-worker join",
        "rank-one-workers012",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-safe-observations-no-card",),
    ),
    _domain(
        "direct-dte",
        "Direct DTE",
        "full-card-16-rank",
        positive=DTE_NCC,
        negative=("test/Board/wafer_board_direct_dte_timeout_test.py",),
        tests=(
            "wafer-board-direct-dte-outer-deadline",
            "wafer-runtime-dte-ncc-execution-probe-no-card",
        ),
    ),
    _domain(
        "multi-tile-arrival",
        "multi-tile arrival",
        "full-card-and-subgroup",
        positive=FULL_CARD_BARRIER,
        negative=(
            "test/Board/wafer_full_card_barrier_contract_test.py",
            "test/Board/wafer_board_full_card_barrier_probe_test.py",
        ),
        tests=(
            "wafer-full-card-barrier-contract-python",
            "wafer-runtime-full-card-barrier-probe-no-card",
        ),
    ),
    _domain(
        "host-launch-runtime",
        "host launch/runtime",
        "rank-one-and-full-card",
        positive=(
            "test/Board/wafer_board_all_rank_add_test.py",
            "test/Board/wafer_board_single_op_add_test.py",
            "test/Board/wafer_board_direct_dte_collective_test.py",
        ),
        negative=("test/Board/wafer_board_direct_dte_timeout_test.py",),
        tests=(
            "wafer-runtime-adapter-python",
            "wafer-runtime-kernel-grid-add-no-card",
            "wafer-runtime-cluster-direct-dte-no-card",
        ),
    ),
    _domain(
        "ncc-pmu-basis",
        "NCC PMU basis",
        "rank-one-worker0",
        positive=NCC_PMU + NCC_EXECUTION,
        tests=(
            "wafer-runtime-ncc-pmu-readonly-probe-no-card",
            "wafer-runtime-ncc-safe-observations-no-card",
        ),
    ),
    _domain(
        "transport-pmu-basis",
        "DTE/SPM/TMNOC PMU",
        "full-card-and-rank-one",
        positive=DTE_NCC + TRANSPORT_PMU_CALIBRATION,
        negative=TRANSPORT_PMU_CALIBRATION,
        tests=(
            "wafer-transport-pmu-calibration-catalog-python",
            "wafer-runtime-dte-ncc-execution-probe-no-card",
        ),
    ),
    _domain(
        "scalar-csr-ordinary-issue",
        "SCALAR/CSR ordinary issue",
        "static-negative",
        negative=(
            "test/Board/wafer_ncc_probe_protocol_test.py",
            "test/Board/Inputs/wafer_ncc_execution_probe.c",
        ),
        tests=("wafer-ncc-probe-protocol-python",),
    ),
)


DOMAINS_BY_KEY = {domain.key: domain for domain in CALIBRATION_DOMAINS}


CALIBRATION_LEAVES_BY_DOMAIN = {
    "profile-qualification": (
        _board_leaf(
            "profile-identity-readonly",
            "calibration",
            "rank-one-read-only",
            NCC_PMU_CATALOG,
            "profile-identity-readonly",
            resource_budget="read-only-registers",
        ),
        _board_leaf(
            "known-good-heartbeat",
            "held-out",
            "rank-one-runtime-session",
            RUNTIME_RANK_ONE_CATALOG,
            "rank-one-per-rank-add",
            resource_budget="runtime-owned-rank-one-resources",
            oracle=("independent-f16-expected", "full-result"),
            guards=("schema-v7-resource-binding", "rank-one-domain"),
            completion=("terminal-status", "D2H", "cleanup"),
        ),
    ),
    "constructor-ownership": (
        _board_leaf(
            "constructor-return-address-nonnull",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "constructor-return-address-nonnull",
            resource_budget="one-owned-packet-builder",
            oracle=("constructor-address-nonnull", "exact-packet-result"),
        ),
        _board_leaf(
            "constructor-builder-release",
            "held-out",
            "rank-one-worker0",
            NCC_CATALOG,
            "constructor-builder-release",
            resource_budget="bounded-packet-builders",
        ),
    ),
    "execute-result": (
        _board_leaf(
            "execute-success-requires-side-effects",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "execute-success-requires-side-effects",
            resource_budget="one-known-good-packet",
        ),
        _non_board_leaf(
            "execute-invalid-type-return",
            "held-out",
            "static-negative",
            NCC_CATALOG,
            "execute-invalid-type-return",
            reason=(
                "invalid ordinary-issue types are rejected by the typed host "
                "gate because the raw return value is not a success oracle"
            ),
        ),
        _non_board_leaf(
            "execute-engine-none-host-negative",
            "held-out",
            "static-negative",
            NCC_CATALOG,
            "execute-engine-none-static-negative",
            reason=(
                "typed Plan validation rejects the Engine.NONE wire sentinel "
                "before serialization or device submission"
            ),
        ),
    ),
    "packet-routing-range": (
        _board_leaf(
            "routing-five-engines-worker0",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "routing-five-engines-worker0",
            resource_budget="one-packet-per-engine",
        ),
        _board_leaf(
            "routing-ct-workers012",
            "held-out",
            "rank-one-workers012",
            NCC_CATALOG,
            "routing-ct-workers012",
            resource_budget="one-ct-packet-per-worker",
        ),
        _board_leaf(
            "range-materialization-ct-ne",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "range-materialization-ct-ne",
            resource_budget="ct-ne-register-range",
        ),
        _board_leaf(
            "range-materialization-rdma-wdma-tdma",
            "held-out",
            "rank-one-worker0",
            NCC_CATALOG,
            "range-materialization-rdma-wdma-tdma",
            resource_budget="dma-descriptor-range",
        ),
        _non_board_leaf(
            "execute-worker-out-of-range-host-negative",
            "held-out",
            "static-negative",
            NCC_CATALOG,
            "execute-worker-out-of-range-static-negative",
            reason=(
                "typed Lane validation rejects worker ids outside 0..2 "
                "before serialization or device submission"
            ),
        ),
    ),
    "ct-numeric-form": (
        _board_leaf(
            "ct-vector-forms-main-tail",
            "calibration",
            "rank-one-worker0",
            CT_VECTOR_CATALOG,
            "ct-vector-forms-main-tail",
            resource_budget="two-64k-ddr-slots",
        ),
        _board_leaf(
            "ct-fp16-bf16-fp32-finite-boundaries",
            "held-out",
            "rank-one-worker0-manual",
            CT_VECTOR_CATALOG,
            "ct-finite-boundaries-all-float-dtypes",
            resource_budget="two-64k-ddr-slots",
        ),
        _board_leaf(
            "ct-zero-domain-positive",
            "held-out",
            "rank-one-worker0-manual",
            CT_VECTOR_CATALOG,
            "ct-zero-domain-positive",
            resource_budget="two-64k-ddr-slots",
        ),
        _observation_leaf(
            "ct-zero-domain-observation",
            "held-out",
            "rank-one-worker0-manual",
            CT_VECTOR_CATALOG,
            "ct-zero-domain-observed",
            resource_budget="two-64k-ddr-slots",
            reason=(
                "zero-domain rows without a frozen result classification "
                "are captured with full guards but do not qualify semantics"
            ),
        ),
        _board_leaf(
            "ct-special-values-positive",
            "held-out",
            "rank-one-worker0-manual",
            CT_VECTOR_CATALOG,
            "ct-special-values-positive",
            resource_budget="two-64k-ddr-slots",
        ),
        _observation_leaf(
            "ct-special-values-observation",
            "held-out",
            "rank-one-worker0-manual",
            CT_VECTOR_CATALOG,
            "ct-special-values-observed",
            resource_budget="two-64k-ddr-slots",
            reason=(
                "NaN payload/quieting and selected Inf/domain behavior are "
                "safe to capture but lack a profile-independent exact oracle"
            ),
        ),
        _board_leaf(
            "ct-relation-logic-value-bool",
            "calibration",
            "rank-one-worker0",
            CT_VECTOR_CATALOG,
            "ct-relation-logic-value-bool",
            resource_budget="two-64k-ddr-slots",
        ),
        _board_leaf(
            "ct-transcendental-activation",
            "calibration",
            "rank-one-worker0",
            CT_VECTOR_CATALOG,
            "ct-transcendental-activation",
            resource_budget="two-64k-ddr-slots",
        ),
        _observation_leaf(
            "ct-observed-activation-semantics",
            "held-out",
            "rank-one-worker0-manual",
            CT_VECTOR_CATALOG,
            "ct-observed-activation-semantics",
            resource_budget="two-64k-ddr-slots",
            reason=(
                "SatRelu and LeakyReLU are safe to capture with full guards, "
                "but their implicit parameter semantics lack an independent "
                "expected oracle"
            ),
        ),
        _board_leaf(
            "ct-convert-all-routes-main-tail",
            "calibration",
            "rank-one-worker0",
            CT_CONVERT_CATALOG,
            "ct-convert-all-routes-main-tail-nearest-even",
            resource_budget="two-64k-ddr-slots",
        ),
        _board_leaf(
            "ct-convert-halfway-extrema-positive",
            "held-out",
            "rank-one-worker0-manual",
            CT_CONVERT_CATALOG,
            "ct-convert-halfway-extrema-positive",
            resource_budget="two-64k-ddr-slots",
        ),
        _observation_leaf(
            "ct-convert-halfway-extrema-observation",
            "held-out",
            "rank-one-worker0-manual",
            CT_CONVERT_CATALOG,
            "ct-convert-halfway-extrema-observed",
            resource_budget="two-64k-ddr-slots",
            reason=(
                "overflow/saturation rows whose destination behavior is not "
                "yet frozen are captured without qualifying exact semantics"
            ),
        ),
        _board_leaf(
            "ct-convert-directed-rounding",
            "held-out",
            "rank-one-worker0-manual",
            CT_CONVERT_CATALOG,
            "ct-convert-directed-rounding-positive",
            resource_budget="two-64k-ddr-slots",
        ),
        _observation_leaf(
            "ct-convert-zero-point-observation",
            "held-out",
            "rank-one-worker0-manual",
            CT_CONVERT_CATALOG,
            "ct-convert-zero-point-observed",
            resource_budget="two-64k-ddr-slots",
            reason=(
                "the constructor exposes zero-point fields, but the exact "
                "application order remains profile-observed"
            ),
        ),
        _observation_leaf(
            "ct-convert-stochastic",
            "held-out",
            "rank-one-worker0-manual",
            CT_CONVERT_CATALOG,
            "ct-convert-stochastic-observed",
            resource_budget="two-64k-ddr-slots",
            reason=(
                "repeat each bounded stochastic route and retain raw output, "
                "physical span, guard, PMU and completion without assuming a "
                "seeded sequence or distribution"
            ),
        ),
        _board_leaf(
            "ct-reduce-pool-unpool-peripheral-positive",
            "calibration",
            "rank-one-worker0",
            CT_DISPOSITION_CATALOG,
            "ct-reduce-pool-unpool-peripheral-positive",
            resource_budget="shared-instruction-family-package",
        ),
        _observation_leaf(
            "ct-reduce-pool-unpool-peripheral-observation",
            "held-out",
            "rank-one-worker0",
            CT_DISPOSITION_CATALOG,
            "ct-reduce-pool-unpool-peripheral-observed",
            resource_budget="shared-instruction-family-package",
            reason=(
                "composite-only IndexedMax, Unpool-index, negative-domain "
                "ArgMin, Bilinear, Factorize, LUT32, RandGen and ElemMask "
                "have bounded raw capture but no profile-independent numeric "
                "oracle"
            ),
        ),
        _non_board_leaf(
            "ct-reduce-pool-unpool-peripheral-static-negative",
            "held-out",
            "static-negative",
            CT_DISPOSITION_CATALOG,
            "ct-reduce-pool-unpool-peripheral-static-negative",
            reason=(
                "reduce axes absent from the public enum, pool/unpool Cx, "
                "and scalar unpool index forms are rejected before packet "
                "construction"
            ),
        ),
        _non_board_leaf(
            "ct-reduce-raw-axis-isolated-deferred",
            "held-out",
            "isolated-deferred",
            CT_DISPOSITION_CATALOG,
            "ct-reduce-raw-axis-isolated-deferred",
            reason=(
                "historical raw N/HWC dimensions are absent from the "
                "version-matched public enum; an isolated dimension-3 "
                "launch exceeded its completion deadline and may "
                "permanently wait, so all eight variants are fail-closed"
            ),
        ),
    ),
    "instruction-physical-layout": (
        _leaf(
            "instruction-layout-native-positive",
            "calibration",
            "board-positive",
            "rank-one-worker0",
            bindings=(
                _catalog_groups(
                    DATAMOVE_CATALOG,
                    "instruction-layout-native-positive",
                )
                + _catalog_groups(
                    CT_VECTOR_CATALOG,
                    "ct-vector-forms-main-tail",
                )
                + _catalog_groups(
                    NE_CATALOG,
                    "ne-gemm-dtype-orientation-main-tail-batch",
                )
                + _catalog_groups(
                    NCC_CATALOG,
                    "routing-five-engines-worker0",
                )
                + _catalog_groups(
                    DATAMOVE_CATALOG,
                    "transpose-mirror-rotate-large",
                    "cx-ncx-channel-boundaries",
                )
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="shared-datamove-package",
        ),
        _leaf(
            "instruction-layout-composite-positive",
            "held-out",
            "board-positive",
            "rank-one-worker0",
            bindings=(
                _catalog_groups(
                    DATAMOVE_CATALOG,
                    "instruction-layout-composite-positive",
                )
                + _catalog_groups(
                    DATAMOVE_EXTENDED_CATALOG,
                    "materialize-then-consume",
                )
            ),
            oracle=("materialize-then-consume-exact", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="bounded-datamove-extended-package",
        ),
        _non_board_leaf(
            "instruction-layout-static-negative",
            "held-out",
            "static-negative",
            DATAMOVE_CATALOG,
            "instruction-layout-static-negative",
            reason=(
                "layout/engine combinations without a native or explicit "
                "materialization contract are rejected before issue"
            ),
        ),
        _board_leaf(
            "cx-ncx-c63-c64-c65-c127-c129",
            "held-out",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "cx-ncx-channel-boundaries",
            resource_budget="two-64k-ddr-slots",
        ),
        _board_leaf(
            "cx-ncx-padding-poison-n-slice",
            "held-out",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "cx-ncx-padding-poison-n-slice",
            resource_budget="two-64k-ddr-slots",
        ),
    ),
    "datamove-layout": (
        _board_leaf(
            "datamove-transpose-mirror-rotate",
            "calibration",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "transpose-mirror-rotate-large",
            resource_budget="two-64k-ddr-slots",
        ),
        _board_leaf(
            "datamove-nchw-nhwc",
            "calibration",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "nchw-nhwc-large",
            resource_budget="two-64k-ddr-slots",
        ),
        _board_leaf(
            "datamove-concat-c-w-h-hw",
            "calibration",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "concat-c-w-h-hw",
            resource_budget="two-64k-ddr-slots",
        ),
        _leaf(
            "datamove-raw-concat-disposition",
            "held-out",
            "board-observation",
            "rank-one-worker0",
            bindings=(
                _catalog_groups(
                    DATAMOVE_CATALOG,
                    "raw-concat-observation",
                )
                + _catalog_groups(
                    DATAMOVE_EXTENDED_CATALOG,
                    "raw-concat-c-w-h",
                )
            ),
            oracle=("raw-full-result-capture", "request-echo"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="two-128k-ddr-slots",
            reason=(
                "raw opcode 131 C/W/H completion and guards are bounded, but "
                "their numeric semantics are not yet exact-qualified; native "
                "dims=HW is statically invalid"
            ),
        ),
        _non_board_leaf(
            "datamove-native-concat-hw-static-negative",
            "held-out",
            "static-negative",
            DATAMOVE_CATALOG,
            "raw-concat-hw-static-negative",
            reason=(
                "native dims=HW is an invalid instruction use and must never "
                "be emitted or executed; legal HW-axis concat lowers through "
                "typed GatherScatter"
            ),
        ),
        _board_leaf(
            "datamove-compiler-concat-materialization",
            "production-vertical",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "compiler-concat-materialization",
            resource_budget="shared-datamove-package",
        ),
        _board_leaf(
            "datamove-broadcast-scalar-channel-row",
            "calibration",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "broadcast-scalar-channel-row",
            resource_budget="two-64k-ddr-slots",
        ),
        _leaf(
            "datamove-pad-img2col-large",
            "held-out",
            "board-positive",
            "rank-one-worker0",
            bindings=(
                _catalog_groups(
                    DATAMOVE_CATALOG,
                    "pad-img2col-large",
                )
                + _catalog_groups(
                    DATAMOVE_EXTENDED_CATALOG,
                    "large-pad-img2col",
                )
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="two-128k-ddr-slots",
        ),
        _board_leaf(
            "datamove-gather-contiguous-strided-tail",
            "calibration",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "gather-contiguous-strided-tail",
            resource_budget="two-64k-ddr-slots",
        ),
        _leaf(
            "datamove-mask-gather",
            "held-out",
            "board-observation",
            "rank-one-worker0",
            bindings=(
                _catalog_groups(
                    DATAMOVE_CATALOG,
                    "mask-gather-disposition",
                )
                + _catalog_groups(
                    DATAMOVE_EXTENDED_CATALOG,
                    "mask-gather-and-bit-vector",
                )
            ),
            oracle=("raw-full-result-capture", "request-echo"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="two-128k-ddr-slots",
            reason=(
                "MaskGather and MaskGather_bV are safely bounded captures, "
                "while mask-index ordering remains profile-observed"
            ),
        ),
        _leaf(
            "datamove-tensor-normalization",
            "held-out",
            "board-observation",
            "rank-one-worker0",
            bindings=(
                _catalog_groups(
                    DATAMOVE_CATALOG,
                    "tensor-normalization",
                )
                + _catalog_groups(
                    DATAMOVE_EXTENDED_CATALOG,
                    "tensor-nom",
                )
            ),
            oracle=("raw-full-result-capture", "request-echo"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="two-128k-ddr-slots",
            reason=(
                "TensorNom is safely executed with raw output retention, "
                "but its normalization policy remains profile-observed"
            ),
        ),
    ),
    "ne-numeric-layout": (
        _board_leaf(
            "ne-gemm-dtype-orientation-main-tail-batch",
            "calibration",
            "rank-one-worker0",
            NE_CATALOG,
            "ne-gemm-dtype-orientation-main-tail-batch",
            resource_budget="bounded-ne-shared-package",
        ),
        _board_leaf(
            "ne-long-accumulation-cancellation",
            "held-out",
            "rank-one-worker0",
            NE_CATALOG,
            "ne-long-accumulation-cancellation",
            resource_budget="bounded-ne-shared-package",
        ),
        _observation_leaf(
            "ne-bf16-special-values",
            "held-out",
            "rank-one-worker0-manual",
            NE_CATALOG,
            "ne-bf16-special-values",
            resource_budget="bounded-ne-shared-package",
            reason=(
                "BF16 NaN/Inf/subnormal/signed-zero behavior is captured "
                "with physical guards before a numeric policy is frozen"
            ),
        ),
        _board_leaf(
            "ne-one-factor-options-positive",
            "calibration",
            "rank-one-worker0",
            NE_CATALOG,
            "ne-one-factor-options-positive",
            resource_budget="bounded-ne-shared-package",
        ),
        _observation_leaf(
            "ne-one-factor-options-observation",
            "held-out",
            "rank-one-worker0-manual",
            NE_CATALOG,
            "ne-one-factor-options-observed",
            resource_budget="bounded-ne-shared-package",
            reason=(
                "option rows without a frozen parameter application order "
                "are safe captures rather than exact capability evidence"
            ),
        ),
        _observation_leaf(
            "ne-quant",
            "held-out",
            "rank-one-worker0",
            NE_CATALOG,
            "ne-quant-observed",
            resource_budget="bounded-ne-shared-package",
            reason=(
                "INT8 quantization is safely executed and captured while "
                "the profile-specific scale/rounding policy remains observed"
            ),
        ),
        _non_board_leaf(
            "ne-sparse",
            "held-out",
            "static-negative",
            NE_CATALOG,
            "ne-sparse-static-negative",
            reason=(
                "the owned TsmGemm ABI exposes no sparse configuration "
                "field, so sparse issue is rejected before construction"
            ),
        ),
        _non_board_leaf(
            "ne-pad-unpad",
            "held-out",
            "static-negative",
            NE_CATALOG,
            "ne-pad-unpad-static-negative",
            reason=(
                "the typed GEMM ABI has no pad or unpad field and rejects "
                "both combinations before packet construction"
            ),
        ),
        _observation_leaf(
            "ne-conv-large-held-out",
            "calibration",
            "rank-one-worker0",
            NE_CATALOG,
            "ne-conv-large-heldout",
            resource_budget="bounded-ne-shared-package",
            reason=(
                "ordinary-Conv rows retain bounded raw output; three "
                "one-factor feature/weight/output fingerprints compare "
                "distinct NCx/Cx physical candidates without promoting an "
                "unmatched capture to exact semantics"
            ),
        ),
        _observation_leaf(
            "ne-depthwise-backward-conv",
            "held-out",
            "rank-one-worker0",
            NE_CATALOG,
            "ne-depthwise-backward-conv-observed",
            resource_budget="bounded-ne-shared-package",
            reason=(
                "FP16/BF16 depthwise and backward-convolution outputs are "
                "bounded captures pending a frozen numerical policy"
            ),
        ),
        _board_leaf(
            "ne-batch-broadcast",
            "held-out",
            "rank-one-worker0",
            NE_CATALOG,
            "ne-batch-broadcast-positive",
            resource_budget="bounded-ne-shared-package",
        ),
    ),
    "rdma-wdma-descriptor": (
        _board_leaf(
            "dma-contiguous-64k",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "dma-contiguous-64k",
            resource_budget="256k-ncc-resource-with-64k-payload",
        ),
        _board_leaf(
            "dma-1d-2d-3d-stride-holes",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "dma-1d-2d-3d-stride-holes",
            resource_budget="64k-payload-plus-guards",
        ),
        _board_leaf(
            "dma-offset-alignment-tail",
            "held-out",
            "rank-one-worker0",
            MEMORY_DESCRIPTOR_CATALOG,
            "dma-offset-alignment-tail",
            resource_budget="256k-memory-descriptor-resource",
        ),
    ),
    "tdma-memset": (
        _board_leaf(
            "tdma-i8-whole",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "tdma-i8-whole-positive",
            resource_budget="4k-payload-plus-guards",
        ),
        _board_leaf(
            "tdma-fp16-bf16-strided",
            "held-out",
            "rank-one-worker0",
            DATAMOVE_EXTENDED_CATALOG,
            "tdma-fp16-bf16-strided",
            resource_budget="two-128k-ddr-slots",
        ),
        _board_leaf(
            "tdma-fp16-bf16-raw-crt",
            "held-out",
            "rank-one-worker0",
            DATAMOVE_EXTENDED_CATALOG,
            "tdma-fp16-bf16-raw-crt",
            resource_budget="two-128k-ddr-slots",
        ),
    ),
    "tdma-bool-fill": (
        _non_board_leaf(
            "tdma-native-bool-exclusion",
            "calibration",
            "static-negative",
            NCC_CATALOG,
            "tdma-native-bool-exclusion",
            reason=(
                "native BOOL fill did not provide bounded completion on the "
                "qualified profile and is excluded from ordinary execution"
            ),
        ),
        _board_leaf(
            "tdma-bool-to-i8-physical-fill",
            "held-out",
            "rank-one-worker0-manual",
            NCC_CATALOG,
            "tdma-bool-to-i8-physical-fill",
            resource_budget="one-physical-block-plus-guards",
        ),
    ),
    "tdma-movement-variants": (
        _board_leaf(
            "tdma-layout-materialization",
            "held-out",
            "rank-one-worker0",
            DATAMOVE_CATALOG,
            "tdma-layout-materialization",
            resource_budget="two-64k-ddr-slots",
        ),
    ),
    "spm-capacity-reservation": (
        _board_leaf(
            "spm-lower-upper-capacity-boundaries",
            "calibration",
            "rank-one-worker0",
            SPM_CATALOG,
            "lower-upper-capacity-boundaries",
            resource_budget="profile-allocatable-spm",
        ),
        _non_board_leaf(
            "spm-reservation-range-negatives",
            "held-out",
            "static-negative",
            SPM_CATALOG,
            "reservation-range-negatives",
            reason=(
                "reserved, crossing, zero-length, overflow and out-of-range "
                "requests are rejected before packet construction"
            ),
        ),
    ),
    "spm-alignment-bank": (
        _board_leaf(
            "spm-relative-offset-sweep",
            "calibration",
            "rank-one-worker0",
            SPM_CATALOG,
            "relative-offset-sweep",
            resource_budget="profile-allocatable-spm",
        ),
        _leaf(
            "spm-non-preferred-geometry-observation",
            "held-out",
            "board-observation",
            "rank-one-worker0",
            bindings=_catalog_groups(
                SPM_CATALOG,
                "non-preferred-geometry-roundtrip",
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="profile-allocatable-spm",
            reason=(
                "256-byte alignment is preferred rather than statically "
                "proven legal; bounded exact round-trips observe whether "
                "64/128/192-byte bases and 128/384-byte lengths execute"
            ),
        ),
        _leaf(
            "spm-rdma-wdma-engine-access-delegated",
            "calibration",
            "delegated-positive",
            "delegated-board-runner",
            bindings=_catalog_groups(
                SPM_CATALOG,
                "rdma-wdma-engine-access-delegated",
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="profile-allocatable-spm",
            reason=(
                "RDMA and WDMA SPM access are executed by the exact "
                "round-trip package referenced by each delegation row"
            ),
        ),
        _leaf(
            "spm-ct-ne-tdma-engine-access-delegated",
            "calibration",
            "delegated-positive",
            "delegated-board-runner",
            bindings=_catalog_groups(
                SPM_CATALOG,
                "ct-ne-tdma-engine-access-delegated",
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="profile-allocatable-spm",
            reason=(
                "CT, NE and TDMA SPM access are executed by the dedicated "
                "memory-descriptor package referenced by each row"
            ),
        ),
        _board_leaf(
            "spm-five-engine-access-concrete",
            "calibration",
            "rank-one-worker0",
            MEMORY_DESCRIPTOR_CATALOG,
            "spm-five-engine-access",
            resource_budget="256k-memory-descriptor-resource",
        ),
        _leaf(
            "spm-address-relations-delegated",
            "held-out",
            "delegated-positive",
            "delegated-board-runner",
            bindings=_catalog_groups(
                SPM_CATALOG,
                "exact-partial-adjacent-disjoint-strided",
            ),
            oracle=("raw-full-result-capture", "request-echo"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="profile-allocatable-spm",
            reason=(
                "the SPM planner rows delegate bounded execution to the "
                "memory-descriptor relation package"
            ),
        ),
        _observation_leaf(
            "spm-address-relations-observation",
            "held-out",
            "rank-one-worker0",
            MEMORY_DESCRIPTOR_CATALOG,
            "spm-exact-partial-adjacent-disjoint-strided",
            resource_budget="256k-memory-descriptor-resource",
            reason=(
                "RAW/WAR/WAW/RAR across exact, partial, adjacent, disjoint "
                "and strided ranges are captured without assuming overlap"
            ),
        ),
        _leaf(
            "spm-physical-layout",
            "held-out",
            "delegated-positive",
            "delegated-board-runner",
            bindings=_catalog_groups(
                SPM_CATALOG,
                "tensor-cx-ncx-physical-layout",
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="bounded-datamove-package",
            reason=(
                "physical-layout execution is owned by the concrete bounded "
                "DataMove cases referenced by each delegation row"
            ),
        ),
        _board_leaf(
            "spm-slot-lifetime-reuse",
            "production-vertical",
            "rank-one-worker0",
            SPM_CATALOG,
            "slot-lifetime-reuse",
            resource_budget="two-independent-spm-slots",
        ),
        _non_board_leaf(
            "spm-slot-lifetime-negatives",
            "production-vertical",
            "static-negative",
            SPM_CATALOG,
            "slot-lifetime-negatives",
            reason=(
                "reuse before completion, dangling views, insufficient "
                "double-slot capacity and physical slot aliasing are "
                "rejected before device submission"
            ),
        ),
        _leaf(
            "spm-bank-engine-pair-controls-delegated",
            "calibration",
            "delegated-positive",
            "delegated-board-runner",
            bindings=_catalog_groups(
                SPM_CATALOG,
                "bank-engine-pair-controls",
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="profile-allocatable-spm",
            reason=(
                "SPM bank pair rows delegate to serial/window controls at "
                "three distinct relative offsets per engine pair"
            ),
        ),
        _board_leaf(
            "spm-bank-engine-pair-controls-concrete",
            "calibration",
            "rank-one-worker0",
            MEMORY_DESCRIPTOR_CATALOG,
            "spm-bank-engine-pair-controls",
            resource_budget="256k-memory-descriptor-resource",
        ),
    ),
    "ddr-cache-coherence": (
        _board_leaf(
            "cache-four-visibility-directions",
            "calibration",
            "rank-one-worker0",
            CACHE_CATALOG,
            "four-visibility-directions",
            resource_budget="16k-payload-plus-guards",
        ),
        _non_board_leaf(
            "cache-same-session-stale-invalidation",
            "held-out",
            "isolated-deferred",
            CACHE_CATALOG,
            "cache-same-session-stale-invalidation",
            reason=(
                "the current one-shot runtime frees allocations after each "
                "launch, so the state is unreachable.  Reactivate only with "
                "persistent-session allocation reuse, where host mutation of "
                "a cache-primed address creates a real invalidate boundary"
            ),
        ),
        _board_leaf(
            "ddr-same-allocation-offset-sweep",
            "calibration",
            "rank-one-worker0",
            CACHE_CATALOG,
            "ddr-same-allocation-offset-sweep",
            resource_budget="two-64k-ddr-windows",
        ),
        _board_leaf(
            "ddr-rdma-wdma-pair-controls",
            "calibration",
            "rank-one-worker0-manual",
            CACHE_CATALOG,
            "ddr-rdma-wdma-pair-controls",
            resource_budget="two-64k-ddr-windows",
        ),
        _leaf(
            "ddr-large-stride-burst-tail-delegated",
            "held-out",
            "delegated-positive",
            "delegated-board-runner",
            bindings=_catalog_groups(
                CACHE_CATALOG,
                "ddr-large-stride-burst-tail",
            ),
            oracle=("independent-expected", "full-result"),
            guards=("physical-span", "prefix-suffix-canary"),
            completion=(
                "matching-completion",
                "terminal-status",
                "cleanup",
            ),
            resource_budget="256k-memory-descriptor-resource",
            reason=(
                "legacy cache rows delegate 64KiB, 1D/2D/3D, default-burst "
                "boundary and tail execution to the descriptor package"
            ),
        ),
        _board_leaf(
            "ddr-large-stride-default-burst-tail",
            "held-out",
            "rank-one-worker0",
            MEMORY_DESCRIPTOR_CATALOG,
            "ddr-large-stride-default-burst-tail",
            resource_budget="256k-memory-descriptor-resource",
        ),
        _non_board_leaf(
            "ddr-configurable-burst-knob",
            "held-out",
            "static-negative",
            MEMORY_DESCRIPTOR_CATALOG,
            "ddr-configurable-burst-static-boundary",
            reason=(
                "the owned typed RDMA/WDMA ABI exposes no configurable "
                "burst field; default boundaries remain executable"
            ),
        ),
    ),
    "queue-shape-submission": (
        _board_leaf(
            "five-engine-n1-n2-n4",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "five-engine-n1-n2-n4",
            resource_budget="documented-safe-ordinary-depth",
        ),
        _board_leaf(
            "documented-depth-manual",
            "held-out",
            "rank-one-worker0-manual",
            NCC_CATALOG,
            "documented-depth-manual",
            resource_budget="exact-documented-depth",
        ),
        _board_leaf(
            "depth-plus-one-manual",
            "held-out",
            "rank-one-worker0-manual",
            NCC_CATALOG,
            "depth-plus-one-manual",
            resource_budget="exact-documented-depth-plus-one",
        ),
    ),
    "worker-scope": (
        _board_leaf(
            "workers012-disjoint-routing",
            "calibration",
            "rank-one-workers012",
            NCC_CATALOG,
            "workers012-disjoint-routing",
            resource_budget="one-disjoint-output-per-worker",
        ),
        _board_leaf(
            "default-byworker-wait-controls",
            "calibration",
            "rank-one-workers012",
            NCC_CATALOG,
            "default-byworker-wait-controls",
            resource_budget="one-disjoint-output-per-worker",
        ),
        _observation_leaf(
            "default-wait-nondefault-scope",
            "held-out",
            "rank-one-workers012",
            NCC_CATALOG,
            "default-wait-nondefault-scope",
            resource_budget="large-ne-backlog-plus-disjoint-control",
            reason=(
                "default and worker-scoped completion are compared with a "
                "bounded pending snapshot, then all workers are safely drained"
            ),
        ),
    ),
    "cross-engine-overlap": (
        _board_leaf(
            "ten-engine-pairs-two-orders-controls",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "ten-engine-pairs-two-orders-controls",
            resource_budget="r2-all-pairs-r4-non-tdma",
        ),
        _board_leaf(
            "large-backlog-compute-movement",
            "held-out",
            "rank-one-worker0",
            NCC_CATALOG,
            "large-backlog-compute-movement",
            resource_budget="256k-resource-with-64k-and-large-ne-ct-backlogs",
        ),
        _board_leaf(
            "multi-engine-sustained-parallel-controls",
            "held-out",
            "rank-one-worker0",
            MEMORY_DESCRIPTOR_CATALOG,
            "multi-engine-sustained-parallel-controls",
            resource_budget="2m-memory-descriptor-resource",
        ),
        _observation_leaf(
            "double-slot-hardware-observation",
            "held-out",
            "rank-one-worker0",
            NCC_CATALOG,
            "double-slot-hardware-observation",
            resource_budget="two-16k-slots-plus-guards",
            reason=(
                "hand-built 1/2/3/4-iteration serial/window double-slot "
                "execution captures hardware behavior without claiming the "
                "production compiler software-pipeline vertical is complete"
            ),
        ),
    ),
    "address-dependency": (
        _board_leaf(
            "dependency-raw-war-waw-rar",
            "calibration",
            "rank-one-worker0-manual",
            NCC_CATALOG,
            "dependency-raw-war-waw-rar",
            resource_budget="qualified-overlap-pairs-only",
        ),
        _board_leaf(
            "dependency-exact-partial-adjacent",
            "calibration",
            "rank-one-worker0-manual",
            NCC_CATALOG,
            "dependency-exact-partial-adjacent",
            resource_budget="qualified-overlap-pairs-only",
        ),
        _observation_leaf(
            "dependency-strided-envelope",
            "held-out",
            "rank-one-worker0",
            NCC_CATALOG,
            "dependency-strided-envelope",
            resource_budget="bounded-1d-2d-3d-envelopes-plus-hole-guards",
            reason=(
                "serial/window strided hazards retain compact output, full "
                "envelope and hole observations before compiler promotion"
            ),
        ),
    ),
    "issue-overhead": (
        _board_leaf(
            "wrapper-prebuilt-same-sequence",
            "calibration",
            "rank-one-worker0-manual",
            NCC_CATALOG,
            "wrapper-prebuilt-same-sequence",
            resource_budget="same-prebuilt-packet-sequence",
        ),
    ),
    "local-completion": (
        _board_leaf(
            "five-engine-wait-each-window",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "five-engine-wait-each-window",
            resource_budget="ordinary-safe-depth",
        ),
        _board_leaf(
            "ncc-producer-consumer-representative",
            "calibration",
            "rank-one-worker0",
            NCC_CATALOG,
            "ncc-producer-consumer-representative-positive",
            resource_budget="one-producer-consumer-chain",
        ),
        _observation_leaf(
            "ncc-producer-consumer-boundaries",
            "held-out",
            "rank-one-worker0",
            NCC_CATALOG,
            "ncc-producer-consumer-all-directions",
            resource_budget="bounded-all-direction-producer-consumer-chains",
            reason=(
                "the group includes mapped-SPM boundary controls that prove "
                "bounded completion and visibility but do not distinguish a "
                "unique completion-scope mechanism"
            ),
        ),
    ),
    "cross-worker-join": (
        _board_leaf(
            "cross-worker-single-pair-triple-masks",
            "calibration",
            "rank-one-workers012",
            NCC_CATALOG,
            "cross-worker-single-pair-triple-masks",
            resource_budget="three-disjoint-worker-outputs",
        ),
        _observation_leaf(
            "cross-worker-unjoined-boundary",
            "held-out",
            "rank-one-workers012",
            NCC_CATALOG,
            "cross-worker-unjoined-boundary",
            resource_budget="three-worker-backlogs-plus-six-proper-subset-masks",
            reason=(
                "six proper-subset joins retain the boundary snapshot and "
                "then drain all participants before teardown"
            ),
        ),
        _non_board_leaf(
            "wait-mask-nonparticipant-host-negative",
            "held-out",
            "static-negative",
            NCC_CATALOG,
            "wait-mask-nonparticipant-static-negative",
            reason=(
                "typed Plan validation rejects wait masks naming a "
                "non-participating worker before serialization"
            ),
        ),
    ),
    "direct-dte": (
        _board_leaf(
            "dte-four-ordered-ncc-modes",
            "calibration",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-ordered-interaction",
            resource_budget="one-logical-slot-per-rank",
        ),
        _board_leaf(
            "dte-event-safe-reuse",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-reuse-after-event",
            resource_budget="two-epoch-slot-reuse",
        ),
        _board_leaf(
            "dte-broadcast-multi-destination",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-broadcast",
            resource_budget="two-destination-slots-per-source-rank",
        ),
        _non_board_leaf(
            "dte-invalid-coordinate-host-negative",
            "held-out",
            "static-negative",
            DTE_NCC_CATALOG,
            "direct-dte-host-static-negative",
            reason=(
                "typed host mesh validation rejects coordinates outside the "
                "qualified mesh before serialization or device submission"
            ),
        ),
        _observation_leaf(
            "dte-device-error-observation",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-device-error-observation",
            resource_budget="one-16b-logical-slot-per-rank",
            reason=(
                "early source reuse, invalid FSM and unknown event are "
                "synchronous bounded error observations; shadow status "
                "retains the error before normal terminal cleanup"
            ),
        ),
        _observation_leaf(
            "dte-sender-raw-async-controls",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-sender-async-controls",
            resource_budget="four-guarded-64k-spm-regions-per-rank",
            reason=(
                "receiver-first test-only raw sender controls repeat serial "
                "and send_async-to-wait_done issue windows without changing "
                "the production CRT; exact guards, return codes, receiver "
                "completion and terminal publication gate PMU comparison"
            ),
        ),
        _observation_leaf(
            "dte-raw-four-source-fanin",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-four-source-fanin",
            resource_budget="four-receiver-fsms-and-four-guarded-source-slots",
            reason=(
                "the owner-backed raw case classifies four-source fan-in "
                "correctness and receiver-FSM capacity only; it is not an "
                "operator collective or a calibrated contention-cost claim"
            ),
        ),
        _observation_leaf(
            "dte-raw-broadcast-fanout-layout",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-raw-broadcast",
            resource_budget="one-guarded-source-and-up-to-fifteen-destinations",
            reason=(
                "raw broadcast fanout and destination-count encoding remain "
                "board observations; exact copies do not qualify an "
                "operator-level collective or device cost"
            ),
        ),
        _observation_leaf(
            "dte-raw-scatter-fanout-layout",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-raw-scatter",
            resource_budget="one-guarded-source-and-up-to-fifteen-destinations",
            reason=(
                "raw scatter fanout, source slicing and destination-count "
                "encoding remain board observations; they do not qualify "
                "operator AllToAll semantics or device cost"
            ),
        ),
        _observation_leaf(
            "dte-raw-shuffle-source-stride",
            "held-out",
            "full-card-16-rank",
            DTE_NCC_CATALOG,
            "direct-dte-raw-shuffle",
            resource_budget="one-guarded-source-and-one-destination",
            reason=(
                "raw shuffle covers one remote destination and source-side "
                "1D strided gather only; it does not qualify "
                "multi-destination communication, a collective, or cost"
            ),
        ),
        _non_board_leaf(
            "dte-receiver-unprepared",
            "held-out",
            "isolated-deferred",
            DTE_NCC_CATALOG,
            "direct-dte-unsafe-isolation",
            reason=(
                "send without a prepared receiver enters direct_sync_wait "
                "without a device timeout and may permanently block"
            ),
        ),
    ),
    "multi-tile-arrival": (
        _board_leaf(
            "full-card-two-epoch-arrival",
            "calibration",
            "full-card-16-rank",
            BARRIER_CATALOG,
            "multi-tile-arrival",
            resource_budget="sixteen-version-matched-slots",
        ),
        _non_board_leaf(
            "unsupported-subgroup-arrival",
            "held-out",
            "static-negative",
            BARRIER_CATALOG,
            "barrier-participant-negative",
            reason=(
                "the version-matched barrier observes sixteen fixed slots; "
                "unsafe subgroup participant counts are rejected by host"
            ),
        ),
    ),
    "host-launch-runtime": (
        _leaf(
            "kernel-rank1-rank16-launch",
            "calibration",
            "board-positive",
            "rank-one-and-full-card",
            bindings=(
                _catalog_groups(
                    RUNTIME_RANK_ONE_CATALOG,
                    "rank-one-per-rank-add",
                )
                + _catalog_groups(
                    RUNTIME_CATALOG,
                    "rank16-kernel-add",
                )
            ),
            oracle=("independent-expected", "all-rank-full-result"),
            guards=("typed-resource-bindings", "rank-slice-domain"),
            completion=("terminal-status", "D2H", "cleanup"),
            resource_budget="runtime-owned-rank-resources",
        ),
        _board_leaf(
            "terminal-d2h-cleanup-heartbeat",
            "held-out",
            "rank-one-and-full-card",
            TRANSPORT_PMU_CATALOG,
            "runtime-publication-positive",
            resource_budget="runtime-owned-rank-resources",
        ),
        _non_board_leaf(
            "runtime-timeout-cleanup-contract",
            "held-out",
            "static-negative",
            TRANSPORT_PMU_CATALOG,
            "runtime-publication-negative",
            reason=(
                "outer deadline and cleanup are validated without issuing an "
                "unsafe hardware protocol violation"
            ),
        ),
    ),
    "ncc-pmu-basis": (
        _board_leaf(
            "ncc-pmu-stable-read-enable-scope",
            "calibration",
            "rank-one-read-only",
            NCC_PMU_CATALOG,
            "stable-read-enable-scope",
            resource_budget="read-only-registers",
        ),
        _board_leaf(
            "ncc-pmu-workload-delta-basis",
            "held-out",
            "rank-one-worker0",
            NCC_CATALOG,
            "pmu-workload-delta-basis",
            resource_budget="one-known-good-engine-window",
        ),
    ),
    "transport-pmu-basis": (
        _observation_leaf(
            "dte-spm-counter-payload-sweep",
            "calibration",
            "full-card-16-rank",
            TRANSPORT_PMU_CATALOG,
            "dte-spm-counter-payload-sweep",
            resource_budget="one-64b-slot-per-rank",
            reason=(
                "the payload sweep retains raw modulo-2^64 deltas, but the "
                "counter measurement basis and event units remain "
                "uncalibrated"
            ),
        ),
        _non_board_leaf(
            "tmnoc-counter-offset-unavailable",
            "held-out",
            "static-negative",
            TRANSPORT_PMU_CATALOG,
            "tmnoc-counter-offset-unavailable",
            reason=(
                "the version-matched header exposes only a base address and "
                "no decoded read-only counter offsets"
            ),
        ),
    ),
    "scalar-csr-ordinary-issue": (
        _non_board_leaf(
            "scalar-csr-ordinary-issue-rejected",
            "calibration",
            "static-negative",
            NCC_CATALOG,
            "scalar-csr-ordinary-issue-rejected",
            reason=(
                "SCALAR and CSR are outside the typed ordinary NCC execute "
                "domain and are rejected before packet submission"
            ),
        ),
    ),
}


LEAVES_BY_KEY = {
    leaf.key: leaf
    for leaves in CALIBRATION_LEAVES_BY_DOMAIN.values()
    for leaf in leaves
}
