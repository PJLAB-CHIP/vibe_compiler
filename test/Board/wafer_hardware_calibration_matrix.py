#!/usr/bin/env python3
"""Machine-auditable preparation state for every hardware calibration row."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CalibrationDomain:
    key: str
    document_label: str
    execution_scope: str
    preparation: str
    positive_assets: tuple[str, ...]
    negative_assets: tuple[str, ...]
    no_card_tests: tuple[str, ...]
    remaining_preparation: tuple[str, ...] = ()


def _domain(
    key: str,
    document_label: str,
    execution_scope: str,
    preparation: str,
    *,
    positive: tuple[str, ...] = (),
    negative: tuple[str, ...] = (),
    tests: tuple[str, ...] = (),
    remaining: tuple[str, ...] = (),
) -> CalibrationDomain:
    return CalibrationDomain(
        key=key,
        document_label=document_label,
        execution_scope=execution_scope,
        preparation=preparation,
        positive_assets=positive,
        negative_assets=negative,
        no_card_tests=tests,
        remaining_preparation=remaining,
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
DTE_NCC = (
    "test/Board/wafer_board_dte_ncc_execution_probe_test.py",
    "test/Board/Inputs/wafer_dte_ncc_execution_probe.c",
)
FULL_CARD_BARRIER = (
    "test/Board/wafer_board_full_card_barrier_probe_test.py",
    "test/Board/Inputs/wafer_full_card_barrier_probe.c",
)
NCC_PMU = (
    "test/Board/wafer_board_ncc_pmu_probe_test.py",
    "test/Board/Inputs/wafer_ncc_pmu_readonly_probe.c",
)


CALIBRATION_DOMAINS = (
    _domain(
        "profile-qualification",
        "profile qualification",
        "rank-one-read-only",
        "ready",
        positive=NCC_PMU,
        tests=("wafer-runtime-ncc-pmu-readonly-probe-no-card",),
    ),
    _domain(
        "constructor-ownership",
        "constructor ownership",
        "rank-one-worker0",
        "ready",
        positive=INSTRUCTION_FAMILY,
        tests=(
            "wafer-instruction-family-catalog-python",
            "wafer-runtime-instruction-family-probe-no-card",
        ),
    ),
    _domain(
        "execute-result",
        "execute result",
        "rank-one-worker0",
        "ready",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    _domain(
        "packet-routing-range",
        "packet routing/range",
        "rank-one-workers012",
        "ready",
        positive=NCC_EXECUTION,
        tests=(
            "wafer-ncc-probe-plan",
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-execution-probe-no-card",
        ),
    ),
    _domain(
        "ct-numeric-form",
        "CT numeric/form",
        "rank-one-worker0",
        "in-progress",
        positive=INSTRUCTION_FAMILY + CT_VECTOR_CALIBRATION,
        tests=(
            "wafer-instruction-family-catalog-python",
            "wafer-ct-vector-calibration-catalog-python",
            "wafer-runtime-ct-vector-calibration-probe-no-card",
        ),
        remaining=(
            "opcode 111..186 disposition and executable/deferred closure",
            "convert special-value and deterministic-rounding held-out rows",
        ),
    ),
    _domain(
        "instruction-physical-layout",
        "instruction × physical layout",
        "rank-one-worker0",
        "in-progress",
        positive=(
            INSTRUCTION_FAMILY
            + CT_VECTOR_CALIBRATION
            + PHYSICAL_TENSOR_CODEC
            + NE_CALIBRATION
        ),
        tests=(
            "wafer-physical-tensor-codec-python",
            "wafer-ne-calibration-catalog-python",
            "wafer-runtime-ne-calibration-probe-no-card",
        ),
        remaining=(
            "CT Tensor-to-Cx/NCx explicit materialization representatives",
            "typed illegal direct-layout negative gates",
        ),
    ),
    _domain(
        "datamove-layout",
        "DataMove/layout",
        "rank-one-worker0",
        "in-progress",
        positive=INSTRUCTION_FAMILY,
        remaining=(
            "concat/broadcast/transform/mask inventory and oracles",
            "device dispatch and no-card link closure",
        ),
    ),
    _domain(
        "ne-numeric-layout",
        "NE numeric/layout",
        "rank-one-worker0",
        "ready",
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
        "ready",
        positive=NCC_EXECUTION,
        tests=(
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-execution-probe-no-card",
        ),
    ),
    _domain(
        "tdma-memset",
        "TDMA Memset",
        "rank-one-worker0",
        "ready",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    _domain(
        "tdma-bool-fill",
        "TDMA BOOL fill",
        "rank-one-worker0-manual",
        "ready",
        positive=NCC_EXECUTION,
        negative=("test/Board/wafer_ncc_probe_protocol_test.py",),
        tests=(
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-execution-probe-no-card",
        ),
    ),
    _domain(
        "tdma-movement-variants",
        "TDMA movement variants",
        "rank-one-worker0",
        "in-progress",
        positive=INSTRUCTION_FAMILY + NCC_EXECUTION,
        remaining=("all public movement kind dispositions and oracles",),
    ),
    _domain(
        "spm-capacity-reservation",
        "SPM capacity/reservation",
        "rank-one-static-and-worker0",
        "ready",
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
        "ready",
        positive=NCC_EXECUTION + SPM_CALIBRATION,
        tests=(
            "wafer-spm-calibration-catalog-python",
            "wafer-runtime-spm-calibration-probe-no-card",
        ),
    ),
    _domain(
        "ddr-cache-coherence",
        "DDR/cache/coherence",
        "rank-one-and-full-card",
        "in-progress",
        positive=DTE_NCC + NCC_EXECUTION,
        remaining=("complete four-direction visibility matrix",),
    ),
    _domain(
        "queue-shape-submission",
        "queue shape与连续提交边界",
        "rank-one-worker0-manual",
        "ready",
        positive=NCC_EXECUTION,
        tests=(
            "wafer-ncc-probe-plan",
            "wafer-ncc-probe-protocol-python",
            "wafer-runtime-ncc-execution-probe-no-card",
        ),
    ),
    _domain(
        "worker-scope",
        "worker scope",
        "rank-one-workers012-manual",
        "ready",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    _domain(
        "cross-engine-overlap",
        "cross-engine overlap",
        "rank-one-worker0",
        "ready",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    _domain(
        "address-dependency",
        "address dependency",
        "rank-one-worker0-manual",
        "ready",
        positive=NCC_EXECUTION,
        negative=(
            "test/Board/wafer_ncc_hazard_relation_test.c",
            "test/Board/Inputs/wafer_ncc_hazard_relation.c",
            "test/Board/Inputs/wafer_ncc_hazard_relation.h",
        ),
        tests=(
            "wafer-ncc-hazard-relation",
            "wafer-ncc-probe-protocol-python",
        ),
    ),
    _domain(
        "issue-overhead",
        "issue overhead",
        "rank-one-worker0-manual",
        "ready",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    _domain(
        "local-completion",
        "local completion",
        "rank-one-workers012-manual",
        "ready",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    _domain(
        "cross-worker-join",
        "cross-worker join",
        "rank-one-workers012",
        "ready",
        positive=NCC_EXECUTION,
        tests=("wafer-runtime-ncc-execution-probe-no-card",),
    ),
    _domain(
        "direct-dte",
        "Direct DTE",
        "full-card-16-rank",
        "ready",
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
        "ready",
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
        "ready",
        positive=(
            "test/Board/wafer_board_all_rank_add_test.py",
            "test/Board/wafer_board_direct_dte_collective_test.py",
        ),
        negative=("test/Board/wafer_board_direct_dte_timeout_test.py",),
        tests=(
            "wafer-runtime-adapter-python",
            "wafer-runtime-kernel-grid-add-no-card",
            "wafer-runtime-model-add-no-card",
            "wafer-runtime-cluster-direct-dte-no-card",
        ),
    ),
    _domain(
        "ncc-pmu-basis",
        "NCC PMU basis",
        "rank-one-worker0",
        "ready",
        positive=NCC_PMU + NCC_EXECUTION,
        tests=(
            "wafer-runtime-ncc-pmu-readonly-probe-no-card",
            "wafer-runtime-ncc-execution-probe-no-card",
        ),
    ),
    _domain(
        "transport-pmu-basis",
        "DTE/SPM/TMNOC PMU",
        "full-card-and-rank-one",
        "in-progress",
        positive=DTE_NCC,
        remaining=("TMNOC scope/unit correlation and held-out sweep",),
    ),
    _domain(
        "scalar-csr-ordinary-issue",
        "SCALAR/CSR ordinary issue",
        "static-negative",
        "ready",
        negative=(
            "test/Board/wafer_ncc_probe_protocol_test.py",
            "test/Board/Inputs/wafer_ncc_execution_probe.c",
        ),
        tests=("wafer-ncc-probe-protocol-python",),
    ),
)


DOMAINS_BY_KEY = {domain.key: domain for domain in CALIBRATION_DOMAINS}
