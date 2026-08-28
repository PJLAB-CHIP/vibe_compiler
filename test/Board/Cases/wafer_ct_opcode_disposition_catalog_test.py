#!/usr/bin/env python3
"""Require exactly one closed disposition for every CT opcode 0..186."""

from __future__ import annotations

from collections import Counter

import wafer_board_ncc_execution_probe_runner as ncc_catalog
import wafer_ct_opcode_disposition_catalog as catalog
import wafer_instruction_family_catalog as instruction_catalog


def main() -> int:
    assert len(catalog.CATALOG) == 187
    assert set(catalog.BY_OPCODE) == set(range(187))
    assert len(catalog.BY_OPCODE) == len(catalog.CATALOG)
    allowed = {
        "board-executable",
        "board-observation",
        "static-negative",
        "isolated-deferred",
        "typed-profile",
    }
    for row in catalog.CATALOG:
        assert row.disposition in allowed
        if row.disposition in {"board-executable", "board-observation"}:
            assert row.evidence
            assert row.reason is None
        elif row.disposition == "isolated-deferred":
            assert not row.evidence
            assert row.reason
        elif row.disposition == "typed-profile":
            assert row.evidence == (
                "wafer-ct-reduce-pool-capability-catalog-python",
            )
            assert row.reason
        else:
            assert row.evidence
            assert row.reason
        assert row.opcode_name
        assert row.family

    counts = Counter(row.disposition for row in catalog.CATALOG)
    assert counts["static-negative"] == 2
    assert counts["isolated-deferred"] == 0
    assert counts["board-executable"] == 162
    assert counts["board-observation"] == 10
    assert counts["typed-profile"] == 13
    assert catalog.BY_OPCODE[139].evidence[:2] == (
        "ct-op139-convert-int8-fp16-main",
        "ct-op139-convert-int8-fp16-tail",
    )
    assert all(
        catalog.BY_OPCODE[opcode].disposition == "typed-profile"
        for opcode in range(111, 124)
    )
    assert catalog.BY_OPCODE[178].disposition == "board-observation"
    assert catalog.BY_OPCODE[178].evidence == (
        "peripheral-argmin-f16",
        "peripheral-argmin-negative-f16-observed",
    )
    assert catalog.BY_OPCODE[182].disposition == "board-observation"
    concrete_evidence = {
        case.name for case in catalog.vector_catalog.CATALOG
    } | {
        case.name for case in catalog.convert_catalog.CATALOG
    } | {
        case.name for case in catalog.datamove_catalog.CATALOG
    } | {
        case.name for case in catalog.datamove_catalog.extended.ALL_CASES
    } | {
        case.name for case in instruction_catalog.CATALOG
    } | {
        case.name for case in ncc_catalog.NO_CARD_PROTOCOL_CASES
    }
    registered_gate_evidence = {
        "wafer-ct-opcode-disposition-catalog-python",
        "wafer-ct-reduce-pool-capability-catalog-python",
        "wafer-ct-vector-calibration-catalog-python",
        "wafer-runtime-ct-vector-calibration-probe-no-card",
    }
    unresolved_evidence = {
        evidence
        for row in catalog.CATALOG
        for evidence in row.evidence
        if evidence not in concrete_evidence
        and evidence not in registered_gate_evidence
    }
    assert not unresolved_evidence, (
        f"opcode dispositions reference unknown evidence: "
        f"{sorted(unresolved_evidence)}"
    )
    positive = catalog.CALIBRATION_LEAF_BINDINGS[
        "ct-reduce-pool-unpool-peripheral-positive"
    ]
    observed = catalog.CALIBRATION_LEAF_BINDINGS[
        "ct-reduce-pool-unpool-peripheral-observed"
    ]
    static_negative = catalog.CALIBRATION_LEAF_BINDINGS[
        "ct-reduce-pool-unpool-peripheral-static-negative"
    ]
    isolated_deferred = catalog.CALIBRATION_LEAF_BINDINGS[
        "ct-reduce-raw-axis-isolated-deferred"
    ]
    assert set(catalog.CALIBRATION_LEAF_BINDINGS) == {
        "ct-reduce-pool-unpool-peripheral-positive",
        "ct-reduce-pool-unpool-peripheral-observed",
        "ct-reduce-pool-unpool-peripheral-static-negative",
        "ct-reduce-raw-axis-isolated-deferred",
    }
    assert positive and observed and static_negative and isolated_deferred
    assert all(row.disposition == "board-executable" for row in positive)
    assert all(row.disposition == "board-observation" for row in observed)
    assert all(
        row.disposition == "static-negative" for row in static_negative
    )
    assert all(
        row.disposition == "isolated-deferred"
        for row in isolated_deferred
    )
    assert not catalog.reduce_pool_catalog.UNKNOWN_ROWS
    bound = positive + observed + static_negative + isolated_deferred
    expected = catalog.reduce_pool_catalog.CATALOG + tuple(
        row
        for row in catalog.CATALOG
        if row.opcode in catalog._PERIPHERAL_OPCODES
    )
    assert len(bound) == len(set(bound)) == len(expected)
    assert set(bound) == set(expected)
    print(
        "wafer_ct_opcode_disposition_catalog_test: "
        f"opcodes=187 dispositions={dict(sorted(counts.items()))} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
