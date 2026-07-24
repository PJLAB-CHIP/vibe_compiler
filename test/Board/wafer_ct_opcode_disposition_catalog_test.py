#!/usr/bin/env python3
"""Require exactly one closed disposition for every CT opcode 0..186."""

from __future__ import annotations

from collections import Counter

import wafer_board_ncc_execution_probe_test as ncc_catalog
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
    }
    for row in catalog.CATALOG:
        assert row.disposition in allowed
        if row.disposition in {"board-executable", "board-observation"}:
            assert row.evidence
            assert row.reason is None
        elif row.disposition == "isolated-deferred":
            assert not row.evidence
            assert row.reason
        else:
            assert row.evidence
            assert row.reason
        assert row.opcode_name
        assert row.family

    counts = Counter(row.disposition for row in catalog.CATALOG)
    assert counts["static-negative"] == 2
    assert counts["isolated-deferred"] == 0
    assert counts["board-executable"] == 177
    assert counts["board-observation"] == 8
    assert catalog.BY_OPCODE[139].evidence[:2] == (
        "ct-op139-convert-int8-fp16-main",
        "ct-op139-convert-int8-fp16-tail",
    )
    concrete_evidence = {
        case.name for case in catalog.vector_catalog.CATALOG
    } | {
        case.name for case in catalog.convert_catalog.CATALOG
    } | {
        case.name for case in catalog.datamove_catalog.CATALOG
    } | {
        case.name for case in catalog.datamove_catalog.extended.CATALOG
    } | {
        case.name for case in instruction_catalog.CATALOG
    } | {
        case.name for case in ncc_catalog.NO_CARD_PROTOCOL_CASES
    }
    registered_gate_evidence = {
        "wafer-ct-opcode-disposition-catalog-python",
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
    assert set(catalog.CALIBRATION_LEAF_BINDINGS) == {
        "ct-reduce-pool-unpool-peripheral-positive",
        "ct-reduce-pool-unpool-peripheral-observed",
        "ct-reduce-pool-unpool-peripheral-static-negative",
    }
    assert positive and observed and static_negative
    assert all(row.disposition == "board-executable" for row in positive)
    assert all(row.disposition == "board-observation" for row in observed)
    assert all(
        row.disposition == "static-negative" for row in static_negative
    )
    bound = positive + observed + static_negative
    expected = tuple(
        row
        for row in catalog.CATALOG
        if row.opcode
        in catalog._REDUCE_POOL_UNPOOL_PERIPHERAL_OPCODES
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
