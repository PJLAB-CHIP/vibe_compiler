#!/usr/bin/env python3
"""Validate collective-traffic global sources and exact host oracles."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

import wafer_board_collective_traffic_behavior_runner as driver
import wafer_collective_traffic_behavior_catalog as catalog


def validate_all_to_all(case: catalog.TrafficBehaviorCase) -> None:
    values = driver.payloads(case)
    lanes = case.input_shape[1]
    assert values.input.shape == (16, 16, lanes, 2)
    assert values.expected.shape == (16, 16 * lanes, 2)
    for destination in range(16):
        for source in range(16):
            begin = source * lanes
            end = begin + lanes
            assert np.array_equal(
                values.expected[destination, begin:end],
                values.input[source, destination],
            )
    assert "stablehlo.transpose" in driver.module_text(case)


def validate_permute(case: catalog.TrafficBehaviorCase) -> None:
    values = driver.payloads(case)
    assert values.input.shape == (16, *case.input_shape)
    assert values.expected.shape == (16, *case.output_shape)
    current = values.input
    for epoch in case.epochs:
        next_values = np.zeros_like(current)
        for source, target in epoch:
            next_values[target] = current[source]
        current = next_values
    assert np.array_equal(values.expected, current)
    module = driver.module_text(case)
    assert "stablehlo.slice" in module
    assert "stablehlo.concatenate" in module


def main() -> int:
    catalog.validate_catalog()
    assert driver.CASES_BY_KEY is catalog.CASES_BY_KEY
    assert tuple(case.case_order for case in catalog.CASES) == tuple(range(11))
    assert {case.participant_count for case in catalog.CASES} == {16}
    assert {case.element_type for case in catalog.CASES} == {"f16"}
    assert {case.disposition for case in catalog.CASES} == {
        catalog.CaseDisposition.SOURCE_CONTRACT_PENDING_LOWERING
    }
    assert all(
        case.structural_evidence == catalog.StructuralEvidenceKind.SOURCE_GRAPH_ONLY
        for case in catalog.CASES
    )
    assert not any(case.proves_target_peer_graph for case in catalog.CASES)
    assert not any(case.proves_physical_route for case in catalog.CASES)
    assert not any(case.proves_device_contention_cost for case in catalog.CASES)

    for case in catalog.CASES:
        driver.validate_source(case)
        if case.collective_kind == catalog.CollectiveKind.ALL_TO_ALL:
            validate_all_to_all(case)
        else:
            validate_permute(case)

    double_epoch = catalog.CASES_BY_KEY[
        "collective-permute-two-epoch-chain-4096b"
    ]
    values = driver.payloads(double_epoch)
    first_epoch = driver.simulate_permute_epochs(
        values.input, (double_epoch.epochs[0],)
    )
    second_epoch = driver.simulate_permute_epochs(
        values.input, (double_epoch.epochs[1],)
    )
    assert not np.array_equal(values.expected, values.input)
    assert not np.array_equal(values.expected, first_epoch)
    assert not np.array_equal(values.expected, second_epoch)

    nearest = catalog.CASES_BY_KEY[
        "collective-permute-cycle-forward-4096b"
    ]
    opposite = catalog.CASES_BY_KEY[
        "collective-permute-opposite-pairs-4096b"
    ]
    nearest_hops = driver.intended_min_hop_summary(nearest)
    opposite_hops = driver.intended_min_hop_summary(opposite)
    assert nearest_hops["maximum_minimum_hops"] == 1
    assert opposite_hops["maximum_minimum_hops"] == 4
    assert opposite_hops["total_minimum_hops"] > nearest_hops[
        "total_minimum_hops"
    ]

    source_rows = {
        item
        for item in catalog.COVERAGE_ITEMS
        if item.disposition
        == catalog.CoverageDisposition.SOURCE_CONTRACT_PENDING_LOWERING
    }
    assert source_rows
    assert {
        case_key for item in source_rows for case_key in item.case_keys
    } == set(catalog.CASE_KEYS)
    assert all(
        item.execution_gate
        == catalog.ExecutionGate.CURRENT_GLOBAL_LOWERING
        for item in source_rows
    )
    raw_rows = {
        item
        for item in catalog.COVERAGE_ITEMS
        if item.disposition
        in {
            catalog.CoverageDisposition.CURRENT_RAW_PROBE,
            catalog.CoverageDisposition.PENDING_BOARD_EXECUTION,
        }
    }
    assert raw_rows
    assert all(item.evidence_refs for item in raw_rows)

    with tempfile.TemporaryDirectory() as directory:
        source = driver.write_source(pathlib.Path(directory), opposite)
        assert (source / "functions" / "forward.mlir").is_file()
        assert (source / "functions" / "forward.meta").is_file()
    print(f"collective_traffic_source_contract: cases={len(catalog.CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
