#!/usr/bin/env python3
"""Validate collective traffic cases, exact oracles, and blocked coverage."""

from __future__ import annotations

import pathlib

import numpy as np

import wafer_board_collective_traffic_behavior_test as driver
import wafer_collective_traffic_behavior_catalog as catalog
import wafer_collective_traffic_post_spmd_carrier as carrier


EXPECTED_CASE_KEYS = (
    "all-to-all-src-dst-lane-256b",
    "all-to-all-src-dst-lane-4096b",
    "all-to-all-src-dst-lane-65536b",
    "collective-permute-cycle-forward-256b",
    "collective-permute-cycle-forward-4096b",
    "collective-permute-cycle-forward-65536b",
    "collective-permute-cycle-reverse-4096b",
    "collective-permute-opposite-pairs-4096b",
    "collective-permute-disjoint-adjacent-pairs-4096b",
    "collective-permute-sparse-roles-4096b",
    "collective-permute-two-epoch-chain-4096b",
)

EXPECTED_COVERAGE_KEYS = (
    "all-to-all-equal-split-traffic-semantics",
    "all-to-all-segmented-or-ragged-traffic",
    "collective-permute-cycle-directions",
    "collective-permute-sparse-role-semantics",
    "collective-permute-double-epoch-semantics",
    "collective-permute-same-physical-buffer-reuse",
    "dte-nearest-endpoint-direction-pair",
    "dte-long-distance-endpoint-graph",
    "dte-disjoint-endpoint-graph",
    "dte-fanout-fanin-one-payload-sweep",
    "dte-fanout-fanin-two-payload-sweep",
    "dte-native-broadcast-fanout-layout-matrix",
    "dte-native-scatter-fanout-layout-matrix",
    "dte-native-shuffle-fanout-layout-matrix",
    "dte-four-source-fanin",
    "dte-eight-source-fanin",
    "dte-fifteen-source-fanin",
    "dte-device-phase-contention-cost",
    "dte-physical-link-route",
    "dte-cross-card-route",
    "tmnoc-per-link-counter",
)


def assert_array_differs(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    assert not np.array_equal(left, right)


def assert_all_to_all_oracle() -> None:
    for payload_bytes in catalog.PAYLOAD_POINTS:
        case = driver.runtime_case(
            catalog.CASES_BY_KEY[
                f"all-to-all-src-dst-lane-{payload_bytes}b"
            ]
        )
        inputs, outputs = driver.payloads(case)
        lanes_per_peer = payload_bytes // (catalog.RANK_COUNT * 4)
        assert len(inputs) == len(outputs) == catalog.RANK_COUNT
        assert all(
            rank[0].shape == (16, lanes_per_peer, 2) for rank in inputs
        )
        assert all(
            rank[0].shape == (1, lanes_per_peer * 16, 2)
            for rank in outputs
        )
        assert all(rank[0].nbytes == payload_bytes for rank in inputs)
        assert all(rank[0].nbytes == payload_bytes for rank in outputs)
        assert all(np.all(np.isfinite(rank[0])) for rank in inputs)
        assert all(np.all(np.isfinite(rank[0])) for rank in outputs)

        lane_samples = sorted({0, 1, lanes_per_peer // 2, lanes_per_peer - 1})
        for destination in range(catalog.RANK_COUNT):
            expected = outputs[destination][0].view(np.uint16)
            for source in range(catalog.RANK_COUNT):
                for lane in lane_samples:
                    record = expected[
                        0, source * lanes_per_peer + lane
                    ]
                    assert record.tolist() == [
                        int(
                            driver.ALL_TO_ALL_PAIR_BITS_BASE
                            + source * catalog.RANK_COUNT
                            + destination
                        ),
                        int(driver.ALL_TO_ALL_LANE_BITS_BASE + lane),
                    ]
            assert_array_differs(
                outputs[destination][0],
                np.roll(
                    outputs[destination][0],
                    lanes_per_peer,
                    axis=1,
                ),
            )
            assert_array_differs(
                outputs[destination][0],
                np.roll(outputs[destination][0], 1, axis=1),
            )
            assert np.any(outputs[destination][0] != 0)
        for destination in range(catalog.RANK_COUNT - 1):
            assert_array_differs(
                outputs[destination][0],
                outputs[destination + 1][0],
            )


def payload_origin(payload: np.ndarray) -> int | None:
    bits = payload.view(np.uint16)
    if not np.any(bits):
        return None
    rank = int(bits[0, 0] - driver.PERMUTE_RANK_BITS_BASE)
    assert 0 <= rank < catalog.RANK_COUNT
    assert np.all(
        bits[:, 0] == driver.PERMUTE_RANK_BITS_BASE + rank
    )
    lanes = np.arange(payload.shape[0], dtype=np.uint16)
    assert np.array_equal(
        bits[:, 1],
        driver.PERMUTE_LANE_BITS_BASE + lanes,
    )
    return rank


def expected_sparse_origins() -> tuple[int | None, ...]:
    origins: list[int | None] = [None] * catalog.RANK_COUNT
    for source, target in catalog.SPARSE_ROLE_PAIRS:
        origins[target] = source
    return tuple(origins)


def epoch_origins(
    epoch: tuple[tuple[int, int], ...],
) -> tuple[int | None, ...]:
    origins: list[int | None] = [None] * catalog.RANK_COUNT
    for source, target in epoch:
        origins[target] = source
    return tuple(origins)


def assert_permute_oracles() -> None:
    expected_origins: dict[str, tuple[int | None, ...]] = {
        "collective-permute-cycle-forward-256b": epoch_origins(
            catalog.FULL_CYCLE_FORWARD
        ),
        "collective-permute-cycle-forward-4096b": epoch_origins(
            catalog.FULL_CYCLE_FORWARD
        ),
        "collective-permute-cycle-forward-65536b": epoch_origins(
            catalog.FULL_CYCLE_FORWARD
        ),
        "collective-permute-cycle-reverse-4096b": epoch_origins(
            catalog.FULL_CYCLE_REVERSE
        ),
        "collective-permute-opposite-pairs-4096b": epoch_origins(
            catalog.OPPOSITE_PAIRS
        ),
        "collective-permute-disjoint-adjacent-pairs-4096b": epoch_origins(
            catalog.DISJOINT_ADJACENT_PAIRS
        ),
        "collective-permute-sparse-roles-4096b": expected_sparse_origins(),
        "collective-permute-two-epoch-chain-4096b": tuple(
            (rank - 4) % catalog.RANK_COUNT
            for rank in range(catalog.RANK_COUNT)
        ),
    }
    rank_inputs = [
        driver.permute_rank_payload(rank)
        for rank in range(catalog.RANK_COUNT)
    ]
    assert len({payload.tobytes() for payload in rank_inputs}) == catalog.RANK_COUNT
    for payload in rank_inputs:
        assert np.any(payload != 0)
        assert np.all(np.isfinite(payload))
        for shift in (1, 64, 256):
            assert_array_differs(payload, np.roll(payload, shift, axis=0))

    for key, origins in expected_origins.items():
        case = driver.runtime_case(catalog.CASES_BY_KEY[key])
        inputs, outputs = driver.payloads(case)
        assert len(inputs) == len(outputs) == catalog.RANK_COUNT
        assert tuple(payload_origin(rank[0]) for rank in outputs) == origins
        module = driver.module_text(case)
        assert module.count('"stablehlo.collective_permute"') == len(
            case.contract.epochs
        )
        for rank, origin in enumerate(origins):
            if origin is None:
                assert np.count_nonzero(outputs[rank][0]) == 0
            else:
                assert np.array_equal(
                    outputs[rank][0],
                    inputs[origin][0],
                )

    double_epoch = catalog.CASES_BY_KEY[
        "collective-permute-two-epoch-chain-4096b"
    ]
    final = driver.simulate_permute_epochs(rank_inputs, double_epoch.epochs)
    first_only = driver.simulate_permute_epochs(
        rank_inputs, (double_epoch.epochs[0],)
    )
    second_only = driver.simulate_permute_epochs(
        rank_inputs, (double_epoch.epochs[1],)
    )
    assert any(
        not np.array_equal(actual, mutation)
        for actual, mutation in zip(final, first_only, strict=True)
    )
    assert any(
        not np.array_equal(actual, mutation)
        for actual, mutation in zip(final, second_only, strict=True)
    )
    assert any(
        not np.array_equal(actual, mutation)
        for actual, mutation in zip(final, rank_inputs, strict=True)
    )


def assert_source_contracts() -> None:
    all_to_all = driver.runtime_case(
        catalog.CASES_BY_KEY["all-to-all-src-dst-lane-4096b"]
    )
    module = driver.module_text(all_to_all)
    assert module.count('"stablehlo.all_to_all"') == 1
    assert "split_dimension = 0" in module
    assert "concat_dimension = 1" in module
    assert "split_count = 16" in module
    assert "tensor<16x64x2xf16>" in module
    assert "tensor<1x1024x2xf16>" in module
    assert carrier.require_supported_module(module) == '"stablehlo.all_to_all"'

    double_epoch = driver.runtime_case(
        catalog.CASES_BY_KEY["collective-permute-two-epoch-chain-4096b"]
    )
    double_module = driver.module_text(double_epoch)
    assert double_module.count('"stablehlo.collective_permute"') == 2
    assert carrier.require_supported_module(
        double_module
    ) == '"stablehlo.collective_permute"'
    try:
        carrier.require_supported_module(module + double_module)
    except RuntimeError:
        pass
    else:
        raise AssertionError("carrier accepted mixed collective traffic kinds")
    try:
        carrier.require_supported_module(double_module + double_module)
    except RuntimeError:
        pass
    else:
        raise AssertionError("carrier accepted four Permute epochs")

    binding = carrier.replicated_binding(
        "argument_index",
        0,
        {"shape": [1024, 2], "dtype": "float16"},
        catalog.RANK_COUNT,
    )
    assert binding["distribution"] == "replicated"
    assert binding["global_shape"] == binding["local_shape"] == [1024, 2]
    assert [row["replica_id"] for row in binding["ranks"]] == list(
        range(catalog.RANK_COUNT)
    )


def assert_intended_demand_summaries() -> None:
    for payload_bytes in catalog.PAYLOAD_POINTS:
        case = catalog.CASES_BY_KEY[
            f"all-to-all-src-dst-lane-{payload_bytes}b"
        ]
        epochs = driver.intended_graph_epochs(case)
        assert len(epochs) == 1
        assert len(epochs[0]) == catalog.RANK_COUNT * (catalog.RANK_COUNT - 1)
        assert all(source != target for source, target in epochs[0])
        summary = driver.intended_min_hop_summary(case)
        assert len(summary) == 1
        assert summary[0]["remote_edge_count"] == 240
        assert summary[0]["local_self_count"] == 0
        assert (
            summary[0]["issue_bytes_per_remote_edge"]
            == payload_bytes // catalog.RANK_COUNT
        )
        assert set(summary[0]["row_major_min_hop_histogram"]) == {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        }

    expected_histograms = {
        "collective-permute-cycle-forward-4096b": {"1": 16},
        "collective-permute-cycle-reverse-4096b": {"1": 16},
        "collective-permute-opposite-pairs-4096b": {"4": 16},
        "collective-permute-disjoint-adjacent-pairs-4096b": {"1": 16},
    }
    for key, histogram in expected_histograms.items():
        summary = driver.intended_min_hop_summary(catalog.CASES_BY_KEY[key])
        assert len(summary) == 1
        assert summary[0]["row_major_min_hop_histogram"] == histogram
        assert summary[0]["remote_edge_count"] == 16
        assert summary[0]["issue_bytes_per_remote_edge"] == 4096


def assert_coverage(repo: pathlib.Path) -> None:
    assert catalog.CASE_KEYS == EXPECTED_CASE_KEYS
    assert tuple(driver.CASE_KEYS) == EXPECTED_CASE_KEYS
    assert catalog.COVERAGE_KEYS == EXPECTED_COVERAGE_KEYS
    assert tuple(case.board_order for case in catalog.CASES) == tuple(range(11))
    assert all(case.element_type == "f16" for case in catalog.CASES)
    assert all(
        case.disposition == catalog.CaseDisposition.PENDING_BOARD_CORRECTNESS
        for case in catalog.CASES
    )
    assert all(
        case.device_measurement
        == catalog.DeviceMeasurementState.BLOCKED_MISSING_DEVICE_PHASE_BASIS
        for case in catalog.CASES
    )
    assert not any(
        case.proves_target_peer_graph
        or case.proves_physical_route
        or case.proves_device_contention_cost
        or case.proves_same_physical_buffer
        for case in catalog.CASES
    )

    for item in catalog.COVERAGE_ITEMS:
        for relative in item.evidence_refs:
            assert (repo / relative).is_file(), (
                f"{item.key}: missing evidence reference {relative}"
            )
    raw_names: set[str] = set()
    for semantic in ("broadcast", "scatter", "shuffle"):
        item = catalog.COVERAGE_BY_KEY[
            f"dte-native-{semantic}-fanout-layout-matrix"
        ]
        assert (
            item.disposition
            == catalog.CoverageDisposition.PENDING_BOARD_EXECUTION
        )
        assert (
            item.execution_gate
            == catalog.ExecutionGate.QUALIFIED_RAW_DTE_CORRECTNESS
        )
        assert (
            catalog.PromotionGate.TYPED_CONCURRENT_MULTI_ENDPOINT_GRAPH
            in item.promotion_gates
        )
        semantic_cases = tuple(
            case
            for case in catalog.raw_dte.RAW_MULTIDEST_CASES
            if case.semantic == semantic
        )
        assert set(item.raw_case_names) == {
            case.name for case in semantic_cases
        }
        assert {
            (case.fanout, case.target_layout) for case in semantic_cases
        } == {
            (fanout, layout)
            for fanout in (2, 4, 8, 15)
            for layout in ("adjacent", "interleaved")
        }
        raw_names.update(item.raw_case_names)
    assert raw_names == {
        case.name for case in catalog.raw_dte.RAW_MULTIDEST_CASES
    }
    fanin4 = catalog.COVERAGE_BY_KEY["dte-four-source-fanin"]
    assert (
        fanin4.disposition
        == catalog.CoverageDisposition.PENDING_BOARD_EXECUTION
    )
    assert fanin4.raw_case_names == (
        "dte-four-source-fanin-correctness",
    )
    for fanin in ("eight", "fifteen"):
        item = catalog.COVERAGE_BY_KEY[f"dte-{fanin}-source-fanin"]
        assert (
            item.disposition
            == catalog.CoverageDisposition.BLOCKED_FAIL_CLOSED
        )
        assert catalog.PromotionGate.RECEIVER_FSM_CAPACITY in (
            item.promotion_gates
        )
    assert (
        catalog.COVERAGE_BY_KEY[
            "dte-fanout-fanin-two-payload-sweep"
        ].disposition
        == catalog.CoverageDisposition.EXISTING_BOARD_EVIDENCE
    )
    assert (
        catalog.COVERAGE_BY_KEY["tmnoc-per-link-counter"].disposition
        == catalog.CoverageDisposition.EXISTING_STATIC_NEGATIVE
    )
    assert (
        catalog.PromotionGate.FINAL_BUFFER_ALIAS_EVIDENCE
        in catalog.COVERAGE_BY_KEY[
            "collective-permute-same-physical-buffer-reuse"
        ].promotion_gates
    )
    assert all(
        catalog.row_major_manhattan_distance(source, target) == 1
        for source, target in catalog.FULL_CYCLE_FORWARD
    )
    assert set(catalog.FULL_CYCLE_REVERSE) == {
        (target, source) for source, target in catalog.FULL_CYCLE_FORWARD
    }
    assert all(
        catalog.row_major_manhattan_distance(source, target) == 4
        for source, target in catalog.OPPOSITE_PAIRS
    )
    assert all(
        catalog.row_major_manhattan_distance(source, target) == 1
        for source, target in catalog.DISJOINT_ADJACENT_PAIRS
    )


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    catalog.validate_catalog()
    assert_coverage(repo)
    assert_source_contracts()
    assert_intended_demand_summaries()
    assert_all_to_all_oracle()
    assert_permute_oracles()


if __name__ == "__main__":
    main()
