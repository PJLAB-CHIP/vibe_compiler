#!/usr/bin/env python3
"""Validate collective characterization catalog and execution inventories."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import numpy as np

import wafer_collective_hardware_characterization_catalog as catalog


EXPECTED_CASE_KEYS = (
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


def _load_python_asset(
    repo: pathlib.Path, relative: str, module_name: str
) -> object:
    spec = importlib.util.spec_from_file_location(module_name, repo / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _cmake_set(cmake: str, variable: str) -> tuple[str, ...]:
    match = re.search(
        rf"set\({re.escape(variable)}\s+(?P<body>.*?)\n\s*\)",
        cmake,
        re.DOTALL,
    )
    assert match is not None, f"missing CMake registry {variable}"
    return tuple(match["body"].split())


def _assert_no_short_period(values: np.ndarray) -> None:
    for shift in (1, 32, 256):
        if shift < values.size:
            assert not np.array_equal(values, np.roll(values, shift))


def _assert_reduction_mutation_adequacy(
    expected: np.ndarray, source_contributions: list[np.ndarray]
) -> None:
    expected_bytes = expected.reshape(-1).view(np.uint8)
    expected_u16 = expected_bytes.astype(np.uint16)
    contributions = [
        contribution.reshape(-1).view(np.uint8).astype(np.uint16)
        for contribution in source_contributions
    ]
    assert len(contributions) == 16
    assert all(contribution.shape == expected_u16.shape for contribution in contributions)
    assert len({contribution.tobytes() for contribution in contributions}) == 16

    for source_index, source in enumerate(contributions):
        _assert_no_short_period(source)
        missing_source = ((expected_u16 - source) & 0xFF).astype(np.uint8)
        assert not np.array_equal(missing_source, expected_bytes)
        for replacement_index, replacement in enumerate(contributions):
            if replacement_index == source_index:
                continue
            duplicate_source = (
                (expected_u16 - source + replacement) & 0xFF
            ).astype(np.uint8)
            assert not np.array_equal(duplicate_source, expected_bytes)


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    catalog.validate_catalog()

    assert catalog.CASE_KEYS == EXPECTED_CASE_KEYS
    assert tuple(catalog.CASES_BY_KEY) == EXPECTED_CASE_KEYS
    assert tuple(case.board_order for case in catalog.CASES) == tuple(range(9))
    assert {case.payload_bytes for case in catalog.CASES} == {256, 4096, 65536}
    assert {case.rank_count for case in catalog.CASES} == {16}
    assert {case.element_type for case in catalog.CASES} == {"i8"}
    assert {
        case.disposition for case in catalog.CASES
    } == {
        catalog.CharacterizationDisposition.PENDING_BOARD_CHARACTERIZATION
    }
    assert {
        case.numeric_oracle for case in catalog.CASES
    } == {catalog.NumericOracleKind.FULL_OUTPUT_EXACT}
    assert all(case.same_source_required for case in catalog.CASES)
    assert not any(
        case.performance_is_promotion_evidence for case in catalog.CASES
    )
    assert all(
        case.left_expected_phases
        and case.right_expected_phases
        and case.left_expected_phases.isdisjoint(case.right_expected_phases)
        for case in catalog.CASES
    )

    by_kind = {
        kind: tuple(
            case for case in catalog.CASES if case.collective_kind == kind
        )
        for kind in catalog.CollectiveKind
    }
    assert all(len(cases) == 3 for cases in by_kind.values())
    assert {
        (
            cases[0].left_alternative.value,
            cases[0].right_alternative.value,
        )
        for cases in by_kind.values()
    } == {
        ("all-gather-direct", "all-gather-ring"),
        ("reduce-scatter-direct", "reduce-scatter-ring"),
        ("all-reduce-ring", "all-reduce-tree"),
    }
    assert catalog.ALTERNATIVE_PHASES[
        catalog.CollectiveAlternative.ALL_REDUCE_TREE
    ] == {"all_reduce_tree_reduce", "all_reduce_tree_broadcast"}

    driver_path = (
        repo / "test/Board/wafer_board_collective_characterization_test.py"
    )
    assert driver_path.is_file(), "missing executable characterization driver"
    driver = _load_python_asset(
        repo,
        "test/Board/wafer_board_collective_characterization_test.py",
        "wafer_board_collective_characterization_test_catalog_check",
    )
    assert tuple(driver.CASE_KEYS) == EXPECTED_CASE_KEYS
    assert driver.CASES_BY_KEY is catalog.CASES_BY_KEY
    synthetic_rows = {
        rank: {
            "messages": [
                {
                    "direction": "recv",
                    "peer": (rank - 1) % 16,
                    "communication_id": 91,
                    "phase": "all_gather_ring",
                    "round": 0,
                    "payload_slice": (rank - 1) % 16,
                    "issue_bytes": 16,
                    "constant_loop_multiplicity": 1,
                    "executed_bytes": 16,
                },
                {
                    "direction": "send",
                    "peer": (rank + 1) % 16,
                    "communication_id": 91,
                    "phase": "all_gather_ring",
                    "round": 0,
                    "payload_slice": rank,
                    "issue_bytes": 16,
                    "constant_loop_multiplicity": 1,
                    "executed_bytes": 16,
                },
            ]
        }
        for rank in range(16)
    }
    driver.require_cross_rank_message_matching(synthetic_rows, "synthetic-ring")
    driver.require_single_cycle(
        {rank: (rank + 1) % 16 for rank in range(16)}, "synthetic-ring"
    )
    synthetic_rows[7]["messages"][0]["round"] = 1
    try:
        driver.require_cross_rank_message_matching(
            synthetic_rows, "tampered-ring"
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("message tuple tampering was not rejected")
    try:
        driver.require_single_cycle(
            {rank: (rank + 1) % 8 + (rank // 8) * 8 for rank in range(16)},
            "split-ring",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("two disjoint cycles were not rejected")

    wrong_progression_rows = {rank: {"messages": []} for rank in range(16)}
    for rank in range(16):
        for round_ in range(15):
            wrong_progression_rows[rank]["messages"].extend(
                [
                    {
                        "direction": "recv",
                        "peer": (rank - 1) % 16,
                        "communication_id": 92,
                        "phase": "all_gather_ring",
                        "round": round_,
                        "payload_slice": (rank - round_ - 2) % 16,
                        "issue_bytes": 16,
                        "constant_loop_multiplicity": 1,
                        "executed_bytes": 16,
                    },
                    {
                        "direction": "send",
                        "peer": (rank + 1) % 16,
                        "communication_id": 92,
                        "phase": "all_gather_ring",
                        "round": round_,
                        "payload_slice": (rank - round_ - 1) % 16,
                        "issue_bytes": 16,
                        "constant_loop_multiplicity": 1,
                        "executed_bytes": 16,
                    },
                ]
            )
    driver.require_cross_rank_message_matching(
        wrong_progression_rows, "matched-but-wrong-ring"
    )
    try:
        driver.require_ring(
            wrong_progression_rows,
            catalog.CollectiveAlternative.ALL_GATHER_RING.value,
            15,
            16,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong ring round/slice progression was accepted")

    def synthetic_tree_rows(
        use_depth: bool, use_child_slice: bool = True
    ) -> dict[int, dict[str, object]]:
        rows: dict[int, dict[str, object]] = {
            rank: {"messages": []} for rank in range(16)
        }

        def append_message(
            rank: int,
            direction: str,
            peer: int,
            phase: str,
            round_: int,
            payload_slice: int,
        ) -> None:
            rows[rank]["messages"].append(
                {
                    "direction": direction,
                    "peer": peer,
                    "communication_id": 93,
                    "phase": phase,
                    "round": round_,
                    "payload_slice": payload_slice,
                    "issue_bytes": 16,
                    "constant_loop_multiplicity": 1,
                    "executed_bytes": 16,
                }
            )

        for child in range(15):
            parent = child + 1
            round_ = 15 - child if use_depth else 0
            payload_slice = child if use_child_slice else (child + 1) % 16
            append_message(
                child,
                "send",
                parent,
                "all_reduce_tree_reduce",
                round_,
                payload_slice,
            )
            append_message(
                parent,
                "recv",
                child,
                "all_reduce_tree_reduce",
                round_,
                payload_slice,
            )
            append_message(
                parent,
                "send",
                child,
                "all_reduce_tree_broadcast",
                round_,
                payload_slice,
            )
            append_message(
                child,
                "recv",
                parent,
                "all_reduce_tree_broadcast",
                round_,
                payload_slice,
            )
        return rows

    exact_tree_rows = synthetic_tree_rows(True)
    driver.require_cross_rank_message_matching(exact_tree_rows, "exact-tree")
    driver.require_ordered_tree(exact_tree_rows, "exact-tree", 16)
    wrong_tree_rows = synthetic_tree_rows(False)
    driver.require_cross_rank_message_matching(wrong_tree_rows, "wrong-tree")
    try:
        driver.require_ordered_tree(wrong_tree_rows, "wrong-tree", 16)
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong tree depth/round mapping was accepted")
    wrong_slice_tree_rows = synthetic_tree_rows(True, False)
    driver.require_cross_rank_message_matching(
        wrong_slice_tree_rows, "wrong-tree-slice"
    )
    try:
        driver.require_ordered_tree(
            wrong_slice_tree_rows, "wrong-tree-slice", 16
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong tree child/slice mapping was accepted")

    carrier = _load_python_asset(
        repo,
        "test/Board/wafer_collective_characterization_spmd_carrier.py",
        "wafer_collective_characterization_spmd_carrier_catalog_check",
    )
    all_gather_boundary = carrier.collective_boundary(
        '"stablehlo.all_gather"',
        {"shape": [16], "dtype": "i8"},
        {"shape": [256], "dtype": "i8"},
        16,
    )
    assert all_gather_boundary["inputs"][0]["distribution"] == "partitioned"
    assert all_gather_boundary["inputs"][0]["global_shape"] == [256]
    assert all_gather_boundary["inputs"][0]["local_shape"] == [16]
    assert all_gather_boundary["inputs"][0]["ranks"][15] == {
        "rank": 15,
        "replica_id": 0,
        "offsets": [240],
        "sizes": [16],
        "strides": [1],
    }
    assert all_gather_boundary["outputs"][0]["distribution"] == "replicated"
    assert {
        row["replica_id"]
        for row in all_gather_boundary["outputs"][0]["ranks"]
    } == set(range(16))

    reduce_scatter_boundary = carrier.collective_boundary(
        '"stablehlo.reduce_scatter"',
        {"shape": [256], "dtype": "i8"},
        {"shape": [16], "dtype": "i8"},
        16,
    )
    assert reduce_scatter_boundary["inputs"][0]["distribution"] == "partitioned"
    assert reduce_scatter_boundary["inputs"][0]["global_shape"] == [4096]
    assert reduce_scatter_boundary["inputs"][0]["local_shape"] == [256]
    assert reduce_scatter_boundary["outputs"][0]["distribution"] == "partitioned"
    assert reduce_scatter_boundary["outputs"][0]["global_shape"] == [256]
    assert reduce_scatter_boundary["outputs"][0]["local_shape"] == [16]
    assert reduce_scatter_boundary["outputs"][0]["ranks"][15] == {
        "rank": 15,
        "replica_id": 0,
        "offsets": [240],
        "sizes": [16],
        "strides": [1],
    }

    for contract in catalog.CASES:
        inputs, outputs = driver.payloads(driver.runtime_case(contract))
        local_inputs = [rank_inputs[0] for rank_inputs in inputs]
        local_outputs = [rank_outputs[0] for rank_outputs in outputs]
        assert len({values.tobytes() for values in local_inputs}) == 16
        if contract.collective_kind == catalog.CollectiveKind.ALL_GATHER:
            assert len({values.tobytes() for values in local_outputs}) == 1
            expected = local_outputs[0]
            _assert_no_short_period(expected)
            chunks = np.split(expected, 16)
            assert len({chunk.tobytes() for chunk in chunks}) == 16
            for chunk in chunks:
                _assert_no_short_period(chunk)
            for source in range(16):
                destination = (source + 1) % 16
                swapped_chunks = list(chunks)
                swapped_chunks[source], swapped_chunks[destination] = (
                    swapped_chunks[destination],
                    swapped_chunks[source],
                )
                assert not np.array_equal(
                    np.concatenate(swapped_chunks), expected
                )
            continue

        reduced = (
            np.concatenate(local_outputs)
            if contract.collective_kind
            == catalog.CollectiveKind.REDUCE_SCATTER
            else local_outputs[0]
        )
        _assert_no_short_period(reduced)
        if (
            contract.collective_kind
            == catalog.CollectiveKind.REDUCE_SCATTER
        ):
            assert len({values.tobytes() for values in local_outputs}) == 16
            chunk_bytes = local_outputs[0].size
            for destination, output in enumerate(local_outputs):
                _assert_no_short_period(output)
                begin = destination * chunk_bytes
                end = begin + chunk_bytes
                _assert_reduction_mutation_adequacy(
                    output,
                    [
                        source.reshape(-1)[begin:end]
                        for source in local_inputs
                    ],
                )
        else:
            assert len({values.tobytes() for values in local_outputs}) == 1
            _assert_reduction_mutation_adequacy(reduced, local_inputs)

    runner = _load_python_asset(
        repo,
        "tools/run_hardware_calibration.py",
        "run_hardware_calibration_collective_catalog_check",
    )
    assert (
        tuple(runner.COLLECTIVE_CHARACTERIZATION_CASES)
        == EXPECTED_CASE_KEYS
    )
    assert runner.SELECTABLE_BATCHES["collective-characterization"] == tuple(
        f"collective-characterization-{key}" for key in EXPECTED_CASE_KEYS
    )

    cmake = (repo / "test/CMakeLists.txt").read_text()
    assert _cmake_set(
        cmake, "_wafer_collective_characterization_cases"
    ) == EXPECTED_CASE_KEYS
    assert (
        "wafer-runtime-collective-characterization-"
        "${_wafer_collective_characterization_case}-no-card"
    ) in cmake
    assert (
        "wafer-board-collective-characterization-"
        "${_wafer_collective_characterization_case}"
    ) in cmake
    assert cmake.count(
        "foreach(_wafer_collective_characterization_case IN LISTS"
    ) == 2
    board_properties = re.search(
        r"set_tests_properties\(\s*"
        r"wafer-board-collective-characterization-"
        r"\$\{_wafer_collective_characterization_case\}\s*"
        r"PROPERTIES(?P<body>.*?)\n\s*\)",
        cmake,
        re.DOTALL,
    )
    assert board_properties is not None
    assert "board;hardware;collective-characterization" in board_properties[
        "body"
    ]
    assert ";pending" in board_properties["body"]
    assert (
        'RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"'
        in board_properties["body"]
    )
    assert "SKIP_RETURN_CODE 77" in board_properties["body"]
    assert "TIMEOUT 3600" in board_properties["body"]

    optimizer_catalog = _load_python_asset(
        repo,
        "test/Board/wafer_compiler_optimization_campaign_catalog.py",
        "wafer_compiler_optimization_catalog_collective_check",
    )
    direct_transport = optimizer_catalog.CASES_BY_KEY["direct-all-reduce"]
    assert direct_transport.family == "direct-dte-transport-evidence"
    assert (
        direct_transport.pair_requirement
        == optimizer_catalog.PackagePairRequirement.EXISTING_SINGLE_PACKAGE_BASELINE
    )
    assert set(EXPECTED_CASE_KEYS).isdisjoint(optimizer_catalog.CASES_BY_KEY)


if __name__ == "__main__":
    main()
