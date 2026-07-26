#!/usr/bin/env python3
"""Validate production-compiler optimization campaign coverage and oracles."""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import re
import sys

import wafer_board_compiler_optimization_campaign_test as driver
import wafer_compiler_optimization_campaign_catalog as catalog


EXPECTED_CASE_KEYS = {
    "reciprocal-implementation",
    "modular-common-factor",
    "resident-fanout-share",
    "consumer-local-recompute",
    "ready-order-movement-first",
    "gemm-aligned-physical-route",
    "gemm-tail-physical-route",
    "direct-all-reduce",
    "tree-all-reduce",
}

EXPECTED_AXIS_KEYS = {
    "target-implementation-selection",
    "dependent-tiling-and-tail-coverage",
    "producer-fusion-and-relation-propagation",
    "physical-encoding-view-materialization",
    "transfer-route-storage-realization",
    "fixed-cx-ncx-gemm-absorption",
    "physical-version-reuse",
    "movement-resident-cut-elimination",
    "whole-tensor-share-winner",
    "consumer-local-recompute-winner",
    "relaxed-f16-bf16-algebra",
    "integer-modular-reassociation",
    "integer-modular-reduction-tree",
    "integer-modular-distribution",
    "integer-modular-common-factor",
    "static-loop-invariant-hoist",
    "static-buffering-ready-order",
    "collective-direct",
    "collective-ring-all-gather",
    "collective-ordered-tree-all-reduce",
    "stablehlo-normalization-and-canonicalization",
    "function-boundary-bufferization-alias",
    "index-relation-and-traversal-legality",
    "candidate-rewrite-effect-and-numeric-negatives",
    "ready-order-hazard-negatives",
    "resource-aware-neighbor-generation",
    "bounded-joint-search-and-baseline-fallback",
    "whole-variant-pareto-and-atomic-commit",
    "spm-lifetime-placement-and-packing",
    "ddr-lifetime-placement-and-packing",
    "typed-instruction-lowering-legality",
    "software-pipeline-queue-occupancy",
    "software-pipeline-spm-bank-cost",
    "software-pipeline-ddr-bank-cost",
    "software-pipeline-cross-worker-completion",
}


def load_python_asset(
    repo: pathlib.Path, relative: str, module_name: str
) -> object:
    spec = importlib.util.spec_from_file_location(module_name, repo / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cmake_set(cmake: str, variable: str) -> tuple[str, ...]:
    match = re.search(
        rf"set\({re.escape(variable)}\s+(?P<body>.*?)\n\s*\)",
        cmake,
        re.DOTALL,
    )
    assert match is not None, f"missing CMake registry {variable}"
    return tuple(match["body"].split())


def validate_board_output_parser() -> None:
    lifecycle = (
        "board_stage: launch\n"
        "board_stage: completion\n"
        "board_stage: device-to-host\n"
        "board_stage: cleanup\n"
        "board_execution: true\n"
    )
    rank_one = driver.CASES["reciprocal-implementation"]
    driver.verify_board_output(
        lifecycle
        + "output_compare: resource=7 bytes=32768 exact=true\n"
        + "terminal_completion: 3 kind=entry_return\n",
        rank_one,
        {7},
        {(3, 0)},
    )
    try:
        driver.verify_board_output(
            lifecycle
            + "output_compare: resource=7 bytes=32768 exact=true\n"
            + "output_compare: resource=7 bytes=32768 exact=true\n"
            + "terminal_completion: 3 kind=entry_return\n",
            rank_one,
            {7},
            {(3, 0)},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate board output comparison was accepted")

    collective = driver.CASES["tree-all-reduce"]
    output_ids = set(range(100, 116))
    completions = {(rank + 200, rank) for rank in range(16)}
    collective_stdout = (
        lifecycle
        + "launch_pattern: cluster-prepare-main-x16\n"
        + "logical_tile_domain: 0..15\n"
        + "".join(
            f"output_compare: resource={resource} bytes=4096 exact=true\n"
            for resource in sorted(output_ids)
        )
        + "".join(
            "terminal_completion: "
            f"{completion} kind=entry_return rank={rank}\n"
            for completion, rank in sorted(completions, key=lambda item: item[1])
        )
    )
    driver.verify_board_output(
        collective_stdout, collective, output_ids, completions
    )


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    validate_board_output_parser()
    calibration_runner = load_python_asset(
        repo,
        "tools/run_hardware_calibration.py",
        "wafer_hardware_calibration_catalog_contract",
    )
    direct_driver = load_python_asset(
        repo,
        "test/Board/wafer_board_direct_dte_collective_test.py",
        "wafer_direct_dte_catalog_contract",
    )

    assert set(catalog.CASES_BY_KEY) == EXPECTED_CASE_KEYS
    assert len(catalog.CASES_BY_KEY) == len(catalog.CAMPAIGN_CASES)
    paired_case_keys = {
        case.key for case in catalog.CAMPAIGN_CASES if case.requires_pair
    }
    assert paired_case_keys == set(driver.CASES), (
        "new paired campaign entries must have a concrete executable driver case"
    )
    assert set(catalog.AXES_BY_KEY) == EXPECTED_AXIS_KEYS
    assert len(catalog.AXES_BY_KEY) == len(catalog.OPTIMIZATION_AXES)
    assert set(catalog.PRODUCTION_OWNER_BY_AXIS) == EXPECTED_AXIS_KEYS

    for anchor in catalog.PRODUCTION_PIPELINE_ANCHORS:
        asset = repo / anchor.asset
        assert asset.is_file(), f"missing production pipeline asset {anchor.asset}"
        source = asset.read_text()
        for marker in anchor.markers:
            assert marker in source, (
                f"production pipeline marker {marker!r} is absent from {anchor.asset}"
            )
    for axis in catalog.OPTIMIZATION_AXES:
        owner = axis.production_owner
        assert owner == catalog.PRODUCTION_OWNER_BY_AXIS[axis.key]
        asset = repo / owner.asset
        assert asset.is_file(), f"{axis.key}: missing owner {owner.asset}"
        source = asset.read_text()
        assert owner.markers, f"{axis.key}: owner has no source marker"
        for marker in owner.markers:
            assert marker in source, (
                f"{axis.key}: owner marker {marker!r} is absent from {owner.asset}"
            )
    claimed_execution_capabilities = set().union(
        *(
            case.oracle.executable_capabilities
            - catalog.STRUCTURAL_EVIDENCE_CAPABILITIES
            for case in catalog.CAMPAIGN_CASES
        )
    )
    assert claimed_execution_capabilities == set(
        catalog.EXECUTION_CAPABILITY_ANCHORS
    )
    for capability, anchor in catalog.EXECUTION_CAPABILITY_ANCHORS.items():
        asset = repo / anchor.asset
        assert asset.is_file(), f"{capability.value}: missing {anchor.asset}"
        source = asset.read_text()
        for marker in anchor.markers:
            assert marker in source, (
                f"{capability.value}: marker {marker!r} is absent from "
                f"{anchor.asset}"
            )

    assert {
        axis.disposition for axis in catalog.OPTIMIZATION_AXES
    } == set(catalog.AxisDisposition)
    assert catalog.BOARD_AXES
    assert catalog.HOST_ONLY_AXES
    assert catalog.FUTURE_SOFTWARE_PIPELINE_AXES
    disposition_partitions = (
        {axis.key for axis in catalog.BOARD_AXES},
        {axis.key for axis in catalog.HOST_ONLY_AXES},
        {axis.key for axis in catalog.FUTURE_SOFTWARE_PIPELINE_AXES},
    )
    assert set().union(*disposition_partitions) == EXPECTED_AXIS_KEYS
    assert all(
        not lhs & rhs
        for index, lhs in enumerate(disposition_partitions)
        for rhs in disposition_partitions[index + 1 :]
    ), "every optimization axis must have exactly one disposition"

    board_orders = [case.board_order for case in catalog.CAMPAIGN_CASES]
    assert len(set(board_orders)) == len(board_orders)
    assert all(case.priority == "P0" for case in catalog.CAMPAIGN_CASES)
    assert all(case.rank_count in {1, 16} for case in catalog.CAMPAIGN_CASES)
    assert all(case.completion_contract for case in catalog.CAMPAIGN_CASES)
    assert tuple(
        case.key
        for case in sorted(
            catalog.CAMPAIGN_CASES, key=lambda entry: entry.board_order
        )
    ) == (
        "reciprocal-implementation",
        "modular-common-factor",
        "resident-fanout-share",
        "consumer-local-recompute",
        "ready-order-movement-first",
        "gemm-aligned-physical-route",
        "gemm-tail-physical-route",
        "direct-all-reduce",
        "tree-all-reduce",
    )

    for case in catalog.CAMPAIGN_CASES:
        assert case.oracle.is_strong, f"{case.key}: weak board oracle"
        assert case.oracle.performance_is_promotion_evidence is False
        assert case.oracle.structural_checks
        assert case.oracle.numeric_checks
        assert case.oracle.performance_checks
        assert case.oracle.executable_capabilities
        assert case.key in catalog.AXES_BY_CASE
        assert catalog.AXES_BY_CASE[case.key]
        assert (repo / case.execution_asset).is_file(), (
            f"{case.key}: missing execution asset {case.execution_asset}"
        )
        assert {
            axis.disposition for axis in catalog.AXES_BY_CASE[case.key]
        } == {case.disposition}
        if case.requires_pair:
            assert case.disposition == (
                catalog.AxisDisposition.NEW_PAIRED_PACKAGE_BOARD_FAMILY
            )
            assert case.package_roles == ("control", "optimized")
            assert len(case.pair_contract) >= 4
            assert (
                case.oracle.performance_kind
                == catalog.PerformanceOracleKind.BALANCED_HOST_PROCESS_OBSERVATION
            )
            assert case.execution_asset == catalog.PAIRED_CAMPAIGN_DRIVER
            assert driver.CASES[case.key].rank_count == case.rank_count
            binding = case.driver_binding
            assert binding is not None
            executable = driver.CASES[case.key]
            assert executable.structural_oracle.__name__ == (
                binding.structural_oracle_symbol
            )
            structural_source = inspect.getsource(executable.structural_oracle)
            claimed_structural = (
                case.oracle.executable_capabilities
                & catalog.STRUCTURAL_EVIDENCE_CAPABILITIES
            )
            bound_structural = {
                capability
                for capability, _ in binding.structural_capability_markers
            }
            assert claimed_structural == bound_structural, (
                f"{case.key}: typed structural capabilities are not bound "
                "one-for-one to the executable oracle"
            )
            for capability, marker in binding.structural_capability_markers:
                assert marker in structural_source, (
                    f"{case.key}: {capability.value} marker {marker!r} "
                    "is absent from the driver oracle"
                )
            payload_source = inspect.getsource(executable.payload_factory)
            assert binding.payload_factory_marker in payload_source, (
                f"{case.key}: payload binding is stale"
            )
        else:
            assert (
                case.disposition
                == catalog.AxisDisposition.EXISTING_BOARD_FAMILY
            )
            assert case.package_roles == ("selected",)
            assert case.existing_assets
            assert (
                case.oracle.performance_kind
                == catalog.PerformanceOracleKind.REPEATED_LIFECYCLE_OBSERVATION
            )
            binding = case.driver_binding
            assert binding is not None
            structural = getattr(
                direct_driver, binding.structural_oracle_symbol
            )
            payload = getattr(direct_driver, "write_raw_files")
            structural_source = inspect.getsource(structural)
            payload_source = inspect.getsource(payload)
            assert binding.payload_factory_marker in payload_source
            claimed_structural = (
                case.oracle.executable_capabilities
                & catalog.STRUCTURAL_EVIDENCE_CAPABILITIES
            )
            bound_structural = {
                capability
                for capability, _ in binding.structural_capability_markers
            }
            assert claimed_structural == bound_structural
            for capability, marker in binding.structural_capability_markers:
                assert marker in structural_source, (
                    f"{case.key}: {capability.value} marker {marker!r} "
                    "is absent from the existing execution asset"
                )
        if case.board_batch == catalog.BoardBatch.RANK_ONE_LOCAL:
            assert case.rank_count == 1
        elif (
            case.board_batch
            == catalog.BoardBatch.RANK_SIXTEEN_COMMUNICATION
        ):
            assert case.rank_count == 16
        else:
            raise AssertionError(f"{case.key}: unknown board batch")
        for asset in case.existing_assets:
            assert (repo / asset).is_file(), f"{case.key}: missing {asset}"

    mapped_case_keys = {
        axis.evidence_key for axis in catalog.BOARD_AXES
    }
    assert mapped_case_keys == EXPECTED_CASE_KEYS
    for axis in catalog.BOARD_AXES:
        assert axis.board_observables
        assert axis.evidence_key in catalog.CASES_BY_KEY
        case = catalog.CASES_BY_KEY[axis.evidence_key]
        assert case.oracle.is_strong, (
            f"{axis.key}: board mapping has no strong oracle"
        )
        assert case.disposition == axis.disposition
        assert not axis.host_assets

    for axis in catalog.HOST_ONLY_AXES:
        assert not axis.board_observables
        assert axis.host_assets
        assert axis.evidence_key not in catalog.CASES_BY_KEY
        for asset in axis.host_assets:
            assert (repo / asset).is_file(), f"{axis.key}: missing {asset}"

    for axis in catalog.FUTURE_SOFTWARE_PIPELINE_AXES:
        assert not axis.board_observables
        assert axis.host_assets == catalog.SOFTWARE_PIPELINE_PLAN
        assert axis.evidence_key not in catalog.CASES_BY_KEY
        for asset in axis.host_assets:
            assert (repo / asset).is_file(), f"{axis.key}: missing {asset}"

    shared_families = {
        family: {
            case.key
            for case in catalog.CAMPAIGN_CASES
            if case.family == family
        }
        for family in {case.family for case in catalog.CAMPAIGN_CASES}
    }
    assert shared_families["tile-layout-route"] == {
        "gemm-aligned-physical-route",
        "gemm-tail-physical-route",
    }
    assert shared_families["resident-dataflow"] == {
        "resident-fanout-share",
        "consumer-local-recompute",
    }
    assert shared_families["numeric-dag-and-implementation"] == {
        "modular-common-factor",
        "reciprocal-implementation",
    }
    assert shared_families["resource-aware-order"] == {
        "ready-order-movement-first",
    }
    assert shared_families["collective-algorithms"] == {
        "direct-all-reduce",
        "tree-all-reduce",
    }

    assert any(
        len(axes) > 1 for axes in catalog.AXES_BY_CASE.values()
    ), "catalog accidentally regressed to one board case per pass/axis"
    assert {
        axis.evidence_key
        for axis in catalog.FUTURE_SOFTWARE_PIPELINE_AXES
    } == {
        "queue-saturation-response",
        "spm-conflict-equivalence",
        "ddr-conflict-equivalence",
        "worker-wait-and-subset-join-exclusion",
    }

    paired_by_board_order = tuple(
        case.key
        for case in sorted(
            (
                case
                for case in catalog.CAMPAIGN_CASES
                if case.requires_pair
            ),
            key=lambda entry: entry.board_order,
        )
    )
    assert tuple(driver.CASES) == paired_by_board_order
    assert tuple(calibration_runner.COMPILER_OPTIMIZATION_PAIRED_CASES) == (
        paired_by_board_order
    ), "runner registry must be the board-order view of executable paired cases"

    cmake = (repo / "test/CMakeLists.txt").read_text()
    rank_one_cases = cmake_set(
        cmake, "_wafer_compiler_optimization_rank_one_cases"
    )
    collective_cases = cmake_set(
        cmake, "_wafer_compiler_optimization_collective_cases"
    )
    assert rank_one_cases == paired_by_board_order[:-1]
    assert collective_cases == paired_by_board_order[-1:]
    assert cmake.count(
        "foreach(_wafer_compiler_optimization_case IN LISTS"
    ) == 3
    assert (
        "wafer-runtime-compiler-optimization-"
        "${_wafer_compiler_optimization_case}-no-card"
    ) in cmake
    assert (
        "wafer-board-compiler-optimization-"
        "${_wafer_compiler_optimization_case}"
    ) in cmake
    for case_key in paired_by_board_order:
        assert f"wafer-board-compiler-optimization-{case_key}" not in cmake, (
            f"{case_key}: board registration bypasses the shared foreach registry"
        )

    print(
        "wafer_compiler_optimization_campaign_catalog_test: "
        f"axes={len(catalog.OPTIMIZATION_AXES)} "
        f"board_axes={len(catalog.BOARD_AXES)} "
        f"cases={len(catalog.CAMPAIGN_CASES)} "
        f"host_only={len(catalog.HOST_ONLY_AXES)} "
        f"future={len(catalog.FUTURE_SOFTWARE_PIPELINE_AXES)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
