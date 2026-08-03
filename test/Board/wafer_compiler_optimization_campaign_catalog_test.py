#!/usr/bin/env python3
"""Validate production-compiler optimization campaign coverage and oracles."""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import re
import sys
import tempfile

import torch

import wafer_board_compiler_optimization_campaign_test as driver
import wafer_compiler_optimization_campaign_catalog as catalog


EXPECTED_CASE_KEYS = {
    "reciprocal-implementation",
    "f16-common-factor",
    "resident-fanout-share",
    "consumer-local-recompute",
    "long-steady-elementwise-add",
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
    "static-fixed-slot-overlap-selection",
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
        + "output_capture: resource=7 bytes=32768 path=/tmp/rank-one.raw\n"
        + "terminal_completion: 3 kind=entry_return\n",
        rank_one,
        {7},
        {(3, 0)},
    )
    try:
        driver.verify_board_output(
            lifecycle
            + "output_capture: resource=7 bytes=32768 path=/tmp/a.raw\n"
            + "output_capture: resource=7 bytes=32768 path=/tmp/b.raw\n"
            + "terminal_completion: 3 kind=entry_return\n",
            rank_one,
            {7},
            {(3, 0)},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate board output comparison was accepted")

    relaxed = driver.CASES["f16-common-factor"]
    driver.verify_board_output(
        lifecycle
        + "output_capture: resource=8 bytes=32768 path=/tmp/relaxed.raw\n"
        + "terminal_completion: 4 kind=entry_return\n",
        relaxed,
        {8},
        {(4, 0)},
    )
    try:
        driver.verify_board_output(
            lifecycle
            + "output_capture: resource=9 bytes=32768 path=/tmp/wrong.raw\n"
            + "terminal_completion: 4 kind=entry_return\n",
            relaxed,
            {8},
            {(4, 0)},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "board output accepted a capture for the wrong resource"
        )

    collective = driver.CASES["tree-all-reduce"]
    output_ids = set(range(100, 116))
    completions = {(rank + 200, rank) for rank in range(16)}
    collective_stdout = (
        lifecycle
        + "launch_pattern: cluster-x16\n"
        + "logical_tile_domain: 0..15\n"
        + "".join(
            f"output_capture: resource={resource} bytes=4096 "
            f"path=/tmp/rank-{resource}.raw\n"
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


def validate_paired_payload_preflight() -> None:
    lightweight_cases = {
        key: case
        for key, case in driver.CASES.items()
        if key
        not in {
            "long-steady-elementwise-add",
            "noc-resident-large-gemm",
            "noc-resident-m-sharded-gemm",
        }
    }
    for case in lightweight_cases.values():
        driver.validate_paired_payloads(case, case.payload_factory())

    case = driver.CASES["f16-common-factor"]
    payloads = case.payload_factory()
    assert len(payloads.inputs) == 1
    assert len(payloads.baseline_outputs) == 1
    assert len(payloads.winner_outputs) == 1
    a, b, c = payloads.inputs[0]
    baseline = payloads.baseline_outputs[0][0]
    winner = payloads.winner_outputs[0][0]
    source = (a * b) + (a * c)
    torch.testing.assert_close(baseline, source, rtol=0, atol=0)
    torch.testing.assert_close(winner, source, rtol=0, atol=0)
    bindings = {
        variant: {
            (0, "user_input", 0): base,
            (0, "user_input", 1): base + 1,
            (0, "user_input", 2): base + 2,
            (0, "output", 0): base + 3,
        }
        for variant, base in (("baseline", 10), ("winner", 20))
    }
    with tempfile.TemporaryDirectory() as directory:
        arguments = driver.write_payloads(
            pathlib.Path(directory), case, bindings, payloads
        )
    for variant in ("baseline", "winner"):
        assert "--output" in arguments[variant]
        assert "--expected-f16-relaxed" not in arguments[variant]
        assert "--expected" not in arguments[variant]

    mismatching_baseline = [[baseline.clone()]]
    mismatching_winner = [[winner.clone()]]
    mismatching_winner[0][0][2] = (
        mismatching_baseline[0][0][2]
        + torch.tensor(1.0, dtype=torch.float16)
    )
    try:
        driver.validate_paired_payloads(
            case,
            driver.PairedPayloads(
                payloads.inputs,
                mismatching_baseline,
                mismatching_winner,
            ),
        )
    except RuntimeError as error:
        assert "exceed the numeric policy" in str(error)
        assert "element 2" in str(error)
    else:
        raise AssertionError(
            "paired payload preflight accepted an out-of-policy finite mismatch"
        )

    nonfinite_winner = [[winner.clone()]]
    nonfinite_winner[0][0][3] = torch.inf
    try:
        driver.validate_paired_payloads(
            case,
            driver.PairedPayloads(
                payloads.inputs,
                [[baseline.clone()]],
                nonfinite_winner,
            ),
        )
    except RuntimeError as error:
        assert "NaN or infinity" in str(error)
        assert "element 3" in str(error)
    else:
        raise AssertionError(
            "paired payload preflight accepted a nonfinite oracle"
        )


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    validate_board_output_parser()
    validate_paired_payload_preflight()
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
    assert paired_case_keys.issubset(driver.CASES), (
        "new paired campaign entries must have a concrete executable driver case"
    )
    for key, executable in driver.CASES.items():
        dtypes = {
            spec.mlir_dtype
            for spec in (*executable.inputs, *executable.outputs)
        }
        assert dtypes == {"f16"}, (
            f"{key}: generic board workloads must use f16"
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
    assert catalog.PENDING_CONFIGURED_BOARD_AXES
    disposition_partitions = (
        {axis.key for axis in catalog.BOARD_AXES},
        {axis.key for axis in catalog.HOST_ONLY_AXES},
        {axis.key for axis in catalog.PENDING_CONFIGURED_BOARD_AXES},
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
        "f16-common-factor",
        "resident-fanout-share",
        "consumer-local-recompute",
        "long-steady-elementwise-add",
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

    for axis in catalog.PENDING_CONFIGURED_BOARD_AXES:
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
        "f16-common-factor",
        "reciprocal-implementation",
    }
    assert shared_families["resource-aware-order"] == {
        "ready-order-movement-first",
    }
    assert shared_families["static-fixed-slot-overlap"] == {
        "long-steady-elementwise-add",
    }
    assert shared_families["collective-algorithms"] == {"tree-all-reduce"}
    assert shared_families["direct-dte-transport-evidence"] == {
        "direct-all-reduce",
    }
    direct_transport = catalog.CASES_BY_KEY["direct-all-reduce"]
    assert (
        direct_transport.pair_requirement
        == catalog.PackagePairRequirement.EXISTING_SINGLE_PACKAGE_BASELINE
    )
    assert not direct_transport.requires_pair
    assert any(
        "transport evidence only" in statement
        for statement in direct_transport.pair_contract
    )

    assert any(
        len(axes) > 1 for axes in catalog.AXES_BY_CASE.values()
    ), "catalog accidentally regressed to one board case per pass/axis"
    assert {
        axis.evidence_key
        for axis in catalog.PENDING_CONFIGURED_BOARD_AXES
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
    assert tuple(
        case_key for case_key in driver.CASES if case_key in paired_case_keys
    ) == paired_by_board_order
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
    assert tuple(
        case_key
        for case_key in collective_cases
        if case_key in paired_case_keys
    ) == paired_by_board_order[-1:]
    assert cmake.count(
        "foreach(_wafer_compiler_optimization_case IN LISTS"
    ) == 2
    assert (
        "wafer-runtime-compiler-optimization-"
        "${_wafer_compiler_optimization_case}-no-card"
    ) not in cmake
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
        f"pending_configured_board="
        f"{len(catalog.PENDING_CONFIGURED_BOARD_AXES)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
