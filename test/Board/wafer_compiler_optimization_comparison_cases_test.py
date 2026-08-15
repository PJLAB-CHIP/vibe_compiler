#!/usr/bin/env python3
"""Validate current optimizer-comparison source and oracle contracts."""

from __future__ import annotations

import json
import pathlib
import tempfile

import wafer_board_compiler_optimization_comparison_test as driver
import wafer_board_m_tiled_gemm_profile_test as m_tiled_profile
import wafer_compiler_optimization_comparison_cases as catalog


EXPECTED_CASE_KEYS = {
    "reciprocal-implementation",
    "f16-common-factor",
    "resident-fanout-share",
    "consumer-local-recompute",
    "long-steady-elementwise-add",
    "ready-order-movement-first",
    "gemm-aligned-physical-route",
    "gemm-tail-physical-route",
    "noc-resident-large-gemm",
    "noc-resident-m-tiled-gemm",
    "direct-dte-reduction",
    "tree-all-reduce",
}


def validate_driver_cases() -> None:
    required_driver_cases = EXPECTED_CASE_KEYS - {"direct-dte-reduction"}
    assert required_driver_cases.issubset(driver.CASES)
    for key in required_driver_cases:
        case = driver.CASES[key]
        assert {spec.mlir_dtype for spec in (*case.inputs, *case.outputs)} == {
            "f16"
        }
        driver.validate_source(case)

    for key in (
        "reciprocal-implementation",
        "f16-common-factor",
        "resident-fanout-share",
        "ready-order-movement-first",
        "gemm-aligned-physical-route",
        "gemm-tail-physical-route",
        "tree-all-reduce",
    ):
        case = driver.CASES[key]
        driver.validate_payloads(case, case.payload_factory())

    case = driver.CASES["tree-all-reduce"]
    assert case.inputs[0].shape == (driver.TILE_COUNT, 4096)
    assert case.outputs[0].shape == (4096,)
    with tempfile.TemporaryDirectory() as directory:
        source = driver.write_source(pathlib.Path(directory), case)
        module = (source / "functions" / "forward.mlir").read_text()
        metadata = json.loads(
            (source / "functions" / "forward.meta").read_text()
        )
        assert "stablehlo.reduce" in module
        assert "mhlo.sharding" not in module
        assert "@Sharding" not in module
        assert "distributed_boundary" not in metadata
        assert metadata["input_signature"][0]["shape"] == [16, 4096]


def validate_m_tiled_profile_runner() -> None:
    assert m_tiled_profile.CASE is driver.CASES["noc-resident-m-tiled-gemm"]
    with tempfile.TemporaryDirectory() as directory:
        source = m_tiled_profile.write_source_program(pathlib.Path(directory))
        module = (source / "functions" / "forward.mlir").read_text()
        metadata = json.loads(
            (source / "functions" / "forward.meta").read_text()
        )
        assert "stablehlo.dot_general" in module
        assert "mhlo.sharding" not in module
        assert metadata["input_signature"] == [
            {"shape": [4096, 1024], "dtype": "float16", "dynamic_dims": []},
            {"shape": [1024, 4096], "dtype": "float16", "dynamic_dims": []},
        ]

    base = {
        "package: id=0 cards=1 tiles=16",
        "invocation_tiles: 16",
        "board_execution: false",
    }
    m_tiled_profile.verify_no_card("\n".join(sorted(base)), profile=False)
    m_tiled_profile.verify_no_card(
        "\n".join(
            sorted(base | {m_tiled_profile.profile_support.PROFILE_INSTRUMENTATION_READY})
        ),
        profile=True,
    )


def validate_catalog(repo: pathlib.Path) -> None:
    assert set(catalog.CASES_BY_KEY) == EXPECTED_CASE_KEYS
    assert len(catalog.CASES_BY_KEY) == len(
        catalog.OPTIMIZATION_COMPARISON_CASES
    )
    assert catalog.BOARD_AXES
    assert catalog.SOURCE_COMPARISON_AXES
    assert catalog.HOST_ONLY_AXES
    assert catalog.PENDING_CONFIGURED_BOARD_AXES

    case_orders = [
        case.board_order for case in catalog.OPTIMIZATION_COMPARISON_CASES
    ]
    assert len(case_orders) == len(set(case_orders))
    for case in catalog.OPTIMIZATION_COMPARISON_CASES:
        assert case.priority == "P0"
        assert case.participant_count in {1, driver.TILE_COUNT}
        assert case.execution_scope in {
            catalog.ExecutionScope.PROGRAM_LOCAL,
            catalog.ExecutionScope.TILE_COLLECTIVE,
        }
        assert case.oracle.structural_checks
        assert case.oracle.numeric_checks
        assert case.oracle.performance_checks
        assert case.completion_contract
        assert (repo / case.execution_asset).is_file()
        for asset in case.existing_assets:
            assert (repo / asset).is_file()
        if case.requires_pair:
            assert case.key in driver.CASES
            assert case.disposition == (
                catalog.AxisDisposition.SOURCE_COMPARISON_PENDING_LOWERING
            )
            assert case.package_roles == ("none", "search")
            assert not case.oracle.executable_capabilities
            assert not case.oracle.is_strong
            assert case.driver_binding is None
            assert driver.CASES[case.key].structural_checks
        else:
            assert case.disposition == catalog.AxisDisposition.EXISTING_BOARD_FAMILY
            assert case.oracle.is_strong

    for anchor in (
        *catalog.PIPELINE_IMPLEMENTATION_REFERENCES,
        *catalog.IMPLEMENTATION_REFERENCE_BY_AXIS.values(),
        *catalog.EXECUTION_CAPABILITY_ANCHORS.values(),
    ):
        source = (repo / anchor.asset).read_text()
        assert anchor.markers
        assert all(marker in source for marker in anchor.markers)

    axis_keys = {axis.key for axis in catalog.OPTIMIZATION_AXES}
    assert axis_keys == set(catalog.IMPLEMENTATION_REFERENCE_BY_AXIS)
    assert set(catalog.AXES_BY_CASE) == EXPECTED_CASE_KEYS
    for case_key, axes in catalog.AXES_BY_CASE.items():
        assert axes
        assert all(axis.evidence_key == case_key for axis in axes)


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    validate_driver_cases()
    validate_m_tiled_profile_runner()
    validate_catalog(repo)
    print(
        "compiler_optimization_source_contract: "
        f"cases={len(driver.CASES)} policies=none,search"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
