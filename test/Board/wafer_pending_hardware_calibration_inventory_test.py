#!/usr/bin/env python3
"""Audit every pending calibration family against executable repository assets."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence

import wafer_pending_hardware_calibration_inventory as inventory


EXPECTED_PENDING_BOARD_COUNTS = {
    "production-optimizer-paired-qualification": (8, 8),
    "collective-algorithm-characterization": (9, 9),
    "collective-traffic-semantics": (11, 11),
    "single-engine-and-engine-pair-characterization": (62, 381),
    "queue-saturation-response": (30, 30),
    "worker-wait-scope-exclusion": (18, 18),
    "worker-subset-join-exclusion": (12, 12),
    "worker-placement-and-bounded-progress": (8, 32),
    "spm-conflict-equivalence": (2, 136),
    "spm-sustained-conflict-pilot": (4, 16),
    "ddr-conflict-equivalence": (2, 492),
    "ddr-active-rank-contention": (9, 45),
    "argmin-tie-and-nan-domain": (2, 2),
    "unpool-repeated-overlap-collision": (2, 2),
    "native-concat-hw-isolated-requalification": (1, 1),
}


def _load_module(
    repo: pathlib.Path, relative: str, cache: dict[str, object]
) -> object:
    if relative in cache:
        return cache[relative]
    path = repo / relative
    assert path.is_file(), f"missing pending calibration asset {relative}"
    module_name = "_pending_calibration_" + re.sub(r"[^a-zA-Z0-9]", "_", relative)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    cache[relative] = module
    return module


def _resolve_binding(
    repo: pathlib.Path,
    binding: inventory.CatalogBinding,
    cache: dict[str, object],
) -> tuple[object, ...]:
    module = _load_module(repo, binding.asset, cache)
    assert hasattr(module, binding.symbol), (
        f"{binding.asset}: missing symbol {binding.symbol}"
    )
    collection = getattr(module, binding.symbol)
    if binding.identifier_field == "@callable":
        assert callable(collection)
        rows = tuple(collection())
        assert rows, f"{binding.asset}:{binding.symbol} returned no cases"
        return rows
    if binding.identifier_field == "@mapping":
        assert isinstance(collection, Mapping)
        missing = set(binding.identifiers) - set(collection)
        assert not missing, (
            f"{binding.asset}:{binding.symbol} missing {sorted(missing)}"
        )
        return tuple(collection[key] for key in binding.identifiers)
    assert isinstance(collection, Sequence) and not isinstance(
        collection, (str, bytes)
    )
    rows = tuple(collection)
    assert rows, f"{binding.asset}:{binding.symbol} is empty"
    if not binding.identifiers:
        return rows
    indexed = {
        getattr(row, binding.identifier_field): row
        for row in rows
    }
    missing = set(binding.identifiers) - set(indexed)
    assert not missing, (
        f"{binding.asset}:{binding.symbol} missing {sorted(missing)}"
    )
    return tuple(indexed[key] for key in binding.identifiers)


def _cmake_set(cmake: str, variable: str) -> tuple[str, ...]:
    match = re.search(
        rf"set\({re.escape(variable)}\s+(?P<body>.*?)\n\s*\)",
        cmake,
        re.DOTALL,
    )
    assert match is not None, f"missing CMake registry {variable}"
    return tuple(match["body"].split())


def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    inventory.validate_inventory()
    cache: dict[str, object] = {}
    resolved_pending_counts: dict[str, tuple[int, int]] = {}
    for family in inventory.FAMILIES:
        resolved = tuple(
            row
            for binding in family.bindings
            for row in _resolve_binding(repo, binding, cache)
        )
        assert resolved, f"{family.key}: no concrete catalog rows resolved"
        if family.disposition == inventory.PENDING_BOARD:
            resolved_pending_counts[family.key] = (
                len(family.board_ctests),
                len(resolved),
            )
    assert resolved_pending_counts == EXPECTED_PENDING_BOARD_COUNTS
    assert sum(
        ctests for ctests, _ in resolved_pending_counts.values()
    ) == 180
    assert sum(
        cells for _, cells in resolved_pending_counts.values()
    ) == 1195

    memory = _load_module(
        repo,
        "test/Board/wafer_memory_descriptor_calibration_catalog.py",
        cache,
    )
    spm_family = inventory.FAMILIES_BY_KEY["spm-conflict-equivalence"]
    assert spm_family.bindings[0].symbol == (
        "PENDING_CONFLICT_EQUIVALENCE_CASES"
    )
    pending_spm_names = {
        case.name for case in memory.PENDING_CONFLICT_EQUIVALENCE_CASES
    }
    baseline_spm_names = {
        case.name for case in memory.CONFLICT_EQUIVALENCE_BASELINE_CASES
    }
    assert len(pending_spm_names) == 88
    assert len(baseline_spm_names) == 8
    assert pending_spm_names.isdisjoint(baseline_spm_names)
    assert len(memory.CONFLICT_EQUIVALENCE_CASES) == 96

    ncc = _load_module(
        repo,
        "test/Board/wafer_board_ncc_execution_probe_test.py",
        cache,
    )
    assert tuple(case.name for case in ncc.V2_QUEUE_SATURATION_CASES) == (
        inventory.QUEUE_SATURATION_CASES
    )
    assert tuple(case.name for case in ncc.V2_WORKER_WAIT_SCOPE_CASES) == (
        inventory.WORKER_WAIT_SCOPE_CASES
    )
    assert tuple(case.name for case in ncc.V2_WORKER_SUBSET_SCOPE_CASES) == (
        inventory.WORKER_SUBSET_SCOPE_CASES
    )

    cmake = (repo / "test/CMakeLists.txt").read_text()
    assert _cmake_set(
        cmake, "_wafer_pending_ncc_queue_saturation_cases"
    ) == inventory.QUEUE_SATURATION_CASES
    assert _cmake_set(
        cmake, "_wafer_pending_ncc_worker_wait_scope_cases"
    ) == inventory.WORKER_WAIT_SCOPE_CASES
    assert _cmake_set(
        cmake, "_wafer_pending_ncc_worker_subset_scope_cases"
    ) == inventory.WORKER_SUBSET_SCOPE_CASES
    assert _cmake_set(
        cmake, "_wafer_pending_argmin_domain_cases"
    ) == inventory.ARGMIN_DOMAIN_CASES
    assert _cmake_set(
        cmake, "_wafer_pending_unpool_collision_cases"
    ) == inventory.UNPOOL_COLLISION_CASES
    assert _cmake_set(
        cmake, "_wafer_collective_traffic_behavior_cases"
    ) == inventory.COLLECTIVE_TRAFFIC_CASES
    assert (
        "wafer-board-collective-traffic-behavior-"
        "${_wafer_collective_traffic_case}"
    ) in cmake
    assert (
        "wafer-runtime-collective-traffic-behavior-"
        "${_wafer_collective_traffic_case}-no-card"
    ) in cmake
    assert "--emit-board-group-keys" in cmake
    assert (
        "wafer-board-engine-pipeline-"
        "${_wafer_engine_pipeline_group}"
    ) in cmake
    assert "--group ${_wafer_engine_pipeline_group}" in cmake
    assert "NAME wafer-board-ne-tail-throughput-single-ne-tail" in cmake
    assert "--emit-board-case-keys" in cmake
    assert (
        "wafer-board-spm-sustained-"
        "${_wafer_spm_sustained_group}"
    ) in cmake
    assert "--case ${_wafer_spm_sustained_group}" in cmake
    assert (
        "wafer-board-ddr-active-rank-"
        "${_wafer_ddr_active_rank_group}"
    ) in cmake
    assert "--group ${_wafer_ddr_active_rank_group}" in cmake
    assert (
        "wafer-board-worker-placement-"
        "${_wafer_worker_placement_group}"
    ) in cmake
    assert "--group ${_wafer_worker_placement_group}" in cmake
    for ctest in (
        "wafer-runtime-engine-pipeline-characterization-no-card",
        "wafer-engine-pipeline-production-gate-python",
        "wafer-engine-pipeline-characterization-catalog-python",
        "wafer-collective-traffic-behavior-catalog-python",
        "wafer-runtime-spm-sustained-conflict-probe-no-card",
        "wafer-spm-sustained-conflict-contract-python",
        "wafer-runtime-ne-tail-throughput-no-card",
        "wafer-ne-tail-throughput-catalog-python",
        "wafer-runtime-ddr-active-rank-contention-no-card",
        "wafer-ddr-active-rank-contention-contract-python",
        "wafer-runtime-worker-placement-characterization-no-card",
        "wafer-worker-placement-characterization-python",
        "wafer-unrepresentable-hardware-behavior-python",
    ):
        assert f"NAME {ctest}" in cmake, f"missing CTest {ctest}"
    for ctest in (
        "wafer-board-spm-conflict-equivalence-rank-one",
        "wafer-board-spm-conflict-equivalence-cross-tile",
        "wafer-board-ddr-conflict-equivalence-rank-one",
        "wafer-board-ddr-conflict-equivalence-cross-tile",
    ):
        assert f"NAME {ctest}" in cmake, f"missing board CTest {ctest}"
    assert "wafer-board-ncc-${test_suffix}" in cmake

    runner = _load_module(
        repo, "tools/run_hardware_calibration.py", cache
    )
    steps_by_ctest = {
        step.ctest_name: step for step in runner.EXPLICIT_ONLY_STEPS
    }
    for family in inventory.FAMILIES:
        if family.disposition != inventory.PENDING_BOARD:
            continue
        missing_steps = set(family.board_ctests) - set(steps_by_ctest)
        assert not missing_steps, (
            f"{family.key}: board CTests lack runner steps "
            f"{sorted(missing_steps)}"
        )
        if family.runner_batch is not None:
            assert family.runner_batch in runner.SELECTABLE_BATCHES
            batch_keys = set(runner.SELECTABLE_BATCHES[family.runner_batch])
            for ctest in family.board_ctests:
                assert steps_by_ctest[ctest].key in batch_keys, (
                    f"{family.key}: {ctest} is outside "
                    f"{family.runner_batch}"
                )

    assert tuple(runner.QUEUE_SATURATION_CASE_NAMES) == (
        inventory.QUEUE_SATURATION_CASES
    )
    assert tuple(runner.WORKER_WAIT_SCOPE_CASE_NAMES) == (
        inventory.WORKER_WAIT_SCOPE_CASES
    )
    assert tuple(runner.WORKER_SUBSET_SCOPE_CASE_NAMES) == (
        inventory.WORKER_SUBSET_SCOPE_CASES
    )
    assert tuple(runner.ARGMIN_PENDING_DOMAIN_CASE_NAMES) == (
        inventory.ARGMIN_DOMAIN_CASES
    )
    assert tuple(runner.UNPOOL_PENDING_COLLISION_CASE_NAMES) == (
        inventory.UNPOOL_COLLISION_CASES
    )
    assert tuple(runner.COLLECTIVE_TRAFFIC_BEHAVIOR_CASES) == (
        inventory.COLLECTIVE_TRAFFIC_CASES
    )
    assert tuple(runner.ENGINE_PIPELINE_BOARD_CELL_KEYS) == (
        inventory.ENGINE_PIPELINE_BOARD_CELL_KEYS
    )
    assert tuple(runner.ENGINE_PIPELINE_BOARD_GROUP_KEYS) == (
        inventory.ENGINE_PIPELINE_BOARD_GROUP_KEYS
    )
    assert tuple(runner.ENGINE_PIPELINE_EXTERNAL_GROUP_KEYS) == (
        inventory.ENGINE_PIPELINE_EXTERNAL_GROUP_KEYS
    )
    assert tuple(runner.SPM_SUSTAINED_BOARD_GROUP_KEYS) == (
        inventory.SPM_SUSTAINED_GROUP_KEYS
    )
    assert tuple(runner.DDR_ACTIVE_RANK_CASE_KEYS) == (
        inventory.DDR_ACTIVE_RANK_CASE_KEYS
    )
    assert tuple(runner.DDR_ACTIVE_RANK_BOARD_GROUP_KEYS) == (
        inventory.DDR_ACTIVE_RANK_GROUP_KEYS
    )
    assert tuple(runner.WORKER_PLACEMENT_CASE_KEYS) == (
        inventory.WORKER_PLACEMENT_CASE_KEYS
    )
    assert tuple(runner.WORKER_PLACEMENT_BOARD_GROUP_KEYS) == (
        inventory.WORKER_PLACEMENT_GROUP_KEYS
    )
    pending_ctests = {
        ctest
        for family in inventory.FAMILIES
        if family.disposition == inventory.PENDING_BOARD
        for ctest in family.board_ctests
    }
    master_batch = runner.SELECTABLE_BATCHES[
        "pending-hardware-calibration"
    ]
    master_steps = {
        step.key: step for step in runner.EXPLICIT_ONLY_STEPS
    }
    selected_ctests = tuple(
        master_steps[key].ctest_name for key in master_batch
    )
    assert len(selected_ctests) == len(pending_ctests)
    assert set(selected_ctests) == pending_ctests
    assert master_batch[-1] == "datamove-native-concat-hw-isolated"


if __name__ == "__main__":
    main()
