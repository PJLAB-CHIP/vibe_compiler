#!/usr/bin/env python3
"""Validate current calibration inventory bindings and execution classes."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence

import wafer_pending_hardware_calibration_inventory as inventory


def load_module(
    repo: pathlib.Path, relative: str, cache: dict[str, object]
) -> object:
    if relative in cache:
        return cache[relative]
    path = repo / relative
    assert path.is_file(), f"missing calibration asset {relative}"
    name = "_wafer_calibration_" + re.sub(r"[^a-zA-Z0-9]", "_", relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cache[relative] = module
    return module


def resolve_binding(
    repo: pathlib.Path,
    binding: inventory.CatalogBinding,
    cache: dict[str, object],
) -> tuple[object, ...]:
    module = load_module(repo, binding.asset, cache)
    assert hasattr(module, binding.symbol)
    collection = getattr(module, binding.symbol)
    if binding.identifier_field == "@callable":
        assert callable(collection)
        return tuple(collection())
    if binding.identifier_field == "@mapping":
        assert isinstance(collection, Mapping)
        assert set(binding.identifiers).issubset(collection)
        return tuple(collection[key] for key in binding.identifiers)
    assert isinstance(collection, Sequence) and not isinstance(
        collection, (str, bytes)
    )
    rows = tuple(collection)
    if not binding.identifiers:
        return rows
    indexed = {getattr(row, binding.identifier_field): row for row in rows}
    assert set(binding.identifiers).issubset(indexed)
    return tuple(indexed[key] for key in binding.identifiers)


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    board_cmake = (repo / "test/Board/CMakeLists.txt").read_text()
    inventory.validate_inventory()
    cache: dict[str, object] = {}
    for family in inventory.FAMILIES:
        resolved = tuple(
            row
            for binding in family.bindings
            for row in resolve_binding(repo, binding, cache)
        )
        assert resolved, f"{family.key}: catalog binding is empty"
        names = (
            *family.host_ctests,
            *family.no_card_ctests,
            *family.board_ctests,
        )
        assert len(names) == len(set(names)), f"{family.key}: duplicate CTest"
        assert not any(
            token in name
            for name in names
            for token in ("rank-one", "all-rank", "execution-ranks")
        )
        assert all(
            name in board_cmake for name in family.host_ctests
        ), f"{family.key}: host CTest is not registered"
        if family.disposition == inventory.BLOCKED_CURRENT_PIPELINE:
            assert all(
                name not in board_cmake for name in family.no_card_ctests
            ), f"{family.key}: blocked no-card CTest remains registered"
        else:
            assert all(
                name in board_cmake for name in family.no_card_ctests
            ), f"{family.key}: no-card CTest is not registered"
        assert all(
            name in board_cmake for name in family.board_ctests
        ), f"{family.key}: Board CTest is not registered"
        if family.runner_batch is not None:
            assert family.board_ctests

    for key, expected_host_test in (
        (
            "collective-algorithm-characterization",
            "wafer-collective-algorithm-source-contract",
        ),
        (
            "collective-traffic-semantics",
            "wafer-collective-traffic-source-contract",
        ),
    ):
        family = inventory.FAMILIES_BY_KEY[key]
        assert family.disposition == inventory.BLOCKED_EXTERNAL
        assert family.execution_scope == "current-global-source-contract"
        assert family.host_ctests == (expected_host_test,)
        assert not family.no_card_ctests
        assert not family.board_ctests
        assert family.runner_batch is None
        assert "current" in family.blocker

    raw_dte = inventory.FAMILIES_BY_KEY[
        "direct-dte-raw-multidestination-and-fanin"
    ]
    assert raw_dte.disposition == inventory.BLOCKED_CURRENT_PIPELINE
    assert "wafer-runtime-dte-ncc-execution-probe-no-card" in (
        raw_dte.no_card_ctests
    )
    assert raw_dte.board_ctests

    pending = tuple(
        family
        for family in inventory.FAMILIES
        if family.disposition == inventory.PENDING_BOARD
    )
    blocked = tuple(
        family
        for family in inventory.FAMILIES
        if family.disposition == inventory.BLOCKED_EXTERNAL
    )
    pipeline_blocked = tuple(
        family
        for family in inventory.FAMILIES
        if family.disposition == inventory.BLOCKED_CURRENT_PIPELINE
    )
    assert not pending and pipeline_blocked and blocked
    assert all(family.no_card_ctests for family in pending)
    assert all(
        family.board_ctests
        for family in pending
        if family.runner_batch is not None
    )
    assert all(family.blocker for family in blocked)
    assert all(
        family.blocker == inventory.CURRENT_PIPELINE_BLOCKER
        for family in pipeline_blocked
    )
    print(
        "calibration_inventory: "
        f"families={len(inventory.FAMILIES)} "
        f"board_pending={len(pending)} "
        f"pipeline_blocked={len(pipeline_blocked)} "
        f"blocked={len(blocked)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
