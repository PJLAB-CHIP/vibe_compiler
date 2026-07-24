#!/usr/bin/env python3
"""Validate complete calibration preparation and explicit evidence sharing."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence

import wafer_hardware_calibration_matrix as matrix


EXPECTED_KEYS = {
    "profile-qualification",
    "constructor-ownership",
    "execute-result",
    "packet-routing-range",
    "ct-numeric-form",
    "instruction-physical-layout",
    "datamove-layout",
    "ne-numeric-layout",
    "rdma-wdma-descriptor",
    "tdma-memset",
    "tdma-bool-fill",
    "tdma-movement-variants",
    "spm-capacity-reservation",
    "spm-alignment-bank",
    "ddr-cache-coherence",
    "queue-shape-submission",
    "worker-scope",
    "cross-engine-overlap",
    "address-dependency",
    "issue-overhead",
    "local-completion",
    "cross-worker-join",
    "direct-dte",
    "multi-tile-arrival",
    "host-launch-runtime",
    "ncc-pmu-basis",
    "transport-pmu-basis",
    "scalar-csr-ordinary-issue",
}

EXPECTED_SHARED_GROUP_REFERENCES = {
    (
        "test/Board/wafer_board_single_op_add_test.py",
        "rank-one-per-rank-add",
    ): 2,
    (
        "test/Board/wafer_board_ncc_execution_probe_test.py",
        "routing-five-engines-worker0",
    ): 2,
    (
        "test/Board/wafer_ct_vector_calibration_catalog.py",
        "ct-vector-forms-main-tail",
    ): 2,
    (
        "test/Board/wafer_datamove_calibration_catalog.py",
        "cx-ncx-channel-boundaries",
    ): 2,
    (
        "test/Board/wafer_datamove_calibration_catalog.py",
        "transpose-mirror-rotate-large",
    ): 2,
    (
        "test/Board/wafer_ne_calibration_catalog.py",
        "ne-gemm-dtype-orientation-main-tail-batch",
    ): 2,
}


def load_asset_module(
    repo: pathlib.Path,
    relative: str,
    cache: dict[str, object],
) -> object:
    if relative in cache:
        return cache[relative]
    path = repo / relative
    module_name = (
        "_wafer_calibration_leaf_"
        + relative.replace("/", "_").replace(".", "_")
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, (
        f"cannot load calibration asset {relative}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    cache[relative] = module
    return module


def resolve_binding(
    repo: pathlib.Path,
    binding: matrix.CatalogBinding,
    cache: dict[str, object],
) -> tuple[object, ...]:
    path = repo / binding.asset
    assert path.is_file(), f"missing leaf asset {binding.asset}"
    module = load_asset_module(repo, binding.asset, cache)
    assert hasattr(module, binding.symbol), (
        f"{binding.asset}: missing symbol {binding.symbol}"
    )
    collection = getattr(module, binding.symbol)
    if binding.identifier_field == "@mapping":
        assert isinstance(collection, Mapping), (
            f"{binding.asset}:{binding.symbol} is not a mapping"
        )
        resolved: list[object] = []
        for identifier in binding.identifiers:
            assert identifier in collection, (
                f"{binding.asset}:{binding.symbol} has no leaf group "
                f"{identifier!r}"
            )
            rows = collection[identifier]
            assert isinstance(rows, Sequence) and not isinstance(
                rows, (str, bytes)
            ), (
                f"{binding.asset}:{binding.symbol}[{identifier!r}] "
                "must be a row sequence"
            )
            assert rows, (
                f"{binding.asset}:{binding.symbol}[{identifier!r}] is empty"
            )
            for row in rows:
                assert not isinstance(row, (str, bytes, int, float)), (
                    f"{binding.asset}:{binding.symbol}[{identifier!r}] "
                    "must reference concrete catalog/contract objects"
                )
                if isinstance(row, Mapping):
                    assert any(
                        field in row for field in ("case_id", "id", "key", "name")
                    ) or "opcode" in row or (
                        "engine" in row and "layout" in row
                    ), (
                        f"{binding.asset}:{binding.symbol}[{identifier!r}] "
                        "mapping rows need a stable case identifier"
                    )
                else:
                    assert any(
                        hasattr(row, field)
                        for field in ("case_id", "id", "key", "name")
                    ) or hasattr(row, "opcode") or (
                        hasattr(row, "engine") and hasattr(row, "layout")
                    ), (
                        f"{binding.asset}:{binding.symbol}[{identifier!r}] "
                        "objects need a stable case identifier"
                    )
            resolved.extend(rows)
        return tuple(resolved)

    if isinstance(collection, Mapping):
        rows = tuple(collection.values())
    else:
        assert isinstance(collection, Sequence) and not isinstance(
            collection, (str, bytes)
        )
        rows = tuple(collection)
    indexed = {
        getattr(row, binding.identifier_field): row
        for row in rows
    }
    missing = set(binding.identifiers) - set(indexed)
    assert not missing, (
        f"{binding.asset}:{binding.symbol} missing identifiers "
        f"{sorted(missing, key=str)}"
    )
    return tuple(indexed[identifier] for identifier in binding.identifiers)


def row_value(row: object, field: str) -> object | None:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def stable_row_id(row: object) -> tuple[str, object]:
    for field in ("case_id", "id", "key", "name", "opcode"):
        value = row_value(row, field)
        if value is not None:
            return field, value
    engine = row_value(row, "engine")
    layout = row_value(row, "layout")
    if engine is not None and layout is not None:
        return "engine-layout", (engine, layout)
    raise AssertionError("calibration row has no stable identifier")


def is_non_board_row(row: object) -> bool:
    if row_value(row, "is_safe") is False:
        return True
    for field in ("disposition_name", "disposition", "preparation"):
        value = row_value(row, field)
        if isinstance(value, str):
            normalized = value.lower().replace("_", "-")
            if any(
                marker in normalized
                for marker in (
                    "deferred",
                    "negative",
                    "excluded",
                    "unsupported",
                    "unprepared",
                    "delegated",
                    "unknown",
                )
            ):
                return True
    return False


def non_board_disposition(row: object) -> str | None:
    for field in ("disposition_name", "disposition", "preparation"):
        value = row_value(row, field)
        if not isinstance(value, str):
            continue
        normalized = value.lower().replace("_", "-")
        if normalized == "static-negative":
            return "static-negative"
        if normalized == "isolated-deferred":
            return "isolated-deferred"
        if normalized == "unknown":
            return "unknown"
    return None


def is_observation_row(row: object) -> bool:
    for field in ("disposition_name", "disposition"):
        value = row_value(row, field)
        if isinstance(value, str):
            normalized = value.lower().replace("_", "-")
            if "observed" in normalized or "observation" in normalized:
                return True
    return False


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    board_dir = str(repo / "test" / "Board")
    if board_dir not in sys.path:
        sys.path.insert(0, board_dir)
    domains = matrix.CALIBRATION_DOMAINS
    assert len(domains) == len(EXPECTED_KEYS)
    assert set(matrix.DOMAINS_BY_KEY) == EXPECTED_KEYS
    assert len(matrix.DOMAINS_BY_KEY) == len(domains)
    assert set(matrix.CALIBRATION_LEAVES_BY_DOMAIN) == EXPECTED_KEYS
    all_leaves = tuple(
        leaf
        for leaves in matrix.CALIBRATION_LEAVES_BY_DOMAIN.values()
        for leaf in leaves
    )
    assert len(matrix.LEAVES_BY_KEY) == len(all_leaves)
    assert len({leaf.key for leaf in all_leaves}) == len(all_leaves)

    labels = {domain.document_label for domain in domains}
    assert len(labels) == len(domains)
    calibration = (
        repo / "docs" / "tx81-compiler-hardware-calibration.md"
    ).read_text()
    matrix_body = calibration[
        calibration.index("## 3. Compiler-sensitive calibration matrix") :
        calibration.index("## 4. 当前profile已经闭合的事实")
    ]
    document_rows = {
        match.group(1)
        for match in re.finditer(r"^\| ([^|]+?) \|", matrix_body, re.M)
        if match.group(1) not in {"域", "---"}
    }
    assert labels == document_rows

    cmake = (repo / "test" / "CMakeLists.txt").read_text()
    allowed_states = {"ready", "in-progress"}
    allowed_layers = {
        "calibration",
        "held-out",
        "production-vertical",
    }
    allowed_dispositions = {
        "board-positive",
        "board-observation",
        "delegated-positive",
        "static-negative",
        "isolated-deferred",
        "unknown",
    }
    module_cache: dict[str, object] = {}
    referenced_groups: Counter[tuple[str, str | int]] = Counter()
    for domain in domains:
        assert domain.preparation in allowed_states
        assert domain.execution_scope
        assets = domain.positive_assets + domain.negative_assets
        assert len(set(domain.positive_assets)) == len(
            domain.positive_assets
        ), f"{domain.key}: duplicate positive assets hide accounting mistakes"
        assert len(set(domain.negative_assets)) == len(
            domain.negative_assets
        ), f"{domain.key}: duplicate negative assets hide accounting mistakes"
        assert domain.no_card_tests, (
            f"{domain.key}: preparation requires at least one no-card gate"
        )
        if domain.preparation == "ready":
            assert assets
            assert not domain.remaining_preparation
        else:
            assert domain.remaining_preparation
        if domain.execution_scope != "static-negative":
            assert domain.positive_assets, (
                f"{domain.key}: executable domain has no positive asset"
            )
            assert any(
                "/wafer_board_" in relative
                for relative in domain.positive_assets
            ), f"{domain.key}: executable domain has no board runner"
        for relative in assets:
            path = repo / relative
            assert path.is_file(), f"{domain.key}: missing asset {relative}"
        for test_name in domain.no_card_tests:
            generated_launch_test = test_name in {
                "wafer-runtime-kernel-grid-add-no-card",
                "wafer-runtime-model-add-no-card",
            }
            registered = f"NAME {test_name}" in cmake
            if generated_launch_test:
                registered = (
                    "kernel-grid model" in cmake
                    and "wafer-runtime-${_wafer_no_card_launch_test}-add-no-card"
                    in cmake
                )
            assert registered, (
                f"{domain.key}: CTest {test_name} is not registered"
            )

        leaves = matrix.CALIBRATION_LEAVES_BY_DOMAIN[domain.key]
        assert leaves, f"{domain.key}: domain has no leaf requirements"
        for leaf in leaves:
            assert leaf.layer in allowed_layers, (
                f"{leaf.key}: invalid layer {leaf.layer}"
            )
            assert leaf.disposition in allowed_dispositions, (
                f"{leaf.key}: invalid disposition {leaf.disposition}"
            )
            assert leaf.execution_scope
            assert leaf.resource_budget
            assert leaf.bindings, f"{leaf.key}: no concrete catalog binding"
            for binding in leaf.bindings:
                if binding.identifier_field == "@mapping":
                    referenced_groups.update(
                        (binding.asset, identifier)
                        for identifier in binding.identifiers
                    )
            resolved = tuple(
                row
                for binding in leaf.bindings
                for row in resolve_binding(repo, binding, module_cache)
            )
            assert resolved, f"{leaf.key}: binding resolved no rows"
            row_ids = tuple(stable_row_id(row) for row in resolved)
            assert len(set(row_ids)) == len(row_ids), (
                f"{leaf.key}: duplicate concrete rows in leaf binding"
            )
            if leaf.preparation == "ready":
                assert not leaf.remaining_preparation
            else:
                assert leaf.remaining_preparation
            if leaf.disposition in {
                "board-positive",
                "board-observation",
            }:
                assert leaf.oracle, f"{leaf.key}: missing independent oracle"
                assert leaf.guards, f"{leaf.key}: missing physical guards"
                assert leaf.completion, (
                    f"{leaf.key}: missing completion/publication contract"
                )
                assert domain.positive_assets, (
                    f"{leaf.key}: parent domain has no board-positive assets"
                )
                assert not any(is_non_board_row(row) for row in resolved), (
                    f"{leaf.key}: board leaf binds deferred/negative "
                    "catalog rows"
                )
                assert any(
                    "/wafer_board_" in relative
                    for relative in domain.positive_assets
                ), f"{leaf.key}: parent domain has no board runner"
                if leaf.disposition == "board-positive":
                    assert not leaf.reason
                    assert not any(
                        is_observation_row(row) for row in resolved
                    ), (
                        f"{leaf.key}: exact/tolerance positive leaf binds "
                        "observation-only rows"
                    )
                else:
                    assert leaf.reason, (
                        f"{leaf.key}: board observation requires the "
                        "unresolved semantic reason"
                    )
                    assert all(
                        is_observation_row(row) for row in resolved
                    ), (
                        f"{leaf.key}: board-observation leaf must contain "
                        "only observation rows"
                    )
            elif leaf.disposition == "delegated-positive":
                assert leaf.reason, (
                    f"{leaf.key}: delegated execution needs an owner reason"
                )
                assert leaf.oracle and leaf.guards and leaf.completion, (
                    f"{leaf.key}: delegated execution lost its board oracle"
                )
                assert domain.positive_assets, (
                    f"{leaf.key}: parent domain has no positive asset"
                )
                for row in resolved:
                    disposition = row_value(row, "disposition")
                    assert (
                        isinstance(disposition, str)
                        and "delegated" in disposition.lower()
                    ), (
                        f"{leaf.key}: delegated-positive leaf contains a "
                        "non-delegation row"
                    )
                    evidence = row_value(row, "evidence")
                    assert isinstance(evidence, Sequence) and evidence, (
                        f"{leaf.key}: delegation has no concrete evidence"
                    )
                    assert all(
                        not is_non_board_row(item) for item in evidence
                    ), (
                        f"{leaf.key}: delegation references non-board "
                        "evidence"
                    )
                    tuple(stable_row_id(item) for item in evidence)
            else:
                assert leaf.reason, (
                    f"{leaf.key}: non-board disposition requires a reason"
                )
                assert all(is_non_board_row(row) for row in resolved), (
                    f"{leaf.key}: non-board leaf contains a board-executable "
                    "row instead of only typed negative/deferred rows"
                )
                assert all(
                    non_board_disposition(row) == leaf.disposition
                    for row in resolved
                ), (
                    f"{leaf.key}: leaf disposition {leaf.disposition} does "
                    "not exactly match every concrete catalog row"
                )

        derived_preparation = (
            "ready"
            if all(leaf.preparation == "ready" for leaf in leaves)
            else "in-progress"
        )
        assert domain.preparation == derived_preparation, (
            f"{domain.key}: row state {domain.preparation} does not match "
            f"leaf-derived state {derived_preparation}"
        )

    grouped_references: dict[str, set[str | int]] = {}
    for asset, identifier in referenced_groups:
        grouped_references.setdefault(asset, set()).add(identifier)
    for asset, identifiers in grouped_references.items():
        module = load_asset_module(repo, asset, module_cache)
        groups = getattr(module, "CALIBRATION_LEAF_BINDINGS")
        assert set(groups) == identifiers, (
            f"{asset}: leaf groups are not fully accounted; "
            f"unreferenced={sorted(set(groups) - identifiers, key=str)}, "
            f"missing={sorted(identifiers - set(groups), key=str)}"
        )
    shared_references = {
        key: count for key, count in referenced_groups.items() if count > 1
    }
    assert shared_references == EXPECTED_SHARED_GROUP_REFERENCES, (
        "shared leaf-group references changed without an explicit "
        f"accounting update: {shared_references}"
    )

    incomplete = tuple(
        leaf.key for leaf in all_leaves if leaf.preparation != "ready"
    )
    assert not incomplete
    print(
        "wafer_hardware_calibration_matrix_test: "
        f"domains={len(domains)} leaves={len(all_leaves)} "
        f"incomplete={len(incomplete)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
