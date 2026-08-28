#!/usr/bin/env python3
"""Validate collective algorithm source and host-oracle contracts."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

import wafer_board_collective_characterization_runner as driver
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


def validate_payload(case: catalog.CollectiveCharacterizationCase) -> None:
    values = driver.payloads(case)
    assert values.input.dtype == np.dtype("<f2")
    assert values.expected.dtype == np.dtype("<f2")
    assert values.input.shape[0] == driver.TILE_COUNT
    assert len(values.tile_contributions) == driver.TILE_COUNT
    assert len({value.tobytes() for value in values.tile_contributions}) == 16
    assert np.all(np.isfinite(values.input))
    assert np.all(np.isfinite(values.expected))

    if case.collective_kind == catalog.CollectiveKind.ALL_GATHER:
        flat = values.input.reshape(-1)
        assert values.expected.shape == (16, flat.size)
        assert all(np.array_equal(row, flat) for row in values.expected)
        return

    reduced = np.sum(values.input.astype(np.int32), axis=0).astype("<f2")
    if case.collective_kind == catalog.CollectiveKind.REDUCE_SCATTER:
        assert np.array_equal(values.expected.reshape(-1), reduced)
    else:
        assert values.expected.shape == (16, reduced.size)
        assert all(np.array_equal(row, reduced) for row in values.expected)

    for source in values.tile_contributions:
        missing = (
            values.input.astype(np.int32).sum(axis=0)
            - source.astype(np.int32)
        ).astype("<f2")
        assert not np.array_equal(missing, reduced)


def validate_message_relations() -> None:
    rows = {tile: {"messages": []} for tile in range(16)}
    for tile in range(16):
        successor = (tile + 1) % 16
        predecessor = (tile - 1) % 16
        rows[tile]["messages"].extend(
            (
                {
                    "direction": "send", "peer": successor,
                    "communication_id": 7, "phase": "all_gather_ring",
                    "round": 0, "payload_slice": tile,
                    "executed_bytes": 16,
                },
                {
                    "direction": "recv", "peer": predecessor,
                    "communication_id": 7, "phase": "all_gather_ring",
                    "round": 0, "payload_slice": predecessor,
                    "executed_bytes": 16,
                },
            )
        )
    driver.require_cross_tile_message_matching(rows, "ring")
    driver.require_single_cycle({tile: (tile + 1) % 16 for tile in range(16)}, "ring")
    driver.require_ring(rows, "ring", 1, 16)

    tampered = {tile: {"messages": list(row["messages"])} for tile, row in rows.items()}
    tampered[0]["messages"] = [dict(message) for message in tampered[0]["messages"]]
    tampered[0]["messages"][0]["payload_slice"] = 9
    try:
        driver.require_cross_tile_message_matching(tampered, "tampered-ring")
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched ring message was accepted")

    tree = {tile: {"messages": []} for tile in range(16)}
    for child in range(15):
        parent = child + 1
        for tile, direction, peer, phase in (
            (child, "send", parent, "all_reduce_tree_reduce"),
            (parent, "recv", child, "all_reduce_tree_reduce"),
            (parent, "send", child, "all_reduce_tree_broadcast"),
            (child, "recv", parent, "all_reduce_tree_broadcast"),
        ):
            tree[tile]["messages"].append(
                {
                    "direction": direction, "peer": peer,
                    "communication_id": 9, "phase": phase,
                    "round": 15 - child, "payload_slice": child,
                    "executed_bytes": 256,
                }
            )
    driver.require_ordered_tree(tree, "tree", 256)


def main() -> int:
    catalog.validate_catalog()
    assert catalog.CASE_KEYS == EXPECTED_CASE_KEYS
    assert driver.CASES_BY_KEY is catalog.CASES_BY_KEY
    assert tuple(case.case_order for case in catalog.CASES) == tuple(range(9))
    assert {case.participant_count for case in catalog.CASES} == {16}
    assert {case.element_type for case in catalog.CASES} == {"f16"}
    assert {case.disposition for case in catalog.CASES} == {
        catalog.CharacterizationDisposition.SOURCE_COMPARISON_PENDING_LOWERING
    }
    assert {case.performance_evidence for case in catalog.CASES} == {
        catalog.PerformanceEvidenceKind.NOT_COLLECTED
    }
    assert all(case.same_source_required for case in catalog.CASES)
    assert not any(case.performance_is_promotion_evidence for case in catalog.CASES)

    for case in catalog.CASES:
        driver.validate_source(case)
        validate_payload(case)
    with tempfile.TemporaryDirectory() as directory:
        case = catalog.CASES_BY_KEY[EXPECTED_CASE_KEYS[0]]
        source = driver.write_source(pathlib.Path(directory), case)
        assert (source / "functions" / "forward.mlir").is_file()
        assert (source / "functions" / "forward.meta").is_file()
    validate_message_relations()
    print(f"collective_source_contract: cases={len(catalog.CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
