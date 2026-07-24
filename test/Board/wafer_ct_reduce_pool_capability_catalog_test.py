#!/usr/bin/env python3
"""Validate typed, non-opcode-wide capability accounting for CT 111..123."""

from __future__ import annotations

from collections import Counter

import wafer_ct_reduce_pool_capability_catalog as catalog
import wafer_instruction_family_catalog as instruction_catalog


def main() -> int:
    assert catalog.CATALOG
    assert len(catalog.BY_KEY) == len(catalog.CATALOG)
    assert set(catalog.ROWS_BY_OPCODE) == set(range(111, 124))
    assert all(catalog.ROWS_BY_OPCODE.values())
    assert all(
        row.key
        == (row.opcode, row.dtype, row.axis_layout, row.geometry)
        for row in catalog.CATALOG
    )

    allowed = {
        "board-executable",
        "board-observation",
        "static-negative",
        "isolated-deferred",
    }
    evidence_names = {
        case.name for case in instruction_catalog.CATALOG
    }
    for row in catalog.CATALOG:
        assert row.disposition in allowed
        if row.disposition in {"board-executable", "board-observation"}:
            assert row.evidence
            assert row.oracle
            assert all(item in evidence_names for item in row.evidence)
        else:
            assert not row.evidence
            assert row.oracle is None
            assert row.reason

    reduce_rows = tuple(
        row for row in catalog.CATALOG if row.family == "reduce"
    )
    for opcode in range(111, 115):
        for dtype in catalog.DTYPES:
            rows = tuple(
                row
                for row in reduce_rows
                if row.opcode == opcode
                and row.dtype == dtype
                and row.axis_layout in catalog._REDUCE_AXIS_GEOMETRIES
            )
            assert {row.axis_layout for row in rows} == {
                "C/NCx",
                "W/NCx",
                "H/NCx",
                "HW/NCx",
            }
    qualified_reduce = {
        (row.opcode, row.dtype, row.axis_layout)
        for row in reduce_rows
        if row.disposition == "board-executable"
    }
    assert qualified_reduce == {
        (opcode, dtype, axis_layout)
        for opcode in range(111, 115)
        for dtype in catalog.DTYPES
        for axis_layout in catalog._REDUCE_AXIS_GEOMETRIES
    } | {
        (opcode, "f16", "axis-specific/Cx")
        for opcode in range(111, 115)
    }
    assert not any(
        row.disposition == "unknown" for row in reduce_rows
    )
    raw_reduce = tuple(
        row
        for row in reduce_rows
        if row.axis_layout in {"N/NCx", "HWC/NCx"}
    )
    assert len(raw_reduce) == 8
    assert all(
        row.dtype == "f16"
        and row.disposition == "isolated-deferred"
        and row.oracle is None
        and not row.evidence
        and row.qualification == "isolated-timeout"
        and "may permanently wait" in (row.reason or "")
        for row in raw_reduce
    )
    existing_board_passed = {
        (111, "f16", "W/NCx"),
        (112, "bf16", "W/NCx"),
        (113, "f16", "W/NCx"),
        (114, "bf16", "W/NCx"),
    }
    assert {
        (row.opcode, row.dtype, row.axis_layout)
        for row in reduce_rows
        if row.qualification == "board-passed"
    } == existing_board_passed

    independent_indexed_max = catalog.BY_KEY[
        (
            118,
            "f16",
            "HW-window/NCx",
            "k3x2-s2x1-p0-n1h3w5c64",
        )
    ]
    assert independent_indexed_max.disposition == "board-executable"
    assert independent_indexed_max.qualification == "pending-board"
    assert independent_indexed_max.evidence == (
        "pool-indexed-max-f16-k3x2-s2x1",
    )
    composite_only_indexed_max = catalog.BY_KEY[
        (
            118,
            "f16",
            "HW-window/NCx",
            "k2x2-s2x2-p0-n1h2w4c64",
        )
    ]
    assert composite_only_indexed_max.disposition == "board-observation"
    assert "predecessor" in (composite_only_indexed_max.reason or "")

    pool_rows = tuple(
        row for row in catalog.CATALOG if row.family == "pool"
    )
    assert not any(row.disposition == "unknown" for row in pool_rows)
    assert sum(
        row.disposition == "board-executable"
        and row.qualification == "pending-board"
        for row in pool_rows
    ) == 17
    assert {
        row.geometry
        for row in pool_rows
        if row.disposition == "board-observation"
        and row.qualification == "pending-board"
    } == {
        "k3x2-s2x1-asymmetric-pad",
        "indexed-tie",
    }

    opcode_121_rows = catalog.ROWS_BY_OPCODE[121]
    assert not any(
        row.disposition == "board-executable" for row in opcode_121_rows
    )
    assert any(
        row.disposition == "board-observation" for row in opcode_121_rows
    )
    opcode_123_evidence = {
        evidence
        for row in catalog.ROWS_BY_OPCODE[123]
        for evidence in row.evidence
    }
    assert opcode_123_evidence == {
        "unpool-f16",
        "unpool-mask-bf16",
        "unpool-mask-f32",
        "unpool-mask-f16-k3x2-s2x1",
        "unpool-mask-f16-repeated-overlap-observed",
    }
    assert all("tdma-" not in item for item in opcode_123_evidence)

    unpool_rows = tuple(
        row for row in catalog.CATALOG if row.family == "unpool"
    )
    assert {
        row.evidence
        for row in unpool_rows
        if row.disposition == "board-observation"
        and row.qualification == "board-observed"
    } == {
        ("unpool-index-f16",),
        ("unpool-mask-f32",),
        ("unpool-mask-f16-k3x2-s2x1",),
    }
    assert sum(
        row.disposition == "board-observation"
        and row.qualification == "pending-board"
        for row in unpool_rows
    ) == 6

    for opcode in range(115, 124):
        assert any(
            row.disposition == "static-negative"
            and "/Cx" in row.axis_layout
            for row in catalog.ROWS_BY_OPCODE[opcode]
        )

    counts = Counter(row.disposition for row in catalog.CATALOG)
    assert counts == {
        "board-executable": 80,
        "board-observation": 18,
        "static-negative": 11,
        "isolated-deferred": 8,
    }
    assert catalog.ISOLATED_DEFERRED_ROWS == raw_reduce
    assert not catalog.UNKNOWN_ROWS
    print(
        "wafer_ct_reduce_pool_capability_catalog_test: "
        f"typed_rows={len(catalog.CATALOG)} "
        f"dispositions={dict(sorted(counts.items()))} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
