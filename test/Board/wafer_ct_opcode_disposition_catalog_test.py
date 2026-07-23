#!/usr/bin/env python3
"""Require exactly one closed disposition for every CT opcode 0..186."""

from __future__ import annotations

from collections import Counter

import wafer_ct_opcode_disposition_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 187
    assert set(catalog.BY_OPCODE) == set(range(187))
    assert len(catalog.BY_OPCODE) == len(catalog.CATALOG)
    allowed = {
        "board-executable",
        "static-negative",
        "isolated-deferred",
    }
    for row in catalog.CATALOG:
        assert row.disposition in allowed
        if row.disposition == "board-executable":
            assert row.evidence
            assert row.reason is None
        elif row.disposition == "isolated-deferred":
            assert not row.evidence
            assert row.reason
        else:
            assert row.evidence
            assert row.reason
        assert row.opcode_name
        assert row.family

    counts = Counter(row.disposition for row in catalog.CATALOG)
    assert counts["board-executable"] == 173
    assert counts["static-negative"] == 2
    assert counts["isolated-deferred"] == 12
    assert {
        row.opcode
        for row in catalog.CATALOG
        if row.disposition == "isolated-deferred"
    } == {
        115,
        116,
        119,
        120,
        122,
        136,
        137,
        180,
        182,
        184,
        185,
        186,
    }
    assert catalog.BY_OPCODE[139].evidence == (
        "ct-op139-convert-int8-fp16-main",
        "ct-op139-convert-int8-fp16-tail",
    )
    print(
        "wafer_ct_opcode_disposition_catalog_test: "
        f"opcodes=187 dispositions={dict(sorted(counts.items()))} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
