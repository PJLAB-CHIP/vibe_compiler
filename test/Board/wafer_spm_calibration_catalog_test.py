#!/usr/bin/env python3
"""Validate SPM boundary and relative-offset calibration preparation."""

from __future__ import annotations

import wafer_spm_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 84
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert len(catalog.BOARD_CASES) == 19
    assert len(catalog.STATIC_NEGATIVE_CASES) == 15
    assert len(catalog.DEFERRED_CASES) == 0
    assert len(catalog.DELEGATED_CASES) == 50
    assert {case.domain for case in catalog.CATALOG} == {
        "capacity-reservation",
        "alignment-bank",
        "address-relation",
        "physical-layout",
        "lifetime-reuse",
        "engine-access",
        "bank-engine-pair",
    }
    assert {case.disposition for case in catalog.CATALOG}.issubset(
        catalog.DISPOSITIONS
    )
    for seed, case in enumerate(catalog.BOARD_CASES, start=1):
        assert catalog.ALLOCATABLE_BEGIN <= case.address
        assert case.address < case.end <= catalog.ALLOCATABLE_END
        payload = case.payload(seed)
        assert len(payload) == case.transfer_bytes * case.iterations
        assert payload != bytes(len(payload))
        assert len(set(payload[: min(len(payload), 256)])) > 16
        built = catalog.build_case_payload(case, seed)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert built.expected == case.payload(seed + 1)
        assert case.address % 256 == 0
        assert case.transfer_bytes % 256 == 0
        assert case.transfer_bytes * case.iterations <= catalog.SLOT_BYTES
    violations = {
        case.expected_violation for case in catalog.STATIC_NEGATIVE_CASES
    }
    assert {
        "below-allocatable-window",
        "crosses-reserved-window",
        "reserved-window",
        "outside-hardware-window",
        "zero-length",
        "address-overflow",
    }.issubset(violations)
    assert {
        "base-not-256-aligned",
        "length-not-256-aligned",
        "length-not-supported-alignment",
    }.issubset(violations)
    assert {case.address % 256 for case in catalog.ALIGNMENT_CASES} == {0}
    assert any(
        case.address % 1024 == 0 for case in catalog.ALIGNMENT_CASES
    )
    assert any(
        case.address % 1024 != 0 for case in catalog.ALIGNMENT_CASES
    )
    assert set(catalog.ALIGNMENT_OFFSETS) == {
        0,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        65792,
    }
    assert {case.effect for case in catalog.ADDRESS_RELATION_CASES} == {
        "RAW",
        "WAR",
        "WAW",
        "RAR",
    }
    assert {case.relation for case in catalog.ADDRESS_RELATION_CASES} == {
        "exact",
        "half-partial",
        "adjacent",
        "far-disjoint",
        "strided",
    }
    assert all(
        case.disposition == "DELEGATED_BOARD_CASE"
        and case.expected_violation is None
        and case.evidence
        for case in catalog.ADDRESS_RELATION_CASES
    )
    assert {case.iterations for case in catalog.LIFETIME_CASES} == {1, 4, 5}
    assert all(case.evidence for case in catalog.PHYSICAL_LAYOUT_CASES)
    assert all(case.evidence for case in catalog.DELEGATED_CASES)
    assert all(
        row in catalog.datamove.CATALOG
        or row in catalog.BOARD_CASES
        or row in catalog.memory_descriptor.CATALOG
        for case in catalog.DELEGATED_CASES
        for row in case.evidence
    )
    assert {case.effect for case in catalog.FIVE_ENGINE_ACCESS_CASES} == {
        "CT",
        "NE",
        "RDMA",
        "WDMA",
        "TDMA",
    }
    assert len(catalog.BANK_ENGINE_PAIR_CASES) == 20
    assert {case.relation for case in catalog.BANK_ENGINE_PAIR_CASES} == {
        "serial",
        "window",
    }
    assert catalog.CALIBRATION_LEAF_BINDINGS
    allowed_leaf_objects = set(catalog.CATALOG)
    for key, rows in catalog.CALIBRATION_LEAF_BINDINGS.items():
        assert key
        assert rows
        assert all(row in allowed_leaf_objects for row in rows)
    print(
        "wafer_spm_calibration_catalog_test: "
        f"cases={len(catalog.CATALOG)} "
        f"board={len(catalog.BOARD_CASES)} "
        f"negative={len(catalog.STATIC_NEGATIVE_CASES)} "
        f"deferred={len(catalog.DEFERRED_CASES)} "
        f"delegated={len(catalog.DELEGATED_CASES)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
