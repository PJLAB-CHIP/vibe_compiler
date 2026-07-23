#!/usr/bin/env python3
"""Validate SPM boundary and relative-offset calibration preparation."""

from __future__ import annotations

import wafer_spm_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 22
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert len(catalog.BOARD_CASES) == 16
    assert len(catalog.STATIC_NEGATIVE_CASES) == 6
    assert {case.domain for case in catalog.CATALOG} == {
        "capacity-reservation",
        "alignment-bank",
    }
    assert {case.disposition for case in catalog.CATALOG} == (
        catalog.DISPOSITIONS
    )
    for seed, case in enumerate(catalog.BOARD_CASES, start=1):
        assert catalog.ALLOCATABLE_BEGIN <= case.address
        assert case.address < case.end <= catalog.ALLOCATABLE_END
        payload = case.payload(seed)
        assert len(payload) == case.transfer_bytes
        assert payload != bytes(len(payload))
        assert len(set(payload[: min(len(payload), 256)])) > 16
        built = catalog.build_case_payload(case, seed)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert built.expected == case.payload(seed + 1)
    violations = {
        case.expected_violation for case in catalog.STATIC_NEGATIVE_CASES
    }
    assert violations == {
        "below-allocatable-window",
        "crosses-reserved-window",
        "reserved-window",
        "outside-hardware-window",
        "zero-length",
        "address-overflow",
    }
    assert {case.address % 256 for case in catalog.ALIGNMENT_CASES} == {
        0,
        64,
        128,
        192,
    }
    assert any(
        case.address % 1024 == 0 for case in catalog.ALIGNMENT_CASES
    )
    assert any(
        case.address % 1024 != 0 for case in catalog.ALIGNMENT_CASES
    )
    print(
        "wafer_spm_calibration_catalog_test: "
        f"cases={len(catalog.CATALOG)} "
        f"board={len(catalog.BOARD_CASES)} "
        f"negative={len(catalog.STATIC_NEGATIVE_CASES)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
