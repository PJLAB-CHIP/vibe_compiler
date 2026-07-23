#!/usr/bin/env python3
"""Validate four independent cache/coherence visibility directions."""

from __future__ import annotations

import wafer_cache_coherence_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 4
    assert len(catalog.CASES_BY_ID) == 4
    assert len(catalog.CASES_BY_NAME) == 4
    assert {case.direction for case in catalog.CATALOG} == {
        "host-h2d->kcore",
        "kcore-store->rdma",
        "wdma->kcore",
        "wdma->host-d2h",
    }
    assert catalog.OUTPUT0_OFFSET % 64 == 0
    assert catalog.OUTPUT1_OFFSET % 64 == 0
    assert (
        catalog.OUTPUT1_OFFSET + catalog.PAYLOAD_BYTES
        + catalog.SPM_GUARD_BYTES
        <= catalog.RESOURCE_BYTES
    )
    total_phases = 0
    for case in catalog.CATALOG:
        assert case.phases in (1, 2)
        assert case.cache_control
        assert case.output_regions in (0, 1, 2)
        for sample in range(case.phases):
            built = catalog.build_case_payload(case, sample)
            assert len(built.request) == catalog.RESOURCE_BYTES
            assert len(built.payload) == catalog.RESOURCE_BYTES
            assert len(built.input_pattern) == catalog.PAYLOAD_BYTES
            assert len(built.post_control_expected) == catalog.PAYLOAD_BYTES
            if case.case_id == 1:
                assert built.input_pattern != built.post_control_expected
            else:
                assert built.input_pattern == built.post_control_expected
            total_phases += 1
    assert total_phases == 5
    print(
        "wafer_cache_coherence_calibration_catalog_test: "
        "directions=4 phases=5 payload_bytes=16384 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
