#!/usr/bin/env python3
"""Validate four independent cache/coherence visibility directions."""

from __future__ import annotations

import wafer_cache_coherence_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CACHE_CASES) == 4
    assert len(catalog.DDR_BANK_CASES) == 54
    assert len(catalog.CATALOG) == 58
    assert len(catalog.CASES_BY_ID) == 58
    assert len(catalog.CASES_BY_NAME) == 58
    assert len(catalog.CACHE_SESSION_DISPOSITIONS) == 1
    assert {case.direction for case in catalog.CACHE_CASES} == {
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
        assert case.phases == 1
        assert case.cache_control
        assert case.output_regions in (0, 1, 2)
        for sample in range(case.phases):
            built = catalog.build_case_payload(case, sample)
            assert len(built.request) == catalog.RESOURCE_BYTES
            assert len(built.payload) == catalog.RESOURCE_BYTES
            assert len(built.input_pattern) == case.payload_bytes
            assert len(built.post_control_expected) == case.payload_bytes
            if case.case_id == 1:
                assert built.input_pattern != built.post_control_expected
            else:
                assert built.input_pattern == built.post_control_expected
            total_phases += 1
            if case.kind == "ddr-bank-pair":
                assert case.pair_kind in {
                    "rdma-rdma",
                    "wdma-wdma",
                    "rdma-wdma",
                }
                assert case.schedule in {"serial", "window"}
                assert case.bank_offset in catalog.BANK_OFFSETS
                assert len(built.expected_regions) == case.output_regions
                for offset, expected in built.expected_regions:
                    assert offset % 256 == 0
                    assert len(expected) == catalog.BANK_PAYLOAD_BYTES
                    assert (
                        offset + len(expected) + catalog.SPM_GUARD_BYTES
                        <= catalog.RESOURCE_BYTES
                    )
    assert total_phases == 58
    assert {
        (case.pair_kind, case.schedule, case.bank_offset)
        for case in catalog.DDR_BANK_CASES
    } == {
        (pair, schedule, offset)
        for pair in ("rdma-rdma", "wdma-wdma", "rdma-wdma")
        for offset in catalog.BANK_OFFSETS
        for schedule in ("serial", "window")
    }
    assert all(
        row.disposition == "isolated-deferred" and row.reason
        for row in catalog.DDR_LARGE_DESCRIPTOR_DISPOSITIONS
    )
    assert all(
        row.disposition == "isolated-deferred"
        and row.phases == 0
        and row.reason
        for row in catalog.CACHE_SESSION_DISPOSITIONS
    )
    assert catalog.CALIBRATION_LEAF_BINDINGS
    allowed_leaf_objects = set(catalog.CATALOG)
    allowed_leaf_objects.update(catalog.DDR_LARGE_DESCRIPTOR_DISPOSITIONS)
    allowed_leaf_objects.update(catalog.CACHE_SESSION_DISPOSITIONS)
    for key, rows in catalog.CALIBRATION_LEAF_BINDINGS.items():
        assert key
        assert rows
        assert all(row in allowed_leaf_objects for row in rows)
    print(
        "wafer_cache_coherence_calibration_catalog_test: "
        "directions=4 cache_phases=4 ddr_pair_cases=54 "
        "offsets=9 deferred_session=1 deferred_large_descriptor=3 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
