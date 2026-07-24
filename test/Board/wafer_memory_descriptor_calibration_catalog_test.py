#!/usr/bin/env python3
"""Validate bounded memory descriptor rows and host payload/oracle masks."""

from __future__ import annotations

import pathlib

import wafer_memory_descriptor_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 59
    assert len(catalog.DMA_CASES) == 14
    assert len(catalog.ENGINE_ACCESS_CASES) == 5
    assert len(catalog.RELATION_CASES) == 20
    assert len(catalog.PAIR_CASES) == 20
    assert [case.case_id for case in catalog.CATALOG] == list(range(59))
    assert len(catalog.CASES_BY_ID) == len(catalog.CATALOG)
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert {case.domain for case in catalog.CATALOG} == {
        "dma-ddr-descriptor",
        "spm-engine-access",
        "spm-address-relation",
        "spm-bank-engine-pair",
    }
    assert all(case.is_exact for case in catalog.DMA_CASES)
    assert all(case.is_exact for case in catalog.ENGINE_ACCESS_CASES)
    assert all(not case.is_exact for case in catalog.RELATION_CASES)
    assert all(case.is_exact for case in catalog.PAIR_CASES)
    assert {case.engine_a for case in catalog.ENGINE_ACCESS_CASES} == set(
        range(5)
    )
    assert {
        (case.engine_a, case.engine_b)
        for case in catalog.PAIR_CASES
    } == {
        (left, right)
        for left in range(5)
        for right in range(left + 1, 5)
    }
    assert {case.schedule for case in catalog.PAIR_CASES} == {
        catalog.SCHEDULE_SERIAL,
        catalog.SCHEDULE_WINDOW,
    }
    assert {case.effect for case in catalog.RELATION_CASES} == {
        catalog.EFFECT_RAW,
        catalog.EFFECT_WAR,
        catalog.EFFECT_WAW,
        catalog.EFFECT_RAR,
    }
    assert {case.relation for case in catalog.RELATION_CASES} == {
        catalog.RELATION_EXACT,
        catalog.RELATION_PARTIAL,
        catalog.RELATION_ADJACENT,
        catalog.RELATION_DISJOINT,
        catalog.RELATION_STRIDED,
    }

    for sample, case in enumerate(catalog.CATALOG):
        descriptor = case.descriptor
        assert descriptor.inner_bytes > 0
        assert descriptor.compact_bytes > 0
        assert descriptor.envelope_bytes >= descriptor.inner_bytes
        assert case.spm_a % 256 == 0
        assert sum(case.expected_counts) > 0
        built = catalog.build_case_payload(case, sample)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert len(built.expected_output) == catalog.RESOURCE_BYTES
        assert built.allowed_ranges
        for begin, end in built.allowed_ranges:
            assert 0 <= begin < end <= catalog.RESOURCE_BYTES
        for begin, end in built.exact_ranges:
            assert (begin, end) in built.allowed_ranges or any(
                outer_begin <= begin < end <= outer_end
                for outer_begin, outer_end in built.allowed_ranges
            )

    assert max(
        case.dst_ddr_offset + case.descriptor.envelope_bytes
        for case in catalog.DMA_CASES
    ) + catalog.OUTPUT_DATA_OFFSET <= catalog.RESOURCE_BYTES
    assert max(
        case.src_ddr_offset + case.descriptor.envelope_bytes
        for case in catalog.DMA_CASES
    ) + catalog.PAYLOAD_DATA_OFFSET <= catalog.RESOURCE_BYTES
    large = catalog.CASES_BY_NAME["ddr-large-contiguous-65536"]
    assert large.descriptor.compact_bytes == 65536
    assert {
        catalog.CASES_BY_NAME[name].descriptor.compact_bytes
        for name in (
            "ddr-large-1d-stride-holes",
            "ddr-large-2d-stride-holes",
            "ddr-large-3d-stride-holes",
        )
    } == {32768}
    assert catalog.CALIBRATION_LEAF_BINDINGS
    assert set().union(
        *map(set, catalog.CALIBRATION_LEAF_BINDINGS.values())
    ) == set(catalog.CATALOG) | set(catalog.STATIC_BOUNDARIES)
    assert all(
        boundary.disposition == "static-negative" and boundary.reason
        for boundary in catalog.STATIC_BOUNDARIES
    )

    probe = (
        pathlib.Path(__file__).parent
        / "Inputs"
        / "wafer_memory_descriptor_calibration_probe.c"
    ).read_text()
    assert "wafer_mdc_checked_descriptor" in probe
    assert "wafer_mdc_validate_kind" in probe
    assert "wafer_mdc_execute_dma" in probe
    assert "wafer_mdc_execute_engines" in probe
    assert "wafer_mdc_execute_relation" in probe
    assert "wafer_tx81_local_fence" in probe
    print(
        "wafer_memory_descriptor_calibration_catalog_test: "
        "cases=59 exact=39 observation=20 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
