#!/usr/bin/env python3
"""Validate bounded memory descriptor rows and host payload/oracle masks."""

from __future__ import annotations

import pathlib
import types

import wafer_board_memory_descriptor_calibration_probe_test as runner
import wafer_memory_descriptor_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 99
    assert len(catalog.DMA_CASES) == 14
    assert len(catalog.ENGINE_ACCESS_CASES) == 5
    assert len(catalog.RELATION_CASES) == 20
    assert len(catalog.PAIR_CASES) == 60
    assert [case.case_id for case in catalog.CATALOG] == list(range(99))
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
    new_offsets = runner.select_cases(
        types.SimpleNamespace(
            selected_cases=None,
            domain=["spm-bank-engine-pair"],
            oracle=None,
            relative_spm_offset=[4352, 65536],
        )
    )
    assert len(new_offsets) == 40
    assert {
        abs(case.spm_b - case.spm_a) for case in new_offsets
    } == {4352, 65536}
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
    assert {
        case.spm_b - case.spm_a for case in catalog.PAIR_CASES
    } == set(catalog.SPM_PAIR_RELATIVE_OFFSETS) == {4352, 8192, 65536}
    assert all(
        sum(
            candidate.engine_a == left
            and candidate.engine_b == right
            and candidate.spm_b - candidate.spm_a == relative_offset
            for candidate in catalog.PAIR_CASES
        )
        == 2
        for left in range(5)
        for right in range(left + 1, 5)
        for relative_offset in catalog.SPM_PAIR_RELATIVE_OFFSETS
    )
    assert all(
        case.spm_b - case.spm_a
        >= catalog.SPM_PAIR_SLOT_BYTES + 2 * catalog.SPM_GUARD_BYTES
        for case in catalog.PAIR_CASES
    )
    assert {
        case.name
        for case in catalog.PAIR_CASES
        if case.spm_b - case.spm_a == 8192
    } == {
        f"spm-bank-pair-{catalog.ENGINE_NAMES[left].lower()}-"
        f"{catalog.ENGINE_NAMES[right].lower()}-{schedule}"
        for left in range(5)
        for right in range(left + 1, 5)
        for schedule in ("serial", "window")
    }
    assert catalog.SPM_BANK_PMU_REPETITIONS >= 3
    assert all(
        case.repetitions == catalog.SPM_BANK_PMU_REPETITIONS
        for case in catalog.PAIR_CASES
    )
    assert all(
        case.repetitions == 1
        for case in catalog.CATALOG
        if case not in catalog.PAIR_CASES
    )
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
        "cases=99 exact=79 observation=20 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
