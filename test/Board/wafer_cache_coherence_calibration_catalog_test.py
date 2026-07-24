#!/usr/bin/env python3
"""Validate four independent cache/coherence visibility directions."""

from __future__ import annotations

import pathlib

import wafer_cache_coherence_calibration_catalog as catalog


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    protocol = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_cache_coherence_calibration_probe_protocol.h"
    ).read_text()
    probe = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_cache_coherence_calibration_probe.c"
    ).read_text()
    for name, value in (
        ("SCHEMA", catalog.SCHEMA),
        ("REQUEST_WORDS", catalog.REQUEST_WORDS),
        ("RECORD_WORDS", catalog.RECORD_WORDS),
        (
            "BANK_SEED_RDMA_INSTRUCTIONS",
            catalog.BANK_SEED_RDMA_INSTRUCTIONS,
        ),
        (
            "BANK_READBACK_WDMA_INSTRUCTIONS",
            catalog.BANK_READBACK_WDMA_INSTRUCTIONS,
        ),
    ):
        assert f"#define WAFER_CCH_{name} {value}U" in protocol
    for field, index in catalog.REQ.items():
        assert f"WAFER_CCH_REQ_{field} = {index}" in protocol
    for field, index in catalog.REC.items():
        assert f"WAFER_CCH_REC_{field} = {index}" in protocol
    assert "wafer_cch_seed_spm_bank" not in probe
    assert "wafer_cch_mismatch_spm_bank" not in probe
    pair_begin = probe.index("if (selected.pair_kind != 0U) {")
    pair_end = probe.index("} else {", pair_begin)
    pair_path = probe[pair_begin:pair_end]
    assert "get_spm_memory_mapping" not in pair_path
    assert pair_path.count("wafer_tx81_local_fence();") == 1
    assert (
        pair_path.index("wafer_cch_seed_bank_slots")
        < pair_path.index("wafer_cch_issue_bank_pair")
        < pair_path.index("wafer_cch_readback_bank_slots")
        < pair_path.index("wafer_tx81_local_fence();")
    )
    issue_begin = probe.index("static void wafer_cch_issue_bank_pair")
    issue_end = probe.index("static void wafer_cch_seed_bank_slots")
    assert (
        probe[issue_begin:issue_end].count("wafer_tx81_local_fence();")
        == 3
    )
    seed_begin = issue_end
    readback_begin = probe.index(
        "static void wafer_cch_readback_bank_slots"
    )
    entry_begin = probe.index(
        "__attribute__((visibility(\"hidden\")))", readback_begin
    )
    assert "wafer_tx81_local_fence();" not in probe[
        seed_begin:readback_begin
    ]
    assert "wafer_tx81_local_fence();" not in probe[
        readback_begin:entry_begin
    ]

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
        assert case.repetitions == (
            catalog.DDR_BANK_PMU_REPETITIONS
            if case.kind == "ddr-bank-pair"
            else 1
        )
        assert case.cache_control
        assert case.output_regions in (0, 1, 2)
        for sample in range(case.phases * case.repetitions):
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
                assert len(built.expected_regions) == 2
                assert case.output_regions == 2
                for offset, expected in built.expected_regions:
                    assert offset % 64 == 0
                    assert len(expected) == catalog.BANK_SLOT_BYTES
                    assert expected[: catalog.SPM_GUARD_BYTES] == bytes(
                        [catalog.SPM_GUARD_VALUE]
                    ) * catalog.SPM_GUARD_BYTES
                    assert expected[-catalog.SPM_GUARD_BYTES :] == bytes(
                        [catalog.SPM_GUARD_VALUE]
                    ) * catalog.SPM_GUARD_BYTES
                    assert (
                        offset + len(expected) <= catalog.RESOURCE_BYTES
                    )
    assert total_phases == (
        len(catalog.CACHE_CASES)
        + len(catalog.DDR_BANK_CASES)
        * catalog.DDR_BANK_PMU_REPETITIONS
    )
    assert {
        (case.pair_kind, case.schedule, case.bank_offset)
        for case in catalog.DDR_BANK_CASES
    } == {
        (pair, schedule, offset)
        for pair in ("rdma-rdma", "wdma-wdma", "rdma-wdma")
        for offset in catalog.BANK_OFFSETS
        for schedule in ("serial", "window")
    }
    assert {
        pair: {
            (case.expected_rdma, case.expected_wdma)
            for case in catalog.DDR_BANK_CASES
            if case.pair_kind == pair
        }
        for pair in ("rdma-rdma", "wdma-wdma", "rdma-wdma")
    } == {
        "rdma-rdma": {(4, 2)},
        "wdma-wdma": {(2, 4)},
        "rdma-wdma": {(3, 3)},
    }
    assert all(
        row.disposition == "delegated-board-case"
        and row.reason
        and row.evidence
        for row in catalog.DDR_LARGE_DESCRIPTOR_DISPOSITIONS
    )
    assert all(
        row.disposition == "isolated-deferred"
        and row.phases == 0
        and row.reason
        and row.evidence == (catalog.CACHE_CASES[0], catalog.CACHE_CASES[2])
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
        "directions=4 cache_phases=4 ddr_pair_cases=54 ddr_samples=162 "
        "offsets=9 deferred_session=1 delegated_large_descriptor=3 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
