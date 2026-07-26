#!/usr/bin/env python3
"""Validate four independent cache/coherence visibility directions."""

from __future__ import annotations

import pathlib
import struct
import types

import wafer_cache_coherence_calibration_catalog as catalog
import wafer_board_cache_coherence_calibration_probe_test as host


def _valid_conflict_raw(
    case: catalog.CacheCoherenceCase,
    built: catalog.CasePayload,
) -> bytearray:
    raw = bytearray([catalog.OUTPUT_CANARY]) * catalog.RESOURCE_BYTES
    words = [0] * catalog.RECORD_WORDS
    expected_record = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "SAMPLE": 0,
        "PAYLOAD_BYTES": case.transfer_bytes,
        "EXPECTED_RDMA": case.expected_rdma,
        "EXPECTED_WDMA": case.expected_wdma,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT0_OFFSET": catalog.OUTPUT0_OFFSET,
        "OUTPUT1_OFFSET": catalog.OUTPUT1_OFFSET,
        "MISMATCH_BEFORE": 0,
        "MISMATCH_AFTER": 0,
        "SPM_GUARD_MISMATCHES": 0,
        "OUTPUT_GUARD_MISMATCHES": 0,
        "RDMA_INST_DELTA": case.expected_rdma,
        "WDMA_INST_DELTA": case.expected_wdma,
        "CACHE_CONTROL_MASK": 8,
        "RDMA_EXEC_DELTA": 100,
        "WDMA_EXEC_DELTA": 100,
        "PAIR_KIND": 1,
        "SCHEDULE": 0,
        "BANK_OFFSET": case.bank_offset,
        "MODE": catalog.CONFLICT_MODE,
        "TRANSFER_BYTES": case.transfer_bytes,
        "BASE_RELATION": case.base_relation,
        "BASE_TRANSLATION": case.base_translation,
        "DDR_BASE_A": 0x100000,
        "DDR_BASE_B": 0x100000,
        "DDR_ADDRESS_A": (
            0x100000
            + catalog.CONFLICT_DATA_BASE
            + case.base_translation
        ),
        "DDR_ADDRESS_B": (
            0x100000
            + catalog.CONFLICT_DATA_BASE
            + case.base_translation
            + catalog.BANK_REGION_GAP
            + case.bank_offset
        ),
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    for field, value in expected_record.items():
        words[catalog.REC[field]] = value
    struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *words)

    row_ordinal = 0
    for repetition in range(catalog.CONFLICT_REPETITIONS):
        for schedule in (1, 2):
            for issue_order in (0, 1):
                row = [0] * catalog.CONFLICT_ROW_WORDS
                control = (schedule - 1) * 2 + issue_order
                archive0 = (
                    catalog.CONFLICT_ARCHIVE_BASE
                    + control * catalog.CONFLICT_ARCHIVE_STRIDE
                )
                expected_row = {
                    "MAGIC": catalog.CONFLICT_ROW_MAGIC,
                    "IDENTITY": (
                        repetition
                        | schedule << 8
                        | issue_order << 16
                    ),
                    "REPETITION": repetition,
                    "SCHEDULE": schedule,
                    "ISSUE_ORDER": issue_order,
                    "RDMA_INST_DELTA": 2,
                    "WDMA_INST_DELTA": 0,
                    "RDMA_EXEC_DELTA": 10,
                    "WDMA_EXEC_DELTA": 0,
                    "PLAN_CYCLES": 20,
                    "ARCHIVE0_OFFSET": archive0,
                    "ARCHIVE1_OFFSET": (
                        archive0
                        + case.transfer_bytes
                        + 2 * catalog.SPM_GUARD_BYTES
                    ),
                    "DDR_ADDRESS_A": words[
                        catalog.REC["DDR_ADDRESS_A"]
                    ],
                    "DDR_ADDRESS_B": words[
                        catalog.REC["DDR_ADDRESS_B"]
                    ],
                    "COMPLETED": 1,
                    "GUARD": catalog.CONFLICT_ROW_GUARD,
                    "RESULT_MISMATCHES": 0,
                    "SCHEDULE_POSITION": (
                        schedule - 1
                        if (repetition + issue_order) % 2 == 0
                        else 2 - schedule
                    ),
                }
                for field, value in expected_row.items():
                    row[catalog.CONFLICT_ROW[field]] = value
                begin = (
                    catalog.RECORD_WORDS
                    + row_ordinal * catalog.CONFLICT_ROW_WORDS
                ) * 8
                struct.pack_into(
                    f"<{catalog.CONFLICT_ROW_WORDS}Q",
                    raw,
                    begin,
                    *row,
                )
                row_ordinal += 1
    for begin, expected in built.expected_regions:
        raw[begin : begin + len(expected)] = expected
    return raw


def _assert_conflict_host_oracle_rejects(
    raw: bytearray,
    case: catalog.CacheCoherenceCase,
    built: catalog.CasePayload,
) -> None:
    try:
        host._validate_conflict_output(bytes(raw), case, built, 0)
    except RuntimeError:
        return
    raise AssertionError("conflict host oracle accepted corrupted output")


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
        ("CONFLICT_MODE", catalog.CONFLICT_MODE),
        (
            "CONFLICT_FIRST_CASE_ID",
            catalog.CONFLICT_FIRST_CASE_ID,
        ),
        (
            "CONFLICT_REPETITIONS",
            catalog.CONFLICT_REPETITIONS,
        ),
        (
            "CONFLICT_DATA_BASE",
            catalog.CONFLICT_DATA_BASE,
        ),
        (
            "CONFLICT_ARCHIVE_BASE",
            catalog.CONFLICT_ARCHIVE_BASE,
        ),
        (
            "CONFLICT_ARCHIVE_STRIDE",
            catalog.CONFLICT_ARCHIVE_STRIDE,
        ),
        (
            "CONFLICT_ROW_WORDS",
            catalog.CONFLICT_ROW_WORDS,
        ),
    ):
        assert f"#define WAFER_CCH_{name} {value}U" in protocol
    for field, index in catalog.REQ.items():
        assert f"WAFER_CCH_REQ_{field} = {index}" in protocol
    for field, index in catalog.REC.items():
        assert f"WAFER_CCH_REC_{field} = {index}" in protocol
    for field, index in catalog.CONFLICT_ROW.items():
        suffix = "_WORD" if field in {"MAGIC", "GUARD"} else ""
        assert (
            f"WAFER_CCH_CONFLICT_ROW_{field}{suffix} = {index}"
            in protocol
        )
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
    conflict_cycle_begin = probe.index(
        "static uint64_t wafer_cch_cycle", readback_begin
    )
    assert "wafer_tx81_local_fence();" not in probe[
        seed_begin:readback_begin
    ]
    assert "wafer_tx81_local_fence();" not in probe[
        readback_begin:conflict_cycle_begin
    ]

    conflict_issue_begin = probe.index(
        "static void wafer_cch_issue_conflict_pair"
    )
    conflict_issue_end = probe.index(
        "static uint32_t wafer_cch_conflict_identity",
        conflict_issue_begin,
    )
    conflict_issue = probe[conflict_issue_begin:conflict_issue_end]
    assert conflict_issue.count("wafer_tx81_local_fence();") == 2
    assert (
        "if (schedule == 1U && ordinal == 0U)\n"
        "      wafer_tx81_local_fence();"
    ) in conflict_issue

    conflict_execute_begin = probe.index(
        "static uint32_t wafer_cch_execute_conflict"
    )
    conflict_execute_end = probe.index(
        "__attribute__((visibility(\"hidden\")))",
        conflict_execute_begin,
    )
    conflict_execute = probe[
        conflict_execute_begin:conflict_execute_end
    ]
    seed_fence = conflict_execute.index(
        "wafer_tx81_local_fence();",
        conflict_execute.index("seed_b - WAFER_CCH_SPM_GUARD_BYTES"),
    )
    pair_before = conflict_execute.index(
        "WaferCCHPMU before = wafer_cch_read_pmu();"
    )
    pair_issue = conflict_execute.index(
        "wafer_cch_issue_conflict_pair", pair_before
    )
    pair_after = conflict_execute.index(
        "WaferCCHPMU after = wafer_cch_read_pmu();", pair_issue
    )
    readback = conflict_execute.index(
        "if (selected->pair_kind == 1U)", pair_after
    )
    assert seed_fence < pair_before < pair_issue < pair_after < readback

    assert len(catalog.CACHE_CASES) == 4
    assert len(catalog.DDR_BANK_CASES) == 54
    assert len(catalog.CATALOG) == 58
    assert len(catalog.CASES_BY_ID) == 58
    assert len(catalog.CASES_BY_NAME) == 58
    assert len(catalog.PENDING_DDR_CONFLICT_CASES) == 108
    assert len(catalog.ALL_CASES) == 166
    assert len(catalog.ALL_CASES_BY_ID) == 166
    assert len(catalog.ALL_CASES_BY_NAME) == 166
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
    expected_conflict_axes = {
        (
            pair_kind,
            transfer,
            relation,
            translation,
            offset,
        )
        for pair_kind in ("rdma-rdma", "wdma-wdma")
        for transfer in catalog.CONFLICT_TRANSFER_BYTES
        for relation in (
            (
                catalog.CONFLICT_BASE_RELATION_SAME,
                catalog.CONFLICT_BASE_RELATION_CROSS,
            )
            if pair_kind == "rdma-rdma"
            else (catalog.CONFLICT_BASE_RELATION_SAME,)
        )
        for translation in catalog.CONFLICT_BASE_TRANSLATIONS
        for offset in catalog.BANK_OFFSETS
    }
    assert {
        (
            case.pair_kind,
            case.transfer_bytes,
            case.base_relation,
            case.base_translation,
            case.bank_offset,
        )
        for case in catalog.PENDING_DDR_CONFLICT_CASES
    } == expected_conflict_axes
    for case in catalog.PENDING_DDR_CONFLICT_CASES:
        assert case.disposition == "pending-board-executable"
        assert case.repetitions == 1
        assert case.batch_repetitions == catalog.CONFLICT_REPETITIONS
        assert case.schedule is None
        built = catalog.build_case_payload(case, 0)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        request_words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert request_words[catalog.REQ["MODE"]] == catalog.CONFLICT_MODE
        assert (
            request_words[catalog.REQ["TRANSFER_BYTES"]]
            == case.transfer_bytes
        )
        assert (
            request_words[catalog.REQ["BASE_RELATION"]]
            == case.base_relation
        )
        assert (
            request_words[catalog.REQ["BASE_TRANSLATION"]]
            == case.base_translation
        )
        assert (
            request_words[catalog.REQ["BATCH_ROWS"]]
            == catalog.CONFLICT_BATCH_ROWS
        )
        assert (
            request_words[catalog.REQ["ARCHIVE_STRIDE"]]
            == catalog.CONFLICT_ARCHIVE_STRIDE
        )
        assert len(built.expected_regions) == (
            8 if case.pair_kind == "rdma-rdma" else 2
        )
        assert all(
            len(expected)
            == case.transfer_bytes + 2 * catalog.SPM_GUARD_BYTES
            for _, expected in built.expected_regions
        )
        for offset, expected in built.expected_regions:
            assert 0 <= offset < catalog.RESOURCE_BYTES
            assert offset + len(expected) <= catalog.RESOURCE_BYTES
            assert expected[: catalog.SPM_GUARD_BYTES] == bytes(
                [catalog.SPM_GUARD_VALUE]
            ) * catalog.SPM_GUARD_BYTES
            assert expected[-catalog.SPM_GUARD_BYTES :] == bytes(
                [catalog.SPM_GUARD_VALUE]
            ) * catalog.SPM_GUARD_BYTES
        assert (
            case.expected_rdma,
            case.expected_wdma,
        ) == catalog._conflict_expected_counts(case.pair_kind)
    oracle_case = next(
        case
        for case in catalog.PENDING_DDR_CONFLICT_CASES
        if case.pair_kind == "rdma-rdma"
        and case.base_relation == catalog.CONFLICT_BASE_RELATION_SAME
        and case.transfer_bytes == catalog.CONFLICT_TRANSFER_BYTES[0]
        and case.base_translation == 0
        and case.bank_offset == 0
    )
    oracle_built = catalog.build_case_payload(oracle_case, 0)
    valid_raw = _valid_conflict_raw(oracle_case, oracle_built)
    host._validate_conflict_output(
        bytes(valid_raw), oracle_case, oracle_built, 0
    )
    first_row = catalog.RECORD_WORDS * 8
    mismatch_raw = bytearray(valid_raw)
    struct.pack_into(
        "<Q",
        mismatch_raw,
        first_row
        + catalog.CONFLICT_ROW["RESULT_MISMATCHES"] * 8,
        1,
    )
    _assert_conflict_host_oracle_rejects(
        mismatch_raw, oracle_case, oracle_built
    )
    position_raw = bytearray(valid_raw)
    struct.pack_into(
        "<Q",
        position_raw,
        first_row
        + catalog.CONFLICT_ROW["SCHEDULE_POSITION"] * 8,
        1,
    )
    _assert_conflict_host_oracle_rejects(
        position_raw, oracle_case, oracle_built
    )
    archive_raw = bytearray(valid_raw)
    archive_raw[oracle_built.expected_regions[0][0]] ^= 0xFF
    _assert_conflict_host_oracle_rejects(
        archive_raw, oracle_case, oracle_built
    )
    assert len(catalog.DDR_CONFLICT_DISPOSITIONS) == 1
    assert host.select_cases(
        types.SimpleNamespace(
            ddr_conflict_equivalence=False,
            selected_cases=None,
        )
    ) == catalog.CATALOG
    assert host.select_cases(
        types.SimpleNamespace(
            ddr_conflict_equivalence=True,
            selected_cases=None,
        )
    ) == catalog.PENDING_DDR_CONFLICT_CASES
    try:
        host.select_cases(
            types.SimpleNamespace(
                ddr_conflict_equivalence=False,
                selected_cases=[
                    catalog.PENDING_DDR_CONFLICT_CASES[0].name
                ],
            )
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "pending DDR conflict case bypassed its explicit selector"
        )
    assert (
        catalog.DDR_CONFLICT_DISPOSITIONS[0].disposition
        == "isolated-deferred"
    )
    assert catalog.DDR_CONFLICT_DISPOSITIONS[0].reason
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
        "offsets=9 ddr_conflict_batches=108 "
        "ddr_conflict_cells=1728 deferred_wdma_cross_allocation=1 "
        "deferred_session=1 delegated_large_descriptor=3 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
