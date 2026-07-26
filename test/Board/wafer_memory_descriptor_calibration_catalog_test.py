#!/usr/bin/env python3
"""Validate bounded memory descriptor rows and host payload/oracle masks."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import pathlib
import struct
import tempfile
import types

import wafer_board_memory_descriptor_calibration_probe_test as runner
import wafer_memory_descriptor_calibration_catalog as catalog


def main() -> int:
    assert catalog.SCHEMA == 3
    assert catalog.REQUEST_WORDS == 40
    assert catalog.RECORD_WORDS == 64
    assert catalog.RESOURCE_BYTES == 2 * 1024 * 1024
    assert len(catalog.CATALOG) == 155
    assert len(catalog.DMA_CASES) == 14
    assert len(catalog.ENGINE_ACCESS_CASES) == 5
    assert len(catalog.RELATION_CASES) == 20
    assert len(catalog.GENERAL_PAIR_CASES) == 60
    assert len(catalog.BANK_PERIOD_PAIR_CASES) == 16
    assert len(catalog.ALIGNMENT_PAIR_CASES) == 6
    assert len(catalog.PAIR_CASES) == 82
    assert len(catalog.PARALLEL_ADDRESS_SWEEP_CASES) == 26
    assert len(catalog.SUSTAINED_PARALLEL_PAIR_CASES) == 20
    assert len(catalog.CROSS_WORKER_PARALLEL_PAIR_CASES) == 2
    assert len(catalog.DEPENDENCY_PARALLEL_PAIR_CASES) == 12
    assert len(catalog.PARALLEL_PAIR_CASES) == 34
    assert len(catalog.PENDING_CONFLICT_EQUIVALENCE_CASES) == 88
    assert len(catalog.CONFLICT_EQUIVALENCE_PAIRS) == 48
    assert catalog.CONFLICT_SAMPLES == 4
    assert [case.case_id for case in catalog.CATALOG] == list(range(155))
    assert [case.case_id for case in catalog.PENDING_CASES] == list(
        range(155, 243)
    )
    assert len(catalog.CASES_BY_ID) == len(catalog.CATALOG)
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert len(catalog.ALL_CASES_BY_ID) == len(catalog.ALL_CASES) == 243
    assert len(catalog.ALL_CASES_BY_NAME) == len(catalog.ALL_CASES)
    assert {case.domain for case in catalog.CATALOG} == {
        "dma-ddr-descriptor",
        "spm-engine-access",
        "spm-address-relation",
        "spm-bank-engine-pair",
        "multi-engine-parallel-window",
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
    equivalence = runner.select_cases(
        types.SimpleNamespace(
            selected_cases=None,
            domain=None,
            oracle=None,
            relative_spm_offset=None,
            parallel_address_sweep=False,
            parallel_pair_expanded=False,
            conflict_equivalence=True,
        )
    )
    assert equivalence == catalog.CONFLICT_EQUIVALENCE_CASES
    try:
        runner.select_cases(
            types.SimpleNamespace(
                selected_cases=[
                    catalog.PENDING_CONFLICT_EQUIVALENCE_CASES[0].name
                ],
                domain=None,
                oracle=None,
                relative_spm_offset=None,
                parallel_address_sweep=False,
                parallel_pair_expanded=False,
                conflict_equivalence=False,
            )
        )
    except RuntimeError as error:
        assert "unknown memory descriptor cases" in str(error)
    else:
        raise AssertionError(
            "pending conflict case bypassed the explicit selector"
        )
    assert len(catalog.CONFLICT_EQUIVALENCE_BASELINE_CASES) == 8
    held_out = catalog.PENDING_CONFLICT_EQUIVALENCE_CASES
    assert {
        case.spm_a - catalog.SPM_BASE - case.spm_a % 256
        for case in held_out
    } == {0, *catalog.SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS}
    assert {
        case.spm_a % 256 for case in held_out
    } == set(catalog.SPM_PARALLEL_ALIGNMENT_PHASES)
    assert {
        case.schedule for case in held_out
    } == {catalog.SCHEDULE_SERIAL, catalog.SCHEDULE_WINDOW}
    assert {
        case.descriptor.compact_bytes for case in held_out
    } == set(catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS)
    assert {
        case.issue_order for case in held_out
    } == set(catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS)
    assert {
        (
            case.spm_a
            - catalog.SPM_BASE
            - case.spm_a % 256,
            case.spm_a % 256,
            case.descriptor.compact_bytes,
            case.issue_order,
            case.schedule,
        )
        for case in catalog.CONFLICT_EQUIVALENCE_CASES
    } == {
        (
            translation,
            phase,
            transfer,
            issue_order,
            schedule,
        )
        for translation in (
            0,
            *catalog.SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS,
        )
        for phase in catalog.SPM_PARALLEL_ALIGNMENT_PHASES
        for transfer in catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS
        for issue_order in catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
        for schedule in (
            catalog.SCHEDULE_SERIAL,
            catalog.SCHEDULE_WINDOW,
        )
    }
    assert {
        (
            pair.translation,
            pair.phase,
            pair.transfer_bytes,
            pair.issue_order,
        )
        for pair in catalog.CONFLICT_EQUIVALENCE_PAIRS
    } == {
        (translation, phase, transfer, issue_order)
        for translation in (
            0,
            *catalog.SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS,
        )
        for phase in catalog.SPM_PARALLEL_ALIGNMENT_PHASES
        for transfer in catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS
        for issue_order in catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
    }
    for pair in catalog.CONFLICT_EQUIVALENCE_PAIRS:
        first_positions = []
        for sample in range(catalog.CONFLICT_SAMPLES):
            invocation = catalog.build_conflict_equivalence_invocation(
                pair, sample
            )
            assert len(invocation.request) == catalog.RESOURCE_BYTES
            assert len(invocation.payload) == catalog.RESOURCE_BYTES
            assert (
                invocation.serial_built.payload
                == invocation.window_built.payload
                == invocation.payload
            )
            serial_words = struct.unpack_from(
                f"<{catalog.REQUEST_WORDS}Q",
                invocation.request,
                catalog.CONFLICT_SERIAL_REQUEST_WORD * 8,
            )
            window_words = struct.unpack_from(
                f"<{catalog.REQUEST_WORDS}Q",
                invocation.request,
                catalog.CONFLICT_WINDOW_REQUEST_WORD * 8,
            )
            assert {
                index
                for index, (serial, window) in enumerate(
                    zip(serial_words, window_words, strict=True)
                )
                if serial != window
            } == {catalog.REQ["CASE"], catalog.REQ["SCHEDULE"]}
            meta = struct.unpack_from(
                f"<{catalog.CONFLICT_REQUEST_META_WORDS}Q",
                invocation.request,
                catalog.CONFLICT_REQUEST_META_WORD * 8,
            )
            assert (
                meta[catalog.CONFLICT_REQ["MAGIC"]]
                == catalog.CONFLICT_REQUEST_MAGIC
            )
            assert meta[catalog.CONFLICT_REQ["COORDINATE"]] == (
                pair.coordinate_id
            )
            assert meta[catalog.CONFLICT_REQ["SAMPLE"]] == sample
            assert meta[catalog.CONFLICT_REQ["FIRST_SCHEDULE"]] == (
                invocation.first_schedule
            )
            first_positions.append(invocation.first_schedule)
        assert first_positions.count(catalog.SCHEDULE_SERIAL) == 2
        assert first_positions.count(catalog.SCHEDULE_WINDOW) == 2
    assert all(
        case.spm_b - case.spm_a == 8192
        and case.sweep == "conflict-equivalence"
        and case not in catalog.CATALOG
        for case in held_out
    )
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
    } == set(catalog.SPM_PAIR_RELATIVE_OFFSETS)
    assert set(catalog.SPM_BANK_PERIOD_RELATIVE_OFFSETS) == {
        4096 + index * 256 for index in range(9)
    }
    assert all(
        sum(
            candidate.engine_a == left
            and candidate.engine_b == right
            and candidate.spm_b - candidate.spm_a == relative_offset
            for candidate in catalog.GENERAL_PAIR_CASES
        )
        == 2
        for left in range(5)
        for right in range(left + 1, 5)
        for relative_offset in catalog.SPM_GENERAL_PAIR_RELATIVE_OFFSETS
    )
    assert {
        (case.engine_a, case.engine_b)
        for case in catalog.BANK_PERIOD_PAIR_CASES
        + catalog.ALIGNMENT_PAIR_CASES
    } == {catalog.SPM_PARALLEL_ENGINES}
    assert {
        case.spm_b - case.spm_a
        for case in catalog.PARALLEL_ADDRESS_SWEEP_CASES
        if case.spm_a % 256 == 0
        and case.spm_b - case.spm_a <= 6144
    } == set(catalog.SPM_BANK_PERIOD_RELATIVE_OFFSETS)
    assert {
        case.spm_a % 256
        for case in catalog.PARALLEL_ADDRESS_SWEEP_CASES
        if case.spm_b - case.spm_a == 8192
    } == set(catalog.SPM_PARALLEL_ALIGNMENT_PHASES)
    assert all(
        case.spm_b - case.spm_a
        >= catalog.SPM_PAIR_SLOT_BYTES + 2 * catalog.SPM_GUARD_BYTES
        for case in catalog.PAIR_CASES
    )
    assert {
        case.name
        for case in catalog.GENERAL_PAIR_CASES
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
        for case in catalog.PAIR_CASES + catalog.PARALLEL_PAIR_CASES
    )
    assert all(
        case.repetitions == 1
        for case in catalog.CATALOG
        if case
        not in catalog.PAIR_CASES + catalog.PARALLEL_PAIR_CASES
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
    assert all(
        case.kind == catalog.KIND_PARALLEL_PAIR and case.is_exact
        for case in catalog.PARALLEL_PAIR_CASES
    )
    assert {
        (case.engine_a, case.engine_b)
        for case in catalog.SUSTAINED_PARALLEL_PAIR_CASES
    } == {
        (left, right)
        for left in range(5)
        for right in range(left + 1, 5)
    }
    assert all(
        case.rounds == catalog.PERF_MAX_ROUNDS == 4
        and case.buffer_count == catalog.PERF_MAX_BUFFERS == 2
        and case.worker_a == case.worker_b == 0
        and case.issue_order == 0
        and case.relation == catalog.RELATION_DISJOINT
        and case.effect == catalog.EFFECT_NONE
        for case in catalog.SUSTAINED_PARALLEL_PAIR_CASES
    )
    assert all(
        (case.engine_a, case.engine_b)
        == (catalog.ENGINE_CT, catalog.ENGINE_RDMA)
        and (case.worker_a, case.worker_b) == (0, 1)
        and case.rounds == 4
        and case.buffer_count == 2
        for case in catalog.CROSS_WORKER_PARALLEL_PAIR_CASES
    )
    assert {
        case.relation
        for case in catalog.DEPENDENCY_PARALLEL_PAIR_CASES
    } == {
        catalog.RELATION_DISJOINT,
        catalog.RELATION_EXACT,
        catalog.RELATION_PARTIAL,
    }
    assert all(
        (case.engine_a, case.engine_b)
        == (catalog.ENGINE_RDMA, catalog.ENGINE_WDMA)
        and case.rounds == 1
        and case.buffer_count == 1
        and case.effect
        == (
            catalog.EFFECT_RAW
            if case.issue_order == 0
            else catalog.EFFECT_WAR
        )
        for case in catalog.DEPENDENCY_PARALLEL_PAIR_CASES
    )
    assert {
        (case.relation, case.issue_order, case.schedule)
        for case in catalog.DEPENDENCY_PARALLEL_PAIR_CASES
    } == {
        (relation, issue_order, schedule)
        for relation in (
            catalog.RELATION_DISJOINT,
            catalog.RELATION_EXACT,
            catalog.RELATION_PARTIAL,
        )
        for issue_order in (0, 1)
        for schedule in (
            catalog.SCHEDULE_SERIAL,
            catalog.SCHEDULE_WINDOW,
        )
    }
    assert tuple(
        catalog.REQ[name]
        for name in (
            "WORKER_A",
            "WORKER_B",
            "ROUNDS",
            "BUFFER_COUNT",
            "ISSUE_ORDER",
            "GUARD",
        )
    ) == tuple(range(34, 40))
    assert tuple(
        catalog.REC[name]
        for name in (
            "WORKER_A",
            "WORKER_B",
            "ROUNDS",
            "BUFFER_COUNT",
            "ISSUE_ORDER",
            "LANE_A_WORKER_INST_DELTA",
            "LANE_B_WORKER_INST_DELTA",
            "LANE_A_WORKER_BLOCKING_DELTA",
            "LANE_B_WORKER_BLOCKING_DELTA",
            "WORKER_MASK",
            "CONTROL_FINAL",
            "RECORD_GUARD",
        )
    ) == tuple(range(52, 64))
    assert "tensor<524288xf32>" in runner.MODULE
    assert all(
        signature["shape"] == [524288]
        for signature in (
            *runner.METADATA["input_signature"],
            *runner.METADATA["output_signature"],
        )
    )

    for sample, case in enumerate(catalog.CATALOG):
        descriptor = case.descriptor
        assert descriptor.inner_bytes > 0
        assert descriptor.compact_bytes > 0
        assert descriptor.envelope_bytes >= descriptor.inner_bytes
        if case not in catalog.ALIGNMENT_PAIR_CASES:
            assert case.spm_a % 256 == 0
        assert sum(case.expected_counts) > 0
        built = catalog.build_case_payload(case, sample)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert len(built.expected_output) == catalog.RESOURCE_BYTES
        request_words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert request_words[catalog.REQ["MAGIC"]] == catalog.REQUEST_MAGIC
        assert request_words[catalog.REQ["SCHEMA_AND_WORDS"]] == (
            catalog.SCHEMA << 32
        ) | catalog.REQUEST_WORDS
        assert request_words[catalog.REQ["WORKER_A"]] == case.worker_a
        assert request_words[catalog.REQ["WORKER_B"]] == case.worker_b
        assert request_words[catalog.REQ["ROUNDS"]] == case.rounds
        assert (
            request_words[catalog.REQ["BUFFER_COUNT"]]
            == case.buffer_count
        )
        assert request_words[catalog.REQ["ISSUE_ORDER"]] == case.issue_order
        assert request_words[catalog.REQ["GUARD"]] == catalog.REQUEST_GUARD
        assert built.request[catalog.REQUEST_WORDS * 8 :] == bytes(
            [catalog.RESOURCE_CANARY]
        ) * (catalog.RESOURCE_BYTES - catalog.REQUEST_WORDS * 8)
        assert built.allowed_ranges
        assert all(
            guard_range in built.allowed_ranges
            for guard_range in catalog.spm_dump_ranges(case)
        )
        assert tuple(
            output_range
            for output_range in built.allowed_ranges
            if output_range not in catalog.spm_dump_ranges(case)
        ) == catalog.output_result_ranges(case)
        if case.kind != catalog.KIND_PARALLEL_PAIR:
            assert built.payload[
                catalog.PAYLOAD_SPM_SEED_OFFSET
            ] == catalog.SPM_GUARD
            assert built.payload[
                catalog.PAYLOAD_SPM_SEED_OFFSET
                + catalog.SPM_GUARD_BYTES
                + catalog.spm_slot_spans(case)[0]
            ] == catalog.SPM_GUARD
        if case.kind == catalog.KIND_PARALLEL_PAIR:
            transfer = case.descriptor.compact_bytes
            assert catalog.spm_slot_spans(case) == (0, 0)
            assert built.exact_ranges == catalog.output_result_ranges(case)
            assert all(
                end - begin == 2 * catalog.SPM_GUARD_BYTES
                and built.expected_output[begin:end]
                == bytes([catalog.SPM_GUARD]) * (end - begin)
                for begin, end in catalog.spm_dump_ranges(case)
            )
            assert case.output_bytes == 2 * case.rounds * transfer
            if case.rounds == 4:
                assert case.buffer_count == 2
                for lane, engine in enumerate(
                    (case.engine_a, case.engine_b)
                ):
                    read0_begin = (
                        catalog.PERF_PAYLOAD_BASE
                        + lane * catalog.PERF_PAYLOAD_LANE_STRIDE
                        + catalog.PERF_PAYLOAD_READ0_OFFSET
                    )
                    buffers = tuple(
                        built.payload[
                            read0_begin + buffer * transfer :
                            read0_begin + (buffer + 1) * transfer
                        ]
                        for buffer in range(2)
                    )
                    if engine in (
                        catalog.ENGINE_CT,
                        catalog.ENGINE_NE,
                        catalog.ENGINE_WDMA,
                    ):
                        assert buffers[0] != buffers[1]
                    result_begin = (
                        catalog.PERF_OUTPUT_RESULT_BASE
                        + lane * catalog.PERF_OUTPUT_LANE_STRIDE
                    )
                    rounds = tuple(
                        built.expected_output[
                            result_begin + round_index * transfer :
                            result_begin + (round_index + 1) * transfer
                        ]
                        for round_index in range(4)
                    )
                    assert len(set(rounds)) >= 2
                    if engine in (
                        catalog.ENGINE_CT,
                        catalog.ENGINE_NE,
                        catalog.ENGINE_WDMA,
                    ):
                        assert rounds[0] == rounds[2]
                        assert rounds[1] == rounds[3]
        for begin, end in built.allowed_ranges:
            assert 0 <= begin < end <= catalog.RESOURCE_BYTES
        assert all(
            left_end <= right_begin or right_end <= left_begin
            for left_index, (left_begin, left_end) in enumerate(
                built.allowed_ranges
            )
            for right_begin, right_end in built.allowed_ranges[
                left_index + 1 :
            ]
        )
        for begin, end in built.exact_ranges:
            assert (begin, end) in built.allowed_ranges or any(
                outer_begin <= begin < end <= outer_end
                for outer_begin, outer_end in built.allowed_ranges
            )

    focused = runner.select_cases(
        types.SimpleNamespace(
            selected_cases=None,
            domain=None,
            oracle=None,
            relative_spm_offset=None,
            parallel_address_sweep=True,
        )
    )
    assert focused == catalog.PARALLEL_ADDRESS_SWEEP_CASES
    expanded = runner.select_cases(
        types.SimpleNamespace(
            selected_cases=None,
            domain=None,
            oracle=None,
            relative_spm_offset=None,
            parallel_address_sweep=False,
            parallel_pair_expanded=True,
        )
    )
    assert expanded == catalog.PARALLEL_PAIR_CASES
    expanded_conflicts = (
        {"selected_cases": [catalog.CATALOG[0].name]},
        {"domain": ["multi-engine-parallel-window"]},
        {"relative_spm_offset": [4352]},
        {"parallel_address_sweep": True},
    )
    for override in expanded_conflicts:
        arguments = {
            "selected_cases": None,
            "domain": None,
            "oracle": None,
            "relative_spm_offset": None,
            "parallel_address_sweep": False,
            "parallel_pair_expanded": True,
        }
        arguments.update(override)
        try:
            runner.select_cases(types.SimpleNamespace(**arguments))
        except RuntimeError as error:
            assert "--parallel-pair-expanded cannot be combined" in str(error)
        else:
            raise AssertionError(
                "expanded parallel-pair selector accepted a mixed filter"
            )

    synthetic_cases = (
        catalog.SUSTAINED_PARALLEL_PAIR_CASES[0],
        catalog.CROSS_WORKER_PARALLEL_PAIR_CASES[0],
        next(
            case
            for case in catalog.DEPENDENCY_PARALLEL_PAIR_CASES
            if case.relation == catalog.RELATION_EXACT
            and case.issue_order == 0
            and case.schedule == catalog.SCHEDULE_WINDOW
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        output_path = pathlib.Path(directory) / "output.raw"

        def materialize_output(
            case: catalog.MemoryCase, sample: int
        ) -> tuple[catalog.CasePayload, bytearray]:
            built = catalog.build_case_payload(case, sample)
            raw = bytearray(built.expected_output)
            words = [0] * catalog.RECORD_WORDS
            for key, value in runner._expected_record(case, sample).items():
                words[catalog.REC[key]] = value
            words[catalog.REC["PLAN_CYCLES"]] = 1
            words[catalog.REC["PMU_ENABLE"]] = 1
            words[catalog.REC["STABLE_BEFORE"]] = runner.PMU_STABLE_MASK
            words[catalog.REC["STABLE_AFTER"]] = runner.PMU_STABLE_MASK
            words[catalog.REC["LANE_A_WORKER_INST_DELTA"]] = case.rounds
            words[catalog.REC["LANE_B_WORKER_INST_DELTA"]] = case.rounds
            words[catalog.REC["WORKER_MASK"]] = (
                (1 << case.worker_a) | (1 << case.worker_b)
            )
            words[catalog.REC["CONTROL_FINAL"]] = (
                0x100 | (0x100 << 32)
            )
            struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *words)
            return built, raw

        for case in synthetic_cases:
            built, raw = materialize_output(case, 7)
            output_path.write_bytes(raw)
            observation = runner.validate_output(
                output_path, case, built, 7
            )
            assert observation["correctness"] == (
                "exact-result+full-spm-dump-guard"
            )
            assert observation["parallel_pair"][
                "worker_instruction_deltas"
            ] == [case.rounds, case.rounds]

        strict_case = catalog.SUSTAINED_PARALLEL_PAIR_CASES[0]
        strict_built, valid_raw = materialize_output(strict_case, 11)

        def expect_rejected(label: str, raw: bytearray) -> None:
            output_path.write_bytes(raw)
            try:
                runner.validate_output(
                    output_path, strict_case, strict_built, 11
                )
            except RuntimeError:
                return
            raise AssertionError(
                f"host strict oracle accepted invalid {label}"
            )

        invalid_words = (
            ("request echo", "WORKER_A", strict_case.worker_a + 1),
            ("guard mismatch count", "SPM_GUARD_MISMATCHES", 1),
            ("PMU enable", "PMU_ENABLE", 0),
            (
                "lane instruction count",
                "LANE_A_WORKER_INST_DELTA",
                strict_case.rounds + 1,
            ),
            ("worker mask", "WORKER_MASK", 0),
            ("worker completion controls", "CONTROL_FINAL", 0x100 << 32),
            ("worker blocking consistency", "LANE_A_WORKER_BLOCKING_DELTA", 1),
        )
        for label, field, value in invalid_words:
            raw = bytearray(valid_raw)
            struct.pack_into("<Q", raw, catalog.REC[field] * 8, value)
            expect_rejected(label, raw)

        raw = bytearray(valid_raw)
        result_begin = strict_built.exact_ranges[0][0]
        raw[result_begin] ^= 0xFF
        expect_rejected("result bytes", raw)
        raw = bytearray(valid_raw)
        guard_begin = catalog.spm_dump_ranges(strict_case)[0][0]
        raw[guard_begin] ^= 0xFF
        expect_rejected("SPM guard bytes", raw)

        invocation = catalog.build_conflict_equivalence_invocation(
            catalog.CONFLICT_EQUIVALENCE_PAIRS[0], 0
        )
        conflict_raw = bytearray(
            [catalog.RESOURCE_CANARY] * catalog.RESOURCE_BYTES
        )
        actual_bases = {
            "REQUEST_DDR": 0x10000000,
            "PAYLOAD_DDR": 0x20000000,
            "OUTPUT_DDR": 0x30000000,
        }
        for row, (case, built, output_offset) in enumerate(
            (
                (
                    invocation.pair.serial,
                    invocation.serial_built,
                    catalog.CONFLICT_SERIAL_OUTPUT_OFFSET,
                ),
                (
                    invocation.pair.window,
                    invocation.window_built,
                    catalog.CONFLICT_WINDOW_OUTPUT_OFFSET,
                ),
            )
        ):
            conflict_raw[
                output_offset :
                output_offset + catalog.CONFLICT_OUTPUT_ROW_BYTES
            ] = built.expected_output[
                : catalog.CONFLICT_OUTPUT_ROW_BYTES
            ]
            for begin, end in catalog.spm_dump_ranges(case):
                conflict_raw[
                    output_offset + begin : output_offset + end
                ] = bytes([catalog.SPM_GUARD]) * (end - begin)
            record = [0] * catalog.RECORD_WORDS
            for key, value in runner._expected_record(
                case, invocation.sample
            ).items():
                record[catalog.REC[key]] = value
            record[catalog.REC["PLAN_CYCLES"]] = row + 1
            record[catalog.REC["PMU_ENABLE"]] = 1
            record[catalog.REC["STABLE_BEFORE"]] = (
                runner.PMU_STABLE_MASK
            )
            record[catalog.REC["STABLE_AFTER"]] = (
                runner.PMU_STABLE_MASK
            )
            struct.pack_into(
                f"<{catalog.RECORD_WORDS}Q",
                conflict_raw,
                output_offset,
                *record,
            )

            request_word = (
                catalog.CONFLICT_SERIAL_REQUEST_WORD
                if row == 0
                else catalog.CONFLICT_WINDOW_REQUEST_WORD
            )
            row_output_ddr = (
                actual_bases["OUTPUT_DDR"] + output_offset
            )
            meta = [0] * catalog.CONFLICT_RECORD_META_WORDS
            meta_values = {
                "MAGIC": catalog.CONFLICT_RECORD_MAGIC,
                "SCHEMA_AND_WORDS": (
                    catalog.CONFLICT_SCHEMA << 32
                )
                | catalog.CONFLICT_RECORD_META_WORDS,
                "STATUS": 0,
                "COORDINATE": invocation.pair.coordinate_id,
                "INNER_CASE": case.case_id,
                "SCHEDULE": case.schedule,
                "ISSUE_ORDER": case.issue_order,
                "SAMPLE": invocation.sample,
                "EXECUTION_ORDINAL": (
                    0
                    if case.schedule == invocation.first_schedule
                    else 1
                ),
                "FIRST_SCHEDULE": invocation.first_schedule,
                **actual_bases,
                "INNER_REQUEST_DDR": (
                    actual_bases["REQUEST_DDR"] + request_word * 8
                ),
                "ROW_OUTPUT_DDR": row_output_ddr,
                "CT_INPUT0_DDR": (
                    actual_bases["PAYLOAD_DDR"]
                    + catalog.PAYLOAD_DATA_OFFSET
                ),
                "CT_INPUT1_DDR": (
                    actual_bases["PAYLOAD_DDR"]
                    + catalog.PAYLOAD_DATA_OFFSET
                    + 4096
                ),
                "RDMA_INPUT_DDR": (
                    actual_bases["PAYLOAD_DDR"]
                    + catalog.PAYLOAD_DATA_OFFSET
                    + 16384
                    + 8192
                ),
                "RESULT_A_DDR": (
                    row_output_ddr + catalog.OUTPUT_DATA_OFFSET
                ),
                "RESULT_B_DDR": (
                    row_output_ddr
                    + catalog.OUTPUT_DATA_OFFSET
                    + case.descriptor.compact_bytes
                ),
                "SPM_A": case.spm_a,
                "SPM_B": case.spm_b,
                "WINDOW_FLAGS": catalog.CONFLICT_WINDOW_FLAGS,
                "RECORD_GUARD": catalog.CONFLICT_RECORD_GUARD,
            }
            for key, value in meta_values.items():
                meta[catalog.CONFLICT_REC[key]] = value
            meta_begin = (
                catalog.CONFLICT_RECORD_META_WORD
                + row * catalog.CONFLICT_RECORD_META_STRIDE_WORDS
            ) * 8
            struct.pack_into(
                f"<{catalog.CONFLICT_RECORD_META_WORDS}Q",
                conflict_raw,
                meta_begin,
                *meta,
            )

        output_path.write_bytes(conflict_raw)
        conflict_rows = runner.validate_conflict_equivalence_output(
            output_path, invocation
        )
        assert set(conflict_rows) == {"serial", "window"}
        assert (
            conflict_rows["serial"]["actual_addresses"]["payload_ddr"]
            == conflict_rows["window"]["actual_addresses"]["payload_ddr"]
            == actual_bases["PAYLOAD_DDR"]
        )

        def expect_conflict_rejected(
            label: str, candidate: bytearray
        ) -> None:
            output_path.write_bytes(candidate)
            try:
                runner.validate_conflict_equivalence_output(
                    output_path, invocation
                )
            except RuntimeError:
                return
            raise AssertionError(
                f"conflict host oracle accepted invalid {label}"
            )

        invalid = bytearray(conflict_raw)
        window_meta = (
            catalog.CONFLICT_RECORD_META_WORD
            + catalog.CONFLICT_RECORD_META_STRIDE_WORDS
        ) * 8
        struct.pack_into(
            "<Q",
            invalid,
            window_meta
            + catalog.CONFLICT_REC["PAYLOAD_DDR"] * 8,
            actual_bases["PAYLOAD_DDR"] + 4096,
        )
        expect_conflict_rejected("shared payload base", invalid)
        invalid = bytearray(conflict_raw)
        for row in range(catalog.CONFLICT_ROWS):
            meta_begin = (
                catalog.CONFLICT_RECORD_META_WORD
                + row * catalog.CONFLICT_RECORD_META_STRIDE_WORDS
            ) * 8
            for field in ("REQUEST_DDR", "INNER_REQUEST_DDR"):
                field_offset = (
                    meta_begin + catalog.CONFLICT_REC[field] * 8
                )
                value = struct.unpack_from(
                    "<Q", invalid, field_offset
                )[0]
                struct.pack_into(
                    "<Q", invalid, field_offset, value + 1
                )
        expect_conflict_rejected("unaligned allocation base", invalid)
        invalid = bytearray(conflict_raw)
        struct.pack_into(
            "<Q",
            invalid,
            catalog.CONFLICT_RECORD_META_WORD * 8
            + catalog.CONFLICT_REC["WINDOW_FLAGS"] * 8,
            0,
        )
        expect_conflict_rejected("pair-only PMU window flags", invalid)
        invalid = bytearray(conflict_raw)
        invalid[0xA000] ^= 0xFF
        expect_conflict_rejected("whole-output canary", invalid)

    fake_conflict_observations = {
        pair.name: [
            {
                schedule: {
                    "plan_cycles": (
                        100
                        + pair.coordinate_id
                        + sample
                        + (10 if schedule == "window" else 0)
                    ),
                    "pmu": {
                        "full_execution": (
                            80 + (5 if schedule == "window" else 0)
                        ),
                        "ct_execution": 30,
                        "rdma_execution": 40,
                        "ct_blocking": (
                            3 + (1 if schedule == "window" else 0)
                        ),
                        "rdma_blocking": (
                            4 + (1 if schedule == "window" else 0)
                        ),
                    },
                    "actual_addresses": {
                        "request_ddr": 0x10000000 + sample * 0x100000,
                        "payload_ddr": 0x20000000 + sample * 0x100000,
                        "output_ddr": 0x30000000 + sample * 0x100000,
                    },
                }
                for schedule in ("serial", "window")
            }
            for sample in range(catalog.CONFLICT_SAMPLES)
        ]
        for pair in catalog.CONFLICT_EQUIVALENCE_PAIRS
    }
    summary_output = io.StringIO()
    with contextlib.redirect_stdout(summary_output):
        runner.emit_conflict_equivalence_summary(
            fake_conflict_observations
        )
    final_summary = json.loads(
        summary_output.getvalue().split(
            "spm_conflict_equivalence_summary: "
        )[-1]
    )
    assert final_summary["state"] == (
        "heldout-proxy-direction-consistent"
    )
    assert len(final_summary["phase_rows"]) == (
        len(catalog.SPM_PARALLEL_ALIGNMENT_PHASES)
        * len(catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS)
        * len(catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS)
    )
    zero_signal_observations = {
        pair.name: [
            {
                schedule: {
                    "plan_cycles": 100,
                    "pmu": {
                        "full_execution": 80,
                        "ct_execution": 30,
                        "rdma_execution": 40,
                        "ct_blocking": 0,
                        "rdma_blocking": 0,
                    },
                    "actual_addresses": {
                        "request_ddr": 0x10000000 + sample * 0x100000,
                        "payload_ddr": 0x20000000 + sample * 0x100000,
                        "output_ddr": 0x30000000 + sample * 0x100000,
                    },
                }
                for schedule in ("serial", "window")
            }
            for sample in range(catalog.CONFLICT_SAMPLES)
        ]
        for pair in catalog.CONFLICT_EQUIVALENCE_PAIRS
    }
    zero_summary_output = io.StringIO()
    with contextlib.redirect_stdout(zero_summary_output):
        runner.emit_conflict_equivalence_summary(
            zero_signal_observations
        )
    zero_summary = json.loads(
        zero_summary_output.getvalue().split(
            "spm_conflict_equivalence_summary: "
        )[-1]
    )
    assert zero_summary["state"] == (
        "consistent-but-no-nonzero-cost-signal"
    )
    assert not any(
        row["promotable_proxy_pattern"]
        for row in zero_summary["phase_rows"]
    )

    invalid_phase = dataclasses.replace(
        catalog.ALIGNMENT_PAIR_CASES[0],
        spm_a=catalog.ALIGNMENT_PAIR_CASES[0].spm_a + 1,
        spm_b=catalog.ALIGNMENT_PAIR_CASES[0].spm_b + 1,
    )
    try:
        catalog.build_case_payload(invalid_phase)
    except RuntimeError as error:
        assert "unsupported SPM parallel address coordinate" in str(error)
    else:
        raise AssertionError("sub-64-byte base phase reached serialization")

    for sample, case in enumerate(catalog.PENDING_CASES):
        built = catalog.build_case_payload(case, sample)
        request_words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert request_words[catalog.REQ["CASE"]] == case.case_id
        assert request_words[catalog.REQ["SPM_A"]] == case.spm_a
        assert request_words[catalog.REQ["SPM_B"]] == case.spm_b
        assert request_words[catalog.REQ["SAMPLE"]] == sample
        assert built.allowed_ranges
        assert built.exact_ranges

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
    assert "wafer_mdc_execute_parallel" in probe
    assert "wafer_mdc_execute_conflict_equivalence" in probe
    assert "wafer_mdc_ce_write_meta" in probe
    assert "WAFER_MDC_CE_REC_PAYLOAD_DDR" in probe
    assert "WAFER_MDC_CE_WINDOW_FLAGS" in probe
    assert "wafer_mdc_prepare_parallel_instruction" in probe
    assert "wafer_mdc_drain_worker" in probe
    assert "WAFER_MDC_REC_LANE_A_WORKER_INST_DELTA" in probe
    assert "WAFER_MDC_REC_LANE_B_WORKER_INST_DELTA" in probe
    assert "WAFER_MDC_REC_WORKER_MASK" in probe
    assert "WAFER_MDC_REC_CONTROL_FINAL" in probe
    assert "wafer_tx81_local_fence" in probe
    assert "wafer_mdc_seed_slot" in probe
    assert "wafer_mdc_dump_slot" in probe
    assert "wafer_mdc_first_spm_dump_offset" in probe
    assert "wafer_mdc_output_layout_fits" in probe
    assert "WAFER_MDC_OUTPUT_SPM_DUMP0_OFFSET" not in probe
    assert "WAFER_MDC_REC_PLAN_CYCLES" in probe
    assert "WAFER_MDC_REC_FULL_EXEC_DELTA" in probe
    assert "GR_CSR_SERIAL_MODE_ADDR" in probe
    assert "get_spm_memory_mapping" not in probe
    assert "wafer_mdc_guard_mismatches" not in probe
    engine_window = probe[
        probe.index("static uint32_t wafer_mdc_execute_engines") :
        probe.index("static uint32_t wafer_mdc_issue_parallel_lane")
    ]
    assert (
        engine_window.index("wafer_mdc_prepare_engine")
        < engine_window.index("WaferMDCPMU before")
        < engine_window.index("wafer_mdc_issue_engine")
        < engine_window.index("WaferMDCPMU after")
        < engine_window.index("wafer_mdc_dump_slot")
    )
    print(
        "wafer_memory_descriptor_calibration_catalog_test: "
        f"cases={len(catalog.CATALOG)} "
        f"exact={sum(case.is_exact for case in catalog.CATALOG)} "
        f"observation={sum(not case.is_exact for case in catalog.CATALOG)} "
        "passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
