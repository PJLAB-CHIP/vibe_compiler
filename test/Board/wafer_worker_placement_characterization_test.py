#!/usr/bin/env python3
"""Host tests for the real worker placement/progress board adapter."""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import struct
import tempfile

import wafer_board_worker_placement_characterization_test as driver
import wafer_worker_placement_characterization_catalog as catalog
import wafer_worker_placement_probe_protocol as protocol


def _record(
    case: catalog.WorkerCase, sample: int
) -> tuple[list[int], bytearray]:
    issues = protocol.issues_for_case(case, sample)
    words = [0] * protocol.RECORD_WORDS
    values = {
        "MAGIC": protocol.RECORD_MAGIC,
        "WORD_COUNT": protocol.RECORD_WORDS,
        "STATUS": protocol.STATUS_OK,
        "FLAGS": protocol.ALL_FLAGS,
        "KIND": int(case.kind),
        "ENGINE": int(case.engine),
        "WORKER_MASK": case.worker_mask,
        "TARGET_WORKER": case.target_worker,
        "OBSERVER_WORKER": case.observer_worker,
        "SAMPLE": sample,
        "ISSUE_COUNT": len(issues),
        "PRIMARY_BYTES": case.primary_bytes,
        "SENTINEL_BYTES": case.sentinel_bytes,
        "REQUEST_GUARD": protocol.REQUEST_GUARD,
        "RECORD_GUARD": protocol.RECORD_GUARD,
        "RESOURCE_BYTES": protocol.RESOURCE_BYTES,
        "OUTPUT_SLOT_BASE": protocol.OUTPUT_SLOT_BASE,
        "OUTPUT_SLOT_STRIDE": protocol.OUTPUT_SLOT_STRIDE,
        "OUTPUT_GUARD_BYTES": protocol.GUARD_BYTES,
        "PMU_ENABLE": 1,
        "STABLE_BEFORE": protocol.PMU_STABLE_MASK,
        "STABLE_AFTER": protocol.PMU_STABLE_MASK,
        "START_CYCLE": 100,
        "AFTER_ISSUE_CYCLE": 200,
        "OBSERVER_BOUNDARY_CYCLE": 300,
        "BOUNDARY_ORACLE_CYCLE": 310,
        "FINAL_CYCLE": 400,
        "BOUNDARY_MISMATCHES": 0,
        "FINAL_MISMATCHES": 0,
        "FINAL_GUARD_MISMATCHES": 0,
        "POLL_ROUNDS": 3,
        "OBSERVATION_DEADLINE_CYCLES": (
            protocol.OBSERVATION_DEADLINE_CYCLES
        ),
        "CLEANUP_DEADLINE_CYCLES": protocol.CLEANUP_DEADLINE_CYCLES,
        "ACCEPTED_ISSUE_COUNT": len(issues),
        "ACCEPTED_WORKER_MASK": case.worker_mask,
        "PENDING_WORKER_MASK": case.worker_mask,
        "ATTEMPTED_ISSUE_COUNT": len(issues),
        "ATTEMPTED_WORKER_MASK": case.worker_mask,
    }
    for name, value in values.items():
        words[protocol.REC[name]] = value
    for worker, count in enumerate(case.worker_issues):
        words[
            protocol.REC["ACCEPTED_ISSUES_BY_WORKER_BASE"] + worker
        ] = count
    expected_counts = protocol.expected_instruction_counts(issues)
    for worker in range(protocol.WORKERS):
        worker_bit = 1 << worker
        participating = bool(case.worker_mask & worker_bit)
        words[protocol.REC["CONTROL_BEFORE"] + worker] = (
            protocol.TASK_DONE
        )
        words[protocol.REC["CONTROL_AFTER_ISSUE"] + worker] = (
            1 if participating else protocol.TASK_DONE
        )
        boundary_done = not (
            case.kind == catalog.Kind.CONCURRENT
            and worker == case.target_worker
        )
        words[protocol.REC["CONTROL_BOUNDARY"] + worker] = (
            protocol.TASK_DONE if boundary_done else 1
        )
        words[protocol.REC["CONTROL_FINAL"] + worker] = (
            protocol.TASK_DONE
        )
        if participating:
            words[
                protocol.REC["MATCHING_JOIN_CYCLE_BASE"] + worker
            ] = 3 + worker
            words[
                protocol.REC["MATCHING_JOIN_RC_BASE"] + worker
            ] = protocol.WAIT_SUCCESS
            words[
                protocol.REC["MATCHING_JOIN_CONTROL_BASE"] + worker
            ] = protocol.TASK_DONE
        if participating:
            first_done = (
                350
                if case.kind == catalog.Kind.CONCURRENT
                and worker == case.target_worker
                else 220 + worker * 10
            )
            words[
                protocol.REC["FIRST_DONE_CYCLE_BASE"] + worker
            ] = first_done
            words[protocol.REC["POLL_COUNT_BASE"] + worker] = worker + 1
            words[protocol.REC["MAX_POLL_GAP_BASE"] + worker] = 7 + worker
        for queue in range(protocol.QUEUES):
            index = worker * protocol.QUEUES + queue
            words[protocol.REC["INSTRUCTION_AFTER"] + index] = (
                expected_counts[(worker, queue)]
            )
            words[protocol.REC["BLOCKING_AFTER"] + index] = (
                worker * 10 + queue
            )
    observer_first = words[
        protocol.REC["FIRST_DONE_CYCLE_BASE"] + case.observer_worker
    ]
    target_first = words[
        protocol.REC["FIRST_DONE_CYCLE_BASE"] + case.target_worker
    ]
    words[protocol.REC["OBSERVER_FIRST_DONE_CYCLES"]] = (
        observer_first - values["START_CYCLE"] if observer_first else 0
    )
    words[protocol.REC["TARGET_FIRST_DONE_CYCLES"]] = (
        target_first - values["START_CYCLE"] if target_first else 0
    )
    observer_boundary = words[
        protocol.REC["CONTROL_BOUNDARY"] + case.observer_worker
    ]
    target_boundary = words[
        protocol.REC["CONTROL_BOUNDARY"] + case.target_worker
    ]
    words[protocol.REC["OBSERVER_DONE_BOUNDARY"]] = int(
        bool(observer_boundary & protocol.TASK_DONE)
    )
    words[protocol.REC["TARGET_PENDING_BOUNDARY"]] = int(
        not bool(target_boundary & protocol.TASK_DONE)
    )
    words[protocol.REC["MATCHING_JOIN_MASK"]] = case.worker_mask
    for index in range(protocol.PMU64_COUNTERS):
        words[protocol.REC["PMU64_BEFORE"] + index] = 1000 + index
        words[protocol.REC["PMU64_AFTER"] + index] = 1100 + index

    output = bytearray(
        [protocol.OUTPUT_CANARY]
    ) * protocol.RESOURCE_BYTES
    submitted = [0] * protocol.WORKERS
    for issue in issues:
        submitted[issue.worker] += 1
        ordinal = issue.ordinal
        words[protocol.REC["ISSUE_WORKER_BASE"] + ordinal] = issue.worker
        words[protocol.REC["ISSUE_ENGINE_BASE"] + ordinal] = int(
            issue.engine
        )
        words[protocol.REC["ISSUE_BYTES_BASE"] + ordinal] = (
            issue.output_bytes
        )
        words[protocol.REC["ISSUE_RC_BASE"] + ordinal] = 1
        words[protocol.REC["ISSUE_INTER_TYPE_BASE"] + ordinal] = (
            int(issue.engine) | (issue.worker << 8)
        )
        words[protocol.REC["ISSUE_ORDINAL_BASE"] + ordinal] = ordinal
        words[protocol.REC["ISSUE_SPM_SLOT_BASE"] + ordinal] = (
            issue.spm_slot
        )
        words[
            protocol.REC["ISSUE_READ0_ADDRESS_BASE"] + ordinal
        ] = protocol.spm_read0_address(issue.spm_slot)
        words[
            protocol.REC["ISSUE_READ1_ADDRESS_BASE"] + ordinal
        ] = protocol.spm_read1_address(issue.spm_slot)
        words[
            protocol.REC["ISSUE_WRITE_ADDRESS_BASE"] + ordinal
        ] = protocol.spm_write_address(issue.spm_slot)
        if case.kind not in {
            catalog.Kind.BACKLOG_ONLY,
            catalog.Kind.SENTINEL_ONLY,
            catalog.Kind.CONCURRENT,
        }:
            words[protocol.REC["ISSUE_CYCLE_BASE"] + ordinal] = (
                110 + ordinal * 10
            )
            for worker in range(protocol.WORKERS):
                control = (
                    submitted[worker]
                    if submitted[worker]
                    else protocol.TASK_DONE
                )
                words[
                    protocol.REC["CONTROL_PER_ISSUE_BASE"]
                    + ordinal * protocol.WORKERS
                    + worker
                ] = control
        output_offset = (
            protocol.OUTPUT_SLOT_BASE
            + ordinal * protocol.OUTPUT_SLOT_STRIDE
            + protocol.GUARD_BYTES
        )
        words[protocol.REC["OUTPUT_BYTES_BASE"] + ordinal] = (
            issue.output_bytes
        )
        words[protocol.REC["OUTPUT_OFFSET_BASE"] + ordinal] = output_offset
        slot_begin = output_offset - protocol.GUARD_BYTES
        output[slot_begin:output_offset] = bytes([0xA5]) * protocol.GUARD_BYTES
        expected = driver.expected_result(case, sample, issue)
        output[output_offset : output_offset + len(expected)] = expected
        output[
            output_offset + len(expected) :
            output_offset + len(expected) + protocol.GUARD_BYTES
        ] = bytes([0xA5]) * protocol.GUARD_BYTES
    output[: protocol.RECORD_WORDS * 8] = struct.pack(
        f"<{protocol.RECORD_WORDS}Q", *words
    )
    return words, output


def _failure_record(
    case: catalog.WorkerCase,
    sample: int,
    primary_status: int,
    *,
    poisoned: bool = False,
) -> list[int]:
    words, _ = _record(case, sample)
    issues = protocol.issues_for_case(case, sample)
    if primary_status == protocol.STATUS_ISSUE_FAILED:
        attempted_count = 2
        for ordinal in range(attempted_count - 1, len(issues)):
            words[protocol.REC["ISSUE_RC_BASE"] + ordinal] = 0
        accepted_ordinals = list(range(attempted_count - 1))
        words[protocol.REC["FLAGS"]] = protocol.RECORD_FLAGS[
            "BEFORE_CAPTURED"
        ]
        words[protocol.REC["OBSERVATION_TIMEOUT_MASK"]] = 0
        words[protocol.REC["OBSERVATION_TIMEOUT_CYCLE"]] = 0
    else:
        assert primary_status == protocol.STATUS_OBSERVATION_TIMEOUT
        attempted_count = len(issues)
        accepted_ordinals = list(range(len(issues)))
        words[protocol.REC["FLAGS"]] = (
            protocol.RECORD_FLAGS["BEFORE_CAPTURED"]
            | protocol.RECORD_FLAGS["ISSUES_SUBMITTED"]
        )
        words[protocol.REC["OBSERVATION_TIMEOUT_MASK"]] = (
            1 << case.target_worker
        )
        words[protocol.REC["OBSERVATION_TIMEOUT_CYCLE"]] = 250

    accepted_by_worker = [0] * protocol.WORKERS
    accepted_mask = 0
    for ordinal in accepted_ordinals:
        worker = issues[ordinal].worker
        accepted_by_worker[worker] += 1
        accepted_mask |= 1 << worker
    attempted_mask = 0
    for issue in issues[:attempted_count]:
        attempted_mask |= 1 << issue.worker
    words[protocol.REC["ACCEPTED_ISSUE_COUNT"]] = len(
        accepted_ordinals
    )
    words[protocol.REC["ACCEPTED_WORKER_MASK"]] = accepted_mask
    words[protocol.REC["ATTEMPTED_ISSUE_COUNT"]] = attempted_count
    words[protocol.REC["ATTEMPTED_WORKER_MASK"]] = attempted_mask
    words[protocol.REC["CLEANUP_PARTICIPANT_MASK"]] = attempted_mask
    for worker, count in enumerate(accepted_by_worker):
        words[
            protocol.REC["ACCEPTED_ISSUES_BY_WORKER_BASE"] + worker
        ] = count
    for worker in range(protocol.WORKERS):
        words[protocol.REC["CONTROL_AFTER_ISSUE"] + worker] = (
            1 if attempted_mask & (1 << worker) else protocol.TASK_DONE
        )
        words[protocol.REC["CONTROL_FINAL"] + worker] = (
            protocol.TASK_DONE
        )
        words[
            protocol.REC["MATCHING_JOIN_CYCLE_BASE"] + worker
        ] = 0
        words[protocol.REC["MATCHING_JOIN_RC_BASE"] + worker] = 0
        words[
            protocol.REC["MATCHING_JOIN_CONTROL_BASE"] + worker
        ] = 0
    words[protocol.REC["PENDING_WORKER_MASK"]] = attempted_mask
    words[protocol.REC["MATCHING_JOIN_MASK"]] = 0
    words[protocol.REC["PRIMARY_FAILURE_STATUS"]] = primary_status
    words[protocol.REC["CLEANUP_ATTEMPT_COUNT"]] = 1
    words[protocol.REC["CLEANUP_START_CYCLE"]] = 260
    words[protocol.REC["CLEANUP_END_CYCLE"]] = 300
    if poisoned:
        poisoned_worker = next(
            worker
            for worker in range(protocol.WORKERS)
            if attempted_mask & (1 << worker)
        )
        words[protocol.REC["STATUS"]] = (
            protocol.STATUS_CLEANUP_POISONED
        )
        words[protocol.REC["CLEANUP_FLAGS"]] = (
            protocol.CLEANUP_ATTEMPTED | protocol.CLEANUP_POISONED
        )
        words[protocol.REC["CLEANUP_TIMEOUT_MASK"]] = (
            1 << poisoned_worker
        )
        words[protocol.REC["CONTROL_FINAL"] + poisoned_worker] = 1
    else:
        words[protocol.REC["STATUS"]] = primary_status
        words[protocol.REC["CLEANUP_FLAGS"]] = (
            protocol.CLEANUP_ATTEMPTED | protocol.CLEANUP_SUCCEEDED
        )
        words[protocol.REC["CLEANUP_TIMEOUT_MASK"]] = 0
    return words


def _expect_failure(
    words: list[int],
    case: catalog.WorkerCase,
    sample: int,
    needle: str,
) -> None:
    try:
        protocol.validate_record(tuple(words), case, sample)
    except RuntimeError as error:
        assert needle in str(error), (needle, str(error))
    else:
        raise AssertionError(f"tampered record accepted; wanted {needle!r}")


def main() -> int:
    catalog.validate_catalog()
    driver.validate_static_contract()
    assert len(catalog.CASES) == 44
    assert len(catalog.GROUPS) == 14
    assert len(
        [case for case in catalog.CASES if case.kind == catalog.Kind.PLACEMENT]
    ) == 14
    assert len(
        [
            case
            for case in catalog.CASES
            if case.kind
            in {
                catalog.Kind.BACKLOG_ONLY,
                catalog.Kind.SENTINEL_ONLY,
                catalog.Kind.CONCURRENT,
            }
        ]
    ) == 18
    assert len(
        [
            case
            for case in catalog.CASES
            if case.kind == catalog.Kind.OUTSTANDING
        ]
    ) == 12
    assert all(
        case.issue_count == catalog.PLACEMENT_ISSUES
        for case in catalog.CASES
        if case.kind == catalog.Kind.PLACEMENT
    )
    assert {
        tuple(sorted(count for count in case.worker_issues if count))
        for case in catalog.CASES
        if case.kind == catalog.Kind.PLACEMENT
    } == {(6,), (3, 3), (2, 2, 2)}
    assert len(
        {
            protocol.request_words(case, sample)
            for case in catalog.CASES
            for sample in range(catalog.REPEATS)
        }
    ) == len(catalog.CASES) * catalog.REPEATS
    for group in catalog.GROUPS:
        cases = catalog.GROUP_CASES[group]
        if group.startswith("worker-placement-"):
            assert len(cases) == 7
        elif group.startswith("worker-outstanding-"):
            assert len(cases) == 2
            assert {case.issue_count for case in cases} == {2, 6}
        else:
            assert {case.kind for case in cases} == {
                catalog.Kind.BACKLOG_ONLY,
                catalog.Kind.SENTINEL_ONLY,
                catalog.Kind.CONCURRENT,
            }
    for group in (
        "worker-placement-ct",
        "worker-progress-ne-target-w0-observer-w1",
        "worker-outstanding-ct-w0",
    ):
        cases = catalog.GROUP_CASES[group]
        plan = driver.execution_plan(cases, catalog.REPEATS)
        assert [sample for sample, _ in plan] == [
            sample
            for sample in range(catalog.REPEATS)
            for _ in cases
        ]
        actual_orders = [
            tuple(
                case
                for planned_sample, case in plan
                if planned_sample == sample
            )
            for sample in range(catalog.REPEATS)
        ]
        expected_orders = []
        for sample in range(catalog.REPEATS):
            rotation = sample % len(cases)
            expected = cases[rotation:] + cases[:rotation]
            if sample & 1 and len(cases) > 2:
                expected = tuple(reversed(expected))
            expected_orders.append(expected)
        assert actual_orders == expected_orders
        assert all(set(order) == set(cases) for order in actual_orders)
        assert len(set(actual_orders)) >= min(
            catalog.REPEATS, len(cases)
        )
        observations = [
            protocol.validate_record(
                tuple(_record(case, sample)[0]), case, sample
            )
            for case in cases
            for sample in range(catalog.REPEATS)
        ]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            driver.report_group(cases, observations, catalog.REPEATS)
        assert "worker_placement_decision:" in captured.getvalue()

    selected_key = "worker-progress-ne-target-w0-observer-w1-concurrent"
    selected = catalog.CASES_BY_KEY[selected_key]
    expanded = driver.selected_cases(
        argparse.Namespace(
            case=selected_key,
            group=None,
            no_card=False,
        )
    )
    assert expanded == catalog.GROUP_CASES[selected.group]
    try:
        driver.report_group((selected,), [], catalog.REPEATS)
    except RuntimeError as error:
        assert "complete group" in str(error)
    else:
        raise AssertionError("single case produced a matched decision")

    outstanding_group = catalog.GROUP_CASES[
        "worker-outstanding-ct-w0"
    ]
    indistinguishable = [
        protocol.validate_record(
            tuple(_record(case, sample)[0]), case, sample
        )
        for case in outstanding_group
        for sample in range(catalog.REPEATS)
    ]
    for observation in indistinguishable:
        maxima = observation["observed_max_ib_counter_by_worker"]
        assert isinstance(maxima, list)
        maxima[0] = 1
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        driver.report_group(
            outstanding_group, indistinguishable, catalog.REPEATS
        )
    assert (
        "inconclusive-no-distinguishing-ib-response"
        in captured.getvalue()
    )

    for group in (
        "worker-progress-ne-target-w0-observer-w1",
        "worker-progress-rdma-target-w0-observer-w1",
    ):
        progress_cases = catalog.GROUP_CASES[group]
        backlog = next(
            case
            for case in progress_cases
            if case.kind == catalog.Kind.BACKLOG_ONLY
        )
        sentinel_only = next(
            case
            for case in progress_cases
            if case.kind == catalog.Kind.SENTINEL_ONLY
        )
        concurrent = next(
            case
            for case in progress_cases
            if case.kind == catalog.Kind.CONCURRENT
        )
        assert [
            issue.spm_slot
            for issue in protocol.issues_for_case(backlog, 0)
        ] == [0, 1, 2, 3]
        sentinel_control = protocol.issues_for_case(
            sentinel_only, 0
        )[-1]
        concurrent_sentinel = protocol.issues_for_case(
            concurrent, 0
        )[-1]
        assert sentinel_control.ordinal == 0
        assert concurrent_sentinel.ordinal == 4
        assert (
            sentinel_control.spm_slot
            == concurrent_sentinel.spm_slot
            == protocol.PROGRESS_SENTINEL_SPM_SLOT
        )
        assert protocol.spm_write_address(
            sentinel_control.spm_slot
        ) == protocol.spm_write_address(concurrent_sentinel.spm_slot)

    for boundary in catalog.BOUNDARIES:
        try:
            driver.prepare_selection(boundary.key)
        except driver.PreparationBlocked as error:
            assert boundary.typed_gate in str(error)
            assert boundary.safe_alternative in str(error)
        else:
            raise AssertionError("typed worker boundary became board-positive")

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        for case in catalog.CASES:
            words, output = _record(case, 0)
            observation = protocol.validate_record(tuple(words), case, 0)
            assert observation["case"] == case.key
            assert len(observation["issues"]) == case.issue_count
            path = root / f"{case.key}.raw"
            path.write_bytes(output)
            parsed = driver.parse_output(path, case, 0)
            assert parsed["case"] == case.key

        case = catalog.CASES_BY_KEY["worker-placement-ct-w01"]
        words, _ = _record(case, 0)
        words[protocol.REC["INSTRUCTION_AFTER"]] += 1
        _expect_failure(words, case, 0, "count")

        for stable_field in ("STABLE_BEFORE", "STABLE_AFTER"):
            words, _ = _record(case, 0)
            words[protocol.REC[stable_field]] &= ~1
            _expect_failure(words, case, 0, "PMU snapshot")

        concurrent = next(
            case
            for case in catalog.CASES
            if case.kind == catalog.Kind.CONCURRENT
        )
        words, _ = _record(concurrent, 0)
        words[protocol.REC["OBSERVER_DONE_BOUNDARY"]] = 0
        _expect_failure(
            words, concurrent, 0, "OBSERVER_DONE_BOUNDARY disagrees"
        )

        words, _ = _record(concurrent, 0)
        words[protocol.REC["TARGET_PENDING_BOUNDARY"]] = 0
        _expect_failure(
            words, concurrent, 0, "TARGET_PENDING_BOUNDARY disagrees"
        )

        words, _ = _record(concurrent, 0)
        words[
            protocol.REC["CONTROL_BOUNDARY"]
            + concurrent.target_worker
        ] = protocol.TASK_DONE
        _expect_failure(
            words, concurrent, 0, "TARGET_PENDING_BOUNDARY disagrees"
        )

        words, _ = _record(concurrent, 0)
        words[
            protocol.REC["FIRST_DONE_CYCLE_BASE"]
            + concurrent.target_worker
        ] = words[protocol.REC["OBSERVER_BOUNDARY_CYCLE"]]
        words[protocol.REC["TARGET_FIRST_DONE_CYCLES"]] = (
            words[protocol.REC["OBSERVER_BOUNDARY_CYCLE"]]
            - words[protocol.REC["START_CYCLE"]]
        )
        _expect_failure(
            words, concurrent, 0, "after the boundary result/guard oracle"
        )

        words, _ = _record(concurrent, 0)
        words[protocol.REC["BOUNDARY_ORACLE_CYCLE"]] = 250
        _expect_failure(
            words, concurrent, 0, "device-cycle window is malformed"
        )

        words, _ = _record(concurrent, 0)
        words[protocol.REC["CONTROL_PER_ISSUE_BASE"]] = (
            protocol.TASK_DONE
        )
        _expect_failure(words, concurrent, 0, "tight burst")

        sentinel_only = next(
            case
            for case in catalog.CASES
            if case.kind == catalog.Kind.SENTINEL_ONLY
        )
        words, _ = _record(sentinel_only, 0)
        words[protocol.REC["ISSUE_SPM_SLOT_BASE"]] = 0
        _expect_failure(words, sentinel_only, 0, "route/address differs")

        words, _ = _record(concurrent, 0)
        words[protocol.REC["OBSERVATION_TIMEOUT_MASK"]] = (
            1 << concurrent.target_worker
        )
        words[protocol.REC["OBSERVATION_TIMEOUT_CYCLE"]] = 350
        _expect_failure(words, concurrent, 0, "polling timed out")

        outstanding = next(
            case
            for case in catalog.CASES
            if case.kind == catalog.Kind.OUTSTANDING
            and case.issue_count == catalog.OUTSTANDING_HIGH_ISSUES
        )
        words, _ = _record(outstanding, 0)
        words[protocol.REC["CONTROL_BEFORE"]] = 1
        _expect_failure(words, outstanding, 0, "not idle before issue")

        words, _ = _record(outstanding, 0)
        words[protocol.REC["CONTROL_PER_ISSUE_BASE"]] = 7
        _expect_failure(words, outstanding, 0, "IB_COUNTER")

        words, _ = _record(outstanding, 0)
        words[
            protocol.REC["FIRST_DONE_CYCLE_BASE"]
            + outstanding.target_worker
        ] = 0
        _expect_failure(words, outstanding, 0, "polling evidence")

        words, _ = _record(outstanding, 0)
        words[protocol.REC["MATCHING_JOIN_MASK"]] = 0
        _expect_failure(words, outstanding, 0, "join participants")

        words, _ = _record(outstanding, 0)
        words[
            protocol.REC["MATCHING_JOIN_RC_BASE"]
            + outstanding.target_worker
        ] = 0
        _expect_failure(
            words, outstanding, 0, "matching join RC/post-control"
        )

        words, _ = _record(outstanding, 0)
        words[
            protocol.REC["MATCHING_JOIN_CONTROL_BASE"]
            + outstanding.target_worker
        ] = 1
        _expect_failure(
            words, outstanding, 0, "matching join RC/post-control"
        )

        words, _ = _record(outstanding, 0)
        words[protocol.REC["RESERVED_BASE"]] = 1
        _expect_failure(words, outstanding, 0, "reserved record")

        partial_case = catalog.CASES_BY_KEY[
            "worker-placement-ct-w01"
        ]
        words = _failure_record(
            partial_case, 0, protocol.STATUS_ISSUE_FAILED
        )
        _expect_failure(
            words, partial_case, 0, "bounded-cleanup-succeeded"
        )

        words = _failure_record(
            partial_case, 0, protocol.STATUS_ISSUE_FAILED
        )
        words[protocol.REC["ATTEMPTED_WORKER_MASK"]] = words[
            protocol.REC["ACCEPTED_WORKER_MASK"]
        ]
        _expect_failure(
            words,
            partial_case,
            0,
            "accepted/attempted cleanup participants differ",
        )

        words = _failure_record(
            partial_case, 0, protocol.STATUS_ISSUE_FAILED
        )
        words[protocol.REC["CLEANUP_ATTEMPT_COUNT"]] = 2
        _expect_failure(
            words, partial_case, 0, "attempted exactly once"
        )

        words = _failure_record(
            partial_case, 0, protocol.STATUS_ISSUE_FAILED
        )
        words[
            protocol.REC["CONTROL_FINAL"] + partial_case.target_worker
        ] = 1
        _expect_failure(
            words, partial_case, 0, "lacks idle final CONTROL"
        )

        words = _failure_record(
            partial_case, 0, protocol.STATUS_ISSUE_FAILED
        )
        words[
            protocol.REC["MATCHING_JOIN_RC_BASE"]
            + partial_case.target_worker
        ] = 1
        _expect_failure(
            words, partial_case, 0, "repeated an unbounded matching wait"
        )

        words = _failure_record(
            partial_case,
            0,
            protocol.STATUS_ISSUE_FAILED,
            poisoned=True,
        )
        _expect_failure(words, partial_case, 0, "(poisoned)")

        words = _failure_record(
            partial_case,
            0,
            protocol.STATUS_ISSUE_FAILED,
            poisoned=True,
        )
        words[protocol.REC["CLEANUP_TIMEOUT_MASK"]] = 0
        _expect_failure(
            words, partial_case, 0, "poisoned cleanup evidence"
        )

        words = _failure_record(
            concurrent, 0, protocol.STATUS_OBSERVATION_TIMEOUT
        )
        _expect_failure(
            words, concurrent, 0, "bounded-cleanup-succeeded"
        )

        words, output = _record(case, 0)
        path = root / "tampered-result.raw"
        first = protocol.issues_for_case(case, 0)[0]
        result_offset = (
            protocol.OUTPUT_SLOT_BASE
            + first.ordinal * protocol.OUTPUT_SLOT_STRIDE
            + protocol.GUARD_BYTES
        )
        output[result_offset] ^= 1
        path.write_bytes(output)
        try:
            driver.parse_output(path, case, 0)
        except RuntimeError as error:
            assert "differs at byte" in str(error)
        else:
            raise AssertionError("tampered full output was accepted")

        _, output = _record(case, 0)
        path = root / "tampered-guard.raw"
        guard_offset = (
            protocol.OUTPUT_SLOT_BASE
            + first.ordinal * protocol.OUTPUT_SLOT_STRIDE
        )
        output[guard_offset] ^= 1
        path.write_bytes(output)
        try:
            driver.parse_output(path, case, 0)
        except RuntimeError as error:
            assert "archive guard changed" in str(error)
        else:
            raise AssertionError("tampered archive guard was accepted")

        _, output = _record(case, 0)
        path = root / "tampered-output-tail.raw"
        output[2500] ^= 1
        path.write_bytes(output)
        try:
            driver.parse_output(path, case, 0)
        except RuntimeError as error:
            assert "output tail canary changed at byte 2500" in str(error)
        else:
            raise AssertionError("tampered output tail was accepted")

        unknown_work = root / "unknown-work"
        unknown_work.mkdir()
        preserved = unknown_work / "preserve"
        preserved.write_text("user-owned\n", encoding="utf-8")
        try:
            driver.write_source_program(unknown_work)
        except RuntimeError as error:
            assert "unknown entries" in str(error)
        else:
            raise AssertionError("unknown worker work-dir content was deleted")
        assert preserved.read_text(encoding="utf-8") == "user-owned\n"

    valid_lifecycle = "\n".join(
        (
            *sorted(driver.LIFECYCLE_LINES),
            *(
                "completion: return_after_local_drain tile_id=" + str(tile_id)
                for tile_id in range(16)
            ),
            "invocation_tiles: 16",
            "physical_tile_domain: 0..15",
        )
    )
    driver.validate_board_lifecycle(valid_lifecycle, "worker")
    for invalid in (
        valid_lifecycle.replace(
            "completion: return_after_local_drain tile_id=3", ""
        ),
        valid_lifecycle.replace(
            "completion: return_after_local_drain tile_id=3",
            "completion: return_after_local_drain tile_id=4",
        ),
        valid_lifecycle + "\ncompletion: return_after_local_drain tile_id=3",
    ):
        try:
            driver.validate_board_lifecycle(invalid, "worker")
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "worker missing/mismatched/duplicate completion was accepted"
            )

    carrier = driver.PROBE_C.read_text(encoding="utf-8")
    for needle in (
        "TsmExecute",
        "wafer_wp_observe_completions",
        "WAFER_WP_REC_CONTROL_PER_ISSUE_BASE",
        "WAFER_WP_REC_FIRST_DONE_CYCLE_BASE",
        "WAFER_WP_REC_MAX_POLL_GAP_BASE",
        "TsmWaitfinish_bywork",
        "WAFER_WP_REC_MATCHING_JOIN_CYCLE_BASE",
        "WAFER_WP_REC_MATCHING_JOIN_RC_BASE",
        "WAFER_WP_REC_MATCHING_JOIN_CONTROL_BASE",
        "WAFER_WP_REC_ISSUE_SPM_SLOT_BASE",
        "WAFER_WP_REC_ISSUE_WRITE_ADDRESS_BASE",
        "WAFER_WP_REC_OBSERVATION_TIMEOUT_MASK",
        "WAFER_WP_OBSERVATION_DEADLINE_CYCLES",
        "wafer_wp_bounded_failure_cleanup",
        "WAFER_WP_REC_CLEANUP_PARTICIPANT_MASK",
        "WAFER_WP_REC_ATTEMPTED_WORKER_MASK",
        "WAFER_WP_REC_CLEANUP_ATTEMPT_COUNT",
        "GR_PMU_CT_BLOCKING_TIME",
        "GR_CSR_CONTROL_ADDR",
        "wafer_wp_archive_outputs",
        "WAFER_WP_RECORD_BOUNDARY_CAPTURED",
        "if (high_before == high_after)",
        "*stable_mask |= bit",
    ):
        assert needle in carrier
    assert "GR_CSR_PRIORITY_ADDR" not in carrier
    cleanup_start = carrier.index(
        "static uint32_t wafer_wp_bounded_failure_cleanup("
    )
    cleanup_end = carrier.index(
        "static void wafer_wp_capture_boundary(", cleanup_start
    )
    cleanup_body = carrier[cleanup_start:cleanup_end]
    assert "TsmWaitfinish" not in cleanup_body
    assert cleanup_body.count(
        "WAFER_WP_REC_CLEANUP_ATTEMPT_COUNT"
    ) >= 2
    failure_write_start = carrier.index(
        "static void wafer_wp_write_launch_failure("
    )
    failure_write_end = carrier.index(
        "static int wafer_wp_decode_u32(", failure_write_start
    )
    failure_write_body = carrier[
        failure_write_start:failure_write_end
    ]
    assert failure_write_body.count(
        "wafer_wp_bounded_failure_cleanup("
    ) == 1
    assert "TsmWaitfinish" not in failure_write_body
    main_start = carrier.index("wafer_tx81_worker_placement_probe(")
    main_body = carrier[main_start:]
    assert main_body.count("wafer_wp_write_launch_failure(") == 3
    assert (
        main_body.rindex("wafer_wp_write_launch_failure(")
        < main_body.index("wafer_wp_confirm_matching_joins(")
    )
    assert "attempted_issue_count = ordinal + 1U" in main_body
    initial_idle = carrier.index(
        "wafer_wp_controls(record, WAFER_WP_REC_CONTROL_BEFORE)",
        main_start,
    )
    seed_prepare = carrier.index(
        "wafer_wp_seed_and_prepare(context, instruction)", main_start
    )
    assert initial_idle < seed_prepare
    progress_submit = carrier.index("if (progress_kind)", main_start)
    instrumented_submit = carrier.index("} else {", progress_submit)
    assert (
        "WAFER_WP_REC_CONTROL_PER_ISSUE_BASE"
        not in carrier[progress_submit:instrumented_submit]
    )
    submitted = carrier.index(
        "WAFER_WP_RECORD_ISSUES_SUBMITTED", main_start
    )
    first_poll = carrier.index(
        "wafer_wp_observe_completions(", submitted
    )
    boundary_capture = carrier.index(
        "wafer_wp_capture_boundary(", first_poll
    )
    boundary_oracle = carrier.index(
        "record[WAFER_WP_REC_BOUNDARY_MISMATCHES]", boundary_capture
    )
    boundary_oracle_cycle = carrier.index(
        "record[WAFER_WP_REC_BOUNDARY_ORACLE_CYCLE]",
        boundary_oracle,
    )
    second_poll = carrier.index(
        "wafer_wp_observe_completions(", boundary_oracle_cycle
    )
    safety_drain = carrier.index(
        "wafer_wp_confirm_matching_joins(", second_poll
    )
    assert (
        first_poll
        < boundary_capture
        < boundary_oracle
        < boundary_oracle_cycle
        < second_poll
        < safety_drain
    )
    board_driver = pathlib.Path(driver.__file__).read_text(encoding="utf-8")
    for lifecycle in driver.LIFECYCLE_LINES:
        assert lifecycle in board_driver
    assert "--no-card" in board_driver
    assert "--board" in board_driver
    assert "no retry, reset, or power" in board_driver
    print("wafer_worker_placement_characterization_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
