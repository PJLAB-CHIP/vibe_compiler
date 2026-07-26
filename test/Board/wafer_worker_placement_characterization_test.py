#!/usr/bin/env python3
"""Host tests for the real worker placement/progress board adapter."""

from __future__ import annotations

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
        "SCHEMA_AND_WORDS": (
            protocol.SCHEMA << 32
        ) | protocol.RECORD_WORDS,
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
        "FINAL_CYCLE": 400,
        "OBSERVER_WAIT_CYCLES": 17,
        "TARGET_WAIT_CYCLES": 31,
        "TARGET_PENDING_BOUNDARY": int(
            case.kind == catalog.Kind.CONCURRENT
        ),
        "OBSERVER_DONE_BOUNDARY": int(
            case.kind
            in {catalog.Kind.SENTINEL_ONLY, catalog.Kind.CONCURRENT}
        ),
        "BOUNDARY_MISMATCHES": 0,
        "FINAL_MISMATCHES": 0,
        "FINAL_GUARD_MISMATCHES": 0,
    }
    for name, value in values.items():
        words[protocol.REC[name]] = value
    expected_counts = protocol.expected_instruction_counts(issues)
    for worker in range(protocol.WORKERS):
        if case.worker_mask & (1 << worker):
            words[protocol.REC["CONTROL_FINAL"] + worker] = (
                protocol.TASK_DONE
            )
        if (
            case.kind
            in {catalog.Kind.SENTINEL_ONLY, catalog.Kind.CONCURRENT}
            and worker == case.observer_worker
        ):
            words[protocol.REC["CONTROL_BOUNDARY"] + worker] = (
                protocol.TASK_DONE
            )
        if (
            case.kind == catalog.Kind.CONCURRENT
            and worker == case.target_worker
        ):
            words[protocol.REC["CONTROL_BOUNDARY"] + worker] = 0
        for queue in range(protocol.QUEUES):
            index = worker * protocol.QUEUES + queue
            words[protocol.REC["INSTRUCTION_AFTER"] + index] = (
                expected_counts[(worker, queue)]
            )
            words[protocol.REC["BLOCKING_AFTER"] + index] = (
                worker * 10 + queue
            )
    for index in range(protocol.PMU64_COUNTERS):
        words[protocol.REC["PMU64_BEFORE"] + index] = 1000 + index
        words[protocol.REC["PMU64_AFTER"] + index] = 1100 + index

    output = bytearray(protocol.RESOURCE_BYTES)
    for issue in issues:
        words[protocol.REC["ISSUE_WORKER_BASE"] + issue.slot] = issue.worker
        words[protocol.REC["ISSUE_ENGINE_BASE"] + issue.slot] = int(
            issue.engine
        )
        words[protocol.REC["ISSUE_BYTES_BASE"] + issue.slot] = (
            issue.output_bytes
        )
        words[protocol.REC["ISSUE_RC_BASE"] + issue.slot] = 1
        words[protocol.REC["ISSUE_INTER_TYPE_BASE"] + issue.slot] = (
            int(issue.engine) | (issue.worker << 8)
        )
        output_offset = (
            protocol.OUTPUT_SLOT_BASE
            + issue.slot * protocol.OUTPUT_SLOT_STRIDE
            + protocol.GUARD_BYTES
        )
        words[protocol.REC["OUTPUT_BYTES_BASE"] + issue.slot] = (
            issue.output_bytes
        )
        words[protocol.REC["OUTPUT_OFFSET_BASE"] + issue.slot] = output_offset
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
    assert len(catalog.CASES) == 32
    assert len(catalog.GROUPS) == 8
    assert len(
        [case for case in catalog.CASES if case.kind == catalog.Kind.PLACEMENT]
    ) == 14
    assert len(
        [
            case
            for case in catalog.CASES
            if case.kind != catalog.Kind.PLACEMENT
        ]
    ) == 18
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
        else:
            assert {case.kind for case in cases} == {
                catalog.Kind.BACKLOG_ONLY,
                catalog.Kind.SENTINEL_ONLY,
                catalog.Kind.CONCURRENT,
            }
    for group in (
        "worker-placement-ct",
        "worker-progress-ne-target-w0-observer-w1",
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
            if sample & 1:
                expected = tuple(reversed(expected))
            expected_orders.append(expected)
        assert actual_orders == expected_orders
        assert all(set(order) == set(cases) for order in actual_orders)
        assert len(set(actual_orders)) == catalog.REPEATS
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
        _expect_failure(words, concurrent, 0, "observer")

        words, output = _record(case, 0)
        path = root / "tampered-result.raw"
        first = protocol.issues_for_case(case, 0)[0]
        result_offset = (
            protocol.OUTPUT_SLOT_BASE
            + first.slot * protocol.OUTPUT_SLOT_STRIDE
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
            + first.slot * protocol.OUTPUT_SLOT_STRIDE
        )
        output[guard_offset] ^= 1
        path.write_bytes(output)
        try:
            driver.parse_output(path, case, 0)
        except RuntimeError as error:
            assert "archive guard changed" in str(error)
        else:
            raise AssertionError("tampered archive guard was accepted")

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

    terminal_completion = 3
    valid_lifecycle = "\n".join(
        (
            *sorted(driver.LIFECYCLE_LINES),
            "terminal_completion: 3 kind=entry_return",
        )
    )
    driver.validate_board_lifecycle(
        valid_lifecycle, terminal_completion, "worker"
    )
    for invalid in (
        valid_lifecycle.replace("terminal_completion: 3", ""),
        valid_lifecycle.replace(
            "terminal_completion: 3", "terminal_completion: 4"
        ),
        valid_lifecycle + "\nterminal_completion: 3 kind=entry_return",
    ):
        try:
            driver.validate_board_lifecycle(
                invalid, terminal_completion, "worker"
            )
        except RuntimeError as error:
            assert "matching terminal-completion" in str(error)
        else:
            raise AssertionError(
                "worker missing/mismatched/duplicate terminal was accepted"
            )

    carrier = driver.PROBE_C.read_text(encoding="utf-8")
    for needle in (
        "TsmExecute",
        "TsmWaitfinish_bywork",
        "GR_PMU_CT_BLOCKING_TIME",
        "GR_CSR_CONTROL_ADDR",
        "wafer_wp_archive_outputs",
        "WAFER_WP_RECORD_BOUNDARY_CAPTURED",
        "if (high_before == high_after)",
        "*stable_mask |= bit",
    ):
        assert needle in carrier
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
