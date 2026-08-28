#!/usr/bin/env python3
"""Host codec and oracle for the worker-placement probe wire contract."""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections import Counter

import wafer_worker_placement_characterization_catalog as catalog


PROTOCOL_H = (
    pathlib.Path(__file__).resolve().parent.parent
    / "Inputs"
    / "wafer_worker_placement_probe_protocol.h"
)
_TEXT = PROTOCOL_H.read_text(encoding="utf-8")


def _c_integer(expression: str) -> int:
    expression = re.sub(r"UINT(?:32|64)_C\(([^)]+)\)", r"\1", expression)
    expression = re.sub(
        r"(?<=[0-9a-fA-F])[uUlL]+\b", "", expression
    )
    names = {
        name: value
        for name, value in globals().items()
        if name.startswith("_MACRO_") and isinstance(value, int)
    }
    for name, value in names.items():
        expression = expression.replace(name.removeprefix("_MACRO_"), str(value))
    if re.fullmatch(r"[\s0-9a-fA-FxX()+*<>|_-]+", expression) is None:
        raise RuntimeError(f"unsupported C integer expression {expression!r}")
    return int(eval(expression, {"__builtins__": {}}, {}))


def _raw_macro(name: str) -> str:
    match = re.search(
        rf"^#define\s+{re.escape(name)}\s+(.+?)(?<!\\)\s*$",
        _TEXT,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"missing protocol macro {name}")
    expression = match.group(1).strip()
    while expression.endswith("\\"):
        expression = expression[:-1].strip()
    return expression


def _simple_macro(name: str) -> int:
    return _c_integer(_raw_macro(name))


def _enum(name: str) -> int:
    match = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*([^,]+),?\s*$",
        _TEXT,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"missing protocol enumerator {name}")
    return _c_integer(match.group(1))


REQUEST_MAGIC = _simple_macro("WAFER_WP_REQUEST_MAGIC")
RECORD_MAGIC = _simple_macro("WAFER_WP_RECORD_MAGIC")
REQUEST_GUARD = _simple_macro("WAFER_WP_REQUEST_GUARD")
RECORD_GUARD = _simple_macro("WAFER_WP_RECORD_GUARD")
REQUEST_WORDS = _simple_macro("WAFER_WP_REQUEST_WORDS")
RECORD_WORDS = _simple_macro("WAFER_WP_RECORD_WORDS")
WORKERS = _simple_macro("WAFER_WP_WORKERS")
QUEUES = _simple_macro("WAFER_WP_QUEUES")
PMU64_COUNTERS = _simple_macro("WAFER_WP_PMU64_COUNTERS")
MAX_ISSUES = _simple_macro("WAFER_WP_MAX_ISSUES")
PLACEMENT_ISSUES = _simple_macro("WAFER_WP_PLACEMENT_ISSUES")
BACKLOG_ISSUES = _simple_macro("WAFER_WP_BACKLOG_ISSUES")
OUTSTANDING_LOW_ISSUES = _simple_macro(
    "WAFER_WP_OUTSTANDING_LOW_ISSUES"
)
OUTSTANDING_HIGH_ISSUES = _simple_macro(
    "WAFER_WP_OUTSTANDING_HIGH_ISSUES"
)
PLACEMENT_BYTES = _simple_macro("WAFER_WP_PLACEMENT_BYTES")
RDMA_BACKLOG_BYTES = _simple_macro("WAFER_WP_RDMA_BACKLOG_BYTES")
NE_RESULT_BYTES = _simple_macro("WAFER_WP_NE_RESULT_BYTES")
SENTINEL_BYTES = _simple_macro("WAFER_WP_SENTINEL_BYTES")
LONG_ISSUE_BYTES = _simple_macro("WAFER_WP_LONG_ISSUE_BYTES")
PROGRESS_SENTINEL_SPM_SLOT = _simple_macro(
    "WAFER_WP_PROGRESS_SENTINEL_SPM_SLOT"
)
OBSERVATION_DEADLINE_CYCLES = _simple_macro(
    "WAFER_WP_OBSERVATION_DEADLINE_CYCLES"
)
CLEANUP_DEADLINE_CYCLES = _simple_macro(
    "WAFER_WP_CLEANUP_DEADLINE_CYCLES"
)
WAIT_SUCCESS = _simple_macro("WAFER_WP_WAIT_SUCCESS")
OUTPUT_CANARY = _simple_macro("WAFER_WP_OUTPUT_CANARY")
GUARD_BYTES = _simple_macro("WAFER_WP_GUARD_BYTES")
OUTPUT_SLOT_BASE = _simple_macro("WAFER_WP_OUTPUT_SLOT_BASE")
OUTPUT_SLOT_DATA_BYTES = _simple_macro(
    "WAFER_WP_OUTPUT_SLOT_DATA_BYTES"
)
OUTPUT_SLOT_STRIDE = OUTPUT_SLOT_DATA_BYTES + 2 * GUARD_BYTES
RESOURCE_BYTES = OUTPUT_SLOT_BASE + MAX_ISSUES * OUTPUT_SLOT_STRIDE
SPM_SLOT_BASE = _simple_macro("WAFER_WP_SPM_SLOT_BASE")
SPM_SLOT_STRIDE = _simple_macro("WAFER_WP_SPM_SLOT_STRIDE")
SPM_READ0_OFFSET = _simple_macro("WAFER_WP_SPM_READ0_OFFSET")
SPM_READ1_OFFSET = _simple_macro("WAFER_WP_SPM_READ1_OFFSET")
SPM_WRITE_OFFSET = _simple_macro("WAFER_WP_SPM_WRITE_OFFSET")

REQ_NAMES = (
    "MAGIC",
    "WORD_COUNT",
    "KIND",
    "ENGINE",
    "WORKER_MASK",
    "TARGET_WORKER",
    "OBSERVER_WORKER",
    "SAMPLE",
    "WORKER0_ISSUES",
    "WORKER1_ISSUES",
    "WORKER2_ISSUES",
    "PRIMARY_BYTES",
    "SENTINEL_BYTES",
    "ISSUE_ROTATION",
    "JOIN_ROTATION",
    "RESOURCE_BYTES",
    "GUARD",
    "RESERVED_BASE",
)
REQ = {
    name: _enum(f"WAFER_WP_REQ_{name}") for name in REQ_NAMES
}

REC_NAMES = (
    "MAGIC",
    "WORD_COUNT",
    "STATUS",
    "FLAGS",
    "KIND",
    "ENGINE",
    "WORKER_MASK",
    "TARGET_WORKER",
    "OBSERVER_WORKER",
    "SAMPLE",
    "ISSUE_COUNT",
    "PRIMARY_BYTES",
    "SENTINEL_BYTES",
    "REQUEST_GUARD",
    "RECORD_GUARD",
    "RESOURCE_BYTES",
    "OUTPUT_SLOT_BASE",
    "OUTPUT_SLOT_STRIDE",
    "OUTPUT_GUARD_BYTES",
    "PMU_ENABLE",
    "STABLE_BEFORE",
    "STABLE_AFTER",
    "START_CYCLE",
    "AFTER_ISSUE_CYCLE",
    "OBSERVER_BOUNDARY_CYCLE",
    "FINAL_CYCLE",
    "OBSERVER_FIRST_DONE_CYCLES",
    "TARGET_FIRST_DONE_CYCLES",
    "TARGET_PENDING_BOUNDARY",
    "OBSERVER_DONE_BOUNDARY",
    "BOUNDARY_MISMATCHES",
    "FINAL_MISMATCHES",
    "FINAL_GUARD_MISMATCHES",
    "ISSUE_WORKER_BASE",
    "ISSUE_ENGINE_BASE",
    "ISSUE_BYTES_BASE",
    "ISSUE_RC_BASE",
    "ISSUE_INTER_TYPE_BASE",
    "CONTROL_AFTER_ISSUE",
    "CONTROL_BOUNDARY",
    "CONTROL_FINAL",
    "FIRST_DONE_CYCLE_BASE",
    "PMU64_BEFORE",
    "PMU64_AFTER",
    "INSTRUCTION_BEFORE",
    "INSTRUCTION_AFTER",
    "BLOCKING_BEFORE",
    "BLOCKING_AFTER",
    "OUTPUT_BYTES_BASE",
    "OUTPUT_OFFSET_BASE",
    "SLOT_MISMATCH_BASE",
    "SLOT_GUARD_MISMATCH_BASE",
    "SERIAL_MODE_BASE",
    "CONTROL_BEFORE",
    "CONTROL_PER_ISSUE_BASE",
    "ISSUE_CYCLE_BASE",
    "POLL_COUNT_BASE",
    "MAX_POLL_GAP_BASE",
    "POLL_ROUNDS",
    "MATCHING_JOIN_CYCLE_BASE",
    "MATCHING_JOIN_MASK",
    "ISSUE_ORDINAL_BASE",
    "ISSUE_SPM_SLOT_BASE",
    "ISSUE_READ0_ADDRESS_BASE",
    "ISSUE_READ1_ADDRESS_BASE",
    "ISSUE_WRITE_ADDRESS_BASE",
    "BOUNDARY_ORACLE_CYCLE",
    "OBSERVATION_DEADLINE_CYCLES",
    "OBSERVATION_TIMEOUT_MASK",
    "OBSERVATION_TIMEOUT_CYCLE",
    "MATCHING_JOIN_RC_BASE",
    "MATCHING_JOIN_CONTROL_BASE",
    "ACCEPTED_ISSUE_COUNT",
    "ACCEPTED_WORKER_MASK",
    "ACCEPTED_ISSUES_BY_WORKER_BASE",
    "PENDING_WORKER_MASK",
    "PRIMARY_FAILURE_STATUS",
    "CLEANUP_FLAGS",
    "CLEANUP_START_CYCLE",
    "CLEANUP_END_CYCLE",
    "CLEANUP_DEADLINE_CYCLES",
    "CLEANUP_TIMEOUT_MASK",
    "CLEANUP_ATTEMPT_COUNT",
    "ATTEMPTED_ISSUE_COUNT",
    "ATTEMPTED_WORKER_MASK",
    "CLEANUP_PARTICIPANT_MASK",
    "RESERVED_BASE",
)
REC = {
    name: _enum(f"WAFER_WP_REC_{name}") for name in REC_NAMES
}

STATUS_OK = _enum("WAFER_WP_STATUS_OK")
STATUS_ISSUE_FAILED = _enum("WAFER_WP_STATUS_ISSUE_FAILED")
STATUS_OBSERVATION_TIMEOUT = _enum(
    "WAFER_WP_STATUS_OBSERVATION_TIMEOUT"
)
STATUS_CLEANUP_POISONED = _enum(
    "WAFER_WP_STATUS_CLEANUP_POISONED"
)
CLEANUP_ATTEMPTED = _enum("WAFER_WP_CLEANUP_ATTEMPTED")
CLEANUP_SUCCEEDED = _enum("WAFER_WP_CLEANUP_SUCCEEDED")
CLEANUP_POISONED = _enum("WAFER_WP_CLEANUP_POISONED")
RECORD_FLAG_NAMES = (
    "BEFORE_CAPTURED",
    "ISSUES_SUBMITTED",
    "BOUNDARY_CAPTURED",
    "SAFETY_DRAINED",
    "FINAL_CAPTURED",
    "OUTPUT_ARCHIVED",
    "MATCHING_JOIN_CONFIRMED",
)
RECORD_FLAGS = {
    name: _enum(f"WAFER_WP_RECORD_{name}")
    for name in RECORD_FLAG_NAMES
}
ALL_FLAGS = sum(RECORD_FLAGS.values())
PMU_STABLE_MASK = (1 << PMU64_COUNTERS) - 1
TASK_DONE = 0x100
IB_COUNTER_MASK = 0xFF


@dataclasses.dataclass(frozen=True)
class Issue:
    ordinal: int
    spm_slot: int
    worker: int
    engine: catalog.Engine
    output_bytes: int


def issues_for_case(
    case: catalog.WorkerCase, sample: int
) -> tuple[Issue, ...]:
    if sample < 0:
        raise ValueError("sample must be nonnegative")
    issues: list[Issue] = []
    if case.kind in {
        catalog.Kind.PLACEMENT,
        catalog.Kind.OUTSTANDING,
    }:
        remaining = list(case.worker_issues)
        rotation = sample % WORKERS
        while sum(remaining):
            progressed = False
            for offset in range(WORKERS):
                worker = (rotation + offset) % WORKERS
                if remaining[worker]:
                    issues.append(
                        Issue(
                            ordinal=len(issues),
                            spm_slot=len(issues),
                            worker=worker,
                            engine=case.engine,
                            output_bytes=case.primary_bytes,
                        )
                    )
                    remaining[worker] -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError("worker issue scheduler made no progress")
    else:
        if case.kind != catalog.Kind.SENTINEL_ONLY:
            for _ in range(catalog.BACKLOG_ISSUES):
                issues.append(
                    Issue(
                        ordinal=len(issues),
                        spm_slot=len(issues),
                        worker=case.target_worker,
                        engine=case.engine,
                        output_bytes=case.primary_bytes,
                    )
                )
        if case.kind != catalog.Kind.BACKLOG_ONLY:
            issues.append(
                Issue(
                    ordinal=len(issues),
                    spm_slot=PROGRESS_SENTINEL_SPM_SLOT,
                    worker=case.observer_worker,
                    engine=catalog.Engine.CT,
                    output_bytes=case.sentinel_bytes,
                )
            )
    if len(issues) != case.issue_count or len(issues) > MAX_ISSUES:
        raise RuntimeError(f"{case.key}: issue plan differs from catalog")
    return tuple(issues)


def request_words(
    case: catalog.WorkerCase, sample: int
) -> tuple[int, ...]:
    case.validate()
    issues_for_case(case, sample)
    words = [0] * REQUEST_WORDS
    values = {
        "MAGIC": REQUEST_MAGIC,
        "WORD_COUNT": REQUEST_WORDS,
        "KIND": int(case.kind),
        "ENGINE": int(case.engine),
        "WORKER_MASK": case.worker_mask,
        "TARGET_WORKER": case.target_worker,
        "OBSERVER_WORKER": case.observer_worker,
        "SAMPLE": sample,
        "WORKER0_ISSUES": case.worker_issues[0],
        "WORKER1_ISSUES": case.worker_issues[1],
        "WORKER2_ISSUES": case.worker_issues[2],
        "PRIMARY_BYTES": case.primary_bytes,
        "SENTINEL_BYTES": case.sentinel_bytes,
        "ISSUE_ROTATION": sample % WORKERS,
        "JOIN_ROTATION": (sample + 1) % WORKERS,
        "RESOURCE_BYTES": RESOURCE_BYTES,
        "GUARD": REQUEST_GUARD,
    }
    for name, value in values.items():
        words[REQ[name]] = value
    return tuple(words)


def expected_instruction_counts(
    issues: tuple[Issue, ...],
) -> Counter[tuple[int, int]]:
    return Counter((issue.worker, int(issue.engine)) for issue in issues)


def delta32(after: int, before: int) -> int:
    return (after - before) & 0xFFFFFFFF


def delta64(after: int, before: int) -> int:
    return (after - before) & 0xFFFFFFFFFFFFFFFF


def decode_control(value: int) -> dict[str, int | bool]:
    return {
        "raw": value,
        "ib_counter": value & IB_COUNTER_MASK,
        "task_done": bool(value & TASK_DONE),
    }


def control_is_idle(value: int) -> bool:
    return not (value & IB_COUNTER_MASK) and bool(value & TASK_DONE)


def spm_read0_address(spm_slot: int) -> int:
    return SPM_SLOT_BASE + spm_slot * SPM_SLOT_STRIDE + SPM_READ0_OFFSET


def spm_read1_address(spm_slot: int) -> int:
    return SPM_SLOT_BASE + spm_slot * SPM_SLOT_STRIDE + SPM_READ1_OFFSET


def spm_write_address(spm_slot: int) -> int:
    return SPM_SLOT_BASE + spm_slot * SPM_SLOT_STRIDE + SPM_WRITE_OFFSET


def _validate_launch_failure(
    words: tuple[int, ...],
    case: catalog.WorkerCase,
    issues: tuple[Issue, ...],
) -> None:
    status = words[REC["STATUS"]]
    primary = words[REC["PRIMARY_FAILURE_STATUS"]]
    if primary not in {STATUS_ISSUE_FAILED, STATUS_OBSERVATION_TIMEOUT}:
        raise RuntimeError(
            f"{case.key}: failure record has no qualified primary status"
        )
    if status not in {primary, STATUS_CLEANUP_POISONED}:
        raise RuntimeError(
            f"{case.key}: failure status does not preserve primary failure"
        )

    attempted_count = words[REC["ATTEMPTED_ISSUE_COUNT"]]
    accepted_count = words[REC["ACCEPTED_ISSUE_COUNT"]]
    if not 0 < attempted_count <= len(issues):
        raise RuntimeError(
            f"{case.key}: failure attempted-issue count is malformed"
        )
    issue_rcs = [
        words[REC["ISSUE_RC_BASE"] + ordinal]
        for ordinal in range(MAX_ISSUES)
    ]
    if primary == STATUS_ISSUE_FAILED:
        if (
            issue_rcs[attempted_count - 1] == 1
            or any(rc != 1 for rc in issue_rcs[: attempted_count - 1])
            or any(issue_rcs[attempted_count:])
        ):
            raise RuntimeError(
                f"{case.key}: partial-issue failure prefix is malformed"
            )
    elif (
        attempted_count != len(issues)
        or any(rc != 1 for rc in issue_rcs[:attempted_count])
        or any(issue_rcs[attempted_count:])
    ):
        raise RuntimeError(
            f"{case.key}: observation timeout did not follow full acceptance"
        )

    accepted_ordinals = [
        ordinal
        for ordinal in range(attempted_count)
        if issue_rcs[ordinal] == 1
    ]
    accepted_by_worker = [0] * WORKERS
    for ordinal in accepted_ordinals:
        accepted_by_worker[issues[ordinal].worker] += 1
    accepted_mask = sum(
        1 << worker
        for worker, count in enumerate(accepted_by_worker)
        if count
    )
    attempted_mask = 0
    for issue in issues[:attempted_count]:
        attempted_mask |= 1 << issue.worker
    recorded_accepted_by_worker = [
        words[REC["ACCEPTED_ISSUES_BY_WORKER_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    if (
        accepted_count != len(accepted_ordinals)
        or words[REC["ACCEPTED_WORKER_MASK"]] != accepted_mask
        or recorded_accepted_by_worker != accepted_by_worker
        or words[REC["ATTEMPTED_WORKER_MASK"]] != attempted_mask
        or words[REC["CLEANUP_PARTICIPANT_MASK"]] != attempted_mask
    ):
        raise RuntimeError(
            f"{case.key}: accepted/attempted cleanup participants differ"
        )

    controls_after = [
        words[REC["CONTROL_AFTER_ISSUE"] + worker]
        for worker in range(WORKERS)
    ]
    expected_pending = sum(
        1 << worker
        for worker in range(WORKERS)
        if attempted_mask & (1 << worker)
        and not control_is_idle(controls_after[worker])
    )
    if words[REC["PENDING_WORKER_MASK"]] != expected_pending:
        raise RuntimeError(
            f"{case.key}: pending mask disagrees with post-attempt CONTROL"
        )

    flags = words[REC["FLAGS"]]
    if not flags & RECORD_FLAGS["BEFORE_CAPTURED"]:
        raise RuntimeError(
            f"{case.key}: failure occurred before qualified issue window"
        )
    if flags & (
        RECORD_FLAGS["SAFETY_DRAINED"]
        | RECORD_FLAGS["FINAL_CAPTURED"]
        | RECORD_FLAGS["OUTPUT_ARCHIVED"]
        | RECORD_FLAGS["MATCHING_JOIN_CONFIRMED"]
    ):
        raise RuntimeError(
            f"{case.key}: failure record claims normal completion"
        )
    submitted = bool(flags & RECORD_FLAGS["ISSUES_SUBMITTED"])
    if submitted != (accepted_count == len(issues)):
        raise RuntimeError(
            f"{case.key}: submitted flag disagrees with accepted count"
        )

    observation_timeout_mask = words[
        REC["OBSERVATION_TIMEOUT_MASK"]
    ]
    observation_timeout_cycle = words[
        REC["OBSERVATION_TIMEOUT_CYCLE"]
    ]
    if primary == STATUS_OBSERVATION_TIMEOUT:
        if (
            observation_timeout_mask == 0
            or observation_timeout_mask & ~attempted_mask
            or observation_timeout_cycle
            <= words[REC["AFTER_ISSUE_CYCLE"]]
        ):
            raise RuntimeError(
                f"{case.key}: observation-timeout evidence is malformed"
            )
    elif observation_timeout_mask or observation_timeout_cycle:
        raise RuntimeError(
            f"{case.key}: issue failure contains observation timeout"
        )

    cleanup_flags = words[REC["CLEANUP_FLAGS"]]
    cleanup_attempt_count = words[REC["CLEANUP_ATTEMPT_COUNT"]]
    cleanup_start = words[REC["CLEANUP_START_CYCLE"]]
    cleanup_end = words[REC["CLEANUP_END_CYCLE"]]
    if (
        cleanup_attempt_count != 1
        or not cleanup_flags & CLEANUP_ATTEMPTED
        or cleanup_start == 0
        or cleanup_end < cleanup_start
    ):
        raise RuntimeError(
            f"{case.key}: bounded cleanup was not attempted exactly once"
        )
    succeeded = cleanup_flags == (
        CLEANUP_ATTEMPTED | CLEANUP_SUCCEEDED
    )
    poisoned = cleanup_flags == (
        CLEANUP_ATTEMPTED | CLEANUP_POISONED
    )
    if not (succeeded or poisoned):
        raise RuntimeError(
            f"{case.key}: cleanup success/poison state is ambiguous"
        )
    controls_final = [
        words[REC["CONTROL_FINAL"] + worker]
        for worker in range(WORKERS)
    ]
    if succeeded:
        if (
            status != primary
            or words[REC["CLEANUP_TIMEOUT_MASK"]]
            or any(
                not control_is_idle(controls_final[worker])
                for worker in range(WORKERS)
                if attempted_mask & (1 << worker)
            )
        ):
            raise RuntimeError(
                f"{case.key}: successful cleanup lacks idle final CONTROL"
            )
    else:
        cleanup_timeout_mask = words[REC["CLEANUP_TIMEOUT_MASK"]]
        if (
            status != STATUS_CLEANUP_POISONED
            or cleanup_timeout_mask == 0
            or cleanup_timeout_mask & ~attempted_mask
        ):
            raise RuntimeError(
                f"{case.key}: poisoned cleanup evidence is malformed"
            )

    matching_fields = [
        words[REC["MATCHING_JOIN_MASK"]],
        *(
            words[REC["MATCHING_JOIN_CYCLE_BASE"] + worker]
            for worker in range(WORKERS)
        ),
        *(
            words[REC["MATCHING_JOIN_RC_BASE"] + worker]
            for worker in range(WORKERS)
        ),
        *(
            words[REC["MATCHING_JOIN_CONTROL_BASE"] + worker]
            for worker in range(WORKERS)
        ),
    ]
    if any(matching_fields):
        raise RuntimeError(
            f"{case.key}: failure path repeated an unbounded matching wait"
        )


def validate_record(
    words: tuple[int, ...],
    case: catalog.WorkerCase,
    sample: int,
) -> dict[str, object]:
    if len(words) != RECORD_WORDS:
        raise RuntimeError("worker record has the wrong size")
    issues = issues_for_case(case, sample)
    expected_echo = {
        "MAGIC": RECORD_MAGIC,
        "WORD_COUNT": RECORD_WORDS,
        "KIND": int(case.kind),
        "ENGINE": int(case.engine),
        "WORKER_MASK": case.worker_mask,
        "TARGET_WORKER": case.target_worker,
        "OBSERVER_WORKER": case.observer_worker,
        "SAMPLE": sample,
        "ISSUE_COUNT": len(issues),
        "PRIMARY_BYTES": case.primary_bytes,
        "SENTINEL_BYTES": case.sentinel_bytes,
        "REQUEST_GUARD": REQUEST_GUARD,
        "RECORD_GUARD": RECORD_GUARD,
        "RESOURCE_BYTES": RESOURCE_BYTES,
        "OUTPUT_SLOT_BASE": OUTPUT_SLOT_BASE,
        "OUTPUT_SLOT_STRIDE": OUTPUT_SLOT_STRIDE,
        "OUTPUT_GUARD_BYTES": GUARD_BYTES,
        "OBSERVATION_DEADLINE_CYCLES": OBSERVATION_DEADLINE_CYCLES,
        "CLEANUP_DEADLINE_CYCLES": CLEANUP_DEADLINE_CYCLES,
    }
    for name, expected in expected_echo.items():
        if words[REC[name]] != expected:
            raise RuntimeError(
                f"{case.key}: record {name}={words[REC[name]]:#x}, "
                f"expected {expected:#x}"
            )
    if any(
        words[index]
        for index in range(REC["RESERVED_BASE"], RECORD_WORDS)
    ):
        raise RuntimeError(f"{case.key}: reserved record words are nonzero")
    status = words[REC["STATUS"]]
    if status != STATUS_OK:
        if status in {
            STATUS_ISSUE_FAILED,
            STATUS_OBSERVATION_TIMEOUT,
            STATUS_CLEANUP_POISONED,
        }:
            _validate_launch_failure(words, case, issues)
            cleanup = (
                "poisoned"
                if status == STATUS_CLEANUP_POISONED
                else "bounded-cleanup-succeeded"
            )
            raise RuntimeError(
                f"{case.key}: device launch failed with primary status "
                f"{words[REC['PRIMARY_FAILURE_STATUS']]} ({cleanup})"
            )
        raise RuntimeError(
            f"{case.key}: device launch failed with status {status}"
        )
    if words[REC["FLAGS"]] != ALL_FLAGS:
        raise RuntimeError(
            f"{case.key}: successful record flags are incomplete"
        )
    if (
        words[REC["PMU_ENABLE"]] == 0
        or words[REC["STABLE_BEFORE"]] != PMU_STABLE_MASK
        or words[REC["STABLE_AFTER"]] != PMU_STABLE_MASK
    ):
        raise RuntimeError(f"{case.key}: PMU snapshot is not qualified")
    if any(
        words[REC["SERIAL_MODE_BASE"] + worker] & 1
        for worker in range(WORKERS)
    ):
        raise RuntimeError(f"{case.key}: a worker is in serial mode")
    cycles = [
        words[REC["START_CYCLE"]],
        words[REC["AFTER_ISSUE_CYCLE"]],
        words[REC["OBSERVER_BOUNDARY_CYCLE"]],
        words[REC["BOUNDARY_ORACLE_CYCLE"]],
        words[REC["FINAL_CYCLE"]],
    ]
    if cycles != sorted(cycles) or cycles[-1] == cycles[0]:
        raise RuntimeError(f"{case.key}: device-cycle window is malformed")
    if (
        words[REC["OBSERVATION_TIMEOUT_MASK"]]
        or words[REC["OBSERVATION_TIMEOUT_CYCLE"]]
    ):
        raise RuntimeError(
            f"{case.key}: device task-done polling timed out"
        )

    controls_before = [
        words[REC["CONTROL_BEFORE"] + worker]
        for worker in range(WORKERS)
    ]
    if any(
        (control & IB_COUNTER_MASK) != 0 or not (control & TASK_DONE)
        for control in controls_before
    ):
        raise RuntimeError(
            f"{case.key}: worker CONTROL was not idle before issue"
        )

    observed_issues: list[dict[str, int | str]] = []
    submitted_by_worker = [0] * WORKERS
    per_issue_controls: list[dict[str, object]] = []
    previous_issue_cycle = cycles[0]
    progress_kind = case.kind in {
        catalog.Kind.BACKLOG_ONLY,
        catalog.Kind.SENTINEL_ONLY,
        catalog.Kind.CONCURRENT,
    }
    for ordinal, issue in enumerate(issues):
        route_worker = words[REC["ISSUE_WORKER_BASE"] + ordinal]
        engine = words[REC["ISSUE_ENGINE_BASE"] + ordinal]
        output_bytes = words[REC["ISSUE_BYTES_BASE"] + ordinal]
        rc = words[REC["ISSUE_RC_BASE"] + ordinal]
        inter_type = words[REC["ISSUE_INTER_TYPE_BASE"] + ordinal]
        recorded_ordinal = words[REC["ISSUE_ORDINAL_BASE"] + ordinal]
        recorded_spm_slot = words[REC["ISSUE_SPM_SLOT_BASE"] + ordinal]
        read0_address = words[
            REC["ISSUE_READ0_ADDRESS_BASE"] + ordinal
        ]
        read1_address = words[
            REC["ISSUE_READ1_ADDRESS_BASE"] + ordinal
        ]
        write_address = words[
            REC["ISSUE_WRITE_ADDRESS_BASE"] + ordinal
        ]
        if (
            route_worker != issue.worker
            or engine != int(issue.engine)
            or output_bytes != issue.output_bytes
            or rc != 1
            or inter_type & 0xFF != int(issue.engine)
            or (inter_type >> 8) & 0x3 != issue.worker
            or recorded_ordinal != issue.ordinal
            or recorded_spm_slot != issue.spm_slot
            or read0_address != spm_read0_address(issue.spm_slot)
            or read1_address != spm_read1_address(issue.spm_slot)
            or write_address != spm_write_address(issue.spm_slot)
        ):
            raise RuntimeError(
                f"{case.key}: issue ordinal {ordinal} route/address differs"
            )
        expected_offset = (
            OUTPUT_SLOT_BASE
            + ordinal * OUTPUT_SLOT_STRIDE
            + GUARD_BYTES
        )
        if (
            words[REC["OUTPUT_BYTES_BASE"] + ordinal]
            != issue.output_bytes
            or words[REC["OUTPUT_OFFSET_BASE"] + ordinal]
            != expected_offset
            or words[REC["SLOT_MISMATCH_BASE"] + ordinal]
            or words[REC["SLOT_GUARD_MISMATCH_BASE"] + ordinal]
        ):
            raise RuntimeError(
                f"{case.key}: issue ordinal {ordinal} oracle failed"
            )
        issue_cycle = words[REC["ISSUE_CYCLE_BASE"] + ordinal]
        submitted_by_worker[issue.worker] += 1
        controls = [
            words[
                REC["CONTROL_PER_ISSUE_BASE"]
                + ordinal * WORKERS
                + worker
            ]
            for worker in range(WORKERS)
        ]
        if progress_kind:
            if issue_cycle or any(controls):
                raise RuntimeError(
                    f"{case.key}: progress issue {ordinal} was not "
                    "submitted as a tight burst"
                )
        else:
            if (
                issue_cycle < previous_issue_cycle
                or issue_cycle > cycles[1]
            ):
                raise RuntimeError(
                    f"{case.key}: issue ordinal {ordinal} cycle is "
                    "malformed"
                )
            previous_issue_cycle = issue_cycle
            for sampled_worker, control in enumerate(controls):
                if (
                    (control & IB_COUNTER_MASK)
                    > submitted_by_worker[sampled_worker]
                ):
                    raise RuntimeError(
                        f"{case.key}: issue ordinal {ordinal} worker "
                        f"{sampled_worker} IB_COUNTER exceeds submitted work"
                    )
            per_issue_controls.append(
                {
                    "ordinal": ordinal,
                    "cycle": issue_cycle,
                    "workers": [
                        decode_control(control) for control in controls
                    ],
                }
            )
        observed_issues.append(
            {
                "ordinal": ordinal,
                "spm_slot": recorded_spm_slot,
                "spm_read0_address": read0_address,
                "spm_read1_address": read1_address,
                "spm_write_address": write_address,
                "worker": route_worker,
                "engine": issue.engine.name.lower(),
                "output_bytes": output_bytes,
                "execute_rc": rc,
            }
        )
    for ordinal in range(len(issues), MAX_ISSUES):
        for base in (
            "ISSUE_WORKER_BASE",
            "ISSUE_ENGINE_BASE",
            "ISSUE_BYTES_BASE",
            "ISSUE_RC_BASE",
            "ISSUE_INTER_TYPE_BASE",
            "OUTPUT_BYTES_BASE",
            "OUTPUT_OFFSET_BASE",
            "SLOT_MISMATCH_BASE",
            "SLOT_GUARD_MISMATCH_BASE",
            "ISSUE_CYCLE_BASE",
            "ISSUE_ORDINAL_BASE",
            "ISSUE_SPM_SLOT_BASE",
            "ISSUE_READ0_ADDRESS_BASE",
            "ISSUE_READ1_ADDRESS_BASE",
            "ISSUE_WRITE_ADDRESS_BASE",
        ):
            if words[REC[base] + ordinal] != 0:
                raise RuntimeError(
                    f"{case.key}: unused issue ordinal {ordinal} "
                    f"contains {base}"
                )
        for worker in range(WORKERS):
            if (
                words[
                    REC["CONTROL_PER_ISSUE_BASE"]
                    + ordinal * WORKERS
                    + worker
                ]
                != 0
            ):
                raise RuntimeError(
                    f"{case.key}: unused issue ordinal {ordinal} "
                    "contains CONTROL"
                )
    controls_after_issue = [
        words[REC["CONTROL_AFTER_ISSUE"] + worker]
        for worker in range(WORKERS)
    ]
    for worker, control in enumerate(controls_after_issue):
        if (
            control & IB_COUNTER_MASK
        ) > submitted_by_worker[worker]:
            raise RuntimeError(
                f"{case.key}: worker {worker} post-submit IB_COUNTER "
                "exceeds submitted work"
            )
    expected_pending = sum(
        1 << worker
        for worker, control in enumerate(controls_after_issue)
        if case.worker_mask & (1 << worker)
        and not control_is_idle(control)
    )
    accepted_by_worker = [
        words[REC["ACCEPTED_ISSUES_BY_WORKER_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    if (
        words[REC["ACCEPTED_ISSUE_COUNT"]] != len(issues)
        or words[REC["ACCEPTED_WORKER_MASK"]] != case.worker_mask
        or accepted_by_worker != list(case.worker_issues)
        or words[REC["ATTEMPTED_ISSUE_COUNT"]] != len(issues)
        or words[REC["ATTEMPTED_WORKER_MASK"]] != case.worker_mask
        or words[REC["PENDING_WORKER_MASK"]] != expected_pending
    ):
        raise RuntimeError(
            f"{case.key}: successful accepted/pending state differs"
        )
    for name in (
        "PRIMARY_FAILURE_STATUS",
        "CLEANUP_FLAGS",
        "CLEANUP_START_CYCLE",
        "CLEANUP_END_CYCLE",
        "CLEANUP_TIMEOUT_MASK",
        "CLEANUP_ATTEMPT_COUNT",
        "CLEANUP_PARTICIPANT_MASK",
    ):
        if words[REC[name]]:
            raise RuntimeError(
                f"{case.key}: successful record contains failure cleanup"
            )

    expected_counts = expected_instruction_counts(issues)
    instruction_delta: dict[str, int] = {}
    blocking_delta: dict[str, int] = {}
    for worker in range(WORKERS):
        for queue in range(QUEUES):
            index = worker * QUEUES + queue
            actual = delta32(
                words[REC["INSTRUCTION_AFTER"] + index],
                words[REC["INSTRUCTION_BEFORE"] + index],
            )
            expected = expected_counts[(worker, queue)]
            if actual != expected:
                raise RuntimeError(
                    f"{case.key}: worker{worker}.queue{queue} count "
                    f"{actual}, expected {expected}"
                )
            instruction_delta[f"worker{worker}.queue{queue}"] = actual
            blocking_delta[f"worker{worker}.queue{queue}"] = delta32(
                words[REC["BLOCKING_AFTER"] + index],
                words[REC["BLOCKING_BEFORE"] + index],
            )

    controls_final = [
        words[REC["CONTROL_FINAL"] + worker]
        for worker in range(WORKERS)
    ]
    for worker in range(WORKERS):
        if (
            not controls_final[worker] & TASK_DONE
            or controls_final[worker] & IB_COUNTER_MASK
        ):
            raise RuntimeError(
                f"{case.key}: worker {worker} was not idle after join"
            )
    controls_boundary = [
        words[REC["CONTROL_BOUNDARY"] + worker]
        for worker in range(WORKERS)
    ]
    raw_target_pending = not control_is_idle(
        controls_boundary[case.target_worker]
    )
    recorded_target_pending = bool(
        words[REC["TARGET_PENDING_BOUNDARY"]]
    )
    if recorded_target_pending != raw_target_pending:
        raise RuntimeError(
            f"{case.key}: TARGET_PENDING_BOUNDARY disagrees with raw "
            "CONTROL_BOUNDARY[target]"
        )
    raw_observer_done = control_is_idle(
        controls_boundary[case.observer_worker]
    )
    if bool(words[REC["OBSERVER_DONE_BOUNDARY"]]) != raw_observer_done:
        raise RuntimeError(
            f"{case.key}: OBSERVER_DONE_BOUNDARY disagrees with raw "
            "CONTROL_BOUNDARY[observer]"
        )
    boundary_mask = (
        1 << case.observer_worker
        if case.kind
        in {catalog.Kind.SENTINEL_ONLY, catalog.Kind.CONCURRENT}
        else case.worker_mask
    )
    for worker in range(WORKERS):
        if (
            boundary_mask & (1 << worker)
            and (
                not controls_boundary[worker] & TASK_DONE
                or controls_boundary[worker] & IB_COUNTER_MASK
            )
        ):
            raise RuntimeError(
                f"{case.key}: boundary worker {worker} was not done"
            )
    if (
        words[REC["BOUNDARY_MISMATCHES"]]
        or words[REC["FINAL_MISMATCHES"]]
        or words[REC["FINAL_GUARD_MISMATCHES"]]
    ):
        raise RuntimeError(f"{case.key}: device-side oracle failed")
    if case.kind in {
        catalog.Kind.SENTINEL_ONLY,
        catalog.Kind.CONCURRENT,
    }:
        if (
            not raw_observer_done
        ):
            raise RuntimeError(
                f"{case.key}: observer did not complete at boundary"
            )

    first_done_cycles = [
        words[REC["FIRST_DONE_CYCLE_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    poll_counts = [
        words[REC["POLL_COUNT_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    max_poll_gaps = [
        words[REC["MAX_POLL_GAP_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    if words[REC["POLL_ROUNDS"]] == 0:
        raise RuntimeError(f"{case.key}: task-done polling did not run")
    poll_rounds = words[REC["POLL_ROUNDS"]]
    for worker in range(WORKERS):
        participating = bool(case.worker_mask & (1 << worker))
        if participating:
            if (
                first_done_cycles[worker] < cycles[1]
                or first_done_cycles[worker] > cycles[-1]
                or poll_counts[worker] == 0
                or poll_counts[worker] > poll_rounds
                or max_poll_gaps[worker] > cycles[-1] - cycles[0]
            ):
                raise RuntimeError(
                    f"{case.key}: worker {worker} polling evidence is "
                    "malformed"
                )
        elif (
            first_done_cycles[worker]
            or poll_counts[worker]
            or max_poll_gaps[worker]
        ):
            raise RuntimeError(
                f"{case.key}: inactive worker {worker} has polling evidence"
            )
    if (
        words[REC["OBSERVER_FIRST_DONE_CYCLES"]]
        != (
            first_done_cycles[case.observer_worker] - cycles[0]
            if case.worker_mask & (1 << case.observer_worker)
            else 0
        )
        or words[REC["TARGET_FIRST_DONE_CYCLES"]]
        != (
            first_done_cycles[case.target_worker] - cycles[0]
            if case.worker_mask & (1 << case.target_worker)
            else 0
        )
    ):
        raise RuntimeError(
            f"{case.key}: first-observed completion delta differs"
        )
    if (
        case.kind == catalog.Kind.CONCURRENT
        and raw_target_pending
        and first_done_cycles[case.target_worker] <= cycles[3]
    ):
        raise RuntimeError(
            f"{case.key}: target completion was not observed after the "
            "boundary result/guard oracle"
        )

    matching_join_cycles = [
        words[REC["MATCHING_JOIN_CYCLE_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    matching_join_rcs = [
        words[REC["MATCHING_JOIN_RC_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    matching_join_controls = [
        words[REC["MATCHING_JOIN_CONTROL_BASE"] + worker]
        for worker in range(WORKERS)
    ]
    if words[REC["MATCHING_JOIN_MASK"]] != case.worker_mask:
        raise RuntimeError(
            f"{case.key}: matching join participants differ"
        )
    for worker, join_cycles in enumerate(matching_join_cycles):
        participating = bool(case.worker_mask & (1 << worker))
        if not participating and (
            join_cycles
            or matching_join_rcs[worker]
            or matching_join_controls[worker]
        ):
            raise RuntimeError(
                f"{case.key}: inactive worker {worker} has matching join "
                "evidence"
            )
        if participating and (
            join_cycles > cycles[-1] - cycles[0]
            or matching_join_rcs[worker] != WAIT_SUCCESS
            or not control_is_idle(matching_join_controls[worker])
        ):
            raise RuntimeError(
                f"{case.key}: worker {worker} matching join RC/post-control "
                "is malformed"
            )
    observed_control_rows = [
        controls_after_issue,
        controls_boundary,
    ]
    observed_control_rows.extend(
        [
            int(worker["raw"])
            for worker in row["workers"]
        ]
        for row in per_issue_controls
    )
    observed_max_ib = [
        max(
            int(row[worker]) & IB_COUNTER_MASK
            for row in observed_control_rows
        )
        for worker in range(WORKERS)
    ]
    pmu_delta = [
        delta64(
            words[REC["PMU64_AFTER"] + index],
            words[REC["PMU64_BEFORE"] + index],
        )
        for index in range(PMU64_COUNTERS)
    ]
    return {
        "case": case.key,
        "sample": sample,
        "issues": observed_issues,
        "submitted_issue_count": len(issues),
        "submitted_issues_by_worker": submitted_by_worker,
        "observed_max_ib_counter_by_worker": observed_max_ib,
        "instruction_delta": instruction_delta,
        "blocking_delta": blocking_delta,
        "pmu64_delta": pmu_delta,
        "plan_cycles": cycles[-1] - cycles[0],
        "issue_window_cycles": cycles[1] - cycles[0],
        "observer_first_observed_done_cycles": words[
            REC["OBSERVER_FIRST_DONE_CYCLES"]
        ],
        "target_first_observed_done_cycles": words[
            REC["TARGET_FIRST_DONE_CYCLES"]
        ],
        "target_pending_at_observer_boundary": bool(
            words[REC["TARGET_PENDING_BOUNDARY"]]
        ),
        "observer_done_at_boundary": bool(
            words[REC["OBSERVER_DONE_BOUNDARY"]]
        ),
        "controls_before": [
            decode_control(control) for control in controls_before
        ],
        "controls_per_issue": per_issue_controls,
        "controls_after_issue": [
            decode_control(control) for control in controls_after_issue
        ],
        "controls_boundary": [
            decode_control(control) for control in controls_boundary
        ],
        "controls_final": [
            decode_control(control) for control in controls_final
        ],
        "first_observed_completion_cycles": [
            cycle - cycles[0] if cycle else 0
            for cycle in first_done_cycles
        ],
        "task_done_poll_counts": poll_counts,
        "task_done_max_poll_gaps": max_poll_gaps,
        "task_done_poll_rounds": poll_rounds,
        "matching_join_cycles": matching_join_cycles,
        "matching_join_rcs": matching_join_rcs,
        "matching_join_controls": [
            decode_control(control)
            if case.worker_mask & (1 << worker)
            else None
            for worker, control in enumerate(matching_join_controls)
        ],
        "observation_deadline_cycles": words[
            REC["OBSERVATION_DEADLINE_CYCLES"]
        ],
        "boundary_oracle_cycle": cycles[3] - cycles[0],
    }
