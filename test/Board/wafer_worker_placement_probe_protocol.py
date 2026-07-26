#!/usr/bin/env python3
"""Host codec and oracle for the worker-placement probe wire contract."""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections import Counter

import wafer_worker_placement_characterization_catalog as catalog


PROTOCOL_H = (
    pathlib.Path(__file__).resolve().parent
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
SCHEMA = _simple_macro("WAFER_WP_SCHEMA")
REQUEST_WORDS = _simple_macro("WAFER_WP_REQUEST_WORDS")
RECORD_WORDS = _simple_macro("WAFER_WP_RECORD_WORDS")
WORKERS = _simple_macro("WAFER_WP_WORKERS")
QUEUES = _simple_macro("WAFER_WP_QUEUES")
PMU64_COUNTERS = _simple_macro("WAFER_WP_PMU64_COUNTERS")
MAX_ISSUES = _simple_macro("WAFER_WP_MAX_ISSUES")
PLACEMENT_ISSUES = _simple_macro("WAFER_WP_PLACEMENT_ISSUES")
BACKLOG_ISSUES = _simple_macro("WAFER_WP_BACKLOG_ISSUES")
PLACEMENT_BYTES = _simple_macro("WAFER_WP_PLACEMENT_BYTES")
RDMA_BACKLOG_BYTES = _simple_macro("WAFER_WP_RDMA_BACKLOG_BYTES")
NE_RESULT_BYTES = _simple_macro("WAFER_WP_NE_RESULT_BYTES")
SENTINEL_BYTES = _simple_macro("WAFER_WP_SENTINEL_BYTES")
GUARD_BYTES = _simple_macro("WAFER_WP_GUARD_BYTES")
OUTPUT_SLOT_BASE = _simple_macro("WAFER_WP_OUTPUT_SLOT_BASE")
OUTPUT_SLOT_DATA_BYTES = _simple_macro(
    "WAFER_WP_OUTPUT_SLOT_DATA_BYTES"
)
OUTPUT_SLOT_STRIDE = OUTPUT_SLOT_DATA_BYTES + 2 * GUARD_BYTES
RESOURCE_BYTES = OUTPUT_SLOT_BASE + MAX_ISSUES * OUTPUT_SLOT_STRIDE

REQ_NAMES = (
    "MAGIC",
    "SCHEMA_AND_WORDS",
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
    "SCHEMA_AND_WORDS",
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
    "OBSERVER_WAIT_CYCLES",
    "TARGET_WAIT_CYCLES",
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
    "JOIN_CYCLE_BASE",
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
)
REC = {
    name: _enum(f"WAFER_WP_REC_{name}") for name in REC_NAMES
}

STATUS_OK = _enum("WAFER_WP_STATUS_OK")
ALL_FLAGS = sum(
    _enum(name)
    for name in (
        "WAFER_WP_RECORD_BEFORE_CAPTURED",
        "WAFER_WP_RECORD_ISSUES_SUBMITTED",
        "WAFER_WP_RECORD_BOUNDARY_CAPTURED",
        "WAFER_WP_RECORD_SAFETY_DRAINED",
        "WAFER_WP_RECORD_FINAL_CAPTURED",
        "WAFER_WP_RECORD_OUTPUT_ARCHIVED",
    )
)
PMU_STABLE_MASK = (1 << PMU64_COUNTERS) - 1
TASK_DONE = 0x100


@dataclasses.dataclass(frozen=True)
class Issue:
    slot: int
    worker: int
    engine: catalog.Engine
    output_bytes: int


def issues_for_case(
    case: catalog.WorkerCase, sample: int
) -> tuple[Issue, ...]:
    if sample < 0:
        raise ValueError("sample must be nonnegative")
    issues: list[Issue] = []
    if case.kind == catalog.Kind.PLACEMENT:
        remaining = list(case.worker_issues)
        rotation = sample % WORKERS
        while sum(remaining):
            progressed = False
            for offset in range(WORKERS):
                worker = (rotation + offset) % WORKERS
                if remaining[worker]:
                    issues.append(
                        Issue(
                            slot=len(issues),
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
                        slot=len(issues),
                        worker=case.target_worker,
                        engine=case.engine,
                        output_bytes=case.primary_bytes,
                    )
                )
        if case.kind != catalog.Kind.BACKLOG_ONLY:
            issues.append(
                Issue(
                    slot=len(issues),
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
        "SCHEMA_AND_WORDS": (SCHEMA << 32) | REQUEST_WORDS,
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
        "SCHEMA_AND_WORDS": (SCHEMA << 32) | RECORD_WORDS,
        "STATUS": STATUS_OK,
        "FLAGS": ALL_FLAGS,
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
    }
    for name, expected in expected_echo.items():
        if words[REC[name]] != expected:
            raise RuntimeError(
                f"{case.key}: record {name}={words[REC[name]]:#x}, "
                f"expected {expected:#x}"
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
        words[REC["FINAL_CYCLE"]],
    ]
    if cycles != sorted(cycles) or cycles[-1] == cycles[0]:
        raise RuntimeError(f"{case.key}: device-cycle window is malformed")

    observed_issues: list[dict[str, int | str]] = []
    for slot, issue in enumerate(issues):
        worker = words[REC["ISSUE_WORKER_BASE"] + slot]
        engine = words[REC["ISSUE_ENGINE_BASE"] + slot]
        output_bytes = words[REC["ISSUE_BYTES_BASE"] + slot]
        rc = words[REC["ISSUE_RC_BASE"] + slot]
        inter_type = words[REC["ISSUE_INTER_TYPE_BASE"] + slot]
        if (
            worker != issue.worker
            or engine != int(issue.engine)
            or output_bytes != issue.output_bytes
            or rc != 1
            or inter_type & 0xFF != int(issue.engine)
            or (inter_type >> 8) & 0x3 != issue.worker
        ):
            raise RuntimeError(f"{case.key}: slot {slot} route differs")
        expected_offset = (
            OUTPUT_SLOT_BASE
            + slot * OUTPUT_SLOT_STRIDE
            + GUARD_BYTES
        )
        if (
            words[REC["OUTPUT_BYTES_BASE"] + slot]
            != issue.output_bytes
            or words[REC["OUTPUT_OFFSET_BASE"] + slot]
            != expected_offset
            or words[REC["SLOT_MISMATCH_BASE"] + slot]
            or words[REC["SLOT_GUARD_MISMATCH_BASE"] + slot]
        ):
            raise RuntimeError(f"{case.key}: slot {slot} oracle failed")
        observed_issues.append(
            {
                "slot": slot,
                "worker": worker,
                "engine": issue.engine.name.lower(),
                "output_bytes": output_bytes,
                "execute_rc": rc,
            }
        )
    for slot in range(len(issues), MAX_ISSUES):
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
        ):
            if words[REC[base] + slot] != 0:
                raise RuntimeError(
                    f"{case.key}: unused slot {slot} contains {base}"
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
        if case.worker_mask & (1 << worker):
            if not controls_final[worker] & TASK_DONE:
                raise RuntimeError(
                    f"{case.key}: worker {worker} did not finish"
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
            not words[REC["OBSERVER_DONE_BOUNDARY"]]
            or not (
                words[REC["CONTROL_BOUNDARY"] + case.observer_worker]
                & TASK_DONE
            )
        ):
            raise RuntimeError(
                f"{case.key}: observer did not complete at boundary"
            )

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
        "instruction_delta": instruction_delta,
        "blocking_delta": blocking_delta,
        "pmu64_delta": pmu_delta,
        "plan_cycles": cycles[-1] - cycles[0],
        "issue_window_cycles": cycles[1] - cycles[0],
        "observer_wait_cycles": words[REC["OBSERVER_WAIT_CYCLES"]],
        "target_wait_cycles": words[REC["TARGET_WAIT_CYCLES"]],
        "target_pending_at_observer_boundary": bool(
            words[REC["TARGET_PENDING_BOUNDARY"]]
        ),
        "observer_done_at_boundary": bool(
            words[REC["OBSERVER_DONE_BOUNDARY"]]
        ),
        "controls_after_issue": [
            words[REC["CONTROL_AFTER_ISSUE"] + worker]
            for worker in range(WORKERS)
        ],
        "controls_boundary": [
            words[REC["CONTROL_BOUNDARY"] + worker]
            for worker in range(WORKERS)
        ],
        "controls_final": controls_final,
        "join_cycles": [
            words[REC["JOIN_CYCLE_BASE"] + worker]
            for worker in range(WORKERS)
        ],
    }
