#!/usr/bin/env python3
"""DTE/SPM transport-PMU payload-sweep and fail-closed TMNOC contract."""

from __future__ import annotations

import dataclasses
import statistics


PAYLOAD_SWEEP = (16, 32, 64, 256, 4096)
CALIBRATION_PAYLOADS = (16,)
HELD_OUT_PAYLOADS = (32, 64, 256, 4096)
MODE_NAMES = {
    1: "ncc-producer-local-drain-dte",
    2: "dte-recv-wait-ncc-consumer",
    3: "disjoint-local-wait-first",
    4: "disjoint-dte-wait-first",
    5: "dte-source-destination-reuse-after-events",
    6: "dte-two-destination-broadcast",
    7: "dte-source-reuse-before-send-event-error",
    8: "dte-invalid-fsm-error",
    9: "dte-wait-unknown-event-error",
    10: "dte-sender-raw-async-serial-control",
    11: "dte-sender-raw-async-window",
    12: "dte-destination-reuse-before-receive-event-error",
}
TRANSPORT_PMU_MODES = tuple(range(1, 7))
ERROR_PATH_MODES = (7, 8, 9, 12)
ASYNC_SENDER_MODES = (10, 11)
ASYNC_SENDER_PAYLOAD_BYTES = 64
ASYNC_SENDER_TRANSPORT_BYTES = 65536
ASYNC_SENDER_REPETITIONS = 3
MODE_PAYLOADS = {
    **{mode: PAYLOAD_SWEEP for mode in TRANSPORT_PMU_MODES},
    **{mode: (16,) for mode in ERROR_PATH_MODES},
    **{mode: (ASYNC_SENDER_PAYLOAD_BYTES,) for mode in ASYNC_SENDER_MODES},
}
BOARD_COUNTER_NAMES = (
    "dte_channel0_transfer",
    "dte_channel1_transfer",
    "dte_channel0_execution",
    "dte_channel1_execution",
    "spm_dte_t2_port8",
    "spm_dte_t3_port8",
)
SPLIT_COUNTER_STABLE_RETRIES = 8


@dataclasses.dataclass(frozen=True)
class TransportPmuCase:
    case_id: int
    mode: int
    payload_bytes: int
    split: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "mode": self.mode,
            "name": MODE_NAMES[self.mode],
            "payload_bytes": self.payload_bytes,
            "split": self.split,
            "ranks": 16,
            "oracle": (
                "full-card exact payload+inactive suffix poison+SPM/DDR "
                "guards+terminal+NCC instruction counts"
            ),
            "pmu_gate": (
                "read-only stable split counters+unchanged enable scope+"
                "raw modulo-2^64 deltas"
            ),
            "disposition": "board-executable",
        }


CASES = tuple(
    TransportPmuCase(
        case_id=index,
        mode=mode,
        payload_bytes=payload_bytes,
        split=(
            "calibration"
            if payload_bytes in CALIBRATION_PAYLOADS
            else "held-out"
        ),
    )
    for index, (payload_bytes, mode) in enumerate(
        (payload_bytes, mode)
        for payload_bytes in PAYLOAD_SWEEP
        for mode in TRANSPORT_PMU_MODES
    )
)


@dataclasses.dataclass(frozen=True)
class TransportPmuObservation:
    case_id: int
    name: str
    source_case: str
    mode: int
    payload_bytes: int
    split: str
    disposition: str = "board-observation"
    oracle: str = (
        "raw modulo-2^64 counter deltas across the "
        "16/32/64/256/4096-byte sweep"
    )
    reason: str = (
        "the counter measurement basis and event units remain uncalibrated"
    )


TRANSPORT_PMU_OBSERVATIONS = tuple(
    TransportPmuObservation(
        case.case_id,
        f"dte-spm-pmu-{MODE_NAMES[case.mode]}-{case.payload_bytes}b",
        MODE_NAMES[case.mode],
        case.mode,
        case.payload_bytes,
        case.split,
    )
    for case in CASES
)


@dataclasses.dataclass(frozen=True)
class TransportContractCase:
    name: str
    domain: str
    disposition: str
    verification_scope: str
    gate: str
    reason: str
    mode: int | None = None
    payload_bytes: int | None = None
    transport_bytes: int | None = None
    repetitions: int = 1

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


CONTRACT_CASES = (
    TransportContractCase(
        "dte-source-reuse-before-send-event",
        "direct-dte-reuse",
        "board-observation",
        "board-device-error-path",
        (
            "record sender-active rejection, transport-error status, completed "
            "first send/receive, exact payload, guards, and normal cleanup"
        ),
        (
            "the duplicate prepare is rejected synchronously; the first "
            "prepared transfer is still completed before runtime success"
        ),
        7,
        16,
    ),
    TransportContractCase(
        "dte-destination-reuse-before-receive-event",
        "direct-dte-reuse",
        "board-observation",
        "board-device-error-path",
        (
            "record receiver-active rejection, transport-error status, "
            "completed original send/receive, exact first destination, "
            "untouched rejected destination, guards, and normal cleanup"
        ),
        (
            "the duplicate receive prepare is rejected synchronously; the "
            "original receiver and its matching send still complete before "
            "runtime success"
        ),
        12,
        16,
    ),
    TransportContractCase(
        "dte-receiver-unprepared",
        "direct-dte-receiver",
        "isolated-deferred",
        "isolated-safety-quarantine",
        "never submit a sender wait without its matching receive prepare",
        (
            "direct_sync_wait has no bounded device timeout and can "
            "permanently block the card session"
        ),
    ),
    TransportContractCase(
        "dte-invalid-fsm",
        "direct-dte-routing",
        "board-observation",
        "board-device-error-path",
        (
            "record rejected send/receive tokens, transport-error status, "
            "untouched payload/guards, terminal completion, and cleanup"
        ),
        "both typed uint32 FSM guards return before transport attachment",
        8,
        16,
    ),
    TransportContractCase(
        "dte-invalid-coordinate",
        "direct-dte-routing",
        "static-negative",
        "host-prelaunch-verifier",
        "typed IR/host mesh validation rejects out-of-mesh coordinates",
        "a coordinate outside the qualified 16-tile mesh is never submitted",
    ),
    TransportContractCase(
        "dte-wait-unknown-event",
        "direct-dte-completion",
        "board-observation",
        "board-device-error-path",
        (
            "record unknown-event return, transport-error status, untouched "
            "payload/guards, terminal completion, and cleanup"
        ),
        "the event decoder rejects an unknown token before any transport wait",
        9,
        16,
    ),
    TransportContractCase(
        "dte-sender-raw-async-serial-control",
        "direct-dte-sender-overlap",
        "board-observation",
        "board-device-receiver-first-raw-async",
        (
            "receiver prepare; sender ready wait; attach/send_async/wait_done/"
            "release; then disjoint CT; record all three return codes; exact "
            "64KiB receive/CT, guards, receiver completion, and terminal"
        ),
        (
            "test-only raw control keeps production CRT unchanged and supplies "
            "the serial half of the sender/NCC comparison"
        ),
        10,
        ASYNC_SENDER_PAYLOAD_BYTES,
        ASYNC_SENDER_TRANSPORT_BYTES,
        ASYNC_SENDER_REPETITIONS,
    ),
    TransportContractCase(
        "dte-sender-raw-async-window",
        "direct-dte-sender-overlap",
        "board-observation",
        "board-device-receiver-first-raw-async",
        (
            "receiver prepare; sender ready wait; attach/send_async; issue "
            "disjoint CT before wait_done/release; record all three return "
            "codes; exact 64KiB receive/CT, guards, receiver completion, and "
            "terminal"
        ),
        (
            "the issue window is observable without claiming temporal overlap; "
            "raw PMU remains profile-scoped until repeated board comparison"
        ),
        11,
        ASYNC_SENDER_PAYLOAD_BYTES,
        ASYNC_SENDER_TRANSPORT_BYTES,
        ASYNC_SENDER_REPETITIONS,
    ),
    TransportContractCase(
        "host-readback-before-terminal",
        "runtime-publication",
        "static-negative",
        "host-runtime-lifecycle",
        "runtime waits for terminal publication before D2H",
        "management-plane idle is not an execution completion oracle",
    ),
    TransportContractCase(
        "terminal-completion-d2h-cleanup",
        "runtime-publication",
        "board-executable",
        "board-runtime-lifecycle",
        "runner requires all-rank output capture and normal cleanup",
        "terminal schema, exact output, and cleanup are one lifecycle leaf",
    ),
    TransportContractCase(
        "outer-timeout-stops-batch",
        "runtime-publication",
        "static-negative",
        "host-runtime-lifecycle",
        "one-shot outer timeout without retry/reset/power",
        "a failed or missing terminal publication stops the active batch",
    ),
)


@dataclasses.dataclass(frozen=True)
class CounterDisposition:
    name: str
    block: str
    disposition: str
    interpretation: str
    reason: str


COUNTER_DISPOSITIONS = (
    CounterDisposition(
        "dte_channel0_transfer",
        "DTE",
        "board-observation",
        "payload-correlation; unit remains observed, not assumed",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "dte_channel1_transfer",
        "DTE",
        "board-observation",
        "payload-correlation; unit remains observed, not assumed",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "dte_channel0_execution",
        "DTE",
        "board-observation",
        "activity/cycle trend; no byte-unit inference",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "dte_channel1_execution",
        "DTE",
        "board-observation",
        "activity/cycle trend; no byte-unit inference",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "spm_dte_t2_port8",
        "SPM",
        "board-observation",
        "payload-correlation; event meaning remains profile-bound",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "spm_dte_t3_port8",
        "SPM",
        "board-observation",
        "payload-correlation; event meaning remains profile-bound",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "tmnoc",
        "TMNOC",
        "static-negative",
        "unsupported until a version-matched read-only register contract exists",
        (
            "current version-matched header exposes only TMNOC base addresses; "
            "inventing offsets could read a destructive or unrelated register"
        ),
    ),
)
COUNTERS_BY_NAME = {
    disposition.name: disposition for disposition in COUNTER_DISPOSITIONS
}
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "direct-dte-ordered-interaction": tuple(
        case for case in CASES if case.mode in (1, 2, 3, 4)
    ),
    "direct-dte-reuse-after-event": tuple(
        case for case in CASES if case.mode == 5
    ),
    "direct-dte-broadcast": tuple(case for case in CASES if case.mode == 6),
    "direct-dte-host-static-negative": tuple(
        case
        for case in CONTRACT_CASES
        if case.disposition == "static-negative"
        and case.domain.startswith("direct-dte")
    ),
    "direct-dte-device-error-observation": tuple(
        case
        for case in CONTRACT_CASES
        if case.disposition == "board-observation"
        and case.domain.startswith("direct-dte")
        and case.mode in ERROR_PATH_MODES
    ),
    "direct-dte-sender-async-controls": tuple(
        case
        for case in CONTRACT_CASES
        if case.mode in ASYNC_SENDER_MODES
    ),
    "direct-dte-unsafe-isolation": tuple(
        case
        for case in CONTRACT_CASES
        if case.disposition == "isolated-deferred"
        and case.domain.startswith("direct-dte")
    ),
    "runtime-publication-positive": tuple(
        case
        for case in CONTRACT_CASES
        if case.domain == "runtime-publication"
        and case.disposition == "board-executable"
    ),
    "runtime-publication-negative": tuple(
        case
        for case in CONTRACT_CASES
        if case.domain == "runtime-publication"
        and case.disposition == "static-negative"
    ),
    "dte-spm-counter-payload-sweep": TRANSPORT_PMU_OBSERVATIONS,
    "tmnoc-counter-offset-unavailable": (COUNTERS_BY_NAME["tmnoc"],),
}


@dataclasses.dataclass(frozen=True)
class SplitCounterRead:
    value: int
    stable: bool
    attempts: int


def modulo_counter_delta(after: int, before: int, bits: int) -> int:
    """Return the unsigned hardware-counter delta across one wrap."""

    if bits not in (32, 64):
        raise ValueError("only 32-bit and 64-bit PMU counters are supported")
    mask = (1 << bits) - 1
    if not 0 <= after <= mask or not 0 <= before <= mask:
        raise ValueError(f"counter samples must fit unsigned {bits}-bit words")
    return (after - before) & mask


def stable_high_low_high_read(
    attempts: tuple[tuple[int, int, int], ...],
    max_retries: int = SPLIT_COUNTER_STABLE_RETRIES,
) -> SplitCounterRead:
    """Model the device high-low-high split-counter read without MMIO."""

    if max_retries < 1:
        raise ValueError("split-counter retries must be positive")
    selected = attempts[:max_retries]
    if not selected:
        raise ValueError("at least one split-counter attempt is required")
    word_mask = (1 << 32) - 1
    for index, (high_before, low, high_after) in enumerate(selected, start=1):
        if any(
            not 0 <= word <= word_mask
            for word in (high_before, low, high_after)
        ):
            raise ValueError("split-counter words must fit unsigned 32-bit")
        value = (high_after << 32) | low
        if high_before == high_after:
            return SplitCounterRead(value, True, index)
    return SplitCounterRead(value, False, len(selected))


def classify_payload_series(
    samples: dict[int, tuple[int, ...]],
) -> dict[str, object]:
    """Classify raw counter observations without guessing the counter unit."""

    missing = tuple(size for size in PAYLOAD_SWEEP if not samples.get(size))
    if missing:
        return {
            "state": "inconclusive",
            "reason": "missing-payload-samples",
            "missing_payload_bytes": missing,
        }
    medians = {
        size: statistics.median(samples[size]) for size in PAYLOAD_SWEEP
    }
    values = tuple(medians[size] for size in PAYLOAD_SWEEP)
    if not any(values):
        relation = "all-zero"
    elif all(lhs < rhs for lhs, rhs in zip(values, values[1:])):
        relation = "strictly-increasing"
    elif all(lhs <= rhs for lhs, rhs in zip(values, values[1:])):
        relation = "nondecreasing"
    else:
        relation = "non-monotonic"

    scale_candidates = tuple(
        medians[size] / size for size in PAYLOAD_SWEEP
    )
    exact_integer_scale = (
        all(value > 0 for value in values)
        and len(set(scale_candidates)) == 1
        and scale_candidates[0].is_integer()
    )
    return {
        "state": "observation",
        "raw_medians": medians,
        "payload_relation": relation,
        "integer_scale_candidate": (
            int(scale_candidates[0]) if exact_integer_scale else None
        ),
        "unit": "unclassified",
    }
