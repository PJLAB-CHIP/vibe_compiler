#!/usr/bin/env python3
"""DTE/SPM transport-PMU payload-sweep and fail-closed TMNOC contract."""

from __future__ import annotations

import dataclasses
import statistics


PAYLOAD_SWEEP = (16, 32, 64)
CALIBRATION_PAYLOADS = (16,)
HELD_OUT_PAYLOADS = (32, 64)
MODE_NAMES = {
    1: "ncc-producer-local-drain-dte",
    2: "dte-recv-wait-ncc-consumer",
    3: "disjoint-local-wait-first",
    4: "disjoint-dte-wait-first",
}
BOARD_COUNTER_NAMES = (
    "dte_channel0_transfer",
    "dte_channel1_transfer",
    "dte_channel0_execution",
    "dte_channel1_execution",
    "spm_dte_t2_port8",
    "spm_dte_t3_port8",
)


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
        for mode in MODE_NAMES
    )
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
        "board-executable",
        "payload-correlation; unit remains observed, not assumed",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "dte_channel1_transfer",
        "DTE",
        "board-executable",
        "payload-correlation; unit remains observed, not assumed",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "dte_channel0_execution",
        "DTE",
        "board-executable",
        "activity/cycle trend; no byte-unit inference",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "dte_channel1_execution",
        "DTE",
        "board-executable",
        "activity/cycle trend; no byte-unit inference",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "spm_dte_t2_port8",
        "SPM",
        "board-executable",
        "payload-correlation; event meaning remains profile-bound",
        "version-matched header exposes a decoded low/high split counter",
    ),
    CounterDisposition(
        "spm_dte_t3_port8",
        "SPM",
        "board-executable",
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
    elif values[0] < values[1] < values[2]:
        relation = "strictly-increasing"
    elif values[0] <= values[1] <= values[2]:
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
