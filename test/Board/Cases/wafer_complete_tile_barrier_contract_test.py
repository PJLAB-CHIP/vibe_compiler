#!/usr/bin/env python3
"""Validate complete-Tile-domain positive and subgroup static-negative barrier gates."""

from __future__ import annotations

import wafer_board_complete_tile_barrier_probe_runner as barrier


def main() -> int:
    assert barrier.CALIBRATION_LEAF_BINDINGS
    assert all(barrier.CALIBRATION_LEAF_BINDINGS.values())
    real_objects = {
        id(case)
        for case in (
            barrier.BARRIER_POSITIVE_CASES
            + barrier.BARRIER_NEGATIVE_CASES
        )
    }
    assert all(
        id(case) in real_objects
        for cases in barrier.CALIBRATION_LEAF_BINDINGS.values()
        for case in cases
    )
    assert {
        case.epoch for case in barrier.BARRIER_POSITIVE_CASES
    } == {None, 1, 2}
    assert {
        case.participants for case in barrier.BARRIER_NEGATIVE_CASES
    } == set(barrier.UNSUPPORTED_PARTICIPANT_COUNTS)
    assert all(
        case.disposition == "static-negative"
        for case in barrier.BARRIER_NEGATIVE_CASES
    )
    barrier.validate_participant_count(barrier.TILE_COUNT)
    for participants in barrier.UNSUPPORTED_PARTICIPANT_COUNTS:
        try:
            barrier.validate_participant_count(participants)
        except RuntimeError as error:
            assert "refusing unsafe subgroup" in str(error)
        else:
            raise AssertionError(
                f"unsafe subgroup {participants} was not rejected"
            )
    barrier.validate_host_contract()
    print(
        "wafer_complete_tile_barrier_contract_test: "
        f"positive_leaves={len(barrier.BARRIER_POSITIVE_CASES)} "
        f"negative_leaves={len(barrier.BARRIER_NEGATIVE_CASES)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
