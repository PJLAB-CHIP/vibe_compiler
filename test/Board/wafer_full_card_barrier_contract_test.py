#!/usr/bin/env python3
"""Validate full-card positive and subgroup static-negative barrier gates."""

from __future__ import annotations

import wafer_board_full_card_barrier_probe_test as barrier


def main() -> int:
    barrier.validate_participant_count(barrier.RANK_COUNT)
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
        "wafer_full_card_barrier_contract_test: "
        f"positive={barrier.RANK_COUNT} "
        f"negative={barrier.UNSUPPORTED_PARTICIPANT_COUNTS} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
