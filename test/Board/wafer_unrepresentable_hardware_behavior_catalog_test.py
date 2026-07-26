#!/usr/bin/env python3
"""Host tests for typed unrepresentable hardware-behavior gates."""

from __future__ import annotations

import wafer_unrepresentable_hardware_behavior_catalog as catalog


def main() -> None:
    catalog.validate_catalog()
    assert len(catalog.BEHAVIORS) == 3
    assert set(catalog.BEHAVIORS_BY_KEY) == {
        "queue-active-resident-count-unobservable",
        "spm-physical-bank-color-class-unobservable",
        "argmin-unqualified-domain",
    }
    assert {
        behavior.safe_executable_family for behavior in catalog.BEHAVIORS
    } == {
        "queue-saturation-response",
        "spm-sustained-conflict-pilot",
        "argmin-tie-and-nan-domain",
    }
    argmin = catalog.BEHAVIORS_BY_KEY["argmin-unqualified-domain"]
    argmin_surrogates = " ".join(argmin.rejected_surrogates)
    assert "BF16 or F32" in argmin_surrogates
    assert "signed zero or infinity" in argmin_surrogates
    assert "subnormal" in argmin_surrogates
    assert catalog.ARGMIN_EXPLICIT_BOARD_CASES == (
        "peripheral-argmin-f16",
        "peripheral-argmin-negative-f16-observed",
        "peripheral-argmin-tie-f16-observed",
        "peripheral-argmin-nan-f16-observed",
    )
    for behavior in catalog.BEHAVIORS:
        try:
            catalog.prepare_board_request(behavior.key)
        except catalog.PreparationBlocked as error:
            message = str(error)
            assert behavior.key in message
            assert behavior.safe_executable_family in message
        else:
            raise AssertionError(
                f"{behavior.key}: unobservable behavior became executable"
            )
    try:
        catalog.prepare_board_request("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown behavior key was accepted")
    print("unrepresentable hardware behavior gates: passed")


if __name__ == "__main__":
    main()
