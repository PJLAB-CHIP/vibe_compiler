#!/usr/bin/env python3
"""Validate complete DTE/SPM sweep and fail-closed TMNOC disposition."""

from __future__ import annotations

import pathlib
import re

import wafer_transport_pmu_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CASES) == 12
    assert {(case.mode, case.payload_bytes) for case in catalog.CASES} == {
        (mode, payload)
        for mode in catalog.MODE_NAMES
        for payload in catalog.PAYLOAD_SWEEP
    }
    assert sum(case.split == "calibration" for case in catalog.CASES) == 4
    assert sum(case.split == "held-out" for case in catalog.CASES) == 8
    assert len(catalog.COUNTER_DISPOSITIONS) == 7
    assert set(catalog.BOARD_COUNTER_NAMES) == {
        disposition.name
        for disposition in catalog.COUNTER_DISPOSITIONS
        if disposition.disposition == "board-executable"
    }
    tmnoc = catalog.COUNTERS_BY_NAME["tmnoc"]
    assert tmnoc.disposition == "static-negative"
    assert "inventing offsets" in tmnoc.reason

    repo = pathlib.Path(__file__).resolve().parents[2]
    pmu_header = (
        repo
        / "third_party"
        / "tx8_deps"
        / "tx8-yoc-rt-thread-smp"
        / "include"
        / "components"
        / "oplib_tx81"
        / "riscv"
        / "riscv"
        / "include"
        / "pmu"
        / "pmu_reg.h"
    ).read_text()
    assert "#define SCT_REG_BASE_TMNOC " in pmu_header
    assert "#define SCT_REG_BASE_TMNOC1 " in pmu_header
    decoded_tmnoc_registers = re.findall(
        r"^#define\s+(?!SCT_REG_BASE_TMNOC1?\b)\w*TMNOC\w*",
        pmu_header,
        re.M,
    )
    assert not decoded_tmnoc_registers

    increasing = catalog.classify_payload_series(
        {16: (16, 16), 32: (32, 32), 64: (64, 64)}
    )
    assert increasing["payload_relation"] == "strictly-increasing"
    assert increasing["integer_scale_candidate"] == 1
    flat = catalog.classify_payload_series(
        {16: (7, 7), 32: (7, 7), 64: (7, 7)}
    )
    assert flat["payload_relation"] == "nondecreasing"
    assert flat["integer_scale_candidate"] is None
    missing = catalog.classify_payload_series({16: (1,), 32: (2,)})
    assert missing["state"] == "inconclusive"
    assert missing["missing_payload_bytes"] == (64,)

    print(
        "wafer_transport_pmu_calibration_catalog_test: "
        "cases=12 board_counters=6 tmnoc_static_negative=1 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
