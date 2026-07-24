#!/usr/bin/env python3
"""Validate complete DTE/SPM sweep and fail-closed TMNOC disposition."""

from __future__ import annotations

import pathlib
import re

import wafer_transport_pmu_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CASES) == 18
    assert {(case.mode, case.payload_bytes) for case in catalog.CASES} == {
        (mode, payload)
        for mode in catalog.TRANSPORT_PMU_MODES
        for payload in catalog.PAYLOAD_SWEEP
    }
    assert catalog.ERROR_PATH_MODES == (7, 8, 9)
    assert {
        (case.mode, case.payload_bytes)
        for case in catalog.CONTRACT_CASES
        if case.disposition == "board-observation"
    } == {(7, 16), (8, 16), (9, 16)}
    assert sum(case.split == "calibration" for case in catalog.CASES) == 6
    assert sum(case.split == "held-out" for case in catalog.CASES) == 12
    assert {
        case.mode
        for case in catalog.CALIBRATION_LEAF_BINDINGS[
            "direct-dte-reuse-after-event"
        ]
    } == {5}
    assert {
        case.mode
        for case in catalog.CALIBRATION_LEAF_BINDINGS[
            "direct-dte-broadcast"
        ]
    } == {6}
    assert catalog.CALIBRATION_LEAF_BINDINGS
    assert all(catalog.CALIBRATION_LEAF_BINDINGS.values())
    real_objects = {
        id(case)
        for case in (
            catalog.CASES
            + catalog.CONTRACT_CASES
            + catalog.COUNTER_DISPOSITIONS
        )
    }
    assert all(
        id(case) in real_objects
        for cases in catalog.CALIBRATION_LEAF_BINDINGS.values()
        for case in cases
    )
    static_negative_names = {
        case.name
        for case in catalog.CONTRACT_CASES
        if case.disposition == "static-negative"
    }
    assert static_negative_names == {
        "dte-invalid-coordinate",
        "host-readback-before-terminal",
        "outer-timeout-stops-batch",
    }
    direct_host_negative = catalog.CALIBRATION_LEAF_BINDINGS[
        "direct-dte-host-static-negative"
    ]
    assert {case.name for case in direct_host_negative} == {
        "dte-invalid-coordinate"
    }
    assert {case.verification_scope for case in direct_host_negative} == {
        "host-prelaunch-verifier"
    }
    direct_error_observation = catalog.CALIBRATION_LEAF_BINDINGS[
        "direct-dte-device-error-observation"
    ]
    assert {case.name for case in direct_error_observation} == {
        "dte-source-reuse-before-send-event",
        "dte-invalid-fsm",
        "dte-wait-unknown-event",
    }
    assert {case.disposition for case in direct_error_observation} == {
        "board-observation"
    }
    assert all(
        case.verification_scope == "board-device-error-path"
        for case in direct_error_observation
    )
    direct_unsafe = catalog.CALIBRATION_LEAF_BINDINGS[
        "direct-dte-unsafe-isolation"
    ]
    assert {case.name for case in direct_unsafe} == {
        "dte-receiver-unprepared"
    }
    assert {case.disposition for case in direct_unsafe} == {
        "isolated-deferred"
    }
    assert all(
        case.verification_scope == "isolated-safety-quarantine"
        for case in direct_unsafe
    )
    assert {
        case.disposition
        for case in catalog.CALIBRATION_LEAF_BINDINGS[
            "runtime-publication-positive"
        ]
    } == {"board-executable"}
    assert {
        case.disposition
        for case in catalog.CALIBRATION_LEAF_BINDINGS[
            "runtime-publication-negative"
        ]
    } == {"static-negative"}
    assert catalog.CALIBRATION_LEAF_BINDINGS[
        "tmnoc-counter-offset-unavailable"
    ] == (catalog.COUNTERS_BY_NAME["tmnoc"],)
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

    crt_source = (
        repo / "runtime" / "wafer_crt" / "src" / "wafer_tx81_crt.c"
    ).read_text()
    assert "wafer_direct_dte_sender.active || remote_fsm_id >= 4" in crt_source
    assert "local_fsm_id >= WAFER_DIRECT_DTE_MAX_RECEIVERS" in crt_source
    wait_body = crt_source.split(
        "void wafer_tx81_direct_dte_wait(uint64_t event)", maxsplit=1
    )[1].split("void wafer_tx81_direct_dte_finish", maxsplit=1)[0]
    assert wait_body.rstrip().endswith("wafer_direct_dte_set_error();\n}")
    send_wait = wait_body.split(
        "if (event == WAFER_DIRECT_DTE_SEND_EVENT)", maxsplit=1
    )[1].split(
        "if (event >= WAFER_DIRECT_DTE_RECV_EVENT_BASE", maxsplit=1
    )[0]
    assert send_wait.index("direct_sync_wait(") < send_wait.index(
        "direct_dte_send_async("
    )

    probe_source = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_dte_ncc_execution_probe.c"
    ).read_text()
    reuse = probe_source.split(
        "case WAFER_PROBE_DTE_REUSE_AFTER_EVENTS:", maxsplit=1
    )[1].split(
        "case WAFER_PROBE_DTE_TWO_DESTINATION_BROADCAST:", maxsplit=1
    )[0]
    assert reuse.count("wafer_probe_dte_ring(") == 2
    assert reuse.index("intermediate_mismatches") < reuse.rindex(
        "wafer_probe_dte_ring("
    )
    broadcast = probe_source.split(
        "static void wafer_probe_dte_two_destination_broadcast(", maxsplit=1
    )[1].split(
        "static uint32_t wafer_probe_dte_reuse_before_send_event", maxsplit=1
    )[0]
    assert broadcast.count("wafer_tx81_direct_dte_recv_prepare(") == 2
    assert broadcast.count("wafer_tx81_direct_dte_send_prepare(") == 2
    first_send = broadcast.index("wafer_tx81_direct_dte_send_prepare(")
    second_send = broadcast.index(
        "wafer_tx81_direct_dte_send_prepare(", first_send + 1
    )
    assert broadcast.index("wafer_tx81_direct_dte_wait(send1)") < second_send
    early_reuse = probe_source.split(
        "static uint32_t wafer_probe_dte_reuse_before_send_event(", maxsplit=1
    )[1].split("static uint32_t wafer_probe_dte_invalid_fsm", maxsplit=1)[0]
    assert early_reuse.count("wafer_tx81_direct_dte_send_prepare(") == 2
    assert early_reuse.count("wafer_tx81_direct_dte_recv_prepare(") == 1
    assert early_reuse.index("rejected =") < early_reuse.index(
        "wafer_tx81_direct_dte_wait(send)"
    )
    assert early_reuse.index("wafer_tx81_direct_dte_wait(send)") < (
        early_reuse.index("wafer_tx81_direct_dte_wait(receive)")
    )
    invalid_fsm = probe_source.split(
        "static uint32_t wafer_probe_dte_invalid_fsm(", maxsplit=1
    )[1].split("static void wafer_probe_publish_header", maxsplit=1)[0]
    assert "successor, 4, 0" in invalid_fsm
    assert "predecessor, 4)" in invalid_fsm
    unknown_wait = probe_source.split(
        "case WAFER_PROBE_DTE_WAIT_UNKNOWN_EVENT_ERROR:", maxsplit=1
    )[1].split("}", maxsplit=1)[0]
    assert "wafer_tx81_direct_dte_wait(UINT64_C(0xdeadbeef));" in unknown_wait
    assert probe_source.count(
        "wafer_tx81_direct_dte_begin_after_prepare(status_ddr,"
    ) >= 1

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
        "cases=18 contracts=8 direct_host_negative=1 "
        "direct_error_observation=3 unsafe_isolation=1 board_counters=6 "
        "tmnoc_static_negative=1 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
