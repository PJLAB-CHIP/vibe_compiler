#!/usr/bin/env python3
"""Static contract checks for the cross-Tile SPM conflict probe."""

from __future__ import annotations

import argparse
import pathlib
import struct

import wafer_board_spm_cross_tile_conflict_probe_test as probe
import wafer_memory_descriptor_calibration_catalog as catalog


ROOT = pathlib.Path(__file__).resolve().parent
INPUTS = ROOT / "Inputs"


def main() -> int:
    pairs = probe.conflict_pairs()
    probe.validate_static_contract(pairs)
    expected_keys = {
        (translation, phase, transfer, issue_order)
        for translation in (
            0,
            *catalog.SPM_CONFLICT_EQUIVALENCE_TRANSLATIONS,
        )
        for phase in catalog.SPM_PARALLEL_ALIGNMENT_PHASES
        for transfer in catalog.SPM_CONFLICT_EQUIVALENCE_TRANSFERS
        for issue_order in catalog.SPM_CONFLICT_EQUIVALENCE_ISSUE_ORDERS
    }
    actual_keys = {
        (
            pair.translation,
            pair.phase,
            pair.transfer_bytes,
            pair.issue_order,
        )
        for pair in pairs
    }
    assert actual_keys == expected_keys
    assert len(pairs) == len(expected_keys)
    assert {pair.coordinate_id for pair in pairs} == set(range(len(pairs)))
    selected = probe.select_pair(
        argparse.Namespace(selected_cases=[pairs[0].name]), pairs
    )
    assert selected == pairs[0]
    try:
        probe.select_pair(argparse.Namespace(selected_cases=None), pairs)
    except RuntimeError as error:
        assert "explicit --case" in str(error)
    else:
        raise AssertionError("implicit cross-tile held-out case was accepted")
    probe.validate_physical_coordinates(
        {tile_id: (tile_id % 4, tile_id // 4) for tile_id in range(probe.TILE_COUNT)}
    )
    duplicate_coordinates = {
        tile_id: (tile_id % 4, tile_id // 4) for tile_id in range(probe.TILE_COUNT)
    }
    duplicate_coordinates[15] = duplicate_coordinates[0]
    try:
        probe.validate_physical_coordinates(duplicate_coordinates)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate Tiles were accepted")
    for execution_round in range(probe.COUNTERBALANCED_ROUNDS):
        plans = [
            probe.execution_plan(tile_id, execution_round)
            for tile_id in range(probe.TILE_COUNT)
        ]
        assert {plan.tile_phase for plan in plans} == set(
            range(probe.TILE_COUNT)
        )
        assert {
            plan.tile_order for plan in plans
        } == {
            (
                probe.TILE_ORDER_FORWARD
                if execution_round % 2 == 0
                else probe.TILE_ORDER_REVERSE
            )
        }
        assert sum(
            plan.first_schedule == catalog.SCHEDULE_SERIAL
            for plan in plans
        ) == probe.TILE_COUNT // 2
        assert sum(
            plan.first_schedule == catalog.SCHEDULE_WINDOW
            for plan in plans
        ) == probe.TILE_COUNT // 2

    for pair in pairs:
        request, serial, window, wire_sample = probe.build_tile_request(
            pair, 15, 2
        )
        assert wire_sample == 47
        plan = probe.execution_plan(15, 2)
        assert len(request) == catalog.RESOURCE_BYTES
        meta = struct.unpack_from(
            f"<{probe.REQUEST_META_WORDS}Q",
            request,
            probe.REQUEST_META_BEGIN,
        )
        expected_meta = {
            "MAGIC": probe.REQUEST_MAGIC,
            "WORD_COUNT": probe.REQUEST_META_WORDS,
            "TILE_ID": 15,
            "TILE_COUNT": probe.TILE_COUNT,
            "COORDINATE": pair.coordinate_id,
            "SERIAL_CASE": pair.serial.case_id,
            "WINDOW_CASE": pair.window.case_id,
            "SERIAL_REQUEST_WORD": probe.SERIAL_REQUEST_WORD,
            "WINDOW_REQUEST_WORD": probe.WINDOW_REQUEST_WORD,
            "SPM_A": pair.serial.spm_a,
            "SPM_B": pair.serial.spm_b,
            "TRANSFER_BYTES": pair.transfer_bytes,
            "ISSUE_ORDER": pair.issue_order,
            "SAMPLE": wire_sample,
            "RESOURCE_BYTES": catalog.RESOURCE_BYTES,
            "GUARD": probe.REQUEST_GUARD,
            "EXECUTION_ROUND": plan.execution_round,
            "TILE_ORDER": plan.tile_order,
            "TILE_PHASE": plan.tile_phase,
            "FIRST_SCHEDULE": plan.first_schedule,
        }
        assert all(
            meta[probe.REQ_META[key]] == value
            for key, value in expected_meta.items()
        )
        serial_words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", request
        )
        window_words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q",
            request,
            probe.WINDOW_REQUEST_WORD * 8,
        )
        assert serial_words[catalog.REQ["CASE"]] == pair.serial.case_id
        assert window_words[catalog.REQ["CASE"]] == pair.window.case_id
        assert (
            serial_words[catalog.REQ["SCHEDULE"]]
            == catalog.SCHEDULE_SERIAL
        )
        assert (
            window_words[catalog.REQ["SCHEDULE"]]
            == catalog.SCHEDULE_WINDOW
        )
        assert serial.payload == window.payload
        serial_meta = probe.expected_record_meta(
            pair,
            pair.serial,
            15,
            wire_sample,
            plan,
            catalog.SCHEDULE_SERIAL,
        )
        window_meta = probe.expected_record_meta(
            pair,
            pair.window,
            15,
            wire_sample,
            plan,
            catalog.SCHEDULE_WINDOW,
        )
        assert {
            serial_meta["EXECUTION_ORDINAL"],
            window_meta["EXECUTION_ORDINAL"],
        } == {0, 1}
        assert (
            serial_meta["FIRST_SCHEDULE"]
            == window_meta["FIRST_SCHEDULE"]
            == plan.first_schedule
        )
        assert (
            serial_meta["TILE_PHASE"]
            == window_meta["TILE_PHASE"]
            == plan.tile_phase
        )

    def synthetic_launches(delta: int) -> list[dict[str, object]]:
        launches: list[dict[str, object]] = []
        for execution_round in range(probe.COUNTERBALANCED_ROUNDS):
            rows: list[dict[str, object]] = []
            for tile_id in range(probe.TILE_COUNT):
                plan = probe.execution_plan(tile_id, execution_round)
                rows.append(
                    {
                        "tile_id": tile_id,
                        "execution_round": execution_round,
                        "tile_order": (
                            "forward"
                            if plan.tile_order
                            == probe.TILE_ORDER_FORWARD
                            else "reverse"
                        ),
                        "first_schedule": (
                            "serial"
                            if plan.first_schedule
                            == catalog.SCHEDULE_SERIAL
                            else "window"
                        ),
                        "physical_x": tile_id % 4,
                        "physical_y": tile_id // 4,
                        "request_ddr": 0x10000000 + tile_id * 0x100000,
                        "payload_ddr": 0x20000000 + tile_id * 0x100000,
                        "window_minus_serial": {
                            "plan_cycles": delta,
                            "full_execution": delta,
                            "ct_blocking": 0,
                            "rdma_blocking": 0,
                        },
                    }
                )
            launches.append({"rows": rows})
        return launches

    zero_summary = probe.summarize_repeats(
        pairs[0], synthetic_launches(0)
    )
    assert zero_summary["state"] == (
        "cross-Tile-heldout-consistent-but-zero-signal"
    )
    assert not zero_summary["nonzero_cost_signal"]
    signal_summary = probe.summarize_repeats(
        pairs[0], synthetic_launches(1)
    )
    assert signal_summary["state"] == (
        "cross-Tile-heldout-proxy-consistent"
    )
    assert signal_summary["nonzero_cost_signal"]

    llvm_ir = (INPUTS / "wafer_spm_cross_tile_conflict_probe.ll").read_text()
    device = (
        INPUTS / "wafer_memory_descriptor_calibration_probe.c"
    ).read_text()
    protocol = (
        INPUTS / "wafer_spm_cross_tile_conflict_probe_protocol.h"
    ).read_text()
    assert "define void @__wafer_kernel_prepare" in llvm_ir
    assert "call void @direct_sync_init(i32 16)" in llvm_ir
    assert "%row = mul i64 %pid64, 5" in llvm_ir
    assert "wafer_tx81_spm_cross_tile_conflict_probe" in llvm_ir
    assert "WAFER_SPM_CT_SERIAL_REQUEST_WORD 0U" in protocol
    assert "WAFER_SPM_CT_WINDOW_REQUEST_WORD 40U" in protocol
    assert "WAFER_SPM_CT_REQUEST_META_WORD 80U" in protocol
    assert "WAFER_SPM_CT_RECORD_META_WORD 64U" in protocol
    assert "WAFER_SPM_CT_COUNTERBALANCED_ROUNDS 4U" in protocol
    assert "wafer_tx81_spm_cross_tile_conflict_probe" in device
    assert "hrt_barrier" in device
    assert "WAFER_SPM_CT_TILE_ORDER_REVERSE" in device
    assert "first_schedule == WAFER_MDC_SCHEDULE_SERIAL" in device
    driver = pathlib.Path(probe.__file__).read_text()
    for stage in ("completion", "device-to-host", "cleanup"):
        assert f'"board_stage: {stage}"' in driver
    assert callable(probe.select_pair)
    assert callable(probe.build_probe)
    print(
        "spm cross-tile conflict probe contract tests passed: "
        f"{len(pairs)} paired coordinates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
