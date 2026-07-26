#!/usr/bin/env python3
"""Host contract and oracle-mutation tests for the sustained SPM pilot."""

from __future__ import annotations

import pathlib
import struct
import subprocess
import sys

import wafer_board_spm_sustained_conflict_probe_test as driver
import wafer_spm_sustained_conflict_catalog as catalog


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "test/Board/Inputs/"
    "wafer_spm_sustained_conflict_probe_protocol.h"
)
DEVICE = (
    ROOT
    / "test/Board/Inputs/"
    "wafer_spm_sustained_conflict_probe.c"
)
SCRIPT = (
    ROOT
    / "test/Board/"
    "wafer_board_spm_sustained_conflict_probe_test.py"
)


def _fixture(
    invocation: catalog.InvocationPayload,
) -> bytes:
    raw = bytearray(
        [catalog.RESOURCE_CANARY] * catalog.RESOURCE_BYTES
    )
    addresses = (0x10000000, 0x10200000, 0x10400000)
    global_words = [0] * catalog.RECORD_WORDS
    expected_global = driver._expected_global_record(invocation)
    for key, value in expected_global.items():
        global_words[catalog.REC[key]] = value
    for key, value in zip(
        ("REQUEST_DDR", "PAYLOAD_DDR", "OUTPUT_DDR"),
        addresses,
        strict=True,
    ):
        global_words[catalog.REC[key]] = value
    struct.pack_into(
        f"<{catalog.RECORD_WORDS}Q", raw, 0, *global_words
    )

    for cell in invocation.group.cells:
        row_words = [0] * catalog.ROW_RECORD_WORDS
        expected_row = driver._expected_row_record(invocation, cell)
        for key, value in expected_row.items():
            row_words[catalog.ROW_REC[key]] = value
        for key, value in zip(
            ("REQUEST_DDR", "PAYLOAD_DDR", "OUTPUT_DDR"),
            addresses,
            strict=True,
        ):
            row_words[catalog.ROW_REC[key]] = value
        row_words[catalog.ROW_REC["CT_EXEC_DELTA"]] = 100 + cell.cell_id
        row_words[catalog.ROW_REC["RDMA_EXEC_DELTA"]] = (
            200 + cell.cell_id
        )
        row_words[catalog.ROW_REC["FULL_EXEC_DELTA"]] = (
            300 + cell.cell_id
        )
        row_words[catalog.ROW_REC["CT_BLOCKING_DELTA"]] = cell.cell_id
        row_words[catalog.ROW_REC["RDMA_BLOCKING_DELTA"]] = (
            cell.cell_id + 1
        )
        row_words[catalog.ROW_REC["PLAN_CYCLES"]] = 400 + cell.cell_id
        row_words[catalog.ROW_REC["PMU_ENABLE"]] = 1
        row_words[catalog.ROW_REC["SERIAL_MODE"]] = 0
        row_words[catalog.ROW_REC["STABLE_BEFORE"]] = (
            catalog.PMU_STABLE_MASK
        )
        row_words[catalog.ROW_REC["STABLE_AFTER"]] = (
            catalog.PMU_STABLE_MASK
        )
        row_words[catalog.ROW_REC["FINAL_CONTROL"]] = 0x100
        begin = (
            catalog.ROW_RECORD_BASE_WORD
            + cell.row_index * catalog.ROW_RECORD_WORDS
        ) * 8
        struct.pack_into(
            f"<{catalog.ROW_RECORD_WORDS}Q",
            raw,
            begin,
            *row_words,
        )
        archive = (
            catalog.OUTPUT_ARCHIVE_BASE
            + cell.row_index * catalog.OUTPUT_ARCHIVE_STRIDE
        )
        snapshot = invocation.expected_snapshots[cell.relative_offset]
        raw[archive : archive + len(snapshot)] = snapshot
    return bytes(raw)


def _must_reject(
    raw: bytes,
    invocation: catalog.InvocationPayload,
    label: str,
) -> None:
    try:
        driver.validate_output_bytes(raw, invocation)
    except RuntimeError:
        return
    raise AssertionError(f"oracle mutation was accepted: {label}")


def _validate_inventory() -> None:
    assert len(catalog.CELLS) == 16
    assert len(catalog.GROUPS) == 4
    assert {cell.work_bytes for cell in catalog.CELLS} == {4096, 16384}
    assert {
        cell.relative_offset for cell in catalog.CELLS
    } == {4352, 8192}
    assert {
        cell.issue_order_name for cell in catalog.CELLS
    } == {"a-b", "b-a"}
    assert {
        cell.schedule.value for cell in catalog.CELLS
    } == {"serial", "window"}
    assert all(len(group.cells) == 4 for group in catalog.GROUPS)
    for cell in catalog.CELLS:
        ordered = sorted(
            cell.active_ranges(), key=lambda active: active.begin
        )
        assert all(
            left.end <= right.begin
            for left, right in zip(ordered, ordered[1:])
        )
        assert ordered[0].begin >= catalog.SPM_BASE
        assert (
            ordered[-1].end
            <= catalog.SPM_BASE + cell.owned_spm_bytes
        )


def _validate_wire_and_oracle() -> None:
    for group in catalog.GROUPS:
        observations = []
        for sample in range(catalog.COUNTERBALANCED_REPEATS):
            invocation = catalog.build_invocation(group, sample)
            assert len(invocation.request) == catalog.RESOURCE_BYTES
            assert len(invocation.payload) == catalog.RESOURCE_BYTES
            words = struct.unpack_from(
                f"<{catalog.REQUEST_WORDS}Q", invocation.request
            )
            assert words[catalog.REQ["MAGIC"]] == catalog.REQUEST_MAGIC
            assert words[catalog.REQ["GROUP"]] == group.group_id
            assert words[catalog.REQ["OWNED_SPM_BYTES"]] == (
                group.owned_spm_bytes
            )
            fixture = _fixture(invocation)
            observation = driver.validate_output_bytes(
                fixture, invocation
            )
            observations.append(observation)

            if sample == 0:
                mutated = bytearray(fixture)
                mutated[
                    catalog.OUTPUT_ARCHIVE_BASE
                    + catalog.SPM_READ0_OFFSET
                ] ^= 1
                _must_reject(
                    bytes(mutated), invocation, "full snapshot"
                )

                mutated = bytearray(fixture)
                row_record = catalog.ROW_RECORD_BASE_WORD * 8
                count_byte = (
                    row_record
                    + catalog.ROW_REC["CT_INST_DELTA"] * 8
                )
                mutated[count_byte] ^= 1
                _must_reject(
                    bytes(mutated), invocation, "instruction count"
                )

                mutated = bytearray(fixture)
                control_byte = (
                    row_record
                    + catalog.ROW_REC["FINAL_CONTROL"] * 8
                )
                struct.pack_into("<Q", mutated, control_byte, 1)
                _must_reject(
                    bytes(mutated), invocation, "completion control"
                )

                mutated = bytearray(fixture)
                stable_byte = (
                    row_record
                    + catalog.ROW_REC["STABLE_AFTER"] * 8
                )
                struct.pack_into("<Q", mutated, stable_byte, 0)
                _must_reject(bytes(mutated), invocation, "PMU stability")

                mutated = bytearray(fixture)
                mutated[-1] ^= 1
                _must_reject(
                    bytes(mutated), invocation, "outside typed ranges"
                )

        summary = driver.summarize_group(group, observations)
        assert summary["samples"] == catalog.COUNTERBALANCED_REPEATS
        for cell in group.cells:
            ordinals = {
                int(
                    observation["rows"][cell.key][
                        "execution_ordinal"
                    ]
                )
                for observation in observations
            }
            assert ordinals == set(range(catalog.ROW_COUNT))


def _validate_device_contract() -> None:
    protocol = PROTOCOL.read_text()
    device = DEVICE.read_text()
    for fragment in (
        "#define WAFER_SSC_SCHEMA 1U",
        "#define WAFER_SSC_ROW_COUNT 4U",
        "#define WAFER_SSC_REPEATS 4U",
        "#define WAFER_SSC_TRANSFER_BYTES 4096U",
        "#define WAFER_SSC_CANDIDATE_OFFSET UINT64_C(8192)",
        "#define WAFER_SSC_CONTROL_OFFSET UINT64_C(4352)",
        "WAFER_SSC_ROW_CT_INST_DELTA = 23",
        "WAFER_SSC_ROW_STABLE_AFTER = 37",
        "WAFER_SSC_ROW_FINAL_CONTROL = 38",
    ):
        assert fragment in protocol
    for fragment in (
        "wafer_ssc_seed_owned_spm(request, payload_ddr)",
        "wafer_ssc_prepare_packets(",
        "WaferMDCPMU before = wafer_mdc_read_pmu();",
        "wafer_ssc_issue_measured(",
        "WaferMDCPMU after = wafer_mdc_read_pmu();",
        "wafer_ssc_archive_owned_spm(",
        "(ordinal + request.sample) % WAFER_SSC_ROW_COUNT",
        "WAFER_SSC_ROW_OTHER_INST_DELTA",
        "WAFER_SSC_ROW_WORKER_CT_INST_DELTA",
        "WAFER_SSC_ROW_WORKER_RDMA_INST_DELTA",
    ):
        assert fragment in device
    measured_begin = device.index(
        "WaferMDCPMU before = wafer_mdc_read_pmu();"
    )
    measured_end = device.index(
        "WaferMDCPMU after = wafer_mdc_read_pmu();",
        measured_begin,
    )
    readback = device.index(
        "wafer_ssc_archive_owned_spm(", measured_end
    )
    assert measured_begin < measured_end < readback


def _validate_cli() -> None:
    driver_source = SCRIPT.read_text()
    assert "rank_one_terminal_completion" in driver_source
    assert "validate_board_lifecycle" in driver_source
    emitted = subprocess.run(
        [sys.executable, str(SCRIPT), "--emit-board-case-keys"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert emitted == [group.key for group in catalog.GROUPS]
    listed = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-cases"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert all(cell.key in listed for cell in catalog.CELLS)
    assert all(group.key in listed for group in catalog.GROUPS)


def main() -> int:
    _validate_inventory()
    _validate_wire_and_oracle()
    _validate_device_contract()
    _validate_cli()
    print("spm_sustained_conflict_probe_contract: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
