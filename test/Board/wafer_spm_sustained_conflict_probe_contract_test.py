#!/usr/bin/env python3
"""Host contract and oracle-mutation tests for the sustained SPM pilot."""

from __future__ import annotations

import copy
import pathlib
import re
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
        row_words[
            catalog.ROW_REC["SPM_PORT_PMU_ENABLE_ORIGINAL"]
        ] = 0x3
        for field in (
            "SPM_PORT_PMU_ENABLE_BEFORE",
            "SPM_PORT_PMU_ENABLE_BOUNDARY",
            "SPM_PORT_PMU_ENABLE_AFTER",
        ):
            row_words[catalog.ROW_REC[field]] = (
                catalog.SPM_PORT_PMU_REQUIRED_ENABLE
            )
        row_words[
            catalog.ROW_REC["SPM_PORT_PMU_ENABLE_RESTORED"]
        ] = 0x3
        for field in (
            "SPM_PORT_PMU_STABLE_BEFORE",
            "SPM_PORT_PMU_STABLE_BOUNDARY",
            "SPM_PORT_PMU_STABLE_AFTER",
        ):
            row_words[catalog.ROW_REC[field]] = (
                catalog.SPM_PORT_PMU_STABLE_MASK
            )
        pair_delta = (
            50
            if cell.address_class == catalog.AddressClass.CANDIDATE
            else 30
        )
        readback_delta = (
            40
            if cell.address_class == catalog.AddressClass.CANDIDATE
            else 20
        )
        counter_mask = (1 << 64) - 1
        counter_base = (
            counter_mask - 120
            if invocation.sample == 3
            else 10000 + invocation.sample * 1000 + cell.cell_id * 100
        )
        counter_windows = {
            "SPM_PORT0_T2": (counter_base, pair_delta, 1),
            "SPM_PORT0_T3": (
                counter_base + 100000,
                pair_delta * 2,
                1,
            ),
            "SPM_PORT6_T2": (
                counter_base + 200000,
                2,
                readback_delta,
            ),
            "SPM_PORT6_T3": (
                counter_base + 300000,
                3,
                readback_delta * 2,
            ),
        }
        for prefix, (
            before_value,
            pair_value,
            readback_value,
        ) in counter_windows.items():
            before_value &= counter_mask
            boundary_value = (
                before_value + pair_value
            ) & counter_mask
            row_words[catalog.ROW_REC[f"{prefix}_BEFORE"]] = (
                before_value
            )
            row_words[catalog.ROW_REC[f"{prefix}_BOUNDARY"]] = (
                boundary_value
            )
            row_words[catalog.ROW_REC[f"{prefix}_AFTER"]] = (
                boundary_value + readback_value
            ) & counter_mask
        row_words[catalog.ROW_REC["WDMA_FINAL_CONTROL"]] = 0x100
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
        response = (
            catalog.PORT_RESPONSE_BASE
            + cell.row_index * catalog.PORT_RESPONSE_STRIDE
        )
        raw[
            response :
            response + len(invocation.expected_port_readback)
        ] = invocation.expected_port_readback
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


def _must_reject_poisoned(
    raw: bytes,
    invocation: catalog.InvocationPayload,
    label: str,
) -> None:
    try:
        driver.validate_output_bytes(raw, invocation)
    except RuntimeError as error:
        assert "poisoned" in str(error) and "must stop" in str(error)
        return
    raise AssertionError(f"poisoned mutation was accepted: {label}")


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
        cell.expected_setup_accepted for cell in catalog.CELLS
    } == {3, 10}
    assert {
        cell.expected_archive_accepted for cell in catalog.CELLS
    } == {1, 2}
    assert {
        cell.schedule.value for cell in catalog.CELLS
    } == {"serial", "window"}
    assert all(len(group.cells) == 4 for group in catalog.GROUPS)
    for cell in catalog.CELLS:
        port_contract = cell.as_dict()["spm_port_response"]
        assert isinstance(port_contract, dict)
        assert port_contract["counter_unit"] == "unclassified"
        assert port_contract["bank_identity"] == "not-observed"
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
            assert words[catalog.REQ["PORT_RESPONSE_BASE"]] == (
                catalog.PORT_RESPONSE_BASE
            )
            assert words[catalog.REQ["SPM_PORT_PMU_BASE"]] == (
                catalog.SPM_PORT_PMU_BASE
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
                wdma_count_byte = (
                    row_record
                    + catalog.ROW_REC["WDMA_INST_DELTA"] * 8
                )
                mutated[wdma_count_byte] ^= 1
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "matched WDMA instruction count",
                )

                mutated = bytearray(fixture)
                wdma_control_byte = (
                    row_record
                    + catalog.ROW_REC["WDMA_FINAL_CONTROL"] * 8
                )
                struct.pack_into(
                    "<Q", mutated, wdma_control_byte, 1
                )
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "matched WDMA completion control",
                )

                for field in (
                    "SETUP_ACCEPTED",
                    "MEASURED_PAIR_ACCEPTED",
                    "MATCHED_WDMA_ACCEPTED",
                    "ARCHIVE_ACCEPTED",
                ):
                    mutated = bytearray(fixture)
                    mutated[
                        row_record + catalog.ROW_REC[field] * 8
                    ] ^= 1
                    _must_reject(
                        bytes(mutated),
                        invocation,
                        f"{field} accepted count",
                    )

                for field in (
                    "SETUP_PENDING_AFTER",
                    "MEASURED_PAIR_PENDING_AFTER",
                    "MATCHED_WDMA_PENDING_AFTER",
                    "ARCHIVE_PENDING_AFTER",
                ):
                    mutated = bytearray(fixture)
                    struct.pack_into(
                        "<Q",
                        mutated,
                        row_record + catalog.ROW_REC[field] * 8,
                        1,
                    )
                    _must_reject(
                        bytes(mutated),
                        invocation,
                        f"{field} pending work",
                    )

                for field in (
                    "SETUP_FINAL_CONTROL",
                    "ARCHIVE_FINAL_CONTROL",
                ):
                    mutated = bytearray(fixture)
                    struct.pack_into(
                        "<Q",
                        mutated,
                        row_record + catalog.ROW_REC[field] * 8,
                        1,
                    )
                    _must_reject(
                        bytes(mutated),
                        invocation,
                        f"{field} completion control",
                    )

                mutated = bytearray(fixture)
                struct.pack_into(
                    "<Q",
                    mutated,
                    row_record
                    + catalog.ROW_REC["CLEANUP_ATTEMPTED_MASK"] * 8,
                    catalog.CLEANUP_SETUP,
                )
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "unexpected cleanup attempt on a successful row",
                )

                mutated = bytearray(fixture)
                struct.pack_into(
                    "<Q",
                    mutated,
                    row_record
                    + catalog.ROW_REC["CLEANUP_SUCCEEDED_MASK"] * 8,
                    catalog.CLEANUP_ARCHIVE,
                )
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "unexpected cleanup success on a successful row",
                )

                mutated = bytearray(fixture)
                for field, value in (
                    (
                        "CLEANUP_ATTEMPTED_MASK",
                        catalog.CLEANUP_MATCHED_WDMA,
                    ),
                    ("CLEANUP_SUCCEEDED_MASK", 0),
                    (
                        "POISONED_PHASE_MASK",
                        catalog.CLEANUP_MATCHED_WDMA,
                    ),
                    ("MATCHED_WDMA_PENDING_AFTER", 1),
                    ("WDMA_FINAL_CONTROL", 1),
                ):
                    struct.pack_into(
                        "<Q",
                        mutated,
                        row_record + catalog.ROW_REC[field] * 8,
                        value,
                    )
                _must_reject_poisoned(
                    bytes(mutated),
                    invocation,
                    "single failed drain poisons its row phase",
                )

                mutated = bytearray(fixture)
                struct.pack_into(
                    "<Q",
                    mutated,
                    catalog.REC["STATUS"] * 8,
                    catalog.STATUS_CLEANUP_FAILED,
                )
                _must_reject_poisoned(
                    bytes(mutated),
                    invocation,
                    "global cleanup-failed status",
                )

                mutated = bytearray(fixture)
                stable_byte = (
                    row_record
                    + catalog.ROW_REC["STABLE_AFTER"] * 8
                )
                struct.pack_into("<Q", mutated, stable_byte, 0)
                _must_reject(bytes(mutated), invocation, "PMU stability")

                mutated = bytearray(fixture)
                port_stable_byte = (
                    row_record
                    + catalog.ROW_REC[
                        "SPM_PORT_PMU_STABLE_BOUNDARY"
                    ]
                    * 8
                )
                struct.pack_into(
                    "<Q", mutated, port_stable_byte, 0
                )
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "SPM port-PMU split-read stability",
                )

                mutated = bytearray(fixture)
                restored_byte = (
                    row_record
                    + catalog.ROW_REC[
                        "SPM_PORT_PMU_ENABLE_RESTORED"
                    ]
                    * 8
                )
                struct.pack_into(
                    "<Q", mutated, restored_byte, 0x1F
                )
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "SPM port-PMU enable restore",
                )

                mutated = bytearray(fixture)
                response_byte = catalog.PORT_RESPONSE_BASE
                mutated[response_byte] ^= 1
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "matched WDMA response",
                )

                mutated = bytearray(fixture)
                response_guard_byte = (
                    row_record
                    + catalog.ROW_REC["PORT_RESPONSE_GUARD"] * 8
                )
                mutated[response_guard_byte] ^= 1
                _must_reject(
                    bytes(mutated),
                    invocation,
                    "port response record guard",
                )

                mutated = bytearray(fixture)
                mutated[-1] ^= 1
                _must_reject(
                    bytes(mutated), invocation, "outside typed ranges"
                )

        summary = driver.summarize_group(group, observations)
        assert summary["samples"] == catalog.COUNTERBALANCED_REPEATS
        assert summary["state"] == (
            "raw-port-signal-requires-heldout-review"
        )
        assert summary["compiler_use"] == "no-bank-coloring"
        aggregate_only = copy.deepcopy(observations)
        for observation in aggregate_only:
            rows = observation["rows"]
            assert isinstance(rows, dict)
            for row in rows.values():
                assert isinstance(row, dict)
                response = row["spm_port_response"]
                assert isinstance(response, dict)
                for window_name in (
                    "pair_window",
                    "matched_readback_window",
                ):
                    window = response[window_name]
                    assert isinstance(window, dict)
                    for counter_name in window:
                        window[counter_name] = 0
        aggregate_summary = driver.summarize_group(
            group, aggregate_only
        )
        assert aggregate_summary["state"] == (
            "aggregate-signal-without-stable-port-response"
        )
        assert aggregate_summary["compiler_use"] == "no-bank-coloring"
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

    def parse_enum(enum_name: str, prefix: str) -> dict[str, int]:
        match = re.search(
            rf"enum {enum_name} \{{(?P<body>.*?)\}};",
            protocol,
            flags=re.DOTALL,
        )
        assert match is not None
        return {
            name.removeprefix(prefix): int(value)
            for name, value in re.findall(
                rf"({prefix}[A-Z0-9_]+) = ([0-9]+),",
                match.group("body"),
            )
        }

    def function_body(name: str) -> str:
        match = re.search(rf"\b{name}\s*\([^;]*?\)\s*\{{", device, re.DOTALL)
        assert match is not None
        begin = device.index("{", match.start())
        depth = 0
        for index in range(begin, len(device)):
            if device[index] == "{":
                depth += 1
            elif device[index] == "}":
                depth -= 1
                if depth == 0:
                    return device[begin + 1 : index]
        raise AssertionError(f"unterminated C function {name}")

    assert parse_enum("WaferSSCRequestWord", "WAFER_SSC_REQ_") == (
        catalog.REQ
    )
    assert parse_enum("WaferSSCRecordWord", "WAFER_SSC_REC_") == (
        catalog.REC
    )
    row_fields = parse_enum(
        "WaferSSCRowRecordWord", "WAFER_SSC_ROW_"
    )
    row_fields["MAGIC"] = row_fields.pop("MAGIC_WORD")
    row_fields["ROW_GUARD"] = row_fields.pop("GUARD_WORD")
    assert row_fields == catalog.ROW_REC

    for fragment in (
        "#define WAFER_SSC_SCHEMA 3U",
        "#define WAFER_SSC_REQUEST_WORDS 36U",
        "#define WAFER_SSC_ROW_RECORD_WORDS 86U",
        "#define WAFER_SSC_ROW_COUNT 4U",
        "#define WAFER_SSC_REPEATS 4U",
        "#define WAFER_SSC_TRANSFER_BYTES 4096U",
        "#define WAFER_SSC_CANDIDATE_OFFSET UINT64_C(8192)",
        "#define WAFER_SSC_CONTROL_OFFSET UINT64_C(4352)",
        "#define WAFER_SSC_SPM_PORT_PMU_BASE UINT64_C(0x580000)",
        "#define WAFER_SSC_SPM_PORT_PMU_REQUIRED_ENABLE UINT32_C(0x1f)",
        "WAFER_SSC_ROW_CT_INST_DELTA = 23",
        "WAFER_SSC_ROW_STABLE_AFTER = 37",
        "WAFER_SSC_ROW_FINAL_CONTROL = 38",
        "WAFER_SSC_ROW_SPM_PORT0_T2_BEFORE = 52",
        "WAFER_SSC_ROW_WDMA_INST_DELTA = 64",
        "WAFER_SSC_ROW_PORT_RESPONSE_GUARD = 71",
        "WAFER_SSC_ROW_SETUP_ACCEPTED = 72",
        "WAFER_SSC_ROW_POISONED_PHASE_MASK = 84",
        "WAFER_SSC_ROW_CLEANUP_RECORD_GUARD = 85",
        "WAFER_SSC_STATUS_CLEANUP_FAILED = 6",
    ):
        assert fragment in protocol
    for fragment in (
        "wafer_ssc_seed_owned_spm(",
        "wafer_ssc_prepare_packets(",
        "WaferMDCPMU before = wafer_mdc_read_pmu();",
        "wafer_ssc_issue_measured(",
        "WaferMDCPMU after = wafer_mdc_read_pmu();",
        "wafer_ssc_prepare_matched_wdma(",
        "WaferSSCPortPMU port_before = wafer_ssc_read_spm_port_pmu();",
        "port_boundary = wafer_ssc_read_spm_port_pmu();",
        "wafer_ssc_issue_matched_wdma(",
        "port_after = wafer_ssc_read_spm_port_pmu();",
        "wafer_ssc_restore_spm_port_pmu(",
        "wafer_ssc_archive_owned_spm(",
        "(ordinal + request.sample) % WAFER_SSC_ROW_COUNT",
        "WAFER_SSC_ROW_OTHER_INST_DELTA",
        "WAFER_SSC_ROW_WORKER_CT_INST_DELTA",
        "WAFER_SSC_ROW_WORKER_RDMA_INST_DELTA",
        "#define WAFER_SSC_MAX_DMA_CHUNK 65536U",
    ):
        assert fragment in device

    fail_work = function_body("wafer_ssc_fail_work")
    assert fail_work.count("wafer_mdc_drain_worker(") == 1
    assert "work->cleanup_attempted = 1U;" in fail_work
    assert "work->cleanup_succeeded = 1U;" in fail_work
    assert "work->poisoned = 1U;" in fail_work

    complete_pending = function_body("wafer_ssc_complete_pending")
    assert complete_pending.count("wafer_mdc_drain_worker(") == 1
    assert "wafer_ssc_fail_work(" not in complete_pending
    assert "work->cleanup_attempted = 1U;" in complete_pending
    assert "work->poisoned = 1U;" in complete_pending
    assert "work->cleanup_succeeded = 1U;" not in complete_pending
    setup_issue = function_body("wafer_ssc_issue_setup_rdma")
    assert setup_issue.count("return wafer_ssc_fail_work(work);") == 2
    assert "wafer_ssc_track_accepted(work);" in setup_issue
    archive_issue = function_body("wafer_ssc_issue_archive_wdma")
    assert archive_issue.count("return wafer_ssc_fail_work(work);") == 2
    assert "wafer_ssc_track_accepted(work);" in archive_issue
    seed_chunk = function_body("wafer_ssc_issue_seed_chunk")
    assert "wafer_ssc_issue_setup_rdma(" in seed_chunk
    assert "wafer_ssc_complete_pending(work)" in seed_chunk
    seed_owned = function_body("wafer_ssc_seed_owned_spm")
    assert seed_owned.count("wafer_ssc_issue_setup_rdma(") == 2
    assert seed_owned.count("wafer_ssc_complete_pending(work)") == 3
    issue_lane = function_body("wafer_ssc_issue_lane")
    assert "return wafer_ssc_fail_work(work);" in issue_lane
    assert "wafer_ssc_track_accepted(work);" in issue_lane
    issue_measured = function_body("wafer_ssc_issue_measured")
    assert issue_measured.count("return wafer_ssc_fail_work(work);") == 2
    assert issue_measured.count("wafer_ssc_complete_pending(work)") == 3
    matched_wdma_body = function_body("wafer_ssc_issue_matched_wdma")
    assert "return wafer_ssc_fail_work(work);" in matched_wdma_body
    assert "wafer_ssc_complete_pending(work)" in matched_wdma_body
    archive_body = function_body("wafer_ssc_archive_owned_spm")
    assert "wafer_ssc_issue_archive_wdma(" in archive_body
    assert archive_body.count("wafer_ssc_complete_pending(work)") == 2

    execute_row = function_body("wafer_ssc_execute_row")
    setup_failure = execute_row.index(
        "if (!wafer_ssc_seed_owned_spm("
    )
    setup_cleanup_record = execute_row.index(
        "WAFER_SSC_CLEANUP_SETUP", setup_failure
    )
    setup_failure_return = execute_row.index(
        "return (uint32_t)row_record[WAFER_SSC_ROW_STATUS];",
        setup_cleanup_record,
    )
    measured_failure = execute_row.index(
        "if (!wafer_ssc_issue_measured("
    )
    measured_cleanup_record = execute_row.index(
        "WAFER_SSC_CLEANUP_MEASURED_PAIR", measured_failure
    )
    measured_restore = execute_row.index(
        "wafer_ssc_restore_spm_port_pmu(", measured_cleanup_record
    )
    matched_failure = execute_row.index(
        "if (!(scope_flags & WAFER_SSC_PORT_SCOPE_BOUNDARY_STABLE)"
    )
    matched_cleanup_record = execute_row.index(
        "WAFER_SSC_CLEANUP_MATCHED_WDMA", matched_failure
    )
    matched_restore = execute_row.index(
        "wafer_ssc_restore_spm_port_pmu(", matched_cleanup_record
    )
    archive_failure = execute_row.index(
        "if (!wafer_ssc_archive_owned_spm("
    )
    archive_cleanup_record = execute_row.index(
        "WAFER_SSC_CLEANUP_ARCHIVE", archive_failure
    )
    archive_failure_return = execute_row.index(
        "return (uint32_t)row_record[WAFER_SSC_ROW_STATUS];",
        archive_cleanup_record,
    )
    assert (
        setup_failure
        < setup_cleanup_record
        < setup_failure_return
        < measured_failure
        < measured_cleanup_record
        < measured_restore
        < matched_failure
        < matched_cleanup_record
        < matched_restore
        < archive_failure
        < archive_cleanup_record
        < archive_failure_return
    )
    assert execute_row.count("WAFER_SSC_STATUS_CLEANUP_FAILED") == 4
    measured_begin = device.index(
        "WaferMDCPMU before = wafer_mdc_read_pmu();"
    )
    measured_end = device.index(
        "WaferMDCPMU after = wafer_mdc_read_pmu();",
        measured_begin,
    )
    port_before = device.index(
        "WaferSSCPortPMU port_before = "
        "wafer_ssc_read_spm_port_pmu();"
    )
    port_boundary = device.index(
        "port_boundary = wafer_ssc_read_spm_port_pmu();",
        port_before,
    )
    matched_wdma = device.index(
        "wafer_ssc_issue_matched_wdma(",
        port_boundary,
    )
    port_after = device.index(
        "port_after = wafer_ssc_read_spm_port_pmu();",
        matched_wdma,
    )
    restore = device.index(
        "wafer_ssc_restore_spm_port_pmu(",
        port_after,
    )
    readback = device.index(
        "wafer_ssc_archive_owned_spm(", restore
    )
    assert (
        port_before
        < measured_begin
        < port_boundary
        < measured_end
        < matched_wdma
        < port_after
        < restore
        < readback
    )


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
