#!/usr/bin/env python3
"""Validate complete DTE/SPM sweep and fail-closed TMNOC disposition."""

from __future__ import annotations

import pathlib
import re

import wafer_transport_pmu_calibration_catalog as catalog


def main() -> int:
    assert catalog.PAYLOAD_SWEEP == (16, 32, 64, 256, 4096)
    assert len(catalog.CASES) == 30
    assert {(case.mode, case.payload_bytes) for case in catalog.CASES} == {
        (mode, payload)
        for mode in catalog.TRANSPORT_PMU_MODES
        for payload in catalog.PAYLOAD_SWEEP
    }
    assert catalog.ERROR_PATH_MODES == (7, 8, 9, 12)
    assert catalog.ASYNC_SENDER_MODES == (10, 11)
    assert catalog.RAW_REMOTE_MODE_IDS == tuple(range(15, 39))
    assert (
        catalog.RAW_REMOTE_MULTICAST_CASES
        is catalog.RAW_MULTIDEST_CASES
    )
    assert (
        catalog.RAW_REMOTE_MULTICAST_CASE_BY_MODE
        is catalog.RAW_MULTIDEST_CASE_BY_MODE
    )
    raw_cases = catalog.RAW_REMOTE_MULTICAST_CASES
    assert len(raw_cases) == 24
    assert len({case.key for case in raw_cases}) == 24
    assert len({case.mode for case in raw_cases}) == 24
    assert {case.semantic for case in raw_cases} == {
        "broadcast",
        "scatter",
        "shuffle",
    }
    for semantic in catalog.RAW_MULTIDEST_SEMANTICS:
        semantic_cases = tuple(
            case for case in raw_cases if case.semantic == semantic
        )
        if semantic == "shuffle":
            assert {case.fanout for case in semantic_cases} == {1}
            assert {
                (case.shuffle_sections, case.target_layout)
                for case in semantic_cases
            } == {
                (sections, layout)
                for sections in (2, 4, 8, 15)
                for layout in ("adjacent", "interleaved")
            }
            assert {
                case.target_ranks for case in semantic_cases
            } == {(1,), (8,)}
            assert {
                case.dest_num_register_value for case in semantic_cases
            } == {0}
        else:
            assert {
                (case.fanout, case.target_layout)
                for case in semantic_cases
            } == {
                (fanout, layout)
                for fanout in (2, 4, 8, 15)
                for layout in ("adjacent", "interleaved")
            }
        assert {case.disposition for case in semantic_cases} == {
            "pending-board-observation"
        }
        assert {
            case.verification_scope for case in semantic_cases
        } == {"board-device-owner-backed-raw-dte-registers"}
    assert {
        case.semantic: case.raw_mode for case in raw_cases
    } == {"broadcast": 2, "scatter": 1, "shuffle": 3}
    assert {
        case.semantic: case.mode_register_value for case in raw_cases
    } == {"broadcast": 2, "scatter": 0x101, "shuffle": 3}
    assert all(
        len(case.target_ranks) == case.fanout
        and len(set(case.target_ranks)) == case.fanout
        and 0 not in case.target_ranks
        and case.source_span_bytes <= catalog.RAW_MULTIDEST_PAYLOAD_BYTES
        and case.element_bytes > 0
        and case.dest_num_register_value == case.fanout - 1
        and case.dest_num_encoding == "zero-based-final-active-slot"
        and case.as_dict()["key"] == case.name
        for case in raw_cases
    )
    assert catalog.raw_multidest_target_ranks(4, "adjacent") == (1, 2, 3, 4)
    assert catalog.raw_multidest_target_ranks(4, "interleaved") == (1, 8, 15, 7)
    assert {
        (case.mode, case.payload_bytes)
        for case in catalog.CONTRACT_CASES
        if case.disposition == "board-observation"
    } == {
        (7, 16),
        (8, 16),
        (9, 16),
        (10, 64),
        (11, 64),
        (12, 16),
    }
    sender_controls = catalog.CALIBRATION_LEAF_BINDINGS[
        "direct-dte-sender-async-controls"
    ]
    assert {
        (
            case.mode,
            case.payload_bytes,
            case.transport_bytes,
            case.repetitions,
        )
        for case in sender_controls
    } == {
        (
            10,
            64,
            catalog.ASYNC_SENDER_TRANSPORT_BYTES,
            catalog.ASYNC_SENDER_REPETITIONS,
        ),
        (
            11,
            64,
            catalog.ASYNC_SENDER_TRANSPORT_BYTES,
            catalog.ASYNC_SENDER_REPETITIONS,
        ),
    }
    assert catalog.ASYNC_SENDER_REPETITIONS >= 3
    assert {case.disposition for case in sender_controls} == {
        "board-observation"
    }
    assert {case.verification_scope for case in sender_controls} == {
        "board-device-receiver-first-raw-async"
    }
    assert sum(case.split == "calibration" for case in catalog.CASES) == 6
    assert sum(case.split == "held-out" for case in catalog.CASES) == 24
    pmu_observations = catalog.CALIBRATION_LEAF_BINDINGS[
        "dte-spm-counter-payload-sweep"
    ]
    assert len(pmu_observations) == len(catalog.CASES) == 30
    assert {
        (case.case_id, case.mode, case.payload_bytes, case.split)
        for case in pmu_observations
    } == {
        (case.case_id, case.mode, case.payload_bytes, case.split)
        for case in catalog.CASES
    }
    assert {
        case.disposition for case in pmu_observations
    } == {"board-observation"}
    assert all(
        "uncalibrated" in case.reason for case in pmu_observations
    )
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
    assert {
        case.mode
        for case in catalog.CALIBRATION_LEAF_BINDINGS[
            "direct-dte-four-source-fanin"
        ]
    } == {catalog.FOUR_SOURCE_FANIN_MODE}
    assert {
        key: len(catalog.CALIBRATION_LEAF_BINDINGS[key])
        for key in (
            "direct-dte-raw-broadcast",
            "direct-dte-raw-scatter",
            "direct-dte-raw-shuffle",
        )
    } == {
        "direct-dte-raw-broadcast": 8,
        "direct-dte-raw-scatter": 8,
        "direct-dte-raw-shuffle": 8,
    }
    assert catalog.CALIBRATION_LEAF_BINDINGS
    assert all(catalog.CALIBRATION_LEAF_BINDINGS.values())
    real_objects = {
        id(case)
        for case in (
            catalog.CASES
            + catalog.TRANSPORT_PMU_OBSERVATIONS
            + catalog.CONTRACT_CASES
            + catalog.COUNTER_DISPOSITIONS
            + catalog.RAW_REMOTE_MULTICAST_CASES
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
        "dte-raw-gather-unencodable",
        "host-readback-before-terminal",
        "outer-timeout-stops-batch",
    }
    direct_host_negative = catalog.CALIBRATION_LEAF_BINDINGS[
        "direct-dte-host-static-negative"
    ]
    assert {case.name for case in direct_host_negative} == {
        "dte-invalid-coordinate",
        "dte-raw-gather-unencodable",
    }
    assert {case.verification_scope for case in direct_host_negative} == {
        "host-prelaunch-verifier"
    }
    direct_error_observation = catalog.CALIBRATION_LEAF_BINDINGS[
        "direct-dte-device-error-observation"
    ]
    assert {case.name for case in direct_error_observation} == {
        "dte-source-reuse-before-send-event",
        "dte-destination-reuse-before-receive-event",
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
        if disposition.disposition == "board-observation"
    }
    tmnoc = catalog.COUNTERS_BY_NAME["tmnoc"]
    assert tmnoc.disposition == "static-negative"
    assert "inventing offsets" in tmnoc.reason

    repo = pathlib.Path(__file__).resolve().parents[2]
    cmake = (repo / "test" / "CMakeLists.txt").read_text()
    pending_cmake_specs = tuple(
        (int(mode), name)
        for mode, name in re.findall(
            r'^\s*"(\d+):(dte-(?:four-source-fanin-correctness|'
            r'raw-(?:(?:broadcast|scatter)-fanout\d+-(?:adjacent|'
            r'interleaved)|shuffle-source1d-sections\d+-target\d+)))"$',
            cmake,
            re.M,
        )
    )
    assert pending_cmake_specs == (
        (
            catalog.FOUR_SOURCE_FANIN_MODE,
            catalog.MODE_NAMES[catalog.FOUR_SOURCE_FANIN_MODE],
        ),
        *tuple(
            (case.mode, case.name)
            for case in catalog.RAW_REMOTE_MULTICAST_CASES
        ),
    )
    assert "wafer-runtime-${_wafer_pending_dte_case}-no-card" in cmake
    assert "wafer-board-${_wafer_pending_dte_case}" in cmake
    assert "--host-contract-only" in cmake
    assert cmake.count("--mode ${_wafer_pending_dte_mode}") == 2
    assert 'RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"' in (
        cmake.split(
            "foreach(_wafer_pending_dte_case_spec IN LISTS",
            maxsplit=2,
        )[2]
    )
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
    assert "wafer_direct_dte_sender.active || byte_count == 0" in crt_source
    assert "local_fsm_id >= WAFER_DIRECT_DTE_MAX_RECEIVERS" in crt_source
    assert "if (receiver->active)" in crt_source
    wait_body = crt_source.split(
        "void wafer_tx81_direct_dte_wait(uint64_t event)", maxsplit=1
    )[1].split("void wafer_tx81_direct_dte_finish", maxsplit=1)[0]
    wait_behavior, wait_epilogue = wait_body.split("\ndone:", maxsplit=1)
    assert wait_behavior.rstrip().endswith("wafer_direct_dte_set_error();")
    assert "wafer_profile_direct_dte_end();" in wait_epilogue
    send_wait = wait_body.split(
        "if (event == WAFER_DIRECT_DTE_SEND_EVENT)", maxsplit=1
    )[1].split(
        "if (event >= WAFER_DIRECT_DTE_RECV_EVENT_BASE", maxsplit=1
    )[0]
    assert "direct_sync_wait(" not in send_wait
    assert "direct_dte_attach(" not in send_wait
    assert "direct_dte_send_async(" not in send_wait
    assert "wafer_direct_dte_issue_sender(event, false)" in send_wait
    assert send_wait.index("direct_dte_wait_done(") < send_wait.index(
        "direct_dte_release("
    )
    issue_helper = crt_source.split(
        "static bool wafer_direct_dte_issue_sender(", maxsplit=1
    )[1].split(
        "void wafer_tx81_direct_dte_send_issue_v3(uint64_t event)", maxsplit=1
    )[0]
    assert issue_helper.index("direct_sync_wait(") < issue_helper.index(
        "direct_dte_attach("
    )
    assert issue_helper.index("direct_dte_attach(") < issue_helper.index(
        "direct_dte_send_async("
    )
    assert "direct_dte_wait_done(" not in issue_helper
    send_issue_v3 = crt_source.split(
        "void wafer_tx81_direct_dte_send_issue_v3(uint64_t event)", maxsplit=1
    )[1].split(
        "uint64_t wafer_tx81_direct_dte_recv_prepare", maxsplit=1
    )[0]
    assert "wafer_direct_dte_issue_sender(event, true)" in send_issue_v3

    probe_source = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_dte_ncc_execution_probe.c"
    ).read_text()
    host_driver_source = (
        repo
        / "test"
        / "Board"
        / "wafer_board_dte_ncc_execution_probe_test.py"
    ).read_text()
    for legacy_ordinary_call in (
        "wafer_tx81_rdma(",
        "wafer_tx81_wdma(",
        "wafer_tx81_elementwise_add(",
    ):
        assert legacy_ordinary_call not in probe_source
    assert "wafer_tx81_rdma_v3(" in probe_source
    assert "wafer_tx81_wdma_v3(" in probe_source
    assert "wafer_tx81_elementwise_add_v3(" in probe_source
    assert 'parser.add_argument("--host-contract-only"' in host_driver_source
    assert 'TARGET_PROFILE = "wafer-tx81-single-card-kernel-v3"' in (
        host_driver_source
    )
    assert "f\"--target-profile={TARGET_PROFILE}\"" in host_driver_source
    assert "mode not in (*ISOLATED_DTE_MODES, *PENDING_DTE_MODES)" in (
        host_driver_source
    )
    assert "each pending fan-in/raw-multidestination case must run alone" in (
        host_driver_source
    )
    assert "raw_multidest_elf_verification: passed" in host_driver_source
    assert (
        "final probe ELF dropped the remote-SPM route transform"
        in host_driver_source
    )
    probe_main = probe_source.split(
        "wafer_tx81_dte_ncc_execution_probe(", maxsplit=1
    )[1]
    assert "get_spm_memory_mapping" not in probe_source
    assert "wafer_probe_seed_guarded_region(" in probe_source
    assert "wafer_probe_capture_results(output_ddr, mode, rank);" in probe_main
    reuse = probe_main.split(
        "case WAFER_PROBE_DTE_REUSE_AFTER_EVENTS:", maxsplit=1
    )[1].split(
        "case WAFER_PROBE_DTE_TWO_DESTINATION_BROADCAST:", maxsplit=1
    )[0]
    assert reuse.count("wafer_probe_dte_ring(") == 2
    assert reuse.index(
        "wafer_probe_capture_region(output_ddr, 3"
    ) < reuse.rindex(
        "wafer_probe_dte_ring("
    )
    assert reuse.index(
        "wafer_probe_capture_region(output_ddr, 3"
    ) < reuse.index("wafer_tx81_local_fence();", reuse.index(
        "wafer_probe_capture_region(output_ddr, 3"
    ))
    broadcast = probe_source.split(
        "static void wafer_probe_dte_two_destination_broadcast(", maxsplit=1
    )[1].split(
        "static uint64_t wafer_probe_dte_fanin_destination", maxsplit=1
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
    )[1].split(
        "static uint32_t wafer_probe_dte_reuse_before_recv_event", maxsplit=1
    )[0]
    assert early_reuse.count("wafer_tx81_direct_dte_send_prepare(") == 2
    assert early_reuse.count("wafer_tx81_direct_dte_recv_prepare(") == 1
    assert early_reuse.index("rejected =") < early_reuse.index(
        "wafer_tx81_direct_dte_wait(send)"
    )
    assert early_reuse.index("wafer_tx81_direct_dte_wait(send)") < (
        early_reuse.index("wafer_tx81_direct_dte_wait(receive)")
    )
    early_receive_reuse = probe_source.split(
        "static uint32_t wafer_probe_dte_reuse_before_recv_event(", maxsplit=1
    )[1].split("static uint32_t wafer_probe_dte_invalid_fsm", maxsplit=1)[0]
    assert early_receive_reuse.count(
        "wafer_tx81_direct_dte_recv_prepare("
    ) == 2
    assert early_receive_reuse.count(
        "wafer_tx81_direct_dte_send_prepare("
    ) == 1
    assert "WAFER_PROBE_SPM_DTE_RECV_SECOND" in early_receive_reuse
    assert early_receive_reuse.index("rejected =") < early_receive_reuse.index(
        "wafer_tx81_direct_dte_send_prepare("
    )
    assert early_receive_reuse.index(
        "wafer_tx81_direct_dte_wait(send)"
    ) < early_receive_reuse.index("wafer_tx81_direct_dte_wait(receive)")
    invalid_fsm = probe_source.split(
        "static uint32_t wafer_probe_dte_invalid_fsm(", maxsplit=1
    )[1].split("static void wafer_probe_publish_header", maxsplit=1)[0]
    assert "successor, 4, 0" in invalid_fsm
    assert "predecessor, 4)" in invalid_fsm
    unknown_wait = probe_main.split(
        "case WAFER_PROBE_DTE_WAIT_UNKNOWN_EVENT_ERROR:", maxsplit=1
    )[1].split("}", maxsplit=1)[0]
    assert "wafer_tx81_direct_dte_wait(UINT64_C(0xdeadbeef));" in unknown_wait
    assert probe_source.count(
        "wafer_tx81_direct_dte_begin_after_prepare(status_ddr,"
    ) >= 1
    assert '#include "direct_dte_and_fsm.h"' in probe_source
    assert "sizeof(DirectDTESendInfo) == 64U" in probe_source
    assert "offsetof(DirectDTESendInfo, stride_iterations) == 28U" in (
        probe_source
    )
    assert "offsetof(DirectDTESendInfo, dte_node) == 56U" in probe_source
    raw_async = probe_source.split(
        "wafer_probe_dte_sender_raw_async(", maxsplit=1
    )[1].split(
        "wafer_probe_raw_multidest_user_id(", maxsplit=1
    )[0]
    assert raw_async.index(
        "wafer_tx81_direct_dte_recv_prepare("
    ) < raw_async.index("direct_sync_wait(")
    assert raw_async.index("direct_sync_wait(") < raw_async.index(
        "direct_dte_attach("
    )
    assert raw_async.index("direct_dte_attach(") < raw_async.index(
        "direct_dte_send_async("
    )
    first_ct = raw_async.index("wafer_tx81_elementwise_add_v3(")
    assert raw_async.count("wafer_tx81_elementwise_add_v3(") == 2
    assert raw_async.index("direct_dte_send_async(") < first_ct
    assert first_ct < raw_async.index("direct_dte_wait_done(")
    assert raw_async.index("direct_dte_wait_done(") < raw_async.index(
        "direct_dte_release("
    )
    assert raw_async.index("direct_dte_release(") < raw_async.index(
        "wafer_tx81_direct_dte_wait(receive)"
    )
    assert raw_async.index(
        "wafer_tx81_direct_dte_wait(receive)"
    ) < raw_async.rindex("wafer_tx81_local_fence()")
    assert raw_async.index(
        "wafer_tx81_direct_dte_wait(receive)"
    ) < raw_async.rindex("wafer_tx81_elementwise_add_v3(")
    assert "result.send_result = direct_dte_send_async(&info)" in raw_async
    assert "result.wait_result = direct_dte_wait_done(&info)" in raw_async
    assert "result.release_result = direct_dte_release(info.dte_node)" in (
        raw_async
    )
    assert "WAFER_PROBE_RAW_ASYNC_RC_MARKER" in probe_source
    assert '#include "dte/dte_cfg.h"' in probe_source
    assert '#include "dte/mod_dte.h"' in probe_source
    for owner_offset in (
        "offsetof(sct_dte_cfg_s, src_addr) == 0U",
        "offsetof(sct_dte_cfg_s, dst_addr_0) == 8U",
        "offsetof(sct_dte_cfg_s, user_id_0) == 16U",
        "offsetof(sct_dte_cfg_s, sct_dte_block_mode) == 20U",
        "offsetof(sct_dte_cfg_s, sct_dte_block_length) == 24U",
        "offsetof(sct_dte_cfg_s, sct_dte_block_dest_num) == 28U",
        "offsetof(sct_dte_cfg_s, dst_addr_others) == 80U",
        "offsetof(sct_dte_cfg_s, user_id_others) == 328U",
    ):
        assert owner_offset in probe_source
    raw_multidest_program = probe_source.split(
        "wafer_probe_raw_multidest_program(", maxsplit=1
    )[1].split(
        "wafer_probe_dte_raw_multidest(", maxsplit=1
    )[0]
    raw_multidest_route = probe_source.split(
        "wafer_probe_raw_multidest_route_destination(", maxsplit=1
    )[1].split(
        "wafer_probe_raw_multidest_program(", maxsplit=1
    )[0]
    assert "(mapped_destination << 17U)" in raw_multidest_route
    assert "UINT64_C(0x00ffff0000000000)" in raw_multidest_route
    assert "DirectDTESendInfo" not in raw_multidest_program
    assert "for (uint32_t index = 0; index < fanout; ++index)" in (
        raw_multidest_program
    )
    route_call = (
        "wafer_probe_raw_multidest_route_destination(destination);"
    )
    assert route_call in raw_multidest_program
    assert raw_multidest_program.index(route_call) < (
        raw_multidest_program.index(
            "node->dst_cfg[index].dst_addr = destination;"
        )
    )
    assert "node->dst_cfg[index].dst_addr = destination;" in (
        raw_multidest_program
    )
    assert "node->dst_cfg[index].dst_id = user_id;" in raw_multidest_program
    raw_multidest_user_id = probe_source.split(
        "wafer_probe_raw_multidest_user_id(", maxsplit=1
    )[1].split(
        "wafer_probe_raw_multidest_program(", maxsplit=1
    )[0]
    assert "user_id.field.tgt_npu = 1;" in raw_multidest_user_id
    assert "user_id.field.switch_ddr = 0;" in raw_multidest_user_id
    assert "user_id.field.switch_ddr = 1;" not in raw_multidest_user_id
    raw_multidest_mode = probe_source.split(
        "wafer_probe_raw_multidest_mode_register(", maxsplit=1
    )[1].split(
        "static uint32_t wafer_probe_raw_multidest_fanout(", maxsplit=1
    )[0]
    assert "register_mode.field.mode = raw_mode;" in raw_multidest_mode
    assert "register_mode.field.sg_flag = raw_mode == 1U ? 1U : 0U;" in (
        raw_multidest_mode
    )
    assert "GR_DTE_MODE, mode_register" in raw_multidest_program
    assert "GR_DTE_LENGTH, element_bytes" in raw_multidest_program
    assert "GR_DTE_DEST_NUM, fanout - 1U" in raw_multidest_program
    assert "raw_mode == 3U ? fanout != 1U : fanout < 2U" in (
        raw_multidest_program
    )
    assert "shuffle_sections - 1U" in raw_multidest_program
    assert "wafer_probe_raw_dte_receive_bytes(mode)" in probe_source
    assert "zero-based final active slot encoding" in probe_source
    assert "keeps software dst_cnt" in probe_source
    assert "writes zero to this five-bit register" in probe_source
    assert "GR_DTE_STRIDE0" in raw_multidest_program
    assert "GR_DTE_ITERATION0" in raw_multidest_program
    assert "GR_DTE_CMD_VALID, UINT32_C(1)" in raw_multidest_program
    raw_multidest_execution = probe_source.split(
        "wafer_probe_dte_raw_multidest(", maxsplit=1
    )[1].split(
        "static void wafer_probe_dte_two_destination_broadcast(", maxsplit=1
    )[0]
    assert raw_multidest_execution.index("direct_sync_wait(") < (
        raw_multidest_execution.index("direct_dte_attach(")
    )
    assert "wait_info.dte_node = node;" in raw_multidest_execution
    assert "direct_dte_wait_done(&wait_info)" in raw_multidest_execution
    assert "mod_kuiper_dte_trig_send" not in raw_multidest_execution
    assert "mod_kuiper_dte_check_send_status" not in raw_multidest_execution
    assert "direct_dte_release(node)" in raw_multidest_execution
    assert "WAFER_PROBE_RAW_MULTIDEST_RC_MARKER" in probe_source
    fanin = probe_source.split(
        "static uint32_t wafer_probe_dte_four_source_fanin(", maxsplit=1
    )[1].split(
        "static uint32_t wafer_probe_dte_reuse_before_send_event(", maxsplit=1
    )[0]
    assert fanin.count("wafer_tx81_direct_dte_recv_prepare(") == 4
    assert fanin.count("wafer_tx81_direct_dte_wait(receive") == 4
    assert "WAFER_PROBE_FANIN_SOURCE_COUNT" in fanin
    assert "wafer_tx81_direct_dte_send_prepare(" in fanin
    split_read = probe_source.split(
        "static uint64_t wafer_probe_mmio_read64(", maxsplit=1
    )[1].split("static WaferProbePmu wafer_probe_read_pmu", maxsplit=1)[0]
    high_before = split_read.index(
        "uint32_t high_before = wafer_probe_mmio_read32("
    )
    low = split_read.index(
        "low = wafer_probe_mmio_read32(", high_before
    )
    high_after = split_read.index(
        "high_after = wafer_probe_mmio_read32(", low
    )
    assert high_before < low < high_after
    assert "if (high_before == high_after)" in split_read
    assert "*stable_mask |= stable_bit;" in split_read
    assert "return ((uint64_t)high_after << 32) | low;" in split_read
    assert "WAFER_PROBE_STABLE_RETRIES UINT32_C(8)" in probe_source

    assert catalog.modulo_counter_delta(1, 0xFFFFFFFE, 32) == 3
    assert (
        catalog.modulo_counter_delta(
            1, 0xFFFFFFFFFFFFFFFE, 64
        )
        == 3
    )
    assert catalog.modulo_counter_delta(17, 5, 64) == 12
    rollover = catalog.stable_high_low_high_read(
        (
            (0, 0, 1),
            (1, 0, 1),
        )
    )
    assert rollover == catalog.SplitCounterRead(
        value=0x0000000100000000,
        stable=True,
        attempts=2,
    )
    pre_rollover = catalog.stable_high_low_high_read(
        ((0, 0xFFFFFFFF, 0),)
    )
    assert pre_rollover == catalog.SplitCounterRead(
        value=0x00000000FFFFFFFF,
        stable=True,
        attempts=1,
    )
    unstable_attempts = tuple(
        (index & 1, index, (index & 1) ^ 1)
        for index in range(catalog.SPLIT_COUNTER_STABLE_RETRIES)
    )
    exhausted = catalog.stable_high_low_high_read(
        unstable_attempts + ((9, 7, 9),)
    )
    assert exhausted.stable is False
    assert exhausted.attempts == catalog.SPLIT_COUNTER_STABLE_RETRIES
    assert exhausted.value == (
        (unstable_attempts[-1][2] << 32) | unstable_attempts[-1][1]
    )

    increasing = catalog.classify_payload_series(
        {size: (size, size) for size in catalog.PAYLOAD_SWEEP}
    )
    assert increasing["payload_relation"] == "strictly-increasing"
    assert increasing["integer_scale_candidate"] == 1
    flat = catalog.classify_payload_series(
        {size: (7, 7) for size in catalog.PAYLOAD_SWEEP}
    )
    assert flat["payload_relation"] == "nondecreasing"
    assert flat["integer_scale_candidate"] is None
    missing = catalog.classify_payload_series({16: (1,), 32: (2,)})
    assert missing["state"] == "inconclusive"
    assert missing["missing_payload_bytes"] == (64, 256, 4096)
    late_drop = catalog.classify_payload_series(
        {
            16: (16,),
            32: (32,),
            64: (64,),
            256: (256,),
            4096: (1,),
        }
    )
    assert late_drop["payload_relation"] == "non-monotonic"

    print(
        "wafer_transport_pmu_calibration_catalog_test: "
        "cases=30 contracts=13 direct_host_negative=2 "
        "direct_error_observation=4 sender_async_controls=2 "
        "fanin=1 raw_remote_multicast=24 unsafe_isolation=1 "
        "board_counters=6 wrap_boundaries=5 "
        "tmnoc_static_negative=1 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
