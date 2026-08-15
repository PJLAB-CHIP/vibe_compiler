#!/usr/bin/env python3
"""Deterministic v10 primary-program evidence for the offline analyzer."""

from __future__ import annotations

from typing import Any


NCC_ENGINES = ("CT", "NE", "RDMA", "WDMA", "TDMA")
FINAL_DIGEST = "sha256:" + "f" * 64
RECORD_ABI = "wafer-tx81-profiler-record-v4"


def _cost_metric(
    value: int | None,
    *,
    knowledge: str = "known",
    reason: str = "none",
) -> dict[str, str | None]:
    return {
        "knowledge": knowledge,
        "value": str(value) if value is not None else None,
        "reason": reason,
    }


def _static_cost_model() -> dict[str, Any]:
    tile_rows = []
    for tile_id in range(16):
        directional = {
            direction: _cost_metric(
                None,
                knowledge="unavailable",
                reason="unresolved-noc-route",
            )
            for direction in ("north", "east", "south", "west")
        }
        tile_rows.append(
            {
                "card_id": 0,
                "tile_id": tile_id,
                "launch_slot": tile_id,
                "work": {
                    "npu_f16_bf16_logical_ops": _cost_metric(820_000),
                    "npu_other_logical_ops": _cost_metric(0),
                    "vector_f16_bf16_logical_ops": _cost_metric(5_856),
                    "vector_f32_logical_ops": _cost_metric(0),
                    "vector_other_logical_ops": _cost_metric(0),
                    "ddr_read_bytes": _cost_metric(1_400),
                    "ddr_write_bytes": _cost_metric(1_556),
                    "spm_movement_bytes": _cost_metric(3_980),
                    "noc_transmit_bytes": _cost_metric(1_280),
                    "noc_receive_bytes": _cost_metric(1_280),
                    "directional_noc_transmit_bytes": directional,
                },
            }
        )
    return {
        "model": "tx81-static-throughput-lower-bound",
        "scope": "complete-final-instruction-program-per-physical-tile",
        "rates": {
            "card_ddr_bytes_per_second": 200_000_000_000,
            "directional_noc_bytes_per_second": 128_000_000_000,
            "f16_bf16_npu_logical_ops_per_second_per_tile": (
                8_000_000_000_000
            ),
            "f16_bf16_vector_logical_ops_per_second_per_tile": 64_000_000_000,
            "f32_vector_logical_ops_per_second_per_tile": 32_000_000_000,
            "spm_movement_bytes_per_second": None,
        },
        "tiles": tile_rows,
    }


def _kernel_launch() -> dict[str, Any]:
    return {
        "kind": "kernel",
        "form": "grid",
        "entry_abi": "tile-major-pointer-table",
        "phases": ["main"],
    }


def _counter(delta: int) -> dict[str, int | bool]:
    start = 10_000
    end = start + delta
    return {
        "start": start,
        "end": end,
        "recovery": end,
        "stable": True,
        "enabled": True,
    }


def _pmu_tile(tile: int) -> dict[str, Any]:
    execution_base = 84 + tile
    return {
        "card_id": 0,
        "tile_id": tile,
        "launch_slot": tile,
        "aggregates": {
            "statistics_window": _counter(820 + tile),
            "fu": _counter(390 + tile),
            "ct": _counter(execution_base),
            "ne": _counter(execution_base + 11),
            "rdma": _counter(execution_base + 22),
            "wdma": _counter(execution_base + 33),
            "tdma": _counter(execution_base + 44),
            "scalar": _counter(75 + tile),
        },
        "workers": [
            {
                "worker": worker,
                "engines": [
                    {
                        "engine": engine,
                        "instructions": _counter(1 if worker == 0 else 0),
                        "blocking": _counter(0),
                    }
                    for engine in NCC_ENGINES
                ],
            }
            for worker in range(3)
        ],
    }


def _trace_tile(tile: int) -> dict[str, Any]:
    entry_begin = 4_900 + tile * 100
    events: list[dict[str, Any]] = []

    def append_event(event: dict[str, Any]) -> None:
        event["sequence"] = len(events)
        events.append(event)

    def append_site(site_id: int, site_begin: int, site_end: int) -> None:
        append_event(
            {
                "site_id": site_id,
                "sub_index": 0,
                "engine": None,
                "kind": "target-site",
                "observed_begin_cycle": 0,
                "observed_end_cycle": 0,
                "counter_delta": 0,
                "site_begin_cycle": site_begin,
                "site_end_cycle": site_end,
                "operation_begin_cycle": 0,
                "operation_end_cycle": 0,
                "observation_count": 0,
                "observed_span_valid": False,
                "site_span_valid": True,
                "operation_span_valid": False,
                "positive_delta": False,
                "attribution_ambiguous": False,
                "ncc_counter_valid": None,
                "observation_status": None,
                "worker": None,
                "wait_scope": None,
                "dte_counter_valid": None,
                "dte_role": None,
            }
        )

    for site_id, engine in enumerate(NCC_ENGINES):
        site_begin = entry_begin + 60 + site_id * 42
        operation_begin = site_begin + 6
        observed_begin = operation_begin + 3
        append_site(site_id, site_begin, site_begin + 31)
        append_event(
            {
                "site_id": site_id,
                "sub_index": 1,
                "engine": engine,
                "kind": "ncc-command",
                "observed_begin_cycle": observed_begin,
                "observed_end_cycle": observed_begin + 7 + site_id,
                "counter_delta": 1 + site_id,
                "site_begin_cycle": site_begin,
                "site_end_cycle": site_begin + 31,
                "operation_begin_cycle": operation_begin,
                "operation_end_cycle": operation_begin + 11,
                "observation_count": 2 + site_id,
                "observed_span_valid": True,
                "site_span_valid": True,
                "operation_span_valid": True,
                "positive_delta": True,
                "attribution_ambiguous": False,
                "ncc_counter_valid": True,
                "observation_status": "engine-delta-bounded",
                "worker": 0,
                "wait_scope": None,
                "dte_counter_valid": None,
                "dte_role": None,
            }
        )

    completion_site_begin = entry_begin + 278
    completion_site_end = entry_begin + 312
    append_site(5, completion_site_begin, completion_site_end)
    append_event(
        {
            "site_id": 5,
            "sub_index": 1,
            "engine": None,
            "kind": "ncc-completion-wait",
            "observed_begin_cycle": 0,
            "observed_end_cycle": 0,
            "counter_delta": 0,
            "site_begin_cycle": completion_site_begin,
            "site_end_cycle": completion_site_end,
            "operation_begin_cycle": entry_begin + 283,
            "operation_end_cycle": entry_begin + 307,
            "observation_count": 4,
            "observed_span_valid": False,
            "site_span_valid": True,
            "operation_span_valid": True,
            "positive_delta": False,
            "attribution_ambiguous": False,
            "ncc_counter_valid": None,
            "observation_status": None,
            "worker": 0,
            "wait_scope": "worker",
            "dte_counter_valid": None,
            "dte_role": None,
        }
    )

    append_site(6, entry_begin + 316, entry_begin + 322)

    dte_role = "send" if tile % 2 == 0 else "receive"
    if dte_role == "send":
        issue_site_begin = entry_begin + 326
        issue_site_end = entry_begin + 371
        append_site(7, issue_site_begin, issue_site_end)
        append_event(
            {
                "site_id": 7,
                "sub_index": 1,
                "engine": "DIRECT_DTE",
                "kind": "direct-dte-issue",
                "observed_begin_cycle": issue_site_begin,
                "observed_end_cycle": issue_site_end,
                "counter_delta": 0,
                "site_begin_cycle": issue_site_begin,
                "site_end_cycle": issue_site_end,
                "operation_begin_cycle": entry_begin + 331,
                "operation_end_cycle": entry_begin + 366,
                "observation_count": 2,
                "observed_span_valid": True,
                "site_span_valid": True,
                "operation_span_valid": True,
                "positive_delta": False,
                "attribution_ambiguous": False,
                "ncc_counter_valid": None,
                "observation_status": None,
                "worker": None,
                "wait_scope": None,
                "dte_counter_valid": False,
                "dte_role": dte_role,
            }
        )
        issue_phase_specs = (
            (
                "direct-dte-peer-ready-wait",
                entry_begin + 331,
                entry_begin + 345,
            ),
            ("direct-dte-setup-issue", entry_begin + 350, entry_begin + 365),
        )
        for sub_index, (kind, operation_begin, operation_end) in enumerate(
            issue_phase_specs, start=2
        ):
            append_event(
                {
                    "site_id": 7,
                    "sub_index": sub_index,
                    "engine": "DIRECT_DTE",
                    "kind": kind,
                    "observed_begin_cycle": 0,
                    "observed_end_cycle": 0,
                    "counter_delta": 0,
                    "site_begin_cycle": issue_site_begin,
                    "site_end_cycle": issue_site_end,
                    "operation_begin_cycle": operation_begin,
                    "operation_end_cycle": operation_end,
                    "observation_count": 0,
                    "observed_span_valid": False,
                    "site_span_valid": True,
                    "operation_span_valid": True,
                    "positive_delta": False,
                    "attribution_ambiguous": False,
                    "ncc_counter_valid": None,
                    "observation_status": None,
                    "worker": None,
                    "wait_scope": None,
                    "dte_counter_valid": None,
                    "dte_role": dte_role,
                }
            )

    dte_site_begin = entry_begin + 375
    dte_site_end = entry_begin + 446
    append_site(8, dte_site_begin, dte_site_end)
    append_event(
        {
            "site_id": 8,
            "sub_index": 1,
            "engine": "DIRECT_DTE",
            "kind": "direct-dte-wait",
            "observed_begin_cycle": dte_site_begin,
            "observed_end_cycle": dte_site_end,
            "counter_delta": 0,
            "site_begin_cycle": dte_site_begin,
            "site_end_cycle": dte_site_end,
            "operation_begin_cycle": entry_begin + 380,
            "operation_end_cycle": entry_begin + 441,
            "observation_count": 2,
            "observed_span_valid": True,
            "site_span_valid": True,
            "operation_span_valid": True,
            "positive_delta": False,
            "attribution_ambiguous": False,
            "ncc_counter_valid": None,
            "observation_status": None,
            "worker": None,
            "wait_scope": None,
            "dte_counter_valid": False,
            "dte_role": dte_role,
        }
    )
    phase_specs = (
        (
            "direct-dte-completion-wait",
            entry_begin + 385,
            entry_begin + 420,
        ),
        ("direct-dte-cleanup", entry_begin + 425, entry_begin + 440),
    )
    for sub_index, (kind, operation_begin, operation_end) in enumerate(
        phase_specs, start=2
    ):
        append_event(
            {
                "site_id": 8,
                "sub_index": sub_index,
                "engine": "DIRECT_DTE",
                "kind": kind,
                "observed_begin_cycle": 0,
                "observed_end_cycle": 0,
                "counter_delta": 0,
                "site_begin_cycle": dte_site_begin,
                "site_end_cycle": dte_site_end,
                "operation_begin_cycle": operation_begin,
                "operation_end_cycle": operation_end,
                "observation_count": 0,
                "observed_span_valid": False,
                "site_span_valid": True,
                "operation_span_valid": True,
                "positive_delta": False,
                "attribution_ambiguous": False,
                "ncc_counter_valid": None,
                "observation_status": None,
                "worker": None,
                "wait_scope": None,
                "dte_counter_valid": None,
                "dte_role": dte_role,
            }
        )

    append_site(9, entry_begin + 450, entry_begin + 460)
    entry_end = entry_begin + 500
    return {
        "card_id": 0,
        "tile_id": tile,
        "launch_slot": tile,
        "entry_begin_cycle": entry_begin,
        "entry_end_cycle": entry_end,
        "capacity": len(events) + 2,
        "count": len(events),
        "counted_event_count": len(events),
        "next_sequence": len(events),
        "dropped_event_count": 0,
        "record_flags": 71,
        "trace_state": 2,
        "overflow": False,
        "cost_summary": {
            "ncc_pmu_sample_cycles": 12,
            "dte_pmu_sample_cycles": 7,
            "event_bookkeeping_cycles": 9,
            "status_poll_cycles": 13,
            "site_hook_cycles": 11,
            "completion_loop_bookkeeping_cycles": 5,
            "entry_setup_cycles": 8,
            "entry_teardown_cycles": 6,
        },
        "events": events,
    }


def _sites() -> list[dict[str, Any]]:
    specifications = (
        ("ncc-command", "CT", "compute.ct", "wafer_tx81_tsm_ct_execute"),
        ("ncc-command", "NE", "compute.gemm", "wafer_tx81_tsm_ne_execute"),
        ("ncc-command", "RDMA", "input.read", "wafer_tx81_tsm_rdma_execute"),
        ("ncc-command", "WDMA", "output.write", "wafer_tx81_tsm_wdma_execute"),
        ("ncc-command", "TDMA", "workspace.move", "wafer_tx81_tsm_tdma_execute"),
        (
            "ncc-completion",
            None,
            "completion.ncc",
            "wafer_tx81_ncc_join",
        ),
        (
            "direct-dte-control",
            None,
            "communication.direct-dte.setup",
            "wafer_tx81_direct_dte_setup",
        ),
        (
            "direct-dte-issue",
            "DIRECT_DTE",
            "communication.direct-dte.issue",
            "wafer_tx81_direct_dte_send_issue_v3",
        ),
        (
            "direct-dte-wait",
            "DIRECT_DTE",
            "communication.direct-dte.wait",
            "wafer_tx81_direct_dte_wait",
        ),
        (
            "direct-dte-control",
            None,
            "communication.direct-dte.cleanup",
            "wafer_tx81_direct_dte_cleanup",
        ),
    )
    return [
        {
            "card_id": 0,
            "tile_id": tile,
            "launch_slot": tile,
            "site_id": site_id,
            "site_kind": site_kind,
            "correlation_key": correlation,
            "engine": engine,
            "target_call_ordinal": 10 + site_id,
            "target_call_symbol": symbol,
            "position": f"fixture.mlir:{10 + site_id * 4}:3",
        }
        for tile in range(16)
        for site_id, (site_kind, engine, correlation, symbol) in enumerate(
            specifications
        )
    ]


def make_evidence(*, permute_bindings: bool = False) -> dict[str, Any]:
    """Build one complete, primary-program profile evidence object."""

    target_identity = "wafer-tx81-single-card"
    experiment = {
        "clock": [
            {
                "card_id": 0,
                "tile_id": tile,
                "launch_slot": tile,
                "slope": 1.0,
                "offset": -tile * 101.0,
                "uncertainty": 2.0 + tile * 0.05,
                "round_trips": 32,
                "valid": True,
                "monotonic": True,
            }
            for tile in range(16)
        ],
        "trace": {
            "complete": True,
            "tiles": [_trace_tile(tile) for tile in range(16)],
        },
        "pmu": {"tiles": [_pmu_tile(tile) for tile in range(16)]},
    }
    evidence = {
        "schema": "wafer.profile.evidence",
        "schema_version": 11,
        "run_id": "fixture-primary-program",
        "program": {
            "program_manifest_sha256": FINAL_DIGEST,
            "record_abi": RECORD_ABI,
            "target_identity": target_identity,
            "launch": _kernel_launch(),
            "card_count": 1,
            "tile_count": 16,
            "site_correlation_basis": (
                "target-call-ordinal-ssa-position"
            ),
        },
        "topology": [
            {
                "card_id": 0,
                "tile_id": tile,
                "launch_slot": tile,
                "x": tile % 4,
                "y": tile // 4,
            }
            for tile in range(16)
        ],
        "measurement": {
            "samples": [
                {
                    "sample_id": "primary",
                    "sample_index": 0,
                    "device_elapsed_ns": 8_400,
                    "device_timer_kind": "tx-stream-events",
                    "host_submit_ns": 260_000,
                    "host_launch_to_completion_ns": 1_018_000,
                    "completion_observation_resolution_ns": 100,
                }
            ]
        },
        "output_validation": {
            "mode": "external-expected",
            "resources": [
                {
                    "scope": {"kind": "tile", "card_id": 0, "tile_id": tile},
                    "role": "output",
                    "role_index": 0,
                    "bytes": 4096,
                    "reference_sha256": "sha256:" + f"{tile + 1:064x}",
                    "external_expected_comparison": "exact",
                    "primary_output_validated": True,
                    "diagnostic_captures_match_primary": True,
                }
                for tile in range(16)
            ],
        },
        "sites": _sites(),
        "validity": {
            "environment": True,
            "profile_instrumentation": True,
            "measurement_basis": True,
        },
        "static_cost_model": _static_cost_model(),
        "experiment": experiment,
    }
    if permute_bindings:
        def launch_slot(tile_id: int) -> int:
            return 1 - tile_id if tile_id < 2 else tile_id

        tile_arrays = (
            evidence["topology"],
            evidence["static_cost_model"]["tiles"],
            evidence["experiment"]["clock"],
            evidence["experiment"]["trace"]["tiles"],
            evidence["experiment"]["pmu"]["tiles"],
        )
        for rows in tile_arrays:
            for row in rows:
                row["launch_slot"] = launch_slot(int(row["tile_id"]))
        for site in evidence["sites"]:
            site["launch_slot"] = launch_slot(int(site["tile_id"]))
    return evidence
