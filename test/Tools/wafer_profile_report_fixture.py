#!/usr/bin/env python3
"""Deterministic v4 final-artifact evidence for the offline analyzer."""

from __future__ import annotations

from typing import Any


NCC_ENGINES = ("CT", "NE", "RDMA", "WDMA", "TDMA")
FINAL_DIGEST = "sha256:" + "f" * 64


def _kernel_launch() -> dict[str, Any]:
    return {
        "kind": "kernel",
        "form": "grid",
        "entry_abi": "rank-major-pointer-table-v1",
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
        "tile": tile,
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
    for sequence, engine in enumerate(NCC_ENGINES):
        begin = entry_begin + 100 + sequence * 41
        events.append(
            {
                "sequence": sequence,
                "site_id": sequence,
                "sub_index": 0,
                "engine": engine,
                "observed_begin_cycle": begin,
                "observed_end_cycle": begin + 7 + sequence,
                "counter_delta": 1 + sequence,
                "activity_valid": True,
                "dte_counter_valid": None,
                "dte_role": None,
            }
        )
    # Deliberately differs from the summary capture span. The report must use
    # this trace-local axis for every activity-window coordinate.
    entry_end = entry_begin + 500
    return {
        "tile": tile,
        "entry_begin_cycle": entry_begin,
        "entry_end_cycle": entry_end,
        "capacity": len(events) + 2,
        "count": len(events),
        "preflight_count": len(events),
        "next_sequence": len(events),
        "dropped_event_count": 0,
        "record_flags": 71,
        "trace_state": 2,
        "overflow": False,
        "events": events,
    }


def _summary_tiles() -> list[dict[str, int]]:
    return [
        {
            "tile": tile,
            "entry_begin": 2_000 + tile * 101,
            "entry_end": 3_000 + tile * 102,
        }
        for tile in range(16)
    ]


def _sites() -> list[dict[str, Any]]:
    specifications = (
        ("CT", "compute.ct", "wafer_tx81_tsm_ct_execute"),
        ("NE", "compute.gemm", "wafer_tx81_tsm_ne_execute"),
        ("RDMA", "input.read", "wafer_tx81_tsm_rdma_execute"),
        ("WDMA", "output.write", "wafer_tx81_tsm_wdma_execute"),
        ("TDMA", "workspace.move", "wafer_tx81_tsm_tdma_execute"),
    )
    return [
        {
            "tile": tile,
            "site_id": site_id,
            "correlation_key": correlation,
            "engine": engine,
            "target_call_ordinal": 10 + site_id,
            "target_call_symbol": symbol,
            "position": f"fixture.mlir:{10 + site_id * 4}:3",
        }
        for tile in range(16)
        for site_id, (engine, correlation, symbol) in enumerate(specifications)
    ]


def make_evidence() -> dict[str, Any]:
    """Build one complete, final-artifact-only profile evidence object."""

    target_profile = "wafer-tx81-single-card-kernel-v1"
    sample_durations = (
        1_018_000,
        1_006_000,
        1_012_000,
        1_001_000,
        1_009_000,
        1_004_000,
        1_014_000,
        1_007_000,
        1_011_000,
        1_003_000,
    )
    experiment = {
        "artifact": {
            "digest": FINAL_DIGEST,
            "target_profile": target_profile,
            "launch": _kernel_launch(),
            "execution_ranks": 16,
        },
        "clock": [
            {
                "tile": tile,
                "slope": 1.0,
                "offset": -tile * 101.0,
                "uncertainty": 2.0 + tile * 0.05,
                "round_trips": 32,
                "valid": True,
                "monotonic": True,
            }
            for tile in range(16)
        ],
        "summary": {"tiles": _summary_tiles()},
        "trace": {
            "complete": True,
            "tiles": [_trace_tile(tile) for tile in range(16)],
        },
        "pmu": {"tiles": [_pmu_tile(tile) for tile in range(16)]},
    }
    return {
        "schema": "wafer.profile.evidence",
        "schema_version": 4,
        "run_id": "fixture-final-artifact",
        "identity": {
            "production_manifest_sha256": FINAL_DIGEST,
            "profile_companion_schema_version": 2,
            "target_profile": target_profile,
            "launch": _kernel_launch(),
            "execution_ranks": 16,
            "site_correlation_basis": (
                "heuristic-target-call-signature-occurrence-v1"
            ),
        },
        "topology": [
            {"tile": tile, "x": tile % 8, "y": tile // 8}
            for tile in range(16)
        ],
        "measurement": {
            "samples": [
                {
                    "sample_id": f"final-s{index}",
                    "sample_index": index,
                    "host_elapsed_ns": elapsed,
                    "completion_observation_resolution_ns": 100,
                }
                for index, elapsed in enumerate(sample_durations)
            ]
        },
        "output_validation": {
            "mode": "external-expected",
            "resources": [
                {
                    "logical_rank": tile,
                    "role": "output",
                    "role_index": 0,
                    "bytes": 4096,
                    "reference_sha256": "sha256:" + f"{tile + 1:064x}",
                    "external_expected_comparison": "exact",
                    "production_repeats_exact": True,
                    "diagnostic_captures_exact": True,
                }
                for tile in range(16)
            ],
        },
        "sites": _sites(),
        "validity": {
            "environment": True,
            "package_companion": True,
            "measurement_basis": True,
        },
        "experiment": experiment,
    }
