#!/usr/bin/env python3
"""Deterministic v2 evidence fixture for the offline profile analyzer."""

from __future__ import annotations

from typing import Any


ENGINES = ("CT", "NE", "RDMA", "WDMA", "TDMA")


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


def _pmu_tile(tile: int, candidate: str) -> dict[str, Any]:
    winner = candidate == "winner"
    execution_base = 84 if winner else 105
    instruction_counts = {
        "CT": 1,
        "NE": 1,
        "RDMA": 1 if winner else 2,
        "WDMA": 1,
        "TDMA": 1,
    }
    return {
        "tile": tile,
        "aggregates": {
            "statistics_window": _counter((820 if winner else 1_000) + tile),
            "fu": _counter((390 if winner else 470) + tile),
            "ct": _counter(execution_base + tile),
            "ne": _counter(execution_base + 11 + tile),
            "rdma": _counter(execution_base + 22 + tile),
            "wdma": _counter(execution_base + 33 + tile),
            "tdma": _counter(execution_base + 44 + tile),
            "scalar": _counter((75 if winner else 90) + tile),
        },
        "workers": [
            {
                "worker": worker,
                "engines": [
                    {
                        "engine": engine,
                        "instructions": _counter(
                            instruction_counts[engine] if worker == 0 else 0
                        ),
                        "blocking": _counter(0),
                    }
                    for engine in ENGINES
                ],
            }
            for worker in range(3)
        ],
    }


def _trace_tile(tile: int, candidate: str) -> dict[str, Any]:
    if candidate == "baseline":
        # A baseline-only early site shifts every later candidate-local ID.
        site_ids = (0, 1, 2, 2, 3, 4)
        site_engines = {
            0: "CT",
            1: "TDMA",
            2: "RDMA",
            3: "NE",
            4: "WDMA",
        }
    else:
        site_ids = (0, 1, 2, 3, 4)
        site_engines = {
            0: "CT",
            1: "RDMA",
            2: "NE",
            3: "WDMA",
            4: "TDMA",
        }
    events = []
    for sequence, site_id in enumerate(site_ids):
        begin = 5_000 + tile * 100 + sequence * 41
        events.append(
            {
                "sequence": sequence,
                "site_id": site_id,
                "sub_index": 0 if sequence < 2 else sequence - 1,
                "engine": site_engines[site_id],
                "begin_cycle": begin,
                "return_cycle": begin + 7 + sequence,
                "raw_result": 1,
                "valid": True,
            }
        )
    return {
        "tile": tile,
        "entry_begin_cycle": 4_900 + tile * 100,
        "entry_end_cycle": events[-1]["return_cycle"] + 100,
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


def _measurement_samples(
    candidate: str,
    *,
    winner_scale: float,
    held_out_winner_scale: float | None,
) -> list[dict[str, Any]]:
    orders = ("ABBA", "BAAB", "ABBA", "BAAB", "ABBA")
    block_durations = (1_000, 1_008, 996, 1_004, 1_010)
    result: list[dict[str, Any]] = []
    wanted_letter = "A" if candidate == "baseline" else "B"
    for block, order in enumerate(orders):
        scale = 1.0
        if candidate == "winner":
            scale = (
                held_out_winner_scale
                if block == 4 and held_out_winner_scale is not None
                else winner_scale
            )
        positions = [
            position
            for position, letter in enumerate(order)
            if letter == wanted_letter
        ]
        for ordinal, position in enumerate(positions):
            sample_jitter = -2 if ordinal == 0 else 2
            result.append(
                {
                    "sample_id": f"{candidate}-b{block}-p{position}",
                    "candidate": candidate,
                    "block": block,
                    "position": position,
                    "host_elapsed_ns": int(
                        (block_durations[block] + sample_jitter)
                        * scale
                        * 1_000
                    ),
                    "completion_observation_resolution_ns": 100,
                }
            )
    return result


def _summary_tiles(candidate: str, *, winner_scale: float) -> list[dict[str, Any]]:
    scale = winner_scale if candidate == "winner" else 1.0
    tiles = []
    for tile in range(16):
        # The calibration offset removes tile * 101.  Device cycles remain
        # exact u64 integers; the analyzer must not route them through float.
        begin = 2_000 + tile * 101
        duration = round((1_000 + tile) * scale)
        tiles.append(
            {
                "tile": tile,
                "entry_begin": begin,
                "entry_end": begin + duration,
            }
        )
    return tiles


def make_evidence(
    *,
    winner_scale: float = 0.8,
    held_out_winner_scale: float | None = None,
) -> dict[str, Any]:
    """Build a complete valid profile record.

    ``winner_scale < 1`` is an improvement, ``> 1`` is a regression.  A
    separate held-out scale can deliberately disagree with the four training
    blocks.
    """

    target_profile = "wafer-tx81-single-card-kernel-v1"
    blocks = [
        {
            "block": block,
            "held_out": block == 4,
            "order": "ABBA" if block % 2 == 0 else "BAAB",
        }
        for block in range(5)
    ]
    site_specs = {
        "compute.ct": {
            "engine": "CT",
            "target_call_ordinal": 10,
            "target_call_symbol": "wafer_tx81_tsm_ct_execute",
            "position": "fixture.mlir:10:3",
        },
        "input.read": {
            "engine": "RDMA",
            "target_call_ordinal": 11,
            "target_call_symbol": "wafer_tx81_tsm_rdma_execute",
            "position": "fixture.mlir:14:3",
        },
        "compute.gemm": {
            "engine": "NE",
            "target_call_ordinal": 12,
            "target_call_symbol": "wafer_tx81_tsm_ne_execute",
            "position": "fixture.mlir:18:3",
        },
        "output.write": {
            "engine": "WDMA",
            "target_call_ordinal": 13,
            "target_call_symbol": "wafer_tx81_tsm_wdma_execute",
            "position": "fixture.mlir:22:3",
        },
        "baseline.prefill": {
            "engine": "TDMA",
            "target_call_ordinal": 14,
            "target_call_symbol": "wafer_tx81_tsm_tdma_execute",
            "position": "fixture.mlir:12:3",
        },
        "winner.workspace": {
            "engine": "TDMA",
            "target_call_ordinal": 14,
            "target_call_symbol": "wafer_tx81_tsm_tdma_execute",
            "position": "fixture.mlir:26:3",
        },
    }
    site_layout = {
        "baseline": (
            (0, "compute.ct"),
            (1, "baseline.prefill"),
            (2, "input.read"),
            (3, "compute.gemm"),
            (4, "output.write"),
        ),
        "winner": (
            (0, "compute.ct"),
            (1, "input.read"),
            (2, "compute.gemm"),
            (3, "output.write"),
            (4, "winner.workspace"),
        ),
    }
    sites = []
    for candidate, rows in site_layout.items():
        for tile in range(16):
            for site_id, correlation_key in rows:
                sites.append(
                    {
                        "candidate": candidate,
                        "tile": tile,
                        "site_id": site_id,
                        "correlation_key": correlation_key,
                        **site_specs[correlation_key],
                    }
                )

    experiments: dict[str, Any] = {}
    for candidate in ("baseline", "winner"):
        experiments[candidate] = {
            "artifact": {
                "digest": f"sha256:{candidate}-fixture",
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
            "summary": {
                "tiles": _summary_tiles(
                    candidate, winner_scale=winner_scale
                )
            },
            "trace": {
                "complete": True,
                "tiles": [
                    _trace_tile(tile, candidate) for tile in range(16)
                ],
            },
            "pmu": {
                "tiles": [
                    _pmu_tile(tile, candidate) for tile in range(16)
                ]
            },
        }

    return {
        "schema": "wafer.profile.evidence",
        "schema_version": 2,
        "run_id": "fixture-improved",
        "identity": {
            "production_manifest_sha256": "sha256:winner-fixture",
            "profile_companion_schema_version": 1,
            "target_profile": target_profile,
            "launch": _kernel_launch(),
            "execution_ranks": 16,
            "baseline_same_as_winner": False,
            "site_correlation_basis": (
                "heuristic-target-call-signature-occurrence-v1"
            ),
        },
        # Match the runtime inventory fixture's 8x2 coordinates.  The report
        # must consume physical coordinates rather than invent rank % 4.
        "topology": [
            {
                "tile": tile,
                "x": tile % 8,
                "y": tile // 8,
            }
            for tile in range(16)
        ],
        "measurement": {
            "blocks": blocks,
            "samples": sorted(
                _measurement_samples(
                    "baseline",
                    winner_scale=winner_scale,
                    held_out_winner_scale=held_out_winner_scale,
                )
                + _measurement_samples(
                    "winner",
                    winner_scale=winner_scale,
                    held_out_winner_scale=held_out_winner_scale,
                ),
                key=lambda row: (row["block"], row["position"]),
            ),
        },
        "output_validation": {
            "mode": "external-exact",
            "resources": [
                {
                    "logical_rank": tile,
                    "role": "output",
                    "role_index": 0,
                    "bytes": 4096,
                    "reference_sha256": (
                        "sha256:"
                        + f"{tile + 1:064x}"
                    ),
                    "external_expected_exact": True,
                    "production_winner_repeat_exact": True,
                    "candidate_equivalent_exact": True,
                }
                for tile in range(16)
            ],
        },
        "sites": sites,
        "validity": {
            "environment": True,
            "package_companion": True,
            "measurement_basis": True,
        },
        "experiments": experiments,
    }
