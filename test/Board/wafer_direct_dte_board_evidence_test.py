#!/usr/bin/env python3
"""Focused tests for strict Direct-DTE manifest/board evidence checks."""

from __future__ import annotations

import copy
import dataclasses
import unittest

import wafer_direct_dte_board_evidence as evidence


def valid_manifest() -> dict[str, object]:
    resources: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    completions: list[dict[str, object]] = []
    for rank in range(evidence.RANK_COUNT):
        input_id = rank * 3
        output_id = input_id + 1
        status_id = input_id + 2
        resources.extend(
            [
                {
                    "id": input_id,
                    "rank": rank,
                    "role": "user_input",
                    "role_index": 0,
                    "type": {"dtype": "i8", "shape": [256]},
                    "bytes": 256,
                    "alignment": 256,
                    "access": "read_only",
                    "host_visible": True,
                },
                {
                    "id": output_id,
                    "rank": rank,
                    "role": "output",
                    "role_index": 0,
                    "type": {"dtype": "i8", "shape": [256]},
                    "bytes": 256,
                    "alignment": 256,
                    "access": "write_only",
                    "host_visible": True,
                },
                {
                    "id": status_id,
                    "rank": rank,
                    "role": "transport_status",
                    "role_index": 0,
                    "type": {"dtype": "u32", "shape": [1]},
                    "bytes": evidence.DIRECT_DTE_STATUS_BYTES,
                    "alignment": evidence.DIRECT_DTE_STATUS_ALIGNMENT,
                    "access": "read_write",
                    "host_visible": False,
                },
            ]
        )
        entries.append(
            {
                "id": rank,
                "rank": rank,
                "module": 0,
                "slots": [
                    {
                        "ordinal": 0,
                        "resource": input_id,
                        "access": "read_only",
                    },
                    {
                        "ordinal": 1,
                        "resource": output_id,
                        "access": "write_only",
                    },
                    {
                        "ordinal": 2,
                        "resource": status_id,
                        "access": "read_write",
                    },
                ],
                "terminal_completion": rank,
                "transport": {
                    "kind": "direct_dte",
                    "status_resource": status_id,
                    "status_abi": evidence.DIRECT_DTE_STATUS_ABI,
                    "host_watchdog_required": True,
                },
            }
        )
        completions.append(
            {"id": rank, "rank": rank, "kind": "entry_return"}
        )
    return {
        "schema_version": evidence.SCHEMA_VERSION,
        "target": {
            "launch": {
                "kind": "kernel",
                "form": "cluster",
                "entry_abi": "rank-major-pointer-table-v1",
                "phases": ["prepare", "main"],
            }
        },
        "rank_count": evidence.RANK_COUNT,
        "resources": resources,
        "entries": entries,
        "completions": completions,
    }


def valid_stdout() -> str:
    lines = [
        "board_stage: preflight",
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
    ]
    lines.extend(
        f"terminal_completion: {rank} kind=entry_return rank={rank}"
        for rank in range(evidence.RANK_COUNT)
    )
    lines.extend(
        [
            f"invocation_ranks: {evidence.RANK_COUNT}",
            "launch_pattern: cluster-prepare-main-x16",
            "logical_tile_domain: 0..15",
            "board_execution: true",
        ]
    )
    return "\n".join(lines) + "\n"


class DirectDTEManifestEvidenceTest(unittest.TestCase):
    def test_accepts_exact_status_v2_watchdog_completion_domain(self) -> None:
        result = evidence.validate_direct_dte_manifest(valid_manifest())
        self.assertEqual(
            dict(result.status_resource_by_rank),
            {rank: rank * 3 + 2 for rank in range(evidence.RANK_COUNT)},
        )
        self.assertEqual(
            result.terminal_completion_by_rank,
            tuple((rank, rank) for rank in range(evidence.RANK_COUNT)),
        )
        self.assertEqual(
            result.host_watchdog_ranks, tuple(range(evidence.RANK_COUNT))
        )

    def test_rejects_incomplete_or_malformed_status_domain(self) -> None:
        mutations = []

        missing_status = valid_manifest()
        missing_status["resources"] = missing_status["resources"][:-1]
        mutations.append(missing_status)

        wrong_storage = valid_manifest()
        wrong_storage["resources"][2]["bytes"] = 4
        mutations.append(wrong_storage)

        host_visible = valid_manifest()
        host_visible["resources"][2]["host_visible"] = True
        mutations.append(host_visible)

        unbound = valid_manifest()
        unbound["entries"][0]["slots"] = unbound["entries"][0]["slots"][:-1]
        mutations.append(unbound)

        for manifest in mutations:
            with self.subTest(manifest=manifest), self.assertRaises(
                RuntimeError
            ):
                evidence.validate_direct_dte_manifest(manifest)

    def test_rejects_wrong_abi_watchdog_or_completion_relation(self) -> None:
        wrong_abi = valid_manifest()
        wrong_abi["entries"][7]["transport"]["status_abi"] = (
            "wafer-direct-dte-status-v1"
        )

        no_watchdog = valid_manifest()
        no_watchdog["entries"][11]["transport"][
            "host_watchdog_required"
        ] = False

        wrong_status_resource = valid_manifest()
        wrong_status_resource["entries"][4]["transport"][
            "status_resource"
        ] = 2

        wrong_completion = valid_manifest()
        wrong_completion["entries"][15]["terminal_completion"] = 14

        missing_completion = valid_manifest()
        missing_completion["completions"].pop()

        for manifest in (
            wrong_abi,
            no_watchdog,
            wrong_status_resource,
            wrong_completion,
            missing_completion,
        ):
            with self.subTest(manifest=manifest), self.assertRaises(
                RuntimeError
            ):
                evidence.validate_direct_dte_manifest(manifest)


class DirectDTEBoardEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = evidence.validate_direct_dte_manifest(valid_manifest())

    def test_command_requires_one_bounded_all_rank_board_deadline(self) -> None:
        command = [
            "wafer-run",
            "--package-dir",
            "package",
            "--all-ranks",
            "--board",
            "--completion-timeout-ms",
            "60000",
        ]
        self.assertEqual(
            evidence.validate_direct_dte_board_command(command), 60000
        )
        for mutation in (
            [item for item in command if item != "--all-ranks"],
            command + ["--no-card"],
            command[:-2],
            command[:-1] + ["0"],
            command + ["--completion-timeout-ms", "60000"],
        ):
            with self.subTest(command=mutation), self.assertRaises(
                RuntimeError
            ):
                evidence.validate_direct_dte_board_command(mutation)

    def test_success_log_reports_exact_status_observation_gap(self) -> None:
        result = evidence.validate_direct_dte_board_output(
            valid_stdout(), self.manifest, completion_timeout_ms=60000
        )
        self.assertTrue(result.runtime_all_rank_success_enforced)
        self.assertFalse(result.has_observed_all_rank_success)
        self.assertIsNone(result.observed_status_by_rank)
        self.assertEqual(
            result.status_observation_gap, evidence.STATUS_OBSERVATION_GAP
        )
        self.assertIn(
            "do not expose the 16 per-rank", result.status_observation_gap
        )

    def test_accepts_only_explicit_exact_all_rank_success_records(self) -> None:
        observations = [
            evidence.DirectDTEStatusObservation(
                rank=rank,
                resource=dict(self.manifest.status_resource_by_rank)[rank],
                status_abi=evidence.DIRECT_DTE_STATUS_ABI,
                value=evidence.DIRECT_DTE_STATUS_SUCCESS,
            )
            for rank in range(evidence.RANK_COUNT)
        ]
        result = evidence.validate_direct_dte_board_output(
            valid_stdout(),
            self.manifest,
            completion_timeout_ms=60000,
            status_observations=observations,
        )
        self.assertTrue(result.has_observed_all_rank_success)
        self.assertIsNone(result.status_observation_gap)
        self.assertEqual(
            [row.rank for row in result.observed_status_by_rank],  # type: ignore[union-attr]
            list(range(evidence.RANK_COUNT)),
        )

        partial = observations[:-1]
        duplicate = observations + [copy.copy(observations[-1])]
        wrong_resource = list(observations)
        wrong_resource[5] = dataclasses.replace(
            wrong_resource[5], resource=wrong_resource[4].resource
        )
        wrong_abi = list(observations)
        wrong_abi[6] = dataclasses.replace(
            wrong_abi[6], status_abi="wafer-direct-dte-status-v1"
        )
        transport_error = list(observations)
        transport_error[15] = dataclasses.replace(
            transport_error[15], value=2
        )
        for rows in (
            partial,
            duplicate,
            wrong_resource,
            wrong_abi,
            transport_error,
        ):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                evidence.validate_direct_dte_board_output(
                    valid_stdout(),
                    self.manifest,
                    completion_timeout_ms=60000,
                    status_observations=rows,
                )

    def test_rejects_partial_or_wrong_terminal_completion_output(self) -> None:
        missing_terminal = valid_stdout().replace(
            "terminal_completion: 15 kind=entry_return rank=15\n", ""
        )
        duplicate_rank = valid_stdout().replace(
            "terminal_completion: 15 kind=entry_return rank=15",
            "terminal_completion: 15 kind=entry_return rank=14",
        )
        wrong_completion = valid_stdout().replace(
            "terminal_completion: 7 kind=entry_return rank=7",
            "terminal_completion: 9 kind=entry_return rank=7",
        )
        missing_stage = valid_stdout().replace(
            "board_stage: completion\n", ""
        )
        for stdout in (
            missing_terminal,
            duplicate_rank,
            wrong_completion,
            missing_stage,
        ):
            with self.subTest(stdout=stdout), self.assertRaises(RuntimeError):
                evidence.validate_direct_dte_board_output(
                    stdout, self.manifest, completion_timeout_ms=60000
                )


if __name__ == "__main__":
    unittest.main()
