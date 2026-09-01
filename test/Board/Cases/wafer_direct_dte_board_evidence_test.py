#!/usr/bin/env python3
"""Focused tests for strict Direct-DTE manifest/board evidence checks."""

from __future__ import annotations

import copy
import dataclasses
import unittest

import wafer_direct_dte_board_evidence as evidence
import wafer_runtime_launch_contract as runtime_launch


def valid_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "program": {"id": 0},
        "target": {
            "identity": "wafer-tx81-single-card",
            "runtime_abi": "wafer-tx81-kernel",
            "module_format": "elf-riscv64",
        },
        "launch": dict(runtime_launch.GRID_KERNEL_LAUNCH),
        "card_count": 1,
        "tile_count": evidence.TILE_COUNT,
        "program_data": {
            "relative_path": "data/program-data.bin",
            "total_bytes": 0,
            "base_alignment": 1,
            "digest": (
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649"
                "349ca495991b7852b855"
            ),
        },
        "program_tensors": [],
        "target_tensors": [],
        "inputs": [],
        "outputs": [],
        "modules": [
            {
                "id": 0,
                "path": "modules/tile.so",
                "digest": (
                    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649"
                    "349ca495991b7852b855"
                ),
                "format": "elf-riscv64",
                "exports": [
                    {"role": "main", "symbol": "entry"},
                ],
            }
        ],
        "entries": [
            {
                "id": tile,
                "card_id": 0,
                "tile_id": tile,
                "launch_slot": tile,
                "module": 0,
                "arguments": [],
                "completion": "return_after_local_drain",
                "transport": {"kind": "none"},
            }
            for tile in range(evidence.TILE_COUNT)
        ],
    }
    runtime_launch.configure_direct_dte_tile_package(
        manifest,
        ports=(
            runtime_launch.SharedBoundaryPortSpec(
                table="inputs",
                role_index=0,
                logical_dtype="i8",
                logical_shape=[256],
                dtype="i8",
                layout="tensor",
                shape=[256],
                bytes=256,
                alignment=256,
                access="read_only",
            ),
            runtime_launch.SharedBoundaryPortSpec(
                table="outputs",
                role_index=0,
                logical_dtype="i8",
                logical_shape=[256],
                dtype="i8",
                layout="tensor",
                shape=[256],
                bytes=256,
                alignment=256,
                access="write_only",
            ),
        ),
        status_abi=evidence.DIRECT_DTE_STATUS_ABI,
        status_bytes=evidence.DIRECT_DTE_STATUS_BYTES,
        status_alignment=evidence.DIRECT_DTE_STATUS_ALIGNMENT,
        context="Direct-DTE evidence test",
    )
    return manifest


def valid_stdout() -> str:
    lines = [
        "board_stage: validation",
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
    ]
    lines.extend(
        "completion: return_after_local_drain tile_id=" + str(tile)
        for tile in range(evidence.TILE_COUNT)
    )
    lines.extend(
        [
            f"invocation_tiles: {evidence.TILE_COUNT}",
            "launch_pattern: cluster-x16",
            "physical_tile_domain: 0..15",
            "board_execution: true",
        ]
    )
    return "\n".join(lines) + "\n"


class DirectDTEManifestEvidenceTest(unittest.TestCase):
    def test_accepts_exact_status_watchdog_completion_domain(self) -> None:
        result = evidence.validate_direct_dte_manifest(valid_manifest())
        self.assertEqual(
            dict(result.status_abi_by_tile),
            {
                tile: evidence.DIRECT_DTE_STATUS_ABI
                for tile in range(evidence.TILE_COUNT)
            },
        )
        self.assertEqual(
            result.completion_by_tile,
            tuple(
                (tile, "return_after_local_drain")
                for tile in range(evidence.TILE_COUNT)
            ),
        )
        self.assertEqual(
            result.host_watchdog_tiles, tuple(range(evidence.TILE_COUNT))
        )

    def test_rejects_incomplete_or_malformed_status_domain(self) -> None:
        mutations = []

        missing_status = valid_manifest()
        missing_status["entries"][15]["arguments"] = (
            missing_status["entries"][15]["arguments"][:-1]
        )
        mutations.append(missing_status)

        wrong_storage = valid_manifest()
        wrong_storage["entries"][2]["arguments"][-1]["bytes"] = 4
        mutations.append(wrong_storage)

        host_bound = valid_manifest()
        host_bound["entries"][2]["arguments"][-1]["kind"] = "external_input"
        host_bound["entries"][2]["arguments"][-1]["port"] = 0
        mutations.append(host_bound)

        unbound = valid_manifest()
        unbound["entries"][0]["arguments"] = (
            unbound["entries"][0]["arguments"][:-1]
        )
        mutations.append(unbound)

        for manifest in mutations:
            with self.subTest(manifest=manifest), self.assertRaises(
                RuntimeError
            ):
                evidence.validate_direct_dte_manifest(manifest)

    def test_rejects_wrong_abi_watchdog_or_completion_relation(self) -> None:
        wrong_abi = valid_manifest()
        wrong_abi["entries"][7]["transport"]["status_abi"] = (
            "unsupported-status-abi"
        )

        no_watchdog = valid_manifest()
        no_watchdog["entries"][11]["transport"][
            "host_watchdog_required"
        ] = False

        wrong_status_abi = valid_manifest()
        wrong_status_abi["entries"][4]["transport"]["status_abi"] = (
            "mismatched-status-abi"
        )

        wrong_completion = valid_manifest()
        wrong_completion["entries"][15]["completion"] = "unsupported"

        duplicate_tile = valid_manifest()
        duplicate_tile["entries"][15]["tile_id"] = 14

        for manifest in (
            wrong_abi,
            no_watchdog,
            wrong_status_abi,
            wrong_completion,
            duplicate_tile,
        ):
            with self.subTest(manifest=manifest), self.assertRaises(
                RuntimeError
            ):
                evidence.validate_direct_dte_manifest(manifest)


class DirectDTEBoardEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = evidence.validate_direct_dte_manifest(valid_manifest())

    def test_command_requires_one_bounded_board_deadline(self) -> None:
        command = [
            "wafer-run",
            "--package-dir",
            "package",
            "--board",
            "--completion-timeout-ms",
            "60000",
        ]
        self.assertEqual(
            evidence.validate_direct_dte_board_command(command), 60000
        )
        for mutation in (
            command + ["--all-Tiles"],
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
        self.assertTrue(result.runtime_all_tile_success_enforced)
        self.assertFalse(result.has_observed_all_tile_success)
        self.assertIsNone(result.observed_status_by_tile)
        self.assertEqual(
            result.status_observation_gap, evidence.STATUS_OBSERVATION_GAP
        )
        self.assertIn(
            "do not expose the 16 per-tile", result.status_observation_gap
        )

    def test_accepts_only_explicit_exact_all_tile_success_records(self) -> None:
        observations = [
            evidence.DirectDTEStatusObservation(
                tile=tile_id,
                status_abi=evidence.DIRECT_DTE_STATUS_ABI,
                value=evidence.DIRECT_DTE_STATUS_SUCCESS,
            )
            for tile_id in range(evidence.TILE_COUNT)
        ]
        result = evidence.validate_direct_dte_board_output(
            valid_stdout(),
            self.manifest,
            completion_timeout_ms=60000,
            status_observations=observations,
        )
        self.assertTrue(result.has_observed_all_tile_success)
        self.assertIsNone(result.status_observation_gap)
        self.assertEqual(
            [row.tile for row in result.observed_status_by_tile],  # type: ignore[union-attr]
            list(range(evidence.TILE_COUNT)),
        )

        partial = observations[:-1]
        duplicate = observations + [copy.copy(observations[-1])]
        wrong_identity = list(observations)
        wrong_identity[5] = dataclasses.replace(
            wrong_identity[5], status_abi="wafer-direct-dte-status-other"
        )
        wrong_abi = list(observations)
        wrong_abi[6] = dataclasses.replace(
            wrong_abi[6], status_abi="unsupported-status-abi"
        )
        transport_error = list(observations)
        transport_error[15] = dataclasses.replace(
            transport_error[15], value=2
        )
        for rows in (
            partial,
            duplicate,
            wrong_identity,
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

    def test_rejects_partial_or_wrong_completion_output(self) -> None:
        missing_completion = valid_stdout().replace(
            "completion: return_after_local_drain tile_id=15\n", ""
        )
        duplicate_tile = valid_stdout().replace(
            "completion: return_after_local_drain tile_id=15",
            "completion: return_after_local_drain tile_id=14",
        )
        wrong_completion = valid_stdout().replace(
            "completion: return_after_local_drain tile_id=7",
            "completion: unsupported tile_id=7",
        )
        missing_stage = valid_stdout().replace(
            "board_stage: completion\n", ""
        )
        for stdout in (
            missing_completion,
            duplicate_tile,
            wrong_completion,
            missing_stage,
        ):
            with self.subTest(stdout=stdout), self.assertRaises(RuntimeError):
                evidence.validate_direct_dte_board_output(
                    stdout, self.manifest, completion_timeout_ms=60000
                )


if __name__ == "__main__":
    unittest.main()
