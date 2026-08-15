#!/usr/bin/env python3
"""Host-only contract tests for the Direct-DTE profiler board gate."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


BOARD_DIR = pathlib.Path(__file__).resolve().parent
HARNESS_PATH = BOARD_DIR / "wafer_board_direct_dte_collective_test.py"
sys.path.insert(0, str(BOARD_DIR))
SPEC = importlib.util.spec_from_file_location(
    "wafer_board_direct_dte_profile_harness", HARNESS_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Direct-DTE harness: {HARNESS_PATH}")
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class ProfileReportFixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.package = root / "package"
        self.run_id = "run-direct-dte-host-fixture"
        self.runs = pathlib.Path(f"{self.package}.profile") / "runs"
        self.run_directory = self.runs / self.run_id
        self.run_directory.mkdir(parents=True)
        (self.runs / "current").symlink_to(self.run_directory)

        trace_tiles = []
        analysis_tiles = []
        for tile in range(HARNESS.TILE_COUNT):
            event = {
                "engine": "DIRECT_DTE",
                "kind": "direct-dte-wait",
                "operation_span_valid": True,
                "operation_begin_cycle": 110,
                "operation_end_cycle": 118,
            }
            trace_tiles.append({"tile": tile, "events": [event]})
            analysis_tiles.append(
                {
                    "tile": tile,
                    "trace_entry_cpu_cycles": 20,
                    "semantic_partition": {
                        "entry_cycles": 20,
                        "exclusive_cycles": 20,
                        "exclusive_accounting_valid": True,
                    },
                    "semantic_timeline_segments": [
                        {"category": "dte-completion-wait", "cycles": 20}
                    ],
                    "trace_overhead_overlay": {
                        "exclusive_component_cycles": 2,
                        "inside_entry_known_overhead_cycles": 2,
                        "outside_entry_overhead_cycles": 0,
                        "components_exclusive": True,
                        "non_additive_to_semantic_partition": True,
                        "positioning": "aggregate-only",
                        "coverage": "measured-categories-only",
                        "rows": [
                            {
                                "category": "trace-run-cost-overlay",
                                "cycles": 2,
                            }
                        ]
                    },
                    "engines": [
                        {
                            "engine": "DIRECT_DTE",
                            "wait_windows_valid": True,
                            "wait_window_cpu_cycles": 8,
                            "wait_window_count": 1,
                        }
                    ],
                }
            )

        sample = {
            "sample_id": "primary",
            "sample_index": 0,
            "device_elapsed_ns": 8400,
            "device_timer_kind": "tx-stream-events",
            "host_submit_ns": 2600,
            "host_launch_to_completion_ns": 18000,
            "completion_observation_resolution_ns": 20,
        }
        evidence = {
            "schema": "wafer.profile.evidence",
            "run_id": self.run_id,
            "measurement": {"samples": [sample]},
            "experiment": {
                "trace": {"complete": True, "tiles": trace_tiles}
            },
        }
        analysis = {
            "schema": "wafer.profile.analysis",
            "run_id": self.run_id,
            "valid": True,
            "validity": {"trace": True, "cost_accounting": True},
            "program": {
                "duration": {
                    **sample,
                    "host_non_submit_envelope_ns": 15400,
                    "host_envelope_available": True,
                    "completion_observation_fraction": 20 / 18000,
                    "host_completion_high_resolution": True,
                    "qualified": True,
                },
                "output": {
                    "primary_output_validated": True,
                    "diagnostic_captures_match_primary": True,
                },
                "tiles": analysis_tiles,
                "timeline_events": [
                    {
                        "tile": tile,
                        "engine": "DIRECT_DTE",
                        "kind": "direct-dte-wait",
                        "operation_window_cpu_cycles": 8,
                        "duration_status": "Measured",
                    }
                    for tile in range(HARNESS.TILE_COUNT)
                ],
            },
        }
        (self.run_directory / "evidence.json").write_text(json.dumps(evidence))
        (self.run_directory / "analysis.json").write_text(json.dumps(analysis))
        (self.run_directory / "index.html").write_text(
            "<!doctype html><title>Direct-DTE profile</title>"
        )
        for path in (
            pathlib.Path(f"{self.package}.profile"),
            self.runs,
            self.run_directory,
            *self.run_directory.iterdir(),
        ):
            path.chmod(0o777)

    @property
    def stdout(self) -> str:
        return f"profile_report: {self.runs / 'current' / 'index.html'}\n"


class DirectDTEProfileGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_real_direct_dte_phase_and_nonnegative_cost(self) -> None:
        fixture = ProfileReportFixture(self.root)

        self.assertEqual(
            HARNESS.verify_profile_report(fixture.package, fixture.stdout),
            (8400, fixture.runs / "current" / "index.html"),
        )

    def test_accepts_zero_quantized_host_completion_diagnostics(self) -> None:
        fixture = ProfileReportFixture(self.root)
        evidence_path = fixture.run_directory / "evidence.json"
        analysis_path = fixture.run_directory / "analysis.json"
        evidence = json.loads(evidence_path.read_text())
        analysis = json.loads(analysis_path.read_text())
        sample = evidence["measurement"]["samples"][0]
        sample["host_submit_ns"] = 0
        sample["host_launch_to_completion_ns"] = 0
        sample["completion_observation_resolution_ns"] = 0
        duration = analysis["program"]["duration"]
        duration.update(sample)
        duration["host_non_submit_envelope_ns"] = 0
        duration["host_envelope_available"] = False
        duration["completion_observation_fraction"] = None
        duration["host_completion_high_resolution"] = False
        evidence_path.write_text(json.dumps(evidence))
        analysis_path.write_text(json.dumps(analysis))

        self.assertEqual(
            HARNESS.verify_profile_report(fixture.package, fixture.stdout),
            (8400, fixture.runs / "current" / "index.html"),
        )

    def test_rejects_missing_tile_direct_dte_phase(self) -> None:
        fixture = ProfileReportFixture(self.root)
        analysis_path = fixture.run_directory / "analysis.json"
        analysis = json.loads(analysis_path.read_text())
        analysis["program"]["timeline_events"] = [
            event
            for event in analysis["program"]["timeline_events"]
            if event["tile"] != HARNESS.TILE_COUNT - 1
        ]
        analysis_path.write_text(json.dumps(analysis))

        with self.assertRaisesRegex(
            RuntimeError,
            "measured Direct-DTE phase for one or more tiles",
        ):
            HARNESS.verify_profile_report(fixture.package, fixture.stdout)

    def test_profile_board_path_has_exactly_one_board_runner(self) -> None:
        work_dir = self.root / "work"
        arguments = argparse.Namespace(
            wafer_compile=self.root / "wafer-compile",
            wafer_run=self.root / "wafer-run",
            work_dir=work_dir,
            no_card=False,
            profile=True,
            device_id=0,
            expected_runtime_version=1300,
            expected_device_name="/dev/accel/dev-0",
            expected_pci_bus_id="0000:3b:00.0",
            expected_tile_count=HARNESS.TILE_COUNT,
            expected_runtime_library_sha256="0" * 64,
            completion_timeout_ms=60000,
            repeat=1,
        )
        completed = subprocess.CompletedProcess(
            ["wafer-run"], 0, stdout="", stderr=""
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    {"WAFER_EXECUTE_HARDWARE_TESTS": "1"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS, "parse_args", return_value=arguments
                )
            )
            write_fixture = stack.enter_context(
                mock.patch.object(
                    HARNESS,
                    "write_fixture",
                    return_value=work_dir / "source-program",
                )
            )
            compile_package = stack.enter_context(
                mock.patch.object(HARNESS, "compile_package")
            )
            compare = stack.enter_context(
                mock.patch.object(
                    HARNESS, "require_byte_identical_packages"
                )
            )
            permissions = stack.enter_context(
                mock.patch.object(
                    HARNESS, "require_profile_instrumentation_permissions"
                )
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS, "validate_manifest", return_value={}
                )
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS, "write_raw_files", return_value=[]
                )
            )
            verify_board = stack.enter_context(
                mock.patch.object(HARNESS, "verify_board_evidence")
            )
            verify_report = stack.enter_context(
                mock.patch.object(
                    HARNESS,
                    "verify_profile_report",
                    return_value=(
                        8400,
                        work_dir / "package.profile/runs/current/index.html",
                    ),
                )
            )
            run = stack.enter_context(
                mock.patch.object(
                    HARNESS, "run", return_value=completed
                )
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = HARNESS.main()

        self.assertEqual(result, 0)
        write_fixture.assert_called_once_with(work_dir)
        self.assertEqual(
            compile_package.call_args_list,
            [
                mock.call(
                    arguments,
                    work_dir / "source-program",
                    work_dir / "ordinary-package",
                    profile=False,
                ),
                mock.call(
                    arguments,
                    work_dir / "source-program",
                    work_dir / "package",
                    profile=True,
                ),
            ],
        )
        compare.assert_called_once_with(
            work_dir / "ordinary-package", work_dir / "package"
        )
        permissions.assert_called_once_with(work_dir / "package")
        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0][1:3],
            ["--package-dir", str(work_dir / "package")],
        )
        self.assertEqual(
            run.call_args.kwargs["timeout_seconds"],
            max(
                300.0,
                (
                    HARNESS.PROFILE_MEASUREMENT_COUNT * 60.0
                    + HARNESS.DIRECT_DTE_PROCESS_TIMEOUT_MARGIN_SECONDS
                ),
            ),
        )
        verify_board.assert_called_once()
        verify_report.assert_called_once()
        self.assertIn(
            "direct_dte_profile_collection: pass launches=3 primary=1",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
