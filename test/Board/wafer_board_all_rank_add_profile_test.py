#!/usr/bin/env python3
"""Host-only contract tests for the live-board Add profiler gate."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


BOARD_DIR = pathlib.Path(__file__).resolve().parent
HARNESS_PATH = BOARD_DIR / "wafer_board_all_rank_add_test.py"
sys.path.insert(0, str(BOARD_DIR))
SPEC = importlib.util.spec_from_file_location(
    "wafer_board_all_rank_add_harness", HARNESS_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Add profiler harness: {HARNESS_PATH}")
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


def engine_rows() -> list[dict[str, object]]:
    rows = []
    for engine in ("CT", "NE", "RDMA", "WDMA", "TDMA"):
        rows.append(
            {
                "engine": engine,
                "engine_execution_time_ns": 8 if engine == "CT" else 0,
                "engine_execution_time_valid": True,
                "activity_window_count": 1 if engine == "CT" else 0,
                "wait_window_cpu_cycles": None,
                "raw_pmu_activity": 8 if engine == "CT" else 0,
            }
        )
    rows.append(
        {
            "engine": "DIRECT_DTE",
            "engine_execution_time_ns": None,
            "wait_window_cpu_cycles": 0,
            "wait_window_count": 0,
            "wait_windows_valid": True,
            "raw_pmu_activity": 0,
        }
    )
    return rows


class ProfileReportFixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.package = root / "package"
        self.run_id = "run-fresh-host-fixture"
        self.runs = pathlib.Path(f"{self.package}.profile") / "runs"
        self.run_directory = self.runs / self.run_id
        self.run_directory.mkdir(parents=True)
        (self.runs / "current").symlink_to(self.run_directory)
        evidence = {
            "schema": "wafer.profile.evidence",
            "schema_version": 8,
            "run_id": self.run_id,
            "measurement": {
                "samples": [
                    {
                        "sample_id": "primary",
                        "sample_index": 0,
                        "device_elapsed_ns": 512,
                        "device_timer_kind": "tx-stream-events",
                        "host_submit_ns": 128,
                        "host_launch_to_completion_ns": 1024,
                        "completion_observation_resolution_ns": 8,
                    }
                ]
            },
            "experiment": {
                "trace": {
                    "complete": True,
                    "tiles": [
                        {"tile": tile}
                        for tile in range(HARNESS.RANK_COUNT)
                    ],
                }
            },
        }
        analysis = {
            "schema": "wafer.profile.analysis",
            "schema_version": 7,
            "run_id": self.run_id,
            "validity": {"trace": True, "pmu": True},
            "program": {
                "duration": {
                    "sample_id": "primary",
                    "sample_index": 0,
                    "qualified": True,
                    "device_elapsed_ns": 512,
                    "device_timer_kind": "tx-stream-events",
                    "host_submit_ns": 128,
                    "host_launch_to_completion_ns": 1024,
                    "completion_observation_resolution_ns": 8,
                },
                "output": {
                    "primary_output_validated": True,
                    "diagnostic_captures_match_primary": True,
                },
                "tiles": [
                    {
                        "tile": tile,
                        "trace_entry_cpu_cycles": 12,
                        "engines": engine_rows(),
                    }
                    for tile in range(HARNESS.RANK_COUNT)
                ],
                "timeline_events": [
                    {
                        "tile": tile,
                        "engine": "CT",
                        "activity_window_cpu_cycles": (
                            None if tile == 0 else 8
                        ),
                        "trace_entry_offset_begin_cpu_cycles": 1,
                        "trace_entry_offset_end_cpu_cycles": 9,
                    }
                    for tile in range(HARNESS.RANK_COUNT)
                ],
            },
        }
        (self.run_directory / "evidence.json").write_text(
            json.dumps(evidence)
        )
        (self.run_directory / "analysis.json").write_text(
            json.dumps(analysis)
        )
        (self.run_directory / "index.html").write_text(
            "<!doctype html><title>Final program profile</title>"
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


class AllRankAddProfileGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_complete_nonnegative_report(self) -> None:
        fixture = ProfileReportFixture(self.root)

        result = HARNESS.verify_profile_report(
            fixture.package, fixture.stdout
        )

        self.assertEqual(
            result,
            (
                512,
                fixture.runs / "current" / "index.html",
            ),
        )

    def test_rejects_bad_permissions_and_negative_measurements(self) -> None:
        fixture = ProfileReportFixture(self.root)
        html = fixture.run_directory / "index.html"
        html.chmod(0o755)
        with self.assertRaisesRegex(
            RuntimeError, "permission is not 0777"
        ):
            HARNESS.verify_profile_report(fixture.package, fixture.stdout)

        html.chmod(0o777)
        analysis_path = fixture.run_directory / "analysis.json"
        analysis = json.loads(analysis_path.read_text())
        analysis["program"]["tiles"][0]["engines"][0][
            "engine_execution_time_ns"
        ] = -1
        analysis_path.write_text(json.dumps(analysis))
        analysis_path.chmod(0o777)
        with self.assertRaisesRegex(
            RuntimeError, "negative engine_execution_time_ns"
        ):
            HARNESS.verify_profile_report(fixture.package, fixture.stdout)

    def test_profile_board_path_has_one_bounded_collection(self) -> None:
        work_dir = self.root / "work"
        arguments = argparse.Namespace(
            wafer_compile=self.root / "wafer-compile",
            wafer_run=self.root / "wafer-run",
            work_dir=work_dir,
            no_card=False,
            profile=True,
            launch_kind="kernel",
            device_id=0,
            expected_runtime_version=1300,
            expected_device_name="/dev/accel/dev-0",
            expected_pci_bus_id="0000:3b:00.0",
            expected_tile_count=HARNESS.RANK_COUNT,
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
            write_source = stack.enter_context(
                mock.patch.object(
                    HARNESS,
                    "write_source_program",
                    return_value=work_dir / "source-program",
                )
            )
            compile_package = stack.enter_context(
                mock.patch.object(HARNESS, "compile_package")
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS, "require_byte_identical_packages"
                )
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS, "require_profile_instrumentation_permissions"
                )
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS, "load_boundary_slices", return_value={}
                )
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS, "validate_manifest", return_value={}
                )
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS,
                    "write_rank_payloads",
                    return_value=(
                        [],
                        {2},
                        {(0, 0, 0)},
                        {(0, 0)},
                    ),
                )
            )
            verify_board = stack.enter_context(
                mock.patch.object(HARNESS, "verify_board_evidence")
            )
            stack.enter_context(
                mock.patch.object(
                    HARNESS,
                    "verify_profile_report",
                    return_value=(
                        1024,
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
        write_source.assert_called_once_with(work_dir)
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
        run.assert_called_once()
        collection = run.call_args
        self.assertEqual(
            collection.args[0][1:3],
            ["--package-dir", str(work_dir / "package")],
        )
        self.assertEqual(
            collection.kwargs["timeout_seconds"],
            max(
                300.0,
                (
                    HARNESS.PROFILE_MEASUREMENT_COUNT * 60.0
                    + HARNESS.BOARD_PROCESS_TIMEOUT_MARGIN_SECONDS
                ),
            ),
        )
        verify_board.assert_called_once()
        self.assertIn(
            "board_profile_collection: pass launches=3 primary=1",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
