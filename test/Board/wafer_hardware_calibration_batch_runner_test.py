#!/usr/bin/env python3
"""Host-only contract tests for the current hardware calibration runner."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "tools" / "run_hardware_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "run_hardware_calibration", RUNNER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load calibration runner: {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


FAKE_CMAKE = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

record = pathlib.Path(os.environ["FAKE_RECORD"])
with record.open("a") as output:
    output.write(json.dumps({"kind": "cmake", "argv": sys.argv[1:]}) + "\\n")
print("fake incremental build")
raise SystemExit(int(os.environ.get("FAKE_BUILD_RETURN_CODE", "0")))
"""


FAKE_CTEST = """#!/usr/bin/env python3
import json
import os
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

tests = json.loads(os.environ["FAKE_TESTS"])
record = pathlib.Path(os.environ["FAKE_RECORD"])

if "--show-only=json-v1" in sys.argv:
    inventory = []
    for test in tests:
        properties = [
            {"name": "LABELS", "value": test.get("labels", ["board", "hardware"])},
            {"name": "TIMEOUT", "value": test.get("timeout", 30.0)},
            {"name": "RESOURCE_LOCK", "value": test.get("resource_lock", ["board-0"])},
        ]
        inventory.append(
            {
                "name": test["name"],
                "command": test.get("command", ["python3", "safe_board_probe.py"]),
                "properties": properties,
            }
        )
    print(json.dumps({"kind": "ctestInfo", "tests": inventory}))
    raise SystemExit(0)

regex = sys.argv[sys.argv.index("--tests-regex") + 1]
matches = [test["name"] for test in tests if re.fullmatch(regex, test["name"])]
if len(matches) != 1:
    print(f"selection matched {matches}", file=sys.stderr)
    raise SystemExit(2)
name = matches[0]
with record.open("a") as output:
    output.write(
        json.dumps(
            {
                "kind": "ctest",
                "name": name,
                "parallel": os.environ.get("CTEST_PARALLEL_LEVEL"),
                "calibration_session_id": os.environ.get(
                    "WAFER_CALIBRATION_SESSION_ID"
                ),
            }
        )
        + "\\n"
    )

junit = pathlib.Path(sys.argv[sys.argv.index("--output-junit") + 1])
suite = ET.Element("testsuite", tests="1", failures="0", errors="0", skipped="0")
case = ET.SubElement(suite, "testcase", name=name)
if os.environ.get("FAKE_SKIP_TEST") == name:
    suite.set("skipped", "1")
    ET.SubElement(case, "skipped")
elif os.environ.get("FAKE_FAIL_TEST") == name:
    suite.set("failures", "1")
    ET.SubElement(case, "failure")
ET.ElementTree(suite).write(junit, encoding="unicode")
print(f"fake board test: {name}")
if os.environ.get("FAKE_FAIL_TEST") == name:
    raise SystemExit(1)
"""


def make_executable(path: pathlib.Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def sample_steps() -> tuple[object, ...]:
    return (
        RUNNER.CalibrationStep(
            "first", "sample-a", "wafer-board-sample-a", "first sample"
        ),
        RUNNER.CalibrationStep(
            "second", "sample-b", "wafer-board-sample-b", "second sample"
        ),
        RUNNER.CalibrationStep(
            "third", "sample-c", "wafer-board-sample-c", "third sample"
        ),
    )


class HardwareCalibrationBatchRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.build_dir = self.root / "build"
        self.build_dir.mkdir()
        self.record = self.root / "commands.jsonl"
        self.fake_cmake = self.root / "fake-cmake"
        self.fake_ctest = self.root / "fake-ctest"
        make_executable(self.fake_cmake, FAKE_CMAKE)
        make_executable(self.fake_ctest, FAKE_CTEST)
        self.steps = sample_steps()
        self.environment = dict(os.environ)
        self.environment.update(
            {
                RUNNER.ARM_ENVIRONMENT_VARIABLE: "1",
                "FAKE_RECORD": str(self.record),
                "FAKE_TESTS": json.dumps(
                    [{"name": step.ctest_name} for step in self.steps]
                ),
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def records(self) -> list[dict[str, object]]:
        if not self.record.exists():
            return []
        return [
            json.loads(line)
            for line in self.record.read_text().splitlines()
            if line
        ]

    def execute(self, *, environment: dict[str, str] | None = None) -> int:
        return RUNNER.execute_calibration(
            build_dir=self.build_dir,
            log_dir=self.root / "logs",
            ctest=str(self.fake_ctest),
            cmake=str(self.fake_cmake),
            steps=self.steps,
            environment=self.environment if environment is None else environment,
        )

    def test_default_plan_contains_only_current_registered_board_probes(
        self,
    ) -> None:
        RUNNER.validate_default_plan()
        self.assertEqual(
            [step.key for step in RUNNER.CALIBRATION_STEPS],
            [
                "direct-dte-ncc-execution",
                "complete-tile-barrier",
                "complete-tile-add",
                "complete-tile-add-profile",
                "ddr-tile-offset",
                "worker-placement",
                "spm-cross-tile-conflict",
                "ne-tail-throughput",
                "engine-pipeline-characterization",
                "ddr-active-tile-contention",
            ],
        )
        self.assertEqual(
            RUNNER.CALIBRATION_STEPS[0].ctest_name,
            RUNNER.INITIAL_VALIDATION_CTEST,
        )
        self.assertEqual(RUNNER.EXPLICIT_ONLY_STEPS, ())
        self.assertEqual(RUNNER.SESSION_RISK_STEP_KEYS, ())
        self.assertEqual(
            len({step.ctest_name for step in RUNNER.CALIBRATION_STEPS}), 10
        )
        self.assertFalse(
            any(
                RUNNER.is_forbidden_control_argument(step.ctest_name)
                for step in RUNNER.CALIBRATION_STEPS
            )
        )

    def test_selection_adds_initial_validation_once_in_canonical_order(
        self,
    ) -> None:
        selected = RUNNER.select_calibration_steps(
            ("worker-placement", "complete-tile-barrier")
        )
        self.assertEqual(
            [step.key for step in selected],
            [
                "direct-dte-ncc-execution",
                "complete-tile-barrier",
                "worker-placement",
            ],
        )
        self.assertEqual(
            RUNNER.select_calibration_steps(None, ("current-interface-smoke",)),
            RUNNER.CALIBRATION_STEPS,
        )
        self.assertEqual(
            RUNNER.select_calibration_steps(None, ("transport",)),
            RUNNER.CALIBRATION_STEPS[:1],
        )
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "initial validation is automatic"
        ):
            RUNNER.select_calibration_steps(("direct-dte-ncc-execution",))
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "unknown calibration step"
        ):
            RUNNER.select_calibration_steps(("missing-step",))
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "batch names"
        ):
            RUNNER.select_calibration_steps(None, ("missing-batch",))

    def test_execute_runs_serially_and_records_fresh_session(self) -> None:
        self.environment[RUNNER.SESSION_ENVIRONMENT_VARIABLE] = "stale"
        self.assertEqual(self.execute(), 0)
        records = self.records()
        self.assertEqual(records[0]["kind"], "cmake")
        self.assertEqual(
            records[0]["argv"][-2:],
            ["--parallel", str(max(1, os.cpu_count() or 1))],
        )
        ctest_records = [record for record in records if record["kind"] == "ctest"]
        self.assertEqual(
            [record["name"] for record in ctest_records],
            [step.ctest_name for step in self.steps],
        )
        self.assertTrue(all(record["parallel"] == "1" for record in ctest_records))
        session_ids = {
            record["calibration_session_id"] for record in ctest_records
        }
        self.assertEqual(len(session_ids), 1)
        session_id = session_ids.pop()
        self.assertRegex(session_id, r"^[0-9a-f]{32}$")
        self.assertNotEqual(session_id, "stale")
        summary = json.loads((self.root / "logs" / "session.json").read_text())
        self.assertEqual(summary["calibration_session_id"], session_id)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(all(step["status"] == "passed" for step in summary["steps"]))

    def test_first_failure_stops_without_retry_or_reset(self) -> None:
        environment = dict(self.environment)
        environment["FAKE_FAIL_TEST"] = self.steps[1].ctest_name
        self.assertEqual(self.execute(environment=environment), 1)
        ctest_records = [
            record for record in self.records() if record["kind"] == "ctest"
        ]
        self.assertEqual(
            [record["name"] for record in ctest_records],
            [self.steps[0].ctest_name, self.steps[1].ctest_name],
        )
        summary = json.loads((self.root / "logs" / "session.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertIn("left unexecuted", summary["failure"])
        self.assertFalse(summary["automatic_retry"])
        self.assertFalse(summary["reset_or_power_control"])

    def test_skipped_hardware_test_is_failure(self) -> None:
        environment = dict(self.environment)
        environment["FAKE_SKIP_TEST"] = self.steps[0].ctest_name
        self.assertEqual(self.execute(environment=environment), 1)
        summary = json.loads((self.root / "logs" / "session.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertIn("skipped", summary["failure"])
        self.assertEqual(len(summary["steps"]), 1)

    def test_inventory_rejects_no_card_and_unplanned_board_tests(self) -> None:
        planned = RUNNER.CalibrationStep(
            "sample", "sample", "wafer-board-sample", "sample"
        )
        registered = {
            planned.ctest_name: RUNNER.RegisteredTest(
                planned.ctest_name,
                ("python3", "probe.py", "--no-card"),
                frozenset(("board", "hardware", "no-card")),
                30.0,
                "board-0",
            )
        }
        with self.assertRaisesRegex(RUNNER.CalibrationRunnerError, "no-card path"):
            RUNNER.validate_inventory(registered, (planned,))

        registered[planned.ctest_name] = RUNNER.RegisteredTest(
            planned.ctest_name,
            ("python3", "probe.py"),
            frozenset(("board", "hardware")),
            30.0,
            "board-0",
        )
        registered["wafer-board-extra"] = RUNNER.RegisteredTest(
            "wafer-board-extra",
            ("python3", "extra.py"),
            frozenset(("board", "hardware")),
            30.0,
            "board-0",
        )
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "explicit batch decision"
        ):
            RUNNER.validate_inventory(registered, (planned,))

    def test_archives_current_source_and_runtime_evidence(self) -> None:
        work_dir = self.root / "board-work"
        (work_dir / "raw").mkdir(parents=True)
        (work_dir / "raw" / "request.raw").write_bytes(b"\x01\x02")
        (work_dir / "result.json").write_text('{"status":"ok"}\n')
        (work_dir / "source.mlir").write_text("module {}\n")
        (work_dir / "module.so").write_bytes(b"ELF")
        (work_dir / "ignored.log").write_text("not retained\n")
        compiler = self.root / "wafer-compile"
        runtime = self.root / "wafer-run"
        compiler.write_bytes(b"compiler")
        runtime.write_bytes(b"runtime")
        test = RUNNER.RegisteredTest(
            "wafer-board-sample",
            (
                "python3",
                "probe.py",
                "--wafer-compile",
                str(compiler),
                "--wafer-run",
                str(runtime),
                "--work-dir",
                str(work_dir),
            ),
            frozenset(("board", "hardware", "current-interface", "pending")),
            30.0,
            "board-0",
        )
        destination = self.root / "evidence"
        archived = RUNNER.collect_step_evidence(test, destination)
        self.assertIsNotNone(archived)
        assert archived is not None
        self.assertEqual(archived["file_count"], 4)
        manifest = json.loads((destination / "manifest.json").read_text())
        self.assertEqual(
            {item["path"] for item in manifest["files"]},
            {"raw/request.raw", "result.json", "source.mlir", "module.so"},
        )
        self.assertEqual(
            {item["option"] for item in manifest["tools"]},
            {"--wafer-compile", "--wafer-run"},
        )
        self.assertFalse((destination / "ignored.log").exists())

    def test_execute_mode_requires_environment_arm(self) -> None:
        result = RUNNER.main(
            [
                "--execute",
                "--build-dir",
                str(self.build_dir),
                "--ctest",
                str(self.fake_ctest),
                "--cmake",
                str(self.fake_cmake),
            ],
            environment={},
        )
        self.assertEqual(result, 2)
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
