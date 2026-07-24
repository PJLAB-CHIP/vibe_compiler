#!/usr/bin/env python3
"""Host-only tests for the fail-fast hardware calibration batch runner."""

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
            {
                "name": "RESOURCE_LOCK",
                "value": test.get("resource_lock", ["board-0"]),
            },
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
            }
        )
        + "\\n"
    )

junit = pathlib.Path(sys.argv[sys.argv.index("--output-junit") + 1])
suite = ET.Element(
    "testsuite",
    tests="1",
    failures="0",
    errors="0",
    skipped="0",
)
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
            "first",
            "sample-a",
            "wafer-board-sample-a",
            "first sample",
        ),
        RUNNER.CalibrationStep(
            "second",
            "sample-b",
            "wafer-board-sample-b",
            "second sample",
        ),
        RUNNER.CalibrationStep(
            "third",
            "sample-c",
            "wafer-board-sample-c",
            "third sample",
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

    def test_default_plan_covers_all_board_stages_and_only_repeats_heartbeat(
        self,
    ) -> None:
        RUNNER.validate_default_plan()
        steps = RUNNER.CALIBRATION_STEPS
        self.assertEqual(steps[0].ctest_name, RUNNER.HEARTBEAT_CTEST)
        self.assertEqual(steps[-1].ctest_name, RUNNER.HEARTBEAT_CTEST)
        self.assertEqual(len(steps), 31)
        self.assertEqual(len({step.ctest_name for step in steps}), 30)
        batches = {step.batch for step in steps}
        self.assertTrue(
            {
                "preflight-profile-heartbeat",
                "rank-one-instruction",
                "rank-one-datamove",
                "rank-one-memory",
                "rank-one-ne",
                "rank-one-spm",
                "rank-one-ncc",
                "full-card-runtime",
                "full-card-barrier",
                "full-card-dte",
                "measurement-pmu",
                "terminal-heartbeat",
            }.issubset(batches)
        )
        self.assertFalse(
            any(
                RUNNER.is_forbidden_control_argument(step.ctest_name)
                for step in steps
            )
        )

    def test_execute_runs_serially_and_records_per_step_evidence(self) -> None:
        self.assertEqual(self.execute(), 0)
        records = self.records()
        self.assertEqual(records[0]["kind"], "cmake")
        ctest_records = [
            record for record in records if record["kind"] == "ctest"
        ]
        self.assertEqual(
            [record["name"] for record in ctest_records],
            [step.ctest_name for step in self.steps],
        )
        self.assertTrue(
            all(record["parallel"] == "1" for record in ctest_records)
        )
        summary = json.loads((self.root / "logs" / "session.json").read_text())
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(len(summary["steps"]), len(self.steps))
        self.assertTrue(all(step["status"] == "passed" for step in summary["steps"]))
        self.assertEqual(
            len(list((self.root / "logs").glob("*.log"))),
            len(self.steps) + 1,
        )
        self.assertEqual(
            len(list((self.root / "logs").glob("*.junit.xml"))),
            len(self.steps),
        )

    def test_first_failure_stops_without_retrying_later_steps(self) -> None:
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

    def test_skipped_hardware_test_is_a_failure(self) -> None:
        environment = dict(self.environment)
        environment["FAKE_SKIP_TEST"] = self.steps[0].ctest_name
        self.assertEqual(self.execute(environment=environment), 1)
        summary = json.loads((self.root / "logs" / "session.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertIn("skipped", summary["failure"])
        self.assertEqual(len(summary["steps"]), 1)

    def test_inventory_rejects_no_card_and_unplanned_board_tests(self) -> None:
        registered = {
            "wafer-board-sample-a": RUNNER.RegisteredTest(
                "wafer-board-sample-a",
                ("python3", "probe.py", "--no-card"),
                frozenset(("board", "hardware", "no-card")),
                30.0,
                "board-0",
            ),
            "wafer-board-extra": RUNNER.RegisteredTest(
                "wafer-board-extra",
                ("python3", "extra.py"),
                frozenset(("board", "hardware")),
                30.0,
                "board-0",
            ),
        }
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "no-card path"
        ):
            RUNNER.validate_inventory(
                registered,
                (
                    RUNNER.CalibrationStep(
                        "sample",
                        "sample",
                        "wafer-board-sample-a",
                        "sample",
                    ),
                ),
            )

        registered["wafer-board-sample-a"] = RUNNER.RegisteredTest(
            "wafer-board-sample-a",
            ("python3", "probe.py"),
            frozenset(("board", "hardware")),
            30.0,
            "board-0",
        )
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "explicit batch decision"
        ):
            RUNNER.validate_inventory(
                registered,
                (
                    RUNNER.CalibrationStep(
                        "sample",
                        "sample",
                        "wafer-board-sample-a",
                        "sample",
                    ),
                ),
            )

    def test_execute_mode_requires_both_cli_and_environment_arm(self) -> None:
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
