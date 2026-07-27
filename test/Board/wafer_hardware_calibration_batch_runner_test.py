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
                "calibration_session_id": os.environ.get(
                    "WAFER_CALIBRATION_SESSION_ID"
                ),
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

    def test_default_plan_covers_all_board_stages_and_qualifies_once(
        self,
    ) -> None:
        RUNNER.validate_default_plan()
        steps = RUNNER.CALIBRATION_STEPS
        self.assertEqual(steps[0].ctest_name, RUNNER.HEARTBEAT_CTEST)
        self.assertEqual(
            sum(step.ctest_name == RUNNER.HEARTBEAT_CTEST for step in steps),
            1,
        )
        self.assertEqual(len(steps), 35)
        self.assertEqual(len({step.ctest_name for step in steps}), 35)
        self.assertEqual(
            len(RUNNER.ALL_CALIBRATION_STEPS),
            len(RUNNER.CALIBRATION_STEPS)
            + len(RUNNER.EXPLICIT_ONLY_STEPS),
        )
        default_keys = {step.key for step in steps}
        self.assertTrue(
            {
                "instruction-family-ct-capability",
                "instruction-family-regression",
                "memory-engine-pair-new-offsets",
                "spm-non-preferred-geometry",
            }.issubset(default_keys)
        )
        explicit_keys = {step.key for step in RUNNER.EXPLICIT_ONLY_STEPS}
        self.assertTrue(
            {
                "compiler-optimization-reciprocal-implementation",
                "compiler-optimization-tree-all-reduce",
                "collective-characterization-all-gather-direct-vs-ring-256b",
                "collective-characterization-all-reduce-ring-vs-tree-65536b",
                "ncc-queue-saturation-ct-short-depth5-tight-window",
                "ncc-worker-wait-scope-ne-worker0-default-tight-window",
                "ncc-worker-subset-rdma-target2-include-tight-window",
                "worker-placement-worker-placement-ct",
                "spm-conflict-equivalence-rank-one",
                "ddr-conflict-equivalence-cross-tile",
            }.issubset(explicit_keys),
        )
        self.assertFalse(
            any("native-concat-hw" in key for key in explicit_keys)
        )
        self.assertNotIn(
            "wafer-board-model-add",
            {step.ctest_name for step in RUNNER.ALL_CALIBRATION_STEPS},
        )
        for step in RUNNER.EXPLICIT_ONLY_STEPS:
            self.assertNotIn(step, RUNNER.CALIBRATION_STEPS)
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
            }.issubset(batches)
        )
        self.assertFalse(
            any(
                RUNNER.is_forbidden_control_argument(step.ctest_name)
                for step in steps
            )
        )

    def test_explicit_step_selection_keeps_canonical_order_and_one_heartbeat(
        self,
    ) -> None:
        selected = RUNNER.select_calibration_steps(
            (
                "instruction-family-ct-capability",
                "memory-engine-pair-new-offsets",
                "spm-non-preferred-geometry",
            )
        )
        self.assertEqual(
            [step.key for step in selected],
            [
                "initial-profile-heartbeat",
                "instruction-family-ct-capability",
                "memory-engine-pair-new-offsets",
                "spm-non-preferred-geometry",
            ],
        )
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "more than once"
        ):
            RUNNER.select_calibration_steps(("ct-vector", "ct-vector"))
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "heartbeat steps are automatic"
        ):
            RUNNER.select_calibration_steps(("initial-profile-heartbeat",))
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "unknown calibration step"
        ):
            RUNNER.select_calibration_steps(("missing-step",))

    def test_named_compiler_optimization_batch_is_complete_and_ordered(
        self,
    ) -> None:
        selected = RUNNER.select_calibration_steps(
            None, ("compiler-optimization-paired",)
        )
        self.assertEqual(
            [step.key for step in selected],
            [
                "initial-profile-heartbeat",
                *RUNNER.SELECTABLE_BATCHES["compiler-optimization-paired"],
            ],
        )
        self.assertEqual(
            len(RUNNER.SELECTABLE_BATCHES["compiler-optimization-paired"]),
            8,
        )
        campaign = RUNNER.select_calibration_steps(
            None, ("compiler-optimization-campaign",)
        )
        self.assertEqual(
            [step.key for step in campaign],
            [
                "initial-profile-heartbeat",
                "direct-dte-collective",
                *RUNNER.SELECTABLE_BATCHES["compiler-optimization-paired"],
            ],
        )
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "batch names"
        ):
            RUNNER.select_calibration_steps(None, ("missing-batch",))
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "batches were selected more than once"
        ):
            RUNNER.select_calibration_steps(
                None,
                (
                    "compiler-optimization-paired",
                    "compiler-optimization-paired",
                ),
            )
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "steps were selected more than once"
        ):
            RUNNER.select_calibration_steps(
                ("compiler-optimization-tree-all-reduce",),
                ("compiler-optimization-paired",),
            )

    def test_named_collective_characterization_batch_is_complete_and_ordered(
        self,
    ) -> None:
        batch = RUNNER.SELECTABLE_BATCHES["collective-characterization"]
        self.assertEqual(
            batch,
            tuple(
                f"collective-characterization-{case_name}"
                for case_name in RUNNER.COLLECTIVE_CHARACTERIZATION_CASES
            ),
        )
        self.assertEqual(len(batch), 9)
        selected = RUNNER.select_calibration_steps(
            None, ("collective-characterization",)
        )
        self.assertEqual(
            [step.key for step in selected],
            [
                "initial-profile-heartbeat",
                *batch,
            ],
        )
        self.assertTrue(
            all(
                step.batch == "full-card-collective-characterization"
                for step in selected[1:]
            )
        )
        self.assertFalse(
            set(batch) & {step.key for step in RUNNER.CALIBRATION_STEPS}
        )
        with self.assertRaisesRegex(
            RUNNER.CalibrationRunnerError, "steps were selected more than once"
        ):
            RUNNER.select_calibration_steps(
                (batch[0],), ("collective-characterization",)
            )

    def test_pending_execution_boundary_batch_is_complete_and_explicit(
        self,
    ) -> None:
        batch = RUNNER.SELECTABLE_BATCHES["pending-execution-boundaries"]
        self.assertEqual(
            len(RUNNER.QUEUE_SATURATION_CASE_NAMES),
            30,
        )
        self.assertEqual(
            len(RUNNER.WORKER_WAIT_SCOPE_CASE_NAMES),
            18,
        )
        self.assertEqual(
            len(RUNNER.WORKER_SUBSET_SCOPE_CASE_NAMES),
            12,
        )
        self.assertEqual(
            len(RUNNER.ARGMIN_PENDING_DOMAIN_CASE_NAMES),
            2,
        )
        self.assertEqual(
            len(RUNNER.UNPOOL_PENDING_COLLISION_CASE_NAMES),
            2,
        )
        self.assertEqual(
            len(RUNNER.COLLECTIVE_TRAFFIC_BEHAVIOR_CASES),
            11,
        )
        self.assertEqual(
            len(RUNNER.DTE_PENDING_CASE_SPECS),
            25,
        )
        self.assertEqual(
            len(RUNNER.DTE_PENDING_CASE_NAMES),
            25,
        )
        self.assertEqual(
            len(RUNNER.ENGINE_PIPELINE_BOARD_CELL_KEYS),
            381,
        )
        self.assertEqual(
            len(RUNNER.ENGINE_PIPELINE_BOARD_GROUP_KEYS),
            61,
        )
        self.assertEqual(
            RUNNER.ENGINE_PIPELINE_EXTERNAL_GROUP_KEYS,
            ("single-ne-tail",),
        )
        self.assertEqual(
            len(RUNNER.SPM_SUSTAINED_BOARD_GROUP_KEYS),
            4,
        )
        self.assertEqual(
            len(RUNNER.DDR_ACTIVE_RANK_CASE_KEYS),
            45,
        )
        self.assertEqual(
            len(RUNNER.DDR_ACTIVE_RANK_BOARD_GROUP_KEYS),
            9,
        )
        self.assertEqual(
            len(RUNNER.WORKER_PLACEMENT_CASE_KEYS),
            44,
        )
        self.assertEqual(
            len(RUNNER.WORKER_PLACEMENT_BOARD_GROUP_KEYS),
            14,
        )
        self.assertEqual(
            len(batch),
            (
                30
                + 18
                + 12
                + 2
                + 2
                + 4
                + 11
                + 25
                + 61
                + 1
                + 4
                + 9
                + 14
            ),
        )
        self.assertEqual(len(set(batch)), len(batch))
        selected = RUNNER.select_calibration_steps(
            None, ("pending-execution-boundaries",)
        )
        self.assertEqual(
            [step.key for step in selected],
            [
                "initial-profile-heartbeat",
                *batch,
            ],
        )
        self.assertFalse(
            set(batch) & {step.key for step in RUNNER.CALIBRATION_STEPS}
        )

    def test_direct_dte_raw_batch_is_complete_and_isolated(self) -> None:
        batch = RUNNER.SELECTABLE_BATCHES["direct-dte-raw-behavior"]
        self.assertEqual(
            batch,
            tuple(
                f"direct-dte-pending-{case_name}"
                for case_name in RUNNER.DTE_PENDING_CASE_NAMES
            ),
        )
        self.assertEqual(len(batch), 25)
        self.assertEqual(len(set(batch)), len(batch))
        selected = RUNNER.select_calibration_steps(
            None, ("direct-dte-raw-behavior",)
        )
        self.assertEqual(
            [step.key for step in selected],
            [
                "initial-profile-heartbeat",
                *batch,
            ],
        )
        self.assertTrue(
            all(
                step.batch == "full-card-pending-direct-dte-raw"
                for step in selected[1:]
            )
        )
        self.assertEqual(
            [step.ctest_name for step in selected[1:]],
            [
                f"wafer-board-{case_name}"
                for case_name in RUNNER.DTE_PENDING_CASE_NAMES
            ],
        )

    def test_pending_hardware_calibration_batch_exactly_covers_inventory(
        self,
    ) -> None:
        batch = RUNNER.SELECTABLE_BATCHES["pending-hardware-calibration"]
        explicit_by_key = {
            step.key: step for step in RUNNER.EXPLICIT_ONLY_STEPS
        }
        expected_ctests = {
            ctest
            for family in RUNNER.pending_inventory.FAMILIES
            if (
                family.disposition
                == RUNNER.pending_inventory.PENDING_BOARD
            )
            for ctest in family.board_ctests
        }
        selected_ctests = tuple(
            explicit_by_key[key].ctest_name for key in batch
        )
        self.assertEqual(len(selected_ctests), len(expected_ctests))
        self.assertEqual(set(selected_ctests), expected_ctests)
        self.assertEqual(len(set(batch)), len(batch))
        self.assertFalse(any("native-concat-hw" in key for key in batch))
        selected = RUNNER.select_calibration_steps(
            None, ("pending-hardware-calibration",)
        )
        self.assertEqual(
            [step.key for step in selected],
            [
                "initial-profile-heartbeat",
                *batch,
            ],
        )

    def test_execute_runs_serially_and_records_per_step_evidence(self) -> None:
        self.environment[RUNNER.SESSION_ENVIRONMENT_VARIABLE] = "stale"
        self.assertEqual(self.execute(), 0)
        records = self.records()
        self.assertEqual(records[0]["kind"], "cmake")
        self.assertEqual(records[0]["argv"][-2:], ["--parallel", "128"])
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
        session_ids = {
            record["calibration_session_id"] for record in ctest_records
        }
        self.assertEqual(len(session_ids), 1)
        session_id = session_ids.pop()
        self.assertRegex(session_id, r"^[0-9a-f]{32}$")
        self.assertNotEqual(session_id, "stale")
        self.assertEqual(summary["calibration_session_id"], session_id)
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

    def test_each_execute_uses_a_fresh_calibration_session_id(self) -> None:
        self.assertEqual(self.execute(), 0)
        first = json.loads(
            (self.root / "logs" / "session.json").read_text()
        )["calibration_session_id"]
        self.assertEqual(
            RUNNER.execute_calibration(
                build_dir=self.build_dir,
                log_dir=self.root / "logs-second",
                ctest=str(self.fake_ctest),
                cmake=str(self.fake_cmake),
                steps=self.steps,
                environment=self.environment,
            ),
            0,
        )
        second = json.loads(
            (self.root / "logs-second" / "session.json").read_text()
        )["calibration_session_id"]
        self.assertRegex(first, r"^[0-9a-f]{32}$")
        self.assertRegex(second, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first, second)
        ctest_records = [
            record
            for record in self.records()
            if record["kind"] == "ctest"
        ]
        self.assertEqual(
            [
                {
                    record["calibration_session_id"]
                    for record in ctest_records[: len(self.steps)]
                },
                {
                    record["calibration_session_id"]
                    for record in ctest_records[len(self.steps) :]
                },
            ],
            [{first}, {second}],
        )

    def test_collective_traffic_builds_its_test_compiler(self) -> None:
        self.steps = (
            RUNNER.CalibrationStep(
                "collective-traffic-behavior-alltoall-256b",
                "full-card-pending-collective-traffic",
                "wafer-board-collective-traffic-behavior-alltoall-256b",
                "structured traffic carrier",
            ),
        )
        self.environment["FAKE_TESTS"] = json.dumps(
            [{"name": self.steps[0].ctest_name}]
        )
        self.assertEqual(self.execute(), 0)
        build = self.records()[0]
        self.assertEqual(build["kind"], "cmake")
        self.assertIn("wafer-compile-test", build["argv"])

    def test_archives_raw_and_json_evidence_with_hash_manifest(self) -> None:
        work_dir = self.root / "board-work"
        (work_dir / "raw").mkdir(parents=True)
        (work_dir / "raw" / "case.request.raw").write_bytes(b"\x01\x02")
        (work_dir / "result.json").write_text('{"status":"ok"}\n')
        (work_dir / "source.mlir").write_text("module {}\n")
        (work_dir / "module.so").write_bytes(b"ELF")
        (work_dir / "ignored.log").write_text("not durable evidence\n")
        tools = {}
        for name in (
            "wafer-compile",
            "wafer-compile-test",
            "wafer-run",
            "tx8-objdump",
            "llvm-clangxx",
        ):
            path = self.root / name
            path.write_bytes(name.encode())
            tools[name] = path
        test = RUNNER.RegisteredTest(
            "wafer-board-compiler-optimization-artifact-sample",
            (
                "python3",
                "probe.py",
                "--wafer-compile",
                str(tools["wafer-compile"]),
                "--wafer-compile-test",
                str(tools["wafer-compile-test"]),
                "--wafer-run",
                str(tools["wafer-run"]),
                "--tx8-objdump",
                str(tools["tx8-objdump"]),
                "--llvm-clangxx",
                str(tools["llvm-clangxx"]),
                "--work-dir",
                str(work_dir),
            ),
            frozenset(("board", "hardware", "compiler-optimization")),
            30.0,
            "board-0",
        )
        archived = RUNNER.archive_step_artifacts(
            test, self.root / "archived"
        )
        self.assertIsNotNone(archived)
        assert archived is not None
        self.assertEqual(archived["file_count"], 4)
        manifest = json.loads(
            pathlib.Path(str(archived["manifest"])).read_text()
        )
        self.assertEqual(
            {item["path"] for item in manifest["files"]},
            {
                "raw/case.request.raw",
                "result.json",
                "source.mlir",
                "module.so",
            },
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(
            {item["option"] for item in manifest["tools"]},
            {
                "--wafer-compile",
                "--wafer-compile-test",
                "--wafer-run",
                "--tx8-objdump",
                "--llvm-clangxx",
            },
        )
        self.assertTrue(
            all(len(item["sha256"]) == 64 for item in manifest["files"])
        )
        self.assertTrue(
            all(len(item["sha256"]) == 64 for item in manifest["tools"])
        )
        self.assertFalse((self.root / "archived" / "ignored.log").exists())

    def test_archives_executable_artifacts_for_every_pending_case(
        self,
    ) -> None:
        work_dir = self.root / "pending-board-work"
        work_dir.mkdir()
        (work_dir / "source.mlir").write_text("module {}\n")
        (work_dir / "request.meta").write_text("{}\n")
        (work_dir / "device.so").write_bytes(b"ELF")
        (work_dir / "observations.json").write_text("{}\n")
        test = RUNNER.RegisteredTest(
            "wafer-board-pending-artifact-sample",
            (
                "python3",
                "probe.py",
                "--work-dir",
                str(work_dir),
            ),
            frozenset(("board", "hardware", "ne", "pending")),
            30.0,
            "board-0",
        )

        destination = self.root / "archived-pending"
        archived = RUNNER.archive_step_artifacts(test, destination)

        self.assertIsNotNone(archived)
        assert archived is not None
        self.assertEqual(archived["file_count"], 4)
        self.assertTrue((destination / "source.mlir").is_file())
        self.assertTrue((destination / "request.meta").is_file())
        self.assertTrue((destination / "device.so").is_file())
        manifest = json.loads(
            pathlib.Path(str(archived["manifest"])).read_text()
        )
        self.assertEqual(
            {item["path"] for item in manifest["files"]},
            {
                "source.mlir",
                "request.meta",
                "device.so",
                "observations.json",
            },
        )

    def test_archives_collective_traffic_and_engine_executable_artifacts(
        self,
    ) -> None:
        work_dir = self.root / "board-work"
        work_dir.mkdir()
        (work_dir / "source.mlir").write_text("module {}\n")
        (work_dir / "request.meta").write_text("{}\n")
        (work_dir / "module.so").write_bytes(b"ELF")
        for index, labels in enumerate(
            (
                frozenset(("board", "hardware", "collective-traffic")),
                frozenset(("board", "hardware", "engine", "pipeline")),
            )
        ):
            test = RUNNER.RegisteredTest(
                f"wafer-board-artifact-sample-{index}",
                (
                    "python3",
                    "probe.py",
                    "--work-dir",
                    str(work_dir),
                ),
                labels,
                30.0,
                "board-0",
            )
            destination = self.root / f"archived-{index}"
            archived = RUNNER.archive_step_artifacts(test, destination)
            self.assertIsNotNone(archived)
            self.assertTrue((destination / "source.mlir").is_file())
            self.assertTrue((destination / "request.meta").is_file())
            self.assertTrue((destination / "module.so").is_file())

    def test_inventory_accepts_unbuilt_non_board_test(self) -> None:
        self.environment["FAKE_TESTS"] = json.dumps(
            [
                *[{"name": step.ctest_name} for step in self.steps],
                {
                    "name": "unbuilt-host-unit",
                    "command": None,
                    "labels": [],
                },
            ]
        )
        registered = RUNNER.load_registered_tests(
            str(self.fake_ctest), self.build_dir, self.environment
        )
        self.assertEqual(registered["unbuilt-host-unit"].command, ())
        RUNNER.validate_inventory(
            registered, self.steps, reject_unplanned_board_tests=True
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
