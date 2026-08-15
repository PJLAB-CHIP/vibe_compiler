#!/usr/bin/env python3
"""Run the configured TX board calibration tests as one fail-fast session.

This driver deliberately owns only host-side orchestration.  Every hardware
operation remains inside an explicitly registered CTest, with that test's
bounded timeout and normal runtime cleanup.  The driver runs one CTest process
at a time, never retries a failure, and has no reset or power-control path.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence

BOARD_TEST_DIRECTORY = (
    pathlib.Path(__file__).resolve().parents[1] / "test" / "Board"
)
if str(BOARD_TEST_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BOARD_TEST_DIRECTORY))

ARM_ENVIRONMENT_VARIABLE = "WAFER_EXECUTE_HARDWARE_TESTS"
SESSION_ENVIRONMENT_VARIABLE = "WAFER_CALIBRATION_SESSION_ID"
INITIAL_VALIDATION_CTEST = "wafer-board-dte-ncc-execution-probe"


class CalibrationRunnerError(RuntimeError):
    """A validation or execution failure that must stop the board session."""


@dataclasses.dataclass(frozen=True)
class CalibrationStep:
    key: str
    batch: str
    ctest_name: str
    purpose: str


@dataclasses.dataclass(frozen=True)
class RegisteredTest:
    name: str
    command: tuple[str, ...]
    labels: frozenset[str]
    timeout_seconds: float | None
    resource_lock: str | None
    disabled: bool = False


CALIBRATION_STEPS = (
    CalibrationStep(
        "direct-dte-ncc-execution",
        "transport",
        "wafer-board-dte-ncc-execution-probe",
        "current Direct-DTE and NCC execution with exact output and status",
    ),
    CalibrationStep(
        "complete-tile-barrier",
        "synchronization",
        "wafer-board-complete-tile-barrier-probe",
        "complete 16-Tile barrier completion and visibility",
    ),
    CalibrationStep(
        "complete-tile-add",
        "runtime-execution",
        "wafer-board-complete-tile-add",
        "complete 16-Tile add with full output and repeated completion",
    ),
    CalibrationStep(
        "complete-tile-add-profile",
        "runtime-execution",
        "wafer-board-complete-tile-add-profile",
        "one bounded Primary, Count, and Trace profile collection",
    ),
    CalibrationStep(
        "ddr-tile-offset",
        "memory",
        "wafer-board-ddr-tile-offset-probe",
        "Tile-relative DDR addressing with exact output and guards",
    ),
    CalibrationStep(
        "worker-placement",
        "worker-execution",
        "wafer-board-worker-placement-probe",
        "worker placement and bounded completion on the current runtime ABI",
    ),
    CalibrationStep(
        "spm-cross-tile-conflict",
        "memory",
        "wafer-board-spm-cross-tile-conflict-probe",
        "cross-Tile SPM conflict controls with exact output and guards",
    ),
    CalibrationStep(
        "ne-tail-throughput",
        "engine-execution",
        "wafer-board-ne-tail-throughput-probe",
        "NE tail throughput with device counters and exact output",
    ),
    CalibrationStep(
        "engine-pipeline-characterization",
        "engine-execution",
        "wafer-board-engine-pipeline-characterization-probe",
        "typed engine issue and completion behavior for one current probe",
    ),
    CalibrationStep(
        "ddr-active-tile-contention",
        "memory",
        "wafer-board-ddr-active-tile-contention-probe",
        "active-Tile DDR contention with exact inactive-Tile guards",
    ),
)

EXPLICIT_ONLY_STEPS: tuple[CalibrationStep, ...] = ()
SESSION_RISK_STEP_KEYS: tuple[str, ...] = ()
ALL_CALIBRATION_STEPS = CALIBRATION_STEPS

SELECTABLE_BATCHES = {
    "current-interface-smoke": tuple(
        step.key for step in CALIBRATION_STEPS[1:]
    ),
    "transport": (),
    "synchronization": ("complete-tile-barrier",),
    "runtime-execution": (
        "complete-tile-add",
        "complete-tile-add-profile",
    ),
    "memory": (
        "ddr-tile-offset",
        "spm-cross-tile-conflict",
        "ddr-active-tile-contention",
    ),
    "worker-execution": ("worker-placement",),
    "engine-execution": (
        "ne-tail-throughput",
        "engine-pipeline-characterization",
    ),
    "pending-hardware-calibration": tuple(
        step.key for step in CALIBRATION_STEPS[1:]
    ),
}

def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--list",
        action="store_true",
        help="list and validate the ordered CTest plan without building or executing",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help=(
            "explicitly authorize the default fail-fast board plan, or the "
            "subset named by --step/--batch"
        ),
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=repo_root / "build" / "wafer-dev",
        help="configured board-enabled CMake build directory",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        help=(
            "new directory for the session summary, per-step logs, and JUnit "
            "files; defaults below the build directory"
        ),
    )
    parser.add_argument(
        "--step",
        action="append",
        dest="selected_steps",
        metavar="KEY",
        help=(
            "execute/list only this named calibration step; repeatable. "
            "The initial Direct-DTE/NCC validation is added once automatically, "
            "and execution retains canonical order."
        ),
    )
    parser.add_argument(
        "--batch",
        action="append",
        dest="selected_batches",
        metavar="NAME",
        help=(
            "execute/list every step in this named explicit batch; repeatable. "
            "Named batches retain canonical order after the automatic initial "
            "Direct-DTE/NCC validation."
        ),
    )
    parser.add_argument("--ctest", default="ctest", help="CTest executable")
    parser.add_argument("--cmake", default="cmake", help="CMake executable")
    return parser.parse_args(argv)


def property_map(test: Mapping[str, object]) -> dict[str, object]:
    properties: dict[str, object] = {}
    raw_properties = test.get("properties", [])
    if not isinstance(raw_properties, list):
        raise CalibrationRunnerError("CTest JSON contains malformed test properties")
    for raw_property in raw_properties:
        if not isinstance(raw_property, dict):
            raise CalibrationRunnerError("CTest JSON contains a malformed property")
        name = raw_property.get("name")
        if not isinstance(name, str):
            raise CalibrationRunnerError("CTest JSON property has no string name")
        properties[name] = raw_property.get("value")
    return properties


def load_registered_tests(
    ctest: str,
    build_dir: pathlib.Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, RegisteredTest]:
    if not build_dir.is_dir():
        raise CalibrationRunnerError(f"build directory does not exist: {build_dir}")
    command = [ctest, "--test-dir", str(build_dir), "--show-only=json-v1"]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=None if environment is None else dict(environment),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CalibrationRunnerError(
            f"failed to query CTest inventory ({result.returncode}): {detail}"
        )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CalibrationRunnerError(f"CTest returned invalid JSON: {error}") from error
    raw_tests = document.get("tests")
    if not isinstance(raw_tests, list):
        raise CalibrationRunnerError("CTest JSON has no test inventory")

    registered: dict[str, RegisteredTest] = {}
    for raw_test in raw_tests:
        if not isinstance(raw_test, dict):
            raise CalibrationRunnerError("CTest JSON contains a malformed test")
        name = raw_test.get("name")
        command_value = raw_test.get("command")
        if not isinstance(name, str):
            raise CalibrationRunnerError("CTest JSON test has no string name")
        if command_value is None:
            command: tuple[str, ...] = ()
        elif isinstance(command_value, list):
            command = tuple(str(argument) for argument in command_value)
        else:
            raise CalibrationRunnerError(
                f"CTest {name} has a malformed command"
            )
        if name in registered:
            raise CalibrationRunnerError(f"CTest registered duplicate test name: {name}")
        properties = property_map(raw_test)
        raw_labels = properties.get("LABELS", [])
        if isinstance(raw_labels, str):
            labels = frozenset((raw_labels,))
        elif isinstance(raw_labels, list) and all(
            isinstance(label, str) for label in raw_labels
        ):
            labels = frozenset(raw_labels)
        else:
            raise CalibrationRunnerError(f"CTest {name} has malformed LABELS")
        raw_timeout = properties.get("TIMEOUT")
        timeout = (
            float(raw_timeout)
            if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool)
            else None
        )
        raw_lock = properties.get("RESOURCE_LOCK")
        if isinstance(raw_lock, str):
            resource_lock = raw_lock
        elif (
            isinstance(raw_lock, list)
            and raw_lock
            and all(isinstance(lock, str) and lock for lock in raw_lock)
        ):
            resource_lock = ";".join(raw_lock)
        else:
            resource_lock = None
        raw_disabled = properties.get("DISABLED", False)
        disabled = raw_disabled is True or (
            isinstance(raw_disabled, str)
            and raw_disabled.upper() in {"1", "ON", "TRUE", "YES"}
        )
        registered[name] = RegisteredTest(
            name=name,
            command=command,
            labels=labels,
            timeout_seconds=timeout,
            resource_lock=resource_lock,
            disabled=disabled,
        )
    return registered


def is_forbidden_control_argument(argument: str) -> bool:
    lowered = argument.lower()
    if lowered in {"reset", "power", "power-cycle", "power_cycle"}:
        return True
    if not lowered.startswith("-"):
        return False
    option = lowered.split("=", maxsplit=1)[0]
    return "reset" in option or "power" in option


def validate_step_definition(steps: Sequence[CalibrationStep]) -> None:
    if not steps:
        raise CalibrationRunnerError("hardware calibration plan is empty")
    keys = [step.key for step in steps]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count != 1)
    if duplicate_keys:
        raise CalibrationRunnerError(f"duplicate calibration step keys: {duplicate_keys}")
    for step in steps:
        if not step.batch or not step.ctest_name or not step.purpose:
            raise CalibrationRunnerError(f"incomplete calibration step: {step.key}")
        if is_forbidden_control_argument(step.ctest_name):
            raise CalibrationRunnerError(
                f"calibration step names a forbidden control action: {step.ctest_name}"
            )
    duplicates = {
        name: count
        for name, count in Counter(step.ctest_name for step in steps).items()
        if count > 1
    }
    if duplicates:
        raise CalibrationRunnerError(
            f"calibration CTests must not repeat: {duplicates}"
        )


def validate_default_plan() -> None:
    validate_step_definition(CALIBRATION_STEPS)
    validate_step_definition(ALL_CALIBRATION_STEPS)
    if (
        CALIBRATION_STEPS[0].key != "direct-dte-ncc-execution"
        or CALIBRATION_STEPS[0].ctest_name != INITIAL_VALIDATION_CTEST
    ):
        raise CalibrationRunnerError(
            "default plan must begin with Direct-DTE/NCC validation"
        )
    if ALL_CALIBRATION_STEPS[0] != CALIBRATION_STEPS[0]:
        raise CalibrationRunnerError(
            "selectable plan must retain the initial Direct-DTE/NCC validation"
        )
    default_keys = {step.key for step in CALIBRATION_STEPS}
    explicit_keys = {step.key for step in EXPLICIT_ONLY_STEPS}
    if default_keys & explicit_keys:
        raise CalibrationRunnerError(
            "explicit-only calibration steps must not enter the default plan"
        )


def select_calibration_steps(
    selected_keys: Sequence[str] | None,
    selected_batches: Sequence[str] | None = None,
) -> tuple[CalibrationStep, ...]:
    validate_default_plan()
    if not selected_keys and not selected_batches:
        return CALIBRATION_STEPS
    selected_keys = tuple(selected_keys or ())
    selected_batches = tuple(selected_batches or ())
    duplicate_batches = sorted(
        batch
        for batch, count in Counter(selected_batches).items()
        if count > 1
    )
    if duplicate_batches:
        raise CalibrationRunnerError(
            f"calibration batches were selected more than once: "
            f"{duplicate_batches}"
        )
    unknown_batches = sorted(set(selected_batches) - set(SELECTABLE_BATCHES))
    if unknown_batches:
        raise CalibrationRunnerError(
            f"unknown calibration batch names: {unknown_batches}"
        )
    expanded_keys = list(selected_keys)
    for batch in selected_batches:
        expanded_keys.extend(SELECTABLE_BATCHES[batch])
    duplicates = sorted(
        key for key, count in Counter(expanded_keys).items() if count > 1
    )
    if duplicates:
        raise CalibrationRunnerError(
            f"calibration steps were selected more than once: {duplicates}"
        )
    automatic_keys = {CALIBRATION_STEPS[0].key}
    explicitly_automatic = sorted(set(expanded_keys) & automatic_keys)
    if explicitly_automatic:
        raise CalibrationRunnerError(
            "the initial validation is automatic and cannot be selected "
            "explicitly: "
            f"{explicitly_automatic}"
        )
    selectable = {
        step.key: step for step in ALL_CALIBRATION_STEPS[1:]
    }
    unknown = sorted(set(expanded_keys) - set(selectable))
    if unknown:
        raise CalibrationRunnerError(
            f"unknown calibration step keys: {unknown}"
        )
    selected = set(expanded_keys)
    ordered = tuple(
        step
        for step in ALL_CALIBRATION_STEPS[1:]
        if step.key in selected
    )
    return (
        CALIBRATION_STEPS[0],
        *ordered,
    )


def validate_inventory(
    registered: Mapping[str, RegisteredTest],
    steps: Sequence[CalibrationStep],
    *,
    reject_unplanned_board_tests: bool = True,
) -> None:
    validate_step_definition(steps)
    planned_names = {step.ctest_name for step in steps}
    missing = sorted(planned_names - set(registered))
    if missing:
        raise CalibrationRunnerError(
            "board-enabled build is missing planned CTests: " + ", ".join(missing)
        )

    for name in sorted(planned_names):
        test = registered[name]
        if not {"board", "hardware"}.issubset(test.labels):
            raise CalibrationRunnerError(
                f"planned CTest {name} lacks board+hardware labels: "
                f"{sorted(test.labels)}"
            )
        if "no-card" in test.labels or "--no-card" in test.command:
            raise CalibrationRunnerError(
                f"planned CTest {name} resolves to a no-card path"
            )
        if test.timeout_seconds is None or test.timeout_seconds <= 0:
            raise CalibrationRunnerError(
                f"planned CTest {name} has no positive bounded TIMEOUT"
            )
        if not test.resource_lock:
            raise CalibrationRunnerError(
                f"planned CTest {name} has no board RESOURCE_LOCK"
            )
        if test.disabled:
            raise CalibrationRunnerError(f"planned CTest {name} is disabled")
        forbidden = [
            argument
            for argument in test.command
            if is_forbidden_control_argument(argument)
        ]
        if forbidden:
            raise CalibrationRunnerError(
                f"planned CTest {name} exposes reset/power control: {forbidden}"
            )

    if reject_unplanned_board_tests:
        eligible = {
            name
            for name, test in registered.items()
            if {"board", "hardware"}.issubset(test.labels)
            and "no-card" not in test.labels
        }
        unplanned = sorted(eligible - planned_names)
        if unplanned:
            raise CalibrationRunnerError(
                "new board CTests require an explicit batch decision before execution: "
                + ", ".join(unplanned)
            )


def default_log_dir(build_dir: pathlib.Path) -> pathlib.Path:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        build_dir
        / "hardware-calibration-logs"
        / f"{stamp}-{os.getpid()}"
    )


def write_summary(path: pathlib.Path, summary: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stream_command(
    command: Sequence[str],
    *,
    log_path: pathlib.Path,
    cwd: pathlib.Path,
    environment: Mapping[str, str],
) -> int:
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"started_at: {utc_now()}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"command: {shlex.join(command)}\n\n")
        log.flush()
        with subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise


def junit_result(junit_path: pathlib.Path, expected_name: str) -> str | None:
    if not junit_path.is_file():
        return "CTest did not produce the expected JUnit result"
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError as error:
        return f"CTest produced malformed JUnit XML: {error}"
    cases = root.findall(".//testcase")
    if root.tag == "testcase":
        cases = [root]
    if len(cases) != 1:
        return f"CTest JUnit contains {len(cases)} cases instead of exactly one"
    case = cases[0]
    if case.get("name") != expected_name:
        return (
            f"CTest JUnit reported {case.get('name')!r}, expected {expected_name!r}"
        )
    if case.find("skipped") is not None:
        return "hardware CTest was skipped"
    if case.find("failure") is not None or case.find("error") is not None:
        return "hardware CTest reported a failure"
    for node in (root, case):
        for attribute in ("failures", "errors", "skipped", "disabled"):
            value = node.get(attribute)
            if value not in (None, "0"):
                return f"hardware CTest JUnit has {attribute}={value}"
    return None


def ctest_command(
    ctest: str,
    build_dir: pathlib.Path,
    test_name: str,
    junit_path: pathlib.Path,
) -> list[str]:
    return [
        ctest,
        "--test-dir",
        str(build_dir),
        "--verbose",
        "--output-on-failure",
        "--no-tests=error",
        "--parallel",
        "1",
        "--tests-regex",
        f"^{re.escape(test_name)}$",
        "--output-junit",
        str(junit_path),
    ]


def command_option_value(
    command: Sequence[str], option: str
) -> str | None:
    prefix = option + "="
    for index, argument in enumerate(command):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option:
            if index + 1 >= len(command):
                raise CalibrationRunnerError(
                    f"CTest command ends after {option}"
                )
            return command[index + 1]
    return None


def profile_report_members(
    test: RegisteredTest, work_dir: pathlib.Path
) -> tuple[pathlib.Path, ...]:
    if "profiler" not in test.labels:
        return ()

    runs = work_dir / "package.profile" / "runs"
    current = runs / "current"
    if not current.is_symlink():
        raise CalibrationRunnerError(
            f"{test.name}: profiler did not write a managed current report"
        )
    try:
        canonical_runs = runs.resolve(strict=True)
        run_directory = current.resolve(strict=True)
    except OSError as error:
        raise CalibrationRunnerError(
            f"{test.name}: profiler current report cannot be resolved: {error}"
        ) from error
    if run_directory.parent != canonical_runs:
        raise CalibrationRunnerError(
            f"{test.name}: profiler current report escapes its runs directory"
        )
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise CalibrationRunnerError(
            f"{test.name}: profiler current target is not a real directory"
        )

    expected_names = {"evidence.json", "analysis.json", "index.html"}
    actual_members = tuple(run_directory.iterdir())
    if (
        {path.name for path in actual_members} != expected_names
        or len(actual_members) != len(expected_names)
        or any(not path.is_file() or path.is_symlink() for path in actual_members)
    ):
        raise CalibrationRunnerError(
            f"{test.name}: profiler current report is not the complete "
            "three report files"
        )
    evidence_path = run_directory / "evidence.json"
    try:
        evidence = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationRunnerError(
            f"{test.name}: profiler evidence is not readable JSON: {error}"
        ) from error
    if (
        not isinstance(evidence, dict)
        or evidence.get("run_id") != run_directory.name
    ):
        raise CalibrationRunnerError(
            f"{test.name}: profiler evidence run_id does not match current "
            "report directory"
        )
    return tuple(
        run_directory / name
        for name in ("evidence.json", "analysis.json", "index.html")
    )


def collect_step_evidence(
    test: RegisteredTest,
    destination: pathlib.Path,
) -> dict[str, object] | None:
    raw_work_dir = command_option_value(test.command, "--work-dir")
    if raw_work_dir is None:
        return None
    work_dir = pathlib.Path(raw_work_dir)
    if not work_dir.is_dir():
        raise CalibrationRunnerError(
            f"{test.name}: declared work directory was not produced: {work_dir}"
        )
    durable_suffixes = {".raw", ".json", ".jsonl"}
    preserves_compiler_ir = (
        "pending" in test.labels or "current-interface" in test.labels
    )
    if preserves_compiler_ir:
        # Preserve exact source snapshots and final linked ELFs as read-only
        # evidence for the current calibration execution.
        durable_suffixes.update({".mlir", ".meta", ".so"})
    required_profile_report = profile_report_members(test, work_dir)
    evidence_files = tuple(
        sorted(
            {
                *required_profile_report,
                *(
                    path
                    for path in work_dir.rglob("*")
                    if path.is_file()
                    and (
                        path.suffix in durable_suffixes
                        or path.name in {"session.txt", "summary.txt"}
                    )
                ),
            }
        )
    )
    destination.mkdir(parents=True, exist_ok=False)
    manifest_files: list[dict[str, object]] = []
    for source in evidence_files:
        relative = source.relative_to(work_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest_files.append(
            {
                "path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": digest,
            }
        )
    tool_options = (
        "--wafer-compile",
        "--wafer-compile-test",
        "--wafer-run",
        "--tx8-objdump",
        "--llvm-clangxx",
    )
    tools: list[dict[str, object]] = []
    for option in tool_options:
        value = command_option_value(test.command, option)
        if value is None:
            continue
        tool = pathlib.Path(value)
        if not tool.is_file():
            raise CalibrationRunnerError(
                f"{test.name}: {option} tool does not exist: {tool}"
            )
        tools.append(
            {
                "option": option,
                "path": str(tool),
                "bytes": tool.stat().st_size,
                "sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "ctest": test.name,
        "ctest_command": list(test.command),
        "source_work_dir": str(work_dir),
        "files": manifest_files,
        "tools": tools,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {
        "directory": str(destination),
        "manifest": str(manifest_path),
        "file_count": len(manifest_files),
    }


def execute_calibration(
    *,
    build_dir: pathlib.Path,
    log_dir: pathlib.Path,
    ctest: str,
    cmake: str,
    steps: Sequence[CalibrationStep],
    inventory_steps: Sequence[CalibrationStep] | None = None,
    environment: Mapping[str, str],
) -> int:
    validate_step_definition(steps)
    audited_steps = steps if inventory_steps is None else inventory_steps
    validate_step_definition(audited_steps)
    if not {step.ctest_name for step in steps}.issubset(
        {step.ctest_name for step in audited_steps}
    ):
        raise CalibrationRunnerError(
            "execution plan contains a CTest outside the audited inventory"
        )
    if environment.get(ARM_ENVIRONMENT_VARIABLE) != "1":
        raise CalibrationRunnerError(
            f"execution requires {ARM_ENVIRONMENT_VARIABLE}=1"
        )
    if not build_dir.is_dir():
        raise CalibrationRunnerError(f"build directory does not exist: {build_dir}")
    registered = load_registered_tests(ctest, build_dir, environment)
    validate_inventory(registered, audited_steps)
    log_dir.mkdir(parents=True, exist_ok=False)
    summary_path = log_dir / "session.json"
    calibration_session_id = secrets.token_hex(16)
    summary: dict[str, object] = {
        "status": "running",
        "started_at": utc_now(),
        "build_dir": str(build_dir),
        "calibration_session_id": calibration_session_id,
        "hardware_execution_armed": (
            environment.get(ARM_ENVIRONMENT_VARIABLE) == "1"
        ),
        "automatic_retry": False,
        "reset_or_power_control": False,
        "steps": [],
    }
    write_summary(summary_path, summary)

    child_environment = dict(environment)
    child_environment[SESSION_ENVIRONMENT_VARIABLE] = calibration_session_id
    child_environment["CTEST_PARALLEL_LEVEL"] = "1"
    child_environment["CTEST_OUTPUT_ON_FAILURE"] = "1"

    build_log = log_dir / "000-incremental-build.log"
    build_targets = ["wafer-compile", "wafer-run"]
    build_command = [
        cmake,
        "--build",
        str(build_dir),
        "--target",
        *build_targets,
        "--parallel",
        str(max(1, os.cpu_count() or 1)),
    ]
    print(f"[build] {shlex.join(build_command)}")
    build_started = time.monotonic()
    build_returncode = stream_command(
        build_command,
        log_path=build_log,
        cwd=build_dir,
        environment=child_environment,
    )
    summary["incremental_build"] = {
        "status": "passed" if build_returncode == 0 else "failed",
        "returncode": build_returncode,
        "elapsed_seconds": round(time.monotonic() - build_started, 3),
        "log": str(build_log),
    }
    if build_returncode != 0:
        summary["status"] = "failed"
        summary["failure"] = "incremental build failed before board execution"
        summary["ended_at"] = utc_now()
        write_summary(summary_path, summary)
        return 1

    try:
        registered = load_registered_tests(ctest, build_dir, child_environment)
        validate_inventory(registered, audited_steps)
    except CalibrationRunnerError as error:
        summary["status"] = "failed"
        summary["failure"] = str(error)
        summary["ended_at"] = utc_now()
        write_summary(summary_path, summary)
        print(f"hardware_calibration: {error}", file=sys.stderr)
        return 1

    summary["planned_ctest_count"] = len(steps)
    summary["unique_ctest_count"] = len({step.ctest_name for step in steps})
    write_summary(summary_path, summary)

    step_results = summary["steps"]
    assert isinstance(step_results, list)
    for index, step in enumerate(steps, start=1):
        stem = f"{index:03d}-{step.batch}-{step.key}"
        log_path = log_dir / f"{stem}.log"
        junit_path = log_dir / f"{stem}.junit.xml"
        command = ctest_command(ctest, build_dir, step.ctest_name, junit_path)
        print(
            f"[{index}/{len(steps)}] {step.batch}: {step.ctest_name}",
            flush=True,
        )
        started_at = utc_now()
        started = time.monotonic()
        try:
            returncode = stream_command(
                command,
                log_path=log_path,
                cwd=build_dir,
                environment=child_environment,
            )
        except KeyboardInterrupt:
            result = {
                "key": step.key,
                "batch": step.batch,
                "ctest": step.ctest_name,
                "status": "interrupted",
                "started_at": started_at,
                "ended_at": utc_now(),
                "log": str(log_path),
                "junit": str(junit_path),
            }
            step_results.append(result)
            summary["status"] = "interrupted"
            summary["failure"] = (
                "host orchestration was interrupted; no retry, reset, or power "
                "action was attempted"
            )
            summary["ended_at"] = utc_now()
            write_summary(summary_path, summary)
            raise
        junit_error = (
            junit_result(junit_path, step.ctest_name)
            if returncode == 0
            else f"CTest process exited with code {returncode}"
        )
        evidence_error: str | None = None
        evidence: dict[str, object] | None = None
        try:
            evidence = collect_step_evidence(
                registered[step.ctest_name],
                log_dir / "evidence" / stem,
            )
        except (CalibrationRunnerError, OSError) as error:
            evidence_error = f"failed to archive durable evidence: {error}"
            if junit_error is None:
                junit_error = evidence_error
        status = "passed" if junit_error is None else "failed"
        result = {
            "key": step.key,
            "batch": step.batch,
            "ctest": step.ctest_name,
            "purpose": step.purpose,
            "status": status,
            "returncode": returncode,
            "started_at": started_at,
            "ended_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path),
            "junit": str(junit_path),
        }
        if evidence is not None:
            result["evidence"] = evidence
        if evidence_error is not None:
            result["evidence_error"] = evidence_error
        if junit_error is not None:
            result["failure"] = junit_error
        step_results.append(result)
        write_summary(summary_path, summary)
        if junit_error is not None:
            summary["status"] = "failed"
            summary["failure"] = (
                f"{step.ctest_name}: {junit_error}; all later board tests were "
                "left unexecuted"
            )
            summary["ended_at"] = utc_now()
            write_summary(summary_path, summary)
            print(f"hardware_calibration: {summary['failure']}", file=sys.stderr)
            return 1

    summary["status"] = "passed"
    summary["ended_at"] = utc_now()
    write_summary(summary_path, summary)
    print(f"hardware_calibration: passed; evidence: {log_dir}")
    return 0


def print_plan(
    registered: Mapping[str, RegisteredTest], steps: Sequence[CalibrationStep]
) -> None:
    print("ORDER  BATCH                         CTEST")
    for index, step in enumerate(steps, start=1):
        test = registered[step.ctest_name]
        print(
            f"{index:>5}  {step.batch:<28}  {step.ctest_name} "
            f"(timeout={test.timeout_seconds:g}s)"
        )
    print(
        f"\n{len(steps)} ordered executions, "
        f"{len({step.ctest_name for step in steps})} unique board CTests; "
        "one process at a time, no retry/reset/power."
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    args = parse_args(argv)
    active_environment = dict(os.environ if environment is None else environment)
    build_dir = args.build_dir.resolve()
    try:
        validate_default_plan()
        selected_steps = select_calibration_steps(
            args.selected_steps, args.selected_batches
        )
        if args.list:
            registered = load_registered_tests(
                args.ctest, build_dir, active_environment
            )
            validate_inventory(registered, ALL_CALIBRATION_STEPS)
            print_plan(registered, selected_steps)
            return 0

        if active_environment.get(ARM_ENVIRONMENT_VARIABLE) != "1":
            raise CalibrationRunnerError(
                f"--execute also requires {ARM_ENVIRONMENT_VARIABLE}=1"
            )
        log_dir = (
            args.log_dir.resolve()
            if args.log_dir is not None
            else default_log_dir(build_dir)
        )
        return execute_calibration(
            build_dir=build_dir,
            log_dir=log_dir,
            ctest=args.ctest,
            cmake=args.cmake,
            steps=selected_steps,
            inventory_steps=ALL_CALIBRATION_STEPS,
            environment=active_environment,
        )
    except CalibrationRunnerError as error:
        print(f"hardware_calibration: {error}", file=sys.stderr)
        return 2
    except FileExistsError as error:
        print(
            "hardware_calibration: refusing to overwrite an existing log "
            f"directory: {error.filename}",
            file=sys.stderr,
        )
        return 2
    except OSError as error:
        print(f"hardware_calibration: host orchestration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "hardware_calibration: interrupted; stopped without retry/reset/power",
            file=sys.stderr,
        )
        raise SystemExit(130)
