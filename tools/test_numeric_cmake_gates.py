#!/usr/bin/env python3
"""Exercise fail-closed numeric dependency configuration with real CMake runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import shutil
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class ConfigurationCase:
    name: str
    root_kind: str
    record_kind: str
    expected_diagnostic: str


CASES = (
    ConfigurationCase(
        name="missing-root",
        root_kind="missing",
        record_kind="missing",
        expected_diagnostic="requires a managed dependency root",
    ),
    ConfigurationCase(
        name="missing-record",
        root_kind="directory",
        record_kind="missing",
        expected_diagnostic="requires the completed managed numeric-model conformance record",
    ),
    ConfigurationCase(
        name="invalid-record",
        root_kind="directory",
        record_kind="invalid",
        expected_diagnostic="dependency identity/conformance check failed",
    ),
)


def existing_file(value: str) -> pathlib.Path:
    path = pathlib.Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def existing_directory(value: str) -> pathlib.Path:
    path = pathlib.Path(value).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True, type=existing_file)
    parser.add_argument("--source-dir", required=True, type=existing_directory)
    parser.add_argument("--work-root", required=True, type=pathlib.Path)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--generator-platform")
    parser.add_argument("--generator-toolset")
    parser.add_argument("--make-program", type=existing_file)
    parser.add_argument("--c-compiler", required=True, type=existing_file)
    parser.add_argument("--cxx-compiler", required=True, type=existing_file)
    parser.add_argument("--llvm-dir", required=True, type=existing_directory)
    parser.add_argument("--mlir-dir", required=True, type=existing_directory)
    parser.add_argument("--python", required=True, type=existing_file)
    parser.add_argument("--build-type")
    return parser.parse_args()


def cmake_definition(name: str, value: object) -> str:
    return f"-D{name}={value}"


def common_configure_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.cmake),
        "-S",
        str(args.source_dir),
        "-G",
        args.generator,
        "--no-warn-unused-cli",
        "-Wno-dev",
        cmake_definition("CMAKE_C_COMPILER", args.c_compiler),
        cmake_definition("CMAKE_CXX_COMPILER", args.cxx_compiler),
        cmake_definition("LLVM_DIR", args.llvm_dir),
        cmake_definition("MLIR_DIR", args.mlir_dir),
        cmake_definition("Python3_EXECUTABLE", args.python),
        cmake_definition("CMAKE_FIND_USE_PACKAGE_REGISTRY", "FALSE"),
        cmake_definition("CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY", "FALSE"),
        cmake_definition("FETCHCONTENT_FULLY_DISCONNECTED", "ON"),
        cmake_definition("FETCHCONTENT_UPDATES_DISCONNECTED", "ON"),
        cmake_definition("WAFER_FETCH_GTEST", "OFF"),
        cmake_definition("WAFER_ENABLE_IMPORTER_DEPS", "OFF"),
        cmake_definition("WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS", "OFF"),
        cmake_definition("WAFER_ENABLE_SPMD_PARTITIONER_DEPS", "OFF"),
        cmake_definition("WAFER_ENABLE_RUNTIME_DEPS", "OFF"),
        cmake_definition("WAFER_ENABLE_UNIT_TESTS", "OFF"),
        cmake_definition("WAFER_ENABLE_NUMERIC_MODEL_DEPS", "ON"),
    ]
    if args.generator_platform:
        command.extend(["-A", args.generator_platform])
    if args.generator_toolset:
        command.extend(["-T", args.generator_toolset])
    if args.make_program:
        command.append(cmake_definition("CMAKE_MAKE_PROGRAM", args.make_program))
    if args.build_type:
        command.append(cmake_definition("CMAKE_BUILD_TYPE", args.build_type))
    return command


def prepare_case(
    root: pathlib.Path, case: ConfigurationCase
) -> tuple[pathlib.Path, pathlib.Path]:
    managed_root = root / "managed-root"
    record = managed_root / "numeric-model-deps.json"
    if case.root_kind == "directory":
        managed_root.mkdir(parents=True)
    elif case.root_kind != "missing":
        raise ValueError(f"unknown root kind: {case.root_kind}")

    if case.record_kind == "invalid":
        record.write_text("{}\n", encoding="utf-8")
    elif case.record_kind != "missing":
        raise ValueError(f"unknown record kind: {case.record_kind}")
    return managed_root, record


def run_case(
    args: argparse.Namespace,
    temporary_root: pathlib.Path,
    case: ConfigurationCase,
) -> None:
    case_root = temporary_root / case.name
    case_root.mkdir()
    managed_root, record = prepare_case(case_root, case)
    build_dir = case_root / "build"
    command = common_configure_command(args)
    command.extend(
        [
            "-B",
            str(build_dir),
            cmake_definition("WAFER_NUMERIC_MODEL_DEPS_ROOT", managed_root),
            cmake_definition("WAFER_NUMERIC_MODEL_DEPS_RECORD", record),
        ]
    )
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        raise RuntimeError(
            f"{case.name}: CMake configure unexpectedly succeeded\n{result.stdout}"
        )
    normalized_output = " ".join(result.stdout.split())
    normalized_diagnostic = " ".join(case.expected_diagnostic.split())
    if normalized_diagnostic not in normalized_output:
        raise RuntimeError(
            f"{case.name}: expected diagnostic {case.expected_diagnostic!r} was absent\n"
            f"command: {' '.join(command)}\n{result.stdout}"
        )
    print(f"{case.name}: rejected with expected numeric dependency diagnostic")


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix="numeric-cmake-gates-", dir=work_root)
    )
    try:
        for case in CASES:
            run_case(args, temporary, case)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print(f"checked {len(CASES)} fail-closed numeric CMake configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
