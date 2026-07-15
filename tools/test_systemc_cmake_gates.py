#!/usr/bin/env python3
"""Exercise fail-closed managed SystemC configuration with real CMake runs."""

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
    numeric_enabled: bool
    root_kind: str
    record_kind: str
    expected_diagnostic: str


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
    parser.add_argument("--numeric-root", type=existing_directory)
    parser.add_argument("--numeric-record", type=existing_file)
    parser.add_argument("--systemc-root", type=existing_directory)
    parser.add_argument("--systemc-record", type=existing_file)
    args = parser.parse_args()
    if (args.numeric_root is None) != (args.numeric_record is None):
        parser.error("--numeric-root and --numeric-record must be supplied together")
    if (args.systemc_root is None) != (args.systemc_record is None):
        parser.error("--systemc-root and --systemc-record must be supplied together")
    if args.systemc_root is not None and args.numeric_root is None:
        parser.error("--systemc-root requires --numeric-root")
    return args


def definition(name: str, value: object) -> str:
    return f"-D{name}={value}"


def common_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.cmake),
        "-S",
        str(args.source_dir),
        "-G",
        args.generator,
        "--no-warn-unused-cli",
        "-Wno-dev",
        definition("CMAKE_C_COMPILER", args.c_compiler),
        definition("CMAKE_CXX_COMPILER", args.cxx_compiler),
        definition("LLVM_DIR", args.llvm_dir),
        definition("MLIR_DIR", args.mlir_dir),
        definition("Python3_EXECUTABLE", args.python),
        definition("CMAKE_FIND_USE_PACKAGE_REGISTRY", "FALSE"),
        definition("CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY", "FALSE"),
        definition("FETCHCONTENT_FULLY_DISCONNECTED", "ON"),
        definition("FETCHCONTENT_UPDATES_DISCONNECTED", "ON"),
        definition("WAFER_FETCH_GTEST", "OFF"),
        definition("WAFER_ENABLE_IMPORTER_DEPS", "OFF"),
        definition("WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS", "OFF"),
        definition("WAFER_ENABLE_SPMD_PARTITIONER_DEPS", "OFF"),
        definition("WAFER_ENABLE_RUNTIME_DEPS", "OFF"),
        definition("WAFER_ENABLE_BULK_MODEL_DEPS", "OFF"),
        definition("WAFER_ENABLE_UNIT_TESTS", "OFF"),
        definition("WAFER_ENABLE_SYSTEMC_MODEL", "ON"),
    ]
    if args.generator_platform:
        command.extend(["-A", args.generator_platform])
    if args.generator_toolset:
        command.extend(["-T", args.generator_toolset])
    if args.make_program:
        command.append(definition("CMAKE_MAKE_PROGRAM", args.make_program))
    if args.build_type:
        command.append(definition("CMAKE_BUILD_TYPE", args.build_type))
    return command


def cases(args: argparse.Namespace) -> list[ConfigurationCase]:
    result = [
        ConfigurationCase(
            "numeric-disabled",
            False,
            "missing",
            "missing",
            "requires WAFER_ENABLE_NUMERIC_MODEL_DEPS=ON",
        )
    ]
    if args.numeric_root:
        result.extend(
            [
                ConfigurationCase(
                    "missing-root",
                    True,
                    "missing",
                    "missing",
                    "requires a managed SystemC-model root",
                ),
                ConfigurationCase(
                    "missing-record",
                    True,
                    "directory",
                    "missing",
                    "requires the completed managed SystemC-model dependency record",
                ),
                ConfigurationCase(
                    "invalid-record",
                    True,
                    "directory",
                    "invalid",
                    "SystemC-model dependency identity/conformance check failed",
                ),
            ]
        )
    return result


def run_case(args: argparse.Namespace, temporary: pathlib.Path, case: ConfigurationCase) -> None:
    case_root = temporary / case.name
    case_root.mkdir()
    managed_root = case_root / "managed-root"
    record = managed_root / "systemc-model-deps.json"
    if case.root_kind == "directory":
        managed_root.mkdir()
    if case.record_kind == "invalid":
        record.write_text("{}\n", encoding="utf-8")
    command = common_command(args)
    command.extend(
        [
            "-B",
            str(case_root / "build"),
            definition("WAFER_ENABLE_NUMERIC_MODEL_DEPS", "ON" if case.numeric_enabled else "OFF"),
            definition("WAFER_SYSTEMC_MODEL_DEPS_ROOT", managed_root),
            definition("WAFER_SYSTEMC_MODEL_DEPS_RECORD", record),
        ]
    )
    if case.numeric_enabled:
        command.extend(
            [
                definition("WAFER_NUMERIC_MODEL_DEPS_ROOT", args.numeric_root),
                definition("WAFER_NUMERIC_MODEL_DEPS_RECORD", args.numeric_record),
            ]
        )
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if completed.returncode == 0:
        raise RuntimeError(f"{case.name}: CMake configure unexpectedly succeeded\n{completed.stdout}")
    normalized = " ".join(completed.stdout.split())
    if " ".join(case.expected_diagnostic.split()) not in normalized:
        raise RuntimeError(
            f"{case.name}: expected diagnostic {case.expected_diagnostic!r} was absent\n"
            f"command: {' '.join(command)}\n{completed.stdout}"
        )
    print(f"{case.name}: rejected with expected SystemC dependency diagnostic")


def run_stale_package_cache_case(
    args: argparse.Namespace, temporary: pathlib.Path
) -> None:
    if args.systemc_root is None:
        return
    case_root = temporary / "stale-package-cache"
    stale_package = case_root / "stale-package"
    stale_package.mkdir(parents=True)
    stale_library = stale_package / "libsystemc-stale.a"
    stale_library.write_bytes(b"stale")
    (stale_package / "SystemCLanguageConfig.cmake").write_text(
        "set(SystemCLanguage_VERSION 3.0.2)\n"
        "add_library(SystemC::systemc STATIC IMPORTED)\n"
        f'set_target_properties(SystemC::systemc PROPERTIES '
        f'IMPORTED_LOCATION "{stale_library}")\n',
        encoding="utf-8",
    )
    build = case_root / "build"
    command = common_command(args)
    command.extend(
        [
            "-B",
            str(build),
            definition("WAFER_ENABLE_NUMERIC_MODEL_DEPS", "ON"),
            definition("WAFER_NUMERIC_MODEL_DEPS_ROOT", args.numeric_root),
            definition("WAFER_NUMERIC_MODEL_DEPS_RECORD", args.numeric_record),
            definition("WAFER_SYSTEMC_MODEL_DEPS_ROOT", args.systemc_root),
            definition("WAFER_SYSTEMC_MODEL_DEPS_RECORD", args.systemc_record),
            definition("SystemCLanguage_DIR", stale_package),
        ]
    )
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "stale-package-cache: managed root did not replace stale package cache\n"
            f"command: {' '.join(command)}\n{completed.stdout}"
        )
    expected = (
        args.systemc_root
        / "install"
        / "systemc"
        / "lib"
        / "cmake"
        / "SystemCLanguage"
    ).resolve()
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    if f"SystemCLanguage_DIR:PATH={expected}" not in cache:
        raise RuntimeError(
            "stale-package-cache: CMake cache did not select the validated package"
        )
    print("stale-package-cache: selected the validated managed SystemC package")


def main() -> int:
    args = parse_args()
    args.work_root.resolve().mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix="systemc-cmake-gates-", dir=args.work_root.resolve())
    )
    try:
        for case in cases(args):
            run_case(args, temporary, case)
        run_stale_package_cache_case(args, temporary)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    positive = 1 if args.systemc_root is not None else 0
    print(
        f"checked {len(cases(args))} fail-closed and {positive} "
        "managed-root SystemC CMake configurations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
