#!/usr/bin/env python3
"""Check Wafer dependency pins and importer/backend isolation."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSIONS_FILE = REPO_ROOT / "cmake" / "WaferDependencyVersions.cmake"


REQUIRED_KEYS = [
    "WAFER_LLVM_VERSION",
    "WAFER_STABLEHLO_TAG",
    "WAFER_STABLEHLO_COMMIT",
    "WAFER_SHARDY_COMMIT",
    "WAFER_GOOGLETEST_TAG",
    "WAFER_PYTHON_LIT_VERSION",
]

SOURCE_SUFFIXES = {".cpp", ".h", ".td"}

STABLEHLO_API_NEEDLES = [
    "stablehlo/",
    "mlir::stablehlo",
]

STABLEHLO_API_ALLOWED_PREFIXES = [
    "include/Wafer/Frontend",
    "lib/Wafer/Transforms/StableHLOToLinalg",
    "tools/wafer-import-model",
]

RUNTIME_DRIVER_NEEDLES = [
    "tx_runtime",
    "libhpgr",
    "Tsm",
]

TEST_TOOLING_NEEDLES = [
    "GTest",
    "gtest",
    "FileCheck",
]


def load_versions() -> dict[str, str]:
    text = VERSIONS_FILE.read_text(encoding="utf-8")
    return dict(re.findall(r'set\((WAFER_[A-Z0-9_]+)\s+"([^"]+)"\)', text))


def rev_parse(path: pathlib.Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def check_text_contains(path: pathlib.Path, needle: str) -> None:
    if needle not in path.read_text(encoding="utf-8"):
        raise RuntimeError(f"{path} does not contain required text: {needle}")


def rel(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def is_under(path: pathlib.Path, prefix: str) -> bool:
    relative = rel(path)
    return relative == prefix or relative.startswith(prefix + "/")


def iter_source_files(roots: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
    for root in roots:
        if root.is_file():
            if root.suffix in SOURCE_SUFFIXES:
                yield root
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                yield path


def check_forbidden_needles(
    *,
    label: str,
    roots: Iterable[pathlib.Path],
    needles: Iterable[str],
    allowed_prefixes: Iterable[str] = (),
) -> None:
    violations: list[str] = []
    for path in iter_source_files(roots):
        if any(is_under(path, prefix) for prefix in allowed_prefixes):
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                violations.append(f"{rel(path)} contains {needle!r}")
    if violations:
        raise RuntimeError(
            f"{label} dependency boundary violation(s): " + "; ".join(violations)
        )


def check_cmake_target_visibility() -> None:
    transforms_cmake_path = REPO_ROOT / "lib" / "Wafer" / "Transforms" / "CMakeLists.txt"
    transforms_cmake = transforms_cmake_path.read_text(encoding="utf-8")
    if re.search(
        r"target_link_libraries\(\s*WaferTransforms\s+PUBLIC\s+StablehloOps",
        transforms_cmake,
    ):
        raise RuntimeError(
            "WaferTransforms must not expose StablehloOps as a PUBLIC dependency"
        )
    if "target_link_libraries(WaferTransforms PRIVATE StablehloOps)" not in transforms_cmake:
        raise RuntimeError("WaferTransforms must keep StablehloOps as a PRIVATE dependency")

    for cmake_path in [
        REPO_ROOT / "lib" / "Wafer" / "IR" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "Conversion" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "ABI" / "CMakeLists.txt",
    ]:
        text = cmake_path.read_text(encoding="utf-8")
        for needle in [
            "Stablehlo",
            "Shardy",
            "GTest",
            "FileCheck",
            "tx_runtime",
            "libhpgr",
            "Tsm",
        ]:
            if needle in text:
                raise RuntimeError(f"{rel(cmake_path)} leaks {needle!r}")

    check_text_contains(
        REPO_ROOT / "tools" / "wafer-opt" / "CMakeLists.txt",
        "target_link_libraries(wafer-opt PRIVATE StablehloRegister)",
    )
    check_text_contains(
        REPO_ROOT / "tools" / "wafer-import-model" / "CMakeLists.txt",
        "StablehloRegister",
    )


def check_dependency_layering() -> None:
    production_roots = [
        REPO_ROOT / "include" / "Wafer",
        REPO_ROOT / "lib" / "Wafer",
        REPO_ROOT / "tools" / "wafer-opt",
        REPO_ROOT / "tools" / "wafer-import-model",
    ]
    compiler_library_roots = [
        REPO_ROOT / "include" / "Wafer",
        REPO_ROOT / "lib" / "Wafer",
    ]
    check_forbidden_needles(
        label="StableHLO/Shardy",
        roots=production_roots,
        needles=STABLEHLO_API_NEEDLES,
        allowed_prefixes=STABLEHLO_API_ALLOWED_PREFIXES,
    )
    check_forbidden_needles(
        label="runtime/driver",
        roots=compiler_library_roots,
        needles=RUNTIME_DRIVER_NEEDLES,
    )
    check_forbidden_needles(
        label="test tooling",
        roots=compiler_library_roots,
        needles=TEST_TOOLING_NEEDLES,
    )
    check_cmake_target_visibility()


def print_versions(versions: dict[str, str]) -> None:
    print(f"LLVM/MLIR {versions['WAFER_LLVM_VERSION']}")
    print(f"StableHLO {versions['WAFER_STABLEHLO_TAG']} {versions['WAFER_STABLEHLO_COMMIT']}")
    print(f"Shardy {versions['WAFER_SHARDY_COMMIT']}")
    print(f"googletest {versions['WAFER_GOOGLETEST_TAG']}")
    print(f"lit {versions['WAFER_PYTHON_LIT_VERSION']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions-only", action="store_true")
    args = parser.parse_args()

    versions = load_versions()
    missing = [key for key in REQUIRED_KEYS if key not in versions]
    if missing:
      raise RuntimeError(f"missing dependency pin(s): {', '.join(missing)}")

    print_versions(versions)
    if args.versions_only:
        return 0

    check_text_contains(REPO_ROOT / "CMakeLists.txt", "WAFER_ENABLE_IMPORTER_DEPS")
    check_text_contains(REPO_ROOT / "tools" / "wafer-opt" / "wafer-opt.cpp",
                        "registerImporterDialects")
    check_dependency_layering()

    stablehlo_head = rev_parse(REPO_ROOT / ".deps" / "src" / "stablehlo")
    if stablehlo_head and stablehlo_head != versions["WAFER_STABLEHLO_COMMIT"]:
        raise RuntimeError(f"StableHLO checkout mismatch: {stablehlo_head}")

    shardy_head = rev_parse(REPO_ROOT / ".deps" / "src" / "shardy")
    if shardy_head and shardy_head != versions["WAFER_SHARDY_COMMIT"]:
        raise RuntimeError(f"Shardy checkout mismatch: {shardy_head}")

    print("dependency layering checks passed")
    print("dependency consistency checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
