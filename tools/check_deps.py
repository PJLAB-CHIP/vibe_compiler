#!/usr/bin/env python3
"""Check Wafer dependency pins and importer/backend isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable

from numeric_deps import (
    ValidatedNumericRecord,
    numeric_pins,
    reject_symlink_ancestors,
    sha256_file,
    validate_numeric_record_snapshot,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSIONS_FILE = REPO_ROOT / "cmake" / "third_party" / "WaferDependencyVersions.cmake"
DEPS_ROOT = REPO_ROOT / "third_party"


REQUIRED_KEYS = [
    "WAFER_LLVM_VERSION",
    "WAFER_LLVM_PACKAGE_VERSION",
    "WAFER_LLVM_COMMIT",
    "WAFER_STABLEHLO_TAG",
    "WAFER_STABLEHLO_COMMIT",
    "WAFER_SHARDY_COMMIT",
    "WAFER_OPENXLA_XLA_SHARDY_BASE_COMMIT",
    "WAFER_OPENXLA_XLA_COMMIT",
    "WAFER_GOOGLETEST_TAG",
    "WAFER_GOOGLETEST_COMMIT",
    "WAFER_PYTORCH_VERSION",
    "WAFER_TORCHVISION_VERSION",
    "WAFER_TORCH_XLA_PYTHON_VERSION",
    "WAFER_IMPORTER_PYTHON_MAJOR_MINOR",
    "WAFER_PYTORCH_XLA_COMMIT",
    "WAFER_BAZEL_VERSION",
    "WAFER_BAZEL_LINUX_X64_URL",
    "WAFER_BAZEL_LINUX_X64_SHA256",
    "WAFER_PYTHON_LIT_VERSION",
    "WAFER_MINIMALLOC_COMMIT",
    "WAFER_MINIMALLOC_REPOSITORY",
    "WAFER_EGG_VERSION",
    "WAFER_EGG_COMMIT",
    "WAFER_EGG_REPOSITORY",
    "WAFER_SOFTFLOAT_VERSION",
    "WAFER_SOFTFLOAT_URL",
    "WAFER_SOFTFLOAT_SHA256",
    "WAFER_TESTFLOAT_VERSION",
    "WAFER_TESTFLOAT_URL",
    "WAFER_TESTFLOAT_SHA256",
    "WAFER_M4_VERSION",
    "WAFER_M4_URL",
    "WAFER_M4_SHA256",
    "WAFER_GMP_VERSION",
    "WAFER_GMP_URL",
    "WAFER_GMP_SHA256",
    "WAFER_MPFR_VERSION",
    "WAFER_MPFR_URL",
    "WAFER_MPFR_SHA256",
]

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".td"}

STABLEHLO_API_NEEDLES = [
    "stablehlo/",
    "mlir::stablehlo",
]

STABLEHLO_API_ALLOWED_PREFIXES = [
    "include/Wafer/Frontend",
    "include/Wafer/Conversion/StableHLOToLinalg",
    "lib/Wafer/Frontend",
    "lib/Wafer/Driver",
    "lib/Wafer/Conversion/StableHLOToLinalg",
    "tools/wafer-opt",
    "tools/wafer-verify-program",
]

SHARDY_API_NEEDLES = [
    "shardy/",
    "mlir::sdy",
]

SHARDY_API_ALLOWED_PREFIXES = [
    "include/Wafer/Frontend",
    "include/Wafer/Transforms/SpmdPipelines.h",
    "lib/Wafer/Driver",
    "lib/Wafer/Transforms/SPMD",
    "tools/wafer-opt",
    "tools/wafer-verify-program",
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

NUMERIC_MODEL_API_NEEDLES = [
    "softfloat.h",
    "mpfr.h",
    "gmp.h",
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


def git_is_ancestor(path: pathlib.Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "merge-base", "--is-ancestor", ancestor, descendant],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def existing_checkout(relative: pathlib.Path) -> list[pathlib.Path]:
    candidates = [DEPS_ROOT / relative]
    return [path for path in candidates if (path / ".git").exists()]


def check_text_contains(path: pathlib.Path, needle: str) -> None:
    if needle not in path.read_text(encoding="utf-8"):
        raise RuntimeError(f"{path} does not contain required text: {needle}")


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_bzl_string_constant(path: pathlib.Path, name: str) -> str:
    text = read_text(path)
    match = re.search(rf'{re.escape(name)}\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise RuntimeError(f"{path} does not define {name}")
    return match.group(1)


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
    check_text_contains(
        REPO_ROOT / "CMakeLists.txt",
        "include(cmake/third_party/WaferThirdParty.cmake)",
    )
    check_text_contains(
        REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
        'set(WAFER_DEPS_ROOT "${CMAKE_SOURCE_DIR}/third_party"',
    )
    check_text_contains(
        REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
        "WAFER_ENABLE_IMPORTER_DEPS",
    )
    for needle in [
        "WAFER_LLVM_SOURCE_DIR",
        "WAFER_LLVM_INSTALL_DIR",
        "WAFER_LLVM_BUILD_DIR",
        "WAFER_MINIMALLOC_SOURCE_DIR",
        "WAFER_EGG_SOURCE_DIR",
        "WAFER_RUST_VENDOR_DIR",
        "WAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS",
        "WAFER_ENABLE_SPMD_PARTITIONER_DEPS",
        '${WAFER_DEPS_ROOT}/llvm-project',
        '${WAFER_DEPS_ROOT}/stablehlo',
        '${WAFER_DEPS_ROOT}/shardy',
        '${WAFER_DEPS_ROOT}/xla',
        '${WAFER_DEPS_ROOT}/pytorch-xla',
        "WAFER_IMPORTER_PYTHON_VENV",
        '${WAFER_DEPS_ROOT}/googletest',
        "WAFER_ENABLE_RUNTIME_DEPS",
        "WAFER_TX_RUNTIME_ROOT",
        "WAFER_KMD_UAPI_ROOT",
        "WAFER_ENABLE_NUMERIC_MODEL_DEPS",
        "WAFER_NUMERIC_MODEL_DEPS_ROOT",
        "WAFER_NUMERIC_MODEL_DEPS_RECORD",
    ]:
        check_text_contains(
            REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
            needle,
        )
    check_text_contains(
        REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
        "add_subdirectory(\"${WAFER_STABLEHLO_SOURCE_DIR}\"",
    )
    check_text_contains(
        REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
        "WaferShardyCMake.cmake",
    )
    check_text_contains(
        REPO_ROOT / "CMakeLists.txt",
        "add_dependencies(check-wafer wafer-shardy-cmake-gate)",
    )
    check_text_contains(REPO_ROOT / "CMakeLists.txt", "wafer_add_minimalloc()")
    check_text_contains(
        REPO_ROOT / "CMakeLists.txt", "wafer_add_structured_egraph()"
    )
    for needle in [
        "function(wafer_add_minimalloc)",
        "WaferThirdPartyMiniMalloc",
        'add_subdirectory(\n    "${WAFER_MINIMALLOC_SOURCE_DIR}"',
    ]:
        check_text_contains(
            REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
            needle,
        )
    for needle in [
        "function(wafer_add_structured_egraph)",
        "WaferThirdPartyStructuredEGraph",
        "--release --locked --offline",
    ]:
        check_text_contains(
            REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
            needle,
        )
    shardy_cmake_path = REPO_ROOT / "cmake" / "third_party" / "WaferShardyCMake.cmake"
    for needle in [
        "llvm_update_compile_flags(${target})",
        "add_library(ShardySdyDialect STATIC",
        "add_library(ShardySdyImportPasses STATIC",
        "add_library(ShardySdyExportPasses STATIC",
        "add_library(ShardySdyPropagationPasses STATIC",
        "add_executable(shardy-sdy-opt",
        "add_custom_target(wafer-shardy-cmake-gate",
    ]:
        check_text_contains(shardy_cmake_path, needle)
    check_text_contains(
        REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
        "function(wafer_require_gtest)",
    )
    numeric_cmake_path = (
        REPO_ROOT / "cmake" / "third_party" / "WaferNumericModelDeps.cmake"
    )
    for needle in [
        "tools/check_deps.py",
        "--numeric-only",
        "--emit-numeric-cmake-snapshot",
        "WaferNumeric::SoftFloat",
        "WaferNumeric::GMP",
        "WaferNumeric::MPFR",
        "WaferNumeric::TestFloat",
    ]:
        check_text_contains(numeric_cmake_path, needle)
    numeric_cmake = numeric_cmake_path.read_text(encoding="utf-8")
    for forbidden in ["find_library", "find_path", "FetchContent", "ExternalProject"]:
        if forbidden in numeric_cmake:
            raise RuntimeError(
                "managed numeric-model CMake must not discover host packages or "
                f"fetch sources via {forbidden}"
            )
    if 'file(READ "${WAFER_NUMERIC_MODEL_DEPS_RECORD}"' in numeric_cmake:
        raise RuntimeError(
            "managed numeric-model CMake must consume the validator canonical "
            "snapshot instead of reparsing the mutable record"
        )
    for submodule_path in [
        "third_party/llvm-project",
        "third_party/stablehlo",
        "third_party/shardy",
        "third_party/xla",
        "third_party/googletest",
        "third_party/pytorch-xla",
        "third_party/egg",
    ]:
        check_text_contains(REPO_ROOT / ".gitmodules", submodule_path)
    for requirement in [
        "torch==2.5.0",
        "torchvision==0.20.0",
        "absl-py==2.1.0",
        "pyyaml==6.0.1",
        "requests==2.32.3",
        "wheel==0.44.0",
    ]:
        check_text_contains(REPO_ROOT / "requirements-importer.txt", requirement)
    if "torch_xla" in (REPO_ROOT / "requirements-importer.txt").read_text(encoding="utf-8"):
        raise RuntimeError(
            "requirements-importer.txt must not install torch_xla from a wheel; "
            "build/install third_party/pytorch-xla from source"
        )

    transforms_cmake_path = REPO_ROOT / "lib" / "Wafer" / "Transforms" / "CMakeLists.txt"
    transforms_cmake = transforms_cmake_path.read_text(encoding="utf-8")
    if re.search(
        r"target_link_libraries\(\s*WaferTransforms\s+PUBLIC\s+StablehloOps",
        transforms_cmake,
    ):
        raise RuntimeError(
            "WaferTransforms must not expose StablehloOps as a PUBLIC dependency"
        )
    if "StablehloOps" in transforms_cmake:
        raise RuntimeError("WaferTransforms must not link StablehloOps")
    for needle in [
        "target_compile_definitions(WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)",
        "ShardySdyDialect ShardySdyTransforms",
    ]:
        check_text_contains(transforms_cmake_path, needle)

    conversion_cmake_path = REPO_ROOT / "lib" / "Wafer" / "Conversion" / "CMakeLists.txt"
    conversion_cmake = conversion_cmake_path.read_text(encoding="utf-8")
    if re.search(
        r"target_link_libraries\(\s*WaferStableHLOToLinalg\s+PUBLIC\s+StablehloOps",
        conversion_cmake,
    ):
        raise RuntimeError(
            "WaferStableHLOToLinalg must not expose StablehloOps as a PUBLIC dependency"
        )
    if (
        "target_link_libraries(WaferStableHLOToLinalg PRIVATE StablehloOps)"
        not in conversion_cmake
    ):
        raise RuntimeError(
            "WaferStableHLOToLinalg must keep StablehloOps as a PRIVATE dependency"
        )

    cmake_paths = [
        REPO_ROOT / "lib" / "Wafer" / "Analysis" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "IR" / "CMakeLists.txt",
    ]
    for cmake_path in cmake_paths:
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

    for cmake_path in [
        REPO_ROOT / "lib" / "Wafer" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "Conversion" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "Transforms" / "CMakeLists.txt",
        REPO_ROOT / "tools" / "wafer-opt" / "CMakeLists.txt",
    ]:
        text = cmake_path.read_text(encoding="utf-8")
        if "WaferConversion" in text:
            raise RuntimeError(f"{rel(cmake_path)} still links removed WaferConversion")

    wafer_opt_source_path = REPO_ROOT / "tools" / "wafer-opt" / "wafer-opt.cpp"
    for needle in [
        "mlir::MlirOptMain",
        "mlir::stablehlo::registerAllDialects(registry)",
        "mlir::sdy::registerAllDialects(registry)",
        "mlir::sdy::registerAllSdyPassesAndPipelines()",
    ]:
        check_text_contains(wafer_opt_source_path, needle)
    wafer_opt_source = wafer_opt_source_path.read_text(encoding="utf-8")
    for needle in [
        "Wafer/Frontend",
        "--program-pipeline",
        "--input-program-dir",
        "--output-dir",
        "WAFER_XLA_SPMD_PARTITIONER_HELPER",
        "llvm/Support/FileSystem.h",
        "llvm/Support/Program.h",
        "ExecuteAndWait",
    ]:
        if needle in wafer_opt_source:
            raise RuntimeError(
                f"tools/wafer-opt/wafer-opt.cpp owns program orchestration via {needle!r}"
            )

    wafer_opt_cmake_path = REPO_ROOT / "tools" / "wafer-opt" / "CMakeLists.txt"
    for needle in [
        "MLIRMlirOptMain",
        "target_link_libraries(wafer-opt PRIVATE StablehloRegister)",
        "WAFER_ENABLE_SPMD_PARTITIONER_DEPS",
        "WAFER_ENABLE_SHARDY=1",
        "target_link_libraries(wafer-opt PRIVATE ShardySdyRegister ShardySdyTransforms)",
    ]:
        check_text_contains(wafer_opt_cmake_path, needle)
    wafer_opt_cmake = wafer_opt_cmake_path.read_text(encoding="utf-8")
    for needle in [
        "WaferFrontend",
        "MLIRParser",
        "MLIRPass",
        "WAFER_XLA_SPMD_PARTITIONER_HELPER",
    ]:
        if needle in wafer_opt_cmake:
            raise RuntimeError(
                f"tools/wafer-opt/CMakeLists.txt owns program orchestration via {needle!r}"
            )

    compiler_cmake_path = REPO_ROOT / "lib" / "Wafer" / "CMakeLists.txt"
    for needle in [
        "WaferCompiler",
        "WaferFrontend",
        "WaferStableHLOPipelines",
        "WaferTileRegionPipelines",
        "WaferTransforms",
        "WAFER_ENABLE_IMPORTER_DEPS",
        "StablehloRegister",
        "WAFER_ENABLE_SPMD_PARTITIONER_DEPS",
        "ShardySdyRegister",
    ]:
        check_text_contains(compiler_cmake_path, needle)
    compiler_cmake = compiler_cmake_path.read_text(encoding="utf-8")
    for needle in [*RUNTIME_DRIVER_NEEDLES, *TEST_TOOLING_NEEDLES]:
        if needle in compiler_cmake:
            raise RuntimeError(
                f"lib/Wafer/CMakeLists.txt compiler target leaks {needle!r}"
            )

    compiler_tool_cmake_path = (
        REPO_ROOT / "tools" / "wafer-compile" / "CMakeLists.txt"
    )
    for needle in ["WaferCompiler", "LLVMSupport"]:
        check_text_contains(compiler_tool_cmake_path, needle)
    compiler_tool_cmake = compiler_tool_cmake_path.read_text(encoding="utf-8")
    for needle in [
        "WaferFrontend",
        "WaferStableHLOPipelines",
        "WaferTileRegionPipelines",
        "WaferTransforms",
        "Stablehlo",
        "Shardy",
        *RUNTIME_DRIVER_NEEDLES,
        *TEST_TOOLING_NEEDLES,
    ]:
        if needle in compiler_tool_cmake:
            raise RuntimeError(
                "wafer-compile must delegate compiler implementation ownership "
                f"to WaferCompiler, but directly links {needle!r}"
            )

    check_text_contains(
        REPO_ROOT / "tools" / "wafer-verify-program" / "CMakeLists.txt",
        "StablehloRegister",
    )
    stablehlo_tool_cmake = (
        REPO_ROOT / "tools" / "wafer-verify-program" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    for needle in [
        "WaferCompiler",
        "WaferStableHLOPipelines",
        "WaferTileRegionPipelines",
        "WaferTransforms",
    ]:
        if needle in stablehlo_tool_cmake:
            raise RuntimeError(
                "wafer-verify-program is a frontend verifier tool and must not "
                f"link {needle}"
            )
    stablehlo_tool_source = (
        REPO_ROOT
        / "tools"
        / "wafer-verify-program"
        / "wafer-verify-program.cpp"
    ).read_text(encoding="utf-8")
    for needle in [
        "Wafer/Driver",
        "Wafer/Conversion/StableHLOToLinalg/Pipelines.h",
        "Wafer/Transforms",
    ]:
        if needle in stablehlo_tool_source:
            raise RuntimeError(
                "wafer-verify-program must remain a frontend verifier, but "
                f"its source includes {needle!r}"
            )
    pipelines_cmake_path = REPO_ROOT / "lib" / "Wafer" / "Transforms" / "CMakeLists.txt"
    for needle in [
        "target_compile_definitions(obj.WaferTransforms PRIVATE WAFER_ENABLE_SHARDY=1)",
        "ShardySdyDialect ShardySdyTransforms",
    ]:
        check_text_contains(pipelines_cmake_path, needle)
    for needle in [
        "shardy/dialect/sdy/ir/register.h",
        "mlir::sdy::registerAllDialects(registry)",
    ]:
        check_text_contains(
            REPO_ROOT / "include" / "Wafer" / "Frontend" / "InitImporterDialects.h",
            needle,
        )
    check_text_contains(REPO_ROOT / "test" / "lit.cfg.py", 'add("shardy")')
    check_text_contains(
        REPO_ROOT / "test" / "lit.site.cfg.py.in",
        "config.wafer_enable_spmd_partitioner_deps",
    )


def check_dependency_layering() -> None:
    production_roots = [
        REPO_ROOT / "include" / "Wafer",
        REPO_ROOT / "lib" / "Wafer",
        REPO_ROOT / "tools" / "wafer-compile",
        REPO_ROOT / "tools" / "wafer-opt",
        REPO_ROOT / "tools" / "wafer-verify-program",
    ]
    compiler_library_roots = [
        REPO_ROOT / "include" / "Wafer",
        REPO_ROOT / "lib" / "Wafer",
        REPO_ROOT / "tools" / "wafer-compile",
    ]
    check_forbidden_needles(
        label="StableHLO",
        roots=production_roots,
        needles=STABLEHLO_API_NEEDLES,
        allowed_prefixes=STABLEHLO_API_ALLOWED_PREFIXES,
    )
    check_forbidden_needles(
        label="Shardy",
        roots=production_roots,
        needles=SHARDY_API_NEEDLES,
        allowed_prefixes=SHARDY_API_ALLOWED_PREFIXES,
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
    check_forbidden_needles(
        label="numeric-model",
        roots=compiler_library_roots,
        needles=NUMERIC_MODEL_API_NEEDLES,
        allowed_prefixes=["include/Wafer/Target", "lib/Wafer/Target"],
    )
    for cmake_path in [
        REPO_ROOT / "lib" / "Wafer" / "Analysis" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "IR" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "Conversion" / "CMakeLists.txt",
        REPO_ROOT / "lib" / "Wafer" / "Transforms" / "CMakeLists.txt",
        REPO_ROOT / "tools" / "wafer-compile" / "CMakeLists.txt",
        REPO_ROOT / "tools" / "wafer-opt" / "CMakeLists.txt",
    ]:
        text = cmake_path.read_text(encoding="utf-8")
        for needle in ["WaferNumeric::", "SoftFloat", "TestFloat", "MPFR", "GMP"]:
            if needle in text:
                raise RuntimeError(
                    f"base compiler target {rel(cmake_path)} leaks managed "
                    f"numeric-model dependency {needle!r}"
                )
    check_cmake_target_visibility()


def check_minimalloc_snapshot(versions: dict[str, str]) -> None:
    root = DEPS_ROOT / "minimalloc"
    expected_files = {
        "CMakeLists.txt",
        "LICENSE",
        "PROVENANCE.json",
        "README.wafer.md",
        "include/wafer_third_party/minimalloc/minimalloc.h",
        "src/minimalloc.cc",
        "src/minimalloc_internal.h",
        "src/solver.cc",
        "src/sweeper.cc",
        "src/sweeper.h",
        "tests/minimalloc_test.cc",
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise RuntimeError(
            "curated MiniMalloc source closure changed: "
            f"missing={missing}, unexpected={unexpected}"
        )

    readme = read_text(root / "README.wafer.md")
    commit = versions["WAFER_MINIMALLOC_COMMIT"]
    repository = versions["WAFER_MINIMALLOC_REPOSITORY"]
    if commit not in readme or repository.removesuffix(".git") not in readme:
        raise RuntimeError(
            "curated MiniMalloc provenance does not match dependency pins"
        )
    license_text = read_text(root / "LICENSE")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("curated MiniMalloc Apache-2.0 license is missing")

    provenance_path = root / "PROVENANCE.json"
    try:
        provenance = json.loads(read_text(provenance_path))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"invalid curated MiniMalloc provenance manifest: {error}"
        ) from error
    if set(provenance) != {
        "upstream",
        "curated",
        "semantic_deltas",
    }:
        raise RuntimeError("curated MiniMalloc provenance fields changed")

    upstream = provenance["upstream"]
    if set(upstream) != {
        "repository",
        "commit",
        "git_tree",
        "retained_files",
        "verification_files",
    }:
        raise RuntimeError("curated MiniMalloc upstream provenance fields changed")
    if (
        upstream["repository"] != repository
        or upstream["commit"] != commit
        or upstream["git_tree"] != "221e93c24ae6c5b76458932fd3928bae71a4dffc"
    ):
        raise RuntimeError("curated MiniMalloc upstream identity changed")

    expected_upstream_hashes = {
        "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "src/minimalloc.cc": "451c83f22f5e87ef4221c203a13bc3ba23ea31c7eef6f6625f23f2a5749a6f60",
        "src/minimalloc.h": "d888510cd97294bcde359fd7702ea3edd93291e2b5ae02d6db156db62ff17bc0",
        "src/solver.cc": "a693ed4b02bc5542c7a03a8d8de76395abeee51a62183b2b7b2c587b653a9a1c",
        "src/solver.h": "67b608c5d019978dc9fcd781adfc6c508a74bb146ccaae00479fa077d5658da7",
        "src/sweeper.cc": "24c026e90371ccb9c01faf647bf166b28eedb1bad1628f17a5cfdad52d63d0cd",
        "src/sweeper.h": "e786ef19007a267ed897e695cc2d979085c43a0c6274cf57d04521eb980c2cc7",
    }
    expected_upstream_mappings = {
        "LICENSE": ["LICENSE"],
        "src/minimalloc.cc": ["src/minimalloc.cc"],
        "src/minimalloc.h": [
            "include/wafer_third_party/minimalloc/minimalloc.h",
            "src/minimalloc_internal.h",
        ],
        "src/solver.cc": ["src/solver.cc"],
        "src/solver.h": [
            "include/wafer_third_party/minimalloc/minimalloc.h",
            "src/solver.cc",
        ],
        "src/sweeper.cc": ["src/sweeper.cc"],
        "src/sweeper.h": ["src/sweeper.h"],
    }
    retained = upstream["retained_files"]
    if not isinstance(retained, list) or {
        entry.get("path"): entry.get("sha256") for entry in retained
    } != expected_upstream_hashes:
        raise RuntimeError("curated MiniMalloc retained upstream source map changed")
    for entry in retained:
        if set(entry) != {"path", "sha256", "maps_to", "role"}:
            raise RuntimeError(
                "curated MiniMalloc retained source-map fields changed"
            )
        if not isinstance(entry["maps_to"], list) or not entry["maps_to"]:
            raise RuntimeError(
                "curated MiniMalloc retained source mapping is incomplete"
            )
        if entry["maps_to"] != expected_upstream_mappings[entry["path"]]:
            raise RuntimeError(
                "curated MiniMalloc retained source mapping changed for "
                f"{entry['path']}"
            )
        if not isinstance(entry["role"], str) or not entry["role"]:
            raise RuntimeError(
                "curated MiniMalloc retained source-map role is missing"
            )

    expected_verification_hashes = {
        "tests/minimalloc_test.cc": "9aae27768fb894a36ab056de1f6a0364aa11563ee79839c6de1c162c2db3b3b8",
        "tests/solver_test.cc": "a51fac06d32c367d9a640682a29828f96a2cffdb9e376a9c1ee83b238c54abf5",
        "tests/sweeper_test.cc": "ad182d0e0c49a94808a1b0b69a01286ea678d8240c8dd613efcc2d3dfa6d80ac",
    }
    verification = upstream["verification_files"]
    if not isinstance(verification, list) or any(
        set(entry) != {"path", "sha256"} for entry in verification
    ) or {
        entry.get("path"): entry.get("sha256") for entry in verification
    } != expected_verification_hashes:
        raise RuntimeError(
            "curated MiniMalloc upstream verification source map changed"
        )

    curated = provenance["curated"]
    expected_algorithm_files = sorted(
        {
            "include/wafer_third_party/minimalloc/minimalloc.h",
            "src/minimalloc.cc",
            "src/minimalloc_internal.h",
            "src/solver.cc",
            "src/sweeper.cc",
            "src/sweeper.h",
        }
    )
    expected_distribution_files = sorted(
        expected_files - {"PROVENANCE.json"}
    )
    digest_algorithm = (
        "sha256 over each UTF-8 path, NUL, decimal byte length, NUL, raw "
        "bytes, NUL; paths sorted lexicographically"
    )
    if set(curated) != {
        "digest_algorithm",
        "algorithm_files",
        "algorithm_digest",
        "distribution_files",
        "distribution_digest",
    } or curated["digest_algorithm"] != digest_algorithm:
        raise RuntimeError("curated MiniMalloc digest contract changed")
    if curated["algorithm_files"] != expected_algorithm_files:
        raise RuntimeError("curated MiniMalloc algorithm source set changed")
    if curated["distribution_files"] != expected_distribution_files:
        raise RuntimeError("curated MiniMalloc distribution source set changed")

    mapped_algorithm_files = {
        target
        for entry in retained
        if entry["path"] != "LICENSE"
        for target in entry["maps_to"]
    }
    if mapped_algorithm_files != set(expected_algorithm_files):
        raise RuntimeError(
            "curated MiniMalloc upstream mapping does not cover the algorithm set"
        )

    def canonical_digest(relative_paths: list[str]) -> str:
        digest = hashlib.sha256()
        for relative in sorted(relative_paths):
            data = (root / relative).read_bytes()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(len(data)).encode("ascii"))
            digest.update(b"\0")
            digest.update(data)
            digest.update(b"\0")
        return "sha256:" + digest.hexdigest()

    if curated["algorithm_digest"] != canonical_digest(
        expected_algorithm_files
    ):
        raise RuntimeError("curated MiniMalloc algorithm digest mismatch")
    if curated["distribution_digest"] != canonical_digest(
        expected_distribution_files
    ):
        raise RuntimeError("curated MiniMalloc distribution digest mismatch")
    if sha256_file(root / "LICENSE") != expected_upstream_hashes["LICENSE"]:
        raise RuntimeError("curated MiniMalloc license is not upstream-exact")

    semantic_deltas = provenance["semantic_deltas"]
    if not isinstance(semantic_deltas, list) or not semantic_deltas or any(
        set(entry) != {"id", "description"}
        or not entry["id"]
        or not entry["description"]
        for entry in semantic_deltas
    ):
        raise RuntimeError("curated MiniMalloc semantic delta record is invalid")

    source_paths = [
        root / relative
        for relative in expected_files
        if pathlib.Path(relative).suffix in {".cc", ".h"}
    ]
    for path in source_paths:
        text = read_text(path)
        if "Copyright 2023 Google LLC" not in text:
            raise RuntimeError(f"{rel(path)} lost the upstream copyright notice")
        if "Modified for Wafer" not in text:
            raise RuntimeError(f"{rel(path)} lacks a prominent modification notice")
        for forbidden in ["absl/", "absl::", ".contains(", "std::span"]:
            if forbidden in text:
                raise RuntimeError(
                    f"{rel(path)} reintroduced unsupported dependency {forbidden!r}"
                )

    cmake = read_text(root / "CMakeLists.txt")
    for required in [
        "add_library(WaferThirdPartyMiniMalloc STATIC",
        "CXX_VISIBILITY_PRESET hidden",
        "-fno-exceptions",
        "cxx_std_17",
    ]:
        if required not in cmake:
            raise RuntimeError(
                f"curated MiniMalloc CMake lacks required boundary {required!r}"
            )
    for forbidden in ["FetchContent", "abseil", "pybind", "src/main.cc"]:
        if forbidden in cmake:
            raise RuntimeError(
                f"curated MiniMalloc CMake must remain self-contained: {forbidden!r}"
            )


def check_openxla_stack_pins(versions: dict[str, str]) -> None:
    xla_root = DEPS_ROOT / "xla"
    shardy_root = DEPS_ROOT / "shardy"
    if not (xla_root / ".git").exists():
        return

    xla_llvm = parse_bzl_string_constant(
        xla_root / "third_party" / "llvm" / "workspace.bzl", "LLVM_COMMIT"
    )
    xla_stablehlo = parse_bzl_string_constant(
        xla_root / "third_party" / "stablehlo" / "workspace.bzl", "STABLEHLO_COMMIT"
    )
    xla_shardy = parse_bzl_string_constant(
        xla_root / "third_party" / "shardy" / "workspace.bzl", "SHARDY_COMMIT"
    )

    expected = {
        "WAFER_LLVM_COMMIT": xla_llvm,
        "WAFER_STABLEHLO_COMMIT": xla_stablehlo,
        "WAFER_OPENXLA_XLA_SHARDY_BASE_COMMIT": xla_shardy,
    }
    mismatches = [
        f"{key}={versions[key]} but OpenXLA/XLA pins {actual}"
        for key, actual in expected.items()
        if versions[key] != actual
    ]
    if mismatches:
        raise RuntimeError("OpenXLA stack pin mismatch: " + "; ".join(mismatches))

    if (shardy_root / ".git").exists():
        shardy_llvm = parse_bzl_string_constant(
            shardy_root / "third_party" / "llvm" / "workspace.bzl", "LLVM_COMMIT"
        )
        shardy_stablehlo = parse_bzl_string_constant(
            shardy_root / "third_party" / "stablehlo" / "workspace.bzl",
            "STABLEHLO_COMMIT",
        )
        if shardy_llvm != versions["WAFER_LLVM_COMMIT"]:
            raise RuntimeError(
                f"Shardy LLVM pin mismatch: {shardy_llvm} != {versions['WAFER_LLVM_COMMIT']}"
            )
        if shardy_stablehlo != versions["WAFER_STABLEHLO_COMMIT"]:
            raise RuntimeError(
                "Shardy StableHLO pin mismatch: "
                f"{shardy_stablehlo} != {versions['WAFER_STABLEHLO_COMMIT']}"
            )
        if not git_is_ancestor(
            shardy_root,
            versions["WAFER_OPENXLA_XLA_SHARDY_BASE_COMMIT"],
            versions["WAFER_SHARDY_COMMIT"],
        ):
            raise RuntimeError(
                "Shardy pin must contain the OpenXLA/XLA Shardy base pin: "
                f"{versions['WAFER_OPENXLA_XLA_SHARDY_BASE_COMMIT']} is not an ancestor "
                f"of {versions['WAFER_SHARDY_COMMIT']}"
            )

        xla_stablehlo_patch = xla_root / "third_party" / "stablehlo" / "temporary.patch"
        shardy_stablehlo_patch = (
            shardy_root / "third_party" / "stablehlo" / "temporary.patch"
        )
        if xla_stablehlo_patch.exists() and shardy_stablehlo_patch.exists():
            if xla_stablehlo_patch.read_bytes() != shardy_stablehlo_patch.read_bytes():
                raise RuntimeError(
                    "OpenXLA/XLA and Shardy must use the same StableHLO temporary.patch"
                )


def check_framework_source_alignment(versions: dict[str, str]) -> None:
    pytorch_xla_root = DEPS_ROOT / "pytorch-xla"
    if not (pytorch_xla_root / ".git").exists():
        return

    workspace_path = pytorch_xla_root / "WORKSPACE"
    if not workspace_path.exists():
        raise RuntimeError("third_party/pytorch-xla exists but has no WORKSPACE")

    pytorch_xla_xla = parse_bzl_string_constant(workspace_path, "xla_hash")
    expected_xla = versions["WAFER_OPENXLA_XLA_COMMIT"]
    if pytorch_xla_xla != expected_xla:
        raise RuntimeError(
            "PyTorch/XLA source checkout is not aligned with the Wafer OpenXLA stack: "
            f"WORKSPACE xla_hash={pytorch_xla_xla}, "
            f"WAFER_OPENXLA_XLA_COMMIT={expected_xla}. "
            "Pin PyTorch/XLA to a commit with the same xla_hash, or build it only "
            "through a checked wrapper that passes --override_repository=xla=third_party/xla."
        )


def check_numeric_sources_if_present(
    versions: dict[str, str], numeric_root: pathlib.Path
) -> None:
    numeric_root = reject_symlink_ancestors(numeric_root, allow_missing=True)
    if not numeric_root.exists():
        return
    if not numeric_root.is_dir():
        raise RuntimeError(f"numeric dependency root is not a directory: {numeric_root}")
    pins = numeric_pins(versions)
    downloads = numeric_root / "downloads"
    reject_symlink_ancestors(downloads, allow_missing=True)
    if not downloads.exists():
        return
    if not downloads.is_dir():
        raise RuntimeError(f"numeric download root is not a directory: {downloads}")
    for name, pin in pins.items():
        archive = downloads / pin.archive_name
        if not archive.exists():
            continue
        if not archive.is_file() or archive.is_symlink():
            raise RuntimeError(f"managed numeric archive is not a regular file: {archive}")
        actual = sha256_file(archive)
        if actual != pin.sha256:
            raise RuntimeError(
                f"managed numeric archive mismatch for {name}: "
                f"expected {pin.sha256}, got {actual}"
            )


def check_numeric_record(
    versions: dict[str, str], numeric_root: pathlib.Path, record_path: pathlib.Path
) -> ValidatedNumericRecord:
    return validate_numeric_record_snapshot(record_path, numeric_root, versions)


def print_versions(versions: dict[str, str]) -> None:
    print(f"LLVM/MLIR {versions['WAFER_LLVM_PACKAGE_VERSION']} {versions['WAFER_LLVM_COMMIT']}")
    print(f"StableHLO {versions['WAFER_STABLEHLO_TAG']} {versions['WAFER_STABLEHLO_COMMIT']}")
    print(
        f"Shardy {versions['WAFER_SHARDY_COMMIT']} "
        f"(OpenXLA/XLA base {versions['WAFER_OPENXLA_XLA_SHARDY_BASE_COMMIT']})"
    )
    print(f"OpenXLA/XLA {versions['WAFER_OPENXLA_XLA_COMMIT']}")
    print(
        "PyTorch importer source-build Python packages "
        f"torch {versions['WAFER_PYTORCH_VERSION']} "
        f"torchvision {versions['WAFER_TORCHVISION_VERSION']} "
        "absl-py 2.1.0 pyyaml 6.0.1 requests 2.32.3"
    )
    print(
        f"PyTorch/XLA source runtime {versions['WAFER_TORCH_XLA_PYTHON_VERSION']} "
        f"{versions['WAFER_PYTORCH_XLA_COMMIT']}"
    )
    print(f"googletest {versions['WAFER_GOOGLETEST_TAG']} {versions['WAFER_GOOGLETEST_COMMIT']}")
    print(f"lit {versions['WAFER_PYTHON_LIT_VERSION']}")
    print(f"egg {versions['WAFER_EGG_VERSION']} {versions['WAFER_EGG_COMMIT']}")
    print(
        "numeric-model sources "
        f"SoftFloat {versions['WAFER_SOFTFLOAT_VERSION']} "
        f"TestFloat {versions['WAFER_TESTFLOAT_VERSION']} "
        f"m4 {versions['WAFER_M4_VERSION']} "
        f"GMP {versions['WAFER_GMP_VERSION']} "
        f"MPFR {versions['WAFER_MPFR_VERSION']}"
    )


def check_checkout_pin(
    *, label: str, relative: pathlib.Path, expected_commit: str
) -> None:
    for checkout in existing_checkout(relative):
        actual_commit = rev_parse(checkout)
        if actual_commit and actual_commit != expected_commit:
            raise RuntimeError(
                f"{label} checkout mismatch in {rel(checkout)}: {actual_commit}"
            )


def check_egraph_sources(versions: dict[str, str]) -> None:
    crate_root = (
        REPO_ROOT
        / "lib"
        / "Wafer"
        / "Conversion"
        / "StableHLOToLinalg"
        / "EGraphCore"
    )
    lock = crate_root / "Cargo.lock"
    record_path = DEPS_ROOT / "rust-vendor" / "wafer-egraph-deps.json"
    if not lock.is_file() or not record_path.is_file():
        raise RuntimeError(
            "structured e-graph Cargo lock/vendor record is missing; run "
            "tools/bootstrap_deps.py --egraph-sources"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "egg_version": versions["WAFER_EGG_VERSION"],
        "egg_commit": versions["WAFER_EGG_COMMIT"],
        "cargo_lock_sha256": sha256_file(lock),
    }
    mismatches = [
        f"{key}={record.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "structured e-graph vendor record mismatch: " + "; ".join(mismatches)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions-only", action="store_true")
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="validate only the required managed numeric-model record",
    )
    parser.add_argument(
        "--numeric-root",
        default=str(DEPS_ROOT / "numeric-model"),
        help="managed numeric-model dependency root",
    )
    parser.add_argument(
        "--numeric-record",
        help="numeric-model dependency record (defaults to <numeric-root>/numeric-model-deps.json)",
    )
    parser.add_argument(
        "--emit-numeric-cmake-snapshot",
        action="store_true",
        help=(
            "with --numeric-only, emit only the validated canonical JSON snapshot "
            "for CMake on stdout"
        ),
    )
    args = parser.parse_args()

    versions = load_versions()
    missing = [key for key in REQUIRED_KEYS if key not in versions]
    if missing:
        raise RuntimeError(f"missing dependency pin(s): {', '.join(missing)}")

    if args.emit_numeric_cmake_snapshot and not args.numeric_only:
        raise RuntimeError("--emit-numeric-cmake-snapshot requires --numeric-only")
    if not args.emit_numeric_cmake_snapshot:
        print_versions(versions)
    if args.versions_only:
        return 0

    numeric_root = pathlib.Path(args.numeric_root).absolute()
    numeric_record = (
        pathlib.Path(args.numeric_record).absolute()
        if args.numeric_record
        else numeric_root / "numeric-model-deps.json"
    )
    if args.numeric_only:
        if not numeric_record.is_file():
            raise RuntimeError(
                f"managed numeric-model dependency record is missing: {numeric_record}"
            )
        check_numeric_sources_if_present(versions, numeric_root)
        validated = check_numeric_record(versions, numeric_root, numeric_record)
        if args.emit_numeric_cmake_snapshot:
            print(
                json.dumps(
                    validated.cmake_snapshot(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(
                f"numeric-model dependency conformance record passed: {numeric_record}"
            )
        return 0

    check_text_contains(
        REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
        "WAFER_ENABLE_IMPORTER_DEPS",
    )
    check_minimalloc_snapshot(versions)
    check_egraph_sources(versions)
    check_dependency_layering()
    check_openxla_stack_pins(versions)
    check_framework_source_alignment(versions)
    check_numeric_sources_if_present(versions, numeric_root)
    if args.numeric_record or numeric_record.exists():
        check_numeric_record(versions, numeric_root, numeric_record)
        print(f"numeric-model dependency conformance record passed: {numeric_record}")

    check_checkout_pin(
        label="LLVM/MLIR",
        relative=pathlib.Path("llvm-project"),
        expected_commit=versions["WAFER_LLVM_COMMIT"],
    )
    check_checkout_pin(
        label="StableHLO",
        relative=pathlib.Path("stablehlo"),
        expected_commit=versions["WAFER_STABLEHLO_COMMIT"],
    )
    check_checkout_pin(
        label="Shardy",
        relative=pathlib.Path("shardy"),
        expected_commit=versions["WAFER_SHARDY_COMMIT"],
    )
    check_checkout_pin(
        label="OpenXLA/XLA",
        relative=pathlib.Path("xla"),
        expected_commit=versions["WAFER_OPENXLA_XLA_COMMIT"],
    )
    check_checkout_pin(
        label="PyTorch/XLA",
        relative=pathlib.Path("pytorch-xla"),
        expected_commit=versions["WAFER_PYTORCH_XLA_COMMIT"],
    )
    check_checkout_pin(
        label="googletest",
        relative=pathlib.Path("googletest"),
        expected_commit=versions["WAFER_GOOGLETEST_COMMIT"],
    )
    check_checkout_pin(
        label="egg",
        relative=pathlib.Path("egg"),
        expected_commit=versions["WAFER_EGG_COMMIT"],
    )

    print("dependency layering checks passed")
    print("dependency consistency checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
