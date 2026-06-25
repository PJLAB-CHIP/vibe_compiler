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
    "WAFER_PYTORCH_XLA_COMMIT",
    "WAFER_PYTHON_LIT_VERSION",
]

SOURCE_SUFFIXES = {".cpp", ".h", ".td"}

STABLEHLO_API_NEEDLES = [
    "stablehlo/",
    "mlir::stablehlo",
]

STABLEHLO_API_ALLOWED_PREFIXES = [
    "include/Wafer/Frontend",
    "lib/Wafer/Conversion/StableHLOToLinalg",
    "tools/wafer-opt",
    "tools/wafer-compile-stablehlo",
]

SHARDY_API_NEEDLES = [
    "shardy/",
    "mlir::sdy",
]

SHARDY_API_ALLOWED_PREFIXES = [
    "include/Wafer/Frontend",
    "include/Wafer/Pipelines",
    "lib/Wafer/Pipelines",
    "lib/Wafer/Transforms/SPMD",
    "tools/wafer-opt",
    "tools/wafer-compile-stablehlo",
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
        "WAFER_LEGACY_TSM_RUNTIME_ROOT",
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
    for submodule_path in [
        "third_party/llvm-project",
        "third_party/stablehlo",
        "third_party/shardy",
        "third_party/xla",
        "third_party/googletest",
        "third_party/pytorch-xla",
    ]:
        check_text_contains(REPO_ROOT / ".gitmodules", submodule_path)
    for requirement in [
        "torch==2.5.0",
        "torchvision==0.20.0",
        "absl-py==2.1.0",
        "pyyaml==6.0.1",
        "requests==2.32.3",
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
        "target_link_libraries(WaferTransforms PRIVATE ShardySdyDialect)",
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
        REPO_ROOT / "lib" / "Wafer" / "ABI" / "CMakeLists.txt",
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
        REPO_ROOT / "lib" / "Wafer" / "Pipelines" / "CMakeLists.txt",
        REPO_ROOT / "tools" / "wafer-opt" / "CMakeLists.txt",
    ]:
        text = cmake_path.read_text(encoding="utf-8")
        if "WaferConversion" in text:
            raise RuntimeError(f"{rel(cmake_path)} still links removed WaferConversion")

    check_text_contains(
        REPO_ROOT / "tools" / "wafer-opt" / "CMakeLists.txt",
        "target_link_libraries(wafer-opt PRIVATE StablehloRegister)",
    )
    for needle in [
        "WaferFrontend",
        "MLIRParser",
        "MLIRPass",
        "WAFER_ENABLE_SPMD_PARTITIONER_DEPS",
        "WAFER_ENABLE_SHARDY=1",
        "target_link_libraries(wafer-opt PRIVATE ShardySdyRegister ShardySdyTransforms)",
    ]:
        check_text_contains(REPO_ROOT / "tools" / "wafer-opt" / "CMakeLists.txt", needle)
    check_text_contains(
        REPO_ROOT / "tools" / "wafer-compile-stablehlo" / "CMakeLists.txt",
        "StablehloRegister",
    )
    stablehlo_tool_cmake = (
        REPO_ROOT / "tools" / "wafer-compile-stablehlo" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    if "WaferPipelines" in stablehlo_tool_cmake:
        raise RuntimeError(
            "wafer-compile-stablehlo is a frontend verifier tool and must not "
            "link WaferPipelines"
        )
    for needle in [
        "WAFER_ENABLE_SHARDY=1",
        "target_link_libraries(wafer-compile-stablehlo PRIVATE ShardySdyRegister)",
    ]:
        check_text_contains(
            REPO_ROOT / "tools" / "wafer-compile-stablehlo" / "CMakeLists.txt", needle
        )
    pipelines_cmake_path = REPO_ROOT / "lib" / "Wafer" / "Pipelines" / "CMakeLists.txt"
    for needle in [
        "target_compile_definitions(obj.WaferPipelines PRIVATE WAFER_ENABLE_SHARDY=1)",
        "target_link_libraries(WaferPipelines PUBLIC ShardySdyTransforms)",
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
        REPO_ROOT / "tools" / "wafer-opt",
        REPO_ROOT / "tools" / "wafer-compile-stablehlo",
    ]
    compiler_library_roots = [
        REPO_ROOT / "include" / "Wafer",
        REPO_ROOT / "lib" / "Wafer",
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
    check_cmake_target_visibility()


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


def check_checkout_pin(
    *, label: str, relative: pathlib.Path, expected_commit: str
) -> None:
    for checkout in existing_checkout(relative):
        actual_commit = rev_parse(checkout)
        if actual_commit and actual_commit != expected_commit:
            raise RuntimeError(
                f"{label} checkout mismatch in {rel(checkout)}: {actual_commit}"
            )


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

    check_text_contains(
        REPO_ROOT / "cmake" / "third_party" / "WaferThirdParty.cmake",
        "WAFER_ENABLE_IMPORTER_DEPS",
    )
    check_text_contains(REPO_ROOT / "tools" / "wafer-opt" / "wafer-opt.cpp",
                        "registerImporterDialects")
    check_dependency_layering()
    check_openxla_stack_pins(versions)
    check_framework_source_alignment(versions)

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

    print("dependency layering checks passed")
    print("dependency consistency checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
