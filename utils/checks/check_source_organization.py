#!/usr/bin/env python3
"""Check repository-wide Wafer source ownership without duplicating semantics."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


# Sources intentionally retained for a current numbered task but not compiled.
# The design and deletion gates live in tasks/18-source-organization.md; this
# list only makes the filesystem/CMake distinction fail closed.
DORMANT_LIBRARY_SOURCES = {
}

# These translation units are inputs to the explicitly invoked XLA helper
# build, not host CMake targets. utils/deps/build_xla_spmd_partitioner_helper.py
# owns their exact source manifest and copies them into the external build.
EXTERNAL_HELPER_SOURCES = {
    f"lib/Wafer/Transforms/StableHLO/SPMD/{name}.cpp"
    for name in (
        "XlaSpmdBoundary",
        "XlaSpmdDriver",
        "XlaSpmdFilesystem",
        "XlaSpmdMetadata",
        "XlaSpmdPartitionerMain",
        "XlaSpmdPartitioning",
        "XlaSpmdPayload",
        "XlaSpmdProgram",
    )
}

DORMANT_UNIT_SOURCES = {
}

RETIRED_PATHS = (
    "CLAUDE.md",
    "include/Wafer/Compiler",
    "include/Wafer/Pipelines",
    "lib/Wafer/Compiler",
    "lib/Wafer/Pipelines",
    "unittests/Compiler",
    "include/Wafer/Model",
    "include/Wafer/Program",
    "lib/Wafer/Model",
    "lib/Wafer/Program",
    "unittests/Model",
    "unittests/Program",
    "include/Wafer/IR/Tensor/GroupOps.td",
    "lib/Wafer/IR/Tensor/GroupOps.cpp",
    "lib/Wafer/Transforms/Group",
    "lib/Wafer/Conversion/WaferGroupToTileRegion",
    "include/Wafer/TestSupport",
    "include/Wafer/IR/Program",
    "include/Wafer/IR/Resource",
    "include/Wafer/IR/Target",
    "lib/Wafer/IR/Program",
    "lib/Wafer/IR/Resource",
    "lib/Wafer/IR/Target",
    "lib/Wafer/Analysis/Structured",
    "lib/Wafer/Analysis/PhysicalDataflow",
    "lib/Wafer/Analysis/Scheduling",
    "lib/Wafer/CodeGen/Executable",
    "lib/Wafer/CodeGen/Target",
    "lib/Wafer/Target/Core",
    "lib/Wafer/Target/Execution",
    "lib/Wafer/Target/Layout",
    "lib/Wafer/Target/Numeric",
    "lib/Wafer/Transforms/Bufferization",
    "lib/Wafer/Transforms/DDR",
    "lib/Wafer/Transforms/Execution",
    "lib/Wafer/Transforms/MemoryPlanning",
    "lib/Wafer/Transforms/Scheduling",
    "lib/Wafer/Transforms/SPM",
    "lib/Wafer/Transforms/Target",
    "lib/Wafer/Transforms/Transport",
    "lib/Wafer/Conversion/StandaloneTileModules",
    "lib/Wafer/Conversion/StructuredTiling",
    "lib/Wafer/Conversion/WaferTileRegionToInstr",
    "include/Wafer/Conversion/StableHLOToLinalg/StructuredGraphNormalization.h",
    "lib/Wafer/Transforms/Linalg/EGraphCore",
    "lib/Wafer/Transforms/Tile/CurrentIRLayoutOptimization.cpp",
    "lib/Wafer/Transforms/Tile/CurrentIRLayoutOptimization.h",
    "runtime/wafer_crt",
    "test/Board/Support/wafer_source_program_fixture.py",
    "test/Tools/Inputs/hf/tiny-random-llama-fp16-config.json",
    "test/Tools/Inputs/inspect_tx81_worker_calls.py",
    "test/Tools/Inputs/minimal-profile-summary-target-kernel.ll",
    "test/Tools/Inputs/wafer_pytorch_xla_row_sharded_model.py",
    "unittests/Transforms/Tile/CurrentIRLayoutOptimizationTest.cpp",
    "unittests/Planning/Search",
)

ALLOWED_LIBRARY_DIRECTORIES = {
    "Analysis",
    "CodeGen",
    "Conversion",
    "Driver",
    "Frontend",
    "IR",
    "Package",
    "Planning",
    "Runtime",
    "Simulator",
    "Target",
    "Transforms",
}


def cmake_files(root: Path) -> list[Path]:
    result = [root / "CMakeLists.txt"]
    for directory in ("cmake", "lib", "runtime", "test", "tools", "unittests"):
        base = root / directory
        if not base.is_dir():
            continue
        result.extend(base.rglob("CMakeLists.txt"))
        result.extend(base.rglob("*.cmake"))
    return sorted(set(path for path in result if "third_party" not in path.parts))


def strip_cmake_comments(text: str) -> str:
    return re.sub(r"(?m)#.*$", "", text)


def resolve_source(root: Path, manifest: Path, token: str) -> str | None:
    token = token.strip('"')
    source_prefix = "${CMAKE_SOURCE_DIR}/"
    if token.startswith(source_prefix):
        path = root / token[len(source_prefix) :]
    elif "${" in token or "$<" in token:
        return None
    else:
        path = manifest.parent / token
        owner_relative = manifest.parent.parent / token
        if not path.exists() and owner_relative.exists():
            path = owner_relative
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def source_occurrences(
    root: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    occurrences: dict[str, list[Path]] = defaultdict(list)
    optional: dict[str, list[Path]] = defaultdict(list)
    for manifest in cmake_files(root):
        text = strip_cmake_comments(manifest.read_text(encoding="utf-8"))
        for token in re.findall(r'[^\s()"]+\.cpp|"[^"]+\.cpp"', text):
            source = resolve_source(root, manifest, token)
            if source:
                occurrences[source].append(manifest)
        for body in re.findall(
            r"set\s*\(\s*LLVM_OPTIONAL_SOURCES\b(.*?)\)", text, re.S
        ):
            for token in re.findall(r'[^\s()"]+\.cpp|"[^"]+\.cpp"', body):
                source = resolve_source(root, manifest, token)
                if source:
                    optional[source].append(manifest)
    return occurrences, optional


def check_library_sources(root: Path, errors: list[str]) -> None:
    occurrences, optional = source_occurrences(root)
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "lib/Wafer").rglob("*.cpp")
    }
    declared_dormant = set(DORMANT_LIBRARY_SOURCES)
    declared_external = set(EXTERNAL_HELPER_SOURCES)
    for source in sorted(actual):
        optional_manifests = set(optional.get(source, []))
        active_manifests = [
            manifest
            for manifest in occurrences.get(source, [])
            if manifest not in optional_manifests
        ]
        if active_manifests:
            if len(active_manifests) != 1:
                errors.append(
                    f"active library source has {len(active_manifests)} CMake owners: {source}"
                )
            if source in declared_dormant or source in declared_external:
                errors.append(f"source is both active and non-CMake-owned: {source}")
            continue
        if source not in declared_dormant and source not in declared_external:
            errors.append(f"library source has no active or task-owned disposition: {source}")
    for source in sorted(declared_dormant | declared_external):
        if source not in actual:
            errors.append(f"declared non-CMake source is missing: {source}")


def check_unit_sources(root: Path, errors: list[str]) -> None:
    occurrences, _ = source_occurrences(root)
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "unittests").rglob("*.cpp")
    }
    dormant = set(DORMANT_UNIT_SOURCES)
    for source in sorted(actual):
        active = occurrences.get(source, [])
        if active:
            if source in dormant:
                errors.append(f"unit source is both active and dormant: {source}")
            continue
        if source not in dormant:
            errors.append(f"unit source has no CMake registration or task owner: {source}")
    for source in sorted(dormant):
        if source not in actual:
            errors.append(f"declared dormant unit source is missing: {source}")


def check_directory_owners(root: Path, errors: list[str]) -> None:
    for relative in RETIRED_PATHS:
        if (root / relative).exists():
            errors.append(f"retired source owner still exists: {relative}")
    actual = {
        path.name
        for path in (root / "lib/Wafer").iterdir()
        if path.is_dir()
    }
    unexpected = sorted(actual - ALLOWED_LIBRARY_DIRECTORIES)
    if unexpected:
        errors.append("unexpected top-level library owner(s): " + ", ".join(unexpected))


def check_no_retired_includes(root: Path, errors: list[str]) -> None:
    needles = (
        "Wafer/Compiler/",
        "Wafer/Pipelines/",
        "Wafer/Driver/ProgramData.h",
        "Wafer/TestSupport/",
        "Wafer/IR/Program/",
        "Wafer/IR/Resource/",
        "Wafer/IR/Target/",
        "Wafer/CodeGen/Executable/",
        "Wafer/CodeGen/Target/",
        "Wafer/Target/Core/",
        "Wafer/Target/Execution/",
        "Wafer/Target/Layout/",
        "Wafer/Target/Numeric/",
        "Wafer/Conversion/WaferTileRegionToInstr/",
    )
    suffixes = {".h", ".hpp", ".c", ".cc", ".cpp", ".td"}
    for directory in ("include", "lib", "runtime", "tools", "unittests"):
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in text:
                    errors.append(f"{path.relative_to(root)} retains {needle}")


def check_component_cmake_ownership(root: Path, errors: list[str]) -> None:
    for manifest in (root / "lib/Wafer").rglob("CMakeLists.txt"):
        text = strip_cmake_comments(manifest.read_text(encoding="utf-8"))
        if "PARENT_SCOPE" in text:
            errors.append(
                f"component CMake aggregates ownership through PARENT_SCOPE: "
                f"{manifest.relative_to(root)}"
            )
    unit_manifest = root / "unittests/CMakeLists.txt"
    if unit_manifest.is_file() and "add_executable(WaferUnitTests" in (
        strip_cmake_comments(unit_manifest.read_text(encoding="utf-8"))
    ):
        errors.append(
            "unit tests are aggregated into the retired all-component target"
        )


def check_private_header_guards(root: Path, errors: list[str]) -> None:
    private_root = root / "lib/Wafer"
    for path in sorted(private_root.rglob("*.h")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        guard = re.search(r"^#ifndef\s+(\w+)", text, re.M)
        if not guard:
            continue
        expected = "WAFER_" + re.sub(
            r"[^A-Za-z0-9]", "_", path.relative_to(private_root).as_posix()
        ).upper()
        if guard.group(1) != expected or f"#define {expected}" not in text:
            errors.append(
                "private header guard does not match current owner path: "
                f"{path.relative_to(root)}"
            )


def source_text_files(root: Path, relative: str) -> list[Path]:
    suffixes = {".h", ".hpp", ".c", ".cc", ".cpp", ".td"}
    base = root / relative
    if not base.is_dir():
        return []
    return sorted(
        path for path in base.rglob("*") if path.is_file() and path.suffix in suffixes
    )


def check_forbidden_include_edges(root: Path, errors: list[str]) -> None:
    boundaries = {
        "Transforms": (
            ("include/Wafer/Transforms", "lib/Wafer/Transforms"),
            (
                'Wafer/CodeGen/',
                'Wafer/Conversion/',
                'Wafer/Driver/',
                'Wafer/Runtime/',
                'Wafer/Simulator/',
            ),
        ),
        "Target": (
            ("include/Wafer/Target", "lib/Wafer/Target"),
            (
                'Wafer/Analysis/',
                'Wafer/CodeGen/',
                'Wafer/Conversion/',
                'Wafer/Driver/',
                'Wafer/Frontend/',
                'Wafer/Planning/',
                'Wafer/Runtime/',
                'Wafer/Simulator/',
                'Wafer/Transforms/',
            ),
        ),
        "Analysis": (
            ("include/Wafer/Analysis", "lib/Wafer/Analysis"),
            (
                'Wafer/CodeGen/',
                'Wafer/Conversion/',
                'Wafer/Driver/',
                'Wafer/Planning/',
                'Wafer/Runtime/',
                'Wafer/Simulator/',
                'Wafer/Transforms/',
            ),
        ),
        "Planning": (
            ("include/Wafer/Planning", "lib/Wafer/Planning"),
            (
                'Wafer/CodeGen/',
                'Wafer/Conversion/',
                'Wafer/Driver/',
                'Wafer/Runtime/',
                'Wafer/Simulator/',
                'Wafer/Transforms/',
            ),
        ),
    }
    for owner, (directories, forbidden) in boundaries.items():
        for directory in directories:
            for path in source_text_files(root, directory):
                text = path.read_text(encoding="utf-8", errors="ignore")
                for edge in forbidden:
                    if f'#include "{edge}' in text:
                        errors.append(
                            f"{owner} has a forbidden include edge {edge}: "
                            f"{path.relative_to(root)}"
                        )

    exact_forbidden = {
        "lib/Wafer/Package": ('Wafer/Driver/Compilation',),
        "include/Wafer/Package": ('Wafer/Driver/Compilation',),
        "lib/Wafer/Simulator/Memory": ('Wafer/Simulator/Invocation/',),
        "include/Wafer/Simulator/Memory": ('Wafer/Simulator/Invocation/',),
    }
    for directory, forbidden in exact_forbidden.items():
        for path in source_text_files(root, directory):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for edge in forbidden:
                if edge in text:
                    errors.append(
                        f"forbidden component include {edge}: {path.relative_to(root)}"
                    )

    transform_cmake = (root / "lib/Wafer/Transforms/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    for target in ("WaferStableHLOToLinalg", "WaferTileToInstr", "WaferInstrToLLVM"):
        if target in transform_cmake:
            errors.append(f"WaferTransforms links conversion target {target}")
    qualification_cmake = (
        root / "tools/wafer-cmodel-qualify-onednn/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    if "WaferCompiler" in qualification_cmake:
        errors.append("oneDNN qualification tool links the whole compiler")

    required_cmake_edges = {
        "lib/Wafer/Transforms/CMakeLists.txt": (
            "WaferPlanning",
            "WaferStableHLOTransforms",
        ),
        "lib/Wafer/Simulator/Memory/CMakeLists.txt": ("WaferCodeGen",),
        "lib/Wafer/Package/CMakeLists.txt": ("WaferFrontend",),
        "tools/wafer-cmodel-qualify-onednn/CMakeLists.txt": (
            "WaferCodeGen",
            "WaferPhysicalTensor",
            "WaferSimulatorInvocation",
        ),
    }
    for relative, required in required_cmake_edges.items():
        text = (root / relative).read_text(encoding="utf-8")
        for target in required:
            if target not in text:
                errors.append(f"{relative} is missing direct dependency {target}")


def check_structured_egraph_owner(root: Path, errors: list[str]) -> None:
    crate = root / "lib/Wafer/Transforms/Linalg/StructuredEGraph"
    for relative in ("Cargo.toml", "Cargo.lock", "src/lib.rs"):
        if not (crate / relative).is_file():
            errors.append(f"structured e-graph crate is missing {relative}")
    cmake_text = (
        root / "cmake/third_party/WaferThirdParty.cmake"
    ).read_text(encoding="utf-8")
    if "WaferThirdPartyStructuredEGraph" in cmake_text:
        errors.append("Wafer-owned structured e-graph retains a ThirdParty target")
    if "lib/Wafer/Transforms/Linalg/StructuredEGraph" not in cmake_text:
        errors.append("structured e-graph CMake owner does not name the current crate")


def check_test_support_consumers(root: Path, errors: list[str]) -> None:
    candidates = []
    tools_inputs = root / "test/Tools/Inputs"
    if tools_inputs.is_dir():
        candidates.extend(
            path
            for path in tools_inputs.rglob("*")
            if path.is_file()
            and path.name != "README.md"
            and path.suffix in {".c", ".h", ".json", ".ll", ".py"}
        )
    board_support = root / "test/Board/Support"
    if board_support.is_dir():
        candidates.extend(board_support.glob("*.py"))

    test_text_files = [
        path
        for path in (root / "test").rglob("*")
        if path.is_file()
        and path.suffix in {".c", ".h", ".json", ".ll", ".mlir", ".py", ".test", ".txt"}
    ]
    test_text_files.extend(cmake_files(root))
    texts = {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in test_text_files
    }
    for candidate in sorted(set(candidates)):
        tokens = (candidate.name, candidate.stem)
        if any(
            any(token in text for token in tokens)
            for path, text in texts.items()
            if path != candidate
        ):
            continue
        errors.append(
            "test support asset has no current consumer: "
            f"{candidate.relative_to(root)}"
        )


def check_python_test_registration(root: Path, errors: list[str]) -> None:
    registration_files = [*cmake_files(root)]
    registration_files.extend((root / "test").rglob("*.test"))
    registration_files.extend((root / "test").rglob("*.mlir"))
    registration_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in registration_files
    )
    for path in sorted((root / "test").rglob("*_test.py")):
        if path.name not in registration_text:
            errors.append(
                "Python test has no CTest registration: "
                f"{path.relative_to(root)}"
            )

    board_support = root / "test/Board/Support"
    if board_support.is_dir():
        for path in sorted(board_support.glob("*_test.py")):
            errors.append(
                "Board helper uses a test filename in Support: "
                f"{path.relative_to(root)}"
            )


def check_source_tree_artifacts(root: Path, errors: list[str]) -> None:
    if (root / ".deps").exists():
        errors.append("legacy source-tree dependency artifact exists: .deps")
    owned_directories = (
        "cmake",
        "docs",
        "include",
        "lib",
        "memory",
        "python",
        "runtime",
        "tasks",
        "test",
        "tools",
        "unittests",
        "utils",
    )
    for directory in owned_directories:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_dir() and path.name == "__pycache__":
                errors.append(
                    "source-tree Python cache exists: "
                    f"{path.relative_to(root)}"
                )
            elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
                errors.append(
                    "source-tree Python bytecode exists: "
                    f"{path.relative_to(root)}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    check_directory_owners(root, errors)
    check_no_retired_includes(root, errors)
    check_component_cmake_ownership(root, errors)
    check_private_header_guards(root, errors)
    check_forbidden_include_edges(root, errors)
    check_structured_egraph_owner(root, errors)
    check_library_sources(root, errors)
    check_unit_sources(root, errors)
    check_python_test_registration(root, errors)
    check_test_support_consumers(root, errors)
    check_source_tree_artifacts(root, errors)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Wafer source organization checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
