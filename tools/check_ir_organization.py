#!/usr/bin/env python3
"""Check Wafer IR source/test organization boundaries."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path


OP_FAMILY_LAYOUT = {
    "Program/ModuleOps.td": {
        "cpp": ("Program/ModuleOps.cpp",),
        "tests": "Program/Module",
    },
    "Target/TopologyOps.td": {
        "cpp": ("Target/TopologyOps.cpp",),
        "tests": "Target/Topology",
    },
    "LinalgExt/CollectiveOps.td": {
        "cpp": ("LinalgExt/CollectiveOps.cpp",),
        "tests": "LinalgExt/Collective",
    },
    "LinalgExt/AttentionOps.td": {
        "cpp": ("LinalgExt/AttentionOps.cpp",),
        "tests": "LinalgExt/Attention",
    },
    "Tile/TileRegionOps.td": {
        "cpp": ("Tile/TileRegionOps.cpp",),
        "tests": "Tile/TileRegion",
    },
    "Tile/LayoutOps.td": {
        "cpp": ("Tile/LayoutOps.cpp",),
        "tests": "Tile/Layout",
    },
    "Tile/ComputeOps.td": {
        "cpp": ("Tile/ComputeOps.cpp",),
        "tests": "Tile/Compute",
    },
    "Tile/MoveOps.td": {
        "cpp": ("Tile/MoveOps.cpp",),
        "tests": "Tile/Move",
    },
    "Tile/ViewOps.td": {
        "cpp": ("Tile/ViewOps.cpp",),
        "tests": "Tile/View",
    },
    "Tile/CommOps.td": {
        "cpp": ("Tile/CommOps.cpp",),
        "tests": "Tile/Comm",
    },
    "Resource/SPMOps.td": {
        "cpp": ("Resource/SPMOps.cpp",),
        "tests": "Resource/SPM",
    },
    # InstructionOps.td deliberately splits verifier implementations by
    # instruction family. The ODS include remains the schema truth.
    "Instr/InstructionOps.td": {
        "cpp": (
            "Instr/ComputeOps.cpp",
            "Instr/MovementOps.cpp",
            "Instr/PeripheralOps.cpp",
        ),
        "tests": "Instr/Instruction",
    },
    "Instr/DTEOps.td": {
        "cpp": ("Instr/DTEOps.cpp",),
        "tests": "Instr/DTE",
    },
    "Instr/SyncOps.td": {
        "cpp": ("Instr/SyncOps.cpp",),
        "tests": "Instr/Sync",
    },
}

SUPPORT_TEST_DIRS = {"Common/Attrs"}
IR_LAYERS = {
    "Program",
    "Target",
    "LinalgExt",
    "Tile",
    "Resource",
    "Instr",
    "Runtime",
    "Common",
}
CONVERSION_LIBRARIES = {
    "WaferTileModuleFanout": {
        "include": "include/Wafer/Conversion/WaferTileModuleFanout/WaferTileModuleFanout.h",
        "lib": "lib/Wafer/Conversion/WaferTileModuleFanout/WaferTileModuleFanout.cpp",
    },
    "WaferStructuredTiling": {
        "include": "include/Wafer/Conversion/StructuredTiling.h",
        "lib": "lib/Wafer/Conversion/StructuredTiling/StructuredTiling.cpp",
    },
    "WaferTileRegionToInstr": {
        "include": "include/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h",
        "lib": "lib/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.cpp",
    },
}
STABLEHLO_CONVERSION_SOURCES = [
    "lib/Wafer/Conversion/StableHLOToLinalg/AttentionMatching.cpp",
    "lib/Wafer/Conversion/StableHLOToLinalg/LegalizeStablehloToLinalg.cpp",
    "lib/Wafer/Conversion/StableHLOToLinalg/NormalizeAttention.cpp",
    "lib/Wafer/Conversion/StableHLOToLinalg/NormalizeStablehloCollectives.cpp",
]
FORBIDDEN_IR_STRINGS = (
    "wafer.abi.",
    "wafer.ddr.",
    "wafer.comm.",
    "wafer.compute.",
    "wafer.move.",
    "wafer.layout.",
    "wafer.view.",
    "wafer.sync.",
    "wafer.storage.",
    "wafer.instr.ct.",
    "wafer.instr.ne.",
    "wafer.instr.tdma.",
    "wafer_linalg_ext",
    "wafer.linalg_ext_collective",
    "wafer.tile_region",
    "wafer.tile_yield",
    "WAFER_INSTR_CT_",
    "WAFER_INSTR_NE_",
    "WAFER_INSTR_TDMA_",
    "AbiOps",
    "DDROps",
    "DdrExternalBindingOp",
    "DdrBindingKind",
    "AbiWaitPolicy",
    "WaferLinalgExtDialect",
    "WaferLinalgExtOps",
)
ALLOWED_IR_SPECIALIZATIONS = (
    "wafer.ddr.offset",
)
LEGACY_GROUP_IR_PATHS = (
    "include/Wafer/IR/Tensor/GroupOps.td",
    "lib/Wafer/IR/Tensor/GroupOps.cpp",
    "test/Dialect/Wafer/Tensor/Group",
)
LEGACY_GROUP_IR_PATTERNS = (
    (re.compile(r"(?<![A-Za-z0-9_])wafer\.group\b"), "wafer.group mnemonic"),
    (
        re.compile(r"\b(?:Wafer_)?Group(?:Yield)?Op\b"),
        "wafer.group ODS/C++ op API",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_])(?:Tensor/)?GroupOps\.(?:td|cpp)\b"),
        "wafer.group ODS/source include",
    ),
)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_file(path: Path, errors: list[str]) -> str:
    if not path.exists():
        fail(errors, f"missing required file: {path}")
        return ""
    return path.read_text()


def get_aggregated_op_td_files(root: Path, errors: list[str]) -> set[str]:
    path = root / "include/Wafer/IR/WaferOps.td"
    text = check_file(path, errors)
    if re.search(r"^def Wafer_", text, flags=re.MULTILINE):
        fail(errors, "WaferOps.td must aggregate op-family files, not define ops")
    return {
        match
        for match in re.findall(
            r'^include "Wafer/IR/([^\"]+Ops\.td)"', text, flags=re.MULTILINE
        )
        if match != "WaferOps.td"
    }


def get_defined_op_mnemonics(td_text: str) -> set[str]:
    definition = re.compile(
        r"^def\s+Wafer_[A-Za-z0-9_]+Op\b"
        r"(?:(?!^def\s).)*?"
        r":\s*Wafer_[A-Za-z0-9_]*Op<\s*\"([^\"]+)\"",
        flags=re.MULTILINE | re.DOTALL,
    )
    return set(definition.findall(td_text))


def get_mlir_library_sources(cmake_text: str, target: str) -> set[str]:
    block = re.search(
        rf"add_mlir_library\(\s*{re.escape(target)}\s+(.*?)(?:\n\s*LINK_LIBS|\n\s*DEPENDS)",
        cmake_text,
        flags=re.DOTALL,
    )
    if not block:
        return set()
    return set(
        re.findall(r"^\s*([A-Za-z0-9_./+-]+\.cpp)\s*$", block.group(1), re.MULTILINE)
    )


def check_main_ops_td(root: Path, errors: list[str]) -> None:
    included = get_aggregated_op_td_files(root, errors)
    expected = set(OP_FAMILY_LAYOUT)
    for missing in sorted(expected - included):
        fail(errors, f"WaferOps.td missing op-family include Wafer/IR/{missing}")
    for unowned in sorted(included - expected):
        fail(errors, f"WaferOps.td op-family include has no organization owner: {unowned}")


def check_op_family_files(root: Path, errors: list[str]) -> None:
    td_root = root / "include/Wafer/IR"
    cpp_root = root / "lib/Wafer/IR"
    dialect_cpp = check_file(root / "lib/Wafer/IR/WaferDialect.cpp", errors)

    discovered_mnemonics: set[str] = set()
    for td_relative, spec in OP_FAMILY_LAYOUT.items():
        td_text = check_file(td_root / td_relative, errors)
        mnemonics = get_defined_op_mnemonics(td_text)
        if not mnemonics:
            fail(errors, f"{td_relative} defines no Wafer operations")
        duplicates = discovered_mnemonics & mnemonics
        for mnemonic in sorted(duplicates):
            fail(errors, f"duplicate Wafer operation mnemonic: {mnemonic}")
        discovered_mnemonics.update(mnemonics)

        cpp_texts = [check_file(cpp_root / path, errors) for path in spec["cpp"]]
        if not any("::verify()" in text for text in cpp_texts):
            fail(errors, f"{td_relative} has no verifier implementation owner")

    if "::verify()" in dialect_cpp:
        fail(errors, "WaferDialect.cpp must not own op verifier definitions")

    check_file(cpp_root / "Common/WaferIRVerification.h", errors)
    check_file(cpp_root / "Common/WaferIRVerification.cpp", errors)

    old_td_root = root / "include/Wafer/IR/Ops"
    old_cpp_root = root / "lib/Wafer/IR/Ops"
    if old_td_root.exists():
        fail(errors, f"old op-family ODS directory must be removed: {old_td_root}")
    if old_cpp_root.exists():
        fail(errors, f"old op-family C++ directory must be removed: {old_cpp_root}")


def check_tests(root: Path, errors: list[str]) -> None:
    test_root = root / "test/Dialect/Wafer"
    if not test_root.exists():
        fail(errors, f"missing Wafer dialect test root: {test_root}")
        return

    allowed_dirs = {
        spec["tests"] for spec in OP_FAMILY_LAYOUT.values()
    } | SUPPORT_TEST_DIRS
    for directory in allowed_dirs:
        path = test_root / directory
        if not path.is_dir():
            fail(errors, f"missing Wafer dialect test directory: {path}")
        elif not any(path.glob("*.mlir")):
            fail(errors, f"Wafer dialect test directory is empty: {path}")

    for path in test_root.glob("*.mlir"):
        fail(errors, f"Wafer dialect test must live in a family directory: {path}")

    for path in test_root.iterdir():
        if path.is_dir() and path.name not in IR_LAYERS:
            fail(errors, f"unexpected Wafer dialect test layer directory: {path}")
    actual_dirs = {
        str(path.relative_to(test_root))
        for layer in test_root.iterdir()
        if layer.is_dir()
        for path in layer.iterdir()
        if path.is_dir()
    }
    for directory in sorted(actual_dirs - allowed_dirs):
        fail(errors, f"unowned Wafer dialect test directory: {directory}")


def run_self_test() -> int:
    sample = """
def Wafer_FirstOp : Wafer_Op<"test.first"> { }
def Wafer_SecondOp
    : Wafer_InstrOp<"test.second", [Pure]> { }
"""
    if get_defined_op_mnemonics(sample) != {"test.first", "test.second"}:
        print("error: op-definition parser self-test failed", file=sys.stderr)
        return 1
    cmake_sample = """
add_mlir_library(WaferCompiler
  CompilationStages.cpp
  Nested/TargetModule.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
"""
    if get_mlir_library_sources(cmake_sample, "WaferCompiler") != {
        "CompilationStages.cpp",
        "Nested/TargetModule.cpp",
    }:
        print("error: CMake library-source parser self-test failed", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="wafer-ir-organization-") as temp:
        root = Path(temp)
        aggregator = root / "include/Wafer/IR/WaferOps.td"
        aggregator.parent.mkdir(parents=True)
        included = sorted(OP_FAMILY_LAYOUT)
        aggregator.write_text(
            "\n".join(f'include "Wafer/IR/{path}"' for path in included)
            + "\n"
        )
        errors: list[str] = []
        check_main_ops_td(root, errors)
        if errors:
            print(
                "error: complete op-family fixture was rejected: "
                + "; ".join(errors),
                file=sys.stderr,
            )
            return 1

        aggregator.write_text(
            "\n".join(f'include "Wafer/IR/{path}"' for path in included[1:])
            + "\n"
        )
        errors = []
        check_main_ops_td(root, errors)
        if not any("missing op-family include" in error for error in errors):
            print("error: missing op-family fixture passed", file=sys.stderr)
            return 1

        aggregator.write_text(
            "\n".join(f'include "Wafer/IR/{path}"' for path in included)
            + '\ninclude "Wafer/IR/Test/UnownedOps.td"\n'
        )
        errors = []
        check_main_ops_td(root, errors)
        if not any("has no organization owner" in error for error in errors):
            print("error: unowned op-family fixture passed", file=sys.stderr)
            return 1

        test_root = root / "test/Dialect/Wafer"
        allowed_dirs = {
            spec["tests"] for spec in OP_FAMILY_LAYOUT.values()
        } | SUPPORT_TEST_DIRS
        for directory in allowed_dirs:
            path = test_root / directory
            path.mkdir(parents=True, exist_ok=True)
            (path / "contract.mlir").write_text("module {}\n")
        errors = []
        check_tests(root, errors)
        if errors:
            print(
                "error: complete test-directory fixture was rejected: "
                + "; ".join(errors),
                file=sys.stderr,
            )
            return 1
        unowned = test_root / "Instr/Unowned"
        unowned.mkdir(parents=True)
        (unowned / "contract.mlir").write_text("module {}\n")
        errors = []
        check_tests(root, errors)
        if not any("unowned Wafer dialect test directory" in error for error in errors):
            print("error: unowned test-directory fixture passed", file=sys.stderr)
            return 1

    print("Wafer IR organization self-test passed")
    return 0


def check_forbidden_ir_specializations(root: Path, errors: list[str]) -> None:
    for relative in LEGACY_GROUP_IR_PATHS:
        path = root / relative
        if path.exists():
            fail(errors, f"retired wafer.group IR path must not exist: {path}")

    scan_roots = [
        root / "include/Wafer/IR",
        root / "lib/Wafer/IR",
        root / "lib/Wafer/Conversion",
        root / "test",
        root / "tools",
    ]
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.name in {
                "check_ir_organization.py",
                "check_source_organization.py",
            }:
                continue
            text = path.read_text(errors="ignore")
            for pattern, label in LEGACY_GROUP_IR_PATTERNS:
                if pattern.search(text):
                    fail(errors, f"{path} contains retired {label}")
            for forbidden in FORBIDDEN_IR_STRINGS:
                scan_text = text
                for allowed in ALLOWED_IR_SPECIALIZATIONS:
                    scan_text = scan_text.replace(allowed, "")
                if forbidden in scan_text:
                    fail(errors, f"{path} contains forbidden IR specialization {forbidden}")


def check_conversion_organization(root: Path, errors: list[str]) -> None:
    lib_cmake = check_file(root / "lib/Wafer/CMakeLists.txt", errors)
    conversion_cmake = check_file(root / "lib/Wafer/Conversion/CMakeLists.txt", errors)
    transforms_cmake = check_file(root / "lib/Wafer/Transforms/CMakeLists.txt", errors)

    if "add_subdirectory(Conversion)" not in lib_cmake:
        fail(errors, "lib/Wafer/CMakeLists.txt must add_subdirectory(Conversion)")
    if "../Conversion/" in transforms_cmake:
        fail(errors, "WaferTransforms must not compile sources from ../Conversion")
    if "StableHLOToLinalg/" in transforms_cmake:
        fail(errors, "StableHLOToLinalg conversion sources must not live in WaferTransforms")
    if "add_mlir_conversion_library" not in conversion_cmake:
        fail(errors, "Wafer conversion libraries must use add_mlir_conversion_library")
    if "WaferStableHLOToLinalg" not in conversion_cmake:
        fail(errors, "lib/Wafer/Conversion/CMakeLists.txt missing WaferStableHLOToLinalg")

    for name, paths in CONVERSION_LIBRARIES.items():
        for key, relative in paths.items():
            check_file(root / relative, errors)
        if name not in conversion_cmake:
            fail(errors, f"lib/Wafer/Conversion/CMakeLists.txt missing {name}")
    for relative in STABLEHLO_CONVERSION_SOURCES:
        check_file(root / relative, errors)

    for old_conversion in [
        root / "include/Wafer/Conversion/TileRegionCandidate",
        root / "lib/Wafer/Conversion/TileRegionCandidate",
        root / "include/Wafer/Conversion/WaferGroupToTileRegion",
        root / "lib/Wafer/Conversion/WaferGroupToTileRegion",
    ]:
        if old_conversion.exists():
            fail(errors, f"retired conversion directory must be removed: {old_conversion}")


def check_analysis_organization(root: Path, errors: list[str]) -> None:
    lib_cmake = check_file(root / "lib/Wafer/CMakeLists.txt", errors)
    analysis_cmake = check_file(root / "lib/Wafer/Analysis/CMakeLists.txt", errors)

    if "add_subdirectory(Analysis)" not in lib_cmake:
        fail(errors, "lib/Wafer/CMakeLists.txt must add_subdirectory(Analysis)")
    if "add_mlir_library(WaferAnalysis" not in analysis_cmake:
        fail(errors, "compiler analyses must be owned by WaferAnalysis")
    for legacy_root in (
        root / "include/Wafer/Analysis/Group",
        root / "lib/Wafer/Analysis/Group",
    ):
        if legacy_root.exists():
            fail(errors, f"legacy group analysis directory must be removed: {legacy_root}")
    old_include = root / "include/Wafer/Transforms/Group"
    if old_include.exists():
        fail(errors, f"old transform-owned analysis include directory must be removed: {old_include}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    root = args.root.resolve()
    errors: list[str] = []
    check_main_ops_td(root, errors)
    check_op_family_files(root, errors)
    check_tests(root, errors)
    check_forbidden_ir_specializations(root, errors)
    check_analysis_organization(root, errors)
    check_conversion_organization(root, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Wafer IR organization checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
