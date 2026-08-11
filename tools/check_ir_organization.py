#!/usr/bin/env python3
"""Check Wafer IR source/test organization boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


OP_FAMILIES = {
    "Program": {
        "layer": "Program",
        "td": "ProgramOps.td",
        "cpp": "ProgramOps.cpp",
        "mnemonics": ["card.program", "tile.program"],
        "tests": "Program/Program",
    },
    "TargetTopology": {
        "layer": "Target",
        "td": "TopologyOps.td",
        "cpp": "TopologyOps.cpp",
        "mnemonics": ["target.topology"],
        "tests": "Target/Topology",
    },
    "TileRegion": {
        "layer": "Tile",
        "td": "TileRegionOps.td",
        "cpp": "TileRegionOps.cpp",
        "mnemonics": ["tile.region", "tile.yield"],
        "tests": "Tile/TileRegion",
    },
    "Layout": {
        "layer": "Tile",
        "td": "LayoutOps.td",
        "cpp": "LayoutOps.cpp",
        "mnemonics": ["tile.materialize_layout"],
        "tests": "Tile/Layout",
    },
    "SPM": {
        "layer": "Resource",
        "td": "SPMOps.td",
        "cpp": "SPMOps.cpp",
        "mnemonics": ["tile.load", "tile.store"],
        "tests": "Resource/SPM",
    },
    "Move": {
        "layer": "Tile",
        "td": "MoveOps.td",
        "cpp": "MoveOps.cpp",
        "mnemonics": [
            "tile.extract_slice",
            "tile.insert_slice",
            "tile.copy",
            "tile.transpose",
            "tile.broadcast",
        ],
        "tests": "Tile/Move",
    },
    "View": {
        "layer": "Tile",
        "td": "ViewOps.td",
        "cpp": "ViewOps.cpp",
        "mnemonics": ["tile.reshape"],
        "tests": "Tile/View",
    },
    "Compute": {
        "layer": "Tile",
        "td": "ComputeOps.td",
        "cpp": "ComputeOps.cpp",
        "mnemonics": ["tile.fill", "tile.gemm", "tile.elementwise", "tile.reduce"],
        "tests": "Tile/Compute",
    },
    "Comm": {
        "layer": "Tile",
        "td": "CommOps.td",
        "cpp": "CommOps.cpp",
        "mnemonics": ["tile.peer_send", "tile.peer_recv"],
        "tests": "Tile/Comm",
    },
    "LinalgExtCollective": {
        "layer": "LinalgExt",
        "td": "CollectiveOps.td",
        "cpp": "CollectiveOps.cpp",
        "mnemonics": [
            "linalg_ext.collective.all_gather",
            "linalg_ext.collective.all_reduce",
            "linalg_ext.collective.reduce_scatter",
            "linalg_ext.collective.all_to_all",
            "linalg_ext.collective.collective_permute",
            "linalg_ext.collective.yield",
        ],
        "tests": "LinalgExt/Collective",
    },
    "DTE": {
        "layer": "Instr",
        "td": "DTEOps.td",
        "cpp": "DTEOps.cpp",
        "mnemonics": ["instr.dte_send", "instr.dte_recv", "instr.dte_wait"],
        "tests": "Instr/DTE",
    },
    "Sync": {
        "layer": "Instr",
        "td": "SyncOps.td",
        "cpp": "SyncOps.cpp",
        "mnemonics": ["instr.ncc_join"],
        "tests": "Instr/Sync",
    },
}

SUPPORT_TEST_DIRS = {"Common/Attrs"}
IR_LAYERS = {
    "Program",
    "Target",
    "Tensor",
    "LinalgExt",
    "Tile",
    "Resource",
    "Instr",
    "Runtime",
    "Common",
}
CONVERSION_LIBRARIES = {
    "WaferCardProgramToTileModules": {
        "include": "include/Wafer/Conversion/WaferCardProgramToTileModules/WaferCardProgramToTileModules.h",
        "lib": "lib/Wafer/Conversion/WaferCardProgramToTileModules/WaferCardProgramToTileModules.cpp",
    },
    "WaferTensorProgramToTileRegion": {
        "include": "include/Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h",
        "lib": "lib/Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.cpp",
    },
    "WaferTileRegionToInstr": {
        "include": "include/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h",
        "lib": "lib/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.cpp",
    },
}
STABLEHLO_CONVERSION_SOURCES = [
    "lib/Wafer/Conversion/StableHLOToLinalg/LegalizeStablehloToLinalg.cpp",
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


def check_main_ops_td(root: Path, errors: list[str]) -> None:
    aggregator_texts: dict[str, str] = {}
    for filename in ("WaferOps.td",):
        path = root / "include/Wafer/IR" / filename
        text = check_file(path, errors)
        if not text:
            continue
        aggregator_texts[filename] = text

        direct_defs = re.findall(r"^def Wafer_", text, flags=re.MULTILINE)
        if direct_defs:
            fail(errors, f"{filename} must aggregate op-family files, not define ops")

    for spec in OP_FAMILIES.values():
        aggregator = spec.get("aggregator", "WaferOps.td")
        text = aggregator_texts.get(aggregator, "")
        include = f'include "Wafer/IR/{spec["layer"]}/{spec["td"]}"'
        if include not in text:
            fail(errors, f"{aggregator} missing {include}")


def check_op_family_files(root: Path, errors: list[str]) -> None:
    td_root = root / "include/Wafer/IR"
    cpp_root = root / "lib/Wafer/IR"
    dialect_cpp = check_file(root / "lib/Wafer/IR/WaferDialect.cpp", errors)

    for name, spec in OP_FAMILIES.items():
        td_text = check_file(td_root / spec["layer"] / spec["td"], errors)
        cpp_text = check_file(cpp_root / spec["layer"] / spec["cpp"], errors)
        for mnemonic in spec["mnemonics"]:
            if mnemonic not in td_text:
                fail(errors, f"{spec['td']} missing op mnemonic {mnemonic}")

        if name != "Sync" and "::verify()" not in cpp_text:
            fail(errors, f"{spec['cpp']} must own {name} verifier definitions")

    if "::verify()" in dialect_cpp:
        fail(errors, "WaferDialect.cpp must not own op verifier definitions")

    check_file(cpp_root / "Common/OpVerifierUtils.h", errors)
    check_file(cpp_root / "Common/OpVerifierUtils.cpp", errors)

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

    allowed_dirs = {spec["tests"] for spec in OP_FAMILIES.values()} | SUPPORT_TEST_DIRS
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
            fail(errors, f"old artifact-named conversion directory must be removed: {old_conversion}")


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
    args = parser.parse_args()

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
