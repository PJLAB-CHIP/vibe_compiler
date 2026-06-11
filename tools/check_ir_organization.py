#!/usr/bin/env python3
"""Check Wafer IR source/test organization boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


OP_FAMILIES = {
    "Group": {
        "layer": "Tensor",
        "td": "GroupOps.td",
        "cpp": "GroupOps.cpp",
        "mnemonics": ["group", "group.yield"],
        "tests": "Tensor/Group",
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
    "Placement": {
        "layer": "Resource",
        "td": "PlacementOps.td",
        "cpp": "PlacementOps.cpp",
        "mnemonics": ["placement.map"],
        "tests": "Resource/Placement",
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
        "mnemonics": [
            "tile.send",
            "tile.recv",
            "tile.wait",
            "tile.all_gather",
            "tile.reduce_scatter",
            "tile.all_reduce",
        ],
        "tests": "Tile/Comm",
    },
    "TensorCollective": {
        "layer": "Tensor",
        "td": "TensorCollectiveOps.td",
        "cpp": "TensorCollectiveOps.cpp",
        "mnemonics": [
            "tensor.all_gather",
            "tensor.all_reduce",
            "tensor.reduce_scatter",
            "tensor.all_to_all",
            "tensor.collective_permute",
            "tensor.yield",
        ],
        "tests": "Tensor/TensorCollective",
    },
    "Sync": {
        "layer": "Instr",
        "td": "SyncOps.td",
        "cpp": "SyncOps.cpp",
        "mnemonics": ["instr.local_drain"],
        "tests": "Instr/Sync",
    },
    "Launch": {
        "layer": "Runtime",
        "td": "LaunchOps.td",
        "cpp": "LaunchOps.cpp",
        "mnemonics": ["launch"],
        "tests": "Runtime/Launch",
    },
}

SUPPORT_TEST_DIRS = {"Common/Attrs"}
IR_LAYERS = {"Tensor", "Tile", "Resource", "Instr", "Runtime", "Common"}
CONVERSION_LIBRARIES = {
    "WaferGroupToTileRegion": {
        "include": "include/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h",
        "lib": "lib/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.cpp",
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
GROUP_ANALYSIS_FILES = [
    "include/Wafer/Analysis/Group/LayoutPlanningAnalysis.h",
    "include/Wafer/Analysis/Group/TilingDemandAnalysis.h",
    "lib/Wafer/Analysis/Group/LayoutPlanningAnalysis.cpp",
    "lib/Wafer/Analysis/Group/TilingDemandAnalysis.cpp",
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
    "wafer.tensor_collective",
    "wafer.tile_region",
    "wafer.tile_yield",
    "wafer.group_yield",
    "WAFER_INSTR_CT_",
    "WAFER_INSTR_NE_",
    "WAFER_INSTR_TDMA_",
    "AbiOps",
    "DDROps",
    "DdrExternalBindingOp",
    "DdrBindingKind",
    "AbiWaitPolicy",
)
ALLOWED_IR_SPECIALIZATIONS = (
    "wafer.ddr.offset",
)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_file(path: Path, errors: list[str]) -> str:
    if not path.exists():
        fail(errors, f"missing required file: {path}")
        return ""
    return path.read_text()


def check_main_ops_td(root: Path, errors: list[str]) -> None:
    main_td = root / "include/Wafer/IR/WaferOps.td"
    text = check_file(main_td, errors)
    if not text:
        return

    direct_defs = re.findall(r"^def Wafer_", text, flags=re.MULTILINE)
    if direct_defs:
        fail(errors, "WaferOps.td must aggregate op-family files, not define ops")

    for spec in OP_FAMILIES.values():
        include = f'include "Wafer/IR/{spec["layer"]}/{spec["td"]}"'
        if include not in text:
            fail(errors, f"WaferOps.td missing {include}")


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
            if path.name == "check_ir_organization.py":
                continue
            text = path.read_text(errors="ignore")
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
    ]:
        if old_conversion.exists():
            fail(errors, f"old artifact-named conversion directory must be removed: {old_conversion}")


def check_analysis_organization(root: Path, errors: list[str]) -> None:
    lib_cmake = check_file(root / "lib/Wafer/CMakeLists.txt", errors)
    analysis_cmake = check_file(root / "lib/Wafer/Analysis/CMakeLists.txt", errors)
    transforms_cmake = check_file(root / "lib/Wafer/Transforms/CMakeLists.txt", errors)

    if "add_subdirectory(Analysis)" not in lib_cmake:
        fail(errors, "lib/Wafer/CMakeLists.txt must add_subdirectory(Analysis)")
    if "add_mlir_library(WaferAnalysis" not in analysis_cmake:
        fail(errors, "group analysis must be owned by WaferAnalysis")
    for relative in GROUP_ANALYSIS_FILES:
        check_file(root / relative, errors)
    for source in ["Group/LayoutPlanningAnalysis.cpp", "Group/TilingDemandAnalysis.cpp"]:
        if source in transforms_cmake:
            fail(errors, f"WaferTransforms must not compile group analysis source {source}")
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
