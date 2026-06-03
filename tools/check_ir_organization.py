#!/usr/bin/env python3
"""Check Wafer IR source/test organization boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


OP_FAMILIES = {
    "Group": {
        "td": "GroupOps.td",
        "cpp": "GroupOps.cpp",
        "mnemonics": ["group", "group_yield"],
        "tests": "Group",
    },
    "TileRegion": {
        "td": "TileRegionOps.td",
        "cpp": "TileRegionOps.cpp",
        "mnemonics": ["tile_region", "tile_yield"],
        "tests": "TileRegion",
    },
    "Layout": {
        "td": "LayoutOps.td",
        "cpp": "LayoutOps.cpp",
        "mnemonics": ["layout.materialize"],
        "tests": "Layout",
    },
    "SPM": {
        "td": "SPMOps.td",
        "cpp": "SPMOps.cpp",
        "mnemonics": ["load_tile", "store_tile"],
        "tests": "SPM",
    },
    "Move": {
        "td": "MoveOps.td",
        "cpp": "MoveOps.cpp",
        "mnemonics": [
            "move.extract_slice",
            "move.insert_slice",
            "move.copy",
            "move.transpose",
            "move.broadcast",
        ],
        "tests": "Move",
    },
    "View": {
        "td": "ViewOps.td",
        "cpp": "ViewOps.cpp",
        "mnemonics": ["view.reshape"],
        "tests": "View",
    },
    "DDR": {
        "td": "DDROps.td",
        "cpp": "DDROps.cpp",
        "mnemonics": ["ddr.external_binding"],
        "tests": "DDR",
    },
    "Placement": {
        "td": "PlacementOps.td",
        "cpp": "PlacementOps.cpp",
        "mnemonics": ["placement.map"],
        "tests": "Placement",
    },
    "ABI": {
        "td": "AbiOps.td",
        "cpp": "AbiOps.cpp",
        "mnemonics": [
            "abi.rdma",
            "abi.wdma",
            "abi.gemm",
            "abi.elementwise",
            "abi.reduce",
            "abi.dte_send",
            "abi.dte_recv",
            "abi.dte_wait",
        ],
        "tests": "ABI",
    },
    "Compute": {
        "td": "ComputeOps.td",
        "cpp": "ComputeOps.cpp",
        "mnemonics": ["compute.gemm", "compute.elementwise", "compute.reduce"],
        "tests": "Compute",
    },
    "Comm": {
        "td": "CommOps.td",
        "cpp": "CommOps.cpp",
        "mnemonics": [
            "comm.send",
            "comm.recv",
            "comm.wait",
            "comm.all_gather",
            "comm.reduce_scatter",
            "comm.all_reduce",
        ],
        "tests": "Comm",
    },
    "TensorCollective": {
        "td": "TensorCollectiveOps.td",
        "cpp": "TensorCollectiveOps.cpp",
        "mnemonics": [
            "tensor_collective.all_gather",
            "tensor_collective.all_reduce",
            "tensor_collective.reduce_scatter",
            "tensor_collective.all_to_all",
            "tensor_collective.collective_permute",
            "tensor_collective.yield",
        ],
        "tests": "TensorCollective",
    },
    "Sync": {
        "td": "SyncOps.td",
        "cpp": "SyncOps.cpp",
        "mnemonics": ["sync.local_drain"],
        "tests": "Sync",
    },
    "Launch": {
        "td": "LaunchOps.td",
        "cpp": "LaunchOps.cpp",
        "mnemonics": ["launch"],
        "tests": "Launch",
    },
}

SUPPORT_TEST_DIRS = {"Attrs", "Types"}
FORBIDDEN_IR_STRINGS = (
    "abi.rdma_" + "1d",
    "abi.wdma_" + "1d",
    "AbiRdma" + "1DOp",
    "AbiWdma" + "1DOp",
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
        include = f'include "Wafer/IR/Ops/{spec["td"]}"'
        if include not in text:
            fail(errors, f"WaferOps.td missing {include}")


def check_op_family_files(root: Path, errors: list[str]) -> None:
    td_root = root / "include/Wafer/IR/Ops"
    cpp_root = root / "lib/Wafer/IR/Ops"
    dialect_cpp = check_file(root / "lib/Wafer/IR/WaferDialect.cpp", errors)

    for name, spec in OP_FAMILIES.items():
        td_text = check_file(td_root / spec["td"], errors)
        cpp_text = check_file(cpp_root / spec["cpp"], errors)
        for mnemonic in spec["mnemonics"]:
            if mnemonic not in td_text:
                fail(errors, f"{spec['td']} missing op mnemonic {mnemonic}")

        if name != "Sync" and "::verify()" not in cpp_text:
            fail(errors, f"{spec['cpp']} must own {name} verifier definitions")

    if "::verify()" in dialect_cpp.replace("TileBufferType::verify()", ""):
        fail(errors, "WaferDialect.cpp must not own op verifier definitions")

    check_file(cpp_root / "OpVerifierUtils.h", errors)
    check_file(cpp_root / "OpVerifierUtils.cpp", errors)


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
        if path.is_dir() and path.name not in allowed_dirs:
            fail(errors, f"unexpected Wafer dialect test directory: {path}")


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
                if forbidden in text:
                    fail(errors, f"{path} contains forbidden IR specialization {forbidden}")


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

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Wafer IR organization checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
