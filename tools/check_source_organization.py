#!/usr/bin/env python3
"""Check stable Wafer production-source ownership boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


INSTRUCTION_FAMILY_SOURCES = (
    "ComputeOps.cpp",
    "DTEOps.cpp",
    "MovementOps.cpp",
    "PeripheralOps.cpp",
    "SyncOps.cpp",
)
TILE_REGION_TO_INSTR_SOURCES = (
    "CollectiveLowering.cpp",
    "ComputeLowering.cpp",
    "MovementLowering.cpp",
    "MovementSupport.cpp",
    "WaferTileRegionToInstr.cpp",
)
NUMERIC_SEMANTICS_SOURCES = (
    "NumericCapability.cpp",
    "NumericCommand.cpp",
    "NumericProfiles.cpp",
    "NumericSemanticsInternal.cpp",
)
INSTRUCTION_VERIFIER_FILES = (
    "InstructionVerifierUtils.cpp",
    "InstructionVerifierUtils.h",
)
LIB_WAFER_SUBDIRECTORY_ORDER = (
    "IR",
    "Target",
    "Frontend",
    "Analysis",
    "Conversion",
    "Transforms",
    "Pipelines",
    "Runtime",
    "Compiler",
    "Model",
)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_required(path: Path, errors: list[str]) -> str:
    if not path.is_file():
        fail(errors, f"missing required file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def cmake_target_body(
    text: str, command: str, target: str, cmake_path: Path, errors: list[str]
) -> str:
    match = re.search(
        rf"\b{re.escape(command)}\s*\(\s*{re.escape(target)}\b", text
    )
    if not match:
        fail(errors, f"{cmake_path} missing {command}({target} ...)")
        return ""

    opening = text.find("(", match.start())
    depth = 0
    in_quote = False
    escaped = False
    for index in range(opening, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]

    fail(errors, f"unterminated {command}({target} ...) in {cmake_path}")
    return ""


def check_cmake_sources(
    *,
    body: str,
    required: tuple[str, ...],
    cmake_path: Path,
    target: str,
    prefix: str = "",
    forbidden: tuple[str, ...] = (),
    errors: list[str],
) -> None:
    tokens = set(body.split())
    for source in required:
        entry = prefix + source
        if entry not in tokens:
            fail(errors, f"{cmake_path}: {target} missing source {entry}")
    for source in forbidden:
        entry = prefix + source
        if entry in tokens:
            fail(errors, f"{cmake_path}: {target} still lists legacy source {entry}")


def check_instruction_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/IR/Instr"
    cmake_path = root / "lib/Wafer/IR/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    actual_family_sources = {
        path.name for path in source_root.glob("*Ops.cpp") if path.is_file()
    }
    expected_family_sources = set(INSTRUCTION_FAMILY_SOURCES)
    if actual_family_sources != expected_family_sources:
        missing = sorted(expected_family_sources - actual_family_sources)
        unexpected = sorted(actual_family_sources - expected_family_sources)
        if missing:
            fail(errors, f"instruction family sources missing: {', '.join(missing)}")
        if unexpected:
            fail(
                errors,
                f"unexpected instruction family sources: {', '.join(unexpected)}",
            )

    for filename in INSTRUCTION_VERIFIER_FILES:
        read_required(source_root / filename, errors)

    public_internal_header = (
        root / "include/Wafer/IR/Instr/InstructionVerifierUtils.h"
    )
    if public_internal_header.exists():
        fail(
            errors,
            "instruction verifier helper must remain library-private: "
            f"{public_internal_header}",
        )

    legacy_source = source_root / "InstructionOps.cpp"
    if legacy_source.exists():
        fail(errors, f"legacy instruction aggregate must be removed: {legacy_source}")

    target_body = cmake_target_body(
        cmake_text, "add_mlir_dialect_library", "WaferIR", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=(*INSTRUCTION_FAMILY_SOURCES, "InstructionVerifierUtils.cpp"),
        forbidden=("InstructionOps.cpp",),
        prefix="Instr/",
        cmake_path=cmake_path,
        target="WaferIR",
        errors=errors,
    )


def check_tile_region_to_instr_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Conversion/WaferTileRegionToInstr"
    cmake_path = root / "lib/Wafer/Conversion/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    for filename in (*TILE_REGION_TO_INSTR_SOURCES, "Internal.h"):
        read_required(source_root / filename, errors)

    actual_sources = {
        path.name for path in source_root.glob("*.cpp") if path.is_file()
    }
    expected_sources = set(TILE_REGION_TO_INSTR_SOURCES)
    if actual_sources != expected_sources:
        missing = sorted(expected_sources - actual_sources)
        unexpected = sorted(actual_sources - expected_sources)
        if missing:
            fail(errors, f"tile-region-to-instr sources missing: {', '.join(missing)}")
        if unexpected:
            fail(
                errors,
                "unexpected tile-region-to-instr sources: " + ", ".join(unexpected),
            )

    target_body = cmake_target_body(
        cmake_text,
        "add_mlir_conversion_library",
        "WaferTileRegionToInstr",
        cmake_path,
        errors,
    )
    check_cmake_sources(
        body=target_body,
        required=TILE_REGION_TO_INSTR_SOURCES,
        prefix="WaferTileRegionToInstr/",
        cmake_path=cmake_path,
        target="WaferTileRegionToInstr",
        errors=errors,
    )

    facade_path = source_root / "WaferTileRegionToInstr.cpp"
    facade_text = read_required(facade_path, errors)
    concrete_pattern = re.compile(
        r"\b(?:class|struct)\s+[A-Za-z_][A-Za-z0-9_]*"
        r"\s*(?:final\s*)?:[^;{]*"
        r"(?:OpRewritePattern|OpConversionPattern|RewritePattern|ConversionPattern)",
        flags=re.DOTALL,
    )
    if concrete_pattern.search(facade_text):
        fail(
            errors,
            f"{facade_path} must orchestrate conversion, not define concrete patterns",
        )


def check_numeric_semantics_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Target"
    cmake_path = source_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    for filename in (*NUMERIC_SEMANTICS_SOURCES, "NumericSemanticsInternal.h"):
        read_required(source_root / filename, errors)

    legacy_source = source_root / "NumericSemantics.cpp"
    if legacy_source.exists():
        fail(errors, f"legacy numeric aggregate must be removed: {legacy_source}")

    public_internal_header = root / "include/Wafer/Target/NumericSemanticsInternal.h"
    if public_internal_header.exists():
        fail(
            errors,
            "numeric internal header must remain library-private: "
            f"{public_internal_header}",
        )

    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTarget", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=NUMERIC_SEMANTICS_SOURCES,
        forbidden=("NumericSemantics.cpp",),
        cmake_path=cmake_path,
        target="WaferTarget",
        errors=errors,
    )


def check_lib_wafer_dependency_order(root: Path, errors: list[str]) -> None:
    cmake_path = root / "lib/Wafer/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)
    subdirectories = re.findall(
        r"(?m)^\s*add_subdirectory\(\s*([^\s\)]+)\s*\)", cmake_text
    )

    positions: dict[str, int] = {}
    for expected in LIB_WAFER_SUBDIRECTORY_ORDER:
        occurrences = [
            index
            for index, subdirectory in enumerate(subdirectories)
            if subdirectory == expected
        ]
        if not occurrences:
            fail(errors, f"{cmake_path} missing add_subdirectory({expected})")
        elif len(occurrences) > 1:
            fail(errors, f"{cmake_path} adds subdirectory {expected} more than once")
        else:
            positions[expected] = occurrences[0]

    present_order = [
        name for name in LIB_WAFER_SUBDIRECTORY_ORDER if name in positions
    ]
    if any(
        positions[left] >= positions[right]
        for left, right in zip(present_order, present_order[1:])
    ):
        actual = sorted(present_order, key=positions.get)
        fail(
            errors,
            f"{cmake_path} dependency order must be "
            f"{' -> '.join(LIB_WAFER_SUBDIRECTORY_ORDER)}; found "
            f"{' -> '.join(actual)}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    errors: list[str] = []
    check_instruction_owners(root, errors)
    check_tile_region_to_instr_owners(root, errors)
    check_numeric_semantics_owners(root, errors)
    check_lib_wafer_dependency_order(root, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Wafer source organization checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
