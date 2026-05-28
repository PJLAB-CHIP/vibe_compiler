#!/usr/bin/env python3
"""Check stage-connection tests exercise connected compiler boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CAST_RE = re.compile(r"\b(?:builtin\.)?unrealized_conversion_cast\b")
RUN_RE = re.compile(r"^\s*//\s*RUN:\s*(.*)$", re.MULTILINE)

PUBLIC_SOURCE_MARKERS = (
    "linalg.matmul",
    "linalg.generic",
    "linalg.reduce",
    "stablehlo.",
)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def strip_line_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(line.split("//", 1)[0])
    return "\n".join(lines)


def check_file(path: Path, errors: list[str]) -> None:
    text = path.read_text()
    input_text = strip_line_comments(text)
    if CAST_RE.search(input_text):
        fail(errors, f"{path}: stage-connection tests must not use unrealized_conversion_cast")

    if not any(marker in input_text for marker in PUBLIC_SOURCE_MARKERS):
        fail(errors, f"{path}: expected at least one public upstream op source")

    run_lines = RUN_RE.findall(text)
    if not run_lines:
        fail(errors, f"{path}: missing RUN lines")
        return

    filecheck_runs = [line for line in run_lines if "FileCheck" in line]
    for line in filecheck_runs:
        if "--implicit-check-not=unrealized_conversion_cast" not in line:
            fail(errors, f"{path}: FileCheck RUN must reject unrealized_conversion_cast: {line}")

    has_group_gate = any(
        "--wafer-form-groups" in line
        and "--wafer-materialize-single-tile" not in line
        and "--wafer-lower-tile-region-to-c-abi" not in line
        for line in run_lines
    )
    has_tile_gate = any(
        "--wafer-form-groups" in line
        and "--wafer-materialize-single-tile" in line
        and "--wafer-lower-tile-region-to-c-abi" not in line
        for line in run_lines
    )
    has_abi_gate = any(
        "--wafer-form-groups" in line
        and "--wafer-materialize-single-tile" in line
        and "--wafer-lower-tile-region-to-c-abi" in line
        for line in run_lines
    )

    if not has_group_gate:
        fail(errors, f"{path}: missing source-to-group stage gate")
    if not has_tile_gate:
        fail(errors, f"{path}: missing group-to-tile stage gate")
    if not has_abi_gate:
        fail(errors, f"{path}: missing tile-to-ABI stage gate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    stage_root = root / "test" / "StageConnections"
    errors: list[str] = []

    if not stage_root.is_dir():
        fail(errors, f"missing stage-connection test directory: {stage_root}")
    else:
        tests = sorted(stage_root.glob("*.mlir"))
        if not tests:
            fail(errors, f"stage-connection test directory is empty: {stage_root}")
        for path in tests:
            check_file(path, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Wafer stage-connection test checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
