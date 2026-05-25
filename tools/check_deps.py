#!/usr/bin/env python3
"""Check Wafer dependency pins and importer/backend isolation."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


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

    stablehlo_head = rev_parse(REPO_ROOT / ".deps" / "src" / "stablehlo")
    if stablehlo_head and stablehlo_head != versions["WAFER_STABLEHLO_COMMIT"]:
        raise RuntimeError(f"StableHLO checkout mismatch: {stablehlo_head}")

    shardy_head = rev_parse(REPO_ROOT / ".deps" / "src" / "shardy")
    if shardy_head and shardy_head != versions["WAFER_SHARDY_COMMIT"]:
        raise RuntimeError(f"Shardy checkout mismatch: {shardy_head}")

    print("dependency consistency checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
