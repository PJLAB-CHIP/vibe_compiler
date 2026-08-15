#!/usr/bin/env python3
"""Verify the feature-on tool uses static oneDNN without a pool runtime."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


FORBIDDEN_DYNAMIC_TOKENS = (
    "libdnnl",
    "libonednn",
    "libgomp",
    "libomp",
    "libiomp",
    "libtbb",
)
REQUIRED_RAW_SYMBOLS = ("dnnl_version", "dnnl_primitive_execute")
REQUIRED_ADAPTER_SYMBOLS = (
    "createManagedBulkExecutionEnvironment",
    "executeQualifiedBulkTensorNumeric",
    "validateBulkBackend",
)


def existing_file(value: str) -> pathlib.Path:
    path = pathlib.Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=existing_file)
    parser.add_argument("--readelf", required=True, type=existing_file)
    parser.add_argument("--nm", required=True, type=existing_file)
    args = parser.parse_args()
    try:
        dynamic = run([str(args.readelf), "-d", str(args.binary)]).lower()
        forbidden = [token for token in FORBIDDEN_DYNAMIC_TOKENS if token in dynamic]
        # oneDNN's static build hides its C API symbols after final link; local
        # text symbols still prove that the managed archive was incorporated.
        raw = run([str(args.nm), str(args.binary)])
        missing_raw = [name for name in REQUIRED_RAW_SYMBOLS if name not in raw]
        demangled = run([str(args.nm), "-g", "-C", str(args.binary)])
        missing_adapters = [
            name for name in REQUIRED_ADAPTER_SYMBOLS if name not in demangled
        ]
        if forbidden or missing_raw or missing_adapters:
            raise RuntimeError(
                "bulk feature-on link closure mismatch: "
                f"forbidden dynamic={forbidden}, missing raw={missing_raw}, "
                f"missing adapters={missing_adapters}"
            )
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("bulk feature-on link closure passed: static oneDNN, SEQ runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
