#!/usr/bin/env python3
"""Reject oneDNN or bulk-adapter closure in a feature-off base binary."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


FORBIDDEN_DYNAMIC_TOKENS = ("libdnnl", "libonednn")
FORBIDDEN_RAW_PREFIXES = ("dnnl_",)
FORBIDDEN_ADAPTER_SYMBOLS = (
    "executeQualifiedBulkTensorNumeric",
    "createManagedBulkExecutionEnvironment",
    "calibrateBulkBackend",
    "validateBulkBackend",
)


def existing_file(value: str) -> pathlib.Path:
    path = pathlib.Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=existing_file)
    parser.add_argument("--readelf", required=True, type=existing_file)
    parser.add_argument("--nm", required=True, type=existing_file)
    return parser.parse_args()


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}\n"
            f"{result.stdout}"
        )
    return result.stdout


def main() -> int:
    args = parse_args()
    try:
        dynamic = run([str(args.readelf), "-d", str(args.binary)]).lower()
        dynamic_hits = [token for token in FORBIDDEN_DYNAMIC_TOKENS if token in dynamic]
        raw_symbols = run([str(args.nm), "-g", str(args.binary)])
        raw_hits = sorted(
            {
                fields[-1].split("@", 1)[0]
                for line in raw_symbols.splitlines()
                if (fields := line.split())
                and fields[-1].split("@", 1)[0].startswith(FORBIDDEN_RAW_PREFIXES)
            }
        )
        demangled = run([str(args.nm), "-g", "-C", str(args.binary)])
        adapter_hits = [name for name in FORBIDDEN_ADAPTER_SYMBOLS if name in demangled]
        if dynamic_hits or raw_hits or adapter_hits:
            raise RuntimeError(
                "base binary contains managed bulk link closure: "
                f"dynamic={dynamic_hits}, raw={raw_hits}, adapters={adapter_hits}"
            )
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("base bulk link closure passed: no oneDNN dependency or bulk adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
