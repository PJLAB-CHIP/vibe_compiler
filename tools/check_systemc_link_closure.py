#!/usr/bin/env python3
"""Reject SystemC closure in a feature-off base test binary."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


FORBIDDEN_DYNAMIC_TOKENS = ("libsystemc",)
FORBIDDEN_DEMANGLED_TOKENS = ("sc_core::", "sc_dt::", "sc_main")


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
        demangled = run([str(args.nm), "-g", "-C", str(args.binary)])
        symbol_hits = [token for token in FORBIDDEN_DEMANGLED_TOKENS if token in demangled]
        if dynamic_hits or symbol_hits:
            raise RuntimeError(
                "base binary contains SystemC link closure: "
                f"dynamic={dynamic_hits}, symbols={symbol_hits}"
            )
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("base SystemC link closure passed: no SystemC dependency or symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
