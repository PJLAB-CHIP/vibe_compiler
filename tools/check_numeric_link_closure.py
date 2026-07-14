#!/usr/bin/env python3
"""Reject managed numeric libraries or adapter symbols in a base test binary."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


FORBIDDEN_DYNAMIC_TOKENS = (
    "libmpfr",
    "libgmp",
    "libsoftfloat",
    "libtestfloat",
)

FORBIDDEN_RAW_SYMBOL = re.compile(
    r"^(?:"
    r"mpfr_"
    r"|__gmp"
    r"|softfloat_"
    r"|(?:f16|f32|f64|f128|extF80)(?:M)?_"
    r"|(?:ui32|ui64|i32|i64)_to_(?:f16|f32|f64|f128|extF80)"
    r")"
)

FORBIDDEN_ADAPTER_SYMBOLS = (
    "executeMPFRFormal",
    "executeSoftFloatOracle",
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


def run_tool(command: list[str]) -> str:
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


def check_dynamic_section(args: argparse.Namespace) -> None:
    dynamic = run_tool([str(args.readelf), "-d", str(args.binary)])
    lowered = dynamic.lower()
    present = [token for token in FORBIDDEN_DYNAMIC_TOKENS if token in lowered]
    if present:
        raise RuntimeError(
            "base binary dynamic dependency closure contains managed numeric "
            f"libraries: {', '.join(present)}\n{dynamic}"
        )


def parse_nm_symbol(line: str) -> str | None:
    fields = line.split()
    if not fields:
        return None
    symbol = fields[-1].split("@", 1)[0]
    return symbol


def check_symbols(args: argparse.Namespace) -> None:
    raw_output = run_tool([str(args.nm), "-g", str(args.binary)])
    forbidden_raw = sorted(
        {
            symbol
            for line in raw_output.splitlines()
            if (symbol := parse_nm_symbol(line)) is not None
            and FORBIDDEN_RAW_SYMBOL.match(symbol)
        }
    )
    demangled_output = run_tool([str(args.nm), "-g", "-C", str(args.binary)])
    forbidden_adapters = [
        symbol for symbol in FORBIDDEN_ADAPTER_SYMBOLS if symbol in demangled_output
    ]
    if forbidden_raw or forbidden_adapters:
        details = []
        if forbidden_raw:
            details.append("external symbols=" + ", ".join(forbidden_raw))
        if forbidden_adapters:
            details.append("adapter symbols=" + ", ".join(forbidden_adapters))
        raise RuntimeError(
            "base binary contains managed numeric link closure: " + "; ".join(details)
        )


def main() -> int:
    args = parse_args()
    try:
        check_dynamic_section(args)
        check_symbols(args)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "base numeric link closure passed: no MPFR, GMP, SoftFloat, or TestFloat dependency"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
