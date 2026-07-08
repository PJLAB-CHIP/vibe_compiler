#!/usr/bin/env python3
"""Check Wafer target CRT symbol closure.

The current target CRT contract is declared in tasks/14 and materialized by the
repo-local CRT header/source. This checker intentionally treats Direct DTE as
outside the production CRT closure until the ABI is extended.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


SYMBOL_RE = re.compile(r"\bwafer_tx81_[A-Za-z0-9_]+\b")
PROTOTYPE_RE = re.compile(
    r"\bvoid\s+(wafer_tx81_[A-Za-z0-9_]+)\s*\(([^;{}]*)\)\s*;",
    re.MULTILINE | re.DOTALL,
)

EXCLUDED_SYMBOLS = {
    "wafer_tx81_dte_send",
    "wafer_tx81_dte_recv",
    "wafer_tx81_dte_wait",
}

LOWERING_FAMILY_MARKERS = {
    "wafer_tx81_rdma": 'makeTargetSymbol("rdma")',
    "wafer_tx81_wdma": 'makeTargetSymbol("wdma")',
    "wafer_tx81_gather_scatter": 'makeTargetSymbol("gather_scatter")',
    "wafer_tx81_memset": 'makeTargetSymbol("memset")',
    "wafer_tx81_bit2fp": 'makeTargetSymbol("bit2fp")',
    "wafer_tx81_mask_move": 'makeTargetSymbol("mask_move")',
    "wafer_tx81_gemm": 'makeTargetSymbol("gemm")',
    "wafer_tx81_tdma_pad": 'makeTargetSymbol("tdma_pad")',
    "wafer_tx81_tdma_img2col": 'makeTargetSymbol("tdma_img2col")',
    "wafer_tx81_local_fence": 'makeTargetSymbol("local_fence")',
    "wafer_tx81_elementwise_": 'makeTargetSymbol("elementwise",',
    "wafer_tx81_reduce_": 'makeTargetSymbol("reduce",',
    "wafer_tx81_convert_": 'makeTargetSymbol("convert",',
    "wafer_tx81_pool_": 'makeTargetSymbol("pool",',
    "wafer_tx81_unpool_": 'makeTargetSymbol("unpool",',
    "wafer_tx81_peripheral_": 'makeTargetSymbol("peripheral",',
    "wafer_tx81_conv": "makeConvSymbol",
    "wafer_tx81_depthwise_conv": "makeConvSymbol",
    "wafer_tx81_backward_conv": "makeConvSymbol",
}


def fail(message: str) -> None:
    raise ValueError(message)


def read_text(path: pathlib.Path) -> str:
    if not path.is_file():
        fail(f"required file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def production_symbols_from_design(design_text: str) -> set[str]:
    match = re.search(
        r"以下 symbol 是当前 Q2-Q3 要实现并链接闭合的 production CRT closure"
        r".*?### 7\.5 Wrapper mapping",
        design_text,
        re.DOTALL,
    )
    if not match:
        fail("cannot find Q2-Q3 production CRT closure block in tasks/14")
    symbols = {
        line.strip()
        for line in match.group(0).splitlines()
        if SYMBOL_RE.fullmatch(line.strip())
    }
    symbols -= EXCLUDED_SYMBOLS
    if not symbols:
        fail("no production wafer_tx81_* symbols found in tasks/14")
    return symbols


def parse_prototypes(header_text: str) -> dict[str, str]:
    prototypes: dict[str, str] = {}
    for match in PROTOTYPE_RE.finditer(header_text):
        params = " ".join(match.group(2).split())
        prototypes[match.group(1)] = params
    if not prototypes:
        fail("no wafer_tx81_* prototypes found in CRT header")
    return prototypes


def check_no_excluded_symbols(path: pathlib.Path, text: str) -> None:
    present = sorted(EXCLUDED_SYMBOLS & set(SYMBOL_RE.findall(text)))
    if present:
        fail(f"{path} contains ABI-incomplete Direct DTE symbols: {', '.join(present)}")


def check_lowering_markers(symbols: set[str], lowering_text: str) -> None:
    missing_markers: list[str] = []
    for symbol in sorted(symbols):
        marker = None
        for prefix, candidate in LOWERING_FAMILY_MARKERS.items():
            if symbol == prefix or symbol.startswith(prefix):
                marker = candidate
                break
        if marker is None:
            missing_markers.append(f"{symbol}: no lowering marker rule")
            continue
        if marker not in lowering_text:
            missing_markers.append(f"{symbol}: missing {marker}")
    if missing_markers:
        fail("lowering is missing production CRT symbol families:\n  " + "\n  ".join(missing_markers))


def check_defined_symbols(symbols: set[str], nm_text: str) -> None:
    defined_symbols = set(SYMBOL_RE.findall(nm_text))
    missing = sorted(symbols - defined_symbols)
    excluded = sorted(EXCLUDED_SYMBOLS & defined_symbols)
    if missing:
        fail("missing CRT object definitions: " + ", ".join(missing))
    if excluded:
        fail("CRT object defines ABI-incomplete Direct DTE symbols: " + ", ".join(excluded))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--defined-symbols-stdin",
        action="store_true",
        help="also validate wafer_tx81_* definitions from nm text on stdin",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    design_path = repo_root / "tasks" / "14-target-llvm-golden-packet.md"
    header_path = repo_root / "runtime" / "wafer_crt" / "include" / "wafer_tx81_crt.h"
    source_path = repo_root / "runtime" / "wafer_crt" / "src" / "wafer_tx81_crt.c"
    lowering_path = (
        repo_root / "lib" / "Wafer" / "Transforms" / "Target" / "LowerInstrToTargetLLVM.cpp"
    )

    design_text = read_text(design_path)
    header_text = read_text(header_path)
    source_text = read_text(source_path)
    lowering_text = read_text(lowering_path)

    production_symbols = production_symbols_from_design(design_text)
    prototypes = parse_prototypes(header_text)
    source_symbols = set(SYMBOL_RE.findall(source_text))

    check_no_excluded_symbols(header_path, header_text)
    check_no_excluded_symbols(source_path, source_text)

    header_symbols = set(prototypes)
    missing_header = sorted(production_symbols - header_symbols)
    extra_header = sorted((header_symbols - production_symbols) - EXCLUDED_SYMBOLS)
    missing_source = sorted(production_symbols - source_symbols)

    errors: list[str] = []
    if missing_header:
        errors.append("missing CRT header prototypes: " + ", ".join(missing_header))
    if extra_header:
        errors.append("extra CRT header prototypes outside production closure: " + ", ".join(extra_header))
    if missing_source:
        errors.append("missing CRT source symbol references: " + ", ".join(missing_source))
    if len(production_symbols) != 105:
        errors.append(f"production closure symbol count is {len(production_symbols)}, expected 105")
    if errors:
        fail("\n".join(errors))

    check_lowering_markers(production_symbols, lowering_text)
    if args.defined_symbols_stdin:
        check_defined_symbols(production_symbols, sys.stdin.read())
    print(f"checked {len(production_symbols)} production wafer_tx81_* CRT symbols")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
