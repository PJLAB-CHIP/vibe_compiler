#!/usr/bin/env python3
"""Check Wafer target CRT symbol closure from production code facts.

The compiler lowering owns the set of target symbol families, Wafer TableGen
enums own dynamic symbol suffixes, and the repo-local CRT header/source must
exactly implement the resulting production surface. Target-illegal operations
must not appear in the lowering-derived symbol registry.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


SYMBOL_RE = re.compile(r"\bwafer_tx81_[A-Za-z0-9_]+\b")
SOURCE_SYMBOL_RE = re.compile(
    r"\b(wafer_tx81_[A-Za-z0-9_]+)\b(?=\s*[\(,])"
)
PROTOTYPE_RE = re.compile(
    r"\b(?:void|uint64_t)\s+(wafer_tx81_[A-Za-z0-9_]+)\s*\(([^;{}]*)\)\s*;",
    re.MULTILINE | re.DOTALL,
)
STATIC_TARGET_SYMBOL_RE = re.compile(
    r'\bmakeTargetSymbol\(\s*"([a-z0-9_]+)"\s*\)'
)
DYNAMIC_TARGET_SYMBOL_RE = re.compile(
    r'\bmakeTargetSymbol\(\s*"([a-z0-9_]+)"\s*,'
)
ENUM_CASE_RE = re.compile(
    r'\bdef\s+(Wafer_[A-Za-z0-9_]+)\s*:\s*I32EnumAttrCase<'
    r'\s*"[^"]+"\s*,\s*-?[0-9]+\s*,\s*"([a-z0-9_]+)"\s*>\s*;',
    re.MULTILINE | re.DOTALL,
)

DYNAMIC_SYMBOL_ENUMS = {
    "elementwise": "InstrElementwiseKind",
    "reduce": "InstrReduceKind",
    "convert": "InstrConvertKind",
    "pool": "InstrPoolKind",
    "unpool": "InstrUnpoolKind",
    "peripheral": "InstrPeripheralKind",
}

TARGET_ILLEGAL_SYMBOLS = {
    "wafer_tx81_peripheral_factorize": "peripheral factorize lacks a",
}

VERIFIER_REJECTED_SYMBOLS = {
    "wafer_tx81_peripheral_count":
        "count peripheral writeback is not represented in instruction IR",
}


def fail(message: str) -> None:
    raise ValueError(message)


def read_text(path: pathlib.Path) -> str:
    if not path.is_file():
        fail(f"required file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def parse_enum_spellings(attrs_text: str, enum_name: str) -> set[str]:
    match = re.search(
        rf'\bI32EnumAttr<\s*"{re.escape(enum_name)}"\s*,.*?'
        r'\[(.*?)\]\s*>\s*\{',
        attrs_text,
        re.DOTALL,
    )
    if not match:
        fail(f"cannot find TableGen enum {enum_name}")

    case_spellings = dict(ENUM_CASE_RE.findall(attrs_text))
    case_names = re.findall(r"\bWafer_[A-Za-z0-9_]+\b", match.group(1))
    missing_cases = sorted(set(case_names) - set(case_spellings))
    if missing_cases:
        fail(
            f"TableGen enum {enum_name} references unresolved cases: "
            + ", ".join(missing_cases)
        )
    spellings = {case_spellings[name] for name in case_names}
    if not spellings:
        fail(f"TableGen enum {enum_name} has no cases")
    if len(spellings) != len(case_names):
        fail(f"TableGen enum {enum_name} has duplicate symbol spellings")
    return spellings


def potential_symbols_from_lowering(
    lowering_text: str, attrs_text: str
) -> set[str]:
    static_bases = set(STATIC_TARGET_SYMBOL_RE.findall(lowering_text))
    dynamic_bases = set(DYNAMIC_TARGET_SYMBOL_RE.findall(lowering_text))
    if not static_bases:
        fail("no literal makeTargetSymbol calls found in target lowering")

    unknown_dynamic = sorted(dynamic_bases - set(DYNAMIC_SYMBOL_ENUMS))
    stale_registry = sorted(set(DYNAMIC_SYMBOL_ENUMS) - dynamic_bases)
    if unknown_dynamic:
        fail(
            "dynamic target symbol families have no enum registry: "
            + ", ".join(unknown_dynamic)
        )
    if stale_registry:
        fail(
            "dynamic target symbol enum registry has no lowering call: "
            + ", ".join(stale_registry)
        )

    symbols = {f"wafer_tx81_{base}" for base in static_bases}
    for base, enum_name in DYNAMIC_SYMBOL_ENUMS.items():
        symbols.update(
            f"wafer_tx81_{base}_{spelling}"
            for spelling in parse_enum_spellings(attrs_text, enum_name)
        )
    return symbols


def production_symbols_from_code(
    lowering_text: str, attrs_text: str, instruction_ops_text: str
) -> set[str]:
    potential_symbols = potential_symbols_from_lowering(lowering_text, attrs_text)
    non_production_symbols = set(TARGET_ILLEGAL_SYMBOLS) | set(
        VERIFIER_REJECTED_SYMBOLS
    )
    stale_exclusions = sorted(non_production_symbols - potential_symbols)
    if stale_exclusions:
        fail(
            "non-production target symbol registry is stale: "
            + ", ".join(stale_exclusions)
        )
    for symbol, verifier_marker in VERIFIER_REJECTED_SYMBOLS.items():
        if verifier_marker not in instruction_ops_text:
            fail(f"{symbol} is not proven unreachable by the instruction verifier")
    for symbol, target_marker in TARGET_ILLEGAL_SYMBOLS.items():
        if target_marker not in lowering_text:
            fail(f"{symbol} is not proven unreachable by target lowering")
    return potential_symbols - non_production_symbols


def parse_prototypes(header_text: str) -> dict[str, str]:
    prototypes: dict[str, str] = {}
    for match in PROTOTYPE_RE.finditer(header_text):
        params = " ".join(match.group(2).split())
        prototypes[match.group(1)] = params
    if not prototypes:
        fail("no wafer_tx81_* prototypes found in CRT header")
    return prototypes


def check_no_non_production_symbols(path: pathlib.Path, text: str) -> None:
    non_production_symbols = set(TARGET_ILLEGAL_SYMBOLS) | set(
        VERIFIER_REJECTED_SYMBOLS
    )
    present = sorted(non_production_symbols & set(SYMBOL_RE.findall(text)))
    if present:
        fail(f"{path} contains non-production CRT symbols: {', '.join(present)}")


def check_defined_symbols(symbols: set[str], nm_text: str) -> None:
    defined_symbols = set(SYMBOL_RE.findall(nm_text))
    missing = sorted(symbols - defined_symbols)
    extra = sorted(defined_symbols - symbols)
    if missing:
        fail("missing CRT object definitions: " + ", ".join(missing))
    if extra:
        fail("extra CRT object definitions outside production closure: " + ", ".join(extra))


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
    header_path = repo_root / "runtime" / "wafer_crt" / "include" / "wafer_tx81_crt.h"
    source_path = repo_root / "runtime" / "wafer_crt" / "src" / "wafer_tx81_crt.c"
    attrs_path = repo_root / "include" / "Wafer" / "IR" / "WaferAttrs.td"
    instruction_ops_path = (
        repo_root / "lib" / "Wafer" / "IR" / "Instr" / "InstructionOps.cpp"
    )
    lowering_path = (
        repo_root / "lib" / "Wafer" / "Transforms" / "Target" / "LowerInstrToTargetLLVM.cpp"
    )

    header_text = read_text(header_path)
    source_text = read_text(source_path)
    attrs_text = read_text(attrs_path)
    instruction_ops_text = read_text(instruction_ops_path)
    lowering_text = read_text(lowering_path)

    production_symbols = production_symbols_from_code(
        lowering_text, attrs_text, instruction_ops_text
    )
    prototypes = parse_prototypes(header_text)
    # Macro-generated CRT definitions keep their concrete public symbol as a
    # source token.  Header exactness plus the compiled-object nm gate below
    # distinguish a real definition from an accidental source-only mention.
    source_symbols = set(SOURCE_SYMBOL_RE.findall(source_text))

    check_no_non_production_symbols(header_path, header_text)
    check_no_non_production_symbols(source_path, source_text)

    header_symbols = set(prototypes)
    missing_header = sorted(production_symbols - header_symbols)
    extra_header = sorted(header_symbols - production_symbols)
    missing_source = sorted(production_symbols - source_symbols)
    extra_source = sorted(source_symbols - production_symbols)

    errors: list[str] = []
    if missing_header:
        errors.append("missing CRT header prototypes: " + ", ".join(missing_header))
    if extra_header:
        errors.append("extra CRT header prototypes outside production closure: " + ", ".join(extra_header))
    if missing_source:
        errors.append("missing CRT source definitions: " + ", ".join(missing_source))
    if extra_source:
        errors.append("extra CRT source definitions outside production closure: " + ", ".join(extra_source))
    if errors:
        fail("\n".join(errors))

    if args.defined_symbols_stdin:
        check_defined_symbols(production_symbols, sys.stdin.read())
    print(
        f"checked {len(production_symbols)} production wafer_tx81_* CRT symbols "
        "from target lowering and Wafer enum registry"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
