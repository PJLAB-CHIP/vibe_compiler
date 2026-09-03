#!/usr/bin/env python3
"""Check Wafer target CRT symbol closure from production code facts.

The shared typed target-call registry owns the exact production surface,
Wafer TableGen enums own dynamic semantic spellings, and the repo-local CRT
header/source must exactly implement it. Target lowering must consume the
registry and must not retain a second symbol-construction path.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


TARGET_LOWERING_SOURCES = (
    "LowerInstrToTargetLLVM.cpp",
    "TargetLoweringVerification.cpp",
    "TargetCallLoweringSupport.cpp",
    "MovementTargetCallLowering.cpp",
    "ComputeTargetCallLowering.cpp",
    "DirectDTETargetCallLowering.cpp",
    "PeripheralTargetCallLowering.cpp",
    "SyncTargetCallLowering.cpp",
    "InstructionTargetCallLowering.cpp",
    "TargetLLVMStructure.cpp",
    "TargetLLVMConversionPatterns.cpp",
    "TargetLLVMConversion.cpp",
)


SYMBOL_RE = re.compile(r"\bwafer_tx81_[A-Za-z0-9_]+\b")
SOURCE_SYMBOL_RE = re.compile(
    r"\b(wafer_tx81_[A-Za-z0-9_]+)\b(?=\s*[\(,])"
)
PROTOTYPE_RE = re.compile(
    r"\b(?:void|uint64_t)\s+(wafer_tx81_[A-Za-z0-9_]+)\s*\(([^;{}]*)\)\s*;",
    re.MULTILINE | re.DOTALL,
)
STATIC_REGISTRY_STEM_RE = re.compile(
    r'\badd(?:Void)?\(\s*"([a-z0-9_]+)"\s*,'
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


def parse_target_operation_spellings(
    target_operation_text: str, operation_type: str
) -> dict[str, str]:
    matches = re.findall(
        rf"case\s+{re.escape(operation_type)}::([A-Za-z0-9_]+):\s*"
        r'return\s+"([a-z0-9_]+)";',
        target_operation_text,
        re.DOTALL,
    )
    result = dict(matches)
    if not result:
        fail(f"cannot find {operation_type} canonical spellings")
    return result


def production_symbols_from_registry(
    registry_text: str,
    lowering_text: str,
    attrs_text: str,
    target_operation_text: str,
) -> set[str]:
    if '"Wafer/Target/TargetCall.h"' not in lowering_text:
        fail("target lowering does not include the shared target-call registry")
    if "makeTargetSymbol" in lowering_text:
        fail("target lowering retains the old symbol-construction path")

    static_bases = set(STATIC_REGISTRY_STEM_RE.findall(registry_text))
    if len(static_bases) != 20:
        fail(
            "target-call registry must contain 20 fixed call stems, found "
            f"{len(static_bases)}"
        )
    symbols = {f"wafer_tx81_{base}" for base in static_bases}
    if len(re.findall(r"\baddEnumSelectedCalls\(\s*\)\s*;", registry_text)) != 1:
        fail("target-call registry must instantiate each dynamic family once")
    for base, enum_name in DYNAMIC_SYMBOL_ENUMS.items():
        symbols.update(
            f"wafer_tx81_{base}_{spelling}"
            for spelling in parse_enum_spellings(attrs_text, enum_name)
        )

    conv_stems = set(
        parse_target_operation_spellings(
            target_operation_text, "TargetConvolutionOperation"
        ).values()
    )
    if len(conv_stems) != 3:
        fail(f"expected 3 convolution stems, found {sorted(conv_stems)}")
    symbols.update(f"wafer_tx81_{stem}" for stem in conv_stems)

    peripheral_spellings = parse_target_operation_spellings(
        target_operation_text, "TargetPeripheralOperation"
    )
    peripheral_kinds = set(
        re.findall(
            r"addPeripheral\(TargetPeripheralOperation::([A-Za-z0-9_]+)",
            registry_text,
        )
    )
    if len(peripheral_kinds) != 7:
        fail(
            "target-call registry must contain 7 admitted peripheral kinds, "
            f"found {sorted(peripheral_kinds)}"
        )
    missing_peripheral_spellings = sorted(
        peripheral_kinds - set(peripheral_spellings)
    )
    if missing_peripheral_spellings:
        fail(
            "peripheral registry kinds have no TableGen spelling: "
            + ", ".join(missing_peripheral_spellings)
        )
    symbols.update(
        f"wafer_tx81_peripheral_{peripheral_spellings[kind]}"
        for kind in peripheral_kinds
    )
    if len(symbols) != 114:
        fail(f"target-call registry must close 114 symbols, found {len(symbols)}")

    shared_symbols = {
        "wafer_tx81_ncc_join",
        "wafer_tx81_direct_dte_begin",
        "wafer_tx81_direct_dte_begin_after_prepare",
        "wafer_tx81_direct_dte_send_prepare",
        "wafer_tx81_direct_dte_multisend_prepare",
        "wafer_tx81_direct_dte_multisend_add_destination",
        "wafer_tx81_direct_dte_recv_prepare",
        "wafer_tx81_direct_dte_wait",
        "wafer_tx81_direct_dte_finish",
        "wafer_tx81_direct_dte_send_issue",
    }
    ordinary_symbols = {
        symbol
        for symbol in symbols
        if symbol not in shared_symbols
    }
    if len(ordinary_symbols) != 104:
        fail(
            "target-call registry must close 104 current ordinary "
            f"symbols, found {len(ordinary_symbols)}"
        )
    if "wafer_tx81_direct_dte_send_issue" not in symbols:
        fail("current ABI must expose the explicit Direct DTE issue call")
    return symbols


def production_symbols_from_code(
    registry_text: str,
    lowering_text: str,
    attrs_text: str,
    instruction_ops_text: str,
    target_operation_text: str,
) -> set[str]:
    production_symbols = production_symbols_from_registry(
        registry_text, lowering_text, attrs_text, target_operation_text
    )
    non_production_symbols = set(TARGET_ILLEGAL_SYMBOLS) | set(
        VERIFIER_REJECTED_SYMBOLS
    )
    accidentally_registered = sorted(non_production_symbols & production_symbols)
    if accidentally_registered:
        fail(
            "target-call registry contains non-production symbols: "
            + ", ".join(accidentally_registered)
        )
    for symbol, verifier_marker in VERIFIER_REJECTED_SYMBOLS.items():
        if verifier_marker not in instruction_ops_text:
            fail(f"{symbol} is not proven unreachable by the instruction verifier")
    for symbol, target_marker in TARGET_ILLEGAL_SYMBOLS.items():
        if target_marker not in lowering_text:
            fail(f"{symbol} is not proven unreachable by target lowering")
    return production_symbols


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
    header_path = repo_root / "runtime" / "crt" / "include" / "wafer_tx81_crt.h"
    source_path = repo_root / "runtime" / "crt" / "src" / "wafer_tx81_crt.c"
    attrs_path = repo_root / "include" / "Wafer" / "IR" / "WaferAttrs.td"
    instruction_ops_dir = repo_root / "lib" / "Wafer" / "IR" / "Instr"
    target_lowering_dir = (
        repo_root / "lib" / "Wafer" / "Conversion" / "InstrToLLVM"
    )
    registry_path = (
        repo_root / "lib" / "Wafer" / "Target" / "TargetCall.cpp"
    )
    target_operation_path = (
        repo_root / "lib" / "Wafer" / "Target" / "TargetOperation.cpp"
    )

    header_text = read_text(header_path)
    source_text = read_text(source_path)
    attrs_text = read_text(attrs_path)
    instruction_sources = sorted(instruction_ops_dir.glob("*.cpp"))
    if not instruction_sources:
        fail(f"no instruction implementation sources found in {instruction_ops_dir}")
    instruction_ops_text = "\n".join(read_text(path) for path in instruction_sources)
    lowering_text = "\n".join(
        read_text(target_lowering_dir / source) for source in TARGET_LOWERING_SOURCES
    )
    registry_text = read_text(registry_path)
    target_operation_text = read_text(target_operation_path)

    production_symbols = production_symbols_from_code(
        registry_text,
        lowering_text,
        attrs_text,
        instruction_ops_text,
        target_operation_text,
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
        "from the shared typed target-call registry"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
