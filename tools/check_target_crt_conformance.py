#!/usr/bin/env python3
"""Check Wafer target CRT conformance from compiler and CRT code facts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import re


TARGET_LOWERING_SOURCES = (
    "LowerInstrToTargetLLVM.cpp",
    "TargetCallPreflight.cpp",
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


def fail(message: str) -> None:
    raise ValueError(message)


def read_text(path: pathlib.Path) -> str:
    if not path.is_file():
        fail(f"required file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def function_body(text: str, name: str) -> str:
    match = re.search(
        r"\b(?:static\s+)?(?:void|bool|uint64_t)\s+"
        + re.escape(name)
        + r"\s*\(",
        text,
    )
    if not match:
        fail(f"cannot find function {name}")
    start = text.find("{", match.end())
    if start == -1:
        fail(f"cannot find body for function {name}")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    fail(f"unterminated body for function {name}")


def require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label}: missing `{needle}`")


def require_absent(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"{label}: forbidden `{needle}`")


def require_pattern(text: str, pattern: str, label: str) -> None:
    if not re.search(pattern, text, re.DOTALL):
        fail(f"{label}: missing pattern `{pattern}`")


def require_in_order(text: str, needles: list[str], label: str) -> None:
    cursor = 0
    for needle in needles:
        found = text.find(needle, cursor)
        if found == -1:
            fail(f"{label}: missing ordered token `{needle}`")
        cursor = found + len(needle)


def require_macro_body(text: str, name: str) -> str:
    marker = f"#define {name}"
    start = text.find(marker)
    if start == -1:
        fail(f"cannot find macro {name}")
    next_macro = text.find("\n#define ", start + len(marker))
    if next_macro == -1:
        next_macro = len(text)
    return text[start:next_macro]


def initializer_body(text: str, marker: str) -> str:
    marker_start = text.find(marker)
    if marker_start == -1:
        fail(f"cannot find initializer {marker}")
    start = text.find("{", marker_start + len(marker))
    if start == -1:
        fail(f"cannot find body for initializer {marker}")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    fail(f"unterminated initializer {marker}")


def require_exact_set(actual: set[str], expected: set[str], label: str) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    fail(f"{label}: missing={missing}, extra={extra}")


@dataclass(frozen=True)
class LogicalFormatFact:
    enum_name: str
    spelling: str


@dataclass(frozen=True)
class ConvertRouteFact:
    opcode: int
    spelling: str
    source: str
    destination: str
    parameter: str


def vendor_format_suffix(spelling: str) -> str:
    integer = re.fullmatch(r"i(\d+)", spelling)
    if integer:
        return f"INT{integer.group(1)}"
    unsigned = re.fullmatch(r"u(\d+)", spelling)
    if unsigned:
        return f"UINT{unsigned.group(1)}"
    floating = re.fullmatch(r"f(\d+)", spelling)
    if floating:
        return f"FP{floating.group(1)}"
    return spelling.upper()


def parse_logical_formats(target_format_text: str) -> list[LogicalFormatFact]:
    body = initializer_body(target_format_text, "kLogicalFormats[]")
    matches = re.findall(
        r'\{\s*Format::(\w+)\s*,\s*"([^"]+)"\s*,[^{}]*\}',
        body,
    )
    facts = [LogicalFormatFact(enum_name, spelling) for enum_name, spelling in matches]
    if len(facts) != 13:
        fail(f"logical format registry: expected 13 rows, found {len(facts)}")
    if len({fact.enum_name for fact in facts}) != len(facts):
        fail("logical format registry: duplicate enum identity")
    if len({fact.spelling for fact in facts}) != len(facts):
        fail("logical format registry: duplicate canonical spelling")
    return facts


def parse_target_data_format_codes(target_format_text: str) -> dict[str, int]:
    body = initializer_body(target_format_text, "kTargetDataFormatCodes[]")
    matches = re.findall(
        r"\{\s*kProfile\s*,\s*Format::(\w+)\s*,\s*(\d+)\s*\}", body
    )
    if len(matches) != 13:
        fail(f"target Data_Format registry: expected 13 rows, found {len(matches)}")
    result: dict[str, int] = {}
    seen_codes: set[int] = set()
    for format_name, code_text in matches:
        code = int(code_text)
        if format_name in result:
            fail(f"target Data_Format registry: duplicate format {format_name}")
        if code in seen_codes:
            fail(f"target Data_Format registry: duplicate code {code}")
        result[format_name] = code
        seen_codes.add(code)
    return result


def parse_vendor_data_format_codes(vendor_header_text: str) -> dict[str, int]:
    match = re.search(
        r"typedef\s+enum\s+Data_Format\s*\{(.*?)\}\s*Data_Format\s*;",
        vendor_header_text,
        re.DOTALL,
    )
    if not match:
        fail("vendor header: cannot find Data_Format enum")
    result: dict[str, int] = {}
    for suffix, code_text in re.findall(
        r"\bFmt_([A-Z0-9]+)\s*=\s*(\d+)", match.group(1)
    ):
        if suffix == "UNUSED":
            continue
        if suffix in result:
            fail(f"vendor Data_Format enum: duplicate Fmt_{suffix}")
        result[suffix] = int(code_text)
    if len(result) != 13:
        fail(f"vendor Data_Format enum: expected 13 explicit codes, found {len(result)}")
    return result


def check_data_format_code_contract(
    target_format_text: str, vendor_header_text: str
) -> tuple[list[LogicalFormatFact], dict[str, int]]:
    logical_formats = parse_logical_formats(target_format_text)
    project_codes = parse_target_data_format_codes(target_format_text)
    vendor_codes = parse_vendor_data_format_codes(vendor_header_text)

    logical_names = {fact.enum_name for fact in logical_formats}
    require_exact_set(
        set(project_codes), logical_names, "target Data_Format logical domain"
    )
    expected_vendor_names = {
        vendor_format_suffix(fact.spelling) for fact in logical_formats
    }
    require_exact_set(
        set(vendor_codes), expected_vendor_names, "vendor Data_Format logical domain"
    )
    for fact in logical_formats:
        vendor_name = vendor_format_suffix(fact.spelling)
        if project_codes[fact.enum_name] != vendor_codes[vendor_name]:
            fail(
                "Data_Format code mismatch for "
                f"{fact.enum_name}: project={project_codes[fact.enum_name]}, "
                f"vendor Fmt_{vendor_name}={vendor_codes[vendor_name]}"
            )
    return logical_formats, project_codes


def check_encoding_matrix_contract(
    target_format_text: str,
    logical_formats: list[LogicalFormatFact],
    project_codes: dict[str, int],
) -> int:
    engine_body = initializer_body(target_format_text, "kTargetFormatEngines[]")
    engines = re.findall(r"Engine::(\w+)", engine_body)
    if len(engines) != 5 or len(set(engines)) != len(engines):
        fail(f"target format engine registry: expected 5 unique engines, found {engines}")

    body = initializer_body(target_format_text, "kTargetFormatEncodings[]")
    row_pattern = re.compile(
        r"\b(supported|unsupported)\(\s*Engine::(\w+)\s*,\s*"
        r"Format::(\w+)(?:\s*,\s*(?:Constraint|Reason)::(\w+))?\s*\)"
    )
    rows = row_pattern.findall(body)
    expected_count = len(engines) * len(logical_formats)
    if len(rows) != expected_count or len(rows) != 65:
        fail(f"target format matrix: expected 65 rows, found {len(rows)}")

    actual_pairs: set[tuple[str, str]] = set()
    supported_count = 0
    for support, engine, format_name, detail in rows:
        pair = (engine, format_name)
        if pair in actual_pairs:
            fail(f"target format matrix: duplicate row {engine} x {format_name}")
        actual_pairs.add(pair)
        if support == "supported":
            supported_count += 1
            if format_name not in project_codes:
                fail(
                    "target format matrix: supported row has no profile code "
                    f"{engine} x {format_name}"
                )
            if detail and detail not in {
                "BitpackedLayoutAndCheckedElementCount",
                "BoolSpecificCTOpKindAndBitpackedLayout",
                "BitpackedPhysicalFootprintFill",
            }:
                fail(
                    "target format matrix: supported row has unexpected constraint "
                    f"{detail}"
                )
        elif not detail:
            fail(
                "target format matrix: unsupported row lacks an explicit reason "
                f"{engine} x {format_name}"
            )

    expected_pairs = {
        (engine, fact.enum_name) for engine in engines for fact in logical_formats
    }
    if actual_pairs != expected_pairs:
        fail("target format matrix: rows are not the exact engine x logical domain")
    if supported_count != 30:
        fail(f"target format matrix: expected 30 supported rows, found {supported_count}")

    require_pattern(
        target_format_text,
        r"supported\(Engine\s+engine,\s*Format\s+format.*?"
        r"Reason::None,\s*findDataFormatCode\(kProfile,\s*format\)\}",
        "supported target format row code join",
    )
    require_pattern(
        target_format_text,
        r"unsupported\(Engine\s+engine,\s*Format\s+format.*?"
        r"reason,\s*std::nullopt\}",
        "unsupported target format row has no usable code",
    )
    return len(rows)


def parse_convert_routes(target_format_text: str) -> list[ConvertRouteFact]:
    body = initializer_body(target_format_text, "kTargetConvertRoutes[]")
    matches = re.findall(
        r'\{\s*kProfile\s*,\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*'
        r"Format::(\w+)\s*,\s*Format::(\w+)\s*,\s*"
        r"Parameter::(\w+)\s*\}",
        body,
        re.DOTALL,
    )
    routes = [
        ConvertRouteFact(int(opcode), spelling, source, destination, parameter)
        for opcode, spelling, source, destination, parameter in matches
    ]
    if len(routes) != 36:
        fail(f"target convert registry: expected 36 routes, found {len(routes)}")
    return routes


def check_convert_route_contract(
    target_format_text: str,
    header_text: str,
    source_text: str,
    lowering_text: str,
    registry_text: str,
    logical_formats: list[LogicalFormatFact],
) -> tuple[int, int, int]:
    routes = parse_convert_routes(target_format_text)
    logical_by_name = {fact.enum_name: fact for fact in logical_formats}
    parameter_to_group = {
        "ZeroPoint": "ZP",
        "RoundingMode": "ROUND",
        "None": "PLAIN",
    }
    expected_symbols: dict[str, tuple[str, str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    group_counts = {"ZP": 0, "ROUND": 0, "PLAIN": 0}
    for index, route in enumerate(routes):
        expected_opcode = 139 + index
        if route.opcode != expected_opcode:
            fail(
                f"target convert registry: expected opcode {expected_opcode}, "
                f"found {route.opcode}"
            )
        if route.source not in logical_by_name or route.destination not in logical_by_name:
            fail(f"target convert registry: unknown logical format in {route.spelling}")
        source_token = vendor_format_suffix(
            logical_by_name[route.source].spelling
        ).lower()
        destination_token = vendor_format_suffix(
            logical_by_name[route.destination].spelling
        ).lower()
        expected_spelling = f"{source_token}_{destination_token}"
        if route.spelling != expected_spelling:
            fail(
                f"target convert registry: route {route.opcode} spelling "
                f"{route.spelling} does not match {expected_spelling}"
            )
        pair = (route.source, route.destination)
        if pair in seen_pairs:
            fail(f"target convert registry: duplicate typed route {pair}")
        seen_pairs.add(pair)
        if route.parameter not in parameter_to_group:
            fail(
                f"target convert registry: unknown parameter group {route.parameter}"
            )
        group = parameter_to_group[route.parameter]
        symbol = f"wafer_tx81_convert_{route.spelling}"
        if symbol in expected_symbols:
            fail(f"target convert registry: duplicate symbol {symbol}")
        expected_symbols[symbol] = (group, route.spelling.upper())
        group_counts[group] += 1

    if group_counts != {"ZP": 4, "ROUND": 23, "PLAIN": 9}:
        fail(f"target convert registry: wrong parameter partition {group_counts}")

    invocations = re.findall(
        r"^\s*WAFER_DEFINE_CONVERT_(ZP|ROUND|PLAIN)\(\s*"
        r"(wafer_tx81_convert_[a-z0-9_]+)\s*,\s*([A-Z0-9_]+)\s*\)\s*$",
        source_text,
        re.MULTILINE,
    )
    if len(invocations) != 36:
        fail(f"runtime CRT convert definitions: expected 36, found {len(invocations)}")
    actual_symbols: dict[str, tuple[str, str]] = {}
    for group, symbol, method in invocations:
        if symbol in actual_symbols:
            fail(f"runtime CRT convert definitions: duplicate symbol {symbol}")
        actual_symbols[symbol] = (group, method)
    require_exact_set(
        set(actual_symbols), set(expected_symbols), "runtime CRT convert symbols"
    )
    for symbol, expected in expected_symbols.items():
        if actual_symbols[symbol] != expected:
            fail(
                f"runtime CRT convert definition mismatch for {symbol}: "
                f"expected={expected}, actual={actual_symbols[symbol]}"
            )

    declarations = re.findall(
        r"\bvoid\s+(wafer_tx81_convert_[a-z0-9_]+)\s*\(([^;{}]*)\)\s*;",
        header_text,
        re.DOTALL,
    )
    if len(declarations) != 36:
        fail(f"runtime CRT convert declarations: expected 36, found {len(declarations)}")
    declaration_symbols = [symbol for symbol, _ in declarations]
    if len(set(declaration_symbols)) != len(declaration_symbols):
        fail("runtime CRT convert declarations: duplicate symbol")
    require_exact_set(
        set(declaration_symbols), set(expected_symbols), "runtime CRT convert declarations"
    )
    expected_signature = (
        "uint64_t src, uint64_t dst, uint32_t elem_count, "
        "uint32_t zero_point, uint32_t rounding_mode"
    )
    for symbol, signature in declarations:
        normalized = re.sub(r"\s+", " ", signature).strip()
        if normalized != expected_signature:
            fail(f"runtime CRT convert declaration has wrong ABI for {symbol}")

    require_pattern(
        lowering_text,
        r"Case<InstrConvertOp>\(.*?verifyTargetConvertRoute\(typedOp,\s*"
        r"targetProfile\)",
        "target convert preflight route verification",
    )
    require_pattern(
        lowering_text,
        r"lowerConvert\(InstrConvertOp\s+op\).*?"
        r"emitCall\(op\.getLoc\(\),\s*"
        r"getTargetCallDescriptor\(op\.getKind\(\)\),\s*args\)",
        "target convert typed descriptor lowering",
    )
    require_pattern(
        registry_text,
        r'\("convert_"\s*\+\s*stringifyEnum\(\*kind\)\)\.str\(\),\s*'
        r"signature\(2,\s*3\),\s*\*kind",
        "target convert shared registry ABI",
    )
    return group_counts["ZP"], group_counts["ROUND"], group_counts["PLAIN"]


def check_no_old_abi_or_helpers(source_text: str) -> None:
    forbidden = [
        "__Gemm",
        "__AddVV",
        "__Rdma",
        "__Wdma",
        "__Send",
        "__Recv",
        "__Gelu",
        "op_gelu",
        "__mxfp",
        "__ReduceMul",
        "ChannelNorm",
        "DechannelNorm",
        "libvr",
    ]
    for needle in forbidden:
        require_absent(source_text, needle, "runtime CRT source")


def check_direct_dte_lifecycle(source_text: str) -> None:
    begin = function_body(source_text, "wafer_tx81_direct_dte_begin")
    require_contains(
        begin,
        "wafer_direct_dte_reset_state(status_addr);",
        "standalone Direct DTE begin state reset",
    )
    require_contains(
        begin,
        "direct_sync_init((int)rank_count);",
        "standalone Direct DTE begin synchronization initialization",
    )

    after_prepare = function_body(
        source_text, "wafer_tx81_direct_dte_begin_after_prepare"
    )
    require_contains(
        after_prepare,
        "(void)rank_count;",
        "prepared Direct DTE begin rank-count contract",
    )
    require_contains(
        after_prepare,
        "wafer_direct_dte_reset_state(status_addr);",
        "prepared Direct DTE begin state reset",
    )
    require_absent(
        after_prepare,
        "direct_sync_init",
        "prepared Direct DTE begin synchronization ownership",
    )

    send_prepare = function_body(
        source_text, "wafer_tx81_direct_dte_send_prepare"
    )
    require_in_order(
        send_prepare,
        [
            "get_tile_spm_addr_base(",
            "remote_dst > UINT64_MAX - remote_spm_base",
            "info.src_addr = (uintptr_t)src;",
            "info.dst_addr = (uintptr_t)(remote_spm_base + remote_dst);",
        ],
        "Direct DTE sender local/remote SPM address materialization",
    )

    recv_prepare = function_body(
        source_text, "wafer_tx81_direct_dte_recv_prepare"
    )
    require_contains(
        recv_prepare,
        "direct_fsm_monitor_init((int)local_fsm_id, (uintptr_t)dst,",
        "Direct DTE receiver local SPM offset",
    )
    require_absent(
        recv_prepare,
        "get_spm_memory_mapping",
        "Direct DTE receiver local SPM offset",
    )
    require_absent(
        recv_prepare,
        "get_tile_spm_addr_base",
        "Direct DTE receiver local SPM offset",
    )


def check_dma(source_text: str) -> None:
    conversion = function_body(source_text, "wafer_elem_count_from_bytes")
    for needle in [
        "Fmt_BOOL",
        "UINT32_MAX / 8U",
        "*elem_count = bytes * 8U",
        "bytes % elem_bytes != 0",
        "*elem_count = bytes / elem_bytes",
    ]:
        require_contains(conversion, needle, "DMA checked byte to element conversion")

    rdma = function_body(source_text, "wafer_tx81_rdma")
    require_contains(rdma, "(void)byte_count;", "RDMA")
    require_in_order(rdma, ["rdma->AddSrcDst", "rdma->ConfigStrideIteration"], "RDMA")
    require_contains(
        rdma,
        "!wafer_elem_count_from_bytes(inner_bytes, format, &inner_elements)",
        "RDMA checked inner byte to element conversion",
    )
    require_contains(rdma, "&instr, inner_elements", "RDMA checked element count")

    wdma = function_body(source_text, "wafer_tx81_wdma")
    require_contains(wdma, "(void)byte_count;", "WDMA")
    require_in_order(wdma, ["wdma->AddSrcDst", "wdma->ConfigStrideIteration"], "WDMA")
    require_contains(
        wdma,
        "!wafer_elem_count_from_bytes(inner_bytes, format, &inner_elements)",
        "WDMA checked inner byte to element conversion",
    )
    require_contains(wdma, "&instr, inner_elements", "WDMA checked element count")


def check_gather_scatter_and_mask(
    source_text: str, header_text: str, lowering_text: str
) -> None:
    gather = function_body(source_text, "wafer_tx81_gather_scatter")
    for needle in ["wafer_stride_iteration", "move->GatherScatter", "inner_bytes", "&src_si", "&dst_si"]:
        require_contains(gather, needle, "GatherScatter")

    mask_move = function_body(source_text, "wafer_tx81_mask_move")
    require_contains(mask_move, "move->MaskMove", "MaskMove")
    signature = (
        r"void\s+wafer_tx81_mask_move\s*\(\s*uint64_t\s+src\s*,\s*"
        r"uint32_t\s+mask\s*,\s*uint64_t\s+dst\s*,\s*"
        r"uint32_t\s+elem_count\s*,\s*uint32_t\s+format\s*\)"
    )
    require_pattern(header_text, signature + r"\s*;", "MaskMove public header ABI")
    require_pattern(source_text, signature + r"\s*\{", "MaskMove source ABI")
    require_absent(mask_move, "(uint32_t)mask", "MaskMove hidden mask narrowing")
    require_contains(
        mask_move,
        "move->MaskMove(&instr, src, mask, dst",
        "MaskMove checked mask forwarding",
    )
    require_pattern(
        lowering_text,
        r"lowerMaskMove\(InstrMaskMoveOp\s+op\).*?"
        r"getStaticUInt32MaskAddress\(op,\s*op\.getMask\(\)\).*?"
        r"args\.push_back\(constantI32\(op\.getLoc\(\),\s*\*mask\)\);",
        "MaskMove compiler uint32 proof and call ABI",
    )


def check_arg_writeback(
    source_text: str, instruction_ops_text: str, lowering_text: str
) -> None:
    require_pattern(
        instruction_ops_text,
        r"case\s+InstrPeripheralKind::ArgMax:\s*"
        r"case\s+InstrPeripheralKind::ArgMin:\s*return\s+\{1,\s*2\};",
        "argmax/argmin instruction value/index destinations",
    )
    require_pattern(
        instruction_ops_text,
        r"for\s*\(mlir::Value\s+dest\s*:\s*getDests\(\)\)\s*\{.*?"
        r"verifySPMMemRef\(getOperation\(\),\s*dest\.getType\(\),\s*"
        r'"dest"\)',
        "peripheral instruction SPM destination verification",
    )
    require_pattern(
        lowering_text,
        r"mlir::LogicalResult\s+(?:FunctionLowering::)?lowerPeripheral"
        r"\(InstrPeripheralOp\s+op\)\s*"
        r"\{.*?for\s*\(mlir::Value\s+destValue\s*:\s*op\.getDests\(\)\)"
        r"\s*\{.*?materializeAddress\(op,\s*destValue,\s*"
        r'"peripheral dest"\).*?args\.push_back\(\*address\);',
        "peripheral target destination lowering",
    )
    require_pattern(
        lowering_text,
        r"resolveAddress\(mlir::Operation\s*\*op,\s*mlir::Value\s+value,\s*"
        r"llvm::StringRef\s+role\)\s*\{.*?"
        r"if\s*\(isWaferSPMMemRefType\(rootType\)\)\s*\{.*?"
        r"getAttrOfType<SPMOffsetAttr>\(kWaferSPMOffsetAttrName\).*?"
        r"address\.staticOffset\s*=\s*offset\.getOffset\(\);",
        "target SPM offset address lowering",
    )
    require_contains(
        source_text,
        "extern int8_t *get_spm_memory_mapping(uint64_t offset);",
        "SPM mapping declaration",
    )
    require_pattern(
        source_text,
        r"static\s+uint64_t\s+wafer_spm_mapped_addr\(uint64_t\s+offset\)\s*"
        r"\{\s*return\s+\(uint64_t\)\(uintptr_t\)"
        r"get_spm_memory_mapping\(offset\);\s*\}",
        "SPM mapping helper",
    )
    body = function_body(source_text, "wafer_arg_writeback")
    require_contains(body, "TsmWaitfinish()", "argmax/argmin writeback wait")
    require_pattern(
        body,
        r"wafer_store_value\s*\(\s*wafer_spm_mapped_addr\s*\(\s*value_dst\s*\)\s*,\s*format\s*,\s*instr->param\.wb_data0\s*\)",
        "argmax/argmin value mapped store",
    )
    require_pattern(
        body,
        r"wafer_store_u32\s*\(\s*wafer_spm_mapped_addr\s*\(\s*index_dst\s*\)\s*,\s*\(uint32_t\)instr->param\.wb_data1\s*\)",
        "argmax/argmin index mapped store",
    )
    for symbol in [
        "wafer_tx81_peripheral_argmax",
        "wafer_tx81_peripheral_argmin",
    ]:
        require_contains(
            function_body(source_text, symbol),
            "wafer_arg_writeback(value_dst, index_dst, format, &instr);",
            f"{symbol} mapped writeback",
        )


def check_relation_logic_convert(source_text: str) -> None:
    for macro in [
        "WAFER_DEFINE_RELATION",
        "WAFER_DEFINE_LOGIC_UNARY",
        "WAFER_DEFINE_LOGIC_BINARY",
    ]:
        body = require_macro_body(source_text, macro)
        require_contains(body, "if (format == Fmt_BOOL)", macro)

    zp = require_macro_body(source_text, "WAFER_DEFINE_CONVERT_ZP")
    require_contains(zp, "(void)rounding_mode;", "zero-point convert group")
    require_contains(
        zp,
        "convert->METHOD(&instr, src, zero_point, dst, elem_count)",
        "zero-point convert group",
    )
    rounding = require_macro_body(source_text, "WAFER_DEFINE_CONVERT_ROUND")
    require_contains(rounding, "(void)zero_point;", "rounding convert group")
    require_contains(rounding, "wafer_rounding(rounding_mode)", "rounding convert group")

    plain = require_macro_body(source_text, "WAFER_DEFINE_CONVERT_PLAIN")
    require_contains(plain, "(void)zero_point;", "plain convert group")
    require_contains(plain, "(void)rounding_mode;", "plain convert group")
    require_contains(
        plain,
        "convert->METHOD(&instr, src, dst, elem_count)",
        "plain convert group",
    )


def check_gemm_conv(source_text: str) -> None:
    gemm = function_body(source_text, "wafer_tx81_gemm")
    gemm_sequence = [
            "gemm->AddInput",
            "gemm->ConfigMKN",
            "gemm->ConfigBatch",
            "gemm->SetTransflag",
            "gemm->SetPsum(&instr, 0, 0, Fmt_UNUSED)",
            "gemm->SetQuant(&instr, 0, 0, 0, 0)",
            "gemm->AddBias(&instr, 0, 0)",
            "gemm->SetNegativeAxisScale(&instr, 0, 0)",
            "gemm->SetPositiveAxisScale(&instr, 0, 0)",
            "gemm->DisableRelu(&instr)",
            "gemm->DisableLeakyRelu(&instr)",
            "gemm->AddOutput",
        ]
    require_in_order(
        gemm,
        gemm_sequence,
        "GEMM wrapper sequence",
    )
    require_contains(
        gemm,
        "gemm->SetTransflag(&instr, 0, 0);",
        "v1 GEMM fixed orientation",
    )

    oriented_gemm = function_body(source_text, "wafer_tx81_gemm_oriented_v2")
    require_in_order(
        oriented_gemm,
        gemm_sequence,
        "oriented GEMM wrapper sequence",
    )
    require_contains(
        oriented_gemm,
        "gemm->SetTransflag(&instr, lhs_orientation, rhs_orientation);",
        "oriented GEMM typed orientation",
    )

    conv = require_macro_body(source_text, "WAFER_CONFIGURE_CONV")
    for needle in [
        "OP->AddBias(INSTR, 0, 0)",
        "OP->SetNegativeAxisScale(INSTR, 0, 0)",
        "OP->SetPositiveAxisScale(INSTR, 0, 0)",
        "OP->SetSparse(INSTR, 0, 0)",
        "OP->SetPsum(INSTR, 0, 0, Fmt_UNUSED)",
        "OP->SetQuant(INSTR, 0, 0, 0, 0)",
        "OP->DisableRelu(INSTR)",
        "OP->DisableLeakyRelu(INSTR)",
    ]:
        require_contains(conv, needle, "Conv optional/fused feature disablement")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    header_text = read_text(
        repo_root / "runtime" / "wafer_crt" / "include" / "wafer_tx81_crt.h"
    )
    source_text = read_text(repo_root / "runtime" / "wafer_crt" / "src" / "wafer_tx81_crt.c")
    instruction_ops_dir = repo_root / "lib" / "Wafer" / "IR" / "Instr"
    instruction_sources = sorted(instruction_ops_dir.glob("*.cpp"))
    if not instruction_sources:
        fail(f"no instruction implementation sources found in {instruction_ops_dir}")
    instruction_ops_text = "\n".join(read_text(path) for path in instruction_sources)
    target_lowering_dir = repo_root / "lib" / "Wafer" / "Transforms" / "Target"
    lowering_text = "\n".join(
        read_text(target_lowering_dir / source) for source in TARGET_LOWERING_SOURCES
    )
    registry_text = read_text(
        repo_root
        / "lib"
        / "Wafer"
        / "Target"
        / "TargetCall.cpp"
    )
    target_format_text = read_text(
        repo_root / "lib" / "Wafer" / "Target" / "TargetFormat.cpp"
    )
    vendor_header_text = read_text(
        repo_root / "third_party" / "tx8_deps" / "include" / "instr_def.h"
    )

    logical_formats, project_codes = check_data_format_code_contract(
        target_format_text, vendor_header_text
    )
    encoding_row_count = check_encoding_matrix_contract(
        target_format_text, logical_formats, project_codes
    )
    zp_count, round_count, plain_count = check_convert_route_contract(
        target_format_text,
        header_text,
        source_text,
        lowering_text,
        registry_text,
        logical_formats,
    )

    check_no_old_abi_or_helpers(source_text)
    check_direct_dte_lifecycle(source_text)
    check_dma(source_text)
    check_gather_scatter_and_mask(source_text, header_text, lowering_text)
    check_arg_writeback(source_text, instruction_ops_text, lowering_text)
    check_relation_logic_convert(source_text)
    check_gemm_conv(source_text)
    print(
        "checked Wafer target CRT conformance rules from compiler and CRT code: "
        f"formats={len(logical_formats)}, encoding_rows={encoding_row_count}, "
        f"convert_routes={zp_count + round_count + plain_count}, "
        f"groups={zp_count}/{round_count}/{plain_count}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}")
        raise SystemExit(1)
