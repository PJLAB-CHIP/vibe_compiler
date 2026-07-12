#!/usr/bin/env python3
"""Check Wafer target CRT conformance from compiler and CRT code facts."""

from __future__ import annotations

import argparse
import pathlib
import re


def fail(message: str) -> None:
    raise ValueError(message)


def read_text(path: pathlib.Path) -> str:
    if not path.is_file():
        fail(f"required file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def function_body(text: str, name: str) -> str:
    match = re.search(
        r"\b(?:static\s+)?(?:void|bool)\s+" + re.escape(name) + r"\s*\(",
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
        r"mlir::LogicalResult\s+lowerPeripheral\(InstrPeripheralOp\s+op\)\s*"
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
    for symbol in [
        "wafer_tx81_convert_int8_fp16",
        "wafer_tx81_convert_int8_bf16",
        "wafer_tx81_convert_int8_fp32",
        "wafer_tx81_convert_int8_tf32",
    ]:
        require_contains(source_text, f"WAFER_DEFINE_CONVERT_ZP({symbol}", "zero-point convert symbols")

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
    require_in_order(
        gemm,
        [
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
        ],
        "GEMM wrapper sequence",
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
    instruction_ops_text = read_text(
        repo_root
        / "lib"
        / "Wafer"
        / "IR"
        / "Instr"
        / "InstructionOps.cpp"
    )
    lowering_text = read_text(
        repo_root
        / "lib"
        / "Wafer"
        / "Transforms"
        / "Target"
        / "LowerInstrToTargetLLVM.cpp"
    )

    check_no_old_abi_or_helpers(source_text)
    check_dma(source_text)
    check_gather_scatter_and_mask(source_text, header_text, lowering_text)
    check_arg_writeback(source_text, instruction_ops_text, lowering_text)
    check_relation_logic_convert(source_text)
    check_gemm_conv(source_text)
    print("checked Wafer target CRT conformance rules from compiler and CRT code")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}")
        raise SystemExit(1)
