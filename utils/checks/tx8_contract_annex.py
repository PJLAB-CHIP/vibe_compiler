#!/usr/bin/env python3
"""Generate TX8 API/struct contract annexes from reversed dependency inputs."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path("third_party/tx8_deps")
OUT = Path("docs/tx8-deps-reverse-engineering/tx8-api-struct-contract-annex.md")

INSTR_ADAPTER_PLAT = ROOT / "include/instr_adapter_plat.h"
INSTR_ADAPTER = ROOT / "include/instr_adapter.h"
INSTR_DEF = ROOT / "include/instr_def.h"
RISCV_H = (
    ROOT
    / "tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/riscv.h"
)
DTE_CFG_H = (
    ROOT
    / "tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/dte/dte_cfg.h"
)
MOD_DTE_H = (
    ROOT
    / "tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/dte/mod_dte.h"
)
STREAM_RT_H = (
    ROOT
    / "tx8-yoc-rt-thread-smp/include/components/oplib_tx81/riscv/riscv/include/stream/stream_rt.h"
)
RUNTIME_SO = ROOT / "profiling_tool/examples/engtest_example/libtx8_runtime.so"


STRUCT_NAMES = [
    "Data_Shape",
    "St_Elem_Shape",
    "St_StrideIteration",
    "InstrParamHead",
    "InstrCTParam",
    "InstrNEParam",
    "InstrRDMAParam",
    "InstrWDMAParam",
    "InstrTDMAParam",
    "InstrInvalidInfo",
    "Ncc_CT_GR_Ctl_Regs",
    "Ncc_CT_GR_Param_Regs",
    "CT_Param",
    "Ncc_NE_GR_Ctl_Regs",
    "Ncc_NE_GR_Param_Regs",
    "TsmNeInstr",
    "Ncc_DMA_GR_Ctl_Regs",
    "Ncc_DMA_GR_Param_Regs",
    "DMA_Param",
    "Ncc_TDMA_GR_Ctl_Regs",
    "Ncc_TDMA_GR_Param_Regs",
    "TD_Param",
    "Ncc_SCALAR_GR_Ctl_Regs",
    "Ncc_SCALAR_GR_Param_Regs",
    "SC_Param",
    "NCC_CSR",
    "EXCEP_SERI",
    "D_BootParamHead",
    "D_BootParamDyninfo",
    "D_DynTLV_Terminate",
    "D_KcoreCfgInfo",
    "D_ProfilingConfig",
    "GroupDataInfo",
    "D_GroupDataDumpCfg",
    "TileDteCfg",
    "D_DteCfgList",
    "TileMappingTable",
    "D_DynTLV",
    "D_Cfg_Pmu_Info",
    "D_DynTLV_Cfgpmu",
    "mode_kuiper_dte_dst_config_t",
    "mod_kuiper_dte_node_t",
]

WRAPPER_STRUCTS = [
    "TsmConv",
    "TsmDepthwiseConv",
    "TsmGemm",
    "TsmRdma",
    "TsmWdma",
    "TsmArith",
    "TsmRelation",
    "TsmLogic",
    "TsmTranscendental",
    "TsmActivation",
    "TsmReduce",
    "TsmPool",
    "TsmUnPool",
    "TsmMaskDataMove",
    "TsmConvert",
    "TsmPeripheral",
    "TsmDataMove",
    "TsmStream",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def extract_typedef_struct(text: str, name: str) -> str | None:
    pattern = re.compile(
        r"typedef\s+struct\s+" + re.escape(name) + r"\s*\{.*?\}\s*" + re.escape(name) + r"\s*;",
        re.S,
    )
    match = pattern.search(text)
    if match:
        return match.group(0)

    anon = re.compile(
        r"typedef\s+struct\s*\{.*?\}\s*" + re.escape(name) + r"\s*;",
        re.S,
    )
    match = anon.search(text)
    if match:
        return match.group(0)
    return None


def extract_named_struct(text: str, name: str) -> str | None:
    pattern = re.compile(r"struct\s+" + re.escape(name) + r"\s*\{.*?\}\s*(?:__aligned\(8\))?\s*;", re.S)
    match = pattern.search(text)
    return match.group(0) if match else None


def extract_enum(text: str, name: str) -> str | None:
    patterns = [
        re.compile(r"typedef\s+enum\s+" + re.escape(name) + r"\s*\{.*?\}\s*" + re.escape(name) + r"\s*;", re.S),
        re.compile(r"typedef\s+enum\s*\{.*?\}\s*" + re.escape(name) + r"\s*;", re.S),
        re.compile(r"enum\s+" + re.escape(name) + r"\s*\{.*?\}\s*;", re.S),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def collect_blocks(paths: list[Path], names: list[str]) -> dict[str, str]:
    texts = [(path, read(path)) for path in paths]
    found: dict[str, str] = {}
    for name in names:
        for _, text in texts:
            block = extract_typedef_struct(text, name) or extract_named_struct(text, name)
            if block:
                found[name] = block
                break
    return found


def collect_enums() -> dict[str, str]:
    texts = [read(INSTR_DEF), read(RISCV_H), read(MOD_DTE_H)]
    names = [
        "OP_INSTR_TYPE",
        "OP_INSTR_WORKER",
        "RND_MODE",
        "OP_FUNC_CGRA",
        "Data_Format",
        "Tensor_Fmt",
        "Reduce_Dim",
        "D_DynDataType",
        "KrtRetCode",
        "kuiper_dte_mode_t",
    ]
    out: dict[str, str] = {}
    for name in names:
        for text in texts:
            block = extract_enum(text, name)
            if block:
                out[name] = block
                break
    return out


def host_exports() -> list[str]:
    cmd = ["nm", "-D", "-C", "--defined-only", str(RUNTIME_SO)]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] == "T" and parts[2].startswith("Tsm"):
            rows.append(parts[2])
    return sorted(set(rows))


def public_c_decls() -> list[str]:
    text = read(INSTR_ADAPTER_PLAT) + "\n" + read(INSTR_ADAPTER)
    decls = collect_function_declarations(text)
    rows = []
    for decl in decls:
        if re.match(r"(Tsm\w+\s*\*\s*|void\s+|uint8_t\s+|uint64_t\s+)Tsm\w+\s*\(", decl):
            rows.append(decl)
    return rows


def dte_stream_decls() -> list[str]:
    text = read(MOD_DTE_H) + "\n" + read(STREAM_RT_H)
    decls = collect_function_declarations(text)
    rows = []
    for decl in decls:
        if re.match(r"(int|uint\w+_t|void|mod_kuiper_dte_node_t\s*\*)\s+\w+\s*\(", decl):
            rows.append(decl)
    return rows


def collect_function_declarations(text: str) -> list[str]:
    declarations: list[str] = []
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//") or line.startswith("*"):
            continue
        if not buf and "(" not in line:
            continue
        buf.append(line)
        if ";" in line:
            decl = " ".join(buf)
            decl = re.sub(r"\s+", " ", decl)
            decl = re.sub(r"\s*;\s*$", ";", decl)
            if "(" in decl and ")" in decl and not decl.startswith("typedef"):
                declarations.append(decl)
            buf = []
    return declarations


def code_block(code: str, lang: str = "c") -> str:
    return f"```{lang}\n{code.strip()}\n```"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    struct_paths = [INSTR_ADAPTER_PLAT, INSTR_ADAPTER, INSTR_DEF, RISCV_H, DTE_CFG_H, MOD_DTE_H]
    structs = collect_blocks(struct_paths, STRUCT_NAMES + WRAPPER_STRUCTS)
    enums = collect_enums()

    lines: list[str] = []
    lines += [
        "# TX8 API and Struct Contract Annex",
        "",
        "This annex expands concrete API signatures and C struct shapes from the reversed dependency package. It is generated from current `tx8_deps` headers and `libtx8_runtime.so` symbols, then interpreted with the disassembly notes in `tx8-interface-contract.md`.",
        "",
        "## Host Runtime Exported Signatures",
        "",
    ]
    lines += [f"- `{sig}`" for sig in host_exports()]
    lines += ["", "## Kcore Wrapper Public Constructors/Destructors/CSR", ""]
    lines += [f"- `{decl}`" for decl in public_c_decls()]
    lines += ["", "## DTE and Stream Public Declarations", ""]
    lines += [f"- `{decl}`" for decl in dte_stream_decls()]
    lines += ["", "## Enums", ""]
    for name in sorted(enums):
        lines += [f"### `{name}`", "", code_block(enums[name]), ""]
    lines += ["## Wrapper Function Pointer Structs", ""]
    for name in WRAPPER_STRUCTS:
        block = structs.get(name)
        if block:
            lines += [f"### `{name}`", "", code_block(block), ""]
    lines += ["## Core Instruction and Runtime Structs", ""]
    for name in STRUCT_NAMES:
        block = structs.get(name)
        if block:
            lines += [f"### `{name}`", "", code_block(block), ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} sections={len(structs) + len(enums)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
