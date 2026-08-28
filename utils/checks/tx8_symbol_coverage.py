#!/usr/bin/env python3
"""Generate a symbol coverage matrix for tx8_deps.

The matrix is intentionally rule-based. It is not a replacement for the
reverse-engineering reference; it is an audit aid that makes symbol coverage
repeatable and easy to diff when tx8_deps changes.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TX8_ROOT_DEFAULT = Path("third_party/tx8_deps")
TX8_DOC_DIR = Path("docs/tx8-deps-reverse-engineering")
DEFAULT_CSV = TX8_DOC_DIR / "tx8-symbol-coverage-matrix.csv"
DEFAULT_MD = TX8_DOC_DIR / "tx8-symbol-coverage-matrix.md"

SKIP_SYMBOL_RE = re.compile(
    r"^("
    r"\.L|\.LC|\.LVL|\.LANCHOR|\.LASF|\.LLST|\.Ldebug|"
    r"\$|"
    r"__FRAME_END__|__GNU_EH_FRAME_HDR|"
    r"_DYNAMIC|_GLOBAL_OFFSET_TABLE_|_edata|_end|__bss_start|"
    r"completed\.\d+|__dso_handle"
    r")"
)

NM_PAYLOAD_RE = re.compile(
    r"^(?P<addr>[0-9A-Fa-f]+|\s+)\s+(?P<kind>[A-Za-z?])\s+(?P<symbol>.+)$"
)

ABI_PREFIXES = (
    "Tsm",
    "HrtWatchRpcTask",
    "HrtUnWatchRpcTask",
    "HrtWatchRpcTaskInit",
    "HrtWatchRpcTaskDeinit",
    "OnlineStream",
    "OnlineStreamPreload",
    "OfflineStream",
    "WaitStream",
    "ReqStream",
    "PushStream",
    "PopStream",
    "SendMailbox",
    "GenPayload",
    "tlv_",
)

KCORE_API_PREFIXES = (
    "kuiper_",
    "mod_kuiper_",
    "direct_",
    "mailbox_",
    "get_spm_",
    "get_tile_spm_",
    "tile_sync_",
    "tile_ready_",
    "hrt_barrier",
    "atomic_barrier_",
    "init_atomic_barrier",
    "tsm_ep_log",
    "tx8_log_",
    "monitor_write_log",
    "kcore_write_log",
)

DEPENDENCY_PREFIXES = (
    "rt_",
    "dfs_",
    "libc_",
    "pthread_",
    "posix_",
    "board_",
    "drv_",
    "csi_",
    "aos_",
    "hal_",
    "yoc_",
    "cli_",
    "finsh_",
    "mnt_",
    "vnode",
    "devfs",
    "sys",
    "_ctype_",
    "__libc_",
    "__wrap_",
)

HARDWARE_VERIFY_PATTERNS = (
    "pmu",
    "PMU",
    "mhu_power",
    "power",
    "Power",
    "tx81_mhu",
    "mod_mhu",
    "mhu_send",
    "mhu_recv",
)

IMPLEMENTED_PATTERNS = (
    "RuntimeApiImplHw::",
    "RuntimeApiDecorator::",
    "RuntimeApiErrorDecorator::",
    "RuntimeApiLogDecorator::",
    "RuntimeApiProfDecorator::",
    "Runtime::",
    "HrtBootParam::",
    "HostParamElem",
    "buildRiscv",
    "findAndExtractNumbers",
    "extractNumberFromFolderName",
    "get_tensor_info",
    "get_multi_card_common_info",
    "hrt_get_dtype_size",
    "__execute_",
    "common_",
    "tensor",
    "dtype",
)


@dataclass(frozen=True)
class SymbolRow:
    library: str
    member: str
    symbol: str
    kind: str
    category: str
    scope: str
    note: str


def is_toolchain_path(path: Path) -> bool:
    return "Xuantie-900-gcc-elf-newlib" in str(path)


def discover_libraries(root: Path, include_toolchain: bool) -> list[Path]:
    suffixes = {".a", ".so", ".o"}
    libs = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in suffixes and (include_toolchain or not is_toolchain_path(p))
    ]
    return sorted(libs, key=lambda p: str(p))


def run_nm(path: Path) -> list[str]:
    cmd = ["nm", "-A", "-C", "--defined-only", str(path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return []
    return proc.stdout.splitlines()


def split_nm_line(line: str) -> tuple[str, str, str, str] | None:
    if ":" not in line:
        return None
    prefix, payload = line.rsplit(":", 1)
    match = NM_PAYLOAD_RE.match(payload)
    if not match:
        return None

    library = prefix
    member = ""
    archive_marker = ".a:"
    idx = prefix.find(archive_marker)
    if idx != -1:
        library = prefix[: idx + 2]
        member = prefix[idx + len(archive_marker) :]

    symbol = match.group("symbol").strip()
    kind = match.group("kind")
    if not symbol or SKIP_SYMBOL_RE.match(symbol):
        return None
    return library, member, kind, symbol


def classify(library: str, member: str, symbol: str) -> tuple[str, str, str]:
    lib = Path(library).name
    lower_symbol = symbol.lower()
    lower_member = member.lower()
    lower_path = library.lower()

    if "xuantie-900-gcc-elf-newlib" in lower_path:
        return (
            "IgnoredToolchain",
            "toolchain-runtime",
            "Upstream Xuantie/newlib/libgcc symbol; not TX8 compiler/runtime ABI.",
        )

    if "__execute_sc" in symbol or "scalar" in lower_symbol:
        return (
            "Reserved",
            "reserved-scalar",
            "SCALAR path is stub/reserved in the current reversed dependency set.",
        )

    if "liblibc_stub" in lib or "libc_stub" in lower_path:
        return (
            "Dependency",
            "libc-shim",
            "Kcore libc compatibility shim; dependency boundary, not compiler ABI.",
        )

    if symbol.startswith(ABI_PREFIXES):
        return (
            "ABI",
            "public-runtime-api",
            "Named public wrapper/API entry covered by the reverse-engineering reference.",
        )

    if any(pattern in symbol for pattern in HARDWARE_VERIFY_PATTERNS):
        return (
            "HardwareVerify",
            "hardware-measured-runtime",
            "Static ABI is documented; timing/state semantics require board verification.",
        )

    if any(symbol.startswith(prefix) for prefix in KCORE_API_PREFIXES):
        return (
            "ABI",
            "kcore-runtime-api",
            "Kcore runtime entry covered as device-side ABI.",
        )

    if any(pattern in symbol for pattern in IMPLEMENTED_PATTERNS):
        return (
            "Implemented",
            "reversed-implementation",
            "Implementation symbol covered by function/API-level reverse engineering.",
        )

    if (
        any(symbol.startswith(prefix) for prefix in DEPENDENCY_PREFIXES)
        or lower_member.startswith(("board_", "drv_", "dfs_", "libc", "pthread", "rt_"))
        or "rt-thread" in lower_path
        and not any(token in lower_symbol for token in ("kuiper", "dte", "stream", "mailbox", "pmu", "tlv"))
    ):
        return (
            "Dependency",
            "rtos-board-dependency",
            "RTThread/POSIX/board support dependency; not modeled as TX8 compiler ABI.",
        )

    if lib in {"libinstr_tx81.a", "libcommon_util.a"}:
        return (
            "Implemented",
            "tx8-support-library",
            "TX8 support-library symbol; behavior is captured by headers and disassembly.",
        )

    if lib.startswith("libtx8_profiling"):
        return (
            "ABI",
            "profiling-api",
            "Profiling public/support API; host trigger path is documented.",
        )

    if lib == "libtx8_runtime.so":
        return (
            "Implemented",
            "host-runtime-implementation",
            "Host runtime implementation/support symbol from libtx8_runtime.so.",
        )

    return (
        "Dependency",
        "unclassified-dependency",
        "Named dependency/support symbol outside the compiler-facing ABI surface.",
    )


def collect_symbols(root: Path, include_toolchain: bool) -> list[SymbolRow]:
    rows: list[SymbolRow] = []
    for lib_path in discover_libraries(root, include_toolchain):
        for line in run_nm(lib_path):
            parsed = split_nm_line(line)
            if parsed is None:
                continue
            library, member, kind, symbol = parsed
            category, scope, note = classify(library, member, symbol)
            rows.append(SymbolRow(library, member, symbol, kind, category, scope, note))
    rows.sort(key=lambda r: (r.library, r.member, r.category, r.symbol))
    return rows


def write_csv(path: Path, rows: Iterable[SymbolRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["library", "member", "symbol", "kind", "category", "scope", "note"])
        for row in rows:
            writer.writerow(
                [row.library, row.member, row.symbol, row.kind, row.category, row.scope, row.note]
            )


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def sample_rows(rows: list[SymbolRow], category: str, limit: int = 30) -> list[list[str]]:
    selected = [r for r in rows if r.category == category]
    selected = selected[:limit]
    return [[Path(r.library).name, r.member, f"`{r.symbol}`", r.scope] for r in selected]


def write_markdown(
    path: Path,
    rows: list[SymbolRow],
    root: Path,
    csv_path: Path,
    include_toolchain: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_category = Counter(r.category for r in rows)
    by_library = defaultdict(Counter)
    for row in rows:
        by_library[Path(row.library).name][row.category] += 1

    category_rows = [
        [category, str(by_category[category])]
        for category in ["ABI", "Implemented", "HardwareVerify", "Reserved", "Dependency", "IgnoredToolchain"]
        if by_category[category]
    ]
    library_rows = []
    for lib in sorted(by_library):
        counts = by_library[lib]
        library_rows.append(
            [
                f"`{lib}`",
                str(sum(counts.values())),
                str(counts.get("ABI", 0)),
                str(counts.get("Implemented", 0)),
                str(counts.get("HardwareVerify", 0)),
                str(counts.get("Reserved", 0)),
                str(counts.get("Dependency", 0)),
                str(counts.get("IgnoredToolchain", 0)),
            ]
        )

    important_scopes = Counter(r.scope for r in rows)
    scope_rows = [[f"`{scope}`", str(count)] for scope, count in sorted(important_scopes.items())]

    md = [
        "# TX8 Symbol Coverage Matrix",
        "",
        f"Source root: `{root}`",
        f"Full CSV: `{csv_path}`",
        f"Total named symbols: `{len(rows)}`",
        f"Toolchain symbols expanded: `{'yes' if include_toolchain else 'no'}`",
        "",
        "This matrix is generated by `tools/tx8_symbol_coverage.py`. It is a repeatable audit view over the reversed dependency package, not a replacement for `tx8-deps-reverse-engineering-reference.md`.",
        "",
        "Default output expands TX8-owned runtime/support libraries only. Use `--include-toolchain` when you need the upstream Xuantie/newlib/libgcc symbols expanded as `IgnoredToolchain` rows.",
        "",
        "## Category Semantics",
        "",
        table(
            ["category", "meaning"],
            [
                ["`ABI`", "Public or compiler-facing TX8 API/runtime entry."],
                ["`Implemented`", "Internal TX8 implementation symbol whose behavior is captured by headers/disassembly."],
                ["`HardwareVerify`", "Static ABI is known, but timing/state/corner semantics require board validation."],
                ["`Reserved`", "Known reserved/stub path, currently not a production lowering target."],
                ["`Dependency`", "RTOS, board, libc shim, or support dependency outside compiler ABI."],
                ["`IgnoredToolchain`", "Upstream Xuantie/newlib/libgcc symbol, intentionally excluded from TX8 ABI modeling."],
            ],
        ),
        "",
        "## Summary By Category",
        "",
        table(["category", "symbols"], category_rows),
        "",
        "## Summary By Library",
        "",
        table(
            [
                "library",
                "total",
                "ABI",
                "Implemented",
                "HardwareVerify",
                "Reserved",
                "Dependency",
                "IgnoredToolchain",
            ],
            library_rows,
        ),
        "",
        "## Summary By Scope",
        "",
        table(["scope", "symbols"], scope_rows),
        "",
        "## Boundary Samples",
        "",
        "### HardwareVerify",
        "",
        table(["library", "member", "symbol", "scope"], sample_rows(rows, "HardwareVerify")),
        "",
        "### Reserved",
        "",
        table(["library", "member", "symbol", "scope"], sample_rows(rows, "Reserved")),
        "",
        "### ABI Samples",
        "",
        table(["library", "member", "symbol", "scope"], sample_rows(rows, "ABI")),
        "",
    ]
    path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TX8_ROOT_DEFAULT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--include-toolchain",
        action="store_true",
        help="Also expand Xuantie/newlib/libgcc archives. This is large and mostly noise.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_symbols(args.root, args.include_toolchain)
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows, args.root, args.csv, args.include_toolchain)
    print(f"symbols={len(rows)} csv={args.csv} markdown={args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
