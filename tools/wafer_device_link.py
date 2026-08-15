#!/usr/bin/env python3
"""Compile Wafer LLVM IR into a TX8 kcore shared object."""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile


DEFAULT_TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
DEFAULT_GCC_VERSION = "10.4.0"
DEFAULT_MARCH = "rv64imafdc"
DEFAULT_CRT_MCPU = "c908"
DEFAULT_MABI = "lp64d"
DEFAULT_LOADER_ABI = "tx8-kcore-loader-v1"
PROFILE_CAPTURE_KINDS = ("none", "count", "trace")
BASE_LOADER_ABI_UNDEFINED_SYMBOLS = frozenset(
    {
        "get_log_level",
        "get_spm_memory_mapping",
        "get_tile_spm_addr_base",
        "direct_dte_attach",
        "direct_dte_release",
        "direct_dte_send_async",
        "direct_dte_wait_done",
        "direct_fsm_monitor_deinit",
        "direct_fsm_monitor_init",
        "direct_fsm_monitor_receive",
        "direct_sync_init",
        "direct_sync_post",
        "direct_sync_wait",
        "csi_kernel_free",
        "csi_kernel_malloc",
        "monitor_write_log",
        "rt_thread_mdelay",
        "rt_free",
        "rt_malloc",
        "tsm_ep_log",
        "tx8_kernel_printf",
        "tx8_kernel_vprintf",
    }
)
LOADER_ABI_UNDEFINED_SYMBOLS = {
    "tx8-kcore-loader-v1": BASE_LOADER_ABI_UNDEFINED_SYMBOLS,
    "tx8-kcore-loader-grid-v1": BASE_LOADER_ABI_UNDEFINED_SYMBOLS
    | {"__get_pid"},
    "tx8-kcore-loader-cluster-v1": BASE_LOADER_ABI_UNDEFINED_SYMBOLS
    | {"__get_pid", "init_tile_id"},
}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TX8_DEPS_DIR = REPO_ROOT / "third_party" / "tx8_deps"
DEFAULT_WAFER_INCLUDE_DIR = REPO_ROOT / "include"
DEFAULT_WAFER_CRT_SOURCE = REPO_ROOT / "runtime" / "wafer_crt" / "src" / "wafer_tx81_crt.c"
DEFAULT_WAFER_CRT_INCLUDE_DIR = REPO_ROOT / "runtime" / "wafer_crt" / "include"


def fail(message: str) -> None:
    raise ValueError(message)


def resolve_clangxx(value: str | None) -> str:
    if value:
        return value
    env_value = os.environ.get("LLVM_CLANGXX")
    if env_value:
        return env_value
    llvm_binary_dir = os.environ.get("LLVM_BINARY_DIR")
    if llvm_binary_dir:
        return str(pathlib.Path(llvm_binary_dir) / "clang++")
    return "clang++"


def resolve_tx8_gcc(
    tx8_deps_root: pathlib.Path, toolchain_dir_name: str, value: str | None
) -> str:
    if value:
        return value
    return str(
        tx8_deps_root
        / toolchain_dir_name
        / "bin"
        / "riscv64-unknown-elf-gcc"
    )


def resolve_tx8_objcopy(
    tx8_deps_root: pathlib.Path, toolchain_dir_name: str, value: str | None
) -> str:
    if value:
        return value
    return str(
        tx8_deps_root
        / toolchain_dir_name
        / "bin"
        / "riscv64-unknown-elf-objcopy"
    )


def resolve_tx8_nm(
    tx8_deps_root: pathlib.Path, toolchain_dir_name: str, value: str | None
) -> str:
    if value:
        return value
    return str(
        tx8_deps_root
        / toolchain_dir_name
        / "bin"
        / "riscv64-unknown-elf-nm"
    )


def object_output_path(output: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    if output.suffix:
        return output.with_suffix(".o")
    return output.parent / f"{output.name}.o"


def crt_object_output_path(output: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    if output.suffix:
        return output.with_suffix(".wafer_crt.o")
    return output.parent / f"{output.name}.wafer_crt.o"


def tool_exists(tool: str) -> bool:
    path = pathlib.Path(tool)
    if path.parent != pathlib.Path(".") or path.is_absolute():
        return path.exists() and os.access(path, os.X_OK)
    return shutil.which(tool) is not None


def library_exists(library_dir: pathlib.Path, library_name: str) -> bool:
    return any(
        (library_dir / f"lib{library_name}{suffix}").is_file()
        for suffix in (".a", ".so")
    )


def resolve_tx8_include_dir(
    tx8_deps_root: pathlib.Path, value: str | None
) -> pathlib.Path:
    if value:
        return pathlib.Path(value)
    return tx8_deps_root / "include"


def resolve_tx8_sysroot(
    tx8_deps_root: pathlib.Path, toolchain_dir_name: str, value: str | None
) -> pathlib.Path:
    if value:
        return pathlib.Path(value)
    return tx8_deps_root / toolchain_dir_name / "riscv64-unknown-elf"


def build_commands(
    args: argparse.Namespace,
) -> tuple[list[str], list[list[str]], list[str], list[list[str]], list[str], list[str]]:
    llvm_ir = pathlib.Path(args.llvm_ir)
    if not llvm_ir.exists():
        fail(f"LLVM IR input does not exist: {llvm_ir}")
    if not llvm_ir.is_file():
        fail(f"LLVM IR input is not a file: {llvm_ir}")

    output = pathlib.Path(args.output)
    object_output = object_output_path(output, args.object_output)
    crt_object_output = crt_object_output_path(output, args.crt_object_output)
    tx8_deps_root = pathlib.Path(args.tx8_deps_root)
    wafer_crt_source = pathlib.Path(args.wafer_crt_source)
    wafer_crt_include_dir = pathlib.Path(args.wafer_crt_include_dir)
    tx8_include_dir = resolve_tx8_include_dir(tx8_deps_root, args.tx8_include_dir)
    toolchain_root = tx8_deps_root / args.toolchain_dir_name
    libc_dir = toolchain_root / "riscv64-unknown-elf" / "lib" / args.march / args.mabi
    libgcc_dir = (
        toolchain_root
        / "lib"
        / "gcc"
        / "riscv64-unknown-elf"
        / args.gcc_version
        / args.march
        / args.mabi
    )
    clangxx = resolve_clangxx(args.llvm_clangxx)
    tx8_gcc = resolve_tx8_gcc(tx8_deps_root, args.toolchain_dir_name, args.tx8_gcc)
    tx8_objcopy = resolve_tx8_objcopy(
        tx8_deps_root, args.toolchain_dir_name, args.tx8_objcopy
    )
    tx8_nm = resolve_tx8_nm(tx8_deps_root, args.toolchain_dir_name, args.tx8_nm)

    compile_cmd = [
        clangxx,
        str(llvm_ir),
        "-O2",
        "-c",
        "-fPIC",
        "--target=riscv64-unknown-elf",
        f"-march={args.march}",
        "-o",
        str(object_output),
    ]

    normalize_cmds: list[list[str]] = []
    if not args.keep_riscv_attributes:
        normalize_cmds.append(
            [
                tx8_objcopy,
                "-R",
                ".riscv.attributes",
                str(object_output),
            ]
        )

    compile_crt_cmd = [
        tx8_gcc,
        str(wafer_crt_source),
        "-O2",
        "-c",
        "-fPIC",
        "-ffunction-sections",
        "-fdata-sections",
        "-fvisibility=hidden",
        "-DCONFIG_NO_PLATFORM_HOOK_H",
        "-DUSING_RISCV",
        f"-I{wafer_crt_include_dir}",
        f"-I{DEFAULT_WAFER_INCLUDE_DIR}",
        f"-I{tx8_include_dir}",
        f"-mcpu={args.crt_mcpu}",
        f"-mabi={args.mabi}",
        "-o",
        str(crt_object_output),
    ]
    if args.profile_capture in {"count", "trace"}:
        compile_crt_cmd.insert(-2, "-DWAFER_TX81_PROFILE_TRACE_CRT=1")

    normalize_crt_cmds: list[list[str]] = []
    if not args.keep_riscv_attributes:
        normalize_crt_cmds.append(
            [
                tx8_objcopy,
                "-R",
                ".riscv.attributes",
                str(crt_object_output),
            ]
        )

    link_cmd = [
        tx8_gcc,
        "-shared",
        f"-march={args.march}",
        "-O2",
        "-nostartfiles",
        "-Wl,--allow-shlib-undefined",
        f"-mabi={args.mabi}",
        "-Wl,--no-dynamic-linker",
        str(object_output),
        str(crt_object_output),
    ]
    link_cmd.extend(str(path) for path in args.extra_object)
    link_cmd.extend(
        [
            f"-L{libc_dir}",
            f"-L{libgcc_dir}",
            f"-L{tx8_deps_root / 'lib'}",
        ]
    )
    link_cmd.extend(f"-L{path}" for path in args.extra_library_dir)
    link_cmd.extend(
        [
            "-Wl,--exclude-libs,ALL",
            "-Wl,--start-group",
            "-lcommon_util",
            "-linstr_tx81",
            "-llibc_stub",
        ]
    )
    link_cmd.extend(f"-l{library}" for library in args.extra_library)
    link_cmd.extend(
        [
            "-Wl,--end-group",
            "-lm",
            "-Wl,--gc-sections",
            "-Wl,--unique=.rodata.name",
            "-lc",
            "-lgcc",
            "-o",
            str(output),
        ]
    )
    required_symbol_scan_cmd = [tx8_nm, "-u", str(output)]
    return (
        compile_cmd,
        normalize_cmds,
        compile_crt_cmd,
        normalize_crt_cmds,
        link_cmd,
        required_symbol_scan_cmd,
    )


def validate_execute_inputs(
    args: argparse.Namespace,
    compile_cmd: list[str],
    normalize_cmds: list[list[str]],
    compile_crt_cmd: list[str],
    normalize_crt_cmds: list[list[str]],
    link_cmd: list[str],
    required_symbol_scan_cmd: list[str],
) -> None:
    if not tool_exists(compile_cmd[0]):
        fail(f"LLVM clang++ is not executable: {compile_cmd[0]}")
    for normalize_cmd in normalize_cmds:
        if not tool_exists(normalize_cmd[0]):
            fail(f"TX8 RISC-V objcopy is not executable: {normalize_cmd[0]}")
    if not tool_exists(compile_crt_cmd[0]):
        fail(f"TX8 RISC-V GCC is not executable: {compile_crt_cmd[0]}")
    for normalize_cmd in normalize_crt_cmds:
        if not tool_exists(normalize_cmd[0]):
            fail(f"TX8 RISC-V objcopy is not executable: {normalize_cmd[0]}")
    if not tool_exists(link_cmd[0]):
        fail(f"TX8 RISC-V GCC is not executable: {link_cmd[0]}")
    if not tool_exists(required_symbol_scan_cmd[0]):
        fail(f"TX8 RISC-V nm is not executable: {required_symbol_scan_cmd[0]}")

    wafer_crt_source = pathlib.Path(args.wafer_crt_source)
    if not wafer_crt_source.is_file():
        fail(f"Wafer CRT source does not exist: {wafer_crt_source}")
    wafer_crt_include_dir = pathlib.Path(args.wafer_crt_include_dir)
    if not wafer_crt_include_dir.is_dir():
        fail(f"Wafer CRT include dir does not exist: {wafer_crt_include_dir}")
    if not DEFAULT_WAFER_INCLUDE_DIR.is_dir():
        fail(f"Wafer public include dir does not exist: {DEFAULT_WAFER_INCLUDE_DIR}")
    tx8_include_dir = resolve_tx8_include_dir(
        pathlib.Path(args.tx8_deps_root), args.tx8_include_dir
    )
    if not tx8_include_dir.is_dir():
        fail(f"TX8 deps include dir does not exist: {tx8_include_dir}")

    tx8_deps_root = pathlib.Path(args.tx8_deps_root)
    tx8_lib_dir = tx8_deps_root / "lib"
    if not tx8_lib_dir.is_dir():
        fail(f"TX8 deps lib dir does not exist: {tx8_lib_dir}")
    for library in ("common_util", "instr_tx81", "libc_stub"):
        if not library_exists(tx8_lib_dir, library):
            fail(f"TX8 deps library -l{library} does not exist in {tx8_lib_dir}")
    toolchain_root = tx8_deps_root / args.toolchain_dir_name
    libc_dir = toolchain_root / "riscv64-unknown-elf" / "lib" / args.march / args.mabi
    if not libc_dir.is_dir():
        fail(f"TX8 libc multilib dir does not exist: {libc_dir}")
    libgcc_dir = (
        toolchain_root
        / "lib"
        / "gcc"
        / "riscv64-unknown-elf"
        / args.gcc_version
        / args.march
        / args.mabi
    )
    if not libgcc_dir.is_dir():
        fail(f"TX8 libgcc multilib dir does not exist: {libgcc_dir}")

    for path in args.extra_object:
        if not pathlib.Path(path).is_file():
            fail(f"extra object does not exist: {path}")
    for path in args.extra_library_dir:
        if not pathlib.Path(path).is_dir():
            fail(f"extra library dir does not exist: {path}")


def print_commands(
    compile_cmd: list[str],
    normalize_cmds: list[list[str]],
    compile_crt_cmd: list[str],
    normalize_crt_cmds: list[list[str]],
    link_cmd: list[str],
    required_symbol_scan_cmd: list[str],
) -> None:
    print(f"compile: {shlex.join(compile_cmd)}")
    for normalize_cmd in normalize_cmds:
        print(f"normalize: {shlex.join(normalize_cmd)}")
    print(f"compile-crt: {shlex.join(compile_crt_cmd)}")
    for normalize_cmd in normalize_crt_cmds:
        print(f"normalize-crt: {shlex.join(normalize_cmd)}")
    print(f"link: {shlex.join(link_cmd)}")
    print(f"required-symbol-scan: {shlex.join(required_symbol_scan_cmd)}")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_required_symbol_scan(command: list[str], loader_abi: str) -> None:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    undefined = {
        line.split()[-1]
        for line in completed.stdout.splitlines()
        if line.split()
    }
    missing_wafer_symbols = sorted(
        symbol for symbol in undefined if symbol.startswith("wafer_tx81_")
    )
    if missing_wafer_symbols:
        fail(
            "undefined Wafer target CRT symbols remain after link: "
            + ", ".join(missing_wafer_symbols)
        )

    allowed = LOADER_ABI_UNDEFINED_SYMBOLS[loader_abi]
    unexpected = sorted(undefined - allowed)
    if unexpected:
        fail(
            "target_symbol_not_allowed: undefined symbols are not allowed by "
            f"loader ABI {loader_abi}: " + ", ".join(unexpected)
        )


def install_intermediate(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = pathlib.Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def execute_staged_link(args: argparse.Namespace) -> None:
    output = pathlib.Path(args.output)
    object_output = object_output_path(output, args.object_output)
    crt_object_output = crt_object_output_path(output, args.crt_object_output)

    destinations = [output, object_output, crt_object_output]
    absolute_destinations = [os.path.abspath(path) for path in destinations]
    if len(set(absolute_destinations)) != len(absolute_destinations):
        fail("output, object-output and crt-object-output must be distinct")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_prefix = f".{output.name or 'wafer-output'}.staging-"
    with tempfile.TemporaryDirectory(
        prefix=staging_prefix, dir=output.parent
    ) as staging_name:
        staging_dir = pathlib.Path(staging_name)
        staged_args = argparse.Namespace(**vars(args))
        staged_args.output = str(staging_dir / "linked.so")
        staged_args.object_output = str(staging_dir / "input.o")
        staged_args.crt_object_output = str(staging_dir / "wafer_crt.o")

        (
            compile_cmd,
            normalize_cmds,
            compile_crt_cmd,
            normalize_crt_cmds,
            link_cmd,
            required_symbol_scan_cmd,
        ) = build_commands(staged_args)

        run_command(compile_cmd)
        for normalize_cmd in normalize_cmds:
            run_command(normalize_cmd)
        run_command(compile_crt_cmd)
        for normalize_cmd in normalize_crt_cmds:
            run_command(normalize_cmd)
        run_command(link_cmd)
        run_required_symbol_scan(required_symbol_scan_cmd, args.loader_abi)

        install_intermediate(pathlib.Path(staged_args.object_output), object_output)
        install_intermediate(
            pathlib.Path(staged_args.crt_object_output), crt_object_output
        )
        os.replace(pathlib.Path(staged_args.output), output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile Wafer LLVM IR to a TX8 kcore shared object."
    )
    parser.add_argument("--llvm-ir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-output")
    parser.add_argument(
        "--tx8-deps-root",
        default=str(DEFAULT_TX8_DEPS_DIR),
        help="repo-vendored TX8 dependency root",
    )
    parser.add_argument(
        "--wafer-crt-source",
        default=str(DEFAULT_WAFER_CRT_SOURCE),
        help="repo-local Wafer CRT C source",
    )
    parser.add_argument(
        "--wafer-crt-include-dir",
        default=str(DEFAULT_WAFER_CRT_INCLUDE_DIR),
        help="repo-local Wafer CRT include directory",
    )
    parser.add_argument("--tx8-include-dir")
    parser.add_argument("--llvm-clangxx")
    parser.add_argument("--tx8-gcc")
    parser.add_argument("--tx8-objcopy")
    parser.add_argument("--tx8-nm")
    parser.add_argument("--toolchain-dir-name", default=DEFAULT_TOOLCHAIN_DIR)
    parser.add_argument("--gcc-version", default=DEFAULT_GCC_VERSION)
    parser.add_argument("--march", default=DEFAULT_MARCH)
    parser.add_argument(
        "--crt-mcpu",
        default=DEFAULT_CRT_MCPU,
        help="Xuantie CPU selected for the TX81 target CRT",
    )
    parser.add_argument("--mabi", default=DEFAULT_MABI)
    parser.add_argument(
        "--loader-abi",
        choices=sorted(LOADER_ABI_UNDEFINED_SYMBOLS),
        default=DEFAULT_LOADER_ABI,
        help="versioned loader ABI used to validate all undefined symbols",
    )
    parser.add_argument(
        "--profile-capture",
        choices=PROFILE_CAPTURE_KINDS,
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--crt-object-output")
    parser.add_argument(
        "--keep-riscv-attributes",
        action="store_true",
        help=(
            "keep LLVM-emitted .riscv.attributes in generated objects; by "
            "default they are removed before the Xuantie GNU ld link because "
            "LLVM 21 split-extension attributes are not accepted by this "
            "binutils 2.35 toolchain"
        ),
    )
    parser.add_argument("--extra-object", action="append", default=[])
    parser.add_argument("--extra-library-dir", action="append", default=[])
    parser.add_argument("--extra-library", action="append", default=[])
    parser.add_argument(
        "--print-commands",
        "--dry-run",
        action="store_true",
        help="print the compile and link commands without executing them",
    )
    args = parser.parse_args()

    (
        compile_cmd,
        normalize_cmds,
        compile_crt_cmd,
        normalize_crt_cmds,
        link_cmd,
        required_symbol_scan_cmd,
    ) = build_commands(args)
    if args.print_commands:
        print_commands(
            compile_cmd,
            normalize_cmds,
            compile_crt_cmd,
            normalize_crt_cmds,
            link_cmd,
            required_symbol_scan_cmd,
        )
        return 0

    validate_execute_inputs(
        args,
        compile_cmd,
        normalize_cmds,
        compile_crt_cmd,
        normalize_crt_cmds,
        link_cmd,
        required_symbol_scan_cmd,
    )
    execute_staged_link(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
