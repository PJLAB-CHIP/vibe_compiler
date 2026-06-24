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


DEFAULT_TOOLCHAIN_DIR = "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"
DEFAULT_GCC_VERSION = "10.4.0"
DEFAULT_MARCH = "rv64imfdc"
DEFAULT_MABI = "lp64d"
DEFAULT_WAFER_SHIM_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "runtime" / "wafer_cabi_shim.c"
)


def fail(message: str) -> None:
    raise ValueError(message)


def resolve_required_path(
    value: str | None, env_name: str, description: str
) -> pathlib.Path:
    if value:
        return pathlib.Path(value)
    env_value = os.environ.get(env_name)
    if env_value:
        return pathlib.Path(env_value)
    option_name = env_name.lower().replace("_", "-")
    fail(f"{description} must be provided with --{option_name} or {env_name}")


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


def object_output_path(output: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    if output.suffix:
        return output.with_suffix(".o")
    return output.parent / f"{output.name}.o"


def shim_object_output_path(output: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    if output.suffix:
        return output.with_suffix(".wafer_cabi_shim.o")
    return output.parent / f"{output.name}.wafer_cabi_shim.o"


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
    env_value = os.environ.get("TX8_INCLUDE_DIR")
    if env_value:
        return pathlib.Path(env_value)
    return tx8_deps_root / "include"


def resolve_tx8_sysroot(
    tx8_deps_root: pathlib.Path, toolchain_dir_name: str, value: str | None
) -> pathlib.Path:
    if value:
        return pathlib.Path(value)
    env_value = os.environ.get("TX8_SYSROOT")
    if env_value:
        return pathlib.Path(env_value)
    return tx8_deps_root / toolchain_dir_name / "riscv64-unknown-elf"


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[list[str]], list[str]]:
    llvm_ir = pathlib.Path(args.llvm_ir)
    if not llvm_ir.exists():
        fail(f"LLVM IR input does not exist: {llvm_ir}")
    if not llvm_ir.is_file():
        fail(f"LLVM IR input is not a file: {llvm_ir}")

    output = pathlib.Path(args.output)
    object_output = object_output_path(output, args.object_output)
    tx8_deps_root = resolve_required_path(
        args.tx8_deps_root, "TX8_DEPS_ROOT", "TX8 deps root"
    )
    tx8_include_dir = resolve_tx8_include_dir(tx8_deps_root, args.tx8_include_dir)
    tx8_sysroot = resolve_tx8_sysroot(
        tx8_deps_root, args.toolchain_dir_name, args.tx8_sysroot
    )
    wafer_crt_lib_dir = resolve_required_path(
        args.wafer_crt_lib_dir, "WAFER_CRT_LIB_DIR", "Wafer CRT lib dir"
    )
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

    shim_compile_cmds: list[list[str]] = []
    shim_objects: list[pathlib.Path] = []
    if not args.no_default_wafer_shim:
        shim_source = pathlib.Path(args.wafer_shim_source)
        if not shim_source.exists():
            fail(f"Wafer C ABI shim source does not exist: {shim_source}")
        if not shim_source.is_file():
            fail(f"Wafer C ABI shim source is not a file: {shim_source}")
        shim_object = shim_object_output_path(output, args.wafer_shim_object_output)
        shim_compile_cmds.append(
            [
                clangxx,
                "-x",
                "c",
                str(shim_source),
                "-O2",
                "-c",
                "-fPIC",
                "--target=riscv64-unknown-elf",
                f"-march={args.march}",
                f"-mabi={args.mabi}",
                f"--sysroot={tx8_sysroot}",
                "-DUSING_RISCV",
                "-DCONFIG_NO_PLATFORM_HOOK_H",
                f"-I{tx8_include_dir}",
                "-o",
                str(shim_object),
            ]
        )
        shim_objects.append(shim_object)

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
    ]
    link_cmd.extend(str(path) for path in shim_objects)
    link_cmd.extend(str(path) for path in args.extra_object)
    link_cmd.extend(
        [
            f"-L{wafer_crt_lib_dir}",
            f"-L{libc_dir}",
            f"-L{libgcc_dir}",
            f"-L{tx8_deps_root / 'lib'}",
        ]
    )
    link_cmd.extend(f"-L{path}" for path in args.extra_library_dir)
    link_cmd.extend(
        [
            "-Wl,--start-group",
            "-lcommon_util",
            "-linstr_tx81",
            "-llibc_stub",
            "-lvr",
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
    return compile_cmd, shim_compile_cmds, link_cmd


def validate_execute_inputs(
    args: argparse.Namespace,
    compile_cmd: list[str],
    shim_compile_cmds: list[list[str]],
    link_cmd: list[str],
) -> None:
    if not tool_exists(compile_cmd[0]):
        fail(f"LLVM clang++ is not executable: {compile_cmd[0]}")
    for shim_compile_cmd in shim_compile_cmds:
        if not tool_exists(shim_compile_cmd[0]):
            fail(f"LLVM clang++ is not executable: {shim_compile_cmd[0]}")
    if not tool_exists(link_cmd[0]):
        fail(f"TX8 RISC-V GCC is not executable: {link_cmd[0]}")

    wafer_crt_lib_dir = resolve_required_path(
        args.wafer_crt_lib_dir, "WAFER_CRT_LIB_DIR", "Wafer CRT lib dir"
    )
    if not wafer_crt_lib_dir.is_dir():
        fail(f"Wafer CRT lib dir does not exist: {wafer_crt_lib_dir}")

    tx8_deps_root = resolve_required_path(
        args.tx8_deps_root, "TX8_DEPS_ROOT", "TX8 deps root"
    )
    tx8_lib_dir = tx8_deps_root / "lib"
    if not tx8_lib_dir.is_dir():
        fail(f"TX8 deps lib dir does not exist: {tx8_lib_dir}")
    for library in ("common_util", "instr_tx81", "libc_stub"):
        if not library_exists(tx8_lib_dir, library):
            fail(f"TX8 deps library -l{library} does not exist in {tx8_lib_dir}")
    tx8_include_dir = resolve_tx8_include_dir(tx8_deps_root, args.tx8_include_dir)
    if shim_compile_cmds and not tx8_include_dir.is_dir():
        fail(f"TX8 deps include dir does not exist: {tx8_include_dir}")
    tx8_sysroot = resolve_tx8_sysroot(
        tx8_deps_root, args.toolchain_dir_name, args.tx8_sysroot
    )
    if shim_compile_cmds and not tx8_sysroot.is_dir():
        fail(f"TX8 sysroot does not exist: {tx8_sysroot}")
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
    if not library_exists(wafer_crt_lib_dir, "vr"):
        fail(f"Wafer CRT library -lvr does not exist in {wafer_crt_lib_dir}")

    for path in args.extra_object:
        if not pathlib.Path(path).is_file():
            fail(f"extra object does not exist: {path}")
    for path in args.extra_library_dir:
        if not pathlib.Path(path).is_dir():
            fail(f"extra library dir does not exist: {path}")


def print_commands(
    compile_cmd: list[str], shim_compile_cmds: list[list[str]], link_cmd: list[str]
) -> None:
    print(f"compile: {shlex.join(compile_cmd)}")
    for shim_compile_cmd in shim_compile_cmds:
        print(f"shim-compile: {shlex.join(shim_compile_cmd)}")
    print(f"link: {shlex.join(link_cmd)}")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile Wafer LLVM IR to a TX8 kcore shared object."
    )
    parser.add_argument("--llvm-ir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-output")
    parser.add_argument("--tx8-deps-root")
    parser.add_argument("--wafer-crt-lib-dir")
    parser.add_argument("--llvm-clangxx")
    parser.add_argument("--tx8-gcc")
    parser.add_argument("--tx8-include-dir")
    parser.add_argument("--tx8-sysroot")
    parser.add_argument(
        "--wafer-shim-source", default=str(DEFAULT_WAFER_SHIM_SOURCE)
    )
    parser.add_argument("--wafer-shim-object-output")
    parser.add_argument(
        "--no-default-wafer-shim",
        action="store_true",
        help="do not compile and link the default wafer_* C ABI shim",
    )
    parser.add_argument("--toolchain-dir-name", default=DEFAULT_TOOLCHAIN_DIR)
    parser.add_argument("--gcc-version", default=DEFAULT_GCC_VERSION)
    parser.add_argument("--march", default=DEFAULT_MARCH)
    parser.add_argument("--mabi", default=DEFAULT_MABI)
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

    compile_cmd, shim_compile_cmds, link_cmd = build_commands(args)
    if args.print_commands:
        print_commands(compile_cmd, shim_compile_cmds, link_cmd)
        return 0

    validate_execute_inputs(args, compile_cmd, shim_compile_cmds, link_cmd)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    object_output_path(pathlib.Path(args.output), args.object_output).parent.mkdir(
        parents=True, exist_ok=True
    )
    for shim_compile_cmd in shim_compile_cmds:
        pathlib.Path(shim_compile_cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
    run_command(compile_cmd)
    for shim_compile_cmd in shim_compile_cmds:
        run_command(shim_compile_cmd)
    run_command(link_cmd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
