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
DEFAULT_MARCH = "rv64imafdc"
DEFAULT_MABI = "lp64d"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TX8_DEPS_DIR = REPO_ROOT / "third_party" / "tx8_deps"
DEFAULT_WAFER_CRT_LIB_PATH = REPO_ROOT / "third_party" / "wafer_crt" / "lib"


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


def object_output_path(output: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    if output.suffix:
        return output.with_suffix(".o")
    return output.parent / f"{output.name}.o"


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
) -> tuple[list[str], list[list[str]], list[str]]:
    llvm_ir = pathlib.Path(args.llvm_ir)
    if not llvm_ir.exists():
        fail(f"LLVM IR input does not exist: {llvm_ir}")
    if not llvm_ir.is_file():
        fail(f"LLVM IR input is not a file: {llvm_ir}")

    output = pathlib.Path(args.output)
    object_output = object_output_path(output, args.object_output)
    tx8_deps_root = pathlib.Path(args.tx8_deps_root)
    wafer_crt_lib_dir = pathlib.Path(args.wafer_crt_lib_dir)
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
    return compile_cmd, normalize_cmds, link_cmd


def validate_execute_inputs(
    args: argparse.Namespace,
    compile_cmd: list[str],
    normalize_cmds: list[list[str]],
    link_cmd: list[str],
) -> None:
    if not tool_exists(compile_cmd[0]):
        fail(f"LLVM clang++ is not executable: {compile_cmd[0]}")
    for normalize_cmd in normalize_cmds:
        if not tool_exists(normalize_cmd[0]):
            fail(f"TX8 RISC-V objcopy is not executable: {normalize_cmd[0]}")
    if not tool_exists(link_cmd[0]):
        fail(f"TX8 RISC-V GCC is not executable: {link_cmd[0]}")

    wafer_crt_lib_dir = pathlib.Path(args.wafer_crt_lib_dir)
    if not wafer_crt_lib_dir.is_dir():
        fail(f"Wafer CRT lib dir does not exist: {wafer_crt_lib_dir}")

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
    if not library_exists(wafer_crt_lib_dir, "vr"):
        fail(f"Wafer CRT library -lvr does not exist in {wafer_crt_lib_dir}")

    for path in args.extra_object:
        if not pathlib.Path(path).is_file():
            fail(f"extra object does not exist: {path}")
    for path in args.extra_library_dir:
        if not pathlib.Path(path).is_dir():
            fail(f"extra library dir does not exist: {path}")


def print_commands(
    compile_cmd: list[str],
    normalize_cmds: list[list[str]],
    link_cmd: list[str],
) -> None:
    print(f"compile: {shlex.join(compile_cmd)}")
    for normalize_cmd in normalize_cmds:
        print(f"normalize: {shlex.join(normalize_cmd)}")
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
    parser.add_argument(
        "--tx8-deps-root",
        default=str(DEFAULT_TX8_DEPS_DIR),
        help="repo-vendored TX8 dependency root",
    )
    parser.add_argument(
        "--wafer-crt-lib-dir",
        default=str(DEFAULT_WAFER_CRT_LIB_PATH),
        help="repo-local Wafer CRT library directory",
    )
    parser.add_argument("--llvm-clangxx")
    parser.add_argument("--tx8-gcc")
    parser.add_argument("--tx8-objcopy")
    parser.add_argument("--toolchain-dir-name", default=DEFAULT_TOOLCHAIN_DIR)
    parser.add_argument("--gcc-version", default=DEFAULT_GCC_VERSION)
    parser.add_argument("--march", default=DEFAULT_MARCH)
    parser.add_argument("--mabi", default=DEFAULT_MABI)
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

    compile_cmd, normalize_cmds, link_cmd = build_commands(args)
    if args.print_commands:
        print_commands(compile_cmd, normalize_cmds, link_cmd)
        return 0

    validate_execute_inputs(
        args, compile_cmd, normalize_cmds, link_cmd
    )
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    object_output_path(pathlib.Path(args.output), args.object_output).parent.mkdir(
        parents=True, exist_ok=True
    )
    run_command(compile_cmd)
    for normalize_cmd in normalize_cmds:
        run_command(normalize_cmd)
    run_command(link_cmd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
