#!/usr/bin/env python3
#
# Arm SCP/MCP Software
# Copyright (c) 2019-2020, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

"""
    Check for usage of banned API (banned_api.lst).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import fnmatch


BANNED_API = list()

#
# Directories to exclude
#
EXCLUDE_DIRECTORIES = [
    '.git',
    'build',
    'tools',
    'cmsis'
]

#
# Exclude patterns (applied to files only)
#
EXCLUDE = [
    '*.html',
    '*.xml',
    '*.css',
    '*.gif',
    '*.dat',
    '*.pyc',
    '*.jar',
    '*.md',
    '*.swp',
    '*.a',
    '*.pdf',
]

#
# File types to check
#
FILE_TYPES = [
    '*.c',
    '*.h',
    '*.inc',
]


def is_valid_type(filename):
    for file_type in FILE_TYPES:
        if fnmatch.fnmatch(filename, file_type):
            return True

    return False

import argparse
import subprocess
import re
import sys

def find_riscv_jumps(elf_file: str, objdump_cmd: str = "riscv64-unknown-elf-objdump"):
    """
    在 RISC-V ELF 文件的反汇编代码中查找 'jal' 和 'j' 跳转指令。

    Args:
        elf_file (str): 要分析的 ELF 文件路径。
        objdump_cmd (str): 用于反汇编的 objdump 命令。
                           可以根据你的工具链修改 (例如 'riscv32-unknown-elf-objdump')。
    """
    # 检查 objdump 命令是否存在
    try:
        subprocess.run([objdump_cmd, '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"错误: objdump 命令 '{objdump_cmd}' 未找到或无法执行。", file=sys.stderr)
        print("请确保 RISC-V binutils 工具链已经安装并且在系统的 PATH 环境变量中。", file=sys.stderr)
        sys.exit(1)

    # 构建 objdump 命令
    command = [objdump_cmd, "-d", elf_file]
    print(f"[*] 正在执行命令: {' '.join(command)}")

    try:
        # 执行命令并捕获输出
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        disassembly = result.stdout
        #print(disassembly)
        # 定义用于匹配 'j' 或 'jal' 指令的正则表达式
        # - ^\s*[a-f0-9]+:    匹配行首的地址
        # - \s+               匹配空格
        # - ([a-f0-9]{2}\s)+  匹配指令的十六进制表示
        # - \s+(j|jal)\s+.* 匹配 'j' 或 'jal' 指令及其操作数
        # 我们主要关心指令本身，所以可以简化正则，提高匹配速度
        # 简化版正则：寻找以 'j' 或 'jal' 开头的指令词
        jump_pattern = re.compile(r"\s+(j|jal)\s+.*$")
        #print(f"\n[*] 在 '{elf_file}' 中找到的跳转指令:\n")
        # 逐行解析反汇编代码
        line_list = []
        for line in disassembly.splitlines():
            if re.search(jump_pattern, line):
                line_list.append(line.strip())

        found = False
        for assam_code in line_list:
            #print("assam_code",assam_code)
            #  遍历函数列表并检查调用
            for func_name in BANNED_API:
                # 为了避免匹配到变量名或字符串的一部分
                # re.escape() 处理函数名中可能存在的特殊正则字符
                pattern =  re.escape("<"+ func_name + ">")
                # re.search() 会在整个字符串中查找模式
                # re.MULTILINE 标志不是必需的，但有时处理跨行代码有帮助
                if re.search(pattern, assam_code):
                    found = True
                    print("banned api {} in {}".format(func_name,assam_code),
                          file=sys.stderr)
        if  found:
            sys.exit(-1)
        else:
            sys.exit(0)
    except FileNotFoundError:
        print(f"错误: ELF 文件 '{elf_file}' 不存在。", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"错误: objdump 命令执行失败，返回码 {e.returncode}", file=sys.stderr)
        print(f"错误输出:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="在 RISC-V ELF 文件的反汇编中查找 'j' 和 'jal' 跳转指令。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--elf_file",
        help="要分析的 RISC-V ELF 文件路径。" ,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--tool",
        default="riscv64-unknown-elf-objdump",
        help="指定 objdump 工具的命令名称 (默认: riscv64-unknown-elf-objdump)。"
    )
    parser.add_argument(
        "-l",
        "--api_list",
        required=True,
        help="需要api",
    )
    args = parser.parse_args()
    with open(args.api_list) as file:
        for fname in file:
            fname = fname.rstrip()
            if fname == "":
                continue
            if fname[0] == '#':
                continue
            BANNED_API.append(fname)
        print("\tBanned API list: {}".format(BANNED_API))
    find_riscv_jumps(args.elf_file, args.tool)
    sys.exit(-1)