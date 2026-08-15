#!/usr/bin/env python3

import argparse
import re
import subprocess


CALL_WORKER_LOCATIONS = {
    "wafer_tx81_rdma": ("stack", 24),
    "wafer_tx81_wdma": ("stack", 24),
    "wafer_tx81_elementwise_add": ("register", "a5"),
}

CALLER_SAVED_REGISTERS = {
    "ra",
    "t0",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
}


def parse_integer(value):
    return int(value, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect constant NCC worker arguments in a TX81 ELF entry"
    )
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--elf", required=True)
    arguments = parser.parse_args()

    disassembly = subprocess.run(
        [arguments.objdump, "-dr", arguments.elf],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    function = re.search(
        r"^[0-9a-f]+ <main>:\n(?P<body>.*?)(?=^[0-9a-f]+ <[^>]+>:\n)",
        disassembly,
        re.MULTILINE | re.DOTALL,
    )
    if function is None:
        raise RuntimeError("ELF disassembly has no bounded main function")

    registers = {"zero": 0}
    stack = {}
    workers = {symbol: [] for symbol in CALL_WORKER_LOCATIONS}
    join_masks = []

    for line in function.group("body").splitlines():
        instruction = re.match(
            r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+([a-z0-9.]+)"
            r"(?:\s+([^#]+?))?\s*(?:#.*)?$",
            line,
        )
        if instruction is None:
            continue
        mnemonic = instruction.group(1)
        operands = [
            operand.strip()
            for operand in (instruction.group(2) or "").split(",")
            if operand.strip()
        ]

        call = re.search(r"<(wafer_tx81_[^>]+)>", line)
        if call is not None and mnemonic in {"jal", "j"}:
            symbol = call.group(1)
            if symbol in CALL_WORKER_LOCATIONS:
                location, key = CALL_WORKER_LOCATIONS[symbol]
                value = registers.get(key) if location == "register" else stack.get(key)
                if value is None:
                    raise RuntimeError(
                        f"worker argument for {symbol} is not a constant"
                    )
                workers[symbol].append(value)
            elif symbol == "wafer_tx81_ncc_join":
                value = registers.get("a0")
                if value is None:
                    raise RuntimeError("participant mask for NCC join is not a constant")
                join_masks.append(value)
            for register in CALLER_SAVED_REGISTERS:
                registers.pop(register, None)
            registers["zero"] = 0
            continue

        if mnemonic == "li" and len(operands) == 2:
            registers[operands[0]] = parse_integer(operands[1])
            continue
        if mnemonic == "lui" and len(operands) == 2:
            registers[operands[0]] = parse_integer(operands[1]) << 12
            continue
        if mnemonic == "mv" and len(operands) == 2:
            value = registers.get(operands[1])
            if value is None:
                registers.pop(operands[0], None)
            else:
                registers[operands[0]] = value
            continue
        if mnemonic in {"addi", "addiw"} and len(operands) == 3:
            source = registers.get(operands[1])
            if source is None:
                registers.pop(operands[0], None)
            else:
                registers[operands[0]] = source + parse_integer(operands[2])
            continue
        if mnemonic == "sd" and len(operands) == 2:
            address = re.fullmatch(r"(-?(?:0x)?[0-9a-f]+)\(sp\)", operands[1])
            if address is not None:
                offset = parse_integer(address.group(1))
                value = registers.get(operands[0])
                if value is None:
                    stack.pop(offset, None)
                else:
                    stack[offset] = value
            continue

        # Loads and arithmetic overwrite their first register operand. The
        # worker constants of interest are rematerialized immediately before
        # each call, so dropping unknown values is conservative.
        if operands and re.fullmatch(r"(?:[ast][0-9]+|ra|sp)", operands[0]):
            registers.pop(operands[0], None)

    for symbol, values in workers.items():
        print(f"{symbol} workers=" + ",".join(str(value) for value in values))
    print("wafer_tx81_ncc_join participant_masks=" +
          ",".join(str(value) for value in join_masks))


if __name__ == "__main__":
    main()
