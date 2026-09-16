"""Execute production DMA packet construction with the SDK calls intercepted."""

import argparse
import pathlib
import re
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
source = (args.repo_root / "runtime/crt/src/wafer_tx81_crt.c").read_text()
functions = []
# Only packet construction is host-executable. NCC issue and other CRT
# functions contain RISC-V instructions and are tested by device linking.
for name in ("wafer_format", "wafer_format_bytes", "wafer_elem_count_from_bytes",
             "wafer_dma_packet_format", "wafer_tx81_rdma", "wafer_tx81_wdma"):
    match = re.search(
        r"(?:static\s+)?(?:void|bool|uint32_t|Data_Format)\s+" + name
        + r"\s*\([^;{}]*\)\s*\{", source,
    )
    if not match:
        raise RuntimeError(f"missing production function: {name}")
    depth = 1
    end = match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    functions.append(source[match.start():end])
template = pathlib.Path(__file__).with_suffix(".c").read_text()
output = args.output.with_suffix(".c")
output.write_text(template.replace("/* PRODUCTION_FUNCTIONS */", "\n".join(functions)))
subprocess.run([
    "cc", "-std=c11", "-O2", "-DCONFIG_NO_PLATFORM_HOOK_H",
    "-I" + str(args.repo_root / "third_party/tx8_deps/include"),
    str(output), "-o", str(args.output),
], check=True)
subprocess.run([str(args.output)], check=True)
