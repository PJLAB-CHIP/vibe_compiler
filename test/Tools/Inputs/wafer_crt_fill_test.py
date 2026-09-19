"""Execute the production fill with SDK construction/issue intercepted."""

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
for name in ("wafer_format_bytes", "wafer_memset_span_bytes", "wafer_tx81_memset"):
    match = re.search(
        r"(?:static\s+)?(?:void|bool|uint32_t)\s+" + name
        + r"\s*\([^;{}]*\)\s*\{", source,
    )
    if not match:
        raise RuntimeError(f"missing production function: {name}")
    depth, end = 1, match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    functions.append(source[match.start():end])
template = pathlib.Path(__file__).with_suffix(".c").read_text()
output = args.output.with_suffix(".c")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(template.replace("/* PRODUCTION_FUNCTIONS */", "\n".join(functions)))
subprocess.run([
    "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
    "-DCONFIG_NO_PLATFORM_HOOK_H",
    "-I", str(args.repo_root / "third_party/tx8_deps/include"),
    str(output), "-o", str(args.output),
], check=True)
subprocess.run([str(args.output)], check=True)
