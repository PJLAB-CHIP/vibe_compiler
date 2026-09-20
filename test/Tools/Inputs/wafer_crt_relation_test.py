"""Execute production relation ABI dispatch with public SDK methods intercepted."""

import argparse
import pathlib
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
source = (args.repo_root / "runtime/crt/src/wafer_tx81_crt.c").read_text()
match = re.search(r"static Data_Format wafer_format\([^;{}]*\)\s*\{", source)
if not match:
    raise RuntimeError("missing production format function")
depth, end = 1, match.end()
while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
functions = [source[match.start():end]]
start = source.index("#define WAFER_DEFINE_RELATION(")
end = source.index("#define WAFER_DEFINE_LOGIC_UNARY", start)
functions.append(source[start:end])
functions.extend(re.findall(r"WAFER_DEFINE_RELATION\(wafer_tx81_[^)]*\)", source))
template = pathlib.Path(__file__).with_suffix(".c").read_text()
output = args.output.with_suffix(".c")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(template.replace("/* PRODUCTION_FUNCTIONS */", "\n".join(functions)))
subprocess.run([
    "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-DCONFIG_NO_PLATFORM_HOOK_H",
    "-I", str(args.repo_root / "third_party/tx8_deps/include"),
    "-I", str(args.repo_root / "runtime/crt/include"),
    "-I", str(args.repo_root / "include"),
    str(output), "-o", str(args.output),
], check=True)
subprocess.run([str(args.output)], check=True)
