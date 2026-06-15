#!/usr/bin/env python3
"""Run manual heavy candidate-selection tests.

This is a dedicated stress runner, not a lit/ctest regression.  It keeps large
shape candidate-search coverage out of the default test suite while still
making the cases reproducible from the repository checkout.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import subprocess
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    m: int
    k: int
    n: int
    extra_roots: tuple[str, ...]
    require_split: bool = False


CASES = {
    "multi_output_if_traversal_tile": Case(
        name="multi_output_if_traversal_tile",
        m=257,
        k=4096,
        n=257,
        extra_roots=("add",),
    ),
    "multi_output_if_k_split": Case(
        name="multi_output_if_k_split",
        m=257,
        k=8192,
        n=257,
        extra_roots=("add",),
        require_split=True,
    ),
    "three_output_if_multi_op": Case(
        name="three_output_if_multi_op",
        m=257,
        k=4096,
        n=257,
        extra_roots=("add", "mul"),
    ),
}


SUMMARY_RE = re.compile(
    r"wafer\.select_group_tile selected group @(?P<name>[A-Za-z0-9_]+)#0 "
    r"mode=(?P<mode>[^ ]+) tile=\[(?P<tile>[^\]]*)\] "
    r"split=\[(?P<split>[^\]]*)\] estimated_cycles=(?P<cycles>[0-9]+) "
    r"candidates=(?P<candidates>[0-9]+) rejected=(?P<rejected>[0-9]+) "
    r"representatives=(?P<representatives>[0-9]+)"
)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def tensor_type(rows: int, cols: int) -> str:
    return f"tensor<{rows}x{cols}xf16>"


def linalg_binary_root(name: str, op: str, lhs: str, rhs: str, out: str, shape: str) -> str:
    arith_op = {"add": "arith.addf", "mul": "arith.mulf"}[op]
    return f"""      %{name} = linalg.generic {{
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        }} ins(%{lhs}, %{rhs} : {shape}, {shape})
          outs(%{out} : {shape}) {{
      ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
        %{name}_value = {arith_op} %lhs_el, %rhs_el : f16
        linalg.yield %{name}_value : f16
      }} -> {shape}
"""


def build_case_ir(case: Case) -> str:
    result_type = tensor_type(case.m, case.n)
    lhs_type = tensor_type(case.m, case.k)
    rhs_type = tensor_type(case.k, case.n)
    result_count = 1 + len(case.extra_roots)

    extra_input_names: list[str] = []
    extra_input_decls: list[str] = []
    for index, op in enumerate(case.extra_roots):
        lhs_name = f"{op}_lhs"
        rhs_name = f"{op}_rhs"
        extra_input_names.extend([lhs_name, rhs_name])
        extra_input_decls.extend([f"%{lhs_name}: {result_type}", f"%{rhs_name}: {result_type}"])

    out_decls = [f"%out{index}: {result_type}" for index in range(result_count)]
    ret_types = ", ".join([result_type] * result_count)
    result_values = ", ".join([f"%result#{index}" for index in range(result_count)])
    out_values = ", ".join([f"%out{index}" for index in range(result_count)])

    group_inputs = ["%lhs", "%rhs"] + [f"%{name}" for name in extra_input_names]
    group_input_types = [lhs_type, rhs_type] + [result_type] * len(extra_input_names)
    group_outs = [f"%out{index}" for index in range(result_count)]
    group_out_types = [result_type] * result_count

    block_args = [("%arg0", lhs_type), ("%arg1", rhs_type)]
    for index in range(len(extra_input_names)):
        block_args.append((f"%arg{2 + index}", result_type))
    output_arg_base = 2 + len(extra_input_names)
    for index in range(result_count):
        block_args.append((f"%arg{output_arg_base + index}", result_type))

    roots = [
        f"""      %mm = linalg.matmul
          ins(%arg0, %arg1 : {lhs_type}, {rhs_type})
          outs(%arg{output_arg_base} : {result_type}) -> {result_type}
"""
    ]

    yielded = ["%mm"]
    for index, op in enumerate(case.extra_roots):
        input_base = 2 + index * 2
        out_arg = output_arg_base + 1 + index
        root_name = f"{op}_root"
        roots.append(
            linalg_binary_root(
                root_name,
                op,
                f"arg{input_base}",
                f"arg{input_base + 1}",
                f"arg{out_arg}",
                result_type,
            )
        )
        yielded.append(f"%{root_name}")

    func_args = [f"%lhs: {lhs_type}", f"%rhs: {rhs_type}"] + extra_input_decls + out_decls + [
        "%cond: i1"
    ]

    return f"""func.func @{case.name}(
    {", ".join(func_args)}) -> ({ret_types}) {{
  %result:{result_count} = scf.if %cond -> ({ret_types}) {{
    %group:{result_count} = wafer.group
        ins({", ".join(group_inputs)} : {", ".join(group_input_types)})
        outs({", ".join(group_outs)} : {", ".join(group_out_types)}) {{
    ^bb0({", ".join(f"{name}: {typ}" for name, typ in block_args)}):
{''.join(roots)}      wafer.group.yield {", ".join(yielded)} : {ret_types}
    }} : {ret_types}
    scf.yield {", ".join(f"%group#{index}" for index in range(result_count))}
        : {ret_types}
  }} else {{
    scf.yield {out_values} : {ret_types}
  }}
  return {result_values} : {ret_types}
}}
"""


def run_case(case: Case, wafer_opt: pathlib.Path, timeout_seconds: int) -> bool:
    mlir = build_case_ir(case)
    command = [
        str(wafer_opt),
        "--wafer-select-group-tile=print-candidate-summary max-candidates-per-dim=3 preferred-tile-sizes=64",
        "-",
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=mlir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"FAIL {case.name}: timed out after {timeout_seconds}s")
        return False

    elapsed = time.monotonic() - start
    output = completed.stdout
    match = SUMMARY_RE.search(output)
    if completed.returncode != 0 or not match:
        print(f"FAIL {case.name}: wafer-opt failed or emitted no summary")
        print(output)
        return False

    rejected = int(match.group("rejected"))
    candidates = int(match.group("candidates"))
    representatives = int(match.group("representatives"))
    tile_sizes = parse_int_list(match.group("tile"))
    split = match.group("split")

    required_needles = [
        "scf.if",
        "wafer.tile.region",
        "wafer.spm.offset",
        "wafer.instr.gemm",
        "wafer.instr.elementwise <add>",
        "wafer.instr.wdma",
    ]
    if "mul" in case.extra_roots:
        required_needles.append("wafer.instr.elementwise <mul>")

    failures: list[str] = []
    if candidates <= 1:
        failures.append("expected more than one candidate")
    if rejected <= 0:
        failures.append("expected at least one rejected candidate")
    expected_representatives = 1
    for dim_size, tile_size in zip((case.m, case.n), tile_sizes):
        if dim_size > tile_size:
            expected_representatives *= 2
    if representatives < expected_representatives:
        failures.append(
            f"expected at least {expected_representatives} representative tile classes"
        )
    if case.require_split and not split.strip():
        failures.append("expected a non-empty reduction split")
    for needle in required_needles:
        if needle not in output:
            failures.append(f"missing lowered IR needle: {needle}")
    for forbidden in ("wafer.group", "linalg."):
        if forbidden in output:
            failures.append(f"committed output still contains {forbidden}")

    if failures:
        print(f"FAIL {case.name}: " + "; ".join(failures))
        print(output)
        return False

    print(
        "PASS {name}: tile=[{tile}] split=[{split}] candidates={candidates} "
        "rejected={rejected} representatives={representatives} elapsed={elapsed:.2f}s".format(
            name=case.name,
            tile=match.group("tile"),
            split=split,
            candidates=candidates,
            rejected=rejected,
            representatives=representatives,
            elapsed=elapsed,
        )
    )
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wafer-opt",
        type=pathlib.Path,
        default=REPO_ROOT / "build" / "wafer-dev" / "bin" / "wafer-opt",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="Case to run. Defaults to all cases.",
    )
    args = parser.parse_args(argv)

    selected = args.case or sorted(CASES)
    ok = True
    for name in selected:
        ok = run_case(CASES[name], args.wafer_opt, args.timeout) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
