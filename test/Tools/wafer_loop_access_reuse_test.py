#!/usr/bin/env python3
"""Qualify loop peer access reuse through PyTorch, the compiler, and SystemC."""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "Board" / "PyTorch"))
import wafer_pytorch_board_cases as cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--extent", type=int, choices=(1024, 1025, 1031), required=True)
    args = parser.parse_args()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)
    case = cases.CASE_FACTORIES[f"gemm-long-k-{args.extent}"](
        getattr(torch, args.dtype), 20260803
    )
    source = args.work_dir / "source-program"
    case.export_program(source)
    dump = args.work_dir / "compiler-ir"
    command = [
        str(args.wafer_compile), "--input-program-dir", str(source),
        "--output-dir", str(args.work_dir / "package"), "--num-partitions=1",
        "--optimization-policy=none", "--test-communication-candidate=access-reuse-peer",
        "--target-model", "--model-atol=0", "--model-rtol=0",
        "--target-model-max-fused-multiply-adds=100000000",
        "--target-model-max-scalar-evaluations=10000000",
        "--target-model-max-movement-bytes=536870912",
        "--target-model-max-movement-segments=1000000",
        "--dump-compiler-ir", str(dump),
    ]
    for role, tensors in (("input", case.inputs),
                          ("expected", case.materialize_expected_outputs())):
        for index, tensor in enumerate(tensors):
            path = args.work_dir / f"{role}_{index}.npy"
            tensor = tensor.detach().cpu().contiguous()
            array = (tensor.view(torch.uint16).numpy().view("V2")
                     if tensor.dtype == torch.bfloat16 else tensor.numpy())
            np.save(path, array, allow_pickle=False)
            command.extend([f"--model-{role}", f"{index}={path}"])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"loop peer access-reuse qualification failed: {completed.returncode}")
    if "target model outputs matched; tiles=16" not in completed.stdout:
        raise RuntimeError("functional model did not compare every output")
    # Supplement the typed compiler checks and full numeric result with a
    # witness that this test exercised repeated transport, not only static DTE.
    nested_send = nested_receive = False
    for path in (dump / "instruction").glob("tile_*.mlir"):
        lines = path.read_text().splitlines()
        loop_indents = [len(line) - len(line.lstrip()) for line in lines
                        if "scf.for " in line]
        if not loop_indents:
            continue
        for line in lines:
            if len(line) - len(line.lstrip()) <= min(loop_indents):
                continue
            nested_send |= "wafer.instr.dte_send " in line
            nested_receive |= "wafer.instr.dte_recv " in line
    if not (nested_send and nested_receive):
        raise RuntimeError("qualified program contains no loop send/receive witness")


if __name__ == "__main__":
    main()
