#!/usr/bin/env python3
"""Profile DDR/DTE choices for handwritten AllGather IR, without an oracle."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import statistics
import struct

import wafer_board_source_program as source_program
from wafer_board_ncc_execution_probe_runner import run


def write_source(work: pathlib.Path, extent: int) -> pathlib.Path:
    shapes = [(16, 16, 1, extent), (16, 1, extent)]
    lhs, rhs = ["tensor<" + "x".join(map(str, shape)) + "xf16>" for shape in shapes]
    ir = f"""module {{
  func.func @main(%lhs: {lhs}, %rhs: {rhs}) -> {lhs} {{
    %double = stablehlo.add %rhs, %rhs : {rhs}
    %gather = "stablehlo.broadcast_in_dim"(%double) {{broadcast_dimensions = array<i64: 1, 2, 3>}} : ({rhs}) -> {lhs}
    %result = stablehlo.add %lhs, %gather : {lhs}
    return %result : {lhs}
  }}
}}
"""
    signatures = [{"shape": list(shape), "dtype": "float16", "dynamic_dims": []}
                  for shape in shapes]
    metadata = {
        "name": "forward", "stablehlo_version": "0.0.0",
        "input_signature": signatures, "output_signature": [signatures[0]],
        "input_locations": [{"type_": "input_arg", "position": i, "name": name}
                            for i, name in enumerate(("lhs", "rhs"))],
        "unused_inputs": [],
    }
    (work / "input.mlir").write_text(ir)
    return source_program.write_program(work / "source", ir, metadata)


def verify_measurement(report: pathlib.Path, messages: int) -> dict:
    analysis = json.loads((report / "analysis.json").read_text())
    if not all(analysis["validity"][key] for key in
               ("trace", "pmu", "pmu_restore", "environment", "program_contract")):
        raise RuntimeError("performance trace is incomplete")
    communication = analysis["program"]["communication"]
    if (communication["send_count"] != 16 * messages
            or communication["receive_count"] != 16 * messages
            or communication["issue_event_count"] != 16 * messages
            or communication["wait_event_count"] != 32 * messages):
        raise RuntimeError("dynamic DTE events differ from actual Instr")
    events = analysis["program"]["timeline_events"]
    summary = {}
    for kind, per_message in (("direct-dte-peer-ready-wait", 1),
                              ("direct-dte-setup-issue", 1),
                              ("direct-dte-completion-wait", 2),
                              ("direct-dte-cleanup", 2)):
        for tile in range(16):
            phases = [event for event in events if event["kind"] == kind and event["tile"] == tile]
            if len(phases) != per_message * messages or any(
                    event["operation_window_status"] != "Measured"
                    or event["operation_window_cpu_cycles"] <= 0 for event in phases):
                raise RuntimeError("DTE phases lack valid per-Tile timing")
        for role in ("send", "receive"):
            cycles = [event["operation_window_cpu_cycles"] for event in events
                      if event["kind"] == kind and event["dte_role"] == role]
            if cycles:
                summary[kind + "/" + role] = {"unit": "cycles", "samples": len(cycles),
                                               "median": statistics.median(cycles)}
    for engine in ("RDMA", "WDMA"):
        rows = [row for tile in analysis["program"]["tiles"] for row in tile["engines"]
                if row["engine"] == engine]
        if len(rows) != 16 or any(not row["engine_execution_time_valid"]
                                 or row["engine_execution_time_ns"] <= 0 for row in rows):
            raise RuntimeError("DDR engines lack valid per-Tile timing")
        summary[engine] = {"unit": "ns", "samples": 16, "median": statistics.median(
            row["engine_execution_time_ns"] for row in rows)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("peer", "shared-ddr"), required=True)
    parser.add_argument("--extent", type=int, choices=(1024, 1025, 4096), required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True,
                        help="internal wafer-compile-test executable")
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    keys = ("runtime-version", "device-name", "pci-bus-id", "tile-count",
            "runtime-library-sha256")
    for key in keys:
        parser.add_argument("--expected-" + key)
    args = parser.parse_args()
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            parser.error("hardware execution is not armed")
        if any(getattr(args, "expected_" + k.replace("-", "_")) is None for k in keys):
            parser.error("board execution requires complete qualification")
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=False)
    source = write_source(work, args.extent)
    compiled = run([str(args.wafer_compile), "--input-program-dir", str(source),
                    "--output-dir", str(work / "delivery"), "--num-partitions=1",
                    "--optimization-policy=none", "--profile",
                    "--test-communication-candidate=" + args.transport,
                    "--dump-compiler-ir", str(work / "compiler-ir")], 1800)
    (work / "compile.log").write_text(compiled.stdout + compiled.stderr)
    tiles = sorted((work / "compiler-ir/instruction").glob("*.mlir"))
    if len(tiles) != 16:
        raise RuntimeError("expected actual Instr for all 16 Tiles")
    messages = 15 if args.transport == "peer" else 0
    for tile in tiles:
        ir = tile.read_text()
        for kind in ("send", "recv"):
            payloads = re.findall(r"wafer\.instr\.dte_" + kind + r" .*?bytes = (\d+) : i64", ir)
            if list(map(int, payloads)) != [args.extent * 2] * messages:
                raise RuntimeError("actual DTE messages/payload differ from requested experiment")
        if ir.count("wafer.instr.dte_wait ") != 2 * messages:
            raise RuntimeError("actual DTE waits differ from requested experiment")
    package = work / "delivery/package"
    manifest = json.loads((package / "manifest.json").read_text())
    arguments = []
    for port in manifest["inputs"]:
        if (port["dtype"] != "f16" or port["layout"] != "tensor"
                or port["bytes"] != 2 * math.prod(port["shape"])):
            raise RuntimeError("performance input requires a compact FP16 tensor")
        path = work / f"input-{port['id']}.raw"
        path.write_bytes(struct.pack("<e", 0.125) * math.prod(port["shape"]))
        arguments += ["--resource", f"{port['id']}={path}"]
    # The runner requires output bindings. Captures are discarded after normal
    # completion; no expected files or external numerical comparison are used.
    captures = [work / f"output-{port['id']}.raw" for port in manifest["outputs"]]
    for port, path in zip(manifest["outputs"], captures):
        arguments += ["--output", f"{port['id']}={path}"]
    command = [str(args.wafer_run), "--package-dir", str(package)]
    no_card = run(command + ["--no-card", "--direct-dte-status-abi",
                            "wafer-direct-dte-status", "--supports-host-watchdog"])
    if "profile_instrumentation: ready cards=1 tiles=16" not in no_card.stdout:
        raise RuntimeError("no-card omitted verified profile instrumentation")
    (work / "no-card.log").write_text(no_card.stdout)
    print(f"performance_no_card: {args.transport} extent={args.extent} "
          f"sends_per_tile={messages} profile_ready=true numeric_reference=false", flush=True)
    if args.no_card:
        return
    command += ["--board", "--device-id", str(args.device_id),
                "--completion-timeout-ms", "60000"]
    for key in keys:
        command += ["--expected-" + key, getattr(args, "expected_" + key.replace("-", "_"))]
    result = run(command + arguments, 240)
    (work / "board.log").write_text(result.stdout + result.stderr)
    if "board_execution: true" not in result.stdout or "board_stage: cleanup" not in result.stdout:
        raise RuntimeError("performance invocation did not complete normally")
    for capture in captures:
        capture.unlink()
    report = pathlib.Path(str(package) + ".profile") / "runs/current"
    print("performance_measurement: " + json.dumps(verify_measurement(report, messages)), flush=True)
    print(f"performance: {args.transport} extent={args.extent} numeric_reference=false "
          f"report={report / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
