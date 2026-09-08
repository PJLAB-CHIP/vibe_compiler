#!/usr/bin/env python3
"""Measure Kcore clock rate and uninstrumented Direct-DTE call latency."""

import argparse
import hashlib
import json
import os
import pathlib
import re
import struct
import sys

from wafer_board_communication_performance import write_source
from wafer_board_ncc_execution_probe_runner import run, TOOLCHAIN_DIR
import wafer_runtime_launch_contract as launch


ROOT = pathlib.Path(__file__).resolve().parents[3]
MAGIC = 0x4454454C4154454E
GUARD = 0xA5A5A5A5A5A5A5A5
SLICE_BYTES = 32768
PHASES = ("recv_prepare", "send_prepare", "send_issue", "send_wait", "recv_wait")
NCC_PHASES = ("idle_join", "rdma_issue", "rdma_join", "wdma_issue", "wdma_join")
COMBINED_PHASES = ("idle_combined", "idle_split", "wdma_issue", "pending_combined", "pending_split")


def read_records(raw, clock_cycles, payload_bytes, ncc_sync=False, participants=1):
    if len(raw) != 16 * SLICE_BYTES:
        raise RuntimeError("latency output byte count differs from its port")
    tiles = []
    for tile in range(16):
        chunk = raw[tile * SLICE_BYTES:(tile + 1) * SLICE_BYTES]
        words = struct.unpack_from("<256Q", chunk)
        count = 0 if clock_cycles else 32
        if (words[:3] != (MAGIC, tile, clock_cycles)
                or words[5:7] != (count, payload_bytes) or words[255] != GUARD
                or words[8] != (1 if ncc_sync else GUARD)
                or words[9] != (participants if participants > 1 else GUARD)):
            raise RuntimeError("latency record identity/count/guard is invalid")
        result = {"tile": tile, "cycle_read_overhead": words[7]}
        if clock_cycles:
            if not clock_cycles <= words[3] <= clock_cycles * 1.01 or words[4] == 0:
                raise RuntimeError("bounded clock probe did not reach its measurement interval")
            result.update(cycles=words[3], timer_ticks=words[4])
        else:
            for slot in range(3 if participants > 1 else 2):
                base = 2048 + slot * (payload_bytes + 512)
                if (chunk[base:base + 256] != b"\xa5" * 256
                        or chunk[base + 256 + payload_bytes:base + 512 + payload_bytes] != b"\xa5" * 256):
                    raise RuntimeError("DTE source/receive SPM guard was modified")
            if participants > 1:
                base = 2048 + 3 * (payload_bytes + 512)
                if chunk[base:base + 512] != b"\xa5" * 512:
                    raise RuntimeError("combined join source SPM guard was modified")
            phases = COMBINED_PHASES if participants > 1 else NCC_PHASES if ncc_sync else PHASES
            samples = [dict(zip(phases, words[16 + i * 5:21 + i * 5])) for i in range(count)]
            if any(value <= 0 for row in samples for value in row.values()):
                raise RuntimeError("latency phase has no measured duration")
            result["samples_cycles"] = samples
        tiles.append(result)
    return tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path, required=True)
    parser.add_argument("--clock-cycles", type=int, choices=(0, 2000000, 8000000, 20000000), default=0)
    parser.add_argument("--payload-bytes", type=int, choices=(2048, 8192), default=2048)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--ncc-sync", action="store_true")
    parser.add_argument("--ncc-participants", type=int, choices=(1, 2, 3), default=1)
    keys = ("runtime-version", "device-name", "pci-bus-id", "tile-count", "runtime-library-sha256")
    for key in keys:
        parser.add_argument("--expected-" + key)
    args = parser.parse_args()
    if args.ncc_sync and args.clock_cycles:
        parser.error("NCC synchronization and clock calibration are separate probes")
    if args.ncc_participants > 1 and not args.ncc_sync:
        parser.error("multiple NCC participants require --ncc-sync")
    if not args.no_card and (os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1" or any(
            getattr(args, "expected_" + key.replace("-", "_")) is None for key in keys)):
        parser.error("board execution requires arming and complete qualification")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    source = write_source(work, 1024)
    package = work / "package"
    run([str(args.wafer_compile), "--input-program-dir", str(source), "--output-dir", str(package),
         "--num-partitions=1", "--optimization-policy=none", "--test-communication-candidate=peer"], 300)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    launch.require_manifest_launch(manifest, launch.CLUSTER_KERNEL_LAUNCH, context="DTE latency seed")
    entries = launch.require_complete_tile_domain(manifest, context="DTE latency seed")
    if len(manifest["modules"]) != 1 or len(manifest["outputs"]) != 1:
        raise RuntimeError("latency seed must have one module and one output")
    args0 = entries[0]["arguments"]
    input_slot = next(a["ordinal"] for a in args0 if a["kind"] == "external_input" and a["port"] == 0)
    output_slot = next(a["ordinal"] for a in args0 if a["kind"] == "external_output")
    status_slot = next(a["ordinal"] for a in args0 if a["kind"] == "transport_status")
    if (any(entry["arguments"] != args0 or entry["module"] != 0 for entry in entries)
            or manifest["outputs"][0]["bytes"] != 16 * SLICE_BYTES
            or manifest["inputs"][0]["bytes"] != 16 * SLICE_BYTES):
        raise RuntimeError("latency seed pointer-table/port layout is invalid")
    ir = '''target triple = "riscv64-unknown-unknown-elf"
declare i32 @__get_pid(i32)
declare i32 @init_tile_id(i32, i32)
declare void @direct_sync_init(i32)
declare void @wafer_dte_latency_probe(i32, i64, i64, i64)
define void @__wafer_kernel_prepare(ptr %slots) {
  %pid = call i32 @__get_pid(i32 0)
  %unused = call i32 @init_tile_id(i32 %pid, i32 4)
  call void @direct_sync_init(i32 16)
  ret void
}
define void @entry(ptr %slots) {
  %pid = call i32 @__get_pid(i32 0)
  %tile = zext i32 %pid to i64
'''
    ir += f"  %row = mul i64 %tile, {len(args0)}\n  %offset = mul i64 %tile, {SLICE_BYTES}\n"
    for name, slot in (("input", input_slot), ("output", output_slot), ("status", status_slot)):
        ir += (f"  %{name}.index = add i64 %row, {slot}\n"
               f"  %{name}.slot = getelementptr i64, ptr %slots, i64 %{name}.index\n"
               f"  %{name}.base = load i64, ptr %{name}.slot, align 8\n")
        if name != "status":
            ir += f"  %{name} = add i64 %{name}.base, %offset\n"
    ir += "  call void @wafer_dte_latency_probe(i32 %pid, i64 %input, i64 %output, i64 %status.base)\n  ret void\n}\n"
    (work / "probe.ll").write_text(ir)
    deps = ROOT / "third_party/tx8_deps"
    tool_bin = deps / TOOLCHAIN_DIR / "bin"
    obj = work / "probe.o"
    run([str(tool_bin / "riscv64-unknown-elf-gcc"), "-std=c11", "-O2", "-c", "-fPIC",
         "-ffreestanding", "-fno-stack-protector", "-ffunction-sections", "-fdata-sections",
         "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror", "-DCONFIG_NO_PLATFORM_HOOK_H",
         "-DUSING_RISCV", "-mcpu=c908", "-mabi=lp64d",
         f"-DWAFER_LATENCY_CLOCK_CYCLES={args.clock_cycles}ULL",
         f"-DWAFER_LATENCY_PAYLOAD_BYTES={args.payload_bytes}U",
         f"-DWAFER_LATENCY_NCC_SYNC={int(args.ncc_sync)}",
         f"-DWAFER_LATENCY_NCC_PARTICIPANTS={args.ncc_participants}",
         "-I" + str(ROOT / "runtime/crt/include"), "-I" + str(ROOT / "include"),
         "-I" + str(deps / "include"),
         str(ROOT / "test/Board/Inputs/wafer_dte_latency_probe.c"), "-o", str(obj)])
    run([str(tool_bin / "riscv64-unknown-elf-objcopy"), "-R", ".riscv.attributes", str(obj)])
    module = manifest["modules"][0]
    module_path = package / module["path"]
    linked = work / "probe.so"
    run([sys.executable, str(ROOT / "tools/wafer_device_link.py"), "--llvm-ir", str(work / "probe.ll"),
         "--llvm-clangxx", str(args.llvm_clangxx), "--output", str(linked),
         "--loader-abi", "tx8-kcore-loader-cluster", "--extra-object", str(obj)], 120)
    module_path.write_bytes(linked.read_bytes())
    module["digest"] = "sha256:" + hashlib.sha256(module_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    command = [str(args.wafer_run), "--package-dir", str(package)]
    no_card = run(command + ["--no-card", "--direct-dte-status-abi", "wafer-direct-dte-status", "--supports-host-watchdog"])
    if "invocation_tiles: 16" not in no_card.stdout:
        raise RuntimeError("latency no-card did not verify every Tile")
    (work / "no-card.log").write_text(no_card.stdout)
    print(f"latency_no_card: cycles={args.clock_cycles} payload={args.payload_bytes} tiles=16 "
          f"ncc_sync={args.ncc_sync} participants={args.ncc_participants}", flush=True)
    if args.no_card:
        return
    bindings = []
    for port in manifest["inputs"]:
        data = bytearray(b"\xa5" * port["bytes"])
        if port["id"] == 0:
            for tile in range(16):
                start = tile * SLICE_BYTES + 256
                data[start:start + args.payload_bytes] = struct.pack("<e", 0.125) * (args.payload_bytes // 2)
        path = work / f"input-{port['id']}.raw"
        path.write_bytes(data)
        bindings += ["--resource", f"{port['id']}={path}"]
    capture = work / "output.raw"
    command += ["--board", "--device-id", "0", "--completion-timeout-ms", "60000", "--device-timing"]
    for key in keys:
        command += ["--expected-" + key, getattr(args, "expected_" + key.replace("-", "_"))]
    result = run(command + bindings + ["--output", f"{manifest['outputs'][0]['id']}={capture}"], 120)
    (work / "board.log").write_text(result.stdout + result.stderr)
    if "board_execution: true" not in result.stdout or "board_stage: cleanup" not in result.stdout:
        raise RuntimeError("latency execution did not complete normally")
    match = re.search(r"board_timing: kind=tx-stream-events device_elapsed_ns=(\d+)", result.stdout)
    if not match:
        raise RuntimeError("latency execution omitted device timing")
    measurement = {"clock_cycles": args.clock_cycles, "payload_bytes": args.payload_bytes,
                   "ncc_sync": args.ncc_sync,
                   "ncc_participants": args.ncc_participants,
                   "device_elapsed_ns": int(match[1]),
                   "tiles": read_records(capture.read_bytes(), args.clock_cycles, args.payload_bytes,
                                         args.ncc_sync, args.ncc_participants)}
    (work / "measurement.json").write_text(json.dumps(measurement, indent=2) + "\n")
    print("latency_measurement: " + str(work / "measurement.json"), flush=True)


if __name__ == "__main__":
    main()
