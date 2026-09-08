#!/usr/bin/env python3
"""Convert this session's guarded clock/DTE/NCC timing records to nanoseconds."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics

from wafer_board_dte_latency import read_records


def distribution(values, scale):
    ordered = sorted(value * scale for value in values)
    return {"samples": len(ordered), "median_ns": statistics.median(ordered),
            "p90_ns": ordered[math.ceil(0.9 * len(ordered)) - 1],
            "min_ns": ordered[0], "max_ns": ordered[-1]}


def summarize(paths):
    rows, identities, evidence = [], set(), []
    for path in paths:
        row = json.loads(path.read_text())
        work = path.parent
        actual = read_records((work / "output.raw").read_bytes(), row["clock_cycles"],
                              row["payload_bytes"], row.get("ncc_sync", False), row.get("ncc_participants", 1))
        if actual != row["tiles"]:
            raise RuntimeError("measurement differs from the guarded raw record")
        log = (work / "board.log").read_text()
        fields = dict(item.split("=", 1) for item in log.splitlines()[0].split()[1:])
        identities.add(tuple(fields[key] for key in
                             ("name", "pci_bus_id", "runtime_version", "tiles", "runtime_library_digest")))
        if ("board_execution: true" not in log or "board_stage: cleanup" not in log
                or f"device_elapsed_ns={row['device_elapsed_ns']}\n" not in log):
            raise RuntimeError("clock record lacks matching successful device timing")
        evidence.append({"measurement": str(path), "manifest_sha256": hashlib.sha256(
            (work / "package/manifest.json").read_bytes()).hexdigest()})
        rows.append(row)
    if len(identities) != 1:
        raise RuntimeError("clock and latency measurements have different device identities")
    clocks = sorted((row for row in rows if row["clock_cycles"]), key=lambda row: row["clock_cycles"])
    if len(clocks) != 3 or len({row["clock_cycles"] for row in clocks}) != 3:
        raise RuntimeError("clock calibration requires three distinct bounded intervals")
    cycles = [statistics.median(tile["cycles"] for tile in row["tiles"]) for row in clocks]
    elapsed = [row["device_elapsed_ns"] for row in clocks]
    scale = (elapsed[2] - elapsed[0]) / (cycles[2] - cycles[0])
    adjacent = [(elapsed[i + 1] - elapsed[i]) / (cycles[i + 1] - cycles[i]) for i in range(2)]
    if scale <= 0 or any(abs(slope / scale - 1) > 0.05 for slope in adjacent):
        raise RuntimeError("device events do not establish a consistent cycle scale")
    ticks = [statistics.median(tile["timer_ticks"] for tile in row["tiles"]) for row in clocks]
    ratios = [tile["cycles"] / tile["timer_ticks"] for row in clocks for tile in row["tiles"]]
    ratio = statistics.median(ratios)
    if any(abs(value / ratio - 1) > 0.05 for value in ratios):
        raise RuntimeError("Tile clock and CLINT intervals are inconsistent")
    tick_scale = (elapsed[2] - elapsed[0]) / (ticks[2] - ticks[0])
    results = []
    for path, row in zip(paths, rows):
        if row["clock_cycles"]:
            continue
        cold, warm = [], []
        for tile in row["tiles"]:
            for index, sample in enumerate(tile["samples_cycles"]):
                values = dict(sample)
                if not row.get("ncc_sync", False):
                    values["sender_lifecycle"] = sum(sample[key] for key in
                                                     ("send_prepare", "send_issue", "send_wait"))
                    values["both_endpoints"] = sum(sample.values())
                (cold if index == 0 else warm).append(values)
        results.append({"measurement": str(path), "payload_bytes": row["payload_bytes"],
                        "ncc_participants": row.get("ncc_participants", 1),
                        "kind": "ncc" if row.get("ncc_sync", False) else "dte",
                        "first": {key: distribution([sample[key] for sample in cold], scale) for key in cold[0]},
                        "subsequent": {key: distribution([sample[key] for sample in warm], scale) for key in warm[0]}})
    return {"scope": "local CRT call boundaries, includes polling; not pure fabric latency",
            "device_identity": list(next(iter(identities))), "evidence": evidence,
            "ns_per_cycle": scale, "adjacent_ns_per_cycle": adjacent,
            "ns_per_timer_tick": tick_scale,
            "middle_interval_residual_ns": elapsed[1] - elapsed[0] - (cycles[1] - cycles[0]) * scale,
            "cycle_read_overhead_cycles": [tile["cycle_read_overhead"] for tile in clocks[0]["tiles"]],
            "runs": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=pathlib.Path, nargs="+")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.measurements), indent=2) + "\n")


if __name__ == "__main__":
    main()
