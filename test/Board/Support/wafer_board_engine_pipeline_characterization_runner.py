#!/usr/bin/env python3
"""Audit or serially run the engine/pipeline characterization matrix."""

from __future__ import annotations

import argparse
import enum
import json
import os
import pathlib
import re
import sys
from collections.abc import Iterable, Mapping

import wafer_board_ncc_execution_probe_runner as ncc_driver
import wafer_engine_pipeline_characterization_catalog as catalog
import wafer_runtime_launch_contract as runtime_launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("audit", "no-card", "board", "production-gate"),
        default="audit",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(family.value for family in catalog.Family),
        help="select every cell in a family for audit/no-card listing",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="select an exact matrix cell; repeatable",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        choices=tuple(catalog.BOARD_ACTIVATION_GROUPS_BY_KEY),
        help=(
            "select one complete board activation group; repeatable and "
            "preferred for durable characterization"
        ),
    )
    parser.add_argument(
        "--activation-group",
        action="append",
        default=[],
        help=(
            "single:<engine> or "
            "pair:<a>-<b>:<same-worker|cross-worker>:i<2|4|8>"
        ),
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--emit-board-cell-keys",
        action="store_true",
        help="print one CMake-list-safe board-executable cell key per line",
    )
    parser.add_argument(
        "--emit-board-group-keys",
        action="store_true",
        help="print one complete board activation group key per line",
    )
    parser.add_argument("--production-package", type=pathlib.Path)
    parser.add_argument("--require-production-ready", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=10000)
    parser.add_argument("--repeat", type=int, default=catalog.MIN_REPEATS)
    return parser.parse_args()


def _parse_engine(value: str) -> catalog.Engine:
    try:
        return catalog.Engine(value)
    except ValueError as error:
        raise RuntimeError(f"unknown engine {value!r}") from error


def activation_group_cell_keys(spec: str) -> tuple[str, ...]:
    parts = spec.split(":")
    if len(parts) == 2 and parts[0] == "single":
        return catalog.single_activation_cell_keys(_parse_engine(parts[1]))
    if len(parts) == 4 and parts[0] == "pair":
        pair = parts[1].split("-")
        if len(pair) != 2:
            raise RuntimeError(f"malformed pair activation group {spec!r}")
        engine_a, engine_b = map(_parse_engine, pair)
        try:
            placement = catalog.Placement(parts[2])
        except ValueError as error:
            raise RuntimeError(
                f"malformed placement in activation group {spec!r}"
            ) from error
        iteration_text = parts[3]
        if not iteration_text.startswith("i"):
            raise RuntimeError(f"malformed iteration in group {spec!r}")
        try:
            iterations = int(iteration_text[1:])
        except ValueError as error:
            raise RuntimeError(
                f"malformed iteration in group {spec!r}"
            ) from error
        return catalog.pair_activation_cell_keys(
            engine_a, engine_b, placement, iterations
        )
    raise RuntimeError(f"malformed activation group {spec!r}")


def selected_cell_keys(args: argparse.Namespace) -> tuple[str, ...]:
    keys: list[str] = list(args.cell)
    for group_key in args.group:
        keys.extend(
            catalog.BOARD_ACTIVATION_GROUPS_BY_KEY[group_key].cell_keys
        )
    for spec in args.activation_group:
        keys.extend(activation_group_cell_keys(spec))
    if args.family:
        requested = {catalog.Family(value) for value in args.family}
        keys.extend(
            cell.key
            for cell in catalog.ALL_CELLS
            if cell.family in requested
        )
    if not keys:
        if args.mode in ("audit", "no-card"):
            keys.extend(cell.key for cell in catalog.ALL_CELLS)
        elif args.mode == "production-gate":
            keys.extend(cell.key for cell in catalog.THREE_STAGE_CELLS)
    missing = sorted(set(keys) - set(catalog.CELLS_BY_KEY))
    if missing:
        raise RuntimeError(f"unknown characterization cells: {missing}")
    return tuple(dict.fromkeys(keys))


def inventory_record(keys: Iterable[str]) -> dict[str, object]:
    cells = tuple(catalog.CELLS_BY_KEY[key] for key in keys)
    by_disposition: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for cell in cells:
        by_disposition[cell.disposition.value] = (
            by_disposition.get(cell.disposition.value, 0) + 1
        )
        by_family[cell.family.value] = (
            by_family.get(cell.family.value, 0) + 1
        )
    return {
        "cell_count": len(cells),
        "probe_count": sum(len(cell.probes) for cell in cells),
        "by_family": dict(sorted(by_family.items())),
        "by_disposition": dict(sorted(by_disposition.items())),
        "cells": [cell.as_dict() for cell in cells],
    }


def _require_execution_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("wafer_compile", "wafer_run", "llvm_clangxx", "work_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise RuntimeError(
            "no-card/board execution requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    if args.repeat < catalog.MIN_REPEATS:
        raise RuntimeError(
            f"performance characterization requires >= "
            f"{catalog.MIN_REPEATS} repeats"
        )


def _raw_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=args.repo_root.resolve(),
        wafer_compile=args.wafer_compile.resolve(),
        wafer_run=args.wafer_run.resolve(),
        llvm_clangxx=args.llvm_clangxx.resolve(),
        work_dir=ncc_driver.validate_work_dir(
            args.repo_root, args.work_dir
        ),
        no_card=args.mode == "no-card",
        device_id=args.device_id,
        expected_runtime_version=args.expected_runtime_version,
        expected_device_name=args.expected_device_name,
        expected_pci_bus_id=args.expected_pci_bus_id,
        expected_tile_count=args.expected_tile_count,
        expected_runtime_library_sha256=(
            args.expected_runtime_library_sha256
        ),
        completion_timeout_ms=args.completion_timeout_ms,
        repeat=args.repeat,
    )


def require_calibration_session(
    environment: Mapping[str, str],
) -> str:
    session_id = environment.get(
        catalog.CALIBRATION_SESSION_ENVIRONMENT, ""
    )
    if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
        raise RuntimeError(
            f"board characterization requires "
            f"{catalog.CALIBRATION_SESSION_ENVIRONMENT} as 32 lowercase "
            "hex digits"
        )
    return session_id


def board_qualification(args: argparse.Namespace) -> dict[str, object]:
    digest = str(args.expected_runtime_library_sha256).lower()
    qualification = {
        "target_identity": ncc_driver.TARGET_IDENTITY,
        "launch": runtime_launch.GRID_KERNEL_LAUNCH,
        "device_id": args.device_id,
        "expected_runtime_version": args.expected_runtime_version,
        "expected_device_name": args.expected_device_name,
        "expected_pci_bus_id": args.expected_pci_bus_id,
        "expected_tile_count": args.expected_tile_count,
        "expected_runtime_library_sha256": digest,
    }
    if (
        type(qualification["device_id"]) is not int
        or type(qualification["expected_runtime_version"]) is not int
        or not isinstance(qualification["expected_device_name"], str)
        or not isinstance(qualification["expected_pci_bus_id"], str)
        or type(qualification["expected_tile_count"]) is not int
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RuntimeError(
            "engine probe board qualification fingerprint is incomplete"
        )
    return qualification


def package_completion_kind(package: pathlib.Path) -> str:
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_complete_tile_domain(
        manifest, context="engine probe"
    )
    return runtime_launch.LOCAL_DRAIN_COMPLETION


def require_exact_board_completion(
    stdout: str, completion_kind: str, context: str
) -> None:
    if completion_kind != runtime_launch.LOCAL_DRAIN_COMPLETION:
        raise RuntimeError(f"{context}: package completion kind is invalid")
    lines = stdout.splitlines()
    stage_lines = [
        f"board_stage: {stage}" for stage in catalog.RUNTIME_LIFECYCLE
    ]
    stage_positions: list[int] = []
    for line in stage_lines:
        if lines.count(line) != 1:
            raise RuntimeError(
                f"{context}: wafer-run omitted or repeated lifecycle stage "
                f"{line!r}"
            )
        stage_positions.append(lines.index(line))
    if stage_positions != sorted(stage_positions):
        raise RuntimeError(
            f"{context}: wafer-run lifecycle stages are out of order"
        )
    runtime_launch.require_board_completion(stdout, context=context)
    first_completion = lines.index(
        f"completion: {completion_kind} tile_id=0"
    )
    if not (
        stage_positions[-1]
        < first_completion
        < lines.index("board_execution: true")
    ):
        raise RuntimeError(
            f"{context}: Tile completion is outside the completed "
            "runtime lifecycle"
        )


def counterbalanced_probe_schedule(
    probes: Iterable[catalog.ProbeCase], repeat: int
) -> tuple[tuple[catalog.ProbeCase, int], ...]:
    ordered = tuple(probes)
    if not ordered:
        raise ValueError("counterbalanced schedule requires at least one probe")
    if repeat < 1:
        raise ValueError("counterbalanced schedule repeat must be positive")
    half_rotation = max(1, len(ordered) // 2)
    schedule: list[tuple[catalog.ProbeCase, int]] = []
    for sample in range(repeat):
        sweep = ordered if sample % 2 == 0 else tuple(reversed(ordered))
        offset = ((sample // 2) * half_rotation) % len(sweep)
        sweep = sweep[offset:] + sweep[:offset]
        schedule.extend((probe, sample) for probe in sweep)
    return tuple(schedule)


def prepare_probe_package(
    args: argparse.Namespace,
    probes: tuple[catalog.ProbeCase, ...],
) -> tuple[argparse.Namespace, pathlib.Path, tuple[int, int, int], str]:
    raw_args = _raw_args(args)
    ncc_driver.validate_no_card_protocol_cases()
    ncc_driver.validate_catalog_resource_layout(probes)
    source = ncc_driver.write_source_program(raw_args.work_dir)
    package = ncc_driver.compile_seed_package(raw_args, source)
    module_path, resource_ids, slots_per_tile = (
        ncc_driver.locate_probe_bindings(package)
    )
    ncc_driver.build_probe(
        raw_args, package, module_path, slots_per_tile
    )
    ncc_driver.verify_no_card(raw_args, package)
    completion_kind = package_completion_kind(package)
    return raw_args, package, resource_ids, completion_kind


def execute_probes(
    args: argparse.Namespace,
    raw_args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    probes: Iterable[catalog.ProbeCase],
    completion_kind: str,
) -> list[dict[str, object]]:
    raw_dir = raw_args.work_dir / "raw"
    raw_dir.mkdir()
    observations: list[dict[str, object]] = []
    for launch_ordinal, (probe, sample) in enumerate(
        counterbalanced_probe_schedule(probes, args.repeat)
    ):
        request = raw_dir / f"{probe.name}.{sample}.request.raw"
        payload = raw_dir / f"{probe.name}.{sample}.payload.raw"
        output = raw_dir / f"{probe.name}.{sample}.output.raw"
        ncc_driver.write_request(request, probe, sample)
        ncc_driver.write_payload(payload, probe)
        result = ncc_driver.run(
            ncc_driver.board_command(
                raw_args,
                package,
                resource_ids,
                request,
                payload,
                output,
            ),
            timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0,
        )
        require_exact_board_completion(
            result.stdout, completion_kind, probe.key
        )
        observation = ncc_driver.parse_record(output, probe, sample)
        observation["launch_ordinal"] = launch_ordinal
        observation["runtime_lifecycle"] = catalog.RUNTIME_LIFECYCLE
        observation["runtime_completion_kind"] = completion_kind
        observations.append(observation)
        print(
            "engine_pipeline_sample: "
            + json.dumps(observation, sort_keys=True)
        )
    return observations


def activation_decisions(
    keys: Iterable[str],
    observations: list[Mapping[str, object]],
) -> tuple[catalog.ActivationDecision, ...]:
    cells = tuple(catalog.CELLS_BY_KEY[key] for key in keys)
    decisions: list[catalog.ActivationDecision] = []
    single_engines = {
        engine
        for engine in catalog.Engine
        if any(
            cell.key in catalog.single_activation_cell_keys(engine)
            for cell in cells
        )
    }
    for engine in sorted(single_engines, key=lambda item: item.value):
        decisions.append(
            catalog.evaluate_single_engine_activation(engine, observations)
        )

    pair_scopes = {
        (
            catalog.Engine(str(cell.dimension("engine_a"))),
            catalog.Engine(str(cell.dimension("engine_b"))),
            catalog.Placement(str(cell.dimension("placement"))),
            int(cell.dimension("iterations")),
        )
        for cell in cells
        if cell.family == catalog.Family.ENGINE_PAIR_STAGE_BALANCE
    }
    for engine_a, engine_b, placement, iterations in sorted(
        pair_scopes,
        key=lambda value: tuple(
            item.value if isinstance(item, enum.Enum) else item
            for item in value
        ),
    ):
        decisions.append(
            catalog.evaluate_pair_activation(
                engine_a,
                engine_b,
                placement,
                iterations,
                observations,
            )
        )
    return tuple(decisions)


def main() -> int:
    args = parse_args()
    catalog.validate_catalog()
    if args.emit_board_cell_keys:
        print(
            "\n".join(
                cell.key
                for cell in catalog.ALL_CELLS
                if cell.disposition == catalog.Disposition.BOARD_EXECUTABLE
            )
        )
        return 0
    if args.emit_board_group_keys:
        print(
            "\n".join(
                group.key for group in catalog.BOARD_ACTIVATION_GROUPS
            )
        )
        return 0
    keys = selected_cell_keys(args)
    record = inventory_record(keys)
    if args.list_cases or args.mode == "audit":
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    if args.mode == "production-gate":
        decision = catalog.production_pipeline_preparation_gate(
            args.production_package
        )
        print(
            "production_pipeline_preparation: "
            + json.dumps(decision.as_dict(), sort_keys=True)
        )
        return int(args.require_production_ready and not decision.ready)

    _require_execution_args(args)
    selected = tuple(catalog.CELLS_BY_KEY[key] for key in keys)
    non_board = [
        (cell.key, cell.disposition.value)
        for cell in selected
        if cell.disposition != catalog.Disposition.BOARD_EXECUTABLE
    ]
    if args.mode == "board" and non_board:
        raise RuntimeError(
            "board selection contains delegated/fail-closed cells: "
            + repr(non_board)
        )
    probes = tuple(
        probe
        for cell in selected
        if cell.disposition == catalog.Disposition.BOARD_EXECUTABLE
        for probe in cell.probes
    )
    if not probes:
        raise RuntimeError("selection contains no board-executable probes")
    calibration_session_id: str | None = None
    qualification: dict[str, object] | None = None
    if args.mode == "board":
        if not args.cell and not args.activation_group and not args.group:
            raise RuntimeError(
                "board mode requires explicit --cell, --group, or "
                "--activation-group; "
                "a whole-family implicit run is intentionally forbidden"
            )
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "wafer_board_engine_pipeline_characterization_runner: "
                "hardware execution is not armed; set "
                "WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        raw_probe_args = _raw_args(args)
        ncc_driver.validate_board_args(raw_probe_args)
        calibration_session_id = require_calibration_session(os.environ)
        qualification = board_qualification(raw_probe_args)

    raw_args, package, resource_ids, completion_kind = (
        prepare_probe_package(args, probes)
    )
    if args.mode == "no-card":
        print(
            "engine_pipeline_no_card: "
            + json.dumps(
                {
                    "selected_cells": len(selected),
                    "serialized_probes": len(probes),
                    "package": str(package),
                    "completion_kind": completion_kind,
                    "board_execution": False,
                },
                sort_keys=True,
            )
        )
        return 0

    observations = execute_probes(
        args,
        raw_args,
        package,
        resource_ids,
        probes,
        completion_kind,
    )
    decisions = activation_decisions(keys, observations)
    assert calibration_session_id is not None
    assert qualification is not None
    archive = {
        "calibration_session_id": calibration_session_id,
        "board_qualification": qualification,
        "catalog": record,
        "repeat": args.repeat,
        "completion_kind": completion_kind,
        "execution_order": (
            "sample-major alternating forward/reverse sweeps with paired "
            "half-rotation"
        ),
        "observations": observations,
        "activation_decisions": [
            decision.as_dict() for decision in decisions
        ],
        "interpretation": (
            "only activated decisions may seed a narrow profile-scoped cost; "
            "unknown retains the conservative compiler fallback"
        ),
    }
    archive_path = raw_args.work_dir / "engine-pipeline-observations.json"
    archive_path.write_text(
        json.dumps(archive, indent=2, sort_keys=True) + "\n"
    )
    for decision in decisions:
        print(
            "engine_pipeline_activation: "
            + json.dumps(decision.as_dict(), sort_keys=True)
        )
    print(f"engine_pipeline_archive: {archive_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            "wafer_board_engine_pipeline_characterization_runner: "
            + str(error),
            file=sys.stderr,
        )
        raise SystemExit(1)
