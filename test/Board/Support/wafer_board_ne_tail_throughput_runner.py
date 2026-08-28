#!/usr/bin/env python3
"""Build and serially execute the device-PMU-backed NE tail group."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys
from collections.abc import Iterable, Mapping

import wafer_board_engine_pipeline_characterization_runner as engine_driver
import wafer_board_ne_calibration_probe_runner as ne_driver
import wafer_engine_pipeline_characterization_catalog as engine_catalog
import wafer_ne_calibration_catalog as ne_catalog
import wafer_ne_tail_throughput_catalog as catalog
import wafer_runtime_launch_contract as runtime_launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument(
        "--group",
        choices=(catalog.GROUP_KEY,),
        default=catalog.GROUP_KEY,
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--emit-board-cell-key", action="store_true")
    parser.add_argument("--emit-board-group-key", action="store_true")
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    parser.add_argument("--repeat", type=int, default=catalog.MINIMUM_REPEATS)
    return parser.parse_args()


def selected_cases(group: str) -> tuple[catalog.NETailThroughputCase, ...]:
    if group != catalog.GROUP_KEY:
        raise RuntimeError(f"unknown NE tail activation group {group!r}")
    return catalog.CASES


def _require_repeat(repeat: int) -> None:
    if repeat < catalog.MINIMUM_REPEATS:
        raise RuntimeError(
            f"NE tail characterization requires >= "
            f"{catalog.MINIMUM_REPEATS} repeats"
        )


def require_calibration_session(
    environment: Mapping[str, str],
) -> str:
    session_id = environment.get(
        engine_catalog.CALIBRATION_SESSION_ENVIRONMENT, ""
    )
    if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
        raise RuntimeError(
            f"NE tail characterization requires "
            f"{engine_catalog.CALIBRATION_SESSION_ENVIRONMENT} as 32 "
            "lowercase hex digits"
        )
    return session_id


def board_qualification(args: argparse.Namespace) -> dict[str, object]:
    digest = str(args.expected_runtime_library_sha256).lower()
    qualification = {
        "target_identity": ne_driver.package_support.TARGET_IDENTITY,
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
            "NE tail board qualification fingerprint is incomplete"
        )
    return qualification


def normalize_observation(
    case: catalog.NETailThroughputCase,
    sample: int,
    validated: Mapping[str, object],
    completion_kind: str,
) -> dict[str, object]:
    pmu = validated.get("pmu")
    if not isinstance(pmu, Mapping):
        raise RuntimeError(f"{case.key}: validated result omitted PMU data")
    ne_instructions = pmu.get("ne_instruction_delta")
    ne_execution = pmu.get("ne_execution_delta")
    ne_blocking = pmu.get("ne_blocking_delta")
    if (
        ne_instructions != case.expected_ne_instructions
        or type(ne_execution) is not int
        or ne_execution <= 0
        or type(ne_blocking) is not int
    ):
        raise RuntimeError(f"{case.key}: incompatible NE PMU observation")
    return {
        "case": case.as_dict(),
        "sample": sample,
        "source_case": validated["case"],
        "instruction_delta": {"worker0.ne": ne_instructions},
        "execution_delta": {"ne": ne_execution},
        "blocking_delta": {"ne": ne_blocking},
        "pmu_enable": pmu["enable"],
        "pmu_base": pmu["base"],
        "execute_result": validated["execute_result"],
        "logical_sha256": validated["logical_sha256"],
        "physical_sha256": validated["physical_sha256"],
        "runtime_lifecycle": engine_catalog.RUNTIME_LIFECYCLE,
        "runtime_completion_kind": completion_kind,
    }


def package_completion_kind(package: pathlib.Path) -> str:
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_complete_tile_domain(manifest, context="NE tail")
    return runtime_launch.LOCAL_DRAIN_COMPLETION


def require_exact_board_completion(
    stdout: str, completion_kind: str, context: str
) -> None:
    if completion_kind != runtime_launch.LOCAL_DRAIN_COMPLETION:
        raise RuntimeError(f"{context}: package completion kind is invalid")
    lines = stdout.splitlines()
    stage_lines = [
        f"board_stage: {stage}" for stage in engine_catalog.RUNTIME_LIFECYCLE
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


def fresh_ne_prerequisite_probes() -> tuple[engine_catalog.ProbeCase, ...]:
    cell_keys = engine_catalog.single_activation_cell_keys(
        engine_catalog.Engine.NE
    )
    expected = catalog.CASES[0].activation_prerequisite_cells
    if cell_keys != expected:
        raise RuntimeError(
            "NE tail fresh prerequisite cells differ from the activation "
            "contract"
        )
    probes = tuple(
        probe
        for cell_key in cell_keys
        for probe in engine_catalog.CELLS_BY_KEY[cell_key].probes
    )
    if len(probes) != len(cell_keys):
        raise RuntimeError(
            "NE tail fresh prerequisite cells do not map one-to-one to probes"
        )
    return probes


def require_complete_ne_prerequisites(
    observations: Iterable[Mapping[str, object]],
) -> None:
    decision = engine_catalog.evaluate_single_engine_activation(
        engine_catalog.Engine.NE, observations
    )
    expected_tail_reason = (
        f"{engine_catalog.NE_TAIL_CHARACTERIZATION_KEY}: needs "
        f"{engine_catalog.MIN_REPEATS} distinct external samples"
    )
    if decision.reasons != (expected_tail_reason,):
        raise RuntimeError(
            "NE tail activation prerequisite observations are incomplete: "
            + "; ".join(decision.reasons or ("unexpected tail evidence",))
        )


def _with_work_dir(
    args: argparse.Namespace, work_dir: pathlib.Path
) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.work_dir = work_dir
    return copied


def reset_work_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.repo_root is None or args.work_dir is None:
        raise RuntimeError(
            "NE tail execution requires --repo-root and --work-dir"
        )
    resolved = engine_driver.ncc_driver.validate_work_dir(
        args.repo_root, args.work_dir
    )
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)
    args.work_dir = resolved
    return resolved


def prepare_fresh_ne_prerequisites(
    args: argparse.Namespace,
) -> tuple[
    argparse.Namespace,
    pathlib.Path,
    tuple[int, int, int],
    int,
    tuple[engine_catalog.ProbeCase, ...],
]:
    prerequisite_args = _with_work_dir(
        args, args.work_dir / "ne-small-steady"
    )
    prerequisite_args.mode = "no-card" if args.no_card else "board"
    engine_driver._require_execution_args(prerequisite_args)
    probes = fresh_ne_prerequisite_probes()
    raw_args, package, resource_ids, completion_kind = (
        engine_driver.prepare_probe_package(prerequisite_args, probes)
    )
    return raw_args, package, resource_ids, completion_kind, probes


def prepare_package(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, tuple[int, int, int], str]:
    ne_driver.configure_package_support()
    ne_driver.package_support.require_build_args(args)
    source = ne_driver.package_support.write_source_program(args)
    package = ne_driver.package_support.compile_seed_package(args, source)
    module_path, resource_ids, slots_per_tile = (
        ne_driver.package_support.locate_bindings(package)
    )
    ne_driver.package_support.build_probe(
        args, package, module_path, slots_per_tile
    )
    ne_driver.package_support.verify_no_card(args, package)
    completion_kind = package_completion_kind(package)
    return package, resource_ids, completion_kind


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.NETailThroughputCase],
    completion_kind: str,
) -> list[dict[str, object]]:
    raw_dir = args.work_dir / "raw"
    raw_dir.mkdir()
    observations: list[dict[str, object]] = []
    for case in cases:
        source_case = case.source_case
        for sample in range(args.repeat):
            built = ne_catalog.build_case_payload(source_case, sample)
            stem = f"{case.name}.sample-{sample}"
            request = raw_dir / f"{stem}.request.raw"
            payload = raw_dir / f"{stem}.payload.raw"
            output = raw_dir / f"{stem}.output.raw"
            request.write_bytes(built.request)
            payload.write_bytes(built.payload)
            result = ne_driver.package_support.run(
                ne_driver.package_support.board_command(
                    args,
                    package,
                    resource_ids,
                    request,
                    payload,
                    output,
                ),
                timeout_seconds=args.completion_timeout_ms / 1000.0 + 30.0,
            )
            require_exact_board_completion(
                result.stdout, completion_kind, case.key
            )
            validated = ne_driver.validate_output(
                output, source_case, built, sample
            )
            observation = normalize_observation(
                case, sample, validated, completion_kind
            )
            observations.append(observation)
            print(
                "ne_tail_throughput_sample: "
                + json.dumps(observation, sort_keys=True)
            )
    return observations


def main() -> int:
    args = parse_args()
    catalog.validate_catalog()
    if args.emit_board_cell_key:
        print(catalog.CELL_KEY)
        return 0
    if args.emit_board_group_key:
        print(catalog.GROUP_KEY)
        return 0
    cases = selected_cases(args.group)
    if args.list_cases:
        print(
            json.dumps(
                [case.as_dict() for case in cases],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    _require_repeat(args.repeat)
    calibration_session_id: str | None = None
    qualification: dict[str, object] | None = None
    if not args.no_card:
        if os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
            print(
                "wafer_board_ne_tail_throughput_runner: hardware execution "
                "is not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        ne_driver.package_support.require_board_args(args)
        calibration_session_id = require_calibration_session(os.environ)
        qualification = board_qualification(args)
    reset_work_dir(args)
    (
        prerequisite_args,
        prerequisite_package,
        prerequisite_resource_ids,
        prerequisite_completion_kind,
        prerequisite_probes,
    ) = prepare_fresh_ne_prerequisites(args)
    prerequisite_observations: tuple[Mapping[str, object], ...]
    if args.no_card:
        prerequisite_observations = ()
    else:
        prerequisite_observations = tuple(
            engine_driver.execute_probes(
                args,
                prerequisite_args,
                prerequisite_package,
                prerequisite_resource_ids,
                prerequisite_probes,
                prerequisite_completion_kind,
            )
        )
        require_complete_ne_prerequisites(prerequisite_observations)
    tail_args = _with_work_dir(args, args.work_dir / "tail")
    package, resource_ids, completion_kind = prepare_package(tail_args)
    if args.no_card:
        print(
            "ne_tail_throughput_no_card: "
            + json.dumps(
                {
                    "activation_group": catalog.GROUP_KEY,
                    "board_execution": False,
                    "case_count": len(cases),
                    "prerequisite_case_count": len(prerequisite_probes),
                    "prerequisite_package": str(prerequisite_package),
                    "package": str(package),
                    "completion_kind": completion_kind,
                },
                sort_keys=True,
            )
        )
        return 0
    observations = execute_cases(
        tail_args, package, resource_ids, cases, completion_kind
    )
    activation = engine_catalog.evaluate_single_engine_activation(
        engine_catalog.Engine.NE,
        (*prerequisite_observations, *observations),
    )
    assert calibration_session_id is not None
    assert qualification is not None
    archive = {
        "calibration_session_id": calibration_session_id,
        "board_qualification": qualification,
        "activation_group": catalog.GROUP_KEY,
        "repeat": args.repeat,
        "completion_kind": completion_kind,
        "prerequisite_generation": "fresh-invocation",
        "prerequisite_observations": prerequisite_observations,
        "observations": observations,
        "activation_decision": activation.as_dict(),
        "interpretation": (
            "raw NE device counters are retained and activation is fail-closed "
            "over the merged small/steady/tail device-PMU triplet"
        ),
    }
    archive_path = args.work_dir / "ne-tail-throughput-observations.json"
    archive_path.write_text(
        json.dumps(archive, indent=2, sort_keys=True) + "\n"
    )
    print(
        "ne_tail_throughput_activation: "
        + json.dumps(activation.as_dict(), sort_keys=True)
    )
    print(f"ne_tail_throughput_archive: {archive_path}")
    if not activation.activated:
        raise RuntimeError(
            "NE small/steady/tail activation remains unknown: "
            + "; ".join(activation.reasons)
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            f"wafer_board_ne_tail_throughput_runner: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
