#!/usr/bin/env python3
"""Build and serially execute the device-PMU-backed NE tail group."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from collections.abc import Iterable, Mapping

import wafer_board_ne_calibration_probe_test as ne_driver
import wafer_engine_pipeline_characterization_catalog as engine_catalog
import wafer_ne_calibration_catalog as ne_catalog
import wafer_ne_tail_throughput_catalog as catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--wafer-compile", type=pathlib.Path)
    parser.add_argument("--wafer-run", type=pathlib.Path)
    parser.add_argument("--llvm-clangxx", type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument(
        "--single-engine-observations",
        type=pathlib.Path,
        help=(
            "engine-pipeline-observations.json from the same calibration "
            "session; required for board activation"
        ),
    )
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
        "target_profile": ne_driver.package_support.TARGET_PROFILE,
        "launch_abi": ne_driver.package_support.LAUNCH_ABI,
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
    terminal_completion: int,
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
        "runtime_terminal_completion": terminal_completion,
    }


def rank_one_terminal_completion(package: pathlib.Path) -> int:
    manifest = json.loads((package / "manifest.json").read_text())
    entries = manifest.get("entries")
    completions = manifest.get("completions")
    if (
        manifest.get("schema_version") != 5
        or manifest.get("rank_count") != 1
        or not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(completions, list)
        or len(completions) != 1
    ):
        raise RuntimeError(
            "NE tail package terminal completion domain is not rank-one exact"
        )
    entry = entries[0]
    completion = completions[0]
    if not isinstance(entry, Mapping) or not isinstance(completion, Mapping):
        raise RuntimeError("NE tail entry/completion records are not objects")
    completion_id = completion.get("id")
    if (
        entry.get("id") != 0
        or entry.get("rank") != 0
        or type(completion_id) is not int
        or completion_id < 0
        or entry.get("terminal_completion") != completion_id
        or completion.get("rank") != 0
        or completion.get("kind") != "entry_return"
    ):
        raise RuntimeError(
            "NE tail entry does not bind its exact terminal completion"
        )
    return completion_id


def require_exact_board_completion(
    stdout: str, terminal_completion: int, context: str
) -> None:
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
    expected_terminal = (
        f"terminal_completion: {terminal_completion} kind=entry_return"
    )
    terminal_lines = [
        line for line in lines if line.startswith("terminal_completion:")
    ]
    if terminal_lines != [expected_terminal]:
        raise RuntimeError(
            f"{context}: wafer-run terminal completion differs from the "
            "package manifest"
        )
    if lines.count("board_execution: true") != 1:
        raise RuntimeError(
            f"{context}: wafer-run omitted or repeated board execution state"
        )
    if not (
        stage_positions[-1]
        < lines.index(expected_terminal)
        < lines.index("board_execution: true")
    ):
        raise RuntimeError(
            f"{context}: terminal completion is outside the completed "
            "runtime lifecycle"
        )


def load_single_engine_observations(
    path: pathlib.Path | None,
    calibration_session_id: str,
    qualification: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    if path is None:
        raise RuntimeError(
            "NE tail activation requires --single-engine-observations"
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(
            f"NE tail activation prerequisite archive is missing: {resolved}"
        )
    archive = json.loads(resolved.read_text())
    if not isinstance(archive, dict):
        raise RuntimeError(
            "NE tail activation prerequisite archive has an incompatible "
            "schema"
        )
    observations = archive.get("observations")
    if (
        archive.get("schema_version") != 1
        or not isinstance(observations, list)
        or any(not isinstance(row, Mapping) for row in observations)
    ):
        raise RuntimeError(
            "NE tail activation prerequisite archive has an incompatible "
            "schema"
        )
    if archive.get("calibration_session_id") != calibration_session_id:
        raise RuntimeError(
            "NE tail activation prerequisite archive is stale or belongs to "
            "another calibration session"
        )
    if archive.get("board_qualification") != dict(qualification):
        raise RuntimeError(
            "NE tail activation prerequisite board qualification differs "
            "from the current target/runtime/device profile"
        )
    return tuple(observations)


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


def prepare_package(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, tuple[int, int, int], int]:
    ne_driver.configure_package_support()
    ne_driver.package_support.require_build_args(args)
    source = ne_driver.package_support.write_source_program(args)
    package = ne_driver.package_support.compile_seed_package(args, source)
    module_path, resource_ids = ne_driver.package_support.locate_bindings(
        package
    )
    ne_driver.package_support.build_probe(args, package, module_path)
    ne_driver.package_support.verify_no_card(args, package)
    terminal_completion = rank_one_terminal_completion(package)
    return package, resource_ids, terminal_completion


def execute_cases(
    args: argparse.Namespace,
    package: pathlib.Path,
    resource_ids: tuple[int, int, int],
    cases: Iterable[catalog.NETailThroughputCase],
    terminal_completion: int,
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
                result.stdout, terminal_completion, case.key
            )
            validated = ne_driver.validate_output(
                output, source_case, built, sample
            )
            observation = normalize_observation(
                case, sample, validated, terminal_completion
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
                "wafer_board_ne_tail_throughput_test: hardware execution "
                "is not armed; set WAFER_EXECUTE_HARDWARE_TESTS=1",
                file=sys.stderr,
            )
            return 77
        ne_driver.package_support.require_board_args(args)
        calibration_session_id = require_calibration_session(os.environ)
        qualification = board_qualification(args)
    prerequisite_observations: tuple[Mapping[str, object], ...] = ()
    if not args.no_card:
        assert calibration_session_id is not None
        assert qualification is not None
        prerequisite_observations = load_single_engine_observations(
            args.single_engine_observations,
            calibration_session_id,
            qualification,
        )
        require_complete_ne_prerequisites(prerequisite_observations)
    package, resource_ids, terminal_completion = prepare_package(args)
    if args.no_card:
        print(
            "ne_tail_throughput_no_card: "
            + json.dumps(
                {
                    "activation_group": catalog.GROUP_KEY,
                    "board_execution": False,
                    "case_count": len(cases),
                    "package": str(package),
                    "terminal_completion": terminal_completion,
                },
                sort_keys=True,
            )
        )
        return 0
    observations = execute_cases(
        args, package, resource_ids, cases, terminal_completion
    )
    activation = engine_catalog.evaluate_single_engine_activation(
        engine_catalog.Engine.NE,
        (*prerequisite_observations, *observations),
    )
    assert calibration_session_id is not None
    assert qualification is not None
    archive = {
        "schema_version": 1,
        "calibration_session_id": calibration_session_id,
        "board_qualification": qualification,
        "activation_group": catalog.GROUP_KEY,
        "repeat": args.repeat,
        "terminal_completion": terminal_completion,
        "single_engine_observations": str(
            args.single_engine_observations.resolve()
        ),
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
            f"wafer_board_ne_tail_throughput_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
