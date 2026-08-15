#!/usr/bin/env python3
"""Validate the model-scale M-sharded GEMM production/profile output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

import wafer_board_all_rank_add_test as profile_support
import wafer_board_compiler_optimization_campaign_test as campaign
import wafer_runtime_launch_contract as runtime_launch


def require_stable_production_structure(
    ordinary: campaign.TargetStructure, profiled: campaign.TargetStructure
) -> None:
    campaign.require_call(ordinary.counts, "_gemm", present=True)
    campaign.require_call(profiled.counts, "_gemm", present=True)
    if ordinary != profiled:
        raise RuntimeError(
            "ordinary and profiled production target structures differ"
        )


CASE = campaign.CampaignCase(
    key="m-sharded-replicated-gemm-profile",
    family="compiler-search-scalability",
    rank_count=campaign.NOC_RESIDENT_M_SHARDED_GEMM_RANKS,
    launch_kind=runtime_launch.KERNEL_LAUNCH_KIND,
    inputs=(
        campaign.F16_NOC_RESIDENT_M_SHARDED_GEMM_LHS_LOCAL,
        campaign.F16_NOC_RESIDENT_M_SHARDED_GEMM_RHS_LOCAL,
    ),
    outputs=(campaign.F16_NOC_RESIDENT_M_SHARDED_GEMM_OUTPUT_LOCAL,),
    module_factory=campaign.noc_resident_m_sharded_gemm_module,
    payload_factory=campaign.noc_resident_m_sharded_gemm_payloads,
    structural_oracle=require_stable_production_structure,
    expected_launch=runtime_launch.GRID_KERNEL_LAUNCH,
)
PROFILE_CAMPAIGN_LAUNCH_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafer-compile", type=pathlib.Path, required=True)
    parser.add_argument("--wafer-run", type=pathlib.Path, required=True)
    parser.add_argument("--tx8-objdump", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--expected-runtime-version", type=int)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--expected-pci-bus-id")
    parser.add_argument("--expected-tile-count", type=int)
    parser.add_argument("--expected-runtime-library-sha256")
    parser.add_argument("--completion-timeout-ms", type=int, default=60000)
    return parser.parse_args()


def write_source(work_dir: pathlib.Path) -> pathlib.Path:
    known_children = {
        "source-program",
        "ordinary-package",
        "profile-package",
        "profile-package.profile",
        "raw",
        "target-structure.json",
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    unknown_children = [
        child for child in work_dir.iterdir() if child.name not in known_children
    ]
    if unknown_children:
        raise RuntimeError(
            "--work-dir contains unknown entries; refusing cleanup: "
            + ", ".join(sorted(child.name for child in unknown_children))
        )
    for name in sorted(known_children):
        child = work_dir / name
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    source = work_dir / "source-program"
    (source / "functions").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "functions" / "forward.mlir").write_text(CASE.module_factory())
    (source / "functions" / "forward.meta").write_text(
        json.dumps(campaign.metadata(CASE), separators=(",", ":")) + "\n"
    )
    return source


def require_byte_identical_packages(
    ordinary: pathlib.Path, profiled: pathlib.Path
) -> None:
    def file_digests(root: pathlib.Path) -> dict[pathlib.Path, str]:
        return {
            path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    ordinary_digests = file_digests(ordinary)
    profiled_digests = file_digests(profiled)
    if not ordinary_digests or ordinary_digests != profiled_digests:
        raise RuntimeError(
            "ordinary and profiled production packages are not byte-identical"
        )


def require_profile_no_card(stdout: str, expected: bool) -> None:
    if "board_execution: false" not in stdout:
        raise RuntimeError("no-card invocation omitted execution state")
    instrumentation_ready = profile_support.PROFILE_INSTRUMENTATION_READY in stdout
    if instrumentation_ready != expected:
        raise RuntimeError(
            "no-card invocation did not prove the expected profile instrumentation "
            "activation boundary"
        )


def main() -> int:
    args = parse_args()
    if args.completion_timeout_ms < 1:
        raise RuntimeError("completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("M-sharded GEMM profile hardware execution is not armed")
        return 77
    if not args.no_card:
        qualification = (
            args.expected_runtime_version,
            args.expected_device_name,
            args.expected_pci_bus_id,
            args.expected_tile_count,
            args.expected_runtime_library_sha256,
        )
        if any(value is None for value in qualification):
            raise RuntimeError(
                "board execution requires complete qualification arguments"
            )
        if args.expected_tile_count != CASE.rank_count:
            raise RuntimeError(
                f"--expected-tile-count must be {CASE.rank_count}"
            )

    payloads = CASE.payload_factory()
    campaign.validate_paired_payloads(CASE, payloads)
    source = write_source(args.work_dir)
    packages = {
        "baseline": args.work_dir / "ordinary-package",
        "winner": args.work_dir / "profile-package",
    }
    campaign.compile_package(
        args.wafer_compile,
        source,
        packages["baseline"],
        CASE,
        reserved_baseline=False,
    )
    campaign.compile_package(
        args.wafer_compile,
        source,
        packages["winner"],
        CASE,
        reserved_baseline=False,
        profile=True,
    )
    require_byte_identical_packages(packages["baseline"], packages["winner"])
    profile_support.require_profile_instrumentation_permissions(packages["winner"])

    (
        bindings_by_variant,
        output_ids_by_variant,
        completion_evidence_by_variant,
    ) = campaign.validate_paired_packages(
        packages["baseline"], packages["winner"], CASE
    )
    structures = {
        name: campaign.target_structure(package, args.tx8_objdump)
        for name, package in packages.items()
    }
    require_stable_production_structure(
        structures["baseline"], structures["winner"]
    )
    (args.work_dir / "target-structure.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case": CASE.key,
                "launch": CASE.launch_contract,
                "rank_count": CASE.rank_count,
                "ordinary_module_digests": list(
                    campaign.module_digests(packages["baseline"])
                ),
                "profiled_module_digests": list(
                    campaign.module_digests(packages["winner"])
                ),
                "scheduler_body_sha256": list(
                    structures["winner"].scheduler_body_sha256
                ),
                "static_target_callsite_counts": dict(
                    sorted(structures["winner"].counts.items())
                ),
                "workspace_bytes": structures["winner"].workspace_bytes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    resource_arguments = campaign.write_payloads(
        args.work_dir, CASE, bindings_by_variant, payloads
    )

    if args.no_card:
        for name, package in packages.items():
            result = campaign.run(
                [
                    str(args.wafer_run),
                    "--package-dir",
                    str(package),
                    "--all-ranks",
                    "--no-card",
                ]
            )
            require_profile_no_card(result.stdout, expected=name == "winner")
        print(
            "m_sharded_gemm_profile_no_card: "
            "ordinary_profile_identical=true grid_launch=true "
            "profile_instrumentation=true"
        )
        return 0

    result = campaign.run(
        campaign.board_command(
            args, packages["winner"], CASE, resource_arguments["winner"]
        ),
        timeout_seconds=max(
            300.0,
            PROFILE_CAMPAIGN_LAUNCH_COUNT
            * args.completion_timeout_ms
            / 1000.0
            + campaign.PROCESS_TIMEOUT_MARGIN_SECONDS,
        ),
    )
    campaign.verify_board_output(
        result.stdout,
        CASE,
        output_ids_by_variant["winner"],
        completion_evidence_by_variant["winner"],
    )
    campaign.compare_captured_outputs(
        args.work_dir, CASE, payloads, "winner"
    )
    device_duration, report = profile_support.verify_profile_report(
        packages["winner"], result.stdout, expected_active_engine="NE"
    )
    print(
        "m_sharded_gemm_profile_board: pass "
        f"device_duration_ns={device_duration} report={report}"
    )
    print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"wafer_board_m_sharded_gemm_profile_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
