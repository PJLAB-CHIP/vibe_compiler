#!/usr/bin/env python3
"""Build and validate the matched Direct-DTE/compute overlap qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

import torch

import wafer_board_compiler_optimization_campaign_test as campaign


TARGET_IDENTITY = "wafer-tx81-single-card"
BASELINE_SELECTION = "WAFER_TEST_SELECT_SERIALIZED_DIRECT_DTE_COMPUTE"
OVERLAP_SELECTION = "WAFER_TEST_SELECT_DIRECT_DTE_COMPUTE_OVERLAP"
SELECTION_ENVIRONMENT_VARIABLES = (
    BASELINE_SELECTION,
    OVERLAP_SELECTION,
    "WAFER_TEST_SELECT_RESERVED_BASELINE",
    "WAFER_TEST_SELECT_STATIC_FIXED_SLOT",
    "WAFER_TEST_SELECT_WORKER_PLACEMENT",
    "WAFER_TEST_SELECT_NOC_RESIDENT_FIXED_SLOT_WORKER",
    "WAFER_TEST_COLLECTIVE_CHARACTERIZATION_ALTERNATIVE",
    "WAFER_TEST_COLLECTIVE_CHARACTERIZATION_REPORT",
)
ELEMENT_COUNT = 524288


def overlap_module() -> str:
    return f"""\
module {{
  func.func @main(
      %lhs: tensor<{ELEMENT_COUNT}xf16>
          {{mhlo.sharding = "{{replicated}}"}},
      %rhs: tensor<{ELEMENT_COUNT}xf16>
          {{mhlo.sharding = "{{replicated}}"}})
      -> (tensor<{ELEMENT_COUNT}xf16>
          {{mhlo.sharding = "{{replicated}}"}}) {{
    %add = stablehlo.add %lhs, %lhs : tensor<{ELEMENT_COUNT}xf16>
    %mul = stablehlo.multiply %rhs, %rhs : tensor<{ELEMENT_COUNT}xf16>
    %result = stablehlo.add %add, %mul : tensor<{ELEMENT_COUNT}xf16>
    return %result : tensor<{ELEMENT_COUNT}xf16>
  }}
}}
"""


def overlap_payloads() -> campaign.PairedPayloads:
    lhs = campaign.random_f16((ELEMENT_COUNT,), 400)
    rhs = campaign.random_f16((ELEMENT_COUNT,), 401)
    expected = (lhs + lhs) + (rhs * rhs)
    inputs = [[lhs.clone(), rhs.clone()] for _ in range(16)]
    outputs = [[expected.clone()] for _ in range(16)]
    return campaign.PairedPayloads(
        inputs,
        outputs,
        [[expected.clone()] for _ in range(16)],
    )


def overlap_structure_oracle(
    baseline: campaign.TargetStructure,
    winner: campaign.TargetStructure,
) -> None:
    del baseline, winner


CASE = campaign.CampaignCase(
    "direct-dte-compute-overlap",
    "direct-dte-compute-overlap",
    16,
    campaign.CLUSTER_LAUNCH_KIND,
    (campaign.F16_524288, campaign.F16_524288),
    (campaign.F16_524288,),
    overlap_module,
    overlap_payloads,
    overlap_structure_oracle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafer-compile-test", type=pathlib.Path, required=True)
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
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def write_source(work_dir: pathlib.Path) -> pathlib.Path:
    known_children = {
        "source-program",
        "baseline-package",
        "baseline-package.qualification",
        "overlap-package",
        "overlap-package.qualification",
        "raw",
        "target-structure.json",
        "board-observations.json",
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    unknown = [
        child for child in work_dir.iterdir() if child.name not in known_children
    ]
    if unknown:
        raise RuntimeError(
            "--work-dir contains unknown entries; refusing cleanup: "
            + ", ".join(sorted(child.name for child in unknown))
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


def compile_variant(
    compiler: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    selection: str,
) -> None:
    environment = os.environ.copy()
    for variable in (
        *SELECTION_ENVIRONMENT_VARIABLES,
        "WAFER_TEST_FAIL_AFTER_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_TARGET_LOGICAL_RANK",
        "WAFER_TEST_FAIL_AFTER_PACKAGE_LOGICAL_RANK",
    ):
        environment.pop(variable, None)
    environment[selection] = "1"
    result = campaign.run(
        [
            str(compiler),
            "--input-program-dir",
            str(source),
            "--output-program-dir",
            str(output),
            f"--execution-ranks={CASE.rank_count}",
            f"--launch-kind={CASE.launch_kind}",
        ],
        environment=environment,
        timeout_seconds=1800.0,
    )
    expected = (
        "wafer-compile: published verified package with "
        f"execution-ranks={CASE.rank_count}"
    )
    if expected not in result.stdout:
        raise RuntimeError(f"compiler did not publish the {selection} package")


def digest(path: pathlib.Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def validate_overlap_companion(
    package: pathlib.Path,
) -> dict[str, object]:
    companion = pathlib.Path(str(package) + ".qualification")
    attestation_path = companion / "attestation.json"
    activation_path = companion / "activation.json"
    if set(path.name for path in companion.iterdir()) != {
        "attestation.json",
        "activation.json",
    }:
        raise RuntimeError("overlap qualification companion inventory is invalid")
    attestation = json.loads(attestation_path.read_text())
    activation = json.loads(activation_path.read_text())
    if (
        attestation.get("schema") != "wafer-static-fixed-slot-qualification"
        or attestation.get("schema_version") != 1
        or attestation.get("selection_kind")
        != "direct-dte-compute-overlap"
        or attestation.get("target")
        != {
            "identity": TARGET_IDENTITY,
            "rank_count": CASE.rank_count,
            "logical_ranks": list(range(CASE.rank_count)),
        }
    ):
        raise RuntimeError("overlap qualification identity is invalid")
    if (
        attestation.get("manifest_sha256") != digest(package / "manifest.json")
        or activation.get("manifest_sha256")
        != attestation.get("manifest_sha256")
        or activation.get("attestation_sha256") != digest(attestation_path)
    ):
        raise RuntimeError("overlap qualification digest binding is invalid")
    ranks = attestation.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != CASE.rank_count:
        raise RuntimeError("overlap qualification rank domain is invalid")
    for expected_rank, rank in enumerate(ranks):
        if (
            not isinstance(rank, dict)
            or rank.get("logical_rank") != expected_rank
            or not isinstance(
                rank.get("direct_dte_compute_overlap_window_count"), int
            )
            or rank["direct_dte_compute_overlap_window_count"] < 1
            or not rank.get("static_loops")
            or not rank.get("dte", {}).get("issues")
            or not rank.get("dte", {}).get("waits")
        ):
            raise RuntimeError(
                f"rank {expected_rank} has no closed overlap witness"
            )
        tokens = sorted(
            token
            for wait in rank["dte"]["waits"]
            for token in wait.get("tokens", [])
        )
        if tokens != list(range(rank["dte"]["token_count"])):
            raise RuntimeError(
                f"rank {expected_rank} DTE issue/wait closure is invalid"
            )
    return attestation


def validate_target_structure(
    baseline: campaign.TargetStructure,
    overlap: campaign.TargetStructure,
) -> None:
    for structure in (baseline, overlap):
        for fragment in (
            "elementwise_add",
            "elementwise_mul",
            "direct_dte_send_issue",
            "direct_dte_recv_prepare",
            "direct_dte_wait",
        ):
            campaign.require_call(structure.counts, fragment, present=True)
    if baseline.scheduler_body_sha256 == overlap.scheduler_body_sha256:
        raise RuntimeError(
            "overlap candidate scheduler is identical to its serialized baseline"
        )


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.completion_timeout_ms < 1:
        raise RuntimeError("repeat and completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("Direct-DTE compute-overlap hardware execution is not armed")
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
        "baseline": args.work_dir / "baseline-package",
        "winner": args.work_dir / "overlap-package",
    }
    compile_variant(
        args.wafer_compile_test,
        source,
        packages["baseline"],
        BASELINE_SELECTION,
    )
    compile_variant(
        args.wafer_compile_test,
        source,
        packages["winner"],
        OVERLAP_SELECTION,
    )
    attestation = validate_overlap_companion(packages["winner"])
    (
        bindings_by_variant,
        output_ids_by_variant,
        completion_evidence_by_variant,
    ) = campaign.validate_paired_packages(
        packages["baseline"],
        packages["winner"],
        CASE,
        target_identity=TARGET_IDENTITY,
    )
    structures = {
        name: campaign.target_structure(package, args.tx8_objdump)
        for name, package in packages.items()
    }
    validate_target_structure(structures["baseline"], structures["winner"])

    identity = {
        "case": "direct-dte-compute-overlap",
        "source_snapshots_sha256": {
            str(relative): hashlib.sha256(
                (packages["winner"] / relative).read_bytes()
            ).hexdigest()
            for relative in campaign.SOURCE_SNAPSHOT_PATHS
        },
        "target_identity": TARGET_IDENTITY,
        "launch": CASE.launch_contract,
        "rank_count": CASE.rank_count,
        "module_digests": {
            name: list(campaign.module_digests(package))
            for name, package in packages.items()
        },
        "accepted_instr_digests": [
            rank["accepted_instr_sha256"] for rank in attestation["ranks"]
        ],
        "overlap_window_counts": [
            rank["direct_dte_compute_overlap_window_count"]
            for rank in attestation["ranks"]
        ],
    }
    structure_record = {
        "schema_version": 1,
        "identity": identity,
        "variants": {
            name: {
                "scheduler_body_sha256": list(
                    structure.scheduler_body_sha256
                ),
                "static_target_callsite_counts": dict(
                    sorted(structure.counts.items())
                ),
                "workspace_bytes": structure.workspace_bytes,
            }
            for name, structure in structures.items()
        },
    }
    (args.work_dir / "target-structure.json").write_text(
        json.dumps(structure_record, indent=2, sort_keys=True) + "\n"
    )
    resource_arguments = campaign.write_payloads(
        args.work_dir, CASE, bindings_by_variant, payloads
    )

    if args.no_card:
        for name, package in packages.items():
            result = campaign.run(
                campaign.no_card_command(args.wafer_run, package, CASE)
            )
            if "board_execution: false" not in result.stdout:
                raise RuntimeError(
                    f"{name} no-card output omitted execution state"
                )
        print(
            "direct_dte_compute_overlap_no_card: "
            "matched_packages=true all_rank_windows=true exact_oracle=true"
        )
        return 0

    rows: list[dict[str, object]] = []
    for sample, name in enumerate(campaign.balanced_order(args.repeat), start=1):
        start_ns = time.monotonic_ns()
        result = campaign.run(
            campaign.board_command(
                args, packages[name], CASE, resource_arguments[name]
            ),
            timeout_seconds=(
                args.completion_timeout_ms / 1000
                + campaign.PROCESS_TIMEOUT_MARGIN_SECONDS
            ),
        )
        elapsed_ns = time.monotonic_ns() - start_ns
        campaign.verify_board_output(
            result.stdout,
            CASE,
            output_ids_by_variant[name],
            completion_evidence_by_variant[name],
        )
        campaign.compare_captured_outputs(
            args.work_dir, CASE, payloads, name
        )
        row = {
            "sample": sample,
            "variant": name,
            "host_process_elapsed_ns": elapsed_ns,
            "torch_reference_matched": True,
        }
        rows.append(row)
        print(
            "direct_dte_compute_overlap_sample: "
            + json.dumps(row, sort_keys=True)
        )
        print(result.stdout, end="")
    (args.work_dir / "board-observations.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "measurement": (
                    "matched host process elapsed observation; not PMU, "
                    "device cycles, or compiler profitability evidence"
                ),
                "samples": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"wafer_board_direct_dte_compute_overlap_test: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
