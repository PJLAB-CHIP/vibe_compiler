#!/usr/bin/env python3
"""Compile and profile the current global-source M-tiled f16 GEMM."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys

import torch

import wafer_board_complete_tile_add_test as profile_support
import wafer_board_compiler_optimization_comparison_test as source_contract
import wafer_runtime_launch_contract as runtime_launch


PYTORCH_BOARD_DIR = pathlib.Path(__file__).resolve().parent / "PyTorch"
if str(PYTORCH_BOARD_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_BOARD_DIR))

import wafer_pytorch_board_common as torch_reference  # noqa: E402


CASE_KEY = "noc-resident-m-tiled-gemm"
CASE = source_contract.CASES[CASE_KEY]
TILE_COUNT = source_contract.TILE_COUNT
TARGET_IDENTITY = "wafer-tx81-single-card"
STATUS_ABI = "wafer-direct-dte-status"
PROCESS_TIMEOUT_MARGIN_SECONDS = 60


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
    parser.add_argument("--completion-timeout-ms", type=int, default=600000)
    return parser.parse_args()


def run(
    command: list[str], timeout_seconds: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        for partial in (error.stdout, error.stderr):
            if partial:
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                print(partial, end="", file=sys.stderr)
        raise RuntimeError(
            "one M-tiled GEMM profile collection exceeded its outer deadline; "
            "the process was killed without retry, reset, or power operation"
        ) from error
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}"
        )
    return result


def write_source_program(work_dir: pathlib.Path) -> pathlib.Path:
    known_children = {
        "source-program",
        "ordinary-package",
        "package",
        "raw",
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
    source_contract.validate_source(CASE)
    return source_contract.write_source(work_dir, CASE)


def compile_package(
    compiler: pathlib.Path,
    source: pathlib.Path,
    package: pathlib.Path,
    *,
    profile: bool,
) -> None:
    command = [
        str(compiler),
        "--input-program-dir",
        str(source),
        "--output-package-dir",
        str(package),
        "--num-partitions=1",
            ]
    if profile:
        command.append("--profile")
    result = run(command)
    if "wrote verified package with num-partitions=1 tiles=16" not in result.stdout:
        raise RuntimeError("compiler did not write a current complete-Tile package")
    instrumentation = "wafer-compile: wrote profile instrumentation:"
    if profile and instrumentation not in result.stdout:
        raise RuntimeError("profile compilation omitted instrumentation")
    if not profile and instrumentation in result.stdout:
        raise RuntimeError("ordinary compilation wrote profile instrumentation")


def validate_manifest(
    package: pathlib.Path,
) -> tuple[dict[tuple[str, int], int], int]:
    manifest = json.loads((package / "manifest.json").read_text())
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="M-tiled GEMM",
    )
    if (
        manifest.get("card_count") != 1
        or manifest.get("tile_count") != TILE_COUNT
        or manifest.get("target", {}).get("identity") != TARGET_IDENTITY
    ):
        raise RuntimeError("M-tiled GEMM package target fields are invalid")

    modules = manifest.get("modules")
    expected_exports = runtime_launch.expected_kernel_module_exports(
        runtime_launch.CLUSTER_KERNEL_LAUNCH
    )
    if (
        not isinstance(modules, list)
        or len(modules) != 1
        or modules[0].get("exports") != expected_exports
        or not (package / modules[0].get("path", "")).is_file()
    ):
        raise RuntimeError("M-tiled GEMM must produce one cluster shared ELF")

    entries = runtime_launch.require_complete_tile_domain(
        manifest, context="M-tiled GEMM"
    )
    module_id = modules[0].get("id")
    if any(
        entry.get("module") != module_id
        or entry.get("transport", {}).get("kind") != "direct_dte"
        or entry["transport"].get("status_abi") != STATUS_ABI
        or entry["transport"].get("host_watchdog_required") is not True
        for entry in entries
    ):
        raise RuntimeError("M-tiled GEMM Direct-DTE contract is invalid")

    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise RuntimeError("M-tiled GEMM port tables are not lists")
    expected_resources = {
        ("user_input", index): (
            {"dtype": spec.mlir_dtype, "shape": list(spec.shape)},
            math.prod(spec.shape) * spec.element_bytes,
        )
        for index, spec in enumerate(CASE.inputs)
    }
    expected_resources.update(
        {
            ("output", index): (
                {"dtype": spec.mlir_dtype, "shape": list(spec.shape)},
                math.prod(spec.shape) * spec.element_bytes,
            )
            for index, spec in enumerate(CASE.outputs)
        }
    )
    bindings: dict[tuple[str, int], int] = {}
    for table, role in (("inputs", "user_input"), ("outputs", "output")):
        for record in manifest.get(table, []):
            if not isinstance(record, dict):
                raise RuntimeError("M-tiled GEMM port records must be objects")
            role_index = record.get("role_index")
            key = (role, role_index)
            if key not in expected_resources or key in bindings:
                raise RuntimeError("M-tiled GEMM has an unexpected host port")
            type_, bytes_ = expected_resources[key]
            if (
                record.get("dtype") != type_["dtype"]
                or record.get("shape") != type_["shape"]
                or record.get("bytes") != bytes_
                or not isinstance(record.get("id"), int)
            ):
                raise RuntimeError(f"M-tiled GEMM port {key} is invalid")
            bindings[key] = record["id"]
    if set(bindings) != set(expected_resources):
        raise RuntimeError("M-tiled GEMM host bindings are incomplete")
    for entry in entries:
        argument_ports: dict[int, tuple[str, str]] = {}
        for argument in entry.get("arguments", []):
            if not isinstance(argument, dict):
                raise RuntimeError("M-tiled GEMM entry arguments must be objects")
            kind = argument.get("kind")
            port = argument.get("port")
            access = argument.get("access")
            if kind not in ("external_input", "external_output"):
                continue
            if not isinstance(port, int) or not isinstance(access, str):
                raise RuntimeError("M-tiled GEMM port argument is invalid")
            argument_ports[port] = (kind, access)
        for (role, role_index), port in bindings.items():
            expected_argument = (
                ("external_input", "read_only")
                if role == "user_input"
                else ("external_output", "write_only")
            )
            if argument_ports.get(port) != expected_argument:
                raise RuntimeError(
                    f"M-tiled GEMM entry argument for {role} {role_index} is invalid"
                )
    return bindings, bindings[("output", 0)]


def require_gemm_call(package: pathlib.Path, objdump: pathlib.Path) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    disassembly = "\n".join(
        run([str(objdump), "-d", str(package / module["path"])]).stdout
        for module in manifest["modules"]
    )
    if re.search(r"<wafer_tx81_gemm(?:_oriented)?(?:\+[^>]*)?>", disassembly) is None:
        raise RuntimeError("M-tiled GEMM target module omitted the GEMM call")


def write_payloads(
    work_dir: pathlib.Path,
    bindings: dict[tuple[str, int], int],
) -> tuple[list[str], pathlib.Path, torch.Tensor]:
    values = CASE.payload_factory()
    source_contract.validate_payloads(CASE, values)
    raw = work_dir / "raw"
    raw.mkdir()
    arguments: list[str] = []
    for index, tensor in enumerate(values.inputs):
        path = raw / f"user_input_{index}.f16.raw"
        torch_reference.write_tensor_raw(path, tensor)
        arguments.extend(
            ["--resource", f"{bindings[('user_input', index)]}={path}"]
        )
    output = raw / "output.capture.f16.raw"
    arguments.extend(["--output", f"{bindings[('output', 0)]}={output}"])
    return arguments, output, values.expected_outputs[0]


def verify_no_card(stdout: str, *, profile: bool) -> None:
    required = {
        "package: id=0 cards=1 tiles=16",
        "invocation_tiles: 16",
        "board_execution: false",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("M-tiled GEMM no-card output is incomplete")
    instrumentation = profile_support.PROFILE_INSTRUMENTATION_READY in stdout
    if instrumentation != profile:
        raise RuntimeError("profile instrumentation activation is not exact")


def verify_board(stdout: str, output_id: int) -> None:
    required = {
        "board_stage: validation",
        "board_stage: device-selection",
        "board_stage: resource-allocation",
        "board_stage: host-to-device",
        "board_stage: module-load",
        "board_stage: entry-resolve",
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "invocation_tiles: 16",
        "launch_pattern: cluster-x16",
        "physical_tile_domain: 0..15",
        "board_execution: true",
    }
    if not required.issubset(set(stdout.splitlines())):
        raise RuntimeError("M-tiled GEMM board output is incomplete")
    captures = re.findall(
        r"^output_capture: port=(\d+) bytes=\d+ path=.+$",
        stdout,
        re.MULTILINE,
    )
    if captures != [str(output_id)]:
        raise RuntimeError("M-tiled GEMM output capture is not exact")
    tile_ids = sorted(
        int(tile)
        for tile in re.findall(
            r"^board_tile: tile_id=(\d+) launch_slot=\d+ available=true "
            r"physical_x=\d+ physical_y=\d+$",
            stdout,
            re.MULTILINE,
        )
    )
    if tile_ids != list(range(TILE_COUNT)):
        raise RuntimeError("M-tiled GEMM did not prove Tiles 0..15")
    runtime_launch.require_board_completion(stdout, context="M-tiled GEMM")


def main() -> int:
    args = parse_args()
    if args.completion_timeout_ms < 1:
        raise RuntimeError("completion timeout must be positive")
    if not args.no_card and os.environ.get("WAFER_EXECUTE_HARDWARE_TESTS") != "1":
        print("M-tiled GEMM profile hardware execution is not armed")
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
            raise RuntimeError("board execution requires complete qualification")
        if args.expected_tile_count != TILE_COUNT:
            raise RuntimeError(f"--expected-tile-count must be {TILE_COUNT}")

    source = write_source_program(args.work_dir)
    ordinary = args.work_dir / "ordinary-package"
    package = args.work_dir / "package"
    compile_package(args.wafer_compile, source, ordinary, profile=False)
    compile_package(args.wafer_compile, source, package, profile=True)
    # The profiled output directory is the common delivery root; the
    # package root inside it keeps the <root>.profile sibling rule.
    package = package / "package"
    profile_support.require_byte_identical_packages(ordinary, package)
    profile_support.require_profile_instrumentation_permissions(package)
    bindings, output_id = validate_manifest(package)
    require_gemm_call(package, args.tx8_objdump)

    if args.no_card:
        for candidate, profiled in ((ordinary, False), (package, True)):
            result = run(
                [
                    str(args.wafer_run),
                    "--package-dir",
                    str(candidate),
                    "--no-card",
                    "--direct-dte-status-abi",
                    STATUS_ABI,
                    "--supports-host-watchdog",
                ]
            )
            verify_no_card(result.stdout, profile=profiled)
        print("m_tiled_gemm_profile_no_card: verified")
        return 0

    resource_arguments, capture, expected = write_payloads(
        args.work_dir, bindings
    )
    result = run(
        [
            str(args.wafer_run),
            "--package-dir",
            str(package),
            "--board",
            "--device-id",
            str(args.device_id),
            "--expected-runtime-version",
            str(args.expected_runtime_version),
            "--expected-device-name",
            args.expected_device_name,
            "--expected-pci-bus-id",
            args.expected_pci_bus_id,
            "--expected-tile-count",
            str(args.expected_tile_count),
            "--expected-runtime-library-sha256",
            args.expected_runtime_library_sha256,
            "--completion-timeout-ms",
            str(args.completion_timeout_ms),
            *resource_arguments,
        ],
        timeout_seconds=max(
            300.0,
            profile_support.PROFILE_MEASUREMENT_COUNT
            * args.completion_timeout_ms
            / 1000.0
            + PROCESS_TIMEOUT_MARGIN_SECONDS,
        ),
    )
    verify_board(result.stdout, output_id)
    torch_reference.assert_raw_capture_matches(
        capture,
        expected,
        policy=torch_reference.EXACT,
        context="M-tiled GEMM profile output",
    )
    duration, report = profile_support.verify_profile_report(
        package, result.stdout, expected_active_engine="NE"
    )
    print(
        "m_tiled_gemm_profile_board: pass "
        f"device_duration_ns={duration} report={report}"
    )
    print(result.stdout, end="")
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
        print(f"wafer_board_m_tiled_gemm_profile_test: {error}", file=sys.stderr)
        raise SystemExit(1)
