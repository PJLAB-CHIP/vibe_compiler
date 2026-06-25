#!/usr/bin/env python3
"""Construct and validate Wafer runtime launch plans from package manifests."""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from wafer_package_manifest import validate_manifest  # noqa: E402


TX_RUNTIME_LIBRARY_NAMES = (
    "libhpgr.so",
    "libtx_runtime.so",
)

TX_REQUIRED_SYMBOLS = (
    "txSetDevice",
    "txMalloc",
    "txFree",
    "txMemcpy",
    "txModuleLoad",
    "txModuleGetFunction",
    "txLaunchKernel",
    "txStreamSynchronize",
)


@dataclass(frozen=True)
class LaunchPlan:
    manifest: dict[str, Any]
    package_name: str
    runtime_mode: str
    completion_source: str
    program_id: str
    entrypoint: str
    device_codes: list[dict[str, Any]]
    ddr_bindings: list[dict[str, Any]]
    workspace_buffers: list[dict[str, Any]]
    resident_constants: list[dict[str, Any]]

    @property
    def launch_arg_bytes(self) -> int:
        total = 0
        for item in (
            self.ddr_bindings + self.workspace_buffers + self.resident_constants
        ):
            total += int(item.get("bytes", 0))
        return total


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        fail("manifest must be a JSON object")
    validate_manifest(manifest)
    return manifest


def build_launch_plan(manifest: dict[str, Any]) -> LaunchPlan:
    runtime = manifest["runtime"]
    program = manifest["program"]
    return LaunchPlan(
        manifest=manifest,
        package_name=manifest["package_name"],
        runtime_mode=runtime["mode"],
        completion_source=runtime["completion_source"],
        program_id=program["id"],
        entrypoint=program["entrypoint"],
        device_codes=list(manifest["device_code"]),
        ddr_bindings=list(manifest["ddr_bindings"]),
        workspace_buffers=list(manifest["workspace_buffers"]),
        resident_constants=list(manifest["resident_constants"]),
    )


def format_bool_flag(value: Any, true_text: str) -> str:
    return true_text if bool(value) else "device_only"


def emit_plan_summary(plan: LaunchPlan, backend_name: str) -> list[str]:
    lines = [
        f"backend: {backend_name}",
        f"package: {plan.package_name}",
        f"runtime_mode: {plan.runtime_mode}",
        f"completion_source: {plan.completion_source}",
    ]
    for device_code in plan.device_codes:
        lines.append(
            "device_code: "
            f"{device_code['name']} {device_code['kind']} {device_code['artifact']}"
        )
    for binding in plan.ddr_bindings:
        lines.append(
            "allocation: "
            f"{binding['kind']} {binding['name']} {binding['bytes']} bytes "
            f"{format_bool_flag(binding.get('host_visible'), 'host_visible')}"
        )
    for workspace in plan.workspace_buffers:
        lines.append(
            f"allocation: workspace {workspace['name']} {workspace['bytes']} bytes"
        )
    for constant in plan.resident_constants:
        lines.append(
            f"allocation: resident_constant {constant['name']} "
            f"{constant['bytes']} bytes source={constant['source']}"
        )
    lines.append(
        "launch: module_kernel "
        f"entrypoint={plan.entrypoint} grid=(1,1,1) block=(1,1,1)"
    )
    lines.append(f"completion: wait {plan.completion_source}")
    return lines


def run_dry_run(plan: LaunchPlan) -> list[str]:
    return emit_plan_summary(plan, "dry-run")


def first_device_code(plan: LaunchPlan) -> dict[str, Any]:
    if not plan.device_codes:
        fail("launch plan contains no device code")
    return plan.device_codes[0]


def run_fake_tx(plan: LaunchPlan, device_id: int) -> list[str]:
    code = first_device_code(plan)
    lines = ["backend: fake-tx", f"txSetDevice device={device_id}"]
    for binding in plan.ddr_bindings:
        lines.append(f"txMalloc name={binding['name']} bytes={binding['bytes']}")
        if binding["kind"] == "input":
            lines.append(f"txMemcpyH2D name={binding['name']} bytes={binding['bytes']}")
    for workspace in plan.workspace_buffers:
        lines.append(f"txMalloc name={workspace['name']} bytes={workspace['bytes']}")
    for constant in plan.resident_constants:
        if constant["source"] == "embedded_constant":
            lines.append(f"txMalloc name={constant['name']} bytes={constant['bytes']}")
            lines.append(
                f"txMemcpyH2D name={constant['name']} bytes={constant['bytes']}"
            )
    lines.extend(
        [
            f"txModuleLoad artifact={code['artifact']}",
            f"txModuleGetFunction entrypoint={plan.entrypoint}",
            f"txLaunchKernel entrypoint={plan.entrypoint} "
            f"arg_bytes={plan.launch_arg_bytes}",
            f"txStreamSynchronize completion_source={plan.completion_source}",
        ]
    )
    for binding in plan.ddr_bindings:
        if binding["kind"] == "output":
            lines.append(f"txMemcpyD2H name={binding['name']} bytes={binding['bytes']}")
    return lines


def runtime_library_candidates(
    runtime_root: pathlib.Path | None, runtime_library: pathlib.Path | None
) -> list[pathlib.Path]:
    if runtime_library is not None:
        return [runtime_library]

    roots: list[pathlib.Path] = []
    if runtime_root is not None:
        roots.append(runtime_root)
    else:
        repo_root = SCRIPT_DIR.parent
        roots.extend(
            [
                repo_root / "third_party" / "tx_runtime",
                pathlib.Path("/usr"),
                pathlib.Path("/usr/local"),
            ]
        )

    candidates: list[pathlib.Path] = []
    for root in roots:
        for lib_dir in ("lib", "lib64", "lib/x86_64-linux-gnu"):
            for library_name in TX_RUNTIME_LIBRARY_NAMES:
                candidates.append(root / lib_dir / library_name)
        for library_name in TX_RUNTIME_LIBRARY_NAMES:
            candidates.append(root / library_name)
    return candidates


def find_tx_runtime_library(
    runtime_root: pathlib.Path | None, runtime_library: pathlib.Path | None
) -> pathlib.Path:
    candidates = runtime_library_candidates(runtime_root, runtime_library)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(item) for item in candidates)
    fail(f"tx runtime library was not found; searched: {searched}")


def run_tx_discovery(
    plan: LaunchPlan,
    runtime_root: pathlib.Path | None,
    runtime_library: pathlib.Path | None,
) -> list[str]:
    library_path = find_tx_runtime_library(runtime_root, runtime_library)
    library = ctypes.CDLL(str(library_path))
    missing = [symbol for symbol in TX_REQUIRED_SYMBOLS if not hasattr(library, symbol)]
    if missing:
        fail(
            "tx runtime library is missing required symbol(s): " + ", ".join(missing)
        )
    lines = emit_plan_summary(plan, "tx")
    lines.insert(1, f"tx_runtime_library: {library_path}")
    lines.append("tx_runtime_symbols: ok")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--backend", choices=("dry-run", "fake-tx", "tx"), required=True
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--runtime-root", type=pathlib.Path)
    parser.add_argument("--runtime-library", type=pathlib.Path)
    args = parser.parse_args()

    manifest = load_manifest(pathlib.Path(args.manifest))
    plan = build_launch_plan(manifest)
    if args.backend == "dry-run":
        lines = run_dry_run(plan)
    elif args.backend == "fake-tx":
        lines = run_fake_tx(plan, args.device_id)
    else:
        lines = run_tx_discovery(plan, args.runtime_root, args.runtime_library)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
