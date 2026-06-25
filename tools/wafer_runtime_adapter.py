#!/usr/bin/env python3
"""Construct and validate Wafer runtime launch plans from package metadata."""

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

from wafer_package_metadata import validate_package_metadata  # noqa: E402


TX_RUNTIME_LIBRARY_NAMES = (
    "libhpgr.so",
    "libtx_runtime.so",
)

TX_BASE_REQUIRED_SYMBOLS = (
    "txSetDevice",
    "txMalloc",
    "txFree",
    "txMemcpy",
    "txStreamSynchronize",
)

TX_ENTRYPOINT_REQUIRED_SYMBOLS = {
    "tx.module": (
        "txModuleLoad",
        "txModuleGetFunction",
        "txLaunchKernel",
    ),
    "tx.cluster": (
        "txModuleLoad",
        "txModuleGetFunction",
        "txLaunchClusterKernel",
    ),
    "tx.model": (
        "txLaunchModel",
        "txLaunchModelSync",
    ),
    "tx.graph": (
        "txLoadGraph",
        "txUnloadGraph",
    ),
}


@dataclass(frozen=True)
class RuntimePlan:
    metadata: dict[str, Any]
    package: str
    runtime_mode: str
    completion_source: str
    model: dict[str, Any]
    modules: dict[str, dict[str, Any]]
    entrypoints: dict[str, dict[str, Any]]
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    parameters: list[dict[str, Any]]
    workspace: list[dict[str, Any]]
    resident_constants: list[dict[str, Any]]
    binding_bytes: dict[str, int]


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_package_metadata(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        fail("metadata must be a JSON object")
    validate_package_metadata(metadata)
    return metadata


def bytes_from_binding(item: dict[str, Any]) -> int:
    binding = item.get("binding")
    if isinstance(binding, dict):
        return int(binding["bytes"])
    return int(item["bytes"])


def build_runtime_plan(metadata: dict[str, Any]) -> RuntimePlan:
    runtime = metadata["runtime"]
    model = metadata["model"]
    interface = model["interface"]
    modules = {item["name"]: item for item in metadata["modules"]}
    entrypoints = {item["name"]: item for item in metadata["entrypoints"]}
    inputs = list(interface["inputs"])
    outputs = list(interface["outputs"])
    parameters = list(interface["parameters"])
    workspace = list(interface["workspace"])
    resident_constants = list(interface["resident_constants"])
    binding_bytes: dict[str, int] = {}
    for item in inputs + outputs + parameters + workspace + resident_constants:
        binding_bytes[item["name"]] = bytes_from_binding(item)
    return RuntimePlan(
        metadata=metadata,
        package=metadata["name"],
        runtime_mode=runtime["mode"],
        completion_source=runtime["completion_source"],
        model=model,
        modules=modules,
        entrypoints=entrypoints,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        workspace=workspace,
        resident_constants=resident_constants,
        binding_bytes=binding_bytes,
    )


def format_bool_flag(value: Any, true_text: str) -> str:
    return true_text if bool(value) else "device_only"


def resident_source_label(item: dict[str, Any]) -> str:
    source = item["source"]
    if source in {"launch_input", "parameter"}:
        return f"{source}:{item['source_binding']}"
    return source


def binding_summary_lines(plan: RuntimePlan) -> list[str]:
    lines: list[str] = []
    for item in plan.inputs:
        binding = item["binding"]
        lines.append(
            "binding: input "
            f"{item['name']} {binding['bytes']} bytes "
            f"{format_bool_flag(binding.get('host_visible'), 'host_visible')}"
        )
    for item in plan.outputs:
        binding = item["binding"]
        lines.append(
            "binding: output "
            f"{item['name']} {binding['bytes']} bytes "
            f"{format_bool_flag(binding.get('host_visible'), 'host_visible')}"
        )
    for item in plan.parameters:
        binding = item["binding"]
        lines.append(
            "binding: parameter "
            f"{item['name']} {binding['bytes']} bytes "
            f"{format_bool_flag(binding.get('host_visible'), 'host_visible')}"
        )
    for item in plan.workspace:
        lines.append(f"binding: workspace {item['name']} {item['bytes']} bytes")
    for item in plan.resident_constants:
        lines.append(
            "binding: resident_constant "
            f"{item['name']} {item['bytes']} bytes source={resident_source_label(item)}"
        )
    return lines


def entrypoint_summary(entrypoint: dict[str, Any]) -> str:
    executor = entrypoint["executor"]
    name = entrypoint["name"]
    if executor == "tx.model":
        state = entrypoint["bpm_descriptor"]["state"]
        return f"entrypoint: {name} {executor} bpm={state}"
    if executor in {"tx.module", "tx.cluster"}:
        return (
            f"entrypoint: {name} {executor} module={entrypoint['module']} "
            f"function={entrypoint['function']} debug"
        )
    if executor == "tx.graph":
        return (
            f"entrypoint: {name} {executor} module={entrypoint['module']} "
            f"mod_symbol={entrypoint['mod_symbol']}"
        )
    return f"entrypoint: {name} {executor}"


def emit_plan_summary(plan: RuntimePlan, backend_name: str) -> list[str]:
    lines = [
        f"backend: {backend_name}",
        f"package: {plan.package}",
        f"runtime_mode: {plan.runtime_mode}",
        f"completion_source: {plan.completion_source}",
        f"model: {plan.model['id']} abi={plan.model['abi']}",
    ]
    lines.extend(binding_summary_lines(plan))
    for module in plan.modules.values():
        lines.append(
            f"module: {module['name']} {module['format']} {module['path']}"
        )
    for entrypoint in plan.entrypoints.values():
        lines.append(entrypoint_summary(entrypoint))
    lines.append(f"completion: wait {plan.completion_source}")
    return lines


def run_dry_run(plan: RuntimePlan) -> list[str]:
    return emit_plan_summary(plan, "dry-run")


def select_entrypoint(plan: RuntimePlan, entrypoint_name: str | None) -> dict[str, Any]:
    if entrypoint_name is None:
        fail("--entrypoint is required for this backend")
    if entrypoint_name not in plan.entrypoints:
        fail(f"entrypoint was not found in package metadata: {entrypoint_name}")
    return plan.entrypoints[entrypoint_name]


def binding_order_arg_bytes(plan: RuntimePlan, entrypoint: dict[str, Any]) -> int:
    return sum(plan.binding_bytes[name] for name in entrypoint["binding_order"])


def module_for_entrypoint(plan: RuntimePlan, entrypoint: dict[str, Any]) -> dict[str, Any]:
    module_name = entrypoint["module"]
    if module_name not in plan.modules:
        fail(f"entrypoint module was not found: {module_name}")
    return plan.modules[module_name]


def emit_common_fake_tx_allocations(plan: RuntimePlan) -> list[str]:
    lines: list[str] = []
    for item in plan.inputs:
        binding = item["binding"]
        lines.append(f"txMalloc name={item['name']} bytes={binding['bytes']}")
        lines.append(f"txMemcpyH2D name={item['name']} bytes={binding['bytes']}")
    for item in plan.parameters:
        binding = item["binding"]
        lines.append(f"txMalloc name={item['name']} bytes={binding['bytes']}")
        lines.append(f"txMemcpyH2D name={item['name']} bytes={binding['bytes']}")
    for item in plan.outputs:
        binding = item["binding"]
        lines.append(f"txMalloc name={item['name']} bytes={binding['bytes']}")
    for item in plan.workspace:
        lines.append(f"txMalloc name={item['name']} bytes={item['bytes']}")
    for item in plan.resident_constants:
        lines.append(f"txMalloc name={item['name']} bytes={item['bytes']}")
        source = item["source"]
        if source == "embedded_constant":
            lines.append(f"txMemcpyH2D name={item['name']} bytes={item['bytes']}")
        else:
            lines.append(
                f"txMemcpyH2D name={item['name']} bytes={item['bytes']} "
                f"source={item['source_binding']}"
            )
    return lines


def emit_fake_tx_writebacks(plan: RuntimePlan) -> list[str]:
    lines: list[str] = []
    for item in plan.outputs:
        binding = item["binding"]
        lines.append(f"txMemcpyD2H name={item['name']} bytes={binding['bytes']}")
    return lines


def run_fake_tx(
    plan: RuntimePlan, device_id: int, entrypoint_name: str | None
) -> list[str]:
    entrypoint = select_entrypoint(plan, entrypoint_name)
    executor = entrypoint["executor"]
    lines = [
        "backend: fake-tx",
        f"entrypoint: {entrypoint['name']} {executor}",
        f"txSetDevice device={device_id}",
    ]
    if executor == "tx.model":
        state = entrypoint["bpm_descriptor"]["state"]
        if state != "materialized":
            fail(
                f"tx.model entrypoint {entrypoint['name']} requires a materialized BPM descriptor"
            )
        lines.append("txLaunchModel bpm_descriptor=materialized")
        lines.append(f"txStreamSynchronize completion_source={plan.completion_source}")
        return lines

    if executor == "tx.graph":
        module = plan.modules[entrypoint["module"]]
        lines.append(
            f"txLoadGraph path={module['path']} mod_symbol={entrypoint['mod_symbol']}"
        )
        lines.append(f"txStreamSynchronize completion_source={plan.completion_source}")
        return lines

    if executor not in {"tx.module", "tx.cluster"}:
        fail(f"fake-tx backend does not support entrypoint executor: {executor}")

    module = module_for_entrypoint(plan, entrypoint)
    lines.extend(emit_common_fake_tx_allocations(plan))
    lines.append(f"txModuleLoad module={module['path']}")
    lines.append(f"txModuleGetFunction function={entrypoint['function']}")
    launch_api = (
        "txLaunchClusterKernel" if executor == "tx.cluster" else "txLaunchKernel"
    )
    lines.append(
        f"{launch_api} entrypoint={entrypoint['name']} "
        f"function={entrypoint['function']} "
        f"arg_bytes={binding_order_arg_bytes(plan, entrypoint)}"
    )
    lines.append(f"txStreamSynchronize completion_source={plan.completion_source}")
    lines.extend(emit_fake_tx_writebacks(plan))
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


def required_symbols_for_entrypoint(entrypoint: dict[str, Any] | None) -> tuple[str, ...]:
    if entrypoint is None:
        return TX_BASE_REQUIRED_SYMBOLS
    return TX_BASE_REQUIRED_SYMBOLS + TX_ENTRYPOINT_REQUIRED_SYMBOLS.get(
        entrypoint["executor"], ()
    )


def run_tx_discovery(
    plan: RuntimePlan,
    entrypoint_name: str | None,
    runtime_root: pathlib.Path | None,
    runtime_library: pathlib.Path | None,
) -> list[str]:
    entrypoint = select_entrypoint(plan, entrypoint_name) if entrypoint_name else None
    library_path = find_tx_runtime_library(runtime_root, runtime_library)
    library = ctypes.CDLL(str(library_path))
    missing = [
        symbol
        for symbol in required_symbols_for_entrypoint(entrypoint)
        if not hasattr(library, symbol)
    ]
    if missing:
        fail(
            "tx runtime library is missing required symbol(s): " + ", ".join(missing)
        )
    lines = emit_plan_summary(plan, "tx")
    lines.insert(1, f"tx_runtime_library: {library_path}")
    if entrypoint is not None:
        lines.insert(
            2, f"selected_entrypoint: {entrypoint['name']} {entrypoint['executor']}"
        )
    lines.append("tx_runtime_symbols: ok")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-metadata", required=True)
    parser.add_argument(
        "--backend", choices=("dry-run", "fake-tx", "tx"), required=True
    )
    parser.add_argument("--entrypoint")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--runtime-root", type=pathlib.Path)
    parser.add_argument("--runtime-library", type=pathlib.Path)
    args = parser.parse_args()

    metadata = load_package_metadata(pathlib.Path(args.package_metadata))
    plan = build_runtime_plan(metadata)
    if args.backend == "dry-run":
        lines = run_dry_run(plan)
    elif args.backend == "fake-tx":
        lines = run_fake_tx(plan, args.device_id, args.entrypoint)
    else:
        lines = run_tx_discovery(
            plan, args.entrypoint, args.runtime_root, args.runtime_library
        )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
