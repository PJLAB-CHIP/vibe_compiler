#!/usr/bin/env python3
"""Validate and roundtrip Wafer runtime package manifests."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


ALLOWED_COMPLETION_SOURCES = {
    "hpgr_stream_event",
    "hpgr_command_slot",
    "kcore_local_drain",
    "legacy_tsm_run_sync",
}

KNOWN_STUB_FENCES = {
    "TsmDeviceSynchronize",
    "TsmLaunch",
    "TsmLaunchPg",
    "KmdDoorbellOnly",
}


def canonical_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def fail(message: str) -> None:
    raise ValueError(message)


def require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{name} must be an object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{name} must be a list")
    return value


def require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{name} must be a non-empty string")
    return value


def require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        fail(f"{name} must be a positive integer")
    return value


def validate_tensor(tensor: Any, name: str) -> str:
    item = require_dict(tensor, name)
    tensor_name = require_non_empty_string(item.get("name"), f"{name}.name")
    shape = require_list(item.get("shape"), f"{name}.shape")
    if not shape:
        fail(f"{name}.shape must be non-empty")
    for index, dim in enumerate(shape):
        require_positive_int(dim, f"{name}.shape[{index}]")
    require_non_empty_string(item.get("dtype"), f"{name}.dtype")
    if item.get("layout") != "tensor":
        fail(f"{name}.layout must be tensor")
    return tensor_name


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        fail("schema_version must be 1")
    require_non_empty_string(manifest.get("package_name"), "package_name")

    runtime = require_dict(manifest.get("runtime"), "runtime")
    completion_source = require_non_empty_string(
        runtime.get("completion_source"), "runtime.completion_source"
    )
    if completion_source in KNOWN_STUB_FENCES:
        fail("completion source is a known stub fence")
    if completion_source not in ALLOWED_COMPLETION_SOURCES:
        fail("completion source is not in the allowed runtime fence set")

    signature = require_dict(manifest.get("launch_signature"), "launch_signature")
    input_names = {
        validate_tensor(item, f"launch_signature.inputs[{index}]")
        for index, item in enumerate(require_list(signature.get("inputs"), "launch_signature.inputs"))
    }
    output_names = {
        validate_tensor(item, f"launch_signature.outputs[{index}]")
        for index, item in enumerate(require_list(signature.get("outputs"), "launch_signature.outputs"))
    }

    resources = require_dict(manifest.get("resources"), "resources")
    spm_bytes = require_positive_int(resources.get("spm_bytes"), "resources.spm_bytes")
    if spm_bytes > 0x2F0000:
        fail("resources.spm_bytes exceeds usable SPM capacity")

    abi_ops = require_list(manifest.get("abi_ops"), "abi_ops")
    for index, op in enumerate(abi_ops):
        item = require_dict(op, f"abi_ops[{index}]")
        mnemonic = require_non_empty_string(item.get("op"), f"abi_ops[{index}].op")
        if mnemonic not in {
            "wafer.abi.rdma_1d",
            "wafer.abi.wdma_1d",
            "wafer.abi.gemm",
        }:
            fail(f"abi_ops[{index}].op is not supported by the M0 manifest")
        if item.get("wait_policy") != "issue_only":
            fail(f"abi_ops[{index}].wait_policy must be issue_only")

    input_bytes = 0
    output_bytes = 0
    for index, binding in enumerate(
        require_list(manifest.get("ddr_bindings"), "ddr_bindings")
    ):
        item = require_dict(binding, f"ddr_bindings[{index}]")
        kind = require_non_empty_string(item.get("kind"), f"ddr_bindings[{index}].kind")
        name = require_non_empty_string(item.get("name"), f"ddr_bindings[{index}].name")
        bytes_value = require_positive_int(item.get("bytes"), f"ddr_bindings[{index}].bytes")
        require_positive_int(item.get("alignment"), f"ddr_bindings[{index}].alignment")
        if not isinstance(item.get("host_visible"), bool):
            fail(f"ddr_bindings[{index}].host_visible must be boolean")
        if not isinstance(item.get("read_only"), bool):
            fail(f"ddr_bindings[{index}].read_only must be boolean")

        if kind == "input":
            if name not in input_names:
                fail(f"ddr_bindings[{index}] input name is not in launch signature")
            if item["read_only"] is not True:
                fail(f"ddr_bindings[{index}] input must be read-only")
            input_bytes += bytes_value
        elif kind == "output":
            if name not in output_names:
                fail(f"ddr_bindings[{index}] output name is not in launch signature")
            if item["read_only"] is not False:
                fail(f"ddr_bindings[{index}] output must be writable")
            output_bytes += bytes_value
        else:
            fail(f"ddr_bindings[{index}].kind must be input or output")

    if resources.get("ddr_external_input_bytes") != input_bytes:
        fail("resources.ddr_external_input_bytes does not match DDR bindings")
    if resources.get("ddr_external_output_bytes") != output_bytes:
        fail("resources.ddr_external_output_bytes does not match DDR bindings")


def load_manifest(path: str) -> dict[str, Any]:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    return require_dict(json.loads(text), "manifest")


def m0_smoke_manifest(completion_source: str = "hpgr_stream_event") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package_name": "m0_single_tile",
        "runtime": {
            "mode": "hpgr",
            "completion_source": completion_source,
        },
        "device_code": [
            {
                "name": "single_matmul",
                "kind": "c_abi_skeleton",
                "artifact": "m0_single_tile.o",
            }
        ],
        "launch_signature": {
            "inputs": [
                {"name": "lhs", "shape": [4, 8], "dtype": "f16", "layout": "tensor"},
                {"name": "rhs", "shape": [8, 16], "dtype": "f16", "layout": "tensor"},
            ],
            "outputs": [
                {"name": "out", "shape": [4, 16], "dtype": "f16", "layout": "tensor"},
            ],
        },
        "ddr_bindings": [
            {
                "kind": "input",
                "name": "lhs",
                "bytes": 64,
                "alignment": 256,
                "read_only": True,
                "host_visible": True,
            },
            {
                "kind": "input",
                "name": "rhs",
                "bytes": 256,
                "alignment": 256,
                "read_only": True,
                "host_visible": True,
            },
            {
                "kind": "output",
                "name": "out",
                "bytes": 128,
                "alignment": 256,
                "read_only": False,
                "host_visible": True,
            },
        ],
        "resources": {
            "spm_bytes": 2816,
            "ddr_external_input_bytes": 320,
            "ddr_external_output_bytes": 128,
            "workspace_bytes": 0,
            "resident_constant_bytes": 0,
        },
        "abi_ops": [
            {"op": "wafer.abi.rdma_1d", "bytes": 64, "wait_policy": "issue_only"},
            {"op": "wafer.abi.rdma_1d", "bytes": 256, "wait_policy": "issue_only"},
            {"op": "wafer.abi.gemm", "m": 4, "k": 8, "n": 16, "wait_policy": "issue_only"},
            {"op": "wafer.abi.wdma_1d", "bytes": 128, "wait_policy": "issue_only"},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-m0-smoke", action="store_true")
    mode.add_argument("--emit-stub-smoke", action="store_true")
    mode.add_argument("--roundtrip")
    mode.add_argument("--validate")
    args = parser.parse_args()

    if args.emit_m0_smoke:
        sys.stdout.write(canonical_json(m0_smoke_manifest()))
        return 0
    if args.emit_stub_smoke:
        sys.stdout.write(canonical_json(m0_smoke_manifest("TsmDeviceSynchronize")))
        return 0

    manifest = load_manifest(args.roundtrip or args.validate)
    validate_manifest(manifest)
    if args.roundtrip:
        sys.stdout.write(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
