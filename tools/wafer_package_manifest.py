#!/usr/bin/env python3
"""Validate and roundtrip Wafer runtime package manifests."""

from __future__ import annotations

import argparse
import copy
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
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        fail(f"{name} must be a positive integer")
    return value


def require_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        fail(f"{name} must be a non-negative integer")
    return value


def validate_tensor(tensor: Any, name: str) -> tuple[str, list[int]]:
    item = require_dict(tensor, name)
    tensor_name = require_non_empty_string(item.get("name"), f"{name}.name")
    shape = require_list(item.get("shape"), f"{name}.shape")
    if not shape:
        fail(f"{name}.shape must be non-empty")
    checked_shape = []
    for index, dim in enumerate(shape):
        checked_shape.append(require_positive_int(dim, f"{name}.shape[{index}]"))
    require_non_empty_string(item.get("dtype"), f"{name}.dtype")
    if item.get("layout") != "tensor":
        fail(f"{name}.layout must be tensor")
    return tensor_name, checked_shape


def physical_tile_id(coord: list[int], topology: dict[str, int]) -> int:
    card_y, card_x, tile_y, tile_x = coord
    card_index = card_y * topology["card_x_count"] + card_x
    tile_row = card_index * topology["tile_y_count"] + tile_y
    return tile_row * topology["tile_x_count"] + tile_x


def validate_physical_coord(
    coord_value: Any, name: str, topology: dict[str, int]
) -> tuple[list[int], int]:
    coord = require_list(coord_value, name)
    if len(coord) != 4:
        fail(f"{name} must contain card_y, card_x, tile_y, tile_x")

    checked = [
        require_non_negative_int(coord[0], f"{name}[0]"),
        require_non_negative_int(coord[1], f"{name}[1]"),
        require_non_negative_int(coord[2], f"{name}[2]"),
        require_non_negative_int(coord[3], f"{name}[3]"),
    ]
    if (
        checked[0] >= topology["card_y_count"]
        or checked[1] >= topology["card_x_count"]
        or checked[2] >= topology["tile_y_count"]
        or checked[3] >= topology["tile_x_count"]
    ):
        fail(f"{name} is outside target topology")
    return checked, physical_tile_id(checked, topology)


def validate_tile_id_list(
    value: Any, name: str, total_tile_count: int
) -> set[int]:
    tile_ids = set()
    for index, tile_id in enumerate(require_list(value, name)):
        checked = require_non_negative_int(tile_id, f"{name}[{index}]")
        if checked >= total_tile_count:
            fail(f"{name}[{index}] must be within physical topology")
        tile_ids.add(checked)
    return tile_ids


def validate_local_shards(
    value: Any, name: str, tensor_shapes: dict[str, list[int]]
) -> None:
    shards = require_list(value, name)
    if not shards:
        fail(f"{name} must be non-empty")

    for index, shard in enumerate(shards):
        item = require_dict(shard, f"{name}[{index}]")
        tensor_name = require_non_empty_string(item.get("name"), f"{name}[{index}].name")
        if tensor_name not in tensor_shapes:
            fail(f"{name}[{index}].name is not in launch signature")
        tensor_shape = tensor_shapes[tensor_name]

        offsets = require_list(item.get("offsets"), f"{name}[{index}].offsets")
        sizes = require_list(item.get("sizes"), f"{name}[{index}].sizes")
        if len(offsets) != len(tensor_shape) or len(sizes) != len(tensor_shape):
            fail(f"{name}[{index}] rank must match tensor shape")

        for dim, (offset, size, tensor_dim) in enumerate(
            zip(offsets, sizes, tensor_shape)
        ):
            checked_offset = require_non_negative_int(
                offset, f"{name}[{index}].offsets[{dim}]"
            )
            checked_size = require_positive_int(size, f"{name}[{index}].sizes[{dim}]")
            if checked_offset + checked_size > tensor_dim:
                fail(f"{name}[{index}] exceeds tensor shape")


def validate_placement(
    placement_value: Any, tensor_shapes: dict[str, list[int]]
) -> None:
    placement = require_dict(placement_value, "placement")
    logical_rank_count = require_positive_int(
        placement.get("logical_rank_count"), "placement.logical_rank_count"
    )

    topology_value = require_dict(placement.get("topology"), "placement.topology")
    topology = {
        key: require_positive_int(topology_value.get(key), f"placement.topology.{key}")
        for key in (
            "card_y_count",
            "card_x_count",
            "tile_y_count",
            "tile_x_count",
        )
    }
    total_tile_count = (
        topology["card_y_count"]
        * topology["card_x_count"]
        * topology["tile_y_count"]
        * topology["tile_x_count"]
    )

    good_tile_ids = validate_tile_id_list(
        placement.get("good_tile_ids"), "placement.good_tile_ids", total_tile_count
    )
    if not good_tile_ids:
        fail("placement.good_tile_ids must be non-empty")
    bad_tile_ids = validate_tile_id_list(
        placement.get("bad_tile_ids"), "placement.bad_tile_ids", total_tile_count
    )
    if good_tile_ids & bad_tile_ids:
        fail("placement good and bad tile sets must be disjoint")

    ranks = require_list(placement.get("ranks"), "placement.ranks")
    if len(ranks) != logical_rank_count:
        fail("placement ranks must cover every logical rank")

    seen_ranks = set()
    seen_tiles = set()
    seen_blocks = set()
    for index, rank in enumerate(ranks):
        item = require_dict(rank, f"placement.ranks[{index}]")
        logical_rank = require_non_negative_int(
            item.get("logical_rank"), f"placement.ranks[{index}].logical_rank"
        )
        if logical_rank >= logical_rank_count or logical_rank in seen_ranks:
            fail("placement logical ranks must be dense and unique")
        seen_ranks.add(logical_rank)

        block_id = require_non_negative_int(
            item.get("block_id"), f"placement.ranks[{index}].block_id"
        )
        if block_id in seen_blocks:
            fail("placement block_id values must be unique")
        seen_blocks.add(block_id)

        _, tile_id = validate_physical_coord(
            item.get("physical_coord"),
            f"placement.ranks[{index}].physical_coord",
            topology,
        )
        if tile_id not in good_tile_ids:
            fail(
                f"placement.ranks[{index}] maps to tile id {tile_id} "
                "that is not marked good"
            )
        if tile_id in bad_tile_ids:
            fail(f"placement.ranks[{index}] maps to bad tile id {tile_id}")
        if tile_id in seen_tiles:
            fail("placement physical tile ids must be unique")
        seen_tiles.add(tile_id)

        validate_local_shards(
            item.get("local_shards"),
            f"placement.ranks[{index}].local_shards",
            tensor_shapes,
        )

    if seen_ranks != set(range(logical_rank_count)):
        fail("placement ranks must cover every logical rank")


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
    input_tensors = [
        validate_tensor(item, f"launch_signature.inputs[{index}]")
        for index, item in enumerate(
            require_list(signature.get("inputs"), "launch_signature.inputs")
        )
    ]
    output_tensors = [
        validate_tensor(item, f"launch_signature.outputs[{index}]")
        for index, item in enumerate(
            require_list(signature.get("outputs"), "launch_signature.outputs")
        )
    ]
    input_names = {name for name, _ in input_tensors}
    output_names = {name for name, _ in output_tensors}
    tensor_shapes = {name: shape for name, shape in input_tensors + output_tensors}
    validate_placement(manifest.get("placement"), tensor_shapes)

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
        "placement": {
            "logical_rank_count": 1,
            "topology": {
                "card_y_count": 1,
                "card_x_count": 1,
                "tile_y_count": 4,
                "tile_x_count": 4,
            },
            "good_tile_ids": [0],
            "bad_tile_ids": [],
            "ranks": [
                {
                    "logical_rank": 0,
                    "physical_coord": [0, 0, 0, 0],
                    "block_id": 0,
                    "local_shards": [
                        {"name": "lhs", "offsets": [0, 0], "sizes": [4, 8]},
                        {"name": "rhs", "offsets": [0, 0], "sizes": [8, 16]},
                        {"name": "out", "offsets": [0, 0], "sizes": [4, 16]},
                    ],
                }
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


def bad_placement_smoke_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(m0_smoke_manifest())
    manifest["placement"]["ranks"][0]["physical_coord"] = [0, 0, 0, 1]
    return manifest


def bad_local_shard_smoke_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(m0_smoke_manifest())
    manifest["placement"]["ranks"][0]["local_shards"][0]["sizes"] = [5, 8]
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-m0-smoke", action="store_true")
    mode.add_argument("--emit-stub-smoke", action="store_true")
    mode.add_argument("--emit-bad-placement-smoke", action="store_true")
    mode.add_argument("--emit-bad-local-shard-smoke", action="store_true")
    mode.add_argument("--roundtrip")
    mode.add_argument("--validate")
    args = parser.parse_args()

    if args.emit_m0_smoke:
        sys.stdout.write(canonical_json(m0_smoke_manifest()))
        return 0
    if args.emit_stub_smoke:
        sys.stdout.write(canonical_json(m0_smoke_manifest("TsmDeviceSynchronize")))
        return 0
    if args.emit_bad_placement_smoke:
        sys.stdout.write(canonical_json(bad_placement_smoke_manifest()))
        return 0
    if args.emit_bad_local_shard_smoke:
        sys.stdout.write(canonical_json(bad_local_shard_smoke_manifest()))
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
