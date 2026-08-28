#!/usr/bin/env python3
"""Strict reusable evidence checks for one 16-tile Direct-DTE board run.

The package manifest and the board process expose different evidence:

* the manifest exposes the typed transport-status argument, status ABI,
  watchdog requirement, and terminal-completion domain;
* a successful ``wafer-run --board`` process currently exposes lifecycle and
  completion rows, but not the raw per-tile status readbacks.

Keep those facts separate.  Callers may provide ``DirectDTEStatusObservation``
records only when an output surface actually exposes the raw status values.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence

import wafer_runtime_launch_contract as runtime_launch


TILE_COUNT = 16
DIRECT_DTE_STATUS_ABI = "wafer-direct-dte-status"
DIRECT_DTE_STATUS_BYTES = 64
DIRECT_DTE_STATUS_ALIGNMENT = 64
DIRECT_DTE_STATUS_SUCCESS = 1
MAXIMUM_COMPLETION_TIMEOUT_MS = 60 * 60 * 1000

STATUS_OBSERVATION_GAP = (
    "wafer-run board stdout and BoardRuntimeInvocationResult do not expose "
    "the 16 per-tile Direct-DTE status values; the "
    "successful runtime path enforces all-tile status Success internally, "
    "but this output cannot independently reconstruct those readbacks"
)


@dataclasses.dataclass(frozen=True)
class DirectDTEManifestEvidence:
    """Typed Direct-DTE package facts indexed by logical tile."""

    status_abi_by_tile: tuple[tuple[int, str], ...]
    completion_by_tile: tuple[tuple[int, str], ...]
    host_watchdog_tiles: tuple[int, ...] = tuple(range(TILE_COUNT))


@dataclasses.dataclass(frozen=True)
class DirectDTEStatusObservation:
    """One raw status row from an output surface that actually exposes it."""

    tile: int
    status_abi: str
    value: int


@dataclasses.dataclass(frozen=True)
class DirectDTEBoardEvidence:
    """Evidence available after validating one successful board invocation."""

    completion_timeout_ms: int
    completion_by_tile: tuple[tuple[int, str], ...]
    runtime_all_tile_success_enforced: bool
    observed_status_by_tile: (
        tuple[DirectDTEStatusObservation, ...] | None
    )
    status_observation_gap: str | None

    @property
    def has_observed_all_tile_success(self) -> bool:
        return self.observed_status_by_tile is not None


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_record_list(
    manifest: Mapping[str, object], field: str
) -> list[Mapping[str, object]]:
    raw = manifest.get(field)
    if not isinstance(raw, list) or not all(
        isinstance(record, Mapping) for record in raw
    ):
        raise RuntimeError(f"Direct-DTE manifest {field} is not a record list")
    return raw


def _entries_by_dense_tile(
    records: Iterable[Mapping[str, object]],
    *,
    context: str,
) -> dict[int, Mapping[str, object]]:
    by_tile: dict[int, Mapping[str, object]] = {}
    for record in records:
        tile = record.get("tile_id")
        if not _is_integer(tile) or not 0 <= tile < TILE_COUNT:
            raise RuntimeError(f"{context} has an invalid logical tile")
        if tile in by_tile:
            raise RuntimeError(f"{context} duplicates logical tile {tile}")
        by_tile[tile] = record
    if set(by_tile) != set(range(TILE_COUNT)):
        raise RuntimeError(f"{context} is not the exact 16-tile domain")
    return by_tile


def _status_arguments(
    entry: Mapping[str, object],
    *,
    context: str,
) -> list[Mapping[str, object]]:
    arguments = entry.get("arguments")
    if not isinstance(arguments, list) or not all(
        isinstance(argument, Mapping) for argument in arguments
    ):
        raise RuntimeError(f"{context} has an invalid argument list")
    return [
        argument
        for argument in arguments
        if argument.get("kind") == "transport_status"
    ]


def validate_direct_dte_manifest(
    manifest: Mapping[str, object],
) -> DirectDTEManifestEvidence:
    """Validate the current status/watchdog/completion package slice."""

    target = manifest.get("target")
    runtime_launch.require_manifest_launch(
        manifest,
        runtime_launch.CLUSTER_KERNEL_LAUNCH,
        context="Direct-DTE",
    )
    if (
        manifest.get("tile_count") != TILE_COUNT
        or not isinstance(target, Mapping)
    ):
        raise RuntimeError(
            "Direct-DTE package does not use the current 16-tile cluster "
            "launch contract"
        )

    entries = _require_record_list(manifest, "entries")
    if len(entries) != TILE_COUNT:
        raise RuntimeError("Direct-DTE entry domain is not exactly 16 records")
    entry_by_tile = _entries_by_dense_tile(
        entries, context="Direct-DTE entry domain"
    )
    entry_ids = {entry.get("id") for entry in entries}
    if (
        entry_ids != set(range(TILE_COUNT))
        or not all(_is_integer(entry_id) for entry_id in entry_ids)
    ):
        raise RuntimeError(
            "Direct-DTE EntryId domain is not dense zero-based"
        )

    completion_by_tile: dict[int, str] = {}
    status_abi_by_tile: dict[int, str] = {}
    for tile, entry in entry_by_tile.items():
        transport = entry.get("transport")
        status_arguments = _status_arguments(
            entry, context=f"tile {tile} Direct-DTE"
        )
        if len(status_arguments) != 1:
            raise RuntimeError(
                f"tile {tile} Direct-DTE status is not bound by exactly "
                "one transport_status argument"
            )
        status = status_arguments[0]
        status_abi = status.get("status_abi")
        if (
            not _is_integer(status.get("ordinal"))
            or status_abi != DIRECT_DTE_STATUS_ABI
            or status.get("bytes") != DIRECT_DTE_STATUS_BYTES
            or status.get("alignment") != DIRECT_DTE_STATUS_ALIGNMENT
            or status.get("access") != "read_write"
        ):
            raise RuntimeError(
                f"tile {tile} Direct-DTE transport_status argument is not "
                "the internal 64/64 read-write status slot"
            )
        if (
            entry.get("card_id") != 0
            or entry.get("launch_slot") != tile
            or not isinstance(transport, Mapping)
            or transport.get("kind") != "direct_dte"
            or transport.get("status_abi") != status_abi
            or transport.get("host_watchdog_required") is not True
        ):
            raise RuntimeError(
                f"tile {tile} Direct-DTE entry/status contract is invalid"
            )

        completion = entry.get("completion")
        if completion != "return_after_local_drain":
            raise RuntimeError(
                f"tile {tile} completion contract is invalid"
            )
        completion_by_tile[tile] = completion
        status_abi_by_tile[tile] = str(status_abi)

    return DirectDTEManifestEvidence(
        status_abi_by_tile=tuple(sorted(status_abi_by_tile.items())),
        completion_by_tile=tuple(sorted(completion_by_tile.items())),
    )


def validate_direct_dte_board_command(command: Sequence[str]) -> int:
    """Validate that a board command carries one bounded completion deadline."""

    if (
        command.count("--board") != 1
        or "--no-card" in command
        or "--all-Tiles" in command
    ):
        raise RuntimeError(
            "Direct-DTE board command must select one board invocation"
        )
    if command.count("--completion-timeout-ms") != 1:
        raise RuntimeError(
            "Direct-DTE board command must carry one completion timeout"
        )
    option_index = command.index("--completion-timeout-ms")
    if option_index + 1 >= len(command):
        raise RuntimeError("Direct-DTE completion timeout value is missing")
    try:
        completion_timeout_ms = int(command[option_index + 1], 10)
    except ValueError as error:
        raise RuntimeError(
            "Direct-DTE completion timeout is not an integer"
        ) from error
    if not 1 <= completion_timeout_ms <= MAXIMUM_COMPLETION_TIMEOUT_MS:
        raise RuntimeError(
            "Direct-DTE completion timeout is outside the runtime range"
        )
    return completion_timeout_ms


def _validate_status_observations(
    observations: Iterable[DirectDTEStatusObservation],
    manifest: DirectDTEManifestEvidence,
) -> tuple[DirectDTEStatusObservation, ...]:
    expected_abi_by_tile = dict(manifest.status_abi_by_tile)
    by_tile: dict[int, DirectDTEStatusObservation] = {}
    for observation in observations:
        if (
            not _is_integer(observation.tile)
            or not _is_integer(observation.value)
        ):
            raise RuntimeError(
                "Direct-DTE status observation has a non-integer "
                "tile/value"
            )
        if observation.tile in by_tile:
            raise RuntimeError(
                f"Direct-DTE status observations duplicate tile "
                f"{observation.tile}"
            )
        if (
            observation.tile not in expected_abi_by_tile
            or observation.status_abi
            != expected_abi_by_tile[observation.tile]
        ):
            raise RuntimeError(
                "Direct-DTE status observation does not match its typed "
                "manifest status ABI"
            )
        if observation.value != DIRECT_DTE_STATUS_SUCCESS:
            raise RuntimeError(
                f"tile {observation.tile} Direct-DTE status is not Success"
            )
        by_tile[observation.tile] = observation
    if set(by_tile) != set(range(TILE_COUNT)):
        raise RuntimeError(
            "Direct-DTE status observations are not the exact 16-tile domain"
        )
    return tuple(by_tile[tile] for tile in range(TILE_COUNT))


def validate_direct_dte_board_output(
    stdout: str,
    manifest: DirectDTEManifestEvidence,
    *,
    completion_timeout_ms: int,
    status_observations: (
        Iterable[DirectDTEStatusObservation] | None
    ) = None,
) -> DirectDTEBoardEvidence:
    """Validate one board success log without inventing missing status rows."""

    if (
        not _is_integer(completion_timeout_ms)
        or not 1 <= completion_timeout_ms <= MAXIMUM_COMPLETION_TIMEOUT_MS
    ):
        raise RuntimeError(
            "Direct-DTE completion timeout is outside the runtime range"
        )

    lines = stdout.splitlines()
    required_once = (
        "board_stage: launch",
        "board_stage: completion",
        "board_stage: device-to-host",
        "board_stage: cleanup",
        "launch_pattern: cluster-x16",
        "physical_tile_domain: 0..15",
        f"invocation_tiles: {TILE_COUNT}",
        "board_execution: true",
    )
    for required in required_once:
        if lines.count(required) != 1:
            raise RuntimeError(
                f"Direct-DTE board output must contain exactly one {required!r}"
            )
    lifecycle = [
        lines.index(f"board_stage: {stage}")
        for stage in ("launch", "completion", "device-to-host", "cleanup")
    ]
    if lifecycle != sorted(lifecycle) or lifecycle[-1] >= lines.index(
        "board_execution: true"
    ):
        raise RuntimeError("Direct-DTE board lifecycle evidence is out of order")

    completion_lines = [
        line for line in lines if line.startswith("completion:")
    ]
    completion_matches = [
        re.fullmatch(
            r"completion: (return_after_local_drain) tile_id=(\d+)",
            line,
        )
        for line in completion_lines
    ]
    if (
        len(completion_lines) != TILE_COUNT
        or any(match is None for match in completion_matches)
    ):
        raise RuntimeError(
            "Direct-DTE board output does not expose exactly 16 typed "
            "completion rows"
        )
    observed_completion_by_tile: dict[int, str] = {}
    for match in completion_matches:
        assert match is not None
        completion, tile_text = match.groups()
        tile = int(tile_text)
        if tile in observed_completion_by_tile:
            raise RuntimeError(
                f"Direct-DTE board output duplicates completion tile {tile}"
            )
        observed_completion_by_tile[tile] = completion
    if tuple(sorted(observed_completion_by_tile.items())) != (
        manifest.completion_by_tile
    ):
        raise RuntimeError(
            "Direct-DTE board completions differ from the manifest"
        )

    observed_status_by_tile = (
        None
        if status_observations is None
        else _validate_status_observations(status_observations, manifest)
    )
    return DirectDTEBoardEvidence(
        completion_timeout_ms=completion_timeout_ms,
        completion_by_tile=manifest.completion_by_tile,
        runtime_all_tile_success_enforced=True,
        observed_status_by_tile=observed_status_by_tile,
        status_observation_gap=(
            STATUS_OBSERVATION_GAP
            if observed_status_by_tile is None
            else None
        ),
    )
