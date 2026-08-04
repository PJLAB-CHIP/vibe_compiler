#!/usr/bin/env python3
"""Strict reusable evidence checks for one 16-rank Direct-DTE board run.

The package manifest and the board process expose different evidence:

* the manifest exposes the typed status resource, status ABI, watchdog
  requirement, and terminal-completion domain;
* a successful ``wafer-run --board`` process currently exposes lifecycle and
  terminal-completion rows, but not the raw per-rank status readbacks.

Keep those facts separate.  Callers may provide ``DirectDTEStatusObservation``
records only when an output surface actually exposes the raw status values.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence

import wafer_runtime_launch_contract as runtime_launch


SCHEMA_VERSION = runtime_launch.PACKAGE_SCHEMA_VERSION
RANK_COUNT = 16
CLUSTER_LAUNCH_KIND = runtime_launch.KERNEL_LAUNCH_KIND
DIRECT_DTE_STATUS_ABI = "wafer-direct-dte-status-v2"
DIRECT_DTE_STATUS_BYTES = 64
DIRECT_DTE_STATUS_ALIGNMENT = 64
DIRECT_DTE_STATUS_SUCCESS = 1
MAXIMUM_COMPLETION_TIMEOUT_MS = 60 * 60 * 1000

STATUS_OBSERVATION_GAP = (
    "wafer-run board stdout and BoardRuntimeInvocationResult do not expose "
    "the 16 per-rank Direct-DTE status resource IDs and raw u32 values; the "
    "successful runtime path enforces all-rank status-v2 Success internally, "
    "but this output cannot independently reconstruct those readbacks"
)


@dataclasses.dataclass(frozen=True)
class DirectDTEManifestEvidence:
    """Typed Direct-DTE package facts indexed by logical rank."""

    status_resource_by_rank: tuple[tuple[int, int], ...]
    terminal_completion_by_rank: tuple[tuple[int, int], ...]
    status_abi: str = DIRECT_DTE_STATUS_ABI
    host_watchdog_ranks: tuple[int, ...] = tuple(range(RANK_COUNT))

    @property
    def status_resources(self) -> frozenset[int]:
        return frozenset(
            resource for _, resource in self.status_resource_by_rank
        )


@dataclasses.dataclass(frozen=True)
class DirectDTEStatusObservation:
    """One raw status row from an output surface that actually exposes it."""

    rank: int
    resource: int
    status_abi: str
    value: int


@dataclasses.dataclass(frozen=True)
class DirectDTEBoardEvidence:
    """Evidence available after validating one successful board invocation."""

    completion_timeout_ms: int
    terminal_completion_by_rank: tuple[tuple[int, int], ...]
    runtime_all_rank_success_enforced: bool
    observed_status_by_rank: (
        tuple[DirectDTEStatusObservation, ...] | None
    )
    status_observation_gap: str | None

    @property
    def has_observed_all_rank_success(self) -> bool:
        return self.observed_status_by_rank is not None


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


def _records_by_dense_rank(
    records: Iterable[Mapping[str, object]],
    *,
    context: str,
) -> dict[int, Mapping[str, object]]:
    by_rank: dict[int, Mapping[str, object]] = {}
    for record in records:
        rank = record.get("rank")
        if not _is_integer(rank) or not 0 <= rank < RANK_COUNT:
            raise RuntimeError(f"{context} has an invalid logical rank")
        if rank in by_rank:
            raise RuntimeError(f"{context} duplicates logical rank {rank}")
        by_rank[rank] = record
    if set(by_rank) != set(range(RANK_COUNT)):
        raise RuntimeError(f"{context} is not the exact 16-rank domain")
    return by_rank


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
        manifest.get("rank_count") != RANK_COUNT
        or not isinstance(target, Mapping)
    ):
        raise RuntimeError(
            "Direct-DTE package does not use the current schema-v7 "
            "16-rank cluster launch contract"
        )

    resources = _require_record_list(manifest, "resources")
    resource_ids: set[int] = set()
    for resource in resources:
        resource_id = resource.get("id")
        if not _is_integer(resource_id) or resource_id in resource_ids:
            raise RuntimeError(
                "Direct-DTE manifest resource IDs are invalid or duplicated"
            )
        resource_ids.add(resource_id)

    status_records = [
        resource
        for resource in resources
        if resource.get("role") == "transport_status"
    ]
    status_by_rank = _records_by_dense_rank(
        status_records, context="Direct-DTE status resource domain"
    )
    status_resource_by_rank: dict[int, int] = {}
    for rank, status in status_by_rank.items():
        resource_id = status.get("id")
        if (
            not _is_integer(resource_id)
            or status.get("role_index") != 0
            or status.get("type") != {"dtype": "u32", "shape": [1]}
            or status.get("bytes") != DIRECT_DTE_STATUS_BYTES
            or status.get("alignment") != DIRECT_DTE_STATUS_ALIGNMENT
            or status.get("access") != "read_write"
            or status.get("host_visible") is not False
        ):
            raise RuntimeError(
                f"rank {rank} Direct-DTE status resource is not the "
                "internal u32[1] status-v2 64/64 read-write slot"
            )
        status_resource_by_rank[rank] = resource_id

    entries = _require_record_list(manifest, "entries")
    if len(entries) != RANK_COUNT:
        raise RuntimeError("Direct-DTE entry domain is not exactly 16 records")
    entry_by_rank = _records_by_dense_rank(
        entries, context="Direct-DTE entry domain"
    )
    entry_ids = {entry.get("id") for entry in entries}
    if (
        entry_ids != set(range(RANK_COUNT))
        or not all(_is_integer(entry_id) for entry_id in entry_ids)
    ):
        raise RuntimeError(
            "Direct-DTE EntryId domain is not dense zero-based"
        )

    terminal_by_rank: dict[int, int] = {}
    for rank, entry in entry_by_rank.items():
        transport = entry.get("transport")
        status_resource = status_resource_by_rank[rank]
        if (
            not isinstance(transport, Mapping)
            or transport.get("kind") != "direct_dte"
            or transport.get("status_resource") != status_resource
            or transport.get("status_abi") != DIRECT_DTE_STATUS_ABI
            or transport.get("host_watchdog_required") is not True
        ):
            raise RuntimeError(
                f"rank {rank} Direct-DTE status ABI/watchdog contract is invalid"
            )

        slots = entry.get("slots")
        if not isinstance(slots, list) or not all(
            isinstance(slot, Mapping) for slot in slots
        ):
            raise RuntimeError(f"rank {rank} Direct-DTE slots are invalid")
        status_slots = [
            slot for slot in slots if slot.get("resource") == status_resource
        ]
        if (
            len(status_slots) != 1
            or status_slots[0].get("access") != "read_write"
            or not _is_integer(status_slots[0].get("ordinal"))
        ):
            raise RuntimeError(
                f"rank {rank} Direct-DTE status resource is not bound by "
                "exactly one read-write ABI slot"
            )

        completion = entry.get("terminal_completion")
        if not _is_integer(completion):
            raise RuntimeError(
                f"rank {rank} terminal completion reference is invalid"
            )
        terminal_by_rank[rank] = completion

    completions = _require_record_list(manifest, "completions")
    if len(completions) != RANK_COUNT:
        raise RuntimeError(
            "Direct-DTE terminal completion domain is not exactly 16 records"
        )
    completion_by_rank = _records_by_dense_rank(
        completions, context="Direct-DTE terminal completion domain"
    )
    completion_ids = {completion.get("id") for completion in completions}
    if (
        completion_ids != set(range(RANK_COUNT))
        or not all(
            _is_integer(completion_id) for completion_id in completion_ids
        )
    ):
        raise RuntimeError(
            "Direct-DTE CompletionId domain is not dense zero-based"
        )
    for rank, completion in completion_by_rank.items():
        if (
            completion.get("kind") != "entry_return"
            or terminal_by_rank[rank] != completion.get("id")
        ):
            raise RuntimeError(
                f"rank {rank} entry/terminal-completion relation is invalid"
            )

    return DirectDTEManifestEvidence(
        status_resource_by_rank=tuple(sorted(status_resource_by_rank.items())),
        terminal_completion_by_rank=tuple(sorted(terminal_by_rank.items())),
    )


def validate_direct_dte_board_command(command: Sequence[str]) -> int:
    """Validate that a board command carries one bounded all-rank deadline."""

    if (
        command.count("--board") != 1
        or command.count("--all-ranks") != 1
        or "--no-card" in command
    ):
        raise RuntimeError(
            "Direct-DTE board command must select one all-rank board invocation"
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
    expected_resource_by_rank = dict(manifest.status_resource_by_rank)
    by_rank: dict[int, DirectDTEStatusObservation] = {}
    for observation in observations:
        if (
            not _is_integer(observation.rank)
            or not _is_integer(observation.resource)
            or not _is_integer(observation.value)
        ):
            raise RuntimeError(
                "Direct-DTE status observation has a non-integer "
                "rank/resource/value"
            )
        if observation.rank in by_rank:
            raise RuntimeError(
                f"Direct-DTE status observations duplicate rank "
                f"{observation.rank}"
            )
        if (
            observation.rank not in expected_resource_by_rank
            or observation.resource
            != expected_resource_by_rank[observation.rank]
            or observation.status_abi != DIRECT_DTE_STATUS_ABI
        ):
            raise RuntimeError(
                "Direct-DTE status observation does not match its typed "
                "manifest resource/ABI"
            )
        if observation.value != DIRECT_DTE_STATUS_SUCCESS:
            raise RuntimeError(
                f"rank {observation.rank} Direct-DTE status is not Success"
            )
        by_rank[observation.rank] = observation
    if set(by_rank) != set(range(RANK_COUNT)):
        raise RuntimeError(
            "Direct-DTE status observations are not the exact 16-rank domain"
        )
    return tuple(by_rank[rank] for rank in range(RANK_COUNT))


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
        "logical_tile_domain: 0..15",
        f"invocation_ranks: {RANK_COUNT}",
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

    terminal_lines = [
        line for line in lines if line.startswith("terminal_completion:")
    ]
    terminal_matches = [
        re.fullmatch(
            r"terminal_completion: (\d+) kind=entry_return rank=(\d+)",
            line,
        )
        for line in terminal_lines
    ]
    if (
        len(terminal_lines) != RANK_COUNT
        or any(match is None for match in terminal_matches)
    ):
        raise RuntimeError(
            "Direct-DTE board output does not expose exactly 16 typed "
            "terminal-completion rows"
        )
    observed_terminal_by_rank: dict[int, int] = {}
    for match in terminal_matches:
        assert match is not None
        completion, rank = (int(value) for value in match.groups())
        if rank in observed_terminal_by_rank:
            raise RuntimeError(
                f"Direct-DTE board output duplicates terminal rank {rank}"
            )
        observed_terminal_by_rank[rank] = completion
    if tuple(sorted(observed_terminal_by_rank.items())) != (
        manifest.terminal_completion_by_rank
    ):
        raise RuntimeError(
            "Direct-DTE board terminal completions differ from the manifest"
        )

    observed_status_by_rank = (
        None
        if status_observations is None
        else _validate_status_observations(status_observations, manifest)
    )
    return DirectDTEBoardEvidence(
        completion_timeout_ms=completion_timeout_ms,
        terminal_completion_by_rank=manifest.terminal_completion_by_rank,
        runtime_all_rank_success_enforced=True,
        observed_status_by_rank=observed_status_by_rank,
        status_observation_gap=(
            STATUS_OBSERVATION_GAP
            if observed_status_by_rank is None
            else None
        ),
    )
