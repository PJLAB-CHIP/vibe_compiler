#!/usr/bin/env python3
"""Typed fail-closed gates for hardware behavior with no observable ABI.

Pipeline position:
- Upstream artifact / IR:
  Version-matched NCC request/record fields and the sustained SPM pilot
  request/record contract, plus the explicitly qualified ArgMin catalog rows.
- Current stage responsibility:
  Prove that queue resident occupancy and physical SPM bank identity cannot
  be represented by those contracts; keep every unenumerated ArgMin numeric
  domain closed; and reject any board request that claims otherwise.
- Output artifact / IR:
  A typed preparation rejection.  No board package, scheduler hint, bank
  side table, or inferred occupancy value is produced.
- Downstream consumer:
  The pending hardware-calibration inventory and conservative compiler cost
  fallback.
- User-level driver / named pipeline:
  ``wafer_unrepresentable_hardware_behavior_catalog_test.py``.
- Explicit non-goals:
  Treating task-done state as queue occupancy, treating SPM offsets as bank
  numbers, sampling undocumented MMIO, or extrapolating one numeric domain
  from another.
- Completion gate:
  Every key is rejected, the versioned records contain no surrogate field,
  and the nearest safe executable characterization remains named.
"""

from __future__ import annotations

import dataclasses

import wafer_instruction_family_catalog as instruction_family
import wafer_ncc_probe_protocol as ncc
import wafer_spm_sustained_conflict_catalog as spm


class PreparationBlocked(RuntimeError):
    """No honest board request can be serialized for this behavior."""


ARGMIN_EXPLICIT_BOARD_CASES = (
    "peripheral-argmin-f16",
    "peripheral-argmin-negative-f16-observed",
    "peripheral-argmin-tie-f16-observed",
    "peripheral-argmin-nan-f16-observed",
)


@dataclasses.dataclass(frozen=True)
class UnrepresentableBehavior:
    key: str
    missing_surface: str
    rejected_surrogates: tuple[str, ...]
    safe_executable_family: str
    unblock_condition: str

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


BEHAVIORS = (
    UnrepresentableBehavior(
        key="queue-active-resident-count-unobservable",
        missing_surface=(
            "versioned read-only queue head/tail/occupancy or resident-count "
            "record on the same issue boundary"
        ),
        rejected_surrogates=(
            "documented static pending depth",
            "task-done control bit",
            "total completed submissions",
            "last-issue host elapsed time",
        ),
        safe_executable_family="queue-saturation-response",
        unblock_condition=(
            "an owner-backed read-only occupancy surface with unit, scope, "
            "sampling boundary, and wrap semantics"
        ),
    ),
    UnrepresentableBehavior(
        key="spm-physical-bank-color-class-unobservable",
        missing_surface=(
            "owner-backed address-to-bank/port mapping or a decoded "
            "bank-specific read-only counter"
        ),
        rejected_surrogates=(
            "relative SPM offset",
            "base modulo alignment",
            "aggregate engine execution cycles",
            "serial/window timing direction alone",
        ),
        safe_executable_family="spm-sustained-conflict-pilot",
        unblock_condition=(
            "a version-matched bank/port mapping or bank-specific counter "
            "whose identity and sampling scope are owned by the target ABI"
        ),
    ),
    UnrepresentableBehavior(
        key="argmin-unqualified-domain",
        missing_surface=(
            "an explicit typed ArgMin board row and coherent value/index "
            "classification for any domain outside the four reviewed F16 "
            "positive-finite, negative-finite, tie, and NaN rows"
        ),
        rejected_surrogates=(
            "neighboring CT special-value behavior",
            "F16 behavior extrapolated to BF16 or F32",
            "finite values extrapolated to signed zero or infinity",
            "normal values extrapolated to subnormal handling",
        ),
        safe_executable_family="argmin-tie-and-nan-domain",
        unblock_condition=(
            "a separately named typed catalog row with distinguishing inputs, "
            "coherent value/index oracle, bounded writeback, guards, and "
            "explicit board qualification"
        ),
    ),
)
BEHAVIORS_BY_KEY = {behavior.key: behavior for behavior in BEHAVIORS}


def prepare_board_request(key: str) -> None:
    if key not in BEHAVIORS_BY_KEY:
        raise KeyError(key)
    behavior = BEHAVIORS_BY_KEY[key]
    raise PreparationBlocked(
        f"{behavior.key}: missing {behavior.missing_surface}; "
        f"safe alternative is {behavior.safe_executable_family}"
    )


def validate_catalog() -> None:
    if len(BEHAVIORS_BY_KEY) != len(BEHAVIORS):
        raise RuntimeError("unrepresentable behavior keys are not unique")
    if any(
        not behavior.rejected_surrogates
        or not behavior.safe_executable_family
        or not behavior.unblock_condition
        for behavior in BEHAVIORS
    ):
        raise RuntimeError("unrepresentable behavior gate is incomplete")

    ncc_record_names = set(ncc.REC)
    forbidden_queue_fields = {
        "QUEUE_HEAD",
        "QUEUE_TAIL",
        "QUEUE_OCCUPANCY",
        "RESIDENT_COUNT",
        "ACTIVE_RESIDENT_COUNT",
    }
    if ncc_record_names & forbidden_queue_fields:
        raise RuntimeError(
            "queue gate is stale: an occupancy field now exists and must be "
            "reviewed instead of rejected"
        )

    spm_record_names = set(spm.REC) | set(spm.ROW_REC)
    forbidden_spm_fields = {
        "BANK",
        "BANK_ID",
        "PORT",
        "PORT_ID",
        "BANK_CONFLICT_COUNT",
    }
    if spm_record_names & forbidden_spm_fields:
        raise RuntimeError(
            "SPM bank gate is stale: a physical-class field now exists and "
            "must be reviewed instead of rejected"
        )

    actual_argmin_cases = {
        case.name
        for case in instruction_family.CATALOG
        if case.symbol.startswith("PERIPHERAL_ARGMIN_")
        or case.symbol == "PERIPHERAL_ARGMIN_F16"
    }
    if actual_argmin_cases != set(ARGMIN_EXPLICIT_BOARD_CASES):
        raise RuntimeError(
            "ArgMin domain gate is stale: catalog domains changed and must be "
            "reviewed before board qualification"
        )


validate_catalog()
