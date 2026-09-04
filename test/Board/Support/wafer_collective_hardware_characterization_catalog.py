#!/usr/bin/env python3
"""Typed contract for collective algorithm source comparisons.

The production compiler and host oracle are current.  This catalog only checks
source construction and host-side relations; board execution and performance
evidence remain pending.  It does not provide a test-only compiler selector or
consume historical board output.
"""

from __future__ import annotations

import dataclasses
import enum
from collections import Counter


class CollectiveKind(str, enum.Enum):
    ALL_GATHER = "all-gather"
    REDUCE_SCATTER = "reduce-scatter"
    ALL_REDUCE = "all-reduce"

    def __str__(self) -> str:
        return self.value


class CollectiveAlternative(str, enum.Enum):
    ALL_GATHER_DIRECT = "all-gather-direct"
    ALL_GATHER_RING = "all-gather-ring"
    REDUCE_SCATTER_DIRECT = "reduce-scatter-direct"
    REDUCE_SCATTER_RING = "reduce-scatter-ring"
    ALL_REDUCE_RING = "all-reduce-ring"
    # Synthetic traffic shape for characterization; not a production lowering.
    ALL_REDUCE_TREE = "all-reduce-tree"

    def __str__(self) -> str:
        return self.value


class CharacterizationDisposition(str, enum.Enum):
    SOURCE_COMPARISON_PENDING_BOARD_EXECUTION = (
        "source-comparison-pending-board-execution"
    )


class NumericOracleKind(str, enum.Enum):
    FULL_OUTPUT_EXACT = "full-output-exact"


class PerformanceEvidenceKind(str, enum.Enum):
    NOT_COLLECTED = "not-collected"


ALTERNATIVE_PHASES: dict[CollectiveAlternative, frozenset[str]] = {
    CollectiveAlternative.ALL_GATHER_DIRECT: frozenset(
        {"all_gather_direct"}
    ),
    CollectiveAlternative.ALL_GATHER_RING: frozenset({"all_gather_ring"}),
    CollectiveAlternative.REDUCE_SCATTER_DIRECT: frozenset(
        {"reduce_scatter_direct"}
    ),
    CollectiveAlternative.REDUCE_SCATTER_RING: frozenset(
        {"reduce_scatter_ring"}
    ),
    CollectiveAlternative.ALL_REDUCE_RING: frozenset({"all_reduce_ring"}),
    CollectiveAlternative.ALL_REDUCE_TREE: frozenset(
        {"all_reduce_tree_reduce", "all_reduce_tree_broadcast"}
    ),
}


@dataclasses.dataclass(frozen=True)
class CollectiveCharacterizationCase:
    key: str
    collective_kind: CollectiveKind
    payload_bytes: int
    left_alternative: CollectiveAlternative
    right_alternative: CollectiveAlternative
    case_order: int
    participant_count: int = 16
    element_type: str = "f16"
    disposition: CharacterizationDisposition = (
        CharacterizationDisposition.SOURCE_COMPARISON_PENDING_BOARD_EXECUTION
    )
    numeric_oracle: NumericOracleKind = NumericOracleKind.FULL_OUTPUT_EXACT
    performance_evidence: PerformanceEvidenceKind = (
        PerformanceEvidenceKind.NOT_COLLECTED
    )
    same_source_required: bool = True
    performance_is_promotion_evidence: bool = False

    @property
    def left_expected_phases(self) -> frozenset[str]:
        return ALTERNATIVE_PHASES[self.left_alternative]

    @property
    def right_expected_phases(self) -> frozenset[str]:
        return ALTERNATIVE_PHASES[self.right_alternative]


_PAIR_SPECS = (
    (
        CollectiveKind.ALL_GATHER,
        CollectiveAlternative.ALL_GATHER_DIRECT,
        CollectiveAlternative.ALL_GATHER_RING,
    ),
    (
        CollectiveKind.REDUCE_SCATTER,
        CollectiveAlternative.REDUCE_SCATTER_DIRECT,
        CollectiveAlternative.REDUCE_SCATTER_RING,
    ),
    (
        CollectiveKind.ALL_REDUCE,
        CollectiveAlternative.ALL_REDUCE_RING,
        CollectiveAlternative.ALL_REDUCE_TREE,
    ),
)

PAYLOAD_BYTES = (256, 4096, 65536)


def _alternative_suffix(alternative: CollectiveAlternative) -> str:
    return alternative.value.rsplit("-", maxsplit=1)[-1]


CASES = tuple(
    CollectiveCharacterizationCase(
        key=(
            f"{collective_kind.value}-"
            f"{_alternative_suffix(left_alternative)}-vs-"
            f"{_alternative_suffix(right_alternative)}-"
            f"{payload_bytes}b"
        ),
        collective_kind=collective_kind,
        payload_bytes=payload_bytes,
        left_alternative=left_alternative,
        right_alternative=right_alternative,
        case_order=pair_index * len(PAYLOAD_BYTES) + payload_index,
        element_type="f16",
    )
    for pair_index, (
        collective_kind,
        left_alternative,
        right_alternative,
    ) in enumerate(_PAIR_SPECS)
    for payload_index, payload_bytes in enumerate(PAYLOAD_BYTES)
)

CASES_BY_KEY = {case.key: case for case in CASES}
CASE_KEYS = tuple(case.key for case in CASES)


def validate_catalog() -> None:
    """Fail closed when the characterization matrix stops being rectangular."""

    if len(CASES_BY_KEY) != len(CASES):
        duplicates = sorted(
            key
            for key, count in Counter(case.key for case in CASES).items()
            if count != 1
        )
        raise ValueError(f"duplicate characterization case keys: {duplicates}")
    if len({case.case_order for case in CASES}) != len(CASES):
        raise ValueError("collective characterization case order is not unique")
    if set(ALTERNATIVE_PHASES) != {
        alternative
        for _, left, right in _PAIR_SPECS
        for alternative in (left, right)
    }:
        raise ValueError("collective alternative phase registry is incomplete")
    for collective_kind, left, right in _PAIR_SPECS:
        matching = tuple(
            case
            for case in CASES
            if case.collective_kind == collective_kind
        )
        if tuple(case.payload_bytes for case in matching) != PAYLOAD_BYTES:
            raise ValueError(
                f"{collective_kind.value}: payload matrix is incomplete"
            )
        if any(
            (case.left_alternative, case.right_alternative) != (left, right)
            for case in matching
        ):
            raise ValueError(
                f"{collective_kind.value}: alternative pair is inconsistent"
            )
        if any(
            not case.same_source_required
            or case.performance_is_promotion_evidence
            or case.participant_count != 16
            or case.element_type != "f16"
            for case in matching
        ):
            raise ValueError(
                f"{collective_kind.value}: characterization contract weakened"
            )


validate_catalog()
