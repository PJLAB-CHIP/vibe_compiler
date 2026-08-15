#!/usr/bin/env python3
"""Typed inventory for collective traffic and Direct-DTE behavior probes.

Pipeline position:
- Upstream IR / input: explicit post-SPMD StableHLO AllToAll or
  CollectivePermute with a verified sixteen-rank distributed boundary.
- Current stage responsibility: define exact traffic-semantics workloads and
  fail closed when a requested route/contention claim is not observable.
- Output IR / files: a verified Direct-DTE package plus, after an armed
  board run, full-output exact correctness and transport lifecycle evidence.
- Downstream consumer: hardware-calibration review and future compiler cost or
  legality work after its separately named activation gates are satisfied.
- User-level driver / named pipeline: the collective traffic behavior driver
  invokes the full structured compilation pipeline with explicit test-only
  post-SPMD carrier injection.
- Explicit non-goals: inventing an AllToAll/Permute algorithm A/B, inferring a
  physical N/E/S/W route from endpoints, treating host time as device cost, or
  representing unsupported concurrent fanout/fanin with sequential operations.
- Completion gate: every executable case has a strong exact oracle and a
  qualified-board activation gate; every unobservable calibration request has
  an explicit typed blocked row rather than an unsafe executable surrogate.
"""

from __future__ import annotations

import dataclasses
import enum
from collections import Counter

import wafer_transport_pmu_calibration_catalog as raw_dte


RANK_COUNT = 16
PAYLOAD_POINTS = (256, 4096, 65536)


class CollectiveKind(str, enum.Enum):
    ALL_TO_ALL = "all-to-all"
    COLLECTIVE_PERMUTE = "collective-permute"


class TrafficGraphKind(str, enum.Enum):
    COMPLETE_EXCHANGE = "complete-exchange"
    FULL_CYCLE_FORWARD = "full-cycle-forward"
    FULL_CYCLE_REVERSE = "full-cycle-reverse"
    OPPOSITE_PAIRS = "opposite-pairs"
    DISJOINT_ADJACENT_PAIRS = "disjoint-adjacent-pairs"
    SPARSE_ROLES = "sparse-roles"
    TWO_EPOCH_CHAIN = "two-epoch-chain"


class CaseDisposition(str, enum.Enum):
    PENDING_BOARD_CORRECTNESS = "pending-board-correctness"


class NumericOracleKind(str, enum.Enum):
    FULL_OUTPUT_EXACT = "full-output-exact"


class StructuralEvidenceKind(str, enum.Enum):
    GENERIC_DIRECT_DTE_CALLS_ONLY = "generic-direct-dte-calls-only"


class DeviceMeasurementState(str, enum.Enum):
    BLOCKED_MISSING_DEVICE_PHASE_BASIS = (
        "blocked-missing-device-phase-basis"
    )


class CoverageDisposition(str, enum.Enum):
    EXISTING_BOARD_EVIDENCE = "existing-board-evidence"
    PENDING_BOARD_EXECUTION = "pending-board-execution"
    BLOCKED_FAIL_CLOSED = "blocked-fail-closed"
    EXISTING_STATIC_NEGATIVE = "existing-static-negative"


class ExecutionGate(str, enum.Enum):
    QUALIFIED_BOARD_CORRECTNESS = "qualified-board-correctness"
    QUALIFIED_RAW_DTE_CORRECTNESS = "qualified-raw-dte-correctness"
    ALREADY_EXECUTED_REFERENCE = "already-executed-reference"
    BLOCKED_MISSING_TYPED_SURFACE = "blocked-missing-typed-surface"
    BLOCKED_UNSUPPORTED_ABI = "blocked-unsupported-abi"
    STATIC_ONLY = "static-only"


class PromotionGate(str, enum.Enum):
    ACCEPTED_MESSAGE_TUPLE_REPORT = "accepted-message-tuple-report"
    DEVICE_PHASE_BASIS = "device-phase-basis"
    FINAL_BUFFER_ALIAS_EVIDENCE = "final-buffer-alias-evidence"
    DECODED_TMNOC_COUNTER = "decoded-tmnoc-counter"
    PHYSICAL_ROUTE_CONTROL_OR_REPORT = "physical-route-control-or-report"
    TYPED_CONCURRENT_MULTI_ENDPOINT_GRAPH = (
        "typed-concurrent-multi-endpoint-graph"
    )
    RECEIVER_FSM_CAPACITY = "receiver-fsm-capacity"
    SEGMENTED_ALL_TO_ALL_SEMANTICS = "segmented-all-to-all-semantics"
    MULTI_CARD_TRANSPORT_ABI = "multi-card-transport-abi"
    NATIVE_MULTICAST_ABI = "native-multicast-abi"


Pair = tuple[int, int]
Epoch = tuple[Pair, ...]


def cycle_pairs(delta: int) -> Epoch:
    return tuple((rank, (rank + delta) % RANK_COUNT) for rank in range(RANK_COUNT))


def all_to_all_shapes(
    payload_bytes: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if payload_bytes % (RANK_COUNT * 4) != 0:
        raise ValueError("AllToAll sentinel payload is not record-aligned")
    lanes_per_peer = payload_bytes // (RANK_COUNT * 4)
    return (
        (RANK_COUNT, lanes_per_peer, 2),
        (1, lanes_per_peer * RANK_COUNT, 2),
    )


def permute_shape(payload_bytes: int) -> tuple[int, ...]:
    if payload_bytes % 4 != 0:
        raise ValueError("Permute sentinel payload is not record-aligned")
    return (payload_bytes // 4, 2)


def cycle_from_order(order: tuple[int, ...]) -> Epoch:
    if len(order) != RANK_COUNT or set(order) != set(range(RANK_COUNT)):
        raise ValueError("Permute cycle order is not the exact rank domain")
    return tuple(
        (source, order[(index + 1) % len(order)])
        for index, source in enumerate(order)
    )


def reverse_epoch(epoch: Epoch) -> Epoch:
    return tuple(sorted((target, source) for source, target in epoch))


def row_major_manhattan_distance(source: int, target: int) -> int:
    source_y, source_x = divmod(source, 4)
    target_y, target_x = divmod(target, 4)
    return abs(source_y - target_y) + abs(source_x - target_x)


NEAREST_CYCLE_ORDER = (
    0,
    1,
    2,
    3,
    7,
    6,
    5,
    9,
    10,
    11,
    15,
    14,
    13,
    12,
    8,
    4,
)
FULL_CYCLE_FORWARD = cycle_from_order(NEAREST_CYCLE_ORDER)
FULL_CYCLE_REVERSE = reverse_epoch(FULL_CYCLE_FORWARD)
OPPOSITE_PAIRS = tuple(
    (
        rank,
        ((rank // 4 + 2) % 4) * 4 + ((rank % 4 + 2) % 4),
    )
    for rank in range(RANK_COUNT)
)
DISJOINT_ADJACENT_PAIRS = tuple(
    (rank, rank ^ 1) for rank in range(RANK_COUNT)
)
SPARSE_ROLE_PAIRS: Epoch = (
    (0, 1),
    (2, 2),
    (3, 4),
    (5, 6),
    (6, 5),
    (7, 8),
    (8, 9),
    (9, 7),
)
TWO_EPOCH_CHAIN = (cycle_pairs(1), cycle_pairs(3))


@dataclasses.dataclass(frozen=True)
class TrafficBehaviorCase:
    key: str
    collective_kind: CollectiveKind
    graph_kind: TrafficGraphKind
    epochs: tuple[Epoch, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    payload_bytes: int
    board_order: int
    objectives: tuple[str, ...]
    rank_count: int = RANK_COUNT
    element_type: str = "f16"
    disposition: CaseDisposition = CaseDisposition.PENDING_BOARD_CORRECTNESS
    numeric_oracle: NumericOracleKind = NumericOracleKind.FULL_OUTPUT_EXACT
    structural_evidence: StructuralEvidenceKind = (
        StructuralEvidenceKind.GENERIC_DIRECT_DTE_CALLS_ONLY
    )
    device_measurement: DeviceMeasurementState = (
        DeviceMeasurementState.BLOCKED_MISSING_DEVICE_PHASE_BASIS
    )
    proves_target_peer_graph: bool = False
    proves_physical_route: bool = False
    proves_device_contention_cost: bool = False
    proves_same_physical_buffer: bool = False


_ALL_TO_ALL_CASES = tuple(
    TrafficBehaviorCase(
        key=f"all-to-all-src-dst-lane-{payload_bytes}b",
        collective_kind=CollectiveKind.ALL_TO_ALL,
        graph_kind=TrafficGraphKind.COMPLETE_EXCHANGE,
        epochs=(),
        input_shape=all_to_all_shapes(payload_bytes)[0],
        output_shape=all_to_all_shapes(payload_bytes)[1],
        payload_bytes=payload_bytes,
        board_order=payload_index,
        objectives=(
            "all sixteen source-to-destination slots",
            "source concat order",
            "lane order",
            "self-slot and fifteen remote slots",
            (
                f"{payload_bytes // RANK_COUNT}-byte payload per peer at "
                f"{payload_bytes} bytes per rank"
            ),
        ),
    )
    for payload_index, payload_bytes in enumerate(PAYLOAD_POINTS)
)

_PERMUTE_FORWARD_CASES = tuple(
    TrafficBehaviorCase(
        key=f"collective-permute-cycle-forward-{payload_bytes}b",
        collective_kind=CollectiveKind.COLLECTIVE_PERMUTE,
        graph_kind=TrafficGraphKind.FULL_CYCLE_FORWARD,
        epochs=(FULL_CYCLE_FORWARD,),
        input_shape=permute_shape(payload_bytes),
        output_shape=permute_shape(payload_bytes),
        payload_bytes=payload_bytes,
        board_order=len(_ALL_TO_ALL_CASES) + payload_index,
        objectives=(
            "one-in/one-out full cycle",
            "forward endpoint direction",
            "nearest intended peer graph",
            f"{payload_bytes}-byte per-rank message",
        ),
    )
    for payload_index, payload_bytes in enumerate(PAYLOAD_POINTS)
)

CASES = (
    *_ALL_TO_ALL_CASES,
    *_PERMUTE_FORWARD_CASES,
    TrafficBehaviorCase(
        key="collective-permute-cycle-reverse-4096b",
        collective_kind=CollectiveKind.COLLECTIVE_PERMUTE,
        graph_kind=TrafficGraphKind.FULL_CYCLE_REVERSE,
        epochs=(FULL_CYCLE_REVERSE,),
        input_shape=permute_shape(4096),
        output_shape=permute_shape(4096),
        payload_bytes=4096,
        board_order=6,
        objectives=(
            "one-in/one-out full cycle",
            "reverse endpoint direction",
            "nearest intended peer graph",
        ),
    ),
    TrafficBehaviorCase(
        key="collective-permute-opposite-pairs-4096b",
        collective_kind=CollectiveKind.COLLECTIVE_PERMUTE,
        graph_kind=TrafficGraphKind.OPPOSITE_PAIRS,
        epochs=(OPPOSITE_PAIRS,),
        input_shape=permute_shape(4096),
        output_shape=permute_shape(4096),
        payload_bytes=4096,
        board_order=7,
        objectives=(
            "same bytes as the cycle cases",
            "long-distance intended endpoints",
            "eight disjoint two-cycles",
        ),
    ),
    TrafficBehaviorCase(
        key="collective-permute-disjoint-adjacent-pairs-4096b",
        collective_kind=CollectiveKind.COLLECTIVE_PERMUTE,
        graph_kind=TrafficGraphKind.DISJOINT_ADJACENT_PAIRS,
        epochs=(DISJOINT_ADJACENT_PAIRS,),
        input_shape=permute_shape(4096),
        output_shape=permute_shape(4096),
        payload_bytes=4096,
        board_order=8,
        objectives=(
            "same bytes as the cycle cases",
            "nearest intended endpoints",
            "eight disjoint two-cycles",
        ),
    ),
    TrafficBehaviorCase(
        key="collective-permute-sparse-roles-4096b",
        collective_kind=CollectiveKind.COLLECTIVE_PERMUTE,
        graph_kind=TrafficGraphKind.SPARSE_ROLES,
        epochs=(SPARSE_ROLE_PAIRS,),
        input_shape=permute_shape(4096),
        output_shape=permute_shape(4096),
        payload_bytes=4096,
        board_order=9,
        objectives=(
            "send-only and receive-only ranks",
            "self copy",
            "unmapped zero fill",
            "two-cycle and three-cycle ranks with both roles",
        ),
    ),
    TrafficBehaviorCase(
        key="collective-permute-two-epoch-chain-4096b",
        collective_kind=CollectiveKind.COLLECTIVE_PERMUTE,
        graph_kind=TrafficGraphKind.TWO_EPOCH_CHAIN,
        epochs=TWO_EPOCH_CHAIN,
        input_shape=permute_shape(4096),
        output_shape=permute_shape(4096),
        payload_bytes=4096,
        board_order=10,
        objectives=(
            "two sequential communication epochs",
            "second epoch consumes the first epoch result",
            "composition differs from either single epoch and from identity",
        ),
    ),
)

CASES_BY_KEY = {case.key: case for case in CASES}
CASE_KEYS = tuple(case.key for case in CASES)


@dataclasses.dataclass(frozen=True)
class CoverageItem:
    key: str
    disposition: CoverageDisposition
    execution_gate: ExecutionGate
    case_keys: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    promotion_gates: tuple[PromotionGate, ...]
    evidence_boundary: str
    raw_case_names: tuple[str, ...] = ()


RAW_DTE_DRIVER = "test/Board/wafer_board_dte_ncc_execution_probe_test.py"
RAW_DTE_CATALOG = "test/Board/wafer_transport_pmu_calibration_catalog.py"
RAW_DTE_PROBE = "test/Board/Inputs/wafer_dte_ncc_execution_probe.c"


COVERAGE_ITEMS = (
    CoverageItem(
        key="all-to-all-equal-split-traffic-semantics",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_BOARD_CORRECTNESS,
        case_keys=tuple(case.key for case in _ALL_TO_ALL_CASES),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.ACCEPTED_MESSAGE_TUPLE_REPORT,
            PromotionGate.DEVICE_PHASE_BASIS,
        ),
        evidence_boundary=(
            "Exact output can prove equal-split traffic semantics. Generic ELF "
            "DTE calls cannot prove the final per-rank peer tuples or a cost."
        ),
    ),
    CoverageItem(
        key="all-to-all-segmented-or-ragged-traffic",
        disposition=CoverageDisposition.BLOCKED_FAIL_CLOSED,
        execution_gate=ExecutionGate.BLOCKED_MISSING_TYPED_SURFACE,
        case_keys=(),
        evidence_refs=(),
        promotion_gates=(PromotionGate.SEGMENTED_ALL_TO_ALL_SEMANTICS,),
        evidence_boundary=(
            "Current structured AllToAll requires an equal split count matching "
            "the rank group; it cannot stand in for segmented or ragged traffic."
        ),
    ),
    CoverageItem(
        key="collective-permute-cycle-directions",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_BOARD_CORRECTNESS,
        case_keys=(
            "collective-permute-cycle-forward-256b",
            "collective-permute-cycle-forward-4096b",
            "collective-permute-cycle-forward-65536b",
            "collective-permute-cycle-reverse-4096b",
        ),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.ACCEPTED_MESSAGE_TUPLE_REPORT,
            PromotionGate.DEVICE_PHASE_BASIS,
        ),
        evidence_boundary=(
            "Exact output distinguishes both logical directions; it does not "
            "identify the physical link route or device phase duration."
        ),
    ),
    CoverageItem(
        key="collective-permute-sparse-role-semantics",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_BOARD_CORRECTNESS,
        case_keys=("collective-permute-sparse-roles-4096b",),
        evidence_refs=(),
        promotion_gates=(PromotionGate.ACCEPTED_MESSAGE_TUPLE_REPORT,),
        evidence_boundary=(
            "Exact output covers send-only, receive-only, self and unmapped "
            "zero-fill roles without inferring transport timing."
        ),
    ),
    CoverageItem(
        key="collective-permute-double-epoch-semantics",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_BOARD_CORRECTNESS,
        case_keys=("collective-permute-two-epoch-chain-4096b",),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.ACCEPTED_MESSAGE_TUPLE_REPORT,
            PromotionGate.FINAL_BUFFER_ALIAS_EVIDENCE,
        ),
        evidence_boundary=(
            "The chained oracle proves two logical epochs. It deliberately does "
            "not claim that final allocation reused one physical buffer."
        ),
    ),
    CoverageItem(
        key="collective-permute-same-physical-buffer-reuse",
        disposition=CoverageDisposition.BLOCKED_FAIL_CLOSED,
        execution_gate=ExecutionGate.BLOCKED_MISSING_TYPED_SURFACE,
        case_keys=("collective-permute-two-epoch-chain-4096b",),
        evidence_refs=(),
        promotion_gates=(PromotionGate.FINAL_BUFFER_ALIAS_EVIDENCE,),
        evidence_boundary=(
            "The package has no accepted buffer-alias report, so a successful "
            "double-epoch run cannot certify same-physical-buffer reuse."
        ),
    ),
    CoverageItem(
        key="dte-nearest-endpoint-direction-pair",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_BOARD_CORRECTNESS,
        case_keys=(
            "collective-permute-cycle-forward-4096b",
            "collective-permute-cycle-reverse-4096b",
        ),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.ACCEPTED_MESSAGE_TUPLE_REPORT,
            PromotionGate.DEVICE_PHASE_BASIS,
        ),
        evidence_boundary=(
            "The source graphs hold bytes constant and reverse endpoints. A "
            "future accepted message report must prove the target graph."
        ),
    ),
    CoverageItem(
        key="dte-long-distance-endpoint-graph",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_BOARD_CORRECTNESS,
        case_keys=("collective-permute-opposite-pairs-4096b",),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.ACCEPTED_MESSAGE_TUPLE_REPORT,
            PromotionGate.DEVICE_PHASE_BASIS,
        ),
        evidence_boundary=(
            "The logical opposite-pair graph is runnable, but endpoint distance "
            "is not a physical-route or timing observation."
        ),
    ),
    CoverageItem(
        key="dte-disjoint-endpoint-graph",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_BOARD_CORRECTNESS,
        case_keys=("collective-permute-disjoint-adjacent-pairs-4096b",),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.ACCEPTED_MESSAGE_TUPLE_REPORT,
            PromotionGate.DEVICE_PHASE_BASIS,
        ),
        evidence_boundary=(
            "The logical graph is eight disjoint nearest pairs; exact output "
            "alone cannot prove physical path disjointness."
        ),
    ),
    CoverageItem(
        key="dte-fanout-fanin-one-payload-sweep",
        disposition=CoverageDisposition.EXISTING_BOARD_EVIDENCE,
        execution_gate=ExecutionGate.ALREADY_EXECUTED_REFERENCE,
        case_keys=(),
        evidence_refs=(RAW_DTE_DRIVER, RAW_DTE_CATALOG),
        promotion_gates=(PromotionGate.DEVICE_PHASE_BASIS,),
        evidence_boundary=(
            "Existing receiver-first ring modes cover one destination/source "
            "at 16, 32, 64, 256 and 4096 bytes with exact guards and lifecycle."
        ),
    ),
    CoverageItem(
        key="dte-fanout-fanin-two-payload-sweep",
        disposition=CoverageDisposition.EXISTING_BOARD_EVIDENCE,
        execution_gate=ExecutionGate.ALREADY_EXECUTED_REFERENCE,
        case_keys=(),
        evidence_refs=(RAW_DTE_DRIVER, RAW_DTE_CATALOG),
        promotion_gates=(PromotionGate.DEVICE_PHASE_BASIS,),
        evidence_boundary=(
            "Existing mode 6 covers two destinations and matching two-source "
            "receive behavior; its sequential send waits are not contention."
        ),
    ),
    *tuple(
        CoverageItem(
            key=f"dte-native-{semantic}-fanout-layout-matrix",
            disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
            execution_gate=ExecutionGate.QUALIFIED_RAW_DTE_CORRECTNESS,
            case_keys=(),
            evidence_refs=(
                RAW_DTE_DRIVER,
                RAW_DTE_CATALOG,
                RAW_DTE_PROBE,
            ),
            promotion_gates=(
                PromotionGate.NATIVE_MULTICAST_ABI,
                PromotionGate.TYPED_CONCURRENT_MULTI_ENDPOINT_GRAPH,
            ),
            evidence_boundary=(
                f"The owner-backed raw DTE {semantic} rows separately cover "
                "fanout 2/4/8/15 and adjacent/interleaved destination order. "
                "Exact destination payloads, inactive ranks, guards, raw mode/"
                "dest_num/user-id echo, completion and cleanup are required; "
                "the result does not establish device cost or physical route."
            ),
            raw_case_names=tuple(
                case.name
                for case in raw_dte.RAW_MULTIDEST_CASES
                if case.semantic == semantic
            ),
        )
        for semantic in raw_dte.RAW_MULTIDEST_SEMANTICS
    ),
    CoverageItem(
        key="dte-four-source-fanin",
        disposition=CoverageDisposition.PENDING_BOARD_EXECUTION,
        execution_gate=ExecutionGate.QUALIFIED_RAW_DTE_CORRECTNESS,
        case_keys=(),
        evidence_refs=(RAW_DTE_DRIVER, RAW_DTE_CATALOG, RAW_DTE_PROBE),
        promotion_gates=(
            PromotionGate.TYPED_CONCURRENT_MULTI_ENDPOINT_GRAPH,
            PromotionGate.RECEIVER_FSM_CAPACITY,
        ),
        evidence_boundary=(
            "Four source ranks issue to four receiver FSMs and disjoint guarded "
            "slots on one target. Exact source-slot identity and lifecycle can "
            "establish four-source fan-in correctness, but not timing overlap "
            "or a contention cost."
        ),
        raw_case_names=("dte-four-source-fanin-correctness",),
    ),
    CoverageItem(
        key="dte-eight-source-fanin",
        disposition=CoverageDisposition.BLOCKED_FAIL_CLOSED,
        execution_gate=ExecutionGate.BLOCKED_UNSUPPORTED_ABI,
        case_keys=(),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.TYPED_CONCURRENT_MULTI_ENDPOINT_GRAPH,
            PromotionGate.RECEIVER_FSM_CAPACITY,
        ),
        evidence_boundary=(
            "The qualified receiver helper exposes only four FSM ids, so an "
            "eight-source fan-in cannot be prepared safely."
        ),
    ),
    CoverageItem(
        key="dte-fifteen-source-fanin",
        disposition=CoverageDisposition.BLOCKED_FAIL_CLOSED,
        execution_gate=ExecutionGate.BLOCKED_UNSUPPORTED_ABI,
        case_keys=(),
        evidence_refs=(),
        promotion_gates=(
            PromotionGate.TYPED_CONCURRENT_MULTI_ENDPOINT_GRAPH,
            PromotionGate.RECEIVER_FSM_CAPACITY,
        ),
        evidence_boundary=(
            "AllToAll reaches fifteen sources through separate exchanges; the "
            "four-FSM receiver surface cannot represent fifteen live sources "
            "to one target."
        ),
    ),
    CoverageItem(
        key="dte-device-phase-contention-cost",
        disposition=CoverageDisposition.BLOCKED_FAIL_CLOSED,
        execution_gate=ExecutionGate.BLOCKED_MISSING_TYPED_SURFACE,
        case_keys=(
            "collective-permute-cycle-forward-4096b",
            "collective-permute-cycle-reverse-4096b",
            "collective-permute-opposite-pairs-4096b",
            "collective-permute-disjoint-adjacent-pairs-4096b",
        ),
        evidence_refs=(),
        promotion_gates=(PromotionGate.DEVICE_PHASE_BASIS,),
        evidence_boundary=(
            "Host elapsed time and the non-monotonic DTE execution counter are "
            "not a credible device phase basis."
        ),
    ),
    CoverageItem(
        key="dte-physical-link-route",
        disposition=CoverageDisposition.BLOCKED_FAIL_CLOSED,
        execution_gate=ExecutionGate.BLOCKED_MISSING_TYPED_SURFACE,
        case_keys=(
            "collective-permute-cycle-forward-4096b",
            "collective-permute-cycle-reverse-4096b",
            "collective-permute-opposite-pairs-4096b",
            "collective-permute-disjoint-adjacent-pairs-4096b",
        ),
        evidence_refs=(),
        promotion_gates=(PromotionGate.PHYSICAL_ROUTE_CONTROL_OR_REPORT,),
        evidence_boundary=(
            "Logical endpoints and minimum-hop demand do not identify physical "
            "N/E/S/W links, arbitration, or per-link load."
        ),
    ),
    CoverageItem(
        key="dte-cross-card-route",
        disposition=CoverageDisposition.BLOCKED_FAIL_CLOSED,
        execution_gate=ExecutionGate.BLOCKED_UNSUPPORTED_ABI,
        case_keys=(),
        evidence_refs=(),
        promotion_gates=(PromotionGate.MULTI_CARD_TRANSPORT_ABI,),
        evidence_boundary=(
            "The current package and launch contract is single-card only."
        ),
    ),
    CoverageItem(
        key="tmnoc-per-link-counter",
        disposition=CoverageDisposition.EXISTING_STATIC_NEGATIVE,
        execution_gate=ExecutionGate.STATIC_ONLY,
        case_keys=(),
        evidence_refs=(RAW_DTE_CATALOG,),
        promotion_gates=(PromotionGate.DECODED_TMNOC_COUNTER,),
        evidence_boundary=(
            "Only a TMNOC base is known; without a decoded read-only offset the "
            "counter remains static-negative and is never sampled speculatively."
        ),
    ),
)

COVERAGE_BY_KEY = {item.key: item for item in COVERAGE_ITEMS}
COVERAGE_KEYS = tuple(item.key for item in COVERAGE_ITEMS)


def _validate_epoch(epoch: Epoch, case_key: str) -> None:
    sources = [source for source, _ in epoch]
    targets = [target for _, target in epoch]
    if any(
        type(rank) is not int or not 0 <= rank < RANK_COUNT
        for pair in epoch
        for rank in pair
    ):
        raise ValueError(f"{case_key}: source/target rank is out of range")
    if len(sources) != len(set(sources)):
        raise ValueError(f"{case_key}: one epoch has duplicate sources")
    if len(targets) != len(set(targets)):
        raise ValueError(f"{case_key}: one epoch has duplicate targets")


def validate_catalog() -> None:
    """Validate completeness and all fail-closed evidence boundaries."""

    if len(CASES_BY_KEY) != len(CASES):
        duplicates = sorted(
            key
            for key, count in Counter(case.key for case in CASES).items()
            if count != 1
        )
        raise ValueError(f"duplicate traffic case keys: {duplicates}")
    if tuple(case.board_order for case in CASES) != tuple(range(len(CASES))):
        raise ValueError("traffic case board order is not contiguous")
    if len(COVERAGE_BY_KEY) != len(COVERAGE_ITEMS):
        raise ValueError("duplicate traffic coverage keys")

    for case in CASES:
        if (
            case.rank_count != RANK_COUNT
            or case.payload_bytes not in PAYLOAD_POINTS
            or case.element_type != "f16"
            or case.disposition != CaseDisposition.PENDING_BOARD_CORRECTNESS
            or case.numeric_oracle != NumericOracleKind.FULL_OUTPUT_EXACT
            or case.structural_evidence
            != StructuralEvidenceKind.GENERIC_DIRECT_DTE_CALLS_ONLY
            or case.device_measurement
            != DeviceMeasurementState.BLOCKED_MISSING_DEVICE_PHASE_BASIS
            or case.proves_target_peer_graph
            or case.proves_physical_route
            or case.proves_device_contention_cost
            or case.proves_same_physical_buffer
        ):
            raise ValueError(f"{case.key}: evidence boundary was weakened")
        input_bytes = 2
        for dimension in case.input_shape:
            input_bytes *= dimension
        output_bytes = 2
        for dimension in case.output_shape:
            output_bytes *= dimension
        if (
            input_bytes != case.payload_bytes
            or output_bytes != case.payload_bytes
        ):
            raise ValueError(
                f"{case.key}: payload shape differs from its byte contract"
            )
        if case.collective_kind == CollectiveKind.ALL_TO_ALL:
            if case.epochs or case.graph_kind != TrafficGraphKind.COMPLETE_EXCHANGE:
                raise ValueError(f"{case.key}: AllToAll graph contract is invalid")
            if (
                case.input_shape,
                case.output_shape,
            ) != all_to_all_shapes(case.payload_bytes):
                raise ValueError(f"{case.key}: AllToAll shape contract is invalid")
        else:
            if not case.epochs:
                raise ValueError(f"{case.key}: Permute has no epoch")
            if (
                case.input_shape != permute_shape(case.payload_bytes)
                or case.output_shape != permute_shape(case.payload_bytes)
            ):
                raise ValueError(f"{case.key}: Permute shape contract is invalid")
            for epoch in case.epochs:
                _validate_epoch(epoch, case.key)

    if tuple(case.payload_bytes for case in _ALL_TO_ALL_CASES) != PAYLOAD_POINTS:
        raise ValueError("AllToAll payload matrix is incomplete")
    if (
        tuple(case.payload_bytes for case in _PERMUTE_FORWARD_CASES)
        != PAYLOAD_POINTS
    ):
        raise ValueError("Permute payload matrix is incomplete")

    expected_cases = set(CASE_KEYS)
    referenced_cases = {
        case_key for item in COVERAGE_ITEMS for case_key in item.case_keys
    }
    if referenced_cases != expected_cases:
        raise ValueError(
            "coverage rows do not cover executable traffic cases: "
            f"missing={sorted(expected_cases - referenced_cases)} "
            f"unknown={sorted(referenced_cases - expected_cases)}"
        )
    expected_raw_cases = {
        case.name for case in raw_dte.RAW_MULTIDEST_CASES
    } | {"dte-four-source-fanin-correctness"}
    referenced_raw_cases = Counter(
        name for item in COVERAGE_ITEMS for name in item.raw_case_names
    )
    if (
        set(referenced_raw_cases) != expected_raw_cases
        or any(count != 1 for count in referenced_raw_cases.values())
    ):
        raise ValueError(
            "raw DTE coverage rows are incomplete or duplicated: "
            f"missing={sorted(expected_raw_cases - set(referenced_raw_cases))} "
            f"unknown={sorted(set(referenced_raw_cases) - expected_raw_cases)} "
            f"duplicates={sorted(name for name, count in referenced_raw_cases.items() if count != 1)}"
        )
    for item in COVERAGE_ITEMS:
        if any(case_key not in CASES_BY_KEY for case_key in item.case_keys):
            raise ValueError(f"{item.key}: coverage refers to an unknown case")
        if item.disposition == CoverageDisposition.EXISTING_BOARD_EVIDENCE:
            if (
                item.execution_gate != ExecutionGate.ALREADY_EXECUTED_REFERENCE
                or item.case_keys
                or not item.evidence_refs
            ):
                raise ValueError(f"{item.key}: existing evidence is incomplete")
        elif item.disposition == CoverageDisposition.PENDING_BOARD_EXECUTION:
            structured_pending = (
                item.execution_gate
                == ExecutionGate.QUALIFIED_BOARD_CORRECTNESS
                and bool(item.case_keys)
                and not item.raw_case_names
            )
            raw_pending = (
                item.execution_gate
                == ExecutionGate.QUALIFIED_RAW_DTE_CORRECTNESS
                and bool(item.raw_case_names)
                and not item.case_keys
                and bool(item.evidence_refs)
            )
            if not structured_pending and not raw_pending:
                raise ValueError(f"{item.key}: pending execution has no case")
        elif item.disposition == CoverageDisposition.EXISTING_STATIC_NEGATIVE:
            if (
                item.execution_gate != ExecutionGate.STATIC_ONLY
                or not item.evidence_refs
            ):
                raise ValueError(f"{item.key}: static-negative row is invalid")
        elif not item.execution_gate.value.startswith("blocked-"):
            raise ValueError(f"{item.key}: blocked row has an executable gate")

    double_epoch = CASES_BY_KEY["collective-permute-two-epoch-chain-4096b"]
    if len(double_epoch.epochs) != 2:
        raise ValueError("double-epoch case no longer has two epochs")
    composed = {
        source: target
        for source, middle in double_epoch.epochs[0]
        for middle_source, target in double_epoch.epochs[1]
        if middle_source == middle
    }
    if (
        len(composed) != RANK_COUNT
        or any(source == target for source, target in composed.items())
        or set(composed.items()) == set(double_epoch.epochs[0])
        or set(composed.items()) == set(double_epoch.epochs[1])
    ):
        raise ValueError("double-epoch composition is not mutation-discriminating")

    if (
        any(
            row_major_manhattan_distance(source, target) != 1
            for source, target in FULL_CYCLE_FORWARD
        )
        or set(FULL_CYCLE_REVERSE)
        != {(target, source) for source, target in FULL_CYCLE_FORWARD}
    ):
        raise ValueError("nearest cycle/reverse endpoint contract is invalid")
    if any(
        row_major_manhattan_distance(source, target) != 4
        for source, target in OPPOSITE_PAIRS
    ):
        raise ValueError("opposite-pair endpoint distance is not uniform four-hop")
    if any(
        row_major_manhattan_distance(source, target) != 1
        for source, target in DISJOINT_ADJACENT_PAIRS
    ):
        raise ValueError("disjoint adjacent-pair endpoint contract is invalid")


validate_catalog()
