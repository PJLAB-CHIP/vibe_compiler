#!/usr/bin/env python3
"""Typed capability inventory for CT opcodes 111..123.

This catalog intentionally does not assign capability to a bare opcode.  A
row is identified by (opcode, dtype, axis/layout, geometry), because each of
those fields can change packet legality, numeric behavior, or the physical
SPM span.  Cross-products are limited to equivalence classes that change one
of those contracts; unimplemented classes remain explicit non-board rows.
"""

from __future__ import annotations

import dataclasses


DTYPES = ("f16", "bf16", "f32")

OPCODE_NAMES = {
    111: "reduce-sum",
    112: "reduce-avg",
    113: "reduce-max",
    114: "reduce-min",
    115: "pool-avg",
    116: "pool-sum",
    117: "pool-max",
    118: "pool-indexed-max",
    119: "pool-min",
    120: "pool-indexed-min",
    121: "unpool-index",
    122: "unpool-avg",
    123: "unpool-mask",
}


@dataclasses.dataclass(frozen=True)
class CTCapabilityRow:
    opcode: int
    opcode_name: str
    family: str
    dtype: str
    axis_layout: str
    geometry: str
    disposition: str
    evidence: tuple[str, ...]
    oracle: str | None
    qualification: str
    reason: str | None = None

    @property
    def key(self) -> tuple[int, str, str, str]:
        return (
            self.opcode,
            self.dtype,
            self.axis_layout,
            self.geometry,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            **dataclasses.asdict(self),
        }


def _row(
    opcode: int,
    family: str,
    dtype: str,
    axis_layout: str,
    geometry: str,
    disposition: str,
    *,
    evidence: tuple[str, ...] = (),
    oracle: str | None = None,
    qualification: str,
    reason: str | None = None,
) -> CTCapabilityRow:
    if disposition in {"board-executable", "board-observation"}:
        if not evidence or oracle is None:
            raise RuntimeError(
                f"{opcode}/{dtype}/{axis_layout}/{geometry}: "
                "board row requires evidence and oracle"
            )
    elif reason is None:
        raise RuntimeError(
            f"{opcode}/{dtype}/{axis_layout}/{geometry}: "
            "non-board row requires a reason"
        )
    return CTCapabilityRow(
        opcode,
        OPCODE_NAMES[opcode],
        family,
        dtype,
        axis_layout,
        geometry,
        disposition,
        evidence,
        oracle,
        qualification,
        reason,
    )


def _positive(
    opcode: int,
    family: str,
    dtype: str,
    axis_layout: str,
    geometry: str,
    evidence: str,
    *,
    qualification: str = "board-passed",
) -> CTCapabilityRow:
    return _row(
        opcode,
        family,
        dtype,
        axis_layout,
        geometry,
        "board-executable",
        evidence=(evidence,),
        oracle="exact-bits+physical-span+guard",
        qualification=qualification,
    )


def _observation(
    opcode: int,
    family: str,
    dtype: str,
    axis_layout: str,
    geometry: str,
    evidence: str,
    reason: str,
    *,
    qualification: str = "board-observed",
) -> CTCapabilityRow:
    return _row(
        opcode,
        family,
        dtype,
        axis_layout,
        geometry,
        "board-observation",
        evidence=(evidence,),
        oracle="bounded-raw+physical-span+guard",
        qualification=qualification,
        reason=reason,
    )


def _deferred(
    opcode: int,
    family: str,
    dtype: str,
    axis_layout: str,
    geometry: str,
    reason: str,
) -> CTCapabilityRow:
    return _row(
        opcode,
        family,
        dtype,
        axis_layout,
        geometry,
        "unknown",
        qualification="remaining",
        reason=reason,
    )


def _isolated_deferred(
    opcode: int,
    family: str,
    dtype: str,
    axis_layout: str,
    geometry: str,
    reason: str,
) -> CTCapabilityRow:
    return _row(
        opcode,
        family,
        dtype,
        axis_layout,
        geometry,
        "isolated-deferred",
        qualification="isolated-timeout",
        reason=reason,
    )


def _negative(
    opcode: int,
    family: str,
    dtype: str,
    axis_layout: str,
    geometry: str,
    reason: str,
) -> CTCapabilityRow:
    return _row(
        opcode,
        family,
        dtype,
        axis_layout,
        geometry,
        "static-negative",
        qualification="host-negative",
        reason=reason,
    )


_REDUCE_EVIDENCE = {
    (111, "f16"): "reduce-sum-f16",
    (112, "bf16"): "reduce-avg-bf16",
    (113, "f16"): "reduce-max-f16",
    (114, "bf16"): "reduce-min-bf16",
}

_REDUCE_AXIS_GEOMETRIES = {
    "W/NCx": "n1-h1-w4-c64",
    "C/NCx": "n1-h2-w3-c65",
    "H/NCx": "n1-h3-w2-c65",
    "HW/NCx": "n1-h3-w2-c65",
}


def _reduce_rows() -> list[CTCapabilityRow]:
    rows: list[CTCapabilityRow] = []
    for opcode in range(111, 115):
        for dtype in DTYPES:
            for axis_layout, geometry in _REDUCE_AXIS_GEOMETRIES.items():
                evidence = _REDUCE_EVIDENCE.get((opcode, dtype))
                if axis_layout == "W/NCx" and evidence is not None:
                    rows.append(
                        _positive(
                            opcode,
                            "reduce",
                            dtype,
                            axis_layout,
                            geometry,
                            evidence,
                        )
                    )
                    continue
                rows.append(
                    _positive(
                        opcode,
                        "reduce",
                        dtype,
                        axis_layout,
                        geometry,
                        f"{OPCODE_NAMES[opcode]}-{dtype}-"
                        f"{axis_layout.split('/')[0].lower()}-ncx-"
                        f"{geometry.replace('-', '')}",
                        qualification="pending-board",
                    )
                )
        rows.append(
            _positive(
                opcode,
                "reduce",
                "f16",
                "axis-specific/Cx",
                "w4-c8",
                f"{OPCODE_NAMES[opcode]}-f16-c-cx-w4c8",
                qualification="pending-board",
            )
        )
        for axis in ("N", "HWC"):
            rows.append(
                _isolated_deferred(
                    opcode,
                    "reduce",
                    "f16",
                    f"{axis}/NCx",
                    "public-enum-absent",
                    "the version-matched Reduce_Dim enum exposes only "
                    "C/W/H/HW; isolated case 196 using historical raw "
                    "dimension 3 exceeded its outer completion deadline and "
                    "may permanently wait.  Raw N/HWC variants are "
                    "fail-closed until a separately authorized isolated "
                    "protocol is designed",
                )
            )
    return rows


_POOL_SYMMETRIC_EVIDENCE = {
    (115, "f16"): "pool-avg-f16",
    (116, "f16"): "pool-sum-f16",
    (117, "f16"): "pool-f16",
    (117, "bf16"): "pool-bf16",
    (119, "f16"): "pool-min-f16",
    (120, "f16"): "pool-indexed-min-f16",
}


def _pool_rows() -> list[CTCapabilityRow]:
    rows: list[CTCapabilityRow] = []
    symmetric = "k2x2-s2x2-p0-n1h2w4c64"
    asymmetric = "k3x2-s2x1-p0-n1h3w5c64"
    padded = "k3x2-s2x1-asymmetric-pad"
    for opcode in range(115, 121):
        for dtype in DTYPES:
            evidence = _POOL_SYMMETRIC_EVIDENCE.get((opcode, dtype))
            if evidence is not None:
                rows.append(
                    _positive(
                        opcode,
                        "pool",
                        dtype,
                        "HW-window/NCx",
                        symmetric,
                        evidence,
                    )
                )
            elif opcode == 118 and dtype == "f16":
                rows.append(
                    _observation(
                        opcode,
                        "pool",
                        dtype,
                        "HW-window/NCx",
                        symmetric,
                        "unpool-f16",
                        "indexed-max only ran as an unpool predecessor; its "
                        "value/index outputs were not independently checked",
                    )
                )
            else:
                rows.append(
                    _positive(
                        opcode,
                        "pool",
                        dtype,
                        "HW-window/NCx",
                        symmetric,
                        f"{OPCODE_NAMES[opcode]}-{dtype}-k2x2-s2x2",
                        qualification="pending-board",
                    )
                )

        if opcode == 118:
            rows.append(
                _positive(
                    opcode,
                    "pool",
                    "f16",
                    "HW-window/NCx",
                    asymmetric,
                    "pool-indexed-max-f16-k3x2-s2x1",
                    qualification="pending-board",
                )
            )
        else:
            rows.append(
                _positive(
                    opcode,
                    "pool",
                    "f16",
                    "HW-window/NCx",
                    asymmetric,
                    f"{OPCODE_NAMES[opcode]}-f16-k3x2-s2x1",
                    qualification="pending-board",
                )
            )
        rows.append(
            _observation(
                opcode,
                "pool",
                "f16",
                "HW-window/NCx",
                padded,
                f"{OPCODE_NAMES[opcode]}-f16-k3x2-s2x1-padded-observed",
                "padding changes the numeric boundary convention; the case "
                "therefore checks packet completion and bounded writeback "
                "without publishing a numeric capability",
                qualification="pending-board",
            )
        )
        rows.append(
            _negative(
                opcode,
                "pool",
                "any",
                "HW-window/Cx",
                "any",
                "native Pool has no layout operand; Cx must be explicitly "
                "materialized to NCx before packet construction",
            )
        )
    for opcode in (118, 120):
        rows.append(
            _observation(
                opcode,
                "pool",
                "f16",
                "HW-window/NCx",
                "indexed-tie",
                f"{OPCODE_NAMES[opcode]}-f16-tie-observed",
                "indexed tie-breaking is a distinct numeric contract; the "
                "case records the bounded raw value/index result",
                qualification="pending-board",
            )
        )
    return rows


_UNPOOL_EVIDENCE = {
    121: ("unpool-index-f16", "observation"),
    122: ("unpool-avg-f16", "exact"),
    123: ("unpool-f16", "exact"),
}

_UNPOOL_TYPED_EVIDENCE = {
    (121, "bf16"): ("unpool-index-bf16-observed", "observation"),
    (121, "f32"): ("unpool-index-f32-observed", "observation"),
    (122, "bf16"): ("unpool-avg-bf16", "exact"),
    (122, "f32"): ("unpool-avg-f32", "exact"),
    (123, "bf16"): ("unpool-mask-bf16", "exact"),
    (123, "f32"): ("unpool-mask-f32", "observation"),
}

_UNPOOL_ASYMMETRIC_EVIDENCE = {
    121: ("unpool-index-f16-k3x2-s2x1-observed", "observation"),
    122: ("unpool-avg-f16-k3x2-s2x1-observed", "observation"),
    123: ("unpool-mask-f16-k3x2-s2x1", "observation"),
}

_UNPOOL_COLLISION_EVIDENCE = {
    121: "unpool-index-f16-repeated-overlap-observed",
    123: "unpool-mask-f16-repeated-overlap-observed",
}


def _unpool_rows() -> list[CTCapabilityRow]:
    rows: list[CTCapabilityRow] = []
    symmetric = "k2x2-s2x2-n1h1w1c64-to-n1h2w2c64"
    asymmetric = "k3x2-s2x1-n1h2w2c64-to-n1h3w5c64"
    for opcode in range(121, 124):
        axis_layout = (
            "i16-spm-index/NCx" if opcode in {121, 123} else "no-index/NCx"
        )
        for dtype in DTYPES:
            typed_evidence = (
                _UNPOOL_EVIDENCE[opcode]
                if dtype == "f16"
                else _UNPOOL_TYPED_EVIDENCE[(opcode, dtype)]
            )
            evidence, oracle = typed_evidence
            if oracle == "exact":
                rows.append(
                    _positive(
                        opcode,
                        "unpool",
                        dtype,
                        axis_layout,
                        symmetric,
                        evidence,
                        qualification=(
                            "board-passed"
                            if dtype == "f16"
                            else "pending-board"
                        ),
                    )
                )
            else:
                board_observed = (opcode, dtype) in {
                    (121, "f16"),
                    (123, "f32"),
                }
                rows.append(
                    _observation(
                        opcode,
                        "unpool",
                        dtype,
                        axis_layout,
                        symmetric,
                        evidence,
                        (
                            "mask scatter completed with bounded writes, but "
                            "the f32 numeric layout differs from the host "
                            "semantic reference"
                            if (opcode, dtype) == (123, "f32")
                            else
                            "indexed scatter completed with bounded writes; "
                            "the numeric contract remains an observation "
                            "until the new typed vector is executed on board"
                        ),
                        qualification=(
                            "board-observed"
                            if board_observed
                            else "pending-board"
                        ),
                    )
                )
        evidence, oracle = _UNPOOL_ASYMMETRIC_EVIDENCE[opcode]
        if oracle == "exact":
            rows.append(
                _positive(
                    opcode,
                    "unpool",
                    "f16",
                    axis_layout,
                    asymmetric,
                    evidence,
                    qualification="pending-board",
                )
            )
        else:
            rows.append(
                _observation(
                    opcode,
                    "unpool",
                    "f16",
                    axis_layout,
                    asymmetric,
                    evidence,
                    (
                        "asymmetric mask scatter completed with bounded "
                        "writes, but only the final 64-channel source window "
                        "matched the semantic scatter reference"
                        if opcode == 123
                        else
                        "asymmetric X/Y geometry has a bounded board vector "
                        "but its numeric result remains an observation"
                    ),
                    qualification=(
                        "board-observed"
                        if opcode == 123
                        else "pending-board"
                    ),
                )
            )
        rows.append(
            _negative(
                opcode,
                "unpool",
                "any",
                axis_layout.replace("NCx", "Cx"),
                "any",
                "native Unpool has no layout operand; Cx must be explicitly "
                "materialized to NCx before packet construction",
            )
        )
    for opcode in (121, 123):
        rows.append(
            _negative(
                opcode,
                "unpool",
                "any",
                "scalar-inline-index/NCx",
                "any",
                "the terminal IR requires an i16 SPM index memref; scalar "
                "index attrs are rejected before target lowering",
            )
        )
        rows.append(
            _observation(
                opcode,
                "unpool",
                "f16",
                "i16-spm-index/NCx",
                "repeated-or-overlapping-index",
                _UNPOOL_COLLISION_EVIDENCE[opcode],
                "repeated destinations deliberately exercise collision "
                "behavior; preserve bounded raw output until board evidence "
                "selects a deterministic numeric contract",
                qualification="pending-board",
            )
        )
    return rows


CATALOG = tuple(_reduce_rows() + _pool_rows() + _unpool_rows())
BY_KEY = {row.key: row for row in CATALOG}
ROWS_BY_OPCODE = {
    opcode: tuple(row for row in CATALOG if row.opcode == opcode)
    for opcode in range(111, 124)
}

BOARD_POSITIVE_ROWS = tuple(
    row for row in CATALOG if row.disposition == "board-executable"
)
BOARD_OBSERVATION_ROWS = tuple(
    row for row in CATALOG if row.disposition == "board-observation"
)
STATIC_NEGATIVE_ROWS = tuple(
    row for row in CATALOG if row.disposition == "static-negative"
)
ISOLATED_DEFERRED_ROWS = tuple(
    row for row in CATALOG if row.disposition == "isolated-deferred"
)
UNKNOWN_ROWS = tuple(
    row for row in CATALOG if row.disposition == "unknown"
)
