#!/usr/bin/env python3
"""One auditable disposition for every public CT opcode 0..186."""

from __future__ import annotations

import dataclasses

import wafer_ct_convert_calibration_catalog as convert_catalog
import wafer_ct_vector_calibration_catalog as vector_catalog
import wafer_datamove_calibration_catalog as datamove_catalog


@dataclasses.dataclass(frozen=True)
class CTOpcodeDisposition:
    opcode: int
    opcode_name: str
    family: str
    disposition: str
    evidence: tuple[str, ...]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _board(
    opcode: int, family: str, *evidence: str
) -> CTOpcodeDisposition:
    if not evidence:
        raise RuntimeError(f"opcode {opcode}: board row requires evidence")
    return CTOpcodeDisposition(
        opcode,
        vector_catalog.OPCODE_NAMES[opcode],
        family,
        "board-executable",
        tuple(evidence),
    )


def _observation(
    opcode: int, family: str, *evidence: str
) -> CTOpcodeDisposition:
    if not evidence:
        raise RuntimeError(
            f"opcode {opcode}: board observation requires evidence"
        )
    return CTOpcodeDisposition(
        opcode,
        vector_catalog.OPCODE_NAMES[opcode],
        family,
        "board-observation",
        tuple(evidence),
    )


def _negative(
    opcode: int, family: str, reason: str
) -> CTOpcodeDisposition:
    return CTOpcodeDisposition(
        opcode,
        vector_catalog.OPCODE_NAMES[opcode],
        family,
        "static-negative",
        ("wafer-ct-opcode-disposition-catalog-python",),
        reason,
    )


_REDUCE_POOL = {
    111: _board(111, "reduce", "reduce-sum-f16"),
    112: _board(112, "reduce", "reduce-avg-bf16"),
    113: _board(113, "reduce", "reduce-max-f16"),
    114: _board(114, "reduce", "reduce-min-bf16"),
    115: _board(115, "pool", "pool-avg-f16"),
    116: _board(116, "pool", "pool-sum-f16"),
    117: _board(117, "pool", "pool-f16", "pool-bf16"),
    118: _board(118, "pool", "unpool-f16"),
    119: _board(119, "pool", "pool-min-f16"),
    120: _board(120, "pool", "pool-indexed-min-f16"),
}

_PERIPHERAL = {
    175: _negative(
        175,
        "peripheral",
        "Count scalar writeback is not represented by the current typed IR/CRT contract",
    ),
    176: _negative(
        176,
        "peripheral",
        "BitCount scalar writeback is not represented by the current typed IR/CRT contract",
    ),
    177: _board(177, "peripheral", "peripheral-argmax-f16"),
    178: _board(178, "peripheral", "peripheral-argmin-f16"),
    179: _board(
        179,
        "peripheral",
        "tdma-crt-i8-physical16",
        "tdma-crt-bool-to-i8-physical17",
    ),
    180: _observation(
        180, "peripheral", "peripheral-factorize-f32-observed"
    ),
    181: _board(
        181,
        "peripheral",
        "select-bit2fp-maskmove-f16",
        "select-bit2fp-maskmove-bf16",
    ),
    182: _board(182, "peripheral", "peripheral-bilinear-f16"),
    183: _board(183, "peripheral", "peripheral-lut16-f16"),
    184: _observation(184, "peripheral", "peripheral-lut32-observed"),
    185: _observation(
        185, "peripheral", "peripheral-randgen-f16-observed"
    ),
    186: _observation(
        186, "peripheral", "peripheral-elemmask-f16-observed"
    ),
}


def build_catalog() -> tuple[CTOpcodeDisposition, ...]:
    rows: list[CTOpcodeDisposition] = []
    vector_evidence = (
        "wafer-ct-vector-calibration-catalog-python",
        "wafer-runtime-ct-vector-calibration-probe-no-card",
    )
    for opcode in range(111):
        rows.append(
            _board(opcode, vector_catalog._family(opcode).lower(), *vector_evidence)
        )
    rows.extend(_REDUCE_POOL[opcode] for opcode in range(111, 121))

    datamove_by_opcode = {
        row.opcode: row for row in datamove_catalog.PUBLIC_DISPOSITIONS
    }
    for opcode in range(121, 139):
        movement = datamove_by_opcode[opcode]
        rows.append(
            CTOpcodeDisposition(
                opcode,
                vector_catalog.OPCODE_NAMES[opcode],
                "datamove",
                movement.disposition,
                movement.evidence,
                movement.reason,
            )
        )

    convert_evidence_by_opcode = {
        opcode: tuple(
            case.name
            for case in convert_catalog.SAFE_CASES
            if case.opcode == opcode
        )
        for opcode in range(139, 175)
    }
    for opcode in range(139, 175):
        rows.append(
            _board(
                opcode,
                "convert",
                *convert_evidence_by_opcode[opcode],
            )
        )
    rows.extend(_PERIPHERAL[opcode] for opcode in range(175, 187))
    return tuple(rows)


CATALOG = build_catalog()
BY_OPCODE = {row.opcode: row for row in CATALOG}
_REDUCE_POOL_UNPOOL_PERIPHERAL_OPCODES = frozenset(
    range(111, 124)
) | frozenset(range(175, 187))
CALIBRATION_LEAF_BINDINGS: dict[str, tuple[object, ...]] = {
    "ct-reduce-pool-unpool-peripheral-positive": tuple(
        row
        for row in CATALOG
        if row.opcode in _REDUCE_POOL_UNPOOL_PERIPHERAL_OPCODES
        and row.disposition == "board-executable"
    ),
    "ct-reduce-pool-unpool-peripheral-observed": tuple(
        row
        for row in CATALOG
        if row.opcode in _REDUCE_POOL_UNPOOL_PERIPHERAL_OPCODES
        and row.disposition == "board-observation"
    ),
    "ct-reduce-pool-unpool-peripheral-static-negative": tuple(
        row
        for row in CATALOG
        if row.opcode in _REDUCE_POOL_UNPOOL_PERIPHERAL_OPCODES
        and row.disposition == "static-negative"
    ),
}
