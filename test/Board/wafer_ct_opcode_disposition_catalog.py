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


def _deferred(
    opcode: int, family: str, reason: str
) -> CTOpcodeDisposition:
    return CTOpcodeDisposition(
        opcode,
        vector_catalog.OPCODE_NAMES[opcode],
        family,
        "isolated-deferred",
        (),
        reason,
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
    115: _deferred(
        115,
        "pool",
        "AvgPool lacks a large-shape exact physical oracle in the current ABI",
    ),
    116: _deferred(
        116,
        "pool",
        "SumPool lacks a large-shape exact physical oracle in the current ABI",
    ),
    117: _board(117, "pool", "pool-f16", "pool-bf16"),
    118: _board(118, "pool", "unpool-f16-index-source"),
    119: _deferred(
        119,
        "pool",
        "MinPool lacks a large-shape exact physical oracle in the current ABI",
    ),
    120: _deferred(
        120,
        "pool",
        "IndexedMinPool value/index dual-write span is not qualified",
    ),
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
    180: _deferred(
        180,
        "peripheral",
        "Factorize has three writeback destinations without a qualified ownership/span contract",
    ),
    181: _board(
        181,
        "peripheral",
        "select-bit2fp-maskmove-f16",
        "select-bit2fp-maskmove-bf16",
    ),
    182: _deferred(
        182,
        "peripheral",
        "Bilinear geometry and physical write span are not exposed by the current typed ABI",
    ),
    183: _board(183, "peripheral", "peripheral-lut16-f16"),
    184: _deferred(
        184,
        "peripheral",
        "LUT32 table/index width and full physical output oracle are not qualified",
    ),
    185: _deferred(
        185,
        "peripheral",
        "RandGen is stateful/random and has no bounded deterministic seed/recovery contract",
    ),
    186: _deferred(
        186,
        "peripheral",
        "ElemMask is probabilistic and has no deterministic seed/recovery contract",
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
            for case in convert_catalog.CATALOG
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
