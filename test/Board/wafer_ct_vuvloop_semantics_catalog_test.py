#!/usr/bin/env python3
"""Validate supported VuVLoop controls and host-only negatives."""

from __future__ import annotations

import pathlib
import struct

import wafer_ct_vector_calibration_catalog as vector_catalog
import wafer_ct_vuvloop_semantics_catalog as catalog


def _decode_f16(raw: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(raw) // 2}e", raw)


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    protocol = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_ct_vuvloop_semantics_protocol.h"
    ).read_text()
    probe = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_ct_vuvloop_semantics_probe.c"
    ).read_text()

    constants = {
        "REQUEST_MAGIC": f"UINT64_C(0x{catalog.REQUEST_MAGIC:016x})",
        "RECORD_MAGIC": f"UINT64_C(0x{catalog.RECORD_MAGIC:016x})",
        "REQUEST_WORDS": f"{catalog.REQUEST_WORDS}U",
        "RECORD_WORDS": f"{catalog.RECORD_WORDS}U",
        "CASE_BASE": f"{catalog.CASE_BASE}U",
        "OPCODE": f"{catalog.OPCODE}U",
        "DTYPE": f"{catalog.DTYPE}U",
        "ELEMENT_BYTES": f"{catalog.ELEMENT_BYTES}U",
        "PHYSICAL_BLOCK_BYTES": f"{catalog.PHYSICAL_BLOCK_BYTES}U",
        "RESOURCE_BYTES": f"{catalog.RESOURCE_BYTES}U",
        "LHS_OWNED_BYTES": f"{catalog.LHS_OWNED_BYTES}U",
        "RHS_OWNED_BYTES": f"{catalog.RHS_OWNED_BYTES}U",
        "OUTPUT_CAPTURE_BYTES": f"{catalog.OUTPUT_CAPTURE_BYTES}U",
        "GUARD_BYTES": f"{catalog.GUARD_BYTES}U",
        "GUARD_BYTE": f"UINT8_C(0x{catalog.GUARD_BYTE:02x})",
        "REQUEST_PADDING_BYTE": (
            f"UINT8_C(0x{catalog.REQUEST_PADDING_BYTE:02x})"
        ),
        "REQUEST_GUARD": f"UINT64_C(0x{catalog.REQUEST_GUARD:016x})",
        "RECORD_GUARD": f"UINT64_C(0x{catalog.RECORD_GUARD:016x})",
    }
    for name, value in constants.items():
        assert f"#define WAFER_CTVL_{name} {value}" in protocol
    for field, index in catalog.REQ.items():
        assert f"WAFER_CTVL_REQ_{field} = {index}" in protocol
    for field, index in catalog.REC.items():
        assert f"WAFER_CTVL_REC_{field} = {index}" in protocol

    assert "host-only verifier negatives" in protocol
    assert "get_spm_memory_mapping" not in probe
    assert "TsmWaitfinish" not in probe
    assert probe.count("wafer_tx81_rdma(") == 3
    assert probe.count("wafer_tx81_wdma(") == 1
    assert probe.count("wafer_tx81_ncc_join(1U);") == 1
    assert "selected->unit_elem_count != 64U" in probe
    assert (
        "(uint64_t)selected->full_elem_count *" in probe
        and "(uint64_t)selected->elem_count *" in probe
    )

    reference = catalog.EXISTING_RECTANGULAR_REFERENCE
    existing = vector_catalog.CASES_BY_NAME[reference.case_name]
    assert (
        existing.elem_count,
        existing.unit_elements,
        existing.full_elements,
        existing.full_unit_elements,
    ) == (
        reference.geometry.elem_count,
        reference.geometry.unit_elem_count,
        reference.geometry.full_elem_count,
        reference.geometry.full_unit_elem_count,
    )

    assert len(catalog.CATALOG) == 2
    assert len(catalog.CASES_BY_NAME) == 2
    assert all(case.exact_legality for case in catalog.CATALOG)
    assert all(
        case.geometry.satisfies_existing_catalog_gate
        for case in catalog.CATALOG
    )
    assert all(
        case.geometry.unit_elem_count == 64 for case in catalog.CATALOG
    )
    assert all(
        case.geometry.full_elem_count
        * case.geometry.unit_elem_count
        == case.geometry.elem_count
        * case.geometry.full_unit_elem_count
        for case in catalog.CATALOG
    )
    assert set(catalog.STATIC_NEGATIVE_GEOMETRIES) == {
        "unit-not-64",
        "ratio-mismatch",
        "elem-unit-remainder",
    }
    assert all(
        not geometry.satisfies_existing_catalog_gate
        for geometry in catalog.STATIC_NEGATIVE_GEOMETRIES.values()
    )

    for sample, case in enumerate(catalog.CATALOG):
        built = catalog.build_observation_payload(case, sample)
        geometry = case.geometry
        assert built.expected_result == case.expected_result
        assert built.expected_result is not None
        assert len(built.request_resource) == catalog.RESOURCE_BYTES
        assert len(built.payload_resource) == catalog.RESOURCE_BYTES
        words = struct.unpack(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert words[catalog.REQ["CASE"]] == case.case_id
        assert words[catalog.REQ["ELEM_COUNT"]] == geometry.elem_count
        assert (
            words[catalog.REQ["UNIT_ELEM_COUNT"]]
            == geometry.unit_elem_count
        )
        assert (
            words[catalog.REQ["FULL_ELEM_COUNT"]]
            == geometry.full_elem_count
        )
        assert (
            words[catalog.REQ["FULL_UNIT_ELEM_COUNT"]]
            == geometry.full_unit_elem_count
        )

        lhs = _decode_f16(built.lhs.body)
        rhs = _decode_f16(built.rhs.body)
        expected = _decode_f16(built.expected_result)
        assert len(expected) == geometry.full_elem_count
        for index, value in enumerate(expected):
            outer = index // geometry.elem_count
            rhs_index = (
                outer * geometry.unit_elem_count
                + index % geometry.elem_count
                % geometry.unit_elem_count
            )
            assert value == lhs[index] + rhs[rhs_index]
        guard = bytes([catalog.GUARD_BYTE]) * catalog.GUARD_BYTES
        for region in (built.lhs, built.rhs, built.output):
            assert region.prefix_guard == guard
            assert region.suffix_guard == guard

    print(
        "wafer_ct_vuvloop_semantics_catalog_test: "
        "board_controls=2 host_negatives=3 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
