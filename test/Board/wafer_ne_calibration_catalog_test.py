#!/usr/bin/env python3
"""Validate all prepared large/tail/batch NE calibration payloads."""

from __future__ import annotations

import struct

import wafer_ne_calibration_catalog as catalog
import wafer_physical_tensor_codec as physical


def main() -> int:
    assert len(catalog.CATALOG) == 24
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    coverage = {
        (
            case.dtype_name,
            case.geometry_name,
            case.orientation_name,
        )
        for case in catalog.CATALOG
    }
    assert coverage == {
        (dtype, geometry, orientation)
        for dtype in catalog.DTYPES
        for geometry in catalog.GEOMETRIES
        for orientation in catalog.ORIENTATIONS
    }
    for sample, case in enumerate(catalog.CATALOG):
        built = catalog.build_case_payload(case, sample)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert words[catalog.REQ["CASE"]] == case.case_id
        assert words[catalog.REQ["SAMPLE"]] == sample
        assert words[catalog.REQ["GUARD"]] == catalog.REQUEST_GUARD
        assert max(case.lhs_span, case.rhs_span, case.output_span) <= (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET
        )
        unpacked = physical.unpack_scalar_bytes(
            case.output_shape,
            case.layout,
            2,
            built.expected_physical,
        )
        assert unpacked == built.expected_logical
        assert any(
            value != b"\x00\x00" for value in built.expected_logical
        )
    print(
        "wafer_ne_calibration_catalog_test: "
        f"cases={len(catalog.CATALOG)} "
        f"max_span={max(case.rhs_span for case in catalog.CATALOG)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
