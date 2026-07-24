#!/usr/bin/env python3
"""Validate all prepared large/tail/batch NE calibration payloads."""

from __future__ import annotations

import struct

import wafer_ne_calibration_catalog as catalog
import wafer_physical_tensor_codec as physical


def main() -> int:
    assert len(catalog.SAFE_CASES) == 58
    assert len(catalog.CATALOG) == 67
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    coverage = {
        (
            case.dtype_name,
            case.geometry_name,
            case.orientation_name,
        )
        for case in catalog.SAFE_CASES
        if case.kind_name == "GEMM"
        and case.profile_name == "DENSE"
        and case.option_name == "NONE"
    }
    assert coverage == {
        (dtype, geometry, orientation)
        for dtype in catalog.DTYPES
        for geometry in catalog.GEOMETRIES
        for orientation in catalog.ORIENTATIONS
    }
    for sample, case in enumerate(catalog.SAFE_CASES):
        built = catalog.build_case_payload(case, sample)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert words[catalog.REQ["CASE"]] == case.case_id
        assert words[catalog.REQ["SAMPLE"]] == sample
        assert words[catalog.REQ["KIND"]] == case.kind
        assert words[catalog.REQ["PROFILE"]] == case.profile
        assert words[catalog.REQ["OPTION"]] == case.option
        assert words[catalog.REQ["AUX_SPAN"]] == case.aux_span
        assert words[catalog.REQ["DISPOSITION"]] == case.disposition
        assert words[catalog.REQ["GUARD"]] == catalog.REQUEST_GUARD
        assert max(case.lhs_span, case.rhs_span, case.output_span) <= (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET
        )
        if case.exact:
            assert built.expected_logical is not None
            assert built.expected_physical is not None
            unpacked = physical.unpack_scalar_bytes(
                case.output_shape,
                case.output_layout,
                2,
                built.expected_physical,
            )
            assert unpacked == built.expected_logical
        else:
            assert built.expected_logical is None
            assert built.expected_physical is None

    for case in catalog.CALIBRATION_LEAF_BINDINGS[
        "ne-gemm-dtype-orientation-main-tail-batch"
    ]:
        lhs, rhs = catalog._gemm_inputs(case)
        assert all(value != 0.0 for value in lhs)
        assert all(value != 0.0 for value in rhs)
        expected = catalog._gemm_expected_values(case, lhs, rhs)
        assert len(expected) == case.batch * case.m * case.n
        for batch in range(case.batch):
            rows = tuple(
                expected[
                    (batch * case.m + row) * case.n :
                    (batch * case.m + row + 1) * case.n
                ]
                for row in range(case.m)
            )
            column_signatures = {
                tuple(rows[row][column] for row in range(case.m))
                for column in range(case.n)
            }
            assert len(column_signatures) == case.n
            assert all(
                len(set(row)) >= min(8, case.n) for row in rows
            )
        if case.batch > 1:
            batch_elements = case.m * case.n
            assert (
                expected[:batch_elements]
                != expected[batch_elements : 2 * batch_elements]
            )
        if (
            case.dtype_name == "BF16"
            and case.geometry_name == "tail"
            and case.orientation_name == "NN"
        ):
            expected_logical = tuple(
                catalog._encode(case.dtype_name, value)
                for value in expected
            )
            permuted_logical = tuple(
                expected_logical[
                    (batch * case.m + row) * case.n
                    + (column + 1) % case.n
                ]
                for batch in range(case.batch)
                for row in range(case.m)
                for column in range(case.n)
            )
            assert physical.pack_scalar_bytes(
                case.output_shape,
                case.output_layout,
                2,
                permuted_logical,
                padding=catalog.SLOT_CANARY,
            ) != physical.pack_scalar_bytes(
                case.output_shape,
                case.output_layout,
                2,
                expected_logical,
                padding=catalog.SLOT_CANARY,
            )

    assert {
        case.profile_name
        for case in catalog.SAFE_CASES
        if case.profile_name.startswith("BF16_")
    } == {
        "BF16_CANCELLATION",
        "BF16_ROUNDING",
        "BF16_SIGNED_ZERO",
        "BF16_SUBNORMAL",
        "BF16_OVERFLOW_INF",
        "BF16_NAN",
    }
    special_cases = {
        case.profile_name: case
        for case in catalog.SAFE_CASES
        if case.profile_name.startswith("BF16_")
    }
    assert {
        name
        for name, case in special_cases.items()
        if case.disposition_name == "BOARD_OBSERVED"
    } == {
        "BF16_SIGNED_ZERO",
        "BF16_SUBNORMAL",
        "BF16_OVERFLOW_INF",
        "BF16_NAN",
    }

    def lhs_bits(profile_name: str) -> set[int]:
        case = special_cases[profile_name]
        built = catalog.build_case_payload(case)
        physical_lhs = built.payload[
            catalog.BODY_OFFSET : catalog.BODY_OFFSET + case.lhs_span
        ]
        logical_lhs = physical.unpack_scalar_bytes(
            case.lhs_shape,
            case.lhs_layout,
            2,
            physical_lhs,
        )
        return {
            struct.unpack("<H", scalar)[0] for scalar in logical_lhs
        }

    assert {0x0000, 0x8000, 0x3F80}.issubset(
        lhs_bits("BF16_SIGNED_ZERO")
    )
    assert {0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080}.issubset(
        lhs_bits("BF16_SUBNORMAL")
    )
    assert {0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x4000, 0xC000}.issubset(
        lhs_bits("BF16_OVERFLOW_INF")
    )
    assert set(catalog.BF16_NAN_INPUT_BITS).issubset(
        lhs_bits("BF16_NAN")
    )
    assert 0x3F80 in lhs_bits("BF16_NAN")
    assert {
        case.option_name
        for case in catalog.SAFE_CASES
        if case.option_name != "NONE"
    } == set(tuple(catalog.OPTIONS)[1:])
    conv_cases = tuple(
        case for case in catalog.SAFE_CASES if case.kind_name == "CONV"
    )
    assert len(conv_cases) == 16
    assert {case.n for case in conv_cases} == {65, 96}
    assert all(
        max(case.lhs_span, case.rhs_span, case.output_span)
        <= catalog.SLOT_BYTES - catalog.BODY_OFFSET
        for case in conv_cases
    )
    deferred = tuple(
        case for case in catalog.CATALOG if not case.is_safe
    )
    assert len(deferred) == 9
    assert all(case.reason for case in deferred)

    bound = tuple(
        case
        for cases in catalog.CALIBRATION_LEAF_BINDINGS.values()
        for case in cases
    )
    assert all(catalog.CALIBRATION_LEAF_BINDINGS.values())
    assert len(bound) == len(set(bound)) == len(catalog.CATALOG)
    assert set(bound) == set(catalog.CATALOG)
    max_span = max(
        max(
            case.lhs_span,
            case.rhs_span,
            case.output_span,
            case.aux_span,
        )
        for case in catalog.SAFE_CASES
    )
    assert max_span <= catalog.SLOT_BYTES - catalog.BODY_OFFSET
    print(
        "wafer_ne_calibration_catalog_test: "
        f"cases={len(catalog.CATALOG)} "
        f"safe={len(catalog.SAFE_CASES)} "
        f"max_span={max_span} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
