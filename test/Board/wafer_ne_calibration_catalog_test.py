#!/usr/bin/env python3
"""Validate all prepared large/tail/batch NE calibration payloads."""

from __future__ import annotations

import pathlib
import struct
import tempfile

import wafer_board_ne_calibration_probe_test as runner
import wafer_ne_calibration_catalog as catalog
import wafer_physical_tensor_codec as physical


def main() -> int:
    assert len(catalog.SAFE_CASES) == 67
    assert len(catalog.CATALOG) == 70
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
        for dtype in catalog.FLOAT_DTYPES
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
        assert words[catalog.REQ["BATCH_PAIR"]] == (
            (case.lhs_batch << 32) | case.rhs_batch
        )
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
                case.element_bytes,
                built.expected_physical,
            )
            assert unpacked == built.expected_logical
        else:
            assert built.expected_logical is None
            assert built.expected_physical is None

    tail = catalog.CASES_BY_NAME["ne-f16-tail-nn"]
    tail_built = catalog.build_case_payload(tail)
    assert tail_built.expected_physical is not None
    tail_layout = physical.physical_layout(
        tail.output_shape,
        tail.output_layout,
        tail.element_bytes,
    )
    logical_output_bytes = set()
    for coordinate in physical.coordinates(tail.output_shape):
        begin = (
            physical.physical_element_offset(tail_layout, coordinate)
            * tail.element_bytes
        )
        logical_output_bytes.update(
            range(begin, begin + tail.element_bytes)
        )
    padding_output_bytes = set(range(tail.output_span)) - logical_output_bytes
    assert padding_output_bytes
    assert {
        tail_built.expected_physical[index]
        for index in padding_output_bytes
    } == {catalog.OUTPUT_PADDING}
    output_seed = tail_built.payload[
        2 * catalog.SLOT_BYTES
        + catalog.BODY_OFFSET :
        2 * catalog.SLOT_BYTES
        + catalog.BODY_OFFSET
        + tail.output_span
    ]
    assert {
        output_seed[index] for index in padding_output_bytes
    } == {catalog.SLOT_CANARY}

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
    bias_cases = tuple(
        case
        for case in catalog.SAFE_CASES
        if case.option_name == "BIAS"
    )
    assert {
        (case.kind_name, case.dtype_name) for case in bias_cases
    } == {
        (kind, dtype)
        for kind in ("GEMM", "CONV")
        for dtype in catalog.FLOAT_DTYPES
    }
    for case in bias_cases:
        assert case.disposition_name == "BOARD_OBSERVED"
        assert "base output unchanged" in case.reason
        assert "bias_en" in case.reason
        bias_built = catalog.build_case_payload(case)
        assert bias_built.expected_logical is None
        assert bias_built.expected_physical is None
    bias = catalog.CASES_BY_NAME["ne-f16-large-nn-bias"]
    bias_lhs, bias_rhs = catalog._gemm_inputs(bias)
    bias_base = catalog._gemm_expected_values(
        bias, bias_lhs, bias_rhs
    )
    bias_auxiliary, _, _ = catalog._option_auxiliary(bias)
    bias_additive = tuple(
        value + bias_auxiliary[index % bias.n]
        for index, value in enumerate(bias_base)
    )
    assert tuple(
        catalog._encode(bias.dtype_name, value) for value in bias_base
    ) != tuple(
        catalog._encode(bias.dtype_name, value)
        for value in bias_additive
    )
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

    quant = catalog.CASES_BY_NAME["ne-gemm-quant-observed"]
    assert quant.dtype_name == "I8"
    assert quant.element_bytes == 1
    assert quant.disposition_name == "BOARD_OBSERVED"
    assert (quant.lhs_span, quant.rhs_span, quant.output_span) == (
        256,
        256,
        256,
    )
    quant_built = catalog.build_case_payload(quant)
    quant_lhs = physical.unpack_scalar_bytes(
        quant.lhs_shape,
        quant.lhs_layout,
        quant.element_bytes,
        quant_built.payload[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + quant.lhs_span
        ],
    )
    quant_rhs = physical.unpack_scalar_bytes(
        quant.rhs_shape,
        quant.rhs_layout,
        quant.element_bytes,
        quant_built.payload[
            catalog.SLOT_BYTES + catalog.BODY_OFFSET :
            catalog.SLOT_BYTES + catalog.BODY_OFFSET + quant.rhs_span
        ],
    )
    assert {-7, 0, 7}.issubset(
        {struct.unpack("<b", value)[0] for value in quant_lhs}
    )
    assert {-6, 0, 6}.issubset(
        {struct.unpack("<b", value)[0] for value in quant_rhs}
    )
    assert quant_built.expected_logical is None
    assert quant_built.expected_physical is None
    raw = bytearray(
        [runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES
    )
    record = [0] * catalog.RECORD_WORDS
    record_values = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": quant.case_id,
        "DTYPE": quant.dtype,
        "LHS_ORIENTATION": quant.lhs_orientation,
        "RHS_ORIENTATION": quant.rhs_orientation,
        "BATCH": quant.batch,
        "M": quant.m,
        "K": quant.k,
        "N": quant.n,
        "LHS_SPAN": quant.lhs_span,
        "RHS_SPAN": quant.rhs_span,
        "OUTPUT_SPAN": quant.output_span,
        "SAMPLE": 4,
        "EXECUTE_RESULT": 1,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "KIND": quant.kind,
        "PROFILE": quant.profile,
        "OPTION": quant.option,
        "AUX_SPAN": quant.aux_span,
        "DISPOSITION": quant.disposition,
        "LHS_BATCH": quant.lhs_batch,
        "RHS_BATCH": quant.rhs_batch,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    for name, value in record_values.items():
        record[catalog.REC[name]] = value
    struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *record)
    output_seed = quant_built.payload[
        2 * catalog.SLOT_BYTES :
        3 * catalog.SLOT_BYTES
    ]
    begin = catalog.OUTPUT_DDR_OFFSET
    raw[begin : begin + catalog.SLOT_BYTES] = output_seed
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "quant-observed.raw"
        output.write_bytes(raw)
        try:
            runner.validate_output(output, quant, quant_built, sample=4)
        except RuntimeError as error:
            assert "without any bounded writeback" in str(error)
        else:
            raise AssertionError("unchanged NE observation seed was accepted")
        raw[begin + catalog.BODY_OFFSET + 3] ^= 0x1
        output.write_bytes(raw)
        runner.validate_output(output, quant, quant_built, sample=4)

    observed_conv = catalog.CALIBRATION_LEAF_BINDINGS[
        "ne-depthwise-backward-conv-observed"
    ]
    assert {case.kind_name for case in observed_conv} == {
        "DEPTHWISE_CONV",
        "BACKWARD_CONV",
    }
    assert {case.dtype_name for case in observed_conv} == {
        "F16",
        "BF16",
    }
    assert all(
        case.disposition_name == "BOARD_OBSERVED"
        and case.output_shape == (1, 4, 4, 64)
        and case.output_span == 2048
        for case in observed_conv
    )
    probe = (
        pathlib.Path(__file__).resolve().parent
        / "Inputs"
        / "wafer_ne_calibration_probe.c"
    ).read_text()
    for function, op_type in (
        ("wafer_nec_issue_depthwise", "1U"),
        ("wafer_nec_issue_backward", "2U"),
    ):
        begin = probe.index(f"{function}(")
        body = probe[begin : probe.index("static ", begin + 16)]
        assert body.index(f"SetOpType(&instruction, {op_type})") < body.index(
            "AddWeight(&instruction"
        )

    unequal = catalog.CALIBRATION_LEAF_BINDINGS[
        "ne-batch-broadcast-positive"
    ]
    assert len(unequal) == 4
    assert {
        (case.dtype_name, case.lhs_batch, case.rhs_batch)
        for case in unequal
    } == {
        (dtype, lhs_batch, rhs_batch)
        for dtype in catalog.FLOAT_DTYPES
        for lhs_batch, rhs_batch in ((1, 2), (2, 1))
    }
    for case in unequal:
        assert case.exact
        lhs, rhs = catalog._gemm_inputs(case)
        assert len(lhs) == case.lhs_batch * case.m * case.k
        assert len(rhs) == case.rhs_batch * case.k * case.n
        expected = catalog._gemm_expected_values(case, lhs, rhs)
        batch_elements = case.m * case.n
        assert expected[:batch_elements] != expected[batch_elements:]
        built = catalog.build_case_payload(case)
        assert built.expected_logical == tuple(
            catalog._encode(case.dtype_name, value) for value in expected
        )

    nonexecuting = tuple(
        case for case in catalog.CATALOG if not case.is_safe
    )
    assert len(nonexecuting) == 3
    assert all(case.reason for case in nonexecuting)
    assert all(
        case.disposition_name == "STATIC_NEGATIVE"
        for case in nonexecuting
    )
    assert catalog.CASES_BY_NAME[
        "ne-gemm-sparse-static-unsupported"
    ].reason.startswith("TsmGemm has no SetSparse")

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
