#!/usr/bin/env python3
"""No-card tests for instruction qualification catalog and host oracles."""

from __future__ import annotations

import pathlib
import re
import struct
import tempfile

import wafer_instruction_family_catalog as catalog
import wafer_board_instruction_family_probe_test as runner


def _encoded_words(
    dtype_name: str, values: tuple[float, ...] | list[float]
) -> tuple[int, ...]:
    if dtype_name == "F16":
        raw = struct.pack(f"<{len(values)}e", *values)
        return struct.unpack(f"<{len(values)}H", raw)
    assert dtype_name == "BF16"
    words = []
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        words.append(((bits + rounding_bias) >> 16) & 0xFFFF)
    return tuple(words)


def validate_rounding_and_bit2fp_oracles() -> None:
    assert "convert-i8-f16-zp0" in catalog.CASES_BY_NAME
    assert "convert-i8-bf16-zp0" in catalog.CASES_BY_NAME
    assert "convert-i8-f16-zp" not in catalog.CASES_BY_NAME
    assert "convert-i8-bf16-zp" not in catalog.CASES_BY_NAME

    bf16_case = catalog.CASES_BY_NAME["convert-f16-bf16-round"]
    bf16_built = catalog.build_case_payload(bf16_case)
    f16_source = struct.unpack_from(
        "<128e", bf16_built.payload, catalog.BODY_OFFSET
    )
    bf16_golden = struct.unpack_from(
        "<128H", bf16_built.expected_output_slot, catalog.BODY_OFFSET
    )
    bf16_rne: list[int] = []
    bf16_truncated: list[int] = []
    for value in f16_source:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bf16_rne.append(
            ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
        )
        bf16_truncated.append((bits >> 16) & 0xFFFF)
    assert tuple(bf16_rne) == bf16_golden
    assert any(
        expected != truncated
        for expected, truncated in zip(
            bf16_golden, bf16_truncated, strict=True
        )
    )

    i16_case = catalog.CASES_BY_NAME["convert-f16-i16-round"]
    i16_built = catalog.build_case_payload(i16_case)
    i16_source = struct.unpack_from(
        "<128e", i16_built.payload, catalog.BODY_OFFSET
    )
    i16_golden = struct.unpack_from(
        "<128h", i16_built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert tuple(round(value) for value in i16_source) == i16_golden
    assert any(
        expected != int(value)
        for expected, value in zip(i16_golden, i16_source, strict=True)
    )

    protocol = (
        pathlib.Path(__file__).resolve().parent
        / "Inputs"
        / "wafer_instruction_family_probe_protocol.h"
    ).read_text()
    for dtype, macro in (
        ("F16", "WAFER_IFP_BIT2FP_TRUE_F16"),
        ("BF16", "WAFER_IFP_BIT2FP_TRUE_BF16"),
    ):
        match = re.search(
            rf"#define {macro} UINT16_C\(0x([0-9a-fA-F]+)\)",
            protocol,
        )
        assert match is not None
        assert int(match.group(1), 16) == catalog.BIT2FP_TRUE_WORDS[dtype]
        case = catalog.CASES_BY_NAME[
            f"select-bit2fp-maskmove-{dtype.lower()}"
        ]
        expected = catalog.bit2fp_expected_words(case)
        assert expected[:8] == (catalog.BIT2FP_TRUE_WORDS[dtype],) * 8
        assert expected[8:16] == (catalog.BIT2FP_FALSE_WORD,) * 8


def validate_ct_add_f16_tail130_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-f16-tail130"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        260,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<130e", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<130e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<130e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(
        left + right for left, right in zip(lhs, rhs, strict=True)
    )
    assert lhs[128:] == (-3.0, -2.0)
    assert rhs[128:] == (1.0, -1.0)
    assert expected[128:] == (-2.0, -3.0)


def validate_ct_add_bf16_tail130_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-bf16-tail130"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        260,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<130H", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<130H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<130H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    lhs_pattern = (-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    rhs_pattern = (1.0, -1.0, 2.0, 2.0, 0.5, -2.0, 4.0, -0.5)
    lhs_values = [lhs_pattern[index % len(lhs_pattern)] for index in range(130)]
    rhs_values = [rhs_pattern[index % len(rhs_pattern)] for index in range(130)]
    expected_values = [
        left + right
        for left, right in zip(lhs_values, rhs_values, strict=True)
    ]
    assert lhs == _encoded_words("BF16", lhs_values)
    assert rhs == _encoded_words("BF16", rhs_values)
    assert expected == _encoded_words("BF16", expected_values)
    assert expected_values[128:] == [-2.0, -3.0]


def validate_ct_add_f32_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-f32"]
    assert case.is_safe
    assert case.dtype_name == "F32"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        512,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128f", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<128f",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<128f", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert all(value.is_integer() for value in lhs + rhs)
    assert expected == tuple(
        left + right for left, right in zip(lhs, rhs, strict=True)
    )


def validate_ct_add_special_f16_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-special-f16"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        256,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<128H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<128H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert lhs[:12] == (
        0x0000,
        0x8000,
        0x0000,
        0x8000,
        0x7C00,
        0xFC00,
        0x7BFF,
        0xFBFF,
        0x0400,
        0x8400,
        0x0001,
        0x8001,
    )
    assert rhs[:4] == (0x0000, 0x0000, 0x8000, 0x8000)
    assert expected[:4] == (0x0000, 0x0000, 0x0000, 0x8000)
    assert expected[4:12] == lhs[4:12]
    assert all((word & 0x7C00) != 0x7C00 or (word & 0x03FF) == 0 for word in lhs)


def validate_ct_add_special_bf16_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-special-bf16"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        256,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<128H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<128H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert lhs[:12] == (
        0x0000,
        0x8000,
        0x0000,
        0x8000,
        0x7F80,
        0xFF80,
        0x7F7F,
        0xFF7F,
        0x0080,
        0x8080,
        0x0001,
        0x8001,
    )
    assert rhs[:4] == (0x0000, 0x0000, 0x8000, 0x8000)
    assert expected[:4] == (0x0000, 0x0000, 0x0000, 0x8000)
    assert expected[4:12] == lhs[4:12]
    assert all((word & 0x7F80) != 0x7F80 or (word & 0x007F) == 0 for word in lhs)


def validate_gemm_padding_domain() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16"]
    built = catalog.build_case_payload(case, sample=3)
    raw = bytearray([runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES)
    words = [0] * catalog.RECORD_WORDS
    rec = catalog.REC
    words[rec["MAGIC"]] = catalog.RECORD_MAGIC
    words[rec["SCHEMA_AND_WORDS"]] = (
        catalog.SCHEMA << 32
    ) | catalog.RECORD_WORDS
    words[rec["STATUS"]] = 0
    words[rec["CASE"]] = case.case_id
    words[rec["DISPOSITION"]] = case.disposition
    words[rec["FAMILY"]] = case.family
    words[rec["DTYPE"]] = case.dtype
    words[rec["ORACLE"]] = case.oracle
    words[rec["RESULT_BYTES"]] = case.result_bytes
    words[rec["OUTPUT_SPAN"]] = case.output_span
    words[rec["AUX_SPAN"]] = case.aux_span
    words[rec["SAMPLE"]] = 3
    words[rec["REQUEST_GUARD"]] = catalog.REQUEST_GUARD
    words[rec["OUTPUT_DDR_OFFSET"]] = catalog.OUTPUT_DDR_OFFSET
    words[rec["SLOT_BYTES"]] = catalog.SLOT_BYTES
    words[rec["BODY_OFFSET"]] = catalog.BODY_OFFSET
    words[rec["STEP_FLAGS"]] = (
        catalog.STEP_TARGET_ISSUED | catalog.STEP_FINAL_FENCE_COMPLETED
    )
    words[rec["RECORD_GUARD"]] = catalog.RECORD_GUARD
    struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *words)
    output_begin = catalog.OUTPUT_DDR_OFFSET
    raw[output_begin : output_begin + catalog.SLOT_BYTES] = (
        built.expected_output_slot
    )

    padding_index = (
        output_begin + catalog.BODY_OFFSET + case.result_bytes
    )
    raw[padding_index] ^= 0x5A
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "gemm.raw"
        output.write_bytes(raw)
        runner.validate_output(
            output, case, built.expected_output_slot, sample=3
        )

        logical = bytearray(raw)
        logical[output_begin + catalog.BODY_OFFSET] ^= 1
        output.write_bytes(logical)
        try:
            runner.validate_output(
                output, case, built.expected_output_slot, sample=3
            )
        except RuntimeError as error:
            assert "logical result" in str(error)
        else:
            raise AssertionError("GEMM logical result mismatch was accepted")

        suffix = bytearray(raw)
        suffix[
            output_begin + catalog.BODY_OFFSET + case.output_span
        ] ^= 1
        output.write_bytes(suffix)
        try:
            runner.validate_output(
                output, case, built.expected_output_slot, sample=3
            )
        except RuntimeError as error:
            assert "SPM guard" in str(error)
        else:
            raise AssertionError("GEMM suffix guard mismatch was accepted")


def validate_gemm_dtype_oracles() -> None:
    values = tuple(float(value) for value in (1, 2, 3, 4) * 4)
    identity = tuple(
        1.0 if row == column else 0.0
        for row in range(16)
        for column in range(16)
    )

    def encode(dtype_name: str, source: tuple[float, ...]) -> bytes:
        if dtype_name == "F16":
            return struct.pack(f"<{len(source)}e", *source)
        assert dtype_name == "BF16"
        words = {
            -13.0: 0xC150,
            0.0: 0x0000,
            1.0: 0x3F80,
            2.0: 0x4000,
            3.0: 0x4040,
            4.0: 0x4080,
        }
        return struct.pack(
            f"<{len(source)}H", *(words[value] for value in source)
        )

    for name, dtype_name in (
        ("ne-gemm-f16", "F16"),
        ("ne-gemm-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 32
        assert case.output_span == 256
        assert case.aux_span == 0

        built = catalog.build_case_payload(case)
        lhs = encode(dtype_name, values)
        lhs_begin = catalog.BODY_OFFSET
        assert built.payload[lhs_begin : lhs_begin + len(lhs)] == lhs
        assert built.payload[lhs_begin + len(lhs) : lhs_begin + 256] == bytes(
            256 - len(lhs)
        )

        rhs = encode(dtype_name, identity)
        rhs_begin = catalog.SLOT_BYTES + catalog.BODY_OFFSET
        assert built.payload[rhs_begin : rhs_begin + len(rhs)] == rhs
        assert (
            built.expected_output_slot[
                catalog.BODY_OFFSET : catalog.BODY_OFFSET + len(lhs)
            ]
            == lhs
        )

        seed = encode(dtype_name, (-13.0,) * 128)
        output_begin = 2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET
        assert (
            built.payload[output_begin : output_begin + len(seed)] == seed
        )


def validate_gemm_bf16_accum_round_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-accum-round"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert case.result_bytes == 32
    assert case.output_span == 256
    assert case.aux_span == 0

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (0x3F80,) * 16
    assert lhs[16:] == (0,) * 112

    rhs = struct.unpack_from(
        "<256H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs[:16] == tuple(0x3F80 + column for column in range(16))
    for row in range(1, 16):
        assert rhs[row * 16 : (row + 1) * 16] == tuple(
            0x3B00 if column % 2 == 0 else 0x3980
            for column in range(16)
        )

    expected = struct.unpack_from(
        "<16H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == (
        0x3F84,
        0x3F81,
        0x3F86,
        0x3F83,
        0x3F88,
        0x3F85,
        0x3F8A,
        0x3F87,
        0x3F8C,
        0x3F89,
        0x3F8E,
        0x3F8B,
        0x3F90,
        0x3F8D,
        0x3F92,
        0x3F8F,
    )


def validate_gemm_f16_accum_round_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-accum-round"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (32, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (0x3C00,) * 16
    assert lhs[16:] == (0,) * 112

    rhs = struct.unpack_from(
        "<256H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs[:16] == tuple(0x3C00 + column for column in range(16))
    for row in range(1, 16):
        assert rhs[row * 16 : (row + 1) * 16] == tuple(
            0x0C00
            if column % 2 == 0
            else 0x0800
            if row <= 8
            else 0x8800
            for column in range(16)
        )

    expected = struct.unpack_from(
        "<16H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(
        0x3C00 + column + (4 if column % 2 == 0 else 0)
        for column in range(16)
    )


def validate_gemm_f16_m4_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-m4"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        128,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    expected_rows = (
        tuple(float(value) for value in range(1, 17)),
        tuple(float(value) for value in range(33, 49)),
        tuple(float(-value) for value in range(65, 81)),
        tuple(float(value) for value in range(97, 113)),
    )
    expected = tuple(value for row in expected_rows for value in row)
    assert lhs[:64] == expected
    assert lhs[64:] == (0.0,) * 64
    assert len(set(expected_rows)) == 4

    rhs = struct.unpack_from(
        "<256e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == tuple(
        1.0 if row == column else 0.0
        for row in range(16)
        for column in range(16)
    )
    actual_expected = struct.unpack_from(
        "<64e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_f16_batch2_m8_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-batch2-m8"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        512,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<256e", built.payload, catalog.BODY_OFFSET)
    assert lhs == tuple(float(value) for value in range(1, 257))

    rhs = struct.unpack_from(
        "<512e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    identity = tuple(
        1.0 if row == column else 0.0
        for row in range(16)
        for column in range(16)
    )
    negative_identity = tuple(-value for value in identity)
    assert rhs == identity + negative_identity

    expected = struct.unpack_from(
        "<256e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(value) for value in range(1, 129)) + tuple(
        float(-value) for value in range(129, 257)
    )


def validate_gemm_bf16_batch2_m8_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-batch2-m8"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        512,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<256H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words(
        "BF16", [float(value) for value in range(1, 257)]
    )

    rhs = struct.unpack_from(
        "<512H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    rhs_values = [
        float(
            1
            if batch == 0 and row == column
            else -1
            if batch == 1 and row == column
            else 0
        )
        for batch in range(2)
        for row in range(16)
        for column in range(16)
    ]
    assert rhs == _encoded_words("BF16", rhs_values)

    expected_values = [float(value) for value in range(1, 129)] + [
        float(-value) for value in range(129, 257)
    ]
    expected = struct.unpack_from(
        "<256H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words("BF16", expected_values)


def validate_gemm_f16_n17_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-n17"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (34, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (1.0,) * 16
    assert lhs[16:] == (0.0,) * 112

    rhs = struct.unpack_from(
        "<512e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    for row in range(16):
        physical_row = rhs[row * 32 : (row + 1) * 32]
        assert physical_row[17:] == (-29.0,) * 15
        assert tuple(
            column
            for column, value in enumerate(physical_row[:17])
            if value != 0.0
        ) == tuple(
            column for column in range(17) if row == column % 16
        )
    expected = struct.unpack_from(
        "<17e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(value) for value in range(1, 18))


def validate_gemm_f16_k17_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-k17"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (32, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:17] == tuple(float(value) for value in range(1, 18))
    assert lhs[17:32] == (-29.0,) * 15
    assert lhs[32:] == (0.0,) * 96

    rhs = struct.unpack_from(
        "<384e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    for row in range(16):
        assert rhs[row * 16 : (row + 1) * 16] == (1.0,) * 16
    assert rhs[16 * 16 : 17 * 16] == tuple(
        float(value) for value in range(2, 18)
    )
    assert rhs[17 * 16 :] == (-31.0,) * 112

    expected = struct.unpack_from(
        "<16e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(170 + 17 * column) for column in range(16))


def validate_gemm_bf16_k17_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-k17"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (32, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words(
        "BF16", [1.0] * 17 + [-29.0] * 15 + [0.0] * 96
    )

    rhs = struct.unpack_from(
        "<384H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected_rhs = [
        float(1 if row < 16 else column + 2)
        for row in range(17)
        for column in range(16)
    ] + [-31.0] * 112
    assert rhs == _encoded_words("BF16", expected_rhs)

    expected = struct.unpack_from(
        "<16H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words(
        "BF16", [float(18 + column) for column in range(16)]
    )


def validate_gemm_f16_n65_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-n65"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        130,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (1.0,) * 16
    assert lhs[16:] == (0.0,) * 112

    rhs = struct.unpack_from(
        "<1152e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    full_block = rhs[: 16 * 64]
    for row in range(16):
        expected_row = (
            (1.0,) * 64
            if row < 15
            else tuple(float(value) for value in range(100, 164))
        )
        assert full_block[row * 64 : (row + 1) * 64] == expected_row
    tail = rhs[16 * 64 : 16 * 68]
    for row in range(16):
        assert tail[row * 4 : (row + 1) * 4] == (
            float(1 if row < 15 else 164),
            -29.0,
            -29.0,
            -29.0,
        )
    assert rhs[16 * 68 :] == (-31.0,) * 64

    expected = struct.unpack_from(
        "<65e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(115 + column) for column in range(65))


def validate_gemm_bf16_n65_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-n65"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        130,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words("BF16", [1.0] * 16 + [0.0] * 112)

    rhs = struct.unpack_from(
        "<1152H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    physical_rhs = (
        [
            float(1 if row < 15 else 100 + column)
            for row in range(16)
            for column in range(64)
        ]
        + [
            value
            for row in range(16)
            for value in (
                float(1 if row < 15 else 164),
                -29.0,
                -29.0,
                -29.0,
            )
        ]
        + [-31.0] * 64
    )
    assert rhs == _encoded_words("BF16", physical_rhs)

    expected = struct.unpack_from(
        "<65H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words(
        "BF16", [float(115 + column) for column in range(65)]
    )


def validate_gemm_f16_oriented_nt_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-oriented-nt"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = tuple(
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    )
    semantic_rhs = tuple(
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    )
    stored_rhs = tuple(
        semantic_rhs[contracting * 8 + column]
        for column in range(8)
        for contracting in range(16)
    )
    expected = tuple(
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:64] == semantic_lhs
    assert lhs[64:] == (0.0,) * 64
    rhs = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == stored_rhs
    assert stored_rhs != semantic_rhs
    actual_expected = struct.unpack_from(
        "<32e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_bf16_oriented_nt_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-oriented-nt"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = [
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    ]
    semantic_rhs = [
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    ]
    stored_rhs = [
        semantic_rhs[contracting * 8 + column]
        for column in range(8)
        for contracting in range(16)
    ]
    expected_values = [
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    ]

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words("BF16", semantic_lhs + [0.0] * 64)
    rhs = struct.unpack_from(
        "<128H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == _encoded_words("BF16", stored_rhs)
    assert stored_rhs != semantic_rhs
    expected = struct.unpack_from(
        "<32H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words("BF16", expected_values)


def validate_gemm_f16_oriented_tn_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-oriented-tn"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = tuple(
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    )
    semantic_rhs = tuple(
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    )
    stored_lhs = tuple(
        semantic_lhs[row * 16 + contracting]
        for contracting in range(16)
        for row in range(4)
    )
    expected = tuple(
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:64] == stored_lhs
    assert lhs[64:] == (0.0,) * 64
    assert stored_lhs != semantic_lhs
    rhs = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == semantic_rhs
    actual_expected = struct.unpack_from(
        "<32e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_f16_oriented_tt_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-oriented-tt"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = tuple(
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    )
    semantic_rhs = tuple(
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    )
    stored_lhs = tuple(
        semantic_lhs[row * 16 + contracting]
        for contracting in range(16)
        for row in range(4)
    )
    stored_rhs = tuple(
        semantic_rhs[contracting * 8 + column]
        for column in range(8)
        for contracting in range(16)
    )
    expected = tuple(
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:64] == stored_lhs
    assert lhs[64:] == (0.0,) * 64
    rhs = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == stored_rhs
    assert stored_lhs != semantic_lhs
    assert stored_rhs != semantic_rhs
    actual_expected = struct.unpack_from(
        "<32e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_f16_psum_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-psum"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_COMPOSITE"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        32,
        256,
        256,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (1.0,) * 16
    assert lhs[16:] == (0.0,) * 112
    rhs = struct.unpack_from(
        "<256e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == tuple(
        float(column + 1)
        for _row in range(16)
        for column in range(16)
    )
    psum = struct.unpack_from(
        "<128e",
        built.payload,
        3 * catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert psum[:16] == tuple(float(1000 + column) for column in range(16))
    assert psum[16:] == (-23.0,) * 112

    expected = struct.unpack_from(
        "<16e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    overwrite_only = tuple(float(16 * (column + 1)) for column in range(16))
    assert expected == tuple(
        float(1016 + 17 * column) for column in range(16)
    )
    assert expected != overwrite_only


def validate_arg_extrema_composite_oracles() -> None:
    expected_cases = (
        ("peripheral-argmax-f16", max, 73, 100.0),
        ("peripheral-argmin-f16", min, 42, 0.5),
    )
    for name, operation, expected_index, expected_value in expected_cases:
        case = catalog.CASES_BY_NAME[name]
        assert case.oracle_name == "EXACT_COMPOSITE"
        assert case.result_bytes == 8
        assert case.output_span == 8
        assert case.aux_span == 0
        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<128e", built.payload, catalog.BODY_OFFSET
        )
        assert operation(source) == expected_value
        assert source.count(expected_value) == 1
        assert source.index(expected_value) == expected_index

        result = built.expected_output_slot[
            catalog.BODY_OFFSET : catalog.BODY_OFFSET + case.result_bytes
        ]
        assert struct.unpack_from("<e", result, 0)[0] == expected_value
        assert result[2:4] == _output_seed_padding()
        assert struct.unpack_from("<I", result, 4)[0] == expected_index
        output_payload_offset = 2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET
        assert (
            built.payload[
                output_payload_offset + 2 : output_payload_offset + 4
            ]
            == result[2:4]
        )


def validate_pool_max_oracle() -> None:
    source_values = [
        float(100 * row + 10 * column + channel % 8)
        for row in range(2)
        for column in range(4)
        for channel in range(64)
    ]
    expected_values = [
        float(100 + 10 * (2 * output_column + 1) + channel % 8)
        for output_column in range(2)
        for channel in range(64)
    ]
    for name, dtype_name in (
        ("pool-f16", "F16"),
        ("pool-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 256
        assert case.output_span == 256
        assert case.aux_span == 0

        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<512H", built.payload, catalog.BODY_OFFSET
        )
        assert source == _encoded_words(dtype_name, source_values)
        result = struct.unpack_from(
            "<128H", built.expected_output_slot, catalog.BODY_OFFSET
        )
        assert result == _encoded_words(dtype_name, expected_values)
        assert built.expected_output_slot[: catalog.BODY_OFFSET] == bytes(
            [catalog.SLOT_CANARY]
        ) * catalog.BODY_OFFSET
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.output_span :
        ] == bytes([catalog.SLOT_CANARY]) * (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
        )


def validate_unpool_composite_oracle() -> None:
    case = catalog.CASES_BY_NAME["unpool-f16"]
    assert case.is_safe
    assert case.oracle_name == "EXACT_COMPOSITE"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        512,
        512,
        256,
    )

    built = catalog.build_case_payload(case)
    source = struct.unpack_from(
        "<256e", built.payload, catalog.BODY_OFFSET
    )
    assert source == tuple(
        float(128 + channel if position == channel % 4 else 1 + position)
        for position in range(4)
        for channel in range(64)
    )
    output_seed = built.payload[
        2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET :
        2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET + case.output_span
    ]
    assert output_seed == bytes(case.output_span)
    expected = struct.unpack_from(
        "<256e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(
        float(128 + channel if position == channel % 4 else 0)
        for position in range(4)
        for channel in range(64)
    )

    probe = (
        pathlib.Path(__file__).resolve().parent
        / "Inputs"
        / "wafer_instruction_family_probe.c"
    ).read_text()
    case_body = probe[
        probe.index("case WAFER_IFP_CASE_UNPOOL_F16:") :
        probe.index("break;", probe.index("case WAFER_IFP_CASE_UNPOOL_F16:"))
    ]
    pool = case_body.index("wafer_tx81_pool_indexedmax")
    fence = case_body.index("wafer_tx81_local_fence")
    unpool = case_body.index("wafer_tx81_unpool_mask")
    assert pool < fence < unpool
    assert "(uint32_t)auxiliary" in case_body


def validate_conv_oracle() -> None:
    source_values = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    weight_values = (
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
        17.0,
    )
    expected_values = (1.0, 16.0, 3.0, 11.0, 2.0, 32.0, 5.0, 13.0)

    for name, dtype_name in (
        ("conv-f16", "F16"),
        ("conv-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 16
        assert case.output_span == 256
        assert case.aux_span == 0

        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<12H", built.payload, catalog.BODY_OFFSET
        )
        assert source == _encoded_words(dtype_name, source_values)
        weight = struct.unpack_from(
            "<16H",
            built.payload,
            catalog.SLOT_BYTES + catalog.BODY_OFFSET,
        )
        assert weight == _encoded_words(dtype_name, weight_values)
        result = struct.unpack_from(
            "<8H", built.expected_output_slot, catalog.BODY_OFFSET
        )
        assert result == _encoded_words(dtype_name, expected_values)

        padding_elements = (case.output_span - case.result_bytes) // 2
        padding_word = _encoded_words(dtype_name, (-13.0,))[0]
        padding = struct.pack("<H", padding_word) * padding_elements
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.result_bytes :
            catalog.BODY_OFFSET + case.output_span
        ] == padding
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.output_span :
        ] == bytes([catalog.SLOT_CANARY]) * (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
        )


def validate_img2col_oracle() -> None:
    for name, dtype_name in (
        ("tdma-img2col-f16", "F16"),
        ("tdma-img2col-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 2048
        assert case.output_span == 2048
        assert case.aux_span == 0

        if dtype_name == "F16":
            source_values = [
                float(1 + 256 * row + 64 * column + channel)
                for row in range(3)
                for column in range(3)
                for channel in range(64)
            ]
        else:
            source_values = [
                float(1 + 64 * row + 16 * column + channel % 8)
                for row in range(3)
                for column in range(3)
                for channel in range(64)
            ]
        source_words = _encoded_words(dtype_name, source_values)
        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<576H", built.payload, catalog.BODY_OFFSET
        )
        assert source == source_words

        expected = tuple(
            source_words[
                ((output_row + kernel_y) * 3 + output_column + kernel_x) * 64
                + channel
            ]
            for kernel_y in range(2)
            for kernel_x in range(2)
            for output_row in range(2)
            for output_column in range(2)
            for channel in range(64)
        )
        result = struct.unpack_from(
            "<1024H", built.expected_output_slot, catalog.BODY_OFFSET
        )
        assert result == expected
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.output_span :
        ] == bytes([catalog.SLOT_CANARY]) * (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
        )


def validate_lut16_oracle() -> None:
    case = catalog.CASES_BY_NAME["peripheral-lut16-f16"]
    assert case.is_safe
    assert case.oracle_name == "EXACT_BITS"
    assert case.result_bytes == 256
    assert case.output_span == 256
    assert case.aux_span == 0

    built = catalog.build_case_payload(case)
    source = struct.unpack_from(
        "<128H", built.payload, catalog.BODY_OFFSET
    )
    table = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected_indices = tuple(
        (37 * index + 11) % 128 for index in range(128)
    )
    assert source == tuple(2 * index for index in expected_indices)
    assert all(offset % 2 == 0 and offset < 256 for offset in source)
    assert table == tuple(float(index - 64) for index in range(128))

    result = struct.unpack_from(
        "<128e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert result == tuple(table[index] for index in expected_indices)


def _output_seed_padding() -> bytes:
    return struct.pack("<e", -13.0)


def main() -> int:
    assert len(catalog.SAFE_CASES) == 60
    assert len(catalog.CATALOG) == 60
    assert {case.case_id for case in catalog.SAFE_CASES} == (
        set(range(1, 30))
        | {
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
        }
    )
    assert {
        (case.symbol, case.reason_name)
        for case in catalog.CATALOG
        if not case.is_safe
    } == set()
    for name in (
        "reduce-sum-f16",
        "reduce-max-f16",
        "reduce-min-bf16",
        "reduce-avg-bf16",
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.result_bytes == 128
        assert case.output_span == 256

    for case in catalog.SAFE_CASES:
        built = catalog.build_case_payload(case, sample=7)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert len(built.expected_output_slot) == catalog.SLOT_BYTES
        words = struct.unpack_from(f"<{catalog.REQUEST_WORDS}Q", built.request)
        assert words[catalog.REQ["MAGIC"]] == catalog.REQUEST_MAGIC
        assert words[catalog.REQ["CASE"]] == case.case_id
        assert words[catalog.REQ["FAMILY"]] == case.family
        assert words[catalog.REQ["DTYPE"]] == case.dtype
        assert words[catalog.REQ["RESULT_BYTES"]] == case.result_bytes
        assert words[catalog.REQ["OUTPUT_SPAN"]] == case.output_span
        assert words[catalog.REQ["AUX_SPAN"]] == case.aux_span
        assert words[catalog.REQ["SAMPLE"]] == 7
        assert words[catalog.REQ["GUARD"]] == catalog.REQUEST_GUARD
        assert any(built.payload[: catalog.SLOT_BYTES])
        assert (
            built.expected_output_slot[: catalog.BODY_OFFSET]
            == bytes([catalog.SLOT_CANARY]) * catalog.BODY_OFFSET
        )
        assert (
            built.expected_output_slot[
                catalog.BODY_OFFSET + case.output_span :
            ]
            == bytes([catalog.SLOT_CANARY])
            * (catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span)
        )

    for case in catalog.CATALOG:
        if case.is_safe:
            continue
        try:
            catalog.build_case_payload(case)
        except RuntimeError as error:
            assert "deferred case" in str(error)
        else:
            raise AssertionError(f"{case.name}: deferred case became executable")

    select = catalog.build_case_payload(
        catalog.CASES_BY_NAME["select-bit2fp-maskmove-f16"]
    )
    mask = select.payload[
        catalog.SLOT_BYTES + catalog.BODY_OFFSET :
        catalog.SLOT_BYTES + catalog.BODY_OFFSET + 16
    ]
    assert mask == bytes(
        0xFF if byte % 2 == 0 else 0x00 for byte in range(16)
    )
    validate_rounding_and_bit2fp_oracles()
    validate_ct_add_f16_tail130_oracle()
    validate_ct_add_bf16_tail130_oracle()
    validate_ct_add_f32_oracle()
    validate_ct_add_special_f16_oracle()
    validate_ct_add_special_bf16_oracle()
    validate_gemm_padding_domain()
    validate_gemm_dtype_oracles()
    validate_gemm_bf16_accum_round_oracle()
    validate_gemm_f16_accum_round_oracle()
    validate_gemm_f16_m4_oracle()
    validate_gemm_f16_batch2_m8_oracle()
    validate_gemm_bf16_batch2_m8_oracle()
    validate_gemm_f16_n17_oracle()
    validate_gemm_f16_k17_oracle()
    validate_gemm_bf16_k17_oracle()
    validate_gemm_f16_n65_oracle()
    validate_gemm_bf16_n65_oracle()
    validate_gemm_f16_oriented_nt_oracle()
    validate_gemm_bf16_oriented_nt_oracle()
    validate_gemm_f16_oriented_tn_oracle()
    validate_gemm_f16_oriented_tt_oracle()
    validate_gemm_f16_psum_oracle()
    validate_arg_extrema_composite_oracles()
    validate_conv_oracle()
    validate_pool_max_oracle()
    validate_unpool_composite_oracle()
    validate_img2col_oracle()
    validate_lut16_oracle()

    print("wafer_instruction_family_catalog_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
