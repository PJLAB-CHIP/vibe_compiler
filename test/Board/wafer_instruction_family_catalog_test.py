#!/usr/bin/env python3
"""No-card tests for instruction qualification catalog and host oracles."""

from __future__ import annotations

import pathlib
import re
import struct
import tempfile

import wafer_instruction_family_catalog as catalog
import wafer_board_instruction_family_probe_test as runner


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
    case = catalog.CASES_BY_NAME["pool-f16"]
    assert case.is_safe
    assert case.oracle_name == "EXACT_BITS"
    assert case.result_bytes == 256
    assert case.output_span == 256
    assert case.aux_span == 0

    built = catalog.build_case_payload(case)
    source = struct.unpack_from(
        "<512e", built.payload, catalog.BODY_OFFSET
    )
    assert source[0] == 0.0
    assert source[63] == 7.0
    assert source[64] == 10.0
    assert source[4 * 64] == 100.0
    assert source[-1] == 137.0

    result = struct.unpack_from(
        "<128e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert result[:8] == tuple(float(110 + channel) for channel in range(8))
    assert result[64:72] == tuple(
        float(130 + channel) for channel in range(8)
    )
    assert built.expected_output_slot[: catalog.BODY_OFFSET] == bytes(
        [catalog.SLOT_CANARY]
    ) * catalog.BODY_OFFSET
    assert built.expected_output_slot[
        catalog.BODY_OFFSET + case.output_span :
    ] == bytes([catalog.SLOT_CANARY]) * (
        catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
    )


def validate_img2col_oracle() -> None:
    case = catalog.CASES_BY_NAME["tdma-img2col-f16"]
    assert case.is_safe
    assert case.oracle_name == "EXACT_BITS"
    assert case.result_bytes == 2048
    assert case.output_span == 2048
    assert case.aux_span == 0

    built = catalog.build_case_payload(case)
    source = struct.unpack_from(
        "<576e", built.payload, catalog.BODY_OFFSET
    )
    assert source[0] == 1.0
    assert source[64] == 65.0
    assert source[3 * 64] == 257.0
    assert source[-1] == 704.0

    result = struct.unpack_from(
        "<1024e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    expected = tuple(
        source[((output_row + kernel_y) * 3 + output_column + kernel_x) * 64
               + channel]
        for kernel_y in range(2)
        for kernel_x in range(2)
        for output_row in range(2)
        for output_column in range(2)
        for channel in range(64)
    )
    assert result == expected
    assert result[::64] == (
        1.0,
        65.0,
        257.0,
        321.0,
        65.0,
        129.0,
        321.0,
        385.0,
        257.0,
        321.0,
        513.0,
        577.0,
        321.0,
        385.0,
        577.0,
        641.0,
    )
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
    assert len(catalog.SAFE_CASES) == 35
    assert len(catalog.CATALOG) == 37
    assert {case.case_id for case in catalog.SAFE_CASES} == (
        set(range(1, 30)) | {100, 101, 103, 105, 106, 107}
    )
    assert {
        case.reason_name
        for case in catalog.CATALOG
        if not case.is_safe
    } == {
        "REASON_GEOMETRY_UNQUALIFIED",
    }
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
    validate_gemm_padding_domain()
    validate_gemm_dtype_oracles()
    validate_arg_extrema_composite_oracles()
    validate_pool_max_oracle()
    validate_img2col_oracle()
    validate_lut16_oracle()

    print("wafer_instruction_family_catalog_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
