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


def main() -> int:
    assert len(catalog.SAFE_CASES) == 29
    assert len(catalog.CATALOG) == 36
    assert {case.case_id for case in catalog.SAFE_CASES} == set(range(1, 30))
    assert {
        case.reason_name
        for case in catalog.CATALOG
        if not case.is_safe
    } == {
        "REASON_ISOLATED_COMPLETION_WRITEBACK_UNQUALIFIED",
        "REASON_GEOMETRY_UNQUALIFIED",
        "REASON_NUMERIC_UNQUALIFIED",
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

    print("wafer_instruction_family_catalog_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
