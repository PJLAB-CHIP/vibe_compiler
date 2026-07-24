#!/usr/bin/env python3
"""Validate extended DataMove payloads, dispatch metadata and oracles."""

from __future__ import annotations

import pathlib
import struct
import tempfile

import wafer_board_datamove_extended_calibration_probe_test as runner
import wafer_datamove_extended_calibration_catalog as catalog


def validate_concat_physical_spans() -> None:
    expected_spans = {
        "C": 24576,
        "W": 17408,
        "H": 17920,
        "HW": 9216,
    }
    assert {
        axis: catalog._raw_concat_physical_span(axis)
        for axis in catalog.RAW_CONCAT_SPECS
    } == expected_spans
    assert catalog.NATIVE_CONCAT_AXES == {"C", "W", "H"}
    assert catalog.UNQUALIFIED_NATIVE_CONCAT_AXES == {"HW"}
    for sample, case in enumerate(catalog.CONCAT_CASES):
        axis = str(case.semantic_axis)
        left_shape, right_shape, output_shape, _ = (
            catalog.RAW_CONCAT_SPECS[axis]
        )
        left_span = catalog._raw_concat_operand_span(left_shape)
        right_span = catalog._raw_concat_operand_span(right_shape)
        assert case.input_bytes == left_span + right_span
        assert case.output_span == expected_spans[axis]
        assert case.output_span == left_span + right_span
        assert case.result_bytes == catalog._compact_bytes(output_shape)

        source, _ = catalog.build_input_expected(case, sample + 1)
        left = source[:left_span]
        right = source[left_span:]
        assert len(right) == right_span
        assert catalog.codec.unpack_scalar_bytes(
            left_shape, "NCx", catalog.FP16_BYTES, left
        ) == catalog._logical_values(
            catalog._product(left_shape), sample + 1
        )
        assert catalog.codec.unpack_scalar_bytes(
            right_shape, "NCx", catalog.FP16_BYTES, right
        ) == catalog._logical_values(
            catalog._product(right_shape), sample + 42
        )

    held_out_left = (3, 5, 11, 17)
    held_out_right = (3, 5, 11, 15)
    assert (
        catalog._raw_concat_operand_span(held_out_left)
        + catalog._raw_concat_operand_span(held_out_right)
        > catalog._span(
            catalog._compact_bytes(held_out_left)
            + catalog._compact_bytes(held_out_right)
        )
    )

    concat_c = catalog.CASES_BY_NAME[
        "datamove-raw-concat-c-n2h7w9-c33-c32"
    ]
    output_slot = bytearray([catalog.SLOT_CANARY] * catalog.SLOT_BYTES)
    begin = catalog.BODY_OFFSET
    output_slot[begin : begin + concat_c.output_span] = bytes(
        concat_c.output_span
    )
    old_compact_span = catalog._span(concat_c.result_bytes)
    assert old_compact_span == 16384
    assert (
        runner.output_guard_mismatches(bytes(output_slot), old_compact_span)
        == 8192
    )
    assert (
        runner.output_guard_mismatches(
            bytes(output_slot), concat_c.output_span
        )
        == 0
    )
    output_slot[begin + concat_c.output_span] ^= 1
    assert (
        runner.output_guard_mismatches(
            bytes(output_slot), concat_c.output_span
        )
        == 1
    )


def validate_resource_canaries() -> None:
    case = catalog.CASES_BY_NAME[
        "datamove-pad-large-n2h5w7c65-to-n2h7w10c65"
    ]
    built = catalog.build_case_payload(case, sample=0)
    raw = bytearray(
        [runner.package_support.OUTPUT_INITIAL_CANARY]
        * catalog.RESOURCE_BYTES
    )
    raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ] = built.expected_output_slot
    words = [0] * catalog.RECORD_WORDS
    expected = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (catalog.SCHEMA << 32) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "INPUT_BYTES": case.input_bytes,
        "RESULT_BYTES": case.result_bytes,
        "OUTPUT_SPAN": case.output_span,
        "TDMA_INSTRUCTIONS": case.expected_tdma_instructions,
        "CT_INSTRUCTIONS": case.expected_ct_instructions,
        "NE_INSTRUCTIONS": case.expected_ne_instructions,
        "ORACLE": case.oracle,
        "SAMPLE": 0,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "OUTPUT_GUARD_MISMATCHES": 0,
        "TDMA_INST_DELTA": case.expected_tdma_instructions,
        "CT_INST_DELTA": case.expected_ct_instructions,
        "NE_INST_DELTA": case.expected_ne_instructions,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    for name, value in expected.items():
        words[catalog.REC[name]] = value
    raw[: catalog.RECORD_WORDS * 8] = struct.pack(
        f"<{catalog.RECORD_WORDS}Q", *words
    )
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "output.raw"
        output.write_bytes(raw)
        runner.validate_output(output, case, built, sample=0)
        raw[catalog.RECORD_WORDS * 8] ^= 1
        output.write_bytes(raw)
        try:
            runner.validate_output(output, case, built, sample=0)
        except RuntimeError as error:
            assert "output changed outside record/slot" in str(error)
        else:
            raise AssertionError("output resource canary corruption accepted")


def main() -> int:
    assert len(catalog.CATALOG) == 17
    assert len(catalog.CASES_BY_ID) == len(catalog.CATALOG)
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert [case.case_id for case in catalog.CATALOG] == [
        0,
        1,
        2,
        *range(4, 18),
    ]
    assert {case.opcode for case in catalog.CATALOG} >= {
        123,
        131,
        132,
        133,
        135,
        136,
        137,
        138,
    }
    assert {case.semantic_axis for case in catalog.CONCAT_CASES} == {
        "C",
        "W",
        "H",
    }
    assert all(not case.is_exact for case in catalog.CONCAT_CASES)
    assert all(case.is_exact for case in catalog.LARGE_TYPED_CASES)
    assert all(
        not case.is_exact for case in catalog.RAW_OBSERVATION_CASES
    )
    assert all(case.is_exact for case in catalog.COMPOSITE_CASES)
    assert all(case.is_exact for case in catalog.TDMA_DESCRIPTOR_CASES)
    assert catalog.IMG2COL_RESULT_BYTES > 65536
    assert catalog.BODY_OFFSET + catalog.IMG2COL_RESULT_BYTES <= (
        catalog.SLOT_BYTES
    )
    validate_concat_physical_spans()

    for sample, case in enumerate(catalog.CATALOG):
        assert case.result_bytes > 0
        assert case.output_span % 256 == 0
        assert case.result_bytes <= case.output_span
        assert catalog.BODY_OFFSET + case.input_bytes <= catalog.SLOT_BYTES
        assert catalog.BODY_OFFSET + case.output_span <= catalog.SLOT_BYTES
        assert (
            case.expected_tdma_instructions
            + case.expected_ct_instructions
            + case.expected_ne_instructions
            > 0
        )
        built = catalog.build_case_payload(case, sample)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert len(built.expected_output_slot) == catalog.SLOT_BYTES
        source, expected = catalog.build_input_expected(case, sample + 1)
        assert len(source) == case.input_bytes
        if case.is_exact:
            assert len(expected) == case.result_bytes
            assert (
                built.expected_output_slot[
                    catalog.BODY_OFFSET :
                    catalog.BODY_OFFSET + case.result_bytes
                ]
                == expected
            )

    cx = catalog.CASES_BY_NAME[
        "datamove-cx-materialize-ct-add-n2c65"
    ]
    ncx = catalog.CASES_BY_NAME[
        "datamove-ncx-materialize-ct-add-n2c65"
    ]
    assert cx.expected_tdma_instructions == 2
    assert ncx.expected_tdma_instructions == 4
    assert cx.expected_ct_instructions == ncx.expected_ct_instructions == 1
    assert catalog.LAYOUT_CX_BYTES == catalog.LAYOUT_NCX_BYTES == 512
    assert catalog.CALIBRATION_LEAF_BINDINGS
    assert set().union(*map(set, catalog.CALIBRATION_LEAF_BINDINGS.values())) == (
        set(catalog.CATALOG)
    )

    probe = (
        pathlib.Path(__file__).parent
        / "Inputs"
        / "wafer_datamove_extended_calibration_probe.c"
    ).read_text()
    assert "wafer_dmx_cases[]" in probe
    assert "move->Concat" in probe
    assert "move->TensorNom" in probe
    assert "move->MaskGather" in probe
    assert "wafer_tx81_tdma_pad" in probe
    assert "wafer_tx81_tdma_img2col" in probe
    assert "wafer_tx81_elementwise_add" in probe
    assert "wafer_tx81_gemm" in probe
    for row in (
        "{0U, 24576U, 16380U, 24576U, 0U, 1U, 0U, 1U, 1U}",
        "{1U, 17408U, 16380U, 17408U, 0U, 1U, 0U, 1U, 1U}",
        "{2U, 17920U, 16380U, 17920U, 0U, 1U, 0U, 1U, 1U}",
        "{3U, 9216U, 8060U, 9216U, 0U, 1U, 0U, 1U, 0U}",
    ):
        assert row in probe
    assert "selected->executable == 0U" in probe
    assert "case 1U:" in probe
    assert "case 2U:" in probe
    assert "case 3U:" not in probe
    for source1_offset in (16384, 7680):
        assert f"input + {source1_offset}U" in probe
    assert "input + 3072U" not in probe
    validate_resource_canaries()
    print(
        "wafer_datamove_extended_calibration_catalog_test: "
        "cases=17 exact=11 observation=6 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
