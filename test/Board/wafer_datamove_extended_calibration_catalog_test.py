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
    }
    assert {
        axis: catalog._raw_concat_physical_span(axis)
        for axis in catalog.RAW_CONCAT_SPECS
    } == expected_spans
    assert catalog.NATIVE_CONCAT_AXES == {"C", "W", "H"}
    assert catalog.INVALID_NATIVE_CONCAT_AXES == {"HW"}
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


def validate_large_typed_physical_spans() -> None:
    cases = (
        (
            catalog.CASES_BY_NAME[
                "datamove-pad-large-n2h5w7c65-to-n2h7w10c65"
            ],
            catalog.PAD_SOURCE_SHAPE,
            catalog.PAD_DESTINATION_SHAPE,
        ),
        (
            catalog.CASES_BY_NAME[
                "datamove-img2col-large-n2h9w11c65-k3x2-s2x1"
            ],
            catalog.IMG2COL_SOURCE_SHAPE,
            catalog.IMG2COL_DESTINATION_SHAPE,
        ),
    )
    for sample, (case, source_shape, destination_shape) in enumerate(cases):
        source_layout = catalog.codec.physical_layout(
            source_shape, "NCx", catalog.FP16_BYTES
        )
        destination_layout = catalog.codec.physical_layout(
            destination_shape, "NCx", catalog.FP16_BYTES
        )
        assert case.input_bytes == source_layout.physical_bytes
        assert case.result_bytes == destination_layout.physical_bytes
        assert case.output_span == destination_layout.physical_bytes
        source, expected = catalog.build_input_expected(case, sample + 1)
        assert catalog.codec.unpack_scalar_bytes(
            source_shape, "NCx", catalog.FP16_BYTES, source
        ) == catalog._logical_values(
            catalog._product(source_shape), sample + 1
        )
        logical_expected = catalog.codec.unpack_scalar_bytes(
            destination_shape,
            "NCx",
            catalog.FP16_BYTES,
            expected,
        )
        assert len(logical_expected) == catalog._product(destination_shape)
        assert (
            catalog.codec.pack_scalar_bytes(
                destination_shape,
                "NCx",
                catalog.FP16_BYTES,
                logical_expected,
                padding=0,
                batch_padding=(
                    catalog.SLOT_CANARY
                    if case.operation == "img2col-large"
                    else None
                ),
            )
            == expected
        )

    pad = cases[0][0]
    old_compact_span = catalog._span(
        catalog._compact_bytes(catalog.PAD_DESTINATION_SHAPE)
    )
    assert old_compact_span == 18432
    assert pad.output_span == 19456
    assert pad.output_span - old_compact_span == 1024
    output_slot = bytearray([catalog.SLOT_CANARY] * catalog.SLOT_BYTES)
    begin = catalog.BODY_OFFSET
    output_slot[begin : begin + pad.output_span] = bytes(pad.output_span)
    assert (
        runner.output_guard_mismatches(
            bytes(output_slot), old_compact_span
        )
        == 1024
    )
    assert (
        runner.output_guard_mismatches(bytes(output_slot), pad.output_span)
        == 0
    )

    img2col = cases[1][0]
    _, img2col_expected = catalog.build_input_expected(img2col, 2)
    img2col_layout = catalog.codec.physical_layout(
        catalog.IMG2COL_DESTINATION_SHAPE,
        "NCx",
        catalog.FP16_BYTES,
    )
    active_bytes = (
        img2col_layout.hw_elements
        * img2col_layout.aligned_c
        * catalog.FP16_BYTES
    )
    batch_bytes = (
        img2col_layout.batch_elements * catalog.FP16_BYTES
    )
    assert active_bytes == 44064
    assert batch_bytes == 44288
    for batch in range(catalog.IMG2COL_DESTINATION_SHAPE[0]):
        begin = batch * batch_bytes + active_bytes
        end = (batch + 1) * batch_bytes
        assert img2col_expected[begin:end] == bytes(
            [catalog.SLOT_CANARY]
        ) * (end - begin)


def validate_tensor_nom_physical_span() -> None:
    case = catalog.CASES_BY_NAME[
        "datamove-raw-tensornom-n2h7w9c65"
    ]
    layout = catalog.codec.physical_layout(
        catalog.TENSOR_NOM_SHAPE,
        "NCx",
        catalog.FP16_BYTES,
    )
    assert not case.is_exact
    assert case.input_bytes == catalog._compact_bytes(
        catalog.TENSOR_NOM_SHAPE
    )
    assert case.result_bytes == layout.physical_bytes == 17408
    assert case.output_span == layout.physical_bytes

    source, semantic_expected = catalog.build_input_expected(case, 1)
    logical = catalog._logical_values(
        catalog._product(catalog.TENSOR_NOM_SHAPE), 1
    )
    assert source == b"".join(logical)
    assert semantic_expected == catalog.codec.pack_scalar_bytes(
        catalog.TENSOR_NOM_SHAPE,
        "NCx",
        catalog.FP16_BYTES,
        logical,
        padding=0,
    )
    assert catalog.codec.unpack_scalar_bytes(
        catalog.TENSOR_NOM_SHAPE,
        "NCx",
        catalog.FP16_BYTES,
        semantic_expected,
    ) == logical

    old_compact_span = catalog._span(case.input_bytes)
    assert old_compact_span == 16384
    output_slot = bytearray([catalog.SLOT_CANARY] * catalog.SLOT_BYTES)
    begin = catalog.BODY_OFFSET
    output_slot[
        begin : begin + case.output_span
    ] = semantic_expected
    assert (
        runner.output_guard_mismatches(
            bytes(output_slot), old_compact_span
        )
        == 1024
    )
    assert (
        runner.output_guard_mismatches(
            bytes(output_slot), case.output_span
        )
        == 0
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
    cmake = (
        pathlib.Path(__file__).resolve().parents[1] / "CMakeLists.txt"
    ).read_text()
    assert "wafer-board-datamove-native-concat-hw" not in cmake
    assert "--allow-isolated-native-concat-hw" not in cmake
    assert runner.execution_sample_count(
        exact=True, observation_samples=3
    ) == 1
    assert runner.execution_sample_count(
        exact=False, observation_samples=3
    ) == 3
    try:
        runner.execution_sample_count(
            exact=False, observation_samples=0
        )
    except ValueError as error:
        assert "--observation-samples must be positive" in str(error)
    else:
        raise AssertionError(
            "zero DataMove observation sample count was accepted"
    )
    assert len(catalog.CATALOG) == 17
    assert len(catalog.ALL_CASES) == 17
    assert len(catalog.CASES_BY_ID) == len(catalog.ALL_CASES)
    assert len(catalog.CASES_BY_NAME) == len(catalog.ALL_CASES)
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
    assert all(case.semantic_axis != "HW" for case in catalog.ALL_CASES)
    assert tuple(case.case_id for case in catalog.ALL_CASES) == (
        0,
        1,
        2,
        *range(4, 18),
    )
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
    validate_large_typed_physical_spans()
    validate_tensor_nom_physical_span()
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
    assert "if (dimension > 2U)" in probe
    assert "wafer_tx81_tdma_pad" in probe
    assert "wafer_tx81_tdma_img2col" in probe
    assert "wafer_tx81_elementwise_add" in probe
    assert "wafer_tx81_gemm" in probe
    entry = probe[
        probe.index("wafer_tx81_instruction_family_probe(uint64_t request_ddr")
        :
    ]
    assert "get_spm_memory_mapping" not in probe
    assert "wafer_dmx_fill" not in probe
    assert "wafer_dmx_guard_mismatches" not in probe
    input_rdma = entry.index("wafer_tx81_rdma(payload_ddr,")
    canary_rdma = entry.index(
        "wafer_tx81_rdma(request_ddr + WAFER_DMX_SLOT_BYTES,"
    )
    issue = entry.index("wafer_dmx_issue(&selected, &raw_execute_rc)")
    wdma = entry.index("wafer_tx81_wdma(", issue)
    terminal = entry.index("wafer_tx81_local_fence();", wdma)
    assert input_rdma < canary_rdma < issue < wdma < terminal
    assert entry.count("wafer_tx81_local_fence();") == 1
    for row in (
        "{0U, 24576U, 16380U, 24576U, 0U, 1U, 0U, 1U}",
        "{1U, 17408U, 16380U, 17408U, 0U, 1U, 0U, 1U}",
        "{2U, 17920U, 16380U, 17920U, 0U, 1U, 0U, 1U}",
        "{4U, 9728U, 19456U, 19456U, 1U, 0U, 0U, 0U}",
        "{5U, 27136U, 88576U, 88576U, 1U, 0U, 0U, 0U}",
        "{8U, 16380U, 17408U, 17408U, 1U, 0U, 0U, 1U}",
    ):
        assert row in probe
    assert "case 1U:" in probe
    assert "case 2U:" in probe
    assert "case 3U:" not in probe
    assert "{3U, 9216U, 8060U, 9216U" not in probe
    assert "wafer_dmx_shape(2U, 1U, 31U, 65U)" not in probe
    for source1_offset in (16384, 7680):
        assert f"input + {source1_offset}U" in probe
    assert "input + 3072U" not in probe
    validate_resource_canaries()
    print(
        "wafer_datamove_extended_calibration_catalog_test: "
        "default=17 all=17 exact=11 observation=6 native_hw=static-negative "
        "passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
