#!/usr/bin/env python3
"""Validate DataMove board cases and all opcode 121..138 dispositions."""

from __future__ import annotations

import pathlib
import struct
import tempfile

import wafer_board_datamove_calibration_probe_test as runner
import wafer_datamove_calibration_catalog as catalog


def validate_resource_canaries() -> None:
    case = catalog.CASES_BY_NAME["datamove-transpose-37x53"]
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
        "EXPECTED_INSTRUCTIONS": case.expected_instructions,
        "SAMPLE": 0,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "OUTPUT_GUARD_MISMATCHES": 0,
        "TDMA_INST_DELTA": case.expected_instructions,
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
    repo = pathlib.Path(__file__).resolve().parents[2]
    assert len(catalog.CATALOG) == 46
    assert len(catalog.CASES_BY_ID) == len(catalog.CATALOG)
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert len(catalog.PUBLIC_DISPOSITIONS) == 18
    assert set(row.opcode for row in catalog.PUBLIC_DISPOSITIONS) == set(
        range(121, 139)
    )
    assert {
        row.opcode
        for row in catalog.PUBLIC_DISPOSITIONS
        if row.disposition == "isolated-deferred"
    } == set()
    for row in catalog.PUBLIC_DISPOSITIONS:
        if row.disposition in {"board-executable", "board-observation"}:
            assert row.evidence
            assert row.reason is None
        else:
            assert row.reason
        for evidence in catalog._EXISTING_EVIDENCE.get(row.opcode, ()):
            assert row.opcode in catalog.EXISTING_EVIDENCE_OPCODES[evidence], (
                f"{row.opcode}: evidence {evidence} does not issue this "
                "public opcode"
            )

    assert len(catalog.INSTRUCTION_LAYOUT_DISPOSITIONS) == 20
    assert {
        (row.instruction_family, row.layout)
        for row in catalog.INSTRUCTION_LAYOUT_DISPOSITIONS
    } == {
        (family, layout)
        for family in ("CT", "NE", "RDMA", "WDMA", "TDMA")
        for layout in ("Tensor", "NTensor", "Cx", "NCx")
    }
    for row in catalog.INSTRUCTION_LAYOUT_DISPOSITIONS:
        if row.disposition == "static-negative":
            assert not row.evidence
            assert row.reason
        else:
            assert row.evidence
            assert row.reason is None
    assert set(catalog.INSTRUCTION_LAYOUT_POSITIVE_DISPOSITIONS).isdisjoint(
        catalog.INSTRUCTION_LAYOUT_COMPOSITE_DEFERRED_DISPOSITIONS
    )
    assert set(catalog.INSTRUCTION_LAYOUT_POSITIVE_DISPOSITIONS).isdisjoint(
        catalog.INSTRUCTION_LAYOUT_NONBOARD_DISPOSITIONS
    )
    assert set(
        catalog.INSTRUCTION_LAYOUT_COMPOSITE_DEFERRED_DISPOSITIONS
    ).isdisjoint(catalog.INSTRUCTION_LAYOUT_NONBOARD_DISPOSITIONS)
    assert (
        set(catalog.INSTRUCTION_LAYOUT_POSITIVE_DISPOSITIONS)
        | set(catalog.INSTRUCTION_LAYOUT_COMPOSITE_DEFERRED_DISPOSITIONS)
        | set(catalog.INSTRUCTION_LAYOUT_NONBOARD_DISPOSITIONS)
    ) == set(catalog.INSTRUCTION_LAYOUT_DISPOSITIONS)
    assert all(
        row.disposition == "native-board-executable"
        for row in catalog.CALIBRATION_LEAF_BINDINGS[
            "instruction-layout-native-positive"
        ]
    )
    assert all(
        row.disposition == "composite-board-executable"
        for row in catalog.CALIBRATION_LEAF_BINDINGS[
            "instruction-layout-composite-positive"
        ]
    )
    assert all(
        row.disposition == "static-negative"
        for row in catalog.CALIBRATION_LEAF_BINDINGS[
            "instruction-layout-static-negative"
        ]
    )

    operations = {case.operation for case in catalog.CATALOG}
    assert {
        "transpose",
        "mirror",
        "rotate90",
        "rotate180",
        "rotate270",
        "nchw2nhwc",
        "nhwc2nchw",
        "concat",
        "broadcast-row",
        "broadcast-column",
        "tensor-to-cx",
        "cx-to-tensor",
        "tensor-to-ncx",
        "ncx-to-tensor",
        "gather-scatter-strided",
    }.issubset(operations)
    assert {
        "broadcast-scalar",
        "broadcast-channel",
        "broadcast-row-large",
        "gather-contiguous-large",
        "gather-1d-holes-large",
        "gather-2d-holes-large",
        "gather-3d-holes-large",
        "gather-tail-large",
    }.issubset(operations)
    for case in catalog.CATALOG:
        assert case.input_bytes > 0
        assert case.result_bytes > 0
        assert case.output_span % 256 == 0
        assert case.output_span >= case.result_bytes
        assert case.expected_instructions > 0
        assert (
            catalog.BODY_OFFSET + max(case.input_bytes, case.output_span)
            <= catalog.SLOT_BYTES
        )
        built = catalog.build_case_payload(case)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert len(built.expected_output_slot) == catalog.SLOT_BYTES
        assert (
            built.expected_output_slot[
                catalog.BODY_OFFSET + case.result_bytes :
            ]
            == bytes([catalog.SLOT_CANARY])
            * (catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.result_bytes)
        )

    assert catalog.CX_BYTES != catalog.NCX_BYTES
    raw_concat = {
        row.semantic: row for row in catalog.RAW_CONCAT_DISPOSITIONS
    }
    assert set(raw_concat) == {
        "concat-axis-C",
        "concat-axis-W",
        "concat-axis-H",
        "concat-axis-HW",
    }
    assert all(
        row.disposition == "board-observation"
        and row.evidence
        and row.reason is None
        for row in raw_concat.values()
    )
    assert (
        catalog.CALIBRATION_LEAF_BINDINGS["raw-concat-disposition"]
        == catalog.RAW_CONCAT_DISPOSITIONS
    )
    materialized_concat = {
        case.semantic_axis
        for case in catalog.CATALOG
        if case.operation == "concat" and case.case_id >= 15
    }
    assert materialized_concat == {"C", "W", "H", "HW", "N"}
    hw_concat = next(
        case
        for case in catalog.CATALOG
        if case.operation == "concat" and case.semantic_axis == "HW"
    )
    assert hw_concat.source1_shape is not None
    assert (
        hw_concat.source_shape[1] > 1
        and hw_concat.source1_shape[1] > 1
        and hw_concat.source_shape[1:3] != hw_concat.source1_shape[1:3]
        and hw_concat.destination_shape[1] == 1
        and hw_concat.destination_shape[2]
        == (
            hw_concat.source_shape[1] * hw_concat.source_shape[2]
            + hw_concat.source1_shape[1] * hw_concat.source1_shape[2]
        )
    )
    device_probe = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_datamove_calibration_probe.c"
    ).read_text()
    assert "{18U, 8060U, 8060U, 8192U, 2U}" in device_probe
    assert (
        "wafer_dmc_concat_materialized(2600U, 1300U, "
        "5460U, 2730U, 2U)"
    ) in device_probe
    validate_resource_canaries()
    assert all(
        case.opcode == 135
        for case in catalog.CATALOG
        if case.operation == "concat"
    )
    for channels in (63, 64, 65, 127, 129):
        rows = tuple(
            row
            for row in catalog.CALIBRATION_LEAF_BINDINGS[
                "cx-ncx-channel-boundaries"
            ]
            if row.destination_shape == (2, 7, 9, channels)
        )
        assert len(rows) == 4
        assert {row.operation for row in rows} == {
            "tensor-to-cx",
            "cx-to-tensor",
            "tensor-to-ncx",
            "ncx-to-tensor",
        }
    allowed_leaf_objects = set(catalog.CATALOG)
    allowed_leaf_objects.update(catalog.PUBLIC_DISPOSITIONS)
    allowed_leaf_objects.update(catalog.INSTRUCTION_LAYOUT_DISPOSITIONS)
    allowed_leaf_objects.update(catalog.RAW_CONCAT_DISPOSITIONS)
    allowed_leaf_objects.update(catalog.EXTENDED_DATAMOVE_DISPOSITIONS)
    assert catalog.CALIBRATION_LEAF_BINDINGS
    for key, rows in catalog.CALIBRATION_LEAF_BINDINGS.items():
        assert key
        assert rows
        assert all(row in allowed_leaf_objects for row in rows)
    assert all(
        row.disposition == "board-executable"
        and row.evidence
        and row.reason is None
        for row in catalog.EXTENDED_DATAMOVE_DISPOSITIONS
    )
    assert {
        row.opcode
        for row in catalog.PUBLIC_DISPOSITIONS
        if row.disposition == "board-observation"
    } == {131, 133, 136, 137}
    print(
        "wafer_datamove_calibration_catalog_test: "
        "base_board_cases=46 public_opcodes=18 layout_combinations=20 "
        "deferred=0 extended_evidence=18 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
