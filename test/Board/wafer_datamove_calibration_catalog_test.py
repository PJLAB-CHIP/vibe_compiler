#!/usr/bin/env python3
"""Validate DataMove board cases and all opcode 121..138 dispositions."""

from __future__ import annotations

import wafer_datamove_calibration_catalog as catalog


def main() -> int:
    assert len(catalog.CATALOG) == 15
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
    } == {122, 136, 137}
    for row in catalog.PUBLIC_DISPOSITIONS:
        if row.disposition == "board-executable":
            assert row.evidence
            assert row.reason is None
        else:
            assert row.reason

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
    } == operations
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
    print(
        "wafer_datamove_calibration_catalog_test: "
        "board_cases=15 public_opcodes=18 layout_combinations=20 "
        "deferred=3 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
