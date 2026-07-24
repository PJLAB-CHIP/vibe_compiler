#!/usr/bin/env python3
"""Validate complete CT convert inventory and independent host payloads."""

from __future__ import annotations

import pathlib
import struct
import tempfile

import wafer_board_ct_convert_calibration_probe_test as runner
import wafer_ct_convert_calibration_catalog as catalog


def validate_resource_canaries() -> None:
    case = catalog.CASES_BY_NAME[
        "ct-op139-convert-int8-fp16-main"
    ]
    built = catalog.build_case_payload(case)
    assert built.expected_output_slot is not None
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
        "OPCODE": case.opcode,
        "SRC_TYPE": catalog.TYPE_CODES[case.source_type],
        "DST_TYPE": catalog.TYPE_CODES[case.destination_type],
        "ELEMENTS": case.elements,
        "INPUT_BYTES": case.input_bytes,
        "RESULT_BYTES": case.result_bytes,
        "OUTPUT_SPAN": case.output_span,
        "ROUNDING": case.rounding_mode,
        "SAMPLE": 0,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "OUTPUT_GUARD_MISMATCHES": 0,
        "CT_INST_DELTA": 1,
        "DOMAIN": case.domain,
        "ZERO_POINT": case.zero_point,
        "DISPOSITION": case.disposition,
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
    assert len(catalog.CATALOG) == 204
    assert len(catalog.SAFE_CASES) == 204
    assert len(catalog.DEFERRED_CASES) == 0
    assert set(case.opcode for case in catalog.CATALOG) == set(range(139, 175))
    assert set(case.shape_name for case in catalog.CATALOG) == {"main", "tail"}
    assert len(catalog.CASES_BY_ID) == len(catalog.CATALOG)
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    route_shapes = {
        (case.opcode, case.shape_name)
        for case in catalog.CATALOG
        if case.domain_name == "NORMAL"
    }
    assert len(route_shapes) == 72

    rounding_sensitive = 0
    special_value_routes = 0
    for case in catalog.SAFE_CASES:
        source, destination = catalog._parse_route(case.opcode)
        assert (case.source_type, case.destination_type) == (
            source,
            destination,
        )
        assert case.elements in (catalog.MAIN_ELEMENTS, catalog.TAIL_ELEMENTS)
        assert case.input_bytes == case.elements * catalog.TYPE_BYTES[source]
        assert (
            case.result_bytes
            == case.elements * catalog.TYPE_BYTES[destination]
        )
        assert case.output_span % 256 == 0
        assert case.output_span >= case.result_bytes
        assert (
            catalog.BODY_OFFSET + max(case.input_bytes, case.output_span)
            <= catalog.SLOT_BYTES
        )
        built = catalog.build_case_payload(case)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        if case.exact:
            assert built.expected_output_slot is not None
            assert len(built.expected_output_slot) == catalog.SLOT_BYTES
            assert (
                built.expected_output_slot[
                    catalog.BODY_OFFSET + case.result_bytes :
                ]
                == bytes([catalog.SLOT_CANARY])
                * (
                    catalog.SLOT_BYTES
                    - catalog.BODY_OFFSET
                    - case.result_bytes
                )
            )
        else:
            assert built.expected_output_slot is None
        if (
            case.domain_name == "NORMAL"
            and (source.startswith("fp") or source in ("bf16", "tf32"))
        ):
            rounding_sensitive += 1
            if not destination.startswith("int"):
                special_value_routes += 1

    assert rounding_sensitive == 48
    assert special_value_routes == 24
    assert {
        case.rounding_mode
        for case in catalog.CATALOG
        if case.domain_name == "DIRECTED"
    } == {
        catalog.RND_ZERO,
        catalog.RND_POS_INF,
        catalog.RND_NEG_INF,
    }
    directed_by_opcode = {
        opcode: tuple(
            case
            for case in catalog.CATALOG
            if case.domain_name == "DIRECTED" and case.opcode == opcode
        )
        for opcode in catalog.ROUNDING_OPCODES
    }
    for opcode, cases in directed_by_opcode.items():
        outputs = {
            catalog.build_case_payload(case).expected_output_slot
            for case in cases
        }
        if opcode in catalog.ROUNDING_INVARIANT_OPCODES:
            assert len(outputs) == 1
        else:
            assert len(outputs) == len(cases), (
                f"opcode {opcode} directed-rounding vectors do not "
                "distinguish all three modes"
            )
    assert {
        case.opcode
        for case in catalog.CATALOG
        if case.domain_name == "ZERO_POINT"
    } == catalog.ZERO_POINT_OPCODES
    assert all(
        case.reason
        and case.disposition_name == "BOARD_OBSERVED"
        and catalog.build_case_payload(case).expected_output_slot is None
        for case in catalog.CATALOG
        if case.domain_name == "STOCHASTIC"
    )
    bound = tuple(
        case
        for cases in catalog.CALIBRATION_LEAF_BINDINGS.values()
        for case in cases
    )
    assert all(catalog.CALIBRATION_LEAF_BINDINGS.values())
    assert len(bound) == len(set(bound)) == len(catalog.CATALOG)
    assert set(bound) == set(catalog.CATALOG)
    validate_resource_canaries()
    print(
        "wafer_ct_convert_calibration_catalog_test: "
        f"opcodes=36 cases={len(catalog.CATALOG)} "
        f"safe={len(catalog.SAFE_CASES)} main=8192 tail=8197 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
