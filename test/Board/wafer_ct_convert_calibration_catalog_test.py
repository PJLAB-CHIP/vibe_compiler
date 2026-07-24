#!/usr/bin/env python3
"""Validate complete CT convert inventory and independent host payloads."""

from __future__ import annotations

import wafer_ct_convert_calibration_catalog as catalog


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
    print(
        "wafer_ct_convert_calibration_catalog_test: "
        f"opcodes=36 cases={len(catalog.CATALOG)} "
        f"safe={len(catalog.SAFE_CASES)} main=8192 tail=8197 passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
