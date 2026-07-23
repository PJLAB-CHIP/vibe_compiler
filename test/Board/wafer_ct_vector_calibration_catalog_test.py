#!/usr/bin/env python3
"""No-card completeness and oracle checks for CT vector calibration."""

from __future__ import annotations

import collections
import struct

import wafer_ct_vector_calibration_catalog as catalog
import wafer_board_ct_vector_calibration_probe_test as runner


def main() -> int:
    assert len(catalog.OPCODE_NAMES) == 187
    assert len(catalog.CATALOG) == 626
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert collections.Counter(
        case.family_name for case in catalog.CATALOG
    ) == {
        "UNARY": 36,
        "BINARY": 144,
        "RELATION": 288,
        "LOGIC_VALUE": 60,
        "LOGIC_BOOL": 20,
        "TRANSCENDENTAL": 42,
        "ACTIVATION": 36,
    }

    for opcode in range(111):
        cases = tuple(
            case for case in catalog.CATALOG if case.opcode == opcode
        )
        expected_dtypes = (
            {"F16"} if 88 <= opcode <= 97 else set(catalog.DTYPES)
        )
        assert {case.dtype_name for case in cases} == expected_dtypes
        assert {case.shape_name for case in cases} == {"main", "tail"}
        assert {case.elements for case in cases} == {
            catalog.MAIN_ELEMENTS,
            catalog.TAIL_ELEMENTS,
        }
        assert {case.opcode_name for case in cases} == {
            catalog.OPCODE_NAMES[opcode]
        }

    expected_forms = {
        "VV",
        "VS",
        "VuV",
        "VuVLoop",
    }
    assert {
        catalog.form_name(opcode) for opcode in range(6, 30)
    } == expected_forms
    assert {
        catalog.form_name(opcode) for opcode in range(30, 78)
    } == expected_forms
    assert {
        (
            catalog.form_name(case.opcode),
            case.as_dict()["output_storage"],
        )
        for case in catalog.CATALOG
        if case.family_name == "RELATION"
    } == {
        (form, storage)
        for form in expected_forms
        for storage in ("value", "bitpacked-bool")
    }

    for sample, case in enumerate(catalog.CATALOG):
        built = catalog.build_case_payload(case, sample)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        assert len(built.expected_output_slot) == catalog.SLOT_BYTES
        words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert words[catalog.REQ["MAGIC"]] == catalog.REQUEST_MAGIC
        assert words[catalog.REQ["CASE"]] == case.case_id
        assert words[catalog.REQ["OPCODE"]] == case.opcode
        assert words[catalog.REQ["DTYPE"]] == case.dtype
        assert words[catalog.REQ["ELEMENTS"]] == case.elements
        assert words[catalog.REQ["RESULT_BYTES"]] == case.result_bytes
        assert words[catalog.REQ["OUTPUT_SPAN"]] == case.output_span
        assert words[catalog.REQ["SAMPLE"]] == sample
        assert built.input_a_bytes <= catalog.SLOT_BYTES
        assert built.input_b_bytes <= catalog.SLOT_BYTES
        if case.disposition_name == "BOARD_OBSERVED":
            assert built.expected_result is None
        else:
            assert built.expected_result is not None
            assert len(built.expected_result) == case.result_bytes

    vv = catalog.CASES_BY_NAME[
        "ct-op014-arithop_v_vv_add-f16-main"
    ]
    vs = catalog.CASES_BY_NAME[
        "ct-op015-arithop_v_vs_add-f16-main"
    ]
    vuv = catalog.CASES_BY_NAME[
        "ct-op016-arithop_v_vuv_add-f16-main"
    ]
    loop = catalog.CASES_BY_NAME[
        "ct-op017-arithop_v_vuv_add_loop-f16-tail"
    ]
    vv_payload = catalog.build_case_payload(vv)
    vs_payload = catalog.build_case_payload(vs)
    vuv_payload = catalog.build_case_payload(vuv)
    loop_payload = catalog.build_case_payload(loop)
    assert vv_payload.input_b_bytes == catalog.MAIN_ELEMENTS * 2
    assert vs_payload.input_b_bytes == 0
    assert vs_payload.scalar_bits == 0x4000
    assert vuv_payload.input_b_bytes == catalog.UNIT_ELEMENTS * 2
    assert loop_payload.input_b_bytes == catalog.UNIT_ELEMENTS * 2
    assert loop.elements == catalog.TAIL_ELEMENTS
    assert len({
        vv_payload.expected_result,
        vs_payload.expected_result,
        vuv_payload.expected_result,
        loop_payload.expected_result,
    }) == 4

    bool_tail = catalog.CASES_BY_NAME[
        "ct-op097-logicop_bv_bvubv_xor_loop-f16-tail"
    ]
    bool_payload = catalog.build_case_payload(bool_tail)
    assert bool_tail.result_bytes == (catalog.TAIL_ELEMENTS + 7) // 8
    assert bool_payload.expected_result is not None
    unused = 8 - catalog.TAIL_ELEMENTS % 8
    assert bool_payload.expected_result[-1] >> (8 - unused) == 0

    runner.configure_package_support()
    assert runner.package_support.catalog is catalog
    assert runner.package_support.PROBE_C == runner.PROBE_C
    print(
        "wafer_ct_vector_calibration_catalog_test: "
        f"opcodes=111 cases={len(catalog.CATALOG)} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
