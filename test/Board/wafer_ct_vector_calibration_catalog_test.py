#!/usr/bin/env python3
"""No-card completeness and oracle checks for CT vector calibration."""

from __future__ import annotations

import collections
import contextlib
import io
import pathlib
import struct
import tempfile
import types

import wafer_ct_vector_calibration_catalog as catalog
import wafer_board_ct_vector_calibration_probe_test as runner


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    assert len(catalog.OPCODE_NAMES) == 187
    assert len(catalog.CATALOG) == 653
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert catalog.DTYPES == {
        "F16": 0,
        "BF16": 1,
        "F32": 2,
        "BOOL": 3,
    }
    assert catalog.HARDWARE_FORMATS == {
        "F16": 2,
        "BF16": 3,
        "F32": 5,
        "BOOL": 7,
    }
    protocol = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_ct_vector_calibration_probe_protocol.h"
    ).read_text()
    for dtype_name, wire_id in catalog.DTYPES.items():
        assert f"WAFER_CTV_{dtype_name} = {wire_id}" in protocol
    for name, value in (
        ("MAIN_ELEMENTS", catalog.MAIN_ELEMENTS),
        ("TAIL_ELEMENTS", catalog.TAIL_ELEMENTS),
        ("MAIN_UNIT_ELEMENTS", catalog.MAIN_UNIT_ELEMENTS),
        ("TAIL_UNIT_ELEMENTS", catalog.TAIL_UNIT_ELEMENTS),
        ("VUVLOOP_UNIT_ELEMENTS", catalog.VUVLOOP_UNIT_ELEMENTS),
        ("MAIN_LOOP_ELEMENTS", catalog.MAIN_LOOP_ELEMENTS),
        (
            "TAIL_LOOP_FULL_ELEMENTS",
            catalog.TAIL_LOOP_FULL_ELEMENTS,
        ),
        ("TAIL_LOOP_ELEMENTS", catalog.TAIL_LOOP_ELEMENTS),
    ):
        assert f"#define WAFER_CTV_{name} {value}U" in protocol
    assert (
        "#define WAFER_CTV_SPM_OUTPUT UINT64_C(0x50000)"
        in protocol
    )
    assert "WAFER_CTV_SPM_SCRATCH" not in protocol
    probe = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_ct_vector_calibration_probe.c"
    ).read_text()
    assert "_Static_assert(Fmt_BOOL == 7" in probe
    assert "case WAFER_CTV_BOOL:\n    return Fmt_BOOL;" in probe
    assert "if (selected->full_elements != 0)" in probe
    assert probe.count("instruction.param.full_elem_count =") == 1
    assert probe.count("instruction.param.full_unit_elem_count =") == 1
    assert (
        "is_loop != 0 ? WAFER_CTV_TAIL_LOOP_FULL_ELEMENTS\n"
        "                                        : WAFER_CTV_TAIL_ELEMENTS"
        in probe
    )
    assert "uint32_t shape = elements != WAFER_CTV_MAIN_ELEMENTS;" in probe
    issue = probe[
        probe.index("static uint64_t wafer_ctv_issue")
        : probe.index("static void wafer_ctv_init_record")
    ]
    entry = probe[
        probe.index("wafer_tx81_instruction_family_probe(uint64_t request_ddr")
        :
    ]
    assert "get_spm_memory_mapping" not in probe
    assert "wafer_ctv_fill_output" not in probe
    assert "wafer_ctv_guard_mismatches" not in probe
    assert (
        "static uint64_t wafer_ctv_issue(const WaferCTVCase *selected)"
        in issue
    )
    assert (
        "uint32_t output = (uint32_t)(WAFER_CTV_SPM_OUTPUT + "
        "WAFER_CTV_BODY_OFFSET);"
        in issue
    )
    assert "return TsmExecute(&instruction);" in issue
    assert "TsmWaitfinish" not in issue
    assert "wafer_tx81_ncc_join" not in issue
    entry_parse = entry.index(
        "wafer_ctv_invalidate(request_ddr, WAFER_CTV_RESOURCE_BYTES);"
    )
    first_rdma = entry.index("wafer_tx81_rdma(")
    assert entry_parse < first_rdma
    assert "WAFER_CTV_SPM_SCRATCH" not in entry
    observed = entry.index(
        "uint64_t execute_result = wafer_ctv_issue(&selected);",
        first_rdma,
    )
    assert probe.count("wafer_ctv_issue(") == 2
    assert first_rdma < observed
    second_rdma = entry.index(
        "wafer_tx81_rdma(payload_ddr + WAFER_CTV_SLOT_BYTES,"
    )
    canary_rdma = entry.index(
        "wafer_tx81_rdma(request_ddr + WAFER_CTV_SLOT_BYTES,"
    )
    assert second_rdma < canary_rdma < observed
    assert "wafer_tx81_ncc_join(1U);" not in entry[first_rdma:observed]
    wdma = entry.index("wafer_tx81_wdma(", observed)
    output_drain = entry.index("wafer_tx81_ncc_join(1U);", wdma)
    assert wdma < output_drain
    assert "wafer_tx81_ncc_join(1U);" not in entry[observed:wdma]
    assert entry.count("wafer_tx81_ncc_join(1U);") == 1

    selected_for_continue = catalog.CATALOG[:2]
    original_board_command = runner.package_support.board_command
    original_run = runner.package_support.run
    original_validate_output = runner.validate_output
    run_calls: list[object] = []
    validation_calls: list[str] = []

    def fake_board_command(*args: object) -> list[str]:
        return ["wafer-run"]

    def fake_run(command: object, timeout_seconds: float) -> object:
        run_calls.append((command, timeout_seconds))
        return types.SimpleNamespace(
            stdout="\n".join(
                (
                    "board_stage: completion",
                    "board_stage: device-to-host",
                    "board_stage: cleanup",
                    "board_execution: true",
                )
            )
        )

    def fake_validate_output(
        path: pathlib.Path,
        case: catalog.CTVectorCase,
        built: catalog.CasePayload,
        sample: int,
    ) -> dict[str, object]:
        del path, built
        validation_calls.append(case.name)
        if sample == 0:
            raise RuntimeError(f"{case.name}: synthetic oracle failure")
        return {"case": case.name}

    runner.package_support.board_command = fake_board_command
    runner.package_support.run = fake_run
    runner.validate_output = fake_validate_output
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = types.SimpleNamespace(
                work_dir=pathlib.Path(temp_dir),
                completion_timeout_ms=100,
                continue_on_validation_failure=True,
            )
            stderr = io.StringIO()
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr), (
                    contextlib.redirect_stdout(stdout)
                ):
                    runner.execute_cases(
                        args,
                        pathlib.Path("synthetic.package"),
                        (1, 2, 3),
                        selected_for_continue,
                    )
            except RuntimeError as error:
                assert "CT vector validation failures" in str(error)
                assert selected_for_continue[0].name in str(error)
            else:
                raise AssertionError("validation failures stopped aggregating")
            assert len(run_calls) == len(selected_for_continue)
            assert validation_calls == [
                case.name for case in selected_for_continue
            ]
            assert "ct_vector_calibration_failure:" in stderr.getvalue()
            assert "ct_vector_calibration:" in stdout.getvalue()

        run_calls.clear()
        validation_calls.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            args = types.SimpleNamespace(
                work_dir=pathlib.Path(temp_dir),
                completion_timeout_ms=100,
                continue_on_validation_failure=False,
            )
            try:
                runner.execute_cases(
                    args,
                    pathlib.Path("synthetic.package"),
                    (1, 2, 3),
                    selected_for_continue,
                )
            except RuntimeError as error:
                assert "synthetic oracle failure" in str(error)
            else:
                raise AssertionError("default validation failure did not stop")
            assert len(run_calls) == 1
            assert validation_calls == [selected_for_continue[0].name]

        run_calls.clear()
        validation_calls.clear()

        def fake_incomplete_run(
            command: object, timeout_seconds: float
        ) -> object:
            run_calls.append((command, timeout_seconds))
            return types.SimpleNamespace(stdout="board_stage: completion")

        runner.package_support.run = fake_incomplete_run
        with tempfile.TemporaryDirectory() as temp_dir:
            args = types.SimpleNamespace(
                work_dir=pathlib.Path(temp_dir),
                completion_timeout_ms=100,
                continue_on_validation_failure=True,
            )
            try:
                runner.execute_cases(
                    args,
                    pathlib.Path("synthetic.package"),
                    (1, 2, 3),
                    selected_for_continue,
                )
            except RuntimeError as error:
                assert "omitted complete lifecycle evidence" in str(error)
            else:
                raise AssertionError("lifecycle failure was incorrectly caught")
            assert len(run_calls) == 1
            assert validation_calls == []
    finally:
        runner.package_support.board_command = original_board_command
        runner.package_support.run = original_run
        runner.validate_output = original_validate_output

    assert collections.Counter(
        case.family_name for case in catalog.CATALOG
    ) == {
        "UNARY": 54,
        "BINARY": 153,
        "RELATION": 288,
        "LOGIC_VALUE": 60,
        "LOGIC_BOOL": 20,
        "TRANSCENDENTAL": 42,
        "ACTIVATION": 36,
    }

    for opcode in range(111):
        cases = tuple(
            case
            for case in catalog.CATALOG
            if case.opcode == opcode and case.domain_name == "NORMAL"
        )
        expected_dtypes = (
            {"BOOL"} if 88 <= opcode <= 97 else set(catalog.FLOAT_DTYPES)
        )
        assert {case.dtype_name for case in cases} == expected_dtypes
        assert {case.shape_name for case in cases} == {"main", "tail"}
        expected_elements = {
            catalog.MAIN_ELEMENTS,
            (
                catalog.TAIL_LOOP_FULL_ELEMENTS
                if catalog.form_name(opcode) == "VuVLoop"
                else catalog.TAIL_ELEMENTS
            ),
        }
        assert {case.elements for case in cases} == expected_elements
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
    bool_cases = tuple(
        case
        for case in catalog.CATALOG
        if case.family_name == "LOGIC_BOOL"
    )
    assert len(bool_cases) == 20
    assert {case.dtype_name for case in bool_cases} == {"BOOL"}
    assert {case.dtype for case in bool_cases} == {catalog.DTYPES["BOOL"]}
    assert {case.as_dict()["dtype"] for case in bool_cases} == {"bool"}
    assert all(
        case.result_bytes == (case.elements + 7) // 8
        for case in bool_cases
    )
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

    for case in (
        case for case in catalog.CATALOG if catalog._is_vuv(case.opcode)
    ):
        if catalog._is_loop(case.opcode):
            assert case.unit_elements == catalog.VUVLOOP_UNIT_ELEMENTS
            assert case.full_elements == case.elements
            assert case.full_elements % case.elem_count == 0
            assert case.elem_count % case.unit_elements == 0
            assert case.full_unit_elements % case.unit_elements == 0
            assert (
                case.full_elements * case.unit_elements
                == case.elem_count * case.full_unit_elements
            )
            assert case.full_elements // case.elem_count > 1
            assert case.rhs_elements == case.full_unit_elements
        else:
            assert 1 <= case.unit_elements <= 64
            assert case.elem_count % case.unit_elements == 0
            assert case.elem_count == case.elements
            assert case.full_elements == 0
            assert case.full_unit_elements == 0
            assert case.rhs_elements == case.unit_elements

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
        assert words[catalog.REQ["DOMAIN"]] == case.domain
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
    assert catalog.decode_values(
        vv.dtype_name, vv_payload.expected_result or b""
    )[:8] == (-6.0, -5.0, 3.0, 1.0, 0.0, 2.5, 11.0, 4.0)
    assert vs_payload.input_b_bytes == 0
    assert vs_payload.scalar_bits == 0x4000
    assert vuv.elem_count == catalog.MAIN_ELEMENTS
    assert vuv.unit_elements == catalog.MAIN_UNIT_ELEMENTS
    assert vuv.full_elements == vuv.full_unit_elements == 0
    assert (
        vuv_payload.input_b_bytes
        == catalog.MAIN_UNIT_ELEMENTS * 2
    )
    assert loop.elem_count == catalog.TAIL_LOOP_ELEMENTS
    assert loop.unit_elements == catalog.VUVLOOP_UNIT_ELEMENTS
    assert loop.full_elements == catalog.TAIL_LOOP_FULL_ELEMENTS
    assert (
        loop.full_unit_elements
        == catalog.TAIL_LOOP_FULL_ELEMENTS
        // catalog.TAIL_LOOP_ELEMENTS
        * catalog.VUVLOOP_UNIT_ELEMENTS
    )
    assert loop_payload.input_b_bytes == loop.full_unit_elements * 2
    assert loop.elements == catalog.TAIL_LOOP_FULL_ELEMENTS
    loop_rhs = catalog.decode_values(
        loop.dtype_name,
        loop_payload.payload[
            catalog.SLOT_BYTES :
            catalog.SLOT_BYTES + loop_payload.input_b_bytes
        ],
    )
    outer_units = tuple(
        loop_rhs[begin : begin + loop.unit_elements]
        for begin in range(0, len(loop_rhs), loop.unit_elements)
    )
    assert len(outer_units) == len(set(outer_units)) == (
        loop.full_elements // loop.elem_count
    )
    loop_lhs = catalog.decode_values(
        loop.dtype_name,
        loop_payload.payload[: loop_payload.input_a_bytes],
    )
    loop_expected = catalog.decode_values(
        loop.dtype_name, loop_payload.expected_result or b""
    )
    for index in (
        0,
        loop.elem_count - 1,
        loop.elem_count,
        loop.elements - 1,
    ):
        assert loop_expected[index] == (
            loop_lhs[index]
            + loop_rhs[catalog._rhs_index(loop, index)]
        )
    record_words = [0] * catalog.RECORD_WORDS
    for field, value in (
        ("MAGIC", catalog.RECORD_MAGIC),
        (
            "WORD_COUNT",
            catalog.RECORD_WORDS,
        ),
        ("STATUS", 0),
        ("CASE", loop.case_id),
        ("DISPOSITION", loop.disposition),
        ("FAMILY", loop.family),
        ("DTYPE", loop.dtype),
        ("OPCODE", loop.opcode),
        ("RESULT_BYTES", loop.result_bytes),
        ("OUTPUT_SPAN", loop.output_span),
        ("ELEMENTS", loop.elements),
        ("SAMPLE", 0),
        ("OUTPUT_GUARD_MISMATCHES", 0),
        ("EXECUTE_RESULT", 1),
        ("REQUEST_GUARD", catalog.REQUEST_GUARD),
        ("OUTPUT_DDR_OFFSET", catalog.OUTPUT_DDR_OFFSET),
        ("SLOT_BYTES", catalog.SLOT_BYTES),
        ("BODY_OFFSET", catalog.BODY_OFFSET),
        ("INPUT_A_BYTES", loop_payload.input_a_bytes),
        ("INPUT_B_BYTES", loop_payload.input_b_bytes),
        ("SCALAR_BITS", loop_payload.scalar_bits),
        ("UNIT_ELEMENTS", loop.unit_elements),
        ("DOMAIN", loop.domain),
        ("ELEM_COUNT", loop.elem_count),
        ("FULL_ELEMENTS", loop.full_elements),
        ("FULL_UNIT_ELEMENTS", loop.full_unit_elements),
        ("RECORD_GUARD", catalog.RECORD_GUARD),
    ):
        record_words[catalog.REC[field]] = value
    raw_record = bytearray(catalog.RESOURCE_BYTES)
    struct.pack_into(
        f"<{catalog.RECORD_WORDS}Q", raw_record, 0, *record_words
    )
    runner._validate_record(bytes(raw_record), loop, loop_payload, 0)
    record_words[catalog.REC["FULL_UNIT_ELEMENTS"]] += 1
    struct.pack_into(
        f"<{catalog.RECORD_WORDS}Q", raw_record, 0, *record_words
    )
    try:
        runner._validate_record(bytes(raw_record), loop, loop_payload, 0)
    except RuntimeError as error:
        assert "record/execute mirror failed" in str(error)
    else:
        raise AssertionError("raw loop-count mirror stopped being enforced")
    assert len({
        vv_payload.expected_result,
        vs_payload.expected_result,
        vuv_payload.expected_result,
        loop_payload.expected_result,
    }) == 4

    for case in (
        case
        for case in catalog.CATALOG
        if case.domain_name == "NORMAL"
        and case.opcode in (30, 31, 38, 39)
    ):
        built = catalog.build_case_payload(case)
        lhs = catalog.decode_values(
            case.dtype_name,
            built.payload[: built.input_a_bytes],
        )
        rhs = catalog.decode_values(
            case.dtype_name,
            built.payload[
                catalog.SLOT_BYTES :
                catalog.SLOT_BYTES + built.input_b_bytes
            ],
        )
        equal_lanes = tuple(
            left == right for left, right in zip(lhs, rhs, strict=True)
        )
        assert set(equal_lanes) == {False, True}
        expected_truth = tuple(
            value if case.opcode < 38 else not value
            for value in equal_lanes
        )
        assert built.expected_result is not None
        if catalog._is_bool_output(case.opcode):
            actual_truth = tuple(
                bool(
                    (
                        built.expected_result[index // 8]
                        >> (index % 8)
                    )
                    & 1
                )
                for index in range(case.elements)
            )
        else:
            actual_truth = tuple(
                bool(value)
                for value in catalog.decode_values(
                    case.dtype_name, built.expected_result
                )
            )
        assert actual_truth == expected_truth

    logic_value_cases = tuple(
        case
        for case in catalog.CATALOG
        if case.family_name == "LOGIC_VALUE"
    )
    assert len(logic_value_cases) == 60
    assert {
        catalog.form_name(case.opcode) for case in logic_value_cases
    } == {"V", "VV", "VuV", "VuVLoop"}
    for case in logic_value_cases:
        built = catalog.build_case_payload(case)
        assert built.expected_result is not None
        lhs = catalog.decode_values(
            case.dtype_name,
            built.payload[: built.input_a_bytes],
        )
        if case.opcode == 78:
            expected_truth = tuple(not bool(value) for value in lhs)
        else:
            rhs = catalog.decode_values(
                case.dtype_name,
                built.payload[
                    catalog.SLOT_BYTES :
                    catalog.SLOT_BYTES + built.input_b_bytes
                ],
            )
            function = (
                lambda left, right: left and right,
                lambda left, right: left or right,
                lambda left, right: left != right,
            )[(case.opcode - 79) % 3]
            expected_truth = tuple(
                function(
                    bool(left),
                    bool(rhs[catalog._rhs_index(case, index)]),
                )
                for index, left in enumerate(lhs)
            )
        actual_values = catalog.decode_values(
            case.dtype_name, built.expected_result
        )
        assert actual_values == tuple(
            float(value) for value in expected_truth
        )

    logic_value_negative = catalog.CASES_BY_NAME[
        "ct-op078-logicop_v_v_not-f16-tail"
    ]
    logic_value_expected = catalog.build_case_payload(
        logic_value_negative
    ).expected_result
    assert logic_value_expected is not None
    wrong_logic_value = bytearray(logic_value_expected)
    wrong_logic_value[0] ^= 1
    try:
        runner._validate_numeric(
            logic_value_negative,
            bytes(wrong_logic_value),
            logic_value_expected,
        )
    except RuntimeError as error:
        assert "exact result differs at byte 0" in str(error)
    else:
        raise AssertionError("logic-value exact oracle stopped rejecting errors")

    bool_tail = catalog.CASES_BY_NAME[
        "ct-op097-logicop_bv_bvubv_xor_loop-bool-tail"
    ]
    bool_payload = catalog.build_case_payload(bool_tail)
    assert bool_tail.result_bytes == (bool_tail.elements + 7) // 8
    assert bool_payload.expected_result is not None
    unused = (-bool_tail.elements) % 8
    if unused:
        assert bool_payload.expected_result[-1] >> (8 - unused) == 0

    for opcode in (92, 93, 94):
        bitpacked_vuv_tail = next(
            case
            for case in catalog.CATALOG
            if case.opcode == opcode and case.shape_name == "tail"
        )
        built = catalog.build_case_payload(bitpacked_vuv_tail)
        expected = built.expected_result
        assert expected is not None
        assert bitpacked_vuv_tail.unit_elements == 37
        assert built.input_b_bytes == 5
        rhs = built.payload[
            catalog.SLOT_BYTES :
            catalog.SLOT_BYTES + built.input_b_bytes
        ]
        assert rhs[-1] & 0b1110_0000 == 0
        assert catalog._bitpacked_vuv_rhs_index(
            bitpacked_vuv_tail, 36
        ) == 36
        assert catalog._bitpacked_vuv_rhs_index(
            bitpacked_vuv_tail, 37
        ) == 37
        assert catalog._bitpacked_vuv_rhs_index(
            bitpacked_vuv_tail, 39
        ) == 39
        assert catalog._bitpacked_vuv_rhs_index(
            bitpacked_vuv_tail, 40
        ) == 0
        covered = bitpacked_vuv_tail.elements // 40 * 40
        assert covered == 8200
        assert catalog._bitpacked_vuv_rhs_index(
            bitpacked_vuv_tail, covered - 1
        ) == 39
        assert catalog._bitpacked_vuv_rhs_index(
            bitpacked_vuv_tail, covered
        ) is None
        lhs = built.payload[: built.input_a_bytes]
        for index in range(covered, bitpacked_vuv_tail.elements):
            left = (lhs[index // 8] >> (index % 8)) & 1
            actual = (expected[index // 8] >> (index % 8)) & 1
            assert actual == (
                0 if opcode == 92 else left
            )

    relation_bool_tail = catalog.CASES_BY_NAME[
        "ct-op031-relaop_bv_vv_eq-f16-tail"
    ]
    relation_bool_payload = catalog.build_case_payload(relation_bool_tail)
    relation_bool_expected = relation_bool_payload.expected_result
    assert relation_bool_expected is not None
    relation_bool_actual = bytearray(relation_bool_expected)
    valid_tail_bits = relation_bool_tail.elements % 8
    valid_mask = (1 << valid_tail_bits) - 1
    relation_bool_actual[-1] = (
        relation_bool_actual[-1] & valid_mask
    ) | (catalog.SLOT_CANARY & (0xFF ^ valid_mask))
    assert runner._validate_numeric(
        relation_bool_tail,
        bytes(relation_bool_actual),
        relation_bool_expected,
    ) == {"mismatches": 0}
    wrong_bool_value = bytearray(relation_bool_actual)
    wrong_bool_value[-1] ^= 1
    try:
        runner._validate_numeric(
            relation_bool_tail,
            bytes(wrong_bool_value),
            relation_bool_expected,
        )
    except RuntimeError as error:
        assert "exact result differs at byte" in str(error)
    else:
        raise AssertionError("bitpacked bool valid-bit oracle weakened")
    alternate_unused_bits = bytearray(relation_bool_actual)
    alternate_unused_bits[-1] ^= 0x80
    assert runner._validate_numeric(
        relation_bool_tail,
        bytes(alternate_unused_bits),
        relation_bool_expected,
    ) == {"mismatches": 0}
    relation_bool_slot = bytearray(
        [catalog.SLOT_CANARY] * catalog.SLOT_BYTES
    )
    result_begin = catalog.BODY_OFFSET
    result_end = result_begin + relation_bool_tail.result_bytes
    relation_bool_slot[result_begin:result_end] = relation_bool_actual
    runner._validate_output_guard(
        relation_bool_tail, bytes(relation_bool_slot)
    )
    relation_bool_slot[result_end] ^= 1
    try:
        runner._validate_output_guard(
            relation_bool_tail,
            bytes(relation_bool_slot),
        )
    except RuntimeError as error:
        assert f"output guard differs at slot byte {result_end}" in str(error)
    else:
        raise AssertionError("bitpacked bool post-result guard weakened")

    special = catalog.CASES_BY_NAME[
        "ct-op014-arithop_v_vv_add-bf16-special"
    ]
    qnan = catalog.CASES_BY_NAME[
        "ct-op014-arithop_v_vv_add-f32-qnan"
    ]
    negative_sqrt = catalog.CASES_BY_NAME[
        "ct-op003-arithop_v_v_sqrt-f16-sqrt-negative"
    ]
    assert catalog.build_case_payload(special).expected_result is not None
    assert catalog.build_case_payload(qnan).expected_result is None
    assert (
        catalog.build_case_payload(negative_sqrt).expected_result is None
    )

    f16_neg = catalog.CASES_BY_NAME[
        "ct-op005-arithop_v_v_neg-f16-main"
    ]
    f16_neg_signed_zero = catalog.CASES_BY_NAME[
        "ct-op005-arithop_v_v_neg-f16-signed-zero"
    ]
    bf16_neg = catalog.CASES_BY_NAME[
        "ct-op005-arithop_v_v_neg-bf16-main"
    ]
    bf16_neg_signed_zero = catalog.CASES_BY_NAME[
        "ct-op005-arithop_v_v_neg-bf16-signed-zero"
    ]
    f32_neg = catalog.CASES_BY_NAME[
        "ct-op005-arithop_v_v_neg-f32-main"
    ]
    f32_neg_signed_zero = catalog.CASES_BY_NAME[
        "ct-op005-arithop_v_v_neg-f32-signed-zero"
    ]
    assert (
        f16_neg.output_zero_sign_policy_name
        == f16_neg_signed_zero.output_zero_sign_policy_name
        == "CANONICAL_POSITIVE"
    )
    assert (
        f16_neg.as_dict()["output_zero_sign_policy"]
        == "canonical-positive"
    )
    f16_neg_expected = catalog.build_case_payload(f16_neg).expected_result
    f16_signed_zero_expected = catalog.build_case_payload(
        f16_neg_signed_zero
    ).expected_result
    assert f16_neg_expected is not None
    assert f16_signed_zero_expected is not None
    assert f16_neg_expected[6:8] == b"\x00\x00"
    assert runner._validate_numeric(
        f16_neg, f16_neg_expected, f16_neg_expected
    ) == {"mismatches": 0}
    wrong_f16_zero_sign = bytearray(f16_neg_expected)
    wrong_f16_zero_sign[7] = 0x80
    try:
        runner._validate_numeric(
            f16_neg, bytes(wrong_f16_zero_sign), f16_neg_expected
        )
    except RuntimeError as error:
        assert "exact result differs at byte 7" in str(error)
    else:
        raise AssertionError("F16 Neg zero-sign policy stopped being exact")
    assert set(
        struct.unpack(
            f"<{len(f16_signed_zero_expected) // 2}H",
            f16_signed_zero_expected,
        )
    ) == {0}
    assert (
        bf16_neg.output_zero_sign_policy_name
        == bf16_neg_signed_zero.output_zero_sign_policy_name
        == "CANONICAL_POSITIVE"
    )
    assert (
        f32_neg.output_zero_sign_policy_name
        == f32_neg_signed_zero.output_zero_sign_policy_name
        == "CANONICAL_POSITIVE"
    )
    bf16_neg_expected = catalog.build_case_payload(bf16_neg).expected_result
    bf16_signed_zero_expected = catalog.build_case_payload(
        bf16_neg_signed_zero
    ).expected_result
    f32_neg_expected = catalog.build_case_payload(f32_neg).expected_result
    f32_signed_zero_expected = catalog.build_case_payload(
        f32_neg_signed_zero
    ).expected_result
    assert bf16_neg_expected is not None
    assert bf16_signed_zero_expected is not None
    assert f32_neg_expected is not None
    assert f32_signed_zero_expected is not None
    assert bf16_neg_expected[6:8] == b"\x00\x00"
    wrong_bf16_zero_sign = bytearray(bf16_neg_expected)
    wrong_bf16_zero_sign[7] = 0x80
    try:
        runner._validate_numeric(
            bf16_neg, bytes(wrong_bf16_zero_sign), bf16_neg_expected
        )
    except RuntimeError as error:
        assert "exact result differs at byte 7" in str(error)
    else:
        raise AssertionError("BF16 Neg zero-sign policy stopped being exact")
    assert set(
        struct.unpack(
            f"<{len(bf16_signed_zero_expected) // 2}H",
            bf16_signed_zero_expected,
        )
    ) == {0}
    assert f32_neg_expected[12:16] == b"\x00\x00\x00\x00"
    wrong_f32_zero_sign = bytearray(f32_neg_expected)
    wrong_f32_zero_sign[15] = 0x80
    try:
        runner._validate_numeric(
            f32_neg, bytes(wrong_f32_zero_sign), f32_neg_expected
        )
    except RuntimeError as error:
        assert "exact result differs at byte 15" in str(error)
    else:
        raise AssertionError("F32 Neg zero-sign policy stopped being exact")
    assert set(
        struct.unpack(
            f"<{len(f32_signed_zero_expected) // 4}I",
            f32_signed_zero_expected,
        )
    ) == {0}

    bound = tuple(
        case
        for cases in catalog.CALIBRATION_LEAF_BINDINGS.values()
        for case in cases
    )
    assert all(catalog.CALIBRATION_LEAF_BINDINGS.values())
    assert len(bound) == len(set(bound)) == len(catalog.CATALOG)
    assert set(bound) == set(catalog.CATALOG)

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
