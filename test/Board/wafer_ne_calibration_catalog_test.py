#!/usr/bin/env python3
"""Validate all prepared large/tail/batch NE calibration payloads."""

from __future__ import annotations

import pathlib
import struct
import tempfile
from collections import Counter

import wafer_board_ne_calibration_probe_test as runner
import wafer_ne_calibration_catalog as catalog
import wafer_physical_tensor_codec as physical


def validate_pure_ncc_probe() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    probe = (
        repo
        / "test"
        / "Board"
        / "Inputs"
        / "wafer_ne_calibration_probe.c"
    ).read_text()
    entry = probe[
        probe.index("wafer_tx81_instruction_family_probe(uint64_t request_ddr")
        :
    ]
    assert "get_spm_memory_mapping" not in probe
    assert "wafer_nec_output_guard_mismatches" not in probe
    assert "TsmWaitfinish" not in probe

    first_rdma = entry.index("wafer_tx81_rdma_v3(payload_ddr,")
    output_seed = entry.index(
        "wafer_tx81_rdma_v3(payload_ddr + 2U * WAFER_NEC_SLOT_BYTES,"
    )
    aux_rdma = entry.index(
        "wafer_tx81_rdma_v3(payload_ddr + 3U * WAFER_NEC_SLOT_BYTES,"
    )
    pmu_before = entry.index(
        "WaferNECPMU before = wafer_nec_read_pmu();", aux_rdma
    )
    execute = entry.index("uint64_t execute_result = 0U;", pmu_before)
    wdma = entry.index("wafer_tx81_wdma_v3(", execute)
    terminal_fence = entry.index("wafer_tx81_ncc_join(1U);", wdma)
    pmu_after = entry.index(
        "WaferNECPMU after = wafer_nec_read_pmu();", terminal_fence
    )
    assert first_rdma < output_seed < aux_rdma < execute
    assert execute < wdma < terminal_fence
    assert aux_rdma < pmu_before < execute < pmu_after
    assert entry.count("wafer_tx81_ncc_join(1U);") == 1
    assert "GR_PMU_NE_INST_NUMS" in probe
    assert "GR_PMU_NE_BLOCKING_TIME" in probe
    assert "GR_PMU_NE_EXE_TIME" in probe

    runner_source = pathlib.Path(runner.__file__).read_text()
    assert "OUTPUT_GUARD_MISMATCHES" not in runner_source


def _observation_raw(
    case: catalog.NECase,
    built: catalog.CasePayload,
    *,
    sample: int,
    actual_physical: bytes,
) -> bytearray:
    assert len(actual_physical) == case.output_span
    raw = bytearray(
        [runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES
    )
    record = [0] * catalog.RECORD_WORDS
    values = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": case.case_id,
        "DTYPE": case.dtype,
        "LHS_ORIENTATION": case.lhs_orientation,
        "RHS_ORIENTATION": case.rhs_orientation,
        "BATCH": case.batch,
        "M": case.m,
        "K": case.k,
        "N": case.n,
        "LHS_SPAN": case.lhs_span,
        "RHS_SPAN": case.rhs_span,
        "OUTPUT_SPAN": case.output_span,
        "SAMPLE": sample,
        "EXECUTE_RESULT": 1,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "KIND": case.kind,
        "PROFILE": case.profile,
        "OPTION": case.option,
        "AUX_SPAN": case.aux_span,
        "DISPOSITION": case.disposition,
        "LHS_BATCH": case.lhs_batch,
        "RHS_BATCH": case.rhs_batch,
        "PMU_ENABLE": 1,
        "NE_INST_DELTA": 1,
        "NE_BLOCKING_DELTA": 0,
        "NE_EXEC_DELTA": 123,
        "PMU_BASE": catalog.PMU_BASE,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    for name, value in values.items():
        record[catalog.REC[name]] = value
    struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *record)
    output_slot = built.payload[
        2 * catalog.SLOT_BYTES : 3 * catalog.SLOT_BYTES
    ]
    begin = catalog.OUTPUT_DDR_OFFSET
    raw[begin : begin + catalog.SLOT_BYTES] = output_slot
    physical_begin = begin + catalog.BODY_OFFSET
    raw[
        physical_begin : physical_begin + case.output_span
    ] = actual_physical
    return raw


def main() -> int:
    validate_pure_ncc_probe()
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
        raise AssertionError("zero NE observation sample count was accepted")
    assert len(catalog.SAFE_CASES) == 70
    assert len(catalog.CATALOG) == 73
    assert len(catalog.CASES_BY_NAME) == len(catalog.CATALOG)
    assert Counter(case.disposition_name for case in catalog.CATALOG) == {
        "BOARD_EXACT": 32,
        "BOARD_OBSERVED": 38,
        "STATIC_NEGATIVE": 3,
    }
    coverage = {
        (
            case.dtype_name,
            case.geometry_name,
            case.orientation_name,
        )
        for case in catalog.SAFE_CASES
        if case.kind_name == "GEMM"
        and case.profile_name == "DENSE"
        and case.option_name == "NONE"
    }
    assert coverage == {
        (dtype, geometry, orientation)
        for dtype in catalog.FLOAT_DTYPES
        for geometry in catalog.GEOMETRIES
        for orientation in catalog.ORIENTATIONS
    }
    for sample, case in enumerate(catalog.SAFE_CASES):
        built = catalog.build_case_payload(case, sample)
        assert len(built.request) == catalog.RESOURCE_BYTES
        assert len(built.payload) == catalog.RESOURCE_BYTES
        words = struct.unpack_from(
            f"<{catalog.REQUEST_WORDS}Q", built.request
        )
        assert words[catalog.REQ["CASE"]] == case.case_id
        assert words[catalog.REQ["SAMPLE"]] == sample
        assert words[catalog.REQ["KIND"]] == case.kind
        assert words[catalog.REQ["PROFILE"]] == case.profile
        assert words[catalog.REQ["OPTION"]] == case.option
        assert words[catalog.REQ["AUX_SPAN"]] == case.aux_span
        assert words[catalog.REQ["DISPOSITION"]] == case.disposition
        assert words[catalog.REQ["BATCH_PAIR"]] == (
            (case.lhs_batch << 32) | case.rhs_batch
        )
        assert words[catalog.REQ["GUARD"]] == catalog.REQUEST_GUARD
        assert max(case.lhs_span, case.rhs_span, case.output_span) <= (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET
        )
        if case.exact:
            assert built.expected_logical is not None
            assert built.expected_physical is not None
            unpacked = physical.unpack_scalar_bytes(
                case.output_shape,
                case.output_layout,
                case.element_bytes,
                built.expected_physical,
            )
            assert unpacked == built.expected_logical
        else:
            assert built.expected_logical is None
            assert built.expected_physical is None

    tail = catalog.CASES_BY_NAME["ne-f16-tail-nn"]
    tail_built = catalog.build_case_payload(tail)
    assert tail_built.expected_physical is not None
    tail_layout = physical.physical_layout(
        tail.output_shape,
        tail.output_layout,
        tail.element_bytes,
    )
    logical_output_bytes = set()
    for coordinate in physical.coordinates(tail.output_shape):
        begin = (
            physical.physical_element_offset(tail_layout, coordinate)
            * tail.element_bytes
        )
        logical_output_bytes.update(
            range(begin, begin + tail.element_bytes)
        )
    padding_output_bytes = set(range(tail.output_span)) - logical_output_bytes
    assert padding_output_bytes
    assert {
        tail_built.expected_physical[index]
        for index in padding_output_bytes
    } == {catalog.OUTPUT_PADDING}
    output_seed = tail_built.payload[
        2 * catalog.SLOT_BYTES
        + catalog.BODY_OFFSET :
        2 * catalog.SLOT_BYTES
        + catalog.BODY_OFFSET
        + tail.output_span
    ]
    assert {
        output_seed[index] for index in padding_output_bytes
    } == {catalog.SLOT_CANARY}

    for case in catalog.CALIBRATION_LEAF_BINDINGS[
        "ne-gemm-dtype-orientation-main-tail-batch"
    ]:
        lhs, rhs = catalog._gemm_inputs(case)
        assert all(value != 0.0 for value in lhs)
        assert all(value != 0.0 for value in rhs)
        expected = catalog._gemm_expected_values(case, lhs, rhs)
        assert len(expected) == case.batch * case.m * case.n
        for batch in range(case.batch):
            rows = tuple(
                expected[
                    (batch * case.m + row) * case.n :
                    (batch * case.m + row + 1) * case.n
                ]
                for row in range(case.m)
            )
            column_signatures = {
                tuple(rows[row][column] for row in range(case.m))
                for column in range(case.n)
            }
            assert len(column_signatures) == case.n
            assert all(
                len(set(row)) >= min(8, case.n) for row in rows
            )
        if case.batch > 1:
            batch_elements = case.m * case.n
            assert (
                expected[:batch_elements]
                != expected[batch_elements : 2 * batch_elements]
            )
        if (
            case.dtype_name == "BF16"
            and case.geometry_name == "tail"
            and case.orientation_name == "NN"
        ):
            expected_logical = tuple(
                catalog._encode(case.dtype_name, value)
                for value in expected
            )
            permuted_logical = tuple(
                expected_logical[
                    (batch * case.m + row) * case.n
                    + (column + 1) % case.n
                ]
                for batch in range(case.batch)
                for row in range(case.m)
                for column in range(case.n)
            )
            assert physical.pack_scalar_bytes(
                case.output_shape,
                case.output_layout,
                2,
                permuted_logical,
                padding=catalog.SLOT_CANARY,
            ) != physical.pack_scalar_bytes(
                case.output_shape,
                case.output_layout,
                2,
                expected_logical,
                padding=catalog.SLOT_CANARY,
            )

    assert {
        case.profile_name
        for case in catalog.SAFE_CASES
        if case.profile_name.startswith("BF16_")
    } == {
        "BF16_CANCELLATION",
        "BF16_ROUNDING",
        "BF16_SIGNED_ZERO",
        "BF16_SUBNORMAL",
        "BF16_OVERFLOW_INF",
        "BF16_NAN",
    }
    special_cases = {
        case.profile_name: case
        for case in catalog.SAFE_CASES
        if case.profile_name.startswith("BF16_")
    }
    assert {
        name
        for name, case in special_cases.items()
        if case.disposition_name == "BOARD_OBSERVED"
    } == {
        "BF16_SIGNED_ZERO",
        "BF16_SUBNORMAL",
        "BF16_OVERFLOW_INF",
        "BF16_NAN",
    }

    def lhs_bits(profile_name: str) -> set[int]:
        case = special_cases[profile_name]
        built = catalog.build_case_payload(case)
        physical_lhs = built.payload[
            catalog.BODY_OFFSET : catalog.BODY_OFFSET + case.lhs_span
        ]
        logical_lhs = physical.unpack_scalar_bytes(
            case.lhs_shape,
            case.lhs_layout,
            2,
            physical_lhs,
        )
        return {
            struct.unpack("<H", scalar)[0] for scalar in logical_lhs
        }

    assert {0x0000, 0x8000, 0x3F80}.issubset(
        lhs_bits("BF16_SIGNED_ZERO")
    )
    assert {0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080}.issubset(
        lhs_bits("BF16_SUBNORMAL")
    )
    assert {0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x4000, 0xC000}.issubset(
        lhs_bits("BF16_OVERFLOW_INF")
    )
    assert set(catalog.BF16_NAN_INPUT_BITS).issubset(
        lhs_bits("BF16_NAN")
    )
    assert 0x3F80 in lhs_bits("BF16_NAN")
    assert {
        case.option_name
        for case in catalog.SAFE_CASES
        if case.option_name != "NONE"
    } == set(tuple(catalog.OPTIONS)[1:])
    bias_cases = tuple(
        case
        for case in catalog.SAFE_CASES
        if case.option_name == "BIAS"
    )
    assert {
        (case.kind_name, case.dtype_name) for case in bias_cases
    } == {
        (kind, dtype)
        for kind in ("GEMM", "CONV")
        for dtype in catalog.FLOAT_DTYPES
    }
    for case in bias_cases:
        assert case.disposition_name == "BOARD_OBSERVED"
        if case.kind_name == "GEMM":
            assert "base output unchanged" in case.reason
            assert "bias_en" in case.reason
        else:
            assert "NCx/HWOI numeric oracle" in case.reason
        bias_built = catalog.build_case_payload(case)
        assert bias_built.expected_logical is None
        assert bias_built.expected_physical is None
    bias = catalog.CASES_BY_NAME["ne-f16-large-nn-bias"]
    bias_lhs, bias_rhs = catalog._gemm_inputs(bias)
    bias_base = catalog._gemm_expected_values(
        bias, bias_lhs, bias_rhs
    )
    bias_auxiliary, _, _ = catalog._option_auxiliary(bias)
    bias_additive = tuple(
        value + bias_auxiliary[index % bias.n]
        for index, value in enumerate(bias_base)
    )
    assert tuple(
        catalog._encode(bias.dtype_name, value) for value in bias_base
    ) != tuple(
        catalog._encode(bias.dtype_name, value)
        for value in bias_additive
    )
    relu_cases = tuple(
        case
        for case in catalog.SAFE_CASES
        if case.option_name == "RELU"
    )
    assert {
        (case.kind_name, case.dtype_name) for case in relu_cases
    } == {
        (kind, dtype)
        for kind in ("GEMM", "CONV")
        for dtype in catalog.FLOAT_DTYPES
    }
    assert all(
        case.disposition_name == "BOARD_OBSERVED"
        and catalog.build_case_payload(case).expected_logical is None
        and catalog.build_case_payload(case).expected_physical is None
        for case in relu_cases
    )
    for dtype in catalog.FLOAT_DTYPES:
        relu = catalog.CASES_BY_NAME[
            f"ne-{dtype.lower()}-large-nn-relu"
        ]
        relu_lhs, relu_rhs = catalog._gemm_inputs(relu)
        relu_base = catalog._gemm_expected_values(
            relu, relu_lhs, relu_rhs
        )
        assert any(value < 0.0 for value in relu_base)
        encoded_base = tuple(
            catalog._encode(dtype, value) for value in relu_base
        )
        encoded_clamped = tuple(
            catalog._encode(dtype, max(value, 0.0))
            for value in relu_base
        )
        assert encoded_base != encoded_clamped
        assert "retained negative base results" in relu.reason
    conv_cases = tuple(
        case
        for case in catalog.SAFE_CASES
        if case.profile_name in {"CONV_LARGE", "CONV_HELDOUT"}
    )
    assert len(conv_cases) == 16
    assert {case.n for case in conv_cases} == {65, 96}
    assert all(
        case.disposition_name == "BOARD_OBSERVED"
        and "physical indexing" in case.reason
        and catalog.build_case_payload(case).expected_logical is None
        and catalog.build_case_payload(case).expected_physical is None
        for case in conv_cases
    )
    assert all(
        max(case.lhs_span, case.rhs_span, case.output_span)
        <= catalog.SLOT_BYTES - catalog.BODY_OFFSET
        for case in conv_cases
    )

    conv_index_cases = tuple(
        case
        for case in catalog.SAFE_CASES
        if case.profile_name in catalog.CONV_INDEX_PROFILES
    )
    assert tuple(
        case.profile_name for case in conv_index_cases
    ) == catalog.CONV_INDEX_PROFILE_ORDER
    assert all(
        case.dtype_name == "F16"
        and case.option_name == "NONE"
        and case.disposition_name == "BOARD_OBSERVED"
        and case.as_dict()["oracle"]
        == "conv-index-candidates+raw-physical-guard"
        for case in conv_index_cases
    )
    expected_candidate_names = {
        "CONV_FEATURE_INDEX": {"feature:ncx", "feature:cx"},
        "CONV_WEIGHT_INDEX": {"weight:ncx", "weight:cx"},
        "CONV_OUTPUT_INDEX": {"output:ncx", "output:cx"},
    }
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "conv-index-observed.raw"
        for sample, case in enumerate(conv_index_cases, start=40):
            built = catalog.build_case_payload(case, sample=sample)
            candidates = catalog.build_conv_index_candidates(case, built)
            assert set(candidates) == expected_candidate_names[
                case.profile_name
            ]
            assert len(set(candidates.values())) == 2
            for candidate_name, candidate in candidates.items():
                output.write_bytes(
                    _observation_raw(
                        case,
                        built,
                        sample=sample,
                        actual_physical=candidate,
                    )
                )
                result = runner.validate_output(
                    output, case, built, sample=sample
                )
                assert result["candidate_matches"] == (candidate_name,)
                assert (
                    result["candidate_byte_mismatches"][candidate_name]
                    == 0
                )
            unknown = bytearray(next(iter(candidates.values())))
            unknown[0] ^= 0x1
            assert bytes(unknown) not in set(candidates.values())
            output.write_bytes(
                _observation_raw(
                    case,
                    built,
                    sample=sample,
                    actual_physical=bytes(unknown),
                )
            )
            result = runner.validate_output(
                output, case, built, sample=sample
            )
            assert result["candidate_matches"] == ()
            assert all(
                mismatches > 0
                for mismatches in result[
                    "candidate_byte_mismatches"
                ].values()
            )

    quant = catalog.CASES_BY_NAME["ne-gemm-quant-observed"]
    assert quant.dtype_name == "I8"
    assert quant.element_bytes == 1
    assert quant.disposition_name == "BOARD_OBSERVED"
    assert (quant.lhs_span, quant.rhs_span, quant.output_span) == (
        256,
        256,
        256,
    )
    quant_built = catalog.build_case_payload(quant)
    quant_lhs = physical.unpack_scalar_bytes(
        quant.lhs_shape,
        quant.lhs_layout,
        quant.element_bytes,
        quant_built.payload[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + quant.lhs_span
        ],
    )
    quant_rhs = physical.unpack_scalar_bytes(
        quant.rhs_shape,
        quant.rhs_layout,
        quant.element_bytes,
        quant_built.payload[
            catalog.SLOT_BYTES + catalog.BODY_OFFSET :
            catalog.SLOT_BYTES + catalog.BODY_OFFSET + quant.rhs_span
        ],
    )
    assert {-7, 0, 7}.issubset(
        {struct.unpack("<b", value)[0] for value in quant_lhs}
    )
    assert {-6, 0, 6}.issubset(
        {struct.unpack("<b", value)[0] for value in quant_rhs}
    )
    assert quant_built.expected_logical is None
    assert quant_built.expected_physical is None
    raw = bytearray(
        [runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES
    )
    record = [0] * catalog.RECORD_WORDS
    record_values = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": quant.case_id,
        "DTYPE": quant.dtype,
        "LHS_ORIENTATION": quant.lhs_orientation,
        "RHS_ORIENTATION": quant.rhs_orientation,
        "BATCH": quant.batch,
        "M": quant.m,
        "K": quant.k,
        "N": quant.n,
        "LHS_SPAN": quant.lhs_span,
        "RHS_SPAN": quant.rhs_span,
        "OUTPUT_SPAN": quant.output_span,
        "SAMPLE": 4,
        "EXECUTE_RESULT": 1,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "KIND": quant.kind,
        "PROFILE": quant.profile,
        "OPTION": quant.option,
        "AUX_SPAN": quant.aux_span,
        "DISPOSITION": quant.disposition,
        "LHS_BATCH": quant.lhs_batch,
        "RHS_BATCH": quant.rhs_batch,
        "PMU_ENABLE": 1,
        "NE_INST_DELTA": 1,
        "NE_BLOCKING_DELTA": 0,
        "NE_EXEC_DELTA": 123,
        "PMU_BASE": catalog.PMU_BASE,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    for name, value in record_values.items():
        record[catalog.REC[name]] = value
    struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *record)
    output_seed = quant_built.payload[
        2 * catalog.SLOT_BYTES :
        3 * catalog.SLOT_BYTES
    ]
    begin = catalog.OUTPUT_DDR_OFFSET
    raw[begin : begin + catalog.SLOT_BYTES] = output_seed
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "quant-observed.raw"
        output.write_bytes(raw)
        try:
            runner.validate_output(output, quant, quant_built, sample=4)
        except RuntimeError as error:
            assert "without any bounded writeback" in str(error)
        else:
            raise AssertionError("unchanged NE observation seed was accepted")
        raw[begin + catalog.BODY_OFFSET + 3] ^= 0x1
        output.write_bytes(raw)
        runner.validate_output(output, quant, quant_built, sample=4)

    observed_conv = catalog.CALIBRATION_LEAF_BINDINGS[
        "ne-depthwise-backward-conv-observed"
    ]
    assert {case.kind_name for case in observed_conv} == {
        "DEPTHWISE_CONV",
        "BACKWARD_CONV",
    }
    assert {case.dtype_name for case in observed_conv} == {
        "F16",
        "BF16",
    }
    depthwise = tuple(
        case
        for case in observed_conv
        if case.kind_name == "DEPTHWISE_CONV"
    )
    backward = tuple(
        case
        for case in observed_conv
        if case.kind_name == "BACKWARD_CONV"
    )
    assert all(
        case.disposition_name == "BOARD_OBSERVED"
        and case.output_shape == (1, 4, 4, 64)
        and case.output_span == 2048
        for case in depthwise
    )
    assert all(
        case.disposition_name == "BOARD_OBSERVED"
        and case.output_shape == case.rhs_shape == (1, 1, 64, 64)
        and case.output_span == case.rhs_span == 8192
        and "tfr_1" in case.reason
        for case in backward
    )
    probe = (
        pathlib.Path(__file__).resolve().parent
        / "Inputs"
        / "wafer_ne_calibration_probe.c"
    ).read_text()
    conv_decode_begin = probe.index(
        "} else if (kind == WAFER_NEC_CONV) {"
    )
    conv_decode_end = probe.index(
        "  } else {\n    uint32_t expected_profile",
        conv_decode_begin,
    )
    conv_decode = probe[conv_decode_begin:conv_decode_end]
    assert (
        "expected_disposition = WAFER_NEC_BOARD_OBSERVED;"
        in conv_decode
        and "wafer_nec_option_disposition(option)" not in conv_decode
    )
    for function, op_type in (
        ("wafer_nec_issue_depthwise", "1U"),
        ("wafer_nec_issue_backward", "2U"),
    ):
        begin = probe.index(f"{function}(")
        body = probe[begin : probe.index("static ", begin + 16)]
        assert body.index(f"SetOpType(&instruction, {op_type})") < body.index(
            "AddWeight(&instruction"
        )
    assert (
        "output_span = kind == WAFER_NEC_DEPTHWISE_CONV"
        in probe
        and ": rhs_span;" in probe
    )

    backward_case = catalog.CASES_BY_NAME[
        "ne-backward-conv-f16-observed"
    ]
    backward_built = catalog.build_case_payload(backward_case, sample=9)
    backward_raw = bytearray(
        [runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES
    )
    backward_record = [0] * catalog.RECORD_WORDS
    backward_record_values = {
        "MAGIC": catalog.RECORD_MAGIC,
        "SCHEMA_AND_WORDS": (
            catalog.SCHEMA << 32
        ) | catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": backward_case.case_id,
        "DTYPE": backward_case.dtype,
        "LHS_ORIENTATION": backward_case.lhs_orientation,
        "RHS_ORIENTATION": backward_case.rhs_orientation,
        "BATCH": backward_case.batch,
        "M": backward_case.m,
        "K": backward_case.k,
        "N": backward_case.n,
        "LHS_SPAN": backward_case.lhs_span,
        "RHS_SPAN": backward_case.rhs_span,
        "OUTPUT_SPAN": backward_case.output_span,
        "SAMPLE": 9,
        "EXECUTE_RESULT": 1,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "KIND": backward_case.kind,
        "PROFILE": backward_case.profile,
        "OPTION": backward_case.option,
        "AUX_SPAN": backward_case.aux_span,
        "DISPOSITION": backward_case.disposition,
        "LHS_BATCH": backward_case.lhs_batch,
        "RHS_BATCH": backward_case.rhs_batch,
        "PMU_ENABLE": 1,
        "NE_INST_DELTA": 1,
        "NE_BLOCKING_DELTA": 0,
        "NE_EXEC_DELTA": 123,
        "PMU_BASE": catalog.PMU_BASE,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    for name, value in backward_record_values.items():
        backward_record[catalog.REC[name]] = value
    struct.pack_into(
        f"<{catalog.RECORD_WORDS}Q",
        backward_raw,
        0,
        *backward_record,
    )
    backward_slot = backward_built.payload[
        2 * catalog.SLOT_BYTES : 3 * catalog.SLOT_BYTES
    ]
    backward_begin = catalog.OUTPUT_DDR_OFFSET
    backward_raw[
        backward_begin : backward_begin + catalog.SLOT_BYTES
    ] = backward_slot
    physical_begin = backward_begin + catalog.BODY_OFFSET
    backward_raw[
        physical_begin : physical_begin + backward_case.output_span
    ] = bytes(backward_case.output_span)
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "backward-observed.raw"
        output.write_bytes(backward_raw)
        runner.validate_output(
            output, backward_case, backward_built, sample=9
        )
        backward_raw[
            physical_begin + backward_case.output_span
        ] ^= 0x1
        output.write_bytes(backward_raw)
        try:
            runner.validate_output(
                output, backward_case, backward_built, sample=9
            )
        except RuntimeError as error:
            assert "output suffix guard changed" in str(error)
        else:
            raise AssertionError(
                "BackwardConv byte outside derived footprint was accepted"
            )

    unequal = catalog.CALIBRATION_LEAF_BINDINGS[
        "ne-batch-broadcast-positive"
    ]
    assert len(unequal) == 4
    assert {
        (case.dtype_name, case.lhs_batch, case.rhs_batch)
        for case in unequal
    } == {
        (dtype, lhs_batch, rhs_batch)
        for dtype in catalog.FLOAT_DTYPES
        for lhs_batch, rhs_batch in ((1, 2), (2, 1))
    }
    for case in unequal:
        assert case.exact
        lhs, rhs = catalog._gemm_inputs(case)
        assert len(lhs) == case.lhs_batch * case.m * case.k
        assert len(rhs) == case.rhs_batch * case.k * case.n
        expected = catalog._gemm_expected_values(case, lhs, rhs)
        batch_elements = case.m * case.n
        assert expected[:batch_elements] != expected[batch_elements:]
        built = catalog.build_case_payload(case)
        assert built.expected_logical == tuple(
            catalog._encode(case.dtype_name, value) for value in expected
        )

    nonexecuting = tuple(
        case for case in catalog.CATALOG if not case.is_safe
    )
    assert len(nonexecuting) == 3
    assert all(case.reason for case in nonexecuting)
    assert all(
        case.disposition_name == "STATIC_NEGATIVE"
        for case in nonexecuting
    )
    assert catalog.CASES_BY_NAME[
        "ne-gemm-sparse-static-unsupported"
    ].reason.startswith("TsmGemm has no SetSparse")

    bound = tuple(
        case
        for cases in catalog.CALIBRATION_LEAF_BINDINGS.values()
        for case in cases
    )
    assert all(catalog.CALIBRATION_LEAF_BINDINGS.values())
    assert len(bound) == len(set(bound)) == len(catalog.CATALOG)
    assert set(bound) == set(catalog.CATALOG)
    max_span = max(
        max(
            case.lhs_span,
            case.rhs_span,
            case.output_span,
            case.aux_span,
        )
        for case in catalog.SAFE_CASES
    )
    assert max_span <= catalog.SLOT_BYTES - catalog.BODY_OFFSET
    print(
        "wafer_ne_calibration_catalog_test: "
        f"cases={len(catalog.CATALOG)} "
        f"safe={len(catalog.SAFE_CASES)} "
        f"max_span={max_span} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
