#!/usr/bin/env python3
"""No-card tests for instruction qualification catalog and host oracles."""

from __future__ import annotations

import pathlib
import re
import struct
import tempfile
import types

import wafer_instruction_family_catalog as catalog
import wafer_board_instruction_family_probe_runner as runner


def _build_args(
    repo: pathlib.Path, work_dir: pathlib.Path
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        repo_root=repo,
        wafer_compile=repo / "build/wafer-dev/bin/wafer-compile",
        wafer_run=repo / "build/wafer-dev/bin/wafer-run",
        llvm_clangxx=repo / "build/wafer-dev/bin/clang++",
        work_dir=work_dir,
    )


def validate_work_dir_cleanup_is_bounded() -> None:
    repo = pathlib.Path(__file__).resolve().parents[3]
    forbidden = tuple(
        dict.fromkeys(
            (
                repo,
                *repo.parents,
            )
        )
    )
    for work_dir in forbidden:
        args = _build_args(repo, work_dir)
        try:
            runner.require_build_args(args)
        except RuntimeError as error:
            assert "too broad" in str(error)
        else:
            raise AssertionError(f"broad work directory was accepted: {work_dir}")

    with tempfile.TemporaryDirectory() as directory:
        temporary = pathlib.Path(directory)
        unknown = temporary / "unknown"
        unknown.mkdir()
        preserved = unknown / "preserve.txt"
        preserved.write_text("user-owned\n", encoding="utf-8")
        args = _build_args(repo, unknown)
        runner.require_build_args(args)
        try:
            runner.write_source_program(args)
        except RuntimeError as error:
            assert "unknown entries" in str(error)
        else:
            raise AssertionError("unknown work-dir content was deleted")
        assert preserved.read_text(encoding="utf-8") == "user-owned\n"

        managed = temporary / "managed"
        (managed / "source-program").mkdir(parents=True)
        stale = managed / "source-program" / "stale"
        stale.write_text("generated\n", encoding="utf-8")
        external = temporary / "external"
        external.mkdir()
        external_file = external / "keep"
        external_file.write_text("outside\n", encoding="utf-8")
        (managed / "raw").symlink_to(external, target_is_directory=True)
        args = _build_args(repo, managed)
        runner.require_build_args(args)
        source = runner.write_source_program(args)
        assert source == managed / "source-program"
        assert not stale.exists()
        assert not (managed / "raw").exists()
        assert external_file.read_text(encoding="utf-8") == "outside\n"
        assert (source / "functions" / "forward.stablehlo.bc").is_file()
        assert not (source / "functions" / "forward.mlir").exists()
        assert (source / "functions" / "forward.meta").is_file()


def validate_matching_tile_completion() -> None:
    valid = "\n".join(
        (
            "board_stage: completion",
            "board_stage: device-to-host",
            "board_stage: cleanup",
            "board_execution: true",
            *(
                "completion: return_after_local_drain tile_id=" + str(tile_id)
                for tile_id in range(16)
            ),
            "invocation_tiles: 16",
            "physical_tile_domain: 0..15",
        )
    )
    runner.validate_board_lifecycle(valid, "return_after_local_drain")
    for invalid in (
        valid.replace(
            "completion: return_after_local_drain tile_id=7", ""
        ),
        valid.replace(
            "completion: return_after_local_drain tile_id=7",
            "completion: return_after_local_drain tile_id=8",
        ),
        valid
        + "\n"
        + "completion: return_after_local_drain tile_id=7",
    ):
        try:
            runner.validate_board_lifecycle(invalid, "return_after_local_drain")
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "missing, mismatched, or duplicate Tile completion "
                "was accepted"
            )


def _encoded_words(
    dtype_name: str, values: tuple[float, ...] | list[float]
) -> tuple[int, ...]:
    if dtype_name == "F16":
        raw = struct.pack(f"<{len(values)}e", *values)
        return struct.unpack(f"<{len(values)}H", raw)
    assert dtype_name == "BF16"
    words = []
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        words.append(((bits + rounding_bias) >> 16) & 0xFFFF)
    return tuple(words)


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
        pathlib.Path(__file__).resolve().parent.parent
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


def validate_ct_add_f16_tail130_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-f16-tail130"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        260,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<130e", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<130e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<130e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(
        left + right for left, right in zip(lhs, rhs, strict=True)
    )
    assert lhs[128:] == (-3.0, -2.0)
    assert rhs[128:] == (1.0, -1.0)
    assert expected[128:] == (-2.0, -3.0)


def validate_ct_add_bf16_tail130_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-bf16-tail130"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        260,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<130H", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<130H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<130H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    lhs_pattern = (-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    rhs_pattern = (1.0, -1.0, 2.0, 2.0, 0.5, -2.0, 4.0, -0.5)
    lhs_values = [lhs_pattern[index % len(lhs_pattern)] for index in range(130)]
    rhs_values = [rhs_pattern[index % len(rhs_pattern)] for index in range(130)]
    expected_values = [
        left + right
        for left, right in zip(lhs_values, rhs_values, strict=True)
    ]
    assert lhs == _encoded_words("BF16", lhs_values)
    assert rhs == _encoded_words("BF16", rhs_values)
    assert expected == _encoded_words("BF16", expected_values)
    assert expected_values[128:] == [-2.0, -3.0]


def validate_ct_add_f32_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-f32"]
    assert case.is_safe
    assert case.dtype_name == "F32"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        512,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128f", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<128f",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<128f", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert all(value.is_integer() for value in lhs + rhs)
    assert expected == tuple(
        left + right for left, right in zip(lhs, rhs, strict=True)
    )


def validate_ct_add_special_f16_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-special-f16"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        256,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<128H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<128H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert lhs[:12] == (
        0x0000,
        0x8000,
        0x0000,
        0x8000,
        0x7C00,
        0xFC00,
        0x7BFF,
        0xFBFF,
        0x0400,
        0x8400,
        0x0001,
        0x8001,
    )
    assert rhs[:4] == (0x0000, 0x0000, 0x8000, 0x8000)
    assert expected[:4] == (0x0000, 0x0000, 0x0000, 0x8000)
    assert expected[4:12] == lhs[4:12]
    assert all((word & 0x7C00) != 0x7C00 or (word & 0x03FF) == 0 for word in lhs)


def validate_ct_add_special_bf16_oracle() -> None:
    case = catalog.CASES_BY_NAME["ct-add-special-bf16"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        256,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    rhs = struct.unpack_from(
        "<128H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected = struct.unpack_from(
        "<128H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert lhs[:12] == (
        0x0000,
        0x8000,
        0x0000,
        0x8000,
        0x7F80,
        0xFF80,
        0x7F7F,
        0xFF7F,
        0x0080,
        0x8080,
        0x0001,
        0x8001,
    )
    assert rhs[:4] == (0x0000, 0x0000, 0x8000, 0x8000)
    assert expected[:4] == (0x0000, 0x0000, 0x0000, 0x8000)
    assert expected[4:12] == lhs[4:12]
    assert all((word & 0x7F80) != 0x7F80 or (word & 0x007F) == 0 for word in lhs)


def validate_gemm_padding_domain() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16"]
    built = catalog.build_case_payload(case, sample=3)
    raw = bytearray([runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES)
    words = [0] * catalog.RECORD_WORDS
    rec = catalog.REC
    words[rec["MAGIC"]] = catalog.RECORD_MAGIC
    words[rec["WORD_COUNT"]] = catalog.RECORD_WORDS
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
    expected_aux_slot = built.payload[
        3 * catalog.SLOT_BYTES : 4 * catalog.SLOT_BYTES
    ]
    raw[
        catalog.AUX_DDR_OFFSET :
        catalog.AUX_DDR_OFFSET + catalog.SLOT_BYTES
    ] = expected_aux_slot

    padding_index = (
        output_begin + catalog.BODY_OFFSET + case.result_bytes
    )
    raw[padding_index] ^= 0x5A
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "gemm.raw"
        output.write_bytes(raw)
        runner.validate_output(
            output,
            case,
            built.expected_output_slot,
            expected_aux_slot,
            sample=3,
        )

        logical = bytearray(raw)
        logical[output_begin + catalog.BODY_OFFSET] ^= 1
        output.write_bytes(logical)
        try:
            runner.validate_output(
                output,
                case,
                built.expected_output_slot,
                expected_aux_slot,
                sample=3,
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
                output,
                case,
                built.expected_output_slot,
                expected_aux_slot,
                sample=3,
            )
        except RuntimeError as error:
            assert "SPM guard" in str(error)
        else:
            raise AssertionError("GEMM suffix guard mismatch was accepted")


def validate_gemm_dtype_oracles() -> None:
    values = tuple(float(value) for value in (1, 2, 3, 4) * 4)
    identity = tuple(
        1.0 if row == column else 0.0
        for row in range(16)
        for column in range(16)
    )

    def encode(dtype_name: str, source: tuple[float, ...]) -> bytes:
        if dtype_name == "F16":
            return struct.pack(f"<{len(source)}e", *source)
        assert dtype_name == "BF16"
        words = {
            -13.0: 0xC150,
            0.0: 0x0000,
            1.0: 0x3F80,
            2.0: 0x4000,
            3.0: 0x4040,
            4.0: 0x4080,
        }
        return struct.pack(
            f"<{len(source)}H", *(words[value] for value in source)
        )

    for name, dtype_name in (
        ("ne-gemm-f16", "F16"),
        ("ne-gemm-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 32
        assert case.output_span == 256
        assert case.aux_span == 0

        built = catalog.build_case_payload(case)
        lhs = encode(dtype_name, values)
        lhs_begin = catalog.BODY_OFFSET
        assert built.payload[lhs_begin : lhs_begin + len(lhs)] == lhs
        assert built.payload[lhs_begin + len(lhs) : lhs_begin + 256] == bytes(
            256 - len(lhs)
        )

        rhs = encode(dtype_name, identity)
        rhs_begin = catalog.SLOT_BYTES + catalog.BODY_OFFSET
        assert built.payload[rhs_begin : rhs_begin + len(rhs)] == rhs
        assert (
            built.expected_output_slot[
                catalog.BODY_OFFSET : catalog.BODY_OFFSET + len(lhs)
            ]
            == lhs
        )

        seed = encode(dtype_name, (-13.0,) * 128)
        output_begin = 2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET
        assert (
            built.payload[output_begin : output_begin + len(seed)] == seed
        )


def validate_gemm_bf16_accum_round_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-accum-round"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert case.result_bytes == 32
    assert case.output_span == 256
    assert case.aux_span == 0

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (0x3F80,) * 16
    assert lhs[16:] == (0,) * 112

    rhs = struct.unpack_from(
        "<256H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs[:16] == tuple(0x3F80 + column for column in range(16))
    for row in range(1, 16):
        assert rhs[row * 16 : (row + 1) * 16] == tuple(
            0x3B00 if column % 2 == 0 else 0x3980
            for column in range(16)
        )

    expected = struct.unpack_from(
        "<16H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == (
        0x3F84,
        0x3F81,
        0x3F86,
        0x3F83,
        0x3F88,
        0x3F85,
        0x3F8A,
        0x3F87,
        0x3F8C,
        0x3F89,
        0x3F8E,
        0x3F8B,
        0x3F90,
        0x3F8D,
        0x3F92,
        0x3F8F,
    )


def validate_gemm_f16_accum_round_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-accum-round"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (32, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (0x3C00,) * 16
    assert lhs[16:] == (0,) * 112

    rhs = struct.unpack_from(
        "<256H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs[:16] == tuple(0x3C00 + column for column in range(16))
    for row in range(1, 16):
        assert rhs[row * 16 : (row + 1) * 16] == tuple(
            0x0C00
            if column % 2 == 0
            else 0x0800
            if row <= 8
            else 0x8800
            for column in range(16)
        )

    expected = struct.unpack_from(
        "<16H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(
        0x3C00 + column + (4 if column % 2 == 0 else 0)
        for column in range(16)
    )


def validate_gemm_f16_m4_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-m4"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        128,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    expected_rows = (
        tuple(float(value) for value in range(1, 17)),
        tuple(float(value) for value in range(33, 49)),
        tuple(float(-value) for value in range(65, 81)),
        tuple(float(value) for value in range(97, 113)),
    )
    expected = tuple(value for row in expected_rows for value in row)
    assert lhs[:64] == expected
    assert lhs[64:] == (0.0,) * 64
    assert len(set(expected_rows)) == 4

    rhs = struct.unpack_from(
        "<256e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == tuple(
        1.0 if row == column else 0.0
        for row in range(16)
        for column in range(16)
    )
    actual_expected = struct.unpack_from(
        "<64e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_f16_batch2_m8_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-batch2-m8"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        512,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<256e", built.payload, catalog.BODY_OFFSET)
    assert lhs == tuple(float(value) for value in range(1, 257))

    rhs = struct.unpack_from(
        "<512e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    identity = tuple(
        1.0 if row == column else 0.0
        for row in range(16)
        for column in range(16)
    )
    negative_identity = tuple(-value for value in identity)
    assert rhs == identity + negative_identity

    expected = struct.unpack_from(
        "<256e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(value) for value in range(1, 129)) + tuple(
        float(-value) for value in range(129, 257)
    )


def validate_gemm_bf16_batch2_m8_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-batch2-m8"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        512,
        512,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<256H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words(
        "BF16", [float(value) for value in range(1, 257)]
    )

    rhs = struct.unpack_from(
        "<512H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    rhs_values = [
        float(
            1
            if batch == 0 and row == column
            else -1
            if batch == 1 and row == column
            else 0
        )
        for batch in range(2)
        for row in range(16)
        for column in range(16)
    ]
    assert rhs == _encoded_words("BF16", rhs_values)

    expected_values = [float(value) for value in range(1, 129)] + [
        float(-value) for value in range(129, 257)
    ]
    expected = struct.unpack_from(
        "<256H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words("BF16", expected_values)


def validate_gemm_f16_n17_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-n17"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (34, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (1.0,) * 16
    assert lhs[16:] == (0.0,) * 112

    rhs = struct.unpack_from(
        "<512e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    for row in range(16):
        physical_row = rhs[row * 32 : (row + 1) * 32]
        assert physical_row[17:] == (-29.0,) * 15
        assert tuple(
            column
            for column, value in enumerate(physical_row[:17])
            if value != 0.0
        ) == tuple(
            column for column in range(17) if row == column % 16
        )
    expected = struct.unpack_from(
        "<17e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(value) for value in range(1, 18))


def validate_gemm_f16_k17_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-k17"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (32, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:17] == tuple(float(value) for value in range(1, 18))
    assert lhs[17:32] == (-29.0,) * 15
    assert lhs[32:] == (0.0,) * 96

    rhs = struct.unpack_from(
        "<384e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    for row in range(16):
        assert rhs[row * 16 : (row + 1) * 16] == (1.0,) * 16
    assert rhs[16 * 16 : 17 * 16] == tuple(
        float(value) for value in range(2, 18)
    )
    assert rhs[17 * 16 :] == (-31.0,) * 112

    expected = struct.unpack_from(
        "<16e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(170 + 17 * column) for column in range(16))


def validate_gemm_bf16_k17_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-k17"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (32, 256, 0)

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words(
        "BF16", [1.0] * 17 + [-29.0] * 15 + [0.0] * 96
    )

    rhs = struct.unpack_from(
        "<384H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected_rhs = [
        float(1 if row < 16 else column + 2)
        for row in range(17)
        for column in range(16)
    ] + [-31.0] * 112
    assert rhs == _encoded_words("BF16", expected_rhs)

    expected = struct.unpack_from(
        "<16H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words(
        "BF16", [float(18 + column) for column in range(16)]
    )


def validate_gemm_f16_n65_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-n65"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        130,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (1.0,) * 16
    assert lhs[16:] == (0.0,) * 112

    rhs = struct.unpack_from(
        "<1152e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    full_block = rhs[: 16 * 64]
    for row in range(16):
        expected_row = (
            (1.0,) * 64
            if row < 15
            else tuple(float(value) for value in range(100, 164))
        )
        assert full_block[row * 64 : (row + 1) * 64] == expected_row
    tail = rhs[16 * 64 : 16 * 68]
    for row in range(16):
        assert tail[row * 4 : (row + 1) * 4] == (
            float(1 if row < 15 else 164),
            -29.0,
            -29.0,
            -29.0,
        )
    assert rhs[16 * 68 :] == (-31.0,) * 64

    expected = struct.unpack_from(
        "<65e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(float(115 + column) for column in range(65))


def validate_gemm_bf16_n65_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-n65"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        130,
        256,
        0,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words("BF16", [1.0] * 16 + [0.0] * 112)

    rhs = struct.unpack_from(
        "<1152H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    physical_rhs = (
        [
            float(1 if row < 15 else 100 + column)
            for row in range(16)
            for column in range(64)
        ]
        + [
            value
            for row in range(16)
            for value in (
                float(1 if row < 15 else 164),
                -29.0,
                -29.0,
                -29.0,
            )
        ]
        + [-31.0] * 64
    )
    assert rhs == _encoded_words("BF16", physical_rhs)

    expected = struct.unpack_from(
        "<65H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words(
        "BF16", [float(115 + column) for column in range(65)]
    )


def validate_gemm_f16_oriented_nt_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-oriented-nt"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = tuple(
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    )
    semantic_rhs = tuple(
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    )
    stored_rhs = tuple(
        semantic_rhs[contracting * 8 + column]
        for column in range(8)
        for contracting in range(16)
    )
    expected = tuple(
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:64] == semantic_lhs
    assert lhs[64:] == (0.0,) * 64
    rhs = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == stored_rhs
    assert stored_rhs != semantic_rhs
    actual_expected = struct.unpack_from(
        "<32e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_bf16_oriented_nt_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-bf16-oriented-nt"]
    assert case.is_safe
    assert case.dtype_name == "BF16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = [
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    ]
    semantic_rhs = [
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    ]
    stored_rhs = [
        semantic_rhs[contracting * 8 + column]
        for column in range(8)
        for contracting in range(16)
    ]
    expected_values = [
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    ]

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128H", built.payload, catalog.BODY_OFFSET)
    assert lhs == _encoded_words("BF16", semantic_lhs + [0.0] * 64)
    rhs = struct.unpack_from(
        "<128H",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == _encoded_words("BF16", stored_rhs)
    assert stored_rhs != semantic_rhs
    expected = struct.unpack_from(
        "<32H", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == _encoded_words("BF16", expected_values)


def validate_gemm_f16_oriented_tn_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-oriented-tn"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = tuple(
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    )
    semantic_rhs = tuple(
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    )
    stored_lhs = tuple(
        semantic_lhs[row * 16 + contracting]
        for contracting in range(16)
        for row in range(4)
    )
    expected = tuple(
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:64] == stored_lhs
    assert lhs[64:] == (0.0,) * 64
    assert stored_lhs != semantic_lhs
    rhs = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == semantic_rhs
    actual_expected = struct.unpack_from(
        "<32e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_f16_oriented_tt_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-oriented-tt"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_BITS"
    assert (case.result_bytes, case.output_span, case.aux_span) == (64, 256, 0)

    semantic_lhs = tuple(
        float((row * 5 + contracting * 3) % 7 - 3)
        for row in range(4)
        for contracting in range(16)
    )
    semantic_rhs = tuple(
        float((contracting * 2 + column * 5) % 9 - 4)
        for contracting in range(16)
        for column in range(8)
    )
    stored_lhs = tuple(
        semantic_lhs[row * 16 + contracting]
        for contracting in range(16)
        for row in range(4)
    )
    stored_rhs = tuple(
        semantic_rhs[contracting * 8 + column]
        for column in range(8)
        for contracting in range(16)
    )
    expected = tuple(
        float(
            sum(
                semantic_lhs[row * 16 + contracting]
                * semantic_rhs[contracting * 8 + column]
                for contracting in range(16)
            )
        )
        for row in range(4)
        for column in range(8)
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:64] == stored_lhs
    assert lhs[64:] == (0.0,) * 64
    rhs = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == stored_rhs
    assert stored_lhs != semantic_lhs
    assert stored_rhs != semantic_rhs
    actual_expected = struct.unpack_from(
        "<32e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert actual_expected == expected


def validate_gemm_f16_psum_oracle() -> None:
    case = catalog.CASES_BY_NAME["ne-gemm-f16-psum"]
    assert case.is_safe
    assert case.dtype_name == "F16"
    assert case.oracle_name == "EXACT_COMPOSITE"
    assert (case.result_bytes, case.output_span, case.aux_span) == (
        32,
        256,
        256,
    )

    built = catalog.build_case_payload(case)
    lhs = struct.unpack_from("<128e", built.payload, catalog.BODY_OFFSET)
    assert lhs[:16] == (1.0,) * 16
    assert lhs[16:] == (0.0,) * 112
    rhs = struct.unpack_from(
        "<256e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert rhs == tuple(
        float(column + 1)
        for _row in range(16)
        for column in range(16)
    )
    psum = struct.unpack_from(
        "<128e",
        built.payload,
        3 * catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    assert psum[:16] == tuple(float(1000 + column) for column in range(16))
    assert psum[16:] == (-23.0,) * 112

    expected = struct.unpack_from(
        "<16e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    overwrite_only = tuple(float(16 * (column + 1)) for column in range(16))
    assert expected == tuple(
        float(1016 + 17 * column) for column in range(16)
    )
    assert expected != overwrite_only


def validate_arg_extrema_composite_oracles() -> None:
    expected_cases = (
        ("peripheral-argmax-f16", max, 73, 100.0),
        ("peripheral-argmin-f16", min, 42, 0.5),
    )
    for name, operation, expected_index, expected_value in expected_cases:
        case = catalog.CASES_BY_NAME[name]
        assert case.oracle_name == "EXACT_COMPOSITE"
        assert case.result_bytes == 8
        assert case.output_span == 8
        assert case.aux_span == 0
        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<128e", built.payload, catalog.BODY_OFFSET
        )
        assert operation(source) == expected_value
        assert source.count(expected_value) == 1
        assert source.index(expected_value) == expected_index

        result = built.expected_output_slot[
            catalog.BODY_OFFSET : catalog.BODY_OFFSET + case.result_bytes
        ]
        assert struct.unpack_from("<e", result, 0)[0] == expected_value
        assert result[2:4] == _output_seed_padding()
        assert struct.unpack_from("<I", result, 4)[0] == expected_index
        output_payload_offset = 2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET
        assert (
            built.payload[
                output_payload_offset + 2 : output_payload_offset + 4
            ]
            == result[2:4]
        )

    negative = catalog.CASES_BY_NAME[
        "peripheral-argmin-negative-f16-observed"
    ]
    assert negative.is_observation
    assert (negative.result_bytes, negative.output_span, negative.aux_span) == (
        8,
        8,
        0,
    )
    sources = []
    for sample, expected_index in enumerate((37, 83, 109)):
        built = catalog.build_case_payload(negative, sample=sample)
        source = struct.unpack_from(
            "<128e", built.payload, catalog.BODY_OFFSET
        )
        sources.append(source)
        assert all(value < 0.0 for value in source)
        assert source.index(min(source)) == expected_index
        assert source.count(min(source)) == 1
        assert min(source) == float(-100 - sample)
        assert built.expected_output_slot[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + negative.output_span
        ] == _output_seed_padding() * 4
    assert len(set(sources)) == 3


def validate_argmin_pending_domain_observations() -> None:
    tie = catalog.CASES_BY_NAME[
        "peripheral-argmin-tie-f16-observed"
    ]
    nan = catalog.CASES_BY_NAME[
        "peripheral-argmin-nan-f16-observed"
    ]
    assert tie.is_observation and nan.is_observation
    assert (tie.case_id, nan.case_id) == (247, 248)
    assert (tie.result_bytes, tie.output_span, tie.aux_span) == (8, 8, 256)
    assert (nan.result_bytes, nan.output_span, nan.aux_span) == (8, 8, 256)

    tie_sources = []
    tie_source_pairs = []
    for sample, candidates in enumerate(catalog.ARGMIN_TIE_INDEX_SETS):
        built = catalog.build_case_payload(tie, sample=sample)
        tied_value_bits = struct.unpack(
            "<H", struct.pack("<e", catalog.ARGMIN_TIE_VALUES[sample])
        )[0]
        source_bits = struct.unpack_from(
            "<128H", built.payload, catalog.BODY_OFFSET
        )
        tie_sources.append(source_bits)
        tie_source_pairs.append(
            {(bits, index) for index, bits in enumerate(source_bits)}
        )
        assert {
            index
            for index, bits in enumerate(source_bits)
            if bits == tied_value_bits
        } == set(candidates)
        assert built.payload[
            3 * catalog.SLOT_BYTES
            + catalog.BODY_OFFSET :
            3 * catalog.SLOT_BYTES
            + catalog.BODY_OFFSET
            + tie.aux_span
        ] == built.payload[
            catalog.BODY_OFFSET : catalog.BODY_OFFSET + tie.aux_span
        ]
        for selected in candidates:
            result = (
                struct.pack("<H", tied_value_bits)
                + _output_seed_padding()
                + struct.pack("<I", selected)
            )
            classified = catalog.classify_argmin_domain_observation(
                tie, sample, result
            )
            assert classified is not None
            assert classified["classification"] == "tied-minimum"
            assert tuple(classified["candidate_indices"]) == candidates
    assert len(set(tie_sources)) == 3
    assert all(
        tie_source_pairs[lhs].isdisjoint(tie_source_pairs[rhs])
        for lhs in range(3)
        for rhs in range(lhs + 1, 3)
    )
    stale_sample_zero = (
        struct.pack("<H", 0x3800)
        + _output_seed_padding()
        + struct.pack("<I", 5)
    )
    try:
        catalog.classify_argmin_domain_observation(
            tie, 1, stale_sample_zero
        )
    except RuntimeError as error:
        assert "incoherent" in str(error)
    else:
        raise AssertionError("stale prior-sample ArgMin pair was accepted")

    nan_sources = []
    nan_source_pairs = []
    for sample, (
        nan_index,
        nan_bits,
        finite_index,
    ) in enumerate(catalog.ARGMIN_NAN_VECTORS):
        built = catalog.build_case_payload(nan, sample=sample)
        source_bits = struct.unpack_from(
            "<128H", built.payload, catalog.BODY_OFFSET
        )
        nan_sources.append(source_bits)
        nan_source_pairs.append(
            {(bits, index) for index, bits in enumerate(source_bits)}
        )
        assert source_bits[nan_index] == nan_bits
        assert source_bits[finite_index] == 0x3800
        finite_result = (
            struct.pack("<H", 0x3800)
            + _output_seed_padding()
            + struct.pack("<I", finite_index)
        )
        finite_class = catalog.classify_argmin_domain_observation(
            nan, sample, finite_result
        )
        assert finite_class is not None
        assert finite_class["classification"] == "finite-minimum-selected"
        nan_result = (
            struct.pack("<H", nan_bits | 0x0200)
            + _output_seed_padding()
            + struct.pack("<I", nan_index)
        )
        nan_class = catalog.classify_argmin_domain_observation(
            nan, sample, nan_result
        )
        assert nan_class is not None
        assert nan_class["classification"].startswith("nan-selected-")
    assert len(set(nan_sources)) == 3
    assert all(
        nan_source_pairs[lhs].isdisjoint(nan_source_pairs[rhs])
        for lhs in range(3)
        for rhs in range(lhs + 1, 3)
    )

    incoherent = (
        struct.pack("<H", 0x3800)
        + _output_seed_padding()
        + struct.pack("<I", 126)
    )
    try:
        catalog.classify_argmin_domain_observation(nan, 0, incoherent)
    except RuntimeError as error:
        assert "incoherent" in str(error)
    else:
        raise AssertionError("incoherent ArgMin value/index pair was accepted")

    sample = 0
    built = catalog.build_case_payload(tie, sample=sample)
    tied_value_bits = struct.unpack(
        "<H", struct.pack("<e", catalog.ARGMIN_TIE_VALUES[sample])
    )[0]
    actual_slot = bytearray(built.expected_output_slot)
    actual_slot[
        catalog.BODY_OFFSET : catalog.BODY_OFFSET + tie.result_bytes
    ] = (
        struct.pack("<H", tied_value_bits)
        + _output_seed_padding()
        + struct.pack("<I", catalog.ARGMIN_TIE_INDEX_SETS[sample][0])
    )
    expected_aux_slot = built.payload[
        3 * catalog.SLOT_BYTES : 4 * catalog.SLOT_BYTES
    ]
    raw = bytearray(
        [runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES
    )
    words = [0] * catalog.RECORD_WORDS
    rec = catalog.REC
    values = {
        "MAGIC": catalog.RECORD_MAGIC,
        "WORD_COUNT": catalog.RECORD_WORDS,
        "STATUS": 0,
        "CASE": tie.case_id,
        "DISPOSITION": tie.disposition,
        "FAMILY": tie.family,
        "DTYPE": tie.dtype,
        "ORACLE": tie.oracle,
        "RESULT_BYTES": tie.result_bytes,
        "OUTPUT_SPAN": tie.output_span,
        "AUX_SPAN": tie.aux_span,
        "SAMPLE": sample,
        "REQUEST_GUARD": catalog.REQUEST_GUARD,
        "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
        "SLOT_BYTES": catalog.SLOT_BYTES,
        "BODY_OFFSET": catalog.BODY_OFFSET,
        "STEP_FLAGS": (
            catalog.STEP_TARGET_ISSUED
            | catalog.STEP_FINAL_FENCE_COMPLETED
            | catalog.STEP_ARGMIN_INPUT_SNAPSHOTTED
            | catalog.STEP_ARGMIN_DIAGNOSTIC_RETURNED
        ),
        "ARGMIN_DIAGNOSTIC_FLAGS": (
            catalog.ARGMIN_VALUE_VALID
            | catalog.ARGMIN_INDEX_VALID
            | catalog.ARGMIN_TASK_DRAINED
        ),
        "ARGMIN_WRITEBACK_POLLS": 1,
        "ARGMIN_VALUE_RAW": (1 << 32) | tied_value_bits,
        "ARGMIN_INDEX_RAW": (
            (1 << 32) | catalog.ARGMIN_TIE_INDEX_SETS[sample][0]
        ),
        "ARGMIN_ARRIVAL_POLLS": (1 << 32) | 1,
        "ARGMIN_TASKSTATUS_FIRST_LAST": (1 << 32) | 1,
        "ARGMIN_IBCOUNTER_FIRST_LAST": 0,
        "ARGMIN_STATE_POLLS": 1,
        "ARGMIN_WRITEBACK_BUDGET": catalog.ARGMIN_WRITEBACK_POLL_BUDGET,
        "ARGMIN_STATE_BUDGET": catalog.ARGMIN_STATE_POLL_BUDGET,
        "RECORD_GUARD": catalog.RECORD_GUARD,
    }
    for name, value in values.items():
        words[rec[name]] = value
    struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *words)
    raw[
        catalog.OUTPUT_DDR_OFFSET :
        catalog.OUTPUT_DDR_OFFSET + catalog.SLOT_BYTES
    ] = actual_slot
    raw[
        catalog.AUX_DDR_OFFSET :
        catalog.AUX_DDR_OFFSET + catalog.SLOT_BYTES
    ] = expected_aux_slot
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "argmin.raw"
        output.write_bytes(raw)
        observation = runner.validate_output(
            output,
            tie,
            built.expected_output_slot,
            expected_aux_slot,
            sample,
        )
        assert (
            observation["semantic_observation"]["classification"]
            == "tied-minimum"
        )
        corrupted = bytearray(raw)
        corrupted[
            catalog.AUX_DDR_OFFSET + catalog.BODY_OFFSET
        ] ^= 1
        output.write_bytes(corrupted)
        try:
            runner.validate_output(
                output,
                tie,
                built.expected_output_slot,
                expected_aux_slot,
                sample,
            )
        except RuntimeError as error:
            assert "post-ArgMin input snapshot differs" in str(error)
        else:
            raise AssertionError("corrupt post-ArgMin snapshot was accepted")

        bounded = bytearray(
            [runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES
        )
        bounded_words = [0] * catalog.RECORD_WORDS
        bounded_values = {
            **values,
            "STATUS": catalog.STATUS_DIAGNOSTIC_BOUNDED,
            "STEP_FLAGS": (
                catalog.STEP_TARGET_ISSUED
                | catalog.STEP_ARGMIN_DIAGNOSTIC_RETURNED
            ),
            "ARGMIN_DIAGNOSTIC_FLAGS": (
                catalog.ARGMIN_VALUE_VALID
                | catalog.ARGMIN_WRITEBACK_BUDGET_EXHAUSTED
                | catalog.ARGMIN_TASK_DRAINED
            ),
            "ARGMIN_WRITEBACK_POLLS": (
                catalog.ARGMIN_WRITEBACK_POLL_BUDGET
            ),
            "ARGMIN_INDEX_RAW": 0,
            "ARGMIN_ARRIVAL_POLLS": 1,
        }
        for name, value in bounded_values.items():
            bounded_words[rec[name]] = value
        struct.pack_into(
            f"<{catalog.RECORD_WORDS}Q", bounded, 0, *bounded_words
        )
        output.write_bytes(bounded)
        bounded_observation = runner.validate_output(
            output,
            tie,
            built.expected_output_slot,
            expected_aux_slot,
            sample,
        )
        assert (
            bounded_observation["semantic_observation"]["classification"]
            == "bounded-diagnostic"
        )


def validate_probe_seed_is_ncc_local_and_completed() -> None:
    probe = (
        pathlib.Path(__file__).resolve().parent.parent
        / "Inputs"
        / "wafer_instruction_family_probe.c"
    ).read_text()
    start = probe.index("static void wafer_ifp_seed")
    body = probe[start : probe.index("\n}", start) + 2]
    assert "wafer_tx81_rdma" in body
    assert "payload_ddr + slot * WAFER_IFP_SLOT_BYTES" in body
    assert "destinations[slot]" in body
    assert "get_spm_memory_mapping" not in body
    assert body.index("wafer_tx81_rdma") < body.index(
        "wafer_ifp_wait_worker0_drain"
    )
    wait_start = probe.index("static void wafer_ifp_wait_worker0_drain")
    wait_body = probe[
        wait_start : probe.index("\n}", wait_start) + 2
    ]
    assert "TsmGetCsrIbcounter() != 0U" in wait_body
    assert "TsmGetCsrTaskstatus() != 1U" in wait_body
    assert "TsmWaitfinish();" not in wait_body


def validate_argmin_bounded_probe_protocol() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    probe = (root / "Inputs" / "wafer_instruction_family_probe.c").read_text()
    protocol = (
        root / "Inputs" / "wafer_instruction_family_probe_protocol.h"
    ).read_text()
    start = probe.index("wafer_ifp_argmin_bounded")
    body = probe[start : probe.index("\n}", start) + 2]
    for required in (
        "instruction.param.wb_data0 = 0;",
        "instruction.param.wb_data1 = 0;",
        "prelaunch_value_raw = getreg(WAFER_IFP_ARGMIN_VALUE_CSR)",
        "prelaunch_index_raw = getreg(WAFER_IFP_ARGMIN_INDEX_CSR)",
        "(void)TsmExecute(&instruction);",
        "getreg(WAFER_IFP_ARGMIN_VALUE_CSR)",
        "getreg(WAFER_IFP_ARGMIN_INDEX_CSR)",
        "observed_invalid_pair",
        "wafer_ifp_argmin_pair_is_fresh",
        "wafer_ifp_argmin_pair_is_coherent",
        "poll <= WAFER_IFP_ARGMIN_WRITEBACK_POLL_BUDGET",
        "poll <= WAFER_IFP_ARGMIN_STATE_POLL_BUDGET",
        "TsmGetCsrTaskstatus()",
        "TsmGetCsrIbcounter()",
        "WAFER_IFP_STEP_ARGMIN_DIAGNOSTIC_RETURNED",
    ):
        assert required in body
    assert body.index("prelaunch_value_raw = getreg") < body.index(
        "(void)TsmExecute(&instruction);"
    )
    assert body.index("prelaunch_index_raw = getreg") < body.index(
        "(void)TsmExecute(&instruction);"
    )
    for forbidden in (
        "__execute_ct_argmaxmin",
        "__ct_init_argmaxmin",
        "__ct_execute_argmaxmin",
        "__ct_get_argmaxmin_result",
        "TsmWaitfinish",
        "while (",
    ):
        assert forbidden not in body
    assert "WAFER_IFP_REC_ARGMIN_VALUE_RAW = 22" in protocol
    assert "WAFER_IFP_REC_ARGMIN_INDEX_RAW = 23" in protocol
    assert "WAFER_IFP_REC_ARGMIN_ARRIVAL_POLLS = 24" in protocol


def validate_pool_max_oracle() -> None:
    source_values = [
        float(100 * row + 10 * column + channel % 8)
        for row in range(2)
        for column in range(4)
        for channel in range(64)
    ]
    expected_values = [
        float(100 + 10 * (2 * output_column + 1) + channel % 8)
        for output_column in range(2)
        for channel in range(64)
    ]
    for name, dtype_name in (
        ("pool-f16", "F16"),
        ("pool-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 256
        assert case.output_span == 256
        assert case.aux_span == 0

        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<512H", built.payload, catalog.BODY_OFFSET
        )
        assert source == _encoded_words(dtype_name, source_values)
        result = struct.unpack_from(
            "<128H", built.expected_output_slot, catalog.BODY_OFFSET
        )
        assert result == _encoded_words(dtype_name, expected_values)
        assert built.expected_output_slot[: catalog.BODY_OFFSET] == bytes(
            [catalog.SLOT_CANARY]
        ) * catalog.BODY_OFFSET
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.output_span :
        ] == bytes([catalog.SLOT_CANARY]) * (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
        )


def validate_unpool_rows() -> None:
    for name, oracle_name in (
        ("unpool-f16", "EXACT_COMPOSITE"),
        ("unpool-index-f16", "NO_ORACLE"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.oracle_name == oracle_name
        assert (case.result_bytes, case.output_span, case.aux_span) == (
            512,
            512,
            256,
        )

        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<256e", built.payload, catalog.BODY_OFFSET
        )
        assert source == tuple(
            float(
                128 + channel
                if position == channel % 4
                else 1 + position
            )
            for position in range(4)
            for channel in range(64)
        )
        output_seed = built.payload[
            2 * catalog.SLOT_BYTES + catalog.BODY_OFFSET :
            2 * catalog.SLOT_BYTES
            + catalog.BODY_OFFSET
            + case.output_span
        ]
        if case.is_observation:
            assert output_seed == struct.pack("<256e", *([-13.0] * 256))
            assert (
                built.expected_output_slot[
                    catalog.BODY_OFFSET :
                    catalog.BODY_OFFSET + case.output_span
                ]
                == output_seed
            )
        else:
            assert output_seed == bytes(case.output_span)
            expected = struct.unpack_from(
                "<256e", built.expected_output_slot, catalog.BODY_OFFSET
            )
            assert expected == tuple(
                float(128 + channel if position == channel % 4 else 0)
                for position in range(4)
                for channel in range(64)
            )

    probe = (
        pathlib.Path(__file__).resolve().parent.parent
        / "Inputs"
        / "wafer_instruction_family_probe.c"
    ).read_text()
    for symbol, wrapper, opcode in (
        (
            "UNPOOL_F16",
            "wafer_tx81_unpool_mask",
            "OP_FUNC_CGRATensor_DataMoveOp_T_T_maskunpool",
        ),
        (
            "UNPOOL_INDEX_F16",
            "wafer_tx81_unpool_unpool",
            "OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool",
        ),
    ):
        start = probe.index(f"case WAFER_IFP_CASE_{symbol}:")
        case_body = probe[start : probe.index("break;", start)]
        pool = case_body.index("wafer_tx81_pool_indexedmax")
        unpool = case_body.index(wrapper)
        assert pool < unpool
        assert "wafer_tx81_ncc_join" not in case_body
        assert opcode in case_body
        assert "(uint32_t)auxiliary" in case_body


def validate_unpool_capability_rows() -> None:
    protocol = (
        pathlib.Path(__file__).resolve().parent.parent
        / "Inputs"
        / "wafer_instruction_family_probe_protocol.h"
    ).read_text()
    assert (
        "#define WAFER_IFP_REPEATED_SENTINEL_OFFSET 2048U"
        in protocol
    )
    assert (
        "#define WAFER_IFP_REPEATED_SENTINEL_BYTES 512U"
        in protocol
    )
    assert (
        "#define WAFER_IFP_REPEATED_AUX_BYTES 512U"
        in protocol
    )
    assert (
        "#define WAFER_IFP_REPEATED_AUX_SNAPSHOT_OFFSET 512U"
        in protocol
    )
    assert (
        "WAFER_IFP_STEP_REPEATED_OVERLAP_VALUES_STAGED = "
        "UINT32_C(1) << 4"
        in protocol
    )
    assert (
        "WAFER_IFP_STEP_REPEATED_OVERLAP_AUX_SNAPSHOTTED = "
        "UINT32_C(1) << 6"
        in protocol
    )
    assert catalog.REPEATED_UNPOOL_SENTINEL_OFFSET == 2048
    assert catalog.REPEATED_UNPOOL_SENTINEL_BYTES == 512
    assert catalog.REPEATED_UNPOOL_AUX_BYTES == 512
    assert catalog.REPEATED_UNPOOL_AUX_SNAPSHOT_OFFSET == 512
    assert catalog.STEP_REPEATED_OVERLAP_VALUES_STAGED == 1 << 4
    assert catalog.STEP_REPEATED_OVERLAP_AUX_SNAPSHOTTED == 1 << 6

    rotations = tuple(
        catalog._repeated_unpool_sentinels(sample)
        for sample in range(4)
    )
    assert len(set(rotations)) == 4
    for position in range(4):
        assert {
            rotation[position][0] for rotation in rotations
        } == {256.0, 320.0, 384.0, 448.0}

    expected_rows = {
        "unpool-index-bf16-observed": (236, "BF16", "NO_ORACLE", 512, 512, 256),
        "unpool-index-f32-observed": (237, "F32", "NO_ORACLE", 1024, 1024, 512),
        "unpool-avg-bf16": (238, "BF16", "EXACT_BITS", 512, 512, 0),
        "unpool-avg-f32": (239, "F32", "EXACT_BITS", 1024, 1024, 0),
        "unpool-mask-bf16": (240, "BF16", "EXACT_COMPOSITE", 512, 512, 256),
        "unpool-mask-f32": (241, "F32", "NO_ORACLE", 1024, 1024, 512),
        "unpool-index-f16-k3x2-s2x1-observed": (
            242, "F16", "NO_ORACLE", 1920, 2048, 512
        ),
        "unpool-avg-f16-k3x2-s2x1-observed": (
            243, "F16", "NO_ORACLE", 1920, 2048, 0
        ),
        "unpool-mask-f16-k3x2-s2x1": (
            244, "F16", "NO_ORACLE", 1920, 2048, 512
        ),
        "unpool-index-f16-repeated-overlap-observed": (
            245, "F16", "NO_ORACLE", 1920, 2048, 1024
        ),
        "unpool-mask-f16-repeated-overlap-observed": (
            246, "F16", "NO_ORACLE", 1920, 2048, 1024
        ),
    }
    for name, expected_row in expected_rows.items():
        case = catalog.CASES_BY_NAME[name]
        assert (
            case.case_id,
            case.dtype_name,
            case.oracle_name,
            case.result_bytes,
            case.output_span,
            case.aux_span,
        ) == expected_row
        built = catalog.build_case_payload(case)
        assert built.expected_output_slot[: catalog.BODY_OFFSET] == bytes(
            [catalog.SLOT_CANARY]
        ) * catalog.BODY_OFFSET
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.output_span :
        ] == bytes([catalog.SLOT_CANARY]) * (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
        )

    for name in ("unpool-avg-bf16", "unpool-avg-f32"):
        case = catalog.CASES_BY_NAME[name]
        built = catalog.build_case_payload(case)
        source = [
            float(4 * (channel % 16 + 1)) for channel in range(64)
        ]
        expected = [
            source[channel] / 4.0
            for _position in range(4)
            for channel in range(64)
        ]
        assert built.payload[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + len(catalog._fp(case.dtype_name, source))
        ] == catalog._fp(case.dtype_name, source)
        assert built.expected_output_slot[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + case.result_bytes
        ] == catalog._fp(case.dtype_name, expected)

    asymmetric = catalog.CASES_BY_NAME[
        "unpool-mask-f16-k3x2-s2x1"
    ]
    built = catalog.build_case_payload(asymmetric)
    source = struct.unpack_from(
        "<960e", built.payload, catalog.BODY_OFFSET
    )
    _, _, semantic_bytes = catalog._unpool(asymmetric)
    semantic = struct.unpack("<960e", semantic_bytes)
    expected_seed = struct.unpack_from(
        "<960e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected_seed == (-13.0,) * 960
    nonzero_positions = {
        (1, 2),
        (1, 4),
        (2, 2),
        (2, 4),
    }
    for row in range(3):
        for column in range(5):
            for channel in range(64):
                ordinal = (row * 5 + column) * 64 + channel
                assert semantic[ordinal] == (
                    source[ordinal]
                    if (row, column) in nonzero_positions
                    else 0.0
                )

    for name in (
        "unpool-index-f16-repeated-overlap-observed",
        "unpool-mask-f16-repeated-overlap-observed",
    ):
        case = catalog.CASES_BY_NAME[name]
        built = catalog.build_case_payload(case, sample=2)
        source = struct.unpack_from(
            "<960e", built.payload, catalog.BODY_OFFSET
        )
        relative_indices = []
        global_positions = []
        for output_row in range(2):
            for output_column in range(2):
                window = [
                    (
                        source[
                            ((output_row + kernel_row) * 5
                             + 2 * output_column + kernel_column) * 64
                        ],
                        kernel_row * 3 + kernel_column,
                        (
                            output_row + kernel_row,
                            2 * output_column + kernel_column,
                        ),
                    )
                    for kernel_row in range(2)
                    for kernel_column in range(3)
                ]
                maximum = max(window)
                relative_indices.append(maximum[1])
                global_positions.append(maximum[2])
        assert relative_indices == [5, 3, 2, 0]
        assert global_positions == [(1, 2)] * 4
        sentinel_begin = (
            catalog.BODY_OFFSET + catalog.REPEATED_UNPOOL_SENTINEL_OFFSET
        )
        sentinels = struct.unpack_from(
            "<256e", built.payload, sentinel_begin
        )
        assert sentinels == tuple(
            value
            for position in catalog._repeated_unpool_sentinels(2)
            for value in position
        )
        assert (
            built.payload[
                catalog.BODY_OFFSET + case.result_bytes : sentinel_begin
            ]
            == bytes([catalog.SLOT_CANARY])
            * (
                catalog.REPEATED_UNPOOL_SENTINEL_OFFSET
                - case.result_bytes
            )
        )

    probe = (
        pathlib.Path(__file__).resolve().parent.parent
        / "Inputs"
        / "wafer_instruction_family_probe.c"
    ).read_text()
    for symbol, wrapper in (
        ("UNPOOL_INDEX_BF16_OBSERVED", "wafer_tx81_unpool_unpool"),
        ("UNPOOL_INDEX_F32_OBSERVED", "wafer_tx81_unpool_unpool"),
        ("UNPOOL_AVG_BF16", "wafer_tx81_unpool_avg"),
        ("UNPOOL_AVG_F32", "wafer_tx81_unpool_avg"),
        ("UNPOOL_MASK_BF16", "wafer_tx81_unpool_mask"),
        ("UNPOOL_MASK_F32", "wafer_tx81_unpool_mask"),
        (
            "UNPOOL_INDEX_F16_ASYMMETRIC_OBSERVED",
            "wafer_tx81_unpool_unpool",
        ),
        (
            "UNPOOL_AVG_F16_ASYMMETRIC_OBSERVED",
            "wafer_tx81_unpool_avg",
        ),
        ("UNPOOL_MASK_F16_ASYMMETRIC", "wafer_tx81_unpool_mask"),
        (
            "UNPOOL_INDEX_F16_REPEATED_OVERLAP_OBSERVED",
            "wafer_tx81_unpool_unpool",
        ),
        (
            "UNPOOL_MASK_F16_REPEATED_OVERLAP_OBSERVED",
            "wafer_tx81_unpool_mask",
        ),
    ):
        start = probe.index(f"case WAFER_IFP_CASE_{symbol}:")
        body = probe[start : probe.index("break;", start)]
        assert wrapper in body
        if "AVG" not in symbol:
            assert "wafer_tx81_pool_indexedmax" in body
            assert "wafer_tx81_ncc_join" not in body
        if "REPEATED_OVERLAP" in symbol:
            assert "wafer_ifp_wait_worker0_drain();" not in body
            assert "wafer_ifp_copy_spm_bytes(" not in body
            assert body.count("wafer_ifp_copy_spm_bytes_ncc(") == 2
            pool = body.index("wafer_tx81_pool_indexedmax")
            snapshot = body.index(
                "WAFER_IFP_REPEATED_AUX_SNAPSHOT_OFFSET"
            )
            stage = body.index("WAFER_IFP_REPEATED_SENTINEL_OFFSET")
            unpool = body.index(wrapper)
            assert pool < snapshot < stage < unpool
            assert (
                "WAFER_IFP_STEP_REPEATED_OVERLAP_VALUES_STAGED"
                in body
            )
            assert (
                "WAFER_IFP_STEP_REPEATED_OVERLAP_AUX_SNAPSHOTTED"
                in body
            )


def validate_repeated_unpool_bounded_oracle() -> None:
    for name in (
        "unpool-index-f16-repeated-overlap-observed",
        "unpool-mask-f16-repeated-overlap-observed",
    ):
        case = catalog.CASES_BY_NAME[name]
        sample = 1
        built = catalog.build_case_payload(case, sample=sample)
        raw = bytearray(
            [runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES
        )
        words = [0] * catalog.RECORD_WORDS
        values = {
            "MAGIC": catalog.RECORD_MAGIC,
            "WORD_COUNT": catalog.RECORD_WORDS,
            "STATUS": 0,
            "CASE": case.case_id,
            "DISPOSITION": case.disposition,
            "FAMILY": case.family,
            "DTYPE": case.dtype,
            "ORACLE": case.oracle,
            "RESULT_BYTES": case.result_bytes,
            "OUTPUT_SPAN": case.output_span,
            "AUX_SPAN": case.aux_span,
            "SAMPLE": sample,
            "REQUEST_GUARD": catalog.REQUEST_GUARD,
            "OUTPUT_DDR_OFFSET": catalog.OUTPUT_DDR_OFFSET,
            "SLOT_BYTES": catalog.SLOT_BYTES,
            "BODY_OFFSET": catalog.BODY_OFFSET,
            "STEP_FLAGS": (
                catalog.STEP_TARGET_ISSUED
                | catalog.STEP_FINAL_FENCE_COMPLETED
                | catalog.STEP_REPEATED_OVERLAP_VALUES_STAGED
                | catalog.STEP_REPEATED_OVERLAP_AUX_SNAPSHOTTED
            ),
            "RECORD_GUARD": catalog.RECORD_GUARD,
        }
        for field, value in values.items():
            words[catalog.REC[field]] = value
        struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *words)

        output_begin = catalog.OUTPUT_DDR_OFFSET
        output_slot = bytearray(built.expected_output_slot)
        logical_begin = catalog.BODY_OFFSET
        output_slot[
            logical_begin : logical_begin + case.result_bytes
        ] = bytes(case.result_bytes)
        target_begin = (
            logical_begin
            + catalog.REPEATED_UNPOOL_TARGET_POSITION * 64 * 2
        )
        winner_position = 2
        struct.pack_into(
            "<64e",
            output_slot,
            target_begin,
            *catalog._repeated_unpool_sentinels(sample)[winner_position],
        )
        raw[output_begin : output_begin + catalog.SLOT_BYTES] = output_slot
        expected_aux = catalog.repeated_unpool_expected_aux_slot()
        raw[
            catalog.AUX_DDR_OFFSET :
            catalog.AUX_DDR_OFFSET + catalog.SLOT_BYTES
        ] = expected_aux

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "repeated-unpool.raw"
            output.write_bytes(raw)
            observation = runner.validate_output(
                output,
                case,
                built.expected_output_slot,
                built.payload[
                    3 * catalog.SLOT_BYTES : 4 * catalog.SLOT_BYTES
                ],
                sample,
            )
            semantic = observation["semantic_observation"]
            assert semantic["classification"] == "uniform-candidate-subset"
            assert semantic["candidate_source_masks"] == (
                1 << winner_position,
            )
            assert set(semantic["channel_candidate_source_masks"]) == {
                (1 << winner_position,)
            }
            assert set(semantic["channel_candidate_value_masks"]) == {
                (1 << ((winner_position + sample) % 4),)
            }
            assert semantic["non_target_zero_positions"] == 14
            assert semantic["padding_mode"] == "seed"

            heterogeneous = bytearray(raw)
            sentinels = catalog._repeated_unpool_sentinels(sample)
            for channel in range(64):
                position = channel % 4
                struct.pack_into(
                    "<e",
                    heterogeneous,
                    output_begin + target_begin + channel * 2,
                    sentinels[position][channel],
                )
            output.write_bytes(heterogeneous)
            heterogeneous_observation = runner.validate_output(
                output,
                case,
                built.expected_output_slot,
                built.payload[
                    3 * catalog.SLOT_BYTES : 4 * catalog.SLOT_BYTES
                ],
                sample,
            )["semantic_observation"]
            assert (
                heterogeneous_observation["classification"]
                == "lane-varying-candidate-subsets"
            )
            assert (
                heterogeneous_observation["candidate_source_masks"] == ()
            )
            assert {
                (entry["masks"], entry["channels"])
                for entry in heterogeneous_observation[
                    "candidate_source_mask_histogram"
                ]
            } == {
                ((1 << position,), 16) for position in range(4)
            }

            if case.symbol == (
                "UNPOOL_INDEX_F16_REPEATED_OVERLAP_OBSERVED"
            ):
                zero_fill = bytearray(raw)
                zero_fill[
                    output_begin + catalog.BODY_OFFSET :
                    output_begin + catalog.BODY_OFFSET + case.output_span
                ] = bytes(case.output_span)
                output.write_bytes(zero_fill)
                zero_observation = runner.validate_output(
                    output,
                    case,
                    built.expected_output_slot,
                    built.payload[
                        3 * catalog.SLOT_BYTES :
                        4 * catalog.SLOT_BYTES
                    ],
                    sample,
                )["semantic_observation"]
                assert (
                    zero_observation["classification"]
                    == "zero-fill-no-scatter-baseline"
                )
                assert zero_observation["collision_evidence"] is False
                assert zero_observation["candidate_source_masks"] == (0,)
            else:
                zero_fill = bytearray(raw)
                zero_fill[
                    output_begin + catalog.BODY_OFFSET :
                    output_begin + catalog.BODY_OFFSET + case.output_span
                ] = bytes(case.output_span)
                output.write_bytes(zero_fill)
                try:
                    runner.validate_output(
                        output,
                        case,
                        built.expected_output_slot,
                        built.payload[
                            3 * catalog.SLOT_BYTES :
                            4 * catalog.SLOT_BYTES
                        ],
                        sample,
                    )
                except RuntimeError as error:
                    assert "bounded winner/accumulation subset" in str(error)
                else:
                    raise AssertionError(
                        f"{case.name}: mask-Unpool drop-all was accepted"
                    )

            mutations = (
                (
                    output_begin
                    + catalog.BODY_OFFSET
                    + case.result_bytes,
                    "physical padding",
                ),
                (
                    output_begin
                    + catalog.BODY_OFFSET
                    + case.output_span,
                    "SPM guard",
                ),
                (
                    catalog.AUX_DDR_OFFSET + catalog.BODY_OFFSET,
                    "post-consumer auxiliary",
                ),
                (
                    catalog.AUX_DDR_OFFSET
                    + catalog.BODY_OFFSET
                    + catalog.REPEATED_UNPOOL_AUX_SNAPSHOT_OFFSET,
                    "pre-consumer auxiliary snapshot",
                ),
                (output_begin + target_begin, "target channel"),
                (output_begin + catalog.BODY_OFFSET, "non-target position"),
            )
            for offset, diagnostic in mutations:
                corrupted = bytearray(raw)
                corrupted[offset] ^= 1
                output.write_bytes(corrupted)
                try:
                    runner.validate_output(
                        output,
                        case,
                        built.expected_output_slot,
                        built.payload[
                            3 * catalog.SLOT_BYTES :
                            4 * catalog.SLOT_BYTES
                        ],
                        sample,
                    )
                except RuntimeError as error:
                    assert diagnostic in str(error)
                else:
                    raise AssertionError(
                        f"{case.name}: {diagnostic} corruption was accepted"
                    )

            padding_only = bytearray(raw)
            padding_only[
                output_begin :
                output_begin + catalog.SLOT_BYTES
            ] = built.expected_output_slot
            padding_only[
                output_begin
                + catalog.BODY_OFFSET
                + case.result_bytes
            ] ^= 1
            output.write_bytes(padding_only)
            try:
                runner.validate_output(
                    output,
                    case,
                    built.expected_output_slot,
                    built.payload[
                        3 * catalog.SLOT_BYTES :
                        4 * catalog.SLOT_BYTES
                    ],
                    sample,
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    f"{case.name}: padding-only writeback was accepted"
                )


def validate_conv_oracle() -> None:
    source_values = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    weight_values = (
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
        17.0,
    )
    expected_values = (1.0, 16.0, 3.0, 11.0, 2.0, 32.0, 5.0, 13.0)

    for name, dtype_name in (
        ("conv-f16", "F16"),
        ("conv-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 16
        assert case.output_span == 256
        assert case.aux_span == 0

        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<12H", built.payload, catalog.BODY_OFFSET
        )
        assert source == _encoded_words(dtype_name, source_values)
        weight = struct.unpack_from(
            "<16H",
            built.payload,
            catalog.SLOT_BYTES + catalog.BODY_OFFSET,
        )
        assert weight == _encoded_words(dtype_name, weight_values)
        result = struct.unpack_from(
            "<8H", built.expected_output_slot, catalog.BODY_OFFSET
        )
        assert result == _encoded_words(dtype_name, expected_values)

        padding_elements = (case.output_span - case.result_bytes) // 2
        padding_word = _encoded_words(dtype_name, (-13.0,))[0]
        padding = struct.pack("<H", padding_word) * padding_elements
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.result_bytes :
            catalog.BODY_OFFSET + case.output_span
        ] == padding
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.output_span :
        ] == bytes([catalog.SLOT_CANARY]) * (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
        )


def validate_img2col_oracle() -> None:
    for name, dtype_name in (
        ("tdma-img2col-f16", "F16"),
        ("tdma-img2col-bf16", "BF16"),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.is_safe
        assert case.dtype_name == dtype_name
        assert case.oracle_name == "EXACT_BITS"
        assert case.result_bytes == 2048
        assert case.output_span == 2048
        assert case.aux_span == 0

        if dtype_name == "F16":
            source_values = [
                float(1 + 256 * row + 64 * column + channel)
                for row in range(3)
                for column in range(3)
                for channel in range(64)
            ]
        else:
            source_values = [
                float(1 + 64 * row + 16 * column + channel % 8)
                for row in range(3)
                for column in range(3)
                for channel in range(64)
            ]
        source_words = _encoded_words(dtype_name, source_values)
        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<576H", built.payload, catalog.BODY_OFFSET
        )
        assert source == source_words

        expected = tuple(
            source_words[
                ((output_row + kernel_y) * 3 + output_column + kernel_x) * 64
                + channel
            ]
            for kernel_y in range(2)
            for kernel_x in range(2)
            for output_row in range(2)
            for output_column in range(2)
            for channel in range(64)
        )
        result = struct.unpack_from(
            "<1024H", built.expected_output_slot, catalog.BODY_OFFSET
        )
        assert result == expected
        assert built.expected_output_slot[
            catalog.BODY_OFFSET + case.output_span :
        ] == bytes([catalog.SLOT_CANARY]) * (
            catalog.SLOT_BYTES - catalog.BODY_OFFSET - case.output_span
        )


def validate_lut16_oracle() -> None:
    case = catalog.CASES_BY_NAME["peripheral-lut16-f16"]
    assert case.is_safe
    assert case.oracle_name == "EXACT_BITS"
    assert case.result_bytes == 256
    assert case.output_span == 256
    assert case.aux_span == 0

    built = catalog.build_case_payload(case)
    source = struct.unpack_from(
        "<128H", built.payload, catalog.BODY_OFFSET
    )
    table = struct.unpack_from(
        "<128e",
        built.payload,
        catalog.SLOT_BYTES + catalog.BODY_OFFSET,
    )
    expected_indices = tuple(
        (37 * index + 11) % 128 for index in range(128)
    )
    assert source == tuple(2 * index for index in expected_indices)
    assert all(offset % 2 == 0 and offset < 256 for offset in source)
    assert table == tuple(float(index - 64) for index in range(128))

    result = struct.unpack_from(
        "<128e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert result == tuple(table[index] for index in expected_indices)


def _output_seed_padding() -> bytes:
    return struct.pack("<e", -13.0)


def validate_extended_pool_unpool_oracles() -> None:
    for name, operation in (
        ("pool-avg-f16", lambda values: sum(values) / 4.0),
        ("pool-sum-f16", sum),
        ("pool-min-f16", min),
    ):
        case = catalog.CASES_BY_NAME[name]
        assert case.oracle_name == "EXACT_BITS"
        assert (case.result_bytes, case.output_span) == (256, 256)
        built = catalog.build_case_payload(case)
        source = struct.unpack_from(
            "<512e", built.payload, catalog.BODY_OFFSET
        )
        expected = struct.unpack_from(
            "<128e", built.expected_output_slot, catalog.BODY_OFFSET
        )
        oracle = tuple(
            operation(
                tuple(
                    source[(row * 4 + column) * 64 + channel]
                    for row in range(2)
                    for column in range(
                        2 * output_column, 2 * output_column + 2
                    )
                )
            )
            for output_column in range(2)
            for channel in range(64)
        )
        assert expected == oracle

    indexed = catalog.CASES_BY_NAME["pool-indexed-min-f16"]
    assert indexed.oracle_name == "EXACT_COMPOSITE"
    assert (indexed.result_bytes, indexed.output_span) == (512, 512)
    built = catalog.build_case_payload(indexed)
    source = struct.unpack_from(
        "<512e", built.payload, catalog.BODY_OFFSET
    )
    values = struct.unpack_from(
        "<128e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    indices = struct.unpack_from(
        "<128H",
        built.expected_output_slot,
        catalog.BODY_OFFSET + 256,
    )
    for output_column in range(2):
        for channel in range(64):
            ordinal = output_column * 64 + channel
            window = tuple(
                source[(row * 4 + column) * 64 + channel]
                for row in range(2)
                for column in range(
                    2 * output_column, 2 * output_column + 2
                )
            )
            assert values[ordinal] == min(window)
            assert indices[ordinal] == window.index(min(window))

    indexed_max = catalog.CASES_BY_NAME[
        "pool-indexed-max-f16-k3x2-s2x1"
    ]
    assert indexed_max.oracle_name == "EXACT_COMPOSITE"
    assert (
        indexed_max.result_bytes,
        indexed_max.output_span,
    ) == (1024, 1024)
    built = catalog.build_case_payload(indexed_max)
    source = struct.unpack_from(
        "<960e", built.payload, catalog.BODY_OFFSET
    )
    values = struct.unpack_from(
        "<256e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    indices = struct.unpack_from(
        "<256H",
        built.expected_output_slot,
        catalog.BODY_OFFSET + 512,
    )
    for output_row in range(2):
        for output_column in range(2):
            for channel in range(64):
                ordinal = (
                    (output_row * 2 + output_column) * 64 + channel
                )
                window = tuple(
                    source[
                        ((output_row + kernel_y) * 5
                         + 2 * output_column + kernel_x) * 64
                        + channel
                    ]
                    for kernel_y in range(2)
                    for kernel_x in range(3)
                )
                assert values[ordinal] == max(window)
                assert indices[ordinal] == 5
                assert indices[ordinal] == window.index(max(window))

    unpool = catalog.CASES_BY_NAME["unpool-avg-f16"]
    built = catalog.build_case_payload(unpool)
    source = struct.unpack_from(
        "<64e", built.payload, catalog.BODY_OFFSET
    )
    expected = struct.unpack_from(
        "<256e", built.expected_output_slot, catalog.BODY_OFFSET
    )
    assert expected == tuple(
        source[channel] / 4.0
        for _position in range(4)
        for channel in range(64)
    )


def validate_pool_capability_matrix() -> None:
    cases = tuple(
        case
        for case in catalog.SAFE_CASES
        if catalog._POOL_SYMMETRIC_BASE
        <= case.case_id
        <= catalog._POOL_TIE_END
    )
    assert len(cases) == 24
    assert {case.case_id for case in cases} == {
        205,
        206,
        208,
        209,
        212,
        214,
        215,
        217,
        218,
        220,
        221,
        222,
        223,
        224,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
    }
    assert sum(case.is_observation for case in cases) == 8
    for case in cases:
        built = catalog.build_case_payload(case)
        assert case.output_span == case.result_bytes
        assert case.aux_span == 0
        assert any(
            built.payload[
                catalog.BODY_OFFSET :
                catalog.BODY_OFFSET + case.result_bytes
            ]
        )
        actual = built.expected_output_slot[
            catalog.BODY_OFFSET :
            catalog.BODY_OFFSET + case.result_bytes
        ]
        if case.is_observation:
            assert actual == struct.pack(
                f"<{case.output_span // 2}e",
                *([-13.0] * (case.output_span // 2)),
            )
        else:
            assert actual != bytes(case.result_bytes)

    for name in (
        "pool-indexed-max-f32-k2x2-s2x2",
        "pool-indexed-min-f32-k2x2-s2x2",
    ):
        indexed_f32 = catalog.CASES_BY_NAME[name]
        assert (indexed_f32.result_bytes, indexed_f32.output_span) == (
            1024,
            1024,
        )
        built = catalog.build_case_payload(indexed_f32)
        indices = struct.unpack_from(
            "<128I",
            built.expected_output_slot,
            catalog.BODY_OFFSET + 512,
        )
        assert indices == (3,) * 128

    asymmetric = catalog.CASES_BY_NAME["pool-avg-f16-k3x2-s2x1"]
    built = catalog.build_case_payload(asymmetric)
    assert len(
        built.payload[
            catalog.BODY_OFFSET : catalog.BODY_OFFSET + 3 * 5 * 64 * 2
        ]
    ) == 3 * 5 * 64 * 2
    assert asymmetric.result_bytes == 2 * 2 * 64 * 2

    tie = catalog.CASES_BY_NAME["pool-indexed-max-f16-tie-observed"]
    built = catalog.build_case_payload(tie)
    source = struct.unpack_from(
        "<512e", built.payload, catalog.BODY_OFFSET
    )
    for output_column in range(2):
        for channel in range(64):
            window = tuple(
                source[(row * 4 + column) * 64 + channel]
                for row in range(2)
                for column in range(
                    2 * output_column, 2 * output_column + 2
                )
            )
            assert window[0] == window[3] == max(window)


def validate_peripheral_exact_and_observation_rows() -> None:
    bilinear = catalog.CASES_BY_NAME["peripheral-bilinear-f16"]
    assert bilinear.is_observation
    built = catalog.build_case_payload(bilinear)
    source = built.payload[
        catalog.BODY_OFFSET : catalog.BODY_OFFSET + bilinear.result_bytes
    ]
    expected = built.expected_output_slot[
        catalog.BODY_OFFSET : catalog.BODY_OFFSET + bilinear.result_bytes
    ]
    assert source != expected
    assert expected == struct.pack("<128e", *([-13.0] * 128))

    observed_names = {
        "unpool-index-f16",
        "unpool-index-bf16-observed",
        "unpool-index-f32-observed",
        "unpool-index-f16-k3x2-s2x1-observed",
        "unpool-avg-f16-k3x2-s2x1-observed",
        "unpool-mask-f16-k3x2-s2x1",
        "unpool-index-f16-repeated-overlap-observed",
        "unpool-mask-f16-repeated-overlap-observed",
        "peripheral-bilinear-f16",
        "peripheral-argmin-negative-f16-observed",
        "peripheral-argmin-tie-f16-observed",
        "peripheral-argmin-nan-f16-observed",
        "peripheral-factorize-f32-observed",
        "peripheral-lut32-observed",
        "peripheral-randgen-f16-observed",
        "peripheral-elemmask-f16-observed",
    }
    observed = tuple(
        case for case in catalog.SAFE_CASES if case.is_observation
    )
    assert observed_names.issubset({case.name for case in observed})
    assert {
        case.name
        for case in observed
        if catalog._POOL_PADDED_BASE
        <= case.case_id
        <= catalog._POOL_TIE_END
    } == {
        "pool-avg-f16-k3x2-s2x1-padded-observed",
        "pool-sum-f16-k3x2-s2x1-padded-observed",
        "pool-max-f16-k3x2-s2x1-padded-observed",
        "pool-indexed-max-f16-k3x2-s2x1-padded-observed",
        "pool-min-f16-k3x2-s2x1-padded-observed",
        "pool-indexed-min-f16-k3x2-s2x1-padded-observed",
        "pool-indexed-max-f16-tie-observed",
        "pool-indexed-min-f16-tie-observed",
    }
    selected = runner.select_cases(
        types.SimpleNamespace(selected_cases=None, suite="observed")
    )
    assert selected == observed
    ct_capability = runner.select_cases(
        types.SimpleNamespace(selected_cases=None, suite="ct-capability")
    )
    assert len(ct_capability) == 90
    assert sum(not case.is_observation for case in ct_capability) == 72
    assert sum(case.is_observation for case in ct_capability) == 18
    assert ct_capability[0].name == "pool-indexed-max-f16-k3x2-s2x1"
    assert not ({case.case_id for case in ct_capability} & set(range(196, 204)))
    assert {case.case_id for case in ct_capability} == {
        case.case_id
        for case in catalog.SAFE_CASES
        if 143 <= case.case_id <= 248
    }
    raw_reduce_name = "reduce-sum-f16-n-raw-observed"
    try:
        runner.select_cases(
            types.SimpleNamespace(
                selected_cases=[raw_reduce_name],
                suite="safe",
            )
        )
    except RuntimeError as error:
        assert "deferred catalog rows" in str(error)
        assert raw_reduce_name in str(error)
    else:
        raise AssertionError("raw-axis Reduce bypassed host fail-closed selection")

    probe = (
        pathlib.Path(__file__).resolve().parent.parent
        / "Inputs"
        / "wafer_instruction_family_probe.c"
    ).read_text()
    assert "get_spm_memory_mapping" in probe
    assert "wafer_ifp_copy_spm_bytes" in probe
    assert "wafer_ifp_guard_mismatches" not in probe
    assert "output_ddr + WAFER_IFP_AUX_DDR_OFFSET" in probe
    assert "WAFER_IFP_REDUCE_RAW_BASE" not in probe
    assert "dimension = (offset & 1U) == 0U ? 3U : 5U" not in probe
    qualification_regression = runner.select_cases(
        types.SimpleNamespace(
            selected_cases=None,
            suite="qualification-regression",
        )
    )
    assert tuple(case.name for case in qualification_regression) == (
        runner.QUALIFICATION_REGRESSION_CASES
    )

    case = catalog.CASES_BY_NAME["peripheral-elemmask-f16-observed"]
    built = catalog.build_case_payload(case, sample=2)
    raw = bytearray([runner.OUTPUT_INITIAL_CANARY] * catalog.RESOURCE_BYTES)
    words = [0] * catalog.RECORD_WORDS
    for field, value in (
        ("MAGIC", catalog.RECORD_MAGIC),
        ("WORD_COUNT", catalog.RECORD_WORDS),
        ("STATUS", 0),
        ("CASE", case.case_id),
        ("DISPOSITION", case.disposition),
        ("FAMILY", case.family),
        ("DTYPE", case.dtype),
        ("ORACLE", case.oracle),
        ("RESULT_BYTES", case.result_bytes),
        ("OUTPUT_SPAN", case.output_span),
        ("AUX_SPAN", case.aux_span),
        ("SAMPLE", 2),
        ("REQUEST_GUARD", catalog.REQUEST_GUARD),
        ("OUTPUT_DDR_OFFSET", catalog.OUTPUT_DDR_OFFSET),
        ("SLOT_BYTES", catalog.SLOT_BYTES),
        ("BODY_OFFSET", catalog.BODY_OFFSET),
        (
            "STEP_FLAGS",
            catalog.STEP_TARGET_ISSUED
            | catalog.STEP_FINAL_FENCE_COMPLETED,
        ),
        ("RECORD_GUARD", catalog.RECORD_GUARD),
    ):
        words[catalog.REC[field]] = value
    struct.pack_into(f"<{catalog.RECORD_WORDS}Q", raw, 0, *words)
    begin = catalog.OUTPUT_DDR_OFFSET
    raw[begin : begin + catalog.SLOT_BYTES] = built.expected_output_slot
    expected_aux_slot = built.payload[
        3 * catalog.SLOT_BYTES : 4 * catalog.SLOT_BYTES
    ]
    raw[
        catalog.AUX_DDR_OFFSET :
        catalog.AUX_DDR_OFFSET + catalog.SLOT_BYTES
    ] = expected_aux_slot
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "observation.raw"
        output.write_bytes(raw)
        try:
            runner.validate_output(
                output,
                case,
                built.expected_output_slot,
                expected_aux_slot,
                sample=2,
            )
        except RuntimeError as error:
            assert "without any bounded writeback" in str(error)
        else:
            raise AssertionError("unchanged observation seed was accepted")
        raw[begin + catalog.BODY_OFFSET + 7] ^= 0x5A
        output.write_bytes(raw)
        runner.validate_output(
            output,
            case,
            built.expected_output_slot,
            expected_aux_slot,
            sample=2,
        )


def validate_reduce_capability_matrix() -> None:
    reduce_cases = tuple(
        case for case in catalog.CATALOG if 144 <= case.case_id <= 203
    )
    assert len(reduce_cases) == 60
    assert sum(case.is_safe for case in reduce_cases) == 52
    assert sum(case.is_observation for case in reduce_cases) == 0
    deferred = tuple(case for case in reduce_cases if not case.is_safe)
    assert len(deferred) == 8
    assert {case.case_id for case in deferred} == set(range(196, 204))
    assert all(
        case.reason_name == "REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT"
        and (case.result_bytes, case.output_span, case.aux_span) == (0, 0, 0)
        for case in deferred
    )

    for case in reduce_cases:
        if not case.is_safe:
            continue
        built = catalog.build_case_payload(case)
        offsets = catalog.reduce_exact_result_byte_offsets(case)
        assert len(offsets) == case.result_bytes
        assert len(set(offsets)) == case.result_bytes
        assert min(offsets) >= catalog.BODY_OFFSET
        assert max(offsets) < catalog.BODY_OFFSET + case.output_span

    sum_c = catalog.CASES_BY_NAME[
        "reduce-sum-f16-c-ncx-n1h2w3c65"
    ]
    built = catalog.build_case_payload(sum_c)
    offsets = catalog.reduce_exact_result_byte_offsets(sum_c)
    logical = bytes(built.expected_output_slot[offset] for offset in offsets)
    assert struct.unpack("<6e", logical) == (
        65.0,
        130.0,
        260.0,
        520.0,
        1040.0,
        65.0,
    )
    assert offsets != tuple(
        range(catalog.BODY_OFFSET, catalog.BODY_OFFSET + sum_c.result_bytes)
    )

    cx = catalog.CASES_BY_NAME["reduce-sum-f16-c-cx-w4c8"]
    cx_offsets = catalog.reduce_exact_result_byte_offsets(cx)
    assert cx_offsets == tuple(
        offset
        for scalar in (0, 8, 16, 24)
        for offset in (
            catalog.BODY_OFFSET + scalar,
            catalog.BODY_OFFSET + scalar + 1,
        )
    )


def main() -> int:
    assert len(catalog.SAFE_CASES) == 162
    assert len(catalog.CATALOG) == 170
    assert {case.case_id for case in catalog.SAFE_CASES} == (
        set(range(1, 30))
        | set(range(100, 144))
        | set(range(144, 196))
        | {
            205,
            206,
            208,
            209,
            212,
            214,
            215,
            217,
            218,
            220,
            221,
            222,
            223,
            224,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
        }
        | set(range(236, 249))
    )
    assert {
        (case.symbol, case.reason_name)
        for case in catalog.CATALOG
        if not case.is_safe
    } == {
        (symbol, "REASON_ISOLATED_POSSIBLE_PERMANENT_WAIT")
        for symbol in (
            "REDUCE_SUM_F16_N_RAW",
            "REDUCE_SUM_F16_HWC_RAW",
            "REDUCE_AVG_F16_N_RAW",
            "REDUCE_AVG_F16_HWC_RAW",
            "REDUCE_MAX_F16_N_RAW",
            "REDUCE_MAX_F16_HWC_RAW",
            "REDUCE_MIN_F16_N_RAW",
            "REDUCE_MIN_F16_HWC_RAW",
        )
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
    validate_ct_add_f16_tail130_oracle()
    validate_ct_add_bf16_tail130_oracle()
    validate_ct_add_f32_oracle()
    validate_ct_add_special_f16_oracle()
    validate_ct_add_special_bf16_oracle()
    validate_gemm_padding_domain()
    validate_gemm_dtype_oracles()
    validate_gemm_bf16_accum_round_oracle()
    validate_gemm_f16_accum_round_oracle()
    validate_gemm_f16_m4_oracle()
    validate_gemm_f16_batch2_m8_oracle()
    validate_gemm_bf16_batch2_m8_oracle()
    validate_gemm_f16_n17_oracle()
    validate_gemm_f16_k17_oracle()
    validate_gemm_bf16_k17_oracle()
    validate_gemm_f16_n65_oracle()
    validate_gemm_bf16_n65_oracle()
    validate_gemm_f16_oriented_nt_oracle()
    validate_gemm_bf16_oriented_nt_oracle()
    validate_gemm_f16_oriented_tn_oracle()
    validate_gemm_f16_oriented_tt_oracle()
    validate_gemm_f16_psum_oracle()
    validate_arg_extrema_composite_oracles()
    validate_argmin_pending_domain_observations()
    validate_probe_seed_is_ncc_local_and_completed()
    validate_argmin_bounded_probe_protocol()
    validate_conv_oracle()
    validate_pool_max_oracle()
    validate_unpool_rows()
    validate_unpool_capability_rows()
    validate_repeated_unpool_bounded_oracle()
    validate_img2col_oracle()
    validate_lut16_oracle()
    validate_reduce_capability_matrix()
    validate_extended_pool_unpool_oracles()
    validate_pool_capability_matrix()
    validate_peripheral_exact_and_observation_rows()
    validate_work_dir_cleanup_is_bounded()
    validate_matching_tile_completion()

    print("wafer_instruction_family_catalog_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
