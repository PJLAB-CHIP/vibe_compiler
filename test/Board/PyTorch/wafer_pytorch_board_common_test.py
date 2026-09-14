#!/usr/bin/env python3
"""Contract tests for PyTorch board tensor read/write and comparison."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import torch

import wafer_pytorch_board_common as common
import wafer_board_pytorch_test as board_runner


class CompilerIRDumpTest(unittest.TestCase):
    def test_inactive_function_symbol_and_ddr_result_do_not_imply_work(self) -> None:
        for symbol in ("entry", "other_symbol"):
            ir = f"""module {{
              func.func @{symbol}() -> memref<1x2x1031xf16, #wafer.memory<ddr, tensor>> {{
                %result = memref.alloc() : memref<1x2x1031xf16, #wafer.memory<ddr, tensor>>
                return %result : memref<1x2x1031xf16, #wafer.memory<ddr, tensor>>
              }}
            }}"""
            board_runner.verify_pre_instruction_boundary(ir, ir)
            with self.assertRaisesRegex(RuntimeError, "gained instructions"):
                board_runner.verify_pre_instruction_boundary(ir, "wafer.instr.fill")

    def test_instruction_dump_cannot_replace_pre_instruction_evidence(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "pre-Instr"):
            board_runner.verify_pre_instruction_boundary("wafer.instr.fill", "wafer.instr.fill")


class PyTorchBoardCommonTest(unittest.TestCase):
    def test_integer_ports_preserve_large_values_and_reject_one_bit_errors(self) -> None:
        # Exercise real-size/tail buffers and values above floating-point exact
        # integer ranges. Neither raw transport nor comparison may cast them.
        for extent in (1024, 1025, 1031):
            for dtype, value in ((torch.int32, 10000), (torch.int64, 2**60),
                                 (torch.uint8, 200), (torch.uint32, 2**31)):
                with self.subTest(extent=extent, dtype=dtype):
                    expected = torch.full((1, extent, 4), value, dtype=dtype)
                    actual = expected.clone()
                    actual[0, -1, -1] = value + 1
                    with tempfile.TemporaryDirectory() as directory:
                        path = pathlib.Path(directory) / "integer.raw"
                        common.write_tensor_raw(path, expected)
                        common.assert_raw_capture_matches(path, expected, context="integer raw")
                        common.write_tensor_raw(path, actual)
                        for policy in (common.PYTORCH_DEFAULT,
                                       common.ComparisonPolicy(rtol=0.002, atol=0.004)):
                            with self.assertRaisesRegex(AssertionError, "1/.* elements"):
                                common.assert_raw_capture_matches(
                                    path, expected, context="integer tail", policy=policy,
                                )
                    error = board_runner.summarize_output_error(actual, expected)
                    self.assertEqual(error["mismatched_elements"], 1)
                    self.assertIsNone(error["max_abs_error"])
                    self.assertEqual(
                        board_runner.summarize_output_error(expected, expected)["max_abs_error"], 0,
                    )

    def test_round_trip_preserves_dtype(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20260803)
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int32,
        ):
            expected = (
                torch.randn((4, 6), dtype=dtype, generator=generator)
                if dtype.is_floating_point
                else torch.randint(
                    -16, 17, (4, 6), dtype=dtype, generator=generator
                )
            )
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "tensor.raw"
                common.write_tensor_raw(path, expected)
                common.assert_raw_capture_matches(
                    path, expected, context=f"round-trip {dtype}"
                )

    def test_round_trip_uses_logical_contiguous_view_only(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20260805)
        storage = torch.randn(
            (8, 6), dtype=torch.float32, generator=generator
        )
        expected = storage[2:6, :]
        self.assertTrue(expected.is_contiguous())
        self.assertNotEqual(expected.storage_offset(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "view.raw"
            common.write_tensor_raw(path, expected)
            self.assertEqual(path.stat().st_size, expected.numel() * 4)
            common.assert_raw_capture_matches(
                path, expected, context="logical contiguous view"
            )

    def test_fault_injection_is_rejected(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20260804)
        expected = torch.randn(
            (4, 4), dtype=torch.float16, generator=generator
        )
        actual = expected.clone()
        actual[2, 1] += torch.tensor(1, dtype=torch.float16)
        with self.assertRaises(AssertionError):
            common.assert_tensor_matches(
                actual, expected, context="fault-injected output"
            )

    def test_shape_dtype_and_byte_mismatch_are_rejected(self) -> None:
        expected = torch.zeros((2, 3), dtype=torch.float16)
        with self.assertRaises(RuntimeError):
            common.assert_tensor_matches(
                expected.reshape(3, 2), expected, context="bad shape"
            )
        with self.assertRaises(RuntimeError):
            common.assert_tensor_matches(
                expected.float(), expected, context="bad dtype"
            )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "short.raw"
            path.write_bytes(b"\x00")
            with self.assertRaises(RuntimeError):
                common.assert_raw_capture_matches(
                    path, expected, context="short capture"
                )

    def test_pytorch_directory_has_no_numpy_reference_import(self) -> None:
        root = pathlib.Path(__file__).resolve().parent
        forbidden = ("import " + "numpy", "from " + "numpy")
        offenders = []
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in forbidden):
                offenders.append(path.name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
