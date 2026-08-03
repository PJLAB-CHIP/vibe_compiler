#!/usr/bin/env python3
"""Contract tests for PyTorch board tensor read/write and comparison."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import torch

import wafer_pytorch_board_common as common


class PyTorchBoardCommonTest(unittest.TestCase):
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
