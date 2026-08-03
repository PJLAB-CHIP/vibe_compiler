#!/usr/bin/env python3
"""Contract tests for framework-owned PyTorch board cases."""

from __future__ import annotations

import collections
import unittest

import torch

import wafer_pytorch_board_cases as cases


class FakeSpmdModule:
    def __init__(self) -> None:
        self.marks: list[tuple[object, tuple[object, ...]]] = []

    def mark_sharding(
        self, value: object, _mesh: object, spec: tuple[object, ...]
    ) -> None:
        self.marks.append((value, spec))


class PyTorchBoardCasesTest(unittest.TestCase):
    def test_hf_megatron_parameter_specs_are_explicit_and_complete(self) -> None:
        config = dict(
            cases.capture.load_hf_transformer_config(
                cases.HF_LLAMA2_7B_CONFIG
            )
        )
        self.assertEqual(int(config["hidden_size"]), 4096)
        self.assertEqual(int(config["intermediate_size"]), 11008)
        self.assertEqual(int(config["num_attention_heads"]), 32)
        self.assertEqual(int(config["head_dim"]), 128)
        module = cases.capture._make_hf_llama_decoder_block_module(
            torch,
            config,
            sequence_length=cases.HF_LLAMA2_7B_SEQUENCE_LENGTH,
        )
        input_tensor = torch.empty((1, 16, 4096), dtype=torch.float16)
        spmd = FakeSpmdModule()
        cases.capture.apply_hf_megatron_sharding_marks(
            spmd_module=spmd,
            mesh=object(),
            input_tensor=input_tensor,
            reference_module=module,
        )

        self.assertIs(spmd.marks[0][0], input_tensor)
        self.assertEqual(
            spmd.marks[0][1], cases.capture.HF_MEGATRON_INPUT_SPEC
        )
        parameter_marks = spmd.marks[1:]
        self.assertEqual(
            {id(value) for value, _ in parameter_marks},
            {id(parameter) for parameter in module.parameters()},
        )
        self.assertEqual(
            collections.Counter(spec for _, spec in parameter_marks),
            collections.Counter(
                {
                    cases.capture.HF_MEGATRON_REPLICATED_VECTOR_SPEC: 2,
                    cases.capture.HF_MEGATRON_COLUMN_PARALLEL_WEIGHT_SPEC: 5,
                    cases.capture.HF_MEGATRON_ROW_PARALLEL_WEIGHT_SPEC: 2,
                }
            ),
        )

    def test_torch_random_seed_and_llama_policy_reject_fault(self) -> None:
        first = cases._random_tensor(
            (8, 8),
            dtype=torch.float16,
            generator=torch.Generator(device="cpu").manual_seed(20260803),
        )
        repeat = cases._random_tensor(
            (8, 8),
            dtype=torch.float16,
            generator=torch.Generator(device="cpu").manual_seed(20260803),
        )
        different = cases._random_tensor(
            (8, 8),
            dtype=torch.float16,
            generator=torch.Generator(device="cpu").manual_seed(20260804),
        )
        self.assertTrue(torch.equal(first, repeat))
        self.assertFalse(torch.equal(first, different))
        fault = first.clone()
        fault[0, 0] += torch.tensor(1.0, dtype=fault.dtype)
        with self.assertRaises(AssertionError):
            cases.common.assert_tensor_matches(
                fault,
                first,
                policy=cases.HF_LLAMA2_7B_COMPARISON,
                context="HF Megatron fault injection",
            )


if __name__ == "__main__":
    unittest.main()
