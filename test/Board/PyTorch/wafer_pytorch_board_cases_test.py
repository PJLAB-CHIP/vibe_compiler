#!/usr/bin/env python3
"""Contract tests for framework-owned PyTorch board cases."""

from __future__ import annotations

import collections
import inspect
import math
import pathlib
import re
import tempfile
import types
import unittest
from unittest import mock

import torch

import wafer_board_pytorch_test as board_runner
import wafer_pytorch_board_cases as cases


class FakeSpmdModule:
    def __init__(self) -> None:
        self.marks: list[tuple[object, tuple[object, ...]]] = []

    def mark_sharding(
        self, value: object, _mesh: object, spec: tuple[object, ...]
    ) -> None:
        self.marks.append((value, spec))


class PyTorchBoardCasesTest(unittest.TestCase):
    def test_no_card_runner_materializes_both_functional_decode_packages(
        self,
    ) -> None:
        step_one_outputs = (
            torch.zeros((1, 1, 1), dtype=torch.float16),
            torch.ones((1, 1, 2, 1), dtype=torch.float16),
            torch.full((1, 1, 2, 1), 2.0, dtype=torch.float16),
        )
        step_two_outputs = (
            torch.zeros((1, 1, 1), dtype=torch.float16),
            torch.ones((1, 1, 3, 1), dtype=torch.float16),
            torch.full((1, 1, 3, 1), 2.0, dtype=torch.float16),
        )
        received_state: list[tuple[torch.Tensor, ...]] = []

        def make_step_two(
            outputs: tuple[torch.Tensor, ...],
        ) -> cases.PyTorchBoardCase:
            received_state.append(outputs)
            return cases.PyTorchBoardCase(
                name="decode-step-two",
                rank_count=1,
                dtype=torch.float16,
                inputs=(outputs[1], outputs[2]),
                expected_outputs_factory=lambda: step_two_outputs,
                export_program=lambda _path: None,
                required_structured_ir=(),
                expected_all_reduce_count=0,
                comparison_policy=cases.ATTENTION_COMPARISON,
            )

        step_one = cases.PyTorchBoardCase(
            name="decode-step-one",
            rank_count=1,
            dtype=torch.float16,
            inputs=(torch.zeros((1,), dtype=torch.float16),),
            expected_outputs_factory=lambda: step_one_outputs,
            export_program=lambda _path: None,
            required_structured_ir=(),
            expected_all_reduce_count=0,
            comparison_policy=cases.ATTENTION_COMPARISON,
            continuation_factory=make_step_two,
        )
        args = types.SimpleNamespace(
            case="attention-decode-kv-cache",
            dtype="float16",
            seed=17,
            wafer_compile=pathlib.Path("wafer-compile"),
            wafer_run=pathlib.Path("wafer-run"),
            work_dir=pathlib.Path("work"),
            dump_compiler_ir=None,
            compile_timing=False,
            require_implementation_alternative=False,
            no_card=True,
            device_id=0,
            expected_runtime_version=None,
            expected_device_name=None,
            expected_pci_bus_id=None,
            expected_tile_count=None,
            expected_runtime_library_sha256=None,
            completion_timeout_ms=1,
            repeat=1,
        )

        def prepare_step(
            _args: object,
            case: cases.PyTorchBoardCase,
            **_kwargs: object,
        ) -> tuple[object, ...]:
            outputs = (
                step_one_outputs
                if case is step_one
                else step_two_outputs
            )
            return outputs, [], {}, set(), {}

        with (
            mock.patch.object(board_runner, "parse_args", return_value=args),
            mock.patch.object(
                board_runner.board_cases,
                "make_case",
                return_value=step_one,
            ),
            mock.patch.object(board_runner, "prepare_work_dir"),
            mock.patch.object(
                board_runner,
                "case_step_paths",
                side_effect=[
                    (pathlib.Path("s1"), pathlib.Path("src1"), pathlib.Path("pkg1")),
                    (pathlib.Path("s2"), pathlib.Path("src2"), pathlib.Path("pkg2")),
                ],
            ),
            mock.patch.object(
                board_runner,
                "prepare_case_step",
                side_effect=prepare_step,
            ) as prepare_mock,
            mock.patch.object(
                board_runner,
                "base_runtime_command",
                return_value=["wafer-run"],
            ),
            mock.patch.object(
                board_runner,
                "run",
                return_value=types.SimpleNamespace(stdout=""),
            ) as run_mock,
            mock.patch.object(board_runner, "verify_no_card"),
        ):
            self.assertEqual(board_runner.main(), 0)

        self.assertEqual(prepare_mock.call_count, 2)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(len(received_state), 1)
        for actual, expected in zip(
            received_state[0], step_one_outputs, strict=True
        ):
            self.assertTrue(torch.equal(actual, expected))

    def test_board_continuation_reads_the_step_one_captures(self) -> None:
        expected = (
            torch.arange(4, dtype=torch.float16).reshape(1, 1, 4),
            torch.arange(8, dtype=torch.float16).reshape(1, 1, 4, 2),
            torch.arange(8, 16, dtype=torch.float16).reshape(1, 1, 4, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            paths = {}
            for index, tensor in enumerate(expected):
                path = root / f"output-{index}.raw"
                cases.common.write_tensor_raw(path, tensor)
                paths[(0, index)] = path
            actual = board_runner.read_rank_one_continuation_outputs(
                paths, expected
            )
        for actual_tensor, expected_tensor in zip(
            actual, expected, strict=True
        ):
            self.assertTrue(torch.equal(actual_tensor, expected_tensor))

    def test_q49_board_ready_matrix_is_callable_and_not_deferred(self) -> None:
        expected = {
            ("heterogeneous-tiling-dataflow", "float16", 1),
            ("heterogeneous-tiling-dataflow-tp16", "bfloat16", 16),
            ("attention-prefill", "float16", 1),
            ("attention-prefill", "bfloat16", 1),
            ("attention-decode-kv-cache", "float16", 1),
            ("attention-decode-kv-cache", "bfloat16", 1),
            ("hf-megatron-transformer-block", "float16", 16),
            ("hf-megatron-transformer-block", "bfloat16", 16),
        }
        actual = {
            (entry.case_name, entry.dtype_name, entry.rank_count)
            for entry in cases.Q49_BOARD_READY_WORKLOADS
        }
        self.assertEqual(actual, expected)
        self.assertTrue(
            all(
                entry.case_name in cases.CASE_FACTORIES
                for entry in cases.Q49_BOARD_READY_WORKLOADS
            )
        )
        self.assertEqual(
            {
                (entry.case_name, entry.dtype_name): entry.package_count
                for entry in cases.Q49_BOARD_READY_WORKLOADS
                if entry.case_name == "attention-decode-kv-cache"
            },
            {
                ("attention-decode-kv-cache", "float16"): 2,
                ("attention-decode-kv-cache", "bfloat16"): 2,
            },
        )
        self.assertEqual(
            sum(
                entry.package_count
                for entry in cases.Q49_BOARD_READY_WORKLOADS
            ),
            10,
        )

        cmake = (
            pathlib.Path(__file__).resolve().parents[3]
            / "test"
            / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        for entry in cases.Q49_BOARD_READY_WORKLOADS:
            match = re.search(
                rf"add_test\(NAME {re.escape(entry.ctest_name)}\n"
                r"(?P<body>.*?)\n\s*\)",
                cmake,
                re.DOTALL,
            )
            self.assertIsNotNone(match, entry.ctest_name)
            body = match.group("body")
            self.assertIn(f"--case {entry.case_name}", body)
            self.assertIn(f"--dtype {entry.dtype_name}", body)
            self.assertIn("--no-card", body)
            self.assertEqual(
                "--require-implementation-alternative" in body,
                entry.require_implementation_alternative,
            )

        runner_source = inspect.getsource(board_runner.main)
        self.assertNotIn("torch_eager_reference=deferred", runner_source)
        prepare_source = inspect.getsource(board_runner.prepare_case_step)
        self.assertIn("case.materialize_expected_outputs()", prepare_source)
        self.assertLess(
            runner_source.index(") = prepare_case_step("),
            runner_source.index("if args.no_card:"),
        )

    def test_heterogeneous_tiling_dataflow_has_mixed_output_domains(self) -> None:
        case = cases._heterogeneous_tiling_rank_one(torch.float16, seed=19)
        self.assertEqual(case.rank_count, 1)
        matrix, row_reduction = case.materialize_expected_outputs()
        self.assertEqual(matrix.shape, (96, 80))
        self.assertEqual(row_reduction.shape, (96,))
        self.assertEqual(matrix.dtype, torch.float16)
        self.assertTrue(torch.isfinite(matrix).all())
        self.assertTrue(torch.isfinite(row_reduction).all())

        tp16 = cases._heterogeneous_tiling_tp16(torch.bfloat16, seed=19)
        self.assertEqual(tp16.rank_count, cases.RANK_COUNT)
        bf16_matrix, bf16_row_reduction = tp16.materialize_expected_outputs()
        self.assertEqual(bf16_matrix.dtype, torch.bfloat16)
        self.assertEqual(bf16_row_reduction.dtype, torch.bfloat16)

    def test_heterogeneous_tiling_export_is_source_derived(self) -> None:
        case = cases._heterogeneous_tiling_rank_one(torch.float16, seed=23)
        with tempfile.TemporaryDirectory() as directory:
            program = pathlib.Path(directory) / "heterogeneous-program"
            case.export_program(program)
            stablehlo = (program / "functions" / "forward.mlir").read_text()
        self.assertIn("stablehlo.dot_general", stablehlo)
        self.assertIn("stablehlo.maximum", stablehlo)
        self.assertIn("stablehlo.reduce", stablehlo)
        self.assertNotIn("wafer", stablehlo.lower())

    def test_bfloat16_export_preserves_parameter_and_buffer_storage(self) -> None:
        class StatefulBFloat16Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor([1.0, -2.0], dtype=torch.bfloat16),
                    requires_grad=False,
                )
                self.register_buffer(
                    "offset",
                    torch.tensor([0.5, -0.25], dtype=torch.bfloat16),
                )

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value * self.weight + self.offset

        module = StatefulBFloat16Module().eval()
        inputs = (torch.ones((2,), dtype=torch.bfloat16),)
        numpy_module = cases.capture._import_numpy()
        with tempfile.TemporaryDirectory() as directory:
            program = pathlib.Path(directory) / "bf16-state-program"
            cases._save_exported_program(program, module, inputs)
            for name, tensor in module.state_dict().items():
                actual = numpy_module.load(
                    program / "data" / name, allow_pickle=False
                )
                expected = cases.capture._torch_to_workload_storage(
                    torch, numpy_module, tensor, "bfloat16"
                )
                self.assertEqual(actual.dtype, numpy_module.dtype("|V2"))
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_split_kv_lse_merge_matches_pytorch_sdpa(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(73)
        query = torch.randn((1, 2, 3, 8), generator=generator)
        key = torch.randn((1, 2, 11, 8), generator=generator)
        value = torch.randn((1, 2, 11, 8), generator=generator)
        mask = torch.zeros((1, 1, 3, 11))
        mask[..., 0, -2:] = float("-inf")
        reference = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask
        )

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(8)
        scores = scores + mask
        partial_outputs: list[torch.Tensor] = []
        partial_lses: list[torch.Tensor] = []
        for begin, end in ((0, 4), (4, 8), (8, 11)):
            partial_scores = scores[..., begin:end]
            partial_lses.append(torch.logsumexp(partial_scores, dim=-1))
            partial_outputs.append(
                torch.matmul(
                    torch.softmax(partial_scores, dim=-1),
                    value[..., begin:end, :],
                )
            )
        lses = torch.stack(partial_lses, dim=0)
        global_lse = torch.logsumexp(lses, dim=0)
        weights = torch.exp(lses - global_lse.unsqueeze(0)).unsqueeze(-1)
        merged = torch.sum(
            weights * torch.stack(partial_outputs, dim=0), dim=0
        )
        torch.testing.assert_close(merged, reference, rtol=1.0e-5, atol=1.0e-6)

    def test_hf_prefill_uses_the_official_additive_mask_values(self) -> None:
        source = inspect.getsource(cases._read_only_attention)
        self.assertNotIn('float("-inf")', source)
        case = cases._attention_prefill(torch.float16, seed=37)
        mask = case.inputs[-1]
        self.assertEqual(mask.shape, (1, 1, 1024, 1024))
        self.assertFalse(torch.isinf(mask).any())
        self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(
            mask[0, 0, 0, 1].item(), torch.finfo(torch.float16).min
        )
        self.assertEqual(mask[0, 0, -1, -1].item(), 0.0)

    def test_hf_functional_decode_threads_exact_updated_cache(self) -> None:
        case = cases._attention_decode_kv_cache(torch.float16, seed=41)
        self.assertEqual(case.name, "attention-decode-kv-cache")
        self.assertEqual(case.rank_count, 1)
        attention, updated_key, updated_value = (
            case.expected_outputs_factory()
        )
        hidden_states, past_key, past_value, _ = case.inputs
        self.assertEqual(hidden_states.shape, (1, 1, 4096))
        self.assertEqual(past_key.shape, (1, 32, 1023, 128))
        self.assertEqual(past_value.shape, (1, 32, 1023, 128))
        self.assertEqual(attention.shape[:-1], hidden_states.shape[:-1])
        self.assertEqual(updated_key.shape[-2], past_key.shape[-2] + 1)
        self.assertEqual(updated_value.shape[-2], past_value.shape[-2] + 1)
        self.assertTrue(torch.equal(updated_key[..., :-1, :], past_key))
        self.assertTrue(torch.equal(updated_value[..., :-1, :], past_value))
        self.assertIsNotNone(case.continuation_factory)
        continuation = case.continuation_factory(
            (attention, updated_key, updated_value)
        )
        self.assertEqual(
            continuation.name, "attention-decode-kv-cache-continuation"
        )
        self.assertIsNone(continuation.continuation_factory)
        self.assertTrue(torch.equal(continuation.inputs[1], updated_key))
        self.assertTrue(torch.equal(continuation.inputs[2], updated_value))
        _, twice_updated_key, twice_updated_value = (
            continuation.materialize_expected_outputs()
        )
        self.assertEqual(twice_updated_key.shape[-2], 1025)
        self.assertEqual(twice_updated_value.shape[-2], 1025)
        self.assertTrue(
            torch.equal(twice_updated_key[..., :-1, :], updated_key)
        )
        self.assertTrue(
            torch.equal(twice_updated_value[..., :-1, :], updated_value)
        )
        with tempfile.TemporaryDirectory() as directory:
            step_one_program = pathlib.Path(directory) / "decode-step-one"
            step_two_program = pathlib.Path(directory) / "decode-step-two"
            case.export_program(step_one_program)
            continuation.export_program(step_two_program)
            step_one_stablehlo = (
                step_one_program / "functions" / "forward.mlir"
            ).read_text()
            step_two_stablehlo = (
                step_two_program / "functions" / "forward.mlir"
            ).read_text()
        for stablehlo in (step_one_stablehlo, step_two_stablehlo):
            self.assertGreaterEqual(
                stablehlo.count("stablehlo.concatenate"), 2
            )
            self.assertNotIn("stablehlo.cosine", stablehlo)
            self.assertNotIn("stablehlo.sine", stablehlo)
        self.assertIn("tensor<1x32x1023x128xf16>", step_one_stablehlo)
        self.assertIn("tensor<1x32x1024x128xf16>", step_two_stablehlo)

    def test_hf_block_uses_official_math_and_official_precomputed_inputs(
        self,
    ) -> None:
        from transformers import LlamaConfig
        from transformers.masking_utils import create_causal_mask
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaMLP,
            LlamaRMSNorm,
            LlamaRotaryEmbedding,
        )

        helper_source = inspect.getsource(
            cases.capture._make_hf_llama_decoder_block_module
        )
        for forbidden in (
            "wafer_accumulation_dtype",
            "torch_module.softmax",
            "torch_module.rsqrt",
            "torch_module.exp",
            'float("-inf")',
            "def _rotate_half",
            "def _silu",
        ):
            self.assertNotIn(forbidden, helper_source)

        for dtype, dtype_name in (
            (torch.float16, "float16"),
            (torch.bfloat16, "bfloat16"),
        ):
            config = dict(
                cases.capture.load_hf_transformer_config(
                    cases.HF_LLAMA2_7B_CONFIG
                )
            )
            config.update(
                hidden_size=16,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=4,
                torch_dtype=dtype_name,
            )
            module = cases.capture._make_hf_llama_decoder_block_module(
                torch, config, sequence_length=4
            ).eval()
            self.assertIsInstance(module.decoder_layer, LlamaDecoderLayer)
            self.assertIsInstance(
                module.decoder_layer.input_layernorm, LlamaRMSNorm
            )
            self.assertIsInstance(module.decoder_layer.mlp, LlamaMLP)

            hf_config = LlamaConfig.from_dict(config)
            hf_config._attn_implementation = "eager"
            positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
            rotary_input = torch.empty((1, 4, 16), dtype=dtype)
            rotary = LlamaRotaryEmbedding(hf_config).eval()
            with torch.no_grad():
                expected_cos, expected_sin = rotary(rotary_input, positions)
                expected_mask = create_causal_mask(
                    config=hf_config,
                    inputs_embeds=rotary_input,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=positions,
                )
            self.assertTrue(torch.equal(module.rotary_cos, expected_cos))
            self.assertTrue(torch.equal(module.rotary_sin, expected_sin))
            self.assertTrue(torch.equal(module.causal_mask, expected_mask))

            value = torch.randn(
                (1, 4, 16),
                dtype=dtype,
                generator=torch.Generator(device="cpu").manual_seed(7),
            )
            with torch.no_grad():
                result = module(value)
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(torch.isfinite(result).all())

    def test_hf_megatron_board_dtype_rejects_f32(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "float16 or bfloat16"):
            cases._hf_megatron_transformer_block(torch.float32, seed=11)

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
