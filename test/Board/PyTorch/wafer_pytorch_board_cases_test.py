#!/usr/bin/env python3
"""Contract tests for framework-owned PyTorch board cases."""

from __future__ import annotations

import inspect
import json
import pathlib
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

import wafer_board_pytorch_test as board_runner
import wafer_pytorch_board_cases as cases


class PyTorchBoardCasesTest(unittest.TestCase):
    def test_search_policy_evidence_requires_actual_fusion(self) -> None:
        case = cases.PyTorchBoardCase(
            name="fusion-required",
            num_partitions=1,
            dtype=torch.float16,
            inputs=(torch.zeros((1,), dtype=torch.float16),),
            expected_outputs_factory=lambda: (
                torch.zeros((1,), dtype=torch.float16),
            ),
            export_program=lambda _path: None,
            required_structured_ir=(),
            expected_all_reduce_count=0,
            comparison_policy=cases.ATTENTION_COMPARISON,
            minimum_search_actual_fused_edges=1,
        )
        board_runner.validate_compiler_search_evidence(
            "wafer-compile: whole-card-selection admitted=1 "
            "actual_fused_edges=3 makespan_ps=7\n",
            case,
            "search",
        )
        with self.assertRaisesRegex(RuntimeError, "required operator fusion"):
            board_runner.validate_compiler_search_evidence(
                "wafer-compile: whole-card-selection admitted=1 "
                "actual_fused_edges=0 makespan_ps=7\n",
                case,
                "search",
            )
        # The conservative comparison baseline is intentionally not judged by
        # the search winner's optimization-effectiveness contract.
        board_runner.validate_compiler_search_evidence("", case, "none")

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
                num_partitions=1,
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
            num_partitions=1,
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
            optimization_policy="none",
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
                side_effect=lambda *_args: ["wafer-run"],
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
        self.assertEqual(
            [call.kwargs["source"] for call in prepare_mock.call_args_list],
            [pathlib.Path("src1"), pathlib.Path("src2")],
        )
        self.assertEqual(
            [call.kwargs["package"] for call in prepare_mock.call_args_list],
            [pathlib.Path("pkg1"), pathlib.Path("pkg2")],
        )
        for call in run_mock.call_args_list:
            self.assertEqual(
                call.args[0],
                [
                    "wafer-run",
                    "--no-card",
                    "--direct-dte-status-abi",
                    "wafer-direct-dte-status-v2",
                    "--supports-host-watchdog",
                ],
            )
        self.assertEqual(len(received_state), 1)
        for actual, expected in zip(
            received_state[0], step_one_outputs, strict=True
        ):
            self.assertTrue(torch.equal(actual, expected))

    def test_runner_exposes_and_forwards_the_public_optimization_policy(
        self,
    ) -> None:
        required_args = [
            "wafer_board_pytorch_test.py",
            "--case",
            "single-card-gemm",
            "--wafer-compile",
            "wafer-compile",
            "--wafer-run",
            "wafer-run",
            "--work-dir",
            "work",
        ]
        with mock.patch.object(sys, "argv", required_args):
            self.assertEqual(
                board_runner.parse_args().optimization_policy,
                "search",
            )
        with mock.patch.object(
            sys,
            "argv",
            [*required_args, "--optimization-policy", "none"],
        ):
            self.assertEqual(
                board_runner.parse_args().optimization_policy,
                "none",
            )
        self.assertEqual(
            cases.OPTIMIZATION_POLICIES,
            ("search", "none"),
        )

        case = types.SimpleNamespace(
            num_partitions=1,
            export_program=mock.Mock(),
            materialize_expected_outputs=mock.Mock(
                return_value=(torch.zeros((1,), dtype=torch.float16),)
            ),
        )
        args = types.SimpleNamespace(
            wafer_compile=pathlib.Path("wafer-compile"),
            compile_timing=False,
            optimization_policy="none",
        )
        with (
            mock.patch.object(
                board_runner,
                "run",
                return_value=types.SimpleNamespace(
                    stdout=(
                        "wafer-compile: wrote verified package with "
                        "num-partitions=1 physical-tiles=16"
                    ),
                    stderr="",
                ),
            ) as run_mock,
            mock.patch.object(board_runner, "validate_structured_program"),
            mock.patch.object(
                board_runner,
                "prepare_runtime_payloads",
                return_value=([], {}, set(), {}),
            ),
        ):
            board_runner.prepare_case_step(
                args,
                case,
                step_index=0,
                step_dir=pathlib.Path("step"),
                source=pathlib.Path("source"),
                package=pathlib.Path("package"),
                dump_compiler_ir=None,
            )
        compile_command = run_mock.call_args.args[0]
        self.assertIn("--optimization-policy=none", compile_command)
        self.assertEqual(
            [arg for arg in compile_command if arg.startswith("--optimization-")],
            ["--optimization-policy=none"],
        )

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
                paths[index] = path
            actual = board_runner.read_single_card_continuation_outputs(
                paths, expected
            )
        for actual_tensor, expected_tensor in zip(
            actual, expected, strict=True
        ):
            self.assertTrue(torch.equal(actual_tensor, expected_tensor))

    def test_runner_accepts_nonidentity_physical_tile_binding(self) -> None:
        entries = [
            {
                "card_id": 0,
                "tile_id": 1 if launch_slot == 0 else (
                    0 if launch_slot == 1 else launch_slot
                ),
                "launch_slot": launch_slot,
                "completion": "return_after_local_drain",
            }
            for launch_slot in range(board_runner.PHYSICAL_TILE_COUNT)
        ]
        manifest = {
            "schema_version": 8,
            "target": {"identity": board_runner.TARGET_IDENTITY},
            "card_count": 1,
            "tile_count": board_runner.PHYSICAL_TILE_COUNT,
            "resources": [
                {
                    "id": 7,
                    "scope": {"kind": "card", "card_id": 0},
                    "role": "user_input",
                    "role_index": 0,
                    "host_visible": True,
                },
                {
                    "id": 8,
                    "scope": {"kind": "card", "card_id": 0},
                    "role": "output",
                    "role_index": 0,
                    "host_visible": True,
                },
            ],
            "entries": entries,
        }
        with tempfile.TemporaryDirectory() as directory:
            package = pathlib.Path(directory)
            manifest_path = package / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            resources, output_ids = board_runner._manifest_resources(package)
            self.assertEqual(
                set(resources), {("user_input", 0), ("output", 0)}
            )
            self.assertEqual(output_ids, {8})

            entries[1]["tile_id"] = entries[0]["tile_id"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                board_runner._manifest_resources(package)

        self.assertEqual(
            board_runner.base_runtime_command(
                pathlib.Path("wafer-run"), pathlib.Path("package")
            ),
            ["wafer-run", "--package-dir", "package"],
        )

    def test_source_no_card_matrix_is_callable_and_not_deferred(self) -> None:
        expected_search = {
            ("single-card-gemm", "float16", 1),
            ("single-card-gemm", "bfloat16", 1),
            ("heterogeneous-tiling-dataflow", "float16", 1),
            ("heterogeneous-tiling-dataflow", "bfloat16", 1),
            ("conv-mixed-dag", "float16", 1),
            ("conv-mixed-dag", "bfloat16", 1),
            ("attention-prefill", "float16", 1),
            ("attention-prefill", "bfloat16", 1),
            ("attention-decode-kv-cache", "float16", 1),
            ("attention-decode-kv-cache", "bfloat16", 1),
            ("llama-2-7b-block", "float16", 1),
            ("llama-2-7b-block", "bfloat16", 1),
        }
        expected = {
            (*workload, "search")
            for workload in expected_search
        } | {
            ("attention-prefill", "float16", 1, "none"),
            ("attention-decode-kv-cache", "float16", 1, "none"),
            ("llama-2-7b-block", "float16", 1, "none"),
        }
        actual = {
            (
                entry.case_name,
                entry.dtype_name,
                entry.num_partitions,
                entry.optimization_policy,
            )
            for entry in cases.SOURCE_NO_CARD_WORKLOADS
        }
        self.assertEqual(actual, expected)
        self.assertTrue(
            all(
                entry.case_name in cases.CASE_FACTORIES
                for entry in cases.SOURCE_NO_CARD_WORKLOADS
            )
        )
        self.assertEqual(
            {
                (
                    entry.case_name,
                    entry.dtype_name,
                    entry.optimization_policy,
                ): entry.package_count
                for entry in cases.SOURCE_NO_CARD_WORKLOADS
                if entry.case_name == "attention-decode-kv-cache"
            },
            {
                ("attention-decode-kv-cache", "float16", "search"): 2,
                ("attention-decode-kv-cache", "bfloat16", "search"): 2,
                ("attention-decode-kv-cache", "float16", "none"): 2,
            },
        )
        self.assertEqual(
            sum(
                entry.package_count
                for entry in cases.SOURCE_NO_CARD_WORKLOADS
            ),
            18,
        )

        cmake = (
            pathlib.Path(__file__).resolve().parents[3]
            / "test"
            / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        registered_source_no_card = re.findall(
            r"add_test\(NAME "
            r"(wafer-runtime-pytorch-[^\s()]+-no-card)\s",
            cmake,
        )
        catalog_names = [
            entry.ctest_name for entry in cases.SOURCE_NO_CARD_WORKLOADS
        ]
        self.assertEqual(
            len(registered_source_no_card),
            len(set(registered_source_no_card)),
        )
        self.assertCountEqual(registered_source_no_card, catalog_names)
        for entry in cases.SOURCE_NO_CARD_WORKLOADS:
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
            self.assertIn(
                f"--optimization-policy {entry.optimization_policy}",
                body,
            )
            self.assertIn("--no-card", body)

        matched_policy_pairs = (
            (
                "wafer-runtime-pytorch-attention-decode-kv-cache-no-card",
                "wafer-runtime-pytorch-attention-decode-kv-cache-fp16-optimization-none-no-card",
                "attention-decode-kv-cache",
            ),
            (
                "wafer-runtime-pytorch-llama-2-7b-block-fp16-no-card",
                "wafer-runtime-pytorch-llama-2-7b-block-fp16-optimization-none-no-card",
                "llama-2-7b-block",
            ),
        )
        for search_name, none_name, case_name in matched_policy_pairs:
            bodies = []
            for test_name in (search_name, none_name):
                match = re.search(
                    rf"add_test\(NAME {re.escape(test_name)}\n"
                    r"(?P<body>.*?)\n\s*\)",
                    cmake,
                    re.DOTALL,
                )
                self.assertIsNotNone(match, test_name)
                bodies.append(match.group("body"))
            search_body, none_body = bodies
            for body in bodies:
                self.assertIn("Board/PyTorch/wafer_board_pytorch_test.py", body)
                self.assertIn(f"--case {case_name}", body)
                self.assertIn("--dtype float16", body)
                self.assertIn("--seed 20260803", body)
                self.assertIn("--no-card", body)
            self.assertIn("--optimization-policy search", search_body)
            self.assertIn("--optimization-policy none", none_body)

        runner_source = inspect.getsource(board_runner.main)
        self.assertNotIn("torch_eager_reference=deferred", runner_source)
        prepare_source = inspect.getsource(board_runner.prepare_case_step)
        self.assertIn("case.materialize_expected_outputs()", prepare_source)
        self.assertLess(
            runner_source.index(") = prepare_case_step("),
            runner_source.index("if args.no_card:"),
        )

    def test_matched_board_policy_matrix_is_independent_and_serialized(
        self,
    ) -> None:
        cmake = (
            pathlib.Path(__file__).resolve().parents[3]
            / "test"
            / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        matched_tests = {
            "wafer-board-pytorch-attention-decode-kv-cache": (
                "attention-decode-kv-cache",
                "search",
            ),
            "wafer-board-pytorch-attention-decode-kv-cache-optimization-none": (
                "attention-decode-kv-cache",
                "none",
            ),
            "wafer-board-pytorch-llama-2-7b-block": (
                "llama-2-7b-block",
                "search",
            ),
            "wafer-board-pytorch-llama-2-7b-block-optimization-none": (
                "llama-2-7b-block",
                "none",
            ),
        }
        work_dirs = set()
        for test_name, (case_name, policy) in matched_tests.items():
            command_match = re.search(
                rf"add_test\(NAME {re.escape(test_name)}\n"
                r"(?P<body>.*?)\n\s*\)",
                cmake,
                re.DOTALL,
            )
            self.assertIsNotNone(command_match, test_name)
            command = command_match.group("body")
            self.assertIn("Board/PyTorch/wafer_board_pytorch_test.py", command)
            self.assertIn(f"--case {case_name}", command)
            self.assertIn("--dtype float16", command)
            self.assertIn("--seed 20260803", command)
            self.assertIn(f"--optimization-policy {policy}", command)
            self.assertIn("--repeat 3", command)
            self.assertNotIn("--no-card", command)
            work_dir_match = re.search(
                r"--work-dir\s+\$\{CMAKE_CURRENT_BINARY_DIR\}/(?P<path>\S+)",
                command,
            )
            self.assertIsNotNone(work_dir_match, test_name)
            work_dirs.add(work_dir_match.group("path"))

            properties_match = re.search(
                rf"set_tests_properties\(\s*{re.escape(test_name)} PROPERTIES"
                r"(?P<body>.*?)\n\s*\)",
                cmake,
                re.DOTALL,
            )
            self.assertIsNotNone(properties_match, test_name)
            properties = properties_match.group("body")
            self.assertIn("matched-ab", properties)
            self.assertIn(f"optimization-{policy}", properties)
            self.assertIn(
                'RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"',
                properties,
            )
        self.assertEqual(len(work_dirs), len(matched_tests))

        baseline_name = (
            "wafer-board-pytorch-attention-prefill-optimization-none"
        )
        baseline_match = re.search(
            rf"add_test\(NAME {re.escape(baseline_name)}\n"
            r"(?P<body>.*?)\n\s*\)",
            cmake,
            re.DOTALL,
        )
        self.assertIsNotNone(baseline_match, baseline_name)
        baseline_command = baseline_match.group("body")
        self.assertIn("--case attention-prefill", baseline_command)
        self.assertIn("--dtype float16", baseline_command)
        self.assertIn("--seed 20260803", baseline_command)
        self.assertIn("--optimization-policy none", baseline_command)
        self.assertIn("--repeat 3", baseline_command)
        self.assertNotIn("--no-card", baseline_command)
        baseline_properties_match = re.search(
            rf"set_tests_properties\(\s*{re.escape(baseline_name)} PROPERTIES"
            r"(?P<body>.*?)\n\s*\)",
            cmake,
            re.DOTALL,
        )
        self.assertIsNotNone(baseline_properties_match, baseline_name)
        baseline_properties = baseline_properties_match.group("body")
        self.assertIn("baseline", baseline_properties)
        self.assertIn("optimization-none", baseline_properties)
        self.assertNotIn("matched-ab", baseline_properties)
        self.assertIn(
            'RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"',
            baseline_properties,
        )

    def test_heterogeneous_tiling_dataflow_has_mixed_output_domains(self) -> None:
        case = cases._heterogeneous_tiling_single_card(torch.float16, seed=19)
        self.assertEqual(case.num_partitions, 1)
        matrix, row_reduction = case.materialize_expected_outputs()
        self.assertEqual(matrix.shape, (96, 80))
        self.assertEqual(row_reduction.shape, (96,))
        self.assertEqual(matrix.dtype, torch.float16)
        self.assertTrue(torch.isfinite(matrix).all())
        self.assertTrue(torch.isfinite(row_reduction).all())

        bf16 = cases._heterogeneous_tiling_single_card(
            torch.bfloat16, seed=19
        )
        bf16_matrix, bf16_row_reduction = bf16.materialize_expected_outputs()
        self.assertEqual(bf16_matrix.dtype, torch.bfloat16)
        self.assertEqual(bf16_row_reduction.dtype, torch.bfloat16)

    def test_heterogeneous_tiling_export_is_source_derived(self) -> None:
        case = cases._heterogeneous_tiling_single_card(torch.float16, seed=23)
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

    def test_hf_prefill_preserves_the_official_additive_mask(self) -> None:
        from transformers.masking_utils import create_causal_mask

        observed_masks = []

        def record_official_mask(*args: object, **kwargs: object) -> object:
            mask = create_causal_mask(*args, **kwargs)
            observed_masks.append(mask)
            return mask

        with mock.patch(
            "transformers.masking_utils.create_causal_mask",
            side_effect=record_official_mask,
        ) as create_mask_mock:
            case = cases._attention_prefill(torch.float16, seed=37)

        self.assertEqual(create_mask_mock.call_count, 1)
        mask = case.inputs[-1]
        self.assertIs(mask, observed_masks[0])
        self.assertEqual(mask.shape, (1, 1, 1024, 1024))
        call = create_mask_mock.call_args
        expected = create_causal_mask(*call.args, **call.kwargs)
        self.assertTrue(torch.equal(mask, expected))

    def test_hf_llama_block_wraps_the_official_decoder_layer(self) -> None:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        config_dict = config.to_dict()
        config_dict["torch_dtype"] = "float16"
        module = cases.capture._make_hf_llama_decoder_block_module(
            torch, config_dict, sequence_length=4
        )
        self.assertIsInstance(module.decoder_layer, LlamaDecoderLayer)

    def test_hf_functional_decode_threads_exact_updated_cache(self) -> None:
        case = cases._attention_decode_kv_cache(torch.float16, seed=41)
        self.assertEqual(case.name, "attention-decode-kv-cache")
        self.assertEqual(case.num_partitions, 1)
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
        self.assertEqual(updated_key.ndim, 4)
        self.assertEqual(updated_value.ndim, 4)
        self.assertEqual(updated_key.shape, (1, 32, 1024, 128))
        self.assertEqual(updated_value.shape, (1, 32, 1024, 128))
        self.assertEqual(continuation.inputs[1].ndim, 4)
        self.assertEqual(continuation.inputs[2].ndim, 4)
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

    def test_hf_functional_decode_uses_official_causal_mask_path(self) -> None:
        from transformers.masking_utils import create_causal_mask

        observed_masks = []

        def record_official_mask(*args: object, **kwargs: object) -> object:
            mask = create_causal_mask(*args, **kwargs)
            observed_masks.append(mask)
            return mask

        with mock.patch(
            "transformers.masking_utils.create_causal_mask",
            side_effect=record_official_mask,
        ) as create_mask_mock:
            case = cases._attention_decode_kv_cache(torch.float16, seed=43)
            outputs = case.materialize_expected_outputs()
            continuation = case.continuation_factory(outputs)

        self.assertEqual(create_mask_mock.call_count, 2)
        self.assertIs(case.inputs[-1], observed_masks[0])
        self.assertIs(continuation.inputs[-1], observed_masks[1])
        for ordinal, call in enumerate(
            create_mask_mock.call_args_list,
            start=1023,
        ):
            self.assertIsNone(call.kwargs["attention_mask"])
            self.assertEqual(
                call.kwargs["position_ids"].tolist(),
                [[ordinal]],
            )
            self.assertEqual(
                call.kwargs["past_key_values"].get_seq_length(),
                ordinal,
            )
            self.assertEqual(call.kwargs["inputs_embeds"].shape, (1, 1, 4096))
            self.assertEqual(observed_masks[ordinal - 1023].ndim, 4)

    def test_torch_random_seed_and_attention_policy_reject_fault(self) -> None:
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
                policy=cases.ATTENTION_COMPARISON,
                context="attention fault injection",
            )


if __name__ == "__main__":
    unittest.main()
