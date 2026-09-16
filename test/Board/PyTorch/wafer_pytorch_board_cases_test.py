#!/usr/bin/env python3
"""Contract tests for framework-owned PyTorch board cases."""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

import wafer_board_pytorch_test as board_runner
import wafer_pytorch_board_cases as cases


def read_portable_stablehlo(program: pathlib.Path) -> str:
    translator = os.environ.get("WAFER_STABLEHLO_TRANSLATE")
    if not translator:
        raise RuntimeError("WAFER_STABLEHLO_TRANSLATE is not configured")
    with tempfile.TemporaryDirectory() as directory:
        text = pathlib.Path(directory) / "program.mlir"
        subprocess.run(
            [
                translator,
                "--deserialize",
                str(program / "functions" / "forward.stablehlo.bc"),
                "-o",
                str(text),
            ],
            check=True,
        )
        return text.read_text()


class PyTorchBoardCasesTest(unittest.TestCase):
    def test_mixed_ports_export_and_payload_preserve_runtime_integer_indices(self):
        class Lookup(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                weight = torch.arange(1024 * 8).reshape(1024, 8).float() / 2048
                self.embedding = torch.nn.Embedding.from_pretrained(weight.to(dtype))

            def forward(self, data, indices):
                return self.embedding(indices).float() + data.float(), indices.clone()

        for extent in (1024, 1025, 1031):
            for dtype, index_dtype in ((torch.float16, torch.int64),
                                       (torch.bfloat16, torch.int32)):
                with self.subTest(extent=extent, dtype=dtype):
                    module = Lookup(dtype).eval()
                    indices = (torch.arange(2 * extent) % 1024).to(index_dtype).reshape(1, 2, extent)
                    inputs = (torch.ones((1, 2, extent, 8), dtype=dtype), indices)
                    case = cases.PyTorchBoardCase(
                        name="mixed-ports", num_partitions=1, dtype=dtype, inputs=inputs,
                        expected_outputs_factory=lambda: module(*inputs),
                        export_program=lambda path: cases._save_exported_program(path, module, inputs),
                        comparison_policy=cases.common.PYTORCH_DEFAULT,
                    )
                    with mock.patch.dict(cases.CASE_FACTORIES, {"mixed-ports": lambda _dtype, _seed: case}):
                        self.assertIs(cases.make_case("mixed-ports", dtype=dtype, seed=0), case)
                    expected = case.materialize_expected_outputs()
                    self.assertEqual(tuple(t.dtype for t in expected), (torch.float32, index_dtype))
                    torch.testing.assert_close(expected[1], indices, rtol=0, atol=0)
                    with tempfile.TemporaryDirectory() as directory:
                        root = pathlib.Path(directory)
                        source = root / "source"
                        case.export_program(source)
                        metadata = json.loads((source / "functions/forward.meta").read_text())
                        runtime_signatures = {
                            location["position"]: signature
                            for signature, location in zip(metadata["input_signature"], metadata["input_locations"])
                            if location["type_"] == "input_arg"
                        }
                        for index, tensor in enumerate(inputs):
                            self.assertEqual(runtime_signatures[index]["dtype"], str(tensor.dtype).removeprefix("torch."))
                            self.assertEqual(runtime_signatures[index]["shape"], list(tensor.shape))
                        for signature, tensor in zip(metadata["output_signature"], expected):
                            self.assertEqual(signature["dtype"], str(tensor.dtype).removeprefix("torch."))
                            self.assertEqual(signature["shape"], list(tensor.shape))
                        self.assertEqual(len(metadata["output_signature"]), len(expected))
                        self.assertIn("stablehlo.gather", read_portable_stablehlo(source))

                        # Exercise the existing manifest/file boundary. This is
                        # a port fixture, not an ExecutablePackage qualification.
                        dtype_names = {value: key for key, value in cases.common.MANIFEST_DTYPES.items()}
                        manifest = {
                            "card_count": 1, "tile_count": board_runner.PHYSICAL_TILE_COUNT,
                            "target": {"identity": board_runner.TARGET_IDENTITY},
                            "entries": [{"card_id": 0, "tile_id": tile, "launch_slot": tile,
                                         "completion": "return_after_local_drain"}
                                        for tile in range(board_runner.PHYSICAL_TILE_COUNT)],
                        }
                        for table, tensors in (("inputs", inputs), ("outputs", expected)):
                            manifest[table] = [
                                {"id": index + (10 if table == "outputs" else 0), "role_index": index,
                                 "dtype": dtype_names[tensor.dtype], "shape": list(tensor.shape),
                                 "bytes": cases.common.tensor_nbytes(tensor)}
                                for index, tensor in enumerate(tensors)
                            ]
                        package = root / "ports"
                        package.mkdir()
                        (package / "manifest.json").write_text(json.dumps(manifest))
                        args, captures, output_ids, _ = board_runner.prepare_runtime_payloads(
                            root, source, package, case, expected,
                        )
                        self.assertEqual(output_ids, {10, 11})
                        for index, tensor in enumerate(inputs):
                            path = next(pathlib.Path(value.split("=", 1)[1]) for value in args
                                        if value.startswith(f"{index}="))
                            cases.common.assert_raw_capture_matches(
                                path, tensor, context="mixed input payload", policy=cases.common.EXACT,
                            )
                        self.assertEqual(len(captures), len(expected))
                        for index, tensor in enumerate(expected):
                            path = root / "raw" / f"card_00_output_{index}.expected.{dtype_names[tensor.dtype]}.raw"
                            cases.common.assert_raw_capture_matches(
                                path, tensor, context="mixed reference payload", policy=cases.common.EXACT,
                            )
                        for field, bad in (("dtype", "f16"), ("shape", [1, 2, extent - 1]),
                                           ("bytes", 0), ("role_index", 4)):
                            corrupted = json.loads(json.dumps(manifest))
                            corrupted["inputs"][1][field] = bad
                            (package / "manifest.json").write_text(json.dumps(corrupted))
                            work = root / field
                            work.mkdir()
                            with self.assertRaises(RuntimeError):
                                board_runner.prepare_runtime_payloads(work, source, package, case, expected)

    def test_case_rejects_unsupported_port_dtype_and_non_cpu_tensor(self):
        for invalid in (torch.zeros((1, 1024, 1), dtype=torch.float64),
                        torch.empty((1, 1024, 1), dtype=torch.float16, device="meta")):
            for role in ("input", "output"):
                with self.subTest(role=role, dtype=invalid.dtype, device=invalid.device):
                    valid = torch.zeros((1, 1024, 1), dtype=torch.float16)
                    case = cases.PyTorchBoardCase(
                        name="invalid-port", num_partitions=1, dtype=torch.float16,
                        inputs=(invalid if role == "input" else valid,),
                        expected_outputs_factory=lambda: (invalid if role == "output" else valid,),
                        export_program=lambda _path: None,
                        comparison_policy=cases.common.PYTORCH_DEFAULT,
                    )
                    with mock.patch.dict(cases.CASE_FACTORIES, {"invalid-port": lambda _dtype, _seed: case}):
                        with self.assertRaisesRegex(RuntimeError, f"{role} 0 requires"):
                            case = cases.make_case("invalid-port", dtype=torch.float16, seed=0)
                            case.materialize_expected_outputs()

    def test_single_layer_lm_keeps_embedding_decoder_norm_and_all_logits(self):
        import transformers

        constructor = transformers.LlamaForCausalLM
        initialize = constructor.__init__
        # S16 is the specified primary LM configuration. Large/tail exports
        # have separate registered source/no-card cases; this checks the oracle.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                models = []

                def construct(model, config):
                    initialize(model, config)
                    models.append(model)

                with mock.patch.object(constructor, "__init__", autospec=True, side_effect=construct):
                    case = cases.make_case("llama-2-7b-single-layer-lm", dtype=dtype, seed=20260803)
                original, = models
                self.assertEqual(len(original.model.layers), 1)
                self.assertEqual(original.model.embed_tokens.weight.shape, (32000, 4096))
                self.assertEqual(original.lm_head.weight.shape, (32000, 4096))
                self.assertIsNot(original.model.embed_tokens.weight, original.lm_head.weight)
                self.assertEqual(original.config.intermediate_size, 11008)
                self.assertEqual(original.config.num_attention_heads, 32)
                self.assertEqual(original.config.max_position_embeddings, 4096)
                self.assertEqual(case.inputs[0].dtype, torch.int64)
                self.assertEqual(case.inputs[0][0, :3].tolist(), [0, 31999, 0])
                expected, = case.materialize_expected_outputs()
                with torch.no_grad():
                    actual = original(input_ids=case.inputs[0], use_cache=False, logits_to_keep=0).logits
                self.assertEqual(actual.shape, (1, 16, 32000))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    destination = pathlib.Path(directory) / "source"
                    with mock.patch("torch.export.export", side_effect=AssertionError("Dynamo must not run")):
                        case.export_program(destination)
                    subprocess.run(["wafer-verify-program", "--program-dir", str(destination)], check=True)
                    metadata = json.loads((destination / "functions/forward.meta").read_text())
                    ports = [(location, signature) for location, signature in
                             zip(metadata["input_locations"], metadata["input_signature"])]
                    runtime = [(location["position"], signature["shape"], signature["dtype"])
                               for location, signature in ports if location["type_"] == "input_arg"]
                    self.assertEqual(runtime, [(0, [1, 16], "int64")])
                    self.assertEqual(sum(location["type_"] == "parameter" and signature["shape"] == [32000, 4096]
                                         for location, signature in ports), 2)
                    self.assertEqual(metadata["output_signature"], [{"shape": [1, 16, 32000],
                                      "dtype": str(dtype).removeprefix("torch."), "dynamic_dims": []}])
                    self.assertIn("stablehlo.gather", read_portable_stablehlo(destination))
                case.inputs[0][0, -1] = (case.inputs[0][0, -1] + 1) % 32000
                changed, = case.materialize_expected_outputs()
                self.assertFalse(torch.equal(changed[:, -1], expected[:, -1]))
                torch.testing.assert_close(changed[:, :-1], expected[:, :-1], rtol=0, atol=0)
                for invalid in (-1, 32000):
                    case.inputs[0][0, -1] = invalid
                    with self.assertRaisesRegex(ValueError, "within the vocabulary"):
                        case.materialize_expected_outputs()
                    with tempfile.TemporaryDirectory() as directory:
                        destination = pathlib.Path(directory) / "source"
                        with self.assertRaisesRegex(ValueError, "within the vocabulary"):
                            case.export_program(destination)
                        self.assertFalse(destination.exists())

    def test_gemm_wide_partial_oracle_matches_full_pytorch(self):
        import wafer_instruction_family_catalog as catalog

        for dtype_name, dtype in (("F16", torch.float16), ("BF16", torch.bfloat16)):
            case = catalog.CASES_BY_NAME[f"ne-gemm-{dtype_name.lower()}-wide-partial"]
            lhs, rhs = catalog.gemm_wide_partial_inputs(dtype_name)
            a = torch.tensor([lhs], dtype=dtype)
            b = torch.tensor(rhs, dtype=dtype)
            expected = a @ b
            _, _, raw = catalog._gemm(case)
            reference = torch.frombuffer(bytearray(raw), dtype=dtype).reshape_as(expected)
            torch.testing.assert_close(reference, expected, rtol=0, atol=0)
            narrow = torch.zeros_like(expected)
            wide = torch.zeros_like(expected, dtype=torch.float32)
            for begin, end in ((0, 16), (16, 32), (32, 33)):
                narrow += a[:, begin:end] @ b[begin:end]
                wide += a[:, begin:end].float() @ b[begin:end].float()
            self.assertTrue(torch.any(narrow != expected))
            torch.testing.assert_close(wide.to(dtype), expected, rtol=0, atol=0)

    def test_profile_keeps_device_watchdog_without_timing_out_host_report(self):
        outputs = (torch.zeros((1, 1, 1024), dtype=torch.float16),)
        case = cases.PyTorchBoardCase(
            name="launch-reference", num_partitions=1, dtype=torch.float16,
            inputs=outputs, expected_outputs_factory=lambda: outputs,
            export_program=lambda _path: None,
            comparison_policy=cases.ATTENTION_COMPARISON,
        )
        for profile in (False, True):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                argv = [
                    "runner", "--case", "attention-prefill", "--wafer-compile", "compile",
                    "--wafer-run", "run", "--work-dir", str(root / "work"),
                    "--expected-runtime-version", "1300", "--expected-device-name", "device",
                    "--expected-pci-bus-id", "bus", "--expected-tile-count", "16",
                    "--expected-runtime-library-sha256", "digest", "--completion-timeout-ms", "7000",
                ] + (["--profile"] if profile else [])
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.dict(os.environ, {"WAFER_EXECUTE_HARDWARE_TESTS": "1"}),
                    mock.patch.object(cases, "make_case", return_value=case),
                    mock.patch.object(board_runner, "prepare_case_step", return_value=(
                        outputs, [], {}, set(), {},
                    )),
                    mock.patch.object(board_runner, "run", return_value=types.SimpleNamespace(
                        stdout="", stderr="",
                    )) as run,
                    mock.patch.object(board_runner, "verify_no_card"),
                    mock.patch.object(board_runner, "verify_board"),
                    mock.patch.object(board_runner, "read_single_card_continuation_outputs",
                                      return_value=outputs),
                    mock.patch.object(board_runner, "file_sha256", return_value="digest"),
                ):
                    self.assertEqual(board_runner.main(), 0)
                self.assertEqual(run.call_count, 2)
                no_card, launch = run.call_args_list
                self.assertIn("--no-card", no_card.args[0])
                command = launch.args[0]
                self.assertIn("--board", command)
                self.assertEqual(command[command.index("--completion-timeout-ms") + 1], "7000")
                self.assertEqual(launch.kwargs["timeout_seconds"], None if profile else 67)

    def test_compact_contribution_assembly_preserves_sources_and_global_window(self):
        # The full input has extent 1025; this is its real 64-element Tile 1 tail shard.
        lines = [
            "wafer.tile.region(%input : memref<16x1025x1xf16>) -> () {",
            "%local = memref.subview %input[1, 65, 0] [1, 64, 1] [1, 1, 1]",
        ]
        for peer in range(16):
            source = "%local" if peer == 1 else f"%recv_{peer}"
            if peer != 1:
                lines.append(f"wafer.tile.peer_recv {source} {{peer = {peer} : i64}}")
            lines.extend([
                f"%slot_{peer} = memref.subview %assembly[{peer}, 0, 0] [1, 64, 1] [1, 1, 1]",
                f"%layout_{peer} = wafer.tile.materialize_layout {source} : memref<1x64x1xf16>",
                f"wafer.tile.copy_into %layout_{peer} into %slot_{peer} : memref<1x64x1xf16>",
            ])
        lines.extend([
            "%sum = wafer.tile.reduce <sum> %assembly, %zero {dimensions = array<i64: 0>} : (memref<16x64x1xf16, #wafer.memory<spm, ncx>>, f16)",
            "}",
        ])
        text = "\n".join(lines)
        for shard in (None, (65, 64)):
            self.assertEqual(
                board_runner.verify_reduce_scatter_contributions(text, 1, 1025, shard),
                (65, 64),
            )
        faults = (
            text.replace("peer = 0 : i64", "peer = 2 : i64"),
            text.replace("[1, 65, 0]", "[1, 64, 0]"),
            text.replace("wafer.tile.copy_into %layout_0 into %slot_0", "removed %layout_0 %slot_0"),
            text.replace("%slot_2 : memref", "%slot_0 : memref"),
        )
        for corrupted in faults:
            with self.assertRaises(RuntimeError):
                board_runner.verify_reduce_scatter_contributions(corrupted, 1, 1025, (65, 64))

    def test_prepared_source_checks_graph_parameters_and_file_set(self):
        # File-boundary negative tests; production-size no-card covers execution.
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            prepared, fresh = root / "prepared", root / "fresh"
            for path in (prepared, fresh):
                (path / "functions").mkdir(parents=True)
                (path / "data").mkdir()
                (path / "functions/forward.stablehlo.bc").write_bytes(b"graph")
                (path / "data/weight").write_bytes(b"weight")
            board_runner.verify_prepared_source(prepared, fresh)
            for name in ("functions/forward.stablehlo.bc", "data/weight"):
                path = fresh / name
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "prepared source differs"):
                    board_runner.verify_prepared_source(prepared, fresh)
                path.write_bytes(original)
            (fresh / "data/weight").unlink()
            with self.assertRaisesRegex(RuntimeError, "prepared source differs"):
                board_runner.verify_prepared_source(prepared, fresh)

    def test_prepared_output_cannot_remove_or_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            sentinel = prepared / "capture"
            sentinel.write_bytes(b"keep")
            for output in (prepared, prepared / "child", root):
                with self.assertRaisesRegex(RuntimeError, "must not overlap"):
                    board_runner.validate_prepared_directories(prepared, output)
            output = root / "invocation"
            board_runner.validate_prepared_directories(prepared, output)
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "fresh output"):
                board_runner.validate_prepared_directories(prepared, output)
            self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_prepared_step_reuses_package_but_rebuilds_payload_and_reference(self):
        for profile in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                prepared, invocation = root / "prepared", root / "invocation"
                invocation.mkdir()
                package = prepared / ("package/package" if profile else "package")
                package.mkdir(parents=True)
                (package / "manifest.json").write_text("{}")
                def export(path):
                    path.mkdir()
                    (path / "graph").write_bytes(b"same-current-graph")
                export(prepared / "source-program")
                expected = (torch.zeros((1, 2, 1031), dtype=torch.float16),)
                case = types.SimpleNamespace(
                    export_program=export, gemm_dimensions=None,
                    materialize_expected_outputs=mock.Mock(return_value=expected),
                )
                args = types.SimpleNamespace(
                    qualify_communication=None, optimization_policy="search",
                    target_model=False,
                )
                with (
                    mock.patch.object(board_runner, "run") as compiler,
                    mock.patch.object(board_runner, "prepare_runtime_payloads",
                                      return_value=([], {}, set(), {})) as payload,
                ):
                    result = board_runner.prepare_case_step(
                        args, case, step_index=0, step_dir=invocation,
                        source=invocation / "source-program", package=package,
                        dump_compiler_ir=None, prepared_step=prepared,
                    )
                compiler.assert_not_called()
                case.materialize_expected_outputs.assert_called_once()
                self.assertIs(result[0], expected)
                payload.assert_called_once_with(
                    invocation, invocation / "source-program", package, case, expected
                )

    def test_precision_cases_keep_eager_operator_rounding_and_default_tolerance(self):
        for extent in (1024, 1025, 1031):
            conv = cases.make_biased_conv(torch.float16, 20260803, extent=extent)
            value, weight, bias = conv.inputs
            expected, = conv.materialize_expected_outputs()
            separately_rounded = torch.nn.functional.conv2d(value, weight, padding=1)
            separately_rounded = separately_rounded + bias[None, :, None, None]
            self.assertEqual(conv.comparison_policy, cases.common.PYTORCH_DEFAULT)
            with self.assertRaises(AssertionError):
                cases.common.assert_tensor_matches(
                    separately_rounded, expected, policy=conv.comparison_policy,
                    context="bias rounded after the entire convolution",
                )
            sigmoid = cases.make_sigmoid(torch.float16, 20260803, extent=extent)
            expected, = sigmoid.materialize_expected_outputs()
            self.assertEqual(expected.shape, (2, 4, extent))
            self.assertEqual(sigmoid.comparison_policy, cases.common.PYTORCH_DEFAULT)
            missing_tail = expected.clone()
            missing_tail.flatten()[-1] += 1
            with self.assertRaises(AssertionError):
                cases.common.assert_tensor_matches(
                    missing_tail, expected, policy=sigmoid.comparison_policy,
                    context="sigmoid full tail",
                )

    def test_local_conv_reference_detects_weight_axis_and_tail_errors(self) -> None:
        for extent, kernel in ((1024, (3, 3)), (1025, (2, 3)), (1031, (3, 2))):
            case = cases.make_local_conv(
                torch.float16, 20260803, extent=extent, kernel=kernel
            )
            value, weight = case.inputs
            expected, = case.materialize_expected_outputs()
            self.assertEqual(value.shape, (1, 16, 8, extent))
            self.assertEqual(
                expected.shape, (1, 24, 11 - kernel[0], extent + 3 - kernel[1])
            )
            wrong_weight = cases.LocalConv()(value, weight.flip(2))
            missing_tail = expected.clone()
            missing_tail.flatten()[-1] += 1
            for actual in (wrong_weight, missing_tail):
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        actual, expected, policy=case.comparison_policy,
                        context=f"LocalConv L={extent}",
                    )

    def test_local_reduce_reference_covers_both_axes_and_output_tail(self) -> None:
        for extent in (1024, 1025, 1031):
            case = cases.make_local_reduce(torch.float16, 20260803, extent=extent)
            value, = case.inputs
            width_sum, spatial_sum = case.materialize_expected_outputs()
            self.assertEqual(value.shape, (1, 24, 8, extent))
            self.assertEqual(width_sum.shape, (1, 24, 8))
            self.assertEqual(spatial_sum.shape, (1, 24))
            torch.testing.assert_close(
                width_sum.sum(dim=2), spatial_sum, rtol=0, atol=0
            )
            for expected in (width_sum, spatial_sum):
                missing_tail = expected.clone()
                missing_tail.flatten()[-1] += 1
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        missing_tail, expected, policy=case.comparison_policy,
                        context=f"LocalReduce L={extent}",
                    )

    def test_alltoall_reference_detects_wrong_source_and_output_tail(self) -> None:
        for extent in (1024, 1025, 1031):
            case = cases.make_alltoall_transpose(torch.float16, 20260803, extent=extent)
            lhs, rhs = case.inputs
            expected, = case.materialize_expected_outputs()
            self.assertEqual(lhs.shape, (extent, 16, 1))
            self.assertEqual(rhs.shape, (16, extent, 1))
            wrong_source = lhs + (rhs + rhs).roll(1, dims=0).transpose(0, 1)
            missing_tail = expected.clone()
            missing_tail[-1, -1, 0] += 1
            for actual in (wrong_source, missing_tail):
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        actual, expected, policy=case.comparison_policy,
                        context=f"AllToAll L={extent}",
                    )

    def test_reduce_scatter_reference_detects_missing_source_and_tail(self) -> None:
        for extent in (1024, 1025, 1031):
            case = cases.make_reduce_scatter_sum(torch.float16, 20260803, extent=extent)
            rhs, = case.inputs
            expected, = case.materialize_expected_outputs()
            self.assertEqual(expected.shape, (1, extent, 1))
            missing_source = (rhs[:-1] + rhs[:-1]).sum(dim=0, keepdim=True)
            wrong_tail = expected.clone()
            wrong_tail[0, -1, 0] += 1
            for actual in (missing_source, wrong_tail):
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        actual, expected, policy=case.comparison_policy,
                        context=f"ReduceScatter L={extent}",
                    )

    def test_all_reduce_reference_detects_contribution_broadcast_and_tail_errors(self) -> None:
        for extent in (1024, 1025, 1031):
            case = cases.make_all_reduce_sum(torch.float16, 20260803, extent=extent)
            lhs, rhs = case.inputs
            expected, = case.materialize_expected_outputs()
            self.assertEqual(expected.shape, (16, extent, 1))
            missing_source = lhs + (rhs[:-1] + rhs[:-1]).sum(dim=0, keepdim=True)
            duplicate = rhs.clone()
            duplicate[-1] = duplicate[0]
            duplicated_source = lhs + (duplicate + duplicate).sum(dim=0, keepdim=True)
            missing_destination = expected.clone()
            missing_destination[-1] = lhs[-1]
            wrong_tail = expected.clone()
            wrong_tail[-1, -1, 0] += 1
            for actual in (missing_source, duplicated_source, missing_destination, wrong_tail):
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        actual, expected, policy=case.comparison_policy,
                        context=f"AllReduce L={extent}",
                    )

    def test_gemm_reference_detects_missing_k_and_output_tails(self) -> None:
        for name in ("single-card-gemm", "single-card-gemm-tail-1025",
                     "single-card-gemm-tail-1031"):
            case = cases.make_case(name, dtype=torch.float16, seed=20260803)
            m, k, n = case.gemm_dimensions
            self.assertGreaterEqual(m, 1024)
            self.assertEqual(case.inputs[0].shape, (1, m, k))
            self.assertEqual(case.inputs[1].shape, (1, k, n))
            expected, = case.materialize_expected_outputs()
            self.assertEqual(expected.shape, (1, m, n))
            for fault in ("missing-k", "last-row", "last-column"):
                if fault == "missing-k":
                    actual = torch.matmul(
                        case.inputs[0][..., :-1], case.inputs[1][..., :-1, :]
                    )
                else:
                    actual = expected.clone()
                    if fault == "last-row":
                        actual[:, -1, :] = 0
                    else:
                        actual[:, :, -1] = 0
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        actual, expected, policy=case.comparison_policy,
                        context=f"{name} {fault}",
                    )

    def test_allgather_reference_detects_missing_shard_and_last_element(self) -> None:
        for extent in (1024, 1025, 1031):
            case = cases.make_allgather_add(torch.float16, 17, extent=extent)
            expected, = case.materialize_expected_outputs()
            self.assertEqual(expected.shape, (16, 16, 1, extent))
            self.assertEqual(case.allgather_payload_elements, extent)
            for fault in ("missing-shard", "last-element"):
                actual = expected.clone()
                if fault == "missing-shard":
                    actual[:, 15] = actual[:, 14]
                else:
                    actual[-1, -1, -1, -1] += 16
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        actual, expected, policy=case.comparison_policy,
                        context=f"AllGather {extent} {fault}",
                    )

    def test_launch_reference_checks_full_tensor_and_tail(self) -> None:
        for length in (1024, 1025, 1031):
            case = cases.make_launch_case("complete-tile-add", local_elements=length)
            expected, = case.materialize_expected_outputs()
            self.assertEqual(expected.shape, (16 * length,))
            self.assertEqual(expected.dtype, torch.float16)
            actual = expected.clone()
            actual[-1] += 16
            with self.assertRaises(AssertionError):
                cases.common.assert_tensor_matches(
                    actual, expected, policy=case.comparison_policy,
                    context=f"fault at the last element, local length {length}",
                )

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
                comparison_policy=cases.ATTENTION_COMPARISON,
            )

        step_one = cases.PyTorchBoardCase(
            name="decode-step-one",
            num_partitions=1,
            dtype=torch.float16,
            inputs=(torch.zeros((1,), dtype=torch.float16),),
            expected_outputs_factory=lambda: step_one_outputs,
            export_program=lambda _path: None,
            comparison_policy=cases.ATTENTION_COMPARISON,
            continuation_factory=make_step_two,
        )
        args = types.SimpleNamespace(
            case="attention-decode-kv-cache",
            dtype="float16",
            seed=17,
            prepared_work_dir=None, search_width=None, search_trials=None,
            compile_timeout_seconds=1800,
            wafer_compile=pathlib.Path("wafer-compile"),
            wafer_run=pathlib.Path("wafer-run"),
            work_dir=pathlib.Path("work"),
            dump_compiler_ir=None,
            compile_timing=False, profile=False, profile_trace_event_limit=None,
            optimization_policy="none",
            qualify_communication=None,
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
                    "wafer-direct-dte-status",
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
                "none",
            )
        with mock.patch.object(
            sys,
            "argv",
            [*required_args, "--optimization-policy", "search"],
        ):
            self.assertEqual(
                board_runner.parse_args().optimization_policy,
                "search",
            )
        self.assertEqual(
            cases.OPTIMIZATION_POLICIES,
            ("search", "none"),
        )

        case = types.SimpleNamespace(
            num_partitions=1,
            allgather_payload_elements=None,
            gemm_dimensions=None,
            alltoall_extent=None,
            reduce_scatter_extent=None, all_reduce_extent=None,
            export_program=mock.Mock(),
            materialize_expected_outputs=mock.Mock(
                return_value=(torch.zeros((1,), dtype=torch.float16),)
            ),
        )
        args = types.SimpleNamespace(
            prepared_work_dir=None, search_width=None, search_trials=None,
            target_model=False,
            compile_timeout_seconds=1800,
            wafer_compile=pathlib.Path("wafer-compile"),
            compile_timing=False, profile=False, profile_trace_event_limit=None,
            optimization_policy="none",
            qualify_communication=None,
        )
        with (
            mock.patch.object(
                board_runner,
                "run",
                return_value=types.SimpleNamespace(
                    stdout=(
                        "wafer-compile: wrote verified package with "
                        "num-partitions=1 tiles=16"
                    ),
                    stderr="",
                ),
            ) as run_mock,
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

    def test_product_policies_ignore_communication_expectations(self) -> None:
        case = types.SimpleNamespace(
            num_partitions=1, allgather_payload_elements=1024,
            alltoall_extent=1024, reduce_scatter_extent=1024, all_reduce_extent=1024,
            gemm_dimensions=None, export_program=mock.Mock(),
            materialize_expected_outputs=mock.Mock(return_value=(torch.zeros(1),)),
        )
        for policy in ("none", "search"):
            args = types.SimpleNamespace(
                prepared_work_dir=None,
                search_width=16 if policy == "search" else None,
                search_trials=126 if policy == "search" else None,
                compile_timeout_seconds=2400,
                wafer_compile=pathlib.Path("wafer-compile"), compile_timing=False,
                profile=False, profile_trace_event_limit=None,
                optimization_policy=policy, qualify_communication=None,
                target_model=False,
            )
            with (
                mock.patch.object(board_runner, "run", return_value=types.SimpleNamespace(
                    stdout="wrote verified package with num-partitions=1 tiles=16", stderr=""
                )) as command,
                mock.patch.object(board_runner, "prepare_runtime_payloads",
                                  return_value=([], {}, set(), {})),
                mock.patch.object(board_runner, "verify_ring_allgather") as gather,
                mock.patch.object(board_runner, "verify_personalized_exchange") as exchange,
                mock.patch.object(board_runner, "verify_all_reduce") as reduce,
            ):
                board_runner.prepare_case_step(
                    args, case, step_index=0, step_dir=pathlib.Path("step"),
                    source=pathlib.Path("source"), package=pathlib.Path("package"),
                    dump_compiler_ir=None,
                )
                gather.assert_not_called()
                exchange.assert_not_called()
                reduce.assert_not_called()
                compile_command = command.call_args.args[0]
                self.assertFalse(any("test-communication" in arg for arg in compile_command))
                self.assertEqual(command.call_args.kwargs["timeout_seconds"], 2400)
                for option, expected in (("--search-width", "16"), ("--search-trials", "126")):
                    if policy == "search":
                        self.assertEqual(compile_command[compile_command.index(option) + 1], expected)
                    else:
                        self.assertNotIn(option, compile_command)
            args.qualify_communication = "ring-allgather"
            args.optimization_policy = "search"
            with mock.patch.object(board_runner, "run") as command:
                with self.assertRaisesRegex(RuntimeError, "cannot run search"):
                    board_runner.prepare_case_step(
                        args, case, step_index=0, step_dir=pathlib.Path("step"),
                        source=pathlib.Path("source"), package=pathlib.Path("package"),
                        dump_compiler_ir=None,
                    )
                command.assert_not_called()

    def test_output_error_audit_preserves_valid_json_for_nonfinite_values(self) -> None:
        # Tiny scalar oracle for the diagnostic format, not numeric qualification.
        finite = torch.tensor([1.0, 2.0])
        audit = board_runner.summarize_output_error(finite + 0.25, finite)
        self.assertEqual(audit["max_abs_error"], 0.25)
        special = torch.tensor([float("nan"), float("inf")])
        audit = board_runner.summarize_output_error(special, special)
        self.assertIsNone(audit["max_abs_error"])
        json.dumps(audit, allow_nan=False)

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
            "target": {"identity": board_runner.TARGET_IDENTITY},
            "card_count": 1,
            "tile_count": board_runner.PHYSICAL_TILE_COUNT,
            "inputs": [
                {
                    "id": 7,
                    "role_index": 0,
                    "logical_dtype": "f16",
                    "logical_shape": [16],
                    "dtype": "f16",
                    "layout": "tensor",
                    "shape": [16],
                    "bytes": 32,
                    "alignment": 256,
                },
            ],
            "outputs": [
                {
                    "id": 8,
                    "role_index": 0,
                    "logical_dtype": "f16",
                    "logical_shape": [16],
                    "dtype": "f16",
                    "layout": "tensor",
                    "shape": [16],
                    "bytes": 32,
                    "alignment": 256,
                },
            ],
            "entries": entries,
        }
        with tempfile.TemporaryDirectory() as directory:
            package = pathlib.Path(directory)
            manifest_path = package / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            ports, output_ids = board_runner._manifest_ports(package)
            self.assertEqual(
                set(ports), {("user_input", 0), ("output", 0)}
            )
            self.assertEqual(output_ids, {8})

            entries[1]["tile_id"] = entries[0]["tile_id"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                board_runner._manifest_ports(package)

        self.assertEqual(
            board_runner.base_runtime_command(
                pathlib.Path("wafer-run"), pathlib.Path("package")
            ),
            ["wafer-run", "--package-dir", "package"],
        )

    def test_source_no_card_matrix_is_registered_for_both_policies(
        self,
    ) -> None:
        expected = {
            (case_name, dtype_name, 1, optimization_policy)
            for case_name in cases.PRODUCTION_SOURCE_CASES
            for dtype_name, _ in cases.PRODUCTION_SOURCE_DTYPES
            for optimization_policy in (
                cases.SOURCE_NO_CARD_OPTIMIZATION_POLICIES
            )
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
                (
                    "attention-decode-kv-cache",
                    dtype_name,
                    optimization_policy,
                ): 2
                for dtype_name, _ in cases.PRODUCTION_SOURCE_DTYPES
                for optimization_policy in (
                    cases.SOURCE_NO_CARD_OPTIMIZATION_POLICIES
                )
            },
        )
        self.assertEqual(
            sum(
                entry.package_count
                for entry in cases.SOURCE_NO_CARD_WORKLOADS
            ),
            20,
        )

        cmake = (
            pathlib.Path(__file__).resolve().parents[3]
            / "test"
            / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("wafer_add_pytorch_source_no_card_test", cmake)
        for case_name in cases.PRODUCTION_SOURCE_CASES:
            self.assertIn(case_name, cmake)
        for dtype_name, dtype_label in cases.PRODUCTION_SOURCE_DTYPES:
            self.assertIn(dtype_name, cmake)
            self.assertIn(dtype_label, cmake)
        for optimization_policy in cases.SOURCE_NO_CARD_OPTIMIZATION_POLICIES:
            self.assertIn(optimization_policy, cmake)

        runner_source = inspect.getsource(board_runner.main)
        self.assertNotIn("torch_eager_reference=deferred", runner_source)
        prepare_source = inspect.getsource(board_runner.prepare_case_step)
        self.assertIn("case.materialize_expected_outputs()", prepare_source)
        self.assertLess(
            runner_source.index(") = prepare_case_step("),
            runner_source.index("if args.no_card:"),
        )

    def test_board_workloads_use_the_baseline_policy(self) -> None:
        cmake = (
            pathlib.Path(__file__).resolve().parents[3]
            / "test"
            / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        baseline_tests = {
            "wafer-board-pytorch-attention-prefill-optimization-none":
                "attention-prefill",
            "wafer-board-pytorch-attention-decode-kv-cache-optimization-none":
                "attention-decode-kv-cache",
            "wafer-board-pytorch-llama-2-7b-block-optimization-none":
                "llama-2-7b-block",
        }
        for test_name, case_name in baseline_tests.items():
            command_match = re.search(
                rf"add_test\(NAME {re.escape(test_name)}\n"
                r"(?P<body>.*?)\n\s*\)",
                cmake,
                re.DOTALL,
            )
            self.assertIsNotNone(command_match, test_name)
            command = command_match.group("body")
            self.assertIn(f"--case {case_name}", command)
            self.assertIn("--dtype float16", command)
            self.assertIn("--optimization-policy none", command)
            self.assertIn("--repeat 3", command)
            self.assertNotIn("--no-card", command)

            properties_match = re.search(
                rf"set_tests_properties\(\s*{re.escape(test_name)} PROPERTIES"
                r"(?P<body>.*?)\n\s*\)",
                cmake,
                re.DOTALL,
            )
            self.assertIsNotNone(properties_match, test_name)
            properties = properties_match.group("body")
            self.assertIn("baseline", properties)
            self.assertIn("optimization-none", properties)
            self.assertNotIn("matched-ab", properties)
            self.assertIn(
                'RESOURCE_LOCK "wafer-board-${WAFER_BOARD_TEST_DEVICE_ID}"',
                properties,
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
            stablehlo = read_portable_stablehlo(program)
        self.assertIn("stablehlo.dot_general", stablehlo)
        self.assertIn("stablehlo.maximum", stablehlo)
        self.assertIn("stablehlo.reduce", stablehlo)
        self.assertNotIn("wafer", stablehlo.lower())

    def test_conv_mixed_dag_pairs_aligned_and_ragged_real_scale_widths(
        self,
    ) -> None:
        fp16 = cases._conv_mixed_dag(torch.float16, seed=29)
        bf16 = cases._conv_mixed_dag(torch.bfloat16, seed=29)
        self.assertEqual(fp16.inputs[0].shape, (1, 16, 8, 1024))
        self.assertEqual(bf16.inputs[0].shape, (1, 16, 8, 1025))
        self.assertEqual(fp16.inputs[0].dtype, torch.float16)
        self.assertEqual(bf16.inputs[0].dtype, torch.bfloat16)

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

        for extent in (1024, 1025, 1031):
            with self.subTest(extent=extent):
                observed_masks = []

                def record_official_mask(*args: object, **kwargs: object) -> object:
                    mask = create_causal_mask(*args, **kwargs)
                    observed_masks.append(mask)
                    return mask

                with mock.patch(
                    "transformers.masking_utils.create_causal_mask",
                    side_effect=record_official_mask,
                ) as create_mask_mock:
                    case = cases._attention_prefill(
                        torch.float16, seed=20260803, extent=extent
                    )

                self.assertEqual(create_mask_mock.call_count, 1)
                mask = case.inputs[-1]
                self.assertIs(mask, observed_masks[0])
                self.assertEqual(mask.shape, (1, 1, extent, extent))
                call = create_mask_mock.call_args
                expected_mask = create_causal_mask(*call.args, **call.kwargs)
                self.assertTrue(torch.equal(mask, expected_mask))
                expected, = case.materialize_expected_outputs()
                self.assertEqual(expected.shape, (1, 1, extent, 64))
                self.assertTrue(torch.isfinite(expected).all())
                missing_tail = expected.clone()
                missing_tail.flatten()[-1] += 1
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        missing_tail, expected, policy=case.comparison_policy,
                        context=f"causal prefill L={extent} full tail",
                    )

    def test_llama_prefill_uses_full_context_and_causal_multihead_reference(self) -> None:
        case = cases.make_case(
            "attention-prefill-llama-2-7b", dtype=torch.float16, seed=20260803
        )
        query, key, value, mask = case.inputs
        for tensor in (query, key, value):
            self.assertEqual(tensor.shape, (1, 32, 4096, 128))
        self.assertEqual(mask.shape, (1, 1, 4096, 4096))
        self.assertEqual(mask[0, 0, 0, 0].item(), 0)
        self.assertLess(mask[0, 0, 0, -1].item(), -10000)
        self.assertEqual(mask[0, 0, -1, 0].item(), 0)
        # Full-context eager reference runs in the registered product no-card
        # case. Exercise multihead causality here without duplicating its cost.
        case = cases._read_only_attention(
            torch.float16, 20260803, name="multihead-causal-reference",
            query_length=1031, key_value_length=1031, causal=True,
            num_heads=2, head_dim=128,
        )
        expected, = case.materialize_expected_outputs()
        self.assertEqual(expected.shape, (1, 2, 1031, 128))
        # Every head's first causal query must observe only its own first V.
        torch.testing.assert_close(expected[:, :, 0], case.inputs[2][:, :, 0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(expected).all())

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

    def test_gqa_preserves_runtime_kv_heads_and_official_group_mapping(self) -> None:
        for extent in (1024, 1025):
            with self.subTest(extent=extent):
                name = "attention-gqa" if extent == 1024 else "attention-gqa-tail-1025"
                case = cases.make_case(name, dtype=torch.float16, seed=20260803)
                query, key, value, mask = case.inputs
                self.assertEqual(query.shape, (1, 32, extent, 128))
                self.assertEqual(key.shape, (1, 8, extent, 128))
                self.assertEqual(value.shape, key.shape)
                self.assertEqual(mask.shape, (1, 1, extent, extent))
                expected, = case.materialize_expected_outputs()
                self.assertEqual(expected.shape, query.shape)
                self.assertTrue(torch.isfinite(expected).all())
                # Only token zero is visible to the first causal query. Each
                # contiguous group of four Q heads must read its one KV head.
                for head in range(32):
                    torch.testing.assert_close(
                        expected[:, head, 0], value[:, head // 4, 0],
                        rtol=0, atol=0,
                    )
                corrupted = expected.clone()
                corrupted[0, -1, -1, -1] += 1
                with self.assertRaises(AssertionError):
                    cases.common.assert_tensor_matches(
                        corrupted, expected, policy=case.comparison_policy,
                        context=f"GQA S={extent} final head/token",
                    )
                with tempfile.TemporaryDirectory() as directory:
                    program = pathlib.Path(directory) / "gqa-program"
                    case.export_program(program)
                    stablehlo = read_portable_stablehlo(program)
                signatures = [
                    line for line in stablehlo.splitlines()
                    if re.match(r"\s*func\.func\b", line) and " private " not in line
                ]
                self.assertEqual(len(signatures), 1)
                signature = signatures[0]
                self.assertEqual(signature.count(f"tensor<1x8x{extent}x128xf16>"), 2)
                self.assertIn(f"tensor<1x32x{extent}x128xf16>", signature)
                self.assertIn("stablehlo.broadcast_in_dim", stablehlo)
                self.assertIn("stablehlo.dot_general", stablehlo)

    def test_attention_rejects_invalid_q_kv_head_groups(self) -> None:
        for heads, kv_heads in ((0, 8), (32, 0), (32, 7)):
            with self.subTest(heads=heads, kv_heads=kv_heads):
                with self.assertRaisesRegex(ValueError, "Q divisible by KV"):
                    cases._read_only_attention(
                        torch.float16, 20260803, name="invalid-heads",
                        query_length=1024, key_value_length=1024, causal=True,
                        num_heads=heads, num_key_value_heads=kv_heads,
                    )

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
        self.assertIsNotNone(case.validate_actual_outputs)
        case.validate_actual_outputs((attention, updated_key, updated_value))
        for index in (1, 2):
            corrupted = [attention, updated_key, updated_value]
            corrupted[index] = corrupted[index].clone()
            # A single stored-bit change must fail even within numeric tolerance.
            bits = corrupted[index].view(torch.int16)
            bits[0, 0, 0, 0] ^= 1
            with self.assertRaisesRegex(RuntimeError, "existing KV prefix"):
                case.validate_actual_outputs(tuple(corrupted))
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
            step_one_stablehlo = read_portable_stablehlo(step_one_program)
            step_two_stablehlo = read_portable_stablehlo(step_two_program)
        for stablehlo in (step_one_stablehlo, step_two_stablehlo):
            self.assertGreaterEqual(
                stablehlo.count("stablehlo.concatenate"), 2
            )
            self.assertNotIn("stablehlo.cosine", stablehlo)
            self.assertNotIn("stablehlo.sine", stablehlo)
        self.assertIn("tensor<1x32x1023x128xf16>", step_one_stablehlo)
        self.assertIn("tensor<1x32x1024x128xf16>", step_two_stablehlo)

    def test_hf_long_decode_reaches_context_limit_with_actual_continuation(self) -> None:
        name = "attention-decode-kv-cache-long-4096"
        case = cases.make_case(name, dtype=torch.float16, seed=20260803)
        self.assertEqual(case.name, name)
        self.assertEqual(case.inputs[0].shape, (1, 1, 4096))
        for tensor in case.inputs[1:3]:
            self.assertEqual(tensor.shape, (1, 32, 4094, 128))
        first = case.materialize_expected_outputs()
        self.assertEqual(first[0].shape, (1, 1, 4096))
        for tensor in first[1:]:
            self.assertEqual(tensor.shape, (1, 32, 4095, 128))
        case.validate_actual_outputs(first)

        # Stand in for readback with distinguishable newly appended tokens.
        # The original prefix stays exact; the next step must use these values.
        actual = tuple(tensor.clone() for tensor in first)
        actual[1][0, 0, -1, 0] += 1
        actual[2][0, 0, -1, 0] -= 1
        case.validate_actual_outputs(actual)
        continuation = case.continuation_factory(actual)
        self.assertEqual(continuation.name, f"{name}-continuation")
        self.assertIsNone(continuation.continuation_factory)
        for index in (1, 2):
            self.assertTrue(torch.equal(continuation.inputs[index], actual[index]))
            self.assertFalse(torch.equal(continuation.inputs[index], first[index]))
        second = continuation.materialize_expected_outputs()
        self.assertEqual(second[0].shape, (1, 1, 4096))
        continuation.validate_actual_outputs(second)
        for index in (1, 2):
            self.assertEqual(second[index].shape, (1, 32, 4096, 128))
            self.assertTrue(torch.equal(second[index][..., :-1, :], actual[index]))
            corrupt = list(second)
            corrupt[index] = corrupt[index].clone()
            corrupt[index].view(torch.int16)[0, 31, 4094, 127] ^= 1
            with self.assertRaisesRegex(RuntimeError, "existing KV prefix"):
                continuation.validate_actual_outputs(tuple(corrupt))
        with tempfile.TemporaryDirectory() as directory:
            for step, length in ((case, 4094), (continuation, 4095)):
                program = pathlib.Path(directory) / str(length)
                step.export_program(program)
                stablehlo = read_portable_stablehlo(program)
                self.assertIn(f"tensor<1x32x{length}x128xf16>", stablehlo)
                self.assertIn(f"tensor<1x32x{length + 1}x128xf16>", stablehlo)
                self.assertGreaterEqual(stablehlo.count("stablehlo.concatenate"), 2)
        for invalid in (-1, 4095):
            with self.assertRaisesRegex(ValueError, "original context"):
                cases._attention_decode_kv_cache(
                    torch.float16, seed=20260803, past_length=invalid
                )

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
