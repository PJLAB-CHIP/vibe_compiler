#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

import numpy


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = (
    REPO_ROOT / "test" / "Tools" / "Inputs" / "wafer_pytorch_xla_capture.py"
)


def load_tool_module():
    spec = importlib.util.spec_from_file_location("wafer_pytorch_xla_capture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype

    def __matmul__(self, other):
        return FakeTensor((self.shape[0], other.shape[1]), self.dtype)

    def __add__(self, other):
        return FakeTensor(self.shape, self.dtype)


class FakeParameter(FakeTensor):
    def __init__(self, value, dtype="float32"):
        if isinstance(value, FakeTensor):
            super().__init__(value.shape, value.dtype)
        else:
            super().__init__(value, dtype)


class FakeModule:
    def eval(self):
        self.was_eval = True
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakeReferenceModule(FakeModule):
    def __init__(self):
        self.weight = FakeParameter((4096, 4096))
        self.bias = FakeParameter((4096,))


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.exported_program = object()
        self.export_calls = []
        self.empty_calls = []
        self.no_grad_entered = False
        self.float32 = "float32"
        self.bfloat16 = "bfloat16"
        self.uint16 = "uint16"

        self.nn = types.SimpleNamespace(Module=FakeModule, Parameter=FakeParameter)
        self.export = types.SimpleNamespace(export=self.export_model)

    def empty(self, *shape, dtype=None):
        self.empty_calls.append((shape, dtype))
        return FakeTensor(shape, dtype or "float32")

    def export_model(self, model, args):
        self.export_calls.append((model, args))
        if hasattr(model, "forward"):
            model(*args)
        return self.exported_program

    def tanh(self, tensor):
        return FakeTensor(tensor.shape, tensor.dtype)

    def no_grad(self):
        fake_torch = self

        class NoGrad:
            def __enter__(self):
                fake_torch.no_grad_entered = True

            def __exit__(self, exc_type, exc, tb):
                return False

        return NoGrad()


class FakeStableHLO(types.ModuleType):
    def __init__(self):
        super().__init__("torch_xla.stablehlo")
        self.calls = []
        self.last_options = None
        self.VariableType = types.SimpleNamespace(PARAMETER="parameter")

        class StableHLOExportOptions:
            def __init__(self):
                self.export_weights = True
                self.inline_all_constant = True
                self.include_human_readable_text = True

        self.StableHLOExportOptions = StableHLOExportOptions

    def exported_program_to_stablehlo(self, exported, options=None):
        self.calls.append(exported)
        self.last_options = options
        return FakeStableHLOProgram()


class FakeStableHLOProgram:
    def __init__(self):
        self.saved_path = None

    def save(self, path):
        self.saved_path = pathlib.Path(path)
        functions = self.saved_path / "functions"
        data = self.saved_path / "data"
        functions.mkdir(parents=True)
        data.mkdir()
        (functions / "forward.mlir").write_text(
            "module @IrToHlo.16 {\n"
            "  func.func @main(%arg0: tensor<4096xf32>, "
            "%arg1: tensor<4096x4096xf32>, "
            "%arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {\n"
            "    return %arg2 : tensor<4096x4096xf32>\n"
            "  }\n"
            "}\n"
        )
        (functions / "forward.meta").write_text(
            '{"name":"forward","input_signature":['
            '{"shape":[4096],"dtype":"float32","dynamic_dims":[]},'
            '{"shape":[4096,4096],"dtype":"float32","dynamic_dims":[]},'
            '{"shape":[4096,4096],"dtype":"float32","dynamic_dims":[]}],'
            '"output_signature":[{"shape":[4096,4096],"dtype":"float32",'
            '"dynamic_dims":[]}],"input_locations":['
            '{"type_":"parameter","position":-1,"name":"bias"},'
            '{"type_":"parameter","position":-1,"name":"weight"},'
            '{"type_":"input_arg","position":0,"name":""}]}\n'
        )
        (functions / "forward.bytecode").write_bytes(b"bytecode")
        (data / "weight").write_bytes(b"0" * 16)
        (data / "bias").write_bytes(b"0" * 16)


class FakeMesh:
    def __init__(self, device_ids, mesh_shape, axis_names):
        self.device_ids = tuple(device_ids)
        self.mesh_shape = tuple(mesh_shape)
        self.axis_names = tuple(axis_names)


class FakeSpmd(types.ModuleType):
    def __init__(self):
        super().__init__("torch_xla.distributed.spmd")
        self.meshes = []
        self.mark_calls = []

    def Mesh(self, device_ids, mesh_shape, axis_names):
        mesh = FakeMesh(device_ids, mesh_shape, axis_names)
        self.meshes.append(mesh)
        return mesh

    def mark_sharding(self, tensor, mesh, partition_spec):
        self.mark_calls.append((tensor, mesh, partition_spec))
        return tensor


class WaferPyTorchXlaCaptureContractTest(unittest.TestCase):
    def setUp(self):
        self.saved_modules = dict(sys.modules)
        self.fake_torch = FakeTorch()
        self.fake_stablehlo = FakeStableHLO()
        sys.modules["torch"] = self.fake_torch
        sys.modules["torch_xla"] = types.ModuleType("torch_xla")
        sys.modules["torch_xla.stablehlo"] = self.fake_stablehlo
        self.tool = load_tool_module()

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.saved_modules)

    def test_capture_uses_torch_xla_runtime_and_saves_program_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            program_dir = pathlib.Path(tmp) / "program"

            self.tool.emit_reference_stablehlo_program(
                program_dir=program_dir,
                torch_module=self.fake_torch,
                stablehlo_module=self.fake_stablehlo,
                reference_module_factory=FakeReferenceModule,
            )

            self.assertEqual(self.fake_stablehlo.calls, [self.fake_torch.exported_program])
            self.assertTrue(self.fake_stablehlo.last_options.export_weights)
            self.assertTrue(self.fake_stablehlo.last_options.save_weights)
            self.assertTrue(self.fake_stablehlo.last_options.include_human_readable_text)
            self.assertTrue(self.fake_torch.no_grad_entered)
            self.assertTrue((program_dir / "functions" / "forward.mlir").is_file())
            self.assertTrue((program_dir / "functions" / "forward.meta").is_file())
            self.assertTrue((program_dir / "data" / "weight").is_file())

    def test_emit_rejects_missing_program_dir_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            program_dir = pathlib.Path(tmp) / "program"
            program_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "missing StableHLO program directory file"):
                self.tool._verify_program_dir_layout(program_dir)

    def test_sharding_strategy_matrix_names_are_fixed(self):
        self.assertEqual(
            self.tool.SHARDING_STRATEGY_NAMES,
            (
                "data",
                "column",
                "row",
                "2d-output",
                "2d-contracting-output",
                "partial-replication",
            ),
        )

    def test_sharding_strategy_applies_framework_marks_to_inputs_only(self):
        fake_spmd = FakeSpmd()
        strategy = self.tool.get_sharding_strategy("row")
        mesh = self.tool.create_spmd_mesh(fake_spmd, strategy)
        module = types.SimpleNamespace(weight=object(), bias=object())
        input_tensor = object()

        self.tool.apply_strategy_marks(
            spmd_module=fake_spmd,
            strategy=strategy,
            mesh=mesh,
            input_tensor=input_tensor,
            reference_module=module,
        )

        self.assertEqual(fake_spmd.meshes[0].mesh_shape, (16,))
        self.assertEqual(fake_spmd.meshes[0].axis_names, ("tp",))
        self.assertEqual(
            fake_spmd.mark_calls,
            [
                (input_tensor, mesh, (None, "tp")),
                (module.weight, mesh, ("tp", None)),
                (module.bias, mesh, (None,)),
            ],
        )

    def test_sharding_strategy_accepts_bias_free_gemm(self):
        fake_spmd = FakeSpmd()
        strategy = self.tool.get_sharding_strategy("row")
        mesh = self.tool.create_spmd_mesh(fake_spmd, strategy)
        module = types.SimpleNamespace(weight=object(), bias=None)
        input_tensor = object()

        self.tool.apply_strategy_marks(
            spmd_module=fake_spmd,
            strategy=strategy,
            mesh=mesh,
            input_tensor=input_tensor,
            reference_module=module,
        )

        self.assertEqual(
            fake_spmd.mark_calls,
            [
                (input_tensor, mesh, (None, "tp")),
                (module.weight, mesh, ("tp", None)),
            ],
        )

    def test_hf_megatron_marks_explicit_llama_parameter_specs(self):
        fake_spmd = FakeSpmd()
        mesh = fake_spmd.Mesh(list(range(16)), (16,), ("tensor",))
        input_norm = object()
        post_attention_norm = object()
        q_projection = object()
        k_projection = object()
        v_projection = object()
        o_projection = object()
        gate_projection = object()
        up_projection = object()
        down_projection = object()
        parameters = (
            input_norm,
            post_attention_norm,
            q_projection,
            k_projection,
            v_projection,
            o_projection,
            gate_projection,
            up_projection,
            down_projection,
        )
        parameter_specs = (
            (input_norm, (None,)),
            (post_attention_norm, (None,)),
            (q_projection, ("tensor", None)),
            (k_projection, ("tensor", None)),
            (v_projection, ("tensor", None)),
            (o_projection, (None, "tensor")),
            (gate_projection, ("tensor", None)),
            (up_projection, ("tensor", None)),
            (down_projection, (None, "tensor")),
        )
        module = types.SimpleNamespace(
            parameters=lambda: iter(parameters),
            wafer_parameter_sharding_specs=lambda: parameter_specs,
        )
        input_tensor = object()

        self.tool.apply_hf_megatron_sharding_marks(
            spmd_module=fake_spmd,
            mesh=mesh,
            input_tensor=input_tensor,
            reference_module=module,
        )

        self.assertEqual(
            fake_spmd.mark_calls,
            [
                (input_tensor, mesh, (None, None, None)),
                (input_norm, mesh, (None,)),
                (post_attention_norm, mesh, (None,)),
                (q_projection, mesh, ("tensor", None)),
                (k_projection, mesh, ("tensor", None)),
                (v_projection, mesh, ("tensor", None)),
                (o_projection, mesh, (None, "tensor")),
                (gate_projection, mesh, ("tensor", None)),
                (up_projection, mesh, ("tensor", None)),
                (down_projection, mesh, (None, "tensor")),
            ],
        )

    def test_hf_transformer_config_loader_preserves_llama_shape_fields(self):
        config_path = (
            REPO_ROOT
            / "test"
            / "Tools"
            / "Inputs"
            / "hf"
            / "tiny-random-llama-config.json"
        )

        config = self.tool.load_hf_transformer_config(config_path)

        self.assertEqual(config["model_type"], "llama")
        self.assertEqual(config["hidden_size"], 16)
        self.assertEqual(config["intermediate_size"], 64)
        self.assertEqual(config["num_attention_heads"], 4)

    def test_workload_corpus_spec_fixes_source_and_payload_facts(self):
        spec = self.tool.load_workload_corpus_spec(
            self.tool.DEFAULT_WORKLOAD_CORPUS_SPEC
        )

        self.assertEqual(spec["corpus_id"], "wafer-single-card-vertical")
        self.assertEqual(
            spec["source"]["framework"],
            {
                "name": "PyTorch",
                "version": "2.5.0+cpu",
                "git_revision": "32f585d9346e316e554c8d9bf7548af9f62141fc",
            },
        )
        self.assertEqual(
            spec["source"]["exporter"]["git_revision"],
            "396608c7105b3763874fe3800dfabdfa2b38a28a",
        )
        self.assertEqual(
            [case["id"] for case in spec["cases"]],
            [
                "linear-residual-mlp-f32",
                "simple-gemm-f16",
                "simple-gemm-bf16",
                "large-gemm-f32",
                "large-gemm-f16",
                "tiny-llama-decoder-f32",
            ],
        )
        expected_dtypes = {
            "linear-residual-mlp-f32": "float32",
            "simple-gemm-f16": "float16",
            "simple-gemm-bf16": "bfloat16",
            "large-gemm-f32": "float32",
            "large-gemm-f16": "float16",
            "tiny-llama-decoder-f32": "float32",
        }
        for case in spec["cases"]:
            self.assertEqual(case["dtype"], expected_dtypes[case["id"]])
            self.assertGreater(case["seed"], 0)
            self.assertEqual(
                case["source_revision"], case["digests"]["source_config"]
            )
            for digest in case["digests"].values():
                self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")

    def test_llama_scale_payload_is_explicit(self):
        spec_path = (
            REPO_ROOT
            / "test"
            / "Tools"
            / "Inputs"
            / "workloads"
            / "llama-2-7b-block.json"
        )
        spec = self.tool.load_workload_corpus_spec(spec_path)
        case = spec["cases"][0]

        self.assertEqual(
            case["config"]["payload"],
            {
                "algorithm": (
                    "wafer-exact-f16-splitmix64-counter-byte-scaled"
                ),
                "input_denominator": 128,
                "normalization_delta_denominator": 4096,
                "projection_denominator": 8192,
            },
        )
        self.assertNotIn("payload_algorithm", spec["source"])
        self.assertEqual(
            self.tool._load_llama_scale_payload_config(
                case["id"], case["config"]
            ),
            (128, 8192, 4096),
        )
        self.assertEqual(case["reference"]["atol"], 0.004)
        self.assertEqual(case["reference"]["rtol"], 0.002)

    def test_llama_scale_payload_rejects_implicit_or_unknown_algorithm(self):
        with self.assertRaisesRegex(
            RuntimeError, "requires explicit config.payload"
        ):
            self.tool._load_llama_scale_payload_config("scale-case", {})
        with self.assertRaisesRegex(
            RuntimeError, "unsupported payload algorithm"
        ):
            self.tool._load_llama_scale_payload_config(
                "scale-case",
                {
                    "payload": {
                        "algorithm": "unsupported-algorithm",
                        "input_denominator": 128,
                        "projection_denominator": 8192,
                        "normalization_delta_denominator": 4096,
                    }
                },
            )

    def test_diagnostic_variant_is_disjoint_from_fixed_corpus(self):
        spec_path = (
            REPO_ROOT
            / "test"
            / "Tools"
            / "Inputs"
            / "workloads"
            / "llama-2-7b-block.json"
        )
        spec = self.tool.load_workload_corpus_spec(spec_path)
        base = spec["cases"][0]
        fixed_seed = base["seed"]
        fixed_digests = dict(base["digests"])
        variant = self.tool._make_diagnostic_variant_case(base, 20260729)

        self.assertEqual(base["seed"], fixed_seed)
        self.assertEqual(base["digests"], fixed_digests)
        self.assertEqual(variant["diagnostic_base_case_id"], base["id"])
        self.assertEqual(variant["seed"], 20260729)
        self.assertNotIn("digests", variant)
        self.assertIn("diagnostic-seed-20260729", variant["id"])

        with self.assertRaisesRegex(RuntimeError, "Llama scale base case"):
            self.tool._make_diagnostic_variant_case(
                {"id": "not-scale", "kind": "simple_gemm"}, 1
            )
        for invalid in (-1, 1 << 64, True, "1"):
            with self.assertRaisesRegex(RuntimeError, "fit uint64"):
                self.tool._make_diagnostic_variant_case(base, invalid)

    def test_llama_scale_counter_payload_is_exact_and_chunk_independent(self):
        expected_bits = numpy.array(
            [
                46656,
                46464,
                46848,
                14480,
                12800,
                14752,
                47792,
                14208,
                14736,
                14848,
                46208,
                45824,
                47952,
                47936,
                46464,
                43776,
            ],
            dtype=numpy.dtype("<u2"),
        )
        payloads = [
            self.tool._deterministic_counter_float16_array(
                numpy,
                (2, 8),
                seed=0,
                stream=0,
                denominator=128,
                chunk_elements=chunk_elements,
            )
            for chunk_elements in (1, 5, 1 << 20)
        ]
        for payload in payloads:
            self.assertEqual(payload.dtype, numpy.dtype("float16"))
            numpy.testing.assert_array_equal(
                payload.reshape(-1).view(numpy.dtype("<u2")), expected_bits
            )
        numpy.testing.assert_array_equal(payloads[0], payloads[1])
        numpy.testing.assert_array_equal(payloads[1], payloads[2])

    def test_llama_scale_counter_payload_separates_streams_and_lag257(self):
        first = self.tool._deterministic_counter_float16_array(
            numpy, (8192,), seed=20260715, stream=3, denominator=8192
        )
        second = self.tool._deterministic_counter_float16_array(
            numpy, (8192,), seed=20260715, stream=4, denominator=8192
        )
        self.assertFalse(numpy.array_equal(first, second))
        self.assertFalse(numpy.array_equal(first[:-257], first[257:]))
        self.assertLess(
            int(numpy.count_nonzero(first[:-257] == first[257:])),
            first.size // 16,
        )

    def test_llama_scale_counter_payload_rejects_invalid_domain(self):
        invalid = (
            ({"seed": -1, "stream": 0, "denominator": 128}, "seed"),
            (
                {"seed": 0, "stream": 1 << 64, "denominator": 128},
                "stream",
            ),
            ({"seed": 0, "stream": 0, "denominator": True}, "denominator"),
            ({"seed": 0, "stream": 0, "denominator": "128"}, "denominator"),
            ({"seed": 0, "stream": 0, "denominator": 3}, "denominator"),
            (
                {"seed": 0, "stream": 0, "denominator": 1 << 25},
                "denominator",
            ),
        )
        for arguments, message in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.tool._deterministic_counter_float16_array(
                        numpy, (2,), **arguments
                    )
        with self.assertRaisesRegex(RuntimeError, "chunk size"):
            self.tool._deterministic_counter_float16_array(
                numpy,
                (2,),
                seed=0,
                stream=0,
                denominator=128,
                chunk_elements=0,
            )

    def test_nonfinite_payload_is_rejected_with_value_name(self):
        finite = numpy.zeros((2,), dtype=numpy.float16)
        cases = (
            (
                "input 0",
                {"input_array": numpy.array([0.0, numpy.nan], dtype=numpy.float16)},
            ),
            (
                "parameter 'q_proj.weight'",
                {
                    "parameters": {
                        "q_proj.weight": numpy.array(
                            [0.0, numpy.inf], dtype=numpy.float16
                        )
                    }
                },
            ),
            (
                "expected output 0",
                {"expected": numpy.array([0.0, -numpy.inf], dtype=numpy.float16)},
            ),
        )
        for value_name, override in cases:
            arguments = {
                "case_id": "scale-case",
                "input_array": finite,
                "extra_inputs": {},
                "parameters": {},
                "expected": finite,
                **override,
            }
            with self.subTest(value_name=value_name):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"workload case 'scale-case' {value_name} contains 1 "
                    r"non-finite value",
                ):
                    self.tool._verify_workload_payload_finite(
                        numpy, **arguments
                    )

    def test_finite_validation_covers_chunks_bfloat16_and_empty(self):
        chunk_elements = 1 << 20
        across_chunks = numpy.zeros((1, chunk_elements + 8), dtype=numpy.float16)
        across_chunks[0, chunk_elements + 3] = numpy.nan
        across_chunks[0, chunk_elements + 7] = numpy.inf
        with self.assertRaises(RuntimeError) as failure:
            self.tool._verify_finite_workload_array(
                numpy,
                across_chunks,
                case_id="chunk-case",
                value_name="parameter 'large.weight'",
            )
        self.assertIn("contains 2 non-finite value(s)", str(failure.exception))
        self.assertIn(
            f"first_index=(0, {chunk_elements + 3})", str(failure.exception)
        )

        bfloat16_nonfinite = self.tool._float32_to_bfloat16_storage(
            numpy, numpy.array([0.0, -numpy.inf], dtype=numpy.float32)
        )
        with self.assertRaisesRegex(
            RuntimeError, r"contains 1 non-finite value.*first_index=\(1,\)"
        ):
            self.tool._verify_finite_workload_array(
                numpy,
                bfloat16_nonfinite,
                case_id="bf16-case",
                value_name="expected output 0",
            )

        for empty in (
            numpy.empty((0, 3), dtype=numpy.float16),
            self.tool._float32_to_bfloat16_storage(
                numpy, numpy.empty((0,), dtype=numpy.float32)
            ),
        ):
            self.tool._verify_finite_workload_array(
                numpy,
                empty,
                case_id="empty-case",
                value_name="input 0",
            )

    def test_bfloat16_storage_is_canonical_little_endian(self):
        values = numpy.array([1.0, -2.0, 0.5], dtype=numpy.float32)
        storage = self.tool._float32_to_bfloat16_storage(numpy, values)
        self.assertEqual(storage.tobytes(), b"\x80\x3f\x00\xc0\x00\x3f")
        numpy.testing.assert_array_equal(
            self.tool._bfloat16_storage_to_float32(numpy, storage), values
        )

    def test_lazy_bfloat16_constant_uses_canonical_workload_storage(self):
        raw_words = numpy.array([0x3F80, 0xC000], dtype=numpy.uint16)

        class FakeUInt16Tensor:
            def numpy(self):
                return raw_words

        class FakeBFloat16Tensor:
            shape = (2,)
            dtype = self.fake_torch.bfloat16

            def detach(self):
                return self

            def cpu(self):
                return self

            def view(self, dtype):
                self_outer.assertEqual(dtype, self_outer.fake_torch.uint16)
                return FakeUInt16Tensor()

            def numpy(self):
                raise AssertionError("BF16 must not use Tensor.numpy() directly")

        self_outer = self
        constant = FakeBFloat16Tensor()
        input_tensor = FakeTensor((2,), self.fake_torch.bfloat16)
        output_tensor = FakeTensor((2,), self.fake_torch.bfloat16)
        input_location = types.SimpleNamespace(
            input_arg=lambda position: ("input_arg", position),
            parameter=lambda name: ("parameter", name),
            constant=lambda position: ("constant", position),
        )
        stablehlo = types.SimpleNamespace(
            InputLocation=input_location,
            VariableSignature=lambda **kwargs: types.SimpleNamespace(**kwargs),
            StableHLOFunctionMeta=lambda **kwargs: types.SimpleNamespace(**kwargs),
            StableHLOFunc=lambda **kwargs: types.SimpleNamespace(**kwargs),
            StableHLOModelBundle=lambda **kwargs: types.SimpleNamespace(**kwargs),
        )
        xla_model = types.SimpleNamespace(
            get_stablehlo=lambda outputs: "module {}",
            get_stablehlo_bytecode=lambda outputs: b"bytecode",
        )
        xlac = types.SimpleNamespace(
            _xla_get_tensor_id=lambda tensor: id(tensor),
            _get_tensors_xla_device_data_node=lambda outputs: (
                [id(constant)],
                [constant],
            ),
        )
        reference_module = types.SimpleNamespace(named_parameters=lambda: [])

        exported_model = self.tool._build_lazy_stablehlo_program(
            torch_module=self.fake_torch,
            stablehlo_module=stablehlo,
            xla_model_module=xla_model,
            xlac_module=xlac,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            reference_module=reference_module,
            state_dict={},
        )

        self.assertEqual(len(exported_model.additional_constants), 1)
        saved = exported_model.additional_constants[0]
        self.assertEqual(saved.dtype, numpy.dtype("|V2"))
        self.assertEqual(saved.tobytes(), b"\x80\x3f\x00\xc0")
        metadata = exported_model.stablehlo_funcs[0].meta
        self.assertEqual(metadata.input_locations, [("constant", 0)])
        self.assertEqual(metadata.input_signature[0].dtype, "bfloat16")

    def test_public_npy_payloads_are_canonical_little_endian(self):
        big_endian_f16 = numpy.array(
            [1.0, -2.0], dtype=numpy.dtype(">f2")
        )
        canonical_f16 = self.tool._canonical_little_endian_array(
            numpy, big_endian_f16
        )
        self.assertEqual(canonical_f16.dtype, numpy.dtype("<f2"))
        self.assertEqual(canonical_f16.tobytes(), b"\x00\x3c\x00\xc0")

        with tempfile.TemporaryDirectory() as tmp:
            destination = pathlib.Path(tmp) / "payload.npy"
            self.tool._save_workload_array(
                numpy, destination, big_endian_f16
            )
            saved = numpy.load(destination, allow_pickle=False)
        self.assertEqual(saved.dtype, numpy.dtype("<f2"))
        self.assertEqual(saved.tobytes(), b"\x00\x3c\x00\xc0")

        opaque_bf16 = numpy.array(
            [b"\x80\x3f", b"\x00\xc0"], dtype=numpy.dtype("|V2")
        )
        canonical_bf16 = self.tool._canonical_little_endian_array(
            numpy, opaque_bf16
        )
        self.assertEqual(canonical_bf16.dtype, numpy.dtype("|V2"))
        self.assertEqual(canonical_bf16.tobytes(), b"\x80\x3f\x00\xc0")

    def test_public_npy_scalars_preserve_rank_zero(self):
        scalar_payloads = (
            (
                numpy.array(1.0, dtype=numpy.dtype(">f4")),
                numpy.dtype("<f4"),
                b"\x00\x00\x80\x3f",
            ),
            (
                numpy.array(b"\x80\x3f", dtype=numpy.dtype("|V2")),
                numpy.dtype("|V2"),
                b"\x80\x3f",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (source, expected_dtype, expected_bytes) in enumerate(
                scalar_payloads
            ):
                with self.subTest(dtype=expected_dtype):
                    canonical = self.tool._canonical_little_endian_array(
                        numpy, source
                    )
                    self.assertEqual(canonical.shape, ())
                    self.assertEqual(canonical.dtype, expected_dtype)
                    self.assertEqual(canonical.tobytes(), expected_bytes)

                    destination = pathlib.Path(tmp) / f"scalar-{index}.npy"
                    self.tool._save_workload_array(
                        numpy, destination, source
                    )
                    saved = numpy.load(destination, allow_pickle=False)
                    self.assertEqual(saved.shape, ())
                    self.assertEqual(saved.dtype, expected_dtype)
                    self.assertEqual(saved.tobytes(), expected_bytes)

    def test_cpu_reference_is_independent_and_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = pathlib.Path(tmp) / "first"
            second = pathlib.Path(tmp) / "second"

            self.tool.emit_workload_cpu_reference(
                self.tool.DEFAULT_WORKLOAD_CORPUS_SPEC,
                "tiny-llama-decoder-f32",
                first,
            )
            self.tool.emit_workload_cpu_reference(
                self.tool.DEFAULT_WORKLOAD_CORPUS_SPEC,
                "tiny-llama-decoder-f32",
                second,
            )

            first_files = sorted(
                path.relative_to(first) for path in first.rglob("*") if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second)
                for path in second.rglob("*")
                if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual(
                    (first / relative).read_bytes(), (second / relative).read_bytes()
                )
            self.assertEqual(self.fake_torch.export_calls, [])

if __name__ == "__main__":
    unittest.main()
