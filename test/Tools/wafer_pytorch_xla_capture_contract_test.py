#!/usr/bin/env python3

import importlib.util
import json
import numpy as np
import pathlib
import sys
import tempfile
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "wafer_pytorch_xla_capture.py"


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


class FakeShardData:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeShard:
    def __init__(self, array, indices, shard_device, replica_id):
        self.data = FakeShardData(array)
        self.indices = indices
        self.shard_device = shard_device
        self.replica_id = replica_id

    @property
    def unpadded_data(self):
        return self.data


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

    def test_capture_uses_torch_xla_runtime_and_saves_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = pathlib.Path(tmp) / "bundle"

            self.tool.emit_reference_stablehlo_bundle(
                bundle_path=bundle_path,
                torch_module=self.fake_torch,
                stablehlo_module=self.fake_stablehlo,
                reference_module_factory=FakeReferenceModule,
            )

            self.assertEqual(self.fake_stablehlo.calls, [self.fake_torch.exported_program])
            self.assertTrue(self.fake_stablehlo.last_options.export_weights)
            self.assertTrue(self.fake_stablehlo.last_options.save_weights)
            self.assertTrue(self.fake_stablehlo.last_options.include_human_readable_text)
            self.assertTrue(self.fake_torch.no_grad_entered)
            self.assertTrue((bundle_path / "functions" / "forward.mlir").is_file())
            self.assertTrue((bundle_path / "functions" / "forward.meta").is_file())
            self.assertTrue((bundle_path / "data" / "weight").is_file())

    def test_emit_rejects_missing_bundle_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = pathlib.Path(tmp) / "bundle"
            bundle_path.mkdir()

            with self.assertRaisesRegex(RuntimeError, "missing StableHLO bundle file"):
                self.tool._verify_bundle_layout(bundle_path)

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

    def test_default_input_strategy_uses_replicated_fallback(self):
        strategy = self.tool.create_default_input_sharding_strategy(
            tile_count=16, size=31
        )

        self.assertEqual(strategy.mesh_shape, (16,))
        self.assertEqual(strategy.axis_names, ("tile",))
        self.assertEqual(strategy.input_spec, (None, None))
        self.assertEqual(strategy.weight_spec, (None, None))
        self.assertEqual(strategy.bias_spec, (None,))

    def test_partitioned_export_disables_hlo_fusion(self):
        flags = self.tool._with_disabled_hlo_pass(
            "--xla_dump_to=/tmp/xla --xla_disable_hlo_passes=cse", "fusion"
        )

        self.assertIn("--xla_dump_to=/tmp/xla", flags)
        self.assertIn("--xla_disable_hlo_passes=cse,fusion", flags)

    def test_runtime_shard_binding_writes_rank_payload_files(self):
        signature = types.SimpleNamespace(shape=[2, 4], dtype="float32")
        location = types.SimpleNamespace(type_="parameter", name="weight")
        func = types.SimpleNamespace(
            meta=types.SimpleNamespace(
                name="forward",
                input_signature=[signature],
                input_locations=[location],
            )
        )
        bundle = types.SimpleNamespace(stablehlo_funcs=[func])
        global_state = {"weight": np.zeros((4, 4), dtype=np.float32)}
        runtime_shards = {
            "weight": [
                FakeShard(
                    np.ones((2, 4), dtype=np.float32),
                    [slice(0, 2, 1), slice(0, 4, 1)],
                    "CPU:0",
                    0,
                ),
                FakeShard(
                    np.full((2, 4), 2, dtype=np.float32),
                    [slice(2, 4, 1), slice(0, 4, 1)],
                    "CPU:1",
                    0,
                ),
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = pathlib.Path(tmp) / "bundle"
            (bundle_path / "functions").mkdir(parents=True)
            (bundle_path / "data").mkdir()
            (bundle_path / "data" / "weight").write_bytes(b"global")

            self.tool.write_parameter_shard_bindings(
                bundle_path=bundle_path,
                bundle=bundle,
                global_state_dict=global_state,
                parameter_shards=runtime_shards,
            )

            manifest = json.loads(
                (bundle_path / "functions" / "forward.parameter_shards.json")
                .read_text()
            )

            self.assertEqual(manifest["parameter_shards_version"], 2)
            self.assertEqual(manifest["logical_rank_count"], 2)
            parameter = manifest["parameters"][0]
            self.assertNotIn("source", parameter)
            self.assertEqual(parameter["global_shape"], [4, 4])
            self.assertEqual(parameter["local_shape"], [2, 4])
            self.assertEqual(parameter["shards"][0]["file"],
                             "parameter_shards/weight/rank_00000.npy")
            self.assertEqual(parameter["shards"][1]["offsets"], [2, 0])
            self.assertTrue(
                (bundle_path / "parameter_shards" / "weight" / "rank_00000.npy")
                .is_file()
            )
            self.assertFalse((bundle_path / "data" / "weight").exists())
            np.testing.assert_array_equal(
                np.load(bundle_path / parameter["shards"][1]["file"]),
                np.full((2, 4), 2, dtype=np.float32),
            )

if __name__ == "__main__":
    unittest.main()
