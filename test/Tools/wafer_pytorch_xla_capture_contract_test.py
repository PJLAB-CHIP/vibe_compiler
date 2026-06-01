#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest


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

if __name__ == "__main__":
    unittest.main()
