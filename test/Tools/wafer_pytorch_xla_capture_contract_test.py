#!/usr/bin/env python3

import importlib.util
import json
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


class FakeParameter(FakeTensor):
    pass


class FakeModule:
    def eval(self):
        self.was_eval = True
        return self


class FakeSmokeModule(FakeModule):
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
        return self.exported_program

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
        return fake_stablehlo_program()


def fake_stablehlo_program():
    meta = types.SimpleNamespace(
        input_locations=[
            types.SimpleNamespace(type_="input_arg", position=0, name=""),
            types.SimpleNamespace(type_="parameter", position=-1, name="weight"),
            types.SimpleNamespace(type_="parameter", position=-1, name="bias"),
        ],
        input_signature=[
            types.SimpleNamespace(shape=[4096, 4096], dtype="float32"),
            types.SimpleNamespace(shape=[4096, 4096], dtype="float32"),
            types.SimpleNamespace(shape=[4096], dtype="float32"),
        ],
    )
    func = types.SimpleNamespace(meta=meta)
    bundle = types.SimpleNamespace(stablehlo_funcs=[func])
    text = """module {
  func.func public @forward(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>, %arg2: tensor<4096xf32>) -> tensor<4096x4096xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) : (tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1 = stablehlo.tanh %0 : tensor<4096x4096xf32>
    return %1 : tensor<4096x4096xf32>
  }
}
"""
    return types.SimpleNamespace(_bundle=bundle, get_stablehlo_text=lambda: text)


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

    def test_capture_uses_torch_xla_runtime_and_disables_weight_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            mlir_path = pathlib.Path(tmp) / "artifact.mlir"
            sidecar_path = pathlib.Path(tmp) / "artifact.json"
            config_path = pathlib.Path(tmp) / "compile_config.json"

            self.tool.emit_p2f1_smoke_artifact(
                mlir_path=mlir_path,
                sidecar_path=sidecar_path,
                config_path=config_path,
                torch_module=self.fake_torch,
                stablehlo_module=self.fake_stablehlo,
                smoke_module_factory=FakeSmokeModule,
            )

            self.assertEqual(self.fake_stablehlo.calls, [self.fake_torch.exported_program])
            self.assertFalse(self.fake_stablehlo.last_options.export_weights)
            self.assertFalse(self.fake_stablehlo.last_options.inline_all_constant)
            self.assertTrue(self.fake_stablehlo.last_options.include_human_readable_text)
            self.assertTrue(self.fake_torch.no_grad_entered)

    def test_emit_adds_constant_arg_metadata_and_sidecar_without_parameter_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            mlir_path = pathlib.Path(tmp) / "artifact.mlir"
            sidecar_path = pathlib.Path(tmp) / "artifact.json"
            config_path = pathlib.Path(tmp) / "compile_config.json"

            self.tool.emit_p2f1_smoke_artifact(
                mlir_path=mlir_path,
                sidecar_path=sidecar_path,
                config_path=config_path,
                torch_module=self.fake_torch,
                stablehlo_module=self.fake_stablehlo,
                smoke_module_factory=FakeSmokeModule,
            )

            mlir_text = mlir_path.read_text()
            self.assertIn('tensor<4096x4096xf32> {wafer.frontend.constant = "const_arg_1"}', mlir_text)
            self.assertIn('tensor<4096xf32> {wafer.frontend.constant = "const_arg_2"}', mlir_text)
            self.assertNotIn("weight", mlir_text)
            self.assertNotIn("bias", mlir_text)

            sidecar = json.loads(sidecar_path.read_text())
            self.assertEqual(sidecar["version"], 0)
            self.assertEqual(
                sidecar["constants"],
                [
                    {
                        "function": "forward",
                        "arg": 1,
                        "resource_key": "const_arg_1",
                        "shape": [4096, 4096],
                        "dtype": "f32",
                        "byte_size": 67108864,
                    },
                    {
                        "function": "forward",
                        "arg": 2,
                        "resource_key": "const_arg_2",
                        "shape": [4096],
                        "dtype": "f32",
                        "byte_size": 16384,
                    },
                ],
            )

            config = json.loads(config_path.read_text())
            self.assertEqual(config["version"], 0)
            self.assertEqual(config["artifact_kind"], "p2f1_framework_capture_smoke")
            self.assertEqual(config["input_shape"], [4096, 4096])
            self.assertEqual(config["input_dtype"], "f32")
            self.assertEqual(config["constant_binding"], "function_arg_sidecar")
            self.assertEqual(
                config["sharding_import_policy"],
                "preserve_stablehlo_sdy_annotations",
            )


if __name__ == "__main__":
    unittest.main()
