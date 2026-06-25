#!/usr/bin/env python3
"""Unit tests for no-card Wafer runtime adapter behavior."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    return parser.parse_args()


class WaferRuntimeAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        args = parse_args()
        cls.repo_root = pathlib.Path(args.repo_root)
        cls.tool = cls.repo_root / "tools" / "wafer_runtime_adapter.py"
        cls.valid_package = (
            cls.repo_root / "test" / "Tools" / "valid-model-package.json"
        )
        cls.invalid_completion_package = (
            cls.repo_root / "test" / "Tools" / "invalid-completion-package.json"
        )

    def run_adapter(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.tool), *extra_args],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def load_valid_package(self) -> dict:
        with self.valid_package.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_package(self, directory: pathlib.Path, package: dict) -> pathlib.Path:
        path = directory / "package.json"
        path.write_text(json.dumps(package), encoding="utf-8")
        return path

    def build_fake_tx_library(
        self, directory: pathlib.Path, symbols: list[str]
    ) -> pathlib.Path:
        source = directory / "fake_tx_runtime.c"
        library = directory / "libfake_tx_runtime.so"
        source.write_text(
            "\n".join(f"int {symbol}(void) {{ return 0; }}" for symbol in symbols),
            encoding="utf-8",
        )
        result = subprocess.run(
            ["cc", "-shared", "-fPIC", str(source), "-o", str(library)],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return library

    def test_dry_run_describes_launch_plan(self) -> None:
        result = self.run_adapter(
            "--package-metadata", str(self.valid_package), "--backend", "dry-run"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend: dry-run", result.stdout)
        self.assertIn("package: model_package_sample", result.stdout)
        self.assertIn("runtime_mode: tx", result.stdout)
        self.assertIn("completion_source: runtime_stream_wait", result.stdout)
        self.assertIn("model: model_package_sample abi=wafer-cabi-v0", result.stdout)
        self.assertIn("binding: input lhs 16 bytes host_visible", result.stdout)
        self.assertIn("binding: input rhs 16 bytes host_visible", result.stdout)
        self.assertIn("binding: output out 16 bytes host_visible", result.stdout)
        self.assertIn("binding: workspace tmp 16 bytes", result.stdout)
        self.assertIn(
            "binding: resident_constant rhs_resident 16 bytes source=launch_input:rhs",
            result.stdout,
        )
        self.assertIn(
            "module: kernel tx.kcore model_package.so",
            result.stdout,
        )
        self.assertIn(
            "entrypoint: model_bpm tx.model bpm=descriptor_only",
            result.stdout,
        )
        self.assertIn(
            "entrypoint: debug_kernel tx.module module=kernel "
            "function=model_package_sample_abi debug",
            result.stdout,
        )
        self.assertIn("completion: wait runtime_stream_wait", result.stdout)
        self.assertNotIn("launch: module_kernel", result.stdout)

    def test_dry_run_describes_selected_runtime_session(self) -> None:
        result = self.run_adapter(
            "--package-metadata",
            str(self.valid_package),
            "--backend",
            "dry-run",
            "--entrypoint",
            "debug_kernel",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        expected_lines = [
            "session: package=model_package_sample runtime=tx completion=runtime_stream_wait",
            "session_binding: lhs role=input bytes=16 lifecycle=import_or_allocate,query,bind,copy_h2d read_only=true host_visible=true",
            "session_binding: out role=output bytes=16 lifecycle=allocate,query,bind,copy_d2h read_only=false host_visible=true",
            "session_binding: tmp role=workspace bytes=16 lifecycle=allocate,query,bind read_only=false host_visible=false",
            "session_binding: rhs_resident role=resident_constant bytes=16 lifecycle=allocate,query,bind,copy_h2d source=launch_input:rhs read_only=true host_visible=false",
            "module_resolve: kernel format=tx.kcore path=model_package.so",
            "launch_arg: 0 lhs role=input bytes=16",
            "launch_arg: 2 out role=output bytes=16",
            "launch_arg: 4 rhs_resident role=resident_constant bytes=16",
            "entrypoint_plan: debug_kernel executor=tx.module launch_api=txLaunchKernel module=kernel function=model_package_sample_abi arg_bytes=80",
            "completion_plan: wait runtime_stream_wait",
        ]
        for line in expected_lines:
            self.assertIn(line, result.stdout)

    def test_fake_tx_backend_constructs_tx_call_sequence(self) -> None:
        result = self.run_adapter(
            "--package-metadata",
            str(self.valid_package),
            "--backend",
            "fake-tx",
            "--entrypoint",
            "debug_kernel",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        expected_lines = [
            "backend: fake-tx",
            "entrypoint: debug_kernel tx.module",
            "txSetDevice device=0",
            "txMalloc name=lhs bytes=16",
            "txMemcpyH2D name=lhs bytes=16",
            "txMalloc name=rhs bytes=16",
            "txMemcpyH2D name=rhs bytes=16",
            "txMalloc name=out bytes=16",
            "txMalloc name=tmp bytes=16",
            "txMalloc name=rhs_resident bytes=16",
            "txMemcpyH2D name=rhs_resident bytes=16 source=rhs",
            "txModuleLoad module=model_package.so",
            "txModuleGetFunction function=model_package_sample_abi",
            "txLaunchKernel entrypoint=debug_kernel function=model_package_sample_abi arg_bytes=80",
            "txStreamSynchronize completion_source=runtime_stream_wait",
            "txMemcpyD2H name=out bytes=16",
        ]
        for line in expected_lines:
            self.assertIn(line, result.stdout)

    def test_fake_tx_backend_uses_provider_lifecycle_trace(self) -> None:
        result = self.run_adapter(
            "--package-metadata",
            str(self.valid_package),
            "--backend",
            "fake-tx",
            "--entrypoint",
            "debug_kernel",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        expected_lines = [
            "provider: fake-tx",
            "provider.call: set_device device=0",
            "provider.call: allocate_or_import name=lhs role=input bytes=16",
            "provider.call: query name=lhs",
            "provider.call: bind name=lhs",
            "provider.call: copy_h2d name=lhs bytes=16",
            "provider.call: allocate name=out role=output bytes=16",
            "provider.call: copy_d2h name=out bytes=16",
            "provider.call: wait_completion source=runtime_stream_wait",
        ]
        for line in expected_lines:
            self.assertIn(line, result.stdout)

    def test_fake_tx_backend_constructs_cluster_kernel_sequence(self) -> None:
        package = self.load_valid_package()
        package["entrypoints"].append(
            {
                "name": "cluster_debug",
                "executor": "tx.cluster",
                "module": "kernel",
                "function": "model_package_sample_cluster",
                "debug": True,
                "grid": [1, 1, 1],
                "block": [1, 1, 1],
                "cluster": [1, 1, 1],
                "binding_order": ["lhs", "rhs", "out", "tmp", "rhs_resident"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_package(pathlib.Path(tmp), package)
            result = self.run_adapter(
                "--package-metadata",
                str(path),
                "--backend",
                "fake-tx",
                "--entrypoint",
                "cluster_debug",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("entrypoint: cluster_debug tx.cluster", result.stdout)
        self.assertIn(
            "provider.call: launch_cluster_kernel entrypoint=cluster_debug "
            "function=model_package_sample_cluster arg_bytes=80",
            result.stdout,
        )
        self.assertIn(
            "txLaunchClusterKernel entrypoint=cluster_debug "
            "function=model_package_sample_cluster arg_bytes=80",
            result.stdout,
        )

    def test_fake_tx_backend_constructs_graph_sequence(self) -> None:
        package = self.load_valid_package()
        package["modules"].append(
            {"name": "graph_module", "format": "tx.graph", "path": "model.graph"}
        )
        package["entrypoints"].append(
            {
                "name": "graph_debug",
                "executor": "tx.graph",
                "module": "graph_module",
                "mod_symbol": "graph_main",
                "binding_order": ["lhs", "rhs", "out", "tmp", "rhs_resident"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_package(pathlib.Path(tmp), package)
            result = self.run_adapter(
                "--package-metadata",
                str(path),
                "--backend",
                "fake-tx",
                "--entrypoint",
                "graph_debug",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("entrypoint: graph_debug tx.graph", result.stdout)
        self.assertIn(
            "provider.call: load_graph module=model.graph mod_symbol=graph_main",
            result.stdout,
        )
        self.assertIn("txLoadGraph path=model.graph mod_symbol=graph_main", result.stdout)

    def test_fake_tx_backend_constructs_materialized_model_sequence(self) -> None:
        package = self.load_valid_package()
        package["entrypoints"][0]["bpm_descriptor"]["state"] = "materialized"
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_package(pathlib.Path(tmp), package)
            result = self.run_adapter(
                "--package-metadata",
                str(path),
                "--backend",
                "fake-tx",
                "--entrypoint",
                "model_bpm",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("entrypoint: model_bpm tx.model", result.stdout)
        self.assertIn(
            "provider.call: launch_model entrypoint=model_bpm bpm=materialized arg_bytes=80",
            result.stdout,
        )
        self.assertIn("txLaunchModel bpm_descriptor=materialized", result.stdout)

    def test_fake_tx_backend_rejects_descriptor_only_bpm_entrypoint(self) -> None:
        result = self.run_adapter(
            "--package-metadata",
            str(self.valid_package),
            "--backend",
            "fake-tx",
            "--entrypoint",
            "model_bpm",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "error: tx.model entrypoint model_bpm requires a materialized BPM descriptor",
            result.stderr,
        )

    def test_fake_tx_backend_rejects_legacy_tsm_entrypoint(self) -> None:
        package = self.load_valid_package()
        package["runtime"]["mode"] = "legacy_tsm"
        package["runtime"]["completion_source"] = "legacy_model_sync"
        package["entrypoints"] = [
            {
                "name": "legacy_run",
                "executor": "legacy.tsm",
                "binding_order": ["lhs", "rhs", "out", "tmp", "rhs_resident"],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_package(pathlib.Path(tmp), package)
            result = self.run_adapter(
                "--package-metadata",
                str(path),
                "--backend",
                "fake-tx",
                "--entrypoint",
                "legacy_run",
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "error: legacy.tsm entrypoint legacy_run requires the legacy board gate",
            result.stderr,
        )

    def test_tx_backend_reports_missing_runtime_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_adapter(
                "--package-metadata",
                str(self.valid_package),
                "--backend",
                "tx",
                "--entrypoint",
                "debug_kernel",
                "--runtime-root",
                str(pathlib.Path(tmp) / "missing-runtime-root"),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error: tx runtime library was not found", result.stderr)

    def test_tx_backend_binds_required_symbols_without_board_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = self.build_fake_tx_library(
                pathlib.Path(tmp),
                [
                    "txSetDevice",
                    "txMalloc",
                    "txFree",
                    "txMemcpy",
                    "txStreamSynchronize",
                    "txModuleLoad",
                    "txModuleGetFunction",
                    "txLaunchKernel",
                ],
            )
            result = self.run_adapter(
                "--package-metadata",
                str(self.valid_package),
                "--backend",
                "tx",
                "--entrypoint",
                "debug_kernel",
                "--runtime-library",
                str(library),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"tx_runtime_library: {library}", result.stdout)
        self.assertIn("tx_runtime_provider: ctypes", result.stdout)
        self.assertIn("tx_runtime_symbols: ok", result.stdout)
        self.assertIn(
            "board_launch_gate: not executed; enable a board environment to launch",
            result.stdout,
        )

    def test_tx_backend_reports_missing_selected_executor_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = self.build_fake_tx_library(
                pathlib.Path(tmp),
                [
                    "txSetDevice",
                    "txMalloc",
                    "txFree",
                    "txMemcpy",
                    "txStreamSynchronize",
                    "txModuleLoad",
                    "txModuleGetFunction",
                ],
            )
            result = self.run_adapter(
                "--package-metadata",
                str(self.valid_package),
                "--backend",
                "tx",
                "--entrypoint",
                "debug_kernel",
                "--runtime-library",
                str(library),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "error: tx runtime library is missing required symbol(s) for tx.module: txLaunchKernel",
            result.stderr,
        )

    def test_stub_completion_is_rejected_before_backend_selection(self) -> None:
        result = self.run_adapter(
            "--package-metadata",
            str(self.invalid_completion_package),
            "--backend",
            "dry-run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error: completion source is a known stub fence", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
