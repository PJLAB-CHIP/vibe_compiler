#!/usr/bin/env python3
"""Unit tests for no-card Wafer runtime adapter behavior."""

from __future__ import annotations

import argparse
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
        cls.valid_manifest = (
            cls.repo_root / "test" / "Tools" / "valid-package-manifest.json"
        )
        cls.invalid_completion_manifest = (
            cls.repo_root / "test" / "Tools" / "invalid-completion-manifest.json"
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

    def test_dry_run_describes_launch_plan(self) -> None:
        result = self.run_adapter(
            "--manifest", str(self.valid_manifest), "--backend", "dry-run"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend: dry-run", result.stdout)
        self.assertIn("package: manifest_schema_fixture", result.stdout)
        self.assertIn("runtime_mode: tx", result.stdout)
        self.assertIn("completion_source: runtime_stream_wait", result.stdout)
        self.assertIn("model: manifest_schema_fixture abi=wafer-cabi-v0", result.stdout)
        self.assertIn("binding: input lhs 16 bytes host_visible", result.stdout)
        self.assertIn("binding: input rhs 16 bytes host_visible", result.stdout)
        self.assertIn("binding: output out 16 bytes host_visible", result.stdout)
        self.assertIn("binding: workspace tmp 16 bytes", result.stdout)
        self.assertIn(
            "binding: resident_constant rhs_resident 16 bytes source=launch_input:rhs",
            result.stdout,
        )
        self.assertIn(
            "artifact: schema_fixture_kernel kcore_shared_object schema_fixture.so",
            result.stdout,
        )
        self.assertIn(
            "strategy: model_bpm tx_model_bpm bpm=descriptor_only",
            result.stdout,
        )
        self.assertIn(
            "strategy: debug_kernel tx_module_kernel artifact=schema_fixture_kernel "
            "entrypoint=manifest_schema_fixture_abi debug_or_bringup",
            result.stdout,
        )
        self.assertIn("completion: wait runtime_stream_wait", result.stdout)
        self.assertNotIn("launch: module_kernel", result.stdout)

    def test_fake_tx_backend_constructs_tx_call_sequence(self) -> None:
        result = self.run_adapter(
            "--manifest",
            str(self.valid_manifest),
            "--backend",
            "fake-tx",
            "--strategy",
            "debug_kernel",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        expected_lines = [
            "backend: fake-tx",
            "strategy: debug_kernel tx_module_kernel",
            "txSetDevice device=0",
            "txMalloc name=lhs bytes=16",
            "txMemcpyH2D name=lhs bytes=16",
            "txMalloc name=rhs bytes=16",
            "txMemcpyH2D name=rhs bytes=16",
            "txMalloc name=out bytes=16",
            "txMalloc name=tmp bytes=16",
            "txMalloc name=rhs_resident bytes=16",
            "txMemcpyH2D name=rhs_resident bytes=16 source=rhs",
            "txModuleLoad artifact=schema_fixture.so",
            "txModuleGetFunction entrypoint=manifest_schema_fixture_abi",
            "txLaunchKernel strategy=debug_kernel entrypoint=manifest_schema_fixture_abi arg_bytes=80",
            "txStreamSynchronize completion_source=runtime_stream_wait",
            "txMemcpyD2H name=out bytes=16",
        ]
        for line in expected_lines:
            self.assertIn(line, result.stdout)

    def test_fake_tx_backend_rejects_descriptor_only_bpm_strategy(self) -> None:
        result = self.run_adapter(
            "--manifest",
            str(self.valid_manifest),
            "--backend",
            "fake-tx",
            "--strategy",
            "model_bpm",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "error: tx_model_bpm strategy model_bpm requires a materialized BPM descriptor",
            result.stderr,
        )

    def test_tx_backend_reports_missing_runtime_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_adapter(
                "--manifest",
                str(self.valid_manifest),
                "--backend",
                "tx",
                "--strategy",
                "debug_kernel",
                "--runtime-root",
                str(pathlib.Path(tmp) / "missing-runtime-root"),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error: tx runtime library was not found", result.stderr)

    def test_stub_completion_is_rejected_before_backend_selection(self) -> None:
        result = self.run_adapter(
            "--manifest",
            str(self.invalid_completion_manifest),
            "--backend",
            "dry-run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error: completion source is a known stub fence", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
