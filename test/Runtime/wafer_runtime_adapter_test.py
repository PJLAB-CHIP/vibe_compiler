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
        self.assertIn(
            "device_code: schema_fixture_kernel kcore_shared_object schema_fixture.so",
            result.stdout,
        )
        self.assertIn("allocation: input lhs 16 bytes host_visible", result.stdout)
        self.assertIn("allocation: input rhs 16 bytes host_visible", result.stdout)
        self.assertIn("allocation: output out 16 bytes host_visible", result.stdout)
        self.assertIn("allocation: workspace tmp 16 bytes", result.stdout)
        self.assertIn(
            "launch: module_kernel entrypoint=manifest_schema_fixture_abi "
            "grid=(1,1,1) block=(1,1,1)",
            result.stdout,
        )
        self.assertIn("completion: wait runtime_stream_wait", result.stdout)

    def test_fake_tx_backend_constructs_tx_call_sequence(self) -> None:
        result = self.run_adapter(
            "--manifest", str(self.valid_manifest), "--backend", "fake-tx"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        expected_lines = [
            "backend: fake-tx",
            "txSetDevice device=0",
            "txMalloc name=lhs bytes=16",
            "txMemcpyH2D name=lhs bytes=16",
            "txMalloc name=rhs bytes=16",
            "txMemcpyH2D name=rhs bytes=16",
            "txMalloc name=out bytes=16",
            "txMalloc name=tmp bytes=16",
            "txModuleLoad artifact=schema_fixture.so",
            "txModuleGetFunction entrypoint=manifest_schema_fixture_abi",
            "txLaunchKernel entrypoint=manifest_schema_fixture_abi arg_bytes=80",
            "txStreamSynchronize completion_source=runtime_stream_wait",
            "txMemcpyD2H name=out bytes=16",
        ]
        for line in expected_lines:
            self.assertIn(line, result.stdout)

    def test_tx_backend_reports_missing_runtime_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_adapter(
                "--manifest",
                str(self.valid_manifest),
                "--backend",
                "tx",
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
