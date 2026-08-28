#!/usr/bin/env python3
"""Tests that the Python runtime adapter delegates all acceptance to C++."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--wafer-run", required=True)
    return parser.parse_args()


class WaferRuntimeAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        args = parse_args()
        cls.repo_root = pathlib.Path(args.repo_root)
        cls.wafer_run = pathlib.Path(args.wafer_run)
        cls.tool = cls.repo_root / "tools" / "wafer_runtime_adapter.py"
        cls.manifest = (
            cls.repo_root
            / "test"
            / "Runtime"
            / "Inputs"
            / "typed-package-manifest.json"
        )

    def make_package(self, root: pathlib.Path) -> pathlib.Path:
        package = root / "package"
        (package / "modules").mkdir(parents=True)
        (package / "data").mkdir()
        (package / "modules" / "tile_00000.so").touch()
        (package / "data" / "program-data.bin").touch()
        shutil.copyfile(self.manifest, package / "manifest.json")
        return package

    def run_adapter(
        self, package: pathlib.Path, *extra_args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(self.tool),
                "--wafer-run",
                str(self.wafer_run),
                "--package-dir",
                str(package),
                *extra_args,
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_delegates_verified_no_card_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self.make_package(pathlib.Path(temporary))
            result = self.run_adapter(package, "--no-card")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("package: id=0 cards=1 tiles=16", result.stdout)
        self.assertIn(
            "target_identity: wafer-tx81-single-card", result.stdout
        )
        self.assertIn("board_execution: false", result.stdout)

    def test_cpp_rejection_is_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self.make_package(pathlib.Path(temporary))
            result = self.run_adapter(
                package, "--no-card", "--max-resource-bytes", "32"
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "runtime environment invocation capacity is insufficient",
            result.stderr,
        )

    def test_adapter_does_not_offer_alternate_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self.make_package(pathlib.Path(temporary))
            result = self.run_adapter(package, "--backend", "fake-tx")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
