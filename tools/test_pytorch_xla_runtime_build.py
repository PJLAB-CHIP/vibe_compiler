#!/usr/bin/env python3
"""Fast contract tests for the source-built PyTorch/XLA build entrypoint."""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest import mock

import build_pytorch_xla_runtime
import bootstrap_deps


class ImporterPythonPathTest(unittest.TestCase):
    def test_default_matches_bootstrap_importer_environment(self) -> None:
        with mock.patch.object(sys, "argv", ["build_pytorch_xla_runtime.py"]):
            args = build_pytorch_xla_runtime.parse_args()
        self.assertEqual(
            args.python,
            build_pytorch_xla_runtime.REPO_ROOT
            / "third_party"
            / "python-importer"
            / "bin"
            / "python",
        )

    def test_absolute_path_preserves_virtualenv_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = pathlib.Path(temporary_directory) / "importer"
            venv.EnvBuilder(with_pip=False).create(environment)
            interpreter = environment / "bin" / "python"
            if not interpreter.is_symlink():
                interpreter.unlink()
                interpreter.symlink_to(pathlib.Path(sys.executable).resolve())

            command_path = (
                build_pytorch_xla_runtime.make_absolute_without_resolving(interpreter)
            )

            self.assertEqual(command_path, interpreter)
            self.assertNotEqual(command_path, interpreter.resolve())
            actual_prefix = subprocess.check_output(
                [str(command_path), "-c", "import sys; print(sys.prefix)"],
                text=True,
            ).strip()
            self.assertEqual(pathlib.Path(actual_prefix), environment)

    def test_upstream_gcc10_toolchain_is_preferred_when_available(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(
            build_pytorch_xla_runtime.shutil,
            "which",
            side_effect=lambda command: (
                f"/usr/bin/{command}" if command in {"gcc-10", "g++-10"} else None
            ),
        ), mock.patch.object(sys, "argv", ["build_pytorch_xla_runtime.py"]):
            args = build_pytorch_xla_runtime.parse_args()
        self.assertEqual(args.cc, pathlib.Path("gcc-10"))
        self.assertEqual(args.cxx, pathlib.Path("g++-10"))

    def test_compiler_wrapper_uses_selected_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            compiler = root / "selected-cxx"
            compiler.write_text("#!/bin/sh\nprintf '%s\\n' selected \"$@\"\n")
            compiler.chmod(0o755)
            wrapper = root / "g++"

            build_pytorch_xla_runtime.materialize_compiler_wrapper(
                wrapper, compiler
            )

            output = subprocess.check_output(
                [str(wrapper), "-std=c++17"], text=True
            ).splitlines()
            self.assertEqual(output, ["selected", "-std=c++17"])

    def test_bazel_wrapper_overrides_repository_and_action_compilers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            bazel = root / "bazel-real"
            bazel.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            bazel.chmod(0o755)
            wrapper = root / "bazel"
            cc = pathlib.Path("/toolchain/gcc")
            cxx = pathlib.Path("/toolchain/g++")

            build_pytorch_xla_runtime.materialize_bazel_wrapper(
                wrapper,
                bazel,
                cc,
                cxx,
                root / "torch",
                root / "pypi-torch",
                root / "llvm",
                root / "zlib",
                root / "zstd",
            )

            arguments = subprocess.check_output(
                [str(wrapper), "build", "//:runtime"], text=True
            ).splitlines()
            self.assertIn("--repo_env=CC=/toolchain/gcc", arguments)
            self.assertIn("--repo_env=CXX=/toolchain/g++", arguments)
            self.assertIn("--action_env=CC=/toolchain/gcc", arguments)
            self.assertIn("--action_env=CXX=/toolchain/g++", arguments)
            info_arguments = subprocess.check_output(
                [str(wrapper), "info", "bazel-bin"], text=True
            ).splitlines()
            self.assertIn(
                f"--override_repository=xla={build_pytorch_xla_runtime.REPO_ROOT}/third_party/xla",
                info_arguments,
            )

    def test_move_contract_probe_rejects_incompatible_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            compiler = pathlib.Path(temporary_directory) / "incompatible-cxx"
            compiler.write_text("#!/bin/sh\nexit 1\n")
            compiler.chmod(0o755)

            with self.assertRaisesRegex(RuntimeError, "select a newer GCC"):
                build_pytorch_xla_runtime.verify_cxx_move_support(compiler)

    def test_extensions_are_installed_from_persistent_bazel_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            source = root / "source"
            bazel_bin = root / "bazel-bin"
            source.mkdir()
            bazel_bin.mkdir()
            for name in ("_XLAC", "_XLAC_cuda_functions"):
                (bazel_bin / f"{name}.so").write_bytes(
                    f"persistent {name}\n".encode()
                )

            build_pytorch_xla_runtime.copy_built_extensions(
                source, {"ext_suffix": ".test.so"}, bazel_bin
            )

            self.assertEqual(
                (source / "_XLAC.test.so").read_bytes(),
                b"persistent _XLAC\n",
            )
            self.assertEqual(
                (source / "_XLAC_cuda_functions.test.so").read_bytes(),
                b"persistent _XLAC_cuda_functions\n",
            )

    def test_importer_bazel_is_checksum_pinned_and_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            payload = root / "bazel-download"
            payload.write_bytes(b"fixture bazel executable\n")
            versions = {
                "WAFER_BAZEL_VERSION": "6.5.0",
                "WAFER_BAZEL_LINUX_X64_URL": payload.as_uri(),
                "WAFER_BAZEL_LINUX_X64_SHA256": hashlib.sha256(
                    payload.read_bytes()
                ).hexdigest(),
            }
            prefix = root / "managed"

            default = bootstrap_deps.fetch_importer_bazel(versions, prefix)

            self.assertTrue(default.is_symlink())
            binary = default.resolve()
            self.assertEqual(binary.read_bytes(), payload.read_bytes())
            self.assertTrue(binary.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
