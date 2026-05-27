#!/usr/bin/env python3
"""Build and install torch_xla from the pinned third_party/pytorch-xla source."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSIONS_FILE = REPO_ROOT / "cmake" / "third_party" / "WaferDependencyVersions.cmake"


def run(command: list[str], *, cwd: pathlib.Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def load_versions() -> dict[str, str]:
    text = VERSIONS_FILE.read_text(encoding="utf-8")
    return {
        name: value
        for name, value in re.findall(r'set\((WAFER_[A-Z0-9_]+)\s+"([^"]+)"\)', text)
    }


def ensure_removed(path: pathlib.Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def symlink(target: pathlib.Path, link: pathlib.Path) -> None:
    ensure_removed(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def write_executable(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def get_python_layout(python: pathlib.Path) -> dict[str, str]:
    code = r"""
import json
import pathlib
import sysconfig
import sys
import torch
import torchgen

torch_dir = pathlib.Path(torch.__file__).resolve().parent
torchgen_dir = pathlib.Path(torchgen.__file__).resolve().parent
print(json.dumps({
    "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "site_packages": str(torch_dir.parent),
    "torch": str(torch_dir),
    "torchgen": str(torchgen_dir),
    "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
}))
"""
    return json.loads(capture([str(python), "-c", code]))


def materialize_torch_repo(torch_repo: pathlib.Path, layout: dict[str, str]) -> None:
    torch_dir = pathlib.Path(layout["torch"])
    torchgen_dir = pathlib.Path(layout["torchgen"])
    torchgen_aten = torchgen_dir / "packaged" / "ATen"
    shape_inference = torch_dir / "include" / "torch" / "csrc" / "lazy" / "core" / "shape_inference.h"

    if not (torch_dir / "include").is_dir():
        raise RuntimeError(f"torch headers not found under {torch_dir / 'include'}")
    if not (torch_dir / "lib" / "libtorch.so").is_file():
        raise RuntimeError(f"torch libraries not found under {torch_dir / 'lib'}")
    if not torchgen_aten.is_dir():
        raise RuntimeError(f"torchgen packaged ATen files not found under {torchgen_aten}")
    if not shape_inference.is_file():
        raise RuntimeError(f"torch lazy shape_inference.h not found at {shape_inference}")

    ensure_removed(torch_repo)
    torch_repo.mkdir(parents=True)
    (torch_repo / "WORKSPACE").write_text('workspace(name = "torch")\n', encoding="utf-8")
    (torch_repo / "BUILD").write_text(
        """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["torch/include/**/*.h"], exclude = ["torch/include/google/protobuf/**/*.h"]),
    strip_include_prefix = "torch/include",
)

cc_library(
    name = "runtime_headers",
    hdrs = glob(["torch/include/torch/csrc/api/include/**/*.h"]),
    strip_include_prefix = "torch/include/torch/csrc/api/include",
)

filegroup(
    name = "torchgen_deps",
    srcs = glob([
        "aten/src/ATen/**/*",
        "torch/csrc/lazy/core/shape_inference.h",
    ]),
)

cc_import(
    name = "libtorch",
    shared_library = "build/lib/libtorch.so",
)

cc_import(
    name = "libtorch_cpu",
    shared_library = "build/lib/libtorch_cpu.so",
)

cc_import(
    name = "libtorch_python",
    shared_library = "build/lib/libtorch_python.so",
)

cc_import(
    name = "libc10",
    shared_library = "build/lib/libc10.so",
)
""".lstrip(),
        encoding="utf-8",
    )
    symlink(torch_dir / "include", torch_repo / "torch" / "include")
    symlink(torch_dir / "lib", torch_repo / "build" / "lib")
    symlink(torchgen_aten, torch_repo / "aten" / "src" / "ATen")
    symlink(shape_inference, torch_repo / "torch" / "csrc" / "lazy" / "core" / "shape_inference.h")


def materialize_pypi_torch_repo(pypi_torch_repo: pathlib.Path, layout: dict[str, str]) -> None:
    site_packages = pathlib.Path(layout["site_packages"])
    ensure_removed(pypi_torch_repo)
    pypi_torch_repo.mkdir(parents=True)
    (pypi_torch_repo / "WORKSPACE").write_text('workspace(name = "pypi_torch")\n', encoding="utf-8")
    (pypi_torch_repo / "BUILD").write_text(
        """
package(default_visibility = ["//visibility:public"])

py_library(
    name = "pkg",
    srcs = [],
    data = glob([
        "site-packages/torchgen/**",
        "site-packages/typing_extensions.py",
        "site-packages/typing_extensions-*.dist-info/**",
    ]),
    imports = ["site-packages"],
)
""".lstrip(),
        encoding="utf-8",
    )
    symlink(site_packages, pypi_torch_repo / "site-packages")


def materialize_llvm_raw_repo(llvm_raw_repo: pathlib.Path) -> None:
    llvm_source = REPO_ROOT / "third_party" / "llvm-project"
    if not (llvm_source / "utils" / "bazel" / "configure.bzl").is_file():
        raise RuntimeError(f"LLVM Bazel configure.bzl not found under {llvm_source}")

    ensure_removed(llvm_raw_repo)
    llvm_raw_repo.mkdir(parents=True)
    (llvm_raw_repo / "WORKSPACE").write_text('workspace(name = "llvm-raw")\n', encoding="utf-8")
    (llvm_raw_repo / "BUILD").write_text('exports_files(["WORKSPACE"])\n', encoding="utf-8")
    for child in llvm_source.iterdir():
        if child.name == ".git":
            continue
        symlink(child, llvm_raw_repo / child.name)


def materialize_empty_cc_repo(repo: pathlib.Path, repo_name: str, target_name: str) -> None:
    ensure_removed(repo)
    repo.mkdir(parents=True)
    (repo / "WORKSPACE").write_text(f'workspace(name = "{repo_name}")\n', encoding="utf-8")
    (repo / "BUILD").write_text(
        f"""
package(default_visibility = ["//visibility:public"])

cc_library(name = "{target_name}")
""".lstrip(),
        encoding="utf-8",
    )


def copy_built_extensions(pytorch_xla: pathlib.Path, layout: dict[str, str]) -> None:
    build_libs = sorted(
        pytorch_xla.glob("build/lib.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not build_libs:
        raise RuntimeError(f"no build/lib.* directory found under {pytorch_xla}")
    build_lib = build_libs[0]
    ext_suffix = layout["ext_suffix"]
    for name in ["_XLAC", "_XLAC_cuda_functions"]:
        source = build_lib / f"{name}{ext_suffix}"
        if not source.is_file():
            raise RuntimeError(f"built extension not found: {source}")
        shutil.copy2(source, pytorch_xla / source.name)


def materialize_bazel_wrapper(
    wrapper: pathlib.Path,
    bazel: pathlib.Path,
    torch_repo: pathlib.Path,
    pypi_torch_repo: pathlib.Path,
    llvm_raw_repo: pathlib.Path,
    llvm_zlib_repo: pathlib.Path,
    llvm_zstd_repo: pathlib.Path,
) -> None:
    write_executable(
        wrapper,
        f"""#!/usr/bin/env bash
set -euo pipefail
real_bazel={str(bazel)!r}
repo_root={str(REPO_ROOT)!r}
torch_repo={str(torch_repo)!r}
pypi_torch_repo={str(pypi_torch_repo)!r}
llvm_raw_repo={str(llvm_raw_repo)!r}
llvm_zlib_repo={str(llvm_zlib_repo)!r}
llvm_zstd_repo={str(llvm_zstd_repo)!r}

case "${{1:-}}" in
  build|test|run|query|cquery|aquery|coverage)
    exec "$real_bazel" "$@" \\
      "--override_repository=xla=$repo_root/third_party/xla" \\
      "--override_repository=llvm-raw=$llvm_raw_repo" \\
      "--override_repository=llvm_zlib=$llvm_zlib_repo" \\
      "--override_repository=llvm_zstd=$llvm_zstd_repo" \\
      "--override_repository=stablehlo=$repo_root/third_party/stablehlo" \\
      "--override_repository=torch=$torch_repo" \\
      "--override_repository=pypi_torch=$pypi_torch_repo"
    ;;
  *)
    exec "$real_bazel" "$@"
    ;;
esac
""",
    )


def parse_args() -> argparse.Namespace:
    default_python = REPO_ROOT / "third_party" / "python-importer-py311" / "bin" / "python"
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=pathlib.Path, default=default_python)
    parser.add_argument("--bazel", type=pathlib.Path, default=REPO_ROOT / "third_party" / "tools" / "bazel")
    parser.add_argument("--build-root", type=pathlib.Path, default=REPO_ROOT / "build" / "pytorch-xla-source-build")
    parser.add_argument("--jobs", default=os.getenv("BAZEL_JOBS", "8"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = args.python.resolve()
    bazel = args.bazel.resolve()
    build_root = args.build_root.resolve()
    pytorch_xla = REPO_ROOT / "third_party" / "pytorch-xla"

    if not python.exists():
        raise RuntimeError(f"importer Python not found: {python}")
    if not bazel.exists():
        raise RuntimeError(f"Bazel not found: {bazel}")
    if not pytorch_xla.is_dir():
        raise RuntimeError(f"PyTorch/XLA source checkout not found: {pytorch_xla}")

    versions = load_versions()
    layout = get_python_layout(python)
    torch_repo = build_root / "torch-python-bazel-repo"
    pypi_torch_repo = build_root / "pypi-torch-bazel-repo"
    llvm_raw_repo = build_root / "llvm-raw-bazel-repo"
    llvm_zlib_repo = build_root / "llvm-zlib-empty-bazel-repo"
    llvm_zstd_repo = build_root / "llvm-zstd-empty-bazel-repo"
    wrapper = build_root / "bin" / "bazel"

    materialize_torch_repo(torch_repo, layout)
    materialize_pypi_torch_repo(pypi_torch_repo, layout)
    materialize_llvm_raw_repo(llvm_raw_repo)
    materialize_empty_cc_repo(llvm_zlib_repo, "llvm_zlib", "zlib")
    materialize_empty_cc_repo(llvm_zstd_repo, "llvm_zstd", "zstd")
    materialize_bazel_wrapper(
        wrapper,
        bazel,
        torch_repo,
        pypi_torch_repo,
        llvm_raw_repo,
        llvm_zlib_repo,
        llvm_zstd_repo,
    )

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [
            str(python.parent),
            str(wrapper.parent),
            str(bazel.parent),
            env.get("PATH", ""),
        ]
    )
    env["BAZEL_JOBS"] = args.jobs
    env["BUILD_CPP_TESTS"] = "0"
    env["BUNDLE_LIBTPU"] = "0"
    env["GIT_VERSIONED_XLA_BUILD"] = "0"
    env["HERMETIC_PYTHON_VERSION"] = layout["major_minor"]
    env["TORCH_XLA_VERSION"] = versions["WAFER_TORCH_XLA_PYTHON_VERSION"]

    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "--no-build-isolation",
            "--no-deps",
            "--config-settings",
            "editable_mode=compat",
        ],
        cwd=pytorch_xla,
        env=env,
    )
    copy_built_extensions(pytorch_xla, layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
