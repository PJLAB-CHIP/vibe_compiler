#!/usr/bin/env python3
"""Build the pinned-XLA SPMD partitioner helper for Wafer."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


BUILD_FILE = """\
load("@tsl//tsl/platform:rules_cc.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "wafer_xla_spmd_partitioner",
    srcs = ["wafer_xla_spmd_partitioner.cc"],
    deps = [
        "//xla:shape_util",
        "//xla:autotuning_proto_cc",
        "//xla:autotuning_proto_cc_impl",
        "//xla:xla_data_proto_cc",
        "//xla:xla_data_proto_cc_impl",
        "//xla:xla_proto_cc",
        "//xla:xla_proto_cc_impl",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_proto_cc_impl",
        "//xla/service:hlo_verifier",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:backend_configs_cc_impl",
        "//xla/service/spmd:spmd_partitioner",
        "//xla/service/spmd:spmd_prepare",
        "//xla/stream_executor:stream_executor_impl",
        "//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/profiler/lib:scoped_annotation",
        "@tsl//tsl/protobuf:dnn_proto_cc",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/platform:statusor",
    ],
)
"""


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _default_bazel() -> str:
    for candidate in (
        os.environ.get("BAZEL"),
        shutil.which("bazelisk"),
        shutil.which("bazel"),
        "/root/.cache/wafer-tools/bin/bazelisk",
    ):
        if candidate and pathlib.Path(candidate).exists():
            return str(candidate)
    return "bazelisk"


def _default_cc() -> str:
    for candidate in (
        os.environ.get("CC"),
        shutil.which("clang"),
        "/usr/bin/clang",
    ):
        if candidate and pathlib.Path(candidate).exists():
            return str(candidate)
    return ""


def _symlink(src: pathlib.Path, dst: pathlib.Path) -> None:
    target_is_directory = src.is_dir()
    os.symlink(src, dst, target_is_directory=target_is_directory)


def _populate_workspace(
    *,
    repo: pathlib.Path,
    xla_source: pathlib.Path,
    workspace: pathlib.Path,
) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    for child in xla_source.iterdir():
        if child.name.startswith("bazel-"):
            continue
        if child.name == "xla":
            continue
        _symlink(child, workspace / child.name)

    xla_dst = workspace / "xla"
    xla_dst.mkdir()
    for child in (xla_source / "xla").iterdir():
        if child.name == "wafer_tools":
            continue
        _symlink(child, xla_dst / child.name)

    helper_pkg = xla_dst / "wafer_tools"
    helper_pkg.mkdir()
    _symlink(
        repo / "lib" / "Wafer" / "Transforms" / "SPMD" / "XlaSpmdPartitionerMain.cpp",
        helper_pkg / "wafer_xla_spmd_partitioner.cc",
    )
    (helper_pkg / "BUILD.bazel").write_text(BUILD_FILE)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xla-source",
        type=pathlib.Path,
        default=repo / "third_party" / "xla",
        help="pinned XLA source workspace",
    )
    parser.add_argument(
        "--build-root",
        type=pathlib.Path,
        default=repo / "build" / "xla-spmd-helper",
        help="generated Bazel overlay and output directory",
    )
    parser.add_argument(
        "--bazel",
        default=_default_bazel(),
        help="bazel or bazelisk executable",
    )
    parser.add_argument(
        "--cc",
        default=_default_cc(),
        help="C/C++ compiler used by Bazel local_config_cc; defaults to clang when available",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="path for the copied helper binary",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="only materialize the overlay workspace",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo = _repo_root()
    xla_source = args.xla_source.resolve()
    build_root = args.build_root.resolve()
    workspace = build_root / "workspace"
    output = args.output or build_root / "wafer_xla_spmd_partitioner"

    if not (xla_source / "WORKSPACE").is_file():
        raise RuntimeError(f"not an XLA Bazel workspace: {xla_source}")

    _populate_workspace(repo=repo, xla_source=xla_source, workspace=workspace)
    if args.skip_build:
        print(workspace)
        return 0

    command = [args.bazel, "build"]
    if args.cc:
        command.extend(
            [
                f"--repo_env=CC={args.cc}",
                f"--repo_env=BAZEL_COMPILER={args.cc}",
                f"--action_env=CC={args.cc}",
            ]
        )
    command.append("//xla/wafer_tools:wafer_xla_spmd_partitioner")
    subprocess.run(
        command,
        cwd=workspace,
        check=True,
    )
    built = workspace / "bazel-bin" / "xla" / "wafer_tools" / "wafer_xla_spmd_partitioner"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, output)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"build_xla_spmd_partitioner_helper: {error}", file=sys.stderr)
        raise SystemExit(1)
