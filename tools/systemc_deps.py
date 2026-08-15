#!/usr/bin/env python3
"""Managed SystemC identity and conformance record for the event-model lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
from typing import Any

from numeric_deps import sha256_archive_tree, sha256_tree


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSIONS_FILE = REPO_ROOT / "cmake" / "third_party" / "WaferDependencyVersions.cmake"
RECORD_KIND = "wafer-systemc-model-deps"
RECORD_STATUS = "complete"
SNAPSHOT_KIND = "wafer-systemc-model-canonical-snapshot"
SOURCE_DIRECTORY_PREFIX = "systemc-"
BUILD_OPTIONS = {
    "BUILD_SHARED_LIBS": "OFF",
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_CXX_EXTENSIONS": "OFF",
    "CMAKE_CXX_STANDARD": "17",
    "CMAKE_CXX_STANDARD_REQUIRED": "ON",
    "CMAKE_INSTALL_LIBDIR": "lib",
    "DISABLE_COPYRIGHT_MESSAGE": "ON",
    "ENABLE_ASSERTIONS": "ON",
    "ENABLE_EXAMPLES": "OFF",
    "ENABLE_PTHREADS": "OFF",
    "ENABLE_REGRESSION": "OFF",
    "ENABLE_STD_THREADS": "OFF",
    "INSTALL_TO_LIB_BUILD_TYPE_DIR": "OFF",
    "INSTALL_TO_LIB_TARGET_ARCH_DIR": "OFF",
    "SYSTEMC_UNITY_BUILD": "OFF",
}
REQUIRED_FILES = {
    "systemc-archive",
    "systemc-library",
    "systemc-header",
    "tlm-header",
    "cmake-config",
    "cmake-config-version",
    "cmake-targets",
    "license",
    "notice",
    "delta-event-smoke",
}
REQUIRED_GATES = {
    "configure",
    "build-install",
    "consumer-configure",
    "consumer-build",
    "delta-event-run",
}


def load_versions() -> dict[str, str]:
    text = VERSIONS_FILE.read_text(encoding="utf-8")
    return dict(re.findall(r'set\((WAFER_[A-Z0-9_]+)\s+"([^"]+)"\)', text))


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlink_path(path: pathlib.Path, *, allow_missing: bool = False) -> pathlib.Path:
    absolute = pathlib.Path(os.path.abspath(path))
    current = pathlib.Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                continue
            raise RuntimeError(f"managed SystemC path component is missing: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"managed SystemC path contains a symlink: {current}")
    return absolute


def _regular_file(path: pathlib.Path) -> pathlib.Path:
    path = reject_symlink_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise RuntimeError(f"managed SystemC file is not a regular file: {path}")
    return path


def _closed_object(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be an object")
    actual = set(value)
    if actual != keys:
        raise RuntimeError(
            f"{context} fields mismatch: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return value


def _relative_file(root: pathlib.Path, record: dict[str, Any], name: str) -> pathlib.Path:
    entry = _closed_object(
        record["artifacts"][name], {"path", "sha256", "size"}, f"file {name}"
    )
    relative = entry["path"]
    if (
        not isinstance(relative, str)
        or not relative
        or pathlib.PurePosixPath(relative).is_absolute()
    ):
        raise RuntimeError(f"file {name} path must be a nonempty relative path")
    path = _regular_file(root / pathlib.PurePosixPath(relative))
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"file {name} escapes the managed root") from error
    if entry["sha256"] != sha256_file(path) or entry["size"] != path.stat().st_size:
        raise RuntimeError(f"file {name} identity mismatch")
    return path


def validate_record(
    record_path: pathlib.Path, root: pathlib.Path, versions: dict[str, str]
) -> dict[str, pathlib.Path]:
    root = reject_symlink_path(root)
    if not root.is_dir():
        raise RuntimeError(f"managed SystemC root is not a directory: {root}")
    record_path = _regular_file(record_path)
    try:
        record_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("SystemC dependency record must be inside its managed root") from error
    raw = record_path.read_bytes()
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"SystemC dependency record is invalid JSON: {error}") from error
    record = _closed_object(
        record,
        {
            "kind",
            "status",
            "dependency",
            "build",
            "toolchain",
            "artifacts",
            "licenses",
            "conformance",
        },
        "SystemC dependency record",
    )
    if raw != canonical_json(record).encode("utf-8"):
        raise RuntimeError("SystemC dependency record is not canonical JSON")
    if record["kind"] != RECORD_KIND or record["status"] != RECORD_STATUS:
        raise RuntimeError("SystemC dependency record kind/status mismatch")

    dependency = _closed_object(
        record["dependency"],
        {"name", "version", "commit", "url", "archive_sha256", "source_tree_sha256"},
        "SystemC dependency identity",
    )
    expected_dependency = {
        "name": "Accellera SystemC",
        "version": versions["WAFER_SYSTEMC_VERSION"],
        "commit": versions["WAFER_SYSTEMC_COMMIT"],
        "url": versions["WAFER_SYSTEMC_URL"],
        "archive_sha256": versions["WAFER_SYSTEMC_SHA256"],
    }
    for key, expected in expected_dependency.items():
        if dependency[key] != expected:
            raise RuntimeError(f"SystemC dependency {key} mismatch")
    if not isinstance(dependency["source_tree_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", dependency["source_tree_sha256"]
    ):
        raise RuntimeError("SystemC dependency source tree digest is invalid")

    archive = _relative_file(root, record, "systemc-archive")
    expected_archive_name = f"systemc-{versions['WAFER_SYSTEMC_VERSION']}.tar.gz"
    if archive.name != expected_archive_name or sha256_file(archive) != versions["WAFER_SYSTEMC_SHA256"]:
        raise RuntimeError("SystemC dependency source archive mismatch")
    source_top = SOURCE_DIRECTORY_PREFIX + versions["WAFER_SYSTEMC_VERSION"]
    source = reject_symlink_path(root / "sources" / source_top)
    if not source.is_dir():
        raise RuntimeError("SystemC dependency source tree is missing")
    archive_tree_digest = sha256_archive_tree(archive, source_top)
    if dependency["source_tree_sha256"] != archive_tree_digest:
        raise RuntimeError("SystemC source-tree record is not derived from pinned archive")
    if sha256_tree(source) != archive_tree_digest:
        raise RuntimeError("SystemC dependency extracted source tree mismatch")

    build = _closed_object(
        record["build"],
        {"options", "generator", "install_prefix", "install_tree_sha256"},
        "SystemC build",
    )
    if build["options"] != BUILD_OPTIONS:
        raise RuntimeError("SystemC dependency build options mismatch")
    if build["install_prefix"] != "install/systemc":
        raise RuntimeError("SystemC dependency install prefix mismatch")
    if not isinstance(build["generator"], str) or not build["generator"]:
        raise RuntimeError("SystemC dependency generator is missing")
    if not isinstance(build["install_tree_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", build["install_tree_sha256"]
    ):
        raise RuntimeError("SystemC dependency install tree digest is invalid")

    toolchain = _closed_object(record["toolchain"], {"cmake", "cxx"}, "toolchain")
    for name, identity in toolchain.items():
        identity = _closed_object(identity, {"path", "version", "sha256"}, f"toolchain {name}")
        if not all(isinstance(identity[key], str) and identity[key] for key in identity):
            raise RuntimeError(f"toolchain {name} identity is incomplete")

    files = record["artifacts"]
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
        raise RuntimeError("SystemC dependency file closure mismatch")
    resolved = {name: _relative_file(root, record, name) for name in sorted(files)}

    licenses = _closed_object(record["licenses"], {"license", "notice"}, "licenses")
    if licenses != {
        "license": {"artifact": "license", "spdx": "Apache-2.0"},
        "notice": {"artifact": "notice", "spdx": "Apache-2.0"},
    }:
        raise RuntimeError("SystemC dependency license closure mismatch")

    gates = _closed_object(record["conformance"], {"gates"}, "conformance")["gates"]
    if (
        not isinstance(gates, list)
        or len(gates) != len(REQUIRED_GATES)
        or {gate.get("name") for gate in gates if isinstance(gate, dict)} != REQUIRED_GATES
    ):
        raise RuntimeError("SystemC dependency conformance gate closure mismatch")
    seen_logs: set[pathlib.Path] = set()
    for gate in gates:
        gate = _closed_object(
            gate, {"name", "command", "log", "log_sha256", "exit_code"}, "conformance gate"
        )
        if (
            gate["exit_code"] != 0
            or not isinstance(gate["command"], list)
            or not gate["command"]
            or not all(isinstance(item, str) and item for item in gate["command"])
        ):
            raise RuntimeError(f"SystemC dependency gate did not complete: {gate['name']}")
        relative_log = gate["log"]
        if (
            not isinstance(relative_log, str)
            or not relative_log
            or pathlib.PurePosixPath(relative_log).is_absolute()
        ):
            raise RuntimeError(f"SystemC dependency gate log path is invalid: {gate['name']}")
        log = _regular_file(root / pathlib.PurePosixPath(relative_log))
        try:
            log.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"SystemC dependency gate log escapes the managed root: {gate['name']}") from error
        if log in seen_logs:
            raise RuntimeError("SystemC dependency gates reuse one log file")
        seen_logs.add(log)
        if (
            not isinstance(gate["log_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", gate["log_sha256"])
            or sha256_file(log) != gate["log_sha256"]
        ):
            raise RuntimeError(f"SystemC dependency gate log identity is invalid: {gate['name']}")
    install_prefix = reject_symlink_path(root / "install" / "systemc")
    if not install_prefix.is_dir() or sha256_tree(install_prefix) != build["install_tree_sha256"]:
        raise RuntimeError("SystemC dependency install tree mismatch")
    return resolved


def make_snapshot(
    record_path: pathlib.Path,
    root: pathlib.Path,
    versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = reject_symlink_path(root)
    versions = load_versions() if versions is None else versions
    files = validate_record(record_path, root, versions)
    record_path = _regular_file(record_path)
    return {
        "kind": SNAPSHOT_KIND,
        "root": root.as_posix(),
        "record": record_path.as_posix(),
        "record_sha256": sha256_file(record_path),
        "version": versions["WAFER_SYSTEMC_VERSION"],
        "commit": versions["WAFER_SYSTEMC_COMMIT"],
        "artifacts": {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in sorted(files.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--record", required=True, type=pathlib.Path)
    parser.add_argument("--emit-cmake-snapshot", action="store_true")
    args = parser.parse_args()
    try:
        snapshot = make_snapshot(args.record.absolute(), args.root.absolute())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.emit_cmake_snapshot:
        sys.stdout.write(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))
        sys.stdout.write("\n")
    else:
        print(f"SystemC dependency record valid: {snapshot['record_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
