#!/usr/bin/env python3
"""Shared source and identity helpers for managed numeric-model dependencies.

This module deliberately has no network entry point.  ``bootstrap_deps.py``
owns explicit downloads and clean builds; CMake and ``check_deps.py`` consume
the resulting fail-closed conformance record.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import tarfile
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

import fcntl


RECORD_SCHEMA_VERSION = 2
RECORD_KIND = "wafer-numeric-model-deps"
RECORD_STATUS = "conformance-passed"
CONFORMANCE_POLICY = "wafer-numeric-model-conformance-v1"
SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "wafer-numeric-model-canonical-snapshot"
TOOLCHAIN_POLICY = "wafer-host-numeric-build-environment-v1"
NUMERIC_ROOT_TOKEN = "${NUMERIC_ROOT}"

SOFTFLOAT_PLATFORM = "Linux-x86_64-GCC"
SOFTFLOAT_SPECIALIZATION = "ARM-VFPv2-defaultNaN"
SOFTFLOAT_THREAD_LOCAL = "_Thread_local"
SOFTFLOAT_RAISE_FLAGS = "non-trapping"

CONFIGURE_OPTIONS = {
    "m4": ["--disable-dependency-tracking"],
    "gmp": [
        "--enable-shared",
        "--disable-static",
        "--with-pic",
    ],
    "mpfr": [
        "--enable-shared",
        "--disable-static",
        "--enable-thread-safe",
        "--disable-dependency-tracking",
        "--with-gmp=${GMP_PREFIX}",
    ],
}

LINKAGE_POLICY = {
    "softfloat": "static",
    "gmp": "shared-only",
    "mpfr": "shared-only",
}

REQUIRED_ARTIFACTS = {
    "m4",
    "softfloat",
    "softfloat-header",
    "softfloat-types-header",
    "testsoftfloat",
    "gmp",
    "gmp-soname",
    "gmp-header",
    "mpfr",
    "mpfr-soname",
    "mpfr-header",
    "license-softfloat",
    "license-testfloat",
    "license-m4",
    "license-gmp-copying",
    "license-gmp-gpl-v2",
    "license-gmp-gpl-v3",
    "license-gmp-lgpl-v3",
    "license-mpfr-copying",
    "license-mpfr-lesser",
}

REQUIRED_CONFORMANCE_GATES = {
    "m4-configure",
    "m4-build",
    "m4-check",
    "m4-install",
    "m4-version",
    "gmp-configure",
    "gmp-build",
    "gmp-check",
    "gmp-install",
    "mpfr-configure",
    "mpfr-build",
    "mpfr-check",
    "mpfr-install",
    "softfloat-build",
    "testfloat-build",
    "softfloat-policy-compile",
    "softfloat-tls-default-nan",
    "testsoftfloat-f16-mulAdd",
    "testsoftfloat-f32-mulAdd",
    "testsoftfloat-all1",
    "testsoftfloat-all2",
    "managed-version-thread-safe-compile",
    "managed-version-thread-safe",
}

TOOL_EXECUTABLE_NAMES = {
    "cc": "gcc",
    "cxx": "g++",
    "make": "make",
    "ar": "ar",
    "ranlib": "ranlib",
    "nm": "nm",
    "ld": "ld",
    "readelf": "readelf",
    "shell": "sh",
    "false": "false",
}

LICENSE_SOURCE_FILES = {
    "license-softfloat": ("softfloat", "COPYING.txt"),
    "license-testfloat": ("testfloat", "COPYING.txt"),
    "license-m4": ("m4", "COPYING"),
    "license-gmp-copying": ("gmp", "COPYING"),
    "license-gmp-gpl-v2": ("gmp", "COPYINGv2"),
    "license-gmp-gpl-v3": ("gmp", "COPYINGv3"),
    "license-gmp-lgpl-v3": ("gmp", "COPYING.LESSERv3"),
    "license-mpfr-copying": ("mpfr", "COPYING"),
    "license-mpfr-lesser": ("mpfr", "COPYING.LESSER"),
}


@dataclass(frozen=True)
class NumericPin:
    name: str
    version: str
    url: str
    sha256: str
    archive_name: str
    source_directory: str


@dataclass(frozen=True)
class ValidatedNumericRecord:
    record: dict[str, Any]
    record_sha256: str
    root: pathlib.Path
    record_path: pathlib.Path
    artifacts: dict[str, pathlib.Path]

    def cmake_snapshot(self) -> dict[str, Any]:
        artifact_records = self.record["artifacts"]
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "kind": SNAPSHOT_KIND,
            "record_sha256": self.record_sha256,
            "root": self.root.as_posix(),
            "record": self.record_path.as_posix(),
            "artifacts": {
                name: {
                    **identity,
                    "path": self.artifacts[name].as_posix(),
                }
                for name, identity in sorted(artifact_records.items())
            },
        }


_PIN_FIELDS = {
    "softfloat": (
        "WAFER_SOFTFLOAT_VERSION",
        "WAFER_SOFTFLOAT_URL",
        "WAFER_SOFTFLOAT_SHA256",
        "SoftFloat-3e",
    ),
    "testfloat": (
        "WAFER_TESTFLOAT_VERSION",
        "WAFER_TESTFLOAT_URL",
        "WAFER_TESTFLOAT_SHA256",
        "TestFloat-3e",
    ),
    "m4": (
        "WAFER_M4_VERSION",
        "WAFER_M4_URL",
        "WAFER_M4_SHA256",
        "m4-1.4.21",
    ),
    "gmp": (
        "WAFER_GMP_VERSION",
        "WAFER_GMP_URL",
        "WAFER_GMP_SHA256",
        "gmp-6.3.0",
    ),
    "mpfr": (
        "WAFER_MPFR_VERSION",
        "WAFER_MPFR_URL",
        "WAFER_MPFR_SHA256",
        "mpfr-4.2.2",
    ),
}


def numeric_pins(versions: dict[str, str]) -> dict[str, NumericPin]:
    pins: dict[str, NumericPin] = {}
    for name, (version_key, url_key, sha_key, source_directory) in _PIN_FIELDS.items():
        missing = [key for key in (version_key, url_key, sha_key) if key not in versions]
        if missing:
            raise RuntimeError(
                f"missing numeric dependency pin(s) for {name}: {', '.join(missing)}"
            )
        url = versions[url_key]
        archive_name = pathlib.PurePosixPath(urlparse(url).path).name
        if not archive_name:
            raise RuntimeError(f"numeric dependency URL has no archive name: {url}")
        digest = versions[sha_key]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError(f"invalid SHA256 pin for {name}: {digest}")
        pins[name] = NumericPin(
            name=name,
            version=versions[version_key],
            url=url,
            sha256=digest,
            archive_name=archive_name,
            source_directory=source_directory,
        )
    return pins


def sha256_file(path: pathlib.Path) -> str:
    return _regular_file_snapshot(path)["sha256"]


def _absolute_path(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(path))


def reject_symlink_ancestors(
    path: pathlib.Path, *, allow_missing: bool = False
) -> pathlib.Path:
    """Reject every symlink component in an absolute filesystem path."""

    absolute = _absolute_path(path)
    current = pathlib.Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                return absolute
            raise RuntimeError(f"managed path component does not exist: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"managed path contains a symlink component: {current}")
    return absolute


def prepare_managed_directory(path: pathlib.Path) -> pathlib.Path:
    absolute = reject_symlink_ancestors(path, allow_missing=True)
    absolute.mkdir(parents=True, exist_ok=True)
    reject_symlink_ancestors(absolute)
    if not absolute.is_dir():
        raise RuntimeError(f"managed path is not a directory: {absolute}")
    return absolute


def _assert_managed_subpath(root: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    root = reject_symlink_ancestors(root)
    path = reject_symlink_ancestors(path, allow_missing=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"path escapes managed numeric root: {path}") from error
    return path


@contextmanager
def numeric_build_lock(root: pathlib.Path) -> Iterator[None]:
    root = prepare_managed_directory(root)
    lock_path = _assert_managed_subpath(root, root / ".numeric-model-deps.lock")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"managed numeric-model root is already being modified: {root}"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _open_regular_file(path: pathlib.Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"identity path is not a regular file: {path}")
    return descriptor


def _regular_file_snapshot(path: pathlib.Path) -> dict[str, Any]:
    path = reject_symlink_ancestors(path)
    descriptor = _open_regular_file(path)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    if stable_fields != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    ):
        raise RuntimeError(f"identity file changed while hashing: {path}")
    return {
        "sha256": digest.hexdigest(),
        "size": before.st_size,
        "file_type": "regular",
        "mode": stat.S_IMODE(before.st_mode),
    }


def _read_regular_file_bytes(path: pathlib.Path) -> tuple[bytes, dict[str, Any]]:
    path = reject_symlink_ancestors(path)
    descriptor = _open_regular_file(path)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    ):
        raise RuntimeError(f"identity file changed while reading: {path}")
    content = b"".join(chunks)
    return content, {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": before.st_size,
        "file_type": "regular",
        "mode": stat.S_IMODE(before.st_mode),
    }


def _version_command(tool_name: str, path: pathlib.Path) -> list[str]:
    if tool_name == "shell":
        return [str(path), "-c", "printf 'managed-shell\\n'"]
    return [str(path), "--version"]


def _tool_identity(tool_name: str, path: pathlib.Path) -> dict[str, Any]:
    path = reject_symlink_ancestors(path.resolve(strict=True))
    file_identity = _regular_file_snapshot(path)
    if tool_name == "false":
        output = "managed-false-command\n"
    else:
        result = subprocess.run(
            _version_command(tool_name, path),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"LC_ALL": "C", "LANG": "C", "PATH": str(path.parent)},
        )
        output = result.stdout
    return {
        "path": path.as_posix(),
        **file_identity,
        "version_first_line": output.splitlines()[0] if output.splitlines() else "",
        "version_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def capture_host_toolchain() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for logical_name, executable_name in TOOL_EXECUTABLE_NAMES.items():
        discovered = shutil.which(executable_name, path=os.defpath)
        if discovered is None:
            raise RuntimeError(
                f"managed numeric build requires host tool {executable_name!r}"
            )
        tools[logical_name] = _tool_identity(logical_name, pathlib.Path(discovered))
    return {"policy": TOOLCHAIN_POLICY, "tools": tools}


def _tool_path(toolchain: dict[str, Any], name: str) -> str:
    return str(toolchain["tools"][name]["path"])


def numeric_build_environments(toolchain: dict[str, Any]) -> dict[str, dict[str, str]]:
    tool_directories: list[str] = []
    for tool in toolchain["tools"].values():
        directory = pathlib.PurePosixPath(str(tool["path"])).parent.as_posix()
        if directory not in tool_directories:
            tool_directories.append(directory)
    base = {
        "AR": _tool_path(toolchain, "ar"),
        "CC": _tool_path(toolchain, "cc"),
        "CXX": _tool_path(toolchain, "cxx"),
        "HOME": f"{NUMERIC_ROOT_TOKEN}/build/home",
        "LANG": "C",
        "LC_ALL": "C",
        "LD": _tool_path(toolchain, "ld"),
        "MAKE": _tool_path(toolchain, "make"),
        "NM": _tool_path(toolchain, "nm"),
        "PATH": os.pathsep.join(tool_directories),
        "RANLIB": _tool_path(toolchain, "ranlib"),
        "SHELL": _tool_path(toolchain, "shell"),
        "SOURCE_DATE_EPOCH": "0",
        "TMPDIR": f"{NUMERIC_ROOT_TOKEN}/build/tmp",
        "TZ": "UTC",
    }
    managed = {
        **base,
        "M4": f"{NUMERIC_ROOT_TOKEN}/install/m4/bin/m4",
        "PATH": os.pathsep.join(
            [f"{NUMERIC_ROOT_TOKEN}/install/m4/bin", base["PATH"]]
        ),
        "PKG_CONFIG": _tool_path(toolchain, "false"),
    }
    mpfr = {
        **managed,
        "CPPFLAGS": f"-I{NUMERIC_ROOT_TOKEN}/install/gmp/include",
        "LDFLAGS": (
            f"-L{NUMERIC_ROOT_TOKEN}/install/gmp/lib "
            f"-Wl,-rpath,{NUMERIC_ROOT_TOKEN}/install/gmp/lib"
        ),
        "LD_LIBRARY_PATH": f"{NUMERIC_ROOT_TOKEN}/install/gmp/lib",
    }
    version_runtime = {
        **managed,
        "LD_LIBRARY_PATH": os.pathsep.join(
            [
                f"{NUMERIC_ROOT_TOKEN}/install/mpfr/lib",
                f"{NUMERIC_ROOT_TOKEN}/install/gmp/lib",
            ]
        ),
    }
    return {
        "base": base,
        "managed": managed,
        "mpfr": mpfr,
        "version-runtime": version_runtime,
    }


def expand_numeric_root(value: str, root: pathlib.Path) -> str:
    return value.replace(NUMERIC_ROOT_TOKEN, root.as_posix())


def expand_environment(
    environment: dict[str, str], root: pathlib.Path
) -> dict[str, str]:
    return {name: expand_numeric_root(value, root) for name, value in environment.items()}


def numeric_gate_contracts(
    *, jobs: int, toolchain: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    make = _tool_path(toolchain, "make")
    cc = _tool_path(toolchain, "cc")
    root = NUMERIC_ROOT_TOKEN

    def command(
        values: list[str], cwd: str, environment: str
    ) -> dict[str, Any]:
        return {"command": values, "cwd": cwd, "environment": environment}

    gates: dict[str, dict[str, Any]] = {}
    for dependency, source_directory, environment in (
        ("m4", "m4-1.4.21", "base"),
        ("gmp", "gmp-6.3.0", "managed"),
        ("mpfr", "mpfr-4.2.2", "mpfr"),
    ):
        options = list(CONFIGURE_OPTIONS[dependency])
        options = [
            option.replace("${GMP_PREFIX}", f"{root}/install/gmp")
            for option in options
        ]
        gates[f"{dependency}-configure"] = command(
            [
                f"{root}/sources/{source_directory}/configure",
                f"--prefix={root}/install/{dependency}",
                *options,
            ],
            f"{root}/build/{dependency}",
            environment,
        )
        gates[f"{dependency}-build"] = command(
            [make, f"-j{jobs}"], f"{root}/build/{dependency}", environment
        )
        gates[f"{dependency}-check"] = command(
            [make, f"-j{jobs}", "check"],
            f"{root}/build/{dependency}",
            environment,
        )
        gates[f"{dependency}-install"] = command(
            [make, "install"], f"{root}/build/{dependency}", environment
        )
    gates["m4-version"] = command(
        [f"{root}/install/m4/bin/m4", "--version"], root, "base"
    )
    gates["softfloat-build"] = command(
        [
            make,
            f"-j{jobs}",
            f"SOURCE_DIR={root}/sources/SoftFloat-3e/source",
            f"SPECIALIZE_TYPE={SOFTFLOAT_SPECIALIZATION}",
        ],
        f"{root}/build/softfloat",
        "managed",
    )
    gates["testfloat-build"] = command(
        [
            make,
            f"-j{jobs}",
            "testsoftfloat",
            f"SOURCE_DIR={root}/sources/TestFloat-3e/source",
            f"SOFTFLOAT_INCLUDE_DIR={root}/sources/SoftFloat-3e/source/include",
            f"SOFTFLOAT_LIB={root}/build/softfloat/softfloat.a",
        ],
        f"{root}/build/testfloat",
        "managed",
    )
    gates["softfloat-policy-compile"] = command(
        [
            cc,
            "-std=c11",
            f"-DTHREAD_LOCAL={SOFTFLOAT_THREAD_LOCAL}",
            f"-I{root}/install/softfloat/include",
            f"{root}/build/softfloat/wafer-softfloat-policy-probe.c",
            f"{root}/install/softfloat/lib/libsoftfloat.a",
            "-pthread",
            "-o",
            f"{root}/build/softfloat/wafer-softfloat-policy-probe",
        ],
        root,
        "managed",
    )
    gates["softfloat-tls-default-nan"] = command(
        [f"{root}/build/softfloat/wafer-softfloat-policy-probe"],
        root,
        "managed",
    )
    selectors = {
        "testsoftfloat-f16-mulAdd": "f16_mulAdd",
        "testsoftfloat-f32-mulAdd": "f32_mulAdd",
        "testsoftfloat-all1": "-all1",
        "testsoftfloat-all2": "-all2",
    }
    for name, selector in selectors.items():
        gates[name] = command(
            [
                f"{root}/install/testfloat/bin/testsoftfloat",
                "-seed",
                "1",
                "-level",
                "1",
                "-errorstop",
                "-rnear_even",
                "-tininessafter",
                selector,
            ],
            root,
            "managed",
        )
    gates["managed-version-thread-safe-compile"] = command(
        [
            cc,
            f"-I{root}/install/gmp/include",
            f"-I{root}/install/mpfr/include",
            f"{root}/build/managed-version-thread-safe.c",
            f"{root}/install/mpfr/lib/libmpfr.so.6.2.2",
            f"{root}/install/gmp/lib/libgmp.so.10.5.0",
            f"-Wl,-rpath,{root}/install/mpfr/lib",
            f"-Wl,-rpath,{root}/install/gmp/lib",
            "-pthread",
            "-o",
            f"{root}/build/managed-version-thread-safe",
        ],
        root,
        "managed",
    )
    gates["managed-version-thread-safe"] = command(
        [f"{root}/build/managed-version-thread-safe"],
        root,
        "version-runtime",
    )
    if set(gates) != REQUIRED_CONFORMANCE_GATES:
        raise RuntimeError("internal numeric conformance gate contract is incomplete")
    return gates


def _update_tree_digest(
    digest: Any, relative: str, executable: bool, content_digest: str
) -> None:
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(b"1" if executable else b"0")
    digest.update(b"\0")
    digest.update(bytes.fromhex(content_digest))
    digest.update(b"\n")


def sha256_tree(root: pathlib.Path) -> str:
    """Hash regular-file content, relative names, and executable identity.

    Archive timestamps, owners, and directory modes are intentionally omitted;
    they are not source semantics and vary across extraction hosts.
    """

    root = reject_symlink_ancestors(root)
    if not root.is_dir():
        raise RuntimeError(f"source tree is not a directory: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"managed source tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"managed source tree contains a special file: {path}")
        _update_tree_digest(
            digest,
            path.relative_to(root).as_posix(),
            bool(path.stat().st_mode & 0o111),
            sha256_file(path),
        )
    return digest.hexdigest()


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def sha256_archive_tree(archive: pathlib.Path, expected_top_directory: str) -> str:
    """Hash the regular-file tree represented by a safe pinned archive."""

    archive = reject_symlink_ancestors(archive)
    expected = _safe_member_path(expected_top_directory)
    if len(expected.parts) != 1:
        raise RuntimeError("expected archive top directory must be one path component")
    entries: list[tuple[str, bool, str]] = []
    saw_top = False
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                path = _safe_member_path(member.filename.rstrip("/"))
                if path.parts[0] != expected_top_directory:
                    raise RuntimeError(
                        f"archive member is outside {expected_top_directory}: {member.filename}"
                    )
                saw_top = True
                unix_mode = member.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                if file_type == stat.S_IFLNK:
                    raise RuntimeError(f"archive symlink is not permitted: {member.filename}")
                if member.is_dir():
                    continue
                if file_type not in (0, stat.S_IFREG):
                    raise RuntimeError(
                        f"archive special file is not permitted: {member.filename}"
                    )
                relative = pathlib.PurePosixPath(*path.parts[1:]).as_posix()
                if not relative or relative == ".":
                    raise RuntimeError(f"archive top entry must be a directory: {member.filename}")
                with source.open(member) as stream:
                    entries.append(
                        (relative, bool(unix_mode & 0o111), _sha256_stream(stream))
                    )
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, mode="r:*") as source:
            for member in source.getmembers():
                path = _safe_member_path(member.name.rstrip("/"))
                if path.parts[0] != expected_top_directory:
                    raise RuntimeError(
                        f"archive member is outside {expected_top_directory}: {member.name}"
                    )
                saw_top = True
                if member.isdir():
                    continue
                if not member.isfile():
                    raise RuntimeError(
                        f"archive links and special files are not permitted: {member.name}"
                    )
                relative = pathlib.PurePosixPath(*path.parts[1:]).as_posix()
                if not relative or relative == ".":
                    raise RuntimeError(f"archive top entry must be a directory: {member.name}")
                stream = source.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"could not read archive member: {member.name}")
                with stream:
                    entries.append(
                        (relative, bool(member.mode & 0o111), _sha256_stream(stream))
                    )
    else:
        raise RuntimeError(f"unsupported numeric dependency archive: {archive}")
    if not saw_top:
        raise RuntimeError(f"archive does not contain {expected_top_directory}")
    digest = hashlib.sha256()
    seen: set[str] = set()
    for relative, executable, content_digest in sorted(entries):
        if relative in seen:
            raise RuntimeError(f"duplicate archive member path: {relative}")
        seen.add(relative)
        _update_tree_digest(digest, relative, executable, content_digest)
    return digest.hexdigest()


def _safe_member_path(name: str) -> pathlib.PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise RuntimeError(f"unsafe archive member name: {name!r}")
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    return path


def _write_stream(
    stream: Any,
    destination: pathlib.Path,
    mode: int,
    *,
    mtime: float | None = None,
    chunk_size: int = 1024 * 1024,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)
    os.chmod(destination, mode & 0o777)
    if mtime is not None:
        os.utime(destination, (mtime, mtime))


def _extract_zip(archive: pathlib.Path, staging: pathlib.Path) -> None:
    with zipfile.ZipFile(archive) as source:
        seen: set[pathlib.PurePosixPath] = set()
        for member in source.infolist():
            relative = _safe_member_path(member.filename.rstrip("/"))
            if relative in seen:
                raise RuntimeError(f"duplicate archive member path: {member.filename}")
            seen.add(relative)
            destination = staging.joinpath(*relative.parts)
            unix_mode = member.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if file_type == stat.S_IFLNK:
                raise RuntimeError(f"archive symlink is not permitted: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if file_type not in (0, stat.S_IFREG):
                raise RuntimeError(f"archive special file is not permitted: {member.filename}")
            mode = unix_mode & 0o777 if unix_mode else 0o644
            mtime = datetime(*member.date_time, tzinfo=timezone.utc).timestamp()
            with source.open(member) as stream:
                _write_stream(stream, destination, mode, mtime=mtime)


def _extract_tar(archive: pathlib.Path, staging: pathlib.Path) -> None:
    with tarfile.open(archive, mode="r:*") as source:
        seen: set[pathlib.PurePosixPath] = set()
        for member in source.getmembers():
            relative = _safe_member_path(member.name.rstrip("/"))
            if relative in seen:
                raise RuntimeError(f"duplicate archive member path: {member.name}")
            seen.add(relative)
            destination = staging.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"archive links and special files are not permitted: {member.name}"
                )
            stream = source.extractfile(member)
            if stream is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            with stream:
                _write_stream(stream, destination, member.mode, mtime=member.mtime)


def safe_extract_archive(
    archive: pathlib.Path, destination: pathlib.Path, expected_top_directory: str
) -> None:
    """Safely and atomically replace one pinned source directory."""

    archive = reject_symlink_ancestors(archive)
    destination = _absolute_path(destination)
    expected = _safe_member_path(expected_top_directory)
    if len(expected.parts) != 1:
        raise RuntimeError("expected archive top directory must be one path component")
    prepare_managed_directory(destination.parent)
    reject_symlink_ancestors(destination, allow_missing=True)
    staging = destination.parent / f".{destination.name}.extract-{uuid.uuid4().hex}"
    os.mkdir(staging, 0o700)
    reject_symlink_ancestors(staging)
    try:
        if zipfile.is_zipfile(archive):
            _extract_zip(archive, staging)
        elif tarfile.is_tarfile(archive):
            _extract_tar(archive, staging)
        else:
            raise RuntimeError(f"unsupported numeric dependency archive: {archive}")
        children = sorted(item.name for item in staging.iterdir())
        if children != [expected_top_directory]:
            raise RuntimeError(
                f"archive must contain exactly {expected_top_directory!r}, got {children}"
            )
        extracted = staging / expected_top_directory
        if not extracted.is_dir() or extracted.is_symlink():
            raise RuntimeError(
                f"archive top-level entry is not a regular directory: {extracted}"
            )
        if destination.is_symlink():
            raise RuntimeError(f"managed extraction destination is a symlink: {destination}")
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                raise RuntimeError(
                    f"managed extraction destination is not a directory: {destination}"
                )
        os.replace(extracted, destination)
        reject_symlink_ancestors(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _absolute_without_resolving(path: pathlib.Path) -> pathlib.Path:
    return _absolute_path(path)


def _managed_root(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    unresolved = _absolute_without_resolving(root)
    reject_symlink_ancestors(unresolved)
    if not unresolved.is_dir():
        raise RuntimeError(f"managed numeric root is not a directory: {root}")
    resolved = unresolved.resolve(strict=True)
    if resolved != unresolved:
        raise RuntimeError(f"managed numeric root is not canonical: {root}")
    return unresolved, resolved


def _reject_managed_symlink_components(
    root: pathlib.Path, relative: pathlib.PurePosixPath
) -> pathlib.Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(
                f"managed identity path contains a symlink component: {relative}"
            )
    return current


def relative_managed_path(root: pathlib.Path, path: pathlib.Path) -> str:
    unresolved_root, resolved_root = _managed_root(root)
    unresolved_path = _absolute_without_resolving(path)
    try:
        relative = unresolved_path.relative_to(unresolved_root)
    except ValueError as error:
        raise RuntimeError(f"path escapes managed numeric root: {path}") from error
    pure = _safe_member_path(relative.as_posix())
    candidate = _reject_managed_symlink_components(resolved_root, pure)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"path escapes managed numeric root: {path}") from error
    return relative.as_posix()


def resolve_managed_path(
    root: pathlib.Path,
    relative: str,
    *,
    regular_file: bool = False,
    directory: bool = False,
) -> pathlib.Path:
    pure = _safe_member_path(relative)
    _, root = _managed_root(root)
    candidate = _reject_managed_symlink_components(root, pure)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"managed record path escapes numeric root: {relative}") from error
    if regular_file and not resolved.is_file():
        raise RuntimeError(f"managed record path is not a regular file: {relative}")
    if directory and not resolved.is_dir():
        raise RuntimeError(f"managed record path is not a directory: {relative}")
    return resolved


def elf_identity(
    path: pathlib.Path, readelf: pathlib.Path | None
) -> dict[str, Any] | None:
    descriptor = _open_regular_file(reject_symlink_ancestors(path))
    try:
        magic = os.read(descriptor, 4)
    finally:
        os.close(descriptor)
    if magic != b"\x7fELF":
        return None
    if readelf is None:
        discovered = shutil.which("readelf", path=os.defpath)
        if discovered is None:
            raise RuntimeError(f"cannot inspect ELF identity without readelf: {path}")
        readelf = pathlib.Path(discovered).resolve(strict=True)
    readelf = reject_symlink_ancestors(readelf.resolve(strict=True))
    result = subprocess.run(
        [str(readelf), "--wide", "--file-header", "--notes", "--dynamic", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"LC_ALL": "C", "LANG": "C", "PATH": str(readelf.parent)},
    )
    output = result.stdout

    def header(label: str) -> str:
        match = re.search(rf"^\s*{re.escape(label)}:\s*(.+?)\s*$", output, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"readelf did not report {label} for {path}")
        return match.group(1)

    data = header("Data")
    if "little endian" in data:
        byte_order = "little"
    elif "big endian" in data:
        byte_order = "big"
    else:
        raise RuntimeError(f"readelf reported unknown ELF byte order for {path}: {data}")
    elf_type = header("Type").split()[0]
    build_id_match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", output)
    soname_match = re.search(r"\(SONAME\).*?\[([^]]+)\]", output)

    def dynamic_paths(tag: str) -> list[str]:
        values = re.findall(rf"\({tag}\).*?\[([^]]*)\]", output)
        result_paths: list[str] = []
        for value in values:
            result_paths.extend(item for item in value.split(":") if item)
        return result_paths

    return {
        "class": header("Class"),
        "byte_order": byte_order,
        "type": elf_type,
        "machine": header("Machine"),
        "build_id": build_id_match.group(1).lower() if build_id_match else None,
        "soname": soname_match.group(1) if soname_match else None,
        "needed": re.findall(r"\(NEEDED\).*?\[([^]]+)\]", output),
        "rpath": dynamic_paths("RPATH"),
        "runpath": dynamic_paths("RUNPATH"),
    }


def artifact_identity(
    root: pathlib.Path,
    path: pathlib.Path,
    *,
    readelf: pathlib.Path | None = None,
) -> dict[str, Any]:
    relative = relative_managed_path(root, path)
    before = _regular_file_snapshot(path)
    elf = elf_identity(path, readelf)
    after = _regular_file_snapshot(path)
    if before != after:
        raise RuntimeError(f"numeric artifact changed during identity capture: {path}")
    return {
        "path": relative,
        **before,
        "elf": elf,
    }


def atomic_write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path = _absolute_path(path)
    prepare_managed_directory(path.parent)
    reject_symlink_ancestors(path, allow_missing=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file_noreplace(candidate: pathlib.Path, destination: pathlib.Path) -> None:
    candidate = reject_symlink_ancestors(candidate)
    destination = _absolute_path(destination)
    reject_symlink_ancestors(destination, allow_missing=True)
    if destination.exists():
        raise RuntimeError(
            f"managed numeric dependency record is already published: {destination}"
        )
    try:
        os.link(candidate, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"managed numeric dependency record was concurrently published: {destination}"
        ) from error
    os.chmod(destination, 0o444)
    candidate.unlink()
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate JSON key in numeric dependency record: {key}")
        value[key] = item
    return value


def _decode_record(content: bytes) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("numeric dependency record is not UTF-8") from error
    value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise RuntimeError("numeric dependency record must be a JSON object")
    return value


def load_record(path: pathlib.Path) -> dict[str, Any]:
    content, _ = _read_regular_file_bytes(path)
    return _decode_record(content)


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{label} must be at least {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    value = _require_string(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{label} must be a lowercase SHA256 digest")
    return value


def _validate_toolchain(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("numeric build toolchain identity must be an object")
    _require_exact_keys(value, {"policy", "tools"}, "numeric build toolchain")
    if value["policy"] != TOOLCHAIN_POLICY:
        raise RuntimeError("numeric build toolchain policy mismatch")
    tools = value["tools"]
    if not isinstance(tools, dict):
        raise RuntimeError("numeric build tools must be an object")
    _require_exact_keys(tools, set(TOOL_EXECUTABLE_NAMES), "numeric build tools")
    validated: dict[str, Any] = {}
    for name, identity in tools.items():
        if not isinstance(identity, dict):
            raise RuntimeError(f"numeric build tool {name} identity must be an object")
        _require_exact_keys(
            identity,
            {
                "path",
                "sha256",
                "size",
                "file_type",
                "mode",
                "version_first_line",
                "version_output_sha256",
            },
            f"numeric build tool {name}",
        )
        path_text = _require_string(identity["path"], f"numeric build tool {name}.path")
        path = pathlib.Path(path_text)
        if not path.is_absolute():
            raise RuntimeError(f"numeric build tool {name} path must be absolute")
        _require_sha256(identity["sha256"], f"numeric build tool {name}.sha256")
        _require_sha256(
            identity["version_output_sha256"],
            f"numeric build tool {name}.version_output_sha256",
        )
        _require_int(identity["size"], f"numeric build tool {name}.size", minimum=0)
        _require_int(identity["mode"], f"numeric build tool {name}.mode", minimum=0)
        _require_string(
            identity["version_first_line"],
            f"numeric build tool {name}.version_first_line",
        )
        if identity["file_type"] != "regular":
            raise RuntimeError(f"numeric build tool {name} must be a regular file")
        actual = _tool_identity(name, path)
        if identity != actual:
            raise RuntimeError(f"numeric build tool identity mismatch for {name}: {path}")
        validated[name] = identity
    return {"policy": TOOLCHAIN_POLICY, "tools": validated}


def _validate_artifact(
    root: pathlib.Path,
    value: Any,
    label: str,
    *,
    readelf: pathlib.Path,
) -> pathlib.Path:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} identity must be an object")
    _require_exact_keys(
        value,
        {"path", "sha256", "size", "file_type", "mode", "elf"},
        label,
    )
    _require_string(value["path"], f"{label}.path")
    _require_sha256(value["sha256"], f"{label}.sha256")
    _require_int(value["size"], f"{label}.size", minimum=0)
    _require_int(value["mode"], f"{label}.mode", minimum=0)
    if value["file_type"] != "regular":
        raise RuntimeError(f"{label}.file_type must be regular")
    if value["elf"] is not None and not isinstance(value["elf"], dict):
        raise RuntimeError(f"{label}.elf must be an object or null")
    path = resolve_managed_path(root, value["path"], regular_file=True)
    actual = artifact_identity(root, path, readelf=readelf)
    if value != actual:
        raise RuntimeError(f"{label} identity mismatch: {path}")
    return path


def validate_numeric_record_snapshot(
    record_path: pathlib.Path, root: pathlib.Path, versions: dict[str, str]
) -> ValidatedNumericRecord:
    """Validate every identity needed before numeric-model dependencies exist."""

    unresolved_root, root = _managed_root(root)
    unresolved_record = _absolute_without_resolving(record_path)
    try:
        record_relative = unresolved_record.relative_to(unresolved_root)
    except ValueError as error:
        raise RuntimeError("numeric dependency record must be inside its managed root") from error
    record_path = resolve_managed_path(
        root, record_relative.as_posix(), regular_file=True
    )

    record_content, record_file_identity = _read_regular_file_bytes(record_path)
    record = _decode_record(record_content)
    _require_exact_keys(
        record,
        {
            "schema_version",
            "kind",
            "status",
            "pins",
            "build",
            "artifacts",
            "licenses",
            "conformance",
        },
        "numeric dependency record",
    )
    if _require_int(record["schema_version"], "numeric schema_version") != RECORD_SCHEMA_VERSION:
        raise RuntimeError("numeric dependency record schema mismatch")
    if record["kind"] != RECORD_KIND or record["status"] != RECORD_STATUS:
        raise RuntimeError("numeric dependency record is not conformance-passed")

    pins = numeric_pins(versions)
    pin_records = record["pins"]
    if not isinstance(pin_records, dict):
        raise RuntimeError("numeric dependency pins must be an object")
    _require_exact_keys(pin_records, set(pins), "numeric dependency pins")
    for name, pin in pins.items():
        value = pin_records[name]
        if not isinstance(value, dict):
            raise RuntimeError(f"numeric pin record for {name} must be an object")
        _require_exact_keys(
            value,
            {
                "version",
                "url",
                "archive_sha256",
                "archive",
                "source",
                "source_tree_sha256",
            },
            f"numeric pin {name}",
        )
        expected = (pin.version, pin.url, pin.sha256)
        for field in (
            "version",
            "url",
            "archive_sha256",
            "archive",
            "source",
            "source_tree_sha256",
        ):
            _require_string(value[field], f"numeric pin {name}.{field}")
        _require_sha256(value["archive_sha256"], f"numeric pin {name}.archive_sha256")
        _require_sha256(
            value["source_tree_sha256"], f"numeric pin {name}.source_tree_sha256"
        )
        actual = (value["version"], value["url"], value["archive_sha256"])
        if actual != expected:
            raise RuntimeError(f"numeric pin identity mismatch for {name}: {actual}")
        archive = resolve_managed_path(root, value["archive"], regular_file=True)
        if archive.name != pin.archive_name or sha256_file(archive) != pin.sha256:
            raise RuntimeError(f"numeric source archive mismatch for {name}: {archive}")
        source = resolve_managed_path(root, value["source"], directory=True)
        if source.name != pin.source_directory:
            raise RuntimeError(f"numeric source directory mismatch for {name}: {source}")
        archive_tree_digest = sha256_archive_tree(archive, pin.source_directory)
        if value["source_tree_sha256"] != archive_tree_digest:
            raise RuntimeError(
                f"numeric source-tree record is not derived from pinned archive for {name}"
            )
        if sha256_tree(source) != archive_tree_digest:
            raise RuntimeError(f"numeric source tree mismatch for {name}: {source}")

    build = record["build"]
    if not isinstance(build, dict):
        raise RuntimeError("numeric dependency build identity must be an object")
    _require_exact_keys(
        build,
        {
            "platform",
            "softfloat_specialization",
            "softfloat_thread_local",
            "softfloat_raise_flags",
            "configure_options",
            "linkage",
            "managed_m4",
            "pkg_config",
            "jobs",
            "toolchain",
            "environments",
            "mpfr_patches",
            "elf_identity_policy",
        },
        "numeric dependency build identity",
    )
    jobs = _require_int(build["jobs"], "numeric build jobs", minimum=1)
    toolchain = _validate_toolchain(build["toolchain"])
    expected_environments = numeric_build_environments(toolchain)
    expected_build_scalars = {
        "platform": SOFTFLOAT_PLATFORM,
        "softfloat_specialization": SOFTFLOAT_SPECIALIZATION,
        "softfloat_thread_local": SOFTFLOAT_THREAD_LOCAL,
        "softfloat_raise_flags": SOFTFLOAT_RAISE_FLAGS,
        "configure_options": CONFIGURE_OPTIONS,
        "linkage": LINKAGE_POLICY,
        "managed_m4": "install/m4/bin/m4",
        "pkg_config": "disabled",
        "mpfr_patches": "",
        "elf_identity_policy": "sha256-build-id-soname-needed-rpath-v1",
    }
    for field, expected in expected_build_scalars.items():
        if build[field] != expected:
            raise RuntimeError(
                f"numeric dependency build policy/option identity mismatch: {field}"
            )
    if build["environments"] != expected_environments:
        raise RuntimeError("numeric dependency effective environment identity mismatch")
    readelf = pathlib.Path(_tool_path(toolchain, "readelf"))

    artifacts = record["artifacts"]
    if not isinstance(artifacts, dict):
        raise RuntimeError("numeric dependency artifacts must be an object")
    _require_exact_keys(artifacts, REQUIRED_ARTIFACTS, "numeric dependency artifacts")
    artifact_paths = {
        name: _validate_artifact(
            root, value, f"numeric artifact {name}", readelf=readelf
        )
        for name, value in artifacts.items()
    }
    expected_artifact_paths = {
        "m4": "install/m4/bin/m4",
        "softfloat": "install/softfloat/lib/libsoftfloat.a",
        "softfloat-header": "install/softfloat/include/softfloat.h",
        "softfloat-types-header": "install/softfloat/include/softfloat_types.h",
        "testsoftfloat": "install/testfloat/bin/testsoftfloat",
        "gmp": "install/gmp/lib/libgmp.so.10.5.0",
        "gmp-soname": "install/gmp/lib/libgmp.so.10",
        "gmp-header": "install/gmp/include/gmp.h",
        "mpfr": "install/mpfr/lib/libmpfr.so.6.2.2",
        "mpfr-soname": "install/mpfr/lib/libmpfr.so.6",
        "mpfr-header": "install/mpfr/include/mpfr.h",
        **{
            name: f"install/licenses/{name.removeprefix('license-')}.txt"
            for name in LICENSE_SOURCE_FILES
        },
    }
    for name, expected_relative in expected_artifact_paths.items():
        if artifacts[name]["path"] != expected_relative:
            raise RuntimeError(f"numeric artifact path mismatch for {name}")
    for executable_name in ("m4", "testsoftfloat"):
        if artifacts[executable_name]["mode"] & 0o111 == 0:
            raise RuntimeError(f"numeric artifact is not executable: {executable_name}")
    if artifact_paths["softfloat"].suffix != ".a":
        raise RuntimeError("SoftFloat must be a managed static archive")
    if ".so." not in artifact_paths["gmp"].name or ".so." not in artifact_paths["mpfr"].name:
        raise RuntimeError("GMP and MPFR must identify real versioned shared objects")
    for name in ("gmp", "mpfr"):
        soname = artifact_paths[f"{name}-soname"]
        library = artifact_paths[name]
        elf = artifacts[name]["elf"]
        if not isinstance(elf, dict) or elf.get("type") != "DYN":
            raise RuntimeError(f"{name} artifact is not a shared ELF object")
        expected_elf_keys = {
            "class",
            "byte_order",
            "type",
            "machine",
            "build_id",
            "soname",
            "needed",
            "rpath",
            "runpath",
        }
        _require_exact_keys(elf, expected_elf_keys, f"numeric {name} ELF identity")
        if not isinstance(elf["build_id"], str) or not re.fullmatch(
            r"[0-9a-f]+", elf["build_id"]
        ):
            raise RuntimeError(f"{name} shared object has no usable ELF build-id")
        if not isinstance(elf["soname"], str) or not elf["soname"].startswith(
            f"lib{name}.so."
        ):
            raise RuntimeError(f"{name} shared object has an invalid SONAME")
        if soname == library or soname.name != elf["soname"]:
            raise RuntimeError(f"{name} loader-facing SONAME artifact is invalid")
        if artifacts[f"{name}-soname"]["sha256"] != artifacts[name]["sha256"]:
            raise RuntimeError(f"{name} SONAME artifact does not match its real library")
        if artifacts[f"{name}-soname"]["elf"] != elf:
            raise RuntimeError(f"{name} SONAME ELF identity does not match its real library")
    gmp_soname = artifacts["gmp"]["elf"]["soname"]
    if gmp_soname not in artifacts["mpfr"]["elf"]["needed"]:
        raise RuntimeError("MPFR ELF identity does not name the managed GMP SONAME")
    expected_gmp_runtime_path = (root / "install/gmp/lib").as_posix()
    mpfr_runtime_paths = [
        *artifacts["mpfr"]["elf"]["rpath"],
        *artifacts["mpfr"]["elf"]["runpath"],
    ]
    if mpfr_runtime_paths != [expected_gmp_runtime_path]:
        raise RuntimeError("MPFR ELF runtime path does not exactly name managed GMP")

    licenses = record["licenses"]
    if not isinstance(licenses, dict):
        raise RuntimeError("numeric dependency licenses must be an object")
    _require_exact_keys(licenses, set(LICENSE_SOURCE_FILES), "numeric licenses")
    for artifact_name, (dependency, source_relative) in LICENSE_SOURCE_FILES.items():
        value = licenses[artifact_name]
        if not isinstance(value, dict):
            raise RuntimeError(f"numeric license {artifact_name} must be an object")
        _require_exact_keys(
            value,
            {"dependency", "source", "artifact", "sha256"},
            f"numeric license {artifact_name}",
        )
        expected_source = (
            f"sources/{pins[dependency].source_directory}/{source_relative}"
        )
        if value["dependency"] != dependency or value["source"] != expected_source:
            raise RuntimeError(f"numeric license source mismatch for {artifact_name}")
        if value["artifact"] != artifact_name:
            raise RuntimeError(f"numeric license artifact mismatch for {artifact_name}")
        _require_sha256(value["sha256"], f"numeric license {artifact_name}.sha256")
        source = resolve_managed_path(root, expected_source, regular_file=True)
        source_digest = sha256_file(source)
        if value["sha256"] != source_digest:
            raise RuntimeError(f"numeric license source digest mismatch for {artifact_name}")
        if artifacts[artifact_name]["sha256"] != source_digest:
            raise RuntimeError(f"numeric license copy mismatch for {artifact_name}")

    conformance = record["conformance"]
    if not isinstance(conformance, dict):
        raise RuntimeError("numeric conformance identity must be an object")
    _require_exact_keys(conformance, {"policy", "gates"}, "numeric conformance")
    if conformance["policy"] != CONFORMANCE_POLICY:
        raise RuntimeError("numeric conformance policy mismatch")
    gates = conformance["gates"]
    if not isinstance(gates, list):
        raise RuntimeError("numeric conformance gates must be an array")
    expected_contracts = numeric_gate_contracts(jobs=jobs, toolchain=toolchain)
    by_name: dict[str, dict[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, dict):
            raise RuntimeError("numeric conformance gate must be an object")
        _require_exact_keys(
            gate,
            {"name", "command", "cwd", "environment", "exit_code", "log"},
            "numeric conformance gate",
        )
        name = gate["name"]
        if not isinstance(name, str) or name in by_name:
            raise RuntimeError(f"duplicate or invalid numeric conformance gate: {name!r}")
        if name not in expected_contracts:
            raise RuntimeError(f"unexpected numeric conformance gate: {name}")
        expected = expected_contracts[name]
        for field in ("command", "cwd", "environment"):
            if gate[field] != expected[field]:
                raise RuntimeError(
                    f"numeric conformance gate {name} {field} does not match frozen policy"
                )
        if _require_int(gate["exit_code"], f"numeric gate {name}.exit_code") != 0:
            raise RuntimeError(f"numeric conformance gate {name} did not pass")
        log = _validate_artifact(
            root,
            gate["log"],
            f"numeric conformance log {name}",
            readelf=readelf,
        )
        if gate["log"]["path"] != f"conformance/{name}.log":
            raise RuntimeError(f"numeric conformance log path mismatch: {name}")
        if log.stat().st_size == 0:
            raise RuntimeError(f"numeric conformance log is empty: {name}")
        by_name[name] = gate
    if set(by_name) != REQUIRED_CONFORMANCE_GATES:
        raise RuntimeError(
            "numeric conformance gate closure mismatch: "
            f"missing={sorted(REQUIRED_CONFORMANCE_GATES - set(by_name))}, "
            f"unexpected={sorted(set(by_name) - REQUIRED_CONFORMANCE_GATES)}"
        )
    return ValidatedNumericRecord(
        record=record,
        record_sha256=record_file_identity["sha256"],
        root=root,
        record_path=record_path,
        artifacts=artifact_paths,
    )


def validate_numeric_record(
    record_path: pathlib.Path, root: pathlib.Path, versions: dict[str, str]
) -> dict[str, Any]:
    return validate_numeric_record_snapshot(record_path, root, versions).record
