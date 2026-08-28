#!/usr/bin/env python3
"""Fast negative and identity tests for managed numeric dependencies."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

from bootstrap_deps import (
    build_numeric_model_dependencies,
    fetch_numeric_sources,
)
from numeric_deps import (
    CONFIGURE_OPTIONS,
    CONFORMANCE_POLICY,
    LICENSE_SOURCE_FILES,
    LINKAGE_POLICY,
    RECORD_KIND,
    RECORD_STATUS,
    REQUIRED_FILES,
    REQUIRED_CONFORMANCE_GATES,
    SOFTFLOAT_PLATFORM,
    SOFTFLOAT_RAISE_FLAGS,
    SOFTFLOAT_SPECIALIZATION,
    SOFTFLOAT_THREAD_LOCAL,
    TOOLCHAIN_POLICY,
    file_identity,
    atomic_write_json,
    capture_host_toolchain,
    load_record,
    numeric_build_environments,
    numeric_build_lock,
    numeric_gate_contracts,
    numeric_pins,
    relative_managed_path,
    safe_extract_archive,
    sha256_file,
    sha256_tree,
    validate_numeric_record,
    validate_numeric_record_snapshot,
)


PIN_DATA = {
    "softfloat": ("SOFTFLOAT", "3e", "SoftFloat-3e.zip", "SoftFloat-3e"),
    "testfloat": ("TESTFLOAT", "3e", "TestFloat-3e.zip", "TestFloat-3e"),
    "m4": ("M4", "1.4.21", "m4-1.4.21.tar.xz", "m4-1.4.21"),
    "gmp": ("GMP", "6.3.0", "gmp-6.3.0.tar.xz", "gmp-6.3.0"),
    "mpfr": ("MPFR", "4.2.2", "mpfr-4.2.2.tar.xz", "mpfr-4.2.2"),
}


def _zip_file(output: zipfile.ZipFile, name: str, content: bytes) -> None:
    member = zipfile.ZipInfo(name)
    member.external_attr = 0o100644 << 16
    output.writestr(member, content)


def make_versions(root: pathlib.Path) -> tuple[dict[str, str], dict[str, pathlib.Path]]:
    versions: dict[str, str] = {}
    archives: dict[str, pathlib.Path] = {}
    downloads = root / "downloads"
    downloads.mkdir()
    license_files_by_dependency: dict[str, list[str]] = {}
    for dependency, source_relative in LICENSE_SOURCE_FILES.values():
        license_files_by_dependency.setdefault(dependency, []).append(source_relative)
    for name, (prefix, version, archive_name, source_name) in PIN_DATA.items():
        archive = downloads / archive_name
        with zipfile.ZipFile(archive, "w") as output:
            directory = zipfile.ZipInfo(f"{source_name}/")
            directory.external_attr = (0o40755 << 16) | 0x10
            output.writestr(directory, b"")
            _zip_file(
                output,
                f"{source_name}/source.txt",
                f"{name}-source\n".encode(),
            )
            for license_relative in license_files_by_dependency.get(name, []):
                _zip_file(
                    output,
                    f"{source_name}/{license_relative}",
                    f"license:{name}:{license_relative}\n".encode(),
                )
        versions[f"WAFER_{prefix}_VERSION"] = version
        versions[f"WAFER_{prefix}_URL"] = f"https://example.invalid/{archive_name}"
        versions[f"WAFER_{prefix}_SHA256"] = sha256_file(archive)
        archives[name] = archive
    return versions, archives


def _write_executable(path: pathlib.Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n# {label}\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def _build_fixture_shared_objects(
    root: pathlib.Path, toolchain: dict[str, object]
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    cc = str(toolchain["tools"]["cc"]["path"])  # type: ignore[index]
    fixture = root / "fixture"
    fixture.mkdir()
    gmp_source = fixture / "gmp.c"
    mpfr_source = fixture / "mpfr.c"
    gmp_source.write_text("int wafer_gmp_fixture(void) { return 17; }\n", encoding="utf-8")
    mpfr_source.write_text(
        "extern int wafer_gmp_fixture(void);\n"
        "int wafer_mpfr_fixture(void) { return wafer_gmp_fixture(); }\n",
        encoding="utf-8",
    )
    gmp_dir = root / "install/gmp/lib"
    mpfr_dir = root / "install/mpfr/lib"
    gmp_dir.mkdir(parents=True)
    mpfr_dir.mkdir(parents=True)
    gmp = gmp_dir / "libgmp.so.10.5.0"
    mpfr = mpfr_dir / "libmpfr.so.6.2.2"
    environment = {"LC_ALL": "C", "LANG": "C", "PATH": os.defpath}
    subprocess.run(
        [
            cc,
            "-shared",
            "-fPIC",
            "-Wl,--build-id=sha1",
            "-Wl,-soname,libgmp.so.10",
            str(gmp_source),
            "-o",
            str(gmp),
        ],
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            cc,
            "-shared",
            "-fPIC",
            "-Wl,--build-id=sha1",
            "-Wl,-soname,libmpfr.so.6",
            str(mpfr_source),
            str(gmp),
            f"-Wl,-rpath,{gmp_dir}",
            "-o",
            str(mpfr),
        ],
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    gmp_soname = gmp_dir / "libgmp.so.10"
    mpfr_soname = mpfr_dir / "libmpfr.so.6"
    os.link(gmp, gmp_soname)
    os.link(mpfr, mpfr_soname)
    return gmp, gmp_soname, mpfr, mpfr_soname


def make_record(
    root: pathlib.Path,
    versions: dict[str, str],
    archives: dict[str, pathlib.Path],
    *,
    jobs: int = 2,
) -> pathlib.Path:
    pins = numeric_pins(versions)
    sources = root / "sources"
    sources.mkdir()
    pin_records = {}
    for name, pin in pins.items():
        source = sources / pin.source_directory
        safe_extract_archive(archives[name], source, pin.source_directory)
        pin_records[name] = {
            "version": pin.version,
            "url": pin.url,
            "archive_sha256": pin.sha256,
            "archive": relative_managed_path(root, archives[name]),
            "source": relative_managed_path(root, source),
            "source_tree_sha256": sha256_tree(source),
        }

    toolchain = capture_host_toolchain()
    self = unittest.TestCase()
    self.assertEqual(toolchain["policy"], TOOLCHAIN_POLICY)
    readelf = pathlib.Path(str(toolchain["tools"]["readelf"]["path"]))
    gmp, gmp_soname, mpfr, mpfr_soname = _build_fixture_shared_objects(
        root, toolchain
    )
    file_paths = {
        "m4": root / "install/m4/bin/m4",
        "softfloat": root / "install/softfloat/lib/libsoftfloat.a",
        "softfloat-header": root / "install/softfloat/include/softfloat.h",
        "softfloat-types-header": root / "install/softfloat/include/softfloat_types.h",
        "testsoftfloat": root / "install/testfloat/bin/testsoftfloat",
        "gmp": gmp,
        "gmp-soname": gmp_soname,
        "gmp-header": root / "install/gmp/include/gmp.h",
        "mpfr": mpfr,
        "mpfr-soname": mpfr_soname,
        "mpfr-header": root / "install/mpfr/include/mpfr.h",
    }
    _write_executable(file_paths["m4"], "managed m4 fixture")
    _write_executable(file_paths["testsoftfloat"], "TestFloat fixture")
    for name in (
        "softfloat",
        "softfloat-header",
        "softfloat-types-header",
        "gmp-header",
        "mpfr-header",
    ):
        path = file_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"file-{name}\n".encode())

    license_records: dict[str, dict[str, str]] = {}
    for file_name, (dependency, source_relative) in LICENSE_SOURCE_FILES.items():
        source = sources / pins[dependency].source_directory / source_relative
        destination = (
            root
            / "install/licenses"
            / f"{file_name.removeprefix('license-')}.txt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        file_paths[file_name] = destination
        license_records[file_name] = {
            "dependency": dependency,
            "source": relative_managed_path(root, source),
            "artifact": file_name,
            "sha256": sha256_file(source),
        }
    self.assertEqual(set(file_paths), REQUIRED_FILES)

    contracts = numeric_gate_contracts(jobs=jobs, toolchain=toolchain)
    self.assertEqual(set(contracts), REQUIRED_CONFORMANCE_GATES)
    gates = []
    for name, contract in sorted(contracts.items()):
        log = root / "conformance" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"passed {name}\n", encoding="utf-8")
        gates.append(
            {
                "name": name,
                **contract,
                "exit_code": 0,
                "log": file_identity(root, log, readelf=readelf),
            }
        )

    record = {
        "kind": RECORD_KIND,
        "status": RECORD_STATUS,
        "pins": pin_records,
        "build": {
            "platform": SOFTFLOAT_PLATFORM,
            "softfloat_specialization": SOFTFLOAT_SPECIALIZATION,
            "softfloat_thread_local": SOFTFLOAT_THREAD_LOCAL,
            "softfloat_raise_flags": SOFTFLOAT_RAISE_FLAGS,
            "configure_options": CONFIGURE_OPTIONS,
            "linkage": LINKAGE_POLICY,
            "managed_m4": "install/m4/bin/m4",
            "pkg_config": "disabled",
            "jobs": jobs,
            "toolchain": toolchain,
            "environments": numeric_build_environments(toolchain),
            "mpfr_patches": "",
            "elf_identity_policy": "sha256-build-id-soname-needed-rpath",
        },
        "artifacts": {
            name: file_identity(root, path, readelf=readelf)
            for name, path in file_paths.items()
        },
        "licenses": license_records,
        "conformance": {"policy": CONFORMANCE_POLICY, "gates": gates},
    }
    record_path = root / "numeric-model-deps.json"
    atomic_write_json(record_path, record)
    return record_path


def record_readelf(value: dict[str, object]) -> pathlib.Path:
    return pathlib.Path(
        str(value["build"]["toolchain"]["tools"]["readelf"]["path"])  # type: ignore[index]
    )


class NumericDependencyTest(unittest.TestCase):
    def test_invalid_rebuild_preserves_installed_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            record = root / "numeric-model-deps.json"
            record.write_text("installed record\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "job count"):
                build_numeric_model_dependencies({}, root, 0)
            self.assertEqual(record.read_text(encoding="utf-8"), "installed record\n")

    def test_valid_installed_record_is_reused_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            before = sha256_file(record)
            self.assertEqual(
                build_numeric_model_dependencies(versions, root, 2), record
            )
            self.assertEqual(sha256_file(record), before)

    def test_record_accepts_complete_identity_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            validate_numeric_record(record, root, versions)
            gmp = root / "install/gmp/lib/libgmp.so.10.5.0"
            gmp.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                validate_numeric_record(record, root, versions)

    def test_record_rejects_source_not_derived_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            source = root / "sources/SoftFloat-3e/source.txt"
            source.write_text("different source\n", encoding="utf-8")
            value = load_record(record)
            value["pins"]["softfloat"]["source_tree_sha256"] = sha256_tree(
                source.parent
            )
            atomic_write_json(record, value)
            with self.assertRaisesRegex(RuntimeError, "not derived from pinned archive"):
                validate_numeric_record(record, root, versions)

    def test_record_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "record.json"
            path.write_text('{"kind":"first","kind":"second"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                load_record(path)

    def test_record_rejects_missing_thread_safe_build_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            value = load_record(record)
            value["build"]["configure_options"]["mpfr"].remove(
                "--enable-thread-safe"
            )
            atomic_write_json(record, value)
            with self.assertRaisesRegex(RuntimeError, "build policy/option"):
                validate_numeric_record(record, root, versions)

    def test_record_rejects_any_changed_gate_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            value = load_record(record)
            gate = next(
                gate
                for gate in value["conformance"]["gates"]
                if gate["name"] == "mpfr-check"
            )
            gate["command"] = ["true"]
            atomic_write_json(record, value)
            with self.assertRaisesRegex(RuntimeError, "frozen policy"):
                validate_numeric_record(record, root, versions)

    def test_record_rejects_bool_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            value = load_record(record)
            value["conformance"]["gates"][0]["exit_code"] = False
            atomic_write_json(record, value)
            with self.assertRaisesRegex(RuntimeError, "must be an integer"):
                validate_numeric_record(record, root, versions)

    def test_record_rejects_toolchain_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            value = load_record(record)
            value["build"]["toolchain"]["tools"]["cc"][
                "version_first_line"
            ] += " changed"
            atomic_write_json(record, value)
            with self.assertRaisesRegex(RuntimeError, "tool identity mismatch"):
                validate_numeric_record(record, root, versions)

    def test_record_rejects_removed_executable_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            (root / "install/testfloat/bin/testsoftfloat").chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                validate_numeric_record(record, root, versions)

    def test_record_binds_license_copy_to_pinned_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            value = load_record(record)
            license_path = pathlib.Path(
                root / value["artifacts"]["license-mpfr-lesser"]["path"]
            )
            license_path.write_text("wrong license\n", encoding="utf-8")
            value["artifacts"]["license-mpfr-lesser"] = file_identity(
                root, license_path, readelf=record_readelf(value)
            )
            atomic_write_json(record, value)
            with self.assertRaisesRegex(RuntimeError, "license copy mismatch"):
                validate_numeric_record(record, root, versions)

    def test_record_symlink_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            real_record = root / "real-record.json"
            record.rename(real_record)
            record.symlink_to(real_record.name)
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                validate_numeric_record(record, root, versions)

    def test_managed_parent_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            gmp = root / "install/gmp"
            real_gmp = root / "install/gmp-real"
            gmp.rename(real_gmp)
            gmp.symlink_to(real_gmp.name, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                validate_numeric_record(record, root, versions)

    def test_managed_root_ancestor_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            real = parent / "real"
            real.mkdir()
            root = real / "numeric"
            root.mkdir()
            versions, archives = make_versions(root)
            make_record(root, versions, archives)
            (parent / "alias").symlink_to(real, target_is_directory=True)
            alias_root = parent / "alias/numeric"
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                validate_numeric_record(
                    alias_root / "numeric-model-deps.json", alias_root, versions
                )

    def test_download_directory_symlink_is_rejected_before_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            archive_root = parent / "archive-root"
            archive_root.mkdir()
            versions, _ = make_versions(archive_root)
            root = parent / "numeric"
            root.mkdir()
            (root / "downloads").symlink_to(
                archive_root / "downloads", target_is_directory=True
            )
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                fetch_numeric_sources(versions, root)

    def test_build_lock_rejects_concurrent_modifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with numeric_build_lock(root):
                with self.assertRaisesRegex(RuntimeError, "already being modified"):
                    with numeric_build_lock(root):
                        self.fail("nested lock unexpectedly succeeded")

    def test_canonical_snapshot_uses_validated_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versions, archives = make_versions(root)
            record = make_record(root, versions, archives)
            validated = validate_numeric_record_snapshot(record, root, versions)
            snapshot = validated.cmake_snapshot()
            self.assertEqual(snapshot["record_sha256"], sha256_file(record))
            self.assertTrue(pathlib.Path(snapshot["root"]).is_absolute())
            self.assertTrue(pathlib.Path(snapshot["record"]).is_absolute())
            for file_record in snapshot["artifacts"].values():
                self.assertTrue(pathlib.Path(file_record["path"]).is_absolute())
            json.dumps(snapshot, sort_keys=True)

    def test_zip_path_traversal_is_rejected_without_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape", b"bad")
            with self.assertRaisesRegex(RuntimeError, "unsafe archive member"):
                safe_extract_archive(archive, root / "source", "source")
            self.assertFalse((root / "escape").exists())
            self.assertFalse((root / "source").exists())

    def test_tar_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as output:
                directory = tarfile.TarInfo("source")
                directory.type = tarfile.DIRTYPE
                output.addfile(directory)
                link = tarfile.TarInfo("source/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                output.addfile(link)
            with self.assertRaisesRegex(RuntimeError, "links and special files"):
                safe_extract_archive(archive, root / "source", "source")
            self.assertFalse((root / "source").exists())

    def test_zip_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                directory = zipfile.ZipInfo("source/")
                directory.external_attr = (0o40755 << 16) | 0x10
                output.writestr(directory, b"")
                link = zipfile.ZipInfo("source/link")
                link.external_attr = 0o120777 << 16
                output.writestr(link, b"/etc/passwd")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                safe_extract_archive(archive, root / "source", "source")


if __name__ == "__main__":
    unittest.main()
