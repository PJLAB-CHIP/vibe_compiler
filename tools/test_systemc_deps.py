#!/usr/bin/env python3
"""Fast closed-schema and identity tests for managed SystemC files."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import tarfile
import tempfile
import unittest

from bootstrap_deps import write_systemc_consumer
from systemc_deps import (
    BUILD_OPTIONS,
    RECORD_KIND,
    RECORD_SCHEMA_VERSION,
    RECORD_STATUS,
    REQUIRED_FILES,
    REQUIRED_GATES,
    SNAPSHOT_KIND,
    canonical_json,
    load_versions,
    make_snapshot,
    sha256_file,
    sha256_tree,
    validate_record,
)


def make_fixture(root: pathlib.Path) -> tuple[pathlib.Path, dict[str, object]]:
    versions = load_versions()
    source_top = "systemc-" + versions["WAFER_SYSTEMC_VERSION"]
    source = root / "sources" / source_top
    source.mkdir(parents=True, exist_ok=True)
    (source / "LICENSE").write_text("fixture source\n", encoding="utf-8")
    archive = root / "downloads" / f"systemc-{versions['WAFER_SYSTEMC_VERSION']}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname=source_top)
    versions = dict(versions)
    versions["WAFER_SYSTEMC_SHA256"] = sha256_file(archive)
    files: dict[str, dict[str, object]] = {}
    for name in sorted(REQUIRED_FILES):
        path = (
            archive
            if name == "systemc-archive"
            else root / "install" / "systemc" / f"{name}.file"
        )
        if name != "systemc-archive":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"managed SystemC fixture: {name}\n".encode())
        files[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    record: dict[str, object] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "kind": RECORD_KIND,
        "status": RECORD_STATUS,
        "dependency": {
            "name": "Accellera SystemC",
            "version": versions["WAFER_SYSTEMC_VERSION"],
            "commit": versions["WAFER_SYSTEMC_COMMIT"],
            "url": versions["WAFER_SYSTEMC_URL"],
            "archive_sha256": versions["WAFER_SYSTEMC_SHA256"],
            "source_tree_sha256": sha256_tree(source),
        },
        "build": {
            "options": BUILD_OPTIONS,
            "generator": "fixture-generator",
            "install_prefix": "install/systemc",
            "install_tree_sha256": "",
        },
        "toolchain": {
            name: {"path": f"/fixture/{name}", "version": "fixture 1", "sha256": "2" * 64}
            for name in ("cmake", "cxx")
        },
        "artifacts": files,
        "licenses": {
            "license": {"artifact": "license", "spdx": "Apache-2.0"},
            "notice": {"artifact": "notice", "spdx": "Apache-2.0"},
        },
        "conformance": {
            "gates": [
                {
                    "name": name,
                    "command": ["fixture", name],
                    "log": f"conformance/{name}.log",
                    "log_sha256": "",
                    "exit_code": 0,
                }
                for name in sorted(REQUIRED_GATES)
            ]
        },
    }
    for gate in record["conformance"]["gates"]:
        log = root / gate["log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"fixture gate: {gate['name']}\n", encoding="utf-8")
        gate["log_sha256"] = sha256_file(log)
    indirect_header = root / "install" / "systemc" / "include" / "indirect-header"
    indirect_header.parent.mkdir(parents=True)
    indirect_header.write_text("fixture indirect header\n", encoding="utf-8")
    record["build"]["install_tree_sha256"] = sha256_tree(root / "install" / "systemc")
    record_path = root / "systemc-model-deps.json"
    record_path.write_text(canonical_json(record), encoding="utf-8")
    return record_path, record


class SystemCDependencyRecordTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="wafer-systemc-deps-")
        self.root = pathlib.Path(self.temporary.name)
        self.record_path, self.record = make_fixture(self.root)
        self.versions = load_versions()
        self.versions["WAFER_SYSTEMC_SHA256"] = self.record["dependency"]["archive_sha256"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def rewrite(self, record: dict[str, object]) -> None:
        self.record_path.write_text(canonical_json(record), encoding="utf-8")

    def test_generated_consumer_accepts_managed_host_cmake_baseline(self) -> None:
        source = self.root / "consumer"
        write_systemc_consumer(source)
        cmake_lists = (source / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("cmake_minimum_required(VERSION 3.16)", cmake_lists)
        self.assertNotIn("cmake_minimum_required(VERSION 3.24)", cmake_lists)

    def test_valid_record_produces_closed_absolute_snapshot(self) -> None:
        resolved = validate_record(self.record_path, self.root, self.versions)
        self.assertEqual(set(resolved), REQUIRED_FILES)
        snapshot = make_snapshot(self.record_path, self.root, self.versions)
        self.assertEqual(snapshot["kind"], SNAPSHOT_KIND)
        self.assertEqual(snapshot["record_sha256"], sha256_file(self.record_path))
        self.assertTrue(
            all(pathlib.Path(item["path"]).is_absolute() for item in snapshot["artifacts"].values())
        )

    def test_unknown_and_noncanonical_records_are_rejected(self) -> None:
        changed = copy.deepcopy(self.record)
        changed["unknown"] = False
        self.rewrite(changed)
        with self.assertRaisesRegex(RuntimeError, "fields mismatch"):
            validate_record(self.record_path, self.root, self.versions)
        self.record_path.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not canonical JSON"):
            validate_record(self.record_path, self.root, self.versions)

    def test_changed_file_source_and_gate_are_rejected(self) -> None:
        entry = self.record["artifacts"]["systemc-library"]
        (self.root / entry["path"]).write_bytes(b"tampered\n")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            validate_record(self.record_path, self.root, self.versions)

        (self.root / entry["path"]).write_bytes(
            b"managed SystemC fixture: systemc-library\n"
        )
        source = self.root / "sources" / ("systemc-" + self.versions["WAFER_SYSTEMC_VERSION"])
        (source / "LICENSE").write_text("tampered source\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source tree mismatch"):
            validate_record(self.record_path, self.root, self.versions)

        (source / "LICENSE").write_text("fixture source\n", encoding="utf-8")
        first_gate = self.record["conformance"]["gates"][0]
        (self.root / first_gate["log"]).write_text("tampered log\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "gate log identity"):
            validate_record(self.record_path, self.root, self.versions)

    def test_build_option_and_gate_closure_are_rejected(self) -> None:
        changed = copy.deepcopy(self.record)
        changed["build"]["options"]["ENABLE_EXAMPLES"] = "ON"
        self.rewrite(changed)
        with self.assertRaisesRegex(RuntimeError, "build options mismatch"):
            validate_record(self.record_path, self.root, self.versions)
        changed = copy.deepcopy(self.record)
        changed["conformance"]["gates"].pop()
        self.rewrite(changed)
        with self.assertRaisesRegex(RuntimeError, "gate closure mismatch"):
            validate_record(self.record_path, self.root, self.versions)

    def test_changed_indirect_install_file_is_rejected(self) -> None:
        indirect_header = self.root / "install" / "systemc" / "include" / "indirect-header"
        indirect_header.write_text("tampered indirect header\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "install tree mismatch"):
            validate_record(self.record_path, self.root, self.versions)

    def test_record_outside_root_and_symlink_component_are_rejected(self) -> None:
        outside = self.root.parent / f"{self.root.name}-outside.json"
        try:
            outside.write_text(canonical_json(self.record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inside its managed root"):
                validate_record(outside, self.root, self.versions)
        finally:
            outside.unlink(missing_ok=True)
        link = self.root.parent / f"{self.root.name}-link"
        try:
            os.symlink(self.root, link)
            with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
                validate_record(link / self.record_path.name, link, self.versions)
        finally:
            link.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
