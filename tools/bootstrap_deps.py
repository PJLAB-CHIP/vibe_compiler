#!/usr/bin/env python3
"""Fetch pinned development dependencies for the Wafer compiler prototype."""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid

from numeric_deps import (
    CONFIGURE_OPTIONS,
    CONFORMANCE_POLICY,
    LINKAGE_POLICY,
    NumericPin,
    RECORD_KIND,
    RECORD_SCHEMA_VERSION,
    RECORD_STATUS,
    SOFTFLOAT_PLATFORM,
    SOFTFLOAT_RAISE_FLAGS,
    SOFTFLOAT_SPECIALIZATION,
    SOFTFLOAT_THREAD_LOCAL,
    artifact_identity,
    atomic_write_json,
    capture_host_toolchain,
    expand_environment,
    expand_numeric_root,
    LICENSE_SOURCE_FILES,
    numeric_build_environments,
    numeric_build_lock,
    numeric_gate_contracts,
    numeric_pins,
    prepare_managed_directory,
    publish_file_noreplace,
    reject_symlink_ancestors,
    relative_managed_path,
    safe_extract_archive,
    sha256_file,
    sha256_tree,
    validate_numeric_record,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSIONS_FILE = REPO_ROOT / "cmake" / "third_party" / "WaferDependencyVersions.cmake"


def load_versions() -> dict[str, str]:
    text = VERSIONS_FILE.read_text(encoding="utf-8")
    versions: dict[str, str] = {}
    for name, value in re.findall(r'set\((WAFER_[A-Z0-9_]+)\s+"([^"]+)"\)', text):
        versions[name] = value
    return versions


def run(command: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_venv(prefix: pathlib.Path, name: str, requirements: pathlib.Path) -> pathlib.Path:
    venv = prefix / name
    python = venv / "bin" / "python"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(venv)])
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-r", str(requirements)])
    return python


def clone_or_update(repo: str, commit: str, destination: pathlib.Path) -> None:
    if not destination.exists():
        destination.mkdir(parents=True)
        run(["git", "init"], cwd=destination)
        run(["git", "remote", "add", "origin", repo], cwd=destination)
    run(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
    run(["git", "checkout", "--detach", commit], cwd=destination)


def sync_submodule(path: pathlib.Path, commit: str) -> None:
    relative = path.relative_to(REPO_ROOT)
    run(["git", "submodule", "update", "--init", "--depth", "1", str(relative)], cwd=REPO_ROOT)
    run(["git", "checkout", "--detach", commit], cwd=path)


def get_remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    return int(content_length)


def is_complete_file(path: pathlib.Path, expected_size: int | None) -> bool:
    if not path.exists():
        return False
    if expected_size is None:
        return True
    return path.stat().st_size == expected_size


def download_with_resume(url: str, destination: pathlib.Path) -> None:
    if destination.is_symlink():
        raise RuntimeError(f"download destination must not be a symlink: {destination}")
    expected_size = get_remote_content_length(url)
    if is_complete_file(destination, expected_size):
        return

    part = destination.with_name(destination.name + ".part")
    if part.is_symlink():
        raise RuntimeError(f"partial download must not be a symlink: {part}")
    if destination.exists():
        if expected_size is not None and destination.stat().st_size > expected_size:
            destination.unlink()
        elif not part.exists() or destination.stat().st_size > part.stat().st_size:
            destination.replace(part)
        else:
            destination.unlink()

    existing_size = part.stat().st_size if part.exists() else 0
    if expected_size is not None and existing_size == expected_size:
        part.replace(destination)
        return

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        append = existing_size > 0 and response.status == 206
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(part, flags, 0o644)
        with os.fdopen(descriptor, "ab" if append else "wb") as output:
            shutil.copyfileobj(response, output)

    if not is_complete_file(part, expected_size):
        actual = part.stat().st_size if part.exists() else 0
        raise RuntimeError(
            f"incomplete download for {destination.name}: got {actual} bytes"
        )
    part.replace(destination)


def fetch_llvm_prebuilt(versions: dict[str, str], prefix: pathlib.Path) -> pathlib.Path:
    version = versions["WAFER_LLVM_VERSION"]
    url = versions["WAFER_LLVM_LINUX_X64_URL"]
    if not url:
        raise RuntimeError(
            "this dependency stack does not provide a pinned LLVM prebuilt; "
            "use --llvm-source and build/install llvm-project at the pinned commit"
        )
    llvm_root = prefix / "llvm" / version
    mlir_config = llvm_root / "lib" / "cmake" / "mlir" / "MLIRConfig.cmake"
    if mlir_config.exists():
        return llvm_root

    downloads = prefix / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / pathlib.Path(url).name
    download_with_resume(url, archive)

    extract_dir = prefix / "llvm" / f"extract-{version}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(extract_dir)

    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise RuntimeError(f"expected one LLVM directory in {extract_dir}, got {children}")
    llvm_root.parent.mkdir(parents=True, exist_ok=True)
    if llvm_root.exists() or llvm_root.is_symlink():
        llvm_root.unlink()
    llvm_root.symlink_to(children[0], target_is_directory=True)
    return llvm_root


def run_logged(
    command: list[str],
    log: pathlib.Path,
    *,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    append: bool = False,
) -> None:
    print("+", " ".join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a" if append else "w", encoding="utf-8") as output:
        output.write("+ " + " ".join(command) + "\n")
        output.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
        )


def run_conformance_gate(
    name: str,
    *,
    numeric_root: pathlib.Path,
    conformance_root: pathlib.Path,
    contracts: dict[str, dict[str, object]],
    environments: dict[str, dict[str, str]],
) -> dict[str, object]:
    contract = contracts[name]
    command = contract["command"]
    cwd = contract["cwd"]
    environment_name = contract["environment"]
    if (
        not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or not isinstance(cwd, str)
        or not isinstance(environment_name, str)
    ):
        raise RuntimeError(f"internal numeric gate contract is invalid: {name}")
    actual_command = [expand_numeric_root(item, numeric_root) for item in command]
    actual_cwd = pathlib.Path(expand_numeric_root(cwd, numeric_root))
    reject_symlink_ancestors(actual_cwd)
    actual_environment = expand_environment(
        environments[environment_name], numeric_root
    )
    log = conformance_root / f"{name}.log"
    run_logged(
        actual_command,
        log,
        cwd=actual_cwd,
        env=actual_environment,
    )
    return {
        "name": name,
        "command": command,
        "cwd": cwd,
        "environment": environment_name,
        "exit_code": 0,
        "log_path": log,
    }


def fetch_numeric_archive(pin: NumericPin, downloads: pathlib.Path) -> pathlib.Path:
    downloads = prepare_managed_directory(downloads)
    archive = downloads / pin.archive_name
    reject_symlink_ancestors(archive, allow_missing=True)
    part = archive.with_name(archive.name + ".part")
    reject_symlink_ancestors(part, allow_missing=True)
    if archive.exists() and sha256_file(archive) == pin.sha256:
        return archive
    if archive.exists():
        archive.unlink()
    part.unlink(missing_ok=True)
    download_with_resume(pin.url, archive)
    actual = sha256_file(archive)
    if actual != pin.sha256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {pin.name}: expected {pin.sha256}, got {actual}"
        )
    return archive


def _fetch_numeric_sources_unlocked(
    versions: dict[str, str], numeric_root: pathlib.Path
) -> dict[str, tuple[NumericPin, pathlib.Path, pathlib.Path]]:
    """Fetch, checksum, and atomically extract all numeric source archives."""

    numeric_root = prepare_managed_directory(numeric_root)
    downloads = prepare_managed_directory(numeric_root / "downloads")
    sources = prepare_managed_directory(numeric_root / "sources")
    result: dict[str, tuple[NumericPin, pathlib.Path, pathlib.Path]] = {}
    for name, pin in numeric_pins(versions).items():
        archive = fetch_numeric_archive(pin, downloads)
        source = sources / pin.source_directory
        safe_extract_archive(archive, source, pin.source_directory)
        result[name] = (pin, archive, source)
        print(
            f"Verified {pin.name} {pin.version}: {archive.name} {pin.sha256}",
            flush=True,
        )
    return result


def fetch_numeric_sources(
    versions: dict[str, str], numeric_root: pathlib.Path
) -> dict[str, tuple[NumericPin, pathlib.Path, pathlib.Path]]:
    numeric_root = prepare_managed_directory(numeric_root)
    with numeric_build_lock(numeric_root):
        record = numeric_root / "numeric-model-deps.json"
        if record.exists():
            raise RuntimeError(
                "managed numeric source root is immutable after conformance record publication"
            )
        return _fetch_numeric_sources_unlocked(versions, numeric_root)


def clean_directory(path: pathlib.Path, numeric_root: pathlib.Path) -> None:
    numeric_root = reject_symlink_ancestors(numeric_root)
    path = pathlib.Path(os.path.abspath(path))
    reject_symlink_ancestors(path, allow_missing=True)
    try:
        path.relative_to(numeric_root)
    except ValueError as error:
        raise RuntimeError(f"clean path escapes managed numeric root: {path}") from error
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise RuntimeError(f"refusing to clean managed symlink: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            raise RuntimeError(f"managed clean path is not a directory: {path}")
    path.mkdir(parents=True)
    reject_symlink_ancestors(path)


def add_thread_local_to_platform_header(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "#define THREAD_LOCAL" in text:
        if f"#define THREAD_LOCAL {SOFTFLOAT_THREAD_LOCAL}" not in text:
            raise RuntimeError(f"unexpected THREAD_LOCAL policy in {path}")
        return
    text += (
        "\n/* Wafer managed numeric-model state isolation policy. */\n"
        f"#define THREAD_LOCAL {SOFTFLOAT_THREAD_LOCAL}\n"
    )
    path.write_text(text, encoding="utf-8")


def build_softfloat_and_testfloat(
    *,
    softfloat_source: pathlib.Path,
    testfloat_source: pathlib.Path,
    build_root: pathlib.Path,
    install_root: pathlib.Path,
    conformance_root: pathlib.Path,
    numeric_root: pathlib.Path,
    contracts: dict[str, dict[str, object]],
    environments: dict[str, dict[str, str]],
) -> tuple[dict[str, pathlib.Path], list[dict[str, object]]]:
    if platform.system() != "Linux" or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        raise RuntimeError(
            f"managed SoftFloat/TestFloat policy only supports Linux x86_64, got "
            f"{platform.system()} {platform.machine()}"
        )
    softfloat_template = softfloat_source / "build" / SOFTFLOAT_PLATFORM
    softfloat_makefile = softfloat_template / "Makefile"
    make_text = softfloat_makefile.read_text(encoding="utf-8")
    if (
        "SPECIALIZE_TYPE ?=" not in make_text
        or "$(SOURCE_DIR)/$(SPECIALIZE_TYPE)" not in make_text
    ):
        raise RuntimeError("SoftFloat release Makefile cannot enforce SPECIALIZE_TYPE")
    specialization = softfloat_source / "source" / SOFTFLOAT_SPECIALIZATION
    raise_flags = specialization / "softfloat_raiseFlags.c"
    specialize_header = specialization / "specialize.h"
    if (
        not specialization.is_dir()
        or "softfloat_exceptionFlags |= flags"
        not in raise_flags.read_text(encoding="utf-8")
        or "#define init_detectTininess softfloat_tininess_beforeRounding"
        not in specialize_header.read_text(encoding="utf-8")
    ):
        raise RuntimeError(
            "SoftFloat non-trapping/default-tininess specialization is unavailable"
        )

    softfloat_build = build_root / "softfloat"
    clean_directory(softfloat_build, numeric_root)
    shutil.copytree(softfloat_template, softfloat_build, dirs_exist_ok=True)
    add_thread_local_to_platform_header(softfloat_build / "platform.h")
    gates: list[dict[str, object]] = [
        run_conformance_gate(
            "softfloat-build",
            numeric_root=numeric_root,
            conformance_root=conformance_root,
            contracts=contracts,
            environments=environments,
        )
    ]
    softfloat_library = softfloat_build / "softfloat.a"
    if not softfloat_library.is_file():
        raise RuntimeError("SoftFloat build did not produce softfloat.a")

    softfloat_install = install_root / "softfloat"
    clean_directory(softfloat_install, numeric_root)
    (softfloat_install / "lib").mkdir()
    shutil.copy2(softfloat_library, softfloat_install / "lib" / "libsoftfloat.a")
    shutil.copytree(
        softfloat_source / "source" / "include",
        softfloat_install / "include",
    )

    testfloat_template = testfloat_source / "build" / SOFTFLOAT_PLATFORM
    testfloat_build = build_root / "testfloat"
    clean_directory(testfloat_build, numeric_root)
    shutil.copytree(testfloat_template, testfloat_build, dirs_exist_ok=True)
    add_thread_local_to_platform_header(testfloat_build / "platform.h")
    gates.append(
        run_conformance_gate(
            "testfloat-build",
            numeric_root=numeric_root,
            conformance_root=conformance_root,
            contracts=contracts,
            environments=environments,
        )
    )
    testsoftfloat_built = testfloat_build / "testsoftfloat"
    if not testsoftfloat_built.is_file():
        raise RuntimeError("TestFloat build did not produce testsoftfloat")
    testfloat_install = install_root / "testfloat"
    clean_directory(testfloat_install, numeric_root)
    (testfloat_install / "bin").mkdir()
    testsoftfloat = testfloat_install / "bin" / "testsoftfloat"
    shutil.copy2(testsoftfloat_built, testsoftfloat)

    policy_probe_source = softfloat_build / "wafer-softfloat-policy-probe.c"
    policy_probe = softfloat_build / "wafer-softfloat-policy-probe"
    policy_probe_source.write_text(
        r'''#include <pthread.h>
#include <stdint.h>
#include "softfloat.h"

static void *probe_thread(void *unused) {
  (void)unused;
  if (softfloat_roundingMode != softfloat_round_near_even ||
      softfloat_detectTininess != softfloat_tininess_beforeRounding ||
      extF80_roundingPrecision != 80 ||
      softfloat_exceptionFlags != 0) return (void *)(uintptr_t)1;
  softfloat_roundingMode = softfloat_round_max;
  softfloat_detectTininess = softfloat_tininess_afterRounding;
  extF80_roundingPrecision = 32;
  softfloat_raiseFlags(softfloat_flag_overflow);
  if (softfloat_roundingMode != softfloat_round_max ||
      softfloat_detectTininess != softfloat_tininess_afterRounding ||
      extF80_roundingPrecision != 32 ||
      softfloat_exceptionFlags != softfloat_flag_overflow)
    return (void *)(uintptr_t)2;
  return 0;
}

int main(void) {
  pthread_t thread;
  softfloat_roundingMode = softfloat_round_min;
  softfloat_detectTininess = softfloat_tininess_afterRounding;
  extF80_roundingPrecision = 64;
  softfloat_exceptionFlags = softfloat_flag_inexact;
  if (pthread_create(&thread, 0, probe_thread, 0)) return 2;
  void *thread_result = 0;
  if (pthread_join(thread, &thread_result) || thread_result) return 3;
  if (softfloat_roundingMode != softfloat_round_min ||
      softfloat_detectTininess != softfloat_tininess_afterRounding ||
      extF80_roundingPrecision != 64 ||
      softfloat_exceptionFlags != softfloat_flag_inexact) return 4;
  softfloat_roundingMode = softfloat_round_near_even;
  softfloat_detectTininess = softfloat_tininess_beforeRounding;
  extF80_roundingPrecision = 80;
  softfloat_exceptionFlags = 0;
  float32_t signaling_nan = { UINT32_C(0x7f800001) };
  float32_t one = { UINT32_C(0x3f800000) };
  if (f32_add(signaling_nan, one).v != UINT32_C(0x7fc00000)) return 5;
  if (!(softfloat_exceptionFlags & softfloat_flag_invalid)) return 6;
  softfloat_exceptionFlags = 0;
  softfloat_raiseFlags(softfloat_flag_overflow);
  if (softfloat_exceptionFlags != softfloat_flag_overflow) return 7;
  return 0;
}
''',
        encoding="utf-8",
    )
    gates.append(
        run_conformance_gate(
            "softfloat-policy-compile",
            numeric_root=numeric_root,
            conformance_root=conformance_root,
            contracts=contracts,
            environments=environments,
        )
    )
    gates.append(
        run_conformance_gate(
            "softfloat-tls-default-nan",
            numeric_root=numeric_root,
            conformance_root=conformance_root,
            contracts=contracts,
            environments=environments,
        )
    )
    tests = {
        "testsoftfloat-f16-mulAdd": "f16_mulAdd",
        "testsoftfloat-f32-mulAdd": "f32_mulAdd",
        "testsoftfloat-all1": "-all1",
        "testsoftfloat-all2": "-all2",
    }
    for name in tests:
        gates.append(
            run_conformance_gate(
                name,
                numeric_root=numeric_root,
                conformance_root=conformance_root,
                contracts=contracts,
                environments=environments,
            )
        )

    artifacts = {
        "softfloat": softfloat_install / "lib" / "libsoftfloat.a",
        "softfloat-header": softfloat_install / "include" / "softfloat.h",
        "softfloat-types-header": softfloat_install / "include" / "softfloat_types.h",
        "testsoftfloat": testsoftfloat,
    }
    return artifacts, gates


def find_versioned_shared_object(directory: pathlib.Path, stem: str) -> pathlib.Path:
    candidates = sorted(
        path
        for path in directory.glob(f"lib{stem}.so.*")
        if path.is_file() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one real versioned lib{stem} shared object in "
            f"{directory}, got {candidates}"
        )
    return candidates[0]


def materialize_loader_soname(
    directory: pathlib.Path, stem: str, library: pathlib.Path
) -> pathlib.Path:
    candidates = sorted(
        path
        for path in directory.glob(f"lib{stem}.so.*")
        if path.is_symlink() and path.resolve(strict=True) == library.resolve(strict=True)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one loader-facing lib{stem} SONAME symlink in "
            f"{directory}, got {candidates}"
        )
    soname = candidates[0]
    soname.unlink()
    try:
        os.link(library, soname)
    except OSError:
        shutil.copy2(library, soname)
    if soname.is_symlink() or sha256_file(soname) != sha256_file(library):
        raise RuntimeError(f"failed to materialize managed lib{stem} SONAME artifact")
    return soname


def copy_numeric_licenses(
    sources: dict[str, tuple[NumericPin, pathlib.Path, pathlib.Path]],
    install_root: pathlib.Path,
) -> tuple[dict[str, pathlib.Path], dict[str, dict[str, str]]]:
    licenses = prepare_managed_directory(install_root / "licenses")
    result: dict[str, pathlib.Path] = {}
    records: dict[str, dict[str, str]] = {}
    numeric_root = install_root.parent
    for name, (dependency, source_relative) in LICENSE_SOURCE_FILES.items():
        source = sources[dependency][2] / source_relative
        if not source.is_file():
            raise RuntimeError(f"numeric dependency license is missing: {source}")
        destination = licenses / f"{name.removeprefix('license-')}.txt"
        shutil.copy2(source, destination)
        digest = sha256_file(source)
        if sha256_file(destination) != digest:
            raise RuntimeError(f"numeric dependency license copy mismatch: {name}")
        result[name] = destination
        records[name] = {
            "dependency": dependency,
            "source": relative_managed_path(numeric_root, source),
            "artifact": name,
            "sha256": digest,
        }
    return result, records


def build_numeric_model_dependencies(
    versions: dict[str, str], numeric_root: pathlib.Path, jobs: int
) -> pathlib.Path:
    """Clean-build and publish one fully conformance-tested dependency record."""

    if jobs < 1:
        raise RuntimeError("numeric dependency job count must be positive")
    numeric_root = prepare_managed_directory(numeric_root)
    record_path = numeric_root / "numeric-model-deps.json"
    with numeric_build_lock(numeric_root):
        reject_symlink_ancestors(record_path, allow_missing=True)
        if record_path.exists():
            validate_numeric_record(record_path, numeric_root, versions)
            print(
                f"Numeric model dependency record already published and valid: {record_path}",
                flush=True,
            )
            return record_path

        candidate_record = (
            numeric_root / f".numeric-model-deps.candidate-{uuid.uuid4().hex}.json"
        )
        reject_symlink_ancestors(candidate_record, allow_missing=True)
        sources = _fetch_numeric_sources_unlocked(versions, numeric_root)
        build_root = numeric_root / "build"
        install_root = numeric_root / "install"
        conformance_root = numeric_root / "conformance"
        clean_directory(build_root, numeric_root)
        clean_directory(install_root, numeric_root)
        clean_directory(conformance_root, numeric_root)
        prepare_managed_directory(build_root / "home")
        prepare_managed_directory(build_root / "tmp")

        toolchain = capture_host_toolchain()
        environments = numeric_build_environments(toolchain)
        contracts = numeric_gate_contracts(jobs=jobs, toolchain=toolchain)

        gates: list[dict[str, object]] = []
        for dependency in ("m4", "gmp", "mpfr"):
            prepare_managed_directory(build_root / dependency)
            for action in ("configure", "build", "check", "install"):
                gates.append(
                    run_conformance_gate(
                        f"{dependency}-{action}",
                        numeric_root=numeric_root,
                        conformance_root=conformance_root,
                        contracts=contracts,
                        environments=environments,
                    )
                )
            if dependency == "m4":
                gates.append(
                    run_conformance_gate(
                        "m4-version",
                        numeric_root=numeric_root,
                        conformance_root=conformance_root,
                        contracts=contracts,
                        environments=environments,
                    )
                )

        managed_m4 = install_root / "m4" / "bin" / "m4"
        if not managed_m4.is_file() or not os.access(managed_m4, os.X_OK):
            raise RuntimeError("managed m4 install did not produce executable bin/m4")

        soft_artifacts, soft_gates = build_softfloat_and_testfloat(
            softfloat_source=sources["softfloat"][2],
            testfloat_source=sources["testfloat"][2],
            build_root=build_root,
            install_root=install_root,
            conformance_root=conformance_root,
            numeric_root=numeric_root,
            contracts=contracts,
            environments=environments,
        )
        gates.extend(soft_gates)

        gmp_prefix = install_root / "gmp"
        mpfr_prefix = install_root / "mpfr"
        gmp_library = find_versioned_shared_object(gmp_prefix / "lib", "gmp")
        mpfr_library = find_versioned_shared_object(mpfr_prefix / "lib", "mpfr")
        if gmp_library.name != "libgmp.so.10.5.0":
            raise RuntimeError(f"unexpected GMP shared library identity: {gmp_library}")
        if mpfr_library.name != "libmpfr.so.6.2.2":
            raise RuntimeError(f"unexpected MPFR shared library identity: {mpfr_library}")
        gmp_soname = materialize_loader_soname(
            gmp_prefix / "lib", "gmp", gmp_library
        )
        mpfr_soname = materialize_loader_soname(
            mpfr_prefix / "lib", "mpfr", mpfr_library
        )

        version_probe_source = build_root / "managed-version-thread-safe.c"
        version_probe_source.write_text(
            f'''#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "gmp.h"
#include "mpfr.h"

static mpfr_exp_t main_emin_value;
static mpfr_exp_t thread_emin_value;

static void *probe_thread(void *unused) {{
  (void)unused;
  if (mpfr_get_default_prec() == 113 ||
      mpfr_get_emin() == main_emin_value || mpfr_overflow_p())
    return (void *)(uintptr_t)1;
  mpfr_set_default_prec(211);
  if (mpfr_set_emin(thread_emin_value)) return (void *)(uintptr_t)2;
  mpfr_flags_clear(MPFR_FLAGS_ALL);
  mpfr_set_underflow();
  if (mpfr_get_default_prec() != 211 ||
      mpfr_get_emin() != thread_emin_value || !mpfr_underflow_p() ||
      mpfr_overflow_p())
    return (void *)(uintptr_t)3;
  return 0;
}}

int main(void) {{
  if (strcmp(gmp_version, "{sources['gmp'][0].version}") ||
      __GNU_MP_VERSION != 6 || __GNU_MP_VERSION_MINOR != 3 ||
      __GNU_MP_VERSION_PATCHLEVEL != 0 ||
      strcmp(mpfr_get_version(), "{sources['mpfr'][0].version}") ||
      strcmp(mpfr_get_version(), MPFR_VERSION_STRING) ||
      strcmp(mpfr_get_patches(), "") || !mpfr_buildopt_tls_p())
    return 1;
  mpfr_prec_t original_prec = mpfr_get_default_prec();
  mpfr_exp_t original_emin = mpfr_get_emin();
  mpfr_flags_t original_flags = mpfr_flags_save();
  main_emin_value = original_emin > mpfr_get_emin_min() + 2
                      ? original_emin - 1 : original_emin + 1;
  thread_emin_value = main_emin_value > mpfr_get_emin_min() + 2
                        ? main_emin_value - 1 : main_emin_value + 1;
  mpfr_set_default_prec(113);
  if (mpfr_set_emin(main_emin_value)) return 2;
  mpfr_flags_clear(MPFR_FLAGS_ALL);
  mpfr_set_overflow();
  pthread_t thread;
  if (pthread_create(&thread, 0, probe_thread, 0)) return 3;
  void *thread_result = 0;
  if (pthread_join(thread, &thread_result) || thread_result) return 4;
  if (mpfr_get_default_prec() != 113 ||
      mpfr_get_emin() != main_emin_value || !mpfr_overflow_p() ||
      mpfr_underflow_p())
    return 5;
  mpfr_set_default_prec(original_prec);
  if (mpfr_set_emin(original_emin)) return 6;
  mpfr_flags_restore(original_flags, MPFR_FLAGS_ALL);
  printf("gmp=%s\\nmpfr=%s\\npatches=%s\\ntls=%d\\n",
         gmp_version, mpfr_get_version(), mpfr_get_patches(),
         mpfr_buildopt_tls_p());
  return 0;
}}
''',
            encoding="utf-8",
        )
        gates.append(
            run_conformance_gate(
                "managed-version-thread-safe-compile",
                numeric_root=numeric_root,
                conformance_root=conformance_root,
                contracts=contracts,
                environments=environments,
            )
        )
        gates.append(
            run_conformance_gate(
                "managed-version-thread-safe",
                numeric_root=numeric_root,
                conformance_root=conformance_root,
                contracts=contracts,
                environments=environments,
            )
        )

        license_artifacts, license_records = copy_numeric_licenses(
            sources, install_root
        )
        artifacts: dict[str, pathlib.Path] = {
            "m4": managed_m4,
            **soft_artifacts,
            "gmp": gmp_library,
            "gmp-soname": gmp_soname,
            "gmp-header": gmp_prefix / "include" / "gmp.h",
            "mpfr": mpfr_library,
            "mpfr-soname": mpfr_soname,
            "mpfr-header": mpfr_prefix / "include" / "mpfr.h",
            **license_artifacts,
        }
        pin_records = {}
        for name, (pin, archive, source) in sources.items():
            pin_records[name] = {
                "version": pin.version,
                "url": pin.url,
                "archive_sha256": pin.sha256,
                "archive": relative_managed_path(numeric_root, archive),
                "source": relative_managed_path(numeric_root, source),
                "source_tree_sha256": sha256_tree(source),
            }

        readelf = pathlib.Path(str(toolchain["tools"]["readelf"]["path"]))
        gate_records = []
        for gate in gates:
            log_path = gate["log_path"]
            if not isinstance(log_path, pathlib.Path):
                raise RuntimeError("internal numeric conformance log path is invalid")
            gate_records.append(
                {
                    **{key: value for key, value in gate.items() if key != "log_path"},
                    "log": artifact_identity(
                        numeric_root, log_path, readelf=readelf
                    ),
                }
            )

        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
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
                "environments": environments,
                "mpfr_patches": "",
                "elf_identity_policy": "sha256-build-id-soname-needed-rpath-v1",
            },
            "artifacts": {
                name: artifact_identity(numeric_root, path, readelf=readelf)
                for name, path in sorted(artifacts.items())
            },
            "licenses": license_records,
            "conformance": {
                "policy": CONFORMANCE_POLICY,
                "gates": sorted(gate_records, key=lambda gate: str(gate["name"])),
            },
        }
        try:
            atomic_write_json(candidate_record, record)
            validate_numeric_record(candidate_record, numeric_root, versions)
            publish_file_noreplace(candidate_record, record_path)
        finally:
            candidate_record.unlink(missing_ok=True)
        print(f"Numeric model dependency record published: {record_path}", flush=True)
        return record_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=str(REPO_ROOT / "third_party"))
    parser.add_argument("--python", action="store_true", help="install pinned Python dev tools")
    parser.add_argument("--llvm", action="store_true", help="download pinned LLVM/MLIR prebuilt when available")
    parser.add_argument("--llvm-source", action="store_true", help="sync pinned llvm-project source submodule")
    parser.add_argument(
        "--importer-sources",
        action="store_true",
        help="sync PyTorch/XLA baseline, StableHLO, Shardy, and OpenXLA/XLA source submodules",
    )
    parser.add_argument(
        "--importer-python",
        action="store_true",
        help="install pinned importer Python packages for the PyTorch/XLA source build; build/install torch_xla from third_party/pytorch-xla source separately",
    )
    parser.add_argument("--test-sources", action="store_true", help="sync pinned googletest submodule")
    parser.add_argument(
        "--numeric-model-sources",
        action="store_true",
        help=(
            "explicitly fetch, checksum, and safely extract pinned "
            "numeric-model source archives"
        ),
    )
    parser.add_argument(
        "--numeric-model-deps",
        action="store_true",
        help=(
            "clean-build and fully self-test managed numeric-model dependencies, "
            "then atomically publish their conformance record"
        ),
    )
    parser.add_argument(
        "--numeric-model-root",
        help="managed numeric-model root (defaults to <prefix>/numeric-model)",
    )
    parser.add_argument(
        "--numeric-jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="parallel make job count for --numeric-model-deps",
    )
    parser.add_argument("--all", action="store_true", help="fetch every pinned dependency")
    args = parser.parse_args()

    prefix = pathlib.Path(args.prefix).absolute()
    prefix.mkdir(parents=True, exist_ok=True)
    numeric_root = (
        pathlib.Path(args.numeric_model_root).absolute()
        if args.numeric_model_root
        else prefix / "numeric-model"
    )
    if args.numeric_model_deps or args.numeric_model_sources or args.all:
        prepare_managed_directory(numeric_root)
    versions = load_versions()

    if args.all or args.python:
        python = ensure_venv(prefix, "python", REPO_ROOT / "requirements-dev.txt")
        print(f"Python tools installed: {python}")

    if args.all or args.importer_python:
        importer_python = ensure_venv(prefix, "python-importer", REPO_ROOT / "requirements-importer.txt")
        print(f"Importer Python tools installed: {importer_python}")

    if args.llvm:
        llvm_root = fetch_llvm_prebuilt(versions, prefix)
        print(f"LLVM/MLIR installed: {llvm_root}")
        print(f"Configure with: -DMLIR_DIR={llvm_root / 'lib/cmake/mlir'}")

    if args.all or args.llvm_source:
        sync_submodule(REPO_ROOT / "third_party" / "llvm-project", versions["WAFER_LLVM_COMMIT"])
        print(f"LLVM/MLIR source installed under: {REPO_ROOT / 'third_party' / 'llvm-project'}")

    if args.all or args.importer_sources:
        sync_submodule(prefix / "pytorch-xla", versions["WAFER_PYTORCH_XLA_COMMIT"])
        sync_submodule(prefix / "stablehlo", versions["WAFER_STABLEHLO_COMMIT"])
        sync_submodule(prefix / "shardy", versions["WAFER_SHARDY_COMMIT"])
        sync_submodule(prefix / "xla", versions["WAFER_OPENXLA_XLA_COMMIT"])
        print(f"Compiler source dependencies installed under: {prefix}")

    if args.all or args.test_sources:
        sync_submodule(prefix / "googletest", versions["WAFER_GOOGLETEST_COMMIT"])
        print(f"Test sources installed under: {prefix}")

    if args.numeric_model_deps:
        build_numeric_model_dependencies(versions, numeric_root, args.numeric_jobs)
    elif args.all or args.numeric_model_sources:
        fetch_numeric_sources(versions, numeric_root)
        print(f"Numeric model sources installed under: {numeric_root / 'sources'}")

    if not (
        args.all
        or args.python
        or args.llvm
        or args.llvm_source
        or args.importer_sources
        or args.importer_python
        or args.test_sources
        or args.numeric_model_sources
        or args.numeric_model_deps
    ):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
