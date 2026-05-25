#!/usr/bin/env python3
"""Fetch pinned development dependencies for the Wafer compiler prototype."""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSIONS_FILE = REPO_ROOT / "cmake" / "WaferDependencyVersions.cmake"


def load_versions() -> dict[str, str]:
    text = VERSIONS_FILE.read_text(encoding="utf-8")
    versions: dict[str, str] = {}
    for name, value in re.findall(r'set\((WAFER_[A-Z0-9_]+)\s+"([^"]+)"\)', text):
        versions[name] = value
    return versions


def run(command: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_venv(prefix: pathlib.Path) -> pathlib.Path:
    venv = prefix / "python"
    python = venv / "bin" / "python"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(venv)])
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements-dev.txt")])
    return python


def clone_or_update(repo: str, commit: str, destination: pathlib.Path) -> None:
    if not destination.exists():
        destination.mkdir(parents=True)
        run(["git", "init"], cwd=destination)
        run(["git", "remote", "add", "origin", repo], cwd=destination)
    run(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
    run(["git", "checkout", "--detach", commit], cwd=destination)


def fetch_llvm_prebuilt(versions: dict[str, str], prefix: pathlib.Path) -> pathlib.Path:
    version = versions["WAFER_LLVM_VERSION"]
    url = versions["WAFER_LLVM_LINUX_X64_URL"]
    llvm_root = prefix / "llvm" / version
    mlir_config = llvm_root / "lib" / "cmake" / "mlir" / "MLIRConfig.cmake"
    if mlir_config.exists():
        return llvm_root

    downloads = prefix / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / pathlib.Path(url).name
    if not archive.exists():
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=str(REPO_ROOT / ".deps"))
    parser.add_argument("--python", action="store_true", help="install pinned Python dev tools")
    parser.add_argument("--llvm", action="store_true", help="download pinned LLVM/MLIR prebuilt")
    parser.add_argument("--importer-sources", action="store_true", help="clone StableHLO and Shardy pins")
    parser.add_argument("--all", action="store_true", help="fetch every pinned dependency")
    args = parser.parse_args()

    prefix = pathlib.Path(args.prefix).resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    versions = load_versions()

    if args.all or args.python:
        python = ensure_venv(prefix)
        print(f"Python tools installed: {python}")

    if args.all or args.llvm:
        llvm_root = fetch_llvm_prebuilt(versions, prefix)
        print(f"LLVM/MLIR installed: {llvm_root}")
        print(f"Configure with: -DMLIR_DIR={llvm_root / 'lib/cmake/mlir'}")

    if args.all or args.importer_sources:
        src = prefix / "src"
        clone_or_update(
            versions["WAFER_STABLEHLO_REPOSITORY"],
            versions["WAFER_STABLEHLO_COMMIT"],
            src / "stablehlo",
        )
        clone_or_update(
            versions["WAFER_SHARDY_REPOSITORY"],
            versions["WAFER_SHARDY_COMMIT"],
            src / "shardy",
        )
        print(f"Importer sources installed under: {src}")

    if not (args.all or args.python or args.llvm or args.importer_sources):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
