#!/usr/bin/env python3
"""Check repository-wide Wafer source ownership without duplicating semantics."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


# Sources intentionally retained for a current numbered task but not compiled.
# The design and deletion gates live in tasks/18-source-organization.md; this
# list only makes the filesystem/CMake distinction fail closed.
DORMANT_LIBRARY_SOURCES = {
    "lib/Wafer/Analysis/PhysicalDataflow/GlobalTileRelation.cpp": "Q50.H",
    "lib/Wafer/Analysis/Topology/CollectiveTopologyAnalysis.cpp": "Q50.H",
    "lib/Wafer/Analysis/Topology/ExecutionTopologyAnalysis.cpp": "Q50.H",
    "lib/Wafer/Conversion/WaferTileRegionToInstr/CollectiveLowering.cpp": "Q50.H",
    "lib/Wafer/Transforms/PhysicalDataflow/CompleteRankMaterialization.cpp": "Q50.E/H",
    "lib/Wafer/Transforms/PhysicalDataflow/LayoutMovementOptimization.cpp": "Q50.G/H",
    "lib/Wafer/Transforms/PhysicalDataflow/ReadyOrder.cpp": "Q50.J",
    "lib/Wafer/Transforms/PhysicalDataflow/StructuredOpInterfaceModels.cpp": "Q50.J",
    "lib/Wafer/Transforms/Scheduling/FixedSlotPipeline.cpp": "Q50.K",
    "lib/Wafer/Transforms/Scheduling/RankTileMaterialization.cpp": "Q50.E/H",
    "lib/Wafer/Transforms/Scheduling/WorkerPlacement.cpp": "Q50.J",
    "lib/Wafer/Transforms/Transport/CoordinatedCommunicationAction.cpp": "Q50.H",
    "lib/Wafer/Transforms/Transport/NoCCommunicationAction.cpp": "Q50.H",
    "lib/Wafer/Transforms/Transport/NoCIntermediateRouting.cpp": "Q50.H",
    "lib/Wafer/Transforms/Transport/NoCPartialDataflow.cpp": "Q50.H",
}

# These translation units are inputs to the explicitly invoked XLA helper
# build, not host CMake targets. tools/build_xla_spmd_partitioner_helper.py
# owns their exact source manifest and copies them into the external build.
EXTERNAL_HELPER_SOURCES = {
    f"lib/Wafer/Transforms/SPMD/{name}.cpp"
    for name in (
        "XlaSpmdBoundary",
        "XlaSpmdDriver",
        "XlaSpmdFilesystem",
        "XlaSpmdMetadata",
        "XlaSpmdPartitionerMain",
        "XlaSpmdPartitioning",
        "XlaSpmdPayload",
        "XlaSpmdProgram",
    )
}

DORMANT_UNIT_SOURCES = {
    "unittests/Analysis/PhysicalDataflow/GlobalTileRelationTest.cpp": "Q50.H",
    "unittests/Analysis/Topology/ExecutionTopologyAnalysisTest.cpp": "Q50.H",
    "unittests/Conversion/CollectiveCompletionTest.cpp": "Q50.H/Q63",
    "unittests/Conversion/CollectiveTopologyTest.cpp": "Q50.H",
    "unittests/Conversion/CommunicationAlternativesTest.cpp": "Q50.H",
    "unittests/Transforms/PhysicalDataflow/CandidateRewritesTest.cpp": "Q50.G/H",
    "unittests/Transforms/PhysicalDataflow/CompleteRankMaterializationTest.cpp": "Q50.E/H",
    "unittests/Transforms/PhysicalDataflow/ReadyOrderTest.cpp": "Q50.J",
    "unittests/Transforms/Scheduling/FixedSlotPipelineTest.cpp": "Q50.K",
    "unittests/Transforms/Scheduling/RankTileMaterializationTest.cpp": "Q50.E/H",
    "unittests/Transforms/Scheduling/WorkerPlacementTest.cpp": "Q50.J",
    "unittests/Transforms/Transport/NoCCommunicationActionTest.cpp": "Q50.H",
    "unittests/Transforms/Transport/NoCIntermediateRoutingTest.cpp": "Q50.H",
    "unittests/Transforms/Transport/NoCPartialDataflowTest.cpp": "Q50.H",
    "unittests/Model/SystemC/SystemCTargetModelDTEComputeAccessTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelDTEIntegrationTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelDTELateJoinTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelDTEMismatchTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelDTEPreIssueTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelDTEReadinessTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelEventFailureTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelNCCVisibilityTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelNoProgressTest.cpp": "Q63",
    "unittests/Model/SystemC/SystemCTargetModelTestSupport.cpp": "Q63",
}

RETIRED_PATHS = (
    "include/Wafer/Compiler",
    "include/Wafer/Pipelines",
    "lib/Wafer/Compiler",
    "lib/Wafer/Pipelines",
    "unittests/Compiler",
    "include/Wafer/IR/Tensor/GroupOps.td",
    "lib/Wafer/IR/Tensor/GroupOps.cpp",
    "lib/Wafer/Transforms/Group",
    "lib/Wafer/Conversion/WaferGroupToTileRegion",
)

ALLOWED_LIBRARY_DIRECTORIES = {
    "Analysis",
    "CodeGen",
    "Conversion",
    "Driver",
    "Frontend",
    "IR",
    "Model",
    "Package",
    "Planning",
    "Program",
    "Runtime",
    "Target",
    "Transforms",
}


def cmake_files(root: Path) -> list[Path]:
    result = [root / "CMakeLists.txt"]
    for directory in ("cmake", "lib", "runtime", "test", "tools", "unittests"):
        base = root / directory
        if not base.is_dir():
            continue
        result.extend(base.rglob("CMakeLists.txt"))
        result.extend(base.rglob("*.cmake"))
    return sorted(set(path for path in result if "third_party" not in path.parts))


def strip_cmake_comments(text: str) -> str:
    return re.sub(r"(?m)#.*$", "", text)


def resolve_source(root: Path, manifest: Path, token: str) -> str | None:
    token = token.strip('"')
    source_prefix = "${CMAKE_SOURCE_DIR}/"
    if token.startswith(source_prefix):
        path = root / token[len(source_prefix) :]
    elif "${" in token or "$<" in token:
        return None
    else:
        path = manifest.parent / token
        owner_relative = manifest.parent.parent / token
        if not path.exists() and owner_relative.exists():
            path = owner_relative
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def source_occurrences(root: Path) -> tuple[dict[str, list[Path]], set[str]]:
    occurrences: dict[str, list[Path]] = defaultdict(list)
    optional: set[str] = set()
    for manifest in cmake_files(root):
        text = strip_cmake_comments(manifest.read_text(encoding="utf-8"))
        for token in re.findall(r'[^\s()"]+\.cpp|"[^"]+\.cpp"', text):
            source = resolve_source(root, manifest, token)
            if source:
                occurrences[source].append(manifest)
        for body in re.findall(
            r"set\s*\(\s*LLVM_OPTIONAL_SOURCES\b(.*?)\)", text, re.S
        ):
            for token in re.findall(r'[^\s()"]+\.cpp|"[^"]+\.cpp"', body):
                source = resolve_source(root, manifest, token)
                if source:
                    optional.add(source)
    return occurrences, optional


def check_library_sources(root: Path, errors: list[str]) -> None:
    occurrences, optional = source_occurrences(root)
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "lib/Wafer").rglob("*.cpp")
    }
    declared_dormant = set(DORMANT_LIBRARY_SOURCES)
    declared_external = set(EXTERNAL_HELPER_SOURCES)
    for source in sorted(actual):
        active_manifests = [] if source in optional else occurrences.get(source, [])
        if active_manifests:
            if len(active_manifests) != 1:
                errors.append(
                    f"active library source has {len(active_manifests)} CMake owners: {source}"
                )
            if source in declared_dormant or source in declared_external:
                errors.append(f"source is both active and non-CMake-owned: {source}")
            continue
        if source not in declared_dormant and source not in declared_external:
            errors.append(f"library source has no active or task-owned disposition: {source}")
    for source in sorted(declared_dormant | declared_external):
        if source not in actual:
            errors.append(f"declared non-CMake source is missing: {source}")


def check_unit_sources(root: Path, errors: list[str]) -> None:
    occurrences, _ = source_occurrences(root)
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "unittests").rglob("*.cpp")
    }
    dormant = set(DORMANT_UNIT_SOURCES)
    for source in sorted(actual):
        active = occurrences.get(source, [])
        if active:
            if source in dormant:
                errors.append(f"unit source is both active and dormant: {source}")
            continue
        if source not in dormant:
            errors.append(f"unit source has no CMake registration or task owner: {source}")
    for source in sorted(dormant):
        if source not in actual:
            errors.append(f"declared dormant unit source is missing: {source}")


def check_directory_owners(root: Path, errors: list[str]) -> None:
    for relative in RETIRED_PATHS:
        if (root / relative).exists():
            errors.append(f"retired source owner still exists: {relative}")
    actual = {
        path.name
        for path in (root / "lib/Wafer").iterdir()
        if path.is_dir()
    }
    unexpected = sorted(actual - ALLOWED_LIBRARY_DIRECTORIES)
    if unexpected:
        errors.append("unexpected top-level library owner(s): " + ", ".join(unexpected))


def check_no_retired_includes(root: Path, errors: list[str]) -> None:
    needles = ("Wafer/Compiler/", "Wafer/Pipelines/")
    suffixes = {".h", ".hpp", ".c", ".cc", ".cpp", ".td"}
    for directory in ("include", "lib", "runtime", "tools", "unittests"):
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in text:
                    errors.append(f"{path.relative_to(root)} retains {needle}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    check_directory_owners(root, errors)
    check_no_retired_includes(root, errors)
    check_library_sources(root, errors)
    check_unit_sources(root, errors)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Wafer source organization checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
