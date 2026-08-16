#!/usr/bin/env python3
"""Check stable Wafer production-source ownership boundaries."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


INSTRUCTION_FAMILY_SOURCES = (
    "ComputeOps.cpp",
    "DTEOps.cpp",
    "MovementOps.cpp",
    "PeripheralOps.cpp",
    "SyncOps.cpp",
)
TILE_REGION_TO_INSTR_DORMANT_SOURCES = ("CollectiveLowering.cpp",)
NUMERIC_SEMANTICS_SOURCES = (
    "NumericCapability.cpp",
    "NumericCommand.cpp",
    "NumericProfiles.cpp",
    "NumericSemanticsInternal.cpp",
)
TENSOR_PROGRAM_TO_TILE_REGION_DORMANT_SOURCES = (
    "AttentionSemantics.cpp",
    "CompleteTraversal.cpp",
    "MaterializeFlashAttention.cpp",
    "MaterializeFlashDecoding.cpp",
)
RETIRED_TASK_LOCAL_SELECTION_PATHS = (
    "CandidateAnalysis.cpp",
    "CandidateEvaluation.cpp",
    "CandidateSelection.cpp",
    "ScheduleTensorProgram.cpp",
    "ScheduleTensorProgramInternal.h",
    "StructuredSchedulingScope.cpp",
    "StructuredSchedulingScope.h",
)
RETIRED_RANK_CANDIDATE_PATHS = (
    "lib/Wafer/Compiler/NoCResidentDataflow.cpp",
    "lib/Wafer/Compiler/NoCResidentDataflow.h",
    "lib/Wafer/Compiler/WholeVariantAttemptPlan.cpp",
    "lib/Wafer/Compiler/WholeVariantAttemptPlan.h",
    "lib/Wafer/Compiler/WholeVariantCoordinator.h",
)
MEMORY_PLANNING_SOURCES = (
    "LifetimeAnalysis.cpp",
    "MiniMallocPacking.cpp",
    "StaticIndexRange.cpp",
    "StaticMemoryPacking.cpp",
)
MEMORY_PLANNING_HEADERS = (
    "LifetimeAnalysis.h",
    "MiniMallocPacking.h",
    "StaticIndexRange.h",
    "StaticMemoryPacking.h",
)
MEMORY_PLANNING_TEST_SOURCES = (
    "LifetimeAnalysisTest.cpp",
    "MiniMallocPackingTest.cpp",
    "StaticMemoryPackingTest.cpp",
)
TARGET_LLVM_SOURCES = (
    "ComputeTargetCallLowering.cpp",
    "DirectDTETargetCallLowering.cpp",
    "InstructionTargetCallLowering.cpp",
    "LowerInstrToTargetLLVM.cpp",
    "MovementTargetCallLowering.cpp",
    "PeripheralTargetCallLowering.cpp",
    "SyncTargetCallLowering.cpp",
    "TargetCallLoweringSupport.cpp",
    "TargetLoweringVerification.cpp",
    "TargetLLVMConversion.cpp",
    "TargetLLVMConversionPatterns.cpp",
    "TargetLLVMStructure.cpp",
)
NUMERIC_DEPENDENCY_SOURCES = (
    "NumericDependencyBuildConfig.cpp",
    "NumericDependencyConformance.cpp",
    "NumericDependencyELF.cpp",
    "NumericDependencyFilesystem.cpp",
    "NumericDependencyGate.cpp",
    "NumericDependencyManifest.cpp",
    "NumericDependencyProcess.cpp",
    "NumericDependencyRuntime.cpp",
)
FRONTEND_PROGRAM_SOURCES = (
    "DistributedBoundary.cpp",
    "DistributedSupport.cpp",
    "FunctionBoundary.cpp",
    "NpyPayload.cpp",
    "ParameterShards.cpp",
    "Program.cpp",
    "ProgramMetadata.cpp",
    "ProgramSupport.cpp",
)
WAFER_COMPILE_SOURCES = (
    "CompilerIRDump.cpp",
    "DriverOptions.cpp",
    "TargetModelGate.cpp",
    "wafer-compile.cpp",
)
REFERENCE_EXECUTOR_LEGACY_PATHS = (
    "include/Wafer/Compiler/ReferenceExecutor.h",
    "lib/Wafer/Compiler/ReferenceExecutor.cpp",
    "lib/Wafer/Compiler/ReferenceExecutorInternal.h",
    "lib/Wafer/Compiler/ReferenceExecutorNumeric.cpp",
    "lib/Wafer/Compiler/ReferenceProgramControlMemory.cpp",
    "lib/Wafer/Compiler/ReferenceProgramDTETransport.cpp",
    "lib/Wafer/Compiler/ReferenceProgramExecution.cpp",
    "lib/Wafer/Compiler/ReferenceProgramInterpreter.cpp",
    "lib/Wafer/Compiler/ReferenceProgramInterpreterInternal.h",
    "lib/Wafer/Compiler/ReferenceProgramMovement.cpp",
    "lib/Wafer/Compiler/ReferenceProgramNumeric.cpp",
    "lib/Wafer/Compiler/ReferenceProgramProjection.cpp",
    "lib/Wafer/Compiler/ReferenceProgramProjectionDTE.cpp",
    "lib/Wafer/Compiler/ReferenceProgramProjectionInternal.h",
    "lib/Wafer/Compiler/ReferenceProgramProjectionMemrefControl.cpp",
    "lib/Wafer/Compiler/ReferenceProgramProjectionMovement.cpp",
    "lib/Wafer/Compiler/ReferenceProgramProjectionNumeric.cpp",
    "tools/wafer-compile/ReferenceGate.cpp",
    "unittests/Compiler/ReferenceExecutorTest.cpp",
)
TARGET_MODEL_KERNEL_SOURCES = (
    "TargetModelControl.cpp",
    "TargetModelKernel.cpp",
    "TargetModelMovement.cpp",
    "TargetModelTensorNumeric.cpp",
    "TargetModelCommandValidation.cpp",
)
FORMAL_NUMERIC_SOURCES = (
    "FormalNumericConvert.cpp",
    "FormalNumericElementwise.cpp",
    "FormalNumericGemm.cpp",
    "FormalNumericSupport.cpp",
    "FormalNumericValidation.cpp",
)
BULK_TENSOR_NUMERIC_SOURCES = (
    "QualifiedBulkExecution.cpp",
    "BulkExecutionEnvironment.cpp",
    "BulkQualificationExecution.cpp",
    "BulkTensorCodec.cpp",
    "BulkTensorNumericSupport.cpp",
    "OneDNNGemmAdapter.cpp",
)
BULK_QUALIFICATION_SOURCES = (
    "BulkQualification.cpp",
    "BulkQualificationCalibration.cpp",
    "BulkQualificationCase.cpp",
    "BulkQualificationComparison.cpp",
    "BulkQualificationPolicy.cpp",
    "BulkQualificationValidation.cpp",
)
COMPILATION_SOURCES = (
    "Compilation.cpp",
    "CompilationOrchestration.cpp",
    "CompilationStages.cpp",
    "TensorProgramCompilation.cpp",
    "ProgramDirectoryTransaction.cpp",
    "SpmdCompilationBridge.cpp",
    "WriteExecutablePackage.cpp",
)
TARGET_CODE_GENERATION_SOURCES = (
    "TargetABIPreparation.cpp",
    "TargetLLVMModule.cpp",
    "TargetModuleLinking.cpp",
    "TargetDeviceLink.cpp",
    "TargetLLVMTranslation.cpp",
    "TargetModuleReadback.cpp",
    "CompileCardExecutableLLVMModules.cpp",
)
PACKAGE_SUPPORT_SOURCES = (
    "PackageManifest.cpp",
    "PackageManifestJson.cpp",
    "PackageManifestReadback.cpp",
    "PackageManifestSerialization.cpp",
    "PackageManifestVerification.cpp",
    "ProfileInstrumentationModel.cpp",
)
RUNTIME_SOURCES = (
    "BoardRuntime.cpp",
    "ProfileInstrumentation.cpp",
    "ProfilerRecord.cpp",
    "RuntimeInvocationPlanning.cpp",
)
WAFER_RUN_SOURCES = (
    "TxBoardRuntime.cpp",
    "WaferProfileCollection.cpp",
    "WaferRunBoardIO.cpp",
    "wafer-run.cpp",
)
RETIRED_PROFILE_SCHEMA_MARKERS = (
    "kProfileInstrumentationVariantsFileName",
    "ProfileVariantPackage",
    "ProfileVariantRole",
    "ProfileVariantSiteMap",
    "getProductionManifestDigest",
    '"variants.json"',
    '"variant_metadata"',
    '"execution_packages"',
    '"variant_id"',
)
STABLEHLO_NORMALIZATION_SOURCES = (
    "ConstantTensorFolding.cpp",
    "NormalizeStablehloCollectives.cpp",
)
XLA_SPMD_HELPER_SOURCES = (
    "XlaSpmdBoundary.cpp",
    "XlaSpmdDriver.cpp",
    "XlaSpmdFilesystem.cpp",
    "XlaSpmdMetadata.cpp",
    "XlaSpmdPartitionerMain.cpp",
    "XlaSpmdPartitioning.cpp",
    "XlaSpmdPayload.cpp",
    "XlaSpmdProgram.cpp",
)
INSTRUCTION_VERIFIER_FILES = (
    "InstructionVerification.cpp",
    "InstructionVerification.h",
)
LIB_WAFER_SUBDIRECTORY_ORDER = (
    "IR",
    "Target",
    "Frontend",
    "Analysis",
    "Conversion",
    "Transforms",
    "Pipelines",
    "Runtime",
    "Compiler",
    "Model",
)
LEGACY_GROUP_PATHS = (
    "include/Wafer/IR/Tensor/GroupOps.td",
    "lib/Wafer/IR/Tensor/GroupOps.cpp",
    "test/Dialect/Wafer/Tensor/Group",
    "include/Wafer/Analysis/Group",
    "lib/Wafer/Analysis/Group",
    "include/Wafer/Conversion/WaferGroupToTileRegion",
    "lib/Wafer/Conversion/WaferGroupToTileRegion",
    "lib/Wafer/Transforms/Group",
    "lib/Wafer/Compiler/GroupedProgramCompilation.cpp",
    "tools/run_heavy_candidate_selection_tests.py",
    "unittests/Conversion/WaferGroupToTileRegionTest.cpp",
)
LEGACY_GROUP_API_PATTERNS = (
    (re.compile(r"(?<![A-Za-z0-9_])wafer\.group\b"), "wafer.group mnemonic"),
    (
        re.compile(r"\b(?:Wafer_)?Group(?:Yield)?Op\b"),
        "wafer.group ODS/C++ op API",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_])(?:Tensor/)?GroupOps\.(?:td|cpp)\b"),
        "wafer.group ODS/source include",
    ),
    (
        re.compile(
            r"\b(?:GroupLayoutPlan|GroupTilingDemand|collectGroupLayoutPlan|"
            r"dumpGroupLayoutPlan|collectGroupTilingDemand|dumpGroupTilingDemand)\b"
        ),
        "retired group analysis API",
    ),
    (
        re.compile(
            r"\b(?:cloneGroupToStandaloneModule|lowerGroupToTileRegionModule|"
            r"lowerCandidateGroupToTileRegionModule|"
            r"lowerCompleteCandidateGroupToTileRegionModule|"
            r"dumpGroupToTileRegionModule)\b"
        ),
        "retired group-to-tile-region API",
    ),
    (
        re.compile(r"\bcompileGroupedProgramToCardExecutable(?:Impl)?\b"),
        "retired grouped-program compiler API",
    ),
    (
        re.compile(
            r"\bcreate(?:FormLogicalGroups|DumpGroupTilingDemand|"
            r"DumpGroupLayoutPlan|DumpGroupToTileRegion|"
            r"DumpCandidateDdrTileViews|SelectGroupTile|"
            r"ConvertGroupToTileRegion)Pass\b"
        ),
        "retired group pass API",
    ),
    (
        re.compile(
            r"\bbuild(?:FormLogicalGroups|LowerGroupsToTileRegion|"
            r"LowerGroupsToInstr|LowerGroupsToMemoryPlannedInstr|"
            r"LowerGroupsToDDRMemoryPlannedInstr|LowerGroupsToTargetLLVM|"
            r"LowerGroupsToSelectedInstr)Pipeline\b"
        ),
        "retired group pipeline API",
    ),
)
LEGACY_GROUP_CLI_NAMES = (
    "wafer-form-logical-groups",
    "wafer-dump-group-tiling-demand",
    "wafer-dump-group-layout-plan",
    "wafer-dump-group-to-tile-region",
    "wafer-dump-candidate-ddr-tile-views",
    "wafer-select-group-tile",
    "wafer-convert-group-to-tile-region",
    "wafer-lower-groups-to-tile-region",
    "wafer-lower-groups-to-instr",
    "wafer-lower-groups-to-memory-planned-instr",
    "wafer-lower-groups-to-ddr-memory-planned-instr",
    "wafer-lower-groups-to-target-llvm",
    "wafer-lower-groups-to-selected-instr",
)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_legacy_group_surfaces_retired(root: Path, errors: list[str]) -> None:
    for relative in LEGACY_GROUP_PATHS:
        path = root / relative
        if path.exists():
            fail(errors, f"retired wafer.group path must not exist: {path}")

    scan_roots = (
        root / "include/Wafer",
        root / "lib/Wafer",
        root / "tools",
        root / "unittests",
        root / "cmake",
    )
    source_suffixes = {
        ".h",
        ".hpp",
        ".cpp",
        ".cc",
        ".td",
        ".inc",
        ".cmake",
        ".py",
    }
    paths = [root / "CMakeLists.txt"]
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        paths.extend(path for path in scan_root.rglob("*") if path.is_file())

    for path in paths:
        if "__pycache__" in path.parts:
            continue
        if path.name in {
            "check_ir_organization.py",
            "check_source_organization.py",
        }:
            continue
        if path.name != "CMakeLists.txt" and path.suffix not in source_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in LEGACY_GROUP_API_PATTERNS:
            if pattern.search(text):
                fail(errors, f"{path} contains {label}")
        for cli_name in LEGACY_GROUP_CLI_NAMES:
            if cli_name in text:
                fail(errors, f"{path} contains retired group CLI/pipeline {cli_name}")


def read_required(path: Path, errors: list[str]) -> str:
    if not path.is_file():
        fail(errors, f"missing required file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def cmake_code(text: str) -> str:
    """Remove line comments while preserving quoted CMake arguments."""
    result: list[str] = []
    in_quote = False
    in_comment = False
    escaped = False
    for character in text:
        if in_comment:
            if character == "\n":
                result.append(character)
                in_comment = False
            continue
        if escaped:
            result.append(character)
            escaped = False
            continue
        if character == "\\" and in_quote:
            result.append(character)
            escaped = True
            continue
        if character == '"':
            result.append(character)
            in_quote = not in_quote
            continue
        if character == "#" and not in_quote:
            in_comment = True
            continue
        result.append(character)
    return "".join(result)


def cmake_target_body(
    text: str, command: str, target: str, cmake_path: Path, errors: list[str]
) -> str:
    text = cmake_code(text)
    match = re.search(
        rf"(?m)^[ \t]*{re.escape(command)}\s*\(\s*{re.escape(target)}\b",
        text,
    )
    if not match:
        fail(errors, f"{cmake_path} missing {command}({target} ...)")
        return ""

    opening = text.find("(", match.start())
    depth = 0
    in_quote = False
    escaped = False
    for index in range(opening, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]

    fail(errors, f"unterminated {command}({target} ...) in {cmake_path}")
    return ""


def check_cmake_sources(
    *,
    body: str,
    required: tuple[str, ...],
    cmake_path: Path,
    target: str,
    prefix: str = "",
    forbidden: tuple[str, ...] = (),
    errors: list[str],
) -> None:
    tokens = body.split()
    for source in required:
        entry = prefix + source
        occurrences = tokens.count(entry)
        if occurrences == 0:
            fail(errors, f"{cmake_path}: {target} missing source {entry}")
        elif occurrences > 1:
            fail(
                errors,
                f"{cmake_path}: {target} lists source {entry} more than once",
            )
    for source in forbidden:
        entry = prefix + source
        if entry in tokens:
            fail(errors, f"{cmake_path}: {target} still lists legacy source {entry}")



def check_cmake_links(
    *,
    body: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
    cmake_path: Path,
    target: str,
    errors: list[str],
) -> None:
    """Check one CMake target's LINK_LIBS boundary: required libraries must be
    present and forbidden ones absent, so dependency direction is enforced."""
    tokens = cmake_tokens(body)
    if "LINK_LIBS" not in tokens:
        fail(errors, f"{cmake_path}: {target} has no LINK_LIBS block")
        return
    link_index = tokens.index("LINK_LIBS")
    links = [
        token
        for token in tokens[link_index + 1 :]
        if token not in ("PUBLIC", "PRIVATE", "INTERFACE")
    ]
    for library in required:
        if library not in links:
            fail(
                errors,
                f"{cmake_path}: {target} must link {library}",
            )
    for library in forbidden:
        if library in links:
            fail(
                errors,
                f"{cmake_path}: {target} must not link {library}",
            )


def cmake_tokens(text: str) -> list[str]:

    return [
        token.strip('"')
        for token in re.findall(r'"(?:\\.|[^"\\])*"|[^\s()]+', cmake_code(text))
    ]


def cmake_target_cpp_source_names(
    *,
    body: str,
    prefix: str,
    cmake_path: Path,
    target: str,
    errors: list[str],
) -> tuple[str, ...]:
    """Return one target directory's active C++ sources from its CMake body."""

    entries = [
        token
        for token in cmake_tokens(body)
        if token.startswith(prefix) and token.endswith(".cpp")
    ]
    duplicates = sorted({entry for entry in entries if entries.count(entry) > 1})
    if duplicates:
        fail(
            errors,
            f"{cmake_path}: {target} lists source more than once: "
            + ", ".join(duplicates),
        )
    return tuple(Path(entry).name for entry in entries)


def check_active_and_dormant_directory_sources(
    *,
    source_root: Path,
    active: tuple[str, ...],
    dormant: tuple[str, ...],
    label: str,
    errors: list[str],
) -> None:
    """Check filesystem sources against CMake truth plus a policy allowlist."""

    active_set = set(active)
    dormant_set = set(dormant)
    overlap = sorted(active_set & dormant_set)
    if overlap:
        fail(errors, f"{label} sources are both active and dormant: {', '.join(overlap)}")
    actual = {path.name for path in source_root.glob("*.cpp") if path.is_file()}
    expected = active_set | dormant_set
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        fail(errors, f"{label} sources missing: {', '.join(missing)}")
    if unexpected:
        fail(errors, f"unexpected {label} sources: {', '.join(unexpected)}")


def check_cmake_source_ownership(
    *,
    text: str,
    required: tuple[str, ...],
    cmake_path: Path,
    target: str,
    prefix: str = "",
    errors: list[str],
) -> None:
    tokens = cmake_tokens(text)
    for source in required:
        entry = prefix + source
        occurrences = tokens.count(entry)
        if occurrences != 1:
            fail(
                errors,
                f"{cmake_path}: {entry} must be owned exactly once by {target}; "
                f"found {occurrences} CMake entries",
            )


def check_repository_cmake_source_ownership(
    *,
    root: Path,
    source: str,
    expected_cmake_path: Path,
    label: str,
    errors: list[str],
) -> None:
    """Require a source basename to occur in exactly one repository CMake manifest."""

    manifests = [root / "CMakeLists.txt"]
    for directory_name in ("cmake", "lib", "test", "tools", "unittests"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        manifests.extend(directory.rglob("CMakeLists.txt"))
        manifests.extend(directory.rglob("*.cmake"))

    occurrences: list[Path] = []
    for manifest in sorted(set(manifests)):
        text = read_required(manifest, errors)
        for token in cmake_tokens(text):
            if Path(token).name == source:
                occurrences.append(manifest)

    resolved_expected = expected_cmake_path.resolve()
    if len(occurrences) != 1 or occurrences[0].resolve() != resolved_expected:
        owners = ", ".join(str(path) for path in occurrences) or "none"
        fail(
            errors,
            f"{label} {source} must have exactly one repository CMake entry "
            f"({expected_cmake_path}); found {owners}",
        )


def check_cmake_sources_absent(
    *,
    text: str,
    forbidden: tuple[str, ...],
    cmake_path: Path,
    label: str,
    prefixes: tuple[str, ...] = ("",),
    errors: list[str],
) -> None:
    tokens = cmake_tokens(text)
    for source in forbidden:
        entries = tuple(prefix + source for prefix in prefixes)
        found = [entry for entry in entries if entry in tokens]
        if found:
            fail(
                errors,
                f"{cmake_path}: {label} must not own {', '.join(found)}",
            )


def source_includes(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', text, flags=re.M)
    )


def cpp_code(text: str) -> str:
    """Remove comments and literals before checking implementation markers."""
    result: list[str] = []
    index = 0
    state = "code"
    quote = ""
    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "line-comment":
            if character == "\n":
                result.append(character)
                state = "code"
            else:
                result.append(" ")
            index += 1
            continue
        if state == "block-comment":
            if character == "*" and following == "/":
                result.extend((" ", " "))
                index += 2
                state = "code"
            else:
                result.append("\n" if character == "\n" else " ")
                index += 1
            continue
        if state == "literal":
            if character == "\\":
                result.append(" ")
                if following:
                    result.append("\n" if following == "\n" else " ")
                    index += 2
                else:
                    index += 1
                continue
            result.append("\n" if character == "\n" else " ")
            index += 1
            if character == quote:
                state = "code"
            continue
        if character == "/" and following == "/":
            result.extend((" ", " "))
            index += 2
            state = "line-comment"
            continue
        if character == "/" and following == "*":
            result.extend((" ", " "))
            index += 2
            state = "block-comment"
            continue
        if character in ('"', "'"):
            result.append(" ")
            index += 1
            state = "literal"
            quote = character
            continue
        result.append(character)
        index += 1
    return "".join(result)


def check_exact_sources(
    source_root: Path,
    expected: tuple[str, ...],
    label: str,
    errors: list[str],
) -> None:
    actual = {path.name for path in source_root.glob("*.cpp") if path.is_file()}
    wanted = set(expected)
    missing = sorted(wanted - actual)
    unexpected = sorted(actual - wanted)
    if missing:
        fail(errors, f"{label} sources missing: {', '.join(missing)}")
    if unexpected:
        fail(errors, f"unexpected {label} sources: {', '.join(unexpected)}")


def check_required_sources(
    source_root: Path,
    required: tuple[str, ...],
    label: str,
    errors: list[str],
) -> None:
    for filename in required:
        read_required(source_root / filename, errors)


def check_private_header(
    private_path: Path,
    public_path: Path,
    label: str,
    errors: list[str],
) -> None:
    read_required(private_path, errors)
    include_root = next(
        (parent for parent in public_path.parents if parent.name == "include"),
        None,
    )
    leaked = (
        sorted(path for path in include_root.rglob(private_path.name) if path.is_file())
        if include_root and include_root.is_dir()
        else ([public_path] if public_path.exists() else [])
    )
    for path in leaked:
        fail(errors, f"{label} header must remain library-private: {path}")


def check_instruction_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/IR/Instr"
    cmake_path = root / "lib/Wafer/IR/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    actual_family_sources = {
        path.name for path in source_root.glob("*Ops.cpp") if path.is_file()
    }
    expected_family_sources = set(INSTRUCTION_FAMILY_SOURCES)
    if actual_family_sources != expected_family_sources:
        missing = sorted(expected_family_sources - actual_family_sources)
        unexpected = sorted(actual_family_sources - expected_family_sources)
        if missing:
            fail(errors, f"instruction family sources missing: {', '.join(missing)}")
        if unexpected:
            fail(
                errors,
                f"unexpected instruction family sources: {', '.join(unexpected)}",
            )

    for filename in INSTRUCTION_VERIFIER_FILES:
        read_required(source_root / filename, errors)

    public_internal_header = (
        root / "include/Wafer/IR/Instr/InstructionVerification.h"
    )
    if public_internal_header.exists():
        fail(
            errors,
            "instruction verifier helper must remain library-private: "
            f"{public_internal_header}",
        )

    legacy_source = source_root / "InstructionOps.cpp"
    if legacy_source.exists():
        fail(errors, f"legacy instruction aggregate must be removed: {legacy_source}")

    target_body = cmake_target_body(
        cmake_text, "add_mlir_dialect_library", "WaferIR", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=(*INSTRUCTION_FAMILY_SOURCES, "InstructionVerification.cpp"),
        forbidden=("InstructionOps.cpp",),
        prefix="Instr/",
        cmake_path=cmake_path,
        target="WaferIR",
        errors=errors,
    )


def check_tile_region_to_instr_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Conversion/WaferTileRegionToInstr"
    cmake_path = root / "lib/Wafer/Conversion/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    target_body = cmake_target_body(
        cmake_text,
        "add_mlir_conversion_library",
        "WaferTileRegionToInstr",
        cmake_path,
        errors,
    )
    active_sources = cmake_target_cpp_source_names(
        body=target_body,
        prefix="WaferTileRegionToInstr/",
        cmake_path=cmake_path,
        target="WaferTileRegionToInstr",
        errors=errors,
    )
    check_active_and_dormant_directory_sources(
        source_root=source_root,
        active=active_sources,
        dormant=TILE_REGION_TO_INSTR_DORMANT_SOURCES,
        label="tile-region-to-instr",
        errors=errors,
    )
    for filename in (*active_sources, *TILE_REGION_TO_INSTR_DORMANT_SOURCES, "Internal.h"):
        read_required(source_root / filename, errors)

    facade_path = source_root / "WaferTileRegionToInstr.cpp"
    facade_text = read_required(facade_path, errors)
    concrete_pattern = re.compile(
        r"\b(?:class|struct)\s+[A-Za-z_][A-Za-z0-9_]*"
        r"\s*(?:final\s*)?:[^;{]*"
        r"(?:OpRewritePattern|OpConversionPattern|RewritePattern|ConversionPattern)",
        flags=re.DOTALL,
    )
    if concrete_pattern.search(facade_text):
        fail(
            errors,
            f"{facade_path} must orchestrate conversion, not define concrete patterns",
        )


def check_retired_profile_surfaces_removed(root: Path, errors: list[str]) -> None:
    retired_markers = ("DTEProtocolPhase", *RETIRED_PROFILE_SCHEMA_MARKERS)
    production_paths = (
        root / "include/Wafer/IR/WaferAttrs.td",
        root / "include/Wafer/Runtime/ProfileInstrumentation.h",
        root / "lib/Wafer/Runtime/ProfileInstrumentation.cpp",
        root / "lib/Wafer/Compiler/WriteExecutablePackage.cpp",
        root / "tools/wafer-run/WaferProfileCollection.cpp",
        root / "tools/wafer-run/wafer-run.cpp",
    )
    for path in production_paths:
        text = read_required(path, errors)
        for marker in retired_markers:
            if marker in text:
                fail(errors, f"{path} contains retired contract marker {marker}")


def check_numeric_semantics_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Target"
    cmake_path = source_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    for filename in (*NUMERIC_SEMANTICS_SOURCES, "NumericSemanticsInternal.h"):
        read_required(source_root / filename, errors)

    legacy_source = source_root / "NumericSemantics.cpp"
    if legacy_source.exists():
        fail(errors, f"legacy numeric aggregate must be removed: {legacy_source}")

    public_internal_header = root / "include/Wafer/Target/NumericSemanticsInternal.h"
    if public_internal_header.exists():
        fail(
            errors,
            "numeric internal header must remain library-private: "
            f"{public_internal_header}",
        )

    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTarget", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=NUMERIC_SEMANTICS_SOURCES,
        forbidden=("NumericSemantics.cpp",),
        cmake_path=cmake_path,
        target="WaferTarget",
        errors=errors,
    )


def check_tensor_program_to_tile_region_owners(
    root: Path, errors: list[str]
) -> None:
    source_root = root / "lib/Wafer/Conversion/WaferTensorProgramToTileRegion"
    cmake_path = root / "lib/Wafer/Conversion/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    check_private_header(
        source_root / "Internal.h",
        root
        / "include/Wafer/Conversion/WaferTensorProgramToTileRegion/Internal.h",
        "tensor-program-to-tile-region internal",
        errors,
    )
    target_body = cmake_target_body(
        cmake_text,
        "add_mlir_conversion_library",
        "WaferTensorProgramToTileRegion",
        cmake_path,
        errors,
    )
    active_sources = cmake_target_cpp_source_names(
        body=target_body,
        prefix="WaferTensorProgramToTileRegion/",
        cmake_path=cmake_path,
        target="WaferTensorProgramToTileRegion",
        errors=errors,
    )
    check_active_and_dormant_directory_sources(
        source_root=source_root,
        active=active_sources,
        dormant=TENSOR_PROGRAM_TO_TILE_REGION_DORMANT_SOURCES,
        label="tensor-program-to-tile-region",
        errors=errors,
    )
    facade = read_required(
        source_root / "WaferTensorProgramToTileRegion.cpp", errors
    )
    if "OpRewritePattern" in facade or "OpConversionPattern" in facade:
        fail(
            errors,
            "tensor-program-to-tile-region facade must not own concrete "
            "rewrite patterns",
        )

    for legacy_root in (
        root / "include/Wafer/Conversion/WaferGroupToTileRegion",
        root / "lib/Wafer/Conversion/WaferGroupToTileRegion",
    ):
        if legacy_root.exists():
            fail(errors, f"legacy group conversion directory must be removed: {legacy_root}")


def check_legacy_task_local_selection_removed(
    root: Path, errors: list[str]
) -> None:
    source_root = root / "lib/Wafer/Transforms/Scheduling"
    cmake_path = root / "lib/Wafer/Transforms/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    for filename in RETIRED_TASK_LOCAL_SELECTION_PATHS:
        path = source_root / filename
        if path.exists():
            fail(errors, f"legacy task-local selection path must be removed: {path}")
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTransforms", cmake_path, errors
    )
    for filename in RETIRED_TASK_LOCAL_SELECTION_PATHS:
        if not filename.endswith(".cpp"):
            continue
        source = f"Scheduling/{filename}"
        if source in target_body:
            fail(
                errors,
                f"WaferTransforms still lists removed task-local selection source: {source}",
            )

    legacy_root = root / "lib/Wafer/Transforms/Group"
    if legacy_root.exists():
        fail(errors, f"legacy group transform directory must be removed: {legacy_root}")


def check_retired_rank_candidate_sources_removed(
    root: Path, errors: list[str]
) -> None:
    for relative in RETIRED_RANK_CANDIDATE_PATHS:
        path = root / relative
        if path.exists():
            fail(errors, f"retired rank-candidate path must be removed: {path}")

    cmake_path = root / "lib/Wafer/Compiler/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferCompiler", cmake_path, errors
    )
    for filename in ("NoCResidentDataflow.cpp", "WholeVariantAttemptPlan.cpp"):
        if filename in target_body:
            fail(
                errors,
                f"WaferCompiler still lists retired rank-candidate source: {filename}",
            )


def check_memory_planning_owners(root: Path, errors: list[str]) -> None:
    transforms_root = root / "lib/Wafer/Transforms"
    source_root = transforms_root / "MemoryPlanning"
    cmake_path = transforms_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)
    unit_root = root / "unittests/Transforms/MemoryPlanning"
    unit_cmake_path = root / "unittests/CMakeLists.txt"
    unit_cmake_text = read_required(unit_cmake_path, errors)

    check_exact_sources(
        source_root, MEMORY_PLANNING_SOURCES, "memory-planning core", errors
    )
    private_headers = [source_root / header for header in MEMORY_PLANNING_HEADERS]
    actual_headers = {
        path.name for path in source_root.glob("*.h") if path.is_file()
    }
    if actual_headers != set(MEMORY_PLANNING_HEADERS):
        fail(
            errors,
            "memory-planning headers must be exactly "
            f"{', '.join(MEMORY_PLANNING_HEADERS)}; found "
            f"{', '.join(sorted(actual_headers)) or 'none'}",
        )
    for private_header in private_headers:
        check_private_header(
            private_header,
            root / "include/Wafer/Transforms/MemoryPlanning" / private_header.name,
            "memory-planning internal",
            errors,
        )
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTransforms", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=MEMORY_PLANNING_SOURCES,
        prefix="MemoryPlanning/",
        cmake_path=cmake_path,
        target="WaferTransforms",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=cmake_text,
        required=MEMORY_PLANNING_SOURCES,
        prefix="MemoryPlanning/",
        cmake_path=cmake_path,
        target="WaferTransforms",
        errors=errors,
    )
    for source in MEMORY_PLANNING_SOURCES:
        check_repository_cmake_source_ownership(
            root=root,
            source=source,
            expected_cmake_path=cmake_path,
            label="memory-planning core",
            errors=errors,
        )

    detail_namespace = re.compile(
        r"\bnamespace\s+wafer::memory_planning::detail\s*\{"
    )
    legacy_namespace = re.compile(
        r"\bnamespace\s+wafer::memory_planning\s*\{"
    )
    for detail_path in private_headers + [
        source_root / source for source in MEMORY_PLANNING_SOURCES
    ]:
        detail_code = cpp_code(read_required(detail_path, errors))
        if not detail_namespace.search(detail_code):
            fail(
                errors,
                f"{detail_path} must keep shared symbols in "
                "wafer::memory_planning::detail",
            )
        if legacy_namespace.search(detail_code):
            fail(
                errors,
                f"{detail_path} must not declare the non-detail "
                "wafer::memory_planning namespace",
            )

    check_exact_sources(
        unit_root,
        MEMORY_PLANNING_TEST_SOURCES,
        "memory-planning unit mirror",
        errors,
    )
    unit_target_body = cmake_target_body(
        unit_cmake_text,
        "add_executable",
        "WaferUnitTests",
        unit_cmake_path,
        errors,
    )
    check_cmake_sources(
        body=unit_target_body,
        required=MEMORY_PLANNING_TEST_SOURCES,
        prefix="Transforms/MemoryPlanning/",
        cmake_path=unit_cmake_path,
        target="WaferUnitTests",
        errors=errors,
    )
    expected_test_include = {
        "LifetimeAnalysisTest.cpp": "MemoryPlanning/LifetimeAnalysis.h",
        "MiniMallocPackingTest.cpp": "MemoryPlanning/MiniMallocPacking.h",
        "StaticMemoryPackingTest.cpp": "MemoryPlanning/StaticMemoryPacking.h",
    }
    unit_test_paths = []
    for source in MEMORY_PLANNING_TEST_SOURCES:
        check_repository_cmake_source_ownership(
            root=root,
            source=source,
            expected_cmake_path=unit_cmake_path,
            label="memory-planning unit mirror",
            errors=errors,
        )
        unit_test_path = unit_root / source
        unit_test_paths.append(unit_test_path)
        unit_test_text = read_required(unit_test_path, errors)
        shared_include = expected_test_include[source]
        if source_includes(unit_test_text).count(shared_include) != 1:
            fail(
                errors,
                f"{unit_test_path} must include {shared_include} exactly once",
            )

    owner_paths = (
        transforms_root / "DDR/PlanDDRMemory.cpp",
        transforms_root / "SPM/PlanSPMMemory.cpp",
    )
    forbidden_markers = (
        "struct PathCondition",
        "struct LiveSegment",
        "struct EventInfo",
        "struct RootRef",
        "struct SPMDemand {",
        "struct DDRDemand {",
        "struct AssignedSPMInterval {",
        "struct AssignedDDROffset {",
        "struct PendingLocalIssue {",
        "struct LifetimeDataflow {",
        "class StructuredTimeline",
        "class LifetimeDataflow",
        "class LocalCompletionTracker",
        "areCompatible(",
        "mergeConditions(",
        "conditionImplies(",
        "withBranch(",
        "appendConditionDifference(",
        "segmentsOverlap(",
        "lifetimesOverlap(",
        "assignRegionEvents(",
        "assignBlockEvents(",
        "assignOperationEvents(",
        "addLiveSegment(",
        "recordDemandUse(",
        "computeLifetimeBounds(",
        "getLifetimeSpan(",
        "hasHigherPlanningPriority(",
        "findFirstFitOffset",
        "computePlanningPriorities",
        "conflictBytes",
        "alignUp(",
        "packFirstFit(",
    )
    lifetime_include = "MemoryPlanning/LifetimeAnalysis.h"
    packing_include = "MemoryPlanning/StaticMemoryPacking.h"
    for owner_path in owner_paths:
        owner_text = read_required(owner_path, errors)
        for shared_include in (lifetime_include, packing_include):
            if source_includes(owner_text).count(shared_include) != 1:
                fail(
                    errors,
                    f"{owner_path} must include {shared_include} exactly once",
                )
        owner_code = cpp_code(owner_text)
        for marker in forbidden_markers:
            if marker in owner_code:
                fail(
                    errors,
                    f"{owner_path} still owns shared lifetime marker {marker}",
                )

    check_no_textual_source_includes(
        [source_root / source for source in MEMORY_PLANNING_SOURCES]
        + private_headers
        + unit_test_paths
        + list(owner_paths),
        "memory-planning core",
        errors,
    )


def check_target_llvm_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Transforms/Target"
    cmake_path = root / "lib/Wafer/Transforms/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    for filename in TARGET_LLVM_SOURCES:
        read_required(source_root / filename, errors)
    check_private_header(
        source_root / "LowerInstrToTargetLLVMInternal.h",
        root / "include/Wafer/Transforms/Target/LowerInstrToTargetLLVMInternal.h",
        "target-LLVM internal",
        errors,
    )
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTransforms", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=TARGET_LLVM_SOURCES,
        prefix="Target/",
        cmake_path=cmake_path,
        target="WaferTransforms",
        errors=errors,
    )
    facade = read_required(source_root / "LowerInstrToTargetLLVM.cpp", errors)
    if "OpConversionPattern" in facade or "ConversionPattern" in facade:
        fail(errors, "target-LLVM facade must not own concrete conversion patterns")


def check_numeric_dependency_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Target"
    cmake_path = source_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    for filename in NUMERIC_DEPENDENCY_SOURCES:
        read_required(source_root / filename, errors)
    check_private_header(
        source_root / "NumericDependencyConformanceInternal.h",
        root / "include/Wafer/Target/NumericDependencyConformanceInternal.h",
        "numeric-dependency internal",
        errors,
    )
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTarget", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=NUMERIC_DEPENDENCY_SOURCES,
        cmake_path=cmake_path,
        target="WaferTarget",
        errors=errors,
    )


def check_frontend_program_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Frontend"
    cmake_path = source_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    check_exact_sources(
        source_root, FRONTEND_PROGRAM_SOURCES, "frontend program", errors
    )
    check_private_header(
        source_root / "ProgramInternal.h",
        root / "include/Wafer/Frontend/ProgramInternal.h",
        "frontend program internal",
        errors,
    )
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferFrontend", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=FRONTEND_PROGRAM_SOURCES,
        cmake_path=cmake_path,
        target="WaferFrontend",
        errors=errors,
    )
    facade = read_required(source_root / "Program.cpp", errors)
    for implementation in ("llvm/Support/JSON.h", "NUMPY", "NpyPayloadMetadata"):
        if implementation in facade:
            fail(errors, f"frontend program facade still owns {implementation}")


def check_wafer_compile_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "tools/wafer-compile"
    cmake_path = source_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    check_exact_sources(source_root, WAFER_COMPILE_SOURCES, "wafer-compile", errors)
    read_required(source_root / "DriverInternal.h", errors)
    source_body = cmake_target_body(
        cmake_text, "set", "_wafer_compile_sources", cmake_path, errors
    )
    check_cmake_sources(
        body=source_body,
        required=WAFER_COMPILE_SOURCES,
        cmake_path=cmake_path,
        target="_wafer_compile_sources",
        errors=errors,
    )
    for target in ("wafer-compile", "wafer-compile-test"):
        target_body = cmake_target_body(
            cmake_text, "add_executable", target, cmake_path, errors
        )
        if "${_wafer_compile_sources}" not in target_body:
            fail(errors, f"{cmake_path}: {target} must use shared driver source set")
    facade = read_required(source_root / "wafer-compile.cpp", errors)
    for implementation in (
        "bool parseCommandLine(",
        "bool runTargetModelGate(",
    ):
        if implementation in facade:
            fail(errors, f"wafer-compile main still defines {implementation}")


def check_no_textual_source_includes(
    paths: list[Path], label: str, errors: list[str]
) -> None:
    pattern = re.compile(r'^\s*#\s*include\s*[<\"][^>\"]*\.(?:cc|cpp|cxx)[>\"]', re.M)
    for path in paths:
        text = read_required(path, errors)
        if pattern.search(text):
            fail(errors, f"{label} must not use textual source includes: {path}")


def check_xla_helper_source_manifest(
    builder: str, builder_path: Path, errors: list[str]
) -> None:
    try:
        module = ast.parse(builder, filename=str(builder_path))
    except SyntaxError as error:
        fail(errors, f"{builder_path} is not valid Python: {error}")
        return

    def assignment(name: str) -> ast.expr | None:
        values: list[ast.expr] = []
        for statement in module.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            ):
                values.append(statement.value)
        if len(values) != 1:
            fail(
                errors,
                f"{builder_path}: expected one top-level {name} assignment; "
                f"found {len(values)}",
            )
            return None
        return values[0]

    manifest_value = assignment("HELPER_SOURCES")
    if manifest_value is None:
        return
    try:
        manifest = ast.literal_eval(manifest_value)
    except (TypeError, ValueError, SyntaxError):
        fail(errors, f"{builder_path}: HELPER_SOURCES must be a literal source map")
        return
    if not isinstance(manifest, (tuple, list)) or any(
        not isinstance(mapping, (tuple, list))
        or len(mapping) != 2
        or not all(isinstance(value, str) for value in mapping)
        for mapping in manifest
    ):
        fail(errors, f"{builder_path}: HELPER_SOURCES has invalid entries")
        return

    source_names = [mapping[0] for mapping in manifest]
    overlay_names = [mapping[1] for mapping in manifest]
    expected_sources = set(XLA_SPMD_HELPER_SOURCES)
    configured_sources = set(source_names)
    missing = sorted(expected_sources - configured_sources)
    unexpected = sorted(configured_sources - expected_sources)
    if missing:
        fail(errors, f"XLA helper overlay sources missing: {', '.join(missing)}")
    if unexpected:
        fail(errors, f"unexpected XLA helper overlay sources: {', '.join(unexpected)}")
    if len(source_names) != len(configured_sources):
        fail(errors, f"{builder_path}: HELPER_SOURCES repeats a production source")
    if len(overlay_names) != len(set(overlay_names)):
        fail(errors, f"{builder_path}: HELPER_SOURCES repeats a Bazel overlay source")
    if any(not name.endswith(".cc") for name in overlay_names):
        fail(errors, f"{builder_path}: helper overlay sources must use .cc names")

    build_sources_value = assignment("HELPER_BUILD_SOURCES")
    if build_sources_value is not None:
        build_source_names = {
            node.id
            for node in ast.walk(build_sources_value)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if build_source_names != {"HELPER_SOURCES", "overlay_name"}:
            fail(
                errors,
                f"{builder_path}: Bazel srcs must derive only from HELPER_SOURCES",
            )
    build_file_value = assignment("BUILD_FILE")
    if build_file_value is not None:
        formatted_values = [
            node.value
            for node in ast.walk(build_file_value)
            if isinstance(node, ast.FormattedValue)
        ]
        if (
            len(formatted_values) != 1
            or not isinstance(formatted_values[0], ast.Name)
            or formatted_values[0].id != "HELPER_BUILD_SOURCES"
        ):
            fail(errors, f"{builder_path}: BUILD_FILE bypasses HELPER_BUILD_SOURCES")
        static_build_text = "".join(
            node.value
            for node in ast.walk(build_file_value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if re.search(r'"[^"\n]+\.cc"', static_build_text):
            fail(errors, f"{builder_path}: BUILD_FILE hardcodes Bazel source names")

    populate = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_populate_workspace"
        ),
        None,
    )
    manifest_loops = (
        [
            node
            for node in ast.walk(populate)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "HELPER_SOURCES"
        ]
        if populate
        else []
    )
    if len(manifest_loops) != 1 or not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_symlink"
        for node in ast.walk(manifest_loops[0])
    ):
        fail(
            errors,
            f"{builder_path}: overlay materialization must consume HELPER_SOURCES once",
        )

    string_literals = [
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    literal_cpp_sources = sorted(
        value
        for value in string_literals
        if re.fullmatch(r"XlaSpmd[^/]*\.cpp", value)
    )
    if literal_cpp_sources != sorted(source_names):
        fail(
            errors,
            f"{builder_path}: production .cpp names must have HELPER_SOURCES as "
            "their single fact source",
        )
    literal_overlay_sources = sorted(
        value
        for value in string_literals
        if re.fullmatch(r"(?:wafer_)?xla_spmd[^/]*\.cc", value)
    )
    if literal_overlay_sources != sorted(overlay_names):
        fail(
            errors,
            f"{builder_path}: Bazel .cc names must have HELPER_SOURCES as their "
            "single fact source",
        )
    header_name = "XlaSpmdPartitionerInternal.h"
    build_header_mentions = (
        sum(
            node.value.count(header_name)
            for node in ast.walk(build_file_value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if build_file_value is not None
        else 0
    )
    populate_header_mentions = (
        sum(
            node.value.count(header_name)
            for node in ast.walk(populate)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if populate is not None
        else 0
    )
    if build_header_mentions != 1 or populate_header_mentions != 2:
        fail(
            errors,
            f"{builder_path}: private header must be declared and materialized "
            "from one source path exactly once",
        )


def check_reference_executor_retired(root: Path, errors: list[str]) -> None:
    for relative in REFERENCE_EXECUTOR_LEGACY_PATHS:
        path = root / relative
        if path.exists():
            fail(errors, f"retired reference executor path still exists: {path}")

    production_files = [root / "unittests/CMakeLists.txt"]
    for production_root in (
        root / "include",
        root / "lib",
        root / "tools/wafer-compile",
    ):
        production_files.extend(
            path
            for path in production_root.rglob("*")
            if path.is_file()
            and (path.suffix in {".h", ".cpp", ".td", ".inc"}
                 or path.name == "CMakeLists.txt")
        )
    for path in production_files:
        text = read_required(path, errors)
        for marker in (
            "ReferenceExecutor",
            "ReferenceProgram",
            "ReferenceExecutionResult",
            "ReferenceGate",
            "runReferenceGate",
            "--reference-input",
            "--reference-expected",
            "--target-model-oracle",
        ):
            if marker in text:
                fail(errors, f"{path}: retired reference marker remains: {marker}")


def check_target_model_owners(root: Path, errors: list[str]) -> None:
    model_root = root / "lib/Wafer/Model"
    model_cmake = model_root / "CMakeLists.txt"
    model_text = read_required(model_cmake, errors)
    check_required_sources(
        model_root, TARGET_MODEL_KERNEL_SOURCES, "target-model kernel", errors
    )
    check_private_header(
        model_root / "TargetModelKernelInternal.h",
        root / "include/Wafer/Model/TargetModelKernelInternal.h",
        "target-model kernel internal",
        errors,
    )
    model_body = cmake_target_body(
        model_text,
        "add_mlir_library",
        "WaferTargetFunctionalModel",
        model_cmake,
        errors,
    )
    check_cmake_sources(
        body=model_body,
        required=TARGET_MODEL_KERNEL_SOURCES,
        cmake_path=model_cmake,
        target="WaferTargetFunctionalModel",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=model_text,
        required=TARGET_MODEL_KERNEL_SOURCES,
        cmake_path=model_cmake,
        target="WaferTargetFunctionalModel",
        errors=errors,
    )
    model_facade = read_required(model_root / "TargetModelKernel.cpp", errors)
    model_facade_code = cpp_code(model_facade)
    for implementation in (
        "template <typename Enum>",
        "getSegmentAddresses(",
        "getNumericOperation(",
        "readSnapshot(",
    ):
        if implementation in model_facade_code:
            fail(errors, f"target-model kernel facade still owns {implementation}")

    check_no_textual_source_includes(
        [model_root / name for name in TARGET_MODEL_KERNEL_SOURCES]
        + [model_root / "TargetModelKernelInternal.h"],
        "target model",
        errors,
    )


def check_numeric_bulk_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Target"
    cmake_path = source_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    check_required_sources(
        source_root, FORMAL_NUMERIC_SOURCES, "formal numeric", errors
    )
    check_private_header(
        source_root / "FormalNumericInternal.h",
        root / "include/Wafer/Target/FormalNumericInternal.h",
        "formal numeric internal",
        errors,
    )
    legacy_formal = source_root / "FormalNumeric.cpp"
    if legacy_formal.exists():
        fail(
            errors,
            f"legacy formal numeric aggregate must be removed: {legacy_formal}",
        )
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTarget", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=FORMAL_NUMERIC_SOURCES,
        forbidden=("FormalNumeric.cpp",),
        cmake_path=cmake_path,
        target="WaferTarget",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=cmake_text,
        required=FORMAL_NUMERIC_SOURCES,
        cmake_path=cmake_path,
        target="WaferTarget",
        errors=errors,
    )

    bulk_sources = (*BULK_TENSOR_NUMERIC_SOURCES, *BULK_QUALIFICATION_SOURCES)
    check_required_sources(source_root, bulk_sources, "bulk numeric", errors)
    for private_name in (
        "BulkTensorNumericInternal.h",
        "BulkQualificationInternal.h",
    ):
        check_private_header(
            source_root / private_name,
            root / "include/Wafer/Target" / private_name,
            private_name,
            errors,
        )
    legacy_bulk = source_root / "BulkTensorNumeric.cpp"
    if legacy_bulk.exists():
        fail(errors, f"legacy bulk tensor aggregate must be removed: {legacy_bulk}")
    bulk_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferBulkModel", cmake_path, errors
    )
    check_cmake_sources(
        body=bulk_body,
        required=bulk_sources,
        forbidden=("BulkTensorNumeric.cpp",),
        cmake_path=cmake_path,
        target="WaferBulkModel",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=cmake_text,
        required=bulk_sources,
        cmake_path=cmake_path,
        target="WaferBulkModel",
        errors=errors,
    )
    qualification_facade = read_required(
        source_root / "BulkQualification.cpp", errors
    )
    qualification_facade_code = cpp_code(qualification_facade)
    for implementation in (
        "generateValues(",
        "struct Comparison",
        "runQualification(",
        "calibrateBulkBackend(",
        "freezeBulkBackendPolicy(",
        "validateBulkBackend(",
    ):
        if implementation in qualification_facade_code:
            fail(errors, f"bulk qualification facade still owns {implementation}")
    check_no_textual_source_includes(
        [source_root / name for name in (*FORMAL_NUMERIC_SOURCES, *bulk_sources)]
        + [
            source_root / "FormalNumericInternal.h",
            source_root / "BulkTensorNumericInternal.h",
            source_root / "BulkQualificationInternal.h",
        ],
        "numeric/bulk",
        errors,
    )


def check_compiler_target_module_and_package_sources(
    root: Path, errors: list[str]
) -> None:
    compiler_root = root / "lib/Wafer/Compiler"
    compiler_cmake = compiler_root / "CMakeLists.txt"
    compiler_text = read_required(compiler_cmake, errors)
    compiler_sources = (*COMPILATION_SOURCES, *TARGET_CODE_GENERATION_SOURCES)
    check_required_sources(
        compiler_root, compiler_sources, "compiler target module", errors
    )
    for private_name in ("CompilationInternal.h", "TargetCodeGenInternal.h"):
        check_private_header(
            compiler_root / private_name,
            root / "include/Wafer/Compiler" / private_name,
            private_name,
            errors,
        )
    compiler_body = cmake_target_body(
        compiler_text, "add_mlir_library", "WaferCompiler", compiler_cmake, errors
    )
    check_cmake_sources(
        body=compiler_body,
        required=compiler_sources,
        cmake_path=compiler_cmake,
        target="WaferCompiler",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=compiler_text,
        required=compiler_sources,
        cmake_path=compiler_cmake,
        target="WaferCompiler",
        errors=errors,
    )
    compilation_facade = read_required(compiler_root / "Compilation.cpp", errors)
    compilation_includes = set(source_includes(compilation_facade))
    if (
        "mlir/Parser/Parser.h" in compilation_includes
        or "filesystem" in compilation_includes
    ):
        fail(errors, "compilation facade still owns stage or filesystem implementation")
    target_module_facade = read_required(
        compiler_root / "TargetLLVMModule.cpp", errors
    )
    target_module_includes = source_includes(target_module_facade)
    for implementation in ("llvm/Object/", "llvm/Linker/", "mlir/Target/"):
        if any(path.startswith(implementation) for path in target_module_includes):
            fail(errors, f"target-module facade still owns {implementation}")

    package_root = root / "lib/Wafer/Package"
    package_cmake = package_root / "CMakeLists.txt"
    package_text = read_required(package_cmake, errors)
    check_exact_sources(package_root, PACKAGE_SUPPORT_SOURCES, "package", errors)
    check_private_header(
        package_root / "PackageManifestInternal.h",
        root / "include/Wafer/Package/PackageManifestInternal.h",
        "package manifest internal",
        errors,
    )
    package_body = cmake_target_body(
        package_text, "add_mlir_library", "WaferPackageSupport", package_cmake,
        errors
    )
    check_cmake_sources(
        body=package_body,
        required=PACKAGE_SUPPORT_SOURCES,
        cmake_path=package_cmake,
        target="WaferPackageSupport",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=package_text,
        required=PACKAGE_SUPPORT_SOURCES,
        cmake_path=package_cmake,
        target="WaferPackageSupport",
        errors=errors,
    )
    manifest_facade = read_required(package_root / "PackageManifest.cpp", errors)
    manifest_includes = set(source_includes(manifest_facade))
    if (
        "llvm/Support/JSON.h" in manifest_includes
        or "filesystem" in manifest_includes
    ):
        fail(errors, "package-manifest facade still owns JSON or filesystem I/O")

    runtime_root = root / "lib/Wafer/Runtime"
    runtime_cmake = runtime_root / "CMakeLists.txt"
    runtime_text = read_required(runtime_cmake, errors)
    check_exact_sources(runtime_root, RUNTIME_SOURCES, "runtime", errors)
    runtime_body = cmake_target_body(
        runtime_text, "add_mlir_library", "WaferRuntime", runtime_cmake, errors
    )
    check_cmake_sources(
        body=runtime_body,
        required=RUNTIME_SOURCES,
        cmake_path=runtime_cmake,
        target="WaferRuntime",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=runtime_text,
        required=RUNTIME_SOURCES,
        cmake_path=runtime_cmake,
        target="WaferRuntime",
        errors=errors,
    )
    check_cmake_links(
        body=runtime_body,
        required=["WaferPackageSupport"],
        forbidden=["WaferCompiler"],
        cmake_path=runtime_cmake,
        target="WaferRuntime",
        errors=errors,
    )
    # Boundary gate: the compiler package writer consumes the neutral
    # package support library and must not link the board runtime
    # implementation to write packages.
    compiler_cmake = compiler_root / "CMakeLists.txt"
    compiler_text = read_required(compiler_cmake, errors)
    compiler_body = cmake_target_body(
        compiler_text, "add_mlir_library", "WaferCompiler", compiler_cmake,
        errors
    )
    check_cmake_links(
        body=compiler_body,
        required=["WaferPackageSupport"],
        forbidden=["WaferRuntime"],
        cmake_path=compiler_cmake,
        target="WaferCompiler",
        errors=errors,
    )
    check_no_textual_source_includes(
        [compiler_root / name for name in compiler_sources]
        + [
            compiler_root / "CompilationInternal.h",
            compiler_root / "TargetCodeGenInternal.h",
        ]
        + [package_root / name for name in PACKAGE_SUPPORT_SOURCES]
        + [package_root / "PackageManifestInternal.h"]
        + [runtime_root / name for name in RUNTIME_SOURCES],
        "compiler/target-module/package",
        errors,
    )

    runner_root = root / "tools/wafer-run"
    runner_cmake = runner_root / "CMakeLists.txt"
    runner_text = read_required(runner_cmake, errors)
    check_exact_sources(runner_root, WAFER_RUN_SOURCES, "wafer-run", errors)
    runner_body = cmake_target_body(
        runner_text, "add_executable", "wafer-run", runner_cmake, errors
    )
    check_cmake_sources(
        body=runner_body,
        required=("wafer-run.cpp",),
        cmake_path=runner_cmake,
        target="wafer-run",
        errors=errors,
    )
    if "target_sources(wafer-run PRIVATE TxBoardRuntime.cpp)" not in runner_text:
        fail(errors, "wafer-run must own the optional TX board adapter")
    check_cmake_source_ownership(
        text=runner_text,
        required=WAFER_RUN_SOURCES,
        cmake_path=runner_cmake,
        target="wafer-run",
        errors=errors,
    )


def check_frontend_bridge_owners(root: Path, errors: list[str]) -> None:
    conversion_root = root / "lib/Wafer/Conversion/StableHLOToLinalg"
    conversion_cmake = root / "lib/Wafer/Conversion/CMakeLists.txt"
    conversion_text = read_required(conversion_cmake, errors)
    check_required_sources(
        conversion_root,
        STABLEHLO_NORMALIZATION_SOURCES,
        "StableHLO normalization",
        errors,
    )
    check_private_header(
        conversion_root / "ConstantTensorFoldingInternal.h",
        root
        / "include/Wafer/Conversion/StableHLOToLinalg/ConstantTensorFoldingInternal.h",
        "constant tensor folding internal",
        errors,
    )
    conversion_body = cmake_target_body(
        conversion_text,
        "add_mlir_conversion_library",
        "WaferStableHLOToLinalg",
        conversion_cmake,
        errors,
    )
    check_cmake_sources(
        body=conversion_body,
        required=STABLEHLO_NORMALIZATION_SOURCES,
        prefix="StableHLOToLinalg/",
        cmake_path=conversion_cmake,
        target="WaferStableHLOToLinalg",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=conversion_text,
        required=STABLEHLO_NORMALIZATION_SOURCES,
        prefix="StableHLOToLinalg/",
        cmake_path=conversion_cmake,
        target="WaferStableHLOToLinalg",
        errors=errors,
    )
    normalize_facade = read_required(
        conversion_root / "NormalizeStablehloCollectives.cpp", errors
    )
    if "foldConstantLinalgGeneric" in cpp_code(normalize_facade):
        fail(errors, "StableHLO collective owner still embeds constant folding")

    helper_root = root / "lib/Wafer/Transforms/SPMD"
    transforms_cmake = root / "lib/Wafer/Transforms/CMakeLists.txt"
    transforms_text = read_required(transforms_cmake, errors)
    check_cmake_sources_absent(
        text=transforms_text,
        forbidden=XLA_SPMD_HELPER_SOURCES,
        prefixes=("", "SPMD/"),
        cmake_path=transforms_cmake,
        label="core WaferTransforms target; XLA helper sources are Bazel-only",
        errors=errors,
    )
    check_required_sources(
        helper_root, XLA_SPMD_HELPER_SOURCES, "XLA SPMD helper", errors
    )
    actual_helper_sources = {
        path.name for path in helper_root.glob("XlaSpmd*.cpp") if path.is_file()
    }
    expected_helper_sources = set(XLA_SPMD_HELPER_SOURCES)
    missing_helper_sources = sorted(expected_helper_sources - actual_helper_sources)
    unexpected_helper_sources = sorted(actual_helper_sources - expected_helper_sources)
    if missing_helper_sources:
        fail(
            errors,
            "XLA SPMD helper sources missing: " + ", ".join(missing_helper_sources),
        )
    if unexpected_helper_sources:
        fail(
            errors,
            "unexpected XLA SPMD helper sources: "
            + ", ".join(unexpected_helper_sources),
        )
    check_private_header(
        helper_root / "XlaSpmdPartitionerInternal.h",
        root / "include/Wafer/Transforms/SPMD/XlaSpmdPartitionerInternal.h",
        "XLA SPMD helper internal",
        errors,
    )
    helper_main = read_required(helper_root / "XlaSpmdPartitionerMain.cpp", errors)
    helper_main_code = cpp_code(helper_main)
    helper_main_includes = source_includes(helper_main)
    for implementation in ("xla/", "llvm/Support/JSON.h"):
        if any(
            path == implementation or path.startswith(implementation)
            for path in helper_main_includes
        ):
            fail(errors, f"XLA SPMD helper main still owns {implementation}")
    if re.search(r"\bmakeNpyPayload\s*\(", helper_main_code):
        fail(errors, "XLA SPMD helper main still owns makeNpyPayload")

    builder_path = root / "tools/build_xla_spmd_partitioner_helper.py"
    builder = read_required(builder_path, errors)
    check_xla_helper_source_manifest(builder, builder_path, errors)

    check_no_textual_source_includes(
        [conversion_root / name for name in STABLEHLO_NORMALIZATION_SOURCES]
        + [conversion_root / "ConstantTensorFoldingInternal.h"]
        + [helper_root / name for name in XLA_SPMD_HELPER_SOURCES]
        + [helper_root / "XlaSpmdPartitionerInternal.h"],
        "frontend bridge",
        errors,
    )


def check_lib_wafer_dependency_order(root: Path, errors: list[str]) -> None:
    cmake_path = root / "lib/Wafer/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)
    subdirectories = re.findall(
        r"(?m)^\s*add_subdirectory\(\s*([^\s\)]+)\s*\)", cmake_text
    )

    positions: dict[str, int] = {}
    for expected in LIB_WAFER_SUBDIRECTORY_ORDER:
        occurrences = [
            index
            for index, subdirectory in enumerate(subdirectories)
            if subdirectory == expected
        ]
        if not occurrences:
            fail(errors, f"{cmake_path} missing add_subdirectory({expected})")
        elif len(occurrences) > 1:
            fail(errors, f"{cmake_path} adds subdirectory {expected} more than once")
        else:
            positions[expected] = occurrences[0]

    present_order = [
        name for name in LIB_WAFER_SUBDIRECTORY_ORDER if name in positions
    ]
    if any(
        positions[left] >= positions[right]
        for left, right in zip(present_order, present_order[1:])
    ):
        actual = sorted(present_order, key=positions.get)
        fail(
            errors,
            f"{cmake_path} dependency order must be "
            f"{' -> '.join(LIB_WAFER_SUBDIRECTORY_ORDER)}; found "
            f"{' -> '.join(actual)}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    errors: list[str] = []
    check_legacy_group_surfaces_retired(root, errors)
    check_instruction_owners(root, errors)
    check_tile_region_to_instr_owners(root, errors)
    check_retired_profile_surfaces_removed(root, errors)
    check_numeric_semantics_owners(root, errors)
    check_tensor_program_to_tile_region_owners(root, errors)
    check_legacy_task_local_selection_removed(root, errors)
    check_retired_rank_candidate_sources_removed(root, errors)
    check_memory_planning_owners(root, errors)
    check_target_llvm_owners(root, errors)
    check_numeric_dependency_owners(root, errors)
    check_frontend_program_owners(root, errors)
    check_wafer_compile_owners(root, errors)
    check_reference_executor_retired(root, errors)
    check_target_model_owners(root, errors)
    check_numeric_bulk_owners(root, errors)
    check_compiler_target_module_and_package_sources(root, errors)
    check_frontend_bridge_owners(root, errors)
    check_lib_wafer_dependency_order(root, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Wafer source organization checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
