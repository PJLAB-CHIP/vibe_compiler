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
TILE_REGION_TO_INSTR_SOURCES = (
    "CollectiveLowering.cpp",
    "ComputeLowering.cpp",
    "MovementLowering.cpp",
    "MovementSupport.cpp",
    "WaferTileRegionToInstr.cpp",
)
NUMERIC_SEMANTICS_SOURCES = (
    "NumericCapability.cpp",
    "NumericCommand.cpp",
    "NumericProfiles.cpp",
    "NumericSemanticsInternal.cpp",
)
GROUP_TO_TILE_REGION_SOURCES = (
    "BodyEmitter.cpp",
    "CandidateMaterialization.cpp",
    "CandidateSupport.cpp",
    "CollectiveLowering.cpp",
    "CompleteTraversal.cpp",
    "GenericLowering.cpp",
    "GroupLowering.cpp",
    "NamedComputeLowering.cpp",
    "TensorControlFlowLowering.cpp",
    "TileMaterialization.cpp",
    "WaferGroupToTileRegion.cpp",
)
CANDIDATE_SELECTION_SOURCES = (
    "CandidateAnalysis.cpp",
    "CandidateCommit.cpp",
    "CandidateEvaluation.cpp",
    "CandidateSelection.cpp",
    "SelectGroupTile.cpp",
)
MEMORY_PLANNING_SOURCES = (
    "LifetimeAnalysis.cpp",
)
MEMORY_PLANNING_TEST_SOURCES = (
    "LifetimeAnalysisTest.cpp",
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
    "TargetCallPreflight.cpp",
    "TargetLLVMConversion.cpp",
    "TargetLLVMConversionPatterns.cpp",
    "TargetLLVMStructure.cpp",
)
NUMERIC_DEPENDENCY_SOURCES = (
    "NumericDependencyBuildIdentity.cpp",
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
    "TargetModelTransactionSchema.cpp",
)
FORMAL_NUMERIC_SOURCES = (
    "FormalNumericConvert.cpp",
    "FormalNumericElementwise.cpp",
    "FormalNumericGemm.cpp",
    "FormalNumericSupport.cpp",
    "FormalNumericValidation.cpp",
)
BULK_TENSOR_NUMERIC_SOURCES = (
    "BulkAdmission.cpp",
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
    "GroupedProgramCompilation.cpp",
    "ProgramDirectoryTransaction.cpp",
    "SpmdCompilationBridge.cpp",
    "TargetPackagePublication.cpp",
)
TARGET_ARTIFACT_SOURCES = (
    "TargetABIPreparation.cpp",
    "TargetArtifact.cpp",
    "TargetArtifactPublication.cpp",
    "TargetDeviceLink.cpp",
    "TargetLLVMTranslation.cpp",
    "TargetModuleReadback.cpp",
    "TargetRankPreflight.cpp",
)
PACKAGE_MANIFEST_SOURCES = (
    "PackageManifest.cpp",
    "PackageManifestJson.cpp",
    "PackageManifestReadback.cpp",
    "PackageManifestSerialization.cpp",
    "PackageManifestVerification.cpp",
    "RuntimeSessionPreflight.cpp",
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
    "InstructionVerifierUtils.cpp",
    "InstructionVerifierUtils.h",
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


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


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


def cmake_tokens(text: str) -> list[str]:
    return [
        token.strip('"')
        for token in re.findall(r'"(?:\\.|[^"\\])*"|[^\s()]+', cmake_code(text))
    ]


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


def check_project_cmake_source_ownership(
    *,
    root: Path,
    source: str,
    expected_cmake_path: Path,
    label: str,
    errors: list[str],
) -> None:
    """Require a source basename to occur in exactly one project CMake manifest."""

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
            f"{label} {source} must have exactly one project CMake owner "
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
        root / "include/Wafer/IR/Instr/InstructionVerifierUtils.h"
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
        required=(*INSTRUCTION_FAMILY_SOURCES, "InstructionVerifierUtils.cpp"),
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

    for filename in (*TILE_REGION_TO_INSTR_SOURCES, "Internal.h"):
        read_required(source_root / filename, errors)

    actual_sources = {
        path.name for path in source_root.glob("*.cpp") if path.is_file()
    }
    expected_sources = set(TILE_REGION_TO_INSTR_SOURCES)
    if actual_sources != expected_sources:
        missing = sorted(expected_sources - actual_sources)
        unexpected = sorted(actual_sources - expected_sources)
        if missing:
            fail(errors, f"tile-region-to-instr sources missing: {', '.join(missing)}")
        if unexpected:
            fail(
                errors,
                "unexpected tile-region-to-instr sources: " + ", ".join(unexpected),
            )

    target_body = cmake_target_body(
        cmake_text,
        "add_mlir_conversion_library",
        "WaferTileRegionToInstr",
        cmake_path,
        errors,
    )
    check_cmake_sources(
        body=target_body,
        required=TILE_REGION_TO_INSTR_SOURCES,
        prefix="WaferTileRegionToInstr/",
        cmake_path=cmake_path,
        target="WaferTileRegionToInstr",
        errors=errors,
    )

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


def check_group_to_tile_region_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Conversion/WaferGroupToTileRegion"
    cmake_path = root / "lib/Wafer/Conversion/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    check_exact_sources(
        source_root, GROUP_TO_TILE_REGION_SOURCES, "group-to-tile-region", errors
    )
    check_private_header(
        source_root / "Internal.h",
        root / "include/Wafer/Conversion/WaferGroupToTileRegion/Internal.h",
        "group-to-tile-region internal",
        errors,
    )
    target_body = cmake_target_body(
        cmake_text,
        "add_mlir_conversion_library",
        "WaferGroupToTileRegion",
        cmake_path,
        errors,
    )
    check_cmake_sources(
        body=target_body,
        required=GROUP_TO_TILE_REGION_SOURCES,
        prefix="WaferGroupToTileRegion/",
        cmake_path=cmake_path,
        target="WaferGroupToTileRegion",
        errors=errors,
    )
    facade = read_required(source_root / "WaferGroupToTileRegion.cpp", errors)
    if "OpRewritePattern" in facade or "OpConversionPattern" in facade:
        fail(
            errors,
            "group-to-tile-region facade must not own concrete rewrite patterns",
        )


def check_candidate_selection_owners(root: Path, errors: list[str]) -> None:
    source_root = root / "lib/Wafer/Transforms/Group"
    cmake_path = root / "lib/Wafer/Transforms/CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)

    for filename in CANDIDATE_SELECTION_SOURCES:
        read_required(source_root / filename, errors)
    check_private_header(
        source_root / "SelectGroupTileInternal.h",
        root / "include/Wafer/Transforms/Group/SelectGroupTileInternal.h",
        "candidate-selection internal",
        errors,
    )
    target_body = cmake_target_body(
        cmake_text, "add_mlir_library", "WaferTransforms", cmake_path, errors
    )
    check_cmake_sources(
        body=target_body,
        required=CANDIDATE_SELECTION_SOURCES,
        prefix="Group/",
        cmake_path=cmake_path,
        target="WaferTransforms",
        errors=errors,
    )
    facade = read_required(source_root / "SelectGroupTile.cpp", errors)
    for implementation in (
        "struct CandidateRecord",
        "struct CandidateCost",
        "class CandidateAnalysis",
    ):
        if implementation in facade:
            fail(errors, f"candidate-selection facade still owns {implementation}")


def check_memory_planning_owners(root: Path, errors: list[str]) -> None:
    transforms_root = root / "lib/Wafer/Transforms"
    source_root = transforms_root / "MemoryPlanning"
    cmake_path = transforms_root / "CMakeLists.txt"
    cmake_text = read_required(cmake_path, errors)
    unit_root = root / "unittests/Transforms/MemoryPlanning"
    unit_cmake_path = root / "unittests/CMakeLists.txt"
    unit_cmake_text = read_required(unit_cmake_path, errors)

    check_exact_sources(
        source_root, MEMORY_PLANNING_SOURCES, "memory-planning analysis", errors
    )
    private_header = source_root / "LifetimeAnalysis.h"
    analysis_source = source_root / "LifetimeAnalysis.cpp"
    actual_headers = {
        path.name for path in source_root.glob("*.h") if path.is_file()
    }
    if actual_headers != {private_header.name}:
        fail(
            errors,
            "memory-planning analysis headers must be exactly "
            f"{private_header.name}; found {', '.join(sorted(actual_headers)) or 'none'}",
        )
    check_private_header(
        private_header,
        root / "include/Wafer/Transforms/MemoryPlanning/LifetimeAnalysis.h",
        "memory-planning lifetime analysis",
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
    check_project_cmake_source_ownership(
        root=root,
        source="LifetimeAnalysis.cpp",
        expected_cmake_path=cmake_path,
        label="memory-planning analysis",
        errors=errors,
    )

    detail_namespace = re.compile(
        r"\bnamespace\s+wafer::memory_planning::detail\s*\{"
    )
    legacy_namespace = re.compile(
        r"\bnamespace\s+wafer::memory_planning\s*\{"
    )
    for detail_path in (private_header, analysis_source):
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
    check_project_cmake_source_ownership(
        root=root,
        source="LifetimeAnalysisTest.cpp",
        expected_cmake_path=unit_cmake_path,
        label="memory-planning unit mirror",
        errors=errors,
    )
    unit_test_path = unit_root / "LifetimeAnalysisTest.cpp"
    unit_test_text = read_required(unit_test_path, errors)
    shared_include = "MemoryPlanning/LifetimeAnalysis.h"
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
    )
    for owner_path in owner_paths:
        owner_text = read_required(owner_path, errors)
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
        + [private_header, unit_test_path, *owner_paths],
        "memory-planning analysis",
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


def check_compiler_artifact_package_owners(root: Path, errors: list[str]) -> None:
    compiler_root = root / "lib/Wafer/Compiler"
    compiler_cmake = compiler_root / "CMakeLists.txt"
    compiler_text = read_required(compiler_cmake, errors)
    compiler_sources = (*COMPILATION_SOURCES, *TARGET_ARTIFACT_SOURCES)
    check_required_sources(
        compiler_root, compiler_sources, "compiler artifact", errors
    )
    for private_name in ("CompilationInternal.h", "TargetArtifactInternal.h"):
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
    artifact_facade = read_required(compiler_root / "TargetArtifact.cpp", errors)
    artifact_includes = source_includes(artifact_facade)
    for implementation in ("llvm/Object/", "llvm/Linker/", "mlir/Target/"):
        if any(path.startswith(implementation) for path in artifact_includes):
            fail(errors, f"target-artifact facade still owns {implementation}")

    runtime_root = root / "lib/Wafer/Runtime"
    runtime_cmake = runtime_root / "CMakeLists.txt"
    runtime_text = read_required(runtime_cmake, errors)
    check_exact_sources(
        runtime_root, PACKAGE_MANIFEST_SOURCES, "package manifest", errors
    )
    check_private_header(
        runtime_root / "PackageManifestInternal.h",
        root / "include/Wafer/Runtime/PackageManifestInternal.h",
        "package manifest internal",
        errors,
    )
    runtime_body = cmake_target_body(
        runtime_text, "add_mlir_library", "WaferRuntime", runtime_cmake, errors
    )
    check_cmake_sources(
        body=runtime_body,
        required=PACKAGE_MANIFEST_SOURCES,
        cmake_path=runtime_cmake,
        target="WaferRuntime",
        errors=errors,
    )
    check_cmake_source_ownership(
        text=runtime_text,
        required=PACKAGE_MANIFEST_SOURCES,
        cmake_path=runtime_cmake,
        target="WaferRuntime",
        errors=errors,
    )
    manifest_facade = read_required(runtime_root / "PackageManifest.cpp", errors)
    manifest_includes = set(source_includes(manifest_facade))
    if (
        "llvm/Support/JSON.h" in manifest_includes
        or "filesystem" in manifest_includes
    ):
        fail(errors, "package-manifest facade still owns JSON or filesystem I/O")
    check_no_textual_source_includes(
        [compiler_root / name for name in compiler_sources]
        + [
            compiler_root / "CompilationInternal.h",
            compiler_root / "TargetArtifactInternal.h",
        ]
        + [runtime_root / name for name in PACKAGE_MANIFEST_SOURCES]
        + [runtime_root / "PackageManifestInternal.h"],
        "compiler/artifact/package",
        errors,
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
    check_instruction_owners(root, errors)
    check_tile_region_to_instr_owners(root, errors)
    check_numeric_semantics_owners(root, errors)
    check_group_to_tile_region_owners(root, errors)
    check_candidate_selection_owners(root, errors)
    check_memory_planning_owners(root, errors)
    check_target_llvm_owners(root, errors)
    check_numeric_dependency_owners(root, errors)
    check_frontend_program_owners(root, errors)
    check_wafer_compile_owners(root, errors)
    check_reference_executor_retired(root, errors)
    check_target_model_owners(root, errors)
    check_numeric_bulk_owners(root, errors)
    check_compiler_artifact_package_owners(root, errors)
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
