//===- CompilationOrchestration.cpp - Compiler transaction orchestration ===//

#include "../Pipelines/QualificationInternal.h"
#include "CompilationInternal.h"

#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Support/OptimizationArtifactDigest.h"
#include "Wafer/Support/OptimizationMechanism.h"
#include "Wafer/Target/TargetProfile.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

static std::optional<OptimizationDigest>
digestFrontendImportInput(llvm::StringRef sourceProgramDirectory,
                          const ExecutionConfig &config) {
  std::optional<OptimizationDigest> directory =
      digestOptimizationArtifactDirectoryV1(
          sourceProgramDirectory, "wafer.frontend-import-program-directory-v1");
  if (!directory)
    return std::nullopt;
  std::vector<uint8_t> bytes(directory->begin(), directory->end());
  uint64_t rankCount = config.getRankCount();
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(rankCount >> shift));
  llvm::StringRef profile =
      stringifyTargetProfileId(config.getTargetProfileId());
  bytes.insert(bytes.end(), profile.bytes_begin(), profile.bytes_end());
  return digestOptimizationBytesV1("wafer.frontend-import-input-snapshot-v1",
                                   bytes);
}

mlir::LogicalResult runCompilationTransaction(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<ExecutableBundle> *retainedExecutableBundle,
    std::optional<TargetLLVMModuleBundle> *retainedTargetLLVMModuleBundle,
    const CompilationOptimizationPolicyV1 &optimizationPolicy) {
#if !defined(WAFER_ENABLE_STABLEHLO) || !defined(WAFER_ENABLE_SHARDY)
  (void)request;
  (void)outputProgramDirectory;
  (void)xlaSpmdPartitionerHelper;
  (void)targetToolchain;
  (void)failAfterLogicalRank;
  (void)failAfterTargetLogicalRank;
  (void)failAfterPackageLogicalRank;
  (void)retainedExecutableBundle;
  (void)retainedTargetLLVMModuleBundle;
  reject(diagnostics,
         "StableHLO and SPMD partitioner dependencies are required");
  return mlir::failure();
#else
  if (outputProgramDirectory.empty()) {
    reject(diagnostics, "output program directory must not be empty");
    return mlir::failure();
  }
  if (xlaSpmdPartitionerHelper.empty()) {
    reject(diagnostics, "XLA SPMD partitioner helper path must not be empty");
    return mlir::failure();
  }

  llvm::SmallString<256> canonicalSource;
  if (std::error_code error = llvm::sys::fs::real_path(
          request.getSourceProgramDirectory(), canonicalSource)) {
    reject(diagnostics, "failed to resolve source program directory '" +
                            request.getSourceProgramDirectory().str() +
                            "': " + error.message());
    return mlir::failure();
  }
  if (!isDirectory(canonicalSource)) {
    reject(diagnostics, "source program directory is not a directory: '" +
                            canonicalSource.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> absoluteOutput;
  if (makeAbsoluteNormalizedPath(outputProgramDirectory, absoluteOutput,
                                 diagnostics))
    return mlir::failure();
  llvm::StringRef outputName = llvm::sys::path::filename(absoluteOutput);
  if (outputName.empty()) {
    reject(diagnostics, "output program directory must name a directory");
    return mlir::failure();
  }
  if (pathEntryExists(absoluteOutput)) {
    reject(diagnostics,
           "refusing to replace existing output program directory: '" +
               absoluteOutput.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> outputParent(absoluteOutput);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";

  llvm::SmallString<256> prospectiveCanonicalOutputParent;
  if (resolveThroughExistingAncestor(
          outputParent, prospectiveCanonicalOutputParent, diagnostics))
    return mlir::failure();
  llvm::SmallString<256> prospectiveCanonicalOutput(
      prospectiveCanonicalOutputParent);
  llvm::sys::path::append(prospectiveCanonicalOutput, outputName);
  if (pathIsWithin(prospectiveCanonicalOutput, canonicalSource)) {
    reject(diagnostics,
           "output program directory must not equal or be nested under the "
           "source program directory");
    return mlir::failure();
  }

  if (createDirectory(outputParent, diagnostics))
    return mlir::failure();

  llvm::SmallString<256> canonicalOutputParent;
  if (std::error_code error =
          llvm::sys::fs::real_path(outputParent, canonicalOutputParent)) {
    reject(diagnostics, "failed to resolve output parent directory '" +
                            outputParent.str().str() + "': " + error.message());
    return mlir::failure();
  }
  llvm::SmallString<256> canonicalOutput(canonicalOutputParent);
  llvm::sys::path::append(canonicalOutput, outputName);
  if (pathIsWithin(canonicalOutput, canonicalSource)) {
    reject(diagnostics,
           "output program directory must not equal or be nested under the "
           "source program directory");
    return mlir::failure();
  }
  if (pathEntryExists(canonicalOutput)) {
    reject(diagnostics,
           "refusing to replace existing output program directory: '" +
               canonicalOutput.str().str() + "'");
    return mlir::failure();
  }

  llvm::SmallString<256> stagingPrefix(canonicalOutputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-compile-staging");
  llvm::SmallString<256> transactionRoot;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          stagingPrefix, transactionRoot)) {
    reject(diagnostics, "failed to create compilation staging directory: " +
                            error.message());
    return mlir::failure();
  }
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(transactionRoot); });

  llvm::SmallString<256> sourceSnapshot(transactionRoot);
  llvm::sys::path::append(sourceSnapshot, "source");
  if (copyDirectory(canonicalSource, sourceSnapshot, diagnostics))
    return mlir::failure();

  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        diagnostic.print(diagnostics);
        diagnostics << "\n";
        return mlir::success();
      });

  std::optional<OptimizationDigest> frontendInput =
      digestFrontendImportInput(sourceSnapshot, request.getExecutionConfig());
  if (!frontendInput) {
    reject(diagnostics, "cannot establish frontend import input digest");
    return mlir::failure();
  }
  OptimizationInvocationTokenV1 frontendImportToken;
  std::string frontendTelemetryDiagnostic;
  if (!beginOptimizationInvocationV1(
          mechanism::FrontendProgramImport,
          OptimizationCutPoint::FrontendProgramImport,
          /*invocationOrdinal=*/0, *frontendInput, frontendImportToken,
          &frontendTelemetryDiagnostic)) {
    reject(diagnostics, "cannot begin frontend import invocation: " +
                            frontendTelemetryDiagnostic);
    return mlir::failure();
  }
  mlir::OwningOpRef<mlir::ModuleOp> sourceModule =
      parseProgramDirectoryModule(sourceSnapshot, context);
  OptimizationInvocationTelemetry frontendTelemetry;
  frontendTelemetry.key = mechanism::FrontendProgramImport;
  frontendTelemetry.cutPoint = OptimizationCutPoint::FrontendProgramImport;
  frontendTelemetry.outcome =
      sourceModule ? InvocationOutcome::NoChange : InvocationOutcome::Invalid;
  frontendTelemetry.inputSnapshotDigest = *frontendInput;
  if (sourceModule)
    sourceModule->walk(
        [&](mlir::Operation *) { ++frontendTelemetry.workUnits; });
  if (!commitOptimizationInvocationV1(frontendImportToken, frontendTelemetry,
                                      &frontendTelemetryDiagnostic)) {
    reject(diagnostics, "invalid frontend import invocation terminal: " +
                            frontendTelemetryDiagnostic);
    return mlir::failure();
  }
  if (!sourceModule)
    return mlir::failure();
  if (hasPostSpmdMarker(*sourceModule, sourceSnapshot)) {
    reject(diagnostics,
           "source program is already SPMD-partitioned; typed compilation "
           "requires an exporter program");
    return mlir::failure();
  }
  if (containsDialectSemantics(*sourceModule, "sdy")) {
    reject(diagnostics,
           "source program contains Shardy semantics that the external XLA "
           "SPMD helper boundary cannot consume");
    return mlir::failure();
  }
  if (mlir::failed(verifyStablehloStageOperations(*sourceModule)))
    return mlir::failure();
  if (mlir::failed(materializeOrVerifyExactExecutionConfig(
          *sourceModule, request.getExecutionConfig())))
    return mlir::failure();
  if (mlir::failed(verifyProgramDirectoryMetadata(*sourceModule, sourceSnapshot,
                                                  diagnostics)))
    return mlir::failure();
  llvm::SmallString<256> propagatedProgram(transactionRoot);
  llvm::sys::path::append(propagatedProgram, "propagated");
  if (copyDirectory(sourceSnapshot, propagatedProgram, diagnostics))
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> helperModule = sourceModule->clone();
  eraseTargetTopologyAndExecutionMesh(*helperModule);
  if (mlir::failed(mlir::verify(*helperModule)))
    return mlir::failure();
  if (writeProgramModule(*helperModule, propagatedProgram, diagnostics))
    return mlir::failure();

  llvm::SmallString<256> tensorProgram(transactionRoot);
  llvm::sys::path::append(tensorProgram, "tensor-program");
  if (runSpmdHelper(xlaSpmdPartitionerHelper, propagatedProgram, tensorProgram,
                    request.getExecutionConfig(), diagnostics))
    return mlir::failure();

  if (validateRegularDirectoryTree(tensorProgram, diagnostics))
    return mlir::failure();
  for (llvm::StringRef requiredMember :
       {llvm::StringRef("forward.mlir"), llvm::StringRef("forward.meta")}) {
    if (!isRegularFile(programFile(
            tensorProgram, {llvm::StringRef("functions"), requiredMember}))) {
      reject(diagnostics,
             "XLA SPMD partitioner output is missing required regular member "
             "'functions/" +
                 requiredMember.str() + "'");
      return mlir::failure();
    }
  }
  if (!isRegularFile(programFile(
          tensorProgram, {llvm::StringRef("functions"),
                          llvm::StringRef("forward.parameter_shards.json")}))) {
    reject(diagnostics,
           "XLA SPMD partitioner output is missing its partition marker");
    return mlir::failure();
  }
  if (mergeMissingProgramMembers(sourceSnapshot, tensorProgram, diagnostics) ||
      validateRegularDirectoryTree(tensorProgram, diagnostics))
    return mlir::failure();

  mlir::OwningOpRef<mlir::ModuleOp> tensorModule =
      parseProgramDirectoryModule(tensorProgram, context);
  if (!tensorModule)
    return mlir::failure();
  if (mlir::failed(materializeOrVerifyExactExecutionConfig(
          *tensorModule, request.getExecutionConfig())))
    return mlir::failure();
  if (!hasPostSpmdMarker(*tensorModule, tensorProgram)) {
    reject(diagnostics,
           "XLA SPMD partitioner output is missing its partition marker");
    return mlir::failure();
  }
  if (containsDialectSemantics(*tensorModule, "sdy")) {
    reject(diagnostics,
           "XLA SPMD partitioner output still contains Shardy semantics");
    return mlir::failure();
  }
  if (mlir::failed(verifyStablehloStageOperations(*tensorModule)))
    return mlir::failure();
  if (mlir::failed(verifyProgramDirectoryMetadata(*tensorModule, tensorProgram,
                                                  diagnostics)))
    return mlir::failure();
  mlir::PassManager tensorPipeline(tensorModule->getContext());
  std::string optimizationDiagnostic;
  if (!wafer::qualification_internal::buildStablehloToLinalgPipeline(
          tensorPipeline, optimizationPolicy.proposal,
          optimizationPolicy.optimizationConfiguration, &optimizationDiagnostic,
          optimizationPolicy.inputVariant,
          optimizationPolicy.requiredTensorNormalizationRepetitions)) {
    reject(diagnostics,
           "invalid compile optimization policy: " + optimizationDiagnostic);
    return mlir::failure();
  }
  if (mlir::failed(tensorPipeline.run(*tensorModule)))
    return mlir::failure();
  if (containsDialectSemantics(*tensorModule, "stablehlo") ||
      containsDialectSemantics(*tensorModule, "sdy")) {
    reject(diagnostics,
           "tensor program still contains frontend or sharding operations");
    return mlir::failure();
  }
  if (mlir::failed(verifyTensorProgramStageOperations(*tensorModule)))
    return mlir::failure();
  if (writeProgramModule(*tensorModule, tensorProgram, diagnostics))
    return mlir::failure();

  mlir::OwningOpRef<mlir::ModuleOp> verifiedTensorModule =
      parseProgramDirectoryModule(tensorProgram, context);
  if (!verifiedTensorModule)
    return mlir::failure();
  if (mlir::failed(verifyExactExecutionConfigInternal(
          *verifiedTensorModule, request.getExecutionConfig())))
    return mlir::failure();
  if (!hasPostSpmdMarker(*verifiedTensorModule, tensorProgram)) {
    reject(diagnostics,
           "tensor-program readback is missing its partition marker");
    return mlir::failure();
  }
  if (containsDialectSemantics(*verifiedTensorModule, "stablehlo") ||
      containsDialectSemantics(*verifiedTensorModule, "sdy")) {
    reject(diagnostics,
           "tensor-program readback contains frontend or sharding semantics");
    return mlir::failure();
  }
  if (mlir::failed(verifyTensorProgramStageOperations(*verifiedTensorModule)) ||
      mlir::failed(verifyProgramDirectoryMetadata(*verifiedTensorModule,
                                                  tensorProgram, diagnostics)))
    return mlir::failure();

  std::optional<ExecutableBundle> executableBundle;
  std::optional<TargetLLVMModuleBundle> targetLLVMModules;
  if (mlir::failed(stageTargetPackage(
          tensorProgram, transactionRoot, request.getExecutionConfig(),
          targetToolchain, diagnostics, failAfterLogicalRank,
          failAfterTargetLogicalRank, failAfterPackageLogicalRank,
          executableBundle, targetLLVMModules, optimizationPolicy)))
    return mlir::failure();

  llvm::SmallString<256> stagedPackage(transactionRoot);
  llvm::sys::path::append(stagedPackage, "package");
  if (publishDirectoryNoReplace(stagedPackage, canonicalOutput, diagnostics))
    return mlir::failure();
  if (retainedExecutableBundle)
    retainedExecutableBundle->emplace(std::move(*executableBundle));
  if (retainedTargetLLVMModuleBundle)
    retainedTargetLLVMModuleBundle->emplace(std::move(*targetLLVMModules));
  return mlir::success();
#endif
}

} // namespace wafer::compiler::detail
