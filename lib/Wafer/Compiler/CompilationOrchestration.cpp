//===- CompilationOrchestration.cpp - Compiler transaction orchestration ===//

#include "CompilationInternal.h"

#include "Wafer/Pipelines/Pipelines.h"

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

namespace wafer::compiler::detail {

mlir::LogicalResult runCompilationTransaction(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionMode selectionMode, CompilationOptions options,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<ExecutableBundle> *retainedExecutableBundle,
    std::optional<TargetLLVMModuleBundle> *retainedTargetLLVMModuleBundle) {
#if !defined(WAFER_ENABLE_STABLEHLO) || !defined(WAFER_ENABLE_SHARDY)
  (void)request;
  (void)outputProgramDirectory;
  (void)xlaSpmdPartitionerHelper;
  (void)targetToolchain;
  (void)selectionMode;
  (void)options;
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
  if (options.shouldProduceProfileCompanion() &&
      selectionMode != WholeVariantSelectionMode::Production) {
    reject(diagnostics,
           "profile companion requires production whole-variant selection");
    return mlir::failure();
  }
  if (options.shouldProduceProfileCompanion() &&
      request.getExecutionConfig().getRankCount() != 16) {
    reject(diagnostics, "profile compilation requires execution-ranks=16");
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
  llvm::SmallString<256> canonicalProfileOutput(canonicalOutput);
  canonicalProfileOutput += ".profile";
  if (options.shouldProduceProfileCompanion() &&
      pathEntryExists(canonicalProfileOutput)) {
    reject(diagnostics,
           "refusing to replace existing profile companion directory: '" +
               canonicalProfileOutput.str().str() + "'");
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

  mlir::OwningOpRef<mlir::ModuleOp> sourceModule =
      parseProgramDirectoryModule(sourceSnapshot, context);
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
  if (runPassPipeline(*tensorModule, wafer::buildStablehloToLinalgPipeline))
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
  if (options.shouldProduceProfileCompanion()) {
    if (mlir::failed(stageProfileTargetPackages(
            tensorProgram, transactionRoot, outputName,
            request.getExecutionConfig(), targetToolchain, diagnostics,
            failAfterLogicalRank, failAfterTargetLogicalRank,
            failAfterPackageLogicalRank, executableBundle, targetLLVMModules)))
      return mlir::failure();
  } else if (mlir::failed(stageTargetPackage(
                 tensorProgram, transactionRoot, request.getExecutionConfig(),
                 targetToolchain, diagnostics, selectionMode,
                 failAfterLogicalRank, failAfterTargetLogicalRank,
                 failAfterPackageLogicalRank, executableBundle,
                 targetLLVMModules))) {
    return mlir::failure();
  }

  llvm::SmallString<256> stagedPackage(transactionRoot);
  llvm::sys::path::append(stagedPackage, "package");
  if (publishDirectoryNoReplace(stagedPackage, canonicalOutput, diagnostics))
    return mlir::failure();
  if (options.shouldProduceProfileCompanion()) {
    llvm::SmallString<256> stagedCompanion(transactionRoot);
    llvm::sys::path::append(stagedCompanion, "profile-companion");
    if (publishDirectoryNoReplace(stagedCompanion, canonicalProfileOutput,
                                  diagnostics)) {
      if (std::error_code rollbackError =
              llvm::sys::fs::rename(canonicalOutput, stagedPackage))
        reject(diagnostics,
               "failed to roll back package after profile companion "
               "publication failure: " +
                   rollbackError.message());
      return mlir::failure();
    }
  }
  if (retainedExecutableBundle)
    retainedExecutableBundle->emplace(std::move(*executableBundle));
  if (retainedTargetLLVMModuleBundle)
    retainedTargetLLVMModuleBundle->emplace(std::move(*targetLLVMModules));
  return mlir::success();
#endif
}

} // namespace wafer::compiler::detail
