//===- CompilationInternal.h - Compiler orchestration internals -*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATIONINTERNAL_H
#define WAFER_COMPILER_COMPILATIONINTERNAL_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/Package.h"
#include "Wafer/Compiler/ProgramData.h"
#include "Wafer/Compiler/TargetCodeGen.h"
#include "Wafer/Frontend/Program.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace wafer::compiler::detail {

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message);

/// Transaction-local current stage. Each stable pipeline phase enters its
/// stage before running; on failure the transaction reports the stage so the
/// library entry can classify the error without parsing diagnostics.
struct CompilationStageTracker {
  CompilationStage current = CompilationStage::SourceVerification;

  void enter(CompilationStage stage) { current = stage; }
};

/// Digests the profile instrumentation writer computed from the staged files
/// before commit. The transaction re-verifies them against the installed
/// activation.json after the commit rename.
struct ProfileInstrumentationIdentity {
  std::string primaryManifestDigest;
  std::string planDigest;
  std::string siteMapDigest;
};

bool pathEntryExists(llvm::StringRef path);
bool isDirectory(llvm::StringRef path);
bool isRegularFile(llvm::StringRef path);
bool createDirectory(llvm::StringRef path, llvm::raw_ostream &diagnostics);
std::string programFile(llvm::StringRef programDirectory,
                        llvm::ArrayRef<llvm::StringRef> components);
bool copyDirectory(llvm::StringRef sourceDirectory,
                   llvm::StringRef destinationDirectory,
                   llvm::raw_ostream &diagnostics,
                   llvm::ArrayRef<llvm::StringRef> skipMembers = {});
bool validateRegularDirectoryTree(llvm::StringRef root,
                                  llvm::raw_ostream &diagnostics);
bool mergeMissingProgramMembers(llvm::StringRef sourceDirectory,
                                llvm::StringRef destinationDirectory,
                                llvm::raw_ostream &diagnostics);
bool writeProgramModule(mlir::ModuleOp module, llvm::StringRef programDirectory,
                        llvm::raw_ostream &diagnostics);
bool makeAbsoluteNormalizedPath(llvm::StringRef path,
                                llvm::SmallVectorImpl<char> &storage,
                                llvm::raw_ostream &diagnostics);
bool resolveThroughExistingAncestor(llvm::StringRef path,
                                    llvm::SmallVectorImpl<char> &storage,
                                    llvm::raw_ostream &diagnostics);
bool pathIsWithin(llvm::StringRef path, llvm::StringRef directory);
bool renameDirectoryNoReplace(llvm::StringRef source,
                              llvm::StringRef destination,
                              llvm::raw_ostream &diagnostics);


mlir::OwningOpRef<mlir::ModuleOp>
parseProgramDirectoryModule(llvm::StringRef programDirectory,
                            mlir::MLIRContext &context);
mlir::LogicalResult verifyProgramDirectoryMetadata(
    mlir::ModuleOp module, llvm::StringRef programDirectory,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result = nullptr,
    const wafer::frontend::ProgramPayloadResolver *resolver = nullptr);
bool hasPostSpmdMarker(mlir::ModuleOp module, llvm::StringRef programDirectory);
bool containsDialectSemantics(mlir::ModuleOp module,
                              llvm::StringRef dialectNamespace);
mlir::LogicalResult verifyStablehloStageOperations(mlir::ModuleOp module);
mlir::LogicalResult verifyTensorProgramStageOperations(mlir::ModuleOp module);
mlir::LogicalResult
verifyExactExecutionConfigInternal(mlir::ModuleOp module,
                                   const ExecutionConfig &config);
mlir::LogicalResult
materializeOrVerifyExactExecutionConfig(mlir::ModuleOp module,
                                        const ExecutionConfig &config);
void eraseTargetTopologyAndExecutionMesh(mlir::ModuleOp module);
void registerCompilationDialects(mlir::DialectRegistry &registry);
mlir::LogicalResult
runPassPipeline(mlir::ModuleOp module, llvm::StringRef pipelineLabel,
                llvm::function_ref<void(mlir::OpPassManager &)> builder);

mlir::LogicalResult runSpmdHelper(llvm::StringRef helper,
                                  llvm::StringRef inputProgramDirectory,
                                  llvm::StringRef outputProgramDirectory,
                                  const ExecutionConfig &config,
                                  llvm::raw_ostream &diagnostics);

llvm::Expected<CardExecutable>
compileTensorProgramToCardExecutable(
    llvm::StringRef tensorProgramDirectory, ExecutionConfig executionConfig,
    OptimizationConfig optimizations, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace);

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<CardExecutable> &cardExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace, CompilationStageTracker &stages);

mlir::LogicalResult stageProfileTargetPackages(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<CardExecutable> &cardExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace, CompilationStageTracker &stages,
    ProfileInstrumentationIdentity &profileIdentity);

mlir::LogicalResult runCompilationTransaction(
    CompilationRequest request, llvm::StringRef outputPackageDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    CompilationOptions options, std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<CardExecutable> *retainedCardExecutable,
    std::optional<TargetLLVMModules> *retainedTargetLLVMModules,
    std::optional<CompilationIRTrace> *retainedIRTrace = nullptr,
    std::optional<ExecutablePackage> *retainedPackage = nullptr,
    std::optional<ProfileInstrumentationProduct> *retainedProfileProduct =
        nullptr,
    CompilationStage *failureStage = nullptr,
    bool failCommitVerification = false);

/// Pre-publication binding for the co-committed profile instrumentation:
/// digests the staged ordinary manifest and requires the staged
/// activation.json to carry exactly that primary digest and the staged
/// plan/site-map digests. The single publication rename then makes the
/// verified inodes visible, so no post-rename readback exists. Returns an
/// error on any mismatch; the runtime strict loader remains the semantic
/// reader at launch.
llvm::Error verifyProfileInstrumentationBinding(
    llvm::StringRef packageRoot, llvm::StringRef instrumentationRoot,
    const ProfileInstrumentationIdentity &identity);

} // namespace wafer::compiler::detail

namespace wafer::compiler {

/// Internal constructor for the committed profile instrumentation product.
struct ProfileInstrumentationProductBuilder {
  static ProfileInstrumentationProduct
  make(llvm::StringRef rootDirectory,
       const detail::ProfileInstrumentationIdentity &identity) {
    return ProfileInstrumentationProduct(rootDirectory,
                                         identity.primaryManifestDigest,
                                         identity.planDigest,
                                         identity.siteMapDigest);
  }
};

} // namespace wafer::compiler

#endif // WAFER_COMPILER_COMPILATIONINTERNAL_H
