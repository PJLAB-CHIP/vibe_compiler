//===- CompilationInternal.h - Compiler orchestration internals -*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATIONINTERNAL_H
#define WAFER_COMPILER_COMPILATIONINTERNAL_H

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/Compilation.h"
#include "Wafer/Driver/CompilationResult.h"
#include "Wafer/Driver/ProgramData.h"
#include "Wafer/Frontend/Program.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message);

/// Transaction-local current stage. Each stable pipeline phase enters its
/// stage before running; on failure the transaction reports the stage so the
/// library entry can classify the error without parsing diagnostics.
struct CompilationStageTracker {
  CompilationStage current = CompilationStage::SourceVerification;

  void enter(CompilationStage stage) { current = stage; }
};

/// Digests the profile instrumentation writer computed from the staged files.
/// The transaction binds those exact staged bytes before the commit rename.
struct ProfileInstrumentationIdentity {
  std::string primaryManifestDigest;
  std::string planDigest;
  std::string siteMapDigest;
  std::array<std::string, 2> captureManifestDigests;
};

struct BoundProfileInstrumentation {
  std::unique_ptr<llvm::MemoryBuffer> activationBuffer;
  std::unique_ptr<llvm::MemoryBuffer> planBuffer;
  std::unique_ptr<llvm::MemoryBuffer> siteMapBuffer;
  std::vector<runtime::detail::BoundExecutablePackage> capturePackages;
};

enum class CommitFailureInjection {
  None,
  CorruptPackageProgramData,
  CorruptProfilePlan,
  FailAfterVerification,
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
mlir::LogicalResult runSpmdHelper(llvm::StringRef helper,
                                  llvm::StringRef inputProgramDirectory,
                                  llvm::StringRef outputProgramDirectory,
                                  const ExecutionConfig &config,
                                  llvm::raw_ostream &diagnostics);

llvm::Expected<DeviceExecutable> compileTensorProgramToDeviceExecutable(
    llvm::StringRef tensorProgramDirectory, ExecutionConfig executionConfig,
    OptimizationConfig optimizations, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot, ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace);

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<DeviceExecutable> &deviceExecutable,
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
    std::optional<DeviceExecutable> &deviceExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace, CompilationStageTracker &stages,
    ProfileInstrumentationIdentity &profileIdentity);

mlir::LogicalResult runCompilationTransaction(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    CompilationOptions options, std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<DeviceExecutable> *retainedDeviceExecutable,
    std::optional<TargetLLVMModules> *retainedTargetLLVMModules,
    std::optional<CompilationIRTrace> *retainedIRTrace = nullptr,
    std::optional<ExecutablePackage> *retainedPackage = nullptr,
    std::optional<ProfileInstrumentationProduct> *retainedProfileProduct =
        nullptr,
    CompilationStage *failureStage = nullptr,
    CommitFailureInjection commitFailureInjection =
        CommitFailureInjection::None);

/// Binds the all-and-only staged instrumentation resources and checks their
/// actual bytes against the writer identity. Capture packages are strictly
/// loaded through the shared package binder; no path is reopened after commit.
llvm::Expected<BoundProfileInstrumentation> bindProfileInstrumentation(
    const runtime::detail::BoundExecutablePackage &primaryPackage,
    llvm::StringRef instrumentationRoot,
    const ProfileInstrumentationIdentity &identity);

} // namespace wafer::compiler::detail

namespace wafer::compiler {

/// Internal constructor for the committed profile instrumentation product.
struct ProfileInstrumentationProductBuilder {
  static ProfileInstrumentationProduct
  make(std::string rootDirectory,
       detail::ProfileInstrumentationIdentity identity,
       detail::BoundProfileInstrumentation instrumentation) {
    return ProfileInstrumentationProduct(
        std::move(rootDirectory), std::move(identity.primaryManifestDigest),
        std::move(identity.planDigest), std::move(identity.siteMapDigest),
        std::move(instrumentation.activationBuffer),
        std::move(instrumentation.planBuffer),
        std::move(instrumentation.siteMapBuffer),
        std::move(instrumentation.capturePackages));
  }
};

} // namespace wafer::compiler

#endif // WAFER_COMPILER_COMPILATIONINTERNAL_H
