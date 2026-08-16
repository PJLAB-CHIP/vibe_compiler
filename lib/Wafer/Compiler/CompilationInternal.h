//===- CompilationInternal.h - Compiler orchestration internals -*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATIONINTERNAL_H
#define WAFER_COMPILER_COMPILATIONINTERNAL_H

#include "Wafer/Compiler/Compilation.h"
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

using DirectoryRenameFunction = bool (*)(llvm::StringRef source,
                                         llvm::StringRef destination,
                                         llvm::raw_ostream &diagnostics);

/// Renames a package and profile directory to their final paths. If the second
/// rename fails, the package is renamed back to its staging path.
mlir::LogicalResult renamePackageAndProfileNoReplace(
    llvm::StringRef stagedPackage, llvm::StringRef outputPackage,
    llvm::StringRef stagedInstrumentation,
    llvm::StringRef outputInstrumentation, llvm::raw_ostream &diagnostics,
    DirectoryRenameFunction renameDirectory);

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
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace);

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<CardExecutable> &cardExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace);

mlir::LogicalResult stageProfileTargetPackages(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<CardExecutable> &cardExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace);

mlir::LogicalResult runCompilationTransaction(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    CompilationOptions options, std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<CardExecutable> *retainedCardExecutable,
    std::optional<TargetLLVMModules> *retainedTargetLLVMModules,
    std::optional<CompilationIRTrace> *retainedIRTrace = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COMPILATIONINTERNAL_H
