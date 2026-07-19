//===- CompilationInternal.h - Compiler orchestration internals -*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATIONINTERNAL_H
#define WAFER_COMPILER_COMPILATIONINTERNAL_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/Support/OptimizationQualification.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace wafer::compiler::detail {

struct CompilationOptimizationPolicyV1 {
  OptimizationQualificationProposal proposal;
  OptimizationConfiguration optimizationConfiguration;
  EquivalentInputVariantV1 inputVariant = EquivalentInputVariantV1::Original;
};

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message);

bool pathEntryExists(llvm::StringRef path);
bool isDirectory(llvm::StringRef path);
bool isRegularFile(llvm::StringRef path);
bool createDirectory(llvm::StringRef path, llvm::raw_ostream &diagnostics);
std::string programFile(llvm::StringRef programDirectory,
                        llvm::ArrayRef<llvm::StringRef> components);
bool copyDirectory(llvm::StringRef sourceDirectory,
                   llvm::StringRef destinationDirectory,
                   llvm::raw_ostream &diagnostics);
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
bool publishDirectoryNoReplace(llvm::StringRef source,
                               llvm::StringRef destination,
                               llvm::raw_ostream &diagnostics);

mlir::OwningOpRef<mlir::ModuleOp>
parseProgramDirectoryModule(llvm::StringRef programDirectory,
                            mlir::MLIRContext &context);
mlir::LogicalResult verifyProgramDirectoryMetadata(
    mlir::ModuleOp module, llvm::StringRef programDirectory,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result = nullptr);
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
bool runPassPipeline(mlir::ModuleOp module,
                     void (*builder)(mlir::OpPassManager &));

bool runSpmdHelper(llvm::StringRef helper,
                   llvm::StringRef inputProgramDirectory,
                   llvm::StringRef outputProgramDirectory,
                   const ExecutionConfig &config,
                   llvm::raw_ostream &diagnostics);

llvm::Expected<ExecutableBundle> compileTensorProgramToExecutableBundleImpl(
    llvm::StringRef tensorProgramDirectory, ExecutionConfig executionConfig,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLogicalRank,
    const CompilationOptimizationPolicyV1 &optimizationPolicy);

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<ExecutableBundle> &executableBundle,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle,
    const CompilationOptimizationPolicyV1 &optimizationPolicy);

mlir::LogicalResult runCompilationTransaction(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<ExecutableBundle> *retainedExecutableBundle,
    std::optional<TargetLLVMModuleBundle> *retainedTargetLLVMModuleBundle,
    const CompilationOptimizationPolicyV1 &optimizationPolicy);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COMPILATIONINTERNAL_H
