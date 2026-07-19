//===- TargetArtifactInternal.h - Internal target build --------*- C++ -*-===//

#ifndef WAFER_COMPILER_TARGETARTIFACTINTERNAL_H
#define WAFER_COMPILER_TARGETARTIFACTINTERNAL_H

#include "Wafer/Compiler/TargetArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

struct TargetLLVMModuleBundleBuilder {
  static TargetLLVMModule
  makeModule(int64_t logicalRank, llvm::StringRef entrySymbol,
             TargetProfileId targetProfile, TargetIdentityId targetIdentity,
             KernelRuntimeABIId kernelRuntimeABI, llvm::StringRef moduleFormat,
             std::vector<KernelABISlot> kernelABISlots,
             std::unique_ptr<llvm::LLVMContext> context,
             std::unique_ptr<llvm::Module> module) {
    return TargetLLVMModule(logicalRank, entrySymbol, targetProfile,
                            targetIdentity, kernelRuntimeABI, moduleFormat,
                            std::move(kernelABISlots), std::move(context),
                            std::move(module));
  }

  static TargetLLVMModuleBundle
  makeBundle(ExecutionConfig executionConfig,
             std::vector<TargetLLVMModule> modules) {
    return TargetLLVMModuleBundle(executionConfig, std::move(modules));
  }
};

struct TargetArtifactBundleBuilder {
  static VerifiedTargetModule
  makeModule(int64_t logicalRank, llvm::StringRef entrySymbol,
             llvm::StringRef relativePath, llvm::StringRef contentDigest,
             TargetProfileId targetProfile, TargetIdentityId targetIdentity,
             KernelRuntimeABIId kernelRuntimeABI, llvm::StringRef moduleFormat,
             std::vector<KernelABISlot> kernelABISlots) {
    return VerifiedTargetModule(logicalRank, entrySymbol, relativePath,
                                contentDigest, targetProfile, targetIdentity,
                                kernelRuntimeABI, moduleFormat,
                                std::move(kernelABISlots));
  }

  static TargetArtifactBundle
  makeBundle(llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
             std::vector<VerifiedTargetModule> modules) {
    return TargetArtifactBundle(rootDirectory, executionConfig,
                                std::move(modules));
  }
};

namespace detail {

struct PreparedTargetRank {
  explicit PreparedTargetRank(const ExecutionConfig &executionConfig);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<KernelABISlot> slots;
  TargetProfileId targetProfile;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
  std::string moduleFormat;
  int64_t logicalRank = -1;
  int64_t defaultDDRArenaArgumentIndex = -1;
  int64_t transportStatusArgumentIndex = -1;
};

enum class TargetLLVMLoweringPurpose : uint8_t {
  WholeVariantCandidateProof,
  PublishedArtifact,
  Testing,
};

struct TargetLLVMLoweringInvocation {
  TargetLLVMLoweringPurpose purpose;
  uint64_t attemptOrdinal = 0;
};

struct TargetModuleReadback {
  std::string contentDigest;
  std::string moduleFormat;
};

llvm::Error fail(llvm::raw_ostream &diagnostics, llvm::StringRef message);
bool isRegularTargetFile(llvm::StringRef path);

mlir::FailureOr<PreparedTargetRank>
prepareTargetABI(const RankExecutable &rankExecutable,
                 const ExecutionConfig &executionConfig);
mlir::LogicalResult
lowerToTargetLLVM(PreparedTargetRank &prepared,
                  TargetLLVMLoweringInvocation invocation);
mlir::LogicalResult verifyLoweredKernelABI(PreparedTargetRank &prepared,
                                           llvm::StringRef entrySymbol);
llvm::Expected<TargetLLVMModule>
translatePreparedTargetRank(PreparedTargetRank prepared,
                            llvm::StringRef entrySymbol);
llvm::Error verifyTargetLLVMModule(const llvm::Module &module,
                                   int64_t expectedLogicalRank,
                                   llvm::StringRef expectedEntrySymbol,
                                   TargetProfileId expectedProfile,
                                   TargetIdentityId expectedTargetIdentity,
                                   KernelRuntimeABIId expectedKernelRuntimeABI,
                                   llvm::StringRef expectedModuleFormat,
                                   llvm::ArrayRef<KernelABISlot> expectedSlots);

llvm::Error writeLLVMIR(const llvm::Module &module, llvm::StringRef path);
llvm::Error runDeviceLink(const TargetToolchain &toolchain,
                          llvm::StringRef llvmIR, llvm::StringRef module,
                          llvm::StringRef object, llvm::StringRef crtObject,
                          int64_t logicalRank);
llvm::Expected<TargetModuleReadback>
verifyTargetModule(llvm::StringRef path, llvm::StringRef entrySymbol,
                   TargetProfileId expectedProfile);

/// Runs production entry-only ABI preparation, target lowering, and lowered
/// entry verification on an owned clone. This narrow hook lets unit tests
/// prove accepted multi-function closure handling without invoking the
/// external object/link toolchain.
mlir::LogicalResult
lowerTargetABIForTesting(const RankExecutable &rankExecutable,
                         const ExecutionConfig &executionConfig);

/// Verifies a genuinely linked module with the production ELF readback path
/// and exposes the immutable typed facts that production stores per rank.
llvm::Expected<VerifiedTargetModule>
verifyLinkedTargetModuleForTesting(llvm::StringRef path,
                                   llvm::StringRef entrySymbol,
                                   TargetProfileId targetProfile);

/// Re-runs the production target LLVM module readback against the immutable
/// typed facts stored by the owner-backed entry.
llvm::Error
verifyTargetLLVMModuleForTesting(const TargetLLVMModule &targetModule);

llvm::Expected<TargetLLVMModuleBundle>
compileExecutableBundleToTargetLLVMModulesImpl(
    const ExecutableBundle &executableBundle, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank);

llvm::Expected<TargetArtifactBundle>
compileExecutableBundleToTargetArtifactsImpl(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank);

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_TARGETARTIFACTINTERNAL_H
