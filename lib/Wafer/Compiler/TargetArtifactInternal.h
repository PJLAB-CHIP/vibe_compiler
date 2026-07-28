//===- TargetArtifactInternal.h - Internal target build --------*- C++ -*-===//

#ifndef WAFER_COMPILER_TARGETARTIFACTINTERNAL_H
#define WAFER_COMPILER_TARGETARTIFACTINTERNAL_H

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Target/TargetCall.h"

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
             RuntimeLaunchContract runtimeLaunchContract,
             std::vector<TargetLLVMModule> modules) {
    return TargetLLVMModuleBundle(
        executionConfig, std::move(runtimeLaunchContract), std::move(modules));
  }
};

struct TargetArtifactBundleBuilder {
  static VerifiedTargetExport makeExport(TargetExportRole role,
                                         llvm::StringRef symbol) {
    return VerifiedTargetExport(role, symbol);
  }

  static VerifiedTargetModule
  makeModule(TargetArtifactModuleId id, llvm::StringRef relativePath,
             llvm::StringRef contentDigest, TargetProfileId targetProfile,
             TargetIdentityId targetIdentity,
             KernelRuntimeABIId kernelRuntimeABI, llvm::StringRef moduleFormat,
             std::vector<VerifiedTargetExport> exports) {
    return VerifiedTargetModule(id, relativePath, contentDigest, targetProfile,
                                targetIdentity, kernelRuntimeABI, moduleFormat,
                                std::move(exports));
  }

  static VerifiedTargetRankInterface
  makeRankInterface(int64_t logicalRank, TargetArtifactModuleId moduleId,
                    std::vector<KernelABISlot> kernelABISlots) {
    return VerifiedTargetRankInterface(logicalRank, moduleId,
                                       std::move(kernelABISlots));
  }

  static TargetArtifactBundle
  makeBundle(llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
             RuntimeLaunchContract runtimeLaunchContract,
             std::vector<VerifiedTargetModule> modules,
             std::vector<VerifiedTargetRankInterface> rankInterfaces) {
    return TargetArtifactBundle(rootDirectory, executionConfig,
                                std::move(runtimeLaunchContract),
                                std::move(modules), std::move(rankInterfaces));
  }
};

namespace detail {

inline constexpr llvm::StringLiteral kKernelPrepareExportSymbol =
    "__wafer_kernel_prepare";

/// Compiler-internal target capture kinds. They are implementation details of
/// the single profiling product and are never user-selectable driver modes.
enum class ProfileCaptureKind : uint8_t { None, Summary, Count, Trace };

llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture);
uint64_t getProfileCaptureRecordBytes(ProfileCaptureKind capture);
llvm::Error
verifyProfileCaptureKernelABISlots(llvm::ArrayRef<KernelABISlot> slots,
                                   ProfileCaptureKind capture);

struct ProfileTSMCallSite {
  uint64_t siteId = 0;
  uint64_t functionOrdinal = 0;
  uint64_t blockOrdinal = 0;
  uint64_t instructionOrdinal = 0;
  uint64_t targetCallOrdinal = 0;
  std::string targetCallSymbol;
  TargetCallTSMEngine engine = TargetCallTSMEngine::CT;
  std::string correlationKey;
};

/// Collects all and only entry-reachable registered target calls whose closed
/// semantic mapping reaches one of the five NCC execution helpers or the
/// Direct-DTE wait/completion helper.
llvm::Expected<std::vector<ProfileTSMCallSite>>
collectProfileTSMCallSites(const llvm::Module &module,
                           llvm::StringRef entrySymbol);

/// Verifies that a trace clone preserves the final production artifact's
/// rank-local site order and exact typed target-call identity. Instrumentation
/// may shift LLVM instruction ordinals, so only the production module owns the
/// canonical source position written to the companion.
llvm::Error verifyProfileTargetCallSiteIdentity(
    const llvm::Module &productionModule, llvm::StringRef productionEntrySymbol,
    const llvm::Module &traceModule, llvm::StringRef traceEntrySymbol);

/// Adds profile entry bracketing for every non-None capture and per-site
/// bracketing only for Trace. Count relies on the profile CRT helpers to count
/// the real NCC issue and Direct-DTE wait calls without per-site branches.
llvm::Error instrumentProfileTargetModule(llvm::Module &module,
                                          llvm::StringRef entrySymbol,
                                          ProfileCaptureKind capture);
llvm::Error
verifyProfileTargetModuleInstrumentation(const llvm::Module &module,
                                         llvm::StringRef entrySymbol,
                                         ProfileCaptureKind capture);

/// Owns one synthesized LLVM module and its uniquing context. Declaration
/// order ensures the module is destroyed before its context.
struct OwnedTargetLLVMModule {
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
};

struct PreparedTargetRank {
  PreparedTargetRank(const ExecutionConfig &executionConfig,
                     bool transportPreparedBeforeEntry);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<KernelABISlot> slots;
  TargetProfileId targetProfile;
  bool transportPreparedBeforeEntry = false;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
  std::string moduleFormat;
  int64_t logicalRank = -1;
  int64_t defaultDDRArenaArgumentIndex = -1;
  int64_t transportStatusArgumentIndex = -1;
  int64_t profileRecordArgumentIndex = -1;
  ProfileCaptureKind profileCapture = ProfileCaptureKind::None;
};

struct TargetModuleReadback {
  std::string contentDigest;
  std::string moduleFormat;
};

llvm::Error fail(llvm::raw_ostream &diagnostics, llvm::StringRef message);
bool isRegularTargetFile(llvm::StringRef path);

mlir::FailureOr<PreparedTargetRank>
prepareTargetABI(const RankExecutable &rankExecutable,
                 const ExecutionConfig &executionConfig,
                 bool transportPreparedBeforeEntry,
                 ProfileCaptureKind profileCapture = ProfileCaptureKind::None);
mlir::LogicalResult lowerToTargetLLVM(PreparedTargetRank &prepared);
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

llvm::Error
writeLLVMIR(const llvm::Module &module, llvm::StringRef entrySymbol,
            llvm::ArrayRef<KernelABISlot> slots,
            const RuntimeLaunchContract &runtimeLaunchContract,
            int64_t logicalRank, int64_t rankCount, llvm::StringRef path,
            ProfileCaptureKind profileCapture = ProfileCaptureKind::None);
llvm::Error writeTargetLLVMIR(const llvm::Module &module, llvm::StringRef path);

/// Imports the complete rank domain into one context, scopes every supported
/// definition by rank, links it, and creates the contract's ordered exports.
llvm::Expected<OwnedTargetLLVMModule> buildKernelAggregateTargetModule(
    const TargetLLVMModuleBundle &targetLLVMModules);
llvm::Error
runDeviceLink(const TargetToolchain &toolchain, llvm::StringRef llvmIR,
              llvm::StringRef module, llvm::StringRef object,
              llvm::StringRef crtObject,
              const RuntimeLaunchContract &runtimeLaunchContract,
              ProfileCaptureKind profileCapture = ProfileCaptureKind::None);
llvm::Expected<TargetModuleReadback>
verifyTargetModule(llvm::StringRef path,
                   llvm::ArrayRef<VerifiedTargetExport> expectedExports,
                   TargetProfileId expectedProfile,
                   const RuntimeLaunchContract &expectedLaunch);

/// Verifies a genuinely linked module with the production ELF readback path
/// and exposes the immutable typed facts that production stores per module.
llvm::Expected<VerifiedTargetModule> verifyLinkedTargetModuleForTesting(
    llvm::StringRef path, llvm::StringRef entrySymbol,
    TargetProfileId targetProfile,
    const RuntimeLaunchContract &runtimeLaunchContract);

/// Re-runs the production target LLVM module readback against the immutable
/// typed facts stored by the owner-backed entry.
llvm::Error
verifyTargetLLVMModuleForTesting(const TargetLLVMModule &targetModule);

/// Runs the side-effect-free all-rank runtime-launch contract preflight used
/// before publication creates its private staging directory.
llvm::Error validateRuntimeLaunchContractDomainForTesting(
    const TargetLLVMModuleBundle &targetLLVMModules);

llvm::Expected<TargetLLVMModuleBundle>
compileExecutableBundleToTargetLLVMModulesImpl(
    const ExecutableBundle &executableBundle, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    ProfileCaptureKind profileCapture = ProfileCaptureKind::None);

llvm::Expected<TargetArtifactBundle>
compileTargetLLVMModuleBundleToTargetArtifactsImpl(
    const TargetLLVMModuleBundle &targetLLVMModules,
    llvm::StringRef outputDirectory, const TargetToolchain &toolchain,
    llvm::raw_ostream &diagnostics, ProfileCaptureKind profileCapture);

llvm::Expected<TargetArtifactBundle>
compileExecutableBundleToTargetArtifactsImpl(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank);

} // namespace detail
} // namespace wafer::compiler

#endif // WAFER_COMPILER_TARGETARTIFACTINTERNAL_H
