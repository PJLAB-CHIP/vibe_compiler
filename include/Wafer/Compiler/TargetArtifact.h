//===- TargetArtifact.h - Typed Wafer target artifacts ---------*- C++ -*-===//

#ifndef WAFER_COMPILER_TARGETARTIFACT_H
#define WAFER_COMPILER_TARGETARTIFACT_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

enum class KernelABISlotRole {
  UserInput,
  Parameter,
  Constant,
  Output,
  Workspace,
  TransportStatus,
};

struct KernelABISlot {
  int64_t ordinal = -1;
  KernelABISlotRole role = KernelABISlotRole::UserInput;
  int64_t resourceIndex = -1;
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t byteSize = -1;
  int64_t alignment = -1;
};

/// One fully translated target LLVM module and the context that owns all of
/// its uniqued IR state. The module is immutable after construction: target
/// publication and host-side consumers must share this verified translation
/// instead of independently lowering the accepted rank again.
class TargetLLVMModule {
public:
  ~TargetLLVMModule();
  TargetLLVMModule(TargetLLVMModule &&);
  TargetLLVMModule &operator=(TargetLLVMModule &&);
  TargetLLVMModule(const TargetLLVMModule &) = delete;
  TargetLLVMModule &operator=(const TargetLLVMModule &) = delete;

  int64_t getLogicalRank() const { return logicalRank; }
  llvm::StringRef getEntrySymbol() const { return entrySymbol; }
  TargetProfileId getTargetProfileId() const { return targetProfile; }
  TargetIdentityId getTargetIdentityId() const { return targetIdentity; }
  KernelRuntimeABIId getKernelRuntimeABIId() const { return kernelRuntimeABI; }
  llvm::StringRef getModuleFormat() const { return moduleFormat; }
  llvm::StringRef getModuleIdentifier() const;
  llvm::StringRef getTargetTriple() const;
  const std::vector<KernelABISlot> &getKernelABISlots() const {
    return kernelABISlots;
  }
  const llvm::Module &getModule() const;

private:
  friend struct TargetLLVMModuleBundleBuilder;

  TargetLLVMModule(int64_t logicalRank, llvm::StringRef entrySymbol,
                   TargetProfileId targetProfile,
                   TargetIdentityId targetIdentity,
                   KernelRuntimeABIId kernelRuntimeABI,
                   llvm::StringRef moduleFormat,
                   std::vector<KernelABISlot> kernelABISlots,
                   std::unique_ptr<llvm::LLVMContext> context,
                   std::unique_ptr<llvm::Module> module);

  int64_t logicalRank;
  std::string entrySymbol;
  TargetProfileId targetProfile;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
  std::string moduleFormat;
  std::vector<KernelABISlot> kernelABISlots;
  // Declaration order is intentional: reverse destruction destroys the
  // module before the context that owns its uniqued state.
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
};

/// Atomic owner of the complete target LLVM rank domain. This is an
/// invocation-local boundary and deliberately has no serialization form.
class TargetLLVMModuleBundle {
public:
  ~TargetLLVMModuleBundle();
  TargetLLVMModuleBundle(TargetLLVMModuleBundle &&);
  TargetLLVMModuleBundle &operator=(TargetLLVMModuleBundle &&);
  TargetLLVMModuleBundle(const TargetLLVMModuleBundle &) = delete;
  TargetLLVMModuleBundle &operator=(const TargetLLVMModuleBundle &) = delete;

  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const std::vector<TargetLLVMModule> &getModules() const { return modules; }

private:
  friend struct TargetLLVMModuleBundleBuilder;

  TargetLLVMModuleBundle(ExecutionConfig executionConfig,
                         std::vector<TargetLLVMModule> modules)
      : executionConfig(executionConfig), modules(std::move(modules)) {}

  ExecutionConfig executionConfig;
  std::vector<TargetLLVMModule> modules;
};

class TargetToolchain {
public:
  static llvm::Expected<TargetToolchain>
  create(llvm::StringRef pythonExecutable, llvm::StringRef deviceLinkerScript);

  llvm::StringRef getPythonExecutable() const { return pythonExecutable; }
  llvm::StringRef getDeviceLinkerScript() const { return deviceLinkerScript; }

private:
  TargetToolchain(llvm::StringRef pythonExecutable,
                  llvm::StringRef deviceLinkerScript)
      : pythonExecutable(pythonExecutable.str()),
        deviceLinkerScript(deviceLinkerScript.str()) {}

  std::string pythonExecutable;
  std::string deviceLinkerScript;
};

/// Prepares the fixed Kernel Runtime ABI, lowers, translates, and verifies
/// every accepted rank before atomically returning an owner-backed LLVM
/// module bundle. No file or package artifact is produced by this boundary.
llvm::Expected<TargetLLVMModuleBundle>
compileExecutableBundleToTargetLLVMModules(
    const ExecutableBundle &executableBundle, llvm::raw_ostream &diagnostics);

class VerifiedTargetModule {
public:
  int64_t getLogicalRank() const { return logicalRank; }
  llvm::StringRef getEntrySymbol() const { return entrySymbol; }
  llvm::StringRef getRelativePath() const { return relativePath; }
  llvm::StringRef getContentDigest() const { return contentDigest; }
  TargetProfileId getTargetProfileId() const { return targetProfile; }
  TargetIdentityId getTargetIdentityId() const { return targetIdentity; }
  KernelRuntimeABIId getKernelRuntimeABIId() const { return kernelRuntimeABI; }
  llvm::StringRef getModuleFormat() const { return moduleFormat; }
  const std::vector<KernelABISlot> &getKernelABISlots() const {
    return kernelABISlots;
  }

private:
  friend struct TargetArtifactBundleBuilder;

  VerifiedTargetModule(int64_t logicalRank, llvm::StringRef entrySymbol,
                       llvm::StringRef relativePath,
                       llvm::StringRef contentDigest,
                       TargetProfileId targetProfile,
                       TargetIdentityId targetIdentity,
                       KernelRuntimeABIId kernelRuntimeABI,
                       llvm::StringRef moduleFormat,
                       std::vector<KernelABISlot> kernelABISlots)
      : logicalRank(logicalRank), entrySymbol(entrySymbol.str()),
        relativePath(relativePath.str()), contentDigest(contentDigest.str()),
        targetProfile(targetProfile), targetIdentity(targetIdentity),
        kernelRuntimeABI(kernelRuntimeABI), moduleFormat(moduleFormat.str()),
        kernelABISlots(std::move(kernelABISlots)) {}

  int64_t logicalRank;
  std::string entrySymbol;
  std::string relativePath;
  std::string contentDigest;
  TargetProfileId targetProfile;
  TargetIdentityId targetIdentity;
  KernelRuntimeABIId kernelRuntimeABI;
  std::string moduleFormat;
  std::vector<KernelABISlot> kernelABISlots;
};

class TargetArtifactBundle {
public:
  TargetArtifactBundle(TargetArtifactBundle &&) = default;
  TargetArtifactBundle &operator=(TargetArtifactBundle &&) = default;
  TargetArtifactBundle(const TargetArtifactBundle &) = delete;
  TargetArtifactBundle &operator=(const TargetArtifactBundle &) = delete;

  llvm::StringRef getRootDirectory() const { return rootDirectory; }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const std::vector<VerifiedTargetModule> &getModules() const {
    return modules;
  }

private:
  friend struct TargetArtifactBundleBuilder;

  TargetArtifactBundle(llvm::StringRef rootDirectory,
                       ExecutionConfig executionConfig,
                       std::vector<VerifiedTargetModule> modules)
      : rootDirectory(rootDirectory.str()), executionConfig(executionConfig),
        modules(std::move(modules)) {}

  std::string rootDirectory;
  ExecutionConfig executionConfig;
  std::vector<VerifiedTargetModule> modules;
};

/// Links and publishes the exact verified LLVM modules owned by the bundle.
/// This consumer never re-runs ABI preparation, target lowering, or LLVM
/// translation.
llvm::Expected<TargetArtifactBundle>
compileTargetLLVMModuleBundleToTargetArtifacts(
    const TargetLLVMModuleBundle &targetLLVMModules,
    llvm::StringRef outputDirectory, const TargetToolchain &toolchain,
    llvm::raw_ostream &diagnostics);

/// Compiles and links every executable rank into a private transaction
/// directory, verifies the complete rank/module/entry/ABI/digest domain, and
/// publishes the target-artifact root only after all ranks pass.
llvm::Expected<TargetArtifactBundle> compileExecutableBundleToTargetArtifacts(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_TARGETARTIFACT_H
