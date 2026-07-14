//===- TargetArtifact.h - Typed Wafer target artifacts ---------*- C++ -*-===//

#ifndef WAFER_COMPILER_TARGETARTIFACT_H
#define WAFER_COMPILER_TARGETARTIFACT_H

#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
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

/// Compiles and links every executable rank into a private transaction
/// directory, verifies the complete rank/module/entry/ABI/digest domain, and
/// publishes the target-artifact root only after all ranks pass.
llvm::Expected<TargetArtifactBundle> compileExecutableBundleToTargetArtifacts(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_TARGETARTIFACT_H
