//===- Package.h - Typed Wafer compiler package bundle ---------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGE_H
#define WAFER_COMPILER_PACKAGE_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

class PackageBundle {
public:
  PackageBundle(PackageBundle &&) = default;
  PackageBundle &operator=(PackageBundle &&) = default;
  PackageBundle(const PackageBundle &) = delete;
  PackageBundle &operator=(const PackageBundle &) = delete;

  llvm::StringRef getRootDirectory() const { return rootDirectory; }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const runtime::VerifiedPackageManifest &getManifest() const {
    return manifest;
  }

private:
  friend struct PackageBundleBuilder;

  PackageBundle(llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
                runtime::VerifiedPackageManifest manifest)
      : rootDirectory(rootDirectory.str()), executionConfig(executionConfig),
        manifest(std::move(manifest)) {}

  std::string rootDirectory;
  ExecutionConfig executionConfig;
  runtime::VerifiedPackageManifest manifest;
};

/// Copies the verified source tensor-program checkpoint and target modules
/// into a private transaction, constructs and reads back the canonical typed
/// manifest, and publishes the complete package only after all members pass
/// verification.
llvm::Expected<PackageBundle>
assemblePackageBundle(llvm::StringRef tensorProgramDirectory,
                      const ExecutableBundle &executableBundle,
                      const TargetArtifactBundle &targetArtifacts,
                      llvm::StringRef outputDirectory,
                      llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PACKAGE_H
