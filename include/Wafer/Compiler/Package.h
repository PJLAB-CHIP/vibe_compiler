//===- Package.h - Verified Wafer compiler package ------------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGE_H
#define WAFER_COMPILER_PACKAGE_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetCodeGen.h"
#include "Wafer/Package/PackageManifest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

class VerifiedPackage {
public:
  VerifiedPackage(VerifiedPackage &&) = default;
  VerifiedPackage &operator=(VerifiedPackage &&) = default;
  VerifiedPackage(const VerifiedPackage &) = delete;
  VerifiedPackage &operator=(const VerifiedPackage &) = delete;

  llvm::StringRef getRootDirectory() const { return rootDirectory; }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const runtime::VerifiedPackageManifest &getManifest() const {
    return manifest;
  }

private:
  friend struct VerifiedPackageBuilder;

  VerifiedPackage(llvm::StringRef rootDirectory,
                  ExecutionConfig executionConfig,
                  runtime::VerifiedPackageManifest manifest)
      : rootDirectory(rootDirectory.str()), executionConfig(executionConfig),
        manifest(std::move(manifest)) {}

  std::string rootDirectory;
  ExecutionConfig executionConfig;
  runtime::VerifiedPackageManifest manifest;
};

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PACKAGE_H
