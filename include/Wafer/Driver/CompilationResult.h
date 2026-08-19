//===- CompilationResult.h - Typed compilation product ----------*- C++ -*-===//

#ifndef WAFER_DRIVER_COMPILATIONRESULT_H
#define WAFER_DRIVER_COMPILATIONRESULT_H

#include "Wafer/Driver/Compilation.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Package/Manifest/PackageManifest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

/// Compiler spelling of the single package-layer owner also returned by the
/// runtime strict loader. This is an alias, never a second verified wrapper.
using ExecutablePackage = runtime::ExecutablePackage;

/// Compiler-side owner of profile instrumentation committed together with the
/// ordinary package. It owns the exact activation/plan/site-map buffers and
/// both strictly bound capture packages selected before the publication
/// rename. The runtime loader remains the semantic parser at launch.
class ProfileInstrumentationProduct {
public:
  ProfileInstrumentationProduct(ProfileInstrumentationProduct &&) = default;
  ProfileInstrumentationProduct &operator=(ProfileInstrumentationProduct &&) =
      default;
  ProfileInstrumentationProduct(const ProfileInstrumentationProduct &) =
      delete;
  ProfileInstrumentationProduct &
  operator=(const ProfileInstrumentationProduct &) = delete;

  llvm::StringRef getRootDirectory() const { return rootDirectory; }
  llvm::StringRef getPrimaryManifestDigest() const {
    return primaryManifestDigest;
  }
  llvm::StringRef getPlanDigest() const { return planDigest; }
  llvm::StringRef getSiteMapDigest() const { return siteMapDigest; }

private:
  friend struct ProfileInstrumentationProductBuilder;

  ProfileInstrumentationProduct(
      std::string rootDirectory, std::string primaryManifestDigest,
      std::string planDigest, std::string siteMapDigest,
      std::unique_ptr<llvm::MemoryBuffer> activationBuffer,
      std::unique_ptr<llvm::MemoryBuffer> planBuffer,
      std::unique_ptr<llvm::MemoryBuffer> siteMapBuffer,
      std::vector<runtime::detail::BoundExecutablePackage> capturePackages)
      : rootDirectory(std::move(rootDirectory)),
        primaryManifestDigest(std::move(primaryManifestDigest)),
        planDigest(std::move(planDigest)),
        siteMapDigest(std::move(siteMapDigest)),
        activationBuffer(std::move(activationBuffer)),
        planBuffer(std::move(planBuffer)),
        siteMapBuffer(std::move(siteMapBuffer)),
        capturePackages(std::move(capturePackages)) {}

  std::string rootDirectory;
  std::string primaryManifestDigest;
  std::string planDigest;
  std::string siteMapDigest;
  std::unique_ptr<llvm::MemoryBuffer> activationBuffer;
  std::unique_ptr<llvm::MemoryBuffer> planBuffer;
  std::unique_ptr<llvm::MemoryBuffer> siteMapBuffer;
  std::vector<runtime::detail::BoundExecutablePackage> capturePackages;
};

/// The complete typed product of one production compilation transaction. The
/// ordinary package is always present; the profile instrumentation is present
/// exactly when it was explicitly requested and was committed together with
/// the ordinary package. CardExecutable, target LLVM modules and IR trace are
/// deliberately not part of this result; internal qualification consumers use
/// compileProgramWithTargetLLVMModules.
class CompilationResult {
public:
  CompilationResult(CompilationResult &&) = default;
  CompilationResult &operator=(CompilationResult &&) = default;
  CompilationResult(const CompilationResult &) = delete;
  CompilationResult &operator=(const CompilationResult &) = delete;

  const ExecutablePackage &getPackage() const { return package; }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const std::optional<ProfileInstrumentationProduct>
      &getProfileInstrumentation() const {
    return profileInstrumentation;
  }

private:
  friend struct CompilationResultBuilder;
  friend llvm::Expected<CompilationResult> compileProgram(
      CompilationRequest request, llvm::StringRef outputDirectory,
      llvm::StringRef xlaSpmdPartitionerHelper,
      const TargetToolchain &targetToolchain, CompilationOptions options,
      llvm::raw_ostream &diagnostics);

  CompilationResult(ExecutionConfig executionConfig, ExecutablePackage package,
                    std::optional<ProfileInstrumentationProduct>
                        profileInstrumentation)
      : executionConfig(executionConfig), package(std::move(package)),
        profileInstrumentation(std::move(profileInstrumentation)) {}

  ExecutionConfig executionConfig;
  ExecutablePackage package;
  std::optional<ProfileInstrumentationProduct> profileInstrumentation;
};

/// Compiles the source program and atomically publishes the requested output
/// directory. Success means the target package root is visible and every
/// fallible verification step closed before the single publication rename;
/// failure means no target output from this invocation is visible. The
/// ordinary case publishes the package as the output directory itself. When
/// profiling is requested, the output directory is the common delivery root
/// published by one rename: it contains exactly `package/` (the ordinary
/// package root) and `package.profile/` (the instrumentation root), so the
/// runtime sibling rule `<package-root>.profile` is preserved and both
/// products become visible atomically. Detailed diagnostics are rendered on
/// the caller-owned stream; the returned error classifies the failing
/// transaction stage.
llvm::Expected<CompilationResult>
compileProgram(CompilationRequest request,
               llvm::StringRef outputDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_DRIVER_COMPILATIONRESULT_H
