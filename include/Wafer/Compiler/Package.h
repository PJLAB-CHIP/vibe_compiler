//===- Package.h - Typed Wafer compiler package result ----------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGE_H
#define WAFER_COMPILER_PACKAGE_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetCodeGen.h"
#include "Wafer/Package/PackageManifest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

/// Move-only owner of one committed, readback-verified package: the canonical
/// installed root, the execution configuration, and the verified manifest
/// loaded back from the installed root. The manifest readback closes the
/// whole package tree, so module and program-data members are bound by the
/// same identity. It is the primary product of the compiler library entry.
class ExecutablePackage {
public:
  ExecutablePackage(ExecutablePackage &&) = default;
  ExecutablePackage &operator=(ExecutablePackage &&) = default;
  ExecutablePackage(const ExecutablePackage &) = delete;
  ExecutablePackage &operator=(const ExecutablePackage &) = delete;

  llvm::StringRef getRootDirectory() const { return rootDirectory; }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const runtime::VerifiedPackageManifest &getManifest() const {
    return manifest;
  }

private:
  friend struct ExecutablePackageBuilder;

  ExecutablePackage(llvm::StringRef rootDirectory,
                    ExecutionConfig executionConfig,
                    runtime::VerifiedPackageManifest manifest)
      : rootDirectory(rootDirectory.str()), executionConfig(executionConfig),
        manifest(std::move(manifest)) {}

  std::string rootDirectory;
  ExecutionConfig executionConfig;
  runtime::VerifiedPackageManifest manifest;
};

/// Compiler-side identity of one profile instrumentation committed together
/// with the ordinary package. The digests are the values the instrumentation
/// writer computed from the staged files and re-verified against the
/// installed activation.json after the commit rename. The runtime strict
/// loader remains the single semantic reader at launch; this type only binds
/// the co-commit identity and never duplicates plan/site-map parsing.
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

  ProfileInstrumentationProduct(llvm::StringRef rootDirectory,
                                llvm::StringRef primaryManifestDigest,
                                llvm::StringRef planDigest,
                                llvm::StringRef siteMapDigest)
      : rootDirectory(rootDirectory.str()),
        primaryManifestDigest(primaryManifestDigest.str()),
        planDigest(planDigest.str()), siteMapDigest(siteMapDigest.str()) {}

  std::string rootDirectory;
  std::string primaryManifestDigest;
  std::string planDigest;
  std::string siteMapDigest;
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
  const std::optional<ProfileInstrumentationProduct>
      &getProfileInstrumentation() const {
    return profileInstrumentation;
  }

private:
  friend llvm::Expected<CompilationResult> compileProgram(
      CompilationRequest request, llvm::StringRef outputPackageDirectory,
      llvm::StringRef xlaSpmdPartitionerHelper,
      const TargetToolchain &targetToolchain, CompilationOptions options,
      llvm::raw_ostream &diagnostics);

  CompilationResult(ExecutablePackage package,
                    std::optional<ProfileInstrumentationProduct>
                        profileInstrumentation)
      : package(std::move(package)),
        profileInstrumentation(std::move(profileInstrumentation)) {}

  ExecutablePackage package;
  std::optional<ProfileInstrumentationProduct> profileInstrumentation;
};

/// Compiles the source program, atomically commits the requested package
/// directory and returns the readback-verified committed result. Success means
/// the target package root is visible and its manifest identity was read back
/// from the installed root; failure means no target package from this
/// invocation is visible. When profiling is requested, the ordinary package
/// remains the production package and a verified sibling `<output>.profile`
/// instrumentation is committed together with it; both are read back before
/// success is reported. Detailed diagnostics are rendered on the caller-owned
/// stream; the returned error classifies the failing transaction stage.
llvm::Expected<CompilationResult>
compileProgram(CompilationRequest request,
               llvm::StringRef outputPackageDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PACKAGE_H
