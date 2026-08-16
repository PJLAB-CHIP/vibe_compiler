//===- Package.h - Typed Wafer compiler package result ----------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGE_H
#define WAFER_COMPILER_PACKAGE_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetCodeGen.h"
#include "Wafer/Package/PackageManifest.h"

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

/// Move-only owner of one committed, readback-verified package: the canonical
/// root, the execution configuration, the verified manifest and the opened
/// all-and-only package members (manifest, modules and program data). The
/// members are pinned by open file descriptors opened before the single
/// publication rename, so later deletion or replacement of the paths cannot
/// invalidate the result content; reopening by root is not ownership.
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
  /// Opened canonical manifest member, the same bytes the manifest verifier
  /// consumed.
  const llvm::MemoryBuffer &getManifestBuffer() const {
    return *manifestBuffer;
  }
  /// Opened module members in the verified manifest module order.
  const std::vector<std::unique_ptr<llvm::MemoryBuffer>> &
  getModuleBuffers() const {
    return moduleBuffers;
  }
  /// Opened target-ready program data member (canonical empty file when the
  /// package carries no program data).
  const llvm::MemoryBuffer &getProgramDataBuffer() const {
    return *programDataBuffer;
  }

private:
  friend struct ExecutablePackageBuilder;

  ExecutablePackage(
      llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
      runtime::VerifiedPackageManifest manifest,
      std::unique_ptr<llvm::MemoryBuffer> manifestBuffer,
      std::vector<std::unique_ptr<llvm::MemoryBuffer>> moduleBuffers,
      std::unique_ptr<llvm::MemoryBuffer> programDataBuffer)
      : rootDirectory(rootDirectory.str()), executionConfig(executionConfig),
        manifest(std::move(manifest)),
        manifestBuffer(std::move(manifestBuffer)),
        moduleBuffers(std::move(moduleBuffers)),
        programDataBuffer(std::move(programDataBuffer)) {}

  std::string rootDirectory;
  ExecutionConfig executionConfig;
  runtime::VerifiedPackageManifest manifest;
  // Declaration order keeps the manifest alive as long as any member; the
  // buffers are independent open handles and are destroyed in reverse order.
  std::unique_ptr<llvm::MemoryBuffer> manifestBuffer;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> moduleBuffers;
  std::unique_ptr<llvm::MemoryBuffer> programDataBuffer;
};

/// Compiler-side identity of one profile instrumentation committed together
/// with the ordinary package. The digests are the values the instrumentation
/// writer computed from the staged files and re-verified against the staged
/// activation.json before the single publication rename; the published
/// inodes are the verified ones. The runtime strict loader remains the
/// single semantic reader at launch; this type only binds the co-commit
/// identity and never duplicates plan/site-map parsing.
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
               llvm::StringRef outputPackageDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_PACKAGE_H
