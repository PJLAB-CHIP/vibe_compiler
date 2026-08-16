//===- PackageInternal.h - Internal package assembly -----------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGEINTERNAL_H
#define WAFER_COMPILER_PACKAGEINTERNAL_H

#include "Wafer/Compiler/Package.h"

#include "llvm/Support/MemoryBuffer.h"

#include <optional>

namespace wafer::compiler {

/// Opened all-and-only package members pinned before the publication rename:
/// the canonical manifest, every verified module in manifest order, and the
/// target-ready program data member.
struct OpenedPackageMembers {
  std::unique_ptr<llvm::MemoryBuffer> manifest;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> modules;
  std::unique_ptr<llvm::MemoryBuffer> programData;
};

/// Internal constructor for the committed executable package. Only the
/// commit stage of the compilation transaction may construct it.
struct ExecutablePackageBuilder {
  static ExecutablePackage
  makePackage(llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
              runtime::VerifiedPackageManifest manifest,
              OpenedPackageMembers members) {
    return ExecutablePackage(rootDirectory, executionConfig,
                             std::move(manifest), std::move(members.manifest),
                             std::move(members.modules),
                             std::move(members.programData));
  }
};

} // namespace wafer::compiler

namespace wafer::compiler::detail {

/// Resource names are diagnostic payload only. These package-boundary
/// predicates intentionally compare typed identity and storage facts.
bool doesPackageSlotMatchProgramBinding(const TileEntryArgument &slot,
                                        const ProgramResourceBinding &binding);
bool isValidPackageCompilerManagedSlot(const TileEntryArgument &slot);

/// Opens and binds the all-and-only package members listed by the verified
/// manifest: the canonical manifest, every module (content digest compared
/// against the verified record) and the program data member (byte size
/// compared against the verified record). All handles are opened before the
/// publication rename, so later path deletion or replacement cannot change
/// the returned content.
llvm::Expected<OpenedPackageMembers>
openPackageMembers(llvm::StringRef packageRoot,
                   const runtime::VerifiedPackageManifest &manifest);

/// Stages the package into a private staging root, verifies it, reads back
/// the canonical manifest and returns the narrow staged readback. The
/// committed ExecutablePackage is constructed only by the outer transaction
/// commit stage.
llvm::Expected<runtime::VerifiedPackageManifest>
writePackage(llvm::StringRef tensorProgramDirectory,
             const CardExecutable &cardExecutable,
             const LinkedTargetModules &targetModules,
             llvm::StringRef outputDirectory, llvm::raw_ostream &diagnostics,
             std::optional<int64_t> failAfterLaunchSlot);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PACKAGEINTERNAL_H
