//===- PackageInternal.h - Internal package assembly -----------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGEINTERNAL_H
#define WAFER_COMPILER_PACKAGEINTERNAL_H

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/CompilationResult.h"

#include <optional>

namespace wafer::compiler {

struct CompilationResultBuilder {
  static CompilationResult
  make(ExecutionConfig executionConfig, ExecutablePackage package,
       std::optional<ProfileInstrumentationProduct> profileInstrumentation) {
    return CompilationResult(executionConfig, std::move(package),
                             std::move(profileInstrumentation));
  }
};

} // namespace wafer::compiler

namespace wafer::compiler::detail {

/// Resource names are diagnostic payload only. These package-boundary
/// predicates intentionally compare typed identity and storage facts.
bool doesPackageSlotMatchProgramBinding(const TileEntryArgument &slot,
                                        const ProgramResourceBinding &binding);
bool isValidPackageCompilerManagedSlot(const TileEntryArgument &slot);

/// Stages the package into a private staging root, verifies it, reads back
/// the canonical manifest and returns the narrow staged readback. The
/// committed ExecutablePackage is constructed only by the outer transaction
/// commit stage.
llvm::Expected<runtime::VerifiedPackageManifest>
writePackage(llvm::StringRef tensorProgramDirectory,
             const DeviceExecutable &deviceExecutable,
             const LinkedTargetModules &targetModules,
             llvm::StringRef outputDirectory, llvm::raw_ostream &diagnostics,
             std::optional<int64_t> failAfterLaunchSlot);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PACKAGEINTERNAL_H
