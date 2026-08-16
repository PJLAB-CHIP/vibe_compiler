//===- PackageInternal.h - Internal package assembly -----------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGEINTERNAL_H
#define WAFER_COMPILER_PACKAGEINTERNAL_H

#include "Wafer/Compiler/Package.h"

#include <optional>

namespace wafer::compiler::detail {

/// Resource names are diagnostic payload only. These package-boundary
/// predicates intentionally compare typed identity and storage facts.
bool doesPackageSlotMatchProgramBinding(const TileEntryArgument &slot,
                                        const ProgramResourceBinding &binding);
bool isValidPackageCompilerManagedSlot(const TileEntryArgument &slot);

llvm::Expected<VerifiedPackage>
writePackage(llvm::StringRef tensorProgramDirectory,
             const CardExecutable &cardExecutable,
             const LinkedTargetModules &targetModules,
             llvm::StringRef outputDirectory, llvm::raw_ostream &diagnostics,
             std::optional<int64_t> failAfterLaunchSlot);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PACKAGEINTERNAL_H
