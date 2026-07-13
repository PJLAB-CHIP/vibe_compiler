//===- PackageInternal.h - Internal package assembly -----------*- C++ -*-===//

#ifndef WAFER_COMPILER_PACKAGEINTERNAL_H
#define WAFER_COMPILER_PACKAGEINTERNAL_H

#include "Wafer/Compiler/Package.h"

#include <optional>

namespace wafer::compiler::detail {

llvm::Expected<PackageBundle>
assemblePackageBundleImpl(llvm::StringRef groupedProgramDirectory,
                          const ExecutableBundle &executableBundle,
                          const TargetArtifactBundle &targetArtifacts,
                          llvm::StringRef outputDirectory,
                          llvm::raw_ostream &diagnostics,
                          std::optional<int64_t> failAfterLogicalRank);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PACKAGEINTERNAL_H
