//===- TargetArtifactInternal.h - Internal target build --------*- C++ -*-===//

#ifndef WAFER_COMPILER_TARGETARTIFACTINTERNAL_H
#define WAFER_COMPILER_TARGETARTIFACTINTERNAL_H

#include "Wafer/Compiler/TargetArtifact.h"

#include <optional>

namespace wafer::compiler::detail {

/// Runs production entry-only ABI preparation, target lowering, and lowered
/// entry verification on an owned clone. This narrow hook lets unit tests
/// prove accepted multi-function closure handling without invoking the
/// external object/link toolchain.
mlir::LogicalResult
lowerTargetABIForTesting(const RankExecutable &rankExecutable);

llvm::Expected<TargetArtifactBundle>
compileExecutableBundleToTargetArtifactsImpl(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_TARGETARTIFACTINTERNAL_H
