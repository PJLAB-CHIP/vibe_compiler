//===- OptimizationArtifactDigest.h - Exact artifact snapshots -*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONARTIFACTDIGEST_H
#define WAFER_SUPPORT_OPTIMIZATIONARTIFACTDIGEST_H

#include "Wafer/Support/OptimizationMechanism.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace wafer {

std::optional<OptimizationDigest>
digestOptimizationArtifactFileV1(llvm::StringRef path,
                                 llvm::StringRef domain);

/// Canonical sorted relative-path/content view. Symlinks and filesystem
/// errors fail closed rather than acquiring host-path-dependent semantics.
std::optional<OptimizationDigest>
digestOptimizationArtifactDirectoryV1(llvm::StringRef directory,
                                      llvm::StringRef domain);

OptimizationDigest digestOptimizationBytesV1(llvm::StringRef domain,
                                              llvm::ArrayRef<uint8_t> bytes);

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONARTIFACTDIGEST_H
