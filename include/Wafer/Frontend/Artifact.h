//===- Artifact.h - Wafer frontend artifact verifier ----------*- C++ -*-===//

#ifndef WAFER_FRONTEND_ARTIFACT_H
#define WAFER_FRONTEND_ARTIFACT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::frontend {

struct ArtifactVerificationResult {
  unsigned sidecarConstantCount = 0;
};

mlir::LogicalResult
verifyFrontendArtifact(mlir::ModuleOp module, llvm::StringRef sidecarPath,
                       llvm::raw_ostream &diagnostics,
                       ArtifactVerificationResult *result = nullptr);

} // namespace wafer::frontend

#endif // WAFER_FRONTEND_ARTIFACT_H
