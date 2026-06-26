//===- Program.h - Wafer frontend program verifier -----------*- C++ -*-===//

#ifndef WAFER_FRONTEND_PROGRAM_H
#define WAFER_FRONTEND_PROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::frontend {

struct FrontendProgramVerificationResult {
  unsigned programParameterCount = 0;
  unsigned programUserInputCount = 0;
  unsigned programConstantCount = 0;
  unsigned programParameterShardCount = 0;
};

mlir::LogicalResult
verifyFrontendProgram(mlir::ModuleOp module, llvm::raw_ostream &diagnostics,
                      FrontendProgramVerificationResult *result = nullptr);

mlir::LogicalResult
verifyStableHLOProgramDir(mlir::ModuleOp module, llvm::StringRef programDir,
                          llvm::raw_ostream &diagnostics,
                          FrontendProgramVerificationResult *result = nullptr);

mlir::LogicalResult verifyAndMaterializeStableHLOProgramDir(
    mlir::ModuleOp module, llvm::StringRef programDir,
    llvm::raw_ostream &diagnostics,
    FrontendProgramVerificationResult *result = nullptr);

} // namespace wafer::frontend

#endif // WAFER_FRONTEND_PROGRAM_H
