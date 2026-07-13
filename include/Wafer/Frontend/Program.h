//===- Program.h - Wafer frontend program verifier -----------*- C++ -*-===//

#ifndef WAFER_FRONTEND_PROGRAM_H
#define WAFER_FRONTEND_PROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::frontend {

enum class ProgramDistributionKind { Replicated, Partitioned };

struct ProgramRankSlice {
  int64_t logicalRank = -1;
  int64_t replicaId = -1;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::string payloadPath;
};

struct ProgramBoundaryBinding {
  int64_t index = -1;
  ProgramDistributionKind distribution = ProgramDistributionKind::Replicated;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::string dtype;
  std::vector<ProgramRankSlice> rankSlices;
};

struct ProgramParameterBinding {
  int64_t argumentIndex = -1;
  std::string name;
  ProgramDistributionKind distribution = ProgramDistributionKind::Replicated;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::string dtype;
  std::vector<ProgramRankSlice> rankSlices;
};

struct ProgramConstantBinding {
  int64_t argumentIndex = -1;
  int64_t position = -1;
  std::vector<int64_t> shape;
  std::string dtype;
  std::string payloadPath;
};

struct FrontendProgramVerificationResult {
  unsigned programParameterCount = 0;
  unsigned programUserInputCount = 0;
  unsigned programConstantCount = 0;
  unsigned programParameterShardCount = 0;
  int64_t logicalRankCount = 0;
  std::vector<ProgramBoundaryBinding> distributedInputs;
  std::vector<ProgramBoundaryBinding> distributedOutputs;
  std::vector<ProgramParameterBinding> parameters;
  std::vector<ProgramConstantBinding> constants;
};

mlir::LogicalResult
verifyFrontendProgram(mlir::ModuleOp module, llvm::raw_ostream &diagnostics,
                      FrontendProgramVerificationResult *result = nullptr);

mlir::LogicalResult
verifyProgramDirectory(mlir::ModuleOp module, llvm::StringRef programDir,
                       llvm::raw_ostream &diagnostics,
                       FrontendProgramVerificationResult *result = nullptr);

} // namespace wafer::frontend

#endif // WAFER_FRONTEND_PROGRAM_H
