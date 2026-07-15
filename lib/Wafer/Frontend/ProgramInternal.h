//===- ProgramInternal.h - Frontend program internals --------*- C++ -*-===//

#ifndef WAFER_LIB_FRONTEND_PROGRAMINTERNAL_H
#define WAFER_LIB_FRONTEND_PROGRAMINTERNAL_H

#include "Wafer/Frontend/Program.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::frontend::program_detail {

struct ProgramSignature {
  std::vector<int64_t> shape;
  std::string dtype;
};

struct ProgramInputLocation {
  std::string type;
  int64_t position = -1;
  std::string name;
};

struct DistributedBoundaryRank {
  int64_t rank = -1;
  int64_t replicaId = -1;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct DistributedBoundaryBinding {
  int64_t index = -1;
  std::string distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::string dtype;
  std::vector<DistributedBoundaryRank> ranks;
};

struct DistributedBoundary {
  int64_t version = 0;
  int64_t logicalRankCount = 0;
  std::vector<DistributedBoundaryBinding> inputs;
  std::vector<DistributedBoundaryBinding> outputs;
};

struct ProgramMetadata {
  std::string name;
  std::vector<ProgramSignature> inputSignatures;
  std::vector<ProgramSignature> outputSignatures;
  std::vector<ProgramInputLocation> inputLocations;
  std::optional<DistributedBoundary> distributedBoundary;
};

bool rejectProgramDirectory(llvm::StringRef reason,
                            llvm::raw_ostream &diagnostics);

bool readStringField(const llvm::json::Object &object, llvm::StringRef field,
                     std::string &out, llvm::raw_ostream &diagnostics);
bool readIntegerField(const llvm::json::Object &object, llvm::StringRef field,
                      int64_t &out, llvm::raw_ostream &diagnostics);
bool readIntegerArrayField(const llvm::json::Object &object,
                           llvm::StringRef field, std::vector<int64_t> &values,
                           llvm::raw_ostream &diagnostics,
                           bool requirePositive = false);

mlir::FailureOr<ProgramMetadata>
parseProgramMetadata(llvm::StringRef metaPath, llvm::raw_ostream &diagnostics);

std::string dtypeString(mlir::Type elementType);
std::string normalizeProgramDtype(llvm::StringRef dtype);
bool checkedMulUint64(uint64_t lhs, uint64_t rhs, uint64_t &result);
std::optional<uint64_t> checkedRawByteSize(llvm::ArrayRef<int64_t> shape,
                                           mlir::Type elementType);

bool verifyNpyTensorPayloadFile(llvm::StringRef path,
                                llvm::StringRef displayName,
                                llvm::ArrayRef<int64_t> expectedShape,
                                mlir::Type elementType,
                                llvm::raw_ostream &diagnostics);

mlir::FailureOr<mlir::func::FuncOp>
findSingleFunction(mlir::ModuleOp module, llvm::raw_ostream &diagnostics);
std::string programPath(llvm::StringRef programDir,
                        llvm::ArrayRef<llvm::StringRef> components);
bool fileExists(llvm::StringRef path);

bool verifyProgramMetadata(mlir::ModuleOp module, llvm::StringRef programDir,
                           const ProgramMetadata &meta,
                           llvm::raw_ostream &diagnostics,
                           FrontendProgramVerificationResult *result);
bool hasSpmdParameterShardings(mlir::ModuleOp module);
ProgramDistributionKind
getVerifiedDistributionKind(llvm::StringRef distribution);
mlir::FailureOr<int64_t>
getSingleExecutionMeshRankCount(mlir::ModuleOp module,
                                llvm::raw_ostream &diagnostics);
bool verifyDistributedBoundary(mlir::ModuleOp module,
                               const ProgramMetadata &meta,
                               mlir::func::FuncOp func, bool postSpmdMarker,
                               llvm::raw_ostream &diagnostics,
                               FrontendProgramVerificationResult *result);
bool verifyParameterShards(mlir::ModuleOp module, llvm::StringRef programDir,
                           const ProgramMetadata &meta, mlir::func::FuncOp func,
                           llvm::raw_ostream &diagnostics,
                           FrontendProgramVerificationResult *result);

} // namespace wafer::frontend::program_detail

#endif // WAFER_LIB_FRONTEND_PROGRAMINTERNAL_H
