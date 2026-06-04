//===- TilingDemandAnalysis.h - Group tiling demand analysis ----*- C++ -*-===//

#ifndef WAFER_ANALYSIS_GROUP_TILINGDEMANDANALYSIS_H
#define WAFER_ANALYSIS_GROUP_TILINGDEMANDANALYSIS_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer {

enum class TilingDemandValueRole {
  Input,
  Output,
  Result,
};

struct TilingDemandSlice {
  llvm::SmallVector<unsigned, 4> loopDims;
};

struct TilingDemandValue {
  TilingDemandValueRole role;
  unsigned index = 0;
  mlir::Value value;
  mlir::Type type;
  TilingDemandSlice slice;
};

struct TilingDemandAccumulator {
  unsigned resultIndex = 0;
  llvm::SmallVector<unsigned, 4> reductionDims;
};

enum class OpTilingDemandKind {
  Support,
  Linalg,
  TensorCollective,
  Failure,
};

struct OpTilingDemand {
  OpTilingDemandKind kind = OpTilingDemandKind::Failure;
  mlir::Operation *op = nullptr;
  unsigned opIndex = 0;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes;
  llvm::SmallVector<TilingDemandValue, 4> values;
  llvm::SmallVector<TilingDemandAccumulator, 2> accumulators;
  WaferTensorCollectiveInfo collectiveInfo;
  std::string failureReason;
};

struct GroupTilingDemand {
  GroupOp group;
  llvm::SmallVector<TilingDemandValue, 4> boundaryValues;
  llvm::SmallVector<TilingDemandValue, 2> resultTiles;
  llvm::SmallVector<OpTilingDemand, 8> ops;
  bool succeeded = true;
  std::string failureReason;
};

mlir::LogicalResult collectGroupTilingDemand(GroupOp group,
                                             GroupTilingDemand &demand);

void dumpGroupTilingDemand(const GroupTilingDemand &demand,
                           llvm::StringRef groupLabel, llvm::raw_ostream &os);

} // namespace wafer

#endif // WAFER_ANALYSIS_GROUP_TILINGDEMANDANALYSIS_H
