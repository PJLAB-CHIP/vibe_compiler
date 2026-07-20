//===- TilingDemandAnalysis.h - Scheduling tiling demand analysis -*- C++
//-*-===//

#ifndef WAFER_ANALYSIS_SCHEDULING_TILINGDEMANDANALYSIS_H
#define WAFER_ANALYSIS_SCHEDULING_TILINGDEMANDANALYSIS_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  LinalgExtCollective,
  Failure,
};

struct OpTilingDemand {
  OpTilingDemandKind kind = OpTilingDemandKind::Failure;
  mlir::Operation *op = nullptr;
  unsigned opIndex = 0;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes;
  llvm::SmallVector<TilingDemandValue, 4> values;
  llvm::SmallVector<TilingDemandAccumulator, 2> accumulators;
  std::string failureReason;
};

/// Tiling demand derived from a standalone structured scheduling function.
/// The entry block arguments are partitioned into read-only inputs followed by
/// output destinations. The func.return operands are the yielded roots.
struct StructuredSchedulingTilingDemand {
  mlir::func::FuncOp function;
  unsigned inputCount = 0;
  llvm::SmallVector<TilingDemandValue, 4> boundaryValues;
  llvm::SmallVector<TilingDemandValue, 2> resultTiles;
  llvm::SmallVector<OpTilingDemand, 8> ops;
  bool succeeded = true;
  std::string failureReason;
};

mlir::LogicalResult collectStructuredSchedulingTilingDemand(
    mlir::func::FuncOp function, unsigned inputCount,
    StructuredSchedulingTilingDemand &demand);

} // namespace wafer

#endif // WAFER_ANALYSIS_SCHEDULING_TILINGDEMANDANALYSIS_H
