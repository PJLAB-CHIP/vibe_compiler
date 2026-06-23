//===- LayoutPlanningAnalysis.h - Group layout planning analysis -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_GROUP_LAYOUTPLANNINGANALYSIS_H
#define WAFER_ANALYSIS_GROUP_LAYOUTPLANNINGANALYSIS_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Analysis/Group/TilingDemandAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer {

enum class LayoutPlanValueRole {
  Input,
  Output,
  Result,
};

enum class LayoutPlanRelation {
  Exact,
  Broadcast,
};

struct LayoutPlanValue {
  LayoutPlanValueRole role;
  unsigned index = 0;
  mlir::Value value;
  mlir::Type type;
  TilingDemandSlice slice;
  MemLayout layout = MemLayout::Tensor;
  LayoutPlanRelation relation = LayoutPlanRelation::Exact;
};

struct LayoutMaterializationDemand {
  unsigned index = 0;
  std::string source;
  std::string target;
  mlir::Type type;
  MemLayout sourceLayout = MemLayout::Tensor;
  MemLayout targetLayout = MemLayout::Tensor;
};

struct OpLayoutPlan {
  OpTilingDemandKind kind = OpTilingDemandKind::Failure;
  mlir::Operation *op = nullptr;
  unsigned opIndex = 0;
  llvm::SmallVector<LayoutPlanValue, 4> values;
  llvm::SmallVector<TilingDemandAccumulator, 2> accumulators;
  llvm::SmallVector<LayoutMaterializationDemand, 2> materializations;
  WaferLinalgExtCollectiveInfo collectiveInfo;
  std::string failureReason;
};

struct GroupLayoutPlan {
  GroupOp group;
  llvm::SmallVector<LayoutPlanValue, 4> boundaryValues;
  llvm::SmallVector<OpLayoutPlan, 8> ops;
  llvm::SmallVector<LayoutMaterializationDemand, 2> resultMaterializations;
  bool succeeded = true;
  std::string failureReason;
};

mlir::LogicalResult collectGroupLayoutPlan(GroupOp group,
                                           GroupLayoutPlan &plan);

void dumpGroupLayoutPlan(const GroupLayoutPlan &plan,
                         llvm::StringRef groupLabel, llvm::raw_ostream &os);

} // namespace wafer

#endif // WAFER_ANALYSIS_GROUP_LAYOUTPLANNINGANALYSIS_H
