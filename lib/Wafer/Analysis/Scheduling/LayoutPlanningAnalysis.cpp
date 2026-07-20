//===- LayoutPlanningAnalysis.cpp - Structured scheduling layout analysis ===//

#include "Wafer/Analysis/Scheduling/LayoutPlanningAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace {

static bool sameSlice(const TilingDemandSlice &lhs,
                      const TilingDemandSlice &rhs) {
  return lhs.loopDims == rhs.loopDims;
}

static bool isTensorLike(mlir::Type type) {
  return mlir::isa<mlir::RankedTensorType>(type);
}

static bool isContractionLike(const OpTilingDemand &demand) {
  return !demand.accumulators.empty() ||
         mlir::isa_and_nonnull<
             mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
             mlir::linalg::MatmulTransposeBOp, mlir::linalg::BatchMatmulOp,
             mlir::linalg::BatchMatmulTransposeAOp,
             mlir::linalg::BatchMatmulTransposeBOp>(demand.op);
}

static LayoutPlanRelation relationToResult(const TilingDemandValue &value,
                                           const TilingDemandSlice &result) {
  if (sameSlice(value.slice, result))
    return LayoutPlanRelation::Exact;
  return LayoutPlanRelation::Broadcast;
}

class LayoutPlanner {
public:
  mlir::LogicalResult run(mlir::func::FuncOp function,
                          unsigned schedulingInputCount,
                          StructuredSchedulingLayoutPlan &plan) {
    plan = {};
    plan.function = function;
    plan.inputCount = schedulingInputCount;
    inputCount = schedulingInputCount;

    StructuredSchedulingTilingDemand tilingDemand;
    if (mlir::failed(collectStructuredSchedulingTilingDemand(
            function, schedulingInputCount, tilingDemand)))
      return mlir::failure();
    if (!tilingDemand.succeeded) {
      plan.succeeded = false;
      plan.failureReason =
          "tiling demand failed: " + tilingDemand.failureReason;
      return mlir::success();
    }

    mlir::Block &block = function.getBody().front();
    initializeBoundary(block, plan);
    for (const OpTilingDemand &opDemand : tilingDemand.ops)
      collectOpPlan(opDemand, plan);
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
    if (returnOp)
      collectResultMaterializations(returnOp.getOperands(), "function result",
                                    plan);
    return mlir::success();
  }

private:
  llvm::DenseMap<mlir::Value, MemLayout> valueLayouts;
  llvm::DenseMap<mlir::Operation *, unsigned> opOrdinals;
  unsigned nextMaterialization = 0;
  unsigned inputCount = 0;

  void initializeBoundary(mlir::Block &block,
                          StructuredSchedulingLayoutPlan &plan) {
    for (auto [index, arg] : llvm::enumerate(block.getArguments())) {
      if (index < inputCount) {
        plan.boundaryValues.push_back({LayoutPlanValueRole::Input,
                                       static_cast<unsigned>(index),
                                       arg,
                                       arg.getType(),
                                       {},
                                       MemLayout::Tensor});
      } else {
        plan.boundaryValues.push_back(
            {LayoutPlanValueRole::Output,
             static_cast<unsigned>(index - inputCount),
             arg,
             arg.getType(),
             {},
             MemLayout::Tensor});
      }
      valueLayouts[arg] = MemLayout::Tensor;
    }
  }

  MemLayout lookupLayout(mlir::Value value) const {
    auto it = valueLayouts.find(value);
    if (it != valueLayouts.end())
      return it->second;
    return MemLayout::Tensor;
  }

  std::string describeValue(mlir::Value value) const {
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      unsigned argNumber = arg.getArgNumber();
      std::string label;
      llvm::raw_string_ostream os(label);
      if (argNumber < inputCount)
        os << "boundary input #" << argNumber;
      else
        os << "boundary output #" << (argNumber - inputCount);
      return os.str();
    }

    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      std::string label;
      llvm::raw_string_ostream os(label);
      auto it = opOrdinals.find(result.getOwner());
      if (it != opOrdinals.end())
        os << "op #" << it->second << " result #" << result.getResultNumber();
      else
        os << result.getOwner()->getName().getStringRef() << " result #"
           << result.getResultNumber();
      return os.str();
    }

    return "<unknown>";
  }

  static std::string describeUse(const OpTilingDemand &opDemand,
                                 const LayoutPlanValue &value) {
    std::string label;
    llvm::raw_string_ostream os(label);
    os << "op #" << opDemand.opIndex << " ";
    switch (value.role) {
    case LayoutPlanValueRole::Input:
      os << "input";
      break;
    case LayoutPlanValueRole::Output:
      os << "output";
      break;
    case LayoutPlanValueRole::Result:
      os << "result";
      break;
    }
    os << " #" << value.index;
    return os.str();
  }

  void addMaterializationIfNeeded(OpLayoutPlan &opPlan,
                                  const OpTilingDemand &opDemand,
                                  const LayoutPlanValue &value) {
    if (value.role == LayoutPlanValueRole::Result ||
        value.relation == LayoutPlanRelation::Broadcast)
      return;

    MemLayout sourceLayout = lookupLayout(value.value);
    if (sourceLayout == value.layout)
      return;

    opPlan.materializations.push_back(
        {nextMaterialization++, describeValue(value.value),
         describeUse(opDemand, value), value.type, sourceLayout, value.layout});
  }

  void collectOpPlan(const OpTilingDemand &opDemand,
                     StructuredSchedulingLayoutPlan &plan) {
    opOrdinals[opDemand.op] = opDemand.opIndex;

    OpLayoutPlan opPlan;
    opPlan.kind = opDemand.kind;
    opPlan.op = opDemand.op;
    opPlan.opIndex = opDemand.opIndex;
    opPlan.accumulators = opDemand.accumulators;
    opPlan.collectiveInfo = opDemand.collectiveInfo;
    opPlan.failureReason = opDemand.failureReason;

    if (opDemand.kind == OpTilingDemandKind::Failure) {
      plan.succeeded = false;
      plan.failureReason = opDemand.failureReason;
      plan.ops.push_back(std::move(opPlan));
      return;
    }

    if (opDemand.kind == OpTilingDemandKind::Support) {
      for (mlir::Value result : opDemand.op->getResults())
        valueLayouts[result] = MemLayout::Tensor;
      plan.ops.push_back(std::move(opPlan));
      return;
    }

    MemLayout defaultLayout = chooseDefaultLayout(opDemand);
    TilingDemandSlice resultSlice = chooseResultSlice(opDemand);

    for (const TilingDemandValue &tilingValue : opDemand.values) {
      LayoutPlanValue value{toLayoutRole(tilingValue.role),
                            tilingValue.index,
                            tilingValue.value,
                            tilingValue.type,
                            tilingValue.slice,
                            defaultLayout,
                            LayoutPlanRelation::Exact};
      if (opDemand.kind == OpTilingDemandKind::Linalg &&
          !isContractionLike(opDemand) &&
          value.role == LayoutPlanValueRole::Input) {
        value.relation = relationToResult(tilingValue, resultSlice);
        if (value.relation == LayoutPlanRelation::Broadcast)
          value.layout = MemLayout::Tensor;
      }
      if (opDemand.kind == OpTilingDemandKind::LinalgExtCollective)
        value.layout = MemLayout::Tensor;

      addMaterializationIfNeeded(opPlan, opDemand, value);

      if (value.role == LayoutPlanValueRole::Result)
        valueLayouts[value.value] = value.layout;
      opPlan.values.push_back(value);
    }

    plan.ops.push_back(std::move(opPlan));
  }

  static LayoutPlanValueRole toLayoutRole(TilingDemandValueRole role) {
    switch (role) {
    case TilingDemandValueRole::Input:
      return LayoutPlanValueRole::Input;
    case TilingDemandValueRole::Output:
      return LayoutPlanValueRole::Output;
    case TilingDemandValueRole::Result:
      return LayoutPlanValueRole::Result;
    }
    llvm_unreachable("unknown tiling demand role");
  }

  MemLayout chooseDefaultLayout(const OpTilingDemand &opDemand) const {
    if (opDemand.kind == OpTilingDemandKind::LinalgExtCollective)
      return MemLayout::Tensor;
    if (opDemand.kind == OpTilingDemandKind::Linalg &&
        isContractionLike(opDemand))
      return MemLayout::Cx;

    for (const TilingDemandValue &value : opDemand.values) {
      if (value.role == TilingDemandValueRole::Input &&
          isTensorLike(value.type))
        return lookupLayout(value.value);
    }
    return MemLayout::Tensor;
  }

  static TilingDemandSlice chooseResultSlice(const OpTilingDemand &opDemand) {
    for (const TilingDemandValue &value : opDemand.values) {
      if (value.role == TilingDemandValueRole::Result)
        return value.slice;
    }
    return {};
  }

  void collectResultMaterializations(mlir::ValueRange yieldedValues,
                                     llvm::StringRef resultLabel,
                                     StructuredSchedulingLayoutPlan &plan) {
    for (auto [index, value] : llvm::enumerate(yieldedValues)) {
      MemLayout sourceLayout = lookupLayout(value);
      if (sourceLayout == MemLayout::Tensor)
        continue;
      std::string label;
      llvm::raw_string_ostream os(label);
      os << resultLabel << " #" << index;
      plan.resultMaterializations.push_back(
          {nextMaterialization++, describeValue(value), os.str(),
           value.getType(), sourceLayout, MemLayout::Tensor});
    }
  }
};

} // namespace

mlir::LogicalResult wafer::collectStructuredSchedulingLayoutPlan(
    mlir::func::FuncOp function, unsigned inputCount,
    StructuredSchedulingLayoutPlan &plan) {
  LayoutPlanner planner;
  return planner.run(function, inputCount, plan);
}
