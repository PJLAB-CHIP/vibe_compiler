//===- LayoutPlanningAnalysis.cpp - Wafer group layout planning analysis --===//

#include "Wafer/Transforms/Group/LayoutPlanningAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;

namespace {

static llvm::StringRef layoutName(MemLayout layout) {
  switch (layout) {
  case MemLayout::Tensor:
    return "tensor";
  case MemLayout::NTensor:
    return "ntensor";
  case MemLayout::Cx:
    return "cx";
  case MemLayout::NCx:
    return "ncx";
  }
  llvm_unreachable("unknown memory layout");
}

static void printSlice(const TilingDemandSlice &slice, llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, dim] : llvm::enumerate(slice.loopDims)) {
    if (index != 0)
      os << ",";
    os << "d" << dim;
  }
  os << "]";
}

static void printDimList(llvm::ArrayRef<unsigned> dims, llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, dim] : llvm::enumerate(dims)) {
    if (index != 0)
      os << ",";
    os << "d" << dim;
  }
  os << "]";
}

static bool sameSlice(const TilingDemandSlice &lhs,
                      const TilingDemandSlice &rhs) {
  return lhs.loopDims == rhs.loopDims;
}

static bool isTensorLike(mlir::Type type) {
  return mlir::isa<mlir::RankedTensorType>(type);
}

static bool isContractionLike(const OpTilingDemand &demand) {
  return !demand.accumulators.empty() ||
         mlir::isa_and_nonnull<mlir::linalg::MatmulOp>(demand.op);
}

static LayoutPlanRelation relationToResult(const TilingDemandValue &value,
                                           const TilingDemandSlice &result) {
  if (sameSlice(value.slice, result))
    return LayoutPlanRelation::Exact;
  return LayoutPlanRelation::Broadcast;
}

class LayoutPlanner {
public:
  mlir::LogicalResult run(GroupOp group, GroupLayoutPlan &plan) {
    plan = {};
    plan.group = group;

    GroupTilingDemand tilingDemand;
    if (mlir::failed(collectGroupTilingDemand(group, tilingDemand)))
      return mlir::failure();
    if (!tilingDemand.succeeded) {
      plan.succeeded = false;
      plan.failureReason =
          "tiling demand failed: " + tilingDemand.failureReason;
      return mlir::success();
    }

    initializeBoundary(group, plan);
    for (const OpTilingDemand &opDemand : tilingDemand.ops)
      collectOpPlan(opDemand, plan);
    collectGroupResultMaterializations(group, plan);
    return mlir::success();
  }

private:
  llvm::DenseMap<mlir::Value, MemLayout> valueLayouts;
  llvm::DenseMap<mlir::Operation *, unsigned> opOrdinals;
  unsigned nextMaterialization = 0;

  void initializeBoundary(GroupOp group, GroupLayoutPlan &plan) {
    mlir::Block &block = group.getBody().front();
    unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
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
      auto group = mlir::cast<GroupOp>(arg.getOwner()->getParentOp());
      unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
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

  void collectOpPlan(const OpTilingDemand &opDemand, GroupLayoutPlan &plan) {
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
      if (opDemand.kind == OpTilingDemandKind::TensorCollective)
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
    if (opDemand.kind == OpTilingDemandKind::TensorCollective)
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

  void collectGroupResultMaterializations(GroupOp group,
                                          GroupLayoutPlan &plan) {
    auto yield =
        mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
    if (!yield)
      return;

    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      MemLayout sourceLayout = lookupLayout(value);
      if (sourceLayout == MemLayout::Tensor)
        continue;
      std::string label;
      llvm::raw_string_ostream os(label);
      os << "group result #" << index;
      plan.resultMaterializations.push_back({nextMaterialization++,
                                             describeValue(value),
                                             os.str(), value.getType(),
                                             sourceLayout, MemLayout::Tensor});
    }
  }
};

static void printRole(LayoutPlanValueRole role, llvm::raw_ostream &os) {
  switch (role) {
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
}

static llvm::StringRef collectiveKindName(WaferTensorCollectiveKind kind) {
  switch (kind) {
  case WaferTensorCollectiveKind::AllGather:
    return "all_gather";
  case WaferTensorCollectiveKind::ReduceScatter:
    return "reduce_scatter";
  case WaferTensorCollectiveKind::AllReduce:
    return "all_reduce";
  case WaferTensorCollectiveKind::AllToAll:
    return "all_to_all";
  case WaferTensorCollectiveKind::CollectivePermute:
    return "collective_permute";
  }
  llvm_unreachable("unknown tensor collective kind");
}

static void printI64List(llvm::ArrayRef<int64_t> values,
                         llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      os << ",";
    os << value;
  }
  os << "]";
}

static void dumpMaterialization(const LayoutMaterializationDemand &demand,
                                llvm::raw_ostream &os) {
  os << "    materialization #" << demand.index << " " << demand.source << " ";
  if (demand.source != demand.target)
    os << "-> " << demand.target << " ";
  os << layoutName(demand.sourceLayout) << "->"
     << layoutName(demand.targetLayout) << " type=" << demand.type << "\n";
}

} // namespace

mlir::LogicalResult wafer::collectGroupLayoutPlan(GroupOp group,
                                                  GroupLayoutPlan &plan) {
  LayoutPlanner planner;
  return planner.run(group, plan);
}

void wafer::dumpGroupLayoutPlan(const GroupLayoutPlan &plan,
                                llvm::StringRef groupLabel,
                                llvm::raw_ostream &os) {
  os << "wafer.layout_plan group " << groupLabel << "\n";
  if (!plan.succeeded && plan.ops.empty()) {
    os << "  failure " << plan.failureReason << "\n";
    return;
  }

  for (const LayoutPlanValue &boundary : plan.boundaryValues) {
    os << "  boundary ";
    printRole(boundary.role, os);
    os << " #" << boundary.index << " layout=" << layoutName(boundary.layout)
       << " type=" << boundary.type << "\n";
  }

  for (const OpLayoutPlan &opPlan : plan.ops) {
    os << "  op #" << opPlan.opIndex << " "
       << opPlan.op->getName().getStringRef() << "\n";

    if (opPlan.kind == OpTilingDemandKind::Failure) {
      os << "    failure " << opPlan.failureReason << "\n";
      continue;
    }

    if (opPlan.kind == OpTilingDemandKind::TensorCollective) {
      os << "    collective kind="
         << collectiveKindName(opPlan.collectiveInfo.kind);
      if (!opPlan.collectiveInfo.rankGroup.empty()) {
        os << " rank_group=";
        printI64List(opPlan.collectiveInfo.rankGroup, os);
      }
      os << "\n";
    }

    for (const LayoutPlanValue &value : opPlan.values) {
      os << "    ";
      printRole(value.role, os);
      os << " #" << value.index << " layout=" << layoutName(value.layout)
         << " slice=";
      printSlice(value.slice, os);
      if (value.relation == LayoutPlanRelation::Broadcast)
        os << " relation=broadcast";
      os << "\n";
    }

    for (const TilingDemandAccumulator &accumulator : opPlan.accumulators) {
      os << "    accumulator result #" << accumulator.resultIndex
         << " layout=cx reduction_dims=";
      printDimList(accumulator.reductionDims, os);
      os << "\n";
    }

    for (const LayoutMaterializationDemand &materialization :
         opPlan.materializations)
      dumpMaterialization(materialization, os);
  }

  for (const LayoutMaterializationDemand &materialization :
       plan.resultMaterializations)
    dumpMaterialization(materialization, os);
}
