//===- SPMHighWater.cpp - Accepted SPM address high-water ------*- C++ -*-===//

#include "Internal.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace wafer::analysis::detail {
namespace {

static mlir::Value resolveSPMRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArg.getOwner();
      auto tileRegion =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (!tileRegion || tileRegion.getBody().empty() ||
          owner != &tileRegion.getBody().front() ||
          blockArg.getArgNumber() >= tileRegion.getInputs().size())
        return value;
      value = tileRegion.getInputs()[blockArg.getArgNumber()];
      continue;
    }
    mlir::Operation *def = value.getDefiningOp();
    if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(def)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      auto yield = tileRegion.getBody().empty()
                       ? TileYieldOp{}
                       : mlir::dyn_cast<TileYieldOp>(
                             tileRegion.getBody().front().getTerminator());
      if (!result || !yield ||
          result.getResultNumber() >= yield.getNumOperands())
        return value;
      value = yield.getOperand(result.getResultNumber());
      continue;
    }
    auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(def);
    if (!view)
      return value;
    value = view.getViewSource();
  }
  return value;
}

class SPMHighWaterAnalysis {
public:
  SPMHighWaterAnalysis(InstructionProgramCost &cost,
                       const TargetScheduleCostPolicy &policy)
      : metric(cost.spmHighWaterBytes), policy(policy) {}

  void run(mlir::Operation *root) {
    llvm::SmallVector<mlir::Operation *, 8> scopes{root};
    llvm::DenseSet<mlir::Operation *> seenScopes;
    while (!scopes.empty()) {
      mlir::Operation *scope = scopes.pop_back_val();
      if (!seenScopes.insert(scope).second)
        continue;
      scope->walk([&](mlir::memref::AllocOp alloc) {
        if (isWaferSPMMemRefType(alloc.getType()))
          accountAllocation(alloc);
      });
      scope->walk([&](mlir::Operation *op) {
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          mlir::func::FuncOp callee =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                  call, call.getCalleeAttr());
          if (callee && !callee.isDeclaration())
            scopes.push_back(callee.getOperation());
        }
        if (!mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp>(op))
          return;
        for (mlir::Value value : op->getOperands())
          accountSPMValue(value);
        for (mlir::Value value : op->getResults())
          accountSPMValue(value);
      });
    }
  }

private:
  void accountSPMValue(mlir::Value value) {
    if (!isWaferSPMMemRefType(value.getType()))
      return;
    mlir::Value root = resolveSPMRoot(value);
    auto alloc = root.getDefiningOp<mlir::memref::AllocOp>();
    if (!alloc) {
      degrade(metric, ScheduleCostKnowledge::Unknown,
              ScheduleCostReason::UnsupportedSPMRoot);
      return;
    }
    accountAllocation(alloc);
  }

  void accountAllocation(mlir::memref::AllocOp alloc) {
    if (!seenAllocs.insert(alloc.getOperation()).second)
      return;
    auto offset = alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
    if (!offset) {
      degrade(metric, ScheduleCostKnowledge::Unknown,
              ScheduleCostReason::MissingAcceptedSPMOffset);
      return;
    }
    if (offset.getOffset() < 0 ||
        static_cast<uint64_t>(offset.getOffset()) < policy.spmAddressBase) {
      degrade(metric, ScheduleCostKnowledge::Unsupported,
              ScheduleCostReason::InvalidAcceptedSPMOffset);
      return;
    }
    std::optional<WaferPhysicalTensorInfo> physical =
        computeWaferPhysicalTensorInfo(alloc.getType());
    if (!physical || physical->physicalBytes < 0) {
      degrade(metric, ScheduleCostKnowledge::Unknown,
              ScheduleCostReason::UnknownPhysicalGeometry);
      return;
    }
    uint64_t end = 0;
    if (!checkedAdd(static_cast<uint64_t>(offset.getOffset()),
                    static_cast<uint64_t>(physical->physicalBytes), end)) {
      degrade(metric, ScheduleCostKnowledge::Overflow,
              ScheduleCostReason::ArithmeticOverflow);
      return;
    }
    if (metric.isKnown())
      metric.value = std::max(metric.value, end - policy.spmAddressBase);
  }

  ScheduleCostMetric &metric;
  const TargetScheduleCostPolicy &policy;
  llvm::DenseSet<mlir::Operation *> seenAllocs;
};

} // namespace

void collectSPMHighWater(mlir::Operation *root, InstructionProgramCost &cost,
                         const TargetScheduleCostPolicy &policy) {
  SPMHighWaterAnalysis(cost, policy).run(root);
}

} // namespace wafer::analysis::detail
