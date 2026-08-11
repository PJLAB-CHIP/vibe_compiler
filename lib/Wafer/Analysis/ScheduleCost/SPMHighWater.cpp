//===- SPMHighWater.cpp - Accepted SPM address high-water ------*- C++ -*-===//

#include "Internal.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace wafer::analysis::detail {
namespace {

struct SPMRootResolution {
  llvm::SmallVector<mlir::memref::AllocOp, 4> allocations;
  bool complete = true;
};

static SPMRootResolution resolveSPMRoots(mlir::Value initialValue) {
  SPMRootResolution resolution;
  llvm::SmallVector<mlir::Value, 8> worklist{initialValue};
  llvm::DenseSet<mlir::Value> seenValues;
  llvm::DenseSet<mlir::Operation *> seenAllocations;

  auto enqueueLoopCarriedOrigins = [&](mlir::scf::ForOp loop,
                                       unsigned iterArgNumber) {
    auto initArgs = loop.getInitArgs();
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    if (iterArgNumber >= initArgs.size() || !yield ||
        iterArgNumber >= yield.getResults().size()) {
      resolution.complete = false;
      return;
    }
    worklist.push_back(initArgs[iterArgNumber]);
    worklist.push_back(yield.getResults()[iterArgNumber]);
  };

  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!value || !seenValues.insert(value).second)
      continue;

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArg.getOwner();
      auto tileRegion =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (tileRegion && !tileRegion.getBody().empty() &&
          owner == &tileRegion.getBody().front() &&
          blockArg.getArgNumber() < tileRegion.getInputs().size()) {
        worklist.push_back(tileRegion.getInputs()[blockArg.getArgNumber()]);
        continue;
      }

      auto loop =
          owner ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(owner->getParentOp())
                : mlir::scf::ForOp{};
      if (loop && owner == loop.getBody() && blockArg.getArgNumber() > 0) {
        enqueueLoopCarriedOrigins(loop, blockArg.getArgNumber() - 1);
        continue;
      }

      resolution.complete = false;
      continue;
    }

    mlir::Operation *def = value.getDefiningOp();
    if (auto alloc = mlir::dyn_cast_or_null<mlir::memref::AllocOp>(def)) {
      if (!isWaferSPMMemRefType(alloc.getType())) {
        resolution.complete = false;
        continue;
      }
      if (seenAllocations.insert(alloc.getOperation()).second)
        resolution.allocations.push_back(alloc);
      continue;
    }

    if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(def)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      auto yield = tileRegion.getBody().empty()
                       ? TileYieldOp{}
                       : mlir::dyn_cast<TileYieldOp>(
                             tileRegion.getBody().front().getTerminator());
      if (!result || !yield ||
          result.getResultNumber() >= yield.getNumOperands()) {
        resolution.complete = false;
        continue;
      }
      worklist.push_back(yield.getOperand(result.getResultNumber()));
      continue;
    }

    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(def)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      if (!result) {
        resolution.complete = false;
        continue;
      }
      enqueueLoopCarriedOrigins(loop, result.getResultNumber());
      continue;
    }

    if (auto ifOp = mlir::dyn_cast_or_null<mlir::scf::IfOp>(def)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      auto thenYield = ifOp.thenBlock() ? mlir::dyn_cast<mlir::scf::YieldOp>(
                                              ifOp.thenBlock()->getTerminator())
                                        : mlir::scf::YieldOp{};
      auto elseYield = ifOp.elseBlock() ? mlir::dyn_cast<mlir::scf::YieldOp>(
                                              ifOp.elseBlock()->getTerminator())
                                        : mlir::scf::YieldOp{};
      if (!result || !thenYield || !elseYield ||
          result.getResultNumber() >= thenYield.getNumOperands() ||
          result.getResultNumber() >= elseYield.getNumOperands()) {
        resolution.complete = false;
        continue;
      }
      worklist.push_back(thenYield.getOperand(result.getResultNumber()));
      worklist.push_back(elseYield.getOperand(result.getResultNumber()));
      continue;
    }

    if (auto select =
            mlir::dyn_cast_or_null<mlir::SelectLikeOpInterface>(def)) {
      worklist.push_back(select.getTrueValue());
      worklist.push_back(select.getFalseValue());
      continue;
    }

    auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(def);
    if (view) {
      worklist.push_back(view.getViewSource());
      continue;
    }

    resolution.complete = false;
  }

  if (resolution.allocations.empty())
    resolution.complete = false;
  return resolution;
}

class SPMHighWaterAnalysis {
public:
  SPMHighWaterAnalysis(
      InstructionProgramCost &cost, const TargetScheduleCostPolicy &policy,
      llvm::function_ref<bool(mlir::Operation *)> includeOperation)
      : metric(cost.spmHighWaterBytes),
        bufferCount(cost.compilerOwnedSPMBufferCount), policy(policy),
        includeOperation(includeOperation) {}

  void run(mlir::Operation *root) {
    llvm::SmallVector<mlir::Operation *, 8> scopes;
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(root)) {
      for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
        if (!function.isPrivate())
          scopes.push_back(function.getOperation());
    } else {
      scopes.push_back(root);
    }
    llvm::DenseSet<mlir::Operation *> seenScopes;
    while (!scopes.empty()) {
      mlir::Operation *scope = scopes.pop_back_val();
      if (!seenScopes.insert(scope).second)
        continue;
      scope->walk([&](mlir::memref::AllocOp alloc) {
        if (isWaferSPMMemRefType(alloc.getType()) &&
            includeOperation(alloc.getOperation()))
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
        if (!includeOperation(op) ||
            !mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(op))
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
    SPMRootResolution roots = resolveSPMRoots(value);
    if (!roots.complete) {
      degrade(metric, ScheduleCostKnowledge::Unavailable,
              ScheduleCostReason::UnsupportedSPMRoot);
      return;
    }
    for (mlir::memref::AllocOp alloc : roots.allocations)
      accountAllocation(alloc);
  }

  void accountAllocation(mlir::memref::AllocOp alloc) {
    if (!includeOperation(alloc.getOperation()))
      return;
    if (!seenAllocs.insert(alloc.getOperation()).second)
      return;
    add(bufferCount, Quantity{1});
    auto offset = alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
    if (!offset) {
      degrade(metric, ScheduleCostKnowledge::Unavailable,
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
      degrade(metric, ScheduleCostKnowledge::Unavailable,
              ScheduleCostReason::UnavailablePhysicalGeometry);
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
  ScheduleCostMetric &bufferCount;
  const TargetScheduleCostPolicy &policy;
  llvm::function_ref<bool(mlir::Operation *)> includeOperation;
  llvm::DenseSet<mlir::Operation *> seenAllocs;
};

} // namespace

void collectSPMHighWater(mlir::Operation *root, InstructionProgramCost &cost,
                         const TargetScheduleCostPolicy &policy) {
  collectSPMHighWater(root, cost, policy,
                      [](mlir::Operation *) { return true; });
}

void collectSPMHighWater(
    mlir::Operation *root, InstructionProgramCost &cost,
    const TargetScheduleCostPolicy &policy,
    llvm::function_ref<bool(mlir::Operation *)> includeOperation) {
  SPMHighWaterAnalysis(cost, policy, includeOperation).run(root);
}

} // namespace wafer::analysis::detail
