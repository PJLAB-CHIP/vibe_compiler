//===- DDRHighWater.cpp - Accepted DDR arena high-water -----------------===//

#include "Internal.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace wafer::analysis::detail {
namespace {

class DDRHighWaterAnalysis {
public:
  explicit DDRHighWaterAnalysis(InstructionProgramCost &cost)
      : metric(cost.ddrHighWaterBytes),
        bufferCount(cost.compilerOwnedDDRBufferCount) {}

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
    llvm::DenseSet<mlir::Operation *> seenAllocations;
    while (!scopes.empty()) {
      mlir::Operation *scope = scopes.pop_back_val();
      if (!scope || !seenScopes.insert(scope).second)
        continue;
      scope->walk([&](mlir::memref::AllocOp allocation) {
        if (!isWaferDDRMemRefType(allocation.getType()) ||
            !seenAllocations.insert(allocation.getOperation()).second)
          return;
        account(allocation);
      });
      scope->walk([&](mlir::func::CallOp call) {
        mlir::func::FuncOp callee =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                call, call.getCalleeAttr());
        if (callee && !callee.isDeclaration())
          scopes.push_back(callee.getOperation());
      });
    }
  }

private:
  void account(mlir::memref::AllocOp allocation) {
    add(bufferCount, Quantity{1});
    auto offset =
        allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
    if (!offset) {
      degrade(metric, ScheduleCostKnowledge::Unknown,
              ScheduleCostReason::MissingAcceptedDDROffset);
      return;
    }
    if (offset.getOffset() < 0) {
      degrade(metric, ScheduleCostKnowledge::Unsupported,
              ScheduleCostReason::InvalidAcceptedDDROffset);
      return;
    }
    std::optional<WaferPhysicalTensorInfo> physical =
        computeWaferPhysicalTensorInfo(allocation.getType());
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
      metric.value = std::max(metric.value, end);
  }

  ScheduleCostMetric &metric;
  ScheduleCostMetric &bufferCount;
};

} // namespace

void collectDDRHighWater(mlir::Operation *root, InstructionProgramCost &cost) {
  DDRHighWaterAnalysis(cost).run(root);
}

} // namespace wafer::analysis::detail
