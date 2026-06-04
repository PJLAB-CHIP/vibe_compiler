//===- DumpTilingDemand.cpp - Dump Wafer group tiling demand -------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Analysis/Group/TilingDemandAnalysis.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer {
#define GEN_PASS_DEF_DUMPGROUPTILINGDEMANDPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

static std::string getNearestSymbolName(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto name = parent->getAttrOfType<mlir::StringAttr>("sym_name"))
      return ("@" + name.getValue()).str();
  }
  return "@<unknown>";
}

struct DumpGroupTilingDemandPass
    : public impl::DumpGroupTilingDemandPassBase<DumpGroupTilingDemandPass> {
  using impl::DumpGroupTilingDemandPassBase<
      DumpGroupTilingDemandPass>::DumpGroupTilingDemandPassBase;

  void runOnOperation() final {
    llvm::DenseMap<mlir::Operation *, unsigned> groupOrdinals;
    getOperation().walk([&](GroupOp group) {
      std::string symbolName = getNearestSymbolName(group.getOperation());
      unsigned ordinal = groupOrdinals[group->getParentOp()]++;
      std::string label;
      llvm::raw_string_ostream labelOs(label);
      labelOs << symbolName << "#" << ordinal;

      GroupTilingDemand demand;
      if (mlir::failed(collectGroupTilingDemand(group, demand))) {
        signalPassFailure();
        return;
      }
      dumpGroupTilingDemand(demand, labelOs.str(), llvm::errs());
    });
    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace wafer
