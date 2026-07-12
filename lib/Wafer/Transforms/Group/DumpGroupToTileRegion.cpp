//===- DumpGroupToTileRegion.cpp - Dump group-to-tile-region lowering -----===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer {
#define GEN_PASS_DEF_DUMPGROUPTOTILEREGIONPASS
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

struct DumpGroupToTileRegionPass
    : public impl::DumpGroupToTileRegionPassBase<DumpGroupToTileRegionPass> {
  using impl::DumpGroupToTileRegionPassBase<
      DumpGroupToTileRegionPass>::DumpGroupToTileRegionPassBase;

  void runOnOperation() final {
    if (logicalRank < 0) {
      getOperation()->emitError()
          << "missing_logical_rank: group-to-tile-region dump requires an "
             "explicit non-negative logical-rank";
      signalPassFailure();
      return;
    }
    llvm::DenseMap<mlir::Operation *, unsigned> groupOrdinals;
    getOperation().walk([&](GroupOp group) {
      std::string symbolName = getNearestSymbolName(group.getOperation());
      unsigned ordinal = groupOrdinals[group->getParentOp()]++;
      std::string label;
      llvm::raw_string_ostream labelOs(label);
      labelOs << symbolName << "#" << ordinal;

      mlir::OwningOpRef<mlir::ModuleOp> loweredModule;
      std::string failureReason;
      if (mlir::failed(lowerGroupToTileRegionModule(group, loweredModule,
                                                    &failureReason,
                                                    logicalRank))) {
        llvm::errs() << "wafer.group_to_tile_region group " << labelOs.str()
                     << "\n";
        llvm::errs() << "  failure "
                     << (failureReason.empty()
                             ? llvm::StringRef("group-to-tile-region lowering "
                                               "failed")
                             : llvm::StringRef(failureReason))
                     << "\n";
        return;
      }
      dumpGroupToTileRegionModule(*loweredModule, labelOs.str(), llvm::errs());
    });
    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace wafer
