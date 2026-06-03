//===- DumpTileRegionCandidate.cpp - Dump Wafer tile-region candidates ---===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Group/TileRegionCandidate.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer {
namespace {

static std::string getNearestSymbolName(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto name = parent->getAttrOfType<mlir::StringAttr>("sym_name"))
      return ("@" + name.getValue()).str();
  }
  return "@<unknown>";
}

struct DumpTileRegionCandidatePass
    : public mlir::PassWrapper<DumpTileRegionCandidatePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpTileRegionCandidatePass)

  llvm::StringRef getArgument() const final {
    return "wafer-dump-tile-region-candidate";
  }

  llvm::StringRef getDescription() const final {
    return "dump R3.2c provisional wafer.tile_region candidates";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, wafer::WaferDialect>();
  }

  void runOnOperation() final {
    llvm::DenseMap<mlir::Operation *, unsigned> groupOrdinals;
    getOperation().walk([&](GroupOp group) {
      std::string symbolName = getNearestSymbolName(group.getOperation());
      unsigned ordinal = groupOrdinals[group->getParentOp()]++;
      std::string label;
      llvm::raw_string_ostream labelOs(label);
      labelOs << symbolName << "#" << ordinal;

      TileRegionCandidate candidate;
      if (mlir::failed(buildTileRegionCandidate(group, candidate))) {
        signalPassFailure();
        return;
      }
      dumpTileRegionCandidate(candidate, labelOs.str(), llvm::errs());
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createDumpTileRegionCandidatePass() {
  return std::make_unique<DumpTileRegionCandidatePass>();
}

} // namespace wafer
