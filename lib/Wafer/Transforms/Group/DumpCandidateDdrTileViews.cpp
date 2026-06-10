//===- DumpCandidateDdrTileViews.cpp - Dump candidate DDR tile views ------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer {
#define GEN_PASS_DEF_DUMPCANDIDATEDDRTILEVIEWSPASS
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

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
parseI64List(llvm::StringRef text, llvm::StringRef optionName,
             mlir::Operation *anchor) {
  llvm::SmallVector<int64_t, 4> values;
  if (text.empty()) {
    anchor->emitError() << optionName << " must not be empty";
    return mlir::failure();
  }

  llvm::SmallVector<llvm::StringRef, 4> parts;
  llvm::SplitString(text, parts, ",");
  for (llvm::StringRef part : parts) {
    int64_t value = 0;
    if (part.trim().getAsInteger(10, value)) {
      anchor->emitError() << "invalid integer in " << optionName << ": "
                          << part;
      return mlir::failure();
    }
    values.push_back(value);
  }
  return values;
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

struct DumpCandidateDdrTileViewsPass
    : public impl::DumpCandidateDdrTileViewsPassBase<
          DumpCandidateDdrTileViewsPass> {
  using impl::DumpCandidateDdrTileViewsPassBase<
      DumpCandidateDdrTileViewsPass>::DumpCandidateDdrTileViewsPassBase;

  void runOnOperation() final {
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> parsedOffsets = parseI64List(
        candidateTileOffsets, "candidate-tile-offsets", getOperation());
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> parsedSizes = parseI64List(
        candidateTileSizes, "candidate-tile-sizes", getOperation());
    if (mlir::failed(parsedOffsets) || mlir::failed(parsedSizes)) {
      signalPassFailure();
      return;
    }
    if ((*parsedOffsets).size() != (*parsedSizes).size()) {
      getOperation()->emitError()
          << "candidate-tile-offsets and candidate-tile-sizes rank mismatch";
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
      if (mlir::failed(lowerCandidateGroupToTileRegionModule(
              group, *parsedOffsets, *parsedSizes, loweredModule,
              &failureReason))) {
        llvm::errs() << "wafer.candidate_ddr_tile_views group " << labelOs.str()
                     << "\n";
        llvm::errs() << "  failure "
                     << (failureReason.empty()
                             ? llvm::StringRef("candidate DDR tile-view "
                                               "materialization failed")
                             : llvm::StringRef(failureReason))
                     << "\n";
        return;
      }

      llvm::errs() << "wafer.candidate_ddr_tile_views group " << labelOs.str()
                   << " offsets=";
      printI64List(*parsedOffsets, llvm::errs());
      llvm::errs() << " sizes=";
      printI64List(*parsedSizes, llvm::errs());
      llvm::errs() << "\n";
      loweredModule->print(llvm::errs());
      llvm::errs() << "\n";
    });
    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace wafer
