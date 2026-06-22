//===- MaterializeTargetTopology.cpp - Materialize Wafer topology facts ---===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <memory>

namespace wafer {
#define GEN_PASS_DEF_MATERIALIZETARGETTOPOLOGYPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
parseOptionalI64List(llvm::StringRef text, llvm::StringRef optionName,
                     mlir::Operation *anchor) {
  llvm::SmallVector<int64_t, 8> values;
  text = text.trim();
  if (text.empty())
    return values;

  if (text.starts_with(",") || text.ends_with(",")) {
    anchor->emitError() << "topology_failure: invalid empty entry in "
                        << optionName;
    return mlir::failure();
  }

  llvm::StringRef rest = text;
  while (!rest.empty()) {
    auto split = rest.split(',');
    llvm::StringRef part = split.first.trim();
    if (part.empty()) {
      anchor->emitError() << "topology_failure: invalid empty entry in "
                          << optionName;
      return mlir::failure();
    }
    int64_t value = 0;
    if (part.getAsInteger(10, value)) {
      anchor->emitError() << "topology_failure: invalid integer in "
                          << optionName << ": " << part;
      return mlir::failure();
    }
    values.push_back(value);
    rest = split.second;
  }
  return values;
}

static mlir::LogicalResult
validateTopologyShape(mlir::ModuleOp moduleOp, int64_t cardYCount,
                      int64_t cardXCount, int64_t tileYCount,
                      int64_t tileXCount) {
  if (cardYCount <= 0 || cardXCount <= 0 || tileYCount <= 0 ||
      tileXCount <= 0)
    return moduleOp.emitError()
           << "topology_failure: target topology dimensions must be positive";

  int64_t cardCount = 0;
  int64_t rowCount = 0;
  int64_t totalTileCount = 0;
  if (!checkedMul(cardYCount, cardXCount, cardCount) ||
      !checkedMul(cardCount, tileYCount, rowCount) ||
      !checkedMul(rowCount, tileXCount, totalTileCount))
    return moduleOp.emitError()
           << "topology_failure: target topology tile count is too large";

  return mlir::success();
}

static bool isSupportedCardInterconnect(llvm::StringRef interconnect) {
  return interconnect == "mesh" || interconnect == "torus";
}

static mlir::LogicalResult materializeTargetTopology(
    mlir::ModuleOp moduleOp, llvm::StringRef topologyName, int64_t cardYCount,
    int64_t cardXCount, llvm::StringRef cardInterconnect, int64_t tileYCount,
    int64_t tileXCount, llvm::StringRef unavailableTilesOption) {
  bool hasTopology = false;
  moduleOp.walk([&](TargetTopologyOp) { hasTopology = true; });
  if (hasTopology)
    return moduleOp.emitError()
           << "topology_failure: module already contains "
              "wafer.target.topology";

  if (topologyName.empty())
    return moduleOp.emitError()
           << "topology_failure: topology symbol name must not be empty";

  if (!isSupportedCardInterconnect(cardInterconnect))
    return moduleOp.emitError()
           << "topology_failure: card-interconnect must be mesh or torus";

  if (mlir::failed(validateTopologyShape(moduleOp, cardYCount, cardXCount,
                                         tileYCount, tileXCount)))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<int64_t, 8>> unavailableTiles =
      parseOptionalI64List(unavailableTilesOption, "unavailable-tiles",
                           moduleOp);
  if (mlir::failed(unavailableTiles))
    return mlir::failure();
  if (unavailableTiles->size() % 4 != 0)
    return moduleOp.emitError()
           << "topology_failure: unavailable-tiles must contain "
              "card_y/card_x/tile_y/tile_x tuples";

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  builder.create<TargetTopologyOp>(
      moduleOp.getLoc(), builder.getStringAttr(topologyName),
      mlir::DenseI64ArrayAttr::get(builder.getContext(),
                                   {cardYCount, cardXCount}),
      builder.getStringAttr(cardInterconnect),
      mlir::DenseI64ArrayAttr::get(builder.getContext(),
                                   {tileYCount, tileXCount}),
      mlir::DenseI64ArrayAttr::get(builder.getContext(), *unavailableTiles));

  return mlir::success();
}

struct MaterializeTargetTopologyPass
    : public impl::MaterializeTargetTopologyPassBase<
          MaterializeTargetTopologyPass> {
  using impl::MaterializeTargetTopologyPassBase<
      MaterializeTargetTopologyPass>::MaterializeTargetTopologyPassBase;

  void runOnOperation() final {
    if (mlir::failed(materializeTargetTopology(
            getOperation(), topologyName, cardYCount, cardXCount,
            cardInterconnect, tileYCount, tileXCount, unavailableTiles))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace wafer
