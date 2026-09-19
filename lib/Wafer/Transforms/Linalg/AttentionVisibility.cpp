//===- AttentionVisibility.cpp - Selected attention block visibility
//-------===//

#include "AttentionVisibility.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace wafer {

mlir::LogicalResult materializeAttentionVisibility(mlir::RewriterBase &rewriter,
                                                   TileRegionOp region) {
  llvm::SmallVector<LinalgExtOnlineAttentionOp> tiles;
  region.walk([&](LinalgExtOnlineAttentionOp tile) {
    if (!tile.getPositions().empty())
      tiles.push_back(tile);
  });
  for (auto tile : tiles) {
    auto sizes = tile.getStaticLoopRanges();
    auto map = tile.getPositionMapAttr().getValue();
    for (auto expr : map.getResults())
      if (mlir::ShapedType::isDynamic(
              sizes[mlir::cast<mlir::AffineDimExpr>(expr).getPosition()]))
        return tile.emitOpError(
            "visibility requires statically sized selected tiles");
  }
  for (auto tile : tiles) {
    mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPoint(tile);
    auto loc = tile.getLoc();
    auto sizes = tile.getStaticLoopRanges();
    auto map = tile.getPositionMapAttr().getValue();
    auto query = tile.getPositions()[0], key = tile.getPositions()[1];
    auto validEnd = tile.getPositions()[2];
    auto end = [&](mlir::Value start, unsigned position) {
      unsigned dimension =
          mlir::cast<mlir::AffineDimExpr>(map.getResult(position))
              .getPosition();
      auto extent =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, sizes[dimension]);
      return rewriter.createOrFold<mlir::arith::AddIOp>(loc, start, extent);
    };
    auto compare = [&](mlir::arith::CmpIPredicate predicate, mlir::Value lhs,
                       mlir::Value rhs) {
      return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, predicate, lhs,
                                                        rhs);
    };
    auto visible = compare(mlir::arith::CmpIPredicate::ult, key, validEnd);
    auto full = compare(mlir::arith::CmpIPredicate::ule, end(key, 1), validEnd);
    if (tile.getCausal()) {
      auto causalVisible =
          compare(mlir::arith::CmpIPredicate::ult, key, end(query, 0));
      visible = rewriter.createOrFold<mlir::arith::AndIOp>(loc, visible,
                                                           causalVisible);
      auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto firstQueryEnd =
          rewriter.createOrFold<mlir::arith::AddIOp>(loc, query, one);
      auto causalFull =
          compare(mlir::arith::CmpIPredicate::ule, end(key, 1), firstQueryEnd);
      full = rewriter.createOrFold<mlir::arith::AndIOp>(loc, full, causalFull);
    }
    // Select constant predicates here: visibility is an actual materialization
    // boundary, so a downstream consumer must not need canonicalization to
    // discover that a published state has one storage owner.
    using Values = llvm::SmallVector<mlir::Value>;
    auto choose = [&](mlir::Value condition, auto thenBuilder,
                      auto elseBuilder) {
      llvm::APInt constant;
      if (mlir::matchPattern(condition, mlir::m_ConstantInt(&constant)))
        return constant.isZero() ? elseBuilder() : thenBuilder();
      auto branch = rewriter.create<mlir::scf::IfOp>(
          loc, tile->getResultTypes(), condition, true);
      rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
      rewriter.create<mlir::scf::YieldOp>(loc, thenBuilder());
      rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
      rewriter.create<mlir::scf::YieldOp>(loc, elseBuilder());
      rewriter.setInsertionPointAfter(branch);
      return Values(branch->getResults());
    };
    auto selected = choose(
        visible,
        [&]() -> Values {
          // Move pure, single-use input views only into an actual guarded
          // scope. Physical reads are subsequently materialized in this same
          // scope.
          for (mlir::Value input : {tile.getQuery(), tile.getKey(),
                                    tile.getValue(), tile.getMask()}) {
            if (!input)
              continue;
            auto slice = input.getDefiningOp<mlir::tensor::ExtractSliceOp>();
            if (slice && slice->hasOneUse() &&
                slice->getBlock() == tile->getBlock() &&
                rewriter.getInsertionBlock() != tile->getBlock())
              rewriter.moveOpBefore(slice, rewriter.getInsertionBlock(),
                                    rewriter.getInsertionPoint());
          }
          auto clone = [&](bool omitPositions) {
            mlir::IRMapping mapping;
            auto copy = mlir::cast<LinalgExtOnlineAttentionOp>(
                rewriter.clone(*tile, mapping));
            if (omitPositions) {
              copy.getPositionsMutable().clear();
              copy.setCausal(false);
              copy.removePositionMapAttr();
            }
            return Values(copy->getResults());
          };
          return choose(
              full, [&] { return clone(true); }, [&] { return clone(false); });
        },
        [&] {
          return Values{tile.getAccumulator(), tile.getMaximum(),
                        tile.getSum()};
        });
    rewriter.replaceOp(tile, selected);
  }
  return mlir::success();
}
} // namespace wafer
