//===- NumericPredicateFolding.cpp - Compose actual predicate writes
//-------===//

#include "Internal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::tile_region_to_instr {

void foldNumericPredicates(TileRegionOp region, MovementDescriptorCache &cache,
                           mlir::RewriterBase::Listener *listener) {
  llvm::SmallVector<InstrBit2FpOp> conversions;
  region.walk(
      [&](InstrBit2FpOp conversion) { conversions.push_back(conversion); });
  mlir::IRRewriter rewriter(region.getContext(), listener);
  for (auto conversion : conversions) {
    auto boolean = conversion.getSource();
    auto booleanAllocation = boolean.getDefiningOp<mlir::memref::AllocOp>();
    auto numeric = conversion.getDest();
    auto numericAllocation = numeric.getDefiningOp<mlir::memref::AllocOp>();
    if (!booleanAllocation || !numericAllocation ||
        numericAllocation->getBlock() != conversion->getBlock() ||
        !llvm::hasNItems(boolean.use_begin(), boolean.use_end(), 2) ||
        numericAllocation.getNumOperands() != 0)
      continue;
    mlir::Operation *producer = nullptr;
    for (auto *user : boolean.getUsers())
      if (user != conversion)
        producer = user;
    if (!producer || producer->getBlock() != conversion->getBlock() ||
        !producer->isBeforeInBlock(conversion))
      continue;
    auto comparison = mlir::dyn_cast<InstrElementwiseOp>(producer);
    auto fill = mlir::dyn_cast<InstrFillOp>(producer);
    auto numericType = mlir::cast<mlir::MemRefType>(numeric.getType());
    llvm::APInt constant;
    if (comparison) {
      auto inputType =
          mlir::cast<mlir::MemRefType>(comparison.getInputs()[0].getType());
      // For verifier-valid Instr, only relation or BOOL logic can write i1;
      // matching the floating destination type excludes the BOOL logic case.
      if (comparison.getDest() != boolean ||
          inputType.getElementType() != numericType.getElementType())
        continue;
    } else if (!fill || fill.getDest() != boolean ||
               !mlir::matchPattern(fill.getValue(),
                                   mlir::m_ConstantInt(&constant))) {
      continue;
    }
    if (!cache.hasOrProveIdentityPhysicalTraversal(
            mlir::cast<mlir::MemRefType>(boolean.getType()), numericType))
      continue;
    mlir::DominanceInfo dominance(region);
    if (!llvm::all_of(numeric.getUsers(), [&](mlir::Operation *user) {
          return user == conversion ||
                 dominance.properlyDominates(conversion, user);
        }))
      continue;

    // Write the numeric result at the original predicate's position, so an
    // intervening write to a comparison input cannot change its snapshot.
    // Only the actual, unobserved allocation moves; its typed owner remains.
    rewriter.moveOpBefore(numericAllocation, producer);
    if (comparison) {
      rewriter.modifyOpInPlace(
          comparison, [&] { comparison.getDestMutable().assign(numeric); });
    } else {
      rewriter.setInsertionPoint(fill);
      auto value = rewriter.create<mlir::arith::ConstantOp>(
          fill.getLoc(), rewriter.getFloatAttr(numericType.getElementType(),
                                               constant.isZero() ? 0.0 : 1.0));
      rewriter.modifyOpInPlace(fill, [&] {
        fill.getValueMutable().assign(value);
        fill.getDestMutable().assign(numeric);
      });
    }
    rewriter.eraseOp(conversion);
    rewriter.eraseOp(booleanAllocation);
  }
}

} // namespace wafer::tile_region_to_instr
