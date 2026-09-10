//===- TiledOutputStores.cpp - Forward collected tiles to DDR ------------===//

#include "TiledOutputStores.h"

#include "BoundaryMovement.h"
#include "StructuredBufferRelations.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer::compiler::detail {
namespace {

bool isSameSubview(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return true;
  auto left = lhs.getDefiningOp<mlir::memref::SubViewOp>();
  auto right = rhs.getDefiningOp<mlir::memref::SubViewOp>();
  return left && right && left.getSource() == right.getSource() &&
         left.getType() == right.getType() &&
         left.getMixedOffsets() == right.getMixedOffsets() &&
         left.getMixedSizes() == right.getMixedSizes() &&
         left.getMixedStrides() == right.getMixedStrides();
}

// Follow only exact SCF identity forwarding. A nested loop result can be
// the same buffer as an outer iter_arg even though the SSA values differ.
bool forwardsArgument(mlir::Value value, mlir::BlockArgument argument) {
  if (value == argument)
    return true;
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  auto loop = result ? mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner())
                     : mlir::scf::ForOp{};
  if (!loop)
    return false;
  unsigned index = result.getResultNumber();
  auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  return forwardsArgument(yield.getOperand(index),
                          loop.getRegionIterArg(index)) &&
         forwardsArgument(loop.getInitArgs()[index], argument);
}

struct OutputWrites {
  mlir::memref::AllocOp allocation;
  llvm::SmallVector<mlir::Value, 16> aliases;
  llvm::SmallVector<MoveCopyIntoOp, 8> writes;
  llvm::SmallVector<mlir::memref::CopyOp, 8> identityCopies;
};

// This is a proof over existing buffers and effects, not an output inventory.
// Any read, changed loop state or escape leaves the current storage unchanged.
std::optional<OutputWrites> collectOutputWrites(StorageStoreOp terminal) {
  auto region = terminal->getParentOfType<TileRegionOp>();
  auto destination = mlir::dyn_cast<mlir::BlockArgument>(terminal.getDest());
  if (!region || terminal->getBlock() != &region.getBody().front() ||
      !destination || destination.getOwner() != &region.getBody().front() ||
      !destination.hasOneUse())
    return std::nullopt;
  StorageRootMemo roots;
  const auto &sourceRoots = roots.getStorageRoots(terminal.getSource());
  const auto &destinationRoots = roots.getStorageRoots(terminal.getDest());
  if (sourceRoots.size() != 1 || destinationRoots.size() != 1 ||
      !destinationRoots.begin()->getDefiningOp<mlir::memref::AllocOp>())
    return std::nullopt;
  // The region argument dominates every replacement write. A distinct alias
  // used anywhere in this region could observe an earlier DDR write, even if
  // the terminal destination itself has only one use.
  bool destinationAliased = false;
  region.walk([&](mlir::Operation *operation) {
    if (operation == region.getOperation())
      return;
    for (mlir::Value operand : operation->getOperands()) {
      if (operand == destination ||
          !mlir::isa<mlir::BaseMemRefType>(operand.getType()))
        continue;
      if (roots.getStorageRoots(operand).contains(*destinationRoots.begin()))
        destinationAliased = true;
    }
  });
  if (destinationAliased)
    return std::nullopt;
  OutputWrites result;
  result.allocation =
      sourceRoots.begin()->getDefiningOp<mlir::memref::AllocOp>();
  if (!result.allocation ||
      result.allocation->getBlock() != terminal->getBlock() ||
      result.allocation.getType().getShape() !=
          mlir::cast<mlir::MemRefType>(terminal.getSource().getType())
              .getShape())
    return std::nullopt;
  mlir::Value wholeSource = terminal.getSource();
  while (auto loop = wholeSource.getDefiningOp<mlir::scf::ForOp>())
    wholeSource = loop.getInitArgs()[mlir::cast<mlir::OpResult>(wholeSource)
                                         .getResultNumber()];
  if (wholeSource != result.allocation.getResult())
    return std::nullopt;
  llvm::DenseSet<mlir::Value> seen;
  auto append = [&](mlir::Value value) {
    if (seen.insert(value).second)
      result.aliases.push_back(value);
  };
  append(result.allocation.getResult());
  llvm::DenseSet<mlir::Operation *> seenCopies;
  for (size_t index = 0; index < result.aliases.size(); ++index) {
    mlir::Value value = result.aliases[index];
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == terminal && use.getOperandNumber() == 0)
        continue;
      if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(user)) {
        if (use.getOperandNumber() != 0)
          return std::nullopt;
        append(subview.getResult());
        continue;
      }
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(user)) {
        if (use.getOperandNumber() < loop.getNumControlOperands())
          return std::nullopt;
        unsigned argument =
            use.getOperandNumber() - loop.getNumControlOperands();
        auto yield =
            mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        mlir::BlockArgument iterArg = loop.getRegionIterArg(argument);
        if (!forwardsArgument(yield.getOperand(argument), iterArg))
          return std::nullopt;
        append(iterArg);
        append(loop.getResult(argument));
        continue;
      }
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(user)) {
        auto loop = mlir::dyn_cast<mlir::scf::ForOp>(yield->getParentOp());
        if (!loop || !forwardsArgument(
                         value, loop.getRegionIterArg(use.getOperandNumber())))
          return std::nullopt;
        continue;
      }
      if (auto write = mlir::dyn_cast<MoveCopyIntoOp>(user)) {
        if (use.getOperandNumber() != 1)
          return std::nullopt;
        mlir::Operation *beforeStore = write;
        while (beforeStore && beforeStore->getBlock() != terminal->getBlock())
          beforeStore = beforeStore->getParentOp();
        if (!beforeStore || !beforeStore->isBeforeInBlock(terminal))
          return std::nullopt;
        result.writes.push_back(write);
        continue;
      }
      if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user)) {
        if (!isSameSubview(copy.getSource(), copy.getTarget()))
          return std::nullopt;
        if (seenCopies.insert(user).second)
          result.identityCopies.push_back(copy);
        continue;
      }
      return std::nullopt;
    }
  }
  if (result.writes.empty())
    return std::nullopt;
  return result;
}

void applyOutputWrites(StorageStoreOp terminal, const OutputWrites &writes,
                       mlir::IRRewriter &rewriter,
                       BoundaryMovementStatistics &statistics) {
  for (mlir::memref::CopyOp copy : writes.identityCopies)
    rewriter.eraseOp(copy);
  rewriter.replaceAllUsesWith(writes.allocation->getResult(0),
                              terminal.getDest());
  // The worklist follows actual alias edges, so each parent type is available
  // before its subviews and nested loop carriers are updated.
  for (mlir::Value value : llvm::drop_begin(writes.aliases)) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto loop =
          mlir::cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
      unsigned index = argument.getArgNumber() - 1;
      rewriter.modifyOpInPlace(loop, [&] {
        argument.setType(loop.getInitArgs()[index].getType());
        loop.getResult(index).setType(argument.getType());
      });
    } else if (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      auto type = mlir::memref::SubViewOp::inferRankReducedResultType(
          subview.getType().getShape(), subview.getSourceType(),
          subview.getMixedOffsets(), subview.getMixedSizes(),
          subview.getMixedStrides());
      rewriter.modifyOpInPlace(subview, [&] { value.setType(type); });
    }
  }
  for (MoveCopyIntoOp copy : writes.writes) {
    rewriter.setInsertionPoint(copy);
    rewriter.create<StorageStoreOp>(copy.getLoc(), copy.getSource(),
                                    copy.getDest());
    rewriter.eraseOp(copy);
  }
  rewriter.eraseOp(terminal);
  rewriter.eraseOp(writes.allocation);
  statistics.ddrStores += writes.writes.size() - 1;
  ++statistics.streamedOutputCarriers;
}

} // namespace

void materializeTiledOutputStores(mlir::ModuleOp module,
                                  BoundaryMovementStatistics &statistics) {
  llvm::SmallVector<StorageStoreOp, 16> stores;
  module.walk([&](StorageStoreOp store) { stores.push_back(store); });
  mlir::IRRewriter rewriter(module.getContext());
  for (StorageStoreOp store : stores)
    if (auto writes = collectOutputWrites(store))
      applyOutputWrites(store, *writes, rewriter, statistics);
}

} // namespace wafer::compiler::detail
