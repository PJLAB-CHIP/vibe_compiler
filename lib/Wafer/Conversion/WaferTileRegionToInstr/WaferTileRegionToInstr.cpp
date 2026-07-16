//===- WaferTileRegionToInstr.cpp - Tile-region to instr conversion ------===//

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "Internal.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <limits>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace wafer {
#define GEN_PASS_DEF_CONVERTTILEREGIONTOINSTRPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

namespace {

static void configureTileRegionToInstrTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::ModuleOp, TileRegionOp, TileYieldOp, SyncLocalFenceOp,
                    InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrFillOp,
                    InstrElementwiseOp, InstrBit2FpOp, InstrMaskMoveOp,
                    InstrReduceOp, InstrConvertOp, InstrGemmOp, InstrDTESendOp,
                    InstrDTERecvOp, InstrDTEWaitOp>();
  target.addDynamicallyLegalOp<InstrTDMADataMoveOp>([](InstrTDMADataMoveOp op) {
    return !requiresGatherScatterMaterialization(op.getKindAttr().getValue());
  });
  target.addIllegalOp<StorageLoadOp, StorageStoreOp, LayoutMaterializeOp,
                      ComputeFillOp, ComputeConvertOp, ComputeGemmOp,
                      ComputeElementwiseOp, ComputeReduceOp, MoveCopyOp,
                      MoveExtractSliceOp, MoveInsertSliceOp, MoveTransposeOp,
                      MoveBroadcastOp, ViewReshapeOp, CommAllGatherOp,
                      CommReduceScatterOp, CommAllReduceOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

static void
populateTileRegionToInstrPatterns(mlir::RewritePatternSet &patterns,
                                  const TileRegionToInstrOptions &options,
                                  std::string *failureReason) {
  populateMovementLoweringPatterns(patterns, failureReason);
  populateComputeLoweringPatterns(patterns, failureReason);
  populateViewReshapeLoweringPattern(patterns, failureReason);
  populateFillLoweringPattern(patterns);
  populateCollectiveLoweringPatterns(patterns, options, failureReason);
}

/// Remove a private fill whose destination has no reader.  Constant folding
/// of tensor-level select expressions can make the predicate buffer dead
/// after the tile body has already materialized its splat.  Keeping that
/// write would turn a dead i1 value into a real target TDMA command, where the
/// hardware profile correctly rejects the unproven BOOL encoding.
static void eraseDeadPrivateFills(mlir::ModuleOp module) {
  llvm::SmallVector<InstrFillOp, 4> deadFills;
  module.walk([&](InstrFillOp fill) {
    mlir::Value dest = fill.getDest();
    if (dest.hasOneUse() && dest.getDefiningOp<mlir::memref::AllocOp>())
      deadFills.push_back(fill);
  });
  for (InstrFillOp fill : deadFills) {
    mlir::Value dest = fill.getDest();
    mlir::Value scalar = fill.getValue();
    auto alloc = dest.getDefiningOp<mlir::memref::AllocOp>();
    fill->erase();
    alloc->erase();
    if (auto constant = scalar.getDefiningOp<mlir::arith::ConstantOp>();
        constant && constant->use_empty())
      constant->erase();
  }
}

static mlir::LogicalResult parseTileRegionToInstrOptions(
    llvm::StringRef allGatherSchedule, llvm::StringRef allReduceSchedule,
    llvm::StringRef reduceScatterSchedule, TileRegionToInstrOptions &options,
    std::string *failureReason) {
  if (allGatherSchedule == "auto" || allGatherSchedule == "ring") {
    options.allGatherSchedule = AllGatherSchedule::Ring;
  } else if (allGatherSchedule == "direct") {
    options.allGatherSchedule = AllGatherSchedule::Direct;
  } else {
    setFailureReason(failureReason,
                     llvm::Twine("unsupported all_gather schedule: ")
                         .concat(allGatherSchedule)
                         .str());
    return mlir::failure();
  }

  if (allReduceSchedule == "auto" || allReduceSchedule == "ring") {
    options.allReduceSchedule = AllReduceSchedule::Ring;
  } else if (allReduceSchedule == "tree") {
    options.allReduceSchedule = AllReduceSchedule::Tree;
  } else {
    setFailureReason(failureReason,
                     llvm::Twine("unsupported all_reduce schedule: ")
                         .concat(allReduceSchedule)
                         .str());
    return mlir::failure();
  }

  if (reduceScatterSchedule == "auto" || reduceScatterSchedule == "direct") {
    options.reduceScatterSchedule = ReduceScatterSchedule::Direct;
  } else {
    setFailureReason(failureReason,
                     llvm::Twine("unsupported reduce_scatter schedule: ")
                         .concat(reduceScatterSchedule)
                         .str());
    return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult
materializeStructuredLocalFences(mlir::ModuleOp module) {
  mlir::WalkResult result = module.walk([&](TileRegionOp tileRegion) {
    if (!tileRegion.getBody().hasOneBlock()) {
      tileRegion.emitError()
          << "instruction_completion_failure: terminal local completion "
             "requires a single-block wafer.tile.region";
      return mlir::WalkResult::interrupt();
    }

    mlir::WalkResult loopResult = tileRegion.walk([&](mlir::scf::ForOp forOp) {
      mlir::Operation *terminator = forOp.getBody()->getTerminator();
      if (!terminator) {
        forOp.emitError() << "instruction_completion_failure: scf.for has no "
                             "terminator for loop-backedge local completion";
        return mlir::WalkResult::interrupt();
      }
      if (mlir::isa_and_nonnull<SyncLocalFenceOp>(terminator->getPrevNode()))
        return mlir::WalkResult::advance();

      mlir::OpBuilder builder(terminator);
      builder.create<SyncLocalFenceOp>(terminator->getLoc());
      return mlir::WalkResult::advance();
    });
    if (loopResult.wasInterrupted())
      return mlir::WalkResult::interrupt();

    mlir::Block &body = tileRegion.getBody().front();
    mlir::Operation *terminator = body.getTerminator();
    if (!terminator) {
      tileRegion.emitError()
          << "instruction_completion_failure: wafer.tile.region has no "
             "terminator for terminal local completion";
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa_and_nonnull<SyncLocalFenceOp>(terminator->getPrevNode()))
      return mlir::WalkResult::advance();

    mlir::OpBuilder builder(terminator);
    builder.create<SyncLocalFenceOp>(terminator->getLoc());
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

struct ConvertTileRegionToInstrPass
    : public wafer::impl::ConvertTileRegionToInstrPassBase<
          ConvertTileRegionToInstrPass> {
  using wafer::impl::ConvertTileRegionToInstrPassBase<
      ConvertTileRegionToInstrPass>::ConvertTileRegionToInstrPassBase;

  void runOnOperation() final {
    std::string failureReason;
    TileRegionToInstrOptions options;
    if (mlir::failed(parseTileRegionToInstrOptions(
            allGatherSchedule, allReduceSchedule, reduceScatterSchedule,
            options, &failureReason))) {
      getOperation().emitError(failureReason);
      signalPassFailure();
      return;
    }

    if (mlir::succeeded(wafer::convertTileRegionToInstrModule(
            getOperation(), options, &failureReason)))
      return;

    if (!failureReason.empty())
      getOperation().emitError(failureReason);
    else
      getOperation().emitError("tile-region to instruction conversion failed");
    signalPassFailure();
  }
};

} // namespace

wafer::detail::StaticTerminalOperationBudgetStatus
wafer::detail::checkStaticTerminalOperationBudget(mlir::Operation *root,
                                                  uint64_t &operationCount) {
  operationCount = 0;
  bool overflow = false;
  root->walk([&](mlir::Operation *operation) {
    if (overflow ||
        !mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp>(operation))
      return;
    if (operationCount == std::numeric_limits<uint64_t>::max()) {
      overflow = true;
      return;
    }
    ++operationCount;
  });
  if (overflow)
    return StaticTerminalOperationBudgetStatus::CountOverflow;
  if (operationCount > kStaticTerminalOperationBudget)
    return StaticTerminalOperationBudgetStatus::BudgetExceeded;
  return StaticTerminalOperationBudgetStatus::WithinBudget;
}

mlir::LogicalResult
wafer::convertTileRegionToInstrModule(mlir::ModuleOp module,
                                      std::string *failureReason) {
  return wafer::convertTileRegionToInstrModule(
      module, TileRegionToInstrOptions{}, failureReason);
}

mlir::LogicalResult
wafer::convertTileRegionToInstrModule(mlir::ModuleOp module,
                                      const TileRegionToInstrOptions &options,
                                      std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  mlir::MLIRContext *context = module.getContext();
  llvm::SmallVector<mlir::Operation *, 4> selectCandidates;
  module.walk([&](ComputeElementwiseOp op) {
    if (op.getKind() == ComputeElementwiseKind::Select)
      selectCandidates.push_back(op);
  });
  if (!selectCandidates.empty()) {
    mlir::RewritePatternSet canonicalizationPatterns(context);
    populateConstantPredicateSelectCanonicalizationPattern(
        canonicalizationPatterns);
    mlir::FrozenRewritePatternSet frozenPatterns(
        std::move(canonicalizationPatterns));
    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    if (mlir::failed(mlir::applyOpPatternsAndFold(selectCandidates,
                                                  frozenPatterns, config))) {
      setFailureReason(
          failureReason,
          "tile constant-predicate select canonicalization failed");
      return mlir::failure();
    }
  }

  mlir::ConversionTarget target(*context);
  configureTileRegionToInstrTarget(target);

  mlir::RewritePatternSet patterns(context);
  populateTileRegionToInstrPatterns(patterns, options, failureReason);

  bool conversionSucceeded = false;
  {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(
        mlir::applyFullConversion(module, target, std::move(patterns)));
  }

  if (!conversionSucceeded) {
    if (!failureReason || failureReason->empty())
      setFailureReason(failureReason,
                       "tile-region to instruction conversion failed");
    return mlir::failure();
  }
  eraseDeadPrivateFills(module);
  return materializeStructuredLocalFences(module);
}
