//===- TileToInstr.cpp - Tile-region to Instr conversion ---------------===//

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"

#include "Internal.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <memory>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace wafer {
#define GEN_PASS_DEF_CONVERTTILEREGIONTOINSTRPASS
#define GEN_PASS_DEF_CONVERTBUFFERIZATIONCOPIESTOINSTRPASS
#include "Wafer/Conversion/WaferConversionPasses.h.inc"
} // namespace wafer

namespace {

static void configureTileRegionToInstrTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect>();
  target.addIllegalOp<mlir::memref::CopyOp>();
  target.addLegalOp<mlir::ModuleOp, TileRegionOp, TileYieldOp, SyncNCCJoinOp,
                    InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrFillOp,
                    InstrElementwiseOp, InstrBit2FpOp, InstrMaskMoveOp,
                    InstrReduceOp, InstrConvertOp, InstrGemmOp, InstrConvOp,
                    InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>();
  target.addDynamicallyLegalOp<InstrTDMADataMoveOp>([](InstrTDMADataMoveOp op) {
    return !requiresGatherScatterMaterialization(op.getKindAttr().getValue());
  });
  target.addDynamicallyLegalOp<mlir::async::AwaitOp>(
      [](mlir::async::AwaitOp op) {
        mlir::Value operand = op.getOperand();
        return !operand.getDefiningOp<CommPeerSendOp>() &&
               !operand.getDefiningOp<CommPeerRecvOp>() &&
               !operand.getDefiningOp<InstrDTESendOp>() &&
               !operand.getDefiningOp<InstrDTERecvOp>();
      });
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *operation) {
    return !mlir::isa<WaferTileDataflowOpInterface>(operation);
  });
}

static void populateTileRegionToInstrPatterns(
    mlir::RewritePatternSet &patterns,
    TileRegionToInstrBufferRecorder *bufferRecorder,
    MovementDescriptorCache *descriptorCache) {
  populateMovementLoweringPatterns(patterns, bufferRecorder, descriptorCache);
  populateComputeLoweringPatterns(patterns, bufferRecorder, descriptorCache);
  populateViewReshapeLoweringPattern(patterns);
  populateFillLoweringPattern(patterns);
  populatePeerLoweringPatterns(patterns);
}

static void eraseDeadPrivateFills(mlir::Operation *root,
                                  mlir::RewriterBase::Listener *listener) {
  llvm::SmallVector<InstrFillOp, 4> deadFills;
  root->walk([&](InstrFillOp fill) {
    mlir::Value dest = fill.getDest();
    if (dest.hasOneUse() && dest.getDefiningOp<mlir::memref::AllocOp>())
      deadFills.push_back(fill);
  });
  mlir::IRRewriter rewriter(root->getContext(), listener);
  for (InstrFillOp fill : deadFills) {
    mlir::Value dest = fill.getDest();
    mlir::Value scalar = fill.getValue();
    auto alloc = dest.getDefiningOp<mlir::memref::AllocOp>();
    rewriter.eraseOp(fill);
    rewriter.eraseOp(alloc);
    if (auto constant = scalar.getDefiningOp<mlir::arith::ConstantOp>();
        constant && constant->use_empty())
      rewriter.eraseOp(constant);
  }

  llvm::SmallVector<mlir::arith::ConstantOp, 4> deadConstants;
  root->walk([&](mlir::arith::ConstantOp constant) {
    if (constant->use_empty())
      deadConstants.push_back(constant);
  });
  for (mlir::arith::ConstantOp constant : deadConstants)
    rewriter.eraseOp(constant);
}

struct ConvertTileRegionToInstrPass
    : public wafer::impl::ConvertTileRegionToInstrPassBase<
          ConvertTileRegionToInstrPass> {
  using wafer::impl::ConvertTileRegionToInstrPassBase<
      ConvertTileRegionToInstrPass>::ConvertTileRegionToInstrPassBase;

  void runOnOperation() final {
    unsigned sourceOperationCount = 0;
    getOperation().walk(
        [&](WaferTileDataflowOpInterface) { ++sourceOperationCount; });
    TileRegionToInstrLoweringSession session(*getOperation().getContext());
    if (mlir::succeeded(
            wafer::convertTileRegionToInstr(getOperation(), session))) {
      numDataflowOperationsLowered += sourceOperationCount;
      return;
    }
    getOperation().emitError("tile-region to instruction conversion failed");
    signalPassFailure();
  }
};

struct ConvertBufferizationCopiesToInstrPass
    : public wafer::impl::ConvertBufferizationCopiesToInstrPassBase<
          ConvertBufferizationCopiesToInstrPass> {
  using wafer::impl::ConvertBufferizationCopiesToInstrPassBase<
      ConvertBufferizationCopiesToInstrPass>::
      ConvertBufferizationCopiesToInstrPassBase;

  void runOnOperation() final {
    TileRegionToInstrLoweringSession session(*getOperation().getContext());
    if (mlir::succeeded(
            wafer::convertBufferizationCopiesToInstr(getOperation(), session)))
      return;
    getOperation().emitError(
        "bufferization copy to instruction conversion failed");
    signalPassFailure();
  }
};

} // namespace

struct wafer::TileRegionToInstrLoweringSession::Impl {
  explicit Impl(mlir::MLIRContext &context,
                TileRegionToInstrBufferRecorder *bufferRecorder)
      : target(context) {
    configureTileRegionToInstrTarget(target);
    mlir::RewritePatternSet lowering(&context);
    populateTileRegionToInstrPatterns(lowering, bufferRecorder,
                                      &descriptorCache);
    loweringPatterns = mlir::FrozenRewritePatternSet(std::move(lowering));
  }

  MovementDescriptorCache descriptorCache;
  mlir::ConversionTarget target;
  mlir::FrozenRewritePatternSet loweringPatterns;
};

wafer::TileRegionToInstrLoweringSession::TileRegionToInstrLoweringSession(
    mlir::MLIRContext &context, TileRegionToInstrBufferRecorder *bufferRecorder)
    : impl(std::make_unique<Impl>(context, bufferRecorder)) {}

wafer::TileRegionToInstrLoweringSession::~TileRegionToInstrLoweringSession() =
    default;

wafer::detail::StaticExecutableOperationCountStatus
wafer::detail::countStaticExecutableOperations(mlir::Operation *root,
                                               uint64_t &operationCount) {
  operationCount = 0;
  bool overflow = false;
  root->walk([&](mlir::Operation *operation) {
    if (overflow ||
        !mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(operation))
      return;
    if (operationCount == std::numeric_limits<uint64_t>::max()) {
      overflow = true;
      return;
    }
    ++operationCount;
  });
  return overflow ? StaticExecutableOperationCountStatus::CountOverflow
                  : StaticExecutableOperationCountStatus::Counted;
}

mlir::LogicalResult wafer::convertTileRegionToInstr(
    TileRegionOp region, TileRegionToInstrLoweringSession &session,
    mlir::RewriterBase::Listener *listener) {
  if (!region)
    return mlir::failure();
  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::TileToInstructionLowering);
  wafer::support::ScopedCompileTimingSpan conversionTiming(
      "conversion", "tile-region-to-instr", "region-conversion");
  bool conversionSucceeded = false;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr", "full-conversion");
    mlir::ConversionConfig config;
    config.listener = listener;
    conversionSucceeded = mlir::succeeded(mlir::applyFullConversion(
        region.getOperation(), session.impl->target,
        session.impl->loweringPatterns, config));
    if (!conversionSucceeded)
      timing.markFailed();
  }
  if (!conversionSucceeded) {
    conversionTiming.markFailed();
    return mlir::failure();
  }
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering-phase", "tile-region-to-instr", "dead-private-fill-erasure");
    eraseDeadPrivateFills(region.getOperation(), listener);
  }
  return mlir::success();
}

mlir::LogicalResult wafer::convertBufferizationCopiesToInstr(
    mlir::ModuleOp module, TileRegionToInstrLoweringSession &session,
    mlir::RewriterBase::Listener *listener) {
  if (!module)
    return mlir::failure();
  mlir::ConversionConfig config;
  config.listener = listener;
  return mlir::applyPartialConversion(module, session.impl->target,
                                      session.impl->loweringPatterns, config);
}
