//===- PeerLowering.cpp - Physical peer communication lowering ----------===//

#include "Internal.h"

#include "mlir/Dialect/Async/IR/Async.h"

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

template <typename PeerOp, typename InstrOp>
class PeerLowering final : public mlir::OpRewritePattern<PeerOp> {
public:
  using mlir::OpRewritePattern<PeerOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(PeerOp op, mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto lowered = rewriter.create<InstrOp>(
        op.getLoc(), rewriter.getType<mlir::async::TokenType>(), op.getBuffer(),
        mlir::Value(), op.getPeerAttr(), op.getBytesAttr(), op.getMessageAttr(),
        DirectDTEBindingAttr());
    rewriter.replaceOp(op, lowered.getToken());
    return mlir::success();
  }
};

class PeerAwaitLowering final
    : public mlir::OpRewritePattern<mlir::async::AwaitOp> {
public:
  using mlir::OpRewritePattern<mlir::async::AwaitOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::async::AwaitOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    mlir::Value token = op.getOperand();
    if (!mlir::isa<mlir::async::TokenType>(token.getType()) ||
        (!token.getDefiningOp<CommPeerSendOp>() &&
         !token.getDefiningOp<CommPeerRecvOp>() &&
         !token.getDefiningOp<InstrDTESendOp>() &&
         !token.getDefiningOp<InstrDTERecvOp>()))
      return mlir::failure();
    rewriter.create<InstrDTEWaitOp>(op.getLoc(), mlir::ValueRange{token});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

void wafer::tile_region_to_instr::populatePeerLoweringPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<PeerLowering<CommPeerSendOp, InstrDTESendOp>,
               PeerLowering<CommPeerRecvOp, InstrDTERecvOp>, PeerAwaitLowering>(
      patterns.getContext());
}
