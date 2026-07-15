//===- GroupLowering.cpp - wafer.group conversion pattern -------------===//

#include "Internal.h"

using namespace wafer;

namespace wafer::group_to_tile_region {
namespace {

struct GroupToTileRegionLoweringPattern
    : public mlir::OpConversionPattern<GroupOp> {
  GroupToTileRegionLoweringPattern(mlir::MLIRContext *context,
                                   std::string *failureReason,
                                   int64_t currentLogicalRank)
      : mlir::OpConversionPattern<GroupOp>(context),
        failureReason(failureReason), currentLogicalRank(currentLogicalRank) {}

  mlir::LogicalResult
  matchAndRewrite(GroupOp group, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    TileRegionBodyEmitter emitter(failureReason, currentLogicalRank);
    mlir::FailureOr<TileRegionOp> tileRegion =
        emitter.emit(group, adaptor.getInputs(), adaptor.getOuts(), rewriter);
    if (mlir::failed(tileRegion))
      return mlir::failure();

    rewriter.setInsertionPointAfter((*tileRegion).getOperation());
    llvm::SmallVector<mlir::Value, 2> replacements;
    for (mlir::Value result : (*tileRegion).getResults()) {
      auto tensor = rewriter.create<mlir::bufferization::ToTensorOp>(
          group.getLoc(), result, /*restrict=*/true, /*writeable=*/true);
      replacements.push_back(tensor.getResult());
    }

    rewriter.replaceOp(group, replacements);
    return mlir::success();
  }

  std::string *failureReason;
  int64_t currentLogicalRank = -1;
};

} // namespace

void populateGroupToTileRegionPatterns(mlir::RewritePatternSet &patterns,
                                       std::string *failureReason,
                                       int64_t currentLogicalRank) {
  patterns.add<GroupToTileRegionLoweringPattern>(
      patterns.getContext(), failureReason, currentLogicalRank);
}

} // namespace wafer::group_to_tile_region
