//===- LowerStaticStablehloConcatenate.cpp - Preserve concat semantics ---===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace {

#ifdef WAFER_ENABLE_STABLEHLO
struct LowerStaticConcatenate final
    : mlir::OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::ConcatenateOp concatenate,
                  mlir::PatternRewriter &rewriter) const final {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(concatenate.getType());
    if (!resultType || !resultType.hasStaticShape() ||
        concatenate.getInputs().empty())
      return mlir::failure();
    const int64_t rank = resultType.getRank();
    const int64_t dimension = concatenate.getDimension();
    if (dimension < 0 || dimension >= rank)
      return mlir::failure();

    llvm::SmallVector<mlir::RankedTensorType, 4> inputTypes;
    int64_t concatenatedExtent = 0;
    for (mlir::Value input : concatenate.getInputs()) {
      auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      if (!inputType || !inputType.hasStaticShape() ||
          inputType.getRank() != rank ||
          inputType.getElementType() != resultType.getElementType())
        return mlir::failure();
      for (int64_t dim = 0; dim < rank; ++dim)
        if (dim != dimension &&
            inputType.getDimSize(dim) != resultType.getDimSize(dim))
          return mlir::failure();
      concatenatedExtent += inputType.getDimSize(dimension);
      inputTypes.push_back(inputType);
    }
    if (concatenatedExtent != resultType.getDimSize(dimension))
      return mlir::failure();

    mlir::Location loc = concatenate.getLoc();
    mlir::Value result =
        rewriter
            .create<mlir::tensor::EmptyOp>(loc, resultType.getShape(),
                                           resultType.getElementType())
            .getResult();
    llvm::SmallVector<mlir::OpFoldResult, 6> offsets(rank,
                                                     rewriter.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult, 6> strides(rank,
                                                     rewriter.getIndexAttr(1));
    int64_t nextOffset = 0;
    for (auto [input, inputType] :
         llvm::zip_equal(concatenate.getInputs(), inputTypes)) {
      llvm::SmallVector<mlir::OpFoldResult, 6> sizes;
      for (int64_t extent : inputType.getShape())
        sizes.push_back(rewriter.getIndexAttr(extent));
      offsets[dimension] = rewriter.getIndexAttr(nextOffset);
      result = rewriter
                   .create<mlir::tensor::InsertSliceOp>(loc, input, result,
                                                        offsets, sizes, strides)
                   .getResult();
      nextOffset += inputType.getDimSize(dimension);
    }
    rewriter.replaceOp(concatenate, result);
    return mlir::success();
  }
};
#endif

struct LowerStaticStablehloConcatenatePass final
    : mlir::PassWrapper<LowerStaticStablehloConcatenatePass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerStaticStablehloConcatenatePass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-static-stablehlo-concatenate";
  }

  llvm::StringRef getDescription() const final {
    return "Lower static StableHLO concatenate to canonical tensor insertion";
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::Operation *, 8> candidates;
    getOperation().walk([&](mlir::stablehlo::ConcatenateOp concatenate) {
      candidates.push_back(concatenate.getOperation());
    });
    if (candidates.empty())
      return;

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerStaticConcatenate>(&getContext());
    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    if (mlir::failed(
            mlir::applyOpPatternsAndFold(candidates, frozenPatterns, config)))
      signalPassFailure();
#endif
  }
};

} // namespace

std::unique_ptr<mlir::Pass> wafer::createLowerStaticStablehloConcatenatePass() {
  return std::make_unique<LowerStaticStablehloConcatenatePass>();
}
