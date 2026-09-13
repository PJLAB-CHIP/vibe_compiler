//===- ExpandStablehloBatchNormInference.cpp - Inference normalization ----===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {

#define GEN_PASS_DEF_EXPANDSTABLEHLOBATCHNORMINFERENCEPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

#ifdef WAFER_ENABLE_STABLEHLO
struct ExpandBatchNormInference final
    : mlir::OpConversionPattern<mlir::stablehlo::BatchNormInferenceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::BatchNormInferenceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!resultType || !resultType.hasStaticShape() ||
        !mlir::isa<mlir::FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "requires static-ranked floating-point inference");
    for (mlir::Value value : adaptor.getOperands()) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "requires static operands");
    }

    auto elementType = mlir::cast<mlir::FloatType>(resultType.getElementType());
    auto featureType =
        mlir::cast<mlir::RankedTensorType>(adaptor.getVariance().getType());
    llvm::APFloat epsilon = op.getEpsilonAttr().getValue();
    bool losesInfo;
    epsilon.convert(elementType.getFloatSemantics(),
                    llvm::APFloat::rmNearestTiesToEven, &losesInfo);

    // Follow the StableHLO definition without the affine reassociation used by
    // XLA's batchnorm expander. Feature-only work commutes with broadcasting,
    // but the scalar arithmetic and every intermediate dtype stay unchanged.
    mlir::Location loc = op.getLoc();
    auto epsilonValue = rewriter.create<mlir::stablehlo::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(featureType, epsilon));
    auto variance = rewriter.create<mlir::stablehlo::AddOp>(
        loc, adaptor.getVariance(), epsilonValue);
    auto stddev = rewriter.create<mlir::stablehlo::SqrtOp>(loc, variance);
    auto dimensions = rewriter.getDenseI64ArrayAttr(
        {static_cast<int64_t>(op.getFeatureIndex())});
    auto broadcast = [&](mlir::Value value) -> mlir::Value {
      return rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
          loc, resultType, value, dimensions);
    };
    auto centered = rewriter.create<mlir::stablehlo::SubtractOp>(
        loc, adaptor.getOperand(), broadcast(adaptor.getMean()));
    auto normalized = rewriter.create<mlir::stablehlo::DivOp>(
        loc, centered, broadcast(stddev));
    auto scaled = rewriter.create<mlir::stablehlo::MulOp>(
        loc, broadcast(adaptor.getScale()), normalized);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::AddOp>(
        op, scaled, broadcast(adaptor.getOffset()));
    return mlir::success();
  }
};
#endif

struct ExpandStablehloBatchNormInferencePass final
    : impl::ExpandStablehloBatchNormInferencePassBase<
          ExpandStablehloBatchNormInferencePass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addIllegalOp<mlir::stablehlo::BatchNormInferenceOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ExpandBatchNormInference>(&getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))) ||
        mlir::failed(mlir::verify(getOperation())))
      signalPassFailure();
#endif
  }
};

} // namespace
} // namespace wafer
