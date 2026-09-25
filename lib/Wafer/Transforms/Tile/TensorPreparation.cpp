//===- TensorPreparation.cpp - Expose final tensor subsets ----------------===//
#include "TensorPreparation.h"
#include "BooleanReduction.h"
#include "ElementwisePayloads.h"
#include "GatherLowering.h"
#include "LoopSubsetState.h"
#include "StructuredBufferRelations.h"
#include "TensorInitialization.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

namespace wafer::compiler::detail {
namespace {
// Delegate the pinned canonicalizations, keeping coordinate-bearing singleton
// loops intact while they still delimit explicit Tensor read selections.
struct PreserveIterationScope : mlir::OpRewritePattern<mlir::scf::ForOp> {
  explicit PreserveIterationScope(std::unique_ptr<mlir::RewritePattern> pattern)
      : OpRewritePattern(pattern->getContext(), pattern->getBenefit()),
        pattern(std::move(pattern)) {
    setHasBoundedRewriteRecursion(this->pattern->hasBoundedRewriteRecursion());
  }
  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    if (getIterationCoordinates(loop)) {
      auto lower = mlir::getConstantIntValue(loop.getLowerBound());
      auto upper = mlir::getConstantIntValue(loop.getUpperBound());
      auto step = mlir::getConstantIntValue(loop.getStep());
      int64_t extent = 0;
      if (lower && upper && step && *step > 0 &&
          !llvm::SubOverflow(*upper, *lower, extent) && extent > 0 &&
          extent <= *step)
        return rewriter.notifyMatchFailure(loop,
                                           "Tensor iteration scope is live");
    }
    return pattern->matchAndRewrite(loop, rewriter);
  }
  std::unique_ptr<mlir::RewritePattern> pattern;
};

static mlir::LogicalResult
localizeEmptySlices(mlir::ModuleOp module,
                    StructuredMaterializationRelations &relations,
                    mlir::RewriterBase::Listener *externalListener) {
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 16> slices;
  module.walk([&](mlir::tensor::ExtractSliceOp slice) {
    if (slice->getParentOfType<TileRegionOp>())
      slices.push_back(slice);
  });
  StructuredBufferReplacementListener listener(relations, externalListener);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  for (auto slice : slices) {
    auto empty = slice.getSource().getDefiningOp<mlir::tensor::EmptyOp>();
    if (!empty || !slice.getType().hasStaticShape())
      continue;
    // Empty tensors carry no contents; a selected slice needs only its own
    // destination shape. This is the pinned Tensor empty/slice fold.
    rewriter.setInsertionPoint(slice);
    rewriter.replaceOpWithNewOp<mlir::tensor::EmptyOp>(slice, slice.getType(),
                                                       mlir::ValueRange{});
    if (empty->use_empty())
      rewriter.eraseOp(empty);
  }
  return mlir::success(listener.finalizeAfterRewrite());
}

} // namespace

void preserveTensorIterationScopes(mlir::RewritePatternSet &patterns) {
  for (auto &pattern : patterns.getNativePatterns())
    if (pattern->getRootKind() ==
        mlir::OperationName(mlir::scf::ForOp::getOperationName(),
                            patterns.getContext()))
      pattern = mlir::RewritePattern::create<PreserveIterationScope>(
          std::move(pattern));
}

TensorPreparationResult
prepareCurrentTensorInput(mlir::ModuleOp module,
                          StructuredMaterializationRelations &relations,
                          mlir::RewriterBase::Listener *externalListener) {
  TensorPreparationResult result;
  if (!module || mlir::failed(mlir::verify(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.status = TensorPreparationStatus::BrokenContract;
    result.detail =
        "tensor preparation requires verified current IR and relations";
    return result;
  }
  if (module
          .walk([](mlir::Operation *op) {
            return mlir::isa<LinalgExtAttentionOp, LinalgExtOnlineAttentionOp>(
                       op)
                       ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
          })
          .wasInterrupted()) {
    result.status = TensorPreparationStatus::Unsupported;
    result.detail = "tensor preparation requires decomposed attention";
    return result;
  }
  // Make every uniform tensor initializer visible to the layout query,
  // including Generate from standard tiling of an all-padding window.
  {
    StructuredBufferReplacementListener listener(relations, externalListener);
    mlir::IRRewriter rewriter(module.getContext());
    rewriter.setListener(&listener);
    if (mlir::failed(lowerUniformTensorInitializers(rewriter, module))) {
      result.detail =
          "tensor initialization could not be materialized before layout";
      return result;
    }
    if (!listener.finalizeAfterRewrite()) {
      result.detail =
          "tensor initialization left stale current buffer relations";
      return result;
    }
  }
  if (module
          .walk([](mlir::Operation *op) {
            return mlir::isa<mlir::tensor::PadOp, mlir::tensor::GenerateOp>(op)
                       ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
          })
          .wasInterrupted()) {
    result.status = TensorPreparationStatus::Unsupported;
    result.detail = "layout input requires uniform tensor initialization";
    return result;
  }

  if (mlir::failed(lowerTensorGathers(module, relations, externalListener))) {
    result.detail = "selected gather row materialization failed";
    return result;
  }
  if (mlir::failed(
          lowerBooleanReductions(module, relations, externalListener))) {
    result.detail = "boolean reduction representation failed";
    return result;
  }
  if (mlir::failed(
          normalizeLoopSubsetState(module, relations, externalListener))) {
    result.detail = "loop subset state normalization failed";
    return result;
  }
  if (mlir::failed(localizeEmptySlices(module, relations, externalListener))) {
    result.detail = "empty subset normalization left stale relations";
    return result;
  }
  if (mlir::failed(materializeElementwisePayloads(module, relations,
                                                  externalListener))) {
    result.detail = "pointwise payload materialization failed";
    return result;
  }

  if (mlir::failed(mlir::verify(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.detail = "tensor preparation produced invalid IR or stale relations";
    return result;
  }
  result.status = TensorPreparationStatus::Success;
  return result;
}
} // namespace wafer::compiler::detail

namespace wafer {
#define GEN_PASS_DEF_PREPARECURRENTTENSORINPUTPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"
namespace {
struct PrepareCurrentTensorInputPass
    : public impl::PrepareCurrentTensorInputPassBase<
          PrepareCurrentTensorInputPass> {
  void runOnOperation() final {
    StructuredMaterializationRelations relations;
    auto result =
        compiler::detail::prepareCurrentTensorInput(getOperation(), relations);
    if (!result.succeeded()) {
      getOperation().emitError() << result.detail;
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace wafer
