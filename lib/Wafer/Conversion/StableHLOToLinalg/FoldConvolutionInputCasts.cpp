//===- FoldConvolutionInputCasts.cpp - Preserve low-precision storage
//------===//

#include "Wafer/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
#define GEN_PASS_DEF_FOLDCONVOLUTIONINPUTCASTSPASS
#include "Wafer/Conversion/WaferConversionPasses.h.inc"
namespace {

struct WidenedInput {
  mlir::Value source;
  mlir::tensor::PadOp pad;
  mlir::linalg::GenericOp extension;
};

static WidenedInput getWidenedInput(mlir::Value value) {
  auto pad = value.getDefiningOp<mlir::tensor::PadOp>();
  if (pad) {
    auto padding = pad.getConstantPaddingValue();
    auto constant = padding ? padding.getDefiningOp<mlir::arith::ConstantOp>()
                            : mlir::arith::ConstantOp{};
    auto number = constant
                      ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())
                      : mlir::FloatAttr{};
    if (!number || !number.getValue().isZero())
      return {};
    value = pad.getSource();
  }
  auto generic = value.getDefiningOp<mlir::linalg::GenericOp>();
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 ||
      !generic.getMatchingIndexingMap(generic.getDpsInputOperand(0))
           .isIdentity() ||
      !generic.getMatchingIndexingMap(generic.getDpsInitOperand(0))
           .isIdentity() ||
      generic.getNumReductionLoops() != 0)
    return {};
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
      generic.getDpsInputs()[0].getType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!sourceType || !resultType ||
      sourceType.getShape() != resultType.getShape())
    return {};
  mlir::Block &body = generic.getRegion().front();
  if (body.getOperations().size() != 2)
    return {};
  auto extension = mlir::dyn_cast<mlir::arith::ExtFOp>(body.front());
  auto yield = mlir::cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!extension || extension.getIn() != body.getArgument(0) ||
      yield.getValues().front() != extension.getResult() ||
      !extension.getType().isF32() ||
      !(extension.getIn().getType().isF16() ||
        extension.getIn().getType().isBF16()))
    return {};
  return {generic.getDpsInputs()[0], pad, generic};
}

struct FoldConvolutionInputCastsPass final
    : impl::FoldConvolutionInputCastsPassBase<FoldConvolutionInputCastsPass> {
  void runOnOperation() final {
    llvm::SmallVector<mlir::linalg::GenericOp> candidates;
    getOperation().walk([&](mlir::linalg::GenericOp op) {
      if (mlir::succeeded(mlir::linalg::inferConvolutionDims(op)))
        candidates.push_back(op);
    });
    mlir::IRRewriter rewriter(&getContext());
    for (auto op : candidates) {
      if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
          !op.hasPureTensorSemantics())
        continue;
      WidenedInput inputs[] = {getWidenedInput(op.getDpsInputs()[0]),
                               getWidenedInput(op.getDpsInputs()[1])};
      if (!inputs[0].source || !inputs[1].source)
        continue;
      auto elementType =
          mlir::cast<mlir::RankedTensorType>(inputs[0].source.getType())
              .getElementType();
      if (mlir::cast<mlir::RankedTensorType>(inputs[1].source.getType())
              .getElementType() != elementType)
        continue;
      rewriter.setInsertionPoint(op);
      mlir::IRMapping mapping;
      for (unsigned index = 0; index < 2; ++index) {
        mlir::Value source = inputs[index].source;
        if (auto pad = inputs[index].pad) {
          auto type = pad.getType().clone(elementType);
          auto padding = mlir::cast<mlir::FloatAttr>(
              pad.getConstantPaddingValue()
                  .getDefiningOp<mlir::arith::ConstantOp>()
                  .getValue());
          auto zero = rewriter.create<mlir::arith::ConstantOp>(
              pad.getLoc(),
              rewriter.getFloatAttr(
                  elementType, padding.getValue().isNegative() ? -0.0 : 0.0));
          auto replacement = rewriter.create<mlir::tensor::PadOp>(
              pad.getLoc(), type, source, pad.getStaticLow(),
              pad.getStaticHigh(), pad.getLow(), pad.getHigh(),
              pad.getNofold());
          mlir::Block *block = rewriter.createBlock(&replacement.getRegion());
          for (int64_t dim = 0; dim < type.getRank(); ++dim)
            block->addArgument(rewriter.getIndexType(), pad.getLoc());
          rewriter.create<mlir::tensor::YieldOp>(pad.getLoc(),
                                                 zero.getResult());
          source = replacement.getResult();
          rewriter.setInsertionPoint(op);
        }
        mapping.map(op.getDpsInputs()[index], source);
      }
      auto replacement =
          mlir::cast<mlir::linalg::GenericOp>(rewriter.clone(*op, mapping));
      mlir::Block &body = replacement.getRegion().front();
      rewriter.setInsertionPointToStart(&body);
      for (unsigned index = 0; index < 2; ++index) {
        auto argument = body.getArgument(index);
        auto computeType = argument.getType();
        argument.setType(elementType);
        auto extension = rewriter.create<mlir::arith::ExtFOp>(
            op.getLoc(), computeType, argument);
        rewriter.replaceAllUsesExcept(argument, extension.getResult(),
                                      extension);
      }
      rewriter.replaceOp(op, replacement.getResults());
      llvm::SmallPtrSet<mlir::Operation *, 4> visited;
      for (const auto &input : inputs)
        if (input.pad && visited.insert(input.pad).second &&
            input.pad->use_empty())
          rewriter.eraseOp(input.pad);
      for (auto &input : inputs) {
        if (!visited.insert(input.extension).second ||
            !input.extension->use_empty())
          continue;
        auto empty = input.extension.getDpsInits()[0]
                         .getDefiningOp<mlir::tensor::EmptyOp>();
        rewriter.eraseOp(input.extension);
        if (empty && empty->use_empty())
          rewriter.eraseOp(empty);
      }
    }
  }
};
} // namespace
} // namespace wafer
