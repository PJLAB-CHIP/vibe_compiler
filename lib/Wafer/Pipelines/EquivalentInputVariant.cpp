//===- EquivalentInputVariant.cpp - Qualification input construction ----===//

#include "EquivalentInputVariantInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace wafer::qualification_internal {
namespace {

class EquivalentInputVariantPass final
    : public mlir::PassWrapper<EquivalentInputVariantPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EquivalentInputVariantPass)

  explicit EquivalentInputVariantPass(EquivalentInputVariantV1 variant)
      : variant_(variant) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    if (variant_ == EquivalentInputVariantV1::Original)
      return;
    if (variant_ != EquivalentInputVariantV1::Metamorphic) {
      getOperation().emitError("unknown equivalent input variant");
      signalPassFailure();
      return;
    }

    uint64_t constructedSlices = 0;
    getOperation().walk([&](mlir::func::FuncOp function) {
      if (function.isExternal())
        return;
      mlir::Block &entry = function.getBody().front();
      mlir::OpBuilder builder(&entry, entry.begin());
      for (mlir::BlockArgument argument : entry.getArguments()) {
        auto type = mlir::dyn_cast<mlir::RankedTensorType>(argument.getType());
        if (!type || !type.hasStaticShape())
          continue;

        llvm::SmallVector<mlir::OpOperand *, 8> originalUses;
        for (mlir::OpOperand &use : argument.getUses())
          originalUses.push_back(&use);
        llvm::SmallVector<mlir::OpFoldResult, 4> offsets(
            type.getRank(), builder.getIndexAttr(0));
        llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
        sizes.reserve(type.getRank());
        for (int64_t dimension : type.getShape())
          sizes.push_back(builder.getIndexAttr(dimension));
        llvm::SmallVector<mlir::OpFoldResult, 4> strides(
            type.getRank(), builder.getIndexAttr(1));
        auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
            function.getLoc(), argument, offsets, sizes, strides);
        for (mlir::OpOperand *use : originalUses)
          use->set(slice.getResult());
        ++constructedSlices;
      }
    });

    if (constructedSlices == 0) {
      getOperation().emitError(
          "metamorphic input variant requires a static ranked tensor argument");
      signalPassFailure();
    }
  }

private:
  EquivalentInputVariantV1 variant_ = EquivalentInputVariantV1::Original;
};

} // namespace

std::unique_ptr<mlir::Pass>
createEquivalentInputVariantPass(EquivalentInputVariantV1 variant) {
  return std::make_unique<EquivalentInputVariantPass>(variant);
}

} // namespace wafer::qualification_internal
