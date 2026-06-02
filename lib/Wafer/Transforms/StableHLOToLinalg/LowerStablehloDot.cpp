//===- LowerStablehloDot.cpp - StableHLO dot to structured IR ------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
namespace {

#ifdef WAFER_ENABLE_STABLEHLO
static bool isSimple2DMatmul(mlir::stablehlo::DotGeneralOp dot) {
  auto lhsType = dot.getLhs().getType();
  auto rhsType = dot.getRhs().getType();
  auto resultType = dot.getType();
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return false;

  mlir::stablehlo::DotDimensionNumbersAttr dims =
      dot.getDotDimensionNumbers();
  return dims.getLhsBatchingDimensions().empty() &&
         dims.getRhsBatchingDimensions().empty() &&
         llvm::ArrayRef<int64_t>(dims.getLhsContractingDimensions()).equals(
             {1}) &&
         llvm::ArrayRef<int64_t>(dims.getRhsContractingDimensions()).equals(
             {0});
}

static mlir::TypedAttr getZeroAttr(mlir::OpBuilder &builder, mlir::Type type) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return builder.getFloatAttr(floatType, 0.0);
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type))
    return builder.getIntegerAttr(integerType, 0);
  return {};
}

static mlir::Value createZeroFilledTensor(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::RankedTensorType resultType) {
  mlir::TypedAttr zeroAttr = getZeroAttr(builder, resultType.getElementType());
  if (!zeroAttr)
    return {};

  auto zero = builder.create<mlir::arith::ConstantOp>(loc, zeroAttr);
  auto empty = builder.create<mlir::tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  auto filled = builder.create<mlir::linalg::FillOp>(
      loc, mlir::TypeRange{resultType}, mlir::ValueRange{zero},
      mlir::ValueRange{empty.getResult()});
  return filled.getResult(0);
}

static bool lowerSimple2DMatmul(mlir::stablehlo::DotGeneralOp dot) {
  mlir::OpBuilder builder(dot);
  mlir::RankedTensorType resultType = dot.getType();
  mlir::Value filled =
      createZeroFilledTensor(builder, dot.getLoc(), resultType);
  if (!filled)
    return false;

  auto matmul = builder.create<mlir::linalg::MatmulOp>(
      dot.getLoc(), mlir::TypeRange{resultType},
      mlir::ValueRange{dot.getLhs(), dot.getRhs()},
      mlir::ValueRange{filled});

  dot.getResult().replaceAllUsesWith(matmul.getResult(0));
  dot.erase();
  return true;
}
#endif

struct LowerStablehloDotPass
    : public mlir::PassWrapper<LowerStablehloDotPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStablehloDotPass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-stablehlo-dot";
  }

  llvm::StringRef getDescription() const final {
    return "lower simple StableHLO dot_general ops to structured tensor IR";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect>();
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::stablehlo::DotGeneralOp> dots;
    getOperation().walk([&](mlir::stablehlo::DotGeneralOp dot) {
      if (isSimple2DMatmul(dot))
        dots.push_back(dot);
    });

    for (mlir::stablehlo::DotGeneralOp dot : dots) {
      if (!lowerSimple2DMatmul(dot)) {
        dot.emitOpError("has unsupported result element type for zero init");
        signalPassFailure();
        return;
      }
    }
#endif
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerStablehloDotPass() {
  return std::make_unique<LowerStablehloDotPass>();
}

} // namespace wafer
