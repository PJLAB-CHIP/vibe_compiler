//===- LowerStablehloDot.cpp - StableHLO dot to structured IR ------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
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
enum class DotLoweringKind { Matmul2D, AttentionScoreQKt };

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

static bool hasStaticShape(mlir::stablehlo::DotGeneralOp dot) {
  return dot.getLhs().getType().hasStaticShape() &&
         dot.getRhs().getType().hasStaticShape() &&
         dot.getType().hasStaticShape();
}

static bool sameElementType(mlir::stablehlo::DotGeneralOp dot) {
  return dot.getLhs().getType().getElementType() ==
             dot.getType().getElementType() &&
         dot.getRhs().getType().getElementType() ==
             dot.getType().getElementType();
}

static bool isAttentionScoreQKt(mlir::stablehlo::DotGeneralOp dot) {
  auto lhsType = dot.getLhs().getType();
  auto rhsType = dot.getRhs().getType();
  auto resultType = dot.getType();
  if (lhsType.getRank() != 4 || rhsType.getRank() != 4 ||
      resultType.getRank() != 4 || !hasStaticShape(dot) ||
      !sameElementType(dot))
    return false;

  mlir::stablehlo::DotDimensionNumbersAttr dims =
      dot.getDotDimensionNumbers();
  if (!llvm::ArrayRef<int64_t>(dims.getLhsBatchingDimensions()).equals(
          {0, 1}) ||
      !llvm::ArrayRef<int64_t>(dims.getRhsBatchingDimensions()).equals(
          {0, 1}) ||
      !llvm::ArrayRef<int64_t>(dims.getLhsContractingDimensions()).equals(
          {3}) ||
      !llvm::ArrayRef<int64_t>(dims.getRhsContractingDimensions()).equals({3}))
    return false;

  return lhsType.getDimSize(0) == rhsType.getDimSize(0) &&
         lhsType.getDimSize(1) == rhsType.getDimSize(1) &&
         lhsType.getDimSize(3) == rhsType.getDimSize(3) &&
         resultType.getDimSize(0) == lhsType.getDimSize(0) &&
         resultType.getDimSize(1) == lhsType.getDimSize(1) &&
         resultType.getDimSize(2) == lhsType.getDimSize(2) &&
         resultType.getDimSize(3) == rhsType.getDimSize(2);
}

static std::optional<DotLoweringKind>
getDotLoweringKind(mlir::stablehlo::DotGeneralOp dot) {
  if (isSimple2DMatmul(dot))
    return DotLoweringKind::Matmul2D;
  if (isAttentionScoreQKt(dot))
    return DotLoweringKind::AttentionScoreQKt;
  return std::nullopt;
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

static mlir::ArrayAttr getAttentionIteratorTypes(mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  return builder.getArrayAttr({
      mlir::linalg::IteratorTypeAttr::get(context,
                                          mlir::utils::IteratorType::parallel),
      mlir::linalg::IteratorTypeAttr::get(context,
                                          mlir::utils::IteratorType::parallel),
      mlir::linalg::IteratorTypeAttr::get(context,
                                          mlir::utils::IteratorType::parallel),
      mlir::linalg::IteratorTypeAttr::get(context,
                                          mlir::utils::IteratorType::parallel),
      mlir::linalg::IteratorTypeAttr::get(context,
                                          mlir::utils::IteratorType::reduction),
  });
}

static mlir::ArrayAttr getAttentionIndexingMaps(mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::AffineExpr b = builder.getAffineDimExpr(0);
  mlir::AffineExpr h = builder.getAffineDimExpr(1);
  mlir::AffineExpr q = builder.getAffineDimExpr(2);
  mlir::AffineExpr k = builder.getAffineDimExpr(3);
  mlir::AffineExpr d = builder.getAffineDimExpr(4);
  llvm::SmallVector<mlir::AffineMap> indexingMaps = {
      mlir::AffineMap::get(5, 0, {b, h, q, d}, context),
      mlir::AffineMap::get(5, 0, {b, h, k, d}, context),
      mlir::AffineMap::get(5, 0, {b, h, q, k}, context),
  };
  return builder.getAffineMapArrayAttr(indexingMaps);
}

static mlir::Value createMulAdd(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value lhs, mlir::Value rhs,
                                mlir::Value accumulator) {
  mlir::Type type = accumulator.getType();
  if (mlir::isa<mlir::FloatType>(type)) {
    mlir::Value product =
        builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::AddFOp>(loc, accumulator, product);
  }
  if (mlir::isa<mlir::IntegerType>(type)) {
    mlir::Value product =
        builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::AddIOp>(loc, accumulator, product);
  }
  return {};
}

static bool lowerAttentionScoreQKt(mlir::stablehlo::DotGeneralOp dot) {
  mlir::OpBuilder builder(dot);
  mlir::RankedTensorType resultType = dot.getType();
  mlir::Value filled =
      createZeroFilledTensor(builder, dot.getLoc(), resultType);
  if (!filled)
    return false;

  auto generic = builder.create<mlir::linalg::GenericOp>(
      dot.getLoc(), mlir::TypeRange{resultType},
      mlir::ValueRange{dot.getLhs(), dot.getRhs()},
      mlir::ValueRange{filled}, getAttentionIndexingMaps(builder),
      getAttentionIteratorTypes(builder), mlir::StringAttr{},
      mlir::StringAttr{});

  mlir::Block *body = new mlir::Block();
  generic.getRegion().push_back(body);
  mlir::Location loc = dot.getLoc();
  mlir::Type elementType = resultType.getElementType();
  body->addArgument(elementType, loc);
  body->addArgument(elementType, loc);
  body->addArgument(elementType, loc);

  mlir::OpBuilder nestedBuilder = mlir::OpBuilder::atBlockEnd(body);
  mlir::Value combined =
      createMulAdd(nestedBuilder, loc, body->getArgument(0),
                   body->getArgument(1), body->getArgument(2));
  if (!combined)
    return false;
  nestedBuilder.create<mlir::linalg::YieldOp>(loc, combined);

  dot.getResult().replaceAllUsesWith(generic.getResult(0));
  dot.erase();
  return true;
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
      if (getDotLoweringKind(dot))
        dots.push_back(dot);
    });

    for (mlir::stablehlo::DotGeneralOp dot : dots) {
      std::optional<DotLoweringKind> kind = getDotLoweringKind(dot);
      bool lowered = false;
      if (kind == DotLoweringKind::Matmul2D)
        lowered = lowerSimple2DMatmul(dot);
      else if (kind == DotLoweringKind::AttentionScoreQKt)
        lowered = lowerAttentionScoreQKt(dot);

      if (!lowered) {
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
