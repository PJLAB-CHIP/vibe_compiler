//===- LowerStablehloElementwise.cpp - StableHLO elementwise lowering -----===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
namespace {

#ifdef WAFER_ENABLE_STABLEHLO
struct ElementwiseInput {
  mlir::Value value;
  mlir::AffineMap indexingMap;
  mlir::Operation *sourceBroadcast = nullptr;
};

static mlir::Value createEmptyTensor(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType type) {
  return builder
      .create<mlir::tensor::EmptyOp>(loc, type.getShape(),
                                     type.getElementType())
      .getResult();
}

static bool isSplatOne(mlir::Attribute attr) {
  if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!dense.isSplat())
      return false;
    if (auto floats = mlir::dyn_cast<mlir::DenseFPElementsAttr>(dense))
      return floats.getSplatValue<llvm::APFloat>().convertToDouble() == 1.0;
    if (auto ints = mlir::dyn_cast<mlir::DenseIntElementsAttr>(dense))
      return ints.getSplatValue<llvm::APInt>().isOne();
  }

  if (auto value = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return value.getValue().convertToDouble() == 1.0;
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return value.getValue().isOne();
  return false;
}

static bool isConstantOne(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return isSplatOne(constant.getValue());
  if (auto constant = value.getDefiningOp<mlir::stablehlo::ConstantOp>())
    return isSplatOne(constant.getValue());
  return false;
}

static mlir::AffineMap getIdentityMap(mlir::OpBuilder &builder,
                                      int64_t rank) {
  return mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
}

static mlir::AffineMap getBroadcastInputMap(
    mlir::OpBuilder &builder, int64_t resultRank,
    mlir::stablehlo::BroadcastInDimOp broadcast) {
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int64_t dim : broadcast.getBroadcastDimensions())
    exprs.push_back(builder.getAffineDimExpr(dim));
  return mlir::AffineMap::get(resultRank, 0, exprs, builder.getContext());
}

static ElementwiseInput getElementwiseInput(mlir::OpBuilder &builder,
                                            mlir::Value input,
                                            int64_t resultRank) {
  auto broadcast =
      input.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!broadcast)
    return {input, getIdentityMap(builder, resultRank)};

  auto operandType =
      mlir::dyn_cast<mlir::RankedTensorType>(broadcast.getOperand().getType());
  if (!operandType)
    return {input, getIdentityMap(builder, resultRank)};
  if (operandType.getRank() !=
      static_cast<int64_t>(broadcast.getBroadcastDimensions().size()))
    return {input, getIdentityMap(builder, resultRank)};

  return {broadcast.getOperand(),
          getBroadcastInputMap(builder, resultRank, broadcast),
          broadcast.getOperation()};
}

static bool canCreateScalarElementwise(mlir::Type type,
                                       wafer::ComputeElementwiseKind kind,
                                       size_t arity) {
  bool isFloat = mlir::isa<mlir::FloatType>(type);
  bool isInteger = mlir::isa<mlir::IntegerType>(type);
  switch (kind) {
  case wafer::ComputeElementwiseKind::Add:
  case wafer::ComputeElementwiseKind::Sub:
  case wafer::ComputeElementwiseKind::Mul:
  case wafer::ComputeElementwiseKind::Div:
  case wafer::ComputeElementwiseKind::Max:
  case wafer::ComputeElementwiseKind::Min:
    return arity == 2 && (isFloat || isInteger);
  case wafer::ComputeElementwiseKind::Neg:
    return arity == 1 && (isFloat || isInteger);
  case wafer::ComputeElementwiseKind::Recip:
  case wafer::ComputeElementwiseKind::Sqrt:
  case wafer::ComputeElementwiseKind::Rsqrt:
  case wafer::ComputeElementwiseKind::Exp:
  case wafer::ComputeElementwiseKind::Tanh:
    return arity == 1 && isFloat;
  }
  return false;
}

static mlir::Value createScalarConstant(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Type type,
                                        double value) {
  mlir::TypedAttr attr;
  if (mlir::isa<mlir::FloatType>(type))
    attr = builder.getFloatAttr(type, value);
  else if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type))
    attr = builder.getIntegerAttr(integerType, static_cast<int64_t>(value));
  else
    return {};
  return builder.create<mlir::arith::ConstantOp>(loc, attr).getResult();
}

static mlir::Value createScalarElementwise(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           wafer::ComputeElementwiseKind kind,
                                           mlir::ValueRange args) {
  mlir::Type type = args.front().getType();
  if (!canCreateScalarElementwise(type, kind, args.size()))
    return {};

  bool isFloat = mlir::isa<mlir::FloatType>(type);
  switch (kind) {
  case wafer::ComputeElementwiseKind::Add:
    if (isFloat)
      return builder.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
    return builder.create<mlir::arith::AddIOp>(loc, args[0], args[1]);
  case wafer::ComputeElementwiseKind::Sub:
    if (isFloat)
      return builder.create<mlir::arith::SubFOp>(loc, args[0], args[1]);
    return builder.create<mlir::arith::SubIOp>(loc, args[0], args[1]);
  case wafer::ComputeElementwiseKind::Mul:
    if (isFloat)
      return builder.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
    return builder.create<mlir::arith::MulIOp>(loc, args[0], args[1]);
  case wafer::ComputeElementwiseKind::Div:
    if (isFloat)
      return builder.create<mlir::arith::DivFOp>(loc, args[0], args[1]);
    return builder.create<mlir::arith::DivSIOp>(loc, args[0], args[1]);
  case wafer::ComputeElementwiseKind::Max:
    if (isFloat)
      return builder.create<mlir::arith::MaximumFOp>(loc, args[0], args[1]);
    return builder.create<mlir::arith::MaxSIOp>(loc, args[0], args[1]);
  case wafer::ComputeElementwiseKind::Min:
    if (isFloat)
      return builder.create<mlir::arith::MinimumFOp>(loc, args[0], args[1]);
    return builder.create<mlir::arith::MinSIOp>(loc, args[0], args[1]);
  case wafer::ComputeElementwiseKind::Neg:
    if (isFloat)
      return builder.create<mlir::arith::NegFOp>(loc, args[0]);
    if (mlir::Value zero = createScalarConstant(builder, loc, type, 0.0))
      return builder.create<mlir::arith::SubIOp>(loc, zero, args[0]);
    return {};
  case wafer::ComputeElementwiseKind::Recip:
    if (mlir::Value one = createScalarConstant(builder, loc, type, 1.0))
      return builder.create<mlir::arith::DivFOp>(loc, one, args[0]);
    return {};
  case wafer::ComputeElementwiseKind::Sqrt:
    return builder.create<mlir::math::SqrtOp>(loc, args[0]);
  case wafer::ComputeElementwiseKind::Rsqrt:
    return builder.create<mlir::math::RsqrtOp>(loc, args[0]);
  case wafer::ComputeElementwiseKind::Exp:
    return builder.create<mlir::math::ExpOp>(loc, args[0]);
  case wafer::ComputeElementwiseKind::Tanh:
    return builder.create<mlir::math::TanhOp>(loc, args[0]);
  }
  return {};
}

static bool lowerElementwise(mlir::Operation *op,
                             wafer::ComputeElementwiseKind kind,
                             llvm::ArrayRef<mlir::Value> inputs) {
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  if (!canCreateScalarElementwise(resultType.getElementType(), kind,
                                  inputs.size()))
    return false;

  mlir::OpBuilder builder(op);
  llvm::SmallVector<ElementwiseInput> mappedInputs;
  llvm::SmallVector<mlir::Value> loweredInputs;
  llvm::SmallVector<mlir::AffineMap> indexingMaps;
  int64_t resultRank = resultType.getRank();
  for (mlir::Value input : inputs) {
    ElementwiseInput mapped = getElementwiseInput(builder, input, resultRank);
    loweredInputs.push_back(mapped.value);
    indexingMaps.push_back(mapped.indexingMap);
    mappedInputs.push_back(mapped);
  }

  indexingMaps.push_back(getIdentityMap(builder, resultRank));
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      resultRank, mlir::utils::IteratorType::parallel);

  mlir::Value empty = createEmptyTensor(builder, op->getLoc(), resultType);
  auto lowered = builder.create<mlir::linalg::GenericOp>(
      op->getLoc(), mlir::TypeRange{resultType}, loweredInputs,
      mlir::ValueRange{empty}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc,
          mlir::ValueRange blockArgs) {
        mlir::Value scalar = createScalarElementwise(
            nestedBuilder, loc, kind, blockArgs.take_front(inputs.size()));
        nestedBuilder.create<mlir::linalg::YieldOp>(loc, scalar);
      });

  op->getResult(0).replaceAllUsesWith(lowered.getOperation()->getResult(0));
  op->erase();

  llvm::SmallPtrSet<mlir::Operation *, 2> maybeDeadBroadcasts;
  for (const ElementwiseInput &mapped : mappedInputs)
    if (mapped.sourceBroadcast)
      maybeDeadBroadcasts.insert(mapped.sourceBroadcast);
  for (mlir::Operation *broadcast : maybeDeadBroadcasts)
    if (broadcast->use_empty())
      broadcast->erase();

  return true;
}

static bool lowerBinary(mlir::Operation *op, mlir::Value lhs, mlir::Value rhs,
                        wafer::ComputeElementwiseKind kind) {
  return lowerElementwise(op, kind, {lhs, rhs});
}

static bool lowerUnary(mlir::Operation *op, mlir::Value input,
                       wafer::ComputeElementwiseKind kind) {
  return lowerElementwise(op, kind, {input});
}
#endif

struct LowerStablehloElementwisePass
    : public mlir::PassWrapper<LowerStablehloElementwisePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStablehloElementwisePass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-stablehlo-elementwise";
  }

  llvm::StringRef getDescription() const final {
    return "lower StableHLO elementwise ops to linalg.generic";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect, mlir::tensor::TensorDialect>();
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::Operation *> ops;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                    mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                    mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp,
                    mlir::stablehlo::NegOp, mlir::stablehlo::SqrtOp,
                    mlir::stablehlo::RsqrtOp, mlir::stablehlo::ExpOp,
                    mlir::stablehlo::TanhOp>(op))
        ops.push_back(op);
    });

    for (mlir::Operation *op : ops) {
      if (auto add = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
        (void)lowerBinary(op, add.getLhs(), add.getRhs(),
                          wafer::ComputeElementwiseKind::Add);
      } else if (auto sub =
                     mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
        (void)lowerBinary(op, sub.getLhs(), sub.getRhs(),
                          wafer::ComputeElementwiseKind::Sub);
      } else if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
        (void)lowerBinary(op, mul.getLhs(), mul.getRhs(),
                          wafer::ComputeElementwiseKind::Mul);
      } else if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(op)) {
        mlir::Value lhs = div.getLhs();
        if (isConstantOne(lhs)) {
          if (lowerUnary(op, div.getRhs(),
                         wafer::ComputeElementwiseKind::Recip))
            if (mlir::Operation *constant = lhs.getDefiningOp();
                constant && constant->use_empty())
              constant->erase();
        } else {
          (void)lowerBinary(op, div.getLhs(), div.getRhs(),
                            wafer::ComputeElementwiseKind::Div);
        }
      } else if (auto max = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
        (void)lowerBinary(op, max.getLhs(), max.getRhs(),
                          wafer::ComputeElementwiseKind::Max);
      } else if (auto min = mlir::dyn_cast<mlir::stablehlo::MinOp>(op)) {
        (void)lowerBinary(op, min.getLhs(), min.getRhs(),
                          wafer::ComputeElementwiseKind::Min);
      } else if (auto neg = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
        (void)lowerUnary(op, neg.getOperand(),
                         wafer::ComputeElementwiseKind::Neg);
      } else if (auto sqrt = mlir::dyn_cast<mlir::stablehlo::SqrtOp>(op)) {
        (void)lowerUnary(op, sqrt.getOperand(),
                         wafer::ComputeElementwiseKind::Sqrt);
      } else if (auto rsqrt = mlir::dyn_cast<mlir::stablehlo::RsqrtOp>(op)) {
        (void)lowerUnary(op, rsqrt.getOperand(),
                         wafer::ComputeElementwiseKind::Rsqrt);
      } else if (auto exp = mlir::dyn_cast<mlir::stablehlo::ExpOp>(op)) {
        (void)lowerUnary(op, exp.getOperand(),
                         wafer::ComputeElementwiseKind::Exp);
      } else if (auto tanh = mlir::dyn_cast<mlir::stablehlo::TanhOp>(op)) {
        (void)lowerUnary(op, tanh.getOperand(),
                         wafer::ComputeElementwiseKind::Tanh);
      }
    }
#endif
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerStablehloElementwisePass() {
  return std::make_unique<LowerStablehloElementwisePass>();
}

} // namespace wafer
