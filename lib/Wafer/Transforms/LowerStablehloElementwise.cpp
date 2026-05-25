//===- LowerStablehloElementwise.cpp - StableHLO elementwise lowering -----===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

static bool lowerElementwise(mlir::Operation *op,
                             mlir::linalg::ElementwiseKind kind,
                             llvm::ArrayRef<mlir::Value> inputs) {
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
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
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr(
      "kind",
      mlir::linalg::ElementwiseKindAttr::get(builder.getContext(), kind)));
  attrs.push_back(builder.getNamedAttr(
      "indexing_maps", builder.getAffineMapArrayAttr(indexingMaps)));

  mlir::Value empty = createEmptyTensor(builder, op->getLoc(), resultType);
  auto lowered = builder.create<mlir::linalg::ElementwiseOp>(
      op->getLoc(), loweredInputs, mlir::ValueRange{empty}, attrs);

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
                        mlir::linalg::ElementwiseKind kind) {
  return lowerElementwise(op, kind, {lhs, rhs});
}

static bool lowerUnary(mlir::Operation *op, mlir::Value input,
                       mlir::linalg::ElementwiseKind kind) {
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
    return "lower StableHLO elementwise ops to linalg.elementwise";
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
                          mlir::linalg::ElementwiseKind::add);
      } else if (auto sub =
                     mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
        (void)lowerBinary(op, sub.getLhs(), sub.getRhs(),
                          mlir::linalg::ElementwiseKind::sub);
      } else if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
        (void)lowerBinary(op, mul.getLhs(), mul.getRhs(),
                          mlir::linalg::ElementwiseKind::mul);
      } else if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(op)) {
        mlir::Value lhs = div.getLhs();
        if (isConstantOne(lhs)) {
          if (lowerUnary(op, div.getRhs(),
                         mlir::linalg::ElementwiseKind::reciprocal))
            if (mlir::Operation *constant = lhs.getDefiningOp();
                constant && constant->use_empty())
              constant->erase();
        } else {
          (void)lowerBinary(op, div.getLhs(), div.getRhs(),
                            mlir::linalg::ElementwiseKind::div);
        }
      } else if (auto max = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
        (void)lowerBinary(op, max.getLhs(), max.getRhs(),
                          mlir::linalg::ElementwiseKind::max_signed);
      } else if (auto min = mlir::dyn_cast<mlir::stablehlo::MinOp>(op)) {
        (void)lowerBinary(op, min.getLhs(), min.getRhs(),
                          mlir::linalg::ElementwiseKind::min_signed);
      } else if (auto neg = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
        (void)lowerUnary(op, neg.getOperand(),
                         mlir::linalg::ElementwiseKind::negf);
      } else if (auto sqrt = mlir::dyn_cast<mlir::stablehlo::SqrtOp>(op)) {
        (void)lowerUnary(op, sqrt.getOperand(),
                         mlir::linalg::ElementwiseKind::sqrt);
      } else if (auto rsqrt = mlir::dyn_cast<mlir::stablehlo::RsqrtOp>(op)) {
        (void)lowerUnary(op, rsqrt.getOperand(),
                         mlir::linalg::ElementwiseKind::rsqrt);
      } else if (auto exp = mlir::dyn_cast<mlir::stablehlo::ExpOp>(op)) {
        (void)lowerUnary(op, exp.getOperand(),
                         mlir::linalg::ElementwiseKind::exp);
      } else if (auto tanh = mlir::dyn_cast<mlir::stablehlo::TanhOp>(op)) {
        (void)lowerUnary(op, tanh.getOperand(),
                         mlir::linalg::ElementwiseKind::tanh);
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
