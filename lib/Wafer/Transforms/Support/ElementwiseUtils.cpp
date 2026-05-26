//===- ElementwiseUtils.cpp - Elementwise generic helpers -----------------===//

#include "Support/ElementwiseUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer {
namespace {

static bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic &&
         rhs != mlir::ShapedType::kDynamic && lhs != rhs;
}

static bool areBlockArguments(mlir::Value lhs, mlir::Value rhs,
                              mlir::BlockArgument arg0,
                              mlir::BlockArgument arg1) {
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
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

static bool isSplatZero(mlir::Attribute attr) {
  if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!dense.isSplat())
      return false;
    if (auto floats = mlir::dyn_cast<mlir::DenseFPElementsAttr>(dense))
      return floats.getSplatValue<llvm::APFloat>().isZero();
    if (auto ints = mlir::dyn_cast<mlir::DenseIntElementsAttr>(dense))
      return ints.getSplatValue<llvm::APInt>().isZero();
  }

  if (auto value = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return value.getValue().isZero();
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return value.getValue().isZero();
  return false;
}

static bool isConstantOne(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return isSplatOne(constant.getValue());
  return false;
}

static bool isConstantZero(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return isSplatZero(constant.getValue());
  return false;
}

static std::optional<wafer::ComputeElementwiseKind>
matchUnaryBody(mlir::Operation *op, mlir::BlockArgument input) {
  if (auto neg = mlir::dyn_cast_or_null<mlir::arith::NegFOp>(op))
    if (neg.getOperand() == input)
      return wafer::ComputeElementwiseKind::Neg;
  if (auto sqrt = mlir::dyn_cast_or_null<mlir::math::SqrtOp>(op))
    if (sqrt.getOperand() == input)
      return wafer::ComputeElementwiseKind::Sqrt;
  if (auto rsqrt = mlir::dyn_cast_or_null<mlir::math::RsqrtOp>(op))
    if (rsqrt.getOperand() == input)
      return wafer::ComputeElementwiseKind::Rsqrt;
  if (auto exp = mlir::dyn_cast_or_null<mlir::math::ExpOp>(op))
    if (exp.getOperand() == input)
      return wafer::ComputeElementwiseKind::Exp;
  if (auto tanh = mlir::dyn_cast_or_null<mlir::math::TanhOp>(op))
    if (tanh.getOperand() == input)
      return wafer::ComputeElementwiseKind::Tanh;
  if (auto div = mlir::dyn_cast_or_null<mlir::arith::DivFOp>(op))
    if (isConstantOne(div.getLhs()) && div.getRhs() == input)
      return wafer::ComputeElementwiseKind::Recip;
  if (auto sub = mlir::dyn_cast_or_null<mlir::arith::SubIOp>(op))
    if (isConstantZero(sub.getLhs()) && sub.getRhs() == input)
      return wafer::ComputeElementwiseKind::Neg;
  return std::nullopt;
}

static std::optional<wafer::ComputeElementwiseKind>
matchBinaryBody(mlir::Operation *op, mlir::BlockArgument lhs,
                mlir::BlockArgument rhs) {
  if (auto addf = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(op))
    if (areBlockArguments(addf.getLhs(), addf.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Add;
  if (auto addi = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(op))
    if (areBlockArguments(addi.getLhs(), addi.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Add;
  if (auto subf = mlir::dyn_cast_or_null<mlir::arith::SubFOp>(op))
    if (subf.getLhs() == lhs && subf.getRhs() == rhs)
      return wafer::ComputeElementwiseKind::Sub;
  if (auto subi = mlir::dyn_cast_or_null<mlir::arith::SubIOp>(op))
    if (subi.getLhs() == lhs && subi.getRhs() == rhs)
      return wafer::ComputeElementwiseKind::Sub;
  if (auto mulf = mlir::dyn_cast_or_null<mlir::arith::MulFOp>(op))
    if (areBlockArguments(mulf.getLhs(), mulf.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Mul;
  if (auto muli = mlir::dyn_cast_or_null<mlir::arith::MulIOp>(op))
    if (areBlockArguments(muli.getLhs(), muli.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Mul;
  if (auto divf = mlir::dyn_cast_or_null<mlir::arith::DivFOp>(op))
    if (divf.getLhs() == lhs && divf.getRhs() == rhs)
      return wafer::ComputeElementwiseKind::Div;
  if (auto divi = mlir::dyn_cast_or_null<mlir::arith::DivSIOp>(op))
    if (divi.getLhs() == lhs && divi.getRhs() == rhs)
      return wafer::ComputeElementwiseKind::Div;
  if (auto maxf = mlir::dyn_cast_or_null<mlir::arith::MaximumFOp>(op))
    if (areBlockArguments(maxf.getLhs(), maxf.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Max;
  if (auto maxi = mlir::dyn_cast_or_null<mlir::arith::MaxSIOp>(op))
    if (areBlockArguments(maxi.getLhs(), maxi.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Max;
  if (auto minf = mlir::dyn_cast_or_null<mlir::arith::MinimumFOp>(op))
    if (areBlockArguments(minf.getLhs(), minf.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Min;
  if (auto mini = mlir::dyn_cast_or_null<mlir::arith::MinSIOp>(op))
    if (areBlockArguments(mini.getLhs(), mini.getRhs(), lhs, rhs))
      return wafer::ComputeElementwiseKind::Min;
  return std::nullopt;
}

} // namespace

std::optional<wafer::ComputeElementwiseKind>
matchElementwiseGeneric(mlir::linalg::GenericOp generic) {
  if (generic.getRegion().empty() || generic->getNumResults() != 1 ||
      generic.getDpsInits().size() != 1)
    return std::nullopt;

  mlir::Block &body = generic.getRegion().front();
  if (body.getNumArguments() !=
      generic.getDpsInputs().size() + generic.getDpsInits().size())
    return std::nullopt;

  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return std::nullopt;

  mlir::Operation *yielded = yield.getValues()[0].getDefiningOp();
  if (!yielded || yielded->getBlock() != &body)
    return std::nullopt;

  if (generic.getDpsInputs().size() == 1)
    return matchUnaryBody(yielded, body.getArgument(0));
  if (generic.getDpsInputs().size() == 2)
    return matchBinaryBody(yielded, body.getArgument(0), body.getArgument(1));
  return std::nullopt;
}

bool isLimitedBroadcastElementwiseGeneric(mlir::linalg::GenericOp generic) {
  if (!matchElementwiseGeneric(generic))
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](auto iteratorType) {
        return iteratorType == mlir::utils::IteratorType::parallel;
      }))
    return false;

  llvm::SmallVector<mlir::Value> inputs = generic.getDpsInputs();
  mlir::OperandRange outputs = generic.getDpsInits();
  if (outputs.size() != 1 || generic->getNumResults() != 1)
    return false;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultType || outputs[0].getType() != resultType)
    return false;

  llvm::SmallVector<mlir::AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != inputs.size() + outputs.size())
    return false;
  mlir::AffineMap resultMap = maps.back();
  if (resultMap.getNumDims() != resultType.getRank() ||
      resultMap.getNumSymbols() != 0 || !resultMap.isIdentity())
    return false;

  for (auto [index, input] : llvm::enumerate(inputs)) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || inputType.getElementType() != resultType.getElementType())
      return false;

    mlir::AffineMap inputMap = maps[index];
    if (inputMap.getNumDims() != resultType.getRank() ||
        inputMap.getNumSymbols() != 0 ||
        inputMap.getNumResults() != inputType.getRank() ||
        !inputMap.isProjectedPermutation())
      return false;

    for (auto [dim, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr || dimExpr.getPosition() >= resultType.getRank())
        return false;
      if (hasStaticMismatch(inputType.getDimSize(dim),
                            resultType.getDimSize(dimExpr.getPosition())))
        return false;
    }
  }

  return true;
}

} // namespace wafer
