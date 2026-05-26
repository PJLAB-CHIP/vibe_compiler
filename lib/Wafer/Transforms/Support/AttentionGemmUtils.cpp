//===- AttentionGemmUtils.cpp - Attention contraction helpers -------------===//

#include "Support/AttentionGemmUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer {
namespace {

static bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic &&
         rhs != mlir::ShapedType::kDynamic && lhs != rhs;
}

static bool isDimList(mlir::AffineMap map, llvm::ArrayRef<unsigned> dims) {
  if (map.getNumDims() != 5 || map.getNumSymbols() != 0 ||
      map.getNumResults() != dims.size())
    return false;
  for (auto [expr, expectedDim] : llvm::zip(map.getResults(), dims)) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != expectedDim)
      return false;
  }
  return true;
}

static bool hasAttentionIteratorTypes(mlir::linalg::GenericOp generic) {
  auto iteratorTypes = generic.getIteratorTypesArray();
  if (iteratorTypes.size() != 5)
    return false;
  for (unsigned i = 0; i < 4; ++i)
    if (iteratorTypes[i] != mlir::utils::IteratorType::parallel)
      return false;
  return iteratorTypes[4] == mlir::utils::IteratorType::reduction;
}

static bool areBodyArguments(mlir::Value lhs, mlir::Value rhs,
                             mlir::BlockArgument arg0,
                             mlir::BlockArgument arg1) {
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

static mlir::Value getMulProduct(mlir::Value value, mlir::BlockArgument arg0,
                                 mlir::BlockArgument arg1) {
  if (auto mulf = value.getDefiningOp<mlir::arith::MulFOp>())
    if (areBodyArguments(mulf->getOperand(0), mulf->getOperand(1), arg0, arg1))
      return value;
  if (auto muli = value.getDefiningOp<mlir::arith::MulIOp>())
    if (areBodyArguments(muli->getOperand(0), muli->getOperand(1), arg0, arg1))
      return value;
  return {};
}

static bool isMulAddContractionBody(mlir::linalg::GenericOp generic) {
  if (generic.getRegion().empty())
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  mlir::Value yielded = yield.getValues()[0];
  mlir::BlockArgument lhs = body.getArgument(0);
  mlir::BlockArgument rhs = body.getArgument(1);
  mlir::BlockArgument accumulator = body.getArgument(2);

  auto isValidAdd = [&](mlir::Value addLhs, mlir::Value addRhs) {
    return (addLhs == accumulator && getMulProduct(addRhs, lhs, rhs)) ||
           (addRhs == accumulator && getMulProduct(addLhs, lhs, rhs));
  };

  if (auto addf = yielded.getDefiningOp<mlir::arith::AddFOp>())
    return isValidAdd(addf->getOperand(0), addf->getOperand(1));
  if (auto addi = yielded.getDefiningOp<mlir::arith::AddIOp>())
    return isValidAdd(addi->getOperand(0), addi->getOperand(1));
  return false;
}

static std::optional<AttentionGemmDims>
matchAttentionIndexingMaps(mlir::linalg::GenericOp generic) {
  llvm::SmallVector<mlir::AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 3)
    return std::nullopt;

  if (!isDimList(maps[0], {0, 1, 2, 4}) || !isDimList(maps[2], {0, 1, 2, 3}))
    return std::nullopt;

  AttentionGemmDims dims;
  if (isDimList(maps[1], {0, 1, 3, 4})) {
    dims.rhsContractingDim = 3;
    dims.rhsNDim = 2;
    return dims;
  }
  if (isDimList(maps[1], {0, 1, 4, 3})) {
    dims.rhsContractingDim = 2;
    dims.rhsNDim = 3;
    return dims;
  }
  return std::nullopt;
}

static bool hasValidAttentionShapes(mlir::linalg::GenericOp generic,
                                    AttentionGemmDims dims) {
  llvm::SmallVector<mlir::Value> inputs = generic.getDpsInputs();
  mlir::OperandRange inits = generic.getDpsInits();
  if (inputs.size() != 2 || inits.size() != 1 || generic->getNumResults() != 1)
    return false;

  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(inputs[0].getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(inputs[1].getType());
  auto initType = mlir::dyn_cast<mlir::RankedTensorType>(inits[0].getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!lhsType || !rhsType || !initType || !resultType)
    return false;
  if (lhsType.getRank() != 4 || rhsType.getRank() != 4 ||
      resultType.getRank() != 4 || initType != resultType)
    return false;
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != resultType.getElementType())
    return false;

  for (int64_t dim : {0, 1}) {
    if (hasStaticMismatch(lhsType.getDimSize(dim), rhsType.getDimSize(dim)) ||
        hasStaticMismatch(lhsType.getDimSize(dim), resultType.getDimSize(dim)))
      return false;
  }
  if (hasStaticMismatch(lhsType.getDimSize(dims.lhsMDim),
                        resultType.getDimSize(dims.resultMDim)))
    return false;
  if (hasStaticMismatch(rhsType.getDimSize(dims.rhsNDim),
                        resultType.getDimSize(dims.resultNDim)))
    return false;
  if (hasStaticMismatch(lhsType.getDimSize(dims.lhsContractingDim),
                        rhsType.getDimSize(dims.rhsContractingDim)))
    return false;
  return true;
}

static int64_t getStaticBatchCount(mlir::RankedTensorType resultType) {
  int64_t batchCount = 1;
  for (int64_t dim : {0, 1})
    batchCount *= resultType.getDimSize(dim);
  return batchCount;
}

} // namespace

std::optional<AttentionGemmDims>
matchAttentionGemm(mlir::linalg::GenericOp generic) {
  if (!hasAttentionIteratorTypes(generic) || !isMulAddContractionBody(generic))
    return std::nullopt;
  std::optional<AttentionGemmDims> dims = matchAttentionIndexingMaps(generic);
  if (!dims || !hasValidAttentionShapes(generic, *dims))
    return std::nullopt;
  return dims;
}

void setAttentionGemmAttrs(mlir::Operation *op, mlir::OpBuilder &builder,
                           mlir::RankedTensorType resultType,
                           AttentionGemmDims dims) {
  op->setAttr("batch_count",
              builder.getI64IntegerAttr(getStaticBatchCount(resultType)));
  op->setAttr("lhs_batch_dims", builder.getDenseI64ArrayAttr({0, 1}));
  op->setAttr("rhs_batch_dims", builder.getDenseI64ArrayAttr({0, 1}));
  op->setAttr("result_batch_dims", builder.getDenseI64ArrayAttr({0, 1}));
  op->setAttr("lhs_m_dim", builder.getI64IntegerAttr(dims.lhsMDim));
  op->setAttr("lhs_contracting_dim",
              builder.getI64IntegerAttr(dims.lhsContractingDim));
  op->setAttr("rhs_contracting_dim",
              builder.getI64IntegerAttr(dims.rhsContractingDim));
  op->setAttr("rhs_n_dim", builder.getI64IntegerAttr(dims.rhsNDim));
  op->setAttr("result_m_dim", builder.getI64IntegerAttr(dims.resultMDim));
  op->setAttr("result_n_dim", builder.getI64IntegerAttr(dims.resultNDim));
}

} // namespace wafer
