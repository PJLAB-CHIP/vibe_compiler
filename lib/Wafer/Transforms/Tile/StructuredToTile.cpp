//===- StructuredToTile.cpp - Lower current structured compute --------===//

#include "StructuredToTile.h"

#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace wafer::compiler::detail {
namespace {

enum class LoweringKind : uint8_t {
  Fill,
  CapturedFill,
  Contraction,
  Convolution,
  Reduction,
  Elementwise,
};

struct LoweringPlan {
  mlir::linalg::LinalgOp operation;
  LoweringKind kind = LoweringKind::Elementwise;
};

struct ExprValue {
  mlir::Value buffer;
  mlir::AffineMap indexingMap;
};

static StructuredToTileResult fail(StructuredToTileFailureKind kind,
                                   llvm::StringRef detail) {
  StructuredToTileResult result;
  result.failure = kind;
  result.detail = detail.str();
  return result;
}

static mlir::MemRefType getMemRef(mlir::Value value) {
  return value ? mlir::dyn_cast<mlir::MemRefType>(value.getType())
               : mlir::MemRefType{};
}

static mlir::MemRefType changeElementType(mlir::MemRefType source,
                                          mlir::Type elementType) {
  return mlir::MemRefType::get(source.getShape(), elementType,
                               source.getLayout(), source.getMemorySpace());
}

static mlir::MemRefType getOwnedType(mlir::MemRefType source) {
  return mlir::MemRefType::get(source.getShape(), source.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               source.getMemorySpace());
}

static mlir::MemRefType changeShapeAndElementType(mlir::MemRefType source,
                                                  llvm::ArrayRef<int64_t> shape,
                                                  mlir::Type elementType) {
  return mlir::MemRefType::get(shape, elementType,
                               mlir::MemRefLayoutAttrInterface{},
                               source.getMemorySpace());
}

static bool isStaticPositive(mlir::MemRefType type) {
  return type && type.hasStaticShape() &&
         llvm::all_of(type.getShape(),
                      [](int64_t extent) { return extent > 0; });
}

static bool hasOnlyParallelIterators(mlir::linalg::LinalgOp operation) {
  return llvm::all_of(operation.getIteratorTypesArray(), [](auto iterator) {
    return iterator == mlir::utils::IteratorType::parallel;
  });
}

static bool hasReductionIterator(mlir::linalg::LinalgOp operation) {
  return llvm::any_of(operation.getIteratorTypesArray(), [](auto iterator) {
    return iterator == mlir::utils::IteratorType::reduction;
  });
}

static bool matchesPair(mlir::Value lhs, mlir::Value rhs,
                        mlir::Value expectedLhs, mlir::Value expectedRhs) {
  return (lhs == expectedLhs && rhs == expectedRhs) ||
         (lhs == expectedRhs && rhs == expectedLhs);
}

static bool hasNoIntegerOverflowFlags(mlir::Operation *operation) {
  if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(operation))
    return add.getOverflowFlags() == mlir::arith::IntegerOverflowFlags::none;
  if (auto sub = mlir::dyn_cast<mlir::arith::SubIOp>(operation))
    return sub.getOverflowFlags() == mlir::arith::IntegerOverflowFlags::none;
  if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(operation))
    return mul.getOverflowFlags() == mlir::arith::IntegerOverflowFlags::none;
  return true;
}

static mlir::Value stripFloatCast(mlir::Value value,
                                  mlir::BlockArgument expected) {
  if (value == expected)
    return value;
  if (auto ext = value.getDefiningOp<mlir::arith::ExtFOp>())
    return ext.getIn() == expected ? ext.getIn() : mlir::Value{};
  if (auto trunc = value.getDefiningOp<mlir::arith::TruncFOp>())
    return trunc.getIn() == expected ? trunc.getIn() : mlir::Value{};
  return {};
}

static bool
hasExactMultiplyAccumulatePayload(mlir::linalg::LinalgOp operation) {
  if (!operation || operation->getNumRegions() != 1 ||
      operation->getRegion(0).empty() ||
      operation.getRegionInputArgs().size() != 2 ||
      operation.getRegionOutputArgs().size() != 1)
    return false;
  mlir::Block &body = operation->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  mlir::BlockArgument lhs = operation.getRegionInputArgs()[0];
  mlir::BlockArgument rhs = operation.getRegionInputArgs()[1];
  mlir::BlockArgument accumulator = operation.getRegionOutputArgs()[0];
  mlir::Value product;
  mlir::Value sum;
  unsigned arithmeticOperations = 0;
  for (mlir::Operation &nested : body.without_terminator()) {
    if (mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp>(nested))
      continue;
    ++arithmeticOperations;
    if (auto multiply = mlir::dyn_cast<mlir::arith::MulFOp>(nested)) {
      if (!stripFloatCast(multiply.getLhs(), lhs) ||
          !stripFloatCast(multiply.getRhs(), rhs)) {
        if (!stripFloatCast(multiply.getLhs(), rhs) ||
            !stripFloatCast(multiply.getRhs(), lhs))
          return false;
      }
      product = multiply.getResult();
      continue;
    }
    if (auto multiply = mlir::dyn_cast<mlir::arith::MulIOp>(nested)) {
      if (!hasNoIntegerOverflowFlags(multiply) ||
          !matchesPair(multiply.getLhs(), multiply.getRhs(), lhs, rhs))
        return false;
      product = multiply.getResult();
      continue;
    }
    if (auto add = mlir::dyn_cast<mlir::arith::AddFOp>(nested)) {
      if (!product ||
          !matchesPair(add.getLhs(), add.getRhs(), product, accumulator))
        return false;
      sum = add.getResult();
      continue;
    }
    if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(nested)) {
      if (!hasNoIntegerOverflowFlags(add) || !product ||
          !matchesPair(add.getLhs(), add.getRhs(), product, accumulator))
        return false;
      sum = add.getResult();
      continue;
    }
    return false;
  }
  return arithmeticOperations == 2 && product && sum &&
         yield.getValues().front() == sum;
}

static std::optional<ComputeReduceKind>
getReductionKind(mlir::linalg::LinalgOp operation) {
  if (!operation || operation.getRegionInputArgs().size() != 1 ||
      operation.getRegionOutputArgs().size() != 1)
    return std::nullopt;
  mlir::Block &body = operation->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  auto nested = body.without_terminator();
  if (!yield || yield.getValues().size() != 1 ||
      std::distance(nested.begin(), nested.end()) != 1)
    return std::nullopt;
  mlir::Operation *combiner = &*nested.begin();
  if (!hasNoIntegerOverflowFlags(combiner) || combiner->getNumOperands() != 2 ||
      combiner->getNumResults() != 1 ||
      !matchesPair(combiner->getOperand(0), combiner->getOperand(1),
                   operation.getRegionInputArgs().front(),
                   operation.getRegionOutputArgs().front()) ||
      yield.getValues().front() != combiner->getResult(0))
    return std::nullopt;
  if (mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(combiner))
    return ComputeReduceKind::Sum;
  if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp>(combiner))
    return ComputeReduceKind::Max;
  if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinSIOp>(combiner))
    return ComputeReduceKind::Min;
  return std::nullopt;
}

static std::optional<ComputeElementwiseKind>
getFloatCompareKind(mlir::arith::CmpFPredicate predicate) {
  switch (predicate) {
  case mlir::arith::CmpFPredicate::OEQ:
    return ComputeElementwiseKind::Eq;
  case mlir::arith::CmpFPredicate::UNE:
    return ComputeElementwiseKind::Ne;
  case mlir::arith::CmpFPredicate::OLT:
    return ComputeElementwiseKind::Lt;
  case mlir::arith::CmpFPredicate::OLE:
    return ComputeElementwiseKind::Le;
  case mlir::arith::CmpFPredicate::OGT:
    return ComputeElementwiseKind::Gt;
  case mlir::arith::CmpFPredicate::OGE:
    return ComputeElementwiseKind::Ge;
  default:
    return std::nullopt;
  }
}

static std::optional<ComputeElementwiseKind>
getIntegerCompareKind(mlir::arith::CmpIPredicate predicate) {
  switch (predicate) {
  case mlir::arith::CmpIPredicate::eq:
    return ComputeElementwiseKind::Eq;
  case mlir::arith::CmpIPredicate::ne:
    return ComputeElementwiseKind::Ne;
  case mlir::arith::CmpIPredicate::slt:
    return ComputeElementwiseKind::Lt;
  case mlir::arith::CmpIPredicate::sle:
    return ComputeElementwiseKind::Le;
  case mlir::arith::CmpIPredicate::sgt:
    return ComputeElementwiseKind::Gt;
  case mlir::arith::CmpIPredicate::sge:
    return ComputeElementwiseKind::Ge;
  default:
    return std::nullopt;
  }
}

static std::optional<ComputeElementwiseKind>
getScalarElementwiseKind(mlir::Operation *operation) {
  if (!operation || operation->getNumResults() != 1 ||
      !hasNoIntegerOverflowFlags(operation))
    return std::nullopt;
  if (mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(operation))
    return ComputeElementwiseKind::Add;
  if (mlir::isa<mlir::arith::SubFOp, mlir::arith::SubIOp>(operation))
    return ComputeElementwiseKind::Sub;
  if (mlir::isa<mlir::arith::MulFOp, mlir::arith::MulIOp>(operation))
    return ComputeElementwiseKind::Mul;
  if (mlir::isa<mlir::arith::DivFOp, mlir::arith::DivSIOp>(operation))
    return ComputeElementwiseKind::Div;
  if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp>(operation))
    return ComputeElementwiseKind::Max;
  if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinSIOp>(operation))
    return ComputeElementwiseKind::Min;
  if (mlir::isa<mlir::arith::NegFOp>(operation))
    return ComputeElementwiseKind::Neg;
  if (mlir::isa<mlir::math::ExpOp>(operation))
    return ComputeElementwiseKind::Exp;
  if (mlir::isa<mlir::math::LogOp>(operation))
    return ComputeElementwiseKind::Ln;
  if (mlir::isa<mlir::math::SqrtOp>(operation))
    return ComputeElementwiseKind::Sqrt;
  if (mlir::isa<mlir::math::RsqrtOp>(operation))
    return ComputeElementwiseKind::Rsqrt;
  if (mlir::isa<mlir::math::TanhOp>(operation))
    return ComputeElementwiseKind::Tanh;
  if (auto power = mlir::dyn_cast<mlir::math::PowFOp>(operation)) {
    auto constant = power.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto attribute = constant
                         ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())
                         : mlir::FloatAttr{};
    if (attribute && attribute.getValue().isExactlyValue(2.0))
      return ComputeElementwiseKind::Square;
    return std::nullopt;
  }
  if (auto compare = mlir::dyn_cast<mlir::arith::CmpFOp>(operation))
    return getFloatCompareKind(compare.getPredicate());
  if (auto compare = mlir::dyn_cast<mlir::arith::CmpIOp>(operation))
    return getIntegerCompareKind(compare.getPredicate());
  if (mlir::isa<mlir::arith::SelectOp>(operation))
    return ComputeElementwiseKind::Select;
  return std::nullopt;
}

static bool isSupportedElementwiseBody(mlir::linalg::LinalgOp operation) {
  if (!operation || operation.getNumDpsInits() != 1 ||
      !hasOnlyParallelIterators(operation) || operation->getNumRegions() != 1 ||
      operation->getRegion(0).empty())
    return false;
  llvm::SmallVector<mlir::AffineMap, 4> maps = operation.getIndexingMapsArray();
  mlir::MemRefType resultType = getMemRef(operation.getDpsInits().front());
  if (!resultType ||
      maps.size() != static_cast<size_t>(operation.getNumDpsInputs() + 1) ||
      maps.back().getNumDims() != resultType.getRank() ||
      !maps.back().isIdentity())
    return false;
  mlir::Block &body = operation->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;
  llvm::SetVector<mlir::Value> captures;
  mlir::getUsedValuesDefinedAbove(operation->getRegion(0), captures);
  if (llvm::any_of(captures, [](mlir::Value value) {
        return mlir::isa<mlir::ShapedType>(value.getType());
      }))
    return false;
  for (mlir::Operation &nested : body.without_terminator()) {
    if (mlir::isOpTriviallyDead(&nested))
      continue;
    if (mlir::isa<mlir::arith::ConstantOp, mlir::arith::ExtFOp,
                  mlir::arith::TruncFOp>(nested))
      continue;
    if (!getScalarElementwiseKind(&nested))
      return false;
  }
  return true;
}

static mlir::Value getCapturedFillValue(mlir::linalg::LinalgOp operation) {
  if (!operation || operation.getNumDpsInputs() != 0 ||
      operation.getNumDpsInits() != 1 || !hasOnlyParallelIterators(operation) ||
      operation->getNumRegions() != 1 || operation->getRegion(0).empty())
    return {};
  mlir::Block &body = operation->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      mlir::isa<mlir::ShapedType>(yield.getValues().front().getType()) ||
      llvm::any_of(body.without_terminator(), [](mlir::Operation &nested) {
        return !mlir::isOpTriviallyDead(&nested);
      }))
    return {};
  mlir::Value value = yield.getValues().front();
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    if (argument.getOwner() == &body)
      return {};
  if (mlir::Operation *definition = value.getDefiningOp())
    if (definition->getBlock() == &body)
      return {};
  return value;
}

struct GemmDescriptor {
  bool rankTwo = false;
  GemmOrientation lhsOrientation = GemmOrientation::Normal;
  GemmOrientation rhsOrientation = GemmOrientation::Normal;
  int64_t batchCount = 0;
  llvm::SmallVector<int64_t, 4> lhsBatchDims;
  int64_t lhsMDim = 0;
  int64_t lhsContractingDim = 0;
  llvm::SmallVector<int64_t, 4> rhsBatchDims;
  int64_t rhsContractingDim = 0;
  int64_t rhsNDim = 0;
  llvm::SmallVector<int64_t, 4> resultBatchDims;
  int64_t resultMDim = 0;
  int64_t resultNDim = 0;
};

struct ConvDescriptor {
  llvm::SmallVector<int64_t, 4> inputToNHWC;
  llvm::SmallVector<int64_t, 4> weightToXYOI;
  llvm::SmallVector<int64_t, 4> outputToNHWC;
  llvm::SmallVector<int64_t, 4> outputFromNHWC;
  llvm::SmallVector<int64_t, 4> pads;
  llvm::SmallVector<int64_t, 4> unpads;
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> dilations;
};

static std::optional<unsigned> findOperandDim(mlir::AffineMap map,
                                              unsigned loop) {
  std::optional<unsigned> result;
  for (auto [dimension, expression] : llvm::enumerate(map.getResults())) {
    auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dim || dim.getPosition() != loop)
      continue;
    if (result)
      return std::nullopt;
    result = static_cast<unsigned>(dimension);
  }
  return result;
}

static std::optional<unsigned> findMapResult(mlir::AffineMap map,
                                             mlir::AffineExpr expected) {
  expected =
      mlir::simplifyAffineExpr(expected, map.getNumDims(), map.getNumSymbols());
  std::optional<unsigned> result;
  for (auto [index, expression] : llvm::enumerate(map.getResults())) {
    mlir::AffineExpr simplified = mlir::simplifyAffineExpr(
        expression, map.getNumDims(), map.getNumSymbols());
    if (simplified != expected)
      continue;
    if (result)
      return std::nullopt;
    result = index;
  }
  return result;
}

static bool isPermutation(llvm::ArrayRef<int64_t> permutation, int64_t rank) {
  if (permutation.size() != static_cast<size_t>(rank))
    return false;
  llvm::SmallVector<bool, 6> seen(rank, false);
  for (int64_t dimension : permutation) {
    if (dimension < 0 || dimension >= rank || seen[dimension])
      return false;
    seen[dimension] = true;
  }
  return true;
}

static mlir::FailureOr<ConvDescriptor>
buildConvDescriptor(mlir::linalg::LinalgOp operation) {
  if (!operation || operation.getNumDpsInputs() != 2 ||
      operation.getNumDpsInits() != 1 ||
      !hasExactMultiplyAccumulatePayload(operation))
    return mlir::failure();
  mlir::FailureOr<mlir::linalg::ConvolutionDimensions> inferred =
      mlir::linalg::inferConvolutionDims(operation);
  if (mlir::failed(inferred) || inferred->batch.size() != 1 ||
      inferred->outputImage.size() != 2 ||
      inferred->outputChannel.size() != 1 || inferred->filterLoop.size() != 2 ||
      inferred->inputChannel.size() != 1 || !inferred->depth.empty() ||
      inferred->strides.size() != 2 || inferred->dilations.size() != 2)
    return mlir::failure();
  mlir::MemRefType input = getMemRef(operation.getDpsInputs()[0]);
  mlir::MemRefType weight = getMemRef(operation.getDpsInputs()[1]);
  mlir::MemRefType output = getMemRef(operation.getDpsInits()[0]);
  if (!isStaticPositive(input) || !isStaticPositive(weight) ||
      !isStaticPositive(output) || input.getRank() != 4 ||
      weight.getRank() != 4 || output.getRank() != 4)
    return mlir::failure();
  llvm::SmallVector<mlir::AffineMap, 3> maps = operation.getIndexingMapsArray();
  if (maps.size() != 3 || llvm::any_of(maps, [](mlir::AffineMap map) {
        return map.getNumSymbols() != 0 || map.getNumResults() != 4;
      }))
    return mlir::failure();
  mlir::MLIRContext *context = operation.getContext();
  auto loopExpr = [&](unsigned loop) {
    return mlir::getAffineDimExpr(loop, context);
  };
  auto findLoop = [&](mlir::AffineMap map, unsigned loop) {
    return findMapResult(map, loopExpr(loop));
  };
  std::optional<unsigned> inputBatch =
      findLoop(maps[0], inferred->batch.front());
  std::optional<unsigned> inputChannel =
      findLoop(maps[0], inferred->inputChannel.front());
  std::optional<unsigned> weightOutputChannel =
      findLoop(maps[1], inferred->outputChannel.front());
  std::optional<unsigned> weightInputChannel =
      findLoop(maps[1], inferred->inputChannel.front());
  std::optional<unsigned> outputBatch =
      findLoop(maps[2], inferred->batch.front());
  std::optional<unsigned> outputChannel =
      findLoop(maps[2], inferred->outputChannel.front());
  if (!inputBatch || !inputChannel || !weightOutputChannel ||
      !weightInputChannel || !outputBatch || !outputChannel)
    return mlir::failure();

  llvm::SmallVector<unsigned, 2> inputSpatial;
  llvm::SmallVector<unsigned, 2> weightSpatial;
  llvm::SmallVector<unsigned, 2> outputSpatial;
  for (unsigned index = 0; index < 2; ++index) {
    mlir::AffineExpr window =
        loopExpr(inferred->outputImage[index]) * inferred->strides[index] +
        loopExpr(inferred->filterLoop[index]) * inferred->dilations[index];
    std::optional<unsigned> inputDimension = findMapResult(maps[0], window);
    std::optional<unsigned> weightDimension =
        findLoop(maps[1], inferred->filterLoop[index]);
    std::optional<unsigned> outputDimension =
        findLoop(maps[2], inferred->outputImage[index]);
    if (!inputDimension || !weightDimension || !outputDimension)
      return mlir::failure();
    inputSpatial.push_back(*inputDimension);
    weightSpatial.push_back(*weightDimension);
    outputSpatial.push_back(*outputDimension);
  }

  ConvDescriptor descriptor;
  descriptor.inputToNHWC = {static_cast<int64_t>(*inputBatch),
                            static_cast<int64_t>(inputSpatial[0]),
                            static_cast<int64_t>(inputSpatial[1]),
                            static_cast<int64_t>(*inputChannel)};
  descriptor.weightToXYOI = {static_cast<int64_t>(weightSpatial[1]),
                             static_cast<int64_t>(weightSpatial[0]),
                             static_cast<int64_t>(*weightOutputChannel),
                             static_cast<int64_t>(*weightInputChannel)};
  descriptor.outputToNHWC = {static_cast<int64_t>(*outputBatch),
                             static_cast<int64_t>(outputSpatial[0]),
                             static_cast<int64_t>(outputSpatial[1]),
                             static_cast<int64_t>(*outputChannel)};
  if (!isPermutation(descriptor.inputToNHWC, 4) ||
      !isPermutation(descriptor.weightToXYOI, 4) ||
      !isPermutation(descriptor.outputToNHWC, 4))
    return mlir::failure();
  descriptor.outputFromNHWC.assign(4, -1);
  for (auto [canonicalDimension, sourceDimension] :
       llvm::enumerate(descriptor.outputToNHWC))
    descriptor.outputFromNHWC[sourceDimension] = canonicalDimension;
  descriptor.strides.assign(inferred->strides.begin(), inferred->strides.end());
  descriptor.dilations.assign(inferred->dilations.begin(),
                              inferred->dilations.end());

  auto permutedShape = [](mlir::MemRefType type,
                          llvm::ArrayRef<int64_t> order) {
    llvm::SmallVector<int64_t, 4> shape;
    for (int64_t dimension : order)
      shape.push_back(type.getDimSize(dimension));
    return shape;
  };
  llvm::SmallVector<int64_t, 4> inputShape =
      permutedShape(input, descriptor.inputToNHWC);
  llvm::SmallVector<int64_t, 4> weightShape =
      permutedShape(weight, descriptor.weightToXYOI);
  llvm::SmallVector<int64_t, 4> outputShape =
      permutedShape(output, descriptor.outputToNHWC);
  auto validOutput = [](int64_t inputExtent, int64_t kernel, int64_t stride,
                        int64_t dilation) -> std::optional<int64_t> {
    if (inputExtent <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 ||
        kernel - 1 > (std::numeric_limits<int64_t>::max() - 1) / dilation)
      return std::nullopt;
    int64_t effectiveKernel = (kernel - 1) * dilation + 1;
    if (inputExtent < effectiveKernel)
      return std::nullopt;
    return (inputExtent - effectiveKernel) / stride + 1;
  };
  std::optional<int64_t> expectedH =
      validOutput(inputShape[1], weightShape[1], descriptor.strides[0],
                  descriptor.dilations[0]);
  std::optional<int64_t> expectedW =
      validOutput(inputShape[2], weightShape[0], descriptor.strides[1],
                  descriptor.dilations[1]);
  if (!expectedH || !expectedW || outputShape[0] != inputShape[0] ||
      inputShape[3] != weightShape[3] || outputShape[3] != weightShape[2] ||
      outputShape[1] != *expectedH || outputShape[2] != *expectedW)
    return mlir::failure();
  descriptor.pads.assign(4, 0);
  descriptor.unpads.assign(4, 0);
  return descriptor;
}

static mlir::FailureOr<GemmDescriptor>
buildGemmDescriptor(mlir::linalg::LinalgOp operation) {
  if (!operation || operation.getNumDpsInputs() != 2 ||
      operation.getNumDpsInits() != 1 ||
      !hasExactMultiplyAccumulatePayload(operation))
    return mlir::failure();
  mlir::FailureOr<mlir::linalg::ContractionDimensions> dimensions =
      mlir::linalg::inferContractionDims(operation);
  if (mlir::failed(dimensions) || dimensions->m.size() != 1 ||
      dimensions->n.size() != 1 || dimensions->k.size() != 1)
    return mlir::failure();
  llvm::SmallVector<mlir::AffineMap, 3> maps = operation.getIndexingMapsArray();
  if (maps.size() != 3 || llvm::any_of(maps, [](mlir::AffineMap map) {
        return map.getNumSymbols() != 0 || !map.isProjectedPermutation();
      }))
    return mlir::failure();
  mlir::MemRefType lhs = getMemRef(operation.getDpsInputs()[0]);
  mlir::MemRefType rhs = getMemRef(operation.getDpsInputs()[1]);
  mlir::MemRefType result = getMemRef(operation.getDpsInits()[0]);
  if (!isStaticPositive(lhs) || !isStaticPositive(rhs) ||
      !isStaticPositive(result))
    return mlir::failure();

  std::optional<unsigned> lhsM = findOperandDim(maps[0], dimensions->m[0]);
  std::optional<unsigned> lhsK = findOperandDim(maps[0], dimensions->k[0]);
  std::optional<unsigned> rhsK = findOperandDim(maps[1], dimensions->k[0]);
  std::optional<unsigned> rhsN = findOperandDim(maps[1], dimensions->n[0]);
  std::optional<unsigned> resultM = findOperandDim(maps[2], dimensions->m[0]);
  std::optional<unsigned> resultN = findOperandDim(maps[2], dimensions->n[0]);
  if (!lhsM || !lhsK || !rhsK || !rhsN || !resultM || !resultN)
    return mlir::failure();

  GemmDescriptor descriptor;
  descriptor.rankTwo =
      lhs.getRank() == 2 && rhs.getRank() == 2 && result.getRank() == 2;
  descriptor.lhsMDim = *lhsM;
  descriptor.lhsContractingDim = *lhsK;
  descriptor.rhsContractingDim = *rhsK;
  descriptor.rhsNDim = *rhsN;
  descriptor.resultMDim = *resultM;
  descriptor.resultNDim = *resultN;
  if (descriptor.rankTwo) {
    if (!dimensions->batch.empty() || *resultM != 0 || *resultN != 1)
      return mlir::failure();
    if (*lhsM == 0 && *lhsK == 1)
      descriptor.lhsOrientation = GemmOrientation::Normal;
    else if (*lhsM == 1 && *lhsK == 0)
      descriptor.lhsOrientation = GemmOrientation::Transpose;
    else
      return mlir::failure();
    if (*rhsK == 0 && *rhsN == 1)
      descriptor.rhsOrientation = GemmOrientation::Normal;
    else if (*rhsK == 1 && *rhsN == 0)
      descriptor.rhsOrientation = GemmOrientation::Transpose;
    else
      return mlir::failure();
    return descriptor;
  }

  descriptor.batchCount = 1;
  for (unsigned loop : dimensions->batch) {
    std::optional<unsigned> lhsBatch = findOperandDim(maps[0], loop);
    std::optional<unsigned> rhsBatch = findOperandDim(maps[1], loop);
    std::optional<unsigned> resultBatch = findOperandDim(maps[2], loop);
    if (!lhsBatch || !rhsBatch || !resultBatch)
      return mlir::failure();
    int64_t extent = result.getDimSize(*resultBatch);
    if (extent <= 0 ||
        descriptor.batchCount > std::numeric_limits<int64_t>::max() / extent)
      return mlir::failure();
    descriptor.batchCount *= extent;
    descriptor.lhsBatchDims.push_back(*lhsBatch);
    descriptor.rhsBatchDims.push_back(*rhsBatch);
    descriptor.resultBatchDims.push_back(*resultBatch);
  }
  if (descriptor.lhsBatchDims.empty())
    return mlir::failure();
  return descriptor;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getReductionInputDimensions(mlir::linalg::LinalgOp operation) {
  if (!operation || operation.getNumDpsInputs() != 1 ||
      operation.getNumDpsInits() != 1)
    return mlir::failure();
  llvm::SmallVector<mlir::AffineMap, 2> maps = operation.getIndexingMapsArray();
  if (maps.size() != 2 || maps[0].getNumSymbols() != 0 ||
      !maps[0].isProjectedPermutation())
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> result;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iterators =
      operation.getIteratorTypesArray();
  for (auto [inputDimension, expression] :
       llvm::enumerate(maps[0].getResults())) {
    auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dim || dim.getPosition() >= iterators.size())
      return mlir::failure();
    if (iterators[dim.getPosition()] == mlir::utils::IteratorType::reduction)
      result.push_back(inputDimension);
  }
  if (result.empty())
    return mlir::failure();
  return result;
}

static mlir::LogicalResult preflight(mlir::ModuleOp module,
                                     llvm::SmallVectorImpl<LoweringPlan> &plans,
                                     std::string &detail) {
  bool failed = false;
  module.walk([&](mlir::linalg::LinalgOp operation) {
    if (failed)
      return mlir::WalkResult::interrupt();
    if (!operation->getParentOfType<TileRegionOp>()) {
      detail = "executable Linalg operation is not owned by a TileRegion";
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    if (operation->getNumResults() != 0 ||
        llvm::any_of(operation->getOperandTypes(), [](mlir::Type type) {
          return !mlir::isa<mlir::MemRefType>(type) &&
                 mlir::isa<mlir::ShapedType>(type);
        })) {
      detail = "structured-to-Tile requires bufferized Linalg current IR";
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa<mlir::linalg::FillOp>(operation.getOperation())) {
      if (operation.getNumDpsInputs() != 1 || operation.getNumDpsInits() != 1 ||
          !getMemRef(operation.getDpsInits().front())) {
        detail = "linalg.fill has no typed scalar/destination contract";
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      plans.push_back({operation, LoweringKind::Fill});
      return mlir::WalkResult::advance();
    }
    if (getCapturedFillValue(operation)) {
      plans.push_back({operation, LoweringKind::CapturedFill});
      return mlir::WalkResult::advance();
    }
    if (mlir::succeeded(buildGemmDescriptor(operation))) {
      plans.push_back({operation, LoweringKind::Contraction});
      return mlir::WalkResult::advance();
    }
    if (mlir::succeeded(buildConvDescriptor(operation))) {
      plans.push_back({operation, LoweringKind::Convolution});
      return mlir::WalkResult::advance();
    }
    if (hasReductionIterator(operation)) {
      if (!getReductionKind(operation) ||
          mlir::failed(getReductionInputDimensions(operation))) {
        detail = "reduction Linalg operation has no exact Tile reduce form";
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      plans.push_back({operation, LoweringKind::Reduction});
      return mlir::WalkResult::advance();
    }
    if (!isSupportedElementwiseBody(operation)) {
      detail = "parallel Linalg operation has no exact Tile expression form";
      if (operation->getNumRegions() == 1 && !operation->getRegion(0).empty())
        for (mlir::Operation &nested :
             operation->getRegion(0).front().without_terminator())
          if (!mlir::isa<mlir::arith::ConstantOp, mlir::arith::ExtFOp,
                         mlir::arith::TruncFOp>(nested) &&
              !mlir::isOpTriviallyDead(&nested) &&
              !getScalarElementwiseKind(&nested)) {
            detail += ": unsupported scalar op " +
                      nested.getName().getStringRef().str();
            break;
          }
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    plans.push_back({operation, LoweringKind::Elementwise});
    return mlir::WalkResult::advance();
  });
  return mlir::success(!failed);
}

static ComputeFillOp
findLastFillBefore(mlir::Value destination, mlir::Operation *operation,
                   llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {
  ComputeFillOp selected;
  if (!destination || !operation)
    return selected;
  for (mlir::Operation *user : destination.getUsers()) {
    auto fill = mlir::dyn_cast<ComputeFillOp>(user);
    if (!fill || fill->getBlock() != operation->getBlock() ||
        !fill->isBeforeInBlock(operation))
      continue;
    if (!selected || selected->isBeforeInBlock(fill))
      selected = fill;
  }
  if (selected)
    return selected;
  auto materialize = destination.getDefiningOp<LayoutMaterializeOp>();
  if (materialize && visited.insert(materialize).second)
    return findLastFillBefore(materialize.getSource(), operation, visited);
  return selected;
}

static mlir::LogicalResult
lowerCapturedFill(mlir::linalg::LinalgOp operation, mlir::IRRewriter &rewriter,
                  StructuredToTileStatistics &statistics) {
  mlir::Value value = getCapturedFillValue(operation);
  mlir::Value destination = operation.getDpsInits().front();
  if (!value || !getMemRef(destination))
    return mlir::failure();
  rewriter.setInsertionPoint(operation);
  rewriter.create<ComputeFillOp>(operation.getLoc(), destination, value,
                                 FillDomainAttr{});
  rewriter.eraseOp(operation);
  ++statistics.fills;
  return mlir::success();
}

static ComputeFillOp findLastFillBefore(mlir::Value destination,
                                        mlir::Operation *operation) {
  llvm::SmallPtrSet<mlir::Operation *, 4> visited;
  return findLastFillBefore(destination, operation, visited);
}

static bool isPositiveZero(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  mlir::Attribute attribute =
      constant ? constant.getValue() : mlir::Attribute{};
  if (auto floating = mlir::dyn_cast_or_null<mlir::FloatAttr>(attribute))
    return floating.getValue().isZero() && !floating.getValue().isNegative();
  if (auto integer = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute))
    return integer.getValue().isZero();
  return false;
}

static void replaceDominatedUses(mlir::ModuleOp module, mlir::Value oldValue,
                                 mlir::Value newValue,
                                 mlir::Operation *sourceOperation) {
  if (oldValue == newValue)
    return;
  mlir::DominanceInfo dominance(module);
  llvm::SmallVector<mlir::OpOperand *, 8> replacements;
  for (mlir::OpOperand &use : oldValue.getUses())
    if (use.getOwner() != sourceOperation &&
        dominance.properlyDominates(newValue, use.getOwner()))
      replacements.push_back(&use);
  for (mlir::OpOperand *use : replacements)
    use->set(newValue);
}

static mlir::Value
publishComputedValue(mlir::ModuleOp module, mlir::Operation *sourceOperation,
                     mlir::Value destination, mlir::Value computed,
                     mlir::IRRewriter &rewriter,
                     StructuredToTileStatistics &statistics) {
  if (destination.getType() == computed.getType()) {
    replaceDominatedUses(module, destination, computed, sourceOperation);
    return computed;
  }
  mlir::MemRefType destinationType = getMemRef(destination);
  mlir::MemRefType computedType = getMemRef(computed);
  if (!destinationType || !computedType ||
      destinationType.getShape() != computedType.getShape() ||
      destinationType.getElementType() != computedType.getElementType() ||
      destinationType.getMemorySpace() != computedType.getMemorySpace())
    return {};
  rewriter.create<MoveCopyIntoOp>(sourceOperation->getLoc(), computed,
                                  destination);
  ++statistics.passthroughMovements;
  return destination;
}

static mlir::ArrayAttr
getIndexingMapsAttr(mlir::OpBuilder &builder,
                    llvm::ArrayRef<mlir::AffineMap> maps) {
  llvm::SmallVector<mlir::Attribute, 4> attributes;
  attributes.reserve(maps.size());
  for (mlir::AffineMap map : maps)
    attributes.push_back(mlir::AffineMapAttr::get(map));
  return builder.getArrayAttr(attributes);
}

static mlir::MemRefType getShapedType(mlir::MemRefType source,
                                      llvm::ArrayRef<int64_t> shape) {
  return mlir::MemRefType::get(shape, source.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               source.getMemorySpace());
}

static bool isIdentityPermutation(llvm::ArrayRef<int64_t> permutation) {
  return llvm::all_of(llvm::enumerate(permutation), [](auto indexed) {
    return static_cast<int64_t>(indexed.index()) == indexed.value();
  });
}

static mlir::FailureOr<mlir::Value>
permuteBuffer(mlir::Value source, llvm::ArrayRef<int64_t> resultToSource,
              mlir::IRRewriter &rewriter, mlir::Location location,
              StructuredToTileStatistics &statistics) {
  mlir::MemRefType sourceType = getMemRef(source);
  if (!sourceType ||
      resultToSource.size() != static_cast<size_t>(sourceType.getRank()))
    return mlir::failure();
  llvm::SmallVector<bool, 6> seen(sourceType.getRank(), false);
  llvm::SmallVector<int64_t, 6> shape;
  for (int64_t sourceDimension : resultToSource) {
    if (sourceDimension < 0 || sourceDimension >= sourceType.getRank() ||
        seen[sourceDimension])
      return mlir::failure();
    seen[sourceDimension] = true;
    shape.push_back(sourceType.getDimSize(sourceDimension));
  }
  if (isIdentityPermutation(resultToSource))
    return source;
  auto transpose = rewriter.create<MoveTransposeOp>(
      location, getShapedType(sourceType, shape), source,
      rewriter.getDenseI64ArrayAttr(resultToSource));
  ++statistics.passthroughMovements;
  return transpose.getResult();
}

static mlir::FailureOr<mlir::Value>
reshapeBuffer(mlir::Value source, llvm::ArrayRef<int64_t> shape,
              mlir::IRRewriter &rewriter, mlir::Location location,
              StructuredToTileStatistics &statistics) {
  mlir::MemRefType sourceType = getMemRef(source);
  if (!sourceType)
    return mlir::failure();
  if (sourceType.getShape() == shape)
    return source;
  mlir::MemRefType resultType = getShapedType(sourceType, shape);
  if (mlir::succeeded(
          analysis::TransferRealizability::proveStaticReshapeMetadataView(
              sourceType, resultType, /*destinationMayWrite=*/true))) {
    auto reshape = rewriter.create<ViewReshapeOp>(location, resultType, source);
    return reshape.getResult();
  }
  auto movement = rewriter.create<MoveReshapeOp>(location, resultType, source);
  ++statistics.passthroughMovements;
  return movement.getResult();
}

static llvm::SmallVector<int64_t, 6>
invertPermutation(llvm::ArrayRef<int64_t> resultToSource) {
  llvm::SmallVector<int64_t, 6> inverse(resultToSource.size(), -1);
  for (auto [resultDimension, sourceDimension] :
       llvm::enumerate(resultToSource))
    if (sourceDimension >= 0 &&
        sourceDimension < static_cast<int64_t>(inverse.size()))
      inverse[sourceDimension] = resultDimension;
  return inverse;
}

static mlir::AffineMap getIdentityMap(mlir::MLIRContext *context,
                                      int64_t rank) {
  return mlir::AffineMap::getMultiDimIdentityMap(rank, context);
}

static mlir::FailureOr<ExprValue>
materializeExprMap(ExprValue value, mlir::MemRefType targetShapeType,
                   mlir::IRRewriter &rewriter, mlir::Location location,
                   StructuredToTileStatistics &statistics) {
  mlir::MemRefType sourceType = getMemRef(value.buffer);
  if (!sourceType || !targetShapeType ||
      sourceType.getElementType() != targetShapeType.getElementType() ||
      value.indexingMap.getNumDims() != targetShapeType.getRank() ||
      value.indexingMap.getNumSymbols() != 0 ||
      value.indexingMap.getNumResults() != sourceType.getRank() ||
      !value.indexingMap.isProjectedPermutation())
    return mlir::failure();
  mlir::AffineMap identity =
      getIdentityMap(rewriter.getContext(), targetShapeType.getRank());
  if (value.indexingMap.isIdentity() &&
      sourceType.getShape() == targetShapeType.getShape())
    return ExprValue{value.buffer, identity};

  mlir::MemRefType movedType = changeShapeAndElementType(
      sourceType, targetShapeType.getShape(), sourceType.getElementType());
  mlir::Value moved;
  if (sourceType.getRank() == targetShapeType.getRank()) {
    llvm::SmallVector<int64_t, 4> resultToSource(sourceType.getRank(), -1);
    for (auto [sourceDimension, expression] :
         llvm::enumerate(value.indexingMap.getResults())) {
      auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!dim || dim.getPosition() >= resultToSource.size() ||
          resultToSource[dim.getPosition()] != -1)
        return mlir::failure();
      resultToSource[dim.getPosition()] = sourceDimension;
    }
    if (llvm::is_contained(resultToSource, -1))
      return mlir::failure();
    auto transpose = rewriter.create<MoveTransposeOp>(
        location, movedType, value.buffer,
        rewriter.getDenseI64ArrayAttr(resultToSource));
    moved = transpose.getResult();
  } else {
    llvm::SmallVector<int64_t, 4> dimensions;
    for (mlir::AffineExpr expression : value.indexingMap.getResults()) {
      auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!dim)
        return mlir::failure();
      dimensions.push_back(dim.getPosition());
    }
    auto broadcast = rewriter.create<MoveBroadcastOp>(
        location, movedType, value.buffer,
        rewriter.getDenseI64ArrayAttr(dimensions));
    moved = broadcast.getResult();
  }
  ++statistics.passthroughMovements;
  return ExprValue{moved, identity};
}

static mlir::FailureOr<mlir::Value>
materializeBufferAs(mlir::Value source, mlir::MemRefType targetType,
                    mlir::IRRewriter &rewriter, mlir::Location location,
                    StructuredToTileStatistics &statistics) {
  mlir::MemRefType sourceType = getMemRef(source);
  MemoryAttr sourceMemory =
      sourceType ? getWaferMemoryAttr(sourceType) : MemoryAttr{};
  MemoryAttr targetMemory =
      targetType ? getWaferMemoryAttr(targetType) : MemoryAttr{};
  if (!sourceType || !targetType || !sourceMemory || !targetMemory ||
      sourceType.getShape() != targetType.getShape() ||
      sourceType.getElementType() != targetType.getElementType() ||
      sourceMemory.getSpace() != targetMemory.getSpace())
    return mlir::failure();
  if (sourceType == targetType)
    return source;
  mlir::Value result;
  if (sourceMemory.getLayout() == targetMemory.getLayout())
    result =
        rewriter.create<MoveCopyOp>(location, targetType, source).getResult();
  else
    result = rewriter.create<LayoutMaterializeOp>(location, targetType, source)
                 .getResult();
  ++statistics.passthroughMovements;
  return result;
}

static mlir::FailureOr<mlir::Value>
materializeExprAs(ExprValue value, mlir::MemRefType targetType,
                  mlir::IRRewriter &rewriter, mlir::Location location,
                  StructuredToTileStatistics &statistics) {
  mlir::FailureOr<ExprValue> mapped =
      materializeExprMap(value, targetType, rewriter, location, statistics);
  if (mlir::failed(mapped))
    return mlir::failure();
  mlir::Value result = mapped->buffer;
  if (result.getType() == targetType)
    return result;
  return materializeBufferAs(result, targetType, rewriter, location,
                             statistics);
}

static mlir::FailureOr<mlir::Value>
createConvert(mlir::Value source, mlir::MemRefType resultType,
              mlir::IRRewriter &rewriter, mlir::Location location,
              StructuredToTileStatistics &statistics) {
  mlir::MemRefType sourceType = getMemRef(source);
  if (!sourceType || !resultType ||
      sourceType.getShape() != resultType.getShape() ||
      sourceType.getElementType() == resultType.getElementType())
    return mlir::failure();
  analysis::IndexRelationResult identity =
      analysis::IndexRelation::identity(resultType.getShape());
  if (!identity.isExact())
    return mlir::failure();
  if (mlir::succeeded(analysis::TransferRealizability::provePhysicalTraversal(
          sourceType, resultType, resultType.getShape(), *identity.get(),
          *identity.get()))) {
    auto converted =
        rewriter.create<ComputeConvertOp>(location, resultType, source);
    ++statistics.converts;
    return converted.getResult();
  }

  auto tensorType = [&](mlir::MemRefType type) {
    return mlir::MemRefType::get(type.getShape(), type.getElementType(),
                                 mlir::MemRefLayoutAttrInterface{},
                                 MemoryAttr::get(type.getContext(),
                                                 MemorySpace::SPM,
                                                 MemLayout::Tensor));
  };
  mlir::MemRefType tensorSourceType = tensorType(sourceType);
  mlir::MemRefType tensorResultType = tensorType(resultType);
  mlir::Value tensorSource = source;
  if (sourceType != tensorSourceType) {
    mlir::FailureOr<mlir::Value> materialized = materializeBufferAs(
        tensorSource, tensorSourceType, rewriter, location, statistics);
    if (mlir::failed(materialized))
      return mlir::failure();
    tensorSource = *materialized;
  }
  auto converted = rewriter.create<ComputeConvertOp>(location, tensorResultType,
                                                     tensorSource);
  ++statistics.converts;
  if (tensorResultType == resultType)
    return converted.getResult();
  return materializeBufferAs(converted.getResult(), resultType, rewriter,
                             location, statistics);
}

static mlir::FailureOr<ExprValue>
createScalarFill(mlir::Value scalar, mlir::MemRefType resultShape,
                 mlir::IRRewriter &rewriter, mlir::Location location,
                 StructuredToTileStatistics &statistics) {
  if (!scalar || mlir::isa<mlir::ShapedType>(scalar.getType()))
    return mlir::failure();
  auto tensorMemory = MemoryAttr::get(rewriter.getContext(), MemorySpace::SPM,
                                      MemLayout::Tensor);
  auto type = mlir::MemRefType::get(resultShape.getShape(), scalar.getType(),
                                    resultShape.getLayout(), tensorMemory);
  auto allocation = rewriter.create<mlir::memref::AllocOp>(location, type);
  rewriter.create<ComputeFillOp>(location, allocation.getResult(), scalar,
                                 FillDomainAttr{});
  ++statistics.fills;
  return ExprValue{allocation.getResult(),
                   getIdentityMap(rewriter.getContext(), type.getRank())};
}

static mlir::FailureOr<ExprValue>
createElementwise(ComputeElementwiseKind kind, llvm::ArrayRef<ExprValue> inputs,
                  mlir::Type resultElementType, mlir::MemRefType resultShape,
                  mlir::IRRewriter &rewriter, mlir::Location location,
                  StructuredToTileStatistics &statistics) {
  llvm::SmallVector<mlir::Value, 3> buffers;
  llvm::SmallVector<mlir::AffineMap, 4> maps;
  for (ExprValue input : inputs) {
    if (!input.buffer)
      return mlir::failure();
    buffers.push_back(input.buffer);
    maps.push_back(input.indexingMap);
  }
  mlir::MemRefType resultType =
      changeElementType(resultShape, resultElementType);
  mlir::AffineMap identity =
      getIdentityMap(rewriter.getContext(), resultType.getRank());
  maps.push_back(identity);
  auto operation = rewriter.create<ComputeElementwiseOp>(
      location, resultType,
      ComputeElementwiseKindAttr::get(rewriter.getContext(), kind), buffers,
      getIndexingMapsAttr(rewriter, maps));
  ++statistics.elementwiseOperations;
  return ExprValue{operation.getResult(), identity};
}

static mlir::LogicalResult lowerFill(mlir::linalg::LinalgOp operation,
                                     mlir::IRRewriter &rewriter,
                                     StructuredToTileStatistics &statistics) {
  rewriter.setInsertionPoint(operation);
  rewriter.create<ComputeFillOp>(
      operation.getLoc(), operation.getDpsInits().front(),
      operation.getDpsInputs().front(), FillDomainAttr{});
  rewriter.eraseOp(operation);
  ++statistics.fills;
  return mlir::success();
}

static mlir::LogicalResult
lowerContraction(mlir::ModuleOp module, mlir::linalg::LinalgOp operation,
                 mlir::IRRewriter &rewriter,
                 StructuredToTileStatistics &statistics) {
  mlir::FailureOr<GemmDescriptor> descriptor = buildGemmDescriptor(operation);
  if (mlir::failed(descriptor))
    return mlir::failure();
  mlir::Value lhs = operation.getDpsInputs()[0];
  mlir::Value rhs = operation.getDpsInputs()[1];
  mlir::Value destination = operation.getDpsInits()[0];
  mlir::MemRefType destinationType = getMemRef(destination);
  if (!destinationType)
    return mlir::failure();
  mlir::MemRefType resultType = getOwnedType(destinationType);
  rewriter.setInsertionPoint(operation);
  for (mlir::Value *input : {&lhs, &rhs}) {
    mlir::MemRefType inputType = getMemRef(*input);
    if (!inputType)
      return mlir::failure();
    if (inputType.getElementType() == resultType.getElementType())
      continue;
    mlir::MemRefType convertedType =
        changeElementType(inputType, resultType.getElementType());
    mlir::FailureOr<mlir::Value> converted = createConvert(
        *input, convertedType, rewriter, operation.getLoc(), statistics);
    if (mlir::failed(converted))
      return mlir::failure();
    *input = *converted;
  }

  GemmOrientationAttr lhsOrientation;
  GemmOrientationAttr rhsOrientation;
  mlir::IntegerAttr batchCount;
  mlir::DenseI64ArrayAttr lhsBatchDims;
  mlir::IntegerAttr lhsMDim;
  mlir::IntegerAttr lhsContractingDim;
  mlir::DenseI64ArrayAttr rhsBatchDims;
  mlir::IntegerAttr rhsContractingDim;
  mlir::IntegerAttr rhsNDim;
  mlir::DenseI64ArrayAttr resultBatchDims;
  mlir::IntegerAttr resultMDim;
  mlir::IntegerAttr resultNDim;
  mlir::MemRefType gemmResultType = resultType;
  llvm::SmallVector<int64_t, 6> resultCanonicalOrder;
  if (descriptor->rankTwo) {
    if (descriptor->lhsOrientation != GemmOrientation::Normal ||
        descriptor->rhsOrientation != GemmOrientation::Normal) {
      lhsOrientation = GemmOrientationAttr::get(rewriter.getContext(),
                                                descriptor->lhsOrientation);
      rhsOrientation = GemmOrientationAttr::get(rewriter.getContext(),
                                                descriptor->rhsOrientation);
    }
  } else {
    llvm::SmallVector<int64_t, 6> lhsOrder(descriptor->lhsBatchDims.begin(),
                                           descriptor->lhsBatchDims.end());
    lhsOrder.push_back(descriptor->lhsMDim);
    lhsOrder.push_back(descriptor->lhsContractingDim);
    llvm::SmallVector<int64_t, 6> rhsOrder(descriptor->rhsBatchDims.begin(),
                                           descriptor->rhsBatchDims.end());
    rhsOrder.push_back(descriptor->rhsContractingDim);
    rhsOrder.push_back(descriptor->rhsNDim);
    resultCanonicalOrder.assign(descriptor->resultBatchDims.begin(),
                                descriptor->resultBatchDims.end());
    resultCanonicalOrder.push_back(descriptor->resultMDim);
    resultCanonicalOrder.push_back(descriptor->resultNDim);
    mlir::FailureOr<mlir::Value> canonicalLhs =
        permuteBuffer(lhs, lhsOrder, rewriter, operation.getLoc(), statistics);
    mlir::FailureOr<mlir::Value> canonicalRhs =
        permuteBuffer(rhs, rhsOrder, rewriter, operation.getLoc(), statistics);
    if (mlir::failed(canonicalLhs) || mlir::failed(canonicalRhs))
      return mlir::failure();
    lhs = *canonicalLhs;
    rhs = *canonicalRhs;
    mlir::MemRefType canonicalLhsType = getMemRef(lhs);
    mlir::MemRefType canonicalRhsType = getMemRef(rhs);
    mlir::FailureOr<mlir::Value> flattenedLhs = reshapeBuffer(
        lhs,
        {descriptor->batchCount,
         canonicalLhsType.getDimSize(canonicalLhsType.getRank() - 2),
         canonicalLhsType.getDimSize(canonicalLhsType.getRank() - 1)},
        rewriter, operation.getLoc(), statistics);
    mlir::FailureOr<mlir::Value> flattenedRhs = reshapeBuffer(
        rhs,
        {descriptor->batchCount,
         canonicalRhsType.getDimSize(canonicalRhsType.getRank() - 2),
         canonicalRhsType.getDimSize(canonicalRhsType.getRank() - 1)},
        rewriter, operation.getLoc(), statistics);
    if (mlir::failed(flattenedLhs) || mlir::failed(flattenedRhs))
      return mlir::failure();
    lhs = *flattenedLhs;
    rhs = *flattenedRhs;
    gemmResultType = getShapedType(resultType, {descriptor->batchCount,
                                                getMemRef(lhs).getDimSize(1),
                                                getMemRef(rhs).getDimSize(2)});
    batchCount = rewriter.getI64IntegerAttr(descriptor->batchCount);
    lhsBatchDims = rewriter.getDenseI64ArrayAttr({0});
    lhsMDim = rewriter.getI64IntegerAttr(1);
    lhsContractingDim = rewriter.getI64IntegerAttr(2);
    rhsBatchDims = rewriter.getDenseI64ArrayAttr({0});
    rhsContractingDim = rewriter.getI64IntegerAttr(1);
    rhsNDim = rewriter.getI64IntegerAttr(2);
    resultBatchDims = rewriter.getDenseI64ArrayAttr({0});
    resultMDim = rewriter.getI64IntegerAttr(1);
    resultNDim = rewriter.getI64IntegerAttr(2);
  }
  auto gemm = rewriter.create<ComputeGemmOp>(
      operation.getLoc(), gemmResultType, lhs, rhs, lhsOrientation,
      rhsOrientation, batchCount, lhsBatchDims, lhsMDim, lhsContractingDim,
      rhsBatchDims, rhsContractingDim, rhsNDim, resultBatchDims, resultMDim,
      resultNDim);
  mlir::Value replacement = gemm.getResult();
  if (!descriptor->rankTwo) {
    llvm::SmallVector<int64_t, 6> expandedShape;
    for (int64_t dimension : descriptor->resultBatchDims)
      expandedShape.push_back(resultType.getDimSize(dimension));
    expandedShape.push_back(resultType.getDimSize(descriptor->resultMDim));
    expandedShape.push_back(resultType.getDimSize(descriptor->resultNDim));
    mlir::FailureOr<mlir::Value> expanded = reshapeBuffer(
        replacement, expandedShape, rewriter, operation.getLoc(), statistics);
    if (mlir::failed(expanded))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> restored =
        permuteBuffer(*expanded, invertPermutation(resultCanonicalOrder),
                      rewriter, operation.getLoc(), statistics);
    if (mlir::failed(restored) || (*restored).getType() != resultType)
      return mlir::failure();
    replacement = *restored;
  }
  ComputeFillOp fill = findLastFillBefore(destination, operation);
  if (!fill || !isPositiveZero(fill.getValue())) {
    auto combined = rewriter.create<ComputeElementwiseOp>(
        operation.getLoc(), resultType,
        ComputeElementwiseKindAttr::get(rewriter.getContext(),
                                        ComputeElementwiseKind::Add),
        mlir::ValueRange{destination, replacement},
        getIndexingMapsAttr(
            rewriter,
            {getIdentityMap(rewriter.getContext(), resultType.getRank()),
             getIdentityMap(rewriter.getContext(), resultType.getRank()),
             getIdentityMap(rewriter.getContext(), resultType.getRank())}));
    replacement = combined.getResult();
    ++statistics.elementwiseOperations;
  }
  if (!publishComputedValue(module, operation, destination, replacement,
                            rewriter, statistics))
    return mlir::failure();
  rewriter.eraseOp(operation);
  ++statistics.contractions;
  return mlir::success();
}

static mlir::LogicalResult
lowerConvolution(mlir::ModuleOp module, mlir::linalg::LinalgOp operation,
                 mlir::IRRewriter &rewriter,
                 StructuredToTileStatistics &statistics) {
  mlir::FailureOr<ConvDescriptor> descriptor = buildConvDescriptor(operation);
  if (mlir::failed(descriptor))
    return mlir::failure();
  mlir::Value input = operation.getDpsInputs()[0];
  mlir::Value weight = operation.getDpsInputs()[1];
  mlir::Value destination = operation.getDpsInits()[0];
  mlir::MemRefType destinationType = getMemRef(destination);
  mlir::MemRefType inputType = getMemRef(input);
  mlir::MemRefType weightType = getMemRef(weight);
  if (!destinationType || !inputType || !weightType ||
      inputType.getElementType() != weightType.getElementType() ||
      inputType.getElementType() != destinationType.getElementType())
    return mlir::failure();
  mlir::MemRefType resultType = getOwnedType(destinationType);
  rewriter.setInsertionPoint(operation);
  mlir::FailureOr<mlir::Value> canonicalInput = permuteBuffer(
      input, descriptor->inputToNHWC, rewriter, operation.getLoc(), statistics);
  mlir::FailureOr<mlir::Value> canonicalWeight =
      permuteBuffer(weight, descriptor->weightToXYOI, rewriter,
                    operation.getLoc(), statistics);
  if (mlir::failed(canonicalInput) || mlir::failed(canonicalWeight))
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> canonicalResultShape;
  for (int64_t dimension : descriptor->outputToNHWC)
    canonicalResultShape.push_back(resultType.getDimSize(dimension));
  auto convolution = rewriter.create<ComputeConvOp>(
      operation.getLoc(), getShapedType(resultType, canonicalResultShape),
      *canonicalInput, *canonicalWeight,
      rewriter.getDenseI64ArrayAttr(descriptor->pads),
      rewriter.getDenseI64ArrayAttr(descriptor->unpads),
      rewriter.getDenseI64ArrayAttr(descriptor->strides),
      rewriter.getDenseI64ArrayAttr(descriptor->dilations));
  mlir::FailureOr<mlir::Value> restored =
      permuteBuffer(convolution.getResult(), descriptor->outputFromNHWC,
                    rewriter, operation.getLoc(), statistics);
  if (mlir::failed(restored) || (*restored).getType() != resultType)
    return mlir::failure();
  mlir::Value replacement = *restored;
  ComputeFillOp fill = findLastFillBefore(destination, operation);
  if (!fill || !isPositiveZero(fill.getValue())) {
    auto combined = rewriter.create<ComputeElementwiseOp>(
        operation.getLoc(), resultType,
        ComputeElementwiseKindAttr::get(rewriter.getContext(),
                                        ComputeElementwiseKind::Add),
        mlir::ValueRange{destination, replacement},
        getIndexingMapsAttr(
            rewriter,
            {getIdentityMap(rewriter.getContext(), resultType.getRank()),
             getIdentityMap(rewriter.getContext(), resultType.getRank()),
             getIdentityMap(rewriter.getContext(), resultType.getRank())}));
    replacement = combined.getResult();
    ++statistics.elementwiseOperations;
  }
  if (!publishComputedValue(module, operation, destination, replacement,
                            rewriter, statistics))
    return mlir::failure();
  rewriter.eraseOp(operation);
  ++statistics.convolutions;
  return mlir::success();
}

static mlir::Value createReductionIdentity(ComputeReduceKind kind,
                                           mlir::Type elementType,
                                           mlir::OpBuilder &builder,
                                           mlir::Location location) {
  mlir::arith::AtomicRMWKind atomicKind;
  if (kind == ComputeReduceKind::Sum)
    atomicKind = mlir::isa<mlir::FloatType>(elementType)
                     ? mlir::arith::AtomicRMWKind::addf
                     : mlir::arith::AtomicRMWKind::addi;
  else if (kind == ComputeReduceKind::Max)
    atomicKind = mlir::isa<mlir::FloatType>(elementType)
                     ? mlir::arith::AtomicRMWKind::maximumf
                     : mlir::arith::AtomicRMWKind::maxs;
  else if (kind == ComputeReduceKind::Min)
    atomicKind = mlir::isa<mlir::FloatType>(elementType)
                     ? mlir::arith::AtomicRMWKind::minimumf
                     : mlir::arith::AtomicRMWKind::mins;
  else
    return {};
  return mlir::arith::getIdentityValue(atomicKind, elementType, builder,
                                       location);
}

static ComputeElementwiseKind
getAccumulatorElementwiseKind(ComputeReduceKind kind) {
  switch (kind) {
  case ComputeReduceKind::Sum:
    return ComputeElementwiseKind::Add;
  case ComputeReduceKind::Max:
    return ComputeElementwiseKind::Max;
  case ComputeReduceKind::Min:
    return ComputeElementwiseKind::Min;
  case ComputeReduceKind::Avg:
    break;
  }
  llvm_unreachable("average has no exact accumulator elementwise kind");
}

static mlir::LogicalResult
lowerReduction(mlir::ModuleOp module, mlir::linalg::LinalgOp operation,
               mlir::IRRewriter &rewriter,
               StructuredToTileStatistics &statistics) {
  std::optional<ComputeReduceKind> kind = getReductionKind(operation);
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> dimensions =
      getReductionInputDimensions(operation);
  if (!kind || mlir::failed(dimensions))
    return mlir::failure();
  mlir::Value input = operation.getDpsInputs().front();
  mlir::Value destination = operation.getDpsInits().front();
  mlir::MemRefType destinationType = getMemRef(destination);
  mlir::MemRefType inputType = getMemRef(input);
  if (!destinationType || !inputType)
    return mlir::failure();
  mlir::MemRefType resultType = getOwnedType(destinationType);
  rewriter.setInsertionPoint(operation);
  ComputeFillOp fill = findLastFillBefore(destination, operation);
  mlir::Value init =
      fill ? fill.getValue()
           : createReductionIdentity(*kind, inputType.getElementType(),
                                     rewriter, operation.getLoc());
  if (!init || !init.getDefiningOp<mlir::arith::ConstantOp>())
    return mlir::failure();
  auto reduce = rewriter.create<ComputeReduceOp>(
      operation.getLoc(), resultType,
      ComputeReduceKindAttr::get(rewriter.getContext(), *kind), input, init,
      rewriter.getDenseI64ArrayAttr(*dimensions), mlir::TypedAttr{});
  mlir::Value replacement = reduce.getResult();
  if (!fill) {
    auto combined = rewriter.create<ComputeElementwiseOp>(
        operation.getLoc(), resultType,
        ComputeElementwiseKindAttr::get(rewriter.getContext(),
                                        getAccumulatorElementwiseKind(*kind)),
        mlir::ValueRange{destination, replacement},
        getIndexingMapsAttr(
            rewriter,
            {getIdentityMap(rewriter.getContext(), resultType.getRank()),
             getIdentityMap(rewriter.getContext(), resultType.getRank()),
             getIdentityMap(rewriter.getContext(), resultType.getRank())}));
    replacement = combined.getResult();
    ++statistics.elementwiseOperations;
  }
  if (!publishComputedValue(module, operation, destination, replacement,
                            rewriter, statistics))
    return mlir::failure();
  rewriter.eraseOp(operation);
  ++statistics.reductions;
  return mlir::success();
}

static mlir::FailureOr<ExprValue>
lookupExpr(mlir::Value value, llvm::DenseMap<mlir::Value, ExprValue> &values) {
  auto found = values.find(value);
  if (found == values.end())
    return mlir::failure();
  return found->second;
}

static mlir::LogicalResult
lowerElementwise(mlir::ModuleOp module, mlir::linalg::LinalgOp operation,
                 mlir::IRRewriter &rewriter,
                 StructuredToTileStatistics &statistics) {
  mlir::Value destination = operation.getDpsInits().front();
  mlir::MemRefType destinationType = getMemRef(destination);
  if (!destinationType)
    return mlir::failure();
  mlir::MemRefType resultType = getOwnedType(destinationType);
  llvm::SmallVector<mlir::AffineMap, 4> maps = operation.getIndexingMapsArray();
  mlir::Block &body = operation->getRegion(0).front();
  if (body.getNumArguments() != operation.getNumDpsInputs() + 1 ||
      maps.size() != body.getNumArguments())
    return mlir::failure();

  rewriter.setInsertionPoint(operation);
  llvm::DenseMap<mlir::Value, ExprValue> values;
  unsigned argumentIndex = 0;
  for (mlir::Value input : operation.getDpsInputs()) {
    mlir::BlockArgument argument = body.getArgument(argumentIndex);
    if (mlir::MemRefType inputType = getMemRef(input)) {
      values.try_emplace(argument, ExprValue{input, maps[argumentIndex]});
    } else {
      mlir::FailureOr<ExprValue> filled = createScalarFill(
          input, resultType, rewriter, operation.getLoc(), statistics);
      if (mlir::failed(filled))
        return mlir::failure();
      values.try_emplace(argument, *filled);
    }
    ++argumentIndex;
  }
  values.try_emplace(body.getArgument(argumentIndex),
                     ExprValue{destination, maps[argumentIndex]});
  llvm::SetVector<mlir::Value> captures;
  mlir::getUsedValuesDefinedAbove(operation->getRegion(0), captures);
  for (mlir::Value capture : captures) {
    if (mlir::isa<mlir::ShapedType>(capture.getType()))
      return mlir::failure();
    mlir::FailureOr<ExprValue> filled = createScalarFill(
        capture, resultType, rewriter, operation.getLoc(), statistics);
    if (mlir::failed(filled))
      return mlir::failure();
    values.try_emplace(capture, *filled);
  }

  for (mlir::Operation &nested : body.without_terminator()) {
    if (mlir::isOpTriviallyDead(&nested))
      continue;
    mlir::Location location = mlir::FusedLoc::get(
        rewriter.getContext(), {operation.getLoc(), nested.getLoc()});
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(nested)) {
      auto cloned = rewriter.create<mlir::arith::ConstantOp>(
          location, constant.getValue());
      mlir::FailureOr<ExprValue> filled = createScalarFill(
          cloned.getResult(), resultType, rewriter, location, statistics);
      if (mlir::failed(filled))
        return mlir::failure();
      values[constant.getResult()] = *filled;
      continue;
    }
    if (mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp>(nested)) {
      mlir::Value source = nested.getOperand(0);
      mlir::FailureOr<ExprValue> input = lookupExpr(source, values);
      if (mlir::failed(input))
        return mlir::failure();
      mlir::MemRefType inputType = getMemRef(input->buffer);
      if (!inputType)
        return mlir::failure();
      mlir::MemRefType fullInputType = changeShapeAndElementType(
          inputType, resultType.getShape(), inputType.getElementType());
      mlir::FailureOr<mlir::Value> fullInput = materializeExprAs(
          *input, fullInputType, rewriter, location, statistics);
      if (mlir::failed(fullInput))
        return mlir::failure();
      mlir::MemRefType convertedType =
          changeElementType(fullInputType, nested.getResult(0).getType());
      mlir::FailureOr<mlir::Value> converted = createConvert(
          *fullInput, convertedType, rewriter, location, statistics);
      if (mlir::failed(converted))
        return mlir::failure();
      values[nested.getResult(0)] =
          ExprValue{*converted, getIdentityMap(rewriter.getContext(),
                                               convertedType.getRank())};
      continue;
    }
    std::optional<ComputeElementwiseKind> kind =
        getScalarElementwiseKind(&nested);
    if (!kind)
      return mlir::failure();
    llvm::SmallVector<ExprValue, 3> operands;
    mlir::ValueRange scalarOperands = nested.getOperands();
    if (*kind == ComputeElementwiseKind::Square)
      scalarOperands = scalarOperands.take_front(1);
    for (mlir::Value operand : scalarOperands) {
      mlir::FailureOr<ExprValue> value = lookupExpr(operand, values);
      if (mlir::failed(value))
        return mlir::failure();
      operands.push_back(*value);
    }
    mlir::FailureOr<ExprValue> result =
        createElementwise(*kind, operands, nested.getResult(0).getType(),
                          resultType, rewriter, location, statistics);
    if (mlir::failed(result))
      return mlir::failure();
    values[nested.getResult(0)] = *result;
  }

  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return mlir::failure();
  mlir::FailureOr<ExprValue> yielded =
      lookupExpr(yield.getValues().front(), values);
  if (mlir::failed(yielded))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> replacement = materializeExprAs(
      *yielded, resultType, rewriter, operation.getLoc(), statistics);
  if (mlir::failed(replacement))
    return mlir::failure();
  if (!publishComputedValue(module, operation, destination, *replacement,
                            rewriter, statistics))
    return mlir::failure();
  rewriter.eraseOp(operation);
  ++statistics.elementwiseExpressions;
  return mlir::success();
}

static void eraseDeadPrivateStorage(mlir::ModuleOp module) {
  bool changed = true;
  mlir::IRRewriter rewriter(module.getContext());
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *, 16> dead;
    module.walk([&](mlir::Operation *operation) {
      if (auto materialize = mlir::dyn_cast<LayoutMaterializeOp>(operation)) {
        if (materialize.getResult().use_empty())
          dead.push_back(operation);
        return;
      }
      if (auto fill = mlir::dyn_cast<ComputeFillOp>(operation)) {
        if (fill.getDest().hasOneUse())
          dead.push_back(operation);
        return;
      }
      if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
        if (allocation.getResult().use_empty())
          dead.push_back(operation);
        return;
      }
      if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation))
        if (constant.getResult().use_empty())
          dead.push_back(operation);
    });
    for (mlir::Operation *operation : llvm::reverse(dead)) {
      if (!operation->getBlock())
        continue;
      if (auto fill = mlir::dyn_cast<ComputeFillOp>(operation)) {
        if (!fill.getDest().hasOneUse())
          continue;
      } else if (!operation->use_empty()) {
        continue;
      }
      rewriter.eraseOp(operation);
      changed = true;
    }
  }
}

} // namespace

mlir::LogicalResult verifyStructuredComputeLowered(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)))
    return mlir::failure();
  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::linalg::LinalgOp, LinalgExtAttentionOp,
                  LinalgExtOnlineAttentionOp>(operation)) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    auto dataflow = mlir::dyn_cast<WaferTileDataflowOpInterface>(operation);
    if (!dataflow)
      return mlir::WalkResult::advance();
    for (mlir::Type type : operation->getOperandTypes())
      if (mlir::isa<mlir::ShapedType>(type) && !isWaferSPMMemRefType(type)) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    for (mlir::Type type : operation->getResultTypes())
      if (mlir::isa<mlir::ShapedType>(type) && !isWaferSPMMemRefType(type)) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    return mlir::WalkResult::advance();
  });
  return mlir::success(!illegal);
}

StructuredToTileResult
lowerStructuredComputeToTile(mlir::ModuleOp module,
                             StructuredMaterializationRelations &relations) {
  if (!module || mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyLayoutResolvedTileRegions(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return fail(StructuredToTileFailureKind::BrokenContract,
                "structured-to-Tile requires layout-resolved current IR and "
                "live buffer relations");

  llvm::SmallVector<LoweringPlan, 32> plans;
  std::string detail;
  if (mlir::failed(preflight(module, plans, detail)))
    return fail(StructuredToTileFailureKind::Unsupported, detail);
  StructuredToTileResult result;
  mlir::IRRewriter rewriter(module.getContext());
  for (const LoweringPlan &plan : plans) {
    mlir::LogicalResult lowered = mlir::failure();
    switch (plan.kind) {
    case LoweringKind::Fill:
      lowered = lowerFill(plan.operation, rewriter, result.statistics);
      break;
    case LoweringKind::CapturedFill:
      lowered = lowerCapturedFill(plan.operation, rewriter, result.statistics);
      break;
    case LoweringKind::Contraction:
      lowered =
          lowerContraction(module, plan.operation, rewriter, result.statistics);
      break;
    case LoweringKind::Convolution:
      lowered =
          lowerConvolution(module, plan.operation, rewriter, result.statistics);
      break;
    case LoweringKind::Reduction:
      lowered =
          lowerReduction(module, plan.operation, rewriter, result.statistics);
      break;
    case LoweringKind::Elementwise:
      lowered =
          lowerElementwise(module, plan.operation, rewriter, result.statistics);
      break;
    }
    if (mlir::failed(lowered)) {
      llvm::StringRef kind;
      switch (plan.kind) {
      case LoweringKind::Fill:
        kind = "fill";
        break;
      case LoweringKind::CapturedFill:
        kind = "captured fill";
        break;
      case LoweringKind::Contraction:
        kind = "contraction";
        break;
      case LoweringKind::Convolution:
        kind = "convolution";
        break;
      case LoweringKind::Reduction:
        kind = "reduction";
        break;
      case LoweringKind::Elementwise:
        kind = "elementwise";
        break;
      }
      std::string failureDetail =
          ("preflighted structured-to-Tile lowering failed while rewriting "
           "current " +
           kind + " IR")
              .str();
      if (plan.kind == LoweringKind::Elementwise &&
          plan.operation->getNumRegions() == 1 &&
          !plan.operation->getRegion(0).empty()) {
        failureDetail += "; scalar body=";
        llvm::interleave(
            plan.operation->getRegion(0).front().without_terminator(),
            [&](mlir::Operation &nested) {
              failureDetail += nested.getName().getStringRef().str();
            },
            [&] { failureDetail += ","; });
      }
      return fail(StructuredToTileFailureKind::CompilerFailure, failureDetail);
    }
  }
  eraseDeadPrivateStorage(module);
  rebuildCurrentBufferOwnerRelations(module, relations);
  retainCurrentStructuredBufferRelations(module, relations);
  if (mlir::failed(verifyStructuredComputeLowered(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return fail(StructuredToTileFailureKind::CompilerFailure,
                "structured-to-Tile produced invalid current IR or buffer "
                "relations");
  return result;
}

} // namespace wafer::compiler::detail
