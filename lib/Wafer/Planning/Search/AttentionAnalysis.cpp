//===- AttentionAnalysis.cpp - Query-local attention analysis -----------===//

#include "AttentionAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include <limits>

namespace wafer::tensor_program_alternatives {
namespace {

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

static mlir::func::FuncOp findSingleTensorProgram(mlir::ModuleOp module) {
  mlir::func::FuncOp found;
  bool multiple = false;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.isExternal())
      return;
    if (found) {
      multiple = true;
      return;
    }
    found = function;
  });
  return multiple ? mlir::func::FuncOp{} : found;
}

static bool hasExactlyUses(mlir::Value value,
                           llvm::ArrayRef<mlir::Operation *> expected) {
  if (static_cast<size_t>(std::distance(
          value.getUses().begin(), value.getUses().end())) != expected.size())
    return false;
  for (mlir::Operation *operation : expected)
    if (llvm::count_if(value.getUses(), [&](mlir::OpOperand &use) {
          return use.getOwner() == operation;
        }) != 1)
      return false;
  return true;
}

static mlir::Value getFillScalar(mlir::Value value) {
  auto fill = value.getDefiningOp<mlir::linalg::FillOp>();
  if (!fill)
    return {};
  mlir::linalg::LinalgOp linalg = fill;
  if (linalg.getNumDpsInputs() != 1 || linalg.getNumDpsInits() != 1 ||
      fill->getNumResults() != 1)
    return {};
  return linalg.getDpsInputs().front();
}

static bool
hasIteratorKinds(mlir::linalg::LinalgOp operation,
                 llvm::ArrayRef<mlir::utils::IteratorType> expected) {
  return llvm::equal(operation.getIteratorTypesArray(), expected);
}

static bool hasMaps(mlir::linalg::LinalgOp operation,
                    llvm::ArrayRef<mlir::AffineMap> expected) {
  return llvm::equal(operation.getIndexingMapsArray(), expected);
}

template <typename OpTy>
static bool matchBinaryBody(mlir::linalg::GenericOp generic,
                            bool exactOperandOrder) {
  if (generic.getRegionInputArgs().size() != 2 ||
      generic.getRegionOutputArgs().size() != 1)
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  auto operation = mlir::dyn_cast<OpTy>(&body.front());
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!operation || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != operation->getResult(0))
    return false;
  mlir::Value left = generic.getRegionInputArgs()[0];
  mlir::Value right = generic.getRegionInputArgs()[1];
  if (operation->getOperand(0) == left && operation->getOperand(1) == right)
    return true;
  return !exactOperandOrder && operation->getOperand(0) == right &&
         operation->getOperand(1) == left;
}

template <typename OpTy>
static bool matchUnaryBody(mlir::linalg::GenericOp generic) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1)
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  auto operation = mlir::dyn_cast<OpTy>(&body.front());
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return operation && yield && yield.getValues().size() == 1 &&
         operation->getOperand(0) == generic.getRegionInputArgs().front() &&
         yield.getValues().front() == operation->getResult(0);
}

template <typename OpTy>
static bool matchReductionBody(mlir::linalg::GenericOp generic) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1)
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return false;
  auto operation = mlir::dyn_cast<OpTy>(&body.front());
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!operation || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != operation->getResult(0))
    return false;
  mlir::Value input = generic.getRegionInputArgs().front();
  mlir::Value accumulator = generic.getRegionOutputArgs().front();
  return (operation->getOperand(0) == input &&
          operation->getOperand(1) == accumulator) ||
         (operation->getOperand(1) == input &&
          operation->getOperand(0) == accumulator);
}

static bool matchBroadcastBody(mlir::linalg::GenericOp generic) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1)
    return false;
  mlir::Block &body = generic.getRegion().front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return body.without_terminator().empty() && yield &&
         yield.getValues().size() == 1 &&
         yield.getValues().front() == generic.getRegionInputArgs().front();
}

static bool matchMultiplyAccumulateBody(mlir::linalg::LinalgOp operation) {
  if (operation.getRegionInputArgs().size() != 2 ||
      operation.getRegionOutputArgs().size() != 1)
    return false;
  mlir::Block &body = operation->getRegion(0).front();
  if (std::distance(body.begin(), body.end()) != 3)
    return false;
  auto multiply = mlir::dyn_cast<mlir::arith::MulFOp>(&body.front());
  auto add = mlir::dyn_cast<mlir::arith::AddFOp>(&*std::next(body.begin()));
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!multiply || !add || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != add.getResult())
    return false;
  mlir::Value left = operation.getRegionInputArgs()[0];
  mlir::Value right = operation.getRegionInputArgs()[1];
  mlir::Value accumulator = operation.getRegionOutputArgs().front();
  bool exactMultiply =
      (multiply.getLhs() == left && multiply.getRhs() == right) ||
      (multiply.getLhs() == right && multiply.getRhs() == left);
  bool exactAdd =
      (add.getLhs() == multiply.getResult() && add.getRhs() == accumulator) ||
      (add.getRhs() == multiply.getResult() && add.getLhs() == accumulator);
  return exactMultiply && exactAdd;
}

static mlir::AffineMap mapForDims(mlir::MLIRContext *context, unsigned loopRank,
                                  llvm::ArrayRef<unsigned> dims) {
  llvm::SmallVector<mlir::AffineExpr, 6> expressions;
  for (unsigned dim : dims)
    expressions.push_back(mlir::getAffineDimExpr(dim, context));
  return mlir::AffineMap::get(loopRank, 0, expressions, context);
}

static llvm::SmallVector<unsigned, 6> sequence(unsigned count) {
  llvm::SmallVector<unsigned, 6> result;
  for (unsigned index = 0; index < count; ++index)
    result.push_back(index);
  return result;
}

static bool isValueAncestor(mlir::Value ancestor, mlir::Value value) {
  llvm::SmallVector<mlir::Value, 16> worklist{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (current == ancestor)
      return true;
    if (!visited.insert(current).second)
      continue;
    mlir::Operation *definition = current.getDefiningOp();
    if (definition)
      llvm::append_range(worklist, definition->getOperands());
  }
  return false;
}

static mlir::Value getStaticViewSource(mlir::Operation *operation) {
  mlir::Value source;
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation))
    source = expand.getSrc();
  else if (auto collapse =
               mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation))
    source = collapse.getSrc();
  else if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(operation))
    source = cast.getSource();
  if (!source || operation->getNumResults() != 1)
    return {};
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(operation->getResult(0).getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      sourceType.getNumElements() != resultType.getNumElements())
    return {};
  return source;
}

static mlir::Value getPointwiseFloatConvertSource(mlir::Operation *operation) {
  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(operation);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
    return {};
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(
      generic.getDpsInputs().front().getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic.getResult(0).getType());
  if (!inputType || !resultType || !inputType.hasStaticShape() ||
      inputType.getShape() != resultType.getShape() ||
      !mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
      !mlir::isa<mlir::FloatType>(resultType.getElementType()))
    return {};
  unsigned rank = resultType.getRank();
  mlir::AffineMap identity =
      mlir::AffineMap::getMultiDimIdentityMap(rank, operation->getContext());
  llvm::SmallVector<mlir::utils::IteratorType, 6> parallel(
      rank, mlir::utils::IteratorType::parallel);
  if (!hasMaps(generic, {identity, identity}) ||
      !hasIteratorKinds(generic, parallel))
    return {};
  mlir::Block &body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return {};
  mlir::Operation &conversion = body.front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp>(conversion) ||
      conversion.getNumOperands() != 1 || conversion.getNumResults() != 1 ||
      conversion.getOperand(0) != generic.getRegionInputArgs().front() ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != conversion.getResult(0))
    return {};
  return generic.getDpsInputs().front();
}

static mlir::FailureOr<mlir::Value>
traceTransparentChain(mlir::Value value, bool allowFloatConvert,
                      bool allowSharedBoundaryValue = false) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    mlir::Operation *operation = value.getDefiningOp();
    if (!operation)
      return value;
    mlir::Value source = getStaticViewSource(operation);
    if (!source && allowFloatConvert)
      source = getPointwiseFloatConvertSource(operation);
    if (!source)
      return value;
    if (!value.hasOneUse()) {
      if (allowSharedBoundaryValue)
        return value;
      return mlir::failure();
    }
    value = source;
  }
  return mlir::failure();
}

static std::optional<int64_t>
checkedLeadingProduct(mlir::RankedTensorType type) {
  int64_t product = 1;
  for (int64_t dim = 0; dim + 2 < type.getRank(); ++dim) {
    int64_t extent = type.getDimSize(dim);
    if (extent <= 0 || product > std::numeric_limits<int64_t>::max() / extent)
      return std::nullopt;
    product *= extent;
  }
  return product;
}

} // namespace

mlir::FailureOr<AttentionSemantics>
analyzeAttentionSemantics(mlir::Operation *root, std::string *failureReason) {
  auto fail =
      [&](llvm::StringRef reason) -> mlir::FailureOr<AttentionSemantics> {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  };
  if (!root || root->getNumResults() != 1)
    return fail("attention requires one observable tensor result");
  mlir::Value observableOutput = root->getResult(0);
  mlir::FailureOr<mlir::Value> physicalOutput =
      traceTransparentChain(observableOutput, /*allowFloatConvert=*/false);
  if (mlir::failed(physicalOutput))
    return fail("attention output view chain is not exclusive and exact");
  auto output = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
      physicalOutput->getDefiningOp());
  if (!output || output->getNumResults() != 1 ||
      output.getNumDpsInputs() != 2 || output.getNumDpsInits() != 1 ||
      (!mlir::isa<mlir::linalg::GenericOp, mlir::linalg::BatchMatmulOp>(
           output.getOperation()) ||
       !matchMultiplyAccumulateBody(output)))
    return fail("attention requires one exact value contraction root");

  AttentionSemantics semantics;
  semantics.output = output;
  semantics.observableOutput = observableOutput;
  semantics.probabilityStorage = output.getDpsInputs()[0];
  semantics.physicalValues = output.getDpsInputs()[1];
  mlir::FailureOr<mlir::Value> probability = traceTransparentChain(
      semantics.probabilityStorage, /*allowFloatConvert=*/true);
  mlir::FailureOr<mlir::Value> logicalValues = traceTransparentChain(
      semantics.physicalValues, /*allowFloatConvert=*/false,
      /*allowSharedBoundaryValue=*/true);
  if (mlir::failed(probability) || mlir::failed(logicalValues))
    return fail("attention contraction view/conversion chain is not exact");
  semantics.probability = probability->getDefiningOp<mlir::linalg::GenericOp>();
  semantics.values = *logicalValues;
  if (!semantics.probability ||
      !matchBinaryBody<mlir::arith::DivFOp>(semantics.probability, true) ||
      !semantics.probability->getResult(0).hasOneUse())
    return fail("attention requires a single-use exact probability divide");

  semantics.exponential = semantics.probability.getDpsInputs()[0]
                              .getDefiningOp<mlir::linalg::GenericOp>();
  semantics.sumBroadcast = semantics.probability.getDpsInputs()[1]
                               .getDefiningOp<mlir::linalg::GenericOp>();
  if (!semantics.exponential || !semantics.sumBroadcast ||
      !matchUnaryBody<mlir::math::ExpOp>(semantics.exponential) ||
      !matchBroadcastBody(semantics.sumBroadcast))
    return fail("attention requires exp divided by a broadcast row sum");

  semantics.rowSum = semantics.sumBroadcast.getDpsInputs()[0]
                         .getDefiningOp<mlir::linalg::GenericOp>();
  semantics.shifted = semantics.exponential.getDpsInputs()[0]
                          .getDefiningOp<mlir::linalg::GenericOp>();
  if (!semantics.rowSum || !semantics.shifted ||
      !matchReductionBody<mlir::arith::AddFOp>(semantics.rowSum) ||
      !matchBinaryBody<mlir::arith::SubFOp>(semantics.shifted, true) ||
      semantics.rowSum.getDpsInputs()[0] != semantics.exponential.getResult(0))
    return fail("attention requires max-shifted exp and an exact row sum");

  semantics.scores = semantics.shifted.getDpsInputs()[0];
  semantics.maxBroadcast = semantics.shifted.getDpsInputs()[1]
                               .getDefiningOp<mlir::linalg::GenericOp>();
  if (!semantics.maxBroadcast || !matchBroadcastBody(semantics.maxBroadcast))
    return fail("attention requires one broadcast row maximum");
  semantics.rowMax = semantics.maxBroadcast.getDpsInputs()[0]
                         .getDefiningOp<mlir::linalg::GenericOp>();
  if (!semantics.rowMax ||
      !matchReductionBody<mlir::arith::MaximumFOp>(semantics.rowMax) ||
      semantics.rowMax.getDpsInputs()[0] != semantics.scores)
    return fail("attention requires an IEEE maximum row reduction");

  if (!hasExactlyUses(semantics.rowMax.getResult(0),
                      {semantics.maxBroadcast.getOperation()}) ||
      !hasExactlyUses(semantics.maxBroadcast.getResult(0),
                      {semantics.shifted.getOperation()}) ||
      !hasExactlyUses(semantics.shifted.getResult(0),
                      {semantics.exponential.getOperation()}) ||
      !hasExactlyUses(semantics.exponential.getResult(0),
                      {semantics.rowSum.getOperation(),
                       semantics.probability.getOperation()}) ||
      !hasExactlyUses(semantics.rowSum.getResult(0),
                      {semantics.sumBroadcast.getOperation()}) ||
      !hasExactlyUses(semantics.sumBroadcast.getResult(0),
                      {semantics.probability.getOperation()}))
    return fail("attention probability chain has an observable extra use");

  semantics.scoreType =
      mlir::dyn_cast<mlir::RankedTensorType>(semantics.scores.getType());
  semantics.valueType =
      mlir::dyn_cast<mlir::RankedTensorType>(semantics.values.getType());
  auto observableOutputType = mlir::dyn_cast<mlir::RankedTensorType>(
      semantics.observableOutput.getType());
  semantics.physicalOutputType =
      mlir::dyn_cast<mlir::RankedTensorType>(output->getResult(0).getType());
  auto probabilityStorageType = mlir::dyn_cast<mlir::RankedTensorType>(
      semantics.probabilityStorage.getType());
  if (!semantics.scoreType || !semantics.valueType || !observableOutputType ||
      !semantics.physicalOutputType || !probabilityStorageType ||
      !semantics.scoreType.hasStaticShape() ||
      !semantics.valueType.hasStaticShape() ||
      !observableOutputType.hasStaticShape() ||
      !semantics.physicalOutputType.hasStaticShape() ||
      !probabilityStorageType.hasStaticShape() ||
      semantics.scoreType.getRank() < 2 ||
      semantics.scoreType.getRank() != semantics.valueType.getRank())
    return fail("attention requires compatible static logical tensor ranks");
  auto softmaxElementType =
      mlir::dyn_cast<mlir::FloatType>(semantics.scoreType.getElementType());
  if (!softmaxElementType ||
      !mlir::isa<mlir::FloatType>(semantics.valueType.getElementType()) ||
      !mlir::isa<mlir::FloatType>(observableOutputType.getElementType()) ||
      !mlir::isa<mlir::FloatType>(
          semantics.physicalOutputType.getElementType()) ||
      !mlir::isa<mlir::FloatType>(probabilityStorageType.getElementType()))
    return fail("attention requires floating-point storage and accumulation");
  if (probabilityStorageType.getNumElements() !=
      semantics.scoreType.getNumElements())
    return fail("attention probability storage changed the logical domain");

  unsigned rank = semantics.scoreType.getRank();
  unsigned prefixRank = rank - 2;
  for (unsigned dim = 0; dim < prefixRank; ++dim)
    if (semantics.scoreType.getDimSize(dim) !=
        semantics.valueType.getDimSize(dim))
      return fail("attention batch/head dimensions do not agree");
  if (semantics.scoreType.getDimSize(rank - 1) !=
          semantics.valueType.getDimSize(prefixRank) ||
      semantics.valueType.getDimSize(rank - 1) <= 0)
    return fail("attention Q/K/V contraction dimensions do not agree");
  llvm::SmallVector<int64_t, 6> logicalOutputShape(
      semantics.scoreType.getShape().begin(),
      semantics.scoreType.getShape().end());
  logicalOutputShape.back() = semantics.valueType.getDimSize(rank - 1);
  semantics.outputType = mlir::RankedTensorType::get(
      logicalOutputShape, semantics.physicalOutputType.getElementType());
  if (observableOutputType.getNumElements() !=
          semantics.outputType.getNumElements() ||
      observableOutputType.getElementType() !=
          semantics.outputType.getElementType())
    return fail("attention output view changed the logical value domain");
  semantics.reductionExtent = semantics.scoreType.getDimSize(rank - 1);
  if (semantics.reductionExtent <= 0)
    return fail("attention requires a nonempty static K/V domain");

  mlir::MLIRContext *context = root->getContext();
  llvm::SmallVector<unsigned, 6> identity = sequence(rank);
  llvm::SmallVector<unsigned, 6> row = sequence(rank - 1);
  llvm::SmallVector<mlir::utils::IteratorType, 6> pointwise(
      rank, mlir::utils::IteratorType::parallel);
  llvm::SmallVector<mlir::utils::IteratorType, 6> reduction = pointwise;
  reduction.back() = mlir::utils::IteratorType::reduction;
  mlir::AffineMap identityMap = mapForDims(context, rank, identity);
  mlir::AffineMap rowMap = mapForDims(context, rank, row);
  if (!hasMaps(semantics.rowMax, {identityMap, rowMap}) ||
      !hasMaps(semantics.rowSum, {identityMap, rowMap}) ||
      !hasIteratorKinds(semantics.rowMax, reduction) ||
      !hasIteratorKinds(semantics.rowSum, reduction) ||
      !hasMaps(semantics.maxBroadcast, {rowMap, identityMap}) ||
      !hasMaps(semantics.sumBroadcast, {rowMap, identityMap}) ||
      !hasMaps(semantics.shifted, {identityMap, identityMap, identityMap}) ||
      !hasMaps(semantics.exponential, {identityMap, identityMap}) ||
      !hasMaps(semantics.probability,
               {identityMap, identityMap, identityMap}) ||
      !hasIteratorKinds(semantics.maxBroadcast, pointwise) ||
      !hasIteratorKinds(semantics.sumBroadcast, pointwise) ||
      !hasIteratorKinds(semantics.shifted, pointwise) ||
      !hasIteratorKinds(semantics.exponential, pointwise) ||
      !hasIteratorKinds(semantics.probability, pointwise))
    return fail("attention softmax indexing relation is not exact");

  auto physicalValueType = mlir::dyn_cast<mlir::RankedTensorType>(
      semantics.physicalValues.getType());
  if (!physicalValueType || !physicalValueType.hasStaticShape() ||
      probabilityStorageType.getRank() < 2 ||
      probabilityStorageType.getRank() != physicalValueType.getRank() ||
      probabilityStorageType.getRank() !=
          semantics.physicalOutputType.getRank())
    return fail("attention physical contraction ranks are not compatible");
  unsigned physicalRank = probabilityStorageType.getRank();
  unsigned physicalPrefixRank = physicalRank - 2;
  for (unsigned dim = 0; dim < physicalPrefixRank; ++dim)
    if (probabilityStorageType.getDimSize(dim) !=
            physicalValueType.getDimSize(dim) ||
        probabilityStorageType.getDimSize(dim) !=
            semantics.physicalOutputType.getDimSize(dim))
      return fail("attention physical batch dimensions do not agree");
  if (probabilityStorageType.getDimSize(physicalPrefixRank) !=
          semantics.physicalOutputType.getDimSize(physicalPrefixRank) ||
      probabilityStorageType.getDimSize(physicalRank - 1) !=
          physicalValueType.getDimSize(physicalPrefixRank) ||
      physicalValueType.getDimSize(physicalRank - 1) !=
          semantics.physicalOutputType.getDimSize(physicalRank - 1))
    return fail("attention physical contraction dimensions do not agree");
  auto logicalLeading = checkedLeadingProduct(semantics.scoreType);
  auto physicalLeading = checkedLeadingProduct(probabilityStorageType);
  if (!logicalLeading || !physicalLeading ||
      *logicalLeading != *physicalLeading)
    return fail("attention view reassociation changed the leading domain");

  const unsigned outputLoopRank = physicalRank + 1;
  llvm::SmallVector<unsigned, 6> probabilityDims = sequence(physicalRank - 1);
  probabilityDims.push_back(physicalRank);
  llvm::SmallVector<unsigned, 6> valueDims = sequence(physicalPrefixRank);
  valueDims.push_back(physicalRank);
  valueDims.push_back(physicalRank - 1);
  llvm::SmallVector<unsigned, 6> outputDims = sequence(physicalRank);
  llvm::SmallVector<mlir::utils::IteratorType, 6> outputIterators(
      outputLoopRank, mlir::utils::IteratorType::parallel);
  outputIterators.back() = mlir::utils::IteratorType::reduction;
  if (!hasMaps(output, {mapForDims(context, outputLoopRank, probabilityDims),
                        mapForDims(context, outputLoopRank, valueDims),
                        mapForDims(context, outputLoopRank, outputDims)}) ||
      !hasIteratorKinds(output, outputIterators))
    return fail("attention value contraction indexing relation is not exact");

  semantics.maxInit = getFillScalar(semantics.rowMax.getDpsInits().front());
  semantics.sumInit = getFillScalar(semantics.rowSum.getDpsInits().front());
  semantics.outputInit = getFillScalar(output.getDpsInits().front());
  if (!semantics.maxInit || !semantics.sumInit || !semantics.outputInit ||
      semantics.maxInit.getType() != softmaxElementType ||
      semantics.sumInit.getType() != softmaxElementType ||
      semantics.outputInit.getType() !=
          semantics.physicalOutputType.getElementType())
    return fail("attention reductions require typed scalar init values");

  if (failureReason)
    failureReason->clear();
  return semantics;
}

mlir::FailureOr<AttentionSemantics>
analyzeAttentionSemantics(mlir::ModuleOp module, std::string *failureReason) {
  mlir::func::FuncOp function = findSingleTensorProgram(module);
  if (!function || function.getNumResults() == 0 ||
      !function.getBody().hasOneBlock()) {
    setFailureReason(failureReason,
                     "attention requires one standalone tensor program");
    return mlir::failure();
  }
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != function.getNumResults()) {
    setFailureReason(failureReason,
                     "attention function result boundary is invalid");
    return mlir::failure();
  }

  std::optional<AttentionSemantics> matched;
  std::string lastFailure =
      "no observable function subgraph is a fused attention output";
  mlir::WalkResult walk = function.walk([&](mlir::linalg::LinalgOp root) {
    if (root->getNumResults() != 1)
      return mlir::WalkResult::advance();
    std::string localFailure;
    mlir::FailureOr<AttentionSemantics> candidate =
        analyzeAttentionSemantics(root.getOperation(), &localFailure);
    if (mlir::failed(candidate)) {
      if (!localFailure.empty())
        lastFailure = std::move(localFailure);
      return mlir::WalkResult::advance();
    }
    std::optional<unsigned> outputIndex;
    for (auto [index, returned] : llvm::enumerate(returnOp.getOperands())) {
      if (!isValueAncestor(candidate->output->getResult(0), returned))
        continue;
      if (outputIndex) {
        setFailureReason(
            failureReason,
            "one attention value is observable through multiple results");
        return mlir::WalkResult::interrupt();
      }
      outputIndex = index;
    }
    if (!outputIndex)
      return mlir::WalkResult::advance();
    if (matched) {
      setFailureReason(
          failureReason,
          "attention program has multiple independently observable subgraphs");
      return mlir::WalkResult::interrupt();
    }
    candidate->outputIndex = *outputIndex;
    matched = *candidate;
    return mlir::WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return mlir::failure();
  if (!matched) {
    setFailureReason(failureReason, lastFailure);
    return mlir::failure();
  }
  if (failureReason)
    failureReason->clear();
  return *matched;
}

} // namespace wafer::tensor_program_alternatives
