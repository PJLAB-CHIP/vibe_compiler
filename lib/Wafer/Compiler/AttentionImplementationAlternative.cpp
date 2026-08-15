//===- AttentionImplementationAlternative.cpp --------------------------===//

#include "AttentionImplementationAlternative.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/AttentionSemantics.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/Internal.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

using wafer::tensor_program_to_tile_region::AttentionImplementationKind;
using wafer::tensor_program_to_tile_region::AttentionSemantics;
using wafer::tensor_program_to_tile_region::DecodeAttentionSemantics;
using wafer::tensor_program_to_tile_region::TensorProgramScope;

static constexpr llvm::StringLiteral kProviderStableKey =
    "wafer.structured-attention";

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

static void appendShape(llvm::raw_ostream &stream,
                        llvm::ArrayRef<int64_t> shape) {
  stream << '[';
  llvm::interleaveComma(shape, stream);
  stream << ']';
}

static std::string
buildPointKey(llvm::StringRef implementation,
              const StructuredAlternativeDomain &domain,
              const StructuredAlternativeParameters &parameters) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << kProviderStableKey << '/' << implementation << "/output=";
  appendShape(stream, domain.outputShape);
  stream << "/output-tile=";
  appendShape(stream, parameters.outputTileSizes);
  stream << "/reduction=";
  appendShape(stream, domain.reductionShape);
  stream << "/reduction-tile=";
  appendShape(stream, parameters.reductionTileSizes);
  stream << "/partitions=" << parameters.parallelPartitionCount;
  return key;
}

static std::vector<StructuredAlternativeTileShape>
canonicalTileShapes(llvm::ArrayRef<StructuredAlternativeTileShape> requested,
                    llvm::ArrayRef<int64_t> scalarSeeds,
                    llvm::ArrayRef<int64_t> domain) {
  std::vector<StructuredAlternativeTileShape> result;
  result.emplace_back(domain.begin(), domain.end());
  if (!requested.empty() || !scalarSeeds.empty()) {
    for (const StructuredAlternativeTileShape &shape : requested) {
      if (shape.size() != domain.size())
        continue;
      bool valid = true;
      for (auto [tile, extent] : llvm::zip_equal(shape, domain))
        valid &= tile > 0 && extent > 0 && tile <= extent;
      if (valid)
        result.push_back(shape);
    }
    for (int64_t seed : scalarSeeds) {
      if (seed <= 0)
        continue;
      StructuredAlternativeTileShape shape;
      shape.reserve(domain.size());
      for (int64_t extent : domain) {
        if (extent <= 0) {
          shape.clear();
          break;
        }
        shape.push_back(std::min(seed, extent));
      }
      if (!shape.empty())
        result.push_back(std::move(shape));
    }
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

/// Adds attention-owned output shapes that keep the value-channel axis
/// complete while independently tiling the remaining logical output axes.
/// The last-axis role is proven by AttentionSemantics' value contraction; it
/// is not inferred by the common search from a shape or a provider key.
static void appendCompleteValueChannelTileShapes(
    std::vector<StructuredAlternativeTileShape> &result,
    llvm::ArrayRef<int64_t> scalarSeeds, llvm::ArrayRef<int64_t> domain) {
  if (domain.empty())
    return;
  for (int64_t seed : scalarSeeds) {
    if (seed <= 0)
      continue;
    StructuredAlternativeTileShape shape;
    shape.reserve(domain.size());
    for (int64_t extent : domain) {
      if (extent <= 0) {
        shape.clear();
        break;
      }
      shape.push_back(std::min(seed, extent));
    }
    if (shape.empty())
      continue;
    shape.back() = domain.back();
    result.push_back(std::move(shape));
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
}

static std::vector<int64_t>
canonicalSplitCounts(llvm::ArrayRef<int64_t> requested,
                     int64_t reductionExtent) {
  std::vector<int64_t> result;
  for (int64_t count : requested)
    if (count > 1 && count <= reductionExtent)
      result.push_back(count);
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

static std::optional<uint64_t> checkedMultiply(uint64_t left, uint64_t right) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    return std::nullopt;
  return left * right;
}

static std::optional<uint64_t> checkedAdd(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left)
    return std::nullopt;
  return left + right;
}

static std::optional<uint64_t> checkedProduct(llvm::ArrayRef<int64_t> values) {
  uint64_t product = 1;
  for (int64_t value : values) {
    if (value <= 0)
      return std::nullopt;
    std::optional<uint64_t> next =
        checkedMultiply(product, static_cast<uint64_t>(value));
    if (!next)
      return std::nullopt;
    product = *next;
  }
  return product;
}

static std::optional<uint64_t> checkedTensorBytes(uint64_t elements,
                                                  mlir::Type elementType) {
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  const uint64_t bitWidth = elementType.getIntOrFloatBitWidth();
  const uint64_t bytesPerElement = (bitWidth + 7) / 8;
  return checkedMultiply(elements, bytesPerElement);
}

struct AttentionScoreContractionModel {
  uint64_t batchElements = 0;
  uint64_t queryElements = 0;
  uint64_t keyValueElements = 0;
  uint64_t contractionDepth = 0;
  mlir::Type leftElementType;
  mlir::Type rightElementType;
};

static std::optional<uint64_t>
checkedLoopProduct(llvm::ArrayRef<int64_t> loopRanges,
                   llvm::ArrayRef<unsigned> dimensions) {
  uint64_t product = 1;
  for (unsigned dimension : dimensions) {
    if (dimension >= loopRanges.size() || loopRanges[dimension] <= 0)
      return std::nullopt;
    std::optional<uint64_t> next =
        checkedMultiply(product, static_cast<uint64_t>(loopRanges[dimension]));
    if (!next)
      return std::nullopt;
    product = *next;
  }
  return product;
}

/// Finds the nearest unique structured contraction in the current score SSA.
/// Traversal follows tensor data inputs, stopping each branch at its first
/// contraction, so projection GEMMs behind Q/K cannot be mistaken for the
/// score contraction.  Pointwise scale, source mask, dtype conversion and
/// static views remain ordinary SSA and require no special cases.
static bool hasEquivalentScoreDomain(mlir::Value value,
                                     mlir::RankedTensorType scoreType) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 2 ||
      type.getDimSize(type.getRank() - 2) !=
          scoreType.getDimSize(scoreType.getRank() - 2) ||
      type.getDimSize(type.getRank() - 1) !=
          scoreType.getDimSize(scoreType.getRank() - 1))
    return false;
  return checkedProduct(type.getShape().drop_back(2)) ==
         checkedProduct(scoreType.getShape().drop_back(2));
}

static bool isIdentityPointwiseScoreOp(mlir::linalg::LinalgOp linalg,
                                       mlir::RankedTensorType scoreType) {
  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(linalg.getOperation());
  if (!generic || generic->getNumResults() != 1 ||
      !hasEquivalentScoreDomain(generic->getResult(0), scoreType))
    return false;
  const unsigned rank = scoreType.getRank();
  llvm::SmallVector<mlir::utils::IteratorType, 6> parallel(
      rank, mlir::utils::IteratorType::parallel);
  if (!llvm::equal(generic.getIteratorTypesArray(), parallel))
    return false;
  mlir::AffineMap identityMap =
      mlir::AffineMap::getMultiDimIdentityMap(rank, generic.getContext());
  llvm::SmallVector<mlir::AffineMap, 6> maps = generic.getIndexingMapsArray();
  if (maps.size() != generic.getNumDpsInputs() + generic.getNumDpsInits() ||
      llvm::any_of(maps,
                   [&](mlir::AffineMap map) { return map != identityMap; }))
    return false;
  return llvm::all_of(generic.getDpsInputs(), [&](mlir::Value input) {
    return hasEquivalentScoreDomain(input, scoreType);
  });
}

static mlir::Value
getEquivalentStaticScoreViewSource(mlir::Operation *operation,
                                   mlir::RankedTensorType scoreType) {
  mlir::Value source;
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation))
    source = expand.getSrc();
  else if (auto collapse =
               mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation))
    source = collapse.getSrc();
  else if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(operation))
    source = cast.getSource();
  if (!source || operation->getNumResults() != 1 ||
      !hasEquivalentScoreDomain(operation->getResult(0), scoreType) ||
      !hasEquivalentScoreDomain(source, scoreType))
    return {};
  return source;
}

static std::optional<mlir::linalg::LinalgOp>
findNearestScoreContraction(mlir::Value scores,
                            mlir::RankedTensorType scoreType) {
  llvm::SmallVector<mlir::Value, 8> worklist{scores};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    llvm::SmallVector<mlir::Value, 16> next;
    llvm::DenseSet<mlir::Operation *> contractions;
    for (mlir::Value value : worklist) {
      if (!value || !visited.insert(value).second)
        continue;
      mlir::Operation *definition = value.getDefiningOp();
      if (!definition)
        continue;
      auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(definition);
      if (linalg && mlir::linalg::isaContractionOpInterface(linalg)) {
        contractions.insert(definition);
        continue;
      }
      if (linalg && isIdentityPointwiseScoreOp(linalg, scoreType)) {
        for (mlir::Value input : linalg.getDpsInputs())
          next.push_back(input);
        continue;
      }
      mlir::Value viewSource =
          getEquivalentStaticScoreViewSource(definition, scoreType);
      if (viewSource)
        next.push_back(viewSource);
    }
    if (!contractions.empty()) {
      if (contractions.size() != 1)
        return std::nullopt;
      return mlir::cast<mlir::linalg::LinalgOp>(*contractions.begin());
    }
    worklist = std::move(next);
  }
  return std::nullopt;
}

static std::optional<AttentionScoreContractionModel>
buildAttentionScoreContractionModel(const AttentionSemantics &semantics) {
  std::optional<mlir::linalg::LinalgOp> contraction =
      findNearestScoreContraction(semantics.scores, semantics.scoreType);
  if (!contraction || (*contraction)->getNumResults() != 1)
    return std::nullopt;
  mlir::FailureOr<mlir::linalg::ContractionDimensions> dimensions =
      mlir::linalg::inferContractionDims(*contraction);
  if (mlir::failed(dimensions))
    return std::nullopt;
  llvm::SmallVector<int64_t, 6> loopRanges = contraction->getStaticLoopRanges();
  if (llvm::any_of(loopRanges, [](int64_t extent) { return extent <= 0; }))
    return std::nullopt;

  std::optional<uint64_t> batch =
      checkedLoopProduct(loopRanges, dimensions->batch);
  std::optional<uint64_t> query = checkedLoopProduct(loopRanges, dimensions->m);
  std::optional<uint64_t> keyValue =
      checkedLoopProduct(loopRanges, dimensions->n);
  std::optional<uint64_t> depth = checkedLoopProduct(loopRanges, dimensions->k);
  std::optional<uint64_t> logicalBatch =
      checkedProduct(semantics.scoreType.getShape().drop_back(2));
  if (!batch || !query || !keyValue || !depth || !logicalBatch ||
      *batch != *logicalBatch ||
      *query != static_cast<uint64_t>(semantics.scoreType.getDimSize(
                    semantics.scoreType.getRank() - 2)) ||
      *keyValue != static_cast<uint64_t>(semantics.reductionExtent))
    return std::nullopt;

  auto leftType = mlir::dyn_cast<mlir::RankedTensorType>(
      contraction->getDpsInputs()[0].getType());
  auto rightType = mlir::dyn_cast<mlir::RankedTensorType>(
      contraction->getDpsInputs()[1].getType());
  if (!leftType || !rightType || !leftType.hasStaticShape() ||
      !rightType.hasStaticShape() ||
      !leftType.getElementType().isIntOrFloat() ||
      !rightType.getElementType().isIntOrFloat())
    return std::nullopt;
  return AttentionScoreContractionModel{*batch,
                                        *query,
                                        *keyValue,
                                        *depth,
                                        leftType.getElementType(),
                                        rightType.getElementType()};
}

static std::optional<uint64_t>
getMaximumPartitionExtent(int64_t reductionExtent,
                          int64_t parallelPartitionCount) {
  if (reductionExtent <= 0 || parallelPartitionCount <= 0 ||
      parallelPartitionCount > reductionExtent)
    return std::nullopt;
  return static_cast<uint64_t>((reductionExtent + parallelPartitionCount - 1) /
                               parallelPartitionCount);
}

/// Target-independent peak-live prior for the block recurrence represented by
/// one point.  It deliberately describes the provider's typed algorithm, not
/// current lowering accidents: one score and probability block, one V block,
/// and the online `(m, l, o)` state are live; split-K/V keeps one state per
/// partition until the typed merge.  Exact Tile/Instr lifetime and SPM
/// placement remain downstream gates.
static std::optional<uint64_t> estimateAttentionPeakLiveBytes(
    const AttentionSemantics &semantics,
    const StructuredAlternativeParameters &parameters,
    const std::optional<AttentionScoreContractionModel> &contraction) {
  if (!contraction)
    return std::nullopt;
  llvm::ArrayRef<int64_t> outputTile(parameters.outputTileSizes);
  if (outputTile.size() < 2 || parameters.reductionTileSizes.size() != 1 ||
      parameters.parallelPartitionCount <= 0)
    return std::nullopt;

  std::optional<uint64_t> outer = checkedProduct(outputTile.drop_back(2));
  if (!outer)
    return std::nullopt;
  const uint64_t queryTile = static_cast<uint64_t>(outputTile.end()[-2]);
  const uint64_t valueWidth = static_cast<uint64_t>(outputTile.back());
  std::optional<uint64_t> maximumPartitionExtent = getMaximumPartitionExtent(
      semantics.reductionExtent, parameters.parallelPartitionCount);
  if (!maximumPartitionExtent)
    return std::nullopt;
  const uint64_t reductionTile =
      std::min(static_cast<uint64_t>(parameters.reductionTileSizes.front()),
               *maximumPartitionExtent);

  auto product3 = [](uint64_t first, uint64_t second,
                     uint64_t third) -> std::optional<uint64_t> {
    std::optional<uint64_t> prefix = checkedMultiply(first, second);
    return prefix ? checkedMultiply(*prefix, third) : std::nullopt;
  };
  std::optional<uint64_t> scoreElements =
      product3(*outer, queryTile, reductionTile);
  std::optional<uint64_t> valueElements =
      product3(*outer, reductionTile, valueWidth);
  std::optional<uint64_t> outputElements =
      product3(*outer, queryTile, valueWidth);
  std::optional<uint64_t> rowElements = checkedMultiply(*outer, queryTile);
  if (!scoreElements || !valueElements || !outputElements || !rowElements)
    return std::nullopt;

  std::optional<uint64_t> scoreBytes =
      checkedTensorBytes(*scoreElements, semantics.scoreType.getElementType());
  std::optional<uint64_t> valueBytes =
      checkedTensorBytes(*valueElements, semantics.valueType.getElementType());
  std::optional<uint64_t> outputBytes = checkedTensorBytes(
      *outputElements, semantics.outputType.getElementType());
  std::optional<uint64_t> rowBytes =
      checkedTensorBytes(*rowElements, semantics.scoreType.getElementType());
  if (!scoreBytes || !valueBytes || !outputBytes || !rowBytes)
    return std::nullopt;

  std::optional<uint64_t> scoreAndProbability = checkedMultiply(*scoreBytes, 2);
  std::optional<uint64_t> twoRows = checkedMultiply(*rowBytes, 2);
  std::optional<uint64_t> stateBytes =
      twoRows ? checkedAdd(*outputBytes, *twoRows) : std::nullopt;
  std::optional<uint64_t> partitionStates =
      stateBytes
          ? checkedMultiply(*stateBytes, static_cast<uint64_t>(
                                             parameters.parallelPartitionCount))
          : std::nullopt;
  std::optional<uint64_t> ordinaryBlocks =
      scoreAndProbability ? checkedAdd(*scoreAndProbability, *valueBytes)
                          : std::nullopt;
  std::optional<uint64_t> queryElements =
      product3(*outer, queryTile, contraction->contractionDepth);
  std::optional<uint64_t> keyElements =
      product3(*outer, reductionTile, contraction->contractionDepth);
  std::optional<uint64_t> queryBytes =
      queryElements
          ? checkedTensorBytes(*queryElements, contraction->leftElementType)
          : std::nullopt;
  std::optional<uint64_t> keyBytes =
      keyElements
          ? checkedTensorBytes(*keyElements, contraction->rightElementType)
          : std::nullopt;
  std::optional<uint64_t> operands = queryBytes && keyBytes
                                         ? checkedAdd(*queryBytes, *keyBytes)
                                         : std::nullopt;
  std::optional<uint64_t> completeBlocks =
      operands && ordinaryBlocks ? checkedAdd(*operands, *ordinaryBlocks)
                                 : std::nullopt;
  if (!completeBlocks)
    return std::nullopt;
  return partitionStates ? checkedAdd(*completeBlocks, *partitionStates)
                         : std::nullopt;
}

/// Target-independent scalar-operation prior for the provider algorithm.  It
/// counts the score contraction, exact softmax chain, value contraction, and
/// online/split state updates.  Most importantly, score work is multiplied by
/// the number of value-channel tiles because the current complete graph
/// materializes one recurrence per output tile.  This exposes recomputation
/// without inspecting provider keys or adding an attention rule to common
/// search; final instruction cost remains authoritative.
static std::optional<uint64_t> estimateAttentionComputeScalarOps(
    const AttentionSemantics &semantics,
    const StructuredAlternativeParameters &parameters,
    const std::optional<AttentionScoreContractionModel> &contraction) {
  if (!contraction || parameters.outputTileSizes.empty() ||
      parameters.reductionTileSizes.size() != 1)
    return std::nullopt;
  std::optional<uint64_t> scoreElements =
      checkedProduct(semantics.scoreType.getShape());
  std::optional<uint64_t> outputElements =
      checkedProduct(semantics.outputType.getShape());
  std::optional<uint64_t> rowElements =
      checkedProduct(semantics.outputType.getShape().drop_back());
  if (!scoreElements || !outputElements || !rowElements)
    return std::nullopt;

  const uint64_t valueExtent = static_cast<uint64_t>(
      semantics.outputType.getDimSize(semantics.outputType.getRank() - 1));
  const uint64_t valueTile =
      static_cast<uint64_t>(parameters.outputTileSizes.back());
  if (valueTile == 0 || valueTile > valueExtent)
    return std::nullopt;
  const uint64_t valueTileCount =
      valueExtent / valueTile + static_cast<uint64_t>(valueExtent % valueTile);

  std::optional<uint64_t> contractionOps =
      checkedMultiply(*scoreElements, contraction->contractionDepth);
  if (contractionOps)
    contractionOps = checkedMultiply(*contractionOps, 2);
  if (contractionOps)
    contractionOps = checkedMultiply(*contractionOps, valueTileCount);
  std::optional<uint64_t> softmaxOps = checkedMultiply(*scoreElements, 4);
  if (softmaxOps)
    softmaxOps = checkedMultiply(*softmaxOps, valueTileCount);
  std::optional<uint64_t> valueOps = checkedMultiply(*outputElements, 2);
  if (valueOps)
    valueOps = checkedMultiply(
        *valueOps, static_cast<uint64_t>(semantics.reductionExtent));
  if (!contractionOps || !softmaxOps || !valueOps)
    return std::nullopt;
  std::optional<uint64_t> total = checkedAdd(*contractionOps, *softmaxOps);
  if (total)
    total = checkedAdd(*total, *valueOps);
  if (!total)
    return std::nullopt;

  const uint64_t partitions =
      static_cast<uint64_t>(parameters.parallelPartitionCount);
  if (partitions > 1) {
    std::optional<uint64_t> mergeRows =
        checkedMultiply(*rowElements, valueTileCount);
    std::optional<uint64_t> mergePerPartition =
        mergeRows ? checkedAdd(*outputElements, *mergeRows) : std::nullopt;
    std::optional<uint64_t> merge =
        mergePerPartition ? checkedMultiply(*mergePerPartition, partitions - 1)
                          : std::nullopt;
    if (!merge)
      return std::nullopt;
    total = checkedAdd(*total, *merge);
  }
  return total;
}

static mlir::FailureOr<TensorProgramScope>
getMaterializationScope(mlir::ModuleOp module, std::string *failureReason) {
  if (!module) {
    setFailureReason(failureReason,
                     "structured attention requires an isolated module");
    return mlir::failure();
  }
  mlir::func::FuncOp function =
      wafer::tensor_program_to_tile_region::findSingleStandaloneTensorProgram(
          module);
  if (!function) {
    setFailureReason(failureReason,
                     "structured attention requires one standalone tensor "
                     "program");
    return mlir::failure();
  }
  return TensorProgramScope(function);
}

static llvm::DenseSet<mlir::Operation *>
snapshotTopLevelOperations(TensorProgramScope scope) {
  llvm::DenseSet<mlir::Operation *> operations;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    operations.insert(&operation);
  return operations;
}

static void recordCoveredTopLevelOperations(
    TensorProgramScope scope,
    const llvm::DenseSet<mlir::Operation *> &operationsBefore,
    StructuredImplementationAlternativeMaterialization *result) {
  if (!result)
    return;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    if (!operationsBefore.contains(&operation))
      result->coveredTopLevelOperations.push_back(&operation);
}

/// Production only advertises the online recurrence as a FlashAttention
/// implementation when the score computation itself is an interface-tilable
/// current-SSA producer.  This is deliberately not a QK/mask pattern: any
/// producer implementing TilingInterface is eligible, and its operands retain
/// exactly the source graph's values.  Requiring the two proven softmax uses
/// to be the only uses ensures dead-closure cleanup cannot leave a full score
/// tensor beside the recurrence.
static bool hasFusibleScoreComputation(AttentionSemantics semantics) {
  auto scoreResult = mlir::dyn_cast<mlir::OpResult>(semantics.scores);
  mlir::Operation *producer = scoreResult ? scoreResult.getOwner() : nullptr;
  if (!producer || !mlir::isa<mlir::TilingInterface>(producer) ||
      producer->getBlock() != semantics.rowMax->getBlock())
    return false;

  unsigned useCount = 0;
  bool sawRowMaximum = false;
  bool sawShift = false;
  for (mlir::OpOperand &use : scoreResult.getUses()) {
    ++useCount;
    sawRowMaximum |= use.getOwner() == semantics.rowMax.getOperation();
    sawShift |= use.getOwner() == semantics.shifted.getOperation();
    if (use.getOwner() != semantics.rowMax.getOperation() &&
        use.getOwner() != semantics.shifted.getOperation())
      return false;
  }
  return useCount == 2 && sawRowMaximum && sawShift;
}

static mlir::LogicalResult reprovePointDomain(
    mlir::ModuleOp module, const StructuredAlternativeDomain &expected,
    AttentionImplementationKind implementation, std::string *failureReason) {
  mlir::FailureOr<AttentionSemantics> current = mlir::failure();
  if (implementation == AttentionImplementationKind::SplitKV) {
    auto decode =
        wafer::tensor_program_to_tile_region::analyzeDecodeAttentionSemantics(
            module, failureReason);
    if (mlir::failed(decode))
      return mlir::failure();
    current = decode->attention;
  } else {
    current = wafer::tensor_program_to_tile_region::analyzeAttentionSemantics(
        module, failureReason);
    if (mlir::failed(current))
      return mlir::failure();
  }
  if (!current->outputType || !current->outputType.hasStaticShape() ||
      !llvm::equal(current->outputType.getShape(), expected.outputShape) ||
      expected.reductionShape.size() != 1 ||
      expected.reductionShape.front() != current->reductionExtent) {
    setFailureReason(
        failureReason,
        "structured attention point domain changed before materialization");
    return mlir::failure();
  }
  if (!hasFusibleScoreComputation(*current)) {
    setFailureReason(
        failureReason,
        "structured attention score computation is not interface-fusible");
    return mlir::failure();
  }
  return mlir::success();
}

class OnlineAttentionAlternativePoint final
    : public StructuredImplementationAlternativePoint {
public:
  OnlineAttentionAlternativePoint(
      StructuredAlternativeKey key, StructuredAlternativeDomain domain,
      StructuredAlternativeParameters parameters,
      StructuredAlternativeStructuralEstimates estimates)
      : StructuredImplementationAlternativePoint(
            std::move(key), std::move(domain), std::move(parameters),
            std::move(estimates)) {}

  mlir::LogicalResult materialize(
      mlir::ModuleOp isolatedStructuredModule, std::string *failureReason,
      StructuredImplementationAlternativeMaterialization *result) const final {
    if (result)
      *result = {};
    mlir::FailureOr<TensorProgramScope> scope =
        getMaterializationScope(isolatedStructuredModule, failureReason);
    if (mlir::failed(scope) ||
        mlir::failed(reprovePointDomain(isolatedStructuredModule, getDomain(),
                                        AttentionImplementationKind::Online,
                                        failureReason)))
      return mlir::failure();
    llvm::DenseSet<mlir::Operation *> operationsBefore =
        snapshotTopLevelOperations(*scope);
    mlir::LogicalResult materialized =
        wafer::tensor_program_to_tile_region::materializeCompleteFlashTraversal(
            *scope, getParameters().outputTileSizes,
            getParameters().reductionTileSizes,
            AttentionImplementationKind::Online, failureReason);
    if (mlir::succeeded(materialized))
      recordCoveredTopLevelOperations(*scope, operationsBefore, result);
    return materialized;
  }
};

class SplitKVAttentionAlternativePoint final
    : public StructuredImplementationAlternativePoint {
public:
  SplitKVAttentionAlternativePoint(
      StructuredAlternativeKey key, StructuredAlternativeDomain domain,
      StructuredAlternativeParameters parameters,
      StructuredAlternativeStructuralEstimates estimates)
      : StructuredImplementationAlternativePoint(
            std::move(key), std::move(domain), std::move(parameters),
            std::move(estimates)) {}

  mlir::LogicalResult materialize(
      mlir::ModuleOp isolatedStructuredModule, std::string *failureReason,
      StructuredImplementationAlternativeMaterialization *result) const final {
    if (result)
      *result = {};
    mlir::FailureOr<TensorProgramScope> scope =
        getMaterializationScope(isolatedStructuredModule, failureReason);
    if (mlir::failed(scope) ||
        mlir::failed(reprovePointDomain(isolatedStructuredModule, getDomain(),
                                        AttentionImplementationKind::SplitKV,
                                        failureReason)))
      return mlir::failure();
    llvm::DenseSet<mlir::Operation *> operationsBefore =
        snapshotTopLevelOperations(*scope);
    llvm::SmallVector<int64_t, 2> materializerParameters(
        getParameters().reductionTileSizes.begin(),
        getParameters().reductionTileSizes.end());
    materializerParameters.push_back(getParameters().parallelPartitionCount);
    mlir::LogicalResult materialized =
        wafer::tensor_program_to_tile_region::materializeCompleteFlashTraversal(
            *scope, getParameters().outputTileSizes, materializerParameters,
            AttentionImplementationKind::SplitKV, failureReason);
    if (mlir::succeeded(materialized))
      recordCoveredTopLevelOperations(*scope, operationsBefore, result);
    return materialized;
  }
};

enum class PendingPointKind { Online, SplitKV };

struct PendingPoint {
  PendingPointKind kind = PendingPointKind::Online;
  StructuredAlternativeDomain domain;
  StructuredAlternativeParameters parameters;
  StructuredAlternativeStructuralEstimates estimates;
  std::string stableKey;
};

static void appendPendingPoint(
    PendingPointKind kind, const AttentionSemantics &semantics,
    const std::optional<AttentionScoreContractionModel> &contraction,
    const StructuredAlternativeDomain &domain,
    StructuredAlternativeParameters parameters,
    std::vector<PendingPoint> &pending) {
  mlir::FailureOr<StructuredAlternativeStructuralEstimates> estimates =
      buildStructuredAlternativeStructuralEstimates(domain, parameters);
  if (mlir::failed(estimates))
    return;
  estimates->estimatedPeakLiveBytes =
      estimateAttentionPeakLiveBytes(semantics, parameters, contraction);
  estimates->estimatedComputeScalarOps =
      estimateAttentionComputeScalarOps(semantics, parameters, contraction);
  llvm::StringRef implementation =
      kind == PendingPointKind::Online ? "online-recurrence" : "split-kv";
  pending.push_back({kind, domain, std::move(parameters), *estimates, {}});
  pending.back().stableKey = buildPointKey(
      implementation, pending.back().domain, pending.back().parameters);
}

} // namespace

llvm::StringRef
AttentionImplementationAlternativeProvider::getStableKey() const {
  return kProviderStableKey;
}

mlir::LogicalResult AttentionImplementationAlternativeProvider::query(
    mlir::ModuleOp currentStructuredModule,
    const StructuredImplementationAlternativeQuery &query,
    StructuredImplementationAlternativePoints &points,
    std::string *failureReason) const {
  if (!currentStructuredModule) {
    setFailureReason(failureReason,
                     "structured attention query requires a module");
    return mlir::failure();
  }

  bool supportsSplitKV = false;
  std::optional<AttentionSemantics> semantics;
  mlir::FailureOr<DecodeAttentionSemantics> decode =
      wafer::tensor_program_to_tile_region::analyzeDecodeAttentionSemantics(
          currentStructuredModule);
  if (mlir::succeeded(decode)) {
    semantics = decode->attention;
    supportsSplitKV = true;
  } else {
    mlir::FailureOr<AttentionSemantics> online =
        wafer::tensor_program_to_tile_region::analyzeAttentionSemantics(
            currentStructuredModule);
    if (mlir::failed(online))
      return mlir::success();
    semantics = *online;
  }
  if (!hasFusibleScoreComputation(*semantics))
    return mlir::success();

  StructuredAlternativeDomain domain;
  domain.outputShape.assign(semantics->outputType.getShape().begin(),
                            semantics->outputType.getShape().end());
  domain.reductionShape.push_back(semantics->reductionExtent);

  std::vector<StructuredAlternativeTileShape> outputTiles = canonicalTileShapes(
      query.outputTileShapes, query.outputTileSizeSeeds, domain.outputShape);
  appendCompleteValueChannelTileShapes(outputTiles, query.outputTileSizeSeeds,
                                       domain.outputShape);
  std::vector<StructuredAlternativeTileShape> reductionTiles =
      canonicalTileShapes(query.reductionTileShapes,
                          query.reductionTileSizeSeeds, domain.reductionShape);
  std::vector<int64_t> splitCounts = canonicalSplitCounts(
      query.parallelPartitionCounts, semantics->reductionExtent);
  if (outputTiles.empty() || reductionTiles.empty())
    return mlir::success();

  std::optional<AttentionScoreContractionModel> contraction =
      buildAttentionScoreContractionModel(*semantics);

  std::vector<PendingPoint> pending;
  for (const StructuredAlternativeTileShape &outputTile : outputTiles)
    for (const StructuredAlternativeTileShape &reductionTile : reductionTiles) {
      StructuredAlternativeParameters online;
      online.outputTileSizes.assign(outputTile.begin(), outputTile.end());
      online.reductionTileSizes.assign(reductionTile.begin(),
                                       reductionTile.end());
      appendPendingPoint(PendingPointKind::Online, *semantics, contraction,
                         domain, std::move(online), pending);

      if (!supportsSplitKV)
        continue;
      for (int64_t splitCount : splitCounts) {
        StructuredAlternativeParameters split;
        split.outputTileSizes.assign(outputTile.begin(), outputTile.end());
        split.reductionTileSizes.assign(reductionTile.begin(),
                                        reductionTile.end());
        split.parallelPartitionCount = splitCount;
        appendPendingPoint(PendingPointKind::SplitKV, *semantics, contraction,
                           domain, std::move(split), pending);
      }
    }

  llvm::sort(pending, [](const PendingPoint &left, const PendingPoint &right) {
    return left.stableKey < right.stableKey;
  });
  pending.erase(
      std::unique(pending.begin(), pending.end(),
                  [](const PendingPoint &left, const PendingPoint &right) {
                    return left.stableKey == right.stableKey;
                  }),
      pending.end());

  for (auto [ordinal, point] : llvm::enumerate(pending)) {
    StructuredAlternativeKey key{point.stableKey,
                                 static_cast<uint64_t>(ordinal)};
    if (point.kind == PendingPointKind::Online) {
      points.push_back(std::make_unique<OnlineAttentionAlternativePoint>(
          std::move(key), std::move(point.domain), std::move(point.parameters),
          std::move(point.estimates)));
      continue;
    }
    points.push_back(std::make_unique<SplitKVAttentionAlternativePoint>(
        std::move(key), std::move(point.domain), std::move(point.parameters),
        std::move(point.estimates)));
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
