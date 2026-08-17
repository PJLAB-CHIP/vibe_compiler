//===- StructuredDAGExactDemandQuery.cpp - Typed logical demand -----------===//

#include "StructuredDAGExactDemandQuery.h"

#include "StructuredDAGCandidateSchedule.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace wafer::compiler::detail {
namespace {

using analysis::DemandEdgeKind;
using analysis::ExactDemandResult;
using analysis::ExactDemandStatus;
using analysis::IREpoch;
using analysis::IndexRelation;
using analysis::IndexRelationResult;
using analysis::IndexRelationStatus;
using analysis::IndexSetResult;
using analysis::LogicalExecutionShard;
using analysis::LogicalNodeTrial;
using analysis::LogicalShardTrial;
using analysis::LogicalTileBinding;
using analysis::TileRole;

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

analysis::IndexRelationResult relationFailure(IndexRelationStatus status,
                                              llvm::StringRef reason) {
  analysis::IndexRelationResult result;
  result.status = status;
  result.reason = reason.str();
  return result;
}

ExactDemandResult demandResult(ExactDemandStatus status, uint32_t edge,
                               llvm::StringRef detail) {
  ExactDemandResult result;
  result.status = status;
  result.edge = edge;
  result.detail = detail.str();
  return result;
}

/// Typed content key for one rectangular demand image: the image is a pure
/// function of the edge relation and the destination rectangle, so the
/// rectangle content itself is the identity.
struct RectangularDemandKey {
  StructuredDAGEdgeID edge = 0;
  llvm::SmallVector<int64_t, 8> rectangle;

  bool operator<(const RectangularDemandKey &other) const {
    if (edge != other.edge)
      return edge < other.edge;
    return std::lexicographical_compare(
        rectangle.begin(), rectangle.end(), other.rectangle.begin(),
        other.rectangle.end());
  }
};

/// One pure support path between a producer result and a consumer operand.
struct SupportPath {
  /// Support operations in producer-to-consumer order; `sourceOperands[i]`
  /// is the operand of `ops[i]` that carries this producer's data.
  llvm::SmallVector<mlir::Operation *, 4> ops;
  llvm::SmallVector<unsigned, 4> sourceOperands;
};

enum class SupportPathStatus : uint8_t {
  Resolved,
  /// One support operation carries this producer's data through several of
  /// its tensor operands; a single edge relation cannot express that.
  Ambiguous,
  /// A support operation has no typed exact relation.
  Unsupported,
  /// The path crosses side effects, another scheduled operation, or a cycle.
  InvalidStructure,
};

struct SupportPathResolution {
  SupportPathStatus status = SupportPathStatus::InvalidStructure;
  std::optional<SupportPath> path;
  std::string detail;
};

SupportPathResolution resolveSupportPath(mlir::Operation *producer,
                                         unsigned producerResult,
                                         mlir::Operation *consumer,
                                         unsigned consumerOperand) {
  SupportPathResolution resolution;
  if (!producer || !consumer ||
      producer->getBlock() != consumer->getBlock() ||
      producerResult >= producer->getNumResults() ||
      consumerOperand >= consumer->getNumOperands()) {
    resolution.detail = "demand does not name one in-block structured dependency";
    return resolution;
  }
  mlir::Value source = producer->getResult(producerResult);
  mlir::Value current = consumer->getOperand(consumerOperand);
  llvm::DenseMap<mlir::Value, bool> sourceDependencyMemo;
  std::function<bool(mlir::Value)> dependsOnSource =
      [&](mlir::Value value) -> bool {
    if (value == source)
      return true;
    auto found = sourceDependencyMemo.find(value);
    if (found != sourceDependencyMemo.end())
      return found->second;
    mlir::Operation *operation = value.getDefiningOp();
    // DAG nodes are exactly the DestinationStyle + Tiling structured
    // operations. The DPS check comes first so tensor ops that declare
    // TilingInterface only through external models (pad/pack/unpack) are
    // short-circuited in contexts where those models are not registered.
    if (!operation || operation->getBlock() != producer->getBlock() ||
        (mlir::isa<mlir::DestinationStyleOpInterface>(operation) &&
         mlir::isa<mlir::TilingInterface>(operation))) {
      sourceDependencyMemo[value] = false;
      return false;
    }
    // Break malformed cycles conservatively while descending.
    sourceDependencyMemo[value] = false;
    const bool result =
        llvm::any_of(operation->getOperands(), [&](mlir::Value operand) {
          return mlir::isa<mlir::RankedTensorType>(operand.getType()) &&
                 dependsOnSource(operand);
        });
    sourceDependencyMemo[value] = result;
    return result;
  };

  SupportPath path;
  llvm::DenseSet<mlir::Value> visited;
  while (current != source) {
    if (!current || !visited.insert(current).second) {
      resolution.detail = "structured dependency support chain is cyclic";
      return resolution;
    }
    mlir::Operation *operation = current.getDefiningOp();
    if (!operation || operation->getBlock() != producer->getBlock() ||
        operation == producer || !mlir::isMemoryEffectFree(operation)) {
      resolution.detail =
          "structured dependency is not a pure in-block support chain";
      return resolution;
    }
    // DAG nodes are exactly the DestinationStyle + Tiling structured
    // operations; crossing one means the dependency was misclassified by the
    // DAG. The DPS check comes first so tensor ops that declare
    // TilingInterface only through external models (pad/pack/unpack) are
    // short-circuited in contexts where those models are not registered.
    if (mlir::isa<mlir::DestinationStyleOpInterface>(operation) &&
        mlir::isa<mlir::TilingInterface>(operation)) {
      resolution.detail =
          "structured dependency crosses another scheduled operation";
      return resolution;
    }
    std::optional<unsigned> tensorSource;
    for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
      if (!mlir::isa<mlir::RankedTensorType>(operand.getType()))
        continue;
      if (dependsOnSource(operand)) {
        if (tensorSource) {
          resolution.status = SupportPathStatus::Ambiguous;
          resolution.detail =
              "structured dependency support operation has multiple paths "
              "from one producer";
          return resolution;
        }
        tensorSource = index;
      }
    }
    if (!tensorSource) {
      resolution.detail =
          "structured dependency support operation has no tensor source";
      return resolution;
    }
    path.ops.push_back(operation);
    path.sourceOperands.push_back(*tensorSource);
    current = operation->getOperand(*tensorSource);
  }
  std::reverse(path.ops.begin(), path.ops.end());
  std::reverse(path.sourceOperands.begin(), path.sourceOperands.end());
  resolution.status = SupportPathStatus::Resolved;
  resolution.path = std::move(path);
  return resolution;
}

IndexRelationResult
insertSliceRemainderRelation(llvm::ArrayRef<int64_t> resultShape,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes);

/// Resolves a static OpFoldResult vector; any dynamic entry fails closed.
std::optional<llvm::SmallVector<int64_t, 4>>
resolveConstantIndexes(llvm::ArrayRef<mlir::OpFoldResult> values) {
  llvm::SmallVector<int64_t, 4> result;
  for (mlir::OpFoldResult value : values) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(value);
    if (!constant)
      return std::nullopt;
    result.push_back(*constant);
  }
  return result;
}

/// Exact relation of one support operation: result coordinates (destination)
/// to the given source operand coordinates.
IndexRelationResult deriveSupportRelation(mlir::Operation *operation,
                                          unsigned sourceOperand) {
  mlir::Value source = operation->getOperand(sourceOperand);
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getResult(0).getType());
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  if (!resultType || !sourceType || !resultType.hasStaticShape() ||
      !sourceType.hasStaticShape())
    return relationFailure(IndexRelationStatus::Unsupported,
                           "support operation requires static tensor types");

  return llvm::TypeSwitch<mlir::Operation *, IndexRelationResult>(operation)
      .Case<mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(
          [&](auto) {
            return IndexRelation::staticReshape(resultType.getShape(),
                                                sourceType.getShape());
          })
      .Case<mlir::tensor::ExtractSliceOp>([&](auto op) {
        std::optional<llvm::SmallVector<int64_t, 4>> offsets =
            resolveConstantIndexes(op.getMixedOffsets());
        if (!offsets)
          return relationFailure(IndexRelationStatus::Unsupported,
                                 "extract_slice offset is not constant");
        std::optional<llvm::SmallVector<int64_t, 4>> strides =
            resolveConstantIndexes(op.getMixedStrides());
        if (!strides)
          return relationFailure(IndexRelationStatus::Unsupported,
                                 "extract_slice stride is not constant");
        return IndexRelation::staticSlice(resultType.getShape(),
                                          sourceType.getShape(), *offsets,
                                          *strides);
      })
      .Case<mlir::tensor::InsertSliceOp>([&](auto op) {
        if (source == op.getSource()) {
          if (!op.hasUnitStride())
            return relationFailure(
                IndexRelationStatus::Unsupported,
                "strided insert_slice is not a typed support relation");
          std::optional<llvm::SmallVector<int64_t, 4>> offsets =
              resolveConstantIndexes(op.getMixedOffsets());
          if (!offsets)
            return relationFailure(IndexRelationStatus::Unsupported,
                                   "insert_slice offset is not constant");
          // The destination is the whole result, the source only the
          // inserted piece: destination coordinates outside the piece have
          // no image and are excluded by the source bounds.
          return IndexRelation::staticInsertSlice(
              resultType.getShape(), sourceType.getShape(), *offsets);
        }
        if (source == op.getDest()) {
          if (!op.hasUnitStride())
            return relationFailure(
                IndexRelationStatus::Unsupported,
                "strided insert_slice remainder is not a typed support "
                "relation");
          std::optional<llvm::SmallVector<int64_t, 4>> offsets =
              resolveConstantIndexes(op.getMixedOffsets());
          if (!offsets)
            return relationFailure(IndexRelationStatus::Unsupported,
                                   "insert_slice offset is not constant");
          auto sourcePieceType = mlir::dyn_cast<mlir::RankedTensorType>(
              op.getSource().getType());
          if (!sourcePieceType)
            return relationFailure(IndexRelationStatus::Invalid,
                                   "insert_slice source is not a tensor");
          return insertSliceRemainderRelation(
              resultType.getShape(), *offsets, sourcePieceType.getShape());
        }
        return relationFailure(
            IndexRelationStatus::Invalid,
            "insert_slice support source does not carry the dependency");
      })
      .Case<mlir::tensor::PadOp>([&](auto op) {
        std::optional<llvm::SmallVector<int64_t, 4>> lowPad =
            resolveConstantIndexes(op.getMixedLowPad());
        if (!lowPad)
          return relationFailure(IndexRelationStatus::Unsupported,
                                 "tensor.pad low pad is not constant");
        // The pad interior is an insert of the source at the low-pad offset;
        // pad boundary coordinates carry the constant fill and have no image.
        return IndexRelation::staticInsertSlice(resultType.getShape(),
                                                sourceType.getShape(),
                                                *lowPad);
      })
      .Case<mlir::linalg::TransposeOp>([&](auto op) {
        mlir::AffineMap permutation = mlir::AffineMap::getPermutationMap(
            op.getPermutation(), operation->getContext());
        return IndexRelation::fromAffineMap(permutation,
                                            resultType.getShape(),
                                            sourceType.getShape());
      })
      .Case<mlir::tensor::CastOp>([&](auto) {
        return IndexRelation::staticReshape(resultType.getShape(),
                                            sourceType.getShape());
      })
      .Default([&](mlir::Operation *op) {
        return relationFailure(IndexRelationStatus::Unsupported,
                               (llvm::Twine("support operation ") +
                                op->getName().getStringRef() +
                                " has no typed exact relation")
                                   .str());
      });
}

/// Exact relation of the destination operand of one unit-stride
/// insert_slice: result coordinates outside the inserted piece map to equal
/// destination-operand coordinates; coordinates inside the piece are not part
/// of this producer's demand.
IndexRelationResult
insertSliceRemainderRelation(llvm::ArrayRef<int64_t> resultShape,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes) {
  if (resultShape.size() != offsets.size() ||
      offsets.size() != sizes.size())
    return relationFailure(IndexRelationStatus::Invalid,
                           "insert slice remainder requires one offset and "
                           "size per dimension");
  for (auto [index, size] : llvm::enumerate(sizes)) {
    int64_t pieceEnd = 0;
    if (offsets[index] < 0 || size < 0 ||
        llvm::AddOverflow(offsets[index], size, pieceEnd) ||
        pieceEnd > resultShape[index])
      return relationFailure(IndexRelationStatus::Invalid,
                             "insert slice remainder piece exceeds result "
                             "domain");
  }

  std::optional<mlir::presburger::PresburgerSet> complement;
  for (unsigned index = 0; index < resultShape.size(); ++index) {
    for (bool upper : {false, true}) {
      llvm::SmallVector<int64_t, 4> slabOffsets(resultShape.size(), 0);
      llvm::SmallVector<int64_t, 4> slabSizes(resultShape.begin(),
                                              resultShape.end());
      if (!upper) {
        if (offsets[index] == 0)
          continue;
        slabSizes[index] = offsets[index];
      } else {
        int64_t start = offsets[index] + sizes[index];
        if (start >= resultShape[index])
          continue;
        slabOffsets[index] = start;
        slabSizes[index] = resultShape[index] - start;
      }
      IndexSetResult slab =
          IndexRelation::staticRectangularDomain(slabOffsets, slabSizes);
      if (!slab.isExact())
        return relationFailure(slab.status, slab.reason);
      complement = complement ? complement->unionSet(*slab.set) : *slab.set;
    }
  }
  if (!complement) {
    // The inserted piece covers the whole result: the destination operand
    // contributes nothing and the relation has an empty destination domain.
    llvm::SmallVector<int64_t, 4> zeroOffsets(resultShape.size(), 0);
    llvm::SmallVector<int64_t, 4> zeroSizes(resultShape.size(), 0);
    IndexSetResult empty =
        IndexRelation::staticRectangularDomain(zeroOffsets, zeroSizes);
    if (!empty.isExact())
      return relationFailure(empty.status, empty.reason);
    complement = *empty.set;
  }
  IndexRelationResult identity = IndexRelation::identity(resultShape);
  if (!identity.isExact())
    return identity;
  return identity.get()->intersectDestinationDomain(*complement);
}

const LogicalNodeTrial *findNodeTrial(const LogicalShardTrial &trial,
                                      uint32_t node) {
  auto found = llvm::find_if(trial.nodes, [&](const LogicalNodeTrial &entry) {
    return entry.node == node;
  });
  return found == trial.nodes.end() ? nullptr : &*found;
}

} // namespace

class StructuredDAGExactDemandQuery::Impl {
public:
  Impl(const StructuredDAGAnalysis &dag, IREpoch epoch)
      : dag(dag), epoch(epoch),
        functionOperation(dag.getFunction().getOperation()),
        bodyOperationCount(
            dag.getFunction().getBody().front().getOperations().size()) {}

  ExactDemandResult query(StructuredDAGEdgeID edgeId,
                          const LogicalShardTrial &trial) {
    // A trial from another IR generation, or a DAG whose function changed
    // under this query, cannot use any cached or derived fact.
    if (!trial.epoch.isValid() || trial.epoch != epoch)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                          "logical trial belongs to another IR epoch");
    // Real IR invalidation: the query's structural snapshot of the borrowed
    // function (operation identity plus body operation count) must still
    // match. An in-place mutation invalidates every derived fact, including
    // the per-edge relation cache; the token is only a borrow identity.
    if (functionOperation != dag.getFunction().getOperation() ||
        bodyOperationCount !=
            dag.getFunction().getBody().front().getOperations().size())
      return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                          "current IR changed under the demand query");
    const StructuredDAGEdge *edge = dag.getEdge(edgeId);
    if (!edge)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                          "demand references an unknown edge");
    const LogicalNodeTrial *producerTrial =
        findNodeTrial(trial, edge->producer);
    const LogicalNodeTrial *consumerTrial =
        findNodeTrial(trial, edge->consumer);
    if (!producerTrial || !consumerTrial ||
        !consumerTrial->completeIterationDomain)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                          "trial does not cover every edge endpoint");
    // Ownership is per producer result; only this edge's result is
    // validated and consumed.
    llvm::SmallVector<const LogicalTileBinding *, 16> resultBindings;
    for (const LogicalTileBinding &binding : producerTrial->bindings)
      if (binding.resultIndex == edge->producerResult)
        resultBindings.push_back(&binding);
    if (resultBindings.empty())
      return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                          "trial producer result has no ownership");
    for (const LogicalTileBinding *binding : resultBindings)
      if (!binding->ownedDomain)
        return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                            "trial producer ownership domain is missing");
    llvm::SmallVector<TileId, 16> sortedTiles;
    for (const LogicalTileBinding *binding : resultBindings)
      sortedTiles.push_back(binding->tile);
    llvm::sort(sortedTiles, [](TileId lhs, TileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    for (auto [lhs, rhs] : llvm::zip(sortedTiles, llvm::drop_begin(sortedTiles)))
      if (lhs == rhs)
        return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                            "trial binds one Tile to several owner domains");

    return deriveDemand(*edge, *producerTrial, *consumerTrial);
  }

private:
  ExactDemandResult deriveDemand(const StructuredDAGEdge &edge,
                                 const LogicalNodeTrial &producerTrial,
                                 const LogicalNodeTrial &consumerTrial) {
    const StructuredDAGNode *producerNode = dag.getNode(edge.producer);
    const StructuredDAGNode *consumerNode = dag.getNode(edge.consumer);
    mlir::Operation *producer = producerNode->operation;
    mlir::Operation *consumer = consumerNode->operation;
    if (edge.producerResult >= producer->getNumResults() ||
        edge.consumerOperand >= consumer->getNumOperands())
      return demandResult(ExactDemandStatus::IndeterminateFailure, edge.id,
                          "demand edge indexes are outside their operations");

    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        producer->getResult(edge.producerResult).getType());
    auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
        consumer->getOperand(edge.consumerOperand).getType());
    if (!producerType || !producerType.hasStaticShape())
      return demandResult(ExactDemandStatus::UnsupportedSemanticRelation,
                          edge.id,
                          "producer result requires a static tensor type");
    if (!operandType || !operandType.hasStaticShape())
      return demandResult(ExactDemandStatus::UnsupportedSemanticRelation,
                          edge.id,
                          "consumer operand requires a static tensor type");

    // Typed dependency role: DPS data input or DPS init operand. Anything
    // else is a missing upstream role, never a placement failure.
    DemandEdgeKind kind = DemandEdgeKind::DataInput;
    if (auto consumerDps =
            mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumer)) {
      const bool isDataInput =
          llvm::any_of(consumerDps.getDpsInputOperands(),
                       [&](mlir::OpOperand *operand) {
                         return operand->getOperandNumber() ==
                                edge.consumerOperand;
                       });
      bool isInitInput = false;
      for (int64_t index = 0; index < consumerDps.getNumDpsInits(); ++index)
        if (consumerDps.getDpsInitOperand(index)->getOperandNumber() ==
            edge.consumerOperand)
          isInitInput = true;
      if (isInitInput)
        kind = DemandEdgeKind::InitInput;
      else if (!isDataInput)
        return demandResult(ExactDemandStatus::IndeterminateFailure, edge.id,
                            "consumer operand has no typed DPS role");
    } else {
      return demandResult(ExactDemandStatus::UnsupportedSemanticRelation,
                          edge.id,
                          "consumer is not a DestinationStyle structured op");
    }

    SupportPathResolution resolution = resolveSupportPath(
        producer, edge.producerResult, consumer, edge.consumerOperand);
    if (resolution.status == SupportPathStatus::Ambiguous)
      return demandResult(ExactDemandStatus::UnsupportedSemanticRelation,
                          edge.id, resolution.detail);
    if (resolution.status == SupportPathStatus::InvalidStructure)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edge.id,
                          resolution.detail);
    if (!resolution.path)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edge.id,
                          "support path resolution produced no path");

    // Placement-independent relation proof, derived once per edge.
    IndexRelationResult relation = deriveEdgeRelation(edge, producerType,
                                                      operandType,
                                                      *resolution.path);
    if (!relation.isExact())
      return demandResult(mapIndexRelationStatus(relation.status), edge.id,
                          relation.reason);
    if (relation.get()->getSourceRank() !=
            static_cast<unsigned>(producerType.getRank()) ||
        relation.get()->getDestinationRank() !=
            consumerTrial.completeIterationDomain->getSpace()
                .getNumSetDimVars())
      return demandResult(ExactDemandStatus::IndeterminateFailure, edge.id,
                          "exact demand relation ranks do not meet the trial");

    IndexSetResult demandSet = imageDemand(
        edge.id, *relation.get(), *consumerTrial.completeIterationDomain);
    if (!demandSet.isExact())
      return demandResult(mapIndexRelationStatus(demandSet.status), edge.id,
                          demandSet.reason);

    return deriveCoverage(edge, *relation.get(), producerTrial,
                          consumerTrial, *demandSet.set, kind);
  }

  /// Exact image of one destination domain under the edge relation. A dense
  /// rectangle under a projected rectangle pattern relation images
  /// arithmetically; every other case uses the generic Presburger image.
  /// Rectangular images are memoized by (edge, rectangle content): the image
  /// is a pure function of the relation and the domain, the relation is
  /// immutable per edge during this query's lifetime, and the key is the
  /// domain content itself. The memo never enters legality decisions beyond
  /// the exact image it names.
  IndexSetResult imageDemand(StructuredDAGEdgeID edgeId,
                             const IndexRelation &relation,
                             const mlir::presburger::PresburgerSet &domain) {
    analysis::StaticRectangularIndexSetResult rectangle =
        IndexSetResult{IndexRelationStatus::Exact, domain, {}}
            .getExactStaticRectangularDomain();
    if (rectangle.isExact()) {
      RectangularDemandKey key;
      key.edge = edgeId;
      key.rectangle.append(rectangle.domain->offsets.begin(),
                           rectangle.domain->offsets.end());
      key.rectangle.append(rectangle.domain->sizes.begin(),
                           rectangle.domain->sizes.end());
      auto cached = rectangularDemandMemo.find(key);
      if (cached != rectangularDemandMemo.end())
        return IndexSetResult{IndexRelationStatus::Exact, cached->second, {}};
      analysis::StaticRectangularIndexSetResult exactImage =
          relation.getExactStaticRectangularImage(rectangle.domain->offsets,
                                                  rectangle.domain->sizes);
      if (exactImage.isExact()) {
        IndexSetResult result = IndexRelation::staticRectangularDomain(
            exactImage.domain->offsets, exactImage.domain->sizes);
        if (result.isExact()) {
          rectangularDemandMemo.emplace(std::move(key), *result.set);
          return result;
        }
      }
    }
    return relation.image(domain);
  }

  IndexRelationResult deriveEdgeRelation(const StructuredDAGEdge &edge,
                                         mlir::RankedTensorType producerType,
                                         mlir::RankedTensorType operandType,
                                         const SupportPath &path) {
    auto cached = relationCache.find(edge.id);
    if (cached != relationCache.end())
      return IndexRelationResult{IndexRelationStatus::Exact,
                                 cached->second, {}};

    auto consumerLinalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(
        dag.getNode(edge.consumer)->operation);
    if (!consumerLinalg)
      return relationFailure(IndexRelationStatus::Unsupported,
                             "consumer is not a static Linalg structured op");
    llvm::SmallVector<int64_t, 4> loopShape =
        consumerLinalg.getStaticLoopRanges();
    llvm::SmallVector<mlir::AffineMap, 4> maps =
        consumerLinalg.getIndexingMapsArray();
    if (llvm::any_of(loopShape, [](int64_t extent) { return extent < 0; }))
      return relationFailure(IndexRelationStatus::Unsupported,
                             "consumer requires static loop ranges");
    if (edge.consumerOperand >= maps.size())
      return relationFailure(IndexRelationStatus::Invalid,
                             "dependent edge indexing map is missing");

    IndexRelationResult result = IndexRelation::fromAffineMap(
        maps[edge.consumerOperand], loopShape, operandType.getShape());
    if (!result.isExact())
      return result;
    // Compose the support chain from the consumer side back to the producer
    // result: current relation maps iteration coordinates to the support
    // value nearest the consumer; each support relation maps one support
    // result to its source operand.
    IndexRelation current = std::move(*result.relation);
    for (auto [operation, sourceOperand] :
         llvm::zip(llvm::reverse(path.ops), llvm::reverse(path.sourceOperands))) {
      IndexRelationResult support = deriveSupportRelation(operation, sourceOperand);
      if (!support.isExact())
        return support;
      IndexRelationResult composed = current.compose(*support.relation);
      if (!composed.isExact())
        return composed;
      current = std::move(*composed.relation);
    }
    if (current.getSourceRank() !=
        static_cast<unsigned>(producerType.getRank()))
      return relationFailure(IndexRelationStatus::Invalid,
                             "support chain does not meet the producer "
                             "result domain");
    // The producer result shape bounds the relation source domain.
    IndexSetResult producerDomain =
        IndexRelation::staticDomain(producerType.getShape());
    if (!producerDomain.isExact())
      return relationFailure(producerDomain.status, producerDomain.reason);
    IndexRelationResult bounded =
        current.intersectSourceDomain(*producerDomain.set);
    if (!bounded.isExact())
      return bounded;
    relationCache.emplace(edge.id, *bounded.relation);
    return bounded;
  }

  ExactDemandResult
  deriveCoverage(const StructuredDAGEdge &edge, const IndexRelation &relation,
                 const LogicalNodeTrial &producerTrial,
                 const LogicalNodeTrial &consumerTrial,
                 const mlir::presburger::PresburgerSet &demand,
                 DemandEdgeKind kind) {
    ExactDemandResult result;
    result.status = ExactDemandStatus::Satisfied;
    result.edge = edge.id;
    result.producerResult = edge.producerResult;
    result.consumerOperand = edge.consumerOperand;
    result.dependencyKind = kind;
    if (consumerTrial.completeIterationDomain)
      result.consumerIterationDomain =
          *consumerTrial.completeIterationDomain;
    result.producerDemand = demand;

    // Only this edge's producer result participates in the coverage proof;
    // other results of the same node carry their own ownership.
    llvm::SmallVector<const LogicalTileBinding *, 16> resultBindings;
    for (const LogicalTileBinding &binding : producerTrial.bindings)
      if (binding.resultIndex == edge.producerResult)
        resultBindings.push_back(&binding);

    // Unique-partition owners tile the result domain; an overlap with any
    // other owner is a provable partition contradiction. The witness is the
    // union of every pairwise overlap that involves at least one
    // unique-partition owner. One pass replaces the all-pairs scan: the
    // overlap of an owner with the union of all earlier owners equals the
    // union of its pairwise overlaps, and union accumulation is commutative,
    // so the outcome never depends on the caller's binding enumeration
    // order.
    std::optional<mlir::presburger::PresburgerSet> overlapWitness;
    std::optional<mlir::presburger::PresburgerSet> ownedUnion;
    std::optional<mlir::presburger::PresburgerSet> coveredUnique;
    bool hasReplication = false;
    bool hasPartialContribution = false;
    for (const LogicalTileBinding *binding : resultBindings) {
      const bool unique = binding->role == TileRole::UniquePartition;
      hasReplication |= binding->role == TileRole::ExplicitReplication;
      hasPartialContribution |=
          binding->role == TileRole::PartialReductionContribution;
      const std::optional<mlir::presburger::PresburgerSet> &prior =
          unique ? ownedUnion : coveredUnique;
      if (prior) {
        mlir::presburger::PresburgerSet overlap =
            binding->ownedDomain->intersect(*prior);
        if (!overlap.isIntegerEmpty())
          overlapWitness = overlapWitness ? overlapWitness->unionSet(overlap)
                                          : std::move(overlap);
      }
      ownedUnion = ownedUnion ? ownedUnion->unionSet(*binding->ownedDomain)
                              : *binding->ownedDomain;
      if (unique)
        coveredUnique =
            coveredUnique ? coveredUnique->unionSet(*binding->ownedDomain)
                          : *binding->ownedDomain;
      mlir::presburger::PresburgerSet intersection =
          demand.intersect(*binding->ownedDomain);
      if (!intersection.isIntegerEmpty()) {
        if (binding->role == TileRole::PartialReductionContribution)
          result.mergeObligation = true;
        result.ownershipIntersections.push_back(
            analysis::ExactOwnershipIntersection{binding->tile,
                                                 std::move(intersection)});
      }
    }
    if (overlapWitness) {
      result.status = ExactDemandStatus::ProvenLogicalInfeasible;
      result.uncoveredWitness = std::move(*overlapWitness);
      result.detail =
          "unique-partition owners overlap in their result domains";
      return result;
    }

    // Per-destination facts for every consumer execution shard, ordered by
    // Tile id: each shard's exact demand, per-owner intersections and
    // per-shard uncovered witness derive from the same relation proof.
    llvm::SmallVector<const LogicalExecutionShard *, 16> sortedShards;
    for (const LogicalExecutionShard &shard : consumerTrial.executionShards)
      if (shard.executionDomain)
        sortedShards.push_back(&shard);
    llvm::sort(sortedShards, [](const LogicalExecutionShard *lhs,
                                const LogicalExecutionShard *rhs) {
      return lhs->tile.getValue() < rhs->tile.getValue();
    });
    for (const LogicalExecutionShard *shard : sortedShards) {
      IndexSetResult shardDemand =
          imageDemand(edge.id, relation, *shard->executionDomain);
      if (!shardDemand.isExact())
        return demandResult(mapIndexRelationStatus(shardDemand.status),
                            edge.id, shardDemand.reason);
      analysis::ExactDestinationDemand entry;
      entry.destinationTile = shard->tile;
      entry.consumerExecutionDomain = *shard->executionDomain;
      entry.producerDemand = *shardDemand.set;
      // The shard demand is a subset of the whole-edge demand, so only
      // owners with a non-empty whole-edge intersection can contribute; the
      // shard intersection equals the intersection with that smaller set.
      for (const analysis::ExactOwnershipIntersection &owner :
           result.ownershipIntersections) {
        mlir::presburger::PresburgerSet intersection =
            shardDemand.set->intersect(*owner.set);
        if (!intersection.isIntegerEmpty())
          entry.ownershipIntersections.push_back(
              analysis::ExactOwnershipIntersection{owner.tile,
                                                   std::move(intersection)});
      }
      llvm::sort(entry.ownershipIntersections,
                 [](const analysis::ExactOwnershipIntersection &lhs,
                    const analysis::ExactOwnershipIntersection &rhs) {
                   return lhs.tile.getValue() < rhs.tile.getValue();
                 });
      mlir::presburger::PresburgerSet shardUncovered =
          shardDemand.set->subtract(*ownedUnion);
      if (!shardUncovered.isIntegerEmpty())
        entry.uncoveredWitness = std::move(shardUncovered);
      result.perDestination.push_back(std::move(entry));
    }

    mlir::presburger::PresburgerSet uncovered =
        demand.subtract(*ownedUnion);
    if (!uncovered.isIntegerEmpty()) {
      result.status = ExactDemandStatus::ProvenLogicalInfeasible;
      result.uncoveredWitness = std::move(uncovered);
      result.detail =
          "producer shard ownership does not cover exact demand";
      return result;
    }
    result.role = hasPartialContribution
                      ? TileRole::PartialReductionContribution
                      : (hasReplication ? TileRole::ExplicitReplication
                                        : TileRole::UniquePartition);
    // Deterministic outcome: per-owner intersections ordered by Tile id,
    // never by the caller's binding enumeration order.
    llvm::sort(result.ownershipIntersections,
               [](const analysis::ExactOwnershipIntersection &lhs,
                  const analysis::ExactOwnershipIntersection &rhs) {
                 return lhs.tile.getValue() < rhs.tile.getValue();
               });
    return result;
  }

public:
  IREpoch getEpoch() const { return epoch; }

private:
  const StructuredDAGAnalysis &dag;
  IREpoch epoch;
  mlir::Operation *functionOperation = nullptr;
  size_t bodyOperationCount = 0;
  /// Placement-independent relation proofs per edge; the query instance is
  /// bound to one IR epoch and the DAG is immutable during its lifetime.
  std::map<StructuredDAGEdgeID, IndexRelation> relationCache;
  /// Rectangular demand images by (edge, rectangle content); typed-content
  /// keyed, deterministic, epoch-scoped with the query instance.
  std::map<RectangularDemandKey, mlir::presburger::PresburgerSet>
      rectangularDemandMemo;
};

StructuredDAGExactDemandQuery::StructuredDAGExactDemandQuery(
    const StructuredDAGAnalysis &dag, IREpoch epoch)
    : impl(std::make_unique<Impl>(dag, epoch)) {}

StructuredDAGExactDemandQuery::~StructuredDAGExactDemandQuery() = default;
StructuredDAGExactDemandQuery::StructuredDAGExactDemandQuery(
    StructuredDAGExactDemandQuery &&) noexcept = default;
StructuredDAGExactDemandQuery &StructuredDAGExactDemandQuery::operator=(
    StructuredDAGExactDemandQuery &&) noexcept = default;

IREpoch StructuredDAGExactDemandQuery::getEpoch() const {
  return impl->getEpoch();
}

ExactDemandResult
StructuredDAGExactDemandQuery::query(StructuredDAGEdgeID edge,
                                     const LogicalShardTrial &trial) {
  return impl->query(edge, trial);
}

/// Exact result-space shard of one Tile in the balanced placement domain. A
/// result whose rank cannot express the shard axis yields its complete
/// domain (the replication fallback).
mlir::FailureOr<mlir::presburger::PresburgerSet>
deriveBalancedShardDomain(mlir::RankedTensorType type,
                          unsigned shardDimension,
                          llvm::ArrayRef<TileId> group, TileId tile,
                          std::string *failureReason) {
  if (!type || !type.hasStaticShape() || group.empty())
    return fail<mlir::presburger::PresburgerSet>(
        failureReason, "logical trial placement has an invalid static shard");
  if (shardDimension >= static_cast<unsigned>(type.getRank())) {
    IndexSetResult full = IndexRelation::staticDomain(type.getShape());
    if (!full.isExact())
      return fail<mlir::presburger::PresburgerSet>(failureReason,
                                                   full.reason);
    return *full.set;
  }
  auto found = llvm::find(group, tile);
  if (found == group.end())
    return fail<mlir::presburger::PresburgerSet>(
        failureReason, "logical trial Tile is outside its node group");
  const int64_t extent = type.getDimSize(shardDimension);
  if (extent <= 0 || group.size() > static_cast<size_t>(extent))
    return fail<mlir::presburger::PresburgerSet>(
        failureReason,
        "logical trial cannot form nonempty balanced shards");
  const int64_t ordinal = std::distance(group.begin(), found);
  const int64_t participants = static_cast<int64_t>(group.size());
  const int64_t base = extent / participants;
  const int64_t larger = extent % participants;
  llvm::SmallVector<int64_t, 4> offsets(type.getRank(), 0);
  llvm::SmallVector<int64_t, 4> sizes(type.getShape().begin(),
                                      type.getShape().end());
  offsets[shardDimension] =
      ordinal * base + std::min<int64_t>(ordinal, larger);
  sizes[shardDimension] = base + (ordinal < larger);
  IndexSetResult domain =
      IndexRelation::staticRectangularDomain(offsets, sizes);
  if (!domain.isExact())
    return fail<mlir::presburger::PresburgerSet>(failureReason,
                                                 domain.reason);
  return *domain.set;
}

mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
buildBalancedOwnership(mlir::RankedTensorType type,
                       unsigned shardDimension,
                       llvm::ArrayRef<TileId> group, uint32_t resultIndex,
                       std::string *failureReason) {
  llvm::SmallVector<analysis::LogicalTileBinding, 16> result;
  if (!type || !type.hasStaticShape() || group.empty())
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason, "logical trial placement has an invalid static shard");
  const bool replicated =
      shardDimension >= static_cast<unsigned>(type.getRank());
  for (TileId tile : group) {
    auto domain =
        deriveBalancedShardDomain(type, shardDimension, group, tile,
                                  failureReason);
    if (mlir::failed(domain))
      return mlir::failure();
    result.push_back(analysis::LogicalTileBinding{
        tile, resultIndex, *domain,
        replicated ? analysis::TileRole::ExplicitReplication
                   : analysis::TileRole::UniquePartition});
  }
  return result;
}

/// Per-Tile consumer execution shards: for every Tile of the group, the
/// exact iteration-space domain that produces the Tile's result shards,
/// computed as the union over every result of the preimage of the result's
/// balanced shard under the result indexing map. Presburger budget
/// exhaustion yields an empty shard list: the query reports the same
/// machinery failure from the whole-edge derivation as IndeterminateFailure,
/// so the trial build never pre-empts the typed outcome.
mlir::FailureOr<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>
deriveConsumerExecutionShards(mlir::linalg::LinalgOp linalg,
                              const StructuredDAGNodePlacement &placement,
                              std::string *failureReason) {
  llvm::SmallVector<analysis::LogicalExecutionShard, 16> result;
  llvm::SmallVector<int64_t, 4> loopShape = linalg.getStaticLoopRanges();
  // The per-result iteration-to-result relations are Tile-independent and
  // derived once per node.
  llvm::SmallVector<std::pair<mlir::RankedTensorType, IndexRelation>, 2>
      resultRelations;
  for (unsigned resultIndex = 0;
       resultIndex < linalg->getNumResults(); ++resultIndex) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        linalg->getResult(resultIndex).getType());
    if (!resultType || !resultType.hasStaticShape())
      return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
          failureReason,
          "consumer execution shards require static tensor results");
    mlir::AffineMap resultMap = linalg.getIndexingMapMatchingResult(
        linalg->getResult(resultIndex));
    if (!resultMap || resultMap.getNumDims() != loopShape.size() ||
        resultMap.getNumResults() !=
            static_cast<unsigned>(resultType.getRank()))
      return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
          failureReason,
          "consumer execution shard has an inconsistent result map");
    IndexRelationResult iterationToResult = IndexRelation::fromAffineMap(
        resultMap, loopShape, resultType.getShape());
    if (!iterationToResult.isExact()) {
      if (iterationToResult.status == IndexRelationStatus::ResourceExhausted)
        return llvm::SmallVector<analysis::LogicalExecutionShard, 16>{};
      return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
          failureReason, "consumer execution shard relation is not exact");
    }
    resultRelations.emplace_back(resultType, *iterationToResult.relation);
  }
  for (TileId tile : placement.tiles) {
    std::optional<mlir::presburger::PresburgerSet> executionDomain;
    for (auto &[resultType, iterationToResult] : resultRelations) {
      auto shard = deriveBalancedShardDomain(
          resultType, placement.shardDimension, placement.tiles, tile,
          failureReason);
      if (mlir::failed(shard))
        return mlir::failure();
      IndexSetResult preimage = iterationToResult.preimage(*shard);
      if (!preimage.isExact()) {
        if (preimage.status == IndexRelationStatus::ResourceExhausted)
          return llvm::SmallVector<analysis::LogicalExecutionShard, 16>{};
        return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
            failureReason, "consumer execution shard preimage is not exact");
      }
      executionDomain = executionDomain
                            ? executionDomain->unionSet(*preimage.set)
                            : *preimage.set;
    }
    if (!executionDomain)
      return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
            failureReason, "consumer execution shard is empty");
    result.push_back(analysis::LogicalExecutionShard{tile,
                                                     executionDomain});
  }
  return result;
}

mlir::FailureOr<analysis::LogicalShardTrial> buildEdgeShardTrial(
    const StructuredDAGAnalysis &dag,
    const StructuredDAGNodePlacement &producerPlacement,
    const StructuredDAGNodePlacement &consumerPlacement,
    analysis::IREpoch epoch, std::string *failureReason) {
  analysis::LogicalShardTrial trial;
  trial.epoch = epoch;
  for (const StructuredDAGNodePlacement &placement :
       {producerPlacement, consumerPlacement}) {
    const StructuredDAGNode *node = dag.getNode(placement.node);
    if (!node || !node->operation)
      return fail<analysis::LogicalShardTrial>(
          failureReason, "dependent edge demand has an unavailable node");
    auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(node->operation);
    if (!linalg)
      return fail<analysis::LogicalShardTrial>(
          failureReason,
          "dependent edge demand requires a static Linalg node");
    llvm::SmallVector<int64_t, 4> loopShape = linalg.getStaticLoopRanges();
    if (llvm::any_of(loopShape, [](int64_t extent) { return extent < 0; }))
      return fail<analysis::LogicalShardTrial>(
          failureReason, "dependent edge demand requires static loop ranges");
    analysis::IndexSetResult iterationDomain =
        analysis::IndexRelation::staticDomain(loopShape);
    if (!iterationDomain.isExact())
      return fail<analysis::LogicalShardTrial>(failureReason,
                                               iterationDomain.reason);

    analysis::LogicalNodeTrial entry;
    entry.node = placement.node;
    entry.completeIterationDomain = *iterationDomain.set;
    auto executionShards =
        deriveConsumerExecutionShards(linalg, placement, failureReason);
    if (mlir::failed(executionShards))
      return mlir::failure();
    entry.executionShards = std::move(*executionShards);
    for (uint32_t resultIndex = 0;
         resultIndex < node->operation->getNumResults(); ++resultIndex) {
      auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
          node->operation->getResult(resultIndex).getType());
      if (!resultType || !resultType.hasStaticShape())
        return fail<analysis::LogicalShardTrial>(
            failureReason,
            "dependent edge demand requires static tensor results");
      mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
          ownership = buildBalancedOwnership(
              resultType, placement.shardDimension, placement.tiles,
              resultIndex, failureReason);
      if (mlir::failed(ownership))
        return mlir::failure();
      entry.bindings.append(std::move(*ownership));
    }
    trial.nodes.push_back(std::move(entry));
  }
  llvm::sort(trial.nodes, [](const analysis::LogicalNodeTrial &lhs,
                             const analysis::LogicalNodeTrial &rhs) {
    return lhs.node < rhs.node;
  });
  return trial;
}

mlir::FailureOr<analysis::LogicalShardTrial> buildLogicalShardTrial(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> nodePlacements,
    analysis::IREpoch epoch, std::string *failureReason) {
  if (nodePlacements.size() != dag.getNodes().size())
    return fail<analysis::LogicalShardTrial>(
        failureReason, "logical trial must place every DAG node once");
  llvm::SmallVector<const StructuredDAGNodePlacement *, 16> placements(
      dag.getNodes().size(), nullptr);
  for (const StructuredDAGNodePlacement &placement : nodePlacements) {
    if (placement.node >= placements.size() || placements[placement.node] ||
        placement.tiles.empty())
      return fail<analysis::LogicalShardTrial>(
          failureReason,
          "logical trial has an invalid or duplicate node placement");
    placements[placement.node] = &placement;
  }

  analysis::LogicalShardTrial trial;
  trial.epoch = epoch;
  trial.nodes.reserve(dag.getNodes().size());
  for (const StructuredDAGNode &node : dag.getNodes()) {
    const StructuredDAGNodePlacement &placement = *placements[node.id];
    mlir::Operation *operation = node.operation;
    auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
    if (!linalg)
      return fail<analysis::LogicalShardTrial>(
          failureReason, "logical trial requires a static Linalg node");
    llvm::SmallVector<int64_t, 4> loopShape = linalg.getStaticLoopRanges();
    if (llvm::any_of(loopShape, [](int64_t extent) { return extent < 0; }))
      return fail<analysis::LogicalShardTrial>(
          failureReason, "logical trial requires static loop ranges");
    analysis::IndexSetResult iterationDomain =
        analysis::IndexRelation::staticDomain(loopShape);
    if (!iterationDomain.isExact())
      return fail<analysis::LogicalShardTrial>(failureReason,
                                               iterationDomain.reason);

    analysis::LogicalNodeTrial nodeTrial;
    nodeTrial.node = node.id;
    nodeTrial.completeIterationDomain = *iterationDomain.set;
    auto executionShards =
        deriveConsumerExecutionShards(linalg, placement, failureReason);
    if (mlir::failed(executionShards))
      return mlir::failure();
    nodeTrial.executionShards = std::move(*executionShards);
    for (uint32_t resultIndex = 0; resultIndex < operation->getNumResults();
         ++resultIndex) {
      auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
          operation->getResult(resultIndex).getType());
      if (!resultType || !resultType.hasStaticShape())
        return fail<analysis::LogicalShardTrial>(
            failureReason,
            "logical trial requires static tensor results");
      mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
          ownership = buildBalancedOwnership(
              resultType, placement.shardDimension, placement.tiles,
              resultIndex, failureReason);
      if (mlir::failed(ownership))
        return mlir::failure();
      nodeTrial.bindings.append(std::move(*ownership));
    }
    trial.nodes.push_back(std::move(nodeTrial));
  }
  llvm::sort(trial.nodes, [](const analysis::LogicalNodeTrial &lhs,
                             const analysis::LogicalNodeTrial &rhs) {
    return lhs.node < rhs.node;
  });
  return trial;
}

} // namespace wafer::compiler::detail
