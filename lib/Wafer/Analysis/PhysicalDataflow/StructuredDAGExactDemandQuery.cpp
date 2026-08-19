//===- StructuredDAGExactDemandQuery.cpp - Typed logical demand -----------===//

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"

#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace wafer::compiler::detail {
namespace {

using analysis::ConsumerInputDemand;
using analysis::DemandEdgeKind;
using analysis::ExactDemandResult;
using analysis::ExactDemandStatus;
using analysis::IndexRelation;
using analysis::IndexRelationResult;
using analysis::IndexRelationStatus;
using analysis::IndexSetResult;
using analysis::IREpoch;
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

using StaticRectangle = analysis::StaticRectangularIndexSet;

static std::optional<StaticRectangle>
getStaticRectangle(const mlir::presburger::PresburgerSet &set) {
  analysis::StaticRectangularIndexSetResult rectangle = IndexSetResult{
      IndexRelationStatus::Exact, set, {}}.getExactStaticRectangularDomain();
  if (!rectangle.isExact())
    return std::nullopt;
  return std::move(*rectangle.domain);
}

static bool exactSetsEqual(const mlir::presburger::PresburgerSet &lhs,
                           const mlir::presburger::PresburgerSet &rhs);

static mlir::presburger::PresburgerSet
subtractExactSets(const mlir::presburger::PresburgerSet &lhs,
                  const mlir::presburger::PresburgerSet &rhs) {
  if (lhs.getSpace().getNumSetDimVars() != 0)
    return lhs.subtract(rhs);
  if (lhs.isIntegerEmpty() || rhs.isIntegerEmpty())
    return lhs;
  return mlir::presburger::PresburgerSet::getEmpty(lhs.getSpace());
}

static std::optional<StaticRectangle>
intersectRectangles(const StaticRectangle &lhs, const StaticRectangle &rhs) {
  if (lhs.offsets.size() != rhs.offsets.size() ||
      lhs.sizes.size() != lhs.offsets.size() ||
      rhs.sizes.size() != rhs.offsets.size())
    return std::nullopt;
  StaticRectangle result;
  result.offsets.reserve(lhs.offsets.size());
  result.sizes.reserve(lhs.offsets.size());
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.offsets, lhs.sizes, rhs.offsets, rhs.sizes)) {
    int64_t lhsLimit = 0;
    int64_t rhsLimit = 0;
    if (lhsSize <= 0 || rhsSize <= 0 ||
        llvm::AddOverflow(lhsOffset, lhsSize, lhsLimit) ||
        llvm::AddOverflow(rhsOffset, rhsSize, rhsLimit))
      return std::nullopt;
    const int64_t offset = std::max(lhsOffset, rhsOffset);
    const int64_t limit = std::min(lhsLimit, rhsLimit);
    if (offset >= limit)
      return std::nullopt;
    result.offsets.push_back(offset);
    result.sizes.push_back(limit - offset);
  }
  return result;
}

static bool rectangleContains(const StaticRectangle &outer,
                              const StaticRectangle &inner) {
  if (outer.offsets.size() != inner.offsets.size() ||
      outer.sizes.size() != outer.offsets.size() ||
      inner.sizes.size() != inner.offsets.size())
    return false;
  for (auto [outerOffset, outerSize, innerOffset, innerSize] : llvm::zip_equal(
           outer.offsets, outer.sizes, inner.offsets, inner.sizes)) {
    int64_t outerLimit = 0;
    int64_t innerLimit = 0;
    if (outerSize <= 0 || innerSize <= 0 ||
        llvm::AddOverflow(outerOffset, outerSize, outerLimit) ||
        llvm::AddOverflow(innerOffset, innerSize, innerLimit) ||
        innerOffset < outerOffset || innerLimit > outerLimit)
      return false;
  }
  return true;
}

static bool sameRectangle(const StaticRectangle &lhs,
                          const StaticRectangle &rhs) {
  return lhs.offsets == rhs.offsets && lhs.sizes == rhs.sizes;
}

static std::optional<mlir::presburger::PresburgerSet>
materializeRectangle(const StaticRectangle &rectangle) {
  IndexSetResult set = IndexRelation::staticRectangularDomain(rectangle.offsets,
                                                              rectangle.sizes);
  if (!set.isExact())
    return std::nullopt;
  return std::move(*set.set);
}

enum class RectangleCoverageStatus : uint8_t {
  Covered,
  Uncovered,
  ResourceExhausted,
};

struct RectangleCoverageResult {
  RectangleCoverageStatus status = RectangleCoverageStatus::ResourceExhausted;
  std::optional<StaticRectangle> uncoveredWitness;
};

/// Proves coverage of one integer rectangle by a finite rectangle union using
/// only half-open interval arithmetic. Every recursive slab boundary comes
/// from an input rectangle, so an uncovered leaf is itself an exact nonempty
/// rectangular witness. The work bound makes unusual high-rank/many-piece
/// inputs fail closed instead of falling into unbounded generic Presburger
/// equality/subtraction.
static RectangleCoverageResult
proveRectangleCoverage(const StaticRectangle &target,
                       llvm::ArrayRef<StaticRectangle> pieces) {
  constexpr uint64_t kMaximumSlabVisits = 100000;
  llvm::SmallVector<StaticRectangle, 16> clipped;
  clipped.reserve(pieces.size());
  for (const StaticRectangle &piece : pieces)
    if (std::optional<StaticRectangle> intersection =
            intersectRectangles(target, piece))
      clipped.push_back(std::move(*intersection));

  if (target.offsets.size() != target.sizes.size())
    return {};
  if (clipped.empty())
    return RectangleCoverageResult{RectangleCoverageStatus::Uncovered, target};
  if (target.offsets.empty())
    return RectangleCoverageResult{RectangleCoverageStatus::Covered,
                                   std::nullopt};

  StaticRectangle witness;
  witness.offsets.assign(target.offsets.begin(), target.offsets.end());
  witness.sizes.assign(target.sizes.begin(), target.sizes.end());
  uint64_t slabVisits = 0;
  std::function<RectangleCoverageStatus(unsigned, llvm::ArrayRef<unsigned>)>
      coverDimension = [&](unsigned dimension,
                           llvm::ArrayRef<unsigned> activePieces) {
        if (dimension == target.offsets.size())
          return activePieces.empty() ? RectangleCoverageStatus::Uncovered
                                      : RectangleCoverageStatus::Covered;
        if (activePieces.empty())
          return RectangleCoverageStatus::Uncovered;

        int64_t targetLimit = 0;
        if (llvm::AddOverflow(target.offsets[dimension],
                              target.sizes[dimension], targetLimit))
          return RectangleCoverageStatus::ResourceExhausted;
        llvm::SmallVector<int64_t, 34> boundaries = {target.offsets[dimension],
                                                     targetLimit};
        for (unsigned pieceIndex : activePieces) {
          const StaticRectangle &piece = clipped[pieceIndex];
          int64_t pieceLimit = 0;
          if (llvm::AddOverflow(piece.offsets[dimension],
                                piece.sizes[dimension], pieceLimit))
            return RectangleCoverageStatus::ResourceExhausted;
          boundaries.push_back(piece.offsets[dimension]);
          boundaries.push_back(pieceLimit);
        }
        llvm::sort(boundaries);
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                         boundaries.end());
        for (auto [begin, end] :
             llvm::zip(boundaries, llvm::drop_begin(boundaries))) {
          if (begin >= end)
            continue;
          if (++slabVisits > kMaximumSlabVisits)
            return RectangleCoverageStatus::ResourceExhausted;
          llvm::SmallVector<unsigned, 16> slabPieces;
          for (unsigned pieceIndex : activePieces) {
            const StaticRectangle &piece = clipped[pieceIndex];
            int64_t pieceLimit = 0;
            if (llvm::AddOverflow(piece.offsets[dimension],
                                  piece.sizes[dimension], pieceLimit))
              return RectangleCoverageStatus::ResourceExhausted;
            if (piece.offsets[dimension] <= begin && end <= pieceLimit)
              slabPieces.push_back(pieceIndex);
          }
          RectangleCoverageStatus status =
              coverDimension(dimension + 1, slabPieces);
          if (status != RectangleCoverageStatus::Covered) {
            witness.offsets[dimension] = begin;
            witness.sizes[dimension] = end - begin;
            return status;
          }
        }
        return RectangleCoverageStatus::Covered;
      };

  llvm::SmallVector<unsigned, 16> allPieces;
  for (unsigned index = 0; index < clipped.size(); ++index)
    allPieces.push_back(index);
  RectangleCoverageStatus status = coverDimension(0, allPieces);
  return RectangleCoverageResult{
      status, status == RectangleCoverageStatus::Uncovered
                  ? std::optional<StaticRectangle>(std::move(witness))
                  : std::nullopt};
}

static bool sameConstraintRow(llvm::ArrayRef<llvm::DynamicAPInt> lhs,
                              llvm::ArrayRef<llvm::DynamicAPInt> rhs,
                              bool equality) {
  if (lhs == rhs)
    return true;
  if (!equality || lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip_equal(lhs, rhs), [](auto coefficients) {
    return std::get<0>(coefficients) == -std::get<1>(coefficients);
  });
}

static bool sameConstraintRows(const mlir::presburger::IntegerRelation &lhs,
                               const mlir::presburger::IntegerRelation &rhs,
                               bool equality) {
  const unsigned lhsCount =
      equality ? lhs.getNumEqualities() : lhs.getNumInequalities();
  const unsigned rhsCount =
      equality ? rhs.getNumEqualities() : rhs.getNumInequalities();
  if (lhsCount != rhsCount)
    return false;
  llvm::SmallVector<bool, 16> matched(rhsCount, false);
  for (unsigned lhsRow = 0; lhsRow < lhsCount; ++lhsRow) {
    llvm::ArrayRef<llvm::DynamicAPInt> lhsCoefficients =
        equality ? lhs.getEquality(lhsRow) : lhs.getInequality(lhsRow);
    bool found = false;
    for (unsigned rhsRow = 0; rhsRow < rhsCount; ++rhsRow) {
      if (matched[rhsRow])
        continue;
      llvm::ArrayRef<llvm::DynamicAPInt> rhsCoefficients =
          equality ? rhs.getEquality(rhsRow) : rhs.getInequality(rhsRow);
      if (!sameConstraintRow(lhsCoefficients, rhsCoefficients, equality))
        continue;
      matched[rhsRow] = true;
      found = true;
      break;
    }
    if (!found)
      return false;
  }
  return true;
}

/// Compare two Presburger disjuncts after only linear-time local cleanup.
/// Constraint and union ordering are not semantic; local-variable ordering is
/// deliberately not rewritten, so representations requiring actual integer
/// reasoning still fail closed.
static bool
sameDirectConstraintSystem(const mlir::presburger::IntegerRelation &lhsInput,
                           const mlir::presburger::IntegerRelation &rhsInput) {
  if (!lhsInput.getSpace().isEqual(rhsInput.getSpace()))
    return false;
  mlir::presburger::IntegerRelation lhs = lhsInput;
  mlir::presburger::IntegerRelation rhs = rhsInput;
  lhs.removeTrivialRedundancy();
  rhs.removeTrivialRedundancy();
  return sameConstraintRows(lhs, rhs, /*equality=*/true) &&
         sameConstraintRows(lhs, rhs, /*equality=*/false);
}

static bool
sameDirectDisjunctUnion(const mlir::presburger::PresburgerSet &lhs,
                        const mlir::presburger::PresburgerSet &rhs) {
  if (!lhs.getSpace().isEqual(rhs.getSpace()) ||
      lhs.getNumDisjuncts() != rhs.getNumDisjuncts())
    return false;
  llvm::SmallVector<bool, 8> matched(rhs.getNumDisjuncts(), false);
  for (const mlir::presburger::IntegerRelation &lhsDisjunct :
       lhs.getAllDisjuncts()) {
    bool found = false;
    for (auto [rhsIndex, rhsDisjunct] :
         llvm::enumerate(rhs.getAllDisjuncts())) {
      if (matched[rhsIndex] ||
          !sameDirectConstraintSystem(lhsDisjunct, rhsDisjunct))
        continue;
      matched[rhsIndex] = true;
      found = true;
      break;
    }
    if (!found)
      return false;
  }
  return true;
}

/// Equality on the baseline's rectangular unions must remain bounded. The
/// direct representation check handles identical sets at constant structural
/// cost; otherwise each stored rectangular disjunct is checked against the
/// opposite union with finite interval-slab arithmetic. A non-rectangular
/// representation fails closed here instead of entering Presburger set
/// subtraction, whose symbolic lexicographic optimizer has no useful bound
/// for this compiler decision.
static bool exactSetsEqual(const mlir::presburger::PresburgerSet &lhs,
                           const mlir::presburger::PresburgerSet &rhs) {
  if (!lhs.getSpace().isCompatible(rhs.getSpace()))
    return false;
  if (lhs.isObviouslyEqual(rhs) || sameDirectDisjunctUnion(lhs, rhs))
    return true;
  if (lhs.getSpace().getNumSetDimVars() == 0)
    return lhs.isIntegerEmpty() == rhs.isIntegerEmpty();

  analysis::StaticRectangularIndexSetPiecesResult lhsPieces = IndexSetResult{
      IndexRelationStatus::Exact, lhs, {}}.getExactStaticRectangularDisjuncts();
  analysis::StaticRectangularIndexSetPiecesResult rhsPieces = IndexSetResult{
      IndexRelationStatus::Exact, rhs, {}}.getExactStaticRectangularDisjuncts();
  if (!lhsPieces.isExact() || !rhsPieces.isExact())
    return false;
  auto coveredBy = [](llvm::ArrayRef<StaticRectangle> targets,
                      llvm::ArrayRef<StaticRectangle> pieces) {
    return llvm::all_of(targets, [&](const StaticRectangle &target) {
      return proveRectangleCoverage(target, pieces).status ==
             RectangleCoverageStatus::Covered;
    });
  };
  return coveredBy(lhsPieces.domains, rhsPieces.domains) &&
         coveredBy(rhsPieces.domains, lhsPieces.domains);
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
    return std::lexicographical_compare(rectangle.begin(), rectangle.end(),
                                        other.rectangle.begin(),
                                        other.rectangle.end());
  }
};

/// One pure producer-to-consumer tensor chain between a producer result and a
/// consumer operand.
struct ProducerToConsumerChain {
  /// Support operations in producer-to-consumer order; `sourceOperands[i]`
  /// is the operand of `ops[i]` that carries this producer's data.
  llvm::SmallVector<mlir::Operation *, 4> ops;
  llvm::SmallVector<unsigned, 4> sourceOperands;
};

enum class ProducerToConsumerChainStatus : uint8_t {
  Resolved,
  /// One tensor transform carries this producer's data through several of
  /// its tensor operands; a single edge relation cannot express that.
  Ambiguous,
  /// A tensor transform has no typed exact relation.
  Unsupported,
  /// The path crosses side effects, another scheduled operation, or a cycle.
  InvalidStructure,
};

struct ProducerToConsumerChainResult {
  ProducerToConsumerChainStatus status =
      ProducerToConsumerChainStatus::InvalidStructure;
  std::optional<ProducerToConsumerChain> path;
  std::string detail;
};

ProducerToConsumerChainResult resolveProducerToConsumerChain(
    mlir::Operation *producer, unsigned producerResult,
    mlir::Operation *consumer, unsigned consumerOperand) {
  ProducerToConsumerChainResult resolution;
  if (!producer || !consumer || producer->getBlock() != consumer->getBlock() ||
      producerResult >= producer->getNumResults() ||
      consumerOperand >= consumer->getNumOperands()) {
    resolution.detail =
        "demand does not name one in-block structured dependency";
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

  ProducerToConsumerChain path;
  llvm::DenseSet<mlir::Value> visited;
  while (current != source) {
    if (!current || !visited.insert(current).second) {
      resolution.detail =
          "structured dependency producer-to-consumer tensor chain is cyclic";
      return resolution;
    }
    mlir::Operation *operation = current.getDefiningOp();
    if (!operation || operation->getBlock() != producer->getBlock() ||
        operation == producer || !mlir::isMemoryEffectFree(operation)) {
      resolution.detail = "structured dependency is not a pure in-block "
                          "producer-to-consumer tensor chain";
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
          resolution.status = ProducerToConsumerChainStatus::Ambiguous;
          resolution.detail =
              "structured dependency tensor transform has multiple paths "
              "from one producer";
          return resolution;
        }
        tensorSource = index;
      }
    }
    if (!tensorSource) {
      resolution.detail =
          "structured dependency tensor transform has no tensor source";
      return resolution;
    }
    path.ops.push_back(operation);
    path.sourceOperands.push_back(*tensorSource);
    current = operation->getOperand(*tensorSource);
  }
  std::reverse(path.ops.begin(), path.ops.end());
  std::reverse(path.sourceOperands.begin(), path.sourceOperands.end());
  resolution.status = ProducerToConsumerChainStatus::Resolved;
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

/// Exact relation of one tensor transform: result coordinates (destination)
/// to the given source operand coordinates.
IndexRelationResult deriveTensorTransformRelation(mlir::Operation *operation,
                                                  unsigned sourceOperand) {
  mlir::Value source = operation->getOperand(sourceOperand);
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(operation->getResult(0).getType());
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  if (!resultType || !sourceType || !resultType.hasStaticShape() ||
      !sourceType.hasStaticShape())
    return relationFailure(IndexRelationStatus::Unsupported,
                           "tensor transform requires static tensor types");

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
        return IndexRelation::staticSlice(
            resultType.getShape(), sourceType.getShape(), *offsets, *strides);
      })
      .Case<mlir::tensor::InsertSliceOp>([&](auto op) {
        if (source == op.getSource()) {
          if (!op.hasUnitStride())
            return relationFailure(IndexRelationStatus::Unsupported,
                                   "strided insert_slice is not a typed tensor "
                                   "transform relation");
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
          auto sourcePieceType =
              mlir::dyn_cast<mlir::RankedTensorType>(op.getSource().getType());
          if (!sourcePieceType)
            return relationFailure(IndexRelationStatus::Invalid,
                                   "insert_slice source is not a tensor");
          return insertSliceRemainderRelation(resultType.getShape(), *offsets,
                                              sourcePieceType.getShape());
        }
        return relationFailure(IndexRelationStatus::Invalid,
                               "insert_slice tensor transform source does not "
                               "carry the dependency");
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
                                                sourceType.getShape(), *lowPad);
      })
      .Case<mlir::linalg::TransposeOp>([&](auto op) {
        mlir::AffineMap permutation = mlir::AffineMap::getPermutationMap(
            op.getPermutation(), operation->getContext());
        return IndexRelation::fromAffineMap(permutation, resultType.getShape(),
                                            sourceType.getShape());
      })
      .Case<mlir::tensor::CastOp>([&](auto) {
        return IndexRelation::staticReshape(resultType.getShape(),
                                            sourceType.getShape());
      })
      .Default([&](mlir::Operation *op) {
        return relationFailure(IndexRelationStatus::Unsupported,
                               (llvm::Twine("tensor transform ") +
                                op->getName().getStringRef() +
                                " has no typed exact relation")
                                   .str());
      });
}

struct TensorTransformContract {
  analysis::TensorTransformKind kind;
  llvm::SmallVector<unsigned, 2> tensorOperands;
};

/// Returns the same closed support-operation contract consumed by the exact
/// transfer query and by the typed reconstruction recipe.  There is no
/// materializer-side semantic matcher.
std::optional<TensorTransformContract>
getTensorTransformContract(mlir::Operation *operation) {
  if (mlir::isa<mlir::tensor::ExpandShapeOp>(operation))
    return TensorTransformContract{analysis::TensorTransformKind::ExpandShape,
                                   {0}};
  if (mlir::isa<mlir::tensor::CollapseShapeOp>(operation))
    return TensorTransformContract{analysis::TensorTransformKind::CollapseShape,
                                   {0}};
  if (mlir::isa<mlir::tensor::ExtractSliceOp>(operation))
    return TensorTransformContract{analysis::TensorTransformKind::ExtractSlice,
                                   {0}};
  if (mlir::isa<mlir::tensor::InsertSliceOp>(operation))
    return TensorTransformContract{analysis::TensorTransformKind::InsertSlice,
                                   {0, 1}};
  if (mlir::isa<mlir::tensor::PadOp>(operation))
    return TensorTransformContract{analysis::TensorTransformKind::Pad, {0}};
  if (mlir::isa<mlir::tensor::CastOp>(operation))
    return TensorTransformContract{analysis::TensorTransformKind::Cast, {0}};
  return std::nullopt;
}

/// Exact relation of the destination operand of one unit-stride
/// insert_slice: result coordinates outside the inserted piece map to equal
/// destination-operand coordinates; coordinates inside the piece are not part
/// of this producer's demand.
IndexRelationResult
insertSliceRemainderRelation(llvm::ArrayRef<int64_t> resultShape,
                             llvm::ArrayRef<int64_t> offsets,
                             llvm::ArrayRef<int64_t> sizes) {
  if (resultShape.size() != offsets.size() || offsets.size() != sizes.size())
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

std::optional<std::string>
validateExecutionShards(const LogicalNodeTrial &trial) {
  // Empty is the explicit whole-edge-only query form. Once any destination
  // shard is present, the typed assignment must be closed all-and-only.
  if (trial.executionShards.empty())
    return std::nullopt;
  if (!trial.completeIterationDomain)
    return "consumer trial has no complete iteration domain";

  llvm::SmallVector<const LogicalExecutionShard *, 16> shards;
  shards.reserve(trial.executionShards.size());
  for (const LogicalExecutionShard &shard : trial.executionShards) {
    if (!shard.executionDomain)
      return "consumer execution shard has no exact domain";
    shards.push_back(&shard);
  }
  llvm::sort(shards, [](const LogicalExecutionShard *lhs,
                        const LogicalExecutionShard *rhs) {
    return lhs->tile.getValue() < rhs->tile.getValue();
  });
  for (auto [lhs, rhs] : llvm::zip(shards, llvm::drop_begin(shards)))
    if (lhs->tile == rhs->tile)
      return "consumer execution shards repeat one Tile";

  const mlir::presburger::PresburgerSet &complete =
      *trial.completeIterationDomain;
  for (const LogicalExecutionShard *shard : shards) {
    const mlir::presburger::PresburgerSet &domain = *shard->executionDomain;
    if (!domain.getSpace().isCompatible(complete.getSpace()))
      return "consumer execution shard has an incompatible iteration space";
    if (domain.isIntegerEmpty())
      return "consumer execution shard has an empty domain";
  }

  // A rank-zero iteration space contains exactly one point. Generic
  // Presburger subset/equality currently enters symbolic lexicographic
  // optimization, whose contract requires at least one non-symbol variable.
  // The typed shard contract is exact here: one nonempty compatible shard is
  // the complete singleton; two would overlap.
  if (complete.getSpace().getNumSetDimVars() == 0) {
    if (shards.size() != 1)
      return "rank-zero consumer execution requires one singleton shard";
    return std::nullopt;
  }

  // Production balanced shards and future explicitly rectangular assignments
  // carry an exact box per Tile. Prove their all-and-only partition directly;
  // constructing a 16-disjunct union and asking generic Presburger equality
  // loses this proof and can be exponentially slower than the IR itself.
  std::optional<StaticRectangle> completeRectangle =
      getStaticRectangle(complete);
  llvm::SmallVector<StaticRectangle, 16> shardRectangles;
  if (completeRectangle) {
    for (const LogicalExecutionShard *shard : shards) {
      std::optional<StaticRectangle> rectangle =
          getStaticRectangle(*shard->executionDomain);
      if (!rectangle) {
        shardRectangles.clear();
        break;
      }
      shardRectangles.push_back(std::move(*rectangle));
    }
  }
  if (completeRectangle && shardRectangles.size() == shards.size()) {
    for (const StaticRectangle &rectangle : shardRectangles)
      if (!rectangleContains(*completeRectangle, rectangle))
        return "consumer execution shard exceeds the complete iteration "
               "domain";
    for (size_t lhs = 0; lhs < shardRectangles.size(); ++lhs)
      for (size_t rhs = lhs + 1; rhs < shardRectangles.size(); ++rhs)
        if (intersectRectangles(shardRectangles[lhs], shardRectangles[rhs]))
          return "consumer execution shards overlap";
    RectangleCoverageResult coverage =
        proveRectangleCoverage(*completeRectangle, shardRectangles);
    if (coverage.status == RectangleCoverageStatus::ResourceExhausted)
      return "consumer execution shard rectangular coverage exceeds work "
             "budget";
    if (coverage.status == RectangleCoverageStatus::Uncovered)
      return "consumer execution shards do not cover the complete iteration "
             "domain";
    return std::nullopt;
  }

  mlir::presburger::PresburgerSet covered =
      mlir::presburger::PresburgerSet::getEmpty(complete.getSpace());
  for (const LogicalExecutionShard *shard : shards) {
    const mlir::presburger::PresburgerSet &domain = *shard->executionDomain;
    if (!domain.isSubsetOf(complete))
      return "consumer execution shard exceeds the complete iteration domain";
    if (!covered.intersect(domain).isIntegerEmpty())
      return "consumer execution shards overlap";
    covered = covered.unionSet(domain);
  }
  if (!covered.isEqual(complete))
    return "consumer execution shards do not cover the complete iteration "
           "domain";
  return std::nullopt;
}

} // namespace

class StructuredDAGExactDemandQuery::Impl {
public:
  Impl(const StructuredDAGAnalysis &dag, IREpoch epoch)
      : dag(dag), epoch(epoch),
        functionOperation(dag.getFunction().getOperation()),
        functionFingerprint(functionOperation) {}

  ExactDemandResult query(StructuredDAGEdgeID edgeId,
                          const LogicalShardTrial &trial) {
    // A trial from another IR generation, or a DAG whose function changed
    // under this query, cannot use any cached or derived fact.
    if (!trial.epoch.isValid() || trial.epoch != epoch)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                          "logical trial belongs to another IR epoch");
    // Real IR invalidation: MLIR's operation fingerprint observes operation
    // identity, nesting, attributes/properties, operands, successors, result
    // types, blocks and locations across the complete borrowed function. Any
    // in-place semantic mutation invalidates every derived fact, including the
    // per-edge relation cache; the token is only a borrow identity.
    if (functionOperation != dag.getFunction().getOperation() ||
        functionFingerprint != mlir::OperationFingerPrint(functionOperation))
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
    std::optional<std::string> invalidExecutionShards = [&]() {
      wafer::support::ScopedCompileTimingSpan timing(
          "query", "exact-demand", "validate-execution-shards");
      return validateExecutionShards(*consumerTrial);
    }();
    if (invalidExecutionShards)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                          *invalidExecutionShards);
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
    const bool hasPartialContribution =
        llvm::any_of(resultBindings, [](const LogicalTileBinding *binding) {
          return binding->role == TileRole::PartialReductionContribution;
        });
    if (hasPartialContribution != producerTrial->reductionMergeTile.has_value())
      return demandResult(
          ExactDemandStatus::IndeterminateFailure, edgeId,
          "trial partial-reduction ownership and merge owner disagree");
    llvm::SmallVector<TileId, 16> sortedTiles;
    for (const LogicalTileBinding *binding : resultBindings)
      sortedTiles.push_back(binding->tile);
    llvm::sort(sortedTiles, [](TileId lhs, TileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    for (auto [lhs, rhs] :
         llvm::zip(sortedTiles, llvm::drop_begin(sortedTiles)))
      if (lhs == rhs)
        return demandResult(ExactDemandStatus::IndeterminateFailure, edgeId,
                            "trial binds one Tile to several owner domains");

    wafer::support::ScopedCompileTimingSpan timing("query", "exact-demand",
                                                   "derive-edge-demand");
    return deriveDemand(*edge, *producerTrial, *consumerTrial);
  }

  ConsumerInputDemand queryOperand(StructuredDAGNodeID consumerId,
                                   uint32_t consumerOperand,
                                   const LogicalShardTrial &trial) {
    ConsumerInputDemand result;
    result.consumerNode = consumerId;
    result.consumerOperand = consumerOperand;
    auto failOperand = [&](ExactDemandStatus status, llvm::StringRef detail) {
      result.status = status;
      result.detail = detail.str();
      result.perDestination.clear();
      return result;
    };

    if (!trial.epoch.isValid() || trial.epoch != epoch)
      return failOperand(ExactDemandStatus::IndeterminateFailure,
                         "logical trial belongs to another IR epoch");
    if (functionOperation != dag.getFunction().getOperation() ||
        functionFingerprint != mlir::OperationFingerPrint(functionOperation))
      return failOperand(ExactDemandStatus::IndeterminateFailure,
                         "current IR changed under the demand query");
    const StructuredDAGNode *consumerNode = dag.getNode(consumerId);
    const LogicalNodeTrial *consumerTrial = findNodeTrial(trial, consumerId);
    if (!consumerNode || !consumerNode->operation || !consumerTrial ||
        consumerOperand >= consumerNode->operation->getNumOperands())
      return failOperand(ExactDemandStatus::IndeterminateFailure,
                         "operand demand references an unknown consumer");
    result.consumer = consumerNode->operation;
    std::optional<std::string> invalidExecutionShards =
        validateExecutionShards(*consumerTrial);
    if (invalidExecutionShards)
      return failOperand(ExactDemandStatus::IndeterminateFailure,
                         *invalidExecutionShards);
    if (consumerTrial->executionShards.empty())
      return failOperand(
          ExactDemandStatus::IndeterminateFailure,
          "operand demand requires destination execution shards");

    llvm::SmallVector<const StructuredDAGEdge *, 4> edges;
    for (StructuredDAGEdgeID edgeId : consumerNode->incomingEdges) {
      const StructuredDAGEdge *edge = dag.getEdge(edgeId);
      if (edge && edge->consumerOperand == consumerOperand)
        edges.push_back(edge);
    }
    if (edges.empty())
      return failOperand(
          ExactDemandStatus::IndeterminateFailure,
          "consumer operand has no structured producer boundary");
    llvm::sort(edges,
               [](const StructuredDAGEdge *lhs, const StructuredDAGEdge *rhs) {
                 return std::tie(lhs->producer, lhs->producerResult, lhs->id) <
                        std::tie(rhs->producer, rhs->producerResult, rhs->id);
               });

    llvm::SmallVector<ExactDemandResult, 4> edgeDemands;
    edgeDemands.reserve(edges.size());
    for (const StructuredDAGEdge *edge : edges) {
      ExactDemandResult edgeDemand = query(edge->id, trial);
      if (edgeDemand.status != ExactDemandStatus::Satisfied)
        return failOperand(edgeDemand.status, edgeDemand.detail);
      if (edgeDemands.empty())
        result.dependencyKind = edgeDemand.dependencyKind;
      else if (result.dependencyKind != edgeDemand.dependencyKind)
        return failOperand(ExactDemandStatus::IndeterminateFailure,
                           "one consumer operand has inconsistent DPS roles");
      edgeDemands.push_back(std::move(edgeDemand));
    }

    auto consumerLinalg =
        mlir::dyn_cast<mlir::linalg::LinalgOp>(consumerNode->operation);
    auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
        consumerNode->operation->getOperand(consumerOperand).getType());
    if (!consumerLinalg || !operandType || !operandType.hasStaticShape())
      return failOperand(ExactDemandStatus::UnsupportedSemanticRelation,
                         "operand demand requires a static Linalg consumer");
    llvm::SmallVector<mlir::AffineMap, 4> maps =
        consumerLinalg.getIndexingMapsArray();
    llvm::SmallVector<int64_t, 4> loopShape =
        consumerLinalg.getStaticLoopRanges();
    if (consumerOperand >= maps.size() ||
        llvm::any_of(loopShape, [](int64_t extent) { return extent < 0; }))
      return failOperand(ExactDemandStatus::UnsupportedSemanticRelation,
                         "consumer operand has no static indexing relation");
    IndexRelationResult operandRelation = IndexRelation::fromAffineMap(
        maps[consumerOperand], loopShape, operandType.getShape());
    if (!operandRelation.isExact())
      return failOperand(mapIndexRelationStatus(operandRelation.status),
                         operandRelation.reason);

    llvm::SmallVector<const LogicalExecutionShard *, 16> sortedShards;
    for (const LogicalExecutionShard &shard : consumerTrial->executionShards)
      sortedShards.push_back(&shard);
    llvm::sort(sortedShards, [](const LogicalExecutionShard *lhs,
                                const LogicalExecutionShard *rhs) {
      return lhs->tile.getValue() < rhs->tile.getValue();
    });

    struct MutableStep {
      mlir::Operation *operation = nullptr;
      analysis::TensorTransformKind kind = analysis::TensorTransformKind::Cast;
      std::optional<mlir::presburger::PresburgerSet> outputDemand;
      llvm::SmallVector<analysis::TensorTransformInputDemand, 2> operandDemands;
    };
    auto unionInto = [](std::optional<mlir::presburger::PresburgerSet> &target,
                        const mlir::presburger::PresburgerSet &addition) {
      target = target ? target->unionSet(addition) : addition;
    };

    for (const LogicalExecutionShard *shard : sortedShards) {
      if (!shard->executionDomain)
        return failOperand(ExactDemandStatus::IndeterminateFailure,
                           "consumer execution shard has no exact domain");
      IndexSetResult operandSet =
          operandRelation.get()->image(*shard->executionDomain);
      if (!operandSet.isExact())
        return failOperand(mapIndexRelationStatus(operandSet.status),
                           operandSet.reason);

      analysis::ConsumerInputReconstruction recipe;
      recipe.destinationTile = shard->tile;
      recipe.consumerExecutionDomain = *shard->executionDomain;
      recipe.operandDemand = *operandSet.set;

      llvm::SmallVector<std::optional<mlir::presburger::PresburgerSet>, 4>
          propagatedBoundaries(edges.size());
      for (auto [edgeIndex, edge] : llvm::enumerate(edges)) {
        const ExactDemandResult &edgeDemand = edgeDemands[edgeIndex];
        auto destination =
            llvm::find_if(edgeDemand.perDestination,
                          [&](const analysis::ExactDestinationDemand &entry) {
                            return entry.destinationTile == shard->tile;
                          });
        if (destination == edgeDemand.perDestination.end() ||
            !destination->producerDemand)
          return failOperand(ExactDemandStatus::IndeterminateFailure,
                             "edge demand omitted one destination Tile");
        const StructuredDAGNode *producerNode = dag.getNode(edge->producer);
        if (!producerNode || !producerNode->operation)
          return failOperand(ExactDemandStatus::IndeterminateFailure,
                             "operand boundary has no structured producer");
        recipe.boundaries.push_back(analysis::ProducerValueRequirement{
            edge->id, edge->producer, edge->producerResult,
            producerNode->operation, *destination->producerDemand,
            destination->ownershipIntersections});
      }

      llvm::DenseMap<mlir::Value, mlir::presburger::PresburgerSet> processed;
      llvm::SmallVector<MutableStep, 4> mutableSteps;
      ExactDemandStatus propagationStatus = ExactDemandStatus::Satisfied;
      std::string propagationDetail;
      std::function<mlir::LogicalResult(
          mlir::Value, const mlir::presburger::PresburgerSet &)>
          propagate = [&](mlir::Value value,
                          const mlir::presburger::PresburgerSet &demand) {
            if (demand.isIntegerEmpty())
              return mlir::success();
            mlir::presburger::PresburgerSet delta = demand;
            auto already = processed.find(value);
            if (already != processed.end()) {
              delta = subtractExactSets(demand, already->second);
              already->second = already->second.unionSet(demand);
              if (delta.isIntegerEmpty())
                return mlir::success();
            } else {
              processed.try_emplace(value, demand);
            }

            auto opResult = mlir::dyn_cast<mlir::OpResult>(value);
            if (!opResult)
              return mlir::success();
            mlir::Operation *operation = opResult.getOwner();
            if (!operation ||
                operation->getBlock() != &dag.getFunction().getBody().front()) {
              propagationStatus = ExactDemandStatus::IndeterminateFailure;
              propagationDetail =
                  "tensor operand escaped the structured function body";
              return mlir::failure();
            }

            auto structuredEdge =
                llvm::find_if(edges, [&](const StructuredDAGEdge *edge) {
                  const StructuredDAGNode *producer =
                      dag.getNode(edge->producer);
                  return producer && producer->operation == operation &&
                         edge->producerResult == opResult.getResultNumber();
                });
            if (structuredEdge != edges.end()) {
              size_t index = static_cast<size_t>(
                  std::distance(edges.begin(), structuredEdge));
              unionInto(propagatedBoundaries[index], delta);
              return mlir::success();
            }
            if (mlir::isa<mlir::tensor::EmptyOp>(operation)) {
              propagationStatus =
                  ExactDemandStatus::UnsupportedSemanticRelation;
              propagationDetail = "consumer input reconstruction requires an "
                                  "uninitialized tensor";
              return mlir::failure();
            }
            if (operation->hasTrait<mlir::OpTrait::ConstantLike>())
              return mlir::success();
            if (!mlir::isMemoryEffectFree(operation) ||
                opResult.getResultNumber() != 0) {
              propagationStatus =
                  ExactDemandStatus::UnsupportedSemanticRelation;
              propagationDetail = "consumer input reconstruction crosses an "
                                  "unsupported operation";
              return mlir::failure();
            }
            std::optional<TensorTransformContract> contract =
                getTensorTransformContract(operation);
            if (!contract) {
              propagationStatus =
                  ExactDemandStatus::UnsupportedSemanticRelation;
              propagationDetail = (llvm::Twine("tensor transform ") +
                                   operation->getName().getStringRef() +
                                   " has no typed reconstruction contract")
                                      .str();
              return mlir::failure();
            }

            auto step =
                llvm::find_if(mutableSteps, [&](const MutableStep &item) {
                  return item.operation == operation;
                });
            size_t stepIndex = 0;
            if (step == mutableSteps.end()) {
              mutableSteps.push_back(
                  MutableStep{operation, contract->kind, std::nullopt, {}});
              stepIndex = mutableSteps.size() - 1;
            } else {
              stepIndex = static_cast<size_t>(
                  std::distance(mutableSteps.begin(), step));
            }
            unionInto(mutableSteps[stepIndex].outputDemand, delta);
            for (unsigned operandNumber : contract->tensorOperands) {
              if (operandNumber >= operation->getNumOperands() ||
                  !mlir::isa<mlir::RankedTensorType>(
                      operation->getOperand(operandNumber).getType())) {
                propagationStatus = ExactDemandStatus::IndeterminateFailure;
                propagationDetail = "consumer input reconstruction operand "
                                    "contract is malformed";
                return mlir::failure();
              }
              IndexRelationResult relation =
                  deriveTensorTransformRelation(operation, operandNumber);
              if (!relation.isExact()) {
                propagationStatus = mapIndexRelationStatus(relation.status);
                propagationDetail = relation.reason;
                return mlir::failure();
              }
              IndexSetResult operandRead = relation.get()->image(delta);
              if (!operandRead.isExact()) {
                propagationStatus = mapIndexRelationStatus(operandRead.status);
                propagationDetail = operandRead.reason;
                return mlir::failure();
              }
              auto read = llvm::find_if(
                  mutableSteps[stepIndex].operandDemands,
                  [&](const analysis::TensorTransformInputDemand &item) {
                    return item.operand == operandNumber;
                  });
              if (read == mutableSteps[stepIndex].operandDemands.end()) {
                mutableSteps[stepIndex].operandDemands.push_back(
                    analysis::TensorTransformInputDemand{
                        static_cast<uint32_t>(operandNumber),
                        *operandRead.set});
              } else {
                unionInto(read->demand, *operandRead.set);
              }
              if (mlir::failed(propagate(operation->getOperand(operandNumber),
                                         *operandRead.set)))
                return mlir::failure();
            }
            return mlir::success();
          };

      if (mlir::failed(
              propagate(consumerNode->operation->getOperand(consumerOperand),
                        *operandSet.set)))
        return failOperand(propagationStatus, propagationDetail);

      for (auto [edgeIndex, boundary] : llvm::enumerate(recipe.boundaries)) {
        if (!boundary.requiredDomain)
          return failOperand(ExactDemandStatus::IndeterminateFailure,
                             "operand boundary has no exact demand");
        const bool expectedEmpty = boundary.requiredDomain->isIntegerEmpty();
        const bool reached = propagatedBoundaries[edgeIndex].has_value();
        // The per-edge query and this grouped walk consume the same typed
        // producer-to-consumer transform chain. The edge query owns the exact
        // boundary domain and its ownership proof; the grouped walk owns the
        // reconstruction steps. Requiring the walk to reach every and only
        // nonempty boundary proves their structural correspondence. Asking a
        // generic Presburger equality query to re-prove the already composed
        // relation adds no semantic fact and can enter unbounded integer
        // simplex work on large reshape/insert chains.
        if (expectedEmpty == reached)
          return failOperand(
              ExactDemandStatus::IndeterminateFailure,
              "grouped consumer input does not reach every and only required "
              "producer boundary");
      }

      llvm::sort(mutableSteps,
                 [](const MutableStep &lhs, const MutableStep &rhs) {
                   return lhs.operation->isBeforeInBlock(rhs.operation);
                 });
      for (MutableStep &step : mutableSteps) {
        llvm::sort(step.operandDemands,
                   [](const analysis::TensorTransformInputDemand &lhs,
                      const analysis::TensorTransformInputDemand &rhs) {
                     return lhs.operand < rhs.operand;
                   });
        recipe.steps.push_back(analysis::TensorTransform{
            step.operation, /*result=*/0, step.kind,
            std::move(step.outputDemand), std::move(step.operandDemands)});
      }
      result.perDestination.push_back(std::move(recipe));
    }
    result.status = ExactDemandStatus::Satisfied;
    result.detail.clear();
    return result;
  }

  analysis::StaticRectangularIndexSetPiecesResult getExactProducerDemandPieces(
      StructuredDAGEdgeID edgeId,
      const mlir::presburger::PresburgerSet &consumerExecutionDomain) {
    auto failPieces = [](IndexRelationStatus status, llvm::StringRef reason) {
      return analysis::StaticRectangularIndexSetPiecesResult{
          status, {}, reason.str()};
    };
    if (functionOperation != dag.getFunction().getOperation() ||
        functionFingerprint != mlir::OperationFingerPrint(functionOperation))
      return failPieces(IndexRelationStatus::Invalid,
                        "current IR changed under the demand query");
    const StructuredDAGEdge *edge = dag.getEdge(edgeId);
    if (!edge)
      return failPieces(IndexRelationStatus::Invalid,
                        "demand references an unknown edge");
    const StructuredDAGNode *producerNode = dag.getNode(edge->producer);
    const StructuredDAGNode *consumerNode = dag.getNode(edge->consumer);
    if (!producerNode || !consumerNode || !producerNode->operation ||
        !consumerNode->operation ||
        edge->producerResult >= producerNode->operation->getNumResults() ||
        edge->consumerOperand >= consumerNode->operation->getNumOperands())
      return failPieces(IndexRelationStatus::Invalid,
                        "demand edge indexes are outside their operations");
    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        producerNode->operation->getResult(edge->producerResult).getType());
    auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
        consumerNode->operation->getOperand(edge->consumerOperand).getType());
    if (!producerType || !producerType.hasStaticShape() || !operandType ||
        !operandType.hasStaticShape())
      return failPieces(IndexRelationStatus::Unsupported,
                        "demand decomposition requires static tensor types");
    ProducerToConsumerChainResult resolution = resolveProducerToConsumerChain(
        producerNode->operation, edge->producerResult, consumerNode->operation,
        edge->consumerOperand);
    if (!resolution.path)
      return failPieces(resolution.status ==
                                ProducerToConsumerChainStatus::Unsupported
                            ? IndexRelationStatus::Unsupported
                            : IndexRelationStatus::Invalid,
                        resolution.detail);
    IndexRelationResult relation =
        deriveEdgeRelation(*edge, producerType, operandType, *resolution.path);
    if (!relation.isExact())
      return failPieces(relation.status, relation.reason);
    analysis::StaticRectangularIndexSetResult destination = IndexSetResult{
        IndexRelationStatus::Exact,
        consumerExecutionDomain,
        {}}.getExactStaticRectangularDomain();
    if (!destination.isExact())
      return failPieces(destination.status, destination.reason);
    return relation.get()->getExactStaticRectangularImagePieces(
        destination.domain->offsets, destination.domain->sizes);
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
      const bool isDataInput = llvm::any_of(
          consumerDps.getDpsInputOperands(), [&](mlir::OpOperand *operand) {
            return operand->getOperandNumber() == edge.consumerOperand;
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

    ProducerToConsumerChainResult resolution = resolveProducerToConsumerChain(
        producer, edge.producerResult, consumer, edge.consumerOperand);
    if (resolution.status == ProducerToConsumerChainStatus::Ambiguous)
      return demandResult(ExactDemandStatus::UnsupportedSemanticRelation,
                          edge.id, resolution.detail);
    if (resolution.status == ProducerToConsumerChainStatus::InvalidStructure)
      return demandResult(ExactDemandStatus::IndeterminateFailure, edge.id,
                          resolution.detail);
    if (!resolution.path)
      return demandResult(
          ExactDemandStatus::IndeterminateFailure, edge.id,
          "producer-to-consumer tensor chain resolution produced no path");

    // Placement-independent relation proof, derived once per edge.
    IndexRelationResult relation = [&]() {
      wafer::support::ScopedCompileTimingSpan timing("query", "exact-demand",
                                                     "derive-index-relation");
      return deriveEdgeRelation(edge, producerType, operandType,
                                *resolution.path);
    }();
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

    IndexSetResult demandSet = [&]() {
      wafer::support::ScopedCompileTimingSpan timing("query", "exact-demand",
                                                     "image-complete-demand");
      return imageDemand(edge.id, *relation.get(),
                         *consumerTrial.completeIterationDomain);
    }();
    if (!demandSet.isExact())
      return demandResult(mapIndexRelationStatus(demandSet.status), edge.id,
                          demandSet.reason);

    wafer::support::ScopedCompileTimingSpan timing("query", "exact-demand",
                                                   "prove-ownership-coverage");
    return deriveCoverage(edge, *relation.get(), producerTrial, consumerTrial,
                          *demandSet.set, kind);
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
    analysis::StaticRectangularIndexSetResult rectangle = IndexSetResult{
        IndexRelationStatus::Exact,
        domain,
        {}}.getExactStaticRectangularDomain();
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
                                         const ProducerToConsumerChain &path) {
    auto cached = relationCache.find(edge.id);
    if (cached != relationCache.end())
      return IndexRelationResult{
          IndexRelationStatus::Exact, cached->second, {}};

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
    // Compose the producer-to-consumer tensor chain from the consumer side back
    // to the producer result: current relation maps iteration coordinates to
    // the support value nearest the consumer; each tensor transform relation
    // maps one support result to its source operand.
    IndexRelation current = std::move(*result.relation);
    for (auto [operation, sourceOperand] : llvm::zip(
             llvm::reverse(path.ops), llvm::reverse(path.sourceOperands))) {
      IndexRelationResult support =
          deriveTensorTransformRelation(operation, sourceOperand);
      if (!support.isExact())
        return support;
      IndexRelationResult composed = current.compose(*support.relation);
      if (!composed.isExact())
        return composed;
      current = std::move(*composed.relation);
    }
    if (current.getSourceRank() !=
        static_cast<unsigned>(producerType.getRank()))
      return relationFailure(
          IndexRelationStatus::Invalid,
          "producer-to-consumer tensor chain does not meet the producer "
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

  ExactDemandResult deriveRectangularCoverage(
      const StructuredDAGEdge &edge, const IndexRelation &relation,
      const LogicalNodeTrial &consumerTrial,
      llvm::ArrayRef<const LogicalTileBinding *> resultBindings,
      const mlir::presburger::PresburgerSet &demand, DemandEdgeKind kind,
      bool &applicable) {
    applicable = false;
    const std::string edgeDetail =
        (llvm::Twine("edge=") + llvm::Twine(edge.id)).str();
    std::optional<StaticRectangle> demandRectangle = [&]() {
      wafer::support::ScopedCompileTimingSpan timing(
          "query", "exact-demand-rectangle", "recover-complete-demand",
          edgeDetail);
      return getStaticRectangle(demand);
    }();
    if (!demandRectangle)
      return {};

    llvm::SmallVector<StaticRectangle, 16> ownerRectangles;
    ownerRectangles.reserve(resultBindings.size());
    for (const LogicalTileBinding *binding : resultBindings) {
      std::optional<StaticRectangle> rectangle = [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "exact-demand-rectangle", "recover-owner-domain",
            edgeDetail);
        return getStaticRectangle(*binding->ownedDomain);
      }();
      if (!rectangle)
        return {};
      ownerRectangles.push_back(std::move(*rectangle));
    }

    struct DestinationRectangle {
      const LogicalExecutionShard *shard = nullptr;
      mlir::presburger::PresburgerSet demand;
      StaticRectangle rectangle;
    };
    llvm::SmallVector<const LogicalExecutionShard *, 16> sortedShards;
    for (const LogicalExecutionShard &shard : consumerTrial.executionShards)
      if (shard.executionDomain)
        sortedShards.push_back(&shard);
    llvm::sort(sortedShards, [](const LogicalExecutionShard *lhs,
                                const LogicalExecutionShard *rhs) {
      return lhs->tile.getValue() < rhs->tile.getValue();
    });
    llvm::SmallVector<DestinationRectangle, 16> destinationRectangles;
    destinationRectangles.reserve(sortedShards.size());
    for (const LogicalExecutionShard *shard : sortedShards) {
      IndexSetResult shardDemand = [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "exact-demand-rectangle", "image-destination-demand",
            edgeDetail);
        return imageDemand(edge.id, relation, *shard->executionDomain);
      }();
      if (!shardDemand.isExact())
        return {};
      std::optional<StaticRectangle> rectangle = [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "exact-demand-rectangle", "recover-destination-demand",
            edgeDetail);
        return getStaticRectangle(*shardDemand.set);
      }();
      if (!rectangle)
        return {};
      destinationRectangles.push_back(DestinationRectangle{
          shard, std::move(*shardDemand.set), std::move(*rectangle)});
    }
    applicable = true;

    ExactDemandResult result;
    result.status = ExactDemandStatus::Satisfied;
    result.edge = edge.id;
    result.producerResult = edge.producerResult;
    result.consumerOperand = edge.consumerOperand;
    result.dependencyKind = kind;
    if (consumerTrial.completeIterationDomain)
      result.consumerIterationDomain = *consumerTrial.completeIterationDomain;
    result.producerDemand = demand;

    bool hasReplication = false;
    bool hasPartialContribution = false;
    for (const LogicalTileBinding *binding : resultBindings) {
      hasReplication |= binding->role == TileRole::ExplicitReplication;
      hasPartialContribution |=
          binding->role == TileRole::PartialReductionContribution;
    }
    result.role = hasPartialContribution
                      ? TileRole::PartialReductionContribution
                      : (hasReplication ? TileRole::ExplicitReplication
                                        : TileRole::UniquePartition);

    for (auto [binding, owner] :
         llvm::zip_equal(resultBindings, ownerRectangles)) {
      std::optional<StaticRectangle> intersection =
          intersectRectangles(*demandRectangle, owner);
      if (!intersection)
        continue;
      std::optional<mlir::presburger::PresburgerSet> intersectionSet =
          materializeRectangle(*intersection);
      if (!intersectionSet) {
        result.status = ExactDemandStatus::IndeterminateFailure;
        result.detail = "cannot materialize rectangular ownership demand";
        return result;
      }
      if (binding->role == TileRole::PartialReductionContribution)
        result.mergeObligation = true;
      result.ownershipIntersections.push_back(
          analysis::ExactOwnershipIntersection{binding->tile,
                                               std::move(*intersectionSet)});
    }

    std::optional<mlir::presburger::PresburgerSet> overlapWitness;
    for (size_t lhs = 0; lhs < resultBindings.size(); ++lhs) {
      for (size_t rhs = lhs + 1; rhs < resultBindings.size(); ++rhs) {
        if (resultBindings[lhs]->role != TileRole::UniquePartition &&
            resultBindings[rhs]->role != TileRole::UniquePartition)
          continue;
        std::optional<StaticRectangle> overlap =
            intersectRectangles(ownerRectangles[lhs], ownerRectangles[rhs]);
        if (!overlap)
          continue;
        std::optional<mlir::presburger::PresburgerSet> overlapSet =
            materializeRectangle(*overlap);
        if (!overlapSet) {
          result.status = ExactDemandStatus::IndeterminateFailure;
          result.detail = "cannot materialize rectangular ownership overlap";
          return result;
        }
        overlapWitness = overlapWitness ? overlapWitness->unionSet(*overlapSet)
                                        : std::move(*overlapSet);
      }
    }
    if (overlapWitness) {
      result.status = ExactDemandStatus::ProvenLogicalInfeasible;
      result.uncoveredWitness = std::move(*overlapWitness);
      result.detail = "unique-partition owners overlap in their result domains";
      return result;
    }

    for (const DestinationRectangle &destination : destinationRectangles) {
      analysis::ExactDestinationDemand entry;
      entry.destinationTile = destination.shard->tile;
      entry.consumerExecutionDomain = *destination.shard->executionDomain;
      entry.producerDemand = destination.demand;
      for (auto [binding, owner] :
           llvm::zip_equal(resultBindings, ownerRectangles)) {
        std::optional<StaticRectangle> intersection =
            intersectRectangles(destination.rectangle, owner);
        if (!intersection)
          continue;
        std::optional<mlir::presburger::PresburgerSet> intersectionSet =
            materializeRectangle(*intersection);
        if (!intersectionSet) {
          result.status = ExactDemandStatus::IndeterminateFailure;
          result.detail =
              "cannot materialize per-destination rectangular ownership";
          return result;
        }
        entry.ownershipIntersections.push_back(
            analysis::ExactOwnershipIntersection{binding->tile,
                                                 std::move(*intersectionSet)});
      }
      RectangleCoverageResult coverage =
          proveRectangleCoverage(destination.rectangle, ownerRectangles);
      if (coverage.status == RectangleCoverageStatus::ResourceExhausted) {
        result.status = ExactDemandStatus::IndeterminateFailure;
        result.detail =
            "per-destination rectangular coverage exceeds work budget";
        return result;
      }
      if (coverage.status == RectangleCoverageStatus::Uncovered) {
        std::optional<mlir::presburger::PresburgerSet> witness =
            materializeRectangle(*coverage.uncoveredWitness);
        if (!witness) {
          result.status = ExactDemandStatus::IndeterminateFailure;
          result.detail = "cannot materialize rectangular uncovered witness";
          return result;
        }
        entry.uncoveredWitness = std::move(*witness);
      }
      result.perDestination.push_back(std::move(entry));
    }

    RectangleCoverageResult coverage =
        proveRectangleCoverage(*demandRectangle, ownerRectangles);
    if (coverage.status == RectangleCoverageStatus::ResourceExhausted) {
      result.status = ExactDemandStatus::IndeterminateFailure;
      result.detail = "rectangular ownership coverage exceeds work budget";
      return result;
    }
    if (coverage.status == RectangleCoverageStatus::Uncovered) {
      std::optional<mlir::presburger::PresburgerSet> witness =
          materializeRectangle(*coverage.uncoveredWitness);
      if (!witness) {
        result.status = ExactDemandStatus::IndeterminateFailure;
        result.detail = "cannot materialize rectangular uncovered witness";
        return result;
      }
      result.status = ExactDemandStatus::ProvenLogicalInfeasible;
      result.uncoveredWitness = std::move(*witness);
      result.detail = "producer shard ownership does not cover exact demand";
    }
    return result;
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
      result.consumerIterationDomain = *consumerTrial.completeIterationDomain;
    result.producerDemand = demand;
    result.reductionMergeTile = producerTrial.reductionMergeTile;

    // Only this edge's producer result participates in the coverage proof;
    // other results of the same node carry their own ownership.
    llvm::SmallVector<const LogicalTileBinding *, 16> resultBindings;
    for (const LogicalTileBinding &binding : producerTrial.bindings)
      if (binding.resultIndex == edge.producerResult)
        resultBindings.push_back(&binding);
    llvm::sort(resultBindings, [](const LogicalTileBinding *lhs,
                                  const LogicalTileBinding *rhs) {
      return lhs->tile.getValue() < rhs->tile.getValue();
    });

    bool rectangularCoverageApplicable = false;
    ExactDemandResult rectangularCoverage =
        deriveRectangularCoverage(edge, relation, consumerTrial, resultBindings,
                                  demand, kind, rectangularCoverageApplicable);
    if (rectangularCoverageApplicable) {
      rectangularCoverage.reductionMergeTile = producerTrial.reductionMergeTile;
      return rectangularCoverage;
    }

    wafer::support::ScopedCompileTimingSpan genericTiming(
        "query", "exact-demand", "generic-ownership-coverage");

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
        coveredUnique = coveredUnique
                            ? coveredUnique->unionSet(*binding->ownedDomain)
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
    result.role = hasPartialContribution
                      ? TileRole::PartialReductionContribution
                      : (hasReplication ? TileRole::ExplicitReplication
                                        : TileRole::UniquePartition);
    // The bindings were ordered by Tile id before any set accumulation, so
    // successful and failure payloads share the same deterministic order.
    llvm::sort(result.ownershipIntersections,
               [](const analysis::ExactOwnershipIntersection &lhs,
                  const analysis::ExactOwnershipIntersection &rhs) {
                 return lhs.tile.getValue() < rhs.tile.getValue();
               });
    if (overlapWitness) {
      result.status = ExactDemandStatus::ProvenLogicalInfeasible;
      result.uncoveredWitness = std::move(*overlapWitness);
      result.detail = "unique-partition owners overlap in their result domains";
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
        return demandResult(mapIndexRelationStatus(shardDemand.status), edge.id,
                            shardDemand.reason);
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
          subtractExactSets(*shardDemand.set, *ownedUnion);
      if (!shardUncovered.isIntegerEmpty())
        entry.uncoveredWitness = std::move(shardUncovered);
      result.perDestination.push_back(std::move(entry));
    }

    mlir::presburger::PresburgerSet uncovered =
        subtractExactSets(demand, *ownedUnion);
    if (!uncovered.isIntegerEmpty()) {
      result.status = ExactDemandStatus::ProvenLogicalInfeasible;
      result.uncoveredWitness = std::move(uncovered);
      result.detail = "producer shard ownership does not cover exact demand";
      return result;
    }
    return result;
  }

public:
  IREpoch getEpoch() const { return epoch; }

private:
  const StructuredDAGAnalysis &dag;
  IREpoch epoch;
  mlir::Operation *functionOperation = nullptr;
  mlir::OperationFingerPrint functionFingerprint;
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

ConsumerInputDemand
StructuredDAGExactDemandQuery::queryOperand(StructuredDAGNodeID consumer,
                                            uint32_t consumerOperand,
                                            const LogicalShardTrial &trial) {
  return impl->queryOperand(consumer, consumerOperand, trial);
}

analysis::StaticRectangularIndexSetPiecesResult
StructuredDAGExactDemandQuery::getExactProducerDemandPieces(
    StructuredDAGEdgeID edge,
    const mlir::presburger::PresburgerSet &consumerExecutionDomain) {
  return impl->getExactProducerDemandPieces(edge, consumerExecutionDomain);
}

/// Exact result-space shard of one Tile in the balanced placement domain. A
/// result whose rank cannot express the shard axis yields its complete
/// domain (the replication fallback).
mlir::FailureOr<mlir::presburger::PresburgerSet> deriveBalancedPartitionDomain(
    llvm::ArrayRef<int64_t> shape, unsigned shardDimension,
    llvm::ArrayRef<TileId> group, TileId tile, std::string *failureReason) {
  if (shape.empty() ||
      llvm::any_of(shape, [](int64_t extent) { return extent < 0; }) ||
      shardDimension >= shape.size() || group.empty())
    return fail<mlir::presburger::PresburgerSet>(
        failureReason, "logical trial placement has an invalid static shard");
  auto found = llvm::find(group, tile);
  if (found == group.end())
    return fail<mlir::presburger::PresburgerSet>(
        failureReason, "logical trial Tile is outside its node group");
  const int64_t extent = shape[shardDimension];
  if (extent <= 0 || group.size() > static_cast<size_t>(extent))
    return fail<mlir::presburger::PresburgerSet>(
        failureReason, "logical trial cannot form nonempty balanced shards");
  const int64_t ordinal = std::distance(group.begin(), found);
  const int64_t participants = static_cast<int64_t>(group.size());
  const int64_t base = extent / participants;
  const int64_t larger = extent % participants;
  llvm::SmallVector<int64_t, 4> offsets(shape.size(), 0);
  llvm::SmallVector<int64_t, 4> sizes(shape.begin(), shape.end());
  offsets[shardDimension] = ordinal * base + std::min<int64_t>(ordinal, larger);
  sizes[shardDimension] = base + (ordinal < larger);
  IndexSetResult domain =
      IndexRelation::staticRectangularDomain(offsets, sizes);
  if (!domain.isExact())
    return fail<mlir::presburger::PresburgerSet>(failureReason, domain.reason);
  return *domain.set;
}

mlir::FailureOr<mlir::presburger::PresburgerSet>
deriveBalancedShardDomain(mlir::RankedTensorType type, unsigned shardDimension,
                          llvm::ArrayRef<TileId> group, TileId tile,
                          std::string *failureReason) {
  if (!type || !type.hasStaticShape() || group.empty())
    return fail<mlir::presburger::PresburgerSet>(
        failureReason, "logical trial placement has an invalid static shard");
  if (shardDimension >= static_cast<unsigned>(type.getRank())) {
    IndexSetResult full = IndexRelation::staticDomain(type.getShape());
    if (!full.isExact())
      return fail<mlir::presburger::PresburgerSet>(failureReason, full.reason);
    return *full.set;
  }
  return deriveBalancedPartitionDomain(type.getShape(), shardDimension, group,
                                       tile, failureReason);
}

mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
buildBalancedOwnership(mlir::RankedTensorType type, unsigned shardDimension,
                       llvm::ArrayRef<TileId> group, uint32_t resultIndex,
                       std::string *failureReason) {
  llvm::SmallVector<analysis::LogicalTileBinding, 16> result;
  if (!type || !type.hasStaticShape() || group.empty())
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason, "logical trial placement has an invalid static shard");
  const bool replicated =
      shardDimension >= static_cast<unsigned>(type.getRank());
  for (TileId tile : group) {
    auto domain = deriveBalancedShardDomain(type, shardDimension, group, tile,
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

/// Per-Tile execution shards are defined directly in iteration space by the
/// complete iterator factor vector and its row-major physical embedding.
/// Result ownership is derived afterwards as the exact image of these
/// disjoint shards; partitioned reduction images retain explicit partial roles.
mlir::FailureOr<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>
deriveConsumerExecutionShards(mlir::linalg::LinalgOp linalg,
                              const StructuredDAGNodePlacement &placement,
                              std::string *failureReason) {
  llvm::SmallVector<analysis::LogicalExecutionShard, 16> result;
  llvm::SmallVector<int64_t, 4> loopShape = linalg.getStaticLoopRanges();
  llvm::SmallVector<uint32_t, 4> factors = placement.iteratorPartitionFactors;
  if (factors.size() != loopShape.size() || placement.tiles.empty())
    return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
        failureReason,
        "logical trial requires one factor for every structured iterator");

  uint64_t participantCount = 1;
  for (auto [factor, extent] : llvm::zip_equal(factors, loopShape)) {
    if (factor == 0 || extent <= 0 || factor > static_cast<uint64_t>(extent) ||
        participantCount > std::numeric_limits<uint64_t>::max() / factor)
      return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
          failureReason, "logical trial has an invalid iterator factor");
    participantCount *= factor;
  }
  if (participantCount != placement.tiles.size())
    return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
        failureReason,
        "logical trial factor product does not match its Tile embedding");

  llvm::SmallVector<TileId, 16> uniqueTiles(placement.tiles.begin(),
                                            placement.tiles.end());
  llvm::sort(uniqueTiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  if (std::adjacent_find(uniqueTiles.begin(), uniqueTiles.end()) !=
      uniqueTiles.end())
    return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
        failureReason, "logical trial binds one Tile to several owner domains");

  for (auto [linearCoordinate, tile] : llvm::enumerate(placement.tiles)) {
    uint64_t remaining = linearCoordinate;
    llvm::SmallVector<uint32_t, 4> coordinates(factors.size(), 0);
    for (size_t reverse = 0; reverse < factors.size(); ++reverse) {
      const size_t dimension = factors.size() - reverse - 1;
      coordinates[dimension] = remaining % factors[dimension];
      remaining /= factors[dimension];
    }
    if (remaining != 0)
      return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
          failureReason, "logical trial coordinate exceeds its factor mesh");

    llvm::SmallVector<int64_t, 4> offsets(loopShape.size(), 0);
    llvm::SmallVector<int64_t, 4> sizes(loopShape.size(), 0);
    for (size_t dimension = 0; dimension < loopShape.size(); ++dimension) {
      const int64_t factor = factors[dimension];
      const int64_t coordinate = coordinates[dimension];
      const int64_t base = loopShape[dimension] / factor;
      const int64_t larger = loopShape[dimension] % factor;
      offsets[dimension] =
          coordinate * base + std::min<int64_t>(coordinate, larger);
      sizes[dimension] = base + (coordinate < larger);
    }
    IndexSetResult executionDomain =
        IndexRelation::staticRectangularDomain(offsets, sizes);
    if (!executionDomain.isExact())
      return fail<llvm::SmallVector<analysis::LogicalExecutionShard, 16>>(
          failureReason, executionDomain.reason);
    result.push_back(
        analysis::LogicalExecutionShard{tile, *executionDomain.set});
  }
  return result;
}

bool hasSpatialReductionPartition(mlir::linalg::LinalgOp linalg,
                                  const StructuredDAGNodePlacement &placement) {
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      linalg.getIteratorTypesArray();
  if (iteratorTypes.size() != placement.iteratorPartitionFactors.size())
    return false;
  return llvm::any_of(
      llvm::zip_equal(iteratorTypes, placement.iteratorPartitionFactors),
      [](auto iteratorAndFactor) {
        return std::get<0>(iteratorAndFactor) ==
                   mlir::utils::IteratorType::reduction &&
               std::get<1>(iteratorAndFactor) > 1;
      });
}

/// Exact per-result ownership induced by the already closed execution shards.
/// Disjoint images that cover the result are a unique partition; identical
/// full-result images are explicit replication. Any other overlap/hole needs a
/// richer typed role from the upstream placement assignment and fails closed.
mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
deriveResultOwnership(
    mlir::linalg::LinalgOp linalg, uint32_t resultIndex,
    llvm::ArrayRef<analysis::LogicalExecutionShard> executionShards,
    bool partialReduction, std::string *failureReason) {
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      linalg->getResult(resultIndex).getType());
  llvm::SmallVector<int64_t, 4> loopShape = linalg.getStaticLoopRanges();
  if (!resultType || !resultType.hasStaticShape())
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason, "logical trial requires static tensor results");
  mlir::AffineMap resultMap =
      linalg.getIndexingMapMatchingResult(linalg->getResult(resultIndex));
  if (!resultMap || resultMap.getNumDims() != loopShape.size() ||
      resultMap.getNumResults() != static_cast<unsigned>(resultType.getRank()))
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason, "logical trial result has an inconsistent indexing map");
  IndexRelationResult iterationToResult =
      IndexRelation::fromAffineMap(resultMap, loopShape, resultType.getShape());
  if (!iterationToResult.isExact())
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason, "logical trial result relation is not exact");
  IndexSetResult fullResult =
      IndexRelation::staticDomain(resultType.getShape());
  if (!fullResult.isExact())
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason, fullResult.reason);

  llvm::SmallVector<analysis::LogicalTileBinding, 16> bindings;
  bindings.reserve(executionShards.size());
  for (const analysis::LogicalExecutionShard &shard : executionShards) {
    if (!shard.executionDomain)
      return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
          failureReason, "logical trial execution shard has no domain");
    IndexSetResult owned;
    analysis::StaticRectangularIndexSetResult executionRectangle =
        IndexSetResult{IndexRelationStatus::Exact, *shard.executionDomain, {}}
            .getExactStaticRectangularDomain();
    if (executionRectangle.isExact()) {
      analysis::StaticRectangularIndexSetResult ownershipRectangle =
          iterationToResult.get()->getExactStaticRectangularImage(
              executionRectangle.domain->offsets,
              executionRectangle.domain->sizes);
      if (ownershipRectangle.isExact())
        owned = IndexRelation::staticRectangularDomain(
            ownershipRectangle.domain->offsets,
            ownershipRectangle.domain->sizes);
    }
    if (!owned.isExact())
      owned = iterationToResult.get()->image(*shard.executionDomain);
    if (!owned.isExact() || owned.set->isIntegerEmpty())
      return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
          failureReason, "logical trial result ownership is not exact");
    bindings.push_back(
        analysis::LogicalTileBinding{shard.tile, resultIndex, *owned.set,
                                     analysis::TileRole::UniquePartition});
  }

  bool everyOwnerHasFullResult = bindings.size() > 1;
  std::optional<StaticRectangle> fullRectangle =
      getStaticRectangle(*fullResult.set);
  llvm::SmallVector<StaticRectangle, 16> bindingRectangles;
  if (fullRectangle) {
    for (const analysis::LogicalTileBinding &binding : bindings) {
      std::optional<StaticRectangle> rectangle =
          getStaticRectangle(*binding.ownedDomain);
      if (!rectangle) {
        bindingRectangles.clear();
        break;
      }
      bindingRectangles.push_back(std::move(*rectangle));
    }
  }
  if (fullRectangle && bindingRectangles.size() == bindings.size()) {
    bool overlaps = false;
    for (const StaticRectangle &rectangle : bindingRectangles) {
      everyOwnerHasFullResult &= sameRectangle(rectangle, *fullRectangle);
      if (!rectangleContains(*fullRectangle, rectangle))
        return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
            failureReason,
            "logical trial result ownership exceeds the result domain");
    }
    for (size_t lhs = 0; lhs < bindingRectangles.size(); ++lhs)
      for (size_t rhs = lhs + 1; rhs < bindingRectangles.size(); ++rhs)
        overlaps |= static_cast<bool>(intersectRectangles(
            bindingRectangles[lhs], bindingRectangles[rhs]));
    RectangleCoverageResult coverage =
        proveRectangleCoverage(*fullRectangle, bindingRectangles);
    if (coverage.status == RectangleCoverageStatus::ResourceExhausted)
      return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
          failureReason,
          "logical trial rectangular ownership exceeds work budget");
    if (coverage.status == RectangleCoverageStatus::Uncovered)
      return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
          failureReason,
          "logical trial result ownership does not cover the result domain");
    if (partialReduction) {
      for (analysis::LogicalTileBinding &binding : bindings)
        binding.role = analysis::TileRole::PartialReductionContribution;
      return bindings;
    }
    if (overlaps && !everyOwnerHasFullResult)
      return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
          failureReason,
          "logical trial result ownership overlap needs an explicit role");
    if (everyOwnerHasFullResult)
      for (analysis::LogicalTileBinding &binding : bindings)
        binding.role = analysis::TileRole::ExplicitReplication;
    return bindings;
  }

  std::optional<mlir::presburger::PresburgerSet> covered;
  bool overlaps = false;
  for (const analysis::LogicalTileBinding &binding : bindings) {
    everyOwnerHasFullResult &=
        exactSetsEqual(*binding.ownedDomain, *fullResult.set);
    if (covered && !covered->intersect(*binding.ownedDomain).isIntegerEmpty())
      overlaps = true;
    covered = covered ? covered->unionSet(*binding.ownedDomain)
                      : *binding.ownedDomain;
  }
  if (!covered || !exactSetsEqual(*covered, *fullResult.set))
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason,
        "logical trial result ownership does not cover the result domain");
  if (partialReduction) {
    for (analysis::LogicalTileBinding &binding : bindings)
      binding.role = analysis::TileRole::PartialReductionContribution;
    return bindings;
  }
  if (overlaps && !everyOwnerHasFullResult)
    return fail<llvm::SmallVector<analysis::LogicalTileBinding, 16>>(
        failureReason,
        "logical trial result ownership overlap needs an explicit role");
  if (everyOwnerHasFullResult)
    for (analysis::LogicalTileBinding &binding : bindings)
      binding.role = analysis::TileRole::ExplicitReplication;
  return bindings;
}

mlir::FailureOr<analysis::LogicalShardTrial>
buildEdgeShardTrial(const StructuredDAGAnalysis &dag,
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
          failureReason, "dependent edge demand requires a static Linalg node");
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
    const bool partialReduction =
        hasSpatialReductionPartition(linalg, placement);
    if (partialReduction != placement.reductionMergeTile.has_value())
      return fail<analysis::LogicalShardTrial>(
          failureReason,
          "logical trial reduction partition and merge owner disagree");
    entry.reductionMergeTile = placement.reductionMergeTile;
    for (uint32_t resultIndex = 0;
         resultIndex < node->operation->getNumResults(); ++resultIndex) {
      mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
          ownership =
              deriveResultOwnership(linalg, resultIndex, entry.executionShards,
                                    partialReduction, failureReason);
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
    const bool partialReduction =
        hasSpatialReductionPartition(linalg, placement);
    if (partialReduction != placement.reductionMergeTile.has_value())
      return fail<analysis::LogicalShardTrial>(
          failureReason,
          "logical trial reduction partition and merge owner disagree");
    nodeTrial.reductionMergeTile = placement.reductionMergeTile;
    for (uint32_t resultIndex = 0; resultIndex < operation->getNumResults();
         ++resultIndex) {
      mlir::FailureOr<llvm::SmallVector<analysis::LogicalTileBinding, 16>>
          ownership = deriveResultOwnership(linalg, resultIndex,
                                            nodeTrial.executionShards,
                                            partialReduction, failureReason);
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
