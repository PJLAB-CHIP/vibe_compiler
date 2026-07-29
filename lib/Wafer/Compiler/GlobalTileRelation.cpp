//===- GlobalTileRelation.cpp - Rank/global static tile relation --------===//

#include "Wafer/Compiler/GlobalTileRelation.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace wafer::compiler {
namespace {

static llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "global_tile_relation: %s",
                                 detail.str().c_str());
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedSub(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    return false;
  result = lhs - rhs;
  return true;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static llvm::Error verifyRankDomain(const frontend::ProgramRankSlice &rankSlice,
                                    llvm::ArrayRef<int64_t> globalShape,
                                    llvm::ArrayRef<int64_t> localShape) {
  const size_t rank = globalShape.size();
  if (rank == 0 || localShape.size() != rank ||
      rankSlice.offsets.size() != rank || rankSlice.sizes.size() != rank ||
      rankSlice.strides.size() != rank)
    return invalid("rank slice and boundary shapes must have one equal, "
                   "non-zero rank");
  for (size_t dim = 0; dim < rank; ++dim) {
    if (globalShape[dim] <= 0 || localShape[dim] <= 0 ||
        rankSlice.offsets[dim] < 0 || rankSlice.sizes[dim] <= 0 ||
        rankSlice.strides[dim] <= 0 || rankSlice.sizes[dim] != localShape[dim])
      return invalid("rank slice has an invalid shape, offset, or stride");
    int64_t span = 0;
    int64_t last = 0;
    if (!checkedMul(rankSlice.sizes[dim] - 1, rankSlice.strides[dim], span) ||
        !checkedAdd(rankSlice.offsets[dim], span, last) ||
        last >= globalShape[dim])
      return invalid("rank slice exceeds the global logical shape");
  }
  return llvm::Error::success();
}

static llvm::Error verifyTile(const StaticTileRegion &tile, size_t rank) {
  if (tile.offsets.size() != rank || tile.sizes.size() != rank ||
      tile.strides.size() != rank)
    return invalid("tile rank does not match the boundary relation");
  for (size_t dim = 0; dim < rank; ++dim)
    if (tile.offsets[dim] < 0 || tile.sizes[dim] <= 0 || tile.strides[dim] <= 0)
      return invalid("tile has a negative offset or non-positive extent");
  return llvm::Error::success();
}

static std::optional<unsigned> getOperationOrdinal(mlir::Operation *operation) {
  mlir::Block *block = operation ? operation->getBlock() : nullptr;
  if (!block)
    return std::nullopt;
  unsigned ordinal = 0;
  for (mlir::Operation &candidate : *block) {
    if (&candidate == operation)
      return ordinal;
    // Number typed sibling occurrences, rather than every operation in the
    // block.  Earlier materializers may insert rank-specific peer transport
    // around an otherwise identical subview/loop path.  Such unrelated
    // operations must not change the identity of the existing typed
    // occurrence, while two sibling operations of the same kind must remain
    // distinguishable.
    if (candidate.getName() == operation->getName())
      ++ordinal;
  }
  return std::nullopt;
}

static std::optional<unsigned> getBlockOrdinal(mlir::Block *block) {
  mlir::Region *region = block ? block->getParent() : nullptr;
  if (!region)
    return std::nullopt;
  unsigned ordinal = 0;
  for (mlir::Block &candidate : *region) {
    if (&candidate == block)
      return ordinal;
    ++ordinal;
  }
  return std::nullopt;
}

static std::optional<unsigned> getRegionOrdinal(mlir::Region *region) {
  mlir::Operation *parent = region ? region->getParentOp() : nullptr;
  if (!parent)
    return std::nullopt;
  for (auto [ordinal, candidate] : llvm::enumerate(parent->getRegions()))
    if (&candidate == region)
      return static_cast<unsigned>(ordinal);
  return std::nullopt;
}

static bool isPublicEntryFunction(mlir::func::FuncOp function) {
  return function && !function.isPrivate() && !function.isDeclaration() &&
         static_cast<bool>(function->getParentOfType<mlir::ModuleOp>());
}

/// Compare one typed occurrence path from an operation to its function root.
/// Symbol names are intentionally absent, while operation/region/block
/// positions distinguish sibling or differently nested structured instances.
static bool equivalentStructuredPath(mlir::Operation *lhs,
                                     mlir::Operation *rhs) {
  while (lhs && rhs) {
    const bool lhsFunction = mlir::isa<mlir::func::FuncOp>(lhs);
    const bool rhsFunction = mlir::isa<mlir::func::FuncOp>(rhs);
    if (lhsFunction || rhsFunction) {
      if (!lhsFunction || !rhsFunction)
        return false;
      return isPublicEntryFunction(mlir::cast<mlir::func::FuncOp>(lhs)) &&
             isPublicEntryFunction(mlir::cast<mlir::func::FuncOp>(rhs));
    }
    if (lhs->getName() != rhs->getName())
      return false;
    mlir::Block *lhsBlock = lhs->getBlock();
    mlir::Block *rhsBlock = rhs->getBlock();
    std::optional<unsigned> lhsOperation = getOperationOrdinal(lhs);
    std::optional<unsigned> rhsOperation = getOperationOrdinal(rhs);
    std::optional<unsigned> lhsBlockPosition = getBlockOrdinal(lhsBlock);
    std::optional<unsigned> rhsBlockPosition = getBlockOrdinal(rhsBlock);
    std::optional<unsigned> lhsRegion =
        getRegionOrdinal(lhsBlock ? lhsBlock->getParent() : nullptr);
    std::optional<unsigned> rhsRegion =
        getRegionOrdinal(rhsBlock ? rhsBlock->getParent() : nullptr);
    if (!lhsOperation || !rhsOperation || !lhsBlockPosition ||
        !rhsBlockPosition || !lhsRegion || !rhsRegion ||
        *lhsOperation != *rhsOperation ||
        *lhsBlockPosition != *rhsBlockPosition || *lhsRegion != *rhsRegion)
      return false;
    lhs = lhsBlock->getParentOp();
    rhs = rhsBlock->getParentOp();
  }
  return !lhs && !rhs;
}

static std::optional<llvm::hash_code>
hashStructuredPath(mlir::Operation *operation) {
  llvm::hash_code hash = llvm::hash_combine(0U);
  while (operation) {
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation))
      return isPublicEntryFunction(function)
                 ? std::optional<llvm::hash_code>(hash)
                 : std::nullopt;
    mlir::Block *block = operation->getBlock();
    std::optional<unsigned> operationOrdinal = getOperationOrdinal(operation);
    std::optional<unsigned> blockOrdinal = getBlockOrdinal(block);
    std::optional<unsigned> regionOrdinal =
        getRegionOrdinal(block ? block->getParent() : nullptr);
    if (!operationOrdinal || !blockOrdinal || !regionOrdinal)
      return std::nullopt;
    hash = llvm::hash_combine(hash, operation->getName().getStringRef(),
                              *operationOrdinal, *blockOrdinal, *regionOrdinal);
    operation = block->getParentOp();
  }
  return std::nullopt;
}

static llvm::Expected<int64_t> getLastCoordinate(int64_t offset, int64_t size,
                                                 int64_t stride) {
  int64_t span = 0;
  int64_t last = 0;
  if (!checkedMul(size - 1, stride, span) || !checkedAdd(offset, span, last))
    return invalid("tile coordinate arithmetic overflows int64");
  return last;
}

static mlir::Value resolveTileRegionBoundaryAlias(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = argument.getOwner();
      auto region =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp();
      if (!region || region.getBody().empty() ||
          owner != &region.getBody().front() ||
          argument.getArgNumber() >= region.getInputs().size())
        break;
      value = region.getInputs()[argument.getArgNumber()];
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    auto region = result ? mlir::dyn_cast<TileRegionOp>(result.getOwner())
                         : TileRegionOp();
    if (!region || region.getBody().empty() ||
        result.getResultNumber() >=
            region.getBody().front().getTerminator()->getNumOperands())
      break;
    value = region.getBody().front().getTerminator()->getOperand(
        result.getResultNumber());
  }
  return value;
}

static bool isTransparentCast(mlir::memref::CastOp cast) {
  auto source = mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
  auto result = mlir::dyn_cast<mlir::MemRefType>(cast.getType());
  return source && result && source.getShape() == result.getShape() &&
         source.getElementType() == result.getElementType() &&
         source.getMemorySpace() == result.getMemorySpace();
}

static std::optional<std::pair<int64_t, int64_t>>
getStructuredIndexBounds(mlir::OpFoldResult foldResult) {
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(foldResult))
    return std::pair<int64_t, int64_t>{*constant, *constant};
  mlir::Value value = foldResult.dyn_cast<mlir::Value>();
  if (!value || !value.getType().isIndex())
    return std::nullopt;

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    auto loop =
        owner ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(owner->getParentOp())
              : mlir::scf::ForOp();
    if (loop && value == loop.getInductionVar()) {
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(loop.getUpperBound());
      std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *lower >= *upper)
        return std::nullopt;
      int64_t closedUpper = 0;
      int64_t distance = 0;
      if (!checkedSub(*upper, 1, closedUpper) ||
          !checkedSub(closedUpper, *lower, distance))
        return std::nullopt;
      int64_t tripOffset = distance / *step;
      int64_t scaled = 0;
      int64_t maximum = 0;
      if (!checkedMul(tripOffset, *step, scaled) ||
          !checkedAdd(*lower, scaled, maximum))
        return std::nullopt;
      return std::pair<int64_t, int64_t>{*lower, maximum};
    }
  }

  using mlir::ValueBoundsConstraintSet;
  using mlir::presburger::BoundType;
  ValueBoundsConstraintSet::Variable variable(value);
  mlir::FailureOr<int64_t> lower =
      ValueBoundsConstraintSet::computeConstantBound(BoundType::LB, variable,
                                                     nullptr,
                                                     /*closedUB=*/true);
  mlir::FailureOr<int64_t> upper =
      ValueBoundsConstraintSet::computeConstantBound(BoundType::UB, variable,
                                                     nullptr,
                                                     /*closedUB=*/true);
  if (mlir::failed(lower) || mlir::failed(upper) || *lower > *upper)
    return std::nullopt;
  return std::pair<int64_t, int64_t>{*lower, *upper};
}

static bool validateViewStep(mlir::memref::SubViewOp subview) {
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
  auto resultType =
      mlir::dyn_cast<mlir::MemRefType>(subview.getResult().getType());
  if (!sourceType || !resultType ||
      sourceType.getRank() != resultType.getRank() ||
      !sourceType.hasStaticShape() ||
      subview.getMixedOffsets().size() !=
          static_cast<size_t>(sourceType.getRank()) ||
      subview.getMixedSizes().size() !=
          static_cast<size_t>(sourceType.getRank()) ||
      subview.getMixedStrides().size() !=
          static_cast<size_t>(sourceType.getRank()))
    return false;

  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    std::optional<int64_t> size =
        mlir::getConstantIntValue(subview.getMixedSizes()[dim]);
    std::optional<int64_t> stride =
        mlir::getConstantIntValue(subview.getMixedStrides()[dim]);
    std::optional<std::pair<int64_t, int64_t>> offsetBounds =
        getStructuredIndexBounds(subview.getMixedOffsets()[dim]);
    if (!size || !stride || !offsetBounds || *size <= 0 || *stride <= 0 ||
        offsetBounds->first < 0)
      return false;
    int64_t span = 0;
    int64_t last = 0;
    if (!checkedMul(*size - 1, *stride, span) ||
        !checkedAdd(offsetBounds->second, span, last) ||
        last >= sourceType.getDimSize(dim))
      return false;
  }
  return true;
}

class StructuredIndexEquivalence {
public:
  bool equivalent(mlir::OpFoldResult lhs, mlir::OpFoldResult rhs) {
    std::optional<int64_t> lhsConstant = mlir::getConstantIntValue(lhs);
    std::optional<int64_t> rhsConstant = mlir::getConstantIntValue(rhs);
    if (lhsConstant || rhsConstant)
      return lhsConstant && rhsConstant && *lhsConstant == *rhsConstant;
    return equivalent(lhs.dyn_cast<mlir::Value>(), rhs.dyn_cast<mlir::Value>());
  }

private:
  bool equivalent(mlir::Value lhs, mlir::Value rhs) {
    if (!lhs || !rhs || lhs.getType() != rhs.getType())
      return false;
    if (lhs == rhs)
      return true;
    std::pair<mlir::Value, mlir::Value> key{lhs, rhs};
    if (!active.insert(key).second)
      return false;
    auto finish = [&](bool result) {
      active.erase(key);
      return result;
    };

    auto lhsArgument = mlir::dyn_cast<mlir::BlockArgument>(lhs);
    auto rhsArgument = mlir::dyn_cast<mlir::BlockArgument>(rhs);
    if (lhsArgument || rhsArgument) {
      if (!lhsArgument || !rhsArgument)
        return finish(false);
      auto lhsLoop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          lhsArgument.getOwner()->getParentOp());
      auto rhsLoop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          rhsArgument.getOwner()->getParentOp());
      if (!lhsLoop || !rhsLoop || lhs != lhsLoop.getInductionVar() ||
          rhs != rhsLoop.getInductionVar() ||
          !equivalentStructuredPath(lhsLoop.getOperation(),
                                    rhsLoop.getOperation()))
        return finish(false);
      return finish(
          equivalent(lhsLoop.getLowerBound(), rhsLoop.getLowerBound()) &&
          equivalent(lhsLoop.getUpperBound(), rhsLoop.getUpperBound()) &&
          equivalent(lhsLoop.getStep(), rhsLoop.getStep()));
    }

    auto lhsResult = mlir::dyn_cast<mlir::OpResult>(lhs);
    auto rhsResult = mlir::dyn_cast<mlir::OpResult>(rhs);
    if (!lhsResult || !rhsResult ||
        lhsResult.getResultNumber() != rhsResult.getResultNumber())
      return finish(false);
    mlir::Operation *lhsDefinition = lhsResult.getOwner();
    mlir::Operation *rhsDefinition = rhsResult.getOwner();
    if (!mlir::isMemoryEffectFree(lhsDefinition) ||
        !mlir::isMemoryEffectFree(rhsDefinition))
      return finish(false);
    return finish(mlir::OperationEquivalence::isEquivalentTo(
        lhsDefinition, rhsDefinition,
        [&](mlir::Value lhsOperand,
            mlir::Value rhsOperand) -> mlir::LogicalResult {
          return mlir::success(equivalent(lhsOperand, rhsOperand));
        },
        /*markEquivalent=*/nullptr,
        mlir::OperationEquivalence::IgnoreLocations));
  }

  llvm::DenseSet<std::pair<mlir::Value, mlir::Value>> active;
};

class StructuredIndexHash {
public:
  std::optional<llvm::hash_code> compute(mlir::OpFoldResult value) {
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
      return llvm::hash_combine(0U, *constant);
    return compute(value.dyn_cast<mlir::Value>());
  }

private:
  std::optional<llvm::hash_code> compute(mlir::Value value) {
    if (!value || !active.insert(value).second)
      return std::nullopt;
    auto finish = [&](std::optional<llvm::hash_code> result) {
      active.erase(value);
      return result;
    };

    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (!loop || value != loop.getInductionVar())
        return finish(std::nullopt);
      std::optional<llvm::hash_code> path =
          hashStructuredPath(loop.getOperation());
      if (!path)
        return finish(std::nullopt);
      std::optional<llvm::hash_code> lower = compute(loop.getLowerBound());
      std::optional<llvm::hash_code> upper = compute(loop.getUpperBound());
      std::optional<llvm::hash_code> step = compute(loop.getStep());
      if (!lower || !upper || !step)
        return finish(std::nullopt);
      return finish(llvm::hash_combine(1U, *path, *lower, *upper, *step));
    }

    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *definition = result ? result.getOwner() : nullptr;
    if (!definition || !mlir::isMemoryEffectFree(definition))
      return finish(std::nullopt);
    bool valid = true;
    llvm::hash_code hash = mlir::OperationEquivalence::computeHash(
        definition,
        [&](mlir::Value operand) {
          std::optional<llvm::hash_code> operandHash = compute(operand);
          if (!operandHash) {
            valid = false;
            return llvm::hash_code();
          }
          return *operandHash;
        },
        mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::IgnoreLocations);
    return finish(valid ? std::optional<llvm::hash_code>(hash) : std::nullopt);
  }

  llvm::DenseSet<mlir::Value> active;
};

static bool equivalentFoldResults(llvm::ArrayRef<mlir::OpFoldResult> lhs,
                                  llvm::ArrayRef<mlir::OpFoldResult> rhs,
                                  StructuredIndexEquivalence &equivalence) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [&](const auto &pair) {
           return equivalence.equivalent(std::get<0>(pair), std::get<1>(pair));
         });
}

static llvm::hash_code hashStaticTile(const StaticTileRegion &tile) {
  llvm::hash_code hash = llvm::hash_combine(0U);
  for (int64_t offset : tile.offsets)
    hash = llvm::hash_combine(hash, offset);
  for (int64_t size : tile.sizes)
    hash = llvm::hash_combine(hash, size);
  for (int64_t stride : tile.strides)
    hash = llvm::hash_combine(hash, stride);
  return hash;
}

/// Rebuild the exact logical tile-to-global relation from the current static
/// tile.  The Presburger-backed IndexRelation is deliberately invocation
/// local: it proves that two independently discovered view paths denote the
/// same logical points without persisting another relation beside the IR.
static std::optional<bool>
haveEquivalentExactIndexRelations(const StaticTileRegion &lhs,
                                  const StaticTileRegion &rhs,
                                  llvm::ArrayRef<int64_t> globalShape) {
  analysis::IndexRelationResult lhsRelation =
      analysis::IndexRelation::staticSlice(lhs.sizes, globalShape, lhs.offsets,
                                           lhs.strides);
  analysis::IndexRelationResult rhsRelation =
      analysis::IndexRelation::staticSlice(rhs.sizes, globalShape, rhs.offsets,
                                           rhs.strides);
  if (!lhsRelation.isExact() || !rhsRelation.isExact())
    return std::nullopt;
  analysis::IndexRelationQueryResult equivalent =
      lhsRelation.get()->isEquivalentTo(*rhsRelation.get());
  if (equivalent.status != analysis::IndexRelationStatus::Exact ||
      !equivalent.value)
    return std::nullopt;
  return *equivalent.value;
}

} // namespace

llvm::Expected<StaticTileRegion>
mapRankLocalTileToGlobal(const frontend::ProgramRankSlice &rankSlice,
                         llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<int64_t> localShape,
                         const StaticTileRegion &localTile) {
  if (llvm::Error error = verifyRankDomain(rankSlice, globalShape, localShape))
    return std::move(error);
  if (llvm::Error error = verifyTile(localTile, globalShape.size()))
    return std::move(error);

  StaticTileRegion global;
  global.offsets.resize(globalShape.size());
  global.sizes = localTile.sizes;
  global.strides.resize(globalShape.size());
  for (size_t dim = 0; dim < globalShape.size(); ++dim) {
    llvm::Expected<int64_t> localLast = getLastCoordinate(
        localTile.offsets[dim], localTile.sizes[dim], localTile.strides[dim]);
    if (!localLast)
      return localLast.takeError();
    if (*localLast >= localShape[dim])
      return invalid("rank-local tile exceeds the local boundary shape");

    int64_t scaledOffset = 0;
    if (!checkedMul(localTile.offsets[dim], rankSlice.strides[dim],
                    scaledOffset) ||
        !checkedAdd(rankSlice.offsets[dim], scaledOffset,
                    global.offsets[dim]) ||
        !checkedMul(localTile.strides[dim], rankSlice.strides[dim],
                    global.strides[dim]))
      return invalid("rank-to-global tile composition overflows int64");
    llvm::Expected<int64_t> globalLast = getLastCoordinate(
        global.offsets[dim], global.sizes[dim], global.strides[dim]);
    if (!globalLast)
      return globalLast.takeError();
    if (*globalLast >= globalShape[dim])
      return invalid("composed tile exceeds the global boundary shape");
  }
  return global;
}

llvm::Expected<StaticTileRegion>
mapGlobalTileToRankLocal(const frontend::ProgramRankSlice &rankSlice,
                         llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<int64_t> localShape,
                         const StaticTileRegion &globalTile) {
  if (llvm::Error error = verifyRankDomain(rankSlice, globalShape, localShape))
    return std::move(error);
  if (llvm::Error error = verifyTile(globalTile, globalShape.size()))
    return std::move(error);

  StaticTileRegion local;
  local.offsets.resize(globalShape.size());
  local.sizes = globalTile.sizes;
  local.strides.resize(globalShape.size());
  for (size_t dim = 0; dim < globalShape.size(); ++dim) {
    llvm::Expected<int64_t> globalLast =
        getLastCoordinate(globalTile.offsets[dim], globalTile.sizes[dim],
                          globalTile.strides[dim]);
    if (!globalLast)
      return globalLast.takeError();
    if (*globalLast >= globalShape[dim] ||
        globalTile.offsets[dim] < rankSlice.offsets[dim])
      return invalid("global tile is outside the boundary or rank slice");

    int64_t relative = globalTile.offsets[dim] - rankSlice.offsets[dim];
    if (relative % rankSlice.strides[dim] != 0 ||
        globalTile.strides[dim] % rankSlice.strides[dim] != 0)
      return invalid("global tile is not integral in rank-local coordinates");
    local.offsets[dim] = relative / rankSlice.strides[dim];
    local.strides[dim] = globalTile.strides[dim] / rankSlice.strides[dim];
    llvm::Expected<int64_t> localLast = getLastCoordinate(
        local.offsets[dim], local.sizes[dim], local.strides[dim]);
    if (!localLast)
      return localLast.takeError();
    if (*localLast >= localShape[dim])
      return invalid("global tile is not fully covered by the rank slice");
  }

  llvm::Expected<StaticTileRegion> roundTrip =
      mapRankLocalTileToGlobal(rankSlice, globalShape, localShape, local);
  if (!roundTrip)
    return roundTrip.takeError();
  if (*roundTrip != globalTile)
    return invalid("rank-local projection does not exactly round-trip");
  return local;
}

StaticTileRelation compareStaticTiles(const StaticTileRegion &lhs,
                                      const StaticTileRegion &rhs) {
  if (lhs == rhs)
    return StaticTileRelation::Equivalent;
  if (lhs.offsets.size() != rhs.offsets.size() ||
      lhs.sizes.size() != lhs.offsets.size() ||
      rhs.sizes.size() != rhs.offsets.size() ||
      lhs.strides.size() != lhs.offsets.size() ||
      rhs.strides.size() != rhs.offsets.size())
    return StaticTileRelation::OverlappingOrUnknown;

  for (size_t dim = 0; dim < lhs.offsets.size(); ++dim) {
    llvm::Expected<int64_t> leftLast =
        getLastCoordinate(lhs.offsets[dim], lhs.sizes[dim], lhs.strides[dim]);
    llvm::Expected<int64_t> rightLast =
        getLastCoordinate(rhs.offsets[dim], rhs.sizes[dim], rhs.strides[dim]);
    if (!leftLast || !rightLast) {
      if (!leftLast)
        llvm::consumeError(leftLast.takeError());
      if (!rightLast)
        llvm::consumeError(rightLast.takeError());
      return StaticTileRelation::OverlappingOrUnknown;
    }
    if (*leftLast < rhs.offsets[dim] || *rightLast < lhs.offsets[dim])
      return StaticTileRelation::Disjoint;
  }
  return StaticTileRelation::OverlappingOrUnknown;
}

std::optional<ResolvedBoundaryTileView>
resolveBoundaryTileView(mlir::Value value, llvm::ArrayRef<int64_t> localShape) {
  if (!value || localShape.empty() ||
      llvm::any_of(localShape, [](int64_t extent) { return extent <= 0; }))
    return std::nullopt;

  mlir::Value original = value;
  value = resolveTileRegionBoundaryAlias(value);
  llvm::SmallVector<SymbolicTileViewStep, 4> reversedSteps;
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(definition)) {
      if (!isTransparentCast(cast))
        return std::nullopt;
      value = resolveTileRegionBoundaryAlias(cast.getSource());
      continue;
    }
    auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(definition);
    if (!subview || !validateViewStep(subview))
      return std::nullopt;
    reversedSteps.push_back({subview.getOperation(),
                             llvm::to_vector(subview.getMixedOffsets()),
                             llvm::to_vector(subview.getMixedSizes()),
                             llvm::to_vector(subview.getMixedStrides())});
    value = resolveTileRegionBoundaryAlias(subview.getSource());
  }

  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || !argument.getOwner() ||
      !mlir::isa_and_nonnull<mlir::func::FuncOp>(
          argument.getOwner()->getParentOp()))
    return std::nullopt;
  auto function =
      mlir::cast<mlir::func::FuncOp>(argument.getOwner()->getParentOp());
  // A private/helper argument is not a program boundary merely because it has
  // the same ordinal. Until call-site operands are represented in the
  // symbolic view path, fail closed instead of conflating distinct calls.
  if (!isPublicEntryFunction(function))
    return std::nullopt;
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
  auto leafType = mlir::dyn_cast<mlir::MemRefType>(original.getType());
  if (!rootType || !leafType || rootType.getShape() != localShape ||
      rootType.getRank() != leafType.getRank())
    return std::nullopt;

  ResolvedBoundaryTileView result;
  result.argumentIndex = argument.getArgNumber();
  result.leafType = leafType;
  result.viewSteps.assign(reversedSteps.rbegin(), reversedSteps.rend());

  StaticTileRegion tile;
  tile.offsets.assign(localShape.size(), 0);
  tile.sizes.assign(localShape.begin(), localShape.end());
  tile.strides.assign(localShape.size(), 1);
  bool fullyStatic = true;
  for (const SymbolicTileViewStep &step : result.viewSteps) {
    if (step.offsets.size() != tile.offsets.size() ||
        step.sizes.size() != tile.offsets.size() ||
        step.strides.size() != tile.offsets.size()) {
      fullyStatic = false;
      break;
    }
    for (size_t dim = 0; dim < tile.offsets.size(); ++dim) {
      std::optional<int64_t> offset =
          mlir::getConstantIntValue(step.offsets[dim]);
      std::optional<int64_t> size = mlir::getConstantIntValue(step.sizes[dim]);
      std::optional<int64_t> stride =
          mlir::getConstantIntValue(step.strides[dim]);
      if (!offset || !size || !stride) {
        fullyStatic = false;
        break;
      }
      int64_t scaledOffset = 0;
      int64_t composedOffset = 0;
      int64_t composedStride = 0;
      if (!checkedMul(*offset, tile.strides[dim], scaledOffset) ||
          !checkedAdd(tile.offsets[dim], scaledOffset, composedOffset) ||
          !checkedMul(tile.strides[dim], *stride, composedStride)) {
        fullyStatic = false;
        break;
      }
      tile.offsets[dim] = composedOffset;
      tile.sizes[dim] = *size;
      tile.strides[dim] = composedStride;
    }
    if (!fullyStatic)
      break;
  }
  if (fullyStatic)
    result.staticLocalTile = std::move(tile);
  return result;
}

StaticTileRelation
compareRankBoundaryTileViews(const ResolvedBoundaryTileView &lhs,
                             const frontend::ProgramRankSlice &lhsSlice,
                             const ResolvedBoundaryTileView &rhs,
                             const frontend::ProgramRankSlice &rhsSlice,
                             llvm::ArrayRef<int64_t> globalShape,
                             llvm::ArrayRef<int64_t> localShape) {
  if (lhs.argumentIndex != rhs.argumentIndex || lhs.leafType != rhs.leafType ||
      lhs.viewSteps.size() != rhs.viewSteps.size())
    return StaticTileRelation::OverlappingOrUnknown;
  for (auto [lhsStep, rhsStep] : llvm::zip(lhs.viewSteps, rhs.viewSteps))
    if (!equivalentStructuredPath(lhsStep.viewOperation, rhsStep.viewOperation))
      return StaticTileRelation::OverlappingOrUnknown;

  StaticTileRegion fullLocal;
  fullLocal.offsets.assign(localShape.size(), 0);
  fullLocal.sizes.assign(localShape.begin(), localShape.end());
  fullLocal.strides.assign(localShape.size(), 1);
  llvm::Expected<StaticTileRegion> lhsDomain =
      mapRankLocalTileToGlobal(lhsSlice, globalShape, localShape, fullLocal);
  llvm::Expected<StaticTileRegion> rhsDomain =
      mapRankLocalTileToGlobal(rhsSlice, globalShape, localShape, fullLocal);
  if (!lhsDomain || !rhsDomain) {
    if (!lhsDomain)
      llvm::consumeError(lhsDomain.takeError());
    if (!rhsDomain)
      llvm::consumeError(rhsDomain.takeError());
    return StaticTileRelation::OverlappingOrUnknown;
  }

  if (lhs.staticLocalTile && rhs.staticLocalTile) {
    llvm::Expected<StaticTileRegion> lhsGlobal = mapRankLocalTileToGlobal(
        lhsSlice, globalShape, localShape, *lhs.staticLocalTile);
    llvm::Expected<StaticTileRegion> rhsGlobal = mapRankLocalTileToGlobal(
        rhsSlice, globalShape, localShape, *rhs.staticLocalTile);
    if (!lhsGlobal || !rhsGlobal) {
      if (!lhsGlobal)
        llvm::consumeError(lhsGlobal.takeError());
      if (!rhsGlobal)
        llvm::consumeError(rhsGlobal.takeError());
      return StaticTileRelation::OverlappingOrUnknown;
    }
    std::optional<bool> equivalent =
        haveEquivalentExactIndexRelations(*lhsGlobal, *rhsGlobal, globalShape);
    if (equivalent && *equivalent)
      return StaticTileRelation::Equivalent;
    return compareStaticTiles(*lhsGlobal, *rhsGlobal);
  }

  // Replica IDs and logical-rank IDs intentionally do not participate:
  // verified replicated slices have distinct IDs but identical geometry.
  if (lhsSlice.offsets != rhsSlice.offsets ||
      lhsSlice.sizes != rhsSlice.sizes || lhsSlice.strides != rhsSlice.strides)
    return StaticTileRelation::OverlappingOrUnknown;

  StructuredIndexEquivalence equivalence;
  for (auto [lhsStep, rhsStep] : llvm::zip(lhs.viewSteps, rhs.viewSteps))
    if (!equivalentFoldResults(lhsStep.offsets, rhsStep.offsets, equivalence) ||
        !equivalentFoldResults(lhsStep.sizes, rhsStep.sizes, equivalence) ||
        !equivalentFoldResults(lhsStep.strides, rhsStep.strides, equivalence))
      return StaticTileRelation::OverlappingOrUnknown;
  return StaticTileRelation::Equivalent;
}

std::optional<size_t>
hashRankBoundaryTileView(const ResolvedBoundaryTileView &view,
                         const frontend::ProgramRankSlice &rankSlice,
                         llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<int64_t> localShape) {
  if (view.staticLocalTile) {
    llvm::Expected<StaticTileRegion> global = mapRankLocalTileToGlobal(
        rankSlice, globalShape, localShape, *view.staticLocalTile);
    if (!global) {
      llvm::consumeError(global.takeError());
      return std::nullopt;
    }
    llvm::hash_code hash =
        llvm::hash_combine(view.argumentIndex, hashStaticTile(*global));
    for (const SymbolicTileViewStep &step : view.viewSteps) {
      std::optional<llvm::hash_code> path =
          hashStructuredPath(step.viewOperation);
      if (!path)
        return std::nullopt;
      hash = llvm::hash_combine(hash, *path);
    }
    return static_cast<size_t>(hash);
  }

  StaticTileRegion full;
  full.offsets.assign(localShape.size(), 0);
  full.sizes.assign(localShape.begin(), localShape.end());
  full.strides.assign(localShape.size(), 1);
  llvm::Expected<StaticTileRegion> verifiedDomain =
      mapRankLocalTileToGlobal(rankSlice, globalShape, localShape, full);
  if (!verifiedDomain) {
    llvm::consumeError(verifiedDomain.takeError());
    return std::nullopt;
  }

  llvm::hash_code hash = llvm::hash_combine(view.argumentIndex);
  for (int64_t offset : rankSlice.offsets)
    hash = llvm::hash_combine(hash, offset);
  for (int64_t size : rankSlice.sizes)
    hash = llvm::hash_combine(hash, size);
  for (int64_t stride : rankSlice.strides)
    hash = llvm::hash_combine(hash, stride);
  StructuredIndexHash indexHash;
  for (const SymbolicTileViewStep &step : view.viewSteps) {
    std::optional<llvm::hash_code> path =
        hashStructuredPath(step.viewOperation);
    if (!path)
      return std::nullopt;
    hash = llvm::hash_combine(hash, *path);
    for (llvm::ArrayRef<mlir::OpFoldResult> values :
         {llvm::ArrayRef<mlir::OpFoldResult>(step.offsets),
          llvm::ArrayRef<mlir::OpFoldResult>(step.sizes),
          llvm::ArrayRef<mlir::OpFoldResult>(step.strides)}) {
      for (mlir::OpFoldResult value : values) {
        std::optional<llvm::hash_code> valueHash = indexHash.compute(value);
        if (!valueHash)
          return std::nullopt;
        hash = llvm::hash_combine(hash, *valueHash);
      }
    }
  }
  return static_cast<size_t>(hash);
}

bool haveEquivalentStructuredOperationPaths(mlir::Operation *lhs,
                                            mlir::Operation *rhs) {
  return equivalentStructuredPath(lhs, rhs);
}

std::optional<size_t> hashStructuredOperationPath(mlir::Operation *operation) {
  std::optional<llvm::hash_code> hash = hashStructuredPath(operation);
  return hash ? std::optional<size_t>(static_cast<size_t>(*hash))
              : std::nullopt;
}

} // namespace wafer::compiler
