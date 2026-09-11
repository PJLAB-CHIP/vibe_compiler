//===- LayoutOptimization.cpp - Current layout/bufferization -----------===//

#include "Wafer/Transforms/Tile/LayoutOptimization.h"

#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/Analysis/Tile/PhysicalLayoutRelation.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr size_t kMaximumLayoutTupleStates = 4096;

struct ValueUnion {
  std::vector<unsigned> parent;

  unsigned add() {
    unsigned index = parent.size();
    parent.push_back(index);
    return index;
  }

  unsigned find(unsigned value) {
    if (parent[value] != value)
      parent[value] = find(parent[value]);
    return parent[value];
  }

  void unite(unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (lhs > rhs)
      std::swap(lhs, rhs);
    parent[rhs] = lhs;
  }
};

struct ValueGroup {
  llvm::SmallVector<mlir::Value, 4> values;
  llvm::SmallVector<MemLayout, 4> layouts;
  uint32_t variable = 0;
};

struct UseBinding {
  mlir::Operation *owner = nullptr;
  unsigned operandNumber = 0;
  mlir::Value source;
  unsigned sourceGroup = 0;
  llvm::SmallVector<MemLayout, 4> layouts;
  uint32_t variable = 0;
};

struct ResultBinding {
  mlir::OpResult result;
  unsigned publishedGroup = 0;
  MemLayout computeLayout = MemLayout::Tensor;
};

struct TupleVariable {
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<uint32_t, 4> coordinates;
  llvm::SmallVector<llvm::SmallVector<MemLayout, 4>, 4> coordinateLayouts;
  llvm::SmallVector<llvm::SmallVector<uint32_t, 4>, 4> states;
  uint32_t variable = 0;
};

struct UseCohort {
  mlir::Value source;
  mlir::Block *block = nullptr;
  unsigned sourceGroup = 0;
  llvm::SmallVector<unsigned, 4> uses;
  mlir::Operation *lastOwner = nullptr;
};

enum class ActivationState : uint32_t {
  Inactive = 0,
  SourceIsTarget = 1,
  Materialized = 2,
};

struct ConversionActivation {
  unsigned cohort = 0;
  MemLayout layout = MemLayout::Tensor;
  uint32_t variable = 0;
};

static ExactPBQPCost addCost(ExactPBQPCost lhs, ExactPBQPCost rhs) {
  if (lhs == kExactPBQPInfinity || rhs == kExactPBQPInfinity ||
      rhs > kExactPBQPInfinity - lhs)
    return kExactPBQPInfinity;
  return lhs + rhs;
}

class FactorBuilder {
public:
  explicit FactorBuilder(const ExactPBQPProblem &problem) : problem(problem) {}

  template <typename CostFn> void add(uint32_t lhs, uint32_t rhs, CostFn cost) {
    if (lhs == rhs)
      return;
    bool transpose = lhs > rhs;
    if (transpose)
      std::swap(lhs, rhs);
    const uint32_t lhsStates = problem.variables[lhs].unaryCosts.size();
    const uint32_t rhsStates = problem.variables[rhs].unaryCosts.size();
    auto [iterator, inserted] = factors.try_emplace(
        std::make_pair(lhs, rhs),
        ExactPBQPBinaryFactor{
            lhs, rhs, lhsStates, rhsStates,
            std::vector<ExactPBQPCost>(
                static_cast<size_t>(lhsStates) * rhsStates, 0)});
    ExactPBQPBinaryFactor &factor = iterator->second;
    for (uint32_t lhsState = 0; lhsState < lhsStates; ++lhsState)
      for (uint32_t rhsState = 0; rhsState < rhsStates; ++rhsState) {
        ExactPBQPCost next =
            !transpose ? cost(lhsState, rhsState) : cost(rhsState, lhsState);
        ExactPBQPCost &entry =
            factor.costs[static_cast<size_t>(lhsState) * rhsStates + rhsState];
        entry = addCost(entry, next);
      }
  }

  void appendTo(ExactPBQPProblem &target) {
    for (auto &entry : factors)
      target.factors.push_back(std::move(entry.second));
  }

private:
  const ExactPBQPProblem &problem;
  std::map<std::pair<uint32_t, uint32_t>, ExactPBQPBinaryFactor> factors;
};

static bool isTensorValue(mlir::Value value) {
  return value && mlir::isa<mlir::RankedTensorType>(value.getType());
}

static bool isFunctionEntryArgument(mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  auto function = argument ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(
                                 argument.getOwner()->getParentOp())
                           : mlir::func::FuncOp{};
  return function && !function.isExternal() &&
         argument.getOwner() == &function.getBody().front();
}

static bool isTileLocalTensor(mlir::Value value) {
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
    mlir::Operation *owner = result.getOwner();
    return mlir::isa<TileRegionOp>(owner) ||
           static_cast<bool>(owner->getParentOfType<TileRegionOp>());
  }
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  mlir::Operation *parent = argument && argument.getOwner()
                                ? argument.getOwner()->getParentOp()
                                : nullptr;
  return mlir::isa_and_nonnull<TileRegionOp>(parent) ||
         (parent && static_cast<bool>(parent->getParentOfType<TileRegionOp>()));
}

static bool isTensorViewOperation(mlir::Operation *operation) {
  return mlir::isa<mlir::tensor::CastOp, mlir::tensor::CollapseShapeOp,
                   mlir::tensor::ExpandShapeOp, mlir::tensor::ExtractSliceOp,
                   mlir::tensor::InsertSliceOp, mlir::tensor::ExtractOp,
                   mlir::tensor::InsertOp>(operation);
}

static llvm::SmallVector<MemLayout, 4>
getLayoutDomain(mlir::RankedTensorType type, bool externalBoundary) {
  llvm::SmallVector<MemLayout, 4> result{MemLayout::Tensor};
  if (externalBoundary || type.getRank() == 0)
    return result;
  if (type.getRank() >= 2)
    result.push_back(MemLayout::NTensor);
  result.push_back(MemLayout::Cx);
  if (type.getRank() >= 3)
    result.push_back(MemLayout::NCx);
  return result;
}

static std::optional<MemLayout>
getExplicitTensorAllocationLayout(mlir::Value value) {
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  auto allocation = result ? mlir::dyn_cast<mlir::bufferization::AllocTensorOp>(
                                 result.getOwner())
                           : mlir::bufferization::AllocTensorOp{};
  auto memory =
      allocation ? mlir::dyn_cast_or_null<MemoryAttr>(
                       allocation.getMemorySpace().value_or(mlir::Attribute{}))
                 : MemoryAttr{};
  if (!memory || memory.getSpace() != MemorySpace::SPM)
    return std::nullopt;
  return memory.getLayout();
}

static MemLayout getComputeLayout(mlir::RankedTensorType type) {
  if (type.getRank() == 0)
    return MemLayout::Tensor;
  return type.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
}

static MemLayout getComputeOperandLayout(mlir::linalg::LinalgOp operation,
                                         mlir::OpOperand &operand,
                                         mlir::RankedTensorType type) {
  // The convolution interface identifies RHS as the filter from current
  // indexing maps. Its canonical HWOI storage is one Cx volume, not one
  // independently aligned NCx slice per kernel row.
  if (operand.getOperandNumber() == 1 &&
      mlir::linalg::isaConvolutionOpInterface(operation))
    return MemLayout::Cx;
  return getComputeLayout(type);
}

static mlir::MemRefType getMemRefType(mlir::RankedTensorType tensor,
                                      MemorySpace space, MemLayout layout) {
  return mlir::MemRefType::get(
      tensor.getShape(), tensor.getElementType(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(tensor.getContext(), space, layout));
}

static ExactPBQPCost getLayoutMaterializationCost(mlir::RankedTensorType type,
                                                   MemLayout layout) {
  if (!type || !type.hasStaticShape())
    return kExactPBQPInfinity;
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(
          getMemRefType(type, MemorySpace::SPM, layout));
  if (!info || info->physicalBytes < 0)
    return kExactPBQPInfinity;
  // Keep one unit for the materialization itself and include the actual
  // physical footprint (including layout padding). This remains a query-local
  // ordering cost; legality and final SPM capacity still come only from the
  // materialized current IR and MiniMalloc.
  uint64_t bytes = static_cast<uint64_t>(info->physicalBytes);
  if (bytes == std::numeric_limits<uint64_t>::max())
    return kExactPBQPInfinity;
  ++bytes;
  return bytes > kExactPBQPInfinity ? kExactPBQPInfinity
                                     : static_cast<ExactPBQPCost>(bytes);
}

static bool proveReshapeLayoutAlias(mlir::RankedTensorType source,
                                    mlir::RankedTensorType result,
                                    MemLayout layout) {
  if (!source || !result || !source.hasStaticShape() ||
      !result.hasStaticShape() ||
      source.getElementType() != result.getElementType() ||
      source.getEncoding() != result.getEncoding())
    return false;
  analysis::IndexRelationResult resultToSource =
      source.getShape() == result.getShape()
          ? analysis::IndexRelation::identity(result.getShape())
          : analysis::IndexRelation::staticReshape(result.getShape(),
                                                   source.getShape());
  analysis::IndexRelationResult resultIdentity =
      analysis::IndexRelation::identity(result.getShape());
  if (!resultToSource.isExact() || !resultIdentity.isExact() ||
      (!resultToSource.get()->hasCanonicalRowMajorReshapeConstruction() &&
       source.getShape() != result.getShape()))
    return false;
  mlir::MemRefType sourceMemref =
      getMemRefType(source, MemorySpace::SPM, layout);
  mlir::MemRefType resultMemref =
      getMemRefType(result, MemorySpace::SPM, layout);
  mlir::FailureOr<analysis::PhysicalLayoutRelation> sourcePhysical =
      analysis::PhysicalLayoutRelation::create(sourceMemref);
  mlir::FailureOr<analysis::PhysicalLayoutRelation> resultPhysical =
      analysis::PhysicalLayoutRelation::create(resultMemref);
  if (mlir::failed(sourcePhysical) || mlir::failed(resultPhysical))
    return false;
  bool sameCoordinates = false;
  switch (layout) {
  case MemLayout::Tensor:
  case MemLayout::NTensor:
    sameCoordinates = true;
    break;
  case MemLayout::Cx:
    sameCoordinates = !source.getShape().empty() &&
                      !result.getShape().empty() &&
                      source.getShape().back() == result.getShape().back();
    break;
  case MemLayout::NCx:
    sameCoordinates = source.getRank() >= 3 && result.getRank() >= 3 &&
                      source.getShape().front() == result.getShape().front() &&
                      source.getShape().back() == result.getShape().back();
    break;
  }
  return sameCoordinates &&
         sourcePhysical->getPhysicalFootprintBytes() ==
             resultPhysical->getPhysicalFootprintBytes() &&
         sourcePhysical->getMinimumAlignmentBytes() ==
             resultPhysical->getMinimumAlignmentBytes() &&
         sourcePhysical->getValidElementCount() ==
             resultPhysical->getValidElementCount() &&
         sourcePhysical->getPaddingElementCount() ==
             resultPhysical->getPaddingElementCount();
}

static bool viewGroupSupportsLayout(llvm::ArrayRef<mlir::Value> values,
                                    MemLayout layout) {
  llvm::SmallPtrSet<mlir::Operation *, 8> checked;
  auto check = [&](mlir::Operation *operation) {
    if (!operation || !isTensorViewOperation(operation) ||
        !checked.insert(operation).second)
      return true;
    mlir::Value source;
    mlir::Value result;
    bool tensorCast = false;
    if (auto collapse =
            mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation)) {
      source = collapse.getSrc();
      result = collapse.getResult();
    } else if (auto expand =
                   mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation)) {
      source = expand.getSrc();
      result = expand.getResult();
    } else if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(operation)) {
      source = cast.getSource();
      result = cast.getDest();
      tensorCast = true;
    } else if (mlir::isa<mlir::tensor::InsertSliceOp, mlir::tensor::InsertOp,
                         mlir::tensor::ExtractOp>(operation)) {
      // Insert destination/result pairs have identical full-tensor physical
      // coordinates; the inserted source is a separate buffer/use and is not
      // part of this alias group. Scalar extract/insert likewise does not
      // introduce a second tensor view type.
      return true;
    } else {
      // extract_slice needs a separate base-offset/range alias proof. Keep it
      // on standard-view layouts until that proof exists.
      return layout == MemLayout::Tensor || layout == MemLayout::NTensor;
    }
    auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (tensorCast && sourceType && resultType &&
        (!sourceType.hasStaticShape() || !resultType.hasStaticShape()))
      return (layout == MemLayout::Tensor || layout == MemLayout::NTensor) &&
             mlir::tensor::CastOp::areCastCompatible(sourceType, resultType);
    return proveReshapeLayoutAlias(sourceType, resultType, layout);
  };
  for (mlir::Value value : values) {
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
      if (!check(result.getOwner()))
        return false;
    for (mlir::Operation *user : value.getUsers())
      if (!check(user))
        return false;
  }
  return true;
}

static void retargetRelationValue(StructuredMaterializationRelations &relations,
                                  mlir::Value oldValue, mlir::Value newValue) {
  for (MaterializedBufferRelation &relation : relations.buffers)
    if (relation.buffer == oldValue)
      relation.buffer = newValue;
  for (StructuredOutputRelation &relation : relations.structuralOutputs)
    if (relation.endpoint == oldValue)
      relation.endpoint = newValue;
  for (StructuredBoundaryRelation &relation : relations.boundaryRelations) {
    if (relation.sourceEndpoint == oldValue)
      relation.sourceEndpoint = newValue;
    if (relation.destinationEndpoint == oldValue)
      relation.destinationEndpoint = newValue;
  }
}

static bool
relationContainsEndpoint(const StructuredMaterializationRelations &relations,
                         mlir::Value value) {
  return llvm::any_of(relations.boundaryRelations, [&](const auto &relation) {
    return relation.sourceEndpoint == value ||
           relation.destinationEndpoint == value;
  });
}

struct OutputPiecePlan {
  unsigned relationIndex = 0;
  TileRegionOp region;
  unsigned resultIndex = 0;
  mlir::Value piece;
  mlir::RankedTensorType fullType;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  bool preserveFullResult = false;
  unsigned materializedResultIndex = 0;
};

struct BoundarySourcePlan {
  TileRegionOp region;
  unsigned resultIndex = 0;
  mlir::Value piece;
  bool preserveFullResult = false;
  unsigned materializedResultIndex = 0;
  llvm::SmallVector<unsigned, 2> relationIndices;
};

struct SameTileBoundaryUsePlan {
  TileRegionOp consumer;
  unsigned inputIndex = 0;
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 2> extracts;
};

struct SameTileBoundaryPiecePlan {
  TileRegionOp producer;
  unsigned resultIndex = 0;
  mlir::Value piece;
  mlir::tensor::InsertSliceOp insert;
  mlir::tensor::EmptyOp empty;
  llvm::SmallVector<SameTileBoundaryUsePlan, 2> uses;
};

static std::optional<llvm::SmallVector<int64_t, 4>>
getStaticValues(llvm::ArrayRef<mlir::OpFoldResult> values) {
  llvm::SmallVector<int64_t, 4> result;
  for (mlir::OpFoldResult value : values) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(value);
    if (!constant)
      return std::nullopt;
    result.push_back(*constant);
  }
  return result;
}

static bool destinationConsumesExactPiece(mlir::Value destination,
                                          llvm::ArrayRef<int64_t> pieceOffsets,
                                          llvm::ArrayRef<int64_t> pieceSizes,
                                          mlir::RankedTensorType pieceType) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(destination);
  auto region = argument && argument.getOwner()
                    ? mlir::dyn_cast_or_null<TileRegionOp>(
                          argument.getOwner()->getParentOp())
                    : TileRegionOp{};
  if (!argument || !region)
    return false;
  if (argument.getType() == pieceType)
    return true;
  if (argument.use_empty())
    return false;
  return llvm::all_of(argument.getUsers(), [&](mlir::Operation *user) {
    auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user);
    auto offsets =
        extract ? getStaticValues(extract.getMixedOffsets()) : std::nullopt;
    auto sizes =
        extract ? getStaticValues(extract.getMixedSizes()) : std::nullopt;
    auto strides =
        extract ? getStaticValues(extract.getMixedStrides()) : std::nullopt;
    return extract && extract.getSource() == argument && offsets && sizes &&
           strides && llvm::ArrayRef<int64_t>(*offsets) == pieceOffsets &&
           llvm::ArrayRef<int64_t>(*sizes) == pieceSizes &&
           llvm::all_of(*strides, [](int64_t stride) { return stride == 1; }) &&
           extract.getResult().getType() == pieceType;
  });
}

static mlir::FailureOr<llvm::SmallVector<BoundarySourcePlan, 8>>
preflightBoundarySources(StructuredMaterializationRelations &relations,
                         std::string &detail) {
  llvm::SmallVector<BoundarySourcePlan, 8> plans;
  llvm::DenseSet<mlir::Value> sourcesWithPartialDestination;
  for (const StructuredBoundaryRelation &relation :
       relations.boundaryRelations) {
    auto source = mlir::dyn_cast<mlir::OpResult>(relation.sourceEndpoint);
    auto region = source ? mlir::dyn_cast<TileRegionOp>(source.getOwner())
                         : TileRegionOp{};
    auto yield = region && region.getBody().hasOneBlock()
                     ? mlir::dyn_cast<TileYieldOp>(
                           region.getBody().front().getTerminator())
                     : TileYieldOp{};
    auto insert = yield && source.getResultNumber() < yield.getNumOperands()
                      ? yield.getOperand(source.getResultNumber())
                            .getDefiningOp<mlir::tensor::InsertSliceOp>()
                      : mlir::tensor::InsertSliceOp{};
    auto offsets =
        insert ? getStaticValues(insert.getMixedOffsets()) : std::nullopt;
    auto sizes =
        insert ? getStaticValues(insert.getMixedSizes()) : std::nullopt;
    auto pieceType =
        insert ? mlir::dyn_cast<mlir::RankedTensorType>(insert.getSourceType())
               : mlir::RankedTensorType{};
    if (insert && offsets && sizes && pieceType &&
        !destinationConsumesExactPiece(relation.destinationEndpoint, *offsets,
                                       *sizes, pieceType))
      sourcesWithPartialDestination.insert(source);
  }
  for (auto [relationIndex, relation] :
       llvm::enumerate(relations.boundaryRelations)) {
    auto source = mlir::dyn_cast<mlir::OpResult>(relation.sourceEndpoint);
    auto region = source ? mlir::dyn_cast<TileRegionOp>(source.getOwner())
                         : TileRegionOp{};
    if (!region || !region.getBody().hasOneBlock() ||
        source.getResultNumber() >= region.getNumResults()) {
      detail = "cross-Tile source endpoint is not a current TileRegion result";
      return mlir::failure();
    }
    if (sourcesWithPartialDestination.contains(source))
      continue;
    auto yield =
        mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
    if (!yield || source.getResultNumber() >= yield.getNumOperands()) {
      detail = "cross-Tile source endpoint has no current yielded value";
      return mlir::failure();
    }
    auto insert = yield.getOperand(source.getResultNumber())
                      .getDefiningOp<mlir::tensor::InsertSliceOp>();
    if (!insert)
      continue;
    auto empty = insert.getDest().getDefiningOp<mlir::tensor::EmptyOp>();
    auto offsets = getStaticValues(insert.getMixedOffsets());
    auto sizes = getStaticValues(insert.getMixedSizes());
    auto strides = getStaticValues(insert.getMixedStrides());
    auto pieceType =
        mlir::dyn_cast<mlir::RankedTensorType>(insert.getSourceType());
    if (!empty || empty.getType() != source.getType() || !offsets || !sizes ||
        !strides || !pieceType ||
        llvm::any_of(*strides, [](int64_t stride) { return stride != 1; }) ||
        llvm::ArrayRef<int64_t>(*sizes) != pieceType.getShape()) {
      // This source has real assembly semantics or a relation that is not
      // expressible as one metadata piece. Preserve the current endpoint for
      // movement instead of replacing it with a guessed slice.
      continue;
    }
    auto existing = llvm::find_if(plans, [&](const BoundarySourcePlan &plan) {
      return plan.region == region &&
             plan.resultIndex == source.getResultNumber();
    });
    if (existing != plans.end()) {
      if (existing->piece != insert.getSource()) {
        detail = "one cross-Tile source endpoint maps to different pieces";
        return mlir::failure();
      }
      existing->relationIndices.push_back(relationIndex);
      continue;
    }
    const bool observable =
        llvm::any_of(relations.structuralOutputs,
                     [&](const StructuredOutputRelation &output) {
                       return output.endpoint == source;
                     });
    plans.push_back({region,
                     source.getResultNumber(),
                     insert.getSource(),
                     !source.use_empty() || observable,
                     0,
                     {static_cast<unsigned>(relationIndex)}});
  }
  return plans;
}

static mlir::LogicalResult elideBoundarySourceViews(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    llvm::SmallVectorImpl<BoundarySourcePlan> &plans,
    LayoutOptimizationStatistics &statistics, std::string &detail) {
  llvm::SmallVector<
      std::pair<mlir::Operation *, llvm::SmallVector<unsigned, 2>>, 8>
      byRegion;
  for (auto [index, plan] : llvm::enumerate(plans)) {
    auto existing = llvm::find_if(byRegion, [&](const auto &entry) {
      return entry.first == plan.region.getOperation();
    });
    if (existing == byRegion.end()) {
      byRegion.push_back({plan.region, {}});
      existing = std::prev(byRegion.end());
    }
    existing->second.push_back(index);
  }
  mlir::IRRewriter rewriter(module.getContext());
  for (auto &[operation, indices] : byRegion) {
    auto oldRegion = mlir::cast<TileRegionOp>(operation);
    auto oldYield =
        mlir::cast<TileYieldOp>(oldRegion.getBody().front().getTerminator());
    llvm::SmallVector<mlir::Type, 8> resultTypes(
        oldRegion.getResultTypes().begin(), oldRegion.getResultTypes().end());
    llvm::SmallVector<mlir::Value, 8> yieldValues(oldYield.getValues().begin(),
                                                  oldYield.getValues().end());
    for (unsigned planIndex : indices) {
      BoundarySourcePlan &plan = plans[planIndex];
      if (plan.preserveFullResult) {
        plan.materializedResultIndex = resultTypes.size();
        resultTypes.push_back(plan.piece.getType());
        yieldValues.push_back(plan.piece);
      } else {
        plan.materializedResultIndex = plan.resultIndex;
        resultTypes[plan.resultIndex] = plan.piece.getType();
        yieldValues[plan.resultIndex] = plan.piece;
      }
    }
    rewriter.setInsertionPoint(oldRegion);
    auto newRegion = rewriter.create<TileRegionOp>(
        oldRegion.getLoc(), resultTypes, oldRegion.getInputs());
    newRegion.getBody().takeBody(oldRegion.getBody());
    auto newYield =
        mlir::cast<TileYieldOp>(newRegion.getBody().front().getTerminator());
    rewriter.modifyOpInPlace(
        newYield, [&] { newYield.getValuesMutable().assign(yieldValues); });
    for (auto [oldResult, newResult] : llvm::zip_equal(
             oldRegion.getResults(),
             newRegion.getResults().take_front(oldRegion.getNumResults()))) {
      if (oldResult.getType() == newResult.getType())
        rewriter.replaceAllUsesWith(oldResult, newResult);
      else if (!oldResult.use_empty()) {
        detail = "cross-Tile source piece rewrite found an unexpected full use";
        return mlir::failure();
      }
      retargetRelationValue(relations, oldResult, newResult);
    }
    rewriter.eraseOp(oldRegion);
    for (unsigned planIndex : indices) {
      BoundarySourcePlan &plan = plans[planIndex];
      plan.region = newRegion;
      mlir::Value piece = newRegion.getResult(plan.materializedResultIndex);
      for (unsigned relationIndex : plan.relationIndices)
        relations.boundaryRelations[relationIndex].sourceEndpoint = piece;
      ++statistics.boundarySourceViewsElided;
    }
  }
  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<SameTileBoundaryPiecePlan, 8>>
preflightSameTileBoundaryPieces(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations) {
  llvm::SmallVector<SameTileBoundaryPiecePlan, 8> plans;
  module.walk([&](TileRegionOp producer) {
    if (!producer.getBody().hasOneBlock())
      return;
    auto yield =
        mlir::dyn_cast<TileYieldOp>(producer.getBody().front().getTerminator());
    if (!yield)
      return;
    auto producerTile = producer->getParentOfType<TileModuleOp>();
    for (auto [resultIndex, result] : llvm::enumerate(producer.getResults())) {
      if (result.use_empty() || relationContainsEndpoint(relations, result) ||
          llvm::any_of(relations.structuralOutputs,
                       [&](const StructuredOutputRelation &output) {
                         return output.endpoint == result;
                       }))
        continue;
      auto insert = yield.getOperand(resultIndex)
                        .getDefiningOp<mlir::tensor::InsertSliceOp>();
      auto empty = insert
                       ? insert.getDest().getDefiningOp<mlir::tensor::EmptyOp>()
                       : mlir::tensor::EmptyOp{};
      auto offsets =
          insert ? getStaticValues(insert.getMixedOffsets()) : std::nullopt;
      auto sizes =
          insert ? getStaticValues(insert.getMixedSizes()) : std::nullopt;
      auto strides =
          insert ? getStaticValues(insert.getMixedStrides()) : std::nullopt;
      auto pieceType =
          insert
              ? mlir::dyn_cast<mlir::RankedTensorType>(insert.getSourceType())
              : mlir::RankedTensorType{};
      if (!insert || !empty || empty.getType() != result.getType() ||
          !offsets || !sizes || !strides || !pieceType ||
          !insert.getResult().hasOneUse() ||
          *insert.getResult().user_begin() != yield.getOperation() ||
          llvm::ArrayRef<int64_t>(*sizes) != pieceType.getShape() ||
          llvm::any_of(*strides, [](int64_t stride) { return stride != 1; }))
        continue;

      SameTileBoundaryPiecePlan plan{producer,
                                     static_cast<unsigned>(resultIndex),
                                     insert.getSource(),
                                     insert,
                                     empty,
                                     {}};
      bool exact = static_cast<bool>(producerTile);
      for (mlir::OpOperand &use : result.getUses()) {
        auto consumer = mlir::dyn_cast<TileRegionOp>(use.getOwner());
        unsigned inputIndex = use.getOperandNumber();
        if (!consumer ||
            consumer->getParentOfType<TileModuleOp>() != producerTile ||
            !consumer.getBody().hasOneBlock() ||
            inputIndex >= consumer.getBody().front().getNumArguments()) {
          exact = false;
          break;
        }
        mlir::BlockArgument argument =
            consumer.getBody().front().getArgument(inputIndex);
        SameTileBoundaryUsePlan consumerPlan{consumer, inputIndex, {}};
        for (mlir::Operation *user : argument.getUsers()) {
          auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user);
          auto consumerOffsets =
              extract ? getStaticValues(extract.getMixedOffsets())
                      : std::nullopt;
          auto consumerSizes =
              extract ? getStaticValues(extract.getMixedSizes()) : std::nullopt;
          auto consumerStrides =
              extract ? getStaticValues(extract.getMixedStrides())
                      : std::nullopt;
          if (!extract || extract.getSource() != argument || !consumerOffsets ||
              !consumerSizes || !consumerStrides ||
              *consumerOffsets != *offsets || *consumerSizes != *sizes ||
              *consumerStrides != *strides ||
              extract.getResult().getType() != pieceType) {
            exact = false;
            break;
          }
          consumerPlan.extracts.push_back(extract);
        }
        if (!exact || consumerPlan.extracts.empty()) {
          exact = false;
          break;
        }
        plan.uses.push_back(std::move(consumerPlan));
      }
      if (exact && !plan.uses.empty())
        plans.push_back(std::move(plan));
    }
  });
  return plans;
}

static mlir::LogicalResult elideSameTileBoundaryPieces(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    LayoutOptimizationStatistics &statistics, std::string &detail) {
  auto plans = preflightSameTileBoundaryPieces(module, relations);
  if (mlir::failed(plans))
    return mlir::failure();
  mlir::IRRewriter rewriter(module.getContext());
  for (SameTileBoundaryPiecePlan &plan : *plans) {
    mlir::OpResult result =
        mlir::cast<mlir::OpResult>(plan.producer.getResult(plan.resultIndex));
    auto yield = mlir::cast<TileYieldOp>(
        plan.producer.getBody().front().getTerminator());
    rewriter.modifyOpInPlace(plan.producer, [&] {
      result.setType(plan.piece.getType());
      yield->setOperand(plan.resultIndex, plan.piece);
    });
    for (SameTileBoundaryUsePlan &use : plan.uses) {
      mlir::BlockArgument argument =
          use.consumer.getBody().front().getArgument(use.inputIndex);
      rewriter.modifyOpInPlace(use.consumer, [&] {
        use.consumer->setOperand(use.inputIndex, result);
        argument.setType(plan.piece.getType());
      });
      for (mlir::tensor::ExtractSliceOp extract : use.extracts) {
        rewriter.replaceAllUsesWith(extract.getResult(), argument);
        rewriter.eraseOp(extract);
      }
    }
    if (!plan.insert->use_empty()) {
      detail = "same-Tile compact boundary left a live insert wrapper";
      return mlir::failure();
    }
    rewriter.eraseOp(plan.insert);
    if (plan.empty->use_empty())
      rewriter.eraseOp(plan.empty);
    ++statistics.boundarySourceViewsElided;
  }
  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<OutputPiecePlan, 8>>
preflightOutputPieces(StructuredMaterializationRelations &relations,
                      std::string &detail) {
  llvm::SmallVector<OutputPiecePlan, 8> plans;
  std::set<std::pair<mlir::Operation *, unsigned>> seen;
  for (auto [relationIndex, relation] :
       llvm::enumerate(relations.structuralOutputs)) {
    auto endpoint = mlir::dyn_cast<mlir::OpResult>(relation.endpoint);
    auto region = endpoint ? mlir::dyn_cast<TileRegionOp>(endpoint.getOwner())
                           : TileRegionOp{};
    if (!region || !region.getBody().hasOneBlock() ||
        endpoint.getResultNumber() >= region.getNumResults() ||
        !seen.insert({region, endpoint.getResultNumber()}).second) {
      detail = "observable output is not one unique current TileRegion result";
      return mlir::failure();
    }
    auto fullType = mlir::dyn_cast<mlir::RankedTensorType>(endpoint.getType());
    auto function = region->getParentOfType<mlir::func::FuncOp>();
    auto module = region->getParentOfType<mlir::ModuleOp>();
    auto yield =
        mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
    if (!function || !module ||
        !mlir::SymbolTable::symbolKnownUseEmpty(function, module) ||
        !fullType || !fullType.hasStaticShape() || !yield ||
        endpoint.getResultNumber() >= yield.getNumOperands()) {
      detail = "observable output requires an uncalled entry and a static "
               "ranked TileRegion yield";
      return mlir::failure();
    }

    mlir::Value yielded = yield.getOperand(endpoint.getResultNumber());
    mlir::Value piece = yielded;
    llvm::SmallVector<int64_t, 4> offsets(fullType.getRank(), 0);
    llvm::SmallVector<int64_t, 4> sizes(fullType.getShape());
    if (auto insert = yielded.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
      auto empty = insert.getDest().getDefiningOp<mlir::tensor::EmptyOp>();
      // Only an insert into a fresh empty tensor is a structural publication
      // shell that can be narrowed to its source piece. A temporal loop or
      // peeled tail may yield an insert into an actual loop-carried result;
      // that value is already the complete observable tensor and must remain
      // intact for DPS binding.
      if (!empty && !insert.getDest().getDefiningOp<mlir::scf::ForOp>()) {
        detail = "observable insert_slice is neither one exact static output "
                 "piece nor one temporal loop result";
        return mlir::failure();
      }
      if (empty) {
        auto staticOffsets = getStaticValues(insert.getMixedOffsets());
        auto staticSizes = getStaticValues(insert.getMixedSizes());
        auto staticStrides = getStaticValues(insert.getMixedStrides());
        auto pieceType =
            mlir::dyn_cast<mlir::RankedTensorType>(insert.getSourceType());
        if (empty.getType() != fullType || !staticOffsets || !staticSizes ||
            !staticStrides || !pieceType ||
            llvm::any_of(*staticStrides,
                         [](int64_t stride) { return stride != 1; }) ||
            llvm::ArrayRef<int64_t>(*staticSizes) != pieceType.getShape()) {
          detail =
              "observable insert_slice is not one exact static output piece";
          return mlir::failure();
        }
        offsets = std::move(*staticOffsets);
        sizes = std::move(*staticSizes);
        piece = insert.getSource();
      }
    }
    for (auto [offset, size, extent] :
         llvm::zip_equal(offsets, sizes, fullType.getShape()))
      if (offset < 0 || size <= 0 || offset > extent - size) {
        detail = "observable output piece is outside its destination";
        return mlir::failure();
      }

    // Several merge groups may publish disjoint pieces of the same program
    // result on one Tile. Their actual insert_slice bounds, rather than a
    // one-output-per-Tile restriction, establish whether sharing a DDR result
    // root is legal.
    for (const OutputPiecePlan &previous : plans) {
      if (relations.structuralOutputs[previous.relationIndex].outputIndex !=
              relation.outputIndex ||
          previous.region->getParentOfType<TileModuleOp>() !=
              region->getParentOfType<TileModuleOp>())
        continue;
      if (previous.fullType != fullType) {
        detail = "same-Tile output pieces have different full result types";
        return mlir::failure();
      }
      bool disjoint = false;
      for (size_t axis = 0; axis < offsets.size(); ++axis)
        disjoint |=
            offsets[axis] + sizes[axis] <= previous.offsets[axis] ||
            previous.offsets[axis] + previous.sizes[axis] <= offsets[axis];
      if (!disjoint) {
        detail = "same-Tile output pieces overlap";
        return mlir::failure();
      }
    }
    plans.push_back(OutputPiecePlan{
        static_cast<unsigned>(relationIndex), region,
        endpoint.getResultNumber(), piece, fullType, std::move(offsets),
        std::move(sizes),
        !endpoint.use_empty() || relationContainsEndpoint(relations, endpoint),
        0});
  }
  return plans;
}

static mlir::LogicalResult materializeOutputDestinations(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    llvm::SmallVectorImpl<OutputPiecePlan> &plans,
    LayoutOptimizationStatistics &statistics, std::string &detail) {
  llvm::SmallVector<
      std::pair<mlir::Operation *, llvm::SmallVector<unsigned, 2>>, 8>
      byRegion;
  for (auto [index, plan] : llvm::enumerate(plans)) {
    auto existing = llvm::find_if(byRegion, [&](const auto &entry) {
      return entry.first == plan.region.getOperation();
    });
    if (existing == byRegion.end()) {
      byRegion.push_back({plan.region, {}});
      existing = std::prev(byRegion.end());
    }
    existing->second.push_back(index);
  }

  mlir::IRRewriter rewriter(module.getContext());
  for (auto &[operation, indices] : byRegion) {
    auto oldRegion = mlir::cast<TileRegionOp>(operation);
    auto oldYield =
        mlir::cast<TileYieldOp>(oldRegion.getBody().front().getTerminator());
    llvm::SmallVector<mlir::Type, 8> resultTypes(
        oldRegion.getResultTypes().begin(), oldRegion.getResultTypes().end());
    llvm::SmallVector<mlir::Value, 8> yieldValues(oldYield.getValues().begin(),
                                                  oldYield.getValues().end());
    for (unsigned planIndex : indices) {
      OutputPiecePlan &plan = plans[planIndex];
      if (plan.preserveFullResult) {
        plan.materializedResultIndex = resultTypes.size();
        resultTypes.push_back(plan.piece.getType());
        yieldValues.push_back(plan.piece);
      } else {
        plan.materializedResultIndex = plan.resultIndex;
        resultTypes[plan.resultIndex] = plan.piece.getType();
        yieldValues[plan.resultIndex] = plan.piece;
      }
    }

    rewriter.setInsertionPoint(oldRegion);
    auto newRegion = rewriter.create<TileRegionOp>(
        oldRegion.getLoc(), resultTypes, oldRegion.getInputs());
    newRegion.getBody().takeBody(oldRegion.getBody());
    auto newYield =
        mlir::cast<TileYieldOp>(newRegion.getBody().front().getTerminator());
    rewriter.modifyOpInPlace(
        newYield, [&] { newYield.getValuesMutable().assign(yieldValues); });
    for (auto [oldResult, newResult] : llvm::zip_equal(
             oldRegion.getResults(),
             newRegion.getResults().take_front(oldRegion.getNumResults()))) {
      if (oldResult.getType() == newResult.getType())
        rewriter.replaceAllUsesWith(oldResult, newResult);
      else if (!oldResult.use_empty()) {
        detail = "output piece signature rewrite found an unexpected "
                 "full-result use";
        return mlir::failure();
      }
      retargetRelationValue(relations, oldResult, newResult);
    }
    rewriter.eraseOp(oldRegion);
    for (unsigned planIndex : indices) {
      OutputPiecePlan &plan = plans[planIndex];
      plan.region = newRegion;
      relations.structuralOutputs[plan.relationIndex].endpoint =
          newRegion.getResult(plan.materializedResultIndex);
    }
  }

  for (OutputPiecePlan &plan : plans) {
    mlir::Value piece =
        relations.structuralOutputs[plan.relationIndex].endpoint;
    auto function =
        piece.getDefiningOp()
            ? piece.getDefiningOp()->getParentOfType<mlir::func::FuncOp>()
            : mlir::func::FuncOp{};
    if (!function || function.isExternal() ||
        !function.getBody().hasOneBlock()) {
      detail = "observable output piece has no current entry function";
      return mlir::failure();
    }
    mlir::MemRefType destinationType =
        getMemRefType(plan.fullType, MemorySpace::DDR, MemLayout::Tensor);
    const unsigned argumentIndex = function.getNumArguments();
    function.insertArgument(argumentIndex, destinationType,
                            mlir::DictionaryAttr{}, piece.getLoc());
    mlir::BlockArgument destination =
        function.getBody().front().getArgument(argumentIndex);

    mlir::OpBuilder builder(plan.region);
    builder.setInsertionPointAfter(plan.region);
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (auto [offset, size] : llvm::zip_equal(plan.offsets, plan.sizes)) {
      offsets.push_back(builder.getIndexAttr(offset));
      sizes.push_back(builder.getIndexAttr(size));
      strides.push_back(builder.getIndexAttr(1));
    }
    auto subview = builder.create<mlir::memref::SubViewOp>(
        piece.getLoc(), destination, offsets, sizes, strides);
    auto publication =
        builder.create<mlir::bufferization::MaterializeInDestinationOp>(
            piece.getLoc(), piece, subview.getResult());
    publication.setRestrict(true);
    publication.setWritable(true);
    relations.structuralOutputs[plan.relationIndex].endpoint =
        subview.getResult();
    ++statistics.outputDestinations;
    ++statistics.outputSubviews;
  }

  // The full-tensor insert/empty wrappers that were replaced above are now
  // ordinary dead pure IR.  Close only that local dead chain.
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *, 8> dead;
    module.walk([&](mlir::Operation *operation) {
      if (operation->use_empty() && mlir::isMemoryEffectFree(operation) &&
          mlir::isa<mlir::tensor::InsertSliceOp, mlir::tensor::EmptyOp>(
              operation))
        dead.push_back(operation);
    });
    for (mlir::Operation *operation : llvm::reverse(dead)) {
      if (!operation->use_empty())
        continue;
      rewriter.eraseOp(operation);
      changed = true;
    }
  }
  return mlir::success();
}

static void collectTensorValues(mlir::ModuleOp module,
                                llvm::SmallVectorImpl<mlir::Value> &values,
                                llvm::DenseMap<mlir::Value, unsigned> &indices,
                                ValueUnion &unions) {
  auto append = [&](mlir::Value value) {
    if (!isTensorValue(value) || indices.count(value))
      return;
    indices.try_emplace(value, unions.add());
    values.push_back(value);
  };
  module.walk([&](mlir::Operation *operation) {
    for (mlir::Value operand : operation->getOperands())
      append(operand);
    for (mlir::Value result : operation->getResults())
      append(result);
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          append(argument);
  });
}

static void
buildBufferEquivalence(mlir::ModuleOp module,
                       const llvm::DenseMap<mlir::Value, unsigned> &indices,
                       ValueUnion &unions) {
  auto unite = [&](mlir::Value lhs, mlir::Value rhs) {
    auto lhsIt = indices.find(lhs);
    auto rhsIt = indices.find(rhs);
    if (lhsIt != indices.end() && rhsIt != indices.end())
      unions.unite(lhsIt->second, rhsIt->second);
  };
  module.walk([&](mlir::Operation *operation) {
    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(operation))
      for (unsigned index = 0;
           index < dps.getNumDpsInits() && index < operation->getNumResults();
           ++index)
        if (isTensorValue(dps.getDpsInitOperand(index)->get()) &&
            isTensorValue(operation->getResult(index)))
          unite(dps.getDpsInitOperand(index)->get(),
                operation->getResult(index));
    if (auto insert = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(operation))
      unite(insert.getDest(), insert.getResult());
    if (auto insert = mlir::dyn_cast<mlir::tensor::InsertOp>(operation))
      unite(insert.getDest(), insert.getResult());
    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(operation))
      unite(extract.getSource(), extract.getResult());
    if (auto collapse =
            mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation))
      unite(collapse.getSrc(), collapse.getResult());
    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation))
      unite(expand.getSrc(), expand.getResult());
    if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(operation))
      unite(cast.getSource(), cast.getDest());
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
      for (unsigned index = 0; index < loop.getInitArgs().size(); ++index) {
        unite(loop.getInitArgs()[index], loop.getRegionIterArgs()[index]);
        unite(loop.getResult(index), loop.getRegionIterArgs()[index]);
        if (yield && index < yield.getNumOperands())
          unite(yield.getOperand(index), loop.getRegionIterArgs()[index]);
      }
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
      for (unsigned index = 0; index < branch.getNumResults(); ++index)
        for (mlir::Region *region :
             {&branch.getThenRegion(), &branch.getElseRegion()}) {
          if (region->empty())
            continue;
          auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
              region->front().getTerminator());
          if (yield && index < yield.getNumOperands())
            unite(branch.getResult(index), yield.getOperand(index));
        }
    }
    if (auto region = mlir::dyn_cast<TileRegionOp>(operation)) {
      if (!region.getBody().hasOneBlock())
        return;
      auto yield =
          mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
      if (!yield)
        return;
      for (unsigned index = 0;
           index < region.getNumResults() && index < yield.getNumOperands();
           ++index)
        unite(region.getResult(index), yield.getOperand(index));
    }
  });
}

static mlir::FailureOr<llvm::SmallVector<ValueGroup, 32>> buildValueGroups(
    llvm::ArrayRef<mlir::Value> values,
    const llvm::DenseMap<mlir::Value, unsigned> &indices, ValueUnion &unions,
    llvm::DenseMap<mlir::Value, unsigned> &groupByValue, std::string &detail) {
  llvm::DenseMap<unsigned, unsigned> groupByRoot;
  llvm::SmallVector<ValueGroup, 32> groups;
  for (mlir::Value value : values) {
    unsigned root = unions.find(indices.lookup(value));
    auto [iterator, inserted] = groupByRoot.try_emplace(root, groups.size());
    if (inserted)
      groups.emplace_back();
    unsigned group = iterator->second;
    groups[group].values.push_back(value);
    groupByValue.try_emplace(value, group);
  }
  for (ValueGroup &group : groups) {
    bool external = llvm::any_of(group.values, isFunctionEntryArgument);
    std::optional<llvm::SmallVector<MemLayout, 4>> intersection;
    std::optional<MemLayout> explicitLayout;
    for (mlir::Value value : group.values) {
      auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
      llvm::SmallVector<MemLayout, 4> current = getLayoutDomain(type, external);
      if (!intersection) {
        intersection = std::move(current);
      } else {
        llvm::erase_if(*intersection, [&](MemLayout layout) {
          return !llvm::is_contained(current, layout);
        });
      }
      if (std::optional<MemLayout> layout =
              getExplicitTensorAllocationLayout(value)) {
        if (explicitLayout && *explicitLayout != *layout) {
          detail =
              "one buffer-equivalent group has conflicting explicit layouts";
          return mlir::failure();
        }
        explicitLayout = *layout;
      }
    }
    if (intersection && explicitLayout)
      llvm::erase_if(*intersection, [&](MemLayout layout) {
        return layout != *explicitLayout;
      });
    if (intersection)
      llvm::erase_if(*intersection, [&](MemLayout layout) {
        return !viewGroupSupportsLayout(group.values, layout);
      });
    if (!intersection || intersection->empty()) {
      std::string groupSummary;
      llvm::raw_string_ostream stream(groupSummary);
      stream << "one buffer-equivalent tensor group has no common layout:";
      for (mlir::Value value : group.values) {
        stream << " [";
        value.getType().print(stream);
        stream << " from ";
        if (mlir::Operation *owner = value.getDefiningOp())
          stream << owner->getName();
        else
          stream << "block-argument";
        stream << "]";
      }
      stream.flush();
      detail = std::move(groupSummary);
      return mlir::failure();
    }
    group.layouts = std::move(*intersection);
  }
  return groups;
}

static bool hasReductionIterator(mlir::linalg::LinalgOp operation) {
  return llvm::any_of(operation.getIteratorTypesArray(),
                      mlir::linalg::isReductionIterator);
}

static bool isFixedComputeLayoutOp(mlir::linalg::LinalgOp operation);

static std::optional<MemLayout> getLayout(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  MemoryAttr memory = memref ? getWaferMemoryAttr(memref) : MemoryAttr{};
  return memory ? std::optional<MemLayout>(memory.getLayout()) : std::nullopt;
}

static mlir::FailureOr<llvm::SmallVector<UseBinding, 64>>
buildUseBindings(mlir::ModuleOp module,
                 const llvm::DenseMap<mlir::Value, unsigned> &groupByValue,
                 std::string &detail) {
  llvm::SmallVector<UseBinding, 64> uses;
  module.walk([&](mlir::linalg::LinalgOp operation) {
    // Layout-polymorphic Linalg accepts each current operand layout directly.
    // Only fixed compute tuples introduce use bindings.
    if (!isFixedComputeLayoutOp(operation))
      return;
    for (mlir::OpOperand &operand : operation->getOpOperands()) {
      if (!isTensorValue(operand.get()))
        continue;
      auto group = groupByValue.find(operand.get());
      if (group == groupByValue.end()) {
        detail = "Linalg tensor use has no current value group";
        continue;
      }
      UseBinding use;
      use.owner = operation;
      use.operandNumber = operand.getOperandNumber();
      use.source = operand.get();
      use.sourceGroup = group->second;
      auto tensor = mlir::cast<mlir::RankedTensorType>(operand.get().getType());
      use.layouts = getLayoutDomain(tensor, /*externalBoundary=*/false);
      if (isFixedComputeLayoutOp(operation)) {
        MemLayout required =
            mlir::isa<mlir::linalg::FillOp>(operation.getOperation())
                ? MemLayout::Tensor
                : getComputeOperandLayout(operation, operand, tensor);
        use.layouts.assign({required});
      }
      uses.push_back(std::move(use));
    }
  });
  if (!detail.empty())
    return mlir::failure();
  return uses;
}

static llvm::SmallVector<ResultBinding, 32>
buildResultBindings(mlir::ModuleOp module,
                    const llvm::DenseMap<mlir::Value, unsigned> &groupByValue) {
  llvm::SmallVector<ResultBinding, 32> results;
  module.walk([&](mlir::linalg::LinalgOp operation) {
    if (!isFixedComputeLayoutOp(operation))
      return;
    for (mlir::Value value : operation->getResults()) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
      auto group = groupByValue.find(value);
      if (!result || result.use_empty() || !tensor ||
          group == groupByValue.end())
        continue;
      results.push_back(
          {result, group->second,
           mlir::isa<mlir::linalg::FillOp>(operation.getOperation())
               ? MemLayout::Tensor
               : getComputeLayout(tensor)});
    }
  });
  return results;
}

static bool isFixedComputeLayoutOp(mlir::linalg::LinalgOp operation) {
  return mlir::linalg::isaContractionOpInterface(operation) ||
         hasReductionIterator(operation) ||
         mlir::isa<mlir::linalg::FillOp>(operation.getOperation());
}

static bool hasInterveningWrite(mlir::Operation *earlier,
                                mlir::Operation *later) {
  if (!earlier || !later || earlier->getBlock() != later->getBlock() ||
      !earlier->isBeforeInBlock(later))
    return true;
  for (mlir::Operation *operation = earlier->getNextNode();
       operation && operation != later; operation = operation->getNextNode()) {
    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!effects) {
      if (!mlir::isMemoryEffectFree(operation))
        return true;
      continue;
    }
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> instances;
    effects.getEffects(instances);
    if (llvm::any_of(instances, [](const auto &instance) {
          return mlir::isa<mlir::MemoryEffects::Write,
                           mlir::MemoryEffects::Free>(instance.getEffect());
        }))
      return true;
  }
  return false;
}

static bool
tupleStateIsLegal(mlir::linalg::LinalgOp operation,
                  llvm::ArrayRef<mlir::RankedTensorType> coordinateTypes,
                  llvm::ArrayRef<MemLayout> layouts) {
  if (!isFixedComputeLayoutOp(operation))
    return true;
  if (mlir::isa<mlir::linalg::FillOp>(operation.getOperation()))
    return llvm::all_of(
        layouts, [](MemLayout layout) { return layout == MemLayout::Tensor; });
  unsigned coordinate = 0;
  for (mlir::OpOperand &operand : operation->getOpOperands()) {
    if (!isTensorValue(operand.get()))
      continue;
    if (layouts[coordinate] !=
        getComputeOperandLayout(operation, operand,
                                coordinateTypes[coordinate]))
      return false;
    ++coordinate;
  }
  return true;
}

static bool hasOnlyReadUses(LayoutMaterializeOp operation) {
  return llvm::all_of(
      operation.getResult().getUses(), [](mlir::OpOperand &use) {
        auto effects =
            mlir::dyn_cast<mlir::MemoryEffectOpInterface>(use.getOwner());
        if (!effects)
          return false;
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> instances;
        effects.getEffectsOnValue(use.get(), instances);
        return !instances.empty() &&
               llvm::all_of(instances, [](const auto &instance) {
                 return mlir::isa<mlir::MemoryEffects::Read>(
                     instance.getEffect());
               });
      });
}

static bool hasNoInterveningWrite(LayoutMaterializeOp prior,
                                  LayoutMaterializeOp current,
                                  mlir::AliasAnalysis &aliases) {
  if (prior->getBlock() != current->getBlock() ||
      !prior->isBeforeInBlock(current))
    return false;
  for (mlir::Operation *operation = prior->getNextNode();
       operation && operation != current.getOperation();
       operation = operation->getNextNode())
    if (aliases.getModRef(operation, current.getSource()).isMod())
      return false;
  return true;
}

static mlir::LogicalResult convertLayoutCopies(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    LayoutOptimizationStatistics &statistics, std::string &detail) {
  llvm::SmallVector<mlir::memref::CopyOp, 16> copies;
  module.walk([&](mlir::memref::CopyOp copy) { copies.push_back(copy); });
  mlir::IRRewriter rewriter(module.getContext());
  mlir::DominanceInfo copyDominance(module);
  for (mlir::memref::CopyOp copy : copies) {
    if (copy.getSource() == copy.getTarget()) {
      rewriter.eraseOp(copy);
      continue;
    }
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(copy.getSource().getType());
    auto destType =
        mlir::dyn_cast<mlir::MemRefType>(copy.getTarget().getType());
    if (!sourceType || !destType || !isWaferSPMMemRefType(sourceType) ||
        !isWaferSPMMemRefType(destType) ||
        sourceType.getShape() != destType.getShape() ||
        sourceType.getElementType() != destType.getElementType() ||
        getLayout(sourceType) == getLayout(destType))
      continue;
    analysis::IndexRelationResult identity =
        analysis::IndexRelation::identity(destType.getShape());
    mlir::MemRefType ownedDestType = mlir::MemRefType::get(
        destType.getShape(), destType.getElementType(),
        mlir::MemRefLayoutAttrInterface{}, destType.getMemorySpace());
    if (!identity.isExact() ||
        mlir::failed(analysis::TransferRealizability::proveGatherScatter(
            sourceType, ownedDestType, *identity.get()))) {
      llvm::raw_string_ostream detailStream(detail);
      detailStream
          << "selected layout conversion has no exact GatherScatter proof: "
             "source=";
      sourceType.print(detailStream);
      detailStream << ", destination=";
      ownedDestType.print(detailStream);
      detailStream << ", source_owner=";
      if (mlir::Operation *owner = copy.getSource().getDefiningOp())
        detailStream << owner->getName().getStringRef();
      else
        detailStream << "block_argument";
      detailStream << ", destination_owner=";
      if (mlir::Operation *owner = copy.getTarget().getDefiningOp())
        detailStream << owner->getName().getStringRef();
      else
        detailStream << "block_argument";
      return mlir::failure();
    }
    if (auto subview =
            copy.getTarget().getDefiningOp<mlir::memref::SubViewOp>()) {
      rewriter.setInsertionPoint(copy);
      auto materialize = rewriter.create<LayoutMaterializeOp>(
          copy.getLoc(), ownedDestType, copy.getSource());
      mlir::Value destination = copy.getTarget();
      auto baseType = mlir::dyn_cast<mlir::MemRefType>(subview.getSourceType());
      const bool fullSubview =
          baseType && baseType.getShape() == destType.getShape() &&
          llvm::all_of(subview.getMixedOffsets(),
                       [](mlir::OpFoldResult value) {
                         return mlir::getConstantIntValue(value) == 0;
                       }) &&
          llvm::all_of(
              llvm::zip_equal(subview.getMixedSizes(), baseType.getShape()),
              [](auto valueAndExtent) {
                return mlir::getConstantIntValue(std::get<0>(valueAndExtent)) ==
                       std::get<1>(valueAndExtent);
              }) &&
          llvm::all_of(subview.getMixedStrides(), [](mlir::OpFoldResult value) {
            return mlir::getConstantIntValue(value) == 1;
          });
      if (fullSubview)
        destination = subview.getSource();
      rewriter.create<MoveCopyIntoOp>(copy.getLoc(), materialize.getResult(),
                                      destination);
      rewriter.eraseOp(copy);
      if (subview.getResult().use_empty())
        rewriter.eraseOp(subview);
      continue;
    }
    auto allocation = copy.getTarget().getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || allocation->getBlock() != copy->getBlock() ||
        !allocation->isBeforeInBlock(copy) ||
        !llvm::all_of(copy.getTarget().getUsers(), [&](mlir::Operation *user) {
          return user == copy || copyDominance.properlyDominates(copy, user);
        }))
      continue;
    rewriter.setInsertionPoint(copy);
    auto materialize = rewriter.create<LayoutMaterializeOp>(
        copy.getLoc(), ownedDestType, copy.getSource());
    mlir::Value oldTarget = copy.getTarget();
    retargetRelationValue(relations, oldTarget, materialize.getResult());
    rewriter.replaceAllUsesWith(oldTarget, materialize.getResult());
    rewriter.eraseOp(copy);
    if (allocation->use_empty())
      rewriter.eraseOp(allocation);
  }

  module.walk(
      [&](LayoutMaterializeOp) { ++statistics.layoutMaterializationsBefore; });

  // Close dead and exactly shareable layout materializations on current IR.
  mlir::DominanceInfo dominance(module);
  mlir::AliasAnalysis aliases(module);
  llvm::DenseMap<std::pair<mlir::Value, mlir::Type>,
                 llvm::SmallVector<LayoutMaterializeOp, 2>>
      available;
  llvm::SmallVector<LayoutMaterializeOp, 8> materializations;
  module.walk([&](LayoutMaterializeOp op) { materializations.push_back(op); });
  for (LayoutMaterializeOp operation : materializations) {
    if (operation.getResult().use_empty()) {
      rewriter.eraseOp(operation);
      ++statistics.unusedMaterializationsErased;
      continue;
    }
    if (!hasOnlyReadUses(operation))
      continue;
    auto key =
        std::make_pair(operation.getSource(), operation.getResult().getType());
    auto &owners = available[key];
    auto replacement = llvm::find_if(owners, [&](LayoutMaterializeOp prior) {
      return hasNoInterveningWrite(prior, operation, aliases) &&
             llvm::all_of(operation.getResult().getUses(),
                          [&](mlir::OpOperand &use) {
                            return dominance.dominates(prior.getResult(),
                                                       use.getOwner());
                          });
    });
    if (replacement == owners.end()) {
      owners.push_back(operation);
      continue;
    }
    retargetRelationValue(relations, operation.getResult(),
                          replacement->getResult());
    rewriter.replaceAllUsesWith(operation.getResult(),
                                replacement->getResult());
    rewriter.eraseOp(operation);
    ++statistics.sharedMaterializationsReused;
  }
  return mlir::success();
}

static bool isAllowedTensorBoundaryOperation(mlir::Operation *operation) {
  return mlir::isa<TileRegionOp, TileYieldOp, mlir::bufferization::ToMemrefOp,
                   mlir::bufferization::ToTensorOp>(operation);
}

static void
recordCurrentBuffers(mlir::ModuleOp module,
                     StructuredMaterializationRelations &relations) {
  relations.buffers.clear();
  auto append = [&](mlir::Operation *owner, mlir::Value buffer,
                    MaterializedBufferRole role) {
    if (!buffer || !mlir::isa<mlir::BaseMemRefType>(buffer.getType()) ||
        llvm::any_of(relations.buffers, [&](const auto &relation) {
          return relation.owner == owner && relation.buffer == buffer &&
                 relation.role == role;
        }))
      return;
    relations.buffers.push_back({owner, buffer, role});
  };
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::ModuleOp, TileModuleOp, TileRegionOp,
                  mlir::func::FuncOp>(operation))
      return;
    for (mlir::Value operand : operation->getOperands())
      append(operation, operand, MaterializedBufferRole::Operand);
    for (mlir::Value result : operation->getResults())
      append(operation, result,
             mlir::isa<mlir::memref::AllocOp>(operation)
                 ? MaterializedBufferRole::Scratch
                 : MaterializedBufferRole::Result);
    if (mlir::isa<LayoutMaterializeOp, mlir::memref::CopyOp>(operation))
      for (mlir::Value operand : operation->getOperands())
        append(operation, operand, MaterializedBufferRole::Movement);
  });
}

static mlir::LogicalResult localizeBufferizationGlobals(mlir::ModuleOp module,
                                                        std::string &detail) {
  llvm::DenseMap<mlir::StringAttr, mlir::memref::GlobalOp> moduleGlobals;
  for (mlir::memref::GlobalOp global :
       module.getBody()->getOps<mlir::memref::GlobalOp>())
    if (!moduleGlobals.try_emplace(global.getSymNameAttr(), global).second) {
      detail = "bufferization created duplicate module globals";
      return mlir::failure();
    }

  llvm::DenseSet<mlir::StringAttr> localizedNames;
  for (TileModuleOp tile : module.getOps<TileModuleOp>()) {
    llvm::SmallVector<mlir::StringAttr, 4> required;
    tile.walk([&](mlir::memref::GetGlobalOp getGlobal) {
      mlir::StringAttr name = getGlobal.getNameAttr().getAttr();
      if (!llvm::is_contained(required, name))
        required.push_back(name);
    });
    llvm::sort(required, [](mlir::StringAttr lhs, mlir::StringAttr rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    mlir::OpBuilder builder(&tile.getBody().front(),
                            tile.getBody().front().begin());
    for (mlir::StringAttr name : required) {
      mlir::memref::GlobalOp source = moduleGlobals.lookup(name);
      if (!source) {
        detail = "bufferized Tile references an unknown memref.global";
        return mlir::failure();
      }
      mlir::Operation *existing =
          mlir::SymbolTable::lookupSymbolIn(tile, name.getValue());
      if (existing) {
        auto local = mlir::dyn_cast<mlir::memref::GlobalOp>(existing);
        if (!local || local.getType() != source.getType()) {
          detail = "Tile symbol conflicts with a bufferization global";
          return mlir::failure();
        }
      } else {
        builder.clone(*source);
      }
      localizedNames.insert(name);
    }
  }

  llvm::DenseSet<mlir::StringAttr> externallyReferenced;
  module.walk([&](mlir::memref::GetGlobalOp getGlobal) {
    if (!getGlobal->getParentOfType<TileModuleOp>())
      externallyReferenced.insert(getGlobal.getNameAttr().getAttr());
  });
  for (mlir::StringAttr name : localizedNames)
    if (!externallyReferenced.contains(name))
      moduleGlobals.lookup(name).erase();
  return mlir::success();
}

static bool
sharesOutputStorage(mlir::Value value,
                    const StructuredMaterializationRelations &relations,
                    StorageRootMemo &memo) {
  const llvm::DenseSet<mlir::Value> &roots = memo.getStorageRoots(value);
  return llvm::any_of(
      relations.structuralOutputs, [&](const StructuredOutputRelation &output) {
        if (!output.endpoint ||
            !mlir::isa<mlir::BaseMemRefType>(output.endpoint.getType()))
          return false;
        const llvm::DenseSet<mlir::Value> &outputRoots =
            memo.getStorageRoots(output.endpoint);
        return llvm::any_of(roots, [&](mlir::Value root) {
          return outputRoots.contains(root);
        });
      });
}

} // namespace

mlir::LogicalResult verifyLayoutResolvedTileRegions(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)))
    return mlir::failure();
  bool valid = true;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<LinalgExtAttentionOp, LinalgExtOnlineAttentionOp>(
            operation)) {
      valid = false;
      return mlir::WalkResult::interrupt();
    }
    bool hasTensor =
        llvm::any_of(operation->getOperandTypes(),
                     [](mlir::Type type) {
                       return mlir::isa<mlir::TensorType>(type);
                     }) ||
        llvm::any_of(operation->getResultTypes(), [](mlir::Type type) {
          return mlir::isa<mlir::TensorType>(type);
        });
    if (!hasTensor)
      for (mlir::Region &region : operation->getRegions())
        for (mlir::Block &block : region)
          hasTensor |=
              llvm::any_of(block.getArgumentTypes(), [](mlir::Type type) {
                return mlir::isa<mlir::TensorType>(type);
              });
    if (hasTensor && !isAllowedTensorBoundaryOperation(operation)) {
      valid = false;
      return mlir::WalkResult::interrupt();
    }
    if (auto materialize = mlir::dyn_cast<LayoutMaterializeOp>(operation))
      if (materialize.getResult().use_empty() ||
          getLayout(materialize.getSource().getType()) ==
              getLayout(materialize.getResult().getType())) {
        valid = false;
        return mlir::WalkResult::interrupt();
      }
    return mlir::WalkResult::advance();
  });
  return mlir::success(valid);
}

static void localizeEmptySlices(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations) {
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 16> slices;
  module.walk([&](mlir::tensor::ExtractSliceOp slice) {
    if (slice->getParentOfType<TileRegionOp>())
      slices.push_back(slice);
  });
  StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  for (auto slice : slices) {
    auto empty = slice.getSource().getDefiningOp<mlir::tensor::EmptyOp>();
    if (!empty || !slice.getType().hasStaticShape())
      continue;
    // Empty tensors carry no contents; a selected slice needs only its own
    // destination shape. This is the pinned Tensor empty/slice fold.
    rewriter.setInsertionPoint(slice);
    rewriter.replaceOpWithNewOp<mlir::tensor::EmptyOp>(slice, slice.getType(),
                                                       mlir::ValueRange{});
    if (empty->use_empty())
      rewriter.eraseOp(empty);
  }
}

LayoutOptimizationResult
resolveCurrentLayoutsAndBufferize(mlir::ModuleOp module,
                                  StructuredMaterializationRelations &relations,
                                  uint64_t workLimit) {
  LayoutOptimizationResult result;
  result.statistics.invocations = 1;
  if (!module) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail = "current layout transformation has no module";
    return result;
  }
  if (mlir::failed(mlir::verify(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.detail = "current layout input has invalid IR or stale relations";
    return result;
  }
  bool residualAttention = false;
  bool alreadyBufferized = false;
  module.walk([&](mlir::Operation *operation) {
    residualAttention |=
        mlir::isa<LinalgExtAttentionOp, LinalgExtOnlineAttentionOp>(operation);
    if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation))
      alreadyBufferized |=
          llvm::any_of(linalg->getOperandTypes(), [](mlir::Type type) {
            return mlir::isa<mlir::MemRefType>(type);
          });
  });
  if (residualAttention || alreadyBufferized) {
    result.detail = residualAttention
                        ? "layout input still contains attention semantics"
                        : "layout/bufferization stage was already applied";
    return result;
  }

  localizeEmptySlices(module, relations);

  std::string detail;
  auto boundaryPlans = preflightBoundarySources(relations, detail);
  if (mlir::failed(boundaryPlans)) {
    result.status = ExactPBQPStatus::NoSolution;
    result.detail = std::move(detail);
    return result;
  }
  if (mlir::failed(elideBoundarySourceViews(module, relations, *boundaryPlans,
                                            result.statistics, detail))) {
    result.detail = std::move(detail);
    return result;
  }
  if (mlir::failed(elideSameTileBoundaryPieces(module, relations,
                                               result.statistics, detail))) {
    result.detail = std::move(detail);
    return result;
  }
  auto outputPlans = preflightOutputPieces(relations, detail);
  if (mlir::failed(outputPlans)) {
    result.status = ExactPBQPStatus::NoSolution;
    result.detail = std::move(detail);
    return result;
  }
  if (mlir::failed(materializeOutputDestinations(
          module, relations, *outputPlans, result.statistics, detail))) {
    result.detail = std::move(detail);
    return result;
  }

  llvm::SmallVector<mlir::Value, 64> values;
  llvm::DenseMap<mlir::Value, unsigned> valueIndices;
  ValueUnion unions;
  collectTensorValues(module, values, valueIndices, unions);
  if (values.empty()) {
    result.detail = "current layout input contains no tensor SSA values";
    return result;
  }
  buildBufferEquivalence(module, valueIndices, unions);
  llvm::DenseMap<mlir::Value, unsigned> groupByValue;
  auto maybeGroups =
      buildValueGroups(values, valueIndices, unions, groupByValue, detail);
  if (mlir::failed(maybeGroups)) {
    result.status = ExactPBQPStatus::NoSolution;
    result.detail = std::move(detail);
    return result;
  }
  llvm::SmallVector<ValueGroup, 32> groups = std::move(*maybeGroups);
  auto maybeUses = buildUseBindings(module, groupByValue, detail);
  if (mlir::failed(maybeUses)) {
    result.detail = std::move(detail);
    return result;
  }
  llvm::SmallVector<UseBinding, 64> uses = std::move(*maybeUses);
  llvm::SmallVector<ResultBinding, 32> resultBindings =
      buildResultBindings(module, groupByValue);
  result.statistics.valueGroups = groups.size();
  result.statistics.useBindings = uses.size();

  // For the materialization-count objective, a group layout that is neither a
  // current fixed-compute publication layout nor a current fixed-use layout is
  // strictly dominated by every relevant layout available to that group. If
  // no relevant layout is available, every state has the same objective and
  // the first canonical state is sufficient. This reduction changes only the
  // query-local objective-equivalent PBQP domain; it is not persisted as a
  // separate layout frontier or search axis.
  for (auto [groupIndex, group] : llvm::enumerate(groups)) {
    if (group.layouts.empty()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "current value group has no legal layout state";
      return result;
    }
    llvm::SmallVector<MemLayout, 4> relevantLayouts;
    for (const ResultBinding &binding : resultBindings)
      if (binding.publishedGroup == groupIndex &&
          !llvm::is_contained(relevantLayouts, binding.computeLayout))
        relevantLayouts.push_back(binding.computeLayout);
    for (const UseBinding &use : uses)
      if (use.sourceGroup == groupIndex)
        for (MemLayout layout : use.layouts)
          if (!llvm::is_contained(relevantLayouts, layout))
            relevantLayouts.push_back(layout);
    const size_t oldSize = group.layouts.size();
    const MemLayout canonicalLayout = group.layouts.front();
    group.layouts.erase(llvm::remove_if(group.layouts,
                                        [&](MemLayout layout) {
                                          return !llvm::is_contained(
                                              relevantLayouts, layout);
                                        }),
                        group.layouts.end());
    if (group.layouts.empty())
      group.layouts.push_back(canonicalLayout);
    result.statistics.dominatedLayoutStatesPruned +=
        oldSize - group.layouts.size();
  }

  ExactPBQPProblem problem;
  for (auto [groupIndex, group] : llvm::enumerate(groups)) {
    group.variable = problem.variables.size();
    std::vector<ExactPBQPCost> costs(group.layouts.size(), 0);
    for (const ResultBinding &binding : resultBindings) {
      if (binding.publishedGroup != groupIndex)
        continue;
      for (auto [state, layout] : llvm::enumerate(group.layouts))
        if (layout != binding.computeLayout)
          costs[state] = addCost(
              costs[state],
              getLayoutMaterializationCost(
                  mlir::cast<mlir::RankedTensorType>(binding.result.getType()),
                  layout));
    }
    problem.variables.push_back(ExactPBQPVariable{std::move(costs)});
  }
  for (UseBinding &use : uses) {
    use.variable = problem.variables.size();
    problem.variables.push_back(
        ExactPBQPVariable{std::vector<ExactPBQPCost>(use.layouts.size(), 0)});
  }

  FactorBuilder factors(problem);
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 4>> usesByOp;
  for (auto [index, use] : llvm::enumerate(uses))
    usesByOp[use.owner].push_back(index);
  llvm::SmallVector<TupleVariable, 16> tuples;
  bool exactTupleDomainBounded = true;
  module.walk([&](mlir::linalg::LinalgOp operation) {
    if (!isFixedComputeLayoutOp(operation))
      return;
    TupleVariable tuple;
    tuple.operation = operation;
    llvm::SmallVector<mlir::RankedTensorType, 4> coordinateTypes;
    for (unsigned useIndex : usesByOp.lookup(operation)) {
      tuple.coordinates.push_back(uses[useIndex].variable);
      tuple.coordinateLayouts.push_back(uses[useIndex].layouts);
      coordinateTypes.push_back(
          mlir::cast<mlir::RankedTensorType>(uses[useIndex].source.getType()));
    }
    if (tuple.coordinates.empty())
      return;
    llvm::SmallVector<uint32_t, 4> current(tuple.coordinates.size(), 0);
    std::function<void(unsigned)> enumerate = [&](unsigned coordinate) {
      if (tuple.states.size() > kMaximumLayoutTupleStates)
        return;
      if (coordinate == current.size()) {
        llvm::SmallVector<MemLayout, 4> layouts;
        for (auto [index, state] : llvm::enumerate(current))
          layouts.push_back(tuple.coordinateLayouts[index][state]);
        if (tupleStateIsLegal(operation, coordinateTypes, layouts))
          tuple.states.push_back(current);
        return;
      }
      for (uint32_t state = 0;
           state < tuple.coordinateLayouts[coordinate].size(); ++state) {
        current[coordinate] = state;
        enumerate(coordinate + 1);
      }
    };
    enumerate(0);
    tuples.push_back(std::move(tuple));
  });
  for (TupleVariable &tuple : tuples) {
    if (tuple.states.empty()) {
      result.status = ExactPBQPStatus::NoSolution;
      result.detail = "one current Linalg op has no supported layout tuple";
      return result;
    }
    if (tuple.states.size() > kMaximumLayoutTupleStates) {
      // The first legal tuple remains a complete typed assignment witness.
      // Keep the bounded prefix only for factor validation and force the
      // solver to return that witness as Feasible; never claim optimality over
      // a truncated tuple domain.
      exactTupleDomainBounded = false;
    }
    tuple.variable = problem.variables.size();
    problem.variables.push_back(
        ExactPBQPVariable{std::vector<ExactPBQPCost>(tuple.states.size(), 0)});
    for (auto [coordinate, variable] : llvm::enumerate(tuple.coordinates)) {
      factors.add(tuple.variable, variable,
                  [&](uint32_t tupleState, uint32_t coordinateState) {
                    return tuple.states[tupleState][coordinate] ==
                                   coordinateState
                               ? ExactPBQPCost{0}
                               : kExactPBQPInfinity;
                  });
    }
  }
  result.statistics.tupleVariables = tuples.size();

  llvm::SmallVector<UseCohort, 32> cohorts;
  for (auto [useIndex, use] : llvm::enumerate(uses)) {
    mlir::Block *block = use.owner->getBlock();
    UseCohort *found = nullptr;
    for (UseCohort &cohort : llvm::reverse(cohorts))
      if (cohort.source == use.source && cohort.block == block &&
          !hasInterveningWrite(cohort.lastOwner, use.owner)) {
        found = &cohort;
        break;
      }
    if (!found) {
      cohorts.push_back({use.source, block, use.sourceGroup, {}, use.owner});
      found = &cohorts.back();
    }
    found->uses.push_back(useIndex);
    found->lastOwner = use.owner;
  }

  llvm::SmallVector<ConversionActivation, 64> activations;
  for (auto [cohortIndex, cohort] : llvm::enumerate(cohorts)) {
    llvm::SmallVector<MemLayout, 4> layouts =
        groups[cohort.sourceGroup].layouts;
    for (unsigned useIndex : cohort.uses)
      for (MemLayout layout : uses[useIndex].layouts)
        if (!llvm::is_contained(layouts, layout))
          layouts.push_back(layout);
    llvm::sort(layouts, [](MemLayout lhs, MemLayout rhs) {
      return static_cast<unsigned>(lhs) < static_cast<unsigned>(rhs);
    });
    for (MemLayout layout : layouts) {
      ConversionActivation activation;
      activation.cohort = cohortIndex;
      activation.layout = layout;
      activation.variable = problem.variables.size();
      const mlir::RankedTensorType sourceType =
          mlir::cast<mlir::RankedTensorType>(cohort.source.getType());
      const ExactPBQPCost materializationCost =
          getLayoutMaterializationCost(sourceType, layout);
      // The activation state remains an exact current-IR materialization
      // decision, but its finite objective includes physical bytes/padding so
      // a copy-count tie cannot prefer a much larger layout blindly.
      problem.variables.push_back(ExactPBQPVariable{{0, 0, materializationCost}});
      activations.push_back(activation);

      const ValueGroup &sourceGroup = groups[cohort.sourceGroup];
      factors.add(
          sourceGroup.variable, activation.variable,
          [&](uint32_t sourceState, uint32_t activationState) {
            const bool sourceIsTarget =
                sourceGroup.layouts[sourceState] == layout;
            if (sourceIsTarget)
              return activationState == static_cast<uint32_t>(
                                            ActivationState::Materialized)
                         ? kExactPBQPInfinity
                         : ExactPBQPCost{0};
            return activationState == static_cast<uint32_t>(
                                          ActivationState::SourceIsTarget)
                       ? kExactPBQPInfinity
                       : ExactPBQPCost{0};
          });
      for (unsigned useIndex : cohort.uses) {
        const UseBinding &use = uses[useIndex];
        factors.add(use.variable, activation.variable,
                    [&](uint32_t useState, uint32_t activationState) {
                      if (use.layouts[useState] != layout)
                        return ExactPBQPCost{0};
                      return activationState == static_cast<uint32_t>(
                                                    ActivationState::Inactive)
                                 ? kExactPBQPInfinity
                                 : ExactPBQPCost{0};
                    });
      }
    }
  }
  result.statistics.conversionActivations = activations.size();
  factors.appendTo(problem);
  result.statistics.pbqpVariables = problem.variables.size();
  result.statistics.pbqpFactors = problem.factors.size();

  constexpr uint32_t unassigned = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> canonicalAssignment(problem.variables.size(),
                                            unassigned);
  for (const ValueGroup &group : groups)
    canonicalAssignment[group.variable] = 0;
  for (const TupleVariable &tuple : tuples) {
    canonicalAssignment[tuple.variable] = 0;
    for (auto [coordinate, variable] : llvm::enumerate(tuple.coordinates)) {
      uint32_t state = tuple.states.front()[coordinate];
      uint32_t &selected = canonicalAssignment[variable];
      if (selected != unassigned && selected != state) {
        result.status = ExactPBQPStatus::BrokenContract;
        result.detail =
            "canonical layout assignment has conflicting op tuple states";
        return result;
      }
      selected = state;
    }
  }
  for (const UseBinding &use : uses)
    if (canonicalAssignment[use.variable] == unassigned)
      canonicalAssignment[use.variable] = 0;
  for (const ConversionActivation &activation : activations) {
    const UseCohort &cohort = cohorts[activation.cohort];
    const ValueGroup &sourceGroup = groups[cohort.sourceGroup];
    uint32_t sourceState = canonicalAssignment[sourceGroup.variable];
    if (sourceState >= sourceGroup.layouts.size()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail =
          "canonical layout assignment selected an invalid group state";
      return result;
    }
    const bool needed = llvm::any_of(cohort.uses, [&](unsigned useIndex) {
      const UseBinding &use = uses[useIndex];
      uint32_t useState = canonicalAssignment[use.variable];
      return useState < use.layouts.size() &&
             use.layouts[useState] == activation.layout;
    });
    ActivationState state = ActivationState::Inactive;
    if (needed)
      state = sourceGroup.layouts[sourceState] == activation.layout
                  ? ActivationState::SourceIsTarget
                  : ActivationState::Materialized;
    canonicalAssignment[activation.variable] = static_cast<uint32_t>(state);
  }
  if (llvm::is_contained(canonicalAssignment, unassigned)) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail = "canonical layout assignment did not cover every variable";
    return result;
  }
  ++result.statistics.canonicalAssignmentsBuilt;

  ExactPBQPSolveOptions solveOptions;
  solveOptions.workLimit = exactTupleDomainBounded ? workLimit : 0;
  solveOptions.semanticTieVariableCount = static_cast<uint32_t>(groups.size());
  solveOptions.initialFeasibleAssignment = std::move(canonicalAssignment);
  ExactPBQPResult solved = solveExactPBQP(problem, solveOptions);
  result.status = solved.status;
  result.statistics.solverWork = solved.work;
  if (solved.status == ExactPBQPStatus::Feasible)
    ++result.statistics.canonicalAssignmentFallbacks;
  if (solved.status != ExactPBQPStatus::Optimal &&
      solved.status != ExactPBQPStatus::Feasible) {
    result.detail =
        "current value/use layout PBQP did not produce a complete legal "
        "assignment; "
        "status=" +
        std::to_string(static_cast<unsigned>(solved.status)) +
        ", work=" + std::to_string(solved.work);
    return result;
  }
  if (solved.assignment.size() != problem.variables.size()) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail =
        "current value/use layout PBQP returned an incomplete assignment";
    return result;
  }

  llvm::DenseMap<mlir::Value, MemLayout> selectedLayouts;
  for (const ValueGroup &group : groups) {
    uint32_t state = solved.assignment[group.variable];
    if (state >= group.layouts.size()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "layout PBQP selected an invalid value-group state";
      return result;
    }
    for (mlir::Value value : group.values)
      selectedLayouts.try_emplace(value, group.layouts[state]);
  }

  mlir::IRRewriter assignmentRewriter(module.getContext());
  for (const ConversionActivation &activation : activations) {
    uint32_t state = solved.assignment[activation.variable];
    if (state != static_cast<uint32_t>(ActivationState::Materialized))
      continue;
    const UseCohort &cohort = cohorts[activation.cohort];
    llvm::SmallVector<unsigned, 4> selectedUses;
    for (unsigned useIndex : cohort.uses) {
      const UseBinding &use = uses[useIndex];
      uint32_t useState = solved.assignment[use.variable];
      if (useState >= use.layouts.size()) {
        result.status = ExactPBQPStatus::BrokenContract;
        result.detail = "layout PBQP selected an invalid use state";
        return result;
      }
      if (use.layouts[useState] == activation.layout)
        selectedUses.push_back(useIndex);
    }
    if (selectedUses.empty()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "selected layout materialization has no actual use";
      return result;
    }
    mlir::Operation *firstOwner = uses[selectedUses.front()].owner;
    for (unsigned useIndex : selectedUses)
      if (uses[useIndex].owner->isBeforeInBlock(firstOwner))
        firstOwner = uses[useIndex].owner;
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(cohort.source.getType());
    if (!tensorType || !isTileLocalTensor(cohort.source)) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail =
          "layout materialization source is not Tile-local tensor SSA";
      return result;
    }
    assignmentRewriter.setInsertionPoint(firstOwner);
    auto copy = assignmentRewriter.create<mlir::bufferization::AllocTensorOp>(
        firstOwner->getLoc(), tensorType, mlir::ValueRange{}, cohort.source);
    copy.setMemorySpaceAttr(MemoryAttr::get(
        module.getContext(), MemorySpace::SPM, activation.layout));
    for (unsigned useIndex : selectedUses) {
      UseBinding &use = uses[useIndex];
      assignmentRewriter.modifyOpInPlace(use.owner, [&] {
        use.owner->getOpOperand(use.operandNumber).set(copy.getResult());
      });
    }
    ++result.statistics.selectedMaterializations;
  }

  for (ResultBinding &binding : resultBindings) {
    const ValueGroup &published = groups[binding.publishedGroup];
    uint32_t state = solved.assignment[published.variable];
    if (state >= published.layouts.size()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail =
          "layout PBQP selected an invalid result publication state";
      return result;
    }
    MemLayout targetLayout = published.layouts[state];
    if (targetLayout == binding.computeLayout || binding.result.use_empty())
      continue;
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(binding.result.getType());
    if (!tensorType ||
        !binding.result.getOwner()->getParentOfType<TileRegionOp>()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail =
          "fixed-layout result publication is not Tile-local tensor SSA";
      return result;
    }
    assignmentRewriter.setInsertionPointAfter(binding.result.getOwner());
    auto publication =
        assignmentRewriter.create<mlir::bufferization::AllocTensorOp>(
            binding.result.getLoc(), tensorType, mlir::ValueRange{},
            binding.result);
    publication.setMemorySpaceAttr(
        MemoryAttr::get(module.getContext(), MemorySpace::SPM, targetLayout));
    assignmentRewriter.replaceAllUsesExcept(
        binding.result, publication.getResult(), publication);
    retargetRelationValue(relations, binding.result, publication.getResult());
    ++result.statistics.selectedMaterializations;
  }

  mlir::bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.allowUnknownOps = true;
  options.allowReturnAllocsFromLoops = true;
  options.analysisHeuristic = mlir::bufferization::OneShotBufferizationOptions::
      AnalysisHeuristic::BottomUpFromTerminators;
  options.inferFunctionResultLayout = true;
  options.bufferAlignment = 256;
  options.defaultMemorySpaceFn =
      [](mlir::TensorType tensorType) -> std::optional<mlir::Attribute> {
    return MemoryAttr::get(tensorType.getContext(), MemorySpace::DDR,
                           MemLayout::Tensor);
  };
  bool missingLayout = false;
  options.unknownTypeConverterFn =
      [&](mlir::Value value, mlir::Attribute,
          const mlir::bufferization::BufferizationOptions &)
      -> mlir::BaseMemRefType {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    auto found = selectedLayouts.find(value);
    if (!tensor || found == selectedLayouts.end()) {
      missingLayout = true;
      return {};
    }
    const bool tileLocal = isTileLocalTensor(value);
    return getMemRefType(tensor,
                         tileLocal ? MemorySpace::SPM : MemorySpace::DDR,
                         tileLocal ? found->second : MemLayout::Tensor);
  };
  options.functionArgTypeConverterFn =
      [module](mlir::TensorType tensorType, mlir::Attribute,
               mlir::func::FuncOp function,
               const mlir::bufferization::BufferizationOptions &)
      -> mlir::BaseMemRefType {
    const MemorySpace space =
        mlir::SymbolTable::symbolKnownUseEmpty(function, module)
            ? MemorySpace::DDR
            : MemorySpace::SPM;
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(tensorType);
    if (!ranked)
      return mlir::UnrankedMemRefType::get(
          tensorType.getElementType(),
          MemoryAttr::get(tensorType.getContext(), space, MemLayout::Tensor));
    return getMemRefType(ranked, space, MemLayout::Tensor);
  };
  // Identify only state edges that One-Shot cannot keep in the same buffer.
  // The analysis sees the actual layout materializations; in-place loops and
  // metadata aliases require no additional destination constraint.
  struct LoopStateBinding {
    mlir::scf::YieldOp yield;
    unsigned index;
    mlir::BlockArgument destination;
    MemLayout layout;
  };
  llvm::SmallVector<LoopStateBinding, 8> loopBindings;
  {
    mlir::bufferization::OneShotAnalysisState state(module, options);
    if (mlir::failed(mlir::bufferization::analyzeModuleOp(module, state))) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "One-Shot loop state analysis failed";
      return result;
    }
    auto collected = module.walk([&](mlir::scf::ForOp loop) {
      if (!loop->getParentOfType<TileRegionOp>())
        return mlir::WalkResult::advance();
      auto yield =
          mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
      for (auto [index, argument] : llvm::enumerate(loop.getRegionIterArgs())) {
        mlir::Value value = yield.getOperand(index);
        if (!isTensorValue(argument) ||
            state.areEquivalentBufferizedValues(value, argument))
          continue;
        auto layout = selectedLayouts.find(argument);
        if (layout == selectedLayouts.end())
          return mlir::WalkResult::interrupt();
        loopBindings.push_back(
            {yield, static_cast<unsigned>(index), argument, layout->second});
      }
      return mlir::WalkResult::advance();
    });
    if (collected.wasInterrupted()) {
      result.status = ExactPBQPStatus::BrokenContract;
      result.detail = "loop state destination has no selected current layout";
      return result;
    }
  }
  // Destroy the analysis before mutation. A fresh One-Shot analysis below
  // resolves all read/write conflicts introduced by the destination bindings.
  for (LoopStateBinding state : loopBindings) {
    assignmentRewriter.setInsertionPoint(state.yield);
    auto binding =
        assignmentRewriter
            .create<mlir::bufferization::MaterializeInDestinationOp>(
                state.yield.getLoc(), state.yield.getOperand(state.index),
                state.destination);
    selectedLayouts.try_emplace(binding.getResult(), state.layout);
    assignmentRewriter.modifyOpInPlace(state.yield, [&] {
      state.yield->setOperand(state.index, binding.getResult());
    });
  }
  ++result.statistics.bufferizationInvocations;
  if (mlir::failed(
          mlir::bufferization::runOneShotModuleBufferize(module, options)) ||
      missingLayout) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail =
        missingLayout
            ? "One-Shot Bufferization requested an unassigned tensor value"
            : "One-Shot Bufferization failed on selected current layouts";
    return result;
  }
  if (mlir::failed(localizeBufferizationGlobals(module, detail))) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail = std::move(detail);
    return result;
  }

  if (mlir::failed(
          convertLayoutCopies(module, relations, result.statistics, detail))) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail = std::move(detail);
    return result;
  }

  StorageRootMemo roots;
  module.walk([&](mlir::memref::CopyOp copy) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(copy.getSource().getType());
    auto destType =
        mlir::dyn_cast<mlir::MemRefType>(copy.getTarget().getType());
    if (!sourceType || !destType)
      return;
    if (isWaferDDRMemRefType(sourceType) && isWaferDDRMemRefType(destType) &&
        sharesOutputStorage(copy.getTarget(), relations, roots))
      ++result.statistics.redundantPublicationCopies;
    else
      ++result.statistics.necessaryCopies;
  });
  if (result.statistics.redundantPublicationCopies != 0) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail = "bufferization produced a redundant DDR publication copy";
    return result;
  }

  module.walk([&](LayoutMaterializeOp) {
    ++result.statistics.layoutMaterializationsAfter;
  });
  recordCurrentBuffers(module, relations);
  retainCurrentStructuredBufferRelations(module, relations);
  if (mlir::failed(verifyLayoutResolvedTileRegions(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations))) {
    result.status = ExactPBQPStatus::BrokenContract;
    result.detail =
        "layout/bufferization produced invalid current IR or relations";
    return result;
  }
  result.status = solved.status;
  if (solved.status == ExactPBQPStatus::Feasible)
    result.detail =
        "exact layout optimization did not complete; applied the factor-valid "
        "canonical assignment";
  return result;
}

} // namespace wafer::compiler::detail

namespace wafer {

#define GEN_PASS_DEF_RESOLVECURRENTLAYOUTSANDBUFFERIZEPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

struct ResolveCurrentLayoutsAndBufferizePass
    : public impl::ResolveCurrentLayoutsAndBufferizePassBase<
          ResolveCurrentLayoutsAndBufferizePass> {
  using impl::ResolveCurrentLayoutsAndBufferizePassBase<
      ResolveCurrentLayoutsAndBufferizePass>::
      ResolveCurrentLayoutsAndBufferizePassBase;

  void runOnOperation() final {
    StructuredMaterializationRelations relations;
    compiler::detail::LayoutOptimizationResult result =
        compiler::detail::resolveCurrentLayoutsAndBufferize(getOperation(),
                                                            relations);
    if (result.succeeded())
      return;
    getOperation().emitError()
        << "current layout/bufferization failed: " << result.detail;
    signalPassFailure();
  }
};

} // namespace
} // namespace wafer
