//===- TensorAssemblyRead.cpp - Bounded current assembly accesses --------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace wafer::analysis {
namespace {
TensorAssemblyReadResult fail(TensorAssemblyStatus status,
                              llvm::StringRef detail) {
  return {status, {}, {}, detail.str()};
}
TensorAssemblyStatus assemblyStatus(TensorResultIndexingStatus status) {
  switch (status) {
  case TensorResultIndexingStatus::Exact:
    return TensorAssemblyStatus::Exact;
  case TensorResultIndexingStatus::Unsupported:
    return TensorAssemblyStatus::Unsupported;
  case TensorResultIndexingStatus::ResourceExhausted:
    return TensorAssemblyStatus::ResourceExhausted;
  case TensorResultIndexingStatus::BrokenContract:
    return TensorAssemblyStatus::BrokenContract;
  }
  llvm_unreachable("unknown tensor indexing status");
}
struct ReadCoordinate {
  int64_t first = 0;
  std::optional<TensorLoopGrid> grid;
};

TensorAssemblyReadResult
queryAssemblyRead(mlir::Value value, llvm::ArrayRef<ReadCoordinate> coordinates,
                  llvm::ArrayRef<int64_t> sizes,
                  const IndexRelationLimits &limits) {
  if (!value || !value.getDefiningOp<mlir::SubsetInsertionOpInterface>())
    return fail(TensorAssemblyStatus::NotAssembly, "value has no insert chain");
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() ||
      coordinates.size() != static_cast<size_t>(type.getRank()) ||
      sizes.size() != coordinates.size())
    return fail(TensorAssemblyStatus::BrokenContract,
                "assembly read has inconsistent static dimensions");
  TensorAssemblyReadResult result;
  result.status = TensorAssemblyStatus::Exact;
  result.shape.assign(sizes.begin(), sizes.end());
  llvm::SmallVector<std::optional<TensorLoopGrid>, 4> grids;
  llvm::SmallVector<unsigned, 4> loopIndices;
  llvm::SmallVector<int64_t, 4> firstOffsets;
  for (auto [coordinate, size, extent] :
       llvm::zip_equal(coordinates, sizes, type.getShape())) {
    if (size <= 0)
      return fail(TensorAssemblyStatus::Unsupported,
                  "assembly read requires positive static sizes");
    auto grid = coordinate.grid;
    int64_t first = coordinate.first, last = coordinate.first;
    unsigned loopIndex = 0;
    if (grid) {
      last = *getTensorLoopGridFloor(*grid, grid->upper - 1);
      auto found = llvm::find_if(result.loops, [&](const auto &loop) {
        return loop.induction == grid->induction;
      });
      loopIndex = std::distance(result.loops.begin(), found);
      if (found == result.loops.end()) {
        auto loop = mlir::cast<mlir::scf::ForOp>(
            mlir::cast<mlir::BlockArgument>(grid->induction)
                .getOwner()
                ->getParentOp());
        result.loops.push_back(
            {grid->induction,
             *mlir::getConstantIntValue(loop.getStep()),
             {*mlir::getConstantIntValue(loop.getLowerBound()),
              *mlir::getConstantIntValue(loop.getUpperBound())}});
      }
    }
    if (first < 0 || last > extent || size > extent - last)
      return fail(TensorAssemblyStatus::Unsupported,
                  "assembly read family is outside its source");
    firstOffsets.push_back(first);
    grids.push_back(std::move(grid));
    loopIndices.push_back(loopIndex);
  }
  if (result.loops.size() > limits.maxVariables)
    return fail(TensorAssemblyStatus::ResourceExhausted,
                "assembly read exceeds its loop variable budget");

  uint64_t work = 0;
  auto charge = [&]() { return ++work <= limits.maxConstraintWork; };
  llvm::DenseMap<mlir::OpOperand *, mlir::AffineMap> sourceMaps;
  // Only actual insert boundaries can change last-writer or overlap shape.
  // A boundary strictly inside a moving window changes a static piece size;
  // isolate those bounded crossing instances. No other iteration is unrolled.
  auto current = value;
  while (auto insert =
             current.getDefiningOp<mlir::SubsetInsertionOpInterface>()) {
    if (!charge())
      return fail(TensorAssemblyStatus::ResourceExhausted,
                  "assembly read exceeded its SSA traversal budget");
    auto indexing =
        deriveTensorResultIndexing(mlir::cast<mlir::OpResult>(current), limits);
    if (!indexing.isExact())
      return fail(assemblyStatus(indexing.status), indexing.detail);
    const TensorOperandIndexing *source = nullptr;
    for (const auto &operand : indexing.indexing->operands)
      if (operand.role == TensorIndexingOperandRole::Source)
        source = &operand;
    if (!source || source->offsets.size() != sizes.size() ||
        source->sizes.size() != sizes.size())
      return fail(TensorAssemblyStatus::BrokenContract,
                  "insert source has no current rectangular indexing");
    auto sourceMap =
        source->resultToOperand.getProjectedAffineMap(value.getContext());
    if (!sourceMap)
      return fail(TensorAssemblyStatus::Unsupported,
                  "insert source has no affine coordinate projection");
    sourceMaps[&insert->getOpOperand(source->operand)] = *sourceMap;
    for (auto [axis, grid] : llvm::enumerate(grids)) {
      if (!grid)
        continue;
      auto &loop = result.loops[loopIndices[axis]];
      auto addBoundary = [&](int64_t coordinate) {
        int64_t index = loop.boundaries[0] +
                        (coordinate - grid->lower) / grid->step * loop.step;
        if (index > loop.boundaries[0] && index < loop.boundaries[1] &&
            !llvm::is_contained(loop.boundaries, index))
          loop.boundaries.push_back(index);
        return loop.boundaries.size() <= limits.maxRectangularPieces;
      };
      for (int64_t boundary : {source->offsets[axis],
                               source->offsets[axis] + source->sizes[axis]}) {
        // Coordinates are nonnegative and sizes fit the verified source.
        auto crossing =
            getTensorLoopGridCeil(*grid, boundary - sizes[axis] + 1);
        while (crossing && *crossing < boundary) {
          if (!charge() || !addBoundary(*crossing))
            return fail(TensorAssemblyStatus::ResourceExhausted,
                        "assembly read exceeded its boundary partition budget");
          int64_t next = 0;
          if (llvm::AddOverflow(*crossing, grid->step, next))
            return fail(TensorAssemblyStatus::ResourceExhausted,
                        "assembly boundary coordinate overflow");
          if (next >= grid->upper)
            break;
          if (!addBoundary(next))
            return fail(TensorAssemblyStatus::ResourceExhausted,
                        "assembly read exceeded its boundary partition budget");
          crossing = next;
        }
        if (auto after = getTensorLoopGridCeil(*grid, boundary))
          if (!addBoundary(*after))
            return fail(TensorAssemblyStatus::ResourceExhausted,
                        "assembly read exceeded its boundary partition budget");
      }
    }
    current = insert.getDestinationOperand().get();
  }
  uint64_t caseCount = 1;
  for (auto &loop : result.loops) {
    llvm::sort(loop.boundaries);
    auto count = loop.boundaries.size() - 1;
    if (caseCount > limits.maxRectangularPieces / count)
      return fail(TensorAssemblyStatus::ResourceExhausted,
                  "assembly read exceeded its partition product budget");
    caseCount *= count;
  }
  if (caseCount > limits.maxRectangularPieces)
    return fail(TensorAssemblyStatus::ResourceExhausted,
                "assembly read exceeded its partition product budget");
  uint64_t pieceCount = 0;
  auto *context = value.getContext();
  for (uint64_t ordinal = 0; ordinal < caseCount; ++ordinal) {
    if (!charge())
      return fail(TensorAssemblyStatus::ResourceExhausted,
                  "assembly read exceeded its case query budget");
    TensorAssemblyReadCase part;
    uint64_t remaining = ordinal;
    for (const auto &loop : result.loops) {
      auto interval = remaining % (loop.boundaries.size() - 1);
      remaining /= loop.boundaries.size() - 1;
      part.lowerBounds.push_back(loop.boundaries[interval]);
      part.upperBounds.push_back(loop.boundaries[interval + 1]);
    }
    StaticRectangularIndexSet requested;
    requested.offsets = firstOffsets;
    requested.sizes.assign(sizes.begin(), sizes.end());
    for (auto [axis, grid] : llvm::enumerate(grids))
      if (grid)
        requested.offsets[axis] =
            grid->lower + (part.lowerBounds[loopIndices[axis]] -
                           result.loops[loopIndices[axis]].boundaries.front()) /
                              result.loops[loopIndices[axis]].step * grid->step;
    auto demand = queryTensorAssemblyDemand(value, requested, limits);
    if (!demand.isExact())
      return fail(demand.status, demand.detail);
    for (const auto &piece : demand.pieces) {
      if (++pieceCount > limits.maxRectangularPieces || !charge())
        return fail(TensorAssemblyStatus::ResourceExhausted,
                    "assembly read exceeded its generated piece budget");
      TensorAssemblyReadPiece output;
      output.source = piece.source;
      output.sourceSizes = piece.sourceWindow.sizes;
      llvm::SmallVector<mlir::AffineExpr, 4> expressions;
      for (unsigned axis = 0; axis < sizes.size(); ++axis) {
        int64_t local =
            piece.resultWindow.offsets[axis] - requested.offsets[axis];
        output.resultWindow.offsets.push_back(local);
        output.resultWindow.sizes.push_back(piece.resultWindow.sizes[axis]);
        bool moving =
            grids[axis] && part.upperBounds[loopIndices[axis]] -
                                   part.lowerBounds[loopIndices[axis]] >
                               result.loops[loopIndices[axis]].step;
        if (moving &&
            (local != 0 || piece.resultWindow.sizes[axis] != sizes[axis]))
          return fail(TensorAssemblyStatus::BrokenContract,
                      "assembly boundary partition has a varying local piece");
        auto expression = mlir::getAffineConstantExpr(
            piece.resultWindow.offsets[axis], context);
        if (moving) {
          auto delta = mlir::getAffineDimExpr(loopIndices[axis], context) -
                       part.lowerBounds[loopIndices[axis]];
          int64_t inductionStep = result.loops[loopIndices[axis]].step;
          // All executions are on this actual loop grid. Preserve an integral
          // linear form when available; downstream need not rediscover an
          // avoidable div/mul cancellation.
          expression =
              expression +
              (grids[axis]->step % inductionStep == 0
                   ? delta * (grids[axis]->step / inductionStep)
                   : delta.floorDiv(inductionStep) * grids[axis]->step);
        }
        expressions.push_back(expression);
      }
      auto resultOffsets =
          mlir::AffineMap::get(result.loops.size(), 0, expressions, context);
      // Preserve the original insert's coordinate projection. A clipped kept
      // dimension can become unit-sized; recomputing rank reduction from that
      // tile shape would then silently drop the wrong dynamic coordinate.
      auto sourceMap =
          piece.sourceOperand
              ? sourceMaps.lookup(piece.sourceOperand)
              : mlir::AffineMap::getMultiDimIdentityMap(sizes.size(), context);
      if (!sourceMap)
        return fail(TensorAssemblyStatus::BrokenContract,
                    "assembly piece lost its current operand projection");
      output.sourceOffsets =
          mlir::simplifyAffineMap(sourceMap.compose(resultOffsets));
      part.pieces.push_back(std::move(output));
    }
    result.cases.push_back(std::move(part));
  }
  return result;
}
} // namespace

TensorAssemblyReadResult queryTensorAssemblyRead(
    mlir::Value value, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<int64_t> sizes, const IndexRelationLimits &limits) {
  auto type = value ? mlir::dyn_cast<mlir::RankedTensorType>(value.getType())
                    : mlir::RankedTensorType{};
  if (!type || !type.hasStaticShape() || offsets.size() != sizes.size() ||
      offsets.size() != static_cast<size_t>(type.getRank()))
    return fail(TensorAssemblyStatus::BrokenContract,
                "assembly read has inconsistent static dimensions");
  mlir::Value assembly = value;
  std::optional<TensorViewIndexing> view;
  if (!assembly.getDefiningOp<mlir::SubsetInsertionOpInterface>()) {
    auto *definition = assembly.getDefiningOp();
    if (!definition || !mlir::isa<WaferTensorIndexingOpInterface>(definition))
      return fail(TensorAssemblyStatus::NotAssembly,
                  "read source is an opaque tensor leaf");
    auto derived = deriveTensorViewIndexing(value, limits);
    if (!derived.isExact())
      return fail(assemblyStatus(derived.status), derived.detail);
    assembly = derived.indexing->source;
    if (!assembly.getDefiningOp<mlir::SubsetInsertionOpInterface>())
      return fail(TensorAssemblyStatus::NotAssembly,
                  "read has no current assembly source");
    view = std::move(*derived.indexing);
  }
  llvm::SmallVector<ReadCoordinate, 4> coordinates;
  llvm::SmallVector<TensorLoopGrid, 4> loops;
  llvm::SmallVector<unsigned, 4> loopIndices;
  StaticRectangularIndexSet first;
  first.sizes.assign(sizes.begin(), sizes.end());
  llvm::SmallVector<int64_t, 4> span;
  for (auto [offset, size, extent] :
       llvm::zip_equal(offsets, sizes, type.getShape())) {
    ReadCoordinate coordinate;
    unsigned loopIndex = 0;
    int64_t last = 0;
    if (auto constant = mlir::getConstantIntValue(offset)) {
      coordinate.first = last = *constant;
    } else {
      auto grid = queryTensorLoopGrid(offset, limits);
      if (grid.status != TensorResultIndexingStatus::Exact)
        return fail(assemblyStatus(grid.status), grid.detail);
      coordinate.grid = std::move(grid.grid);
      coordinate.first = coordinate.grid->lower;
      last =
          *getTensorLoopGridFloor(*coordinate.grid, coordinate.grid->upper - 1);
      auto found = llvm::find_if(loops, [&](const auto &loop) {
        return loop.induction == coordinate.grid->induction;
      });
      loopIndex = std::distance(loops.begin(), found);
      if (found == loops.end())
        loops.push_back(*coordinate.grid);
    }
    if (size <= 0 || coordinate.first < 0 || last > extent ||
        size > extent - last)
      return fail(TensorAssemblyStatus::Unsupported,
                  "assembly read family is outside its source");
    first.offsets.push_back(coordinate.first);
    span.push_back(last - coordinate.first + size);
    loopIndices.push_back(loopIndex);
    coordinates.push_back(std::move(coordinate));
  }
  if (!view)
    return queryAssemblyRead(assembly, coordinates, sizes, limits);
  auto fromRelation = [&](IndexRelationStatus status, llvm::StringRef reason) {
    return fail(status == IndexRelationStatus::ResourceExhausted
                    ? TensorAssemblyStatus::ResourceExhausted
                : status == IndexRelationStatus::Invalid
                    ? TensorAssemblyStatus::BrokenContract
                    : TensorAssemblyStatus::Unsupported,
                reason);
  };
  if (loops.empty()) {
    auto source =
        getTensorViewTileSource(*view, type.getShape(), first, limits);
    if (!source.isExact())
      return fromRelation(source.status, source.reason);
    coordinates.clear();
    for (int64_t offset : source.domain->offsets)
      coordinates.push_back({offset, std::nullopt});
    return queryAssemblyRead(assembly, coordinates, source.domain->sizes,
                             limits);
  }
  // The existing family proof is defined on a zero-origin tile grid. Translate
  // its domain to this current read's first origin; holes remain a subset of
  // that proved grid and are never passed to the last-writer demand query.
  for (auto [coordinate, size] : llvm::zip_equal(coordinates, sizes))
    if (coordinate.grid && coordinate.grid->step % size != 0)
      return fail(TensorAssemblyStatus::Unsupported,
                  "view read needs an aligned rectangular tile family");
  llvm::SmallVector<int64_t, 4> strides(type.getRank(), 1);
  auto shifted = IndexRelation::staticSlice(span, type.getShape(),
                                            first.offsets, strides, limits);
  if (!shifted.isExact())
    return fromRelation(shifted.status, shifted.reason);
  shifted = shifted.get()->compose(view->resultToSource, limits);
  if (!shifted.isExact())
    return fromRelation(shifted.status, shifted.reason);
  auto sourceType = mlir::cast<mlir::RankedTensorType>(assembly.getType());
  auto *context = value.getContext();
  auto family = shifted.get()->getRectangularTileImage(
      context, span, sourceType.getShape(), sizes, limits);
  if (!family.isExact())
    return fromRelation(family.status, family.reason);
  auto constant = [&](int64_t n) {
    return mlir::getAffineConstantExpr(n, context);
  };
  llvm::SmallVector<mlir::AffineExpr, 8> parameters;
  for (auto [axis, coordinate] : llvm::enumerate(coordinates))
    parameters.push_back(
        coordinate.grid ? mlir::getAffineDimExpr(loopIndices[axis], context) *
                              coordinate.grid->step
                        : constant(0));
  for (int64_t size : sizes)
    parameters.push_back(constant(size));
  llvm::SmallVector<mlir::AffineExpr, 4> origins;
  llvm::SmallVector<int64_t, 4> sourceSizes;
  for (unsigned axis = 0; axis < sourceType.getRank(); ++axis) {
    origins.push_back(mlir::simplifyAffineExpr(
        family.image->offsetMap.getResult(axis).replaceDimsAndSymbols(
            parameters, {}),
        loops.size(), 0));
    auto size =
        mlir::dyn_cast<mlir::AffineConstantExpr>(mlir::simplifyAffineExpr(
            family.image->sizeMap.getResult(axis).replaceDimsAndSymbols(
                llvm::ArrayRef<mlir::AffineExpr>(parameters)
                    .drop_front(sizes.size()),
                {}),
            0, 0));
    if (!size || size.getValue() <= 0)
      return fail(TensorAssemblyStatus::Unsupported,
                  "view family has no static compact source shape");
    sourceSizes.push_back(size.getValue());
  }
  // Prove local row-major order for the whole actual loop family, including
  // correlated coordinates. Matching one example or just volume is
  // insufficient.
  llvm::SmallVector<int64_t, 8> domain;
  for (const auto &loop : loops)
    domain.push_back((loop.upper - loop.lower - 1) / loop.step + 1);
  domain.append(sizes.begin(), sizes.end());
  llvm::SmallVector<mlir::AffineExpr, 4> actualMap, expectedMap;
  auto linear = constant(0);
  for (unsigned axis = 0; axis < sizes.size(); ++axis) {
    auto local = mlir::getAffineDimExpr(loops.size() + axis, context);
    actualMap.push_back(parameters[axis] + first.offsets[axis] + local);
    linear = linear * sizes[axis] + local;
  }
  int64_t trailing = 1;
  expectedMap.resize(sourceSizes.size());
  for (unsigned axis = sourceSizes.size(); axis-- > 0;) {
    expectedMap[axis] =
        origins[axis] + linear.floorDiv(trailing) % sourceSizes[axis];
    if (llvm::MulOverflow(trailing, sourceSizes[axis], trailing))
      return fail(TensorAssemblyStatus::ResourceExhausted,
                  "view compact coordinate volume overflow");
  }
  auto actual = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(domain.size(), 0, actualMap, context), domain,
      type.getShape(), limits);
  if (actual.isExact())
    actual = actual.get()->compose(view->resultToSource, limits);
  auto expected = IndexRelation::fromAffineMap(
      mlir::AffineMap::get(domain.size(), 0, expectedMap, context), domain,
      sourceType.getShape(), limits);
  for (const auto *relation : {&actual, &expected})
    if (!relation->isExact())
      return fromRelation(relation->status, relation->reason);
  auto order = actual.get()->isEquivalentTo(*expected.get(), limits);
  if (!order.isProvenTrue())
    return fromRelation(order.status,
                        "view family does not prove compact row-major order");

  coordinates.clear();
  llvm::SmallVector<mlir::AffineExpr, 4> zeros(loops.size(), constant(0));
  for (auto origin : origins) {
    auto atZero =
        mlir::dyn_cast<mlir::AffineConstantExpr>(mlir::simplifyAffineExpr(
            origin.replaceDimsAndSymbols(zeros, {}), 0, 0));
    if (!atZero)
      return fail(TensorAssemblyStatus::Unsupported,
                  "view source origin is not a bounded coordinate");
    ReadCoordinate coordinate{atZero.getValue(), std::nullopt};
    auto reconstructed = constant(coordinate.first);
    for (unsigned i = 0; i < loops.size(); ++i) {
      auto point = zeros;
      point[i] = constant(1);
      auto atOne =
          mlir::dyn_cast<mlir::AffineConstantExpr>(mlir::simplifyAffineExpr(
              origin.replaceDimsAndSymbols(point, {}), 0, 0));
      if (!atOne)
        return fail(TensorAssemblyStatus::Unsupported,
                    "view source origin is not a linear loop coordinate");
      int64_t step = atOne.getValue() - coordinate.first;
      if (step == 0)
        continue;
      if (step < 0 || coordinate.grid)
        return fail(TensorAssemblyStatus::Unsupported,
                    "view source coordinate requires a coupled loop partition");
      int64_t end = 0;
      if (llvm::MulOverflow(domain[i], step, end) ||
          llvm::AddOverflow(coordinate.first, end, end))
        return fail(TensorAssemblyStatus::ResourceExhausted,
                    "view source coordinate overflow");
      coordinate.grid = TensorLoopGrid{
          {}, loops[i].induction, 0, 1, coordinate.first, end, step};
      reconstructed = reconstructed + mlir::getAffineDimExpr(i, context) * step;
    }
    if (mlir::simplifyAffineExpr(origin - reconstructed, loops.size(), 0) !=
        constant(0))
      return fail(TensorAssemblyStatus::Unsupported,
                  "view source coordinate requires a periodic loop partition");
    coordinates.push_back(std::move(coordinate));
  }
  return queryAssemblyRead(assembly, coordinates, sourceSizes, limits);
}

} // namespace wafer::analysis
