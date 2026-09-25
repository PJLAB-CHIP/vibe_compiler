//===- TensorAssemblyRead.cpp - Bounded current assembly accesses --------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

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
queryStaticViewRead(const TensorViewIndexing &view,
                    llvm::ArrayRef<int64_t> resultShape,
                    const StaticRectangularIndexSet &requested,
                    const IndexRelationLimits &limits) {
  auto fromRelation = [&](IndexRelationStatus status, llvm::StringRef reason) {
    return fail(status == IndexRelationStatus::ResourceExhausted
                    ? TensorAssemblyStatus::ResourceExhausted
                : status == IndexRelationStatus::Invalid
                    ? TensorAssemblyStatus::BrokenContract
                    : TensorAssemblyStatus::Unsupported,
                reason);
  };
  auto images = view.resultToSource.getExactStaticRectangularImagePieces(
      requested.offsets, requested.sizes, limits);
  if (!images.isExact())
    return fromRelation(images.status, "view source image: " + images.reason);
  TensorAssemblyReadResult result;
  result.status = TensorAssemblyStatus::Exact;
  result.shape = requested.sizes;
  result.cases.emplace_back();
  auto &pieces = result.cases.front().pieces;
  uint64_t work = 0;
  int64_t covered = 0, volume = 1;
  for (int64_t size : requested.sizes)
    if (llvm::MulOverflow(volume, size, volume))
      return fail(TensorAssemblyStatus::ResourceExhausted,
                  "view read volume overflow");
  for (const auto &image : images.domains) {
    if (++work > limits.maxConstraintWork)
      return fail(TensorAssemblyStatus::ResourceExhausted,
                  "view read exceeded its piece query budget");
    auto demand = queryTensorAssemblyDemand(view.source, image, limits);
    if (!demand.isExact())
      return fail(demand.status, demand.detail);
    for (const auto &piece : demand.pieces) {
      if (++work > limits.maxConstraintWork ||
          pieces.size() >= limits.maxRectangularPieces)
        return fail(TensorAssemblyStatus::ResourceExhausted,
                    "view read exceeded its piece query budget");
      auto query = getTensorViewTilePiece(view, resultShape, requested,
                                          piece.resultWindow, limits);
      if (!query.isExact()) {
        std::string detail = "view result piece: " + query.reason;
        llvm::raw_string_ostream stream(detail);
        stream << "; source offsets=[";
        llvm::interleaveComma(piece.resultWindow.offsets, stream);
        stream << "], sizes=[";
        llvm::interleaveComma(piece.resultWindow.sizes, stream);
        stream << "]";
        return fromRelation(query.status, detail);
      }
      if (!query.piece ||
          query.piece->source.offsets != piece.resultWindow.offsets ||
          query.piece->source.sizes != piece.resultWindow.sizes)
        return fail(TensorAssemblyStatus::BrokenContract,
                    "view piece lost part of its exact assembly demand");
      auto window = std::move(query.piece->result);
      int64_t pieceVolume = 1;
      for (unsigned axis = 0; axis < window.offsets.size(); ++axis) {
        window.offsets[axis] -= requested.offsets[axis];
        if (llvm::MulOverflow(pieceVolume, window.sizes[axis], pieceVolume))
          return fail(TensorAssemblyStatus::ResourceExhausted,
                      "view read piece volume overflow");
      }
      for (const auto &previous : pieces) {
        bool disjoint = false;
        for (unsigned axis = 0; axis < window.offsets.size(); ++axis) {
          if (++work > limits.maxConstraintWork)
            return fail(TensorAssemblyStatus::ResourceExhausted,
                        "view read exceeded its coverage proof budget");
          disjoint |= window.offsets[axis] + window.sizes[axis] <=
                          previous.resultWindow.offsets[axis] ||
                      previous.resultWindow.offsets[axis] +
                              previous.resultWindow.sizes[axis] <=
                          window.offsets[axis];
        }
        if (!disjoint)
          return fail(TensorAssemblyStatus::BrokenContract,
                      "view read has overlapping result pieces");
      }
      if (llvm::AddOverflow(covered, pieceVolume, covered))
        return fail(TensorAssemblyStatus::ResourceExhausted,
                    "view read coverage volume overflow");
      llvm::SmallVector<mlir::AffineExpr, 4> offsets;
      for (int64_t offset : piece.sourceWindow.offsets)
        offsets.push_back(
            mlir::getAffineConstantExpr(offset, view.source.getContext()));
      pieces.push_back(
          {piece.source,
           mlir::AffineMap::get(0, 0, offsets, view.source.getContext()),
           piece.sourceWindow.sizes, std::move(window)});
    }
  }
  if (covered != volume)
    return fail(TensorAssemblyStatus::BrokenContract,
                "view read does not cover its complete result");
  return result;
}

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
    if (source.status == IndexRelationStatus::Unsupported)
      return queryStaticViewRead(*view, type.getShape(), first, limits);
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

TensorSubsetDemandResult
queryTensorSubsetDemand(mlir::Value value,
                        llvm::ArrayRef<mlir::OpFoldResult> offsets,
                        llvm::ArrayRef<int64_t> sizes, mlir::Operation *scope,
                        IndexRelationWork &work) {
  auto reject = [](TensorAssemblyStatus status, llvm::StringRef detail) {
    return TensorSubsetDemandResult{status, std::nullopt, detail.str()};
  };
  auto relationFailure = [&](IndexRelationStatus status,
                             llvm::StringRef detail) {
    return reject(status == IndexRelationStatus::ResourceExhausted
                      ? TensorAssemblyStatus::ResourceExhausted
                  : status == IndexRelationStatus::Invalid
                      ? TensorAssemblyStatus::BrokenContract
                      : TensorAssemblyStatus::Unsupported,
                  detail);
  };
  auto exhausted = [&] {
    return reject(TensorAssemblyStatus::ResourceExhausted,
                  "tensor subset exceeded the cumulative request budget");
  };
  auto type = value ? mlir::dyn_cast<mlir::RankedTensorType>(value.getType())
                    : mlir::RankedTensorType{};
  if (!type || !type.hasStaticShape() || offsets.size() != sizes.size() ||
      sizes.size() != static_cast<size_t>(type.getRank()) ||
      llvm::any_of(sizes, [](int64_t size) { return size < 0; }))
    return reject(TensorAssemblyStatus::BrokenContract,
                  "tensor subset has inconsistent static dimensions");
  auto *context = value.getContext();
  if (!scope || scope->getContext() != context)
    return reject(TensorAssemblyStatus::BrokenContract,
                  "tensor subset requires its actual consumer scope");
  struct ScopeComparison {
    mlir::arith::CmpIPredicate predicate;
    unsigned lhs;
    unsigned rhs;
    bool complement;
  };
  llvm::SmallVector<ScopeComparison> comparisons;
  llvm::SmallVector<mlir::OpFoldResult> indexInputs(offsets);
  bool emptyScope = false;
  for (auto *child = scope; child->getParentOp();
       child = child->getParentOp()) {
    auto *parent = child->getParentOp();
    if (!work.charge())
      return exhausted();
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
      indexInputs.push_back(loop.getInductionVar());
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent)) {
      bool taken = child->getParentRegion() == &branch.getThenRegion();
      if (auto constant = mlir::getConstantIntValue(branch.getCondition())) {
        emptyScope |= ((*constant != 0) != taken);
        continue;
      }
      auto compare = branch.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
      if (!compare)
        return reject(
            TensorAssemblyStatus::Unsupported,
            "tensor subset scope has an unsupported branch condition");
      unsigned lhs = indexInputs.size();
      indexInputs.push_back(compare.getLhs());
      indexInputs.push_back(compare.getRhs());
      comparisons.push_back({compare.getPredicate(), lhs, lhs + 1, !taken});
    }
    if (parent->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      break;
  }
  auto indices = queryTensorIndexExpressions(context, indexInputs, work);
  if (!indices.isExact())
    return reject(assemblyStatus(indices.status), indices.detail);
  TensorSubsetDemand demand;
  demand.parameters = std::move(indices.expressions->parameters);
  demand.shape.assign(sizes.begin(), sizes.end());
  unsigned parameters = demand.parameters.size();
  unsigned dimensions = parameters + sizes.size();
  if (dimensions > work.getLimits().maxVariables)
    return exhausted();
  auto dim = [&](unsigned i) { return mlir::getAffineDimExpr(i, context); };
  llvm::SmallVector<mlir::AffineExpr> domain, coordinates;
  llvm::SmallVector<bool> equalities;
  auto bound = [&](mlir::AffineExpr expr) {
    domain.push_back(expr);
    equalities.push_back(false);
  };
  // Express actual IVs using bounded iteration counters during proof. This
  // retains every loop step without relaxing the domain to a continuous grid.
  llvm::SmallVector<mlir::AffineExpr> proofParameters;
  bool empty = emptyScope || llvm::is_contained(sizes, 0);
  for (auto [i, parameter] : llvm::enumerate(demand.parameters)) {
    empty |= parameter.upper <= parameter.lower;
    bound(dim(i) - parameter.lower);
    bound(parameter.upper - 1 - dim(i));
    domain.push_back((dim(i) - parameter.lower) % parameter.step);
    equalities.push_back(true);
    proofParameters.push_back(dim(i) * parameter.step + parameter.lower);
  }
  for (unsigned i = 0; i < sizes.size(); ++i) {
    bound(dim(parameters + i));
    bound(sizes[i] - 1 - dim(parameters + i));
    coordinates.push_back(indices.expressions->map.getResult(i) +
                          dim(parameters + i));
    proofParameters.push_back(dim(parameters + i));
  }
  if (domain.empty()) {
    domain.push_back(mlir::getAffineConstantExpr(0, context));
    equalities.push_back(true);
  }
  demand.domain = mlir::IntegerSet::get(dimensions, 0, domain, equalities);
  for (auto comparison : comparisons) {
    auto lhs = indices.expressions->map.getResult(comparison.lhs);
    auto rhs = indices.expressions->map.getResult(comparison.rhs);
    mlir::AffineExpr constraint;
    bool equality = false;
    using P = mlir::arith::CmpIPredicate;
    switch (comparison.predicate) {
    case P::eq:
      constraint = lhs - rhs;
      equality = true;
      break;
    case P::ne:
      constraint = lhs - rhs;
      equality = true;
      comparison.complement = !comparison.complement;
      break;
    case P::slt:
      constraint = rhs - lhs - 1;
      break;
    case P::sle:
      constraint = rhs - lhs;
      break;
    case P::sgt:
      constraint = lhs - rhs - 1;
      break;
    case P::sge:
      constraint = lhs - rhs;
      break;
    default:
      return reject(TensorAssemblyStatus::Unsupported,
                    "tensor subset scope needs a signed index comparison");
    }
    demand.scopeConditions.push_back(
        {mlir::IntegerSet::get(dimensions, 0, {constraint}, {equality}),
         comparison.complement});
  }
  if (empty)
    return {TensorAssemblyStatus::Exact, std::move(demand), {}};
  auto rootCoordinates =
      mlir::AffineMap::get(dimensions, 0, coordinates, context);
  llvm::SmallVector<ClosedIndexInterval> coordinateBounds;
  for (auto parameter : demand.parameters)
    coordinateBounds.push_back({parameter.lower, parameter.upper - 1});
  for (auto size : sizes)
    coordinateBounds.push_back({0, size - 1});
  llvm::DenseMap<mlir::AffineExpr, mlir::AffineExpr> boundedExpressions;
  IndexRelationStatus boundedStatus = IndexRelationStatus::Exact;
  std::string boundedReason;
  // Simplify only using the actual request's enclosing box. Scope guards
  // and last-writer paths can narrow it further, but are never assumed here.
  // Reusing one cache across the entire DAG avoids repeating the same
  // quotient proof at each insertion in a fragment assembly.
  std::function<mlir::AffineExpr(mlir::AffineExpr)> foldBounded =
      [&](mlir::AffineExpr expression) -> mlir::AffineExpr {
    if (!work.charge()) {
      boundedStatus = IndexRelationStatus::ResourceExhausted;
      boundedReason = "bounded subset expression exceeded request budget";
      return {};
    }
    if (auto found = boundedExpressions.find(expression);
        found != boundedExpressions.end())
      return found->second;
    auto result = expression;
    if (auto axis = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
      auto bound = coordinateBounds[axis.getPosition()];
      if (bound.minimum == bound.maximum)
        result = mlir::getAffineConstantExpr(bound.minimum, context);
    } else if (auto binary =
                   mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
      auto lhs = foldBounded(binary.getLHS());
      auto rhs = foldBounded(binary.getRHS());
      if (!lhs || !rhs)
        return {};
      auto composed = composeIndexMap(
          mlir::AffineMap::get(2, 0, mlir::getAffineBinaryOpExpr(
                                         expression.getKind(), dim(0),
                                         expression.getKind() == mlir::AffineExprKind::Add ? dim(1) : rhs)),
          mlir::AffineMap::get(dimensions, 0, {lhs, rhs}, context), work);
      if (!composed.isExact()) {
        boundedStatus = composed.status;
        boundedReason = composed.reason;
        return {};
      }
      result = composed.map.getResult(0);
      if (expression.getKind() == mlir::AffineExprKind::FloorDiv ||
          expression.getKind() == mlir::AffineExprKind::CeilDiv ||
          expression.getKind() == mlir::AffineExprKind::Mod) {
        auto divisor = mlir::dyn_cast<mlir::AffineConstantExpr>(rhs);
        if (!divisor || divisor.getValue() <= 0) {
          boundedStatus = IndexRelationStatus::Unsupported;
          boundedReason = "bounded subset requires constant positive divisors";
          return {};
        }
        auto bound = boundIndexExpression(
            mlir::AffineMap::get(dimensions, 0, lhs), coordinateBounds, work);
        if (bound.status == IndexRelationStatus::ResourceExhausted ||
            bound.status == IndexRelationStatus::Invalid) {
          boundedStatus = bound.status;
          boundedReason = bound.reason;
          return {};
        }
        if (bound.interval) {
          auto divide = [&](int64_t value) {
            return expression.getKind() == mlir::AffineExprKind::CeilDiv
                       ? llvm::divideCeilSigned(value, divisor.getValue())
                       : llvm::divideFloorSigned(value, divisor.getValue());
          };
          int64_t lower = divide(bound.interval->minimum);
          if (lower == divide(bound.interval->maximum)) {
            if (expression.getKind() != mlir::AffineExprKind::Mod) {
              result = mlir::getAffineConstantExpr(lower, context);
            } else {
              int64_t offset;
              if (llvm::MulOverflow(lower, divisor.getValue(), offset) || offset == INT64_MIN) {
                boundedStatus = IndexRelationStatus::ResourceExhausted;
                boundedReason = "bounded subset remainder offset overflow";
                return {};
              }
              auto translated = composeIndexMap(
                  mlir::AffineMap::get(1, 0, dim(0) - offset),
                  mlir::AffineMap::get(dimensions, 0, lhs), work);
              if (!translated.isExact()) {
                boundedStatus = translated.status;
                boundedReason = translated.reason;
                return {};
              }
              result = translated.map.getResult(0);
            }
          }
        }
      }
    }
    boundedExpressions.try_emplace(expression, result);
    return result;
  };
  auto foldBoundedMap = [&](mlir::AffineMap map) -> IndexMapResult {
    llvm::SmallVector<mlir::AffineExpr> expressions;
    for (auto expression : map.getResults()) {
      auto folded = foldBounded(expression);
      if (!folded)
        return {boundedStatus, {}, boundedReason};
      expressions.push_back(folded);
    }
    return {IndexRelationStatus::Exact,
            mlir::AffineMap::get(dimensions, 0, expressions, context), {}};
  };
  struct DomainComposition {
    IndexRelationStatus status;
    mlir::IntegerSet domain;
    std::string reason;
  };
  auto composeSet = [&](mlir::IntegerSet set,
                        mlir::AffineMap map, bool fold = true) -> DomainComposition {
    auto composed = composeIndexMap(
        mlir::AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                             set.getConstraints(), context),
        map, work);
    if (!composed.isExact())
      return {composed.status, {}, composed.reason};
    // The proof substitution below changes IV dimensions into iteration
    // counters. Only the original request coordinate system uses this box.
    if (fold) {
      composed = foldBoundedMap(composed.map);
      if (!composed.isExact())
        return {composed.status, {}, composed.reason};
    }
    return {IndexRelationStatus::Exact,
            mlir::IntegerSet::get(dimensions, 0, composed.map.getResults(),
                                  set.getEqFlags()),
            {}};
  };
  auto proofMap = mlir::AffineMap::get(dimensions, 0, proofParameters, context);
  auto emptyPath = [&](llvm::ArrayRef<IndexDomainCondition> path)
      -> IndexRelationQueryResult {
    llvm::SmallVector<IndexDomainCondition> proof;
    llvm::SmallVector<IndexDomainCondition> conditions{{demand.domain, false}};
    conditions.append(demand.scopeConditions);
    conditions.append(path.begin(), path.end());
    for (auto condition : conditions) {
      auto composed = composeSet(condition.set, proofMap, false);
      if (composed.status != IndexRelationStatus::Exact)
        return {composed.status, std::nullopt, composed.reason};
      proof.push_back({composed.domain, condition.complement});
    }
    return proveIndexDomainEmpty(proof, work);
  };
  // A verifier-valid dynamic slice still needs a bounds proof before it can
  // authorize newly constructed source accesses.
  llvm::SmallVector<mlir::AffineExpr> bounds;
  llvm::SmallVector<bool> boundEqualities;
  for (unsigned i = 0; i < sizes.size(); ++i) {
    bounds.push_back(coordinates[i]);
    bounds.push_back(type.getDimSize(i) - 1 - coordinates[i]);
    boundEqualities.append(2, false);
  }
  if (!bounds.empty()) {
    auto set = mlir::IntegerSet::get(dimensions, 0, bounds, boundEqualities);
    auto valid = emptyPath({IndexDomainCondition{set, true}});
    if (!valid.isProvenTrue())
      return relationFailure(valid.status,
                             "tensor subset bounds: " + valid.reason);
  }
  struct Pending {
    mlir::Value value;
    mlir::AffineMap coordinates;
    llvm::SmallVector<IndexDomainCondition, 4> conditions;
  };
  llvm::SmallVector<Pending> pending{{value, rootCoordinates, {}}}, visited;
  llvm::DenseMap<mlir::Value, TensorResultIndexingResult> indexingCache;
  while (!pending.empty()) {
    if (!work.charge())
      return exhausted();
    Pending current = std::move(pending.back());
    pending.pop_back();
    bool repeated = false;
    for (const auto &previous : visited) {
      if (!work.charge(1 + previous.conditions.size()))
        return exhausted();
      if (previous.value != current.value ||
          previous.coordinates != current.coordinates ||
          previous.conditions.size() != current.conditions.size())
        continue;
      repeated =
          llvm::all_of(llvm::zip_equal(previous.conditions, current.conditions),
                       [](auto pair) {
                         const auto &[a, b] = pair;
                         return a.set == b.set && a.complement == b.complement;
                       });
      if (repeated)
        break;
    }
    if (repeated)
      continue;
    visited.push_back(current);
    auto result = mlir::dyn_cast<mlir::OpResult>(current.value);
    if (current.value.getDefiningOp<mlir::tensor::EmptyOp>()) {
      auto undefined = emptyPath(current.conditions);
      if (!undefined.isProvenTrue())
        return relationFailure(
            undefined.status,
            "tensor subset cannot prove undefined destination is unread: " +
                undefined.reason);
      continue;
    }
    if (!result ||
        !mlir::isa<WaferTensorIndexingOpInterface>(result.getOwner())) {
      if (result &&
          (mlir::isa<mlir::ViewLikeOpInterface, mlir::SubsetOpInterface,
                     mlir::arith::SelectOp>(result.getOwner()) ||
           (mlir::isa<mlir::tensor::TensorDialect>(
                result.getOwner()->getDialect()) &&
            llvm::any_of(result.getOwner()->getOperandTypes(),
                         [](mlir::Type type) {
                           return mlir::isa<mlir::RankedTensorType>(type);
                         }))))
        return reject(
            TensorAssemblyStatus::Unsupported,
            "structural tensor source lacks a supported indexing relation");
      if (demand.sources.size() >= work.getLimits().maxRectangularPieces)
        return exhausted();
      demand.sources.push_back(
          {current.value, current.coordinates, std::move(current.conditions)});
      continue;
    }
    auto found = indexingCache.find(current.value);
    if (found == indexingCache.end()) {
      auto indexing = deriveTensorResultIndexing(result, work);
      if (!indexing.isExact())
        return reject(assemblyStatus(indexing.status), indexing.detail);
      found =
          indexingCache.try_emplace(current.value, std::move(indexing)).first;
    }
    const auto &indexing = *found->second.indexing;
    if (indexing.kind == TensorIndexingTransformKind::Pad)
      return reject(TensorAssemblyStatus::Unsupported,
                    "tensor padding requires its explicit preparation first");
    const TensorOperandIndexing *source = nullptr, *destination = nullptr;
    for (const auto &operand : indexing.operands)
      (operand.role == TensorIndexingOperandRole::Source ? source
                                                         : destination) =
          &operand;
    if (!source)
      return reject(TensorAssemblyStatus::BrokenContract,
                    "structural tensor result has no source relation");
    auto function = source->resultToOperand.getIndexFunction(context, work);
    if (!function.isExact())
      return relationFailure(function.status, function.reason);
    if (!work.charge(current.coordinates.getNumResults() +
                     function.function->map.getNumResults() +
                     function.function->domain.getNumConstraints()))
      return exhausted();
    // Every current coordinate tuple is in this SSA value's type bounds:
    // proved at the root, then preserved by each total view or insert branch.
    // Remove those implied box constraints before substituting a periodic
    // view. Otherwise a loose interval for x-c*floor(x/c) can hide a valid
    // whole-block copy behind an unnecessarily strong guard.
    auto currentType =
        mlir::cast<mlir::RankedTensorType>(current.value.getType());
    llvm::SmallVector<mlir::AffineExpr> sourceConstraints;
    llvm::SmallVector<bool> sourceEqualities;
    for (auto [constraint, equality] :
         llvm::zip_equal(function.function->domain.getConstraints(),
                         function.function->domain.getEqFlags())) {
      bool implied = false;
      for (unsigned axis = 0;
           !equality && axis < unsigned(currentType.getRank()); ++axis) {
        if (!work.charge())
          return exhausted();
        auto coordinate = mlir::getAffineDimExpr(axis, context);
        implied |= constraint == coordinate ||
                   constraint == currentType.getDimSize(axis) - 1 - coordinate;
      }
      if (!implied) {
        sourceConstraints.push_back(constraint);
        sourceEqualities.push_back(equality);
      }
    }
    if (sourceConstraints.empty()) {
      sourceConstraints.push_back(mlir::getAffineConstantExpr(0, context));
      sourceEqualities.push_back(true);
    }
    auto sourceDomain =
        composeSet(mlir::IntegerSet::get(currentType.getRank(), 0,
                                         sourceConstraints, sourceEqualities),
                   current.coordinates);
    if (sourceDomain.status != IndexRelationStatus::Exact)
      return relationFailure(sourceDomain.status, sourceDomain.reason);
    if (destination) {
      demand.hasDestinationUpdates = true;
      Pending old{result.getOwner()->getOperand(destination->operand),
                  current.coordinates, current.conditions};
      old.conditions.push_back({sourceDomain.domain, true});
      pending.push_back(std::move(old));
    } else if (!source->resultToOperand
                    .hasTotalBoundedAffineMapConstruction()) {
      auto invalid = current.conditions;
      invalid.push_back({sourceDomain.domain, true});
      auto valid = emptyPath(invalid);
      if (!valid.isProvenTrue())
        return relationFailure(valid.status,
                               "transparent tensor domain: " + valid.reason);
    }
    // A transparent relation is total on the current path, either by its
    // construction or the proof above. Repeating that implied domain at every
    // view would multiply complement proof work without adding information.
    if (destination)
      current.conditions.push_back({sourceDomain.domain, false});
    auto composed =
        composeIndexMap(function.function->map, current.coordinates, work);
    if (!composed.isExact())
      return relationFailure(composed.status, composed.reason);
    composed = foldBoundedMap(composed.map);
    if (!composed.isExact())
      return relationFailure(composed.status, composed.reason);
    pending.push_back({result.getOwner()->getOperand(source->operand),
                       composed.map, std::move(current.conditions)});
  }
  return {TensorAssemblyStatus::Exact, std::move(demand), {}};
}

} // namespace wafer::analysis
