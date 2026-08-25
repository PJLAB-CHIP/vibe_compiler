//===- WindowedInsertSlice.cpp - tensor.insert_slice windowing -----------===//

#include "ProducerTileFusionInternal.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <algorithm>
#include <functional>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

// Infer the iteration range [first, last] (inclusive, in induction-variable
// values) that a dynamic window offset can take when it is (a constant shift
// of) an enclosing scf.for induction variable.  A traversal wave window whose
// offset is such a value keeps a static intersection size with an inserted
// extent: the runtime overlap [max(insert, wave), min(insertEnd, wave + size))
// is constant over the whole wave, and the extreme iterations are checked to
// prove it.
struct LoopCarriedWaveRange {
  int64_t first = 0;
  int64_t last = 0;
  int64_t step = 0;
};

static std::optional<LoopCarriedWaveRange>
getLoopCarriedWaveRange(mlir::Value offset) {
  int64_t shift = 0;
  mlir::Value base = offset;
  while (mlir::Operation *def = base.getDefiningOp()) {
    if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(def)) {
      auto lhs = mlir::getConstantIntValue(addi.getLhs());
      auto rhs = mlir::getConstantIntValue(addi.getRhs());
      if (lhs && !rhs) {
        shift += *lhs;
        base = addi.getRhs();
        continue;
      }
      if (rhs && !lhs) {
        shift += *rhs;
        base = addi.getLhs();
        continue;
      }
      return std::nullopt;
    }
    if (auto subi = mlir::dyn_cast<mlir::arith::SubIOp>(def)) {
      auto rhs = mlir::getConstantIntValue(subi.getRhs());
      if (rhs) {
        shift -= *rhs;
        base = subi.getLhs();
        continue;
      }
      return std::nullopt;
    }
    if (auto muli = mlir::dyn_cast<mlir::arith::MulIOp>(def)) {
      auto lhs = mlir::getConstantIntValue(muli.getLhs());
      auto rhs = mlir::getConstantIntValue(muli.getRhs());
      if ((lhs && *lhs != 1) || (rhs && *rhs != 1))
        return std::nullopt;
      base = lhs ? muli.getRhs() : muli.getLhs();
      continue;
    }
    return std::nullopt;
  }
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(base);
  if (!blockArg)
    return std::nullopt;
  auto forOp =
      mlir::dyn_cast<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp || forOp.getInductionVar() != base)
    return std::nullopt;
  auto lb = mlir::getConstantIntValue(forOp.getLowerBound());
  auto ub = mlir::getConstantIntValue(forOp.getUpperBound());
  auto step = mlir::getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0)
    return std::nullopt;
  const int64_t first = *lb + shift;
  const int64_t last = *lb + ((*ub - 1 - *lb) / *step) * *step + shift;
  return LoopCarriedWaveRange{first, last, *step};
}

/// Returns every positive overlap extent that a fixed-size window can have
/// with a fixed insert interval while its offset traverses one arithmetic
/// progression.  The overlap is piecewise linear and changes slope only at
/// the four interval-boundary events below, so sampling the neighboring
/// progression points is complete without expanding every temporal wave.
static llvm::SmallVector<int64_t, 3>
getWaveOverlapExtents(const LoopCarriedWaveRange &wave, int64_t windowSize,
                      int64_t insertLo, int64_t insertHi) {
  llvm::SmallVector<int64_t, 3> extents;
  if (windowSize <= 0 || insertHi <= insertLo || wave.step <= 0 ||
      wave.last < wave.first)
    return extents;
  const int64_t iterationCount = (wave.last - wave.first) / wave.step + 1;
  auto floorDiv = [](int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    int64_t remainder = numerator % denominator;
    if (remainder < 0)
      --quotient;
    return quotient;
  };
  auto sample = [&](int64_t iteration) {
    if (iteration < 0 || iteration >= iterationCount)
      return;
    const int64_t windowLo = wave.first + iteration * wave.step;
    const int64_t windowHi = windowLo + windowSize;
    const int64_t overlap =
        std::min(insertHi, windowHi) - std::max(insertLo, windowLo);
    if (overlap > 0)
      extents.push_back(overlap);
  };
  sample(/*iteration=*/0);
  sample(iterationCount - 1);
  for (int64_t breakpoint :
       {insertLo - windowSize, insertLo, insertHi - windowSize, insertHi}) {
    const int64_t nearest = floorDiv(breakpoint - wave.first, wave.step);
    for (int64_t delta : {-1, 0, 1})
      sample(nearest + delta);
  }
  llvm::sort(extents);
  extents.erase(std::unique(extents.begin(), extents.end()), extents.end());
  return extents;
}

void materializeWindowedInsertSlice(
    mlir::IRRewriter &rewriter, mlir::tensor::ExtractSliceOp slice,
    mlir::tensor::InsertSliceOp insert, mlir::OpResult producerResult,
    EnqueueProducerSlices enqueueSlices,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes,
    llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
        &materializedCoupledTiles) {
  auto copyNodeMappings = [&](mlir::Operation *mapped) {
    if (!operationNodes || !mapped)
      return;
    llvm::SmallVector<StructuredOperationNodeMapping, 2> additions;
    for (const StructuredOperationNodeMapping &mapping : *operationNodes)
      if (mapping.operation == insert.getOperation())
        additions.push_back({mapped, mapping.structuredNodeId,
                             mapping.coupledComponentIndices});
    operationNodes->append(additions.begin(), additions.end());
  };
  const size_t rank = slice.getMixedOffsets().size();
  llvm::SmallVector<int64_t, 4> insertOffsets;
  llvm::SmallVector<int64_t, 4> insertSizes;
  llvm::SmallVector<int64_t, 4> insertStrides;
  llvm::SmallVector<int64_t, 4> windowOffsets;
  llvm::SmallVector<int64_t, 4> windowSizes;
  insertOffsets.reserve(rank);
  insertSizes.reserve(rank);
  insertStrides.reserve(rank);
  windowOffsets.reserve(rank);
  windowSizes.reserve(rank);
  bool insertFullyStatic = rank == insert.getMixedOffsets().size() &&
                           rank == insert.getMixedSizes().size() &&
                           rank == insert.getMixedStrides().size();
  for (size_t dimension = 0; insertFullyStatic && dimension < rank;
       ++dimension) {
    auto insertOffset =
        mlir::getConstantIntValue(insert.getMixedOffsets()[dimension]);
    auto insertSize =
        mlir::getConstantIntValue(insert.getMixedSizes()[dimension]);
    auto insertStride =
        mlir::getConstantIntValue(insert.getMixedStrides()[dimension]);
    insertFullyStatic &= insertOffset && insertSize && insertStride;
    if (!insertFullyStatic)
      break;
    insertOffsets.push_back(*insertOffset);
    insertSizes.push_back(*insertSize);
    insertStrides.push_back(*insertStride);
  }
  if (!insertFullyStatic)
    return;
  bool windowFullyStatic = rank == slice.getMixedOffsets().size() &&
                           rank == slice.getMixedSizes().size();
  for (size_t dimension = 0; windowFullyStatic && dimension < rank;
       ++dimension) {
    auto windowOffset =
        mlir::getConstantIntValue(slice.getMixedOffsets()[dimension]);
    auto windowSize =
        mlir::getConstantIntValue(slice.getMixedSizes()[dimension]);
    windowFullyStatic &= windowOffset && windowSize;
    if (!windowFullyStatic)
      break;
    windowOffsets.push_back(*windowOffset);
    windowSizes.push_back(*windowSize);
  }
  if (windowFullyStatic) {
    llvm::SmallVector<int64_t, 4> relativeOffsets(rank);
    llvm::SmallVector<int64_t, 4> intersectSizes(rank);
    bool disjoint = false;
    bool fullyCovered = true;
    for (size_t dimension = 0; dimension < rank; ++dimension) {
      const int64_t insertLo = insertOffsets[dimension];
      const int64_t insertHi = insertLo + insertSizes[dimension];
      const int64_t windowLo = windowOffsets[dimension];
      const int64_t windowHi = windowLo + windowSizes[dimension];
      const int64_t intersectLo = std::max(insertLo, windowLo);
      const int64_t intersectHi = std::min(insertHi, windowHi);
      if (intersectHi <= intersectLo) {
        disjoint = true;
        break;
      }
      relativeOffsets[dimension] = intersectLo - insertLo;
      intersectSizes[dimension] = intersectHi - intersectLo;
      fullyCovered &= intersectLo == windowLo && intersectHi == windowHi &&
                      insertStrides[dimension] == 1;
    }
    rewriter.setInsertionPoint(slice);
    llvm::SmallVector<mlir::OpFoldResult, 4> unitStrides(
        rank, rewriter.getIndexAttr(1));
    // A window fully covered by the inserted region takes every value
    // from the inserted source; the destination's contribution is dead.
    // Materializing the destination window would pull the destination's
    // structured producer (an exact overwritten demand the closure
    // already proved empty) into this scope, so use a fresh destination
    // window instead.
    mlir::Value destSlice;
    if (!disjoint && fullyCovered) {
      auto windowType = mlir::cast<mlir::RankedTensorType>(slice.getType());
      auto empty = rewriter.create<mlir::tensor::EmptyOp>(
          slice.getLoc(), windowType.getShape(), windowType.getElementType());
      destSlice = empty.getResult();
    } else {
      destSlice =
          rewriter
              .create<mlir::tensor::ExtractSliceOp>(
                  slice.getLoc(), slice.getType(), insert.getDest(),
                  slice.getMixedOffsets(), slice.getMixedSizes(), unitStrides)
              .getResult();
    }
    mlir::Value tiled = destSlice;
    if (!disjoint) {
      auto sourceType =
          mlir::cast<mlir::RankedTensorType>(insert.getSource().getType());
      auto inputSliceType = mlir::RankedTensorType::get(
          intersectSizes, sourceType.getElementType());
      llvm::SmallVector<mlir::OpFoldResult, 4> inputOffsets;
      llvm::SmallVector<mlir::OpFoldResult, 4> inputSizes;
      inputOffsets.reserve(rank);
      inputSizes.reserve(rank);
      for (size_t dimension = 0; dimension < rank; ++dimension) {
        inputOffsets.push_back(
            rewriter.getIndexAttr(relativeOffsets[dimension]));
        inputSizes.push_back(rewriter.getIndexAttr(intersectSizes[dimension]));
      }
      mlir::Value inputSlice =
          rewriter
              .create<mlir::tensor::ExtractSliceOp>(
                  slice.getLoc(), inputSliceType, insert.getSource(),
                  inputOffsets, inputSizes, unitStrides)
              .getResult();
      llvm::SmallVector<mlir::OpFoldResult, 4> windowRelativeOffsets;
      llvm::SmallVector<mlir::OpFoldResult, 4> tiledInsertSizes;
      windowRelativeOffsets.reserve(rank);
      tiledInsertSizes.reserve(rank);
      for (size_t dimension = 0; dimension < rank; ++dimension) {
        const int64_t intersectLo =
            std::max(insertOffsets[dimension], windowOffsets[dimension]);
        windowRelativeOffsets.push_back(
            rewriter.getIndexAttr(intersectLo - windowOffsets[dimension]));
        tiledInsertSizes.push_back(
            rewriter.getIndexAttr(intersectSizes[dimension]));
      }
      auto tiledInsert = rewriter.create<mlir::tensor::InsertSliceOp>(
          slice.getLoc(), inputSlice, destSlice, windowRelativeOffsets,
          tiledInsertSizes, insert.getMixedStrides());
      if (auto resource = insert->getAttrOfType<CardDDRResourceAttr>(
              kWaferCardDDRMovementAttrName))
        tiledInsert->setAttr(kWaferCardDDRMovementAttrName, resource);
      copyNodeMappings(tiledInsert);
      tiled = tiledInsert.getResult();
      // The intersecting input slice continues the fusion walk upstream
      // toward the structured producer of the inserted value.
      enqueueSlices({inputSlice.getDefiningOp()}, insert.getOperation(),
                    {tiledInsert.getOperation()});
    }
    materializedCoupledTiles.push_back(MaterializedCoupledProducerTile{
        producerResult, slice->getBlock(), slice.getType(),
        llvm::to_vector(slice.getMixedOffsets()),
        llvm::to_vector(slice.getMixedSizes()),
        llvm::to_vector(slice.getMixedStrides()), tiled});
    slice.getResult().replaceAllUsesWith(tiled);
    if (slice->use_empty())
      rewriter.eraseOp(slice);
    // The windowed destination may itself be the result of a nested
    // insert (a tiled concatenation).  Continue the walk through that
    // inner insert so every read lands on the innermost producer instead
    // of retaining the chained intermediate value in physical storage.
    // The enqueue must happen after the rewiring above: the destination
    // slice only has users once the original slice's uses have been
    // replaced (a pure destination copy forwards those users directly).
    if (mlir::Operation *destProducer = insert.getDest().getDefiningOp()) {
      llvm::SmallVector<mlir::Operation *, 4> destUsers;
      for (mlir::Operation *user : destSlice.getUsers())
        destUsers.push_back(user);
      enqueueSlices({destSlice.getDefiningOp()}, insert.getOperation(),
                    destUsers);
    }
    return;
  }
  // A dynamic window dimension (a temporal traversal induction variable)
  // needs a runtime intersection with the inserted region.  Compute the
  // per-dimension overlap [max(insert, window), min(insertEnd, windowEnd))
  // and select between the pure destination copy and the intersecting
  // insert through scf.if.
  //
  // The intersection must keep a static shape: per dimension the inserted
  // extent is a single element (the decode one-row cache write), or the
  // window lies completely inside the inserted extent with a statically
  // aligned start (a windowed read of a full-width cache write), or the
  // inserted extent covers the whole dimension (a full-width cache write
  // read through a traversal wave), or the window offset is an enclosing
  // loop induction variable whose wave has either one constant overlap or
  // a finite prologue/steady/tail overlap class with the inserted extent.
  llvm::SmallVector<int64_t, 4> intersectSizes(rank);
  bool staticIntersect = true;
  std::optional<unsigned> varyingIntersectDimension;
  llvm::SmallVector<int64_t, 3> varyingIntersectExtents;
  llvm::ArrayRef<int64_t> destDimSizes =
      mlir::cast<mlir::RankedTensorType>(insert.getDest().getType()).getShape();
  for (size_t dimension = 0; staticIntersect && dimension < rank; ++dimension) {
    if (insertSizes[dimension] == 1) {
      intersectSizes[dimension] = 1;
      continue;
    }
    auto windowOffset =
        mlir::getConstantIntValue(slice.getMixedOffsets()[dimension]);
    auto windowSize =
        mlir::getConstantIntValue(slice.getMixedSizes()[dimension]);
    if (!windowSize) {
      staticIntersect = false;
      break;
    }
    if (windowOffset) {
      // The inserted extent covers the whole dimension: the window lies
      // inside it whatever its static offset, so the overlap is the full
      // window extent (a full-width cache write read through a windowed
      // read, with the window starting past the cache's origin).
      if (insertOffsets[dimension] == 0 &&
          insertOffsets[dimension] + insertSizes[dimension] ==
              destDimSizes[dimension]) {
        intersectSizes[dimension] = *windowSize;
        continue;
      }
      // A static window offset keeps the overlap extent static even when
      // the window straddles the inserted region boundary (a row wave
      // clipped at a concatenation boundary).  The runtime disjoint check
      // below still selects the reachable branch; the intersecting branch
      // only touches the computed overlap.
      const int64_t windowHi = *windowOffset + *windowSize;
      const int64_t intersectLo =
          std::max(insertOffsets[dimension], *windowOffset);
      const int64_t intersectHi =
          std::min(insertOffsets[dimension] + insertSizes[dimension], windowHi);
      const int64_t intersectSize = intersectHi - intersectLo;
      if (intersectSize <= 0) {
        staticIntersect = false;
        break;
      }
      intersectSizes[dimension] = intersectSize;
      continue;
    }
    const int64_t insertHi = insertOffsets[dimension] + insertSizes[dimension];
    if (insertOffsets[dimension] == 0 && insertHi == destDimSizes[dimension]) {
      intersectSizes[dimension] = *windowSize;
      continue;
    }
    auto waveRange = getLoopCarriedWaveRange(
        mlir::cast<mlir::Value>(slice.getMixedOffsets()[dimension]));
    if (!waveRange || waveRange->step != *windowSize) {
      staticIntersect = false;
      break;
    }
    llvm::SmallVector<int64_t, 3> extents = getWaveOverlapExtents(
        *waveRange, *windowSize, insertOffsets[dimension], insertHi);
    if (extents.empty()) {
      staticIntersect = false;
      break;
    }
    if (extents.size() > 1) {
      if (varyingIntersectDimension) {
        staticIntersect = false;
        break;
      }
      varyingIntersectDimension = static_cast<unsigned>(dimension);
      varyingIntersectExtents = extents;
    }
    intersectSizes[dimension] = extents.front();
  }
  if (!staticIntersect)
    return;
  rewriter.setInsertionPoint(slice);
  llvm::SmallVector<mlir::Value, 4> windowLoValues;
  llvm::SmallVector<mlir::Value, 4> intersectLoValues;
  llvm::SmallVector<mlir::Value, 4> intersectSizeValues;
  auto toIndexValue = [&](mlir::OpFoldResult ofr) -> mlir::Value {
    if (auto value = mlir::dyn_cast<mlir::Value>(ofr))
      return value;
    return rewriter.create<mlir::arith::ConstantIndexOp>(
        slice.getLoc(), *mlir::getConstantIntValue(ofr));
  };
  mlir::Value disjoint =
      rewriter.create<mlir::arith::ConstantIntOp>(slice.getLoc(), 0, 1);
  for (size_t dimension = 0; dimension < rank; ++dimension) {
    mlir::Value windowLo = toIndexValue(slice.getMixedOffsets()[dimension]);
    mlir::Value windowHi = toIndexValue(slice.getMixedSizes()[dimension]);
    windowHi = rewriter.create<mlir::arith::AddIOp>(slice.getLoc(), windowLo,
                                                    windowHi);
    const int64_t insertLoValue = insertOffsets[dimension];
    const int64_t insertHiValue =
        insertOffsets[dimension] + insertSizes[dimension];
    mlir::Value insertHi = rewriter.create<mlir::arith::ConstantIndexOp>(
        slice.getLoc(), insertHiValue);
    mlir::Value insertLo = rewriter.create<mlir::arith::ConstantIndexOp>(
        slice.getLoc(), insertLoValue);
    const bool insertCoversDimension =
        insertLoValue == 0 && insertHiValue == destDimSizes[dimension];
    mlir::Value intersectLo = insertCoversDimension
                                  ? windowLo
                                  : rewriter.create<mlir::arith::MaxSIOp>(
                                        slice.getLoc(), insertLo, windowLo);
    mlir::Value intersectHi = insertCoversDimension
                                  ? windowHi
                                  : rewriter.create<mlir::arith::MinSIOp>(
                                        slice.getLoc(), insertHi, windowHi);
    // A window extracted from the destination is necessarily contained
    // in a full-dimension insert.  Do not manufacture an always-false
    // disjoint branch: retaining it hides a straight-line physical loop
    // from exact scheduling and selected buffering.
    if (!insertCoversDimension) {
      mlir::Value emptyDim = rewriter.create<mlir::arith::CmpIOp>(
          slice.getLoc(), mlir::arith::CmpIPredicate::sle, intersectHi,
          intersectLo);
      disjoint = rewriter.create<mlir::arith::OrIOp>(slice.getLoc(), disjoint,
                                                     emptyDim);
    }
    windowLoValues.push_back(windowLo);
    intersectLoValues.push_back(intersectLo);
    intersectSizeValues.push_back(rewriter.create<mlir::arith::SubIOp>(
        slice.getLoc(), intersectHi, intersectLo));
  }
  llvm::SmallVector<mlir::OpFoldResult, 4> unitStrides(
      rank, rewriter.getIndexAttr(1));
  mlir::Value destSlice =
      rewriter
          .create<mlir::tensor::ExtractSliceOp>(
              slice.getLoc(), slice.getType(), insert.getDest(),
              slice.getMixedOffsets(), slice.getMixedSizes(), unitStrides)
          .getResult();
  auto materializeStaticIntersection =
      [&](mlir::OpBuilder &branchBuilder,
          llvm::ArrayRef<int64_t> staticIntersectSizes) -> mlir::Value {
    auto sourceType =
        mlir::cast<mlir::RankedTensorType>(insert.getSource().getType());
    auto inputSliceType = mlir::RankedTensorType::get(
        staticIntersectSizes, sourceType.getElementType());
    llvm::SmallVector<mlir::OpFoldResult, 4> inputOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> inputSizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> windowRelativeOffsets;
    inputOffsets.reserve(rank);
    inputSizes.reserve(rank);
    windowRelativeOffsets.reserve(rank);
    for (size_t dimension = 0; dimension < rank; ++dimension) {
      mlir::Value intersectLo = intersectLoValues[dimension];
      auto clampRelativeOffset = [&](mlir::Value relative, int64_t maximum) {
        mlir::Value zero = branchBuilder.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), 0);
        mlir::Value upper = branchBuilder.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), maximum);
        relative = branchBuilder.create<mlir::arith::MaxSIOp>(slice.getLoc(),
                                                              relative, zero);
        return branchBuilder
            .create<mlir::arith::MinSIOp>(slice.getLoc(), relative, upper)
            .getResult();
      };
      mlir::Value inputRelative =
          branchBuilder
              .create<mlir::arith::SubIOp>(
                  slice.getLoc(), intersectLo,
                  branchBuilder.create<mlir::arith::ConstantIndexOp>(
                      slice.getLoc(), insertOffsets[dimension]))
              .getResult();
      inputOffsets.push_back(clampRelativeOffset(
          inputRelative,
          insertSizes[dimension] - staticIntersectSizes[dimension]));
      inputSizes.push_back(
          branchBuilder.getIndexAttr(staticIntersectSizes[dimension]));
      mlir::Value windowRelative =
          branchBuilder
              .create<mlir::arith::SubIOp>(slice.getLoc(), intersectLo,
                                           windowLoValues[dimension])
              .getResult();
      windowRelativeOffsets.push_back(clampRelativeOffset(
          windowRelative, slice.getType().getDimSize(dimension) -
                              staticIntersectSizes[dimension]));
    }
    mlir::Value inputSlice =
        branchBuilder
            .create<mlir::tensor::ExtractSliceOp>(
                slice.getLoc(), inputSliceType, insert.getSource(),
                inputOffsets, inputSizes, unitStrides)
            .getResult();
    auto inserted = branchBuilder.create<mlir::tensor::InsertSliceOp>(
        slice.getLoc(), inputSlice, destSlice, windowRelativeOffsets,
        inputSizes, insert.getMixedStrides());
    if (auto resource = insert->getAttrOfType<CardDDRResourceAttr>(
            kWaferCardDDRMovementAttrName))
      inserted->setAttr(kWaferCardDDRMovementAttrName, resource);
    copyNodeMappings(inserted);
    enqueueSlices({inputSlice.getDefiningOp()}, insert.getOperation(),
                  {inserted.getOperation()});
    return inserted.getResult();
  };
  auto ifOp = rewriter.create<mlir::scf::IfOp>(
      slice.getLoc(), mlir::TypeRange{slice.getType()}, disjoint, true);
  // disjoint branch: the insert does not touch this window, forward the
  // pre-insert destination version unchanged.
  {
    mlir::OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    mlir::OpBuilder::InsertionGuard guard(thenBuilder);
    thenBuilder.create<mlir::scf::YieldOp>(slice.getLoc(), destSlice);
  }
  {
    // Intersecting branch: every carried wave has one of the statically
    // proven overlap shapes.  When an insertion boundary cuts the final
    // wave (for example a 1023-row cache plus one new row under 64-row
    // traversal), select the finite static shape with nested scf.if
    // rather than retaining the full functional tensor in SPM.
    mlir::OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
    mlir::OpBuilder::InsertionGuard guard(elseBuilder);
    mlir::Value intersected;
    if (!varyingIntersectDimension) {
      intersected = materializeStaticIntersection(elseBuilder, intersectSizes);
    } else {
      std::function<mlir::Value(mlir::OpBuilder &, size_t)> buildChoice;
      buildChoice = [&](mlir::OpBuilder &choiceBuilder,
                        size_t extentIndex) -> mlir::Value {
        llvm::SmallVector<int64_t, 4> selectedSizes(intersectSizes);
        selectedSizes[*varyingIntersectDimension] =
            varyingIntersectExtents[extentIndex];
        if (extentIndex + 1 == varyingIntersectExtents.size())
          return materializeStaticIntersection(choiceBuilder, selectedSizes);
        auto expected = choiceBuilder.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), varyingIntersectExtents[extentIndex]);
        auto matches = choiceBuilder.create<mlir::arith::CmpIOp>(
            slice.getLoc(), mlir::arith::CmpIPredicate::eq,
            intersectSizeValues[*varyingIntersectDimension], expected);
        auto choice = choiceBuilder.create<mlir::scf::IfOp>(
            slice.getLoc(), mlir::TypeRange{slice.getType()}, matches, true);
        {
          mlir::OpBuilder thenBuilder = choice.getThenBodyBuilder();
          mlir::OpBuilder::InsertionGuard nestedGuard(thenBuilder);
          thenBuilder.create<mlir::scf::YieldOp>(
              slice.getLoc(),
              materializeStaticIntersection(thenBuilder, selectedSizes));
        }
        {
          mlir::OpBuilder nestedElseBuilder = choice.getElseBodyBuilder();
          mlir::OpBuilder::InsertionGuard nestedGuard(nestedElseBuilder);
          nestedElseBuilder.create<mlir::scf::YieldOp>(
              slice.getLoc(), buildChoice(nestedElseBuilder, extentIndex + 1));
        }
        return choice.getResult(0);
      };
      intersected = buildChoice(elseBuilder, /*extentIndex=*/0);
    }
    elseBuilder.create<mlir::scf::YieldOp>(slice.getLoc(), intersected);
  }
  mlir::Value tiled = ifOp.getResult(0);
  materializedCoupledTiles.push_back(MaterializedCoupledProducerTile{
      producerResult, slice->getBlock(), slice.getType(),
      llvm::to_vector(slice.getMixedOffsets()),
      llvm::to_vector(slice.getMixedSizes()),
      llvm::to_vector(slice.getMixedStrides()), tiled});
  slice.getResult().replaceAllUsesWith(tiled);
  if (slice->use_empty())
    rewriter.eraseOp(slice);
  // The windowed destination may itself be the result of a nested
  // insert (a tiled concatenation).  Continue the walk through that
  // inner insert so every read lands on the innermost producer instead
  // of retaining the chained intermediate value in physical storage.
  // The enqueue must happen after the rewiring above: the destination
  // slice only has users once the original slice's uses have been
  // replaced (a pure destination copy forwards those users directly).
  if (mlir::Operation *destProducer = insert.getDest().getDefiningOp()) {
    llvm::SmallVector<mlir::Operation *, 4> destUsers;
    for (mlir::Operation *user : destSlice.getUsers())
      destUsers.push_back(user);
    enqueueSlices({destSlice.getDefiningOp()}, insert.getOperation(),
                  destUsers);
  }
  return;
}

} // namespace wafer::tensor_program_to_tile_region
