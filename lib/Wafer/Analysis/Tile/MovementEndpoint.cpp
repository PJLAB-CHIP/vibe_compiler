//===- MovementEndpoint.cpp - Current physical movement endpoints --------===//
#include "Wafer/Analysis/Tile/MovementEndpoint.h"
#include "../Instr/StaticIndexRange.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

namespace wafer::analysis {
mlir::FailureOr<MovementEndpoint>
resolveStaticMovementEndpoint(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return mlir::failure();
  auto relation = analysis::IndexRelation::identity(type.getShape());
  if (!relation.isExact())
    return mlir::failure();
  while (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    auto viewType = subview.getType();
    auto baseType = subview.getSourceType();
    auto dropped = subview.getDroppedDims();
    llvm::SmallVector<int64_t, 4> fullViewShape;
    unsigned viewAxis = 0;
    for (unsigned axis = 0; axis < dropped.size(); ++axis)
      fullViewShape.push_back(
          dropped.test(axis) ? 1 : viewType.getDimSize(viewAxis++));
    auto expand = analysis::IndexRelation::staticReshape(viewType.getShape(),
                                                         fullViewShape);
    auto slice = analysis::IndexRelation::slice(
        fullViewShape, baseType.getShape(), subview.getMixedOffsets(),
        subview.getMixedStrides());
    if (!expand.isExact() || !slice.isExact())
      return mlir::failure();
    auto step = expand.get()->compose(*slice.get());
    if (!step.isExact())
      return mlir::failure();
    relation = relation.get()->compose(*step.get());
    if (!relation.isExact())
      return mlir::failure();
    value = subview.getSource();
  }
  return MovementEndpoint{value, std::move(*relation.relation)};
}

namespace {
// A blocked window uses the allocation's coordinate system, including its
// global block stride. Separate proven dynamic translations from the static
// relation; descriptors remain rooted in the actual allocation.
mlir::FailureOr<MovementEndpoint>
resolveBlockedMovementEndpoint(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return mlir::failure();
  auto relation = analysis::IndexRelation::identity(type.getShape());
  struct LogicalTerm {
    unsigned axis;
    mlir::Value value;
    int64_t factor;
  };
  llvm::SmallVector<LogicalTerm, 4> terms;
  while (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    auto sourceType = subview.getSourceType();
    auto dropped = subview.getDroppedDims();
    llvm::SmallVector<int64_t> shape, offsets, strides;
    llvm::SmallVector<unsigned> sourceAxes;
    unsigned resultAxis = 0;
    for (unsigned axis = 0; axis < dropped.size(); ++axis) {
      if (dropped.test(axis))
        shape.push_back(1);
      else {
        sourceAxes.push_back(axis);
        shape.push_back(subview.getType().getDimSize(resultAxis++));
      }
    }
    for (mlir::OpFoldResult stride : subview.getMixedStrides()) {
      auto constant = mlir::getConstantIntValue(stride);
      if (!constant || *constant <= 0)
        return mlir::failure();
      strides.push_back(*constant);
    }
    for (LogicalTerm &term : terms) {
      term.axis = sourceAxes[term.axis];
      if (llvm::MulOverflow(term.factor, strides[term.axis], term.factor))
        return mlir::failure();
    }
    for (auto [axis, offset] : llvm::enumerate(subview.getMixedOffsets())) {
      if (auto constant = mlir::getConstantIntValue(offset))
        offsets.push_back(*constant);
      else {
        offsets.push_back(0);
        terms.push_back(
            {static_cast<unsigned>(axis), mlir::cast<mlir::Value>(offset), 1});
      }
    }
    auto expand = analysis::IndexRelation::staticReshape(
        subview.getType().getShape(), shape);
    auto slice = analysis::IndexRelation::staticSlice(
        shape, sourceType.getShape(), offsets, strides);
    if (!relation.isExact() || !expand.isExact() || !slice.isExact())
      return mlir::failure();
    auto step = expand.get()->compose(*slice.get());
    if (!step.isExact())
      return mlir::failure();
    relation = relation.get()->compose(*step.get());
    value = subview.getSource();
  }
  if (!relation.isExact())
    return mlir::failure();
  auto rootType = mlir::cast<mlir::MemRefType>(value.getType());
  MovementEndpoint endpoint{value, std::move(*relation.relation), {}};
  if (terms.empty())
    return endpoint;
  auto calculator = WaferStaticPhysicalOffsetCalculator::create(rootType);
  if (!calculator)
    return mlir::failure();
  const auto &geometry = calculator->getInfo();
  // Full blocks have one translation for every touched C block. A retained
  // narrow C tail has a different row stride and needs a piecewise route.
  if (geometry.tailC || geometry.bitPackedElement || geometry.cBlock <= 0)
    return mlir::failure();
  llvm::SmallVector<int64_t> origin(rootType.getRank(), 0);
  auto base = calculator->getByteOffset(origin);
  if (!base)
    return mlir::failure();
  for (const LogicalTerm &term : terms) {
    const bool channel = term.axis + 1 == rootType.getRank();
    int64_t quantum = channel ? geometry.cBlock : 1;
    if (rootType.getDimSize(term.axis) <= quantum) {
      auto range = memory_planning::detail::evaluateNonNegativeStaticIndexRange(
          term.value);
      if (!range.succeeded() || (!range.range.empty && range.range.max != 0))
        return mlir::failure();
      continue;
    }
    origin[term.axis] = quantum;
    auto translated = calculator->getByteOffset(origin);
    origin[term.axis] = 0;
    if (!translated || *translated < *base)
      return mlir::failure();
    int64_t common = std::gcd(quantum, term.factor);
    int64_t denominator = quantum / common;
    if (denominator != 1) {
      auto remainder = memory_planning::detail::getKnownIndexRemainder(
          term.value, denominator);
      if (!remainder || *remainder)
        return mlir::failure();
    }
    int64_t numerator;
    if (llvm::MulOverflow(*translated - *base, term.factor / common, numerator))
      return mlir::failure();
    endpoint.dynamicOffsets.push_back({term.value, numerator, denominator});
  }
  return endpoint;
}

} // namespace

// Preserve an existing standard endpoint whose dynamic base address is carried
// by its current SSA view. Static and blocked subviews use the common relation.
mlir::FailureOr<MovementEndpoint> resolveMovementEndpoint(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return mlir::failure();
  auto memory = getWaferMemoryAttr(type);
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (memory &&
      (memory.getLayout() == MemLayout::Tensor ||
       memory.getLayout() == MemLayout::NTensor) &&
      mlir::succeeded(mlir::getStridesAndOffset(type, strides, offset)) &&
      mlir::ShapedType::isDynamic(offset)) {
    auto identity = analysis::IndexRelation::identity(type.getShape());
    if (!identity.isExact())
      return mlir::failure();
    return MovementEndpoint{value, std::move(*identity.relation)};
  }
  if (memory && (memory.getLayout() == MemLayout::Cx ||
                 memory.getLayout() == MemLayout::NCx))
    return resolveBlockedMovementEndpoint(value);
  return resolveStaticMovementEndpoint(value);
}

} // namespace wafer::analysis
