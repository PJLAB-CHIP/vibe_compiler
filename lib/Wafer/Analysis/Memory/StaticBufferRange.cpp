//===- StaticBufferRange.cpp - Proven static buffer intervals -----------===//

#include "Wafer/Analysis/Memory/StaticBufferRange.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace wafer::analysis {
namespace {

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs))
    return false;
  result = lhs * rhs;
  return true;
}

static bool isDenseContiguous(llvm::ArrayRef<int64_t> shape,
                              llvm::ArrayRef<int64_t> strides,
                              int64_t &elementCount) {
  if (shape.size() != strides.size())
    return false;

  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> ordered;
  for (auto [extent, stride] : llvm::zip_equal(shape, strides)) {
    if (extent == mlir::ShapedType::kDynamic || extent <= 0 ||
        stride == mlir::ShapedType::kDynamic || stride < 0)
      return false;
    if (extent > 1)
      ordered.emplace_back(stride, extent);
  }
  llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  int64_t expectedStride = 1;
  for (auto [stride, extent] : ordered) {
    if (stride != expectedStride ||
        !checkedMul(expectedStride, extent, expectedStride))
      return false;
  }
  elementCount = expectedStride;
  return true;
}

static bool haveCompatibleStorageTypes(mlir::MemRefType lhs,
                                       mlir::MemRefType rhs) {
  return lhs && rhs && lhs.getElementType() == rhs.getElementType() &&
         lhs.getMemorySpace() == rhs.getMemorySpace();
}

/// Recompute a rank-preserving subview's result layout from its source. The
/// result type is not accepted as an independent assertion: offsets are
/// multiplied by the source's static element strides, and result strides are
/// the source strides multiplied by the explicit subview strides.
static bool validateStaticSubview(mlir::memref::SubViewOp subview) {
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
  auto resultType =
      mlir::dyn_cast<mlir::MemRefType>(subview.getResult().getType());
  if (!haveCompatibleStorageTypes(sourceType, resultType) ||
      sourceType.getRank() != resultType.getRank() ||
      !sourceType.hasStaticShape() || !resultType.hasStaticShape())
    return false;

  llvm::SmallVector<int64_t, 4> sourceStrides;
  llvm::SmallVector<int64_t, 4> resultStrides;
  int64_t sourceOffset = 0;
  int64_t resultOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(sourceType, sourceStrides, sourceOffset)) ||
      mlir::failed(
          mlir::getStridesAndOffset(resultType, resultStrides, resultOffset)) ||
      sourceOffset == mlir::ShapedType::kDynamic || sourceOffset < 0 ||
      resultOffset == mlir::ShapedType::kDynamic || resultOffset < 0 ||
      subview.getStaticOffsets().size() !=
          static_cast<size_t>(sourceType.getRank()) ||
      subview.getStaticSizes().size() !=
          static_cast<size_t>(sourceType.getRank()) ||
      subview.getStaticStrides().size() !=
          static_cast<size_t>(sourceType.getRank()))
    return false;

  int64_t expectedOffset = sourceOffset;
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    int64_t offset = subview.getStaticOffsets()[dim];
    int64_t size = subview.getStaticSizes()[dim];
    int64_t stride = subview.getStaticStrides()[dim];
    int64_t sourceStride = sourceStrides[dim];
    if (offset == mlir::ShapedType::kDynamic ||
        size == mlir::ShapedType::kDynamic ||
        stride == mlir::ShapedType::kDynamic ||
        sourceStride == mlir::ShapedType::kDynamic || offset < 0 || size <= 0 ||
        stride <= 0 || sourceStride < 0 || resultType.getDimSize(dim) != size)
      return false;

    int64_t scaledOffset = 0;
    int64_t expectedStride = 0;
    int64_t span = 0;
    int64_t last = 0;
    if (!checkedMul(offset, sourceStride, scaledOffset) ||
        !checkedAdd(expectedOffset, scaledOffset, expectedOffset) ||
        !checkedMul(sourceStride, stride, expectedStride) ||
        resultStrides[dim] != expectedStride ||
        !checkedMul(size - 1, stride, span) ||
        !checkedAdd(offset, span, last) || last >= sourceType.getDimSize(dim))
      return false;
  }
  return expectedOffset == resultOffset;
}

static mlir::Value resolveTileRegionAlias(mlir::Value value) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    auto region =
        owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
              : TileRegionOp();
    if (region && !region.getBody().empty() &&
        owner == &region.getBody().front() &&
        argument.getArgNumber() < region.getInputs().size())
      return region.getInputs()[argument.getArgNumber()];
    return {};
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  auto region =
      result ? mlir::dyn_cast<TileRegionOp>(result.getOwner()) : TileRegionOp();
  if (!region || region.getBody().empty() ||
      result.getResultNumber() >=
          region.getBody().front().getTerminator()->getNumOperands())
    return {};
  return region.getBody().front().getTerminator()->getOperand(
      result.getResultNumber());
}

static bool isSupportedRoot(mlir::Value value) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    auto function =
        owner ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(owner->getParentOp())
              : mlir::func::FuncOp();
    return function && !function.empty() && owner == &function.front();
  }
  mlir::Operation *definition = value.getDefiningOp();
  return definition && (mlir::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp,
                                  mlir::memref::GetGlobalOp>(definition));
}

} // namespace

std::optional<StaticByteRange> getStaticByteRange(mlir::Value value) {
  auto type = value ? mlir::dyn_cast<mlir::MemRefType>(value.getType())
                    : mlir::MemRefType();
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->bitPackedElement || info->elementBytes <= 0 ||
      (info->layout != MemLayout::Tensor && info->layout != MemLayout::NTensor))
    return std::nullopt;

  llvm::SmallVector<int64_t, 4> strides;
  int64_t offsetElements = 0;
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offsetElements)) ||
      offsetElements == mlir::ShapedType::kDynamic || offsetElements < 0)
    return std::nullopt;

  int64_t elementCount = 0;
  int64_t begin = 0;
  int64_t length = 0;
  int64_t end = 0;
  if (!isDenseContiguous(type.getShape(), strides, elementCount) ||
      !checkedMul(offsetElements, info->elementBytes, begin) ||
      !checkedMul(elementCount, info->elementBytes, length) ||
      !checkedAdd(begin, length, end))
    return std::nullopt;
  return StaticByteRange{begin, end};
}

std::optional<StaticBufferRange> resolveStaticBufferRange(mlir::Value value) {
  std::optional<StaticByteRange> leafRange = getStaticByteRange(value);
  if (!leafRange)
    return std::nullopt;

  llvm::DenseSet<mlir::Value> seen;
  mlir::Value current = value;
  while (current && seen.insert(current).second) {
    if (mlir::Value alias = resolveTileRegionAlias(current)) {
      current = alias;
      continue;
    }
    mlir::Operation *definition = current.getDefiningOp();
    if (auto subview =
            mlir::dyn_cast_or_null<mlir::memref::SubViewOp>(definition)) {
      if (!validateStaticSubview(subview))
        return std::nullopt;
      current = subview.getSource();
      continue;
    }
    if (auto cast = mlir::dyn_cast_or_null<mlir::memref::CastOp>(definition)) {
      auto sourceType =
          mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
      auto resultType = mlir::dyn_cast<mlir::MemRefType>(cast.getType());
      if (!haveCompatibleStorageTypes(sourceType, resultType))
        return std::nullopt;
      current = cast.getSource();
      continue;
    }
    if (definition && mlir::isa<mlir::ViewLikeOpInterface>(definition))
      return std::nullopt;
    break;
  }
  if (!current || !isSupportedRoot(current))
    return std::nullopt;

  std::optional<StaticByteRange> rootRange = getStaticByteRange(current);
  if (!rootRange || !staticByteRangeContains(*rootRange, *leafRange))
    return std::nullopt;
  return StaticBufferRange{current,
                           StaticByteRange{leafRange->begin - rootRange->begin,
                                           leafRange->end - rootRange->begin}};
}

bool staticByteRangesOverlap(const StaticByteRange &lhs,
                             const StaticByteRange &rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool staticByteRangesAreDisjoint(const StaticByteRange &lhs,
                                 const StaticByteRange &rhs) {
  return !staticByteRangesOverlap(lhs, rhs);
}

bool staticByteRangeContains(const StaticByteRange &container,
                             const StaticByteRange &contained) {
  return container.begin <= contained.begin && contained.end <= container.end;
}

} // namespace wafer::analysis
