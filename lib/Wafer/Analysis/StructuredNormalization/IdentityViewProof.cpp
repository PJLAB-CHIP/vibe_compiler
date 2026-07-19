//===- IdentityViewProof.cpp - Recomputable identity view proof ----------===//

#include "Wafer/Analysis/IdentityViewProof.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <algorithm>

namespace wafer {
namespace {

static std::optional<int64_t> proveExactIntegerValue(mlir::Value value,
                                                     unsigned depth = 0) {
  if (!value || depth > 16)
    return std::nullopt;
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return constant;
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return proveExactIntegerValue(cast.getIn(), depth + 1);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    std::optional<int64_t> input =
        proveExactIntegerValue(cast.getIn(), depth + 1);
    if (input && *input >= 0)
      return input;
    return std::nullopt;
  }

  auto proveBinary = [&](mlir::Value lhs, mlir::Value rhs,
                         bool takeMaximum) -> std::optional<int64_t> {
    std::optional<int64_t> lhsValue =
        proveExactIntegerValue(lhs, depth + 1);
    std::optional<int64_t> rhsValue =
        proveExactIntegerValue(rhs, depth + 1);
    if (!lhsValue || !rhsValue)
      return std::nullopt;
    return takeMaximum ? std::max(*lhsValue, *rhsValue)
                       : std::min(*lhsValue, *rhsValue);
  };
  if (auto minimum = value.getDefiningOp<mlir::arith::MinSIOp>()) {
    if (std::optional<int64_t> exact =
            proveBinary(minimum.getLhs(), minimum.getRhs(), false))
      return exact;
    auto proveUpperClamp = [&](mlir::Value upperValue,
                               mlir::Value flooredValue)
        -> std::optional<int64_t> {
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(upperValue);
      auto maximum = flooredValue.getDefiningOp<mlir::arith::MaxSIOp>();
      if (!upper || !maximum)
        return std::nullopt;
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(maximum.getLhs());
      if (!lower)
        lower = mlir::getConstantIntValue(maximum.getRhs());
      return lower && *lower >= *upper ? upper : std::nullopt;
    };
    if (std::optional<int64_t> exact =
            proveUpperClamp(minimum.getLhs(), minimum.getRhs()))
      return exact;
    return proveUpperClamp(minimum.getRhs(), minimum.getLhs());
  }
  if (auto maximum = value.getDefiningOp<mlir::arith::MaxSIOp>()) {
    if (std::optional<int64_t> exact =
            proveBinary(maximum.getLhs(), maximum.getRhs(), true))
      return exact;
    auto proveLowerClamp = [&](mlir::Value lowerValue,
                               mlir::Value cappedValue)
        -> std::optional<int64_t> {
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(lowerValue);
      auto minimum = cappedValue.getDefiningOp<mlir::arith::MinSIOp>();
      if (!lower || !minimum)
        return std::nullopt;
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(minimum.getLhs());
      if (!upper)
        upper = mlir::getConstantIntValue(minimum.getRhs());
      return upper && *upper <= *lower ? lower : std::nullopt;
    };
    if (std::optional<int64_t> exact =
            proveLowerClamp(maximum.getLhs(), maximum.getRhs()))
      return exact;
    return proveLowerClamp(maximum.getRhs(), maximum.getLhs());
  }
  return std::nullopt;
}

static std::optional<int64_t>
proveExactIntegerValue(mlir::OpFoldResult value) {
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return constant;
  auto dynamic = mlir::dyn_cast<mlir::Value>(value);
  return dynamic ? proveExactIntegerValue(dynamic) : std::nullopt;
}

template <typename ViewOp>
static IdentityViewProof proveStaticIdentityView(ViewOp view) {
  auto sourceType = mlir::dyn_cast<mlir::ShapedType>(view.getSource().getType());
  auto resultType = mlir::dyn_cast<mlir::ShapedType>(view.getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return {IdentityProofStatus::Unsupported, 0};
  if (sourceType.getRank() != resultType.getRank())
    return {IdentityProofStatus::ProvenNonIdentity, 0};

  uint64_t dimensions = static_cast<uint64_t>(sourceType.getRank());
  auto offsets = view.getMixedOffsets();
  auto sizes = view.getMixedSizes();
  auto strides = view.getMixedStrides();
  if (offsets.size() != dimensions || sizes.size() != dimensions ||
      strides.size() != dimensions)
    return {IdentityProofStatus::Unsupported, dimensions};

  for (uint64_t index = 0; index < dimensions; ++index) {
    std::optional<int64_t> offset = proveExactIntegerValue(offsets[index]);
    std::optional<int64_t> size = proveExactIntegerValue(sizes[index]);
    std::optional<int64_t> stride = proveExactIntegerValue(strides[index]);
    if (!offset || !size || !stride)
      return {IdentityProofStatus::Unsupported, dimensions};
    if (*offset != 0 || *stride != 1 ||
        *size != sourceType.getDimSize(index))
      return {IdentityProofStatus::ProvenNonIdentity, dimensions};
  }
  return {IdentityProofStatus::ProvenIdentity, dimensions};
}

} // namespace

IdentityViewProof proveIdentityView(mlir::tensor::ExtractSliceOp slice) {
  return proveStaticIdentityView(slice);
}

IdentityViewProof proveIdentityView(mlir::memref::SubViewOp subview) {
  return proveStaticIdentityView(subview);
}

} // namespace wafer
