//===- TensorAssemblyMaterialization.cpp - Selected tensor demand --------===//

#include "TensorAssemblyMaterialization.h"
#include "Wafer/Transforms/Linalg/StructuredTiling.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <functional>

namespace wafer::compiler::detail {

std::optional<int64_t> resolveStaticIndex(mlir::OpFoldResult value) {
  if (auto attribute = llvm::dyn_cast_if_present<mlir::IntegerAttr>(
          value.dyn_cast<mlir::Attribute>()))
    return attribute.getInt();
  mlir::Value dynamic = value.dyn_cast<mlir::Value>();
  if (!dynamic || !dynamic.getType().isIndex())
    return std::nullopt;
  mlir::FailureOr<int64_t> constant =
      mlir::ValueBoundsConstraintSet::computeConstantBound(
          mlir::presburger::BoundType::EQ,
          mlir::ValueBoundsConstraintSet::Variable(dynamic));
  if (mlir::succeeded(constant))
    return *constant;
  std::function<std::optional<int64_t>(mlir::Value, unsigned)> resolve =
      [&](mlir::Value current, unsigned depth) -> std::optional<int64_t> {
    if (!current || depth > 8)
      return std::nullopt;
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(current))
      return constant;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      auto loop = argument.getOwner()
                      ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                            argument.getOwner()->getParentOp())
                      : mlir::scf::ForOp{};
      if (!loop || argument != loop.getInductionVar())
        return std::nullopt;
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(loop.getUpperBound());
      std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
          *upper - *lower > *step)
        return std::nullopt;
      return lower;
    }
    mlir::Operation *definition = current.getDefiningOp();
    mlir::AffineMap map;
    mlir::ValueRange operands;
    bool takeMinimum = false;
    bool takeMaximum = false;
    if (auto apply =
            mlir::dyn_cast_or_null<mlir::affine::AffineApplyOp>(definition)) {
      map = apply.getAffineMap();
      operands = apply.getMapOperands();
    } else if (auto minimum = mlir::dyn_cast_or_null<mlir::affine::AffineMinOp>(
                   definition)) {
      map = minimum.getAffineMap();
      operands = minimum.getMapOperands();
      takeMinimum = true;
    } else if (auto maximum = mlir::dyn_cast_or_null<mlir::affine::AffineMaxOp>(
                   definition)) {
      map = maximum.getAffineMap();
      operands = maximum.getMapOperands();
      takeMaximum = true;
    } else {
      return std::nullopt;
    }
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (mlir::Value operand : operands) {
      std::optional<int64_t> operandValue = resolve(operand, depth + 1);
      if (!operandValue)
        return std::nullopt;
      attributes.push_back(mlir::IntegerAttr::get(
          mlir::IndexType::get(current.getContext()), *operandValue));
    }
    llvm::SmallVector<mlir::Attribute, 4> folded;
    if (mlir::failed(map.constantFold(attributes, folded)) || folded.empty())
      return std::nullopt;
    std::optional<int64_t> result;
    for (mlir::Attribute attribute : folded) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute);
      if (!integer)
        return std::nullopt;
      if (!result)
        result = integer.getInt();
      else if (takeMinimum)
        result = std::min(*result, integer.getInt());
      else if (takeMaximum)
        result = std::max(*result, integer.getInt());
      else
        return std::nullopt;
    }
    return result;
  };
  return resolve(dynamic, 0);
}

analysis::TensorAssemblyReadResult
queryTensorAssemblySlice(mlir::tensor::ExtractSliceOp read,
                         const analysis::IndexRelationLimits &limits) {
  if (read.getType().getRank() == 0)
    return {analysis::TensorAssemblyStatus::Unsupported,
            {},
            {},
            "scalar assembly reads have no compact tensor representation"};
  llvm::SmallVector<int64_t, 4> sizes;
  for (auto size : read.getMixedSizes()) {
    auto constant = resolveStaticIndex(size);
    if (!constant || *constant <= 0)
      return {analysis::TensorAssemblyStatus::Unsupported,
              {},
              {},
              "assembly read requires positive static sizes"};
    sizes.push_back(*constant);
  }
  if ((read.getType().getRank() != static_cast<int64_t>(sizes.size()) &&
       (!read.getType().hasStaticShape() ||
        !mlir::computeRankReductionMask(sizes, read.getType().getShape()))) ||
      llvm::any_of(read.getMixedStrides(),
                   [](auto stride) { return resolveStaticIndex(stride) != 1; }))
    return {analysis::TensorAssemblyStatus::Unsupported,
            {},
            {},
            "assembly read requires exact rank reduction and unit strides"};
  auto offsets = read.getMixedOffsets();
  for (auto &offset : offsets)
    if (auto constant = resolveStaticIndex(offset))
      offset = mlir::IntegerAttr::get(mlir::IndexType::get(read.getContext()),
                                      *constant);
  auto result = analysis::queryTensorAssemblyRead(read.getSource(), offsets,
                                                  sizes, limits);
  if (result.isExact())
    for (const auto &part : result.cases)
      for (const auto &piece : part.pieces)
        if (piece.sourceSizes.empty())
          return {
              analysis::TensorAssemblyStatus::Unsupported,
              {},
              {},
              "scalar assembly sources have no compact tensor representation"};
  return result;
}

mlir::FailureOr<MaterializedTensorAssemblyRead>
materializeTensorAssemblyRead(mlir::OpBuilder &builder, mlir::Location location,
                              mlir::RankedTensorType resultType,
                              llvm::ArrayRef<int64_t> sizes,
                              const analysis::TensorAssemblyReadResult &read) {
  if (!read.isExact() || read.cases.size() != 1)
    return mlir::failure();
  MaterializedTensorAssemblyRead result;
  llvm::SmallVector<mlir::Value, 4> inductions;
  for (const auto &loop : read.loops)
    inductions.push_back(loop.induction);
  llvm::SmallVector<TensorAssemblyTilePiece, 4> pieces;
  for (const auto &piece : read.cases.front().pieces) {
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets, pieceSizes, strides;
    for (unsigned axis = 0; axis < piece.sourceSizes.size(); ++axis) {
      auto expression = piece.sourceOffsets.getResult(axis);
      if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression))
        offsets.push_back(builder.getIndexAttr(constant.getValue()));
      else
        offsets.push_back(mlir::affine::makeComposedFoldedAffineApply(
            builder, location, piece.sourceOffsets.getSubMap({axis}),
            llvm::to_vector(llvm::map_range(inductions, [](mlir::Value value) {
              return mlir::OpFoldResult(value);
            }))));
      pieceSizes.push_back(builder.getIndexAttr(piece.sourceSizes[axis]));
      strides.push_back(builder.getIndexAttr(1));
    }
    auto source = builder.create<mlir::tensor::ExtractSliceOp>(
        location, piece.source, offsets, pieceSizes, strides);
    auto pieceType = mlir::RankedTensorType::get(piece.resultWindow.sizes,
                                                 resultType.getElementType(),
                                                 resultType.getEncoding());
    result.sourceReads.push_back(source);
    pieces.push_back(
        {reshapeStaticTensorTile(builder, location, source, pieceType),
         piece.resultWindow});
  }
  auto compactType = mlir::RankedTensorType::get(
      read.shape, resultType.getElementType(), resultType.getEncoding());
  analysis::StaticRectangularIndexSet requested;
  requested.offsets.assign(read.shape.size(), 0);
  requested.sizes = read.shape;
  auto assembled = materializeTensorAssemblyTile(builder, location, compactType,
                                                 requested, pieces);
  if (mlir::failed(assembled))
    return mlir::failure();
  auto preciseType =
      resultType.hasStaticShape()
          ? resultType
          : mlir::RankedTensorType::get(sizes, resultType.getElementType(),
                                        resultType.getEncoding());
  result.value =
      reshapeStaticTensorTile(builder, location, *assembled, preciseType);
  if (preciseType != resultType)
    result.value = builder.create<mlir::tensor::CastOp>(location, resultType,
                                                        result.value);
  return result;
}

mlir::FailureOr<mlir::Value> materializeTensorAssemblyTile(
    mlir::OpBuilder &builder, mlir::Location location,
    mlir::RankedTensorType resultType,
    const analysis::StaticRectangularIndexSet &requested,
    llvm::ArrayRef<TensorAssemblyTilePiece> pieces) {
  const size_t rank = resultType.getRank();
  if (pieces.empty() || !resultType.hasStaticShape() ||
      requested.offsets.size() != rank || requested.sizes.size() != rank ||
      resultType.getShape() != llvm::ArrayRef<int64_t>(requested.sizes))
    return mlir::failure();
  int64_t requestedElements = 1;
  for (int64_t size : requested.sizes)
    if (size <= 0 ||
        llvm::MulOverflow(requestedElements, size, requestedElements))
      return mlir::failure();
  int64_t coveredElements = 0;
  for (const auto &piece : pieces) {
    if (!piece.value)
      return mlir::failure();
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(piece.value.getType());
    if (!type || !type.hasStaticShape() ||
        type.getElementType() != resultType.getElementType() ||
        type.getEncoding() != resultType.getEncoding() ||
        piece.resultWindow.offsets.size() != rank ||
        piece.resultWindow.sizes.size() != rank ||
        type.getShape() != llvm::ArrayRef<int64_t>(piece.resultWindow.sizes))
      return mlir::failure();
    int64_t pieceElements = 1;
    for (auto [origin, extent, offset, size] : llvm::zip_equal(
             requested.offsets, requested.sizes, piece.resultWindow.offsets,
             piece.resultWindow.sizes)) {
      int64_t end = 0;
      if (size <= 0 || origin < 0 || offset < origin || extent <= 0 ||
          llvm::AddOverflow(origin, extent, end) || offset > end ||
          size > end - offset ||
          llvm::MulOverflow(pieceElements, size, pieceElements))
        return mlir::failure();
    }
    if (llvm::AddOverflow(coveredElements, pieceElements, coveredElements))
      return mlir::failure();
  }
  if (coveredElements != requestedElements)
    return mlir::failure();
  for (size_t left = 0; left < pieces.size(); ++left)
    for (size_t right = left + 1; right < pieces.size(); ++right) {
      bool disjoint = false;
      for (size_t axis = 0; axis < rank; ++axis) {
        int64_t leftEnd = pieces[left].resultWindow.offsets[axis] +
                          pieces[left].resultWindow.sizes[axis];
        int64_t rightEnd = pieces[right].resultWindow.offsets[axis] +
                           pieces[right].resultWindow.sizes[axis];
        disjoint |= leftEnd <= pieces[right].resultWindow.offsets[axis] ||
                    rightEnd <= pieces[left].resultWindow.offsets[axis];
      }
      if (!disjoint)
        return mlir::failure();
    }
  if (pieces.size() == 1 &&
      pieces.front().resultWindow.offsets == requested.offsets &&
      pieces.front().resultWindow.sizes == requested.sizes)
    return pieces.front().value;

  mlir::Value assembled = builder.create<mlir::tensor::EmptyOp>(
      location, resultType.getShape(), resultType.getElementType(),
      resultType.getEncoding());
  for (const auto &piece : pieces) {
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets, sizes, strides;
    for (auto [offset, origin, size] :
         llvm::zip_equal(piece.resultWindow.offsets, requested.offsets,
                         piece.resultWindow.sizes)) {
      offsets.push_back(builder.getIndexAttr(offset - origin));
      sizes.push_back(builder.getIndexAttr(size));
      strides.push_back(builder.getIndexAttr(1));
    }
    assembled = builder.create<mlir::tensor::InsertSliceOp>(
        location, piece.value, assembled, offsets, sizes, strides);
  }
  return assembled;
}

} // namespace wafer::compiler::detail
