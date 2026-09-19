//===- ConvertStablehloAttentionToLinalg.cpp - SDPA composite boundary ---===//

#include "Wafer/Conversion/Passes.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

#include <cmath>
#include <limits>

namespace wafer {
#define GEN_PASS_DEF_CONVERTSTABLEHLOATTENTIONTOLINALGPASS
#include "Wafer/Conversion/WaferConversionPasses.h.inc"

namespace {
#ifdef WAFER_ENABLE_STABLEHLO
struct AttentionSignature {
  mlir::RankedTensorType query, key, value, output, mask;
  mlir::FloatAttr scale;
  bool causal;
  int64_t groups;
  int64_t queryStart, keyStart, keyValidEnd;
};

mlir::FailureOr<AttentionSignature>
describeAttention(mlir::stablehlo::CompositeOp op) {
  auto attributes = op.getCompositeAttributes();
  auto causal = attributes.getAs<mlir::BoolAttr>("is_causal");
  auto hasMask = attributes.getAs<mlir::BoolAttr>("has_mask");
  auto gqa = attributes.getAs<mlir::BoolAttr>("enable_gqa");
  auto scale = attributes.getAs<mlir::FloatAttr>("scale");
  auto queryStart = attributes.getAs<mlir::IntegerAttr>("query_start");
  auto keyStart = attributes.getAs<mlir::IntegerAttr>("key_start");
  auto keyValidEnd = attributes.getAs<mlir::IntegerAttr>("key_valid_end");
  if (!causal || !hasMask || !gqa || !scale || !queryStart || !keyStart ||
      !keyValidEnd || !queryStart.getType().isInteger(64) ||
      !keyStart.getType().isInteger(64) ||
      !keyValidEnd.getType().isInteger(64) || queryStart.getInt() < 0 ||
      keyStart.getInt() < 0 || keyValidEnd.getInt() < keyStart.getInt() ||
      !std::isfinite(scale.getValueAsDouble()) || op.getVersion() != 0 ||
      attributes.size() != 7 || op.getNumResults() != 1 ||
      op.getNumOperands() < (hasMask.getValue() ? 4u : 3u))
    return mlir::failure();
  auto tensor = [](mlir::Value value) {
    return mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  };
  auto q = tensor(op.getOperand(0)), k = tensor(op.getOperand(1)),
       v = tensor(op.getOperand(2)), out = tensor(op.getResult(0));
  if (!q || !k || !v || !out || q.getRank() < 3 || k.getRank() != q.getRank() ||
      v.getRank() != q.getRank() || out.getRank() != q.getRank())
    return mlir::failure();
  for (auto type : {q, k, v, out})
    if (!type.hasStaticShape() ||
        llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }) ||
        type.getElementType() != q.getElementType())
      return mlir::failure();
  auto element = mlir::dyn_cast<mlir::FloatType>(q.getElementType());
  if (!element || (element.getWidth() != 16 && !element.isF32()))
    return mlir::failure();
  unsigned rank = q.getRank(), head = rank - 3;
  for (unsigned i = 0; i < head; ++i)
    if (q.getDimSize(i) != k.getDimSize(i) ||
        k.getDimSize(i) != v.getDimSize(i))
      return mlir::failure();
  if (q.getDimSize(rank - 1) != k.getDimSize(rank - 1) ||
      k.getDimSize(rank - 2) != v.getDimSize(rank - 2) ||
      k.getDimSize(head) != v.getDimSize(head) ||
      q.getDimSize(head) % k.getDimSize(head) != 0 ||
      out.getShape().drop_back() != q.getShape().drop_back() ||
      out.getDimSize(rank - 1) != v.getDimSize(rank - 1))
    return mlir::failure();
  int64_t groups = q.getDimSize(head) / k.getDimSize(head);
  if (keyValidEnd.getInt() - keyStart.getInt() > k.getDimSize(rank - 2) ||
      queryStart.getInt() >
          std::numeric_limits<int64_t>::max() - q.getDimSize(rank - 2) ||
      keyStart.getInt() >
          std::numeric_limits<int64_t>::max() - k.getDimSize(rank - 2))
    return mlir::failure();
  if (groups != 1 && !gqa.getValue())
    return mlir::failure();
  mlir::RankedTensorType mask;
  if (hasMask.getValue()) {
    mask = tensor(op.getOperand(3));
    if (!mask || !mask.hasStaticShape() || mask.getRank() > rank ||
        (!mlir::isa<mlir::FloatType>(mask.getElementType()) &&
         !mask.getElementType().isInteger(1)))
      return mlir::failure();
    for (unsigned i = 0; i < mask.getRank(); ++i) {
      unsigned axis = rank - mask.getRank() + i;
      int64_t extent =
          axis == rank - 1 ? k.getDimSize(rank - 2) : q.getDimSize(axis);
      if (mask.getDimSize(i) != 1 && mask.getDimSize(i) != extent)
        return mlir::failure();
    }
  }
  // PyTorch/XLA lifts decomposition scalar constants as trailing captures.
  // They remain part of the source definition, not additional tensor roles.
  for (mlir::Value capture : op.getInputs().drop_front(mask ? 4 : 3)) {
    auto type = tensor(capture);
    if (!type || type.getRank() != 0)
      return mlir::failure();
  }
  return AttentionSignature{q,
                            k,
                            v,
                            out,
                            mask,
                            scale,
                            causal.getValue(),
                            groups,
                            queryStart.getInt(),
                            keyStart.getInt(),
                            keyValidEnd.getInt()};
}

mlir::Value widen(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value) {
  if (value.getType().isF32())
    return value;
  if (mlir::cast<mlir::FloatType>(value.getType()).getWidth() < 32)
    return builder.create<mlir::arith::ExtFOp>(loc, builder.getF32Type(),
                                               value);
  return builder.create<mlir::arith::TruncFOp>(loc, builder.getF32Type(),
                                               value);
}

void convertAttention(mlir::stablehlo::CompositeOp source,
                      const AttentionSignature &signature,
                      mlir::IRRewriter &builder) {
  auto loc = source.getLoc();
  builder.setInsertionPoint(source);
  const unsigned batchRank = signature.query.getRank() - 2;
  const bool grouped = signature.groups != 1;
  const unsigned queryAxis = batchRank + grouped;
  const unsigned contractionAxis = queryAxis + 1;
  const unsigned keyAxis = queryAxis + 2;
  const unsigned valueAxis = queryAxis + 3;
  const unsigned domainRank = valueAxis + 1;
  auto dim = [&](unsigned axis) { return builder.getAffineDimExpr(axis); };
  auto map = [&](llvm::ArrayRef<mlir::AffineExpr> coordinates) {
    return mlir::AffineMap::get(domainRank, 0, coordinates,
                                builder.getContext());
  };
  llvm::SmallVector<mlir::AffineExpr> queryCoordinates, keyCoordinates;
  for (unsigned i = 0; i < batchRank; ++i) {
    queryCoordinates.push_back(dim(i));
    keyCoordinates.push_back(dim(i));
  }
  if (grouped)
    queryCoordinates.push_back(dim(batchRank));
  auto outputCoordinates = queryCoordinates;
  auto valueCoordinates = keyCoordinates;
  queryCoordinates.append({dim(queryAxis), dim(contractionAxis)});
  keyCoordinates.append({dim(keyAxis), dim(contractionAxis)});
  valueCoordinates.append({dim(keyAxis), dim(valueAxis)});
  outputCoordinates.append({dim(queryAxis), dim(valueAxis)});
  llvm::SmallVector<mlir::AffineMap> maps{map(queryCoordinates),
                                          map(keyCoordinates),
                                          map(valueCoordinates), map({})};
  auto splitHeads = [&](mlir::RankedTensorType type) {
    llvm::SmallVector<int64_t> shape(type.getShape());
    if (grouped) {
      shape[batchRank - 1] /= signature.groups;
      shape.insert(shape.begin() + batchRank, signature.groups);
    }
    return mlir::RankedTensorType::get(shape, type.getElementType());
  };
  auto reshape = [&](mlir::Value value,
                     mlir::RankedTensorType type) -> mlir::Value {
    if (value.getType() == type)
      return value;
    return builder.create<mlir::stablehlo::ReshapeOp>(loc, type, value);
  };
  mlir::Value query =
      reshape(source.getOperand(0), splitHeads(signature.query));
  mlir::Value mask;
  if (signature.mask) {
    llvm::SmallVector<int64_t> shape;
    llvm::SmallVector<mlir::AffineExpr> coordinates;
    for (unsigned i = 0; i < signature.mask.getRank(); ++i) {
      int64_t size = signature.mask.getDimSize(i);
      if (size == 1)
        continue;
      unsigned axis = signature.query.getRank() - signature.mask.getRank() + i;
      if (grouped && axis == batchRank - 1) {
        shape.append({size / signature.groups, signature.groups});
        coordinates.append({dim(axis), dim(batchRank)});
      } else {
        shape.push_back(size);
        coordinates.push_back(dim(axis < batchRank    ? axis
                                  : axis == batchRank ? queryAxis
                                                      : keyAxis));
      }
    }
    mask = reshape(
        source.getOperand(3),
        mlir::RankedTensorType::get(shape, signature.mask.getElementType()));
    maps.push_back(map(coordinates));
  }
  maps.push_back(map(outputCoordinates));
  auto outputType = splitHeads(signature.output);
  mlir::Value output = builder.create<mlir::tensor::EmptyOp>(
      loc, outputType.getShape(), outputType.getElementType());
  mlir::Value scale = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF32FloatAttr(signature.scale.getValueAsDouble()));
  llvm::SmallVector<mlir::Value> positions;
  mlir::AffineMapAttr positionMap;
  if (signature.causal ||
      signature.keyValidEnd - signature.keyStart !=
          signature.key.getDimSize(signature.key.getRank() - 2)) {
    for (int64_t position :
         {signature.queryStart, signature.keyStart, signature.keyValidEnd})
      positions.push_back(
          builder.create<mlir::arith::ConstantIndexOp>(loc, position));
    positionMap = mlir::AffineMapAttr::get(map({dim(queryAxis), dim(keyAxis)}));
  }
  auto attention = builder.create<LinalgExtAttentionOp>(
      loc, mlir::TypeRange{outputType}, query, source.getOperand(1),
      source.getOperand(2), scale, mask, output,
      AttentionAlgorithm::FlashAttention, builder.getAffineMapArrayAttr(maps),
      positions, signature.causal, true, positionMap);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *block = builder.createBlock(&attention.getScoreRegion());
    mlir::Value dot = block->addArgument(builder.getF32Type(), loc);
    mlir::Value scaling = block->addArgument(builder.getF32Type(), loc);
    mlir::Value score = builder.create<mlir::arith::MulFOp>(loc, dot, scaling);
    if (mask) {
      auto maskType = signature.mask.getElementType();
      mlir::Value maskScalar = block->addArgument(maskType, loc);
      if (maskType.isInteger(1)) {
        mlir::Value negativeInfinity = builder.create<mlir::arith::ConstantOp>(
            loc,
            builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
        score = builder.create<mlir::arith::SelectOp>(loc, maskScalar, score,
                                                      negativeInfinity);
      } else {
        score = builder.create<mlir::arith::AddFOp>(
            loc, score, widen(builder, loc, maskScalar));
      }
    }
    builder.create<LinalgExtAttentionYieldOp>(loc, score);
  }
  builder.replaceOp(source, reshape(attention.getResult(0), signature.output));
}
#endif

struct ConvertStablehloAttentionToLinalgPass final
    : impl::ConvertStablehloAttentionToLinalgPassBase<
          ConvertStablehloAttentionToLinalgPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    impl::ConvertStablehloAttentionToLinalgPassBase<
        ConvertStablehloAttentionToLinalgPass>::getDependentDialects(registry);
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }
  void runOnOperation() override {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<
        std::pair<mlir::stablehlo::CompositeOp, AttentionSignature>>
        work;
    bool invalid = false;
    getOperation().walk([&](mlir::stablehlo::CompositeOp op) {
      if (op.getName() != "wafer.scaled_dot_product_attention")
        return;
      auto signature = describeAttention(op);
      if (mlir::failed(signature)) {
        op.emitError("unsupported or incomplete scaled dot-product attention "
                     "composite contract");
        invalid = true;
      } else {
        work.emplace_back(op, *signature);
      }
    });
    if (invalid)
      return signalPassFailure();
    mlir::IRRewriter rewriter(&getContext());
    for (auto &[op, signature] : work)
      convertAttention(op, signature, rewriter);
#endif
  }
};
} // namespace
} // namespace wafer
