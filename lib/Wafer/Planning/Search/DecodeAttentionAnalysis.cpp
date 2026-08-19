//===- DecodeAttentionAnalysis.cpp - Functional decode proof -----------===//

#include "AttentionAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseSet.h"

namespace wafer::tensor_program_alternatives {
namespace {

struct FunctionalCacheAppend {
  mlir::Value past;
  mlir::Value appended;
  mlir::Value updated;
  int64_t dimension = -1;
};

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

static mlir::func::FuncOp findSingleTensorProgram(mlir::ModuleOp module) {
  mlir::func::FuncOp found;
  bool multiple = false;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.isExternal())
      return;
    if (found) {
      multiple = true;
      return;
    }
    found = function;
  });
  return multiple ? mlir::func::FuncOp{} : found;
}

static bool isValueAncestor(mlir::Value ancestor, mlir::Value value) {
  llvm::SmallVector<mlir::Value, 16> worklist{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (current == ancestor)
      return true;
    if (!visited.insert(current).second)
      continue;
    if (mlir::Operation *definition = current.getDefiningOp())
      llvm::append_range(worklist, definition->getOperands());
  }
  return false;
}

static std::optional<int64_t> getStaticIndex(mlir::OpFoldResult value) {
  return mlir::getConstantIntValue(value);
}

static mlir::Value stripExactTensorCasts(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (visited.insert(value).second) {
    auto cast = value.getDefiningOp<mlir::tensor::CastOp>();
    if (!cast)
      return value;
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(cast.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(cast.getResult().getType());
    if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
        sourceType != resultType)
      return value;
    value = cast.getSource();
  }
  return {};
}

static bool hasStaticSlice(mlir::tensor::InsertSliceOp insert,
                           llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) {
  if (insert.getMixedOffsets().size() != offsets.size() ||
      insert.getMixedSizes().size() != sizes.size() ||
      insert.getMixedStrides().size() != sizes.size())
    return false;
  for (auto [actual, expected] :
       llvm::zip_equal(insert.getMixedOffsets(), offsets)) {
    std::optional<int64_t> value = getStaticIndex(actual);
    if (!value || *value != expected)
      return false;
  }
  for (auto [actual, expected] :
       llvm::zip_equal(insert.getMixedSizes(), sizes)) {
    std::optional<int64_t> value = getStaticIndex(actual);
    if (!value || *value != expected)
      return false;
  }
  for (mlir::OpFoldResult actual : insert.getMixedStrides()) {
    std::optional<int64_t> value = getStaticIndex(actual);
    if (!value || *value != 1)
      return false;
  }
  return true;
}

static std::optional<FunctionalCacheAppend>
matchFunctionalCacheAppend(mlir::Value returned) {
  mlir::Value updated = stripExactTensorCasts(returned);
  auto append = updated.getDefiningOp<mlir::tensor::InsertSliceOp>();
  if (!append)
    return std::nullopt;
  auto prefix = append.getDest().getDefiningOp<mlir::tensor::InsertSliceOp>();
  if (!prefix)
    return std::nullopt;
  mlir::Value past = prefix.getSource();
  mlir::Value appended = append.getSource();
  auto pastType = mlir::dyn_cast<mlir::RankedTensorType>(past.getType());
  auto appendedType =
      mlir::dyn_cast<mlir::RankedTensorType>(appended.getType());
  auto updatedType = mlir::dyn_cast<mlir::RankedTensorType>(updated.getType());
  if (!pastType || !appendedType || !updatedType ||
      !pastType.hasStaticShape() || !appendedType.hasStaticShape() ||
      !updatedType.hasStaticShape() || pastType.getRank() == 0 ||
      pastType.getRank() != appendedType.getRank() ||
      pastType.getRank() != updatedType.getRank() ||
      pastType.getElementType() != appendedType.getElementType() ||
      pastType.getElementType() != updatedType.getElementType() ||
      !mlir::isa<mlir::BlockArgument>(past) || isValueAncestor(past, appended))
    return std::nullopt;

  std::optional<int64_t> appendDimension;
  for (int64_t dim = 0; dim < updatedType.getRank(); ++dim) {
    int64_t pastExtent = pastType.getDimSize(dim);
    int64_t appendedExtent = appendedType.getDimSize(dim);
    int64_t updatedExtent = updatedType.getDimSize(dim);
    if (updatedExtent == pastExtent + appendedExtent &&
        pastExtent != updatedExtent && appendedExtent != updatedExtent) {
      if (appendDimension)
        return std::nullopt;
      appendDimension = dim;
      continue;
    }
    if (pastExtent != updatedExtent || appendedExtent != updatedExtent)
      return std::nullopt;
  }
  if (!appendDimension)
    return std::nullopt;

  llvm::SmallVector<int64_t, 6> prefixOffsets(updatedType.getRank(), 0);
  llvm::SmallVector<int64_t, 6> appendOffsets(updatedType.getRank(), 0);
  appendOffsets[*appendDimension] = pastType.getDimSize(*appendDimension);
  if (!hasStaticSlice(prefix, prefixOffsets, pastType.getShape()) ||
      !hasStaticSlice(append, appendOffsets, appendedType.getShape()))
    return std::nullopt;

  mlir::Value initial = prefix.getDest();
  if (!initial.getDefiningOp<mlir::tensor::EmptyOp>() &&
      !mlir::isa<mlir::BlockArgument>(initial))
    return std::nullopt;
  return FunctionalCacheAppend{past, appended, updated, *appendDimension};
}

} // namespace

mlir::FailureOr<DecodeAttentionSemantics>
analyzeDecodeAttentionSemantics(mlir::ModuleOp module,
                                std::string *failureReason) {
  auto fail =
      [&](llvm::StringRef reason) -> mlir::FailureOr<DecodeAttentionSemantics> {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  };
  mlir::FailureOr<AttentionSemantics> attention =
      analyzeAttentionSemantics(module, failureReason);
  if (mlir::failed(attention))
    return mlir::failure();
  mlir::func::FuncOp function = findSingleTensorProgram(module);
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());

  std::optional<FunctionalCacheAppend> keyAppend;
  std::optional<FunctionalCacheAppend> valueAppend;
  unsigned keyOutputIndex = 0;
  unsigned valueOutputIndex = 0;
  for (auto [index, returned] : llvm::enumerate(returnOp.getOperands())) {
    if (index == attention->outputIndex)
      continue;
    std::optional<FunctionalCacheAppend> append =
        matchFunctionalCacheAppend(returned);
    if (!append)
      continue;
    if (isValueAncestor(append->updated, attention->physicalValues)) {
      if (valueAppend)
        return fail("decode has multiple value-cache update results");
      valueAppend = *append;
      valueOutputIndex = index;
    }
    if (isValueAncestor(append->updated, attention->scores)) {
      if (keyAppend)
        return fail("decode has multiple key-cache update results");
      keyAppend = *append;
      keyOutputIndex = index;
    }
  }
  if (!keyAppend || !valueAppend)
    return fail("decode requires returned K/V appends consumed by attention");
  if (keyOutputIndex == valueOutputIndex)
    return fail("decode K/V cache updates must be distinct SSA results");

  auto keyPastType =
      mlir::cast<mlir::RankedTensorType>(keyAppend->past.getType());
  auto keyNewType =
      mlir::cast<mlir::RankedTensorType>(keyAppend->appended.getType());
  auto keyUpdatedType =
      mlir::cast<mlir::RankedTensorType>(keyAppend->updated.getType());
  auto valuePastType =
      mlir::cast<mlir::RankedTensorType>(valueAppend->past.getType());
  auto valueNewType =
      mlir::cast<mlir::RankedTensorType>(valueAppend->appended.getType());
  auto valueUpdatedType =
      mlir::cast<mlir::RankedTensorType>(valueAppend->updated.getType());
  if (keyAppend->dimension != valueAppend->dimension ||
      keyPastType.getShape() != valuePastType.getShape() ||
      keyNewType.getShape() != valueNewType.getShape() ||
      keyUpdatedType.getShape() != valueUpdatedType.getShape() ||
      keyUpdatedType.getElementType() != valueUpdatedType.getElementType())
    return fail("decode K/V cache append relations do not agree");

  const unsigned rank = attention->scoreType.getRank();
  const unsigned queryDimension = rank - 2;
  if (keyAppend->dimension != static_cast<int64_t>(queryDimension) ||
      keyUpdatedType.getRank() != rank ||
      keyUpdatedType.getDimSize(keyAppend->dimension) !=
          attention->reductionExtent ||
      keyNewType.getDimSize(keyAppend->dimension) !=
          attention->scoreType.getDimSize(queryDimension))
    return fail("decode query and cache-update domains do not agree");

  DecodeAttentionSemantics result;
  result.attention = *attention;
  result.pastKey = keyAppend->past;
  result.newKey = keyAppend->appended;
  result.updatedKey = keyAppend->updated;
  result.pastValue = valueAppend->past;
  result.newValue = valueAppend->appended;
  result.updatedValue = valueAppend->updated;
  result.keyOutputIndex = keyOutputIndex;
  result.valueOutputIndex = valueOutputIndex;
  result.cacheDimension = keyAppend->dimension;
  if (failureReason)
    failureReason->clear();
  return result;
}

} // namespace wafer::tensor_program_alternatives
