//===- NormalizeTensorReads.cpp - Explicit projected tensor inputs ------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>

namespace wafer {

#define GEN_PASS_DEF_NORMALIZELINALGTENSORREADSPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

mlir::AffineMap getProjectedReadMap(mlir::tensor::ExtractOp read,
                                    mlir::linalg::GenericOp generic,
                                    llvm::ArrayRef<int64_t> loopSizes) {
  auto type =
      mlir::dyn_cast<mlir::RankedTensorType>(read.getTensor().getType());
  if (!type || !type.hasStaticShape() ||
      generic.getRegion().isAncestor(read.getTensor().getParentRegion()))
    return {};

  llvm::SmallVector<mlir::AffineExpr> coordinates;
  for (auto [index, extent] :
       llvm::zip_equal(read.getIndices(), type.getShape())) {
    if (auto dimension = index.getDefiningOp<mlir::linalg::IndexOp>()) {
      if (dimension->getParentOp() != generic ||
          dimension.getDim() >= loopSizes.size() ||
          extent != loopSizes[dimension.getDim()])
        return {};
      coordinates.push_back(
          mlir::getAffineDimExpr(dimension.getDim(), generic.getContext()));
    } else if (extent == 1 && mlir::matchPattern(index, mlir::m_Zero())) {
      coordinates.push_back(
          mlir::getAffineConstantExpr(0, generic.getContext()));
    } else {
      return {};
    }
  }
  return mlir::AffineMap::get(loopSizes.size(), 0, coordinates,
                              generic.getContext());
}

struct ReadNormalizationResult {
  unsigned normalizedReads = 0;
  unsigned addedInputs = 0;
};

ReadNormalizationResult normalizeTensorReads(mlir::linalg::GenericOp generic,
                                             mlir::IRRewriter &rewriter) {
  if (!generic.hasPureTensorSemantics())
    return {};
  llvm::SmallVector<mlir::tensor::ExtractOp> reads;
  for (auto read : generic.getBody()->getOps<mlir::tensor::ExtractOp>())
    if (!read->use_empty())
      reads.push_back(read);
  if (reads.empty())
    return {};

  auto loopSizes = generic.getStaticLoopRanges();
  if (llvm::any_of(loopSizes, mlir::ShapedType::isDynamic))
    return {};
  auto maps = generic.getIndexingMapsArray();
  unsigned originalInputs = generic.getNumDpsInputs();
  struct InputBinding {
    mlir::Value tensor;
    mlir::AffineMap map;
  };
  llvm::SmallVector<InputBinding> inputs;
  for (auto [index, input] : llvm::enumerate(generic.getInputs()))
    inputs.push_back({input, maps[index]});
  struct ReadBinding {
    mlir::tensor::ExtractOp read;
    unsigned input;
  };
  llvm::SmallVector<ReadBinding> replacements;
  for (auto read : reads) {
    auto map = getProjectedReadMap(read, generic, loopSizes);
    if (!map)
      continue;
    auto input = llvm::find_if(inputs, [&](const InputBinding &binding) {
      return binding.tensor == read.getTensor() && binding.map == map;
    });
    unsigned inputNumber = std::distance(inputs.begin(), input);
    if (input == inputs.end())
      inputs.push_back({read.getTensor(), map});
    replacements.push_back({read, inputNumber});
  }
  if (replacements.empty())
    return {};

  // All maps and block-argument positions are known before the first edit.
  // Only input arguments can be reused: an init argument may carry an updated
  // reduction value, whereas a captured tensor is immutable input SSA.
  unsigned addedInputs = inputs.size() - originalInputs;
  if (addedInputs != 0) {
    llvm::SmallVector<mlir::Value> addedValues;
    llvm::SmallVector<mlir::AffineMap> addedMaps;
    for (const InputBinding &input : llvm::drop_begin(inputs, originalInputs)) {
      addedValues.push_back(input.tensor);
      addedMaps.push_back(input.map);
    }
    maps.insert(maps.begin() + originalInputs, addedMaps.begin(),
                addedMaps.end());
    rewriter.modifyOpInPlace(generic, [&] {
      generic.getInputsMutable().append(addedValues);
      generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      for (auto [index, value] : llvm::enumerate(addedValues))
        generic.getBody()->insertArgument(
            originalInputs + index,
            mlir::cast<mlir::RankedTensorType>(value.getType())
                .getElementType(),
            value.getLoc());
    });
  }
  for (const ReadBinding &binding : replacements)
    rewriter.replaceOp(binding.read,
                       generic.getBody()->getArgument(binding.input));
  return {static_cast<unsigned>(replacements.size()), addedInputs};
}

struct NormalizeLinalgTensorReadsPass final
    : impl::NormalizeLinalgTensorReadsPassBase<NormalizeLinalgTensorReadsPass> {
  void runOnOperation() override {
    auto function = getOperation();
    if (mlir::failed(mlir::verify(function))) {
      signalPassFailure();
      return;
    }
    mlir::IRRewriter rewriter(function.getContext());
    function.walk([&](mlir::linalg::GenericOp generic) {
      auto result = normalizeTensorReads(generic, rewriter);
      numNormalizedReads += result.normalizedReads;
      numAddedInputs += result.addedInputs;
    });
    if (mlir::failed(mlir::verify(function)))
      signalPassFailure();
  }
};

} // namespace
} // namespace wafer
