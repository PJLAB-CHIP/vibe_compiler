//===- OperandReuse.cpp - Current operand access invariance ---------------===//

#include "OperandReuse.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

namespace wafer::compiler::detail {

std::optional<llvm::SmallVector<OperandProjection, 4>>
getReadOperandProjections(mlir::Operation *operation,
                          llvm::ArrayRef<int64_t> iterationExtents) {
  auto root = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!root)
    return std::nullopt;
  llvm::SmallVector<int64_t> domain(iterationExtents);
  if (domain.empty())
    domain = root.getStaticLoopRanges();
  if (llvm::any_of(domain, [](int64_t extent) { return extent <= 0; }))
    return std::nullopt;
  llvm::SmallVector<int64_t> zero(domain.size(), 0);
  llvm::SmallVector<OperandProjection, 4> projections;
  unsigned work = 0;
  std::function<bool(mlir::Value, const analysis::IndexRelation &, unsigned)>
      visit;
  visit = [&](mlir::Value value, const analysis::IndexRelation &relation,
              unsigned depth) {
    if (++work > 128 || depth > 16)
      return false;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
      if (auto input = analysis::getSingleExecutionRegionEntryOperand(argument))
        return visit(input, relation, depth + 1);
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (auto producer =
              mlir::dyn_cast<mlir::linalg::LinalgOp>(result.getOwner())) {
        if (!type || !producer.hasPureTensorSemantics())
          return false;
        auto resultMap = analysis::getStructuredResultMap(result);
        auto producerShape = producer.getStaticLoopRanges();
        if (mlir::failed(resultMap) ||
            llvm::any_of(producerShape, [](int64_t size) { return size <= 0; }))
          return false;
        for (mlir::OpOperand *input : producer.getDpsInputOperands()) {
          if (!producer.payloadUsesValueFromOperand(input))
            continue;
          auto inputType =
              mlir::dyn_cast<mlir::RankedTensorType>(input->get().getType());
          llvm::ArrayRef<int64_t> shape =
              inputType ? inputType.getShape() : llvm::ArrayRef<int64_t>{};
          auto operandMap = analysis::getStructuredOperandMap(*input);
          if (mlir::failed(operandMap))
            return false;
          auto edge = analysis::IndexRelation::fromCommonIterationDomain(
              *resultMap, type.getShape(), *operandMap, shape, producerShape);
          if (!edge.isExact())
            return false;
          auto composed = relation.compose(*edge.get());
          if (!composed.isExact() ||
              !visit(input->get(), *composed.get(), depth + 1))
            return false;
        }
        return true;
      }
      if (auto support = analysis::deriveTensorResultIndexing(result);
          support.isExact() && support.indexing->operands.size() == 1 &&
          support.indexing->operands.front().role ==
              TensorIndexingOperandRole::Source) {
        for (const auto &operand : support.indexing->operands) {
          auto composed = relation.compose(operand.resultToOperand);
          if (!composed.isExact() ||
              !visit(result.getOwner()->getOperand(operand.operand),
                     *composed.get(), depth + 1))
            return false;
        }
        return true;
      }
    }
    auto element = type ? type.getElementType() : value.getType();
    if (!element.isIntOrFloat())
      return false;
    auto image = relation.getExactStaticRectangularImage(zero, domain);
    OperandProjection projection;
    if (image.isExact()) {
      projection.offsets = image.domain->offsets;
      projection.sizes = image.domain->sizes;
    } else if (type && type.hasStaticShape()) {
      // Strided images can have holes while still proving invariance. Keep
      // the original logical-size ranking weight, but do not advertise an
      // exact shared source window for coordination.
      projection.sizes.assign(type.getShape().begin(), type.getShape().end());
    } else {
      return false;
    }
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
      if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(
              argument.getOwner()->getParentOp()))
        if (auto identity = function.getArgAttrOfType<ProgramArgumentAttr>(
                argument.getArgNumber(), kWaferProgramArgumentAttrName))
          if (image.isExact())
            projection.programArgument = identity.getIndex();
    projection.bytes = (element.getIntOrFloatBitWidth() + 7) / 8;
    for (auto size : projection.sizes)
      projection.bytes =
          llvm::SaturatingMultiply(projection.bytes, uint64_t(size));
    projection.iterators.resize(domain.size(), true);
    for (unsigned axis = 0; axis < domain.size(); ++axis)
      if (relation.isInvariantOnDestinationDimension(domain, axis)
              .isProvenTrue())
        projection.iterators.reset(axis);
    projections.push_back(std::move(projection));
    return true;
  };
  // Compose through current pure producers: a final cast or view does not
  // erase the input invariance of a reduction producer. Inits are mutable
  // accumulation state, not read-only source data.
  for (mlir::OpOperand *input : root.getDpsInputOperands()) {
    if (!root.payloadUsesValueFromOperand(input))
      continue;
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(input->get().getType());
    auto map = analysis::getStructuredOperandMap(*input);
    if (mlir::failed(map))
      return std::nullopt;
    auto relation = analysis::IndexRelation::fromAffineMap(
        *map, domain, type ? type.getShape() : llvm::ArrayRef<int64_t>{});
    if (!relation.isExact() || !visit(input->get(), *relation.get(), 0))
      return std::nullopt;
  }
  return projections;
}

} // namespace wafer::compiler::detail
