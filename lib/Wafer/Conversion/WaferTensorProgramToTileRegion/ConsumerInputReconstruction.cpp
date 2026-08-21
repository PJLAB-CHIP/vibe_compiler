//===- ConsumerInputReconstruction.cpp --------------------------------===//

#include "ConsumerInputReconstruction.h"
#include "Internal.h"

#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::tensor_program_to_tile_region {
namespace {

mlir::LogicalResult fail(std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

bool matchesOperationKind(const analysis::TensorTransform &step) {
  auto indexing = mlir::dyn_cast_or_null<wafer::WaferTensorIndexingOpInterface>(
      step.operation);
  if (!indexing)
    return false;
  mlir::FailureOr<wafer::TensorIndexingDescription> description =
      indexing.getTensorIndexingDescription(step.result);
  if (mlir::failed(description))
    return false;
  switch (description->kind) {
  case wafer::TensorIndexingTransformKind::ExpandShape:
    return step.kind == analysis::TensorTransformKind::ExpandShape;
  case wafer::TensorIndexingTransformKind::CollapseShape:
    return step.kind == analysis::TensorTransformKind::CollapseShape;
  case wafer::TensorIndexingTransformKind::ExtractSlice:
    return step.kind == analysis::TensorTransformKind::ExtractSlice;
  case wafer::TensorIndexingTransformKind::InsertSlice:
    return step.kind == analysis::TensorTransformKind::InsertSlice;
  case wafer::TensorIndexingTransformKind::Pad:
    return step.kind == analysis::TensorTransformKind::Pad;
  case wafer::TensorIndexingTransformKind::Cast:
    return step.kind == analysis::TensorTransformKind::Cast;
  }
  return false;
}

} // namespace

mlir::LogicalResult reconstructConsumerInput(
    mlir::Operation *consumer, unsigned consumerOperand, TileId currentTile,
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
    llvm::ArrayRef<ProducerValue> producerValues, std::string *failureReason) {
  auto grouped = llvm::find_if(
      operandDemands, [&](const analysis::DependencyDemand &demand) {
        return demand.consumerOperation == consumer &&
               demand.consumerOperand == consumerOperand;
      });
  if (grouped == operandDemands.end())
    return fail(failureReason,
                "consumer input dependency has no grouped exact-demand recipe");
  auto recipe =
      llvm::find_if(grouped->perDestination,
                    [&](const analysis::DestinationDemand &candidate) {
                      return candidate.destinationTile == currentTile;
                    });
  if (recipe == grouped->perDestination.end())
    return fail(failureReason, "consumer input dependency recipe omitted the "
                               "current destination Tile");

  mlir::IRMapping mapping;
  mlir::OpBuilder builder(consumer);
  bool hasStructuredBoundary = false;
  for (const analysis::SourceDemand &boundary : recipe->sources) {
    const auto *structured =
        std::get_if<analysis::StructuredResultSource>(&boundary.source);
    if (!structured)
      continue;
    hasStructuredBoundary = true;
    if (!structured->operation ||
        structured->result >= structured->operation->getNumResults())
      return fail(
          failureReason,
          "consumer input reconstruction has a malformed producer boundary");
    mlir::Value source = structured->operation->getResult(structured->result);
    mlir::Value replacement;
    if (boundary.requiredDomain.isEmpty()) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
      if (!type || !type.hasStaticShape())
        return fail(failureReason, "empty producer value requirement requires "
                                   "one static tensor result");
      replacement =
          builder
              .create<mlir::tensor::EmptyOp>(
                  consumer->getLoc(), type.getShape(), type.getElementType(),
                  mlir::ValueRange{}, type.getEncoding())
              .getResult();
    } else {
      auto materialized =
          llvm::find_if(producerValues, [&](const ProducerValue &candidate) {
            return candidate.producer == structured->operation &&
                   candidate.producerResult == structured->result &&
                   candidate.consumer == consumer &&
                   candidate.consumerOperand == consumerOperand &&
                   candidate.destinationTile == currentTile;
          });
      if (materialized == producerValues.end())
        return fail(failureReason, "nonempty producer value requirement has no "
                                   "verified physical carrier");
      replacement = materialized->value;
      if (replacement.getType() != source.getType())
        return fail(failureReason, "producer value requirement carrier does "
                                   "not preserve the tensor type");
    }
    if (mapping.contains(source) && mapping.lookup(source) != replacement)
      return fail(
          failureReason,
          "producer value requirement has conflicting materializations");
    if (!mapping.contains(source))
      mapping.map(source, replacement);
  }

  mlir::Operation *previousStep = nullptr;
  for (const analysis::TensorTransform &step : recipe->reconstruction.steps) {
    if (!step.operation || step.result >= step.operation->getNumResults() ||
        (previousStep && !previousStep->isBeforeInBlock(step.operation)))
      return fail(failureReason,
                  "consumer input reconstruction is not in stable SSA order");
    if (!matchesOperationKind(step))
      return fail(failureReason, "consumer input reconstruction action "
                                 "disagrees with its immutable operation");
    for (mlir::Value operand : step.operation->getOperands())
      if (!mapping.contains(operand))
        mapping.map(operand, operand);
    builder.clone(*step.operation, mapping);
    if (!mapping.lookupOrNull(step.operation->getResult(step.result)))
      return fail(failureReason,
                  "consumer input reconstruction step did not map its result");
    previousStep = step.operation;
  }

  if (!hasStructuredBoundary && recipe->reconstruction.steps.empty())
    return mlir::success();

  mlir::Value replacement =
      mapping.lookupOrNull(consumer->getOperand(consumerOperand));
  if (!replacement)
    return fail(failureReason, "consumer input reconstruction does not "
                               "reconstruct the selected consumer operand");
  consumer->setOperand(consumerOperand, replacement);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
