//===- ConsumerInputReconstruction.cpp --------------------------------===//

#include "ConsumerInputReconstruction.h"
#include "Internal.h"

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
  switch (step.kind) {
  case analysis::TensorTransformKind::ExpandShape:
    return mlir::isa<mlir::tensor::ExpandShapeOp>(step.operation);
  case analysis::TensorTransformKind::CollapseShape:
    return mlir::isa<mlir::tensor::CollapseShapeOp>(step.operation);
  case analysis::TensorTransformKind::ExtractSlice:
    return mlir::isa<mlir::tensor::ExtractSliceOp>(step.operation);
  case analysis::TensorTransformKind::InsertSlice:
    return mlir::isa<mlir::tensor::InsertSliceOp>(step.operation);
  case analysis::TensorTransformKind::Pad:
    return mlir::isa<mlir::tensor::PadOp>(step.operation);
  case analysis::TensorTransformKind::Cast:
    return mlir::isa<mlir::tensor::CastOp>(step.operation);
  }
  llvm_unreachable("unknown tensor-operand operation kind");
}

} // namespace

mlir::LogicalResult reconstructConsumerInput(
    mlir::Operation *consumer, unsigned consumerOperand, TileId currentTile,
    llvm::ArrayRef<analysis::ConsumerInputDemand> operandDemands,
    llvm::ArrayRef<ProducerValue> producerValues, std::string *failureReason) {
  auto grouped = llvm::find_if(
      operandDemands, [&](const analysis::ConsumerInputDemand &demand) {
        return demand.consumer == consumer &&
               demand.consumerOperand == consumerOperand;
      });
  if (grouped == operandDemands.end())
    return fail(failureReason,
                "consumer input dependency has no grouped exact-demand recipe");
  auto recipe = llvm::find_if(
      grouped->perDestination,
      [&](const analysis::ConsumerInputReconstruction &candidate) {
        return candidate.destinationTile == currentTile;
      });
  if (recipe == grouped->perDestination.end() || !recipe->operandDemand)
    return fail(failureReason, "consumer input dependency recipe omitted the "
                               "current destination Tile");
  if (recipe->steps.empty())
    return fail(failureReason,
                "consumer input dependency recipe has no reconstruction step");

  mlir::IRMapping mapping;
  mlir::OpBuilder builder(consumer);
  for (const analysis::ProducerValueRequirement &boundary :
       recipe->boundaries) {
    if (!boundary.producer || !boundary.requiredDomain ||
        boundary.producerResult >= boundary.producer->getNumResults())
      return fail(
          failureReason,
          "consumer input reconstruction has a malformed producer boundary");
    mlir::Value source = boundary.producer->getResult(boundary.producerResult);
    mlir::Value replacement;
    if (boundary.requiredDomain->isIntegerEmpty()) {
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
            return candidate.producer == boundary.producer &&
                   candidate.producerResult == boundary.producerResult &&
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
  for (const analysis::TensorTransform &step : recipe->steps) {
    if (!step.operation || !step.outputDemand ||
        step.result >= step.operation->getNumResults() ||
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

  mlir::Value replacement =
      mapping.lookupOrNull(consumer->getOperand(consumerOperand));
  if (!replacement)
    return fail(failureReason, "consumer input reconstruction does not "
                               "reconstruct the selected consumer operand");
  consumer->setOperand(consumerOperand, replacement);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
