//===- ConsumerInputReconstruction.h ------------------------*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace wafer::tensor_program_to_tile_region {

/// One already-verified physical value for a nonempty producer value
/// requirement.
struct ProducerValue {
  mlir::Operation *producer = nullptr;
  unsigned producerResult = 0;
  mlir::Operation *consumer = nullptr;
  unsigned consumerOperand = 0;
  TileId destinationTile{0};
  mlir::Value value;
};

/// Applies one exact consumer input reconstruction to one consumer operand.
/// Carrier construction and structured-root traversal are deliberately outside
/// this boundary.
mlir::LogicalResult reconstructConsumerInput(
    mlir::Operation *consumer, unsigned consumerOperand, TileId currentTile,
    llvm::ArrayRef<analysis::ConsumerInputDemand> operandDemands,
    llvm::ArrayRef<ProducerValue> producerValues, std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
