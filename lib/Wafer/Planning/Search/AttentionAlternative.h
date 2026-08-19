//===- AttentionAlternative.h - Structured attention alternative -*- C++ -*-===//
#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <string>

namespace wafer::tensor_program_alternatives {

/// A graph-level attention algorithm proven from the current TensorProgram.
/// Temporal tiling of the resulting structured graph remains a later choice.
enum class AttentionProgramAlgorithm : uint8_t {
  Online,
  SplitKeyValue,
};

/// Replaces one proven materialized-softmax attention graph with a pure tensor
/// online recurrence. The function signature and observable results remain
/// unchanged; no Tile, memory-space, movement, or scheduling fact is created.
mlir::LogicalResult materializeAttentionProgramAlternative(
    mlir::func::FuncOp function, AttentionProgramAlgorithm algorithm,
    int64_t keyValueBlockSize, int64_t keyValuePartitionCount,
    std::string *failureReason = nullptr);

} // namespace wafer::tensor_program_alternatives
