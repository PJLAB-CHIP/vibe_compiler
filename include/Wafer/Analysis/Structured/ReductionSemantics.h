//===- ReductionSemantics.h - Structured reduction legality -*- C++ -*-===//
#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace wafer::analysis {

/// Proves whether partitioning and merging one current Linalg reduction
/// preserves the repository numeric contract. It reads only the current
/// scalar region and never mutates IR or consumes fast-math as a policy switch.
mlir::LogicalResult
verifyReductionPartitionLegality(mlir::linalg::LinalgOp reduction,
                                 bool preservesSequentialReductionOrder,
                                 std::string *failureReason = nullptr);

} // namespace wafer::analysis
