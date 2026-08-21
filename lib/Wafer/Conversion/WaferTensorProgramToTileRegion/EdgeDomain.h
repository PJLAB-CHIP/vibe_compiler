//===- EdgeDomain.h - Selected edge logical domains -*- C++ -*-===//
#pragma once

#include "Internal.h"

namespace wafer::tensor_program_to_tile_region {

mlir::LogicalResult validateEdge(mlir::Operation *producer,
                                 unsigned producerResult,
                                 mlir::Operation *consumer,
                                 unsigned consumerOperand, mlir::Block &body,
                                 std::string *failureReason);

mlir::LogicalResult validateStaticDomain(mlir::RankedTensorType type,
                                         llvm::ArrayRef<int64_t> offsets,
                                         llvm::ArrayRef<int64_t> sizes,
                                         std::string *failureReason,
                                         llvm::StringRef subject);

} // namespace wafer::tensor_program_to_tile_region
