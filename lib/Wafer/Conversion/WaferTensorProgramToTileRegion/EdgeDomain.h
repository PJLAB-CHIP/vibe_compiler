//===- EdgeDomain.h - Selected edge logical domains -*- C++ -*-===//
#pragma once

#include "Internal.h"

namespace wafer::tensor_program_to_tile_region {

mlir::LogicalResult validateEdge(mlir::Operation *producer,
                                 unsigned producerResult,
                                 mlir::Operation *consumer,
                                 unsigned consumerOperand, mlir::Block &body,
                                 std::string *failureReason,
                                 bool dependencyAlreadyValidated);

mlir::LogicalResult validateStaticDomain(mlir::RankedTensorType type,
                                         llvm::ArrayRef<int64_t> offsets,
                                         llvm::ArrayRef<int64_t> sizes,
                                         std::string *failureReason,
                                         llvm::StringRef subject);

mlir::LogicalResult
validateDemandIndexRelation(mlir::Operation *consumer, unsigned consumerOperand,
                            llvm::ArrayRef<int64_t> consumerOffsets,
                            llvm::ArrayRef<int64_t> consumerSizes,
                            llvm::ArrayRef<int64_t> producerOffsets,
                            llvm::ArrayRef<int64_t> producerSizes,
                            std::string *failureReason);

mlir::LogicalResult deriveConsumerDomainFromProducerDemand(
    mlir::Operation *producer, unsigned producerResult,
    mlir::Operation *consumer, unsigned consumerOperand,
    llvm::ArrayRef<int64_t> producerOffsets,
    llvm::ArrayRef<int64_t> producerSizes,
    llvm::SmallVectorImpl<int64_t> &consumerOffsets,
    llvm::SmallVectorImpl<int64_t> &consumerSizes, std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
