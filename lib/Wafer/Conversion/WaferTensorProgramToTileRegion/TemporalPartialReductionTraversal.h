//===- TemporalPartialReductionTraversal.h ------------------*- C++ -*-===//

#pragma once

#include "SingleRootTileRegionInternal.h"

namespace wafer::tensor_program_to_tile_region {

mlir::LogicalResult materializeTemporalPartialReductionShard(
    mlir::func::FuncOp function, uint32_t structuredNodeId,
    llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
    const StructuredNodeTemporalTile &selectedTemporal,
    unsigned &functionalArgumentCount,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
