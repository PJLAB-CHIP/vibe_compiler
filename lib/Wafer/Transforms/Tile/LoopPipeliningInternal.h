//===- LoopPipeliningInternal.h - Tile LoopPipeliningInternal -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LOOPPIPELININGINTERNAL_H
#define WAFER_TRANSFORMS_TILE_LOOPPIPELININGINTERNAL_H

#include "Wafer/Transforms/Tile/LoopPipelining.h"
#include "llvm/ADT/StringRef.h"

namespace wafer::compiler::detail {
std::optional<uint64_t> getStaticTripCount(mlir::scf::ForOp loop);
std::optional<LoopPipeliningFailure>
checkLoopPipeliningDomain(mlir::scf::ForOp loop, uint32_t stageCount);

PipelinedModuleResult
materializationFailure(LoopPipeliningFailureKind kind, llvm::StringRef detail,
                       std::optional<uint32_t> pipeline = std::nullopt);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_LOOPPIPELININGINTERNAL_H
