//===- LoopPipelining.h - Tile LoopPipelining -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LOOPPIPELINING_H
#define WAFER_TRANSFORMS_TILE_LOOPPIPELINING_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {
struct LoopPipeliningLimits {
  uint64_t maxFiniteUnrolledOperations = 4096;
};

enum class LoopPipeliningFailureKind : uint8_t {
  BrokenContract,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

struct LoopPipeliningFailure {
  LoopPipeliningFailureKind kind = LoopPipeliningFailureKind::CompilerBug;
  std::optional<uint32_t> pipeline;
  std::string detail;
};

enum class TilePipelineLowering : uint8_t {
  SCFDistanceOne,
  FiniteUnrolled,
};

struct TilePipelineOperation {
  mlir::Operation *operation = nullptr;
  uint32_t stage = 0;
};

/// One invocation-local choice bound directly to current IR. The operation
/// list must cover all top-level operations in `loop` exactly once. No event,
/// storage-object or future occurrence identity crosses this boundary.
struct TilePipelineChoice {
  mlir::scf::ForOp loop;
  std::vector<TilePipelineOperation> operations;
  TilePipelineLowering lowering = TilePipelineLowering::SCFDistanceOne;
};

struct PreparedTilePipeline {
  mlir::scf::ForOp loop;
  std::vector<TilePipelineOperation> operations;
  TilePipelineLowering lowering = TilePipelineLowering::SCFDistanceOne;
  std::optional<uint64_t> tripCount;
  uint32_t stageCount = 0;
};

struct PreparedLoopPipelines {
  mlir::ModuleOp module;
  std::vector<PreparedTilePipeline> pipelines;
  LoopPipeliningLimits limits;
};

struct PreparedLoopPipelinesResult {
  std::optional<PreparedLoopPipelines> prepared;
  std::optional<LoopPipeliningFailure> failure;

  bool succeeded() const { return prepared.has_value(); }
};

enum class PipelinePhase : uint8_t {
  Prologue,
  Kernel,
  Epilogue,
};

struct PipelinedOperation {
  mlir::Operation *operation = nullptr;
  uint32_t stage = 0;
  PipelinePhase phase = PipelinePhase::Kernel;
  uint64_t staticIteration = 0;
};

struct PipelinedLoop {
  uint32_t stageCount = 0;
  std::optional<uint64_t> kernelTripCount;
  uint64_t originalOperationCount = 0;
  std::vector<PipelinedOperation> operations;
};

struct PipelinedModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<PipelinedLoop> pipelines;
};

struct PipelinedModuleResult {
  std::optional<PipelinedModule> materialized;
  std::optional<LoopPipeliningFailure> failure;

  bool succeeded() const { return materialized.has_value(); }
};

PreparedLoopPipelinesResult
prepareLoopPipelines(mlir::ModuleOp module,
                     llvm::ArrayRef<TilePipelineChoice> pipelines,
                     const LoopPipeliningLimits &limits = {});

PipelinedModuleResult pipelineLoops(mlir::OwningOpRef<mlir::ModuleOp> module,
                                    PreparedLoopPipelines prepared);

mlir::LogicalResult verifyPipelinedModule(const PipelinedModule &module,
                                          std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_LOOPPIPELINING_H
