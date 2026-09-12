//===- ExecutionStructure.h - Current Tile execution rewrite -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_EXECUTIONSTRUCTURE_H
#define WAFER_TRANSFORMS_EXECUTIONSTRUCTURE_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct ExecutionStructureLimits {
  uint64_t maxFiniteUnrolledOperations = 4096;
};

enum class ExecutionStructureFailureKind : uint8_t {
  BrokenContract,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

struct ExecutionStructureFailure {
  ExecutionStructureFailureKind kind =
      ExecutionStructureFailureKind::CompilerBug;
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
  uint64_t tripCount = 0;
  uint32_t stageCount = 0;
};

struct PreparedExecutionStructure {
  mlir::ModuleOp module;
  std::vector<PreparedTilePipeline> pipelines;
  ExecutionStructureLimits limits;
};

struct PreparedExecutionStructureResult {
  std::optional<PreparedExecutionStructure> prepared;
  std::optional<ExecutionStructureFailure> failure;

  bool succeeded() const { return prepared.has_value(); }
};

enum class MaterializedExecutionPhase : uint8_t {
  Prologue,
  Kernel,
  Epilogue,
};

struct MaterializedExecutionOperation {
  mlir::Operation *operation = nullptr;
  uint32_t stage = 0;
  MaterializedExecutionPhase phase = MaterializedExecutionPhase::Kernel;
  uint64_t staticIteration = 0;
};

struct MaterializedExecutionPipeline {
  uint32_t stageCount = 0;
  uint64_t kernelDynamicTripCount = 0;
  uint64_t originalOperationCount = 0;
  std::vector<MaterializedExecutionOperation> operations;
};

struct MaterializedExecutionStructure {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<MaterializedExecutionPipeline> pipelines;
};

struct MaterializedExecutionStructureResult {
  std::optional<MaterializedExecutionStructure> materialized;
  std::optional<ExecutionStructureFailure> failure;

  bool succeeded() const { return materialized.has_value(); }
};

PreparedExecutionStructureResult
prepareTileExecutionStructure(mlir::ModuleOp module,
                              llvm::ArrayRef<TilePipelineChoice> pipelines,
                              const ExecutionStructureLimits &limits = {});

MaterializedExecutionStructureResult
materializeExecutionStructure(mlir::OwningOpRef<mlir::ModuleOp> module,
                              PreparedExecutionStructure prepared);

/// Query the current physical load/consumer graph. No capacity prediction is
/// used; eligibility requires a complete per-iteration definition and no
/// escaping or mutated alias of the selected load destination.
bool hasDistanceOneLoadPipeline(mlir::ModuleOp module);

/// Materialize two actual slots and the existing SCF pipeline in the owned
/// transaction. The ordinary Instr/completion/memory path remains the consumer.
MaterializedExecutionStructureResult materializeDistanceOneLoadPipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations &relations);

struct RotatingAllocationBinding {
  mlir::memref::AllocOp allocation;
  mlir::scf::ForOp loop;
  uint32_t multiplicity = 0;
};

struct RotatingAllocationMaterialization {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SmallVector<mlir::Value, 4> slots;
};

struct RotatingAllocationMaterializationResult {
  std::optional<RotatingAllocationMaterialization> materialized;
  std::optional<ExecutionStructureFailure> failure;

  bool succeeded() const { return materialized.has_value(); }
};

RotatingAllocationMaterializationResult materializeRotatingAllocations(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<RotatingAllocationBinding> bindings,
    StructuredMaterializationRelations &relations);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_EXECUTIONSTRUCTURE_H
