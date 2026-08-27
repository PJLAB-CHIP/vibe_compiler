//===- ExecutionStructureMaterialization.h - Selected SCF phases -*- C++
//-*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREMATERIALIZATION_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREMATERIALIZATION_H

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

/// One selected event and the exact top-level operations emitted for it in J
/// order. Empty groups represent structural event boundaries with no operation
/// of their own. Every actual loop-body operation must belong to exactly one
/// group; the constructor never recovers identity from block order or names.
struct ExecutionEventOperationGroup {
  EventId event;
  std::vector<mlir::Operation *> operations;
};

/// Exact storage fact supplied by the structure-specific storage owner. It is
/// the only way a cross-stage write through loop-external storage is admitted.
struct ExternalStorageStageProof {
  mlir::Value root;
  StorageObjectId object;
};

struct ExecutionStructureLoopBinding {
  PipelineScopeId scope;
  mlir::scf::ForOp steadyLoop;
  std::vector<ExecutionEventOperationGroup> eventOrder;
  std::vector<ExternalStorageStageProof> externalStorageProofs;
};

struct ExecutionStructureMaterializationLimits {
  uint64_t maxFiniteUnrolledOperations = 4096;
};

enum class ExecutionStructureMaterializationFailureKind : uint8_t {
  BrokenContract,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

struct ExecutionStructureMaterializationFailure {
  ExecutionStructureMaterializationFailureKind kind =
      ExecutionStructureMaterializationFailureKind::CompilerBug;
  std::optional<PipelineScopeId> scope;
  std::string detail;
};

struct PreparedExecutionStructureOperation {
  EventId event;
  mlir::Operation *operation = nullptr;
  StageId stage{0};
};

struct PreparedExecutionStructureScope {
  PipelinedExecutionStructure plan;
  mlir::scf::ForOp steadyLoop;
  std::vector<PreparedExecutionStructureOperation> operations;
};

/// Read-only checked construction recipe. Raw handles are valid only while the
/// same module epoch remains unmodified and are consumed by the immediate
/// owned-module materialization call.
struct PreparedExecutionStructure {
  mlir::ModuleOp module;
  std::vector<PreparedExecutionStructureScope> scopes;
  ExecutionStructureMaterializationLimits limits;
};

struct PreparedExecutionStructureResult {
  std::optional<PreparedExecutionStructure> prepared;
  std::optional<ExecutionStructureMaterializationFailure> failure;

  bool succeeded() const { return prepared.has_value(); }
};

enum class MaterializedExecutionPhase : uint8_t {
  Prologue,
  Kernel,
  Epilogue,
};

struct MaterializedExecutionEvent {
  EventId event;
  mlir::Operation *operation = nullptr;
  StageId stage{0};
  MaterializedExecutionPhase phase = MaterializedExecutionPhase::Kernel;
  uint64_t staticIteration = 0;
};

struct MaterializedExecutionStructureScope {
  PipelinedExecutionStructure plan;
  uint32_t stageCount = 0;
  uint64_t kernelDynamicTripCount = 0;
  std::vector<std::pair<EventId, uint32_t>> originalOperationCounts;
  std::vector<MaterializedExecutionEvent> events;
};

struct MaterializedExecutionStructure {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<MaterializedExecutionStructureScope> scopes;
};

struct MaterializedExecutionStructureResult {
  std::optional<MaterializedExecutionStructure> materialized;
  std::optional<ExecutionStructureMaterializationFailure> failure;

  bool succeeded() const { return materialized.has_value(); }
};

PreparedExecutionStructureResult prepareExecutionStructureMaterialization(
    mlir::ModuleOp module, const ExecutionStructurePlan &plan,
    const BufferPlan &buffers,
    llvm::ArrayRef<ExecutionStructureLoopBinding> bindings,
    const ExecutionStructureMaterializationLimits &limits = {});

/// Current-IR entry used by the Tile execution-structure stage. `pipelines`
/// are invocation-local transformation choices whose operation bindings refer
/// directly to the unchanged input module. No BufferPlan, slot-family plan or
/// future event inventory crosses this boundary. An empty pipeline list is the
/// serialized identity.
PreparedExecutionStructureResult prepareTileExecutionStructure(
    mlir::ModuleOp module,
    llvm::ArrayRef<PipelinedExecutionStructure> pipelines,
    llvm::ArrayRef<ExecutionStructureLoopBinding> bindings,
    const ExecutionStructureMaterializationLimits &limits = {});

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
  std::optional<ExecutionStructureMaterializationFailure> failure;

  bool succeeded() const { return materialized.has_value(); }
};

/// Materializes actual rotating allocation roots and loop-local SSA slot
/// selection. Every binding names current IR directly; all preflight completes
/// before mutation. Relations are expanded to the actual slot roots in the same
/// transaction. Offset assignment and completion remain downstream.
RotatingAllocationMaterializationResult materializeRotatingAllocations(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<RotatingAllocationBinding> bindings,
    StructuredMaterializationRelations &relations);

/// Consumes the candidate module because pinned SCF pipelining can modify IR
/// before reporting failure. No partially modified module is returned.
MaterializedExecutionStructureResult
materializeExecutionStructure(mlir::OwningOpRef<mlir::ModuleOp> module,
                              PreparedExecutionStructure prepared);

mlir::LogicalResult verifyMaterializedExecutionStructure(
    const MaterializedExecutionStructure &materialized,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREMATERIALIZATION_H
