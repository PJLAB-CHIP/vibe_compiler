//===- ScheduleMaterialization.h - Selected event schedule IR -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEMATERIALIZATION_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEMATERIALIZATION_H

#include "Wafer/Planning/PhysicalDataflow/ScheduleDomain.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct ScheduleIRModule {
  TileId tile{0};
  mlir::ModuleOp module;
};

/// Exact caller-owned event relation. Empty groups are legal structural or
/// completion markers but still name their owning block.
struct ScheduleEventIRBinding {
  EventId event;
  TileId tile{0};
  /// Primary block retained for the common single-occurrence case.
  mlir::Block *block = nullptr;
  std::vector<mlir::Operation *> operations;
  /// All static occurrence blocks for this EventId. Empty means `{block}`.
  /// Operations must belong to exactly one listed block.
  std::vector<mlir::Block *> blocks;
};

enum class ScheduleMaterializationFailureKind : uint8_t {
  BrokenContract,
  Unsupported,
  CompilerBug,
};

struct ScheduleMaterializationFailure {
  ScheduleMaterializationFailureKind kind =
      ScheduleMaterializationFailureKind::CompilerBug;
  std::optional<EventId> event;
  std::string detail;
};

struct PreparedScheduleScope {
  ControlOrder order;
  std::vector<ScheduleEventIRBinding> events;
};

struct PreparedScheduleMaterialization {
  ClosedSchedulePlan plan;
  std::vector<ScheduleIRModule> modules;
  std::vector<PreparedScheduleScope> scopes;
};

struct PreparedScheduleMaterializationResult {
  std::optional<PreparedScheduleMaterialization> prepared;
  std::optional<ScheduleMaterializationFailure> failure;

  bool succeeded() const { return prepared.has_value(); }
};

struct MaterializedCompletionGroup {
  EventBoundaryId boundary;
  mlir::Block *block = nullptr;
  std::vector<CompletionPlacement> placements;
  std::vector<mlir::Operation *> operations;
};

struct MaterializedSchedule {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  std::vector<TileId> moduleTiles;
  ClosedSchedulePlan plan;
  std::vector<ScheduleEventIRBinding> eventBindings;
  std::vector<MaterializedCompletionGroup> completionGroups;
};

struct MaterializedScheduleResult {
  std::optional<MaterializedSchedule> materialized;
  std::optional<ScheduleMaterializationFailure> failure;

  bool succeeded() const { return materialized.has_value(); }
};

PreparedScheduleMaterializationResult
prepareScheduleMaterialization(const ScheduleDomain &domain,
                               const ClosedSchedulePlan &plan,
                               llvm::ArrayRef<ScheduleIRModule> modules,
                               llvm::ArrayRef<ScheduleEventIRBinding> bindings);

/// Consumes every Tile module because reordering and completion insertion are
/// one atomic candidate transaction. Failure returns no partial module.
MaterializedScheduleResult
materializeSchedule(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules,
                    PreparedScheduleMaterialization prepared);

mlir::LogicalResult
verifyMaterializedSchedule(const MaterializedSchedule &materialized,
                           std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEMATERIALIZATION_H
