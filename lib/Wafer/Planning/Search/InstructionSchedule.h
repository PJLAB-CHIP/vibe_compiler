//===- InstructionSchedule.h - Exact Instr schedule domain ---*- C++ -*-===//

#pragma once

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace wafer::compiler::detail {

struct TileInstructionModule {
  TileId tile{0};
  mlir::ModuleOp module;
};

/// Query-local order for one movable Instr window. Operation references are
/// valid only for the unchanged modules from which the domain was created.
struct InstructionWindowOrder {
  TileId tile{0};
  mlir::Block *block = nullptr;
  llvm::SmallVector<mlir::Operation *, 16> operations;

  friend bool operator==(const InstructionWindowOrder &lhs,
                         const InstructionWindowOrder &rhs) {
    return lhs.tile == rhs.tile && lhs.block == rhs.block &&
           lhs.operations == rhs.operations;
  }
};

struct InstructionWorkerChoice {
  TileId tile{0};
  mlir::Operation *operation = nullptr;
  NCCWorker worker = NCCWorker::Worker0;

  friend bool operator==(const InstructionWorkerChoice &lhs,
                         const InstructionWorkerChoice &rhs) {
    return lhs.tile == rhs.tile && lhs.operation == rhs.operation &&
           lhs.worker == rhs.worker;
  }
};

struct CardInstructionScheduleAssignment {
  llvm::SmallVector<InstructionWindowOrder, 32> windows;
  llvm::SmallVector<InstructionWorkerChoice, 64> workers;

  friend bool operator==(const CardInstructionScheduleAssignment &lhs,
                         const CardInstructionScheduleAssignment &rhs) {
    return lhs.windows == rhs.windows && lhs.workers == rhs.workers;
  }
};

/// Lazy exact domain for one unchanged complete Card Instr epoch. It stores no
/// clone and chooses no winner. Window orders cover every topological order of
/// the hard SSA/effect/token/completion DAG; worker choices cover the complete
/// closed NCC worker enum.
class CardInstructionScheduleDomain {
public:
  static mlir::FailureOr<CardInstructionScheduleDomain>
  create(llvm::ArrayRef<TileInstructionModule> modules,
         std::string *failureReason = nullptr);

  CardInstructionScheduleAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<CardInstructionScheduleAssignment>>
  getNextAssignment(const CardInstructionScheduleAssignment &assignment) const;
  bool contains(const CardInstructionScheduleAssignment &assignment) const;
  bool isCurrent(llvm::ArrayRef<TileInstructionModule> modules) const;

  /// Internal immutable query coordinates exposed only because the domain's
  /// implementation helpers operate on them; callers use assignments.
  struct WindowDomain {
    TileId tile{0};
    mlir::Block *block = nullptr;
    llvm::SmallVector<mlir::Operation *, 16> operations;
    llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> predecessors;
  };

  struct WorkerDomain {
    TileId tile{0};
    mlir::Operation *operation = nullptr;
  };

  struct OperationSnapshot {
    mlir::Operation *operation = nullptr;
    mlir::DictionaryAttr attributes;
    llvm::SmallVector<mlir::Value, 4> operands;
    llvm::SmallVector<mlir::Type, 2> resultTypes;
  };

private:
  CardInstructionScheduleDomain(
      llvm::SmallVector<WindowDomain, 32> windows,
      llvm::SmallVector<WorkerDomain, 64> workers,
      llvm::SmallVector<TileInstructionModule, 16> modules,
      llvm::SmallVector<OperationSnapshot, 128> snapshot)
      : windows(std::move(windows)), workers(std::move(workers)),
        modules(std::move(modules)), snapshot(std::move(snapshot)) {}

  InstructionWindowOrder getFirstOrder(const WindowDomain &window) const;
  mlir::FailureOr<std::optional<InstructionWindowOrder>>
  getNextOrder(const WindowDomain &window,
               const InstructionWindowOrder &order) const;
  bool contains(const WindowDomain &window,
                const InstructionWindowOrder &order) const;
  bool isEpochCurrent() const;

  llvm::SmallVector<WindowDomain, 32> windows;
  llvm::SmallVector<WorkerDomain, 64> workers;
  llvm::SmallVector<TileInstructionModule, 16> modules;
  llvm::SmallVector<OperationSnapshot, 128> snapshot;
};

struct ScheduledInstructionModules {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
};

/// Consumes the complete candidate modules so a failed in-place schedule
/// cannot escape. The selected order and workers are applied once, all old
/// derived joins are removed, and required joins are rebuilt from current IR.
mlir::FailureOr<ScheduledInstructionModules>
applyInstructionSchedule(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules,
                         llvm::ArrayRef<TileId> tileIds,
                         const CardInstructionScheduleDomain &domain,
                         const CardInstructionScheduleAssignment &assignment,
                         std::string *failureReason = nullptr);

enum class InstructionResourceKind : uint8_t {
  CT,
  NE,
  RDMA,
  WDMA,
  TDMA,
  DirectDTE,
  NCCWorker,
  TileSPM,
  CardDDR,
  DirectedPeerLink,
};

struct InstructionResourceKey {
  InstructionResourceKind kind = InstructionResourceKind::CT;
  int64_t first = 0;
  int64_t second = 0;

  friend bool operator==(const InstructionResourceKey &lhs,
                         const InstructionResourceKey &rhs) {
    return std::tie(lhs.kind, lhs.first, lhs.second) ==
           std::tie(rhs.kind, rhs.first, rhs.second);
  }
  friend bool operator<(const InstructionResourceKey &lhs,
                        const InstructionResourceKey &rhs) {
    return std::tie(lhs.kind, lhs.first, lhs.second) <
           std::tie(rhs.kind, rhs.first, rhs.second);
  }
};

struct InstructionResourceEvent {
  TileId tile{0};
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<InstructionResourceKey, 4> resources;
};

struct InstructionOverlapWitness {
  TileId tile{0};
  mlir::Operation *pendingIssue = nullptr;
  mlir::Operation *independentOperation = nullptr;
};

struct SharedInstructionResource {
  InstructionResourceKey resource;
  llvm::SmallVector<mlir::Operation *, 4> users;
};

/// Recomputable facts from selected actual order/worker/completion IR. Shared
/// resources are contention inputs, not legality or profitability decisions.
struct CardInstructionResourceAnalysis {
  llvm::SmallVector<InstructionResourceEvent, 64> events;
  llvm::SmallVector<InstructionOverlapWitness, 16> overlapWitnesses;
  llvm::SmallVector<SharedInstructionResource, 16> sharedResources;
};

mlir::FailureOr<CardInstructionResourceAnalysis>
analyzeInstructionResources(llvm::ArrayRef<TileInstructionModule> modules,
                            std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
