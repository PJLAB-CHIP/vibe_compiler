//===- SPMCapacityEvaluation.h - Scoped fixed-capacity SPM probes -*- C++ -*-===//
//
// Query-local fixed-capacity SPM feasibility evaluation for the deterministic
// baseline controller. Two scopes exist:
//
//  - region scope: one materialized single-root TileRegion is cloned into a
//    private evaluation scope, lowered to Instr, its isolated required joins
//    are rebuilt, and static SPM packing runs without assigning offsets;
//  - function scope: one Tile FuncOp is cloned into a private module and runs
//    the same TileRegion-to-Instr, required-join placement, memory-planning
//    preparation and SPM offset-assignment sequence as the final Q50.0 gate.
//
// Both probes classify Fits, proven capacity overflow, unsupported lifetime
// and indeterminate analysis failure, and attribute every capacity demand to
// structured operation/operand/output nodes through clone-remapped
// materialization relations. A region-scope `RequiresFunctionScope` verdict is
// a scope-escalation request for the owning controller, never a fit and never
// an ignorable result. The probes own no candidate-selection policy and never
// modify the parent IR.
//
//===----------------------------------------------------------------------===//

#ifndef WAFER_COMPILER_SPMCAPACITYEVALUATION_H
#define WAFER_COMPILER_SPMCAPACITYEVALUATION_H

#include "StructuredBufferRelations.h"
#include "TileMemoryPlanning.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace wafer::compiler::detail {

enum class TileRegionSPMCapacityStatus : uint8_t {
  Fits,
  RequiresFunctionScope,
  CapacityExceeded,
  AnalysisFailure,
};

enum class TileRegionSPMCapacityPhase : uint8_t {
  None,
  InputValidation,
  InstructionLowering,
  StaticPacking,
};

/// Classification produced by one fixed-capacity region-scoped SPM query.
/// On failure, `planningFailure` carries evidence attributed through the
/// caller relations remapped into the private evaluation clone; the node
/// witnesses are structured DAG identities and remain stable across the
/// clone.
struct TileRegionSPMCapacityEvaluation {
  TileRegionSPMCapacityStatus status =
      TileRegionSPMCapacityStatus::AnalysisFailure;
  TileRegionSPMCapacityPhase phase = TileRegionSPMCapacityPhase::None;
  std::string detail;
  TileMemoryPlanningFailure planningFailure;

  bool fits() const {
    return status == TileRegionSPMCapacityStatus::Fits;
  }
  bool capacityExceeded() const {
    return status == TileRegionSPMCapacityStatus::CapacityExceeded;
  }
  bool requiresFunctionScope() const {
    return status == TileRegionSPMCapacityStatus::RequiresFunctionScope;
  }
  llvm::StringRef getPhaseDiagnosticLabel() const;
};

/// Runs one region-scoped fixed-capacity SPM probe. `relations` is optional;
/// when provided, its buffers are remapped into the private clone and used to
/// attribute the planning evidence. When absent, the evidence node lists stay
/// empty.
TileRegionSPMCapacityEvaluation evaluateTileRegionSPMCapacity(
    TileRegionOp region, TileRegionToInstrLoweringSession &loweringSession,
    llvm::raw_ostream &diagnostics,
    const StructuredMaterializationRelations *relations = nullptr);

enum class TileFunctionSPMCapacityStatus : uint8_t {
  Fits,
  CapacityExceeded,
  UnsupportedLifetime,
  AnalysisFailure,
};

enum class TileFunctionSPMCapacityPhase : uint8_t {
  None,
  InputValidation,
  InstructionLowering,
  PlanningPreparation,
  StaticPacking,
};

/// Classification produced by one function-scoped fixed-capacity SPM probe.
/// The probe replays the exact TileRegion-to-Instr, required-join placement,
/// memory-planning preparation and SPM offset assignment sequence of the final
/// Q50.0 per-Tile gate on a private clone, so its verdict consumes the same
/// demand set as the final planning. `UnsupportedLifetime` means the nearest
/// legal isolated ancestor itself cannot conclude; the owning controller must
/// treat it as a typed unsupported result.
struct TileFunctionSPMCapacityEvaluation {
  TileFunctionSPMCapacityStatus status =
      TileFunctionSPMCapacityStatus::AnalysisFailure;
  TileFunctionSPMCapacityPhase phase = TileFunctionSPMCapacityPhase::None;
  std::string detail;
  TileMemoryPlanningFailure planningFailure;

  bool fits() const {
    return status == TileFunctionSPMCapacityStatus::Fits;
  }
  bool capacityExceeded() const {
    return status == TileFunctionSPMCapacityStatus::CapacityExceeded;
  }
  bool unsupportedLifetime() const {
    return status == TileFunctionSPMCapacityStatus::UnsupportedLifetime;
  }
  llvm::StringRef getPhaseDiagnosticLabel() const;
};

/// Runs one function-scoped fixed-capacity SPM probe over the given Tile
/// FuncOp. The function and its regions are cloned into a private module;
/// `relations` buffers are remapped into that clone and used to attribute
/// the planning evidence.
TileFunctionSPMCapacityEvaluation evaluateTileFunctionSPMCapacity(
    mlir::func::FuncOp function,
    const StructuredMaterializationRelations &relations,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_SPMCAPACITYEVALUATION_H
