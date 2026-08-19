//===- UnifiedPhysicalDataflow.h - Joint physical assignment -*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Search/Buffering.h"
#include "Wafer/Planning/Search/BufferingApply.h"
#include "Wafer/Planning/Search/ComputeImplementation.h"
#include "Wafer/Planning/Search/SpatialPlacement.h"

namespace wafer::compiler::detail {

struct UnifiedPhysicalDataflowAssignment {
  CardSpatialPlacementAssignment spatial;
  CoupledRegionAssignment coupled;
  CardTemporalAssignment temporal;
  CardComputeImplementationAssignment implementation;
  CardPhysicalRepresentationAssignment representation;
  CardDataMovementAssignment movement;
  CardBufferingAssignment buffering;
};

/// Complete dependent Cartesian domain for one immutable TensorProgram root.
/// Downstream domains are rebuilt from the current parent assignment and never
/// cached as semantic state. Spatial points that do not produce an exact
/// satisfied logical trial are skipped without deleting their siblings.
class UnifiedPhysicalDataflowDomain {
public:
  static mlir::FailureOr<UnifiedPhysicalDataflowDomain>
  create(const CardProgramAnalysis &program, CardId cardId,
         const TargetMemoryPolicy &memory,
         std::string *failureReason = nullptr);

  mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
  getFirstAssignment(std::string *failureReason = nullptr) const;
  mlir::FailureOr<std::optional<UnifiedPhysicalDataflowAssignment>>
  getNextAssignment(const UnifiedPhysicalDataflowAssignment &assignment,
                    std::string *failureReason = nullptr) const;
  bool contains(const UnifiedPhysicalDataflowAssignment &assignment) const;

  mlir::FailureOr<CardCoupledRegionMaterialization>
  materialize(mlir::ModuleOp tensorProgram,
              const UnifiedPhysicalDataflowAssignment &assignment,
              std::string *failureReason = nullptr) const;

  mlir::FailureOr<std::vector<llvm::SmallVector<SelectedBufferingScope, 4>>>
  buildBufferingScopes(const UnifiedPhysicalDataflowAssignment &assignment,
                       std::string *failureReason = nullptr) const;

private:
  UnifiedPhysicalDataflowDomain(const CardProgramAnalysis &program,
                                CardId cardId, TargetMemoryPolicy memory,
                                CardSpatialPlacementDomain spatial,
                                CardComputeImplementationDomain implementation)
      : program(program), cardId(cardId), memory(memory),
        spatial(std::move(spatial)), implementation(std::move(implementation)) {
  }

  mlir::FailureOr<analysis::LogicalShardTrial>
  getTrial(const CardSpatialPlacementAssignment &assignment,
           std::string *failureReason) const;
  mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
  getFirstForSpatial(const CardSpatialPlacementAssignment &assignment,
                     std::string *failureReason) const;

  const CardProgramAnalysis &program;
  CardId cardId{0};
  TargetMemoryPolicy memory;
  CardSpatialPlacementDomain spatial;
  CardComputeImplementationDomain implementation;
};

} // namespace wafer::compiler::detail
