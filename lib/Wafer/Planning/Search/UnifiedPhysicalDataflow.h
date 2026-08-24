//===- UnifiedPhysicalDataflow.h - Joint physical assignment -*- C++ -*-===//

#pragma once

#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"
#include "Wafer/Planning/Search/Buffering.h"
#include "Wafer/Planning/Search/ComputeImplementation.h"

namespace wafer::compiler::detail {

struct UnifiedPhysicalDataflowAssignment {
  SpatialPlan spatial;
  CoupledRegionAssignment coupled;
  CardTemporalAssignment temporal;
  CardComputeImplementationAssignment implementation;
  CardPhysicalRepresentationAssignment representation;
  CardDataMovementAssignment movement;
  CardBufferingAssignment buffering;

  friend bool operator==(const UnifiedPhysicalDataflowAssignment &lhs,
                         const UnifiedPhysicalDataflowAssignment &rhs) {
    return lhs.spatial == rhs.spatial && lhs.coupled == rhs.coupled &&
           lhs.temporal == rhs.temporal &&
           lhs.implementation == rhs.implementation &&
           lhs.representation == rhs.representation &&
           lhs.movement == rhs.movement && lhs.buffering == rhs.buffering;
  }
};

struct ClosedSpatialDemand {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
};

/// Complete dependent Cartesian domain for one immutable TensorProgram root.
/// Downstream domains are rebuilt from the current parent assignment and never
/// cached as semantic state. A spatial point whose exact demand or dependent
/// suffix cannot close is skipped during traversal without being cached as
/// spatial illegality or deleting its siblings.
class UnifiedPhysicalDataflowDomain {
public:
  static mlir::FailureOr<UnifiedPhysicalDataflowDomain>
  create(const CardProgramAnalysis &program, CardId cardId,
         std::string *failureReason = nullptr);

  mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
  getFirstAssignment(std::string *failureReason = nullptr) const;
  /// Returns the first downstream-closed assignment from the spatial
  /// domain's deterministic proposal order. Every proposal is also a raw
  /// successor member; failure does not substitute another policy's result.
  mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
  getConstructiveAssignment(std::string *failureReason = nullptr) const;
  /// Rebuilds the complete dependent suffix after repairing every coupled
  /// connected component toward legal fusion. This is a proposal only and
  /// never removes the corresponding exact-domain states.
  mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
  getFusionOrientedAssignment(std::string *failureReason = nullptr) const;
  mlir::FailureOr<std::optional<UnifiedPhysicalDataflowAssignment>>
  getNextAssignment(const UnifiedPhysicalDataflowAssignment &assignment,
                    std::string *failureReason = nullptr) const;
  bool contains(const UnifiedPhysicalDataflowAssignment &assignment) const;

  mlir::FailureOr<CardCoupledRegionMaterialization>
  materialize(mlir::ModuleOp tensorProgram,
              const UnifiedPhysicalDataflowAssignment &assignment,
              std::string *failureReason = nullptr) const;

private:
  UnifiedPhysicalDataflowDomain(const CardProgramAnalysis &program,
                                CardId cardId, SpatialPlanDomain spatial,
                                CardComputeImplementationDomain implementation)
      : program(program), cardId(cardId), spatial(std::move(spatial)),
        implementation(std::move(implementation)) {}

  mlir::FailureOr<ClosedSpatialDemand>
  getSpatialDemand(const SpatialPlan &assignment,
                   std::string *failureReason) const;
  mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
  getFirstForSpatial(const SpatialPlan &assignment, std::string *failureReason,
                     bool fusionOriented = false) const;

  const CardProgramAnalysis &program;
  CardId cardId{0};
  SpatialPlanDomain spatial;
  CardComputeImplementationDomain implementation;
};

} // namespace wafer::compiler::detail
