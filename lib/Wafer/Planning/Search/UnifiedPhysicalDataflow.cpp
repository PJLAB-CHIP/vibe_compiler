//===- UnifiedPhysicalDataflow.cpp - Joint physical assignment --------===//

#include "Wafer/Planning/Search/UnifiedPhysicalDataflow.h"

#include "Wafer/Planning/Search/BufferingApply.h"

namespace wafer::compiler::detail {

mlir::FailureOr<UnifiedPhysicalDataflowDomain>
UnifiedPhysicalDataflowDomain::create(const CardProgramAnalysis &program,
                                      CardId cardId,
                                      std::string *failureReason) {
  auto spatial =
      CardSpatialPlacementDomain::create(program.dag, program.availableTileIds);
  auto implementation =
      CardComputeImplementationDomain::create(program, failureReason);
  if (mlir::failed(spatial) || mlir::failed(implementation))
    return mlir::failure();
  return UnifiedPhysicalDataflowDomain(
      program, cardId, std::move(*spatial), std::move(*implementation));
}

mlir::FailureOr<ClosedSpatialDemand>
UnifiedPhysicalDataflowDomain::getSpatialDemand(
    const CardSpatialPlacementAssignment &assignment,
    std::string *failureReason) const {
  CardSpatialPlacementEvaluation evaluation =
      spatial.evaluate(program.dag, assignment);
  const analysis::ExactDemandProof *proof =
      evaluation.demand ? analysis::getExactDemandProof(*evaluation.demand)
                        : nullptr;
  if (!evaluation.assignment || !proof) {
    if (failureReason)
      *failureReason = evaluation.detail;
    return mlir::failure();
  }
  return ClosedSpatialDemand{std::move(*evaluation.assignment), *proof};
}

mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
UnifiedPhysicalDataflowDomain::getFirstForSpatial(
    const CardSpatialPlacementAssignment &spatialAssignment,
    std::string *failureReason, bool fusionOriented) const {
  auto closed = getSpatialDemand(spatialAssignment, failureReason);
  if (mlir::failed(closed))
    return mlir::failure();
  auto coupled = CoupledRegionDomain::create(
      program.dag, closed->spatial, closed->demand, failureReason);
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements =
      spatial.getNodePlacements(spatialAssignment);
  auto temporal =
      CardTemporalDomain::create(program.dag, placements, failureReason);
  if (mlir::failed(coupled) || mlir::failed(temporal))
    return mlir::failure();
  CoupledRegionAssignment coupledAssignment =
      fusionOriented ? coupled->getFusionOrientedAssignment()
                     : coupled->getFirstAssignment();
  CardTemporalAssignment temporalAssignment = temporal->getFirstAssignment();
  CardComputeImplementationAssignment implementationAssignment =
      implementation.getFirstAssignment();
  auto representation = CardPhysicalRepresentationDomain::create(
      program, closed->spatial, closed->demand, *coupled, coupledAssignment,
      *temporal, temporalAssignment, failureReason);
  if (mlir::failed(representation))
    return mlir::failure();
  CardPhysicalRepresentationAssignment representationAssignment =
      representation->getFirstAssignment();
  auto movement = CardDataMovementDomain::create(
      program, cardId, closed->spatial, closed->demand, *coupled,
      coupledAssignment, *temporal, temporalAssignment, *representation,
      representationAssignment, failureReason);
  if (mlir::failed(movement))
    return mlir::failure();
  CardDataMovementAssignment movementAssignment =
      movement->getFirstAssignment();
  auto buffering = CardBufferingDomain::create(
      program, closed->spatial, closed->demand, *coupled, coupledAssignment,
      *temporal, temporalAssignment, *representation,
      representationAssignment, *movement, movementAssignment, failureReason);
  if (mlir::failed(buffering))
    return mlir::failure();
  return UnifiedPhysicalDataflowAssignment{spatialAssignment,
                                           std::move(coupledAssignment),
                                           std::move(temporalAssignment),
                                           std::move(implementationAssignment),
                                           std::move(representationAssignment),
                                           std::move(movementAssignment),
                                           buffering->getFirstAssignment()};
}

mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
UnifiedPhysicalDataflowDomain::getFirstAssignment(
    std::string *failureReason) const {
  CardSpatialPlacementAssignment current = spatial.getFirstAssignment();
  while (true) {
    if (auto assignment = getFirstForSpatial(current, failureReason);
        mlir::succeeded(assignment))
      return assignment;
    auto next = spatial.getNextAssignment(current);
    if (mlir::failed(next) || !*next)
      return mlir::failure();
    current = std::move(**next);
  }
}

mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
UnifiedPhysicalDataflowDomain::getConstructiveAssignment(
    std::string *failureReason) const {
  auto proposal =
      spatial.getConstructiveAssignment(program.dag, failureReason);
  if (mlir::failed(proposal))
    return mlir::failure();
  return getFirstForSpatial(*proposal, failureReason);
}

mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
UnifiedPhysicalDataflowDomain::getFusionOrientedAssignment(
    std::string *failureReason) const {
  auto proposal =
      spatial.getConstructiveAssignment(program.dag, failureReason);
  if (mlir::failed(proposal))
    return mlir::failure();
  return getFirstForSpatial(*proposal, failureReason,
                            /*fusionOriented=*/true);
}

bool UnifiedPhysicalDataflowDomain::contains(
    const UnifiedPhysicalDataflowAssignment &assignment) const {
  if (!spatial.contains(assignment.spatial) ||
      !implementation.contains(assignment.implementation))
    return false;
  auto closed = getSpatialDemand(assignment.spatial, nullptr);
  if (mlir::failed(closed))
    return false;
  auto coupled = CoupledRegionDomain::create(
      program.dag, closed->spatial, closed->demand);
  auto temporal = CardTemporalDomain::create(
      program.dag, spatial.getNodePlacements(assignment.spatial));
  if (mlir::failed(coupled) || mlir::failed(temporal) ||
      !coupled->contains(assignment.coupled) ||
      !temporal->contains(assignment.temporal))
    return false;
  auto representation = CardPhysicalRepresentationDomain::create(
      program, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal);
  if (mlir::failed(representation) ||
      !representation->contains(assignment.representation))
    return false;
  auto movement = CardDataMovementDomain::create(
      program, cardId, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, *representation,
      assignment.representation);
  if (mlir::failed(movement) || !movement->contains(assignment.movement))
    return false;
  auto buffering = CardBufferingDomain::create(
      program, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, *representation,
      assignment.representation, *movement, assignment.movement);
  return mlir::succeeded(buffering) &&
         buffering->contains(assignment.buffering);
}

mlir::FailureOr<std::optional<UnifiedPhysicalDataflowAssignment>>
UnifiedPhysicalDataflowDomain::getNextAssignment(
    const UnifiedPhysicalDataflowAssignment &assignment,
    std::string *failureReason) const {
  if (!contains(assignment))
    return mlir::failure();
  auto closed = *getSpatialDemand(assignment.spatial, failureReason);
  auto coupled = *CoupledRegionDomain::create(
      program.dag, closed.spatial, closed.demand, failureReason);
  auto temporal = *CardTemporalDomain::create(
      program.dag, spatial.getNodePlacements(assignment.spatial),
      failureReason);
  auto representation = *CardPhysicalRepresentationDomain::create(
      program, closed.spatial, closed.demand, coupled, assignment.coupled,
      temporal,
      assignment.temporal, failureReason);
  auto movement = *CardDataMovementDomain::create(
      program, cardId, closed.spatial, closed.demand, coupled,
      assignment.coupled, temporal, assignment.temporal, representation,
      assignment.representation, failureReason);
  auto buffering = *CardBufferingDomain::create(
      program, closed.spatial, closed.demand, coupled, assignment.coupled,
      temporal, assignment.temporal, representation,
      assignment.representation, movement, assignment.movement, failureReason);

  UnifiedPhysicalDataflowAssignment next = assignment;
  if (auto advanced = buffering.getNextAssignment(assignment.buffering);
      mlir::succeeded(advanced) && *advanced) {
    next.buffering = std::move(**advanced);
    return std::optional<UnifiedPhysicalDataflowAssignment>(std::move(next));
  }
  if (auto advanced = movement.getNextAssignment(assignment.movement);
      mlir::succeeded(advanced) && *advanced) {
    next.movement = std::move(**advanced);
    auto reset = CardBufferingDomain::create(
        program, closed.spatial, closed.demand, coupled, assignment.coupled,
        temporal, assignment.temporal, representation,
        assignment.representation, movement, next.movement, failureReason);
    if (mlir::failed(reset))
      return mlir::failure();
    next.buffering = reset->getFirstAssignment();
    return std::optional<UnifiedPhysicalDataflowAssignment>(std::move(next));
  }
  if (auto advanced =
          representation.getNextAssignment(assignment.representation);
      mlir::succeeded(advanced) && *advanced) {
    next.representation = std::move(**advanced);
    auto resetMovement = CardDataMovementDomain::create(
        program, cardId, closed.spatial, closed.demand, coupled,
        assignment.coupled, temporal, assignment.temporal, representation,
        next.representation, failureReason);
    if (mlir::failed(resetMovement))
      return mlir::failure();
    next.movement = resetMovement->getFirstAssignment();
    auto resetBuffering = CardBufferingDomain::create(
        program, closed.spatial, closed.demand, coupled, assignment.coupled,
        temporal, assignment.temporal, representation, next.representation,
        *resetMovement, next.movement, failureReason);
    if (mlir::failed(resetBuffering))
      return mlir::failure();
    next.buffering = resetBuffering->getFirstAssignment();
    return std::optional<UnifiedPhysicalDataflowAssignment>(std::move(next));
  }
  if (auto advanced =
          implementation.getNextAssignment(assignment.implementation);
      mlir::succeeded(advanced) && *advanced) {
    next.implementation = std::move(**advanced);
    next.representation = representation.getFirstAssignment();
    auto resetMovement = CardDataMovementDomain::create(
        program, cardId, closed.spatial, closed.demand, coupled,
        assignment.coupled, temporal, assignment.temporal, representation,
        next.representation, failureReason);
    if (mlir::failed(resetMovement))
      return mlir::failure();
    next.movement = resetMovement->getFirstAssignment();
    auto resetBuffering = CardBufferingDomain::create(
        program, closed.spatial, closed.demand, coupled, assignment.coupled,
        temporal, assignment.temporal, representation, next.representation,
        *resetMovement, next.movement, failureReason);
    if (mlir::failed(resetBuffering))
      return mlir::failure();
    next.buffering = resetBuffering->getFirstAssignment();
    return std::optional<UnifiedPhysicalDataflowAssignment>(std::move(next));
  }
  if (auto advanced = temporal.getNextAssignment(assignment.temporal);
      mlir::succeeded(advanced) && *advanced) {
    next.temporal = std::move(**advanced);
    next.implementation = implementation.getFirstAssignment();
    auto resetRepresentation = CardPhysicalRepresentationDomain::create(
        program, closed.spatial, closed.demand, coupled, assignment.coupled,
        temporal, next.temporal, failureReason);
    if (mlir::failed(resetRepresentation))
      return mlir::failure();
    next.representation = resetRepresentation->getFirstAssignment();
    auto resetMovement = CardDataMovementDomain::create(
        program, cardId, closed.spatial, closed.demand, coupled,
        assignment.coupled, temporal, next.temporal, *resetRepresentation,
        next.representation, failureReason);
    if (mlir::failed(resetMovement))
      return mlir::failure();
    next.movement = resetMovement->getFirstAssignment();
    auto resetBuffering = CardBufferingDomain::create(
        program, closed.spatial, closed.demand, coupled, assignment.coupled,
        temporal, next.temporal, *resetRepresentation, next.representation,
        *resetMovement, next.movement, failureReason);
    if (mlir::failed(resetBuffering))
      return mlir::failure();
    next.buffering = resetBuffering->getFirstAssignment();
    return std::optional<UnifiedPhysicalDataflowAssignment>(std::move(next));
  }
  if (auto advanced = coupled.getNextAssignment(assignment.coupled);
      mlir::succeeded(advanced) && *advanced) {
    next.coupled = std::move(**advanced);
    next.temporal = temporal.getFirstAssignment();
    next.implementation = implementation.getFirstAssignment();
    auto resetRepresentation = CardPhysicalRepresentationDomain::create(
        program, closed.spatial, closed.demand, coupled, next.coupled,
        temporal, next.temporal, failureReason);
    if (mlir::failed(resetRepresentation))
      return mlir::failure();
    next.representation = resetRepresentation->getFirstAssignment();
    auto resetMovement = CardDataMovementDomain::create(
        program, cardId, closed.spatial, closed.demand, coupled, next.coupled,
        temporal, next.temporal, *resetRepresentation, next.representation,
        failureReason);
    if (mlir::failed(resetMovement))
      return mlir::failure();
    next.movement = resetMovement->getFirstAssignment();
    auto resetBuffering = CardBufferingDomain::create(
        program, closed.spatial, closed.demand, coupled, next.coupled,
        temporal, next.temporal, *resetRepresentation, next.representation,
        *resetMovement, next.movement, failureReason);
    if (mlir::failed(resetBuffering))
      return mlir::failure();
    next.buffering = resetBuffering->getFirstAssignment();
    return std::optional<UnifiedPhysicalDataflowAssignment>(std::move(next));
  }

  CardSpatialPlacementAssignment spatialAssignment = assignment.spatial;
  while (true) {
    auto advanced = spatial.getNextAssignment(spatialAssignment);
    if (mlir::failed(advanced) || !*advanced)
      return std::optional<UnifiedPhysicalDataflowAssignment>{};
    spatialAssignment = std::move(**advanced);
    if (auto reset = getFirstForSpatial(spatialAssignment, failureReason);
        mlir::succeeded(reset))
      return std::optional<UnifiedPhysicalDataflowAssignment>(
          std::move(*reset));
  }
}

mlir::FailureOr<CardCoupledRegionMaterialization>
UnifiedPhysicalDataflowDomain::materialize(
    mlir::ModuleOp tensorProgram,
    const UnifiedPhysicalDataflowAssignment &assignment,
    std::string *failureReason) const {
  if (!contains(assignment))
    return mlir::failure();
  auto closed = getSpatialDemand(assignment.spatial, failureReason);
  if (mlir::failed(closed))
    return mlir::failure();
  auto coupled = CoupledRegionDomain::create(
      program.dag, closed->spatial, closed->demand, failureReason);
  if (mlir::failed(coupled))
    return mlir::failure();
  auto temporal = CardTemporalDomain::create(
      program.dag, spatial.getNodePlacements(assignment.spatial),
      failureReason);
  if (mlir::failed(temporal))
    return mlir::failure();
  auto representation = CardPhysicalRepresentationDomain::create(
      program, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, failureReason);
  if (mlir::failed(representation))
    return mlir::failure();
  auto movement = CardDataMovementDomain::create(
      program, cardId, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, *representation,
      assignment.representation, failureReason);
  if (mlir::failed(movement))
    return mlir::failure();
  return materializeCardCoupledRegionsWithImplementations(
      tensorProgram, program, cardId, closed->spatial, closed->demand,
      *coupled, assignment.coupled, *temporal, assignment.temporal,
      *representation,
      assignment.representation, implementation, assignment.implementation,
      *movement, assignment.movement, failureReason);
}

mlir::FailureOr<std::vector<llvm::SmallVector<SelectedBufferingScope, 4>>>
UnifiedPhysicalDataflowDomain::buildBufferingScopes(
    const UnifiedPhysicalDataflowAssignment &assignment,
    std::string *failureReason) const {
  if (!contains(assignment))
    return mlir::failure();
  auto closed = getSpatialDemand(assignment.spatial, failureReason);
  if (mlir::failed(closed))
    return mlir::failure();
  auto coupled = CoupledRegionDomain::create(
      program.dag, closed->spatial, closed->demand, failureReason);
  if (mlir::failed(coupled))
    return mlir::failure();
  auto temporal = CardTemporalDomain::create(
      program.dag, spatial.getNodePlacements(assignment.spatial),
      failureReason);
  if (mlir::failed(temporal))
    return mlir::failure();
  auto representation = CardPhysicalRepresentationDomain::create(
      program, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, failureReason);
  if (mlir::failed(representation))
    return mlir::failure();
  auto movement = CardDataMovementDomain::create(
      program, cardId, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, *representation,
      assignment.representation, failureReason);
  if (mlir::failed(movement))
    return mlir::failure();
  auto buffering = CardBufferingDomain::create(
      program, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, *representation,
      assignment.representation, *movement, assignment.movement,
      failureReason);
  if (mlir::failed(buffering))
    return mlir::failure();
  return buildSelectedBufferingScopes(program, *buffering, assignment.buffering,
                                      *movement, assignment.movement,
                                      program.availableTileIds, failureReason);
}

} // namespace wafer::compiler::detail
