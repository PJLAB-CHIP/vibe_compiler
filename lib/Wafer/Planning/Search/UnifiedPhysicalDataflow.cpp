//===- UnifiedPhysicalDataflow.cpp - Joint physical assignment --------===//

#include "Wafer/Planning/Search/UnifiedPhysicalDataflow.h"

#include <type_traits>
#include <variant>

namespace wafer::compiler::detail {

mlir::FailureOr<UnifiedPhysicalDataflowDomain>
UnifiedPhysicalDataflowDomain::create(const CardProgramAnalysis &program,
                                      CardId cardId,
                                      std::string *failureReason) {
  SpatialPlanDomainResult spatial =
      buildSpatialPlanDomain(program.dag, program.topology, cardId);
  auto implementation =
      CardComputeImplementationDomain::create(program, failureReason);
  if (!spatial.succeeded() || mlir::failed(implementation)) {
    if (!spatial.succeeded() && failureReason && spatial.failure)
      *failureReason = spatial.failure->detail;
    return mlir::failure();
  }
  return UnifiedPhysicalDataflowDomain(
      program, cardId, std::move(*spatial.domain), std::move(*implementation));
}

mlir::FailureOr<ClosedSpatialDemand>
UnifiedPhysicalDataflowDomain::getSpatialDemand(
    const SpatialPlan &assignment, std::string *failureReason) const {
  SpatialDomainEvaluation evaluation =
      spatial.evaluate(program.dag, assignment);
  const analysis::ExactDemandProof *proof =
      evaluation.demand ? analysis::getExactDemandProof(*evaluation.demand)
                        : nullptr;
  if (!evaluation.assignment || !proof) {
    if (failureReason) {
      if (evaluation.failure)
        *failureReason = evaluation.failure->detail;
      else if (evaluation.demand)
        *failureReason = std::visit(
            [](const auto &value) -> std::string {
              using T = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
                return "spatial demand proof is unavailable";
              else
                return value.detail;
            },
            *evaluation.demand);
      else
        *failureReason = "spatial demand evaluation produced no outcome";
    }
    return mlir::failure();
  }
  return ClosedSpatialDemand{std::move(*evaluation.assignment), *proof};
}

mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
UnifiedPhysicalDataflowDomain::getFirstForSpatial(
    const SpatialPlan &spatialAssignment, std::string *failureReason,
    bool fusionOriented) const {
  auto closed = getSpatialDemand(spatialAssignment, failureReason);
  if (mlir::failed(closed))
    return mlir::failure();
  auto coupled = CoupledRegionDomain::create(program.dag, closed->spatial,
                                             closed->demand, failureReason);
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
      *temporal, temporalAssignment, *representation, representationAssignment,
      *movement, movementAssignment, failureReason);
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
  SpatialPlan current = spatial.getFirstPlan();
  while (true) {
    if (auto assignment = getFirstForSpatial(current, failureReason);
        mlir::succeeded(assignment))
      return assignment;
    SpatialPlanSuccessor next = spatial.getNextPlan(current);
    if (next.kind == SpatialPlanSuccessorKind::Failure && failureReason &&
        next.failure)
      *failureReason = next.failure->detail;
    if (next.kind != SpatialPlanSuccessorKind::Successor || !next.plan)
      return mlir::failure();
    current = std::move(*next.plan);
  }
}

mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
UnifiedPhysicalDataflowDomain::getConstructiveAssignment(
    std::string *failureReason) const {
  for (const SpatialPlan &proposal : spatial.getProposals())
    if (auto assignment = getFirstForSpatial(proposal, failureReason);
        mlir::succeeded(assignment))
      return assignment;
  return mlir::failure();
}

mlir::FailureOr<UnifiedPhysicalDataflowAssignment>
UnifiedPhysicalDataflowDomain::getFusionOrientedAssignment(
    std::string *failureReason) const {
  for (const SpatialPlan &proposal : spatial.getProposals())
    if (auto assignment = getFirstForSpatial(proposal, failureReason,
                                             /*fusionOriented=*/true);
        mlir::succeeded(assignment))
      return assignment;
  return mlir::failure();
}

bool UnifiedPhysicalDataflowDomain::contains(
    const UnifiedPhysicalDataflowAssignment &assignment) const {
  if (!spatial.contains(assignment.spatial) ||
      !implementation.contains(assignment.implementation))
    return false;
  auto closed = getSpatialDemand(assignment.spatial, nullptr);
  if (mlir::failed(closed))
    return false;
  auto coupled =
      CoupledRegionDomain::create(program.dag, closed->spatial, closed->demand);
  auto temporal = CardTemporalDomain::create(
      program.dag, spatial.getNodePlacements(assignment.spatial));
  if (mlir::failed(coupled) || mlir::failed(temporal) ||
      !coupled->contains(assignment.coupled) ||
      !temporal->contains(assignment.temporal))
    return false;
  auto representation = CardPhysicalRepresentationDomain::create(
      program, closed->spatial, closed->demand, *coupled, assignment.coupled,
      *temporal, assignment.temporal);
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
      program, closed->spatial, closed->demand, *coupled, assignment.coupled,
      *temporal, assignment.temporal, *representation,
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
  auto coupled = *CoupledRegionDomain::create(program.dag, closed.spatial,
                                              closed.demand, failureReason);
  auto temporal = *CardTemporalDomain::create(
      program.dag, spatial.getNodePlacements(assignment.spatial),
      failureReason);
  auto representation = *CardPhysicalRepresentationDomain::create(
      program, closed.spatial, closed.demand, coupled, assignment.coupled,
      temporal, assignment.temporal, failureReason);
  auto movement = *CardDataMovementDomain::create(
      program, cardId, closed.spatial, closed.demand, coupled,
      assignment.coupled, temporal, assignment.temporal, representation,
      assignment.representation, failureReason);
  auto buffering = *CardBufferingDomain::create(
      program, closed.spatial, closed.demand, coupled, assignment.coupled,
      temporal, assignment.temporal, representation, assignment.representation,
      movement, assignment.movement, failureReason);

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
        program, closed.spatial, closed.demand, coupled, next.coupled, temporal,
        next.temporal, failureReason);
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
        program, closed.spatial, closed.demand, coupled, next.coupled, temporal,
        next.temporal, *resetRepresentation, next.representation,
        *resetMovement, next.movement, failureReason);
    if (mlir::failed(resetBuffering))
      return mlir::failure();
    next.buffering = resetBuffering->getFirstAssignment();
    return std::optional<UnifiedPhysicalDataflowAssignment>(std::move(next));
  }

  SpatialPlan spatialAssignment = assignment.spatial;
  while (true) {
    SpatialPlanSuccessor advanced = spatial.getNextPlan(spatialAssignment);
    if (advanced.kind == SpatialPlanSuccessorKind::Failure) {
      if (failureReason && advanced.failure)
        *failureReason = advanced.failure->detail;
      return mlir::failure();
    }
    if (advanced.kind == SpatialPlanSuccessorKind::End || !advanced.plan)
      return std::optional<UnifiedPhysicalDataflowAssignment>{};
    spatialAssignment = std::move(*advanced.plan);
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
  auto coupled = CoupledRegionDomain::create(program.dag, closed->spatial,
                                             closed->demand, failureReason);
  if (mlir::failed(coupled))
    return mlir::failure();
  auto temporal = CardTemporalDomain::create(
      program.dag, spatial.getNodePlacements(assignment.spatial),
      failureReason);
  if (mlir::failed(temporal))
    return mlir::failure();
  auto representation = CardPhysicalRepresentationDomain::create(
      program, closed->spatial, closed->demand, *coupled, assignment.coupled,
      *temporal, assignment.temporal, failureReason);
  if (mlir::failed(representation))
    return mlir::failure();
  auto movement = CardDataMovementDomain::create(
      program, cardId, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, *representation,
      assignment.representation, failureReason);
  if (mlir::failed(movement))
    return mlir::failure();
  return materializeCardCoupledRegionsWithImplementations(
      tensorProgram, program, cardId, closed->spatial, closed->demand, *coupled,
      assignment.coupled, *temporal, assignment.temporal, *representation,
      assignment.representation, implementation, assignment.implementation,
      *movement, assignment.movement, failureReason);
}

} // namespace wafer::compiler::detail
