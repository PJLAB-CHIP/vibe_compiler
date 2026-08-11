//===- PhysicalTileFinalization.cpp - Exact Tile finalization -----------===//

#include "PhysicalTileFinalization.h"
#include "SelectedBufferMaterialization.h"

#include "Wafer/Support/CompileTiming.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetIdentity.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace wafer::compiler::detail {
namespace {

static bool hasPrematureWholeCardFacts(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
      found |= static_cast<bool>(
          allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName));
    if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
      found |= send.getBinding().has_value();
    if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
      found |= recv.getBinding().has_value();
  });
  return found;
}

} // namespace

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> finalizePhysicalTileModule(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    PhysicalTileFinalizationFailure *failure,
    llvm::ArrayRef<SelectedBufferRequest> selectedBufferRequests,
    unsigned *materializedSlotAllocationCount,
    SelectedBufferMaterializationFailure *selectedBufferFailure) {
  if (failure)
    *failure = {};
  if (materializedSlotAllocationCount)
    *materializedSlotAllocationCount = 0;
  if (selectedBufferFailure)
    *selectedBufferFailure = {};
  if (!module) {
    if (failure)
      failure->kind = PhysicalTileFinalizationFailureKind::Contract;
    return mlir::failure();
  }

  auto recordFailure = [&](PhysicalTileFinalizationFailureKind kind) {
    if (failure)
      failure->kind = kind;
  };
  if (containsTileDataflowOperations(module->getOperation())) {
    recordFailure(PhysicalTileFinalizationFailureKind::Contract);
    module->emitError()
        << "physical_tile_finalization_requires_canonical_instr_action: Tile "
           "dataflow must be lowered exactly once while constructing the "
           "canonical Instr parent";
    return mlir::failure();
  }
  if (hasPrematureWholeCardFacts(*module)) {
    recordFailure(PhysicalTileFinalizationFailureKind::PrematureWholeCardFacts);
    module->emitError()
        << "physical_tile_finalization_contains_premature_whole_card_facts: "
           "DDR "
           "placement and "
           "Direct DTE bindings must be recomputed by whole-card admission";
    return mlir::failure();
  }

  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::PhysicalTileFinalization);
  clearPhysicalTileCandidateFacts(*module);
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(PhysicalTileFinalizationFailureKind::Verification);
    module->emitError(
        "executable-finalization Instr parent failed verification");
    return mlir::failure();
  }

  mlir::PassManager preparation(module->getContext());
  wafer::support::attachCompileTiming(preparation, "physical-tile-preparation");
  wafer::buildPreparePhysicalTileCandidatePipeline(preparation);
  if (mlir::failed(preparation.run(*module))) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::FunctionBoundaryBufferization);
    return mlir::failure();
  }
  if (mlir::failed(rebuildMinimumNCCJoins(*module))) {
    recordFailure(PhysicalTileFinalizationFailureKind::Completion);
    return mlir::failure();
  }
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(PhysicalTileFinalizationFailureKind::Verification);
    return mlir::failure();
  }

  // Buffer multiplicity is selected by the whole-card state, but its exact
  // rotating allocation can only be derived after function-boundary
  // preparation and completion normalization expose the canonical Instr loop.
  // It must still run before SPM planning so every cloned slot receives a
  // fresh, nonoverlapping physical allocation below.
  if (!selectedBufferRequests.empty() &&
      mlir::failed(materializeSelectedBuffering(module, selectedBufferRequests,
                                                materializedSlotAllocationCount,
                                                selectedBufferFailure))) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::SelectedBufferMaterialization);
    return mlir::failure();
  }

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  SPMMemoryPlanningFailure spmFailure;
  mlir::LogicalResult spmResult = mlir::success();
  {
    wafer::support::ScopedCompileTimingSpan spmTiming(
        "pass", "physical-tile-spm-planning", "wafer-plan-spm-memory");
    spmResult = planSPMMemoryModule(*module, memory.spmBase, memory.spmLimit,
                                    memory.spmAlignment, &spmFailure);
    if (mlir::failed(spmResult))
      spmTiming.markFailed();
  }
  if (mlir::failed(spmResult)) {
    recordFailure(PhysicalTileFinalizationFailureKind::SPMAllocation);
    if (failure)
      failure->spmCapacityOverflow =
          spmFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
    if (failure) {
      failure->spmLargestDemandLocation = spmFailure.largestDemandLocation;
      failure->spmLargestDemandType = spmFailure.largestDemandType;
      failure->spmLargestDemandBytes = spmFailure.largestDemandBytes;
      failure->spmDemandCount = spmFailure.demandCount;
      failure->spmLargestDemands.reserve(spmFailure.largestDemands.size());
      for (const SPMMemoryPlanningFailure::DemandEvidence &demand :
           spmFailure.largestDemands) {
        PhysicalTileFinalizationFailure::SPMDemandEvidence evidence{
            demand.location, demand.type, demand.bytes, {}};
        evidence.userLocations.append(demand.userLocations.begin(),
                                      demand.userLocations.end());
        failure->spmLargestDemands.push_back(std::move(evidence));
      }
      failure->spmCapacityConflictDemands.reserve(
          spmFailure.capacityConflictDemands.size());
      for (const SPMMemoryPlanningFailure::DemandEvidence &demand :
           spmFailure.capacityConflictDemands) {
        PhysicalTileFinalizationFailure::SPMDemandEvidence evidence{
            demand.location, demand.type, demand.bytes, {}};
        evidence.userLocations.append(demand.userLocations.begin(),
                                      demand.userLocations.end());
        failure->spmCapacityConflictDemands.push_back(std::move(evidence));
      }
      failure->spmIndividuallyOversizedDemands.reserve(
          spmFailure.individuallyOversizedDemands.size());
      for (const SPMMemoryPlanningFailure::DemandEvidence &demand :
           spmFailure.individuallyOversizedDemands) {
        PhysicalTileFinalizationFailure::SPMDemandEvidence evidence{
            demand.location, demand.type, demand.bytes, {}};
        evidence.userLocations.append(demand.userLocations.begin(),
                                      demand.userLocations.end());
        failure->spmIndividuallyOversizedDemands.push_back(std::move(evidence));
      }
    }
    return mlir::failure();
  }
  mlir::PassManager spmCleanup(module->getContext());
  wafer::support::attachCompileTiming(spmCleanup, "physical-tile-spm-planning");
  spmCleanup.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(spmCleanup.run(*module))) {
    recordFailure(PhysicalTileFinalizationFailureKind::SPMAllocation);
    return mlir::failure();
  }
  if (hasPrematureWholeCardFacts(*module)) {
    recordFailure(PhysicalTileFinalizationFailureKind::PrematureWholeCardFacts);
    module->emitError(
        "physical_tile_finalization_created_premature_whole_card_facts");
    return mlir::failure();
  }

  if (failure)
    *failure = {};
  return std::move(module);
}

} // namespace wafer::compiler::detail
