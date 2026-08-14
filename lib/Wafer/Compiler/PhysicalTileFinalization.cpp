//===- PhysicalTileFinalization.cpp - Exact Tile finalization -----------===//

#include "PhysicalTileFinalization.h"
#include "CompilationInternal.h"
#include "SelectedBufferMaterialization.h"
#include "StructuredBufferRelations.h"

#include "Wafer/Support/CompileTiming.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetIdentity.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"

namespace wafer::compiler::detail {
namespace {

static bool hasPreexistingPhysicalAssignments(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
      found |= static_cast<bool>(
          allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName));
      found |= static_cast<bool>(
          allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName));
    }
    if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
      found |= send.getBinding().has_value();
    if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
      found |= recv.getBinding().has_value();
  });
  return found;
}

static bool hasWholeCardAssignments(mlir::ModuleOp module) {
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
    StructuredMaterializationRelations *materializationRelations,
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
  auto requireCurrentBufferRelations =
      [&](llvm::StringRef stage) {
        if (!materializationRelations ||
            mlir::succeeded(checkStructuredBufferRelationsCurrent(
                module->getOperation(), *materializationRelations)))
          return mlir::success();
        module->emitError()
            << "structured_buffer_relation_outside_current_artifact: " << stage;
        return mlir::failure();
      };
  if (containsTileDataflowOperations(module->getOperation())) {
    recordFailure(PhysicalTileFinalizationFailureKind::Contract);
    module->emitError()
        << "physical_tile_finalization_requires_canonical_instr_action: Tile "
           "dataflow must be lowered exactly once while constructing the "
           "canonical Instr parent";
    return mlir::failure();
  }
  if (hasPreexistingPhysicalAssignments(*module)) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::PreexistingPhysicalAssignments);
    module->emitError()
        << "physical_tile_finalization_contains_preexisting_physical_"
           "assignments: SPM/DDR offsets and Direct DTE bindings are outputs "
           "of this or a downstream stage; the input must be the immutable "
           "canonical unplaced artifact";
    return mlir::failure();
  }

  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::PhysicalTileFinalization);
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(PhysicalTileFinalizationFailureKind::Verification);
    module->emitError(
        "executable-finalization Instr parent failed verification");
    return mlir::failure();
  }
  if (mlir::failed(requireCurrentBufferRelations("finalization input"))) {
    recordFailure(PhysicalTileFinalizationFailureKind::Contract);
    return mlir::failure();
  }

  if (mlir::failed(runPassPipeline(
          *module, "instr-memory-planning-preparation",
          wafer::buildPrepareInstrForMemoryPlanningPipeline))) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::InstrMemoryPlanningPreparation);
    return mlir::failure();
  }
  if (materializationRelations)
    retainCurrentStructuredBufferRelations(module->getOperation(),
                                           *materializationRelations);
  if (mlir::failed(
          requireCurrentBufferRelations("memory-planning preparation"))) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::InstrMemoryPlanningPreparation);
    return mlir::failure();
  }
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(PhysicalTileFinalizationFailureKind::Verification);
    return mlir::failure();
  }

  // Buffer multiplicity is selected by the whole-card state, but its exact
  // rotating allocation can only be derived after function-boundary
  // preparation and required-join recomputation expose the canonical Instr
  // loop.
  // It must still run before SPM planning so every cloned slot receives a
  // fresh, nonoverlapping physical allocation below.
  if (!selectedBufferRequests.empty() &&
      mlir::failed(materializeSelectedBuffering(
          module, selectedBufferRequests, materializationRelations,
          materializedSlotAllocationCount, selectedBufferFailure))) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::SelectedBufferMaterialization);
    return mlir::failure();
  }
  if (mlir::failed(
          requireCurrentBufferRelations("selected buffer materialization"))) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::SelectedBufferMaterialization);
    return mlir::failure();
  }

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  SPMMemoryPlanningFailure spmFailure;
  PlanSPMMemoryPassOptions spmOptions;
  spmOptions.spmBase = memory.spmBase;
  spmOptions.spmLimit = memory.spmLimit;
  spmOptions.spmAlignment = memory.spmAlignment;
  mlir::LogicalResult spmResult = runPassPipeline(
      *module, "physical-tile-spm-planning",
      [&](mlir::OpPassManager &manager) {
        wafer::addAssignSPMOffsetsPass(manager, spmOptions, &spmFailure);
      });
  if (mlir::failed(spmResult)) {
    recordFailure(PhysicalTileFinalizationFailureKind::SPMAllocation);
    if (failure)
      failure->spmCapacityOverflow =
          spmFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
    if (failure) {
      failure->spmPlanningFailureKind = spmFailure.kind;
      failure->spmLargestDemandLocation = spmFailure.largestDemandLocation;
      failure->spmLargestDemandType = spmFailure.largestDemandType;
      failure->spmLargestDemandBytes = spmFailure.largestDemandBytes;
      failure->spmDemandCount = spmFailure.demandCount;
      auto convertEvidence = [&](const auto &demand) {
        PhysicalTileFinalizationFailure::SPMDemandEvidence evidence{
            demand.location, demand.allocation, demand.type, demand.bytes, {}};
        evidence.userLocations.append(demand.userLocations.begin(),
                                      demand.userLocations.end());
        if (materializationRelations) {
          auto appendNode = [&](auto &nodes, uint32_t node) {
            if (!llvm::is_contained(nodes, node))
              nodes.push_back(node);
          };
          for (const auto &relation :
               materializationRelations->operationResultBuffers)
            if (shareStructuredBufferStorage(demand.allocation,
                                             relation.buffer))
              appendNode(evidence.operationResultNodes,
                         relation.structuredNodeId);
          for (const auto &relation : materializationRelations->operandBuffers)
            if (shareStructuredBufferStorage(demand.allocation,
                                             relation.buffer))
              appendNode(evidence.operandDemandNodes,
                         relation.structuredNodeId);
          for (const auto &relation : materializationRelations->outputBuffers)
            if (shareStructuredBufferStorage(demand.allocation,
                                             relation.buffer) &&
                !llvm::is_contained(evidence.outputIndices,
                                    relation.outputIndex))
              evidence.outputIndices.push_back(relation.outputIndex);
        }
        return evidence;
      };
      failure->spmLargestDemands.reserve(spmFailure.largestDemands.size());
      for (const auto &demand : spmFailure.largestDemands)
        failure->spmLargestDemands.push_back(convertEvidence(demand));
      failure->spmCapacityConflictDemands.reserve(
          spmFailure.capacityConflictDemands.size());
      for (const auto &demand : spmFailure.capacityConflictDemands)
        failure->spmCapacityConflictDemands.push_back(convertEvidence(demand));
      failure->spmIndividuallyOversizedDemands.reserve(
          spmFailure.individuallyOversizedDemands.size());
      for (const auto &demand : spmFailure.individuallyOversizedDemands)
        failure->spmIndividuallyOversizedDemands.push_back(
            convertEvidence(demand));
    }
    return mlir::failure();
  }
  if (mlir::failed(requireCurrentBufferRelations("SPM offset assignment"))) {
    recordFailure(PhysicalTileFinalizationFailureKind::Contract);
    return mlir::failure();
  }
  if (hasWholeCardAssignments(*module)) {
    recordFailure(
        PhysicalTileFinalizationFailureKind::PreexistingPhysicalAssignments);
    module->emitError(
        "physical_tile_finalization_created_premature_whole_card_facts");
    return mlir::failure();
  }

  if (failure)
    *failure = {};
  return std::move(module);
}

} // namespace wafer::compiler::detail
