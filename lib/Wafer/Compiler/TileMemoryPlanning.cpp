//===- TileMemoryPlanning.cpp - Tile memory planning ----===//

#include "TileMemoryPlanning.h"
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

static bool hasPreexistingPlacementFacts(mlir::ModuleOp module) {
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

static bool hasDDROrTransportAssignments(mlir::ModuleOp module) {
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

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
planTileMemory(mlir::OwningOpRef<mlir::ModuleOp> module,
               TileMemoryPlanningFailure *failure,
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
      failure->kind = TileMemoryPlanningFailureKind::Contract;
    return mlir::failure();
  }

  auto recordFailure = [&](TileMemoryPlanningFailureKind kind) {
    if (failure)
      failure->kind = kind;
  };
  auto requireCurrentBufferRelations = [&](llvm::StringRef stage) {
    if (!materializationRelations ||
        mlir::succeeded(checkStructuredBufferRelationsCurrent(
            module->getOperation(), *materializationRelations)))
      return mlir::success();
    module->emitError() << "structured_buffer_relation_outside_current_ir: "
                        << stage;
    return mlir::failure();
  };
  if (containsTileDataflowOperations(module->getOperation())) {
    recordFailure(TileMemoryPlanningFailureKind::Contract);
    module->emitError()
        << "tile_memory_planning_requires_canonical_instr_ir: Tile "
           "dataflow must be lowered exactly once while constructing the "
           "canonical Instr parent";
    return mlir::failure();
  }
  if (hasPreexistingPlacementFacts(*module)) {
    recordFailure(TileMemoryPlanningFailureKind::PreexistingPlacementFacts);
    module->emitError()
        << "tile_memory_planning_contains_preexisting_placement_facts: "
           "SPM/DDR offsets and Direct DTE bindings are outputs "
           "of this or a downstream stage; the input must be the immutable "
           "canonical unplaced module";
    return mlir::failure();
  }

  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::TileMemoryPlanning);
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(TileMemoryPlanningFailureKind::Verification);
    module->emitError("Tile memory-planning input failed verification");
    return mlir::failure();
  }
  if (mlir::failed(requireCurrentBufferRelations("memory-planning input"))) {
    recordFailure(TileMemoryPlanningFailureKind::Contract);
    return mlir::failure();
  }

  if (mlir::failed(
          runPassPipeline(*module, "instr-memory-planning-preparation",
                          wafer::buildPrepareInstrForMemoryPlanningPipeline))) {
    recordFailure(
        TileMemoryPlanningFailureKind::InstrMemoryPlanningPreparation);
    return mlir::failure();
  }
  if (materializationRelations)
    retainCurrentStructuredBufferRelations(module->getOperation(),
                                           *materializationRelations);
  if (mlir::failed(
          requireCurrentBufferRelations("memory-planning preparation"))) {
    recordFailure(
        TileMemoryPlanningFailureKind::InstrMemoryPlanningPreparation);
    return mlir::failure();
  }
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(TileMemoryPlanningFailureKind::Verification);
    return mlir::failure();
  }

  // Buffer multiplicity is selected by the card state, but its exact
  // rotating allocation can only be derived after function-boundary
  // preparation and required-join recomputation expose the canonical Instr
  // loop.
  // It must still run before SPM planning so every cloned slot receives a
  // fresh, nonoverlapping physical allocation below.
  if (!selectedBufferRequests.empty() &&
      mlir::failed(materializeSelectedBuffering(
          module, selectedBufferRequests, materializationRelations,
          materializedSlotAllocationCount, selectedBufferFailure))) {
    recordFailure(TileMemoryPlanningFailureKind::SelectedBufferMaterialization);
    return mlir::failure();
  }
  if (mlir::failed(
          requireCurrentBufferRelations("selected buffer materialization"))) {
    recordFailure(TileMemoryPlanningFailureKind::SelectedBufferMaterialization);
    return mlir::failure();
  }

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  SPMMemoryPlanningFailure spmFailure;
  PlanSPMMemoryPassOptions spmOptions;
  spmOptions.spmBase = memory.spmBase;
  spmOptions.spmLimit = memory.spmLimit;
  spmOptions.spmAlignment = memory.spmAlignment;
  mlir::LogicalResult spmResult = runPassPipeline(
      *module, "tile-spm-planning", [&](mlir::OpPassManager &manager) {
        wafer::addAssignSPMOffsetsPass(manager, spmOptions, &spmFailure);
      });
  if (mlir::failed(spmResult)) {
    recordFailure(TileMemoryPlanningFailureKind::SPMAllocation);
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
        TileMemoryPlanningFailure::SPMDemandEvidence evidence{
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
    recordFailure(TileMemoryPlanningFailureKind::Contract);
    return mlir::failure();
  }
  if (hasDDROrTransportAssignments(*module)) {
    recordFailure(TileMemoryPlanningFailureKind::PreexistingPlacementFacts);
    module->emitError(
        "tile_memory_planning_created_ddr_or_transport_assignments");
    return mlir::failure();
  }

  if (failure)
    *failure = {};
  return std::move(module);
}

} // namespace wafer::compiler::detail
