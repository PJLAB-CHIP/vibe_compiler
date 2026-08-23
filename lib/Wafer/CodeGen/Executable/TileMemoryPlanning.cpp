//===- TileMemoryPlanning.cpp - Tile memory planning ----===//

#include "Wafer/CodeGen/Executable/TileMemoryPlanning.h"
#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Planning/Search/InstructionSchedule.h"
#include "Wafer/Planning/Search/StagePipeline.h"
#include "Wafer/Transforms/Bufferization/SelectedBufferMaterialization.h"

#include "Wafer/Support/CompileTiming.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Target/Core/TargetIdentity.h"
#include "Wafer/Target/Core/TargetMemory.h"
#include "Wafer/Transforms/MemoryPlanningPipelines.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

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

TileMemoryPlanningFailure convertSPMMemoryPlanningFailure(
    const SPMMemoryPlanningFailure &spmFailure,
    const StructuredMaterializationRelations &relations) {
  TileMemoryPlanningFailure failure;
  failure.kind = TileMemoryPlanningFailureKind::SPMAllocation;
  failure.spmCapacityOverflow =
      spmFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
  failure.spmPlanningFailureKind = spmFailure.kind;
  failure.spmLargestDemandLocation = spmFailure.largestDemandLocation;
  failure.spmLargestDemandType = spmFailure.largestDemandType;
  failure.spmLargestDemandBytes = spmFailure.largestDemandBytes;
  failure.spmDemandCount = spmFailure.demandCount;
  StorageRootMemo storageRoots;
  struct OwnersByStorageRoot {
    llvm::DenseMap<mlir::Value, llvm::SmallVector<uint32_t, 2>> resultNodes;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<uint32_t, 2>> operandNodes;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<uint32_t, 2>> scratchNodes;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<unsigned, 2>> outputs;
  } owners;
  auto appendUnique = [](auto &values, auto value) {
    if (!llvm::is_contained(values, value))
      values.push_back(value);
  };
  auto indexNodeRelation = [&](const StructuredOperationBufferRelation &entry,
                               auto &index) {
    for (mlir::Value root : storageRoots.getStorageRoots(entry.buffer))
      appendUnique(index[root], entry.structuredNodeId);
  };
  for (const auto &relation : relations.operationResultBuffers)
    indexNodeRelation(relation, owners.resultNodes);
  for (const auto &relation : relations.operandBuffers)
    indexNodeRelation(relation, owners.operandNodes);
  for (const auto &relation : relations.scratchBuffers)
    indexNodeRelation(relation, owners.scratchNodes);
  for (const SpatialOutputBufferRelation &relation : relations.outputBuffers)
    for (mlir::Value root : storageRoots.getStorageRoots(relation.buffer))
      appendUnique(owners.outputs[root], relation.outputIndex);

  auto convertEvidence = [&](const auto &demand) {
    TileMemoryPlanningFailure::SPMDemandEvidence evidence{
        demand.location, demand.type, demand.bytes, {}};
    evidence.userLocations.append(demand.userLocations.begin(),
                                  demand.userLocations.end());
    evidence.userOperationNames.append(demand.userOperationNames.begin(),
                                       demand.userOperationNames.end());
    // A staged DDR wave is written by the producer's seal store and read by
    // the consumer's reload; neither endpoint aliases the wave buffer
    // through SSA views, so a direct storage-root match misses both. The
    // witnesses are the typed transfer endpoints of the exact allocation,
    // never a shape/type guess.
    llvm::SmallVector<mlir::Value, 8> witnesses;
    witnesses.push_back(demand.allocation);
    llvm::DenseSet<mlir::Value> visited;
    for (size_t index = 0; index < witnesses.size(); ++index) {
      mlir::Value value = witnesses[index];
      if (!value || !visited.insert(value).second)
        continue;
      for (mlir::Operation *user : value.getUsers()) {
        if (auto store = mlir::dyn_cast<StorageStoreOp>(user)) {
          if (store.getDest() == value)
            witnesses.push_back(store.getSource());
          if (store.getSource() == value)
            witnesses.push_back(store.getDest());
        } else if (auto load = mlir::dyn_cast<StorageLoadOp>(user)) {
          if (load.getSource() == value)
            witnesses.push_back(load.getDest());
          if (load.getDest() == value)
            witnesses.push_back(load.getSource());
        } else if (auto view =
                       mlir::dyn_cast<mlir::ViewLikeOpInterface>(user)) {
          for (mlir::Value result : view->getResults())
            witnesses.push_back(result);
        } else if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(user)) {
          // The typed SPM/DDR transfer endpoints of the exact allocation:
          // the upstream boundary source and the downstream layout dest are
          // both evidence of the same causal dataflow.
          witnesses.push_back(rdma.getSource());
          witnesses.push_back(rdma.getDest());
        } else if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(user)) {
          witnesses.push_back(wdma.getSource());
          witnesses.push_back(wdma.getDest());
        } else if (auto scatter = mlir::dyn_cast<InstrGatherScatterOp>(user)) {
          witnesses.push_back(scatter.getSource());
          witnesses.push_back(scatter.getDest());
        } else if (auto dataMove = mlir::dyn_cast<InstrTDMADataMoveOp>(user)) {
          witnesses.push_back(dataMove.getSource());
          witnesses.push_back(dataMove.getDest());
        } else if (mlir::isa<WaferInstructionOpInterface>(user)) {
          // Lowering-created compute temporaries may be connected to their
          // structured owner only through the actual instruction dataflow:
          // movement fills an input, then GEMM/reduce/elementwise consumes it
          // and writes a result buffer that already has a typed relation.
          // Follow only explicit memref SSA operands/results. Operation names,
          // locations and shapes remain diagnostic facts and never select an
          // owner.
          for (mlir::Value operand : user->getOperands())
            if (mlir::isa<mlir::BaseMemRefType>(operand.getType()))
              witnesses.push_back(operand);
          for (mlir::Value result : user->getResults())
            if (mlir::isa<mlir::BaseMemRefType>(result.getType()))
              witnesses.push_back(result);
        }
      }
    }
    auto appendIndexed = [&](auto &destination, const auto &index,
                             mlir::Value root) {
      auto found = index.find(root);
      if (found == index.end())
        return;
      for (auto value : found->second)
        appendUnique(destination, value);
    };
    for (mlir::Value witness : witnesses)
      for (mlir::Value root : storageRoots.getStorageRoots(witness)) {
        appendIndexed(evidence.operationResultNodes, owners.resultNodes, root);
        appendIndexed(evidence.operandDemandNodes, owners.operandNodes, root);
        appendIndexed(evidence.scratchNodes, owners.scratchNodes, root);
        appendIndexed(evidence.outputIndices, owners.outputs, root);
      }
    llvm::sort(evidence.operationResultNodes);
    llvm::sort(evidence.operandDemandNodes);
    llvm::sort(evidence.scratchNodes);
    llvm::sort(evidence.outputIndices);
    return evidence;
  };
  failure.spmLargestDemands.reserve(spmFailure.largestDemands.size());
  for (const auto &demand : spmFailure.largestDemands)
    failure.spmLargestDemands.push_back(convertEvidence(demand));
  failure.spmCapacityConflictDemands.reserve(
      spmFailure.capacityConflictDemands.size());
  for (const auto &demand : spmFailure.capacityConflictDemands)
    failure.spmCapacityConflictDemands.push_back(convertEvidence(demand));
  failure.spmIndividuallyOversizedDemands.reserve(
      spmFailure.individuallyOversizedDemands.size());
  for (const auto &demand : spmFailure.individuallyOversizedDemands)
    failure.spmIndividuallyOversizedDemands.push_back(convertEvidence(demand));
  return failure;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
planTileMemory(mlir::OwningOpRef<mlir::ModuleOp> module,
               TileMemoryPlanningFailure *failure,
               llvm::ArrayRef<SelectedBufferingScope> selectedBufferingScopes,
               StructuredMaterializationRelations *materializationRelations,
               unsigned *materializedSlotAllocationCount,
               SelectedBufferMaterializationFailure *selectedBufferFailure,
               bool applySelectedInstructionSchedule,
               bool emitSPMCapacityDiagnostics) {
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
  if (materializationRelations &&
      mlir::failed(rebaseStructuredBufferRelationsToStorageRoots(
          *materializationRelations))) {
    recordFailure(TileMemoryPlanningFailureKind::Contract);
    module->emitError(
        "structured buffer relation has no unique current storage root");
    return mlir::failure();
  }

  if (mlir::failed(
          runPassPipeline(*module, "instr-memory-planning-preparation",
                          wafer::buildPrepareInstrForMemoryPlanningPipeline))) {
    recordFailure(
        TileMemoryPlanningFailureKind::InstrMemoryPlanningPreparation);
    return mlir::failure();
  }
  // The preparation pipeline must preserve every attribution relation. A
  // relation whose buffer left the current IR is a probe/final evidence
  // contract violation and fails closed here instead of being silently
  // dropped from the certificate.
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
  if (!selectedBufferingScopes.empty()) {
    if (!materializationRelations) {
      if (selectedBufferFailure) {
        selectedBufferFailure->kind =
            SelectedBufferMaterializationFailureKind::InvalidRequest;
        selectedBufferFailure->detail =
            "selected buffering requires current materialization relations";
      }
      recordFailure(
          TileMemoryPlanningFailureKind::SelectedBufferMaterialization);
      return mlir::failure();
    }
    auto materialized = materializeStagePipelines(
        std::move(module), selectedBufferingScopes,
        std::move(*materializationRelations), selectedBufferFailure);
    if (mlir::failed(materialized)) {
      recordFailure(
          TileMemoryPlanningFailureKind::SelectedBufferMaterialization);
      return mlir::failure();
    }
    module = std::move(materialized->module);
    *materializationRelations =
        std::move(materialized->materializationRelations);
    if (materializedSlotAllocationCount)
      *materializedSlotAllocationCount = materialized->slotAllocationCount;
  }
  if (mlir::failed(
          requireCurrentBufferRelations("selected buffer materialization"))) {
    recordFailure(TileMemoryPlanningFailureKind::SelectedBufferMaterialization);
    return mlir::failure();
  }

  // Every complete candidate enters the selected instruction-schedule
  // query/apply boundary after its
  // final stage structure exists and before physical offsets are assigned.
  // The complete search owner may supply another assignment through the same
  // mechanism; this low-level exact gate consumes the canonical first point.
  if (applySelectedInstructionSchedule) {
    auto scheduleDomain = CardInstructionScheduleDomain::create(
        llvm::ArrayRef<TileInstructionModule>{
            TileInstructionModule{TileId(0), *module}});
    if (mlir::failed(scheduleDomain)) {
      recordFailure(TileMemoryPlanningFailureKind::InstructionScheduling);
      return mlir::failure();
    }
    CardInstructionScheduleAssignment schedule =
        scheduleDomain->getFirstAssignment();
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> scheduledModules;
    scheduledModules.push_back(std::move(module));
    auto scheduled = applyInstructionSchedule(std::move(scheduledModules),
                                              llvm::ArrayRef<TileId>{TileId(0)},
                                              *scheduleDomain, schedule);
    if (mlir::failed(scheduled) || scheduled->modules.size() != 1) {
      recordFailure(TileMemoryPlanningFailureKind::InstructionScheduling);
      return mlir::failure();
    }
    module = std::move(scheduled->modules.front());
    if (mlir::failed(
            requireCurrentBufferRelations("instruction schedule apply"))) {
      recordFailure(TileMemoryPlanningFailureKind::InstructionScheduling);
      return mlir::failure();
    }
  }

  const TargetMemoryPolicy memory = getTargetMemoryPolicy();
  SPMMemoryPlanningFailure spmFailure;
  PlanSPMMemoryPassOptions spmOptions;
  spmOptions.spmBase = memory.spmBase;
  spmOptions.spmLimit = memory.spmLimit;
  spmOptions.spmAlignment = memory.spmAlignment;
  spmOptions.emitCapacityDiagnostics = emitSPMCapacityDiagnostics;
  mlir::LogicalResult spmResult = runPassPipeline(
      *module, "tile-spm-planning", [&](mlir::OpPassManager &manager) {
        manager.addPass(
            wafer::createPlanSPMMemoryPassWithFailure(spmOptions, &spmFailure));
      });
  if (mlir::failed(spmResult)) {
    recordFailure(TileMemoryPlanningFailureKind::SPMAllocation);
    if (failure) {
      // The raw planner certificate is preserved completely on both entry
      // paths: structured relations only add owner attribution on top of the
      // full typed evidence, they never decide which evidence is copied.
      *failure = convertSPMMemoryPlanningFailure(
          spmFailure, materializationRelations
                          ? *materializationRelations
                          : StructuredMaterializationRelations{});
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
