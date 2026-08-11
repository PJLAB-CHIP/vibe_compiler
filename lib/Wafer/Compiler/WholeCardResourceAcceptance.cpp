//===- WholeCardResourceAcceptance.cpp - Physical Tile resource gate ---===//

#include "WholeCardResourceAcceptance.h"

#include "AcceptedCallClosure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace wafer::compiler::detail {

mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeCardResources(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
                         llvm::ArrayRef<PhysicalTileId> physicalTileIds,
                         const ExecutionConfig &executionConfig) {
  if (physicalTileModules.empty())
    return mlir::failure();
  if (physicalTileModules.size() != physicalTileIds.size() ||
      physicalTileModules.size() !=
          static_cast<size_t>(executionConfig.getPhysicalTileCount()))
    return mlir::ModuleOp(physicalTileModules.front()).emitOpError()
           << "whole_card_resource_acceptance: physical Tile domain has "
           << physicalTileModules.size()
           << " modules but ExecutionConfig requires "
           << executionConfig.getPhysicalTileCount();

  llvm::SmallVector<int64_t, 16> sortedTileIds;
  sortedTileIds.reserve(physicalTileIds.size());
  for (PhysicalTileId tileId : physicalTileIds)
    sortedTileIds.push_back(tileId.getValue());
  llvm::sort(sortedTileIds);
  for (auto [expected, actual] : llvm::enumerate(sortedTileIds)) {
    if (actual == static_cast<int64_t>(expected))
      continue;
    return mlir::ModuleOp(physicalTileModules.front()).emitOpError()
           << "whole_card_resource_acceptance: physical Tile ids must cover "
              "the complete domain [0, "
           << executionConfig.getPhysicalTileCount() << "), got " << actual
           << " at sorted position " << expected;
  }

  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy();
  mlir::ModuleOp diagnosticAnchor = physicalTileModules.front();
  if (policy.spmAddressLimit < policy.spmAddressBase)
    return diagnosticAnchor.emitOpError(
        "whole_card_resource_acceptance: target SPM range is invalid");

  llvm::SmallVector<analysis::PhysicalTileInstructionProgram, 16> tilePrograms;
  tilePrograms.reserve(physicalTileModules.size());
  for (auto [module, tileId] :
       llvm::zip_equal(physicalTileModules, physicalTileIds)) {
    mlir::ModuleOp diagnosticModule = module;
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(module);
    if (!closure)
      return diagnosticModule.emitOpError()
             << "whole_card_resource_acceptance: accepted call closure "
                "is invalid: "
             << llvm::toString(closure.takeError());
    tilePrograms.push_back({tileId, closure->entry.getOperation()});
  }
  analysis::WholeCardInstructionProgramCost cost =
      analysis::analyzeWholeCardInstructionProgramCost(tilePrograms, policy);

  const uint64_t perTileSPMCapacity =
      policy.spmAddressLimit - policy.spmAddressBase;
  for (auto &&[tile, tileCost] : llvm::enumerate(cost.tileCosts)) {
    mlir::ModuleOp tileModule = physicalTileModules[tile];
    const int64_t tileId = physicalTileIds[tile].getValue();
    if (!tileCost.spmHighWaterBytes.isKnown())
      return tileModule.emitOpError()
             << "whole_card_resource_acceptance: physical Tile " << tileId
             << " has no exact accepted SPM high-water";
    if (tileCost.spmHighWaterBytes.value > perTileSPMCapacity)
      return tileModule.emitOpError()
             << "whole_card_resource_acceptance: physical Tile " << tileId
             << " SPM high-water " << tileCost.spmHighWaterBytes.value
             << " exceeds per-Tile capacity " << perTileSPMCapacity;
  }

  // Performance-only work collectors may be unavailable. They remain raw
  // analysis data and the numeric estimator disables the corresponding term
  // uniformly for the complete comparison cohort. Typed transport validation,
  // rather than a guessed byte count, owns message legality; this check only
  // adds an exact consistency assertion when both byte totals are available.
  if (cost.aggregateNoC.aggregateTransmitBytes.isKnown() &&
      cost.aggregateNoC.aggregateReceiveBytes.isKnown() &&
      cost.aggregateNoC.aggregateTransmitBytes.value !=
          cost.aggregateNoC.aggregateReceiveBytes.value)
    return diagnosticAnchor.emitOpError()
           << "whole_card_resource_acceptance: matched whole-card NoC "
              "transmit and receive bytes differ ("
           << cost.aggregateNoC.aggregateTransmitBytes.value << " vs "
           << cost.aggregateNoC.aggregateReceiveBytes.value << ")";

  return cost;
}

} // namespace wafer::compiler::detail

namespace wafer::compiler::testing {

mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeCardResources(llvm::ArrayRef<mlir::ModuleOp> physicalTileModules,
                         llvm::ArrayRef<PhysicalTileId> physicalTileIds,
                         const ExecutionConfig &executionConfig) {
  return detail::acceptWholeCardResources(physicalTileModules, physicalTileIds,
                                          executionConfig);
}

} // namespace wafer::compiler::testing
