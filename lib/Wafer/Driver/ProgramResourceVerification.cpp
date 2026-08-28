//===- ProgramResourceVerification.cpp - Program resource verification
//===//

#include "Wafer/Driver/ProgramResourceVerification.h"

#include "Wafer/Analysis/Module/ExecutableCallClosure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace wafer::compiler::detail {

mlir::FailureOr<analysis::InstructionProgramAggregateCost>
verifyProgramResources(llvm::ArrayRef<mlir::ModuleOp> inputModules,
                       llvm::ArrayRef<TileId> tileIds,
                       const ExecutionConfig &executionConfig) {
  if (inputModules.empty())
    return mlir::failure();
  if (inputModules.size() != tileIds.size() ||
      inputModules.size() !=
          static_cast<size_t>(executionConfig.getTileCount()))
    return mlir::ModuleOp(inputModules.front()).emitOpError()
           << "program_resource_verification: Tile domain has "
           << inputModules.size() << " modules but ExecutionConfig requires "
           << executionConfig.getTileCount();

  llvm::SmallVector<int64_t, 16> sortedTileIds;
  sortedTileIds.reserve(tileIds.size());
  for (TileId tileId : tileIds)
    sortedTileIds.push_back(tileId.getValue());
  llvm::sort(sortedTileIds);
  for (auto [expected, actual] : llvm::enumerate(sortedTileIds)) {
    if (actual == static_cast<int64_t>(expected))
      continue;
    return mlir::ModuleOp(inputModules.front()).emitOpError()
           << "program_resource_verification: Tile ids must cover "
              "the complete domain [0, "
           << executionConfig.getTileCount() << "), got " << actual
           << " at sorted position " << expected;
  }

  const TargetMemoryPolicy memory = getTargetMemoryPolicy();
  mlir::ModuleOp diagnosticAnchor = inputModules.front();
  if (memory.spmLimit < memory.spmBase)
    return diagnosticAnchor.emitOpError(
        "program_resource_verification: target SPM range is invalid");

  llvm::SmallVector<analysis::TileInstructionProgram, 16> programs;
  programs.reserve(inputModules.size());
  for (auto [module, tileId] : llvm::zip_equal(inputModules, tileIds)) {
    mlir::ModuleOp diagnosticModule = module;
    llvm::Expected<ExecutableCallClosure> closure =
        analyzeExecutableCallClosure(module);
    if (!closure)
      return diagnosticModule.emitOpError()
             << "program_resource_verification: call closure "
                "is invalid: "
             << llvm::toString(closure.takeError());
    programs.push_back({tileId, closure->entry.getOperation()});
  }
  analysis::InstructionProgramAggregateCost cost =
      analysis::analyzeInstructionProgramAggregateCost(programs, memory);

  const uint64_t perTileSPMCapacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  for (auto &&[tile, tileCost] : llvm::enumerate(cost.tileCosts)) {
    mlir::ModuleOp tileModule = inputModules[tile];
    const int64_t tileId = tileIds[tile].getValue();
    if (!tileCost.spmHighWaterBytes.isKnown())
      return tileModule.emitOpError()
             << "program_resource_verification: Tile " << tileId
             << " has no exact SPM high-water";
    if (tileCost.spmHighWaterBytes.value > perTileSPMCapacity)
      return tileModule.emitOpError()
             << "program_resource_verification: Tile " << tileId
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
           << "program_resource_verification: matched card NoC "
              "transmit and receive bytes differ ("
           << cost.aggregateNoC.aggregateTransmitBytes.value << " vs "
           << cost.aggregateNoC.aggregateReceiveBytes.value << ")";

  return cost;
}

} // namespace wafer::compiler::detail

namespace wafer::compiler::testing {

mlir::FailureOr<analysis::InstructionProgramAggregateCost>
verifyProgramResources(llvm::ArrayRef<mlir::ModuleOp> tileModules,
                       llvm::ArrayRef<TileId> tileIds,
                       const ExecutionConfig &executionConfig) {
  return detail::verifyProgramResources(tileModules, tileIds, executionConfig);
}

} // namespace wafer::compiler::testing
