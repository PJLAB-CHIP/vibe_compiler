//===- CompilePhysicalTileLLVMModules.cpp - Compile Tile LLVM modules ----===//

#include "TargetCodeGenInternal.h"

#include "Wafer/Support/TargetPolicy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

llvm::Expected<TargetLLVMModules>
compilePhysicalTileExecutablesToTargetLLVMModulesImpl(
    const PhysicalTileExecutables &physicalTileExecutables,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProfileCaptureKind profileCapture) {
  const std::vector<PhysicalTileExecutable> &tiles =
      physicalTileExecutables.getPhysicalTileExecutables();
  const ExecutionConfig &executionConfig =
      physicalTileExecutables.getExecutionConfig();
  if (tiles.size() !=
      static_cast<size_t>(executionConfig.getPhysicalTileCount()))
    return fail(diagnostics, "target LLVM physical Tile domain is incomplete");
  if (profileCapture != ProfileCaptureKind::None &&
      executionConfig.getPhysicalTileCount() != WAFER_TX81_PROFILER_TILE_COUNT)
    return fail(diagnostics,
                "profile target LLVM requires the complete 16-Tile domain");

  const RuntimeLaunchContract &runtimeLaunchContract =
      physicalTileExecutables.getRuntimeLaunchContract();
  if (runtimeLaunchContract.getKind() != executionConfig.getRuntimeLaunchKind())
    return fail(diagnostics,
                "executable runtime launch contract does not match the "
                "execution configuration");
  const bool transportPreparedBeforeEntry = llvm::is_contained(
      runtimeLaunchContract.getPhases(), RuntimeLaunchPhaseRole::Prepare);
  std::set<int64_t> physicalTileIds;
  std::set<int64_t> launchSlotIds;
  std::optional<PhysicalCardId> physicalCardId;
  std::vector<PreparedPhysicalTile> preparedTiles;
  preparedTiles.reserve(tiles.size());
  for (const PhysicalTileExecutable &tile : tiles) {
    if (!physicalCardId)
      physicalCardId = tile.getPhysicalCardId();
    if (tile.getPhysicalCardId() != *physicalCardId ||
        tile.getPhysicalCardId().getValue() < 0 ||
        tile.getPhysicalTileId().getValue() < 0 ||
        tile.getLaunchSlotId().getValue() < 0 ||
        !physicalTileIds.insert(tile.getPhysicalTileId().getValue()).second ||
        !launchSlotIds.insert(tile.getLaunchSlotId().getValue()).second)
      return fail(
          diagnostics,
          "target LLVM domain has invalid or duplicate physical identity");
    mlir::FailureOr<PreparedPhysicalTile> prepared = prepareTargetABI(
        tile, executionConfig, transportPreparedBeforeEntry, profileCapture);
    if (mlir::failed(prepared))
      return fail(diagnostics,
                  "target ABI preparation failed for launch_slot=" +
                      std::to_string(tile.getLaunchSlotId().getValue()));
    if (llvm::Error error =
            verifyProfileCaptureKernelABISlots(prepared->slots, profileCapture))
      return fail(diagnostics,
                  "target ABI profiler slot verification failed for physical "
                  "Tile launch_slot=" +
                      std::to_string(tile.getLaunchSlotId().getValue()) + ": " +
                      llvm::toString(std::move(error)));
    if (mlir::failed(lowerToTargetLLVM(*prepared)))
      return fail(diagnostics,
                  "target lowering failed for physical Tile launch_slot=" +
                      std::to_string(tile.getLaunchSlotId().getValue()));
    if (mlir::failed(verifyLoweredKernelABI(*prepared, tile.getEntrySymbol())))
      return fail(diagnostics,
                  "target ABI verification failed for physical Tile "
                  "launch_slot=" +
                      std::to_string(tile.getLaunchSlotId().getValue()));
    if (prepared->physicalCardId != tile.getPhysicalCardId() ||
        prepared->physicalTileId != tile.getPhysicalTileId() ||
        prepared->launchSlotId != tile.getLaunchSlotId() ||
        prepared->targetIdentity != executionConfig.getTargetIdentityId() ||
        prepared->transportPreparedBeforeEntry !=
            transportPreparedBeforeEntry ||
        prepared->kernelRuntimeABI != KernelRuntimeABIId::waferTx81Kernel() ||
        prepared->moduleFormat != kCurrentTargetModuleFormat)
      return fail(
          diagnostics,
          "prepared target identity readback failed for physical Tile " +
              std::to_string(tile.getPhysicalTileId().getValue()));
    preparedTiles.push_back(std::move(*prepared));
  }
  for (int64_t launchSlot = 0; launchSlot < static_cast<int64_t>(tiles.size());
       ++launchSlot)
    if (launchSlotIds.count(launchSlot) == 0)
      return fail(diagnostics,
                  "target LLVM launch-slot domain is not dense and canonical");

  std::vector<TargetLLVMModule> modules;
  modules.reserve(tiles.size());
  for (auto [tileIndex, prepared] : llvm::enumerate(preparedTiles)) {
    const PhysicalTileExecutable &tile = tiles[tileIndex];
    llvm::Expected<TargetLLVMModule> translated = translatePreparedPhysicalTile(
        std::move(prepared), tile.getEntrySymbol());
    if (!translated)
      return fail(diagnostics,
                  "target LLVM translation/readback failed for physical Tile " +
                      std::to_string(tile.getPhysicalTileId().getValue()) +
                      ": " + llvm::toString(translated.takeError()));
    modules.push_back(std::move(*translated));
    if (failAfterLaunchSlot &&
        tile.getLaunchSlotId().getValue() == *failAfterLaunchSlot)
      return fail(diagnostics,
                  "test-only injected target failure after launch_slot=" +
                      std::to_string(*failAfterLaunchSlot));
  }

  return TargetLLVMModulesBuilder::makeModules(
      executionConfig, runtimeLaunchContract, std::move(modules));
}

llvm::Error
verifyTargetLLVMModuleForTesting(const TargetLLVMModule &targetModule) {
  return verifyTargetLLVMModule(
      targetModule.getModule(), targetModule.getPhysicalCardId(),
      targetModule.getPhysicalTileId(), targetModule.getLaunchSlotId(),
      targetModule.getEntrySymbol(), targetModule.getTargetIdentityId(),
      targetModule.getKernelRuntimeABIId(), targetModule.getModuleFormat(),
      targetModule.getKernelABISlots());
}

} // namespace wafer::compiler::detail
