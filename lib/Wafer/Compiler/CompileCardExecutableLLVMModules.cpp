//===- CompileCardExecutableLLVMModules.cpp - Compile Tile LLVM modules ----===//

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
compileCardExecutableToTargetLLVMModulesImpl(
    const CardExecutable &cardExecutable,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProfileCaptureKind profileCapture) {
  const std::vector<TileExecutable> &tiles =
      cardExecutable.getTileExecutables();
  const ExecutionConfig &executionConfig =
      cardExecutable.getExecutionConfig();
  if (tiles.size() !=
      static_cast<size_t>(executionConfig.getTileCount()))
    return fail(diagnostics, "target LLVM Tile domain is incomplete");
  if (profileCapture != ProfileCaptureKind::None &&
      executionConfig.getTileCount() != WAFER_TX81_PROFILER_TILE_COUNT)
    return fail(diagnostics,
                "profile target LLVM requires the complete 16-Tile domain");

  const RuntimeLaunchContract &runtimeLaunchContract =
      cardExecutable.getRuntimeLaunchContract();
  if (runtimeLaunchContract.getKind() != executionConfig.getRuntimeLaunchKind())
    return fail(diagnostics,
                "executable runtime launch contract does not match the "
                "execution configuration");
  const bool transportPreparedBeforeEntry = llvm::is_contained(
      runtimeLaunchContract.getPhases(), RuntimeLaunchPhaseRole::Prepare);
  std::set<int64_t> tileIds;
  std::set<int64_t> launchSlotIds;
  std::optional<CardId> cardId;
  std::vector<PreparedTile> preparedTiles;
  preparedTiles.reserve(tiles.size());
  for (const TileExecutable &tile : tiles) {
    if (!cardId)
      cardId = tile.getCardId();
    if (tile.getCardId() != *cardId ||
        tile.getCardId().getValue() < 0 ||
        tile.getTileId().getValue() < 0 ||
        tile.getLaunchSlotId().getValue() < 0 ||
        !tileIds.insert(tile.getTileId().getValue()).second ||
        !launchSlotIds.insert(tile.getLaunchSlotId().getValue()).second)
      return fail(
          diagnostics,
          "target LLVM domain has invalid or duplicate physical identity");
    mlir::FailureOr<PreparedTile> prepared = prepareTargetABI(
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
                  "target lowering failed for Tile launch_slot=" +
                      std::to_string(tile.getLaunchSlotId().getValue()));
    if (mlir::failed(verifyLoweredKernelABI(*prepared, tile.getEntrySymbol())))
      return fail(diagnostics,
                  "target ABI verification failed for Tile "
                  "launch_slot=" +
                      std::to_string(tile.getLaunchSlotId().getValue()));
    if (prepared->cardId != tile.getCardId() ||
        prepared->tileId != tile.getTileId() ||
        prepared->launchSlotId != tile.getLaunchSlotId() ||
        prepared->targetIdentity != executionConfig.getTargetIdentityId() ||
        prepared->transportPreparedBeforeEntry !=
            transportPreparedBeforeEntry ||
        prepared->kernelRuntimeABI != KernelRuntimeABIId::waferTx81Kernel() ||
        prepared->moduleFormat != kCurrentTargetModuleFormat)
      return fail(
          diagnostics,
          "prepared target identity readback failed for Tile " +
              std::to_string(tile.getTileId().getValue()));
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
    const TileExecutable &tile = tiles[tileIndex];
    llvm::Expected<TargetLLVMModule> translated = translatePreparedTile(
        std::move(prepared), tile.getEntrySymbol());
    if (!translated)
      return fail(diagnostics,
                  "target LLVM translation/readback failed for Tile " +
                      std::to_string(tile.getTileId().getValue()) +
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
      targetModule.getModule(), targetModule.getCardId(),
      targetModule.getTileId(), targetModule.getLaunchSlotId(),
      targetModule.getEntrySymbol(), targetModule.getTargetIdentityId(),
      targetModule.getKernelRuntimeABIId(), targetModule.getModuleFormat(),
      targetModule.getKernelABISlots());
}

} // namespace wafer::compiler::detail
