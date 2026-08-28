//===- CompileDeviceExecutableLLVMModules.cpp - Compile Tile LLVM modules
//----===//

#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "Wafer/Driver/BoundedTileExecutor.h"

#include "Wafer/Target/Core/TargetMemory.h"

#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

llvm::Expected<TargetLLVMModules>
compileDeviceExecutableToTargetLLVMModulesImpl(
    const DeviceExecutable &deviceExecutable, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    ProfileCaptureKind profileCapture,
    TargetLLVMCompilationStatistics *statistics) {
  if (statistics)
    *statistics = {};
  const std::vector<TileExecutable> &tiles =
      deviceExecutable.getTileExecutables();
  const ExecutionConfig &executionConfig =
      deviceExecutable.getExecutionConfig();
  if (tiles.size() != static_cast<size_t>(executionConfig.getTileCount()))
    return fail(diagnostics, "target LLVM Tile domain is incomplete");
  if (profileCapture != ProfileCaptureKind::None &&
      executionConfig.getTileCount() != WAFER_TX81_PROFILER_TILE_COUNT)
    return fail(diagnostics,
                "profile target LLVM requires the complete 16-Tile domain");

  const RuntimeLaunchContract &runtimeLaunchContract =
      deviceExecutable.getRuntimeLaunchContract();
  const bool transportPreparedBeforeEntry = llvm::is_contained(
      runtimeLaunchContract.getPhases(), RuntimeLaunchPhaseRole::Prepare);
  std::set<int64_t> tileIds;
  std::set<int64_t> launchSlotIds;
  std::optional<CardId> cardId;
  std::vector<size_t> tileIndexByLaunchSlot(tiles.size(),
                                            std::numeric_limits<size_t>::max());
  llvm::SmallVector<mlir::ModuleOp, 16> tileModules;
  tileModules.reserve(tiles.size());
  for (auto [tileIndex, tile] : llvm::enumerate(tiles)) {
    if (!cardId)
      cardId = tile.getCardId();
    if (tile.getCardId() != *cardId || tile.getCardId().getValue() < 0 ||
        tile.getTileId().getValue() < 0 ||
        tile.getLaunchSlotId().getValue() < 0 ||
        tile.getLaunchSlotId().getValue() >=
            static_cast<int64_t>(tiles.size()) ||
        !tileIds.insert(tile.getTileId().getValue()).second ||
        !launchSlotIds.insert(tile.getLaunchSlotId().getValue()).second)
      return fail(
          diagnostics,
          "target LLVM domain has invalid or duplicate physical identity");
    tileIndexByLaunchSlot[tile.getLaunchSlotId().getValue()] = tileIndex;
    tileModules.push_back(tile.getModule());
  }
  for (int64_t launchSlot = 0; launchSlot < static_cast<int64_t>(tiles.size());
       ++launchSlot)
    if (launchSlotIds.count(launchSlot) == 0)
      return fail(diagnostics,
                  "target LLVM launch-slot domain is not dense and canonical");

  enum class TileCompilationFailure : uint8_t {
    None,
    ABIPreparation,
    ProfileSlots,
    Lowering,
    ABIVerification,
    IdentityReadback,
    Translation,
  };
  struct TileCompilationResult {
    std::optional<TargetLLVMModule> module;
    TileCompilationFailure failure = TileCompilationFailure::None;
    std::string detail;
    bool abiPreparationAttempted = false;
    bool loweringAttempted = false;
    bool translationAttempted = false;
  };

  std::vector<TileCompilationResult> tileResults(tiles.size());
  unsigned workers = 1;
  {
    mlir::ParallelDiagnosticHandler parallelDiagnostics(
        tileModules.front().getContext());
    workers = runBoundedTileModulePipelines(tileModules, [&](size_t tileIndex) {
      const TileExecutable &tile = tiles[tileIndex];
      parallelDiagnostics.setOrderIDForThread(
          static_cast<size_t>(tile.getLaunchSlotId().getValue()));
      auto eraseDiagnosticOrder = llvm::make_scope_exit(
          [&] { parallelDiagnostics.eraseOrderIDForThread(); });
      TileCompilationResult &result = tileResults[tileIndex];
      result.abiPreparationAttempted = true;
      mlir::FailureOr<PreparedTile> prepared = prepareTargetABI(
          tile, executionConfig, transportPreparedBeforeEntry, profileCapture);
      if (mlir::failed(prepared)) {
        result.failure = TileCompilationFailure::ABIPreparation;
        return;
      }
      if (llvm::Error error = verifyProfileCaptureTileEntryArguments(
              prepared->slots, profileCapture)) {
        result.failure = TileCompilationFailure::ProfileSlots;
        result.detail = llvm::toString(std::move(error));
        return;
      }
      result.loweringAttempted = true;
      if (mlir::failed(lowerToTargetLLVM(*prepared))) {
        result.failure = TileCompilationFailure::Lowering;
        return;
      }
      if (mlir::failed(
              verifyLoweredKernelABI(*prepared, tile.getEntrySymbol()))) {
        result.failure = TileCompilationFailure::ABIVerification;
        return;
      }
      if (prepared->cardId != tile.getCardId() ||
          prepared->tileId != tile.getTileId() ||
          prepared->launchSlotId != tile.getLaunchSlotId() ||
          prepared->targetIdentity != executionConfig.getTargetIdentityId() ||
          prepared->transportPreparedBeforeEntry !=
              transportPreparedBeforeEntry ||
          prepared->kernelRuntimeABI != KernelRuntimeABIId::waferTx81Kernel() ||
          prepared->moduleFormat != kCurrentTargetModuleFormat) {
        result.failure = TileCompilationFailure::IdentityReadback;
        return;
      }
      result.translationAttempted = true;
      llvm::Expected<TargetLLVMModule> translated =
          translatePreparedTile(std::move(*prepared), tile.getEntrySymbol());
      if (!translated) {
        result.failure = TileCompilationFailure::Translation;
        result.detail = llvm::toString(translated.takeError());
        return;
      }
      result.module.emplace(std::move(*translated));
    });
  }

  if (statistics) {
    statistics->maximumTilePipelineWorkers =
        std::max<uint64_t>(statistics->maximumTilePipelineWorkers, workers);
    for (const TileCompilationResult &result : tileResults) {
      statistics->targetABIPreparationAttempts +=
          result.abiPreparationAttempted;
      statistics->targetLoweringAttempts += result.loweringAttempted;
      statistics->targetTranslationAttempts += result.translationAttempted;
    }
  }

  auto reportTileFailure =
      [&](const TileExecutable &tile,
          const TileCompilationResult &result) -> llvm::Error {
    const std::string launchSlot =
        std::to_string(tile.getLaunchSlotId().getValue());
    switch (result.failure) {
    case TileCompilationFailure::None:
      return llvm::Error::success();
    case TileCompilationFailure::ABIPreparation:
      return fail(diagnostics,
                  "target ABI preparation failed for launch_slot=" +
                      launchSlot);
    case TileCompilationFailure::ProfileSlots:
      return fail(diagnostics,
                  "target ABI profiler slot verification failed for physical "
                  "Tile launch_slot=" +
                      launchSlot + ": " + result.detail);
    case TileCompilationFailure::Lowering:
      return fail(diagnostics,
                  "target lowering failed for Tile launch_slot=" + launchSlot);
    case TileCompilationFailure::ABIVerification:
      return fail(diagnostics,
                  "target ABI verification failed for Tile launch_slot=" +
                      launchSlot);
    case TileCompilationFailure::IdentityReadback:
      return fail(diagnostics,
                  "prepared target identity readback failed for Tile " +
                      std::to_string(tile.getTileId().getValue()));
    case TileCompilationFailure::Translation:
      return fail(diagnostics,
                  "target LLVM translation/readback failed for Tile " +
                      std::to_string(tile.getTileId().getValue()) + ": " +
                      result.detail);
    }
    llvm_unreachable("unknown Tile target compilation failure");
  };

  std::vector<TargetLLVMModule> modules;
  modules.reserve(tiles.size());
  for (int64_t launchSlot = 0; launchSlot < static_cast<int64_t>(tiles.size());
       ++launchSlot) {
    const size_t tileIndex = tileIndexByLaunchSlot[launchSlot];
    const TileExecutable &tile = tiles[tileIndex];
    TileCompilationResult &result = tileResults[tileIndex];
    if (result.failure != TileCompilationFailure::None)
      return reportTileFailure(tile, result);
    if (!result.module)
      return fail(diagnostics,
                  "target Tile pipeline completed without an LLVM module for "
                  "launch_slot=" +
                      std::to_string(launchSlot));
    if (failAfterLaunchSlot &&
        tile.getLaunchSlotId().getValue() == *failAfterLaunchSlot)
      return fail(diagnostics,
                  "test-only injected target failure after launch_slot=" +
                      std::to_string(*failAfterLaunchSlot));
    modules.push_back(std::move(*result.module));
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
      targetModule.getTileEntryArguments());
}

} // namespace wafer::compiler::detail
