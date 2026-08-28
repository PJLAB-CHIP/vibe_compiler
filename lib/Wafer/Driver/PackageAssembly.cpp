//===- WriteExecutablePackage.cpp - Write target modules and package ----===//

#include "Wafer/Analysis/Module/ExecutableCallClosure.h"
#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CompilationStatistics.h"
#include "Wafer/Package/Writer/PackageInternal.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Analysis/Instr/ScheduleCostAnalysis.h"
#include "Wafer/Driver/CompilationResult.h"
#include "Wafer/Package/Manifest/PackageManifest.h"
#include "Wafer/Package/Profile/ProfileInstrumentationModel.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/TargetCall.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <array>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

static mlir::LogicalResult
writeJSONFile(llvm::StringRef path,
              llvm::function_ref<void(llvm::json::OStream &)> write,
              llvm::raw_ostream &diagnostics) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    reject(diagnostics, "failed to create profile instrumentation metadata: " +
                            error.message());
    return mlir::failure();
  }
  llvm::json::OStream json(output, /*IndentSize=*/2);
  write(json);
  output << "\n";
  output.close();
  if (output.has_error()) {
    reject(diagnostics, "failed to write profile instrumentation metadata");
    return mlir::failure();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> readback =
      llvm::MemoryBuffer::getFile(path);
  if (!readback) {
    reject(diagnostics,
           "failed to read back profile instrumentation metadata: " +
               readback.getError().message());
    return mlir::failure();
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*readback)->getBuffer());
  if (!parsed || !parsed->getAsObject()) {
    if (!parsed)
      llvm::consumeError(parsed.takeError());
    reject(
        diagnostics,
        "profile instrumentation metadata JSON readback verification failed");
    return mlir::failure();
  }
  return mlir::success();
}

static llvm::Expected<std::string> getFileDigest(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(), "failed to digest file");
  llvm::SHA256 hasher;
  hasher.update((*buffer)->getBuffer());
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static llvm::Expected<std::string>
getPackageManifestDigest(llvm::StringRef packageDirectory) {
  llvm::SmallString<256> manifest(packageDirectory);
  llvm::sys::path::append(manifest, runtime::kPackageManifestFileName);
  return getFileDigest(manifest);
}

static mlir::LogicalResult stageExecutablePackage(
    llvm::StringRef tensorProgramDirectory,
    const DeviceExecutable &deviceExecutable,
    llvm::StringRef stagedTargetModules, llvm::StringRef stagedPackage,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<TargetLLVMModules> &retainedTargetLLVMModules,
    CompilationStageTracker &stages,
    ProfileCaptureKind profileCapture = ProfileCaptureKind::None) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan targetPackageTiming(
      "stage", "executable-to-package", "target-package");
  stages.enter(CompilationStage::TargetCodeGeneration);
  const CompileClock::time_point targetIRStart = CompileClock::now();
  auto targetIRTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "executable-to-package", "target-ir-lowering");
  TargetLLVMCompilationStatistics targetCompilationStatistics;
  llvm::Expected<TargetLLVMModules> translatedModules =
      compileDeviceExecutableToTargetLLVMModulesImpl(
          deviceExecutable, diagnostics, failAfterTargetLaunchSlot,
          profileCapture, &targetCompilationStatistics);
  if (!translatedModules) {
    llvm::consumeError(translatedModules.takeError());
    return mlir::failure();
  }
  const int64_t targetIRWallMs = elapsedCompileMilliseconds(targetIRStart);
  targetIRTiming.reset();
  const CompileClock::time_point targetModuleStart = CompileClock::now();
  auto targetModuleTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "executable-to-package", "target-module");
  llvm::Expected<LinkedTargetModules> targetModules =
      linkTargetLLVMModulesImpl(*translatedModules, stagedTargetModules,
                                targetToolchain, diagnostics, profileCapture);
  if (!targetModules) {
    llvm::consumeError(targetModules.takeError());
    return mlir::failure();
  }
  const int64_t targetModuleWallMs =
      elapsedCompileMilliseconds(targetModuleStart);
  targetModuleTiming.reset();
  const CompileClock::time_point packageStart = CompileClock::now();
  auto packageTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "executable-to-package", "package-assembly");
  stages.enter(CompilationStage::PackageAssembly);
  llvm::Expected<runtime::VerifiedPackageManifest> stagedReadback =
      writePackage(tensorProgramDirectory, deviceExecutable, *targetModules,
                   stagedPackage, diagnostics, failAfterPackageLaunchSlot);
  if (!stagedReadback) {
    llvm::consumeError(stagedReadback.takeError());
    return mlir::failure();
  }
  const int64_t packageWallMs = elapsedCompileMilliseconds(packageStart);
  packageTiming.reset();
  if (wafer::support::getActiveCompileTimingSession()) {
    diagnostics << "wafer-compile: compile-stats stage=target-ir-lowering"
                << " wall_ms=" << targetIRWallMs
                << " peak_rss_kib=" << getCompilePeakRSSKiB()
                << " capture=" << stringifyProfileCaptureKind(profileCapture)
                << " abi_preparations="
                << targetCompilationStatistics.targetABIPreparationAttempts
                << " tile_lowerings="
                << targetCompilationStatistics.targetLoweringAttempts
                << " tile_translations="
                << targetCompilationStatistics.targetTranslationAttempts
                << " tile_pipeline_workers="
                << targetCompilationStatistics.maximumTilePipelineWorkers
                << "\n";
    diagnostics << "wafer-compile: compile-stats stage=target-module"
                << " wall_ms=" << targetModuleWallMs
                << " peak_rss_kib=" << getCompilePeakRSSKiB()
                << " capture=" << stringifyProfileCaptureKind(profileCapture)
                << " module_count=" << targetModules->getModules().size()
                << "\n";
    diagnostics << "wafer-compile: compile-stats stage=package-assembly"
                << " wall_ms=" << packageWallMs
                << " peak_rss_kib=" << getCompilePeakRSSKiB()
                << " capture=" << stringifyProfileCaptureKind(profileCapture)
                << " tile_count="
                << deviceExecutable.getExecutionConfig().getTileCount() << "\n";
    diagnostics << "wafer-compile: compile-stats stage=target-package"
                << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
                << " peak_rss_kib=" << getCompilePeakRSSKiB()
                << " capture=" << stringifyProfileCaptureKind(profileCapture)
                << "\n";
  }
  retainedTargetLLVMModules.emplace(std::move(*translatedModules));
  return mlir::success();
}

static runtime::ProfileStaticCostMetric
makeProfileStaticCostMetric(const analysis::ScheduleCostMetric &metric) {
  runtime::ProfileStaticCostMetric result;
  result.knowledge =
      analysis::stringifyScheduleCostKnowledge(metric.knowledge).str();
  result.reason = analysis::stringifyScheduleCostReason(metric.reason).str();
  if (metric.isKnown())
    result.value = metric.value;
  return result;
}

static llvm::Expected<runtime::ProfileStaticCostModel>
collectProfileStaticCostModel(const DeviceExecutable &deviceExecutable) {
  const auto &tiles = deviceExecutable.getTileExecutables();
  if (tiles.size() !=
      static_cast<size_t>(deviceExecutable.getExecutionConfig().getTileCount()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost Tile domain differs from execution "
        "config");

  std::set<int64_t> tileIds;
  std::vector<const TileExecutable *> tilesByLaunchSlot(tiles.size(), nullptr);
  for (const TileExecutable &tile : tiles) {
    const int64_t launchSlot = tile.getLaunchSlotId().getValue();
    if (tile.getCardId() != CardId(0) || tile.getTileId().getValue() < 0 ||
        launchSlot < 0 || launchSlot >= static_cast<int64_t>(tiles.size()) ||
        !tileIds.insert(tile.getTileId().getValue()).second ||
        tilesByLaunchSlot[launchSlot] != nullptr)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile static cost Tile binding is invalid or "
          "duplicated");
    tilesByLaunchSlot[launchSlot] = &tile;
  }
  if (llvm::is_contained(tilesByLaunchSlot, nullptr))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost launch-slot domain is not dense and complete");

  llvm::SmallVector<analysis::TileInstructionProgram, 16> tileModules;
  tileModules.reserve(tilesByLaunchSlot.size());
  for (const TileExecutable *tile : tilesByLaunchSlot) {
    llvm::Expected<ExecutableCallClosure> closure =
        analyzeExecutableCallClosure(tile->getModule(), tile->getEntrySymbol());
    if (!closure)
      return llvm::joinErrors(
          llvm::createStringError(
              llvm::errc::invalid_argument,
              "profile static cost accepted call closure is invalid"),
          closure.takeError());
    tileModules.push_back({tile->getTileId(), closure->entry.getOperation()});
  }

  const TargetMemoryPolicy memory = getTargetMemoryPolicy();
  analysis::InstructionProgramAggregateCost cost =
      analysis::analyzeInstructionProgramAggregateCost(tileModules, memory);
  if (cost.tileCosts.size() != tiles.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost analysis omitted an accepted Tile");

  runtime::ProfileStaticCostModel model;
  model.model = runtime::kProfileStaticCostModelName.str();
  model.scope = runtime::kProfileStaticCostModelScope.str();
  model.rates = runtime::getTargetProfileStaticCostRates();
  model.tiles.reserve(cost.tileCosts.size());
  for (auto [launchSlot, tileCost] : llvm::enumerate(cost.tileCosts)) {
    const TileExecutable &tile = *tilesByLaunchSlot[launchSlot];
    runtime::ProfileStaticTileWork work;
    work.npuF16Bf16LogicalOps =
        makeProfileStaticCostMetric(tileCost.compute.npuF16Bf16LogicalOps);
    work.npuOtherLogicalOps =
        makeProfileStaticCostMetric(tileCost.compute.npuOtherLogicalOps);
    work.vectorF16Bf16LogicalOps =
        makeProfileStaticCostMetric(tileCost.compute.vectorF16Bf16LogicalOps);
    work.vectorF32LogicalOps =
        makeProfileStaticCostMetric(tileCost.compute.vectorF32LogicalOps);
    work.vectorOtherLogicalOps =
        makeProfileStaticCostMetric(tileCost.compute.vectorOtherLogicalOps);
    work.ddrReadBytes = makeProfileStaticCostMetric(tileCost.ddrReadBytes);
    work.ddrWriteBytes = makeProfileStaticCostMetric(tileCost.ddrWriteBytes);
    work.spmMovementBytes =
        makeProfileStaticCostMetric(tileCost.spmMovementBytes);
    work.nocTransmitBytes =
        makeProfileStaticCostMetric(tileCost.noc.aggregateTransmitBytes);
    work.nocReceiveBytes =
        makeProfileStaticCostMetric(tileCost.noc.aggregateReceiveBytes);
    work.directionalNoCTransmitBytes.north = makeProfileStaticCostMetric(
        tileCost.noc.directional(analysis::NoCDirection::North));
    work.directionalNoCTransmitBytes.east = makeProfileStaticCostMetric(
        tileCost.noc.directional(analysis::NoCDirection::East));
    work.directionalNoCTransmitBytes.south = makeProfileStaticCostMetric(
        tileCost.noc.directional(analysis::NoCDirection::South));
    work.directionalNoCTransmitBytes.west = makeProfileStaticCostMetric(
        tileCost.noc.directional(analysis::NoCDirection::West));
    runtime::ProfileStaticTileCost profiledTile;
    profiledTile.cardId = tile.getCardId();
    profiledTile.tileId = tile.getTileId();
    profiledTile.launchSlot = runtime::LaunchSlotId(
        static_cast<uint64_t>(tile.getLaunchSlotId().getValue()));
    profiledTile.work = std::move(work);
    model.tiles.push_back(std::move(profiledTile));
  }
  return model;
}

struct ProfileTileTargetCallSites {
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlotId{0};
  std::vector<ProfileTargetCallSite> sites;
};

struct ProfileSiteMap {
  std::vector<ProfileTileTargetCallSites> tiles;
};

static llvm::Expected<ProfileSiteMap>
collectTargetCallSites(const TargetLLVMModules &targetLLVMModules) {
  ProfileSiteMap maps;
  std::set<int64_t> tileIds;
  std::vector<const TargetLLVMModule *> modulesByLaunchSlot(
      targetLLVMModules.getModules().size(), nullptr);
  for (const TargetLLVMModule &module : targetLLVMModules.getModules()) {
    const int64_t launchSlot = module.getLaunchSlotId().getValue();
    if (module.getCardId() != CardId(0) || module.getTileId().getValue() < 0 ||
        launchSlot < 0 ||
        launchSlot >=
            static_cast<int64_t>(targetLLVMModules.getModules().size()) ||
        !tileIds.insert(module.getTileId().getValue()).second ||
        modulesByLaunchSlot[launchSlot] != nullptr)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile site-map physical binding is invalid or duplicated");
    modulesByLaunchSlot[launchSlot] = &module;
  }
  if (llvm::is_contained(modulesByLaunchSlot, nullptr))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile site-map launch-slot domain is not dense and complete");

  maps.tiles.reserve(modulesByLaunchSlot.size());
  for (const TargetLLVMModule *module : modulesByLaunchSlot) {
    llvm::Expected<std::vector<ProfileTargetCallSite>> sites =
        collectProfileTargetCallSites(module->getModule(),
                                      module->getEntrySymbol());
    if (!sites)
      return sites.takeError();
    for (auto [expectedSite, site] : llvm::enumerate(*sites))
      if (site.siteId != expectedSite)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "profile site-map IDs are not Tile-local dense");
    maps.tiles.push_back({module->getCardId(), module->getTileId(),
                          module->getLaunchSlotId(), std::move(*sites)});
  }
  return maps;
}

static void writeTargetCallSites(llvm::json::OStream &json,
                                 const ProfileSiteMap &maps) {
  for (const ProfileTileTargetCallSites &tile : maps.tiles) {
    json.object([&] {
      json.attribute("card_id", tile.cardId.getValue());
      json.attribute("tile_id", tile.tileId.getValue());
      json.attribute("launch_slot", tile.launchSlotId.getValue());
      json.attributeArray("sites", [&] {
        for (const ProfileTargetCallSite &site : tile.sites)
          json.object([&] {
            json.attribute("site_id", site.siteId);
            json.attribute("target_call_ordinal", site.targetCallOrdinal);
            json.attribute("target_call_symbol", site.targetCallSymbol);
            json.attribute("site_kind", runtime::stringifyProfileTargetSiteKind(
                                            site.siteKind));
            if (site.engine)
              json.attribute("engine",
                             stringifyTargetCallTSMEngine(*site.engine));
            json.attribute("correlation_key", site.correlationKey);
            json.attribute("function_ordinal", site.functionOrdinal);
            json.attribute("block_ordinal", site.blockOrdinal);
            json.attribute("instruction_ordinal", site.instructionOrdinal);
          });
      });
    });
  }
}

struct ProfileCapturePackageMetadata {
  ProfileCaptureKind capture = ProfileCaptureKind::None;
  std::string packageReference;
  std::string manifestDigest;
};

struct ProfileCapturePackages {
  std::array<ProfileCapturePackageMetadata, 2> captures;
};

static constexpr std::array<ProfileCaptureKind, 2> kProfileCaptures = {
    ProfileCaptureKind::Count, ProfileCaptureKind::Trace};

static mlir::LogicalResult stageCapturePackages(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    llvm::StringRef instrumentationRoot,
    const DeviceExecutable &deviceExecutable,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    ProfileCapturePackages &metadata,
    std::optional<TargetLLVMModules> &traceTargetLLVM,
    CompilationStageTracker &stages) {
  for (auto [index, capture] : llvm::enumerate(kProfileCaptures)) {
    llvm::SmallString<256> package(instrumentationRoot);
    llvm::sys::path::append(package, "captures",
                            stringifyProfileCaptureKind(capture));
    llvm::SmallString<256> targetModulesDirectory(transactionRoot);
    llvm::sys::path::append(targetModulesDirectory,
                            "profile-capture-target-modules",
                            stringifyProfileCaptureKind(capture));
    std::optional<TargetLLVMModules> targetLLVM;
    if (mlir::failed(stageExecutablePackage(
            tensorProgramDirectory, deviceExecutable, targetModulesDirectory,
            package, targetToolchain, diagnostics, std::nullopt, std::nullopt,
            targetLLVM, stages, capture)))
      return mlir::failure();
    if (!targetLLVM) {
      reject(diagnostics, "profile capture target LLVM modules are missing");
      return mlir::failure();
    }
    llvm::Expected<std::string> digest = getPackageManifestDigest(package);
    if (!digest) {
      reject(diagnostics, llvm::toString(digest.takeError()));
      return mlir::failure();
    }
    llvm::SmallString<128> reference;
    llvm::sys::path::append(reference, "captures",
                            stringifyProfileCaptureKind(capture));
    metadata.captures[index] = {capture, reference.str().str(), *digest};
    if (capture == ProfileCaptureKind::Trace)
      traceTargetLLVM.emplace(std::move(*targetLLVM));
  }
  return mlir::success();
}

static mlir::LogicalResult writeProfileInstrumentation(
    llvm::StringRef instrumentationRoot, llvm::StringRef productionPackage,
    const DeviceExecutable &productionExecutables,
    const TargetLLVMModules &productionTargetLLVM,
    const TargetLLVMModules &productionTraceTargetLLVM,
    const ProfileCapturePackages &productionCaptures,
    ProfileInstrumentationIdentity &identity, llvm::raw_ostream &diagnostics) {
  if (createDirectory(instrumentationRoot, diagnostics))
    return mlir::failure();

  llvm::Expected<std::string> productionDigest =
      getPackageManifestDigest(productionPackage);
  if (!productionDigest) {
    reject(diagnostics, llvm::toString(productionDigest.takeError()));
    return mlir::failure();
  }

  if (productionTargetLLVM.getModules().size() !=
      productionTraceTargetLLVM.getModules().size()) {
    reject(diagnostics,
           "profile trace Tile domain differs from the primary program");
    return mlir::failure();
  }
  std::vector<const TargetLLVMModule *> traceByLaunchSlot(
      productionTraceTargetLLVM.getModules().size(), nullptr);
  for (const TargetLLVMModule &traceTile :
       productionTraceTargetLLVM.getModules()) {
    const int64_t launchSlot = traceTile.getLaunchSlotId().getValue();
    if (launchSlot < 0 ||
        launchSlot >= static_cast<int64_t>(traceByLaunchSlot.size()) ||
        traceByLaunchSlot[launchSlot] != nullptr) {
      reject(diagnostics,
             "profile trace launch-slot binding is invalid or duplicated");
      return mlir::failure();
    }
    traceByLaunchSlot[launchSlot] = &traceTile;
  }
  if (llvm::is_contained(traceByLaunchSlot, nullptr)) {
    reject(diagnostics,
           "profile trace launch-slot domain is not dense and complete");
    return mlir::failure();
  }
  for (const TargetLLVMModule &finalTile : productionTargetLLVM.getModules()) {
    const int64_t launchSlot = finalTile.getLaunchSlotId().getValue();
    if (launchSlot < 0 ||
        launchSlot >= static_cast<int64_t>(traceByLaunchSlot.size())) {
      reject(diagnostics,
             "primary program launch-slot binding is outside the trace "
             "domain");
      return mlir::failure();
    }
    const TargetLLVMModule &traceTile = *traceByLaunchSlot[launchSlot];
    if (finalTile.getCardId() != traceTile.getCardId() ||
        finalTile.getTileId() != traceTile.getTileId() ||
        finalTile.getLaunchSlotId() != traceTile.getLaunchSlotId() ||
        finalTile.getEntrySymbol() != traceTile.getEntrySymbol()) {
      reject(diagnostics,
             "profile trace Tile binding or entry differs from the "
             "primary program");
      return mlir::failure();
    }
    if (llvm::Error error = verifyProfileTargetCallSitesMatch(
            finalTile.getModule(), finalTile.getEntrySymbol(),
            traceTile.getModule(), traceTile.getEntrySymbol())) {
      reject(diagnostics, llvm::toString(std::move(error)));
      return mlir::failure();
    }
  }
  llvm::Expected<ProfileSiteMap> productionSites =
      collectTargetCallSites(productionTargetLLVM);
  if (!productionSites) {
    reject(diagnostics, llvm::toString(productionSites.takeError()));
    return mlir::failure();
  }
  llvm::Expected<runtime::ProfileStaticCostModel> productionStaticCost =
      collectProfileStaticCostModel(productionExecutables);
  if (!productionStaticCost) {
    reject(diagnostics, llvm::toString(productionStaticCost.takeError()));
    return mlir::failure();
  }

  llvm::SmallString<256> siteMapPath(instrumentationRoot);
  llvm::sys::path::append(siteMapPath, "site-map.json");
  if (mlir::failed(writeJSONFile(
          siteMapPath,
          [&](llvm::json::OStream &json) {
            json.object([&] {
              json.attribute("schema", "wafer-profile-target-call-site-map");
              json.attribute("card_count",
                             runtime::kProfileInstrumentationCardCount);
              json.attribute("tile_count",
                             runtime::kProfileInstrumentationTileCount);
              json.attribute("site_basis",
                             "verified-target-llvm-entry-reachable-physical-"
                             "tile-target-call-preorder");
              json.attribute("correlation_basis",
                             runtime::kProfileSiteCorrelationBasis);
              json.attribute(
                  "target_call_registry_size",
                  static_cast<int64_t>(getTargetCallDescriptors().size()));
              json.attributeArray("tiles", [&] {
                writeTargetCallSites(json, *productionSites);
              });
            });
          },
          diagnostics)))
    return mlir::failure();

  llvm::SmallString<256> planPath(instrumentationRoot);
  llvm::sys::path::append(planPath, "plan.json");
  if (mlir::failed(writeJSONFile(
          planPath,
          [&](llvm::json::OStream &json) {
            json.object([&] {
              json.attribute("schema", "wafer-profile-plan");
              json.attribute("card_count",
                             runtime::kProfileInstrumentationCardCount);
              json.attribute("tile_count",
                             runtime::kProfileInstrumentationTileCount);
              json.attribute("site_map", "site-map.json");
              json.attribute("site_key_contract",
                             runtime::kProfileSiteKeyContract);
              json.attributeBegin("static_cost_model");
              runtime::writeProfileStaticCostModel(json, *productionStaticCost);
              json.attributeEnd();
              json.attributeArray("capture_packages", [&] {
                for (const ProfileCapturePackageMetadata &capture :
                     productionCaptures.captures)
                  json.object([&] {
                    json.attribute("capture", stringifyProfileCaptureKind(
                                                  capture.capture));
                    json.attribute("package_ref", capture.packageReference);
                    json.attribute("manifest_sha256", capture.manifestDigest);
                    json.attribute("record_abi", runtime::kProfileRecordABI);
                    json.attribute("record_bytes", getProfileCaptureRecordBytes(
                                                       capture.capture));
                  });
              });
            });
          },
          diagnostics)))
    return mlir::failure();

  llvm::Expected<std::string> planDigest = getFileDigest(planPath);
  llvm::Expected<std::string> siteMapDigest = getFileDigest(siteMapPath);
  if (!planDigest || !siteMapDigest) {
    if (!planDigest)
      reject(diagnostics, llvm::toString(planDigest.takeError()));
    if (!siteMapDigest)
      reject(diagnostics, llvm::toString(siteMapDigest.takeError()));
    return mlir::failure();
  }

  // Activation is deliberately the final instrumentation write. Its presence
  // means all referenced metadata was closed, read back, and bound by byte
  // digest.
  llvm::SmallString<256> activationPath(instrumentationRoot);
  llvm::sys::path::append(activationPath, "activation.json");
  if (mlir::failed(writeJSONFile(
          activationPath,
          [&](llvm::json::OStream &json) {
            json.object([&] {
              json.attribute("schema", "wafer-profile-activation");
              json.attribute("primary_manifest_sha256", *productionDigest);
              json.attributeObject("metadata_sha256", [&] {
                json.attribute("plan.json", *planDigest);
                json.attribute("site-map.json", *siteMapDigest);
              });
            });
          },
          diagnostics)))
    return mlir::failure();
  identity.primaryManifestDigest = std::move(*productionDigest);
  identity.planDigest = std::move(*planDigest);
  identity.siteMapDigest = std::move(*siteMapDigest);
  for (auto [index, capture] : llvm::enumerate(productionCaptures.captures))
    identity.captureManifestDigests[index] = capture.manifestDigest;
  return mlir::success();
}

static mlir::LogicalResult
makeProfileInstrumentationWorldAccessible(llvm::StringRef instrumentationRoot,
                                          llvm::raw_ostream &diagnostics) {
  auto setPermissions = [&](llvm::StringRef path) -> mlir::LogicalResult {
    if (std::error_code error =
            llvm::sys::fs::setPermissions(path, llvm::sys::fs::all_all)) {
      reject(diagnostics,
             "failed to set profile instrumentation permissions for '" +
                 path.str() + "': " + error.message());
      return mlir::failure();
    }
    return mlir::success();
  };

  std::error_code walkError;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(instrumentationRoot, walkError, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(walkError)) {
    if (walkError) {
      reject(
          diagnostics,
          "failed to walk profile instrumentation while setting permissions: " +
              walkError.message());
      return mlir::failure();
    }
    if (iterator->type() != llvm::sys::fs::file_type::directory_file &&
        iterator->type() != llvm::sys::fs::file_type::regular_file) {
      reject(diagnostics, "profile instrumentation contains a non-regular "
                          "permission target: '" +
                              iterator->path() + "'");
      return mlir::failure();
    }
    if (mlir::failed(setPermissions(iterator->path())))
      return mlir::failure();
  }
  if (walkError) {
    reject(
        diagnostics,
        "failed to walk profile instrumentation while setting permissions: " +
            walkError.message());
    return mlir::failure();
  }
  return setPermissions(instrumentationRoot);
}

} // namespace

static std::string digestOwnedBuffer(const llvm::MemoryBuffer &buffer) {
  llvm::SHA256 hasher;
  hasher.update(buffer.getBuffer());
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

static llvm::Error verifyExactDirectory(
    llvm::StringRef root,
    llvm::ArrayRef<std::pair<llvm::StringRef, llvm::sys::fs::file_type>>
        expected) {
  std::set<std::string> seen;
  std::error_code walkError;
  for (llvm::sys::fs::directory_iterator iterator(root, walkError), end;
       iterator != end; iterator.increment(walkError)) {
    if (walkError)
      return llvm::createStringError(
          walkError, "failed to walk staged profile instrumentation");
    llvm::StringRef name = llvm::sys::path::filename(iterator->path());
    auto match = llvm::find_if(
        expected, [&](const auto &entry) { return entry.first == name; });
    if (match == expected.end() || iterator->type() != match->second ||
        !seen.insert(name.str()).second)
      return llvm::createStringError(
          llvm::errc::operation_not_permitted,
          "staged profile instrumentation topology is not all-and-only: " +
              name);
  }
  if (walkError)
    return llvm::createStringError(
        walkError, "failed to finish staged profile instrumentation walk");
  if (seen.size() != expected.size())
    return llvm::createStringError(
        llvm::errc::operation_not_permitted,
        "staged profile instrumentation topology is incomplete");
  return llvm::Error::success();
}

llvm::Expected<BoundProfileInstrumentation> bindProfileInstrumentation(
    const runtime::detail::BoundExecutablePackage &primaryPackage,
    llvm::StringRef instrumentationRoot,
    const ProfileInstrumentationIdentity &identity) {
  if (digestOwnedBuffer(*primaryPackage.manifestBuffer) !=
      identity.primaryManifestDigest)
    return llvm::createStringError(
        llvm::errc::operation_not_permitted,
        "owned primary manifest does not match the profile identity");

  if (llvm::Error error = verifyExactDirectory(
          instrumentationRoot,
          {{"activation.json", llvm::sys::fs::file_type::regular_file},
           {"plan.json", llvm::sys::fs::file_type::regular_file},
           {"site-map.json", llvm::sys::fs::file_type::regular_file},
           {"captures", llvm::sys::fs::file_type::directory_file}}))
    return std::move(error);

  llvm::SmallString<256> capturesRoot(instrumentationRoot);
  llvm::sys::path::append(capturesRoot, "captures");
  if (llvm::Error error = verifyExactDirectory(
          capturesRoot, {{"count", llvm::sys::fs::file_type::directory_file},
                         {"trace", llvm::sys::fs::file_type::directory_file}}))
    return std::move(error);

  llvm::SmallString<256> activationPath(instrumentationRoot);
  llvm::sys::path::append(activationPath, "activation.json");
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> activation =
      runtime::detail::openPackageMember(activationPath,
                                         "profile instrumentation activation");
  if (!activation)
    return activation.takeError();
  llvm::SmallString<256> planPath(instrumentationRoot);
  llvm::sys::path::append(planPath, "plan.json");
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> plan =
      runtime::detail::openPackageMember(planPath,
                                         "profile instrumentation plan");
  if (!plan)
    return plan.takeError();
  llvm::SmallString<256> siteMapPath(instrumentationRoot);
  llvm::sys::path::append(siteMapPath, "site-map.json");
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> siteMap =
      runtime::detail::openPackageMember(siteMapPath,
                                         "profile instrumentation site map");
  if (!siteMap)
    return siteMap.takeError();
  if (digestOwnedBuffer(**plan) != identity.planDigest ||
      digestOwnedBuffer(**siteMap) != identity.siteMapDigest)
    return llvm::createStringError(
        llvm::errc::operation_not_permitted,
        "owned profile metadata digest does not match the writer identity");

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*activation)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(llvm::errc::operation_not_permitted,
                                   "staged profile instrumentation "
                                   "activation is not a JSON object");
  if (root->size() != 3)
    return llvm::createStringError(
        llvm::errc::operation_not_permitted,
        "staged profile instrumentation activation fields are not exact");
  auto requireString = [&](llvm::json::Object &object, llvm::StringRef key,
                           llvm::StringRef expected) -> llvm::Error {
    std::optional<llvm::StringRef> value = object.getString(key);
    if (!value || *value != expected)
      return llvm::createStringError(
          llvm::errc::operation_not_permitted,
          "staged profile instrumentation activation field '" + key +
              "' does not match the staged value");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          requireString(*root, "schema", "wafer-profile-activation"))
    return error;
  if (llvm::Error error = requireString(*root, "primary_manifest_sha256",
                                        identity.primaryManifestDigest))
    return error;
  llvm::json::Object *metadata = root->getObject("metadata_sha256");
  if (!metadata)
    return llvm::createStringError(llvm::errc::operation_not_permitted,
                                   "staged profile instrumentation "
                                   "activation is missing metadata_sha256");
  if (metadata->size() != 2)
    return llvm::createStringError(
        llvm::errc::operation_not_permitted,
        "staged profile instrumentation metadata fields are not exact");
  if (llvm::Error error =
          requireString(*metadata, "plan.json", identity.planDigest))
    return error;
  if (llvm::Error error =
          requireString(*metadata, "site-map.json", identity.siteMapDigest))
    return error;

  std::vector<runtime::detail::BoundExecutablePackage> captures;
  captures.reserve(kProfileCaptures.size());
  for (auto [index, capture] : llvm::enumerate(kProfileCaptures)) {
    llvm::SmallString<256> captureRoot(capturesRoot);
    llvm::sys::path::append(captureRoot, stringifyProfileCaptureKind(capture));
    llvm::Expected<runtime::detail::BoundExecutablePackage> package =
        runtime::detail::bindExecutablePackage(captureRoot);
    if (!package)
      return package.takeError();
    if (digestOwnedBuffer(*package->manifestBuffer) !=
        identity.captureManifestDigests[index])
      return llvm::createStringError(
          llvm::errc::operation_not_permitted,
          "owned profile capture manifest does not match the writer identity");
    captures.push_back(std::move(*package));
  }

  return BoundProfileInstrumentation{std::move(*activation), std::move(*plan),
                                     std::move(*siteMap), std::move(captures)};
}

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<DeviceExecutable> &deviceExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace, CompilationStageTracker &stages) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan productTiming(
      "stage", "target-codegen", "executable-package");
  stages.enter(CompilationStage::ExecutableCompilation);
  llvm::Expected<DeviceExecutable> compiledDeviceExecutable =
      compileTensorProgramToDeviceExecutable(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLaunchSlot, programData, resolver, irTrace);
  if (!compiledDeviceExecutable) {
    llvm::consumeError(compiledDeviceExecutable.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> stagedTargetModules(transactionRoot);
  llvm::sys::path::append(stagedTargetModules, "target-modules");
  llvm::SmallString<256> stagedPackage(transactionRoot);
  llvm::sys::path::append(stagedPackage, "package");
  if (mlir::failed(stageExecutablePackage(
          tensorProgramDirectory, *compiledDeviceExecutable,
          stagedTargetModules, stagedPackage, targetToolchain, diagnostics,
          failAfterTargetLaunchSlot, failAfterPackageLaunchSlot,
          targetLLVMModules, stages)))
    return mlir::failure();

  deviceExecutable.emplace(std::move(*compiledDeviceExecutable));
  if (wafer::support::getActiveCompileTimingSession())
    diagnostics << "wafer-compile: compile-stats stage=executable-package"
                << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
                << " peak_rss_kib=" << getCompilePeakRSSKiB()
                << " capture_packages=0 target_module_set_count=1\n";
  return mlir::success();
}

mlir::LogicalResult stageProfileTargetPackages(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<DeviceExecutable> &deviceExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace, CompilationStageTracker &stages,
    ProfileInstrumentationIdentity &profileIdentity) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan productTiming(
      "stage", "target-codegen", "profile-package");
  stages.enter(CompilationStage::ExecutableCompilation);
  llvm::Expected<DeviceExecutable> compiled =
      compileTensorProgramToDeviceExecutable(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLaunchSlot, programData, resolver, irTrace);
  if (!compiled) {
    llvm::consumeError(compiled.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> deliveryRoot(transactionRoot);
  llvm::sys::path::append(deliveryRoot, "delivery");
  llvm::SmallString<256> productionTargetModules(transactionRoot);
  llvm::sys::path::append(productionTargetModules, "target-modules");
  llvm::SmallString<256> productionPackage(deliveryRoot);
  llvm::sys::path::append(productionPackage, "package");
  std::optional<TargetLLVMModules> productionTargetLLVM;
  if (mlir::failed(stageExecutablePackage(
          tensorProgramDirectory, *compiled, productionTargetModules,
          productionPackage, targetToolchain, diagnostics,
          failAfterTargetLaunchSlot, failAfterPackageLaunchSlot,
          productionTargetLLVM, stages)))
    return mlir::failure();

  llvm::SmallString<256> instrumentationRoot(deliveryRoot);
  llvm::sys::path::append(instrumentationRoot, "package.profile");

  ProfileCapturePackages productionCaptures;
  std::optional<TargetLLVMModules> productionTraceTargetLLVM;
  if (mlir::failed(stageCapturePackages(
          tensorProgramDirectory, transactionRoot, instrumentationRoot,
          *compiled, targetToolchain, diagnostics, productionCaptures,
          productionTraceTargetLLVM, stages)))
    return mlir::failure();

  if (!productionTargetLLVM || !productionTraceTargetLLVM ||
      mlir::failed(writeProfileInstrumentation(
          instrumentationRoot, productionPackage, *compiled,
          *productionTargetLLVM, *productionTraceTargetLLVM, productionCaptures,
          profileIdentity, diagnostics)))
    return mlir::failure();
  if (mlir::failed(makeProfileInstrumentationWorldAccessible(
          instrumentationRoot, diagnostics)))
    return mlir::failure();

  deviceExecutable.emplace(std::move(*compiled));
  targetLLVMModules.emplace(std::move(*productionTargetLLVM));
  if (wafer::support::getActiveCompileTimingSession())
    diagnostics << "wafer-compile: compile-stats stage=profile-package"
                << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
                << " peak_rss_kib=" << getCompilePeakRSSKiB()
                << " capture_packages=" << kProfileCaptures.size()
                << " target_module_set_count=" << 1 + kProfileCaptures.size()
                << " target_tile_lowerings="
                << executionConfig.getTileCount() *
                       static_cast<int64_t>(1 + kProfileCaptures.size())
                << "\n";
  return mlir::success();
}

} // namespace wafer::compiler::detail
