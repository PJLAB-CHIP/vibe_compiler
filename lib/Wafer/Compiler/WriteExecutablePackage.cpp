//===- WriteExecutablePackage.cpp - Write target modules and package ----===//

#include "CardExecutableInternal.h"
#include "CompilationInternal.h"
#include "CompilationStatistics.h"
#include "ExecutableCallClosure.h"
#include "PackageInternal.h"
#include "TargetCodeGenInternal.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Package.h"
#include "Wafer/Package/PackageManifest.h"
#include "Wafer/Runtime/ProfileInstrumentation.h"
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
    const CardExecutable &cardExecutable, llvm::StringRef stagedTargetModules,
    llvm::StringRef stagedPackage, const TargetToolchain &targetToolchain,
    llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<TargetLLVMModules> &retainedTargetLLVMModules,
    ProfileCaptureKind profileCapture = ProfileCaptureKind::None) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan targetPackageTiming(
      "stage", "executable-to-package", "target-package");
  const CompileClock::time_point targetIRStart = CompileClock::now();
  auto targetIRTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "executable-to-package", "target-ir-lowering");
  llvm::Expected<TargetLLVMModules> translatedModules =
      compileCardExecutableToTargetLLVMModulesImpl(cardExecutable, diagnostics,
                                                   failAfterTargetLaunchSlot,
                                                   profileCapture);
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
  llvm::Expected<VerifiedPackage> package =
      writePackage(tensorProgramDirectory, cardExecutable, *targetModules,
                   stagedPackage, diagnostics, failAfterPackageLaunchSlot);
  if (!package) {
    llvm::consumeError(package.takeError());
    return mlir::failure();
  }
  const int64_t packageWallMs = elapsedCompileMilliseconds(packageStart);
  packageTiming.reset();
  diagnostics << "wafer-compile: compile-stats stage=target-ir-lowering"
              << " wall_ms=" << targetIRWallMs
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << " tile_lowerings=" << translatedModules->getModules().size()
              << "\n";
  diagnostics << "wafer-compile: compile-stats stage=target-module"
              << " wall_ms=" << targetModuleWallMs
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << " module_count=" << targetModules->getModules().size() << "\n";
  diagnostics << "wafer-compile: compile-stats stage=package-assembly"
              << " wall_ms=" << packageWallMs
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << " tile_count="
              << cardExecutable.getExecutionConfig().getTileCount() << "\n";
  diagnostics << "wafer-compile: compile-stats stage=target-package"
              << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << "\n";
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
collectProfileStaticCostModel(const CardExecutable &cardExecutable) {
  const auto &tiles = cardExecutable.getTileExecutables();
  if (tiles.size() !=
      static_cast<size_t>(cardExecutable.getExecutionConfig().getTileCount()))
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

  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy();
  analysis::CardInstructionProgramCost cost =
      analysis::analyzeCardInstructionProgramCost(tileModules, policy);
  if (cost.tileCosts.size() != tiles.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost analysis omitted an accepted Tile");

  runtime::ProfileStaticCostModel model;
  model.model = runtime::kProfileStaticCostModelName.str();
  model.scope = runtime::kProfileStaticCostModelScope.str();
  model.rates.cardDDRBytesPerSecond = policy.cardDDRBytesPerSecond;
  model.rates.directionalNoCBytesPerSecond =
      policy.directionalNoCBytesPerSecond;
  model.rates.f16Bf16NPULogicalOpsPerSecondPerTile =
      policy.f16Bf16NPULogicalOpsPerSecondPerTile;
  model.rates.f16Bf16VectorLogicalOpsPerSecondPerTile =
      policy.f16Bf16VectorLogicalOpsPerSecondPerTile;
  model.rates.f32VectorLogicalOpsPerSecondPerTile =
      policy.f32VectorLogicalOpsPerSecondPerTile;
  model.rates.spmMovementBytesPerSecond = std::nullopt;
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
    llvm::StringRef instrumentationRoot, const CardExecutable &cardExecutable,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    ProfileCapturePackages &metadata,
    std::optional<TargetLLVMModules> &traceTargetLLVM) {
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
            tensorProgramDirectory, cardExecutable, targetModulesDirectory,
            package, targetToolchain, diagnostics, std::nullopt, std::nullopt,
            targetLLVM, capture)))
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

static mlir::LogicalResult
writeProfileInstrumentation(llvm::StringRef instrumentationRoot,
                            llvm::StringRef productionPackage,
                            const CardExecutable &productionExecutables,
                            const TargetLLVMModules &productionTargetLLVM,
                            const TargetLLVMModules &productionTraceTargetLLVM,
                            const ProfileCapturePackages &productionCaptures,
                            llvm::raw_ostream &diagnostics) {
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
  return writeJSONFile(
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
      diagnostics);
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

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<CardExecutable> &cardExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan productTiming(
      "stage", "target-codegen", "executable-package");
  llvm::Expected<CardExecutable> compiledCardExecutable =
      compileTensorProgramToCardExecutable(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLaunchSlot, programData, resolver, irTrace);
  if (!compiledCardExecutable) {
    llvm::consumeError(compiledCardExecutable.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> stagedTargetModules(transactionRoot);
  llvm::sys::path::append(stagedTargetModules, "target-modules");
  llvm::SmallString<256> stagedPackage(transactionRoot);
  llvm::sys::path::append(stagedPackage, "package");
  if (mlir::failed(stageExecutablePackage(
          tensorProgramDirectory, *compiledCardExecutable, stagedTargetModules,
          stagedPackage, targetToolchain, diagnostics,
          failAfterTargetLaunchSlot, failAfterPackageLaunchSlot,
          targetLLVMModules)))
    return mlir::failure();

  cardExecutable.emplace(std::move(*compiledCardExecutable));
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
    std::optional<CardExecutable> &cardExecutable,
    std::optional<TargetLLVMModules> &targetLLVMModules,
    ProgramDataHandoff &programData,
    const frontend::ProgramPayloadResolver &resolver,
    CompilationIRTrace &irTrace) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan productTiming(
      "stage", "target-codegen", "profile-package");
  llvm::Expected<CardExecutable> compiled =
      compileTensorProgramToCardExecutable(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLaunchSlot, programData, resolver, irTrace);
  if (!compiled) {
    llvm::consumeError(compiled.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> productionTargetModules(transactionRoot);
  llvm::sys::path::append(productionTargetModules, "target-modules");
  llvm::SmallString<256> productionPackage(transactionRoot);
  llvm::sys::path::append(productionPackage, "package");
  std::optional<TargetLLVMModules> productionTargetLLVM;
  if (mlir::failed(stageExecutablePackage(
          tensorProgramDirectory, *compiled, productionTargetModules,
          productionPackage, targetToolchain, diagnostics,
          failAfterTargetLaunchSlot, failAfterPackageLaunchSlot,
          productionTargetLLVM)))
    return mlir::failure();

  llvm::SmallString<256> instrumentationRoot(transactionRoot);
  llvm::sys::path::append(instrumentationRoot, "profile-instrumentation");

  ProfileCapturePackages productionCaptures;
  std::optional<TargetLLVMModules> productionTraceTargetLLVM;
  if (mlir::failed(stageCapturePackages(
          tensorProgramDirectory, transactionRoot, instrumentationRoot,
          *compiled, targetToolchain, diagnostics, productionCaptures,
          productionTraceTargetLLVM)))
    return mlir::failure();

  if (!productionTargetLLVM || !productionTraceTargetLLVM ||
      mlir::failed(writeProfileInstrumentation(
          instrumentationRoot, productionPackage, *compiled,
          *productionTargetLLVM, *productionTraceTargetLLVM, productionCaptures,
          diagnostics)))
    return mlir::failure();
  if (mlir::failed(makeProfileInstrumentationWorldAccessible(
          instrumentationRoot, diagnostics)))
    return mlir::failure();

  cardExecutable.emplace(std::move(*compiled));
  targetLLVMModules.emplace(std::move(*productionTargetLLVM));
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
