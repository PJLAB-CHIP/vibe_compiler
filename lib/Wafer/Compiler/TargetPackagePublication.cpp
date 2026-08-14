//===- TargetPackagePublication.cpp - Staged target/package build -------===//

#include "AcceptedCallClosure.h"
#include "CompilationInternal.h"
#include "CompilationStatistics.h"
#include "ExecutableBundleInternal.h"
#include "PackageInternal.h"
#include "TargetArtifactInternal.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Package.h"
#include "Wafer/Runtime/PackageManifest.h"
#include "Wafer/Runtime/ProfileCompanion.h"
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
    reject(diagnostics,
           "failed to create profile companion metadata: " + error.message());
    return mlir::failure();
  }
  llvm::json::OStream json(output, /*IndentSize=*/2);
  write(json);
  output << "\n";
  output.close();
  if (output.has_error()) {
    reject(diagnostics, "failed to write profile companion metadata");
    return mlir::failure();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> readback =
      llvm::MemoryBuffer::getFile(path);
  if (!readback) {
    reject(diagnostics, "failed to read back profile companion metadata: " +
                            readback.getError().message());
    return mlir::failure();
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*readback)->getBuffer());
  if (!parsed || !parsed->getAsObject()) {
    if (!parsed)
      llvm::consumeError(parsed.takeError());
    reject(diagnostics,
           "profile companion metadata JSON readback verification failed");
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
    const ExecutableBundle &executableBundle,
    llvm::StringRef stagedTargetArtifacts, llvm::StringRef stagedPackage,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle,
    ProfileCaptureKind profileCapture = ProfileCaptureKind::None) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan targetPackageTiming(
      "stage", "executable-to-package", "target-package");
  const CompileClock::time_point targetIRStart = CompileClock::now();
  auto targetIRTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "executable-to-package", "target-ir-lowering");
  llvm::Expected<TargetLLVMModuleBundle> targetLLVMModules =
      compileExecutableBundleToTargetLLVMModulesImpl(
          executableBundle, diagnostics, failAfterTargetLaunchSlot,
          profileCapture);
  if (!targetLLVMModules) {
    llvm::consumeError(targetLLVMModules.takeError());
    return mlir::failure();
  }
  const int64_t targetIRWallMs = elapsedCompileMilliseconds(targetIRStart);
  targetIRTiming.reset();
  const CompileClock::time_point targetArtifactStart = CompileClock::now();
  auto targetArtifactTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "executable-to-package", "target-artifact");
  llvm::Expected<TargetArtifactBundle> targetArtifacts =
      compileTargetLLVMModuleBundleToTargetArtifactsImpl(
          *targetLLVMModules, stagedTargetArtifacts, targetToolchain,
          diagnostics, profileCapture);
  if (!targetArtifacts) {
    llvm::consumeError(targetArtifacts.takeError());
    return mlir::failure();
  }
  const int64_t targetArtifactWallMs =
      elapsedCompileMilliseconds(targetArtifactStart);
  targetArtifactTiming.reset();
  const CompileClock::time_point packageStart = CompileClock::now();
  auto packageTiming =
      std::make_unique<wafer::support::ScopedCompileTimingSpan>(
          "stage", "executable-to-package", "package-assembly");
  llvm::Expected<PackageBundle> package = assemblePackageBundleImpl(
      tensorProgramDirectory, executableBundle, *targetArtifacts, stagedPackage,
      diagnostics, failAfterPackageLaunchSlot);
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
              << " tile_lowerings=" << targetLLVMModules->getModules().size()
              << "\n";
  diagnostics << "wafer-compile: compile-stats stage=target-artifact"
              << " wall_ms=" << targetArtifactWallMs
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << " module_count=" << targetArtifacts->getModules().size()
              << "\n";
  diagnostics << "wafer-compile: compile-stats stage=package-assembly"
              << " wall_ms=" << packageWallMs
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << " tile_count="
              << executableBundle.getExecutionConfig().getPhysicalTileCount()
              << "\n";
  diagnostics << "wafer-compile: compile-stats stage=target-package"
              << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << "\n";
  targetLLVMModuleBundle.emplace(std::move(*targetLLVMModules));
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
collectProfileStaticCostModel(const ExecutableBundle &bundle) {
  const auto &physicalTileExecutables = bundle.getPhysicalTileExecutables();
  if (physicalTileExecutables.size() !=
      static_cast<size_t>(bundle.getExecutionConfig().getPhysicalTileCount()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost physical Tile domain differs from execution "
        "config");

  std::set<int64_t> tileIds;
  std::vector<const PhysicalTileExecutable *> tilesByLaunchSlot(
      physicalTileExecutables.size(), nullptr);
  for (const PhysicalTileExecutable &tile : physicalTileExecutables) {
    const int64_t launchSlot = tile.getLaunchSlotId().getValue();
    if (tile.getPhysicalCardId() != PhysicalCardId(0) ||
        tile.getPhysicalTileId().getValue() < 0 || launchSlot < 0 ||
        launchSlot >= static_cast<int64_t>(physicalTileExecutables.size()) ||
        !tileIds.insert(tile.getPhysicalTileId().getValue()).second ||
        tilesByLaunchSlot[launchSlot] != nullptr)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile static cost physical Tile identity is invalid or "
          "duplicated");
    tilesByLaunchSlot[launchSlot] = &tile;
  }
  if (llvm::is_contained(tilesByLaunchSlot, nullptr))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost launch-slot domain is not dense and complete");

  llvm::SmallVector<analysis::PhysicalTileInstructionProgram, 16> tilePrograms;
  tilePrograms.reserve(tilesByLaunchSlot.size());
  for (const PhysicalTileExecutable *tile : tilesByLaunchSlot) {
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(tile->getModule(), tile->getEntrySymbol());
    if (!closure)
      return llvm::joinErrors(
          llvm::createStringError(
              llvm::errc::invalid_argument,
              "profile static cost accepted call closure is invalid"),
          closure.takeError());
    tilePrograms.push_back(
        {tile->getPhysicalTileId(), closure->entry.getOperation()});
  }

  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy();
  analysis::WholeCardInstructionProgramCost cost =
      analysis::analyzeWholeCardInstructionProgramCost(tilePrograms, policy);
  if (cost.tileCosts.size() != physicalTileExecutables.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost analysis omitted an accepted physical Tile");

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
    const PhysicalTileExecutable &tile = *tilesByLaunchSlot[launchSlot];
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
    profiledTile.cardId = tile.getPhysicalCardId();
    profiledTile.tileId = tile.getPhysicalTileId();
    profiledTile.launchSlot = runtime::LaunchSlotId(
        static_cast<uint64_t>(tile.getLaunchSlotId().getValue()));
    profiledTile.work = std::move(work);
    model.tiles.push_back(std::move(profiledTile));
  }
  return model;
}

struct ProfileTileTargetCallSites {
  PhysicalCardId cardId{0};
  PhysicalTileId tileId{0};
  LaunchSlotId launchSlotId{0};
  std::vector<ProfileTargetCallSite> sites;
};

struct ProfileSiteMap {
  std::vector<ProfileTileTargetCallSites> tiles;
};

static llvm::Expected<ProfileSiteMap>
collectTargetCallSites(const TargetLLVMModuleBundle &bundle) {
  ProfileSiteMap maps;
  std::set<int64_t> tileIds;
  std::vector<const TargetLLVMModule *> modulesByLaunchSlot(
      bundle.getModules().size(), nullptr);
  for (const TargetLLVMModule &module : bundle.getModules()) {
    const int64_t launchSlot = module.getLaunchSlotId().getValue();
    if (module.getPhysicalCardId() != PhysicalCardId(0) ||
        module.getPhysicalTileId().getValue() < 0 || launchSlot < 0 ||
        launchSlot >= static_cast<int64_t>(bundle.getModules().size()) ||
        !tileIds.insert(module.getPhysicalTileId().getValue()).second ||
        modulesByLaunchSlot[launchSlot] != nullptr)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile site-map physical identity is invalid or duplicated");
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
            "profile site-map IDs are not physical-Tile-local dense");
    maps.tiles.push_back({module->getPhysicalCardId(),
                          module->getPhysicalTileId(),
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
    llvm::StringRef companionRoot, const ExecutableBundle &executableBundle,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    ProfileCapturePackages &metadata,
    std::optional<TargetLLVMModuleBundle> &traceTargetLLVM) {
  for (auto [index, capture] : llvm::enumerate(kProfileCaptures)) {
    llvm::SmallString<256> package(companionRoot);
    llvm::sys::path::append(package, "captures",
                            stringifyProfileCaptureKind(capture));
    llvm::SmallString<256> artifacts(transactionRoot);
    llvm::sys::path::append(artifacts, "profile-capture-target-artifacts",
                            stringifyProfileCaptureKind(capture));
    std::optional<TargetLLVMModuleBundle> targetLLVM;
    if (mlir::failed(stageExecutablePackage(
            tensorProgramDirectory, executableBundle, artifacts, package,
            targetToolchain, diagnostics, std::nullopt, std::nullopt,
            targetLLVM, capture)))
      return mlir::failure();
    if (!targetLLVM) {
      reject(diagnostics, "profile capture target LLVM bundle is missing");
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
writeProfileCompanion(llvm::StringRef companionRoot,
                      llvm::StringRef productionPackage,
                      const ExecutableBundle &productionBundle,
                      const TargetLLVMModuleBundle &productionTargetLLVM,
                      const TargetLLVMModuleBundle &productionTraceTargetLLVM,
                      const ProfileCapturePackages &productionCaptures,
                      llvm::raw_ostream &diagnostics) {
  if (createDirectory(companionRoot, diagnostics))
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
           "profile trace physical Tile domain differs from final production");
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
             "profile trace launch-slot identity is invalid or duplicated");
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
             "final production launch-slot identity is outside the trace "
             "domain");
      return mlir::failure();
    }
    const TargetLLVMModule &traceTile = *traceByLaunchSlot[launchSlot];
    if (finalTile.getPhysicalCardId() != traceTile.getPhysicalCardId() ||
        finalTile.getPhysicalTileId() != traceTile.getPhysicalTileId() ||
        finalTile.getLaunchSlotId() != traceTile.getLaunchSlotId() ||
        finalTile.getEntrySymbol() != traceTile.getEntrySymbol()) {
      reject(diagnostics,
             "profile trace physical Tile or entry identity differs from "
             "final production");
      return mlir::failure();
    }
    if (llvm::Error error = verifyProfileTargetCallSiteIdentity(
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
      collectProfileStaticCostModel(productionBundle);
  if (!productionStaticCost) {
    reject(diagnostics, llvm::toString(productionStaticCost.takeError()));
    return mlir::failure();
  }

  llvm::SmallString<256> siteMapPath(companionRoot);
  llvm::sys::path::append(siteMapPath, "site-map.json");
  if (mlir::failed(writeJSONFile(
          siteMapPath,
          [&](llvm::json::OStream &json) {
            json.object([&] {
              json.attribute("schema", "wafer-profile-target-call-site-map");
              json.attribute("schema_version",
                             int64_t(runtime::kProfileCompanionSchemaVersion));
              json.attribute("card_count", runtime::kProfileCompanionCardCount);
              json.attribute("tile_count", runtime::kProfileCompanionTileCount);
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

  llvm::SmallString<256> planPath(companionRoot);
  llvm::sys::path::append(planPath, "plan.json");
  if (mlir::failed(writeJSONFile(
          planPath,
          [&](llvm::json::OStream &json) {
            json.object([&] {
              json.attribute("schema", "wafer-profile-plan");
              json.attribute("schema_version",
                             int64_t(runtime::kProfileCompanionSchemaVersion));
              json.attribute("card_count", runtime::kProfileCompanionCardCount);
              json.attribute("tile_count", runtime::kProfileCompanionTileCount);
              json.attribute("site_map", "site-map.json");
              json.attribute("site_identity",
                             "final-physical-tile-local-typed-target-site-id-"
                             "and-correlation-key");
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

  // Activation is deliberately the final companion write. Its presence means
  // all referenced metadata was closed, read back, and bound by byte digest.
  llvm::SmallString<256> activationPath(companionRoot);
  llvm::sys::path::append(activationPath, "activation.json");
  return writeJSONFile(
      activationPath,
      [&](llvm::json::OStream &json) {
        json.object([&] {
          json.attribute("schema", "wafer-profile-activation");
          json.attribute("schema_version",
                         int64_t(runtime::kProfileCompanionSchemaVersion));
          json.attribute("production_manifest_sha256", *productionDigest);
          json.attributeObject("metadata_sha256", [&] {
            json.attribute("plan.json", *planDigest);
            json.attribute("site-map.json", *siteMapDigest);
          });
        });
      },
      diagnostics);
}

static mlir::LogicalResult
makeProfileCompanionWorldAccessible(llvm::StringRef companionRoot,
                                    llvm::raw_ostream &diagnostics) {
  auto setPermissions = [&](llvm::StringRef path) -> mlir::LogicalResult {
    if (std::error_code error =
            llvm::sys::fs::setPermissions(path, llvm::sys::fs::all_all)) {
      reject(diagnostics, "failed to set profile companion permissions for '" +
                              path.str() + "': " + error.message());
      return mlir::failure();
    }
    return mlir::success();
  };

  std::error_code walkError;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(companionRoot, walkError, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(walkError)) {
    if (walkError) {
      reject(diagnostics,
             "failed to walk profile companion while setting permissions: " +
                 walkError.message());
      return mlir::failure();
    }
    if (iterator->type() != llvm::sys::fs::file_type::directory_file &&
        iterator->type() != llvm::sys::fs::file_type::regular_file) {
      reject(diagnostics,
             "profile companion contains a non-regular permission target: '" +
                 iterator->path() + "'");
      return mlir::failure();
    }
    if (mlir::failed(setPermissions(iterator->path())))
      return mlir::failure();
  }
  if (walkError) {
    reject(diagnostics,
           "failed to walk profile companion while setting permissions: " +
               walkError.message());
    return mlir::failure();
  }
  return setPermissions(companionRoot);
}

} // namespace

mlir::LogicalResult stageTargetPackage(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<ExecutableBundle> &executableBundle,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle,
    CompilationIRTrace &irTrace) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan productTiming(
      "stage", "target-product", "ordinary-product");
  llvm::Expected<ExecutableBundle> compiledExecutableBundle =
      compileTensorProgramToExecutableBundleImpl(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLaunchSlot, irTrace);
  if (!compiledExecutableBundle) {
    llvm::consumeError(compiledExecutableBundle.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> stagedTargetArtifacts(transactionRoot);
  llvm::sys::path::append(stagedTargetArtifacts, "target-artifacts");
  llvm::SmallString<256> stagedPackage(transactionRoot);
  llvm::sys::path::append(stagedPackage, "package");
  if (mlir::failed(stageExecutablePackage(
          tensorProgramDirectory, *compiledExecutableBundle,
          stagedTargetArtifacts, stagedPackage, targetToolchain, diagnostics,
          failAfterTargetLaunchSlot, failAfterPackageLaunchSlot,
          targetLLVMModuleBundle)))
    return mlir::failure();

  executableBundle.emplace(std::move(*compiledExecutableBundle));
  diagnostics << "wafer-compile: compile-stats stage=ordinary-product"
              << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture_packages=0 target_bundle_count=1\n";
  return mlir::success();
}

mlir::LogicalResult stageProfileTargetPackages(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    std::optional<int64_t> failAfterTargetLaunchSlot,
    std::optional<int64_t> failAfterPackageLaunchSlot,
    std::optional<ExecutableBundle> &executableBundle,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle,
    CompilationIRTrace &irTrace) {
  const CompileClock::time_point totalStart = CompileClock::now();
  wafer::support::ScopedCompileTimingSpan productTiming(
      "stage", "target-product", "profile-product");
  llvm::Expected<ExecutableBundle> compiled =
      compileTensorProgramToExecutableBundleImpl(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLaunchSlot, irTrace);
  if (!compiled) {
    llvm::consumeError(compiled.takeError());
    return mlir::failure();
  }

  llvm::SmallString<256> productionTargetArtifacts(transactionRoot);
  llvm::sys::path::append(productionTargetArtifacts, "target-artifacts");
  llvm::SmallString<256> productionPackage(transactionRoot);
  llvm::sys::path::append(productionPackage, "package");
  std::optional<TargetLLVMModuleBundle> productionTargetLLVM;
  if (mlir::failed(stageExecutablePackage(
          tensorProgramDirectory, *compiled, productionTargetArtifacts,
          productionPackage, targetToolchain, diagnostics,
          failAfterTargetLaunchSlot, failAfterPackageLaunchSlot,
          productionTargetLLVM)))
    return mlir::failure();

  llvm::SmallString<256> companionRoot(transactionRoot);
  llvm::sys::path::append(companionRoot, "profile-companion");

  ProfileCapturePackages productionCaptures;
  std::optional<TargetLLVMModuleBundle> productionTraceTargetLLVM;
  if (mlir::failed(stageCapturePackages(
          tensorProgramDirectory, transactionRoot, companionRoot, *compiled,
          targetToolchain, diagnostics, productionCaptures,
          productionTraceTargetLLVM)))
    return mlir::failure();

  if (!productionTargetLLVM || !productionTraceTargetLLVM ||
      mlir::failed(writeProfileCompanion(
          companionRoot, productionPackage, *compiled, *productionTargetLLVM,
          *productionTraceTargetLLVM, productionCaptures, diagnostics)))
    return mlir::failure();
  if (mlir::failed(
          makeProfileCompanionWorldAccessible(companionRoot, diagnostics)))
    return mlir::failure();

  executableBundle.emplace(std::move(*compiled));
  targetLLVMModuleBundle.emplace(std::move(*productionTargetLLVM));
  diagnostics << "wafer-compile: compile-stats stage=profile-product"
              << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture_packages=" << kProfileCaptures.size()
              << " target_bundle_count=" << 1 + kProfileCaptures.size()
              << " target_tile_lowerings="
              << executionConfig.getPhysicalTileCount() *
                     static_cast<int64_t>(1 + kProfileCaptures.size())
              << "\n";
  return mlir::success();
}

} // namespace wafer::compiler::detail
