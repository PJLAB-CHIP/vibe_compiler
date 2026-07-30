//===- TargetPackagePublication.cpp - Staged target/package build -------===//

#include "AcceptedCallClosure.h"
#include "CompilationInternal.h"
#include "CompilationStatistics.h"
#include "ExecutableBundleInternal.h"
#include "PackageInternal.h"
#include "StaticFixedSlotQualification.h"
#include "TargetArtifactInternal.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Package.h"
#include "Wafer/Runtime/PackageManifest.h"
#include "Wafer/Runtime/ProfileCompanion.h"
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
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle,
    ProfileCaptureKind profileCapture = ProfileCaptureKind::None) {
  const CompileClock::time_point totalStart = CompileClock::now();
  const CompileClock::time_point targetIRStart = CompileClock::now();
  llvm::Expected<TargetLLVMModuleBundle> targetLLVMModules =
      compileExecutableBundleToTargetLLVMModulesImpl(
          executableBundle, diagnostics, failAfterTargetLogicalRank,
          profileCapture);
  if (!targetLLVMModules) {
    llvm::consumeError(targetLLVMModules.takeError());
    return mlir::failure();
  }
  const int64_t targetIRWallMs = elapsedCompileMilliseconds(targetIRStart);
  const CompileClock::time_point targetArtifactStart = CompileClock::now();
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
  const CompileClock::time_point packageStart = CompileClock::now();
  llvm::Expected<PackageBundle> package = assemblePackageBundleImpl(
      tensorProgramDirectory, executableBundle, *targetArtifacts, stagedPackage,
      diagnostics, failAfterPackageLogicalRank);
  if (!package) {
    llvm::consumeError(package.takeError());
    return mlir::failure();
  }
  const int64_t packageWallMs = elapsedCompileMilliseconds(packageStart);
  diagnostics << "wafer-compile: compile-stats stage=target-ir-lowering"
              << " wall_ms=" << targetIRWallMs
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture=" << stringifyProfileCaptureKind(profileCapture)
              << " rank_lowerings=" << targetLLVMModules->getModules().size()
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
              << " rank_count="
              << executableBundle.getExecutionConfig().getRankCount() << "\n";
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
  const auto &rankExecutables = bundle.getRankExecutables();
  if (rankExecutables.size() !=
      static_cast<size_t>(bundle.getExecutionConfig().getRankCount()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost rank domain differs from execution config");

  llvm::SmallVector<mlir::Operation *, 16> rankRoots;
  rankRoots.reserve(rankExecutables.size());
  for (auto [expectedRank, rank] : llvm::enumerate(rankExecutables)) {
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile static cost rank domain is not canonical");
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(rank.getModule(), rank.getEntrySymbol());
    if (!closure)
      return llvm::joinErrors(
          llvm::createStringError(
              llvm::errc::invalid_argument,
              "profile static cost accepted call closure is invalid"),
          closure.takeError());
    rankRoots.push_back(closure->entry.getOperation());
  }

  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy(
          bundle.getExecutionConfig().getTargetProfileId());
  analysis::WholeCardInstructionProgramCost cost =
      analysis::analyzeWholeCardInstructionProgramCost(rankRoots, policy);
  if (cost.rankCosts.size() != rankExecutables.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile static cost analysis omitted an accepted rank");

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
  model.ranks.reserve(cost.rankCosts.size());
  for (auto [logicalRank, rankCost] : llvm::enumerate(cost.rankCosts)) {
    runtime::ProfileStaticRankWork work;
    work.npuF16Bf16LogicalOps =
        makeProfileStaticCostMetric(rankCost.compute.npuF16Bf16LogicalOps);
    work.npuOtherLogicalOps =
        makeProfileStaticCostMetric(rankCost.compute.npuOtherLogicalOps);
    work.vectorF16Bf16LogicalOps =
        makeProfileStaticCostMetric(rankCost.compute.vectorF16Bf16LogicalOps);
    work.vectorF32LogicalOps =
        makeProfileStaticCostMetric(rankCost.compute.vectorF32LogicalOps);
    work.vectorOtherLogicalOps =
        makeProfileStaticCostMetric(rankCost.compute.vectorOtherLogicalOps);
    work.ddrReadBytes = makeProfileStaticCostMetric(rankCost.ddrReadBytes);
    work.ddrWriteBytes = makeProfileStaticCostMetric(rankCost.ddrWriteBytes);
    work.spmMovementBytes =
        makeProfileStaticCostMetric(rankCost.spmMovementBytes);
    work.nocTransmitBytes =
        makeProfileStaticCostMetric(rankCost.noc.aggregateTransmitBytes);
    work.nocReceiveBytes =
        makeProfileStaticCostMetric(rankCost.noc.aggregateReceiveBytes);
    work.directionalNoCTransmitBytes.north = makeProfileStaticCostMetric(
        rankCost.noc.directional(analysis::NoCDirection::North));
    work.directionalNoCTransmitBytes.east = makeProfileStaticCostMetric(
        rankCost.noc.directional(analysis::NoCDirection::East));
    work.directionalNoCTransmitBytes.south = makeProfileStaticCostMetric(
        rankCost.noc.directional(analysis::NoCDirection::South));
    work.directionalNoCTransmitBytes.west = makeProfileStaticCostMetric(
        rankCost.noc.directional(analysis::NoCDirection::West));
    work.collectiveNoCTransmitBytes.collectivePermute =
        makeProfileStaticCostMetric(rankCost.noc.collective(
            analysis::NoCCollectiveKind::CollectivePermute));
    work.collectiveNoCTransmitBytes.allToAll = makeProfileStaticCostMetric(
        rankCost.noc.collective(analysis::NoCCollectiveKind::AllToAll));
    work.collectiveNoCTransmitBytes.allGather = makeProfileStaticCostMetric(
        rankCost.noc.collective(analysis::NoCCollectiveKind::AllGather));
    work.collectiveNoCTransmitBytes.reduceScatter = makeProfileStaticCostMetric(
        rankCost.noc.collective(analysis::NoCCollectiveKind::ReduceScatter));
    work.collectiveNoCTransmitBytes.allReduce = makeProfileStaticCostMetric(
        rankCost.noc.collective(analysis::NoCCollectiveKind::AllReduce));
    model.ranks.push_back({static_cast<int64_t>(logicalRank), std::move(work)});
  }
  return model;
}

static void
writeVariantMetadata(llvm::json::OStream &json, llvm::StringRef id,
                     llvm::StringRef role, llvm::StringRef packageReference,
                     llvm::StringRef manifestDigest,
                     const runtime::ProfileStaticCostModel &staticCostModel) {
  json.object([&] {
    json.attribute("id", id);
    json.attribute("role", role);
    json.attribute("package_ref", packageReference);
    json.attribute("manifest_sha256", manifestDigest);
    json.attributeBegin("static_cost_model");
    runtime::writeProfileStaticCostModel(json, staticCostModel);
    json.attributeEnd();
  });
}

struct ProfileVariantSiteMaps {
  std::string variantId;
  std::vector<std::vector<ProfileTargetCallSite>> ranks;
};

static llvm::Expected<ProfileVariantSiteMaps>
collectTargetCallSites(llvm::StringRef variantId,
                       const TargetLLVMModuleBundle &bundle) {
  ProfileVariantSiteMaps maps;
  maps.variantId = variantId.str();
  maps.ranks.reserve(bundle.getModules().size());
  for (auto [expectedRank, rankModule] : llvm::enumerate(bundle.getModules())) {
    if (rankModule.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "profile site-map rank domain is not canonical");
    llvm::Expected<std::vector<ProfileTargetCallSite>> sites =
        collectProfileTargetCallSites(rankModule.getModule(),
                                      rankModule.getEntrySymbol(),
                                      rankModule.getTargetProfileId());
    if (!sites)
      return sites.takeError();
    for (auto [expectedSite, site] : llvm::enumerate(*sites))
      if (site.siteId != expectedSite)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "profile site-map IDs are not rank-local dense");
    maps.ranks.push_back(std::move(*sites));
  }
  return maps;
}

static void writeTargetCallSites(llvm::json::OStream &json,
                                 const ProfileVariantSiteMaps &maps) {
  json.object([&] {
    json.attribute("variant_id", maps.variantId);
    json.attributeArray("ranks", [&] {
      for (auto [logicalRank, sites] : llvm::enumerate(maps.ranks)) {
        json.object([&] {
          json.attribute("logical_rank", static_cast<int64_t>(logicalRank));
          json.attributeArray("sites", [&] {
            for (const ProfileTargetCallSite &site : sites)
              json.object([&] {
                json.attribute("site_id", site.siteId);
                json.attribute("target_call_ordinal", site.targetCallOrdinal);
                json.attribute("target_call_symbol", site.targetCallSymbol);
                json.attribute(
                    "site_kind",
                    runtime::stringifyProfileTargetSiteKind(site.siteKind));
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
    });
  });
}

struct ProfileCapturePackageMetadata {
  ProfileCaptureKind capture = ProfileCaptureKind::None;
  std::string packageReference;
  std::string manifestDigest;
};

struct ProfileVariantCapturePackages {
  std::string variantId;
  std::array<ProfileCapturePackageMetadata, 2> captures;
};

static constexpr std::array<ProfileCaptureKind, 2> kProfileCaptures = {
    ProfileCaptureKind::Count, ProfileCaptureKind::Trace};

static mlir::LogicalResult stageVariantCapturePackages(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    llvm::StringRef companionRoot, llvm::StringRef variantId,
    const ExecutableBundle &executableBundle,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    ProfileVariantCapturePackages &metadata,
    std::optional<TargetLLVMModuleBundle> &traceTargetLLVM) {
  metadata.variantId = variantId.str();
  for (auto [index, capture] : llvm::enumerate(kProfileCaptures)) {
    llvm::SmallString<256> package(companionRoot);
    llvm::sys::path::append(package, "captures", variantId,
                            stringifyProfileCaptureKind(capture));
    llvm::SmallString<256> artifacts(transactionRoot);
    llvm::sys::path::append(artifacts, "profile-capture-target-artifacts",
                            variantId, stringifyProfileCaptureKind(capture));
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
    llvm::sys::path::append(reference, "captures", variantId,
                            stringifyProfileCaptureKind(capture));
    metadata.captures[index] = {capture, reference.str().str(), *digest};
    if (capture == ProfileCaptureKind::Trace)
      traceTargetLLVM.emplace(std::move(*targetLLVM));
  }
  return mlir::success();
}

static mlir::LogicalResult writeProfileCompanion(
    llvm::StringRef companionRoot, llvm::StringRef publishedPackageName,
    llvm::StringRef productionPackage, const ExecutableBundle &productionBundle,
    const TargetLLVMModuleBundle &productionTargetLLVM,
    const TargetLLVMModuleBundle &productionTraceTargetLLVM,
    const ProfileVariantCapturePackages &productionCaptures,
    llvm::raw_ostream &diagnostics) {
  if (createDirectory(companionRoot, diagnostics))
    return mlir::failure();

  llvm::Expected<std::string> productionDigest =
      getPackageManifestDigest(productionPackage);
  if (!productionDigest) {
    reject(diagnostics, llvm::toString(productionDigest.takeError()));
    return mlir::failure();
  }

  std::string productionReference = ("../" + publishedPackageName).str();
  if (productionTargetLLVM.getModules().size() !=
      productionTraceTargetLLVM.getModules().size()) {
    reject(diagnostics,
           "profile trace rank domain differs from final production");
    return mlir::failure();
  }
  for (auto [expectedRank, pair] :
       llvm::enumerate(llvm::zip(productionTargetLLVM.getModules(),
                                 productionTraceTargetLLVM.getModules()))) {
    const TargetLLVMModule &finalRank = std::get<0>(pair);
    const TargetLLVMModule &traceRank = std::get<1>(pair);
    if (finalRank.getLogicalRank() != static_cast<int64_t>(expectedRank) ||
        traceRank.getLogicalRank() != static_cast<int64_t>(expectedRank) ||
        finalRank.getEntrySymbol() != traceRank.getEntrySymbol()) {
      reject(diagnostics,
             "profile trace rank or entry identity differs from final "
             "production");
      return mlir::failure();
    }
    if (llvm::Error error = verifyProfileTargetCallSiteIdentity(
            finalRank.getModule(), finalRank.getEntrySymbol(),
            traceRank.getModule(), traceRank.getEntrySymbol(),
            finalRank.getTargetProfileId())) {
      reject(diagnostics, llvm::toString(std::move(error)));
      return mlir::failure();
    }
  }
  llvm::Expected<ProfileVariantSiteMaps> productionSites =
      collectTargetCallSites("final-artifact", productionTargetLLVM);
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

  llvm::SmallString<256> variantsPath(companionRoot);
  llvm::sys::path::append(variantsPath, "variants.json");
  if (mlir::failed(writeJSONFile(
          variantsPath,
          [&](llvm::json::OStream &json) {
            json.object([&] {
              json.attribute("schema", "wafer-profile-variants");
              json.attribute("schema_version",
                             int64_t(runtime::kProfileCompanionSchemaVersion));
              json.attribute(
                  "rank_count",
                  productionBundle.getExecutionConfig().getRankCount());
              json.attributeArray("variants", [&] {
                writeVariantMetadata(json, "final-artifact", "final-artifact",
                                     productionReference, *productionDigest,
                                     *productionStaticCost);
              });
            });
          },
          diagnostics)))
    return mlir::failure();

  llvm::SmallString<256> siteMapPath(companionRoot);
  llvm::sys::path::append(siteMapPath, "site-map.json");
  if (mlir::failed(writeJSONFile(
          siteMapPath,
          [&](llvm::json::OStream &json) {
            json.object([&] {
              json.attribute("schema", "wafer-profile-target-call-site-map");
              json.attribute("schema_version",
                             int64_t(runtime::kProfileCompanionSchemaVersion));
              json.attribute("site_basis",
                             "verified-target-llvm-entry-reachable-profile-"
                             "target-call-preorder");
              json.attribute("correlation_basis",
                             runtime::kProfileSiteCorrelationBasis);
              json.attribute(
                  "target_call_registry_size",
                  static_cast<int64_t>(getTargetCallDescriptors().size()));
              json.attributeArray("variants", [&] {
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
              json.attribute(
                  "rank_count",
                  productionBundle.getExecutionConfig().getRankCount());
              json.attribute("variant_metadata", "variants.json");
              json.attribute("site_map", "site-map.json");
              json.attribute("site_identity",
                             "final-rank-local-typed-target-site-id-and-"
                             "correlation-key");
              json.attributeArray("execution_packages", [&] {
                json.object([&] {
                  json.attribute("variant_id", "final-artifact");
                  json.attribute("package_ref", productionReference);
                  json.attribute("manifest_sha256", *productionDigest);
                });
              });
              json.attributeArray("capture_packages", [&] {
                for (const ProfileCapturePackageMetadata &capture :
                     productionCaptures.captures)
                  json.object([&] {
                    json.attribute("variant_id", productionCaptures.variantId);
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
  llvm::Expected<std::string> variantsDigest = getFileDigest(variantsPath);
  llvm::Expected<std::string> siteMapDigest = getFileDigest(siteMapPath);
  if (!planDigest || !variantsDigest || !siteMapDigest) {
    if (!planDigest)
      reject(diagnostics, llvm::toString(planDigest.takeError()));
    if (!variantsDigest)
      reject(diagnostics, llvm::toString(variantsDigest.takeError()));
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
            json.attribute("variants.json", *variantsDigest);
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
    WholeVariantSelectionMode selectionMode,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<ExecutableBundle> &executableBundle,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle) {
  const CompileClock::time_point totalStart = CompileClock::now();
  llvm::Expected<ExecutableBundle> compiledExecutableBundle =
      compileTensorProgramToExecutableBundleImpl(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLogicalRank, selectionMode);
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
          failAfterTargetLogicalRank, failAfterPackageLogicalRank,
          targetLLVMModuleBundle)))
    return mlir::failure();

  if (producesStaticFixedSlotQualificationCompanion(selectionMode)) {
    llvm::SmallString<256> qualificationCompanion(transactionRoot);
    llvm::sys::path::append(qualificationCompanion, "qualification-companion");
    if (mlir::failed(stageStaticFixedSlotQualificationCompanion(
            qualificationCompanion, stagedPackage, *compiledExecutableBundle,
            selectionMode ==
                WholeVariantSelectionMode::QualifyDirectDTEComputeOverlap,
            diagnostics)))
      return mlir::failure();
  }

  executableBundle.emplace(std::move(*compiledExecutableBundle));
  diagnostics << "wafer-compile: compile-stats stage=ordinary-product"
              << " wall_ms=" << elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << getCompilePeakRSSKiB()
              << " capture_packages=0 target_bundle_count=1\n";
  return mlir::success();
}

mlir::LogicalResult stageProfileTargetPackages(
    llvm::StringRef tensorProgramDirectory, llvm::StringRef transactionRoot,
    llvm::StringRef publishedPackageName,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank,
    std::optional<int64_t> failAfterTargetLogicalRank,
    std::optional<int64_t> failAfterPackageLogicalRank,
    std::optional<ExecutableBundle> &executableBundle,
    std::optional<TargetLLVMModuleBundle> &targetLLVMModuleBundle) {
  const CompileClock::time_point totalStart = CompileClock::now();
  llvm::Expected<ExecutableBundle> compiled =
      compileTensorProgramToExecutableBundleImpl(
          tensorProgramDirectory, executionConfig, optimizations, diagnostics,
          failAfterLogicalRank, WholeVariantSelectionMode::Production);
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
          failAfterTargetLogicalRank, failAfterPackageLogicalRank,
          productionTargetLLVM)))
    return mlir::failure();

  llvm::SmallString<256> companionRoot(transactionRoot);
  llvm::sys::path::append(companionRoot, "profile-companion");

  ProfileVariantCapturePackages productionCaptures;
  std::optional<TargetLLVMModuleBundle> productionTraceTargetLLVM;
  if (mlir::failed(stageVariantCapturePackages(
          tensorProgramDirectory, transactionRoot, companionRoot,
          "final-artifact", *compiled, targetToolchain, diagnostics,
          productionCaptures, productionTraceTargetLLVM)))
    return mlir::failure();

  if (!productionTargetLLVM || !productionTraceTargetLLVM ||
      mlir::failed(writeProfileCompanion(
          companionRoot, publishedPackageName, productionPackage, *compiled,
          *productionTargetLLVM, *productionTraceTargetLLVM, productionCaptures,
          diagnostics)))
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
              << " target_rank_lowerings="
              << executionConfig.getRankCount() *
                     static_cast<int64_t>(1 + kProfileCaptures.size())
              << "\n";
  return mlir::success();
}

} // namespace wafer::compiler::detail
