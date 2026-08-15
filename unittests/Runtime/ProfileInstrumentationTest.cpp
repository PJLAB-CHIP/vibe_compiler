#include "Wafer/Runtime/ProfileInstrumentation.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetIdentity.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

wafer::RuntimeLaunchContract makeGridLaunch() {
  return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Grid,
      wafer::KernelEntryABI::TileMajorPointerTable,
      {wafer::RuntimeLaunchPhaseRole::Main}));
}

wafer::RuntimeLaunchContract makeClusterLaunch() {
  return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Cluster,
      wafer::KernelEntryABI::TileMajorPointerTable,
      {wafer::RuntimeLaunchPhaseRole::Prepare,
       wafer::RuntimeLaunchPhaseRole::Main}));
}

class ProfileInstrumentationTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "wafer-profile-instrumentation-test", root));
    primary = root;
    llvm::sys::path::append(primary, "package");
    instrumentation = root;
    llvm::sys::path::append(instrumentation, "package.profile");
    ASSERT_FALSE(llvm::sys::fs::create_directories(primary));
    ASSERT_FALSE(llvm::sys::fs::create_directories(instrumentation));
    ASSERT_NO_FATAL_FAILURE(writePackage(primary, /*outputBytes=*/4,
                                         /*recordBytes=*/0));
    ASSERT_NO_FATAL_FAILURE(writeInstrumentation());
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  static std::string moduleDigest() {
    llvm::SHA256 hasher;
    hasher.update(llvm::StringRef("\x7f"
                                  "ELFprofile-instrumentation-test"));
    return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  }

  static void writeText(llvm::StringRef path, llvm::StringRef contents) {
    std::error_code error;
    llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
    ASSERT_FALSE(error);
    output << contents;
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  static std::string digestFile(llvm::StringRef path) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    EXPECT_TRUE(static_cast<bool>(buffer));
    if (!buffer)
      return {};
    llvm::SHA256 hasher;
    hasher.update((*buffer)->getBuffer());
    return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  }

  static std::string manifestDigest(llvm::StringRef package) {
    llvm::SmallString<256> path(package);
    llvm::sys::path::append(path, wafer::runtime::kPackageManifestFileName);
    return digestFile(path);
  }

  static wafer::runtime::ProfileStaticCostMetric
  knownStaticCost(uint64_t value) {
    return {"known", value, "none"};
  }

  static int64_t tileForLaunchSlot(int64_t launchSlot,
                                           bool permuteTiles) {
    if (!permuteTiles || launchSlot > 1)
      return launchSlot;
    return 1 - launchSlot;
  }

  static wafer::runtime::ProfileStaticCostModel
  makeStaticCostModel(bool permuteTiles = false) {
    using namespace wafer::runtime;
    ProfileStaticCostModel model;
    model.model = kProfileStaticCostModelName.str();
    model.scope = kProfileStaticCostModelScope.str();
    model.rates.cardDDRBytesPerSecond = UINT64_C(200000000000);
    model.rates.directionalNoCBytesPerSecond = UINT64_C(128000000000);
    model.rates.f16Bf16NPULogicalOpsPerSecondPerTile = UINT64_C(8000000000000);
    model.rates.f16Bf16VectorLogicalOpsPerSecondPerTile = UINT64_C(64000000000);
    model.rates.f32VectorLogicalOpsPerSecondPerTile = UINT64_C(32000000000);
    for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
      const int64_t tile = tileForLaunchSlot(launchSlot, permuteTiles);
      ProfileStaticTileWork work;
      work.npuF16Bf16LogicalOps = knownStaticCost(0);
      work.npuOtherLogicalOps = {"unsupported", std::nullopt,
                                 "unsupported-compute-type"};
      work.vectorF16Bf16LogicalOps =
          knownStaticCost(static_cast<uint64_t>(tile + 1));
      work.vectorF32LogicalOps = knownStaticCost(0);
      work.vectorOtherLogicalOps = knownStaticCost(0);
      work.ddrReadBytes = knownStaticCost(64);
      work.ddrWriteBytes = knownStaticCost(32);
      work.spmMovementBytes = knownStaticCost(96);
      work.nocTransmitBytes = knownStaticCost(0);
      work.nocReceiveBytes = knownStaticCost(0);
      work.directionalNoCTransmitBytes.north = {"unavailable", std::nullopt,
                                                "unresolved-noc-route"};
      work.directionalNoCTransmitBytes.east = knownStaticCost(0);
      work.directionalNoCTransmitBytes.south = knownStaticCost(0);
      work.directionalNoCTransmitBytes.west = knownStaticCost(0);
      model.tiles.push_back(
          {wafer::CardId(0), wafer::TileId(tile),
           wafer::runtime::LaunchSlotId(launchSlot), std::move(work)});
    }
    return model;
  }

  void writeActivation(llvm::StringRef productionDigestOverride = {}) {
    llvm::SmallString<256> activationPath(instrumentation);
    llvm::sys::path::append(
        activationPath,
        wafer::runtime::kProfileInstrumentationActivationFileName);
    llvm::SmallString<256> planPath(instrumentation);
    llvm::sys::path::append(
        planPath, wafer::runtime::kProfileInstrumentationPlanFileName);
    llvm::SmallString<256> siteMapPath(instrumentation);
    llvm::sys::path::append(
        siteMapPath, wafer::runtime::kProfileInstrumentationSiteMapFileName);
    std::error_code error;
    llvm::raw_fd_ostream output(activationPath, error, llvm::sys::fs::OF_Text);
    ASSERT_FALSE(error);
    llvm::json::OStream json(output, 2);
    json.object([&] {
      json.attribute("schema", "wafer-profile-activation");
      json.attribute(
          "schema_version",
          int64_t(wafer::runtime::kProfileInstrumentationSchemaVersion));
      json.attribute("primary_manifest_sha256", productionDigestOverride.empty()
                                                    ? manifestDigest(primary)
                                                    : productionDigestOverride);
      json.attributeObject("metadata_sha256", [&] {
        json.attribute("plan.json", digestFile(planPath));
        json.attribute("site-map.json", digestFile(siteMapPath));
      });
    });
    output << "\n";
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  static void writePackage(
      llvm::StringRef package, uint64_t outputBytes, uint64_t recordBytes,
      llvm::StringRef profilerName = "tx81_profiler_record",
      llvm::StringRef resourceNamePrefix = "",
      wafer::KernelLaunchForm launchForm = wafer::KernelLaunchForm::Grid,
      wafer::TargetIdentityId targetIdentity =
          wafer::TargetIdentityId::waferTx81SingleCard(),
      bool permuteTiles = false) {
    using namespace wafer::runtime;
    llvm::SmallString<256> modules(package);
    llvm::sys::path::append(modules, "modules");
    ASSERT_FALSE(llvm::sys::fs::create_directories(modules));

    PackageManifest manifest(targetIdentity, wafer::kCurrentKernelRuntimeABI,
                             launchForm == wafer::KernelLaunchForm::Cluster
                                 ? makeClusterLaunch()
                                 : makeGridLaunch(),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    manifest.resources = {{ResourceId(0),
                           CardResourceScope{wafer::CardId(0)},
                           PackageResourceRole::UserInput,
                           0,
                           (resourceNamePrefix + "input").str(),
                           {"f32", {1}},
                           4,
                           4,
                           PackageAccessMode::ReadOnly,
                           true},
                          {ResourceId(1),
                           CardResourceScope{wafer::CardId(0)},
                           PackageResourceRole::Output,
                           0,
                           (resourceNamePrefix + "output").str(),
                           {"f32", {1}},
                           outputBytes,
                           4,
                           PackageAccessMode::WriteOnly,
                           true}};
    llvm::SmallString<256> modulePath(modules);
    llvm::sys::path::append(modulePath, "kernel.so");
    writeText(modulePath, llvm::StringRef("\x7f"
                                          "ELFprofile-instrumentation-test"));
    manifest.modules.push_back(
        {ModuleId(0), "modules/kernel.so", moduleDigest(),
         wafer::kCurrentTargetModuleFormat.str(),
         launchForm == wafer::KernelLaunchForm::Cluster
             ? std::vector<
                   PackageModuleExportRecord>{{PackageModuleExportRole::Prepare,
                                               "prepare"},
                                              {PackageModuleExportRole::Main,
                                               "main"}}
             : std::vector<PackageModuleExportRecord>{
                   {PackageModuleExportRole::Main, "main"}}});
    for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
      const int64_t tile = tileForLaunchSlot(launchSlot, permuteTiles);
      std::vector<PackageABISlotBinding> slots = {
          {0, ResourceId(0), PackageAccessMode::ReadOnly},
          {1, ResourceId(1), PackageAccessMode::WriteOnly}};
      if (recordBytes != 0) {
        ResourceId profiler(static_cast<uint64_t>(launchSlot) + 2);
        manifest.resources.push_back(
            {profiler,
             TileResourceScope{wafer::CardId(0),
                               wafer::TileId(tile)},
             PackageResourceRole::Workspace,
             1,
             profilerName.str(),
             {"u8", {static_cast<int64_t>(recordBytes)}},
             recordBytes,
             WAFER_TX81_PROFILER_BUFFER_ALIGNMENT,
             PackageAccessMode::ReadWrite,
             false});
        slots.push_back({2, profiler, PackageAccessMode::ReadWrite});
      }
      manifest.entries.push_back(
          {EntryId(launchSlot), wafer::CardId(0),
           wafer::TileId(tile), LaunchSlotId(launchSlot), ModuleId(0),
           std::move(slots), PackageEntryCompletionKind::ReturnAfterLocalDrain,
           NoTransportRequirements{}});
    }
    llvm::Expected<VerifiedPackageManifest> verified =
        verifyPackageManifest(std::move(manifest), package);
    ASSERT_TRUE(static_cast<bool>(verified))
        << llvm::toString(verified.takeError());
    llvm::SmallString<256> manifestPath(package);
    llvm::sys::path::append(manifestPath, kPackageManifestFileName);
    writeText(manifestPath, serializeCanonicalPackageJson(*verified));
  }

  void writeInstrumentation(
      bool badSiteSymbol = false, llvm::StringRef finalDigestOverride = {},
      llvm::StringRef profilerName = "tx81_profiler_record",
      llvm::StringRef resourceNamePrefix = "", bool permuteTiles = false) {
    struct CaptureFixture {
      llvm::StringRef name;
      uint64_t recordBytes;
      std::string reference;
      std::string digest;
    };
    std::vector<CaptureFixture> captures;
    for (auto [name, recordBytes] :
         std::array<std::pair<llvm::StringRef, uint64_t>, 2>{
             {{"count", WAFER_TX81_PROFILER_MIN_BUFFER_BYTES},
              {"trace", 1024 * 1024}}}) {
      std::string reference = ("captures/" + name).str();
      llvm::SmallString<256> capturePath(instrumentation);
      llvm::sys::path::append(capturePath, reference);
      ASSERT_FALSE(llvm::sys::fs::create_directories(capturePath));
      ASSERT_NO_FATAL_FAILURE(writePackage(
          capturePath, /*outputBytes=*/4, recordBytes, profilerName,
          resourceNamePrefix, wafer::KernelLaunchForm::Grid,
          wafer::kCurrentTargetIdentity, permuteTiles));
      captures.push_back(
          {name, recordBytes, reference, manifestDigest(capturePath)});
    }

    llvm::SmallString<256> siteMapPath(instrumentation);
    llvm::sys::path::append(
        siteMapPath, wafer::runtime::kProfileInstrumentationSiteMapFileName);
    {
      std::error_code error;
      llvm::raw_fd_ostream output(siteMapPath, error, llvm::sys::fs::OF_Text);
      ASSERT_FALSE(error);
      llvm::json::OStream json(output, 2);
      llvm::ArrayRef<wafer::TargetCallDescriptor> descriptors =
          wafer::getTargetCallDescriptors();
      json.object([&] {
        json.attribute("schema", "wafer-profile-target-call-site-map");
        json.attribute(
            "schema_version",
            int64_t(wafer::runtime::kProfileInstrumentationSchemaVersion));
        json.attribute("card_count", int64_t(1));
        json.attribute("tile_count", int64_t(16));
        json.attribute("site_basis", "verified-target-llvm-entry-reachable-"
                                     "physical-tile-target-call-preorder");
        json.attribute("correlation_basis",
                       wafer::runtime::kProfileSiteCorrelationBasis);
        json.attribute("target_call_registry_size",
                       static_cast<int64_t>(descriptors.size()));
        json.attributeArray("tiles", [&] {
          for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
            const int64_t tile =
                tileForLaunchSlot(launchSlot, permuteTiles);
            json.object([&] {
              json.attribute("card_id", int64_t(0));
              json.attribute("tile_id", tile);
              json.attribute("launch_slot", launchSlot);
              json.attributeArray("sites", [&] {
                struct SiteFixture {
                  wafer::TargetCallBuiltin builtin;
                  llvm::StringRef kind;
                  std::optional<llvm::StringRef> engine;
                };
                std::vector<SiteFixture> sites = {
                    {wafer::TargetCallBuiltin::RDMA, "ncc-command", "RDMA"},
                    {wafer::TargetCallBuiltin::NCCJoin, "ncc-completion",
                     std::nullopt},
                    {wafer::TargetCallBuiltin::DirectDTEBegin,
                     "direct-dte-control", std::nullopt},
                };
                sites.push_back({wafer::TargetCallBuiltin::DirectDTESendIssue,
                                 "direct-dte-issue", "DIRECT_DTE"});
                sites.push_back({wafer::TargetCallBuiltin::DirectDTEWait,
                                 "direct-dte-wait", "DIRECT_DTE"});
                for (auto [siteId, site] : llvm::enumerate(sites)) {
                  const wafer::TargetCallDescriptor &descriptor =
                      wafer::getTargetCallDescriptor(site.builtin);
                  const auto *begin = descriptors.data();
                  const uint64_t ordinal =
                      static_cast<uint64_t>(&descriptor - begin);
                  json.object([&] {
                    json.attribute("site_id", static_cast<int64_t>(siteId));
                    json.attribute("function_ordinal", int64_t(0));
                    json.attribute("block_ordinal", int64_t(0));
                    json.attribute("instruction_ordinal",
                                   static_cast<int64_t>(siteId));
                    json.attribute("target_call_ordinal",
                                   static_cast<int64_t>(ordinal));
                    json.attribute("target_call_symbol",
                                   badSiteSymbol && siteId == 0
                                       ? "wafer_invalid_target_call"
                                       : descriptor.symbol);
                    json.attribute("site_kind", site.kind);
                    if (site.engine)
                      json.attribute("engine", *site.engine);
                    json.attribute("correlation_key",
                                   "registry:" + std::to_string(ordinal) +
                                       ":structural-occurrence:0");
                  });
                }
              });
            });
          }
        });
      });
      output << "\n";
    }

    llvm::SmallString<256> planPath(instrumentation);
    llvm::sys::path::append(
        planPath, wafer::runtime::kProfileInstrumentationPlanFileName);
    {
      std::error_code error;
      llvm::raw_fd_ostream output(planPath, error, llvm::sys::fs::OF_Text);
      ASSERT_FALSE(error);
      llvm::json::OStream json(output, 2);
      json.object([&] {
        json.attribute("schema", "wafer-profile-plan");
        json.attribute(
            "schema_version",
            int64_t(wafer::runtime::kProfileInstrumentationSchemaVersion));
        json.attribute("card_count", int64_t(1));
        json.attribute("tile_count", int64_t(16));
        json.attribute("site_map", "site-map.json");
        json.attribute("site_key_contract",
                       wafer::runtime::kProfileSiteKeyContract);
        json.attributeBegin("static_cost_model");
        wafer::runtime::writeProfileStaticCostModel(
            json, makeStaticCostModel(permuteTiles));
        json.attributeEnd();
        json.attributeArray("capture_packages", [&] {
          for (const CaptureFixture &capture : captures)
            json.object([&] {
              json.attribute("capture", capture.name);
              json.attribute("package_ref", capture.reference);
              json.attribute("manifest_sha256", capture.digest);
              json.attribute("record_abi", wafer::runtime::kProfileRecordABI);
              json.attribute("record_bytes",
                             static_cast<int64_t>(capture.recordBytes));
            });
        });
      });
      output << "\n";
    }
    ASSERT_NO_FATAL_FAILURE(writeActivation(finalDigestOverride));
  }

  llvm::SmallString<256> root;
  llvm::SmallString<256> primary;
  llvm::SmallString<256> instrumentation;
};

TEST_F(ProfileInstrumentationTest,
       LoadsExactBoundInstrumentationAndSixteenTileSiteMap) {
  llvm::Expected<wafer::runtime::VerifiedProfileInstrumentation> loaded =
      wafer::runtime::loadVerifiedProfileInstrumentation(instrumentation,
                                                         primary);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  EXPECT_EQ(loaded->getSchemaVersion(),
            wafer::runtime::kProfileInstrumentationSchemaVersion);
  EXPECT_EQ(loaded->getProfiledPackage().getManifestDigest(),
            manifestDigest(primary));
  EXPECT_EQ(loaded->getCardCount(), 1);
  EXPECT_EQ(loaded->getTileCount(), 16);
  EXPECT_EQ(loaded->getCaptures().size(), 2u);
  EXPECT_EQ(loaded->getSiteMap().size(), 16u);
  EXPECT_EQ(loaded->getSiteCount(), 80u);
  const auto &profiledPackage = loaded->getProfiledPackage();
  EXPECT_EQ(profiledPackage.getPackage().getManifest().cardCount, 1);
  EXPECT_EQ(profiledPackage.getPackage().getManifest().tileCount, 16);
  const auto &staticCost = profiledPackage.getStaticCostModel();
  EXPECT_EQ(staticCost.model, wafer::runtime::kProfileStaticCostModelName);
  EXPECT_EQ(staticCost.scope, wafer::runtime::kProfileStaticCostModelScope);
  ASSERT_EQ(staticCost.tiles.size(), 16u);
  ASSERT_TRUE(
      staticCost.tiles.front().work.vectorF16Bf16LogicalOps.value.has_value());
  EXPECT_EQ(*staticCost.tiles.front().work.vectorF16Bf16LogicalOps.value, 1u);
  EXPECT_EQ(staticCost.tiles.front().work.npuOtherLogicalOps.knowledge,
            "unsupported");
  EXPECT_FALSE(
      staticCost.tiles.front().work.npuOtherLogicalOps.value.has_value());
  EXPECT_EQ(
      staticCost.tiles.front().work.directionalNoCTransmitBytes.north.knowledge,
      "unavailable");
  EXPECT_FALSE(staticCost.tiles.front()
                   .work.directionalNoCTransmitBytes.north.value.has_value());
  EXPECT_FALSE(staticCost.rates.spmMovementBytesPerSecond.has_value());
  const auto *trace =
      loaded->findCapture(wafer::runtime::ProfileCaptureKind::Trace);
  ASSERT_NE(trace, nullptr);
  EXPECT_EQ(trace->getRecordBytes(), UINT64_C(1024) * 1024);
  EXPECT_EQ(trace->getRecordABI(), wafer::runtime::kProfileRecordABI);
  for (wafer::runtime::ProfileCaptureKind capture :
       {wafer::runtime::ProfileCaptureKind::Count,
        wafer::runtime::ProfileCaptureKind::Trace}) {
    ASSERT_NE(loaded->findCapture(capture), nullptr);
  }
  llvm::ArrayRef<wafer::runtime::ProfileTileSiteMap> siteMap =
      loaded->getSiteMap();
  ASSERT_EQ(siteMap.size(), 16u);
  ASSERT_EQ(siteMap.front().sites.size(), 5u);
  EXPECT_EQ(siteMap.front().sites[0].siteKind,
            wafer::runtime::ProfileTargetSiteKind::NCCCommand);
  EXPECT_EQ(siteMap.front().sites[0].engine,
            wafer::runtime::ProfileTSMEngine::RDMA);
  EXPECT_EQ(siteMap.front().sites[1].siteKind,
            wafer::runtime::ProfileTargetSiteKind::NCCCompletion);
  EXPECT_FALSE(siteMap.front().sites[1].engine);
  EXPECT_EQ(siteMap.front().sites[2].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEControl);
  EXPECT_FALSE(siteMap.front().sites[2].engine);
  EXPECT_EQ(siteMap.front().sites[3].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEIssue);
  EXPECT_EQ(siteMap.front().sites[3].engine,
            wafer::runtime::ProfileTSMEngine::DirectDTE);
  EXPECT_EQ(siteMap.front().sites[4].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEWait);
  EXPECT_EQ(siteMap.front().sites[4].engine,
            wafer::runtime::ProfileTSMEngine::DirectDTE);
  EXPECT_FALSE(siteMap.front().sites.front().correlationKey.empty());
}

TEST_F(ProfileInstrumentationTest,
       PreservesExplicitTileAndLaunchSlotBinding) {
  ASSERT_NO_FATAL_FAILURE(writePackage(
      primary, /*outputBytes=*/4, /*recordBytes=*/0,
      /*profilerName=*/"tx81_profiler_record", /*resourceNamePrefix=*/"",
      wafer::KernelLaunchForm::Grid, wafer::kCurrentTargetIdentity,
      /*permuteTiles=*/true));
  ASSERT_NO_FATAL_FAILURE(writeInstrumentation(
      /*badSiteSymbol=*/false, /*finalDigestOverride=*/{},
      /*profilerName=*/"tx81_profiler_record", /*resourceNamePrefix=*/"",
      /*permuteTiles=*/true));

  llvm::Expected<wafer::runtime::VerifiedProfileInstrumentation> loaded =
      wafer::runtime::loadVerifiedProfileInstrumentation(instrumentation,
                                                         primary);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  const auto &profiledPackage = loaded->getProfiledPackage();
  llvm::ArrayRef<wafer::runtime::ProfileTileSiteMap> siteMap =
      loaded->getSiteMap();
  ASSERT_EQ(profiledPackage.getStaticCostModel().tiles.size(), 16u);
  ASSERT_EQ(siteMap.size(), 16u);
  EXPECT_EQ(profiledPackage.getStaticCostModel().tiles[0].tileId,
            wafer::TileId(1));
  EXPECT_EQ(profiledPackage.getStaticCostModel().tiles[0].launchSlot,
            wafer::runtime::LaunchSlotId(0));
  EXPECT_EQ(siteMap[1].tileId, wafer::TileId(0));
  EXPECT_EQ(siteMap[1].launchSlot, wafer::runtime::LaunchSlotId(1));
}

TEST_F(ProfileInstrumentationTest, MissingSiblingIsNotAnError) {
  llvm::SmallString<256> unrelated(root);
  llvm::sys::path::append(unrelated, "ordinary");
  auto loaded =
      wafer::runtime::loadSiblingProfileInstrumentationIfPresent(unrelated);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  EXPECT_FALSE(loaded->has_value());
}

TEST_F(ProfileInstrumentationTest, RejectsStaleManifestDigest) {
  ASSERT_NO_FATAL_FAILURE(writeInstrumentation(
      /*badSiteSymbol=*/false, "sha256:" + std::string(64, '0')));
  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("digest mismatch"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsDifferentRuntimeLaunchContract) {
  ASSERT_NO_FATAL_FAILURE(writePackage(
      primary, /*outputBytes=*/4, /*recordBytes=*/0,
      /*profilerName=*/"tx81_profiler_record",
      /*resourceNamePrefix=*/"", wafer::KernelLaunchForm::Cluster));
  ASSERT_NO_FATAL_FAILURE(writeInstrumentation());
  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(
      llvm::toString(loaded.takeError()).find("target/ABI contract differs"),
      std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsUnknownPlanField) {
  llvm::SmallString<256> planPath(instrumentation);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileInstrumentationPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  ASSERT_FALSE(corrupted.empty());
  corrupted.insert(corrupted.find('{') + 1, "\n  \"unknown\": 1,");
  ASSERT_NO_FATAL_FAILURE(writeText(planPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("unknown field"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest,
       RejectsLegacyRecordABIWithCurrentRecordBytes) {
  llvm::SmallString<256> planPath(instrumentation);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileInstrumentationPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  size_t recordABI = corrupted.find(wafer::runtime::kProfileRecordABI);
  ASSERT_NE(recordABI, std::string::npos);
  corrupted.replace(recordABI, wafer::runtime::kProfileRecordABI.size(),
                    "wafer-tx81-profiler-record-v2");
  ASSERT_NO_FATAL_FAILURE(writeText(planPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("record_abi"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsRetiredSchemaV9) {
  llvm::SmallString<256> activationPath(instrumentation);
  llvm::sys::path::append(
      activationPath,
      wafer::runtime::kProfileInstrumentationActivationFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(activationPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  const std::string current = "\"schema_version\": 10";
  size_t version = corrupted.find(current);
  ASSERT_NE(version, std::string::npos);
  corrupted.replace(version, current.size(), "\"schema_version\": 9");
  ASSERT_NO_FATAL_FAILURE(writeText(activationPath, corrupted));

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError())
                .find("schema_version is not supported"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsNonCanonicalStaticCostValue) {
  llvm::SmallString<256> planPath(instrumentation);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileInstrumentationPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  const std::string value = "\"value\": \"0\"";
  size_t position = corrupted.find(value);
  ASSERT_NE(position, std::string::npos);
  corrupted.replace(position, value.size(), "\"value\": \"00\"");
  ASSERT_NO_FATAL_FAILURE(writeText(planPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError())
                .find("canonical uint64 decimal string"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsRetiredUnknownStaticCostKnowledge) {
  llvm::SmallString<256> planPath(instrumentation);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileInstrumentationPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  const std::string current = "\"knowledge\": \"unavailable\"";
  size_t position = corrupted.find(current);
  ASSERT_NE(position, std::string::npos);
  corrupted.replace(position, current.size(), "\"knowledge\": \"unknown\"");
  ASSERT_NO_FATAL_FAILURE(writeText(planPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(
      llvm::toString(loaded.takeError()).find("knowledge is not supported"),
      std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsEngineOnCompletionSite) {
  llvm::SmallString<256> siteMapPath(instrumentation);
  llvm::sys::path::append(
      siteMapPath, wafer::runtime::kProfileInstrumentationSiteMapFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(siteMapPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  const std::string completion = "\"site_kind\": \"ncc-completion\",";
  size_t position = corrupted.find(completion);
  ASSERT_NE(position, std::string::npos);
  position += completion.size();
  corrupted.insert(position, "\n                      \"engine\": \"CT\",");
  ASSERT_NO_FATAL_FAILURE(writeText(siteMapPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("must not carry an engine"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsMissingEngineOnNCCCommandSite) {
  llvm::SmallString<256> siteMapPath(instrumentation);
  llvm::sys::path::append(
      siteMapPath, wafer::runtime::kProfileInstrumentationSiteMapFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(siteMapPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  const std::string engine = "\"engine\": \"RDMA\",\n";
  size_t position = corrupted.find(engine);
  ASSERT_NE(position, std::string::npos);
  corrupted.erase(position, engine.size());
  ASSERT_NO_FATAL_FAILURE(writeText(siteMapPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(
      llvm::toString(loaded.takeError()).find("requires its typed engine"),
      std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsUnsupportedSummaryCapture) {
  llvm::SmallString<256> planPath(instrumentation);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileInstrumentationPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  size_t capture = corrupted.find("\"capture\": \"count\"");
  ASSERT_NE(capture, std::string::npos);
  capture += std::string("\"capture\": \"").size();
  corrupted.replace(capture, std::string("count").size(), "summary");
  ASSERT_NO_FATAL_FAILURE(writeText(planPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(
      llvm::toString(loaded.takeError()).find("not a supported capture kind"),
      std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsMissingActivation) {
  llvm::SmallString<256> activationPath(instrumentation);
  llvm::sys::path::append(
      activationPath,
      wafer::runtime::kProfileInstrumentationActivationFileName);
  ASSERT_FALSE(llvm::sys::fs::remove(activationPath));
  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("activation"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsMetadataByteTamper) {
  llvm::SmallString<256> planPath(instrumentation);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileInstrumentationPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  ASSERT_NO_FATAL_FAILURE(
      writeText(planPath, (*existing)->getBuffer().str() + " "));
  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError())
                .find("activation metadata digest mismatch"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest, RejectsTargetCallOrdinalSymbolDisagreement) {
  ASSERT_NO_FATAL_FAILURE(writeInstrumentation(/*badSiteSymbol=*/true));
  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("do not agree"),
            std::string::npos);
}

TEST_F(ProfileInstrumentationTest,
       AcceptsRenamedTypedProfilerWorkspaceAndResources) {
  ASSERT_NO_FATAL_FAILURE(writeInstrumentation(
      /*badSiteSymbol=*/false, /*finalDigestOverride=*/{},
      /*profilerName=*/"diagnostic_name_only",
      /*resourceNamePrefix=*/"renamed_"));
  auto loaded = wafer::runtime::loadVerifiedProfileInstrumentation(
      instrumentation, primary);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  EXPECT_EQ(loaded->getCaptures().size(), 2u);
}

} // namespace
