#include "Wafer/Runtime/ProfileCompanion.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetProfile.h"

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

wafer::RuntimeLaunchContract makePerRankLaunch() {
  return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::PerRank,
      wafer::KernelEntryABI::RankLocalPointerBlockV1,
      {wafer::RuntimeLaunchPhaseRole::Main}));
}

wafer::RuntimeLaunchContract makeGridLaunch() {
  return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Grid,
      wafer::KernelEntryABI::RankMajorPointerTableV1,
      {wafer::RuntimeLaunchPhaseRole::Main}));
}

class ProfileCompanionTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "wafer-profile-companion-test", root));
    production = root;
    llvm::sys::path::append(production, "package");
    companion = root;
    llvm::sys::path::append(companion, "package.profile");
    ASSERT_FALSE(llvm::sys::fs::create_directories(production));
    ASSERT_FALSE(llvm::sys::fs::create_directories(companion));
    ASSERT_NO_FATAL_FAILURE(writePackage(production, /*outputBytes=*/4,
                                         /*recordBytes=*/0));
    ASSERT_NO_FATAL_FAILURE(writeCompanion());
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  static std::string moduleDigest() {
    llvm::SHA256 hasher;
    hasher.update(llvm::StringRef("\x7f"
                                  "ELFprofile-companion-test"));
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

  void writeActivation() {
    llvm::SmallString<256> activationPath(companion);
    llvm::sys::path::append(
        activationPath, wafer::runtime::kProfileCompanionActivationFileName);
    llvm::SmallString<256> planPath(companion);
    llvm::sys::path::append(planPath,
                            wafer::runtime::kProfileCompanionPlanFileName);
    llvm::SmallString<256> variantsPath(companion);
    llvm::sys::path::append(variantsPath,
                            wafer::runtime::kProfileCompanionVariantsFileName);
    llvm::SmallString<256> siteMapPath(companion);
    llvm::sys::path::append(siteMapPath,
                            wafer::runtime::kProfileCompanionSiteMapFileName);
    std::error_code error;
    llvm::raw_fd_ostream output(activationPath, error, llvm::sys::fs::OF_Text);
    ASSERT_FALSE(error);
    llvm::json::OStream json(output, 2);
    json.object([&] {
      json.attribute("schema", "wafer-profile-activation");
      json.attribute("schema_version",
                     int64_t(wafer::runtime::kProfileCompanionSchemaVersion));
      json.attribute("production_manifest_sha256", manifestDigest(production));
      json.attributeObject("metadata_sha256", [&] {
        json.attribute("plan.json", digestFile(planPath));
        json.attribute("variants.json", digestFile(variantsPath));
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
      wafer::KernelLaunchForm launchForm = wafer::KernelLaunchForm::PerRank) {
    using namespace wafer::runtime;
    llvm::SmallString<256> modules(package);
    llvm::sys::path::append(modules, "modules");
    ASSERT_FALSE(llvm::sys::fs::create_directories(modules));

    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(
        target.id, target.targetIdentity, target.kernelRuntimeABI,
        launchForm == wafer::KernelLaunchForm::Grid ? makeGridLaunch()
                                                    : makePerRankLaunch(),
        target.moduleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = 16;
    for (int64_t rank = 0; rank < 16; ++rank) {
      std::string rankText = std::to_string(rank);
      std::string moduleName =
          "rank_" + std::string(5 - rankText.size(), '0') + rankText + ".so";
      llvm::SmallString<256> modulePath(modules);
      llvm::sys::path::append(modulePath, moduleName);
      if (launchForm == wafer::KernelLaunchForm::PerRank || rank == 0) {
        writeText(modulePath, llvm::StringRef("\x7f"
                                              "ELFprofile-companion-test"));
      } else if (llvm::sys::fs::exists(modulePath)) {
        ASSERT_FALSE(llvm::sys::fs::remove(modulePath));
      }

      uint64_t resourcesPerRank = recordBytes == 0 ? 2 : 3;
      ResourceId input(static_cast<uint64_t>(rank) * resourcesPerRank);
      ResourceId output(input.getValue() + 1);
      manifest.resources.push_back({input,
                                    rank,
                                    PackageResourceRole::UserInput,
                                    0,
                                    (resourceNamePrefix + "input").str(),
                                    {"f32", {1}},
                                    4,
                                    4,
                                    PackageAccessMode::ReadOnly,
                                    true});
      manifest.resources.push_back({output,
                                    rank,
                                    PackageResourceRole::Output,
                                    0,
                                    (resourceNamePrefix + "output").str(),
                                    {"f32", {1}},
                                    outputBytes,
                                    4,
                                    PackageAccessMode::WriteOnly,
                                    true});
      if (launchForm == wafer::KernelLaunchForm::PerRank || rank == 0)
        manifest.modules.push_back(
            {ModuleId(launchForm == wafer::KernelLaunchForm::Grid ? 0 : rank),
             "modules/" + moduleName,
             moduleDigest(),
             target.moduleFormat.str(),
             {{PackageModuleExportRole::Main, "main"}}});
      std::vector<PackageABISlotBinding> slots = {
          {0, input, PackageAccessMode::ReadOnly},
          {1, output, PackageAccessMode::WriteOnly}};
      if (recordBytes != 0) {
        ResourceId profiler(output.getValue() + 1);
        manifest.resources.push_back(
            {profiler,
             rank,
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
          {EntryId(rank), rank,
           ModuleId(launchForm == wafer::KernelLaunchForm::Grid ? 0 : rank),
           std::move(slots), CompletionId(rank)});
      manifest.completions.push_back(
          {CompletionId(rank), rank, "entry_return"});
    }
    llvm::Expected<VerifiedPackageManifest> verified =
        verifyPackageManifest(std::move(manifest), package);
    ASSERT_TRUE(static_cast<bool>(verified))
        << llvm::toString(verified.takeError());
    llvm::SmallString<256> manifestPath(package);
    llvm::sys::path::append(manifestPath, kPackageManifestFileName);
    writeText(manifestPath, serializeCanonicalPackageJson(*verified));
  }

  void writeCompanion(bool badSiteSymbol = false,
                      llvm::StringRef finalDigestOverride = {},
                      llvm::StringRef profilerName = "tx81_profiler_record",
                      llvm::StringRef resourceNamePrefix = "") {
    std::string digest = finalDigestOverride.empty()
                             ? manifestDigest(production)
                             : finalDigestOverride.str();
    llvm::StringRef packageReference = "../package";
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
      std::string reference = ("captures/final-artifact/" + name).str();
      llvm::SmallString<256> capturePath(companion);
      llvm::sys::path::append(capturePath, reference);
      ASSERT_FALSE(llvm::sys::fs::create_directories(capturePath));
      ASSERT_NO_FATAL_FAILURE(writePackage(capturePath, /*outputBytes=*/4,
                                           recordBytes, profilerName,
                                           resourceNamePrefix));
      captures.push_back(
          {name, recordBytes, reference, manifestDigest(capturePath)});
    }

    llvm::SmallString<256> variantsPath(companion);
    llvm::sys::path::append(variantsPath,
                            wafer::runtime::kProfileCompanionVariantsFileName);
    {
      std::error_code error;
      llvm::raw_fd_ostream output(variantsPath, error, llvm::sys::fs::OF_Text);
      ASSERT_FALSE(error);
      llvm::json::OStream json(output, 2);
      json.object([&] {
        json.attribute("schema", "wafer-profile-variants");
        json.attribute("schema_version",
                       int64_t(wafer::runtime::kProfileCompanionSchemaVersion));
        json.attribute("rank_count", int64_t(16));
        json.attributeArray("variants", [&] {
          json.object([&] {
            json.attribute("id", "final-artifact");
            json.attribute("role", "final-artifact");
            json.attribute("package_ref", packageReference);
            json.attribute("manifest_sha256", digest);
          });
        });
      });
      output << "\n";
    }

    llvm::SmallString<256> siteMapPath(companion);
    llvm::sys::path::append(siteMapPath,
                            wafer::runtime::kProfileCompanionSiteMapFileName);
    {
      std::error_code error;
      llvm::raw_fd_ostream output(siteMapPath, error, llvm::sys::fs::OF_Text);
      ASSERT_FALSE(error);
      llvm::json::OStream json(output, 2);
      llvm::ArrayRef<wafer::TargetCallDescriptor> descriptors =
          wafer::getTargetCallDescriptors();
      json.object([&] {
        json.attribute("schema", "wafer-profile-target-call-site-map");
        json.attribute("schema_version",
                       int64_t(wafer::runtime::kProfileCompanionSchemaVersion));
        json.attribute(
            "site_basis",
            "verified-target-llvm-entry-reachable-profile-target-call-preorder");
        json.attribute("correlation_basis",
                       wafer::runtime::kProfileSiteCorrelationBasis);
        json.attribute("target_call_registry_size",
                       static_cast<int64_t>(descriptors.size()));
        json.attributeArray("variants", [&] {
          json.object([&] {
            json.attribute("variant_id", "final-artifact");
            json.attributeArray("ranks", [&] {
              for (int64_t rank = 0; rank < 16; ++rank) {
                json.object([&] {
                  json.attribute("logical_rank", rank);
                  json.attributeArray("sites", [&] {
                    struct SiteFixture {
                      wafer::TargetCallBuiltin builtin;
                      llvm::StringRef kind;
                      std::optional<llvm::StringRef> engine;
                    };
                    const std::array<SiteFixture, 4> sites = {{
                        {wafer::TargetCallBuiltin::RDMA, "ncc-command", "RDMA"},
                        {wafer::TargetCallBuiltin::LocalFence,
                         "ncc-completion", std::nullopt},
                        {wafer::TargetCallBuiltin::DirectDTEBegin,
                         "direct-dte-control", std::nullopt},
                        {wafer::TargetCallBuiltin::DirectDTEWait,
                         "direct-dte-wait", "DIRECT_DTE"},
                    }};
                    for (auto [siteId, site] : llvm::enumerate(sites)) {
                      const wafer::TargetCallDescriptor &descriptor =
                          wafer::getTargetCallDescriptor(site.builtin);
                      const auto *begin = descriptors.data();
                      const uint64_t ordinal =
                          static_cast<uint64_t>(&descriptor - begin);
                      json.object([&] {
                        json.attribute("site_id",
                                       static_cast<int64_t>(siteId));
                        json.attribute("function_ordinal", int64_t(0));
                        json.attribute("block_ordinal", int64_t(0));
                        json.attribute("instruction_ordinal",
                                       static_cast<int64_t>(siteId));
                        json.attribute("target_call_ordinal",
                                       static_cast<int64_t>(ordinal));
                        json.attribute(
                            "target_call_symbol",
                            badSiteSymbol && siteId == 0
                                ? "wafer_invalid_target_call"
                                : descriptor.symbol);
                        json.attribute("site_kind", site.kind);
                        if (site.engine)
                          json.attribute("engine", *site.engine);
                        json.attribute(
                            "correlation_key",
                            "registry:" + std::to_string(ordinal) +
                                ":structural-occurrence:0");
                      });
                    }
                  });
                });
              }
            });
          });
        });
      });
      output << "\n";
    }

    llvm::SmallString<256> planPath(companion);
    llvm::sys::path::append(planPath,
                            wafer::runtime::kProfileCompanionPlanFileName);
    {
      std::error_code error;
      llvm::raw_fd_ostream output(planPath, error, llvm::sys::fs::OF_Text);
      ASSERT_FALSE(error);
      llvm::json::OStream json(output, 2);
      json.object([&] {
        json.attribute("schema", "wafer-profile-plan");
        json.attribute("schema_version",
                       int64_t(wafer::runtime::kProfileCompanionSchemaVersion));
        json.attribute("rank_count", int64_t(16));
        json.attribute("variant_metadata", "variants.json");
        json.attribute("site_map", "site-map.json");
        json.attribute(
            "site_identity",
            "final-rank-local-typed-target-site-id-and-correlation-key");
        json.attributeArray("execution_packages", [&] {
          json.object([&] {
            json.attribute("variant_id", "final-artifact");
            json.attribute("package_ref", packageReference);
            json.attribute("manifest_sha256", digest);
          });
        });
        json.attributeArray("capture_packages", [&] {
          for (const CaptureFixture &capture : captures)
            json.object([&] {
              json.attribute("variant_id", "final-artifact");
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
    ASSERT_NO_FATAL_FAILURE(writeActivation());
  }

  llvm::SmallString<256> root;
  llvm::SmallString<256> production;
  llvm::SmallString<256> companion;
};

TEST_F(ProfileCompanionTest, LoadsExactBoundCompanionAndSixteenRankSiteMap) {
  llvm::Expected<wafer::runtime::VerifiedProfileCompanion> loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  EXPECT_EQ(loaded->getSchemaVersion(),
            wafer::runtime::kProfileCompanionSchemaVersion);
  EXPECT_EQ(loaded->getProductionManifestDigest(), manifestDigest(production));
  EXPECT_EQ(loaded->getRankCount(), 16);
  EXPECT_EQ(loaded->getVariants().size(), 1u);
  EXPECT_EQ(loaded->getCaptures().size(), 2u);
  EXPECT_EQ(loaded->getSiteMaps().size(), 1u);
  EXPECT_EQ(loaded->getSiteCount(), 64u);
  const auto *final =
      loaded->findVariant(wafer::runtime::ProfileVariantRole::FinalArtifact);
  ASSERT_NE(final, nullptr);
  EXPECT_EQ(final->getPackage().getManifest().rankCount, 16);
  const auto *trace = loaded->findCapture(
      "final-artifact", wafer::runtime::ProfileCaptureKind::Trace);
  ASSERT_NE(trace, nullptr);
  EXPECT_EQ(trace->getRecordBytes(), UINT64_C(1024) * 1024);
  EXPECT_EQ(trace->getRecordABI(), wafer::runtime::kProfileRecordABI);
  for (wafer::runtime::ProfileCaptureKind capture :
       {wafer::runtime::ProfileCaptureKind::Count,
        wafer::runtime::ProfileCaptureKind::Trace}) {
    ASSERT_NE(loaded->findCapture("final-artifact", capture), nullptr);
  }
  EXPECT_EQ(loaded->findCapture(
                "final-artifact",
                wafer::runtime::ProfileCaptureKind::Summary),
            nullptr);
  const auto *siteMap = loaded->findSiteMap("final-artifact");
  ASSERT_NE(siteMap, nullptr);
  ASSERT_EQ(siteMap->ranks.size(), 16u);
  ASSERT_EQ(siteMap->ranks.front().sites.size(), 4u);
  EXPECT_EQ(siteMap->ranks.front().sites[0].siteKind,
            wafer::runtime::ProfileTargetSiteKind::NCCCommand);
  EXPECT_EQ(siteMap->ranks.front().sites[0].engine,
            wafer::runtime::ProfileTSMEngine::RDMA);
  EXPECT_EQ(siteMap->ranks.front().sites[1].siteKind,
            wafer::runtime::ProfileTargetSiteKind::NCCCompletion);
  EXPECT_FALSE(siteMap->ranks.front().sites[1].engine);
  EXPECT_EQ(siteMap->ranks.front().sites[2].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEControl);
  EXPECT_FALSE(siteMap->ranks.front().sites[2].engine);
  EXPECT_EQ(siteMap->ranks.front().sites[3].siteKind,
            wafer::runtime::ProfileTargetSiteKind::DirectDTEWait);
  EXPECT_EQ(siteMap->ranks.front().sites[3].engine,
            wafer::runtime::ProfileTSMEngine::DirectDTE);
  EXPECT_FALSE(siteMap->ranks.front().sites.front().correlationKey.empty());
}

TEST_F(ProfileCompanionTest, MissingSiblingIsNotAnError) {
  llvm::SmallString<256> unrelated(root);
  llvm::sys::path::append(unrelated, "ordinary");
  auto loaded = wafer::runtime::loadSiblingProfileCompanionIfPresent(unrelated);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  EXPECT_FALSE(loaded->has_value());
}

TEST_F(ProfileCompanionTest, RejectsStaleManifestDigest) {
  ASSERT_NO_FATAL_FAILURE(writeCompanion(
      /*badSiteSymbol=*/false, "sha256:" + std::string(64, '0')));
  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("digest mismatch"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsDifferentRuntimeLaunchContract) {
  ASSERT_NO_FATAL_FAILURE(
      writePackage(production, /*outputBytes=*/4, /*recordBytes=*/0,
                   /*profilerName=*/"tx81_profiler_record",
                   /*resourceNamePrefix=*/"", wafer::KernelLaunchForm::Grid));
  ASSERT_NO_FATAL_FAILURE(writeCompanion());
  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(
      llvm::toString(loaded.takeError()).find("target/ABI contract differs"),
      std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsUnknownPlanField) {
  llvm::SmallString<256> planPath(companion);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileCompanionPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  ASSERT_FALSE(corrupted.empty());
  corrupted.insert(corrupted.find('{') + 1, "\n  \"unknown\": 1,");
  ASSERT_NO_FATAL_FAILURE(writeText(planPath, corrupted));
  ASSERT_NO_FATAL_FAILURE(writeActivation());

  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("unknown field"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsLegacyRecordABIWithCurrentRecordBytes) {
  llvm::SmallString<256> planPath(companion);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileCompanionPlanFileName);
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

  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("record_abi"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsVersionThreeActivation) {
  llvm::SmallString<256> activationPath(companion);
  llvm::sys::path::append(
      activationPath, wafer::runtime::kProfileCompanionActivationFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(activationPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  std::string corrupted = (*existing)->getBuffer().str();
  const std::string current = "\"schema_version\": 4";
  size_t version = corrupted.find(current);
  ASSERT_NE(version, std::string::npos);
  corrupted.replace(version, current.size(), "\"schema_version\": 3");
  ASSERT_NO_FATAL_FAILURE(writeText(activationPath, corrupted));

  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError())
                .find("schema_version is not supported"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsEngineOnCompletionSite) {
  llvm::SmallString<256> siteMapPath(companion);
  llvm::sys::path::append(siteMapPath,
                          wafer::runtime::kProfileCompanionSiteMapFileName);
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

  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("must not carry an engine"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsMissingEngineOnNCCCommandSite) {
  llvm::SmallString<256> siteMapPath(companion);
  llvm::sys::path::append(siteMapPath,
                          wafer::runtime::kProfileCompanionSiteMapFileName);
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

  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("requires its typed engine"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsSummaryCaptureInVersionFour) {
  llvm::SmallString<256> planPath(companion);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileCompanionPlanFileName);
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

  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError())
                .find("not a supported capture kind"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsMissingActivation) {
  llvm::SmallString<256> activationPath(companion);
  llvm::sys::path::append(activationPath,
                          wafer::runtime::kProfileCompanionActivationFileName);
  ASSERT_FALSE(llvm::sys::fs::remove(activationPath));
  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("activation"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsMetadataByteTamper) {
  llvm::SmallString<256> planPath(companion);
  llvm::sys::path::append(planPath,
                          wafer::runtime::kProfileCompanionPlanFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> existing =
      llvm::MemoryBuffer::getFile(planPath);
  ASSERT_TRUE(static_cast<bool>(existing));
  ASSERT_NO_FATAL_FAILURE(
      writeText(planPath, (*existing)->getBuffer().str() + " "));
  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError())
                .find("activation metadata digest mismatch"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, RejectsTargetCallOrdinalSymbolDisagreement) {
  ASSERT_NO_FATAL_FAILURE(writeCompanion(/*badSiteSymbol=*/true));
  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(llvm::toString(loaded.takeError()).find("do not agree"),
            std::string::npos);
}

TEST_F(ProfileCompanionTest, AcceptsRenamedTypedProfilerWorkspaceAndResources) {
  ASSERT_NO_FATAL_FAILURE(writeCompanion(
      /*badSiteSymbol=*/false, /*finalDigestOverride=*/{},
      /*profilerName=*/"diagnostic_name_only",
      /*resourceNamePrefix=*/"renamed_"));
  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  EXPECT_EQ(loaded->getCaptures().size(), 2u);
}

TEST_F(ProfileCompanionTest, RejectsProductionReferenceBoundToAnotherPackage) {
  llvm::SmallString<256> other(root);
  llvm::sys::path::append(other, "other-package");
  ASSERT_FALSE(llvm::sys::fs::create_directories(other));
  ASSERT_NO_FATAL_FAILURE(
      writePackage(other, /*outputBytes=*/4, /*recordBytes=*/0));
  auto loaded = wafer::runtime::loadVerifiedProfileCompanion(companion, other);
  ASSERT_FALSE(static_cast<bool>(loaded));
  EXPECT_NE(
      llvm::toString(loaded.takeError()).find("selected ordinary package"),
      std::string::npos);
}

} // namespace
