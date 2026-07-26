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
#include <string>
#include <utility>
#include <vector>

namespace {

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
      json.attribute("schema_version", int64_t(1));
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

  static void
  writePackage(llvm::StringRef package, uint64_t outputBytes,
               uint64_t recordBytes,
               llvm::StringRef profilerName = "tx81_profiler_record",
               llvm::StringRef resourceNamePrefix = "") {
    using namespace wafer::runtime;
    llvm::SmallString<256> modules(package);
    llvm::sys::path::append(modules, "modules");
    ASSERT_FALSE(llvm::sys::fs::create_directories(modules));

    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(
        target.id, target.targetIdentity, target.kernelRuntimeABI,
        wafer::TargetLaunchABIId::perRankPointerBlockV1(), target.moduleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = 16;
    for (int64_t rank = 0; rank < 16; ++rank) {
      std::string rankText = std::to_string(rank);
      std::string moduleName =
          "rank_" + std::string(5 - rankText.size(), '0') + rankText + ".so";
      llvm::SmallString<256> modulePath(modules);
      llvm::sys::path::append(modulePath, moduleName);
      writeText(modulePath, llvm::StringRef("\x7f"
                                            "ELFprofile-companion-test"));

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
      manifest.modules.push_back({ModuleId(rank),
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
      manifest.entries.push_back({EntryId(rank), rank, ModuleId(rank),
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
                      llvm::StringRef winnerDigestOverride = {},
                      llvm::StringRef profilerName = "tx81_profiler_record",
                      llvm::StringRef resourceNamePrefix = "") {
    std::string digest = winnerDigestOverride.empty()
                             ? manifestDigest(production)
                             : winnerDigestOverride.str();
    llvm::StringRef packageReference = "../package";
    struct CaptureFixture {
      llvm::StringRef name;
      uint64_t recordBytes;
      std::string reference;
      std::string digest;
    };
    std::vector<CaptureFixture> captures;
    for (auto [name, recordBytes] :
         std::array<std::pair<llvm::StringRef, uint64_t>, 3>{
             {{"summary", WAFER_TX81_PROFILER_MIN_BUFFER_BYTES},
              {"count", WAFER_TX81_PROFILER_MIN_BUFFER_BYTES},
              {"trace", 1024 * 1024}}}) {
      std::string reference = ("captures/production-winner/" + name).str();
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
        json.attribute("schema_version", int64_t(1));
        json.attribute("rank_count", int64_t(16));
        json.attributeArray("variants", [&] {
          json.object([&] {
            json.attribute("id", "production-winner");
            json.attribute("role", "production-winner");
            json.attribute("package_ref", packageReference);
            json.attribute("manifest_sha256", digest);
          });
          json.object([&] {
            json.attribute("id", "reserved-baseline");
            json.attribute("role", "reserved-baseline");
            json.attribute("package_ref", packageReference);
            json.attribute("manifest_sha256", digest);
            json.attribute("same_as", "production-winner");
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
        json.attribute("schema_version", int64_t(1));
        json.attribute(
            "site_basis",
            "verified-target-llvm-entry-reachable-tsm-call-preorder");
        json.attribute("correlation_basis",
                       "heuristic-target-call-signature-occurrence-v1");
        json.attribute("target_call_registry_size",
                       static_cast<int64_t>(descriptors.size()));
        json.attributeArray("variants", [&] {
          json.object([&] {
            json.attribute("variant_id", "production-winner");
            json.attributeArray("ranks", [&] {
              for (int64_t rank = 0; rank < 16; ++rank) {
                json.object([&] {
                  json.attribute("logical_rank", rank);
                  json.attributeArray("sites", [&] {
                    json.object([&] {
                      json.attribute("site_id", int64_t(0));
                      json.attribute("function_ordinal", int64_t(0));
                      json.attribute("block_ordinal", int64_t(0));
                      json.attribute("instruction_ordinal", int64_t(0));
                      json.attribute("target_call_ordinal", int64_t(0));
                      json.attribute("target_call_symbol",
                                     badSiteSymbol
                                         ? "wafer_invalid_target_call"
                                         : descriptors.front().symbol);
                      json.attribute("engine", "RDMA");
                      json.attribute("correlation_key",
                                     "registry:0:structural-occurrence:0");
                    });
                  });
                });
              }
            });
          });
          json.object([&] {
            json.attribute("variant_id", "reserved-baseline");
            json.attribute("same_as", "production-winner");
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
        json.attribute("schema_version", int64_t(1));
        json.attribute("rank_count", int64_t(16));
        json.attribute("variant_metadata", "variants.json");
        json.attribute("site_map", "site-map.json");
        json.attribute(
            "site_identity",
            "variant-rank-local-tsm-site-id-and-typed-correlation-key");
        json.attributeArray("execution_packages", [&] {
          json.object([&] {
            json.attribute("variant_id", "reserved-baseline");
            json.attribute("package_ref", packageReference);
            json.attribute("manifest_sha256", digest);
          });
          json.object([&] {
            json.attribute("variant_id", "production-winner");
            json.attribute("package_ref", packageReference);
            json.attribute("manifest_sha256", digest);
          });
        });
        json.attributeArray("capture_packages", [&] {
          for (llvm::StringRef variant : {llvm::StringRef("reserved-baseline"),
                                          llvm::StringRef("production-winner")})
            for (const CaptureFixture &capture : captures)
              json.object([&] {
                json.attribute("variant_id", variant);
                json.attribute("capture", capture.name);
                json.attribute("package_ref", capture.reference);
                json.attribute("manifest_sha256", capture.digest);
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
  EXPECT_EQ(loaded->getSchemaVersion(), 1u);
  EXPECT_EQ(loaded->getProductionManifestDigest(), manifestDigest(production));
  EXPECT_EQ(loaded->getRankCount(), 16);
  EXPECT_EQ(loaded->getVariants().size(), 2u);
  EXPECT_EQ(loaded->getCaptures().size(), 6u);
  EXPECT_EQ(loaded->getSiteMaps().size(), 2u);
  EXPECT_EQ(loaded->getSiteCount(), 16u);
  const auto *winner =
      loaded->findVariant(wafer::runtime::ProfileVariantRole::ProductionWinner);
  const auto *baseline =
      loaded->findVariant(wafer::runtime::ProfileVariantRole::ReservedBaseline);
  ASSERT_NE(winner, nullptr);
  ASSERT_NE(baseline, nullptr);
  ASSERT_TRUE(baseline->getSameAs().has_value());
  EXPECT_EQ(*baseline->getSameAs(), winner->getId());
  EXPECT_EQ(winner->getPackage().getManifest().rankCount, 16);
  const auto *trace = loaded->findCapture(
      "production-winner", wafer::runtime::ProfileCaptureKind::Trace);
  ASSERT_NE(trace, nullptr);
  EXPECT_EQ(trace->getRecordBytes(), UINT64_C(1024) * 1024);
  for (wafer::runtime::ProfileCaptureKind capture :
       {wafer::runtime::ProfileCaptureKind::Summary,
        wafer::runtime::ProfileCaptureKind::Count,
        wafer::runtime::ProfileCaptureKind::Trace}) {
    const auto *winnerCapture =
        loaded->findCapture("production-winner", capture);
    const auto *baselineCapture =
        loaded->findCapture("reserved-baseline", capture);
    ASSERT_NE(winnerCapture, nullptr);
    ASSERT_NE(baselineCapture, nullptr);
    EXPECT_EQ(baselineCapture->getPackageReference(),
              winnerCapture->getPackageReference());
    EXPECT_EQ(baselineCapture->getManifestDigest(),
              winnerCapture->getManifestDigest());
    EXPECT_EQ(baselineCapture->getPackageDirectory(),
              winnerCapture->getPackageDirectory());
  }
  const auto *siteMap = loaded->findSiteMap("production-winner");
  ASSERT_NE(siteMap, nullptr);
  ASSERT_EQ(siteMap->ranks.size(), 16u);
  ASSERT_EQ(siteMap->ranks.front().sites.size(), 1u);
  EXPECT_EQ(siteMap->ranks.front().sites.front().engine,
            wafer::runtime::ProfileTSMEngine::RDMA);
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
      /*badSiteSymbol=*/false, /*winnerDigestOverride=*/{},
      /*profilerName=*/"diagnostic_name_only",
      /*resourceNamePrefix=*/"renamed_"));
  auto loaded =
      wafer::runtime::loadVerifiedProfileCompanion(companion, production);
  ASSERT_TRUE(static_cast<bool>(loaded)) << llvm::toString(loaded.takeError());
  EXPECT_EQ(loaded->getCaptures().size(), 6u);
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
