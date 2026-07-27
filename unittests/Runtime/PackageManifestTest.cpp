//===- PackageManifestTest.cpp - Typed package format tests --------------===//

#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <type_traits>
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

wafer::RuntimeLaunchContract makeClusterLaunch() {
  return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Cluster,
      wafer::KernelEntryABI::RankMajorPointerTableV1,
      {wafer::RuntimeLaunchPhaseRole::Prepare,
       wafer::RuntimeLaunchPhaseRole::Main}));
}

wafer::RuntimeLaunchContract makeModelLaunch() {
  return llvm::cantFail(wafer::RuntimeLaunchContract::createModel(
      wafer::ModelEntryABI::Tx81ModelBootParamV1,
      {wafer::RuntimeLaunchPhaseRole::Main}));
}

class PackageManifestTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "wafer-package-manifest-test", root));
    llvm::SmallString<256> modules(root);
    llvm::sys::path::append(modules, "modules");
    ASSERT_FALSE(llvm::sys::fs::create_directories(modules));
    modulePath = modules;
    llvm::sys::path::append(modulePath, "rank_00000.so");
    std::error_code error;
    llvm::raw_fd_ostream output(modulePath, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error);
    output << "\x7f"
              "ELFtyped-package-test";
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  std::string moduleDigest() const {
    llvm::SHA256 hasher;
    hasher.update(llvm::StringRef("\x7f"
                                  "ELFtyped-package-test"));
    return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  }

  void createRankModules(int64_t rankCount) const {
    for (int64_t rank = 1; rank < rankCount; ++rank) {
      std::string rankText = std::to_string(rank);
      llvm::SmallString<256> path(root);
      llvm::sys::path::append(path, "modules",
                              "rank_" + std::string(5 - rankText.size(), '0') +
                                  rankText + ".so");
      std::error_code error;
      llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
      ASSERT_FALSE(error);
      output << "\x7f"
                "ELFtyped-package-test";
      output.close();
      ASSERT_FALSE(output.has_error());
    }
  }

  void removeRankModulesAfterZero(int64_t rankCount) const {
    for (int64_t rank = 1; rank < rankCount; ++rank) {
      std::string rankText = std::to_string(rank);
      llvm::SmallString<256> path(root);
      llvm::sys::path::append(path, "modules",
                              "rank_" + std::string(5 - rankText.size(), '0') +
                                  rankText + ".so");
      ASSERT_FALSE(llvm::sys::fs::remove(path));
    }
  }

  wafer::runtime::PackageManifest makeManifest() const {
    using namespace wafer::runtime;
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(target.id, target.targetIdentity,
                             target.kernelRuntimeABI, makePerRankLaunch(),
                             target.moduleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = 1;
    manifest.resources = {{ResourceId(0),
                           0,
                           PackageResourceRole::UserInput,
                           0,
                           "input",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::ReadOnly,
                           true},
                          {ResourceId(1),
                           0,
                           PackageResourceRole::Output,
                           0,
                           "output",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::WriteOnly,
                           true},
                          {ResourceId(2),
                           0,
                           PackageResourceRole::Workspace,
                           0,
                           "default_ddr_arena",
                           {"u8", {512}},
                           512,
                           256,
                           PackageAccessMode::ReadWrite,
                           false}};
    manifest.modules = {{ModuleId(0),
                         "modules/rank_00000.so",
                         moduleDigest(),
                         target.moduleFormat.str(),
                         {{PackageModuleExportRole::Main, "main"}}}};
    manifest.entries = {{EntryId(0),
                         0,
                         ModuleId(0),
                         {{0, ResourceId(0), PackageAccessMode::ReadOnly},
                          {1, ResourceId(1), PackageAccessMode::WriteOnly},
                          {2, ResourceId(2), PackageAccessMode::ReadWrite}},
                         CompletionId(0)}};
    manifest.completions = {{CompletionId(0), 0, "entry_return"}};
    return manifest;
  }

  wafer::runtime::PackageManifest makeRankManifest(
      int64_t rankCount, bool permuteIdentities, bool allDirectDTE = false,
      bool lastRankDirectDTEOnly = false, bool forceClusterLaunch = false,
      bool forceGridLaunch = false) const {
    using namespace wafer::runtime;
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    const bool cluster =
        forceClusterLaunch ||
        (!forceGridLaunch && (allDirectDTE || lastRankDirectDTEOnly));
    const bool shared = cluster || forceGridLaunch;
    PackageManifest manifest(
        target.id, target.targetIdentity, target.kernelRuntimeABI,
        cluster ? makeClusterLaunch()
                : (forceGridLaunch ? makeGridLaunch() : makePerRankLaunch()),
        target.moduleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = rankCount;

    auto permutedId = [&](int64_t rank, int64_t multiplier,
                          int64_t offset) -> uint64_t {
      if (!permuteIdentities)
        return static_cast<uint64_t>(rank);
      return static_cast<uint64_t>((rank * multiplier + offset) % rankCount);
    };

    uint64_t nextResource = 0;
    for (int64_t rank = 0; rank < rankCount; ++rank) {
      std::string rankText = std::to_string(rank);
      std::string moduleName =
          "rank_" + std::string(5 - rankText.size(), '0') + rankText + ".so";
      ModuleId module(shared ? 0 : permutedId(rank, 7, 5));
      EntryId entry(permutedId(rank, 5, 3));
      CompletionId completion(permutedId(rank, 9, 1));
      if (!shared || rank == 0)
        manifest.modules.push_back(
            {module, "modules/" + moduleName, moduleDigest(),
             target.moduleFormat.str(),
             cluster ? std::vector<
                           PackageModuleExportRecord>{{PackageModuleExportRole::
                                                           Prepare,
                                                       "prepare"},
                                                      {PackageModuleExportRole::
                                                           Main,
                                                       "main"}}
                     : std::vector<PackageModuleExportRecord>{
                           {PackageModuleExportRole::Main, "main"}}});
      manifest.completions.push_back({completion, rank, "entry_return"});

      std::vector<PackageABISlotBinding> slots;
      auto addResource =
          [&](PackageResourceRole role, int64_t roleIndex, llvm::StringRef name,
              llvm::StringRef dtype, std::vector<int64_t> shape, uint64_t bytes,
              uint64_t alignment, PackageAccessMode access, bool hostVisible) {
            ResourceId resource(nextResource++);
            manifest.resources.push_back({resource,
                                          rank,
                                          role,
                                          roleIndex,
                                          name.str(),
                                          {dtype.str(), std::move(shape)},
                                          bytes,
                                          alignment,
                                          access,
                                          hostVisible});
            slots.push_back({slots.size(), resource, access});
            return resource;
          };
      addResource(PackageResourceRole::UserInput, 0, "input", "f32", {16}, 64,
                  256, PackageAccessMode::ReadOnly, true);
      addResource(PackageResourceRole::Output, 0, "output", "f32", {16}, 64,
                  256, PackageAccessMode::WriteOnly, true);
      addResource(PackageResourceRole::Workspace, 0, "default_ddr_arena", "u8",
                  {512}, 512, 256, PackageAccessMode::ReadWrite, false);

      bool directDTE =
          allDirectDTE || (lastRankDirectDTEOnly && rank == rankCount - 1);
      TransportRequirements transport = NoTransportRequirements{};
      if (directDTE) {
        ResourceId status = addResource(
            PackageResourceRole::TransportStatus, 0, "direct_dte_status", "u32",
            {1}, kDirectDTEStatusStorageBytes, kDirectDTEStatusStorageAlignment,
            PackageAccessMode::ReadWrite, false);
        transport = DirectDTETransportRequirements{
            status, kDirectDTEStatusABI.str(), true};
      }
      manifest.entries.push_back({entry, rank, module, std::move(slots),
                                  completion, std::move(transport)});
    }

    if (permuteIdentities) {
      std::reverse(manifest.resources.begin(), manifest.resources.end());
      std::reverse(manifest.modules.begin(), manifest.modules.end());
      std::reverse(manifest.entries.begin(), manifest.entries.end());
      std::reverse(manifest.completions.begin(), manifest.completions.end());
    }
    return manifest;
  }

  wafer::runtime::PackageManifest makeModelManifest() const {
    using namespace wafer::runtime;
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(target.id, target.targetIdentity,
                             target.kernelRuntimeABI, makeModelLaunch(),
                             target.moduleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = 16;
    for (int64_t rank = 0; rank < 16; ++rank) {
      const ResourceId input(static_cast<uint64_t>(rank) * 2);
      const ResourceId output(input.getValue() + 1);
      manifest.resources.push_back({input,
                                    rank,
                                    PackageResourceRole::UserInput,
                                    0,
                                    "input",
                                    {"f32", {1}},
                                    4,
                                    alignof(float),
                                    PackageAccessMode::ReadOnly,
                                    true});
      manifest.resources.push_back({output,
                                    rank,
                                    PackageResourceRole::Output,
                                    0,
                                    "output",
                                    {"f32", {1}},
                                    4,
                                    alignof(float),
                                    PackageAccessMode::WriteOnly,
                                    true});
      std::string rankText = std::to_string(rank);
      manifest.modules.push_back({ModuleId(rank),
                                  "modules/rank_" +
                                      std::string(5 - rankText.size(), '0') +
                                      rankText + ".so",
                                  moduleDigest(),
                                  target.moduleFormat.str(),
                                  {{PackageModuleExportRole::Main, "main"}}});
      manifest.entries.push_back({EntryId(rank),
                                  rank,
                                  ModuleId(rank),
                                  {{0, input, PackageAccessMode::ReadOnly},
                                   {1, output, PackageAccessMode::WriteOnly}},
                                  CompletionId(rank)});
      manifest.completions.push_back(
          {CompletionId(rank), rank, "entry_return"});
    }
    return manifest;
  }

  std::vector<wafer::runtime::RuntimeInvocationBinding>
  makeHostBindings(const wafer::runtime::PackageManifest &manifest) const {
    using namespace wafer::runtime;
    std::vector<RuntimeInvocationBinding> bindings;
    for (const PackageResourceRecord &resource : manifest.resources)
      if (resource.hostVisible)
        bindings.push_back({resource.id, resource.bytes, resource.alignment,
                            resource.access, true});
    return bindings;
  }

  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verify() const {
    return wafer::runtime::verifyPackageManifest(makeManifest(), root);
  }

  llvm::SmallString<256> root;
  llvm::SmallString<256> modulePath;
};

TEST_F(PackageManifestTest, CanonicalRoundtripOwnsTypedManifest) {
  static_assert(
      !std::is_copy_constructible_v<wafer::runtime::VerifiedPackageManifest>);
  static_assert(
      std::is_move_constructible_v<wafer::runtime::VerifiedPackageManifest>);

  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);
  EXPECT_NE(canonical.find("\"schema_version\": 6"), std::string::npos);
  EXPECT_NE(canonical.find("\"profile\": \"wafer-tx81-single-card-kernel-v1\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"launch\": {"), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\": \"kernel\""), std::string::npos);
  EXPECT_NE(canonical.find("\"form\": \"per-rank\""), std::string::npos);
  EXPECT_NE(canonical.find("\"entry_abi\": \"rank-local-pointer-block-v1\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"phases\": ["), std::string::npos);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> parsed =
      wafer::runtime::parseCanonicalPackageJson(canonical, root);
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  canonical.clear();
  ASSERT_EQ(parsed->getManifest().modules.front().exports.size(), 1u);
  EXPECT_EQ(parsed->getManifest().modules.front().exports.front().role,
            wafer::runtime::PackageModuleExportRole::Main);
  EXPECT_EQ(parsed->getManifest().modules.front().exports.front().symbol,
            "main");
  EXPECT_EQ(wafer::runtime::serializeCanonicalPackageJson(*parsed),
            wafer::runtime::serializeCanonicalPackageJson(*verified));
}

TEST_F(PackageManifestTest, RejectsLegacySchemaAndMissingTargetFacts) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);

  std::string legacy = canonical;
  size_t schema = legacy.find("\"schema_version\": 6");
  ASSERT_NE(schema, std::string::npos);
  legacy.replace(schema, std::string("\"schema_version\": 6").size(),
                 "\"schema_version\": 5");
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> rejected =
      wafer::runtime::parseCanonicalPackageJson(legacy, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("schema_version"),
            std::string::npos);

  std::string missingProfile = canonical;
  size_t profile = missingProfile.find(
      "    \"profile\": \"wafer-tx81-single-card-kernel-v1\",\n");
  ASSERT_NE(profile, std::string::npos);
  missingProfile.erase(profile,
                       std::string("    \"profile\": "
                                   "\"wafer-tx81-single-card-kernel-v1\",\n")
                           .size());
  rejected = wafer::runtime::parseCanonicalPackageJson(missingProfile, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("missing field 'profile'"),
      std::string::npos);

  const std::string launchPrefix = "    \"launch\": {\n";
  const std::string moduleFormatPrefix = "    \"module_format\":";
  size_t launchBegin = canonical.find(launchPrefix);
  size_t launchEnd = canonical.find(moduleFormatPrefix);
  ASSERT_NE(launchBegin, std::string::npos);
  ASSERT_NE(launchEnd, std::string::npos);
  ASSERT_LT(launchBegin, launchEnd);

  std::string missingLaunch = canonical;
  missingLaunch.erase(launchBegin, launchEnd - launchBegin);
  rejected = wafer::runtime::parseCanonicalPackageJson(missingLaunch, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing field 'launch'"),
            std::string::npos);

  std::string legacyLaunchABI = canonical;
  legacyLaunchABI.replace(
      launchBegin, launchEnd - launchBegin,
      "    \"launch_abi\": \"per-rank-pointer-block-v1\",\n");
  rejected = wafer::runtime::parseCanonicalPackageJson(legacyLaunchABI, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("unknown field"),
            std::string::npos);

  for (llvm::StringRef oldSpelling :
       {"per-rank-pointer-block-v1", "tx81-kernel-grid-pointer-table-v1",
        "tx81-cluster-direct-dte-prepare-main-v1", "tx81-model-bootparam-v1"}) {
    std::string oldKind = canonical;
    size_t kind = oldKind.find("\"kind\": \"kernel\"");
    ASSERT_NE(kind, std::string::npos);
    oldKind.replace(kind, std::string("\"kind\": \"kernel\"").size(),
                    ("\"kind\": \"" + oldSpelling + "\"").str());
    rejected = wafer::runtime::parseCanonicalPackageJson(oldKind, root);
    ASSERT_FALSE(static_cast<bool>(rejected)) << oldSpelling.str();
    EXPECT_NE(llvm::toString(rejected.takeError()).find("runtime launch kind"),
              std::string::npos)
        << oldSpelling.str();
  }
}

TEST_F(PackageManifestTest, RejectsMalformedTaggedLaunchContract) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);

  auto rejectMutation = [&](llvm::StringRef before, llvm::StringRef after,
                            llvm::StringRef expected) {
    std::string mutated = canonical;
    size_t position = mutated.find(before.str());
    ASSERT_NE(position, std::string::npos);
    mutated.replace(position, before.size(), after.str());
    llvm::Expected<wafer::runtime::VerifiedPackageManifest> rejected =
        wafer::runtime::parseCanonicalPackageJson(mutated, root);
    ASSERT_FALSE(static_cast<bool>(rejected));
    EXPECT_NE(llvm::toString(rejected.takeError()).find(expected.str()),
              std::string::npos);
  };

  rejectMutation("\"form\": \"per-rank\"", "\"form\": \"diagonal\"",
                 "kernel launch form");
  rejectMutation("\"entry_abi\": \"rank-local-pointer-block-v1\"",
                 "\"entry_abi\": \"raw-addresses\"", "kernel entry ABI");
  rejectMutation("\"phases\": [\n        \"main\"\n      ]",
                 "\"phases\": [\n        \"prepare\",\n        \"main\"\n"
                 "      ]",
                 "incompatible");
  rejectMutation("      \"form\": \"per-rank\",\n", "", "missing field 'form'");
  rejectMutation("\"kind\": \"kernel\",\n      \"form\": \"per-rank\",\n"
                 "      \"entry_abi\": \"rank-local-pointer-block-v1\"",
                 "\"kind\": \"model\",\n      \"form\": \"per-rank\",\n"
                 "      \"entry_abi\": \"tx81-model-bootparam-v1\"",
                 "unknown field");
}

TEST_F(PackageManifestTest, ModelLaunchRoundtripUsesExactConditionalFields) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified =
      wafer::runtime::verifyPackageManifest(makeModelManifest(), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);
  EXPECT_NE(canonical.find("\"kind\": \"model\""), std::string::npos);
  EXPECT_NE(canonical.find("\"entry_abi\": \"tx81-model-bootparam-v1\""),
            std::string::npos);
  EXPECT_EQ(canonical.find("\"form\":"), std::string::npos);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> parsed =
      wafer::runtime::parseCanonicalPackageJson(canonical, root);
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  EXPECT_EQ(parsed->getManifest().launch, verified->getManifest().launch);

  const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  wafer::runtime::RuntimeEnvironment environment{
      target.id, target.targetIdentity, target.kernelRuntimeABI,
      target.moduleFormat};
  std::vector<wafer::runtime::RuntimeInvocationBinding> bindings =
      makeHostBindings(parsed->getManifest());
  llvm::Expected<wafer::runtime::RuntimeInvocationPlan> rejected =
      wafer::runtime::preflightNoCardRuntimeInvocation(*parsed, bindings,
                                                       environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedModelEntryABIs = {
      wafer::ModelEntryABI::Tx81ModelBootParamV1};
  llvm::Expected<wafer::runtime::RuntimeInvocationPlan> plan =
      wafer::runtime::preflightNoCardRuntimeInvocation(*parsed, bindings,
                                                       environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->ranks.size(), 16u);
  ASSERT_EQ(plan->ranks.front().phases.size(), 1u);
  EXPECT_EQ(plan->ranks.front().phases.front().role,
            wafer::RuntimeLaunchPhaseRole::Main);
  EXPECT_EQ(plan->ranks.front().phases.front().symbol, "main");
}

TEST_F(PackageManifestTest, RejectsUnknownFieldsAndNonCanonicalJSON) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);
  std::string unknown = canonical;
  unknown.insert(unknown.find("\n"), "\n  \"instructions\": [],");
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> rejected =
      wafer::runtime::parseCanonicalPackageJson(unknown, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("unknown field"),
            std::string::npos);

  std::string nonCanonical = canonical;
  nonCanonical.pop_back();
  rejected = wafer::runtime::parseCanonicalPackageJson(nonCanonical, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("not canonical"),
            std::string::npos);

  std::string duplicate = canonical;
  duplicate.insert(duplicate.find("\n"), "\n  \"schema_version\": 1,");
  rejected = wafer::runtime::parseCanonicalPackageJson(duplicate, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_FALSE(llvm::toString(rejected.takeError()).empty());
}

TEST_F(PackageManifestTest, EnforcesParseLimitsBeforeAcceptance) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical = serializeCanonicalPackageJson(*verified);

  auto expectRejected = [&](const PackageParseLimits &limits,
                            llvm::StringRef expectedMessage) {
    llvm::Expected<VerifiedPackageManifest> result =
        parseCanonicalPackageJson(canonical, root, limits);
    if (result) {
      ADD_FAILURE() << "package unexpectedly passed parse limits";
      return;
    }
    std::string message = llvm::toString(result.takeError());
    if (!expectedMessage.empty())
      EXPECT_NE(message.find(expectedMessage.str()), std::string::npos)
          << message;
  };

  PackageParseLimits limits;
  limits.maxJSONBytes = canonical.size() - 1;
  expectRejected(limits, "JSON byte limit");

  limits = PackageParseLimits{};
  limits.maxRecords = 8;
  expectRejected(limits, "record limit");

  limits = PackageParseLimits{};
  limits.maxStringBytes = 3;
  expectRejected(limits, "invalid length");

  limits = PackageParseLimits{};
  limits.maxShapeRank = 0;
  expectRejected(limits, "rank limit");

  limits = PackageParseLimits{};
  limits.maxJSONNesting = 1;
  expectRejected(limits, "nesting limit");
}

TEST_F(PackageManifestTest, RejectsSlotResourceAndPayloadMismatches) {
  wafer::runtime::PackageManifest manifest = makeManifest();
  manifest.moduleFormat = "elf-other";
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> rejected =
      wafer::runtime::verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("profile mapping"),
            std::string::npos);

  manifest = makeManifest();
  manifest.entries.front().slots[1].ordinal = 2;
  rejected = wafer::runtime::verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("ABI slots"),
            std::string::npos);

  manifest = makeManifest();
  manifest.resources.front().access =
      wafer::runtime::PackageAccessMode::ReadWrite;
  rejected = wafer::runtime::verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("role/access"),
            std::string::npos);

  manifest = makeManifest();
  manifest.resources.front().role =
      static_cast<wafer::runtime::PackageResourceRole>(999);
  rejected = wafer::runtime::verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_FALSE(llvm::toString(rejected.takeError()).empty());

  manifest = makeManifest();
  manifest.modules.front().digest = "sha256:" + std::string(64, '0');
  rejected = wafer::runtime::verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("digest mismatch"),
            std::string::npos);
}

TEST_F(PackageManifestTest, RejectsEmptyKernelGridBeforeRuntimeProvider) {
  using namespace wafer::runtime;
  createRankModules(16);
  PackageManifest manifest = makeRankManifest(16, /*permuteIdentities=*/false);
  manifest.launch = makeGridLaunch();
  manifest.modules.resize(1);
  for (PackageEntrypointRecord &entry : manifest.entries)
    entry.module = manifest.modules.front().id;
  manifest.resources.clear();
  for (PackageEntrypointRecord &entry : manifest.entries)
    entry.slots.clear();

  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("at least one typed ABI slot"),
      std::string::npos);
}

TEST_F(PackageManifestTest,
       GridDirectDTEAndClusterNoTransportAreIndependentlyRepresentable) {
  using namespace wafer::runtime;
  createRankModules(16);
  ASSERT_NO_FATAL_FAILURE(removeRankModulesAfterZero(16));

  PackageManifest gridDirectDTE =
      makeRankManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/true,
                       /*lastRankDirectDTEOnly=*/false,
                       /*forceClusterLaunch=*/false,
                       /*forceGridLaunch=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedGrid =
      verifyPackageManifest(std::move(gridDirectDTE), root);
  ASSERT_TRUE(static_cast<bool>(verifiedGrid))
      << llvm::toString(verifiedGrid.takeError());
  const auto *gridKernel = verifiedGrid->getManifest().launch.getKernel();
  ASSERT_NE(gridKernel, nullptr);
  EXPECT_EQ(gridKernel->form, wafer::KernelLaunchForm::Grid);
  EXPECT_TRUE(
      llvm::all_of(verifiedGrid->getManifest().entries, [](const auto &entry) {
        return std::holds_alternative<DirectDTETransportRequirements>(
            entry.transport);
      }));

  PackageManifest clusterNoTransport =
      makeRankManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/false,
                       /*lastRankDirectDTEOnly=*/false,
                       /*forceClusterLaunch=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedCluster =
      verifyPackageManifest(std::move(clusterNoTransport), root);
  ASSERT_TRUE(static_cast<bool>(verifiedCluster))
      << llvm::toString(verifiedCluster.takeError());
  const auto *clusterKernel = verifiedCluster->getManifest().launch.getKernel();
  ASSERT_NE(clusterKernel, nullptr);
  EXPECT_EQ(clusterKernel->form, wafer::KernelLaunchForm::Cluster);
  EXPECT_TRUE(llvm::all_of(
      verifiedCluster->getManifest().entries, [](const auto &entry) {
        return std::holds_alternative<NoTransportRequirements>(entry.transport);
      }));
}

TEST_F(PackageManifestTest, NoCardPreflightIsExactAndSideEffectFree) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::vector<RuntimeInvocationBinding> bindings = {
      {ResourceId(0), 64, 256, PackageAccessMode::ReadOnly, true},
      {ResourceId(1), 64, 256, PackageAccessMode::WriteOnly, true}};
  const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  RuntimeEnvironment environment{target.id, target.targetIdentity,
                                 target.kernelRuntimeABI, target.moduleFormat,
                                 1024};
  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::PerRank};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::RankLocalPointerBlockV1};
  llvm::Expected<RuntimeSessionPlan> first = preflightNoCardRuntimeSession(
      *verified, EntryId(0), bindings, environment);
  ASSERT_TRUE(static_cast<bool>(first)) << llvm::toString(first.takeError());
  llvm::Expected<RuntimeSessionPlan> second = preflightNoCardRuntimeSession(
      *verified, EntryId(0), bindings, environment);
  ASSERT_TRUE(static_cast<bool>(second)) << llvm::toString(second.takeError());
  EXPECT_EQ(first->launchOrder, second->launchOrder);
  EXPECT_FALSE(first->executesBoard);
  ASSERT_EQ(first->resources.size(), 3u);
  EXPECT_FALSE(first->resources.back().externallyBound);

  llvm::Expected<RuntimeInvocationPlan> invocation =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(invocation))
      << llvm::toString(invocation.takeError());
  EXPECT_EQ(invocation->rankCount, 1);
  ASSERT_EQ(invocation->ranks.size(), 1u);
  EXPECT_EQ(invocation->ranks.front().logicalRank, 0);
  EXPECT_EQ(invocation->ranks.front().launchOrder, first->launchOrder);
  ASSERT_EQ(first->phases.size(), 1u);
  EXPECT_EQ(first->phases.front().role, wafer::RuntimeLaunchPhaseRole::Main);
  EXPECT_EQ(first->phases.front().symbol, "main");

  environment.moduleFormat = "elf-other";
  llvm::Expected<RuntimeSessionPlan> incompatible =
      preflightNoCardRuntimeSession(*verified, EntryId(0), bindings,
                                    environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("incompatible"),
            std::string::npos);
  llvm::Expected<RuntimeInvocationPlan> incompatibleInvocation =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(incompatibleInvocation));
  EXPECT_NE(
      llvm::toString(incompatibleInvocation.takeError()).find("incompatible"),
      std::string::npos);
  environment.moduleFormat = target.moduleFormat.str();

  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::Grid};
  incompatible = preflightNoCardRuntimeSession(*verified, EntryId(0), bindings,
                                               environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::PerRank};
  environment.supportedKernelEntryABIs.clear();
  incompatible = preflightNoCardRuntimeSession(*verified, EntryId(0), bindings,
                                               environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::RankLocalPointerBlockV1};

  bindings.pop_back();
  llvm::Expected<RuntimeSessionPlan> rejected = preflightNoCardRuntimeSession(
      *verified, EntryId(0), bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("does not satisfy"),
            std::string::npos);

  bindings.push_back(
      {ResourceId(1), 64, 256, PackageAccessMode::WriteOnly, true});
  environment.maxResourceBytes = 32;
  rejected = preflightNoCardRuntimeSession(*verified, EntryId(0), bindings,
                                           environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("capacity"),
            std::string::npos);
}

TEST_F(PackageManifestTest,
       AllRankPreflightUsesCanonicalLogicalRankOrderAndExactDomain) {
  using namespace wafer::runtime;
  createRankModules(16);
  PackageManifest manifest = makeRankManifest(16, /*permuteIdentities=*/true);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  ASSERT_NE(verified->getManifest().entries.front().logicalRank, 0);

  std::vector<RuntimeInvocationBinding> bindings =
      makeHostBindings(verified->getManifest());
  const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  RuntimeEnvironment environment{target.id, target.targetIdentity,
                                 target.kernelRuntimeABI, target.moduleFormat,
                                 1024};
  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::PerRank};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::RankLocalPointerBlockV1};
  llvm::Expected<RuntimeInvocationPlan> plan =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(plan->rankCount, 16);
  ASSERT_EQ(plan->ranks.size(), 16u);
  EXPECT_NE(plan->ranks.front().entry, EntryId(0));
  EXPECT_NE(plan->ranks.front().module, ModuleId(0));
  for (int64_t rank = 0; rank < 16; ++rank) {
    const RuntimeSessionPlan &session = plan->ranks[rank];
    EXPECT_EQ(session.logicalRank, rank);
    ASSERT_EQ(session.resources.size(), 3u);
    ASSERT_EQ(session.launchOrder.size(), 3u);
    std::string rankText = std::to_string(rank);
    EXPECT_EQ(session.modulePath, "modules/rank_" +
                                      std::string(5 - rankText.size(), '0') +
                                      rankText + ".so");
  }

  auto findRankResource = [&](int64_t rank, PackageResourceRole role) {
    auto iterator = llvm::find_if(
        verified->getManifest().resources, [&](const auto &resource) {
          return resource.logicalRank == rank && resource.role == role;
        });
    EXPECT_NE(iterator, verified->getManifest().resources.end());
    return iterator->id;
  };
  ResourceId rank15Input = findRankResource(15, PackageResourceRole::UserInput);
  ResourceId rank15Workspace =
      findRankResource(15, PackageResourceRole::Workspace);

  std::vector<RuntimeInvocationBinding> missing = bindings;
  missing.erase(llvm::find_if(missing, [&](const auto &binding) {
    return binding.resource == rank15Input;
  }));
  llvm::Expected<RuntimeInvocationPlan> rejected =
      preflightNoCardRuntimeInvocation(*verified, missing, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing"),
            std::string::npos);

  std::vector<RuntimeInvocationBinding> extra = bindings;
  extra.push_back(
      {rank15Workspace, 512, 256, PackageAccessMode::ReadWrite, true});
  rejected = preflightNoCardRuntimeInvocation(*verified, extra, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("extra"),
            std::string::npos);

  std::vector<RuntimeInvocationBinding> duplicate = bindings;
  duplicate.push_back(*llvm::find_if(duplicate, [&](const auto &binding) {
    return binding.resource == rank15Input;
  }));
  rejected =
      preflightNoCardRuntimeInvocation(*verified, duplicate, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("duplicate"),
            std::string::npos);
}

TEST_F(PackageManifestTest,
       AllRankPreflightRejectsMixedTransportAndMissingCapabilities) {
  using namespace wafer::runtime;
  createRankModules(16);
  const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  RuntimeEnvironment environment{target.id, target.targetIdentity,
                                 target.kernelRuntimeABI, target.moduleFormat,
                                 1024};
  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::PerRank,
                                            wafer::KernelLaunchForm::Cluster};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::RankLocalPointerBlockV1,
      wafer::KernelEntryABI::RankMajorPointerTableV1};

  PackageManifest mixed = makeRankManifest(16, /*permuteIdentities=*/true,
                                           /*allDirectDTE=*/false,
                                           /*lastRankDirectDTEOnly=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedMixed =
      verifyPackageManifest(std::move(mixed), root);
  ASSERT_FALSE(static_cast<bool>(verifiedMixed));
  EXPECT_NE(llvm::toString(verifiedMixed.takeError()).find("transport"),
            std::string::npos);

  PackageManifest legacy = makeRankManifest(16, /*permuteIdentities=*/true,
                                            /*allDirectDTE=*/true);
  for (PackageEntrypointRecord &entry : legacy.entries)
    std::get<DirectDTETransportRequirements>(entry.transport).statusABI =
        kDirectDTEStatusABIV1.str();
  llvm::Expected<VerifiedPackageManifest> verifiedLegacy =
      verifyPackageManifest(std::move(legacy), root);
  ASSERT_FALSE(static_cast<bool>(verifiedLegacy));
  EXPECT_NE(llvm::toString(verifiedLegacy.takeError()).find("Direct DTE"),
            std::string::npos);

  ASSERT_NO_FATAL_FAILURE(removeRankModulesAfterZero(16));

  PackageManifest direct = makeRankManifest(16, /*permuteIdentities=*/true,
                                            /*allDirectDTE=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedDirect =
      verifyPackageManifest(std::move(direct), root);
  ASSERT_TRUE(static_cast<bool>(verifiedDirect))
      << llvm::toString(verifiedDirect.takeError());
  std::vector<RuntimeInvocationBinding> bindings =
      makeHostBindings(verifiedDirect->getManifest());
  llvm::Expected<RuntimeInvocationPlan> rejected =
      preflightNoCardRuntimeInvocation(*verifiedDirect, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("Direct DTE"),
            std::string::npos);

  environment.supportsDirectDTE = true;
  environment.directDTEStatusABI = kDirectDTEStatusABI.str();
  rejected =
      preflightNoCardRuntimeInvocation(*verifiedDirect, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("Direct DTE"),
            std::string::npos);

  environment.supportsHostWatchdog = true;
  llvm::Expected<RuntimeInvocationPlan> plan =
      preflightNoCardRuntimeInvocation(*verifiedDirect, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(plan->rankCount, 16);
  ASSERT_EQ(plan->ranks.size(), 16u);
  ASSERT_EQ(plan->ranks.front().phases.size(), 2u);
  EXPECT_EQ(plan->ranks.front().phases[0].role,
            wafer::RuntimeLaunchPhaseRole::Prepare);
  EXPECT_EQ(plan->ranks.front().phases[0].symbol, "prepare");
  EXPECT_EQ(plan->ranks.front().phases[1].role,
            wafer::RuntimeLaunchPhaseRole::Main);
  EXPECT_EQ(plan->ranks.front().phases[1].symbol, "main");
  EXPECT_TRUE(std::holds_alternative<DirectDTETransportRequirements>(
      plan->ranks.back().transport));
}

TEST_F(PackageManifestTest,
       DirectDTETransportDoesNotSelectTheRuntimeLaunchContract) {
  using namespace wafer::runtime;
  PackageManifest manifest = makeManifest();
  manifest.resources.push_back({ResourceId(3),
                                0,
                                PackageResourceRole::TransportStatus,
                                0,
                                "direct_dte_status",
                                {"u32", {1}},
                                kDirectDTEStatusStorageBytes,
                                kDirectDTEStatusStorageAlignment,
                                PackageAccessMode::ReadWrite,
                                false});
  manifest.entries.front().slots.push_back(
      {3, ResourceId(3), PackageAccessMode::ReadWrite});
  manifest.entries.front().transport = DirectDTETransportRequirements{
      ResourceId(3), kDirectDTEStatusABI.str(), true};
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const auto *kernel = verified->getManifest().launch.getKernel();
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->form, wafer::KernelLaunchForm::PerRank);
  EXPECT_TRUE(std::holds_alternative<DirectDTETransportRequirements>(
      verified->getManifest().entries.front().transport));
}

} // namespace
