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
      llvm::sys::path::append(
          path, "modules",
          "rank_" + std::string(5 - rankText.size(), '0') + rankText +
              ".so");
      std::error_code error;
      llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
      ASSERT_FALSE(error);
      output << "\x7f"
                "ELFtyped-package-test";
      output.close();
      ASSERT_FALSE(output.has_error());
    }
  }

  wafer::runtime::PackageManifest makeManifest() const {
    using namespace wafer::runtime;
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(target.id, target.targetIdentity,
                             target.kernelRuntimeABI, target.moduleFormat);
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
    manifest.modules = {{ModuleId(0), 0, "modules/rank_00000.so",
                         moduleDigest(), target.moduleFormat.str()}};
    manifest.entries = {{EntryId(0),
                         0,
                         ModuleId(0),
                         "main",
                         {{0, ResourceId(0), PackageAccessMode::ReadOnly},
                          {1, ResourceId(1), PackageAccessMode::WriteOnly},
                          {2, ResourceId(2), PackageAccessMode::ReadWrite}},
                         CompletionId(0)}};
    manifest.completions = {{CompletionId(0), 0, "entry_return"}};
    return manifest;
  }

  wafer::runtime::PackageManifest
  makeRankManifest(int64_t rankCount, bool permuteIdentities,
                   bool allDirectDTE = false,
                   bool lastRankDirectDTEOnly = false) const {
    using namespace wafer::runtime;
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(target.id, target.targetIdentity,
                             target.kernelRuntimeABI, target.moduleFormat);
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
          "rank_" + std::string(5 - rankText.size(), '0') + rankText +
          ".so";
      ModuleId module(permutedId(rank, 7, 5));
      EntryId entry(permutedId(rank, 5, 3));
      CompletionId completion(permutedId(rank, 9, 1));
      manifest.modules.push_back({module, rank, "modules/" + moduleName,
                                  moduleDigest(), target.moduleFormat.str()});
      manifest.completions.push_back({completion, rank, "entry_return"});

      std::vector<PackageABISlotBinding> slots;
      auto addResource = [&](PackageResourceRole role, int64_t roleIndex,
                             llvm::StringRef name, llvm::StringRef dtype,
                             std::vector<int64_t> shape, uint64_t bytes,
                             uint64_t alignment, PackageAccessMode access,
                             bool hostVisible) {
        ResourceId resource(nextResource++);
        manifest.resources.push_back(
            {resource, rank, role, roleIndex, name.str(),
             {dtype.str(), std::move(shape)}, bytes, alignment, access,
             hostVisible});
        slots.push_back({slots.size(), resource, access});
        return resource;
      };
      addResource(PackageResourceRole::UserInput, 0, "input", "f32", {16},
                  64, 256, PackageAccessMode::ReadOnly, true);
      addResource(PackageResourceRole::Output, 0, "output", "f32", {16}, 64,
                  256, PackageAccessMode::WriteOnly, true);
      addResource(PackageResourceRole::Workspace, 0, "default_ddr_arena",
                  "u8", {512}, 512, 256, PackageAccessMode::ReadWrite,
                  false);

      bool directDTE = allDirectDTE ||
                       (lastRankDirectDTEOnly && rank == rankCount - 1);
      TransportRequirements transport = NoTransportRequirements{};
      if (directDTE) {
        ResourceId status = addResource(
            PackageResourceRole::TransportStatus, 0, "direct_dte_status",
            "u32", {1}, 4, 4, PackageAccessMode::ReadWrite, false);
        transport = DirectDTETransportRequirements{
            status, kDirectDTEStatusABI.str(), true};
      }
      manifest.entries.push_back({entry, rank, module, "main", std::move(slots),
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
  EXPECT_NE(canonical.find("\"schema_version\": 3"), std::string::npos);
  EXPECT_NE(canonical.find(
                "\"profile\": \"wafer-tx81-single-card-kernel-v1\""),
            std::string::npos);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> parsed =
      wafer::runtime::parseCanonicalPackageJson(canonical, root);
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  canonical.clear();
  EXPECT_EQ(parsed->getManifest().entries.front().symbol, "main");
  EXPECT_EQ(wafer::runtime::serializeCanonicalPackageJson(*parsed),
            wafer::runtime::serializeCanonicalPackageJson(*verified));
}

TEST_F(PackageManifestTest, RejectsLegacySchemaAndMissingTargetProfile) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);

  std::string legacy = canonical;
  size_t schema = legacy.find("\"schema_version\": 3");
  ASSERT_NE(schema, std::string::npos);
  legacy.replace(schema, std::string("\"schema_version\": 3").size(),
                 "\"schema_version\": 2");
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> rejected =
      wafer::runtime::parseCanonicalPackageJson(legacy, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("schema_version"),
            std::string::npos);

  std::string missingProfile = canonical;
  size_t profile = missingProfile.find(
      "    \"profile\": \"wafer-tx81-single-card-kernel-v1\",\n");
  ASSERT_NE(profile, std::string::npos);
  missingProfile.erase(
      profile,
      std::string("    \"profile\": "
                  "\"wafer-tx81-single-card-kernel-v1\",\n")
          .size());
  rejected =
      wafer::runtime::parseCanonicalPackageJson(missingProfile, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing field 'profile'"),
            std::string::npos);
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
  EXPECT_NE(llvm::toString(incompatibleInvocation.takeError())
                .find("incompatible"),
            std::string::npos);
  environment.moduleFormat = target.moduleFormat.str();

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
  ASSERT_NE(verified->getManifest().modules.front().logicalRank, 0);

  std::vector<RuntimeInvocationBinding> bindings =
      makeHostBindings(verified->getManifest());
  const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
      wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  RuntimeEnvironment environment{target.id, target.targetIdentity,
                                 target.kernelRuntimeABI, target.moduleFormat,
                                 1024};
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
    EXPECT_EQ(session.modulePath,
              "modules/rank_" + std::string(5 - rankText.size(), '0') +
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
  ResourceId rank15Input =
      findRankResource(15, PackageResourceRole::UserInput);
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
  extra.push_back({rank15Workspace, 512, 256, PackageAccessMode::ReadWrite,
                   true});
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

  PackageManifest mixed =
      makeRankManifest(16, /*permuteIdentities=*/true,
                       /*allDirectDTE=*/false,
                       /*lastRankDirectDTEOnly=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedMixed =
      verifyPackageManifest(std::move(mixed), root);
  ASSERT_TRUE(static_cast<bool>(verifiedMixed))
      << llvm::toString(verifiedMixed.takeError());
  std::vector<RuntimeInvocationBinding> bindings =
      makeHostBindings(verifiedMixed->getManifest());
  llvm::Expected<RuntimeInvocationPlan> rejected =
      preflightNoCardRuntimeInvocation(*verifiedMixed, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("mixed transport"),
            std::string::npos);

  PackageManifest direct =
      makeRankManifest(16, /*permuteIdentities=*/true,
                       /*allDirectDTE=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedDirect =
      verifyPackageManifest(std::move(direct), root);
  ASSERT_TRUE(static_cast<bool>(verifiedDirect))
      << llvm::toString(verifiedDirect.takeError());
  bindings = makeHostBindings(verifiedDirect->getManifest());
  rejected =
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
  EXPECT_TRUE(std::holds_alternative<DirectDTETransportRequirements>(
      plan->ranks.back().transport));
}

TEST_F(PackageManifestTest,
       DirectDTERequirementBindsInternalStatusAndChecksEnvironment) {
  using namespace wafer::runtime;
  PackageManifest manifest = makeManifest();
  manifest.resources.push_back(
      {ResourceId(3), 0, PackageResourceRole::TransportStatus, 0,
       "direct_dte_status", {"u32", {1}}, 4, 4,
       PackageAccessMode::ReadWrite, false});
  manifest.entries.front().slots.push_back(
      {3, ResourceId(3), PackageAccessMode::ReadWrite});
  manifest.entries.front().transport = DirectDTETransportRequirements{
      ResourceId(3), kDirectDTEStatusABI.str(), true};
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
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
  llvm::Expected<RuntimeSessionPlan> rejected = preflightNoCardRuntimeSession(
      *verified, EntryId(0), bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("Direct DTE"),
            std::string::npos);

  environment.supportsDirectDTE = true;
  environment.directDTEStatusABI = kDirectDTEStatusABI.str();
  environment.supportsHostWatchdog = true;
  llvm::Expected<RuntimeSessionPlan> plan = preflightNoCardRuntimeSession(
      *verified, EntryId(0), bindings, environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->resources.size(), 4u);
  EXPECT_EQ(plan->resources.back().role,
            PackageResourceRole::TransportStatus);
  EXPECT_FALSE(plan->resources.back().externallyBound);
  EXPECT_TRUE(std::holds_alternative<DirectDTETransportRequirements>(
      plan->transport));
}

} // namespace
