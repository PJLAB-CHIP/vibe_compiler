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

wafer::RuntimeLaunchContract makeModelLaunch() {
  return llvm::cantFail(wafer::RuntimeLaunchContract::createModel(
      wafer::ModelEntryABI::Tx81ModelBootParam,
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
    llvm::sys::path::append(modulePath, "tile_00000.so");
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

  void createTileModules(int64_t tileCount) const {
    for (int64_t tile = 1; tile < tileCount; ++tile) {
      std::string tileText = std::to_string(tile);
      llvm::SmallString<256> path(root);
      llvm::sys::path::append(path, "modules",
                              "tile_" + std::string(5 - tileText.size(), '0') +
                                  tileText + ".so");
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
    PackageManifest manifest(
        wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        makeGridLaunch(), wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    manifest.resources = {
        {ResourceId(0),
         CardResourceScope{wafer::PhysicalCardId(0)},
         PackageResourceRole::UserInput,
         0,
         "input",
         {"f32", {16}},
         64,
         256,
         PackageAccessMode::ReadOnly,
         true},
        {ResourceId(1),
         CardResourceScope{wafer::PhysicalCardId(0)},
         PackageResourceRole::Output,
         0,
         "output",
         {"f32", {16}},
         64,
         256,
         PackageAccessMode::WriteOnly,
         true},
    };
    manifest.modules = {{ModuleId(0),
                         "modules/tile_00000.so",
                         moduleDigest(),
                         wafer::kCurrentTargetModuleFormat.str(),
                         {{PackageModuleExportRole::Main, "main"}}}};
    for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
      const int64_t physicalTile = launchSlot < 2 ? 1 - launchSlot : launchSlot;
      ResourceId workspace(static_cast<uint64_t>(launchSlot) + 2);
      manifest.resources.push_back(
          {workspace,
           TileResourceScope{wafer::PhysicalCardId(0),
                             wafer::PhysicalTileId(physicalTile)},
           PackageResourceRole::Workspace,
           0,
           "default_ddr_arena",
           {"u8", {512}},
           512,
           256,
           PackageAccessMode::ReadWrite,
           false});
      manifest.entries.push_back(
          {EntryId(launchSlot), wafer::PhysicalCardId(0),
           wafer::PhysicalTileId(physicalTile), LaunchSlotId(launchSlot),
           ModuleId(0),
           {{0, ResourceId(0), PackageAccessMode::ReadOnly},
            {1, ResourceId(1), PackageAccessMode::WriteOnly},
            {2, workspace, PackageAccessMode::ReadWrite}},
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           NoTransportRequirements{}});
    }
    return manifest;
  }

  wafer::runtime::PackageManifest makeTileManifest(
      int64_t tileCount, bool permuteIdentities, bool allDirectDTE = false,
      bool lastTileDirectDTEOnly = false, bool forceClusterLaunch = false,
      bool forceGridLaunch = false) const {
    using namespace wafer::runtime;
    const bool cluster =
        forceClusterLaunch ||
        (!forceGridLaunch && (allDirectDTE || lastTileDirectDTEOnly));
    const bool shared = true;
    PackageManifest manifest(
        wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        cluster ? makeClusterLaunch() : makeGridLaunch(),
        wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = tileCount;

    auto permutedId = [&](int64_t tile, int64_t multiplier,
                          int64_t offset) -> uint64_t {
      if (!permuteIdentities)
        return static_cast<uint64_t>(tile);
      return static_cast<uint64_t>((tile * multiplier + offset) % tileCount);
    };

    manifest.resources = {{ResourceId(0),
                           CardResourceScope{wafer::PhysicalCardId(0)},
                           PackageResourceRole::UserInput,
                           0,
                           "input",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::ReadOnly,
                           true},
                          {ResourceId(1),
                           CardResourceScope{wafer::PhysicalCardId(0)},
                           PackageResourceRole::Output,
                           0,
                           "output",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::WriteOnly,
                           true}};
    uint64_t nextResource = 2;
    for (int64_t tile = 0; tile < tileCount; ++tile) {
      std::string tileText = std::to_string(tile);
      std::string moduleName =
          "tile_" + std::string(5 - tileText.size(), '0') + tileText + ".so";
      ModuleId module(0);
      EntryId entry(permutedId(tile, 5, 3));
      if (!shared || tile == 0)
        manifest.modules.push_back(
            {module, "modules/" + moduleName, moduleDigest(),
             wafer::kCurrentTargetModuleFormat.str(),
             cluster ? std::vector<
                           PackageModuleExportRecord>{{PackageModuleExportRole::
                                                           Prepare,
                                                       "prepare"},
                                                      {PackageModuleExportRole::
                                                           Main,
                                                       "main"}}
                     : std::vector<PackageModuleExportRecord>{
                           {PackageModuleExportRole::Main, "main"}}});
      std::vector<PackageABISlotBinding> slots = {
          {0, ResourceId(0), PackageAccessMode::ReadOnly},
          {1, ResourceId(1), PackageAccessMode::WriteOnly}};
      auto addResource =
          [&](PackageResourceRole role, int64_t roleIndex, llvm::StringRef name,
              llvm::StringRef dtype, std::vector<int64_t> shape, uint64_t bytes,
              uint64_t alignment, PackageAccessMode access, bool hostVisible) {
            ResourceId resource(nextResource++);
            manifest.resources.push_back(
                {resource,
                 TileResourceScope{wafer::PhysicalCardId(0),
                                   wafer::PhysicalTileId(tile)},
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
      addResource(PackageResourceRole::Workspace, 0, "default_ddr_arena", "u8",
                  {512}, 512, 256, PackageAccessMode::ReadWrite, false);

      bool directDTE =
          allDirectDTE || (lastTileDirectDTEOnly && tile == tileCount - 1);
      TransportRequirements transport = NoTransportRequirements{};
      if (directDTE) {
        ResourceId status = addResource(
            PackageResourceRole::TransportStatus, 0, "direct_dte_status", "u32",
            {1}, kDirectDTEStatusStorageBytes, kDirectDTEStatusStorageAlignment,
            PackageAccessMode::ReadWrite, false);
        transport = DirectDTETransportRequirements{
            status, kDirectDTEStatusABI.str(), true};
      }
      manifest.entries.push_back(
          {entry, wafer::PhysicalCardId(0), wafer::PhysicalTileId(tile),
           LaunchSlotId(tile), module, std::move(slots),
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           std::move(transport)});
    }

    if (permuteIdentities) {
      std::reverse(manifest.resources.begin(), manifest.resources.end());
      std::reverse(manifest.modules.begin(), manifest.modules.end());
      std::reverse(manifest.entries.begin(), manifest.entries.end());
    }
    return manifest;
  }

  wafer::runtime::PackageManifest makeModelManifest() const {
    using namespace wafer::runtime;
    PackageManifest manifest(wafer::kCurrentTargetIdentity,
                             wafer::kCurrentKernelRuntimeABI, makeModelLaunch(),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    manifest.resources = {{ResourceId(0),
                           CardResourceScope{wafer::PhysicalCardId(0)},
                           PackageResourceRole::UserInput,
                           0,
                           "input",
                           {"f32", {1}},
                           4,
                           alignof(float),
                           PackageAccessMode::ReadOnly,
                           true},
                          {ResourceId(1),
                           CardResourceScope{wafer::PhysicalCardId(0)},
                           PackageResourceRole::Output,
                           0,
                           "output",
                           {"f32", {1}},
                           4,
                           alignof(float),
                           PackageAccessMode::WriteOnly,
                           true}};
    for (int64_t tile = 0; tile < 16; ++tile) {
      std::string tileText = std::to_string(tile);
      manifest.modules.push_back({ModuleId(tile),
                                  "modules/tile_" +
                                      std::string(5 - tileText.size(), '0') +
                                      tileText + ".so",
                                  moduleDigest(),
                                  wafer::kCurrentTargetModuleFormat.str(),
                                  {{PackageModuleExportRole::Main, "main"}}});
      manifest.entries.push_back(
          {EntryId(tile),
           wafer::PhysicalCardId(0),
           wafer::PhysicalTileId(tile),
           LaunchSlotId(tile),
           ModuleId(tile),
           {{0, ResourceId(0), PackageAccessMode::ReadOnly},
            {1, ResourceId(1), PackageAccessMode::WriteOnly}},
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           NoTransportRequirements{}});
    }
    return manifest;
  }

  wafer::runtime::PackageManifest makeCardSharedTile16Manifest() const {
    using namespace wafer::runtime;
    PackageManifest manifest(wafer::kCurrentTargetIdentity,
                             wafer::kCurrentKernelRuntimeABI, makeGridLaunch(),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    manifest.resources = {{ResourceId(0),
                           CardResourceScope{wafer::PhysicalCardId(0)},
                           PackageResourceRole::UserInput,
                           0,
                           "input",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::ReadOnly,
                           true},
                          {ResourceId(1),
                           CardResourceScope{wafer::PhysicalCardId(0)},
                           PackageResourceRole::Output,
                           0,
                           "output",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::WriteOnly,
                           true}};
    manifest.modules = {{ModuleId(0),
                         "modules/tile_00000.so",
                         moduleDigest(),
                         wafer::kCurrentTargetModuleFormat.str(),
                         {{PackageModuleExportRole::Main, "main"}}}};
    for (int64_t tile = 0; tile < 16; ++tile) {
      ResourceId workspace(static_cast<uint64_t>(tile) + 2);
      manifest.resources.push_back(
          {workspace,
           TileResourceScope{wafer::PhysicalCardId(0),
                             wafer::PhysicalTileId(tile)},
           PackageResourceRole::Workspace,
           0,
           "default_ddr_arena",
           {"u8", {512}},
           512,
           256,
           PackageAccessMode::ReadWrite,
           false});
      manifest.entries.push_back(
          {EntryId(tile),
           wafer::PhysicalCardId(0),
           wafer::PhysicalTileId(tile),
           LaunchSlotId(tile),
           ModuleId(0),
           {{0, ResourceId(0), PackageAccessMode::ReadOnly},
            {1, ResourceId(1), PackageAccessMode::WriteOnly},
            {2, workspace, PackageAccessMode::ReadWrite}},
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           NoTransportRequirements{}});
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
  EXPECT_NE(canonical.find("\"schema_version\": 8"), std::string::npos);
  EXPECT_NE(canonical.find("\"card_count\": 1"), std::string::npos);
  EXPECT_NE(canonical.find("\"tile_count\": 16"), std::string::npos);
  EXPECT_NE(canonical.find("\"completion\": \"return_after_local_drain\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"identity\": \"wafer-tx81-single-card\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"runtime_abi\": \"wafer-tx81-kernel-v3\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"launch\": {"), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\": \"kernel\""), std::string::npos);
  EXPECT_NE(canonical.find("\"form\": \"grid\""), std::string::npos);
  EXPECT_NE(canonical.find("\"entry_abi\": \"tile-major-pointer-table\""),
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

TEST_F(PackageManifestTest, RejectsUnsupportedSchemaAndMissingTargetFacts) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);

  std::string unsupported = canonical;
  size_t schema = unsupported.find("\"schema_version\": 8");
  ASSERT_NE(schema, std::string::npos);
  unsupported.replace(schema, std::string("\"schema_version\": 8").size(),
                      "\"schema_version\": 999");
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> rejected =
      wafer::runtime::parseCanonicalPackageJson(unsupported, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("schema_version"),
            std::string::npos);

  std::string missingIdentity = canonical;
  size_t identity =
      missingIdentity.find("    \"identity\": \"wafer-tx81-single-card\",\n");
  ASSERT_NE(identity, std::string::npos);
  missingIdentity.erase(identity, std::string("    \"identity\": "
                                              "\"wafer-tx81-single-card\",\n")
                                      .size());
  rejected = wafer::runtime::parseCanonicalPackageJson(missingIdentity, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("missing field 'identity'"),
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

  rejectMutation("\"form\": \"grid\"", "\"form\": \"diagonal\"",
                 "kernel launch form");
  rejectMutation("\"entry_abi\": \"tile-major-pointer-table\"",
                 "\"entry_abi\": \"raw-addresses\"", "kernel entry ABI");
  rejectMutation("\"phases\": [\n        \"main\"\n      ]",
                 "\"phases\": [\n        \"prepare\",\n        \"main\"\n"
                 "      ]",
                 "incompatible");
  rejectMutation("      \"form\": \"grid\",\n", "",
                 "missing field 'form'");
  rejectMutation("\"kind\": \"kernel\",\n      \"form\": \"grid\",\n"
                 "      \"entry_abi\": \"tile-major-pointer-table\"",
                 "\"kind\": \"model\",\n      \"form\": \"grid\",\n"
                 "      \"entry_abi\": \"tx81-model-bootparam\"",
                 "unknown field");
}

TEST_F(PackageManifestTest, ModelLaunchRoundtripUsesExactConditionalFields) {
  createTileModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verified =
      wafer::runtime::verifyPackageManifest(makeModelManifest(), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const std::string canonical =
      wafer::runtime::serializeCanonicalPackageJson(*verified);
  EXPECT_NE(canonical.find("\"kind\": \"model\""), std::string::npos);
  EXPECT_NE(canonical.find("\"entry_abi\": \"tx81-model-bootparam\""),
            std::string::npos);
  EXPECT_EQ(canonical.find("\"form\":"), std::string::npos);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> parsed =
      wafer::runtime::parseCanonicalPackageJson(canonical, root);
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  EXPECT_EQ(parsed->getManifest().launch, verified->getManifest().launch);
  wafer::runtime::RuntimeEnvironment environment{
      wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
      wafer::kCurrentTargetModuleFormat};
  std::vector<wafer::runtime::RuntimeInvocationBinding> bindings =
      makeHostBindings(parsed->getManifest());
  llvm::Expected<wafer::runtime::RuntimeInvocationPlan> rejected =
      wafer::runtime::preflightNoCardRuntimeInvocation(*parsed, bindings,
                                                       environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedModelEntryABIs = {
      wafer::ModelEntryABI::Tx81ModelBootParam};
  llvm::Expected<wafer::runtime::RuntimeInvocationPlan> plan =
      wafer::runtime::preflightNoCardRuntimeInvocation(*parsed, bindings,
                                                       environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->tiles.size(), 16u);
  ASSERT_EQ(plan->tiles.front().phases.size(), 1u);
  EXPECT_EQ(plan->tiles.front().phases.front().role,
            wafer::RuntimeLaunchPhaseRole::Main);
  EXPECT_EQ(plan->tiles.front().phases.front().symbol, "main");
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
  duplicate.insert(duplicate.find("\n"), "\n  \"schema_version\": 8,");
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
  EXPECT_NE(llvm::toString(rejected.takeError()).find("target identity"),
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
  PackageManifest manifest = makeTileManifest(16, /*permuteIdentities=*/false);
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
  PackageManifest gridDirectDTE =
      makeTileManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/true,
                       /*lastTileDirectDTEOnly=*/false,
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
      makeTileManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/false,
                       /*lastTileDirectDTEOnly=*/false,
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

TEST_F(PackageManifestTest,
       WholeCardNoCardPreflightIsExactAndSideEffectFree) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::vector<RuntimeInvocationBinding> bindings = {
      {ResourceId(0), 64, 256, PackageAccessMode::ReadOnly, true},
      {ResourceId(1), 64, 256, PackageAccessMode::WriteOnly, true}};
  RuntimeEnvironment environment{wafer::kCurrentTargetIdentity,
                                 wafer::kCurrentKernelRuntimeABI,
                                 wafer::kCurrentTargetModuleFormat, 1024};
  environment.supportedKernelLaunchForms = {
      wafer::KernelLaunchForm::Grid};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::TileMajorPointerTable};
  llvm::Expected<RuntimeInvocationPlan> invocation =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(invocation))
      << llvm::toString(invocation.takeError());
  llvm::Expected<RuntimeInvocationPlan> repeated =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(repeated))
      << llvm::toString(repeated.takeError());
  EXPECT_EQ(invocation->cardCount, 1);
  EXPECT_EQ(invocation->tileCount, 16);
  ASSERT_EQ(invocation->tiles.size(), 16u);
  ASSERT_EQ(repeated->tiles.size(), 16u);
  EXPECT_EQ(invocation->tiles.front().cardId, wafer::PhysicalCardId(0));
  EXPECT_EQ(invocation->tiles.front().tileId, wafer::PhysicalTileId(1));
  EXPECT_EQ(invocation->tiles.front().launchSlot, LaunchSlotId(0));
  EXPECT_EQ(invocation->tiles.front().launchOrder,
            repeated->tiles.front().launchOrder);
  ASSERT_EQ(invocation->tiles.front().resources.size(), 3u);
  EXPECT_FALSE(
      invocation->tiles.front().resources.back().externallyBound);
  ASSERT_EQ(invocation->tiles.front().phases.size(), 1u);
  EXPECT_EQ(invocation->tiles.front().phases.front().role,
            wafer::RuntimeLaunchPhaseRole::Main);
  EXPECT_EQ(invocation->tiles.front().phases.front().symbol, "main");

  environment.moduleFormat = "elf-other";
  llvm::Expected<RuntimeInvocationPlan> incompatible =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("incompatible"),
            std::string::npos);
  environment.moduleFormat = wafer::kCurrentTargetModuleFormat.str();

  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::Cluster};
  incompatible =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedKernelLaunchForms = {
      wafer::KernelLaunchForm::Grid};
  environment.supportedKernelEntryABIs.clear();
  incompatible =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::TileMajorPointerTable};

  bindings.pop_back();
  llvm::Expected<RuntimeInvocationPlan> rejected =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing"),
            std::string::npos);

  bindings.push_back(
      {ResourceId(1), 64, 256, PackageAccessMode::WriteOnly, true});
  bindings.front().bytes = 32;
  rejected =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("does not satisfy"),
            std::string::npos);

  bindings.front().bytes = 64;
  environment.maxResourceBytes = 32;
  rejected =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("capacity"),
            std::string::npos);
}

TEST_F(PackageManifestTest,
       CardSharedProgramResourcesFeedEveryTileLaunchWithoutDuplication) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(makeCardSharedTile16Manifest(), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const PackageManifest &manifest = verified->getManifest();
  ASSERT_EQ(manifest.resources.size(), 18u);
  for (const PackageEntrypointRecord &entry : manifest.entries) {
    ASSERT_EQ(entry.slots.size(), 3u);
    EXPECT_EQ(entry.slots[0].resource, ResourceId(0));
    EXPECT_EQ(entry.slots[1].resource, ResourceId(1));
    EXPECT_EQ(entry.slots[2].resource,
              ResourceId(static_cast<uint64_t>(entry.tileId.getValue()) + 2));
  }

  RuntimeEnvironment environment{wafer::kCurrentTargetIdentity,
                                 wafer::kCurrentKernelRuntimeABI,
                                 wafer::kCurrentTargetModuleFormat, 1024};
  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::Grid};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::TileMajorPointerTable};
  std::vector<RuntimeInvocationBinding> bindings = {
      {ResourceId(0), 64, 256, PackageAccessMode::ReadOnly, true},
      {ResourceId(1), 64, 256, PackageAccessMode::WriteOnly, true}};
  llvm::Expected<RuntimeInvocationPlan> plan =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->tiles.size(), 16u);
  for (auto [tile, session] : llvm::enumerate(plan->tiles)) {
    ASSERT_EQ(session.launchOrder.size(), 3u);
    EXPECT_EQ(session.launchOrder[0], ResourceId(0));
    EXPECT_EQ(session.launchOrder[1], ResourceId(1));
    EXPECT_EQ(session.launchOrder[2], ResourceId(tile + 2));
  }
}

TEST_F(PackageManifestTest, RejectsSharingTileLocalWorkspaceAcrossEntries) {
  using namespace wafer::runtime;
  PackageManifest manifest = makeCardSharedTile16Manifest();
  for (PackageEntrypointRecord &entry : manifest.entries)
    entry.slots[2].resource = ResourceId(2);
  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("typed physical scope"),
      std::string::npos);
}

TEST_F(PackageManifestTest,
       AllTilePreflightUsesCanonicalLaunchSlotOrderAndExactDomain) {
  using namespace wafer::runtime;
  PackageManifest manifest = makeTileManifest(16, /*permuteIdentities=*/true);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  ASSERT_NE(verified->getManifest().entries.front().launchSlot,
            LaunchSlotId(0));

  std::vector<RuntimeInvocationBinding> bindings =
      makeHostBindings(verified->getManifest());
  RuntimeEnvironment environment{wafer::kCurrentTargetIdentity,
                                 wafer::kCurrentKernelRuntimeABI,
                                 wafer::kCurrentTargetModuleFormat, 1024};
  environment.supportedKernelLaunchForms = {
      wafer::KernelLaunchForm::Grid};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::TileMajorPointerTable};
  llvm::Expected<RuntimeInvocationPlan> plan =
      preflightNoCardRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(plan->cardCount, 1);
  EXPECT_EQ(plan->tileCount, 16);
  ASSERT_EQ(plan->tiles.size(), 16u);
  EXPECT_NE(plan->tiles.front().entry, EntryId(0));
  EXPECT_EQ(plan->tiles.front().module, ModuleId(0));
  for (int64_t tile = 0; tile < 16; ++tile) {
    const RuntimeSessionPlan &session = plan->tiles[tile];
    EXPECT_EQ(session.cardId, wafer::PhysicalCardId(0));
    EXPECT_EQ(session.tileId, wafer::PhysicalTileId(tile));
    EXPECT_EQ(session.launchSlot, LaunchSlotId(tile));
    ASSERT_EQ(session.resources.size(), 3u);
    ASSERT_EQ(session.launchOrder.size(), 3u);
    EXPECT_EQ(session.modulePath, "modules/tile_00000.so");
  }

  auto findResource = [&](PackageResourceRole role) {
    auto iterator = llvm::find_if(
        verified->getManifest().resources, [&](const auto &resource) {
          return std::holds_alternative<CardResourceScope>(resource.scope) &&
                 resource.role == role;
        });
    EXPECT_NE(iterator, verified->getManifest().resources.end());
    return iterator->id;
  };
  ResourceId sharedInput = findResource(PackageResourceRole::UserInput);
  auto workspace = llvm::find_if(
      verified->getManifest().resources, [&](const auto &resource) {
        const auto *scope = std::get_if<TileResourceScope>(&resource.scope);
        return scope && scope->tileId == wafer::PhysicalTileId(15) &&
               resource.role == PackageResourceRole::Workspace;
      });
  ASSERT_NE(workspace, verified->getManifest().resources.end());
  ResourceId tile15Workspace = workspace->id;

  std::vector<RuntimeInvocationBinding> missing = bindings;
  missing.erase(llvm::find_if(missing, [&](const auto &binding) {
    return binding.resource == sharedInput;
  }));
  llvm::Expected<RuntimeInvocationPlan> rejected =
      preflightNoCardRuntimeInvocation(*verified, missing, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing"),
            std::string::npos);

  std::vector<RuntimeInvocationBinding> extra = bindings;
  extra.push_back(
      {tile15Workspace, 512, 256, PackageAccessMode::ReadWrite, true});
  rejected = preflightNoCardRuntimeInvocation(*verified, extra, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("extra"),
            std::string::npos);

  std::vector<RuntimeInvocationBinding> duplicate = bindings;
  duplicate.push_back(*llvm::find_if(duplicate, [&](const auto &binding) {
    return binding.resource == sharedInput;
  }));
  rejected =
      preflightNoCardRuntimeInvocation(*verified, duplicate, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("duplicate"),
            std::string::npos);
}

TEST_F(PackageManifestTest,
       PreflightPreservesNonIdentityPhysicalTileLaunchBinding) {
  using namespace wafer::runtime;
  PackageManifest manifest = makeTileManifest(16, /*permuteIdentities=*/false);
  auto swapTile = [](wafer::PhysicalTileId tileId) {
    if (tileId == wafer::PhysicalTileId(0))
      return wafer::PhysicalTileId(1);
    if (tileId == wafer::PhysicalTileId(1))
      return wafer::PhysicalTileId(0);
    return tileId;
  };
  for (PackageEntrypointRecord &entry : manifest.entries)
    entry.tileId = swapTile(entry.tileId);
  for (PackageResourceRecord &resource : manifest.resources)
    if (auto *scope = std::get_if<TileResourceScope>(&resource.scope))
      scope->tileId = swapTile(scope->tileId);

  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  RuntimeEnvironment environment{wafer::kCurrentTargetIdentity,
                                 wafer::kCurrentKernelRuntimeABI,
                                 wafer::kCurrentTargetModuleFormat, 1024};
  environment.supportedKernelLaunchForms = {
      wafer::KernelLaunchForm::Grid};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::TileMajorPointerTable};
  llvm::Expected<RuntimeInvocationPlan> plan = preflightNoCardRuntimeInvocation(
      *verified, makeHostBindings(verified->getManifest()), environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->tiles.size(), 16u);
  EXPECT_EQ(plan->tiles[0].launchSlot, LaunchSlotId(0));
  EXPECT_EQ(plan->tiles[0].tileId, wafer::PhysicalTileId(1));
  EXPECT_EQ(plan->tiles[1].launchSlot, LaunchSlotId(1));
  EXPECT_EQ(plan->tiles[1].tileId, wafer::PhysicalTileId(0));
}

TEST_F(PackageManifestTest,
       AllTilePreflightRejectsMixedTransportAndMissingCapabilities) {
  using namespace wafer::runtime;
  RuntimeEnvironment environment{wafer::kCurrentTargetIdentity,
                                 wafer::kCurrentKernelRuntimeABI,
                                 wafer::kCurrentTargetModuleFormat, 1024};
  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::Grid,
                                            wafer::KernelLaunchForm::Cluster};
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::TileMajorPointerTable};

  PackageManifest mixed = makeTileManifest(16, /*permuteIdentities=*/true,
                                           /*allDirectDTE=*/false,
                                           /*lastTileDirectDTEOnly=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedMixed =
      verifyPackageManifest(std::move(mixed), root);
  ASSERT_FALSE(static_cast<bool>(verifiedMixed));
  EXPECT_NE(llvm::toString(verifiedMixed.takeError()).find("transport"),
            std::string::npos);

  PackageManifest unsupportedStatus =
      makeTileManifest(16, /*permuteIdentities=*/true,
                       /*allDirectDTE=*/true);
  for (PackageEntrypointRecord &entry : unsupportedStatus.entries)
    std::get<DirectDTETransportRequirements>(entry.transport).statusABI =
        "unsupported-status-abi";
  llvm::Expected<VerifiedPackageManifest> verifiedUnsupportedStatus =
      verifyPackageManifest(std::move(unsupportedStatus), root);
  ASSERT_FALSE(static_cast<bool>(verifiedUnsupportedStatus));
  EXPECT_NE(llvm::toString(verifiedUnsupportedStatus.takeError())
                .find("Direct DTE"),
            std::string::npos);

  PackageManifest direct = makeTileManifest(16, /*permuteIdentities=*/true,
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
  EXPECT_EQ(plan->tileCount, 16);
  ASSERT_EQ(plan->tiles.size(), 16u);
  ASSERT_EQ(plan->tiles.front().phases.size(), 2u);
  EXPECT_EQ(plan->tiles.front().phases[0].role,
            wafer::RuntimeLaunchPhaseRole::Prepare);
  EXPECT_EQ(plan->tiles.front().phases[0].symbol, "prepare");
  EXPECT_EQ(plan->tiles.front().phases[1].role,
            wafer::RuntimeLaunchPhaseRole::Main);
  EXPECT_EQ(plan->tiles.front().phases[1].symbol, "main");
  EXPECT_TRUE(std::holds_alternative<DirectDTETransportRequirements>(
      plan->tiles.back().transport));
}

TEST_F(PackageManifestTest,
       DirectDTETransportDoesNotSelectTheRuntimeLaunchContract) {
  using namespace wafer::runtime;
  PackageManifest manifest =
      makeTileManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/true,
                       /*lastTileDirectDTEOnly=*/false,
                       /*forceClusterLaunch=*/false,
                       /*forceGridLaunch=*/true);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const auto *kernel = verified->getManifest().launch.getKernel();
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->form, wafer::KernelLaunchForm::Grid);
  EXPECT_TRUE(std::holds_alternative<DirectDTETransportRequirements>(
      verified->getManifest().entries.front().transport));
}

} // namespace
