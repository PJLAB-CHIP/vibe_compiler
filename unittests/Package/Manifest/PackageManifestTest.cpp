//===- PackageManifestTest.cpp - Typed package format tests --------------===//

#include "Wafer/Package/Manifest/PackageManifest.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::runtime;
using wafer::runtime::LaunchSlotId;

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

    llvm::SmallString<256> data(root);
    llvm::sys::path::append(data, "data");
    ASSERT_FALSE(llvm::sys::fs::create_directories(data));
    writeEmptyProgramData();
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  static std::string moduleDigest() {
    llvm::SHA256 hasher;
    hasher.update(llvm::StringRef("\x7f"
                                  "ELFtyped-package-test"));
    return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  }

  /// The default target-ready program data payload: 80 bytes 0..79.
  static std::vector<uint8_t> programDataBytes() {
    std::vector<uint8_t> result(80);
    for (size_t index = 0; index < result.size(); ++index)
      result[index] = static_cast<uint8_t>(index);
    return result;
  }

  static std::string programDataDigest() {
    llvm::SHA256 hasher;
    hasher.update(programDataBytes());
    return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  }

  static std::string emptyProgramDataDigest() {
    return "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b78"
           "52b855";
  }

  void writeProgramData(llvm::ArrayRef<uint8_t> bytes) const {
    llvm::SmallString<256> path(root);
    llvm::sys::path::append(path, "data", "program-data.bin");
    std::error_code error;
    llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error);
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  void writeEmptyProgramData() const { writeProgramData(/*bytes=*/{}); }

  void writeFullProgramData() const { writeProgramData(programDataBytes()); }

  /// The fully canonical 16-Tile package: one shared module, one input port,
  /// one output port, two TargetTensors backed by the 80-byte program-data
  /// file, and per-Tile entry-local workspace arguments.
  PackageManifest makeManifest(bool withTargetTensors = true) const {
    PackageManifest manifest(wafer::kCurrentTargetIdentity,
                             wafer::kCurrentKernelRuntimeABI, makeGridLaunch(),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    if (withTargetTensors) {
      manifest.programTensors = {
          {ProgramTensorId(0),
           ProgramTensorRole::Parameter,
           0,
           ProgramElementType::F32,
           {16},
           {16},
           {0},
           {16}},
          {ProgramTensorId(1),
           ProgramTensorRole::Constant,
           0,
           ProgramElementType::F32,
           {4},
           {4},
           {0},
           {4}},
      };
      manifest.targetTensors = {
          {TargetTensorId(0),
           ProgramTensorId(0),
           LogicalFormat::F32,
           PackageMemLayout::Tensor,
           {16},
           64,
           64,
           0},
          {TargetTensorId(1),
           ProgramTensorId(1),
           LogicalFormat::F32,
           PackageMemLayout::Tensor,
           {4},
           16,
           64,
           64},
      };
      manifest.programData = {"data/program-data.bin", 80, 64,
                              programDataDigest()};
    } else {
      manifest.programData = {"data/program-data.bin", 0, 1,
                              emptyProgramDataDigest()};
    }
    manifest.inputs = {
        {PortId(0),
         0,
         ProgramElementType::F32,
         {16},
         LogicalFormat::F32,
         PackageMemLayout::Tensor,
         {16},
         64,
         256},
    };
    manifest.outputs = {
        {PortId(0),
         0,
         ProgramElementType::F32,
         {16},
         LogicalFormat::F32,
         PackageMemLayout::Tensor,
         {16},
         64,
         256},
    };
    manifest.modules = {
        {ModuleId(0),
         "modules/tile_00000.so",
         moduleDigest(),
         wafer::kCurrentTargetModuleFormat.str(),
         {{PackageModuleExportRole::Main, "main"}}},
    };
    for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
      const int64_t tileId = launchSlot < 2 ? 1 - launchSlot : launchSlot;
      std::vector<TileEntryArgumentRecord> arguments = {
          {0, ExternalInputArgument{PortId(0)}, PackageAccessMode::ReadOnly}};
      if (withTargetTensors) {
        arguments.push_back({static_cast<uint64_t>(arguments.size()),
                             TargetTensorArgument{TargetTensorId(0)},
                             PackageAccessMode::ReadOnly});
        arguments.push_back({static_cast<uint64_t>(arguments.size()),
                             TargetTensorArgument{TargetTensorId(1)},
                             PackageAccessMode::ReadOnly});
      }
      arguments.push_back({static_cast<uint64_t>(arguments.size()),
                           ExternalOutputArgument{PortId(0)},
                           PackageAccessMode::WriteOnly});
      arguments.push_back({static_cast<uint64_t>(arguments.size()),
                           WorkspaceArgument{512, 256},
                           PackageAccessMode::ReadWrite});
      manifest.entries.push_back(
          {EntryId(launchSlot), wafer::CardId(0), wafer::TileId(tileId),
           LaunchSlotId(launchSlot), ModuleId(0), std::move(arguments),
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           NoTransportRequirements{}});
    }
    return manifest;
  }

  /// A 16-Tile package with one shared module, per-Tile workspace arguments,
  /// and optional Direct DTE transport status on every Tile or on the last
  /// Tile only. `permuteIdentities` permutes Entry ids across launch slots.
  PackageManifest makeTileManifest(int64_t tileCount, bool permuteIdentities,
                                   bool allDirectDTE = false,
                                   bool lastTileDirectDTEOnly = false,
                                   bool forceClusterLaunch = false,
                                   bool forceGridLaunch = false,
                                   bool tileRowEntryABI = false) const {
    const bool cluster =
        forceClusterLaunch ||
        (!forceGridLaunch && (allDirectDTE || lastTileDirectDTEOnly));
    RuntimeLaunchContract launch =
        tileRowEntryABI
            ? llvm::cantFail(RuntimeLaunchContract::createKernel(
                  KernelLaunchForm::Grid, KernelEntryABI::TileRowPointerTable,
                  {RuntimeLaunchPhaseRole::Main}))
            : (cluster ? makeClusterLaunch() : makeGridLaunch());
    PackageManifest manifest(wafer::kCurrentTargetIdentity,
                             wafer::kCurrentKernelRuntimeABI, std::move(launch),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = tileCount;
    manifest.programData = {"data/program-data.bin", 0, 1,
                            emptyProgramDataDigest()};
    manifest.inputs = {
        {PortId(0),
         0,
         ProgramElementType::F32,
         {16},
         LogicalFormat::F32,
         PackageMemLayout::Tensor,
         {16},
         64,
         256},
    };
    manifest.outputs = {
        {PortId(0),
         0,
         ProgramElementType::F32,
         {16},
         LogicalFormat::F32,
         PackageMemLayout::Tensor,
         {16},
         64,
         256},
    };
    manifest.modules = {
        {ModuleId(0), "modules/tile_00000.so", moduleDigest(),
         wafer::kCurrentTargetModuleFormat.str(),
         cluster
             ? std::vector<
                   PackageModuleExportRecord>{{PackageModuleExportRole::Prepare,
                                               "prepare"},
                                              {PackageModuleExportRole::Main,
                                               "main"}}
             : std::vector<
                   PackageModuleExportRecord>{{PackageModuleExportRole::Main,
                                               "main"}}},
    };
    for (int64_t tile = 0; tile < tileCount; ++tile) {
      const uint64_t entryId =
          permuteIdentities ? static_cast<uint64_t>((tile * 5 + 3) % tileCount)
                            : static_cast<uint64_t>(tile);
      std::vector<TileEntryArgumentRecord> arguments = {
          {0, ExternalInputArgument{PortId(0)}, PackageAccessMode::ReadOnly},
          {1, ExternalOutputArgument{PortId(0)}, PackageAccessMode::WriteOnly},
          {2, WorkspaceArgument{512, 256}, PackageAccessMode::ReadWrite}};
      const bool directDTE =
          allDirectDTE || (lastTileDirectDTEOnly && tile == tileCount - 1);
      TransportRequirements transport = NoTransportRequirements{};
      if (directDTE) {
        arguments.push_back(
            {3,
             TransportStatusArgument{kDirectDTEStatusABI.str(),
                                     kDirectDTEStatusStorageBytes,
                                     kDirectDTEStatusStorageAlignment},
             PackageAccessMode::ReadWrite});
        transport = DirectDTETransportRequirements{};
      }
      manifest.entries.push_back(
          {EntryId(entryId), wafer::CardId(0), wafer::TileId(tile),
           LaunchSlotId(tile), ModuleId(0), std::move(arguments),
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           std::move(transport)});
    }
    return manifest;
  }

  /// A canonical 16-Tile package whose every Tile references the same two
  /// TargetTensors and carries a distinct per-Tile workspace size.
  PackageManifest makeSharedTargetTensorManifest() const {
    PackageManifest manifest = makeManifest();
    for (PackageEntrypointRecord &entry : manifest.entries) {
      const uint64_t tile = entry.tileId.getValue();
      entry.arguments[4] = {4, WorkspaceArgument{256 + 16 * tile, 16},
                            PackageAccessMode::ReadWrite};
    }
    return manifest;
  }

  /// A 16-Tile package with profile-record arguments (and optional Direct DTE
  /// transport status / workspace) on every Tile and no program data.
  PackageManifest makeProfileManifest(bool withTransportStatus,
                                      bool withWorkspace) const {
    PackageManifest manifest(wafer::kCurrentTargetIdentity,
                             wafer::kCurrentKernelRuntimeABI, makeGridLaunch(),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    manifest.programData = {"data/program-data.bin", 0, 1,
                            emptyProgramDataDigest()};
    manifest.inputs = {
        {PortId(0),
         0,
         ProgramElementType::F32,
         {16},
         LogicalFormat::F32,
         PackageMemLayout::Tensor,
         {16},
         64,
         256},
    };
    manifest.outputs = {
        {PortId(0),
         0,
         ProgramElementType::F32,
         {16},
         LogicalFormat::F32,
         PackageMemLayout::Tensor,
         {16},
         64,
         256},
    };
    manifest.modules = {
        {ModuleId(0),
         "modules/tile_00000.so",
         moduleDigest(),
         wafer::kCurrentTargetModuleFormat.str(),
         {{PackageModuleExportRole::Main, "main"}}},
    };
    for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
      std::vector<TileEntryArgumentRecord> arguments = {
          {0, ExternalInputArgument{PortId(0)}, PackageAccessMode::ReadOnly},
          {1, ExternalOutputArgument{PortId(0)}, PackageAccessMode::WriteOnly},
          {2,
           ProfileRecordArgument{WAFER_TX81_PROFILER_RECORD_ABI,
                                 WAFER_TX81_PROFILER_MIN_BUFFER_BYTES,
                                 WAFER_TX81_PROFILER_BUFFER_ALIGNMENT},
           PackageAccessMode::ReadWrite}};
      TransportRequirements transport = NoTransportRequirements{};
      if (withTransportStatus) {
        arguments.push_back(
            {3,
             TransportStatusArgument{kDirectDTEStatusABI.str(),
                                     kDirectDTEStatusStorageBytes,
                                     kDirectDTEStatusStorageAlignment},
             PackageAccessMode::ReadWrite});
        transport = DirectDTETransportRequirements{};
      }
      if (withWorkspace)
        arguments.push_back({static_cast<uint64_t>(arguments.size()),
                             WorkspaceArgument{512, 256},
                             PackageAccessMode::ReadWrite});
      manifest.entries.push_back(
          {EntryId(launchSlot), wafer::CardId(0), wafer::TileId(launchSlot),
           LaunchSlotId(launchSlot), ModuleId(0), std::move(arguments),
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           std::move(transport)});
    }
    return manifest;
  }

  /// A grid package with no ports, tensors, or entry arguments at all.
  PackageManifest makeArgumentlessManifest() const {
    PackageManifest manifest(wafer::kCurrentTargetIdentity,
                             wafer::kCurrentKernelRuntimeABI, makeGridLaunch(),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    manifest.programData = {"data/program-data.bin", 0, 1,
                            emptyProgramDataDigest()};
    manifest.modules = {
        {ModuleId(0),
         "modules/tile_00000.so",
         moduleDigest(),
         wafer::kCurrentTargetModuleFormat.str(),
         {{PackageModuleExportRole::Main, "main"}}},
    };
    for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot)
      manifest.entries.push_back(
          {EntryId(launchSlot),
           wafer::CardId(0),
           wafer::TileId(launchSlot),
           LaunchSlotId(launchSlot),
           ModuleId(0),
           {},
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           NoTransportRequirements{}});
    return manifest;
  }

  RuntimeEnvironment makeEnvironment(uint64_t maxResourceBytes) const {
    RuntimeEnvironment environment{
        wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        wafer::kCurrentTargetModuleFormat, maxResourceBytes};
    environment.supportedKernelLaunchForms = {KernelLaunchForm::Grid,
                                              KernelLaunchForm::Cluster};
    environment.supportedKernelEntryABIs = {
        KernelEntryABI::TileMajorPointerTable,
        KernelEntryABI::TileRowPointerTable};
    return environment;
  }

  std::vector<RuntimeInvocationBinding>
  makeInputBindings(const PackageManifest &manifest) const {
    std::vector<RuntimeInvocationBinding> bindings;
    for (const ExternalPortRecord &port : manifest.inputs)
      bindings.push_back({port.id, port.bytes, port.alignment});
    return bindings;
  }

  llvm::Expected<VerifiedPackageManifest> verify() const {
    writeFullProgramData();
    return verifyPackageManifest(makeManifest(), root);
  }

  template <typename T>
  static void expectRejected(llvm::Expected<T> result,
                             llvm::StringRef expectedMessage) {
    ASSERT_FALSE(static_cast<bool>(result));
    std::string message = llvm::toString(result.takeError());
    if (!expectedMessage.empty()) {
      EXPECT_NE(message.find(expectedMessage.str()), std::string::npos)
          << message;
    }
  }

  llvm::SmallString<256> root;
  llvm::SmallString<256> modulePath;
};

TEST_F(PackageManifestTest, CanonicalRoundtripOwnsTypedManifest) {
  static_assert(
      !std::is_copy_constructible_v<wafer::runtime::VerifiedPackageManifest>);
  static_assert(
      std::is_move_constructible_v<wafer::runtime::VerifiedPackageManifest>);

  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical = serializeCanonicalPackageJson(*verified);
  EXPECT_NE(canonical.find("\"program\":{"), std::string::npos);
  EXPECT_NE(canonical.find("\"target\":{"), std::string::npos);
  EXPECT_NE(canonical.find("\"card_count\":1"), std::string::npos);
  EXPECT_NE(canonical.find("\"tile_count\":16"), std::string::npos);
  EXPECT_NE(canonical.find("\"completion\":\"return_after_local_drain\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"identity\":\"wafer-tx81-single-card\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"runtime_abi\":\"wafer-tx81-kernel\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"module_format\":\"elf-riscv64\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"launch\":{"), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\":\"kernel\""), std::string::npos);
  EXPECT_NE(canonical.find("\"form\":\"grid\""), std::string::npos);
  EXPECT_NE(canonical.find("\"entry_abi\":\"tile-major-pointer-table\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"phases\":["), std::string::npos);
  EXPECT_NE(canonical.find("\"program_data\":{"), std::string::npos);
  EXPECT_NE(canonical.find("\"relative_path\":\"data/program-data.bin\""),
            std::string::npos);
  EXPECT_NE(canonical.find("\"total_bytes\":80"), std::string::npos);
  EXPECT_NE(canonical.find("\"base_alignment\":64"), std::string::npos);
  EXPECT_NE(canonical.find("\"program_tensors\":["), std::string::npos);
  EXPECT_NE(canonical.find("\"role\":\"parameter\""), std::string::npos);
  EXPECT_NE(canonical.find("\"role\":\"constant\""), std::string::npos);
  EXPECT_NE(canonical.find("\"target_tensors\":["), std::string::npos);
  EXPECT_NE(canonical.find("\"layout\":\"tensor\""), std::string::npos);
  EXPECT_NE(canonical.find("\"file_offset\":64"), std::string::npos);
  EXPECT_NE(canonical.find("\"inputs\":["), std::string::npos);
  EXPECT_NE(canonical.find("\"outputs\":["), std::string::npos);
  EXPECT_NE(canonical.find("\"modules\":["), std::string::npos);
  EXPECT_NE(canonical.find("\"entries\":["), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\":\"external_input\""), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\":\"target_tensor\""), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\":\"external_output\""), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\":\"workspace\""), std::string::npos);
  EXPECT_NE(canonical.find("\"access\":\"read_only\""), std::string::npos);
  EXPECT_NE(canonical.find("\"access\":\"write_only\""), std::string::npos);
  EXPECT_NE(canonical.find("\"access\":\"read_write\""), std::string::npos);
  EXPECT_NE(canonical.find("\"transport\":{"), std::string::npos);
  EXPECT_NE(canonical.find("\"kind\":\"none\""), std::string::npos);

  llvm::Expected<VerifiedPackageManifest> parsed =
      parseCanonicalPackageJson(canonical, root);
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  ASSERT_EQ(parsed->getManifest().modules.front().exports.size(), 1u);
  EXPECT_EQ(parsed->getManifest().modules.front().exports.front().role,
            PackageModuleExportRole::Main);
  EXPECT_EQ(parsed->getManifest().modules.front().exports.front().symbol,
            "main");
  ASSERT_EQ(parsed->getManifest().entries.size(), 16u);
  const PackageEntrypointRecord &entry = parsed->getManifest().entries.front();
  ASSERT_EQ(entry.arguments.size(), 5u);
  EXPECT_EQ(entry.arguments[0].ordinal, 0u);
  EXPECT_TRUE(std::holds_alternative<ExternalInputArgument>(
      entry.arguments[0].reference));
  EXPECT_EQ(std::get<ExternalInputArgument>(entry.arguments[0].reference).port,
            PortId(0));
  EXPECT_TRUE(std::holds_alternative<TargetTensorArgument>(
      entry.arguments[1].reference));
  EXPECT_EQ(std::get<TargetTensorArgument>(entry.arguments[1].reference).tensor,
            TargetTensorId(0));
  EXPECT_TRUE(std::holds_alternative<TargetTensorArgument>(
      entry.arguments[2].reference));
  EXPECT_EQ(std::get<TargetTensorArgument>(entry.arguments[2].reference).tensor,
            TargetTensorId(1));
  EXPECT_TRUE(std::holds_alternative<ExternalOutputArgument>(
      entry.arguments[3].reference));
  EXPECT_TRUE(
      std::holds_alternative<WorkspaceArgument>(entry.arguments[4].reference));
  EXPECT_EQ(parsed->getManifest().programData.totalBytes, 80u);
  EXPECT_EQ(parsed->getManifest().programData.baseAlignment, 64u);
  EXPECT_EQ(parsed->getManifest().programData.digest, programDataDigest());
  EXPECT_EQ(serializeCanonicalPackageJson(*parsed),
            serializeCanonicalPackageJson(*verified));
}

TEST_F(PackageManifestTest,
       LargeSharedWorkspaceManifestFitsCanonicalByteBudget) {
  for (uint64_t resourceCount : {2048u, 4000u}) {
    SCOPED_TRACE(resourceCount);
    // Manifest-only scale witness: resource identity/entry references, not an
    // IR tensor-shape test. Each reference remains distinct in the entry ABI.
    auto manifest = makeManifest(/*withTargetTensors=*/false);
    manifest.launch = llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
        wafer::KernelLaunchForm::Grid,
        wafer::KernelEntryABI::TileRowPointerTable,
        {wafer::RuntimeLaunchPhaseRole::Main}));
    for (auto &entry : manifest.entries) {
      auto workspace = entry.arguments.back();
      entry.arguments.pop_back();
      for (uint64_t resource = 0; resource < resourceCount; ++resource)
        entry.arguments.push_back(
            {entry.arguments.size(),
             SharedWorkspaceArgument{resource, resource % 2 ? 64u : 512u, 256},
             entry.tileId == wafer::TileId(resource % 16)
                 ? PackageAccessMode::WriteOnly
             : entry.tileId == wafer::TileId((resource + 1) % 16)
                 ? PackageAccessMode::ReadOnly
                 : PackageAccessMode::None});
      workspace.ordinal = entry.arguments.size();
      entry.arguments.push_back(std::move(workspace));
    }
    auto verified = verifyPackageManifest(std::move(manifest), root);
    ASSERT_TRUE(static_cast<bool>(verified))
        << llvm::toString(verified.takeError());
    const std::string canonical = serializeCanonicalPackageJson(*verified);
    EXPECT_LT(canonical.size(), PackageParseLimits{}.maxJSONBytes);
    if (resourceCount == 4000) {
      EXPECT_GT(canonical.size(), 4u * 1024u * 1024u);
    }
    EXPECT_EQ(canonical.find('\n'), canonical.size() - 1);
    auto parsed = parseCanonicalPackageJson(canonical, root);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError());
    EXPECT_EQ(serializeCanonicalPackageJson(*parsed), canonical);
    auto plan =
        planRuntimeInvocation(*parsed, makeInputBindings(parsed->getManifest()),
                              makeEnvironment(16 * 1024 * 1024));
    ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
    ASSERT_EQ(plan->sharedWorkspaceRanges.size(), resourceCount);
    ASSERT_EQ(plan->tiles.size(), 16u);
    for (const auto &tile : plan->tiles) {
      ASSERT_EQ(tile.argumentAddresses.size(), resourceCount + 3);
      for (size_t index = 0; index < resourceCount; ++index) {
        EXPECT_EQ(tile.argumentAddresses[index + 2].base,
                  RuntimeArgumentAddressBase::Invocation);
        EXPECT_EQ(tile.argumentAddresses[index + 2].offset,
                  plan->sharedWorkspaceRanges[index].offset);
      }
    }
    PackageParseLimits limits;
    limits.maxJSONBytes = canonical.size();
    auto exact = parseCanonicalPackageJson(canonical, root, limits);
    ASSERT_TRUE(static_cast<bool>(exact)) << llvm::toString(exact.takeError());
    --limits.maxJSONBytes;
    expectRejected(parseCanonicalPackageJson(canonical, root, limits),
                   "JSON byte limit");
    limits = PackageParseLimits{};
    limits.maxRecords = 32768;
    expectRejected(parseCanonicalPackageJson(canonical, root, limits),
                   "record limit");
    expectRejected(
        parseCanonicalPackageJson(
            llvm::StringRef(canonical).take_front(canonical.size() / 2), root),
        "");
  }
}

TEST_F(PackageManifestTest, RejectsMissingTargetFacts) {
  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical = serializeCanonicalPackageJson(*verified);

  std::string missingIdentity = canonical;
  size_t identity =
      missingIdentity.find("\"identity\":\"wafer-tx81-single-card\",");
  ASSERT_NE(identity, std::string::npos);
  missingIdentity.erase(identity, std::string("\"identity\":"
                                              "\"wafer-tx81-single-card\",")
                                      .size());
  llvm::Expected<VerifiedPackageManifest> rejected =
      parseCanonicalPackageJson(missingIdentity, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("missing field 'identity'"),
      std::string::npos);

  const std::string launchPrefix = "\"launch\":{";
  const std::string cardCountPrefix = "\"card_count\":";
  size_t launchBegin = canonical.find(launchPrefix);
  size_t launchEnd = canonical.find(cardCountPrefix);
  ASSERT_NE(launchBegin, std::string::npos);
  ASSERT_NE(launchEnd, std::string::npos);
  ASSERT_LT(launchBegin, launchEnd);

  std::string missingLaunch = canonical;
  missingLaunch.erase(launchBegin, launchEnd - launchBegin);
  rejected = parseCanonicalPackageJson(missingLaunch, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing field 'launch'"),
            std::string::npos);
}

TEST_F(PackageManifestTest, RejectsMalformedTaggedLaunchContract) {
  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const std::string canonical = serializeCanonicalPackageJson(*verified);

  auto rejectMutation = [&](llvm::StringRef before, llvm::StringRef after,
                            llvm::StringRef expected) {
    std::string mutated = canonical;
    size_t position = mutated.find(before.str());
    ASSERT_NE(position, std::string::npos);
    mutated.replace(position, before.size(), after.str());
    llvm::Expected<VerifiedPackageManifest> rejected =
        parseCanonicalPackageJson(mutated, root);
    ASSERT_FALSE(static_cast<bool>(rejected));
    EXPECT_NE(llvm::toString(rejected.takeError()).find(expected.str()),
              std::string::npos);
  };

  rejectMutation("\"form\":\"grid\"", "\"form\":\"diagonal\"",
                 "kernel launch form");
  rejectMutation("\"entry_abi\":\"tile-major-pointer-table\"",
                 "\"entry_abi\":\"raw-addresses\"", "kernel entry ABI");
  rejectMutation("\"phases\":[\"main\"]", "\"phases\":[\"prepare\",\"main\"]",
                 "incompatible");
  rejectMutation("\"form\":\"grid\",", "", "missing field 'form'");
  rejectMutation("\"kind\":\"kernel\",\"form\":\"grid\","
                 "\"entry_abi\":\"tile-major-pointer-table\"",
                 "\"kind\":\"graph\",\"form\":\"grid\","
                 "\"entry_abi\":\"tile-major-pointer-table\"",
                 "must be 'kernel'");
}

TEST_F(PackageManifestTest, RejectsUnknownFieldsAndNonCanonicalJSON) {
  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical = serializeCanonicalPackageJson(*verified);
  std::string unknown = canonical;
  unknown.insert(1, "\"instructions\":[],");
  llvm::Expected<VerifiedPackageManifest> rejected =
      parseCanonicalPackageJson(unknown, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("unknown field"),
            std::string::npos);

  std::string nonCanonical = canonical;
  nonCanonical.pop_back();
  rejected = parseCanonicalPackageJson(nonCanonical, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("not canonical"),
            std::string::npos);

  std::string whitespace = canonical;
  size_t cardCount = whitespace.find("\"card_count\":1");
  ASSERT_NE(cardCount, std::string::npos);
  whitespace.replace(cardCount, std::string("\"card_count\":1").size(),
                     "\"card_count\" : 1");
  rejected = parseCanonicalPackageJson(whitespace, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("not canonical"),
            std::string::npos);

  std::string duplicate = canonical;
  duplicate.insert(duplicate.find("\"card_count\""), "\"card_count\":1,\n  ");
  rejected = parseCanonicalPackageJson(duplicate, root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_FALSE(llvm::toString(rejected.takeError()).empty());
}

TEST_F(PackageManifestTest, EnforcesParseLimitsBeforeAcceptance) {
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
    if (!expectedMessage.empty()) {
      EXPECT_NE(message.find(expectedMessage.str()), std::string::npos)
          << message;
    }
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

TEST_F(PackageManifestTest, RejectsEntryAndPayloadMismatches) {
  writeFullProgramData();
  PackageManifest manifest = makeManifest();
  manifest.moduleFormat = "elf-other";
  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "target identity");

  manifest = makeManifest();
  manifest.entries.front().arguments[1].ordinal = 2;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "dense");

  manifest = makeManifest();
  manifest.entries.front().arguments[1].access = PackageAccessMode::ReadWrite;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "access-consistent");

  manifest = makeManifest();
  manifest.modules.front().digest = "sha256:" + std::string(64, '0');
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "digest mismatch");

  manifest = makeManifest();
  manifest.targetTensors[1].fileOffset = 1;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "invalid descriptor or offset");

  manifest = makeManifest();
  manifest.targetTensors[1].bytes = 32;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "physical tensor codec");

  manifest = makeManifest();
  manifest.inputs[0].bytes = 1;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "input target descriptor byte count");

  manifest = makeManifest();
  manifest.outputs[0].logicalShape = {8};
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "element counts disagree");

  manifest = makeManifest();
  manifest.programTensors[0].sliceSizes = {8};
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "invalid identity or shape");

  manifest = makeManifest();
  manifest.programTensors[1].role = ProgramTensorRole::Parameter;
  manifest.programTensors[1].roleIndex = 0;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "role/index is duplicated");

  manifest = makeManifest();
  for (PackageEntrypointRecord &entry : manifest.entries) {
    entry.arguments.erase(entry.arguments.begin() + 2);
    for (auto [index, argument] : llvm::enumerate(entry.arguments))
      argument.ordinal = static_cast<uint64_t>(index);
  }
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "all-and-only");
}

TEST_F(PackageManifestTest, RejectsEmptyKernelGridBeforeRuntimeProvider) {
  PackageManifest manifest = makeArgumentlessManifest();
  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("at least one typed argument"),
      std::string::npos);
}

TEST_F(PackageManifestTest,
       GridDirectDTEAndClusterNoTransportAreIndependentlyRepresentable) {
  PackageManifest gridDirectDTE =
      makeTileManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/true, /*lastTileDirectDTEOnly=*/false,
                       /*forceClusterLaunch=*/false,
                       /*forceGridLaunch=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedGrid =
      verifyPackageManifest(std::move(gridDirectDTE), root);
  ASSERT_TRUE(static_cast<bool>(verifiedGrid))
      << llvm::toString(verifiedGrid.takeError());
  const auto &gridKernel = verifiedGrid->getManifest().launch.getKernel();
  EXPECT_EQ(gridKernel.form, wafer::KernelLaunchForm::Grid);
  EXPECT_TRUE(
      llvm::all_of(verifiedGrid->getManifest().entries, [](const auto &entry) {
        return std::holds_alternative<DirectDTETransportRequirements>(
            entry.transport);
      }));

  PackageManifest clusterNoTransport =
      makeTileManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/false, /*lastTileDirectDTEOnly=*/false,
                       /*forceClusterLaunch=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedCluster =
      verifyPackageManifest(std::move(clusterNoTransport), root);
  ASSERT_TRUE(static_cast<bool>(verifiedCluster))
      << llvm::toString(verifiedCluster.takeError());
  const auto &clusterKernel = verifiedCluster->getManifest().launch.getKernel();
  EXPECT_EQ(clusterKernel.form, wafer::KernelLaunchForm::Cluster);
  EXPECT_TRUE(llvm::all_of(
      verifiedCluster->getManifest().entries, [](const auto &entry) {
        return std::holds_alternative<NoTransportRequirements>(entry.transport);
      }));
}

TEST_F(PackageManifestTest, RuntimeInvocationPlanningIsExactAndSideEffectFree) {
  llvm::Expected<VerifiedPackageManifest> verified = verify();
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::vector<RuntimeInvocationBinding> bindings =
      makeInputBindings(verified->getManifest());
  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);
  llvm::Expected<RuntimeInvocationPlan> invocation =
      planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(invocation))
      << llvm::toString(invocation.takeError());
  llvm::Expected<RuntimeInvocationPlan> repeated =
      planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(repeated))
      << llvm::toString(repeated.takeError());
  EXPECT_EQ(invocation->cardCount, 1);
  EXPECT_EQ(invocation->tileCount, 16);
  ASSERT_EQ(invocation->tiles.size(), 16u);
  ASSERT_EQ(repeated->tiles.size(), 16u);
  EXPECT_EQ(invocation->programDataRequired, true);
  EXPECT_EQ(invocation->programDataBytes, 80u);
  EXPECT_EQ(invocation->programDataAlignment, 64u);
  ASSERT_EQ(invocation->targetTensorRanges.size(), 2u);
  EXPECT_EQ(invocation->targetTensorRanges[0].offset, 0u);
  EXPECT_EQ(invocation->targetTensorRanges[0].bytes, 64u);
  EXPECT_EQ(invocation->targetTensorRanges[1].offset, 64u);
  EXPECT_EQ(invocation->targetTensorRanges[1].bytes, 16u);
  ASSERT_EQ(invocation->inputRanges.size(), 1u);
  EXPECT_EQ(invocation->inputRanges[0].offset, 0u);
  EXPECT_EQ(invocation->inputRanges[0].bytes, 64u);
  ASSERT_EQ(invocation->outputRanges.size(), 1u);
  EXPECT_EQ(invocation->outputRanges[0].offset, 256u);
  EXPECT_EQ(invocation->outputRanges[0].bytes, 64u);
  EXPECT_EQ(invocation->invocationBytes, 8704u);
  EXPECT_EQ(invocation->invocationAlignment, 256u);
  EXPECT_TRUE(invocation->pointerRows.empty());
  EXPECT_EQ(invocation->tiles.front().cardId, wafer::CardId(0));
  EXPECT_EQ(invocation->tiles.front().tileId, wafer::TileId(1));
  EXPECT_EQ(invocation->tiles.front().launchSlot, LaunchSlotId(0));
  ASSERT_EQ(invocation->tiles.front().phases.size(), 1u);
  EXPECT_EQ(invocation->tiles.front().phases.front().role,
            wafer::RuntimeLaunchPhaseRole::Main);
  EXPECT_EQ(invocation->tiles.front().phases.front().symbol, "main");
  EXPECT_EQ(invocation->tiles.front().modulePath, "modules/tile_00000.so");
  ASSERT_EQ(invocation->tiles.front().argumentAddresses.size(), 5u);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[0].base,
            RuntimeArgumentAddressBase::Invocation);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[0].offset, 0u);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[1].base,
            RuntimeArgumentAddressBase::ProgramData);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[1].offset, 0u);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[2].base,
            RuntimeArgumentAddressBase::ProgramData);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[2].offset, 64u);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[3].base,
            RuntimeArgumentAddressBase::Invocation);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[3].offset, 256u);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[4].base,
            RuntimeArgumentAddressBase::Invocation);
  EXPECT_EQ(invocation->tiles.front().argumentAddresses[4].offset, 512u);
  EXPECT_EQ(invocation->tiles[15].tileId, wafer::TileId(15));
  ASSERT_TRUE(invocation->tileRanges[15].workspace.has_value());
  EXPECT_EQ(invocation->tileRanges[15].workspace->offset, 8192u);
  EXPECT_EQ(invocation->tileRanges[15].workspace->bytes, 512u);

  // Planning never touches the package directory.
  std::set<std::string> before;
  std::error_code error;
  for (llvm::sys::fs::directory_iterator iterator(root, error), end;
       iterator != end && !error; iterator.increment(error))
    before.insert(iterator->path());
  ASSERT_FALSE(error);
  llvm::Expected<RuntimeInvocationPlan> sideEffectFree =
      planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(sideEffectFree))
      << llvm::toString(sideEffectFree.takeError());
  std::set<std::string> after;
  error = std::error_code();
  for (llvm::sys::fs::directory_iterator iterator(root, error), end;
       iterator != end && !error; iterator.increment(error))
    after.insert(iterator->path());
  ASSERT_FALSE(error);
  EXPECT_EQ(before, after);

  environment.moduleFormat = "elf-other";
  llvm::Expected<RuntimeInvocationPlan> incompatible =
      planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("incompatible"),
            std::string::npos);
  environment.moduleFormat = wafer::kCurrentTargetModuleFormat.str();

  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::Cluster};
  incompatible = planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::Grid,
                                            wafer::KernelLaunchForm::Cluster};
  environment.supportedKernelEntryABIs.clear();
  incompatible = planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(incompatible));
  EXPECT_NE(llvm::toString(incompatible.takeError()).find("does not support"),
            std::string::npos);
  environment.supportedKernelEntryABIs = {
      wafer::KernelEntryABI::TileMajorPointerTable};

  std::vector<RuntimeInvocationBinding> missing;
  llvm::Expected<RuntimeInvocationPlan> rejected =
      planRuntimeInvocation(*verified, missing, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing"),
            std::string::npos);

  bindings.front().bytes = 32;
  rejected = planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("does not satisfy"),
            std::string::npos);

  bindings.front().bytes = 64;
  std::vector<RuntimeInvocationBinding> extra = bindings;
  extra.push_back({PortId(5), 64, 256});
  rejected = planRuntimeInvocation(*verified, extra, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("duplicate or unknown"),
            std::string::npos);

  std::vector<RuntimeInvocationBinding> duplicate = bindings;
  duplicate.push_back(duplicate.front());
  rejected = planRuntimeInvocation(*verified, duplicate, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("duplicate or unknown"),
            std::string::npos);

  environment.maxResourceBytes = 79;
  rejected = planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("capacity"),
            std::string::npos);

  environment.maxResourceBytes = 8703;
  rejected = planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("capacity"),
            std::string::npos);
}

TEST_F(PackageManifestTest,
       SharedWorkspaceIsAllocatedOnceAndSharedByTypedReadersAndWriter) {
  writeFullProgramData();
  PackageManifest manifest = makeManifest();
  for (PackageEntrypointRecord &entry : manifest.entries) {
    ASSERT_FALSE(entry.arguments.empty());
    TileEntryArgumentRecord workspace = entry.arguments.back();
    entry.arguments.pop_back();
    const bool writer = entry.tileId == wafer::TileId(1);
    entry.arguments.push_back(
        {static_cast<uint64_t>(entry.arguments.size()),
         SharedWorkspaceArgument{/*resource=*/0, /*bytes=*/1024,
                                 /*alignment=*/256},
         writer ? PackageAccessMode::WriteOnly : PackageAccessMode::ReadOnly});
    workspace.ordinal = entry.arguments.size();
    entry.arguments.push_back(std::move(workspace));
  }

  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  llvm::Expected<RuntimeInvocationPlan> plan = planRuntimeInvocation(
      *verified, makeInputBindings(verified->getManifest()),
      makeEnvironment(1024 * 1024));
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->sharedWorkspaceRanges.size(), 1u);
  EXPECT_EQ(plan->sharedWorkspaceRanges.front().offset, 512u);
  EXPECT_EQ(plan->sharedWorkspaceRanges.front().bytes, 1024u);
  ASSERT_EQ(plan->tiles.size(), 16u);
  for (const RuntimeSessionPlan &tile : plan->tiles) {
    ASSERT_EQ(tile.argumentAddresses.size(), 6u);
    EXPECT_EQ(tile.argumentAddresses[4].base,
              RuntimeArgumentAddressBase::Invocation);
    EXPECT_EQ(tile.argumentAddresses[4].offset, 512u);
  }

  EXPECT_TRUE(plan->zeroInitializedSharedWorkspaceRanges.empty());
  PackageManifest initialized = verified->getManifest();
  for (auto &entry : initialized.entries)
    for (auto &argument : entry.arguments)
      if (auto *workspace =
              std::get_if<SharedWorkspaceArgument>(&argument.reference))
        workspace->zeroInitialize = true;
  auto initializedPackage = verifyPackageManifest(initialized, root);
  ASSERT_TRUE(static_cast<bool>(initializedPackage))
      << llvm::toString(initializedPackage.takeError());
  auto initializedPlan = planRuntimeInvocation(
      *initializedPackage, makeInputBindings(initializedPackage->getManifest()),
      makeEnvironment(1024 * 1024));
  ASSERT_TRUE(static_cast<bool>(initializedPlan))
      << llvm::toString(initializedPlan.takeError());
  ASSERT_EQ(initializedPlan->zeroInitializedSharedWorkspaceRanges.size(), 1u);
  EXPECT_EQ(
      initializedPlan->zeroInitializedSharedWorkspaceRanges.front().offset,
      512u);
  EXPECT_EQ(initializedPlan->zeroInitializedSharedWorkspaceRanges.front().bytes,
            1024u);
  for (auto &argument : initialized.entries.front().arguments)
    if (auto *workspace =
            std::get_if<SharedWorkspaceArgument>(&argument.reference))
      workspace->zeroInitialize = false;
  auto inconsistent = verifyPackageManifest(std::move(initialized), root);
  ASSERT_FALSE(static_cast<bool>(inconsistent));
  llvm::consumeError(inconsistent.takeError());

  PackageManifest missingWriter = verified->getManifest();
  for (PackageEntrypointRecord &entry : missingWriter.entries)
    for (TileEntryArgumentRecord &argument : entry.arguments)
      if (std::holds_alternative<SharedWorkspaceArgument>(argument.reference))
        argument.access = PackageAccessMode::ReadOnly;
  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(missingWriter), root);
  EXPECT_FALSE(static_cast<bool>(rejected));
  if (!rejected)
    llvm::consumeError(rejected.takeError());
}

TEST_F(PackageManifestTest,
       SharedTargetTensorsFeedEveryTileLaunchWithoutDuplication) {
  writeFullProgramData();
  PackageManifest manifest = makeSharedTargetTensorManifest();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const PackageManifest &verifiedManifest = verified->getManifest();
  ASSERT_EQ(verifiedManifest.targetTensors.size(), 2u);
  for (const PackageEntrypointRecord &entry : verifiedManifest.entries) {
    ASSERT_EQ(entry.arguments.size(), 5u);
    EXPECT_TRUE(std::holds_alternative<TargetTensorArgument>(
        entry.arguments[1].reference));
    EXPECT_TRUE(std::holds_alternative<TargetTensorArgument>(
        entry.arguments[2].reference));
  }

  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);
  std::vector<RuntimeInvocationBinding> bindings =
      makeInputBindings(verifiedManifest);
  llvm::Expected<RuntimeInvocationPlan> plan =
      planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->tiles.size(), 16u);
  for (auto [launchSlot, session] : llvm::enumerate(plan->tiles)) {
    const int64_t tileId =
        launchSlot < 2 ? 1 - launchSlot : static_cast<int64_t>(launchSlot);
    ASSERT_EQ(session.argumentAddresses.size(), 5u);
    EXPECT_EQ(session.argumentAddresses[1].base,
              RuntimeArgumentAddressBase::ProgramData);
    EXPECT_EQ(session.argumentAddresses[1].offset, 0u);
    EXPECT_EQ(session.argumentAddresses[2].base,
              RuntimeArgumentAddressBase::ProgramData);
    EXPECT_EQ(session.argumentAddresses[2].offset, 64u);
    const RuntimePlannedRange &workspace =
        *plan->tileRanges[launchSlot].workspace;
    EXPECT_EQ(workspace.bytes, 256u + 16u * tileId);
    if (launchSlot != 0) {
      EXPECT_GE(workspace.offset,
                plan->tileRanges[launchSlot - 1].workspace->offset +
                    plan->tileRanges[launchSlot - 1].workspace->bytes);
    }
  }
  EXPECT_EQ(plan->tileRanges[0].workspace->offset, 320u);
  EXPECT_EQ(plan->tileRanges[15].workspace->offset, 5840u);
  EXPECT_EQ(plan->invocationBytes, 6336u);
}

TEST_F(PackageManifestTest, RejectsDuplicateTileLocalWorkspaceInOneEntry) {
  writeFullProgramData();
  PackageManifest manifest = makeManifest();
  for (PackageEntrypointRecord &entry : manifest.entries)
    entry.arguments.push_back(
        {5, WorkspaceArgument{64, 64}, PackageAccessMode::ReadWrite});
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);
  llvm::Expected<RuntimeInvocationPlan> rejected = planRuntimeInvocation(
      *verified, makeInputBindings(verified->getManifest()), environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("more than one workspace"),
      std::string::npos);
}

TEST_F(PackageManifestTest,
       AllTilePlanningUsesCanonicalLaunchSlotOrderAndExactDomain) {
  PackageManifest manifest = makeTileManifest(16, /*permuteIdentities=*/true);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  ASSERT_NE(verified->getManifest().entries.front().launchSlot,
            LaunchSlotId(0));

  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);
  std::vector<RuntimeInvocationBinding> bindings =
      makeInputBindings(verified->getManifest());
  llvm::Expected<RuntimeInvocationPlan> plan =
      planRuntimeInvocation(*verified, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(plan->cardCount, 1);
  EXPECT_EQ(plan->tileCount, 16);
  ASSERT_EQ(plan->tiles.size(), 16u);
  EXPECT_NE(plan->tiles.front().entry, EntryId(0));
  EXPECT_EQ(plan->tiles.front().module, ModuleId(0));
  for (int64_t tile = 0; tile < 16; ++tile) {
    const RuntimeSessionPlan &session = plan->tiles[tile];
    EXPECT_EQ(session.cardId, wafer::CardId(0));
    EXPECT_EQ(session.tileId, wafer::TileId(tile));
    EXPECT_EQ(session.launchSlot, LaunchSlotId(tile));
    EXPECT_EQ(session.entry,
              EntryId(static_cast<uint64_t>((tile * 5 + 3) % 16)));
    EXPECT_EQ(session.modulePath, "modules/tile_00000.so");
  }

  std::vector<RuntimeInvocationBinding> missing;
  llvm::Expected<RuntimeInvocationPlan> rejected =
      planRuntimeInvocation(*verified, missing, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("missing"),
            std::string::npos);

  std::vector<RuntimeInvocationBinding> extra = bindings;
  extra.push_back({PortId(1), 64, 256});
  rejected = planRuntimeInvocation(*verified, extra, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("duplicate or unknown"),
            std::string::npos);

  std::vector<RuntimeInvocationBinding> duplicate = bindings;
  duplicate.push_back(duplicate.front());
  rejected = planRuntimeInvocation(*verified, duplicate, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("duplicate or unknown"),
            std::string::npos);
}

TEST_F(PackageManifestTest, PlanningPreservesNonIdentityTileLaunchBinding) {
  PackageManifest manifest = makeTileManifest(16, /*permuteIdentities=*/false);
  auto swapTile = [](wafer::TileId tileId) {
    if (tileId == wafer::TileId(0))
      return wafer::TileId(1);
    if (tileId == wafer::TileId(1))
      return wafer::TileId(0);
    return tileId;
  };
  for (PackageEntrypointRecord &entry : manifest.entries)
    entry.tileId = swapTile(entry.tileId);

  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);
  llvm::Expected<RuntimeInvocationPlan> plan = planRuntimeInvocation(
      *verified, makeInputBindings(verified->getManifest()), environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->tiles.size(), 16u);
  EXPECT_EQ(plan->tiles[0].launchSlot, LaunchSlotId(0));
  EXPECT_EQ(plan->tiles[0].tileId, wafer::TileId(1));
  EXPECT_EQ(plan->tiles[1].launchSlot, LaunchSlotId(1));
  EXPECT_EQ(plan->tiles[1].tileId, wafer::TileId(0));
}

TEST_F(PackageManifestTest,
       AllTilePlanningRejectsMixedTransportAndMissingCapabilities) {
  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);

  PackageManifest mixed =
      makeTileManifest(16, /*permuteIdentities=*/true,
                       /*allDirectDTE=*/false, /*lastTileDirectDTEOnly=*/true);
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
  EXPECT_NE(
      llvm::toString(verifiedUnsupportedStatus.takeError()).find("Direct DTE"),
      std::string::npos);

  PackageManifest missingStatusArgument =
      makeTileManifest(16, /*permuteIdentities=*/true,
                       /*allDirectDTE=*/true);
  for (PackageEntrypointRecord &entry : missingStatusArgument.entries)
    entry.arguments.pop_back();
  llvm::Expected<VerifiedPackageManifest> verifiedMissingStatus =
      verifyPackageManifest(std::move(missingStatusArgument), root);
  ASSERT_FALSE(static_cast<bool>(verifiedMissingStatus));
  EXPECT_NE(
      llvm::toString(verifiedMissingStatus.takeError()).find("Direct DTE"),
      std::string::npos);

  PackageManifest orphanStatus =
      makeTileManifest(16, /*permuteIdentities=*/true,
                       /*allDirectDTE=*/true);
  for (PackageEntrypointRecord &entry : orphanStatus.entries)
    entry.transport = NoTransportRequirements{};
  llvm::Expected<VerifiedPackageManifest> verifiedOrphanStatus =
      verifyPackageManifest(std::move(orphanStatus), root);
  ASSERT_FALSE(static_cast<bool>(verifiedOrphanStatus));
  EXPECT_NE(llvm::toString(verifiedOrphanStatus.takeError())
                .find("transport status exists"),
            std::string::npos);

  PackageManifest direct = makeTileManifest(16, /*permuteIdentities=*/true,
                                            /*allDirectDTE=*/true);
  llvm::Expected<VerifiedPackageManifest> verifiedDirect =
      verifyPackageManifest(std::move(direct), root);
  ASSERT_TRUE(static_cast<bool>(verifiedDirect))
      << llvm::toString(verifiedDirect.takeError());
  std::vector<RuntimeInvocationBinding> bindings =
      makeInputBindings(verifiedDirect->getManifest());
  llvm::Expected<RuntimeInvocationPlan> rejected =
      planRuntimeInvocation(*verifiedDirect, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("Direct DTE"),
            std::string::npos);

  environment.supportsDirectDTE = true;
  environment.directDTEStatusABI = kDirectDTEStatusABI.str();
  rejected = planRuntimeInvocation(*verifiedDirect, bindings, environment);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("Direct DTE"),
            std::string::npos);

  environment.supportsHostWatchdog = true;
  llvm::Expected<RuntimeInvocationPlan> plan =
      planRuntimeInvocation(*verifiedDirect, bindings, environment);
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
  PackageManifest manifest =
      makeTileManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/true, /*lastTileDirectDTEOnly=*/false,
                       /*forceClusterLaunch=*/false,
                       /*forceGridLaunch=*/true);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const auto &kernel = verified->getManifest().launch.getKernel();
  EXPECT_EQ(kernel.form, wafer::KernelLaunchForm::Grid);
  EXPECT_TRUE(std::holds_alternative<DirectDTETransportRequirements>(
      verified->getManifest().entries.front().transport));
}

TEST_F(PackageManifestTest, EmptyProgramDataRoundtripAndDigest) {
  PackageManifest manifest = makeManifest(/*withTargetTensors=*/false);
  writeEmptyProgramData();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  std::string canonical = serializeCanonicalPackageJson(*verified);
  EXPECT_NE(canonical.find("\"total_bytes\":0"), std::string::npos);
  EXPECT_NE(canonical.find("\"base_alignment\":1"), std::string::npos);
  EXPECT_NE(canonical.find("\"digest\":\"sha256:e3b0c44298fc1c149afbf4c8996"
                           "fb92427ae41e4649b934ca495991b7852b855\""),
            std::string::npos);
  llvm::Expected<VerifiedPackageManifest> parsed =
      parseCanonicalPackageJson(canonical, root);
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  EXPECT_EQ(serializeCanonicalPackageJson(*parsed), canonical);

  manifest = makeManifest(/*withTargetTensors=*/false);
  manifest.programData.totalBytes = 64;
  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "zero bytes and unit alignment");

  manifest = makeManifest(/*withTargetTensors=*/false);
  writeProgramData(/*bytes=*/{1});
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "digest mismatch");

  writeEmptyProgramData();
}

TEST_F(PackageManifestTest, ProgramDataByteCountAndTargetTensorRanges) {
  writeFullProgramData();
  PackageManifest manifest = makeManifest();
  manifest.programData.totalBytes = 81;
  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "byte count does not match the file");

  manifest = makeManifest();
  manifest.targetTensors[1].bytes = 32;
  manifest.programData.totalBytes = 80;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "physical tensor codec");

  manifest = makeManifest();
  manifest.targetTensors[0].fileOffset = 4;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "invalid descriptor or offset");
}

TEST_F(PackageManifestTest, TileRowPointerTablePlanningIsExact) {
  PackageManifest manifest =
      makeTileManifest(16, /*permuteIdentities=*/false,
                       /*allDirectDTE=*/false, /*lastTileDirectDTEOnly=*/false,
                       /*forceClusterLaunch=*/false,
                       /*forceGridLaunch=*/true, /*tileRowEntryABI=*/true);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  EXPECT_EQ(verified->getManifest().launch.getKernel().entryABI,
            wafer::KernelEntryABI::TileRowPointerTable);

  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);
  llvm::Expected<RuntimeInvocationPlan> plan = planRuntimeInvocation(
      *verified, makeInputBindings(verified->getManifest()), environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->pointerRows.size(), 16u);
  EXPECT_EQ(plan->pointerRows[0].offset, 8704u);
  EXPECT_EQ(plan->pointerRows[0].bytes, 24u);
  EXPECT_EQ(plan->pointerRows[15].offset, 9064u);
  EXPECT_EQ(plan->pointerRows[15].bytes, 24u);
  EXPECT_EQ(plan->invocationBytes, 9088u);
  EXPECT_EQ(plan->invocationAlignment, 256u);
  for (const RuntimePlannedRange &row : plan->pointerRows) {
    EXPECT_EQ(row.bytes % alignof(uint64_t), 0u);
    EXPECT_GE(row.offset, plan->outputRanges.front().offset +
                              plan->outputRanges.front().bytes);
  }
}

TEST_F(PackageManifestTest, ProfileRecordArgumentVerificationIsExact) {
  PackageManifest manifest = makeProfileManifest(/*withTransportStatus=*/false,
                                                 /*withWorkspace=*/false);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());
  const PackageEntrypointRecord &entry =
      verified->getManifest().entries.front();
  ASSERT_EQ(entry.arguments.size(), 3u);
  ASSERT_TRUE(std::holds_alternative<ProfileRecordArgument>(
      entry.arguments[2].reference));
  const ProfileRecordArgument &profile =
      std::get<ProfileRecordArgument>(entry.arguments[2].reference);
  EXPECT_EQ(profile.recordABI, WAFER_TX81_PROFILER_RECORD_ABI);
  EXPECT_EQ(profile.bytes, WAFER_TX81_PROFILER_MIN_BUFFER_BYTES);
  EXPECT_EQ(profile.alignment, WAFER_TX81_PROFILER_BUFFER_ALIGNMENT);

  manifest = makeProfileManifest(/*withTransportStatus=*/false,
                                 /*withWorkspace=*/false);
  std::get<ProfileRecordArgument>(manifest.entries[0].arguments[2].reference)
      .bytes = 100;
  llvm::Expected<VerifiedPackageManifest> rejected =
      verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "profile record requirement is invalid");

  manifest = makeProfileManifest(/*withTransportStatus=*/false,
                                 /*withWorkspace=*/false);
  std::get<ProfileRecordArgument>(manifest.entries[0].arguments[2].reference)
      .alignment = 32;
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "profile record requirement is invalid");

  manifest = makeProfileManifest(/*withTransportStatus=*/false,
                                 /*withWorkspace=*/false);
  std::get<ProfileRecordArgument>(manifest.entries[0].arguments[2].reference)
      .recordABI = "unsupported-profile-record-abi";
  rejected = verifyPackageManifest(std::move(manifest), root);
  expectRejected(std::move(rejected), "profile record requirement is invalid");
}

TEST_F(PackageManifestTest,
       InvocationPlanningPacksProfileAndTransportStatusRanges) {
  PackageManifest manifest = makeProfileManifest(/*withTransportStatus=*/true,
                                                 /*withWorkspace=*/true);
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(verified))
      << llvm::toString(verified.takeError());

  RuntimeEnvironment environment = makeEnvironment(1024 * 1024);
  environment.supportsDirectDTE = true;
  environment.directDTEStatusABI = kDirectDTEStatusABI.str();
  environment.supportsHostWatchdog = true;
  llvm::Expected<RuntimeInvocationPlan> plan = planRuntimeInvocation(
      *verified, makeInputBindings(verified->getManifest()), environment);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  ASSERT_EQ(plan->tileRanges.size(), 16u);
  const RuntimeEntryLocalRanges &first = plan->tileRanges[0];
  ASSERT_TRUE(first.profileRecord.has_value());
  ASSERT_TRUE(first.transportStatus.has_value());
  ASSERT_TRUE(first.workspace.has_value());
  EXPECT_EQ(first.profileRecord->offset, 320u);
  EXPECT_EQ(first.profileRecord->bytes, WAFER_TX81_PROFILER_MIN_BUFFER_BYTES);
  EXPECT_EQ(first.transportStatus->offset, 1152u);
  EXPECT_EQ(first.transportStatus->bytes, kDirectDTEStatusStorageBytes);
  EXPECT_EQ(first.workspace->offset, 1280u);
  EXPECT_EQ(first.workspace->bytes, 512u);
  EXPECT_EQ(first.profileRecord->offset + first.profileRecord->bytes,
            first.transportStatus->offset);
  // The 512-byte-aligned workspace is not contiguous with the 64-byte status
  // slot; it is aligned up from the packed tail.
  EXPECT_GE(first.workspace->offset,
            first.transportStatus->offset + first.transportStatus->bytes);
  EXPECT_EQ(first.workspace->offset, 1280u);
  EXPECT_EQ(plan->invocationAlignment, 256u);

  const RuntimeSessionPlan &session = plan->tiles[0];
  ASSERT_EQ(session.argumentAddresses.size(), 5u);
  EXPECT_EQ(session.argumentAddresses[2].base,
            RuntimeArgumentAddressBase::Invocation);
  EXPECT_EQ(session.argumentAddresses[2].offset, first.profileRecord->offset);
  EXPECT_EQ(session.argumentAddresses[3].base,
            RuntimeArgumentAddressBase::Invocation);
  EXPECT_EQ(session.argumentAddresses[3].offset, first.transportStatus->offset);
  EXPECT_EQ(session.argumentAddresses[4].base,
            RuntimeArgumentAddressBase::Invocation);
  EXPECT_EQ(session.argumentAddresses[4].offset, first.workspace->offset);
}

} // namespace

// The strict program-data verification rebuilds the canonical non-overlap
// placement with the writer's stable tie-break. Each rejection path below
// mutates exactly one canonical fact and asserts the typed rejection.

TEST_F(PackageManifestTest, RejectsNonCanonicalTargetTensorIdOrder) {
  PackageManifest manifest = makeManifest();
  std::swap(manifest.targetTensors[0].id, manifest.targetTensors[1].id);
  writeFullProgramData();
  expectRejected(verifyPackageManifest(std::move(manifest), root),
                 "ids do not follow the canonical order");
}

TEST_F(PackageManifestTest, RejectsNonCanonicalTargetTensorOffset) {
  PackageManifest manifest = makeManifest();
  manifest.programTensors = {
      {ProgramTensorId(0),
       ProgramTensorRole::Parameter,
       0,
       ProgramElementType::F32,
       {4},
       {4},
       {0},
       {4}},
      {ProgramTensorId(1),
       ProgramTensorRole::Constant,
       0,
       ProgramElementType::F32,
       {4},
       {4},
       {0},
       {4}},
  };
  manifest.targetTensors = {
      {TargetTensorId(0),
       ProgramTensorId(0),
       LogicalFormat::F32,
       PackageMemLayout::Tensor,
       {4},
       16,
       16,
       0},
      // 48 is aligned but the canonical aligned placement from cursor 16 is
      // 16; the unexplained gap is rejected.
      {TargetTensorId(1),
       ProgramTensorId(1),
       LogicalFormat::F32,
       PackageMemLayout::Tensor,
       {4},
       16,
       16,
       48},
  };
  manifest.programData = {"data/program-data.bin", 80, 16, programDataDigest()};
  writeFullProgramData();
  expectRejected(verifyPackageManifest(std::move(manifest), root),
                 "offset is not canonical");
}

TEST_F(PackageManifestTest, RejectsNonCanonicalProgramDataBaseAlignment) {
  PackageManifest manifest = makeManifest();
  manifest.programData.baseAlignment = 32;
  writeFullProgramData();
  expectRejected(verifyPackageManifest(std::move(manifest), root),
                 "base alignment is not canonical");
}

TEST_F(PackageManifestTest, RejectsNonZeroPaddingBetweenCanonicalTensors) {
  PackageManifest manifest = makeManifest();
  manifest.programTensors = {
      {ProgramTensorId(0),
       ProgramTensorRole::Parameter,
       0,
       ProgramElementType::F32,
       {4},
       {4},
       {0},
       {4}},
      {ProgramTensorId(1),
       ProgramTensorRole::Constant,
       0,
       ProgramElementType::F32,
       {4},
       {4},
       {0},
       {4}},
  };
  manifest.targetTensors = {
      {TargetTensorId(0),
       ProgramTensorId(0),
       LogicalFormat::F32,
       PackageMemLayout::Tensor,
       {4},
       16,
       16,
       0},
      {TargetTensorId(1),
       ProgramTensorId(1),
       LogicalFormat::F32,
       PackageMemLayout::Tensor,
       {4},
       16,
       32,
       32},
  };
  std::vector<uint8_t> bytes(48, UINT8_C(0xAB));
  llvm::SHA256 hasher;
  hasher.update(bytes);
  manifest.programData = {"data/program-data.bin", 48, 32,
                          "sha256:" + llvm::toHex(hasher.final(),
                                                  /*LowerCase=*/true)};
  writeProgramData(bytes);
  expectRejected(verifyPackageManifest(std::move(manifest), root),
                 "padding is not zero");
}

TEST_F(PackageManifestTest, RejectsNonZeroTrailingProgramDataBytes) {
  PackageManifest manifest = makeManifest();
  std::vector<uint8_t> bytes(96, 0);
  for (size_t index = 80; index < bytes.size(); ++index)
    bytes[index] = UINT8_C(0xAB);
  llvm::SHA256 hasher;
  hasher.update(bytes);
  manifest.programData = {"data/program-data.bin", 96, 64,
                          "sha256:" + llvm::toHex(hasher.final(),
                                                  /*LowerCase=*/true)};
  writeProgramData(bytes);
  expectRejected(verifyPackageManifest(std::move(manifest), root),
                 "trailing bytes");
}

TEST_F(PackageManifestTest, RejectsZeroTrailingProgramDataBytes) {
  PackageManifest manifest = makeManifest();
  std::vector<uint8_t> bytes = programDataBytes();
  bytes.resize(96, 0);
  llvm::SHA256 hasher;
  hasher.update(bytes);
  manifest.programData = {"data/program-data.bin", 96, 64,
                          "sha256:" + llvm::toHex(hasher.final(),
                                                  /*LowerCase=*/true)};
  writeProgramData(bytes);
  expectRejected(verifyPackageManifest(std::move(manifest), root),
                 "trailing bytes");
}

// The strict loader closes the whole package root: exactly manifest.json,
// modules/ and data/program-data.bin, all regular files.

TEST_F(PackageManifestTest, StrictLoaderRejectsUndeclaredRootMember) {
  writeFullProgramData();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(makeManifest(), root);
  ASSERT_TRUE(static_cast<bool>(verified));
  llvm::SmallString<256> manifestPath(root);
  llvm::sys::path::append(manifestPath, kPackageManifestFileName);
  std::error_code error;
  llvm::raw_fd_ostream output(manifestPath, error, llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  output << serializeCanonicalPackageJson(*verified);
  output.close();
  llvm::SmallString<256> extra(root);
  llvm::sys::path::append(extra, "undeclared.txt");
  llvm::raw_fd_ostream extraOutput(extra, error, llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  extraOutput << "extra";
  extraOutput.close();
  llvm::Expected<ExecutablePackage> loaded = loadExecutablePackage(root);
  expectRejected(std::move(loaded), "undeclared member");
}

TEST_F(PackageManifestTest, StrictLoaderRejectsUndeclaredDataMember) {
  writeFullProgramData();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(makeManifest(), root);
  ASSERT_TRUE(static_cast<bool>(verified));
  llvm::SmallString<256> manifestPath(root);
  llvm::sys::path::append(manifestPath, kPackageManifestFileName);
  std::error_code error;
  llvm::raw_fd_ostream output(manifestPath, error, llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  output << serializeCanonicalPackageJson(*verified);
  output.close();
  llvm::SmallString<256> extra(root);
  llvm::sys::path::append(extra, "data", "aux.payload");
  llvm::raw_fd_ostream extraOutput(extra, error, llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  extraOutput << "aux";
  extraOutput.close();
  llvm::Expected<ExecutablePackage> loaded = loadExecutablePackage(root);
  expectRejected(std::move(loaded), "undeclared member");
}

TEST_F(PackageManifestTest, StrictLoaderRejectsSymlinkMember) {
  writeFullProgramData();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(makeManifest(), root);
  ASSERT_TRUE(static_cast<bool>(verified));
  llvm::SmallString<256> manifestPath(root);
  llvm::sys::path::append(manifestPath, kPackageManifestFileName);
  std::error_code error;
  llvm::raw_fd_ostream output(manifestPath, error, llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  output << serializeCanonicalPackageJson(*verified);
  output.close();
  llvm::SmallString<256> link(root);
  llvm::sys::path::append(link, "linked-manifest");
  ASSERT_FALSE(llvm::sys::fs::create_link(manifestPath, link));
  llvm::Expected<ExecutablePackage> loaded = loadExecutablePackage(root);
  expectRejected(std::move(loaded), "unsupported member");
}

TEST_F(PackageManifestTest, StrictLoaderRejectsUndeclaredModulesDirectory) {
  writeFullProgramData();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(makeManifest(), root);
  ASSERT_TRUE(static_cast<bool>(verified));
  llvm::SmallString<256> manifestPath(root);
  llvm::sys::path::append(manifestPath, kPackageManifestFileName);
  std::error_code error;
  llvm::raw_fd_ostream output(manifestPath, error, llvm::sys::fs::OF_Text);
  ASSERT_FALSE(error);
  output << serializeCanonicalPackageJson(*verified);
  output.close();
  llvm::SmallString<256> extra(root);
  llvm::sys::path::append(extra, "modules", "undeclared");
  ASSERT_FALSE(llvm::sys::fs::create_directory(extra));
  llvm::Expected<ExecutablePackage> loaded = loadExecutablePackage(root);
  expectRejected(std::move(loaded), "undeclared directory");
}
