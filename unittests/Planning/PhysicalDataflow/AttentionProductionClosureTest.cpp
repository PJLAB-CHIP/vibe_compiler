//===- AttentionProductionClosureTest.cpp ----------------------------===//

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"
#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"

#include "Wafer/Analysis/ScheduleCost/ScheduleCostAnalysis.h"
#include "Wafer/CodeGen/Executable/CardExecutableInternal.h"
#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Package/Writer/PackageInternal.h"
#include "Wafer/Target/Core/TargetMemory.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler::testing;

llvm::Expected<wafer::compiler::TargetToolchain> makeTestToolchain() {
  llvm::SmallString<256> pythonExecutable;
  llvm::SmallString<256> llvmClangXX;
  if (std::error_code error = llvm::sys::fs::real_path(
          WAFER_TEST_PYTHON_EXECUTABLE, pythonExecutable))
    return llvm::createStringError(error,
                                   "failed to resolve test Python executable");
  if (std::error_code error =
          llvm::sys::fs::real_path(WAFER_TEST_LLVM_CLANGXX, llvmClangXX))
    return llvm::createStringError(error, "failed to resolve test clang++");
  return wafer::compiler::TargetToolchain::create(
      pythonExecutable, WAFER_TEST_DEVICE_LINKER_SCRIPT, llvmClangXX,
      WAFER_TEST_TX8_DEPS_ROOT, WAFER_TEST_WAFER_INCLUDE_DIR,
      WAFER_TEST_WAFER_CRT_SOURCE, WAFER_TEST_WAFER_CRT_INCLUDE_DIR);
}

std::string makeSource(bool decoding, int64_t queryExtent,
                       int64_t keyValueExtent, bool withMask,
                       bool bf16 = false) {
  std::string source =
      decoding ? wafer::test::buildFlashDecodingPlanningFixture(queryExtent,
                                                                keyValueExtent)
               : wafer::test::buildFlashAttentionPlanningFixture(
                     queryExtent, keyValueExtent, withMask);
  const size_t scaleArgument = source.find(", %scale: f32");
  EXPECT_NE(scaleArgument, std::string::npos);
  source.erase(scaleArgument, std::string(", %scale: f32").size());
  const size_t outputInit = source.find("    %out = tensor.empty");
  EXPECT_NE(outputInit, std::string::npos);
  source.insert(outputInit,
                "    %scale = arith.constant 0.08838834764831845 : f32\n");
  const size_t moduleBody = source.find("module {");
  EXPECT_NE(moduleBody, std::string::npos);
  source.insert(moduleBody + std::string("module {").size(), R"mlir(
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
)mlir");
  if (bf16)
    for (size_t position = 0;
         (position = source.find("f16", position)) != std::string::npos;
         position += 4)
      source.replace(position, 3, "bf16");
  return source;
}

frontend::FrontendProgramVerificationResult makeMetadata(bool decoding,
                                                         int64_t queryExtent,
                                                         int64_t keyValueExtent,
                                                         bool withMask) {
  frontend::FrontendProgramVerificationResult metadata;
  metadata.numPartitions = 1;
  metadata.programUserInputCount = decoding || withMask ? 4 : 3;
  metadata.distributedInputs = {boundary(0, {2, queryExtent, 128}),
                                boundary(1, {2, keyValueExtent, 128}),
                                boundary(2, {2, keyValueExtent, 64})};
  if (decoding || withMask)
    metadata.distributedInputs.push_back(
        boundary(3, {queryExtent, keyValueExtent}));
  metadata.distributedOutputs = {boundary(0, {2, queryExtent, 64})};
  return metadata;
}

void verifyPackageAndNoCard(const wafer::compiler::CardExecutable &executable,
                            llvm::raw_ostream &diagnostics,
                            bool expectCardWorkspace) {
  llvm::Expected<wafer::compiler::TargetLLVMModules> targetLLVM =
      wafer::compiler::compileCardExecutableToTargetLLVMModules(executable,
                                                                diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVM))
      << llvm::toString(targetLLVM.takeError());
  llvm::Expected<wafer::compiler::TargetToolchain> toolchain =
      makeTestToolchain();
  ASSERT_TRUE(static_cast<bool>(toolchain))
      << llvm::toString(toolchain.takeError());
  llvm::SmallString<256> temporaryDirectory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("wafer-attention-package",
                                                    temporaryDirectory));
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(temporaryDirectory); });
  llvm::SmallString<256> linkedDirectory(temporaryDirectory);
  llvm::sys::path::append(linkedDirectory, "linked");
  llvm::Expected<wafer::compiler::LinkedTargetModules> linked =
      wafer::compiler::linkTargetLLVMModules(*targetLLVM, linkedDirectory,
                                             *toolchain, diagnostics);
  ASSERT_TRUE(static_cast<bool>(linked)) << llvm::toString(linked.takeError());
  llvm::SmallString<256> packageDirectory(temporaryDirectory);
  llvm::sys::path::append(packageDirectory, "package");
  auto package = wafer::compiler::detail::writePackage(
      temporaryDirectory, executable, *linked, packageDirectory, diagnostics,
      std::nullopt);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  bool sawCardWorkspace = false;
  for (const wafer::runtime::PackageEntrypointRecord &entry : manifest.entries)
    for (const wafer::runtime::TileEntryArgumentRecord &argument :
         entry.arguments)
      sawCardWorkspace |=
          std::holds_alternative<wafer::runtime::CardWorkspaceArgument>(
              argument.reference);
  EXPECT_EQ(sawCardWorkspace, expectCardWorkspace);
  std::vector<wafer::runtime::RuntimeInvocationBinding> bindings;
  for (const wafer::runtime::ExternalPortRecord &input : manifest.inputs)
    bindings.push_back({input.id, input.bytes, input.alignment});
  wafer::runtime::RuntimeEnvironment environment{
      manifest.targetIdentity, manifest.runtimeABI, manifest.moduleFormat};
  const wafer::KernelRuntimeLaunchContract &kernel =
      manifest.launch.getKernel();
  environment.supportedKernelLaunchForms = {kernel.form};
  environment.supportedKernelEntryABIs = {kernel.entryABI};
  auto noCard =
      wafer::runtime::planRuntimeInvocation(*package, bindings, environment);
  ASSERT_TRUE(static_cast<bool>(noCard)) << llvm::toString(noCard.takeError());
  EXPECT_EQ(noCard->tiles.size(), 16u);
  EXPECT_EQ(noCard->cardWorkspaceRanges.empty(), !expectCardWorkspace);
}

struct AttentionProductionCase {
  bool decoding;
  int64_t queryExtent;
  int64_t keyValueExtent;
  bool withMask;
  bool searchPolicy;
  bool bf16;
};

class AttentionProductionClosureTest
    : public ::testing::TestWithParam<AttentionProductionCase> {};

TEST(AttentionProductionClosureTest,
     DecodeCardDDRResourcesReachTheTypedTargetABI) {
  constexpr int64_t extent = 1024;
  const std::string source = makeSource(/*decoding=*/true, extent, extent,
                                        /*withMask=*/true);
  const auto metadata = makeMetadata(/*decoding=*/true, extent, extent,
                                     /*withMask=*/true);
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::ExecutionConfig config = executionConfig();
  auto executable = wafer::compiler::detail::buildCardExecutable(
      context, *module, metadata, config, OptimizationConfig::none(),
      diagnostics, std::nullopt, programData);
  diagnostics.flush();
  ASSERT_TRUE(static_cast<bool>(executable))
      << (executable ? "" : llvm::toString(executable.takeError())) << '\n'
      << diagnosticsText;

  struct ResourceAccess {
    int64_t bytes = -1;
    int64_t alignment = -1;
    bool read = false;
    bool write = false;
  };
  std::map<int64_t, ResourceAccess> resources;
  for (const wafer::compiler::TileExecutable &tile :
       executable->getTileExecutables()) {
    auto prepared = wafer::compiler::detail::prepareTargetABI(
        tile, config, /*transportPreparedBeforeEntry=*/false);
    ASSERT_TRUE(mlir::succeeded(prepared));
    EXPECT_EQ(prepared->module->getOps<mlir::memref::GlobalOp>().empty(), true);
    for (const wafer::compiler::TileEntryArgument &slot : prepared->slots) {
      if (slot.kind != wafer::compiler::TileEntryArgumentKind::CardWorkspace)
        continue;
      ASSERT_GE(slot.resourceIndex, 0);
      ResourceAccess &resource = resources[slot.resourceIndex];
      if (resource.bytes < 0) {
        resource.bytes = slot.byteSize;
        resource.alignment = slot.alignment;
      } else {
        EXPECT_EQ(resource.bytes, slot.byteSize);
        EXPECT_EQ(resource.alignment, slot.alignment);
      }
      resource.read |=
          slot.access == wafer::compiler::TileEntryArgumentAccess::ReadOnly ||
          slot.access == wafer::compiler::TileEntryArgumentAccess::ReadWrite;
      resource.write |=
          slot.access == wafer::compiler::TileEntryArgumentAccess::WriteOnly ||
          slot.access == wafer::compiler::TileEntryArgumentAccess::ReadWrite;
    }
  }
  ASSERT_FALSE(resources.empty());
  int64_t expectedResource = 0;
  for (const auto &[resourceId, access] : resources) {
    EXPECT_EQ(resourceId, expectedResource++);
    EXPECT_GT(access.bytes, 0);
    EXPECT_GT(access.alignment, 0);
    EXPECT_TRUE(access.read);
    EXPECT_TRUE(access.write);
  }

  verifyPackageAndNoCard(*executable, diagnostics,
                         /*expectCardWorkspace=*/true);
}

TEST_P(AttentionProductionClosureTest, CompilesIndependentlyInBothPolicies) {
  const AttentionProductionCase &test = GetParam();
  const std::string source =
      makeSource(test.decoding, test.queryExtent, test.keyValueExtent,
                 test.withMask, test.bf16);
  const auto metadata = makeMetadata(test.decoding, test.queryExtent,
                                     test.keyValueExtent, test.withMask);
  OptimizationConfig policy = test.searchPolicy ? OptimizationConfig::search()
                                                : OptimizationConfig::none();
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::buildCardExecutable(
      context, *module, metadata, executionConfig(), policy, diagnostics,
      std::nullopt, programData);
  diagnostics.flush();
  if (!executable) {
    ADD_FAILURE() << llvm::toString(executable.takeError()) << "\n"
                  << diagnosticsText;
    return;
  }
  EXPECT_EQ(executable->getTileExecutables().size(), 16u);
  std::vector<wafer::analysis::TileInstructionProgram> instructionPrograms;
  uint64_t directDTECrossingJoins = 0;
  uint64_t directDTECrossingParticipants = 0;
  uint64_t directDTEWaits = 0;
  for (const wafer::compiler::TileExecutable &tile :
       executable->getTileExecutables()) {
    unsigned attentionOps = 0;
    unsigned linalgOps = 0;
    unsigned waits = 0;
    tile.getModule().walk([&](mlir::Operation *operation) {
      attentionOps += mlir::isa<wafer::LinalgExtAttentionOp>(operation);
      linalgOps += mlir::isa<mlir::linalg::LinalgOp>(operation);
      if (auto join = mlir::dyn_cast<wafer::SyncNCCJoinOp>(operation)) {
        EXPECT_FALSE(join->getParentOfType<mlir::scf::ForOp>());
        mlir::Operation *nextIssue = join->getNextNode();
        while (mlir::isa_and_nonnull<mlir::memref::AllocOp>(nextIssue))
          nextIssue = nextIssue->getNextNode();
        if (mlir::isa_and_nonnull<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(
                nextIssue)) {
          ++directDTECrossingJoins;
          directDTECrossingParticipants += join.getParticipants().size();
        }
      }
      if (auto wait = mlir::dyn_cast<wafer::InstrDTEWaitOp>(operation)) {
        ++waits;
        ++directDTEWaits;
        EXPECT_FALSE(wait->getParentOfType<mlir::scf::ForOp>());
      }
      if (auto combine =
              mlir::dyn_cast<wafer::InstrGatherScatterOp>(operation))
        EXPECT_FALSE(combine.getCardDdrResourceAttr());
    });
    EXPECT_EQ(attentionOps, 0u);
    EXPECT_EQ(linalgOps, 0u);
    if (!test.decoding)
      EXPECT_EQ(waits, 0u);
    instructionPrograms.push_back(
        {tile.getTileId(), tile.getModule().getOperation()});
  }
  wafer::analysis::CardInstructionProgramCost cost =
      wafer::analysis::analyzeCardInstructionProgramCost(
          instructionPrograms, wafer::getTargetMemoryPolicy());
  ASSERT_TRUE(cost.aggregateSteadyStateNCCJoinCount.isKnown());
  ASSERT_TRUE(cost.aggregateNonTerminalNCCJoinCount.isKnown());
  ASSERT_TRUE(cost.aggregateSteadyStateNCCParticipantWaitCount.isKnown());
  ASSERT_TRUE(cost.aggregateNonTerminalNCCParticipantWaitCount.isKnown());
  EXPECT_EQ(cost.aggregateSteadyStateNCCJoinCount.value, 0u);
  EXPECT_EQ(cost.aggregateSteadyStateNCCParticipantWaitCount.value, 0u);
  if (test.decoding) {
    EXPECT_EQ(cost.aggregateNonTerminalNCCJoinCount.value,
              directDTECrossingJoins);
    EXPECT_EQ(cost.aggregateNonTerminalNCCParticipantWaitCount.value,
              directDTECrossingParticipants);
  } else {
    EXPECT_EQ(directDTEWaits, 0u);
    EXPECT_EQ(cost.aggregateNonTerminalNCCJoinCount.value, 0u);
    EXPECT_EQ(cost.aggregateNonTerminalNCCParticipantWaitCount.value, 0u);
  }
  if (policy.isSearch()) {
    EXPECT_EQ(diagnosticsText.find("deterministic-card-executable-baseline"),
              std::string::npos);
    EXPECT_EQ(diagnosticsText.find("physical-search result"),
              std::string::npos);
    EXPECT_EQ(diagnosticsText.find("planning-profile"), std::string::npos);
  } else {
    EXPECT_EQ(diagnosticsText.find("physical-search"), std::string::npos);
  }
  verifyPackageAndNoCard(*executable, diagnostics,
                         /*expectCardWorkspace=*/test.decoding);
}

INSTANTIATE_TEST_SUITE_P(
    AlignedAndRagged, AttentionProductionClosureTest,
    ::testing::Values(
        AttentionProductionCase{false, 1024, 1024, false, true, false},
        AttentionProductionCase{false, 1024, 1024, false, false, false},
        AttentionProductionCase{false, 1025, 1031, true, true, false},
        AttentionProductionCase{false, 1025, 1031, true, false, false},
        AttentionProductionCase{true, 1024, 1024, true, true, false},
        AttentionProductionCase{true, 1024, 1024, true, false, false},
        AttentionProductionCase{true, 1025, 1031, true, true, false},
        AttentionProductionCase{true, 1025, 1031, true, false, false},
        AttentionProductionCase{false, 1024, 1024, false, true, true},
        AttentionProductionCase{false, 1024, 1024, false, false, true},
        AttentionProductionCase{false, 1025, 1031, true, true, true},
        AttentionProductionCase{false, 1025, 1031, true, false, true},
        AttentionProductionCase{true, 1024, 1024, true, true, true},
        AttentionProductionCase{true, 1024, 1024, true, false, true},
        AttentionProductionCase{true, 1025, 1031, true, true, true},
        AttentionProductionCase{true, 1025, 1031, true, false, true}));

} // namespace
