//===- SystemCTargetModelAccessReuseTest.cpp - Actual reuse execution -----===//
#include "TestSupport/Transforms/AccessReuseInputs.h"
#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "Wafer/Simulator/Invocation/ProgramInvocation.h"
#include "Wafer/Simulator/Invocation/TargetModelInvocation.h"
#include "Wafer/Simulator/SystemC/SystemCTargetModel.h"
#include "Wafer/Transforms/Tile/AccessReuse.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

namespace {
using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::compiler::detail;
using namespace wafer::model;

frontend::ProgramBoundaryBinding boundary(llvm::ArrayRef<int64_t> shape) {
  frontend::ProgramBoundaryBinding result;
  result.index = result.programIndex = 0;
  result.globalShape.assign(shape.begin(), shape.end());
  result.localShape = result.globalShape;
  frontend::ProgramPartitionSlice slice;
  slice.partitionId = slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes = result.globalShape;
  slice.strides.assign(shape.size(), 1);
  result.partitionSlices.push_back(std::move(slice));
  return result;
}

using ReuseParameters = std::tuple<const char *, int64_t, AccessReuseKind>;
class SystemCTargetModelAccessReuseTest
    : public ::testing::TestWithParam<ReuseParameters> {};
TEST_P(SystemCTargetModelAccessReuseTest,
       CompleteReadWindowsMatchIndependentInputBytes) {
  const auto &[type, extent, kind] = GetParam();
  llvm::StringRef spelling(type);
  SCOPED_TRACE(::testing::Message()
               << spelling.str() << "/" << extent << "/" << int(kind));
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  bool sliding = kind == AccessReuseKind::Sliding;
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      wafer::testing::makeAccessReuseInput(16, extent, spelling, sliding),
      context.get());
  ASSERT_TRUE(module);
  auto facts = analysis::analyzeAccessReuse(*module);
  AccessReuseChoice choice;
  if (sliding) {
    for (const auto &window : facts.sliding)
      choice.actions.push_back({kind, {window.access.load}, window.scope, {}});
  } else {
    for (const auto &window : facts.scopes) {
      if (!window.outerLoops.empty())
        continue;
      AccessReuseAction action{kind, window.reads, window.scope, {}};
      if (kind == AccessReuseKind::TwoLevel)
        for (const auto &inner : facts.scopes)
          if (inner.source == window.source && inner.windowBytes == 256 * 128) {
            action.innerScope = inner.scope;
            break;
          }
      // Exercise time and space reuse in the same transaction.
      action.shareWindow = true;
      choice.actions.push_back(std::move(action));
    }
  }
  ASSERT_EQ(choice.actions.size(), 16u);
  StructuredMaterializationRelations relations;
  rebuildCurrentBufferOwnerRelations(*module, relations);
  auto reused = materializeAccessReuse(*module, relations, choice);
  ASSERT_TRUE(reused.succeeded()) << reused.detail;
  if (!sliding) {
    EXPECT_EQ(reused.movement.peerReceives, 15u);
  }

  const int64_t inputRows = sliding ? extent + 7 : extent;
  const int64_t batches = 16 * (sliding ? extent : 16);
  const int64_t outputRows = sliding ? 8 : extent;
  frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary({1, inputRows, 64})};
  program.distributedOutputs = {boundary({batches, outputRows, 64})};
  auto format = llvm::cantFail(parseProgramElementType(spelling));
  program.distributedInputs.front().dtype = format;
  program.distributedOutputs.front().dtype = format;
  std::string detail;
  llvm::raw_string_ostream diagnostics(detail);
  ProgramDataHandoff data;
  llvm::SmallVector<TileId> tiles;
  for (unsigned tile = 0; tile < 16; ++tile)
    tiles.push_back(TileId(tile));
  CurrentIRDownstreamOptions options;
  options.communication = CommunicationProposalPolicy::DependencyOrdered;
  auto config = llvm::cantFail(ExecutionConfig::createForSingleCard(1));
  auto compiled = compileCurrentIRCandidateToExecutable(
      std::move(module), std::move(relations), CardId(0), tiles, program,
      config, diagnostics, data, options);
  ASSERT_TRUE(compiled.isAccepted())
      << compiled.gate << ": " << compiled.detail << "\n"
      << detail;
  auto executable = DeviceExecutableBuilder::makeDeviceExecutable(
      config, std::move(compiled.executable->runtimeLaunchContract), context,
      std::move(compiled.executable->tiles),
      std::make_unique<ProgramDataHandoff>(std::move(data)));
  auto target =
      compileDeviceExecutableToTargetLLVMModules(executable, diagnostics);
  ASSERT_TRUE(bool(target)) << detail << llvm::toString(target.takeError());

  std::vector<uint8_t> bytes(inputRows * 64 * 2);
  for (size_t i = 0; i < bytes.size() / 2; ++i) {
    uint16_t bits =
        (spelling == "f16" ? 0x3c00 : 0x3f80) +
        (i * 17 + i / 37 + i / 257) % (spelling == "f16" ? 1024 : 128);
    bytes[2 * i] = bits & 255;
    bytes[2 * i + 1] = bits >> 8;
  }
  std::vector<uint8_t> expected(batches * outputRows * 64 * 2);
  for (int64_t batch = 0; batch < batches; ++batch)
    for (int64_t row = 0; row < outputRows; ++row) {
      int64_t inputRow = sliding ? batch % extent + row : row;
      std::copy_n(bytes.begin() + inputRow * 128, 128,
                  expected.begin() + (batch * outputRows + row) * 128);
    }
  auto input = ProgramTensor::create(format, {1, inputRows, 64}, bytes);
  ASSERT_TRUE(bool(input)) << llvm::toString(input.takeError());
  auto calls = prepareProgramInvocations(executable, {{0, std::move(*input)}});
  ASSERT_TRUE(bool(calls)) << llvm::toString(calls.takeError());
  auto invocation = prepareTargetModelInvocation(executable, *target, *calls);
  ASSERT_TRUE(bool(invocation)) << llvm::toString(invocation.takeError());
  auto result = executeSystemCTargetModel(
      std::move(invocation->getExecutable()), invocation->getInputBindings(),
      TargetModelKernelBudget::create(
          FormalNumericWorkBudget::create(1000000, 1000000),
          256ULL * 1024 * 1024, 10000000));
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedTileCount, 16);
  ASSERT_EQ(result->outputs.size(), 1u);
  EXPECT_EQ(result->outputs.front().bytes, expected);
}
INSTANTIATE_TEST_SUITE_P(
    Windows, SystemCTargetModelAccessReuseTest,
    ::testing::Combine(
        ::testing::Values("f16", "bf16"),
        ::testing::Values(int64_t(1024), int64_t(1025), int64_t(1031)),
        ::testing::Values(AccessReuseKind::Resident, AccessReuseKind::Sliding,
                          AccessReuseKind::TwoLevel)));
} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
