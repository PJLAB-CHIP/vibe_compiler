//===- SystemCTargetModelOutputStoresTest.cpp - Stream actual writers ----===//
#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "Wafer/Simulator/Invocation/ProgramInvocation.h"
#include "Wafer/Simulator/Invocation/TargetModelInvocation.h"
#include "Wafer/Simulator/SystemC/SystemCTargetModel.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/TiledOutputStores.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

namespace {
using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::compiler::detail;
using namespace wafer::model;

frontend::ProgramBoundaryBinding boundary(unsigned index,
                                          ProgramElementType type,
                                          llvm::ArrayRef<int64_t> shape) {
  frontend::ProgramBoundaryBinding result;
  result.index = result.programIndex = index;
  result.dtype = type;
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

std::string makeInput(int64_t rows, llvm::StringRef dtype) {
  auto type = [&](int64_t batches, int64_t count, llvm::StringRef space,
                  llvm::StringRef layout = "") {
    return "memref<" + std::to_string(batches) + "x" + std::to_string(count) +
           "x64x" + dtype.str() + layout.str() + ", #wafer.memory<" +
           space.str() + ", tensor>>";
  };
  auto input = type(1, rows, "ddr");
  auto output = type(16, rows, "ddr");
  auto full = type(1, rows, "spm");
  std::string text;
  llvm::raw_string_ostream out(text);
  out << "module { wafer.execution.mesh @mesh {axes = [\"card\"], shape = "
         "array<i64: 1>} wafer.target.topology @topology {card_grid = "
         "array<i64: 1, 1>, card_interconnect = \"mesh\", tile_grid = "
         "array<i64: 4, 4>, unavailable_tiles = array<i64>} ";
  for (int64_t tile = 0; tile < 16; ++tile) {
    auto destination =
        type(1, rows, "ddr",
             ", strided<[" + std::to_string(rows * 64) +
                 ", 64, 1], offset: " + std::to_string(tile * rows * 64) + ">");
    out << "wafer.tile.module card_id = 0 tile_id = " << tile
        << " { func.func @main(%input: " << input
        << " {wafer.program_argument = #wafer.program_argument<0>}) -> ("
        << output << ", " << output << ") { ";
    for (unsigned sink = 0; sink < 2; ++sink)
      out << "%output" << sink << " = memref.alloc() : " << output << " %part"
          << sink << " = memref.subview %output" << sink << "[" << tile
          << ", 0, 0] [1, " << rows << ", 64] [1, 1, 1] : " << output << " to "
          << destination << " ";
    out << "wafer.tile.region(%input, %part0, %part1 : " << input << ", "
        << destination << ", " << destination
        << ") -> () { ^bb0(%src: " << input << ", %dst0: " << destination
        << ", %dst1: " << destination << "): %full = memref.alloc() : " << full
        << " %c0 = arith.constant 0 : index %end = arith.constant 1024 : index "
           "%step = arith.constant 128 : index %middle = arith.constant 512 : "
           "index ";
    auto emit = [&](int64_t count, llvm::StringRef row, llvm::StringRef root,
                    llvm::StringRef tag, bool conditional) {
      auto layout =
          ", strided<[" + std::to_string(rows * 64) + ", 64, 1], offset: ?>";
      auto read = type(1, count, "ddr", layout);
      auto write = type(1, count, "spm", layout);
      auto compact = type(1, count, "spm");
      out << "%read" << tag << " = memref.subview %src[0, " << row
          << ", 0] [1, " << count << ", 64] [1, 1, 1] : " << input << " to "
          << read << " %write" << tag << " = memref.subview " << root << "[0, "
          << row << ", 0] [1, " << count << ", 64] [1, 1, 1] : " << full
          << " to " << write << " ";
      if (conditional)
        out << "%direct = arith.cmpi eq, " << row
            << ", %middle : index scf.if %direct { ";
      out << "wafer.tile.load %read" << tag << " into %write" << tag << " : "
          << read << " into " << write << " ";
      if (conditional)
        out << "} else { %compact = memref.alloc() : " << compact
            << " wafer.tile.load %read" << tag << " into %compact : " << read
            << " into " << compact
            << " wafer.tile.copy_into %compact into %write" << tag << " : "
            << compact << " into " << write << " } ";
    };
    out << "%result = scf.for %row = %c0 to %end step %step "
           "iter_args(%carrier = %full) -> ("
        << full << ") { ";
    emit(128, "%row", "%carrier", "main", true);
    out << "scf.yield %carrier : " << full << " } ";
    if (rows != 1024)
      emit(rows - 1024, "%end", "%result", "tail", false);
    for (unsigned sink = 0; sink < 2; ++sink)
      out << "wafer.tile.store %result, %dst" << sink << " : " << full << " -> "
          << destination << " ";
    out << "wafer.tile.yield } return %output0, %output1 : " << output << ", "
        << output << " } } ";
  }
  out << "}";
  return text;
}

class SystemCTargetModelOutputStoresTest
    : public ::testing::TestWithParam<std::tuple<int64_t, const char *>> {};

TEST_P(SystemCTargetModelOutputStoresTest,
       MixedWritersPreserveEveryOutputByte) {
  auto [rows, dtype] = GetParam();
  auto format = llvm::cantFail(parseProgramElementType(dtype));
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(makeInput(rows, dtype),
                                                        context.get());
  ASSERT_TRUE(module);
  BoundaryMovementStatistics statistics;
  materializeTiledOutputStores(*module, statistics);
  ASSERT_EQ(statistics.streamedOutputCarriers, 16u);
  StructuredMaterializationRelations relations;
  rebuildCurrentBufferOwnerRelations(*module, relations);
  ASSERT_TRUE(mlir::succeeded(optimizeCurrentIRStorage(*module, relations)));
  frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, format, {1, rows, 64})};
  program.distributedOutputs = {boundary(0, format, {16, rows, 64}),
                                boundary(1, format, {16, rows, 64})};
  ProgramDataHandoff data;
  llvm::SmallVector<TileId> tiles;
  for (unsigned tile = 0; tile < 16; ++tile)
    tiles.push_back(TileId(tile));
  auto config = llvm::cantFail(ExecutionConfig::createForSingleCard(1));
  std::string detail;
  llvm::raw_string_ostream diagnostics(detail);
  auto compiled = compileCurrentIRCandidateToExecutable(
      std::move(module), std::move(relations), CardId(0), tiles, program,
      config, diagnostics, data);
  ASSERT_TRUE(compiled.isAccepted())
      << compiled.gate << ": " << compiled.detail << "\n"
      << detail;
  unsigned steadyStateJoins = 0;
  for (auto &tile : compiled.executable->tiles)
    tile.getModule().walk([&](SyncNCCJoinOp join) {
      steadyStateJoins += bool(join->getParentOfType<mlir::scf::ForOp>());
    });
  EXPECT_EQ(steadyStateJoins, 0u);
  auto executable = DeviceExecutableBuilder::makeDeviceExecutable(
      config, std::move(compiled.executable->runtimeLaunchContract), context,
      std::move(compiled.executable->tiles),
      std::make_unique<ProgramDataHandoff>(std::move(data)));
  auto target =
      compileDeviceExecutableToTargetLLVMModules(executable, diagnostics);
  ASSERT_TRUE(bool(target)) << detail << llvm::toString(target.takeError());
  std::vector<uint8_t> bytes(rows * 64 * 2);
  for (size_t i = 0; i < bytes.size() / 2; ++i) {
    uint16_t bits = (llvm::StringRef(dtype) == "f16" ? 0x3c00 : 0x3f80) +
                    (i * 17 + i / 37) % 128;
    bytes[2 * i] = bits & 255;
    bytes[2 * i + 1] = bits >> 8;
  }
  std::vector<uint8_t> expected;
  for (unsigned tile = 0; tile < 16; ++tile)
    expected.insert(expected.end(), bytes.begin(), bytes.end());
  auto input =
      llvm::cantFail(ProgramTensor::create(format, {1, rows, 64}, bytes));
  auto calls = prepareProgramInvocations(executable, {{0, std::move(input)}});
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
  ASSERT_EQ(result->outputs.size(), 2u);
  for (const auto &output : result->outputs)
    EXPECT_EQ(output.bytes, expected);
}

INSTANTIATE_TEST_SUITE_P(Windows, SystemCTargetModelOutputStoresTest,
                         ::testing::Combine(::testing::Values(int64_t(1024),
                                                              int64_t(1025),
                                                              int64_t(1031)),
                                            ::testing::Values("f16", "bf16")));
} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
