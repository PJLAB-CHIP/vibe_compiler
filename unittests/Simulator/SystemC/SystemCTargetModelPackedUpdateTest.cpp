//===- SystemCTargetModelPackedUpdateTest.cpp - Private BOOL writes ------===//
#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "Wafer/Simulator/Invocation/ProgramInvocation.h"
#include "Wafer/Simulator/Invocation/TargetModelInvocation.h"
#include "Wafer/Simulator/SystemC/SystemCTargetModel.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

namespace {
using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::compiler::detail;
using namespace wafer::model;

frontend::ProgramBoundaryBinding boundary(int64_t index,
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

std::string makeInput(int64_t rows, bool local) {
  std::string text;
  llvm::raw_string_ostream out(text);
  auto tensor = [](int64_t batches, int64_t rows, int64_t columns,
                   llvm::StringRef dtype, llvm::StringRef space) {
    return "memref<" + std::to_string(batches) + "x" + std::to_string(rows) +
           "x" + std::to_string(columns) + "x" + dtype.str() +
           ", #wafer.memory<" + space.str() + ", tensor>>";
  };
  auto initial = tensor(1, rows, 39, "i1", "ddr");
  auto source = tensor(1, rows, 32, "i1", "ddr");
  auto output = tensor(16, rows, 39, "f16", "ddr");
  auto localBits = tensor(1, rows, 39, "i1", "spm");
  auto floats = tensor(1, rows, 39, "f16", "spm");
  out << "module { wafer.execution.mesh @mesh {axes = [\"card\"], shape = "
         "array<i64: 1>} "
         "wafer.target.topology @topology {card_grid = array<i64: 1, 1>, "
         "card_interconnect = \"mesh\", tile_grid = array<i64: 4, 4>, "
         "unavailable_tiles = array<i64>} ";
  for (int64_t tile = 0; tile < 16; ++tile) {
    out << "wafer.tile.module card_id = 0 tile_id = " << tile
        << " { func.func @main(%initial: " << initial
        << " {wafer.program_argument = #wafer.program_argument<0>}, %source: "
        << source
        << " {wafer.program_argument = #wafer.program_argument<1>}) -> "
        << output << " { "
        << "%output = memref.alloc() : " << output << " ";
    if (!local)
      out << "%private = memref.alloc() : " << initial << " ";
    out << "wafer.tile.region(%initial, %source, %output"
        << (local ? "" : ", %private") << " : " << initial << ", " << source
        << ", " << output << (local ? "" : ", " + initial)
        << ") -> () { ^bb0(%old: " << initial << ", %src: " << source
        << ", %out: " << output << (local ? "" : ", %dst: " + initial) << "): "
        << "%bits = memref.alloc() : " << localBits << " "
        << "wafer.tile.load %old into %bits : " << initial << " into "
        << localBits << " ";
    if (!local)
      out << "wafer.tile.store %bits, %dst : " << localBits << " -> " << initial
          << " ";
    out << "%c0 = arith.constant 0 : index %end = arith.constant 1024 : index "
           "%step = arith.constant 256 : index ";
    auto emitUpdate = [&](int64_t count, llvm::StringRef row,
                          llvm::StringRef tag) {
      auto view = [&](int64_t columns, llvm::StringRef space) {
        return "memref<1x" + std::to_string(count) + "x32xi1, strided<[" +
               std::to_string(rows * columns) + ", " + std::to_string(columns) +
               ", 1], offset: ?>, #wafer.memory<" + space.str() + ", tensor>>";
      };
      auto compact = tensor(1, count, 32, "i1", "spm");
      out << "%read" << tag << " = memref.subview %src[0, " << row
          << ", 0] [1, " << count << ", 32] [1, 1, 1] : " << source << " to "
          << view(32, "ddr") << " "
          << "%values" << tag << " = memref.alloc() : " << compact << " "
          << "wafer.tile.load %read" << tag << " into %values" << tag << " : "
          << view(32, "ddr") << " into " << compact << " "
          << "%write" << tag << " = memref.subview "
          << (local ? "%bits" : "%dst") << "[0, " << row << ", " << tile % 8
          << "] [1, " << count
          << ", 32] [1, 1, 1] : " << (local ? localBits : initial) << " to "
          << view(39, local ? "spm" : "ddr") << " ";
      if (local)
        out << "memref.copy %values" << tag << ", %write" << tag << " : "
            << compact << " to " << view(39, "spm") << " ";
      else
        out << "wafer.tile.store %values" << tag << ", %write" << tag << " : "
            << compact << " -> " << view(39, "ddr") << " ";
    };
    out << "scf.for %row = %c0 to %end step %step { ";
    emitUpdate(256, "%row", "main");
    out << "} ";
    if (rows != 1024)
      emitUpdate(rows - 1024, "%end", "tail");
    if (!local)
      out << "wafer.tile.load %dst into %bits : " << initial << " into "
          << localBits << " ";
    auto outputView = "memref<1x" + std::to_string(rows) +
                      "x39xf16, strided<[" + std::to_string(rows * 39) +
                      ", 39, 1], offset: " + std::to_string(tile * rows * 39) +
                      ">, #wafer.memory<ddr, tensor>>";
    out << "%values = memref.alloc() : " << floats << " "
        << "wafer.instr.bit2fp %bits into %values : " << localBits << " to "
        << floats << " "
        << "%publish = memref.subview %out[" << tile << ", 0, 0] [1, " << rows
        << ", 39] [1, 1, 1] : " << output << " to " << outputView << " "
        << "wafer.tile.store %values, %publish : " << floats << " -> "
        << outputView << " "
        << "wafer.tile.yield } return %output : " << output << " } } ";
  }
  out << "}";
  return text;
}

class SystemCTargetModelPackedUpdateTest
    : public ::testing::TestWithParam<std::tuple<int64_t, bool>> {};

TEST_P(SystemCTargetModelPackedUpdateTest,
       MainAndTailPreserveNeighbouringBits) {
  auto [rows, local] = GetParam();
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(makeInput(rows, local),
                                                        context.get());
  ASSERT_TRUE(module);
  auto boolean = llvm::cantFail(parseProgramElementType("i1"));
  frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, boolean, {1, rows, 39}),
                               boundary(1, boolean, {1, rows, 32})};
  program.distributedOutputs = {
      boundary(0, ProgramElementType::F16, {16, rows, 39})};
  StructuredMaterializationRelations relations;
  rebuildCurrentBufferOwnerRelations(*module, relations);
  ASSERT_TRUE(mlir::succeeded(optimizeCurrentIRStorage(*module, relations)));
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
  unsigned bitExpansions = 0;
  unsigned steadyStateJoins = 0;
  for (auto &tile : compiled.executable->tiles) {
    tile.getModule().walk([&](InstrBit2FpOp) { ++bitExpansions; });
    tile.getModule().walk([&](SyncNCCJoinOp join) {
      steadyStateJoins += bool(join->getParentOfType<mlir::scf::ForOp>());
    });
  }
  EXPECT_EQ(steadyStateJoins, 0u);
  EXPECT_GT(bitExpansions,
            16u); // The actual private update survived storage optimization.
  auto executable = DeviceExecutableBuilder::makeDeviceExecutable(
      config, std::move(compiled.executable->runtimeLaunchContract), context,
      std::move(compiled.executable->tiles),
      std::make_unique<ProgramDataHandoff>(std::move(data)));
  auto target =
      compileDeviceExecutableToTargetLLVMModules(executable, diagnostics);
  ASSERT_TRUE(bool(target)) << detail << llvm::toString(target.takeError());
  std::vector<uint8_t> old(rows * 39), values(rows * 32);
  for (size_t i = 0; i < old.size(); ++i)
    old[i] = (i * 7 + i / 39) % 3 == 1;
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (i * 11 + i / 32) % 5 < 2;
  std::vector<uint8_t> expected(16 * rows * 39 * 2);
  for (int64_t tile = 0; tile < 16; ++tile)
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t c = 0; c < 39; ++c) {
        bool replaced = c >= tile % 8 && c < tile % 8 + 32;
        bool bit = replaced ? values[r * 32 + c - tile % 8] : old[r * 39 + c];
        expected[((tile * rows + r) * 39 + c) * 2 + 1] = bit ? 0x3c : 0;
      }
  auto oldTensor =
      llvm::cantFail(ProgramTensor::create(boolean, {1, rows, 39}, old));
  auto valueTensor =
      llvm::cantFail(ProgramTensor::create(boolean, {1, rows, 32}, values));
  auto calls = prepareProgramInvocations(
      executable, {{0, std::move(oldTensor)}, {1, std::move(valueTensor)}});
  ASSERT_TRUE(bool(calls)) << llvm::toString(calls.takeError());
  auto invocation = prepareTargetModelInvocation(executable, *target, *calls);
  ASSERT_TRUE(bool(invocation)) << llvm::toString(invocation.takeError());
  auto result = executeSystemCTargetModel(
      std::move(invocation->getExecutable()), invocation->getInputBindings(),
      TargetModelKernelBudget::create(
          FormalNumericWorkBudget::create(10000000, 1000000),
          256ULL * 1024 * 1024, 10000000));
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedTileCount, 16);
  ASSERT_EQ(result->outputs.size(), 1u);
  EXPECT_EQ(result->outputs.front().bytes, expected);
}

INSTANTIATE_TEST_SUITE_P(Windows, SystemCTargetModelPackedUpdateTest,
                         ::testing::Combine(::testing::Values(int64_t(1024),
                                                              int64_t(1025),
                                                              int64_t(1031)),
                                            ::testing::Bool()));
} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
