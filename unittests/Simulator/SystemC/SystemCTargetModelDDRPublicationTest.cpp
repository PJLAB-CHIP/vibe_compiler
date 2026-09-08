//===- SystemCTargetModelDDRPublicationTest.cpp ---------------------------===//
#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"
#include "Wafer/Simulator/SystemC/SystemCTargetModel.h"
#include "Wafer/Target/TargetFormat.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"
#include <array>

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

namespace {
TEST(SystemCTargetModelDDRPublicationTest, MatchesPublicationAndFirstRead) {
  // A protocol test with real rank-3 payloads. Fifteen readers reach acquire
  // before Tile 15 publishes each of three distinct, immutable DDR resources.
  constexpr std::array<int64_t, 3> extents{1024, 1025, 1031};
  constexpr int64_t elements = 3080;
  auto config = ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  auto launch = RuntimeLaunchContract::createKernel(
      KernelLaunchForm::Grid, KernelEntryABI::TileMajorPointerTable,
      {RuntimeLaunchPhaseRole::Main});
  ASSERT_TRUE(static_cast<bool>(launch));
  auto *format = findTargetDataFormatCode(LogicalFormat::F16);
  ASSERT_NE(format, nullptr);
  std::vector<TargetLLVMModule> modules;
  std::vector<TargetCallTileArguments> arguments;
  for (int64_t tile = 0; tile < 16; ++tile) {
    std::vector<TileEntryArgument> slots;
    auto slot = [&](TileEntryArgumentKind kind, int64_t resource, int64_t count,
                    TileEntryArgumentAccess access, bool ready = false) {
      TileEntryArgument value{static_cast<int64_t>(slots.size()),
                              kind,
                              resource,
                              "",
                              ready ? LogicalFormat::U8 : LogicalFormat::F16,
                              MemLayout::Tensor,
                              ready ? std::vector<int64_t>{64}
                                    : std::vector<int64_t>{1, 1, count},
                              ready ? 64 : count * 2,
                              64,
                              access};
      value.zeroInitialize = ready;
      slots.push_back(std::move(value));
    };
    slot(TileEntryArgumentKind::ExternalInput, 0, elements,
         TileEntryArgumentAccess::ReadOnly);
    slot(TileEntryArgumentKind::ExternalOutput, 0, elements * 16,
         TileEntryArgumentAccess::WriteOnly);
    std::vector<uint64_t> addresses{0x10000000, 0x20000000};
    for (unsigned phase = 0; phase < extents.size(); ++phase) {
      auto access = tile == 15 ? TileEntryArgumentAccess::WriteOnly
                               : TileEntryArgumentAccess::ReadOnly;
      slot(TileEntryArgumentKind::SharedWorkspace, phase * 2, extents[phase],
           access);
      slot(TileEntryArgumentKind::SharedWorkspace, phase * 2 + 1, 64, access,
           true);
      addresses.push_back(0x30000000 + phase * 0x10000);
      addresses.push_back(0x40000000 + phase * 0x10000);
    }
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("ddr-publication", *context);
    auto *i64 = llvm::Type::getInt64Ty(*context);
    std::vector<llvm::Type *> types(slots.size(), i64);
    auto *entry = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context), types, false),
        llvm::GlobalValue::ExternalLinkage, "main", *module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(*context, "entry", entry));
    auto call = [&](TargetCallBuiltin builtin,
                    llvm::ArrayRef<llvm::Value *> values) {
      const auto &descriptor = getTargetCallDescriptor(builtin);
      std::vector<llvm::Type *> parameterTypes;
      for (auto scalar : descriptor.arguments)
        parameterTypes.push_back(
            scalar == TargetCallScalarType::I64 ? i64 : builder.getInt32Ty());
      auto callee = module->getOrInsertFunction(
          descriptor.symbol,
          llvm::FunctionType::get(builder.getVoidTy(), parameterTypes, false));
      builder.CreateCall(callee, values);
    };
    auto join = [&] {
      call(TargetCallBuiltin::NCCJoin, {builder.getInt32(1)});
    };
    auto dma = [&](TargetCallBuiltin kind, llvm::Value *source,
                   llvm::Value *destination, int64_t bytes) {
      call(kind,
           {source, destination, builder.getInt32(bytes),
            builder.getInt32(bytes), builder.getInt32(0), builder.getInt32(0),
            builder.getInt32(0), builder.getInt32(1), builder.getInt32(1),
            builder.getInt32(1), builder.getInt32(format->dataFormatCode),
            builder.getInt32(0)});
    };
    int64_t offset = 0;
    for (unsigned phase = 0; phase < extents.size(); ++phase) {
      int64_t bytes = extents[phase] * 2;
      auto *data = entry->getArg(2 + phase * 2);
      auto *ready = entry->getArg(3 + phase * 2);
      if (tile == 15) {
        dma(TargetCallBuiltin::RDMA,
            builder.CreateAdd(entry->getArg(0), builder.getInt64(offset)),
            builder.getInt64(0x10000), bytes);
        dma(TargetCallBuiltin::WDMA, builder.getInt64(0x10000), data, bytes);
#ifndef WAFER_TEST_DDR_LATE_JOIN
        join();
#endif
        call(TargetCallBuiltin::DDRPublish,
             {data, ready, builder.getInt64(bytes)});
      } else {
        call(TargetCallBuiltin::DDRAcquire,
             {data, ready, builder.getInt64(bytes)});
        dma(TargetCallBuiltin::RDMA, data, builder.getInt64(0x10000), bytes);
      }
      dma(TargetCallBuiltin::WDMA, builder.getInt64(0x10000),
          builder.CreateAdd(entry->getArg(1),
                            builder.getInt64(tile * elements * 2 + offset)),
          bytes);
      join();
      offset += bytes;
    }
    builder.CreateRetVoid();
    ASSERT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    modules.push_back(TargetLLVMModulesBuilder::makeModule(
        CardId(0), TileId(tile), LaunchSlotId(tile), "main",
        config->getTargetIdentityId(), kCurrentKernelRuntimeABI,
        kCurrentTargetModuleFormat, std::move(slots), std::move(context),
        std::move(module)));
    arguments.push_back(
        {CardId(0), TileId(tile), LaunchSlotId(tile), std::move(addresses)});
  }
  auto targets = TargetLLVMModulesBuilder::makeModules(
      *config, std::move(*launch), std::move(modules));
  auto executable = createTargetCallExecutable(targets, arguments);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  std::vector<uint8_t> input(elements * 2);
  for (unsigned index = 0; index < elements; ++index) {
    uint16_t value = 0x3c00 + index % 512;
    input[2 * index] = value & 255;
    input[2 * index + 1] = value >> 8;
  }
  TargetModelInputBinding binding{
      getTargetModelResourceId(CardId(0), TileId(0),
                               TileEntryArgumentKind::ExternalInput, 0),
      input};
  auto result = executeSystemCTargetModel(
      std::move(*executable), {binding},
      TargetModelKernelBudget::create(
          FormalNumericWorkBudget::create(1024, 1024), 1024 * 1024, 1024));
#ifdef WAFER_TEST_DDR_LATE_JOIN
  ASSERT_FALSE(static_cast<bool>(result));
  std::string error = llvm::toString(result.takeError());
  EXPECT_NE(error.find("pending NCC write"), std::string::npos) << error;
  EXPECT_EQ(error.find("no-progress"), std::string::npos) << error;
#else
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedTileCount, 16);
  ASSERT_EQ(result->outputs.size(), 1u);
  std::vector<uint8_t> expected;
  for (unsigned tile = 0; tile < 16; ++tile)
    expected.insert(expected.end(), input.begin(), input.end());
  EXPECT_EQ(result->outputs.front().bytes, expected);
#endif
}
} // namespace
extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
