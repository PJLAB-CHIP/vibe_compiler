//===- TargetModelMemoryTest.cpp - Private model memory tests ------------===//

#include "Wafer/Model/TargetModelMemory.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

template <typename T> std::string expectError(llvm::Expected<T> value) {
  if (value) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(value.takeError());
}

std::string expectError(llvm::Error error) {
  if (!error) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(std::move(error));
}

TargetCallInvocationDescriptor makeInvocation(size_t rankCount = 2) {
  std::vector<TargetCallRankDescriptor> ranks;
  for (size_t rank = 0; rank < rankCount; ++rank) {
    const uint64_t base = UINT64_C(0x100000) + rank * UINT64_C(0x10000);
    ranks.push_back(TargetCallRankDescriptor{
        static_cast<int64_t>(rank),
        {{0,
          KernelABISlotRole::UserInput,
          0,
          "ignored-input-name",
          "f32",
          {4},
          16,
          256},
         {1,
          KernelABISlotRole::Output,
          0,
          "ignored-output-name",
          "f32",
          {4},
          16,
          256},
         {2,
          KernelABISlotRole::Workspace,
          0,
          "ignored-workspace-name",
          "u8",
          {64},
          64,
          256}},
        {base, base + UINT64_C(0x1000), base + UINT64_C(0x2000)},
        TargetIdentityId::waferTx81SingleCard(),
        KernelRuntimeABIId::waferTx81KernelV1()});
  }
  return {TargetProfileId::waferTx81SingleCardKernelV1(), std::move(ranks)};
}

std::vector<TargetModelInputBinding> makeBindings(size_t rankCount = 2) {
  std::vector<TargetModelInputBinding> bindings;
  for (size_t rank = 0; rank < rankCount; ++rank)
    bindings.push_back(
        {static_cast<int64_t>(rank), 0,
         std::vector<uint8_t>(16, static_cast<uint8_t>(rank + 1))});
  return bindings;
}

InvocationMemoryRegistry makeRegistry(size_t rankCount = 2) {
  return llvm::cantFail(InvocationMemoryRegistry::create(
      llvm::cantFail(InvocationAddressPlan::create(makeInvocation(rankCount),
                                                   makeBindings(rankCount)))));
}

TEST(TargetModelMemoryTest, RequiresExactInputsAndOwnsPrivateSlotBytes) {
  TargetCallInvocationDescriptor invocation = makeInvocation(1);
  std::string error = expectError(
      InvocationAddressPlan::create(invocation, /*inputBindings=*/{}));
  EXPECT_NE(error.find("requires exact initial bytes"), std::string::npos);

  std::vector<TargetModelInputBinding> mutableBinding = makeBindings(1);
  mutableBinding.push_back({0, 1, std::vector<uint8_t>(16, 9)});
  error =
      expectError(InvocationAddressPlan::create(invocation, mutableBinding));
  EXPECT_NE(error.find("model-owned"), std::string::npos);

  InvocationMemoryRegistry registry = makeRegistry(1);
  EXPECT_EQ(llvm::cantFail(registry.readSlotSnapshot(0, 0)),
            std::vector<uint8_t>(16, 1));
  EXPECT_EQ(llvm::cantFail(registry.readSlotSnapshot(0, 1)),
            std::vector<uint8_t>(16, 0));

  error = expectError(registry.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::CardDDR, UINT64_C(0x100000), 1, {7}}}));
  EXPECT_NE(error.find("access-denied"), std::string::npos);
}

TEST(TargetModelMemoryTest, RankSPMIsPrivateAndReservedRangesFailClosed) {
  InvocationMemoryRegistry registry = makeRegistry();
  const uint64_t spmBase = registry.getAddressPlan().getSPMBase();
  const uint64_t spmLimit = registry.getAddressPlan().getSPMLimit();
  ASSERT_FALSE(registry.applyAtomically(
      {TargetModelByteWrite{
           0, TargetModelAddressSpace::RankSPM, spmBase, 1, {1, 2, 3, 4}},
       TargetModelByteWrite{
           1, TargetModelAddressSpace::RankSPM, spmBase, 1, {5, 6, 7, 8}}}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, spmBase, 4, 1)),
            (std::vector<uint8_t>{1, 2, 3, 4}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                1, TargetModelAddressSpace::RankSPM, spmBase, 4, 1)),
            (std::vector<uint8_t>{5, 6, 7, 8}));

  std::string error = expectError(registry.readSnapshot(
      0, TargetModelAddressSpace::RankSPM, spmBase - 1, 1, 1));
  EXPECT_NE(error.find("reserved-spm"), std::string::npos);
  error = expectError(registry.readSnapshot(0, TargetModelAddressSpace::RankSPM,
                                            spmLimit - 1, 2, 1));
  EXPECT_NE(error.find("reserved-spm"), std::string::npos);
}

TEST(TargetModelMemoryTest, ResolvesExactEndAndRejectsCrossResource) {
  InvocationMemoryRegistry registry = makeRegistry(1);
  EXPECT_EQ(
      llvm::cantFail(registry.readSnapshot(0, TargetModelAddressSpace::CardDDR,
                                           UINT64_C(0x100000) + 8, 8, 1)),
      std::vector<uint8_t>(8, 1));
  std::string error = expectError(registry.readSnapshot(
      0, TargetModelAddressSpace::CardDDR, UINT64_C(0x100000) + 8, 9, 1));
  EXPECT_NE(error.find("cross-resource"), std::string::npos);
  error = expectError(registry.readSnapshot(0, TargetModelAddressSpace::CardDDR,
                                            UINT64_C(0x100000), 1, 3));
  EXPECT_NE(error.find("address-misaligned"), std::string::npos);
  error = expectError(
      registry.readSnapshot(0, TargetModelAddressSpace::CardDDR,
                            std::numeric_limits<uint64_t>::max(), 2, 1));
  EXPECT_NE(error.find("address-overflow"), std::string::npos);
}

TEST(TargetModelMemoryTest, PendingEffectValidationIsAtomic) {
  InvocationMemoryRegistry registry = makeRegistry(1);
  const uint64_t output = UINT64_C(0x101000);
  const uint64_t spmLimit = registry.getAddressPlan().getSPMLimit();
  std::string error = expectError(registry.applyAtomically(
      {TargetModelByteWrite{
           0, TargetModelAddressSpace::CardDDR, output, 1, {9, 9, 9, 9}},
       TargetModelByteWrite{
           0, TargetModelAddressSpace::RankSPM, spmLimit, 1, {8}}}));
  EXPECT_NE(error.find("reserved-spm"), std::string::npos);
  EXPECT_EQ(llvm::cantFail(registry.readSlotSnapshot(0, 1)),
            std::vector<uint8_t>(16, 0));

  error = expectError(registry.applyAtomically(
      {TargetModelByteWrite{
           0, TargetModelAddressSpace::CardDDR, output, 1, {1, 2}},
       TargetModelByteWrite{
           0, TargetModelAddressSpace::CardDDR, output + 1, 1, {3, 4}}}));
  EXPECT_NE(error.find("invalid-effect"), std::string::npos);
  EXPECT_EQ(llvm::cantFail(registry.readSlotSnapshot(0, 1)),
            std::vector<uint8_t>(16, 0));
}

TEST(TargetModelMemoryTest, RejectsOverlappingSlotsAndIncompleteRankDomain) {
  TargetCallInvocationDescriptor overlapping = makeInvocation();
  overlapping.ranks[1].slotValues[0] = overlapping.ranks[0].slotValues[0];
  std::string error =
      expectError(InvocationAddressPlan::create(overlapping, makeBindings()));
  EXPECT_NE(error.find("overlaps"), std::string::npos);

  TargetCallInvocationDescriptor badRanks = makeInvocation();
  badRanks.ranks[1].logicalRank = 2;
  error = expectError(InvocationAddressPlan::create(badRanks, makeBindings()));
  EXPECT_NE(error.find("logical rank domain"), std::string::npos);
}

} // namespace
