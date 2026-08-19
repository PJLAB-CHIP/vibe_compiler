//===- TargetModelMemoryTest.cpp - Private model memory tests ------------===//

#include "Wafer/Model/Core/TargetModelMemory.h"

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

TargetCallInvocationDescriptor makeInvocation(size_t tileCount = 2) {
  std::vector<TargetCallTileDescriptor> tiles;
  for (size_t launchSlot = 0; launchSlot < tileCount; ++launchSlot) {
    const uint64_t workspace =
        UINT64_C(0x102000) + launchSlot * UINT64_C(0x10000);
    tiles.push_back(TargetCallTileDescriptor{
        CardId(0),
        TileId(static_cast<int64_t>(launchSlot)),
        LaunchSlotId(static_cast<int64_t>(launchSlot)),
        {{0,
          TileEntryArgumentKind::ExternalInput,
          0,
          "ignored-input-name",
          "f32",
          MemLayout::Tensor,
          {4},
          16,
          256,
          TileEntryArgumentAccess::ReadOnly},
         {1,
          TileEntryArgumentKind::ExternalOutput,
          0,
          "ignored-output-name",
          "f32",
          MemLayout::Tensor,
          {4},
          16,
          256,
          TileEntryArgumentAccess::WriteOnly},
         {2,
          TileEntryArgumentKind::Workspace,
          0,
          "ignored-workspace-name",
          "u8",
          MemLayout::Tensor,
          {64},
          64,
          256,
          TileEntryArgumentAccess::ReadWrite}},
        {UINT64_C(0x100000), UINT64_C(0x101000), workspace},
        TargetIdentityId::waferTx81SingleCard(),
        KernelRuntimeABIId::waferTx81Kernel()});
  }
  return {TargetIdentityId::waferTx81SingleCard(), std::move(tiles)};
}

std::vector<TargetModelInputBinding> makeBindings(size_t tileCount = 2) {
  (void)tileCount;
  return {{getTargetModelResourceId(CardId(0), TileId(0),
                                    TileEntryArgumentKind::ExternalInput,
                                    /*resourceIndex=*/0),
           std::vector<uint8_t>(16, 1)}};
}

InvocationMemoryRegistry makeRegistry(size_t tileCount = 2) {
  return llvm::cantFail(InvocationMemoryRegistry::create(
      llvm::cantFail(InvocationAddressPlan::create(makeInvocation(tileCount),
                                                   makeBindings(tileCount)))));
}

TEST(TargetModelMemoryTest, RequiresExactInputsAndOwnsPrivateSlotBytes) {
  TargetCallInvocationDescriptor invocation = makeInvocation(1);
  std::string error = expectError(
      InvocationAddressPlan::create(invocation, /*inputBindings=*/{}));
  EXPECT_NE(error.find("requires exact initial bytes"), std::string::npos);

  std::vector<TargetModelInputBinding> mutableBinding = makeBindings(1);
  mutableBinding.push_back(
      {getTargetModelResourceId(CardId(0), TileId(0),
                                TileEntryArgumentKind::ExternalOutput,
                                /*resourceIndex=*/0),
       std::vector<uint8_t>(16, 9)});
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

TEST(TargetModelMemoryTest, TileSPMIsPrivateAndReservedRangesFailClosed) {
  InvocationMemoryRegistry registry = makeRegistry();
  const uint64_t spmBase = registry.getAddressPlan().getSPMBase();
  const uint64_t spmLimit = registry.getAddressPlan().getSPMLimit();
  ASSERT_FALSE(registry.applyAtomically(
      {TargetModelByteWrite{
           0, TargetModelAddressSpace::TileSPM, spmBase, 1, {1, 2, 3, 4}},
       TargetModelByteWrite{
           1, TargetModelAddressSpace::TileSPM, spmBase, 1, {5, 6, 7, 8}}}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, spmBase, 4, 1)),
            (std::vector<uint8_t>{1, 2, 3, 4}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                1, TargetModelAddressSpace::TileSPM, spmBase, 4, 1)),
            (std::vector<uint8_t>{5, 6, 7, 8}));

  std::string error = expectError(registry.readSnapshot(
      0, TargetModelAddressSpace::TileSPM, spmBase - 1, 1, 1));
  EXPECT_NE(error.find("reserved-spm"), std::string::npos);
  error = expectError(registry.readSnapshot(0, TargetModelAddressSpace::TileSPM,
                                            spmLimit - 1, 2, 1));
  EXPECT_NE(error.find("reserved-spm"), std::string::npos);
}

TEST(TargetModelMemoryTest,
     CardResourcesAliasAcrossTilesWhileWorkspaceRemainsTileOwned) {
  InvocationMemoryRegistry registry = makeRegistry();
  const uint64_t output = UINT64_C(0x101000);
  ASSERT_FALSE(registry.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::CardDDR, output, 1, {1, 2, 3, 4}}}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                1, TargetModelAddressSpace::CardDDR, output, 4, 1)),
            (std::vector<uint8_t>{1, 2, 3, 4}));
  EXPECT_EQ(llvm::cantFail(registry.readSlotSnapshot(0, 1)),
            llvm::cantFail(registry.readSlotSnapshot(1, 1)));

  const uint64_t tile0Workspace = UINT64_C(0x102000);
  const uint64_t tile1Workspace = UINT64_C(0x112000);
  ASSERT_FALSE(registry.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::CardDDR, tile0Workspace, 1, {9}}}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                1, TargetModelAddressSpace::CardDDR, tile1Workspace, 1, 1)),
            (std::vector<uint8_t>{0}));
  std::string error = expectError(registry.readSnapshot(
      1, TargetModelAddressSpace::CardDDR, tile0Workspace, 1, 1));
  EXPECT_NE(error.find("unknown-resource"), std::string::npos);
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
           0, TargetModelAddressSpace::TileSPM, spmLimit, 1, {8}}}));
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

TEST(TargetModelMemoryTest,
     CompactStridedEffectPreservesCrossResourceAndAlignmentSemantics) {
  InvocationMemoryRegistry registry = makeRegistry(1);
  const uint64_t output = UINT64_C(0x101000);
  const uint64_t workspace = UINT64_C(0x102000);
  TargetModelStridedByteLayout crossResource{
      2, {UINT32_C(0x1000), 0, 0}, {2, 1, 1}};
  ASSERT_FALSE(registry.applyAtomically(
      {TargetModelByteWrite{0,
                            TargetModelAddressSpace::CardDDR,
                            output,
                            1,
                            {1, 2, 3, 4},
                            crossResource}}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                0, TargetModelAddressSpace::CardDDR, output, 2, 1)),
            (std::vector<uint8_t>{1, 2}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                0, TargetModelAddressSpace::CardDDR, workspace, 2, 1)),
            (std::vector<uint8_t>{3, 4}));
  EXPECT_EQ(llvm::cantFail(registry.readStridedSnapshot(
                0, TargetModelAddressSpace::CardDDR, output, crossResource, 1)),
            (std::vector<uint8_t>{1, 2, 3, 4}));

  const uint64_t spm = registry.getAddressPlan().getSPMBase();
  TargetModelStridedByteLayout misalignedSegments{1, {1, 0, 0}, {2, 1, 1}};
  std::string error = expectError(registry.applyAtomically(
      {TargetModelByteWrite{0,
                            TargetModelAddressSpace::TileSPM,
                            spm,
                            2,
                            {9, 8},
                            misalignedSegments}}));
  EXPECT_NE(error.find("address-misaligned"), std::string::npos);
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, spm, 2, 1)),
            (std::vector<uint8_t>{0, 0}));

  const uint64_t interleaved = spm + UINT64_C(0x100);
  TargetModelStridedByteLayout everyOtherByte{1, {2, 0, 0}, {2, 1, 1}};
  ASSERT_FALSE(registry.applyAtomically(
      {TargetModelByteWrite{0,
                            TargetModelAddressSpace::TileSPM,
                            interleaved,
                            1,
                            {1, 3},
                            everyOtherByte},
       TargetModelByteWrite{0,
                            TargetModelAddressSpace::TileSPM,
                            interleaved + 1,
                            1,
                            {2, 4},
                            everyOtherByte}}));
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, interleaved, 4, 1)),
            (std::vector<uint8_t>{1, 2, 3, 4}));

  error = expectError(registry.applyAtomically(
      {TargetModelByteWrite{0,
                            TargetModelAddressSpace::TileSPM,
                            interleaved,
                            1,
                            {8, 8},
                            everyOtherByte},
       TargetModelByteWrite{0,
                            TargetModelAddressSpace::TileSPM,
                            interleaved + 2,
                            1,
                            {9, 9},
                            everyOtherByte}}));
  EXPECT_NE(error.find("pending writes overlap"), std::string::npos);
  EXPECT_EQ(llvm::cantFail(registry.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, interleaved, 4, 1)),
            (std::vector<uint8_t>{1, 2, 3, 4}));
}

TEST(TargetModelMemoryTest,
     CompactStridedEffectRejectsOverflowAndLateFailureAtomically) {
  InvocationMemoryRegistry registry = makeRegistry();
  const uint64_t output = UINT64_C(0x101000);
  TargetModelStridedByteLayout overflowing{2, {4, 0, 0}, {2, 1, 1}};
  std::string error = expectError(registry.readStridedSnapshot(
      0, TargetModelAddressSpace::CardDDR,
      std::numeric_limits<uint64_t>::max() - 1, overflowing, 1));
  EXPECT_NE(error.find("address-overflow"), std::string::npos);
  TargetModelStridedByteLayout beyondHostVector{
      1,
      {0, 0, 0},
      {std::numeric_limits<uint32_t>::max(),
       std::numeric_limits<uint32_t>::max(), 1}};
  error = expectError(registry.readStridedSnapshot(
      0, TargetModelAddressSpace::CardDDR, output, beyondHostVector, 1));
  EXPECT_NE(error.find("host vector capacity"), std::string::npos);
  error = expectError(registry.applyAtomically(
      {TargetModelByteWrite{0,
                            TargetModelAddressSpace::CardDDR,
                            std::numeric_limits<uint64_t>::max() - 1,
                            1,
                            {1, 2, 3, 4},
                            overflowing}}));
  EXPECT_NE(error.find("address-overflow"), std::string::npos);

  // The bounding range crosses resources, so validation falls back to the
  // individual segments. The first segment is writable launch-slot-0 output,
  // while the last names a different Tile's private workspace and is not in
  // launch-slot-0's resource domain. Neither this descriptor nor the valid
  // write before it may modify any bytes.
  TargetModelStridedByteLayout lateReadOnly{
      2, {UINT32_C(0xf000), 0, 0}, {2, 1, 1}};
  error = expectError(registry.applyAtomically(
      {TargetModelByteWrite{
           0, TargetModelAddressSpace::CardDDR, output + 4, 1, {9, 9}},
       TargetModelByteWrite{0,
                            TargetModelAddressSpace::CardDDR,
                            output,
                            1,
                            {1, 2, 3, 4},
                            lateReadOnly}}));
  EXPECT_NE(error.find("unknown-resource"), std::string::npos);
  EXPECT_EQ(llvm::cantFail(registry.readSlotSnapshot(0, 1)),
            std::vector<uint8_t>(16, 0));
}

TEST(TargetModelMemoryTest, RejectsOverlappingSlotsAndIncompleteLaunchDomain) {
  TargetCallInvocationDescriptor splitSharedResource = makeInvocation();
  splitSharedResource.tiles[1].slotValues[0] = UINT64_C(0x120000);
  std::string error = expectError(
      InvocationAddressPlan::create(splitSharedResource, makeBindings()));
  EXPECT_NE(error.find("same resource"), std::string::npos);

  TargetCallInvocationDescriptor overlapping = makeInvocation();
  overlapping.tiles[1].slotValues[2] = overlapping.tiles[0].slotValues[0];
  error =
      expectError(InvocationAddressPlan::create(overlapping, makeBindings()));
  EXPECT_NE(error.find("overlapping"), std::string::npos);

  TargetCallInvocationDescriptor badSlots = makeInvocation();
  badSlots.tiles[1].launchSlotId = LaunchSlotId(2);
  error = expectError(InvocationAddressPlan::create(badSlots, makeBindings()));
  EXPECT_NE(error.find("launch-slot domain"), std::string::npos);
}

} // namespace
