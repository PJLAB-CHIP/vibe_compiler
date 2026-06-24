#include "Wafer/ABI/TileAbi.h"

#include "gtest/gtest.h"

namespace {

TEST(TileAbiTest, BuildsRdmaDescriptorWithByteRanges) {
  wafer::abi::Dma1DDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildRdma1D(
      /*ddrSrcAddr=*/0x280000000ULL, /*spmDstOffset=*/0,
      /*bytes=*/64, descriptor, &error))
      << error;

  EXPECT_EQ(descriptor.ddrAddress, 0x280000000ULL);
  EXPECT_EQ(descriptor.spmOffset, 0u);
  EXPECT_EQ(descriptor.bytes, 64u);
  EXPECT_EQ(descriptor.ddrEndExclusive, 0x280000040ULL);
  EXPECT_EQ(descriptor.spmEndExclusive, 64u);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsWdmaDescriptorWithByteRanges) {
  wafer::abi::Dma1DDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildWdma1D(
      /*spmSrcOffset=*/2048, /*ddrDstAddr=*/0x280001000ULL,
      /*bytes=*/128, descriptor, &error))
      << error;

  EXPECT_EQ(descriptor.ddrAddress, 0x280001000ULL);
  EXPECT_EQ(descriptor.spmOffset, 2048u);
  EXPECT_EQ(descriptor.bytes, 128u);
  EXPECT_EQ(descriptor.ddrEndExclusive, 0x280001080ULL);
  EXPECT_EQ(descriptor.spmEndExclusive, 2176u);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsGemmDescriptor) {
  wafer::abi::GemmDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildGemm(/*m=*/4, /*k=*/8, /*n=*/16, descriptor,
                                    &error))
      << error;

  EXPECT_EQ(descriptor.m, 4);
  EXPECT_EQ(descriptor.k, 8);
  EXPECT_EQ(descriptor.n, 16);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsStridedDmaDescriptor) {
  wafer::abi::DmaDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000200ULL, /*spmDstOffset=*/65536,
      /*byteCount=*/512, /*innerBytes=*/64,
      /*strides=*/{128, 0, 0}, /*iterations=*/{4, 1, 1}, descriptor,
      &error))
      << error;

  EXPECT_EQ(descriptor.ddrAddress, 0x280000200ULL);
  EXPECT_EQ(descriptor.spmOffset, 65536u);
  EXPECT_EQ(descriptor.byteCount, 512u);
  EXPECT_EQ(descriptor.innerBytes, 64u);
  EXPECT_EQ(descriptor.strides[0], 128);
  EXPECT_EQ(descriptor.iterations[0], 4);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, RejectsInvalidStridedDmaDescriptor) {
  wafer::abi::DmaDescriptor descriptor;
  std::string error;

  EXPECT_FALSE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000000ULL, /*spmDstOffset=*/0,
      /*byteCount=*/64, /*innerBytes=*/0,
      /*strides=*/{16, 0, 0}, /*iterations=*/{1, 1, 1}, descriptor,
      &error));
  EXPECT_NE(error.find("DMA inner byte count must be positive"),
            std::string::npos);

  error.clear();
  EXPECT_FALSE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000000ULL, /*spmDstOffset=*/0,
      /*byteCount=*/64, /*innerBytes=*/128,
      /*strides=*/{16, 0, 0}, /*iterations=*/{1, 1, 1}, descriptor,
      &error));
  EXPECT_NE(error.find("DMA inner byte count must not exceed byte count"),
            std::string::npos);
}

TEST(TileAbiTest, RejectsInvalidDmaRanges) {
  wafer::abi::Dma1DDescriptor descriptor;
  std::string error;

  EXPECT_FALSE(wafer::abi::buildRdma1D(
      /*ddrSrcAddr=*/0x27fffffffULL, /*spmDstOffset=*/0,
      /*bytes=*/64, descriptor, &error));
  EXPECT_NE(error.find("DDR address below lower bound"), std::string::npos);

  error.clear();
  EXPECT_FALSE(wafer::abi::buildWdma1D(
      /*spmSrcOffset=*/0x2efff0, /*ddrDstAddr=*/0x280000000ULL,
      /*bytes=*/32, descriptor, &error));
  EXPECT_NE(error.find("SPM range exceeds usable capacity"), std::string::npos);
}

} // namespace
