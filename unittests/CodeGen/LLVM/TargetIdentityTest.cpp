//===- TargetIdentityTest.cpp - Current target identity tests ------------===//

#include "Wafer/Target/TargetIdentity.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

namespace {

TEST(TargetIdentityTest, CurrentTargetAndABIIdentifiersRoundTripExactly) {
  llvm::Expected<wafer::TargetIdentityId> target =
      wafer::parseTargetIdentityId("wafer-tx81-single-card");
  ASSERT_TRUE(static_cast<bool>(target));
  EXPECT_EQ(*target, wafer::TargetIdentityId::waferTx81SingleCard());
  EXPECT_EQ(wafer::stringifyTargetIdentityId(*target),
            "wafer-tx81-single-card");

  llvm::Expected<wafer::KernelRuntimeABIId> runtimeABI =
      wafer::parseKernelRuntimeABIId("wafer-tx81-kernel");
  ASSERT_TRUE(static_cast<bool>(runtimeABI));
  EXPECT_EQ(*runtimeABI, wafer::KernelRuntimeABIId::waferTx81Kernel());
  EXPECT_EQ(wafer::stringifyKernelRuntimeABIId(*runtimeABI),
            "wafer-tx81-kernel");
  EXPECT_EQ(wafer::kCurrentTargetModuleFormat, "elf-riscv64");
}

TEST(TargetIdentityTest, RejectsUnknownTargetAndABIIdentifiers) {
  llvm::Expected<wafer::TargetIdentityId> target =
      wafer::parseTargetIdentityId("unknown-target");
  ASSERT_FALSE(static_cast<bool>(target));
  EXPECT_NE(llvm::toString(target.takeError()).find("unknown target identity"),
            std::string::npos);

  llvm::Expected<wafer::KernelRuntimeABIId> runtimeABI =
      wafer::parseKernelRuntimeABIId("unknown-runtime-abi");
  ASSERT_FALSE(static_cast<bool>(runtimeABI));
  EXPECT_NE(
      llvm::toString(runtimeABI.takeError()).find("unknown kernel runtime ABI"),
      std::string::npos);
}

} // namespace
