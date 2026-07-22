//===- TargetProfileTest.cpp - Closed target profile registry tests -------===//

#include "Wafer/Target/TargetProfile.h"
#include "Wafer/Target/TargetLaunchABI.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <string>
#include <type_traits>

namespace {

TEST(TargetProfileTest, RegistryIsClosedCanonicalAndRoundTrips) {
  static_assert(!std::is_default_constructible_v<wafer::TargetProfileId>);
  static_assert(!std::is_default_constructible_v<wafer::TargetIdentityId>);
  static_assert(!std::is_default_constructible_v<wafer::KernelRuntimeABIId>);

  llvm::ArrayRef<wafer::TargetProfileRecord> profiles =
      wafer::getRegisteredTargetProfiles();
  ASSERT_EQ(profiles.size(), 2u);
  EXPECT_EQ(profiles[0].canonicalSpelling, "wafer-tx81-single-card-kernel-v1");
  EXPECT_EQ(profiles[0].kernelRuntimeABISpelling, "wafer-tx81-kernel-v1");
  EXPECT_EQ(profiles[1].canonicalSpelling, "wafer-tx81-single-card-kernel-v2");
  EXPECT_EQ(profiles[1].kernelRuntimeABISpelling, "wafer-tx81-kernel-v2");
  for (const wafer::TargetProfileRecord &profile : profiles) {
    EXPECT_EQ(profile.targetIdentitySpelling, "wafer-tx81-single-card");
    EXPECT_EQ(profile.moduleFormat, "elf-riscv64");
    EXPECT_EQ(profile.formatCompatibilityProfile,
              wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    EXPECT_EQ(profile.numericCompatibilityProfile,
              wafer::TargetProfileId::waferTx81SingleCardKernelV1());

    llvm::Expected<wafer::TargetProfileId> parsedProfile =
        wafer::parseTargetProfileId(profile.canonicalSpelling);
    ASSERT_TRUE(static_cast<bool>(parsedProfile))
        << llvm::toString(parsedProfile.takeError());
    EXPECT_EQ(*parsedProfile, profile.id);
    EXPECT_EQ(wafer::stringifyTargetProfileId(*parsedProfile),
              profile.canonicalSpelling);

    llvm::Expected<wafer::TargetIdentityId> parsedIdentity =
        wafer::parseTargetIdentityId(profile.targetIdentitySpelling);
    ASSERT_TRUE(static_cast<bool>(parsedIdentity))
        << llvm::toString(parsedIdentity.takeError());
    EXPECT_EQ(*parsedIdentity, profile.targetIdentity);

    llvm::Expected<wafer::KernelRuntimeABIId> parsedABI =
        wafer::parseKernelRuntimeABIId(profile.kernelRuntimeABISpelling);
    ASSERT_TRUE(static_cast<bool>(parsedABI))
        << llvm::toString(parsedABI.takeError());
    EXPECT_EQ(*parsedABI, profile.kernelRuntimeABI);
    EXPECT_EQ(wafer::stringifyKernelRuntimeABIId(*parsedABI),
              profile.kernelRuntimeABISpelling);
  }
}

TEST(TargetProfileTest, UnknownAndEmptySpellingsHaveNoFallback) {
  for (llvm::StringRef spelling : {llvm::StringRef(), llvm::StringRef("wafer"),
                                   llvm::StringRef("unknown-profile")}) {
    llvm::Expected<wafer::TargetProfileId> parsed =
        wafer::parseTargetProfileId(spelling);
    ASSERT_FALSE(static_cast<bool>(parsed));
    EXPECT_NE(llvm::toString(parsed.takeError()).find("unknown target profile"),
              std::string::npos);
  }

  llvm::Expected<wafer::TargetIdentityId> identity =
      wafer::parseTargetIdentityId("unknown-target");
  ASSERT_FALSE(static_cast<bool>(identity));
  EXPECT_NE(
      llvm::toString(identity.takeError()).find("unknown target identity"),
      std::string::npos);

  llvm::Expected<wafer::KernelRuntimeABIId> abi =
      wafer::parseKernelRuntimeABIId("unknown-abi");
  ASSERT_FALSE(static_cast<bool>(abi));
  EXPECT_NE(llvm::toString(abi.takeError()).find("unknown kernel runtime ABI"),
            std::string::npos);
}

TEST(TargetLaunchABITest, RegistryIsClosedCanonicalAndRoundTrips) {
  static_assert(!std::is_default_constructible_v<wafer::TargetLaunchABIId>);
  llvm::ArrayRef<wafer::TargetLaunchABIRecord> records =
      wafer::getRegisteredTargetLaunchABIs();
  ASSERT_EQ(records.size(), 4u);
  EXPECT_EQ(records[0].canonicalSpelling, "per-rank-pointer-block-v1");
  EXPECT_EQ(records[1].canonicalSpelling, "tx81-kernel-grid-pointer-table-v1");
  EXPECT_EQ(records[2].canonicalSpelling, "tx81-model-bootparam-v1");
  EXPECT_EQ(records[3].canonicalSpelling,
            "tx81-cluster-direct-dte-prepare-main-v1");
  for (const wafer::TargetLaunchABIRecord &record : records) {
    llvm::Expected<wafer::TargetLaunchABIId> parsed =
        wafer::parseTargetLaunchABIId(record.canonicalSpelling);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError());
    EXPECT_EQ(*parsed, record.id);
    EXPECT_EQ(wafer::stringifyTargetLaunchABIId(*parsed),
              record.canonicalSpelling);
  }

  llvm::Expected<wafer::TargetLaunchABIId> unknown =
      wafer::parseTargetLaunchABIId("unknown-launch-abi");
  ASSERT_FALSE(static_cast<bool>(unknown));
  EXPECT_NE(
      llvm::toString(unknown.takeError()).find("unknown target launch ABI"),
      std::string::npos);

  EXPECT_TRUE(wafer::isTargetLaunchABICompatible(
      wafer::TargetLaunchABIId::perRankPointerBlockV1(),
      wafer::TargetProfileId::waferTx81SingleCardKernelV2()));
  EXPECT_TRUE(wafer::isTargetLaunchABICompatible(
      wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1(),
      wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  EXPECT_TRUE(wafer::isTargetLaunchABICompatible(
      wafer::TargetLaunchABIId::tx81ModelBootParamV1(),
      wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  EXPECT_TRUE(wafer::isTargetLaunchABICompatible(
      wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1(),
      wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  EXPECT_FALSE(wafer::isTargetLaunchABICompatible(
      wafer::TargetLaunchABIId::tx81ModelBootParamV1(),
      wafer::TargetProfileId::waferTx81SingleCardKernelV2()));
  EXPECT_FALSE(wafer::isTargetLaunchABICompatible(
      wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1(),
      wafer::TargetProfileId::waferTx81SingleCardKernelV2()));
}

} // namespace
