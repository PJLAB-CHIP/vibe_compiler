//===- TargetProfileTest.cpp - Closed target profile registry tests -------===//

#include "Wafer/Target/TargetProfile.h"

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
  ASSERT_EQ(profiles.size(), 1u);
  const wafer::TargetProfileRecord &profile = profiles.front();
  EXPECT_EQ(profile.canonicalSpelling, "wafer-tx81-single-card-kernel-v1");
  EXPECT_EQ(profile.targetIdentitySpelling, "wafer-tx81-single-card");
  EXPECT_EQ(profile.kernelRuntimeABISpelling, "wafer-tx81-kernel-v1");
  EXPECT_EQ(profile.moduleFormat, "elf-riscv64");

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
  EXPECT_EQ(wafer::stringifyTargetIdentityId(*parsedIdentity),
            profile.targetIdentitySpelling);

  llvm::Expected<wafer::KernelRuntimeABIId> parsedABI =
      wafer::parseKernelRuntimeABIId(profile.kernelRuntimeABISpelling);
  ASSERT_TRUE(static_cast<bool>(parsedABI))
      << llvm::toString(parsedABI.takeError());
  EXPECT_EQ(*parsedABI, profile.kernelRuntimeABI);
  EXPECT_EQ(wafer::stringifyKernelRuntimeABIId(*parsedABI),
            profile.kernelRuntimeABISpelling);
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

} // namespace
