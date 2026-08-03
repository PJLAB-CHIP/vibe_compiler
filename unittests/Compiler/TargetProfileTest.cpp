//===- TargetProfileTest.cpp - Closed target profile registry tests -------===//

#include "Wafer/Target/TargetProfile.h"
#include "Wafer/Target/RuntimeLaunchContract.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <string>
#include <type_traits>

namespace {

TEST(TargetProfileTest, RegistryIsClosedCanonicalAndRoundTrips) {
  static_assert(!std::is_default_constructible_v<wafer::TargetProfileId>);
  static_assert(!std::is_default_constructible_v<wafer::TargetIdentityId>);
  static_assert(!std::is_default_constructible_v<wafer::KernelRuntimeABIId>);

  llvm::ArrayRef<wafer::TargetProfileRecord> profiles =
      wafer::getRegisteredTargetProfiles();
  ASSERT_EQ(profiles.size(), 3u);
  EXPECT_EQ(profiles[0].canonicalSpelling, "wafer-tx81-single-card-kernel-v1");
  EXPECT_EQ(profiles[0].kernelRuntimeABISpelling, "wafer-tx81-kernel-v1");
  EXPECT_EQ(profiles[1].canonicalSpelling, "wafer-tx81-single-card-kernel-v2");
  EXPECT_EQ(profiles[1].kernelRuntimeABISpelling, "wafer-tx81-kernel-v2");
  EXPECT_EQ(profiles[2].canonicalSpelling, "wafer-tx81-single-card-kernel-v3");
  EXPECT_EQ(profiles[2].kernelRuntimeABISpelling, "wafer-tx81-kernel-v3");
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

TEST(RuntimeLaunchContractTest, ComponentsHaveCanonicalClosedSpellings) {
  static_assert(!std::is_default_constructible_v<wafer::RuntimeLaunchContract>);

  llvm::Expected<wafer::RuntimeLaunchKind> kernel =
      wafer::parseRuntimeLaunchKind("kernel");
  ASSERT_TRUE(static_cast<bool>(kernel));
  EXPECT_EQ(*kernel, wafer::RuntimeLaunchKind::Kernel);
  EXPECT_EQ(wafer::stringifyRuntimeLaunchKind(*kernel), "kernel");

  llvm::Expected<wafer::RuntimeLaunchKind> model =
      wafer::parseRuntimeLaunchKind("model");
  ASSERT_TRUE(static_cast<bool>(model));
  EXPECT_EQ(*model, wafer::RuntimeLaunchKind::Model);
  EXPECT_EQ(wafer::stringifyRuntimeLaunchKind(*model), "model");

  for (auto [spelling, expected] :
       std::array<std::pair<llvm::StringRef, wafer::KernelLaunchForm>, 3>{
           std::pair{"per-rank", wafer::KernelLaunchForm::PerRank},
           std::pair{"grid", wafer::KernelLaunchForm::Grid},
           std::pair{"cluster", wafer::KernelLaunchForm::Cluster}}) {
    llvm::Expected<wafer::KernelLaunchForm> parsed =
        wafer::parseKernelLaunchForm(spelling);
    ASSERT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(*parsed, expected);
    EXPECT_EQ(wafer::stringifyKernelLaunchForm(*parsed), spelling);
  }

  for (auto [spelling, expected] :
       std::array<std::pair<llvm::StringRef, wafer::KernelEntryABI>, 3>{
           std::pair{"rank-local-pointer-block-v1",
                     wafer::KernelEntryABI::RankLocalPointerBlockV1},
           std::pair{"rank-major-pointer-table-v1",
                     wafer::KernelEntryABI::RankMajorPointerTableV1},
           std::pair{"rank-row-pointer-table-v1",
                     wafer::KernelEntryABI::RankRowPointerTableV1}}) {
    llvm::Expected<wafer::KernelEntryABI> parsed =
        wafer::parseKernelEntryABI(spelling);
    ASSERT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(*parsed, expected);
    EXPECT_EQ(wafer::stringifyKernelEntryABI(*parsed), spelling);
  }

  llvm::Expected<wafer::ModelEntryABI> modelABI =
      wafer::parseModelEntryABI("tx81-model-bootparam-v1");
  ASSERT_TRUE(static_cast<bool>(modelABI));
  EXPECT_EQ(*modelABI, wafer::ModelEntryABI::Tx81ModelBootParamV1);
  EXPECT_EQ(wafer::stringifyModelEntryABI(*modelABI),
            "tx81-model-bootparam-v1");

  for (auto [spelling, expected] :
       std::array<std::pair<llvm::StringRef, wafer::RuntimeLaunchPhaseRole>, 2>{
           std::pair{"prepare", wafer::RuntimeLaunchPhaseRole::Prepare},
           std::pair{"main", wafer::RuntimeLaunchPhaseRole::Main}}) {
    llvm::Expected<wafer::RuntimeLaunchPhaseRole> parsed =
        wafer::parseRuntimeLaunchPhaseRole(spelling);
    ASSERT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(*parsed, expected);
    EXPECT_EQ(wafer::stringifyRuntimeLaunchPhaseRole(*parsed), spelling);
  }

  auto expectRejected = [](auto parsed) {
    EXPECT_FALSE(static_cast<bool>(parsed));
    if (!parsed)
      llvm::consumeError(parsed.takeError());
  };
  expectRejected(wafer::parseRuntimeLaunchKind("unknown-launch-kind"));
  expectRejected(wafer::parseRuntimeLaunchKind("direct-dte"));
  expectRejected(wafer::parseRuntimeLaunchKind("cluster"));
  expectRejected(wafer::parseRuntimeLaunchKind("grid"));
  expectRejected(wafer::parseRuntimeLaunchKind("per-rank"));
  expectRejected(wafer::parseKernelLaunchForm("unknown-form"));
  expectRejected(wafer::parseKernelEntryABI("unknown-entry-abi"));
  expectRejected(wafer::parseModelEntryABI("unknown-model-abi"));
  expectRejected(wafer::parseRuntimeLaunchPhaseRole("unknown-phase"));
}

TEST(RuntimeLaunchContractTest, FactoriesAdmitOnlySupportedCrossProducts) {
  constexpr std::array main{wafer::RuntimeLaunchPhaseRole::Main};
  constexpr std::array prepareMain{wafer::RuntimeLaunchPhaseRole::Prepare,
                                   wafer::RuntimeLaunchPhaseRole::Main};

  llvm::Expected<wafer::RuntimeLaunchContract> perRank =
      wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::PerRank,
          wafer::KernelEntryABI::RankLocalPointerBlockV1, main);
  ASSERT_TRUE(static_cast<bool>(perRank));
  ASSERT_NE(perRank->getKernel(), nullptr);
  EXPECT_EQ(perRank->getKind(), wafer::RuntimeLaunchKind::Kernel);
  EXPECT_EQ(perRank->getKernel()->form, wafer::KernelLaunchForm::PerRank);
  EXPECT_EQ(perRank->getPhases(), llvm::ArrayRef(main));

  llvm::Expected<wafer::RuntimeLaunchContract> grid =
      wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::Grid,
          wafer::KernelEntryABI::RankMajorPointerTableV1, main);
  ASSERT_TRUE(static_cast<bool>(grid));
  EXPECT_EQ(grid->getKernel()->form, wafer::KernelLaunchForm::Grid);

  llvm::Expected<wafer::RuntimeLaunchContract> cluster =
      wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::Cluster,
          wafer::KernelEntryABI::RankMajorPointerTableV1, prepareMain);
  ASSERT_TRUE(static_cast<bool>(cluster));
  EXPECT_EQ(cluster->getPhases(), llvm::ArrayRef(prepareMain));

  llvm::Expected<wafer::RuntimeLaunchContract> gridRows =
      wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::Grid,
          wafer::KernelEntryABI::RankRowPointerTableV1, main);
  ASSERT_TRUE(static_cast<bool>(gridRows));
  EXPECT_EQ(gridRows->getKernel()->entryABI,
            wafer::KernelEntryABI::RankRowPointerTableV1);

  llvm::Expected<wafer::RuntimeLaunchContract> clusterRows =
      wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::Cluster,
          wafer::KernelEntryABI::RankRowPointerTableV1, prepareMain);
  ASSERT_TRUE(static_cast<bool>(clusterRows));
  EXPECT_EQ(clusterRows->getKernel()->entryABI,
            wafer::KernelEntryABI::RankRowPointerTableV1);

  llvm::Expected<wafer::RuntimeLaunchContract> model =
      wafer::RuntimeLaunchContract::createModel(
          wafer::ModelEntryABI::Tx81ModelBootParamV1, main);
  ASSERT_TRUE(static_cast<bool>(model));
  EXPECT_EQ(model->getKind(), wafer::RuntimeLaunchKind::Model);
  ASSERT_NE(model->getModel(), nullptr);

  auto expectRejected = [](auto contract) {
    EXPECT_FALSE(static_cast<bool>(contract));
    if (!contract)
      llvm::consumeError(contract.takeError());
  };
  expectRejected(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::PerRank,
      wafer::KernelEntryABI::RankMajorPointerTableV1, main));
  expectRejected(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::PerRank,
      wafer::KernelEntryABI::RankRowPointerTableV1, main));
  expectRejected(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Grid,
      wafer::KernelEntryABI::RankLocalPointerBlockV1, main));
  expectRejected(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Grid,
      wafer::KernelEntryABI::RankMajorPointerTableV1, prepareMain));
  expectRejected(wafer::RuntimeLaunchContract::createKernel(
      wafer::KernelLaunchForm::Cluster,
      wafer::KernelEntryABI::RankMajorPointerTableV1, main));
  expectRejected(wafer::RuntimeLaunchContract::createModel(
      wafer::ModelEntryABI::Tx81ModelBootParamV1, prepareMain));

  EXPECT_TRUE(wafer::isRuntimeLaunchContractCompatible(
      *perRank, wafer::TargetProfileId::waferTx81SingleCardKernelV2()));
  EXPECT_TRUE(wafer::isRuntimeLaunchContractCompatible(
      *grid, wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  EXPECT_TRUE(wafer::isRuntimeLaunchContractCompatible(
      *cluster, wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  EXPECT_TRUE(wafer::isRuntimeLaunchContractCompatible(
      *model, wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
  EXPECT_FALSE(wafer::isRuntimeLaunchContractCompatible(
      *grid, wafer::TargetProfileId::waferTx81SingleCardKernelV2()));
  EXPECT_FALSE(wafer::isRuntimeLaunchContractCompatible(
      *cluster, wafer::TargetProfileId::waferTx81SingleCardKernelV2()));
  EXPECT_FALSE(wafer::isRuntimeLaunchContractCompatible(
      *model, wafer::TargetProfileId::waferTx81SingleCardKernelV2()));
  EXPECT_TRUE(wafer::isRuntimeLaunchContractCompatible(
      *grid, wafer::TargetProfileId::waferTx81SingleCardKernelV3()));
  EXPECT_TRUE(wafer::isRuntimeLaunchContractCompatible(
      *cluster, wafer::TargetProfileId::waferTx81SingleCardKernelV3()));
  EXPECT_TRUE(wafer::isRuntimeLaunchContractCompatible(
      *model, wafer::TargetProfileId::waferTx81SingleCardKernelV3()));
}

} // namespace
