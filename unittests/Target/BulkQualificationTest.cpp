//===- BulkQualificationTest.cpp - Qualified oneDNN execution tests ------===//

#include "Wafer/Target/BulkQualification.h"

#include "gtest/gtest.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace wafer;

static_assert(!std::is_default_constructible_v<BulkNumericWorkBudget>);
static_assert(!std::is_default_constructible_v<BulkBackendAdmission>);
static_assert(!std::is_default_constructible_v<BulkExecutionEnvironment>);

constexpr FormalNumericWorkBudget kSmallFormalBudget =
    FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1000000,
                                    /*maximumFusedMultiplyAdds=*/1000000);
constexpr BulkNumericWorkBudget kBulkBudget =
    BulkNumericWorkBudget::create(/*maximumTotalBytes=*/UINT64_C(100000000),
                                  /*maximumScratchpadBytes=*/UINT64_C(50000000),
                                  /*maximumReorderBytes=*/UINT64_C(50000000));

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

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::error_code error =
        llvm::sys::fs::createUniqueDirectory("wafer-bulk-model", path);
    if (error)
      ADD_FAILURE() << error.message();
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path); }

  std::string getPath(llvm::StringRef name) const {
    llvm::SmallString<256> result(path);
    llvm::sys::path::append(result, name);
    return result.str().str();
  }

private:
  llvm::SmallString<256> path;
};

llvm::Error writeText(llvm::StringRef path, llvm::StringRef text) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::errorCodeToError(error);
  stream << text;
  stream.close();
  return llvm::Error::success();
}

struct QualifiedRow {
  VerifiedBulkQualificationRecord record;
  BulkQualificationCase row;
  BulkBackendAdmission admission;
  std::string policyPath;
  std::string recordPath;
};

llvm::Expected<QualifiedRow>
qualify(const BulkExecutionEnvironment &environment, TemporaryDirectory &files,
        LogicalFormat format, uint64_t m, uint64_t k, uint64_t n,
        uint64_t batchCount = 1,
        FormalNumericWorkBudget formalBudget = kSmallFormalBudget) {
  const NumericTensorLayout layout =
      batchCount == 1 ? NumericTensorLayout::Cx : NumericTensorLayout::NCx;
  llvm::Expected<BulkQualificationSpec> calibrationSpec =
      BulkQualificationSpec::create(format, m, k, n, batchCount, layout, layout,
                                    layout, /*seed=*/11);
  llvm::Expected<BulkQualificationSpec> heldOutSpec =
      BulkQualificationSpec::create(format, m, k, n, batchCount, layout, layout,
                                    layout, /*seed=*/29);
  if (!calibrationSpec)
    return calibrationSpec.takeError();
  if (!heldOutSpec)
    return heldOutSpec.takeError();

  const std::string calibrationSpecPath =
      files.getPath("calibration-spec.json");
  const std::string heldOutSpecPath = files.getPath("held-out-spec.json");
  const std::string calibrationPath = files.getPath("calibration.json");
  const std::string policyPath = files.getPath("policy.json");
  const std::string recordPath = files.getPath("record.json");
  if (llvm::Error error =
          writeBulkQualificationSpec(*calibrationSpec, calibrationSpecPath))
    return std::move(error);
  if (llvm::Error error =
          writeBulkQualificationSpec(*heldOutSpec, heldOutSpecPath))
    return std::move(error);
  if (llvm::Error error =
          calibrateBulkBackend(environment, calibrationSpecPath,
                               calibrationPath, formalBudget, kBulkBudget))
    return std::move(error);
  if (llvm::Error error = freezeBulkBackendPolicy(
          calibrationPath, heldOutSpecPath, policyPath,
          {/*maximumAbsoluteError=*/0.0, /*maximumRelativeError=*/0.0}))
    return std::move(error);
  if (llvm::Error error = validateBulkBackend(
          environment, policyPath, recordPath, formalBudget, kBulkBudget))
    return std::move(error);

  llvm::Expected<VerifiedBulkQualificationRecord> record =
      loadVerifiedBulkQualificationRecord(recordPath);
  if (!record)
    return record.takeError();
  llvm::Expected<BulkQualificationCase> row =
      materializeBulkQualificationCase(std::move(*heldOutSpec), kBulkBudget);
  if (!row)
    return row.takeError();
  llvm::Expected<BulkBackendAdmission> admission =
      record->createAdmission(environment, row->getCommand(), row->getInputs(),
                              row->getDestinationTemplate());
  if (!admission)
    return admission.takeError();
  return QualifiedRow{std::move(*record), std::move(*row),
                      std::move(*admission), policyPath, recordPath};
}

TEST(BulkQualificationTest, ManagedEnvironmentHasClosedReadbackIdentity) {
  llvm::Expected<BulkExecutionEnvironment> first =
      createManagedBulkExecutionEnvironment();
  ASSERT_TRUE(static_cast<bool>(first))
      << (first ? std::string() : llvm::toString(first.takeError()));
  llvm::Expected<BulkExecutionEnvironment> second =
      createManagedBulkExecutionEnvironment();
  ASSERT_TRUE(static_cast<bool>(second))
      << (second ? std::string() : llvm::toString(second.takeError()));
  EXPECT_EQ(first->getDigest(), second->getDigest());
  EXPECT_EQ(first->getBackend().getName(), "oneDNN");
  EXPECT_EQ(first->getBackend().getVersion(), "3.12");
  EXPECT_EQ(first->getBackend().getCommit().size(), 40u);
  EXPECT_TRUE(
      first->getBackend().getDependencyRecordDigest().starts_with("sha256:"));
  EXPECT_TRUE(first->getBackend().getLibraryDigest().starts_with("sha256:"));
  EXPECT_TRUE(first->getHostPlatformDigest().starts_with("sha256:"));
  EXPECT_EQ(first->getFloatingRoundingMode(), FE_TONEAREST);
  EXPECT_EQ(first->getThreadRuntime(), "seq-caller-worker-v1");
}

TEST(BulkQualificationTest,
     CallerControlDriftIsRejectedAndStickyFlagsAreRestored) {
  BulkExecutionEnvironment environment =
      llvm::cantFail(createManagedBulkExecutionEnvironment());
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  auto restore = llvm::make_scope_exit([&] { std::fesetenv(&saved); });
  ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
  EXPECT_NE(expectError(createManagedBulkExecutionEnvironment())
                .find("no longer matches"),
            std::string::npos);
  ASSERT_EQ(std::fesetenv(&saved), 0);

  TemporaryDirectory files;
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, LogicalFormat::F32, 4, 8, 5));
  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_INVALID);
  const int before = std::fetestexcept(FE_ALL_EXCEPT);
  llvm::Expected<BulkTensorNumericResult> result =
      executeAdmittedBulkTensorNumeric(
          environment, qualified.admission, qualified.row.getCommand(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kBulkBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), before);
}

TEST(BulkQualificationTest, PhysicalCodecPreservesCxTailAndRejectsWrongSize) {
  llvm::Expected<NumericTensorKey> key = NumericTensorKey::create(
      LogicalFormat::F16, NumericTensorLayout::Cx, {2, 3});
  ASSERT_TRUE(static_cast<bool>(key))
      << (key ? std::string() : llvm::toString(key.takeError()));
  llvm::Expected<uint64_t> physicalBytes = getBulkTensorPhysicalBytes(*key);
  ASSERT_TRUE(static_cast<bool>(physicalBytes))
      << (physicalBytes ? std::string()
                        : llvm::toString(physicalBytes.takeError()));
  EXPECT_GT(*physicalBytes, UINT64_C(2 * 3 * 2));

  std::vector<RawLogicalValue> values(
      6, RawLogicalValue{LogicalFormat::F16, UINT64_C(0)});
  llvm::Expected<BulkTensorStorage> packed =
      packBulkTensorLogicalValues(*key, values, UINT8_C(0xa5));
  ASSERT_TRUE(static_cast<bool>(packed))
      << (packed ? std::string() : llvm::toString(packed.takeError()));
  EXPECT_NE(std::find(packed->getStorage().begin(), packed->getStorage().end(),
                      UINT8_C(0xa5)),
            packed->getStorage().end());
  llvm::Expected<std::vector<RawLogicalValue>> unpacked =
      unpackBulkTensorLogicalValues(*packed);
  ASSERT_TRUE(static_cast<bool>(unpacked))
      << (unpacked ? std::string() : llvm::toString(unpacked.takeError()));
  ASSERT_EQ(unpacked->size(), values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    EXPECT_EQ((*unpacked)[index].format, values[index].format);
    EXPECT_EQ((*unpacked)[index].bits, values[index].bits);
  }

  std::vector<uint8_t> shortStorage(packed->getStorage().begin(),
                                    packed->getStorage().end() - 1);
  EXPECT_NE(expectError(BulkTensorStorage::create(*key, shortStorage))
                .find("invalid-physical-storage"),
            std::string::npos);
}

class BulkFormatQualificationTest
    : public testing::TestWithParam<LogicalFormat> {};

TEST_P(BulkFormatQualificationTest,
       ThreeStageQualificationIssuesExactExecutableAdmission) {
  TemporaryDirectory files;
  BulkExecutionEnvironment environment =
      llvm::cantFail(createManagedBulkExecutionEnvironment());
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, GetParam(), 4, 8, 5));
  EXPECT_EQ(qualified.record.getKind(), BulkQualificationKind::ProfileBounded);
  EXPECT_EQ(qualified.record.getSpecDigest(),
            qualified.row.getSpec().getDigest());
  EXPECT_EQ(qualified.record.getEnvironmentDigest(), environment.getDigest());

  llvm::Expected<BulkTensorNumericResult> result =
      executeAdmittedBulkTensorNumeric(
          environment, qualified.admission, qualified.row.getCommand(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kBulkBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(result->evidence.matmulInvocations, 1u);
  EXPECT_LE(result->evidence.reorderInvocations, 1u);
  EXPECT_EQ(result->evidence.formalFusedMultiplyAdds, 0u);
  EXPECT_FALSE(result->evidence.implementation.empty());
  EXPECT_FALSE(llvm::StringRef(result->evidence.implementation)
                   .contains_insensitive("ref"));
  EXPECT_TRUE(llvm::StringRef(result->evidence.resolvedDescriptorDigest)
                  .starts_with("sha256:"));
  EXPECT_EQ(computeBulkTensorStorageDigest(result->destination),
            qualified.admission.getExpectedBackendOutputDigest());
}

INSTANTIATE_TEST_SUITE_P(F16BF16F32, BulkFormatQualificationTest,
                         testing::Values(LogicalFormat::F16,
                                         LogicalFormat::BF16,
                                         LogicalFormat::F32));

TEST(BulkQualificationTest,
     LargeGemmExceedsRuntimeFormalBudgetButExecutesOnlyAdmittedMatmul) {
  TemporaryDirectory files;
  BulkExecutionEnvironment environment =
      llvm::cantFail(createManagedBulkExecutionEnvironment());
  constexpr uint64_t kFusedMultiplyAdds = UINT64_C(64 * 64 * 64);
  QualifiedRow qualified = llvm::cantFail(qualify(
      environment, files, LogicalFormat::F32, 64, 64, 64, 1,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1000000,
                                      /*maximumFusedMultiplyAdds=*/
                                      kFusedMultiplyAdds)));

  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      unpackBulkTensorLogicalValues(qualified.row.getInputs()[0]);
  llvm::Expected<std::vector<RawLogicalValue>> rhs =
      unpackBulkTensorLogicalValues(qualified.row.getInputs()[1]);
  ASSERT_TRUE(static_cast<bool>(lhs));
  ASSERT_TRUE(static_cast<bool>(rhs));
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*lhs, *rhs};
  FormalNumericExecutionContext formalContext;
  std::string formalError = expectError(executeFormalTensorNumeric(
      formalContext, qualified.row.getCommand(), views,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1000000,
                                      /*maximumFusedMultiplyAdds=*/1024)));
  EXPECT_NE(formalError.find("multiply-accumulate-work-budget-exceeded"),
            std::string::npos);
  EXPECT_FALSE(formalContext.getAggregateFlags().any());

  llvm::Expected<BulkTensorNumericResult> result =
      executeAdmittedBulkTensorNumeric(
          environment, qualified.admission, qualified.row.getCommand(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kBulkBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(result->evidence.matmulInvocations, 1u);
  EXPECT_EQ(result->evidence.formalFusedMultiplyAdds, 0u);
  EXPECT_EQ(computeBulkTensorStorageDigest(result->destination),
            qualified.admission.getExpectedBackendOutputDigest());
}

TEST(BulkQualificationTest, BatchedNCxRowUsesOneMatmulAndPreservesAdmission) {
  TemporaryDirectory files;
  BulkExecutionEnvironment environment =
      llvm::cantFail(createManagedBulkExecutionEnvironment());
  QualifiedRow qualified = llvm::cantFail(
      qualify(environment, files, LogicalFormat::F32, 3, 4, 5, 2));
  EXPECT_EQ(qualified.row.getInputs()[0].getKey().getLayout(),
            NumericTensorLayout::NCx);
  llvm::Expected<BulkTensorNumericResult> result =
      executeAdmittedBulkTensorNumeric(
          environment, qualified.admission, qualified.row.getCommand(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kBulkBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(result->evidence.matmulInvocations, 1u);
  EXPECT_EQ(result->evidence.formalFusedMultiplyAdds, 0u);
}

TEST(BulkQualificationTest,
     AdmissionAndBudgetsRejectBeforePublishingAnUnqualifiedResult) {
  TemporaryDirectory files;
  BulkExecutionEnvironment environment =
      llvm::cantFail(createManagedBulkExecutionEnvironment());
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, LogicalFormat::F32, 4, 8, 5));

  std::vector<BulkTensorStorage> tamperedInputs(
      qualified.row.getInputs().begin(), qualified.row.getInputs().end());
  std::vector<uint8_t> changed = tamperedInputs[0].getStorage().vec();
  changed[0] ^= UINT8_C(1);
  tamperedInputs[0] = llvm::cantFail(BulkTensorStorage::create(
      tamperedInputs[0].getKey(), std::move(changed)));
  EXPECT_NE(
      expectError(qualified.record.createAdmission(
                      environment, qualified.row.getCommand(), tamperedInputs,
                      qualified.row.getDestinationTemplate()))
          .find("does not exact-match"),
      std::string::npos);
  EXPECT_NE(
      expectError(executeAdmittedBulkTensorNumeric(
                      environment, qualified.admission,
                      qualified.row.getCommand(), tamperedInputs,
                      qualified.row.getDestinationTemplate(), kBulkBudget))
          .find("admission-mismatch"),
      std::string::npos);
  EXPECT_NE(
      expectError(executeAdmittedBulkTensorNumeric(
                      environment, qualified.admission,
                      qualified.row.getCommand(), qualified.row.getInputs(),
                      qualified.row.getDestinationTemplate(),
                      BulkNumericWorkBudget::create(1, 1, 1)))
          .find("total-byte-budget-exceeded"),
      std::string::npos);
}

TEST(BulkQualificationTest,
     SpecAndPublicationSchemasAreClosedCanonicalAndNoReplace) {
  TemporaryDirectory files;
  BulkQualificationSpec spec = llvm::cantFail(BulkQualificationSpec::create(
      LogicalFormat::F32, 2, 3, 4, 1, NumericTensorLayout::Cx,
      NumericTensorLayout::Cx, NumericTensorLayout::Cx, 7));
  const std::string goodPath = files.getPath("good.json");
  ASSERT_FALSE(static_cast<bool>(writeBulkQualificationSpec(spec, goodPath)));
  llvm::Expected<BulkQualificationSpec> loaded =
      loadBulkQualificationSpec(goodPath);
  ASSERT_TRUE(static_cast<bool>(loaded))
      << (loaded ? std::string() : llvm::toString(loaded.takeError()));
  EXPECT_EQ(loaded->getDigest(), spec.getDigest());
  EXPECT_FALSE(expectError(writeBulkQualificationSpec(spec, goodPath)).empty());
  EXPECT_NE(expectError(loadBulkQualificationSpec("good.json"))
                .find("paths must be absolute"),
            std::string::npos);
  const std::string linkPath = files.getPath("good-link.json");
  ASSERT_FALSE(
      static_cast<bool>(llvm::sys::fs::create_link(goodPath, linkPath)));
  EXPECT_NE(expectError(loadBulkQualificationSpec(linkPath)).find("alias"),
            std::string::npos);

  const std::string noncanonicalPath = files.getPath("noncanonical.json");
  ASSERT_FALSE(static_cast<bool>(writeText(
      noncanonicalPath,
      "{ \"batch_count\":1,\"destination_layout\":\"cx\","
      "\"format\":\"f32\",\"k\":3,\"lhs_layout\":\"cx\","
      "\"m\":2,\"n\":4,\"rhs_layout\":\"cx\","
      "\"schema\":\"wafer-bulk-qualification-spec-v1\",\"seed\":7}\n")));
  EXPECT_NE(expectError(loadBulkQualificationSpec(noncanonicalPath))
                .find("not in canonical form"),
            std::string::npos);

  const std::string unknownPath = files.getPath("unknown.json");
  ASSERT_FALSE(static_cast<bool>(writeText(
      unknownPath, "{\"batch_count\":1,\"destination_layout\":\"cx\","
                   "\"format\":\"f32\",\"k\":3,\"lhs_layout\":\"cx\","
                   "\"m\":2,\"n\":4,\"rhs_layout\":\"cx\","
                   "\"schema\":\"wafer-bulk-qualification-spec-v1\",\"seed\":7,"
                   "\"unknown\":false}\n")));
  EXPECT_NE(expectError(loadBulkQualificationSpec(unknownPath))
                .find("unknown, missing or duplicate fields"),
            std::string::npos);

  const std::string malformedPath = files.getPath("malformed.json");
  ASSERT_FALSE(static_cast<bool>(writeText(
      malformedPath, "{\"batch_count\":\"bad\",\"destination_layout\":\"bad\","
                     "\"format\":\"bad\",\"k\":\"bad\",\"lhs_layout\":\"bad\","
                     "\"m\":\"bad\",\"n\":\"bad\",\"rhs_layout\":\"bad\","
                     "\"schema\":\"wafer-bulk-qualification-spec-v1\","
                     "\"seed\":\"bad\"}\n")));
  std::string malformedError =
      expectError(loadBulkQualificationSpec(malformedPath));
  EXPECT_NE(malformedError.find("batch_count"), std::string::npos);
  EXPECT_NE(malformedError.find("field k"), std::string::npos);
  EXPECT_NE(malformedError.find("seed"), std::string::npos);
}

TEST(BulkQualificationTest,
     ExplicitPhysicalPayloadSpecRoundTripsWithoutRegeneration) {
  TemporaryDirectory files;
  BulkQualificationSpec generated =
      llvm::cantFail(BulkQualificationSpec::create(
          LogicalFormat::F32, 2, 3, 4, 1, NumericTensorLayout::Cx,
          NumericTensorLayout::Cx, NumericTensorLayout::Cx, 29));
  BulkQualificationCase generatedCase = llvm::cantFail(
      materializeBulkQualificationCase(std::move(generated), kBulkBudget));
  std::vector<uint8_t> lhs(generatedCase.getInputs()[0].getStorage().begin(),
                           generatedCase.getInputs()[0].getStorage().end());
  std::vector<uint8_t> rhs(generatedCase.getInputs()[1].getStorage().begin(),
                           generatedCase.getInputs()[1].getStorage().end());
  std::vector<uint8_t> destination(
      generatedCase.getDestinationTemplate().getStorage().begin(),
      generatedCase.getDestinationTemplate().getStorage().end());
  BulkQualificationSpec explicitSpec =
      llvm::cantFail(BulkQualificationSpec::createWithPhysicalPayload(
          LogicalFormat::F32, 2, 3, 4, 1, NumericTensorLayout::Cx,
          NumericTensorLayout::Cx, NumericTensorLayout::Cx, 31, lhs, rhs,
          destination));
  EXPECT_TRUE(explicitSpec.hasExplicitPhysicalPayload());
  const std::string path = files.getPath("explicit-spec.json");
  ASSERT_FALSE(
      static_cast<bool>(writeBulkQualificationSpec(explicitSpec, path)));
  BulkQualificationSpec loaded =
      llvm::cantFail(loadBulkQualificationSpec(path));
  EXPECT_EQ(loaded.getDigest(), explicitSpec.getDigest());
  EXPECT_TRUE(loaded.hasExplicitPhysicalPayload());
  BulkQualificationCase loadedCase = llvm::cantFail(
      materializeBulkQualificationCase(std::move(loaded), kBulkBudget));
  EXPECT_EQ(computeBulkTensorPayloadDigest(loadedCase.getInputs()),
            computeBulkTensorPayloadDigest(generatedCase.getInputs()));
  EXPECT_EQ(
      computeBulkTensorStorageDigest(loadedCase.getDestinationTemplate()),
      computeBulkTensorStorageDigest(generatedCase.getDestinationTemplate()));

  lhs.pop_back();
  EXPECT_NE(
      expectError(BulkQualificationSpec::createWithPhysicalPayload(
                      LogicalFormat::F32, 2, 3, 4, 1, NumericTensorLayout::Cx,
                      NumericTensorLayout::Cx, NumericTensorLayout::Cx, 31,
                      std::move(lhs), std::move(rhs), std::move(destination)))
          .find("byte geometry differs"),
      std::string::npos);
}

TEST(BulkQualificationTest, FinalRecordReadbackRejectsEvidenceTampering) {
  TemporaryDirectory files;
  BulkExecutionEnvironment environment =
      llvm::cantFail(createManagedBulkExecutionEnvironment());
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, LogicalFormat::F32, 4, 8, 5));
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(qualified.recordPath);
  ASSERT_TRUE(static_cast<bool>(buffer));
  std::string changed = (*buffer)->getBuffer().str();
  const std::string original = "\"matmul_invocations\":1";
  const size_t position = changed.find(original);
  ASSERT_NE(position, std::string::npos);
  changed.replace(position, original.size(), "\"matmul_invocations\":2");
  const std::string tamperedPath = files.getPath("tampered-record.json");
  ASSERT_FALSE(static_cast<bool>(writeText(tamperedPath, changed)));
  EXPECT_NE(expectError(loadVerifiedBulkQualificationRecord(tamperedPath))
                .find("does not prove one MatMul"),
            std::string::npos);
}

TEST(BulkQualificationTest, FreezeRequiresARegisteredDisjointHeldOutSeed) {
  TemporaryDirectory files;
  BulkExecutionEnvironment environment =
      llvm::cantFail(createManagedBulkExecutionEnvironment());
  BulkQualificationSpec spec = llvm::cantFail(BulkQualificationSpec::create(
      LogicalFormat::F32, 2, 3, 4, 1, NumericTensorLayout::Cx,
      NumericTensorLayout::Cx, NumericTensorLayout::Cx, 17));
  const std::string specPath = files.getPath("spec.json");
  const std::string calibrationPath = files.getPath("calibration.json");
  ASSERT_FALSE(static_cast<bool>(writeBulkQualificationSpec(spec, specPath)));
  ASSERT_FALSE(static_cast<bool>(
      calibrateBulkBackend(environment, specPath, calibrationPath,
                           kSmallFormalBudget, kBulkBudget)));
  std::string error = expectError(freezeBulkBackendPolicy(
      calibrationPath, specPath, files.getPath("policy.json"), {0.0, 0.0}));
  EXPECT_NE(error.find("have disjoint seeds/digests"), std::string::npos);
}

} // namespace
