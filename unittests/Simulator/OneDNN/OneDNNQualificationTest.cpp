//===- OneDNNQualificationTest.cpp - Qualified oneDNN execution tests
//------===//

#include "Wafer/Simulator/OneDNN/OneDNNQualification.h"

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

static_assert(!std::is_default_constructible_v<OneDNNNumericWorkBudget>);
static_assert(!std::is_default_constructible_v<QualifiedOneDNNExecution>);
static_assert(!std::is_default_constructible_v<OneDNNExecutionEnvironment>);

constexpr FormalNumericWorkBudget kSmallFormalBudget =
    FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1000000,
                                    /*maximumFusedMultiplyAdds=*/1000000);
constexpr OneDNNNumericWorkBudget kOneDNNBudget =
    OneDNNNumericWorkBudget::create(
        /*maximumTotalBytes=*/UINT64_C(100000000),
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
        llvm::sys::fs::createUniqueDirectory("wafer-onednn", path);
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
  VerifiedOneDNNQualificationRecord record;
  OneDNNQualificationCase row;
  QualifiedOneDNNExecution execution;
  std::string policyPath;
  std::string recordPath;
};

llvm::Expected<QualifiedRow>
qualify(const OneDNNExecutionEnvironment &environment,
        TemporaryDirectory &files, LogicalFormat format, uint64_t m, uint64_t k,
        uint64_t n, uint64_t batchCount = 1,
        FormalNumericWorkBudget formalBudget = kSmallFormalBudget) {
  const PhysicalTensorLayout layout =
      batchCount == 1 ? PhysicalTensorLayout::Cx : PhysicalTensorLayout::NCx;
  llvm::Expected<OneDNNQualificationSpec> calibrationSpec =
      OneDNNQualificationSpec::create(format, m, k, n, batchCount, layout,
                                      layout, layout, /*seed=*/11);
  llvm::Expected<OneDNNQualificationSpec> heldOutSpec =
      OneDNNQualificationSpec::create(format, m, k, n, batchCount, layout,
                                      layout, layout, /*seed=*/29);
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
          writeOneDNNQualificationSpec(*calibrationSpec, calibrationSpecPath))
    return std::move(error);
  if (llvm::Error error =
          writeOneDNNQualificationSpec(*heldOutSpec, heldOutSpecPath))
    return std::move(error);
  if (llvm::Error error =
          calibrateOneDNNBackend(environment, calibrationSpecPath,
                                 calibrationPath, formalBudget, kOneDNNBudget))
    return std::move(error);
  if (llvm::Error error = freezeOneDNNBackendPolicy(
          calibrationPath, heldOutSpecPath, policyPath,
          {/*maximumAbsoluteError=*/0.0, /*maximumRelativeError=*/0.0}))
    return std::move(error);
  if (llvm::Error error = validateOneDNNBackend(
          environment, policyPath, recordPath, formalBudget, kOneDNNBudget))
    return std::move(error);

  llvm::Expected<VerifiedOneDNNQualificationRecord> record =
      loadVerifiedOneDNNQualificationRecord(recordPath);
  if (!record)
    return record.takeError();
  llvm::Expected<OneDNNQualificationCase> row =
      materializeOneDNNQualificationCase(std::move(*heldOutSpec),
                                         kOneDNNBudget);
  if (!row)
    return row.takeError();
  llvm::Expected<QualifiedOneDNNExecution> execution =
      record->qualifyExecution(environment, row->getOperation(),
                               row->getInputs(), row->getDestinationTemplate());
  if (!execution)
    return execution.takeError();
  return QualifiedRow{std::move(*record), std::move(*row),
                      std::move(*execution), policyPath, recordPath};
}

TEST(OneDNNQualificationTest, ManagedEnvironmentHasClosedReadbackIdentity) {
  llvm::Expected<OneDNNExecutionEnvironment> first =
      createManagedOneDNNExecutionEnvironment();
  ASSERT_TRUE(static_cast<bool>(first))
      << (first ? std::string() : llvm::toString(first.takeError()));
  llvm::Expected<OneDNNExecutionEnvironment> second =
      createManagedOneDNNExecutionEnvironment();
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
  EXPECT_EQ(first->getThreadRuntime(), "sequential-caller-worker");
}

TEST(OneDNNQualificationTest, PartialCannotReuseTwoInputQualification) {
  auto spec = llvm::cantFail(OneDNNQualificationSpec::create(
      LogicalFormat::F16, 2, 1025, 4, 2, PhysicalTensorLayout::NCx,
      PhysicalTensorLayout::NCx, PhysicalTensorLayout::NCx, 11));
  auto row = llvm::cantFail(
      materializeOneDNNQualificationCase(std::move(spec), kOneDNNBudget));
  auto operation = row.getOperation();
  operation.psum = llvm::cantFail(PhysicalTensorDescriptor::create(
      LogicalFormat::F32, PhysicalTensorLayout::NCx, {2, 2, 4}));
  EXPECT_NE(expectError(computeOneDNNGemmProblemDigest(operation))
                .find("no psum qualification"),
            std::string::npos);
  auto environment = llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  EXPECT_NE(expectError(executeManagedReferenceOneDNNTensorNumeric(
                            environment, operation, row.getInputs(),
                            row.getDestinationTemplate(), kOneDNNBudget))
                .find("no psum qualification"),
            std::string::npos);
}

TEST(OneDNNQualificationTest,
     CallerControlDriftIsRejectedAndStickyFlagsAreRestored) {
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  auto restore = llvm::make_scope_exit([&] { std::fesetenv(&saved); });
  ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
  EXPECT_NE(expectError(createManagedOneDNNExecutionEnvironment())
                .find("no longer matches"),
            std::string::npos);
  ASSERT_EQ(std::fesetenv(&saved), 0);

  TemporaryDirectory files;
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, LogicalFormat::F32, 4, 8, 5));
  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_INVALID);
  const int before = std::fetestexcept(FE_ALL_EXCEPT);
  llvm::Expected<OneDNNTensorNumericResult> result =
      executeQualifiedOneDNNTensorNumeric(
          environment, qualified.execution, qualified.row.getOperation(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kOneDNNBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), before);
}

TEST(OneDNNQualificationTest, PhysicalCodecPreservesCxTailAndRejectsWrongSize) {
  llvm::Expected<PhysicalTensorDescriptor> key =
      PhysicalTensorDescriptor::create(LogicalFormat::F16,
                                       PhysicalTensorLayout::Cx, {2, 3});
  ASSERT_TRUE(static_cast<bool>(key))
      << (key ? std::string() : llvm::toString(key.takeError()));
  llvm::Expected<uint64_t> physicalBytes = getOneDNNTensorPhysicalBytes(*key);
  ASSERT_TRUE(static_cast<bool>(physicalBytes))
      << (physicalBytes ? std::string()
                        : llvm::toString(physicalBytes.takeError()));
  EXPECT_GT(*physicalBytes, UINT64_C(2 * 3 * 2));

  std::vector<RawLogicalValue> values(
      6, RawLogicalValue{LogicalFormat::F16, UINT64_C(0)});
  llvm::Expected<OneDNNTensorStorage> packed =
      packOneDNNTensorLogicalValues(*key, values, UINT8_C(0xa5));
  ASSERT_TRUE(static_cast<bool>(packed))
      << (packed ? std::string() : llvm::toString(packed.takeError()));
  EXPECT_NE(std::find(packed->getStorage().begin(), packed->getStorage().end(),
                      UINT8_C(0xa5)),
            packed->getStorage().end());
  llvm::Expected<std::vector<RawLogicalValue>> unpacked =
      unpackOneDNNTensorLogicalValues(*packed);
  ASSERT_TRUE(static_cast<bool>(unpacked))
      << (unpacked ? std::string() : llvm::toString(unpacked.takeError()));
  ASSERT_EQ(unpacked->size(), values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    EXPECT_EQ((*unpacked)[index].format, values[index].format);
    EXPECT_EQ((*unpacked)[index].bits, values[index].bits);
  }

  std::vector<uint8_t> shortStorage(packed->getStorage().begin(),
                                    packed->getStorage().end() - 1);
  EXPECT_NE(expectError(OneDNNTensorStorage::create(*key, shortStorage))
                .find("invalid-physical-storage"),
            std::string::npos);
}

class OneDNNFormatQualificationTest
    : public testing::TestWithParam<LogicalFormat> {};

TEST_P(OneDNNFormatQualificationTest,
       ThreeStageQualificationProducesExactExecution) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, GetParam(), 4, 8, 5));
  EXPECT_EQ(qualified.record.getKind(),
            OneDNNQualificationKind::ProfileBounded);
  EXPECT_EQ(qualified.record.getSpecDigest(),
            qualified.row.getSpec().getDigest());
  EXPECT_EQ(qualified.record.getEnvironmentDigest(), environment.getDigest());

  llvm::Expected<OneDNNTensorNumericResult> result =
      executeQualifiedOneDNNTensorNumeric(
          environment, qualified.execution, qualified.row.getOperation(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kOneDNNBudget);
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
  EXPECT_EQ(result->evidence.implementation,
            qualified.execution.getExpectedImplementation());
  EXPECT_EQ(result->evidence.resolvedDescriptorDigest,
            qualified.execution.getExpectedResolvedDescriptorDigest());
  EXPECT_EQ(computeOneDNNTensorStorageDigest(result->destination),
            qualified.execution.getExpectedBackendOutputDigest());
}

INSTANTIATE_TEST_SUITE_P(F16BF16F32, OneDNNFormatQualificationTest,
                         testing::Values(LogicalFormat::F16,
                                         LogicalFormat::BF16,
                                         LogicalFormat::F32));

TEST(OneDNNQualificationTest,
     RawExactComparisonDoesNotTreatPhysicalPaddingAsTensorPayload) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  // Cx f16 [4, 16] has physical padding in the current target codec.  oneDNN
  // may write that padding even when every logical result bit is exact.
  QualifiedRow qualified = llvm::cantFail(
      qualify(environment, files, LogicalFormat::F16, 4, 16, 16));
  EXPECT_EQ(qualified.record.getKind(),
            OneDNNQualificationKind::ProfileBounded);
}

TEST(OneDNNQualificationTest,
     LargeGemmExceedsRuntimeFormalBudgetButExecutesQualifiedMatmul) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  constexpr uint64_t kFusedMultiplyAdds = UINT64_C(64 * 64 * 64);
  QualifiedRow qualified = llvm::cantFail(qualify(
      environment, files, LogicalFormat::F32, 64, 64, 64, 1,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1000000,
                                      /*maximumFusedMultiplyAdds=*/
                                      kFusedMultiplyAdds)));

  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      unpackOneDNNTensorLogicalValues(qualified.row.getInputs()[0]);
  llvm::Expected<std::vector<RawLogicalValue>> rhs =
      unpackOneDNNTensorLogicalValues(qualified.row.getInputs()[1]);
  ASSERT_TRUE(static_cast<bool>(lhs));
  ASSERT_TRUE(static_cast<bool>(rhs));
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*lhs, *rhs};
  FormalNumericExecutionContext formalContext;
  std::string formalError = expectError(executeFormalTensorNumeric(
      formalContext, qualified.row.getOperation(), views,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1000000,
                                      /*maximumFusedMultiplyAdds=*/1024)));
  EXPECT_NE(formalError.find("multiply-accumulate-work-budget-exceeded"),
            std::string::npos);
  EXPECT_FALSE(formalContext.getAggregateFlags().any());

  llvm::Expected<OneDNNTensorNumericResult> result =
      executeQualifiedOneDNNTensorNumeric(
          environment, qualified.execution, qualified.row.getOperation(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kOneDNNBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(result->evidence.matmulInvocations, 1u);
  EXPECT_EQ(result->evidence.formalFusedMultiplyAdds, 0u);
  EXPECT_EQ(computeOneDNNTensorStorageDigest(result->destination),
            qualified.execution.getExpectedBackendOutputDigest());
}

TEST(OneDNNQualificationTest,
     ManagedReferenceAdmitsFiniteShapeIndependentRowsAndRejectsInfinity) {
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  OneDNNQualificationSpec spec = llvm::cantFail(OneDNNQualificationSpec::create(
      LogicalFormat::F32, 16, 128, 32, 1, PhysicalTensorLayout::Cx,
      PhysicalTensorLayout::Cx, PhysicalTensorLayout::Cx, /*seed=*/41));
  OneDNNQualificationCase row = llvm::cantFail(
      materializeOneDNNQualificationCase(std::move(spec), kOneDNNBudget));
  llvm::Expected<OneDNNTensorNumericResult> result =
      executeManagedReferenceOneDNNTensorNumeric(
          environment, row.getOperation(), row.getInputs(),
          row.getDestinationTemplate(), kOneDNNBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(result->evidence.matmulInvocations, 1u);
  EXPECT_EQ(result->evidence.formalFusedMultiplyAdds, 0u);
  EXPECT_FALSE(result->evidence.implementation.empty());

  std::vector<OneDNNTensorStorage> nonfiniteInputs(row.getInputs().begin(),
                                                   row.getInputs().end());
  std::vector<RawLogicalValue> lhs =
      llvm::cantFail(unpackOneDNNTensorLogicalValues(nonfiniteInputs.front()));
  lhs.front().bits = UINT64_C(0x7f800000);
  nonfiniteInputs.front() = llvm::cantFail(packOneDNNTensorLogicalValues(
      nonfiniteInputs.front().getKey(), lhs, UINT8_C(0)));
  EXPECT_NE(expectError(executeManagedReferenceOneDNNTensorNumeric(
                            environment, row.getOperation(), nonfiniteInputs,
                            row.getDestinationTemplate(), kOneDNNBudget))
                .find("admits only finite inputs"),
            std::string::npos);
}

TEST(OneDNNQualificationTest, BatchedNCxRowUsesOneQualifiedMatmul) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  QualifiedRow qualified = llvm::cantFail(
      qualify(environment, files, LogicalFormat::F32, 3, 4, 5, 2));
  EXPECT_EQ(qualified.row.getInputs()[0].getKey().getLayout(),
            PhysicalTensorLayout::NCx);
  llvm::Expected<OneDNNTensorNumericResult> result =
      executeQualifiedOneDNNTensorNumeric(
          environment, qualified.execution, qualified.row.getOperation(),
          qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
          kOneDNNBudget);
  ASSERT_TRUE(static_cast<bool>(result))
      << (result ? std::string() : llvm::toString(result.takeError()));
  EXPECT_EQ(result->evidence.matmulInvocations, 1u);
  EXPECT_EQ(result->evidence.formalFusedMultiplyAdds, 0u);
}

TEST(OneDNNQualificationTest,
     QualificationAndBudgetsRejectBeforeReturningAResult) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, LogicalFormat::F32, 4, 8, 5));

  std::vector<OneDNNTensorStorage> tamperedInputs(
      qualified.row.getInputs().begin(), qualified.row.getInputs().end());
  std::vector<uint8_t> changed = tamperedInputs[0].getStorage().vec();
  changed[0] ^= UINT8_C(1);
  tamperedInputs[0] = llvm::cantFail(OneDNNTensorStorage::create(
      tamperedInputs[0].getKey(), std::move(changed)));
  EXPECT_NE(
      expectError(qualified.record.qualifyExecution(
                      environment, qualified.row.getOperation(), tamperedInputs,
                      qualified.row.getDestinationTemplate()))
          .find("does not exact-match"),
      std::string::npos);
  EXPECT_NE(
      expectError(executeQualifiedOneDNNTensorNumeric(
                      environment, qualified.execution,
                      qualified.row.getOperation(), tamperedInputs,
                      qualified.row.getDestinationTemplate(), kOneDNNBudget))
          .find("qualification-mismatch"),
      std::string::npos);
  EXPECT_NE(
      expectError(executeQualifiedOneDNNTensorNumeric(
                      environment, qualified.execution,
                      qualified.row.getOperation(), qualified.row.getInputs(),
                      qualified.row.getDestinationTemplate(),
                      OneDNNNumericWorkBudget::create(1, 1, 1)))
          .find("total-byte-budget-exceeded"),
      std::string::npos);
}

TEST(OneDNNQualificationTest,
     SpecAndRecordFilesAreClosedCanonicalAndNoReplace) {
  TemporaryDirectory files;
  OneDNNQualificationSpec spec = llvm::cantFail(OneDNNQualificationSpec::create(
      LogicalFormat::F32, 2, 3, 4, 1, PhysicalTensorLayout::Cx,
      PhysicalTensorLayout::Cx, PhysicalTensorLayout::Cx, 7));
  const std::string goodPath = files.getPath("good.json");
  ASSERT_FALSE(static_cast<bool>(writeOneDNNQualificationSpec(spec, goodPath)));
  llvm::Expected<OneDNNQualificationSpec> loaded =
      loadOneDNNQualificationSpec(goodPath);
  ASSERT_TRUE(static_cast<bool>(loaded))
      << (loaded ? std::string() : llvm::toString(loaded.takeError()));
  EXPECT_EQ(loaded->getDigest(), spec.getDigest());
  EXPECT_FALSE(
      expectError(writeOneDNNQualificationSpec(spec, goodPath)).empty());
  EXPECT_NE(expectError(loadOneDNNQualificationSpec("good.json"))
                .find("paths must be absolute"),
            std::string::npos);
  const std::string linkPath = files.getPath("good-link.json");
  ASSERT_FALSE(
      static_cast<bool>(llvm::sys::fs::create_link(goodPath, linkPath)));
  EXPECT_NE(expectError(loadOneDNNQualificationSpec(linkPath)).find("alias"),
            std::string::npos);

  const std::string noncanonicalPath = files.getPath("noncanonical.json");
  ASSERT_FALSE(static_cast<bool>(writeText(
      noncanonicalPath,
      "{ \"batch_count\":1,\"destination_layout\":\"cx\","
      "\"format\":\"f32\",\"k\":3,\"lhs_layout\":\"cx\","
      "\"m\":2,\"n\":4,\"rhs_layout\":\"cx\","
      "\"schema\":\"wafer-onednn-qualification-spec\",\"seed\":7}\n")));
  EXPECT_NE(expectError(loadOneDNNQualificationSpec(noncanonicalPath))
                .find("not in canonical form"),
            std::string::npos);

  const std::string unknownPath = files.getPath("unknown.json");
  ASSERT_FALSE(static_cast<bool>(writeText(
      unknownPath, "{\"batch_count\":1,\"destination_layout\":\"cx\","
                   "\"format\":\"f32\",\"k\":3,\"lhs_layout\":\"cx\","
                   "\"m\":2,\"n\":4,\"rhs_layout\":\"cx\","
                   "\"schema\":\"wafer-onednn-qualification-spec\",\"seed\":7,"
                   "\"unknown\":false}\n")));
  EXPECT_NE(expectError(loadOneDNNQualificationSpec(unknownPath))
                .find("unknown, missing or duplicate fields"),
            std::string::npos);

  const std::string malformedPath = files.getPath("malformed.json");
  ASSERT_FALSE(static_cast<bool>(writeText(
      malformedPath, "{\"batch_count\":\"bad\",\"destination_layout\":\"bad\","
                     "\"format\":\"bad\",\"k\":\"bad\",\"lhs_layout\":\"bad\","
                     "\"m\":\"bad\",\"n\":\"bad\",\"rhs_layout\":\"bad\","
                     "\"schema\":\"wafer-onednn-qualification-spec\","
                     "\"seed\":\"bad\"}\n")));
  std::string malformedError =
      expectError(loadOneDNNQualificationSpec(malformedPath));
  EXPECT_NE(malformedError.find("batch_count"), std::string::npos);
  EXPECT_NE(malformedError.find("field k"), std::string::npos);
  EXPECT_NE(malformedError.find("seed"), std::string::npos);

  const std::string obsoleteSchemaPath = files.getPath("obsolete-schema.json");
  ASSERT_FALSE(static_cast<bool>(writeText(
      obsoleteSchemaPath,
      "{\"batch_count\":1,\"destination_layout\":\"cx\","
      "\"format\":\"f32\",\"k\":3,\"lhs_layout\":\"cx\","
      "\"m\":2,\"n\":4,\"rhs_layout\":\"cx\","
      "\"schema\":\"invalid-onednn-qualification-spec\",\"seed\":7}\n")));
  EXPECT_NE(expectError(loadOneDNNQualificationSpec(obsoleteSchemaPath))
                .find("schema mismatch"),
            std::string::npos);
}

TEST(OneDNNQualificationTest,
     ExplicitPhysicalPayloadSpecRoundTripsWithoutRegeneration) {
  TemporaryDirectory files;
  OneDNNQualificationSpec generated =
      llvm::cantFail(OneDNNQualificationSpec::create(
          LogicalFormat::F32, 2, 3, 4, 1, PhysicalTensorLayout::Cx,
          PhysicalTensorLayout::Cx, PhysicalTensorLayout::Cx, 29));
  OneDNNQualificationCase generatedCase = llvm::cantFail(
      materializeOneDNNQualificationCase(std::move(generated), kOneDNNBudget));
  std::vector<uint8_t> lhs(generatedCase.getInputs()[0].getStorage().begin(),
                           generatedCase.getInputs()[0].getStorage().end());
  std::vector<uint8_t> rhs(generatedCase.getInputs()[1].getStorage().begin(),
                           generatedCase.getInputs()[1].getStorage().end());
  std::vector<uint8_t> destination(
      generatedCase.getDestinationTemplate().getStorage().begin(),
      generatedCase.getDestinationTemplate().getStorage().end());
  OneDNNQualificationSpec explicitSpec =
      llvm::cantFail(OneDNNQualificationSpec::createWithPhysicalPayload(
          LogicalFormat::F32, 2, 3, 4, 1, PhysicalTensorLayout::Cx,
          PhysicalTensorLayout::Cx, PhysicalTensorLayout::Cx, 31, lhs, rhs,
          destination));
  EXPECT_TRUE(explicitSpec.hasExplicitPhysicalPayload());
  const std::string path = files.getPath("explicit-spec.json");
  ASSERT_FALSE(
      static_cast<bool>(writeOneDNNQualificationSpec(explicitSpec, path)));
  OneDNNQualificationSpec loaded =
      llvm::cantFail(loadOneDNNQualificationSpec(path));
  EXPECT_EQ(loaded.getDigest(), explicitSpec.getDigest());
  EXPECT_TRUE(loaded.hasExplicitPhysicalPayload());
  OneDNNQualificationCase loadedCase = llvm::cantFail(
      materializeOneDNNQualificationCase(std::move(loaded), kOneDNNBudget));
  EXPECT_EQ(computeOneDNNTensorPayloadDigest(loadedCase.getInputs()),
            computeOneDNNTensorPayloadDigest(generatedCase.getInputs()));
  EXPECT_EQ(
      computeOneDNNTensorStorageDigest(loadedCase.getDestinationTemplate()),
      computeOneDNNTensorStorageDigest(generatedCase.getDestinationTemplate()));

  lhs.pop_back();
  EXPECT_NE(
      expectError(OneDNNQualificationSpec::createWithPhysicalPayload(
                      LogicalFormat::F32, 2, 3, 4, 1, PhysicalTensorLayout::Cx,
                      PhysicalTensorLayout::Cx, PhysicalTensorLayout::Cx, 31,
                      std::move(lhs), std::move(rhs), std::move(destination)))
          .find("byte geometry differs"),
      std::string::npos);
}

TEST(OneDNNQualificationTest, FinalRecordReadbackRejectsEvidenceTampering) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
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
  EXPECT_NE(expectError(loadVerifiedOneDNNQualificationRecord(tamperedPath))
                .find("does not prove one MatMul"),
            std::string::npos);
}

TEST(OneDNNQualificationTest,
     RuntimeRejectsFrozenImplementationAndDescriptorDrift) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  QualifiedRow qualified =
      llvm::cantFail(qualify(environment, files, LogicalFormat::F32, 4, 8, 5));
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(qualified.recordPath);
  ASSERT_TRUE(static_cast<bool>(buffer));
  struct DriftCase {
    llvm::StringRef key;
    llvm::StringRef replacement;
    llvm::StringRef fileName;
  };
  const DriftCase cases[] = {
      {"implementation", "tampered-implementation", "implementation.json"},
      {"resolved_descriptor_digest",
       "sha256:"
       "0000000000000000000000000000000000000000000000000000000000000000",
       "descriptor.json"},
  };
  for (const DriftCase &testCase : cases) {
    SCOPED_TRACE(testCase.key.str());
    std::string changed = (*buffer)->getBuffer().str();
    const std::string prefix =
        (llvm::Twine("\"") + testCase.key + "\":\"").str();
    const size_t valueStart = changed.find(prefix);
    ASSERT_NE(valueStart, std::string::npos);
    const size_t first = valueStart + prefix.size();
    const size_t last = changed.find('"', first);
    ASSERT_NE(last, std::string::npos);
    changed.replace(first, last - first, testCase.replacement.str());
    const std::string tamperedPath = files.getPath(testCase.fileName);
    ASSERT_FALSE(static_cast<bool>(writeText(tamperedPath, changed)));

    VerifiedOneDNNQualificationRecord tampered =
        llvm::cantFail(loadVerifiedOneDNNQualificationRecord(tamperedPath));
    QualifiedOneDNNExecution execution =
        llvm::cantFail(tampered.qualifyExecution(
            environment, qualified.row.getOperation(),
            qualified.row.getInputs(), qualified.row.getDestinationTemplate()));
    std::string error = expectError(executeQualifiedOneDNNTensorNumeric(
        environment, execution, qualified.row.getOperation(),
        qualified.row.getInputs(), qualified.row.getDestinationTemplate(),
        kOneDNNBudget));
    EXPECT_NE(error.find("implementation or resolved descriptor changed"),
              std::string::npos)
        << error;
  }
}

TEST(OneDNNQualificationTest, FreezeRequiresARegisteredDisjointHeldOutSeed) {
  TemporaryDirectory files;
  OneDNNExecutionEnvironment environment =
      llvm::cantFail(createManagedOneDNNExecutionEnvironment());
  OneDNNQualificationSpec spec = llvm::cantFail(OneDNNQualificationSpec::create(
      LogicalFormat::F32, 2, 3, 4, 1, PhysicalTensorLayout::Cx,
      PhysicalTensorLayout::Cx, PhysicalTensorLayout::Cx, 17));
  const std::string specPath = files.getPath("spec.json");
  const std::string calibrationPath = files.getPath("calibration.json");
  ASSERT_FALSE(static_cast<bool>(writeOneDNNQualificationSpec(spec, specPath)));
  ASSERT_FALSE(static_cast<bool>(
      calibrateOneDNNBackend(environment, specPath, calibrationPath,
                             kSmallFormalBudget, kOneDNNBudget)));
  std::string error = expectError(freezeOneDNNBackendPolicy(
      calibrationPath, specPath, files.getPath("policy.json"), {0.0, 0.0}));
  EXPECT_NE(error.find("have disjoint seeds/digests"), std::string::npos);
}

} // namespace
