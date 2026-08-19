//===- NumericManagedDependencyIntegrationTest.cpp -----------------------===//

#include "Wafer/Target/Numeric/Dependency/NumericDependencyConformance.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <gmp.h>
#include <mpfr.h>

#include <cstdint>
#include <string>

namespace {

template <typename FunctionPointer>
const void *symbolAddress(FunctionPointer symbol) {
  return reinterpret_cast<const void *>(
      reinterpret_cast<std::uintptr_t>(symbol));
}

void expectLoadedObjectMatchesRecord(
    const wafer::NumericDependencyConformanceRecord &record,
    const wafer::NumericLoadedObject &loaded, llvm::StringRef realName,
    llvm::StringRef loaderName, wafer::NumericLoadedObjectKind expectedKind) {
  const wafer::NumericDependencyFileRecord *real = record.findFile(realName);
  const wafer::NumericDependencyFileRecord *loader =
      record.findFile(loaderName);
  ASSERT_NE(real, nullptr);
  ASSERT_NE(loader, nullptr);
  EXPECT_EQ(loaded.kind, expectedKind);
  EXPECT_TRUE(loaded.resolvedPath == real->resolvedPath ||
              loaded.resolvedPath == loader->resolvedPath);
  EXPECT_EQ(loaded.sha256, real->sha256);
  EXPECT_EQ(loaded.sha256, loader->sha256);
}

TEST(NumericManagedDependencyIntegrationTest,
     RealRecordBindsActuallyLoadedMPFRAndGMP) {
  const llvm::StringRef managedRoot = WAFER_NUMERIC_MODEL_DEPS_ROOT;
  const llvm::StringRef recordPath = WAFER_NUMERIC_MODEL_DEPS_RECORD;
  const llvm::StringRef expectedRecordSHA256 =
      WAFER_NUMERIC_MODEL_DEPS_RECORD_SHA256;

  llvm::Expected<wafer::NumericDependencyConformanceRecord> record =
      wafer::readNumericDependencyConformanceRecord(managedRoot, recordPath,
                                                    expectedRecordSHA256);
  ASSERT_TRUE(static_cast<bool>(record))
      << (record ? std::string() : llvm::toString(record.takeError()));
  EXPECT_EQ(record->getManagedRoot(), managedRoot);
  EXPECT_EQ(record->getRecordPath(), recordPath);
  EXPECT_EQ(record->getRecordSHA256(), expectedRecordSHA256);

  wafer::DladdrNumericLoadedObjectProvider provider(
      symbolAddress(&mpfr_get_version), symbolAddress(&mpz_init));
  llvm::Expected<wafer::NumericDependencyExecutionBinding> execution =
      wafer::bindNumericDependenciesToLoadedObjects(*record, provider);
  ASSERT_TRUE(static_cast<bool>(execution))
      << (execution ? std::string() : llvm::toString(execution.takeError()));
  EXPECT_EQ(execution->getRecordSHA256(), expectedRecordSHA256);
  EXPECT_EQ(execution->getRecordSHA256(), record->getRecordSHA256());
  EXPECT_EQ(execution->getBindingSHA256().size(), 64u);
  expectLoadedObjectMatchesRecord(*record, execution->getMPFR(), "mpfr",
                                  "mpfr-soname",
                                  wafer::NumericLoadedObjectKind::MPFR);
  expectLoadedObjectMatchesRecord(*record, execution->getGMP(), "gmp",
                                  "gmp-soname",
                                  wafer::NumericLoadedObjectKind::GMP);

  EXPECT_STREQ(mpfr_get_version(), MPFR_VERSION_STRING);
  EXPECT_NE(mpfr_buildopt_tls_p(), 0);
  const wafer::NumericDependencySourceRecord *mpfrSource =
      record->findSource("mpfr");
  const wafer::NumericDependencySourceRecord *gmpSource =
      record->findSource("gmp");
  ASSERT_NE(mpfrSource, nullptr);
  ASSERT_NE(gmpSource, nullptr);
  EXPECT_EQ(mpfrSource->version, MPFR_VERSION_STRING);
  EXPECT_STREQ(gmp_version, gmpSource->version.c_str());
}

} // namespace
