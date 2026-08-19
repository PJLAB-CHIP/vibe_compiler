//===- BulkQualificationComparison.cpp - Formal/backend comparison ---===//

#include "BulkQualificationInternal.h"

#include "Wafer/Target/Layout/PhysicalTensorCodec.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace wafer::bulk_qualification_detail {

namespace {

const llvm::fltSemantics &getSemantics(LogicalFormat format) {
  switch (format) {
  case LogicalFormat::F16:
    return llvm::APFloat::IEEEhalf();
  case LogicalFormat::BF16:
    return llvm::APFloat::BFloat();
  case LogicalFormat::F32:
    return llvm::APFloat::IEEEsingle();
  default:
    llvm_unreachable("unsupported qualification floating format");
  }
}

double toDouble(RawLogicalValue value) {
  const LogicalFormatDescriptor &descriptor =
      *findLogicalFormatDescriptor(value.format);
  llvm::APFloat floating(
      getSemantics(value.format),
      llvm::APInt(descriptor.storageBits, value.bits, /*isSigned=*/false));
  return floating.convertToDouble();
}

llvm::Expected<Comparison>
compareOutputs(const FormalTensorNumericResult &formal,
               const BulkTensorStorage &backend) {
  llvm::Expected<std::vector<RawLogicalValue>> backendValues =
      unpackBulkTensorLogicalValues(backend);
  if (!backendValues)
    return backendValues.takeError();
  if (backendValues->size() != formal.values.size())
    return invalid("formal/backend qualification output count mismatch");
  Comparison comparison;
  for (auto [expected, actual] :
       llvm::zip_equal(formal.values, *backendValues)) {
    if (expected.format != actual.format)
      return invalid("formal/backend qualification output format mismatch");
    comparison.rawExact &= expected.bits == actual.bits;
    llvm::Expected<LogicalValueClassification> expectedClass =
        classifyRawLogicalValue(expected, NonCanonicalEncodingPolicy::Reject);
    llvm::Expected<LogicalValueClassification> actualClass =
        classifyRawLogicalValue(actual, NonCanonicalEncodingPolicy::Reject);
    if (llvm::Error error = takeExpectedErrors(expectedClass, actualClass))
      return error;
    if (expectedClass->valueClass != actualClass->valueClass ||
        expectedClass->negative != actualClass->negative)
      return invalid("formal/backend special value classification mismatch");
    if (expectedClass->valueClass == LogicalValueClass::Infinity ||
        expectedClass->valueClass == LogicalValueClass::QuietNaN ||
        expectedClass->valueClass == LogicalValueClass::SignalingNaN)
      continue;
    const double expectedValue = toDouble(expected);
    const double actualValue = toDouble(actual);
    const double absolute = std::abs(actualValue - expectedValue);
    const double relative =
        absolute /
        std::max(std::abs(expectedValue), std::numeric_limits<double>::min());
    comparison.maximumAbsoluteError =
        std::max(comparison.maximumAbsoluteError, absolute);
    comparison.maximumRelativeError =
        std::max(comparison.maximumRelativeError, relative);
  }
  return comparison;
}

} // namespace

llvm::Expected<QualificationRun> runQualification(
    const BulkExecutionEnvironment &environment, BulkQualificationCase testCase,
    FormalNumericWorkBudget formalBudget, BulkNumericWorkBudget bulkBudget) {
  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      unpackBulkTensorLogicalValues(testCase.getInputs()[0]);
  llvm::Expected<std::vector<RawLogicalValue>> rhs =
      unpackBulkTensorLogicalValues(testCase.getInputs()[1]);
  if (llvm::Error error = takeExpectedErrors(lhs, rhs))
    return error;
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*lhs, *rhs};
  FormalNumericExecutionContext context;
  llvm::Expected<FormalTensorNumericResult> formal = executeFormalTensorNumeric(
      context, testCase.getCommand(), views, formalBudget);
  if (!formal)
    return formal.takeError();
  llvm::Expected<detail::UnqualifiedBulkExecutionResult> backend =
      detail::executeBulkTensorForQualification(
          environment, testCase.getCommand(), testCase.getInputs(),
          testCase.getDestinationTemplate(), bulkBudget);
  if (!backend)
    return backend.takeError();
  llvm::Expected<Comparison> comparison =
      compareOutputs(*formal, backend->destination);
  if (!comparison)
    return comparison.takeError();
  const BulkTensorStorage &destinationTemplate =
      testCase.getDestinationTemplate();
  // The target comparator above owns logical raw-bit equality.  Physical
  // layouts may contain padding which is not part of that logical tensor and
  // which a backend is free to preserve or overwrite.  Overlay the formal
  // values on the backend's final storage so that the two storage digests
  // differ exactly when an observable logical element differs, rather than
  // when only unobservable padding differs.  The backend storage digest is
  // still recorded independently so qualified execution can reject backend
  // implementation drift.
  llvm::Expected<std::vector<uint8_t>> formalPhysical =
      packPhysicalTensorLogicalValues(destinationTemplate.getKey(),
                                      formal->values,
                                      backend->destination.getStorage());
  if (!formalPhysical)
    return formalPhysical.takeError();
  llvm::Expected<BulkTensorStorage> formalStorage = BulkTensorStorage::create(
      destinationTemplate.getKey(), std::move(*formalPhysical));
  if (!formalStorage)
    return formalStorage.takeError();
  return QualificationRun{std::move(testCase), std::move(*formal),
                          std::move(*backend), *comparison,
                          computeBulkTensorStorageDigest(*formalStorage)};
}

} // namespace wafer::bulk_qualification_detail
