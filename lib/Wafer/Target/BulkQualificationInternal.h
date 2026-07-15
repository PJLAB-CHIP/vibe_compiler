//===- BulkQualificationInternal.h - Qualification collaboration -*- C++
//-*-===//

#ifndef WAFER_TARGET_BULKQUALIFICATIONINTERNAL_H
#define WAFER_TARGET_BULKQUALIFICATIONINTERNAL_H

#include "Wafer/Target/BulkQualification.h"

#include "BulkTensorNumericInternal.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <initializer_list>
#include <string>
#include <utility>

namespace wafer::bulk_qualification_detail {

inline constexpr llvm::StringLiteral kCalibrationSchema =
    "wafer-bulk-calibration-v1";
inline constexpr llvm::StringLiteral kPolicySchema =
    "wafer-bulk-frozen-policy-v1";
inline constexpr llvm::StringLiteral kFinalSchema =
    "wafer-bulk-qualification-record-v1";
inline constexpr llvm::StringLiteral kValueDomain =
    "deterministic-finite-f32-exact-inputs-v1";
inline constexpr llvm::StringLiteral kTargetComparator = "raw-exact-v1";
inline constexpr llvm::StringLiteral kBackendComparator =
    "absolute-relative-v1";
inline constexpr llvm::StringLiteral kProofBasis =
    "finite-calibration-held-out-exact-payload-v1";

llvm::Error invalid(const llvm::Twine &detail);

template <typename... Values>
llvm::Error takeExpectedErrors(Values &...values) {
  llvm::Error errors = llvm::Error::success();
  auto take = [&](auto &value) {
    if (!value)
      errors = llvm::joinErrors(std::move(errors), value.takeError());
  };
  (take(values), ...);
  return errors;
}

std::string sha256(llvm::StringRef payload);
std::string canonicalJSON(const llvm::json::Value &value);
std::string canonicalJSON(llvm::json::Object &&object);

struct ParsedJSON {
  llvm::json::Value value;
  std::string canonical;
  std::string digest;
};

llvm::Expected<ParsedJSON> loadCanonicalJSON(llvm::StringRef path);
llvm::Error requireFields(const llvm::json::Object &object,
                          std::initializer_list<llvm::StringRef> fields,
                          llvm::StringRef context);
llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context);
llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Object &object,
                                         llvm::StringRef key,
                                         llvm::StringRef context);
llvm::Expected<double>
requireFiniteNonnegative(const llvm::json::Object &object, llvm::StringRef key,
                         llvm::StringRef context);
llvm::Expected<std::string> requireDigest(const llvm::json::Object &object,
                                          llvm::StringRef key,
                                          llvm::StringRef context);

llvm::json::Object specJSON(const BulkQualificationSpec &spec);
llvm::json::Object environmentJSON(const BulkExecutionEnvironment &environment);
std::string
environmentRecordDigest(const BulkExecutionEnvironment &environment);
llvm::Error validateEnvironmentJSON(const llvm::json::Object &object,
                                    llvm::StringRef expectedEnvironmentDigest,
                                    llvm::StringRef expectedBackendDigest);
llvm::Expected<BulkQualificationSpec>
parseSpecObject(const llvm::json::Object &object);

llvm::json::Object flagsJSON(FormalNumericExceptionFlags flags);
llvm::Expected<FormalNumericExceptionFlags>
parseFlags(const llvm::json::Object &object);
llvm::Error publishNoReplace(llvm::StringRef path, llvm::StringRef content);

struct Comparison {
  bool rawExact = true;
  double maximumAbsoluteError = 0.0;
  double maximumRelativeError = 0.0;
};

struct QualificationRun {
  BulkQualificationCase testCase;
  FormalTensorNumericResult formal;
  detail::UnqualifiedBulkExecutionResult backend;
  Comparison comparison;
  std::string formalOutputDigest;
};

llvm::Expected<QualificationRun> runQualification(
    const BulkExecutionEnvironment &environment, BulkQualificationCase testCase,
    FormalNumericWorkBudget formalBudget, BulkNumericWorkBudget bulkBudget);

} // namespace wafer::bulk_qualification_detail

#endif // WAFER_TARGET_BULKQUALIFICATIONINTERNAL_H
