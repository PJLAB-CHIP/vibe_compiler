//===- OneDNNQualificationInternal.h - Qualification collaboration -*- C++
//-*-===//

#ifndef WAFER_TARGET_ONEDNNQUALIFICATIONINTERNAL_H
#define WAFER_TARGET_ONEDNNQUALIFICATIONINTERNAL_H

#include "Wafer/Target/Numeric/Qualification/OneDNNQualification.h"

#include "OneDNNTensorNumericInternal.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <initializer_list>
#include <string>
#include <utility>

namespace wafer::onednn_qualification_detail {

inline constexpr llvm::StringLiteral kCalibrationSchema =
    "wafer-onednn-calibration";
inline constexpr llvm::StringLiteral kPolicySchema = "wafer-onednn-policy";
inline constexpr llvm::StringLiteral kFinalSchema =
    "wafer-onednn-qualification-record";
inline constexpr llvm::StringLiteral kValueDomain =
    "deterministic-finite-f32-exact-inputs";
inline constexpr llvm::StringLiteral kTargetComparator = "raw-exact";
inline constexpr llvm::StringLiteral kBackendComparator = "absolute-relative";
inline constexpr llvm::StringLiteral kProofBasis =
    "finite-calibration-held-out-exact-payload";

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

llvm::json::Object specJSON(const OneDNNQualificationSpec &spec);
llvm::json::Object
environmentJSON(const OneDNNExecutionEnvironment &environment);
std::string
environmentRecordDigest(const OneDNNExecutionEnvironment &environment);
llvm::Error validateEnvironmentJSON(const llvm::json::Object &object,
                                    llvm::StringRef expectedEnvironmentDigest,
                                    llvm::StringRef expectedBackendDigest);
llvm::Expected<OneDNNQualificationSpec>
parseSpecObject(const llvm::json::Object &object);

llvm::json::Object flagsJSON(FormalNumericExceptionFlags flags);
llvm::Expected<FormalNumericExceptionFlags>
parseFlags(const llvm::json::Object &object);
llvm::Error writeFileNoReplace(llvm::StringRef path, llvm::StringRef content);

struct Comparison {
  bool rawExact = true;
  double maximumAbsoluteError = 0.0;
  double maximumRelativeError = 0.0;
};

struct QualificationRun {
  OneDNNQualificationCase testCase;
  FormalTensorNumericResult formal;
  detail::UnqualifiedOneDNNExecutionResult backend;
  Comparison comparison;
  std::string formalOutputDigest;
};

llvm::Expected<QualificationRun>
runQualification(const OneDNNExecutionEnvironment &environment,
                 OneDNNQualificationCase testCase,
                 FormalNumericWorkBudget formalBudget,
                 OneDNNNumericWorkBudget onednnBudget);

} // namespace wafer::onednn_qualification_detail

#endif // WAFER_TARGET_ONEDNNQUALIFICATIONINTERNAL_H
