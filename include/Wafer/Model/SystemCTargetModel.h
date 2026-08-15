//===- SystemCTargetModel.h - SystemC functional-event model -*- C++ -*-===//

#ifndef WAFER_MODEL_SYSTEMCTARGETMODEL_H
#define WAFER_MODEL_SYSTEMCTARGETMODEL_H

#include "Wafer/Model/TargetModelKernel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer::model {

enum class SystemCTargetModelErrorCode : uint8_t {
  InvalidLifecycle,
  InvocationFailure,
  NoProgress,
  ResultInvariantViolation,
};

llvm::StringRef
stringifySystemCTargetModelErrorCode(SystemCTargetModelErrorCode code);

class SystemCTargetModelError final
    : public llvm::ErrorInfo<SystemCTargetModelError> {
public:
  static char ID;

  SystemCTargetModelError(SystemCTargetModelErrorCode code, std::string detail)
      : code(code), detail(std::move(detail)) {}

  SystemCTargetModelErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemCTargetModelErrorCode code;
  std::string detail;
};

/// One card-owned output allocation returned after successful execution.
/// Tile and
/// slot fields identify the canonical ABI view used for decoding; aliased
/// views from other Tiles do not create duplicate result objects.
struct TargetModelOutput {
  TargetModelResourceId resource;
  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  int64_t slotOrdinal = -1;
  int64_t resourceIndex = -1;
  std::vector<uint8_t> bytes;
};

/// Complete model result. SystemC objects and private memory are not retained
/// by this value.
struct TargetModelResult {
  TargetIdentityId targetIdentity;
  ModelProfileId modelProfile;
  int64_t completedTileCount = 0;
  uint64_t issuedCommandCount = 0;
  uint64_t systemCThreadProcessCount = 0;
  uint64_t finalDeltaCount = 0;
  FormalNumericExceptionFlags numericFlags;
  uint64_t formalNumericCommandCount = 0;
  uint64_t managedReferenceNumericCommandCount = 0;
  uint64_t managedReferenceScalarEvaluationCount = 0;
  uint64_t bulkNumericCommandCount = 0;
  uint64_t bulkMatmulInvocationCount = 0;
  uint64_t bulkReorderInvocationCount = 0;
  uint64_t bulkFormalFusedMultiplyAddCount = 0;
  std::vector<std::string> bulkQualificationRecordDigests;
  std::vector<std::string> bulkManagedReferenceEnvironmentDigests;
  std::vector<std::string> managedReferenceTensorEnvironmentDigests;
  std::vector<std::string> managedReferenceTensorImplementations;
  std::string systemCVersion;
  std::string schedulerIdentity;
  std::vector<TargetModelOutput> outputs;
};

/// Runs one already-prepared host target-call executable. This entry is
/// intentionally available only in a SystemC-enabled build and must be called
/// during the initial elaboration of one process; it defines no sc_main.
llvm::Expected<TargetModelResult>
executeSystemCTargetModel(compiler::TargetCallExecutable executable,
                          llvm::ArrayRef<TargetModelInputBinding> inputBindings,
                          TargetModelKernelBudget budget,
                          TargetModelExecutionPolicy policy =
                              TargetModelExecutionPolicy::formalOnly());

} // namespace wafer::model

#endif // WAFER_MODEL_SYSTEMCTARGETMODEL_H
