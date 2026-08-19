//===- TargetBulkModel.h - Qualified model bulk dispatch ------*- C++ -*-===//

#ifndef WAFER_MODEL_TARGETBULKMODEL_H
#define WAFER_MODEL_TARGETBULKMODEL_H

#include "Wafer/Model/Core/TargetModelKernel.h"
#include "Wafer/Target/Numeric/Qualification/BulkQualification.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::model {

/// Managed oneDNN adapter for the feature-independent model dispatch seam.
/// Construction reads the current managed environment and verified final
/// records. Runtime execution remains exact-match qualification only.
class QualifiedTargetModelBulkBackend final : public TargetModelBulkBackend {
public:
  QualifiedTargetModelBulkBackend(const QualifiedTargetModelBulkBackend &) =
      delete;
  QualifiedTargetModelBulkBackend &
  operator=(const QualifiedTargetModelBulkBackend &) = delete;

  static llvm::Expected<std::unique_ptr<QualifiedTargetModelBulkBackend>>
  create(llvm::ArrayRef<std::string> recordPaths, BulkNumericWorkBudget budget);

  llvm::Expected<std::optional<TargetModelBulkResult>>
  tryExecute(const TargetModelNumericRequest &request) const override;

private:
  QualifiedTargetModelBulkBackend(
      BulkExecutionEnvironment environment,
      std::vector<VerifiedBulkQualificationRecord> records,
      BulkNumericWorkBudget budget)
      : environment(std::move(environment)), records(std::move(records)),
        budget(budget) {}

  BulkExecutionEnvironment environment;
  std::vector<VerifiedBulkQualificationRecord> records;
  BulkNumericWorkBudget budget;
};

/// Scalable deterministic oneDNN path for end-to-end model-reference gates.
/// Qualification is structural (supported GEMM semantics, managed environment,
/// finite inputs and explicit byte budgets), not exact payload qualification.
/// It therefore requires a final external-oracle tolerance check and must not
/// be reported as hardware-correlated or raw-exact target arithmetic.
class ManagedReferenceTargetModelBackend final
    : public TargetModelBulkBackend,
      public TargetModelManagedReferenceBackend {
public:
  ManagedReferenceTargetModelBackend(
      const ManagedReferenceTargetModelBackend &) = delete;
  ManagedReferenceTargetModelBackend &
  operator=(const ManagedReferenceTargetModelBackend &) = delete;

  static llvm::Expected<std::unique_ptr<ManagedReferenceTargetModelBackend>>
  create(BulkNumericWorkBudget budget);

  llvm::Expected<std::optional<TargetModelBulkResult>>
  tryExecute(const TargetModelNumericRequest &request) const override;

  llvm::Expected<TargetModelManagedReferenceResult>
  execute(const TargetModelNumericRequest &request,
          FormalNumericWorkBudget scalarBudget) const override;

private:
  ManagedReferenceTargetModelBackend(BulkExecutionEnvironment environment,
                                     BulkNumericWorkBudget budget)
      : environment(std::move(environment)), budget(budget) {}

  BulkExecutionEnvironment environment;
  BulkNumericWorkBudget budget;
};

} // namespace wafer::model

#endif // WAFER_MODEL_TARGETBULKMODEL_H
