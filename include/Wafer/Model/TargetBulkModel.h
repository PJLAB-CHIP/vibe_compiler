//===- TargetBulkModel.h - Qualified model bulk dispatch ------*- C++ -*-===//

#ifndef WAFER_MODEL_TARGETBULKMODEL_H
#define WAFER_MODEL_TARGETBULKMODEL_H

#include "Wafer/Model/TargetModelKernel.h"
#include "Wafer/Target/BulkQualification.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::model {

/// Managed oneDNN adapter for the feature-independent model dispatch seam.
/// Construction reads the current managed environment and verified final
/// records. Runtime execution remains exact-match admission only.
class QualifiedTargetModelBulkBackend final : public TargetModelBulkBackend {
public:
  QualifiedTargetModelBulkBackend(const QualifiedTargetModelBulkBackend &) =
      delete;
  QualifiedTargetModelBulkBackend &
  operator=(const QualifiedTargetModelBulkBackend &) = delete;

  static llvm::Expected<std::unique_ptr<QualifiedTargetModelBulkBackend>>
  create(llvm::ArrayRef<std::string> recordPaths, BulkNumericWorkBudget budget);

  llvm::Expected<std::optional<TargetModelBulkResult>>
  tryExecute(const TargetModelBulkRequest &request) const override;

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

} // namespace wafer::model

#endif // WAFER_MODEL_TARGETBULKMODEL_H
