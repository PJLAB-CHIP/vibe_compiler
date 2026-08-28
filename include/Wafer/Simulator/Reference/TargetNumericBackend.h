//===- TargetNumericBackend.h - Target numeric backend adapters -*- C++ -*-===//

#ifndef WAFER_SIMULATOR_REFERENCE_TARGETNUMERICBACKEND_H
#define WAFER_SIMULATOR_REFERENCE_TARGETNUMERICBACKEND_H

#include "Wafer/Simulator/Kernel/TargetModelKernel.h"
#include "Wafer/Simulator/OneDNN/OneDNNQualification.h"

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
class QualifiedTargetModelOneDNNBackend final
    : public TargetModelOneDNNBackend {
public:
  QualifiedTargetModelOneDNNBackend(const QualifiedTargetModelOneDNNBackend &) =
      delete;
  QualifiedTargetModelOneDNNBackend &
  operator=(const QualifiedTargetModelOneDNNBackend &) = delete;

  static llvm::Expected<std::unique_ptr<QualifiedTargetModelOneDNNBackend>>
  create(llvm::ArrayRef<std::string> recordPaths,
         OneDNNNumericWorkBudget budget);

  llvm::Expected<std::optional<TargetModelOneDNNResult>>
  tryExecute(const TargetModelGemmRequest &request) const override;

private:
  QualifiedTargetModelOneDNNBackend(
      OneDNNExecutionEnvironment environment,
      std::vector<VerifiedOneDNNQualificationRecord> records,
      OneDNNNumericWorkBudget budget)
      : environment(std::move(environment)), records(std::move(records)),
        budget(budget) {}

  OneDNNExecutionEnvironment environment;
  std::vector<VerifiedOneDNNQualificationRecord> records;
  OneDNNNumericWorkBudget budget;
};

/// Scalable deterministic oneDNN path for end-to-end model-reference gates.
/// Qualification is structural (supported GEMM semantics, managed environment,
/// finite inputs and explicit byte budgets), not exact payload qualification.
/// It therefore requires a final external-oracle tolerance check and must not
/// be reported as hardware-correlated or raw-exact target arithmetic.
class ManagedReferenceTargetModelBackend final
    : public TargetModelOneDNNBackend,
      public TargetModelManagedReferenceBackend {
public:
  ManagedReferenceTargetModelBackend(
      const ManagedReferenceTargetModelBackend &) = delete;
  ManagedReferenceTargetModelBackend &
  operator=(const ManagedReferenceTargetModelBackend &) = delete;

  static llvm::Expected<std::unique_ptr<ManagedReferenceTargetModelBackend>>
  create(OneDNNNumericWorkBudget budget);

  llvm::Expected<std::optional<TargetModelOneDNNResult>>
  tryExecute(const TargetModelGemmRequest &request) const override;

  llvm::Expected<TargetModelManagedReferenceResult>
  execute(const TargetModelConvertRequest &request,
          FormalNumericWorkBudget scalarBudget) const override;
  llvm::Expected<TargetModelManagedReferenceResult>
  execute(const TargetModelElementwiseRequest &request,
          FormalNumericWorkBudget scalarBudget) const override;
  llvm::Expected<TargetModelManagedReferenceResult>
  execute(const TargetModelReduceRequest &request,
          FormalNumericWorkBudget scalarBudget) const override;

private:
  ManagedReferenceTargetModelBackend(OneDNNExecutionEnvironment environment,
                                     OneDNNNumericWorkBudget budget)
      : environment(std::move(environment)), budget(budget) {}

  OneDNNExecutionEnvironment environment;
  OneDNNNumericWorkBudget budget;
};

} // namespace wafer::model

#endif // WAFER_SIMULATOR_REFERENCE_TARGETNUMERICBACKEND_H
