//===- TargetBulkModel.cpp - Qualified model bulk dispatch --------------===//

#include "Wafer/Model/TargetBulkModel.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::model {

llvm::Expected<std::unique_ptr<QualifiedTargetModelBulkBackend>>
QualifiedTargetModelBulkBackend::create(llvm::ArrayRef<std::string> recordPaths,
                                        BulkNumericWorkBudget budget) {
  if (recordPaths.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "bulk model requires at least one verified "
                                   "qualification record");
  llvm::Expected<BulkExecutionEnvironment> environment =
      createManagedBulkExecutionEnvironment();
  if (!environment)
    return environment.takeError();
  std::vector<VerifiedBulkQualificationRecord> records;
  records.reserve(recordPaths.size());
  for (const std::string &path : recordPaths) {
    llvm::Expected<VerifiedBulkQualificationRecord> record =
        loadVerifiedBulkQualificationRecord(path);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
  }
  return std::unique_ptr<QualifiedTargetModelBulkBackend>(
      new QualifiedTargetModelBulkBackend(std::move(*environment),
                                          std::move(records), budget));
}

llvm::Expected<std::optional<TargetModelBulkResult>>
QualifiedTargetModelBulkBackend::tryExecute(
    const TargetModelBulkRequest &request) const {
  std::vector<BulkTensorStorage> inputs;
  inputs.reserve(request.inputs.size());
  for (const TargetModelBulkTensor &input : request.inputs) {
    llvm::Expected<BulkTensorStorage> storage =
        BulkTensorStorage::create(input.key, input.storage);
    if (!storage)
      return storage.takeError();
    inputs.push_back(std::move(*storage));
  }
  llvm::Expected<BulkTensorStorage> destinationTemplate =
      BulkTensorStorage::create(request.destinationTemplate.key,
                                request.destinationTemplate.storage);
  if (!destinationTemplate)
    return destinationTemplate.takeError();

  for (const VerifiedBulkQualificationRecord &record : records) {
    llvm::Expected<BulkBackendAdmission> admission = record.createAdmission(
        environment, request.command, inputs, *destinationTemplate);
    if (!admission) {
      llvm::consumeError(admission.takeError());
      continue;
    }
    llvm::Expected<BulkTensorNumericResult> result =
        executeAdmittedBulkTensorNumeric(environment, *admission,
                                         request.command, inputs,
                                         *destinationTemplate, budget);
    if (!result)
      return result.takeError();
    TargetModelBulkResult modelResult{
        {result->destination.getKey(),
         std::vector<uint8_t>(result->destination.getStorage().begin(),
                              result->destination.getStorage().end())},
        result->flags,
        {result->evidence.matmulInvocations,
         result->evidence.reorderInvocations,
         result->evidence.formalFusedMultiplyAdds,
         admission->getRecordDigest().str(), result->evidence.implementation}};
    return std::optional<TargetModelBulkResult>(std::move(modelResult));
  }
  return std::optional<TargetModelBulkResult>();
}

} // namespace wafer::model
