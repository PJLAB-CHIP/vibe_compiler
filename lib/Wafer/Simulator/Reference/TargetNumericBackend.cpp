//===- TargetNumericBackend.cpp - Target numeric backend adapters ---------===//

#include "Wafer/Simulator/Reference/TargetNumericBackend.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::model {

llvm::Expected<std::unique_ptr<QualifiedTargetModelOneDNNBackend>>
QualifiedTargetModelOneDNNBackend::create(
    llvm::ArrayRef<std::string> recordPaths, OneDNNNumericWorkBudget budget) {
  if (recordPaths.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "oneDNN backend requires at least one verified "
        "qualification record");
  llvm::Expected<OneDNNExecutionEnvironment> environment =
      createManagedOneDNNExecutionEnvironment();
  if (!environment)
    return environment.takeError();
  std::vector<VerifiedOneDNNQualificationRecord> records;
  records.reserve(recordPaths.size());
  for (const std::string &path : recordPaths) {
    llvm::Expected<VerifiedOneDNNQualificationRecord> record =
        loadVerifiedOneDNNQualificationRecord(path);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
  }
  return std::unique_ptr<QualifiedTargetModelOneDNNBackend>(
      new QualifiedTargetModelOneDNNBackend(std::move(*environment),
                                            std::move(records), budget));
}

llvm::Expected<std::optional<TargetModelOneDNNResult>>
QualifiedTargetModelOneDNNBackend::tryExecute(
    const TargetModelGemmRequest &request) const {
  std::vector<OneDNNTensorStorage> inputs;
  inputs.reserve(request.tensors.inputs.size());
  for (const TargetModelNumericTensor &input : request.tensors.inputs) {
    llvm::Expected<OneDNNTensorStorage> storage =
        OneDNNTensorStorage::create(input.key, input.storage);
    if (!storage)
      return storage.takeError();
    inputs.push_back(std::move(*storage));
  }
  llvm::Expected<OneDNNTensorStorage> destinationTemplate =
      OneDNNTensorStorage::create(request.tensors.destinationTemplate.key,
                                  request.tensors.destinationTemplate.storage);
  if (!destinationTemplate)
    return destinationTemplate.takeError();

  for (const VerifiedOneDNNQualificationRecord &record : records) {
    llvm::Expected<QualifiedOneDNNExecution> qualifiedExecution =
        record.qualifyExecution(environment, request.operation, inputs,
                                *destinationTemplate);
    if (!qualifiedExecution) {
      llvm::consumeError(qualifiedExecution.takeError());
      continue;
    }
    llvm::Expected<OneDNNTensorNumericResult> result =
        executeQualifiedOneDNNTensorNumeric(environment, *qualifiedExecution,
                                            request.operation, inputs,
                                            *destinationTemplate, budget);
    if (!result)
      return result.takeError();
    TargetModelOneDNNResult modelResult{
        {result->destination.getKey(),
         std::vector<uint8_t>(result->destination.getStorage().begin(),
                              result->destination.getStorage().end())},
        result->flags,
        {result->evidence.matmulInvocations,
         result->evidence.reorderInvocations,
         result->evidence.formalFusedMultiplyAdds,
         TargetModelOneDNNEvidenceKind::ExactQualificationRecord,
         qualifiedExecution->getRecordDigest().str(),
         result->evidence.implementation}};
    return std::optional<TargetModelOneDNNResult>(std::move(modelResult));
  }
  return std::optional<TargetModelOneDNNResult>();
}

llvm::Expected<std::unique_ptr<ManagedReferenceTargetModelBackend>>
ManagedReferenceTargetModelBackend::create(OneDNNNumericWorkBudget budget) {
  llvm::Expected<OneDNNExecutionEnvironment> environment =
      createManagedOneDNNExecutionEnvironment();
  if (!environment)
    return environment.takeError();
  return std::unique_ptr<ManagedReferenceTargetModelBackend>(
      new ManagedReferenceTargetModelBackend(std::move(*environment), budget));
}

llvm::Expected<std::optional<TargetModelOneDNNResult>>
ManagedReferenceTargetModelBackend::tryExecute(
    const TargetModelGemmRequest &request) const {
  std::vector<OneDNNTensorStorage> inputs;
  inputs.reserve(request.tensors.inputs.size());
  for (const TargetModelNumericTensor &input : request.tensors.inputs) {
    llvm::Expected<OneDNNTensorStorage> storage =
        OneDNNTensorStorage::create(input.key, input.storage);
    if (!storage)
      return storage.takeError();
    inputs.push_back(std::move(*storage));
  }
  llvm::Expected<OneDNNTensorStorage> destinationTemplate =
      OneDNNTensorStorage::create(request.tensors.destinationTemplate.key,
                                  request.tensors.destinationTemplate.storage);
  if (!destinationTemplate)
    return destinationTemplate.takeError();
  llvm::Expected<OneDNNTensorNumericResult> result =
      executeManagedReferenceOneDNNTensorNumeric(
          environment, request.operation, inputs, *destinationTemplate, budget);
  if (!result)
    return result.takeError();
  TargetModelOneDNNResult modelResult{
      {result->destination.getKey(),
       std::vector<uint8_t>(result->destination.getStorage().begin(),
                            result->destination.getStorage().end())},
      result->flags,
      {result->evidence.matmulInvocations, result->evidence.reorderInvocations,
       result->evidence.formalFusedMultiplyAdds,
       TargetModelOneDNNEvidenceKind::ManagedReferenceEnvironment,
       environment.getDigest().str(), result->evidence.implementation}};
  return std::optional<TargetModelOneDNNResult>(std::move(modelResult));
}

} // namespace wafer::model
